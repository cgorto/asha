use crate::core::util::{atomic_add_device, smoothstep};
use abi_core::oct_encode;
use abi_core::{GpuPtr, GraphicsPush, ReduceSingleDispatchData};
use abi_light::{light_field_gate, light_field_sample, mesh_rim_contribution, mesh_shade_slim};
use abi_mesh::{
    ClusterCullData, IndirectData, LineworkData, MESH_FLAG_NO_LINEWORK, MESH_FLAG_SKINNED,
    MeshFrameData, MeshInstance, MeshTableEntry, max_world_scale, meshlet_backfacing_to_camera,
    sphere_inside_planes,
};
use abi_mesh::{evaluate_vertex, evaluate_vertex_position};
use glam::{UVec2, UVec3, Vec2, Vec3, Vec4};
use spirv_std::arch::{atomic_load, atomic_store, workgroup_memory_barrier_with_group_sync};
use spirv_std::image::Image2d;
use spirv_std::memory::{Scope, Semantics};
use spirv_std::num_traits::Float;
use spirv_std::spirv;
use spirv_std::{Image, RuntimeArray, Sampler};

/// Computes the shared clip position for opaque mesh vertices.
/// Sharing this law preserves reverse-Z prepass-to-forward equality.
fn cluster_clip_position(
    data: GpuPtr<MeshFrameData>,
    indirect: &IndirectData,
    instance_index: u32,
    vert_id: u32,
) -> Vec4 {
    let batch = data.batches[indirect.batch_index];
    let cluster = data.clusters[instance_index];
    let instance = data.instances[cluster.instance_id];
    let meshlet = data.meshlets[cluster.meshlet_index];
    let mesh = data.mesh_table[batch.mesh_index];
    let vertex_index = data.index_data[meshlet.first_index + vert_id];
    let transform = data.transforms[instance.transform_index];
    let p = mesh.positions[vertex_index];
    let local = evaluate_vertex_position(
        instance.joint_transforms,
        mesh.joint_weights,
        instance.deformer_slot,
        data.deformers,
        vertex_index,
        Vec3::new(p[0], p[1], p[2]),
    );
    let world = transform.model_to_world * local.extend(1.0);
    data.world_to_clip * world
}

/// Rasterizes visible clusters from non-indexed indirect instances.
/// Vulkan `InstanceIndex` includes `cmd.first_instance`.
#[spirv(vertex)]
pub fn mesh_vert(
    #[spirv(push_constant)] push: &GraphicsPush,
    #[spirv(vertex_index)] vert_id: i32,
    #[spirv(instance_index)] instance_index: i32,
    #[spirv(draw_index)] draw_id: i32,
    #[spirv(position)] out_pos: &mut Vec4,
    out_normal_world: &mut Vec3,
    out_position_world: &mut Vec3,
    out_uv: &mut Vec2,
    #[spirv(flat)] out_material_index: &mut u32,
    out_instance_color: &mut Vec4,
) {
    let data = push.vert::<MeshFrameData>();
    let indirect = push.indirect::<IndirectData>()[draw_id];
    let cluster = data.clusters[instance_index as u32];
    let instance = data.instances[cluster.instance_id];
    let meshlet = data.meshlets[cluster.meshlet_index];
    if vert_id as u32 >= meshlet.tri_count * 3 {
        // Whole-triangle padding emits no sliver.
        *out_pos = Vec4::ZERO;
        *out_normal_world = Vec3::ZERO;
        *out_position_world = Vec3::ZERO;
        *out_uv = Vec2::ZERO;
        *out_material_index = 0;
        *out_instance_color = Vec4::ONE;
        return;
    }
    let batch = data.batches[indirect.batch_index];
    let mesh = data.mesh_table[batch.mesh_index];
    let vertex_index = data.index_data[meshlet.first_index + vert_id as u32];
    let transform = data.transforms[instance.transform_index];
    let p = mesh.positions[vertex_index];
    let n = mesh.normals[vertex_index];
    let evaluated = evaluate_vertex(
        instance.joint_transforms,
        mesh.joint_weights,
        instance.deformer_slot,
        data.deformers,
        vertex_index,
        Vec3::new(p[0], p[1], p[2]),
        Vec3::new(n[0], n[1], n[2]),
    );
    let normal = transform.model_to_world_normal * evaluated.1.extend(0.0);
    *out_position_world = (transform.model_to_world * evaluated.0.extend(1.0)).truncate();
    *out_pos = cluster_clip_position(data, &indirect, instance_index as u32, vert_id as u32);
    *out_normal_world = normal.truncate().normalize();
    *out_uv = Vec2::from_array(mesh.uvs[vertex_index]);
    *out_material_index = batch.material_index;
    // Vertex color modulates instance tint; absent color means one.
    let vertex_color = if mesh.colors.is_null() {
        Vec4::ONE
    } else {
        Vec4::from_array(mesh.colors[vertex_index])
    };
    *out_instance_color = Vec4::from_array(instance.instance_color) * vertex_color;
}

/// Emits depth-prepass vertices and their compacted cluster indices.
#[spirv(vertex)]
pub fn mesh_prepass_vert(
    #[spirv(push_constant)] push: &GraphicsPush,
    #[spirv(vertex_index)] vert_id: i32,
    #[spirv(instance_index)] instance_index: i32,
    #[spirv(draw_index)] draw_id: i32,
    #[spirv(position)] out_pos: &mut Vec4,
    #[spirv(flat)] out_cluster_index: &mut u32,
) {
    let data = push.vert::<MeshFrameData>();
    let indirect = push.indirect::<IndirectData>()[draw_id];
    let cluster = data.clusters[instance_index as u32];
    let meshlet = data.meshlets[cluster.meshlet_index];
    if vert_id as u32 >= meshlet.tri_count * 3 {
        // Match mesh vertex padding for whole-triangle degenerates.
        *out_pos = Vec4::ZERO;
        *out_cluster_index = 0;
        return;
    }
    *out_pos = cluster_clip_position(data, &indirect, instance_index as u32, vert_id as u32);
    *out_cluster_index = instance_index as u32;
}

/// Emits mesh silhouettes using the opaque clip-position law.
/// Equal depth is the occlusion contract.
#[spirv(vertex)]
pub fn mesh_silhouette_vert(
    #[spirv(push_constant)] push: &GraphicsPush,
    #[spirv(vertex_index)] vert_id: i32,
    #[spirv(instance_index)] instance_index: i32,
    #[spirv(draw_index)] draw_id: i32,
    #[spirv(position)] out_pos: &mut Vec4,
    #[spirv(flat)] out_outline_group: &mut u32,
) {
    let data = push.vert::<MeshFrameData>();
    let indirect = push.indirect::<IndirectData>()[draw_id];
    let cluster = data.clusters[instance_index as u32];
    let instance = data.instances[cluster.instance_id];
    let meshlet = data.meshlets[cluster.meshlet_index];
    if vert_id as u32 >= meshlet.tri_count * 3 {
        // Match prepass padding; ragged tails cannot create slivers.
        *out_pos = Vec4::ZERO;
        *out_outline_group = 0;
        return;
    }
    *out_pos = cluster_clip_position(data, &indirect, instance_index as u32, vert_id as u32);
    *out_outline_group = instance.outline_group;
}

/// R32_UINT visibility token: bits 31..25 hold the exact meshlet-local
/// primitive id (valid values 0..=123; meshlets are limited to 124 triangles);
/// bits 24..0 hold the compacted cluster index plus one (token 1..=0x01ff_ffff,
/// cluster index 0..=0x01ff_fffe). Zero is the cleared sky/void token.
#[spirv(fragment)]
pub fn mesh_prepass_frag(
    #[spirv(primitive_id, flat)] prim_id: u32,
    #[spirv(flat)] cluster_index: u32,
    out_visibility: &mut u32,
) {
    let visible_cluster_token = cluster_index + 1;
    *out_visibility = ((prim_id & 0x7F) << 25) | (visible_cluster_token & 0x01FF_FFFF);
}

/// Encodes outline groups for JFA as `group / 255`.
#[spirv(fragment)]
pub fn mesh_silhouette_frag(#[spirv(flat)] outline_group: u32, out_mask: &mut f32) {
    if outline_group == 0 {
        spirv_std::arch::kill();
    }
    *out_mask = outline_group as f32 / 255.0;
}

/// Reconstructs world positions using the prepass vertex evaluation law.
fn linework_vertex_world(
    frame: GpuPtr<MeshFrameData>,
    instance: MeshInstance,
    mesh: MeshTableEntry,
    vertex_index: u32,
) -> Vec3 {
    let transform = frame.transforms[instance.transform_index];
    let p = mesh.positions[vertex_index];
    let local = evaluate_vertex_position(
        instance.joint_transforms,
        mesh.joint_weights,
        instance.deformer_slot,
        frame.deformers,
        vertex_index,
        Vec3::new(p[0], p[1], p[2]),
    );
    (transform.model_to_world * local.extend(1.0)).truncate()
}

/// Resolves a nonzero prepass token through the compacted cluster stream.
/// A primitive outside the meshlet's `tri_count`, or an instance marked
/// `MESH_FLAG_NO_LINEWORK`, returns its flags and a zero triangle; callers
/// reject the result without using its vertices.
fn linework_resolve(frame: GpuPtr<MeshFrameData>, token: u32) -> (u32, Vec3, Vec3, Vec3) {
    let primitive = token >> 25;
    let cluster_index = (token & 0x01ff_ffff) - 1;
    let cluster = frame.clusters[cluster_index];
    let meshlet = frame.meshlets[cluster.meshlet_index];
    let instance = frame.instances[cluster.instance_id];
    if primitive >= meshlet.tri_count || instance.flags & MESH_FLAG_NO_LINEWORK != 0 {
        return (instance.flags, Vec3::ZERO, Vec3::ZERO, Vec3::ZERO);
    }
    let batch = frame.batches[instance.batch_index];
    let mesh = frame.mesh_table[batch.mesh_index];
    let first = meshlet.first_index + primitive * 3;
    let i0 = frame.index_data[first];
    let i1 = frame.index_data[first + 1];
    let i2 = frame.index_data[first + 2];
    (
        instance.flags,
        linework_vertex_world(frame, instance, mesh, i0),
        linework_vertex_world(frame, instance, mesh, i1),
        linework_vertex_world(frame, instance, mesh, i2),
    )
}

/// Reconstructs world position from reverse-Z depth.
/// The raster matrix already accounts for +Y-down coordinates.
fn linework_world_position(data: &LineworkData, coord: UVec2, depth: f32) -> Vec3 {
    let screen = Vec2::new(data.screen_size[0] as f32, data.screen_size[1] as f32);
    let ndc = (coord.as_vec2() + Vec2::splat(0.5)) / screen * 2.0 - Vec2::ONE;
    let h = data.clip_to_world * Vec4::new(ndc.x, ndc.y, depth, 1.0);
    h.truncate() / h.w
}

/// Draws screen-space linework from prepass tokens and depth.
/// Token equality checks precede all mesh resolution.
#[spirv(fragment)]
pub fn mesh_linework_frag(
    #[spirv(push_constant)] push: &GraphicsPush,
    #[spirv(descriptor_set = 0, binding = 0)] textures: &RuntimeArray<Image2d>,
    #[spirv(descriptor_set = 1, binding = 0)] textures_rw: &RuntimeArray<
        Image!(2D, format = r32ui, sampled = false),
    >,
    #[spirv(frag_coord)] frag_coord: Vec4,
    out_color: &mut Vec4,
) {
    let data = push.frag::<LineworkData>();
    let coord = frag_coord.truncate().truncate().as_uvec2();
    let visibility = unsafe { textures_rw.index(data.visibility_texture_id as usize) };
    let center_token = visibility.read(coord);

    // Guard one: cleared tokens denote sky or void.
    if center_token == 0 {
        *out_color = Vec4::ZERO;
        return;
    }

    let max = UVec2::new(data.screen_size[0] - 1, data.screen_size[1] - 1);
    let left = if coord.x > 0 { coord.x - 1 } else { 0 };
    let right = if coord.x < max.x { coord.x + 1 } else { max.x };
    let up = if coord.y > 0 { coord.y - 1 } else { 0 };
    let down = if coord.y < max.y { coord.y + 1 } else { max.y };
    let taps = [
        UVec2::new(left, coord.y),
        UVec2::new(right, coord.y),
        UVec2::new(coord.x, up),
        UVec2::new(coord.x, down),
    ];
    let tap_tokens = [
        visibility.read(taps[0]),
        visibility.read(taps[1]),
        visibility.read(taps[2]),
        visibility.read(taps[3]),
    ];

    // Reject interiors before triangle, normal, plane, or depth resolution.
    if tap_tokens[0] == center_token
        && tap_tokens[1] == center_token
        && tap_tokens[2] == center_token
        && tap_tokens[3] == center_token
    {
        *out_color = Vec4::ZERO;
        return;
    }

    let depth_image = unsafe { textures.index(data.depth_texture_id as usize) };
    let center_depth = depth_image.fetch_with_lod(coord, 0).x;
    let center_position = linework_world_position(&data, coord, center_depth);
    let (center_flags, c0, c1, c2) = linework_resolve(data.frame, center_token);
    // Excluded instances contribute no linework.
    if center_flags & MESH_FLAG_NO_LINEWORK != 0 {
        *out_color = Vec4::ZERO;
        return;
    }
    let center_triangle = (c0, c1, c2);
    let center_cross =
        (center_triangle.1 - center_triangle.0).cross(center_triangle.2 - center_triangle.0);
    // Guard two: zero-area triangles have no normal or plane.
    if center_cross.length_squared() <= 1.0e-12 {
        *out_color = Vec4::ZERO;
        return;
    }
    let center_normal = center_cross.normalize();
    // Guard three: unknown depth invalidates the resolved plane.
    if (center_position - center_triangle.0)
        .dot(center_normal)
        .abs()
        > data.plane_epsilon
    {
        *out_color = Vec4::ZERO;
        return;
    }

    let mut edge: f32 = 0.0;
    let mut tap = 0;
    while tap < 4 {
        let token = tap_tokens[tap];
        if token != center_token {
            // Guard four: void neighbors need no token or geometry resolve.
            if token == 0 {
                edge = edge.max(data.step_strength);
            } else {
                let (neighbor_flags, n0, n1, n2) = linework_resolve(data.frame, token);
                let neighbor_triangle = (n0, n1, n2);
                let neighbor_depth = depth_image.fetch_with_lod(taps[tap], 0).x;
                let neighbor_position = linework_world_position(&data, taps[tap], neighbor_depth);
                let neighbor_cross = (neighbor_triangle.1 - neighbor_triangle.0)
                    .cross(neighbor_triangle.2 - neighbor_triangle.0);
                // Excluded neighbors leave no halo; reject degenerate normals.
                if neighbor_flags & MESH_FLAG_NO_LINEWORK == 0
                    && neighbor_cross.length_squared() > 1.0e-12
                {
                    let neighbor_normal = neighbor_cross.normalize();
                    let normal_angle = center_normal.dot(neighbor_normal).clamp(-1.0, 1.0).acos();
                    let normal_threshold = data.normal_cos_threshold.acos();
                    let crease = smoothstep(normal_threshold, normal_threshold * 1.5, normal_angle);
                    edge = edge.max(crease * data.crease_strength);

                    let plane_distance = (neighbor_position - center_position)
                        .dot(center_normal)
                        .abs();
                    let step =
                        smoothstep(data.plane_epsilon, data.plane_epsilon * 1.5, plane_distance);
                    // Reverse-Z: larger depth is nearer.
                    if center_depth > neighbor_depth {
                        edge = edge.max(step * data.step_strength);
                    }
                }
            }
        }
        tap += 1;
    }

    let distance = (center_position - Vec3::from_array(data.eye)).length();
    let fade = 1.0 - smoothstep(data.fade_near, data.fade_far, distance);
    let field = light_field_gate(
        light_field_sample(
            data.light_field,
            data.light_field_dims,
            data.light_field_cell_size,
            center_position,
        ),
        data.light_field_gate,
    );
    let ink = edge * fade * data.darkness_seat * field;
    *out_color = Vec4::new(0.0, 0.0, 0.0, ink);
}

/// Writes HDR color plus the deferred-light surface MRT.
/// Locations 1–3 contain octahedral normal, tinted albedo, and
/// `material_index + 1`; zero is the no-mesh sentinel. Unbound extra
/// attachments discard their outputs.
#[spirv(fragment)]
pub fn mesh_frag(
    #[spirv(push_constant)] push: &GraphicsPush,
    #[spirv(descriptor_set = 0, binding = 0)] textures: &RuntimeArray<Image2d>,
    #[spirv(descriptor_set = 2, binding = 0)] samplers: &RuntimeArray<Sampler>,
    normal_world: Vec3,
    position_world: Vec3,
    uv: Vec2,
    #[spirv(flat)] material_index: u32,
    instance_color: Vec4,
    #[spirv(location = 0)] out_color: &mut Vec4,
    #[spirv(location = 1)] out_surface_normal: &mut Vec2,
    #[spirv(location = 2)] out_surface_albedo: &mut Vec4,
    #[spirv(location = 3)] out_surface_material: &mut f32,
) {
    let data = push.frag::<MeshFrameData>();
    let material = data.materials[material_index];
    let mut albedo = Vec3::new(
        material.base_color_factor[0],
        material.base_color_factor[1],
        material.base_color_factor[2],
    );
    // Multiply the linear base-color map into the glTF factor.
    if material.base_color_map != 0 {
        let sampler_id = if material.base_color_sampler == 0 {
            data.ramp_default_sampler
        } else {
            material.base_color_sampler
        };
        let map = unsafe { textures.index(material.base_color_map as usize) };
        let sampler = *unsafe { samplers.index(sampler_id as usize) };
        let texel: Vec4 = map.sample_by_lod(sampler, uv, 0.0);
        albedo *= texel.truncate();
    }
    let mut rgb = Vec3::from_array(mesh_shade_slim(
        normal_world,
        albedo.to_array(),
        &data.lighting,
    ));
    let n = if normal_world.length_squared() > 1.0e-8 {
        normal_world.normalize()
    } else {
        Vec3::Y
    };
    let field = light_field_gate(
        light_field_sample(
            data.light_field,
            data.light_field_dims,
            data.light_field_cell_size,
            position_world,
        ),
        data.light_field_gate,
    );
    if material.rim_boost > 0.0 {
        rgb += mesh_rim_contribution(
            n,
            position_world,
            Vec3::from_array(data.eye),
            &data.lighting,
            field,
            material.rim_power,
            material.rim_boost,
        );
    }
    // Add factor-only unlit emissive inside instance tint.
    rgb += Vec3::from_array(material.emissive_factor);
    *out_color = Vec4::new(rgb.x, rgb.y, rgb.z, 1.0) * instance_color;
    // Export the guarded normal, tinted albedo, and material-plus-one sentinel.
    *out_surface_normal = oct_encode(n);
    let tinted = albedo * instance_color.truncate();
    *out_surface_albedo = Vec4::new(tinted.x, tinted.y, tinted.z, 1.0);
    *out_surface_material = (material_index + 1) as f32;
}

/// Flat-unlit forward shading. Group fragments under `ReplaceForward` must:
///
/// - Declare all five `mesh_vert` varyings in positional order.
/// - Write all four outputs; unwritten MRT attachments are undefined.
///   Use zero material for no mesh, or `material_index + 1` for relighting.
/// - `instance_color` is smooth and already includes vertex color.
#[spirv(fragment)]
pub fn mesh_flat_frag(
    #[spirv(push_constant)] push: &GraphicsPush,
    normal_world: Vec3,
    _position_world: Vec3,
    _uv: Vec2,
    #[spirv(flat)] material_index: u32,
    instance_color: Vec4,
    #[spirv(location = 0)] out_color: &mut Vec4,
    #[spirv(location = 1)] out_surface_normal: &mut Vec2,
    #[spirv(location = 2)] out_surface_albedo: &mut Vec4,
    #[spirv(location = 3)] out_surface_material: &mut f32,
) {
    let data = push.frag::<MeshFrameData>();
    let material = data.materials[material_index];
    let albedo = Vec3::new(
        material.base_color_factor[0],
        material.base_color_factor[1],
        material.base_color_factor[2],
    ) * instance_color.truncate();
    *out_color = albedo.extend(1.0);
    let n = if normal_world.length_squared() > 1.0e-8 {
        normal_world.normalize()
    } else {
        Vec3::Y
    };
    *out_surface_normal = oct_encode(n);
    *out_surface_albedo = albedo.extend(1.0);
    // Zero material opts out of local lighting.
    *out_surface_material = 0.0;
}

/// Emits constant additive glow `instance_color.rgb * instance_color.w`.
/// Coat fragments must:
///
/// - Declare the same five positional varyings.
/// - Emit a light contribution because blending is One plus One.
/// - Declare surface outputs; masking discards them.
#[spirv(fragment)]
pub fn glow_coat_frag(
    #[spirv(push_constant)] _push: &GraphicsPush,
    _normal_world: Vec3,
    _position_world: Vec3,
    _uv: Vec2,
    #[spirv(flat)] _material_index: u32,
    instance_color: Vec4,
    #[spirv(location = 0)] out_color: &mut Vec4,
    #[spirv(location = 1)] out_surface_normal: &mut Vec2,
    #[spirv(location = 2)] out_surface_albedo: &mut Vec4,
    #[spirv(location = 3)] out_surface_material: &mut f32,
) {
    *out_color = (instance_color.truncate() * instance_color.w).extend(0.0);
    // Masked surface writes are explicitly zero.
    *out_surface_normal = Vec2::ZERO;
    *out_surface_albedo = Vec4::ZERO;
    *out_surface_material = 0.0;
}

/// Adds a time-based warm pulse scaled by `instance_color.x`.
/// `instance_color.y` supplies phase; material-plus-one preserves relighting.
#[spirv(fragment)]
pub fn hazard_pulse_frag(
    #[spirv(push_constant)] push: &GraphicsPush,
    normal_world: Vec3,
    _position_world: Vec3,
    _uv: Vec2,
    #[spirv(flat)] material_index: u32,
    instance_color: Vec4,
    #[spirv(location = 0)] out_color: &mut Vec4,
    #[spirv(location = 1)] out_surface_normal: &mut Vec2,
    #[spirv(location = 2)] out_surface_albedo: &mut Vec4,
    #[spirv(location = 3)] out_surface_material: &mut f32,
) {
    let data = push.frag::<MeshFrameData>();
    let material = data.materials[material_index];
    let albedo = Vec3::new(
        material.base_color_factor[0],
        material.base_color_factor[1],
        material.base_color_factor[2],
    );
    let mut rgb = Vec3::from_array(mesh_shade_slim(
        normal_world,
        albedo.to_array(),
        &data.lighting,
    ));
    // Smoothstep shapes the phase pulse toward its extremes.
    let strain = instance_color.x.clamp(0.0, 1.0);
    let wave = (data.time * 4.0 + instance_color.y).sin() * 0.5 + 0.5;
    let throb = smoothstep(0.15, 0.85, wave);
    const EMBER: Vec3 = Vec3::new(1.0, 0.32, 0.08);
    rgb += EMBER * (strain * (0.25 + 0.75 * throb));
    *out_color = rgb.extend(1.0);
    let n = if normal_world.length_squared() > 1.0e-8 {
        normal_world.normalize()
    } else {
        Vec3::Y
    };
    *out_surface_normal = oct_encode(n);
    *out_surface_albedo = albedo.extend(1.0);
    *out_surface_material = (material_index + 1) as f32;
}

/// Culls `(instance, meshlet)` pairs into batch-exclusive ranges.
#[spirv(compute(threads(64, 4, 1)))]
pub fn cluster_cull(
    #[spirv(push_constant)] data_ptr: &GpuPtr<ClusterCullData>,
    #[spirv(global_invocation_id)] id: UVec3,
) {
    let data = &**data_ptr;
    let instance_id = id.x;
    let local_meshlet = id.y;
    if instance_id >= data.instance_count || local_meshlet >= data.max_meshlets_per_mesh {
        return;
    }
    let instance = data.instances[instance_id];
    if instance.batch_index >= data.batch_count || (instance.flags & data.cull_mask) != 0 {
        return;
    }
    let batch = data.batches[instance.batch_index];
    let mesh = data.mesh_data[batch.mesh_index];
    if local_meshlet >= mesh.meshlet_count {
        return;
    }
    let meshlet_index = mesh.meshlet_offset + local_meshlet;
    let meshlet = data.meshlets[meshlet_index];
    let transform = data.transforms[instance.transform_index];
    let c = meshlet.center;
    let world_center = (transform.model_to_world * Vec4::new(c[0], c[1], c[2], 1.0)).truncate();
    let world_radius =
        meshlet.radius * max_world_scale(&transform.model_to_world) + instance.bounds_dilation;
    // Deformation invalidates cones; the dilated sphere remains conservative.
    if instance.flags & MESH_FLAG_SKINNED == 0
        && instance.deformer_slot == 0
        && meshlet_backfacing_to_camera(
            &meshlet,
            &transform.model_to_world_normal,
            Vec3::from_array(data.camera_pos),
            world_center,
            data.cone_cull_epsilon,
        )
    {
        return;
    }
    let planes = data.frustum_planes;
    if !sphere_inside_planes(world_center, world_radius, &planes) {
        return;
    }

    let slot = atomic_add_device(data.visible_counts.offset(instance.batch_index as i64), 1);
    // Host-provided batch capacity violations fail closed.
    if slot >= batch.cluster_capacity {
        return;
    }
    let cluster_index = batch.cluster_base + slot;
    // WORKAROUND: Wine Vulkan/NVIDIA crashes on whole-struct BDA stores of
    // these mixed-layout records. Keep the fields as individually addressed
    // narrow stores: no struct temporary or hidden memcpy.
    let mut clusters = data.clusters;
    clusters[cluster_index].instance_id = instance_id;
    clusters[cluster_index].meshlet_index = meshlet_index;
}

/// Build one non-indexed indirect record for every candidate batch. Empty
/// batches remain commands with zero instances; the CPU-known draw count is
/// the candidate batch count, never a survivor count.
#[spirv(compute(threads(64)))]
pub fn cluster_build_args(
    #[spirv(push_constant)] data_ptr: &GpuPtr<ClusterCullData>,
    #[spirv(global_invocation_id)] id: UVec3,
) {
    let data = &**data_ptr;
    let batch_index = id.x;
    if batch_index >= data.batch_count {
        return;
    }
    let batch = data.batches[batch_index];
    let mesh = data.mesh_data[batch.mesh_index];
    let visible_count = data.visible_counts[batch_index];
    // WORKAROUND: Wine Vulkan/NVIDIA crashes on whole-struct BDA stores of
    // these mixed-layout records. Keep the fields as individually addressed
    // narrow stores: no struct temporary or hidden memcpy.
    let mut out = data.output_indirect;
    out[batch_index].cmd.vertex_count = mesh.cluster_vertex_count;
    out[batch_index].cmd.instance_count = visible_count;
    out[batch_index].cmd.first_vertex = 0;
    out[batch_index].cmd.first_instance = batch.cluster_base;
    out[batch_index].batch_index = batch_index;
}

const DEVICE: u32 = Scope::Device as u32;
/// Release plus uniform-memory semantics for device atomics.
const REL_UNIFORM: u32 = Semantics::RELEASE.bits() | Semantics::UNIFORM_MEMORY.bits();
const ACQ_UNIFORM: u32 = Semantics::ACQUIRE.bits() | Semantics::UNIFORM_MEMORY.bits();
const ACQ_REL_UNIFORM: u32 = Semantics::ACQUIRE_RELEASE.bits() | Semantics::UNIFORM_MEMORY.bits();

/// Two-stage reduction in one dispatch using the last-workgroup election.
/// Each group publishes its partial with a device-scope RELEASE atomic and
/// increments the device-scope ACQUIRE_RELEASE counter; the group seeing
/// `group_count - 1` is last and reads every partial with ACQUIRE atomics.
/// There is no wait or persistent global barrier: non-elected groups exit.
/// Every cross-workgroup handoff is atomic-only. Under Vulkan's memory model,
/// a counter release does not make neighboring plain physical-pointer stores
/// visible; atomic partials are the coherent handoff.
#[spirv(compute(threads(64)))]
pub fn reduce_single_dispatch(
    #[spirv(push_constant)] data_ptr: &GpuPtr<ReduceSingleDispatchData>,
    #[spirv(workgroup)] lds: &mut [u32; 64],
    #[spirv(workgroup)] elected: &mut u32,
    #[spirv(workgroup_id)] group_id: UVec3,
    #[spirv(local_invocation_index)] lane: u32,
) {
    let data = &**data_ptr;
    let li = lane as usize;

    // Stage one: reduce each workgroup's 64 values.
    lds[li] = data.values[group_id.x * 64 + lane];
    workgroup_memory_barrier_with_group_sync();
    let mut stride = 32u32;
    while stride > 0 {
        if lane < stride {
            lds[li] = lds[li].wrapping_add(lds[(lane + stride) as usize]);
        }
        workgroup_memory_barrier_with_group_sync();
        stride >>= 1;
    }

    // Publish the partial and elect the final workgroup.
    if lane == 0 {
        // SAFETY: `partials` has one element per workgroup.
        unsafe {
            atomic_store::<u32, DEVICE, REL_UNIFORM>(
                &mut *data.partials.offset(group_id.x as i64).as_ptr(),
                lds[0],
            );
            let prev = spirv_std::arch::atomic_i_add::<u32, DEVICE, ACQ_REL_UNIFORM>(
                &mut *data.counter.as_ptr(),
                1,
            );
            *elected = (prev == data.group_count - 1) as u32;
        }
    }
    workgroup_memory_barrier_with_group_sync();
    if *elected == 0 {
        return; // Uniform per workgroup — every barrier below is intact.
    }

    // Stage two: the elected workgroup folds all partials.
    let mut acc = 0u32;
    let mut i = lane;
    while i < data.group_count {
        // SAFETY: loop bounds ensure access; acquire pairs with releases.
        acc = acc.wrapping_add(unsafe {
            atomic_load::<u32, DEVICE, ACQ_UNIFORM>(&*data.partials.offset(i as i64).as_ptr())
        });
        i += 64;
    }
    lds[li] = acc;
    workgroup_memory_barrier_with_group_sync();
    let mut stride = 32u32;
    while stride > 0 {
        if lane < stride {
            lds[li] = lds[li].wrapping_add(lds[(lane + stride) as usize]);
        }
        workgroup_memory_barrier_with_group_sync();
        stride >>= 1;
    }
    if lane == 0 {
        let mut result = data.result;
        result[0u32] = lds[0];
    }
}
