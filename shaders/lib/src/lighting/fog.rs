use crate::core::util::{
    atomic_add_device, atomic_max_device, atomic_min_device, atomic_or_device,
};
use abi_core::ray_direction;
use abi_core::{GpuPtr, GraphicsPush};
use abi_light::{DepthReduceData, MAX_TILES, PrefixSumData, TILE_SIZE, TileDepthBounds};
use abi_light::{
    EXT_MAX, EXT_RGB_FIELD_MAX, FOG_LIGHT_TILE, FOG_SLICE_MAX, FogCompositeData, FogDepthMaxData,
    FogIntegrateData, FogLightData, FogLightGridData, FogParamsData, FogPrimeQuad,
    FogPrimeSpawnData, FogPrimeVertData, OitAccumFragData, OitAccumVertData, OitParticle,
    OitResolveData, OitSplatFragData, OitSplatVertData, PRIME_TILE, ZERO_SLICE_NONE,
    ZERO_TRANS_EPS, ext_dword_index, ext_rgb_decode, ext_rgb_field, ext_rgb_index, ext_rgb_pack,
    extinction_decode, extinction_encode, extinction_to_u8, fog_curve_from, fog_light_tile_bounds,
    fog_point_light_radiance, froxel_params_from, height_fog_optical_depth, height_gradient,
    hg_phase, integrate_step, interleaved_gradient_noise, oit_resolve as oit_resolve_rgb,
    prime_froxel_range, prime_quad_depth, slice_of_z, splat_weights, transmittance, warp_eval,
    warped_slice_of_z, z_of_warped_slice,
};
use glam::{UVec2, UVec3, Vec2, Vec3, Vec4};
use spirv_std::arch::workgroup_memory_barrier_with_group_sync;
use spirv_std::image::{Image2d, Image3d};
use spirv_std::num_traits::Float;
use spirv_std::spirv;
use spirv_std::{Image, RuntimeArray, Sampler};

/// Fetches a managed bindless texture texel at mip `lod`.
/// Index zero is reserved as the null-texture sentinel.
fn heap_fetch(textures: &RuntimeArray<Image2d>, id: u32, coord: UVec2, lod: u32) -> Vec4 {
    unsafe { textures.index(id as usize) }.fetch_with_lod(coord, lod)
}

fn normalize_or(v: Vec3, fallback: Vec3) -> Vec3 {
    let len2 = v.length_squared();
    if len2 > 1.0e-12 {
        v / len2.sqrt()
    } else {
        fallback
    }
}

fn in_unit_cube(v: Vec3) -> bool {
    v.x >= 0.0 && v.y >= 0.0 && v.z >= 0.0 && v.x <= 1.0 && v.y <= 1.0 && v.z <= 1.0
}

fn oit_quad_corner(vert_id: i32) -> Vec2 {
    match vert_id {
        0 | 3 => Vec2::new(-1.0, -1.0),
        1 => Vec2::new(-1.0, 1.0),
        2 | 4 => Vec2::new(1.0, 1.0),
        _ => Vec2::new(1.0, -1.0),
    }
}

fn oit_billboard_vertex(
    view: &abi_core::View,
    particle: OitParticle,
    vert_id: i32,
) -> (Vec4, f32, Vec4) {
    let camera = Vec3::from_array(view.camera_position);
    let forward = Vec3::from_array(view.camera_forward);
    let right = Vec3::from_array(view.camera_right);
    let up = Vec3::from_array(view.camera_up);
    let center = Vec3::from_array(particle.pos);
    let rel_center = center - camera;
    let center_z = forward.dot(rel_center);

    let corner = oit_quad_corner(vert_id);
    let half_size = particle.size * 0.5;
    let world = center + right * (corner.x * half_size) - up * (corner.y * half_size);
    let rel = world - camera;
    let x_scale = 1.0 / (view.tan_half_fov * view.aspect);
    let y_scale = 1.0 / view.tan_half_fov;
    let clip = Vec4::new(
        right.dot(rel) * x_scale,
        -up.dot(rel) * y_scale,
        view.depth_near_plane,
        forward.dot(rel),
    );
    (
        clip,
        center_z,
        Vec4::new(
            particle.color[0],
            particle.color[1],
            particle.color[2],
            particle.alpha,
        ),
    )
}

/// Atomically splats low-resolution transparent extinction into `V_ext`.
#[spirv(vertex)]
pub fn oit_splat_vert(
    #[spirv(push_constant)] push: &GraphicsPush,
    #[spirv(vertex_index)] vert_id: i32,
    #[spirv(instance_index)] inst_id: i32,
    #[spirv(position)] out_pos: &mut Vec4,
    out_view_z: &mut f32,
    out_alpha: &mut f32,
    out_tint: &mut Vec3,
) {
    let data = push.vert::<OitSplatVertData>();
    let particle = data.particles[inst_id];
    let (clip, view_z, color) = oit_billboard_vertex(&data.view, particle, vert_id);
    *out_pos = clip;
    *out_view_z = view_z;
    *out_alpha = color.w;
    *out_tint = Vec3::from_array(particle.tint_od);
}

#[spirv(fragment)]
pub fn oit_splat_frag(
    #[spirv(push_constant)] push: &GraphicsPush,
    #[spirv(frag_coord)] frag_coord: Vec4,
    view_z: f32,
    alpha: f32,
    tint: Vec3,
    out_color: &mut Vec4,
) {
    let data = push.frag::<OitSplatFragData>();
    let pixel = frag_coord.truncate().truncate().as_uvec2();
    if pixel.x >= data.view.output_size[0] || pixel.y >= data.view.output_size[1] {
        *out_color = Vec4::ZERO;
        return;
    }

    let curve = &*data.params;
    let encoded = extinction_encode(alpha);
    let n = curve.params.slice_count_u32;
    // Histogram raw slices; store extinction at warped slices.
    let raw = slice_of_z(&curve.params, view_z);
    let warped = warp_eval(&curve.warp, n, raw);
    let (s0, w0, s1, w1) = splat_weights(warped, n);
    let u0 = extinction_to_u8(encoded * w0);
    let u1 = extinction_to_u8(encoded * w1);
    if u0 != 0 {
        oit_splat_one(data, pixel, s0, u0);
    }
    if u1 != 0 {
        oit_splat_one(data, pixel, s1, u1);
    }
    // Store colored transmission separately; preserve the scalar path.
    let tinted = tint.x > 0.0 || tint.y > 0.0 || tint.z > 0.0;
    if tinted {
        oit_splat_rgb_one(data, pixel, s0, tint, w0);
        oit_splat_rgb_one(data, pixel, s1, tint, w1);
    }
    if u0 != 0 || u1 != 0 || tinted {
        let bin = (raw as u32).min(n - 1);
        atomic_add_device(data.hist.offset(bin as i64), 1);
    }

    *out_color = Vec4::ZERO;
}

fn oit_splat_one(data: GpuPtr<OitSplatFragData>, pixel: UVec2, slice: u32, u: u32) {
    let width = data.view.output_size[0];
    let pixel_column = pixel.y * width + pixel.x;
    let (dword, lane) = ext_dword_index(
        pixel.x,
        pixel.y,
        slice,
        width,
        data.params.params.slice_count_u32,
    );
    let shift = lane * 8;
    atomic_or_device(
        data.occupancy
            .offset((pixel_column * 2 + slice / 32) as i64),
        1 << (slice % 32),
    );
    let prev = atomic_add_device(data.v_ext.offset(dword as i64), u << shift);
    let lane_byte = (prev >> shift) & 0xff;
    if lane_byte + u > 255 {
        atomic_min_device(data.overflow.offset(pixel_column as i64), slice);
    }
}

/// One RGB extinction splat: one packed atomic updates all three channels;
/// each 10-bit field records its first overflowing slice, so integration
/// saturates that channel from there.
fn oit_splat_rgb_one(
    data: GpuPtr<OitSplatFragData>,
    pixel: UVec2,
    slice: u32,
    tint_od: Vec3,
    weight: f32,
) {
    // `extinction_to_u8` expects EXT_MAX-normalized input; raw optical depth
    // must be divided before packing.
    let norm = weight / EXT_MAX;
    let q0 = extinction_to_u8(tint_od.x * norm);
    let q1 = extinction_to_u8(tint_od.y * norm);
    let q2 = extinction_to_u8(tint_od.z * norm);
    if q0 == 0 && q1 == 0 && q2 == 0 {
        return;
    }
    let width = data.view.output_size[0];
    let columns = width * data.view.output_size[1];
    let column = pixel.y * width + pixel.x;
    let n = data.params.params.slice_count_u32;
    atomic_or_device(
        data.occupancy_rgb.offset((column * 2 + slice / 32) as i64),
        1 << (slice % 32),
    );
    let idx = ext_rgb_index(pixel.x, pixel.y, slice, width, n);
    let prev = atomic_add_device(data.v_ext_rgb.offset(idx as i64), ext_rgb_pack(q0, q1, q2));
    if ext_rgb_field(prev, 0) + q0 > EXT_RGB_FIELD_MAX {
        atomic_min_device(data.overflow_rgb.offset(column as i64), slice);
    }
    if ext_rgb_field(prev, 1) + q1 > EXT_RGB_FIELD_MAX {
        atomic_min_device(data.overflow_rgb.offset((columns + column) as i64), slice);
    }
    if ext_rgb_field(prev, 2) + q2 > EXT_RGB_FIELD_MAX {
        atomic_min_device(
            data.overflow_rgb.offset((2 * columns + column) as i64),
            slice,
        );
    }
}

/// Writes full-resolution weighted OIT accumulators for billboards.
#[spirv(vertex)]
pub fn oit_accum_vert(
    #[spirv(push_constant)] push: &GraphicsPush,
    #[spirv(vertex_index)] vert_id: i32,
    #[spirv(instance_index)] inst_id: i32,
    #[spirv(position)] out_pos: &mut Vec4,
    out_color: &mut Vec4,
    out_view_z: &mut f32,
) {
    let data = push.vert::<OitAccumVertData>();
    let particle = data.particles[inst_id];
    let (clip, view_z, color) = oit_billboard_vertex(&data.view, particle, vert_id);
    *out_pos = clip;
    *out_color = color;
    *out_view_z = view_z;
}

#[spirv(fragment)]
pub fn oit_accum_frag(
    #[spirv(push_constant)] push: &GraphicsPush,
    #[spirv(descriptor_set = 0, binding = 0)] textures_3d: &RuntimeArray<Image3d>,
    #[spirv(descriptor_set = 2, binding = 0)] samplers: &RuntimeArray<Sampler>,
    #[spirv(frag_coord)] frag_coord: Vec4,
    color: Vec4,
    view_z: f32,
    #[spirv(location = 0)] out_accum_rgb: &mut Vec4,
    #[spirv(location = 1)] out_accum_moments: &mut Vec4,
) {
    let data = push.frag::<OitAccumFragData>();
    let pixel = frag_coord.truncate().truncate().as_uvec2();
    if pixel.x >= data.view.output_size[0] || pixel.y >= data.view.output_size[1] {
        *out_accum_rgb = Vec4::ZERO;
        *out_accum_moments = Vec4::ZERO;
        return;
    }

    let curve = &*data.params;
    let sample_z = view_z.min(curve.params.f);
    let dither = interleaved_gradient_noise(pixel.x, pixel.y);
    let w = ((warped_slice_of_z(curve, sample_z) - data.fog_sample_bias + dither)
        / curve.params.slice_count)
        .clamp(0.0, 1.0);
    let uv = (pixel.as_vec2() + 0.5) / UVec2::from_array(data.view.output_size).as_vec2();
    let v_int: Vec4 = unsafe { textures_3d.index(data.v_int_texture as usize) }.sample_by_lod(
        *unsafe { samplers.index(data.v_int_sampler as usize) },
        Vec3::new(uv.x, uv.y, w),
        0.0,
    );

    let alpha = color.w.clamp(0.0, 1.0);
    let fog_dim = v_int.w;
    let alpha_w = alpha * fog_dim;
    let neg_log = extinction_encode(alpha) * EXT_MAX;
    // Chromatic medium in front tints emitted radiance; OIT ordering remains
    // the scalar monochrome integral.
    let tint = if data.tinted_enable != 0 {
        let t: Vec4 = unsafe { textures_3d.index(data.v_tint_texture as usize) }.sample_by_lod(
            *unsafe { samplers.index(data.v_int_sampler as usize) },
            Vec3::new(uv.x, uv.y, w),
            0.0,
        );
        t.truncate()
    } else {
        Vec3::ONE
    };
    // `v_int.w` is both relative OIT ordering weight and absolute medium
    // transmittance; the RGB factor survives resolve's alpha normalization.
    *out_accum_rgb = (color.truncate() * fog_dim * tint * alpha_w).extend(0.0);
    *out_accum_moments = Vec4::new(alpha_w, neg_log, 0.0, 0.0);
}

/// Reduces per-tile reverse-Z depth bounds for Forward+ light culling.
/// Out-of-bounds threads contribute zero, representing infinite distance.
#[spirv(compute(threads(32, 32)))] // Matches TILE_SIZE × TILE_SIZE.
pub fn depth_reduce(
    #[spirv(push_constant)] data_ptr: &GpuPtr<DepthReduceData>,
    #[spirv(descriptor_set = 0, binding = 0)] textures: &RuntimeArray<Image2d>,
    #[spirv(workgroup)] shared_min: &mut [f32; TILE_AREA],
    #[spirv(workgroup)] shared_max: &mut [f32; TILE_AREA],
    #[spirv(workgroup_id)] group_id: UVec3,
    #[spirv(local_invocation_id)] local_id: UVec3,
    #[spirv(local_invocation_index)] local_index: u32,
) {
    let data = &**data_ptr;
    let pixel = group_id.truncate() * TILE_SIZE + local_id.truncate();

    let mut depth = 0.0f32;
    if pixel.x < data.screen_size[0] && pixel.y < data.screen_size[1] {
        depth = heap_fetch(textures, data.depth_texture_id, pixel, 0).x;
    }

    let li = local_index as usize;
    shared_min[li] = depth;
    shared_max[li] = depth;
    workgroup_memory_barrier_with_group_sync();

    // Uniform strides make barriers legal inside the reduction.
    let mut stride = (TILE_AREA / 2) as u32;
    while stride > 0 {
        if local_index < stride {
            let other = (local_index + stride) as usize;
            shared_min[li] = shared_min[li].min(shared_min[other]);
            shared_max[li] = shared_max[li].max(shared_max[other]);
        }
        workgroup_memory_barrier_with_group_sync();
        stride >>= 1;
    }

    if local_index == 0 {
        let tile = group_id.y * data.tile_count[0] + group_id.x;
        let mut bounds = data.tile_depth_bounds;
        bounds[tile] = TileDepthBounds {
            min_depth: shared_min[0],
            max_depth: shared_max[0],
        };
    }
}

const TILE_AREA: usize = (TILE_SIZE * TILE_SIZE) as usize;
const _: () = assert!(TILE_SIZE == 32, "threads(32, 32) above must match");

/// Screen-wide maximum finite view depth for the froxel far bound.
#[spirv(compute(threads(32, 32)))]
pub fn fog_depth_max(
    #[spirv(push_constant)] data_ptr: &GpuPtr<FogDepthMaxData>,
    #[spirv(descriptor_set = 0, binding = 0)] textures: &RuntimeArray<Image2d>,
    #[spirv(workgroup)] shared_max: &mut [u32; TILE_AREA],
    #[spirv(global_invocation_id)] gid: UVec3,
    #[spirv(local_invocation_index)] local_index: u32,
) {
    let data = &**data_ptr;
    let pixel = gid.truncate();

    let mut bits = 0u32;
    if pixel.x < data.screen_size[0] && pixel.y < data.screen_size[1] {
        let depth = heap_fetch(textures, data.depth_texture_id, pixel, 0).x;
        if depth > 0.0 {
            let view_z = data.depth_near_plane / depth;
            if view_z.is_finite() {
                bits = view_z.max(0.0).to_bits();
            }
        }
    }

    let li = local_index as usize;
    shared_max[li] = bits;
    workgroup_memory_barrier_with_group_sync();

    let mut stride = (TILE_AREA / 2) as u32;
    while stride > 0 {
        if local_index < stride {
            shared_max[li] = shared_max[li].max(shared_max[(local_index + stride) as usize]);
        }
        workgroup_memory_barrier_with_group_sync();
        stride >>= 1;
    }

    if local_index == 0 && shared_max[0] != 0 {
        atomic_max_device(data.max_depth_bits, shared_max[0]);
    }
}

/// Builds the shared froxel curve from depth and the prior event histogram.
#[spirv(compute(threads(1)))]
pub fn fog_params(#[spirv(push_constant)] data_ptr: &GpuPtr<FogParamsData>) {
    let data = &**data_ptr;
    let max_depth = f32::from_bits(data.max_depth_bits[0u32]);
    let params = froxel_params_from(max_depth, data.slice_count, data.a, data.f_min, data.f_max);
    // Consume and clear the previous raw-slice histogram.
    let mut hist = [0u32; FOG_SLICE_MAX as usize];
    let mut hist_ptr = data.hist;
    let mut i = 0u32;
    while i < FOG_SLICE_MAX {
        hist[i as usize] = hist_ptr[i];
        hist_ptr[i] = 0;
        i += 1;
    }
    let mut out = data.curve_out;
    out[0u32] = fog_curve_from(params, &hist, data.warp_gain, data.warp_bound);
}

/// Bins projected light spheres into one bounded list per 8×8 froxel tile.
/// Overflow marks the tile instead of dropping lights; `fog_light` evaluates
/// the complete local-light array for an overflowed tile.
#[spirv(compute(threads(64)))]
pub fn fog_light_grid(
    #[spirv(push_constant)] data_ptr: &GpuPtr<FogLightGridData>,
    #[spirv(global_invocation_id)] gid: UVec3,
) {
    let data = &**data_ptr;
    if gid.x >= data.light_count {
        return;
    }
    let bounds = fog_light_tile_bounds(&data.view, &data.lights[gid.x], data.tile_count);
    if bounds[0] > bounds[2] || bounds[1] > bounds[3] {
        return;
    }

    let mut y = bounds[1];
    while y <= bounds[3] {
        let mut x = bounds[0];
        while x <= bounds[2] {
            let tile = y * data.tile_count[0] + x;
            let slot = atomic_add_device(data.tile_counts.offset(tile as i64), 1);
            if slot < data.lights_per_tile {
                let mut indices = data.tile_indices;
                indices[tile * data.lights_per_tile + slot] = gid.x;
            } else {
                atomic_max_device(data.tile_overflow.offset(tile as i64), 1);
            }
            x += 1;
        }
        y += 1;
    }
}

/// Computes per-froxel scattering into `V_scatter`.
/// Sun uses HG phase, closed-form height-fog self-shadow, and the occluder;
/// local lights use finite-radius radiance, HG phase, and analytic
/// light-to-sample attenuation. There is no nested march or silent list loss.
const _: () = assert!(
    FOG_LIGHT_TILE == 8,
    "fog_light threads must match grid tiles"
);

#[spirv(compute(threads(8, 8)))]
pub fn fog_light(
    #[spirv(push_constant)] data_ptr: &GpuPtr<FogLightData>,
    #[spirv(descriptor_set = 0, binding = 0)] textures_3d: &RuntimeArray<Image3d>,
    #[spirv(descriptor_set = 1, binding = 0)] textures_rw: &RuntimeArray<
        Image!(3D, format = rgba16f, sampled = false),
    >,
    #[spirv(descriptor_set = 2, binding = 0)] samplers: &RuntimeArray<Sampler>,
    #[spirv(global_invocation_id)] gid: UVec3,
) {
    let data = &**data_ptr;
    if gid.x >= data.view.output_size[0] || gid.y >= data.view.output_size[1] {
        return;
    }

    let curve = &*data.params;
    let pixel = gid.truncate();
    let dir = ray_direction(&data.view, pixel);
    let forward = Vec3::from_array(data.view.camera_forward);
    let view_to_ray = 1.0 / forward.dot(dir).max(1.0e-6);
    let camera = Vec3::from_array(data.view.camera_position);
    let sun_dir = normalize_or(Vec3::from_array(data.sun_dir), Vec3::Y);
    let sun_color = Vec3::from_array(data.sun_color);
    let ambient_color = Vec3::from_array(data.ambient_color);
    let density = data.density.max(0.0);
    let falloff = data.height_falloff.max(0.0);
    let image = unsafe { textures_rw.index(data.v_scatter_texture as usize) };
    let light_tile = (gid.y / FOG_LIGHT_TILE) * data.local_tile_count[0] + gid.x / FOG_LIGHT_TILE;
    let light_overflow = data.local_light_count > 0 && data.local_tile_overflow[light_tile] != 0;
    let local_iterations = if light_overflow {
        data.local_light_count
    } else if data.local_light_count > 0 {
        data.local_tile_counts[light_tile].min(data.local_lights_per_tile)
    } else {
        0
    };

    let slice_count = curve.params.slice_count_u32;
    let mut i = 0u32;
    while i < slice_count {
        let view_z = z_of_warped_slice(curve, i as f32 + 0.5);
        let pos = camera + dir * (view_z * view_to_ray);
        let h = pos.y;
        let sigma_t = density * (-(falloff * (h - data.height_offset).max(0.0))).exp();

        let fog_self_shadow = transmittance(height_fog_optical_depth(
            h,
            sun_dir.y,
            0.0,
            f32::INFINITY,
            density,
            falloff,
            data.height_offset,
        ));
        let occluder_vis = fog_occluder_visibility(data, textures_3d, samplers, pos, sun_dir);
        let sun_vis = fog_self_shadow * occluder_vis;
        let phase = hg_phase(dir.dot(sun_dir), data.anisotropy_g);
        let mut local = Vec3::ZERO;
        if sigma_t > 0.0 {
            let mut j = 0u32;
            while j < local_iterations {
                let light_index = if light_overflow {
                    j
                } else {
                    data.local_tile_indices[light_tile * data.local_lights_per_tile + j]
                };
                local += fog_point_light_radiance(
                    dir,
                    pos,
                    &data.local_lights[light_index],
                    data.anisotropy_g,
                    density,
                    falloff,
                    data.height_offset,
                );
                j += 1;
            }
        }
        let tint = Vec3::from_array(height_gradient(
            h,
            data.gradient_bottom,
            data.gradient_top,
            data.gradient_offset,
            data.gradient_length,
        ));
        let lighting = (sun_color * (phase * sun_vis) + ambient_color + local) * tint;
        let scatter = lighting * sigma_t;

        unsafe {
            image.write(
                UVec3::new(gid.x, gid.y, i),
                Vec4::new(scatter.x, scatter.y, scatter.z, sigma_t),
            );
        }
        i += 1;
    }
}

fn fog_occluder_visibility(
    data: &FogLightData,
    textures_3d: &RuntimeArray<Image3d>,
    samplers: &RuntimeArray<Sampler>,
    pos: Vec3,
    sun_dir: Vec3,
) -> f32 {
    let steps = data.sun_occlusion_steps.min(64);
    if data.occluder_texture == 0 || steps == 0 {
        return 1.0;
    }

    let inv_extent = Vec3::from_array(data.occluder_world_inv_extent);
    let inv_abs = inv_extent.abs();
    let extent = Vec3::new(
        1.0 / inv_abs.x.max(1.0e-6),
        1.0 / inv_abs.y.max(1.0e-6),
        1.0 / inv_abs.z.max(1.0e-6),
    );
    let max_extent = extent.x.max(extent.y).max(extent.z);
    if max_extent <= 0.0 {
        return 1.0;
    }

    let mut denom = 0.0f32;
    let mut spacing = 1.0f32;
    let mut i = 0u32;
    while i < steps {
        denom += spacing;
        spacing *= 2.0;
        i += 1;
    }

    let image = unsafe { textures_3d.index(data.occluder_texture as usize) };
    let sampler = *unsafe { samplers.index(data.occluder_sampler as usize) };
    let world_min = Vec3::from_array(data.occluder_world_min);
    let lod_ramp = data.sun_occlusion_lod_ramp.max(0.0);
    let mut visibility = 1.0f32;
    let mut tap_spacing = max_extent / denom.max(1.0);
    let mut t = 0.0f32;
    let mut lod = 0.0f32;
    i = 0;
    while i < steps {
        t += tap_spacing;
        let uvw = (pos + sun_dir * t - world_min) * inv_extent;
        if in_unit_cube(uvw) {
            let opacity: Vec4 = image.sample_by_lod(sampler, uvw, lod);
            visibility = (visibility * (1.0 - opacity.x.clamp(0.0, 1.0))).max(0.0);
            if visibility <= 0.0 {
                break;
            }
        }
        tap_spacing *= 2.0;
        lod += lod_ramp;
        i += 1;
    }
    visibility
}

/// Prefix-integrates scattering and splatted extinction into `V_int`.
#[spirv(compute(threads(8, 8)))]
pub fn fog_integrate(
    #[spirv(push_constant)] data_ptr: &GpuPtr<FogIntegrateData>,
    #[spirv(descriptor_set = 0, binding = 0)] textures_3d: &RuntimeArray<Image3d>,
    #[spirv(descriptor_set = 1, binding = 0)] textures_rw: &RuntimeArray<
        Image!(3D, format = rgba16f, sampled = false),
    >,
    #[spirv(global_invocation_id)] gid: UVec3,
) {
    let data = &**data_ptr;
    if gid.x >= data.view.output_size[0] || gid.y >= data.view.output_size[1] {
        return;
    }

    let curve = &*data.params;
    let pixel = gid.truncate();
    let dir = ray_direction(&data.view, pixel);
    let forward = Vec3::from_array(data.view.camera_forward);
    let view_to_ray = 1.0 / forward.dot(dir).max(1.0e-6);
    let slice_count = data.volume_depth.min(curve.params.slice_count_u32);
    let scatter_image = unsafe { textures_3d.index(data.v_scatter_texture as usize) };
    let image = unsafe { textures_rw.index(data.v_int_texture as usize) };
    let oit_enabled = data.oit_enable != 0;
    let pixel_column = gid.y * data.view.output_size[0] + gid.x;
    let overflow_slice = if oit_enabled {
        data.overflow[pixel_column]
    } else {
        u32::MAX
    };
    // Skip sparse extinction fetches when occupancy proves the slice empty.
    let occ = if oit_enabled {
        [
            data.occupancy[pixel_column * 2],
            data.occupancy[pixel_column * 2 + 1],
        ]
    } else {
        [0, 0]
    };
    // Merge packed sparse RGB extinction as a per-channel multiplier in
    // `V_tint`; the scalar `V_int.a` and RGB history remain paired per slice.
    let tinted = data.tinted_enable != 0;
    let columns = data.view.output_size[0] * data.view.output_size[1];
    let occ_rgb = if tinted {
        [
            data.occupancy_rgb[pixel_column * 2],
            data.occupancy_rgb[pixel_column * 2 + 1],
        ]
    } else {
        [0, 0]
    };
    let overflow_rgb = if tinted {
        [
            data.overflow_rgb[pixel_column],
            data.overflow_rgb[columns + pixel_column],
            data.overflow_rgb[2 * columns + pixel_column],
        ]
    } else {
        [u32::MAX; 3]
    };
    let mut chroma = Vec3::ONE;

    let mut luminance = Vec3::ZERO;
    let mut throughput = 1.0f32;
    let mut zero_slice = ZERO_SLICE_NONE;
    let mut i = 0u32;
    while i < slice_count {
        if i >= overflow_slice {
            throughput = 0.0;
            if zero_slice == ZERO_SLICE_NONE {
                zero_slice = i;
            }
            unsafe {
                image.write(
                    UVec3::new(gid.x, gid.y, i),
                    Vec4::new(luminance.x, luminance.y, luminance.z, throughput),
                );
                if tinted {
                    let tint_image = textures_rw.index(data.v_tint_texture as usize);
                    tint_image.write(UVec3::new(gid.x, gid.y, i), chroma.extend(1.0));
                }
            }
            i += 1;
            continue;
        }

        let z0 = z_of_warped_slice(curve, i as f32);
        let z1 = z_of_warped_slice(curve, i as f32 + 1.0);
        let v_scatter: Vec4 = scatter_image.fetch(UVec3::new(gid.x, gid.y, i));
        let mut splat_od = 0.0;
        if oit_enabled && occ[(i / 32) as usize] & (1 << (i % 32)) != 0 {
            let (dword, lane) = ext_dword_index(
                gid.x,
                gid.y,
                i,
                data.view.output_size[0],
                curve.params.slice_count_u32,
            );
            let packed = data.v_ext[dword];
            splat_od = extinction_decode((packed >> (lane * 8)) & 0xff);
        }
        if tinted {
            let mut od = Vec3::ZERO;
            if occ_rgb[(i / 32) as usize] & (1 << (i % 32)) != 0 {
                let word = data.v_ext_rgb
                    [ext_rgb_index(gid.x, gid.y, i, data.view.output_size[0], slice_count)];
                od = Vec3::new(
                    ext_rgb_decode(ext_rgb_field(word, 0)),
                    ext_rgb_decode(ext_rgb_field(word, 1)),
                    ext_rgb_decode(ext_rgb_field(word, 2)),
                );
            }
            // Events sit at the slice front, so this slice's in-scatter is already
            // tinted. Each channel saturates from its first overflowing slice.
            chroma = Vec3::new(
                if i >= overflow_rgb[0] {
                    0.0
                } else {
                    chroma.x * transmittance(od.x)
                },
                if i >= overflow_rgb[1] {
                    0.0
                } else {
                    chroma.y * transmittance(od.y)
                },
                if i >= overflow_rgb[2] {
                    0.0
                } else {
                    chroma.z * transmittance(od.z)
                },
            );
        }
        let (added, step_t) = integrate_step(
            v_scatter.truncate().to_array(),
            v_scatter.w,
            splat_od,
            (z1 - z0) * view_to_ray,
            throughput,
        );
        luminance += Vec3::from_array(added) * chroma;
        throughput *= step_t;
        if zero_slice == ZERO_SLICE_NONE && throughput <= ZERO_TRANS_EPS {
            zero_slice = i;
        }
        unsafe {
            image.write(
                UVec3::new(gid.x, gid.y, i),
                Vec4::new(luminance.x, luminance.y, luminance.z, throughput),
            );
            if tinted {
                let tint_image = textures_rw.index(data.v_tint_texture as usize);
                tint_image.write(UVec3::new(gid.x, gid.y, i), chroma.extend(1.0));
            }
        }
        i += 1;
    }

    // Record the first zero-transmittance slice for depth priming.
    let mut zero_out = data.zero_slice;
    zero_out[pixel_column] = zero_slice;
}

/// Appends a conservative cover quad when every touched froxel column saturates.
#[spirv(compute(threads(8, 8)))]
pub fn fog_prime_spawn(
    #[spirv(push_constant)] data_ptr: &GpuPtr<FogPrimeSpawnData>,
    #[spirv(global_invocation_id)] gid: UVec3,
) {
    let data = &**data_ptr;
    let tiles_x = data.screen_size[0].div_ceil(PRIME_TILE);
    let tiles_y = data.screen_size[1].div_ceil(PRIME_TILE);
    if gid.x >= tiles_x || gid.y >= tiles_y {
        return;
    }

    let (fx0, fx1) = prime_froxel_range(gid.x, data.screen_size[0], data.froxel_size[0]);
    let (fy0, fy1) = prime_froxel_range(gid.y, data.screen_size[1], data.froxel_size[1]);
    let mut boundary = 0u32;
    let mut y = fy0;
    while y <= fy1 {
        let mut x = fx0;
        while x <= fx1 {
            let zs = data.zero_slice[y * data.froxel_size[0] + x];
            if zs == ZERO_SLICE_NONE {
                return;
            }
            boundary = boundary.max(zs);
            x += 1;
        }
        y += 1;
    }

    // Place the quad beyond the saturated slice and sampling margin.
    let curve = &*data.params;
    let depth = prime_quad_depth(
        curve,
        boundary as f32 + 1.0 + data.slice_margin,
        data.depth_near_plane,
    );
    if depth <= 0.0 {
        return;
    }

    // Indirect command word one stores instance count.
    let slot = atomic_add_device(data.draw_args.offset(1), 1);
    let mut quads = data.quads;
    quads[slot] = FogPrimeQuad {
        tile: gid.x | (gid.y << 16),
        depth,
    };
}

/// Emits depth-only tile quads at the zero-transmittance boundary.
#[spirv(vertex)]
pub fn fog_prime_vert(
    #[spirv(push_constant)] push: &GraphicsPush,
    #[spirv(vertex_index)] vert_id: i32,
    #[spirv(instance_index)] inst_id: i32,
    #[spirv(position)] out_pos: &mut Vec4,
) {
    let data = push.vert::<FogPrimeVertData>();
    let quad = data.quads[inst_id];
    let tile_x = quad.tile & 0xffff;
    let tile_y = quad.tile >> 16;
    let corner = oit_quad_corner(vert_id);
    // Scissor clips edge overhang; NDC and pixel Y share direction.
    let px = (tile_x * PRIME_TILE) as f32 + (corner.x * 0.5 + 0.5) * PRIME_TILE as f32;
    let py = (tile_y * PRIME_TILE) as f32 + (corner.y * 0.5 + 0.5) * PRIME_TILE as f32;
    let ndc_x = px / data.screen_size[0] as f32 * 2.0 - 1.0;
    let ndc_y = py / data.screen_size[1] as f32 * 2.0 - 1.0;
    *out_pos = Vec4::new(ndc_x, ndc_y, quad.depth, 1.0);
}

/// Depth-only pass; fixed-function depth testing performs compositing.
#[spirv(fragment)]
pub fn fog_prime_frag() {}

/// Composites `V_int` over HDR and continues beyond the froxel far bound.
#[spirv(compute(threads(8, 8)))]
pub fn fog_composite(
    #[spirv(push_constant)] data_ptr: &GpuPtr<FogCompositeData>,
    #[spirv(descriptor_set = 0, binding = 0)] textures_2d: &RuntimeArray<Image2d>,
    #[spirv(descriptor_set = 0, binding = 0)] textures_3d: &RuntimeArray<Image3d>,
    #[spirv(descriptor_set = 1, binding = 0)] textures_rw: &RuntimeArray<
        Image!(2D, format = rgba16f, sampled = false),
    >,
    #[spirv(descriptor_set = 2, binding = 0)] samplers: &RuntimeArray<Sampler>,
    #[spirv(global_invocation_id)] gid: UVec3,
) {
    let data = &**data_ptr;
    if gid.x >= data.view.output_size[0] || gid.y >= data.view.output_size[1] {
        return;
    }

    let curve = &*data.params;
    let pixel = gid.truncate();
    let depth = heap_fetch(textures_2d, data.depth_texture_id, pixel, 0).x;
    let scene_z = if depth > 0.0 {
        data.view.depth_near_plane / depth
    } else {
        f32::INFINITY
    };
    let sample_z = scene_z.min(curve.params.f);
    let dither = interleaved_gradient_noise(gid.x, gid.y);
    let w = ((warped_slice_of_z(curve, sample_z) - data.fog_sample_bias + dither)
        / curve.params.slice_count)
        .clamp(0.0, 1.0);
    let uv = (pixel.as_vec2() + 0.5) / UVec2::from_array(data.view.output_size).as_vec2();

    let v_int: Vec4 = unsafe { textures_3d.index(data.v_int_texture as usize) }.sample_by_lod(
        *unsafe { samplers.index(data.v_int_sampler as usize) },
        Vec3::new(uv.x, uv.y, w),
        0.0,
    );
    let hdr_image = unsafe { textures_rw.index(data.hdr_texture as usize) };
    let hdr: Vec4 = hdr_image.read(pixel);
    // Multiply scalar transmission by per-channel `V_tint`.
    let tint = if data.tinted_enable != 0 {
        let t: Vec4 = unsafe { textures_3d.index(data.v_tint_texture as usize) }.sample_by_lod(
            *unsafe { samplers.index(data.v_int_sampler as usize) },
            Vec3::new(uv.x, uv.y, w),
            0.0,
        );
        t.truncate()
    } else {
        Vec3::ONE
    };
    let mut rgb = hdr.truncate() * (v_int.w * tint) + v_int.truncate();

    if scene_z > curve.params.f {
        let dir = ray_direction(&data.view, pixel);
        let forward = Vec3::from_array(data.view.camera_forward);
        let view_to_ray = 1.0 / forward.dot(dir).max(1.0e-6);
        let t1 = if scene_z.is_finite() {
            scene_z * view_to_ray
        } else {
            f32::INFINITY
        };
        let od = height_fog_optical_depth(
            data.view.camera_position[1],
            dir.y,
            curve.params.f * view_to_ray,
            t1,
            data.density,
            data.height_falloff,
            data.height_offset,
        );
        let beyond_t = transmittance(od);
        // Continue the analytic model only beyond the froxel far bound.
        let sun_dir = normalize_or(Vec3::from_array(data.sun_dir), Vec3::Y);
        let phase = hg_phase(dir.dot(sun_dir), data.anisotropy_g);
        let light = Vec3::from_array(data.sun_color) * phase + Vec3::from_array(data.ambient_color);
        rgb = rgb * beyond_t + light * ((1.0 - beyond_t) * v_int.w);
    }

    unsafe {
        hdr_image.write(pixel, Vec4::new(rgb.x, rgb.y, rgb.z, hdr.w));
    }
}

/// Adds normalized weighted OIT emission to the already-extinguished HDR.
/// Do not apply event transmittance again; `fog_composite` already consumed it.
#[spirv(compute(threads(8, 8)))]
pub fn oit_resolve(
    #[spirv(push_constant)] data_ptr: &GpuPtr<OitResolveData>,
    #[spirv(descriptor_set = 0, binding = 0)] textures: &RuntimeArray<Image2d>,
    #[spirv(descriptor_set = 1, binding = 0)] textures_rw: &RuntimeArray<
        Image!(2D, format = rgba16f, sampled = false),
    >,
    #[spirv(global_invocation_id)] gid: UVec3,
) {
    let data = &**data_ptr;
    if gid.x >= data.view.output_size[0] || gid.y >= data.view.output_size[1] {
        return;
    }

    let pixel = gid.truncate();
    let moments = heap_fetch(textures, data.accum_moments_texture, pixel, 0);
    if moments.y == 0.0 {
        return;
    }

    let accum = heap_fetch(textures, data.accum_rgb_texture, pixel, 0);
    let hdr_image = unsafe { textures_rw.index(data.hdr_texture as usize) };
    // HDR already includes medium and splatted-extinction transport.
    let hdr: Vec4 = hdr_image.read(pixel);
    let rgb = Vec3::from_array(oit_resolve_rgb(
        accum.truncate().to_array(),
        moments.x,
        moments.y,
        hdr.truncate().to_array(),
    ));

    unsafe {
        hdr_image.write(pixel, Vec4::new(rgb.x, rgb.y, rgb.z, hdr.w));
    }
}

/// Performs a Blelloch exclusive scan over `MAX_TILES` light counts.
/// The tail is zero-filled, keeping the tree full and power-of-two.
/// Uniform stride loops make all barriers legal.
#[spirv(compute(threads(256)))]
pub fn prefix_sum(
    #[spirv(push_constant)] data_ptr: &GpuPtr<PrefixSumData>,
    #[spirv(workgroup)] shared_data: &mut [u32; MAX_TILES as usize],
    #[spirv(local_invocation_index)] local_index: u32,
) {
    let data = &**data_ptr;
    let tile_count = data.tile_count.min(MAX_TILES);

    let mut i = local_index;
    while i < tile_count {
        shared_data[i as usize] = data.tile_headers[i].light_count;
        i += 256;
    }
    let mut i = tile_count + local_index;
    while i < MAX_TILES {
        shared_data[i as usize] = 0;
        i += 256;
    }
    workgroup_memory_barrier_with_group_sync();

    // Up-sweep: fold pairs at each tree level.
    let mut offset = 1u32;
    let mut d = MAX_TILES >> 1;
    while d > 0 {
        workgroup_memory_barrier_with_group_sync();
        let mut i = local_index;
        while i < d {
            let ai = (offset * (2 * i + 1) - 1) as usize;
            let bi = (offset * (2 * i + 2) - 1) as usize;
            shared_data[bi] += shared_data[ai];
            i += 256;
        }
        offset *= 2;
        d >>= 1;
    }

    // Lane zero reads the completed root before down-sweep.
    if local_index == 0 {
        let mut total = data.total_light_count;
        *total = shared_data[(MAX_TILES - 1) as usize];
        shared_data[(MAX_TILES - 1) as usize] = 0;
    }

    // Down-sweep: distribute exclusive prefixes.
    let mut d = 1u32;
    while d < MAX_TILES {
        offset >>= 1;
        workgroup_memory_barrier_with_group_sync();
        let mut i = local_index;
        while i < d {
            let ai = (offset * (2 * i + 1) - 1) as usize;
            let bi = (offset * (2 * i + 2) - 1) as usize;
            let t = shared_data[ai];
            shared_data[ai] = shared_data[bi];
            shared_data[bi] += t;
            i += 256;
        }
        d *= 2;
    }
    workgroup_memory_barrier_with_group_sync();

    let mut i = local_index;
    while i < tile_count {
        let mut headers = data.tile_headers;
        headers[i].light_offset = shared_data[i as usize];
        i += 256;
    }
}
