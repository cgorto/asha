//! Bounded software BLAS/TLAS traversal for shadow any-hit queries.

use abi_core::GpuPtr;
use abi_light::{
    CwbvhNode, SHADOW_INVALID_PRIMITIVE, SHADOW_QUERY_FAILED, SHADOW_QUERY_OCCLUDED,
    SHADOW_QUERY_VISIBLE, SHADOW_STATE_FAILED, SHADOW_STATE_INACTIVE, SHADOW_STATE_OCCLUDED,
    SHADOW_STATE_UNRESOLVED, SHADOW_STATE_VISIBLE, ShadowBlas, ShadowBlasQueryData,
    ShadowQueryResult, ShadowSegment, ShadowTlasNode, ShadowWorld, ShadowWorldQueryData,
    ShadowWorldQueryResult, shadow_segment_triangle_t,
};
use abi_light::{DepthMarchQuery, MeshShadowData};
use glam::{Mat4, UVec2, UVec3, Vec2, Vec3, Vec4};
use spirv_std::RuntimeArray;
use spirv_std::image::Image2d;
use spirv_std::spirv;

const STACK_CAPACITY: usize = 32;

#[inline(always)]
fn extract_byte(word: u32, byte: u32) -> u32 {
    (word >> (byte * 8)) & 0xff
}

#[inline(always)]
fn first_bit_high(value: u32) -> u32 {
    31 - value.leading_zeros()
}

#[inline(always)]
fn safe_direction(value: f32) -> f32 {
    if value.abs() < 1.0e-30 {
        if value.to_bits() >> 31 != 0 {
            -1.0e-30
        } else {
            1.0e-30
        }
    } else {
        value
    }
}

#[inline(always)]
fn ray_octant_inv4(direction: Vec3) -> u32 {
    (if direction.x < 0.0 { 0 } else { 0x0404_0404 })
        | (if direction.y < 0.0 { 0 } else { 0x0202_0202 })
        | (if direction.z < 0.0 { 0 } else { 0x0101_0101 })
}

/// Returns the CWBVH mask: low 24 bits primitives, high 8 bits nodes.
#[inline(always)]
fn node_intersect(segment: &ShadowSegment, octant_inv4: u32, node: GpuPtr<CwbvhNode>) -> u32 {
    let origin = Vec3::from_array(segment.origin);
    let direction = Vec3::from_array(segment.direction);
    let safe = Vec3::new(
        safe_direction(direction.x),
        safe_direction(direction.y),
        safe_direction(direction.z),
    );
    let p = Vec3::new(
        f32::from_bits(node.words[0]),
        f32::from_bits(node.words[1]),
        f32::from_bits(node.words[2]),
    );
    let exponent_imask = node.words[3];
    let extent = Vec3::new(
        f32::from_bits(extract_byte(exponent_imask, 0) << 23),
        f32::from_bits(extract_byte(exponent_imask, 1) << 23),
        f32::from_bits(extract_byte(exponent_imask, 2) << 23),
    );
    let adjusted_direction_inverse = extent / safe;
    let adjusted_origin = (p - origin) / safe;
    let mut hit_mask = 0u32;

    let mut half = 0u32;
    while half < 2 {
        let meta4 = node.words[(6 + half) as usize];
        let is_inner4 = (meta4 & (meta4 << 1)) & 0x1010_1010;
        let inner_mask4 = (is_inner4 >> 4).wrapping_mul(0xff);
        let bit_index4 = (meta4 ^ (octant_inv4 & inner_mask4)) & 0x1f1f_1f1f;
        let child_bits4 = (meta4 >> 5) & 0x0707_0707;

        let q_lo_x = node.words[(8 + half) as usize];
        let q_hi_x = node.words[(10 + half) as usize];
        let q_lo_y = node.words[(12 + half) as usize];
        let q_hi_y = node.words[(14 + half) as usize];
        let q_lo_z = node.words[(16 + half) as usize];
        let q_hi_z = node.words[(18 + half) as usize];

        let x_min = if direction.x < 0.0 { q_hi_x } else { q_lo_x };
        let x_max = if direction.x < 0.0 { q_lo_x } else { q_hi_x };
        let y_min = if direction.y < 0.0 { q_hi_y } else { q_lo_y };
        let y_max = if direction.y < 0.0 { q_lo_y } else { q_hi_y };
        let z_min = if direction.z < 0.0 { q_hi_z } else { q_lo_z };
        let z_max = if direction.z < 0.0 { q_lo_z } else { q_hi_z };

        let mut child = 0u32;
        while child < 4 {
            let mut t_min = Vec3::new(
                extract_byte(x_min, child) as f32,
                extract_byte(y_min, child) as f32,
                extract_byte(z_min, child) as f32,
            );
            let mut t_max = Vec3::new(
                extract_byte(x_max, child) as f32,
                extract_byte(y_max, child) as f32,
                extract_byte(z_max, child) as f32,
            );
            t_min = t_min * adjusted_direction_inverse + adjusted_origin;
            t_max = t_max * adjusted_direction_inverse + adjusted_origin;
            let near = t_min.max_element().max(segment.t_min);
            let far = t_max.min_element().min(segment.t_max);
            if near <= far {
                let child_bits = extract_byte(child_bits4, child);
                let bit_index = extract_byte(bit_index4, child);
                hit_mask |= child_bits << bit_index;
            }
            child += 1;
        }
        half += 1;
    }

    hit_mask
}

#[inline(always)]
fn failed_result(node_tests: u32, triangle_tests: u32, max_stack_depth: u32) -> ShadowQueryResult {
    ShadowQueryResult {
        status: SHADOW_QUERY_FAILED,
        primitive_id: SHADOW_INVALID_PRIMITIVE,
        hit_t: f32::INFINITY,
        node_tests,
        triangle_tests,
        max_stack_depth,
    }
}

/// Traverses a BLAS with bounded resources and fail-closed errors.
#[inline(always)]
pub fn blas_any_hit(blas: &ShadowBlas, segment: &ShadowSegment) -> ShadowQueryResult {
    let mut result = ShadowQueryResult {
        status: SHADOW_QUERY_VISIBLE,
        primitive_id: SHADOW_INVALID_PRIMITIVE,
        hit_t: f32::INFINITY,
        node_tests: 0,
        triangle_tests: 0,
        max_stack_depth: 0,
    };
    if !segment.is_active() {
        return result;
    }
    if blas.nodes.is_null()
        || blas.primitive_ids.is_null()
        || blas.positions.is_null()
        || blas.indices.is_null()
        || blas.node_count == 0
        || blas.primitive_count == 0
    {
        return failed_result(0, 0, 0);
    }

    let direction = Vec3::from_array(segment.direction);
    let octant_inv4 = ray_octant_inv4(direction);
    let mut stack = [UVec2::ZERO; STACK_CAPACITY];
    let mut stack_size = 0u32;
    let mut current_group = UVec2::new(0, 0x8000_0000);

    loop {
        let mut primitive_group;
        if current_group.y & 0xff00_0000 != 0 {
            let hit_nodes = current_group.y;
            let child_offset = first_bit_high(hit_nodes);
            let child_base = current_group.x;
            current_group.y &= !(1u32 << child_offset);

            if current_group.y & 0xff00_0000 != 0 {
                if stack_size as usize >= STACK_CAPACITY {
                    return failed_result(
                        result.node_tests,
                        result.triangle_tests,
                        result.max_stack_depth,
                    );
                }
                stack[stack_size as usize] = current_group;
                stack_size += 1;
                result.max_stack_depth = result.max_stack_depth.max(stack_size);
            }

            let slot_index = (child_offset - 24) ^ (octant_inv4 & 0xff);
            let lower_bits = if slot_index == 0 {
                0
            } else {
                (1u32 << slot_index) - 1
            };
            let relative_index = (hit_nodes & lower_bits).count_ones();
            let node_index = child_base + relative_index;
            if node_index >= blas.node_count {
                return failed_result(
                    result.node_tests,
                    result.triangle_tests,
                    result.max_stack_depth,
                );
            }

            let node = blas.nodes.offset(node_index as i64);
            result.node_tests += 1;
            let hit_mask = node_intersect(segment, octant_inv4, node);
            let internal_mask = extract_byte(node.words[3], 3);
            current_group.x = node.words[4];
            current_group.y = (hit_mask & 0xff00_0000) | internal_mask;
            primitive_group = UVec2::new(node.words[5], hit_mask & 0x00ff_ffff);
        } else {
            primitive_group = current_group;
            current_group = UVec2::ZERO;
        }

        while primitive_group.y != 0 {
            let local_primitive = first_bit_high(primitive_group.y);
            primitive_group.y &= !(1u32 << local_primitive);
            let primitive_slot = primitive_group.x + local_primitive;
            if primitive_slot >= blas.primitive_count {
                return failed_result(
                    result.node_tests,
                    result.triangle_tests,
                    result.max_stack_depth,
                );
            }
            let primitive_id = blas.primitive_ids[primitive_slot];
            if primitive_id >= blas.primitive_count {
                return failed_result(
                    result.node_tests,
                    result.triangle_tests,
                    result.max_stack_depth,
                );
            }
            let index_base = primitive_id * 3;
            let i0 = blas.indices[index_base];
            let i1 = blas.indices[index_base + 1];
            let i2 = blas.indices[index_base + 2];
            let p0 = blas.positions[i0];
            let p1 = blas.positions[i1];
            let p2 = blas.positions[i2];
            result.triangle_tests += 1;
            let hit_t = shadow_segment_triangle_t(
                segment,
                Vec3::new(p0[0], p0[1], p0[2]),
                Vec3::new(p1[0], p1[1], p1[2]),
                Vec3::new(p2[0], p2[1], p2[2]),
            );
            if hit_t.is_finite() {
                result.status = SHADOW_QUERY_OCCLUDED;
                result.primitive_id = primitive_id;
                result.hit_t = hit_t;
                return result;
            }
        }

        if current_group.y & 0xff00_0000 == 0 {
            if stack_size == 0 {
                return result;
            }
            stack_size -= 1;
            current_group = stack[stack_size as usize];
        }
    }
}

#[inline(always)]
fn segment_intersects_aabb(segment: &ShadowSegment, node: GpuPtr<ShadowTlasNode>) -> bool {
    let origin = Vec3::from_array(segment.origin);
    let direction = Vec3::from_array(segment.direction);
    let safe = Vec3::new(
        safe_direction(direction.x),
        safe_direction(direction.y),
        safe_direction(direction.z),
    );
    let a = (Vec3::from_array(node.min) - origin) / safe;
    let b = (Vec3::from_array(node.max) - origin) / safe;
    let near = a.min(b).max_element().max(segment.t_min);
    let far = a.max(b).min_element().min(segment.t_max);
    near <= far
}

#[inline(always)]
fn world_failed(
    tlas_node_tests: u32,
    blas_node_tests: u32,
    triangle_tests: u32,
    max_stack_depth: u32,
) -> ShadowWorldQueryResult {
    ShadowWorldQueryResult {
        status: SHADOW_QUERY_FAILED,
        instance_id: u32::MAX,
        primitive_id: SHADOW_INVALID_PRIMITIVE,
        hit_t: f32::INFINITY,
        tlas_node_tests,
        blas_node_tests,
        triangle_tests,
        max_stack_depth,
    }
}

/// Traverses TLAS instances and BLAS geometry with preserved segment parameters.
#[inline(always)]
pub fn world_any_hit(world: &ShadowWorld, segment: &ShadowSegment) -> ShadowWorldQueryResult {
    let mut result = ShadowWorldQueryResult {
        status: SHADOW_QUERY_VISIBLE,
        instance_id: u32::MAX,
        primitive_id: SHADOW_INVALID_PRIMITIVE,
        hit_t: f32::INFINITY,
        tlas_node_tests: 0,
        blas_node_tests: 0,
        triangle_tests: 0,
        max_stack_depth: 0,
    };
    if !segment.is_active() || world.node_count == 0 {
        return result;
    }
    if world.nodes.is_null() || world.instances.is_null() || world.blases.is_null() {
        return world_failed(0, 0, 0, 0);
    }

    let mut stack = [0u32; STACK_CAPACITY];
    let mut stack_size = 1u32;
    stack[0] = 0;
    result.max_stack_depth = 1;
    while stack_size != 0 {
        stack_size -= 1;
        let node_index = stack[stack_size as usize];
        if node_index >= world.node_count {
            return world_failed(
                result.tlas_node_tests,
                result.blas_node_tests,
                result.triangle_tests,
                result.max_stack_depth,
            );
        }
        let node = world.nodes.offset(node_index as i64);
        result.tlas_node_tests += 1;
        if !segment_intersects_aabb(segment, node) {
            continue;
        }

        if node.leaf != 0 {
            let instance_index = node.child_or_instance;
            if instance_index >= world.instance_count {
                return world_failed(
                    result.tlas_node_tests,
                    result.blas_node_tests,
                    result.triangle_tests,
                    result.max_stack_depth,
                );
            }
            let instance = world.instances[instance_index];
            let local_segment = segment.transformed(instance.world_to_local);
            if !local_segment.is_active() {
                return world_failed(
                    result.tlas_node_tests,
                    result.blas_node_tests,
                    result.triangle_tests,
                    result.max_stack_depth,
                );
            }
            let blas_result = blas_any_hit(&world.blases[instance.blas_index], &local_segment);
            result.blas_node_tests += blas_result.node_tests;
            result.triangle_tests += blas_result.triangle_tests;
            result.max_stack_depth = result.max_stack_depth.max(blas_result.max_stack_depth);
            if blas_result.status == SHADOW_QUERY_FAILED {
                return world_failed(
                    result.tlas_node_tests,
                    result.blas_node_tests,
                    result.triangle_tests,
                    result.max_stack_depth,
                );
            }
            if blas_result.status == SHADOW_QUERY_OCCLUDED {
                result.status = SHADOW_QUERY_OCCLUDED;
                result.instance_id = instance.instance_id;
                result.primitive_id = blas_result.primitive_id;
                result.hit_t = blas_result.hit_t;
                return result;
            }
        } else {
            let first_child = node.child_or_instance;
            if first_child >= world.node_count - 1 || stack_size as usize > STACK_CAPACITY - 2 {
                return world_failed(
                    result.tlas_node_tests,
                    result.blas_node_tests,
                    result.triangle_tests,
                    result.max_stack_depth,
                );
            }
            stack[stack_size as usize] = first_child + 1;
            stack[stack_size as usize + 1] = first_child;
            stack_size += 2;
            result.max_stack_depth = result.max_stack_depth.max(stack_size);
        }
    }
    result
}

#[spirv(compute(threads(64)))]
pub fn shadow_world_queries(
    #[spirv(push_constant)] data_ptr: &GpuPtr<ShadowWorldQueryData>,
    #[spirv(global_invocation_id)] gid: UVec3,
) {
    let data = &**data_ptr;
    if gid.x >= data.query_count {
        return;
    }
    let result = world_any_hit(&data.world, &data.queries[gid.x]);
    let mut results = data.results;
    results[gid.x] = result;
}

#[derive(Clone, Copy)]
struct ShadowPair {
    segment: ShadowSegment,
    state: u32,
    active: bool,
}

#[inline(always)]
fn reconstruct_pair(
    data: &MeshShadowData,
    textures: &RuntimeArray<Image2d>,
    pair_index: u32,
) -> ShadowPair {
    let pixel_count = data.screen_size[0] * data.screen_size[1];
    let light_index = pair_index / pixel_count;
    let pixel_index = pair_index - light_index * pixel_count;
    let coord = UVec2::new(
        pixel_index % data.screen_size[0],
        pixel_index / data.screen_size[0],
    );
    let inactive = ShadowPair {
        segment: ShadowSegment::default(),
        state: SHADOW_STATE_INACTIVE,
        active: false,
    };
    let marker = unsafe { textures.index(data.surface_material_texture_id as usize) }
        .fetch_with_lod(coord, 0)
        .x;
    if marker <= 0.0 {
        return inactive;
    }
    let depth = unsafe { textures.index(data.depth_texture_id as usize) }
        .fetch_with_lod(coord, 0)
        .x;
    if depth <= 0.0 {
        return inactive;
    }

    let visible = ShadowPair {
        state: SHADOW_STATE_VISIBLE,
        ..inactive
    };
    let light = data.lights[light_index];
    if light.intensity <= 0.0 || light.radius <= 0.0 {
        return visible;
    }
    let screen = Vec2::new(data.screen_size[0] as f32, data.screen_size[1] as f32);
    let ndc = (coord.as_vec2() + Vec2::splat(0.5)) / screen * 2.0 - Vec2::ONE;
    let h = data.clip_to_world * Vec4::new(ndc.x, ndc.y, depth, 1.0);
    let position_world = h.truncate() / h.w;
    let light_position = Vec3::from_array(light.position);
    let to_light = light_position - position_world;
    if to_light.length_squared() >= light.radius * light.radius {
        return visible;
    }
    let segment = ShadowSegment::between(
        position_world,
        light_position,
        data.origin_bias,
        data.destination_bias,
    );
    if !segment.is_active() {
        return visible;
    }
    ShadowPair {
        segment,
        state: SHADOW_STATE_UNRESOLVED,
        active: true,
    }
}

#[inline(always)]
fn clip_plane(d0: f32, d1: f32, t0: &mut f32, t1: &mut f32) -> bool {
    if d0 < 0.0 && d1 < 0.0 {
        return false;
    }
    if d0 < 0.0 || d1 < 0.0 {
        let t = d0 / (d0 - d1);
        if d0 < 0.0 {
            *t0 = (*t0).max(t);
        } else {
            *t1 = (*t1).min(t);
        }
    }
    *t0 <= *t1
}

/// Describes a clipped segment and its homogeneous endpoint weights.
/// `t0`/`t1` map the clipped span to the input segment.
#[derive(Clone, Copy)]
pub(crate) struct ProjectedSegment {
    pub query: DepthMarchQuery,
    pub projected: bool,
    pub t0: f32,
    pub t1: f32,
    pub w0: f32,
    pub w1: f32,
}

/// Clips against Vulkan's homogeneous view volume and positive-w guard.
#[inline(always)]
pub(crate) fn project_segment(world_to_clip: &Mat4, segment: &ShadowSegment) -> ProjectedSegment {
    let origin = Vec3::from_array(segment.origin);
    let direction = Vec3::from_array(segment.direction);
    let a_world = origin + direction * segment.t_min;
    let b_world = origin + direction * segment.t_max;
    let a = *world_to_clip * a_world.extend(1.0);
    let b = *world_to_clip * b_world.extend(1.0);
    let mut t0 = 0.0;
    let mut t1 = 1.0;
    let clipped = clip_plane(a.x + a.w, b.x + b.w, &mut t0, &mut t1)
        && clip_plane(a.w - a.x, b.w - b.x, &mut t0, &mut t1)
        && clip_plane(a.y + a.w, b.y + b.w, &mut t0, &mut t1)
        && clip_plane(a.w - a.y, b.w - b.y, &mut t0, &mut t1)
        && clip_plane(a.z, b.z, &mut t0, &mut t1)
        && clip_plane(a.w - a.z, b.w - b.z, &mut t0, &mut t1)
        && clip_plane(a.w - 1.0e-6, b.w - 1.0e-6, &mut t0, &mut t1);
    let ac = a + (b - a) * t0;
    let bc = a + (b - a) * t1;
    let start = ac.truncate() / ac.w;
    let end = bc.truncate() / bc.w;
    let projected = clipped && start.is_finite() && end.is_finite();
    ProjectedSegment {
        query: DepthMarchQuery {
            start_ndc: start.to_array(),
            _pad0: 0.0,
            end_ndc: end.to_array(),
            _pad1: 0.0,
        },
        projected,
        t0,
        t1,
        w0: ac.w,
        w1: bc.w,
    }
}

/// Computes one full world traversal for each active light-surface pair.
#[spirv(compute(threads(64)))]
pub fn mesh_exact_shadow_mask(
    #[spirv(push_constant)] data_ptr: &GpuPtr<MeshShadowData>,
    #[spirv(descriptor_set = 0, binding = 0)] textures: &RuntimeArray<Image2d>,
    #[spirv(global_invocation_id)] gid: UVec3,
) {
    let data = &**data_ptr;
    let pair_count = data.screen_size[0] * data.screen_size[1] * data.light_count;
    if gid.x >= pair_count {
        return;
    }
    let pair = reconstruct_pair(data, textures, gid.x);
    let state = if pair.active {
        let result = world_any_hit(&data.world, &pair.segment);
        if result.status == SHADOW_QUERY_OCCLUDED {
            SHADOW_STATE_OCCLUDED
        } else if result.status == SHADOW_QUERY_FAILED {
            SHADOW_STATE_FAILED
        } else {
            SHADOW_STATE_VISIBLE
        }
    } else {
        pair.state
    };
    let mut states = data.states;
    states[gid.x] = state;
}

#[spirv(compute(threads(64)))]
pub fn shadow_blas_queries(
    #[spirv(push_constant)] data_ptr: &GpuPtr<ShadowBlasQueryData>,
    #[spirv(global_invocation_id)] gid: UVec3,
) {
    let data = &**data_ptr;
    if gid.x >= data.query_count {
        return;
    }
    let result = blas_any_hit(&data.blas, &data.queries[gid.x]);
    let mut results = data.results;
    results[gid.x] = result;
}
