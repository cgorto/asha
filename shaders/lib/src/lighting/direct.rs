//! Screen-space depth queries shared by lighting passes.

use abi_core::GpuPtr;
use abi_core::oct_decode;
use abi_light::{
    DepthMarchConfig, DepthMarchData, DepthMarchQuery, DepthMarchResult, LOCAL_SHADOW_FRACTION_ONE,
    LOCAL_SHADOW_REP_NONE, LOCAL_SHADOW_RESOLVE_DEPTH_REL, LOCAL_SHADOW_SLOT_EMPTY,
    LOCAL_SHADOW_SLOTS, LocalLightData, local_shadow_blind, local_shadow_fraction,
    local_shadow_slot_find, local_shadow_slot_get, local_shadow_state,
};
use abi_light::{SHADOW_STATE_FAILED, SHADOW_STATE_OCCLUDED, SHADOW_STATE_VISIBLE};
use abi_light::{
    light_field_gate, light_field_sample, point_light_identity_ramp_contribution,
    point_light_ramp_terms,
};
use glam::{IVec2, UVec2, UVec3, Vec2, Vec3, Vec4};
use spirv_std::image::Image2d;
use spirv_std::spirv;
use spirv_std::{Image, RuntimeArray, Sampler};

pub(crate) const DEPTH_BIAS: f32 = 0.000002;

#[inline(always)]
pub(crate) fn linearized_reverse_depth(depth: f32) -> f32 {
    if depth > 0.0 {
        1.0 / depth
    } else {
        f32::INFINITY
    }
}

#[inline(always)]
fn depth_fetch(depth: &Image2d, coord: IVec2, depth_size: UVec2) -> f32 {
    let hi = depth_size.as_ivec2() - IVec2::ONE;
    let pixel = coord.clamp(IVec2::ZERO, hi).as_uvec2();
    let sample: Vec4 = depth.fetch_with_lod(pixel, 0);
    sample.x
}

/// Reconstructs bilinear depth from point fetches for portable D32 support.
#[inline(always)]
pub(crate) fn depth_point_and_linear(depth: &Image2d, uv: Vec2, depth_size: UVec2) -> (f32, f32) {
    let size = depth_size.as_vec2();
    let texel = uv * size - Vec2::splat(0.5);
    let base_f = texel.floor();
    let base = base_f.as_ivec2();
    let frac = texel - base_f;
    let d00 = depth_fetch(depth, base, depth_size);
    let d10 = depth_fetch(depth, base + IVec2::X, depth_size);
    let d01 = depth_fetch(depth, base + IVec2::Y, depth_size);
    let d11 = depth_fetch(depth, base + IVec2::ONE, depth_size);
    let row0 = d00 + (d10 - d00) * frac.x;
    let row1 = d01 + (d11 - d01) * frac.x;
    let linear = row0 + (row1 - row0) * frac.y;
    let point = depth_fetch(depth, (uv * size).floor().as_ivec2(), depth_size);
    (point, linear)
}

/// Marches a clipped segment against reverse-Z depth.
/// Misses remain unresolved so callers can use geometry fallback.
#[inline(always)]
pub fn depth_raymarch(
    depth: &Image2d,
    depth_size: UVec2,
    query: &DepthMarchQuery,
    config: &DepthMarchConfig,
) -> DepthMarchResult {
    let start = Vec3::from_array(query.start_ndc);
    let end = Vec3::from_array(query.end_ndc);
    let delta = end - start;
    let start_uv = start.truncate() * 0.5 + Vec2::splat(0.5);
    let end_uv = end.truncate() * 0.5 + Vec2::splat(0.5);
    let ray_pixels = (end_uv - start_uv) * depth_size.as_vec2();
    let pixel_steps = ray_pixels.length() as u32;
    let step_count = config.linear_steps.min(pixel_steps).max(2);
    let thickness = config.depth_thickness / config.near_plane;

    let mut step = 0u32;
    while step < step_count {
        let candidate_t = (step as f32 + config.jitter) / step_count as f32;
        let candidate = start + delta * candidate_t;
        let uv = candidate.truncate() * 0.5 + Vec2::splat(0.5);
        let (point_sample, linear_sample) = depth_point_and_linear(depth, uv, depth_size);
        let linear_depth = linearized_reverse_depth(linear_sample);
        let point_depth = linearized_reverse_depth(point_sample);
        let far_surface = linear_depth.max(point_depth);
        let near_surface = linear_depth.min(point_depth);
        let ray_depth = linearized_reverse_depth(candidate.z);
        let distance = far_surface * (1.0 + DEPTH_BIAS) - ray_depth;
        let penetration = ray_depth - near_surface;
        let valid = config.continue_after_deep_penetration == 0 || penetration < thickness;

        if distance < 0.0 && valid {
            if penetration < thickness && distance < thickness {
                return DepthMarchResult {
                    hit: 1,
                    _pad0: 0,
                    hit_t: candidate_t,
                    hit_penetration: penetration * config.near_plane,
                    hit_uv: uv.to_array(),
                    _pad1: [0; 2],
                };
            }
            // Beyond-thickness crossings are unresolved, not occluders.
            if config.continue_after_deep_penetration == 0 {
                break;
            }
        }
        step += 1;
    }

    DepthMarchResult {
        hit: 0,
        _pad0: 0,
        hit_t: 1.0,
        hit_penetration: 0.0,
        hit_uv: end_uv.to_array(),
        _pad1: [0; 2],
    }
}

/// Batches depth queries for parity and capture tooling.
/// Production lighting calls [`depth_raymarch`] directly.
#[spirv(compute(threads(64)))]
pub fn depth_raymarch_queries(
    #[spirv(push_constant)] data_ptr: &GpuPtr<DepthMarchData>,
    #[spirv(descriptor_set = 0, binding = 0)] textures: &RuntimeArray<Image2d>,
    #[spirv(global_invocation_id)] gid: UVec3,
) {
    let data = &**data_ptr;
    if gid.x >= data.query_count {
        return;
    }

    let depth = unsafe { textures.index(data.depth_texture_id as usize) };
    let result = depth_raymarch(
        depth,
        UVec2::from_array(data.depth_size),
        &data.queries[gid.x],
        &data.config,
    );
    let mut results = data.results;
    results[gid.x] = result;
}

#[inline(always)]
fn exact_shadow_visibility(data: &LocalLightData, pixel_index: u32, light_index: u32) -> f32 {
    if data.shadow_states.is_null() {
        return 1.0;
    }
    let pixel_count = data.screen_size[0] * data.screen_size[1];
    let state = data.shadow_states[light_index * pixel_count + pixel_index];
    if state == SHADOW_STATE_VISIBLE {
        1.0
    } else if state == SHADOW_STATE_OCCLUDED {
        0.0
    } else if state == SHADOW_STATE_FAILED {
        -1.0
    } else {
        -1.0
    }
}

/// Returns binary neighbor visibility, or a negative weight-killer.
#[inline(always)]
fn slot_neighbor_visibility(data: &LocalLightData, texel_index: u32, light_index: u32) -> f32 {
    let word = data.slot_map[texel_index];
    let slot = local_shadow_slot_find(word, light_index);
    if slot == LOCAL_SHADOW_SLOTS {
        return -1.0;
    }
    let state = local_shadow_state(data.slot_state[texel_index * LOCAL_SHADOW_SLOTS + slot]);
    if state == SHADOW_STATE_VISIBLE || state == SHADOW_STATE_OCCLUDED {
        // Promoted edges store bounded disk-sample means.
        local_shadow_fraction(data.slot_fraction[texel_index * LOCAL_SHADOW_SLOTS + slot]) as f32
            / LOCAL_SHADOW_FRACTION_ONE as f32
    } else {
        -1.0
    }
}

/// Upsamples half-resolution visibility using light and depth matching.
/// Falls back to the pixel's own answer when neighbor weights vanish.
#[inline(always)]
fn slot_visibility_resolve(
    data: &LocalLightData,
    textures: &RuntimeArray<Image2d>,
    coord: UVec2,
    pixel_depth: f32,
    own_texel_index: u32,
    light_index: u32,
) -> f32 {
    let own = slot_neighbor_visibility(data, own_texel_index, light_index);
    if own < 0.0 {
        return -1.0;
    }
    let pixel_linear = linearized_reverse_depth(pixel_depth);
    let depth_tex = unsafe { textures.index(data.depth_texture_id as usize) };
    let half = UVec2::new(data.half_size[0], data.half_size[1]);
    let hi = half.as_ivec2() - IVec2::ONE;

    // Locate the pixel center in the half-resolution bilinear footprint.
    let hp = (coord.as_vec2() + Vec2::splat(0.5)) * 0.5 - Vec2::splat(0.5);
    let base_f = hp.floor();
    let base = base_f.as_ivec2();
    let frac = hp - base_f;

    let mut sum = 0.0f32;
    let mut weight_sum = 0.0f32;
    let mut corner = 0u32;
    while corner < 4 {
        let offset = IVec2::new((corner & 1) as i32, (corner >> 1) as i32);
        let t = (base + offset).clamp(IVec2::ZERO, hi).as_uvec2();
        let texel_index = t.y * data.half_size[0] + t.x;
        let bx = if offset.x == 0 { 1.0 - frac.x } else { frac.x };
        let by = if offset.y == 0 { 1.0 - frac.y } else { frac.y };
        let bilinear = bx * by;
        if bilinear > 0.0 {
            let visibility = slot_neighbor_visibility(data, texel_index, light_index);
            if visibility >= 0.0 {
                let rep = data.slot_rep[texel_index];
                if rep != LOCAL_SHADOW_REP_NONE {
                    let rep_coord = UVec2::new(rep & 0xFFFF, rep >> 16);
                    let rep_depth: Vec4 = depth_tex.fetch_with_lod(rep_coord, 0);
                    let rep_linear = linearized_reverse_depth(rep_depth.x);
                    if (rep_linear - pixel_linear).abs()
                        <= LOCAL_SHADOW_RESOLVE_DEPTH_REL * pixel_linear
                    {
                        sum += visibility * bilinear;
                        weight_sum += bilinear;
                    }
                }
            }
        }
        corner += 1;
    }
    if weight_sum > 0.0 {
        sum / weight_sum
    } else {
        own
    }
}

/// Computes exact post-opaque point-light ownership per surface pixel.
/// Traversal failures remain visibly distinct from shadow.
#[spirv(compute(threads(8, 8)))]
pub fn mesh_local_light(
    #[spirv(push_constant)] data_ptr: &GpuPtr<LocalLightData>,
    #[spirv(descriptor_set = 0, binding = 0)] textures: &RuntimeArray<Image2d>,
    #[spirv(descriptor_set = 1, binding = 0)] textures_rw: &RuntimeArray<
        Image!(2D, format = rgba16f, sampled = false),
    >,
    #[spirv(descriptor_set = 2, binding = 0)] samplers: &RuntimeArray<Sampler>,
    #[spirv(global_invocation_id)] gid: UVec3,
) {
    let data = &**data_ptr;
    if gid.x >= data.screen_size[0] || gid.y >= data.screen_size[1] {
        return;
    }
    let coord = gid.truncate();

    // Zero material marker denotes background and skips surface fetches.
    let marker = unsafe { textures.index(data.surface_material_texture_id as usize) }
        .fetch_with_lod(coord, 0)
        .x;
    if marker <= 0.0 {
        return;
    }
    let depth = unsafe { textures.index(data.depth_texture_id as usize) }
        .fetch_with_lod(coord, 0)
        .x;
    // Reverse-Z zero denotes cleared depth.
    if depth <= 0.0 {
        return;
    }

    let oct: Vec4 =
        unsafe { textures.index(data.surface_normal_texture_id as usize) }.fetch_with_lod(coord, 0);
    let normal = oct_decode(Vec2::new(oct.x, oct.y));
    // The forward MRT pass already applied texture and instance tint.
    let albedo: Vec3 = unsafe { textures.index(data.surface_albedo_texture_id as usize) }
        .fetch_with_lod(coord, 0)
        .truncate();

    // The raster matrix already accounts for +Y-down coordinates.
    let screen = Vec2::new(data.screen_size[0] as f32, data.screen_size[1] as f32);
    let ndc = (coord.as_vec2() + Vec2::splat(0.5)) / screen * 2.0 - Vec2::ONE;
    let h = data.clip_to_world * Vec4::new(ndc.x, ndc.y, depth, 1.0);
    let position_world = h.truncate() / h.w;

    let field = light_field_gate(
        light_field_sample(
            data.light_field,
            data.light_field_dims,
            data.light_field_cell_size,
            position_world,
        ),
        data.light_field_gate,
    );

    let material = data.materials[marker as u32 - 1];
    let pixel_index = coord.y * data.screen_size[0] + coord.x;
    // Shade selected slots; zero-score unselected lights add nothing.
    let slot_mode = !data.slot_map.is_null();
    let own_texel_index = if slot_mode {
        (coord.y / 2) * data.half_size[0] + coord.x / 2
    } else {
        0
    };
    let iterations = if slot_mode {
        LOCAL_SHADOW_SLOTS
    } else {
        data.light_count
    };
    let mut shadow_failed = false;
    let mut direct = Vec3::ZERO;
    if material.ramp_map == 0 {
        // Preserve the defined identity fallback and accumulation order.
        let mut i = 0u32;
        while i < iterations {
            let mut light_index = i;
            let mut visibility = 1.0f32;
            let mut active = true;
            if slot_mode {
                let word = data.slot_map[own_texel_index];
                light_index = local_shadow_slot_get(word, i);
                if light_index == LOCAL_SHADOW_SLOT_EMPTY || light_index >= data.light_count {
                    active = false;
                } else {
                    visibility = slot_visibility_resolve(
                        data,
                        textures,
                        coord,
                        depth,
                        own_texel_index,
                        light_index,
                    );
                }
            } else {
                visibility = exact_shadow_visibility(data, pixel_index, i);
            }
            if active {
                if visibility < 0.0 {
                    shadow_failed = true;
                } else {
                    direct += albedo
                        * point_light_identity_ramp_contribution(
                            normal,
                            position_world,
                            &data.lights[light_index],
                            data.wrap_w,
                        )
                        * field
                        * visibility;
                }
            }
            i += 1;
        }
    } else {
        let sampler_id = if material.ramp_sampler == 0 {
            data.ramp_default_sampler
        } else {
            material.ramp_sampler
        };
        let ramp = unsafe { textures.index(material.ramp_map as usize) };
        let sampler = *unsafe { samplers.index(sampler_id as usize) };
        let mut ramped = Vec3::ZERO;
        let mut i = 0u32;
        while i < iterations {
            let mut light_index = i;
            let mut visibility = 1.0f32;
            let mut active = true;
            if slot_mode {
                let word = data.slot_map[own_texel_index];
                light_index = local_shadow_slot_get(word, i);
                if light_index == LOCAL_SHADOW_SLOT_EMPTY || light_index >= data.light_count {
                    active = false;
                } else {
                    visibility = slot_visibility_resolve(
                        data,
                        textures,
                        coord,
                        depth,
                        own_texel_index,
                        light_index,
                    );
                }
            } else {
                visibility = exact_shadow_visibility(data, pixel_index, i);
            }
            if active {
                if visibility < 0.0 {
                    shadow_failed = true;
                } else {
                    let (index, scale) = point_light_ramp_terms(
                        normal,
                        position_world,
                        &data.lights[light_index],
                        data.wrap_w,
                        field,
                    );
                    let ramp_rgb: Vec3 = ramp
                        .sample_by_lod(sampler, Vec2::new((index * 255.0 + 0.5) / 256.0, 0.5), 0.0)
                        .truncate();
                    ramped += ramp_rgb * scale * visibility;
                }
            }
            i += 1;
        }
        direct = albedo * ramped;
    }

    if shadow_failed {
        direct += Vec3::new(64.0, 0.0, 64.0);
    }
    // Debug output encodes slot-0 answer provenance by color.
    if slot_mode && data.debug_overlay != 0 {
        let word = data.slot_state[own_texel_index * LOCAL_SHADOW_SLOTS];
        let age = (word >> 8) & 0xFF;
        let state = local_shadow_state(word);
        direct = if state == SHADOW_STATE_OCCLUDED {
            if age == 0 {
                Vec3::new(1.0, 0.0, 0.0)
            } else if local_shadow_blind(word) {
                Vec3::new(0.0, 1.0, 1.0)
            } else {
                Vec3::new(0.0, 1.0, 0.0)
            }
        } else if age == 0 {
            Vec3::new(1.0, 0.0, 0.0)
        } else if age >= 255 {
            Vec3::new(1.0, 1.0, 0.0)
        } else {
            Vec3::new(0.0, 0.0, 1.0)
        } * 2.0;
    }

    let hdr_image = unsafe { textures_rw.index(data.hdr_texture_id as usize) };
    let hdr: Vec4 = hdr_image.read(coord);
    unsafe {
        hdr_image.write(
            coord,
            Vec4::new(hdr.x + direct.x, hdr.y + direct.y, hdr.z + direct.z, hdr.w),
        );
    }
}
