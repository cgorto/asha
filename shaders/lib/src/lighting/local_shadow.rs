//! Half-resolution local-shadow selection, tracing, and reconstruction.
//!
//! Temporal history is checked, never blindly trusted: visible history expires
//! by age or integrated light drift; camera-blind occlusions use age leashes,
//! shortened for dynamic or drifting blockers. Visible occlusions revalidate
//! with a symmetric dual-depth tap at the stored hit parameter, while failed
//! checks ray trace in the same frame. Reprojection must be on-screen, match
//! the receiver depth and light identity, and otherwise counts invalidation
//! and traces authoritatively. Refreshes are bounded and birth phases staggered;
//! center-ray answers and same-frame penumbra reconstruction are deterministic.
//! `refresh_interval == 0` disables every reuse path.

use super::direct::{depth_point_and_linear, depth_raymarch, linearized_reverse_depth};
use super::mesh_shadow::{ProjectedSegment, project_segment, world_any_hit};
use crate::core::util::atomic_add_device;
use abi_core::GpuPtr;
use abi_core::oct_decode;
use abi_light::{
    LOCAL_SHADOW_AGE_MAX, LOCAL_SHADOW_BLIND, LOCAL_SHADOW_DYNAMIC, LOCAL_SHADOW_DYNAMIC_LEASH,
    LOCAL_SHADOW_FRACTION_ONE, LOCAL_SHADOW_REP_NONE, LOCAL_SHADOW_REPROJECT_DEPTH_ABS,
    LOCAL_SHADOW_RESOLVE_DEPTH_REL, LOCAL_SHADOW_SLOT_EMPTY, LOCAL_SHADOW_SLOT_WORD_EMPTY,
    LOCAL_SHADOW_SLOTS, LocalShadowArgsData, LocalShadowData, local_shadow_age, local_shadow_blind,
    local_shadow_dynamic, local_shadow_fresh_age, local_shadow_hit_q, local_shadow_pack,
    local_shadow_slot_find, local_shadow_slot_get, local_shadow_slot_set, local_shadow_state,
};
use abi_light::{PointLight, point_light_contribution};
use abi_light::{
    SHADOW_QUERY_FAILED, SHADOW_QUERY_OCCLUDED, SHADOW_STATE_FAILED, SHADOW_STATE_INACTIVE,
    SHADOW_STATE_OCCLUDED, SHADOW_STATE_VISIBLE, SHADOW_TLAS_INSTANCE_DYNAMIC, ShadowSegment,
};
use glam::{UVec2, UVec3, Vec2, Vec3, Vec4};
use spirv_std::RuntimeArray;
use spirv_std::image::Image2d;
use spirv_std::spirv;

/// Computes the slot-ranking luminance of an unshadowed contribution.
#[inline(always)]
fn contribution_score(contribution: Vec3) -> f32 {
    contribution.x * 0.2126 + contribution.y * 0.7152 + contribution.z * 0.0722
}

/// Append to one of the two request queues; `high` selects counter word 0
/// and the high buffer, otherwise word 1 and the low buffer.
#[inline(always)]
fn enqueue_request(data: &LocalShadowData, id: u32, high: bool) -> bool {
    let counters = data.counters.cast::<u32>();
    let word = if high { COUNTER_HIGH } else { COUNTER_LOW };
    let slot = atomic_add_device(counters.offset(word as i64), 1);
    if slot < data.request_capacity {
        let mut requests = if high {
            data.requests_high
        } else {
            data.requests_low
        };
        requests[slot] = id;
        true
    } else {
        atomic_add_device(counters.offset(COUNTER_OVERFLOW as i64), 1);
        false
    }
}

/// Reconstructs receiver world position from final reverse-Z depth.
#[inline(always)]
fn receiver_world(data: &LocalShadowData, coord: UVec2, depth: f32) -> Vec3 {
    let screen = Vec2::new(data.screen_size[0] as f32, data.screen_size[1] as f32);
    let ndc = (coord.as_vec2() + Vec2::splat(0.5)) / screen * 2.0 - Vec2::ONE;
    let h = data.clip_to_world * Vec4::new(ndc.x, ndc.y, depth, 1.0);
    h.truncate() / h.w
}

/// Bit 31 of a request id marks a promoted (EMA-blending) edge sample.
const REQUEST_PROMOTED: u32 = 1 << 31;

/// Counter slots (u32 offsets into `LocalShadowCounters`).
const COUNTER_HIGH: u64 = 0;
const COUNTER_LOW: u64 = 1;
const COUNTER_OVERFLOW: u64 = 2;
const COUNTER_VALIDATED: u64 = 4;
const COUNTER_REUSED: u64 = 5;
const COUNTER_INVALIDATED: u64 = 6;
const COUNTER_CONTACT: u64 = 7;
const COUNTER_SERVICED_HIGH: u64 = 8;
const COUNTER_SERVICED_LOW: u64 = 9;
const COUNTER_ACTIVE: u64 = 10;
const COUNTER_PROMOTED: u64 = 11;
const COUNTER_BLIND: u64 = 12;
const COUNTER_CHURN: u64 = 13;

#[inline(always)]
fn bump(data: &LocalShadowData, which: u64) {
    atomic_add_device(data.counters.cast::<u32>().offset(which as i64), 1);
}

/// Validates one stored occlusion hit at parameter `t` on the current
/// receiver-to-light segment. Project the hit, compare a symmetric dual-depth
/// band, and accept only a visible surface there; the symmetric band avoids
/// refuting hits exactly on the camera-facing surface. Off-screen,
/// behind-surface, or failed projections never validate.
#[inline(always)]
fn validate_occlusion(
    data: &LocalShadowData,
    depth_tex: &Image2d,
    segment: &ShadowSegment,
    t: f32,
) -> bool {
    let origin = Vec3::from_array(segment.origin);
    let direction = Vec3::from_array(segment.direction);
    let point = origin + direction * t;
    let clip = data.world_to_clip * point.extend(1.0);
    if clip.w <= 1.0e-6 {
        return false;
    }
    let ndc = clip.truncate() / clip.w;
    if ndc.x < -1.0 || ndc.x > 1.0 || ndc.y < -1.0 || ndc.y > 1.0 || ndc.z <= 0.0 || ndc.z > 1.0 {
        return false;
    }
    let uv = ndc.truncate() * 0.5 + Vec2::splat(0.5);
    let size = UVec2::new(data.screen_size[0], data.screen_size[1]);
    let (point_sample, linear_sample) = depth_point_and_linear(depth_tex, uv, size);
    let linear_depth = linearized_reverse_depth(linear_sample);
    let point_depth = linearized_reverse_depth(point_sample);
    let far_surface = linear_depth.max(point_depth);
    let near_surface = linear_depth.min(point_depth);
    let ray_depth = linearized_reverse_depth(ndc.z);
    let band = data.validate_thickness / data.near_plane;
    ray_depth > near_surface - band && ray_depth < far_surface + band
}

/// Detects mixed visible/occluded states around the previous-frame texel.
/// The lookup is a conservative edge promotion: reprojection drift can add
/// rays and cost budget, but cannot change the visibility answer.
#[inline(always)]
fn edge_promoted(data: &LocalShadowData, texel: UVec2, light_index: u32) -> bool {
    let mut seen_visible = false;
    let mut seen_occluded = false;
    let hi_x = data.half_size[0] - 1;
    let hi_y = data.half_size[1] - 1;
    let mut n = 0u32;
    while n < 9 {
        let dx = (n % 3) as i32 - 1;
        let dy = (n / 3) as i32 - 1;
        let nx = (texel.x as i32 + dx).clamp(0, hi_x as i32) as u32;
        let ny = (texel.y as i32 + dy).clamp(0, hi_y as i32) as u32;
        let t = ny * data.half_size[0] + nx;
        let word = data.slot_map_prev[t];
        let slot = local_shadow_slot_find(word, light_index);
        if slot == LOCAL_SHADOW_SLOTS {
            n += 1;
            continue;
        }
        let state = local_shadow_state(data.slot_state_prev[t * LOCAL_SHADOW_SLOTS + slot]);
        if state == SHADOW_STATE_VISIBLE {
            seen_visible = true;
        } else if state == SHADOW_STATE_OCCLUDED {
            seen_occluded = true;
        }
        n += 1;
    }
    seen_visible && seen_occluded
}

/// Performs a short near-field screen-space contact march toward the light.
/// A crossing is packed with a perspective-correct hit parameter. The projected
/// hit `u` is converted through `w1 + u * (w0 - w1)` back through the clipped
/// contact interval and then into the full segment's `t`; misses return zero.
#[inline(always)]
fn contact_answer(
    data: &LocalShadowData,
    depth_tex: &Image2d,
    light: &PointLight,
    position_world: Vec3,
) -> u32 {
    if data.contact.linear_steps == 0 {
        return 0;
    }
    let segment = ShadowSegment::between(
        position_world,
        Vec3::from_array(light.position),
        data.origin_bias,
        data.destination_bias,
    );
    if !segment.is_active() {
        return 0;
    }
    // Limit contact tracing to the configured near-field reach.
    let span = Vec3::from_array(segment.direction).length();
    if span <= 0.0 {
        return 0;
    }
    let reach = (segment.t_min + data.contact_distance / span).min(segment.t_max);
    if reach <= segment.t_min {
        return 0;
    }
    let contact = ShadowSegment {
        t_max: reach,
        ..segment
    };
    let projected = project_segment(&data.world_to_clip, &contact);
    if !projected.projected {
        return 0;
    }
    let size = UVec2::new(data.screen_size[0], data.screen_size[1]);
    let result = depth_raymarch(depth_tex, size, &projected.query, &data.contact);
    if result.hit == 0 {
        return 0;
    }
    // Convert the screen-space hit back to full-segment parameter space.
    let ProjectedSegment { t0, t1, w0, w1, .. } = projected;
    let u = result.hit_t;
    let denominator = w1 + u * (w0 - w1);
    if denominator <= 0.0 {
        return 0;
    }
    let s_clip = u * w0 / denominator;
    let s_contact = t0 + s_clip * (t1 - t0);
    let t_world = contact.t_min + s_contact * (contact.t_max - contact.t_min);
    bump(data, COUNTER_CONTACT);
    local_shadow_pack(
        SHADOW_STATE_OCCLUDED,
        0,
        (t_world.clamp(0.0, 1.0) * 65535.0) as u32,
    )
}

/// Answers a slot from history, returning answered and carried states.
/// Carried states require low-queue refresh; two zeros require high priority.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn history_answer(
    data: &LocalShadowData,
    depth_tex: &Image2d,
    light_index: u32,
    light: &PointLight,
    position_world: Vec3,
    reprojectable: bool,
    prev_texel_index: u32,
) -> (u32, u32) {
    if !reprojectable {
        bump(data, COUNTER_INVALIDATED);
        return (0, 0);
    }
    // Host remapping preserves history across light reordering.
    let prev_id = data.light_remap[light_index];
    if prev_id == LOCAL_SHADOW_SLOT_EMPTY {
        bump(data, COUNTER_INVALIDATED);
        return (0, 0);
    }
    let prev_word = data.slot_map_prev[prev_texel_index];
    let prev_slot = local_shadow_slot_find(prev_word, prev_id);
    if prev_slot == LOCAL_SHADOW_SLOTS {
        bump(data, COUNTER_INVALIDATED);
        return (0, 0);
    }
    let prev_state_word = data.slot_state_prev[prev_texel_index * LOCAL_SHADOW_SLOTS + prev_slot];
    let state = local_shadow_state(prev_state_word);
    let age = local_shadow_age(prev_state_word) + 1;

    // Large light motion invalidates visible and blind history.
    let prev_light = data.prev_lights[prev_id];
    let light_delta =
        (Vec3::from_array(light.position) - Vec3::from_array(prev_light.position)).length_squared();
    let epsilon2 = data.light_epsilon * data.light_epsilon;
    let teleported = light_delta > epsilon2;

    if state == SHADOW_STATE_OCCLUDED {
        if local_shadow_blind(prev_state_word) {
            // Blind hits cannot be depth-validated; bound their reuse by age.
            if teleported {
                bump(data, COUNTER_INVALIDATED);
                return (0, 0);
            }
            let dynamic = local_shadow_dynamic(prev_state_word);
            // Compare integrated squared displacement against the slack.
            let slack2 = epsilon2 * 0.0625;
            let drifted = (age * age) as f32 * light_delta > slack2;
            let leash = if data.occluded_refresh == 0 {
                0
            } else if dynamic || drifted {
                LOCAL_SHADOW_DYNAMIC_LEASH
            } else {
                data.occluded_refresh
            };
            let mut state_bits = SHADOW_STATE_OCCLUDED | LOCAL_SHADOW_BLIND;
            if dynamic {
                state_bits |= LOCAL_SHADOW_DYNAMIC;
            }
            let word = local_shadow_pack(
                state_bits,
                age.min(LOCAL_SHADOW_AGE_MAX),
                local_shadow_hit_q(prev_state_word),
            );
            if leash != 0 && age >= leash {
                // Dynamic or drifting blind history requires high-priority reproof.
                if dynamic || drifted {
                    bump(data, COUNTER_INVALIDATED);
                    return (0, 0);
                }
                return (0, word);
            }
            bump(data, COUNTER_BLIND);
            return (word, 0);
        }
        let segment = ShadowSegment::between(
            position_world,
            Vec3::from_array(light.position),
            data.origin_bias,
            data.destination_bias,
        );
        if segment.is_active() {
            let t = local_shadow_hit_q(prev_state_word) as f32 / 65535.0;
            if validate_occlusion(data, depth_tex, &segment, t) {
                // Depth validation alone can follow a moving planar occluder.
                let slack2 = epsilon2 * 0.0625;
                if (age * age) as f32 * light_delta > slack2 {
                    bump(data, COUNTER_INVALIDATED);
                    return (0, 0);
                }
                bump(data, COUNTER_VALIDATED);
                let word = local_shadow_pack(
                    SHADOW_STATE_OCCLUDED,
                    age.min(LOCAL_SHADOW_AGE_MAX),
                    local_shadow_hit_q(prev_state_word),
                );
                // Periodic reproof bounds validation false positives.
                if data.occluded_refresh != 0 && age >= data.occluded_refresh {
                    return (0, word);
                }
                return (word, 0);
            }
        }
        bump(data, COUNTER_INVALIDATED);
        return (0, 0);
    }
    if state == SHADOW_STATE_VISIBLE && !teleported {
        // Visible history expires by age or integrated light displacement.
        let slack2 = epsilon2 * 0.0625;
        let drifted = (age * age) as f32 * light_delta > slack2;
        if age < data.refresh_interval && !drifted {
            bump(data, COUNTER_REUSED);
            return (local_shadow_pack(SHADOW_STATE_VISIBLE, age, 0), 0);
        }
        // Carry expired visibility while low-priority tracing re-proves it.
        return (
            0,
            local_shadow_pack(SHADOW_STATE_VISIBLE, age.min(LOCAL_SHADOW_AGE_MAX), 0),
        );
    }
    if state == SHADOW_STATE_VISIBLE && teleported {
        bump(data, COUNTER_INVALIDATED);
    }
    (0, 0)
}

/// Selects top-K lights, reuses validated history, and queues unresolved slots.
#[spirv(compute(threads(8, 8)))]
pub fn local_shadow_select(
    #[spirv(push_constant)] data_ptr: &GpuPtr<LocalShadowData>,
    #[spirv(descriptor_set = 0, binding = 0)] textures: &RuntimeArray<Image2d>,
    #[spirv(global_invocation_id)] gid: UVec3,
) {
    let data = &**data_ptr;
    if gid.x >= data.half_size[0] || gid.y >= data.half_size[1] {
        return;
    }
    let texel = gid.truncate();
    let texel_index = texel.y * data.half_size[0] + texel.x;
    let mut slot_map = data.slot_map;
    let mut slot_state = data.slot_state;
    let mut slot_rep = data.slot_rep;
    let mut rep_depth_out = data.rep_depth;

    // Choose the first covered corner; skip out-of-range edge corners.
    let marker_tex = unsafe { textures.index(data.surface_material_texture_id as usize) };
    let depth_tex = unsafe { textures.index(data.depth_texture_id as usize) };
    // Promote edges using the previous frame's dominant light.
    let reuse_enabled = data.history_valid != 0 && data.refresh_interval != 0;
    // Keep representatives fixed; blur owns sub-texel edge softness.
    let mut rep = LOCAL_SHADOW_REP_NONE;
    let mut rep_depth = 0.0f32;
    let mut corner = 0u32;
    while corner < 4 {
        let j = corner;
        let c = texel * 2 + UVec2::new(j & 1, j >> 1);
        if c.x < data.screen_size[0] && c.y < data.screen_size[1] {
            let marker: Vec4 = marker_tex.fetch_with_lod(c, 0);
            let depth: Vec4 = depth_tex.fetch_with_lod(c, 0);
            if marker.x > 0.0 && depth.x > 0.0 {
                rep = c.x | (c.y << 16);
                rep_depth = depth.x;
                corner = 4;
                continue;
            }
        }
        corner += 1;
    }
    slot_rep[texel_index] = rep;
    rep_depth_out[texel_index] = rep_depth;

    let mut slot_fraction = data.slot_fraction;
    if rep == LOCAL_SHADOW_REP_NONE {
        slot_map[texel_index] = LOCAL_SHADOW_SLOT_WORD_EMPTY;
        let mut slot = 0u32;
        while slot < LOCAL_SHADOW_SLOTS {
            slot_state[texel_index * LOCAL_SHADOW_SLOTS + slot] = SHADOW_STATE_INACTIVE;
            slot_fraction[texel_index * LOCAL_SHADOW_SLOTS + slot] = LOCAL_SHADOW_FRACTION_ONE;
            slot += 1;
        }
        return;
    }

    let coord = UVec2::new(rep & 0xFFFF, rep >> 16);
    let position_world = receiver_world(data, coord, rep_depth);
    let oct: Vec4 =
        unsafe { textures.index(data.surface_normal_texture_id as usize) }.fetch_with_lod(coord, 0);
    let normal = oct_decode(Vec2::new(oct.x, oct.y));

    // Reuse requires on-screen reprojection and matching previous depth.
    let mut reprojectable = false;
    let mut prev_texel_index = 0u32;
    if reuse_enabled {
        let prev_clip = data.prev_world_to_clip * position_world.extend(1.0);
        if prev_clip.w > 1.0e-6 {
            let prev_ndc = prev_clip.truncate() / prev_clip.w;
            if prev_ndc.x >= -1.0
                && prev_ndc.x <= 1.0
                && prev_ndc.y >= -1.0
                && prev_ndc.y <= 1.0
                && prev_ndc.z > 0.0
                && prev_ndc.z <= 1.0
            {
                let prev_uv = prev_ndc.truncate() * 0.5 + Vec2::splat(0.5);
                let screen = Vec2::new(data.screen_size[0] as f32, data.screen_size[1] as f32);
                let prev_pixel = (prev_uv * screen).as_uvec2();
                let prev_texel = UVec2::new(
                    (prev_pixel.x / 2).min(data.half_size[0] - 1),
                    (prev_pixel.y / 2).min(data.half_size[1] - 1),
                );
                prev_texel_index = prev_texel.y * data.half_size[0] + prev_texel.x;
                let stored = data.rep_depth_prev[prev_texel_index];
                if stored > 0.0 {
                    // Combine relative linearized-depth and absolute raw-depth gates.
                    let stored_linear = linearized_reverse_depth(stored);
                    let expected_linear = linearized_reverse_depth(prev_ndc.z);
                    reprojectable = (stored - prev_ndc.z).abs() <= LOCAL_SHADOW_REPROJECT_DEPTH_ABS
                        || (stored_linear - expected_linear).abs()
                            <= LOCAL_SHADOW_RESOLVE_DEPTH_REL * expected_linear;
                }
            }
        }
    }

    // Evaluate edge promotion around the reprojected receiver.
    let history_texel_index = if reprojectable {
        prev_texel_index
    } else {
        texel_index
    };

    // Insert top-K scores descending; strict comparison makes ties stable.
    let mut slot_lights = [LOCAL_SHADOW_SLOT_EMPTY; LOCAL_SHADOW_SLOTS as usize];
    let mut slot_scores = [0.0f32; LOCAL_SHADOW_SLOTS as usize];
    let mut i = 0u32;
    while i < data.light_count {
        let light = data.lights[i];
        let score = contribution_score(point_light_contribution(
            normal,
            position_world,
            &light,
            data.wrap_w,
        ));
        if score > slot_scores[LOCAL_SHADOW_SLOTS as usize - 1] {
            let mut j = LOCAL_SHADOW_SLOTS as usize - 1;
            while j > 0 && score > slot_scores[j - 1] {
                slot_scores[j] = slot_scores[j - 1];
                slot_lights[j] = slot_lights[j - 1];
                j -= 1;
            }
            slot_scores[j] = score;
            slot_lights[j] = i;
        }
        i += 1;
    }

    let mut selected_word = LOCAL_SHADOW_SLOT_WORD_EMPTY;
    {
        let mut slot = 0u32;
        while slot < LOCAL_SHADOW_SLOTS {
            selected_word = local_shadow_slot_set(selected_word, slot, slot_lights[slot as usize]);
            slot += 1;
        }
    }
    slot_map[texel_index] = selected_word;

    // Count selection changes after translating identities through the remap.
    if reuse_enabled && reprojectable {
        let prev_word = data.slot_map_prev[prev_texel_index];
        // Equal cardinality plus mutual membership proves set equality.
        let mut same = true;
        let mut filled = 0u32;
        let mut prev_filled = 0u32;
        let mut slot = 0u32;
        while slot < LOCAL_SHADOW_SLOTS {
            let cur = slot_lights[slot as usize];
            if cur != LOCAL_SHADOW_SLOT_EMPTY {
                filled += 1;
                let translated = data.light_remap[cur];
                if translated == LOCAL_SHADOW_SLOT_EMPTY
                    || local_shadow_slot_find(prev_word, translated) == LOCAL_SHADOW_SLOTS
                {
                    same = false;
                }
            }
            if local_shadow_slot_get(prev_word, slot) != LOCAL_SHADOW_SLOT_EMPTY {
                prev_filled += 1;
            }
            slot += 1;
        }
        if !same || filled != prev_filled {
            bump(data, COUNTER_CHURN);
        }
    }
    let mut promoted = false;
    if reuse_enabled && data.edge_promotion != 0 {
        let prev_light = local_shadow_slot_get(data.slot_map_prev[history_texel_index], 0);
        if prev_light != LOCAL_SHADOW_SLOT_EMPTY {
            let history_texel = UVec2::new(
                history_texel_index % data.half_size[0],
                history_texel_index / data.half_size[0],
            );
            promoted = edge_promoted(data, history_texel, prev_light);
        }
    }

    let base = texel_index * LOCAL_SHADOW_SLOTS;
    let mut slot = 0u32;
    while slot < LOCAL_SHADOW_SLOTS {
        let light_index = slot_lights[slot as usize];
        let id = base + slot;
        if light_index == LOCAL_SHADOW_SLOT_EMPTY {
            slot_state[id] = SHADOW_STATE_INACTIVE;
            // Define empty-slot fractions to prevent stale output.
            slot_fraction[id] = LOCAL_SHADOW_FRACTION_ONE;
        } else {
            let light = data.lights[light_index];
            if promoted {
                // Reprove promoted edges; carry prior answers if budget-starved.
                bump(data, COUNTER_PROMOTED);
                let prev_id = data.light_remap[light_index];
                let prev_word = data.slot_map_prev[history_texel_index];
                let prev_slot = if prev_id == LOCAL_SHADOW_SLOT_EMPTY {
                    LOCAL_SHADOW_SLOTS
                } else {
                    local_shadow_slot_find(prev_word, prev_id)
                };
                let mut estimate = local_shadow_pack(SHADOW_STATE_VISIBLE, LOCAL_SHADOW_AGE_MAX, 0);
                if prev_slot < LOCAL_SHADOW_SLOTS {
                    let w =
                        data.slot_state_prev[history_texel_index * LOCAL_SHADOW_SLOTS + prev_slot];
                    let st = local_shadow_state(w);
                    if st == SHADOW_STATE_VISIBLE || st == SHADOW_STATE_OCCLUDED {
                        estimate = w;
                    }
                }
                slot_state[id] = estimate;
                slot_fraction[id] = if local_shadow_state(estimate) == SHADOW_STATE_OCCLUDED {
                    0
                } else {
                    LOCAL_SHADOW_FRACTION_ONE
                };
                if !enqueue_request(data, id, true) {
                    slot_state[id] = SHADOW_STATE_FAILED;
                    slot_fraction[id] = LOCAL_SHADOW_FRACTION_ONE;
                }
                slot += 1;
                continue;
            }
            let (mut answered, carried) = if reuse_enabled {
                history_answer(
                    data,
                    depth_tex,
                    light_index,
                    &light,
                    position_world,
                    reprojectable,
                    prev_texel_index,
                )
            } else {
                (0, 0)
            };
            if answered == 0 {
                answered = contact_answer(data, depth_tex, &light, position_world);
            }
            if answered != 0 {
                slot_state[id] = answered;
                slot_fraction[id] = if local_shadow_state(answered) == SHADOW_STATE_OCCLUDED {
                    0
                } else {
                    LOCAL_SHADOW_FRACTION_ONE
                };
            } else if carried != 0 {
                // Retain the estimate until low-priority refresh completes.
                slot_state[id] = carried;
                slot_fraction[id] = if local_shadow_state(carried) == SHADOW_STATE_OCCLUDED {
                    0
                } else {
                    LOCAL_SHADOW_FRACTION_ONE
                };
                if !enqueue_request(data, id, false) {
                    slot_state[id] = SHADOW_STATE_FAILED;
                    slot_fraction[id] = LOCAL_SHADOW_FRACTION_ONE;
                }
            } else if enqueue_request(data, id, true) {
                // Unproven visibility fails open until high-priority tracing completes.
                slot_state[id] = local_shadow_pack(SHADOW_STATE_VISIBLE, LOCAL_SHADOW_AGE_MAX, 0);
                slot_fraction[id] = LOCAL_SHADOW_FRACTION_ONE;
            } else {
                slot_state[id] = SHADOW_STATE_FAILED;
                slot_fraction[id] = LOCAL_SHADOW_FRACTION_ONE;
            }
        }
        slot += 1;
    }
    if slot_lights[0] != LOCAL_SHADOW_SLOT_EMPTY {
        bump(data, COUNTER_ACTIVE);
    }
}

/// Builds the indirect trace dispatch, prioritizing high requests within budget.
/// A zero budget means unlimited tracing.
#[spirv(compute(threads(1)))]
pub fn local_shadow_args(#[spirv(push_constant)] data_ptr: &GpuPtr<LocalShadowArgsData>) {
    let data = &**data_ptr;
    let high = data.counters.requests_high.min(data.queue_capacity);
    let low = data.counters.requests_low.min(data.queue_capacity);
    let budget = if data.ray_budget == 0 {
        u32::MAX
    } else {
        data.ray_budget
    };
    let serviced_high = high.min(budget);
    let serviced_low = low.min(budget - serviced_high);
    let mut counters = data.counters.cast::<u32>();
    counters[COUNTER_SERVICED_HIGH as u32] = serviced_high;
    counters[COUNTER_SERVICED_LOW as u32] = serviced_low;
    let total = serviced_high + serviced_low;
    let mut args = data.dispatch_args;
    args[0u32] = total.div_ceil(data.group_size);
    args[1u32] = 1;
    args[2u32] = 1;
}

/// Answers compacted requests with authoritative world any-hit queries.
/// Valid occlusions store a quantized hit for later validation; query failure
/// becomes FAILED and non-occluded results are fully lit. An out-of-range
/// request ID only increments OVERFLOW and exits; an empty slot, invalid light,
/// missing representative, or nonpositive depth writes FAILED (and never a
/// visibility answer).
#[spirv(compute(threads(64)))]
pub fn local_shadow_trace(
    #[spirv(push_constant)] data_ptr: &GpuPtr<LocalShadowData>,
    #[spirv(descriptor_set = 0, binding = 0)] textures: &RuntimeArray<Image2d>,
    #[spirv(global_invocation_id)] gid: UVec3,
) {
    let data = &**data_ptr;
    let serviced_high = data.counters.serviced_high;
    let count = serviced_high + data.counters.serviced_low;
    if gid.x >= count {
        return;
    }
    let (raw_id, born) = if gid.x < serviced_high {
        (data.requests_high[gid.x], true)
    } else {
        (data.requests_low[gid.x - serviced_high], false)
    };
    let id = raw_id & !REQUEST_PROMOTED;
    let texel_count = data.half_size[0] * data.half_size[1];
    if id >= texel_count * LOCAL_SHADOW_SLOTS {
        bump(data, COUNTER_OVERFLOW);
        return;
    }
    let texel_index = id / LOCAL_SHADOW_SLOTS;
    let slot = id - texel_index * LOCAL_SHADOW_SLOTS;
    let mut slot_state = data.slot_state;

    let word = data.slot_map[texel_index];
    let light_index = local_shadow_slot_get(word, slot);
    let rep = data.slot_rep[texel_index];
    if light_index == LOCAL_SHADOW_SLOT_EMPTY
        || light_index >= data.light_count
        || rep == LOCAL_SHADOW_REP_NONE
    {
        slot_state[id] = SHADOW_STATE_FAILED;
        return;
    }
    let coord = UVec2::new(rep & 0xFFFF, rep >> 16);
    let depth_tex = unsafe { textures.index(data.depth_texture_id as usize) };
    let depth: Vec4 = depth_tex.fetch_with_lod(coord, 0);
    if depth.x <= 0.0 {
        slot_state[id] = SHADOW_STATE_FAILED;
        return;
    }
    let position_world = receiver_world(data, coord, depth.x);
    let light = data.lights[light_index];
    // Trace center rays; blur reconstructs penumbrae from this frame's field.
    let segment = ShadowSegment::between(
        position_world,
        Vec3::from_array(light.position),
        data.origin_bias,
        data.destination_bias,
    );
    // A degenerate segment lies inside the emitter bias shell.
    let packed = if !segment.is_active() {
        local_shadow_pack(SHADOW_STATE_VISIBLE, 0, 0)
    } else {
        let result = world_any_hit(&data.world, &segment);
        if result.status == SHADOW_QUERY_OCCLUDED {
            let hit_q = (result.hit_t.clamp(0.0, 1.0) * 65535.0) as u32;
            // Classify unobservable hits as blind; dynamic occluders shorten reuse.
            let state = if validate_occlusion(data, depth_tex, &segment, result.hit_t) {
                SHADOW_STATE_OCCLUDED
            } else {
                let dynamic = result.instance_id != u32::MAX
                    && data.world.instances[result.instance_id].flags
                        & SHADOW_TLAS_INSTANCE_DYNAMIC
                        != 0;
                if dynamic {
                    SHADOW_STATE_OCCLUDED | LOCAL_SHADOW_BLIND | LOCAL_SHADOW_DYNAMIC
                } else {
                    SHADOW_STATE_OCCLUDED | LOCAL_SHADOW_BLIND
                }
            };
            // Stagger initial refresh phases to avoid synchronized expiry.
            let age = if born {
                local_shadow_fresh_age(id, 0x0C91, data.occluded_refresh)
            } else {
                0
            };
            local_shadow_pack(state, age, hit_q)
        } else if result.status == SHADOW_QUERY_FAILED {
            local_shadow_pack(SHADOW_STATE_FAILED, 0, 0)
        } else {
            let age = if born {
                local_shadow_fresh_age(id, 0x0A53, data.refresh_interval)
            } else {
                0
            };
            local_shadow_pack(SHADOW_STATE_VISIBLE, age, 0)
        }
    };
    slot_state[id] = packed;
    let mut slot_fraction = data.slot_fraction;
    slot_fraction[id] = if local_shadow_state(packed) == SHADOW_STATE_OCCLUDED {
        0
    } else {
        LOCAL_SHADOW_FRACTION_ONE
    };
}

/// Reconstructs penumbrae from this frame's deterministic binary visibility
/// field with depth-guided tent filtering. Radius is
/// `source_radius * t / (1 - t)`, where `t` is the stored blocker parameter;
/// no temporal accumulation can lag, reset, or ripple.
#[spirv(compute(threads(8, 8)))]
pub fn local_shadow_blur(
    #[spirv(push_constant)] data_ptr: &GpuPtr<LocalShadowData>,
    #[spirv(global_invocation_id)] gid: UVec3,
) {
    let data = &**data_ptr;
    if gid.x >= data.half_size[0] || gid.y >= data.half_size[1] {
        return;
    }
    // Keep malformed zero-radius dispatches inert.
    if data.source_radius <= 0.0 {
        return;
    }
    let texel = gid.truncate();
    let texel_index = texel.y * data.half_size[0] + texel.x;
    let own_depth = data.rep_depth[texel_index];
    if own_depth <= 0.0 {
        return;
    }
    let own_linear = linearized_reverse_depth(own_depth);
    let hi_x = data.half_size[0] as i32 - 1;
    let hi_y = data.half_size[1] as i32 - 1;
    let mut slot_fraction = data.slot_fraction;

    let mut slot = 0u32;
    while slot < LOCAL_SHADOW_SLOTS {
        let word = data.slot_map[texel_index];
        let light_index = local_shadow_slot_get(word, slot);
        if light_index == LOCAL_SHADOW_SLOT_EMPTY {
            slot += 1;
            continue;
        }
        let id = texel_index * LOCAL_SHADOW_SLOTS + slot;
        let own_state = local_shadow_state(data.slot_state[id]);
        if own_state != SHADOW_STATE_VISIBLE && own_state != SHADOW_STATE_OCCLUDED {
            slot += 1;
            continue;
        }

        // Probe a sparse radius-three ring before the full mixed-neighborhood sweep.
        const SCAN: i32 = 3;
        let mut seen_visible = false;
        let mut seen_occluded = false;
        let mut probe = 0u32;
        while probe < 17 {
            let (dx, dy) = if probe < 9 {
                ((probe % 3) as i32 - 1, (probe / 3) as i32 - 1)
            } else {
                // Radius-three corners and edge midpoints.
                match probe - 9 {
                    0 => (-SCAN, -SCAN),
                    1 => (0, -SCAN),
                    2 => (SCAN, -SCAN),
                    3 => (-SCAN, 0),
                    4 => (SCAN, 0),
                    5 => (-SCAN, SCAN),
                    6 => (0, SCAN),
                    _ => (SCAN, SCAN),
                }
            };
            let nx = (texel.x as i32 + dx).clamp(0, hi_x) as u32;
            let ny = (texel.y as i32 + dy).clamp(0, hi_y) as u32;
            let t = ny * data.half_size[0] + nx;
            let neighbor_word = data.slot_map[t];
            let neighbor_slot = local_shadow_slot_find(neighbor_word, light_index);
            if neighbor_slot == LOCAL_SHADOW_SLOTS {
                probe += 1;
                continue;
            }
            let state = local_shadow_state(data.slot_state[t * LOCAL_SHADOW_SLOTS + neighbor_slot]);
            if state == SHADOW_STATE_VISIBLE {
                seen_visible = true;
            } else if state == SHADOW_STATE_OCCLUDED {
                seen_occluded = true;
            }
            probe += 1;
        }
        if !(seen_visible && seen_occluded) {
            slot += 1;
            continue;
        }
        // Mixed visibility warrants the full nearest-blocker sweep.
        let mut min_hit_q = 0xFFFFu32;
        let mut dy = -SCAN;
        while dy <= SCAN {
            let mut dx = -SCAN;
            while dx <= SCAN {
                let nx = (texel.x as i32 + dx).clamp(0, hi_x) as u32;
                let ny = (texel.y as i32 + dy).clamp(0, hi_y) as u32;
                let t = ny * data.half_size[0] + nx;
                let neighbor_word = data.slot_map[t];
                let neighbor_slot = local_shadow_slot_find(neighbor_word, light_index);
                if neighbor_slot == LOCAL_SHADOW_SLOTS {
                    dx += 1;
                    continue;
                }
                let state_word = data.slot_state[t * LOCAL_SHADOW_SLOTS + neighbor_slot];
                if local_shadow_state(state_word) == SHADOW_STATE_OCCLUDED {
                    let hit_q = local_shadow_hit_q(state_word);
                    if hit_q < min_hit_q {
                        min_hit_q = hit_q;
                    }
                }
                dx += 1;
            }
            dy += 1;
        }

        // Convert penumbra width from world units to half-resolution texels.
        let t_blocker = (min_hit_q as f32 / 65535.0).min(0.98);
        let penumbra_world = data.source_radius * t_blocker / (1.0 - t_blocker);
        let eye = data.near_plane * linearized_reverse_depth(own_depth);
        let texel_world = 4.0 * eye / (data.proj_scale * data.screen_size[1] as f32);
        let radius_f = penumbra_world / texel_world.max(1.0e-6);
        let radius = (radius_f + 0.5).clamp(1.0, SCAN as f32) as i32;

        // Gather the binary field with depth-guided tent weights.
        let mut sum = 0.0f32;
        let mut weight_sum = 0.0f32;
        let mut dy = -radius;
        while dy <= radius {
            let mut dx = -radius;
            while dx <= radius {
                let nx = (texel.x as i32 + dx).clamp(0, hi_x) as u32;
                let ny = (texel.y as i32 + dy).clamp(0, hi_y) as u32;
                let t = ny * data.half_size[0] + nx;
                let neighbor_word = data.slot_map[t];
                let neighbor_slot = local_shadow_slot_find(neighbor_word, light_index);
                if neighbor_slot == LOCAL_SHADOW_SLOTS {
                    dx += 1;
                    continue;
                }
                let state_word = data.slot_state[t * LOCAL_SHADOW_SLOTS + neighbor_slot];
                let state = local_shadow_state(state_word);
                if state != SHADOW_STATE_VISIBLE && state != SHADOW_STATE_OCCLUDED {
                    dx += 1;
                    continue;
                }
                let neighbor_depth = data.rep_depth[t];
                if neighbor_depth <= 0.0 {
                    dx += 1;
                    continue;
                }
                let neighbor_linear = linearized_reverse_depth(neighbor_depth);
                if (neighbor_linear - own_linear).abs()
                    > LOCAL_SHADOW_RESOLVE_DEPTH_REL * own_linear
                {
                    dx += 1;
                    continue;
                }
                let chebyshev = dx.abs().max(dy.abs()) as f32;
                let weight = 1.0 - chebyshev / (radius as f32 + 1.0);
                if state == SHADOW_STATE_VISIBLE {
                    sum += weight;
                }
                weight_sum += weight;
                dx += 1;
            }
            dy += 1;
        }
        if weight_sum > 0.0 {
            slot_fraction[id] = (sum / weight_sum * LOCAL_SHADOW_FRACTION_ONE as f32) as u32;
        }
        slot += 1;
    }
}

/// Selects top-K lights without history, then queues every filled slot.
/// It defines all empty-slot outputs before the dense trace runs.
#[spirv(compute(threads(8, 8)))]
pub fn local_shadow_select_direct(
    #[spirv(push_constant)] data_ptr: &GpuPtr<LocalShadowData>,
    #[spirv(descriptor_set = 0, binding = 0)] textures: &RuntimeArray<Image2d>,
    #[spirv(global_invocation_id)] gid: UVec3,
) {
    let data = &**data_ptr;
    if gid.x >= data.half_size[0] || gid.y >= data.half_size[1] {
        return;
    }
    let texel = gid.truncate();
    let texel_index = texel.y * data.half_size[0] + texel.x;
    let mut slot_map = data.slot_map;
    let mut slot_state = data.slot_state;
    let mut slot_rep = data.slot_rep;
    let mut rep_depth_out = data.rep_depth;
    let mut slot_fraction = data.slot_fraction;

    // Choose the first covered corner in deterministic order.
    let marker_tex = unsafe { textures.index(data.surface_material_texture_id as usize) };
    let depth_tex = unsafe { textures.index(data.depth_texture_id as usize) };
    let mut rep = LOCAL_SHADOW_REP_NONE;
    let mut rep_depth = 0.0f32;
    let mut corner = 0u32;
    while corner < 4 {
        let c = texel * 2 + UVec2::new(corner & 1, corner >> 1);
        if c.x < data.screen_size[0] && c.y < data.screen_size[1] {
            let marker: Vec4 = marker_tex.fetch_with_lod(c, 0);
            let depth: Vec4 = depth_tex.fetch_with_lod(c, 0);
            if marker.x > 0.0 && depth.x > 0.0 {
                rep = c.x | (c.y << 16);
                rep_depth = depth.x;
                corner = 4;
                continue;
            }
        }
        corner += 1;
    }
    slot_rep[texel_index] = rep;
    rep_depth_out[texel_index] = rep_depth;

    if rep == LOCAL_SHADOW_REP_NONE {
        slot_map[texel_index] = LOCAL_SHADOW_SLOT_WORD_EMPTY;
        let mut slot = 0u32;
        while slot < LOCAL_SHADOW_SLOTS {
            slot_state[texel_index * LOCAL_SHADOW_SLOTS + slot] = SHADOW_STATE_INACTIVE;
            slot_fraction[texel_index * LOCAL_SHADOW_SLOTS + slot] = LOCAL_SHADOW_FRACTION_ONE;
            slot += 1;
        }
        return;
    }

    let coord = UVec2::new(rep & 0xFFFF, rep >> 16);
    let position_world = receiver_world(data, coord, rep_depth);
    let oct: Vec4 =
        unsafe { textures.index(data.surface_normal_texture_id as usize) }.fetch_with_lod(coord, 0);
    let normal = oct_decode(Vec2::new(oct.x, oct.y));

    // Rank by wrapped-N·L contribution; zero scores receive no slot or ray.
    let mut slot_lights = [LOCAL_SHADOW_SLOT_EMPTY; LOCAL_SHADOW_SLOTS as usize];
    let mut slot_scores = [0.0f32; LOCAL_SHADOW_SLOTS as usize];
    let mut i = 0u32;
    while i < data.light_count {
        let light = data.lights[i];
        let score = contribution_score(point_light_contribution(
            normal,
            position_world,
            &light,
            data.wrap_w,
        ));
        if score > slot_scores[LOCAL_SHADOW_SLOTS as usize - 1] {
            let mut j = LOCAL_SHADOW_SLOTS as usize - 1;
            while j > 0 && score > slot_scores[j - 1] {
                slot_scores[j] = slot_scores[j - 1];
                slot_lights[j] = slot_lights[j - 1];
                j -= 1;
            }
            slot_scores[j] = score;
            slot_lights[j] = i;
        }
        i += 1;
    }

    let mut selected_word = LOCAL_SHADOW_SLOT_WORD_EMPTY;
    let mut slot = 0u32;
    while slot < LOCAL_SHADOW_SLOTS {
        selected_word = local_shadow_slot_set(selected_word, slot, slot_lights[slot as usize]);
        let id = texel_index * LOCAL_SHADOW_SLOTS + slot;
        if slot_lights[slot as usize] == LOCAL_SHADOW_SLOT_EMPTY {
            slot_state[id] = SHADOW_STATE_INACTIVE;
            slot_fraction[id] = LOCAL_SHADOW_FRACTION_ONE;
        } else if !enqueue_request(data, id, true) {
            // Capacity violations fail closed rather than imply visibility.
            slot_state[id] = SHADOW_STATE_FAILED;
            slot_fraction[id] = LOCAL_SHADOW_FRACTION_ONE;
        }
        slot += 1;
    }
    slot_map[texel_index] = selected_word;
    // Atomic append order affects trace locality, not output determinism.
}

/// Traces compacted stateless requests with one thread per `(texel, slot)`.
/// Outputs are per-id and order-independent; occlusions retain hit parameters.
/// Empty queued slots, invalid light IDs, missing representatives, and
/// nonpositive representative depths write FAILED plus a fully lit fraction.
#[spirv(compute(threads(64)))]
pub fn local_shadow_trace_direct(
    #[spirv(push_constant)] data_ptr: &GpuPtr<LocalShadowData>,
    #[spirv(global_invocation_id)] gid: UVec3,
) {
    let data = &**data_ptr;
    let count = data.counters.serviced_high + data.counters.serviced_low;
    if gid.x >= count {
        return;
    }
    let id = data.requests_high[gid.x];
    let texel_count = data.half_size[0] * data.half_size[1];
    if id >= texel_count * LOCAL_SHADOW_SLOTS {
        bump(data, COUNTER_OVERFLOW);
        return;
    }
    let texel_index = id / LOCAL_SHADOW_SLOTS;
    let slot = id - texel_index * LOCAL_SHADOW_SLOTS;
    let light_index = local_shadow_slot_get(data.slot_map[texel_index], slot);
    let mut slot_state = data.slot_state;
    let mut slot_fraction = data.slot_fraction;
    if light_index == LOCAL_SHADOW_SLOT_EMPTY {
        // Empty queued slots indicate corruption.
        slot_state[id] = SHADOW_STATE_FAILED;
        slot_fraction[id] = LOCAL_SHADOW_FRACTION_ONE;
        return;
    }
    let rep = data.slot_rep[texel_index];
    let rep_depth = data.rep_depth[texel_index];
    if light_index >= data.light_count || rep == LOCAL_SHADOW_REP_NONE || rep_depth <= 0.0 {
        // Corrupt requests fail closed: FAILED state plus full-light fraction,
        // never a fabricated visibility answer.
        slot_state[id] = SHADOW_STATE_FAILED;
        slot_fraction[id] = LOCAL_SHADOW_FRACTION_ONE;
        return;
    }
    let coord = UVec2::new(rep & 0xFFFF, rep >> 16);
    let position_world = receiver_world(data, coord, rep_depth);
    let light = data.lights[light_index];
    // Trace the center ray; blur reconstructs penumbrae spatially.
    let segment = ShadowSegment::between(
        position_world,
        Vec3::from_array(light.position),
        data.origin_bias,
        data.destination_bias,
    );
    let packed = if !segment.is_active() {
        // Inside the emitter bias shell, no occlusion is possible.
        local_shadow_pack(SHADOW_STATE_VISIBLE, 0, 0)
    } else {
        let result = world_any_hit(&data.world, &segment);
        if result.status == SHADOW_QUERY_OCCLUDED {
            let hit_q = (result.hit_t.clamp(0.0, 1.0) * 65535.0) as u32;
            local_shadow_pack(SHADOW_STATE_OCCLUDED, 0, hit_q)
        } else if result.status == SHADOW_QUERY_FAILED {
            local_shadow_pack(SHADOW_STATE_FAILED, 0, 0)
        } else {
            local_shadow_pack(SHADOW_STATE_VISIBLE, 0, 0)
        }
    };
    slot_state[id] = packed;
    slot_fraction[id] = if local_shadow_state(packed) == SHADOW_STATE_OCCLUDED {
        0
    } else {
        LOCAL_SHADOW_FRACTION_ONE
    };
}
