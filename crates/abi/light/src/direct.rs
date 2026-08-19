//! Local-light, screen-space depth-query, and Forward+ tiled-lighting
//! vocabulary shared by host and shader. Types arrive here only when a live
//! pass needs them.
//!
//! Depth convention (engine-wide): INFINITE reverse-Z. 1.0 is the near
//! plane, depth falls toward 0.0 at infinite distance; 0.0 doubles as the
//! "nothing here" sentinel. "Nearer" is always `max()`.

use crate::PointLight;
use crate::{GpuPtr, gpu_data};
use abi_mesh::MaterialEntry;
use glam::Mat4;

/// Screen tile edge in pixels. One depth-reduce workgroup is one tile
/// (`TILE_SIZE × TILE_SIZE` threads), so the shader's `threads(32, 32)`
/// attribute must match — rust-gpu wants literals there; the const assert
/// beside the entry point keeps them honest.
pub const TILE_SIZE: u32 = 32;

/// Per-tile depth bounds for 2.5D light culling. Reverse-Z, so `min_depth`
/// is the FARTHEST sample in the tile and `max_depth` the NEAREST.
#[gpu_data]
pub struct TileDepthBounds {
    pub min_depth: f32,
    pub max_depth: f32,
}

/// Compact per-tile light-list header: `light_count` is written by the
/// cull-count pass, `light_offset` by the prefix-sum scan (start index into
/// the global light-index buffer).
#[gpu_data]
pub struct TileHeader {
    pub light_count: u32,
    pub light_offset: u32,
}

/// Upper bound for one prefix-sum workgroup: 8192 `u32` entries fit its
/// 32-KiB shared-memory budget and cover 4K output at 32-pixel tiles
/// (120 × 68 = 8160). The power-of-two scan always processes the tail, which
/// is zero-filled past `tile_count`.
pub const MAX_TILES: u32 = 8192;

/// Dispatch data for `prefix_sum`: ONE workgroup, exclusive-scans every
/// `TileHeader.light_count` into `.light_offset` and writes the grand total.
#[gpu_data]
pub struct PrefixSumData {
    pub tile_headers: GpuPtr<TileHeader>,
    pub total_light_count: GpuPtr<u32>,
    pub tile_count: u32,
}

/// Dispatch data for `depth_reduce`: one workgroup per tile, parallel
/// min/max over the tile's depth samples into `tile_depth_bounds`.
///
/// The pass fetches depth texels directly.
#[gpu_data]
pub struct DepthReduceData {
    pub tile_depth_bounds: GpuPtr<TileDepthBounds>,
    /// Bindless index of the scene depth texture (sampled heap, set 0).
    pub depth_texture_id: u32,
    pub screen_size: [u32; 2],
    pub tile_count: [u32; 2],
}

/// Hard ceiling for the linear depth marcher. Keeping the loop bounded is
/// part of the shader contract: callers tune among small fixed budgets, never
/// turn a malformed dispatch into an unbounded texture walk.
pub const DEPTH_MARCH_MAX_STEPS: u32 = 64;

/// One projected segment for screen-space visibility. Coordinates are Vulkan
/// NDC after perspective divide: x/y in `[-1, 1]`, infinite reverse-Z depth in
/// `(0, 1]`. The caller clips or rejects off-screen segments before dispatch.
#[gpu_data]
pub struct DepthMarchQuery {
    pub start_ndc: [f32; 3],
    pub _pad0: f32,
    pub end_ndc: [f32; 3],
    pub _pad1: f32,
}

const _: () = assert!(core::mem::size_of::<DepthMarchQuery>() == 32);
const _: () = assert!(core::mem::align_of::<DepthMarchQuery>() == 4);

/// Shared controls for a batch of screen-space shadow segments.
///
/// `depth_thickness` and `near_plane` use world/view units; the shader scales
/// thickness by `1 / near_plane` before comparing reciprocal reverse-Z depth.
/// With `continue_after_deep_penetration == 0`, the first sign crossing owns
/// the result and a crossing beyond `depth_thickness` is an uncertain miss.
/// With `1`, deep crossings are skipped and the linear search continues.
#[gpu_data]
pub struct DepthMarchConfig {
    pub linear_steps: u32,
    pub continue_after_deep_penetration: u32,
    pub jitter: f32,
    pub depth_thickness: f32,
    pub near_plane: f32,
    pub _pad: [u32; 3],
}

/// Validate before recording. The shader assumes this contract rather than
/// paying defensive branches per ray.
pub fn depth_march_config_valid(config: &DepthMarchConfig) -> bool {
    config.linear_steps >= 2
        && config.linear_steps <= DEPTH_MARCH_MAX_STEPS
        && config.continue_after_deep_penetration <= 1
        && config.jitter >= 0.0
        && config.jitter <= 1.0
        && config.depth_thickness.is_finite()
        && config.depth_thickness > 0.0
        && config.near_plane.is_finite()
        && config.near_plane > 0.0
}

const _: () = assert!(core::mem::size_of::<DepthMarchConfig>() == 32);
const _: () = assert!(core::mem::align_of::<DepthMarchConfig>() == 4);

/// Result of one conservative screen-space occlusion query. `hit == 0`
/// deliberately means "send to fallback", not "visible".
#[gpu_data]
pub struct DepthMarchResult {
    pub hit: u32,
    pub _pad0: u32,
    pub hit_t: f32,
    pub hit_penetration: f32,
    pub hit_uv: [f32; 2],
    pub _pad1: [u32; 2],
}

const _: () = assert!(core::mem::size_of::<DepthMarchResult>() == 32);
const _: () = assert!(core::mem::align_of::<DepthMarchResult>() == 4);

/// Standalone batch data for screen-space visibility queries.
#[gpu_data]
pub struct DepthMarchData {
    pub queries: GpuPtr<DepthMarchQuery>,
    pub results: GpuPtr<DepthMarchResult>,
    pub depth_texture_id: u32,
    pub query_count: u32,
    pub depth_size: [u32; 2],
    pub config: DepthMarchConfig,
}

const _: () = assert!(core::mem::size_of::<DepthMarchData>() == 64);
const _: () = assert!(core::mem::align_of::<DepthMarchData>() == 4);

/// Largest material index whose `index + 1` marker is exact in R32F.
/// `0` is the invalid/background sentinel; f32 represents every integer
/// through 2²⁴, so the maximum index is 2²⁴ − 1.
pub const SURFACE_MATERIAL_INDEX_MAX: u32 = (1 << 24) - 1;

/// Selected local-shadow lights per half-resolution texel.
///
/// Each slot applies to every full-resolution pixel covered by its texel.
pub const LOCAL_SHADOW_SLOTS: u32 = 4;

/// Logical empty-slot sentinel. The packed slot word stores four 8-bit
/// lanes (`0xFF` = empty lane), so light ids are capped at 254; the
/// accessors below translate lanes to this logical sentinel so comparison
/// sites never see lane width.
pub const LOCAL_SHADOW_SLOT_EMPTY: u32 = 0xFFFF;

/// Largest usable light id under the 8-bit slot lanes.
pub const LOCAL_SHADOW_MAX_LIGHT_ID: u32 = 0xFE;

/// The all-empty packed slot word.
pub const LOCAL_SHADOW_SLOT_WORD_EMPTY: u32 = 0xFFFF_FFFF;

/// Read one slot lane, translating the empty lane to the logical sentinel.
#[inline(always)]
pub fn local_shadow_slot_get(word: u32, slot: u32) -> u32 {
    let id = (word >> (slot * 8)) & 0xFF;
    if id == 0xFF {
        LOCAL_SHADOW_SLOT_EMPTY
    } else {
        id
    }
}

/// Write one slot lane (`LOCAL_SHADOW_SLOT_EMPTY` stores the empty lane).
#[inline(always)]
pub fn local_shadow_slot_set(word: u32, slot: u32, id: u32) -> u32 {
    let lane = if id == LOCAL_SHADOW_SLOT_EMPTY {
        0xFF
    } else {
        id & 0xFF
    };
    (word & !(0xFF << (slot * 8))) | (lane << (slot * 8))
}

/// Find `light`'s slot in a packed word; `LOCAL_SHADOW_SLOTS` when absent.
/// The sentinel never matches — searching for EMPTY finds nothing.
#[inline(always)]
pub fn local_shadow_slot_find(word: u32, light: u32) -> u32 {
    let mut slot = 0u32;
    while slot < LOCAL_SHADOW_SLOTS {
        let id = (word >> (slot * 8)) & 0xFF;
        if id != 0xFF && id == light {
            return slot;
        }
        slot += 1;
    }
    LOCAL_SHADOW_SLOTS
}

/// Sentinel in the packed representative-receiver word: no covered pixel of
/// this half-res texel holds a lit surface.
pub const LOCAL_SHADOW_REP_NONE: u32 = 0xFFFF_FFFF;

/// Absolute RAW reverse-Z tolerance for the reprojection surface check.
/// Far surfaces have tiny reverse-Z values where the reciprocal
/// linearization amplifies f32 matrix-roundtrip noise past any relative
/// gate; roundtrip noise is absolute in raw z (~1e-6), while different
/// surfaces differ by orders of magnitude more than this.
pub const LOCAL_SHADOW_REPROJECT_DEPTH_ABS: f32 = 1.0e-4;

/// Relative linearized-depth tolerance for the resolve's guided weights. A
/// half-res sample whose representative receiver differs more than this from
/// the shaded pixel is not the same surface and contributes nothing.
pub const LOCAL_SHADOW_RESOLVE_DEPTH_REL: f32 = 0.1;

/// Mesh-light dispatch data for exact or selected-shadow visibility.
#[gpu_data]
pub struct LocalLightData {
    pub clip_to_world: Mat4,
    pub materials: GpuPtr<MaterialEntry>,
    pub lights: GpuPtr<PointLight>,
    /// Null selects neutral visibility through `light_field_sample`.
    pub light_field: GpuPtr<f32>,
    /// Null selects the unshadowed path.
    pub shadow_states: GpuPtr<u32>,
    /// Null disables slot selection. Mutually exclusive with `shadow_states`.
    pub slot_map: GpuPtr<u32>,
    /// One packed state word (`local_shadow_state` decodes the low byte)
    /// per `(half-res texel, slot)`.
    pub slot_state: GpuPtr<u32>,
    /// One packed representative full-res coord (x | y << 16) per texel.
    pub slot_rep: GpuPtr<u32>,
    /// 8-bit visibility fraction per `(texel, slot)` (see LocalShadowData).
    pub slot_fraction: GpuPtr<u32>,
    pub depth_texture_id: u32,
    pub surface_normal_texture_id: u32,
    pub surface_albedo_texture_id: u32,
    pub surface_material_texture_id: u32,
    pub hdr_texture_id: u32,
    pub ramp_default_sampler: u32,
    pub screen_size: [u32; 2],
    pub light_count: u32,
    pub wrap_w: f32,
    pub light_field_dims: [u32; 2],
    pub light_field_cell_size: f32,
    pub light_field_gate: f32,
    pub half_size: [u32; 2],
    /// Nonzero visualizes the visibility answer source.
    pub debug_overlay: u32,
    pub _pad0: [u32; 3],
}

const _: () = assert!(core::mem::size_of::<LocalLightData>() == 208);
const _: () = assert!(core::mem::align_of::<LocalLightData>() == 16);
const _: () = assert!(core::mem::offset_of!(LocalLightData, materials) == 64);
const _: () = assert!(core::mem::offset_of!(LocalLightData, shadow_states) == 88);
const _: () = assert!(core::mem::offset_of!(LocalLightData, slot_map) == 96);
const _: () = assert!(core::mem::offset_of!(LocalLightData, slot_fraction) == 120);
const _: () = assert!(core::mem::offset_of!(LocalLightData, depth_texture_id) == 128);
const _: () = assert!(core::mem::offset_of!(LocalLightData, screen_size) == 152);
const _: () = assert!(core::mem::offset_of!(LocalLightData, light_field_dims) == 168);
const _: () = assert!(core::mem::offset_of!(LocalLightData, half_size) == 184);

/// Dispatch data for exact per-light, per-pixel visibility queries.
#[gpu_data]
pub struct MeshShadowData {
    pub clip_to_world: Mat4,
    pub world_to_clip: Mat4,
    pub world: crate::shadow::ShadowWorld,
    pub lights: GpuPtr<PointLight>,
    pub states: GpuPtr<u32>,
    pub depth_texture_id: u32,
    pub surface_material_texture_id: u32,
    pub screen_size: [u32; 2],
    pub light_count: u32,
    pub origin_bias: f32,
    pub destination_bias: f32,
    pub _pad0: [u32; 3],
}

const _: () = assert!(core::mem::size_of::<MeshShadowData>() == 224);
const _: () = assert!(core::mem::align_of::<MeshShadowData>() == 16);
const _: () = assert!(core::mem::offset_of!(MeshShadowData, world_to_clip) == 64);
const _: () = assert!(core::mem::offset_of!(MeshShadowData, world) == 128);
const _: () = assert!(core::mem::offset_of!(MeshShadowData, lights) == 168);
const _: () = assert!(core::mem::offset_of!(MeshShadowData, depth_texture_id) == 184);

/// Saturating age ceiling in the packed slot-state word.
pub const LOCAL_SHADOW_AGE_MAX: u32 = 0xFF;

/// Marks occlusion whose blocker cannot be validated onscreen.
///
/// Such occlusion uses aged trust and low-priority revalidation.
pub const LOCAL_SHADOW_BLIND: u32 = 0x40;

/// Marks occlusion from a dynamic, screen-blind instance.
pub const LOCAL_SHADOW_DYNAMIC: u32 = 0x20;

/// Maximum carried frames for dynamic blind occlusion.
pub const LOCAL_SHADOW_DYNAMIC_LEASH: u32 = 2;

/// Packed per-(texel, slot) state word: `SHADOW_STATE_*` plus the
/// `LOCAL_SHADOW_BLIND` flag in the low byte, saturating age in the next
/// byte, and the quantized segment hit parameter (`t * 65535`,
/// receiver→light) in the high half. ZII: the zero word is
/// `SHADOW_STATE_INACTIVE`, age 0, t 0.
#[inline(always)]
pub fn local_shadow_pack(state: u32, age: u32, hit_q: u32) -> u32 {
    (state & 0xFF) | ((age & 0xFF) << 8) | ((hit_q & 0xFFFF) << 16)
}

#[inline(always)]
pub fn local_shadow_state(word: u32) -> u32 {
    word & 0xFF & !(LOCAL_SHADOW_BLIND | LOCAL_SHADOW_DYNAMIC)
}

#[inline(always)]
pub fn local_shadow_blind(word: u32) -> bool {
    word & LOCAL_SHADOW_BLIND != 0
}

#[inline(always)]
pub fn local_shadow_dynamic(word: u32) -> bool {
    word & LOCAL_SHADOW_DYNAMIC != 0
}

#[inline(always)]
pub fn local_shadow_age(word: u32) -> u32 {
    (word >> 8) & 0xFF
}

#[inline(always)]
pub fn local_shadow_hit_q(word: u32) -> u32 {
    word >> 16
}

/// Raw full visibility is `255 << 8` in the low 16-bit 8.8 field.
pub const LOCAL_SHADOW_FRACTION_ONE: u32 = 255 << 8;

#[inline(always)]
pub fn local_shadow_fraction(word: u32) -> u32 {
    word & 0xFFFF
}

/// One round of PCG-ish integer mixing; cheap, stateless, shared by the
/// trace kernel (penumbra disk samples, refresh phases) and the CPU replays
/// that prove it.
#[inline(always)]
pub fn local_shadow_hash(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^= x >> 16;
    x
}

/// Staggers refresh phases to distribute revalidation work.
#[inline(always)]
pub fn local_shadow_fresh_age(id: u32, salt: u32, interval: u32) -> u32 {
    if interval > 1 {
        // Keep phases within half the refresh interval.
        (local_shadow_hash(id ^ salt) % (interval / 2 + 1)).min(LOCAL_SHADOW_AGE_MAX)
    } else {
        0
    }
}

/// Half-resolution local-shadow dispatch shared by selection and tracing.
///
/// Previous-frame buffers are read-only; `history_valid == 0` disables reuse.
#[gpu_data]
pub struct LocalShadowData {
    pub clip_to_world: Mat4,
    pub world_to_clip: Mat4,
    pub prev_world_to_clip: Mat4,
    pub world: crate::shadow::ShadowWorld,
    pub lights: GpuPtr<PointLight>,
    /// Last frame's light array (stable copy owned by the pass), for the
    /// teleport-invalidation rule. Indices must be stable across frames.
    pub prev_lights: GpuPtr<PointLight>,
    /// One packed word (2 × u16 light id, `LOCAL_SHADOW_SLOT_EMPTY`) per texel.
    pub slot_map: GpuPtr<u32>,
    pub slot_map_prev: GpuPtr<u32>,
    /// One packed state word (see `local_shadow_pack`) per `(texel, slot)`.
    pub slot_state: GpuPtr<u32>,
    pub slot_state_prev: GpuPtr<u32>,
    /// One fraction word per `(texel, slot)`: 8.8 visibility, hard from the
    /// trace, reconstructed into penumbra near edges by `local_shadow_blur`.
    pub slot_fraction: GpuPtr<u32>,
    /// One packed representative full-res coord (x | y << 16) per texel;
    /// `LOCAL_SHADOW_REP_NONE` when the texel covers no lit surface.
    pub slot_rep: GpuPtr<u32>,
    /// The representative receiver's raw reverse-Z depth per texel, written
    /// this frame and read next frame by the reprojection surface check.
    pub rep_depth: GpuPtr<f32>,
    pub rep_depth_prev: GpuPtr<f32>,
    /// High-priority requests for slots without usable history.
    pub requests_high: GpuPtr<u32>,
    /// Low-priority requests refresh carried visible estimates.
    pub requests_low: GpuPtr<u32>,
    pub counters: GpuPtr<LocalShadowCounters>,
    pub depth_texture_id: u32,
    pub surface_material_texture_id: u32,
    pub surface_normal_texture_id: u32,
    pub history_valid: u32,
    pub screen_size: [u32; 2],
    pub half_size: [u32; 2],
    pub light_count: u32,
    pub origin_bias: f32,
    pub destination_bias: f32,
    /// Diffuse wrap parameter used during light selection.
    pub wrap_w: f32,
    pub request_capacity: u32,
    /// Frames a visible answer remains trusted; zero disables reuse.
    pub refresh_interval: u32,
    /// World-unit thickness for the occluded-history validation tap.
    pub validate_thickness: f32,
    pub near_plane: f32,
    /// Movement threshold invalidating visible history.
    pub light_epsilon: f32,
    /// Contact-march reach toward the light.
    pub contact_distance: f32,
    /// Contact-march controls; zero steps disables the stage.
    pub contact: DepthMarchConfig,
    /// Per-frame ray budget; zero is unlimited.
    pub ray_budget: u32,
    /// Enables re-raying mixed visibility edges.
    pub edge_promotion: u32,
    /// Host frame counter driving the promoted-texel corner rotation.
    pub frame_index: u32,
    /// Maximum age before occlusion revalidation; zero disables revalidation.
    pub occluded_refresh: u32,
    /// World-space source radius. The blur width is
    /// `source_radius * t / (1 - t)` for blocker parameter `t`; zero keeps
    /// edges hard.
    pub source_radius: f32,
    /// Projection y scale (`|row1(world_to_clip)|`) used in
    /// `texel_world = 4 * eye_z / (proj_scale * screen_height)`.
    pub proj_scale: f32,
    /// Remaps current lights to previous-frame light indices.
    pub light_remap: GpuPtr<u32>,
    pub _pad0: [u32; 2],
}

const _: () = assert!(core::mem::size_of::<LocalShadowData>() == 480);
const _: () = assert!(core::mem::align_of::<LocalShadowData>() == 16);
const _: () = assert!(core::mem::offset_of!(LocalShadowData, world) == 192);
const _: () = assert!(core::mem::offset_of!(LocalShadowData, lights) == 232);
const _: () = assert!(core::mem::offset_of!(LocalShadowData, slot_state) == 264);
const _: () = assert!(core::mem::offset_of!(LocalShadowData, counters) == 328);
const _: () = assert!(core::mem::offset_of!(LocalShadowData, depth_texture_id) == 336);
const _: () = assert!(core::mem::offset_of!(LocalShadowData, screen_size) == 352);
const _: () = assert!(core::mem::offset_of!(LocalShadowData, light_count) == 368);
const _: () = assert!(core::mem::offset_of!(LocalShadowData, refresh_interval) == 388);

/// Per-frame local-shadow diagnostics and dispatch budget.
#[gpu_data]
pub struct LocalShadowCounters {
    /// High-priority requests for slots without usable history.
    pub requests_high: u32,
    /// Low-priority requests refresh carried visible estimates.
    pub requests_low: u32,
    /// Requests dropped at capacity; this must remain zero.
    pub overflow: u32,
    pub texel_count: u32,
    /// Occluded history confirmed by the validation tap (no ray spent).
    pub validated: u32,
    /// Visible history trusted under the refresh interval (no ray spent).
    pub reused: u32,
    /// History discarded after reprojection or validation failure.
    pub invalidated: u32,
    /// Occlusions proven by the contact march (no ray spent).
    pub contact: u32,
    /// High-priority requests serviced this frame.
    pub serviced_high: u32,
    /// Low-priority requests serviced this frame.
    pub serviced_low: u32,
    /// Texels with at least one filled slot.
    pub active_texels: u32,
    /// Slots re-rayed by visibility-space edge promotion.
    pub promoted: u32,
    /// Screen-blind occlusions carried on the leash (no ray, no tap).
    pub blind: u32,
    /// Texels whose selected lights differ from reprojected history.
    pub churn: u32,
    pub _pad0: [u32; 2],
}

const _: () = assert!(core::mem::size_of::<LocalShadowCounters>() == 64);
const _: () = assert!(core::mem::align_of::<LocalShadowCounters>() == 4);
const _: () = assert!(core::mem::offset_of!(LocalShadowCounters, serviced_high) == 32);
const _: () = assert!(core::mem::offset_of!(LocalShadowCounters, blind) == 48);
const _: () = assert!(core::mem::offset_of!(LocalShadowCounters, churn) == 52);

/// Dispatch data for building the indirect trace plan.
#[gpu_data]
pub struct LocalShadowArgsData {
    pub counters: GpuPtr<LocalShadowCounters>,
    pub dispatch_args: GpuPtr<u32>,
    pub queue_capacity: u32,
    pub group_size: u32,
    /// 0 is unlimited.
    pub ray_budget: u32,
    pub _pad0: u32,
}

const _: () = assert!(core::mem::size_of::<LocalShadowArgsData>() == 32);
const _: () = assert!(core::mem::align_of::<LocalShadowArgsData>() == 4);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_state_word_carries_blind_flag_age_and_hit() {
        let word = local_shadow_pack(
            crate::shadow::SHADOW_STATE_OCCLUDED | LOCAL_SHADOW_BLIND,
            42,
            0xBEEF,
        );
        assert_eq!(
            local_shadow_state(word),
            crate::shadow::SHADOW_STATE_OCCLUDED
        );
        assert!(local_shadow_blind(word));
        assert_eq!(local_shadow_age(word), 42);
        assert_eq!(local_shadow_hit_q(word), 0xBEEF);
        assert!(!local_shadow_blind(local_shadow_pack(
            crate::shadow::SHADOW_STATE_OCCLUDED,
            42,
            0xBEEF
        )));
    }

    #[test]
    fn slot_word_lanes_round_trip_and_find() {
        let mut word = LOCAL_SHADOW_SLOT_WORD_EMPTY;
        for slot in 0..LOCAL_SHADOW_SLOTS {
            assert_eq!(local_shadow_slot_get(word, slot), LOCAL_SHADOW_SLOT_EMPTY);
        }
        word = local_shadow_slot_set(word, 0, 3);
        word = local_shadow_slot_set(word, 2, LOCAL_SHADOW_MAX_LIGHT_ID);
        assert_eq!(local_shadow_slot_get(word, 0), 3);
        assert_eq!(local_shadow_slot_get(word, 1), LOCAL_SHADOW_SLOT_EMPTY);
        assert_eq!(local_shadow_slot_get(word, 2), LOCAL_SHADOW_MAX_LIGHT_ID);
        assert_eq!(local_shadow_slot_find(word, 3), 0);
        assert_eq!(local_shadow_slot_find(word, LOCAL_SHADOW_MAX_LIGHT_ID), 2);
        assert_eq!(local_shadow_slot_find(word, 7), LOCAL_SHADOW_SLOTS);
        assert_eq!(
            local_shadow_slot_find(word, LOCAL_SHADOW_SLOT_EMPTY),
            LOCAL_SHADOW_SLOTS,
            "the sentinel must never match a lane"
        );
        word = local_shadow_slot_set(word, 0, LOCAL_SHADOW_SLOT_EMPTY);
        assert_eq!(local_shadow_slot_get(word, 0), LOCAL_SHADOW_SLOT_EMPTY);
    }

    #[test]
    fn fresh_age_staggers_births_across_the_half_window() {
        assert_eq!(local_shadow_fresh_age(7, 0x0A53, 0), 0);
        assert_eq!(local_shadow_fresh_age(7, 0x0A53, 1), 0);
        let interval = 8;
        let mut seen = [false; 5];
        for id in 0..256 {
            let phase = local_shadow_fresh_age(id, 0x0A53, interval);
            assert!(phase <= interval / 2, "first trust window must keep half");
            seen[phase as usize] = true;
        }
        assert!(
            seen.iter().all(|&s| s),
            "256 ids must cover the half window"
        );
        // Long leashes clamp at the packed age ceiling instead of wrapping.
        for id in 0..64 {
            assert!(local_shadow_fresh_age(id, 0x0C91, 100_000) <= LOCAL_SHADOW_AGE_MAX);
        }
    }
}
