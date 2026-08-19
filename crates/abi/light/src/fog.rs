//! Volumetric-fog curve parameters and scalar twins.

use crate::{GpuPtr, gpu_data};
use crate::{PointLight, point_light_radiance};
use abi_core::View;
use glam::Vec3;

// Scalar transcendentals are std-only; on the GPU they come from the
// num_traits::Float shim spirv-std re-exports.
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

pub const FOG_OPTICAL_DEPTH_CAP: f32 = 88.0;
pub const EXT_MAX: f32 = 5.541_263;
pub const EXT_LANES_PER_DWORD: u32 = 4;
/// Throughput at or below one u8 extinction quantum is represented as zero
/// for conservative zero-transmittance priming.
pub const ZERO_TRANS_EPS: f32 = 1.0 / 255.0;
/// `zero_slice` sentinel: the column never reached `ZERO_TRANS_EPS`.
pub const ZERO_SLICE_NONE: u32 = u32::MAX;
/// Screen-tile edge for the zero-transmittance depth prime, in pixels.
/// Values at least 16 preserve depth compression efficiency.
pub const PRIME_TILE: u32 = 16;
/// One local-light list per 8×8 froxel columns: exactly one `fog_light`
/// workgroup, so every invocation in the group reads the same short list.
pub const FOG_LIGHT_TILE: u32 = 8;
/// Inline tile-list budget. Overflow never drops lights: the tile marks
/// itself and `fog_light` falls back to the bounded complete light array.
pub const FOG_LIGHTS_PER_TILE: u32 = 32;
/// Hard loop bound for malformed/content-heavy frames. Host validation
/// rejects larger arrays; overflow tiles therefore remain bounded too.
pub const FOG_LOCAL_LIGHT_MAX: u32 = 256;
/// Empty projected-bounds sentinel (`min > max`).
pub const FOG_LIGHT_BOUNDS_NONE: [u32; 4] = [1, 1, 0, 0];
const MIN_LINEARIZATION: f32 = 1.0e-4;
const MIN_FAR: f32 = 1.0e-3;
const MIN_CURVE_LOG2: f32 = 1.0e-6;
const DIR_Y_EPS: f32 = 1.0e-6;

/// Per-frame constants for the parametric froxel-Z curve.
#[gpu_data]
pub struct FroxelParams {
    pub f: f32,
    pub a: f32,
    pub inv_a: f32,
    pub slice_count: f32,
    /// `slice_count / log2(f / a + 1)`.
    pub slice_scale: f32,
    /// `log2(f / a + 1) / slice_count`.
    pub z_scale: f32,
    pub slice_count_u32: u32,
    pub _pad: u32,
}

const _: () = assert!(core::mem::size_of::<FroxelParams>() == 32);
const _: () = assert!(core::mem::align_of::<FroxelParams>() == 4);
const _: () = assert!(core::mem::offset_of!(FroxelParams, slice_scale) == 16);

/// Slice capacity supported by the warp lookup table.
pub const FOG_SLICE_MAX: u32 = 64;
/// Warp LUT edge count: slice edges `0..=FOG_SLICE_MAX`.
pub const FOG_WARP_EDGES: usize = FOG_SLICE_MAX as usize + 1;

/// Per-frame froxel curve and its piecewise-linear warp tables.
///
/// Endpoints are pinned to zero and `slice_count`. The zero value is safe and
/// maps all coordinates to slice zero.
#[repr(C)]
#[derive(Copy, Clone)]
#[cfg_attr(not(target_arch = "spirv"), derive(Debug))]
pub struct FogCurve {
    pub params: FroxelParams,
    pub warp: [f32; FOG_WARP_EDGES],
    pub unwarp: [f32; FOG_WARP_EDGES],
}

impl Default for FogCurve {
    fn default() -> Self {
        Self {
            params: FroxelParams::default(),
            warp: [0.0; FOG_WARP_EDGES],
            unwarp: [0.0; FOG_WARP_EDGES],
        }
    }
}

// SAFETY: repr(C), fields are f32/u32 aggregates with explicit padding only
// (FroxelParams carries its own _pad); 552 bytes, no implicit padding, any
// bit pattern is a valid (if degenerate) curve.
#[cfg(not(target_arch = "spirv"))]
unsafe impl bytemuck::Zeroable for FogCurve {}
#[cfg(not(target_arch = "spirv"))]
unsafe impl bytemuck::Pod for FogCurve {}

const _: () = assert!(core::mem::size_of::<FogCurve>() == 32 + 8 * FOG_WARP_EDGES);
const _: () = assert!(core::mem::align_of::<FogCurve>() == 4);
const _: () = assert!(core::mem::offset_of!(FogCurve, warp) == 32);
const _: () = assert!(core::mem::offset_of!(FogCurve, unwarp) == 32 + 4 * FOG_WARP_EDGES);

/// Piecewise-linear warp LUT evaluation over slice edges. Outside `[0, n]`
/// the warp is the IDENTITY — the volume is only warped inside itself, and
/// consumers that extrapolate past the far bound (`slice_of_z` beyond `f`,
/// `prime_quad_depth` with an overflowing margin) must keep the raw curve's
/// analytic continuation. An identity LUT (edge `i` holds exactly `i`)
/// reproduces `x` bit-exactly: `i + (x - i)` loses nothing for `x` in
/// `[0, 64]`.
pub fn warp_eval(lut: &[f32; FOG_WARP_EDGES], n: u32, x: f32) -> f32 {
    let n = n.clamp(1, FOG_SLICE_MAX);
    if x <= 0.0 || x >= n as f32 {
        return x;
    }
    let i = (x as u32).min(n - 1) as usize;
    let lo = lut[i];
    lo + (lut[i + 1] - lo) * (x - i as f32)
}

/// Warped volume-slice coordinate of a view-space distance.
pub fn warped_slice_of_z(curve: &FogCurve, view_z: f32) -> f32 {
    let raw = slice_of_z(&curve.params, view_z);
    warp_eval(&curve.warp, curve.params.slice_count_u32, raw)
}

/// View-space distance at a warped volume-slice coordinate.
pub fn z_of_warped_slice(curve: &FogCurve, slice: f32) -> f32 {
    let raw = warp_eval(&curve.unwarp, curve.params.slice_count_u32, slice);
    z_of_slice(&curve.params, raw)
}

/// Bounded-waterfill slice widths from a raw-slice event histogram. Demand
/// per slice is `1 + gain·ĥ` (`ĥ` normalized to mean 1 over the active
/// slices); widths are `clamp(λ·demand, 1/bound, bound)` with `λ` bisected
/// so they sum to `n` — the bound is exact, never renormalized away. Zero
/// gain, an empty histogram, or `bound <= 1` short-circuit to exact unit
/// widths: the identity warp.
pub fn warp_widths(
    hist: &[u32; FOG_SLICE_MAX as usize],
    n: u32,
    gain: f32,
    bound: f32,
    widths: &mut [f32; FOG_SLICE_MAX as usize],
) {
    let n = n.clamp(1, FOG_SLICE_MAX);
    let gain = if gain.is_finite() { gain.max(0.0) } else { 0.0 };
    let bound = if bound.is_finite() { bound } else { 1.0 };
    let mut total = 0.0f32;
    let mut i = 0u32;
    while i < n {
        total += hist[i as usize] as f32;
        i += 1;
    }
    let mut i = 0u32;
    while i < FOG_SLICE_MAX {
        widths[i as usize] = 1.0;
        i += 1;
    }
    if gain <= 0.0 || total <= 0.0 || bound <= 1.0 {
        return;
    }

    let norm = n as f32 / total;
    let (lo, hi) = (1.0 / bound, bound);
    // Demand >= 1 everywhere, so λ = bound saturates every width (sum >= n)
    // and λ = 0 floors it (sum <= n): the root is bracketed.
    let mut lam_lo = 0.0f32;
    let mut lam_hi = hi;
    let mut iter = 0u32;
    while iter < 32 {
        let lam = 0.5 * (lam_lo + lam_hi);
        let mut sum = 0.0f32;
        let mut i = 0u32;
        while i < n {
            let demand = 1.0 + gain * hist[i as usize] as f32 * norm;
            sum += (lam * demand).clamp(lo, hi);
            i += 1;
        }
        if sum < n as f32 {
            lam_lo = lam;
        } else {
            lam_hi = lam;
        }
        iter += 1;
    }
    let lam = 0.5 * (lam_lo + lam_hi);
    let mut i = 0u32;
    while i < n {
        let demand = 1.0 + gain * hist[i as usize] as f32 * norm;
        widths[i as usize] = (lam * demand).clamp(lo, hi);
        i += 1;
    }
}

/// Assemble the complete per-frame curve: log params, forward warp as the
/// prefix sum of the waterfill widths (endpoint pinned to `n` exactly so
/// consumer clamps stay honest), and the monotone piecewise-linear inverse
/// sampled at integer warped edges.
pub fn fog_curve_from(
    params: FroxelParams,
    hist: &[u32; FOG_SLICE_MAX as usize],
    gain: f32,
    bound: f32,
) -> FogCurve {
    let n = params.slice_count_u32.clamp(1, FOG_SLICE_MAX);
    let mut widths = [1.0f32; FOG_SLICE_MAX as usize];
    warp_widths(hist, n, gain, bound, &mut widths);

    let mut curve = FogCurve {
        params,
        warp: [0.0; FOG_WARP_EDGES],
        unwarp: [0.0; FOG_WARP_EDGES],
    };
    let mut i = 0u32;
    while i < FOG_SLICE_MAX {
        curve.warp[i as usize + 1] = curve.warp[i as usize] + widths[i as usize];
        i += 1;
    }
    let scale = n as f32 / curve.warp[n as usize].max(1.0e-6);
    let mut i = 1u32;
    while i <= FOG_SLICE_MAX {
        curve.warp[i as usize] *= scale;
        i += 1;
    }

    let mut seg = 0u32;
    let mut j = 0u32;
    while j <= FOG_SLICE_MAX {
        let target = j as f32;
        while seg + 1 < FOG_SLICE_MAX && curve.warp[seg as usize + 1] < target {
            seg += 1;
        }
        let w0 = curve.warp[seg as usize];
        let w1 = curve.warp[seg as usize + 1];
        let frac = ((target - w0) / (w1 - w0).max(1.0e-6)).clamp(0.0, 1.0);
        curve.unwarp[j as usize] = seg as f32 + frac;
        j += 1;
    }
    curve
}

/// Dispatch data for `fog_depth_max`.
#[gpu_data]
pub struct FogDepthMaxData {
    pub max_depth_bits: GpuPtr<u32>,
    pub depth_texture_id: u32,
    pub _pad0: u32,
    pub screen_size: [u32; 2],
    pub depth_near_plane: f32,
    pub _pad1: u32,
}

const _: () = assert!(core::mem::size_of::<FogDepthMaxData>() == 32);
const _: () = assert!(core::mem::align_of::<FogDepthMaxData>() == 4);

/// Dispatch data for `fog_params`.
#[gpu_data]
pub struct FogParamsData {
    pub max_depth_bits: GpuPtr<u32>,
    pub curve_out: GpuPtr<FogCurve>,
    /// Previous-frame event histogram used to build this frame's warp.
    pub hist: GpuPtr<u32>,
    pub slice_count: u32,
    pub a: f32,
    pub f_min: f32,
    pub f_max: f32,
    pub warp_gain: f32,
    pub warp_bound: f32,
}

const _: () = assert!(core::mem::size_of::<FogParamsData>() == 48);
const _: () = assert!(core::mem::align_of::<FogParamsData>() == 4);
const _: () = assert!(core::mem::offset_of!(FogParamsData, hist) == 16);
const _: () = assert!(core::mem::offset_of!(FogParamsData, warp_gain) == 40);

/// Dispatch data for `fog_light`.
#[gpu_data]
pub struct FogLightData {
    pub view: abi_core::View,
    pub params: GpuPtr<FogCurve>,
    pub v_scatter_texture: u32,
    /// Generalized occluder volume contract: sampled-heap 3D texture slot
    /// whose value in [0, 1] is opacity at `uvw = (pos - world_min) *
    /// world_inv_extent`. Slot 0 is the ZII sentinel and means no occluder.
    /// Producers present occupancy as opacity: mesh voxelization,
    /// SDF-derived opacity, or baked occupancy.
    pub occluder_texture: u32,
    pub occluder_sampler: u32,
    pub _pad0: u32,
    pub occluder_world_min: [f32; 3],
    pub occluder_world_inv_extent: [f32; 3],
    pub sun_dir: [f32; 3],
    pub sun_color: [f32; 3],
    pub ambient_color: [f32; 3],
    pub anisotropy_g: f32,
    pub gradient_bottom: [f32; 3],
    pub gradient_top: [f32; 3],
    pub gradient_offset: f32,
    pub gradient_length: f32,
    pub density: f32,
    pub height_falloff: f32,
    pub height_offset: f32,
    /// 0 skips occluder marching; fog self-shadow still uses the analytic
    /// closed form, replacing the spec's sun optical-depth volume while the
    /// density remains analytic.
    pub sun_occlusion_steps: u32,
    pub sun_occlusion_lod_ramp: f32,
    pub _pad1: u32,
    /// Froxel-owned Forward+-style local-light grid. Zero count is the ZII
    /// no-local-light path; nonzero count requires every pointer below.
    pub local_lights: GpuPtr<PointLight>,
    pub local_tile_counts: GpuPtr<u32>,
    pub local_tile_indices: GpuPtr<u32>,
    pub local_tile_overflow: GpuPtr<u32>,
    pub local_light_count: u32,
    pub local_tile_count: [u32; 2],
    pub local_lights_per_tile: u32,
}

const _: () = assert!(core::mem::size_of::<FogLightData>() == 272);
const _: () = assert!(core::mem::align_of::<FogLightData>() == 4);
const _: () = assert!(core::mem::offset_of!(FogLightData, params) == 80);
const _: () = assert!(core::mem::offset_of!(FogLightData, v_scatter_texture) == 88);
const _: () = assert!(core::mem::offset_of!(FogLightData, occluder_world_min) == 104);
const _: () = assert!(core::mem::offset_of!(FogLightData, sun_dir) == 128);
const _: () = assert!(core::mem::offset_of!(FogLightData, gradient_offset) == 192);
const _: () = assert!(core::mem::offset_of!(FogLightData, sun_occlusion_steps) == 212);
const _: () = assert!(core::mem::offset_of!(FogLightData, local_lights) == 224);
const _: () = assert!(core::mem::offset_of!(FogLightData, local_tile_counts) == 232);
const _: () = assert!(core::mem::offset_of!(FogLightData, local_light_count) == 256);
const _: () = assert!(core::mem::offset_of!(FogLightData, local_lights_per_tile) == 268);

/// One-thread-per-light culling dispatch that bins conservative projected
/// light-sphere bounds into the froxel workgroup grid.
#[gpu_data]
pub struct FogLightGridData {
    pub view: View,
    pub lights: GpuPtr<PointLight>,
    pub tile_counts: GpuPtr<u32>,
    pub tile_indices: GpuPtr<u32>,
    pub tile_overflow: GpuPtr<u32>,
    pub light_count: u32,
    pub tile_count: [u32; 2],
    pub lights_per_tile: u32,
}

const _: () = assert!(core::mem::size_of::<FogLightGridData>() == 128);
const _: () = assert!(core::mem::align_of::<FogLightGridData>() == 4);
const _: () = assert!(core::mem::offset_of!(FogLightGridData, lights) == 80);
const _: () = assert!(core::mem::offset_of!(FogLightGridData, light_count) == 112);
const _: () = assert!(core::mem::offset_of!(FogLightGridData, lights_per_tile) == 124);

/// Dispatch data for `fog_integrate`.
#[gpu_data]
pub struct FogIntegrateData {
    pub view: abi_core::View,
    pub params: GpuPtr<FogCurve>,
    pub v_scatter_texture: u32,
    pub v_int_texture: u32,
    pub volume_depth: u32,
    pub _pad0: u32,
    pub v_ext: GpuPtr<u32>,
    pub overflow: GpuPtr<u32>,
    /// Per froxel column: first slice index where prefix throughput fell to
    /// [`ZERO_TRANS_EPS`] (or the overflow slice), else [`ZERO_SLICE_NONE`].
    pub zero_slice: GpuPtr<u32>,
    /// Per-column occupancy bits for extinction slices.
    pub occupancy: GpuPtr<u32>,
    pub v_ext_rgb: GpuPtr<u32>,
    pub occupancy_rgb: GpuPtr<u32>,
    pub overflow_rgb: GpuPtr<u32>,
    pub oit_enable: u32,
    /// Nonzero only when the frame carries tinted particles: gates every
    /// RGB read and the `V_tint` write so monochrome frames pay nothing.
    pub tinted_enable: u32,
    pub v_tint_texture: u32,
    pub _pad: u32,
}

const _: () = assert!(core::mem::size_of::<FogIntegrateData>() == 176);
const _: () = assert!(core::mem::align_of::<FogIntegrateData>() == 4);
const _: () = assert!(core::mem::offset_of!(FogIntegrateData, params) == 80);
const _: () = assert!(core::mem::offset_of!(FogIntegrateData, v_scatter_texture) == 88);
const _: () = assert!(core::mem::offset_of!(FogIntegrateData, v_ext) == 104);
const _: () = assert!(core::mem::offset_of!(FogIntegrateData, zero_slice) == 120);
const _: () = assert!(core::mem::offset_of!(FogIntegrateData, occupancy) == 128);
const _: () = assert!(core::mem::offset_of!(FogIntegrateData, v_ext_rgb) == 136);
const _: () = assert!(core::mem::offset_of!(FogIntegrateData, oit_enable) == 160);

/// Dispatch data for `fog_composite`.
#[gpu_data]
pub struct FogCompositeData {
    pub view: abi_core::View,
    pub params: GpuPtr<FogCurve>,
    pub sun_dir: [f32; 3],
    pub anisotropy_g: f32,
    pub sun_color: [f32; 3],
    /// Composite still needs the analytic density model for the beyond-far
    /// continuation; in-volume lighting now comes from `V_scatter`.
    pub density: f32,
    pub ambient_color: [f32; 3],
    pub height_falloff: f32,
    pub height_offset: f32,
    pub depth_texture_id: u32,
    pub v_int_texture: u32,
    pub v_int_sampler: u32,
    pub hdr_texture: u32,
    pub fog_sample_bias: f32,
    pub v_tint_texture: u32,
    pub tinted_enable: u32,
}

const _: () = assert!(core::mem::size_of::<FogCompositeData>() == 168);
const _: () = assert!(core::mem::align_of::<FogCompositeData>() == 4);
const _: () = assert!(core::mem::offset_of!(FogCompositeData, params) == 80);
const _: () = assert!(core::mem::offset_of!(FogCompositeData, sun_dir) == 88);
const _: () = assert!(core::mem::offset_of!(FogCompositeData, ambient_color) == 120);
const _: () = assert!(core::mem::offset_of!(FogCompositeData, fog_sample_bias) == 156);

/// One zero-transmittance cover quad: a screen tile behind which the medium
/// is opaque. `tile` packs (x | y << 16); `depth` is reverse-Z hardware depth.
#[gpu_data]
pub struct FogPrimeQuad {
    pub tile: u32,
    pub depth: f32,
}

const _: () = assert!(core::mem::size_of::<FogPrimeQuad>() == 8);
const _: () = assert!(core::mem::align_of::<FogPrimeQuad>() == 4);

/// Dispatch data for `fog_prime_spawn`.
#[gpu_data]
pub struct FogPrimeSpawnData {
    pub params: GpuPtr<FogCurve>,
    pub zero_slice: GpuPtr<u32>,
    pub quads: GpuPtr<FogPrimeQuad>,
    /// The `DrawIndexedIndirectCommand` viewed as dwords; index 1 is
    /// `instance_count`, atomically bumped per spawned quad.
    pub draw_args: GpuPtr<u32>,
    pub froxel_size: [u32; 2],
    pub screen_size: [u32; 2],
    pub depth_near_plane: f32,
    /// Fractional slices added past the zero boundary before the primed
    /// depth: `fog_sample_bias + 1.5` — consumers sample `V_int` up to
    /// `bias` slices before their own depth, and the trilinear footprint
    /// reaches another 1.5 texels; both must land at or past the boundary
    /// for a culled fragment's contribution to be provably ≤ 1/255.
    pub slice_margin: f32,
    pub _pad: [u32; 2],
}

const _: () = assert!(core::mem::size_of::<FogPrimeSpawnData>() == 64);
const _: () = assert!(core::mem::align_of::<FogPrimeSpawnData>() == 4);
const _: () = assert!(core::mem::offset_of!(FogPrimeSpawnData, froxel_size) == 32);
const _: () = assert!(core::mem::offset_of!(FogPrimeSpawnData, depth_near_plane) == 48);
const _: () = assert!(core::mem::offset_of!(FogPrimeSpawnData, slice_margin) == 52);

/// Vertex data for `fog_prime_vert`.
#[gpu_data]
pub struct FogPrimeVertData {
    pub quads: GpuPtr<FogPrimeQuad>,
    pub screen_size: [u32; 2],
}

const _: () = assert!(core::mem::size_of::<FogPrimeVertData>() == 16);
const _: () = assert!(core::mem::align_of::<FogPrimeVertData>() == 4);

/// Transparent billboard particle consumed by the OIT splat and accum draws.
#[gpu_data]
pub struct OitParticle {
    pub pos: [f32; 3],
    pub size: f32,
    pub color: [f32; 3],
    pub alpha: f32,
    /// Per-channel EXTRA optical depth on top of the scalar coverage —
    /// colored transmission (tinted glass, colored smoke). Zero is the
    /// neutral monochrome path (ZII); `alpha` alone keeps carrying
    /// coverage, ordering weight, and the zero-transmittance boundary.
    /// From an artist transmittance tint: [`tint_od_from_transmittance`].
    pub tint_od: [f32; 3],
    pub _pad: u32,
}

const _: () = assert!(core::mem::size_of::<OitParticle>() == 48);
const _: () = assert!(core::mem::align_of::<OitParticle>() == 4);
const _: () = assert!(core::mem::offset_of!(OitParticle, color) == 16);
const _: () = assert!(core::mem::offset_of!(OitParticle, alpha) == 28);
const _: () = assert!(core::mem::offset_of!(OitParticle, tint_od) == 32);

/// Vertex data for `oit_splat_vert`.
#[gpu_data]
pub struct OitSplatVertData {
    pub view: abi_core::View,
    pub params: GpuPtr<FogCurve>,
    pub particles: GpuPtr<OitParticle>,
}

const _: () = assert!(core::mem::size_of::<OitSplatVertData>() == 96);
const _: () = assert!(core::mem::align_of::<OitSplatVertData>() == 4);
const _: () = assert!(core::mem::offset_of!(OitSplatVertData, params) == 80);

/// Fragment data for `oit_splat_frag`.
#[gpu_data]
pub struct OitSplatFragData {
    pub view: abi_core::View,
    pub params: GpuPtr<FogCurve>,
    pub v_ext: GpuPtr<u32>,
    pub overflow: GpuPtr<u32>,
    /// Per-column occupancy bits for splatted extinction.
    pub occupancy: GpuPtr<u32>,
    /// Current-frame raw-slice demand histogram.
    pub hist: GpuPtr<u32>,
    /// RGB extinction words, one per (column, slice) via [`ext_rgb_index`];
    /// written only by fragments whose `tint_od` is nonzero.
    pub v_ext_rgb: GpuPtr<u32>,
    /// RGB twin of `occupancy` — integrate reads `v_ext_rgb` only where set.
    pub occupancy_rgb: GpuPtr<u32>,
    /// Per-CHANNEL first-overflow min slice, channel-major (c * columns +
    /// column); integration saturates that channel from there on.
    pub overflow_rgb: GpuPtr<u32>,
}

const _: () = assert!(core::mem::size_of::<OitSplatFragData>() == 144);
const _: () = assert!(core::mem::align_of::<OitSplatFragData>() == 4);
const _: () = assert!(core::mem::offset_of!(OitSplatFragData, params) == 80);
const _: () = assert!(core::mem::offset_of!(OitSplatFragData, occupancy) == 104);
const _: () = assert!(core::mem::offset_of!(OitSplatFragData, v_ext_rgb) == 120);

/// Vertex data for `oit_accum_vert`.
#[gpu_data]
pub struct OitAccumVertData {
    pub view: abi_core::View,
    pub params: GpuPtr<FogCurve>,
    pub particles: GpuPtr<OitParticle>,
}

const _: () = assert!(core::mem::size_of::<OitAccumVertData>() == 96);
const _: () = assert!(core::mem::align_of::<OitAccumVertData>() == 4);
const _: () = assert!(core::mem::offset_of!(OitAccumVertData, params) == 80);

/// Fragment data for `oit_accum_frag`.
#[gpu_data]
pub struct OitAccumFragData {
    pub view: abi_core::View,
    pub params: GpuPtr<FogCurve>,
    pub v_int_texture: u32,
    pub v_int_sampler: u32,
    pub fog_sample_bias: f32,
    pub v_tint_texture: u32,
    pub tinted_enable: u32,
    pub _pad: u32,
}

const _: () = assert!(core::mem::size_of::<OitAccumFragData>() == 112);
const _: () = assert!(core::mem::align_of::<OitAccumFragData>() == 4);
const _: () = assert!(core::mem::offset_of!(OitAccumFragData, params) == 80);
const _: () = assert!(core::mem::offset_of!(OitAccumFragData, fog_sample_bias) == 96);
const _: () = assert!(core::mem::offset_of!(OitAccumFragData, v_tint_texture) == 100);

/// Dispatch data for `oit_resolve`.
#[gpu_data]
pub struct OitResolveData {
    pub view: abi_core::View,
    pub params: GpuPtr<FogCurve>,
    pub accum_rgb_texture: u32,
    pub accum_moments_texture: u32,
    pub hdr_texture: u32,
    pub _pad: u32,
}

const _: () = assert!(core::mem::size_of::<OitResolveData>() == 104);
const _: () = assert!(core::mem::align_of::<OitResolveData>() == 4);
const _: () = assert!(core::mem::offset_of!(OitResolveData, params) == 80);

/// Parametric-log froxel slice for a positive view-space distance.
pub fn slice_of_z(params: &FroxelParams, view_z: f32) -> f32 {
    ((view_z.max(0.0) * params.inv_a) + 1.0).log2() * params.slice_scale
}

/// View-space distance at a parametric-log froxel slice boundary.
pub fn z_of_slice(params: &FroxelParams, slice: f32) -> f32 {
    params.a * ((slice * params.z_scale).exp2() - 1.0)
}

/// Construct the per-frame froxel curve constants from the depth reduction.
pub fn froxel_params_from(max_depth: f32, n: u32, a: f32, f_min: f32, f_max: f32) -> FroxelParams {
    let f_lo = finite_or(f_min, MIN_FAR).max(MIN_FAR);
    let f_hi = finite_or(f_max, f_lo).max(f_lo);
    let depth = if max_depth.is_finite() {
        max_depth.max(0.0)
    } else if max_depth.is_sign_positive() {
        f_hi
    } else {
        0.0
    };
    let f = depth.clamp(f_lo, f_hi);
    let a = finite_or(a, MIN_LINEARIZATION).max(MIN_LINEARIZATION);
    let slice_count_u32 = n.max(1);
    let slice_count = slice_count_u32 as f32;
    let inv_a = 1.0 / a;
    let curve_log2 = (f * inv_a + 1.0).log2().max(MIN_CURVE_LOG2);
    FroxelParams {
        f,
        a,
        inv_a,
        slice_count,
        slice_scale: slice_count / curve_log2,
        z_scale: curve_log2 / slice_count,
        slice_count_u32,
        _pad: 0,
    }
}

/// Closed-form optical depth for exponential height fog over `[t0, t1]`.
pub fn height_fog_optical_depth(
    h0: f32,
    dir_y: f32,
    t0: f32,
    t1: f32,
    density: f32,
    falloff: f32,
    height_offset: f32,
) -> f32 {
    let density = density.max(0.0);
    if density == 0.0 || t1 <= t0 || !t0.is_finite() || t1.is_nan() {
        return 0.0;
    }
    let falloff = falloff.max(0.0);
    if falloff == 0.0 {
        return uniform_optical_depth(density, t0, t1);
    }
    if dir_y.abs() <= DIR_Y_EPS {
        let h = (h0 + dir_y * t0 - height_offset).max(0.0);
        let sigma = density * (-falloff * h).exp();
        return uniform_optical_depth(sigma, t0, t1);
    }

    let b = h0 - height_offset;
    let mut od = 0.0;
    if dir_y > 0.0 {
        let cross = -b / dir_y;
        let above_start = if cross > t0 {
            if t1.is_finite() && cross >= t1 {
                return uniform_optical_depth(density, t0, t1);
            }
            od += density * (cross - t0);
            cross
        } else {
            t0
        };
        od += exponential_optical_depth(b, dir_y, above_start, t1, density, falloff);
    } else {
        if !t1.is_finite() {
            return FOG_OPTICAL_DEPTH_CAP;
        }
        let cross = -b / dir_y;
        if cross <= t0 {
            od += density * (t1 - t0);
        } else if cross >= t1 {
            od += exponential_optical_depth(b, dir_y, t0, t1, density, falloff);
        } else {
            od += exponential_optical_depth(b, dir_y, t0, cross, density, falloff);
            od += density * (t1 - cross);
        }
    }
    od.clamp(0.0, FOG_OPTICAL_DEPTH_CAP)
}

/// Beer-Lambert transmittance for a scalar optical depth.
pub fn transmittance(optical_depth: f32) -> f32 {
    (-optical_depth).exp()
}

/// Henyey-Greenstein phase normalized relative to isotropic scattering.
///
/// With `g = 0`, this returns exactly 1 for every angle. The physical
/// `1 / (4*pi)` factor is intentionally omitted so the isotropic case is
/// energy-neutral: the dial changes shape, not brightness.
pub fn hg_phase(cos_theta: f32, g: f32) -> f32 {
    let g = g.clamp(-0.99, 0.99);
    let cos_theta = cos_theta.clamp(-1.0, 1.0);
    let g2 = g * g;
    let d = (1.0 + g2 - 2.0 * g * cos_theta).max(1.0e-6);
    (1.0 - g2) / (d * d.sqrt())
}

/// Local point-light radiance scattered toward the camera at one medium
/// sample. Surface lighting and fog share [`point_light_radiance`]'s finite
/// radius attenuation; fog replaces the surface cosine with HG phase and
/// analytically attenuates the light→sample segment through height fog.
pub fn fog_point_light_radiance(
    view_dir: Vec3,
    position_world: Vec3,
    light: &PointLight,
    anisotropy_g: f32,
    density: f32,
    height_falloff: f32,
    height_offset: f32,
) -> Vec3 {
    if light.intensity <= 0.0 || light.radius <= 0.0 {
        return Vec3::ZERO;
    }
    let incident = point_light_radiance(position_world, light);
    let to_light = Vec3::from_array(light.position) - position_world;
    let distance = to_light.length().max(1.0e-4);
    let light_dir = to_light / distance;
    let phase = hg_phase(view_dir.dot(light_dir), anisotropy_g);
    let light_t = transmittance(height_fog_optical_depth(
        position_world.y,
        light_dir.y,
        0.0,
        distance,
        density,
        height_falloff,
        height_offset,
    ));
    incident * (phase * light_t)
}

/// Conservative inclusive workgroup-tile bounds of one projected light
/// sphere. The perspective-radius bound follows the quotient error bound
/// `r(z + |x|) / (z(z-r))`; when a sphere reaches the camera plane the full
/// screen is the only honest bound. Distance/radius checks in `fog_light`
/// make all over-coverage inert.
///
/// Tile membership mirrors the consumer exactly: `fog_light` reads the list
/// at `froxel / FOG_LIGHT_TILE`, so bounds map NDC → froxel texel
/// (`view.output_size`) → tile. A uniform NDC→`tile_count` split would
/// disagree wherever the froxel dimension is not a multiple of
/// [`FOG_LIGHT_TILE`] (160×100 → 13 y-tiles covering 12.5), starving whole
/// froxel rows of their lights. `tile_count` must be
/// `output_size.div_ceil(FOG_LIGHT_TILE)`; the final clamp only guards list
/// indexing against a malformed view.
pub fn fog_light_tile_bounds(view: &View, light: &PointLight, tile_count: [u32; 2]) -> [u32; 4] {
    if light.intensity <= 0.0 || light.radius <= 0.0 || tile_count[0] == 0 || tile_count[1] == 0 {
        return FOG_LIGHT_BOUNDS_NONE;
    }
    let camera = Vec3::from_array(view.camera_position);
    let rel = Vec3::from_array(light.position) - camera;
    let z = Vec3::from_array(view.camera_forward).dot(rel);
    let radius = light.radius;
    if z + radius <= 0.0 {
        return FOG_LIGHT_BOUNDS_NONE;
    }
    if z <= radius.max(view.depth_near_plane) {
        return [0, 0, tile_count[0] - 1, tile_count[1] - 1];
    }

    let x = Vec3::from_array(view.camera_right).dot(rel);
    let y = -Vec3::from_array(view.camera_up).dot(rel);
    let denom = z * (z - radius);
    let tan_y = view.tan_half_fov.max(1.0e-6);
    let tan_x = (tan_y * view.aspect).max(1.0e-6);
    let center_x = x / (z * tan_x);
    let center_y = y / (z * tan_y);
    let radius_x = radius * (z + x.abs()) / (denom * tan_x);
    let radius_y = radius * (z + y.abs()) / (denom * tan_y);
    let min_x = center_x - radius_x;
    let max_x = center_x + radius_x;
    let min_y = center_y - radius_y;
    let max_y = center_y + radius_y;
    if max_x < -1.0 || min_x > 1.0 || max_y < -1.0 || min_y > 1.0 {
        return FOG_LIGHT_BOUNDS_NONE;
    }

    let to_tile = |ndc: f32, dim: u32, count: u32| -> u32 {
        let dim = dim.max(1);
        let froxel = (((ndc.clamp(-1.0, 1.0) * 0.5 + 0.5) * dim as f32) as u32).min(dim - 1);
        (froxel / FOG_LIGHT_TILE).min(count - 1)
    };
    [
        to_tile(min_x, view.output_size[0], tile_count[0]),
        to_tile(min_y, view.output_size[1], tile_count[1]),
        to_tile(max_x, view.output_size[0], tile_count[0]),
        to_tile(max_y, view.output_size[1], tile_count[1]),
    ]
}

/// Smooth height tint from `bottom` to `top`.
pub fn height_gradient(
    h: f32,
    bottom: [f32; 3],
    top: [f32; 3],
    offset: f32,
    length: f32,
) -> [f32; 3] {
    let t = saturate((h - offset) / length.max(1.0e-3));
    let t = t * t * (3.0 - 2.0 * t);
    [
        bottom[0] + (top[0] - bottom[0]) * t,
        bottom[1] + (top[1] - bottom[1]) * t,
        bottom[2] + (top[2] - bottom[2]) * t,
    ]
}

/// Integrate one constant-scatter, constant-extinction froxel step.
///
/// `scatter_rgb` is sigma_s * lighting for the step, `sigma_t` is fog
/// extinction per world unit, `splat_od` is already optical depth for the
/// full step, and `throughput` is the prefix transmittance entering it.
pub fn integrate_step(
    scatter_rgb: [f32; 3],
    sigma_t: f32,
    splat_od: f32,
    dz: f32,
    throughput: f32,
) -> ([f32; 3], f32) {
    if dz <= 0.0 || !dz.is_finite() {
        return ([0.0; 3], 1.0);
    }

    let sigma_t = sigma_t.max(0.0);
    let splat_od = splat_od.max(0.0);
    let total_od = (sigma_t * dz + splat_od).max(0.0);
    let step_t = transmittance(total_od);
    let scale = if total_od <= 1.0e-4 {
        dz
    } else {
        (1.0 - step_t) / (sigma_t + splat_od / dz).max(1.0e-6)
    } * throughput;

    (
        [
            scatter_rgb[0] * scale,
            scatter_rgb[1] * scale,
            scatter_rgb[2] * scale,
        ],
        step_t,
    )
}

/// Interleaved gradient noise in `[0, 1)`, one-slice amplitude for fog lookups.
pub fn interleaved_gradient_noise(x: u32, y: u32) -> f32 {
    let p = 0.067_110_56 * x as f32 + 0.005_837_15 * y as f32;
    fract(52.982_918 * fract(p))
}

/// Encode alpha as normalized fixed-point optical depth.
pub fn extinction_encode(alpha: f32) -> f32 {
    saturate((-(1.0 - alpha.min(0.999)).ln()) / EXT_MAX)
}

pub fn extinction_to_u8(x: f32) -> u32 {
    ((saturate(x) * 255.0) + 0.5).floor() as u32
}

/// Decode an extinction byte to optical depth.
pub fn extinction_decode(u8_value: u32) -> f32 {
    (u8_value.min(255) as f32 / 255.0) * EXT_MAX
}

/// Inclusive froxel-texel range whose hardware-linear filter footprint can
/// influence any screen pixel of tile `tile_idx` along one axis.
///
/// A pixel center `p + 0.5` maps to texel coordinate
/// `(p + 0.5) * froxel / screen - 0.5`; the filter reads floor/ceil of that
/// coordinate. The doubled integer form below computes those exact bounds
/// without CPU/GPU float-rounding disagreement. Clamping models the sampler.
pub fn prime_froxel_range(tile_idx: u32, screen: u32, froxel: u32) -> (u32, u32) {
    let screen = screen.max(1);
    let froxel_max = if froxel > 0 { froxel - 1 } else { 0 };
    let p0 = (tile_idx * PRIME_TILE).min(screen - 1);
    let p1 = (p0 + PRIME_TILE).min(screen) - 1;
    let denominator = 2 * screen;

    let lo_numerator = (2 * p0 + 1) * froxel;
    let f0 = if lo_numerator > screen {
        (lo_numerator - screen) / denominator
    } else {
        0
    }
    .min(froxel_max);

    let hi_numerator = (2 * p1 + 1) * froxel;
    let f1 = if hi_numerator > screen {
        (hi_numerator - screen).div_ceil(denominator)
    } else {
        0
    }
    .min(froxel_max);
    (f0, f1)
}

/// Reverse-Z hardware depth of the zero-transmittance cull boundary at
/// fractional slice coordinate `slice` (the saturated slice's far edge plus
/// the caller's sampling margin). Only fragments strictly beyond it fail a
/// GREATER depth test against the primed value.
pub fn prime_quad_depth(curve: &FogCurve, slice: f32, depth_near_plane: f32) -> f32 {
    let z = z_of_warped_slice(curve, slice);
    if z <= 0.0 {
        return 1.0;
    }
    (depth_near_plane / z).clamp(0.0, 1.0)
}

/// Packed RGB extinction uses bits 0..9, 10..19, and 20..29: eight quantum
/// bits plus two carry bits per channel. Packed atomic adds can carry into a
/// neighboring field after 1023; the per-channel overflow slice nevertheless
/// saturates integration from the first overflow.
pub const EXT_RGB_FIELD_MAX: u32 = 0x3FF;

/// Pack per-channel u8 quanta (each <= 255 per splat) into one RGB word.
pub fn ext_rgb_pack(r: u32, g: u32, b: u32) -> u32 {
    r | (g << 10) | (b << 20)
}

/// Extract channel `c`'s 10-bit accumulator.
pub fn ext_rgb_field(word: u32, channel: u32) -> u32 {
    (word >> (channel * 10)) & EXT_RGB_FIELD_MAX
}

/// Decode a 10-bit field on the shared u8 quantum scale — quanta past 255
/// are the carry headroom, so the decoded optical depth may legitimately
/// reach 4x [`EXT_MAX`] (still far under [`FOG_OPTICAL_DEPTH_CAP`]).
pub fn ext_rgb_decode(field: u32) -> f32 {
    (field.min(EXT_RGB_FIELD_MAX) as f32 / 255.0) * EXT_MAX
}

/// Dword index of one froxel column's RGB extinction word for `slice` —
/// slices contiguous per column, matching integrate's walk order.
pub fn ext_rgb_index(x: u32, y: u32, slice: u32, width: u32, slice_count: u32) -> u32 {
    (y * width + x) * slice_count + slice
}

/// Artist-facing tint: per-channel transmittance multiplier in `(0, 1]`
/// to extra optical depth. `1` is exactly neutral (ZII zero `tint_od`);
/// the floor keeps black glass finite (its channel saturates through the
/// overflow path instead).
pub fn tint_od_from_transmittance(t: [f32; 3]) -> [f32; 3] {
    [
        -t[0].clamp(1.0e-4, 1.0).ln(),
        -t[1].clamp(1.0e-4, 1.0).ln(),
        -t[2].clamp(1.0e-4, 1.0).ln(),
    ]
}

/// Linear split for a fractional froxel slice.
pub fn splat_weights(slice: f32, slice_count: u32) -> (u32, f32, u32, f32) {
    let n = slice_count.max(1);
    let base = slice.floor();
    let frac = slice - base;
    let i0 = clamp_slice_index(base as i32, n);
    let i1 = clamp_slice_index(base as i32 + 1, n);
    (i0, 1.0 - frac, i1, frac)
}

pub fn ext_column_dwords(slice_count: u32) -> u32 {
    (slice_count.max(1) + EXT_LANES_PER_DWORD - 1) / EXT_LANES_PER_DWORD
}

/// Packed extinction-buffer address for one froxel slice.
pub fn ext_dword_index(x: u32, y: u32, slice: u32, width: u32, slice_count: u32) -> (u32, u32) {
    let lane = slice % EXT_LANES_PER_DWORD;
    let dword = (y * width + x) * ext_column_dwords(slice_count) + slice / EXT_LANES_PER_DWORD;
    (dword, lane)
}

/// Resolve weighted transparent emission over a background that has already
/// traversed the splatted event extinction in `V_int`.
///
/// Unlike conventional WBOIT, the background is added directly: multiplying
/// it by event transmittance again would double-attenuate opaque radiance and
/// fog in-scatter in front of the transparent events. `accum_neg_log` still
/// supplies exact total event coverage for normalizing the weighted emission.
pub fn oit_resolve(
    accum_rgb: [f32; 3],
    accum_alpha_w: f32,
    accum_neg_log: f32,
    preextinguished_background: [f32; 3],
) -> [f32; 3] {
    let coverage = 1.0 - (-accum_neg_log).exp();
    let scale = coverage / accum_alpha_w.max(1.0e-6);
    [
        accum_rgb[0] * scale + preextinguished_background[0],
        accum_rgb[1] * scale + preextinguished_background[1],
        accum_rgb[2] * scale + preextinguished_background[2],
    ]
}

fn finite_or(x: f32, fallback: f32) -> f32 {
    if x.is_finite() { x } else { fallback }
}

fn fract(x: f32) -> f32 {
    x - x.floor()
}

fn saturate(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}

fn clamp_slice_index(i: i32, slice_count: u32) -> u32 {
    i.clamp(0, slice_count as i32 - 1) as u32
}

fn uniform_optical_depth(sigma: f32, t0: f32, t1: f32) -> f32 {
    if !t1.is_finite() {
        return if sigma > 0.0 {
            FOG_OPTICAL_DEPTH_CAP
        } else {
            0.0
        };
    }
    (sigma * (t1 - t0).max(0.0)).min(FOG_OPTICAL_DEPTH_CAP)
}

fn exponential_optical_depth(
    b: f32,
    dir_y: f32,
    t0: f32,
    t1: f32,
    density: f32,
    falloff: f32,
) -> f32 {
    let y0 = (b + dir_y * t0).max(0.0);
    let e0 = (-falloff * y0).exp();
    if !t1.is_finite() {
        if dir_y > 0.0 {
            return density * e0 / (falloff * dir_y).max(1.0e-12);
        }
        return FOG_OPTICAL_DEPTH_CAP;
    }
    let y1 = (b + dir_y * t1).max(0.0);
    let e1 = (-falloff * y1).exp();
    density * (e0 - e1) / (falloff * dir_y)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, eps: f32) {
        assert!((a - b).abs() <= eps, "{a} != {b} (eps {eps})");
    }

    fn approx_rgb(a: [f32; 3], b: [f32; 3], eps: f32) {
        approx(a[0], b[0], eps);
        approx(a[1], b[1], eps);
        approx(a[2], b[2], eps);
    }

    #[test]
    fn froxel_curve_round_trips() {
        let params = froxel_params_from(240.0, 64, 8.0, 8.0, 500.0);
        for i in 0..=256 {
            let z = params.f * i as f32 / 256.0;
            let s = slice_of_z(&params, z);
            let z2 = z_of_slice(&params, s);
            approx(z, z2, (z.abs() * 2.0e-5).max(2.0e-5));
        }
        for i in 0..=64 {
            let s = i as f32;
            let z = z_of_slice(&params, s);
            let s2 = slice_of_z(&params, z);
            approx(s, s2, 2.0e-5);
        }
    }

    #[test]
    fn froxel_params_clamp_domains() {
        let p = froxel_params_from(0.0, 0, 0.0, -10.0, 0.0);
        approx(p.f, MIN_FAR, 0.0);
        approx(p.a, MIN_LINEARIZATION, 0.0);
        assert_eq!(p.slice_count_u32, 1);
        approx(p.slice_count, 1.0, 0.0);

        let p = froxel_params_from(f32::INFINITY, 64, 4.0, 10.0, 40.0);
        approx(p.f, 40.0, 0.0);

        let p = froxel_params_from(80.0, 64, 4.0, 10.0, 40.0);
        approx(p.f, 40.0, 0.0);
    }

    #[test]
    fn height_fog_matches_quadrature() {
        let cases = [
            // Above the offset, rising.
            (8.0, 0.35, 0.5, 22.0, 0.08, 0.22, 2.0),
            // Crosses the max() kink upward.
            (-4.0, 0.45, 0.0, 30.0, 0.12, 0.18, 1.0),
            // Crosses the max() kink downward.
            (12.0, -0.55, 0.0, 28.0, 0.09, 0.24, 3.0),
            // Degenerate direction.
            (6.0, 1.0e-8, 2.0, 40.0, 0.07, 0.5, 0.0),
            // Uniform fog.
            (4.0, -0.2, 0.0, 50.0, 0.03, 0.0, 1.0),
        ];
        for (h0, dir_y, t0, t1, density, falloff, offset) in cases {
            let analytic = height_fog_optical_depth(h0, dir_y, t0, t1, density, falloff, offset);
            let numeric = quadrature(h0, dir_y, t0, t1, density, falloff, offset, 10_000);
            approx(analytic, numeric, (numeric.abs() * 2.0e-3).max(4.0e-4));
        }
    }

    #[test]
    fn height_fog_infinite_segments_converge_or_cap() {
        let finite = height_fog_optical_depth(5.0, 0.5, 0.0, f32::INFINITY, 0.1, 0.2, 0.0);
        approx(finite, 0.1 * (-1.0f32).exp() / (0.2 * 0.5), 1.0e-5);

        let capped = height_fog_optical_depth(5.0, -0.5, 0.0, f32::INFINITY, 0.1, 0.2, 0.0);
        approx(capped, FOG_OPTICAL_DEPTH_CAP, 0.0);

        let no_density = height_fog_optical_depth(5.0, 0.0, 0.0, f32::INFINITY, 0.0, 0.2, 0.0);
        approx(no_density, 0.0, 0.0);
    }

    #[test]
    fn beer_helper_is_exp_negative_od() {
        approx(transmittance(0.0), 1.0, 0.0);
        approx(transmittance(2.0), (-2.0f32).exp(), 0.0);
    }

    #[test]
    fn hg_phase_isotropic_is_exactly_one() {
        for cos_theta in [-1.0, -0.4, 0.0, 0.25, 1.0] {
            approx(hg_phase(cos_theta, 0.0), 1.0, 0.0);
        }
    }

    #[test]
    fn hg_phase_spherical_mean_is_one() {
        let steps = 200_000u32;
        let dx = 2.0 / steps as f32;
        for g in [-0.5, 0.3, 0.8] {
            let mut sum = 0.0;
            for i in 0..steps {
                let cos_theta = -1.0 + (i as f32 + 0.5) * dx;
                sum += hg_phase(cos_theta, g) * dx;
            }
            approx(0.5 * sum, 1.0, 1.0e-3);
        }
    }

    #[test]
    fn height_gradient_clamps_and_smooth_midpoint() {
        let bottom = [0.1, 0.3, 0.5];
        let top = [0.9, 0.7, 0.1];
        approx_rgb(
            height_gradient(-20.0, bottom, top, 10.0, 20.0),
            bottom,
            1.0e-6,
        );
        approx_rgb(height_gradient(40.0, bottom, top, 10.0, 20.0), top, 1.0e-6);
        approx_rgb(
            height_gradient(20.0, bottom, top, 10.0, 20.0),
            [0.5, 0.5, 0.3],
            1.0e-6,
        );
        approx_rgb(
            height_gradient(10.0, bottom, top, 10.0, 0.0),
            bottom,
            1.0e-6,
        );
        approx_rgb(height_gradient(10.001, bottom, top, 10.0, 0.0), top, 1.0e-6);
    }

    #[test]
    fn integrate_step_matches_constant_medium_riemann_sum() {
        let cases = [
            ([0.2, 0.4, 1.0], 0.05, 0.0, 2.0, 0.75),
            ([1.0, 0.5, 0.25], 0.2, 0.1, 1.25, 0.6),
            ([0.8, 0.3, 0.1], 0.0, 2.5, 0.5, 1.0),
            ([0.7, 0.6, 0.5], 1.0e-8, 0.0, 3.0, 0.9),
            ([0.5, 0.1, 0.9], 0.0, 0.0, 4.0, 0.4),
        ];
        for (scatter, sigma_t, splat_od, dz, throughput) in cases {
            let (added, step_t) = integrate_step(scatter, sigma_t, splat_od, dz, throughput);
            let (expected, expected_t) =
                riemann_step(scatter, sigma_t, splat_od, dz, throughput, 1_000);
            approx_rgb(added, expected, 1.5e-4);
            approx(step_t, expected_t, 1.0e-6);
        }

        let (added, step_t) = integrate_step([1.0, 2.0, 3.0], 0.5, 0.2, 0.0, 0.75);
        approx_rgb(added, [0.0; 3], 0.0);
        approx(step_t, 1.0, 0.0);
    }

    #[test]
    fn extinction_fixed_point_round_trips_to_one_quantum() {
        let quantum = EXT_MAX / 255.0;
        for i in 0..=999 {
            let alpha = i as f32 * 0.999 / 999.0;
            let od = (-(1.0 - alpha).ln()).min(EXT_MAX);
            let encoded = extinction_encode(alpha);
            let decoded = extinction_decode(extinction_to_u8(encoded));
            approx(decoded, od, quantum);
        }
    }

    #[test]
    fn local_light_bounds_are_conservative_and_zii() {
        let view = View {
            camera_position: [0.0; 3],
            tan_half_fov: 0.5,
            camera_forward: Vec3::Z.to_array(),
            aspect: 2.0,
            camera_right: Vec3::X.to_array(),
            depth_near_plane: 0.1,
            camera_up: Vec3::Y.to_array(),
            _pad: 0,
            output_size: [320, 200],
            _pad2: [0; 2],
        };
        assert_eq!(
            fog_light_tile_bounds(&view, &PointLight::default(), [40, 25]),
            FOG_LIGHT_BOUNDS_NONE
        );
        let center = PointLight {
            position: [0.0, 0.0, 10.0],
            radius: 2.0,
            color: [1.0; 3],
            intensity: 1.0,
        };
        let b = fog_light_tile_bounds(&view, &center, [40, 25]);
        assert!(b[0] <= 20 && b[2] >= 20 && b[1] <= 12 && b[3] >= 12);

        let camera_crossing = PointLight {
            position: [0.0, 0.0, 0.5],
            radius: 1.0,
            ..center
        };
        assert_eq!(
            fog_light_tile_bounds(&view, &camera_crossing, [40, 25]),
            [0, 0, 39, 24]
        );
        let behind = PointLight {
            position: [0.0, 0.0, -3.0],
            radius: 1.0,
            ..center
        };
        assert_eq!(
            fog_light_tile_bounds(&view, &behind, [40, 25]),
            FOG_LIGHT_BOUNDS_NONE
        );
    }

    /// The culler must agree with `fog_light`'s `froxel / FOG_LIGHT_TILE`
    /// list lookup on grids that don't divide by the tile edge (100 rows →
    /// 13 tiles covering 12.5). A light sitting ON a froxel column's center
    /// ray must always bin into the tile that column reads — the uniform
    /// NDC→tile split this replaces starved 20 of 100 rows.
    #[test]
    fn local_light_bounds_cover_every_consumer_tile_on_nondivisible_grids() {
        let view = View {
            camera_position: [0.0; 3],
            tan_half_fov: 0.5,
            camera_forward: Vec3::Z.to_array(),
            aspect: 1.6,
            camera_right: Vec3::X.to_array(),
            depth_near_plane: 0.1,
            camera_up: Vec3::Y.to_array(),
            _pad: 0,
            output_size: [160, 100],
            _pad2: [0; 2],
        };
        let tiles = [
            160u32.div_ceil(FOG_LIGHT_TILE),
            100u32.div_ceil(FOG_LIGHT_TILE),
        ];
        assert_eq!(tiles, [20, 13]);
        for fy in 0..100u32 {
            for fx in 0..160u32 {
                let dir = abi_core::ray_direction(&view, glam::UVec2::new(fx, fy));
                let pos = dir * (20.0 / dir.z);
                let light = PointLight {
                    position: pos.to_array(),
                    radius: 0.05,
                    color: [1.0; 3],
                    intensity: 1.0,
                };
                let b = fog_light_tile_bounds(&view, &light, tiles);
                let (tx, ty) = (fx / FOG_LIGHT_TILE, fy / FOG_LIGHT_TILE);
                assert!(
                    b[0] <= tx && tx <= b[2] && b[1] <= ty && ty <= b[3],
                    "froxel ({fx}, {fy}): consumer tile ({tx}, {ty}) outside culled {b:?}"
                );
            }
        }
    }

    #[test]
    fn local_fog_radiance_reduces_to_incident_for_isotropic_clear_air() {
        let light = PointLight {
            position: [0.0, 0.0, 4.0],
            radius: 8.0,
            color: [1.0, 0.5, 0.25],
            intensity: 2.0,
        };
        let got = fog_point_light_radiance(Vec3::Z, Vec3::ZERO, &light, 0.0, 0.0, 0.0, 0.0);
        let want = point_light_radiance(Vec3::ZERO, &light);
        assert!((got - want).length() <= 1.0e-6, "{got:?} != {want:?}");
    }

    #[test]
    fn prime_tile_footprint_is_conservative_and_bounded() {
        // A 4:1 screen-to-froxel ratio shares edge texels.
        assert_eq!(prime_froxel_range(0, 1280, 320), (0, 4));
        assert_eq!(prime_froxel_range(1, 1280, 320), (3, 8));
        let (lo, hi) = prime_froxel_range(79, 1280, 320);
        assert!(lo <= hi && hi < 320);

        // Tests render below froxel resolution: pixel centers 0..15 touch
        // froxels 1..62; the final partial tile touches 257..318. The edge
        // sampler clamps before texels 0/319 enter either footprint.
        assert_eq!(prime_froxel_range(0, 80, 320), (1, 62));
        assert_eq!(prime_froxel_range(4, 80, 320), (257, 318));
    }

    #[test]
    fn prime_depth_moves_farther_as_slice_increases() {
        let p = froxel_params_from(96.0, 64, 4.0, 96.0, 96.0);
        let curve = fog_curve_from(p, &[0; FOG_SLICE_MAX as usize], 0.0, 0.0);
        let near = prime_quad_depth(&curve, 12.5, 0.1);
        let far = prime_quad_depth(&curve, 30.5, 0.1);
        assert!(near > far && far > 0.0, "reverse-Z depths: {near}, {far}");
        approx(near, 0.1 / z_of_slice(&p, 12.5), 0.0);
        approx(transmittance(EXT_MAX), ZERO_TRANS_EPS, 2.0e-7);
    }

    /// Zero gain, an empty histogram, or bound <= 1 must reproduce the raw
    /// log curve BIT-exactly — the identity LUT stores exact integers and
    /// `i + (x - i)` roundtrips f32 in [0, 64].
    #[test]
    fn warp_identity_is_bit_exact() {
        let p = froxel_params_from(96.0, 64, 4.0, 96.0, 96.0);
        let curves = [
            fog_curve_from(p, &[0; FOG_SLICE_MAX as usize], 1.0, 4.0),
            fog_curve_from(p, &[7; FOG_SLICE_MAX as usize], 0.0, 4.0),
            fog_curve_from(p, &[7; FOG_SLICE_MAX as usize], 1.0, 1.0),
        ];
        for curve in &curves {
            for i in 0..=64u32 {
                assert_eq!(curve.warp[i as usize].to_bits(), (i as f32).to_bits());
                assert_eq!(curve.unwarp[i as usize].to_bits(), (i as f32).to_bits());
            }
            for z in [0.0, 0.013, 1.7, 24.9, 95.0, 300.0] {
                assert_eq!(
                    warped_slice_of_z(curve, z).to_bits(),
                    slice_of_z(&p, z).to_bits()
                );
            }
            for s in [0.0, 0.5, 12.25, 63.99, 64.0] {
                assert_eq!(
                    z_of_warped_slice(curve, s).to_bits(),
                    z_of_slice(&p, s).to_bits()
                );
            }
        }
    }

    /// A concentrated histogram must widen its slices' z-coverage share up
    /// to — and never past — the bound, keep the LUT monotone with pinned
    /// endpoints, and invert consistently.
    #[test]
    fn warp_is_bounded_monotone_and_invertible() {
        let p = froxel_params_from(96.0, 64, 4.0, 96.0, 96.0);
        let mut hist = [0u32; FOG_SLICE_MAX as usize];
        for h in hist[40..44].iter_mut() {
            *h = 1000;
        }
        let bound = 4.0;
        let curve = fog_curve_from(p, &hist, 8.0, bound);

        assert_eq!(curve.warp[0].to_bits(), 0.0f32.to_bits());
        approx(curve.warp[64], 64.0, 1.0e-3);
        let slack = 1.0e-3;
        for i in 0..64 {
            let width = curve.warp[i + 1] - curve.warp[i];
            assert!(
                width >= 1.0 / bound - slack && width <= bound + slack,
                "slice {i} width {width} escapes [{}, {bound}]",
                1.0 / bound
            );
            assert!(width > 0.0, "monotonicity broke at slice {i}");
        }
        // The hot slices actually won volume resolution.
        let hot = curve.warp[44] - curve.warp[40];
        assert!(hot > 4.0 * 2.0, "hot region got {hot} slices, expected > 8");

        // Roundtrip through both LUTs.
        let mut s = 0.0f32;
        while s <= 64.0 {
            let raw = warp_eval(&curve.unwarp, 64, warp_eval(&curve.warp, 64, s));
            approx(raw, s, 2.0e-2);
            s += 0.37;
        }
        // And through z: warped slice -> z -> warped slice.
        for slice in [0.5, 20.0, 41.5, 63.5] {
            let z = z_of_warped_slice(&curve, slice);
            approx(warped_slice_of_z(&curve, z), slice, 2.0e-2);
        }
    }

    /// The RGB word is three independent 10-bit accumulators on the u8
    /// quantum scale: pack/field roundtrip, carry headroom to 1023, decode
    /// parity with the monochrome scale, and the overflow boundary.
    #[test]
    fn ext_rgb_word_roundtrips_and_carries() {
        let word = ext_rgb_pack(255, 7, 1023);
        assert_eq!(ext_rgb_field(word, 0), 255);
        assert_eq!(ext_rgb_field(word, 1), 7);
        assert_eq!(ext_rgb_field(word, 2), 1023);
        // Repeated adds accumulate into the carry bits without touching
        // the neighbor until a field crosses 1023.
        let mut w = 0u32;
        for _ in 0..4 {
            w += ext_rgb_pack(250, 0, 3);
        }
        assert_eq!(ext_rgb_field(w, 0), 1000);
        assert_eq!(ext_rgb_field(w, 1), 0);
        assert_eq!(ext_rgb_field(w, 2), 12);
        // Decode shares the monochrome quantum scale and extends past it.
        for q in [0u32, 1, 128, 255] {
            approx(ext_rgb_decode(q), extinction_decode(q), 0.0);
        }
        approx(ext_rgb_decode(1023), EXT_MAX * (1023.0 / 255.0), 1.0e-4);
        // Normalize optical depth before quantizing extinction.
        for od in [0.05f32, 0.51, 1.386, 2.526] {
            let q = extinction_to_u8(od / EXT_MAX);
            approx(ext_rgb_decode(q), od, EXT_MAX / 255.0 * 0.51);
        }
        // The overflow predicate the splat uses.
        assert!(ext_rgb_field(w, 0) + 24 > EXT_RGB_FIELD_MAX);
        assert!(ext_rgb_field(w, 0) + 23 <= EXT_RGB_FIELD_MAX);
    }

    #[test]
    fn tint_od_neutral_and_finite() {
        assert_eq!(tint_od_from_transmittance([1.0; 3]), [0.0; 3]);
        let od = tint_od_from_transmittance([0.25, 1.0, 0.0]);
        approx(od[0], 4.0f32.ln(), 1.0e-6);
        assert_eq!(od[1], 0.0);
        assert!(od[2].is_finite() && od[2] > 9.0);
        // ZII: the zero tint_od is exactly the monochrome particle.
        assert_eq!(OitParticle::default().tint_od, [0.0; 3]);
    }

    #[test]
    fn splat_weights_sum_and_clamp() {
        let cases = [
            (-0.75, (0, 0)),
            (3.25, (3, 4)),
            (7.9, (7, 7)),
            (20.5, (7, 7)),
        ];
        for (slice, expected_indices) in cases {
            let (i0, w0, i1, w1) = splat_weights(slice, 8);
            assert_eq!((i0, i1), expected_indices);
            approx(w0 + w1, 1.0, 0.0);
            assert!((0.0..=1.0).contains(&w0));
            assert!((0.0..=1.0).contains(&w1));
        }
    }

    #[test]
    fn extinction_dword_index_is_bijective_by_lane() {
        let width = 3;
        let height = 2;
        let slices = 8;
        let words = ext_column_dwords(slices);
        let mut seen = [false; 48];
        for y in 0..height {
            for x in 0..width {
                for slice in 0..slices {
                    let (dword, lane) = ext_dword_index(x, y, slice, width, slices);
                    assert_eq!(lane, slice % 4);
                    assert!(dword < width * height * words);
                    let bit = (dword * 4 + lane) as usize;
                    assert!(!seen[bit], "duplicate index for ({x}, {y}, {slice})");
                    seen[bit] = true;
                }
            }
        }
        assert!(seen.into_iter().all(core::convert::identity));

        let (dword, lane) = ext_dword_index(1, 1, 5, width, slices);
        assert_eq!(lane, 1);
        assert_eq!(dword, ((width + 1) * words) + 1);
    }

    #[test]
    fn oit_resolve_matches_exact_painter_composite() {
        #[derive(Clone, Copy)]
        struct Layer {
            color: [f32; 3],
            alpha: f32,
            depth: f32,
        }

        let mut layers = [
            Layer {
                color: [0.9, 0.1, 0.2],
                alpha: 0.22,
                depth: 7.0,
            },
            Layer {
                color: [0.1, 0.8, 0.3],
                alpha: 0.45,
                depth: 2.0,
            },
            Layer {
                color: [0.2, 0.3, 1.0],
                alpha: 0.31,
                depth: 5.0,
            },
            Layer {
                color: [1.0, 0.8, 0.1],
                alpha: 0.18,
                depth: 11.0,
            },
        ];
        layers.sort_by(|a, b| a.depth.partial_cmp(&b.depth).unwrap());

        let mut accum_rgb = [0.0; 3];
        let mut accum_alpha_w = 0.0;
        let mut accum_neg_log = 0.0;
        let mut w = 1.0;
        for layer in layers.iter().copied() {
            accum_rgb[0] += layer.color[0] * layer.alpha * w;
            accum_rgb[1] += layer.color[1] * layer.alpha * w;
            accum_rgb[2] += layer.color[2] * layer.alpha * w;
            accum_alpha_w += layer.alpha * w;
            accum_neg_log += -(1.0 - layer.alpha).ln();
            w *= 1.0 - layer.alpha;
        }

        let background = [0.04, 0.08, 0.14];
        let preextinguished_background = [background[0] * w, background[1] * w, background[2] * w];
        let resolved = oit_resolve(
            accum_rgb,
            accum_alpha_w,
            accum_neg_log,
            preextinguished_background,
        );

        let mut painter = background;
        for layer in layers.into_iter().rev() {
            painter[0] = layer.color[0] * layer.alpha + painter[0] * (1.0 - layer.alpha);
            painter[1] = layer.color[1] * layer.alpha + painter[1] * (1.0 - layer.alpha);
            painter[2] = layer.color[2] * layer.alpha + painter[2] * (1.0 - layer.alpha);
        }

        approx(accum_alpha_w, 1.0 - w, 1.0e-6);
        approx_rgb(resolved, painter, 1.0e-6);
    }

    fn riemann_step(
        scatter_rgb: [f32; 3],
        sigma_t: f32,
        splat_od: f32,
        dz: f32,
        throughput: f32,
        steps: u32,
    ) -> ([f32; 3], f32) {
        if dz <= 0.0 {
            return ([0.0; 3], 1.0);
        }
        let extinction = sigma_t.max(0.0) + splat_od.max(0.0) / dz;
        let dt = dz / steps as f32;
        let mut added = [0.0; 3];
        for i in 0..steps {
            let t = (i as f32 + 0.5) * dt;
            let sample_t = (-extinction * t).exp() * throughput;
            added[0] += scatter_rgb[0] * sample_t * dt;
            added[1] += scatter_rgb[1] * sample_t * dt;
            added[2] += scatter_rgb[2] * sample_t * dt;
        }
        (added, (-extinction * dz).exp())
    }

    fn quadrature(
        h0: f32,
        dir_y: f32,
        t0: f32,
        t1: f32,
        density: f32,
        falloff: f32,
        height_offset: f32,
        steps: u32,
    ) -> f32 {
        let dt = (t1 - t0) / steps as f32;
        let mut sum = 0.0;
        for i in 0..steps {
            let t = t0 + (i as f32 + 0.5) * dt;
            let h = (h0 + dir_y * t - height_offset).max(0.0);
            sum += density.max(0.0) * (-falloff.max(0.0) * h).exp() * dt;
        }
        sum.min(FOG_OPTICAL_DEPTH_CAP)
    }
}
