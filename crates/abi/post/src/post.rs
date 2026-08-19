//! Post-processing vocabulary.
//!
//! The split with `shaders/lib`: SAMPLING is machine-specific (the fragment
//! shader taps the bindless heaps, the CPU reference taps a slice), but the
//! tap pattern and every combination applied to the samples live here, once,
//! compiled for both machines. A hardware test that compares the GPU image
//! against these same functions is testing the sampling seam and nothing
//! else — the math cannot drift because there is only one copy of it.
//!
//! Struct fields stay `[f32; N]` (the layout law); math speaks glam.

use crate::gpu_data;
use glam::{UVec2, Vec2, Vec3};
// Scalar `.powf()` (the sRGB curve) is std-only; on the GPU it comes from
// the num_traits::Float shim spirv-std re-exports.
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

/// The 13-tap downsample pattern from Call of Duty: Advanced Warfare
/// (Jimenez), in pixel offsets around the destination pixel's source
/// position:
///
/// ```text
///     4 - 5 - 6
///     | 0 - 1 |
///     7 | 8 | 9
///     | 2 - 3 |
///    10 - 11 - 12
/// ```
pub const BLOOM_TAPS: usize = 13;
pub const BLOOM_COORDS: [Vec2; BLOOM_TAPS] = [
    Vec2::new(-1.0, 1.0),
    Vec2::new(1.0, 1.0),
    Vec2::new(-1.0, -1.0),
    Vec2::new(1.0, -1.0),
    Vec2::new(-2.0, 2.0),
    Vec2::new(0.0, 2.0),
    Vec2::new(2.0, 2.0),
    Vec2::new(-2.0, 0.0),
    Vec2::new(0.0, 0.0),
    Vec2::new(2.0, 0.0),
    Vec2::new(-2.0, -2.0),
    Vec2::new(0.0, -2.0),
    Vec2::new(2.0, -2.0),
];

/// Jimenez's improved weights for the 13-tap pattern (sum = 1.0).
pub const BLOOM_WEIGHTS: [f32; BLOOM_TAPS] = [
    0.125, 0.125, 0.125, 0.125, // inner 2x2
    0.03125, 0.0625, 0.03125, // top row
    0.0625, 0.125, 0.0625, // middle row
    0.03125, 0.0625, 0.03125, // bottom row
];

pub fn max_brightness(c: Vec3) -> f32 {
    c.x.max(c.y).max(c.z)
}

/// Karis's luma-weighted average of a 2×2 block: weights inversely
/// proportional to brightness, so a single very bright pixel (a "firefly")
/// cannot dominate the average and flicker across frames.
pub fn karis_average(s1: Vec3, s2: Vec3, s3: Vec3, s4: Vec3) -> Vec3 {
    let w1 = 1.0 / (max_brightness(s1) + 1.0);
    let w2 = 1.0 / (max_brightness(s2) + 1.0);
    let w3 = 1.0 / (max_brightness(s3) + 1.0);
    let w4 = 1.0 / (max_brightness(s4) + 1.0);
    (s1 * w1 + s2 * w2 + s3 * w3 + s4 * w4) / (w1 + w2 + w3 + w4)
}

/// Partial Karis averaging over the 13 taps: five overlapping 2×2 blocks
/// (see the pattern diagram), each Karis-averaged, combined with Jimenez's
/// 0.5 / 4×0.125 weights. The anti-flicker first pass of the bloom chain.
pub fn bloom_average_partial(s: &[Vec3; BLOOM_TAPS]) -> Vec3 {
    let center = karis_average(s[0], s[1], s[2], s[3]);
    let top_left = karis_average(s[4], s[5], s[8], s[7]);
    let top_right = karis_average(s[5], s[6], s[9], s[8]);
    let bottom_left = karis_average(s[7], s[8], s[11], s[10]);
    let bottom_right = karis_average(s[8], s[9], s[12], s[11]);
    0.5 * center + 0.125 * (top_left + top_right + bottom_left + bottom_right)
}

/// Plain weighted 13-tap combination — every bloom pass after the first.
pub fn bloom_weighted_sum(s: &[Vec3; BLOOM_TAPS]) -> Vec3 {
    let mut result = Vec3::ZERO;
    let mut i = 0;
    while i < BLOOM_TAPS {
        result += BLOOM_WEIGHTS[i] * s[i];
        i += 1;
    }
    result
}

/// Soft-knee brightness threshold (Catlike Coding / bevy's quadratic
/// curve): smoothly attenuates colors below the threshold instead of
/// hard-clipping.
pub fn soft_threshold(color: Vec3, threshold: f32, knee: f32) -> Vec3 {
    let brightness = max_brightness(color);
    let mut soft = brightness - (threshold - knee);
    soft = soft.clamp(0.0, 2.0 * knee);
    soft = soft * soft / (4.0 * knee + 0.00001);
    let contribution = soft.max(brightness - threshold) / brightness.max(0.00001);
    color * contribution
}

/// Clamp to a safe HDR range to keep inf/nan out of the chain.
pub fn safe_hdr(color: Vec3) -> Vec3 {
    color.min(Vec3::splat(65000.0))
}

/// The 9-tap tent filter for the bloom upsample, in pixel offsets around
/// the destination pixel: a 3×3 bilinear-weighted pattern that smooths
/// the 2× upscale.
pub const TENT_TAPS: usize = 9;
pub const TENT_COORDS: [Vec2; TENT_TAPS] = [
    Vec2::new(-1.0, 1.0),
    Vec2::new(0.0, 1.0),
    Vec2::new(1.0, 1.0),
    Vec2::new(-1.0, 0.0),
    Vec2::new(0.0, 0.0),
    Vec2::new(1.0, 0.0),
    Vec2::new(-1.0, -1.0),
    Vec2::new(0.0, -1.0),
    Vec2::new(1.0, -1.0),
];

/// Tent weights (sum = 1.0): the outer product of [1, 2, 1]/4.
pub const TENT_WEIGHTS: [f32; TENT_TAPS] = [
    0.0625, 0.125, 0.0625, //
    0.125, 0.25, 0.125, //
    0.0625, 0.125, 0.0625,
];

/// Weighted 9-tap tent combination — the upsample's filter.
pub fn bloom_tent_sum(s: &[Vec3; TENT_TAPS]) -> Vec3 {
    let mut result = Vec3::ZERO;
    let mut i = 0;
    while i < TENT_TAPS {
        result += TENT_WEIGHTS[i] * s[i];
        i += 1;
    }
    result
}

/// The upsample's combination step: blend the current-resolution
/// downsample with the tent-filtered previous (smaller) level, then clamp.
/// `blend_factor` controls accumulated lower-resolution bloom.
pub fn bloom_upsample_blend(current: Vec3, tent_previous: Vec3, blend_factor: f32) -> Vec3 {
    safe_hdr(current.lerp(tent_previous, blend_factor))
}

/// Fragment data for `bloom_downsample`.
///
/// Device-address data uses C layout without std140 padding.
#[gpu_data]
pub struct BloomDownsampleData {
    /// Bindless index of the source texture (sampled heap, set 0).
    pub src_texture_id: u32,
    /// Bindless index of the sampler (sampler heap, set 2).
    pub src_sampler_id: u32,
    /// 1 / source dimensions.
    pub pixel_size: [f32; 2],
    /// Nonzero on the first pass: Karis partial averaging to kill fireflies.
    pub use_anti_flicker: u32,
    /// Brightness cutoff (0 = disabled).
    pub bloom_threshold: f32,
    /// Soft transition width (0 = hard).
    pub bloom_knee: f32,
    /// Kernel stretch per axis (1,1 = uniform; 4,1 = anamorphic).
    pub bloom_scale: [f32; 2],
}

/// Fragment data for `bloom_upsample`.
#[gpu_data]
pub struct BloomUpsampleData {
    /// Downsample chain at the DESTINATION resolution (sampled heap).
    pub downsample_texture_id: u32,
    /// Previous (half-size) upsample result — or the smallest downsample
    /// on the first step.
    pub previous_texture_id: u32,
    pub sampler_id: u32,
    /// lerp(current, tent(previous), blend_factor).
    pub blend_factor: f32,
    /// 1 / DESTINATION dimensions: the tent spreads at output resolution
    /// (the downsample spreads at source resolution — they differ).
    pub pixel_size: [f32; 2],
    /// Kernel stretch per axis (1,1 = uniform; 4,1 = anamorphic).
    pub bloom_scale: [f32; 2],
}

// ── Tony McMapface ──────────────────────────────────────────────────────
//
// The display transform (Tomasz Stachowiak): Reinhard-style stimulus
// encode into a 48³ LUT stored as a 2304×48 2D strip (48 B-slices side by
// side; hardware bilinear covers R/G, the B axis is two taps + a lerp).
// The strip rides in `assets/luts/tony_mc_mapface_2304x48_rgba16f.bin`
// (rgba16f, top-down rows — row 47 of slice 0 is tony(0,0,0) = black,
// hence the V flip in the taps).

pub const TONY_LUT_DIMS: f32 = 48.0;
pub const TONY_LUT_WIDTH: u32 = 48 * 48;
pub const TONY_LUT_HEIGHT: u32 = 48;

/// HDR → [0,1) stimulus for the LUT.
pub fn tony_encode(color: Vec3) -> Vec3 {
    color / (color + 1.0)
}

/// Two-dimensional strip coordinates for a Tony lookup.
#[derive(Copy, Clone, Default)]
pub struct TonyTaps {
    pub uv_low: Vec2,
    pub uv_high: Vec2,
    pub b_frac: f32,
}

pub fn tony_taps(encoded: Vec3) -> TonyTaps {
    let c = encoded.clamp(Vec3::ZERO, Vec3::ONE) * (TONY_LUT_DIMS - 1.0);
    let b_low = c.floor().z; // glam (libm) floor: f32::floor is std-only.
    let b_high = (b_low + 1.0).min(TONY_LUT_DIMS - 1.0);
    let u_low = (b_low * TONY_LUT_DIMS + c.x + 0.5) / (TONY_LUT_DIMS * TONY_LUT_DIMS);
    let u_high = (b_high * TONY_LUT_DIMS + c.x + 0.5) / (TONY_LUT_DIMS * TONY_LUT_DIMS);
    let v = 1.0 - (c.y + 0.5) / TONY_LUT_DIMS;
    TonyTaps {
        uv_low: Vec2::new(u_low, v),
        uv_high: Vec2::new(u_high, v),
        b_frac: c.z - b_low,
    }
}

// Lateral chromatic aberration uses opposite radial red and blue offsets.

/// The per-pixel CA displacement: red samples at `uv + ca_offset`, blue at
/// `uv - ca_offset`, green at `uv`. `strength` is the red channel's UV
/// shift at the image CORNER (red↔blue separation there is 2·strength) —
/// the dial reads directly as "fringe width as a fraction of the screen",
/// resolution-independent. Positive pushes red outward (the common lens
/// look); negative flips the fringe.
pub fn ca_offset(uv: Vec2, strength: f32) -> Vec2 {
    let d = uv - Vec2::splat(0.5);
    // r²-growth along d; at the corner |d| = √½ and r² = ½, so the 2√2
    // normalizer makes |offset| == strength exactly there.
    d * (d.length_squared() * 2.0 * core::f32::consts::SQRT_2 * strength)
}

/// Fragment data for `aberration_frag`.
#[gpu_data]
pub struct AberrationData {
    /// Source image (sampled heap) — any sampled source; typically the
    /// scene HDR, ahead of bloom.
    pub input_texture_id: u32,
    pub sampler_id: u32,
    /// Red-channel UV shift at the image corner (see [`ca_offset`]).
    pub strength: f32,
}

// Lens remapping blends rectilinear, cylindrical, and equidistant projections.

/// Maps output positions to rectilinear source UVs.
///
/// `field_scale == 1` preserves the rendered field. Larger values may produce
/// out-of-range UVs; smaller values zoom in.
pub fn lens_source_uv(
    p: Vec2,
    aspect: f32,
    field_scale: f32,
    tan_half_fov_src: f32,
    cylindrical: f32,
    fisheye: f32,
) -> Vec2 {
    let t_v = tan_half_fov_src;
    let t_h = t_v * aspect;
    // Bound source half-angles away from cosine singularities.
    let theta_h = t_h.atan();
    let theta_corner = (t_h * t_h + t_v * t_v).sqrt().atan();

    // Rectilinear: the identity. Included as the blend's third endpoint.
    let rect = p;

    // Cylindrical: longitude linear in screen x, so a world vertical — which
    // is a line of constant longitude — stays a straight vertical column.
    // The 1/cos λ on y is not a fudge: in a rectilinear image a direction at
    // (longitude λ, latitude φ) lands at (tan λ, tan φ / cos λ), so undoing
    // that is what keeps the two axes consistent.
    let lon = p.x * theta_h * field_scale;
    let cyl = Vec2::new(lon.tan() / t_h, p.y / lon.cos());

    // Equidistant: angle linear in radius. Scaled by the CORNER angle, not
    // the vertical one — that is what makes the corners land exactly on the
    // source's corner rays instead of demanding a field that was never
    // rendered. r == 0 is the single degenerate direction and maps to center.
    let s = Vec2::new(p.x * aspect, p.y);
    let r = s.length();
    let r_max = (aspect * aspect + 1.0).sqrt();
    let fish = if r > 1e-6 {
        let theta = ((r / r_max) * theta_corner * field_scale).min(1.5533);
        let m = theta.tan() / t_v;
        let d = (s / r) * m;
        Vec2::new(d.x / aspect, d.y)
    } else {
        Vec2::ZERO
    };

    let mixed = rect + (cyl - rect) * cylindrical;
    let mixed = mixed + (fish - mixed) * fisheye;
    // To UV: recenter and flip y (UV grows downward).
    Vec2::new(mixed.x, -mixed.y) * 0.5 + Vec2::splat(0.5)
}

/// The incoming ray's angle from the optical axis for an output pixel, used
/// by the vignette. Keyed to the same corner-referenced geometry as
/// [`lens_source_uv`], so the falloff tracks the rays the remap moved.
pub fn lens_ray_angle(p: Vec2, aspect: f32, tan_half_fov_src: f32, field_scale: f32) -> f32 {
    let t_v = tan_half_fov_src;
    let t_h = t_v * aspect;
    let theta_corner = (t_h * t_h + t_v * t_v).sqrt().atan();
    let s = Vec2::new(p.x * aspect, p.y);
    let r_max = (aspect * aspect + 1.0).sqrt();
    ((s.length() / r_max) * theta_corner * field_scale).min(1.5533)
}

/// Blends from no vignette (`1`) toward powered cosine-fourth falloff.
/// `amount` is an un-clamped blend weight; `power` is clamped to 0.01 before
/// exponentiation, and `1` preserves the physical cos⁴ response.
pub fn lens_vignette(theta: f32, amount: f32, power: f32) -> f32 {
    let c = theta.cos().max(0.0);
    let natural = (c * c * c * c).powf(power.max(0.01));
    1.0 + (natural - 1.0) * amount
}

/// Spectral weight for CA tap `i` of `taps`, as linear RGB.
///
/// Spectral weights integrate to neutral white for full-period taps.
pub fn ca_spectral_weight(i: u32, taps: u32) -> Vec3 {
    let t = (i as f32 + 0.5) / (taps.max(1) as f32);
    let tau = core::f32::consts::TAU;
    Vec3::new(
        0.5 + 0.5 * (tau * t).cos(),
        0.5 + 0.5 * (tau * (t - 1.0 / 3.0)).cos(),
        0.5 + 0.5 * (tau * (t - 2.0 / 3.0)).cos(),
    )
}

/// Where tap `i` of `taps` sits along the CA displacement, in [-1, 1].
/// Short wavelengths land on one side of green, long on the other.
pub fn ca_tap_offset(i: u32, taps: u32) -> f32 {
    if taps <= 1 {
        return 0.0;
    }
    ((i as f32 + 0.5) / (taps as f32)) * 2.0 - 1.0
}

/// Fragment data for `lens_frag`: the projection remap, spectral chromatic
/// aberration, and the natural vignette, in one resample. All three are
/// radial properties of the same optical axis and belong at the same point
/// in the chain, and folding them means the warped image is sampled once
/// instead of once per effect — every extra full-screen resample after a
/// warp compounds softening.
#[gpu_data]
pub struct LensData {
    /// Source image (sampled heap): the tonemapped, display-space frame.
    pub input_texture_id: u32,
    pub sampler_id: u32,
    /// tan of the source render's VERTICAL half-FOV.
    pub tan_half_fov_src: f32,
    /// 1.0 presents exactly the rendered field, redistributed. Above 1 asks
    /// for rays that were never rendered (black surround); below 1 zooms in.
    pub field_scale: f32,
    /// Output width / height.
    pub aspect: f32,
    /// 0 = rectilinear, 1 = cylindrical (verticals stay straight).
    pub cylindrical: f32,
    /// 0 = off, 1 = equidistant fisheye (everything bends).
    pub fisheye: f32,
    /// Red-channel UV shift at the image corner (see [`ca_offset`]).
    pub ca_strength: f32,
    /// Spectral taps along the displacement. 0 disables CA entirely; even
    /// counts integrate to neutral white (see [`ca_spectral_weight`]).
    pub ca_taps: u32,
    /// Vignette blend toward the physical cos⁴ falloff.
    pub vignette: f32,
    /// Exponent on the falloff, for taste. 1.0 is the physical law.
    pub vignette_power: f32,
    /// Impact shake, in `p`-space ([-1, 1] across the frame). This offsets the
    /// **vignette mask only** — the image stays locked underneath it.
    ///
    /// Only the vignette mask moves; the image remains registered.
    pub shake: [f32; 2],
}

const _: () = assert!(core::mem::size_of::<LensData>() == 52);

// Sensor effects run in HDR before display transformation.

/// Rec. 709 luma weights, for deciding how dark a pixel is.
pub const LUMA_709: Vec3 = Vec3::new(0.2126, 0.7152, 0.0722);

/// Three decorrelated values in [0, 1) for a pixel, from an integer hash.
/// Integer-domain hashing (rather than the usual `sin`-and-fract trick)
/// because it has no periodicity to alias against a pixel grid, and because
/// the GPU and the CPU reference must agree bit-for-bit for a test to mean
/// anything.
pub fn sensor_noise(px: UVec2, salt: u32) -> Vec3 {
    Vec3::new(
        hash_to_unit(hash_u32(px.x ^ hash_u32(px.y ^ hash_u32(salt)))),
        hash_to_unit(hash_u32(
            px.x ^ hash_u32(px.y ^ hash_u32(salt ^ 0x9E37_79B9)),
        )),
        hash_to_unit(hash_u32(
            px.x ^ hash_u32(px.y ^ hash_u32(salt ^ 0x85EB_CA6B)),
        )),
    )
}

/// Splits noise into monochrome and chromatic components.
pub fn noise_split(n: Vec3) -> (f32, Vec3) {
    let centered = n - Vec3::splat(0.5);
    let mono = centered.dot(Vec3::ONE) / 3.0;
    (mono, centered - Vec3::splat(mono))
}

/// How strongly noise shows at this brightness. Read noise is a roughly
/// constant number of electrons, so its *relative* size grows as the signal
/// falls — hence a falloff in the signal, not a curve chosen by eye.
/// `bias` sharpens the concentration into shadow.
pub fn sensor_shadow_weight(luma: f32, bias: f32) -> f32 {
    (1.0 / (1.0 + luma.max(0.0) * 4.0)).powf(bias.max(0.01))
}

/// The readout shear. Rows are exposed in sequence, so a camera rotating
/// during readout lays each row down at a different heading and the image
/// leans. `yaw_rate` is radians/second.
///
/// **Soft-saturated, not linear.** A mouse flick produces an unbounded yaw
/// rate, and a linear map turns that into unbounded displacement — which is
/// where the nausea lives, and no amount of lowering `strength` fixes it,
/// because the worst case is set by the fastest flick rather than by the
/// dial. `max_shear` is a hard ceiling in UV that the response approaches
/// asymptotically, so the effect is expressive at ordinary turn rates and
/// simply cannot exceed a known bound at absurd ones.
pub fn rolling_shutter_uv(uv: Vec2, yaw_rate: f32, strength: f32, max_shear: f32) -> Vec2 {
    let x = yaw_rate * strength;
    let saturated = x / (1.0 + x.abs());
    Vec2::new(uv.x + (uv.y - 0.5) * saturated * max_shear, uv.y)
}

// ── Noise distributions ─────────────────────────────────────────────────
//
// Ported from the collection at fragcoord.xyz/s/pxmcvnpc (MIT, © 2026
// @lumiey), which credits the originals: interleaved gradient noise is
// Jimenez's (Next Generation Post Processing in Call of Duty: Advanced
// Warfare), and the blue-noise high-pass follows the shadertoy the
// collection cites.
//
// The DISTRIBUTION is what's borrowed; the hash underneath stays asha's own
// `hash_u32`. Those two choices are separable, and the bit-cast float hashes
// the collection uses have weak spots on exact integers — precisely the input
// domain a pixel index lives in.

/// Interleaved gradient noise. Low-discrepancy in space, and still so in time
/// when the coordinate is scrolled linearly per frame.
///
/// Use this distribution for dithering and stochastic sampling.
pub fn ign(p: Vec2) -> f32 {
    let x = 0.067_110_56 * p.x + 0.005_837_15 * p.y;
    let f = x - libm_floor(x);
    let v = 52.982_918_9 * f;
    v - libm_floor(v)
}

/// IGN scrolled for frame `frame`. The offset is the magic constant from
/// Jimenez's temporal variant; the modulo keeps the coordinate small enough
/// that f32 precision never starts eating the low bits.
pub fn ign_temporal(p: Vec2, frame: u32) -> f32 {
    ign(p + Vec2::splat(5.588_238 * (frame % 64) as f32))
}

/// Three decorrelated IGN samples, for per-channel noise. Offsets are
/// arbitrary but fixed and mutually irrational-ish, so the channels do not
/// correlate into grey.
pub fn ign_triple(p: Vec2, frame: u32) -> Vec3 {
    Vec3::new(
        ign_temporal(p, frame),
        ign_temporal(p + Vec2::new(37.0, 17.0), frame),
        ign_temporal(p + Vec2::new(11.0, 71.0), frame),
    )
}

/// Blue-noise-ish: high-pass a white field by pushing each sample away from
/// the mean of its eight neighbours. Blue noise reads as fine and evenly
/// spread where white noise clumps into visible clusters, which is what makes
/// it the right choice for a *static* pattern the eye has time to study.
pub fn blue_noise(px: UVec2) -> f32 {
    let center = hash_to_unit(hash_u32(px.x ^ hash_u32(px.y)));
    let mut sum = 0.0;
    let mut k = 0i32;
    while k < 9 {
        if k != 4 {
            let ox = (k % 3 - 1) as i32;
            let oy = (k / 3 - 1) as i32;
            let nx = px.x.wrapping_add(ox as u32);
            let ny = px.y.wrapping_add(oy as u32);
            sum += hash_to_unit(hash_u32(nx ^ hash_u32(ny)));
        }
        k += 1;
    }
    (1.125 * center - sum / 8.0) * 0.9 + 0.5
}

/// The grain's own pixel grid, independent of output resolution.
///
/// Indexing noise by the real framebuffer pixel makes grain size a function
/// of window size — the same dial produces fine grain at 4K and coarse grain
/// at 1080p, so it can never be tuned once. Quantizing to a virtual grid
/// instead fixes the apparent size across resolutions, and `cells_high`
/// doubles as the chunkiness dial: fewer cells, chunkier grain.
pub fn grain_cell(uv: Vec2, aspect: f32, cells_high: f32) -> Vec2 {
    let cells = Vec2::new(cells_high * aspect, cells_high);
    let scaled = uv * cells;
    Vec2::new(libm_floor(scaled.x), libm_floor(scaled.y))
}

/// `floor` for both machines: std on the host, the spirv-std shim on the GPU.
fn libm_floor(x: f32) -> f32 {
    x.floor()
}

/// Unsharp cross, in pixels, for the over-sharpen tap pattern.
pub const SHARPEN_TAPS: [Vec2; 4] = [
    Vec2::new(-1.0, 0.0),
    Vec2::new(1.0, 0.0),
    Vec2::new(0.0, -1.0),
    Vec2::new(0.0, 1.0),
];

/// Unsharp mask: push the center away from the mean of its neighbours. Left
/// deliberately un-clamped — the bright rim overshoot IS the cheap-camera
/// look, and clamping it away is what makes sharpening read as "crisp"
/// instead of "processed".
pub fn sharpen_combine(center: Vec3, neighbour_sum: Vec3, amount: f32) -> Vec3 {
    center + (center - neighbour_sum * 0.25) * amount
}

/// Fragment data for `sensor_frag`.
#[gpu_data]
pub struct SensorData {
    /// Source image (sampled heap), scene-referred HDR.
    pub input_texture_id: u32,
    pub sampler_id: u32,
    /// Target resolution in pixels, for pixel-space taps and noise indexing.
    pub resolution: [f32; 2],
    /// Animation salt. Must change per frame or the grain reads as dirt on
    /// the monitor instead of noise in the signal.
    pub frame: u32,
    /// Monochrome read-noise amplitude, in scene-referred units.
    pub grain_luma: f32,
    /// Chromatic read-noise amplitude — the colored shadow speckle. Usually
    /// wants to be LARGER than `grain_luma`, which is the opposite of what
    /// film-grain intuition suggests.
    pub grain_chroma: f32,
    /// Exponent concentrating noise into shadow.
    pub grain_shadow_bias: f32,
    /// Non-animating component: hot pixels and column offsets. Small, and
    /// the reason the noise belongs to a specific sensor.
    pub grain_fixed: f32,
    /// Unsharp amount. 0 disables the four extra taps.
    pub sharpen: f32,
    /// Rolling-shutter response scale (dimensionless; feeds the saturation).
    pub shutter: f32,
    /// Hard ceiling on the shear, in UV. The response saturates toward this,
    /// so a mouse flick cannot exceed it.
    pub shutter_max: f32,
    /// Camera yaw rate, radians/second.
    pub yaw_rate: f32,
    /// Grain cells across the frame HEIGHT. Fixes apparent grain size across
    /// output resolutions, and doubles as the chunkiness dial.
    pub grain_cells: f32,
    /// Output width / height, so the grain grid stays square.
    pub aspect: f32,
}

const _: () = assert!(core::mem::size_of::<SensorData>() == 60);

// Transmission effects run after display transformation.

/// Radial weight, 0 at the optical axis and 1 at the frame corner. It masks
/// tear displacement/hold, chroma desync, and dropout; roll and its seam are
/// unmasked, while snow retains a small center floor.
pub fn transmission_radial(uv: Vec2) -> f32 {
    let d = (uv - Vec2::splat(0.5)) * 2.0;
    (d.length() * core::f32::consts::FRAC_1_SQRT_2).min(1.0)
}

/// Snow: the noise floor of a weak carrier. **Luma only** — deliberately
/// achromatic, which is what keeps it distinct from the sensor's chromatic
/// shadow speckle. Two noise layers of different character read as two
/// different points in the hardware chain; two layers of the same character
/// just read as "noisy".
pub fn snow(px: UVec2, frame: u32) -> f32 {
    hash_to_unit(hash_u32(px.x ^ hash_u32(px.y ^ hash_u32(frame)))) - 0.5
}

/// A slow wander in [0, 1] for modulating snow over time, so the link's
/// quality drifts instead of sitting at a constant. Value-noise: hashed
/// integers with a smoothstep between them, which is coherent rather than
/// jittery.
pub fn signal_wander(t: f32) -> f32 {
    let i = libm_floor(t);
    let f = t - i;
    let f = f * f * (3.0 - 2.0 * f);
    let a = hash_to_unit(hash_u32(i as i32 as u32));
    let b = hash_to_unit(hash_u32((i as i32 as u32).wrapping_add(1)));
    a + (b - a) * f
}

/// Chroma smear taps, trailing in one direction.
///
/// Composite video carries color at a fraction of luma's bandwidth, so color
/// lags and spreads horizontally while edges stay sharp. The **asymmetry** is
/// the whole tell: a symmetric blur reads as "out of focus", a one-sided
/// smear reads as signal. `i` runs 0..taps; the offset is always the same
/// sign.
pub fn chroma_tap_offset(i: u32, taps: u32, width: f32) -> f32 {
    if taps <= 1 {
        return 0.0;
    }
    -width * (i as f32 / (taps - 1) as f32)
}

/// Recombine a smeared color with a sharp luma.
///
/// The cheap route, and identical in effect to a YCbCr round-trip: take the
/// smeared sample's *chromaticity* and rescale it to the center sample's
/// luminance. Color spreads, detail does not.
///
/// Written as an explicit chromaticity — a color divided by its own luma, so
/// luma 1 by construction — rather than as `smeared * (sharp / smeared_luma)`.
/// Those are algebraically equal and numerically not: the naive form is an
/// unbounded gain when the smeared sample is near black, which flares to white
/// wherever the offset crosses a hard edge into shadow. A black sample carries
/// no hue at all, so `w` fades it toward neutral instead of dividing by
/// nothing. Luma comes out exactly `sharp_luma` either way, since both ends of
/// the blend have luma 1.
pub fn chroma_recombine(smeared: Vec3, sharp_luma: f32) -> Vec3 {
    let smeared_luma = smeared.dot(LUMA_709);
    let w = (smeared_luma / 0.02).min(1.0);
    let chromaticity = smeared / smeared_luma.max(1e-4);
    (Vec3::ONE + (chromaticity - Vec3::ONE) * w) * sharp_luma
}

/// Fractional part, for wrapping a coordinate into [0, 1). `f32::rem_euclid`
/// lives in std, which this crate cannot have.
fn wrap01(x: f32) -> f32 {
    x - libm_floor(x)
}

/// Map a raw impulse to the impulse set's drive: **zero below the gate**, then
/// rising to 1.
///
/// Thresholding keeps small impulses visually inactive.
pub fn shock_gate(shock: f32, gate: f32) -> f32 {
    if shock <= gate {
        0.0
    } else {
        ((shock - gate) / (1.0 - gate).max(1e-3)).min(1.0)
    }
}

/// One horizontal tear band's contribution at a given row.
pub struct Tear {
    /// Lateral shift in UV, before radial weighting.
    pub offset: f32,
    /// Source row to read instead of this one.
    pub hold_y: f32,
}

/// Computes a hashed horizontal sync-loss band.
pub fn tear_band(
    uv_y: f32,
    radial: f32,
    drive: f32,
    salt: u32,
    bands: f32,
    max_offset: f32,
) -> Tear {
    let quiet = Tear {
        offset: 0.0,
        hold_y: uv_y,
    };
    // Zero offset disables both shifting and holding.
    if drive <= 0.0 || bands < 1.0 || max_offset <= 0.0 {
        return quiet;
    }
    let row = libm_floor(uv_y * bands);
    let h = hash_u32((row as i32 as u32) ^ hash_u32(salt));
    // Drive controls the fraction of affected rows.
    if hash_to_unit(h) > drive * 0.6 {
        return quiet;
    }
    let shift = hash_to_unit(hash_u32(h ^ 0x9e37_79b9));
    let hold = hash_to_unit(hash_u32(h ^ 0x85eb_ca6b));
    Tear {
        offset: (shift - 0.5) * 2.0 * max_offset * drive * radial,
        // Radial weighting keeps the optical axis unchanged.
        hold_y: uv_y + (row / bands - uv_y) * hold * drive * radial,
    }
}

/// Wraps UV vertically for the roll effect.
pub fn roll_uv(uv: Vec2, roll: f32) -> Vec2 {
    if roll == 0.0 {
        return uv;
    }
    Vec2::new(uv.x, wrap01(uv.y + roll))
}

/// Computes the bright seam accompanying a vertical roll.
pub fn roll_seam(uv_y: f32, roll: f32, width: f32) -> f32 {
    if roll == 0.0 || width <= 0.0 {
        return 0.0;
    }
    // The wrap lands where the source coordinate crosses zero.
    let seam = wrap01(-roll);
    let d = (uv_y - seam).abs();
    let d = d.min(1.0 - d);
    let t = (1.0 - d / width).max(0.0);
    t * t
}

/// Tape dropout as a **comet**: a hard bright head with a horizontal tail
/// decaying to its right.
///
/// Real dropout is not a uniform bar. The head loses contact with the tape, and
/// when it regains it the automatic gain control overshoots and recovers over
/// the following microseconds — which is a bright spike followed by a fading
/// trail. It costs the same as a flat bar, and it is what gives the
/// bright-weighting a physical reason to exist instead of being an arbitrary
/// bias. Returns a signed additive amount in display units.
pub fn dropout(uv: Vec2, drive: f32, salt: u32, rows: f32, length: f32, gain: f32) -> f32 {
    if drive <= 0.0 || rows < 1.0 || length <= 0.0 || gain == 0.0 {
        return 0.0;
    }
    let row = libm_floor(uv.y * rows);
    let h = hash_u32((row as i32 as u32) ^ hash_u32(salt ^ 0x2545_f491));
    if hash_to_unit(h) > drive * 0.25 {
        return 0.0;
    }
    let head = hash_to_unit(hash_u32(h ^ 0x27d4_eb2f));
    let len = length * (0.3 + 0.7 * hash_to_unit(hash_u32(h ^ 0x1656_67b1)));
    let d = uv.x - head;
    if d < 0.0 || d > len {
        return 0.0;
    }
    let t = d / len;
    // Quadratic recovery, plus a hard spike at the head itself.
    let env = (1.0 - t) * (1.0 - t) + if t < 0.05 { 0.6 } else { 0.0 };
    // Emphasize the dropout overshoot.
    let sign = if hash_to_unit(hash_u32(h ^ 0x7feb_352d)) < 0.8 {
        1.0
    } else {
        -1.0
    };
    sign * env * gain * drive
}

/// Fragment data for `transmission_frag`.
#[gpu_data]
pub struct TransmissionData {
    /// Display-referred source (sampled heap): the tonemap's output.
    pub input_texture_id: u32,
    pub sampler_id: u32,
    pub resolution: [f32; 2],
    pub frame: u32,
    /// Snow amplitude at rest, in display units.
    pub snow_base: f32,
    /// Extra snow proportional to camera acceleration.
    pub snow_accel: f32,
    /// Camera acceleration magnitude, units/s².
    pub accel: f32,
    /// Depth of the slow quality wander, as a fraction of the base.
    pub snow_wander: f32,
    /// Wander phase (seconds scaled by a rate on the host).
    pub wander_t: f32,
    /// Chroma smear width in UV. 0 disables the extra taps.
    pub chroma_width: f32,
    /// Taps along the smear.
    pub chroma_taps: u32,
    /// Final output dither step (1/255 for 8-bit, zero disables). This pass
    /// quantizes once at the true output; if it is skipped, tonemap owns the
    /// dither on its float output.
    pub dither_step: f32,

    // Radial masking applies to tear displacement/hold, chroma desync, and
    // dropout; roll and its seam are unmasked, and snow keeps a center floor.
    /// The gated shock, 0..1. Zero means the impulse set is entirely absent.
    pub drive: f32,
    /// Tear bands across the frame height; the band height is its reciprocal.
    pub tear_bands: f32,
    /// Peak tear shift in UV, at `drive == 1` and the frame edge.
    pub tear_offset: f32,
    /// Dropout candidate rows across the frame height.
    pub dropout_rows: f32,
    /// Comet length in UV, and its peak amplitude in display units. The bare
    /// envelope peaks at 1.6, which is a full-scale bar — `gain` is what makes
    /// it a dial rather than a constant.
    pub dropout_length: f32,
    pub dropout_gain: f32,
    /// Vertical roll offset in UV. Host-computed, and on its own higher gate:
    /// this is the one artifact no radial mask can contain.
    pub roll: f32,
    /// Seam half-width in UV, and its brightness.
    pub seam_width: f32,
    pub seam_gain: f32,
    /// Chroma-plane offset at unit drive.
    pub chroma_desync: f32,
    /// Extra snow per unit drive: the burst that ties the impulse set to the
    /// continuous floor instead of leaving them two unrelated effect families.
    pub snow_shock: f32,
}

const _: () = assert!(core::mem::size_of::<TransmissionData>() == 96);

// Auto exposure samples luminance and changes in discrete steps.

/// Taps in the probe's sparse grid. 16×9 keeps the sample pattern matched to a
/// widescreen frame's proportions, so no region is systematically over-weighted.
pub const EXPOSURE_PROBE_TAPS: u32 = 144;
const EXPOSURE_PROBE_COLS: u32 = 16;

/// Tap `i`'s position, on a half-offset grid so no sample lands on an exact
/// frame edge or the dead centre.
pub fn exposure_probe_uv(i: u32) -> Vec2 {
    let rows = EXPOSURE_PROBE_TAPS / EXPOSURE_PROBE_COLS;
    let x = i % EXPOSURE_PROBE_COLS;
    let y = i / EXPOSURE_PROBE_COLS;
    Vec2::new(
        (x as f32 + 0.5) / EXPOSURE_PROBE_COLS as f32,
        (y as f32 + 0.5) / rows as f32,
    )
}

/// Quantizes exposure stops with hysteresis.
pub fn exposure_quantize(stops: f32, step: f32, current: f32, margin: f32) -> f32 {
    if step <= 0.0 {
        return stops;
    }
    let want = stops / step;
    let held = current / step;
    if (want - held).abs() < 0.5 + margin.clamp(0.0, 0.49) {
        current
    } else {
        libm_floor(want + 0.5) * step
    }
}

/// Returns one epsilon-clamped log-luminance sample.
pub fn exposure_log_luma(color: Vec3) -> f32 {
    (color.max(Vec3::ZERO).dot(LUMA_709) + 1e-4).ln()
}

/// Fragment data for `exposure_probe_frag`.
#[gpu_data]
pub struct ExposureProbeData {
    /// The scene-referred HDR frame (sampled heap).
    pub input_texture_id: u32,
    pub sampler_id: u32,
}

const _: () = assert!(core::mem::size_of::<ExposureProbeData>() == 8);

/// Fragment data for the `tony_frag` presentation pass.
#[gpu_data]
pub struct TonemapTonyData {
    /// HDR scene (sampled heap).
    pub hdr_texture_id: u32,
    pub hdr_sampler_id: u32,
    /// The tony strip (sampled heap) + its clamp-linear sampler.
    pub lut_texture_id: u32,
    pub lut_sampler_id: u32,
    /// Output-space dither amplitude; zero disables dithering.
    pub dither_strength: f32,
    /// Scene-referred exposure multiplier; one is neutral.
    pub exposure: f32,
    /// Final bloom texture; zero selects the no-bloom sentinel.
    pub bloom_texture_id: u32,
    /// Bloom multiplier applied before exposure.
    pub bloom_intensity: f32,
}

// Preserve the established tonemap data size.
const _: () = assert!(core::mem::size_of::<TonemapTonyData>() == 32);

// Feedback combines current input with a reprojected, decayed accumulator.

/// Camera basis for the feedback reprojection — the pinhole vocabulary of
/// [`abi_core::ray_direction`] (orthonormal basis + tan-half-fov), not a
/// matrix: directions never see translation, so rotation-only reprojection
/// is structural, not extracted.
#[gpu_data]
pub struct FeedbackCamera {
    pub forward: [f32; 3],
    pub tan_half_fov: f32,
    pub right: [f32; 3],
    pub aspect: f32,
    pub up: [f32; 3],
    pub _pad: u32,
}

/// Unnormalized view ray through a UV — the continuous twin of
/// [`abi_core::ray_direction`] (same +v-down / up-points-up flip).
/// The unnormalized ray is sufficient for projection.
pub fn feedback_ray(cam: &FeedbackCamera, uv: Vec2) -> Vec3 {
    let ndc_x = uv.x * 2.0 - 1.0;
    let ndc_y = -(uv.y * 2.0 - 1.0);
    Vec3::from_array(cam.forward)
        + Vec3::from_array(cam.right) * (ndc_x * cam.tan_half_fov * cam.aspect)
        + Vec3::from_array(cam.up) * (ndc_y * cam.tan_half_fov)
}

/// Reprojects a pixel into the previous camera, then returns
/// `uv + (uv - reproj) * flow`: `-1` samples the reprojected content, `0`
/// leaves UV passive, and positive values exaggerate the motion. Directions
/// on or behind the previous image plane fall back to the passive `uv`.
pub fn feedback_flow_uv(curr: &FeedbackCamera, prev: &FeedbackCamera, uv: Vec2, flow: f32) -> Vec2 {
    let d = feedback_ray(curr, uv);
    let z = d.dot(Vec3::from_array(prev.forward));
    if z <= 1.0e-4 {
        return uv;
    }
    let ndc_x = d.dot(Vec3::from_array(prev.right)) / (z * prev.tan_half_fov * prev.aspect);
    let ndc_y = d.dot(Vec3::from_array(prev.up)) / (z * prev.tan_half_fov);
    let reproj = Vec2::new((ndc_x + 1.0) * 0.5, (-ndc_y + 1.0) * 0.5);
    uv + (uv - reproj) * flow
}

/// Combines fresh input with decayed, drained history.
pub fn feedback_combine(input: Vec3, history: Vec3, decay: f32, floor: f32) -> Vec3 {
    safe_hdr(input.max((history * decay - Vec3::splat(floor)).max(Vec3::ZERO)))
}

/// Fragment data for `feedback_frag`.
#[gpu_data]
pub struct FeedbackData {
    /// Fresh input (sampled heap) — any HDR source; typically the bloom
    /// chain's final level.
    pub input_texture_id: u32,
    /// The OTHER ping-pong accumulator, written last frame.
    pub history_texture_id: u32,
    pub sampler_id: u32,
    /// 0 on the first record after (re)build: the history texture is
    /// undefined memory and must not be sampled (not even multiplied by
    /// zero — garbage fp16 can be NaN, and NaN·0 stays NaN).
    pub sample_history: u32,
    /// exp(−rate·dt), CPU-computed this frame.
    pub decay: f32,
    /// Linear drain this frame (floor-per-second · dt).
    pub floor: f32,
    /// Signed camera-flow scale (see [`feedback_flow_uv`]).
    pub flow: f32,
    pub _pad: u32,
    pub curr: FeedbackCamera,
    pub prev: FeedbackCamera,
}

const _: () = assert!(core::mem::size_of::<FeedbackData>() == 128);

/// Linear → sRGB encode, the piecewise standard curve. The 8-bit swapchain
/// is UNORM in sRGB colorspace: quantizing LINEAR values there puts 4-7%
/// luminance between adjacent codes in dark regions (measured: the low-sun
/// sky banding). Encoding first makes the codes perceptually uniform —
/// Hillaire's reference (PostProcess.hlsl) does the same.
pub fn srgb_encode(c: Vec3) -> Vec3 {
    let f = |x: f32| {
        let x = x.clamp(0.0, 1.0);
        if x <= 0.003_130_8 {
            12.92 * x
        } else {
            1.055 * x.powf(1.0 / 2.4) - 0.055
        }
    };
    Vec3::new(f(c.x), f(c.y), f(c.z))
}

/// lowbias32 (Chris Wellons) — the integer hash under the output dither and
/// the sensor's read noise. Public because both want it and the math must
/// exist in exactly one copy.
pub fn hash_u32(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^= x >> 16;
    x
}

/// Triangular-PDF output dither for one pixel, in [-1, 1): two hashed
/// uniforms summed, so quantization error decorrelates from the signal
/// (plain uniform dither leaves visible noise modulation). DETERMINISTIC
/// per pixel — no frame term — so the CPU reference reproduces the GPU
/// byte-for-byte and verify stays exact. One value for all three channels
/// (chroma-safe; per-channel noise reads as color speckle).
/// Top 24 bits of a hash to [0, 1) — the low bits are the weakest.
pub fn hash_to_unit(x: u32) -> f32 {
    (x >> 8) as f32 * (1.0 / 16_777_216.0)
}

pub fn dither_tri(pixel: UVec2) -> f32 {
    let h1 = hash_u32(pixel.x ^ hash_u32(pixel.y));
    let h2 = hash_u32(h1);
    let u1 = (h1 >> 8) as f32 * (1.0 / 16_777_216.0);
    let u2 = (h2 >> 8) as f32 * (1.0 / 16_777_216.0);
    u1 + u2 - 1.0
}

#[cfg(all(test, not(target_arch = "spirv")))]
mod tests {
    use super::*;

    #[test]
    fn weights_sum_to_one() {
        let sum: f32 = BLOOM_WEIGHTS.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "13-tap weights sum to {sum}");
        let tent: f32 = TENT_WEIGHTS.iter().sum();
        assert!((tent - 1.0).abs() < 1e-6, "tent weights sum to {tent}");
    }

    #[test]
    fn upsample_blend_endpoints() {
        let a = Vec3::splat(1.0);
        let b = Vec3::splat(3.0);
        assert_eq!(bloom_upsample_blend(a, b, 0.0), a);
        assert_eq!(bloom_upsample_blend(a, b, 1.0), b);
        // A uniform field is a tent fixed point.
        let uniform = [Vec3::splat(2.0); TENT_TAPS];
        assert!(
            (bloom_tent_sum(&uniform) - Vec3::splat(2.0))
                .abs()
                .max_element()
                < 1e-6
        );
    }

    #[test]
    fn karis_tames_fireflies() {
        let dim = Vec3::splat(0.1);
        let firefly = Vec3::splat(1000.0);
        let plain = (dim * 3.0 + firefly) / 4.0;
        let karis = karis_average(dim, dim, dim, firefly);
        assert!(karis.x < plain.x / 50.0, "karis {karis} vs plain {plain}");
        // And with uniform inputs it IS the plain average.
        let uniform = karis_average(dim, dim, dim, dim);
        assert!((uniform - dim).abs().max_element() < 1e-6);
    }

    #[test]
    fn tony_taps_geometry() {
        // Black: slice 0, bottom row (the V flip), left column.
        let t = tony_taps(tony_encode(Vec3::ZERO));
        assert!((t.uv_low.x - 0.5 / 2304.0).abs() < 1e-6, "{}", t.uv_low.x);
        assert!(
            (t.uv_low.y - (1.0 - 0.5 / 48.0)).abs() < 1e-6,
            "{}",
            t.uv_low.y
        );
        assert_eq!(t.b_frac, 0.0);
        // Very bright white: approaches the top-right texel of slice 47.
        let t = tony_taps(tony_encode(Vec3::splat(1.0e6)));
        assert!(t.uv_high.x > (2304.0 - 1.0) / 2304.0, "{}", t.uv_high.x);
        assert!(t.uv_high.y < 0.5 / 48.0 + 1e-4, "{}", t.uv_high.y);
        // Encode is monotone and bounded.
        assert!(tony_encode(Vec3::splat(4.0)).x > tony_encode(Vec3::splat(1.0)).x);
        // Saturates AT 1.0 in f32 for huge inputs (1e9 + 1 rounds to 1e9);
        // the taps clamp, so the top texel is the correct destination.
        assert!(tony_encode(Vec3::splat(1.0e9)).max_element() <= 1.0);
    }

    fn feedback_cam(forward: Vec3, right: Vec3, up: Vec3) -> FeedbackCamera {
        FeedbackCamera {
            forward: forward.to_array(),
            tan_half_fov: 0.55,
            right: right.to_array(),
            aspect: 16.0 / 9.0,
            up: up.to_array(),
            _pad: 0,
        }
    }

    #[test]
    fn feedback_flow_identity_camera_is_fixed_point() {
        // A stationary camera reprojects every uv to itself, so the flow
        // scale must be inert at ANY magnitude.
        let cam = feedback_cam(Vec3::NEG_Z, Vec3::X, Vec3::Y);
        for uv in [
            Vec2::new(0.5, 0.5),
            Vec2::new(0.1, 0.8),
            Vec2::new(0.9, 0.05),
        ] {
            for flow in [-1.0, 0.0, 3.0] {
                let got = feedback_flow_uv(&cam, &cam, uv, flow);
                assert!((got - uv).abs().max_element() < 1e-6, "{uv} → {got}");
            }
        }
    }

    #[test]
    fn feedback_flow_rotation_and_endpoints() {
        let curr = feedback_cam(Vec3::NEG_Z, Vec3::X, Vec3::Y);
        // Previous camera yawed a few degrees: its forward tilted toward −X.
        let a = 0.05f32;
        let prev = feedback_cam(
            Vec3::new(-a.sin(), 0.0, -a.cos()),
            Vec3::new(a.cos(), 0.0, -a.sin()),
            Vec3::Y,
        );
        let uv = Vec2::new(0.5, 0.5);
        // flow 0 is exactly passive regardless of motion.
        assert_eq!(feedback_flow_uv(&curr, &prev, uv, 0.0), uv);
        // flow −1 is the pure reprojection: the current center direction
        // (−Z) sits to the RIGHT of the yawed-left previous camera's center
        // (dot(−Z, prev_right) = sin a > 0), same row.
        let stab = feedback_flow_uv(&curr, &prev, uv, -1.0);
        assert!(stab.x > uv.x + 1e-4, "{stab}");
        assert!((stab.y - uv.y).abs() < 1e-6, "{stab}");
        // Hand-checked magnitude: ndc_x = tan(a)/(t·aspect).
        let want_x = 0.5 + 0.5 * a.tan() / (0.55 * (16.0 / 9.0));
        assert!((stab.x - want_x).abs() < 1e-5, "{} vs {want_x}", stab.x);
        // Positive flow pushes the OTHER way, scaled.
        let melt = feedback_flow_uv(&curr, &prev, uv, 2.0);
        assert!((melt.x - (uv.x - 2.0 * (stab.x - uv.x))).abs() < 1e-6);
        // A direction behind the previous camera cannot reproject: passive.
        let behind = feedback_cam(Vec3::Z, Vec3::NEG_X, Vec3::Y);
        assert_eq!(feedback_flow_uv(&curr, &behind, uv, -1.0), uv);
    }

    #[test]
    fn feedback_combine_max_and_drain() {
        let bright = Vec3::splat(4.0);
        let dim = Vec3::splat(0.5);
        // Static bloom passes through untouched (the max keeps current).
        assert_eq!(feedback_combine(bright, bright, 0.9, 0.0), bright);
        // History wins only while its decayed value exceeds the input.
        let trail = feedback_combine(dim, bright, 0.5, 0.0);
        assert_eq!(trail, Vec3::splat(2.0));
        // The linear drain kills near-zero dregs outright, never negative.
        let dreg = feedback_combine(Vec3::ZERO, Vec3::splat(0.05), 1.0, 0.1);
        assert_eq!(dreg, Vec3::ZERO);
        // Per-channel, not luminance: a red trail survives a green input.
        let mixed = feedback_combine(Vec3::new(0.0, 3.0, 0.0), Vec3::new(2.0, 0.0, 0.0), 1.0, 0.0);
        assert_eq!(mixed, Vec3::new(2.0, 3.0, 0.0));
    }

    #[test]
    fn ca_offset_geometry() {
        let s = 0.01;
        // The screen center is exactly zero at any strength.
        assert_eq!(ca_offset(Vec2::splat(0.5), 0.25), Vec2::ZERO);
        // Corners: |offset| == strength (the dial's definition), pointing
        // outward along the diagonal.
        for corner in [
            Vec2::ZERO,
            Vec2::ONE,
            Vec2::new(1.0, 0.0),
            Vec2::new(0.0, 1.0),
        ] {
            let o = ca_offset(corner, s);
            assert!((o.length() - s).abs() < 1e-8, "{corner} → {o}");
            assert!(o.dot(corner - Vec2::splat(0.5)) > 0.0, "must point outward");
        }
        // r³ total growth: halving the radius (same direction) shrinks the
        // shift 8× — |d| halves AND r² quarters.
        let mid = Vec2::new(0.75, 0.75);
        assert!((ca_offset(mid, s).length() - s / 8.0).abs() < 1e-8);
        // Antisymmetric about the center; negative strength flips exactly.
        let uv = Vec2::new(0.9, 0.3);
        let mirrored = Vec2::ONE - uv;
        assert!(
            (ca_offset(uv, s) + ca_offset(mirrored, s))
                .abs()
                .max_element()
                < 1e-8
        );
        assert_eq!(ca_offset(uv, -s), -ca_offset(uv, s));
        // Zero strength is inert everywhere.
        assert_eq!(ca_offset(uv, 0.0), Vec2::ZERO);
    }

    #[test]
    fn soft_threshold_endpoints() {
        // Far below threshold: fully attenuated. Far above: passes through.
        let below = soft_threshold(Vec3::splat(0.01), 1.0, 0.1);
        assert!(
            below.max_element() < 1e-3,
            "below-threshold leaked: {below}"
        );
        let above = soft_threshold(Vec3::splat(10.0), 1.0, 0.1);
        assert!(
            (above.x - 9.0).abs() < 0.01,
            "above-threshold wrong: {above}"
        );
    }
}
