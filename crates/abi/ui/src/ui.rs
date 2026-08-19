//! UI quad GPU contracts and the SDF/color math both machines share.
//!
//! One vertex type serves the whole UI pipeline family: solid rounded rects,
//! per-edge borders, linear gradients, and textured (icon) quads all flow
//! through `UiVertex` in a single CPU-sorted painter's-order stream. The SDF
//! and color-mixing functions are ported faithfully from bevy_ui_render 0.19
//! (`ui.wgsl`, `gradient.wgsl`, `color_operations.wgsl`) and live here — not
//! in the shader crate — so CPU tests exercise the exact code the GPU runs.
//!
//! ## Geometry contract
//!
//! The paint walker emits **4 vertices per quad** in corner order
//! **TL, TR, BR, BL**, positions in physical pixels with the node's affine
//! transform already applied. There is no index buffer: draws are non-indexed
//! triangle lists of `6 * quad_count` vertices, and the vertex shader derives
//! `quad = vertex_index / 6`, `corner = UI_QUAD_TRIANGLES[vertex_index % 6]`.
//! The triangle pattern is (tl, bl, br), (tl, br, tr) — CCW front faces in
//! Vulkan's +y-down NDC, per the engine winding law.
//!
//! ## Gradient encoding
//!
//! Upstream carries gradient start/dir/lengths per vertex and computes the
//! interpolation parameter per fragment. Because `t = dot(p - start, dir)`
//! normalized is affine in position, we compute it **per corner on the CPU**
//! and let the rasterizer interpolate: under `UI_FLAG_GRADIENT`, `uv.x` IS
//! the unclamped segment parameter (may be <0 or >1 — the FILL flags decide),
//! `color`/`color2` are the stop pair. Gradient colors are authored in the
//! interpolation space (sRGB, or HSL under `UI_FLAG_GRADIENT_SPACE_HSLA`,
//! hue normalized 0..1) and converted to linear *after* mixing, exactly as
//! upstream. Gradient stop hints are not encoded: the default hint (0.5)
//! makes upstream's reshaping the identity, and feathers authors no others —
//! if hints are ever needed, extend the ABI then.
//!
//! Non-gradient colors are linear RGBA, straight (non-premultiplied) alpha.
//!
//! ## Material encoding
//!
//! `flags` bits 16–19 select the material mode. Material parameters remain
//! per-quad: `UiMaterialData` is packed into `UiVertex::color2` as
//! `[fixed_channel, variant as f32, 0, 0]`. This avoids per-draw pointers and
//! preserves batching. Material modes do not combine with gradients or
//! textures.

use crate::{GpuPtr, gpu_data};
use glam::{Vec2, Vec3, Vec4};

// Scalar `.powf()` (the sRGB curve) is std-only; on the GPU it comes from
// the num_traits::Float shim spirv-std re-exports.
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

/// Sample the bindless texture at `tex_slot` and multiply into the color.
pub const UI_FLAG_TEXTURED: u32 = 1 << 0;
/// Mix `color`..`color2` by the interpolated `uv.x` parameter.
pub const UI_FLAG_GRADIENT: u32 = 1 << 1;
/// Gradient stops are HSL(A) (hue 0..1); mix in HSL, else mix in sRGB.
pub const UI_FLAG_GRADIENT_SPACE_HSLA: u32 = 1 << 2;
/// Gradient parameter < 0 clamps to the start color instead of transparent.
pub const UI_FLAG_FILL_START: u32 = 1 << 3;
/// Gradient parameter > 1 clamps to the end color instead of transparent.
pub const UI_FLAG_FILL_END: u32 = 1 << 4;
/// Paint outside the shape instead of inside (upstream INVERT).
pub const UI_FLAG_INVERT: u32 = 1 << 5;
/// Per-edge border bits. Values match upstream ui.wgsl for cross-reference.
pub const UI_FLAG_BORDER_LEFT: u32 = 256;
pub const UI_FLAG_BORDER_TOP: u32 = 512;
pub const UI_FLAG_BORDER_RIGHT: u32 = 1024;
pub const UI_FLAG_BORDER_BOTTOM: u32 = 2048;
pub const UI_FLAG_BORDER_ANY: u32 =
    UI_FLAG_BORDER_LEFT | UI_FLAG_BORDER_TOP | UI_FLAG_BORDER_RIGHT | UI_FLAG_BORDER_BOTTOM;
/// Phase-3 material MODE selector, bits 16-19 of `flags`.
pub const UI_MODE_SHIFT: u32 = 16;
pub const UI_MODE_MASK: u32 = 0xF << UI_MODE_SHIFT;

/// Standard SDF, gradient, and texture rendering.
pub const UI_MODE_STANDARD: u32 = 0;
/// MODE 1: the 16px checkerboard (upstream `alpha_pattern.wgsl`). Takes no
/// `UiMaterialData` fields (`variant`/`fixed_channel` are both ignored).
pub const UI_MODE_ALPHA_PATTERN: u32 = 1;
/// MODE 2: the 2D color plane (upstream `color_plane.wgsl`).
/// `UiMaterialData::variant` selects the axis pair (`UI_PLANE_*`);
/// `fixed_channel` is the third, non-plotted color-space component.
pub const UI_MODE_COLOR_PLANE: u32 = 2;

/// [`UiMaterialData::variant`] under `UI_MODE_COLOR_PLANE`: red (uv.x) /
/// green (uv.y), `fixed_channel` is blue. sRGB space.
pub const UI_PLANE_RG: u32 = 0;
/// Red (uv.x) / blue (uv.y), `fixed_channel` is green. sRGB space.
pub const UI_PLANE_RB: u32 = 1;
/// Green (uv.x) / blue (uv.y), `fixed_channel` is red. sRGB space.
pub const UI_PLANE_GB: u32 = 2;
/// Hue (uv.x) / saturation (1 - uv.y), `fixed_channel` is lightness. HSL
/// space.
pub const UI_PLANE_HS: u32 = 3;
/// Hue (uv.x) / lightness (1 - uv.y), `fixed_channel` is saturation. HSL
/// space.
pub const UI_PLANE_HL: u32 = 4;

/// Extract the MODE (0..15) from a vertex/fragment `flags` word.
#[inline]
pub fn ui_mode(flags: u32) -> u32 {
    (flags & UI_MODE_MASK) >> UI_MODE_SHIFT
}

// Upstream's RIGHT_VERTEX/BOTTOM_VERTEX corner-id bits are deliberately
// absent: the walker authors all four corners on the CPU, so the shader
// never reconstructs corner identity.

/// Triangle-list corner indices for one quad: (tl, bl, br), (tl, br, tr) in
/// TL,TR,BR,BL corner order — CCW front faces in +y-down NDC.
pub const UI_QUAD_TRIANGLES: [u32; 6] = [0, 3, 2, 0, 2, 1];

/// One corner of a UI quad. Four per quad, corner order TL, TR, BR, BL.
#[gpu_data]
pub struct UiVertex {
    /// Physical-pixel position, node transform pre-applied.
    pub pos: [f32; 2],
    /// Texture UV; under `UI_FLAG_GRADIENT`, `uv[0]` is the unclamped
    /// gradient parameter and `uv[1]` is unused.
    pub uv: [f32; 2],
    /// Linear RGBA fill/border color; gradient start color in the
    /// interpolation space under `UI_FLAG_GRADIENT`. Straight alpha.
    pub color: [f32; 4],
    /// Gradient end color (same space as `color`); zero otherwise.
    pub color2: [f32; 4],
    /// Corner radii in px: TL, TR, BR, BL.
    pub radius: [f32; 4],
    /// Border widths in px: left, top, right, bottom.
    pub border: [f32; 4],
    /// Node size in px (the SDF box extent).
    pub size: [f32; 2],
    /// This vertex's offset from the node center — the SDF sample point.
    pub point: [f32; 2],
    pub flags: u32,
    /// Bindless sampled-heap index; 0 = untextured (ZII).
    pub tex_slot: u32,
}

const _: () = assert!(core::mem::size_of::<UiVertex>() == 104);
const _: () = assert!(core::mem::align_of::<UiVertex>() == 4);
const _: () = assert!(core::mem::offset_of!(UiVertex, pos) == 0);
const _: () = assert!(core::mem::offset_of!(UiVertex, uv) == 8);
const _: () = assert!(core::mem::offset_of!(UiVertex, color) == 16);
const _: () = assert!(core::mem::offset_of!(UiVertex, color2) == 32);
const _: () = assert!(core::mem::offset_of!(UiVertex, radius) == 48);
const _: () = assert!(core::mem::offset_of!(UiVertex, border) == 64);
const _: () = assert!(core::mem::offset_of!(UiVertex, size) == 80);
const _: () = assert!(core::mem::offset_of!(UiVertex, point) == 88);
const _: () = assert!(core::mem::offset_of!(UiVertex, flags) == 96);
const _: () = assert!(core::mem::offset_of!(UiVertex, tex_slot) == 100);

/// Per-batch UI draw payload, shared by vertex and fragment stages.
#[gpu_data]
pub struct UiDraw {
    /// `4 * quad_count` vertices, corner order TL,TR,BR,BL per quad.
    pub vertices: GpuPtr<UiVertex>,
    /// `clip_xy = px_xy * view.xy + view.zw` — top-left origin, +y down,
    /// matching `TextCamera::xform`. For a W×H-px target:
    /// `[2/W, 2/H, -1, -1]`.
    pub view: [f32; 4],
    pub quad_count: u32,
    /// Bindless sampler-heap index used for every textured quad in the batch.
    pub sampler_slot: u32,
}

const _: () = assert!(core::mem::size_of::<UiDraw>() == 32);
const _: () = assert!(core::mem::align_of::<UiDraw>() == 4);
const _: () = assert!(core::mem::offset_of!(UiDraw, vertices) == 0);
const _: () = assert!(core::mem::offset_of!(UiDraw, view) == 8);
const _: () = assert!(core::mem::offset_of!(UiDraw, quad_count) == 24);
const _: () = assert!(core::mem::offset_of!(UiDraw, sampler_slot) == 28);

/// One corner of a rounded-rectangle shadow quad.
///
/// The quad covers `size + 6 * blur`; `uv` recovers the padded SDF point.
/// Its layout is separate from [`UiVertex`] because shadow shading requires
/// `bounds` and `blur` varyings.
#[gpu_data]
pub struct UiShadowVertex {
    /// Physical-pixel position, node transform (plus the shadow's own
    /// x/y offset) pre-applied.
    pub pos: [f32; 2],
    /// Standard `0..1` quad parameter across `bounds`, clip-displaced.
    pub uv: [f32; 2],
    /// Linear RGBA shadow color. Straight alpha — `ui_shadow_shade` writes
    /// `color.rgb` unchanged and derives the OUTPUT alpha from the coverage
    /// integral, exactly like upstream's fragment stage.
    pub color: [f32; 4],
    /// The shadow box size (node size + spread, resolved), the sharp-edged
    /// rect the blur is centered on. NOT the padded quad extent — see
    /// `bounds`.
    pub size: [f32; 2],
    /// Corner radii in px: TL, TR, BR, BL (spread-scaled — same convention
    /// as [`UiVertex::radius`]).
    pub radius: [f32; 4],
    /// Gaussian blur radius in px, pre-clamp (`ui_shadow_shade` applies
    /// upstream's `max(blur, 0.01)` floor).
    pub blur: f32,
    /// The padded render-quad extent this vertex's corner spans — `size +
    /// 6 * blur` on each axis. Divides into `uv` to recover the SDF sample
    /// point; see the struct doc.
    pub bounds: [f32; 2],
}

const _: () = assert!(core::mem::size_of::<UiShadowVertex>() == 68);
const _: () = assert!(core::mem::align_of::<UiShadowVertex>() == 4);
const _: () = assert!(core::mem::offset_of!(UiShadowVertex, pos) == 0);
const _: () = assert!(core::mem::offset_of!(UiShadowVertex, uv) == 8);
const _: () = assert!(core::mem::offset_of!(UiShadowVertex, color) == 16);
const _: () = assert!(core::mem::offset_of!(UiShadowVertex, size) == 32);
const _: () = assert!(core::mem::offset_of!(UiShadowVertex, radius) == 40);
const _: () = assert!(core::mem::offset_of!(UiShadowVertex, blur) == 56);
const _: () = assert!(core::mem::offset_of!(UiShadowVertex, bounds) == 60);

/// Per-batch box-shadow draw payload — the [`UiDraw`] sibling for the shadow
/// pipeline. No `sampler_slot`: shadows never sample a texture. Both
/// push-constant slots (vert and frag) receive the SAME [`UiShadowDraw`]
/// pointer, exactly like [`UiDraw`] (see `ui::UiPass::record_shadows`).
#[gpu_data]
pub struct UiShadowDraw {
    /// `4 * quad_count` vertices, corner order TL,TR,BR,BL per quad.
    pub vertices: GpuPtr<UiShadowVertex>,
    /// Same `[2/W, 2/H, -1, -1]` convention as [`UiDraw::view`].
    pub view: [f32; 4],
    pub quad_count: u32,
}

const _: () = assert!(core::mem::size_of::<UiShadowDraw>() == 28);
const _: () = assert!(core::mem::align_of::<UiShadowDraw>() == 4);
const _: () = assert!(core::mem::offset_of!(UiShadowDraw, vertices) == 0);
const _: () = assert!(core::mem::offset_of!(UiShadowDraw, view) == 8);
const _: () = assert!(core::mem::offset_of!(UiShadowDraw, quad_count) == 24);

/// Per-quad material parameters packed into `UiVertex::color2`.
#[gpu_data]
pub struct UiMaterialData {
    pub variant: u32,
    pub fixed_channel: f32,
    pub _pad0: [u32; 2],
}

const _: () = assert!(core::mem::size_of::<UiMaterialData>() == 16);
const _: () = assert!(core::mem::align_of::<UiMaterialData>() == 4);
const _: () = assert!(core::mem::offset_of!(UiMaterialData, variant) == 0);
const _: () = assert!(core::mem::offset_of!(UiMaterialData, fixed_channel) == 4);

impl UiMaterialData {
    /// Pack into a `UiVertex::color2` slot — the sole encoding, per the
    /// module doc's "Material encoding" section.
    #[inline]
    pub fn to_color2(self) -> Vec4 {
        Vec4::new(self.fixed_channel, self.variant as f32, 0.0, 0.0)
    }

    /// Inverse of [`Self::to_color2`]. `variant` round-trips exactly for
    /// every value this crate ever writes (small non-negative integers,
    /// see `UI_PLANE_*`) — `f32` represents every `u32` up to 2^24 exactly.
    #[inline]
    pub fn from_color2(color2: Vec4) -> Self {
        UiMaterialData {
            variant: color2.y as u32,
            fixed_channel: color2.x,
            _pad0: [0; 2],
        }
    }
}

#[inline]
pub fn ui_flag(flags: u32, mask: u32) -> bool {
    (flags & mask) != 0
}

/// Shortest signed distance from `point` (relative to box center) to the
/// boundary of a rounded box. Negative inside, zero on the edge, positive
/// outside. Radii ordered TL, TR, BR, BL. Ported verbatim from ui.wgsl.
pub fn sd_rounded_box(point: Vec2, size: Vec2, corner_radii: Vec4) -> f32 {
    // If 0.0 < y select the bottom pair (w=BL, z=BR — swapped so both pairs
    // read left-to-right), else the top pair (x=TL, y=TR).
    let rs = if 0.0 < point.y {
        Vec2::new(corner_radii.w, corner_radii.z)
    } else {
        Vec2::new(corner_radii.x, corner_radii.y)
    };
    let radius = if 0.0 < point.x { rs.y } else { rs.x };
    // Vector from the corner closest to the point, to the point.
    let corner_to_point = point.abs() - 0.5 * size;
    // Vector from the center of the radius circle to the point.
    let q = corner_to_point + radius;
    // Zeroed components drop the point out of the corner circle's quadrant.
    let l = q.max(Vec2::ZERO).length();
    let m = q.x.max(q.y).min(0.0);
    l + m - radius
}

/// Signed distance to the box inset by per-edge `inset` (L, T, R, B), with
/// corner radii shrunk to match. Ported verbatim from ui.wgsl.
pub fn sd_inset_rounded_box(point: Vec2, size: Vec2, radius: Vec4, inset: Vec4) -> f32 {
    let inner_size = size - Vec2::new(inset.x, inset.y) - Vec2::new(inset.z, inset.w);
    let inner_center = Vec2::new(inset.x, inset.y) + 0.5 * inner_size - 0.5 * size;
    let inner_point = point - inner_center;

    let mut r = radius;
    r.x -= inset.x.max(inset.y); // top left
    r.y -= inset.z.max(inset.y); // top right
    r.z -= inset.z.max(inset.w); // bottom right
    r.w -= inset.x.max(inset.w); // bottom left

    let half_size = inner_size * 0.5;
    let min_size = half_size.x.min(half_size.y);
    r = r.max(Vec4::ZERO).min(Vec4::splat(min_size));

    sd_rounded_box(inner_point, inner_size, r)
}

/// Is the nearest border edge one of the flagged ones? Distances are
/// width-normalized (Manhattan-per-edge), per upstream. Ported verbatim.
pub fn nearest_border_active(point_vs_mid: Vec2, size: Vec2, width: Vec4, flags: u32) -> bool {
    if (flags & UI_FLAG_BORDER_ANY) == UI_FLAG_BORDER_ANY {
        return true;
    }

    // Point vs top-left. 0.49999 (not 0.5), per upstream, so edge-exact
    // points stay inside the clamp.
    let point = (point_vs_mid + size * 0.49999).clamp(Vec2::ZERO, size);

    let left = point.x / width.x;
    let top = point.y / width.y;
    let right = (size.x - point.x) / width.z;
    let bottom = (size.y - point.y) / width.w;

    let min_dist = left.min(top).min(right.min(bottom));

    (ui_flag(flags, UI_FLAG_BORDER_LEFT) && min_dist == left)
        || (ui_flag(flags, UI_FLAG_BORDER_TOP) && min_dist == top)
        || (ui_flag(flags, UI_FLAG_BORDER_RIGHT) && min_dist == right)
        || (ui_flag(flags, UI_FLAG_BORDER_BOTTOM) && min_dist == bottom)
}

/// SDF edge coverage. Deliberately NOT `fwidth`-based — upstream found it
/// caused artifacts; the distance is already in pixel units.
#[inline]
pub fn ui_antialias(distance: f32) -> f32 {
    (0.5 - distance).clamp(0.0, 1.0)
}

/// The border ring: inside the outer boundary AND outside the inset inner
/// boundary, restricted to the flagged nearest edges. Ported verbatim.
pub fn draw_uinode_border(
    color: Vec4,
    point: Vec2,
    size: Vec2,
    radius: Vec4,
    border: Vec4,
    flags: u32,
) -> Vec4 {
    let external_distance = sd_rounded_box(point, size, radius);
    let internal_distance = sd_inset_rounded_box(point, size, radius, border);
    let border_distance = external_distance.max(-internal_distance);

    let nearest_border = if nearest_border_active(point, size, border, flags) {
        1.0
    } else {
        0.0
    };

    // Anti-alias only where a non-zero-width border exists, else a hairline
    // outline would appear on borderless external edges.
    let t = if external_distance < internal_distance {
        ui_antialias(border_distance)
    } else if border_distance >= 0.0 {
        0.0
    } else {
        1.0
    };

    // Alpha blending downstream — no premultiply here.
    Vec4::new(
        color.x,
        color.y,
        color.z,
        (color.w * t * nearest_border).clamp(0.0, 1.0),
    )
}

/// The fill: the area inside the border's inner edge (or outside it entirely
/// under INVERT). Ported verbatim.
pub fn draw_uinode_background(
    color: Vec4,
    point: Vec2,
    size: Vec2,
    radius: Vec4,
    border: Vec4,
    flags: u32,
) -> Vec4 {
    let sign = if ui_flag(flags, UI_FLAG_INVERT) {
        -1.0
    } else {
        1.0
    };
    let internal_distance = sd_inset_rounded_box(point, size, radius, border) * sign;
    let t = ui_antialias(internal_distance);
    Vec4::new(color.x, color.y, color.z, (color.w * t).clamp(0.0, 1.0))
}

/// One sRGB channel to linear. The exact curve (not the 2.2 approximation),
/// matching bevy's `color_operations.wgsl` `gamma()`.
pub fn srgb_channel_to_linear(value: f32) -> f32 {
    if value <= 0.0 {
        value
    } else if value <= 0.04045 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}

pub fn srgb_to_linear(color: Vec3) -> Vec3 {
    Vec3::new(
        srgb_channel_to_linear(color.x),
        srgb_channel_to_linear(color.y),
        srgb_channel_to_linear(color.z),
    )
}

/// HSL (hue normalized 0..1, HSL over sRGB components) to linear RGB.
/// Ported verbatim from `color_operations.wgsl`.
pub fn hsl_to_linear_rgb(hsl: Vec3) -> Vec3 {
    let h = hsl.x;
    let s = hsl.y;
    let l = hsl.z;
    let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
    let hp = h * 6.0;
    let x = c * (1.0 - (hp % 2.0 - 1.0).abs());
    let (r, g, b) = if (0.0..1.0).contains(&hp) {
        (c, x, 0.0)
    } else if (1.0..2.0).contains(&hp) {
        (x, c, 0.0)
    } else if (2.0..3.0).contains(&hp) {
        (0.0, c, x)
    } else if (3.0..4.0).contains(&hp) {
        (0.0, x, c)
    } else if (4.0..5.0).contains(&hp) {
        (x, 0.0, c)
    } else if (5.0..6.0).contains(&hp) {
        (c, 0.0, x)
    } else {
        (0.0, 0.0, 0.0)
    };
    let m = l - 0.5 * c;
    srgb_to_linear(Vec3::new(r + m, g + m, b + m))
}

/// Below this saturation an endpoint's hue is meaningless; take the other's.
const HUE_GUARD: f32 = 0.0001;

#[inline]
fn fract(x: f32) -> f32 {
    x - x.floor()
}

/// Shortest-path HSL mix with the desaturated-endpoint hue guard.
/// Ported verbatim from `color_operations.wgsl` `mix_hsl`.
pub fn mix_hsl(a: Vec3, b: Vec3, t: f32) -> Vec3 {
    let mut h = a.x;
    let mut g = b.x;
    if a.y < HUE_GUARD {
        h = g;
    } else if b.y < HUE_GUARD {
        g = h;
    }
    Vec3::new(
        fract(h + (fract(g - h + 0.5) - 0.5) * t),
        a.y + (b.y - a.y) * t,
        a.z + (b.z - a.z) * t,
    )
}

/// Mix two gradient stops at unclamped parameter `t` and convert to linear
/// RGBA. Out-of-range `t` yields transparent unless the FILL flags clamp it.
/// Stops are in the interpolation space named by the flags (alpha always
/// straight, mixed linearly).
pub fn ui_gradient_color(t: f32, start: Vec4, end: Vec4, flags: u32) -> Vec4 {
    if t < 0.0 {
        if ui_flag(flags, UI_FLAG_FILL_START) {
            return ui_gradient_convert(start, flags);
        }
        return Vec4::ZERO;
    }
    if t > 1.0 {
        if ui_flag(flags, UI_FLAG_FILL_END) {
            return ui_gradient_convert(end, flags);
        }
        return Vec4::ZERO;
    }
    let mixed = if ui_flag(flags, UI_FLAG_GRADIENT_SPACE_HSLA) {
        mix_hsl(start.truncate(), end.truncate(), t)
    } else {
        start.truncate().lerp(end.truncate(), t)
    };
    let alpha = start.w + (end.w - start.w) * t;
    ui_gradient_convert(Vec4::new(mixed.x, mixed.y, mixed.z, alpha), flags)
}

/// Convert a color in the gradient interpolation space to linear RGBA.
fn ui_gradient_convert(color: Vec4, flags: u32) -> Vec4 {
    let rgb = if ui_flag(flags, UI_FLAG_GRADIENT_SPACE_HSLA) {
        hsl_to_linear_rgb(color.truncate())
    } else {
        srgb_to_linear(color.truncate())
    };
    Vec4::new(rgb.x, rgb.y, rgb.z, color.w)
}

/// Shades a UI quad before optional texture multiplication.
pub fn ui_fragment_shade(
    color: Vec4,
    color2: Vec4,
    uv: Vec2,
    point: Vec2,
    size: Vec2,
    radius: Vec4,
    border: Vec4,
    flags: u32,
) -> Vec4 {
    let base = if ui_flag(flags, UI_FLAG_GRADIENT) {
        ui_gradient_color(uv.x, color, color2, flags)
    } else {
        color
    };
    if ui_flag(flags, UI_FLAG_BORDER_ANY) {
        draw_uinode_border(base, point, size, radius, border, flags)
    } else {
        draw_uinode_background(base, point, size, radius, border, flags)
    }
}

/// GLSL/WGSL-compatible monotonic Hermite interpolation.
#[inline]
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Evaluates the 16-pixel checkerboard material.
pub fn ui_alpha_pattern_shade(point: Vec2, size: Vec2, radius: Vec4) -> Vec4 {
    let checker = point / 16.0;
    let check = if (fract(checker.x) < 0.5) != (fract(checker.y) < 0.5) {
        1.0
    } else {
        0.0
    };
    let bg = Vec3::new(0.2, 0.2, 0.2).lerp(Vec3::new(0.6, 0.6, 0.6), check);
    let external_distance = sd_rounded_box(point, size, radius);
    let alpha = smoothstep(0.5, -0.5, external_distance);
    Vec4::new(bg.x, bg.y, bg.z, alpha)
}

/// Evaluates a runtime-selected two-dimensional color plane.
///
/// Unknown variants return opaque magenta.
pub fn ui_color_plane_shade(uv: Vec2, variant: u32, fixed_channel: f32) -> Vec4 {
    let rgb = match variant {
        UI_PLANE_RG => srgb_to_linear(Vec3::new(uv.x, uv.y, fixed_channel)),
        UI_PLANE_RB => srgb_to_linear(Vec3::new(uv.x, fixed_channel, uv.y)),
        UI_PLANE_GB => srgb_to_linear(Vec3::new(fixed_channel, uv.x, uv.y)),
        UI_PLANE_HS => hsl_to_linear_rgb(Vec3::new(uv.x, 1.0 - uv.y, fixed_channel)),
        UI_PLANE_HL => hsl_to_linear_rgb(Vec3::new(uv.x, fixed_channel, 1.0 - uv.y)),
        _ => Vec3::new(1.0, 0.0, 1.0),
    };
    Vec4::new(rgb.x, rgb.y, rgb.z, 1.0)
}

/// Dispatches a material mode; unknown modes return magenta.
pub fn ui_material_shade(
    mode: u32,
    uv: Vec2,
    point: Vec2,
    size: Vec2,
    radius: Vec4,
    material: UiMaterialData,
) -> Vec4 {
    match mode {
        UI_MODE_ALPHA_PATTERN => ui_alpha_pattern_shade(point, size, radius),
        UI_MODE_COLOR_PLANE => ui_color_plane_shade(uv, material.variant, material.fixed_channel),
        _ => Vec4::new(1.0, 0.0, 1.0, 1.0),
    }
}

/// Routes standard and material UI shading by mode flags.
#[allow(clippy::too_many_arguments)]
pub fn ui_shade(
    base: Vec4,
    color2: Vec4,
    uv: Vec2,
    point: Vec2,
    size: Vec2,
    radius: Vec4,
    border: Vec4,
    flags: u32,
) -> Vec4 {
    match ui_mode(flags) {
        UI_MODE_STANDARD => ui_fragment_shade(base, color2, uv, point, size, radius, border, flags),
        mode => ui_material_shade(
            mode,
            uv,
            point,
            size,
            radius,
            UiMaterialData::from_color2(color2),
        ),
    }
}

/// Recovers the shadow SDF point from padded quad coordinates.
#[inline]
pub fn ui_shadow_point(uv: Vec2, bounds: Vec2) -> Vec2 {
    (uv - Vec2::splat(0.5)) * bounds
}

/// Vertical samples used by the shadow Gaussian integration.
pub const SHADOW_SAMPLES: u32 = 4;

/// Approximates `erf` with a fourth-degree rational polynomial popularized
/// by Evan Wallace's “Fast Rounded Rectangle Shadows”.
///
/// Maximum absolute error is approximately `8.2e-4` on `[0, 4]`.
pub fn shadow_erf(p: Vec2) -> Vec2 {
    let s = p.signum();
    let a = p.abs();
    let mut result = Vec2::ONE
        + (Vec2::splat(0.278393) + (Vec2::splat(0.230389) + Vec2::splat(0.078108) * (a * a)) * a)
            * a;
    result *= result;
    s - s / (result * result)
}

/// Picks the corner radius nearest `p` (relative to the shadow box center)
/// by quadrant — ported verbatim from upstream's `selectCorner`. Radii
/// ordered TL, TR, BR, BL, matching [`sd_rounded_box`]'s convention (and
/// upstream's own `ResolvedBorderRadius` layout).
fn shadow_select_corner(p: Vec2, c: Vec4) -> f32 {
    let step_x = if p.x >= 0.0 { 1.0 } else { 0.0 };
    let step_y = if p.y >= 0.0 { 1.0 } else { 0.0 };
    let top = c.x + (c.y - c.x) * step_x;
    let bottom = c.w + (c.z - c.w) * step_x;
    top + (bottom - top) * step_y
}

/// The 1D analytic horizontal integral of a Gaussian-blurred rounded-box
/// edge at vertical offset `y` from the box's own coordinate frame — ported
/// verbatim from upstream's `horizontalRoundedBoxShadow`.
fn shadow_horizontal_integral(x: f32, y: f32, blur: f32, corner: f32, half_size: Vec2) -> f32 {
    let d = (half_size.y - corner - y.abs()).min(0.0);
    let c = half_size.x - corner + (corner * corner - d * d).max(0.0).sqrt();
    let integral = Vec2::splat(0.5)
        + 0.5
            * shadow_erf(
                (Vec2::splat(x) + Vec2::new(-c, c)) * (core::f32::consts::FRAC_1_SQRT_2 / blur),
            );
    integral.y - integral.x
}

/// 1D Gaussian PDF, ported verbatim from upstream's `gaussian`.
fn shadow_gaussian(x: f32, sigma: f32) -> f32 {
    (-(x * x) / (2.0 * sigma * sigma)).exp() / ((2.0 * core::f32::consts::PI).sqrt() * sigma)
}

/// Integrates rounded-box shadow coverage over vertical Gaussian samples.
///
/// `blur` must be floored before calling this function.
pub fn ui_shadow_coverage(lower: Vec2, upper: Vec2, point: Vec2, blur: f32, corners: Vec4) -> f32 {
    let half_size = (upper - lower) * 0.5;
    let center = (lower + upper) * 0.5;
    let p = point - center;
    let low = p.y - half_size.y;
    let high = p.y + half_size.y;
    let start = (-3.0 * blur).clamp(low, high);
    let end = (3.0 * blur).clamp(low, high);
    let step = (end - start) / SHADOW_SAMPLES as f32;
    let corner = shadow_select_corner(p, corners);

    let mut y = start + step * 0.5;
    let mut value = 0.0f32;
    for _ in 0..SHADOW_SAMPLES {
        value += shadow_horizontal_integral(p.x, p.y - y, blur, corner, half_size)
            * shadow_gaussian(y, blur)
            * step;
        y += step;
    }
    value
}

/// Shades a shadow quad with Gaussian rounded-box coverage.
///
/// Zero blur is floored to a finite minimum.
pub fn ui_shadow_shade(color: Vec4, point: Vec2, size: Vec2, radius: Vec4, blur: f32) -> Vec4 {
    let effective_blur = blur.max(0.01);
    let coverage = ui_shadow_coverage(-0.5 * size, 0.5 * size, point, effective_blur, radius);
    Vec4::new(color.x, color.y, color.z, color.w * coverage)
}

#[cfg(all(test, not(target_arch = "spirv")))]
mod tests {
    use super::*;

    const EPS: f32 = 1e-4;

    #[test]
    fn sd_rounded_box_center_edge_corner() {
        let size = Vec2::new(100.0, 100.0);
        let r = Vec4::splat(20.0);
        // Center: 50 px from every edge, but corner circles pull the
        // boundary in — the sharp-box distance still governs the axis
        // directions: -(50) + max stays -50? No: nearest boundary point is
        // the edge midpoint, distance 50.
        assert!((sd_rounded_box(Vec2::ZERO, size, r) - (-50.0)).abs() < EPS);
        // Edge midpoint (right edge): exactly on the boundary.
        assert!(sd_rounded_box(Vec2::new(50.0, 0.0), size, r).abs() < EPS);
        // The sharp corner (50,50) with r=20: circle center (30,30),
        // distance = |(50,50)-(30,30)| - 20 = 20*sqrt(2) - 20.
        let d = sd_rounded_box(Vec2::new(50.0, 50.0), size, r);
        assert!((d - (20.0 * core::f32::consts::SQRT_2 - 20.0)).abs() < EPS);
        // Just inside the corner circle along the diagonal: on the boundary.
        let on = Vec2::new(30.0, 30.0) + Vec2::splat(core::f32::consts::FRAC_1_SQRT_2 * 20.0);
        assert!(sd_rounded_box(on, size, r).abs() < EPS);
        // Sharp box (r=0): corner point is on the boundary.
        assert!(sd_rounded_box(Vec2::new(50.0, 50.0), size, Vec4::ZERO).abs() < EPS);
        // Outside along +x.
        assert!((sd_rounded_box(Vec2::new(60.0, 0.0), size, Vec4::ZERO) - 10.0).abs() < EPS);
    }

    #[test]
    fn sd_rounded_box_per_corner_radii() {
        let size = Vec2::new(100.0, 100.0);
        // Only the TL corner rounded (radius vec is TL,TR,BR,BL).
        let r = Vec4::new(20.0, 0.0, 0.0, 0.0);
        // TL corner (-50,-50) is rounded: outside by 20*(sqrt(2)-1).
        let d_tl = sd_rounded_box(Vec2::new(-50.0, -50.0), size, r);
        assert!((d_tl - (20.0 * core::f32::consts::SQRT_2 - 20.0)).abs() < EPS);
        // BR corner (+50,+50) is sharp: on the boundary.
        assert!(sd_rounded_box(Vec2::new(50.0, 50.0), size, r).abs() < EPS);
    }

    #[test]
    fn inset_box_border_ring() {
        let size = Vec2::new(100.0, 60.0);
        let border = Vec4::new(4.0, 4.0, 4.0, 4.0);
        // Center of the left border strip: inside outer, outside inner.
        let p = Vec2::new(-48.0, 0.0);
        let external = sd_rounded_box(p, size, Vec4::ZERO);
        let internal = sd_inset_rounded_box(p, size, Vec4::ZERO, border);
        assert!(external < 0.0 && internal > 0.0);
        // border distance formula puts the point inside the ring.
        assert!(external.max(-internal) < 0.0);
        // Node center: inside the inner box, so outside the ring.
        let internal_c = sd_inset_rounded_box(Vec2::ZERO, size, Vec4::ZERO, border);
        assert!(internal_c < 0.0);
    }

    #[test]
    fn nearest_border_edges() {
        let size = Vec2::new(100.0, 100.0);
        let w = Vec4::splat(4.0);
        // A point hugging the left edge.
        let left_point = Vec2::new(-49.0, 0.0);
        assert!(nearest_border_active(
            left_point,
            size,
            w,
            UI_FLAG_BORDER_LEFT
        ));
        assert!(!nearest_border_active(
            left_point,
            size,
            w,
            UI_FLAG_BORDER_TOP
        ));
        // All-edges shortcut.
        assert!(nearest_border_active(
            left_point,
            size,
            w,
            UI_FLAG_BORDER_ANY
        ));
    }

    #[test]
    fn srgb_round_trip_known_values() {
        // sRGB 0.5 → linear ≈ 0.21404114
        assert!((srgb_channel_to_linear(0.5) - 0.214_041_14).abs() < 1e-6);
        assert!(srgb_channel_to_linear(0.0).abs() < 1e-9);
        assert!((srgb_channel_to_linear(1.0) - 1.0).abs() < 1e-6);
        // The linear-falloff segment.
        assert!((srgb_channel_to_linear(0.04) - 0.04 / 12.92).abs() < 1e-9);
    }

    #[test]
    fn hsl_known_colors() {
        // Pure red: H=0, S=1, L=0.5.
        let red = hsl_to_linear_rgb(Vec3::new(0.0, 1.0, 0.5));
        assert!((red - Vec3::new(1.0, 0.0, 0.0)).length() < 1e-5);
        // Pure green: H=1/3.
        let green = hsl_to_linear_rgb(Vec3::new(1.0 / 3.0, 1.0, 0.5));
        assert!((green - Vec3::new(0.0, 1.0, 0.0)).length() < 1e-5);
        // White: L=1 regardless of H/S.
        let white = hsl_to_linear_rgb(Vec3::new(0.7, 0.3, 1.0));
        assert!((white - Vec3::ONE).length() < 1e-5);
    }

    #[test]
    fn hsl_mix_wraps_hue_shortest_path() {
        // 350° (0.9722) to 10° (0.0278) should pass through 0°, not 180°.
        let a = Vec3::new(350.0 / 360.0, 1.0, 0.5);
        let b = Vec3::new(10.0 / 360.0, 1.0, 0.5);
        let mid = mix_hsl(a, b, 0.5);
        // Midpoint hue = 0° (wrapped), i.e. fract(...) == 0.
        assert!(mid.x < 1e-4 || mid.x > 1.0 - 1e-4);
        // Desaturated endpoint adopts the other's hue.
        let gray = Vec3::new(0.9, 0.0, 0.5);
        let color = Vec3::new(0.25, 1.0, 0.5);
        assert!((mix_hsl(gray, color, 0.5).x - 0.25).abs() < 1e-5);
    }

    #[test]
    fn gradient_fill_and_transparent_ends() {
        let start = Vec4::new(1.0, 0.0, 0.0, 1.0);
        let end = Vec4::new(0.0, 0.0, 1.0, 1.0);
        // Out of range without FILL: transparent.
        assert_eq!(
            ui_gradient_color(-0.5, start, end, UI_FLAG_GRADIENT),
            Vec4::ZERO
        );
        assert_eq!(
            ui_gradient_color(1.5, start, end, UI_FLAG_GRADIENT),
            Vec4::ZERO
        );
        // With FILL_START: clamps to converted start color.
        let filled = ui_gradient_color(-0.5, start, end, UI_FLAG_GRADIENT | UI_FLAG_FILL_START);
        assert!((filled - Vec4::new(1.0, 0.0, 0.0, 1.0)).length() < 1e-5);
        // Midpoint in sRGB space: mixed sRGB 0.5 → linear ≈ 0.214.
        let mid = ui_gradient_color(0.5, start, end, UI_FLAG_GRADIENT);
        assert!((mid.x - 0.214_041_14).abs() < 1e-4);
        assert!((mid.z - 0.214_041_14).abs() < 1e-4);
        assert!((mid.w - 1.0).abs() < 1e-6);
        // The same stops mixed in HSL space differ from the sRGB mix
        // (red→blue through HSL passes through green or magenta hues).
        let mid_hsl = ui_gradient_color(
            0.5,
            Vec4::new(0.0, 1.0, 0.5, 1.0),           // red in HSL
            Vec4::new(240.0 / 360.0, 1.0, 0.5, 1.0), // blue in HSL
            UI_FLAG_GRADIENT | UI_FLAG_GRADIENT_SPACE_HSLA,
        );
        assert!((mid_hsl - mid).length() > 0.05);
    }

    #[test]
    fn fragment_shade_solid_and_border_paths() {
        let size = Vec2::new(100.0, 100.0);
        let color = Vec4::new(0.2, 0.4, 0.8, 1.0);
        // Solid center: full alpha.
        let c = ui_fragment_shade(
            color,
            Vec4::ZERO,
            Vec2::ZERO,
            Vec2::ZERO,
            size,
            Vec4::ZERO,
            Vec4::ZERO,
            0,
        );
        assert!((c.w - 1.0).abs() < 1e-6);
        // Well outside (sharp box, point 10px past the right edge): alpha 0.
        let out = ui_fragment_shade(
            color,
            Vec4::ZERO,
            Vec2::ZERO,
            Vec2::new(60.0, 0.0),
            size,
            Vec4::ZERO,
            Vec4::ZERO,
            0,
        );
        assert!(out.w.abs() < 1e-6);
        // Border path: point in the left border strip with only LEFT flagged.
        let b = ui_fragment_shade(
            color,
            Vec4::ZERO,
            Vec2::ZERO,
            Vec2::new(-48.0, 0.0),
            size,
            Vec4::ZERO,
            Vec4::splat(4.0),
            UI_FLAG_BORDER_LEFT,
        );
        assert!((b.w - 1.0).abs() < 1e-6);
        // Same point, only RIGHT flagged: nearest edge is left → suppressed.
        let b2 = ui_fragment_shade(
            color,
            Vec4::ZERO,
            Vec2::ZERO,
            Vec2::new(-48.0, 0.0),
            size,
            Vec4::ZERO,
            Vec4::splat(4.0),
            UI_FLAG_BORDER_RIGHT,
        );
        assert!(b2.w.abs() < 1e-6);
    }

    #[test]
    fn ui_material_data_color2_roundtrip() {
        let data = UiMaterialData {
            variant: UI_PLANE_HL,
            fixed_channel: 0.375,
            _pad0: [0; 2],
        };
        let packed = data.to_color2();
        assert!((packed.x - 0.375).abs() < EPS);
        assert!((packed.y - UI_PLANE_HL as f32).abs() < EPS);
        let unpacked = UiMaterialData::from_color2(packed);
        assert_eq!(unpacked.variant, UI_PLANE_HL);
        assert!((unpacked.fixed_channel - 0.375).abs() < EPS);
    }

    // The checker's spatial period is EXACTLY 16px per axis (`point / 16.0`
    // feeds a `fract`, which repeats every integer). That means a same-axis
    // offset of exactly 16px ALWAYS returns to the same parity — the square
    // wave completed one full cycle. The parity actually flips at the
    // HALF-period, 8px, where `fract` crosses 0.5. Both facts are asserted
    // below (matching the real math, not the size of one visible checker
    // square, which upstream's own naming ("16px checkerboard pattern")
    // refers to as the tile period, not the flip distance).
    #[test]
    fn alpha_pattern_checker_parity() {
        let size = Vec2::new(1000.0, 1000.0);
        let radius = Vec4::ZERO;
        // Deep inside the box (far from any SDF edge) so alpha is always 1
        // and only the `bg` mix (the checker) varies between samples.
        let c0 = ui_alpha_pattern_shade(Vec2::new(0.0, 0.0), size, radius);
        let c8 = ui_alpha_pattern_shade(Vec2::new(8.0, 0.0), size, radius);
        let c16 = ui_alpha_pattern_shade(Vec2::new(16.0, 0.0), size, radius);
        // Half-period (8px) crossing: parity differs.
        assert!(
            (c0 - c8).length() > 0.1,
            "half-period offset must flip parity: {c0:?} vs {c8:?}"
        );
        // Full period (16px): parity returns to the same state.
        assert!(
            (c0 - c16).length() < EPS,
            "full-period offset must repeat parity: {c0:?} vs {c16:?}"
        );
        // Both are fully opaque (well inside a large sharp box).
        assert!((c0.w - 1.0).abs() < 1e-3);
        assert!((c8.w - 1.0).abs() < 1e-3);
    }

    #[test]
    fn alpha_pattern_colors_are_the_two_grays() {
        let size = Vec2::new(1000.0, 1000.0);
        let dark = Vec3::new(0.2, 0.2, 0.2);
        let light = Vec3::new(0.6, 0.6, 0.6);
        let c0 = ui_alpha_pattern_shade(Vec2::new(0.0, 0.0), size, Vec4::ZERO);
        let c8 = ui_alpha_pattern_shade(Vec2::new(8.0, 0.0), size, Vec4::ZERO);
        let c0_rgb = Vec3::new(c0.x, c0.y, c0.z);
        let c8_rgb = Vec3::new(c8.x, c8.y, c8.z);
        assert!(
            (c0_rgb - dark).length() < EPS || (c0_rgb - light).length() < EPS,
            "unexpected checker color {c0_rgb:?}"
        );
        assert!(
            (c8_rgb - dark).length() < EPS || (c8_rgb - light).length() < EPS,
            "unexpected checker color {c8_rgb:?}"
        );
        assert!(
            (c0_rgb - c8_rgb).length() > 0.1,
            "adjacent half-cells must differ"
        );
    }

    #[test]
    fn alpha_pattern_clips_to_rounded_box() {
        let size = Vec2::new(100.0, 100.0);
        // Sharp box (r=0): 10px outside the right edge is fully clipped.
        let outside = ui_alpha_pattern_shade(Vec2::new(60.0, 0.0), size, Vec4::ZERO);
        assert!(outside.w.abs() < 1e-3);
        // Dead center: fully opaque.
        let center = ui_alpha_pattern_shade(Vec2::ZERO, size, Vec4::ZERO);
        assert!((center.w - 1.0).abs() < 1e-3);
    }

    #[test]
    fn color_plane_rg_rb_gb_match_srgb_to_linear() {
        let uv = Vec2::new(0.75, 0.25);
        let fixed = 0.4;
        let rg = ui_color_plane_shade(uv, UI_PLANE_RG, fixed);
        let want_rg = srgb_to_linear(Vec3::new(uv.x, uv.y, fixed));
        assert!((Vec3::new(rg.x, rg.y, rg.z) - want_rg).length() < EPS);
        assert!((rg.w - 1.0).abs() < EPS);

        let rb = ui_color_plane_shade(uv, UI_PLANE_RB, fixed);
        let want_rb = srgb_to_linear(Vec3::new(uv.x, fixed, uv.y));
        assert!((Vec3::new(rb.x, rb.y, rb.z) - want_rb).length() < EPS);

        let gb = ui_color_plane_shade(uv, UI_PLANE_GB, fixed);
        let want_gb = srgb_to_linear(Vec3::new(fixed, uv.x, uv.y));
        assert!((Vec3::new(gb.x, gb.y, gb.z) - want_gb).length() < EPS);
    }

    #[test]
    fn color_plane_hs_hl_match_hsl_to_linear_with_y_flip() {
        let uv = Vec2::new(0.6, 0.3);
        let fixed = 0.55;
        let hs = ui_color_plane_shade(uv, UI_PLANE_HS, fixed);
        let want_hs = hsl_to_linear_rgb(Vec3::new(uv.x, 1.0 - uv.y, fixed));
        assert!((Vec3::new(hs.x, hs.y, hs.z) - want_hs).length() < EPS);

        let hl = ui_color_plane_shade(uv, UI_PLANE_HL, fixed);
        let want_hl = hsl_to_linear_rgb(Vec3::new(uv.x, fixed, 1.0 - uv.y));
        assert!((Vec3::new(hl.x, hl.y, hl.z) - want_hl).length() < EPS);
    }

    #[test]
    fn color_plane_corners_hand_computed() {
        // RG plane at uv=(0,0), fixed_channel=0: pure black.
        let black = ui_color_plane_shade(Vec2::ZERO, UI_PLANE_RG, 0.0);
        assert!((Vec3::new(black.x, black.y, black.z) - Vec3::ZERO).length() < EPS);
        // RG plane at uv=(1,1), fixed_channel=1: pure white.
        let white = ui_color_plane_shade(Vec2::ONE, UI_PLANE_RG, 1.0);
        assert!((Vec3::new(white.x, white.y, white.z) - Vec3::ONE).length() < 1e-5);
        // HL plane: h = uv.x, s = fixed_channel, l = 1.0 - uv.y (per
        // `UI_PLANE_HL`'s doc comment). uv=(0, 0.5), fixed_channel=1.0 ->
        // (h=0, s=1, l=0.5) -> pure red, the same known value
        // `hsl_known_colors` already checks directly on `hsl_to_linear_rgb`.
        let red = ui_color_plane_shade(Vec2::new(0.0, 0.5), UI_PLANE_HL, 1.0);
        assert!((Vec3::new(red.x, red.y, red.z) - Vec3::new(1.0, 0.0, 0.0)).length() < 1e-4);
    }

    #[test]
    fn color_plane_unknown_variant_is_magenta_sentinel() {
        let c = ui_color_plane_shade(Vec2::ZERO, 99, 0.0);
        assert!((c - Vec4::new(1.0, 0.0, 1.0, 1.0)).length() < EPS);
    }

    #[test]
    fn ui_shade_routes_by_mode() {
        let size = Vec2::new(100.0, 100.0);
        // MODE 0 (standard): identical to calling `ui_fragment_shade` directly.
        let color = Vec4::new(0.2, 0.4, 0.8, 1.0);
        let direct = ui_fragment_shade(
            color,
            Vec4::ZERO,
            Vec2::ZERO,
            Vec2::ZERO,
            size,
            Vec4::ZERO,
            Vec4::ZERO,
            0,
        );
        let routed = ui_shade(
            color,
            Vec4::ZERO,
            Vec2::ZERO,
            Vec2::ZERO,
            size,
            Vec4::ZERO,
            Vec4::ZERO,
            0,
        );
        assert_eq!(direct, routed);

        // MODE ALPHA_PATTERN: routed result matches calling the material
        // shader directly.
        let flags = UI_MODE_ALPHA_PATTERN << UI_MODE_SHIFT;
        let direct_ap = ui_alpha_pattern_shade(Vec2::ZERO, size, Vec4::ZERO);
        let routed_ap = ui_shade(
            color,
            Vec4::ZERO,
            Vec2::ZERO,
            Vec2::ZERO,
            size,
            Vec4::ZERO,
            Vec4::ZERO,
            flags,
        );
        assert_eq!(direct_ap, routed_ap);

        // MODE COLOR_PLANE: color2 carries the packed material data.
        let material = UiMaterialData {
            variant: UI_PLANE_GB,
            fixed_channel: 0.3,
            _pad0: [0; 2],
        };
        let uv = Vec2::new(0.2, 0.9);
        let flags_cp = UI_MODE_COLOR_PLANE << UI_MODE_SHIFT;
        let direct_cp = ui_color_plane_shade(uv, UI_PLANE_GB, 0.3);
        let routed_cp = ui_shade(
            color,
            material.to_color2(),
            uv,
            Vec2::ZERO,
            size,
            Vec4::ZERO,
            Vec4::ZERO,
            flags_cp,
        );
        assert_eq!(direct_cp, routed_cp);
    }

    // Test rounded-rectangle shadow coverage.

    /// A tabulated high-precision erf (from `libm`/Python's `math.erf`,
    /// spot-checked against Abramowitz & Stegun tables) at values spanning
    /// the domain the shadow integral actually evaluates `shadow_erf` over
    /// (arguments are `(x ± c) * FRAC_1_SQRT_2 / blur`, typically O(1..10)).
    /// Tolerance 1.5e-3 — measured max absolute error of this exact
    /// polynomial approximation over `[0, 4]` is ≈8.2e-4 (see
    /// `shadow_erf`'s doc comment); 1.5e-3 leaves headroom without hiding a
    /// transcription bug.
    #[test]
    fn erf_matches_reference_within_documented_tolerance() {
        const TOL: f32 = 1.5e-3;
        let cases: &[(f32, f32)] = &[
            (0.0, 0.0),
            (0.1, 0.112_462_92),
            (0.25, 0.276_326_39),
            (0.5, 0.520_499_88),
            (0.75, 0.711_155_63),
            (1.0, 0.842_700_79),
            (1.5, 0.966_105_15),
            (2.0, 0.995_322_27),
            (3.0, 0.999_977_91),
        ];
        for &(x, want) in cases {
            let got = shadow_erf(Vec2::new(x, -x)).x;
            assert!((got - want).abs() < TOL, "erf({x}) = {got}, want {want}");
            let got_neg = shadow_erf(Vec2::new(x, -x)).y;
            assert!(
                (got_neg + want).abs() < TOL,
                "erf(-{x}) = {got_neg}, want {}",
                -want
            );
        }
        // Odd function: erf(-x) == -erf(x), exactly (both branches share the
        // same `s`/`a` split).
        let p = Vec2::new(0.37, 1.83);
        let e = shadow_erf(p);
        let e_neg = shadow_erf(-p);
        assert!(
            (e + e_neg).length() < 1e-6,
            "erf should be odd: erf(p)={e:?} erf(-p)={e_neg:?}"
        );
    }

    /// Coverage falls off monotonically along a ray from the shadow box's
    /// edge outward into the blurred penumbra (sharp corners, so
    /// [`ui_shadow_coverage`]'s only spatial variation along +x past the
    /// right edge is the blur falloff).
    #[test]
    fn coverage_falls_off_monotonically_outward_from_the_edge() {
        let size = Vec2::new(100.0, 100.0);
        let blur = 8.0;
        let radius = Vec4::ZERO;
        let mut last = f32::INFINITY;
        // Start just past the right edge (x=50) and walk outward.
        for step in 0..12 {
            let x = 50.0 + step as f32 * 4.0;
            let point = Vec2::new(x, 0.0);
            let c = ui_shadow_coverage(-0.5 * size, 0.5 * size, point, blur, radius);
            assert!(
                c <= last + 1e-6,
                "coverage should not increase moving outward: at x={x} got {c}, previous {last}"
            );
            last = c;
        }
        assert!(
            last < 0.05,
            "far outside the blur range, coverage should be ~0: {last}"
        );
    }

    /// Symmetric input (square box, uniform radius) yields symmetric
    /// coverage: reflecting `point` across either axis doesn't change the
    /// result.
    #[test]
    fn coverage_is_symmetric_for_a_symmetric_box() {
        let size = Vec2::new(80.0, 80.0);
        let blur = 6.0;
        let radius = Vec4::splat(10.0);
        let lower = -0.5 * size;
        let upper = 0.5 * size;

        let p = Vec2::new(37.0, -22.0);
        let c = ui_shadow_coverage(lower, upper, p, blur, radius);
        let c_x = ui_shadow_coverage(lower, upper, Vec2::new(-p.x, p.y), blur, radius);
        let c_y = ui_shadow_coverage(lower, upper, Vec2::new(p.x, -p.y), blur, radius);
        let c_xy = ui_shadow_coverage(lower, upper, -p, blur, radius);
        assert!((c - c_x).abs() < 1e-4, "reflect x: {c} vs {c_x}");
        assert!((c - c_y).abs() < 1e-4, "reflect y: {c} vs {c_y}");
        assert!((c - c_xy).abs() < 1e-4, "reflect both: {c} vs {c_xy}");
    }

    /// `ui_shadow_shade` floors `blur` to upstream's `0.01` minimum, so a
    /// literal `blur = 0.0` (a plain spread-only "shadow", or a not-yet-
    /// resolved default) degenerates to a very sharp but finite, non-NaN
    /// edge — never a divide-by-zero.
    #[test]
    fn blur_zero_degenerates_to_a_sharp_finite_edge_not_nan() {
        let color = Vec4::new(0.0, 0.0, 0.0, 0.8);
        let size = Vec2::new(50.0, 50.0);
        let radius = Vec4::ZERO;

        let center = ui_shadow_shade(color, Vec2::ZERO, size, radius, 0.0);
        assert!(
            center.w.is_finite(),
            "center alpha must be finite: {center:?}"
        );
        assert!(
            (center.w - 0.8).abs() < 5e-3,
            "deep inside a sharp box, alpha should be ~full: {center:?}"
        );

        let outside = ui_shadow_shade(color, Vec2::new(40.0, 0.0), size, radius, 0.0);
        assert!(
            outside.w.is_finite(),
            "outside alpha must be finite: {outside:?}"
        );
        assert!(
            outside.w < 0.05,
            "well outside a near-zero-blur box, alpha should be ~0: {outside:?}"
        );

        // Matches the explicitly-floored blur value bit-for-bit-close: the
        // floor is applied inside `ui_shadow_shade`, so blur=0.0 and
        // blur=0.01 must agree.
        let floored = ui_shadow_shade(color, Vec2::ZERO, size, radius, 0.01);
        assert!(
            (center.w - floored.w).abs() < 1e-6,
            "blur=0 should floor to blur=0.01: {center:?} vs {floored:?}"
        );
    }

    /// `ui_shadow_point` is the one-line uv->point recovery `ui_shadow_vert`
    /// calls into `abi-ui` for — sanity-check the affine map directly.
    #[test]
    fn shadow_point_recovers_corner_offset_from_uv_and_bounds() {
        let bounds = Vec2::new(120.0, 80.0);
        assert_eq!(
            ui_shadow_point(Vec2::new(0.0, 0.0), bounds),
            Vec2::new(-60.0, -40.0)
        );
        assert_eq!(
            ui_shadow_point(Vec2::new(1.0, 1.0), bounds),
            Vec2::new(60.0, 40.0)
        );
        assert_eq!(ui_shadow_point(Vec2::new(0.5, 0.5), bounds), Vec2::ZERO);
    }
}
