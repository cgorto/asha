//! Baked debug-visualization textures: the generator vocabulary.
//!
//! Bake requests describe procedural fullscreen textures.
//!
//! Evaluation is shared by host and shader targets. Unknown kinds render
//! magenta, making invalid debug requests visible.

use crate::gpu_data;
use glam::{Vec2, Vec4};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

/// The canonical UV map: color IS the coordinate. Ignores both colors.
pub const BAKE_UV: u32 = 0;
/// Checker of `color_a`/`color_b` with darkened cell-boundary lines.
/// `params = [cells_across, line_half_width_in_cell_units, _, _]`.
pub const BAKE_GRID: u32 = 1;
/// Annulus of `color_a` on `color_b`, centered, radii in UV (0.5 = edge).
/// `params = [r_inner, r_outer, feather, _]`.
pub const BAKE_RING: u32 = 2;
/// Soft disc of `color_a` on `color_b`.
/// `params = [radius, feather, _, _]`.
pub const BAKE_DOT: u32 = 3;

/// One procedural texture request with a stable GPU-facing layout.
#[gpu_data]
pub struct BakeData {
    pub color_a: [f32; 4],
    pub color_b: [f32; 4],
    pub params: [f32; 4],
    pub kind: u32,
    pub _pad: [u32; 3],
}

const _: () = assert!(core::mem::size_of::<BakeData>() == 64);

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Evaluates one texel; all four output channels are meaningful.
pub fn bake_eval(kind: u32, color_a: Vec4, color_b: Vec4, params: Vec4, uv: Vec2) -> Vec4 {
    match kind {
        BAKE_UV => Vec4::new(uv.x, uv.y, 0.0, 1.0),
        BAKE_GRID => {
            let cells = params.x.max(1.0);
            let p = uv * cells;
            let checker = (p.x.floor() + p.y.floor()) * 0.5;
            let base = if checker.fract() < 0.25 {
                color_a
            } else {
                color_b
            };
            // Measure distance to the nearest cell boundary.
            let fx = p.x.fract().min(1.0 - p.x.fract());
            let fy = p.y.fract().min(1.0 - p.y.fract());
            let d = fx.min(fy);
            // Darken lines without changing the checker palette.
            let interior = smoothstep(0.0, params.y.max(1e-4), d);
            let shade = 0.5 + 0.5 * interior;
            Vec4::new(base.x * shade, base.y * shade, base.z * shade, base.w)
        }
        BAKE_RING => {
            let r = (uv - Vec2::splat(0.5)).length();
            let feather = params.z.max(1e-4);
            let band = smoothstep(params.x - feather, params.x, r)
                * (1.0 - smoothstep(params.y, params.y + feather, r));
            color_b.lerp(color_a, band)
        }
        BAKE_DOT => {
            let r = (uv - Vec2::splat(0.5)).length();
            let feather = params.y.max(1e-4);
            let disc = 1.0 - smoothstep(params.x - feather, params.x, r);
            color_b.lerp(color_a, disc)
        }
        // Make unknown generators conspicuous.
        _ => Vec4::new(1.0, 0.0, 1.0, 1.0),
    }
}
