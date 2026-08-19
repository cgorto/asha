//! UI quad and box-shadow entry points. Rounded-rect, border, gradient, and
//! icon shading remains in `abi_ui`; these entries marshal vertices and sample.
//!
//! Vertex pulling emits six vertices per quad in the `(tl, bl, br),
//! (tl, br, tr)` triangle list, CCW in +Y-down NDC. `TEXTURED` and `GRADIENT`
//! are mutually exclusive. Flags bits 16–19 (`UI_MODE_MASK`) select the mode;
//! `ui_shade` owns the standard/material dispatch, including the `color2`/`uv`
//! interpretation for nonzero modes, so this ABI stays one call.

use abi_core::GraphicsPush;
use abi_ui::{
    UI_FLAG_TEXTURED, UiDraw, UiShadowDraw, UiShadowVertex, UiVertex, ui_flag, ui_shade,
    ui_shadow_point, ui_shadow_shade,
};
use glam::{Vec2, Vec4};
use spirv_std::image::Image2d;
use spirv_std::{RuntimeArray, Sampler, spirv};

/// Returns the triangle-list corner for a quad vertex.
fn ui_corner(k: u32) -> u32 {
    match k {
        0 | 3 => 0, // tl
        1 => 3,     // bl
        2 | 4 => 2, // br
        _ => 1,     // tr
    }
}

#[spirv(vertex)]
pub fn ui_vert(
    #[spirv(push_constant)] push: &GraphicsPush,
    #[spirv(vertex_index)] vert_id: i32,
    #[spirv(position)] out_pos: &mut Vec4,
    out_uv: &mut Vec2,
    out_point: &mut Vec2,
    #[spirv(flat)] out_color: &mut Vec4,
    #[spirv(flat)] out_color2: &mut Vec4,
    #[spirv(flat)] out_radius: &mut Vec4,
    #[spirv(flat)] out_border: &mut Vec4,
    #[spirv(flat)] out_size: &mut Vec2,
    #[spirv(flat)] out_flags: &mut u32,
    #[spirv(flat)] out_tex_slot: &mut u32,
) {
    let data = push.vert::<UiDraw>();
    let vid = vert_id as u32;
    let quad = vid / 6;
    if quad >= data.quad_count {
        *out_pos = Vec4::ZERO;
        *out_uv = Vec2::ZERO;
        *out_point = Vec2::ZERO;
        *out_color = Vec4::ZERO;
        *out_color2 = Vec4::ZERO;
        *out_radius = Vec4::ZERO;
        *out_border = Vec4::ZERO;
        *out_size = Vec2::ZERO;
        *out_flags = 0;
        *out_tex_slot = 0;
        return;
    }
    let v: UiVertex = data.vertices[quad * 4 + ui_corner(vid % 6)];

    let view = Vec4::from_array(data.view);
    let px = Vec2::from_array(v.pos);
    let clip = px * Vec2::new(view.x, view.y) + Vec2::new(view.z, view.w);
    *out_pos = Vec4::new(clip.x, clip.y, 0.0, 1.0);
    *out_uv = Vec2::from_array(v.uv);
    *out_point = Vec2::from_array(v.point);
    *out_color = Vec4::from_array(v.color);
    *out_color2 = Vec4::from_array(v.color2);
    *out_radius = Vec4::from_array(v.radius);
    *out_border = Vec4::from_array(v.border);
    *out_size = Vec2::from_array(v.size);
    *out_flags = v.flags;
    *out_tex_slot = v.tex_slot;
}

#[spirv(fragment)]
pub fn ui_frag(
    #[spirv(push_constant)] push: &GraphicsPush,
    #[spirv(descriptor_set = 0, binding = 0)] textures: &RuntimeArray<Image2d>,
    #[spirv(descriptor_set = 2, binding = 0)] samplers: &RuntimeArray<Sampler>,
    uv: Vec2,
    point: Vec2,
    #[spirv(flat)] color: Vec4,
    #[spirv(flat)] color2: Vec4,
    #[spirv(flat)] radius: Vec4,
    #[spirv(flat)] border: Vec4,
    #[spirv(flat)] size: Vec2,
    #[spirv(flat)] flags: u32,
    #[spirv(flat)] tex_slot: u32,
    out_color: &mut Vec4,
) {
    let data = push.frag::<UiDraw>();
    let mut base = color;
    if ui_flag(flags, UI_FLAG_TEXTURED) {
        let image = unsafe { textures.index(tex_slot as usize) };
        let sampler = *unsafe { samplers.index(data.sampler_slot as usize) };
        let sample: Vec4 = image.sample_by_lod(sampler, uv, 0.0);
        base *= sample;
    }
    *out_color = ui_shade(base, color2, uv, point, size, radius, border, flags);
}

/// Emits box-shadow vertices using the shared six-vertex quad layout.
/// Shadow coverage uses an analytic Gaussian integral, not an SDF.
/// The point is reconstructed from `uv` and bounds for ABI compatibility.
#[spirv(vertex)]
pub fn ui_shadow_vert(
    #[spirv(push_constant)] push: &GraphicsPush,
    #[spirv(vertex_index)] vert_id: i32,
    #[spirv(position)] out_pos: &mut Vec4,
    out_point: &mut Vec2,
    #[spirv(flat)] out_color: &mut Vec4,
    #[spirv(flat)] out_size: &mut Vec2,
    #[spirv(flat)] out_radius: &mut Vec4,
    #[spirv(flat)] out_blur: &mut f32,
) {
    let data = push.vert::<UiShadowDraw>();
    let vid = vert_id as u32;
    let quad = vid / 6;
    if quad >= data.quad_count {
        *out_pos = Vec4::ZERO;
        *out_point = Vec2::ZERO;
        *out_color = Vec4::ZERO;
        *out_size = Vec2::ZERO;
        *out_radius = Vec4::ZERO;
        *out_blur = 0.0;
        return;
    }
    let v: UiShadowVertex = data.vertices[quad * 4 + ui_corner(vid % 6)];

    let view = Vec4::from_array(data.view);
    let px = Vec2::from_array(v.pos);
    let clip = px * Vec2::new(view.x, view.y) + Vec2::new(view.z, view.w);
    *out_pos = Vec4::new(clip.x, clip.y, 0.0, 1.0);
    *out_point = ui_shadow_point(Vec2::from_array(v.uv), Vec2::from_array(v.bounds));
    *out_color = Vec4::from_array(v.color);
    *out_size = Vec2::from_array(v.size);
    *out_radius = Vec4::from_array(v.radius);
    *out_blur = v.blur;
}

#[spirv(fragment)]
pub fn ui_shadow_frag(
    point: Vec2,
    #[spirv(flat)] color: Vec4,
    #[spirv(flat)] size: Vec2,
    #[spirv(flat)] radius: Vec4,
    #[spirv(flat)] blur: f32,
    out_color: &mut Vec4,
) {
    *out_color = ui_shadow_shade(color, point, size, radius, blur);
}
