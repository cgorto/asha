use abi_core::{GpuPtr, GraphicsPush};
use abi_ui::{
    TextBandHeader, TextCamera, TextCurve, TextDraw, TextGlyphDescriptor, TextGlyphInstance,
};
use glam::{Vec2, Vec3, Vec4};
use spirv_std::arch::Derivative;
use spirv_std::num_traits::Float;
use spirv_std::spirv;

const TEXT_DILATE_PX: f32 = 1.0;
const TEXT_SUBPIXEL_LCD: bool = true;
const TEXT_SUBPIXEL_BGR: bool = false;
const TEXT_WEIGHT_BOOST: bool = false;

fn text_srgb_to_linear_channel(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

fn text_unpack_color(c: u32) -> Vec4 {
    let scale = 1.0 / 255.0;
    let s = Vec4::new(
        (c & 0xff) as f32 * scale,
        ((c >> 8) & 0xff) as f32 * scale,
        ((c >> 16) & 0xff) as f32 * scale,
        ((c >> 24) & 0xff) as f32 * scale,
    );
    Vec4::new(
        text_srgb_to_linear_channel(s.x),
        text_srgb_to_linear_channel(s.y),
        text_srgb_to_linear_channel(s.z),
        s.w,
    )
}

fn text_corner(vert_id: i32) -> Vec2 {
    Vec2::new((vert_id & 1) as f32, ((vert_id >> 1) & 1) as f32)
}

fn text_vertex(
    camera: &TextCamera,
    inst: TextGlyphInstance,
    desc: TextGlyphDescriptor,
    vert_id: i32,
) -> (Vec4, Vec4, Vec2, Vec4, u32, u32, u32) {
    let corner = text_corner(vert_id);
    let bbox = Vec4::from_array(desc.bbox_em);
    let em = Vec2::new(
        bbox.x + (bbox.z - bbox.x) * corner.x,
        bbox.y + (bbox.w - bbox.y) * corner.y,
    );
    let outdir = corner * 2.0 - Vec2::ONE;
    let ems_per_pixel = 1.0 / (camera.font_px_per_em * camera.zoom);
    let render_coord = em + outdir * (ems_per_pixel * TEXT_DILATE_PX);

    let pen = Vec2::from_array(inst.pen_doc);
    let doc = pen
        + Vec2::new(em.x, -em.y) * camera.font_px_per_em
        + Vec2::new(outdir.x, -outdir.y) * (TEXT_DILATE_PX / camera.zoom);
    let clip = doc * Vec2::new(camera.xform[0], camera.xform[1])
        + Vec2::new(camera.xform[2], camera.xform[3]);
    (
        Vec4::new(clip.x, clip.y, 0.0, 1.0),
        text_unpack_color(inst.color),
        render_coord,
        Vec4::new(
            desc.band_scale[0],
            desc.band_scale[1],
            desc.band_offset[0],
            desc.band_offset[1],
        ),
        desc.hband_base,
        desc.vband_base,
        desc.band_max,
    )
}

fn text_calc_root_code(y1: f32, y2: f32, y3: f32) -> u32 {
    let i1 = y1.to_bits() >> 31;
    let i2 = y2.to_bits() >> 30;
    let i3 = y3.to_bits() >> 29;
    let mut shift = (i2 & 2) | (i1 & !2);
    shift = (i3 & 4) | (shift & !4);
    (0x2e74u32 >> shift) & 0x0101u32
}

fn text_solve_horiz_poly(p12: Vec4, p3: Vec2) -> Vec2 {
    let p1 = Vec2::new(p12.x, p12.y);
    let p2 = Vec2::new(p12.z, p12.w);
    let a = p1 - p2 * 2.0 + p3;
    let b = p1 - p2;
    let ra = 1.0 / a.y;
    let rb = 0.5 / b.y;
    let d = (b.y * b.y - a.y * p12.y).max(0.0).sqrt();
    let mut t1 = (b.y - d) * ra;
    let mut t2 = (b.y + d) * ra;
    if a.y.abs() < 1.0 / 65536.0 {
        t1 = p12.y * rb;
        t2 = t1;
    }
    Vec2::new(
        (a.x * t1 - b.x * 2.0) * t1 + p12.x,
        (a.x * t2 - b.x * 2.0) * t2 + p12.x,
    )
}

fn text_solve_vert_poly(p12: Vec4, p3: Vec2) -> Vec2 {
    let p1 = Vec2::new(p12.x, p12.y);
    let p2 = Vec2::new(p12.z, p12.w);
    let a = p1 - p2 * 2.0 + p3;
    let b = p1 - p2;
    let ra = 1.0 / a.x;
    let rb = 0.5 / b.x;
    let d = (b.x * b.x - a.x * p12.x).max(0.0).sqrt();
    let mut t1 = (b.x - d) * ra;
    let mut t2 = (b.x + d) * ra;
    if a.x.abs() < 1.0 / 65536.0 {
        t1 = p12.x * rb;
        t2 = t1;
    }
    Vec2::new(
        (a.y * t1 - b.y * 2.0) * t1 + p12.y,
        (a.y * t2 - b.y * 2.0) * t2 + p12.y,
    )
}

fn text_calc_coverage(xcov: f32, ycov: f32, xwgt: f32, ywgt: f32) -> f32 {
    let mut coverage = ((xcov * xwgt + ycov * ywgt).abs() / (xwgt + ywgt).max(1.0 / 65536.0))
        .max(xcov.abs().min(ycov.abs()))
        .clamp(0.0, 1.0);
    if TEXT_WEIGHT_BOOST {
        coverage = coverage.sqrt();
    }
    coverage
}

fn text_curve_relative(curve: TextCurve, render_coord: Vec2) -> (Vec4, Vec2) {
    let p1 = Vec2::from_array(curve.p1);
    let p2 = Vec2::from_array(curve.p2);
    let p3 = Vec2::from_array(curve.p3);
    (
        Vec4::new(
            p1.x - render_coord.x,
            p1.y - render_coord.y,
            p2.x - render_coord.x,
            p2.y - render_coord.y,
        ),
        p3 - render_coord,
    )
}

fn text_hband_coverage(
    data: GpuPtr<TextDraw>,
    band: TextBandHeader,
    render_coord: Vec2,
    pixels_per_em: Vec2,
) -> (f32, f32) {
    let mut xcov = 0.0f32;
    let mut xwgt = 0.0f32;
    let mut i = 0u32;
    while i < band.count {
        let curve_index = data.band_curve_indices[band.first + i];
        let (p12, p3) = text_curve_relative(data.curves[curve_index], render_coord);
        if p12.x.max(p12.z).max(p3.x) * pixels_per_em.x < -0.5 {
            break;
        }
        let code = text_calc_root_code(p12.y, p12.w, p3.y);
        if code != 0 {
            let r = text_solve_horiz_poly(p12, p3) * pixels_per_em.x;
            if (code & 1) != 0 {
                xcov += (r.x + 0.5).clamp(0.0, 1.0);
                xwgt = xwgt.max(1.0 - (r.x.abs() * 2.0).clamp(0.0, 1.0));
            }
            if code > 1 {
                xcov -= (r.y + 0.5).clamp(0.0, 1.0);
                xwgt = xwgt.max(1.0 - (r.y.abs() * 2.0).clamp(0.0, 1.0));
            }
        }
        i += 1;
    }
    (xcov, xwgt)
}

fn text_vband_coverage(
    data: GpuPtr<TextDraw>,
    band: TextBandHeader,
    render_coord: Vec2,
    pixels_per_em: Vec2,
) -> (f32, f32) {
    let mut ycov = 0.0f32;
    let mut ywgt = 0.0f32;
    let mut i = 0u32;
    while i < band.count {
        let curve_index = data.band_curve_indices[band.first + i];
        let (p12, p3) = text_curve_relative(data.curves[curve_index], render_coord);
        if p12.y.max(p12.w).max(p3.y) * pixels_per_em.y < -0.5 {
            break;
        }
        let code = text_calc_root_code(p12.x, p12.z, p3.x);
        if code != 0 {
            let r = text_solve_vert_poly(p12, p3) * pixels_per_em.y;
            if (code & 1) != 0 {
                ycov -= (r.x + 0.5).clamp(0.0, 1.0);
                ywgt = ywgt.max(1.0 - (r.x.abs() * 2.0).clamp(0.0, 1.0));
            }
            if code > 1 {
                ycov += (r.y + 0.5).clamp(0.0, 1.0);
                ywgt = ywgt.max(1.0 - (r.y.abs() * 2.0).clamp(0.0, 1.0));
            }
        }
        i += 1;
    }
    (ycov, ywgt)
}

// Slug glyph coverage math is ported from Eric Lengyel's reference shader
// (text root solving, band coverage, and horizontal/vertical combination).
// "Slug shader code Copyright 2017 by Eric Lengyel."
// SPDX-License-Identifier: MIT OR Apache-2.0; patent dedicated to the public domain.
fn text_slug_coverage(
    data: GpuPtr<TextDraw>,
    render_coord: Vec2,
    pixels_per_em: Vec2,
    banding: Vec4,
    hband_base: u32,
    vband_base: u32,
    band_max: u32,
) -> f32 {
    let v_max = band_max & 0xffff;
    let h_max = band_max >> 16;
    let v_band = (render_coord.x * banding.x + banding.z).clamp(0.0, v_max as f32) as u32;
    let h_band = (render_coord.y * banding.y + banding.w).clamp(0.0, h_max as f32) as u32;

    let hband = data.bands[hband_base + h_band];
    let vband = data.bands[vband_base + v_band];
    let (xcov, xwgt) = text_hband_coverage(data, hband, render_coord, pixels_per_em);
    let (ycov, ywgt) = text_vband_coverage(data, vband, render_coord, pixels_per_em);
    text_calc_coverage(xcov, ycov, xwgt, ywgt)
}

fn text_lcd_coverage(
    data: GpuPtr<TextDraw>,
    render_coord: Vec2,
    banding: Vec4,
    hband_base: u32,
    vband_base: u32,
    band_max: u32,
) -> Vec3 {
    let ems_per_pixel = render_coord.fwidth();
    let pixels_per_em = Vec2::ONE / ems_per_pixel;
    let center = text_slug_coverage(
        data,
        render_coord,
        pixels_per_em,
        banding,
        hband_base,
        vband_base,
        band_max,
    );
    if !TEXT_SUBPIXEL_LCD {
        return Vec3::splat(center);
    }

    let dx = ems_per_pixel.x * (1.0 / 3.0);
    let l2 = text_slug_coverage(
        data,
        render_coord - Vec2::new(2.0 * dx, 0.0),
        pixels_per_em,
        banding,
        hband_base,
        vband_base,
        band_max,
    );
    let l1 = text_slug_coverage(
        data,
        render_coord - Vec2::new(dx, 0.0),
        pixels_per_em,
        banding,
        hband_base,
        vband_base,
        band_max,
    );
    let r1 = text_slug_coverage(
        data,
        render_coord + Vec2::new(dx, 0.0),
        pixels_per_em,
        banding,
        hband_base,
        vband_base,
        band_max,
    );
    let r2 = text_slug_coverage(
        data,
        render_coord + Vec2::new(2.0 * dx, 0.0),
        pixels_per_em,
        banding,
        hband_base,
        vband_base,
        band_max,
    );
    let third = 1.0 / 3.0;
    let cov = Vec3::new(
        (l2 + l1 + center) * third,
        (l1 + center + r1) * third,
        (center + r1 + r2) * third,
    );
    if TEXT_SUBPIXEL_BGR {
        Vec3::new(cov.z, cov.y, cov.x)
    } else {
        cov
    }
}

#[spirv(vertex)]
pub fn text_vert(
    #[spirv(push_constant)] push: &GraphicsPush,
    #[spirv(vertex_index)] vert_id: i32,
    #[spirv(instance_index)] inst_id: i32,
    #[spirv(position)] out_pos: &mut Vec4,
    #[spirv(flat)] out_color: &mut Vec4,
    out_render_coord: &mut Vec2,
    #[spirv(flat)] out_banding: &mut Vec4,
    #[spirv(flat)] out_hband_base: &mut u32,
    #[spirv(flat)] out_vband_base: &mut u32,
    #[spirv(flat)] out_band_max: &mut u32,
) {
    let data = push.vert::<TextDraw>();
    if inst_id as u32 >= data.glyph_count {
        *out_pos = Vec4::ZERO;
        *out_color = Vec4::ZERO;
        *out_render_coord = Vec2::ZERO;
        *out_banding = Vec4::ZERO;
        *out_hband_base = 0;
        *out_vband_base = 0;
        *out_band_max = 0;
        return;
    }
    let inst = data.instances[inst_id];
    let desc = data.descriptors[inst.glyph_id];
    let (pos, color, render_coord, banding, hband_base, vband_base, band_max) =
        text_vertex(&data.camera, inst, desc, vert_id);
    *out_pos = pos;
    *out_color = color;
    *out_render_coord = render_coord;
    *out_banding = banding;
    *out_hband_base = hband_base;
    *out_vband_base = vband_base;
    *out_band_max = band_max;
}

#[spirv(fragment)]
pub fn text_cover_frag(
    #[spirv(push_constant)] push: &GraphicsPush,
    #[spirv(flat)] color: Vec4,
    render_coord: Vec2,
    #[spirv(flat)] banding: Vec4,
    #[spirv(flat)] hband_base: u32,
    #[spirv(flat)] vband_base: u32,
    #[spirv(flat)] band_max: u32,
    out_color: &mut Vec4,
) {
    let data = push.frag::<TextDraw>();
    let cov = text_lcd_coverage(
        data,
        render_coord,
        banding,
        hband_base,
        vband_base,
        band_max,
    );
    let ca = cov * color.w;
    *out_color = Vec4::new(ca.x, ca.y, ca.z, ca.max_element());
}

#[spirv(fragment)]
pub fn text_blend_frag(
    #[spirv(push_constant)] push: &GraphicsPush,
    #[spirv(flat)] color: Vec4,
    render_coord: Vec2,
    #[spirv(flat)] banding: Vec4,
    #[spirv(flat)] hband_base: u32,
    #[spirv(flat)] vband_base: u32,
    #[spirv(flat)] band_max: u32,
    out_color: &mut Vec4,
) {
    let data = push.frag::<TextDraw>();
    let cov = text_lcd_coverage(
        data,
        render_coord,
        banding,
        hband_base,
        vband_base,
        band_max,
    );
    let ca = cov * color.w;
    *out_color = Vec4::new(
        color.x * ca.x,
        color.y * ca.y,
        color.z * ca.z,
        ca.dot(Vec3::splat(1.0 / 3.0)),
    );
}
