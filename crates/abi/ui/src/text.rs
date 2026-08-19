//! Slug text GPU contracts.
//!
//! Text data is buffer-backed: glyph descriptors, curves, band headers, band
//! curve indices, and per-draw instances all arrive through BDA pointers.

use crate::{GpuPtr, gpu_data};

/// Per-glyph Slug descriptor, indexed by `TextGlyphInstance::glyph_id`.
#[gpu_data]
pub struct TextGlyphDescriptor {
    pub bbox_em: [f32; 4],
    pub band_scale: [f32; 2],
    pub band_offset: [f32; 2],
    pub hband_base: u32,
    pub vband_base: u32,
    /// Low 16 bits = vertical/X band max, high 16 bits = horizontal/Y band max.
    pub band_max: u32,
    pub _pad0: u32,
}

const _: () = assert!(core::mem::size_of::<TextGlyphDescriptor>() == 48);
const _: () = assert!(core::mem::align_of::<TextGlyphDescriptor>() == 4);
const _: () = assert!(core::mem::offset_of!(TextGlyphDescriptor, bbox_em) == 0);
const _: () = assert!(core::mem::offset_of!(TextGlyphDescriptor, band_scale) == 16);
const _: () = assert!(core::mem::offset_of!(TextGlyphDescriptor, band_offset) == 24);
const _: () = assert!(core::mem::offset_of!(TextGlyphDescriptor, hband_base) == 32);
const _: () = assert!(core::mem::offset_of!(TextGlyphDescriptor, vband_base) == 36);
const _: () = assert!(core::mem::offset_of!(TextGlyphDescriptor, band_max) == 40);
const _: () = assert!(core::mem::offset_of!(TextGlyphDescriptor, _pad0) == 44);

/// One text draw instance: baseline pen position, glyph index, and packed color.
#[gpu_data]
pub struct TextGlyphInstance {
    pub pen_doc: [f32; 2],
    pub glyph_id: u32,
    pub color: u32,
}

const _: () = assert!(core::mem::size_of::<TextGlyphInstance>() == 16);
const _: () = assert!(core::mem::align_of::<TextGlyphInstance>() == 4);
const _: () = assert!(core::mem::offset_of!(TextGlyphInstance, pen_doc) == 0);
const _: () = assert!(core::mem::offset_of!(TextGlyphInstance, glyph_id) == 8);
const _: () = assert!(core::mem::offset_of!(TextGlyphInstance, color) == 12);

/// Header for a horizontal or vertical Slug band.
#[gpu_data]
pub struct TextBandHeader {
    pub first: u32,
    pub count: u32,
}

const _: () = assert!(core::mem::size_of::<TextBandHeader>() == 8);
const _: () = assert!(core::mem::align_of::<TextBandHeader>() == 4);
const _: () = assert!(core::mem::offset_of!(TextBandHeader, first) == 0);
const _: () = assert!(core::mem::offset_of!(TextBandHeader, count) == 4);

/// Quadratic curve control points in glyph-em coordinates.
#[gpu_data]
pub struct TextCurve {
    pub p1: [f32; 2],
    pub p2: [f32; 2],
    pub p3: [f32; 2],
}

const _: () = assert!(core::mem::size_of::<TextCurve>() == 24);
const _: () = assert!(core::mem::align_of::<TextCurve>() == 4);
const _: () = assert!(core::mem::offset_of!(TextCurve, p1) == 0);
const _: () = assert!(core::mem::offset_of!(TextCurve, p2) == 8);
const _: () = assert!(core::mem::offset_of!(TextCurve, p3) == 16);

/// Per-draw text camera.
#[gpu_data]
pub struct TextCamera {
    /// `clip_xy = doc_xy * xform.xy + xform.zw`.
    pub xform: [f32; 4],
    /// Screen pixels per document unit.
    pub zoom: f32,
    /// Document units per glyph em before camera zoom.
    pub font_px_per_em: f32,
    pub _pad0: [f32; 2],
}

const _: () = assert!(core::mem::size_of::<TextCamera>() == 32);
const _: () = assert!(core::mem::align_of::<TextCamera>() == 4);
const _: () = assert!(core::mem::offset_of!(TextCamera, xform) == 0);
const _: () = assert!(core::mem::offset_of!(TextCamera, zoom) == 16);
const _: () = assert!(core::mem::offset_of!(TextCamera, font_px_per_em) == 20);
const _: () = assert!(core::mem::offset_of!(TextCamera, _pad0) == 24);

/// Per-frame/per-batch text draw payload shared by text vertex and fragment stages.
#[gpu_data]
pub struct TextDraw {
    pub instances: GpuPtr<TextGlyphInstance>,
    pub descriptors: GpuPtr<TextGlyphDescriptor>,
    pub curves: GpuPtr<TextCurve>,
    pub bands: GpuPtr<TextBandHeader>,
    pub band_curve_indices: GpuPtr<u32>,
    pub camera: TextCamera,
    pub glyph_count: u32,
    pub flags: u32,
    pub _pad0: [u32; 2],
}

const _: () = assert!(core::mem::size_of::<TextDraw>() == 88);
const _: () = assert!(core::mem::align_of::<TextDraw>() == 4);
const _: () = assert!(core::mem::offset_of!(TextDraw, instances) == 0);
const _: () = assert!(core::mem::offset_of!(TextDraw, descriptors) == 8);
const _: () = assert!(core::mem::offset_of!(TextDraw, curves) == 16);
const _: () = assert!(core::mem::offset_of!(TextDraw, bands) == 24);
const _: () = assert!(core::mem::offset_of!(TextDraw, band_curve_indices) == 32);
const _: () = assert!(core::mem::offset_of!(TextDraw, camera) == 40);
const _: () = assert!(core::mem::offset_of!(TextDraw, glyph_count) == 72);
const _: () = assert!(core::mem::offset_of!(TextDraw, flags) == 76);
const _: () = assert!(core::mem::offset_of!(TextDraw, _pad0) == 80);
