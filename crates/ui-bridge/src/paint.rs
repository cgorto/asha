//! Converts laid-out UI into sorted `abi_ui` paint streams.
//!
//! The walker follows `bevy_ui_render` 0.19's extraction/prepare geometry
//! (`bevy_ui_render/src/lib.rs` and `bevy_ui_render/src/gradient.rs`),
//! adapted to this crate's CPU streams and `ui` pass.
//! Deliberate deviations are one target with no render-world camera routing,
//! outlines as border rings rather than `INVERT`,
//! ordered stops without upstream's defensive re-sort, and sRGB fallback for
//! unsupported gradient spaces. Images cover only the three embedded icons;
//! atlas rectangles, flips, and other image modes are not ported.

use std::ops::Range;
use std::sync::atomic::{AtomicBool, Ordering};

use bevy::camera::visibility::InheritedVisibility;
use bevy::input_focus::InputFocus;
use bevy::math::{Affine2, IRect, Rect, Vec2, Vec4};
use bevy::prelude::*;
use bevy::text::{
    ComputedTextBlock, EditableText, Strikethrough, StrikethroughColor, TextBrush, TextColor,
    TextCursorStyle, TextLayoutInfo, Underline, UnderlineColor,
};
use bevy::ui::widget::{ImageNode, TextScroll};
use bevy::ui::{ComputedStackIndex, UiSystems};
use bevy_color::{Alpha, Color, Hsla, LinearRgba, Srgba};
use parley::PositionedLayoutItem;
use smallvec::SmallVec;

use abi_ui::{
    UI_FLAG_BORDER_ANY, UI_FLAG_BORDER_BOTTOM, UI_FLAG_BORDER_LEFT, UI_FLAG_BORDER_RIGHT,
    UI_FLAG_BORDER_TOP, UI_FLAG_FILL_END, UI_FLAG_FILL_START, UI_FLAG_GRADIENT,
    UI_FLAG_GRADIENT_SPACE_HSLA, UI_FLAG_TEXTURED, UI_MODE_SHIFT, UiMaterialData, UiShadowVertex,
    UiVertex,
};
use text::TextGlyphInstance;

use crate::glyphs::GlyphOutlineProvider;
use crate::gradient::{compute_gradient_line_length, resolve_gradient_stops};
use crate::icons::IconRegistry;
use crate::material::UiMaterialTag;

/// UI painter-order offsets matching the bevy UI table.
pub mod stack_z_offsets {
    pub const BOX_SHADOW: f32 = -0.1;
    pub const BACKGROUND: f32 = 0.0;
    pub const BORDER: f32 = 0.01;
    pub const GRADIENT: f32 = 0.02;
    pub const BORDER_GRADIENT: f32 = 0.03;
    pub const IMAGE: f32 = 0.04;
    pub const MATERIAL: f32 = 0.05;
    pub const TEXT_SELECTION: f32 = 0.055;
    pub const TEXT: f32 = 0.06;
    pub const TEXT_STRIKETHROUGH: f32 = 0.07;
    pub const TEXT_CURSOR: f32 = 0.08;
}

/// TL, TR, BR, BL offsets around a node center.
pub(crate) const QUAD_VERTEX_POSITIONS: [Vec2; 4] = [
    Vec2::new(-0.5, -0.5),
    Vec2::new(0.5, -0.5),
    Vec2::new(0.5, 0.5),
    Vec2::new(-0.5, 0.5),
];

/// Full-image UVs in TL, TR, BR, BL order; positive V points down.
const QUAD_VERTEX_UVS: [Vec2; 4] = [
    Vec2::new(0.0, 0.0),
    Vec2::new(1.0, 0.0),
    Vec2::new(1.0, 1.0),
    Vec2::new(0.0, 1.0),
];

/// Retained per-frame UI vertices and painter-order batches.
#[derive(Resource, Default)]
pub struct UiPaintList {
    /// Four vertices per quad, in painter's order.
    pub vertices: Vec<UiVertex>,
    /// Quad ranges split around interleaved text and shadows.
    pub batches: Vec<UiBatch>,
    /// Number of quads in `vertices`.
    pub quad_count: usize,
    /// Box-shadow vertices in painter's order.
    pub shadow_vertices: Vec<UiShadowVertex>,
    /// Shadow quad ranges; their `order` values form a parallel stream for
    /// merging with quad and text batches.
    pub shadow_batches: Vec<UiShadowBatch>,
    /// `shadow_vertices.len() / 4`.
    pub shadow_quad_count: usize,
}

/// A UI quad range and optional scissor rectangle.
#[derive(Clone, Debug, PartialEq)]
pub struct UiBatch {
    pub quad_range: Range<usize>,
    pub scissor: Option<IRect>,
    /// Shared order for interleaving UI, shadow, and text batches.
    pub order: u32,
}

/// A shadow-vertex range with a parallel painter-order descriptor.
#[derive(Clone, Debug, PartialEq)]
pub struct UiShadowBatch {
    pub quad_range: Range<usize>,
    pub scissor: Option<IRect>,
    /// Shared global sequence used to merge render batches.
    pub order: u32,
}

/// Retained text instances and painter-order batches.
#[derive(Resource, Default)]
pub struct TextPaintList {
    pub instances: Vec<TextGlyphInstance>,
    pub batches: Vec<TextRunBatch>,
}

/// Text instances grouped by font, size, and clip.
#[derive(Clone, Debug, PartialEq)]
pub struct TextRunBatch {
    pub instance_range: Range<usize>,
    pub clip: Option<Rect>,
    pub blob_id: u64,
    pub font_index: u32,
    pub font_px_per_em: f32,
    /// See [`UiBatch::order`] — the same global sequence.
    pub order: u32,
}

/// A sortable paint primitive.
struct PaintItem {
    z_order: f32,
    payload: PaintPayload,
}

enum PaintPayload {
    Quad(QuadRecipe),
    Gradient(GradientRecipe),
    /// A text-run instance range and batch key.
    TextRun(TextRunRecipe),
    /// A material-tagged node.
    Material(MaterialRecipe),
    /// One resolved box-shadow layer.
    Shadow(ShadowRecipe),
}

/// Text-run paint metadata.
struct TextRunRecipe {
    range: Range<usize>,
    clip: Option<Rect>,
    blob_id: u64,
    font_index: u32,
    font_px_per_em: f32,
}

/// A flat-color, bordered, outlined, or tinted quad.
struct QuadRecipe {
    transform: Affine2,
    size: Vec2,
    clip: Option<Rect>,
    color: [f32; 4],
    radius: [f32; 4],
    border: [f32; 4],
    flags: u32,
    tex_slot: u32,
    /// Per-corner UVs; textured quads use the full image.
    uv: [Vec2; 4],
}

/// Gradient geometry split into adjacent stop-pair segments.
struct GradientRecipe {
    transform: Affine2,
    size: Vec2,
    clip: Option<Rect>,
    radius: [f32; 4],
    border: [f32; 4],
    /// Border flags combined with gradient flags during emission.
    base_flags: u32,
    g_start: Vec2,
    g_dir: Vec2,
    segments: SmallVec<[GradientSegment; 4]>,
}

struct GradientSegment {
    start_color: [f32; 4],
    end_color: [f32; 4],
    start_len: f32,
    end_len: f32,
    hsla: bool,
}

/// Material quad geometry and packed shader parameters.
struct MaterialRecipe {
    transform: Affine2,
    size: Vec2,
    clip: Option<Rect>,
    radius: [f32; 4],
    mode: u32,
    data: UiMaterialData,
}

/// Resolved shadow geometry with blur padding.
///
/// `bounds = size + 6 * blur`; `size` includes spread.
struct ShadowRecipe {
    transform: Affine2,
    bounds: Vec2,
    size: Vec2,
    clip: Option<Rect>,
    color: [f32; 4],
    radius: [f32; 4],
    blur: f32,
}

fn linear_rgba_arr(c: LinearRgba) -> [f32; 4] {
    [c.red, c.green, c.blue, c.alpha]
}

fn color_to_linear_arr(c: Color) -> [f32; 4] {
    linear_rgba_arr(c.to_linear())
}

fn border_arr(b: BorderRect) -> [f32; 4] {
    [b.min_inset.x, b.min_inset.y, b.max_inset.x, b.max_inset.y]
}

const BORDER_EDGE_FLAGS: [u32; 4] = [
    UI_FLAG_BORDER_LEFT,
    UI_FLAG_BORDER_TOP,
    UI_FLAG_BORDER_RIGHT,
    UI_FLAG_BORDER_BOTTOM,
];

/// Groups equal-colored border edges into draw items.
fn push_border_items(
    items: &mut Vec<PaintItem>,
    node: &ComputedNode,
    transform: Affine2,
    clip: Option<Rect>,
    stack_index: u32,
    border_color: &BorderColor,
) {
    if node.border() == BorderRect::ZERO {
        return;
    }

    let colors = [
        border_color.left.to_linear(),
        border_color.top.to_linear(),
        border_color.right.to_linear(),
        border_color.bottom.to_linear(),
    ];

    let mut completed_flags = 0u32;
    for (i, &color) in colors.iter().enumerate() {
        if color.is_fully_transparent() {
            continue;
        }

        let mut flags = BORDER_EDGE_FLAGS[i];
        if completed_flags & flags != 0 {
            continue;
        }

        for j in (i + 1)..4 {
            if color == colors[j] {
                flags |= BORDER_EDGE_FLAGS[j];
            }
        }
        completed_flags |= flags;

        items.push(PaintItem {
            z_order: stack_index as f32 + stack_z_offsets::BORDER,
            payload: PaintPayload::Quad(QuadRecipe {
                transform,
                size: node.size(),
                clip,
                color: linear_rgba_arr(color),
                radius: node.border_radius().into(),
                border: border_arr(node.border()),
                flags,
                tex_slot: 0,
                uv: [Vec2::ZERO; 4],
            }),
        });
    }
}

/// Emits an outline using the border-SDF ring technique.
fn push_outline_item(
    items: &mut Vec<PaintItem>,
    node: &ComputedNode,
    transform: Affine2,
    clip: Option<Rect>,
    stack_index: u32,
    outline: &Outline,
) {
    if node.outline_width() <= 0.0 || outline.color.is_fully_transparent() {
        return;
    }

    items.push(PaintItem {
        z_order: stack_index as f32 + stack_z_offsets::BORDER,
        payload: PaintPayload::Quad(QuadRecipe {
            transform,
            size: node.outlined_node_size(),
            clip,
            color: color_to_linear_arr(outline.color),
            radius: node.outline_radius().into(),
            border: [node.outline_width(); 4],
            flags: UI_FLAG_BORDER_ANY,
            tex_slot: 0,
            uv: [Vec2::ZERO; 4],
        }),
    });
}

/// Resolves a shadow value against node and viewport dimensions.
///
/// `Val::Px` scales by the target factor; `Val::Percent` uses `base`;
/// `Val::Vw`/`Val::Vh` use viewport width/height; `Val::VMin`/`Val::VMax`
/// use the viewport's smaller/larger dimension. `Val::Auto` is zero.
fn resolve_shadow_val(val: Val, base: f32, scale_factor: f32, viewport: Vec2) -> f32 {
    match val {
        Val::Auto => 0.0,
        Val::Px(px) => px * scale_factor,
        Val::Percent(percent) => percent / 100.0 * base,
        Val::Vw(percent) => percent / 100.0 * viewport.x,
        Val::Vh(percent) => percent / 100.0 * viewport.y,
        Val::VMin(percent) => percent / 100.0 * viewport.x.min(viewport.y),
        Val::VMax(percent) => percent / 100.0 * viewport.x.max(viewport.y),
    }
}

/// Emits nontransparent shadows before each node background.
///
/// Stable sorting preserves declared shadow order.
fn push_shadow_items(
    items: &mut Vec<PaintItem>,
    node: &ComputedNode,
    transform: Affine2,
    clip: Option<Rect>,
    stack_index: u32,
    box_shadow: &BoxShadow,
    target: &ComputedUiRenderTargetInfo,
) {
    let node_size = node.size();
    let scale_factor = 1.0 / node.inverse_scale_factor().max(f32::EPSILON);
    let viewport = target.physical_size().as_vec2();

    for shadow in box_shadow.iter() {
        if shadow.color.is_fully_transparent() {
            continue;
        }

        let spread_x =
            resolve_shadow_val(shadow.spread_radius, node_size.x, scale_factor, viewport);
        let spread_ratio = (spread_x + node_size.x) / node_size.x;
        let spread = Vec2::new(spread_x, node_size.y * spread_ratio - node_size.y);

        let blur_radius =
            resolve_shadow_val(shadow.blur_radius, node_size.x, scale_factor, viewport);
        let offset = Vec2::new(
            resolve_shadow_val(shadow.x_offset, node_size.x, scale_factor, viewport),
            resolve_shadow_val(shadow.y_offset, node_size.y, scale_factor, viewport),
        );

        let shadow_size = node_size + spread;
        if shadow_size.x <= 0.0 || shadow_size.y <= 0.0 {
            continue;
        }

        let node_radius_arr: [f32; 4] = node.border_radius().into();
        let radius = Vec4::from_array(node_radius_arr) * spread_ratio;
        let bounds = shadow_size + Vec2::splat(6.0 * blur_radius);
        let shadow_transform = transform * Affine2::from_translation(offset);

        items.push(PaintItem {
            z_order: stack_index as f32 + stack_z_offsets::BOX_SHADOW,
            payload: PaintPayload::Shadow(ShadowRecipe {
                transform: shadow_transform,
                bounds,
                size: shadow_size,
                clip,
                color: color_to_linear_arr(shadow.color),
                radius: radius.to_array(),
                blur: blur_radius,
            }),
        });
    }
}

/// Packs RGBA sRGB bytes as `R | G<<8 | B<<16 | A<<24`.
///
/// The shader decodes RGB and preserves alpha.
fn pack_text_color(color: Color) -> u32 {
    let srgba: Srgba = color.into();
    let channel = |c: f32| (c.clamp(0.0, 1.0) * 255.0).round() as u32;
    channel(srgba.red)
        | (channel(srgba.green) << 8)
        | (channel(srgba.blue) << 16)
        | (channel(srgba.alpha) << 24)
}

/// Returns a decoration stroke's center and size.
///
/// Coordinates use unscaled, top-left-origin layout space.
fn decoration_rect(
    bounds_center_x: f32,
    bounds_size_x: f32,
    y: f32,
    thickness: f32,
) -> (Vec2, Vec2) {
    (
        Vec2::new(bounds_center_x, y + 0.5 * thickness),
        Vec2::new(bounds_size_x, thickness),
    )
}

/// Emits strikethrough and underline quads for one glyph run.
#[allow(clippy::too_many_arguments)]
fn push_decoration_items(
    items: &mut Vec<PaintItem>,
    content_transform: Affine2,
    clip: Option<Rect>,
    stack_index: u32,
    bounds_center_x: f32,
    bounds_size_x: f32,
    strikethrough_y: f32,
    strikethrough_thickness: f32,
    underline_y: f32,
    underline_thickness: f32,
    has_strikethrough: bool,
    strikethrough_color: Option<Color>,
    has_underline: bool,
    underline_color: Option<Color>,
    fallback_color: Color,
) {
    if has_strikethrough {
        let (center, size) = decoration_rect(
            bounds_center_x,
            bounds_size_x,
            strikethrough_y,
            strikethrough_thickness,
        );
        items.push(PaintItem {
            z_order: stack_index as f32 + stack_z_offsets::TEXT_STRIKETHROUGH,
            payload: PaintPayload::Quad(QuadRecipe {
                transform: content_transform * Affine2::from_translation(center),
                size,
                clip,
                color: color_to_linear_arr(strikethrough_color.unwrap_or(fallback_color)),
                radius: [0.0; 4],
                border: [0.0; 4],
                flags: 0,
                tex_slot: 0,
                uv: [Vec2::ZERO; 4],
            }),
        });
    }
    if has_underline {
        let (center, size) = decoration_rect(
            bounds_center_x,
            bounds_size_x,
            underline_y,
            underline_thickness,
        );
        items.push(PaintItem {
            z_order: stack_index as f32 + stack_z_offsets::TEXT_STRIKETHROUGH,
            payload: PaintPayload::Quad(QuadRecipe {
                transform: content_transform * Affine2::from_translation(center),
                size,
                clip,
                color: color_to_linear_arr(underline_color.unwrap_or(fallback_color)),
                radius: [0.0; 4],
                border: [0.0; 4],
                flags: 0,
                tex_slot: 0,
                uv: [Vec2::ZERO; 4],
            }),
        });
    }
}

/// Emits glyph instances and decorations from a shaped text layout.
///
/// Each glyph run remains one sortable paint item.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn push_text_items(
    items: &mut Vec<PaintItem>,
    text_list: &mut TextPaintList,
    glyph_provider: &mut GlyphOutlineProvider,
    computed_block: &ComputedTextBlock,
    node: &ComputedNode,
    node_affine: Affine2,
    clip: Option<Rect>,
    stack_index: u32,
    node_color: Color,
    text_colors: &Query<&TextColor>,
    text_decorations: &Query<(
        AnyOf<(&Strikethrough, &Underline)>,
        Option<&StrikethroughColor>,
        Option<&UnderlineColor>,
    )>,
) {
    let content_transform = node_affine * Affine2::from_translation(node.content_box().min);

    for line in computed_block.buffer().lines() {
        for item in line.items() {
            let PositionedLayoutItem::GlyphRun(glyph_run) = item else {
                continue;
            };
            let section_index = glyph_run.style().brush.section_index as usize;
            let section_entity = computed_block
                .entities()
                .get(section_index)
                .map(|t| t.entity);
            let color = section_entity
                .and_then(|entity| text_colors.get(entity).ok())
                .map_or(node_color, |text_color| text_color.0);
            let packed_color = pack_text_color(color);

            let run = glyph_run.run();
            let font = run.font();
            let font_bytes = font.data.as_ref();
            let font_index = font.index;
            let blob_id = font.data.id();
            let font_px_per_em = run.font_size();

            let range_start = text_list.instances.len();
            for glyph in glyph_run.positioned_glyphs() {
                let Some(record) =
                    glyph_provider.resolve(font_bytes, font_index, blob_id, glyph.id)
                else {
                    continue;
                };
                if record.empty {
                    continue;
                }
                let pen_doc = content_transform.transform_point2(Vec2::new(glyph.x, glyph.y));
                text_list.instances.push(TextGlyphInstance {
                    pen_doc: [pen_doc.x, pen_doc.y],
                    glyph_id: record.descriptor_index,
                    color: packed_color,
                });
            }
            let range_end = text_list.instances.len();
            if range_end > range_start {
                items.push(PaintItem {
                    z_order: stack_index as f32 + stack_z_offsets::TEXT,
                    payload: PaintPayload::TextRun(TextRunRecipe {
                        range: range_start..range_end,
                        clip,
                        blob_id,
                        font_index,
                        font_px_per_em,
                    }),
                });
            }

            if let Some((decorations, strike_c, underline_c)) =
                section_entity.and_then(|entity| text_decorations.get(entity).ok())
            {
                let (has_strikethrough, has_underline) = match decorations {
                    (Some(_), Some(_)) => (true, true),
                    (Some(_), None) => (true, false),
                    (None, Some(_)) => (false, true),
                    (None, None) => (false, false),
                };
                let metrics = run.metrics();
                push_decoration_items(
                    items,
                    content_transform,
                    clip,
                    stack_index,
                    glyph_run.offset() + 0.5 * glyph_run.advance(),
                    glyph_run.advance(),
                    glyph_run.baseline() - metrics.strikethrough_offset,
                    metrics.strikethrough_size,
                    glyph_run.baseline() - metrics.underline_offset,
                    metrics.underline_size,
                    has_strikethrough,
                    strike_c.map(|c| c.0),
                    has_underline,
                    underline_c.map(|c| c.0),
                    color,
                );
            }
        }
    }
}

/// Emits glyphs from an editable text layout.
///
/// `content_transform` must include the text scroll offset.
#[allow(clippy::too_many_arguments)]
fn push_editable_text_glyph_items(
    items: &mut Vec<PaintItem>,
    text_list: &mut TextPaintList,
    glyph_provider: &mut GlyphOutlineProvider,
    layout: &parley::Layout<TextBrush>,
    content_transform: Affine2,
    clip: Option<Rect>,
    stack_index: u32,
    color: Color,
) {
    let packed_color = pack_text_color(color);

    for line in layout.lines() {
        for item in line.items() {
            let PositionedLayoutItem::GlyphRun(glyph_run) = item else {
                continue;
            };
            let run = glyph_run.run();
            let font = run.font();
            let font_bytes = font.data.as_ref();
            let font_index = font.index;
            let blob_id = font.data.id();
            let font_px_per_em = run.font_size();

            let range_start = text_list.instances.len();
            for glyph in glyph_run.positioned_glyphs() {
                let Some(record) =
                    glyph_provider.resolve(font_bytes, font_index, blob_id, glyph.id)
                else {
                    continue;
                };
                if record.empty {
                    continue;
                }
                let pen_doc = content_transform.transform_point2(Vec2::new(glyph.x, glyph.y));
                text_list.instances.push(TextGlyphInstance {
                    pen_doc: [pen_doc.x, pen_doc.y],
                    glyph_id: record.descriptor_index,
                    color: packed_color,
                });
            }
            let range_end = text_list.instances.len();
            if range_end > range_start {
                items.push(PaintItem {
                    z_order: stack_index as f32 + stack_z_offsets::TEXT,
                    payload: PaintPayload::TextRun(TextRunRecipe {
                        range: range_start..range_end,
                        clip,
                        blob_id,
                        font_index,
                        font_px_per_em,
                    }),
                });
            }
        }
    }
}

/// Emits editable-text selection, cursor, and IME decoration quads.
///
/// Layout supplies current rectangles and selection colors.
///
#[allow(clippy::too_many_arguments)]
fn push_editable_text_decoration_items(
    items: &mut Vec<PaintItem>,
    content_transform: Affine2,
    clip: Option<Rect>,
    stack_index: u32,
    layout_info: &TextLayoutInfo,
    cursor_style: &TextCursorStyle,
    text_color: Color,
    focused: bool,
) {
    let selection_color = if focused {
        cursor_style.selection_color
    } else {
        cursor_style.unfocused_selection_color
    };
    if !layout_info.selection_rects.is_empty() && !selection_color.is_fully_transparent() {
        let color = color_to_linear_arr(selection_color);
        for rect in layout_info.selection_rects.iter() {
            items.push(PaintItem {
                z_order: stack_index as f32 + stack_z_offsets::TEXT_SELECTION,
                payload: PaintPayload::Quad(QuadRecipe {
                    transform: content_transform * Affine2::from_translation(rect.center()),
                    size: rect.size(),
                    clip,
                    color,
                    radius: [0.0; 4],
                    border: [0.0; 4],
                    flags: 0,
                    tex_slot: 0,
                    uv: [Vec2::ZERO; 4],
                }),
            });
        }
    }

    if let Some((true, cursor_rect)) = layout_info.cursor
        && !cursor_rect.is_empty()
        && !cursor_style.color.is_fully_transparent()
    {
        items.push(PaintItem {
            z_order: stack_index as f32 + stack_z_offsets::TEXT_CURSOR,
            payload: PaintPayload::Quad(QuadRecipe {
                transform: content_transform * Affine2::from_translation(cursor_rect.center()),
                size: cursor_rect.size(),
                clip,
                color: color_to_linear_arr(cursor_style.color),
                radius: [0.0; 4],
                border: [0.0; 4],
                flags: 0,
                tex_slot: 0,
                uv: [Vec2::ZERO; 4],
            }),
        });
    }

    if !layout_info.preedit_underline_rects.is_empty() {
        let color = color_to_linear_arr(text_color);
        for rect in layout_info.preedit_underline_rects.iter() {
            items.push(PaintItem {
                // IME underlines share the text-decoration z offset.
                z_order: stack_index as f32 + stack_z_offsets::TEXT_STRIKETHROUGH,
                payload: PaintPayload::Quad(QuadRecipe {
                    transform: content_transform * Affine2::from_translation(rect.center()),
                    size: rect.size(),
                    clip,
                    color,
                    radius: [0.0; 4],
                    border: [0.0; 4],
                    flags: 0,
                    tex_slot: 0,
                    uv: [Vec2::ZERO; 4],
                }),
            });
        }
    }
}

static NONLINEAR_GRADIENT_WARNED: AtomicBool = AtomicBool::new(false);
static GRADIENT_COLOR_SPACE_WARNED: AtomicBool = AtomicBool::new(false);

fn warn_once(flag: &AtomicBool, msg: &str) {
    if !flag.swap(true, Ordering::Relaxed) {
        bevy::log::warn!("{msg}");
    }
}

fn color_to_space_arr(color: Color, hsla: bool) -> [f32; 4] {
    if hsla {
        let h: Hsla = color.into();
        [h.hue / 360.0, h.saturation, h.lightness, h.alpha]
    } else {
        let s: Srgba = color.into();
        [s.red, s.green, s.blue, s.alpha]
    }
}

/// Extracts a linear gradient into solid or segmented quads.
fn push_linear_gradient(
    items: &mut Vec<PaintItem>,
    scratch: &mut Vec<(Color, f32, f32)>,
    node: &ComputedNode,
    transform: Affine2,
    clip: Option<Rect>,
    stack_index: u32,
    gradient: &LinearGradient,
) {
    if gradient.stops.is_empty() {
        return;
    }
    let z_order = stack_index as f32 + stack_z_offsets::GRADIENT;

    if gradient.stops.len() == 1 {
        items.push(PaintItem {
            z_order,
            payload: PaintPayload::Quad(QuadRecipe {
                transform,
                size: node.size(),
                clip,
                color: color_to_linear_arr(gradient.stops[0].color),
                radius: node.border_radius().into(),
                border: border_arr(node.border()),
                flags: 0,
                tex_slot: 0,
                uv: [Vec2::ZERO; 4],
            }),
        });
        return;
    }

    let length = compute_gradient_line_length(gradient.angle, node.size());
    let scale_factor = 1.0 / node.inverse_scale_factor().max(f32::EPSILON);
    resolve_gradient_stops(&gradient.stops, scale_factor, length, node.size(), scratch);

    let hsla = matches!(
        gradient.color_space,
        InterpolationColorSpace::Hsla | InterpolationColorSpace::HslaLong
    );
    if !hsla && !matches!(gradient.color_space, InterpolationColorSpace::Srgba) {
        warn_once(
            &GRADIENT_COLOR_SPACE_WARNED,
            "ui-bridge: gradient color_space is neither Srgba nor Hsla; \
             the ABI only encodes those two, approximating as Srgba",
        );
    }

    let mut segments: SmallVec<[GradientSegment; 4]> = SmallVec::new();
    for pair in scratch.windows(2) {
        let (color_a, pos_a, _) = pair[0];
        let (color_b, pos_b, _) = pair[1];
        if pos_a == pos_b {
            // Ignore zero-length segments.
            continue;
        }
        segments.push(GradientSegment {
            start_color: color_to_space_arr(color_a, hsla),
            end_color: color_to_space_arr(color_b, hsla),
            start_len: pos_a,
            end_len: pos_b,
            hsla,
        });
    }
    if segments.is_empty() {
        return;
    }

    let corner_points = QUAD_VERTEX_POSITIONS.map(|p| p * node.size());
    // Select the gradient-start corner by angle quadrant.
    let corner_index = (((gradient.angle - std::f32::consts::FRAC_PI_2)
        .rem_euclid(std::f32::consts::TAU))
        / std::f32::consts::FRAC_PI_2) as usize;
    let g_start = corner_points[corner_index.min(3)];
    // CSS angles increase clockwise.
    let g_dir = Vec2::new(gradient.angle.sin(), -gradient.angle.cos());

    items.push(PaintItem {
        z_order,
        payload: PaintPayload::Gradient(GradientRecipe {
            transform,
            size: node.size(),
            clip,
            radius: node.border_radius().into(),
            border: border_arr(node.border()),
            base_flags: 0,
            g_start,
            g_dir,
            segments,
        }),
    });
}

/// Clips transformed corners and displaces SDF points consistently.
///
/// Rotated nodes skip displacement; their transformed AABB handles culling.
fn corners_and_clip(
    transform: Affine2,
    size: Vec2,
    clip: Option<Rect>,
) -> Option<([Vec2; 4], [Vec2; 4])> {
    let positions = QUAD_VERTEX_POSITIONS.map(|p| transform.transform_point2(p * size));
    let points = QUAD_VERTEX_POSITIONS.map(|p| p * size);

    let Some(clip) = clip else {
        return Some((positions, points));
    };

    if transform.matrix2.x_axis.y != 0.0 {
        let min = positions.into_iter().reduce(Vec2::min).unwrap();
        let max = positions.into_iter().reduce(Vec2::max).unwrap();
        if max.x <= clip.min.x || min.x >= clip.max.x || max.y <= clip.min.y || min.y >= clip.max.y
        {
            return None;
        }
        return Some((positions, points));
    }

    let diff = [
        Vec2::new(
            f32::max(clip.min.x - positions[0].x, 0.),
            f32::max(clip.min.y - positions[0].y, 0.),
        ),
        Vec2::new(
            f32::min(clip.max.x - positions[1].x, 0.),
            f32::max(clip.min.y - positions[1].y, 0.),
        ),
        Vec2::new(
            f32::min(clip.max.x - positions[2].x, 0.),
            f32::min(clip.max.y - positions[2].y, 0.),
        ),
        Vec2::new(
            f32::max(clip.min.x - positions[3].x, 0.),
            f32::min(clip.max.y - positions[3].y, 0.),
        ),
    ];

    let transformed_size = transform.transform_vector2(size).abs();
    if diff[0].x - diff[1].x >= transformed_size.x || diff[1].y - diff[2].y >= transformed_size.y {
        return None;
    }

    Some((
        [
            positions[0] + diff[0],
            positions[1] + diff[1],
            positions[2] + diff[2],
            positions[3] + diff[3],
        ],
        [
            points[0] + diff[0],
            points[1] + diff[1],
            points[2] + diff[2],
            points[3] + diff[3],
        ],
    ))
}

fn emit_quad(vertices: &mut Vec<UiVertex>, recipe: &QuadRecipe) {
    let Some((positions, points)) = corners_and_clip(recipe.transform, recipe.size, recipe.clip)
    else {
        return;
    };
    let size = [recipe.size.x, recipe.size.y];
    for i in 0..4 {
        vertices.push(UiVertex {
            pos: [positions[i].x, positions[i].y],
            uv: [recipe.uv[i].x, recipe.uv[i].y],
            color: recipe.color,
            color2: [0.0; 4],
            radius: recipe.radius,
            border: recipe.border,
            size,
            point: [points[i].x, points[i].y],
            flags: recipe.flags,
            tex_slot: recipe.tex_slot,
        });
    }
}

fn emit_gradient(vertices: &mut Vec<UiVertex>, recipe: &GradientRecipe) {
    let Some((positions, points)) = corners_and_clip(recipe.transform, recipe.size, recipe.clip)
    else {
        return;
    };
    let size = [recipe.size.x, recipe.size.y];
    let n = recipe.segments.len();
    for (idx, seg) in recipe.segments.iter().enumerate() {
        let mut flags = recipe.base_flags | UI_FLAG_GRADIENT;
        if seg.hsla {
            flags |= UI_FLAG_GRADIENT_SPACE_HSLA;
        }
        if idx == 0 && seg.start_len > 0.0 {
            flags |= UI_FLAG_FILL_START;
        }
        if idx == n - 1 {
            flags |= UI_FLAG_FILL_END;
        }

        for i in 0..4 {
            let distance = (points[i] - recipe.g_start).dot(recipe.g_dir);
            let t = (distance - seg.start_len) / (seg.end_len - seg.start_len);
            vertices.push(UiVertex {
                pos: [positions[i].x, positions[i].y],
                uv: [t, 0.0],
                color: seg.start_color,
                color2: seg.end_color,
                radius: recipe.radius,
                border: recipe.border,
                size,
                point: [points[i].x, points[i].y],
                flags,
                tex_slot: 0,
            });
        }
    }
}

/// Emits a material quad with packed mode parameters.
fn emit_material(vertices: &mut Vec<UiVertex>, recipe: &MaterialRecipe) {
    let Some((positions, points)) = corners_and_clip(recipe.transform, recipe.size, recipe.clip)
    else {
        return;
    };
    let size = [recipe.size.x, recipe.size.y];
    let flags = recipe.mode << UI_MODE_SHIFT;
    let color2 = recipe.data.to_color2().to_array();
    for i in 0..4 {
        vertices.push(UiVertex {
            pos: [positions[i].x, positions[i].y],
            uv: [QUAD_VERTEX_UVS[i].x, QUAD_VERTEX_UVS[i].y],
            color: [0.0; 4],
            color2,
            radius: recipe.radius,
            border: [0.0; 4],
            size,
            point: [points[i].x, points[i].y],
            flags,
            tex_slot: 0,
        });
    }
}

/// Emits a clipped, padded shadow quad.
///
/// UVs invert the `abi_ui::ui_shadow_point` calculation.
fn emit_shadow(vertices: &mut Vec<UiShadowVertex>, recipe: &ShadowRecipe) {
    let Some((positions, points)) = corners_and_clip(recipe.transform, recipe.bounds, recipe.clip)
    else {
        return;
    };
    let bounds_arr = [recipe.bounds.x, recipe.bounds.y];
    let size_arr = [recipe.size.x, recipe.size.y];
    for i in 0..4 {
        let uv = points[i] / recipe.bounds + Vec2::splat(0.5);
        vertices.push(UiShadowVertex {
            pos: [positions[i].x, positions[i].y],
            uv: [uv.x, uv.y],
            color: recipe.color,
            size: size_arr,
            radius: recipe.radius,
            blur: recipe.blur,
            bounds: bounds_arr,
        });
    }
}

/// Rebuilds UI paint streams after layout, clipping, and stacking.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn paint_ui_system(
    mut list: ResMut<UiPaintList>,
    mut text_list: ResMut<TextPaintList>,
    mut glyph_provider: ResMut<GlyphOutlineProvider>,
    icon_registry: Res<IconRegistry>,
    // Group locals to stay within Bevy's system-parameter limit.
    (mut items, mut gradient_scratch): (Local<Vec<PaintItem>>, Local<Vec<(Color, f32, f32)>>),
    backgrounds: Query<(
        &ComputedNode,
        &ComputedStackIndex,
        &UiGlobalTransform,
        &InheritedVisibility,
        Option<&CalculatedClip>,
        &BackgroundColor,
    )>,
    // Shadows precede each node's background.
    shadows: Query<(
        &ComputedNode,
        &ComputedStackIndex,
        &UiGlobalTransform,
        &InheritedVisibility,
        Option<&CalculatedClip>,
        &BoxShadow,
        &ComputedUiRenderTargetInfo,
    )>,
    borders: Query<(
        &ComputedNode,
        &ComputedStackIndex,
        &UiGlobalTransform,
        &InheritedVisibility,
        Option<&CalculatedClip>,
        AnyOf<(&BorderColor, &Outline)>,
    )>,
    gradients: Query<(
        &ComputedNode,
        &ComputedStackIndex,
        &UiGlobalTransform,
        &InheritedVisibility,
        Option<&CalculatedClip>,
        &BackgroundGradient,
    )>,
    images: Query<(
        &ComputedNode,
        &ComputedStackIndex,
        &UiGlobalTransform,
        &InheritedVisibility,
        Option<&CalculatedClip>,
        &ImageNode,
    )>,
    // Material tags select checkerboard or color-plane shading.
    materials: Query<(
        &ComputedNode,
        &ComputedStackIndex,
        &UiGlobalTransform,
        &InheritedVisibility,
        Option<&CalculatedClip>,
        &UiMaterialTag,
    )>,
    texts: Query<(
        &ComputedNode,
        &ComputedStackIndex,
        &UiGlobalTransform,
        &InheritedVisibility,
        Option<&CalculatedClip>,
        &ComputedTextBlock,
        &TextColor,
    )>,
    text_colors: Query<&TextColor>,
    text_decorations: Query<(
        AnyOf<(&Strikethrough, &Underline)>,
        Option<&StrikethroughColor>,
        Option<&UnderlineColor>,
    )>,
    // Editable text uses its live layout and decoration rectangles.
    editable_texts: Query<(
        Entity,
        &ComputedNode,
        &ComputedStackIndex,
        &UiGlobalTransform,
        &InheritedVisibility,
        Option<&CalculatedClip>,
        &EditableText,
        &TextLayoutInfo,
        &TextCursorStyle,
        &TextColor,
        Option<&TextScroll>,
    )>,
    input_focus: Option<Res<InputFocus>>,
) {
    items.clear();

    for (node, stack, transform, visibility, clip, background) in &backgrounds {
        if !visibility.get() || node.is_empty() || background.0.is_fully_transparent() {
            continue;
        }
        items.push(PaintItem {
            z_order: stack.0 as f32 + stack_z_offsets::BACKGROUND,
            payload: PaintPayload::Quad(QuadRecipe {
                transform: transform.affine(),
                size: node.size(),
                clip: clip.map(|c| c.clip),
                color: color_to_linear_arr(background.0),
                radius: node.border_radius().into(),
                border: border_arr(node.border()),
                flags: 0,
                tex_slot: 0,
                uv: [Vec2::ZERO; 4],
            }),
        });
    }

    for (node, stack, transform, visibility, clip, box_shadow, target) in &shadows {
        if !visibility.get() || node.is_empty() {
            continue;
        }
        push_shadow_items(
            &mut items,
            node,
            transform.affine(),
            clip.map(|c| c.clip),
            stack.0,
            box_shadow,
            target,
        );
    }

    for (node, stack, transform, visibility, clip, (border_color, outline)) in &borders {
        if !visibility.get() {
            continue;
        }
        let affine = transform.affine();
        let clip_rect = clip.map(|c| c.clip);
        if let Some(border_color) = border_color {
            push_border_items(&mut items, node, affine, clip_rect, stack.0, border_color);
        }
        if let Some(outline) = outline {
            push_outline_item(&mut items, node, affine, clip_rect, stack.0, outline);
        }
    }

    for (node, stack, transform, visibility, clip, gradient_layers) in &gradients {
        if !visibility.get() {
            continue;
        }
        let affine = transform.affine();
        let clip_rect = clip.map(|c| c.clip);
        for gradient in gradient_layers.0.iter() {
            match gradient {
                Gradient::Linear(linear) => push_linear_gradient(
                    &mut items,
                    &mut gradient_scratch,
                    node,
                    affine,
                    clip_rect,
                    stack.0,
                    linear,
                ),
                Gradient::Radial(_) | Gradient::Conic(_) => warn_once(
                    &NONLINEAR_GRADIENT_WARNED,
                    "ui-bridge: skipping a Radial/Conic BackgroundGradient layer \
                     (only linear gradients are supported)",
                ),
            }
        }
    }

    for (node, stack, transform, visibility, clip, image) in &images {
        if !visibility.get() || image.color.is_fully_transparent() {
            continue;
        }
        // Registered icons use full-image UVs; others remain tinted quads.
        let (flags, tex_slot, uv) = match icon_registry.logical_slot(image.image.id()) {
            Some(slot) => (UI_FLAG_TEXTURED, slot, QUAD_VERTEX_UVS),
            None => (0, 0, [Vec2::ZERO; 4]),
        };
        items.push(PaintItem {
            z_order: stack.0 as f32 + stack_z_offsets::IMAGE,
            payload: PaintPayload::Quad(QuadRecipe {
                transform: transform.affine(),
                size: node.size(),
                clip: clip.map(|c| c.clip),
                color: color_to_linear_arr(image.color),
                radius: node.border_radius().into(),
                border: border_arr(node.border()),
                flags,
                tex_slot,
                uv,
            }),
        });
    }

    // Materials occupy their defined painter-order slot.
    for (node, stack, transform, visibility, clip, tag) in &materials {
        if !visibility.get() || node.is_empty() {
            continue;
        }
        items.push(PaintItem {
            z_order: stack.0 as f32 + stack_z_offsets::MATERIAL,
            payload: PaintPayload::Material(MaterialRecipe {
                transform: transform.affine(),
                size: node.size(),
                clip: clip.map(|c| c.clip),
                radius: node.border_radius().into(),
                mode: tag.mode,
                data: tag.data,
            }),
        });
    }

    // Store glyph instances; sort their paint items below.
    text_list.instances.clear();
    for (node, stack, transform, visibility, clip, computed_block, text_color) in &texts {
        if !visibility.get() || node.is_empty() {
            continue;
        }
        push_text_items(
            &mut items,
            &mut text_list,
            &mut glyph_provider,
            computed_block,
            node,
            transform.affine(),
            clip.map(|c| c.clip),
            stack.0,
            text_color.0,
            &text_colors,
            &text_decorations,
        );
    }

    // Editable text supplies glyphs and current decorations.
    for (
        entity,
        node,
        stack,
        transform,
        visibility,
        clip,
        editable_text,
        layout_info,
        cursor_style,
        text_color,
        text_scroll,
    ) in &editable_texts
    {
        if !visibility.get() || node.is_empty() {
            continue;
        }
        let scroll = text_scroll.map_or(Vec2::ZERO, |s| s.0);
        let content_transform =
            transform.affine() * Affine2::from_translation(node.content_box().min - scroll);
        let clip_rect = clip.map(|c| c.clip);

        if let Some(layout) = editable_text.editor().try_layout() {
            push_editable_text_glyph_items(
                &mut items,
                &mut text_list,
                &mut glyph_provider,
                layout,
                content_transform,
                clip_rect,
                stack.0,
                text_color.0,
            );
        }

        let focused = input_focus.as_ref().and_then(|f| f.get()) == Some(entity);
        push_editable_text_decoration_items(
            &mut items,
            content_transform,
            clip_rect,
            stack.0,
            layout_info,
            cursor_style,
            text_color.0,
            focused,
        );
    }

    // Finite offsets make this a total, stable painter-order sort.
    items.sort_by(|a, b| a.z_order.total_cmp(&b.z_order));

    // Each payload kind interrupts other open batches.
    // Adjacent compatible text runs merge by batch key.
    list.vertices.clear();
    list.batches.clear();
    list.shadow_vertices.clear();
    list.shadow_batches.clear();
    text_list.batches.clear();
    let mut next_order: u32 = 0;
    let mut quad_start = 0usize;
    let mut quad_open = false;
    let mut shadow_start = 0usize;
    let mut shadow_open = false;

    // Close open batches before switching payload kinds.
    macro_rules! close_quad {
        () => {
            if quad_open {
                let quad_end = list.vertices.len() / 4;
                if quad_end > quad_start {
                    list.batches.push(UiBatch {
                        quad_range: quad_start..quad_end,
                        scissor: None,
                        order: next_order,
                    });
                    next_order += 1;
                }
                quad_open = false;
            }
        };
    }
    macro_rules! close_shadow {
        () => {
            if shadow_open {
                let shadow_end = list.shadow_vertices.len() / 4;
                if shadow_end > shadow_start {
                    list.shadow_batches.push(UiShadowBatch {
                        quad_range: shadow_start..shadow_end,
                        scissor: None,
                        order: next_order,
                    });
                    next_order += 1;
                }
                shadow_open = false;
            }
        };
    }

    for item in items.iter() {
        match &item.payload {
            PaintPayload::Quad(recipe) => {
                close_shadow!();
                if !quad_open {
                    quad_start = list.vertices.len() / 4;
                    quad_open = true;
                }
                emit_quad(&mut list.vertices, recipe);
            }
            PaintPayload::Gradient(recipe) => {
                close_shadow!();
                if !quad_open {
                    quad_start = list.vertices.len() / 4;
                    quad_open = true;
                }
                emit_gradient(&mut list.vertices, recipe);
            }
            PaintPayload::Material(recipe) => {
                close_shadow!();
                if !quad_open {
                    quad_start = list.vertices.len() / 4;
                    quad_open = true;
                }
                emit_material(&mut list.vertices, recipe);
            }
            PaintPayload::Shadow(recipe) => {
                close_quad!();
                if !shadow_open {
                    shadow_start = list.shadow_vertices.len() / 4;
                    shadow_open = true;
                }
                emit_shadow(&mut list.shadow_vertices, recipe);
            }
            PaintPayload::TextRun(recipe) => {
                close_quad!();
                close_shadow!();

                let merged = text_list.batches.last_mut().is_some_and(|last| {
                    last.blob_id == recipe.blob_id
                        && last.font_index == recipe.font_index
                        && last.font_px_per_em == recipe.font_px_per_em
                        && last.clip == recipe.clip
                        && last.instance_range.end == recipe.range.start
                        // The shared order proves painter adjacency.
                        && last.order + 1 == next_order
                });
                if merged {
                    text_list
                        .batches
                        .last_mut()
                        .expect("just checked Some")
                        .instance_range
                        .end = recipe.range.end;
                } else {
                    text_list.batches.push(TextRunBatch {
                        instance_range: recipe.range.clone(),
                        clip: recipe.clip,
                        blob_id: recipe.blob_id,
                        font_index: recipe.font_index,
                        font_px_per_em: recipe.font_px_per_em,
                        order: next_order,
                    });
                    next_order += 1;
                }
            }
        }
    }
    list.quad_count = list.vertices.len() / 4;
    list.shadow_quad_count = list.shadow_vertices.len() / 4;
    // Close final batches without advancing the order.
    if quad_open {
        let quad_end = list.quad_count;
        if quad_end > quad_start {
            list.batches.push(UiBatch {
                quad_range: quad_start..quad_end,
                scissor: None,
                order: next_order,
            });
        }
    }
    if shadow_open {
        let shadow_end = list.shadow_quad_count;
        if shadow_end > shadow_start {
            list.shadow_batches.push(UiShadowBatch {
                quad_range: shadow_start..shadow_end,
                scissor: None,
                order: next_order,
            });
        }
    }
}

/// Adds paint resources and runs the walker after UI layout and stacking.
pub struct UiBridgePlugin;

impl Plugin for UiBridgePlugin {
    fn build(&self, app: &mut App) {
        crate::icons::build(app);
        // Refresh computed camera data before UI layout consumes it.
        crate::camera::build(app);
        app.init_resource::<UiPaintList>()
            .init_resource::<TextPaintList>()
            .init_resource::<GlyphOutlineProvider>()
            .add_systems(
                PostUpdate,
                paint_ui_system
                    .after(UiSystems::PostLayout)
                    .after(UiSystems::Stack),
            );
    }
}
