//! CPU paint-walker tests using hand-built layout state.
//!
//! Tests control geometry, clipping, ordering, gradients, icons, materials,
//! text, and shadows without a window or GPU.

use bevy::app::TaskPoolPlugin;
use bevy::asset::Assets;
use bevy::camera::visibility::InheritedVisibility;
use bevy::image::Image;
use bevy::math::{Affine2, Rect, Vec2};
use bevy::prelude::*;
use bevy::sprite::BorderRect;
use bevy::ui::widget::ImageNode;
use bevy::ui::{
    BackgroundGradient, CalculatedClip, ColorStop, ComputedStackIndex, Gradient,
    InterpolationColorSpace, LinearGradient, Outline, UiGlobalTransform, Val,
};
use bevy_color::Color;

use abi_ui::{
    UI_FLAG_BORDER_ANY, UI_FLAG_FILL_END, UI_FLAG_FILL_START, UI_FLAG_GRADIENT,
    UI_FLAG_GRADIENT_SPACE_HSLA, UI_FLAG_TEXTURED, UI_MODE_ALPHA_PATTERN, UI_MODE_COLOR_PLANE,
    UI_MODE_SHIFT, UI_PLANE_HL, UiMaterialData,
};
use ui_bridge::{
    ColorPlaneAxes, IconRegistry, UiBridgePlugin, UiMaterialTag, UiPaintList, UiShadowBatch,
    stack_z_offsets,
};

fn app_with_ui_bridge() -> App {
    let mut app = App::new();
    app.add_plugins(TaskPoolPlugin::default());
    app.add_plugins(UiBridgePlugin);
    app
}

fn base_computed_node(size: Vec2) -> ComputedNode {
    ComputedNode {
        size,
        unrounded_size: size,
        ..Default::default()
    }
}

const EPS: f32 = 1e-4;

fn approx(a: f32, b: f32) -> bool {
    (a - b).abs() < EPS
}

fn approx_vec2(a: [f32; 2], b: Vec2) -> bool {
    approx(a[0], b.x) && approx(a[1], b.y)
}

// Background, border, and outline retain stable painter order.
#[test]
fn background_then_border_then_outline_in_stable_z_order() {
    let mut app = app_with_ui_bridge();

    let size = Vec2::new(100.0, 80.0);
    let mut node = base_computed_node(size);
    node.border = BorderRect::all(4.0);
    node.outline_width = 3.0;
    node.outline_offset = 1.0;

    app.world_mut().spawn((
        node,
        ComputedStackIndex(0),
        UiGlobalTransform::from(Affine2::IDENTITY),
        InheritedVisibility::VISIBLE,
        BackgroundColor(Color::srgb(1.0, 0.0, 0.0)),
        BorderColor::all(Color::srgb(0.0, 1.0, 0.0)),
        Outline::new(Val::Px(3.0), Val::Px(1.0), Color::srgb(0.0, 0.0, 1.0)),
    ));

    app.update();

    let list = app.world().resource::<UiPaintList>();
    assert_eq!(list.quad_count, 3, "background + border + outline");

    // Background: zero flags, original size, red.
    let bg = &list.vertices[0];
    assert_eq!(bg.flags, 0);
    assert!(approx_vec2(bg.size, size));
    assert!(approx(bg.color[0], 1.0) && approx(bg.color[1], 0.0));

    // Border: grouped edges, original size, green.
    let border = &list.vertices[4];
    assert_eq!(border.flags & UI_FLAG_BORDER_ANY, UI_FLAG_BORDER_ANY);
    assert!(approx_vec2(border.size, size));
    assert!(approx(border.color[1], 1.0) && approx(border.color[0], 0.0));

    // Outline: border ring on the enlarged box.
    let outline = &list.vertices[8];
    assert_eq!(outline.flags & UI_FLAG_BORDER_ANY, UI_FLAG_BORDER_ANY);
    let outlined_size = Vec2::new(size.x + 2.0 * (1.0 + 3.0), size.y + 2.0 * (1.0 + 3.0));
    assert!(approx_vec2(outline.size, outlined_size));
    assert!(approx(outline.color[2], 1.0) && approx(outline.color[0], 0.0));
    assert!(approx(outline.border[0], 3.0));
}

// Stack index controls order, independent of spawn order.
#[test]
fn emission_order_follows_stack_index_not_spawn_order() {
    let mut app = app_with_ui_bridge();

    // Higher stack index renders last.
    let node_a = base_computed_node(Vec2::new(10.0, 10.0));
    app.world_mut().spawn((
        node_a,
        ComputedStackIndex(5),
        UiGlobalTransform::from(Affine2::IDENTITY),
        InheritedVisibility::VISIBLE,
        BackgroundColor(Color::srgb(1.0, 0.0, 0.0)),
    ));

    // Lower stack index renders first.
    let node_b = base_computed_node(Vec2::new(10.0, 10.0));
    app.world_mut().spawn((
        node_b,
        ComputedStackIndex(0),
        UiGlobalTransform::from(Affine2::IDENTITY),
        InheritedVisibility::VISIBLE,
        BackgroundColor(Color::srgb(0.0, 0.0, 1.0)),
    ));

    app.update();

    let list = app.world().resource::<UiPaintList>();
    assert_eq!(list.quad_count, 2);
    // Lower stack index must come first.
    assert!(
        approx(list.vertices[0].color[2], 1.0),
        "B (blue) emitted first"
    );
    assert!(
        approx(list.vertices[4].color[0], 1.0),
        "A (red) emitted second"
    );
}

// Clipping displaces both positions and SDF points.
#[test]
fn clipped_node_displaces_corners_and_point() {
    let mut app = app_with_ui_bridge();

    let size = Vec2::new(100.0, 100.0);
    let node = base_computed_node(size);
    // The node spans [0,100] in both axes.
    let transform = Affine2::from_translation(Vec2::new(50.0, 50.0));
    // The clip exposes only x in [0,60].
    let clip = Rect::new(0.0, 0.0, 60.0, 100.0);

    app.world_mut().spawn((
        node,
        ComputedStackIndex(0),
        UiGlobalTransform::from(transform),
        InheritedVisibility::VISIBLE,
        CalculatedClip { clip },
        BackgroundColor(Color::srgb(1.0, 1.0, 1.0)),
    ));

    app.update();

    let list = app.world().resource::<UiPaintList>();
    assert_eq!(list.quad_count, 1);
    let verts = &list.vertices[0..4];

    // Right corners clamp to the clip maximum.
    assert!(approx(verts[0].pos[0], 0.0), "TL x unclipped");
    assert!(approx(verts[1].pos[0], 60.0), "TR x clamped to clip.max.x");
    assert!(approx(verts[2].pos[0], 60.0), "BR x clamped to clip.max.x");
    assert!(approx(verts[3].pos[0], 0.0), "BL x unclipped");

    // SDF points follow the position displacement.
    assert!(
        approx(verts[1].point[0], 10.0),
        "TR point.x displaced with pos"
    );
    assert!(approx(verts[0].point[0], -50.0), "TL point.x untouched");
}

// Rotated nodes skip clip displacement.
#[test]
fn rotated_node_transforms_correctly_and_skips_clip_displacement() {
    let mut app = app_with_ui_bridge();

    let size = Vec2::new(20.0, 10.0);
    let node = base_computed_node(size);
    let angle = std::f32::consts::FRAC_PI_2;
    let translation = Vec2::new(100.0, 100.0);
    let transform = Affine2::from_angle_translation(angle, translation);

    // This clip would displace unrotated corners.
    let clip = Rect::new(0.0, 0.0, 1000.0, 1000.0);

    app.world_mut().spawn((
        node,
        ComputedStackIndex(0),
        UiGlobalTransform::from(transform),
        InheritedVisibility::VISIBLE,
        CalculatedClip { clip },
        BackgroundColor(Color::WHITE),
    ));

    app.update();

    let list = app.world().resource::<UiPaintList>();
    assert_eq!(list.quad_count, 1);
    let verts = &list.vertices[0..4];

    // Rotate the top-left offset and translate to (100,100).
    let hand_computed = transform.transform_point2(Vec2::new(-10.0, -5.0));
    assert!(
        approx_vec2(verts[0].pos, hand_computed),
        "TL matches hand-computed Affine2"
    );

    // Rotation preserves the untransformed SDF point.
    assert!(approx_vec2(verts[0].point, Vec2::new(-10.0, -5.0)));
    assert!(approx_vec2(verts[2].point, Vec2::new(10.0, 5.0)));
}

// Painter offsets match the reference UI table.
#[test]
fn z_offsets_match_reference_table() {
    assert_eq!(stack_z_offsets::BOX_SHADOW, -0.1);
    assert_eq!(stack_z_offsets::BACKGROUND, 0.0);
    assert_eq!(stack_z_offsets::BORDER, 0.01);
    assert_eq!(stack_z_offsets::GRADIENT, 0.02);
    assert_eq!(stack_z_offsets::BORDER_GRADIENT, 0.03);
    assert_eq!(stack_z_offsets::IMAGE, 0.04);
    assert_eq!(stack_z_offsets::MATERIAL, 0.05);
    assert_eq!(stack_z_offsets::TEXT_SELECTION, 0.055);
    assert_eq!(stack_z_offsets::TEXT, 0.06);
    assert_eq!(stack_z_offsets::TEXT_STRIKETHROUGH, 0.07);
    assert_eq!(stack_z_offsets::TEXT_CURSOR, 0.08);
}

// Three-stop gradients emit segments and normalized parameters.
#[test]
fn three_stop_linear_gradient_emits_two_segments_with_correct_t() {
    let mut app = app_with_ui_bridge();

    let size = Vec2::new(100.0, 100.0);
    let node = base_computed_node(size);

    let stops = vec![
        ColorStop {
            color: Color::srgb(1.0, 0.0, 0.0),
            point: Val::Percent(0.0),
            hint: 0.5,
        },
        ColorStop {
            color: Color::srgb(0.0, 1.0, 0.0),
            point: Val::Percent(50.0),
            hint: 0.5,
        },
        ColorStop {
            color: Color::srgb(0.0, 0.0, 1.0),
            point: Val::Percent(100.0),
            hint: 0.5,
        },
    ];

    let gradient = LinearGradient {
        color_space: InterpolationColorSpace::Srgba,
        angle: LinearGradient::TO_BOTTOM,
        stops,
    };

    app.world_mut().spawn((
        node,
        ComputedStackIndex(0),
        UiGlobalTransform::from(Affine2::IDENTITY),
        InheritedVisibility::VISIBLE,
        BackgroundGradient(vec![Gradient::Linear(gradient)]),
    ));

    app.update();

    let list = app.world().resource::<UiPaintList>();
    assert_eq!(list.quad_count, 2, "one quad per adjacent stop pair");

    for v in &list.vertices[0..8] {
        assert_ne!(v.flags & UI_FLAG_GRADIENT, 0);
        assert_eq!(
            v.flags & UI_FLAG_GRADIENT_SPACE_HSLA,
            0,
            "Srgba space, no HSLA flag"
        );
    }

    // First segment starts at zero and extends beyond its stop.
    let seg0 = &list.vertices[0..4];
    assert!(approx(seg0[0].uv[0], 0.0), "seg0 TL t");
    assert!(approx(seg0[1].uv[0], 0.0), "seg0 TR t");
    assert!(approx(seg0[2].uv[0], 2.0), "seg0 BR t");
    assert!(approx(seg0[3].uv[0], 2.0), "seg0 BL t");
    assert_eq!(seg0[0].flags & UI_FLAG_FILL_START, 0);

    // Final segment sets the end-fill flag.
    let seg1 = &list.vertices[4..8];
    assert!(approx(seg1[0].uv[0], -1.0), "seg1 TL t");
    assert!(approx(seg1[2].uv[0], 1.0), "seg1 BR t");
    assert_ne!(seg1[0].flags & UI_FLAG_FILL_END, 0);
}

#[test]
fn hsla_gradient_space_sets_the_hsla_flag() {
    let mut app = app_with_ui_bridge();

    let size = Vec2::new(50.0, 50.0);
    let node = base_computed_node(size);

    let stops = vec![
        ColorStop {
            color: Color::srgb(1.0, 0.0, 0.0),
            point: Val::Percent(0.0),
            hint: 0.5,
        },
        ColorStop {
            color: Color::srgb(0.0, 0.0, 1.0),
            point: Val::Percent(100.0),
            hint: 0.5,
        },
    ];
    let gradient = LinearGradient {
        color_space: InterpolationColorSpace::Hsla,
        angle: LinearGradient::TO_RIGHT,
        stops,
    };

    app.world_mut().spawn((
        node,
        ComputedStackIndex(0),
        UiGlobalTransform::from(Affine2::IDENTITY),
        InheritedVisibility::VISIBLE,
        BackgroundGradient(vec![Gradient::Linear(gradient)]),
    ));

    app.update();

    let list = app.world().resource::<UiPaintList>();
    assert_eq!(list.quad_count, 1);
    for v in &list.vertices {
        assert_ne!(v.flags & UI_FLAG_GRADIENT_SPACE_HSLA, 0);
    }
    // Red converts to normalized HSL coordinates.
    let start = list.vertices[0].color;
    assert!(approx(start[0], 0.0));
    assert!(approx(start[1], 1.0));
    assert!(approx(start[2], 0.5));
}

// Invisible nodes contribute no paint.
#[test]
fn invisible_node_is_skipped() {
    let mut app = app_with_ui_bridge();

    let node = base_computed_node(Vec2::new(10.0, 10.0));
    app.world_mut().spawn((
        node,
        ComputedStackIndex(0),
        UiGlobalTransform::from(Affine2::IDENTITY),
        InheritedVisibility::HIDDEN,
        BackgroundColor(Color::WHITE),
    ));

    app.update();

    let list = app.world().resource::<UiPaintList>();
    assert_eq!(list.quad_count, 0);
    assert!(list.batches.is_empty());
}

// Unregistered images fall back to untextured tint quads.
#[test]
fn image_node_with_unregistered_asset_emits_untextured_tint_quad() {
    let mut app = app_with_ui_bridge();

    let node = base_computed_node(Vec2::new(32.0, 32.0));
    app.world_mut().spawn((
        node,
        ComputedStackIndex(0),
        UiGlobalTransform::from(Affine2::IDENTITY),
        InheritedVisibility::VISIBLE,
        ImageNode {
            color: Color::srgb(0.2, 0.4, 0.6),
            ..Default::default()
        },
    ));

    app.update();

    let list = app.world().resource::<UiPaintList>();
    assert_eq!(list.quad_count, 1);
    let v = &list.vertices[0];
    assert_eq!(v.tex_slot, 0);
    assert_eq!(v.flags, 0, "unregistered asset: untextured ZII fallback");
}

// Registered images emit textured quads with full-image UVs.
#[test]
fn image_node_with_registered_icon_emits_textured_quad() {
    let mut app = app_with_ui_bridge();

    let handle = {
        let mut images = app.world_mut().resource_mut::<Assets<Image>>();
        images.add(Image::default())
    };
    const LOGICAL_SLOT: u32 = 1;
    app.world_mut()
        .insert_resource(IconRegistry::from_slots([(handle.id(), LOGICAL_SLOT)]));

    let node = base_computed_node(Vec2::new(24.0, 24.0));
    app.world_mut().spawn((
        node,
        ComputedStackIndex(0),
        UiGlobalTransform::from(Affine2::IDENTITY),
        InheritedVisibility::VISIBLE,
        ImageNode {
            image: handle,
            color: Color::srgb(1.0, 0.0, 0.0),
            ..Default::default()
        },
    ));

    app.update();

    let list = app.world().resource::<UiPaintList>();
    assert_eq!(list.quad_count, 1);
    let verts = &list.vertices[0..4];
    for v in verts {
        assert_ne!(
            v.flags & UI_FLAG_TEXTURED,
            0,
            "registered icon: UI_FLAG_TEXTURED set"
        );
        assert_eq!(
            v.tex_slot, LOGICAL_SLOT,
            "tex_slot is the registered logical slot"
        );
    }
    assert_eq!(verts[0].uv, [0.0, 0.0], "TL uv");
    assert_eq!(verts[1].uv, [1.0, 0.0], "TR uv");
    assert_eq!(verts[2].uv, [1.0, 1.0], "BR uv");
    assert_eq!(verts[3].uv, [0.0, 1.0], "BL uv");
}

// Material tags preserve painter order and packed shader data.
#[test]
fn material_tagged_node_emits_mode_flagged_quad_at_material_offset() {
    let mut app = app_with_ui_bridge();

    let size = Vec2::new(64.0, 40.0);

    // Background precedes material at equal stack index.
    app.world_mut().spawn((
        base_computed_node(size),
        ComputedStackIndex(0),
        UiGlobalTransform::from(Affine2::IDENTITY),
        InheritedVisibility::VISIBLE,
        BackgroundColor(Color::srgb(1.0, 0.0, 0.0)),
    ));

    app.world_mut().spawn((
        base_computed_node(size),
        ComputedStackIndex(0),
        UiGlobalTransform::from(Affine2::IDENTITY),
        InheritedVisibility::VISIBLE,
        UiMaterialTag::color_plane(ColorPlaneAxes::HueLightness, 0.5),
    ));

    app.update();

    let list = app.world().resource::<UiPaintList>();
    assert_eq!(list.quad_count, 2, "background + material");

    // Background occupies the lower painter order.
    let bg = &list.vertices[0];
    assert_eq!(bg.flags >> UI_MODE_SHIFT, 0, "background stays MODE 0");
    assert!(approx(bg.color[0], 1.0));

    // Material mode and packed data occupy the second quad.
    let verts = &list.vertices[4..8];
    let expected_color2 = UiMaterialData {
        variant: UI_PLANE_HL,
        fixed_channel: 0.5,
        _pad0: [0; 2],
    }
    .to_color2()
    .to_array();
    for v in verts {
        assert_eq!(
            v.flags >> UI_MODE_SHIFT,
            UI_MODE_COLOR_PLANE,
            "material MODE bits"
        );
        assert_eq!(v.color2, expected_color2, "packed UiMaterialData in color2");
        assert!(approx_vec2(v.size, size));
    }
    assert_eq!(verts[0].uv, [0.0, 0.0], "TL uv");
    assert_eq!(verts[1].uv, [1.0, 0.0], "TR uv");
    assert_eq!(verts[2].uv, [1.0, 1.0], "BR uv");
    assert_eq!(verts[3].uv, [0.0, 1.0], "BL uv");
}

// Alpha-pattern mode leaves variant and fixed channel zero.
#[test]
fn alpha_pattern_tagged_node_emits_mode_1_with_zeroed_material_fields() {
    let mut app = app_with_ui_bridge();
    let size = Vec2::new(30.0, 12.0);

    app.world_mut().spawn((
        base_computed_node(size),
        ComputedStackIndex(0),
        UiGlobalTransform::from(Affine2::IDENTITY),
        InheritedVisibility::VISIBLE,
        UiMaterialTag::alpha_pattern(),
    ));

    app.update();

    let list = app.world().resource::<UiPaintList>();
    assert_eq!(list.quad_count, 1);
    let v = &list.vertices[0];
    assert_eq!(v.flags >> UI_MODE_SHIFT, UI_MODE_ALPHA_PATTERN);
    assert_eq!(v.color2, [0.0; 4]);
}

// Shadows use a separate stream and precede backgrounds.
#[test]
fn box_shadow_emits_padded_quad_before_background_at_correct_order() {
    let mut app = app_with_ui_bridge();

    let size = Vec2::new(100.0, 60.0);
    let node = base_computed_node(size);

    app.world_mut().spawn((
        node,
        ComputedStackIndex(0),
        UiGlobalTransform::from(Affine2::IDENTITY),
        InheritedVisibility::VISIBLE,
        BackgroundColor(Color::srgb(0.1, 0.1, 0.1)),
        BoxShadow::new(
            Color::srgba(1.0, 0.0, 0.0, 0.7),
            Val::Px(5.0),
            Val::Px(8.0),
            Val::Px(4.0),
            Val::Px(6.0),
        ),
        ComputedUiRenderTargetInfo::default(),
    ));

    app.update();

    let list = app.world().resource::<UiPaintList>();
    assert_eq!(list.shadow_quad_count, 1, "one shadow quad");
    assert_eq!(list.quad_count, 1, "one background quad");
    assert_eq!(list.shadow_batches.len(), 1);
    assert_eq!(list.batches.len(), 1);

    let shadow_order = list.shadow_batches[0].order;
    let quad_order = list.batches[0].order;
    assert!(
        shadow_order < quad_order,
        "shadow batch (order {shadow_order}) must precede the background batch \
         (order {quad_order}) — BOX_SHADOW sorts before BACKGROUND for the same node"
    );
}

// Shadow offset, spread, blur, bounds, and radii resolve mathematically.
#[test]
fn box_shadow_resolves_offset_spread_blur_against_known_node_size() {
    let mut app = app_with_ui_bridge();

    let size = Vec2::new(100.0, 60.0);
    let mut node = base_computed_node(size);
    node.border_radius = bevy::ui::ResolvedBorderRadius {
        top_left: 10.0,
        top_right: 10.0,
        bottom_right: 10.0,
        bottom_left: 10.0,
    };

    app.world_mut().spawn((
        node,
        ComputedStackIndex(0),
        UiGlobalTransform::from(Affine2::IDENTITY),
        InheritedVisibility::VISIBLE,
        BoxShadow::new(
            Color::srgba(1.0, 0.0, 0.0, 0.7),
            Val::Px(5.0),
            Val::Px(8.0),
            Val::Px(4.0),
            Val::Px(6.0),
        ),
        ComputedUiRenderTargetInfo::default(),
    ));

    app.update();

    let list = app.world().resource::<UiPaintList>();
    assert_eq!(list.shadow_quad_count, 1);
    let verts = &list.shadow_vertices[0..4];

    // spread_x = 4, spread_ratio = (4 + 100) / 100 = 1.04,
    // spread_y = 60 * 1.04 - 60 = 2.4 -> shadow_size = (104, 62.4).
    let expect_size = Vec2::new(104.0, 62.4);
    // radius = 10 * spread_ratio = 10.4 on every corner.
    let expect_radius = 10.4;
    // Blur radius: 6 pixels.
    let expect_blur = 6.0;
    // Bounds add six blur radii per axis.
    let expect_bounds = Vec2::new(140.0, 98.4);
    // Offset determines the expected top-left corner.
    let expect_tl_pos = Vec2::new(-65.0, -41.2);

    for v in verts {
        assert!(
            approx_vec2(v.size, expect_size),
            "shadow size: {:?}",
            v.size
        );
        assert!(
            v.radius.iter().all(|&r| approx(r, expect_radius)),
            "radius: {:?}",
            v.radius
        );
        assert!(approx(v.blur, expect_blur), "blur: {}", v.blur);
        assert!(
            approx_vec2(v.bounds, expect_bounds),
            "bounds: {:?}",
            v.bounds
        );
        // Alpha remains straight; red is full intensity.
        assert!(
            approx(v.color[0], 1.0) && approx(v.color[3], 0.7),
            "color: {:?}",
            v.color
        );
    }
    assert!(
        approx_vec2(verts[0].pos, expect_tl_pos),
        "TL pos: {:?}",
        verts[0].pos
    );
    // Unclipped top-left UV is zero.
    assert!(
        approx_vec2(verts[0].uv, Vec2::ZERO),
        "TL uv: {:?}",
        verts[0].uv
    );
    // Unclipped bottom-right UV is one.
    assert!(
        approx_vec2(verts[2].uv, Vec2::ONE),
        "BR uv: {:?}",
        verts[2].uv
    );
}

// Multiple shadow layers preserve declaration order.
#[test]
fn multiple_shadows_on_one_node_emit_in_declared_back_to_front_order() {
    let mut app = app_with_ui_bridge();

    let size = Vec2::new(50.0, 50.0);
    let node = base_computed_node(size);

    app.world_mut().spawn((
        node,
        ComputedStackIndex(0),
        UiGlobalTransform::from(Affine2::IDENTITY),
        InheritedVisibility::VISIBLE,
        BoxShadow(vec![
            bevy::ui::ShadowStyle {
                color: Color::srgba(1.0, 0.0, 0.0, 1.0),
                x_offset: Val::Px(0.0),
                y_offset: Val::Px(0.0),
                spread_radius: Val::Px(0.0),
                blur_radius: Val::Px(2.0),
            },
            bevy::ui::ShadowStyle {
                color: Color::srgba(0.0, 0.0, 1.0, 1.0),
                x_offset: Val::Px(0.0),
                y_offset: Val::Px(0.0),
                spread_radius: Val::Px(0.0),
                blur_radius: Val::Px(4.0),
            },
        ]),
        ComputedUiRenderTargetInfo::default(),
    ));

    app.update();

    let list = app.world().resource::<UiPaintList>();
    assert_eq!(
        list.shadow_quad_count, 2,
        "two shadow layers, one quad each"
    );
    assert_eq!(
        list.shadow_batches.len(),
        1,
        "same z_order: one merged batch"
    );

    // Stable sorting preserves the declared layer order.
    let first = &list.shadow_vertices[0];
    let second = &list.shadow_vertices[4];
    assert!(
        approx(first.color[0], 1.0) && approx(first.color[2], 0.0),
        "first is red"
    );
    assert!(approx(first.blur, 2.0));
    assert!(
        approx(second.color[2], 1.0) && approx(second.color[0], 0.0),
        "second is blue"
    );
    assert!(approx(second.blur, 4.0));
}

// Fully transparent shadows are skipped.
#[test]
fn fully_transparent_shadow_is_skipped() {
    let mut app = app_with_ui_bridge();

    let size = Vec2::new(40.0, 40.0);
    let node = base_computed_node(size);

    app.world_mut().spawn((
        node,
        ComputedStackIndex(0),
        UiGlobalTransform::from(Affine2::IDENTITY),
        InheritedVisibility::VISIBLE,
        BoxShadow::new(
            Color::srgba(0.0, 0.0, 0.0, 0.0),
            Val::Px(0.0),
            Val::Px(0.0),
            Val::Px(0.0),
            Val::Px(4.0),
        ),
        ComputedUiRenderTargetInfo::default(),
    ));

    app.update();

    let list = app.world().resource::<UiPaintList>();
    assert_eq!(list.shadow_quad_count, 0);
    assert!(list.shadow_batches.is_empty());
}

// Nonpositive shadow sizes are culled.
#[test]
fn spread_that_collapses_the_shadow_box_is_culled() {
    let mut app = app_with_ui_bridge();

    let size = Vec2::new(20.0, 20.0);
    let node = base_computed_node(size);

    app.world_mut().spawn((
        node,
        ComputedStackIndex(0),
        UiGlobalTransform::from(Affine2::IDENTITY),
        InheritedVisibility::VISIBLE,
        BoxShadow::new(
            Color::srgba(1.0, 1.0, 1.0, 1.0),
            Val::Px(0.0),
            Val::Px(0.0),
            Val::Px(-15.0), // Leaves a positive shadow width.
            Val::Px(2.0),
        ),
        ComputedUiRenderTargetInfo::default(),
    ));
    app.world_mut().spawn((
        base_computed_node(size),
        ComputedStackIndex(1),
        UiGlobalTransform::from(Affine2::IDENTITY),
        InheritedVisibility::VISIBLE,
        BoxShadow::new(
            Color::srgba(1.0, 1.0, 1.0, 1.0),
            Val::Px(0.0),
            Val::Px(0.0),
            Val::Px(-25.0), // Collapses the shadow width.
            Val::Px(2.0),
        ),
        ComputedUiRenderTargetInfo::default(),
    ));

    app.update();

    let list = app.world().resource::<UiPaintList>();
    assert_eq!(
        list.shadow_quad_count, 1,
        "only the non-collapsed shadow survives culling"
    );
}

// Shadow batches contain vertices and no scissor.
#[test]
fn shadow_batch_quad_range_matches_emitted_vertex_count() {
    let mut app = app_with_ui_bridge();

    let size = Vec2::new(30.0, 30.0);
    let node = base_computed_node(size);

    app.world_mut().spawn((
        node,
        ComputedStackIndex(2),
        UiGlobalTransform::from(Affine2::IDENTITY),
        InheritedVisibility::VISIBLE,
        BoxShadow::new(
            Color::BLACK,
            Val::Px(0.0),
            Val::Px(0.0),
            Val::Px(0.0),
            Val::Px(3.0),
        ),
        ComputedUiRenderTargetInfo::default(),
    ));

    app.update();

    let list = app.world().resource::<UiPaintList>();
    let batch: &UiShadowBatch = &list.shadow_batches[0];
    assert_eq!(batch.quad_range, 0..1);
    assert_eq!(batch.scissor, None);
    assert_eq!(list.shadow_vertices.len(), 4);
}
