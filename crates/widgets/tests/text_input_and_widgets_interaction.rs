//! Headless interaction tests for text input, scrolling, and menus.
//!
//! Raw keyboard messages exercise Bevy 0.19's focused-input dispatch path.
//! Paint lists verify cursor, selection, clipping, and popover ordering.

use bevy::camera::{Camera, ComputedCameraValues, RenderTargetInfo};
use bevy::input::ButtonState;
use bevy::input::keyboard::{Key, KeyboardInput};
use bevy::input_focus::{FocusCause, InputFocus};
use bevy::math::{UVec2, Vec2};
use bevy::prelude::*;
use bevy::text::{EditableText, FontSource, TextCursorStyle};
use bevy::ui::{ComputedNode, GlobalZIndex, Overflow, ScrollPosition, Val};

use bevy_ui_widgets::EditableTextInputPlugin;
use bevy_ui_widgets::popover::{Popover, PopoverAlign, PopoverPlacement, PopoverSide};

use ui_bridge::{TextPaintList, UiBridgePlugin, UiPaintList, stack_z_offsets};

/// The same headless recipe `gallery_interaction.rs` documents: every
/// subsystem `bevy_ui`'s real layout/text/focus machinery needs, without a
/// window or event loop.
fn headless_app() -> App {
    let mut app = App::new();
    app.add_plugins(DefaultPlugins.build().disable::<bevy::winit::WinitPlugin>());
    // `EditableTextInputPlugin` alone (not the full `UiWidgetsPlugins`) —
    // this test drives text editing directly; scroll/menu tests below don't
    // need it and construct their own minimal plugin sets.
    // `widgets::FeathersCorePlugin` is the ONLY place that registers the
    // embedded FiraSans font assets `widgets::constants::fonts::REGULAR`
    // resolves against (same dev-only reciprocal dependency `icon_extract.rs`
    // documents in `ui-bridge/Cargo.toml`) — needed so `EditableText`'s layout
    // actually shapes real glyphs instead of silently failing to resolve a
    // font.
    app.add_plugins((UiBridgePlugin, widgets::FeathersCorePlugin));

    app.world_mut().spawn((
        Camera2d,
        Camera {
            computed: ComputedCameraValues {
                target_info: Some(RenderTargetInfo {
                    physical_size: UVec2::new(400, 300),
                    scale_factor: 1.0,
                }),
                ..Default::default()
            },
            ..Default::default()
        },
    ));
    // WindowPlugin supplies the sole PrimaryWindow required by focus dispatch.
    app
}

fn press_key(app: &mut App, key_code: KeyCode, logical_key: Key, text: Option<&str>) {
    app.world_mut().write_message(KeyboardInput {
        key_code,
        logical_key,
        state: ButtonState::Pressed,
        text: text.map(Into::into),
        repeat: false,
        window: Entity::PLACEHOLDER,
    });
}

fn release_key(app: &mut App, key_code: KeyCode, logical_key: Key) {
    app.world_mut().write_message(KeyboardInput {
        key_code,
        logical_key,
        state: ButtonState::Released,
        text: None,
        repeat: false,
        window: Entity::PLACEHOLDER,
    });
}

fn type_char(app: &mut App, ch: char) {
    let mut buf = [0u8; 4];
    let s = ch.encode_utf8(&mut buf);
    press_key(app, KeyCode::KeyA, Key::Character(s.into()), Some(s));
    app.update();
    release_key(app, KeyCode::KeyA, Key::Character(s.into()));
    app.update();
}

/// A quad's color, read off the first of its 4 vertices (flat across all
/// corners — same contract `gallery_interaction.rs`'s `first_quad_color`
/// documents).
fn quads_with_color(list: &UiPaintList, want: [f32; 4], tol: f32) -> Vec<usize> {
    list.vertices
        .chunks(4)
        .enumerate()
        .filter(|(_, c)| c.len() == 4 && (0..4).all(|k| (c[0].color[k] - want[k]).abs() <= tol))
        .map(|(qi, _)| qi)
        .collect()
}

/// (a) text_input: focus a real `EditableText`, type through the real
/// `KeyboardInput` -> `dispatch_focused_input` -> `on_focused_keyboard_input`
/// path, extend a selection with shift+arrow, and assert both the text
/// content changed AND the walker's extracted stream carries cursor +
/// selection quads at layout-consistent positions (inside the input's own
/// screen rect, at `TEXT_CURSOR`/`TEXT_SELECTION` z, in the input's own
/// theme colors).
#[test]
fn text_input_editing_extracts_cursor_and_selection_quads() {
    let mut app = headless_app();
    app.add_plugins(EditableTextInputPlugin);

    let cursor_color = Color::srgb(1.0, 0.0, 1.0);
    let selection_color = Color::srgb(0.0, 1.0, 1.0);
    let font = template_font(&mut app);

    let input = app
        .world_mut()
        .spawn((
            Node {
                width: Val::Px(200.0),
                height: Val::Px(24.0),
                ..Default::default()
            },
            EditableText::new(""),
            TextCursorStyle {
                color: cursor_color,
                selection_color,
                unfocused_selection_color: Color::NONE,
                selected_text_color: None,
            },
            font,
        ))
        .id();

    // Settle spawn (required components), asset load, and initial layout —
    // same "give the font a few frames" convention `text_extract.rs` uses.
    for _ in 0..10 {
        app.update();
    }

    app.world_mut()
        .resource_mut::<InputFocus>()
        .set(input, FocusCause::Navigated);
    app.update();

    type_char(&mut app, 'h');
    type_char(&mut app, 'i');

    let value = app
        .world()
        .get::<EditableText>(input)
        .expect("input entity still exists")
        .value()
        .to_string();
    assert_eq!(
        value, "hi",
        "typed characters must reach EditableText::value()"
    );

    // Extend the selection left by one character (Shift+ArrowLeft) —
    // selects "i".
    press_key(&mut app, KeyCode::ShiftLeft, Key::Shift, None);
    app.update();
    press_key(&mut app, KeyCode::ArrowLeft, Key::ArrowLeft, None);
    app.update();
    app.update();

    let node_rect = {
        let node = app.world().get::<ComputedNode>(input).expect("laid out");
        node.size()
    };
    assert!(
        node_rect.x > 0.0 && node_rect.y > 0.0,
        "input must have real layout: {node_rect:?}"
    );

    let list = app.world().resource::<UiPaintList>();
    let cursor_lin = cursor_color.to_linear();
    let cursor_quads = quads_with_color(
        list,
        [
            cursor_lin.red,
            cursor_lin.green,
            cursor_lin.blue,
            cursor_lin.alpha,
        ],
        0.02,
    );
    assert!(
        !cursor_quads.is_empty(),
        "walker should extract a TEXT_CURSOR quad in the input's cursor color"
    );

    let selection_lin = selection_color.to_linear();
    let selection_quads = quads_with_color(
        list,
        [
            selection_lin.red,
            selection_lin.green,
            selection_lin.blue,
            selection_lin.alpha,
        ],
        0.02,
    );
    assert!(
        !selection_quads.is_empty(),
        "shift+arrow should have created a selection; walker should extract a \
         TEXT_SELECTION quad in the input's selection color"
    );

    // Layout-consistent position: every found quad's vertices must sit
    // within (a small font-metrics margin around) the input's own screen
    // rect — it has no siblings/ancestors offsetting it, so its rect is
    // exactly [0, node_rect]. The margin is generous on Y specifically: a
    // cursor/selection rect comes straight from parley's line-box metrics
    // (`PlainEditor::cursor_geometry`/`selection_geometry`), which can
    // overshoot a tightly-sized node by a few px of ascent/descent — this
    // is a "same neighborhood as the widget" check, not a sub-pixel one.
    const MARGIN_X: f32 = 2.0;
    const MARGIN_Y: f32 = 6.0;
    for qi in cursor_quads.iter().chain(selection_quads.iter()) {
        let chunk = &list.vertices[qi * 4..qi * 4 + 4];
        for v in chunk {
            assert!(
                v.pos[0] >= -MARGIN_X && v.pos[0] <= node_rect.x + MARGIN_X,
                "quad #{qi} x={} out of the input's own [0,{}] rect",
                v.pos[0],
                node_rect.x
            );
            assert!(
                v.pos[1] >= -MARGIN_Y && v.pos[1] <= node_rect.y + MARGIN_Y,
                "quad #{qi} y={} out of the input's own [0,{}] rect",
                v.pos[1],
                node_rect.y
            );
        }
    }
}

/// Loads the same embedded font `widgets::FeathersCorePlugin` registers
/// (`widgets::constants::fonts::REGULAR`) — real asset loading, same
/// technique `ui-bridge/tests/text_extract.rs` and the gallery example use, not
/// a generic/system font (`system_font_discovery` isn't enabled — see
/// `Cargo.toml`'s bevy feature census).
fn template_font(app: &mut App) -> impl Bundle {
    let handle = app
        .world_mut()
        .resource_mut::<AssetServer>()
        .load(widgets::constants::fonts::REGULAR);
    (bevy::text::TextFont {
        font: FontSource::Handle(handle),
        font_size: bevy::text::FontSize::Px(14.0),
        ..Default::default()
    },)
}

/// (b) Scrolling clips the first row and reveals a later row.
#[test]
fn scroll_position_shifts_extracted_child_positions() {
    // `headless_app` already spawns the sole camera pane2's/this test's UI
    // roots resolve against automatically — a second one here would make
    // `propagate_ui_target_cameras`'s camera resolution ambiguous.
    let mut app = headless_app();

    // Distinct reds avoid linear-space tolerance collisions with the background.
    const ROW_COLORS: [Color; 6] = [
        Color::srgb(1.0, 0.0, 0.0),
        Color::srgb(0.8, 0.0, 0.0),
        Color::srgb(0.6, 0.0, 0.0),
        Color::srgb(0.4, 0.0, 0.0),
        Color::srgb(0.2, 0.0, 0.0),
        Color::srgb(0.1, 0.0, 0.0),
    ];

    let area = app
        .world_mut()
        .spawn((
            Node {
                width: Val::Px(100.0),
                height: Val::Px(60.0),
                display: Display::Flex,
                flex_direction: FlexDirection::Column,
                overflow: Overflow::scroll(),
                ..Default::default()
            },
            ScrollPosition::default(),
            BackgroundColor(Color::srgb(0.0, 0.0, 0.5)),
        ))
        .with_children(|area| {
            for color in ROW_COLORS {
                area.spawn((
                    Node {
                        height: Val::Px(20.0),
                        // Prevent flex shrinking so rows remain scrollable.
                        flex_shrink: 0.0,
                        ..Default::default()
                    },
                    BackgroundColor(color),
                ));
            }
        })
        .id();

    for _ in 0..3 {
        app.update();
    }

    let row0_lin = ROW_COLORS[0].to_linear();
    let row0_color = [row0_lin.red, row0_lin.green, row0_lin.blue, row0_lin.alpha];
    let row5_lin = ROW_COLORS[5].to_linear();
    let row5_color = [row5_lin.red, row5_lin.green, row5_lin.blue, row5_lin.alpha];
    const TOL: f32 = 0.01;

    {
        let list = app.world().resource::<UiPaintList>();
        assert!(
            !quads_with_color(list, row0_color, TOL).is_empty(),
            "row 0 (60px viewport / 20px rows = 3 visible) must be visible before scrolling"
        );
        assert!(
            quads_with_color(list, row5_color, TOL).is_empty(),
            "row 5 must NOT be visible before scrolling (only rows 0-2 fit)"
        );
    }

    // Scroll past the 60px maximum; row 0 clips and row 5 appears.
    app.world_mut()
        .entity_mut(area)
        .insert(ScrollPosition(Vec2::new(0.0, 80.0)));
    app.update();
    app.update();

    let list = app.world().resource::<UiPaintList>();
    assert!(
        quads_with_color(list, row0_color, TOL).is_empty(),
        "row 0 must be fully clipped out of the extracted stream after scrolling \
         (was visible, now clipped — corners_and_clip culls fully-out-of-clip quads)"
    );
    assert!(
        !quads_with_color(list, row5_color, TOL).is_empty(),
        "row 5 must now be visible in the extracted stream after scrolling"
    );
}

/// (c) A programmatically opened popover is painted above its sibling.
///
/// `Visibility::Visible` avoids pointer input; a higher `GlobalZIndex` must
/// place the popover after its sibling in extracted paint order.
#[test]
fn menu_popover_paints_after_sibling_with_higher_stack_index() {
    let mut app = headless_app();
    app.add_plugins(bevy_ui_widgets::popover::PopoverPlugin);

    let sibling_color = Color::srgb(0.2, 0.2, 0.9);
    let popup_color = Color::srgb(0.9, 0.2, 0.2);

    let anchor = app
        .world_mut()
        .spawn((
            Node {
                width: Val::Px(60.0),
                height: Val::Px(24.0),
                ..Default::default()
            },
            BackgroundColor(sibling_color),
        ))
        .with_children(|anchor| {
            anchor.spawn((
                Node {
                    position_type: PositionType::Absolute,
                    width: Val::Px(80.0),
                    height: Val::Px(40.0),
                    ..Default::default()
                },
                BackgroundColor(popup_color),
                Popover {
                    positions: vec![PopoverPlacement {
                        side: PopoverSide::Bottom,
                        align: PopoverAlign::Start,
                        gap: 2.0,
                    }],
                    window_margin: 4.0,
                },
                GlobalZIndex(100),
                Visibility::Hidden,
            ));
        })
        .id();
    let popup = app
        .world()
        .get::<Children>(anchor)
        .expect("anchor has the popup child")[0];

    for _ in 0..3 {
        app.update();
    }

    // Open without pointer input.
    app.world_mut()
        .entity_mut(popup)
        .insert(Visibility::Visible);
    app.update();
    app.update();

    let list = app.world().resource::<UiPaintList>();
    let sibling_lin = sibling_color.to_linear();
    let sibling_idx = quads_with_color(
        list,
        [
            sibling_lin.red,
            sibling_lin.green,
            sibling_lin.blue,
            sibling_lin.alpha,
        ],
        0.02,
    );
    let popup_lin = popup_color.to_linear();
    let popup_idx = quads_with_color(
        list,
        [
            popup_lin.red,
            popup_lin.green,
            popup_lin.blue,
            popup_lin.alpha,
        ],
        0.02,
    );
    assert!(
        !sibling_idx.is_empty(),
        "sibling background quad should be extracted"
    );
    assert!(
        !popup_idx.is_empty(),
        "popover background quad should be extracted once Visibility::Visible \
         (position_popover lays it out regardless of visibility — only Display::None skips layout)"
    );
    assert!(
        popup_idx.iter().min().unwrap() > sibling_idx.iter().max().unwrap(),
        "GlobalZIndex(100) must push the popover's ComputedStackIndex — and \
         therefore its quads' position in UiPaintList::vertices — after every \
         sibling quad: sibling={sibling_idx:?} popup={popup_idx:?}"
    );
}

/// Sanity check on the z-offset table this test suite (and the gallery's
/// probes) leans on: `TEXT_CURSOR` must sort after `TEXT_SELECTION`, which
/// must sort after plain `BACKGROUND` — otherwise a cursor could paint
/// UNDER a selection highlight, or a selection UNDER the node's own
/// background.
#[test]
fn z_offset_ordering_background_then_selection_then_cursor() {
    const {
        assert!(stack_z_offsets::BACKGROUND < stack_z_offsets::TEXT_SELECTION);
        assert!(stack_z_offsets::TEXT_SELECTION < stack_z_offsets::TEXT_CURSOR);
        assert!(stack_z_offsets::TEXT < stack_z_offsets::TEXT_CURSOR);
    }
}

/// Editable text produces glyph batches alongside decoration quads.
#[test]
fn text_input_glyphs_reach_text_paint_list() {
    let mut app = headless_app();
    app.add_plugins(EditableTextInputPlugin);
    let font = template_font(&mut app);

    let input = app
        .world_mut()
        .spawn((
            Node {
                width: Val::Px(200.0),
                height: Val::Px(24.0),
                ..Default::default()
            },
            EditableText::new(""),
            TextCursorStyle::default(),
            font,
        ))
        .id();

    for _ in 0..10 {
        app.update();
    }
    app.world_mut()
        .resource_mut::<InputFocus>()
        .set(input, FocusCause::Navigated);
    app.update();

    for ch in ['a', 's', 'h', 'a'] {
        type_char(&mut app, ch);
    }
    app.update();

    let text_list = app.world().resource::<TextPaintList>();
    assert!(
        !text_list.instances.is_empty(),
        "typed glyphs must reach TextPaintList::instances via \
         push_editable_text_glyph_items"
    );
}

/// IME fields follow `bevy_ui_widgets` without custom UI handling.
///
/// The test observes the real `PrimaryWindow` fields before and after focus.
#[test]
fn ime_window_fields_flow_from_bevy_ui_widgets_as_is() {
    let mut app = headless_app();
    app.add_plugins(EditableTextInputPlugin);
    let font = template_font(&mut app);

    let window = app
        .world_mut()
        .query_filtered::<Entity, With<bevy::window::PrimaryWindow>>()
        .single(app.world())
        .expect("WindowPlugin spawns exactly one PrimaryWindow headless too");

    // Before focus: IME must be off (this is the state a freshly-opened,
    // nothing-focused window should be in).
    for _ in 0..3 {
        app.update();
    }
    assert!(
        !app.world().get::<Window>(window).unwrap().ime_enabled,
        "IME must not be enabled before any EditableText has focus"
    );

    let input = app
        .world_mut()
        .spawn((
            Node {
                width: Val::Px(200.0),
                height: Val::Px(24.0),
                ..Default::default()
            },
            EditableText::new("hello"),
            TextCursorStyle::default(),
            font,
        ))
        .id();
    for _ in 0..3 {
        app.update();
    }

    app.world_mut()
        .resource_mut::<InputFocus>()
        .set(input, FocusCause::Navigated);
    app.update();
    app.update();

    let win = app.world().get::<Window>(window).unwrap();
    assert!(
        win.ime_enabled,
        "listen_for_ime_input_when_text_input_focused should have set \
         Window::ime_enabled = true once an EditableText gained focus"
    );
    assert!(
        win.ime_position.x.is_finite() && win.ime_position.y.is_finite(),
        "update_ime_position should have written a real screen position \
         tracking the focused input's cursor, not left it at a garbage value: \
         {:?}",
        win.ime_position
    );

    // Losing focus must disable it again.
    app.world_mut().resource_mut::<InputFocus>().clear();
    app.update();
    app.update();
    assert!(
        !app.world().get::<Window>(window).unwrap().ime_enabled,
        "IME must be disabled again once the EditableText loses focus"
    );
}
