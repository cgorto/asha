//! Feathers widget gallery rendered through asha's UI pipeline.
//!
//! Exercises widget construction, theming, input, layout, and paint extraction.
//!
//! Legacy bundle constructors remain for compatibility. Scene components use
//! `bsn!` so `bevy_scene` receives their required scene metadata.
//!
//! Absolute hosts keep legacy bundle geometry deterministic.
//!
//! Draw calls merge UI, shadow, and text batches by their global paint order.
//!
//! `ASHA_VERIFY=1` renders an offscreen float target and checks paint probes.
//!
//! Run windowed with `cargo run -p widgets --example feathers_gallery`.
//! Run probes with `ASHA_VERIFY=1 cargo run -p widgets --example feathers_gallery`.

use bevy::color::Color;
use bevy::ecs::spawn::Spawn;
use bevy::image::Image;
use bevy::input_focus::{FocusCause, InputFocus};
use bevy::math::Rect;
use bevy::picking::hover::Hovered;
use bevy::prelude::*;
use bevy::text::{EditableText, FontSource, FontWeight};
use bevy::ui::widget::ImageNode;
use bevy::ui::{
    BackgroundColor, BorderRadius, Checked, Display, FlexDirection, Overflow, PositionType,
    ScrollPosition, Val,
};
use bevy::window::{Window, WindowPlugin, WindowResolution};

use bevy_ui_widgets::{
    ControlOrientation, RadioGroup, Scrollbar, ScrollbarThumb, SliderPrecision, UiWidgetsPlugins,
    ValueChange, checkbox_self_update, radio_self_update, slider_self_update,
};

use gpu::{
    CommandBuffer, Gpu, HazardFlags, LoadOp, Memory, OwnedTexture, Queue, RenderAttachment,
    RenderPassDesc, Stage, StoreOp, TextureDesc, TextureFormat, UsageFlags,
};
use render::{AshaRenderPlugin, FrameCtx, PacingPlugin, RenderScene};
use ui_bridge::{AshaRenderPluginExt, ICON_PATHS, TextRunBatch, UiBatch, UiBridge, UiBridgePlugin};
use widgets::controls::{
    ColorChannel, ColorSwatchValue, FeathersColorPlane, FeathersColorSliderProps,
    FeathersColorSwatch, FeathersMenu, FeathersMenuButton, FeathersMenuItem, FeathersMenuPopup,
    FeathersNumberInput, FeathersSlider, FeathersTextInput, FeathersTextInputContainer,
    FeathersToggleSwitch,
};
use widgets::cursor::EntityCursor;
use widgets::rounded_corners::RoundedCorners;
use widgets::theme::{ThemeBackgroundColor, ThemeTextColor, ThemedText, UiTheme};
use widgets::{constants, controls, dark_theme, tokens};
// Aliases distinguish ABI oracle math from UI layout math at call sites;
// both use glam 0.32.1.
use abi_core::glam::{Vec2 as GVec2, Vec4 as GVec4};
use abi_ui::{
    UI_MODE_ALPHA_PATTERN, UI_MODE_COLOR_PLANE, UI_MODE_SHIFT, UiMaterialData, UiShadowVertex,
    UiVertex, ui_alpha_pattern_shade, ui_color_plane_shade, ui_shadow_shade,
};
use text::{TextGlyphInstance, TextPass, TextPassTarget};

// Reserve space for the color-tools panel.
const WINDOW_W: u32 = 1100;
const WINDOW_H: u32 = 400;

const PANE_X: f32 = 20.0;
const PANE_Y: f32 = 20.0;
const PANE_W: f32 = 300.0;
const PANE_H: f32 = 320.0;
const PANE_RADIUS: f32 = 20.0;

const LABEL_X: f32 = 16.0;
const LABEL_Y: f32 = 14.0;
const LABEL_W: f32 = 200.0;
const LABEL_H: f32 = 24.0;
const LABEL_FONT_PX: f32 = 14.0;

const BTN_A_X: f32 = 16.0;
const BTN_A_Y: f32 = 50.0;
const BTN_A_W: f32 = 90.0;
const BTN_A_H: f32 = 28.0;

const BTN_B_X: f32 = 118.0;
const BTN_B_Y: f32 = 50.0;
const BTN_B_W: f32 = 44.0;
const BTN_B_H: f32 = 28.0;
const ICON_SIZE: f32 = 24.0;

const CHECK_X: f32 = 16.0;
const CHECK_Y: f32 = 92.0;
const CHECK_W: f32 = 160.0;
const CHECK_H: f32 = 22.0;

const RADIO_X: f32 = 16.0;
const RADIO_Y: f32 = 124.0;
const RADIO_W: f32 = 200.0;
const RADIO_H: f32 = 80.0;

const TOGGLE_X: f32 = 16.0;
const TOGGLE_Y: f32 = 212.0;
const TOGGLE_W: f32 = 32.0;
const TOGGLE_H: f32 = 18.0;

const SLIDER_X: f32 = 16.0;
const SLIDER_Y: f32 = 244.0;
const SLIDER_W: f32 = 240.0;
const SLIDER_H: f32 = 24.0;

// The flex-layout pane uses dynamic paint-stream probes.
const PANE2_X: f32 = 340.0;
const PANE2_Y: f32 = 20.0;
const PANE2_W: f32 = 460.0;
const PANE2_H: f32 = 360.0;

// Color tools use legacy bundles except scene-component widgets.
const PANE3_X: f32 = 840.0;
const PANE3_Y: f32 = 20.0;
const PANE3_W: f32 = 240.0;
const PANE3_H: f32 = 360.0;

const CP_PLANE_X: f32 = 16.0;
const CP_PLANE_Y: f32 = 16.0;
const CP_PLANE_W: f32 = 208.0;
const CP_PLANE_H: f32 = 140.0;

const CP_SLIDER_X: f32 = 16.0;
const CP_SLIDER_Y: f32 = 168.0;
const CP_SLIDER_W: f32 = 208.0;
const CP_SLIDER_H: f32 = 24.0;

const CP_SWATCH_X: f32 = 16.0;
const CP_SWATCH_Y: f32 = 204.0;
const CP_SWATCH_W: f32 = 40.0;
const CP_SWATCH_H: f32 = 40.0;

/// The scroll area's own background — deliberately NOT one of widgets' theme
/// tokens. `tokens::PANE_BODY_BG`, `tokens::SUBPANE_BODY_BG`, and
/// `tokens::MENU_BG` all resolve to the exact same dark-theme gray
/// (`palette::GRAY_1`, verified in `dark_theme.rs`), so probe (g)'s
/// find-the-scroll-viewport-by-color scan would otherwise ambiguously hit
/// pane2's own root background (also `GRAY_1`) first. A one-off, distinct
/// solid color sidesteps the collision entirely — this widget's OWN theming
/// isn't under test here, its scroll behavior is.
fn scroll_area_bg() -> Color {
    Color::srgb(0.05, 0.45, 0.15)
}

/// Translucent swatch value used by the color-tool probes.
fn swatch_value_color() -> Color {
    Color::srgba(0.3, 0.6, 0.9, 0.4)
}

/// Paint-probe marker for the text input's editable leaf.
#[derive(Component, Clone, Copy, Default)]
struct GalleryTextInput;

/// Marker on the number input's [`EditableText`] leaf entity.
#[derive(Component, Clone, Copy, Default)]
struct GalleryNumberInput;

/// Row hosting the scroll area and its target scrollbar.
#[derive(Component, Clone, Copy, Default)]
struct GalleryScrollRow;

/// Marker on the scrollable content area itself (`Overflow::scroll` +
/// `ScrollArea`).
#[derive(Component, Clone, Copy, Default)]
struct GalleryScrollArea;

/// Identifies a scroll row for paint probes.
#[derive(Component, Clone, Copy, Default)]
struct GalleryScrollRowIndex(u32);

/// Identifies the verification popover.
#[derive(Component, Clone, Copy, Default)]
struct GalleryMenuPopup;

/// Absolutely positioned host for a legacy bundle.
fn abs_node(x: f32, y: f32, w: f32, h: f32) -> Node {
    Node {
        position_type: PositionType::Absolute,
        left: Val::Px(x),
        top: Val::Px(y),
        width: Val::Px(w),
        height: Val::Px(h),
        ..Default::default()
    }
}

/// BSN scene for the flex-layout pane.
///
/// The scrollbar is added after its target exists.
/// Standalone rows supply their own embedded font.
fn scroll_row(idx: u32, caption: &'static str) -> impl Scene {
    bsn! {
        Node { height: px(20) }
        GalleryScrollRowIndex(idx)
        Children [
            (
                Text(caption)
                // Supply an embedded font for standalone rows.
                template(|ctx| {
                    Ok(TextFont {
                        font: FontSource::Handle(ctx.resource::<AssetServer>().load(constants::fonts::REGULAR)),
                        font_size: FontSize::Px(14.0),
                        ..Default::default()
                    })
                })
                ThemedText
            )
        ]
    }
}

fn flex_pane_scene() -> impl Scene {
    bsn! {
        Node {
            position_type: PositionType::Absolute,
            left: px(PANE2_X),
            top: px(PANE2_Y),
            width: px(PANE2_W),
            height: px(PANE2_H),
            display: Display::Flex,
            flex_direction: FlexDirection::Column,
            row_gap: px(10),
            padding: px(16),
            border_radius: {RoundedCorners::All.to_border_radius(PANE_RADIUS)},
        }
        ThemeBackgroundColor(tokens::PANE_BODY_BG)
        Children [
            (
                Node {
                    display: Display::Flex,
                    flex_direction: FlexDirection::Row,
                    align_items: AlignItems::Center,
                    column_gap: px(8),
                }
                Children [
                    (
                        @FeathersTextInputContainer
                        Node { flex_grow: 1.0 }
                        Children [
                            (
                                // Flex sizing avoids font-load measurement races.
                                @FeathersTextInput {
                                    @max_characters: 24usize,
                                }
                                EditableText::new("asha")
                                GalleryTextInput
                            )
                        ]
                    )
                ]
            ),
            (
                @FeathersNumberInput
                GalleryNumberInput
                Node { max_width: px(100) }
            ),
            // Scroll area; the scrollbar is added after spawning.
            (
                Node {
                    display: Display::Flex,
                    flex_direction: FlexDirection::Row,
                    height: px(120),
                    column_gap: px(2),
                }
                GalleryScrollRow
                Children [
                    (
                        Node {
                            display: Display::Flex,
                            flex_direction: FlexDirection::Column,
                            flex_grow: 1.0,
                            padding: px(4),
                            overflow: Overflow::scroll(),
                        }
                        GalleryScrollArea
                        BackgroundColor({scroll_area_bg()})
                        Children [
                            scroll_row(0, "Row 0"),
                            scroll_row(1, "Row 1"),
                            scroll_row(2, "Row 2"),
                            scroll_row(3, "Row 3"),
                            scroll_row(4, "Row 4"),
                            scroll_row(5, "Row 5"),
                            scroll_row(6, "Row 6"),
                            scroll_row(7, "Row 7"),
                        ]
                    )
                ]
            ),
            // Menu row and verification popover.
            (
                @FeathersMenu
                Children [
                    (
                        @FeathersMenuButton {
                            @caption: bsn! { Text("Menu") ThemedText },
                        }
                        Node { flex_grow: 0.0, width: px(90) }
                    ),
                    (
                        @FeathersMenuPopup
                        GalleryMenuPopup
                        Children [
                            (@FeathersMenuItem { @caption: bsn! { Text("Item 1") ThemedText } }),
                            (@FeathersMenuItem { @caption: bsn! { Text("Item 2") ThemedText } }),
                        ]
                    )
                ]
            ),
        ]
    }
}

/// Completes dynamic scrollbar and verification setup.
fn prime_dynamic_widgets(
    mut commands: Commands,
    mut done: Local<bool>,
    q_scroll_row: Query<Entity, With<GalleryScrollRow>>,
    q_scroll_area: Query<Entity, With<GalleryScrollArea>>,
    q_popup: Query<Entity, With<GalleryMenuPopup>>,
    q_text_input: Query<Entity, With<GalleryTextInput>>,
    mut focus: ResMut<InputFocus>,
) {
    if !*done
        && let (Ok(row), Ok(area), Ok(popup)) = (
            q_scroll_row.single(),
            q_scroll_area.single(),
            q_popup.single(),
        )
    {
        *done = true;

        // Build the scrollbar after its target entity exists.
        commands.entity(row).with_children(|row| {
            row.spawn((
                Node {
                    width: Val::Px(10.0),
                    border_radius: BorderRadius::all(Val::Px(3.0)),
                    ..Default::default()
                },
                ThemeBackgroundColor(tokens::SCROLLBAR_BG),
                Scrollbar {
                    orientation: ControlOrientation::Vertical,
                    target: area,
                    min_thumb_length: 8.0,
                },
                Children::spawn(Spawn((
                    Hovered::default(),
                    ThemeBackgroundColor(tokens::SCROLLBAR_THUMB),
                    ScrollbarThumb {
                        border_radius: BorderRadius::all(Val::Px(3.0)),
                        ..Default::default()
                    },
                    EntityCursor::System(bevy::window::SystemCursorIcon::Pointer),
                ))),
            ));
        });

        // Scroll row 0 beyond the clip rectangle for verification.
        commands
            .entity(area)
            .insert(ScrollPosition(Vec2::new(0.0, 40.0)));

        // Verification opens the popover without pointer input.
        if std::env::var_os("ASHA_VERIFY").is_some() {
            commands.entity(popup).insert(Visibility::Visible);
        }
    }

    // Verification keeps the text cursor visible.
    if std::env::var_os("ASHA_VERIFY").is_some()
        && let Ok(input) = q_text_input.single()
    {
        focus.set(input, FocusCause::Navigated);
    }
}

/// Spawns a checked toggle through its scene component.
///
/// `bsn!` supplies the scene metadata required by `bevy_scene`.
fn toggle_switch_scene() -> impl Scene {
    bsn! {
        Node {
            position_type: PositionType::Absolute,
            left: px(TOGGLE_X),
            top: px(TOGGLE_Y),
            width: px(TOGGLE_W),
            height: px(TOGGLE_H),
        }
        Children [(
            @FeathersToggleSwitch
            Checked
            // Controlled widgets require explicit value writeback.
            on(checkbox_self_update)
        )]
    }
}

/// Spawns a mid-value slider through its scene component.
fn slider_scene() -> impl Scene {
    bsn! {
        Node {
            position_type: PositionType::Absolute,
            left: px(SLIDER_X),
            top: px(SLIDER_Y),
            width: px(SLIDER_W),
            height: px(SLIDER_H),
        }
        Children [(
            @FeathersSlider {
                @value: 50.0,
                @min: 0.0,
                @max: 100.0,
            }
            SliderPrecision(0)
            // Controlled widget; write back emitted values.
            on(slider_self_update)
        )]
    }
}

/// Spawns a hue/lightness plane through its enum scene component.
///
/// The column host is required for cross-axis stretching.
fn color_plane_scene() -> impl Scene {
    bsn! {
        Node {
            position_type: PositionType::Absolute,
            left: px(CP_PLANE_X),
            top: px(CP_PLANE_Y),
            width: px(CP_PLANE_W),
            height: px(CP_PLANE_H),
            flex_direction: FlexDirection::Column,
        }
        Children [(
            @FeathersColorPlane::HueLightness
            // Store picked coordinates and update the swatch hue/lightness.
            on(|change: On<ValueChange<bevy::math::Vec2>>,
                mut q_value: Query<&mut widgets::controls::ColorPlaneValue>,
                mut q_swatch: Query<&mut ColorSwatchValue>| {
                if let Ok(mut value) = q_value.get_mut(change.source) {
                    value.0.x = change.value.x;
                    value.0.y = change.value.y;
                }
                for mut swatch in q_swatch.iter_mut() {
                    let hsla = bevy::color::Hsla::from(swatch.0);
                    swatch.0 = bevy::color::Color::from(bevy::color::Hsla {
                        hue: change.value.x * 360.0,
                        lightness: 1.0 - change.value.y,
                        ..hsla
                    });
                }
            })
        )]
    }
}

/// Spawns a translucent swatch with an alpha-pattern background.
fn color_swatch_scene() -> impl Scene {
    bsn! {
        Node {
            position_type: PositionType::Absolute,
            left: px(CP_SWATCH_X),
            top: px(CP_SWATCH_Y),
            width: px(CP_SWATCH_W),
            height: px(CP_SWATCH_H),
        }
        Children [(
            @FeathersColorSwatch
            ColorSwatchValue({swatch_value_color()})
        )]
    }
}

#[allow(deprecated)]
fn spawn_scene(mut commands: Commands, asset_server: Res<AssetServer>) {
    commands.spawn(Camera2d);

    let label_font: Handle<bevy::text::Font> = asset_server.load(constants::fonts::REGULAR);
    let icon: Handle<Image> = asset_server.load(ICON_PATHS[0]);

    let root_id = commands
        .spawn((
            Node {
                width: Val::Px(WINDOW_W as f32),
                height: Val::Px(WINDOW_H as f32),
                ..Default::default()
            },
            ThemeBackgroundColor(tokens::WINDOW_BG),
        ))
        .id();

    // Parent all panes under one root for deterministic paint order.
    let flex_pane_id = commands.spawn_scene(flex_pane_scene()).id();
    commands.entity(flex_pane_id).insert(ChildOf(root_id));

    let pane_id = commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                left: Val::Px(PANE_X),
                top: Val::Px(PANE_Y),
                width: Val::Px(PANE_W),
                height: Val::Px(PANE_H),
                border_radius: RoundedCorners::All.to_border_radius(PANE_RADIUS),
                ..Default::default()
            },
            ThemeBackgroundColor(tokens::PANE_BODY_BG),
        ))
        .id();
    commands.entity(pane_id).insert(ChildOf(root_id));
    commands.entity(pane_id).with_children(|pane| {
        // Clip the label to give probes a stable text-run anchor.
        pane.spawn((Node {
            position_type: PositionType::Absolute,
            left: Val::Px(LABEL_X),
            top: Val::Px(LABEL_Y),
            width: Val::Px(LABEL_W),
            height: Val::Px(LABEL_H),
            overflow: Overflow::clip(),
            ..Default::default()
        },))
            .with_children(|c| {
                c.spawn((
                    Text::new("Feathers Gallery"),
                    TextFont {
                        font: label_font.clone().into(),
                        font_size: FontSize::Px(LABEL_FONT_PX),
                        weight: FontWeight::NORMAL,
                        ..Default::default()
                    },
                    ThemeTextColor(tokens::TEXT_MAIN),
                    ThemedText,
                ));
            });

        pane.spawn(abs_node(BTN_A_X, BTN_A_Y, BTN_A_W, BTN_A_H))
            .with_children(|w| {
                w.spawn(controls::button_bundle(
                    controls::ButtonBundleProps::default(),
                    (),
                    Spawn((Text::new("OK"), ThemedText)),
                ));
            });

        // Icon button.
        pane.spawn(abs_node(BTN_B_X, BTN_B_Y, BTN_B_W, BTN_B_H))
            .with_children(|w| {
                w.spawn(controls::button_bundle(
                    controls::ButtonBundleProps::default(),
                    (),
                    Spawn((
                        Node {
                            width: Val::Px(ICON_SIZE),
                            height: Val::Px(ICON_SIZE),
                            ..Default::default()
                        },
                        ImageNode {
                            image: icon.clone(),
                            color: Color::srgb(1.0, 0.0, 0.0),
                            ..Default::default()
                        },
                    )),
                ));
            });

        pane.spawn(abs_node(CHECK_X, CHECK_Y, CHECK_W, CHECK_H))
            .with_children(|w| {
                w.spawn(controls::checkbox_bundle(
                    (Checked,),
                    Spawn((Text::new("Enabled"), ThemedText)),
                ))
                // Controlled widget — see `toggle_switch_scene`'s comment.
                .observe(checkbox_self_update);
            });

        pane.spawn((
            Node {
                position_type: PositionType::Absolute,
                left: Val::Px(RADIO_X),
                top: Val::Px(RADIO_Y),
                width: Val::Px(RADIO_W),
                height: Val::Px(RADIO_H),
                display: Display::Flex,
                flex_direction: FlexDirection::Column,
                row_gap: Val::Px(4.0),
                ..Default::default()
            },
            RadioGroup,
        ))
        // Controlled widget — the group's `ValueChange<Entity>` must move
        // `Checked` between radios; see `toggle_switch_scene`'s comment.
        .observe(radio_self_update)
        .with_children(|g| {
            g.spawn(controls::radio_bundle(
                (),
                Spawn((Text::new("Alpha"), ThemedText)),
            ));
            g.spawn(controls::radio_bundle(
                (Checked,),
                Spawn((Text::new("Beta"), ThemedText)),
            ));
            g.spawn(controls::radio_bundle(
                (),
                Spawn((Text::new("Gamma"), ThemedText)),
            ));
        });
    });

    // Scene-component widgets are parented after spawning.
    let toggle_id = commands.spawn_scene(toggle_switch_scene()).id();
    commands.entity(toggle_id).insert(ChildOf(pane_id));

    // Mid-value slider.
    let slider_id = commands.spawn_scene(slider_scene()).id();
    commands.entity(slider_id).insert(ChildOf(pane_id));

    // Color-tools panel.
    let pane3_id = commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                left: Val::Px(PANE3_X),
                top: Val::Px(PANE3_Y),
                width: Val::Px(PANE3_W),
                height: Val::Px(PANE3_H),
                border_radius: RoundedCorners::All.to_border_radius(PANE_RADIUS),
                ..Default::default()
            },
            ThemeBackgroundColor(tokens::PANE_BODY_BG),
        ))
        .id();
    commands.entity(pane3_id).insert(ChildOf(root_id));

    // Color plane: hue/lightness axes, default value (0,0,0) — the
    // walker-side probes read its actual variant/fixed_channel back off the
    // extracted `UiVertex` stream rather than assuming this constructor's
    // exact defaults. Spawned first (before the color slider below) so it
    // lands first in `pane3`'s `Children` list, matching the original
    // imperative spawn order. See `color_plane_scene`'s doc comment for why
    // it's `bsn!`-spawned now (`FeathersColorPlane` is a scene component)
    // and for the Column-direction host requirement.
    let color_plane_id = commands.spawn_scene(color_plane_scene()).id();
    commands.entity(color_plane_id).insert(ChildOf(pane3_id));

    commands.entity(pane3_id).with_children(|pane3| {
        // Color slider: hue channel, mid value — its track carries an
        // alpha-pattern checkerboard behind the gradient fill.
        // `color_slider_bundle` never inserts `FeathersColorSlider` (it
        // only inserts the plain `ColorSlider` component), so — unlike
        // toggle switch/slider/color-plane/color-swatch — it never trips
        // the scene-component ERROR hook and is left as the deprecated
        // `*_bundle` call.
        pane3
            .spawn(abs_node(CP_SLIDER_X, CP_SLIDER_Y, CP_SLIDER_W, CP_SLIDER_H))
            .with_children(|w| {
                w.spawn(controls::color_slider_bundle(
                    FeathersColorSliderProps {
                        value: 180.0,
                        channel: ColorChannel::HslHue,
                    },
                    (),
                ))
                // Controlled widget (its value is a plain `SliderValue`) —
                // see `toggle_switch_scene`'s comment.
                .observe(slider_self_update)
                // …and forward the picked hue (HslHue channel, 0..360)
                // into the swatch preview, keeping its saturation/
                // lightness/alpha. Interaction-driven only — probes safe.
                .observe(
                    |change: On<ValueChange<f32>>, mut q_swatch: Query<&mut ColorSwatchValue>| {
                        for mut swatch in q_swatch.iter_mut() {
                            let hsla = bevy::color::Hsla::from(swatch.0);
                            swatch.0 = bevy::color::Color::from(bevy::color::Hsla {
                                hue: change.value,
                                ..hsla
                            });
                        }
                    },
                );
            });
    });

    // Keep the swatch translucent so its checkerboard remains visible.
    let color_swatch_id = commands.spawn_scene(color_swatch_scene()).id();
    commands.entity(color_swatch_id).insert(ChildOf(pane3_id));
}

/// Merges UI, shadow, and text batches by their shared paint order.
///
/// Each batch uses a separate pass call; barriers protect attachment writes.
#[allow(clippy::too_many_arguments)]
fn draw_merged(
    gpu: &Gpu,
    cb: CommandBuffer,
    ui_pass: &ui::UiPass,
    text_pass: &TextPass,
    ui_batches: &[ui::UiBatch],
    ui_orders: &[u32],
    shadow_batches: &[ui::UiShadowBatch],
    shadow_orders: &[u32],
    text_batches: &[text::TextBatchDesc],
    text_orders: &[u32],
    texture: gpu::Texture,
    size: [u32; 2],
    clear_color: [f32; 4],
) {
    #[derive(Clone, Copy)]
    enum Lane {
        Ui,
        Shadow,
        Text,
    }

    let mut i = 0usize;
    let mut j = 0usize;
    let mut k = 0usize;
    let mut first = true;
    loop {
        let candidates = [
            (i < ui_batches.len()).then(|| (ui_orders[i], Lane::Ui)),
            (j < shadow_batches.len()).then(|| (shadow_orders[j], Lane::Shadow)),
            (k < text_batches.len()).then(|| (text_orders[k], Lane::Text)),
        ];
        // Paint orders are globally unique.
        let Some((_, lane)) = candidates
            .into_iter()
            .flatten()
            .min_by_key(|(order, _)| *order)
        else {
            break;
        };

        if !first {
            // Protect consecutive writes to the shared color attachment.
            gpu.cmd_barrier(
                cb,
                Stage::RasterColorOut,
                Stage::RasterColorOut,
                HazardFlags::empty(),
            );
        }
        let load_op = if first { LoadOp::Clear } else { LoadOp::Load };

        match lane {
            Lane::Ui => {
                ui_pass.record(
                    gpu,
                    cb,
                    ui::UiPassTarget {
                        texture,
                        size,
                        load_op,
                        store_op: StoreOp::Store,
                        clear_color,
                    },
                    &ui_batches[i..=i],
                );
                i += 1;
            }
            Lane::Shadow => {
                ui_pass.record_shadows(
                    gpu,
                    cb,
                    ui::UiPassTarget {
                        texture,
                        size,
                        load_op,
                        store_op: StoreOp::Store,
                        clear_color,
                    },
                    &shadow_batches[j..=j],
                );
                j += 1;
            }
            Lane::Text => {
                text_pass.record_batches(
                    gpu,
                    cb,
                    TextPassTarget {
                        texture,
                        size,
                        load_op,
                        store_op: StoreOp::Store,
                        clear_color,
                        scissor: None,
                    },
                    &text_batches[k..=k],
                );
                k += 1;
            }
        }
        first = false;
    }

    if first {
        // Nothing painted at all: still clear the target explicitly.
        gpu.cmd_begin_render_pass(
            cb,
            RenderPassDesc {
                render_area_size: size,
                color_attachments: &[RenderAttachment {
                    texture,
                    load_op: LoadOp::Clear,
                    store_op: StoreOp::Store,
                    clear_color,
                    ..Default::default()
                }],
                ..Default::default()
            },
        );
        gpu.cmd_end_render_pass(cb);
    }
}

/// `ASHA_VERIFY=1` plumbing: an offscreen `Rgba32Float` target (linear —
/// see the module doc) plus a mapped readback buffer, sized to the window.
struct VerifyTarget {
    target: OwnedTexture,
    readback: gpu::Ptr<[f32; 4]>,
}

/// Settle window: layout/theme/icon-load convention borrowed from
/// `text_extract.rs` (settles by frame 10) layered on `icon_extract.rs`'s
/// "tolerate real async timing, give up loudly well past any plausible load
/// time" convention (the gallery has far more plugins/systems in its
/// `Update` graph than either seam test alone).
const VERIFY_MIN_FRAME: u64 = 10;
const GIVE_UP_FRAME: u64 = 180;

struct GalleryScene {
    bridge: UiBridge,
    ui_pass: Option<ui::UiPass>,
    text_pass: Option<TextPass>,
    heap: Option<gpu::HeapSlots>,
    verify: Option<VerifyTarget>,
    verified: bool,
}

impl GalleryScene {
    fn new(gpu: &Gpu) -> Self {
        let ui_pass = ui::UiPass::new(gpu);
        let text_pass = TextPass::new(gpu);
        // 1 reserved + the icon + headroom; 1 reserved + the icon sampler.
        let heap = gpu.heap_slots_create(8, 2, 4);
        let verify = std::env::var_os("ASHA_VERIFY").map(|_| {
            let target = gpu.texture_alloc_and_create(
                TextureDesc {
                    dimensions: [WINDOW_W, WINDOW_H, 1],
                    format: TextureFormat::Rgba32Float,
                    usage: UsageFlags::COLOR_ATTACHMENT | UsageFlags::TRANSFER_SRC,
                    ..Default::default()
                },
                Queue::Main,
                None,
            );
            let readback =
                gpu.alloc_slice::<[f32; 4]>((WINDOW_W * WINDOW_H) as u64, Memory::Readback);
            VerifyTarget { target, readback }
        });

        Self {
            bridge: UiBridge::new(),
            ui_pass: Some(ui_pass),
            text_pass: Some(text_pass),
            heap: Some(heap),
            verify,
            verified: false,
        }
    }

    fn check_probes(&self, ctx: &mut FrameCtx, readback: gpu::Ptr<[f32; 4]>) {
        let theme = UiTheme(dark_theme::create_dark_theme());
        let color_arr = |token: &widgets::theme::ThemeToken| -> [f32; 4] {
            let c = theme.color(token).to_linear();
            [c.red, c.green, c.blue, c.alpha]
        };

        let mut pixels = vec![[0.0f32; 4]; (WINDOW_W * WINDOW_H) as usize];
        // SAFETY: readback was allocated for exactly WINDOW_W*WINDOW_H
        // pixels and the copy that filled it just finished
        // (queue_wait_idle fenced it before this is called).
        unsafe {
            std::ptr::copy_nonoverlapping(readback.cpu, pixels.as_mut_ptr(), pixels.len());
        }
        let pixel_at = |x: u32, y: u32| -> [f32; 4] { pixels[(y * WINDOW_W + x) as usize] };
        let close = |a: [f32; 4], b: [f32; 4], tol: f32| (0..4).all(|k| (a[k] - b[k]).abs() <= tol);

        fn report(ok: &mut bool, name: &str, expected: [f32; 4], actual: [f32; 4], pass: bool) {
            println!(
                "PROBE {name}: expected={expected:?} actual={actual:?} {}",
                if pass { "OK" } else { "FAIL" }
            );
            *ok &= pass;
        }

        let mut ok = true;

        // (a) Button A background == the dark-theme normal-button token.
        let btn_expect = color_arr(&tokens::BUTTON_BG);
        let btn_actual = pixel_at(81, 82);
        report(
            &mut ok,
            "button bg",
            btn_expect,
            btn_actual,
            close(btn_expect, btn_actual, 0.05),
        );

        // (b) Slider track: two probes straddling the mid-value split — the
        // bar color on the left, the (dimmer) track color on the right.
        let bar_expect = color_arr(&tokens::SLIDER_BAR);
        let bar_actual = pixel_at(60, 276);
        report(
            &mut ok,
            "slider bar",
            bar_expect,
            bar_actual,
            close(bar_expect, bar_actual, 0.05),
        );
        let track_expect = color_arr(&tokens::SLIDER_BG);
        let track_actual = pixel_at(250, 276);
        report(
            &mut ok,
            "slider track",
            track_expect,
            track_actual,
            close(track_expect, track_actual, 0.05),
        );
        println!(
            "PROBE slider bar vs track differ: {}",
            bar_actual != track_actual
        );
        ok &= bar_actual != track_actual;

        // (c) Nonzero text coverage in the standalone label's rect: locate
        // its TextRunBatch by clip match (screen rect of the LABEL
        // container), then probe near its first glyph's pen position — same
        // technique as `ui-bridge/tests/text_extract.rs`.
        let want_clip = Rect::new(
            PANE_X + LABEL_X,
            PANE_Y + LABEL_Y,
            PANE_X + LABEL_X + LABEL_W,
            PANE_Y + LABEL_Y + LABEL_H,
        );
        let text_runs = ctx.extracted_host::<TextRunBatch>().to_vec();
        let label_run = text_runs.iter().find(|run| {
            run.clip.is_some_and(|c| {
                (c.min.x - want_clip.min.x).abs() < 1.0
                    && (c.min.y - want_clip.min.y).abs() < 1.0
                    && (c.max.x - want_clip.max.x).abs() < 1.0
                    && (c.max.y - want_clip.max.y).abs() < 1.0
            })
        });
        let pane_bg = color_arr(&tokens::PANE_BODY_BG);
        match label_run {
            Some(run) => {
                let glyphs = ctx.extracted_host_mut::<TextGlyphInstance>();
                let run_glyphs = &glyphs[run.instance_range.clone()];
                if let Some(first) = run_glyphs.first() {
                    let px_ = (first.pen_doc[0] as u32 + 3).min(WINDOW_W - 1);
                    let py_ = (first.pen_doc[1] as u32)
                        .saturating_sub(LABEL_FONT_PX as u32 / 3)
                        .min(WINDOW_H - 1);
                    let sample = pixel_at(px_, py_);
                    let coverage = (sample[0] - pane_bg[0])
                        .max(sample[1] - pane_bg[1])
                        .max(sample[2] - pane_bg[2]);
                    report(
                        &mut ok,
                        "label text coverage",
                        [0.05, 0.0, 0.0, 0.0],
                        [coverage, 0.0, 0.0, 0.0],
                        coverage > 0.05,
                    );
                } else {
                    report(&mut ok, "label text coverage", [0.0; 4], [0.0; 4], false);
                }
            }
            None => report(
                &mut ok,
                "label text coverage (run found)",
                [1.0; 4],
                [0.0; 4],
                false,
            ),
        }

        // (d) Pane corner outside its rounded radius reveals the root
        // window background underneath — proves rounded corners actually
        // discard, not just clip to the node rect.
        let bg_expect = color_arr(&tokens::WINDOW_BG);
        let corner_actual = pixel_at(PANE_X as u32 + 2, PANE_Y as u32 + 2);
        report(
            &mut ok,
            "pane rounded corner reveals window bg",
            bg_expect,
            corner_actual,
            close(bg_expect, corner_actual, 0.05),
        );

        // (e) Icon button: an opaque ink texel reads red-dominant (real
        // per-pixel alpha sampling x tint); a transparent-in-bbox texel
        // reveals the button's OWN background underneath (painted first,
        // same z-order as every other node: BACKGROUND < IMAGE).
        let icon_origin = (
            BTN_B_X as u32 + PANE_X as u32 + 8 + 2,
            BTN_B_Y as u32 + PANE_Y as u32,
        );
        let opaque = pixel_at(icon_origin.0 + 11, icon_origin.1 + 14);
        let opaque_pass =
            opaque[0] > 0.5 && opaque[0] > opaque[1] + 0.2 && opaque[0] > opaque[2] + 0.2;
        report(
            &mut ok,
            "icon opaque texel red-dominant",
            [1.0, 0.0, 0.0, 1.0],
            opaque,
            opaque_pass,
        );
        let btn_bg_expect = color_arr(&tokens::BUTTON_BG);
        let transparent = pixel_at(icon_origin.0 + 10, icon_origin.1 + 2);
        report(
            &mut ok,
            "icon transparent texel shows button bg",
            btn_bg_expect,
            transparent,
            close(btn_bg_expect, transparent, 0.06),
        );

        // Locate flex-layout widgets from extracted paint data.
        let quads = ctx.extracted_host_mut::<UiVertex>();
        let find_quad = |want: [f32; 4], max_size: f32| -> Option<(f32, f32, usize)> {
            quads.chunks(4).enumerate().find_map(|(qi, c)| {
                (c.len() == 4
                    && close(c[0].color, want, 0.02)
                    && c[0].size[0] <= max_size
                    && c[0].size[1] <= max_size)
                    .then(|| {
                        let cx = c.iter().map(|v| v.pos[0]).sum::<f32>() / 4.0;
                        let cy = c.iter().map(|v| v.pos[1]).sum::<f32>() / 4.0;
                        (cx, cy, qi)
                    })
            })
        };

        // Cursor probe: extraction and rasterization use the themed color.
        let cursor_expect = color_arr(&tokens::TEXT_INPUT_CURSOR);
        match find_quad(cursor_expect, 20.0) {
            Some((cx, cy, _)) => {
                let (px_, py_) = (
                    (cx.round() as i64).clamp(0, WINDOW_W as i64 - 1) as u32,
                    (cy.round() as i64).clamp(0, WINDOW_H as i64 - 1) as u32,
                );
                let actual = pixel_at(px_, py_);
                report(
                    &mut ok,
                    "text_input cursor pixel",
                    cursor_expect,
                    actual,
                    close(cursor_expect, actual, 0.06),
                );
            }
            None => report(
                &mut ok,
                "text_input cursor quad found",
                cursor_expect,
                [0.0; 4],
                false,
            ),
        }

        // (g) scroll area: locate the scroll viewport by its own
        // (deliberately unique-in-this-scene) background token, then prove
        // `ScrollPosition(0, 40)` moves content above the clip.
        // moved content — two complementary checks:
        //  (g1) CPU-list evidence: "Row 0"'s glyphs are still in the
        //       extracted stream (the walker never CPU-culls text against
        //       clip — only the GPU scissor does, see
        //       `push_editable_text_glyph_items`'s sibling `push_text_items`
        //       doc) but their pen position now sits above the viewport's
        //       clip rect — "a child that was visible is now clipped".
        //  (g2) GPU-pixel evidence: a row that scrolled INTO the visible
        //       range paints real glyph ink (not the viewport's own plain
        //       background) at its now-visible position — the pixel
        //       genuinely changed from "background" to "content".
        let area_linear = scroll_area_bg().to_linear();
        let area_expect = [
            area_linear.red,
            area_linear.green,
            area_linear.blue,
            area_linear.alpha,
        ];
        match find_quad(area_expect, 500.0) {
            Some((_, _, qi)) => {
                let chunk = &quads[qi * 4..qi * 4 + 4];
                let min_x = chunk.iter().map(|v| v.pos[0]).fold(f32::INFINITY, f32::min);
                let min_y = chunk.iter().map(|v| v.pos[1]).fold(f32::INFINITY, f32::min);
                let max_x = chunk
                    .iter()
                    .map(|v| v.pos[0])
                    .fold(f32::NEG_INFINITY, f32::max);
                let max_y = chunk
                    .iter()
                    .map(|v| v.pos[1])
                    .fold(f32::NEG_INFINITY, f32::max);
                let glyphs = ctx.extracted_host_mut::<TextGlyphInstance>();

                // (g1): some glyph's pen position is above the clip's top
                // edge — "Row 0" (and part of "Row 1") scrolled out.
                let scrolled_above = glyphs.iter().any(|g| {
                    g.pen_doc[1] < min_y && g.pen_doc[0] >= min_x && g.pen_doc[0] <= max_x
                });
                report(
                    &mut ok,
                    "scroll: a row's glyphs scrolled above the clip (was visible, now clipped)",
                    [1.0; 4],
                    [if scrolled_above { 1.0 } else { 0.0 }; 4],
                    scrolled_above,
                );

                // (g2): a glyph safely inside the clip (a few px margin, so
                // the probe pixel isn't itself scissor-boundary-sensitive)
                // paints real ink, not the viewport's plain background.
                let inside = glyphs.iter().find(|g| {
                    g.pen_doc[1] >= min_y + 6.0
                        && g.pen_doc[1] <= max_y - 6.0
                        && g.pen_doc[0] >= min_x + 2.0
                        && g.pen_doc[0] <= max_x - 2.0
                });
                match inside {
                    Some(g) => {
                        // Pen origin differs from ink; nudge into a stroke.
                        let px_ = (g.pen_doc[0] as i64 + 3).clamp(0, WINDOW_W as i64 - 1) as u32;
                        let py_ = (g.pen_doc[1] as i64 - 5).clamp(0, WINDOW_H as i64 - 1) as u32;
                        let actual = pixel_at(px_, py_);
                        let coverage = (actual[0] - area_expect[0]).abs()
                            + (actual[1] - area_expect[1]).abs()
                            + (actual[2] - area_expect[2]).abs();
                        report(
                            &mut ok,
                            "scroll: a row scrolled into view paints real ink",
                            [0.05, 0.0, 0.0, 0.0],
                            [coverage, 0.0, 0.0, 0.0],
                            coverage > 0.05,
                        );
                    }
                    None => report(
                        &mut ok,
                        "scroll: a glyph visible inside the clip",
                        [1.0; 4],
                        [0.0; 4],
                        false,
                    ),
                }
            }
            None => report(
                &mut ok,
                "scroll area quad found",
                area_expect,
                [0.0; 4],
                false,
            ),
        }

        // Popover probe checks stack order and visible background color.
        let quads = ctx.extracted_host_mut::<UiVertex>();
        let find_quad2 = |want: [f32; 4], max_size: f32| -> Option<(f32, f32, usize)> {
            quads.chunks(4).enumerate().find_map(|(qi, c)| {
                (c.len() == 4
                    && close(c[0].color, want, 0.02)
                    && c[0].size[0] <= max_size
                    && c[0].size[1] <= max_size)
                    .then(|| {
                        let cx = c.iter().map(|v| v.pos[0]).sum::<f32>() / 4.0;
                        let cy = c.iter().map(|v| v.pos[1]).sum::<f32>() / 4.0;
                        (cx, cy, qi)
                    })
            })
        };
        let pane2_expect = color_arr(&tokens::PANE_BODY_BG);
        let pane2_idx = quads
            .chunks(4)
            .position(|c| c.len() == 4 && close(c[0].color, pane2_expect, 0.02));
        let menu_expect = color_arr(&tokens::MENU_BG);
        // Reuse the popover center for shadow probes.
        let mut menu_center: Option<(f32, f32)> = None;
        let mut menu_center_actual: Option<[f32; 4]> = None;
        match (pane2_idx, find_quad2(menu_expect, 300.0)) {
            (Some(pane2_idx), Some((cx, cy, menu_idx))) => {
                report(
                    &mut ok,
                    "menu popover paints after pane2 (global stack index)",
                    [1.0; 4],
                    [if menu_idx > pane2_idx { 1.0 } else { 0.0 }; 4],
                    menu_idx > pane2_idx,
                );
                let (px_, py_) = (
                    (cx.round() as i64).clamp(0, WINDOW_W as i64 - 1) as u32,
                    (cy.round() as i64).clamp(0, WINDOW_H as i64 - 1) as u32,
                );
                let actual = pixel_at(px_, py_);
                report(
                    &mut ok,
                    "menu popover pixel over sibling content",
                    menu_expect,
                    actual,
                    close(menu_expect, actual, 0.06),
                );
                menu_center = Some((cx, cy));
                menu_center_actual = Some(actual);
            }
            _ => report(
                &mut ok,
                "menu popover + pane2 quads found",
                [1.0; 4],
                [0.0; 4],
                false,
            ),
        }

        // Shadow probes locate the popover's sole shadow in its paint stream.
        const MENU_SHADOW_COLOR_LINEAR: [f32; 4] = [0.0, 0.0, 0.0, 0.9];
        let shadow_quads = ctx.extracted_host_mut::<UiShadowVertex>();
        let shadow_quad = shadow_quads
            .chunks(4)
            .find(|c| c.len() == 4 && close(c[0].color, MENU_SHADOW_COLOR_LINEAR, 0.02));
        match (shadow_quad, menu_center, menu_center_actual) {
            (Some(c), Some(_menu_center), Some(menu_actual)) => {
                let (min_x, min_y, max_x, max_y) = (
                    c.iter().map(|v| v.pos[0]).fold(f32::INFINITY, f32::min),
                    c.iter().map(|v| v.pos[1]).fold(f32::INFINITY, f32::min),
                    c.iter().map(|v| v.pos[0]).fold(f32::NEG_INFINITY, f32::max),
                    c.iter().map(|v| v.pos[1]).fold(f32::NEG_INFINITY, f32::max),
                );
                let center = GVec2::new((min_x + max_x) * 0.5, (min_y + max_y) * 0.5);
                let half_size = GVec2::new(c[0].size[0], c[0].size[1]) * 0.5;
                let radius = GVec4::from_array(c[0].radius);
                let blur = c[0].blur;
                let shadow_color = GVec4::from_array(c[0].color);

                // (l) shadow paints strictly UNDER the caster: the SAME
                // popover-center pixel probe (h) already validated equals
                // `menu_expect` — restated here as an explicit shadow claim
                // (if the shadow pipeline drew AFTER the popover's own
                // background/border instead of before, this pixel would be
                // darkened by the shadow's own near-total interior coverage
                // and fail).
                report(
                    &mut ok,
                    "shadow paints under the caster (popover center unshadowed)",
                    menu_expect,
                    menu_actual,
                    close(menu_expect, menu_actual, 0.06),
                );

                // (m)/(n): a ray of probe points walking outward from the
                // shadow box's right edge, into the blur halo and past it.
                // `far` sits well beyond `3 * blur` (the integral's own
                // cutoff — see `ui_shadow_coverage`), so its composited
                // color IS the plain background color at that row — used
                // as both lanes' `bg` reference (no assumption about WHICH
                // theme token is behind the popover).
                let offsets = [2.0f32, 6.0, 14.0, 26.0];
                let far_offset = 60.0f32;
                let ray_pixel = |d: f32| -> (u32, u32) {
                    let p = center + GVec2::new(half_size.x + d, 0.0);
                    (
                        (p.x.round() as i64).clamp(0, WINDOW_W as i64 - 1) as u32,
                        (p.y.round() as i64).clamp(0, WINDOW_H as i64 - 1) as u32,
                    )
                };
                let (fx, fy) = ray_pixel(far_offset);
                let bg = pixel_at(fx, fy);
                let darkness = |actual: [f32; 4]| -> f32 {
                    (0..3).map(|k| (bg[k] - actual[k]).max(0.0)).sum::<f32>()
                };

                // (m) monotonic falloff: darkness (how much the background
                // is pulled toward the shadow's own black) strictly
                // decreases moving outward, and the nearest probe is
                // genuinely shadowed (nonzero darkness).
                let ray_actual: Vec<[f32; 4]> = offsets
                    .iter()
                    .map(|&d| {
                        let (x, y) = ray_pixel(d);
                        pixel_at(x, y)
                    })
                    .collect();
                let ray_darkness: Vec<f32> = ray_actual.iter().map(|&a| darkness(a)).collect();
                let monotonic = ray_darkness
                    .windows(2)
                    .all(|w| w[0] > w[1] + 1e-3 || w[0] < 1e-3);
                // The dark theme's own background near the popover is
                // already close to black (same ballpark as the shadow's
                // own black), so "darkness" (a linear-space RGB subtraction)
                // moves in small ABSOLUTE steps even where coverage is
                // large — this threshold is picked relative to the ray's
                // own measured values (see the printed
                // "shadow ray darkness" line), not an assumed magnitude.
                let nearest_shadowed = ray_darkness[0] > 0.003;
                report(
                    &mut ok,
                    "shadow alpha falls off monotonically outward from the caster edge",
                    [1.0; 4],
                    [if monotonic && nearest_shadowed {
                        1.0
                    } else {
                        0.0
                    }; 4],
                    monotonic && nearest_shadowed,
                );
                println!(
                    "PROBE shadow ray darkness (offsets {offsets:?} px past edge): {ray_darkness:?}"
                );

                // (n) a probe inside the blur halo matches the CPU oracle
                // (`ui_shadow_shade` — the exact function `ui_shadow_frag`
                // calls) composited straight-alpha over the SAME `bg`
                // reference the falloff probe above already established.
                let probe_d = offsets[1];
                let (px_, py_) = ray_pixel(probe_d);
                let point = GVec2::new(px_ as f32, py_ as f32) - center;
                let size = GVec2::new(c[0].size[0], c[0].size[1]);
                let oracle = ui_shadow_shade(shadow_color, point, size, radius, blur);
                let a = oracle.w;
                let expect_n = [
                    oracle.x * a + bg[0] * (1.0 - a),
                    oracle.y * a + bg[1] * (1.0 - a),
                    oracle.z * a + bg[2] * (1.0 - a),
                    1.0,
                ];
                let actual_n = pixel_at(px_, py_);
                report(
                    &mut ok,
                    "shadow probe in the blur halo matches the CPU oracle composite",
                    expect_n,
                    actual_n,
                    close(expect_n, actual_n, 0.06),
                );
            }
            _ => report(
                &mut ok,
                "menu shadow quad + popover center found",
                [1.0; 4],
                [0.0; 4],
                false,
            ),
        }

        // Compare color-tool modes with CPU shader-oracle results.
        fn quad_bounds(c: &[UiVertex]) -> (f32, f32, f32, f32) {
            let min_x = c.iter().map(|v| v.pos[0]).fold(f32::INFINITY, f32::min);
            let max_x = c.iter().map(|v| v.pos[0]).fold(f32::NEG_INFINITY, f32::max);
            let min_y = c.iter().map(|v| v.pos[1]).fold(f32::INFINITY, f32::min);
            let max_y = c.iter().map(|v| v.pos[1]).fold(f32::NEG_INFINITY, f32::max);
            (min_x, min_y, max_x, max_y)
        }

        let quads = ctx.extracted_host_mut::<UiVertex>();
        let alpha_pattern_quads: Vec<Vec<UiVertex>> = quads
            .chunks(4)
            .filter(|c| c.len() == 4 && (c[0].flags >> UI_MODE_SHIFT) == UI_MODE_ALPHA_PATTERN)
            .map(<[UiVertex]>::to_vec)
            .collect();
        let color_plane_quads: Vec<Vec<UiVertex>> = quads
            .chunks(4)
            .filter(|c| c.len() == 4 && (c[0].flags >> UI_MODE_SHIFT) == UI_MODE_COLOR_PLANE)
            .map(<[UiVertex]>::to_vec)
            .collect();

        // (i) checkerboard + (c) swatch translucency, combined: the
        // swatch's `ColorSwatchFg` (accent @ 0.5 alpha) covers its ENTIRE
        // rect, so there is no bare, unobstructed checkerboard pixel to
        // sample in isolation — every readback pixel is already the
        // composited (checker `over` translucent fg) result. That's
        // exactly probe (c)'s subject, and it doubles as probe (i)'s
        // checkerboard-math proof: predicting the composite from the SAME
        // `ui_alpha_pattern_shade` oracle blended with the known constant
        // fg alpha only comes out right if the checker math is right too.
        // Picked out from the two alpha-pattern quads (the swatch and the
        // slider's track) by aspect ratio — the track is a long thin bar,
        // the swatch is square.
        let swatch_quad = alpha_pattern_quads.iter().find(|c| {
            let (min_x, min_y, max_x, max_y) = quad_bounds(c);
            let (w, h) = (max_x - min_x, max_y - min_y);
            w > 1.0 && h > 1.0 && (w / h - 1.0).abs() < 0.3
        });
        match swatch_quad {
            Some(c) => {
                let (min_x, min_y, max_x, max_y) = quad_bounds(c);
                let center = GVec2::new((min_x + max_x) * 0.5, (min_y + max_y) * 0.5);
                let size = GVec2::new(max_x - min_x, max_y - min_y);
                let radius = GVec4::from_array(c[0].radius);
                let p0 = GVec2::new(min_x + 4.0, min_y + 4.0);
                let p1 = GVec2::new(min_x + 12.0, min_y + 4.0);
                let actual0 = pixel_at(p0.x.round() as u32, p0.y.round() as u32);
                let actual1 = pixel_at(p1.x.round() as u32, p1.y.round() as u32);

                // Two pixels 8px apart (the checker's half-period — see
                // `abi_ui`'s `alpha_pattern_checker_parity` unit test
                // for why 8px, not the naive-but-wrong 16px) straddling a
                // parity boundary: the composited pixels must differ.
                let parity_differs = (actual0[0] - actual1[0]).abs() > 0.08;
                report(
                    &mut ok,
                    "checkerboard: half-period-apart pixels differ",
                    [1.0; 4],
                    [if parity_differs { 1.0 } else { 0.0 }; 4],
                    parity_differs,
                );

                // `update_swatch_color` (spawn-time `Changed<ColorSwatchValue>`,
                // fires on `Added` too) overwrites `ColorSwatchFg`'s
                // bsn-authored background with `ColorSwatchValue.0` wholesale
                // — so the fg's real color+alpha IS `swatch_value_color()`,
                // not the widget's own default accent tint.
                let fg_linear = {
                    let l = swatch_value_color().to_linear();
                    GVec4::new(l.red, l.green, l.blue, l.alpha)
                };
                let fg_alpha = fg_linear.w;
                let blend = |point: GVec2| -> [f32; 4] {
                    let checker = ui_alpha_pattern_shade(point, size, radius);
                    let rgb =
                        fg_linear.truncate() * fg_alpha + checker.truncate() * (1.0 - fg_alpha);
                    [rgb.x, rgb.y, rgb.z, 1.0]
                };
                let expect0 = blend(p0 - center);
                let expect1 = blend(p1 - center);
                report(
                    &mut ok,
                    "swatch: checkerboard shows through translucent fg (px0)",
                    expect0,
                    actual0,
                    close(expect0, actual0, 0.06),
                );
                report(
                    &mut ok,
                    "swatch: checkerboard shows through translucent fg (px1)",
                    expect1,
                    actual1,
                    close(expect1, actual1, 0.06),
                );
            }
            None => report(
                &mut ok,
                "checkerboard/swatch quad found",
                [1.0; 4],
                [0.0; 4],
                false,
            ),
        }

        // (j)/(k) color-plane corners: two interior points well clear of
        // the thumb (which, at the default `ColorPlaneValue` (0,0,0), sits
        // right on the uv=(0,0) corner) — expected color computed by the
        // exact same `ui_color_plane_shade` the shader dispatches to,
        // fed the vertex's own packed `UiMaterialData` (not assumed).
        match color_plane_quads.first() {
            Some(c) => {
                let (min_x, min_y, max_x, max_y) = quad_bounds(c);
                let size = GVec2::new(max_x - min_x, max_y - min_y);
                let material = UiMaterialData::from_color2(GVec4::from_array(c[0].color2));

                let near_tl = GVec2::new(min_x + 24.0, min_y + 24.0);
                let uv_tl = GVec2::new(24.0 / size.x, 24.0 / size.y);
                let expect_tl_v =
                    ui_color_plane_shade(uv_tl, material.variant, material.fixed_channel);
                let expect_tl = [expect_tl_v.x, expect_tl_v.y, expect_tl_v.z, expect_tl_v.w];
                let actual_tl = pixel_at(near_tl.x.round() as u32, near_tl.y.round() as u32);
                report(
                    &mut ok,
                    "color plane: near-TL corner matches CPU oracle",
                    expect_tl,
                    actual_tl,
                    close(expect_tl, actual_tl, 0.05),
                );

                let near_br = GVec2::new(max_x - 24.0, max_y - 24.0);
                let uv_br = GVec2::new(1.0 - 24.0 / size.x, 1.0 - 24.0 / size.y);
                let expect_br_v =
                    ui_color_plane_shade(uv_br, material.variant, material.fixed_channel);
                let expect_br = [expect_br_v.x, expect_br_v.y, expect_br_v.z, expect_br_v.w];
                let actual_br = pixel_at(near_br.x.round() as u32, near_br.y.round() as u32);
                report(
                    &mut ok,
                    "color plane: near-BR corner matches CPU oracle",
                    expect_br,
                    actual_br,
                    close(expect_br, actual_br, 0.05),
                );
            }
            None => report(&mut ok, "color plane quad found", [1.0; 4], [0.0; 4], false),
        }

        println!(
            "feathers_gallery ASHA_VERIFY: {}",
            if ok { "ALL PROBES OK" } else { "FAILED" }
        );
        assert!(
            ok,
            "one or more ASHA_VERIFY probes failed — see PROBE lines above"
        );
        ctx.request_exit();
    }
}

impl RenderScene for GalleryScene {
    fn draw(&mut self, ctx: &mut FrameCtx) {
        let gpu = ctx.gpu;
        // The swapchain's PHYSICAL extent (logical `WindowResolution` times
        // the OS scale factor) — NOT the compile-time `[WINDOW_W, WINDOW_H]`
        // logical constants, which mis-scale windowed rendering on any
        // scale factor != 1. `ingest`/`ingest_text` write ONE view
        // transform per frame shared by both the windowed backbuffer draw
        // below and (when `ASHA_VERIFY` is set) the offscreen verify draw
        // Both reads use the same extent.
        // `main()` forces the OS scale factor to 1.0 under `ASHA_VERIFY`
        // specifically so `extent == [WINDOW_W, WINDOW_H]` in that case,
        // keeping the verify path byte-stable.
        let extent = ctx.extent;

        // Keep verification state exposed for one frame.
        // verify state.
        {
            let heap = self.heap.as_mut().expect("heap present");
            self.bridge.ingest_icons(gpu, heap, ctx);
        }
        self.bridge.ingest(ctx, extent);
        self.bridge.ingest_text(ctx, extent);

        let ui_orders: Vec<u32> = ctx
            .extracted_host::<UiBatch>()
            .iter()
            .map(|b| b.order)
            .collect();
        let shadow_orders: Vec<u32> = ctx
            .extracted_host::<ui_bridge::UiShadowBatch>()
            .iter()
            .map(|b| b.order)
            .collect();
        let text_orders: Vec<u32> = ctx
            .extracted_host::<TextRunBatch>()
            .iter()
            .map(|b| b.order)
            .collect();
        debug_assert_eq!(ui_orders.len(), self.bridge.batches().len());
        debug_assert_eq!(shadow_orders.len(), self.bridge.shadow_batches().len());
        debug_assert_eq!(text_orders.len(), self.bridge.text_batches().len());

        // Windowed backbuffer draw: always runs, so a human sees the real
        // scene with `ASHA_VERIFY` unset.
        {
            let cb = ctx.cb;
            self.heap.as_ref().unwrap().bind(gpu, cb);
            draw_merged(
                gpu,
                cb,
                self.ui_pass.as_ref().unwrap(),
                self.text_pass.as_ref().unwrap(),
                self.bridge.batches(),
                &ui_orders,
                self.bridge.shadow_batches(),
                &shadow_orders,
                self.bridge.text_batches(),
                &text_orders,
                ctx.backbuffer,
                extent,
                [0.0, 0.0, 0.0, 0.0],
            );
        }

        if self.verified {
            return;
        }
        let Some(verify) = self.verify.as_ref() else {
            return;
        };
        let icon_ready = self.bridge.icon_ready(1);
        if ctx.frame < VERIFY_MIN_FRAME || !icon_ready {
            assert!(
                ctx.frame < GIVE_UP_FRAME,
                "ASHA_VERIFY: gallery never settled by frame {} (icon_ready={icon_ready}) — \
                 either layout/theme never stabilized or the chevron-down icon never finished \
                 loading/uploading",
                ctx.frame
            );
            return;
        }

        let texture = verify.target.texture;
        let readback = verify.readback;
        let cb = gpu.commands_begin(Queue::Main);
        self.heap.as_ref().unwrap().bind(gpu, cb);
        draw_merged(
            gpu,
            cb,
            self.ui_pass.as_ref().unwrap(),
            self.text_pass.as_ref().unwrap(),
            self.bridge.batches(),
            &ui_orders,
            self.bridge.shadow_batches(),
            &shadow_orders,
            self.bridge.text_batches(),
            &text_orders,
            texture,
            [WINDOW_W, WINDOW_H],
            [0.0, 0.0, 0.0, 0.0],
        );
        gpu.cmd_barrier(
            cb,
            Stage::RasterColorOut,
            Stage::Transfer,
            HazardFlags::empty(),
        );
        gpu.cmd_copy_texture_to_buffer(cb, texture, readback.cast());
        gpu.queue_submit(Queue::Main, &[cb]);
        gpu.queue_wait_idle(Queue::Main);

        self.verified = true;
        self.check_probes(ctx, readback);
    }

    fn teardown(&mut self, gpu: &Gpu) {
        if let Some(pass) = self.ui_pass.take() {
            pass.free(gpu);
        }
        if let Some(pass) = self.text_pass.take() {
            pass.free(gpu);
        }
        // Icon textures (owned by the bridge) must free before the heap
        // they registered into — same ordering `icon_extract.rs` documents.
        std::mem::take(&mut self.bridge).free(gpu);
        if let Some(heap) = self.heap.take() {
            heap.free(gpu);
        }
        if let Some(verify) = self.verify.take() {
            gpu.texture_free_and_destroy(verify.target);
            gpu.free(verify.readback);
        }
    }
}

fn main() {
    let verify = std::env::var_os("ASHA_VERIFY").is_some();

    // `ASHA_VERIFY`'s offscreen probes assume `ctx.extent == [WINDOW_W,
    // WINDOW_H]` (the verify target's own fixed allocation size — see
    // `GalleryScene::new`); `ctx.extent` is the swapchain's PHYSICAL
    // extent, logical size times the OS scale factor, so that assumption
    // only holds if the scale factor is forced to 1.0. Normal windowed runs
    // leave the OS scale factor alone — `GalleryScene::draw` derives its
    // view transform from `ctx.extent` precisely so it renders correctly at
    // any scale.
    let resolution = if verify {
        WindowResolution::new(WINDOW_W, WINDOW_H).with_scale_factor_override(1.0)
    } else {
        WindowResolution::new(WINDOW_W, WINDOW_H)
    };

    App::new()
        // Dark theme, before FeathersCorePlugin's `init_resource` (which is
        // a no-op once the resource already exists) — see the module doc.
        .insert_resource(UiTheme(dark_theme::create_dark_theme()))
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "asha × widgets — the feathers gallery".into(),
                resolution,
                visible: !verify,
                ..Default::default()
            }),
            ..Default::default()
        }))
        .add_plugins(PacingPlugin::default())
        .add_plugins(UiBridgePlugin)
        .add_plugins(widgets::FeathersPlugins)
        // The interaction observers (Pressed/Checked/RadioGroup selection/
        // slider drag) — NOT part of `widgets::FeathersPlugins` (which only
        // installs the THEMED styling layer on top). Same crate the
        // headless widget smoke tests (`render/tests/widgets.rs`) exercise
        // directly.
        .add_plugins(UiWidgetsPlugins)
        .add_plugins(
            AshaRenderPlugin::new(GalleryScene::new)
                .extract_ui()
                .extract_text()
                .extract_icons(),
        )
        .add_systems(Startup, spawn_scene)
        .add_systems(Update, prime_dynamic_widgets)
        .add_systems(bevy::app::First, input_debug_log)
        .add_systems(bevy::app::Last, input_debug_focus)
        .add_systems(Update, input_selftest)
        .add_observer(
            |click: On<bevy::picking::events::Pointer<bevy::picking::events::Click>>| {
                if std::env::var_os("ASHA_INPUT_DEBUG").is_some()
                    || std::env::var_os("ASHA_INPUT_SELFTEST").is_some()
                {
                    println!("[input] Pointer<Click> on {}", click.entity);
                }
            },
        )
        .add_observer(|act: On<bevy_ui_widgets::Activate>| {
            if std::env::var_os("ASHA_INPUT_DEBUG").is_some()
                || std::env::var_os("ASHA_INPUT_SELFTEST").is_some()
            {
                println!("[input] Activate on {}", act.entity);
            }
        })
        .add_observer(|ev: On<bevy_ui_widgets::MenuEvent>| {
            if std::env::var_os("ASHA_INPUT_DEBUG").is_some()
                || std::env::var_os("ASHA_INPUT_SELFTEST").is_some()
            {
                println!("[input] MenuEvent {:?} source {}", ev.action, ev.source);
            }
        })
        .add_observer(|vc: On<bevy_ui_widgets::ValueChange<bool>>| {
            if std::env::var_os("ASHA_INPUT_DEBUG").is_some()
                || std::env::var_os("ASHA_INPUT_SELFTEST").is_some()
            {
                println!("[input] ValueChange<bool> {} on {}", vc.value, vc.source);
            }
        })
        .run();
}

/// Logs window, pointer, and button input transitions when enabled.
fn input_debug_log(
    mut window_events: bevy::ecs::message::MessageReader<bevy::window::WindowEvent>,
    mut pointer_inputs: bevy::ecs::message::MessageReader<bevy::picking::pointer::PointerInput>,
    pressed_added: Query<Entity, Added<bevy::ui::Pressed>>,
    mut pressed_removed: RemovedComponents<bevy::ui::Pressed>,
) {
    if std::env::var_os("ASHA_INPUT_DEBUG").is_none() {
        return;
    }
    for ev in window_events.read() {
        match ev {
            bevy::window::WindowEvent::MouseButtonInput(e) => {
                println!(
                    "[input] WindowEvent::MouseButtonInput {:?} {:?}",
                    e.button, e.state
                );
            }
            bevy::window::WindowEvent::CursorMoved(_) => {}
            other => {
                let name = format!("{other:?}");
                println!(
                    "[input] WindowEvent::{}",
                    name.split(['(', ' ']).next().unwrap_or("?")
                );
            }
        }
    }
    for pi in pointer_inputs.read() {
        use bevy::picking::pointer::PointerAction;
        match pi.action {
            PointerAction::Press(b) => println!(
                "[input] PointerInput::Press({b:?}) at {:?}",
                pi.location.position
            ),
            PointerAction::Release(b) => println!("[input] PointerInput::Release({b:?})"),
            _ => {}
        }
    }
    for e in pressed_added.iter() {
        println!("[input] Pressed ADDED on {e}");
    }
    for e in pressed_removed.read() {
        println!("[input] Pressed REMOVED from {e}");
    }
}

/// Logs focus changes for input debugging.
fn input_debug_focus(focus: Res<InputFocus>, frames: Res<bevy::diagnostic::FrameCount>) {
    if std::env::var_os("ASHA_INPUT_DEBUG").is_none()
        && std::env::var_os("ASHA_INPUT_SELFTEST").is_none()
    {
        return;
    }
    if focus.is_changed() {
        println!("[focus] frame {} -> {:?}", frames.0, focus.get());
    }
}

/// Injects a synthetic aggregate-window click when enabled.
fn input_selftest(
    mut frame: Local<u32>,
    mut writer: bevy::ecs::message::MessageWriter<bevy::window::WindowEvent>,
    windows: Query<(Entity, &Window), With<bevy::window::PrimaryWindow>>,
    buttons: Query<
        (&bevy::ui::ComputedNode, &bevy::ui::UiGlobalTransform),
        With<bevy_ui_widgets::Button>,
    >,
    checkboxes: Query<
        (
            Entity,
            &bevy::ui::ComputedNode,
            &bevy::ui::UiGlobalTransform,
            Has<Checked>,
        ),
        With<bevy_ui_widgets::Checkbox>,
    >,
    menu_buttons: Query<&bevy::ui::UiGlobalTransform, With<bevy_ui_widgets::MenuButton>>,
    menu_items: Query<&bevy::ui::UiGlobalTransform, With<bevy_ui_widgets::MenuItem>>,
    popup: Query<&bevy::camera::visibility::Visibility, With<GalleryMenuPopup>>,
    planes: Query<&bevy::ui::UiGlobalTransform, With<FeathersColorPlane>>,
    swatches: Query<&ColorSwatchValue>,
    pressed: Query<Entity, With<bevy::ui::Pressed>>,
    mut was_checked: Local<Option<(Entity, bool)>>,
    mut exit: bevy::ecs::message::MessageWriter<bevy::app::AppExit>,
) {
    if std::env::var_os("ASHA_INPUT_SELFTEST").is_none() {
        return;
    }
    *frame += 1;
    let Ok((window_entity, window)) = windows.single() else {
        return;
    };
    match *frame {
        30 => {
            let Some((node, xform)) = buttons.iter().next() else {
                println!("INPUT SELFTEST: FAIL (no Button entity found)");
                exit.write(bevy::app::AppExit::error());
                return;
            };
            // UiGlobalTransform holds PHYSICAL px; WindowEvent positions
            // are LOGICAL (bevy_winit divides by the scale factor).
            let center_physical = xform.translation;
            let scale = window.scale_factor();
            let center_logical = Vec2::new(center_physical.x, center_physical.y) / scale;
            println!(
                "INPUT SELFTEST: injecting click at logical {center_logical:?} \
                 (physical {center_physical:?}, node size {:?}, scale {scale})",
                node.size()
            );
            writer.write(bevy::window::WindowEvent::CursorMoved(
                bevy::window::CursorMoved {
                    window: window_entity,
                    position: center_logical,
                    delta: Some(Vec2::ZERO),
                },
            ));
        }
        32 => {
            writer.write(bevy::window::WindowEvent::MouseButtonInput(
                bevy::input::mouse::MouseButtonInput {
                    button: bevy::input::mouse::MouseButton::Left,
                    state: bevy::input::ButtonState::Pressed,
                    window: window_entity,
                },
            ));
        }
        35 => {
            let ok = !pressed.is_empty();
            println!(
                "INPUT SELFTEST: press {} (Pressed present on {} entit{})",
                if ok { "PASS" } else { "FAIL" },
                pressed.iter().count(),
                if pressed.iter().count() == 1 {
                    "y"
                } else {
                    "ies"
                },
            );
            // Release the button.
            writer.write(bevy::window::WindowEvent::MouseButtonInput(
                bevy::input::mouse::MouseButtonInput {
                    button: bevy::input::mouse::MouseButton::Left,
                    state: bevy::input::ButtonState::Released,
                    window: window_entity,
                },
            ));
        }
        // Phase 2: controlled state must update on click.
        40 => {
            let Some((entity, node, xform, checked)) = checkboxes.iter().next() else {
                println!("INPUT SELFTEST: FAIL (no Checkbox-widget entity found)");
                exit.write(bevy::app::AppExit::error());
                return;
            };
            *was_checked = Some((entity, checked));
            let scale = window.scale_factor();
            let center = Vec2::new(xform.translation.x, xform.translation.y) / scale;
            println!(
                "INPUT SELFTEST: clicking checkbox-widget {entity} at logical {center:?} \
                 (size {:?}, checked={checked})",
                node.size()
            );
            writer.write(bevy::window::WindowEvent::CursorMoved(
                bevy::window::CursorMoved {
                    window: window_entity,
                    position: center,
                    delta: Some(Vec2::ZERO),
                },
            ));
        }
        42 | 44 => {
            writer.write(bevy::window::WindowEvent::MouseButtonInput(
                bevy::input::mouse::MouseButtonInput {
                    button: bevy::input::mouse::MouseButton::Left,
                    state: if *frame == 42 {
                        bevy::input::ButtonState::Pressed
                    } else {
                        bevy::input::ButtonState::Released
                    },
                    window: window_entity,
                },
            ));
        }
        48 => {
            let (entity, before) = was_checked.expect("phase 2 recorded the widget");
            let after = checkboxes
                .get(entity)
                .map(|(_, _, _, c)| c)
                .unwrap_or(before);
            let ok = after != before;
            println!(
                "INPUT SELFTEST: commit {} (Checked {before} -> {after} on {entity})",
                if ok { "PASS" } else { "FAIL" },
            );
            if !ok {
                exit.write(bevy::app::AppExit::error());
            }
        }
        // Phase 3: the dropdown — button click opens the popup, item click
        // activates and closes it. The popup must start Hidden in live
        // runs (the force-open is verify-only now).
        50 | 60 => {
            let xform = if *frame == 50 {
                let Some(x) = menu_buttons.iter().next() else {
                    println!("INPUT SELFTEST: FAIL (no MenuButton found)");
                    exit.write(bevy::app::AppExit::error());
                    return;
                };
                x
            } else {
                let Some(x) = menu_items.iter().next() else {
                    println!("INPUT SELFTEST: FAIL (no MenuItem found)");
                    exit.write(bevy::app::AppExit::error());
                    return;
                };
                x
            };
            let center =
                Vec2::new(xform.translation.x, xform.translation.y) / window.scale_factor();
            println!(
                "INPUT SELFTEST: moving to {} at logical {center:?}",
                if *frame == 50 {
                    "menu button"
                } else {
                    "menu item 1"
                }
            );
            writer.write(bevy::window::WindowEvent::CursorMoved(
                bevy::window::CursorMoved {
                    window: window_entity,
                    position: center,
                    delta: Some(Vec2::ZERO),
                },
            ));
        }
        52 | 54 | 62 | 64 => {
            writer.write(bevy::window::WindowEvent::MouseButtonInput(
                bevy::input::mouse::MouseButtonInput {
                    button: bevy::input::mouse::MouseButton::Left,
                    state: if matches!(*frame, 52 | 62) {
                        bevy::input::ButtonState::Pressed
                    } else {
                        bevy::input::ButtonState::Released
                    },
                    window: window_entity,
                },
            ));
        }
        58 => {
            let visible = popup
                .single()
                .map(|v| *v == bevy::camera::visibility::Visibility::Visible)
                .unwrap_or(false);
            println!(
                "INPUT SELFTEST: dropdown open {}",
                if visible {
                    "PASS"
                } else {
                    "FAIL (popup not Visible after button click)"
                }
            );
            if !visible {
                exit.write(bevy::app::AppExit::error());
            }
        }
        68 => {
            let hidden = popup
                .single()
                .map(|v| *v != bevy::camera::visibility::Visibility::Visible)
                .unwrap_or(false);
            println!(
                "INPUT SELFTEST: dropdown item-click closes {}",
                if hidden {
                    "PASS"
                } else {
                    "FAIL (popup still Visible after item click)"
                }
            );
            if !hidden {
                exit.write(bevy::app::AppExit::error());
            }
        }
        // Phase 4: the color picker — clicking the plane must move the
        // swatch preview off its authored color.
        70 => {
            let Some(xform) = planes.iter().next() else {
                println!("INPUT SELFTEST: FAIL (no FeathersColorPlane found)");
                exit.write(bevy::app::AppExit::error());
                return;
            };
            let center =
                Vec2::new(xform.translation.x, xform.translation.y) / window.scale_factor();
            println!("INPUT SELFTEST: clicking color plane at logical {center:?}");
            writer.write(bevy::window::WindowEvent::CursorMoved(
                bevy::window::CursorMoved {
                    window: window_entity,
                    position: center,
                    delta: Some(Vec2::ZERO),
                },
            ));
        }
        72 | 74 => {
            writer.write(bevy::window::WindowEvent::MouseButtonInput(
                bevy::input::mouse::MouseButtonInput {
                    button: bevy::input::mouse::MouseButton::Left,
                    state: if *frame == 72 {
                        bevy::input::ButtonState::Pressed
                    } else {
                        bevy::input::ButtonState::Released
                    },
                    window: window_entity,
                },
            ));
        }
        78 => {
            let authored = swatch_value_color();
            let changed = swatches
                .iter()
                .next()
                .map(|s| s.0 != authored)
                .unwrap_or(false);
            println!(
                "INPUT SELFTEST: color preview updates {} ({:?})",
                if changed {
                    "PASS"
                } else {
                    "FAIL (swatch still at authored color)"
                },
                swatches.iter().next().map(|s| s.0),
            );
            exit.write(if changed {
                bevy::app::AppExit::Success
            } else {
                bevy::app::AppExit::error()
            });
        }
        _ => {}
    }
}
