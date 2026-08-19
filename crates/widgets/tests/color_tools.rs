//! Headless color-tool test for material tags and paint extraction.
//!
//! Verifies mode bits and packed data without a window or GPU.

use bevy::camera::{Camera, ComputedCameraValues, RenderTargetInfo};
use bevy::math::UVec2;
use bevy::prelude::*;
use bevy::ui::{PositionType, Val};

use abi_ui::{UI_MODE_ALPHA_PATTERN, UI_MODE_COLOR_PLANE, UI_MODE_SHIFT, UI_PLANE_HL};
use ui_bridge::{UiBridgePlugin, UiPaintList};
#[allow(deprecated)]
use widgets::controls::{
    ColorChannel, FeathersColorPlane, FeathersColorSliderProps, color_plane_bundle,
    color_slider_bundle, color_swatch_bundle,
};
use widgets::dark_theme::create_dark_theme;
use widgets::theme::UiTheme;

fn headless_app() -> App {
    let mut app = App::new();
    app.insert_resource(UiTheme(create_dark_theme()));
    // Same headless recipe `gallery_interaction.rs` documents: every
    // subsystem real `bevy_ui` layout needs, minus the winit window.
    app.add_plugins(DefaultPlugins.build().disable::<bevy::winit::WinitPlugin>());
    app.add_plugins((widgets::FeathersCorePlugin, UiBridgePlugin));

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
    app
}

/// Sized host required for cross-axis stretching and paint extraction.
fn abs_host(w: f32, h: f32) -> Node {
    Node {
        position_type: PositionType::Absolute,
        left: Val::Px(0.0),
        top: Val::Px(0.0),
        width: Val::Px(w),
        height: Val::Px(h),
        ..Default::default()
    }
}

fn mode_and_color2(list: &UiPaintList, mode: u32) -> Vec<[f32; 4]> {
    list.vertices
        .chunks(4)
        .filter(|c| c[0].flags >> UI_MODE_SHIFT == mode)
        .map(|c| c[0].color2)
        .collect()
}

/// The plane emits one quad with packed axes and fixed channel.
#[test]
#[allow(deprecated)]
fn color_plane_widget_emits_color_plane_mode_quad() {
    let mut app = headless_app();
    app.world_mut()
        .spawn(abs_host(120.0, 120.0))
        .with_children(|p| {
            p.spawn(color_plane_bundle(FeathersColorPlane::HueLightness, ()));
        });

    app.update();
    app.update();

    let list = app.world().resource::<UiPaintList>();
    let planes = mode_and_color2(list, UI_MODE_COLOR_PLANE);
    assert_eq!(planes.len(), 1, "exactly one color-plane quad");
    let [fixed_channel, variant, ..] = planes[0];
    assert!(
        (fixed_channel - 0.0).abs() < 1e-3,
        "default ColorPlaneValue.z is 0"
    );
    assert!(
        (variant - UI_PLANE_HL as f32).abs() < 1e-3,
        "HueLightness -> UI_PLANE_HL"
    );
}

/// The slider track emits its alpha-pattern checkerboard quad.
#[test]
#[allow(deprecated)]
fn color_slider_track_emits_alpha_pattern_mode_quad() {
    let mut app = headless_app();
    app.world_mut()
        .spawn(abs_host(200.0, 24.0))
        .with_children(|p| {
            p.spawn(color_slider_bundle(
                FeathersColorSliderProps {
                    value: 0.5,
                    channel: ColorChannel::Alpha,
                },
                (),
            ));
        });

    app.update();
    app.update();

    let list = app.world().resource::<UiPaintList>();
    let patterns = mode_and_color2(list, UI_MODE_ALPHA_PATTERN);
    assert_eq!(
        patterns.len(),
        1,
        "exactly one alpha-pattern quad (the track)"
    );
    assert_eq!(
        patterns[0], [0.0; 4],
        "alpha-pattern MODE ignores variant/fixed_channel"
    );
}

/// The swatch emits an alpha-pattern background quad.
#[test]
#[allow(deprecated)]
fn color_swatch_emits_alpha_pattern_mode_quad() {
    let mut app = headless_app();
    app.world_mut()
        .spawn(abs_host(24.0, 24.0))
        .with_children(|p| {
            p.spawn(color_swatch_bundle(()));
        });

    app.update();
    app.update();

    let list = app.world().resource::<UiPaintList>();
    let patterns = mode_and_color2(list, UI_MODE_ALPHA_PATTERN);
    assert_eq!(
        patterns.len(),
        1,
        "exactly one alpha-pattern quad (the swatch itself)"
    );
}
