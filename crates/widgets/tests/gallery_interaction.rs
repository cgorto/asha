//! Headless interaction test for real Feathers button theming and paint.
//!
//! Injects a pointer event, runs real layout and observers, then inspects paint.

use bevy::camera::{Camera, ComputedCameraValues, NormalizedRenderTarget, RenderTargetInfo};
use bevy::math::UVec2;
use bevy::picking::backend::HitData;
use bevy::picking::events::{Pointer, Press};
use bevy::picking::pointer::{Location, PointerButton, PointerId};
use bevy::prelude::*;
use bevy::ui::Pressed;

use ui_bridge::{UiBridgePlugin, UiPaintList};
#[allow(deprecated)]
use widgets::controls::{ButtonBundleProps, button_bundle};
use widgets::dark_theme::create_dark_theme;
use widgets::theme::UiTheme;
use widgets::tokens;

fn pointer_location() -> Location {
    Location {
        target: NormalizedRenderTarget::None {
            width: 0,
            height: 0,
        },
        position: Vec2::ZERO,
    }
}

fn hit() -> HitData {
    HitData::new(Entity::PLACEHOLDER, 0.0, None, None)
}

/// The background quad(s) painted for `entity`: `paint_ui_system` emits
/// exactly one `PaintItem::Quad` (background, `stack_z_offsets::BACKGROUND`)
/// per visible, non-transparent `BackgroundColor` node — a plain button with
/// no children has exactly one, so its first vertex's `color` (flat across
/// all four corners, per `abi_ui`'s vertex-pulling contract) is the
/// walker's own answer for "what background did this entity paint with".
fn first_quad_color(list: &UiPaintList) -> [f32; 4] {
    assert!(
        list.quad_count > 0,
        "expected at least one quad in UiPaintList — did layout/paint run?"
    );
    list.vertices[0].color
}

fn close(a: [f32; 4], b: [f32; 4]) -> bool {
    (0..4).all(|k| (a[k] - b[k]).abs() < 1e-3)
}

#[test]
#[allow(deprecated)]
fn button_press_recolors_the_walkers_background_quad() {
    let mut app = App::new();
    app.insert_resource(UiTheme(create_dark_theme()));
    // DefaultPlugins minus `WinitPlugin` supplies bevy_ui's layout, text,
    // image, focus, and picking systems without creating a native window.
    // The direct pointer path avoids fragile hit-test setup.
    app.add_plugins(DefaultPlugins.build().disable::<bevy::winit::WinitPlugin>());
    app.add_plugins((
        bevy_ui_widgets::ButtonPlugin,
        // widgets::FeathersCorePlugin brings in ControlsPlugin (widgets' OWN
        // `controls::ButtonPlugin` style system + the theme observers) —
        // exactly what `examples/feathers_gallery.rs` runs.
        widgets::FeathersCorePlugin,
        UiBridgePlugin,
    ));

    // No real Window/RenderTarget exists headless — hand the UI root a
    // camera with an explicit computed viewport, the same recipe
    // `bevy_ui`'s own `propagate_ui_target_cameras` unit tests use.
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

    let button = app
        .world_mut()
        .spawn(button_bundle(ButtonBundleProps::default(), (), ()))
        .id();

    // Let layout, theming, and the paint walker settle on the UNPRESSED
    // state first.
    app.update();
    app.update();

    let before = {
        let list = app.world().resource::<UiPaintList>();
        first_quad_color(list)
    };
    let expect_unpressed = UiTheme(create_dark_theme())
        .color(&tokens::BUTTON_BG)
        .to_linear();
    assert!(
        close(
            before,
            [
                expect_unpressed.red,
                expect_unpressed.green,
                expect_unpressed.blue,
                expect_unpressed.alpha
            ]
        ),
        "unpressed button quad should start at BUTTON_BG: got {before:?}"
    );

    // Inject a targeted pointer press — same idiom
    // `render/tests/widgets.rs::button_activates_after_a_targeted_pointer_click`
    // uses: `bevy_ui_widgets::button_on_pointer_down` inserts `Pressed` on
    // the button entity in response, with no picking backend or window
    // involved at all.
    app.world_mut().trigger(Pointer::new_without_propagate(
        PointerId::Mouse,
        pointer_location(),
        Press {
            button: PointerButton::Primary,
            hit: hit(),
            count: 1,
        },
        button,
    ));

    // One update: widgets' `update_button_styles` reacts to `Added<Pressed>`
    // and inserts `ThemeBackgroundColor(BUTTON_BG_PRESSED)`; its own
    // `on_changed_background` observer (fired by that same `Insert`) updates
    // `BackgroundColor` in the same tick. Another update: `UiBridgePlugin`'s
    // `paint_ui_system` (`PostUpdate`, after layout/stack) re-walks the tree
    // and re-emits the quad with the new color.
    app.update();
    app.update();

    let after = {
        let list = app.world().resource::<UiPaintList>();
        first_quad_color(list)
    };
    let expect_pressed = UiTheme(create_dark_theme())
        .color(&tokens::BUTTON_BG_PRESSED)
        .to_linear();
    let expect_pressed = [
        expect_pressed.red,
        expect_pressed.green,
        expect_pressed.blue,
        expect_pressed.alpha,
    ];

    println!(
        "gallery_interaction: unpressed={before:?} pressed={after:?} expected_pressed={expect_pressed:?}"
    );
    assert!(
        app.world().get::<Pressed>(button).is_some(),
        "bevy_ui_widgets::ButtonPlugin should have inserted Pressed on the targeted press"
    );
    assert!(
        !close(before, after),
        "the walker's quad color should have changed after the press: still {after:?}"
    );
    assert!(
        close(after, expect_pressed),
        "pressed button quad should read BUTTON_BG_PRESSED: got {after:?}, expected {expect_pressed:?}"
    );
}
