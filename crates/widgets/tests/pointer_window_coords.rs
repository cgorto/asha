//! Regression tests for window-coordinate picking.
//!
//! Unlike direct pointer triggers, these inject `PointerInput` messages and
//! run the complete picking pipeline. The 2x case uses a nonzero viewport
//! offset so stale camera scale data cannot remain self-consistent.

use bevy::app::App;
use bevy::camera::{Camera, Camera2d, NormalizedRenderTarget, RenderTarget, Viewport};
use bevy::ecs::component::Component;
use bevy::math::{UVec2, Vec2};
use bevy::picking::hover::Hovered;
use bevy::picking::pointer::{Location, PointerAction, PointerButton, PointerId, PointerInput};
use bevy::prelude::*;
use bevy::ui::{Node, PositionType, Pressed, Val};
use bevy::window::{PrimaryWindow, Window, WindowPlugin, WindowRef, WindowResolution};

use bevy_ui_widgets::{Button, ButtonPlugin};

use ui_bridge::UiBridgePlugin;

/// Logical (`Val::Px`) window size — identical in both the 1x and 2x cases;
/// only the window's PHYSICAL resolution and scale factor differ between
/// them (see the module doc).
const LOGICAL_W: f32 = 400.0;
const LOGICAL_H: f32 = 300.0;

/// The target button's logical rect.
const BTN_X: f32 = 50.0;
const BTN_Y: f32 = 50.0;
const BTN_W: f32 = 100.0;
const BTN_H: f32 = 40.0;

/// A second, non-overlapping button — proves the pointer hits ONLY the
/// intended target, not "whatever's first" or "everything at once".
const DECOY_X: f32 = 250.0;
const DECOY_Y: f32 = 150.0;
const DECOY_W: f32 = 100.0;
const DECOY_H: f32 = 40.0;

/// Marker distinguishing the target button from the decoy in assertions.
#[derive(Component)]
struct TargetButton;

#[derive(Component)]
struct DecoyButton;

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

/// Builds the app, camera (with an optional explicit `viewport` — see the
/// module doc for why the scale-2x case needs one), window (at
/// `scale_factor`), and the target/decoy button pair; settles
/// layout/camera state; returns the app plus both button entities.
fn build_scene(scale_factor: f32, viewport: Option<Viewport>) -> (App, Entity, Entity) {
    let physical_w = (LOGICAL_W * scale_factor).round() as u32;
    let physical_h = (LOGICAL_H * scale_factor).round() as u32;

    let mut app = App::new();
    app.add_plugins(
        DefaultPlugins
            .set(WindowPlugin {
                primary_window: Some(Window {
                    resolution: WindowResolution::new(physical_w, physical_h)
                        .with_scale_factor_override(scale_factor),
                    ..Default::default()
                }),
                ..Default::default()
            })
            .build()
            // Same recipe `widgets/tests/gallery_interaction.rs` uses: every
            // subsystem bevy_ui/bevy_picking need, minus a real OS window
            // and event loop.
            .disable::<bevy::winit::WinitPlugin>(),
    );
    app.add_plugins((
        // The ONLY place `embedded_asset!` actually registers the icon PNG
        // bytes `ui_bridge::icons::ICON_PATHS` resolves against (see
        // `ui_bridge::icons`' module doc) — without it, `ui_bridge::icons::build`'s
        // `Startup` load fails with a (harmless but noisy) "path not found"
        // error every run. Not otherwise needed: this test spawns plain
        // `bevy_ui_widgets::Button` nodes, no widgets widgets or theming.
        widgets::FeathersCorePlugin,
        ButtonPlugin,
        UiBridgePlugin,
    ));

    app.world_mut().spawn((
        Camera2d,
        Camera {
            viewport,
            ..Default::default()
        },
    ));

    let root = app
        .world_mut()
        .spawn(Node {
            width: Val::Px(LOGICAL_W),
            height: Val::Px(LOGICAL_H),
            ..Default::default()
        })
        .id();

    let target = app
        .world_mut()
        .spawn((
            abs_node(BTN_X, BTN_Y, BTN_W, BTN_H),
            Button,
            Hovered::default(),
            TargetButton,
            ChildOf(root),
        ))
        .id();

    let decoy = app
        .world_mut()
        .spawn((
            abs_node(DECOY_X, DECOY_Y, DECOY_W, DECOY_H),
            Button,
            Hovered::default(),
            DecoyButton,
            ChildOf(root),
        ))
        .id();

    // Settle: PostStartup's camera_system (ui_bridge::camera::build) populates
    // Camera::computed on the first pass; a couple more PostUpdate frames
    // let bevy_ui's Prepare -> Propagate -> Content -> Layout -> PostLayout
    // chain converge on stable ComputedNode/UiGlobalTransform for the
    // absolutely-positioned nodes above.
    app.update();
    app.update();
    app.update();

    (app, target, decoy)
}

/// Injects a `PointerInput::Move` to `logical_position` followed by a
/// `PointerInput::Press(Primary)` at the same spot — the same two-event
/// shape a real mouse-then-click produces (`PointerAction::Press` alone
/// never touches `PointerLocation`; see `bevy_picking::pointer::PointerInput::receive`).
fn click_at(app: &mut App, logical_position: Vec2) {
    let window = app
        .world_mut()
        .query_filtered::<Entity, With<PrimaryWindow>>()
        .single(app.world())
        .expect("a primary window should exist");
    let target = RenderTarget::Window(WindowRef::Primary)
        .normalize(Some(window))
        .expect("primary window should normalize to a render target");
    assert!(matches!(target, NormalizedRenderTarget::Window(_)));

    let location = Location {
        target,
        position: logical_position,
    };
    app.world_mut().write_message(PointerInput::new(
        PointerId::Mouse,
        location.clone(),
        PointerAction::Move { delta: Vec2::ZERO },
    ));
    app.world_mut().write_message(PointerInput::new(
        PointerId::Mouse,
        location,
        PointerAction::Press(PointerButton::Primary),
    ));

    // One `PreUpdate` pass is enough for ProcessInput -> Backend -> Hover ->
    // PostHover to run and for `bevy_ui_widgets::ButtonPlugin`'s observer to
    // insert `Pressed` (deferred commands flush by the end of `update()`).
    app.update();
}

fn hovered(app: &App, entity: Entity) -> bool {
    app.world().get::<Hovered>(entity).is_some_and(|h| h.0)
}

fn pressed(app: &App, entity: Entity) -> bool {
    app.world().get::<Pressed>(entity).is_some()
}

/// `scale_factor == 1.0`, no explicit viewport — the realistic recipe
/// `widgets/examples/feathers_gallery.rs` itself uses (it never sets
/// `Camera::viewport`). Per the module doc this case can't discriminate the
/// bug (the pre-fix default and the real value coincide at scale 1.0), but
/// it's real, new coverage: window-coordinate hit-testing through the full
/// picking pipeline, which no existing test exercised before this one.
#[test]
fn pointer_hits_only_the_target_button_at_scale_1x() {
    let (mut app, target, decoy) = build_scene(1.0, None);

    assert!(!hovered(&app, target), "target should start unhovered");
    assert!(!hovered(&app, decoy), "decoy should start unhovered");

    let center = Vec2::new(BTN_X + BTN_W / 2.0, BTN_Y + BTN_H / 2.0);
    click_at(&mut app, center);

    assert!(hovered(&app, target), "target button should be hovered");
    assert!(pressed(&app, target), "target button should be pressed");
    assert!(!hovered(&app, decoy), "decoy should stay unhovered");
    assert!(!pressed(&app, decoy), "decoy should stay unpressed");
}

/// `scale_factor == 2.0`, WITH an explicit `Camera::viewport` — the numeric
/// proof. See the module doc for the full derivation; worked out here:
///
/// - Window: logical 400x300, scale 2.0 -> physical 800x600.
/// - `viewport`: physical position (240, 180), physical size (400, 300) —
///   comfortably inside the 800x600 window.
/// - Target button center: logical (100, 70).
/// - Injected pointer (logical, FIXED, computed once from the CORRECT scale
///   factor): `button_center + viewport_min / correct_scale`
///   = (100, 70) + (240, 180) / 2.0 = (220, 160).
///
/// With `Camera::computed` genuinely refreshed (scale' = 2.0, this fix):
/// `pointer_pos_before_subtract = (220,160) * 2.0 = (440, 320)`, inside the
/// viewport rect [240,640]x[180,480] ✓; after subtracting `viewport.min`:
/// `(200, 140)` — exactly the button's own physical center
/// (`(100,70) * 2.0`). Hit.
///
/// With `Camera::computed` stuck at its pre-fix default (scale' = 1.0,
/// independent of the real 2.0): `pointer_pos_before_subtract = (220,160) *
/// 1.0 = (220, 160)` — OUTSIDE the viewport rect [240,640]x[180,480]
/// (`220 < 240`, `160 < 180`), so `ui_picking` discards the pointer for
/// this camera entirely before it ever reaches node hit-testing. Miss.
#[test]
fn pointer_hits_only_the_target_button_at_scale_2x_with_explicit_viewport() {
    let viewport = Viewport {
        physical_position: UVec2::new(240, 180),
        physical_size: UVec2::new(400, 300),
        depth: 0.0..1.0,
    };
    let (mut app, target, decoy) = build_scene(2.0, Some(viewport));

    assert!(!hovered(&app, target), "target should start unhovered");
    assert!(!hovered(&app, decoy), "decoy should start unhovered");

    let logical_pointer = Vec2::new(220.0, 160.0);
    click_at(&mut app, logical_pointer);

    assert!(
        hovered(&app, target),
        "target button should be hovered at scale_factor=2.0 with an \
         explicit viewport — a Camera::computed stuck at its pre-fix \
         default would have scaled the pointer by 1.0 instead of 2.0, \
         landing outside the viewport rect entirely (see the module doc's \
         worked arithmetic)"
    );
    assert!(
        pressed(&app, target),
        "target button should be pressed at scale_factor=2.0 with an explicit viewport"
    );
    assert!(!hovered(&app, decoy), "decoy should stay unhovered");
    assert!(!pressed(&app, decoy), "decoy should stay unpressed");
}
