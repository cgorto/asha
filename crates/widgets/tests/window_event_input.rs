//! Headless test for `WindowEvent` to `PointerInput` translation.
//!
//! Injects the aggregate winit event stream, then runs real picking and button input.

use bevy::app::App;
use bevy::camera::Camera2d;
use bevy::ecs::component::Component;
use bevy::input::ButtonState;
use bevy::input::mouse::{MouseButton, MouseButtonInput};
use bevy::math::Vec2;
use bevy::picking::hover::Hovered;
use bevy::prelude::*;
use bevy::ui::{Node, PositionType, Pressed, Val};
use bevy::window::{
    CursorMoved, PrimaryWindow, Window, WindowEvent, WindowPlugin, WindowResolution,
};

use bevy_ui_widgets::{Button, ButtonPlugin};

use ui_bridge::UiBridgePlugin;

const LOGICAL_W: f32 = 400.0;
const LOGICAL_H: f32 = 300.0;

const BTN_X: f32 = 50.0;
const BTN_Y: f32 = 50.0;
const BTN_W: f32 = 100.0;
const BTN_H: f32 = 40.0;

#[derive(Component)]
struct TargetButton;

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

/// Same scaffolding as `pointer_window_coords::build_scene`, scale 1.0, no
/// viewport — the realistic gallery-shaped recipe.
fn build_scene() -> (App, Entity) {
    let mut app = App::new();
    app.add_plugins(
        DefaultPlugins
            .set(WindowPlugin {
                primary_window: Some(Window {
                    resolution: WindowResolution::new(LOGICAL_W as u32, LOGICAL_H as u32)
                        .with_scale_factor_override(1.0),
                    ..Default::default()
                }),
                ..Default::default()
            })
            .build()
            .disable::<bevy::winit::WinitPlugin>(),
    );
    app.add_plugins((widgets::FeathersCorePlugin, ButtonPlugin, UiBridgePlugin));

    app.world_mut().spawn(Camera2d);

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

    app.update();
    app.update();
    app.update();

    (app, target)
}

fn primary_window(app: &mut App) -> Entity {
    app.world_mut()
        .query_filtered::<Entity, With<PrimaryWindow>>()
        .single(app.world())
        .expect("a primary window should exist")
}

#[test]
fn raw_window_events_hover_and_press_the_button() {
    let (mut app, target) = build_scene();
    let window = primary_window(&mut app);
    let center = Vec2::new(BTN_X + BTN_W / 2.0, BTN_Y + BTN_H / 2.0);

    // The winit-shaped event sequence for "move onto the button, press":
    // the aggregate `WindowEvent` stream is what `mouse_pick_events` reads.
    app.world_mut()
        .write_message(WindowEvent::CursorMoved(CursorMoved {
            window,
            position: center,
            delta: Some(Vec2::ZERO),
        }));
    app.update();

    let hovered = app
        .world()
        .get::<Hovered>(target)
        .expect("target keeps its Hovered component");
    assert!(
        hovered.get(),
        "CursorMoved WindowEvent over the button must hover it \
         (mouse_pick_events -> ui_picking -> hover map)"
    );

    app.world_mut()
        .write_message(WindowEvent::MouseButtonInput(MouseButtonInput {
            button: MouseButton::Left,
            state: ButtonState::Pressed,
            window,
        }));
    app.update();

    assert!(
        app.world().get::<Pressed>(target).is_some(),
        "MouseButtonInput WindowEvent while hovered must press the button \
         (mouse_pick_events -> pointer_events -> ButtonPlugin)"
    );

    app.world_mut()
        .write_message(WindowEvent::MouseButtonInput(MouseButtonInput {
            button: MouseButton::Left,
            state: ButtonState::Released,
            window,
        }));
    app.update();

    assert!(
        app.world().get::<Pressed>(target).is_none(),
        "release must clear Pressed again"
    );
}
