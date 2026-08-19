//! Headless widget smoke tests without windows, render plugins, or a GPU.
//! Pointer and keyboard messages are dispatched directly to ECS targets.

use std::time::Duration;

use bevy::{
    camera::NormalizedRenderTarget,
    input::{
        ButtonState, InputPlugin,
        keyboard::{Key, KeyCode, KeyboardInput},
    },
    input_focus::{InputDispatchPlugin, InputFocus, InputFocusPlugin},
    picking::{
        backend::HitData,
        events::{Click, Pointer, Press},
        pointer::{Location, PointerButton, PointerId},
    },
    prelude::*,
    window::PrimaryWindow,
};
use bevy_ui_widgets::{
    Activate, Button, ButtonPlugin, Slider, SliderPlugin, SliderRange, SliderStep, SliderValue,
    slider_self_update,
};

#[derive(Resource, Default)]
struct Activated(bool);

fn record_activation(_: On<Activate>, mut activated: ResMut<Activated>) {
    activated.0 = true;
}

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

#[test]
fn button_activates_after_a_targeted_pointer_click() {
    let mut app = App::new();
    app.add_plugins(ButtonPlugin).init_resource::<Activated>();

    let button = app
        .world_mut()
        .spawn(Button)
        .observe(record_activation)
        .id();

    app.update();
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
    app.update();
    app.world_mut().trigger(Pointer::new_without_propagate(
        PointerId::Mouse,
        pointer_location(),
        Click {
            button: PointerButton::Primary,
            hit: hit(),
            duration: Duration::ZERO,
            count: 1,
        },
        button,
    ));

    app.update();

    assert!(app.world().resource::<Activated>().0);
}

#[test]
fn focused_slider_increases_on_right_arrow() {
    let mut app = App::new();
    app.add_plugins((
        InputPlugin,
        InputFocusPlugin,
        InputDispatchPlugin,
        SliderPlugin,
    ))
    .add_observer(slider_self_update);

    let input_dispatch_target = app.world_mut().spawn(PrimaryWindow).id();
    let slider = app
        .world_mut()
        .spawn((
            Slider::default(),
            SliderValue(4.0),
            SliderRange::new(0.0, 10.0),
            SliderStep(0.5),
        ))
        .id();
    app.world_mut()
        .insert_resource(InputFocus::from_entity(slider));

    app.update();
    app.world_mut().write_message(KeyboardInput {
        key_code: KeyCode::ArrowRight,
        logical_key: Key::ArrowRight,
        state: ButtonState::Pressed,
        text: None,
        repeat: false,
        window: input_dispatch_target,
    });
    app.update();

    assert_eq!(
        app.world().get::<SliderValue>(slider),
        Some(&SliderValue(4.5))
    );
}
