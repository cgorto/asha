//! Logs controller connections, buttons, and axes without a renderer.

mod common;

use bevy::input::gamepad::{
    GamepadAxisChangedEvent, GamepadButtonChangedEvent, GamepadConnection, GamepadConnectionEvent,
};
use bevy::prelude::*;
use bevy::window::{Window, WindowPlugin};
use bevy::winit::{UpdateMode, WinitSettings};
use common::esc_to_exit;
use core::time::Duration;

fn main() {
    App::new()
        .insert_resource(WinitSettings {
            focused_mode: UpdateMode::reactive_low_power(Duration::from_millis(16)),
            unfocused_mode: UpdateMode::reactive_low_power(Duration::from_millis(16)),
        })
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "asha gamepad probe — mash buttons, watch stdout".into(),
                resolution: bevy::window::WindowResolution::new(480, 240),
                ..Default::default()
            }),
            ..Default::default()
        }))
        .add_systems(
            Update,
            (log_connections, log_buttons, log_axes, esc_to_exit),
        )
        .run();
}

fn log_connections(mut events: MessageReader<GamepadConnectionEvent>) {
    for ev in events.read() {
        match &ev.connection {
            GamepadConnection::Connected {
                name,
                vendor_id,
                product_id,
            } => {
                println!(
                    "[connect] {:?} name={name:?} vendor={:04x?} product={:04x?}",
                    ev.gamepad, vendor_id, product_id
                );
            }
            GamepadConnection::Disconnected => println!("[disconnect] {:?}", ev.gamepad),
        }
    }
}

fn log_buttons(mut events: MessageReader<GamepadButtonChangedEvent>) {
    for ev in events.read() {
        println!(
            "[button] {:?} {:?} {:?} value={:.3}",
            ev.entity, ev.button, ev.state, ev.value
        );
    }
}

fn log_axes(mut events: MessageReader<GamepadAxisChangedEvent>) {
    for ev in events.read() {
        println!(
            "[axis] {:?} {:?} value={:+.3}",
            ev.entity, ev.axis, ev.value
        );
    }
}
