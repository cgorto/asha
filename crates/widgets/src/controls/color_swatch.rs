use bevy_app::{Plugin, PostUpdate};
use bevy_color::{Alpha, Color};
use bevy_ecs::{
    bundle::Bundle,
    children,
    component::Component,
    hierarchy::Children,
    query::Changed,
    reflect::ReflectComponent,
    system::{Commands, Query},
};
use bevy_reflect::{Reflect, prelude::ReflectDefault};
use bevy_scene::prelude::*;
use bevy_ui::{BackgroundColor, BorderRadius, Node, PositionType, px};

use crate::{alpha_pattern::AlphaPattern, constants::size, palette};

/// Color swatch widget, spawnable as a scene component.
#[derive(SceneComponent, Default, Clone, Reflect)]
#[reflect(Component, Clone, Default)]
pub struct FeathersColorSwatch;

/// Swatch color copied to the child background.
#[derive(Component, Default, Clone, Reflect)]
#[reflect(Component, Clone, Default)]
pub struct ColorSwatchValue(pub Color);

/// Marker used to locate and modify the swatch foreground.
#[derive(Component, Default, Clone, Reflect)]
#[reflect(Component, Clone, Default)]
pub struct ColorSwatchFg;

impl FeathersColorSwatch {
    fn scene() -> impl Scene {
        bsn! {
            Node {
                height: size::ROW_HEIGHT,
                min_width: size::ROW_HEIGHT,
                border_radius: px(5),
            }
            FeathersColorSwatch
            ColorSwatchValue
            AlphaPattern
            Children [(
                Node {
                    position_type: PositionType::Absolute,
                    left: px(0),
                    top: px(0),
                    bottom: px(0),
                    right: px(0),
                    border_radius: px(5),
                }
                ColorSwatchFg
                BackgroundColor({palette::ACCENT.with_alpha(0.5)})
            )]
        }
    }
}

/// Spawns a color swatch.
///
/// # Arguments
/// `overrides` augments the default swatch components.
#[deprecated(since = "0.19.0", note = "Use the color_swatch() BSN function")]
pub fn color_swatch_bundle<B: Bundle>(overrides: B) -> impl Bundle {
    (
        Node {
            height: size::ROW_HEIGHT,
            min_width: size::ROW_HEIGHT,
            border_radius: BorderRadius::all(px(5)),
            ..Default::default()
        },
        FeathersColorSwatch,
        ColorSwatchValue::default(),
        AlphaPattern,
        overrides,
        children![(
            Node {
                position_type: PositionType::Absolute,
                left: px(0),
                top: px(0),
                bottom: px(0),
                right: px(0),
                border_radius: BorderRadius::all(px(5)),
                ..Default::default()
            },
            ColorSwatchFg,
            BackgroundColor(palette::ACCENT.with_alpha(0.5)),
        )],
    )
}

fn update_swatch_color(
    q_swatch: Query<(&ColorSwatchValue, &Children), Changed<ColorSwatchValue>>,
    mut commands: Commands,
) {
    for (value, children) in q_swatch.iter() {
        if let Some(first_child) = children.first() {
            commands
                .entity(*first_child)
                .insert(BackgroundColor(value.0));
        }
    }
}

/// Registers swatch-color observers.
pub struct ColorSwatchPlugin;

impl Plugin for ColorSwatchPlugin {
    fn build(&self, app: &mut bevy_app::App) {
        app.add_systems(PostUpdate, update_swatch_color);
    }
}
