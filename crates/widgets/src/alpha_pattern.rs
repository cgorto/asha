use bevy_app::Plugin;
use bevy_ecs::{
    component::Component, lifecycle::Add, observer::On, reflect::ReflectComponent, system::Commands,
};
use bevy_reflect::{Reflect, prelude::ReflectDefault};

use ui_bridge::UiMaterialTag;

/// Marker for an entity with an alpha-pattern material tag.
#[derive(Component, Default, Clone, Reflect)]
#[reflect(Component, Default)]
pub(crate) struct AlphaPattern;

/// Adds the constant alpha-pattern material tag.
fn on_add_alpha_pattern(add: On<Add, AlphaPattern>, mut commands: Commands) {
    commands
        .entity(add.entity)
        .insert(UiMaterialTag::alpha_pattern());
}

/// Registers alpha-pattern observers.
pub struct AlphaPatternPlugin;

impl Plugin for AlphaPatternPlugin {
    fn build(&self, app: &mut bevy_app::App) {
        app.add_observer(on_add_alpha_pattern);
    }
}
