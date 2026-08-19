//! Inheritable font styles.
use bevy_app::{Propagate, PropagateOver};
use bevy_asset::Handle;
use bevy_ecs::{
    component::Component,
    lifecycle::Insert,
    observer::On,
    reflect::ReflectComponent,
    system::{Commands, Query},
    template::FromTemplate,
};
use bevy_reflect::{Reflect, prelude::ReflectDefault};
use bevy_text::{Font, FontSize, FontWeight, TextFont};

use crate::theme::ThemedText;

/// Font settings that propagate to descendant text marked with [`ThemedText`].
///
/// When inserted, this component supplies the descendant [`TextFont`] values.
#[derive(Component, Default, Clone, Debug, Reflect, FromTemplate)]
#[reflect(Component, Default)]
#[require(ThemedText, PropagateOver::<TextFont>)]
pub struct InheritableFont {
    /// The font handle.
    pub font: Handle<Font>,
    /// Font size.
    pub font_size: FontSize,
    /// Font weight.
    pub weight: FontWeight,
}

/// Propagates inserted font settings to marked descendants.
pub(crate) fn on_changed_font(
    insert: On<Insert, InheritableFont>,
    font_style: Query<&InheritableFont>,
    mut commands: Commands,
) {
    if let Ok(inheritable_font) = font_style.get(insert.entity) {
        commands.entity(insert.entity).insert(Propagate(TextFont {
            font: inheritable_font.font.clone().into(),
            font_size: inheritable_font.font_size,
            weight: inheritable_font.weight,
            ..Default::default()
        }));
    }
}
