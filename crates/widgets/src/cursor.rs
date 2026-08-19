//! Cursor management for hovered UI entities.
use bevy_app::{App, Plugin, PreUpdate};
use bevy_derive::Deref;
use bevy_ecs::{
    component::Component,
    entity::Entity,
    hierarchy::ChildOf,
    query::{With, Without},
    reflect::{ReflectComponent, ReflectResource},
    resource::Resource,
    schedule::IntoScheduleConfigs,
    system::{Commands, Query, Res},
    template::FromTemplate,
};
use bevy_picking::{PickingSystems, hover::HoverMap, pointer::PointerId};
use bevy_reflect::{Reflect, std_traits::ReflectDefault};
#[cfg(feature = "custom_cursor")]
use bevy_window::CustomCursor;
use bevy_window::{CursorIcon, SystemCursorIcon, Window};

/// Default window cursor when no entity is hovered.
#[derive(Deref, Resource, Debug, Clone, Default, Reflect)]
#[reflect(Resource, Debug, Default)]
pub struct DefaultCursor(pub EntityCursor);

/// Cursor shape copied to the window while hovering.
///
/// Supports system and optional custom cursors.
#[derive(Component, Debug, Clone, Reflect, PartialEq, Eq, FromTemplate)]
#[reflect(Component, Debug, Default, PartialEq, Clone)]
pub enum EntityCursor {
    #[cfg(feature = "custom_cursor")]
    /// Custom cursor image.
    Custom(CustomCursor),
    #[default]
    /// System provided cursor icon.
    System(SystemCursorIcon),
}

/// Temporarily overrides entity cursor changes.
#[derive(Deref, Resource, Debug, Clone, Default, Reflect)]
#[reflect(Resource, Default, Clone, Debug)]
pub struct OverrideCursor(pub Option<EntityCursor>);

impl EntityCursor {
    /// Converts this cursor to a window [`CursorIcon`].
    pub fn to_cursor_icon(&self) -> CursorIcon {
        match self {
            #[cfg(feature = "custom_cursor")]
            EntityCursor::Custom(custom_cursor) => CursorIcon::Custom(custom_cursor.clone()),
            EntityCursor::System(icon) => CursorIcon::from(*icon),
        }
    }

    /// Compares this cursor with a window [`CursorIcon`].
    pub fn eq_cursor_icon(&self, cursor_icon: &CursorIcon) -> bool {
        // `as_system` keeps matching valid across cursor features.
        match (self, cursor_icon, cursor_icon.as_system()) {
            #[cfg(feature = "custom_cursor")]
            (EntityCursor::Custom(custom), CursorIcon::Custom(other), _) => custom == other,
            (EntityCursor::System(system), _, Some(cursor_icon)) => *system == *cursor_icon,
            _ => false,
        }
    }
}

impl Default for EntityCursor {
    fn default() -> Self {
        EntityCursor::System(Default::default())
    }
}

/// Updates the window cursor from hovered entities or [`DefaultCursor`].
pub(crate) fn update_cursor(
    mut commands: Commands,
    hover_map: Option<Res<HoverMap>>,
    parent_query: Query<&ChildOf>,
    cursor_query: Query<&EntityCursor, Without<Window>>,
    q_windows: Query<(Entity, Option<&CursorIcon>), With<Window>>,
    r_default_cursor: Res<DefaultCursor>,
    r_override_cursor: Res<OverrideCursor>,
) {
    let cursor = r_override_cursor.0.as_ref().unwrap_or_else(|| {
        hover_map
            .and_then(|hover_map| match hover_map.get(&PointerId::Mouse) {
                Some(hover_set) => hover_set.keys().find_map(|entity| {
                    cursor_query.get(*entity).ok().or_else(|| {
                        parent_query
                            .iter_ancestors(*entity)
                            .find_map(|e| cursor_query.get(e).ok())
                    })
                }),
                None => None,
            })
            .unwrap_or(&r_default_cursor)
    });

    for (entity, prev_cursor) in q_windows.iter() {
        if let Some(prev_cursor) = prev_cursor
            && cursor.eq_cursor_icon(prev_cursor)
        {
            continue;
        }
        commands.entity(entity).insert(cursor.to_cursor_icon());
    }
}

/// Plugin that supports automatically changing the cursor based on the hovered entity.
pub struct CursorIconPlugin;

impl Plugin for CursorIconPlugin {
    fn build(&self, app: &mut App) {
        if app.world().get_resource::<DefaultCursor>().is_none() {
            app.init_resource::<DefaultCursor>();
            app.init_resource::<OverrideCursor>();
        }
        app.add_systems(PreUpdate, update_cursor.in_set(PickingSystems::Last));
    }
}
