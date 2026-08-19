use core::marker::PhantomData;

use bevy_ecs::prelude::*;
use bevy_input_focus::tab_navigation::TabGroup;
use bevy_reflect::Reflect;
use bevy_scene::prelude::*;
use bevy_ui::{FlexDirection, Node, px, widget::Text};
use bevy_ui_widgets::{Activate, observe};

use crate::controls::{FeathersButton, button::ButtonBundleProps, button_bundle};

/// Virtual keyboard widget, spawnable as a scene component with optional
/// [`VirtualKeyboardProps`].
///
/// # Emitted events
/// * [`crate::controls::VirtualKeyPressed<T>`] when a virtual key on the keyboard is activated.
///
///  These events can be disabled by adding an [`bevy_ui::InteractionDisabled`] component to the entity
#[derive(SceneComponent, FromTemplate)]
#[scene(VirtualKeyboardProps<T>)]
#[derive(Reflect)]
#[reflect(Component)]
pub struct VirtualKeyboard<T: AsRef<str> + Clone + Send + Sync + 'static>(PhantomData<fn() -> T>);

/// Properties for a [`VirtualKeyboard`] scene.
pub struct VirtualKeyboardProps<T> {
    /// Keyboard keys grouped by row.
    pub keys: Vec<Vec<T>>,
}

impl<T> Default for VirtualKeyboardProps<T> {
    fn default() -> Self {
        Self {
            keys: Default::default(),
        }
    }
}

impl<T: AsRef<str> + Clone + Send + Sync + 'static> VirtualKeyboard<T> {
    fn scene(props: VirtualKeyboardProps<T>) -> impl Scene {
        let keys = Vec::from_iter(props.keys.into_iter().map(move |row| {
            let key_row = Vec::from_iter(row.into_iter().map(move |key| {
                let key_clone = key.clone();
                bsn! {
                    @FeathersButton
                    Node {
                        flex_grow: 1.0,
                    }
                    on(
                        move |activate: On<Activate>,
                              mut commands: Commands,
                              query: Query<&ChildOf>|
                              -> Result {
                            let virtual_keyboard =
                                query.get(query.get(activate.entity)?.parent())?.parent();
                            commands.trigger(VirtualKeyPressed {
                                entity: virtual_keyboard,
                                key: key.clone(),
                            });
                            Ok(())
                        },
                    )
                    Children [
                        Text::new(key_clone.as_ref())
                    ]
                }
            }));
            bsn! {
                Node {
                    flex_direction: FlexDirection::Row,
                    column_gap: px(4),
                }
                Children [
                    {key_row}
                ]
            }
        }));
        bsn! {
            Node {
                flex_direction: FlexDirection::Column,
                row_gap: px(4),
            }
            TabGroup::new(0)
            Children [
                {keys}
            ]
        }
    }
}

/// Fired when a virtual key is activated.
#[derive(EntityEvent, Reflect)]
#[reflect(Event)]
pub struct VirtualKeyPressed<T> {
    /// Virtual keyboard entity.
    pub entity: Entity,
    /// Activated virtual key.
    pub key: T,
}

/// Spawns a virtual keyboard.
///
/// # Emitted events
/// * [`crate::controls::VirtualKeyPressed<T>`] when a virtual key on the keyboard is activated.
///
///  These events can be disabled by adding an [`bevy_ui::InteractionDisabled`] component to the entity
#[deprecated(since = "0.19.0", note = "Use the virtual_keyboard() BSN function")]
#[expect(deprecated, reason = "uses the deprecated button_bundle")]
pub fn virtual_keyboard_bundle<T>(
    keys: impl Iterator<Item = Vec<T>> + Send + Sync + 'static,
) -> impl Bundle
where
    T: AsRef<str> + Clone + Send + Sync + 'static,
{
    (
        Node {
            flex_direction: FlexDirection::Column,
            row_gap: px(4),
            ..Default::default()
        },
        TabGroup::new(0),
        Children::spawn(SpawnIter(keys.map(move |row| {
            (
                Node {
                    flex_direction: FlexDirection::Row,
                    column_gap: px(4),
                    ..Default::default()
                },
                Children::spawn(SpawnIter(row.into_iter().map(move |key| {
                    (
                        button_bundle(
                            ButtonBundleProps::default(),
                            (),
                            Spawn(Text::new(key.as_ref())),
                        ),
                        observe(
                            move |activate: On<Activate>,
                                  mut commands: Commands,
                                  query: Query<&ChildOf>|
                                  -> Result {
                                let virtual_keyboard =
                                    query.get(query.get(activate.entity)?.parent())?.parent();
                                commands.trigger(VirtualKeyPressed {
                                    entity: virtual_keyboard,
                                    key: key.clone(),
                                });
                                Ok(())
                            },
                        ),
                    )
                }))),
            )
        }))),
    )
}
