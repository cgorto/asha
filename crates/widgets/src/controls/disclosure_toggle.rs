use bevy_app::{App, Plugin, PreUpdate};
use bevy_ecs::{
    hierarchy::Children,
    lifecycle::RemovedComponents,
    query::{Added, Has, Or, With},
    reflect::ReflectComponent,
    schedule::IntoScheduleConfigs,
    system::{Query, Res},
};
use bevy_input_focus::tab_navigation::TabIndex;
use bevy_math::Rot2;
use bevy_picking::PickingSystems;
use bevy_reflect::Reflect;
use bevy_reflect::std_traits::ReflectDefault;
use bevy_scene::{Scene, SceneComponent, bsn};
use bevy_ui::{
    AlignItems, Checked, Display, InteractionDisabled, JustifyContent, Node, UiTransform, px,
    widget::ImageNode,
};
use bevy_ui_widgets::Checkbox;
use bevy_window::SystemCursorIcon;

use crate::{
    constants::icons, cursor::EntityCursor, display::icon, focus::FocusIndicator, theme::UiTheme,
    tokens,
};

/// Toggle button for expanding or collapsing panels, spawnable as a scene
/// component.
///
/// Its [`Checked`] state controls the chevron direction.
#[derive(SceneComponent, Default, Clone, Reflect)]
#[reflect(Component, Default, Clone)]
pub struct FeathersDisclosureToggle;

impl FeathersDisclosureToggle {
    fn scene() -> impl Scene {
        bsn!(
            Node {
                width: px(12),
                height: px(12),
                display: Display::Flex,
                align_items: AlignItems::Center,
                justify_content: JustifyContent::Center,
            }
            Checkbox
            EntityCursor::System(SystemCursorIcon::Pointer)
            FocusIndicator
            TabIndex(0)
            Children [
                icon(icons::CHEVRON_RIGHT)
            ]
        )
    }
}

fn update_toggle_styles(
    mut q_toggle: Query<
        (
            Has<InteractionDisabled>,
            Has<Checked>,
            &mut UiTransform,
            &Children,
        ),
        (
            With<FeathersDisclosureToggle>,
            Or<(Added<Checkbox>, Added<Checked>, Added<InteractionDisabled>)>,
        ),
    >,
    mut q_icon: Query<&mut ImageNode>,
    theme: Res<UiTheme>,
) {
    for (disabled, checked, mut transform, children) in q_toggle.iter_mut() {
        let Some(child_id) = children.first() else {
            continue;
        };
        let Ok(mut icon_child) = q_icon.get_mut(*child_id) else {
            continue;
        };
        set_toggle_styles(
            disabled,
            checked,
            transform.as_mut(),
            &mut icon_child,
            &theme,
        );
    }
}

fn update_toggle_styles_remove(
    mut q_toggle: Query<
        (
            Has<InteractionDisabled>,
            Has<Checked>,
            &mut UiTransform,
            &Children,
        ),
        With<FeathersDisclosureToggle>,
    >,
    mut q_icon: Query<&mut ImageNode>,
    mut removed_disabled: RemovedComponents<InteractionDisabled>,
    mut removed_checked: RemovedComponents<Checked>,
    theme: Res<UiTheme>,
) {
    removed_disabled
        .read()
        .chain(removed_checked.read())
        .for_each(|ent| {
            if let Ok((disabled, checked, mut transform, children)) = q_toggle.get_mut(ent) {
                let Some(child_id) = children.first() else {
                    return;
                };
                let Ok(mut icon_child) = q_icon.get_mut(*child_id) else {
                    return;
                };
                set_toggle_styles(
                    disabled,
                    checked,
                    transform.as_mut(),
                    &mut icon_child,
                    &theme,
                );
            }
        });
}

fn set_toggle_styles(
    disabled: bool,
    checked: bool,
    transform: &mut UiTransform,
    image_node: &mut ImageNode,
    theme: &Res<'_, UiTheme>,
) {
    // Match the plain tool-button caption color.
    let icon_color = match disabled {
        true => theme.color(&tokens::BUTTON_TEXT_DISABLED),
        false => theme.color(&tokens::BUTTON_TEXT),
    };

    if image_node.color != icon_color {
        image_node.color = icon_color;
    }

    match checked {
        true => {
            transform.rotation = Rot2::turn_fraction(0.25);
        }
        false => {
            transform.rotation = Rot2::turn_fraction(0.0);
        }
    };
}

/// Registers disclosure-toggle styling systems.
pub struct DisclosureTogglePlugin;

impl Plugin for DisclosureTogglePlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            PreUpdate,
            (update_toggle_styles, update_toggle_styles_remove).in_set(PickingSystems::Last),
        );
    }
}
