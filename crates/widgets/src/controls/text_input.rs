use bevy_app::{Plugin, PreUpdate, PropagateOver};
use bevy_asset::AssetServer;
use bevy_ecs::{
    change_detection::DetectChanges,
    entity::Entity,
    lifecycle::RemovedComponents,
    query::{Added, Has, With},
    reflect::ReflectComponent,
    schedule::IntoScheduleConfigs,
    system::{Commands, Query, Res},
    template::template,
};
use bevy_input_focus::tab_navigation::TabIndex;
use bevy_picking::PickingSystems;
use bevy_reflect::Reflect;
use bevy_reflect::std_traits::ReflectDefault;
use bevy_scene::prelude::*;
use bevy_text::{
    EditableText, FontSource, FontWeight, LineBreak, TextCursorStyle, TextFont, TextLayout,
};
use bevy_ui::{
    AlignItems, BorderRadius, Display, InteractionDisabled, JustifyContent, Node, UiRect, px,
};

use crate::{
    constants::{fonts, size},
    cursor::EntityCursor,
    focus::FocusWithinIndicator,
    font_styles::InheritableFont,
    theme::{InheritableThemeTextColor, ThemeBackgroundColor, UiTheme},
    tokens,
};

/// Text-input frame for adjacent icons, spawnable as a scene component.
#[derive(SceneComponent, Default, Clone, Reflect)]
#[reflect(Component, Default, Clone)]
pub struct FeathersTextInputContainer;

impl FeathersTextInputContainer {
    fn scene() -> impl Scene {
        bsn! {
            Node {
                height: size::ROW_HEIGHT,
                display: Display::Flex,
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                padding: UiRect {
                    right: px(3.0),
                },
                border: UiRect {
                    left: px(3.0)
                },
                flex_grow: 1.0,
                border_radius: {BorderRadius::all(px(4.0))},
                column_gap: px(4),
            }
            FeathersTextInputContainer
            FocusWithinIndicator
            ThemeBackgroundColor(tokens::TEXT_INPUT_BG)
            InheritableThemeTextColor(tokens::TEXT_INPUT_TEXT)
            InheritableFont {
                font: fonts::REGULAR,
                font_size: size::COMPACT_FONT,
                weight: FontWeight::NORMAL,
            }
        }
    }
}

/// Spawns a text input inside [`FeathersTextInputContainer`].
///
/// Spawnable with optional [`FeathersTextInputProps`].
///
/// ```ignore
/// :FeathersTextInputContainer
/// Children [
///     :FeathersTextInput
/// ]
/// ```
#[derive(SceneComponent, Default, Clone)]
#[scene(FeathersTextInputProps)]
#[derive(Reflect)]
#[reflect(Component, Default, Clone)]
pub struct FeathersTextInput;

/// Properties for a [`FeathersTextInput`] scene.
#[derive(Default, Clone)]
pub struct FeathersTextInputProps {
    /// Visible width.
    pub visible_width: Option<f32>,
    /// Maximum characters.
    pub max_characters: Option<usize>,
}

impl FeathersTextInput {
    fn scene(props: FeathersTextInputProps) -> impl Scene {
        bsn! {
            Node {
                flex_grow: {
                    if props.visible_width.is_some() {
                        0.
                    } else {
                        1.
                    }
                } ,
            }
            FeathersTextInput
            EditableText {
                cursor_width: 0.3,
                visible_width: {props.visible_width},
                max_characters: {props.max_characters},
            }
            TextLayout {
                linebreak: LineBreak::NoWrap,
            }
            TabIndex(0)
            template(|ctx| {
                Ok(TextFont {
                    font: FontSource::Handle(ctx.resource::<AssetServer>().load(fonts::REGULAR)),
                    font_size: size::COMPACT_FONT,
                    weight: FontWeight::NORMAL,
                    ..Default::default()
                })
            })
            PropagateOver<TextFont>
            EntityCursor::System(bevy_window::SystemCursorIcon::Text)
            TextCursorStyle::default()
        }
    }
}

fn update_text_cursor_color(
    mut q_text_input: Query<&mut TextCursorStyle, With<FeathersTextInput>>,
    theme: Res<UiTheme>,
) {
    if theme.is_changed() {
        for mut cursor_style in q_text_input.iter_mut() {
            cursor_style.color = theme.color(&tokens::TEXT_INPUT_CURSOR);
            cursor_style.selection_color = theme.color(&tokens::TEXT_INPUT_SELECTION);
            cursor_style.unfocused_selection_color =
                theme.color(&tokens::TEXT_INPUT_SELECTION_UNFOCUSED);
        }
    }
}

fn update_text_input_styles(
    q_inputs: Query<
        (Entity, Has<InteractionDisabled>, &InheritableThemeTextColor),
        (With<FeathersTextInput>, Added<InteractionDisabled>),
    >,
    mut commands: Commands,
) {
    for (input_ent, disabled, font_color) in q_inputs.iter() {
        set_text_input_styles(input_ent, disabled, font_color, &mut commands);
    }
}

fn update_text_input_styles_remove(
    q_inputs: Query<
        (Entity, Has<InteractionDisabled>, &InheritableThemeTextColor),
        With<FeathersTextInput>,
    >,
    mut removed_disabled: RemovedComponents<InteractionDisabled>,
    mut commands: Commands,
) {
    removed_disabled.read().for_each(|ent| {
        if let Ok((input_ent, disabled, font_color)) = q_inputs.get(ent) {
            set_text_input_styles(input_ent, disabled, font_color, &mut commands);
        }
    });
}

fn set_text_input_styles(
    input_ent: Entity,
    disabled: bool,
    font_color: &InheritableThemeTextColor,
    commands: &mut Commands,
) {
    let font_color_token = match disabled {
        true => tokens::TEXT_INPUT_TEXT_DISABLED,
        false => tokens::TEXT_INPUT_TEXT,
    };

    let cursor_shape = match disabled {
        true => bevy_window::SystemCursorIcon::NotAllowed,
        false => bevy_window::SystemCursorIcon::Text,
    };

    if font_color.0 != font_color_token {
        commands
            .entity(input_ent)
            .insert(InheritableThemeTextColor(font_color_token));
    }

    commands
        .entity(input_ent)
        .insert(EntityCursor::System(cursor_shape));
}

/// Registers text-input styling systems.
pub struct TextInputPlugin;

impl Plugin for TextInputPlugin {
    fn build(&self, app: &mut bevy_app::App) {
        app.add_systems(
            PreUpdate,
            (
                update_text_cursor_color,
                update_text_input_styles,
                update_text_input_styles_remove,
            )
                .in_set(PickingSystems::Last),
        );
    }
}
