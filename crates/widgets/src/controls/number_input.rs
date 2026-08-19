use bevy_app::PropagateOver;
use bevy_ecs::{
    component::Component,
    entity::Entity,
    event::EntityEvent,
    hierarchy::{ChildOf, Children},
    observer::On,
    query::With,
    reflect::{ReflectComponent, ReflectEvent},
    relationship::Relationship,
    system::{Commands, Query, Res},
};
use bevy_input::keyboard::{KeyCode, KeyboardInput};
use bevy_input_focus::{FocusLost, FocusedInput, InputFocus};
use bevy_log::warn;
use bevy_reflect::Reflect;
use bevy_reflect::std_traits::ReflectDefault;
use bevy_scene::prelude::*;
use bevy_text::{
    EditableText, EditableTextFilter, FontSourceTemplate, TextEdit, TextEditChange, TextFont,
};
use bevy_ui::{AlignItems, AlignSelf, Display, JustifyContent, Node, UiRect, px, widget::Text};
use bevy_ui_widgets::{SelectAllOnFocus, ValueChange};

use crate::{
    constants::{fonts, size},
    controls::{FeathersTextInput, FeathersTextInputContainer},
    theme::{ThemeBackgroundColor, ThemeBorderColor, ThemeTextColor, ThemeToken},
    tokens,
};

/// Numeric text input with two-way synchronization, spawnable as a scene
/// component with optional [`FeathersNumberInputProps`].
///
/// Focused inputs emit [`ValueChange`] events as the user types. Their payload
/// is `f32`, `f64`, `i32`, or `i64`, according to [`NumberFormat`]. Unfocused
/// inputs accept [`UpdateNumberInput`] events and replace their text buffer.
///
/// The external numeric value is typically the source of truth: update it from
/// [`ValueChange`] events, and send [`UpdateNumberInput`] when it changes.
/// Avoid sending updates when the value has not changed.
// TODO: Add validation when text-input validation is available.
#[derive(SceneComponent, Default, Clone)]
#[scene(FeathersNumberInputProps)]
#[derive(Reflect)]
#[reflect(Component, Default, Clone)]
pub struct FeathersNumberInput;

/// Properties for a [`FeathersNumberInput`] scene.
pub struct FeathersNumberInputProps {
    /// Colored axis strip; transparent by default.
    pub sigil_color: ThemeToken,
    /// Caption beside the axis strip, usually X, Y, or Z.
    pub label_text: Option<&'static str>,
    /// Numeric format being edited.
    pub number_format: NumberFormat,
}

impl Default for FeathersNumberInputProps {
    fn default() -> Self {
        Self {
            sigil_color: tokens::TEXT_INPUT_BG,
            label_text: None,
            number_format: NumberFormat::F32,
        }
    }
}

impl FeathersNumberInput {
    fn scene(props: FeathersNumberInputProps) -> impl Scene {
        bsn! {
            @FeathersTextInputContainer
            ThemeBorderColor({props.sigil_color})
            FeathersNumberInput
            template_value(props.number_format)
            on(number_input_on_update)
            Children [
                {
                    match props.label_text {
                        Some(text) => Box::new(bsn_list!(
                            Node {
                                display: Display::Flex,
                                align_items: AlignItems::Center,
                                align_self: AlignSelf::Stretch,
                                justify_content: JustifyContent::Center,
                                padding: UiRect::axes(px(6), px(0)),
                            }
                            ThemeBackgroundColor(tokens::TEXT_INPUT_LABEL_BG)
                            Children [
                                Text(text)
                                TextFont {
                                    font: FontSourceTemplate::Handle(fonts::REGULAR),
                                    font_size: size::COMPACT_FONT,
                                }
                                PropagateOver<TextFont>
                                ThemeTextColor(tokens::TEXT_INPUT_TEXT)
                            ]
                        )) as Box<dyn SceneList>,
                        None => Box::new(bsn_list!()) as Box<dyn SceneList>
                    }
                }
                @FeathersTextInput {
                    @max_characters: 20usize,
                }
                SelectAllOnFocus
                on(number_input_on_text_change)
                on(number_input_on_enter_key)
                on(number_input_on_focus_loss)
                EditableTextFilter::new(|c| {
                    c.is_ascii_digit() || matches!(c, '.' | '-' | '+' | 'e' | 'E')
                }),
            ]
        }
    }
}

/// Numeric format determining the payload type of [`ValueChange`] events.
#[derive(Component, Default, Clone, Copy, Reflect)]
#[reflect(Component, Default, Clone)]
pub enum NumberFormat {
    /// 32-bit float.
    #[default]
    F32,
    /// 64-bit float.
    F64,
    /// 32-bit integer.
    I32,
    /// 64-bit integer.
    I64,
}

/// Numeric value in a supported format.
#[derive(Debug, PartialEq, Clone, Copy, Reflect)]
pub enum NumberInputValue {
    /// An `f32` value.
    F32(f32),
    /// An `f64` value.
    F64(f64),
    /// An `i32` value.
    I32(i32),
    /// An `i64` value.
    I64(i64),
}

impl core::fmt::Display for NumberInputValue {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            NumberInputValue::F32(v) => write!(f, "{}", v),
            NumberInputValue::F64(v) => write!(f, "{}", v),
            NumberInputValue::I32(v) => write!(f, "{}", v),
            NumberInputValue::I64(v) => write!(f, "{}", v),
        }
    }
}

/// Updates a number input's displayed value when it is not focused.
#[derive(Clone, EntityEvent, Reflect)]
#[reflect(Event, Clone)]
pub struct UpdateNumberInput {
    /// Target widget.
    pub entity: Entity,

    /// Replacement value.
    pub value: NumberInputValue,
}

fn number_input_on_text_change(
    change: On<TextEditChange>,
    q_parent: Query<&ChildOf>,
    q_number_input: Query<&NumberFormat, With<FeathersNumberInput>>,
    q_text_input: Query<&EditableText>,
    mut commands: Commands,
) {
    let Ok(parent) = q_parent.get(change.event_target()) else {
        return;
    };

    let Ok(number_format) = q_number_input.get(parent.get()) else {
        return;
    };

    let Ok(editable_text) = q_text_input.get(change.event_target()) else {
        return;
    };

    let text_value = editable_text.value().to_string();
    emit_value_change(text_value, *number_format, parent.0, &mut commands, false);
}

fn number_input_on_update(
    update: On<UpdateNumberInput>,
    q_children: Query<&Children>,
    q_number_input: Query<(), With<FeathersNumberInput>>,
    mut q_text_input: Query<&mut EditableText>,
    focus: Res<InputFocus>,
) {
    if !q_number_input.contains(update.event_target()) {
        return;
    };

    let Ok(children) = q_children.get(update.event_target()) else {
        return;
    };

    for child_id in children.iter() {
        if focus.get() != Some(*child_id)
            && let Ok(mut editable_text) = q_text_input.get_mut(*child_id)
        {
            let new_digits = update.value.to_string();
            let old_digits = editable_text.value().to_string();
            if old_digits != new_digits {
                editable_text.queue_edit(TextEdit::SelectAll);
                editable_text.queue_edit(TextEdit::Insert(new_digits.into()));
            }
            break;
        }
    }
}

fn number_input_on_enter_key(
    key_input: On<FocusedInput<KeyboardInput>>,
    q_parent: Query<&ChildOf>,
    q_number_input: Query<&NumberFormat, With<FeathersNumberInput>>,
    q_text_input: Query<&EditableText>,
    mut commands: Commands,
) {
    if key_input.input.key_code != KeyCode::Enter {
        return;
    }

    let Ok(parent) = q_parent.get(key_input.event_target()) else {
        return;
    };

    let Ok(number_format) = q_number_input.get(parent.get()) else {
        return;
    };

    let Ok(editable_text) = q_text_input.get(key_input.event_target()) else {
        return;
    };

    let text_value = editable_text.value().to_string();
    emit_value_change(text_value, *number_format, parent.0, &mut commands, true);
}

fn number_input_on_focus_loss(
    focus_lost: On<FocusLost>,
    q_parent: Query<&ChildOf>,
    q_number_input: Query<&NumberFormat, With<FeathersNumberInput>>,
    mut q_text_input: Query<&mut EditableText>,
    mut commands: Commands,
) {
    let editable_text_id = focus_lost.event_target();

    let Ok(parent) = q_parent.get(editable_text_id) else {
        return;
    };

    let Ok(number_format) = q_number_input.get(parent.get()) else {
        return;
    };

    let Ok(editable_text) = q_text_input.get_mut(editable_text_id) else {
        return;
    };

    let text_value = editable_text.value().to_string();
    emit_value_change(text_value, *number_format, parent.0, &mut commands, true);
}

fn emit_value_change(
    text_value: String,
    format: NumberFormat,
    source: Entity,
    commands: &mut Commands,
    is_final: bool,
) {
    let text_value = text_value.trim();
    if text_value.is_empty() {
        return;
    }

    match format {
        NumberFormat::F32 => {
            match text_value.parse::<f32>() {
                Ok(new_value) => {
                    commands.trigger(ValueChange {
                        source,
                        value: new_value,
                        is_final,
                    });
                }
                Err(_) => {
                    // TODO: Emit a validation error once these are defined
                    warn!("Invalid floating-point number in text edit");
                }
            }
        }
        NumberFormat::F64 => {
            match text_value.parse::<f64>() {
                Ok(new_value) => {
                    commands.trigger(ValueChange {
                        source,
                        value: new_value,
                        is_final,
                    });
                }
                Err(_) => {
                    // TODO: Emit a validation error once these are defined
                    warn!("Invalid floating-point number in text edit");
                }
            }
        }
        NumberFormat::I32 => {
            match text_value.parse::<i32>() {
                Ok(new_value) => {
                    commands.trigger(ValueChange {
                        source,
                        value: new_value,
                        is_final,
                    });
                }
                Err(_) => {
                    // TODO: Emit a validation error once these are defined
                    warn!("Invalid integer number in text edit");
                }
            }
        }
        NumberFormat::I64 => {
            match text_value.parse::<i64>() {
                Ok(new_value) => {
                    commands.trigger(ValueChange {
                        source,
                        value: new_value,
                        is_final,
                    });
                }
                Err(_) => {
                    // TODO: Emit a validation error once these are defined
                    warn!("Invalid integer number in text edit");
                }
            }
        }
    }
}
