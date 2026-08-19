use bevy_app::{Plugin, PreUpdate};
use bevy_ecs::{
    bundle::Bundle,
    component::Component,
    entity::Entity,
    hierarchy::{ChildOf, Children},
    lifecycle::RemovedComponents,
    query::{Added, Changed, Has, Or},
    reflect::ReflectComponent,
    schedule::IntoScheduleConfigs,
    spawn::{SpawnRelated, SpawnableList},
    system::{Commands, Query},
};
use bevy_input_focus::tab_navigation::TabIndex;
use bevy_picking::{PickingSystems, hover::Hovered};
use bevy_reflect::{Reflect, prelude::ReflectDefault};
use bevy_scene::prelude::*;
use bevy_text::FontWeight;
use bevy_ui::{AlignItems, InteractionDisabled, JustifyContent, Node, Pressed, UiRect, px};
use bevy_ui_widgets::Button;

use crate::{
    constants::{fonts, size},
    cursor::EntityCursor,
    focus::FocusIndicator,
    font_styles::InheritableFont,
    rounded_corners::RoundedCorners,
    theme::{InheritableThemeTextColor, ThemeBackgroundColor},
    tokens,
};

/// Button color variants and styling marker.
#[derive(Component, Default, Clone, Reflect, Debug, PartialEq, Eq)]
#[reflect(Component, Clone, Default)]
pub enum ButtonVariant {
    /// Standard button appearance.
    #[default]
    Normal,
    /// Prominent call-to-action appearance.
    Primary,
    /// Transparent until hovered or pressed.
    Plain,
}

/// Button widget, spawnable with optional [`FeathersButtonProps`].
///
/// # Emitted events
/// * [`bevy_ui_widgets::Activate`] when any of the following happens:
///     * the pointer is released while hovering over the button.
///     * the ENTER or SPACE key is pressed while the button has keyboard focus.
///
///  These events can be disabled by adding an [`bevy_ui::InteractionDisabled`] component to the entity
#[derive(SceneComponent, Default, Clone)]
#[scene(FeathersButtonProps)]
#[derive(Reflect)]
#[reflect(Component, Clone, Default)]
pub struct FeathersButton;

/// Properties for a [`FeathersButton`] scene.
pub struct FeathersButtonProps {
    /// Label entities, arranged in a horizontal flexbox.
    pub caption: Box<dyn SceneList>,
    /// Color variant for the button.
    pub variant: ButtonVariant,
    /// Rounded corners options
    pub corners: RoundedCorners,
}

impl Default for FeathersButtonProps {
    fn default() -> Self {
        Self {
            caption: Box::new(bsn_list!()),
            variant: ButtonVariant::default(),
            corners: Default::default(),
        }
    }
}

impl FeathersButton {
    fn scene(props: FeathersButtonProps) -> impl Scene {
        bsn! {
            Node {
                height: size::ROW_HEIGHT,
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                padding: UiRect::horizontal(px(8)),
                border_radius: {props.corners.to_border_radius(4.0)},
            }
            Button
            template_value(props.variant)
            Hovered
            EntityCursor::System(bevy_window::SystemCursorIcon::Pointer)
            TabIndex(0)
            FocusIndicator
            ThemeBackgroundColor(tokens::BUTTON_BG)
            InheritableThemeTextColor(tokens::BUTTON_TEXT)
            InheritableFont {
                font: fonts::REGULAR,
                font_size: size::MEDIUM_FONT,
                weight: FontWeight::NORMAL,
            }
            Children [
                {props.caption}
            ]
        }
    }
}

/// Smaller button for embedding in panel headers.
///
/// Spawnable with optional [`FeathersButtonProps`].
///
/// # Emitted events
/// * [`bevy_ui_widgets::Activate`] when any of the following happens:
///     * the pointer is released while hovering over the button.
///     * the ENTER or SPACE key is pressed while the button has keyboard focus.
///
///  These events can be disabled by adding an [`bevy_ui::InteractionDisabled`] component to the entity
#[derive(SceneComponent, Default, Clone)]
#[scene(FeathersButtonProps)]
#[derive(Reflect)]
#[reflect(Component, Clone, Default)]
pub struct FeathersToolButton;

impl FeathersToolButton {
    fn scene(props: FeathersButtonProps) -> impl Scene {
        bsn! {
            @FeathersButton {
                @caption: {props.caption},
                @variant: {props.variant},
                @corners: {props.corners}
            }
            Node {
                padding: UiRect::horizontal(px(4)),
                min_width: size::ROW_HEIGHT,
            }
        }
    }
}

/// Parameters for the [`button_bundle`] template.
#[derive(Default)]
pub struct ButtonBundleProps {
    /// Color variant for the button.
    pub variant: ButtonVariant,
    /// Rounded corners options
    pub corners: RoundedCorners,
}

/// Spawns a button.
///
/// `props` supplies construction properties.
///
/// # Emitted events
/// * [`bevy_ui_widgets::Activate`] when any of the following happens:
///     * the pointer is released while hovering over the button.
///     * the ENTER or SPACE key is pressed while the button has keyboard focus.
///
///  These events can be disabled by adding an [`bevy_ui::InteractionDisabled`] component to the entity
#[deprecated(since = "0.19.0", note = "Use the button() BSN function")]
pub fn button_bundle<C: SpawnableList<ChildOf> + Send + Sync + 'static, B: Bundle>(
    props: ButtonBundleProps,
    overrides: B,
    children: C,
) -> impl Bundle {
    (
        Node {
            height: size::ROW_HEIGHT,
            justify_content: JustifyContent::Center,
            align_items: AlignItems::Center,
            padding: UiRect::horizontal(px(8)),
            flex_grow: 1.0,
            border_radius: props.corners.to_border_radius(4.0),
            ..Default::default()
        },
        Button,
        props.variant,
        Hovered::default(),
        EntityCursor::System(bevy_window::SystemCursorIcon::Pointer),
        TabIndex(0),
        FocusIndicator,
        ThemeBackgroundColor(tokens::BUTTON_BG),
        InheritableThemeTextColor(tokens::BUTTON_TEXT),
        InheritableFont {
            font_size: size::MEDIUM_FONT,
            weight: FontWeight::NORMAL,
            ..Default::default()
        },
        overrides,
        Children::spawn(children),
    )
}
fn update_button_styles(
    q_buttons: Query<
        (
            Entity,
            &ButtonVariant,
            Has<InteractionDisabled>,
            Has<Pressed>,
            &Hovered,
            &ThemeBackgroundColor,
            &InheritableThemeTextColor,
        ),
        Or<(
            Changed<Hovered>,
            Changed<ButtonVariant>,
            Added<Pressed>,
            Added<InteractionDisabled>,
        )>,
    >,
    mut commands: Commands,
) {
    for (button_ent, variant, disabled, pressed, hovered, bg_color, font_color) in q_buttons.iter()
    {
        set_button_styles(
            button_ent,
            variant,
            disabled,
            pressed,
            hovered.0,
            bg_color,
            font_color,
            &mut commands,
        );
    }
}

fn update_button_styles_remove(
    q_buttons: Query<(
        Entity,
        &ButtonVariant,
        Has<InteractionDisabled>,
        Has<Pressed>,
        &Hovered,
        &ThemeBackgroundColor,
        &InheritableThemeTextColor,
    )>,
    mut removed_disabled: RemovedComponents<InteractionDisabled>,
    mut removed_pressed: RemovedComponents<Pressed>,
    mut commands: Commands,
) {
    removed_disabled
        .read()
        .chain(removed_pressed.read())
        .for_each(|ent| {
            if let Ok((button_ent, variant, disabled, pressed, hovered, bg_color, font_color)) =
                q_buttons.get(ent)
            {
                set_button_styles(
                    button_ent,
                    variant,
                    disabled,
                    pressed,
                    hovered.0,
                    bg_color,
                    font_color,
                    &mut commands,
                );
            }
        });
}

fn set_button_styles(
    button_ent: Entity,
    variant: &ButtonVariant,
    disabled: bool,
    pressed: bool,
    hovered: bool,
    bg_color: &ThemeBackgroundColor,
    font_color: &InheritableThemeTextColor,
    commands: &mut Commands,
) {
    let bg_token = match (variant, disabled, pressed, hovered) {
        (ButtonVariant::Normal, true, _, _) => tokens::BUTTON_BG_DISABLED,
        (ButtonVariant::Normal, false, true, _) => tokens::BUTTON_BG_PRESSED,
        (ButtonVariant::Normal, false, false, true) => tokens::BUTTON_BG_HOVER,
        (ButtonVariant::Normal, false, false, false) => tokens::BUTTON_BG,
        (ButtonVariant::Primary, true, _, _) => tokens::BUTTON_PRIMARY_BG_DISABLED,
        (ButtonVariant::Primary, false, true, _) => tokens::BUTTON_PRIMARY_BG_PRESSED,
        (ButtonVariant::Primary, false, false, true) => tokens::BUTTON_PRIMARY_BG_HOVER,
        (ButtonVariant::Primary, false, false, false) => tokens::BUTTON_PRIMARY_BG,
        (ButtonVariant::Plain, true, _, _) => tokens::BUTTON_PLAIN_BG_DISABLED,
        (ButtonVariant::Plain, false, true, _) => tokens::BUTTON_PLAIN_BG_PRESSED,
        (ButtonVariant::Plain, false, false, true) => tokens::BUTTON_PLAIN_BG_HOVER,
        (ButtonVariant::Plain, false, false, false) => tokens::BUTTON_PLAIN_BG,
    };

    let font_color_token = match (variant, disabled) {
        (ButtonVariant::Primary, true) => tokens::BUTTON_PRIMARY_TEXT_DISABLED,
        (ButtonVariant::Primary, false) => tokens::BUTTON_PRIMARY_TEXT,
        (ButtonVariant::Normal | ButtonVariant::Plain, true) => tokens::BUTTON_TEXT_DISABLED,
        (ButtonVariant::Normal | ButtonVariant::Plain, false) => tokens::BUTTON_TEXT,
    };

    let cursor_shape = match disabled {
        true => bevy_window::SystemCursorIcon::NotAllowed,
        false => bevy_window::SystemCursorIcon::Pointer,
    };

    if bg_color.0 != bg_token {
        commands
            .entity(button_ent)
            .insert(ThemeBackgroundColor(bg_token));
    }

    if font_color.0 != font_color_token {
        commands
            .entity(button_ent)
            .insert(InheritableThemeTextColor(font_color_token));
    }

    commands
        .entity(button_ent)
        .insert(EntityCursor::System(cursor_shape));
}

/// Registers button styling systems.
pub struct ButtonPlugin;

impl Plugin for ButtonPlugin {
    fn build(&self, app: &mut bevy_app::App) {
        app.add_systems(
            PreUpdate,
            (update_button_styles, update_button_styles_remove).in_set(PickingSystems::Last),
        );
    }
}
