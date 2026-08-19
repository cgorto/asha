use bevy_app::{Plugin, PostUpdate};
use bevy_ecs::{
    bundle::Bundle,
    children,
    component::Component,
    entity::Entity,
    hierarchy::{ChildOf, Children},
    observer::On,
    query::{Changed, Has, Or, With},
    reflect::ReflectComponent,
    system::{Commands, Query, Res},
    template::FromTemplate,
};
use bevy_math::{Vec2, Vec3};
use bevy_picking::{
    Pickable,
    events::{Cancel, Drag, DragEnd, DragStart, Pointer, Press},
};
use bevy_reflect::{Reflect, prelude::ReflectDefault};
use bevy_scene::prelude::*;
use bevy_ui::{
    AlignSelf, BorderColor, BorderRadius, ComputedNode, ComputedUiRenderTargetInfo, Display,
    InteractionDisabled, Node, Outline, PositionType, UiGlobalTransform, UiRect, UiScale,
    UiTransform, Val2, percent, px,
};
use bevy_ui_widgets::ValueChange;

use ui_bridge::{ColorPlaneAxes, UiMaterialTag};

use crate::{cursor::EntityCursor, palette, theme::ThemeBackgroundColor, tokens};

/// Two-dimensional color-space picker, spawnable as a scene component.
///
/// Emits [`ValueChange<Vec2>`] with normalized x/y values. The [`Vec3`] value
/// supplied to the control uses z as the fixed channel for the background
/// gradient. Color-space conversion is left to callers.
#[derive(
    SceneComponent, FromTemplate, Debug, Reflect, Copy, PartialEq, Eq, Hash, Default, Clone,
)]
#[reflect(Component)]
#[require(ColorPlaneDragState)]
pub enum FeathersColorPlane {
    /// Show red on the horizontal axis and green on the vertical.
    RedGreen,
    /// Show red on the horizontal axis and blue on the vertical.
    RedBlue,
    /// Show green on the horizontal axis and blue on the vertical.
    GreenBlue,
    /// Show hue on the horizontal axis and saturation on the vertical.
    HueSaturation,
    /// Show hue on the horizontal axis and lightness on the vertical.
    #[default]
    HueLightness,
}

/// Selected x/y values and the gradient's fixed z channel.
///
/// The x/y values position the thumb; z supplies the fixed channel for the
/// background gradient.
#[derive(Component, Default, Clone, Reflect)]
#[reflect(Component, Clone, Default)]
pub struct ColorPlaneValue(pub Vec3);

/// Color-plane inner-element marker.
#[derive(Component, Default, Clone, Reflect)]
#[reflect(Component, Clone, Default)]
struct ColorPlaneInner;

/// Color-plane thumb marker.
#[derive(Component, Default, Clone, Reflect)]
#[reflect(Component, Clone, Default)]
struct ColorPlaneThumb;

/// Tracks color-plane dragging.
#[derive(Component, Default, Reflect)]
#[reflect(Component)]
struct ColorPlaneDragState(bool);

/// Maps widget axes to the bridge's material-tag representation.
fn plane_axes(plane: FeathersColorPlane) -> ColorPlaneAxes {
    match plane {
        FeathersColorPlane::RedGreen => ColorPlaneAxes::RedGreen,
        FeathersColorPlane::RedBlue => ColorPlaneAxes::RedBlue,
        FeathersColorPlane::GreenBlue => ColorPlaneAxes::GreenBlue,
        FeathersColorPlane::HueSaturation => ColorPlaneAxes::HueSaturation,
        FeathersColorPlane::HueLightness => ColorPlaneAxes::HueLightness,
    }
}

impl FeathersColorPlane {
    fn scene() -> impl Scene {
        bsn! {
            Node {
                display: Display::Flex,
                min_height: px(100.0),
                align_self: AlignSelf::Stretch,
                padding: UiRect::all(px(4)),
                border_radius: BorderRadius::all(px(5)),
            }
            ColorPlaneValue
            ThemeBackgroundColor(tokens::COLOR_PLANE_BG)
            EntityCursor::System(bevy_window::SystemCursorIcon::Crosshair)
            Children [(
                Node {
                    align_self: AlignSelf::Stretch,
                    flex_grow: 1.0,
                }
                ColorPlaneInner
                Children [(
                    Node {
                        position_type: PositionType::Absolute,
                        left: percent(0),
                        top: percent(0),
                        width: px(10),
                        height: px(10),
                        border: px(1),
                        border_radius: BorderRadius::MAX,
                    }
                    ColorPlaneThumb
                    BorderColor::all(palette::WHITE)
                    Outline {
                        width: px(1),
                        offset: px(0),
                        color: palette::BLACK
                    }
                    Pickable::IGNORE
                    UiTransform::from_translation(Val2::percent(-50., -50.),)
                )]
            )]
        }
    }
}

/// Spawns a two-dimensional color-space picker.
///
/// Emits [`ValueChange<Vec2>`] with normalized x/y values. The [`Vec3`] value
/// supplied to the control uses z as the fixed channel for the background
/// gradient. Color-space conversion is left to callers.
///
/// # Arguments
/// `overrides` augments the default plane components.
#[deprecated(since = "0.19.0", note = "Use the color_plane() BSN function")]
pub fn color_plane_bundle<B: Bundle>(plane: FeathersColorPlane, overrides: B) -> impl Bundle {
    (
        Node {
            display: Display::Flex,
            min_height: px(100.0),
            align_self: AlignSelf::Stretch,
            padding: UiRect::all(px(4)),
            border_radius: BorderRadius::all(px(5)),
            ..Default::default()
        },
        plane,
        ColorPlaneValue::default(),
        ThemeBackgroundColor(tokens::COLOR_PLANE_BG),
        EntityCursor::System(bevy_window::SystemCursorIcon::Crosshair),
        overrides,
        children![(
            Node {
                align_self: AlignSelf::Stretch,
                flex_grow: 1.0,
                ..Default::default()
            },
            ColorPlaneInner,
            children![(
                Node {
                    position_type: PositionType::Absolute,
                    left: percent(0),
                    top: percent(0),
                    width: px(10),
                    height: px(10),
                    border: UiRect::all(px(1)),
                    border_radius: BorderRadius::MAX,
                    ..Default::default()
                },
                ColorPlaneThumb,
                BorderColor::all(palette::WHITE),
                Outline {
                    width: px(1),
                    offset: px(0),
                    color: palette::BLACK
                },
                Pickable::IGNORE,
                UiTransform::from_translation(Val2::new(percent(-50), percent(-50),))
            )],
        ),],
    )
}

fn update_plane_color(
    q_color_plane: Query<
        (Entity, &FeathersColorPlane, &ColorPlaneValue),
        Or<(Changed<FeathersColorPlane>, Changed<ColorPlaneValue>)>,
    >,
    q_children: Query<&Children>,
    mut q_node: Query<&mut Node>,
    mut commands: Commands,
) {
    for (plane_ent, plane, plane_value) in q_color_plane.iter() {
        let Ok(children) = q_children.get(plane_ent) else {
            continue;
        };
        let Some(inner_ent) = children.first() else {
            continue;
        };

        // Update the material tag in place.
        commands
            .entity(*inner_ent)
            .insert(UiMaterialTag::color_plane(
                plane_axes(*plane),
                plane_value.0.z,
            ));

        let Ok(children_inner) = q_children.get(*inner_ent) else {
            continue;
        };
        let Some(thumb_ent) = children_inner.first() else {
            continue;
        };

        let Ok(mut thumb_node) = q_node.get_mut(*thumb_ent) else {
            continue;
        };

        thumb_node.left = percent(plane_value.0.x * 100.0);
        thumb_node.top = percent(plane_value.0.y * 100.0);
    }
}

fn emit_color_plane_value_change(
    commands: &mut Commands,
    source: Entity,
    node: &ComputedNode,
    node_target: &ComputedUiRenderTargetInfo,
    transform: &UiGlobalTransform,
    pointer_position: Vec2,
    ui_scale: f32,
    is_final: bool,
) {
    let Some(pos) = node.normalize_point(
        *transform,
        pointer_position * node_target.scale_factor() / ui_scale,
    ) else {
        return;
    };

    commands.trigger(ValueChange {
        source,
        value: (pos + Vec2::splat(0.5)).clamp(Vec2::ZERO, Vec2::ONE),
        is_final,
    });
}

fn on_pointer_press(
    mut press: On<Pointer<Press>>,
    q_color_planes: Query<Has<InteractionDisabled>, With<FeathersColorPlane>>,
    q_color_plane_inner: Query<
        (
            &ComputedNode,
            &ComputedUiRenderTargetInfo,
            &UiGlobalTransform,
            &ChildOf,
        ),
        With<ColorPlaneInner>,
    >,
    ui_scale: Res<UiScale>,
    mut commands: Commands,
) {
    if let Ok((node, node_target, transform, parent)) = q_color_plane_inner.get(press.entity)
        && let Ok(disabled) = q_color_planes.get(parent.0)
    {
        press.propagate(false);
        if !disabled {
            emit_color_plane_value_change(
                &mut commands,
                parent.0,
                node,
                node_target,
                transform,
                press.pointer_location.position,
                ui_scale.0,
                false,
            );
        }
    }
}

fn on_drag_start(
    mut drag_start: On<Pointer<DragStart>>,
    mut q_color_planes: Query<
        (&mut ColorPlaneDragState, Has<InteractionDisabled>),
        With<FeathersColorPlane>,
    >,
    q_color_plane_inner: Query<&ChildOf, With<ColorPlaneInner>>,
) {
    if let Ok(parent) = q_color_plane_inner.get(drag_start.entity)
        && let Ok((mut state, disabled)) = q_color_planes.get_mut(parent.0)
    {
        drag_start.propagate(false);
        if !disabled {
            state.0 = true;
        }
    }
}

fn on_drag(
    mut drag: On<Pointer<Drag>>,
    q_color_planes: Query<
        (&ColorPlaneDragState, Has<InteractionDisabled>),
        With<FeathersColorPlane>,
    >,
    q_color_plane_inner: Query<
        (
            &ComputedNode,
            &ComputedUiRenderTargetInfo,
            &UiGlobalTransform,
            &ChildOf,
        ),
        With<ColorPlaneInner>,
    >,
    ui_scale: Res<UiScale>,
    mut commands: Commands,
) {
    if let Ok((node, node_target, transform, parent)) = q_color_plane_inner.get(drag.entity)
        && let Ok((state, disabled)) = q_color_planes.get(parent.0)
    {
        drag.propagate(false);
        if state.0 && !disabled {
            emit_color_plane_value_change(
                &mut commands,
                parent.0,
                node,
                node_target,
                transform,
                drag.pointer_location.position,
                ui_scale.0,
                false,
            );
        }
    }
}

fn on_drag_end(
    mut drag_end: On<Pointer<DragEnd>>,
    mut q_color_planes: Query<
        (&mut ColorPlaneDragState, Has<InteractionDisabled>),
        With<FeathersColorPlane>,
    >,
    q_color_plane_inner: Query<
        (
            &ComputedNode,
            &ComputedUiRenderTargetInfo,
            &UiGlobalTransform,
            &ChildOf,
        ),
        With<ColorPlaneInner>,
    >,
    ui_scale: Res<UiScale>,
    mut commands: Commands,
) {
    if let Ok((node, node_target, transform, parent)) = q_color_plane_inner.get(drag_end.entity)
        && let Ok((mut state, disabled)) = q_color_planes.get_mut(parent.0)
    {
        drag_end.propagate(false);
        if state.0 && !disabled {
            emit_color_plane_value_change(
                &mut commands,
                parent.0,
                node,
                node_target,
                transform,
                drag_end.pointer_location.position,
                ui_scale.0,
                true,
            );
        }
        state.0 = false;
    }
}

fn on_drag_cancel(
    drag_cancel: On<Pointer<Cancel>>,
    mut q_color_planes: Query<&mut ColorPlaneDragState, With<FeathersColorPlane>>,
    q_color_plane_inner: Query<&ChildOf, With<ColorPlaneInner>>,
) {
    if let Ok(parent) = q_color_plane_inner.get(drag_cancel.entity)
        && let Ok(mut state) = q_color_planes.get_mut(parent.0)
    {
        state.0 = false;
    }
}

/// Registers color-plane observers.
pub struct ColorPlanePlugin;

impl Plugin for ColorPlanePlugin {
    fn build(&self, app: &mut bevy_app::App) {
        app.add_systems(PostUpdate, update_plane_color);
        app.add_observer(on_pointer_press)
            .add_observer(on_drag_start)
            .add_observer(on_drag)
            .add_observer(on_drag_end)
            .add_observer(on_drag_cancel);
    }
}
