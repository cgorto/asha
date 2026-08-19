//! Render-free fork of `bevy_feathers` 0.19.0.
//!
//! Bevy render dependencies are removed; asha's UI pass draws these widgets.
//!
//! ## Fork deviations
//!
//! - Bevy dependencies remain pinned to `=0.19.0`; render dependencies are absent.
//! - Upstream WGSL shaders are omitted; their math is implemented in `abi_ui`.
//! - Embedded assets use the `embedded://widgets/` namespace.
//! - Optional `color-tools` replaces upstream material assets and specialization with
//!   `ui_bridge::UiMaterialTag`, consumed by the paint walker and `abi_ui`.
//!   It is optional and remains a one-directional dependency on `ui-bridge`.
//!
//! These styled widgets target editors and inspectors.
//!
//! Widgets should stop handled events from propagating to parent entities.
//! See [`EntityEvent`](bevy_ecs::event::EntityEvent) for propagation details.

extern crate alloc;

use bevy_app::{
    HierarchyPropagatePlugin, Plugin, PluginGroup, PluginGroupBuilder, PostUpdate, PropagateSet,
};
use bevy_asset::embedded_asset;
use bevy_ecs::{query::With, schedule::IntoScheduleConfigs};
use bevy_input_focus::tab_navigation::TabNavigationPlugin;
use bevy_text::{TextColor, TextFont};
use bevy_ui::UiSystems;

#[cfg(feature = "color-tools")]
use crate::alpha_pattern::AlphaPatternPlugin;
use crate::{
    controls::ControlsPlugin,
    cursor::{CursorIconPlugin, DefaultCursor, EntityCursor},
    theme::{ThemedText, UiTheme},
};

#[cfg(feature = "color-tools")]
mod alpha_pattern;
pub mod constants;
pub mod containers;
pub mod controls;
pub mod cursor;
pub mod dark_theme;
pub mod display;
pub mod focus;
pub mod font_styles;
pub mod palette;
pub mod rounded_corners;
pub mod theme;
pub mod tokens;

/// Registers theme, cursor, and control systems.
pub struct FeathersCorePlugin;

impl Plugin for FeathersCorePlugin {
    fn build(&self, app: &mut bevy_app::App) {
        app.init_resource::<UiTheme>();

        embedded_asset!(app, "assets/fonts/FiraSans-Bold.ttf");
        embedded_asset!(app, "assets/fonts/FiraSans-BoldItalic.ttf");
        embedded_asset!(app, "assets/fonts/FiraSans-Regular.ttf");
        embedded_asset!(app, "assets/fonts/FiraSans-Italic.ttf");
        embedded_asset!(app, "assets/fonts/FiraMono-Medium.ttf");

        embedded_asset!(app, "assets/icons/chevron-down.png");
        embedded_asset!(app, "assets/icons/chevron-right.png");
        embedded_asset!(app, "assets/icons/x.png");

        app.add_plugins((
            ControlsPlugin,
            CursorIconPlugin,
            HierarchyPropagatePlugin::<TextColor, With<ThemedText>>::new(PostUpdate),
            HierarchyPropagatePlugin::<TextFont, With<ThemedText>>::new(PostUpdate),
            #[cfg(feature = "color-tools")]
            AlphaPatternPlugin,
            focus::FocusOutlinesPlugin,
        ));

        // Propagate fonts before text measurement and rerender detection.
        app.configure_sets(
            PostUpdate,
            PropagateSet::<TextFont>::default().in_set(UiSystems::Propagate),
        );

        app.insert_resource(DefaultCursor(EntityCursor::System(
            bevy_window::SystemCursorIcon::Default,
        )));

        app.add_systems(PostUpdate, theme::update_theme)
            .add_observer(theme::on_changed_background)
            .add_observer(theme::on_changed_border)
            .add_observer(theme::on_changed_font_color)
            .add_observer(theme::on_changed_text_color)
            .add_observer(font_styles::on_changed_font);
    }
}

/// A plugin group that adds all dependencies for Feathers
pub struct FeathersPlugins;

impl PluginGroup for FeathersPlugins {
    fn build(self) -> PluginGroupBuilder {
        PluginGroupBuilder::start::<Self>()
            .add(TabNavigationPlugin)
            .add(FeathersCorePlugin)
    }
}
