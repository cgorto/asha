//! Interactive Feathers widgets.
#![expect(deprecated, reason = "deprecated control bundles are exported here")]

mod button;
mod checkbox;
#[cfg(feature = "color-tools")]
mod color_plane;
#[cfg(feature = "color-tools")]
mod color_slider;
#[cfg(feature = "color-tools")]
mod color_swatch;
mod disclosure_toggle;
mod listview;
mod menu;
mod number_input;
mod radio;
mod scrollbar;
mod slider;
mod text_input;
mod toggle_switch;
mod virtual_keyboard;

pub use button::*;
pub use checkbox::*;
#[cfg(feature = "color-tools")]
pub use color_plane::*;
#[cfg(feature = "color-tools")]
pub use color_slider::*;
#[cfg(feature = "color-tools")]
pub use color_swatch::*;
pub use disclosure_toggle::*;
pub use listview::*;
pub use menu::*;
pub use number_input::*;
pub use radio::*;
pub use scrollbar::*;
pub use slider::*;
pub use text_input::*;
pub use toggle_switch::*;
pub use virtual_keyboard::*;

use bevy_app::Plugin;

/// Registers all control plugins.
pub struct ControlsPlugin;

impl Plugin for ControlsPlugin {
    fn build(&self, app: &mut bevy_app::App) {
        app.add_plugins((
            ButtonPlugin,
            CheckboxPlugin,
            #[cfg(feature = "color-tools")]
            ColorPlanePlugin,
            #[cfg(feature = "color-tools")]
            ColorSliderPlugin,
            #[cfg(feature = "color-tools")]
            ColorSwatchPlugin,
            DisclosureTogglePlugin,
            ListViewPlugin,
            MenuPlugin,
            RadioPlugin,
            ScrollbarPlugin,
            SliderPlugin,
            TextInputPlugin,
            ToggleSwitchPlugin,
        ));
    }
}
