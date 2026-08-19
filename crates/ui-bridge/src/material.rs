//! UI material tags packed into the `abi_ui` vertex stream.
//!
//! Modes and parameters remain CPU-visible; shaders decode the packed data.

use abi_ui::{
    UI_MODE_ALPHA_PATTERN, UI_MODE_COLOR_PLANE, UI_PLANE_GB, UI_PLANE_HL, UI_PLANE_HS, UI_PLANE_RB,
    UI_PLANE_RG, UiMaterialData,
};
use bevy::prelude::Component;

/// Color axes displayed by a material color-plane quad.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ColorPlaneAxes {
    /// Red (uv.x) / green (uv.y); the fixed channel is blue.
    RedGreen,
    /// Red (uv.x) / blue (uv.y); the fixed channel is green.
    RedBlue,
    /// Green (uv.x) / blue (uv.y); the fixed channel is red.
    GreenBlue,
    /// Hue (uv.x) / saturation; the fixed channel is lightness.
    HueSaturation,
    /// Hue (uv.x) / lightness; the fixed channel is saturation.
    #[default]
    HueLightness,
}

impl ColorPlaneAxes {
    fn variant(self) -> u32 {
        match self {
            Self::RedGreen => UI_PLANE_RG,
            Self::RedBlue => UI_PLANE_RB,
            Self::GreenBlue => UI_PLANE_GB,
            Self::HueSaturation => UI_PLANE_HS,
            Self::HueLightness => UI_PLANE_HL,
        }
    }
}

/// Marks a UI node for material shading.
#[derive(Component, Debug, Clone, Copy)]
pub struct UiMaterialTag {
    pub(crate) mode: u32,
    pub(crate) data: UiMaterialData,
}

impl UiMaterialTag {
    /// Returns a checkerboard alpha-pattern material.
    pub fn alpha_pattern() -> Self {
        Self {
            mode: UI_MODE_ALPHA_PATTERN,
            data: UiMaterialData {
                variant: 0,
                fixed_channel: 0.0,
                _pad0: [0; 2],
            },
        }
    }

    /// Returns a color-plane material with one fixed channel.
    pub fn color_plane(axes: ColorPlaneAxes, fixed_channel: f32) -> Self {
        Self {
            mode: UI_MODE_COLOR_PLANE,
            data: UiMaterialData {
                variant: axes.variant(),
                fixed_channel,
                _pad0: [0; 2],
            },
        }
    }
}
