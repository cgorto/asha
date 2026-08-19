//! BSN scenes for themed text labels.
use bevy_app::PropagateOver;
use bevy_scene::{Scene, bsn};
use bevy_text::{FontSourceTemplate, FontWeight, TextFont};
use bevy_ui::widget::Text;

use crate::{
    constants::{fonts, size},
    theme::ThemeTextColor,
    tokens,
};

/// Text label.
pub fn label(text: impl Into<String>) -> impl Scene {
    bsn! {
        Text(text)
        TextFont {
            font: FontSourceTemplate::Handle(fonts::REGULAR),
            font_size: size::MEDIUM_FONT,
            weight: FontWeight::NORMAL,
        }
        PropagateOver<TextFont>
        ThemeTextColor(tokens::TEXT_MAIN)
    }
}

/// Dimmed text label.
pub fn label_dim(text: impl Into<String>) -> impl Scene {
    bsn! {
        Text(text)
        TextFont {
            font: FontSourceTemplate::Handle(fonts::REGULAR),
            font_size: size::MEDIUM_FONT,
            weight: FontWeight::NORMAL,
        }
        PropagateOver<TextFont>
        ThemeTextColor(tokens::TEXT_DIM)
    }
}

/// Small label for field captions.
pub fn label_small(text: impl Into<String>) -> impl Scene {
    bsn! {
        Text(text)
        TextFont {
            font: FontSourceTemplate::Handle(fonts::REGULAR),
            font_size: size::EXTRA_SMALL_FONT,
            weight: FontWeight::NORMAL,
        }
        PropagateOver<TextFont>
        ThemeTextColor(tokens::TEXT_MAIN)
    }
}
