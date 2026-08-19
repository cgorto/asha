use bevy_ecs::hierarchy::Children;
use bevy_scene::{Scene, bsn};
use bevy_text::FontWeight;
use bevy_ui::{
    AlignItems, AlignSelf, Display, FlexDirection, JustifyContent, Node, PositionType, UiRect, px,
};

use crate::{
    constants::{fonts, size},
    font_styles::InheritableFont,
    rounded_corners::RoundedCorners,
    theme::{InheritableThemeTextColor, ThemeBackgroundColor, ThemeBorderColor},
    tokens,
};

/// Standard pane container.
pub fn pane() -> impl Scene {
    bsn! {
        Node {
            display: Display::Flex,
            flex_direction: FlexDirection::Column,
            align_items: AlignItems::Stretch,
        }
    }
}

/// Pane header.
pub fn pane_header() -> impl Scene {
    bsn! {
        Node {
            display: Display::Flex,
            flex_direction: FlexDirection::Row,
            align_items: AlignItems::Center,
            justify_content: JustifyContent::SpaceBetween,
            padding: px(6),
            border: UiRect {
                left: px(1),
                top: px(1),
                right: px(1),
            },
            min_height: size::HEADER_HEIGHT,
            column_gap: px(6),
            border_radius: {RoundedCorners::Top.to_border_radius(4.0)},
        }
        ThemeBackgroundColor(tokens::PANE_HEADER_BG)
        ThemeBorderColor(tokens::PANE_HEADER_BORDER)
        InheritableThemeTextColor(tokens::PANE_HEADER_TEXT)
        InheritableFont {
            font: fonts::REGULAR,
            font_size: size::MEDIUM_FONT,
            weight: FontWeight::NORMAL,
        }
    }
}

/// Header divider between widget groups.
pub fn pane_header_divider() -> impl Scene {
    bsn! {
        Node {
            width: px(1),
            align_self: AlignSelf::Stretch,
        }
        Children [(
            // Absolute positioning extends through header padding.
            Node {
                position_type: PositionType::Absolute,
                left: px(0),
                right: px(0),
                top: px(-6),
                bottom: px(-6),
            }
            ThemeBackgroundColor(tokens::PANE_HEADER_DIVIDER)
        )]
    }
}

/// Pane body.
pub fn pane_body() -> impl Scene {
    bsn! {
        Node {
            display: Display::Flex,
            flex_direction: FlexDirection::Column,
            row_gap: px(4),
            padding: px(6),
            border_radius: {RoundedCorners::Bottom.to_border_radius(4.0)}
        }
        ThemeBackgroundColor(tokens::PANE_BODY_BG)
    }
}
