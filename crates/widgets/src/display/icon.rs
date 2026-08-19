//! BSN scene for displaying [`ImageNode`]s.
use bevy_scene::{Scene, bsn};
use bevy_ui::{Node, px, widget::ImageNode};

/// Displays an icon.
pub fn icon(image: &'static str) -> impl Scene {
    bsn! {
        Node {
            height: px(14),
        }
        ImageNode {
            image: image
        }
    }
}
