use bevy_scene::{Scene, bsn};
use bevy_ui::Node;

/// Invisible spacing node with positive `flex_grow`.
pub fn flex_spacer() -> impl Scene {
    bsn! {
        Node {
            flex_grow: 1.0,
        }
    }
}
