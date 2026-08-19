//! Bevy UI-to-GPU bridge.
//!
//! Walks laid-out UI, sorts painter's order, and authors `abi_ui` streams.

mod bridge;
mod camera;
mod glyphs;
mod gradient;
mod icons;
mod material;
mod paint;

pub use bridge::{AshaRenderPluginExt, UiBridge};
pub use glyphs::GlyphOutlineProvider;
pub use icons::{ICON_PATHS, IconRegistry, IconUploadPayload, IconUploadQueue};
pub use material::{ColorPlaneAxes, UiMaterialTag};
pub use paint::{
    TextPaintList, TextRunBatch, UiBatch, UiBridgePlugin, UiPaintList, UiShadowBatch,
    stack_z_offsets,
};
