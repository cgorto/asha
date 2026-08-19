//! Bevy-free UI rendering: SDF shapes, gradients, borders, and icons.
//!
//! Shading uses `abi_ui`; `ui-bridge` supplies vertex streams.

mod pass;

pub use pass::{UiBatch, UiPass, UiPassTarget, UiScissor, UiShadowBatch};
