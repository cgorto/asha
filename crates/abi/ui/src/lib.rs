//! UI GPU ABI: the quad vertex contract with its SDF/color math (ported
//! faithfully from bevy_ui_render), and the buffer-backed Slug text
//! vocabulary.
//!
//! Same two-target rules as `abi-core`: everything GPU-legal,
//! no_std-compatible, allocation-free.

#![cfg_attr(target_arch = "spirv", no_std)]

mod text;
mod ui;

pub use abi_core::{GpuPtr, gpu_data};
pub use text::*;
pub use ui::*;
