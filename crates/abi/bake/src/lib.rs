//! Baked debug-visualization textures: the generator vocabulary, compiled
//! for both machines so a host can evaluate the same texel on the CPU and
//! compare.
//!
//! Same two-target rules as `abi-core`: everything GPU-legal,
//! no_std-compatible, allocation-free.

#![cfg_attr(target_arch = "spirv", no_std)]

mod bake;

pub use abi_core::{GpuPtr, gpu_data};
pub use bake::*;
