//! Editor gizmo furniture: the instanced SDF pass's ABI and shape math.
//!
//! Same two-target rules as `abi-core`: everything GPU-legal,
//! no_std-compatible, allocation-free.

#![cfg_attr(target_arch = "spirv", no_std)]

mod gizmo;

pub use abi_core::{GpuPtr, gpu_data};
pub use gizmo::*;
