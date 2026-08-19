//! Post-processing GPU ABI: the sensor/exposure/bloom/lens/feedback
//! vocabulary with its CPU-reference combination math, and the
//! display-space jump-flood outline pass.
//!
//! Same two-target rules as `abi-core`: everything GPU-legal,
//! no_std-compatible, allocation-free.

#![cfg_attr(target_arch = "spirv", no_std)]

mod outline;
mod post;

pub use abi_core::{GpuPtr, gpu_data};
pub use outline::*;
pub use post::*;
