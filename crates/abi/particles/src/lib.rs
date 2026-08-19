//! Particle GPU ABI: the fixed ring pool shared by the particles host passes and
//! the rust-gpu entry points. Deliberately no authoring vocabulary — specs,
//! curve baking, and resolver policy are host-only in `particles`.
//!
//! Same two-target rules as `abi-core`: everything GPU-legal,
//! no_std-compatible, allocation-free.

#![cfg_attr(target_arch = "spirv", no_std)]

mod particles;

pub use abi_core::{GpuPtr, gpu_data};
pub use particles::*;
