//! Mesh-domain GPU ABI: the scene/draw plumbing for the single geometry
//! path (meshlets → cluster cull → multi-draw indirect), the skinning
//! vocabulary, the lattice deformers, and host-side transform composition.
//!
//! Same two-target rules as `abi-core`: everything GPU-legal,
//! no_std-compatible, allocation-free. Surface-shading math lives in
//! `abi-light`; the data it needs is here because `MeshFrameData` embeds it.

#![cfg_attr(target_arch = "spirv", no_std)]

mod deform;
mod scene;
mod skin;
#[cfg(not(target_arch = "spirv"))]
mod transform;

pub use abi_core::{GpuPtr, gpu_data};
pub use deform::*;
pub use scene::*;
pub use skin::*;
#[cfg(not(target_arch = "spirv"))]
pub use transform::*;
