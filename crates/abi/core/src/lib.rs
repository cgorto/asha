//! Shared host/GPU ABI types, camera math, and dispatch data.
//!
//! Shared definitions compile natively and for SPIR-V. GPU-facing types use
//! `#[repr(C)]`, remain allocation-free, and preserve host/shader layout.

#![cfg_attr(target_arch = "spirv", no_std)]

mod kernels;
mod oct;
mod view;

pub use kernels::{
    DepthTriangleData, FillData, ImageGradientData, ReduceSingleDispatchData, TriangleData,
    WriteDrawData,
};
pub use oct::{oct_decode, oct_encode};
pub use view::{HIT_T_MISS, View, hardware_depth, ray_direction};

/// The attribute stanza for GPU-shared structs, spelled once.
pub use abi_macros::gpu_data;

/// The shared math vocabulary, both machines.
pub use glam;

/// The typed device address, from the gpu repo's `gpu-ptr` crate.
/// Re-exported so shader and host code keep one spelling.
pub use gpu_ptr::GpuPtr;

/// The 24-byte graphics push-constant block: `cmd_draw_*` writes `vert`,
/// `frag`, and `indirect` at offsets 0, 8, and 16. Graphics shaders must use
/// these typed accessors; declaring a direct `GpuPtr<T>` reads offset 0 and
/// silently selects the vertex slot. Compute push constants are one pointer at
/// offset zero.
#[gpu_data]
pub struct GraphicsPush {
    vert: GpuPtr<u8>,
    frag: GpuPtr<u8>,
    indirect: GpuPtr<u8>,
}

impl GraphicsPush {
    pub fn vert<T>(&self) -> GpuPtr<T> {
        self.vert.cast()
    }
    pub fn frag<T>(&self) -> GpuPtr<T> {
        self.frag.cast()
    }
    pub fn indirect<T>(&self) -> GpuPtr<T> {
        self.indirect.cast()
    }
}

const _: () = assert!(core::mem::size_of::<GraphicsPush>() == 24);

/// Indexed indirect draw arguments, GPU-writable: matches
/// VkDrawIndexedIndirectCommand (and gpu::DrawIndexedIndirectCommand).
#[gpu_data]
pub struct DrawIndexedIndirectCommand {
    pub index_count: u32,
    pub instance_count: u32,
    pub first_index: u32,
    pub vertex_offset: i32,
    pub first_instance: u32,
}
