//! Dispatch data for the utility and smoke-test kernels: the small shaders
//! that prove a device primitive works before an engine system leans on it.

use crate::{DrawIndexedIndirectCommand, GpuPtr, gpu_data};

/// Dispatch data for the `fill` smoke-test shader: the first struct to cross
/// the host/GPU boundary through a single definition. The host suballocates
/// one of these, writes it, and pushes a `GpuPtr<FillData>` as the push
/// constant — the `cmd_dispatch(data.gpu, ...)` ABI.
#[gpu_data]
pub struct FillData {
    pub dst: GpuPtr<f32>,
    pub count: u32,
    pub value: f32,
}

const _: () = assert!(core::mem::size_of::<FillData>() == 16);

/// Dispatch data for the single-dispatch reduction kernel.
#[gpu_data]
pub struct ReduceSingleDispatchData {
    /// Input values, `group_count` x 64.
    pub values: GpuPtr<u32>,
    /// One partial sum per workgroup.
    pub partials: GpuPtr<u32>,
    /// Zero-initialized election counter.
    pub counter: GpuPtr<u32>,
    /// The grand total (wrapping), written by the elected last workgroup.
    pub result: GpuPtr<u32>,
    pub group_count: u32,
    pub _pad: u32,
}

const _: () = assert!(core::mem::size_of::<ReduceSingleDispatchData>() == 40);

/// Vertex-pulling triangle with pointer-backed positions, colors, and tint.
#[gpu_data]
pub struct TriangleData {
    pub positions: GpuPtr<[f32; 2]>,
    pub colors: GpuPtr<[f32; 4]>,
    pub tint: GpuPtr<[f32; 4]>,
}

/// Vertex-pulling triangle with a real z: positions are xyz + pad
/// (the 16-byte-padded layout mesh vertex pools use). Drives the depth
/// attachment tests; pairs with `triangle_frag` for tinting.
#[gpu_data]
pub struct DepthTriangleData {
    pub positions: GpuPtr<[f32; 4]>,
}

/// Dispatch data for the `image_gradient` smoke-test shader: writes a
/// gradient into a storage image reached through the bindless heap by index.
#[gpu_data]
pub struct ImageGradientData {
    pub dst_texture: u32,
    pub width: u32,
    pub height: u32,
}

/// Data for the write_draw smoke-test shader: GPU-driven draw submission
/// in miniature — the shader writes the draw command AND the draw count.
#[gpu_data]
pub struct WriteDrawData {
    pub cmds: GpuPtr<DrawIndexedIndirectCommand>,
    pub count: GpuPtr<u32>,
}
