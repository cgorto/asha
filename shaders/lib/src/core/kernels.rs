//! Utility kernels for validating core GPU primitives.
//! Dispatch data lives in `abi_core`.

use abi_core::{
    DepthTriangleData, DrawIndexedIndirectCommand, FillData, GpuPtr, GraphicsPush,
    ImageGradientData, TriangleData, WriteDrawData,
};
use glam::{UVec2, UVec3, Vec4};
use spirv_std::spirv;
use spirv_std::{Image, RuntimeArray};

/// Validates push-constant `GpuPtr` access to dispatch data.
#[spirv(compute(threads(64)))]
pub fn fill(
    #[spirv(push_constant)] data_ptr: &GpuPtr<FillData>,
    #[spirv(global_invocation_id)] gid: UVec3,
) {
    if gid.x < data_ptr.count {
        let mut dst = data_ptr.dst;
        dst[gid.x] = data_ptr.value;
    }
}

/// Writes a gradient through the bindless storage-image heap.
/// Descriptor set one contains storage images.
#[spirv(compute(threads(8, 8)))]
pub fn image_gradient(
    #[spirv(push_constant)] data_ptr: &GpuPtr<ImageGradientData>,
    #[spirv(descriptor_set = 1, binding = 0)] textures_rw: &RuntimeArray<
        Image!(2D, format = rgba32f, sampled = false),
    >,
    #[spirv(global_invocation_id)] gid: UVec3,
) {
    let data = &**data_ptr;
    if gid.x >= data.width || gid.y >= data.height {
        return;
    }
    let color = Vec4::new(
        gid.x as f32 / data.width as f32,
        gid.y as f32 / data.height as f32,
        0.25,
        1.0,
    );
    unsafe {
        let image = textures_rw.index(data.dst_texture as usize);
        image.write(UVec2::new(gid.x, gid.y), color);
    }
}

#[spirv(vertex)]
pub fn triangle_vert(
    #[spirv(push_constant)] push: &GraphicsPush,
    #[spirv(vertex_index)] vert_id: i32,
    #[spirv(position)] out_pos: &mut Vec4,
    out_color: &mut Vec4,
) {
    let data = push.vert::<TriangleData>();
    let p = data.positions[vert_id];
    *out_pos = Vec4::new(p[0], p[1], 0.0, 1.0);
    *out_color = Vec4::from_array(data.colors[vert_id]);
}

/// Emits a triangle with unit homogeneous position and white tint.
#[spirv(vertex)]
pub fn depth_triangle_vert(
    #[spirv(push_constant)] push: &GraphicsPush,
    #[spirv(vertex_index)] vert_id: i32,
    #[spirv(position)] out_pos: &mut Vec4,
    out_color: &mut Vec4,
) {
    let data = push.vert::<DepthTriangleData>();
    let p = data.positions[vert_id];
    *out_pos = Vec4::new(p[0], p[1], p[2], 1.0);
    *out_color = Vec4::ONE;
}

#[spirv(fragment)]
pub fn triangle_frag(
    #[spirv(push_constant)] push: &GraphicsPush,
    color: Vec4,
    out_color: &mut Vec4,
) {
    let data = push.frag::<TriangleData>();
    *out_color = color * Vec4::from_array(*data.tint);
}

/// Demonstrates GPU-driven indirect drawing from allocated buffers.
#[spirv(compute(threads(1)))]
pub fn write_draw(#[spirv(push_constant)] data_ptr: &GpuPtr<WriteDrawData>) {
    let mut cmds = data_ptr.cmds;
    let mut count = data_ptr.count;
    cmds[0u32] = DrawIndexedIndirectCommand {
        index_count: 3,
        instance_count: 1,
        first_index: 0,
        vertex_offset: 0,
        first_instance: 0,
    };
    *count = 1;
}
