//! Verifies a bindless storage-image round trip: a texture aliases
//! caller-owned memory, its RW descriptor is written at an arbitrary nonzero
//! slot in the caller-owned set-1 heap, and the shader writes through that
//! descriptor for exact per-pixel readback. Slot zero remains the null sentinel.

use abi_core::ImageGradientData;
use asha_assets::load_spv;
use gpu::{
    AllocationType, Gpu, GpuPtr, HazardFlags, Memory, Queue, Stage, TextureDesc, TextureFormat,
    TextureViewDesc, UsageFlags,
};

#[test]
fn bindless_image_roundtrip() {
    const W: u32 = 64;
    const H: u32 = 64;
    const TEX_IDX: u32 = 5;

    let gpu = Gpu::new(true).expect("vulkan init");
    let shader = gpu.shader_create_compute(&load_spv("image_gradient"), 8, 8, 1, "image_gradient");

    // Texture aliases our own GPU-only allocation.
    let desc = TextureDesc {
        dimensions: [W, H, 1],
        format: TextureFormat::Rgba32Float,
        usage: UsageFlags::STORAGE | UsageFlags::TRANSFER_SRC,
        ..Default::default()
    };
    let (size, align) = gpu.texture_size_and_align(desc);
    let storage = gpu.mem_alloc_raw(size, 1, align, Memory::Gpu, AllocationType::Default);
    let texture = gpu.texture_create(desc, storage, Queue::Main, None);

    let desc_size = gpu.texture_rw_view_descriptor_size() as u64;
    let heap = gpu.mem_alloc_raw(
        desc_size,
        64,
        256,
        Memory::Default,
        AllocationType::Descriptors,
    );
    // Set 1 RW heap: use a nonzero arbitrary slot; zero is the null sentinel.
    let rw_desc = gpu.texture_rw_view_descriptor(texture, TextureViewDesc::default());
    gpu.set_texture_rw_desc(heap, TEX_IDX, rw_desc);

    // Dispatch data and readback target.
    let data = gpu.alloc::<ImageGradientData>(Memory::Default);
    unsafe {
        *data.cpu = ImageGradientData {
            dst_texture: TEX_IDX,
            width: W,
            height: H,
        };
    }
    let readback = gpu.alloc_slice::<f32>((W * H * 4) as u64, Memory::Readback);

    let cb = gpu.commands_begin(Queue::Main);
    gpu.cmd_set_desc_heap(cb, GpuPtr::null(), heap.gpu, GpuPtr::null());
    gpu.cmd_set_compute_shader(cb, shader);
    gpu.cmd_dispatch(cb, data.gpu, W.div_ceil(8), H.div_ceil(8), 1);
    gpu.cmd_barrier(cb, Stage::Compute, Stage::Transfer, HazardFlags::empty());
    gpu.cmd_copy_texture_to_buffer(cb, texture, readback.cast());
    gpu.queue_submit(Queue::Main, &[cb]);
    gpu.queue_wait_idle(Queue::Main);

    unsafe {
        for y in 0..H as usize {
            for x in 0..W as usize {
                let px = readback.cpu.add((y * W as usize + x) * 4);
                let expected = [x as f32 / W as f32, y as f32 / H as f32, 0.25f32, 1.0f32];
                for c in 0..4 {
                    assert_eq!(*px.add(c), expected[c], "pixel ({x},{y}) channel {c}");
                }
            }
        }
    }

    gpu.texture_destroy(texture);
    gpu.shader_destroy(shader);
    gpu.free(data);
    gpu.free(readback);
    gpu.mem_free_raw(heap);
    gpu.mem_free_raw(storage);
}
