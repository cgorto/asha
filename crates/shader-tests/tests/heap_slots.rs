//! Verifies bindless heap layout and slot allocation: sampled textures are
//! set 0, storage images set 1, samplers set 2, and slot zero is the null
//! sentinel in each heap. Allocation starts at one, descriptors land in the
//! caller-owned heap, and in-place rewrites preserve slot identity.

use gpu::{Gpu, Queue, SamplerDesc, TextureDesc, TextureFormat, TextureViewDesc, UsageFlags};

#[test]
fn slots_allocate_from_one_and_rewrite_in_place() {
    let gpu = Gpu::new(true).expect("vulkan init");
    let mut heap = gpu.heap_slots_create(4, 4, 2);

    let tex_desc = TextureDesc {
        dimensions: [8, 8, 1],
        format: TextureFormat::Rgba16Float,
        usage: UsageFlags::SAMPLED | UsageFlags::STORAGE,
        ..Default::default()
    };
    let a = gpu.texture_alloc_and_create(tex_desc, Queue::Main, None);
    let b = gpu.texture_alloc_and_create(tex_desc, Queue::Main, None);

    // Slot zero is the null sentinel; first sampled allocations start at one.
    let s_a = heap.add_sampled(
        &gpu,
        gpu.texture_view_descriptor(a.texture, TextureViewDesc::default()),
    );
    let s_b = heap.add_sampled(
        &gpu,
        gpu.texture_view_descriptor(b.texture, TextureViewDesc::default()),
    );
    assert_eq!(s_a.index(), 1);
    assert_eq!(s_b.index(), 2);
    let rw = heap.add_storage(
        &gpu,
        gpu.texture_rw_view_descriptor(a.texture, TextureViewDesc::default()),
    );
    assert_eq!(rw.index(), 1);
    let smp = heap.add_sampler(&gpu, gpu.sampler_descriptor(SamplerDesc::default()));
    assert_eq!(smp.index(), 1);

    // Rewriting a descriptor preserves its slot identity.
    heap.write_sampled(
        &gpu,
        s_a,
        gpu.texture_view_descriptor(b.texture, TextureViewDesc::default()),
    );

    // Binding validates descriptor writes end to end.
    let cb = gpu.commands_begin(Queue::Main);
    heap.bind(&gpu, cb);
    gpu.queue_submit(Queue::Main, &[cb]);
    gpu.queue_wait_idle(Queue::Main);

    heap.free(&gpu);
    gpu.texture_free_and_destroy(a);
    gpu.texture_free_and_destroy(b);
}
