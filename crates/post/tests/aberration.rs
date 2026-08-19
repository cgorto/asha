//! Headless GPU coverage for chromatic aberration.
//!
//! The tests compare displaced channels with a CPU reference, verify uniform
//! fields remain unchanged, and preserve green as the reference plane.

mod common;

use abi_core::glam::{UVec2, Vec2, Vec3};
use abi_post::ca_offset;
use common::{SIZE, TestAlloc, bilinear, texel};
use gpu::{
    Gpu, HazardFlags, Memory, Queue, SampledSlot, Stage, TextureDesc, TextureFormat,
    TextureViewDesc, UsageFlags,
};
use post::AberrationPass;

/// Smooth, distinct per-channel gradients for sampling tests.
fn input_pixel(x: u32, y: u32) -> Vec3 {
    let u = x as f32 / SIZE as f32;
    let v = y as f32 / SIZE as f32;
    Vec3::new(2.0 * u, 1.5 * v, 0.75 * (u + v))
}

/// CPU reference for `aberration_frag`.
fn fringed(uv: Vec2, strength: f32, pixel: impl Fn(u32, u32) -> Vec3 + Copy) -> Vec3 {
    let ca = ca_offset(uv, strength);
    Vec3::new(
        bilinear(pixel, uv + ca).x,
        bilinear(pixel, uv).y,
        bilinear(pixel, uv - ca).z,
    )
}

struct Harness {
    heap: gpu::HeapSlots,
    sampler: gpu::SamplerSlot,
    pass: Option<AberrationPass>,
}

fn harness(gpu: &Gpu) -> Harness {
    let mut heap = gpu.heap_slots_create(8, 2, 2);
    let sampler = heap.add_sampler(
        gpu,
        gpu.sampler_descriptor(gpu::SamplerDesc {
            address_mode_u: gpu::AddressMode::ClampToEdge,
            address_mode_v: gpu::AddressMode::ClampToEdge,
            address_mode_w: gpu::AddressMode::ClampToEdge,
            ..Default::default()
        }),
    );
    let mut pass = None;
    assert!(AberrationPass::ensure(
        &mut pass,
        gpu,
        &mut heap,
        UVec2::splat(SIZE)
    ));
    assert!(
        !AberrationPass::ensure(&mut pass, gpu, &mut heap, UVec2::splat(SIZE)),
        "same size must not rebuild"
    );
    Harness {
        heap,
        sampler,
        pass,
    }
}

/// Upload one input as a sampled Rgba32Float texture (exact values — only
/// the pass's own target quantizes to fp16).
fn upload_input(
    gpu: &Gpu,
    heap: &mut gpu::HeapSlots,
    pixel: impl Fn(u32, u32) -> Vec3,
) -> (gpu::OwnedTexture, SampledSlot, gpu::Ptr<[f32; 4]>) {
    let tex = gpu.texture_alloc_and_create(
        TextureDesc {
            dimensions: [SIZE, SIZE, 1],
            format: TextureFormat::Rgba32Float,
            usage: UsageFlags::SAMPLED | UsageFlags::TRANSFER_DST,
            ..Default::default()
        },
        Queue::Main,
        None,
    );
    let slot = heap.add_sampled(
        gpu,
        gpu.texture_view_descriptor(tex.texture, TextureViewDesc::default()),
    );
    let upload = gpu.alloc_slice::<[f32; 4]>((SIZE * SIZE) as u64, Memory::Default);
    // SAFETY: fresh host-visible allocation sized SIZE².
    unsafe {
        for y in 0..SIZE {
            for x in 0..SIZE {
                let p = pixel(x, y);
                *upload.cpu.add((y * SIZE + x) as usize) = [p.x, p.y, p.z, 1.0];
            }
        }
    }
    (tex, slot, upload)
}

#[test]
fn aberration_matches_cpu_twin_and_green_holds() {
    let gpu = Gpu::new(true).expect("vulkan init");
    let mut h = harness(&gpu);
    let (tex, slot, up) = upload_input(&gpu, &mut h.heap, input_pixel);
    // Use a displacement well above the comparison tolerance.
    let strength = 0.08f32;

    let mut fa = TestAlloc {
        gpu: &gpu,
        live: Vec::new(),
    };
    let read = gpu.alloc_slice::<[u16; 4]>((SIZE * SIZE) as u64, Memory::Readback);

    let cb = gpu.commands_begin(Queue::Main);
    gpu.cmd_copy_to_texture(cb, tex.texture, up.cast());
    gpu.cmd_barrier(
        cb,
        Stage::Transfer,
        Stage::FragmentShader,
        HazardFlags::empty(),
    );
    h.heap.bind(&gpu, cb);
    let pass = h.pass.as_ref().unwrap();
    let out = pass.record(&gpu, cb, &mut fa, slot, h.sampler, strength);
    assert_ne!(out.index(), slot.index(), "output is its own texture");
    gpu.cmd_barrier(
        cb,
        Stage::RasterColorOut,
        Stage::Transfer,
        HazardFlags::empty(),
    );
    gpu.cmd_copy_texture_to_buffer(cb, pass.texture(), read.cast());
    gpu.queue_submit(Queue::Main, &[cb]);
    gpu.queue_wait_idle(Queue::Main);

    // Compare CPU bilinear sampling with GPU output.
    let mut worst = 0.0f32;
    for y in 0..SIZE {
        for x in 0..SIZE {
            let uv = Vec2::new(
                (x as f32 + 0.5) / SIZE as f32,
                (y as f32 + 0.5) / SIZE as f32,
            );
            let want = fringed(uv, strength, input_pixel);
            let got = texel(read, x, y);
            let err = (got - want).abs().max_element();
            worst = worst.max(err);
            assert!(err < 5e-3, "pixel ({x},{y}): gpu {got} vs cpu {want}");
            // Green remains the reference plane.
            assert!(
                (got.y - input_pixel(x, y).y).abs() < 2e-3,
                "green moved at ({x},{y}): {} vs {}",
                got.y,
                input_pixel(x, y).y
            );
        }
    }
    println!("aberration twin: worst channel error {worst:.2e}");

    // Confirm the configured displacement is observable.
    let fringe_matters = (0..SIZE * SIZE).any(|i| {
        let (x, y) = (i % SIZE, i / SIZE);
        let uv = Vec2::new(
            (x as f32 + 0.5) / SIZE as f32,
            (y as f32 + 0.5) / SIZE as f32,
        );
        (fringed(uv, strength, input_pixel) - input_pixel(x, y))
            .abs()
            .max_element()
            > 5e-2
    });
    assert!(
        fringe_matters,
        "gradient too weak to distinguish the fringe"
    );

    fa.free();
    gpu.free(read);
    gpu.free(up);
    gpu.texture_free_and_destroy(tex);
    use gpu::pass::Pass;
    h.pass.take().unwrap().free(&gpu);
    h.heap.free(&gpu);
}

#[test]
fn aberration_uniform_field_is_fixed_point() {
    let gpu = Gpu::new(true).expect("vulkan init");
    let mut h = harness(&gpu);
    // The uniform field is exactly representable in fp16.
    let uniform = Vec3::new(2.0, 1.0, 0.5);
    let (tex, slot, up) = upload_input(&gpu, &mut h.heap, |_, _| uniform);

    let mut fa = TestAlloc {
        gpu: &gpu,
        live: Vec::new(),
    };
    let read = gpu.alloc_slice::<[u16; 4]>((SIZE * SIZE) as u64, Memory::Readback);

    let cb = gpu.commands_begin(Queue::Main);
    gpu.cmd_copy_to_texture(cb, tex.texture, up.cast());
    gpu.cmd_barrier(
        cb,
        Stage::Transfer,
        Stage::FragmentShader,
        HazardFlags::empty(),
    );
    h.heap.bind(&gpu, cb);
    let pass = h.pass.as_ref().unwrap();
    // Uniform fields remain unchanged despite extreme displacement.
    pass.record(&gpu, cb, &mut fa, slot, h.sampler, 0.5);
    gpu.cmd_barrier(
        cb,
        Stage::RasterColorOut,
        Stage::Transfer,
        HazardFlags::empty(),
    );
    gpu.cmd_copy_texture_to_buffer(cb, pass.texture(), read.cast());
    gpu.queue_submit(Queue::Main, &[cb]);
    gpu.queue_wait_idle(Queue::Main);

    for y in 0..SIZE {
        for x in 0..SIZE {
            assert_eq!(
                texel(read, x, y),
                uniform,
                "uniform field moved at ({x},{y})"
            );
        }
    }

    fa.free();
    gpu.free(read);
    gpu.free(up);
    gpu.texture_free_and_destroy(tex);
    use gpu::pass::Pass;
    h.pass.take().unwrap().free(&gpu);
    h.heap.free(&gpu);
}
