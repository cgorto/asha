//! Headless GPU coverage for feedback accumulation.
//!
//! Tests compare reprojection with a CPU reference and verify decay reaches
//! zero through the linear floor across ping-pong frames.

mod common;

use abi_core::glam::{UVec2, Vec2, Vec3};
use abi_post::{FeedbackCamera, feedback_combine, feedback_flow_uv};
use common::{SIZE, TestAlloc, bilinear, texel};
use gpu::{
    Gpu, HazardFlags, Memory, Queue, SampledSlot, Stage, TextureDesc, TextureFormat,
    TextureViewDesc, UsageFlags,
};
use post::FeedbackPass;

fn camera(forward: Vec3, right: Vec3, up: Vec3) -> FeedbackCamera {
    FeedbackCamera {
        forward: forward.to_array(),
        tan_half_fov: 0.55,
        right: right.to_array(),
        aspect: 1.0,
        up: up.to_array(),
        _pad: 0,
    }
}

/// Deterministic HDR-ish input frames.
fn input_pixel(frame: u32, x: u32, y: u32) -> Vec3 {
    Vec3::new(
        ((x * 3 + y * 7 + frame * 5) % 13) as f32 / 13.0 * 2.0,
        ((x * 5 + y + frame) % 11) as f32 / 11.0,
        ((x + y * 3 + frame * 2) % 7) as f32 / 7.0,
    )
}

struct Harness {
    heap: gpu::HeapSlots,
    sampler: gpu::SamplerSlot,
    pass: Option<FeedbackPass>,
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
    assert!(FeedbackPass::ensure(
        &mut pass,
        gpu,
        &mut heap,
        UVec2::splat(SIZE)
    ));
    assert!(
        !FeedbackPass::ensure(&mut pass, gpu, &mut heap, UVec2::splat(SIZE)),
        "same size must not rebuild"
    );
    Harness {
        heap,
        sampler,
        pass,
    }
}

/// Upload one input frame as a sampled Rgba32Float texture (exact values —
/// only the accumulator itself quantizes to fp16).
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
fn feedback_step_matches_cpu_twin() {
    let gpu = Gpu::new(true).expect("vulkan init");
    let mut h = harness(&gpu);
    let (tex1, slot1, up1) = upload_input(&gpu, &mut h.heap, |x, y| input_pixel(0, x, y));
    let (tex2, slot2, up2) = upload_input(&gpu, &mut h.heap, |x, y| input_pixel(1, x, y));

    // Frame 2 introduces camera motion for reprojection.
    let cam1 = camera(Vec3::NEG_Z, Vec3::X, Vec3::Y);
    let a = 0.04f32;
    let cam2 = camera(
        Vec3::new(-a.sin(), 0.0, -a.cos()),
        Vec3::new(a.cos(), 0.0, -a.sin()),
        Vec3::Y,
    );
    let (decay, floor, flow) = (0.6f32, 0.02f32, 1.5f32);

    let mut fa = TestAlloc {
        gpu: &gpu,
        live: Vec::new(),
    };
    let read1 = gpu.alloc_slice::<[u16; 4]>((SIZE * SIZE) as u64, Memory::Readback);
    let read2 = gpu.alloc_slice::<[u16; 4]>((SIZE * SIZE) as u64, Memory::Readback);

    let cb = gpu.commands_begin(Queue::Main);
    gpu.cmd_copy_to_texture(cb, tex1.texture, up1.cast());
    gpu.cmd_copy_to_texture(cb, tex2.texture, up2.cast());
    gpu.cmd_barrier(
        cb,
        Stage::Transfer,
        Stage::FragmentShader,
        HazardFlags::empty(),
    );
    h.heap.bind(&gpu, cb);
    // The first frame must ignore uninitialized history.
    let pass = h.pass.as_mut().unwrap();
    let out1 = pass.record(
        &gpu, cb, &mut fa, slot1, h.sampler, decay, floor, flow, cam1, cam1,
    );
    gpu.cmd_barrier(
        cb,
        Stage::RasterColorOut,
        Stage::Transfer,
        HazardFlags::empty(),
    );
    gpu.cmd_copy_texture_to_buffer(cb, pass.latest_texture(), read1.cast());
    // The second frame combines fresh input with reprojected history.
    let out2 = pass.record(
        &gpu, cb, &mut fa, slot2, h.sampler, decay, floor, flow, cam2, cam1,
    );
    gpu.cmd_barrier(
        cb,
        Stage::RasterColorOut,
        Stage::Transfer,
        HazardFlags::empty(),
    );
    gpu.cmd_copy_texture_to_buffer(cb, pass.latest_texture(), read2.cast());
    assert_ne!(out1.index(), out2.index(), "ping-pong must alternate slots");
    gpu.queue_submit(Queue::Main, &[cb]);
    gpu.queue_wait_idle(Queue::Main);

    // The first frame only quantizes the input to fp16.
    for y in 0..SIZE {
        for x in 0..SIZE {
            let got = texel(read1, x, y);
            let want = input_pixel(0, x, y);
            assert!(
                (got - want).abs().max_element() < 3e-3,
                "unprimed frame pixel ({x},{y}): {got} vs input {want}"
            );
        }
    }

    // Reference the decoded fp16 history sampled by the GPU.
    let mut worst = 0.0f32;
    for y in 0..SIZE {
        for x in 0..SIZE {
            let uv = Vec2::new(
                (x as f32 + 0.5) / SIZE as f32,
                (y as f32 + 0.5) / SIZE as f32,
            );
            let history_uv = feedback_flow_uv(&cam2, &cam1, uv, flow);
            let history = bilinear(|hx, hy| texel(read1, hx, hy), history_uv);
            let want = feedback_combine(input_pixel(1, x, y), history, decay, floor);
            let got = texel(read2, x, y);
            let err = (got - want).abs().max_element();
            worst = worst.max(err);
            assert!(
                err < 1e-2,
                "flowed step pixel ({x},{y}): gpu {got} vs cpu {want}"
            );
        }
    }
    println!("feedback step: worst channel error {worst:.2e}");

    // Confirm reprojection changes at least one sampled value.
    let flow_matters = (0..SIZE * SIZE).any(|i| {
        let (x, y) = (i % SIZE, i / SIZE);
        let uv = Vec2::new(
            (x as f32 + 0.5) / SIZE as f32,
            (y as f32 + 0.5) / SIZE as f32,
        );
        let flowed = bilinear(
            |hx, hy| texel(read1, hx, hy),
            feedback_flow_uv(&cam2, &cam1, uv, flow),
        );
        let passive = bilinear(|hx, hy| texel(read1, hx, hy), uv);
        let d = feedback_combine(input_pixel(1, x, y), flowed, decay, floor)
            - feedback_combine(input_pixel(1, x, y), passive, decay, floor);
        d.abs().max_element() > 5e-2
    });
    assert!(flow_matters, "camera motion too weak to distinguish flow");

    fa.free();
    gpu.free(read1);
    gpu.free(read2);
    gpu.free(up1);
    gpu.free(up2);
    gpu.texture_free_and_destroy(tex1);
    gpu.texture_free_and_destroy(tex2);
    use gpu::pass::Pass;
    h.pass.take().unwrap().free(&gpu);
    h.heap.free(&gpu);
}

#[test]
fn feedback_trail_decays_and_floor_drains_to_zero() {
    let gpu = Gpu::new(true).expect("vulkan init");
    let mut h = harness(&gpu);
    let bright = Vec3::new(2.0, 1.0, 0.5);
    let (btex, bslot, bup) = upload_input(&gpu, &mut h.heap, |_, _| bright);
    let (ktex, kslot, kup) = upload_input(&gpu, &mut h.heap, |_, _| Vec3::ZERO);

    let cam = camera(Vec3::NEG_Z, Vec3::X, Vec3::Y);
    let decay = 0.7f32;
    let mut fa = TestAlloc {
        gpu: &gpu,
        live: Vec::new(),
    };
    // Decay first, then drain the remaining signal with the floor.
    let reads: Vec<_> = (0..8)
        .map(|_| gpu.alloc_slice::<[u16; 4]>((SIZE * SIZE) as u64, Memory::Readback))
        .collect();

    let cb = gpu.commands_begin(Queue::Main);
    gpu.cmd_copy_to_texture(cb, btex.texture, bup.cast());
    gpu.cmd_copy_to_texture(cb, ktex.texture, kup.cast());
    gpu.cmd_barrier(
        cb,
        Stage::Transfer,
        Stage::FragmentShader,
        HazardFlags::empty(),
    );
    h.heap.bind(&gpu, cb);
    let pass = h.pass.as_mut().unwrap();
    for frame in 0..8 {
        let (slot, d, floor) = match frame {
            0 => (bslot, decay, 0.0),
            1..=5 => (kslot, decay, 0.0),
            _ => (kslot, 1.0, 0.2),
        };
        pass.record(&gpu, cb, &mut fa, slot, h.sampler, d, floor, 0.0, cam, cam);
        gpu.cmd_barrier(
            cb,
            Stage::RasterColorOut,
            Stage::Transfer,
            HazardFlags::empty(),
        );
        gpu.cmd_copy_texture_to_buffer(cb, pass.latest_texture(), reads[frame].cast());
        // The execution dependency orders the readback copy.
        gpu.cmd_barrier(
            cb,
            Stage::Transfer,
            Stage::RasterColorOut,
            HazardFlags::empty(),
        );
    }
    gpu.queue_submit(Queue::Main, &[cb]);
    gpu.queue_wait_idle(Queue::Main);

    // A uniform field isolates per-frame fp16 quantization.
    let probe = (SIZE / 2, SIZE / 2);
    for frame in 0..6 {
        let got = texel(reads[frame], probe.0, probe.1);
        let want = bright * decay.powi(frame as i32);
        assert!(
            (got - want).abs().max_element() < 0.02 * want.max_element(),
            "frame {frame}: {got} vs {want}"
        );
    }
    // The trail persists after the bright input ends.
    assert!(texel(reads[1], probe.0, probe.1).x > 1.0);
    // The linear floor reaches exact zero without an fp16 ghost.
    let after_floor = texel(reads[7], probe.0, probe.1);
    assert_eq!(after_floor, Vec3::ZERO, "floor must kill the trail dead");
    // The midpoint confirms linear, rather than multiplicative, drain.
    let mid = texel(reads[6], probe.0, probe.1);
    let want_mid = (bright.x * decay.powi(5) - 0.2).max(0.0);
    assert!(
        (mid.x - want_mid).abs() < 0.01,
        "linear drain midpoint: {mid} vs {want_mid}"
    );

    fa.free();
    for r in reads {
        gpu.free(r);
    }
    gpu.free(bup);
    gpu.free(kup);
    gpu.texture_free_and_destroy(btex);
    gpu.texture_free_and_destroy(ktex);
    use gpu::pass::Pass;
    h.pass.take().unwrap().free(&gpu);
    h.heap.free(&gpu);
}
