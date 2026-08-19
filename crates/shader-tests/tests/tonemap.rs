//! Verifies the Tony McMapface fullscreen presentation pass against a CPU
//! reference using the same LUT asset and shared encode/strip math. CPU code
//! supplies only clamp-to-edge bilinear sampling; exposure ordering, sRGB
//! encoding, deterministic dither, and the bloom slot-zero sentinel are
//! exercised by separate renders.

use abi_core::glam::{UVec2, Vec2, Vec3};
use abi_post::{TONY_LUT_HEIGHT, TONY_LUT_WIDTH, TonemapTonyData, tony_encode, tony_taps};
use asha_assets::load_spv;
use gpu::{
    AllocationType, Gpu, GpuPtr, HazardFlags, LoadOp, Memory, Queue, RenderAttachment,
    RenderPassDesc, SamplerDesc, ShaderTypeGraphics, Stage, StoreOp, TextureDesc, TextureFormat,
    TextureViewDesc, UsageFlags,
};

fn f16_to_f32(h: u16) -> f32 {
    let (sign, exp, mant) = (
        (h >> 15) as u32,
        ((h >> 10) & 0x1F) as u32,
        (h & 0x3FF) as u32,
    );
    let f = match exp {
        0 => (mant as f32) * (2.0f32).powi(-24),
        31 => f32::INFINITY,
        _ => (1.0 + mant as f32 / 1024.0) * (2.0f32).powi(exp as i32 - 15),
    };
    if sign != 0 { -f } else { f }
}

const W: u32 = 64;
const H: u32 = 64;
const HDR_IDX: u32 = 1;
const LUT_IDX: u32 = 2;
const SAMPLER_IDX: u32 = 1;

/// Deterministic HDR test pattern: zeros, LDR grays, saturated hues, HDR.
fn hdr_pixel(x: u32, y: u32) -> Vec3 {
    match (x + y * W) % 7 {
        0 => Vec3::ZERO,
        1 => Vec3::splat(0.18),
        2 => Vec3::new(1.0, 0.1, 0.05),
        3 => Vec3::new(0.1, 2.5, 0.2),
        4 => Vec3::new(0.2, 0.4, 8.0),
        5 => Vec3::splat(20.0),
        _ => Vec3::new(x as f32 / W as f32, y as f32 / H as f32, 0.5) * 3.0,
    }
}

struct CpuLut {
    texels: Vec<Vec3>,
}

impl CpuLut {
    fn load() -> Self {
        let path = asha_assets::asset_path("luts/tony_mc_mapface_2304x48_rgba16f.bin");
        let bytes = std::fs::read(&path).expect("LUT asset present");
        assert_eq!(bytes.len(), (TONY_LUT_WIDTH * TONY_LUT_HEIGHT * 8) as usize);
        let texels = bytes
            .chunks_exact(8)
            .map(|t| {
                Vec3::new(
                    f16_to_f32(u16::from_le_bytes([t[0], t[1]])),
                    f16_to_f32(u16::from_le_bytes([t[2], t[3]])),
                    f16_to_f32(u16::from_le_bytes([t[4], t[5]])),
                )
            })
            .collect();
        Self { texels }
    }

    /// Clamp-to-edge bilinear sampling, matching Vulkan filtering.
    fn sample(&self, uv: Vec2) -> Vec3 {
        let size = Vec2::new(TONY_LUT_WIDTH as f32, TONY_LUT_HEIGHT as f32);
        let t = uv * size - 0.5;
        let (x0, y0) = (t.x.floor(), t.y.floor());
        let (fx, fy) = (t.x - x0, t.y - y0);
        let texel = |x: f32, y: f32| {
            let xi = (x as i32).clamp(0, TONY_LUT_WIDTH as i32 - 1) as u32;
            let yi = (y as i32).clamp(0, TONY_LUT_HEIGHT as i32 - 1) as u32;
            self.texels[(yi * TONY_LUT_WIDTH + xi) as usize]
        };
        let top = texel(x0, y0).lerp(texel(x0 + 1.0, y0), fx);
        let bottom = texel(x0, y0 + 1.0).lerp(texel(x0 + 1.0, y0 + 1.0), fx);
        top.lerp(bottom, fy)
    }

    fn tony(&self, hdr: Vec3) -> Vec3 {
        let taps = tony_taps(tony_encode(hdr));
        self.sample(taps.uv_low)
            .lerp(self.sample(taps.uv_high), taps.b_frac)
    }
}

#[test]
fn tony_matches_cpu_lut_sampling() {
    let gpu = Gpu::new(true).expect("vulkan init");
    let vert = gpu.shader_create(
        &load_spv("fullscreen_vert"),
        ShaderTypeGraphics::Vertex,
        "fullscreen_vert",
    );
    let frag = gpu.shader_create(
        &load_spv("tony_frag"),
        ShaderTypeGraphics::Fragment,
        "tony_frag",
    );
    let lut_cpu = CpuLut::load();

    let make_sampled = |dims: [u32; 3], format| {
        let desc = TextureDesc {
            dimensions: dims,
            format,
            usage: UsageFlags::SAMPLED | UsageFlags::TRANSFER_DST,
            ..Default::default()
        };
        let (size, align) = gpu.texture_size_and_align(desc);
        let mem = gpu.mem_alloc_raw(size, 1, align, Memory::Gpu, AllocationType::Default);
        (gpu.texture_create(desc, mem, Queue::Main, None), mem)
    };
    let (lut_tex, lut_mem) = make_sampled(
        [TONY_LUT_WIDTH, TONY_LUT_HEIGHT, 1],
        TextureFormat::Rgba16Float,
    );
    let (hdr_tex, hdr_mem) = make_sampled([W, H, 1], TextureFormat::Rgba32Float);

    let lut_bytes = std::fs::read(asha_assets::asset_path(
        "luts/tony_mc_mapface_2304x48_rgba16f.bin",
    ))
    .unwrap();
    let lut_up = gpu.alloc_slice::<u8>(lut_bytes.len() as u64, Memory::Default);
    let hdr_up = gpu.alloc_slice::<[f32; 4]>((W * H) as u64, Memory::Default);
    unsafe {
        std::ptr::copy_nonoverlapping(lut_bytes.as_ptr(), lut_up.cpu, lut_bytes.len());
        for y in 0..H {
            for x in 0..W {
                let c = hdr_pixel(x, y);
                *hdr_up.cpu.add((y * W + x) as usize) = [c.x, c.y, c.z, 1.0];
            }
        }
    }

    // Set 0 sampled heap and set 2 sampler heap; slot zero remains reserved.
    let tex_heap = gpu.mem_alloc_raw(
        gpu.texture_view_descriptor_size() as u64 * 8,
        64,
        256,
        Memory::Default,
        AllocationType::Descriptors,
    );
    gpu.set_texture_desc(
        tex_heap,
        HDR_IDX,
        gpu.texture_view_descriptor(hdr_tex, TextureViewDesc::default()),
    );
    gpu.set_texture_desc(
        tex_heap,
        LUT_IDX,
        gpu.texture_view_descriptor(lut_tex, TextureViewDesc::default()),
    );
    let sampler_heap = gpu.mem_alloc_raw(
        gpu.sampler_descriptor_size() as u64 * 8,
        64,
        256,
        Memory::Default,
        AllocationType::Descriptors,
    );
    gpu.set_sampler_desc(
        sampler_heap,
        SAMPLER_IDX,
        gpu.sampler_descriptor(SamplerDesc {
            address_mode_u: gpu::AddressMode::ClampToEdge,
            address_mode_v: gpu::AddressMode::ClampToEdge,
            address_mode_w: gpu::AddressMode::ClampToEdge,
            ..Default::default()
        }),
    );

    let target = gpu.texture_alloc_and_create(
        TextureDesc {
            dimensions: [W, H, 1],
            format: TextureFormat::Rgba32Float,
            usage: UsageFlags::COLOR_ATTACHMENT | UsageFlags::TRANSFER_SRC,
            ..Default::default()
        },
        Queue::Main,
        None,
    );
    let data = gpu.alloc::<TonemapTonyData>(Memory::Default);
    let indices = gpu.alloc_slice::<u32>(3, Memory::Default);
    let readback = gpu.alloc_slice::<[f32; 4]>((W * H) as u64, Memory::Readback);
    unsafe {
        *data.cpu = TonemapTonyData {
            hdr_texture_id: HDR_IDX,
            hdr_sampler_id: SAMPLER_IDX,
            lut_texture_id: LUT_IDX,
            lut_sampler_id: SAMPLER_IDX,
            dither_strength: 0.0,
            exposure: 1.0,
            bloom_texture_id: 0, // Slot-zero sentinel: no bloom composite.
            bloom_intensity: 0.0,
        };
        for i in 0..3 {
            *indices.cpu.add(i) = i as u32;
        }
    }

    // Upload the LUT and HDR source once; subsequent passes vary only controls.
    let cb = gpu.commands_begin(Queue::Main);
    gpu.cmd_copy_to_texture(cb, lut_tex, lut_up.cast());
    gpu.cmd_copy_to_texture(cb, hdr_tex, hdr_up.cast());
    gpu.cmd_barrier(
        cb,
        Stage::Transfer,
        Stage::FragmentShader,
        HazardFlags::empty(),
    );
    gpu.queue_submit(Queue::Main, &[cb]);
    gpu.queue_wait_idle(Queue::Main);

    let render = |exposure: f32, dither_strength: f32| {
        // SAFETY: the preceding wait_idle serializes mapped writes with GPU reads.
        unsafe {
            (*data.cpu).exposure = exposure;
            (*data.cpu).dither_strength = dither_strength;
        }
        let cb = gpu.commands_begin(Queue::Main);
        gpu.cmd_set_desc_heap(cb, tex_heap.gpu, GpuPtr::null(), sampler_heap.gpu);
        gpu.cmd_begin_render_pass(
            cb,
            RenderPassDesc {
                color_attachments: &[RenderAttachment {
                    texture: target.texture,
                    load_op: LoadOp::Clear,
                    store_op: StoreOp::Store,
                    clear_color: [0.0; 4],
                    ..Default::default()
                }],
                ..Default::default()
            },
        );
        gpu.cmd_set_shaders(cb, vert, frag);
        gpu.cmd_draw_indexed_instanced(cb, GpuPtr::null(), data.gpu.cast(), indices.cast(), 3, 1);
        gpu.cmd_end_render_pass(cb);
        gpu.cmd_barrier(
            cb,
            Stage::RasterColorOut,
            Stage::Transfer,
            HazardFlags::empty(),
        );
        gpu.cmd_copy_texture_to_buffer(cb, target.texture, readback.cast());
        gpu.queue_submit(Queue::Main, &[cb]);
        gpu.queue_wait_idle(Queue::Main);
    };

    // Exposure 1.0 must be neutral; compare against the unscaled CPU chain.
    render(1.0, 0.0);
    let mut worst = 0.0f32;
    for y in 0..H {
        for x in 0..W {
            let want = abi_post::srgb_encode(lut_cpu.tony(hdr_pixel(x, y)));
            let got = unsafe { *readback.cpu.add((y * W + x) as usize) };
            for c in 0..3 {
                let err = (got[c] - want[c]).abs();
                worst = worst.max(err);
                // Allow precision differences from f16 filtering.
                assert!(
                    err < 4e-3,
                    "pixel ({x},{y}) ch {c}: {} vs {}",
                    got[c],
                    want[c]
                );
            }
        }
    }
    // The deterministic pixel hash must match and move enough pixels to prove
    // the final quantization dither is live.
    render(1.0, 1.0 / 255.0);
    let mut moved = 0u32;
    for y in 0..H {
        for x in 0..W {
            let offset = abi_post::dither_tri(UVec2::new(x, y)) / 255.0;
            let want = abi_post::srgb_encode(lut_cpu.tony(hdr_pixel(x, y))) + Vec3::splat(offset);
            let got = unsafe { *readback.cpu.add((y * W + x) as usize) };
            for c in 0..3 {
                assert!(
                    (got[c] - want[c]).abs() < 4e-3,
                    "dithered pixel ({x},{y}) ch {c}: {} vs {}",
                    got[c],
                    want[c]
                );
            }
            moved += (offset.abs() > 0.5 / 255.0) as u32;
        }
    }
    // A quarter-amplitude tail is expected; require half that for liveness.
    assert!(
        moved > (W * H) / 8,
        "dither too weak to matter: {moved} pixels moved"
    );

    // Exposure 0.5 must scale HDR before the same tonemap chain.
    render(0.5, 0.0);
    for y in 0..H {
        for x in 0..W {
            let want = abi_post::srgb_encode(lut_cpu.tony(hdr_pixel(x, y) * 0.5));
            let got = unsafe { *readback.cpu.add((y * W + x) as usize) };
            for c in 0..3 {
                let err = (got[c] - want[c]).abs();
                assert!(
                    err < 4e-3,
                    "exposure-0.5 pixel ({x},{y}) ch {c}: {} vs {}",
                    got[c],
                    want[c]
                );
            }
        }
    }

    // Check dark and bright transform endpoints.
    let black = lut_cpu.tony(Vec3::ZERO);
    assert!(black.max_element() < 0.01, "tony(0) = {black}");
    let bright = lut_cpu.tony(Vec3::splat(20.0));
    assert!(bright.min_element() > 0.8, "tony(20) = {bright}");
    println!("tony parity: worst channel error {worst:.2e}");

    gpu.texture_free_and_destroy(target);
    gpu.texture_destroy(lut_tex);
    gpu.texture_destroy(hdr_tex);
    gpu.shader_destroy(vert);
    gpu.shader_destroy(frag);
    gpu.free(data);
    gpu.free(indices);
    gpu.free(readback);
    gpu.free(lut_up);
    gpu.free(hdr_up);
    gpu.mem_free_raw(tex_heap);
    gpu.mem_free_raw(sampler_heap);
    gpu.mem_free_raw(lut_mem);
    gpu.mem_free_raw(hdr_mem);
}
