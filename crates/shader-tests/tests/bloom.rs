//! Tests the bloom downsample and upsample through the shared fullscreen
//! triangle against CPU references. The downsample draw deliberately passes
//! NULL vertex data: `fullscreen_vert` reads no vertex pointer, while a
//! fragment shader reading the old vertex slot would chase null instead of
//! rendering, so this preserves the GraphicsPush slot regression check.
//! CPU code reimplements only bilinear sampling; combination math remains
//! the shared `abi_post` implementation.

use abi_core::glam::{Vec2, Vec3};
use abi_post::{
    BLOOM_COORDS, BLOOM_TAPS, BloomDownsampleData, BloomUpsampleData, TENT_COORDS, TENT_TAPS,
    bloom_average_partial, bloom_tent_sum, bloom_upsample_blend, bloom_weighted_sum, safe_hdr,
    soft_threshold,
};
use asha_assets::load_spv;
use gpu::{
    AllocationType, Gpu, GpuPtr, HazardFlags, LoadOp, Memory, Queue, RenderAttachment,
    RenderPassDesc, SamplerDesc, ShaderTypeGraphics, Stage, StoreOp, TextureDesc, TextureFormat,
    TextureViewDesc, UsageFlags,
};

const SRC: u32 = 64;
const DST: u32 = 32;
const TEX_IDX: u32 = 2;
const SAMPLER_IDX: u32 = 1;

/// Deterministic HDR source with a bright sample.
fn src_pixel(x: u32, y: u32) -> Vec3 {
    if (x, y) == (20, 20) {
        return Vec3::new(50.0, 60.0, 55.0);
    }
    Vec3::new(
        ((x * 3 + y * 7) % 13) as f32 / 13.0 * 2.0,
        ((x * 5 + y) % 11) as f32 / 11.0,
        ((x + y * 3) % 7) as f32 / 7.0,
    )
}

/// Bilinear sample with Vulkan clamp-to-edge semantics.
fn bilinear(size: u32, texel_at: impl Fn(u32, u32) -> Vec3, uv: Vec2) -> Vec3 {
    let t = uv * size as f32 - 0.5;
    let (x0, y0) = (t.x.floor(), t.y.floor());
    let (fx, fy) = (t.x - x0, t.y - y0);
    let texel = |x: f32, y: f32| {
        texel_at(
            (x as i32).clamp(0, size as i32 - 1) as u32,
            (y as i32).clamp(0, size as i32 - 1) as u32,
        )
    };
    let top = texel(x0, y0).lerp(texel(x0 + 1.0, y0), fx);
    let bottom = texel(x0, y0 + 1.0).lerp(texel(x0 + 1.0, y0 + 1.0), fx);
    top.lerp(bottom, fy)
}

fn sample_bilinear(uv: Vec2) -> Vec3 {
    bilinear(SRC, src_pixel, uv)
}

/// CPU reference using shared post-processing math.
fn cpu_reference(data: &BloomDownsampleData, x: u32, y: u32) -> Vec3 {
    let uv = Vec2::new((x as f32 + 0.5) / DST as f32, (y as f32 + 0.5) / DST as f32);
    let pixel_size = Vec2::from_array(data.pixel_size) * Vec2::from_array(data.bloom_scale);
    let mut samples = [Vec3::ZERO; BLOOM_TAPS];
    for i in 0..BLOOM_TAPS {
        samples[i] = sample_bilinear(uv + BLOOM_COORDS[i] * pixel_size);
    }
    let mut color = if data.use_anti_flicker != 0 {
        bloom_average_partial(&samples)
    } else {
        bloom_weighted_sum(&samples)
    };
    if data.bloom_threshold > 0.0 {
        color = soft_threshold(color, data.bloom_threshold, data.bloom_knee);
    }
    safe_hdr(color)
}

#[test]
fn bloom_downsample_matches_cpu() {
    let gpu = Gpu::new(true).expect("vulkan init");
    let vert = gpu.shader_create(
        &load_spv("fullscreen_vert"),
        ShaderTypeGraphics::Vertex,
        "fullscreen_vert",
    );
    let frag = gpu.shader_create(
        &load_spv("bloom_downsample"),
        ShaderTypeGraphics::Fragment,
        "bloom_downsample",
    );

    let src_desc = TextureDesc {
        dimensions: [SRC, SRC, 1],
        format: TextureFormat::Rgba32Float,
        usage: UsageFlags::SAMPLED | UsageFlags::TRANSFER_DST,
        ..Default::default()
    };
    let (size, align) = gpu.texture_size_and_align(src_desc);
    let src_mem = gpu.mem_alloc_raw(size, 1, align, Memory::Gpu, AllocationType::Default);
    let src_tex = gpu.texture_create(src_desc, src_mem, Queue::Main, None);
    let upload = gpu.alloc_slice::<[f32; 4]>((SRC * SRC) as u64, Memory::Default);
    unsafe {
        for y in 0..SRC {
            for x in 0..SRC {
                let p = src_pixel(x, y);
                *upload.cpu.add((y * SRC + x) as usize) = [p.x, p.y, p.z, 1.0];
            }
        }
    }

    let tex_heap = gpu.mem_alloc_raw(
        gpu.texture_view_descriptor_size() as u64 * 8,
        64,
        256,
        Memory::Default,
        AllocationType::Descriptors,
    );
    gpu.set_texture_desc(
        tex_heap,
        TEX_IDX,
        gpu.texture_view_descriptor(src_tex, TextureViewDesc::default()),
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
            ..Default::default() // linear min/mag
        }),
    );

    let cases = [
        BloomDownsampleData {
            src_texture_id: TEX_IDX,
            src_sampler_id: SAMPLER_IDX,
            pixel_size: [1.0 / SRC as f32; 2],
            use_anti_flicker: 1,
            bloom_threshold: 0.0,
            bloom_knee: 0.0,
            bloom_scale: [1.0, 1.0],
        },
        BloomDownsampleData {
            src_texture_id: TEX_IDX,
            src_sampler_id: SAMPLER_IDX,
            pixel_size: [1.0 / SRC as f32; 2],
            use_anti_flicker: 0,
            bloom_threshold: 1.0,
            bloom_knee: 0.5,
            bloom_scale: [2.0, 1.0],
        },
    ];

    let indices = gpu.alloc_slice::<u32>(3, Memory::Default);
    unsafe {
        for i in 0..3 {
            *indices.cpu.add(i) = i as u32;
        }
    }

    let mut targets = Vec::new();
    let mut readbacks = Vec::new();
    let mut datas = Vec::new();

    let cb = gpu.commands_begin(Queue::Main);
    gpu.cmd_copy_to_texture(cb, src_tex, upload.cast());
    gpu.cmd_barrier(
        cb,
        Stage::Transfer,
        Stage::FragmentShader,
        HazardFlags::empty(),
    );
    gpu.cmd_set_desc_heap(cb, tex_heap.gpu, GpuPtr::null(), sampler_heap.gpu);

    for case in &cases {
        let target = gpu.texture_alloc_and_create(
            TextureDesc {
                dimensions: [DST, DST, 1],
                format: TextureFormat::Rgba32Float,
                usage: UsageFlags::COLOR_ATTACHMENT | UsageFlags::TRANSFER_SRC,
                ..Default::default()
            },
            Queue::Main,
            None,
        );
        let data = gpu.alloc::<BloomDownsampleData>(Memory::Default);
        unsafe { *data.cpu = *case };
        let readback = gpu.alloc_slice::<[f32; 4]>((DST * DST) as u64, Memory::Readback);

        gpu.cmd_begin_render_pass(
            cb,
            RenderPassDesc {
                color_attachments: &[RenderAttachment {
                    texture: target.texture,
                    load_op: LoadOp::Clear,
                    store_op: StoreOp::Store,
                    clear_color: [0.0, 0.0, 0.0, 0.0],
                    ..Default::default()
                }],
                ..Default::default()
            },
        );
        gpu.cmd_set_shaders(cb, vert, frag);
        // Regression: the fullscreen vertex consumes no vertex slot, so the
        // fragment's push data must still come from the fragment slot.
        gpu.cmd_draw_indexed_instanced(cb, GpuPtr::null(), data.gpu.cast(), indices.cast(), 3, 1);
        gpu.cmd_end_render_pass(cb);
        gpu.cmd_barrier(
            cb,
            Stage::RasterColorOut,
            Stage::Transfer,
            HazardFlags::empty(),
        );
        gpu.cmd_copy_texture_to_buffer(cb, target.texture, readback.cast());

        targets.push(target);
        readbacks.push(readback);
        datas.push(data);
    }

    gpu.queue_submit(Queue::Main, &[cb]);
    gpu.queue_wait_idle(Queue::Main);

    for (case_idx, case) in cases.iter().enumerate() {
        let readback = &readbacks[case_idx];
        let mut worst = 0.0f32;
        for y in 0..DST {
            for x in 0..DST {
                let got = unsafe { *readback.cpu.add((y * DST + x) as usize) };
                let want = cpu_reference(case, x, y);
                for (c, (g, w)) in [(got[0], want.x), (got[1], want.y), (got[2], want.z)]
                    .into_iter()
                    .enumerate()
                {
                    let err = (g - w).abs();
                    worst = worst.max(err);
                    assert!(
                        err < 2e-3,
                        "case {case_idx} pixel ({x},{y}) ch {c}: gpu {g} vs cpu {w}"
                    );
                }
                assert_eq!(got[3], 1.0, "alpha must be 1");
            }
        }
        println!("case {case_idx}: worst channel error {worst:.2e}");
    }

    // Distinct outputs verify the selected filtering branch.
    let plain_cfg = BloomDownsampleData {
        use_anti_flicker: 0,
        ..cases[0]
    };
    let karis_ref = cpu_reference(&cases[0], 10, 10);
    let plain_ref = cpu_reference(&plain_cfg, 10, 10);
    assert!(
        plain_ref.x > karis_ref.x * 1.05,
        "pattern too weak to distinguish branches: karis {karis_ref} vs plain {plain_ref}"
    );

    for target in targets {
        gpu.texture_free_and_destroy(target);
    }
    gpu.texture_destroy(src_tex);
    gpu.shader_destroy(vert);
    gpu.shader_destroy(frag);
    for rb in readbacks {
        gpu.free(rb);
    }
    for d in datas {
        gpu.free(d);
    }
    gpu.free(indices);
    gpu.free(upload);
    gpu.mem_free_raw(tex_heap);
    gpu.mem_free_raw(sampler_heap);
    gpu.mem_free_raw(src_mem);
}

const UP_DST: u32 = 32;
const UP_PREV: u32 = 16;
const CUR_IDX: u32 = 3;
const PREV_IDX: u32 = 4;

/// Deterministic current-resolution downsample level.
fn cur_pixel(x: u32, y: u32) -> Vec3 {
    Vec3::new(
        ((x * 7 + y * 3) % 11) as f32 / 11.0 * 1.5,
        ((x + y * 5) % 9) as f32 / 9.0,
        ((x * 2 + y * 2) % 5) as f32 / 5.0,
    )
}

/// Previous upsample result with one hot texel.
fn prev_pixel(x: u32, y: u32) -> Vec3 {
    if (x, y) == (8, 8) {
        return Vec3::new(6.0, 4.0, 2.0);
    }
    Vec3::new(
        ((x * 5 + y) % 7) as f32 / 7.0,
        ((x + y * 7) % 13) as f32 / 13.0,
        ((x * 3 + y * 5) % 8) as f32 / 8.0,
    )
}

/// CPU reference using shared tent-filter math.
fn cpu_upsample_reference(data: &BloomUpsampleData, x: u32, y: u32) -> Vec3 {
    let uv = Vec2::new(
        (x as f32 + 0.5) / UP_DST as f32,
        (y as f32 + 0.5) / UP_DST as f32,
    );
    let pixel_size = Vec2::from_array(data.pixel_size) * Vec2::from_array(data.bloom_scale);
    let current = bilinear(UP_DST, cur_pixel, uv);
    let mut taps = [Vec3::ZERO; TENT_TAPS];
    for i in 0..TENT_TAPS {
        taps[i] = bilinear(UP_PREV, prev_pixel, uv + TENT_COORDS[i] * pixel_size);
    }
    bloom_upsample_blend(current, bloom_tent_sum(&taps), data.blend_factor)
}

#[test]
fn bloom_upsample_matches_cpu() {
    let gpu = Gpu::new(true).expect("vulkan init");
    let vert = gpu.shader_create(
        &load_spv("fullscreen_vert"),
        ShaderTypeGraphics::Vertex,
        "fullscreen_vert",
    );
    let frag = gpu.shader_create(
        &load_spv("bloom_upsample"),
        ShaderTypeGraphics::Fragment,
        "bloom_upsample",
    );

    let make_src = |size: u32, pixel: fn(u32, u32) -> Vec3| {
        let desc = TextureDesc {
            dimensions: [size, size, 1],
            format: TextureFormat::Rgba32Float,
            usage: UsageFlags::SAMPLED | UsageFlags::TRANSFER_DST,
            ..Default::default()
        };
        let (mem_size, align) = gpu.texture_size_and_align(desc);
        let mem = gpu.mem_alloc_raw(mem_size, 1, align, Memory::Gpu, AllocationType::Default);
        let tex = gpu.texture_create(desc, mem, Queue::Main, None);
        let upload = gpu.alloc_slice::<[f32; 4]>((size * size) as u64, Memory::Default);
        // SAFETY: Allocation is host-visible and correctly sized.
        unsafe {
            for y in 0..size {
                for x in 0..size {
                    let p = pixel(x, y);
                    *upload.cpu.add((y * size + x) as usize) = [p.x, p.y, p.z, 1.0];
                }
            }
        }
        (tex, mem, upload)
    };
    let (cur_tex, cur_mem, cur_up) = make_src(UP_DST, cur_pixel);
    let (prev_tex, prev_mem, prev_up) = make_src(UP_PREV, prev_pixel);

    let tex_heap = gpu.mem_alloc_raw(
        gpu.texture_view_descriptor_size() as u64 * 8,
        64,
        256,
        Memory::Default,
        AllocationType::Descriptors,
    );
    gpu.set_texture_desc(
        tex_heap,
        CUR_IDX,
        gpu.texture_view_descriptor(cur_tex, TextureViewDesc::default()),
    );
    gpu.set_texture_desc(
        tex_heap,
        PREV_IDX,
        gpu.texture_view_descriptor(prev_tex, TextureViewDesc::default()),
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
            ..Default::default() // linear min/mag
        }),
    );

    let case = BloomUpsampleData {
        downsample_texture_id: CUR_IDX,
        previous_texture_id: PREV_IDX,
        sampler_id: SAMPLER_IDX,
        blend_factor: 0.925,
        pixel_size: [1.0 / UP_DST as f32; 2],
        bloom_scale: [4.0, 1.0],
    };
    let data = gpu.alloc::<BloomUpsampleData>(Memory::Default);
    let indices = gpu.alloc_slice::<u32>(3, Memory::Default);
    let readback = gpu.alloc_slice::<[f32; 4]>((UP_DST * UP_DST) as u64, Memory::Readback);
    // SAFETY: Allocations are host-visible and correctly sized.
    unsafe {
        *data.cpu = case;
        for i in 0..3 {
            *indices.cpu.add(i) = i as u32;
        }
    }
    let target = gpu.texture_alloc_and_create(
        TextureDesc {
            dimensions: [UP_DST, UP_DST, 1],
            format: TextureFormat::Rgba32Float,
            usage: UsageFlags::COLOR_ATTACHMENT | UsageFlags::TRANSFER_SRC,
            ..Default::default()
        },
        Queue::Main,
        None,
    );

    let cb = gpu.commands_begin(Queue::Main);
    gpu.cmd_copy_to_texture(cb, cur_tex, cur_up.cast());
    gpu.cmd_copy_to_texture(cb, prev_tex, prev_up.cast());
    gpu.cmd_barrier(
        cb,
        Stage::Transfer,
        Stage::FragmentShader,
        HazardFlags::empty(),
    );
    gpu.cmd_set_desc_heap(cb, tex_heap.gpu, GpuPtr::null(), sampler_heap.gpu);
    gpu.cmd_begin_render_pass(
        cb,
        RenderPassDesc {
            color_attachments: &[RenderAttachment {
                texture: target.texture,
                load_op: LoadOp::Clear,
                store_op: StoreOp::Store,
                clear_color: [0.0, 0.0, 0.0, 0.0],
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

    let mut worst = 0.0f32;
    for y in 0..UP_DST {
        for x in 0..UP_DST {
            let got = unsafe { *readback.cpu.add((y * UP_DST + x) as usize) };
            let want = cpu_upsample_reference(&case, x, y);
            for (c, (g, w)) in [(got[0], want.x), (got[1], want.y), (got[2], want.z)]
                .into_iter()
                .enumerate()
            {
                let err = (g - w).abs();
                worst = worst.max(err);
                assert!(err < 2e-3, "pixel ({x},{y}) ch {c}: gpu {g} vs cpu {w}");
            }
            assert_eq!(got[3], 1.0, "alpha must be 1");
        }
    }
    println!("upsample: worst channel error {worst:.2e}");

    // Hot texel dominance verifies blend direction and tent spread.
    let hot = cpu_upsample_reference(&case, 17, 17);
    let cool = cpu_upsample_reference(&case, 17, 25);
    assert!(
        hot.x > cool.x + 1.0,
        "tent spread too weak: hot {hot} vs cool {cool}"
    );

    gpu.texture_free_and_destroy(target);
    gpu.texture_destroy(cur_tex);
    gpu.texture_destroy(prev_tex);
    gpu.shader_destroy(vert);
    gpu.shader_destroy(frag);
    gpu.free(readback);
    gpu.free(data);
    gpu.free(indices);
    gpu.free(cur_up);
    gpu.free(prev_up);
    gpu.mem_free_raw(tex_heap);
    gpu.mem_free_raw(sampler_heap);
    gpu.mem_free_raw(cur_mem);
    gpu.mem_free_raw(prev_mem);
}
