//! Headless GPU coverage for jump-flood outline stages.
//!
//! Synthetic masks isolate seed selection, propagation, rejection, and cutoff.

mod common;

use abi_post::{OutlineCompositeData, OutlineGroup, OutlineJfaFloodData, OutlineJfaInitData};
use common::f16_to_f32;
use gpu::{
    CommandBuffer, Gpu, GpuPtr, HazardFlags, LoadOp, Memory, Queue, RenderAttachment,
    RenderPassDesc, ShaderTypeGraphics, Stage, StoreOp, TextureDesc, TextureFormat,
    TextureViewDesc, UsageFlags,
};

const W: u32 = 8;
const H: u32 = 8;

/// Decoded RGBA16F jump-flood texel.
fn seed_at(readback: gpu::Ptr<[u16; 4]>, x: u32, y: u32) -> [f32; 4] {
    // SAFETY: every readback allocation below contains exactly W × H RGBA16F texels.
    let raw = unsafe { *readback.cpu.add((y * W + x) as usize) };
    raw.map(f16_to_f32)
}

fn assert_close(got: f32, want: f32, label: &str) {
    assert!((got - want).abs() < 1.5e-3, "{label}: {got} vs {want}");
}

fn bind(heap: &gpu::HeapSlots, gpu: &Gpu, cb: CommandBuffer) {
    heap.bind(gpu, cb);
}

#[test]
fn jfa_outline_preserves_seeds_groups_and_composite_contract() {
    let gpu = Gpu::new(true).expect("vulkan init");
    let init_shader = gpu.shader_create_compute(
        &asha_assets::load_spv("outline_jfa_init"),
        8,
        8,
        1,
        "outline_jfa_init",
    );
    let flood_shader = gpu.shader_create_compute(
        &asha_assets::load_spv("outline_jfa_flood"),
        8,
        8,
        1,
        "outline_jfa_flood",
    );
    let fullscreen_vert = gpu.shader_create(
        &asha_assets::load_spv("fullscreen_vert"),
        ShaderTypeGraphics::Vertex,
        "fullscreen_vert",
    );
    let composite_frag = gpu.shader_create(
        &asha_assets::load_spv("outline_composite"),
        ShaderTypeGraphics::Fragment,
        "outline_composite",
    );

    let mask = gpu.texture_alloc_and_create(
        TextureDesc {
            dimensions: [W, H, 1],
            format: TextureFormat::R8Unorm,
            usage: UsageFlags::SAMPLED | UsageFlags::TRANSFER_DST,
            ..Default::default()
        },
        Queue::Main,
        None,
    );
    let jfa_desc = TextureDesc {
        dimensions: [W, H, 1],
        format: TextureFormat::Rgba16Float,
        usage: UsageFlags::SAMPLED | UsageFlags::STORAGE | UsageFlags::TRANSFER_SRC,
        ..Default::default()
    };
    let jfa_a = gpu.texture_alloc_and_create(jfa_desc, Queue::Main, None);
    let jfa_b = gpu.texture_alloc_and_create(jfa_desc, Queue::Main, None);
    let display = gpu.texture_alloc_and_create(
        TextureDesc {
            dimensions: [W, H, 1],
            format: TextureFormat::Rgba32Float,
            usage: UsageFlags::COLOR_ATTACHMENT | UsageFlags::TRANSFER_SRC,
            ..Default::default()
        },
        Queue::Main,
        None,
    );

    let mut heap = gpu.heap_slots_create(4, 3, 2);
    let mask_slot = heap.add_sampled(
        &gpu,
        gpu.texture_view_descriptor(mask.texture, TextureViewDesc::default()),
    );
    let jfa_a_slot = heap.add_sampled(
        &gpu,
        gpu.texture_view_descriptor(jfa_a.texture, TextureViewDesc::default()),
    );
    let jfa_b_slot = heap.add_sampled(
        &gpu,
        gpu.texture_view_descriptor(jfa_b.texture, TextureViewDesc::default()),
    );
    let jfa_a_rw = heap.add_storage(
        &gpu,
        gpu.texture_rw_view_descriptor(jfa_a.texture, TextureViewDesc::default()),
    );
    let jfa_b_rw = heap.add_storage(
        &gpu,
        gpu.texture_rw_view_descriptor(jfa_b.texture, TextureViewDesc::default()),
    );

    let mask_upload = gpu.alloc_slice::<u8>((W * H) as u64, Memory::Default);
    // SAFETY: fresh W × H host-visible mask allocation.
    unsafe {
        std::ptr::write_bytes(mask_upload.cpu, 0, (W * H) as usize);
        *mask_upload.cpu.add((1 * W + 1) as usize) = 1; // group 1 at (1, 1)
        *mask_upload.cpu.add((5 * W + 6) as usize) = 2; // group 2 at (6, 5)
    }
    let init_data = gpu.alloc::<OutlineJfaInitData>(Memory::Default);
    // SAFETY: fresh mapped dispatch allocation.
    unsafe {
        *init_data.cpu = OutlineJfaInitData {
            mask_texture_id: mask_slot.index(),
            output_a_id: jfa_a_rw.index(),
            output_b_id: jfa_b_rw.index(),
            size: [W, H],
            ..Default::default()
        };
    }
    let init_readback = gpu.alloc_slice::<[u16; 4]>((W * H) as u64, Memory::Readback);

    // Isolate initialization before flooding overwrites the texture.
    let cb = gpu.commands_begin(Queue::Main);
    gpu.cmd_copy_to_texture(cb, mask.texture, mask_upload.cast());
    gpu.cmd_barrier(cb, Stage::Transfer, Stage::Compute, HazardFlags::empty());
    bind(&heap, &gpu, cb);
    gpu.cmd_set_compute_shader(cb, init_shader);
    gpu.cmd_dispatch(cb, init_data.gpu, 1, 1, 1);
    gpu.cmd_barrier(cb, Stage::Compute, Stage::Transfer, HazardFlags::empty());
    gpu.cmd_copy_texture_to_buffer(cb, jfa_a.texture, init_readback.cast());
    gpu.queue_submit(Queue::Main, &[cb]);
    gpu.queue_wait_idle(Queue::Main);

    for &(x, y, group) in &[(1, 1, 1u32), (6, 5, 2u32)] {
        let seed = seed_at(init_readback, x, y);
        assert_close(seed[0], x as f32 + 0.5, "init seed x");
        assert_close(seed[1], y as f32 + 0.5, "init seed y");
        assert_close(seed[2], group as f32 / 255.0, "init seed group");
    }
    let empty = seed_at(init_readback, 0, 0);
    assert_eq!(
        empty,
        [-1.0, -1.0, 0.0, 0.0],
        "empty mask texel must be sentinel"
    );

    let flood_datas = [
        gpu.alloc::<OutlineJfaFloodData>(Memory::Default),
        gpu.alloc::<OutlineJfaFloodData>(Memory::Default),
        gpu.alloc::<OutlineJfaFloodData>(Memory::Default),
    ];
    let flood_specs = [
        (jfa_a_slot.index(), jfa_b_rw.index(), 4),
        (jfa_b_slot.index(), jfa_a_rw.index(), 2),
        (jfa_a_slot.index(), jfa_b_rw.index(), 1),
    ];
    // SAFETY: one allocation per jump-flood step.
    unsafe {
        for (data, (input_texture_id, output_texture_id, step_size)) in
            flood_datas.iter().zip(flood_specs)
        {
            *data.cpu = OutlineJfaFloodData {
                input_texture_id,
                output_texture_id,
                step_size,
                size: [W, H],
                region_offset: [0, 0],
                region_size: [W, H],
                ..Default::default()
            };
        }
    }
    let final_readback = gpu.alloc_slice::<[u16; 4]>((W * H) as u64, Memory::Readback);
    let display_readback = gpu.alloc_slice::<[f32; 4]>((W * H) as u64, Memory::Readback);
    let indices = gpu.fullscreen_triangle_indices();
    let groups = [
        OutlineGroup {
            color: [1.0, 0.0, 0.0, 1.0],
            width: 2.5,
            ..Default::default()
        },
        OutlineGroup {
            color: [0.0, 1.0, 0.0, 1.0],
            width: 1.0,
            ..Default::default()
        },
    ];
    let composite_data = gpu.alloc::<OutlineCompositeData>(Memory::Default);
    // SAFETY: fresh mapped fragment-data allocation.
    unsafe {
        *composite_data.cpu = OutlineCompositeData {
            jfa_texture_id: jfa_b_slot.index(), // three steps end B → A → B
            mask_texture_id: mask_slot.index(),
            group_count: groups.len() as u32,
            screen_size: [W, H],
            region_min: [0, 0],
            region_max: [W, H],
            groups: [
                groups[0],
                groups[1],
                OutlineGroup::default(),
                OutlineGroup::default(),
                OutlineGroup::default(),
                OutlineGroup::default(),
                OutlineGroup::default(),
                OutlineGroup::default(),
            ],
            ..Default::default()
        };
    }

    let cb = gpu.commands_begin(Queue::Main);
    bind(&heap, &gpu, cb);
    // Order initialization before the first flood.
    gpu.cmd_barrier(
        cb,
        Stage::Compute,
        Stage::Compute,
        HazardFlags::SHADER_IMAGE,
    );
    gpu.cmd_set_compute_shader(cb, flood_shader);
    for (i, data) in flood_datas.iter().enumerate() {
        gpu.cmd_dispatch(cb, data.gpu, 1, 1, 1);
        if i + 1 < flood_datas.len() {
            gpu.cmd_barrier(
                cb,
                Stage::Compute,
                Stage::Compute,
                HazardFlags::SHADER_IMAGE,
            );
        }
    }
    gpu.cmd_barrier(
        cb,
        Stage::Compute,
        Stage::FragmentShader,
        HazardFlags::SHADER_IMAGE,
    );
    gpu.cmd_begin_render_pass(
        cb,
        RenderPassDesc {
            color_attachments: &[RenderAttachment {
                texture: display.texture,
                load_op: LoadOp::Clear,
                store_op: StoreOp::Store,
                clear_color: [0.0; 4],
                ..Default::default()
            }],
            ..Default::default()
        },
    );
    gpu.cmd_set_shaders(cb, fullscreen_vert, composite_frag);
    gpu.cmd_draw_indexed_instanced(
        cb,
        GpuPtr::null(),
        composite_data.gpu.cast(),
        indices.cast(),
        3,
        1,
    );
    gpu.cmd_end_render_pass(cb);
    gpu.cmd_barrier(cb, Stage::Compute, Stage::Transfer, HazardFlags::empty());
    gpu.cmd_barrier(
        cb,
        Stage::RasterColorOut,
        Stage::Transfer,
        HazardFlags::empty(),
    );
    gpu.cmd_copy_texture_to_buffer(cb, jfa_b.texture, final_readback.cast());
    gpu.cmd_copy_texture_to_buffer(cb, display.texture, display_readback.cast());
    gpu.queue_submit(Queue::Main, &[cb]);
    gpu.queue_wait_idle(Queue::Main);

    let seeds = [(1u32, 1u32, 1u32), (6u32, 5u32, 2u32)];
    for y in 0..H {
        for x in 0..W {
            let distances = seeds.map(|(sx, sy, _)| {
                let dx = x as i32 - sx as i32;
                let dy = y as i32 - sy as i32;
                dx * dx + dy * dy
            });
            if distances[0] == distances[1] {
                continue; // Ties depend on scan order.
            }
            let chosen = if distances[0] < distances[1] {
                seeds[0]
            } else {
                seeds[1]
            };
            let seed = seed_at(final_readback, x, y);
            assert_close(seed[0], chosen.0 as f32 + 0.5, "flood nearest seed x");
            assert_close(seed[1], chosen.1 as f32 + 0.5, "flood nearest seed y");
            assert_close(seed[2], chosen.2 as f32 / 255.0, "flood preserved group");
        }
    }

    // Check interior rejection, edge fading, and width cutoff.
    let pixel = |x: u32, y: u32| unsafe { *display_readback.cpu.add((y * W + x) as usize) };
    assert_eq!(
        pixel(1, 1),
        [0.0; 4],
        "silhouette interior must not composite"
    );
    let edge = pixel(3, 1); // Distance 2, width 2.5.
    assert_close(edge[0], 1.0, "outline exterior red");
    assert_close(edge[3], 0.5, "one-pixel outline fade");
    assert_eq!(
        pixel(4, 1),
        [0.0; 4],
        "outside group width must not composite"
    );

    gpu.free(composite_data);
    gpu.free(indices);
    gpu.free(display_readback);
    gpu.free(final_readback);
    for data in flood_datas {
        gpu.free(data);
    }
    gpu.free(init_readback);
    gpu.free(init_data);
    gpu.free(mask_upload);
    heap.free(&gpu);
    gpu.texture_free_and_destroy(display);
    gpu.texture_free_and_destroy(jfa_b);
    gpu.texture_free_and_destroy(jfa_a);
    gpu.texture_free_and_destroy(mask);
    gpu.shader_destroy(composite_frag);
    gpu.shader_destroy(fullscreen_vert);
    gpu.shader_destroy(flood_shader);
    gpu.shader_destroy(init_shader);
}
