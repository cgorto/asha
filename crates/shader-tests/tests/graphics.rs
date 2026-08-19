//! Verifies pointer-pulled triangle rendering, GPU-written indirect draw
//! arguments, and alpha blending through offscreen attachment readback.
//! These checks keep winding, push-constant plumbing, draw-argument barriers,
//! attachment handling, and blend state from regressing silently.

use abi_core::TriangleData;
use asha_assets::load_spv;
use gpu::{
    Gpu, HazardFlags, LoadOp, Memory, Queue, RenderAttachment, RenderPassDesc, ShaderTypeGraphics,
    Stage, StoreOp, TextureDesc, TextureFormat, UsageFlags,
};

#[test]
fn triangle_renders_offscreen() {
    const W: u32 = 128;
    const H: u32 = 128;

    let gpu = Gpu::new(true).expect("vulkan init");
    let vert = gpu.shader_create(
        &load_spv("triangle_vert"),
        ShaderTypeGraphics::Vertex,
        "triangle_vert",
    );
    let frag = gpu.shader_create(
        &load_spv("triangle_frag"),
        ShaderTypeGraphics::Fragment,
        "triangle_frag",
    );

    let target = gpu.texture_alloc_and_create(
        TextureDesc {
            dimensions: [W, H, 1],
            format: TextureFormat::Rgba8Unorm,
            usage: UsageFlags::COLOR_ATTACHMENT | UsageFlags::TRANSFER_SRC,
            ..Default::default()
        },
        Queue::Main,
        None,
    );

    let positions = gpu.alloc_slice::<[f32; 2]>(3, Memory::Default);
    let colors = gpu.alloc_slice::<[f32; 4]>(3, Memory::Default);
    let tint = gpu.alloc::<[f32; 4]>(Memory::Default);
    let tri = gpu.alloc::<TriangleData>(Memory::Default);
    let indices = gpu.alloc_slice::<u32>(3, Memory::Default);
    unsafe {
        // CCW in Vulkan NDC (+Y down): top, bottom-left, bottom-right.
        *positions.cpu.add(0) = [0.0, -0.6];
        *positions.cpu.add(1) = [-0.55, 0.5];
        *positions.cpu.add(2) = [0.55, 0.5];
        for i in 0..3 {
            *colors.cpu.add(i) = [1.0, 1.0, 1.0, 1.0];
            *indices.cpu.add(i) = i as u32;
        }
        *tint.cpu = [1.0, 0.0, 1.0, 1.0];
        *tri.cpu = TriangleData {
            positions: positions.gpu,
            colors: colors.gpu,
            tint: tint.gpu,
        };
    }

    let readback = gpu.alloc_slice::<u8>((W * H * 4) as u64, Memory::Readback);

    let cb = gpu.commands_begin(Queue::Main);
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
    gpu.cmd_draw_indexed_instanced(cb, tri.gpu.cast(), tri.gpu.cast(), indices.cast(), 3, 1);
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

    let px = |x: u32, y: u32| -> [u8; 4] {
        unsafe {
            let p = readback.cpu.add(((y * W + x) * 4) as usize);
            [*p, *p.add(1), *p.add(2), *p.add(3)]
        }
    };

    // The center must contain the tinted triangle.
    let center = px(W / 2, (H as f32 * 0.55) as u32);
    assert!(
        center[0] > 200 && center[1] < 50 && center[2] > 200,
        "center pixel not magenta: {center:?} — triangle didn't render"
    );
    // Corners remain clear.
    assert_eq!(px(2, 2), [0, 0, 0, 0], "corner should be clear");
    assert_eq!(px(W - 3, 2), [0, 0, 0, 0], "corner should be clear");

    gpu.texture_free_and_destroy(target);
    gpu.shader_destroy(vert);
    gpu.shader_destroy(frag);
    gpu.free(positions);
    gpu.free(colors);
    gpu.free(tint);
    gpu.free(tri);
    gpu.free(indices);
    gpu.free(readback);
}

/// A compute shader writes GPU-only indirect arguments and count; the draw
/// consumes them after the DRAW_ARGUMENTS barrier, without a host draw command.
#[test]
fn gpu_driven_indirect_draw() {
    use abi_core::{DrawIndexedIndirectCommand, WriteDrawData};

    const W: u32 = 128;
    const H: u32 = 128;

    let gpu = Gpu::new(true).expect("vulkan init");
    let vert = gpu.shader_create(
        &load_spv("triangle_vert"),
        ShaderTypeGraphics::Vertex,
        "triangle_vert",
    );
    let frag = gpu.shader_create(
        &load_spv("triangle_frag"),
        ShaderTypeGraphics::Fragment,
        "triangle_frag",
    );
    let writer = gpu.shader_create_compute(&load_spv("write_draw"), 1, 1, 1, "write_draw");

    let target = gpu.texture_alloc_and_create(
        TextureDesc {
            dimensions: [W, H, 1],
            format: TextureFormat::Rgba8Unorm,
            usage: UsageFlags::COLOR_ATTACHMENT | UsageFlags::TRANSFER_SRC,
            ..Default::default()
        },
        Queue::Main,
        None,
    );

    let positions = gpu.alloc_slice::<[f32; 2]>(3, Memory::Default);
    let colors = gpu.alloc_slice::<[f32; 4]>(3, Memory::Default);
    let tint = gpu.alloc::<[f32; 4]>(Memory::Default);
    let tri = gpu.alloc::<TriangleData>(Memory::Default);
    let indices = gpu.alloc_slice::<u32>(3, Memory::Default);
    // GPU-only arguments ensure the host cannot make the draw pass by writing them.
    let cmds = gpu.alloc_slice::<DrawIndexedIndirectCommand>(1, Memory::Gpu);
    let count = gpu.alloc::<u32>(Memory::Gpu);
    let writer_data = gpu.alloc::<WriteDrawData>(Memory::Default);
    unsafe {
        *positions.cpu.add(0) = [0.0, -0.6];
        *positions.cpu.add(1) = [-0.55, 0.5];
        *positions.cpu.add(2) = [0.55, 0.5];
        for i in 0..3 {
            *colors.cpu.add(i) = [1.0, 1.0, 1.0, 1.0];
            *indices.cpu.add(i) = i as u32;
        }
        *tint.cpu = [0.0, 1.0, 0.0, 1.0];
        *tri.cpu = TriangleData {
            positions: positions.gpu,
            colors: colors.gpu,
            tint: tint.gpu,
        };
        *writer_data.cpu = WriteDrawData {
            cmds: cmds.gpu,
            count: count.gpu,
        };
    }
    let readback = gpu.alloc_slice::<u8>((W * H * 4) as u64, Memory::Readback);

    let cb = gpu.commands_begin(Queue::Main);
    gpu.cmd_set_compute_shader(cb, writer);
    gpu.cmd_dispatch(cb, writer_data.gpu, 1, 1, 1);
    gpu.cmd_barrier(cb, Stage::Compute, Stage::All, HazardFlags::DRAW_ARGUMENTS);
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
    gpu.cmd_draw_indexed_instanced_indirect_multi(
        cb,
        tri.gpu.cast(),
        tri.gpu.cast(),
        indices.cast(),
        cmds.cast(),
        size_of::<DrawIndexedIndirectCommand>() as u32,
        count.cast(),
    );
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

    unsafe {
        let center = readback
            .cpu
            .add((((H as f32 * 0.55) as u32 * W + W / 2) * 4) as usize);
        assert!(
            *center < 50 && *center.add(1) > 200 && *center.add(2) < 50,
            "center not green — the GPU-written draw did not happen"
        );
        assert_eq!(*readback.cpu.add(8), 0, "corner should be clear");
    }

    gpu.texture_free_and_destroy(target);
    gpu.shader_destroy(vert);
    gpu.shader_destroy(frag);
    gpu.shader_destroy(writer);
    for p in [
        positions.cast::<u8>(),
        colors.cast(),
        tint.cast(),
        tri.cast(),
        indices.cast(),
        cmds.cast(),
        count.cast(),
        writer_data.cast(),
        readback.cast(),
    ] {
        gpu.mem_free_raw(p);
    }
}

/// A half-transparent blue triangle over an opaque red clear must produce
/// the channel mix, exercising the configured alpha blend state.
#[test]
fn blend_state_alpha() {
    use gpu::{BlendFactor, BlendOp, BlendState};

    const W: u32 = 64;
    const H: u32 = 64;

    let gpu = Gpu::new(true).expect("vulkan init");
    let vert = gpu.shader_create(
        &load_spv("triangle_vert"),
        ShaderTypeGraphics::Vertex,
        "triangle_vert",
    );
    let frag = gpu.shader_create(
        &load_spv("triangle_frag"),
        ShaderTypeGraphics::Fragment,
        "triangle_frag",
    );

    let target = gpu.texture_alloc_and_create(
        TextureDesc {
            dimensions: [W, H, 1],
            format: TextureFormat::Rgba8Unorm,
            usage: UsageFlags::COLOR_ATTACHMENT | UsageFlags::TRANSFER_SRC,
            ..Default::default()
        },
        Queue::Main,
        None,
    );

    let positions = gpu.alloc_slice::<[f32; 2]>(3, Memory::Default);
    let colors = gpu.alloc_slice::<[f32; 4]>(3, Memory::Default);
    let tint = gpu.alloc::<[f32; 4]>(Memory::Default);
    let tri = gpu.alloc::<TriangleData>(Memory::Default);
    let indices = gpu.alloc_slice::<u32>(3, Memory::Default);
    unsafe {
        *positions.cpu.add(0) = [0.0, -0.9];
        *positions.cpu.add(1) = [-0.9, 0.9];
        *positions.cpu.add(2) = [0.9, 0.9];
        for i in 0..3 {
            *colors.cpu.add(i) = [1.0, 1.0, 1.0, 1.0];
            *indices.cpu.add(i) = i as u32;
        }
        *tint.cpu = [0.0, 0.0, 1.0, 0.5];
        *tri.cpu = TriangleData {
            positions: positions.gpu,
            colors: colors.gpu,
            tint: tint.gpu,
        };
    }
    let readback = gpu.alloc_slice::<u8>((W * H * 4) as u64, Memory::Readback);

    let cb = gpu.commands_begin(Queue::Main);
    gpu.cmd_begin_render_pass(
        cb,
        RenderPassDesc {
            color_attachments: &[RenderAttachment {
                texture: target.texture,
                load_op: LoadOp::Clear,
                store_op: StoreOp::Store,
                clear_color: [1.0, 0.0, 0.0, 1.0],
                ..Default::default()
            }],
            ..Default::default()
        },
    );
    gpu.cmd_set_shaders(cb, vert, frag);
    gpu.cmd_set_blend_state(
        cb,
        BlendState {
            enable: true,
            color_op: BlendOp::Add,
            src_color_factor: BlendFactor::SrcAlpha,
            dst_color_factor: BlendFactor::OneMinusSrcAlpha,
            alpha_op: BlendOp::Add,
            src_alpha_factor: BlendFactor::One,
            dst_alpha_factor: BlendFactor::OneMinusSrcAlpha,
            color_write_mask: 0xF,
        },
    );
    gpu.cmd_draw_indexed_instanced(cb, tri.gpu.cast(), tri.gpu.cast(), indices.cast(), 3, 1);
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

    unsafe {
        let center = readback.cpu.add(((H / 2 * W + W / 2) * 4) as usize);
        let (r, g, b) = (*center, *center.add(1), *center.add(2));
        // 0.5 blue over 0.5 red leaves red and blue approximately equal.
        assert!((100..=155).contains(&r), "red channel {r} not blended");
        assert!(g < 30, "green channel {g} should stay dark");
        assert!((100..=155).contains(&b), "blue channel {b} not blended");
    }

    gpu.texture_free_and_destroy(target);
    gpu.shader_destroy(vert);
    gpu.shader_destroy(frag);
    for p in [
        positions.cast::<u8>(),
        colors.cast(),
        tint.cast(),
        tri.cast(),
        indices.cast(),
        readback.cast(),
    ] {
        gpu.mem_free_raw(p);
    }
}
