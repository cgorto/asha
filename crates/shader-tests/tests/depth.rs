//! Verifies reverse-Z depth testing and attachment readback. A red triangle
//! at z=0.5 is drawn before an overlapping blue triangle at z=0.25; GREATER
//! must retain red and 0.5 in the overlap while blue wins its uncovered area.
//! Readback of both color and raw D32 depth also locks the 0.0 infinite-far
//! clear value.

use abi_core::{DepthTriangleData, GpuPtr, TriangleData};
use asha_assets::load_spv;
use gpu::{
    CompareOp, DepthFlags, DepthState, Gpu, HazardFlags, LoadOp, Memory, Queue, RenderAttachment,
    RenderPassDesc, ShaderTypeGraphics, Stage, StoreOp, TextureDesc, TextureFormat, UsageFlags,
};

#[test]
fn depth_test_reverse_z() {
    const W: u32 = 64;
    const H: u32 = 64;

    let gpu = Gpu::new(true).expect("vulkan init");
    let vert = gpu.shader_create(
        &load_spv("depth_triangle_vert"),
        ShaderTypeGraphics::Vertex,
        "depth_triangle_vert",
    );
    let frag = gpu.shader_create(
        &load_spv("triangle_frag"),
        ShaderTypeGraphics::Fragment,
        "triangle_frag",
    );

    let color = gpu.texture_alloc_and_create(
        TextureDesc {
            dimensions: [W, H, 1],
            format: TextureFormat::Rgba8Unorm,
            usage: UsageFlags::COLOR_ATTACHMENT | UsageFlags::TRANSFER_SRC,
            ..Default::default()
        },
        Queue::Main,
        None,
    );
    let depth = gpu.texture_alloc_and_create(
        TextureDesc {
            dimensions: [W, H, 1],
            format: TextureFormat::D32Float,
            usage: UsageFlags::DEPTH_STENCIL_ATTACHMENT | UsageFlags::TRANSFER_SRC,
            ..Default::default()
        },
        Queue::Main,
        None,
    );

    // CCW Vulkan NDC vertices and reverse-Z make B (0.25) farther than A
    // (0.5), exercising the configured winding and comparison direction.
    let pos_a = gpu.alloc_slice::<[f32; 4]>(3, Memory::Default);
    let pos_b = gpu.alloc_slice::<[f32; 4]>(3, Memory::Default);
    let tint_a = gpu.alloc::<[f32; 4]>(Memory::Default);
    let tint_b = gpu.alloc::<[f32; 4]>(Memory::Default);
    let tri_a = gpu.alloc::<DepthTriangleData>(Memory::Default);
    let tri_b = gpu.alloc::<DepthTriangleData>(Memory::Default);
    let frag_a = gpu.alloc::<TriangleData>(Memory::Default);
    let frag_b = gpu.alloc::<TriangleData>(Memory::Default);
    let indices = gpu.alloc_slice::<u32>(3, Memory::Default);
    unsafe {
        *pos_a.cpu.add(0) = [0.0, -0.8, 0.5, 0.0];
        *pos_a.cpu.add(1) = [-0.8, 0.8, 0.5, 0.0];
        *pos_a.cpu.add(2) = [0.8, 0.8, 0.5, 0.0];
        *pos_b.cpu.add(0) = [0.5, -0.8, 0.25, 0.0];
        *pos_b.cpu.add(1) = [-0.3, 0.8, 0.25, 0.0];
        *pos_b.cpu.add(2) = [1.3, 0.8, 0.25, 0.0];
        for i in 0..3 {
            *indices.cpu.add(i) = i as u32;
        }
        *tint_a.cpu = [1.0, 0.0, 0.0, 1.0];
        *tint_b.cpu = [0.0, 0.0, 1.0, 1.0];
        *tri_a.cpu = DepthTriangleData {
            positions: pos_a.gpu,
        };
        *tri_b.cpu = DepthTriangleData {
            positions: pos_b.gpu,
        };
        *frag_a.cpu = TriangleData {
            positions: GpuPtr::null(),
            colors: GpuPtr::null(),
            tint: tint_a.gpu,
        };
        *frag_b.cpu = TriangleData {
            positions: GpuPtr::null(),
            colors: GpuPtr::null(),
            tint: tint_b.gpu,
        };
    }

    let color_rb = gpu.alloc_slice::<u8>((W * H * 4) as u64, Memory::Readback);
    let depth_rb = gpu.alloc_slice::<f32>((W * H) as u64, Memory::Readback);

    let cb = gpu.commands_begin(Queue::Main);
    gpu.cmd_begin_render_pass(
        cb,
        RenderPassDesc {
            color_attachments: &[RenderAttachment {
                texture: color.texture,
                load_op: LoadOp::Clear,
                store_op: StoreOp::Store,
                clear_color: [0.0, 0.0, 0.0, 0.0],
                ..Default::default()
            }],
            depth_attachment: Some(RenderAttachment {
                texture: depth.texture,
                load_op: LoadOp::Clear,
                store_op: StoreOp::Store,
                clear_color: [0.0; 4], // reverse-Z infinite-far clear
                ..Default::default()
            }),
            ..Default::default()
        },
    );
    gpu.cmd_set_shaders(cb, vert, frag);
    gpu.cmd_set_depth_state(
        cb,
        DepthState {
            mode: DepthFlags::READ | DepthFlags::WRITE,
            compare: CompareOp::Greater,
            ..Default::default()
        },
    );
    // The far overlapping draw must fail GREATER, not merely paint over A.
    gpu.cmd_draw_indexed_instanced(
        cb,
        tri_a.gpu.cast(),
        frag_a.gpu.cast(),
        indices.cast(),
        3,
        1,
    );
    gpu.cmd_draw_indexed_instanced(
        cb,
        tri_b.gpu.cast(),
        frag_b.gpu.cast(),
        indices.cast(),
        3,
        1,
    );
    gpu.cmd_end_render_pass(cb);
    // Synchronize attachment writes before transfer readback.
    gpu.cmd_barrier(
        cb,
        Stage::RasterColorOut,
        Stage::Transfer,
        HazardFlags::empty(),
    );
    gpu.cmd_barrier(
        cb,
        Stage::LateFragmentTests,
        Stage::Transfer,
        HazardFlags::empty(),
    );
    gpu.cmd_copy_texture_to_buffer(cb, color.texture, color_rb.cast());
    gpu.cmd_copy_texture_to_buffer(cb, depth.texture, depth_rb.cast());
    gpu.queue_submit(Queue::Main, &[cb]);
    gpu.queue_wait_idle(Queue::Main);

    let px = |x: u32, y: u32| -> [u8; 4] {
        unsafe {
            let p = color_rb.cpu.add(((y * W + x) * 4) as usize);
            [*p, *p.add(1), *p.add(2), *p.add(3)]
        }
    };
    let dz = |x: u32, y: u32| -> f32 { unsafe { *depth_rb.cpu.add((y * W + x) as usize) } };

    // Overlap retains the nearer color and depth.
    let overlap = px(41, 41);
    assert!(
        overlap[0] > 200 && overlap[2] < 50,
        "overlap not red: {overlap:?} — far draw beat near draw (depth test broken)"
    );
    assert!(
        (dz(41, 41) - 0.5).abs() < 1e-6,
        "overlap depth {} != 0.5",
        dz(41, 41)
    );

    // The blue-only area passes against the 0.0 far clear and writes 0.25.
    let solo = px(57, 41);
    assert!(
        solo[2] > 200 && solo[0] < 50,
        "B-only region not blue: {solo:?} — the far draw didn't render at all"
    );
    assert!(
        (dz(57, 41) - 0.25).abs() < 1e-6,
        "B-only depth {} != 0.25",
        dz(57, 41)
    );

    // Untouched corners verify color and depth clear values.
    assert_eq!(px(2, 2), [0, 0, 0, 0], "corner color should be clear");
    assert_eq!(dz(2, 2), 0.0, "corner depth should be the 0.0 far clear");

    gpu.texture_free_and_destroy(color);
    gpu.texture_free_and_destroy(depth);
    gpu.shader_destroy(vert);
    gpu.shader_destroy(frag);
    for p in [
        pos_a.cast::<u8>(),
        pos_b.cast(),
        tint_a.cast(),
        tint_b.cast(),
        tri_a.cast(),
        tri_b.cast(),
        frag_a.cast(),
        frag_b.cast(),
        indices.cast(),
        color_rb.cast(),
        depth_rb.cast(),
    ] {
        gpu.mem_free_raw(p);
    }
}
