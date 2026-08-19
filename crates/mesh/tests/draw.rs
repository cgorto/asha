mod common;

use abi_core::glam::{Mat4, UVec2, Vec3};
use abi_light::mesh_shade_slim;
use abi_mesh::mesh_world_to_clip;
use common::{TestFrameAlloc, gpu_test_lock, mesh_heap, view};
use gpu::{
    Gpu, HazardFlags, LoadOp, Memory, Queue, RenderAttachment, RenderPassDesc, Stage, StoreOp,
    TextureDesc, TextureFormat, UsageFlags,
};
use mesh::cull::ClusterCullPass;
use mesh::primitives::cube;
use mesh::{
    MaterialEntry, MeshDepthPrepass, MeshForwardPass, MeshForwardTargets, MeshRasterView,
    MeshScene, MeshSceneDesc, MeshShadeLighting,
};

fn scene_desc() -> MeshSceneDesc {
    MeshSceneDesc {
        max_meshes: 1,
        max_instances: 1,
        max_materials: 1,
        vertex_capacity: 64,
        joint_weight_capacity: 0,
        index_capacity: 256,
        max_meshlets: 8,
    }
}

/// Verifies prepass depth ownership and forward `Equal` shading.
#[test]
fn forward_pass_draws_cube_color_and_depth() {
    const W: u32 = 65;
    const H: u32 = 65;
    let _gpu_guard = gpu_test_lock();
    let gpu = Gpu::new(true).expect("vulkan init");
    let size = UVec2::new(W, H);

    let color = gpu.texture_alloc_and_create(
        TextureDesc {
            dimensions: [W, H, 1],
            format: TextureFormat::Rgba32Float,
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

    let mut scene = MeshScene::new(&gpu, &scene_desc());
    let mesh = scene.add_mesh(&gpu, cube(1.0).desc());
    let mut material = MaterialEntry::standard();
    material.base_color_factor = [0.8, 0.4, 0.2, 1.0];
    // Nonzero emissive isolates the additive term.
    material.emissive_factor = [0.3, 0.5, 0.7];
    let material = scene.add_material(&gpu, material);
    scene.add_instance(&gpu, mesh, Mat4::IDENTITY, material);

    let lighting = MeshShadeLighting {
        sun_direction: Vec3::NEG_Z.to_array(),
        sun_tint: [1.0, 0.75, 0.5],
        sky_ambient: [0.2, 0.25, 0.3],
        ground_ambient: [0.05, 0.04, 0.03],
        ..MeshShadeLighting::zeroed()
    };
    let view = view(size);
    let raster_view = MeshRasterView {
        world_to_clip: mesh_world_to_clip(&view),
    };
    let mut prepass = MeshDepthPrepass::new(&gpu);
    prepass.resize(&gpu, size);
    let pass = MeshForwardPass::new(&gpu);
    let cull = ClusterCullPass::new(&gpu, &scene, 1);
    let (heap, ramp_default_sampler) = mesh_heap(&gpu);
    let mut frame_alloc = TestFrameAlloc {
        gpu: &gpu,
        ptrs: Vec::new(),
    };

    let color_rb = gpu.alloc_slice::<[f32; 4]>((W * H) as u64, Memory::Readback);
    let depth_rb = gpu.alloc_slice::<f32>((W * H) as u64, Memory::Readback);

    let cb = gpu.commands_begin(Queue::Main);
    gpu.cmd_begin_render_pass(
        cb,
        RenderPassDesc {
            // The prepass owns depth clearing.
            color_attachments: &[RenderAttachment {
                texture: color.texture,
                load_op: LoadOp::Clear,
                store_op: StoreOp::Store,
                clear_color: [0.02, 0.03, 0.04, 1.0],
                ..Default::default()
            }],
            ..Default::default()
        },
    );
    gpu.cmd_end_render_pass(cb);
    gpu.cmd_barrier(
        cb,
        Stage::RasterColorOut,
        Stage::RasterColorOut,
        HazardFlags::empty(),
    );
    cull.record(
        &gpu,
        cb,
        &mut frame_alloc,
        &scene,
        scene.instances(),
        raster_view,
        Vec3::from_array(view.camera_position),
        0,
    );
    prepass.record(
        &gpu,
        cb,
        &mut frame_alloc,
        &scene,
        scene.instances(),
        cull.output(),
        cull.clusters(),
        cull.draw_count_ptr(&gpu, 0),
        depth.texture,
        size,
        raster_view,
    );
    pass.record(
        &gpu,
        cb,
        &mut frame_alloc,
        &heap,
        &scene,
        scene.instances(),
        cull.output(),
        cull.clusters(),
        cull.draw_count_ptr(&gpu, 0),
        MeshForwardTargets {
            color: color.texture,
            depth: depth.texture,
            size,
            color_load_op: LoadOp::Load,
            clear_color: [0.0; 4],
        },
        raster_view,
        Vec3::from_array(view.camera_position),
        lighting,
        ramp_default_sampler,
    );
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

    let pixel = |x: u32, y: u32| -> [f32; 4] {
        // SAFETY: readback covers W*H pixels.
        unsafe { *color_rb.cpu.add((y * W + x) as usize) }
    };
    let depth_at = |x: u32, y: u32| -> f32 {
        // SAFETY: readback covers W*H pixels.
        unsafe { *depth_rb.cpu.add((y * W + x) as usize) }
    };

    let cube_px = pixel(W / 2, H / 2);
    let shaded = mesh_shade_slim(Vec3::NEG_Z, [0.8, 0.4, 0.2], &lighting);
    // Shade plus the material's additive emissive (the term under test).
    let expected = [shaded[0] + 0.3, shaded[1] + 0.5, shaded[2] + 0.7];
    for c in 0..3 {
        assert!(
            (cube_px[c] - expected[c]).abs() < 2.0e-3,
            "cube channel {c}: gpu {} vs cpu {}",
            cube_px[c],
            expected[c]
        );
    }
    assert!((cube_px[3] - 1.0).abs() < 1.0e-6);

    let bg = pixel(2, 2);
    for (got, want) in bg.into_iter().zip([0.02, 0.03, 0.04, 1.0]) {
        assert!((got - want).abs() < 1.0e-6, "background {got} != {want}");
    }
    assert_eq!(depth_at(2, 2), 0.0);

    let front_center = Vec3::new(0.0, 0.0, -1.0);
    let clip = mesh_world_to_clip(&view) * front_center.extend(1.0);
    let expected_depth = clip.z / clip.w;
    let got_depth = depth_at(W / 2, H / 2);
    assert!(
        (got_depth - expected_depth).abs() < 1.0e-3,
        "cube depth {got_depth} != {expected_depth}"
    );

    frame_alloc.free();
    prepass.free(&gpu);
    pass.free(&gpu);
    cull.free(&gpu);
    scene.free(&gpu);
    gpu.texture_free_and_destroy(color);
    gpu.texture_free_and_destroy(depth);
    gpu.free(color_rb);
    gpu.free(depth_rb);
    heap.free(&gpu);
}

/// Verifies frame-local transform uploads reach culling and rasterization.
/// The second frame moves the cube behind the camera.
#[test]
fn stage_world_moves_the_draw_same_frame() {
    const W: u32 = 65;
    const H: u32 = 65;
    let _gpu_guard = gpu_test_lock();
    let gpu = Gpu::new(true).expect("vulkan init");
    let size = UVec2::new(W, H);

    let color = gpu.texture_alloc_and_create(
        TextureDesc {
            dimensions: [W, H, 1],
            format: TextureFormat::Rgba32Float,
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

    let mut scene = MeshScene::new(&gpu, &scene_desc());
    let mesh = scene.add_mesh(&gpu, cube(1.0).desc());
    let mut material = MaterialEntry::standard();
    material.base_color_factor = [0.8, 0.4, 0.2, 1.0];
    let material = scene.add_material(&gpu, material);
    let instance = scene.add_instance(&gpu, mesh, Mat4::IDENTITY, material);

    let lighting = MeshShadeLighting {
        sun_direction: Vec3::NEG_Z.to_array(),
        sun_tint: [1.0, 0.75, 0.5],
        sky_ambient: [0.2, 0.25, 0.3],
        ground_ambient: [0.05, 0.04, 0.03],
        ..MeshShadeLighting::zeroed()
    };
    let view = view(size);
    let raster_view = MeshRasterView {
        world_to_clip: mesh_world_to_clip(&view),
    };
    let mut prepass = MeshDepthPrepass::new(&gpu);
    prepass.resize(&gpu, size);
    let pass = MeshForwardPass::new(&gpu);
    let cull = ClusterCullPass::new(&gpu, &scene, 1);
    let (heap, ramp_default_sampler) = mesh_heap(&gpu);
    let mut frame_alloc = TestFrameAlloc {
        gpu: &gpu,
        ptrs: Vec::new(),
    };
    let staging = gpu.alloc::<mesh::DrawTransform>(Memory::Default);
    let color_rb = gpu.alloc_slice::<[f32; 4]>((W * H) as u64, Memory::Readback);
    const CLEAR: [f32; 4] = [0.02, 0.03, 0.04, 1.0];

    // Stage, cull, rasterize, and read back one frame.
    let mut frame = |frame_alloc: &mut TestFrameAlloc, world: Option<Mat4>| {
        let cb = gpu.commands_begin(Queue::Main);
        if let Some(world) = world {
            // Bracket the upload with required barriers.
            gpu.cmd_barrier(cb, Stage::All, Stage::Transfer, HazardFlags::empty());
            scene.stage_world(&gpu, cb, staging, instance, world);
            gpu.cmd_barrier(cb, Stage::Transfer, Stage::All, HazardFlags::empty());
        }
        gpu.cmd_begin_render_pass(
            cb,
            RenderPassDesc {
                color_attachments: &[RenderAttachment {
                    texture: color.texture,
                    load_op: LoadOp::Clear,
                    store_op: StoreOp::Store,
                    clear_color: CLEAR,
                    ..Default::default()
                }],
                ..Default::default()
            },
        );
        gpu.cmd_end_render_pass(cb);
        gpu.cmd_barrier(
            cb,
            Stage::RasterColorOut,
            Stage::RasterColorOut,
            HazardFlags::empty(),
        );
        cull.record(
            &gpu,
            cb,
            frame_alloc,
            &scene,
            scene.instances(),
            raster_view,
            Vec3::from_array(view.camera_position),
            0,
        );
        prepass.record(
            &gpu,
            cb,
            frame_alloc,
            &scene,
            scene.instances(),
            cull.output(),
            cull.clusters(),
            cull.draw_count_ptr(&gpu, 0),
            depth.texture,
            size,
            raster_view,
        );
        pass.record(
            &gpu,
            cb,
            frame_alloc,
            &heap,
            &scene,
            scene.instances(),
            cull.output(),
            cull.clusters(),
            cull.draw_count_ptr(&gpu, 0),
            MeshForwardTargets {
                color: color.texture,
                depth: depth.texture,
                size,
                color_load_op: LoadOp::Load,
                clear_color: [0.0; 4],
            },
            raster_view,
            Vec3::from_array(view.camera_position),
            lighting,
            ramp_default_sampler,
        );
        gpu.cmd_barrier(
            cb,
            Stage::RasterColorOut,
            Stage::Transfer,
            HazardFlags::empty(),
        );
        gpu.cmd_copy_texture_to_buffer(cb, color.texture, color_rb.cast());
        gpu.queue_submit(Queue::Main, &[cb]);
        gpu.queue_wait_idle(Queue::Main);
    };

    let center = |label: &str, want: [f32; 3], tolerance: f32| {
        // SAFETY: readback covers W*H pixels.
        let got = unsafe { *color_rb.cpu.add(((H / 2) * W + W / 2) as usize) };
        for c in 0..3 {
            assert!(
                (got[c] - want[c]).abs() < tolerance,
                "{label} channel {c}: gpu {} vs cpu {}",
                got[c],
                want[c]
            );
        }
    };

    frame(&mut frame_alloc, None);
    center(
        "cube at origin",
        mesh_shade_slim(Vec3::NEG_Z, [0.8, 0.4, 0.2], &lighting),
        2.0e-3,
    );

    // Behind-camera placement must cull every meshlet.
    let behind = Mat4::from_translation(Vec3::new(0.0, 0.0, -20.0));
    frame(&mut frame_alloc, Some(behind));
    center("cube staged away", [CLEAR[0], CLEAR[1], CLEAR[2]], 1.0e-6);
    let mirrored = scene.transforms_cpu()[instance.index() as usize];
    assert!(
        (mirrored.model_to_world * abi_core::glam::Vec4::new(0.0, 0.0, 0.0, 1.0)).z == -20.0,
        "CPU mirror must follow the staged transform"
    );

    frame_alloc.free();
    prepass.free(&gpu);
    pass.free(&gpu);
    cull.free(&gpu);
    scene.free(&gpu);
    gpu.texture_free_and_destroy(color);
    gpu.texture_free_and_destroy(depth);
    gpu.free(color_rb);
    gpu.free(staging);
    heap.free(&gpu);
}
