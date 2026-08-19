//! Hardware verification of depth and visibility-token prepass outputs.

mod common;

use abi_core::glam::{Mat4, UVec2, Vec3};
use abi_mesh::mesh_world_to_clip;
use common::{TestFrameAlloc, gpu_test_lock, view};
use gpu::{Gpu, HazardFlags, Memory, Queue, Stage, TextureDesc, TextureFormat, UsageFlags};
use mesh::cull::ClusterCullPass;
use mesh::primitives::cube;
use mesh::{MaterialEntry, MeshDepthPrepass, MeshRasterView, MeshScene, MeshSceneDesc};

#[test]
fn prepass_writes_depth_and_visibility_tokens() {
    const W: u32 = 65;
    const H: u32 = 65;
    let _gpu_guard = gpu_test_lock();
    let gpu = Gpu::new(true).expect("vulkan init");
    let size = UVec2::new(W, H);

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

    let mut scene = MeshScene::new(
        &gpu,
        &MeshSceneDesc {
            max_meshes: 1,
            max_instances: 1,
            max_materials: 1,
            vertex_capacity: 64,
            joint_weight_capacity: 0,
            index_capacity: 256,
            max_meshlets: 8,
        },
    );
    let mesh = scene.add_mesh(&gpu, cube(1.0).desc());
    let material = scene.add_material(&gpu, MaterialEntry::standard());
    scene.add_instance(&gpu, mesh, Mat4::IDENTITY, material);

    let view = view(size);
    let raster_view = MeshRasterView {
        world_to_clip: mesh_world_to_clip(&view),
    };
    let mut prepass = MeshDepthPrepass::new(&gpu);
    prepass.resize(&gpu, size);
    let cull = ClusterCullPass::new(&gpu, &scene, 1);
    let mut frame_alloc = TestFrameAlloc {
        gpu: &gpu,
        ptrs: Vec::new(),
    };

    let depth_rb = gpu.alloc_slice::<f32>((W * H) as u64, Memory::Readback);
    let token_rb = gpu.alloc_slice::<u32>((W * H) as u64, Memory::Readback);

    let cb = gpu.commands_begin(Queue::Main);
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
    gpu.cmd_copy_texture_to_buffer(cb, depth.texture, depth_rb.cast());
    gpu.cmd_copy_texture_to_buffer(cb, prepass.visibility_texture(), token_rb.cast());
    gpu.queue_submit(Queue::Main, &[cb]);
    gpu.queue_wait_idle(Queue::Main);

    let depth_at = |x: u32, y: u32| -> f32 {
        // SAFETY: readback covers W*H pixels.
        unsafe { *depth_rb.cpu.add((y * W + x) as usize) }
    };
    let token_at = |x: u32, y: u32| -> u32 {
        // SAFETY: readback covers W*H pixels.
        unsafe { *token_rb.cpu.add((y * W + x) as usize) }
    };

    // Probe depth using the same world-to-clip transform.
    let front_center = Vec3::new(0.0, 0.0, -1.0);
    let clip = mesh_world_to_clip(&view) * front_center.extend(1.0);
    let expected_depth = clip.z / clip.w;
    let got_depth = depth_at(W / 2, H / 2);
    assert!(
        (got_depth - expected_depth).abs() < 1.0e-3,
        "cube depth {got_depth} != {expected_depth}"
    );
    assert_eq!(
        depth_at(2, 2),
        0.0,
        "background depth must stay the far clear"
    );

    // Decode the cluster index and meshlet-local primitive ID.
    let cube_token = token_at(W / 2, H / 2);
    assert_eq!(
        cube_token & 0x01FF_FFFF,
        1,
        "cube pixel draw token: got {cube_token:#010x}"
    );
    assert!(
        cube_token >> 25 <= 0x7F,
        "primitive id field overflows 7 bits"
    );
    // Background is the all-zero sky sentinel.
    assert_eq!(
        token_at(2, 2),
        0,
        "background token must stay the sky sentinel"
    );

    frame_alloc.free();
    prepass.free(&gpu);
    cull.free(&gpu);
    scene.free(&gpu);
    gpu.texture_free_and_destroy(depth);
    gpu.free(depth_rb);
    gpu.free(token_rb);
}
