//! Depth-parity tooth for generic silhouettes: two overlapping outlined
//! cubes share one compacted batch, but only the front visible surface may
//! write the R8 mask under depth Equal.

mod common;

use abi_core::glam::{Mat4, UVec2, Vec3};
use abi_mesh::mesh_world_to_clip;
use common::{TestFrameAlloc, gpu_test_lock, view};
use gpu::{Gpu, HazardFlags, Memory, Queue, Stage, TextureDesc, TextureFormat, UsageFlags};
use mesh::cull::ClusterCullPass;
use mesh::primitives::cube;
use mesh::{
    MaterialEntry, MeshDepthPrepass, MeshRasterView, MeshScene, MeshSceneDesc, MeshSilhouettePass,
};

#[test]
fn silhouette_equal_writes_only_front_visible_group() {
    const W: u32 = 65;
    const H: u32 = 65;
    let _gpu_guard = gpu_test_lock();
    let gpu = Gpu::new(true).expect("vulkan init");
    let size = UVec2::new(W, H);

    let depth = gpu.texture_alloc_and_create(
        TextureDesc {
            dimensions: [W, H, 1],
            format: TextureFormat::D32Float,
            usage: UsageFlags::DEPTH_STENCIL_ATTACHMENT,
            ..Default::default()
        },
        Queue::Main,
        None,
    );
    let mask = gpu.texture_alloc_and_create(
        TextureDesc {
            dimensions: [W, H, 1],
            format: TextureFormat::R8Unorm,
            usage: UsageFlags::COLOR_ATTACHMENT | UsageFlags::TRANSFER_SRC,
            ..Default::default()
        },
        Queue::Main,
        None,
    );
    let mut scene = MeshScene::new(
        &gpu,
        &MeshSceneDesc {
            max_meshes: 1,
            max_instances: 2,
            max_materials: 1,
            vertex_capacity: 64,
            joint_weight_capacity: 0,
            index_capacity: 256,
            max_meshlets: 8,
        },
    );
    let cube_mesh = scene.add_mesh(&gpu, cube(1.0).desc());
    let material = scene.add_material(&gpu, MaterialEntry::standard());
    let front = scene.add_instance(&gpu, cube_mesh, Mat4::IDENTITY, material);
    let rear = scene.add_instance(
        &gpu,
        cube_mesh,
        Mat4::from_translation(Vec3::new(0.0, 0.0, 2.0)),
        material,
    );
    scene.set_outline_group(&gpu, front, 3);
    scene.set_outline_group(&gpu, rear, 2);

    let camera = view(size);
    let raster_view = MeshRasterView {
        world_to_clip: mesh_world_to_clip(&camera),
    };
    let cull = ClusterCullPass::new(&gpu, &scene, 1);
    let mut prepass = MeshDepthPrepass::new(&gpu);
    prepass.resize(&gpu, size);
    let silhouette = MeshSilhouettePass::new(&gpu);
    let mut frame_alloc = TestFrameAlloc {
        gpu: &gpu,
        ptrs: Vec::new(),
    };
    let readback = gpu.alloc_slice::<u8>((W * H) as u64, Memory::Readback);

    let cb = gpu.commands_begin(Queue::Main);
    cull.record(
        &gpu,
        cb,
        &mut frame_alloc,
        &scene,
        scene.instances(),
        raster_view,
        Vec3::from_array(camera.camera_position),
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
    silhouette.record(
        &gpu,
        cb,
        &mut frame_alloc,
        &scene,
        scene.instances(),
        cull.output(),
        cull.clusters(),
        cull.draw_count_ptr(&gpu, 0),
        mask.texture,
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
    gpu.cmd_copy_texture_to_buffer(cb, mask.texture, readback.cast());
    gpu.queue_submit(Queue::Main, &[cb]);
    gpu.queue_wait_idle(Queue::Main);

    let center = unsafe { *readback.cpu.add((H / 2 * W + W / 2) as usize) };
    assert_eq!(center, 3, "front group's visible center must own the mask");
    let mask_bytes = unsafe { std::slice::from_raw_parts(readback.cpu, (W * H) as usize) };
    assert!(
        mask_bytes.iter().any(|&value| value == 3),
        "front cube never masked"
    );
    assert!(
        mask_bytes.iter().all(|&value| value != 2),
        "occluded rear surfaces entered the silhouette mask"
    );

    frame_alloc.free();
    silhouette.free(&gpu);
    prepass.free(&gpu);
    cull.free(&gpu);
    scene.free(&gpu);
    gpu.texture_free_and_destroy(depth);
    gpu.texture_free_and_destroy(mask);
    gpu.free(readback);
}
