//! Cluster culling and indirect `first_instance` verification.

mod common;

use abi_core::glam::{Mat4, UVec2, Vec3, Vec4};
use abi_mesh::{
    ClusterInstance, MESH_FLAG_HIDDEN, MESH_FLAG_SKINNED, extract_frustum_planes, max_world_scale,
    mesh_world_to_clip, meshlet_backfacing_to_camera, sphere_inside_planes,
};
use common::{TestFrameAlloc, gpu_test_lock, mesh_heap, view};
use gpu::{Gpu, HazardFlags, LoadOp, Memory, Queue, Stage, TextureDesc, TextureFormat, UsageFlags};
use mesh::cull::{CONE_CULL_EPSILON, ClusterCullPass};
use mesh::primitives::cube;
use mesh::{
    JointWeights, MaterialEntry, MeshDepthPrepass, MeshForwardPass, MeshForwardTargets,
    MeshRasterView, MeshScene, MeshSceneDesc, MeshShadeLighting,
};

#[test]
fn cluster_cull_compacts_exclusive_ranges_and_first_instance_reaches_cluster() {
    const W: u32 = 65;
    const H: u32 = 65;
    let _gpu_guard = gpu_test_lock();
    let gpu = Gpu::new(true).expect("vulkan init");
    let size = UVec2::new(W, H);
    let mut scene = MeshScene::new(
        &gpu,
        &MeshSceneDesc {
            max_meshes: 1,
            max_instances: 3,
            max_materials: 2,
            vertex_capacity: 64,
            joint_weight_capacity: 0,
            index_capacity: 256,
            max_meshlets: 8,
        },
    );
    let cube_mesh = scene.add_mesh(&gpu, cube(1.0).desc());
    let mut red = MaterialEntry::standard();
    red.base_color_factor = [1.0, 0.0, 0.0, 1.0];
    let red = scene.add_material(&gpu, red);
    let mut green = MaterialEntry::standard();
    green.base_color_factor = [0.0, 1.0, 0.0, 1.0];
    let green = scene.add_material(&gpu, green);
    // Batch 1 starts at a nonzero compacted-cluster base.
    scene.add_instance(
        &gpu,
        cube_mesh,
        Mat4::from_translation(Vec3::new(-3.0, 0.0, 1.0)),
        red,
    );
    scene.add_instance(&gpu, cube_mesh, Mat4::IDENTITY, green);
    let hidden = scene.add_instance(
        &gpu,
        cube_mesh,
        Mat4::from_translation(Vec3::new(3.0, 0.0, 1.0)),
        red,
    );
    scene.set_flags(&gpu, hidden, MESH_FLAG_HIDDEN);
    assert_eq!(
        scene.batch_count(),
        2,
        "color/material differs but only material splits batches"
    );
    assert_eq!(scene.batches_cpu()[1].cluster_base, 2);

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
            usage: UsageFlags::DEPTH_STENCIL_ATTACHMENT,
            ..Default::default()
        },
        Queue::Main,
        None,
    );
    let view = view(size);
    let raster_view = MeshRasterView {
        world_to_clip: mesh_world_to_clip(&view),
    };
    let planes = extract_frustum_planes(&raster_view.world_to_clip);
    let camera_pos = Vec3::from_array(view.camera_position);

    // CPU twin mirrors shader visibility and batch compaction.
    let mut expected_counts = [0u32; 2];
    for (instance_id, instance) in scene.instances_cpu().iter().enumerate() {
        if instance.flags & MESH_FLAG_HIDDEN != 0 {
            continue;
        }
        let batch = scene.batches_cpu()[instance.batch_index as usize];
        let mesh = scene.mesh_data_cpu()[batch.mesh_index as usize];
        let transform = scene.transforms_cpu()[instance.transform_index as usize];
        for local_meshlet in 0..mesh.meshlet_count {
            let meshlet_index = mesh.meshlet_offset + local_meshlet;
            let meshlet = scene.meshlets_cpu()[meshlet_index as usize];
            let c = meshlet.center;
            let center = (transform.model_to_world * Vec4::new(c[0], c[1], c[2], 1.0)).truncate();
            let radius = meshlet.radius * max_world_scale(&transform.model_to_world)
                + instance.bounds_dilation;
            if instance.deformer_slot == 0
                && meshlet_backfacing_to_camera(
                    &meshlet,
                    &transform.model_to_world_normal,
                    camera_pos,
                    center,
                    CONE_CULL_EPSILON,
                )
            {
                continue;
            }
            if sphere_inside_planes(center, radius, &planes) {
                expected_counts[instance.batch_index as usize] += 1;
            }
        }
        assert!(instance_id < scene.instance_count() as usize);
    }

    let cull = ClusterCullPass::new(&gpu, &scene, 1);
    let mut prepass = MeshDepthPrepass::new(&gpu);
    prepass.resize(&gpu, size);
    let forward = MeshForwardPass::new(&gpu);
    let (heap, ramp_default_sampler) = mesh_heap(&gpu);
    let mut frame_alloc = TestFrameAlloc {
        gpu: &gpu,
        ptrs: Vec::new(),
    };
    let args_rb = gpu.alloc_slice::<abi_mesh::IndirectData>(2, Memory::Readback);
    let clusters_rb = gpu.alloc_slice::<ClusterInstance>(3, Memory::Readback);
    let color_rb = gpu.alloc_slice::<[f32; 4]>((W * H) as u64, Memory::Readback);

    let cb = gpu.commands_begin(Queue::Main);
    cull.record(
        &gpu,
        cb,
        &mut frame_alloc,
        &scene,
        scene.instances(),
        raster_view,
        camera_pos,
        0,
    );
    gpu.cmd_barrier(cb, Stage::Compute, Stage::Transfer, HazardFlags::empty());
    gpu.cmd_mem_copy_raw(
        cb,
        args_rb.cast(),
        cull.output().cast(),
        2 * core::mem::size_of::<abi_mesh::IndirectData>() as u64,
    );
    gpu.cmd_mem_copy_raw(
        cb,
        clusters_rb.cast(),
        cull.clusters_output().cast(),
        3 * core::mem::size_of::<ClusterInstance>() as u64,
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
    forward.record(
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
            color_load_op: LoadOp::Clear,
            clear_color: [0.02, 0.03, 0.04, 1.0],
        },
        raster_view,
        camera_pos,
        MeshShadeLighting {
            sun_direction: Vec3::NEG_Z.to_array(),
            sun_tint: [1.0; 3],
            sky_ambient: [0.0; 3],
            ground_ambient: [0.0; 3],
            ..MeshShadeLighting::zeroed()
        },
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

    cull.assert_counts(0, scene.batches_cpu());
    assert_eq!(cull.visible_count(0, 0), expected_counts[0]);
    assert_eq!(cull.visible_count(0, 1), expected_counts[1]);
    // SAFETY: queue idle and readback allocations have exactly two entries.
    let args = unsafe { std::slice::from_raw_parts(args_rb.cpu, 2) };
    assert_eq!(args[0].cmd.first_instance, 0);
    assert_eq!(
        args[1].cmd.first_instance, 2,
        "second batch must start after first exclusive range"
    );
    assert_eq!(args[1].batch_index, 1);
    // SAFETY: queue idle and the compacted table has three output slots.
    let clusters = unsafe { std::slice::from_raw_parts(clusters_rb.cpu, 3) };
    assert_eq!(
        clusters[2].instance_id, 1,
        "batch-one cluster must name green source instance"
    );
    let center = unsafe { *color_rb.cpu.add((H / 2 * W + W / 2) as usize) };
    assert!(
        center[1] > 0.9 && center[0] < 0.05,
        "nonzero first_instance looked up wrong cluster: {center:?}"
    );

    frame_alloc.free();
    forward.free(&gpu);
    prepass.free(&gpu);
    cull.free(&gpu);
    scene.free(&gpu);
    gpu.texture_free_and_destroy(color);
    gpu.texture_free_and_destroy(depth);
    gpu.free(args_rb);
    gpu.free(clusters_rb);
    gpu.free(color_rb);
    heap.free(&gpu);
}

#[test]
fn skinned_clusters_use_conservatively_dilated_rest_spheres() {
    const DILATION: f32 = 4.0;

    let _gpu_guard = gpu_test_lock();
    let gpu = Gpu::new(true).expect("vulkan init");
    let cube = cube(1.0);
    let joint_weights = vec![
        JointWeights {
            joint_indices: [0; 4],
            weights: [1.0, 0.0, 0.0, 0.0],
        };
        cube.positions.len()
    ];
    let mut scene = MeshScene::new(
        &gpu,
        &MeshSceneDesc {
            max_meshes: 2,
            max_instances: 3,
            max_materials: 1,
            vertex_capacity: (cube.positions.len() * 2) as u32,
            joint_weight_capacity: joint_weights.len() as u32,
            index_capacity: 512,
            max_meshlets: 16,
        },
    );
    let static_mesh = scene.add_mesh(&gpu, cube.desc());
    let weighted_mesh = scene.add_mesh(
        &gpu,
        mesh::MeshDesc {
            joint_weights: Some(&joint_weights),
            ..cube.desc()
        },
    );
    let material = scene.add_material(&gpu, MaterialEntry::standard());
    scene.add_instance(
        &gpu,
        static_mesh,
        Mat4::from_translation(Vec3::new(8.0, 0.0, 0.0)),
        material,
    );
    let reaching_skin = scene.add_instance(
        &gpu,
        weighted_mesh,
        Mat4::from_translation(Vec3::new(8.0, 0.0, 0.0)),
        material,
    );
    let farther_skin = scene.add_instance(
        &gpu,
        weighted_mesh,
        Mat4::from_translation(Vec3::new(14.0, 0.0, 0.0)),
        material,
    );
    for instance in [reaching_skin, farther_skin] {
        scene.set_flags(&gpu, instance, MESH_FLAG_SKINNED);
        scene.set_bounds_dilation_direct(&gpu, instance, DILATION);
    }
    assert_eq!(scene.batch_count(), 2);

    let raster_view = MeshRasterView {
        world_to_clip: mesh_world_to_clip(&view(UVec2::new(65, 65))),
    };
    let planes = extract_frustum_planes(&raster_view.world_to_clip);
    let mut expected = [0u32; 2];
    for (instance_index, instance) in scene.instances_cpu().iter().enumerate() {
        let batch = scene.batches_cpu()[instance.batch_index as usize];
        let mesh = scene.mesh_data_cpu()[batch.mesh_index as usize];
        let transform = scene.transforms_cpu()[instance.transform_index as usize];
        for local_meshlet in 0..mesh.meshlet_count {
            let meshlet = scene.meshlets_cpu()[(mesh.meshlet_offset + local_meshlet) as usize];
            let center = (transform.model_to_world
                * Vec4::new(meshlet.center[0], meshlet.center[1], meshlet.center[2], 1.0))
            .truncate();
            let rest_radius = meshlet.radius * max_world_scale(&transform.model_to_world);
            assert!(
                !sphere_inside_planes(center, rest_radius, &planes),
                "all three candidates must be outside on rest bounds"
            );
            let survives =
                sphere_inside_planes(center, rest_radius + instance.bounds_dilation, &planes);
            match instance_index {
                0 => assert!(!survives, "the static off-frustum candidate must cull"),
                1 => assert!(
                    survives,
                    "the displaced candidate's dilated sphere must reach the frustum"
                ),
                2 => assert!(
                    !survives,
                    "the farther skinned candidate must remain beyond the same dilation"
                ),
                _ => unreachable!(),
            }
            if survives {
                expected[instance.batch_index as usize] += 1;
            }
        }
    }
    assert_eq!(expected[0], 0);
    assert_eq!(expected[1], scene.mesh_data(weighted_mesh).meshlet_count);

    let cull = ClusterCullPass::new(&gpu, &scene, 1);
    let mut frame_alloc = TestFrameAlloc {
        gpu: &gpu,
        ptrs: Vec::new(),
    };
    let cb = gpu.commands_begin(Queue::Main);
    cull.record(
        &gpu,
        cb,
        &mut frame_alloc,
        &scene,
        scene.instances(),
        raster_view,
        Vec3::new(0.0, 0.0, -8.0),
        0,
    );
    gpu.queue_submit(Queue::Main, &[cb]);
    gpu.queue_wait_idle(Queue::Main);

    assert_eq!(cull.visible_count(0, 0), expected[0]);
    assert_eq!(cull.visible_count(0, 1), expected[1]);
    cull.assert_counts(0, scene.batches_cpu());

    frame_alloc.free();
    cull.free(&gpu);
    scene.free(&gpu);
}

/// Verifies rasterization across multiple meshlets and a partial tail.
#[test]
fn ragged_meshlet_padding_rasters_all_real_triangles_and_nothing_else() {
    const W: u32 = 65;
    const H: u32 = 65;
    const QUADS: u32 = 9;
    let _gpu_guard = gpu_test_lock();
    let gpu = Gpu::new(true).expect("vulkan init");
    let size = UVec2::new(W, H);

    // This geometry exceeds one meshlet and leaves a partial tail.
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut uvs = Vec::new();
    for y in 0..=QUADS {
        for x in 0..=QUADS {
            let fx = x as f32 / QUADS as f32 * 2.0 - 1.0;
            let fy = y as f32 / QUADS as f32 * 2.0 - 1.0;
            positions.push([fx, fy, 0.0]);
            normals.push([0.0, 0.0, -1.0]);
            uvs.push([x as f32 / QUADS as f32, y as f32 / QUADS as f32]);
        }
    }
    let mut indices = Vec::new();
    for y in 0..QUADS {
        for x in 0..QUADS {
            let base = y * (QUADS + 1) + x;
            // Winding points the geometric normal toward the camera.
            indices.extend_from_slice(&[
                base,
                base + QUADS + 1,
                base + 1,
                base + 1,
                base + QUADS + 1,
                base + QUADS + 2,
            ]);
        }
    }

    let mut scene = MeshScene::new(
        &gpu,
        &MeshSceneDesc {
            max_meshes: 1,
            max_instances: 1,
            max_materials: 1,
            vertex_capacity: 128,
            joint_weight_capacity: 0,
            index_capacity: 1024,
            max_meshlets: 8,
        },
    );
    let plane = scene.add_mesh(
        &gpu,
        mesh::MeshDesc {
            positions: &positions,
            normals: &normals,
            uvs: &uvs,
            indices: &indices,
            tangents: None,
            joint_weights: None,
            colors: None,
        },
    );
    let mut white = MaterialEntry::standard();
    white.base_color_factor = [1.0, 1.0, 1.0, 1.0];
    let white = scene.add_material(&gpu, white);
    scene.add_instance(&gpu, plane, Mat4::IDENTITY, white);

    // Require multiple meshlets and a shorter final cluster.
    let mesh_data = scene.mesh_data_cpu()[0];
    assert!(
        mesh_data.meshlet_count >= 2,
        "plane must split into multiple meshlets, got {}",
        mesh_data.meshlet_count
    );
    let max_tris = (0..mesh_data.meshlet_count)
        .map(|m| scene.meshlets_cpu()[(mesh_data.meshlet_offset + m) as usize].tri_count)
        .max()
        .unwrap();
    let min_tris = (0..mesh_data.meshlet_count)
        .map(|m| scene.meshlets_cpu()[(mesh_data.meshlet_offset + m) as usize].tri_count)
        .min()
        .unwrap();
    assert_eq!(mesh_data.cluster_vertex_count, max_tris * 3);
    assert!(
        min_tris < max_tris,
        "no ragged meshlet ({min_tris} == {max_tris}); grow or shrink QUADS so the tail pads"
    );

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
            usage: UsageFlags::DEPTH_STENCIL_ATTACHMENT,
            ..Default::default()
        },
        Queue::Main,
        None,
    );
    let view = view(size);
    let raster_view = MeshRasterView {
        world_to_clip: mesh_world_to_clip(&view),
    };

    let cull = ClusterCullPass::new(&gpu, &scene, 1);
    let mut prepass = MeshDepthPrepass::new(&gpu);
    prepass.resize(&gpu, size);
    let forward = MeshForwardPass::new(&gpu);
    let (heap, ramp_default_sampler) = mesh_heap(&gpu);
    let mut frame_alloc = TestFrameAlloc {
        gpu: &gpu,
        ptrs: Vec::new(),
    };
    let color_rb = gpu.alloc_slice::<[f32; 4]>((W * H) as u64, Memory::Readback);

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
    forward.record(
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
            color_load_op: LoadOp::Clear,
            clear_color: [0.0, 0.0, 0.0, 1.0],
        },
        raster_view,
        Vec3::from_array(view.camera_position),
        MeshShadeLighting {
            sun_direction: Vec3::NEG_Z.to_array(),
            sun_tint: [1.0; 3],
            sky_ambient: [1.0; 3],
            ground_ambient: [1.0; 3],
            ..MeshShadeLighting::zeroed()
        },
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

    // All meshlets survive the head-on view.
    assert_eq!(cull.visible_count(0, 0), mesh_data.meshlet_count);

    // Project corners with the GPU matrix and allow raster edge rounding.
    let corner = raster_view.world_to_clip * Vec4::new(1.0, 1.0, 0.0, 1.0);
    let ndc_extent = (corner.x / corner.w).abs();
    let half_px = ndc_extent * 0.5 * W as f32;
    let center = (W as f32) * 0.5;
    let lit = |x: u32, y: u32| -> bool {
        // SAFETY: queue idle; readback holds exactly W*H texels.
        let texel = unsafe { *color_rb.cpu.add((y * W + x) as usize) };
        texel[0] > 0.5
    };
    let mut interior = 0u32;
    for y in 0..H {
        for x in 0..W {
            let dx = (x as f32 + 0.5 - center).abs();
            let dy = (y as f32 + 0.5 - center).abs();
            let inside = dx < half_px - 1.5 && dy < half_px - 1.5;
            let outside = dx > half_px + 1.5 || dy > half_px + 1.5;
            if inside {
                interior += 1;
                assert!(
                    lit(x, y),
                    "hole at ({x},{y}): a ragged-tail triangle failed to raster"
                );
            } else if outside {
                assert!(
                    !lit(x, y),
                    "stray pixel at ({x},{y}): padding rastered something"
                );
            }
        }
    }
    assert!(
        interior > 64,
        "probe rectangle too small ({interior} px) to prove anything"
    );

    frame_alloc.free();
    forward.free(&gpu);
    prepass.free(&gpu);
    cull.free(&gpu);
    scene.free(&gpu);
    gpu.texture_free_and_destroy(color);
    gpu.texture_free_and_destroy(depth);
    gpu.free(color_rb);
    heap.free(&gpu);
}
