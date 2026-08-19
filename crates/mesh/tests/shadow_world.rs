//! World-space TLAS/BLAS queries compared with an independent f64 oracle.

mod common;

use abi_core::glam::{Mat4, Quat, Vec3};
use abi_light::{
    SHADOW_QUERY_FAILED, SHADOW_QUERY_OCCLUDED, SHADOW_QUERY_VISIBLE, ShadowSegment,
    ShadowWorldQueryResult, shadow_segment_triangle_oracle,
};
use abi_mesh::world_transform;
use abi_mesh::{MESH_FLAG_HIDDEN, MESH_FLAG_NO_SHADOW, MaterialEntry};
use common::{TestFrameAlloc, gpu_test_lock};
use gpu::pass::Pass;
use gpu::{Gpu, Memory, Queue};
use mesh::primitives::{cube, icosphere};
use mesh::{
    MeshBuffers, MeshInstances, MeshScene, MeshSceneDesc, ShadowBlasDesc, ShadowTlasBuilder,
    ShadowWorldQueryPass,
};

fn random_unit(state: &mut u32) -> f32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    (*state as f32) * (1.0 / u32::MAX as f32)
}

fn random_point(state: &mut u32) -> Vec3 {
    Vec3::new(
        random_unit(state) * 24.0 - 12.0,
        random_unit(state) * 16.0 - 8.0,
        random_unit(state) * 20.0 - 10.0,
    )
}

fn oracle(scene: &MeshScene, meshes: &[&MeshBuffers], segment: &ShadowSegment) -> bool {
    for instance in scene.instances_cpu() {
        if instance.flags & (MESH_FLAG_HIDDEN | MESH_FLAG_NO_SHADOW) != 0 {
            continue;
        }
        let batch = scene.batches_cpu()[instance.batch_index as usize];
        let mesh = meshes[batch.mesh_index as usize];
        let transform = scene.transforms_cpu()[instance.transform_index as usize];
        let local = segment.transformed(transform.model_to_world.inverse());
        for triangle in mesh.indices.chunks_exact(3) {
            if shadow_segment_triangle_oracle(
                &local,
                Vec3::from_array(mesh.positions[triangle[0] as usize]),
                Vec3::from_array(mesh.positions[triangle[1] as usize]),
                Vec3::from_array(mesh.positions[triangle[2] as usize]),
            ) {
                return true;
            }
        }
    }
    false
}

fn run_queries(
    gpu: &Gpu,
    pass: &ShadowWorldQueryPass,
    frame: &mut TestFrameAlloc,
    builder: &mut ShadowTlasBuilder,
    scene: &MeshScene,
    meshes: &[&MeshBuffers],
    queries: &[ShadowSegment],
) -> (u32, u32) {
    let (world, stats) = builder.build(frame, scene);
    assert_eq!(stats.instance_count, 2);
    assert!(stats.node_count > 0);
    assert!(stats.max_depth < 32);

    let query_buffer = gpu.alloc_slice::<ShadowSegment>(queries.len() as u64, Memory::Default);
    let result_buffer =
        gpu.alloc_slice::<ShadowWorldQueryResult>(queries.len() as u64, Memory::Default);
    unsafe {
        std::ptr::copy_nonoverlapping(queries.as_ptr(), query_buffer.cpu, queries.len());
        for i in 0..queries.len() {
            result_buffer.cpu.add(i).write(ShadowWorldQueryResult {
                status: SHADOW_QUERY_FAILED,
                instance_id: u32::MAX,
                primitive_id: u32::MAX,
                hit_t: f32::NAN,
                tlas_node_tests: u32::MAX,
                blas_node_tests: u32::MAX,
                triangle_tests: u32::MAX,
                max_stack_depth: u32::MAX,
            });
        }
    }

    let cb = gpu.commands_begin(Queue::Main);
    pass.record(
        gpu,
        cb,
        frame,
        world,
        query_buffer.gpu,
        result_buffer.gpu,
        queries.len() as u32,
    );
    gpu.queue_submit(Queue::Main, &[cb]);
    gpu.queue_wait_idle(Queue::Main);

    let got = unsafe { std::slice::from_raw_parts(result_buffer.cpu, queries.len()) };
    let mut hits = 0;
    let mut max_stack = 0;
    for (i, (result, segment)) in got.iter().zip(queries).enumerate() {
        assert_ne!(
            result.status, SHADOW_QUERY_FAILED,
            "world query {i} failed: {result:?}"
        );
        let expected = oracle(scene, meshes, segment);
        assert_eq!(
            result.status == SHADOW_QUERY_OCCLUDED,
            expected,
            "world query {i} parity mismatch: {segment:?} -> {result:?}"
        );
        if expected {
            hits += 1;
            assert!(result.instance_id < scene.instance_count());
            assert!(result.hit_t > segment.t_min && result.hit_t < segment.t_max);
        } else {
            assert_eq!(result.status, SHADOW_QUERY_VISIBLE);
        }
        max_stack = max_stack.max(result.max_stack_depth);
    }

    gpu.free(query_buffer);
    gpu.free(result_buffer);
    (hits, max_stack)
}

#[test]
fn refitted_world_tlas_matches_affine_brute_force() {
    let _guard = gpu_test_lock();
    let gpu = Gpu::new(true).expect("vulkan init");
    let cube = cube(1.0);
    let sphere = icosphere(1.0, 1);
    let mut scene = MeshScene::new_with_shadows(
        &gpu,
        &MeshSceneDesc {
            max_meshes: 2,
            max_instances: 4,
            max_materials: 1,
            vertex_capacity: 256,
            joint_weight_capacity: 0,
            index_capacity: 1_024,
            max_meshlets: 8,
        },
        ShadowBlasDesc {
            node_capacity: 256,
            primitive_capacity: 256,
        },
    );
    let cube_mesh = scene.add_mesh(&gpu, cube.desc());
    let sphere_mesh = scene.add_mesh(&gpu, sphere.desc());
    let material = scene.add_material(&gpu, MaterialEntry::standard());
    let cube_instance = scene.add_instance(
        &gpu,
        cube_mesh,
        Mat4::from_scale_rotation_translation(
            Vec3::new(1.8, 0.6, 1.2),
            Quat::from_rotation_y(0.47) * Quat::from_rotation_x(-0.21),
            Vec3::new(-3.0, 0.5, 0.0),
        ),
        material,
    );
    let sphere_instance = scene.add_instance(
        &gpu,
        sphere_mesh,
        Mat4::from_scale_rotation_translation(
            Vec3::new(0.7, 1.6, 1.1),
            Quat::from_rotation_z(0.31),
            Vec3::new(3.0, -0.5, 0.5),
        ),
        material,
    );
    let no_shadow = scene.add_instance(&gpu, cube_mesh, Mat4::IDENTITY, material);
    scene.set_flags(&gpu, no_shadow, MESH_FLAG_NO_SHADOW);
    let hidden = scene.add_instance(
        &gpu,
        sphere_mesh,
        Mat4::from_translation(Vec3::new(0.0, 0.0, 3.0)),
        material,
    );
    scene.set_flags(&gpu, hidden, MESH_FLAG_HIDDEN);

    let mut queries = vec![
        ShadowSegment::between(
            Vec3::new(-3.0, 0.5, 4.0),
            Vec3::new(-3.0, 0.5, -4.0),
            1e-3,
            1e-3,
        ),
        ShadowSegment::between(
            Vec3::new(3.0, -0.5, 4.0),
            Vec3::new(3.0, -0.5, -4.0),
            1e-3,
            1e-3,
        ),
        // The origin cube is NO_SHADOW and must not turn this into a hit.
        ShadowSegment::between(
            Vec3::new(0.0, 0.0, 0.8),
            Vec3::new(0.0, 0.0, -0.8),
            1e-3,
            1e-3,
        ),
        ShadowSegment::between(
            Vec3::new(-10.0, 7.0, 6.0),
            Vec3::new(10.0, 7.0, -6.0),
            1e-3,
            1e-3,
        ),
    ];
    let mut rng = 0x6d2b_79f5;
    for _ in 0..508 {
        let start = random_point(&mut rng);
        let mut end = random_point(&mut rng);
        if start.distance_squared(end) < 1.0e-4 {
            end.x += 1.0;
        }
        queries.push(ShadowSegment::between(start, end, 1e-3, 2e-3));
    }

    let pass = ShadowWorldQueryPass::new(&gpu);
    let mut frame = TestFrameAlloc {
        gpu: &gpu,
        ptrs: Vec::new(),
    };
    let mut builder = ShadowTlasBuilder::new(4);
    let meshes = [&cube, &sphere];
    let (first_hits, first_stack) = run_queries(
        &gpu,
        &pass,
        &mut frame,
        &mut builder,
        &scene,
        &meshes,
        &queries,
    );

    scene.set_world(
        &gpu,
        sphere_instance,
        Mat4::from_scale_rotation_translation(
            Vec3::new(1.3, 0.55, 1.9),
            Quat::from_rotation_x(-0.6) * Quat::from_rotation_y(0.2),
            Vec3::new(6.5, 2.0, -1.5),
        ),
    );
    scene.set_world(
        &gpu,
        cube_instance,
        Mat4::from_scale_rotation_translation(
            Vec3::new(0.65, 2.1, 0.8),
            Quat::from_rotation_z(0.52),
            Vec3::new(-5.0, -1.0, 1.0),
        ),
    );
    let (second_hits, second_stack) = run_queries(
        &gpu,
        &pass,
        &mut frame,
        &mut builder,
        &scene,
        &meshes,
        &queries,
    );

    println!(
        "World-space queries: count={} first_hits={first_hits} second_hits={second_hits} max_stack={}",
        queries.len() * 2,
        first_stack.max(second_stack)
    );
    pass.free(&gpu);
    frame.free();
    scene.free(&gpu);
}

#[test]
fn streamed_instance_transform_drives_shadow_tlas() {
    let _guard = gpu_test_lock();
    let gpu = Gpu::new(true).expect("vulkan init");
    let cube = cube(1.0);
    let mut scene = MeshScene::new_with_shadows(
        &gpu,
        &MeshSceneDesc {
            max_meshes: 1,
            max_instances: 1,
            max_materials: 1,
            vertex_capacity: 32,
            joint_weight_capacity: 0,
            index_capacity: 512,
            max_meshlets: 2,
        },
        ShadowBlasDesc {
            node_capacity: 16,
            primitive_capacity: 16,
        },
    );
    let cube_mesh = scene.add_mesh(&gpu, cube.desc());
    let material = scene.add_material(&gpu, MaterialEntry::standard());
    scene.add_instance(
        &gpu,
        cube_mesh,
        Mat4::from_translation(Vec3::new(100.0, 0.0, 0.0)),
        material,
    );

    // Use the streamed transform; retained transforms would miss this query.
    let base = scene.instances();
    let streamed_transforms = [world_transform(Mat4::IDENTITY)];
    let streamed = MeshInstances {
        transforms_cpu: &streamed_transforms,
        ..base
    };
    let mut frame = TestFrameAlloc {
        gpu: &gpu,
        ptrs: Vec::new(),
    };
    let mut builder = ShadowTlasBuilder::new(1);
    let (world, stats) = builder.build_instances(&mut frame, &scene, streamed);
    assert_eq!(stats.instance_count, 1);
    assert!(stats.topology_rebuilt);

    let query = gpu.alloc::<ShadowSegment>(Memory::Default);
    let result = gpu.alloc::<ShadowWorldQueryResult>(Memory::Default);
    unsafe {
        *query.cpu = ShadowSegment::between(
            Vec3::new(0.0, 0.0, 2.0),
            Vec3::new(0.0, 0.0, -2.0),
            1.0e-3,
            1.0e-3,
        );
        *result.cpu = ShadowWorldQueryResult {
            status: SHADOW_QUERY_FAILED,
            instance_id: u32::MAX,
            primitive_id: u32::MAX,
            hit_t: f32::NAN,
            tlas_node_tests: 0,
            blas_node_tests: 0,
            triangle_tests: 0,
            max_stack_depth: 0,
        };
    }
    let pass = ShadowWorldQueryPass::new(&gpu);
    let cb = gpu.commands_begin(Queue::Main);
    pass.record(&gpu, cb, &mut frame, world, query.gpu, result.gpu, 1);
    gpu.queue_submit(Queue::Main, &[cb]);
    gpu.queue_wait_idle(Queue::Main);
    assert_eq!(unsafe { (*result.cpu).status }, SHADOW_QUERY_OCCLUDED);

    pass.free(&gpu);
    frame.free();
    gpu.free(query);
    gpu.free(result);
    scene.free(&gpu);
}
