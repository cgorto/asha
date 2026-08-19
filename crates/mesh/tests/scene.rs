mod common;

use std::mem::{size_of, size_of_val};
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};

use abi_core::glam::{Mat4, Quat, Vec3};
use common::gpu_test_lock;
use gpu::{Gpu, Memory, Queue};
use mesh::primitives::{cube, icosphere};
use mesh::{JointWeights, MaterialEntry, MeshScene, MeshSceneDesc, MeshTableEntry};

fn scene_desc() -> MeshSceneDesc {
    MeshSceneDesc {
        max_meshes: 4,
        max_instances: 4,
        max_materials: 4,
        vertex_capacity: 512,
        joint_weight_capacity: 0,
        index_capacity: 4096,
        max_meshlets: 64,
    }
}

#[test]
fn register_meshes_and_meshlets() {
    let _gpu_guard = gpu_test_lock();
    let gpu = Gpu::new(true).expect("vulkan init");
    let mut scene = MeshScene::new(&gpu, &scene_desc());
    let cube = cube(1.0);
    let sphere = icosphere(1.0, 2);

    let cube_handle = scene.add_mesh(&gpu, cube.desc());
    let sphere_handle = scene.add_mesh(&gpu, sphere.desc());

    let cube_data = scene.mesh_data(cube_handle);
    let sphere_data = scene.mesh_data(sphere_handle);
    assert_eq!(cube_data.first_index, 0);
    assert_eq!(
        sphere_data.first_index,
        cube_data.idx_count + scene.meshlet_index_count(cube_handle)
    );
    assert_eq!(
        sphere_data.meshlet_offset,
        cube_data.meshlet_offset + cube_data.meshlet_count
    );
    assert_eq!(
        scene.max_meshlets_per_mesh(),
        cube_data.meshlet_count.max(sphere_data.meshlet_count)
    );

    assert_meshlet_invariants(&scene, cube_handle, cube.indices.len() / 3);
    assert_meshlet_invariants(&scene, sphere_handle, sphere.indices.len() / 3);
    assert_global_index_readback(&gpu, &scene);

    scene.free(&gpu);
}

#[test]
fn static_mesh_keeps_joint_storage_zero_and_table_pointer_null() {
    let _gpu_guard = gpu_test_lock();
    let gpu = Gpu::new(true).expect("vulkan init");
    let mut scene = MeshScene::new(&gpu, &scene_desc());
    let cube = cube(1.0);
    scene.add_mesh(&gpu, cube.desc());

    assert!(scene.joint_weights_buffer().is_null());
    assert!(read_mesh_table_entry(&gpu, &scene).joint_weights.is_null());

    scene.free(&gpu);
}

#[test]
fn skinned_mesh_uploads_compact_joint_storage_and_publishes_it() {
    let _gpu_guard = gpu_test_lock();
    let gpu = Gpu::new(true).expect("vulkan init");
    let cube = cube(1.0);
    let joint_weights = (0..cube.positions.len())
        .map(|vertex| JointWeights {
            joint_indices: [vertex as u32 % 4, 0, 0, 0],
            weights: [1.0, 0.0, 0.0, 0.0],
        })
        .collect::<Vec<_>>();
    let desc = MeshSceneDesc {
        max_meshes: 1,
        max_instances: 0,
        max_materials: 0,
        vertex_capacity: cube.positions.len() as u32,
        // This scene's sole skinned mesh consumes exactly one compact row
        // per vertex; static streams do not reserve this pool implicitly.
        joint_weight_capacity: cube.positions.len() as u32,
        index_capacity: 256,
        max_meshlets: 8,
    };
    let mut scene = MeshScene::new(&gpu, &desc);
    scene.add_mesh(
        &gpu,
        mesh::MeshDesc {
            joint_weights: Some(&joint_weights),
            ..cube.desc()
        },
    );

    let entry = read_mesh_table_entry(&gpu, &scene);
    assert_eq!(
        entry.joint_weights.addr(),
        scene.joint_weights_buffer().gpu.addr()
    );
    let readback = gpu.alloc_slice::<JointWeights>(joint_weights.len() as u64, Memory::Readback);
    let cb = gpu.commands_begin(Queue::Main);
    gpu.cmd_mem_copy_raw(
        cb,
        readback.cast(),
        scene.joint_weights_buffer().cast(),
        size_of_val(joint_weights.as_slice()) as u64,
    );
    gpu.queue_submit(Queue::Main, &[cb]);
    gpu.queue_wait_idle(Queue::Main);
    // SAFETY: readback holds exactly `joint_weights.len()` initialized rows.
    let uploaded = unsafe { std::slice::from_raw_parts(readback.cpu, joint_weights.len()) };
    for (expected, uploaded) in joint_weights.iter().zip(uploaded) {
        assert_eq!(uploaded.joint_indices, expected.joint_indices);
        assert_eq!(uploaded.weights, expected.weights);
    }
    gpu.free(readback);

    scene.free(&gpu);
}

#[test]
fn weighted_static_weighted_upload_is_compact() {
    let _gpu_guard = gpu_test_lock();
    let gpu = Gpu::new(true).expect("vulkan init");
    let cube = cube(1.0);
    let first_weights = vec![
        JointWeights {
            joint_indices: [0, 3, 2, 1],
            weights: [1.0, 0.0, 0.0, 0.0],
        };
        cube.positions.len()
    ];
    let second_weights = vec![
        JointWeights {
            joint_indices: [4, 2, 1, 0],
            weights: [0.5, 0.25, 0.125, 0.125],
        };
        cube.positions.len()
    ];
    let mut desc = scene_desc();
    desc.max_meshes = 3;
    desc.vertex_capacity = (cube.positions.len() * 3) as u32;
    desc.joint_weight_capacity = (first_weights.len() + second_weights.len()) as u32;
    let mut scene = MeshScene::new(&gpu, &desc);
    scene.add_mesh(
        &gpu,
        mesh::MeshDesc {
            joint_weights: Some(&first_weights),
            ..cube.desc()
        },
    );
    scene.add_mesh(&gpu, cube.desc());
    scene.add_mesh(
        &gpu,
        mesh::MeshDesc {
            joint_weights: Some(&second_weights),
            ..cube.desc()
        },
    );

    let entries = read_mesh_table_entries::<3>(&gpu, &scene);
    assert_eq!(
        entries[0].joint_weights.addr(),
        scene.joint_weights_buffer().gpu.addr()
    );
    assert!(entries[1].joint_weights.is_null());
    assert_eq!(
        entries[2].joint_weights.addr(),
        entries[0].joint_weights.addr() + (first_weights.len() * size_of::<JointWeights>()) as u64,
        "the static mesh must consume no compact joint rows"
    );

    scene.free(&gpu);
}

#[test]
fn direct_mesh_ingestion_rejects_malformed_joint_weights() {
    let _gpu_guard = gpu_test_lock();
    let gpu = Gpu::new(true).expect("vulkan init");
    let cube = cube(1.0);
    let mut desc = scene_desc();
    desc.joint_weight_capacity = cube.positions.len() as u32;
    let mut scene = MeshScene::new(&gpu, &desc);
    let valid = JointWeights {
        joint_indices: [0; 4],
        weights: [1.0, 0.0, 0.0, 0.0],
    };

    for (bad, expected) in [
        ([f32::NAN, 0.0, 0.0, 0.0], "must be finite"),
        ([1.1, -0.1, 0.0, 0.0], "must be in [0, 1]"),
        ([0.5, 0.25, 0.0, 0.0], "sum must be within 1e-4"),
    ] {
        let mut weights = vec![valid; cube.positions.len()];
        weights[0].weights = bad;
        let panic = catch_unwind(AssertUnwindSafe(|| {
            scene.add_mesh(
                &gpu,
                mesh::MeshDesc {
                    joint_weights: Some(&weights),
                    ..cube.desc()
                },
            );
        }))
        .expect_err("malformed direct joint weights must panic");
        assert!(panic_message(panic).contains(expected));
    }

    scene.free(&gpu);
}

#[test]
#[should_panic(expected = "joint_weight_capacity capacity exceeded")]
fn joint_weight_capacity_is_independent_of_vertex_capacity() {
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
    let mut desc = scene_desc();
    desc.joint_weight_capacity = cube.positions.len() as u32 - 1;
    let mut scene = MeshScene::new(&gpu, &desc);
    let panic = catch_unwind(AssertUnwindSafe(|| {
        scene.add_mesh(
            &gpu,
            mesh::MeshDesc {
                joint_weights: Some(&joint_weights),
                ..cube.desc()
            },
        );
    }));
    scene.free(&gpu);
    resume_unwind(panic.expect_err("joint-weight overflow must panic"));
}

#[test]
fn instance_transform_path() {
    let _gpu_guard = gpu_test_lock();
    let gpu = Gpu::new(true).expect("vulkan init");
    let mut scene = MeshScene::new(&gpu, &scene_desc());
    let cube = cube(1.0);
    let mesh = scene.add_mesh(&gpu, cube.desc());
    let material = scene.add_material(&gpu, MaterialEntry::standard());
    let world = Mat4::from_scale_rotation_translation(
        Vec3::new(2.0, 3.0, 4.0),
        Quat::from_rotation_z(0.4),
        Vec3::new(5.0, 6.0, 7.0),
    );

    let instance = scene.add_instance(&gpu, mesh, world, material);
    let transform = scene.transforms_cpu()[instance.index() as usize];
    assert_mat4_close(transform.model_to_world, world);
    assert_mat4_close(transform.model_to_world_normal, world.inverse().transpose());

    let next_world = Mat4::from_scale_rotation_translation(
        Vec3::new(1.5, 2.5, 3.5),
        Quat::from_rotation_x(0.2),
        Vec3::new(1.0, 2.0, 3.0),
    );
    scene.set_world(&gpu, instance, next_world);
    let updated = scene.transforms_cpu()[instance.index() as usize];
    assert_mat4_close(updated.model_to_world, next_world);
    assert_mat4_close(
        updated.model_to_world_normal,
        next_world.inverse().transpose(),
    );

    scene.free(&gpu);
}

#[test]
#[should_panic(expected = "max_meshes capacity exceeded")]
fn capacity_overflow_panics() {
    let _gpu_guard = gpu_test_lock();
    let gpu = Gpu::new(true).expect("vulkan init");
    let desc = MeshSceneDesc {
        max_meshes: 0,
        max_instances: 0,
        max_materials: 0,
        vertex_capacity: 32,
        joint_weight_capacity: 0,
        index_capacity: 128,
        max_meshlets: 8,
    };
    let mut scene = MeshScene::new(&gpu, &desc);
    let cube = cube(1.0);
    let panic = catch_unwind(AssertUnwindSafe(|| {
        scene.add_mesh(&gpu, cube.desc());
    }));
    scene.free(&gpu);
    resume_unwind(panic.expect_err("add_mesh must panic"));
}

#[test]
#[should_panic(expected = "mesh indices[0] is out of range")]
fn out_of_range_index_panics() {
    let _gpu_guard = gpu_test_lock();
    let gpu = Gpu::new(true).expect("vulkan init");
    let mut scene = MeshScene::new(&gpu, &scene_desc());
    let mut cube = cube(1.0);
    cube.indices[0] = 99;
    let panic = catch_unwind(AssertUnwindSafe(|| {
        scene.add_mesh(&gpu, cube.desc());
    }));
    scene.free(&gpu);
    resume_unwind(panic.expect_err("add_mesh must panic"));
}

#[test]
#[should_panic(expected = "mirrored world transform flips triangle winding")]
fn mirrored_instance_world_panics() {
    let _gpu_guard = gpu_test_lock();
    let gpu = Gpu::new(true).expect("vulkan init");
    let mut scene = MeshScene::new(&gpu, &scene_desc());
    let cube = cube(1.0);
    let mesh = scene.add_mesh(&gpu, cube.desc());
    let material = scene.add_material(&gpu, MaterialEntry::standard());
    let panic = catch_unwind(AssertUnwindSafe(|| {
        scene.add_instance(
            &gpu,
            mesh,
            Mat4::from_scale(Vec3::new(-1.0, 1.0, 1.0)),
            material,
        );
    }));
    scene.free(&gpu);
    resume_unwind(panic.expect_err("add_instance must panic"));
}

fn assert_meshlet_invariants(scene: &MeshScene, mesh: mesh::MeshHandle, source_tri_count: usize) {
    let data = scene.mesh_data(mesh);
    let vertex_count = scene.mesh_vertex_count(mesh);
    let mut coverage = vec![0u8; source_tri_count];

    for meshlet_index in scene.meshlet_range(mesh) {
        let meshlet = scene.meshlets_cpu()[meshlet_index];
        assert!((1..=124).contains(&meshlet.tri_count));
        assert!(scene.meshlet_vertex_counts_cpu()[meshlet_index] <= 64);

        for &primitive_id in
            &scene.meshlet_primitive_ids_cpu()[scene.primitive_id_range(meshlet_index)]
        {
            assert!((primitive_id as usize) < source_tri_count);
            coverage[primitive_id as usize] += 1;
        }

        let first = meshlet.first_index as usize;
        let end = first + meshlet.tri_count as usize * 3;
        assert!(end <= scene.indices_cpu().len());
        for &index in &scene.indices_cpu()[first..end] {
            assert!(index < vertex_count);
        }
    }

    assert_eq!(
        scene.meshlet_range(mesh).len(),
        data.meshlet_count as usize,
        "meshlet range/count mismatch"
    );
    for (tri, count) in coverage.iter().enumerate() {
        assert_eq!(*count, 1, "source triangle {tri} coverage mismatch");
    }
}

fn read_mesh_table_entry(gpu: &Gpu, scene: &MeshScene) -> MeshTableEntry {
    read_mesh_table_entries::<1>(gpu, scene)[0]
}

fn read_mesh_table_entries<const N: usize>(gpu: &Gpu, scene: &MeshScene) -> [MeshTableEntry; N] {
    let readback = gpu.alloc_slice::<MeshTableEntry>(N as u64, Memory::Readback);
    let cb = gpu.commands_begin(Queue::Main);
    gpu.cmd_mem_copy_raw(
        cb,
        readback.cast(),
        scene.mesh_table_buffer().cast(),
        (N * size_of::<MeshTableEntry>()) as u64,
    );
    gpu.queue_submit(Queue::Main, &[cb]);
    gpu.queue_wait_idle(Queue::Main);
    // SAFETY: the transfer initialized all N readback entries before the wait returned.
    let entries = std::array::from_fn(|index| unsafe { *readback.cpu.add(index) });
    gpu.free(readback);
    entries
}

fn assert_global_index_readback(gpu: &Gpu, scene: &MeshScene) {
    let bytes = scene.indices_cpu().len() * size_of::<u32>();
    let readback = gpu.alloc_slice::<u32>(scene.indices_cpu().len() as u64, Memory::Readback);
    let cb = gpu.commands_begin(Queue::Main);
    gpu.cmd_mem_copy_raw(
        cb,
        readback.cast(),
        scene.global_index_buffer().cast(),
        bytes as u64,
    );
    gpu.queue_submit(Queue::Main, &[cb]);
    gpu.queue_wait_idle(Queue::Main);

    // SAFETY: both slices cover `bytes` initialized bytes for the exact same u32 count.
    let got = unsafe { std::slice::from_raw_parts(readback.cpu.cast::<u8>(), bytes) };
    let expected =
        unsafe { std::slice::from_raw_parts(scene.indices_cpu().as_ptr().cast::<u8>(), bytes) };
    assert_eq!(got, expected);
    gpu.free(readback);
}

fn panic_message(panic: Box<dyn std::any::Any + Send>) -> String {
    panic
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| {
            panic
                .downcast_ref::<&str>()
                .map(|message| (*message).to_owned())
        })
        .unwrap_or_default()
}

fn assert_mat4_close(got: Mat4, expected: Mat4) {
    for (i, (got, expected)) in got
        .to_cols_array()
        .into_iter()
        .zip(expected.to_cols_array())
        .enumerate()
    {
        assert!(
            (got - expected).abs() < 1.0e-5,
            "matrix element {i}: {got} != {expected}"
        );
    }
}
