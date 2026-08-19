//! Hardware verification of CWBVH upload and bounded any-hit traversal.
//! Results are compared with an independent f64 triangle oracle.

mod common;

use abi_core::glam::Vec3;
use abi_light::{
    SHADOW_QUERY_FAILED, SHADOW_QUERY_OCCLUDED, ShadowQueryResult, ShadowSegment,
    shadow_segment_triangle_oracle,
};
use common::{TestFrameAlloc, gpu_test_lock};
use gpu::pass::Pass;
use gpu::{Gpu, Memory, Queue};
use mesh::primitives::icosphere;
use mesh::{MeshScene, MeshSceneDesc, ShadowBlasDesc, ShadowBlasQueryPass};

fn brute_force(mesh: &mesh::MeshBuffers, segment: &ShadowSegment) -> bool {
    mesh.indices.chunks_exact(3).any(|triangle| {
        shadow_segment_triangle_oracle(
            segment,
            Vec3::from_array(mesh.positions[triangle[0] as usize]),
            Vec3::from_array(mesh.positions[triangle[1] as usize]),
            Vec3::from_array(mesh.positions[triangle[2] as usize]),
        )
    })
}

fn random_component(state: &mut u32) -> f32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    let unit = (*state >> 8) as f32 * (1.0 / 16_777_215.0);
    unit * 6.0 - 3.0
}

#[test]
fn gpu_cwbvh_any_hit_matches_brute_force() {
    let _guard = gpu_test_lock();
    let gpu = Gpu::new(true).expect("vulkan init");
    let geometry = icosphere(1.0, 2);
    let primitive_count = (geometry.indices.len() / 3) as u32;
    let mut scene = MeshScene::new_with_shadows(
        &gpu,
        &MeshSceneDesc {
            max_meshes: 1,
            max_instances: 1,
            max_materials: 1,
            vertex_capacity: 512,
            joint_weight_capacity: 0,
            index_capacity: 4_096,
            max_meshlets: 8,
        },
        ShadowBlasDesc {
            node_capacity: 512,
            primitive_capacity: 512,
        },
    );
    let mesh = scene.add_mesh(&gpu, geometry.desc());
    let blas = scene.shadow_blas(mesh);
    let stats = scene.shadow_blas_stats(mesh);
    assert_eq!(stats.primitive_count, primitive_count);
    assert!(stats.node_count > 0);
    assert!(stats.max_depth < 32);
    assert_eq!(scene.shadow_allocated_bytes(), 512 * 80 + 512 * 4 + 48);
    assert_eq!(
        scene.shadow_payload_bytes(),
        u64::from(stats.node_count) * 80 + u64::from(primitive_count) * 4 + 48
    );

    // Controlled cases cover endpoint, zero-direction, inside, and miss paths.
    let mut queries = vec![
        ShadowSegment::between(
            Vec3::new(0.0, 0.0, 3.0),
            Vec3::new(0.0, 0.0, -3.0),
            0.0,
            0.0,
        ),
        ShadowSegment::between(
            Vec3::new(3.0, 0.0, 3.0),
            Vec3::new(3.0, 0.0, -3.0),
            0.0,
            0.0,
        ),
        ShadowSegment::between(Vec3::new(0.0, 0.0, 1.0), Vec3::new(0.0, 0.0, 3.0), 0.0, 0.0),
        ShadowSegment::between(Vec3::new(0.0, 0.0, 3.0), Vec3::new(0.0, 0.0, 1.0), 0.0, 0.0),
        ShadowSegment::between(
            Vec3::new(-3.0, 0.2, 0.3),
            Vec3::new(3.0, 0.2, 0.3),
            0.0,
            0.0,
        ),
        ShadowSegment::between(Vec3::ZERO, Vec3::new(0.0, 0.0, 3.0), 1.0e-4, 1.0e-4),
        ShadowSegment::between(
            Vec3::new(-3.0, 0.9, 0.9),
            Vec3::new(3.0, 0.9, 0.9),
            1.0e-4,
            1.0e-4,
        ),
    ];
    let mut seed = 0xA5A5_1312u32;
    while queries.len() < 257 {
        let start = Vec3::new(
            random_component(&mut seed),
            random_component(&mut seed),
            random_component(&mut seed),
        );
        let mut end = Vec3::new(
            random_component(&mut seed),
            random_component(&mut seed),
            random_component(&mut seed),
        );
        if start.distance_squared(end) < 1.0e-4 {
            end.x += 0.25;
        }
        queries.push(ShadowSegment::between(start, end, 1.0e-4, 1.0e-4));
    }
    assert_ne!(queries.len() as u32 % ShadowBlasQueryPass::GROUP_SIZE, 0);
    let expected = queries
        .iter()
        .map(|segment| brute_force(&geometry, segment))
        .collect::<Vec<_>>();

    let query_buffer = gpu.alloc_slice::<ShadowSegment>(queries.len() as u64, Memory::Default);
    let result_buffer = gpu.alloc_slice::<ShadowQueryResult>(queries.len() as u64, Memory::Default);
    unsafe {
        std::ptr::copy_nonoverlapping(queries.as_ptr(), query_buffer.cpu, queries.len());
        for i in 0..queries.len() {
            result_buffer.cpu.add(i).write(ShadowQueryResult {
                status: 0xDEAD_BEEF,
                primitive_id: 0xDEAD_BEEF,
                hit_t: -1.0,
                node_tests: 0xDEAD_BEEF,
                triangle_tests: 0xDEAD_BEEF,
                max_stack_depth: 0xDEAD_BEEF,
            });
        }
    }

    let pass = ShadowBlasQueryPass::new(&gpu);
    let timestamps = gpu.timestamp_pool_create(2);
    let mut frame = TestFrameAlloc {
        gpu: &gpu,
        ptrs: Vec::new(),
    };
    let cb = gpu.commands_begin(Queue::Main);
    gpu.cmd_timestamp_reset(cb, &timestamps);
    gpu.cmd_timestamp(cb, &timestamps, 0);
    pass.record(
        &gpu,
        cb,
        &mut frame,
        blas,
        query_buffer.gpu,
        result_buffer.gpu,
        queries.len() as u32,
    );
    gpu.cmd_timestamp(cb, &timestamps, 1);
    gpu.queue_submit(Queue::Main, &[cb]);
    gpu.queue_wait_idle(Queue::Main);
    let stamps = gpu.timestamp_pool_read(&timestamps);
    let gpu_us = stamps[1].unwrap().saturating_sub(stamps[0].unwrap()) as f64
        * gpu.device_limits().timestamp_period as f64
        / 1.0e3;

    let got = unsafe { std::slice::from_raw_parts(result_buffer.cpu, queries.len()) };
    let mut node_tests = 0u64;
    let mut triangle_tests = 0u64;
    let mut max_stack = 0u32;
    let mut hits = 0u32;
    for (i, ((result, expected), segment)) in got.iter().zip(&expected).zip(&queries).enumerate() {
        assert_ne!(
            result.status, SHADOW_QUERY_FAILED,
            "query {i} failed: {result:?}"
        );
        assert_eq!(
            result.status == SHADOW_QUERY_OCCLUDED,
            *expected,
            "query {i}: {segment:?}, result={result:?}"
        );
        assert!(result.node_tests > 0, "active query {i} tested no nodes");
        assert!(result.max_stack_depth < 32, "query {i} overflowed");
        if *expected {
            hits += 1;
            assert!(
                result.primitive_id < primitive_count,
                "query {i} primitive ID"
            );
            assert!(
                result.hit_t > segment.t_min && result.hit_t < segment.t_max,
                "query {i} hit outside its open interval"
            );
        } else {
            assert_eq!(result.primitive_id, u32::MAX, "miss query {i} primitive");
            assert!(result.hit_t.is_infinite(), "miss query {i} t");
        }
        node_tests += u64::from(result.node_tests);
        triangle_tests += u64::from(result.triangle_tests);
        max_stack = max_stack.max(result.max_stack_depth);
    }
    println!(
        "BLAS queries: count={} hits={hits} nodes={} primitives={} max_depth={} max_stack={max_stack} avg_nodes={:.2} avg_tris={:.2} build_us={} gpu_us={gpu_us:.2}",
        queries.len(),
        stats.node_count,
        stats.primitive_count,
        stats.max_depth,
        node_tests as f64 / queries.len() as f64,
        triangle_tests as f64 / queries.len() as f64,
        stats.build_time.as_micros(),
    );

    gpu.timestamp_pool_destroy(timestamps);
    pass.free(&gpu);
    frame.free();
    gpu.free(query_buffer);
    gpu.free(result_buffer);
    scene.free(&gpu);
}
