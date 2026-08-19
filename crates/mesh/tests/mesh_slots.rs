//! Hardware tests for mutable mesh slots and reservation reuse.

mod common;

use std::panic::{AssertUnwindSafe, catch_unwind};

use abi_core::glam::{Mat4, UVec2, Vec3};
use abi_light::{
    SHADOW_QUERY_FAILED, SHADOW_QUERY_OCCLUDED, ShadowQueryResult, ShadowSegment,
    shadow_segment_triangle_oracle,
};
use abi_mesh::mesh_world_to_clip;
use common::{TestFrameAlloc, gpu_test_lock, mesh_heap, view};
use gpu::pass::Pass;
use gpu::{
    Gpu, HazardFlags, LoadOp, Memory, Queue, RenderAttachment, RenderPassDesc, Stage, StoreOp,
    TextureDesc, TextureFormat, UsageFlags,
};
use mesh::cull::ClusterCullPass;
use mesh::primitives::{MeshBuffers, cube, icosphere};
use mesh::{
    MaterialEntry, MeshDepthPrepass, MeshForwardPass, MeshForwardTargets, MeshRasterView,
    MeshScene, MeshSceneDesc, MeshShadeLighting, ShadowBlasDesc, ShadowBlasQueryPass,
};

const W: u32 = 65;
const H: u32 = 65;
const CLEAR: [f32; 4] = [0.02, 0.03, 0.04, 1.0];

/// Camera-facing plane with independently varying geometry counts.
fn plane(quads: u32, half: f32) -> MeshBuffers {
    let mut mesh = MeshBuffers::default();
    for y in 0..=quads {
        for x in 0..=quads {
            let fx = x as f32 / quads as f32 * 2.0 - 1.0;
            let fy = y as f32 / quads as f32 * 2.0 - 1.0;
            mesh.positions.push([fx * half, fy * half, 0.0]);
            mesh.normals.push([0.0, 0.0, -1.0]);
            mesh.uvs
                .push([x as f32 / quads as f32, y as f32 / quads as f32]);
        }
    }
    for y in 0..quads {
        for x in 0..quads {
            let base = y * (quads + 1) + x;
            mesh.indices.extend_from_slice(&[
                base,
                base + quads + 1,
                base + 1,
                base + 1,
                base + quads + 1,
                base + quads + 2,
            ]);
        }
    }
    mesh
}

fn scene_desc() -> MeshSceneDesc {
    MeshSceneDesc {
        max_meshes: 8,
        max_instances: 4,
        max_materials: 4,
        vertex_capacity: 4096,
        joint_weight_capacity: 0,
        index_capacity: 32_768,
        max_meshlets: 256,
    }
}

fn lighting() -> MeshShadeLighting {
    MeshShadeLighting {
        sun_direction: Vec3::NEG_Z.to_array(),
        sun_tint: [1.0, 0.75, 0.5],
        sky_ambient: [0.2, 0.25, 0.3],
        ground_ambient: [0.05, 0.04, 0.03],
        ..MeshShadeLighting::zeroed()
    }
}

/// Deterministic stream of 1,000 geometries with bounded size variation.
fn edit(i: u32) -> MeshBuffers {
    let quads = 3 + (i * 7 + i / 3) % 6;
    let half = 0.6 + ((i % 11) as f32) * 0.05;
    plane(quads, half)
}

/// Rewrites must stabilize arena high-water marks after warm-up.
#[test]
fn update_mesh_a_thousand_times_leaves_the_arenas_flat() {
    let _gpu_guard = gpu_test_lock();
    let gpu = Gpu::new(true).expect("vulkan init");
    let mut scene = MeshScene::new_with_shadows(
        &gpu,
        &scene_desc(),
        ShadowBlasDesc {
            node_capacity: 4096,
            primitive_capacity: 8192,
        },
    );

    let handle = scene.add_mesh(&gpu, edit(0).desc());
    let first = scene.arena_usage();

    // Warm-up reaches the reservation high-water marks.
    const WARMUP: u32 = 64;
    for i in 1..WARMUP {
        scene.update_mesh(&gpu, handle, edit(i).desc());
    }
    let warm = scene.arena_usage();

    for i in WARMUP..1000 {
        scene.update_mesh(&gpu, handle, edit(i).desc());
        assert_eq!(
            scene.arena_usage(),
            warm,
            "edit {i} grew an arena after the reservation converged"
        );
    }

    assert_eq!(scene.mesh_count(), 1, "one mesh, a thousand edits");
    assert_eq!(scene.mesh_slot_bound(), 1, "one slot, a thousand edits");
    // The handle is still the holder's: `update_mesh` does not bump.
    assert_eq!(
        scene.mesh_data(handle).idx_count,
        edit(999).indices.len() as u32
    );

    println!(
        "drain (update_mesh × 1000): first={first:?} converged={warm:?} \
         append-only would have consumed {}",
        append_only_cost(1000)
    );
    scene.free(&gpu);
}

/// Computes append-only arena demand for comparison.
fn append_only_cost(edits: u32) -> String {
    let mut vertices = 0u64;
    let mut indices = 0u64;
    for i in 0..edits {
        let mesh = edit(i);
        vertices += mesh.positions.len() as u64;
        // Each triangle contributes source and meshlet index storage.
        indices += mesh.indices.len() as u64 * 2;
    }
    format!("{edits} slots / {vertices} verts / {indices} index words")
}

/// Repeated add/remove must reuse slot reservations.
#[test]
fn add_remove_a_thousand_times_leaves_the_arenas_flat() {
    let _gpu_guard = gpu_test_lock();
    let gpu = Gpu::new(true).expect("vulkan init");
    let mut scene = MeshScene::new_with_shadows(
        &gpu,
        &scene_desc(),
        ShadowBlasDesc {
            node_capacity: 4096,
            primitive_capacity: 8192,
        },
    );

    let mut live = scene.add_mesh(&gpu, edit(0).desc());
    const WARMUP: u32 = 64;
    for i in 1..WARMUP {
        // Register before removing, matching the consumer order.
        let next = scene.add_mesh(&gpu, edit(i).desc());
        scene.remove_mesh(live);
        live = next;
    }
    let warm = scene.arena_usage();
    let warm_bound = scene.mesh_slot_bound();

    for i in WARMUP..1000 {
        let next = scene.add_mesh(&gpu, edit(i).desc());
        scene.remove_mesh(live);
        live = next;
        assert_eq!(
            scene.arena_usage(),
            warm,
            "rebuild {i} grew an arena after the reservations converged"
        );
        assert_eq!(scene.mesh_count(), 1, "rebuild {i} leaked a live mesh");
        assert_eq!(
            scene.mesh_slot_bound(),
            warm_bound,
            "rebuild {i} consumed a fresh slot index"
        );
    }

    assert!(
        warm_bound <= 2,
        "add-then-remove needs two slots at most, got {warm_bound}"
    );
    println!(
        "drain (add/remove × 1000): converged={warm:?} slots={warm_bound}; \
         append-only would have consumed {}",
        append_only_cost(1000)
    );
    scene.free(&gpu);
}

/// Renders one cull, prepass, and forward frame to CPU pixels.
fn render(build: impl FnOnce(&Gpu, &mut MeshScene)) -> Vec<[f32; 4]> {
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
    build(&gpu, &mut scene);

    let v = view(size);
    let raster_view = MeshRasterView {
        world_to_clip: mesh_world_to_clip(&v),
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

    let cb = gpu.commands_begin(Queue::Main);
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
        &mut frame_alloc,
        &scene,
        scene.instances(),
        raster_view,
        Vec3::from_array(v.camera_position),
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
        Vec3::from_array(v.camera_position),
        lighting(),
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

    // SAFETY: the readback buffer covers W*H pixels and the queue is idle.
    let pixels = unsafe { std::slice::from_raw_parts(color_rb.cpu, (W * H) as usize) }.to_vec();

    frame_alloc.free();
    prepass.free(&gpu);
    pass.free(&gpu);
    cull.free(&gpu);
    scene.free(&gpu);
    gpu.texture_free_and_destroy(color);
    gpu.texture_free_and_destroy(depth);
    gpu.free(color_rb);
    heap.free(&gpu);
    pixels
}

fn lit_pixels(pixels: &[[f32; 4]]) -> usize {
    pixels.iter().filter(|p| **p != CLEAR).count()
}

/// Updated geometry must match fresh registration for shrink and grow paths.
#[test]
fn updated_geometry_renders_exactly_like_a_fresh_registration() {
    let _gpu_guard = gpu_test_lock();

    for (from, to, label) in [
        (plane(6, 1.0), plane(3, 0.55), "shrink (fits reservation)"),
        (plane(3, 0.55), plane(6, 1.0), "grow (bumps the arena)"),
        (plane(4, 0.8), icosphere(0.9, 2), "different primitive"),
    ] {
        let updated = render(|gpu, scene| {
            let mesh = scene.add_mesh(gpu, from.desc());
            let material = scene.add_material(gpu, MaterialEntry::standard());
            scene.add_instance(gpu, mesh, Mat4::IDENTITY, material);
            scene.update_mesh(gpu, mesh, to.desc());
        });
        let fresh = render(|gpu, scene| {
            let mesh = scene.add_mesh(gpu, to.desc());
            let material = scene.add_material(gpu, MaterialEntry::standard());
            scene.add_instance(gpu, mesh, Mat4::IDENTITY, material);
        });
        assert!(
            lit_pixels(&fresh) > 64,
            "{label}: the fixture must actually draw something"
        );
        assert_eq!(
            updated, fresh,
            "{label}: an updated slot must render its NEW geometry, byte for byte"
        );
    }
}

/// Reclaimed slots must render new geometry without stale data.
#[test]
fn a_reclaimed_slot_renders_exactly_like_a_fresh_registration() {
    let _gpu_guard = gpu_test_lock();
    let first = plane(6, 1.0);
    let second = icosphere(0.9, 2);

    let reclaimed = render(|gpu, scene| {
        let old = scene.add_mesh(gpu, first.desc());
        scene.remove_mesh(old);
        let mesh = scene.add_mesh(gpu, second.desc());
        assert_eq!(mesh.index(), old.index(), "the slot must be reclaimed");
        assert_ne!(
            mesh.generation(),
            old.generation(),
            "reclaiming a slot must bump its generation"
        );
        let material = scene.add_material(gpu, MaterialEntry::standard());
        scene.add_instance(gpu, mesh, Mat4::IDENTITY, material);
    });
    let fresh = render(|gpu, scene| {
        let mesh = scene.add_mesh(gpu, second.desc());
        let material = scene.add_material(gpu, MaterialEntry::standard());
        scene.add_instance(gpu, mesh, Mat4::IDENTITY, material);
    });
    assert!(lit_pixels(&fresh) > 64);
    assert_eq!(reclaimed, fresh);
}

/// **Generation policy, as law** (`update_mesh` doc comment): a handle held
/// across `remove_mesh` dies at its next use; one held across `update_mesh`
/// does not.
#[test]
fn stale_handles_die_and_updated_handles_live() {
    let _gpu_guard = gpu_test_lock();
    let gpu = Gpu::new(true).expect("vulkan init");
    let mut scene = MeshScene::new(&gpu, &scene_desc());
    let cube = cube(1.0);

    let handle = scene.add_mesh(&gpu, cube.desc());
    scene.update_mesh(&gpu, handle, icosphere(1.0, 1).desc());
    // Survives its own contents changing — that is the whole point of a
    // handle to a mesh whose geometry the holder keeps editing.
    let data = scene.mesh_data(handle);
    assert_eq!(data.idx_count, icosphere(1.0, 1).indices.len() as u32);

    scene.remove_mesh(handle);
    let panic = catch_unwind(AssertUnwindSafe(|| scene.mesh_data(handle)))
        .expect_err("a handle held across remove_mesh must panic on use");
    let message = panic
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| panic.downcast_ref::<&str>().map(|s| (*s).to_string()))
        .unwrap_or_default();
    assert!(
        message.contains("invalid mesh handle"),
        "stale handle must name itself, got {message:?}"
    );

    // The reclaimed slot is a different mesh, and the old handle stays dead.
    let reborn = scene.add_mesh(&gpu, cube.desc());
    assert_eq!(reborn.index(), handle.index());
    assert!(
        catch_unwind(AssertUnwindSafe(|| scene.mesh_data(handle))).is_err(),
        "reclaiming a slot must not resurrect the old handle"
    );
    assert_eq!(scene.mesh_data(reborn).idx_count, cube.indices.len() as u32);

    scene.free(&gpu);
}

fn brute_force(mesh: &MeshBuffers, segment: &ShadowSegment) -> bool {
    mesh.indices.chunks_exact(3).any(|triangle| {
        shadow_segment_triangle_oracle(
            segment,
            Vec3::from_array(mesh.positions[triangle[0] as usize]),
            Vec3::from_array(mesh.positions[triangle[1] as usize]),
            Vec3::from_array(mesh.positions[triangle[2] as usize]),
        )
    })
}

fn segments() -> Vec<ShadowSegment> {
    let mut queries = Vec::new();
    let mut seed = 0x1312_A5A5u32;
    let component = |state: &mut u32| {
        *state ^= *state << 13;
        *state ^= *state >> 17;
        *state ^= *state << 5;
        (*state >> 8) as f32 * (1.0 / 16_777_215.0) * 6.0 - 3.0
    };
    while queries.len() < 193 {
        let start = Vec3::new(
            component(&mut seed),
            component(&mut seed),
            component(&mut seed),
        );
        let end = Vec3::new(
            component(&mut seed),
            component(&mut seed),
            component(&mut seed),
        );
        if start.distance_squared(end) < 1.0e-4 {
            continue;
        }
        queries.push(ShadowSegment::between(start, end, 1.0e-4, 1.0e-4));
    }
    queries
}

/// Shadow BLAS pointers and traversal must follow mesh updates.
#[test]
fn shadow_blas_follows_an_update_in_lockstep() {
    let _gpu_guard = gpu_test_lock();
    let gpu = Gpu::new(true).expect("vulkan init");
    let mut scene = MeshScene::new_with_shadows(
        &gpu,
        &scene_desc(),
        ShadowBlasDesc {
            node_capacity: 4096,
            primitive_capacity: 8192,
        },
    );

    // Register a small sphere, then rewrite the slot as a bigger one: the
    // BLAS grows, so this exercises the reservation bump too.
    let small = icosphere(0.4, 1);
    let big = icosphere(1.3, 2);
    let handle = scene.add_mesh(&gpu, small.desc());
    scene.update_mesh(&gpu, handle, big.desc());

    let blas = scene.shadow_blas(handle);
    let stats = scene.shadow_blas_stats(handle);
    assert_eq!(
        stats.primitive_count,
        (big.indices.len() / 3) as u32,
        "the staged BLAS must cover the UPDATED triangle set"
    );

    let queries = segments();
    let expected = queries
        .iter()
        .map(|segment| brute_force(&big, segment))
        .collect::<Vec<_>>();
    assert!(
        expected.iter().any(|hit| *hit) && expected.iter().any(|hit| !*hit),
        "the query set must contain both hits and misses to prove anything"
    );

    let query_buffer = gpu.alloc_slice::<ShadowSegment>(queries.len() as u64, Memory::Default);
    let result_buffer = gpu.alloc_slice::<ShadowQueryResult>(queries.len() as u64, Memory::Default);
    // SAFETY: both are fresh host-visible allocations sized for the queries.
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
    let mut frame = TestFrameAlloc {
        gpu: &gpu,
        ptrs: Vec::new(),
    };
    let cb = gpu.commands_begin(Queue::Main);
    pass.record(
        &gpu,
        cb,
        &mut frame,
        blas,
        query_buffer.gpu,
        result_buffer.gpu,
        queries.len() as u32,
    );
    gpu.queue_submit(Queue::Main, &[cb]);
    gpu.queue_wait_idle(Queue::Main);

    // SAFETY: the dispatch wrote every result before the wait returned.
    let got = unsafe { std::slice::from_raw_parts(result_buffer.cpu, queries.len()) };
    for (i, (result, expected)) in got.iter().zip(&expected).enumerate() {
        assert_ne!(result.status, SHADOW_QUERY_FAILED, "query {i} failed");
        assert_eq!(
            result.status == SHADOW_QUERY_OCCLUDED,
            *expected,
            "query {i} disagrees with the oracle after update_mesh"
        );
    }

    pass.free(&gpu);
    frame.free();
    gpu.free(query_buffer);
    gpu.free(result_buffer);
    scene.free(&gpu);
}

/// Measures update registration's queue-drain cost.
#[test]
fn report_registration_stall() {
    let _gpu_guard = gpu_test_lock();
    let gpu = Gpu::new(true).expect("vulkan init");
    // Use representative chunk-mesh capacity with headroom.
    let mut scene = MeshScene::new(
        &gpu,
        &MeshSceneDesc {
            max_meshes: 8,
            max_instances: 4,
            max_materials: 4,
            vertex_capacity: 1 << 16,
            joint_weight_capacity: 0,
            index_capacity: 1 << 18,
            max_meshlets: 1 << 12,
        },
    );

    let geometry = (0..8).map(|i| plane(48 + i % 3, 1.0)).collect::<Vec<_>>();
    let handle = scene.add_mesh(&gpu, geometry[0].desc());
    for i in 0..16 {
        scene.update_mesh(&gpu, handle, geometry[i % geometry.len()].desc());
    }

    const N: usize = 64;
    let started = std::time::Instant::now();
    for i in 0..N {
        scene.update_mesh(&gpu, handle, geometry[i % geometry.len()].desc());
    }
    let per_update = started.elapsed().as_secs_f64() * 1.0e3 / N as f64;

    let add_started = std::time::Instant::now();
    let mut live = scene.add_mesh(&gpu, geometry[0].desc());
    for i in 1..N {
        let next = scene.add_mesh(&gpu, geometry[i % geometry.len()].desc());
        scene.remove_mesh(live);
        live = next;
    }
    let per_add = add_started.elapsed().as_secs_f64() * 1.0e3 / N as f64;

    // Measure queue drain without staging or CPU preparation.
    let src = gpu.alloc_slice::<u32>(4, Memory::Default);
    let dst = gpu.alloc_slice::<u32>(4, Memory::Default);
    for _ in 0..8 {
        let cb = gpu.commands_begin(Queue::Main);
        gpu.cmd_mem_copy_raw(cb, dst.cast(), src.cast(), 16);
        gpu.queue_submit(Queue::Main, &[cb]);
        gpu.queue_wait_idle(Queue::Main);
    }
    let drain_started = std::time::Instant::now();
    for _ in 0..N {
        let cb = gpu.commands_begin(Queue::Main);
        gpu.cmd_mem_copy_raw(cb, dst.cast(), src.cast(), 16);
        gpu.queue_submit(Queue::Main, &[cb]);
        gpu.queue_wait_idle(Queue::Main);
    }
    let per_drain = drain_started.elapsed().as_secs_f64() * 1.0e3 / N as f64;
    gpu.free(src);
    gpu.free(dst);

    println!(
        "registration stall ({} verts / {} indices, {} build): \
         update_mesh {per_update:.3} ms, add_mesh+remove_mesh {per_add:.3} ms, \
         bare submit+queue_wait_idle {per_drain:.3} ms \
         ({:.1}% of a registration); 5 rebuilds/frame = {:.2} ms",
        geometry[0].positions.len(),
        geometry[0].indices.len(),
        if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        },
        per_drain / per_update * 100.0,
        per_update * 5.0,
    );

    scene.free(&gpu);
}
