//! Hardware test for temporal shadow reuse and same-frame correction.
//! History validation, refresh, invalidation, and light matching are exercised.

use std::sync::Mutex;

use abi_core::GpuPtr;
use abi_core::glam::{Mat4, UVec2, Vec2, Vec3};
use abi_core::oct_encode;
use abi_light::PointLight;
use abi_light::{
    LOCAL_SHADOW_SLOT_EMPTY, LOCAL_SHADOW_SLOTS, local_shadow_age, local_shadow_hit_q,
    local_shadow_state,
};
use abi_light::{
    SHADOW_STATE_INACTIVE, SHADOW_STATE_OCCLUDED, SHADOW_STATE_VISIBLE, ShadowSegment,
    shadow_segment_triangle_oracle,
};
use gpu::pass::{FrameAlloc, Pass};
use gpu::{
    Gpu, HazardFlags, Memory, Queue, Stage, TextureDesc, TextureFormat, TextureViewDesc, UsageFlags,
};
use mesh::{MaterialEntry, MeshRasterView, MeshScene, MeshSceneDesc, ShadowBlasDesc};
use render::{LocalShadowPass, LocalShadowTemporal, MeshSurfaceTargets};

static GPU_TEST_LOCK: Mutex<()> = Mutex::new(());
const W: u32 = 8;
const H: u32 = 8;
const HALF_W: u32 = 4;
const HALF_H: u32 = 4;
const TEXELS: usize = (HALF_W * HALF_H) as usize;
const ORIGIN_BIAS: f32 = 1.0e-3;
const DESTINATION_BIAS: f32 = 1.0e-3;

struct TestAlloc<'a> {
    gpu: &'a Gpu,
    live: Vec<gpu::Ptr<u8>>,
}

impl FrameAlloc for TestAlloc<'_> {
    fn frame_alloc<T: bytemuck::Pod>(&mut self, value: T) -> GpuPtr<T> {
        let p = self.gpu.alloc::<T>(Memory::Default);
        unsafe { *p.cpu = value };
        self.live.push(p.cast());
        p.gpu
    }

    fn frame_alloc_slice<T: bytemuck::Pod>(&mut self, values: &[T]) -> GpuPtr<T> {
        if values.is_empty() {
            return GpuPtr::null();
        }
        let p = self
            .gpu
            .alloc_slice::<T>(values.len() as u64, Memory::Default);
        unsafe { std::ptr::copy_nonoverlapping(values.as_ptr(), p.cpu, values.len()) };
        self.live.push(p.cast());
        p.gpu
    }
}

impl TestAlloc<'_> {
    fn free(self) {
        for p in self.live {
            self.gpu.free(p);
        }
    }
}

fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp32 = ((bits >> 23) & 0xff) as i32;
    let man = bits & 0x007f_ffff;
    let exp = exp32 - 127 + 15;
    if value == 0.0 {
        return sign;
    }
    assert!((1..0x1f).contains(&exp), "test constant outside f16 range");
    sign | ((exp as u32) << 10) as u16 | (man >> 13) as u16
}

fn receiver(x: u32, y: u32) -> Vec3 {
    let ndc = (Vec2::new(x as f32, y as f32) + Vec2::splat(0.5)) / Vec2::new(W as f32, H as f32)
        * 2.0
        - Vec2::ONE;
    Vec3::new(ndc.x, ndc.y, 0.8)
}

fn oracle_occluded(
    cube: &mesh::primitives::MeshBuffers,
    world_inverse: Mat4,
    from: Vec3,
    light: &PointLight,
) -> bool {
    let segment = ShadowSegment::between(
        from,
        Vec3::from_array(light.position),
        ORIGIN_BIAS,
        DESTINATION_BIAS,
    );
    let local = segment.transformed(world_inverse);
    cube.indices.chunks_exact(3).any(|triangle| {
        shadow_segment_triangle_oracle(
            &local,
            Vec3::from_array(cube.positions[triangle[0] as usize]),
            Vec3::from_array(cube.positions[triangle[1] as usize]),
            Vec3::from_array(cube.positions[triangle[2] as usize]),
        )
    })
}

/// Per-texel oracle at the representative receiver (top-left pixel).
fn oracle_texels(
    cube: &mesh::primitives::MeshBuffers,
    world: Mat4,
    light: &PointLight,
) -> Vec<bool> {
    let world_inverse = world.inverse();
    let mut occluded = Vec::with_capacity(TEXELS);
    for ty in 0..HALF_H {
        for tx in 0..HALF_W {
            occluded.push(oracle_occluded(
                cube,
                world_inverse,
                receiver(tx * 2, ty * 2),
                light,
            ));
        }
    }
    occluded
}

#[test]
fn v1_temporal_reuse_spends_rays_only_where_history_cannot_answer() {
    let _guard = GPU_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let gpu = Gpu::new(true).expect("vulkan init");
    let size = UVec2::new(W, H);
    let pixels = (W * H) as usize;
    let mut heap = gpu.heap_slots_create(8, 4, 2);

    let mut surfaces = None;
    assert!(MeshSurfaceTargets::ensure(
        &mut surfaces,
        &gpu,
        &mut heap,
        size
    ));
    let surfaces = surfaces.unwrap();
    let targets = surfaces.forward_targets();

    let depth = gpu.texture_alloc_and_create(
        TextureDesc {
            dimensions: [W, H, 1],
            format: TextureFormat::R32Float,
            usage: UsageFlags::SAMPLED | UsageFlags::TRANSFER_DST,
            ..Default::default()
        },
        Queue::Main,
        None,
    );
    let depth_slot = heap.add_sampled(
        &gpu,
        gpu.texture_view_descriptor(depth.texture, TextureViewDesc::default()),
    );

    let cube = mesh::primitives::cube(0.35);
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
    let home = Mat4::from_translation(Vec3::new(0.0, 0.0, 0.4));
    let cube_instance = scene.add_instance(&gpu, cube_mesh, home, material);

    let base_light = PointLight {
        position: [0.0, 0.0, -2.0],
        radius: 10.0,
        color: [1.0, 0.7, 0.4],
        intensity: 12.0,
    };

    let normal_oct = oct_encode(Vec3::NEG_Z);
    let normal_up = gpu.alloc_slice::<[u16; 2]>(pixels as u64, Memory::Default);
    let marker_up = gpu.alloc_slice::<f32>(pixels as u64, Memory::Default);
    let depth_up = gpu.alloc_slice::<f32>(pixels as u64, Memory::Default);
    unsafe {
        for i in 0..pixels {
            *normal_up.cpu.add(i) = [f32_to_f16(normal_oct.x), f32_to_f16(normal_oct.y)];
            *marker_up.cpu.add(i) = 1.0;
            *depth_up.cpu.add(i) = 0.8;
        }
    }

    let state_read = gpu.alloc_slice::<u32>(
        (TEXELS * LOCAL_SHADOW_SLOTS as usize) as u64,
        Memory::Readback,
    );
    let mut alloc = TestAlloc {
        gpu: &gpu,
        live: Vec::new(),
    };
    let mut shadows = LocalShadowPass::new(&gpu, size, 1, 1, 1);
    assert_eq!(shadows.half(), UVec2::new(HALF_W, HALF_H));
    let view = MeshRasterView {
        world_to_clip: Mat4::IDENTITY,
    };

    let run_frame = |shadows: &mut LocalShadowPass,
                     alloc: &mut TestAlloc,
                     scene: &MeshScene,
                     lights: &[PointLight],
                     temporal: LocalShadowTemporal| {
        let cb = gpu.commands_begin(Queue::Main);
        gpu.cmd_copy_to_texture(cb, targets.normal, normal_up.cast());
        gpu.cmd_copy_to_texture(cb, targets.material, marker_up.cast());
        gpu.cmd_copy_to_texture(cb, depth.texture, depth_up.cast());
        gpu.cmd_barrier(
            cb,
            Stage::Transfer,
            Stage::RasterColorOut,
            HazardFlags::empty(),
        );
        shadows.record(
            &gpu,
            cb,
            alloc,
            &heap,
            scene,
            scene.instances(),
            &surfaces,
            view,
            lights,
            ORIGIN_BIAS,
            DESTINATION_BIAS,
            0.0,
            temporal,
            depth_slot,
            0,
        );
        gpu.cmd_barrier(cb, Stage::Compute, Stage::Transfer, HazardFlags::empty());
        gpu.cmd_mem_copy_raw(
            cb,
            state_read.cast(),
            shadows.slot_state_buffer().cast(),
            (TEXELS * LOCAL_SHADOW_SLOTS as usize) as u64 * 4,
        );
        gpu.queue_submit(Queue::Main, &[cb]);
        gpu.queue_wait_idle(Queue::Main);
        shadows.take_counters(0)
    };

    let words =
        |i: usize| -> u32 { unsafe { *state_read.cpu.add(i * LOCAL_SHADOW_SLOTS as usize) } };
    let assert_states = |oracle: &[bool], phase: &str| {
        for (i, &occ) in oracle.iter().enumerate() {
            let state = local_shadow_state(words(i));
            let expected = if occ {
                SHADOW_STATE_OCCLUDED
            } else {
                SHADOW_STATE_VISIBLE
            };
            assert_eq!(state, expected, "{phase}: state mismatch at texel {i}");
            for upper in 1..LOCAL_SHADOW_SLOTS as usize {
                let empty = local_shadow_state(unsafe {
                    *state_read.cpu.add(i * LOCAL_SHADOW_SLOTS as usize + upper)
                });
                assert_eq!(
                    empty, SHADOW_STATE_INACTIVE,
                    "{phase}: slot {upper} not inactive"
                );
            }
        }
    };

    let reuse = |refresh: u32, thickness: f32| LocalShadowTemporal {
        refresh_interval: refresh,
        validate_thickness: thickness,
        near_plane: 1.0,
        light_epsilon: 0.25,
        contact: None,
        contact_distance: 0.0,
        ray_budget: 0,
        edge_promotion: false,
        occluded_refresh: 0,
        source_radius: 0.0,
    };
    let lights = [base_light];
    let oracle_home = oracle_texels(&cube, home, &base_light);
    let occ_home = oracle_home.iter().filter(|&&o| o).count() as u32;
    let vis_home = TEXELS as u32 - occ_home;
    assert!(
        occ_home >= 2 && vis_home >= 2,
        "degenerate scene: {occ_home} occluded / {vis_home} visible"
    );

    let c = run_frame(&mut shadows, &mut alloc, &scene, &lights, reuse(1, 1000.0));
    assert_eq!(
        c.requests_high, TEXELS as u32,
        "A: first frame must ray everything"
    );
    assert_eq!(c.requests_low, 0, "A: nothing has an estimate to refresh");
    assert_eq!((c.validated, c.reused), (0, 0), "A: no history yet");
    assert_eq!(c.overflow, 0);
    assert_states(&oracle_home, "A");
    let hit_q_a: Vec<u32> = (0..TEXELS).map(|i| local_shadow_hit_q(words(i))).collect();

    let c = run_frame(
        &mut shadows,
        &mut alloc,
        &scene,
        &lights,
        reuse(100, 1000.0),
    );
    assert_eq!(
        c.requests_high + c.requests_low,
        0,
        "B: static frame must spend zero rays"
    );
    assert_eq!(c.validated, occ_home, "B: every occluded slot validates");
    assert_eq!(c.reused, vis_home, "B: every visible slot reuses");
    assert_eq!(c.overflow, 0);
    assert_states(&oracle_home, "B");
    for (i, &occ) in oracle_home.iter().enumerate() {
        assert_eq!(
            local_shadow_age(words(i)),
            1,
            "B: age must tick at texel {i}"
        );
        if occ {
            assert_eq!(
                local_shadow_hit_q(words(i)),
                hit_q_a[i],
                "B: validation must preserve the stored hit parameter"
            );
        }
    }

    let c = run_frame(
        &mut shadows,
        &mut alloc,
        &scene,
        &lights,
        LocalShadowTemporal {
            occluded_refresh: 1,
            source_radius: 0.0,
            ..reuse(100, 1000.0)
        },
    );
    assert_eq!(c.requests_low, occ_home, "B2: old occlusion re-proves low");
    assert_eq!(c.requests_high, 0, "B2: nothing died");
    assert_states(&oracle_home, "B2");

    let away = Mat4::from_translation(Vec3::new(10.0, 0.0, 0.4));
    scene.set_world(&gpu, cube_instance, away);
    let oracle_away = oracle_texels(&cube, away, &base_light);
    assert!(
        oracle_away.iter().all(|&o| !o),
        "C: moved cube still occludes"
    );
    let c = run_frame(
        &mut shadows,
        &mut alloc,
        &scene,
        &lights,
        reuse(100, 1.0e-3),
    );
    assert_eq!(
        c.requests_high, occ_home,
        "C: refuted occlusion re-rays high"
    );
    assert_eq!(c.reused, vis_home, "C: visible reuse is untouched");
    assert_eq!(c.validated, 0, "C: nothing validates under a refuting tap");
    assert!(c.invalidated >= occ_home, "C: refutations must be counted");
    assert_states(&oracle_away, "C");

    scene.set_world(&gpu, cube_instance, home);
    let c = run_frame(
        &mut shadows,
        &mut alloc,
        &scene,
        &lights,
        reuse(100, 1000.0),
    );
    assert_eq!(
        c.requests_high + c.requests_low,
        0,
        "D: young visible trust spends no rays"
    );
    assert_eq!(c.reused, TEXELS as u32, "D: every slot reuses visible");
    let stale = oracle_home
        .iter()
        .enumerate()
        .filter(|&(i, &occ)| occ && local_shadow_state(words(i)) == SHADOW_STATE_VISIBLE)
        .count() as u32;
    assert_eq!(
        stale, occ_home,
        "D: the returned occluder must be invisible to stale trust (documented latency)"
    );

    let c = run_frame(&mut shadows, &mut alloc, &scene, &lights, reuse(1, 1000.0));
    assert_eq!(
        c.requests_low, TEXELS as u32,
        "E: expiry refreshes through the low queue"
    );
    assert_eq!(c.requests_high, 0, "E: expiry is not a history death");
    assert_eq!(c.reused, 0, "E: nothing may outlive the refresh interval");
    assert_states(&oracle_home, "E");

    let far_light = PointLight {
        position: [0.0, 0.0, -2.5],
        ..base_light
    };
    let lights_far = [far_light];
    let oracle_far = oracle_texels(&cube, home, &far_light);
    let c = run_frame(
        &mut shadows,
        &mut alloc,
        &scene,
        &lights_far,
        reuse(100, 1.0e-3),
    );
    assert_eq!(
        c.requests_high, TEXELS as u32,
        "F: teleport must re-ray everything high"
    );
    assert_eq!((c.validated, c.reused), (0, 0), "F: no history survives");
    assert_eq!(c.invalidated, TEXELS as u32, "F: every death is counted");
    assert_states(&oracle_far, "F");

    shadows.invalidate_history();
    let c = run_frame(
        &mut shadows,
        &mut alloc,
        &scene,
        &lights_far,
        reuse(100, 1000.0),
    );
    assert_eq!(
        c.requests_high, TEXELS as u32,
        "cut: dropped history re-rays high"
    );
    assert_states(&oracle_far, "cut");

    println!(
        "V1 temporal proof: texels={TEXELS} occluded={occ_home} visible={vis_home} bytes={}",
        shadows.allocated_bytes()
    );

    shadows.free(&gpu);
    alloc.free();
    gpu.free(state_read);
    gpu.free(normal_up);
    gpu.free(marker_up);
    gpu.free(depth_up);
    gpu.texture_free_and_destroy(depth);
    surfaces.free(&gpu);
    scene.free(&gpu);
    heap.free(&gpu);
}

const _: () = assert!(LOCAL_SHADOW_SLOT_EMPTY == 0xFFFF);
