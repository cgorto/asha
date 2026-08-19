//! Hardware test for bounded screen-blind occlusion reproof.
//! Blind history follows its leash instead of repeatedly invalidating.

use std::sync::Mutex;

use abi_core::GpuPtr;
use abi_core::glam::{Mat4, UVec2, Vec2, Vec3};
use abi_core::oct_encode;
use abi_light::PointLight;
use abi_light::{
    LOCAL_SHADOW_SLOTS, local_shadow_age, local_shadow_blind, local_shadow_fresh_age,
    local_shadow_state,
};
use abi_light::{
    SHADOW_STATE_OCCLUDED, SHADOW_STATE_VISIBLE, ShadowSegment, shadow_segment_triangle_oracle,
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
/// Short leash so the cadence proves within a handful of frames.
const LEASH: u32 = 4;
/// Refresh horizon kept beyond the test run.
const REFRESH: u32 = 1000;
const FRAMES_AFTER_BIRTH: u32 = 9;

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

fn oracle_texels(
    cube: &mesh::primitives::MeshBuffers,
    world: Mat4,
    light: &PointLight,
) -> Vec<bool> {
    let world_inverse = world.inverse();
    let mut occluded = Vec::with_capacity(TEXELS);
    for ty in 0..HALF_H {
        for tx in 0..HALF_W {
            let from = receiver(tx * 2, ty * 2);
            let segment = ShadowSegment::between(
                from,
                Vec3::from_array(light.position),
                ORIGIN_BIAS,
                DESTINATION_BIAS,
            );
            let local = segment.transformed(world_inverse);
            occluded.push(cube.indices.chunks_exact(3).any(|triangle| {
                shadow_segment_triangle_oracle(
                    &local,
                    Vec3::from_array(cube.positions[triangle[0] as usize]),
                    Vec3::from_array(cube.positions[triangle[1] as usize]),
                    Vec3::from_array(cube.positions[triangle[2] as usize]),
                )
            }));
        }
    }
    occluded
}

#[test]
fn v5_blind_occlusion_rides_the_leash_instead_of_churning() {
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

    let light = PointLight {
        position: [0.0, 0.0, -2.0],
        radius: 10.0,
        color: [1.0, 0.7, 0.4],
        intensity: 12.0,
    };
    let lights = [light];

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
    let view = MeshRasterView {
        world_to_clip: Mat4::IDENTITY,
    };
    let temporal = LocalShadowTemporal {
        refresh_interval: REFRESH,
        validate_thickness: 0.05,
        near_plane: 1.0,
        light_epsilon: 0.25,
        contact: None,
        contact_distance: 0.0,
        ray_budget: 0,
        edge_promotion: false,
        occluded_refresh: LEASH,
        source_radius: 0.0,
    };

    let run_frame = |shadows: &mut LocalShadowPass, alloc: &mut TestAlloc, scene: &MeshScene| {
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
            &lights,
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

    let oracle_home = oracle_texels(&cube, home, &light);
    let occluded: Vec<usize> = (0..TEXELS).filter(|&i| oracle_home[i]).collect();
    let visible_count = (TEXELS - occluded.len()) as u32;
    assert!(
        occluded.len() >= 2 && visible_count >= 2,
        "degenerate scene: {} occluded / {visible_count} visible",
        occluded.len()
    );
    for i in 0..TEXELS {
        let phase =
            local_shadow_fresh_age((i * LOCAL_SHADOW_SLOTS as usize) as u32, 0x0A53, REFRESH);
        assert!(
            phase + FRAMES_AFTER_BIRTH + 1 < REFRESH,
            "visible phase at texel {i} too close to the refresh horizon"
        );
    }

    let c = run_frame(&mut shadows, &mut alloc, &scene);
    assert_eq!(
        c.requests_high, TEXELS as u32,
        "A: first frame rays everything"
    );
    assert_eq!(c.validated, 0, "A: no tap can run on the first frame");
    let mut ages: Vec<u32> = Vec::with_capacity(occluded.len());
    for &i in &occluded {
        let word = words(i);
        assert_eq!(
            local_shadow_state(word),
            SHADOW_STATE_OCCLUDED,
            "A: occluded texel {i} must trace occluded"
        );
        assert!(
            local_shadow_blind(word),
            "A: unvalidatable hit at texel {i} must classify BLIND"
        );
        let expected =
            local_shadow_fresh_age((i * LOCAL_SHADOW_SLOTS as usize) as u32, 0x0C91, LEASH);
        assert_eq!(
            local_shadow_age(word),
            expected,
            "A: birth phase mismatch at texel {i}"
        );
        ages.push(expected);
    }

    for frame in 0..FRAMES_AFTER_BIRTH {
        let mut expect_low = 0u32;
        let mut expect_carried = 0u32;
        for age in ages.iter_mut() {
            *age += 1;
            if *age >= LEASH {
                expect_low += 1;
                *age = 0;
            } else {
                expect_carried += 1;
            }
        }
        let c = run_frame(&mut shadows, &mut alloc, &scene);
        assert_eq!(
            c.requests_low, expect_low,
            "B{frame}: leash re-proofs must go low"
        );
        assert_eq!(c.requests_high, 0, "B{frame}: blind carry is not a death");
        assert_eq!(c.blind, expect_carried, "B{frame}: carried blind count");
        assert_eq!(c.validated, 0, "B{frame}: the tap must never run here");
        assert_eq!(c.invalidated, 0, "B{frame}: nothing dies on the leash");
        assert_eq!(c.overflow, 0);
        for (k, &i) in occluded.iter().enumerate() {
            let word = words(i);
            assert_eq!(
                local_shadow_state(word),
                SHADOW_STATE_OCCLUDED,
                "B{frame}: texel {i} must stay occluded"
            );
            assert_eq!(
                local_shadow_age(word),
                ages[k],
                "B{frame}: replay age mismatch at texel {i}"
            );
        }
    }

    let away = Mat4::from_translation(Vec3::new(10.0, 0.0, 0.4));
    scene.set_world(&gpu, cube_instance, away);
    let mut alive: Vec<bool> = vec![true; occluded.len()];
    for frame in 0..LEASH {
        let c = run_frame(&mut shadows, &mut alloc, &scene);
        assert_eq!(
            c.requests_high, 0,
            "C{frame}: recovery must not use the high queue"
        );
        for (k, &i) in occluded.iter().enumerate() {
            if !alive[k] {
                continue;
            }
            ages[k] += 1;
            let word = words(i);
            if ages[k] >= LEASH {
                assert_eq!(
                    local_shadow_state(word),
                    SHADOW_STATE_VISIBLE,
                    "C{frame}: leash re-proof must un-shadow texel {i}"
                );
                alive[k] = false;
            } else {
                assert_eq!(
                    local_shadow_state(word),
                    SHADOW_STATE_OCCLUDED,
                    "C{frame}: pre-leash texel {i} holds (documented staleness)"
                );
            }
        }
    }
    assert!(
        alive.iter().all(|&a| !a),
        "one full leash must recover every blind texel"
    );

    println!(
        "V5 blind-leash proof: texels={TEXELS} blind={} leash={LEASH}",
        occluded.len()
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
