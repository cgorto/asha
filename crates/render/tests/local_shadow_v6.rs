//! Hardware test for deterministic spatial penumbra reconstruction.
//! GPU fractions must match the CPU replay and remain byte-stable.

use std::sync::Mutex;

use abi_core::GpuPtr;
use abi_core::glam::{Mat4, UVec2, Vec3};
use abi_core::oct_encode;
use abi_light::PointLight;
use abi_light::{
    LOCAL_SHADOW_FRACTION_ONE, LOCAL_SHADOW_SLOTS, local_shadow_fraction, local_shadow_hit_q,
    local_shadow_state,
};
use abi_light::{SHADOW_STATE_OCCLUDED, SHADOW_STATE_VISIBLE};
use gpu::pass::{FrameAlloc, Pass};
use gpu::{
    Gpu, HazardFlags, Memory, Queue, Stage, TextureDesc, TextureFormat, TextureViewDesc, UsageFlags,
};
use mesh::{MaterialEntry, MeshRasterView, MeshScene, MeshSceneDesc, ShadowBlasDesc};
use render::{LocalShadowPass, LocalShadowTemporal, MeshSurfaceTargets};

static GPU_TEST_LOCK: Mutex<()> = Mutex::new(());
const W: u32 = 16;
const H: u32 = 16;
const HALF_W: u32 = 8;
const HALF_H: u32 = 8;
const TEXELS: usize = (HALF_W * HALF_H) as usize;
const SLOTS: usize = TEXELS * LOCAL_SHADOW_SLOTS as usize;
const K: usize = LOCAL_SHADOW_SLOTS as usize;
const ORIGIN_BIAS: f32 = 1.0e-3;
const DESTINATION_BIAS: f32 = 1.0e-3;
const NEAR_PLANE: f32 = 1.0;
const RECEIVER_Z: f32 = 0.8;
const SOURCE_RADIUS: f32 = 0.35;
/// The blur kernel's constants, mirrored: scan radius and radius cap.
const SCAN: i32 = 3;

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

/// Replays `local_shadow_blur` over read-back state words.
fn blur_replay(states: &[u32]) -> Vec<u32> {
    let mut fractions: Vec<u32> = states
        .chunks_exact(K)
        .flat_map(|texel| {
            texel.iter().map(|&word| {
                if local_shadow_state(word) == SHADOW_STATE_OCCLUDED {
                    0
                } else {
                    LOCAL_SHADOW_FRACTION_ONE
                }
            })
        })
        .collect();

    let own_linear = 1.0f32 / RECEIVER_Z;
    let eye = NEAR_PLANE * own_linear;
    let texel_world = 4.0 * eye / (1.0 * H as f32);
    for ty in 0..HALF_H as i32 {
        for tx in 0..HALF_W as i32 {
            let texel_index = (ty * HALF_W as i32 + tx) as usize;
            let own_word = states[texel_index * K];
            let own_state = local_shadow_state(own_word);
            if own_state != SHADOW_STATE_VISIBLE && own_state != SHADOW_STATE_OCCLUDED {
                continue;
            }
            let mut seen_visible = false;
            let mut seen_occluded = false;
            let ring = [
                (-SCAN, -SCAN),
                (0, -SCAN),
                (SCAN, -SCAN),
                (-SCAN, 0),
                (SCAN, 0),
                (-SCAN, SCAN),
                (0, SCAN),
                (SCAN, SCAN),
            ];
            let core = (0..9).map(|p| ((p % 3) - 1, (p / 3) - 1));
            for (dx, dy) in core.chain(ring.into_iter()) {
                let nx = (tx + dx).clamp(0, HALF_W as i32 - 1);
                let ny = (ty + dy).clamp(0, HALF_H as i32 - 1);
                let state = local_shadow_state(states[(ny * HALF_W as i32 + nx) as usize * K]);
                if state == SHADOW_STATE_VISIBLE {
                    seen_visible = true;
                } else if state == SHADOW_STATE_OCCLUDED {
                    seen_occluded = true;
                }
            }
            if !(seen_visible && seen_occluded) {
                continue;
            }
            let mut min_hit_q = 0xFFFFu32;
            for dy in -SCAN..=SCAN {
                for dx in -SCAN..=SCAN {
                    let nx = (tx + dx).clamp(0, HALF_W as i32 - 1);
                    let ny = (ty + dy).clamp(0, HALF_H as i32 - 1);
                    let word = states[(ny * HALF_W as i32 + nx) as usize * K];
                    if local_shadow_state(word) == SHADOW_STATE_OCCLUDED {
                        min_hit_q = min_hit_q.min(local_shadow_hit_q(word));
                    }
                }
            }
            let t_blocker = (min_hit_q as f32 / 65535.0).min(0.98);
            let penumbra_world = SOURCE_RADIUS * t_blocker / (1.0 - t_blocker);
            let radius_f = penumbra_world / texel_world.max(1.0e-6);
            let radius = (radius_f + 0.5).clamp(1.0, SCAN as f32) as i32;
            let mut sum = 0.0f32;
            let mut weight_sum = 0.0f32;
            for dy in -radius..=radius {
                for dx in -radius..=radius {
                    let nx = (tx + dx).clamp(0, HALF_W as i32 - 1);
                    let ny = (ty + dy).clamp(0, HALF_H as i32 - 1);
                    let word = states[(ny * HALF_W as i32 + nx) as usize * K];
                    let state = local_shadow_state(word);
                    if state != SHADOW_STATE_VISIBLE && state != SHADOW_STATE_OCCLUDED {
                        continue;
                    }
                    let chebyshev = dx.abs().max(dy.abs()) as f32;
                    let weight = 1.0 - chebyshev / (radius as f32 + 1.0);
                    if state == SHADOW_STATE_VISIBLE {
                        sum += weight;
                    }
                    weight_sum += weight;
                }
            }
            if weight_sum > 0.0 {
                fractions[texel_index * K] =
                    (sum / weight_sum * LOCAL_SHADOW_FRACTION_ONE as f32) as u32;
            }
        }
    }
    fractions
}

#[test]
fn v6_penumbra_is_deterministic_and_matches_the_cpu_replay() {
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

    let cube = mesh::primitives::cube(0.5);
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
    let home = Mat4::from_scale_rotation_translation(
        Vec3::new(1.0, 2.0, 0.2),
        abi_core::glam::Quat::IDENTITY,
        Vec3::new(-0.5, 0.0, 0.4),
    );
    scene.add_instance(&gpu, cube_mesh, home, material);

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
            *depth_up.cpu.add(i) = RECEIVER_Z;
        }
    }

    let state_read = gpu.alloc_slice::<u32>(SLOTS as u64, Memory::Readback);
    let fraction_read = gpu.alloc_slice::<u32>(SLOTS as u64, Memory::Readback);
    let mut alloc = TestAlloc {
        gpu: &gpu,
        live: Vec::new(),
    };
    let mut shadows = LocalShadowPass::new(&gpu, size, 1, 1, 1);
    let view = MeshRasterView {
        world_to_clip: Mat4::IDENTITY,
    };
    let temporal = |source_radius: f32| LocalShadowTemporal {
        refresh_interval: 1000,
        validate_thickness: 0.05,
        near_plane: NEAR_PLANE,
        light_epsilon: 0.25,
        contact: None,
        contact_distance: 0.0,
        ray_budget: 0,
        edge_promotion: true,
        occluded_refresh: 0,
        source_radius,
    };

    let run = |shadows: &mut LocalShadowPass, alloc: &mut TestAlloc, radius: f32| {
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
            &scene,
            scene.instances(),
            &surfaces,
            view,
            &lights,
            ORIGIN_BIAS,
            DESTINATION_BIAS,
            0.0,
            temporal(radius),
            depth_slot,
            0,
        );
        gpu.cmd_barrier(cb, Stage::Compute, Stage::Transfer, HazardFlags::empty());
        gpu.cmd_mem_copy_raw(
            cb,
            state_read.cast(),
            shadows.slot_state_buffer().cast(),
            SLOTS as u64 * 4,
        );
        gpu.cmd_mem_copy_raw(
            cb,
            fraction_read.cast(),
            shadows.slot_fraction_ptr().cast(),
            SLOTS as u64 * 4,
        );
        gpu.queue_submit(Queue::Main, &[cb]);
        gpu.queue_wait_idle(Queue::Main);
    };

    let states = || -> Vec<u32> {
        (0..SLOTS)
            .map(|i| unsafe { *state_read.cpu.add(i) })
            .collect()
    };
    let fractions = || -> Vec<u32> {
        (0..SLOTS)
            .map(|i| unsafe { *fraction_read.cpu.add(i) })
            .collect()
    };

    run(&mut shadows, &mut alloc, SOURCE_RADIUS);
    let states_a = states();
    let fractions_a = fractions();
    let occluded = states_a
        .iter()
        .step_by(2)
        .filter(|&&w| local_shadow_state(w) == SHADOW_STATE_OCCLUDED)
        .count();
    assert!(
        occluded >= 8 && occluded <= TEXELS - 8,
        "degenerate edge scene: {occluded} occluded texels"
    );
    let replay = blur_replay(&states_a);
    let mut softened = 0;
    for i in 0..SLOTS {
        assert_eq!(
            local_shadow_fraction(fractions_a[i]),
            local_shadow_fraction(replay[i]),
            "A: fraction mismatch at slot {i}"
        );
        let f = local_shadow_fraction(fractions_a[i]);
        if f != 0 && f != LOCAL_SHADOW_FRACTION_ONE {
            softened += 1;
        }
    }
    assert!(softened >= 8, "A: the edge band must soften ({softened})");

    run(&mut shadows, &mut alloc, SOURCE_RADIUS);
    let fractions_b = fractions();
    assert_eq!(
        fractions_a, fractions_b,
        "B: a static scene must reproduce its fraction buffer exactly"
    );

    run(&mut shadows, &mut alloc, 0.0);
    for (i, &word) in fractions().iter().enumerate() {
        let f = local_shadow_fraction(word);
        assert!(
            f == 0 || f == LOCAL_SHADOW_FRACTION_ONE,
            "C: fraction {f} at slot {i} must be hard with source_radius 0"
        );
    }

    println!("V6 penumbra proof: texels={TEXELS} occluded={occluded} softened={softened}");

    shadows.free(&gpu);
    alloc.free();
    gpu.free(state_read);
    gpu.free(fraction_read);
    gpu.free(normal_up);
    gpu.free(marker_up);
    gpu.free(depth_up);
    gpu.texture_free_and_destroy(depth);
    surfaces.free(&gpu);
    scene.free(&gpu);
    heap.free(&gpu);
}
