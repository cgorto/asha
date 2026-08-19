//! Hardware test for bounded shadow-ray budgets and convergence.
//! Unproven slots remain visible rather than producing false shadows.

use std::sync::Mutex;

use abi_core::GpuPtr;
use abi_core::glam::{Mat4, UVec2, Vec2, Vec3};
use abi_core::oct_encode;
use abi_light::PointLight;
use abi_light::{
    LOCAL_SHADOW_AGE_MAX, LOCAL_SHADOW_SLOT_EMPTY, LOCAL_SHADOW_SLOTS, local_shadow_age,
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
fn v3_budget_makes_cost_flat_and_convergence_bounded() {
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

    let run_frame = |shadows: &mut LocalShadowPass, alloc: &mut TestAlloc, budget: u32| {
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
            LocalShadowTemporal {
                refresh_interval: 100,
                validate_thickness: 1000.0,
                near_plane: 1.0,
                light_epsilon: 0.25,
                contact: None,
                contact_distance: 0.0,
                ray_budget: budget,
                edge_promotion: false,
                occluded_refresh: 0,
                source_radius: 0.0,
            },
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
    let oracle = oracle_texels(&cube, home, &light);
    let occ = oracle.iter().filter(|&&o| o).count();
    assert!(occ >= 2, "degenerate scene");

    const BUDGET: u32 = 5;
    let mut proven = 0usize;
    for frame in 0..4 {
        let c = run_frame(&mut shadows, &mut alloc, BUDGET);
        let serviced = c.serviced_high + c.serviced_low;
        assert!(
            serviced <= BUDGET,
            "frame {frame}: budget exceeded ({serviced})"
        );
        assert_eq!(c.overflow, 0);
        proven += serviced as usize;
        let mut fresh = 0;
        let mut unproven = 0;
        for i in 0..TEXELS {
            let word = words(i);
            let state = local_shadow_state(word);
            assert!(
                state == SHADOW_STATE_VISIBLE || state == SHADOW_STATE_OCCLUDED,
                "frame {frame}: texel {i} must never be loud under budget"
            );
            if state == SHADOW_STATE_OCCLUDED {
                assert!(oracle[i], "frame {frame}: false shadow at texel {i}");
            }
            if local_shadow_age(word) == LOCAL_SHADOW_AGE_MAX {
                unproven += 1;
            } else if local_shadow_age(word) == 0 && state == SHADOW_STATE_OCCLUDED {
                fresh += 1;
            }
        }
        assert!(
            unproven <= TEXELS - proven.min(TEXELS),
            "frame {frame}: unproven count must shrink with the budget ({unproven})"
        );
        let _ = fresh;
    }
    assert!(proven >= TEXELS, "four budget-5 frames must cover 16 slots");
    for (i, &o) in oracle.iter().enumerate() {
        let want = if o {
            SHADOW_STATE_OCCLUDED
        } else {
            SHADOW_STATE_VISIBLE
        };
        assert_eq!(
            local_shadow_state(words(i)),
            want,
            "converged mismatch at {i}"
        );
    }

    let c = run_frame(&mut shadows, &mut alloc, BUDGET);
    assert_eq!(
        c.requests_high + c.requests_low,
        0,
        "steady state must not need the budget at all"
    );
    assert_eq!(c.validated as usize, occ);
    assert_eq!(c.reused as usize, TEXELS - occ);

    println!(
        "V3 budget proof: texels={TEXELS} occluded={occ} budget={BUDGET} bytes={}",
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
