//! Hardware test for deterministic visibility-edge promotion.
//! Only mixed neighborhoods re-ray; interior history remains untouched.

use std::sync::Mutex;

use abi_core::GpuPtr;
use abi_core::glam::{Mat4, UVec2, Vec2, Vec3};
use abi_core::oct_encode;
use abi_light::PointLight;
use abi_light::{LOCAL_SHADOW_SLOT_EMPTY, LOCAL_SHADOW_SLOTS, local_shadow_state};
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
const W: u32 = 16;
const H: u32 = 16;
const HALF_W: u32 = 8;
const HALF_H: u32 = 8;
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
fn v4_edge_promotion_rays_exactly_the_mixed_neighborhoods() {
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

    let run_frame = |shadows: &mut LocalShadowPass, alloc: &mut TestAlloc, edges: bool| {
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
                ray_budget: 0,
                edge_promotion: edges,
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

    let c = run_frame(&mut shadows, &mut alloc, false);
    assert_eq!(c.requests_high, TEXELS as u32);
    assert_eq!(c.promoted, 0);
    for (i, &occ) in oracle.iter().enumerate() {
        let want = if occ {
            SHADOW_STATE_OCCLUDED
        } else {
            SHADOW_STATE_VISIBLE
        };
        assert_eq!(local_shadow_state(words(i)), want, "A: texel {i}");
    }

    let mixed = |tx: i32, ty: i32| -> bool {
        let mut vis = false;
        let mut occ = false;
        for dy in -1..=1i32 {
            for dx in -1..=1i32 {
                let nx = (tx + dx).clamp(0, HALF_W as i32 - 1);
                let ny = (ty + dy).clamp(0, HALF_H as i32 - 1);
                if oracle[(ny * HALF_W as i32 + nx) as usize] {
                    occ = true;
                } else {
                    vis = true;
                }
            }
        }
        vis && occ
    };
    let promoted_expected: Vec<bool> = (0..TEXELS)
        .map(|i| mixed((i as u32 % HALF_W) as i32, (i as u32 / HALF_W) as i32))
        .collect();
    let promoted_count = promoted_expected.iter().filter(|&&p| p).count() as u32;
    assert!(
        promoted_count >= 4 && promoted_count < TEXELS as u32,
        "degenerate edge set: {promoted_count}"
    );

    let c = run_frame(&mut shadows, &mut alloc, true);
    assert_eq!(c.promoted, promoted_count, "B: promoted set mismatch");
    assert_eq!(
        c.requests_high, promoted_count,
        "B: promoted must re-ray high"
    );
    assert_eq!(c.requests_low, 0);
    assert_eq!(
        c.validated + c.reused,
        TEXELS as u32 - promoted_count,
        "B: the interior must stay history-fed"
    );
    for i in 0..TEXELS {
        let (tx, ty) = (i as u32 % HALF_W, i as u32 / HALF_W);
        let rep = UVec2::new(tx * 2, ty * 2);
        let expected = oracle_occluded(&cube, home.inverse(), receiver(rep.x, rep.y), &light);
        let want = if expected {
            SHADOW_STATE_OCCLUDED
        } else {
            SHADOW_STATE_VISIBLE
        };
        assert_eq!(
            local_shadow_state(words(i)),
            want,
            "B: texel {i} at pinned rep {rep:?}"
        );
    }

    let c = run_frame(&mut shadows, &mut alloc, false);
    assert_eq!(c.promoted, 0, "C: disabled promotion must not run");
    assert_eq!(c.requests_high + c.requests_low, 0, "C: all history-fed");

    println!("V4 edge proof: texels={TEXELS} promoted={promoted_count}");

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
