//! Hardware test for depth contact marching and validation handoff.
//! It verifies the depth-only occlusion path and its disabled behavior.

use std::sync::Mutex;

use abi_core::GpuPtr;
use abi_core::glam::{IVec2, Mat4, UVec2, Vec2, Vec3};
use abi_core::oct_encode;
use abi_light::PointLight;
use abi_light::{DepthMarchConfig, LOCAL_SHADOW_SLOTS, local_shadow_hit_q, local_shadow_state};
use abi_light::{SHADOW_STATE_OCCLUDED, SHADOW_STATE_VISIBLE, ShadowSegment};
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
const PLANE_DEPTH: f32 = 0.8;
const BAND_DEPTH: f32 = 0.9;
const DEPTH_BIAS: f32 = 0.000002;

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
    Vec3::new(ndc.x, ndc.y, depth_at(x as i32, y as i32))
}

/// Synthetic receiver depth with a nearer band on the right.
fn depth_at(x: i32, y: i32) -> f32 {
    let x = x.clamp(0, W as i32 - 1);
    let _ = y;
    if x >= 6 { BAND_DEPTH } else { PLANE_DEPTH }
}

fn linearized(depth: f32) -> f32 {
    if depth > 0.0 {
        1.0 / depth
    } else {
        f32::INFINITY
    }
}

/// CPU replay of `depth_point_and_linear` over the synthetic depth.
fn point_and_linear(uv: Vec2) -> (f32, f32) {
    let size = Vec2::new(W as f32, H as f32);
    let texel = uv * size - Vec2::splat(0.5);
    let base = texel.floor();
    let frac = texel - base;
    let b = IVec2::new(base.x as i32, base.y as i32);
    let d00 = depth_at(b.x, b.y);
    let d10 = depth_at(b.x + 1, b.y);
    let d01 = depth_at(b.x, b.y + 1);
    let d11 = depth_at(b.x + 1, b.y + 1);
    let row0 = d00 + (d10 - d00) * frac.x;
    let row1 = d01 + (d11 - d01) * frac.x;
    let linear = row0 + (row1 - row0) * frac.y;
    let p = (uv * size).floor();
    let point = depth_at(p.x as i32, p.y as i32);
    (point, linear)
}

/// Replays contact marching under the identity projection.
fn cpu_contact(
    from: Vec3,
    light: &PointLight,
    config: &DepthMarchConfig,
    contact_distance: f32,
) -> Option<f32> {
    let segment = ShadowSegment::between(
        from,
        Vec3::from_array(light.position),
        ORIGIN_BIAS,
        DESTINATION_BIAS,
    );
    if !segment.is_active() {
        return None;
    }
    let direction = Vec3::from_array(segment.direction);
    let span = direction.length();
    let reach = (segment.t_min + contact_distance / span).min(segment.t_max);
    if reach <= segment.t_min {
        return None;
    }
    let origin = Vec3::from_array(segment.origin);
    let start = origin + direction * segment.t_min;
    let end = origin + direction * reach;
    for p in [start, end] {
        assert!(
            p.x.abs() <= 1.0 && p.y.abs() <= 1.0 && p.z > 0.0 && p.z <= 1.0,
            "test geometry must not clip: {p:?}"
        );
    }
    let delta = end - start;
    let start_uv = start.truncate() * 0.5 + Vec2::splat(0.5);
    let end_uv = end.truncate() * 0.5 + Vec2::splat(0.5);
    let ray_pixels = (end_uv - start_uv) * Vec2::new(W as f32, H as f32);
    let pixel_steps = ray_pixels.length() as u32;
    let step_count = config.linear_steps.min(pixel_steps).max(2);
    let thickness = config.depth_thickness / config.near_plane;
    for step in 0..step_count {
        let candidate_t = (step as f32 + config.jitter) / step_count as f32;
        let candidate = start + delta * candidate_t;
        let uv = candidate.truncate() * 0.5 + Vec2::splat(0.5);
        let (point_sample, linear_sample) = point_and_linear(uv);
        let linear_depth = linearized(linear_sample);
        let point_depth = linearized(point_sample);
        let far_surface = linear_depth.max(point_depth);
        let near_surface = linear_depth.min(point_depth);
        let ray_depth = linearized(candidate.z);
        let distance = far_surface * (1.0 + DEPTH_BIAS) - ray_depth;
        let penetration = ray_depth - near_surface;
        let valid = config.continue_after_deep_penetration == 0 || penetration < thickness;
        if distance < 0.0 && valid {
            if penetration < thickness && distance < thickness {
                let t_world = segment.t_min + candidate_t * (reach - segment.t_min);
                return Some(t_world);
            }
            if config.continue_after_deep_penetration == 0 {
                return None;
            }
        }
    }
    None
}

#[test]
fn v2_contact_march_occludes_from_depth_and_hands_off_to_validation() {
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
    scene.add_instance(
        &gpu,
        cube_mesh,
        Mat4::from_translation(Vec3::new(100.0, 0.0, 0.5)),
        material,
    );

    let light = PointLight {
        position: [0.9, 0.0, 0.95],
        radius: 10.0,
        color: [1.0, 0.7, 0.4],
        intensity: 12.0,
    };
    let lights = [light];

    let normal_oct = oct_encode(Vec3::Z);
    let normal_up = gpu.alloc_slice::<[u16; 2]>(pixels as u64, Memory::Default);
    let marker_up = gpu.alloc_slice::<f32>(pixels as u64, Memory::Default);
    let depth_up = gpu.alloc_slice::<f32>(pixels as u64, Memory::Default);
    unsafe {
        for y in 0..H {
            for x in 0..W {
                let i = (y * W + x) as usize;
                *normal_up.cpu.add(i) = [f32_to_f16(normal_oct.x), f32_to_f16(normal_oct.y)];
                *marker_up.cpu.add(i) = 1.0;
                *depth_up.cpu.add(i) = depth_at(x as i32, y as i32);
            }
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

    let run_frame =
        |shadows: &mut LocalShadowPass, alloc: &mut TestAlloc, temporal: LocalShadowTemporal| {
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

    let march = DepthMarchConfig {
        linear_steps: 8,
        continue_after_deep_penetration: 1,
        jitter: 0.0,
        depth_thickness: 0.2,
        near_plane: 1.0,
        _pad: [0; 3],
    };
    let contact_distance = 3.0;
    let temporal = |refresh: u32, contact: Option<DepthMarchConfig>| LocalShadowTemporal {
        refresh_interval: refresh,
        validate_thickness: 0.2,
        near_plane: 1.0,
        light_epsilon: 0.25,
        contact,
        contact_distance,
        ray_budget: 0,
        edge_promotion: false,
        occluded_refresh: 0,
        source_radius: 0.0,
    };

    let expected: Vec<Option<f32>> = (0..TEXELS)
        .map(|i| {
            let (tx, ty) = (i as u32 % HALF_W, i as u32 / HALF_W);
            cpu_contact(receiver(tx * 2, ty * 2), &light, &march, contact_distance)
        })
        .collect();
    let contact_expected = expected.iter().filter(|e| e.is_some()).count() as u32;
    assert!(
        contact_expected >= 2 && contact_expected < TEXELS as u32,
        "degenerate contact scene: {contact_expected} march hits"
    );

    let c = run_frame(&mut shadows, &mut alloc, temporal(0, Some(march)));
    assert_eq!(
        c.contact, contact_expected,
        "A: march hits must match replay"
    );
    assert_eq!(
        c.requests_high,
        TEXELS as u32 - contact_expected,
        "A: only march misses may ray"
    );
    assert_eq!(c.overflow, 0);
    for (i, exp) in expected.iter().enumerate() {
        let word = unsafe { *state_read.cpu.add(i * LOCAL_SHADOW_SLOTS as usize) };
        match exp {
            Some(t) => {
                assert_eq!(
                    local_shadow_state(word),
                    SHADOW_STATE_OCCLUDED,
                    "A: texel {i} must be march-occluded"
                );
                let expected_q = (t.clamp(0.0, 1.0) * 65535.0) as u32;
                let dq = local_shadow_hit_q(word).abs_diff(expected_q);
                assert!(dq <= 2, "A: texel {i} hit param off by {dq}");
            }
            None => {
                assert_eq!(
                    local_shadow_state(word),
                    SHADOW_STATE_VISIBLE,
                    "A: texel {i} must ray to visible (BLAS is empty here)"
                );
            }
        }
    }

    let c = run_frame(&mut shadows, &mut alloc, temporal(100, Some(march)));
    assert_eq!(
        c.requests_high + c.requests_low,
        0,
        "B: static frame must spend zero rays"
    );
    assert_eq!(
        c.validated, contact_expected,
        "B: march occlusion must survive by validation tap"
    );
    assert_eq!(c.contact, 0, "B: history answers before the march runs");
    assert_eq!(c.reused, TEXELS as u32 - contact_expected);
    for (i, exp) in expected.iter().enumerate() {
        let word = unsafe { *state_read.cpu.add(i * LOCAL_SHADOW_SLOTS as usize) };
        let want = if exp.is_some() {
            SHADOW_STATE_OCCLUDED
        } else {
            SHADOW_STATE_VISIBLE
        };
        assert_eq!(local_shadow_state(word), want, "B: texel {i} state drifted");
    }

    shadows.invalidate_history();
    let c = run_frame(&mut shadows, &mut alloc, temporal(0, None));
    assert_eq!(c.contact, 0, "C: disabled march must not run");
    assert_eq!(c.requests_high, TEXELS as u32, "C: everything rays");
    for i in 0..TEXELS {
        let word = unsafe { *state_read.cpu.add(i * LOCAL_SHADOW_SLOTS as usize) };
        assert_eq!(
            local_shadow_state(word),
            SHADOW_STATE_VISIBLE,
            "C: texel {i} must be visible without the march"
        );
    }

    println!(
        "V2 contact proof: texels={TEXELS} march_occluded={contact_expected} bytes={}",
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
