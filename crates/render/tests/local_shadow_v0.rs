//! Hardware test for half-resolution slot selection, tracing, and resolve.
//! The complete GPU result is compared with an independent CPU replay.

use std::sync::Mutex;

use abi_core::GpuPtr;
use abi_core::glam::{IVec2, Mat4, UVec2, Vec2, Vec3};
use abi_core::oct_encode;
use abi_light::{LOCAL_SHADOW_SLOT_EMPTY, LOCAL_SHADOW_SLOTS, local_shadow_state};
use abi_light::{PointLight, point_light_contribution};
use abi_light::{
    SHADOW_STATE_INACTIVE, SHADOW_STATE_OCCLUDED, SHADOW_STATE_VISIBLE, ShadowSegment,
    shadow_segment_triangle_oracle,
};
use gpu::pass::{FrameAlloc, Pass};
use gpu::{
    Gpu, HazardFlags, Memory, Queue, SamplerDesc, Stage, TextureDesc, TextureFormat,
    TextureViewDesc, UsageFlags,
};
use mesh::{
    MaterialEntry, MeshLightField, MeshRasterView, MeshScene, MeshSceneDesc, ShadowBlasDesc,
};
use render::{LocalLightPass, LocalShadowPass, LocalShadowTemporal, MeshSurfaceTargets};

static GPU_TEST_LOCK: Mutex<()> = Mutex::new(());
const W: u32 = 8;
const H: u32 = 8;
const HALF_W: u32 = 4;
const HALF_H: u32 = 4;
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
    if exp32 == 0xff {
        return sign | 0x7c00 | (u16::from(man != 0) << 9);
    }
    let exp = exp32 - 127 + 15;
    if exp >= 0x1f {
        return sign | 0x7c00;
    }
    if exp <= 0 {
        if exp < -10 {
            return sign;
        }
        let man = man | 0x0080_0000;
        let drop = (14 - exp) as u32;
        let base = man >> drop;
        let rem = man & ((1u32 << drop) - 1);
        let halfway = 1u32 << (drop - 1);
        let round = u32::from(rem > halfway) | (u32::from(rem == halfway) & (base & 1));
        return sign | (base + round) as u16;
    }
    let base = ((exp as u32) << 10) | (man >> 13);
    let rem = man & 0x1fff;
    let round = u32::from(rem > 0x1000) | (u32::from(rem == 0x1000) & (base & 1));
    sign | (base + round) as u16
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits as u32) & 0x8000) << 16;
    let exponent = ((bits >> 10) & 0x1f) as u32;
    let fraction = (bits & 0x03ff) as u32;
    let out = if exponent == 0 {
        if fraction == 0 {
            sign
        } else {
            let mut fraction = fraction;
            let mut exponent = -14i32;
            while fraction & 0x0400 == 0 {
                fraction <<= 1;
                exponent -= 1;
            }
            fraction &= 0x03ff;
            sign | (((exponent + 127) as u32) << 23) | (fraction << 13)
        }
    } else if exponent == 0x1f {
        sign | 0x7f80_0000 | (fraction << 13)
    } else {
        sign | ((exponent + 112) << 23) | (fraction << 13)
    };
    f32::from_bits(out)
}

/// Reconstructs a full-resolution receiver under identity clip coordinates.
fn receiver(x: u32, y: u32) -> Vec3 {
    let ndc = (Vec2::new(x as f32, y as f32) + Vec2::splat(0.5)) / Vec2::new(W as f32, H as f32)
        * 2.0
        - Vec2::ONE;
    Vec3::new(ndc.x, ndc.y, 0.8)
}

/// Evaluates one receiver/light pair against instanced cube triangles.
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

/// Replays top-2 contribution ranking with strict tie handling.
fn expected_slots(
    normal: Vec3,
    position: Vec3,
    lights: &[PointLight],
    wrap_w: f32,
) -> [u32; LOCAL_SHADOW_SLOTS as usize] {
    const K: usize = LOCAL_SHADOW_SLOTS as usize;
    let mut slot = [LOCAL_SHADOW_SLOT_EMPTY; K];
    let mut score = [0.0f32; K];
    for (i, light) in lights.iter().enumerate() {
        let c = point_light_contribution(normal, position, light, wrap_w);
        let s = c.x * 0.2126 + c.y * 0.7152 + c.z * 0.0722;
        if s > score[K - 1] {
            let mut j = K - 1;
            while j > 0 && s > score[j - 1] {
                score[j] = score[j - 1];
                slot[j] = slot[j - 1];
                j -= 1;
            }
            score[j] = s;
            slot[j] = i as u32;
        }
    }
    slot
}

/// Replays guided bilinear visibility resolve with border clamping.
fn expected_visibility(x: u32, y: u32, texel_visibility: &dyn Fn(u32, u32) -> f32) -> f32 {
    let hp = (Vec2::new(x as f32, y as f32) + Vec2::splat(0.5)) * 0.5 - Vec2::splat(0.5);
    let base_f = hp.floor();
    let base = IVec2::new(base_f.x as i32, base_f.y as i32);
    let frac = hp - base_f;
    let hi = IVec2::new(HALF_W as i32 - 1, HALF_H as i32 - 1);
    let mut sum = 0.0f32;
    let mut weight = 0.0f32;
    for corner in 0..4u32 {
        let offset = IVec2::new((corner & 1) as i32, (corner >> 1) as i32);
        let t = (base + offset).clamp(IVec2::ZERO, hi);
        let bx = if offset.x == 0 { 1.0 - frac.x } else { frac.x };
        let by = if offset.y == 0 { 1.0 - frac.y } else { frac.y };
        let bilinear = bx * by;
        if bilinear > 0.0 {
            sum += texel_visibility(t.x as u32, t.y as u32) * bilinear;
            weight += bilinear;
        }
    }
    sum / weight
}

#[test]
fn v0_slot_pipeline_matches_cpu_model() {
    let _guard = GPU_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let gpu = Gpu::new(true).expect("vulkan init");
    let size = UVec2::new(W, H);
    let pixels = (W * H) as usize;
    let texels = (HALF_W * HALF_H) as usize;
    let mut heap = gpu.heap_slots_create(8, 4, 2);
    let sampler = heap.add_sampler(&gpu, gpu.sampler_descriptor(SamplerDesc::default()));

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
    let hdr = gpu.texture_alloc_and_create(
        TextureDesc {
            dimensions: [W, H, 1],
            format: TextureFormat::Rgba16Float,
            usage: UsageFlags::STORAGE | UsageFlags::TRANSFER_DST | UsageFlags::TRANSFER_SRC,
            ..Default::default()
        },
        Queue::Main,
        None,
    );
    let hdr_rw = heap.add_storage(
        &gpu,
        gpu.texture_rw_view_descriptor(hdr.texture, TextureViewDesc::default()),
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
    let cube_instance = scene.add_instance(&gpu, cube_mesh, Mat4::IDENTITY, material);

    let lights = [
        PointLight {
            position: [0.0, 0.0, -2.0],
            radius: 10.0,
            color: [1.0, 0.7, 0.4],
            intensity: 12.0,
        },
        PointLight {
            position: [0.6, 0.4, -3.0],
            radius: 10.0,
            color: [0.3, 0.5, 1.0],
            intensity: 3.0,
        },
    ];
    let light_buf = gpu.alloc_slice::<PointLight>(2, Memory::Default);
    unsafe {
        *light_buf.cpu = lights[0];
        *light_buf.cpu.add(1) = lights[1];
    }

    let surface_normal = Vec3::NEG_Z;
    let normal_oct = oct_encode(surface_normal);
    let normal_up = gpu.alloc_slice::<[u16; 2]>(pixels as u64, Memory::Default);
    let albedo_up = gpu.alloc_slice::<[u16; 4]>(pixels as u64, Memory::Default);
    let marker_up = gpu.alloc_slice::<f32>(pixels as u64, Memory::Default);
    let depth_up = gpu.alloc_slice::<f32>(pixels as u64, Memory::Default);
    let hdr_up = gpu.alloc_slice::<[u16; 4]>(pixels as u64, Memory::Default);
    unsafe {
        for i in 0..pixels {
            *normal_up.cpu.add(i) = [f32_to_f16(normal_oct.x), f32_to_f16(normal_oct.y)];
            *albedo_up.cpu.add(i) = [
                f32_to_f16(1.0),
                f32_to_f16(1.0),
                f32_to_f16(1.0),
                f32_to_f16(1.0),
            ];
            *marker_up.cpu.add(i) = 1.0;
            *depth_up.cpu.add(i) = 0.8;
            *hdr_up.cpu.add(i) = [0, 0, 0, f32_to_f16(0.5)];
        }
    }

    let slot_map_read = gpu.alloc_slice::<u32>(texels as u64, Memory::Readback);
    let slot_vis_read = gpu.alloc_slice::<u32>(
        texels as u64 * u64::from(LOCAL_SHADOW_SLOTS),
        Memory::Readback,
    );
    let slot_rep_read = gpu.alloc_slice::<u32>(texels as u64, Memory::Readback);
    let hdr_read = gpu.alloc_slice::<[u16; 4]>(pixels as u64, Memory::Readback);
    let mut alloc = TestAlloc {
        gpu: &gpu,
        live: Vec::new(),
    };
    let mut shadows = LocalShadowPass::new(&gpu, size, 2, 1, 1);
    assert_eq!(shadows.half(), UVec2::new(HALF_W, HALF_H));
    let local = LocalLightPass::new(&gpu);
    let view = MeshRasterView {
        world_to_clip: Mat4::IDENTITY,
    };

    let record_frame = |shadows: &mut LocalShadowPass,
                        alloc: &mut TestAlloc,
                        scene: &MeshScene,
                        lights: &[PointLight]| {
        let cb = gpu.commands_begin(Queue::Main);
        gpu.cmd_copy_to_texture(cb, targets.normal, normal_up.cast());
        gpu.cmd_copy_to_texture(cb, targets.albedo, albedo_up.cast());
        gpu.cmd_copy_to_texture(cb, targets.material, marker_up.cast());
        gpu.cmd_copy_to_texture(cb, depth.texture, depth_up.cast());
        gpu.cmd_copy_to_texture(cb, hdr.texture, hdr_up.cast());
        gpu.cmd_barrier(
            cb,
            Stage::Transfer,
            Stage::RasterColorOut,
            HazardFlags::empty(),
        );
        let (slots, stats) = shadows.record(
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
            LocalShadowTemporal {
                refresh_interval: 0,
                validate_thickness: 0.05,
                near_plane: 1.0,
                light_epsilon: 0.0,
                contact: None,
                contact_distance: 0.0,
                ray_budget: 0,
                edge_promotion: false,
                occluded_refresh: 0,
                source_radius: 0.0,
            },
            depth_slot,
            0,
        );
        local.record(
            &gpu,
            cb,
            alloc,
            &heap,
            scene,
            &surfaces,
            view,
            light_buf.gpu,
            2,
            0.0,
            MeshLightField::default(),
            None,
            Some(slots),
            sampler,
            depth_slot,
            hdr_rw,
        );
        gpu.cmd_barrier(cb, Stage::Compute, Stage::Transfer, HazardFlags::empty());
        gpu.cmd_mem_copy_raw(
            cb,
            slot_map_read.cast(),
            shadows.slot_map_buffer().cast(),
            texels as u64 * 4,
        );
        gpu.cmd_mem_copy_raw(
            cb,
            slot_vis_read.cast(),
            shadows.slot_state_buffer().cast(),
            texels as u64 * u64::from(LOCAL_SHADOW_SLOTS) * 4,
        );
        gpu.cmd_mem_copy_raw(
            cb,
            slot_rep_read.cast(),
            shadows.slot_rep_buffer().cast(),
            texels as u64 * 4,
        );
        gpu.cmd_copy_texture_to_buffer(cb, hdr.texture, hdr_read.cast());
        gpu.queue_submit(Queue::Main, &[cb]);
        gpu.queue_wait_idle(Queue::Main);
        stats
    };

    let check_frame = |world: Mat4, phase: &str| {
        let world_inverse = world.inverse();
        let mut texel_slots = vec![[0u32; LOCAL_SHADOW_SLOTS as usize]; texels];
        let mut texel_visibility = vec![[1.0f32; LOCAL_SHADOW_SLOTS as usize]; texels];
        for ty in 0..HALF_H {
            for tx in 0..HALF_W {
                let i = (ty * HALF_W + tx) as usize;
                let rep = UVec2::new(tx * 2, ty * 2);
                let position = receiver(rep.x, rep.y);
                let slots = expected_slots(surface_normal, position, &lights, 0.0);
                texel_slots[i] = slots;
                for (s, &slot_light) in slots.iter().enumerate() {
                    if slot_light == LOCAL_SHADOW_SLOT_EMPTY {
                        continue;
                    }
                    texel_visibility[i][s] = if oracle_occluded(
                        &cube,
                        world_inverse,
                        position,
                        &lights[slot_light as usize],
                    ) {
                        0.0
                    } else {
                        1.0
                    };
                }

                let rep_word = unsafe { *slot_rep_read.cpu.add(i) };
                assert_eq!(
                    rep_word,
                    rep.x | (rep.y << 16),
                    "{phase}: representative mismatch at ({tx},{ty})"
                );
                let map_word = unsafe { *slot_map_read.cpu.add(i) };
                let mut expected_word = abi_light::LOCAL_SHADOW_SLOT_WORD_EMPTY;
                for (s, &id) in slots.iter().enumerate() {
                    expected_word = abi_light::local_shadow_slot_set(expected_word, s as u32, id);
                }
                assert_eq!(
                    map_word, expected_word,
                    "{phase}: slot map mismatch at ({tx},{ty})"
                );
                for s in 0..LOCAL_SHADOW_SLOTS as usize {
                    let state = local_shadow_state(unsafe {
                        *slot_vis_read.cpu.add(i * LOCAL_SHADOW_SLOTS as usize + s)
                    });
                    let expected = if slots[s] == LOCAL_SHADOW_SLOT_EMPTY {
                        SHADOW_STATE_INACTIVE
                    } else if texel_visibility[i][s] > 0.5 {
                        SHADOW_STATE_VISIBLE
                    } else {
                        SHADOW_STATE_OCCLUDED
                    };
                    assert_eq!(
                        state, expected,
                        "{phase}: slot {s} state mismatch at ({tx},{ty})"
                    );
                }
            }
        }

        let mut occluded_any = 0u32;
        for y in 0..H {
            for x in 0..W {
                let i = (y * W + x) as usize;
                let position = receiver(x, y);
                let mut expected_direct = Vec3::ZERO;
                let own = ((y / 2) * HALF_W + x / 2) as usize;
                for s in 0..LOCAL_SHADOW_SLOTS as usize {
                    let light_index = texel_slots[own][s];
                    if light_index == LOCAL_SHADOW_SLOT_EMPTY {
                        continue;
                    }
                    let visibility = expected_visibility(x, y, &|tx, ty| {
                        let t = (ty * HALF_W + tx) as usize;
                        let ns = texel_slots[t]
                            .iter()
                            .position(|&id| id == light_index)
                            .expect("both lights reach every texel here");
                        texel_visibility[t][ns]
                    });
                    if visibility < 1.0 {
                        occluded_any += 1;
                    }
                    expected_direct += point_light_contribution(
                        surface_normal,
                        position,
                        &lights[light_index as usize],
                        0.0,
                    ) * visibility;
                }
                let actual = unsafe { *hdr_read.cpu.add(i) };
                let actual_rgb = Vec3::new(
                    f16_to_f32(actual[0]),
                    f16_to_f32(actual[1]),
                    f16_to_f32(actual[2]),
                );
                assert!(
                    (actual_rgb - expected_direct).abs().max_element() < 1.0e-2,
                    "{phase}: resolve mismatch at ({x},{y}): got {actual_rgb:?} want {expected_direct:?}"
                );
                assert_eq!(actual[3], f32_to_f16(0.5), "{phase}: HDR alpha changed");
            }
        }
        assert!(
            occluded_any > 0,
            "{phase}: the cube shadows nothing — the scene is degenerate"
        );
    };

    let stats = record_frame(&mut shadows, &mut alloc, &scene, &lights);
    assert_eq!(stats.instance_count, 1);
    assert!(stats.topology_rebuilt);
    let counters = shadows.take_counters(0);
    assert_eq!(counters.texel_count, HALF_W * HALF_H);
    assert_eq!(counters.active_texels, HALF_W * HALF_H);
    assert_eq!(
        counters.requests_high,
        HALF_W * HALF_H * 2,
        "two reachable lights must fill two slots of every texel"
    );
    assert_eq!(counters.overflow, 0);
    check_frame(Mat4::IDENTITY, "static");

    let first_states = unsafe {
        std::slice::from_raw_parts(slot_vis_read.cpu, texels * LOCAL_SHADOW_SLOTS as usize)
    }
    .to_vec();
    let moved_world = Mat4::from_translation(Vec3::new(0.55, 0.0, 0.0));
    scene.set_world(&gpu, cube_instance, moved_world);
    let moved_light = PointLight {
        position: [-0.65, 0.0, -2.0],
        ..lights[0]
    };
    unsafe { *light_buf.cpu = moved_light };
    let lights = [moved_light, lights[1]];
    let stats = record_frame(&mut shadows, &mut alloc, &scene, &lights);
    assert!(
        !stats.topology_rebuilt,
        "motion must refit stable TLAS topology"
    );
    let check_moved = |world: Mat4, phase: &str| {
        let world_inverse = world.inverse();
        for ty in 0..HALF_H {
            for tx in 0..HALF_W {
                let i = (ty * HALF_W + tx) as usize;
                let rep = UVec2::new(tx * 2, ty * 2);
                let position = receiver(rep.x, rep.y);
                let slots = expected_slots(surface_normal, position, &lights, 0.0);
                for (s, &slot_light) in slots.iter().enumerate() {
                    if slot_light == LOCAL_SHADOW_SLOT_EMPTY {
                        continue;
                    }
                    let state = local_shadow_state(unsafe {
                        *slot_vis_read.cpu.add(i * LOCAL_SHADOW_SLOTS as usize + s)
                    });
                    let expected = if oracle_occluded(
                        &cube,
                        world_inverse,
                        position,
                        &lights[slot_light as usize],
                    ) {
                        SHADOW_STATE_OCCLUDED
                    } else {
                        SHADOW_STATE_VISIBLE
                    };
                    assert_eq!(
                        state, expected,
                        "{phase}: slot {s} state mismatch at ({tx},{ty})"
                    );
                }
            }
        }
    };
    check_moved(moved_world, "moved");
    let second_states = unsafe {
        std::slice::from_raw_parts(slot_vis_read.cpu, texels * LOCAL_SHADOW_SLOTS as usize)
    }
    .to_vec();
    assert_ne!(
        first_states, second_states,
        "moving the occluder and light left stale slot states"
    );

    println!(
        "V0 slot proof: texels={texels} requests={} overflow={} bytes={}",
        counters.requests_high,
        counters.overflow,
        shadows.allocated_bytes()
    );

    local.free(&gpu);
    shadows.free(&gpu);
    alloc.free();
    gpu.free(slot_map_read);
    gpu.free(slot_vis_read);
    gpu.free(slot_rep_read);
    gpu.free(hdr_read);
    gpu.free(light_buf);
    gpu.free(normal_up);
    gpu.free(albedo_up);
    gpu.free(marker_up);
    gpu.free(depth_up);
    gpu.free(hdr_up);
    gpu.texture_free_and_destroy(depth);
    gpu.texture_free_and_destroy(hdr);
    surfaces.free(&gpu);
    scene.free(&gpu);
    heap.free(&gpu);
}

const _: () = assert!(SHADOW_STATE_INACTIVE == 0, "ZII slot states");
