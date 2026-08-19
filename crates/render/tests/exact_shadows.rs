//! Hardware test for dense per-light visibility and its lighting consumer.
//! A CPU oracle checks static and moved scenes through the real GPU passes.

use std::sync::Mutex;

use abi_core::GpuPtr;
use abi_core::glam::{Mat4, UVec2, Vec2, Vec3};
use abi_core::oct_encode;
use abi_light::PointLight;
use abi_light::{
    SHADOW_STATE_OCCLUDED, SHADOW_STATE_VISIBLE, ShadowSegment, shadow_segment_triangle_oracle,
};
use gpu::pass::{FrameAlloc, Pass};
use gpu::{
    Gpu, HazardFlags, Memory, Queue, SamplerDesc, Stage, TextureDesc, TextureFormat,
    TextureViewDesc, UsageFlags,
};
use mesh::{
    MaterialEntry, MeshLightField, MeshRasterView, MeshScene, MeshSceneDesc, ShadowBlasDesc,
};
use render::{LocalLightPass, MeshShadowPass, MeshSurfaceTargets};

static GPU_TEST_LOCK: Mutex<()> = Mutex::new(());
const W: u32 = 8;
const H: u32 = 8;

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

#[test]
fn exact_mask_matches_cpu_and_gates_local_light() {
    let _guard = GPU_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let gpu = Gpu::new(true).expect("vulkan init");
    let size = UVec2::new(W, H);
    let pixels = (W * H) as usize;
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
    let make_hdr = |heap: &mut gpu::HeapSlots| {
        let texture = gpu.texture_alloc_and_create(
            TextureDesc {
                dimensions: [W, H, 1],
                format: TextureFormat::Rgba16Float,
                usage: UsageFlags::STORAGE | UsageFlags::TRANSFER_DST | UsageFlags::TRANSFER_SRC,
                ..Default::default()
            },
            Queue::Main,
            None,
        );
        let slot = heap.add_storage(
            &gpu,
            gpu.texture_rw_view_descriptor(texture.texture, TextureViewDesc::default()),
        );
        (texture, slot)
    };
    let (shadowed_hdr, shadowed_rw) = make_hdr(&mut heap);
    let (unshadowed_hdr, unshadowed_rw) = make_hdr(&mut heap);

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

    let light = PointLight {
        position: [0.0, 0.0, -2.0],
        radius: 10.0,
        color: [1.0, 0.7, 0.4],
        intensity: 12.0,
    };
    let light_buf = gpu.alloc::<PointLight>(Memory::Default);
    unsafe { *light_buf.cpu = light };

    let normal = oct_encode(Vec3::NEG_Z);
    let normal_up = gpu.alloc_slice::<[u16; 2]>(pixels as u64, Memory::Default);
    let albedo_up = gpu.alloc_slice::<[u16; 4]>(pixels as u64, Memory::Default);
    let marker_up = gpu.alloc_slice::<f32>(pixels as u64, Memory::Default);
    let depth_up = gpu.alloc_slice::<f32>(pixels as u64, Memory::Default);
    let hdr_up = gpu.alloc_slice::<[u16; 4]>(pixels as u64, Memory::Default);
    unsafe {
        for i in 0..pixels {
            *normal_up.cpu.add(i) = [f32_to_f16(normal.x), f32_to_f16(normal.y)];
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

    let mask_read = gpu.alloc_slice::<u32>(pixels as u64, Memory::Readback);
    let shadowed_read = gpu.alloc_slice::<[u16; 4]>(pixels as u64, Memory::Readback);
    let unshadowed_read = gpu.alloc_slice::<[u16; 4]>(pixels as u64, Memory::Readback);
    let mut alloc = TestAlloc {
        gpu: &gpu,
        live: Vec::new(),
    };
    let mut exact = MeshShadowPass::new(&gpu, size, 1, 1);
    let local = LocalLightPass::new(&gpu);
    let view = MeshRasterView {
        world_to_clip: Mat4::IDENTITY,
    };

    let cb = gpu.commands_begin(Queue::Main);
    gpu.cmd_copy_to_texture(cb, targets.normal, normal_up.cast());
    gpu.cmd_copy_to_texture(cb, targets.albedo, albedo_up.cast());
    gpu.cmd_copy_to_texture(cb, targets.material, marker_up.cast());
    gpu.cmd_copy_to_texture(cb, depth.texture, depth_up.cast());
    gpu.cmd_copy_to_texture(cb, shadowed_hdr.texture, hdr_up.cast());
    gpu.cmd_copy_to_texture(cb, unshadowed_hdr.texture, hdr_up.cast());
    gpu.cmd_barrier(
        cb,
        Stage::Transfer,
        Stage::RasterColorOut,
        HazardFlags::empty(),
    );

    local.record(
        &gpu,
        cb,
        &mut alloc,
        &heap,
        &scene,
        &surfaces,
        view,
        light_buf.gpu,
        1,
        0.0,
        MeshLightField::default(),
        None,
        None,
        sampler,
        depth_slot,
        unshadowed_rw,
    );
    let (mask, tlas) = exact.record(
        &gpu,
        cb,
        &mut alloc,
        &heap,
        &scene,
        scene.instances(),
        &surfaces,
        view,
        light_buf.gpu,
        1,
        1.0e-3,
        1.0e-3,
        depth_slot,
    );
    assert_eq!(tlas.instance_count, 1);
    assert!(tlas.topology_rebuilt);
    local.record(
        &gpu,
        cb,
        &mut alloc,
        &heap,
        &scene,
        &surfaces,
        view,
        light_buf.gpu,
        1,
        0.0,
        MeshLightField::default(),
        Some(mask),
        None,
        sampler,
        depth_slot,
        shadowed_rw,
    );
    gpu.cmd_barrier(cb, Stage::Compute, Stage::Transfer, HazardFlags::empty());
    gpu.cmd_mem_copy_raw(
        cb,
        mask_read.cast(),
        exact.states_buffer().cast(),
        pixels as u64 * 4,
    );
    gpu.cmd_copy_texture_to_buffer(cb, shadowed_hdr.texture, shadowed_read.cast());
    gpu.cmd_copy_texture_to_buffer(cb, unshadowed_hdr.texture, unshadowed_read.cast());
    gpu.queue_submit(Queue::Main, &[cb]);
    gpu.queue_wait_idle(Queue::Main);

    let mut occluded = 0;
    let mut visible = 0;
    for y in 0..H {
        for x in 0..W {
            let i = (y * W + x) as usize;
            let ndc = (Vec2::new(x as f32, y as f32) + Vec2::splat(0.5))
                / Vec2::new(W as f32, H as f32)
                * 2.0
                - Vec2::ONE;
            let receiver = Vec3::new(ndc.x, ndc.y, 0.8);
            let segment =
                ShadowSegment::between(receiver, Vec3::from_array(light.position), 1.0e-3, 1.0e-3);
            let expected_hit = cube.indices.chunks_exact(3).any(|triangle| {
                shadow_segment_triangle_oracle(
                    &segment,
                    Vec3::from_array(cube.positions[triangle[0] as usize]),
                    Vec3::from_array(cube.positions[triangle[1] as usize]),
                    Vec3::from_array(cube.positions[triangle[2] as usize]),
                )
            });
            let state = unsafe { *mask_read.cpu.add(i) };
            assert_eq!(
                state,
                if expected_hit {
                    SHADOW_STATE_OCCLUDED
                } else {
                    SHADOW_STATE_VISIBLE
                },
                "mask mismatch at ({x},{y})"
            );
            let shadowed = unsafe { *shadowed_read.cpu.add(i) };
            let unshadowed = unsafe { *unshadowed_read.cpu.add(i) };
            let shadowed_rgb = Vec3::new(
                f16_to_f32(shadowed[0]),
                f16_to_f32(shadowed[1]),
                f16_to_f32(shadowed[2]),
            );
            let unshadowed_rgb = Vec3::new(
                f16_to_f32(unshadowed[0]),
                f16_to_f32(unshadowed[1]),
                f16_to_f32(unshadowed[2]),
            );
            if expected_hit {
                occluded += 1;
                assert_eq!(shadowed_rgb, Vec3::ZERO, "shadow leaked at ({x},{y})");
                assert!(
                    unshadowed_rgb.max_element() > 0.05,
                    "unshadowed control is dark at ({x},{y})"
                );
            } else {
                visible += 1;
                assert!(
                    (shadowed_rgb - unshadowed_rgb).abs().max_element() < 2.0e-3,
                    "visible local light changed at ({x},{y})"
                );
            }
            assert_eq!(shadowed[3], f32_to_f16(0.5));
            assert_eq!(unshadowed[3], f32_to_f16(0.5));
        }
    }
    assert!(
        occluded >= 4 && visible >= 4,
        "weak mask: {occluded}/{visible}"
    );
    let first_states = unsafe { std::slice::from_raw_parts(mask_read.cpu, pixels) }.to_vec();
    let moved_world = Mat4::from_translation(Vec3::new(0.55, 0.0, 0.0));
    scene.set_world(&gpu, cube_instance, moved_world);
    let moved_light = PointLight {
        position: [-0.65, 0.0, -2.0],
        ..light
    };
    unsafe { *light_buf.cpu = moved_light };
    let cb = gpu.commands_begin(Queue::Main);
    let (_, refit) = exact.record(
        &gpu,
        cb,
        &mut alloc,
        &heap,
        &scene,
        scene.instances(),
        &surfaces,
        view,
        light_buf.gpu,
        1,
        1.0e-3,
        1.0e-3,
        depth_slot,
    );
    assert!(
        !refit.topology_rebuilt,
        "motion must refit stable TLAS topology"
    );
    gpu.cmd_barrier(cb, Stage::Compute, Stage::Transfer, HazardFlags::empty());
    gpu.cmd_mem_copy_raw(
        cb,
        mask_read.cast(),
        exact.states_buffer().cast(),
        pixels as u64 * 4,
    );
    gpu.queue_submit(Queue::Main, &[cb]);
    gpu.queue_wait_idle(Queue::Main);

    let mut changed = 0;
    for y in 0..H {
        for x in 0..W {
            let i = (y * W + x) as usize;
            let ndc = (Vec2::new(x as f32, y as f32) + Vec2::splat(0.5))
                / Vec2::new(W as f32, H as f32)
                * 2.0
                - Vec2::ONE;
            let world_segment = ShadowSegment::between(
                Vec3::new(ndc.x, ndc.y, 0.8),
                Vec3::from_array(moved_light.position),
                1.0e-3,
                1.0e-3,
            );
            let local_segment = world_segment.transformed(moved_world.inverse());
            let expected_hit = cube.indices.chunks_exact(3).any(|triangle| {
                shadow_segment_triangle_oracle(
                    &local_segment,
                    Vec3::from_array(cube.positions[triangle[0] as usize]),
                    Vec3::from_array(cube.positions[triangle[1] as usize]),
                    Vec3::from_array(cube.positions[triangle[2] as usize]),
                )
            });
            let state = unsafe { *mask_read.cpu.add(i) };
            assert_eq!(
                state,
                if expected_hit {
                    SHADOW_STATE_OCCLUDED
                } else {
                    SHADOW_STATE_VISIBLE
                },
                "moved mask mismatch at ({x},{y})"
            );
            changed += u32::from(state != first_states[i]);
        }
    }
    assert!(
        changed > 0,
        "moving the instance and light left a stale mask"
    );
    println!(
        "Exact shadow mask: pixels={} occluded={occluded} visible={visible} moved_changes={changed} bytes={}",
        pixels,
        exact.allocated_bytes()
    );

    local.free(&gpu);
    exact.free(&gpu);
    alloc.free();
    gpu.free(mask_read);
    gpu.free(shadowed_read);
    gpu.free(unshadowed_read);
    gpu.free(light_buf);
    gpu.free(normal_up);
    gpu.free(albedo_up);
    gpu.free(marker_up);
    gpu.free(depth_up);
    gpu.free(hdr_up);
    gpu.texture_free_and_destroy(depth);
    gpu.texture_free_and_destroy(shadowed_hdr);
    gpu.texture_free_and_destroy(unshadowed_hdr);
    surfaces.free(&gpu);
    scene.free(&gpu);
    heap.free(&gpu);
}
