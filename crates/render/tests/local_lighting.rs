//! Hardware test for the mesh local-light MRT and HDR accumulation contract.
//! It checks formats, markers, light ordering, shadow gates, and alpha.

use std::sync::Mutex;

use abi_core::GpuPtr;
use abi_core::glam::{Mat4, UVec2, Vec2, Vec3};
use abi_core::{oct_decode, oct_encode};
use abi_light::{
    PointLight, light_field_gate, light_field_sample, mesh_point_lights_identity,
    point_light_ramp_terms,
};
use gpu::pass::{FrameAlloc, Pass};
use gpu::{
    Gpu, HazardFlags, Memory, Queue, SamplerDesc, Stage, TextureDesc, TextureFormat,
    TextureViewDesc, UsageFlags,
};
use mesh::{MaterialEntry, MeshLightField, MeshRasterView, MeshScene, MeshSceneDesc};
use render::{LocalLightPass, MeshSurfaceTargets};

/// Serializes hardware tests that share one Vulkan device.
static GPU_TEST_LOCK: Mutex<()> = Mutex::new(());

/// Non-multiple dimensions exercise rounded dispatch bounds checks.
const W: u32 = 12;
const H: u32 = 9;
const WRAP_W: f32 = 0.3;

/// Checked pixels cover identity, ramp, and gated lighting paths.
const P_IDENTITY: (u32, u32) = (2, 3); // material 1 (identity ramp), out of field
const P_RAMP: (u32, u32) = (8, 2); // material 2 (constant nonidentity ramp)
const P_GATED: (u32, u32) = (9, 5); // material 1, inside the light field
const P_CLEARED: (u32, u32) = (5, 6); // marked but depth 0.0 → no write

const FIELD_DIMS: [u32; 2] = [2, 2];
const FIELD_CELL: f32 = 0.5;
const FIELD_GATE: f32 = 0.75;
const FIELD_CELLS: [f32; 4] = [0.9, 0.2, 0.5, 0.25];

/// IEEE-754 binary16 decode (mirrors the proven post-test helper).
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

/// Encodes binary16 with round-to-nearest-even.
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

/// Test allocator using host-visible allocations freed at teardown.
struct TestAlloc<'a> {
    gpu: &'a Gpu,
    live: Vec<gpu::Ptr<u8>>,
}

impl FrameAlloc for TestAlloc<'_> {
    fn frame_alloc<T: bytemuck::Pod>(&mut self, value: T) -> GpuPtr<T> {
        let p = self.gpu.alloc::<T>(Memory::Default);
        // SAFETY: fresh host-visible allocation for T.
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
        // SAFETY: fresh host-visible allocation sized for the complete slice.
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

#[test]
fn local_light_pass_holds_the_m1_contract() {
    let _gpu_guard = GPU_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let gpu = Gpu::new(true).expect("vulkan init");
    let size = UVec2::new(W, H);
    let px = |x: u32, y: u32| (y * W + x) as usize;
    let pixels = (W * H) as usize;

    let mut normal_bits = vec![[0u16; 2]; pixels];
    let mut albedo_bits = vec![[0u16; 4]; pixels];
    let mut hdr_bits = vec![[0u16; 4]; pixels];
    let mut markers = vec![0.0f32; pixels];
    let mut depths = vec![0.55f32; pixels];
    for y in 0..H {
        for x in 0..W {
            let i = px(x, y);
            let normal = Vec3::new(0.12 * x as f32 - 0.6, 1.0, 0.1 * y as f32 - 0.35).normalize();
            let oct = oct_encode(normal);
            normal_bits[i] = [f32_to_f16(oct.x), f32_to_f16(oct.y)];
            albedo_bits[i] = [
                f32_to_f16(0.1 + 0.05 * x as f32),
                f32_to_f16(0.6 - 0.03 * y as f32),
                f32_to_f16(0.35 + 0.02 * (x + y) as f32),
                f32_to_f16(0.9),
            ];
            hdr_bits[i] = [
                f32_to_f16(0.02 * x as f32),
                f32_to_f16(0.015 * y as f32),
                f32_to_f16(0.25),
                f32_to_f16(0.1 + 0.03 * x as f32 + 0.02 * y as f32),
            ];
        }
    }
    markers[px(P_IDENTITY.0, P_IDENTITY.1)] = 1.0;
    depths[px(P_IDENTITY.0, P_IDENTITY.1)] = 0.8;
    markers[px(P_RAMP.0, P_RAMP.1)] = 2.0;
    depths[px(P_RAMP.0, P_RAMP.1)] = 0.6;
    markers[px(P_GATED.0, P_GATED.1)] = 1.0;
    depths[px(P_GATED.0, P_GATED.1)] = 0.9;
    markers[px(P_CLEARED.0, P_CLEARED.1)] = 1.0;
    depths[px(P_CLEARED.0, P_CLEARED.1)] = 0.0; // reverse-Z cleared sentinel

    let live_lights = [PointLight {
        position: [0.0, 1.2, 0.4],
        radius: 4.0,
        color: [1.0, 0.6, 0.3],
        intensity: 12.0,
    }];
    let all_lights = [
        live_lights[0],
        PointLight {
            position: [0.1, 0.0, 0.6],
            radius: 0.0,
            color: [9.0, 9.0, 9.0],
            intensity: 5.0,
        },
        PointLight {
            position: [-0.3, -0.4, 0.7],
            radius: 3.0,
            color: [8.0, 8.0, 8.0],
            intensity: 0.0,
        },
        PointLight {
            position: [30.0, -40.0, 25.0],
            radius: 1.5,
            color: [7.0, 7.0, 7.0],
            intensity: 50.0,
        },
    ];

    let ramp_rgb = Vec3::new(0.75, 0.375, 0.25); // fp16-exact constants
    let world_at = |x: u32, y: u32| {
        Vec3::new(
            ((x as f32 + 0.5) / W as f32) * 2.0 - 1.0,
            ((y as f32 + 0.5) / H as f32) * 2.0 - 1.0,
            depths[px(x, y)],
        )
    };
    let decoded_normal = |x: u32, y: u32| {
        let b = normal_bits[px(x, y)];
        oct_decode(Vec2::new(f16_to_f32(b[0]), f16_to_f32(b[1])))
    };
    let decoded_albedo = |x: u32, y: u32| {
        let b = albedo_bits[px(x, y)];
        Vec3::new(f16_to_f32(b[0]), f16_to_f32(b[1]), f16_to_f32(b[2]))
    };
    let base_rgb = |x: u32, y: u32| {
        let b = hdr_bits[px(x, y)];
        Vec3::new(f16_to_f32(b[0]), f16_to_f32(b[1]), f16_to_f32(b[2]))
    };
    let field_at = |x: u32, y: u32| {
        light_field_gate(
            light_field_sample(
                Some(&FIELD_CELLS[..]),
                FIELD_DIMS,
                FIELD_CELL,
                world_at(x, y),
            ),
            FIELD_GATE,
        )
    };
    let delta = |x: u32, y: u32, visibility: f32| -> Vec3 {
        let i = px(x, y);
        let normal = decoded_normal(x, y);
        let world = world_at(x, y);
        let albedo = decoded_albedo(x, y);
        if markers[i] == 2.0 {
            let mut ramped = Vec3::ZERO;
            for light in &live_lights {
                let (_, scale) = point_light_ramp_terms(normal, world, light, WRAP_W, visibility);
                ramped += ramp_rgb * scale;
            }
            albedo * ramped
        } else {
            Vec3::from_array(mesh_point_lights_identity(
                normal,
                world,
                albedo.to_array(),
                WRAP_W,
                &live_lights,
                visibility,
            ))
        }
    };

    for &(x, y) in &[P_IDENTITY, P_RAMP, P_GATED] {
        assert!(
            delta(x, y, 1.0).max_element() > 0.05,
            "fixture too dim at ({x},{y}): {:?}",
            delta(x, y, 1.0)
        );
    }
    let gate_id = field_at(P_IDENTITY.0, P_IDENTITY.1); // out of field → sample 0
    let gate_in = field_at(P_GATED.0, P_GATED.1);
    assert!(
        gate_id < 0.9 && gate_in < 0.9,
        "gates too neutral to observe"
    );
    assert!(
        (delta(P_GATED.0, P_GATED.1, 1.0) - delta(P_GATED.0, P_GATED.1, gate_in))
            .abs()
            .max_element()
            > 0.02,
        "field gating indistinguishable at the gated pixel"
    );

    let mut heap = gpu.heap_slots_create(8, 2, 2);
    let sampler = heap.add_sampler(&gpu, gpu.sampler_descriptor(SamplerDesc::default()));

    let mut surfaces = None;
    assert!(MeshSurfaceTargets::ensure(
        &mut surfaces,
        &gpu,
        &mut heap,
        size
    ));
    let surfaces_ref = surfaces.as_ref().unwrap();
    let targets = surfaces_ref.forward_targets();

    let depth_tex = gpu.texture_alloc_and_create(
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
        gpu.texture_view_descriptor(depth_tex.texture, TextureViewDesc::default()),
    );

    let hdr_tex = gpu.texture_alloc_and_create(
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
        gpu.texture_rw_view_descriptor(hdr_tex.texture, TextureViewDesc::default()),
    );

    let ramp_tex = gpu.texture_alloc_and_create(
        TextureDesc {
            dimensions: [2, 1, 1],
            format: TextureFormat::Rgba16Float,
            usage: UsageFlags::SAMPLED | UsageFlags::TRANSFER_DST,
            ..Default::default()
        },
        Queue::Main,
        None,
    );
    let ramp_slot = heap.add_sampled(
        &gpu,
        gpu.texture_view_descriptor(ramp_tex.texture, TextureViewDesc::default()),
    );

    let mut scene = MeshScene::new(
        &gpu,
        &MeshSceneDesc {
            max_meshes: 4,
            max_instances: 4,
            max_materials: 4,
            vertex_capacity: 512,
            joint_weight_capacity: 0,
            index_capacity: 4096,
            max_meshlets: 64,
        },
    );
    scene.add_material(&gpu, MaterialEntry::standard());
    scene.add_material(
        &gpu,
        MaterialEntry {
            ramp_map: ramp_slot.index(),
            ..MaterialEntry::standard()
        },
    );
    assert_eq!(scene.material_count(), 2);

    let normal_up = gpu.alloc_slice::<[u16; 2]>(pixels as u64, Memory::Default);
    let albedo_up = gpu.alloc_slice::<[u16; 4]>(pixels as u64, Memory::Default);
    let marker_up = gpu.alloc_slice::<f32>(pixels as u64, Memory::Default);
    let depth_up = gpu.alloc_slice::<f32>(pixels as u64, Memory::Default);
    let hdr_up = gpu.alloc_slice::<[u16; 4]>(pixels as u64, Memory::Default);
    let ramp_up = gpu.alloc_slice::<[u16; 4]>(2, Memory::Default);
    // SAFETY: fresh host-visible allocations, each sized above.
    unsafe {
        for i in 0..pixels {
            *normal_up.cpu.add(i) = normal_bits[i];
            *albedo_up.cpu.add(i) = albedo_bits[i];
            *marker_up.cpu.add(i) = markers[i];
            *depth_up.cpu.add(i) = depths[i];
            *hdr_up.cpu.add(i) = hdr_bits[i];
        }
        let ramp_texel = [
            f32_to_f16(ramp_rgb.x),
            f32_to_f16(ramp_rgb.y),
            f32_to_f16(ramp_rgb.z),
            f32_to_f16(1.0),
        ];
        *ramp_up.cpu.add(0) = ramp_texel;
        *ramp_up.cpu.add(1) = ramp_texel;
    }

    let lights_buf = gpu.alloc_slice::<PointLight>(all_lights.len() as u64, Memory::Default);
    // SAFETY: fresh host-visible allocation sized for all_lights.
    unsafe {
        for (i, light) in all_lights.iter().enumerate() {
            *lights_buf.cpu.add(i) = *light;
        }
    }
    let cells_buf = gpu.alloc_slice::<f32>(FIELD_CELLS.len() as u64, Memory::Default);
    // SAFETY: fresh host-visible allocation sized for FIELD_CELLS.
    unsafe {
        for (i, cell) in FIELD_CELLS.iter().enumerate() {
            *cells_buf.cpu.add(i) = *cell;
        }
    }

    let read_a = gpu.alloc_slice::<[u16; 4]>(pixels as u64, Memory::Readback);
    let read_b = gpu.alloc_slice::<[u16; 4]>(pixels as u64, Memory::Readback);
    let read_c = gpu.alloc_slice::<[u16; 4]>(pixels as u64, Memory::Readback);

    let pass = LocalLightPass::new(&gpu);
    let mut fa = TestAlloc {
        gpu: &gpu,
        live: Vec::new(),
    };
    let view = MeshRasterView {
        world_to_clip: Mat4::IDENTITY,
    };

    let cb = gpu.commands_begin(Queue::Main);
    gpu.cmd_copy_to_texture(cb, targets.normal, normal_up.cast());
    gpu.cmd_copy_to_texture(cb, targets.albedo, albedo_up.cast());
    gpu.cmd_copy_to_texture(cb, targets.material, marker_up.cast());
    gpu.cmd_copy_to_texture(cb, depth_tex.texture, depth_up.cast());
    gpu.cmd_copy_to_texture(cb, hdr_tex.texture, hdr_up.cast());
    gpu.cmd_copy_to_texture(cb, ramp_tex.texture, ramp_up.cast());
    gpu.cmd_barrier(cb, Stage::Transfer, Stage::Transfer, HazardFlags::empty());

    pass.record(
        &gpu,
        cb,
        &mut fa,
        &heap,
        &scene,
        surfaces_ref,
        view,
        lights_buf.gpu,
        0,
        WRAP_W,
        MeshLightField::default(),
        None,
        None,
        sampler,
        depth_slot,
        hdr_rw,
    );
    gpu.cmd_copy_texture_to_buffer(cb, hdr_tex.texture, read_a.cast());
    gpu.cmd_barrier(cb, Stage::Transfer, Stage::Compute, HazardFlags::empty());

    pass.record(
        &gpu,
        cb,
        &mut fa,
        &heap,
        &scene,
        surfaces_ref,
        view,
        lights_buf.gpu,
        all_lights.len() as u32,
        WRAP_W,
        MeshLightField::default(),
        None,
        None,
        sampler,
        depth_slot,
        hdr_rw,
    );
    gpu.cmd_barrier(cb, Stage::Compute, Stage::Transfer, HazardFlags::empty());
    gpu.cmd_copy_texture_to_buffer(cb, hdr_tex.texture, read_b.cast());
    gpu.cmd_barrier(cb, Stage::Transfer, Stage::Compute, HazardFlags::empty());

    pass.record(
        &gpu,
        cb,
        &mut fa,
        &heap,
        &scene,
        surfaces_ref,
        view,
        lights_buf.gpu,
        all_lights.len() as u32,
        WRAP_W,
        MeshLightField {
            cells: cells_buf.gpu,
            dims: FIELD_DIMS,
            cell_size: FIELD_CELL,
            gate: FIELD_GATE,
        },
        None,
        None,
        sampler,
        depth_slot,
        hdr_rw,
    );
    gpu.cmd_barrier(cb, Stage::Compute, Stage::Transfer, HazardFlags::empty());
    gpu.cmd_copy_texture_to_buffer(cb, hdr_tex.texture, read_c.cast());
    gpu.queue_submit(Queue::Main, &[cb]);
    gpu.queue_wait_idle(Queue::Main);

    // SAFETY: readbacks hold exactly W × H RGBA16F texels after wait_idle.
    let texel = |ptr: gpu::Ptr<[u16; 4]>, x: u32, y: u32| unsafe { *ptr.cpu.add(px(x, y)) };
    let mut worst = 0.0f32;
    for y in 0..H {
        for x in 0..W {
            let i = px(x, y);
            let bg = hdr_bits[i];
            let got_a = texel(read_a, x, y);
            let got_b = texel(read_b, x, y);
            let got_c = texel(read_c, x, y);

            assert_eq!(got_a, bg, "zero-light dispatch touched HDR at ({x},{y})");

            let covered = markers[i] > 0.0 && depths[i] > 0.0;
            if !covered {
                assert_eq!(got_b, bg, "uncovered pixel written at ({x},{y})");
                assert_eq!(
                    got_c, bg,
                    "uncovered pixel written by gated pass at ({x},{y})"
                );
                continue;
            }

            assert_eq!(got_b[3], bg[3], "alpha destroyed at ({x},{y})");
            assert_eq!(
                got_c[3], bg[3],
                "alpha destroyed by gated pass at ({x},{y})"
            );

            let d1 = delta(x, y, 1.0);
            let d2 = delta(x, y, field_at(x, y));
            for (got, want, tol_scale, label) in [
                (got_b, base_rgb(x, y) + d1, 1.0f32, "ungated"),
                (got_c, base_rgb(x, y) + d1 + d2, 2.0f32, "gated"),
            ] {
                let got_rgb = Vec3::new(f16_to_f32(got[0]), f16_to_f32(got[1]), f16_to_f32(got[2]));
                let err = (got_rgb - want).abs().max_element();
                worst = worst.max(err);
                let tol = tol_scale * (4.0e-3 + 4.0e-3 * want.abs().max_element());
                assert!(
                    err < tol,
                    "{label} pixel ({x},{y}): gpu {got_rgb} vs cpu {want} (err {err:.2e})"
                );
            }
        }
    }
    println!("local light twin: worst channel error {worst:.2e}");

    fa.free();
    for buf in [read_a, read_b, read_c, albedo_up, hdr_up, ramp_up] {
        gpu.free(buf);
    }
    gpu.free(normal_up);
    gpu.free(marker_up);
    gpu.free(depth_up);
    gpu.free(lights_buf);
    gpu.free(cells_buf);
    gpu.texture_free_and_destroy(depth_tex);
    gpu.texture_free_and_destroy(hdr_tex);
    gpu.texture_free_and_destroy(ramp_tex);
    surfaces.take().unwrap().free(&gpu);
    pass.free(&gpu);
    scene.free(&gpu);
    heap.free(&gpu);
}
