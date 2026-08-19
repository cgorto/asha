//! Hardware verification of grouped forward and additive coat draws.

mod common;

use abi_core::glam::{Mat4, UVec2, Vec3, Vec4};
use abi_light::mesh_shade_slim;
use abi_mesh::mesh_world_to_clip;
use common::{TestFrameAlloc, gpu_test_lock, mesh_heap, view};
use gpu::{Gpu, HazardFlags, LoadOp, Memory, Queue, Stage, TextureDesc, TextureFormat, UsageFlags};
use mesh::cull::ClusterCullPass;
use mesh::primitives::cube;
use mesh::{
    MaterialEntry, MeshDepthPrepass, MeshForwardPass, MeshForwardSurfaceTargets,
    MeshForwardTargets, MeshLightField, MeshRasterView, MeshScene, MeshSceneDesc,
    MeshShadeLighting, ShaderCoatSlice, ShaderGroupKind, ShaderGroupSlice,
};

const W: u32 = 65;
const H: u32 = 65;
const CLEAR: [f32; 4] = [0.02, 0.03, 0.04, 1.0];
/// Standard-forward cube's base color (material row 0).
const LIT: [f32; 3] = [0.8, 0.4, 0.2];
/// Group cube's base color (row 1) — flat shading must land it VERBATIM.
const FLAT: [f32; 3] = [0.1, 0.9, 0.3];
/// Pulse cube's base color (row 2).
const PULSE: [f32; 3] = [0.3, 0.3, 0.35];
/// World offsets separating the three cube footprints.
const OFFSET_X: f32 = 2.9;
/// Nonzero frame time used to verify shader time propagation.
const TIME: f32 = 1.7;

/// CPU twin of the pulse shader for neutral instance parameters.
fn pulse_want(lighting: &MeshShadeLighting) -> [f32; 3] {
    let base = mesh_shade_slim(Vec3::NEG_Z, PULSE, lighting);
    let wave = (TIME * 4.0 + 1.0).sin() * 0.5 + 0.5;
    let t = ((wave - 0.15) / 0.7).clamp(0.0, 1.0);
    let throb = t * t * (3.0 - 2.0 * t);
    let ember = [1.0, 0.32, 0.08];
    let glow = 0.25 + 0.75 * throb;
    [
        base[0] + ember[0] * glow,
        base[1] + ember[1] * glow,
        base[2] + ember[2] * glow,
    ]
}

#[test]
fn grouped_forward_splits_shaders_by_batch() {
    let _gpu_guard = gpu_test_lock();
    let gpu = Gpu::new(true).expect("vulkan init");
    let size = UVec2::new(W, H);

    let attachment = |format| {
        gpu.texture_alloc_and_create(
            TextureDesc {
                dimensions: [W, H, 1],
                format,
                usage: UsageFlags::COLOR_ATTACHMENT | UsageFlags::TRANSFER_SRC,
                ..Default::default()
            },
            Queue::Main,
            None,
        )
    };
    let hdr = attachment(TextureFormat::Rgba32Float);
    let normal = attachment(TextureFormat::Rg16Float);
    let albedo = attachment(TextureFormat::Rgba16Float);
    let material_tex = attachment(TextureFormat::R32Float);
    let depth = gpu.texture_alloc_and_create(
        TextureDesc {
            dimensions: [W, H, 1],
            format: TextureFormat::D32Float,
            usage: UsageFlags::DEPTH_STENCIL_ATTACHMENT,
            ..Default::default()
        },
        Queue::Main,
        None,
    );

    let mut scene = MeshScene::new(
        &gpu,
        &MeshSceneDesc {
            max_meshes: 1,
            max_instances: 3,
            max_materials: 3,
            vertex_capacity: 64,
            joint_weight_capacity: 0,
            index_capacity: 512,
            max_meshlets: 16,
        },
    );
    let mesh = scene.add_mesh(&gpu, cube(1.0).desc());
    // Distinct materials create distinct batches.
    let mut lit = MaterialEntry::standard();
    lit.base_color_factor = [LIT[0], LIT[1], LIT[2], 1.0];
    let lit = scene.add_material(&gpu, lit);
    let mut flat = MaterialEntry::standard();
    flat.base_color_factor = [FLAT[0], FLAT[1], FLAT[2], 1.0];
    let flat = scene.add_material(&gpu, flat);
    let mut pulse = MaterialEntry::standard();
    pulse.base_color_factor = [PULSE[0], PULSE[1], PULSE[2], 1.0];
    let pulse = scene.add_material(&gpu, pulse);
    // Add order establishes standard and grouped batch ranges.
    scene.add_instance(
        &gpu,
        mesh,
        Mat4::from_translation(Vec3::new(-OFFSET_X, 0.0, 0.0)),
        lit,
    );
    scene.add_instance(
        &gpu,
        mesh,
        Mat4::from_translation(Vec3::new(OFFSET_X, 0.0, 0.0)),
        flat,
    );
    scene.add_instance(&gpu, mesh, Mat4::from_translation(Vec3::ZERO), pulse);

    let lighting = MeshShadeLighting {
        sun_direction: Vec3::NEG_Z.to_array(),
        sun_tint: [1.0, 0.75, 0.5],
        sky_ambient: [0.2, 0.25, 0.3],
        ground_ambient: [0.05, 0.04, 0.03],
        ..MeshShadeLighting::zeroed()
    };
    let view = view(size);
    let raster_view = MeshRasterView {
        world_to_clip: mesh_world_to_clip(&view),
    };
    let mut prepass = MeshDepthPrepass::new(&gpu);
    prepass.resize(&gpu, size);
    let mut pass = MeshForwardPass::with_groups(&gpu, 3, 1);
    pass.register_group(&gpu, 0, None, "mesh_flat_frag", ShaderGroupKind::Replace);
    pass.register_group(&gpu, 1, None, "hazard_pulse_frag", ShaderGroupKind::Replace);
    pass.register_group(&gpu, 2, None, "glow_coat_frag", ShaderGroupKind::Coat);
    let cull = ClusterCullPass::new(&gpu, &scene, 1);
    let (heap, ramp_default_sampler) = mesh_heap(&gpu);
    let mut frame_alloc = TestFrameAlloc {
        gpu: &gpu,
        ptrs: Vec::new(),
    };

    let hdr_rb = gpu.alloc_slice::<[f32; 4]>((W * H) as u64, Memory::Readback);
    let material_rb = gpu.alloc_slice::<f32>((W * H) as u64, Memory::Readback);

    let cb = gpu.commands_begin(Queue::Main);
    cull.record(
        &gpu,
        cb,
        &mut frame_alloc,
        &scene,
        scene.instances(),
        raster_view,
        Vec3::from_array(view.camera_position),
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
    pass.record_grouped(
        &gpu,
        cb,
        &mut frame_alloc,
        &heap,
        &scene,
        scene.instances(),
        cull.output(),
        cull.clusters(),
        1,
        &[
            ShaderGroupSlice {
                group: 0,
                batch_base: 1,
                batch_count: 1,
            },
            ShaderGroupSlice {
                group: 1,
                batch_base: 2,
                batch_count: 1,
            },
        ],
        // The coat adds white under neutral instance parameters.
        &[ShaderCoatSlice {
            group: 2,
            batch_base: 1,
            batch_count: 1,
        }],
        0,
        TIME,
        MeshForwardTargets {
            color: hdr.texture,
            depth: depth.texture,
            size,
            color_load_op: LoadOp::Clear,
            clear_color: CLEAR,
        },
        raster_view,
        Vec3::from_array(view.camera_position),
        lighting,
        ramp_default_sampler,
        MeshLightField::default(),
        MeshForwardSurfaceTargets {
            normal: normal.texture,
            albedo: albedo.texture,
            material: material_tex.texture,
        },
    );
    gpu.cmd_barrier(
        cb,
        Stage::RasterColorOut,
        Stage::Transfer,
        HazardFlags::empty(),
    );
    gpu.cmd_copy_texture_to_buffer(cb, hdr.texture, hdr_rb.cast());
    gpu.cmd_copy_texture_to_buffer(cb, material_tex.texture, material_rb.cast());
    gpu.queue_submit(Queue::Main, &[cb]);
    gpu.queue_wait_idle(Queue::Main);

    let index = |x: u32, y: u32| (y * W + x) as usize;
    // SAFETY (both): readbacks cover W*H texels; the queue idled above.
    let hdr_at = |x: u32, y: u32| -> [f32; 4] { unsafe { *hdr_rb.cpu.add(index(x, y)) } };
    let material_at = |x: u32, y: u32| -> f32 { unsafe { *material_rb.cpu.add(index(x, y)) } };
    let probe_x = |world_x: f32| -> u32 {
        let clip = raster_view.world_to_clip * Vec4::new(world_x, 0.0, -1.0, 1.0);
        let px = ((clip.x / clip.w + 1.0) * 0.5 * W as f32).floor() as u32;
        assert!(px < W, "probe off screen");
        px
    };
    let (lit_px, flat_px, pulse_px, cy) =
        (probe_x(-OFFSET_X), probe_x(OFFSET_X), probe_x(0.0), H / 2);

    // Batch 0: standard forward shading.
    let lit_got = hdr_at(lit_px, cy);
    let lit_want = mesh_shade_slim(Vec3::NEG_Z, LIT, &lighting);
    for c in 0..3 {
        assert!(
            (lit_got[c] - lit_want[c]).abs() < 2.0e-3,
            "standard cube channel {c}: gpu {} vs cpu {}",
            lit_got[c],
            lit_want[c]
        );
    }
    // Its surface material is material row 0 + 1 = 1.0: relit downstream.
    assert_eq!(material_at(lit_px, cy), 1.0);

    // Batch 1: flat base plus white additive coat; surfaces remain masked.
    let flat_got = hdr_at(flat_px, cy);
    for c in 0..3 {
        let want = FLAT[c] + 1.0;
        assert!(
            (flat_got[c] - want).abs() < 2.0e-3,
            "coated group cube channel {c}: gpu {} vs cpu {want} (flat + glow)",
            flat_got[c],
        );
    }
    assert_eq!(material_at(flat_px, cy), 0.0);

    // Batch 2: pulsed shading and material row 2 identity.
    let pulse_got = hdr_at(pulse_px, cy);
    let pulse_want = pulse_want(&lighting);
    for c in 0..3 {
        assert!(
            (pulse_got[c] - pulse_want[c]).abs() < 2.0e-3,
            "pulse cube channel {c}: gpu {} vs cpu {}",
            pulse_got[c],
            pulse_want[c]
        );
    }
    assert_eq!(material_at(pulse_px, cy), 3.0);

    // Background remains untouched.
    let bg = hdr_at(2, 2);
    for (got, want) in bg.into_iter().zip(CLEAR) {
        assert!((got - want).abs() < 1.0e-6, "background {got} != {want}");
    }

    frame_alloc.free();
    prepass.free(&gpu);
    pass.free(&gpu);
    cull.free(&gpu);
    scene.free(&gpu);
    gpu.texture_free_and_destroy(hdr);
    gpu.texture_free_and_destroy(normal);
    gpu.texture_free_and_destroy(albedo);
    gpu.texture_free_and_destroy(material_tex);
    gpu.texture_free_and_destroy(depth);
    gpu.free(hdr_rb);
    gpu.free(material_rb);
    heap.free(&gpu);
}
