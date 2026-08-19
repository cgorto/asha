//! Hardware verification of forward HDR and deferred-light surface MRTs.
//! Tests cover octahedral normals, tinting, material identity, and clears.

mod common;

use abi_core::GpuPtr;
use abi_core::glam::{Mat4, UVec2, Vec2, Vec3, Vec4};
use abi_core::oct_decode;
use abi_light::mesh_shade_slim;
use abi_mesh::{MESH_FLAG_HIDDEN, mesh_world_to_clip};
use common::{TestFrameAlloc, gpu_test_lock, mesh_heap, view};
use gpu::{Gpu, HazardFlags, LoadOp, Memory, Queue, Stage, TextureDesc, TextureFormat, UsageFlags};
use mesh::cull::ClusterCullPass;
use mesh::primitives::cube;
use mesh::{
    MaterialEntry, MeshDepthPrepass, MeshForwardPass, MeshForwardSurfaceTargets,
    MeshForwardTargets, MeshInstance, MeshInstances, MeshLightField, MeshRasterView, MeshScene,
    MeshSceneDesc, MeshShadeLighting,
};

const W: u32 = 65;
const H: u32 = 65;
/// HDR clear and frame-2 forward-pass witness.
const CLEAR: [f32; 4] = [0.02, 0.03, 0.04, 1.0];
/// Base color for material row 1.
const BASE: [f32; 3] = [0.8, 0.4, 0.2];
/// Instance tint for the rotated cube.
const TINT: [f32; 4] = [0.5, 0.25, 1.0, 1.0];
/// World x of the zero-normal cube.
const FLAT_X: f32 = 2.9;

fn f16_to_f32(h: u16) -> f32 {
    let (sign, exp, mant) = (
        (h >> 15) as u32,
        ((h >> 10) & 0x1F) as u32,
        (h & 0x3FF) as u32,
    );
    let f = match (exp, mant) {
        (0, 0) => 0.0,
        (0, m) => m as f32 * 2f32.powi(-24),
        (0x1F, 0) => f32::INFINITY,
        (0x1F, _) => f32::NAN,
        (e, m) => (1.0 + m as f32 / 1024.0) * 2f32.powi(e as i32 - 15),
    };
    if sign == 1 { -f } else { f }
}

#[test]
fn forward_surfaces_land_and_clear() {
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
    // Allocate format-exact deferred-light attachments.
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
            max_meshes: 2,
            max_instances: 2,
            max_materials: 2,
            vertex_capacity: 64,
            joint_weight_capacity: 0,
            index_capacity: 512,
            max_meshlets: 16,
        },
    );
    let mesh_rotated = scene.add_mesh(&gpu, cube(1.0).desc());
    // Zero normals exercise the fragment fallback instead of NaN encoding.
    let mut flat = cube(1.0);
    for n in &mut flat.normals {
        *n = [0.0; 3];
    }
    let mesh_flat = scene.add_mesh(&gpu, flat.desc());

    // A decoy row exposes material-indexing errors.
    let mut decoy = MaterialEntry::standard();
    decoy.base_color_factor = [0.9, 0.0, 0.9, 1.0];
    scene.add_material(&gpu, decoy);
    let mut entry = MaterialEntry::standard();
    entry.base_color_factor = [BASE[0], BASE[1], BASE[2], 1.0];
    let material = scene.add_material(&gpu, entry);

    // Rotation produces a non-axis-aligned world normal.
    let rotation = Mat4::from_rotation_y(30f32.to_radians());
    scene.add_instance(&gpu, mesh_rotated, rotation, material);
    scene.add_instance(
        &gpu,
        mesh_flat,
        Mat4::from_translation(Vec3::new(FLAT_X, 0.0, 0.0)),
        material,
    );
    let expected_n = (rotation * Vec4::new(0.0, 0.0, -1.0, 0.0))
        .truncate()
        .normalize();

    // Host-visible streams carry tint and hidden flags for this test.
    let instances_buf = gpu.alloc_slice::<MeshInstance>(2, Memory::Default);
    let scene_view = scene.instances();
    let make_instances = |flags: u32| -> Vec<MeshInstance> {
        let instance = |batch_index: u32, instance_color: [f32; 4]| MeshInstance {
            batch_index,
            transform_index: batch_index,
            flags,
            outline_group: 0,
            instance_color,
            joint_transforms: GpuPtr::null(),
            deformer_slot: 0,
            bounds_dilation: 0.0,
        };
        vec![instance(0, TINT), instance(1, [1.0; 4])]
    };

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
    let pass = MeshForwardPass::new(&gpu);
    let cull = ClusterCullPass::new(&gpu, &scene, 1);
    let (heap, ramp_default_sampler) = mesh_heap(&gpu);
    let mut frame_alloc = TestFrameAlloc {
        gpu: &gpu,
        ptrs: Vec::new(),
    };

    let hdr_rb = gpu.alloc_slice::<[f32; 4]>((W * H) as u64, Memory::Readback);
    let normal_rb = gpu.alloc_slice::<[u16; 2]>((W * H) as u64, Memory::Readback);
    let albedo_rb = gpu.alloc_slice::<[u16; 4]>((W * H) as u64, Memory::Readback);
    let material_rb = gpu.alloc_slice::<f32>((W * H) as u64, Memory::Readback);

    // Upload, cull, prepass, forward, then read all four targets.
    let frame = |frame_alloc: &mut TestFrameAlloc, flags: u32| {
        let instances_cpu = make_instances(flags);
        // SAFETY: 2-element host-visible allocation; the GPU is idle
        // between frames (each frame ends in queue_wait_idle).
        unsafe {
            for (i, instance) in instances_cpu.iter().enumerate() {
                *instances_buf.cpu.add(i) = *instance;
            }
        }
        let instances = MeshInstances {
            instances: instances_buf.gpu,
            batches: scene_view.batches,
            transforms: scene_view.transforms,
            deformers: GpuPtr::null(),
            instances_cpu: &instances_cpu,
            batches_cpu: scene_view.batches_cpu,
            transforms_cpu: scene_view.transforms_cpu,
        };

        let cb = gpu.commands_begin(Queue::Main);
        cull.record(
            &gpu,
            cb,
            frame_alloc,
            &scene,
            instances,
            raster_view,
            Vec3::from_array(view.camera_position),
            0,
        );
        prepass.record(
            &gpu,
            cb,
            frame_alloc,
            &scene,
            instances,
            cull.output(),
            cull.clusters(),
            cull.draw_count_ptr(&gpu, 0),
            depth.texture,
            size,
            raster_view,
        );
        pass.record_with_surfaces(
            &gpu,
            cb,
            frame_alloc,
            &heap,
            &scene,
            instances,
            cull.output(),
            cull.clusters(),
            cull.draw_count_ptr(&gpu, 0),
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
        gpu.cmd_copy_texture_to_buffer(cb, normal.texture, normal_rb.cast());
        gpu.cmd_copy_texture_to_buffer(cb, albedo.texture, albedo_rb.cast());
        gpu.cmd_copy_texture_to_buffer(cb, material_tex.texture, material_rb.cast());
        gpu.queue_submit(Queue::Main, &[cb]);
        gpu.queue_wait_idle(Queue::Main);
    };

    let index = |x: u32, y: u32| (y * W + x) as usize;
    // SAFETY (all four): each readback covers W*H texels and the frame
    // closure waited the queue idle before any read.
    let hdr_at = |x: u32, y: u32| -> [f32; 4] { unsafe { *hdr_rb.cpu.add(index(x, y)) } };
    let normal_bits_at =
        |x: u32, y: u32| -> [u16; 2] { unsafe { *normal_rb.cpu.add(index(x, y)) } };
    let albedo_at = |x: u32, y: u32| -> [f32; 4] {
        let px = unsafe { *albedo_rb.cpu.add(index(x, y)) };
        [
            f16_to_f32(px[0]),
            f16_to_f32(px[1]),
            f16_to_f32(px[2]),
            f16_to_f32(px[3]),
        ]
    };
    let material_at = |x: u32, y: u32| -> f32 { unsafe { *material_rb.cpu.add(index(x, y)) } };
    let decoded_normal_at = |x: u32, y: u32| -> Vec3 {
        let bits = normal_bits_at(x, y);
        oct_decode(Vec2::new(f16_to_f32(bits[0]), f16_to_f32(bits[1])))
    };
    let assert_near = |label: &str, got: &[f32], want: &[f32], tolerance: f32| {
        for (c, (g, w)) in got.iter().zip(want).enumerate() {
            assert!(
                (g - w).abs() < tolerance,
                "{label} channel {c}: gpu {g} vs cpu {w}"
            );
        }
    };

    // Probe the zero-normal cube through the raster transform.
    let clip = raster_view.world_to_clip * Vec4::new(FLAT_X, 0.0, -1.0, 1.0);
    let flat_px = ((clip.x / clip.w + 1.0) * 0.5 * W as f32).floor() as u32;
    assert!(flat_px < W, "flat-cube probe off screen: {flat_px}");
    assert!(
        (flat_px as i32 - (W / 2) as i32).unsigned_abs() > 12,
        "flat-cube probe {flat_px} overlaps the rotated cube"
    );

    // Frame 1: both cubes visible.
    frame(&mut frame_alloc, 0);

    let (cx, cy) = (W / 2, H / 2);
    // HDR equals shaded base color multiplied by instance tint.
    let shaded = mesh_shade_slim(expected_n, BASE, &lighting);
    let expected_hdr = [
        shaded[0] * TINT[0],
        shaded[1] * TINT[1],
        shaded[2] * TINT[2],
        TINT[3],
    ];
    assert_near("hdr", &hdr_at(cx, cy), &expected_hdr, 2.0e-3);

    // Octahedral normal decodes within RG16F tolerance.
    assert_near(
        "oct normal",
        &decoded_normal_at(cx, cy).to_array(),
        &expected_n.to_array(),
        5.0e-3,
    );
    // Albedo equals base × tint with alpha one.
    let expected_albedo = [BASE[0] * TINT[0], BASE[1] * TINT[1], BASE[2] * TINT[2], 1.0];
    assert_near("albedo", &albedo_at(cx, cy), &expected_albedo, 2.0e-3);
    // Material row 1 encodes as exact identity 2.0.
    assert_eq!(material_at(cx, cy), 2.0, "material + 1 must be exact");

    // The zero-normal cube: only the guarded shading normal (Y fallback)
    // decodes here; the raw interpolant is NaN and would never decode to Y.
    assert_near(
        "guarded normal",
        &decoded_normal_at(flat_px, cy).to_array(),
        &Vec3::Y.to_array(),
        5.0e-3,
    );
    assert_near(
        "untinted albedo",
        &albedo_at(flat_px, cy),
        &[BASE[0], BASE[1], BASE[2], 1.0],
        2.0e-3,
    );
    assert_eq!(material_at(flat_px, cy), 2.0);

    // Background: all three surfaces stay bit-exact zero (0 = no mesh) and
    // HDR holds the forward pass's own clear.
    assert_eq!(normal_bits_at(2, 2), [0, 0]);
    assert_eq!(albedo_at(2, 2), [0.0; 4]);
    assert_eq!(material_at(2, 2), 0.0);
    assert_near("hdr background", &hdr_at(2, 2), &CLEAR, 1.0e-6);

    // Frame 2 hides all instances; clears must remove prior surfaces.
    frame(&mut frame_alloc, MESH_FLAG_HIDDEN);
    for &(x, label) in &[(cx, "rotated cube"), (flat_px, "flat cube")] {
        assert_eq!(
            material_at(x, cy),
            0.0,
            "stale material identity survived the MRT clear at the {label} pixel"
        );
        assert_eq!(normal_bits_at(x, cy), [0, 0], "stale normal at {label}");
        assert_eq!(albedo_at(x, cy), [0.0; 4], "stale albedo at {label}");
        assert_near("hidden-frame hdr", &hdr_at(x, cy), &CLEAR, 1.0e-6);
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
    gpu.free(normal_rb);
    gpu.free(albedo_rb);
    gpu.free(material_rb);
    gpu.free(instances_buf);
    heap.free(&gpu);
}
