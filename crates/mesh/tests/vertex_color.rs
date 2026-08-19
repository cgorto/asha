//! Hardware verification of optional per-vertex color multiplication.

mod common;

use abi_core::glam::{Mat4, UVec2, Vec3};
use abi_light::mesh_shade_slim;
use abi_mesh::mesh_world_to_clip;
use common::{TestFrameAlloc, gpu_test_lock, mesh_heap, view};
use gpu::{
    Gpu, HazardFlags, LoadOp, Memory, Queue, RenderAttachment, RenderPassDesc, Stage, StoreOp,
    TextureDesc, TextureFormat, UsageFlags,
};
use mesh::cull::ClusterCullPass;
use mesh::primitives::cube;
use mesh::{
    MaterialEntry, MeshDepthPrepass, MeshForwardPass, MeshForwardTargets, MeshRasterView,
    MeshScene, MeshSceneDesc, MeshShadeLighting,
};

const W: u32 = 65;
const H: u32 = 65;
const CLEAR: [f32; 4] = [0.02, 0.03, 0.04, 1.0];
const ALBEDO: [f32; 3] = [0.8, 0.4, 0.2];

fn scene_desc() -> MeshSceneDesc {
    MeshSceneDesc {
        max_meshes: 1,
        max_instances: 1,
        max_materials: 1,
        vertex_capacity: 64,
        joint_weight_capacity: 0,
        index_capacity: 256,
        max_meshlets: 8,
    }
}

fn lighting() -> MeshShadeLighting {
    MeshShadeLighting {
        sun_direction: Vec3::NEG_Z.to_array(),
        sun_tint: [1.0, 0.75, 0.5],
        sky_ambient: [0.2, 0.25, 0.3],
        ground_ambient: [0.05, 0.04, 0.03],
        ..MeshShadeLighting::zeroed()
    }
}

/// Authored gradient: red varies with x; green and blue remain one.
fn gradient_color(position: [f32; 3]) -> [f32; 4] {
    [(position[0] + 1.0) * 0.5, 1.0, 1.0, 1.0]
}

/// Maps a front-face pixel to world x under the shared test view.
fn front_face_world_x(px: u32) -> f32 {
    let v = view(UVec2::new(W, H));
    let ndc_x = (px as f32 + 0.5) / W as f32 * 2.0 - 1.0;
    -ndc_x * v.tan_half_fov * v.aspect * 7.0
}

/// Renders one standard frame and returns CPU pixels.
fn render(colors: Option<&[[f32; 4]]>) -> Vec<[f32; 4]> {
    let _gpu_guard = gpu_test_lock();
    let gpu = Gpu::new(true).expect("vulkan init");
    let size = UVec2::new(W, H);

    let color = gpu.texture_alloc_and_create(
        TextureDesc {
            dimensions: [W, H, 1],
            format: TextureFormat::Rgba32Float,
            usage: UsageFlags::COLOR_ATTACHMENT | UsageFlags::TRANSFER_SRC,
            ..Default::default()
        },
        Queue::Main,
        None,
    );
    let depth = gpu.texture_alloc_and_create(
        TextureDesc {
            dimensions: [W, H, 1],
            format: TextureFormat::D32Float,
            usage: UsageFlags::DEPTH_STENCIL_ATTACHMENT | UsageFlags::TRANSFER_SRC,
            ..Default::default()
        },
        Queue::Main,
        None,
    );

    let mut scene = MeshScene::new(&gpu, &scene_desc());
    let buffers = cube(1.0);
    let mesh = scene.add_mesh(
        &gpu,
        mesh::MeshDesc {
            colors,
            ..buffers.desc()
        },
    );
    let mut material = MaterialEntry::standard();
    material.base_color_factor = [ALBEDO[0], ALBEDO[1], ALBEDO[2], 1.0];
    let material = scene.add_material(&gpu, material);
    scene.add_instance(&gpu, mesh, Mat4::IDENTITY, material);

    let v = view(size);
    let raster_view = MeshRasterView {
        world_to_clip: mesh_world_to_clip(&v),
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
    let color_rb = gpu.alloc_slice::<[f32; 4]>((W * H) as u64, Memory::Readback);

    let cb = gpu.commands_begin(Queue::Main);
    gpu.cmd_begin_render_pass(
        cb,
        RenderPassDesc {
            color_attachments: &[RenderAttachment {
                texture: color.texture,
                load_op: LoadOp::Clear,
                store_op: StoreOp::Store,
                clear_color: CLEAR,
                ..Default::default()
            }],
            ..Default::default()
        },
    );
    gpu.cmd_end_render_pass(cb);
    gpu.cmd_barrier(
        cb,
        Stage::RasterColorOut,
        Stage::RasterColorOut,
        HazardFlags::empty(),
    );
    cull.record(
        &gpu,
        cb,
        &mut frame_alloc,
        &scene,
        scene.instances(),
        raster_view,
        Vec3::from_array(v.camera_position),
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
    pass.record(
        &gpu,
        cb,
        &mut frame_alloc,
        &heap,
        &scene,
        scene.instances(),
        cull.output(),
        cull.clusters(),
        cull.draw_count_ptr(&gpu, 0),
        MeshForwardTargets {
            color: color.texture,
            depth: depth.texture,
            size,
            color_load_op: LoadOp::Load,
            clear_color: [0.0; 4],
        },
        raster_view,
        Vec3::from_array(v.camera_position),
        lighting(),
        ramp_default_sampler,
    );
    gpu.cmd_barrier(
        cb,
        Stage::RasterColorOut,
        Stage::Transfer,
        HazardFlags::empty(),
    );
    gpu.cmd_copy_texture_to_buffer(cb, color.texture, color_rb.cast());
    gpu.queue_submit(Queue::Main, &[cb]);
    gpu.queue_wait_idle(Queue::Main);

    // SAFETY: the readback buffer covers W*H pixels and the queue is idle.
    let pixels = unsafe { std::slice::from_raw_parts(color_rb.cpu, (W * H) as usize) }.to_vec();

    frame_alloc.free();
    prepass.free(&gpu);
    pass.free(&gpu);
    cull.free(&gpu);
    scene.free(&gpu);
    gpu.texture_free_and_destroy(color);
    gpu.texture_free_and_destroy(depth);
    gpu.free(color_rb);
    heap.free(&gpu);
    pixels
}

/// Null colors preserve shading; supplied colors multiply each channel.
#[test]
fn vertex_colors_modulate_and_null_is_unchanged() {
    let buffers = cube(1.0);
    let colors = buffers
        .positions
        .iter()
        .map(|&p| gradient_color(p))
        .collect::<Vec<_>>();

    let untinted = render(None);
    let tinted = render(Some(&colors));

    let at = |pixels: &[[f32; 4]], x: u32, y: u32| pixels[(y * W + x) as usize];
    let shaded = mesh_shade_slim(Vec3::NEG_Z, ALBEDO, &lighting());

    // Null colors preserve shading and the outside clear.
    let center_untinted = at(&untinted, W / 2, H / 2);
    for c in 0..3 {
        assert!(
            (center_untinted[c] - shaded[c]).abs() < 2.0e-3,
            "null-colour channel {c}: gpu {} vs cpu {}",
            center_untinted[c],
            shaded[c]
        );
    }
    assert_eq!(at(&untinted, 2, 2), CLEAR, "background must be the clear");
    assert_eq!(
        at(&tinted, 2, 2),
        CLEAR,
        "vertex colour must not leak outside the mesh"
    );

    // Samples prove interpolation across vertices, not per-instance flatness.
    for x in [27, W / 2, 38] {
        let y = H / 2;
        let expected_ramp = (front_face_world_x(x) + 1.0) * 0.5;
        let got = at(&tinted, x, y);
        let base = at(&untinted, x, y);
        assert!(
            (got[0] - base[0] * expected_ramp).abs() < 3.0e-3,
            "x={x}: red {} != base {} * ramp {expected_ramp}",
            got[0],
            base[0]
        );
        // Green and blue remain bit-identical.
        assert_eq!(
            [got[1], got[2], got[3]],
            [base[1], base[2], base[3]],
            "x={x}: a 1.0 tint must not perturb a channel"
        );
    }

    // The image ramp is monotonic; view right points toward -X.
    let left = at(&tinted, 27, H / 2)[0];
    let mid = at(&tinted, W / 2, H / 2)[0];
    let right = at(&tinted, 38, H / 2)[0];
    assert!(
        left > mid && mid > right,
        "red must ramp monotonically: {left} > {mid} > {right}"
    );
}
