//! Offscreen verification of visibility-buffer linework.

mod common;

use abi_core::GpuPtr;
use abi_core::View;
use abi_core::glam::{Mat4, UVec2, Vec3};
use abi_mesh::mesh_world_to_clip;
use common::{TestFrameAlloc, gpu_test_lock};
use gpu::{
    Gpu, HazardFlags, LoadOp, Memory, Queue, RenderAttachment, RenderPassDesc, Stage, StoreOp,
    TextureDesc, TextureFormat, TextureViewDesc, UsageFlags,
};
use mesh::cull::ClusterCullPass;
use mesh::{
    LineworkDials, MaterialEntry, MeshDepthPrepass, MeshLineworkPass, MeshRasterView, MeshScene,
    MeshSceneDesc,
};

const W: u32 = 96;
const H: u32 = 96;

fn front_view() -> View {
    common::view(UVec2::new(W, H))
}

fn linework_dials() -> LineworkDials {
    LineworkDials {
        enabled: true,
        normal_cos_threshold: 15.0f32.to_radians().cos(),
        plane_epsilon: 0.01,
        crease_strength: 1.0,
        step_strength: 1.0,
        fade_near: 0.0,
        fade_far: 1_000.0,
    }
}

fn render(
    view: View,
    setup: impl FnOnce(&Gpu, &mut MeshScene, mesh::MaterialHandle),
) -> Vec<[f32; 4]> {
    let gpu = Gpu::new(true).expect("vulkan init");
    let size = UVec2::new(W, H);
    let depth = gpu.texture_alloc_and_create(
        TextureDesc {
            dimensions: [W, H, 1],
            format: TextureFormat::D32Float,
            usage: UsageFlags::DEPTH_STENCIL_ATTACHMENT | UsageFlags::SAMPLED,
            ..Default::default()
        },
        Queue::Main,
        None,
    );
    let display = gpu.texture_alloc_and_create(
        TextureDesc {
            dimensions: [W, H, 1],
            format: TextureFormat::Rgba32Float,
            usage: UsageFlags::COLOR_ATTACHMENT | UsageFlags::TRANSFER_SRC,
            ..Default::default()
        },
        Queue::Main,
        None,
    );
    let mut scene = MeshScene::new(
        &gpu,
        &MeshSceneDesc {
            max_meshes: 4,
            max_instances: 8,
            max_materials: 1,
            vertex_capacity: 128,
            joint_weight_capacity: 0,
            index_capacity: 384,
            max_meshlets: 16,
        },
    );
    let material = scene.add_material(&gpu, MaterialEntry::standard());
    setup(&gpu, &mut scene, material);

    let mut prepass = MeshDepthPrepass::new(&gpu);
    prepass.resize(&gpu, size);
    let linework = MeshLineworkPass::new(&gpu);
    let cull = ClusterCullPass::new(&gpu, &scene, 1);
    let mut heap = gpu.heap_slots_create(2, 2, 2);
    let depth_slot = heap.add_sampled(
        &gpu,
        gpu.texture_view_descriptor(depth.texture, TextureViewDesc::default()),
    );
    let visibility_slot = heap.add_storage(
        &gpu,
        gpu.texture_rw_view_descriptor(prepass.visibility_texture(), TextureViewDesc::default()),
    );
    let readback = gpu.alloc_slice::<[f32; 4]>((W * H) as u64, Memory::Readback);
    let mut frame_alloc = TestFrameAlloc {
        gpu: &gpu,
        ptrs: Vec::new(),
    };
    let raster_view = MeshRasterView {
        world_to_clip: mesh_world_to_clip(&view),
    };
    let cb = gpu.commands_begin(Queue::Main);
    // White makes black alpha blending observable.
    gpu.cmd_begin_render_pass(
        cb,
        RenderPassDesc {
            color_attachments: &[RenderAttachment {
                texture: display.texture,
                load_op: LoadOp::Clear,
                store_op: StoreOp::Store,
                clear_color: [1.0; 4],
                ..Default::default()
            }],
            ..Default::default()
        },
    );
    gpu.cmd_end_render_pass(cb);
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
    linework.record(
        &gpu,
        cb,
        &mut frame_alloc,
        &heap,
        &scene,
        scene.instances(),
        cull.clusters(),
        raster_view,
        Vec3::from_array(view.camera_position),
        display.texture,
        depth_slot,
        visibility_slot,
        linework_dials(),
    );
    gpu.cmd_barrier(
        cb,
        Stage::RasterColorOut,
        Stage::Transfer,
        HazardFlags::empty(),
    );
    gpu.cmd_copy_texture_to_buffer(cb, display.texture, readback.cast());
    gpu.queue_submit(Queue::Main, &[cb]);
    gpu.queue_wait_idle(Queue::Main);

    // SAFETY: the readback has exactly W × H initialized RGBA32F texels.
    let pixels = unsafe { std::slice::from_raw_parts(readback.cpu, (W * H) as usize).to_vec() };

    frame_alloc.free();
    gpu.free(readback);
    heap.free(&gpu);
    cull.free(&gpu);
    linework.free(&gpu);
    prepass.free(&gpu);
    scene.free(&gpu);
    gpu.texture_free_and_destroy(display);
    gpu.texture_free_and_destroy(depth);
    pixels
}

fn pixel(pixels: &[[f32; 4]], x: u32, y: u32) -> [f32; 4] {
    pixels[(y * W + x) as usize]
}

fn is_white(pixel: [f32; 4]) -> bool {
    pixel[0] > 0.999 && pixel[1] > 0.999 && pixel[2] > 0.999 && pixel[3] > 0.999
}

fn dark_count(pixels: &[[f32; 4]]) -> usize {
    pixels.iter().filter(|&&p| !is_white(p)).count()
}

fn add_double_sided_unit_quad(
    gpu: &Gpu,
    scene: &mut MeshScene,
    material: mesh::MaterialHandle,
    transform: Mat4,
) {
    let positions = [
        [-1.0, -1.0, 0.0],
        [1.0, -1.0, 0.0],
        [1.0, 1.0, 0.0],
        [-1.0, 1.0, 0.0],
    ];
    let normals = [[0.0, 0.0, -1.0]; 4];
    let uvs = [[0.0, 0.0]; 4];
    // Opposite winds make culling irrelevant; Equal depth selects one token.
    let indices = [0, 2, 1, 0, 3, 2, 0, 1, 2, 0, 2, 3];
    let mesh = scene.add_mesh(
        gpu,
        mesh::MeshDesc {
            positions: &positions,
            normals: &normals,
            uvs: &uvs,
            indices: &indices,
            tangents: None,
            joint_weights: None,
            colors: None,
        },
    );
    scene.add_instance(gpu, mesh, transform, material);
}

fn render_uniform_token_frame() -> Vec<[f32; 4]> {
    let gpu = Gpu::new(true).expect("vulkan init");
    let depth = gpu.texture_alloc_and_create(
        TextureDesc {
            dimensions: [W, H, 1],
            format: TextureFormat::D32Float,
            usage: UsageFlags::SAMPLED,
            ..Default::default()
        },
        Queue::Main,
        None,
    );
    let visibility = gpu.texture_alloc_and_create(
        TextureDesc {
            dimensions: [W, H, 1],
            format: TextureFormat::R32Uint,
            usage: UsageFlags::STORAGE | UsageFlags::TRANSFER_DST,
            ..Default::default()
        },
        Queue::Main,
        None,
    );
    let display = gpu.texture_alloc_and_create(
        TextureDesc {
            dimensions: [W, H, 1],
            format: TextureFormat::Rgba32Float,
            usage: UsageFlags::COLOR_ATTACHMENT | UsageFlags::TRANSFER_SRC,
            ..Default::default()
        },
        Queue::Main,
        None,
    );
    let scene = MeshScene::new(
        &gpu,
        &MeshSceneDesc {
            max_meshes: 1,
            max_instances: 1,
            max_materials: 1,
            vertex_capacity: 3,
            joint_weight_capacity: 0,
            index_capacity: 3,
            max_meshlets: 1,
        },
    );
    let linework = MeshLineworkPass::new(&gpu);
    let mut heap = gpu.heap_slots_create(2, 2, 2);
    let depth_slot = heap.add_sampled(
        &gpu,
        gpu.texture_view_descriptor(depth.texture, TextureViewDesc::default()),
    );
    let visibility_slot = heap.add_storage(
        &gpu,
        gpu.texture_rw_view_descriptor(visibility.texture, TextureViewDesc::default()),
    );
    let tokens = gpu.alloc_slice::<u32>((W * H) as u64, Memory::Default);
    // SAFETY: fresh W × H upload allocation, initialized to one valid nonzero token.
    unsafe {
        for i in 0..(W * H) as usize {
            *tokens.cpu.add(i) = 1;
        }
    }
    let readback = gpu.alloc_slice::<[f32; 4]>((W * H) as u64, Memory::Readback);
    let mut frame_alloc = TestFrameAlloc {
        gpu: &gpu,
        ptrs: Vec::new(),
    };
    let cb = gpu.commands_begin(Queue::Main);
    gpu.cmd_begin_render_pass(
        cb,
        RenderPassDesc {
            color_attachments: &[RenderAttachment {
                texture: display.texture,
                load_op: LoadOp::Clear,
                store_op: StoreOp::Store,
                clear_color: [1.0; 4],
                ..Default::default()
            }],
            ..Default::default()
        },
    );
    gpu.cmd_end_render_pass(cb);
    gpu.cmd_copy_to_texture(cb, visibility.texture, tokens.cast());
    // Transfer writes must be visible to fragment storage reads.
    gpu.cmd_barrier(
        cb,
        Stage::Transfer,
        Stage::FragmentShader,
        HazardFlags::empty(),
    );
    linework.record(
        &gpu,
        cb,
        &mut frame_alloc,
        &heap,
        &scene,
        scene.instances(),
        GpuPtr::null(),
        MeshRasterView {
            world_to_clip: Mat4::IDENTITY,
        },
        Vec3::ZERO,
        display.texture,
        depth_slot,
        visibility_slot,
        linework_dials(),
    );
    gpu.cmd_barrier(
        cb,
        Stage::RasterColorOut,
        Stage::Transfer,
        HazardFlags::empty(),
    );
    gpu.cmd_copy_texture_to_buffer(cb, display.texture, readback.cast());
    gpu.queue_submit(Queue::Main, &[cb]);
    gpu.queue_wait_idle(Queue::Main);
    // SAFETY: the readback has exactly W × H initialized RGBA32F texels.
    let pixels = unsafe { std::slice::from_raw_parts(readback.cpu, (W * H) as usize).to_vec() };

    frame_alloc.free();
    gpu.free(readback);
    gpu.free(tokens);
    heap.free(&gpu);
    linework.free(&gpu);
    scene.free(&gpu);
    gpu.texture_free_and_destroy(display);
    gpu.texture_free_and_destroy(visibility);
    gpu.texture_free_and_destroy(depth);
    pixels
}

#[test]
fn concave_crease_darkens_the_shared_edge_but_not_the_room_interior() {
    let _guard = gpu_test_lock();
    let mut view = front_view();
    view.camera_position = [0.0, 2.0, -6.0];
    view.camera_forward = Vec3::new(0.0, -0.2, 1.0).normalize().to_array();
    view.camera_up = Vec3::new(0.0, 1.0, 0.2).normalize().to_array();
    let pixels = render(view, |gpu, scene, material| {
        let positions = [
            [-20.0, 0.0, -20.0],
            [20.0, 0.0, -20.0],
            [20.0, 0.0, 4.0],
            [-20.0, 0.0, 4.0],
            [-20.0, 0.0, 4.0],
            [20.0, 0.0, 4.0],
            [20.0, 20.0, 4.0],
            [-20.0, 20.0, 4.0],
        ];
        let normals = [[0.0, 1.0, 0.0]; 8];
        let uvs = [[0.0, 0.0]; 8];
        let indices = [
            0, 2, 1, 0, 3, 2, 0, 1, 2, 0, 2, 3, // floor, both winds
            4, 6, 5, 4, 7, 6, 4, 5, 6, 4, 6, 7, // wall, both winds
        ];
        let mesh = scene.add_mesh(
            gpu,
            mesh::MeshDesc {
                positions: &positions,
                normals: &normals,
                uvs: &uvs,
                indices: &indices,
                tangents: None,
                joint_weights: None,
                colors: None,
            },
        );
        scene.add_instance(gpu, mesh, Mat4::IDENTITY, material);
    });
    assert!(dark_count(&pixels) > 0, "concave crease produced no ink");
    assert!(
        dark_count(&pixels) < (W * H / 4) as usize,
        "concave crease spread beyond a thin line"
    );
    assert!(
        is_white(pixel(&pixels, W / 2, H / 8)),
        "flat wall interior inked"
    );
}

#[test]
fn grazing_floor_stays_white_without_depth_sobel_false_positives() {
    let _guard = gpu_test_lock();
    let mut view = front_view();
    view.camera_position = [0.0, 2.0, -6.0];
    // Full coverage isolates grazing-interior false positives.
    view.camera_forward = Vec3::new(0.0, -0.68, 1.0).normalize().to_array();
    view.camera_up = Vec3::new(0.0, 1.0, 0.68).normalize().to_array();
    let pixels = render(view, |gpu, scene, material| {
        let positions = [
            [-100.0, 0.0, -100.0],
            [100.0, 0.0, -100.0],
            [100.0, 0.0, 100.0],
            [-100.0, 0.0, 100.0],
        ];
        let normals = [[0.0, 1.0, 0.0]; 4];
        let uvs = [[0.0, 0.0]; 4];
        let indices = [0, 2, 1, 0, 3, 2, 0, 1, 2, 0, 2, 3];
        let mesh = scene.add_mesh(
            gpu,
            mesh::MeshDesc {
                positions: &positions,
                normals: &normals,
                uvs: &uvs,
                indices: &indices,
                tangents: None,
                joint_weights: None,
                colors: None,
            },
        );
        scene.add_instance(gpu, mesh, Mat4::IDENTITY, material);
    });
    assert_eq!(dark_count(&pixels), 0, "flat grazing floor produced ink");
}

#[test]
fn quad_diagonal_and_flush_instance_seam_stay_white() {
    let _guard = gpu_test_lock();
    let diagonal = render(front_view(), |gpu, scene, material| {
        add_double_sided_unit_quad(
            gpu,
            scene,
            material,
            Mat4::from_scale(Vec3::new(100.0, 100.0, 1.0)),
        );
    });
    assert_eq!(
        dark_count(&diagonal),
        0,
        "coplanar quad diagonal produced ink"
    );

    let flush = render(front_view(), |gpu, scene, material| {
        let positions = [
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0],
        ];
        let normals = [[0.0, 0.0, -1.0]; 4];
        let uvs = [[0.0, 0.0]; 4];
        let indices = [0, 2, 1, 0, 3, 2, 0, 1, 2, 0, 2, 3];
        let mesh = scene.add_mesh(
            gpu,
            mesh::MeshDesc {
                positions: &positions,
                normals: &normals,
                uvs: &uvs,
                indices: &indices,
                tangents: None,
                joint_weights: None,
                colors: None,
            },
        );
        scene.add_instance(
            gpu,
            mesh,
            Mat4::from_scale(Vec3::new(50.0, 100.0, 1.0))
                * Mat4::from_translation(Vec3::new(-1.0, 0.0, 0.0)),
            material,
        );
        scene.add_instance(
            gpu,
            mesh,
            Mat4::from_scale(Vec3::new(50.0, 100.0, 1.0))
                * Mat4::from_translation(Vec3::new(1.0, 0.0, 0.0)),
            material,
        );
    });
    assert_eq!(dark_count(&flush), 0, "flush instance seam produced ink");
}

#[test]
fn parallel_floor_step_inks_a_thin_near_rim_only() {
    let _guard = gpu_test_lock();
    let mut view = front_view();
    view.camera_position = [0.0, 4.0, -7.0];
    view.camera_forward = Vec3::new(0.0, -0.35, 1.0).normalize().to_array();
    view.camera_up = Vec3::new(0.0, 1.0, 0.35).normalize().to_array();
    let pixels = render(view, |gpu, scene, material| {
        let positions = [
            [-30.0, 0.0, -30.0],
            [0.0, 0.0, -30.0],
            [0.0, 0.0, 30.0],
            [-30.0, 0.0, 30.0],
            [0.0, -1.0, -30.0],
            [30.0, -1.0, -30.0],
            [30.0, -1.0, 30.0],
            [0.0, -1.0, 30.0],
        ];
        let normals = [[0.0, 1.0, 0.0]; 8];
        let uvs = [[0.0, 0.0]; 8];
        let indices = [
            0, 2, 1, 0, 3, 2, 0, 1, 2, 0, 2, 3, // higher floor, both winds
            4, 6, 5, 4, 7, 6, 4, 5, 6, 4, 6, 7, // lower floor, both winds
        ];
        let mesh = scene.add_mesh(
            gpu,
            mesh::MeshDesc {
                positions: &positions,
                normals: &normals,
                uvs: &uvs,
                indices: &indices,
                tangents: None,
                joint_weights: None,
                colors: None,
            },
        );
        scene.add_instance(gpu, mesh, Mat4::IDENTITY, material);
    });
    let dark = dark_count(&pixels);
    assert!(dark > 0, "parallel step produced no near-side ink");
    assert!(
        dark < (W * H / 8) as usize,
        "step ink was wider than one rim"
    );
    assert!(
        is_white(pixel(&pixels, W / 4, H / 2)),
        "far floor interior inked"
    );
}

#[test]
fn void_silhouette_inks_geometry_side_and_leaves_void_white() {
    let _guard = gpu_test_lock();
    let pixels = render(front_view(), |gpu, scene, material| {
        add_double_sided_unit_quad(gpu, scene, material, Mat4::IDENTITY);
    });
    assert!(dark_count(&pixels) > 0, "void silhouette produced no ink");
    assert!(
        is_white(pixel(&pixels, 0, 0)),
        "void received silhouette ink"
    );
    assert!(
        is_white(pixel(&pixels, W - 1, H - 1)),
        "void received silhouette ink"
    );
}

#[test]
fn uniform_token_fast_path_is_identically_white() {
    let _guard = gpu_test_lock();
    let pixels = render_uniform_token_frame();
    assert_eq!(
        dark_count(&pixels),
        0,
        "uniform tokens must retire before any triangle resolve"
    );
}

#[test]
fn no_linework_flag_silences_own_edges_and_neighbor_halo() {
    let _guard = gpu_test_lock();
    // Full-frame parallel floors isolate hidden-instance halo behavior.
    let mut view = front_view();
    view.camera_position = [0.0, 4.0, -7.0];
    view.camera_forward = Vec3::new(0.0, -0.68, 1.0).normalize().to_array();
    view.camera_up = Vec3::new(0.0, 1.0, 0.68).normalize().to_array();
    let step_scene = |flag_lower: bool| {
        render(view, move |gpu, scene, material| {
            let quad = |scene: &mut MeshScene, gpu: &Gpu, positions: &[[f32; 3]; 4]| {
                scene.add_mesh(
                    gpu,
                    mesh::MeshDesc {
                        positions,
                        normals: &[[0.0, 1.0, 0.0]; 4],
                        uvs: &[[0.0, 0.0]; 4],
                        indices: &[0, 2, 1, 0, 3, 2, 0, 1, 2, 0, 2, 3],
                        tangents: None,
                        joint_weights: None,
                        colors: None,
                    },
                )
            };
            let upper = quad(
                scene,
                gpu,
                &[
                    [-100.0, 0.0, -100.0],
                    [0.0, 0.0, -100.0],
                    [0.0, 0.0, 100.0],
                    [-100.0, 0.0, 100.0],
                ],
            );
            let lower = quad(
                scene,
                gpu,
                &[
                    [0.0, -1.0, -100.0],
                    [100.0, -1.0, -100.0],
                    [100.0, -1.0, 100.0],
                    [0.0, -1.0, 100.0],
                ],
            );
            scene.add_instance(gpu, upper, Mat4::IDENTITY, material);
            let lower_instance = scene.add_instance(gpu, lower, Mat4::IDENTITY, material);
            if flag_lower {
                scene.set_flags(gpu, lower_instance, abi_mesh::MESH_FLAG_NO_LINEWORK);
            }
        })
    };
    let control = step_scene(false);
    assert!(
        dark_count(&control) > 0,
        "unflagged control produced no rim ink — the flag test would be vacuous"
    );
    let flagged = step_scene(true);
    assert_eq!(
        dark_count(&flagged),
        0,
        "NO_LINEWORK instance still inked (own rim or halo on its neighbor)"
    );
}
