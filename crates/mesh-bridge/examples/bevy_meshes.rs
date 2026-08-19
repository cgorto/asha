//! Bevy-native meshes over the bridge: `Mesh` assets and `Mesh3d` entities
//! on the Bevy side, GPU-driven culling and raster passes on the render thread.
//! The host owns pass structure, targets, camera, and lighting; Bevy transforms
//! drive motion without render-side bookkeeping.
//!
//! Run from the workspace root:  cargo run -p mesh-bridge --example bevy_meshes
//!
//! `ASHA_VERIFY=1` renders offscreen, checks a predicted pixel, and exits.

use abi_core::View;
use abi_core::glam as sg;
use abi_light::mesh_shade_slim;
use abi_mesh::{DeformerStack, LatticeDeformer};
use abi_mesh::{MeshShadeLighting, mesh_world_to_clip};
use bevy::prelude::*;
use bevy::window::{Window, WindowPlugin};
use gpu::{
    Gpu, HazardFlags, HeapSlots, LoadOp, Memory, OwnedTexture, Queue, RenderAttachment,
    RenderPassDesc, SamplerDesc, SamplerSlot, Stage, StoreOp, Texture, TextureDesc, TextureFormat,
    UsageFlags,
};
use mesh::cull::ClusterCullPass;
use mesh::{MeshDepthPrepass, MeshForwardPass, MeshForwardTargets, MeshRasterView, MeshSceneDesc};
use mesh_bridge::MeshBridge;
use render::{AshaRenderPlugin, FrameCtx, MeshMaterials, PacingPlugin, RenderScene};

const MAX_INSTANCES: u32 = 1024;
/// Maximum total meshlet capacity for one streamed frame.
const MAX_DRAWS: u32 = 4096;
const IN_FLIGHT: usize = render::FRAMES_IN_FLIGHT as usize;

const WARM: [f32; 4] = [0.8, 0.4, 0.2, 1.0];
const COOL: [f32; 4] = [0.24, 0.62, 1.0, 1.0];
const SPINNERS: u32 = 8;
const SPHERES: u32 = 5;
const CLEAR: [f32; 4] = [0.013, 0.015, 0.03, 1.0];
const TRANSLATED_REST_FACE: sg::Vec3 = sg::Vec3::new(0.0, -2.6, -1.0);
const TRANSLATED_SHIFTED_FACE: sg::Vec3 = sg::Vec3::new(2.0, -2.6, -1.0);

/// Camera at z = −8 looking toward +Z; the center cube fills the probe.
fn view(size: sg::UVec2) -> View {
    View {
        camera_position: [0.0, 0.0, -8.0],
        tan_half_fov: 0.55,
        camera_forward: sg::Vec3::Z.to_array(),
        aspect: size.x as f32 / size.y as f32,
        camera_right: sg::Vec3::NEG_X.to_array(),
        depth_near_plane: 0.1,
        camera_up: sg::Vec3::Y.to_array(),
        _pad: 0,
        output_size: size.to_array(),
        _pad2: [0; 2],
    }
}

/// Lighting with the sun aligned to the center cube's front face.
fn lighting() -> MeshShadeLighting {
    MeshShadeLighting {
        sun_direction: sg::Vec3::NEG_Z.to_array(),
        sun_tint: [1.0, 0.75, 0.5],
        sky_ambient: [0.2, 0.25, 0.3],
        ground_ambient: [0.05, 0.04, 0.03],
        ..MeshShadeLighting::zeroed()
    }
}

fn cube_lattice_base() -> LatticeDeformer {
    let lattice_to_model = sg::Mat4::from_scale(sg::Vec3::splat(2.0))
        * sg::Mat4::from_translation(sg::Vec3::splat(-0.5));
    LatticeDeformer {
        model_to_lattice: lattice_to_model.inverse(),
        lattice_to_model,
        resolution: [2, 2, 2],
        falloff: 0.0,
        offsets: [[0.0; 4]; abi_mesh::MAX_LATTICE_POINTS],
    }
}

fn uniform_lattice_stack(offset: sg::Vec3) -> DeformerStack {
    let mut stack = DeformerStack::zeroed();
    let mut lattice = cube_lattice_base();
    for i in 0..8 {
        lattice.offsets[i] = [offset.x, offset.y, offset.z, 0.0];
    }
    stack.count = 1;
    stack.lattices[0] = lattice;
    stack
}

fn wobble_lattice_stack(frame: u32) -> DeformerStack {
    let mut stack = DeformerStack::zeroed();
    stack.count = 1;
    stack.lattices[0] = wobble_lattice(frame);
    stack
}

fn wobble_lattice(frame: u32) -> LatticeDeformer {
    let mut lattice = cube_lattice_base();
    let t = frame as f32 * 0.09;
    for z in 0..2 {
        for y in 0..2 {
            for x in 0..2 {
                let idx = z * 4 + y * 2 + x;
                let phase = t + x as f32 * 1.7 + y as f32 * 2.3 + z as f32 * 3.1;
                lattice.offsets[idx] = [
                    phase.sin() * 0.035,
                    (phase * 1.37).cos() * 0.025,
                    (phase * 0.73).sin() * 0.02,
                    0.0,
                ];
            }
        }
    }
    lattice
}

fn verify_pixel_for_world(p: sg::Vec3) -> (u32, u32) {
    let view = view(sg::UVec2::splat(VERIFY_SIZE));
    let clip = mesh_world_to_clip(&view) * p.extend(1.0);
    let ndc = clip.truncate() / clip.w;
    let x = ((ndc.x * 0.5 + 0.5) * VERIFY_SIZE as f32).floor() as i32;
    let y = ((ndc.y * 0.5 + 0.5) * VERIFY_SIZE as f32).floor() as i32;
    (
        x.clamp(0, VERIFY_SIZE as i32 - 1) as u32,
        y.clamp(0, VERIFY_SIZE as i32 - 1) as u32,
    )
}

const VERIFY_SIZE: u32 = 128;

/// Verify owns a parallel screen-sized state (its own prepass, depth,
/// target) because `MeshDepthPrepass` is sized to exactly one extent.
struct Verify {
    prepass: MeshDepthPrepass,
    depth: OwnedTexture,
    target: OwnedTexture,
    readback: gpu::Ptr<[f32; 4]>,
    count_seen: u32,
}

/// The GPU-owning half, one struct so teardown can move it out whole
/// (every `free` takes self by value — the Option is for that move alone,
/// the example's own idiom).
struct Pipeline {
    bridge: MeshBridge,
    cull: ClusterCullPass,
    prepass: MeshDepthPrepass,
    forward: MeshForwardPass,
}

struct BevyMeshScene {
    /// The forward shader declares the shared bindless heap even when every
    /// material takes its slot-0 identity ramp path.
    heap: Option<HeapSlots>,
    ramp_default_sampler: SamplerSlot,
    pipeline: Option<Pipeline>,
    /// Screen-sized reverse-Z D32, recreated on resize. None until the
    /// first frame (the screen does not exist at scene construction).
    depth: Option<OwnedTexture>,
    size: sg::UVec2,
    verify: Option<Verify>,
}

impl BevyMeshScene {
    fn new(gpu: &Gpu) -> Self {
        let mut heap = gpu.heap_slots_create(2, 2, 2);
        let ramp_default_sampler =
            heap.add_sampler(gpu, gpu.sampler_descriptor(SamplerDesc::default()));
        let bridge = MeshBridge::new(
            gpu,
            &MeshSceneDesc {
                max_meshes: 8,
                max_instances: MAX_INSTANCES,
                max_materials: 8,
                vertex_capacity: 1 << 14,
                joint_weight_capacity: 0,
                index_capacity: 1 << 18,
                max_meshlets: 2048,
            },
        );
        let verify = std::env::var_os("ASHA_VERIFY").map(|_| Verify {
            prepass: {
                let mut p = MeshDepthPrepass::new(gpu);
                p.resize(gpu, sg::UVec2::splat(VERIFY_SIZE));
                p
            },
            depth: alloc_depth(gpu, sg::UVec2::splat(VERIFY_SIZE)),
            target: gpu.texture_alloc_and_create(
                TextureDesc {
                    dimensions: [VERIFY_SIZE, VERIFY_SIZE, 1],
                    format: TextureFormat::Rgba32Float,
                    usage: UsageFlags::COLOR_ATTACHMENT | UsageFlags::TRANSFER_SRC,
                    ..Default::default()
                },
                Queue::Main,
                None,
            ),
            readback: gpu
                .alloc_slice::<[f32; 4]>((VERIFY_SIZE * VERIFY_SIZE) as u64, Memory::Readback),
            count_seen: 0,
        });
        Self {
            heap: Some(heap),
            ramp_default_sampler,
            pipeline: Some(Pipeline {
                bridge,
                cull: ClusterCullPass::with_capacity(gpu, MAX_DRAWS, MAX_INSTANCES, IN_FLIGHT),
                prepass: MeshDepthPrepass::new(gpu),
                forward: MeshForwardPass::new(gpu),
            }),
            depth: None,
            size: sg::UVec2::ZERO,
            verify,
        }
    }

    fn check_verify_pixels(&self, ctx: &mut FrameCtx) {
        let v = self.verify.as_ref().unwrap();
        let px = |x: u32, y: u32| -> [f32; 4] {
            // SAFETY: readback sized VERIFY_SIZE²; the frame gate proves
            // frame 30's transfer done by frame 33's pacing wait.
            unsafe { *v.readback.cpu.add((y * VERIFY_SIZE + x) as usize) }
        };
        assert_eq!(
            v.count_seen,
            (2 + SPINNERS + SPHERES),
            "extract count wrong — Mesh3d entities did not reach the arena"
        );
        // Center cube face under the aligned sun.
        let expected = mesh_shade_slim(sg::Vec3::NEG_Z, [WARM[0], WARM[1], WARM[2]], &lighting());
        let center = px(VERIFY_SIZE / 2, VERIFY_SIZE / 2);
        for c in 0..3 {
            assert!(
                (center[c] - expected[c]).abs() < 2.0e-3,
                "verify: center channel {c} = {} expected {} — the mesh did not draw",
                center[c],
                expected[c],
            );
        }
        // Corner pixel lies outside every instance.
        let corner = px(2, 2);
        for c in 0..3 {
            assert!(
                (corner[c] - CLEAR[c]).abs() < 1.0e-6,
                "verify: corner channel {c} = {} expected clear {}",
                corner[c],
                CLEAR[c],
            );
        }
        let rest_px = verify_pixel_for_world(TRANSLATED_REST_FACE);
        let rest = px(rest_px.0, rest_px.1);
        assert_eq!(
            rest, CLEAR,
            "verify: translated cube rest pixel {rest_px:?} was not clear"
        );
        let shifted_px = verify_pixel_for_world(TRANSLATED_SHIFTED_FACE);
        let shifted = px(shifted_px.0, shifted_px.1);
        let shifted_expected =
            mesh_shade_slim(sg::Vec3::NEG_Z, [WARM[0], WARM[1], WARM[2]], &lighting());
        // Equal depth detects any prepass/forward position mismatch.
        for c in 0..3 {
            assert_eq!(
                shifted[c].to_bits(),
                shifted_expected[c].to_bits(),
                "verify: shifted channel {c} = {} expected {} at {shifted_px:?}",
                shifted[c],
                shifted_expected[c],
            );
        }
        println!(
            "VERIFY OK center={center:?} instances={} frame={}",
            v.count_seen, ctx.frame
        );
        ctx.request_exit();
    }
}

impl Pipeline {
    /// Records cull, prepass, clear, and forward for one target set.
    #[allow(clippy::too_many_arguments)] // Each argument is a real dependency.
    fn record(
        &self,
        ctx: &mut FrameCtx,
        heap: &HeapSlots,
        ramp_default_sampler: SamplerSlot,
        prepass: &MeshDepthPrepass,
        hdr: Texture,
        depth: Texture,
        size: sg::UVec2,
        clear: [f32; 4],
        slot: usize,
    ) {
        let gpu = ctx.gpu;
        let cb = ctx.cb;
        let view = view(size);
        let raster_view = MeshRasterView {
            world_to_clip: mesh_world_to_clip(&view),
        };
        let camera_pos = sg::Vec3::from_array(view.camera_position);
        let scene = self.bridge.scene();
        let instances = self.bridge.instances();

        let mut fa = mesh_bridge::Alloc(ctx);
        self.cull.record(
            gpu,
            cb,
            &mut fa,
            scene,
            instances,
            raster_view,
            camera_pos,
            slot,
        );
        prepass.record(
            gpu,
            cb,
            &mut fa,
            scene,
            instances,
            self.cull.output(),
            self.cull.clusters(),
            self.cull.draw_count_ptr(gpu, slot),
            depth,
            size,
            raster_view,
        );
        // The forward pass loads its color target; this path clears separately.
        gpu.cmd_begin_render_pass(
            cb,
            RenderPassDesc {
                render_area_size: size.to_array(),
                color_attachments: &[RenderAttachment {
                    texture: hdr,
                    load_op: LoadOp::Clear,
                    store_op: StoreOp::Store,
                    clear_color: clear,
                    ..Default::default()
                }],
                ..Default::default()
            },
        );
        gpu.cmd_end_render_pass(cb);
        // Make the clear visible before the forward load.
        gpu.cmd_barrier(
            cb,
            Stage::RasterColorOut,
            Stage::RasterColorOut,
            HazardFlags::empty(),
        );
        self.forward.record(
            gpu,
            cb,
            &mut fa,
            heap,
            scene,
            instances,
            self.cull.output(),
            self.cull.clusters(),
            self.cull.draw_count_ptr(gpu, slot),
            MeshForwardTargets {
                color: hdr,
                depth,
                size,
                color_load_op: LoadOp::Load,
                clear_color: [0.0; 4],
            },
            raster_view,
            camera_pos,
            lighting(),
            ramp_default_sampler,
        );
    }
}

fn alloc_depth(gpu: &Gpu, size: sg::UVec2) -> OwnedTexture {
    gpu.texture_alloc_and_create(
        TextureDesc {
            dimensions: [size.x, size.y, 1],
            format: TextureFormat::D32Float,
            usage: UsageFlags::DEPTH_STENCIL_ATTACHMENT,
            ..Default::default()
        },
        Queue::Main,
        None,
    )
}

impl RenderScene for BevyMeshScene {
    fn draw(&mut self, ctx: &mut FrameCtx) {
        let slot = (ctx.frame % render::FRAMES_IN_FLIGHT) as usize;
        // Read completed-slot survivor counts before resetting the slot.
        let prior_survivors: u32 = {
            let pipeline = self.pipeline.as_ref().unwrap();
            let instances = pipeline.bridge.instances();
            pipeline.cull.assert_counts(slot, instances.batches_cpu);
            instances
                .batches_cpu
                .iter()
                .enumerate()
                .map(|(batch, _)| pipeline.cull.visible_count(slot, batch as u32))
                .sum()
        };

        // Cross mesh data once per frame.
        self.pipeline.as_mut().unwrap().bridge.ingest(ctx);

        let dims = ctx.backbuffer.dimensions;
        let size = sg::UVec2::new(dims[0], dims[1]);
        if self.size != size {
            // Resize after waiting for GPU idle.
            ctx.gpu.wait_idle();
            if let Some(old) = self.depth.take() {
                ctx.gpu.texture_free_and_destroy(old);
            }
            self.depth = Some(alloc_depth(ctx.gpu, size));
            self.pipeline
                .as_mut()
                .unwrap()
                .prepass
                .resize(ctx.gpu, size);
            self.size = size;
        }

        let pipeline = self.pipeline.as_ref().unwrap();
        let heap = self.heap.as_ref().expect("mesh heap exists");
        if let Some(v) = &self.verify {
            if ctx.frame == 30 {
                pipeline.record(
                    ctx,
                    heap,
                    self.ramp_default_sampler,
                    &v.prepass,
                    v.target.texture,
                    v.depth.texture,
                    sg::UVec2::splat(VERIFY_SIZE),
                    CLEAR,
                    slot,
                );
                let gpu = ctx.gpu;
                gpu.cmd_barrier(
                    ctx.cb,
                    Stage::RasterColorOut,
                    Stage::Transfer,
                    HazardFlags::empty(),
                );
                gpu.cmd_copy_texture_to_buffer(ctx.cb, v.target.texture, v.readback.cast());
                self.verify.as_mut().unwrap().count_seen = pipeline.bridge.instances().count();
                return; // Verify frame renders offscreen only.
            } else if ctx.frame == 36 {
                assert!(
                    prior_survivors > 0,
                    "cluster cull emitted zero visible clusters at frame 33"
                );
                self.check_verify_pixels(ctx);
                return;
            }
        }

        let depth = self.depth.as_ref().unwrap().texture;
        pipeline.record(
            ctx,
            heap,
            self.ramp_default_sampler,
            &pipeline.prepass,
            ctx.backbuffer,
            depth,
            size,
            CLEAR,
            slot,
        );
    }

    fn teardown(&mut self, gpu: &Gpu) {
        if let Some(v) = self.verify.take() {
            v.prepass.free(gpu);
            gpu.texture_free_and_destroy(v.depth);
            gpu.texture_free_and_destroy(v.target);
            gpu.free(v.readback);
        }
        if let Some(depth) = self.depth.take() {
            gpu.texture_free_and_destroy(depth);
        }
        let pipeline = self.pipeline.take().expect("teardown runs once");
        pipeline.cull.free(gpu);
        pipeline.prepass.free(gpu);
        pipeline.forward.free(gpu);
        pipeline.bridge.free(gpu);
        if let Some(heap) = self.heap.take() {
            heap.free(gpu);
        }
    }
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "asha × bevy — native Mesh3d over the mesh bridge".into(),
                resolution: bevy::window::WindowResolution::new(1280, 720),
                ..Default::default()
            }),
            ..Default::default()
        }))
        .add_plugins(AshaRenderPlugin::new(BevyMeshScene::new).extract_meshes())
        .add_plugins(PacingPlugin::default())
        .add_systems(Startup, spawn_world)
        .add_systems(Update, (spin, wobble, esc_to_exit))
        .run();
}

/// Marker for entities driven by the spin system.
#[derive(Component)]
struct Spinner;

#[derive(Component)]
struct Wobbler;

fn spawn_world(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<MeshMaterials>,
) {
    let cube = meshes.add(Cuboid::new(2.0, 2.0, 2.0));
    let sphere = meshes.add(Sphere::new(1.0));

    let mut warm = abi_mesh::MaterialEntry::standard();
    warm.base_color_factor = WARM;
    let warm = materials.add(warm);
    let mut cool = abi_mesh::MaterialEntry::standard();
    cool.base_color_factor = COOL;
    let cool = materials.add(cool);

    // Center cube: sole owner of the verification pixel.
    commands.spawn((
        Mesh3d(cube.clone()),
        warm,
        Transform::IDENTITY,
        uniform_lattice_stack(sg::Vec3::ZERO),
    ));

    // Uniform lattice offsets produce a rigid +2 world-X translation.
    commands.spawn((
        Mesh3d(cube.clone()),
        warm,
        Transform::from_xyz(0.0, -2.6, 0.0),
        uniform_lattice_stack(sg::Vec3::X),
    ));

    // Surrounding cubes rotate through Bevy transform updates.
    // Half-step phases keep them off the verification ray.
    for i in 0..SPINNERS {
        let angle = (i as f32 + 0.5) / SPINNERS as f32 * std::f32::consts::TAU;
        let mut entity = commands.spawn((
            Mesh3d(cube.clone()),
            if i % 2 == 0 { warm } else { cool },
            Transform::from_xyz(angle.cos() * 5.5, 0.0, angle.sin() * 5.5)
                .with_scale(Vec3::splat(0.7)),
            Spinner,
        ));
        if i == 0 {
            entity.insert((Wobbler, wobble_lattice_stack(0)));
        }
    }

    // Overhead icospheres exercise cluster-level culling.
    for i in 0..SPHERES {
        let x = (i as f32 - (SPHERES - 1) as f32 * 0.5) * 2.4;
        commands.spawn((
            Mesh3d(sphere.clone()),
            cool,
            Transform::from_xyz(x, 3.2, 1.0).with_scale(Vec3::splat(0.8)),
        ));
    }
}

fn spin(time: Res<Time>, mut spinners: Query<&mut Transform, With<Spinner>>) {
    for mut transform in &mut spinners {
        transform.rotate_y(time.delta_secs() * 0.9);
        transform.rotate_x(time.delta_secs() * 0.4);
    }
}

fn wobble(mut frame: Local<u32>, mut wobblers: Query<&mut DeformerStack, With<Wobbler>>) {
    *frame = (*frame).wrapping_add(1);
    for mut stack in &mut wobblers {
        stack.count = 1;
        stack.lattices[0] = wobble_lattice(*frame);
    }
}

fn esc_to_exit(keys: Res<ButtonInput<KeyCode>>, mut exit: MessageWriter<AppExit>) {
    if keys.just_pressed(KeyCode::Escape) {
        exit.write(AppExit::Success);
    }
}
