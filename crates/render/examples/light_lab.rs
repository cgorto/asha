//! Deterministic local-light and shadow rendering example.
//! Scenarios are driven by frame number for reproducible captures and analysis.

mod common;

use abi_core::glam::{Mat4, UVec2, Vec3};
use abi_light::DepthMarchConfig;
use abi_light::PointLight;
use abi_mesh::{MaterialEntry, MeshShadeLighting};
use bevy::prelude::*;
use bevy::window::{Window, WindowPlugin};
use common::esc_to_exit;
use gpu::pass::Pass;
use gpu::{
    HazardFlags, HeapSlots, LoadOp, Memory, OwnedTexture, PassTimer, Queue, SamplerDesc, Stage,
    TextureDesc, TextureFormat, TextureViewDesc, UsageFlags,
};
use mesh::{
    ClusterCullPass, DrawTransform, InstanceHandle, MeshDepthPrepass, MeshForwardPass,
    MeshForwardTargets, MeshLightField, MeshRasterView, MeshScene, MeshSceneDesc, ShadowBlasDesc,
};
use render::{
    AshaRenderPlugin, FrameCtx, LocalLightPass, LocalShadowDirectPass, LocalShadowPass,
    LocalShadowTemporal, MeshShadowPass, MeshSurfaceTargets, PacingPlugin, RenderScene,
};

const MAX_DRAWS: u32 = 16;
const MAX_CLUSTERS: u32 = 1024;
const MAX_LIGHTS: u32 = 4;
const SHADOW_ORIGIN_BIAS: f32 = 1.0e-3;
const SHADOW_DESTINATION_BIAS: f32 = 1.0e-3;
/// Deck-like default; the window may be resized by the WM, all buffers track.
const WINDOW_SIZE: (u32, u32) = (1280, 800);
/// LAB_TIMING zone boundaries, in record order.
const PASS_NAMES: [&str; 6] = ["cull", "prepass", "forward", "shadow", "lights", "post"];

fn env_f32(name: &str, default: f32) -> f32 {
    std::env::var(name)
        .ok()
        .map(|v| {
            v.parse()
                .unwrap_or_else(|_| panic!("{name} must be an f32"))
        })
        .unwrap_or(default)
}

fn env_u32(name: &str, default: u32) -> u32 {
    std::env::var(name)
        .ok()
        .map(|v| v.parse().unwrap_or_else(|_| panic!("{name} must be a u32")))
        .unwrap_or(default)
}

#[derive(Clone, Copy, PartialEq)]
enum Scenario {
    Static,
    Orbit,
    Walker,
    CamPan,
    Cut,
    Teleport,
    All,
    GameCam,
}

impl Scenario {
    fn from_env() -> Self {
        match std::env::var("LAB_SCENARIO").as_deref() {
            Ok("static") => Self::Static,
            Ok("orbit") => Self::Orbit,
            Ok("walker") => Self::Walker,
            Ok("campan") => Self::CamPan,
            Ok("cut") => Self::Cut,
            Ok("teleport") => Self::Teleport,
            Ok("gamecam") => Self::GameCam,
            Ok("all") | Err(_) => Self::All,
            Ok(other) => panic!("unknown LAB_SCENARIO {other}"),
        }
    }
}

/// Maximum number of transforms staged each frame.
const MOVER_COUNT: usize = 3;

/// Everything the scripted frame decides: camera, lights, mover poses.
struct FrameScript {
    eye: Vec3,
    focus: Vec3,
    lights: [PointLight; MAX_LIGHTS as usize],
    light_count: u32,
    movers: [Mat4; MOVER_COUNT],
}

/// Produces deterministic camera, light, and instance transforms.
fn script(scenario: Scenario, frame: u64, light_count: u32) -> FrameScript {
    let t = frame as f32 / 60.0;
    let (orbit_on, walk_on, cam) = match scenario {
        Scenario::Static => (false, false, 0),
        Scenario::Orbit => (true, false, 0),
        Scenario::Walker => (false, true, 0),
        Scenario::CamPan => (false, false, 1),
        Scenario::Cut => (false, false, 2),
        Scenario::Teleport => (false, false, 3),
        Scenario::All => (true, true, 4),
        Scenario::GameCam => (true, true, 5),
    };

    let (eye, focus) = match cam {
        1 => {
            let a = t * 0.35;
            (
                Vec3::new(a.sin() * 10.0, 8.5, a.cos() * 10.0),
                Vec3::new(0.0, 0.5, 0.0),
            )
        }
        2 => {
            if (frame / 96) % 2 == 0 {
                (Vec3::new(0.0, 8.5, 10.0), Vec3::new(0.0, 0.5, 0.0))
            } else {
                (Vec3::new(7.0, 6.0, -7.0), Vec3::new(-1.0, 0.5, 1.0))
            }
        }
        4 => {
            let a = (t * 0.12).sin() * 0.35;
            (
                Vec3::new(a.sin() * 10.0, 8.5, a.cos() * 10.0),
                Vec3::new(0.0, 0.5, 0.0),
            )
        }
        5 => {
            let a = t * 0.45;
            (
                Vec3::new(a.sin() * 9.5, 7.8 + (t * 0.9).sin() * 0.9, a.cos() * 9.5),
                Vec3::new(0.0, 0.6, 0.0),
            )
        }
        _ => (Vec3::new(0.0, 8.5, 10.0), Vec3::new(0.0, 0.5, 0.0)),
    };

    let lantern_pos = if scenario == Scenario::Teleport {
        if (frame / 96) % 2 == 0 {
            Vec3::new(3.0, 2.6, 3.0)
        } else {
            Vec3::new(-3.5, 2.6, -2.0)
        }
    } else if orbit_on {
        let a = t * 0.5;
        Vec3::new(a.cos() * 3.5, 2.6, a.sin() * 3.5)
    } else {
        Vec3::new(2.0, 2.6, 3.0)
    };
    let mut lights = [PointLight::default(); MAX_LIGHTS as usize];
    lights[0] = PointLight {
        position: lantern_pos.to_array(),
        radius: 30.0,
        color: [1.0, 0.78, 0.52],
        intensity: 6.0,
    };
    lights[1] = PointLight {
        position: [2.5, 2.4, 1.5],
        radius: 14.0,
        color: [1.0, 0.62, 0.24],
        intensity: 5.0,
    };
    lights[2] = PointLight {
        position: if scenario == Scenario::GameCam {
            [
                (t * 0.8).sin() * 4.0,
                2.3 + (t * 1.6).cos() * 0.35,
                -1.0 + (t * 0.33).sin() * 2.0,
            ]
        } else {
            [-4.0, 2.0, 4.0]
        },
        radius: 12.0,
        color: [0.4, 0.7, 1.0],
        intensity: 3.0,
    };
    lights[3] = PointLight {
        position: [-4.0, 2.0, -4.0],
        radius: 12.0,
        color: [0.7, 1.0, 0.5],
        intensity: 3.0,
    };

    let wx = if walk_on {
        ((t * 1.2).sin()) * 3.5
    } else {
        -1.5
    };
    let walker = Mat4::from_scale_rotation_translation(
        Vec3::new(0.7, 1.1, 0.5),
        abi_core::glam::Quat::IDENTITY,
        Vec3::new(wx, 0.55, 2.5),
    );

    let slab = if scenario == Scenario::GameCam {
        Mat4::from_scale_rotation_translation(
            Vec3::new(2.4, 0.15, 2.4),
            abi_core::glam::Quat::from_rotation_y(t * 1.3)
                * abi_core::glam::Quat::from_rotation_x((t * 0.5).sin() * 0.35),
            Vec3::new(-2.5, 1.6, 2.0),
        )
    } else {
        Mat4::from_scale_rotation_translation(
            Vec3::new(2.4, 0.15, 2.4),
            abi_core::glam::Quat::IDENTITY,
            Vec3::new(-2.5, 1.6, 2.0),
        )
    };

    let tumbler = Mat4::from_scale_rotation_translation(
        Vec3::splat(0.6),
        abi_core::glam::Quat::from_rotation_x(t * 1.9)
            * abi_core::glam::Quat::from_rotation_z(t * 1.1),
        Vec3::new(
            (t * 0.9).cos() * 3.0,
            1.0 + (t * 1.7).sin().abs() * 1.2,
            -3.2,
        ),
    );

    FrameScript {
        eye,
        focus,
        lights,
        light_count,
        movers: [walker, slab, tumbler],
    }
}

fn world_to_clip(eye: Vec3, focus: Vec3, size: UVec2) -> Mat4 {
    let view = Mat4::look_at_rh(eye, focus, Vec3::Y);
    let aspect = size.x as f32 / size.y as f32;
    let mut proj = Mat4::perspective_infinite_reverse_rh(0.9, aspect, NEAR_PLANE);
    proj.y_axis.y *= -1.0; // Vulkan NDC: +y is down.
    proj * view
}

const NEAR_PLANE: f32 = 0.5;

/// Keeps local-light contrast readable in captures.
fn ambient(scale: f32) -> MeshShadeLighting {
    let s = |rgb: [f32; 3]| [rgb[0] * scale, rgb[1] * scale, rgb[2] * scale];
    MeshShadeLighting {
        sun_direction: Vec3::new(-0.35, -0.6, 0.55).normalize().to_array(),
        sun_tint: s([0.25, 0.24, 0.22]),
        sky_ambient: s([0.10, 0.11, 0.13]),
        ground_ambient: s([0.05, 0.045, 0.04]),
        ..MeshShadeLighting::zeroed()
    }
}

struct Dump {
    dir: std::path::PathBuf,
    start: u64,
    count: u64,
}

struct LabScene {
    scenario: Scenario,
    light_count: u32,
    exact: bool,
    v3: bool,
    stats: bool,
    dump: Option<Dump>,

    ambient_scale: f32,

    heap: Option<HeapSlots>,
    ramp_default_sampler: gpu::SamplerSlot,
    clamp_sampler: gpu::SamplerSlot,
    scene: Option<MeshScene>,
    /// Mover transforms in script order.
    movers: Vec<InstanceHandle>,
    mover_staging: [[gpu::Ptr<DrawTransform>; MOVER_COUNT]; render::FRAMES_IN_FLIGHT as usize],

    cull: Option<ClusterCullPass>,
    prepass: Option<MeshDepthPrepass>,
    forward: Option<MeshForwardPass>,
    surfaces: Option<MeshSurfaceTargets>,
    shadows_v2: Option<LocalShadowPass>,
    shadows_v3: Option<LocalShadowDirectPass>,
    shadows_exact: Option<MeshShadowPass>,
    local_lights: Option<LocalLightPass>,
    tonemap: Option<post::TonemapPass>,

    hdr: Option<OwnedTexture>,
    depth: Option<OwnedTexture>,
    display: Option<OwnedTexture>,
    hdr_slot: gpu::SampledSlot,
    hdr_rw_slot: gpu::StorageSlot,
    depth_slot: gpu::SampledSlot,
    size: UVec2,
    timer: Option<PassTimer>,
}

impl LabScene {
    fn new(gpu: &gpu::Gpu) -> Self {
        let mut heap = gpu.heap_slots_create(24, 2, 4);
        let ramp_default_sampler =
            heap.add_sampler(gpu, gpu.sampler_descriptor(SamplerDesc::default()));
        let clamp_sampler = heap.add_sampler(
            gpu,
            gpu.sampler_descriptor(SamplerDesc {
                address_mode_u: gpu::AddressMode::ClampToEdge,
                address_mode_v: gpu::AddressMode::ClampToEdge,
                address_mode_w: gpu::AddressMode::ClampToEdge,
                ..Default::default()
            }),
        );
        let mut tonemap = post::TonemapPass::new(gpu, &mut heap);
        let cb = gpu.commands_begin(Queue::Main);
        tonemap.upload(gpu, cb);
        heap.bind(gpu, cb);
        gpu.queue_submit(Queue::Main, &[cb]);
        gpu.queue_wait_idle(Queue::Main);
        tonemap.upload_finish(gpu);

        let scenario = Scenario::from_env();
        let (scene, movers) = build_scene(gpu, scenario == Scenario::GameCam);
        let (hdr_slot, hdr_rw_slot, depth_slot) = {
            let heap = &mut heap;
            (
                heap.alloc_sampled(),
                heap.alloc_storage(),
                heap.alloc_sampled(),
            )
        };
        let mover_staging = std::array::from_fn(|_| {
            std::array::from_fn(|_| gpu.alloc::<DrawTransform>(Memory::Default))
        });

        let dump = std::env::var_os("LAB_DUMP").map(|dir| {
            let dir = std::path::PathBuf::from(dir);
            std::fs::create_dir_all(&dir).expect("LAB_DUMP dir must be creatable");
            Dump {
                dir,
                start: u64::from(env_u32("LAB_START", 8)),
                count: u64::from(env_u32("LAB_COUNT", 120)),
            }
        });

        Self {
            scenario,
            light_count: env_u32(
                "LAB_LIGHTS",
                if scenario == Scenario::GameCam { 3 } else { 2 },
            )
            .clamp(1, MAX_LIGHTS),
            exact: std::env::var("LAB_EXACT").is_ok_and(|v| v != "0"),
            v3: std::env::var("LAB_V3").is_ok_and(|v| v != "0"),
            stats: std::env::var("LAB_STATS").is_ok_and(|v| v != "0"),
            dump,
            ambient_scale: env_f32("LAB_AMBIENT", 1.0),
            ramp_default_sampler,
            clamp_sampler,
            scene: Some(scene),
            movers,
            mover_staging,
            cull: Some(ClusterCullPass::with_capacity(
                gpu,
                MAX_CLUSTERS,
                MAX_DRAWS,
                render::FRAMES_IN_FLIGHT as usize,
            )),
            prepass: Some(MeshDepthPrepass::new(gpu)),
            forward: Some(MeshForwardPass::new(gpu)),
            surfaces: None,
            shadows_v2: None,
            shadows_v3: None,
            shadows_exact: None,
            local_lights: Some(LocalLightPass::new(gpu)),
            tonemap: Some(tonemap),
            heap: Some(heap),
            hdr: None,
            depth: None,
            display: None,
            hdr_slot,
            hdr_rw_slot,
            depth_slot,
            size: UVec2::ZERO,
            timer: std::env::var("LAB_TIMING")
                .is_ok_and(|v| v != "0")
                .then(|| PassTimer::new(gpu, &PASS_NAMES, render::FRAMES_IN_FLIGHT as usize)),
        }
    }

    fn temporal(&self) -> LocalShadowTemporal {
        LocalShadowTemporal {
            refresh_interval: env_u32("ASHA_SHADOW_REFRESH", 8),
            validate_thickness: env_f32("LAB_VALIDATE_THICK", 1.0),
            near_plane: NEAR_PLANE,
            light_epsilon: env_f32("LAB_LIGHT_EPS", 1.0),
            contact: (std::env::var("ASHA_SHADOW_CONTACT").map_or(true, |v| v != "0")).then_some(
                DepthMarchConfig {
                    linear_steps: 4,
                    continue_after_deep_penetration: 1,
                    jitter: 0.5,
                    depth_thickness: env_f32("LAB_CONTACT_THICK", 1.0),
                    near_plane: NEAR_PLANE,
                    _pad: [0; 3],
                },
            ),
            contact_distance: env_f32("LAB_CONTACT_DIST", 4.0),
            ray_budget: env_u32("ASHA_SHADOW_BUDGET", 0),
            edge_promotion: std::env::var("ASHA_SHADOW_EDGES").map_or(true, |v| v != "0"),
            occluded_refresh: env_u32("ASHA_SHADOW_OCCLUDED_REFRESH", 64),
            source_radius: env_f32("LAB_SOURCE_RADIUS", 0.25),
        }
    }

    fn ensure_targets(&mut self, ctx: &FrameCtx) {
        let size = std::env::var("LAB_RENDER_SIZE").map_or_else(
            |_| UVec2::new(ctx.backbuffer.dimensions[0], ctx.backbuffer.dimensions[1]),
            |v| {
                let (w, h) = v.split_once('x').expect("LAB_RENDER_SIZE must be WxH");
                UVec2::new(
                    w.parse().expect("LAB_RENDER_SIZE width"),
                    h.parse().expect("LAB_RENDER_SIZE height"),
                )
            },
        );
        if self.size == size {
            return;
        }
        let gpu = ctx.gpu;
        let heap = self.heap.as_mut().expect("heap exists");
        if self.hdr.is_some() {
            gpu.queue_wait_idle(Queue::Main);
        }
        if let Some(t) = self.hdr.take() {
            gpu.texture_free_and_destroy(t);
        }
        if let Some(t) = self.depth.take() {
            gpu.texture_free_and_destroy(t);
        }
        if let Some(t) = self.display.take() {
            gpu.texture_free_and_destroy(t);
        }
        let hdr = gpu.texture_alloc_and_create(
            TextureDesc {
                dimensions: [size.x, size.y, 1],
                format: TextureFormat::Rgba32Float,
                usage: UsageFlags::COLOR_ATTACHMENT | UsageFlags::SAMPLED | UsageFlags::STORAGE,
                ..Default::default()
            },
            Queue::Main,
            None,
        );
        heap.write_sampled(
            gpu,
            self.hdr_slot,
            gpu.texture_view_descriptor(hdr.texture, TextureViewDesc::default()),
        );
        heap.write_storage(
            gpu,
            self.hdr_rw_slot,
            gpu.texture_rw_view_descriptor(hdr.texture, TextureViewDesc::default()),
        );
        let depth = gpu.texture_alloc_and_create(
            TextureDesc {
                dimensions: [size.x, size.y, 1],
                format: TextureFormat::D32Float,
                usage: UsageFlags::DEPTH_STENCIL_ATTACHMENT | UsageFlags::SAMPLED,
                ..Default::default()
            },
            Queue::Main,
            None,
        );
        heap.write_sampled(
            gpu,
            self.depth_slot,
            gpu.texture_view_descriptor(depth.texture, TextureViewDesc::default()),
        );
        self.display = Some(gpu.texture_alloc_and_create(
            TextureDesc {
                dimensions: [size.x, size.y, 1],
                format: TextureFormat::Rgba8Unorm,
                usage: UsageFlags::COLOR_ATTACHMENT | UsageFlags::TRANSFER_SRC,
                ..Default::default()
            },
            Queue::Main,
            None,
        ));
        self.hdr = Some(hdr);
        self.depth = Some(depth);

        self.prepass
            .as_mut()
            .expect("prepass exists")
            .resize(gpu, size);
        let heap = self.heap.as_mut().expect("heap exists");
        assert!(MeshSurfaceTargets::ensure(
            &mut self.surfaces,
            gpu,
            heap,
            size
        ));
        if self.exact {
            match &mut self.shadows_exact {
                Some(pass) => {
                    assert!(pass.resize(gpu, size));
                }
                None => {
                    self.shadows_exact =
                        Some(MeshShadowPass::new(gpu, size, MAX_LIGHTS, MAX_DRAWS));
                }
            }
        } else if self.v3 {
            match &mut self.shadows_v3 {
                Some(pass) => {
                    assert!(pass.resize(gpu, size));
                }
                None => {
                    self.shadows_v3 = Some(LocalShadowDirectPass::new(
                        gpu,
                        size,
                        MAX_LIGHTS,
                        MAX_DRAWS,
                        render::FRAMES_IN_FLIGHT as usize,
                    ));
                }
            }
        } else {
            match &mut self.shadows_v2 {
                Some(pass) => {
                    assert!(pass.resize(gpu, size));
                }
                None => {
                    self.shadows_v2 = Some(LocalShadowPass::new(
                        gpu,
                        size,
                        MAX_LIGHTS,
                        MAX_DRAWS,
                        render::FRAMES_IN_FLIGHT as usize,
                    ));
                }
            }
        }
        self.size = size;
    }
}

impl RenderScene for LabScene {
    fn draw(&mut self, ctx: &mut FrameCtx) {
        self.ensure_targets(ctx);
        let gpu = ctx.gpu;
        let cb = ctx.cb;
        let frame = ctx.frame;
        let slot = (frame % render::FRAMES_IN_FLIGHT) as usize;
        let s = script(self.scenario, frame, self.light_count);
        let view = MeshRasterView {
            world_to_clip: world_to_clip(s.eye, s.focus, self.size),
        };

        if let Some(timer) = self.timer.as_mut() {
            timer.collect(gpu, slot);
        }
        let mut stamps = self.timer.as_ref().map(|t| t.frame_begin(gpu, cb, slot));

        let scene = self.scene.as_mut().expect("scene exists");
        gpu.cmd_barrier(cb, Stage::All, Stage::Transfer, HazardFlags::empty());
        for (index, &instance) in self.movers.iter().enumerate() {
            scene.stage_world(
                gpu,
                cb,
                self.mover_staging[slot][index],
                instance,
                s.movers[index],
            );
        }
        gpu.cmd_barrier(cb, Stage::Transfer, Stage::All, HazardFlags::empty());

        let scene = self.scene.as_ref().expect("scene exists");
        let heap = self.heap.as_ref().expect("heap exists");
        let lights = &s.lights[..s.light_count as usize];
        let (lights_gpu, lights_cpu) = ctx.frame_alloc_slice::<PointLight>(s.light_count);
        // SAFETY: fresh arena run sized for light_count, initialized here.
        unsafe { std::ptr::copy_nonoverlapping(lights.as_ptr(), lights_cpu, lights.len()) };

        if let Some(v3) = &self.shadows_v3 {
            let c = v3.take_counters(slot);
            if self.stats {
                println!("v3stats,{frame},rays={}", c.requests_high);
            }
        }
        if !self.exact {
            if let Some(v2) = &self.shadows_v2 {
                let c = v2.take_counters(slot);
                assert_eq!(c.overflow, 0, "lab v2 shadow request queue overflowed");
                if self.stats {
                    println!(
                        "stats,{frame},{},{},{},{},{},{},{},{},{}",
                        c.active_texels,
                        c.requests_high + c.requests_low,
                        c.serviced_high + c.serviced_low,
                        c.validated,
                        c.reused,
                        c.invalidated,
                        c.contact,
                        c.promoted,
                        c.churn,
                    );
                }
            }
        }

        let cull = self.cull.as_mut().expect("cull exists");
        cull.record(gpu, cb, ctx, scene, scene.instances(), view, s.eye, slot);
        if let Some(st) = stamps.as_mut() {
            st.stamp(gpu, cb);
        }
        self.prepass.as_mut().expect("prepass exists").record(
            gpu,
            cb,
            ctx,
            scene,
            scene.instances(),
            cull.output(),
            cull.clusters(),
            cull.draw_count_ptr(gpu, slot),
            self.depth.as_ref().expect("depth exists").texture,
            self.size,
            view,
        );
        if let Some(st) = stamps.as_mut() {
            st.stamp(gpu, cb);
        }
        let surfaces = self.surfaces.as_ref().expect("surfaces ensured");
        self.forward
            .as_mut()
            .expect("forward exists")
            .record_with_surfaces(
                gpu,
                cb,
                ctx,
                heap,
                scene,
                scene.instances(),
                cull.output(),
                cull.clusters(),
                cull.draw_count_ptr(gpu, slot),
                MeshForwardTargets {
                    color: self.hdr.as_ref().expect("hdr exists").texture,
                    depth: self.depth.as_ref().expect("depth exists").texture,
                    size: self.size,
                    color_load_op: LoadOp::Clear,
                    clear_color: [
                        0.013 * self.ambient_scale,
                        0.013 * self.ambient_scale,
                        0.02 * self.ambient_scale,
                        1.0,
                    ],
                },
                view,
                s.eye,
                ambient(self.ambient_scale),
                self.ramp_default_sampler,
                MeshLightField::default(),
                surfaces.forward_targets(),
            );
        if let Some(st) = stamps.as_mut() {
            st.stamp(gpu, cb);
        }

        let (mask, slots) = if self.exact {
            let (mask, _) = self.shadows_exact.as_mut().expect("exact ensured").record(
                gpu,
                cb,
                ctx,
                heap,
                scene,
                scene.instances(),
                surfaces,
                view,
                lights_gpu,
                s.light_count,
                SHADOW_ORIGIN_BIAS,
                SHADOW_DESTINATION_BIAS,
                self.depth_slot,
            );
            (Some(mask), None)
        } else if self.v3 {
            let (slots, _) = self.shadows_v3.as_mut().expect("v3 ensured").record(
                gpu,
                cb,
                ctx,
                heap,
                scene,
                scene.instances(),
                surfaces,
                view,
                lights,
                SHADOW_ORIGIN_BIAS,
                SHADOW_DESTINATION_BIAS,
                0.0,
                env_f32("LAB_SOURCE_RADIUS", 0.25),
                NEAR_PLANE,
                self.depth_slot,
                slot,
            );
            (None, Some(slots))
        } else {
            let temporal = self.temporal();
            let (slots, _) = self.shadows_v2.as_mut().expect("v2 ensured").record(
                gpu,
                cb,
                ctx,
                heap,
                scene,
                scene.instances(),
                surfaces,
                view,
                lights,
                SHADOW_ORIGIN_BIAS,
                SHADOW_DESTINATION_BIAS,
                0.0,
                temporal,
                self.depth_slot,
                slot,
            );
            (None, Some(slots))
        };
        if let Some(st) = stamps.as_mut() {
            st.stamp(gpu, cb);
        }
        self.local_lights
            .as_ref()
            .expect("local lights exist")
            .record(
                gpu,
                cb,
                ctx,
                heap,
                scene,
                surfaces,
                view,
                lights_gpu,
                s.light_count,
                0.0,
                MeshLightField::default(),
                mask,
                slots,
                self.ramp_default_sampler,
                self.depth_slot,
                self.hdr_rw_slot,
            );
        if let Some(st) = stamps.as_mut() {
            st.stamp(gpu, cb);
        }

        let tonemap = self.tonemap.as_ref().expect("tonemap exists");
        tonemap.record(
            gpu,
            cb,
            ctx,
            ctx.backbuffer,
            self.hdr_slot,
            self.clamp_sampler,
            1.0,
            1.0 / 255.0,
            None,
        );
        if let Some(st) = stamps.as_mut() {
            st.stamp(gpu, cb);
        }
        drop(stamps);
        if let Some(timer) = self.timer.as_mut() {
            if let Some(report) = timer.report(ctx.time) {
                println!("timing,{},{}x{}", report.line, self.size.x, self.size.y);
            }
        }
        if env_u32("LAB_EXIT", 0) != 0 && frame >= u64::from(env_u32("LAB_EXIT", 0)) {
            if let Some(timer) = self.timer.as_mut() {
                if let Some(report) = timer.report(f32::MAX) {
                    println!("timing,{},{}x{}", report.line, self.size.x, self.size.y);
                }
            }
            ctx.request_exit();
        }

        if let Some(dump) = &self.dump {
            if frame >= dump.start && frame < dump.start + dump.count {
                let display = self.display.as_ref().expect("display exists").texture;
                tonemap.record(
                    gpu,
                    cb,
                    ctx,
                    display,
                    self.hdr_slot,
                    self.clamp_sampler,
                    1.0,
                    1.0 / 255.0,
                    None,
                );
            }
            if frame > dump.start && frame <= dump.start + dump.count {
                let display = self.display.as_ref().expect("display exists").texture;
                let path = dump.dir.join(format!("f{:05}.png", frame - 1));
                gpu::dump_texture_png(gpu, display, &path);
            }
            if frame == dump.start + dump.count {
                println!(
                    "light_lab: wrote {} frames to {}",
                    dump.count,
                    dump.dir.display()
                );
                ctx.request_exit();
            }
        }
    }

    fn teardown(&mut self, gpu: &gpu::Gpu) {
        if let Some(t) = self.hdr.take() {
            gpu.texture_free_and_destroy(t);
        }
        if let Some(t) = self.depth.take() {
            gpu.texture_free_and_destroy(t);
        }
        if let Some(t) = self.display.take() {
            gpu.texture_free_and_destroy(t);
        }
        if let Some(surfaces) = self.surfaces.take() {
            surfaces.free(gpu);
        }
        if let Some(pass) = self.shadows_v2.take() {
            pass.free(gpu);
        }
        if let Some(pass) = self.shadows_v3.take() {
            pass.free(gpu);
        }
        if let Some(pass) = self.shadows_exact.take() {
            pass.free(gpu);
        }
        if let Some(tonemap) = self.tonemap.take() {
            tonemap.free(gpu);
        }
        if let Some(scene) = self.scene.take() {
            scene.free(gpu);
        }
        if let Some(cull) = self.cull.take() {
            cull.free(gpu);
        }
        if let Some(prepass) = self.prepass.take() {
            prepass.free(gpu);
        }
        if let Some(forward) = self.forward.take() {
            forward.free(gpu);
        }
        if let Some(local_lights) = self.local_lights.take() {
            local_lights.free(gpu);
        }
        if let Some(heap) = self.heap.take() {
            heap.free(gpu);
        }
        for frame_staging in self.mover_staging {
            for staging in frame_staging {
                gpu.free(staging);
            }
        }
        if let Some(timer) = self.timer.take() {
            timer.free(gpu);
        }
    }
}

/// Builds the fixed example geometry and mover list.
fn build_scene(gpu: &gpu::Gpu, gamecam: bool) -> (MeshScene, Vec<InstanceHandle>) {
    let mut scene = MeshScene::new_with_shadows(
        gpu,
        &MeshSceneDesc {
            max_meshes: 2,
            max_instances: 8,
            max_materials: 8,
            vertex_capacity: 64,
            joint_weight_capacity: 0,
            index_capacity: 128,
            max_meshlets: 16,
        },
        ShadowBlasDesc {
            node_capacity: 64,
            primitive_capacity: 64,
        },
    );
    let cube = mesh::primitives::cube(0.5);
    let cube_mesh = scene.add_mesh(gpu, cube.desc());

    let material = |base: [f32; 4]| MaterialEntry {
        base_color_factor: base,
        ..Default::default()
    };
    let floor_mat = scene.add_material(gpu, material([0.62, 0.58, 0.52, 1.0]));
    let wall_mat = scene.add_material(gpu, material([0.55, 0.52, 0.5, 1.0]));
    let wood_mat = scene.add_material(gpu, material([0.45, 0.3, 0.18, 1.0]));
    let slab_mat = scene.add_material(gpu, material([0.35, 0.38, 0.42, 1.0]));
    let walker_mat = scene.add_material(gpu, material([0.3, 0.45, 0.3, 1.0]));

    let place = |scale: Vec3, pos: Vec3| {
        Mat4::from_scale_rotation_translation(scale, abi_core::glam::Quat::IDENTITY, pos)
    };
    scene.add_instance(
        gpu,
        cube_mesh,
        place(Vec3::new(40.0, 0.5, 40.0), Vec3::new(0.0, -0.25, 0.0)),
        floor_mat,
    );
    scene.add_instance(
        gpu,
        cube_mesh,
        place(Vec3::new(16.0, 5.0, 0.6), Vec3::new(0.0, 2.5, -8.0)),
        wall_mat,
    );
    scene.add_instance(
        gpu,
        cube_mesh,
        place(Vec3::new(0.45, 2.7, 0.45), Vec3::new(2.5, 1.35, 1.5)),
        wood_mat,
    );
    let slab = scene.add_instance(
        gpu,
        cube_mesh,
        place(Vec3::new(2.4, 0.15, 2.4), Vec3::new(-2.5, 1.6, 2.0)),
        slab_mat,
    );
    let walker = scene.add_instance(
        gpu,
        cube_mesh,
        place(Vec3::new(0.7, 1.1, 0.5), Vec3::new(-1.5, 0.55, 2.5)),
        walker_mat,
    );
    let mut movers = vec![walker];
    if gamecam {
        let tumbler = scene.add_instance(
            gpu,
            cube_mesh,
            place(Vec3::splat(0.6), Vec3::new(3.0, 1.0, -3.2)),
            walker_mat,
        );
        movers.push(slab);
        movers.push(tumbler);
    }
    (scene, movers)
}

fn main() {
    let (width, height) = std::env::var("LAB_SIZE").map_or(WINDOW_SIZE, |v| {
        let (w, h) = v.split_once('x').expect("LAB_SIZE must be WxH");
        (
            w.parse().expect("LAB_SIZE width"),
            h.parse().expect("LAB_SIZE height"),
        )
    });
    App::new()
        .add_plugins(
            DefaultPlugins.set(WindowPlugin {
                primary_window: Some(Window {
                    title: "asha light lab — v2 local shadows artifact bench".into(),
                    resolution: bevy::window::WindowResolution::new(width, height)
                        .with_scale_factor_override(1.0),
                    ..Default::default()
                }),
                ..Default::default()
            }),
        )
        .add_plugins(AshaRenderPlugin::new(LabScene::new))
        .add_plugins(PacingPlugin::default())
        .add_systems(Update, esc_to_exit)
        .run();
}
