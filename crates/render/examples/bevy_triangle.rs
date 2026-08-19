//! Minimal Bevy host demonstrating extraction and render-thread drawing.
//! `ASHA_VERIFY=1` checks the offscreen result before clean exit.

mod common;

use abi_core::TriangleData;
use asha_assets::load_spv;
use bevy::prelude::*;
use bevy::window::{Window, WindowPlugin};
use common::{VERIFY_CHECK_FRAME, VERIFY_COPY_FRAME, Verify, esc_to_exit};
use gpu::{Gpu, LoadOp, Memory, RenderAttachment, RenderPassDesc, ShaderTypeGraphics, StoreOp};
use render::{AshaRenderPlugin, FrameCtx, RenderScene};

/// ECS-owned render input extracted every frame as plain `Pod` data.
#[repr(C)]
#[derive(Component, Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct Tint {
    color: [f32; 4],
}

/// Animation curve shared by update and verification.
fn breathe(t: f32) -> f32 {
    0.75 + 0.25 * (t * 2.0).sin()
}

const VERIFY_W: u32 = 128;
const VERIFY_H: u32 = 128;

/// Triangle scene and optional offscreen verification state.
struct TriangleScene {
    vert: gpu::Shader,
    frag: gpu::Shader,
    positions: gpu::Ptr<[f32; 2]>,
    colors: gpu::Ptr<[f32; 4]>,
    white: gpu::Ptr<[f32; 4]>,
    indices: gpu::Ptr<u32>,
    verify: Option<Verify>,
    expected_tint: f32,
}

impl TriangleScene {
    fn new(gpu: &Gpu) -> Self {
        let vert = gpu.shader_create(
            &load_spv("triangle_vert"),
            ShaderTypeGraphics::Vertex,
            "triangle_vert",
        );
        let frag = gpu.shader_create(
            &load_spv("triangle_frag"),
            ShaderTypeGraphics::Fragment,
            "triangle_frag",
        );

        let positions = gpu.alloc_slice::<[f32; 2]>(3, Memory::Default);
        let colors = gpu.alloc_slice::<[f32; 4]>(3, Memory::Default);
        let white = gpu.alloc::<[f32; 4]>(Memory::Default);
        let indices = gpu.alloc_slice::<u32>(3, Memory::Default);
        unsafe {
            *positions.cpu.add(0) = [0.0, -0.6];
            *positions.cpu.add(1) = [-0.55, 0.5];
            *positions.cpu.add(2) = [0.55, 0.5];
            *colors.cpu.add(0) = [1.0, 0.2, 0.2, 1.0];
            *colors.cpu.add(1) = [0.2, 0.4, 1.0, 1.0];
            *colors.cpu.add(2) = [0.2, 1.0, 0.2, 1.0];
            *white.cpu = [1.0, 1.0, 1.0, 1.0];
            for i in 0..3 {
                *indices.cpu.add(i) = i as u32;
            }
        }

        Self {
            vert,
            frag,
            positions,
            colors,
            white,
            indices,
            verify: Verify::from_env(gpu, VERIFY_W, VERIFY_H),
            expected_tint: 0.0,
        }
    }

    /// Records using the extracted tint pointer in the frame arena.
    fn record_pass(&self, ctx: &mut FrameCtx, target: gpu::Texture, clear: [f32; 4]) {
        let (tints, tint_count) = ctx.extracted::<Tint>();
        let tint = if tint_count > 0 {
            tints.cast()
        } else {
            self.white.gpu
        };
        let tri = ctx.frame_alloc(TriangleData {
            positions: self.positions.gpu,
            colors: self.colors.gpu,
            tint,
        });

        ctx.gpu.cmd_begin_render_pass(
            ctx.cb,
            RenderPassDesc {
                color_attachments: &[RenderAttachment {
                    texture: target,
                    load_op: LoadOp::Clear,
                    store_op: StoreOp::Store,
                    clear_color: clear,
                    ..Default::default()
                }],
                ..Default::default()
            },
        );
        ctx.gpu.cmd_set_shaders(ctx.cb, self.vert, self.frag);
        ctx.gpu.cmd_draw_indexed_instanced(
            ctx.cb,
            tri.cast(),
            tri.cast(),
            self.indices.cast(),
            3,
            1,
        );
        ctx.gpu.cmd_end_render_pass(ctx.cb);
    }

    fn check_verify_pixels(&self, ctx: &mut FrameCtx) {
        let v = self.verify.as_ref().unwrap();
        let center = v.pixel(VERIFY_W / 2, (VERIFY_H as f32 * 0.5667) as u32);
        let expect = [
            (255.0 * (1.4 / 3.0) * self.expected_tint) as i32,
            (255.0 * (1.6 / 3.0) * self.expected_tint) as i32,
            (255.0 * (1.4 / 3.0) * self.expected_tint) as i32,
        ];
        for c in 0..3 {
            let got = center[c] as i32;
            assert!(
                (got - expect[c]).abs() <= 8,
                "verify: channel {c} = {got}, expected ~{} (tint {}) — extract or draw broken",
                expect[c],
                self.expected_tint,
            );
        }
        assert_eq!(
            v.pixel(2, 2),
            [0, 0, 0, 0],
            "verify: corner should be clear"
        );
        println!(
            "VERIFY OK center={center:?} tint={} frame={}",
            self.expected_tint, ctx.frame
        );
        ctx.request_exit();
    }
}

impl RenderScene for TriangleScene {
    fn draw(&mut self, ctx: &mut FrameCtx) {
        if self.verify.is_some() {
            if ctx.frame == VERIFY_COPY_FRAME {
                let (_, tint_count) = ctx.extracted::<Tint>();
                assert!(
                    tint_count > 0,
                    "BSN tint must have spawned by the verify frame"
                );
                self.expected_tint = breathe(ctx.time);
                let target = self.verify.as_ref().unwrap().target.texture;
                self.record_pass(ctx, target, [0.0, 0.0, 0.0, 0.0]);
                self.verify.as_ref().unwrap().copy_frame(ctx);
            } else if ctx.frame == VERIFY_CHECK_FRAME {
                self.check_verify_pixels(ctx);
            }
        }

        let t = ctx.time;
        let clear = [
            0.03 + 0.02 * (t * 0.7).sin().abs(),
            0.03,
            0.06 + 0.03 * (t * 0.4).cos().abs(),
            1.0,
        ];
        self.record_pass(ctx, ctx.backbuffer, clear);
    }

    fn teardown(&mut self, gpu: &Gpu) {
        if let Some(v) = self.verify.take() {
            v.teardown(gpu);
        }
        gpu.shader_destroy(self.vert);
        gpu.shader_destroy(self.frag);
        gpu.free(self.positions);
        gpu.free(self.colors);
        gpu.free(self.white);
        gpu.free(self.indices);
    }
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "asha × bevy — first pixels, second engine, third thread".into(),
                resolution: bevy::window::WindowResolution::new(960, 720),
                ..Default::default()
            }),
            ..Default::default()
        }))
        .add_plugins(AshaRenderPlugin::new(TriangleScene::new).extract::<Tint>())
        .add_systems(Startup, spawn_scene)
        .add_systems(Update, (esc_to_exit, breathe_tint))
        .run();
}

fn spawn_scene(mut commands: Commands) {
    commands.spawn_scene(bsn! {
        Tint { color: {[1.0, 1.0, 1.0, 1.0]} }
    });
}

/// Animates the ECS tint.
fn breathe_tint(time: Res<Time>, mut tints: Query<&mut Tint>) {
    let s = breathe(time.elapsed_secs());
    for mut tint in &mut tints {
        tint.color = [s, s, s, 1.0];
    }
}
