//! Integration test for icon decoding, upload, and sampling.
//!
//! Verifies registered alpha sampling and unregistered tint fallback.

use bevy::prelude::*;
use bevy::ui::{Node, PositionType, Val};
use bevy::window::{Window, WindowPlugin};

use gpu::{
    Gpu, HazardFlags, Memory, OwnedTexture, Queue, Stage, TextureDesc, TextureFormat, UsageFlags,
};
use render::{AshaRenderPlugin, FrameCtx, RenderScene};
use ui_bridge::{AshaRenderPluginExt, ICON_PATHS, UiBridge, UiBridgePlugin};

const W: u32 = 64;
const H: u32 = 32;
const CLEAR: [f32; 4] = [0.02, 0.03, 0.02, 1.0];

/// Registered icon position aligned to its 24x24 texel grid.
const ICON_A_POS: (f32, f32) = (4.0, 4.0);
const ICON_SIZE: f32 = 24.0;

/// Unregistered icon position, separate from the registered icon.
const ICON_B_POS: (f32, f32) = (36.0, 4.0);

const RED: Color = Color::srgb(1.0, 0.0, 0.0);
const GREEN: Color = Color::srgb(0.0, 1.0, 0.0);

/// Known opaque chevron texel.
const OPAQUE_TEXEL: (u32, u32) = (11, 14);
/// Known transparent texel inside the icon bounds.
const TRANSPARENT_TEXEL: (u32, u32) = (10, 2);

fn spawn_scene(mut commands: Commands, asset_server: Res<AssetServer>) {
    // Match the registry's first icon path.
    let chevron_down: Handle<Image> = asset_server.load(ICON_PATHS[0]);

    commands.spawn(Camera2d);

    commands
        .spawn(Node {
            width: Val::Px(W as f32),
            height: Val::Px(H as f32),
            ..Default::default()
        })
        .with_children(|root| {
            root.spawn((
                Node {
                    position_type: PositionType::Absolute,
                    left: Val::Px(ICON_A_POS.0),
                    top: Val::Px(ICON_A_POS.1),
                    width: Val::Px(ICON_SIZE),
                    height: Val::Px(ICON_SIZE),
                    ..Default::default()
                },
                ImageNode {
                    image: chevron_down,
                    color: RED,
                    ..Default::default()
                },
            ));

            root.spawn((
                Node {
                    position_type: PositionType::Absolute,
                    left: Val::Px(ICON_B_POS.0),
                    top: Val::Px(ICON_B_POS.1),
                    width: Val::Px(ICON_SIZE),
                    height: Val::Px(ICON_SIZE),
                    ..Default::default()
                },
                ImageNode {
                    color: GREEN,
                    ..Default::default()
                },
            ));
        });
}

struct IconExtractCheckScene {
    bridge: UiBridge,
    pass: Option<ui::UiPass>,
    heap: Option<gpu::HeapSlots>,
    target: OwnedTexture,
    readback: gpu::Ptr<[f32; 4]>,
}

impl IconExtractCheckScene {
    fn new(gpu: &Gpu) -> Self {
        let pass = ui::UiPass::new(gpu);
        let heap = gpu.heap_slots_create(8, 2, 4);
        let target = gpu.texture_alloc_and_create(
            TextureDesc {
                dimensions: [W, H, 1],
                format: TextureFormat::Rgba32Float,
                usage: UsageFlags::COLOR_ATTACHMENT | UsageFlags::TRANSFER_SRC,
                ..Default::default()
            },
            Queue::Main,
            None,
        );
        let readback = gpu.alloc_slice::<[f32; 4]>((W * H) as u64, Memory::Readback);

        Self {
            bridge: UiBridge::new(),
            pass: Some(pass),
            heap: Some(heap),
            target,
            readback,
        }
    }

    /// Uploads pending icons and reports whether slot one is ready.
    fn ingest_icons(&mut self, ctx: &mut FrameCtx) -> bool {
        let gpu = ctx.gpu;
        let heap = self.heap.as_mut().expect("heap present");
        self.bridge.ingest_icons(gpu, heap, ctx);
        self.bridge.icon_ready(1)
    }

    fn check(&mut self, ctx: &mut FrameCtx) {
        let gpu = ctx.gpu;
        let heap = self.heap.as_mut().expect("heap present");

        self.bridge.ingest(ctx, [W, H]);
        let batches = self.bridge.batches().to_vec();
        assert_eq!(
            batches.len(),
            1,
            "one UiPaintList batch spanning both quads"
        );

        let cb = gpu.commands_begin(Queue::Main);
        heap.bind(gpu, cb);
        self.pass.as_ref().unwrap().record(
            gpu,
            cb,
            ui::UiPassTarget::clear(self.target.texture, [W, H], CLEAR),
            &batches,
        );
        gpu.cmd_barrier(cb, Stage::All, Stage::Transfer, HazardFlags::empty());
        gpu.cmd_copy_texture_to_buffer(cb, self.target.texture, self.readback.cast());
        gpu.queue_submit(Queue::Main, &[cb]);
        gpu.queue_wait_idle(Queue::Main);

        let mut pixels = vec![[0.0f32; 4]; (W * H) as usize];
        // SAFETY: readback size matches the completed texture copy.
        unsafe {
            std::ptr::copy_nonoverlapping(self.readback.cpu, pixels.as_mut_ptr(), pixels.len());
        }
        let pixel_at = |x: u32, y: u32| pixels[(y * W + x) as usize];

        let (ox, oy) = OPAQUE_TEXEL;
        let opaque = pixel_at(ICON_A_POS.0 as u32 + ox, ICON_A_POS.1 as u32 + oy);
        println!("icon_extract opaque ink pixel = {opaque:?}");
        assert!(
            opaque[0] > 0.5 && opaque[0] > opaque[1] + 0.2 && opaque[0] > opaque[2] + 0.2,
            "opaque ink texel should read red-dominant (real texture sample x red tint): {opaque:?}"
        );

        let (tx, ty) = TRANSPARENT_TEXEL;
        let transparent = pixel_at(ICON_A_POS.0 as u32 + tx, ICON_A_POS.1 as u32 + ty);
        println!("icon_extract transparent-in-bbox pixel = {transparent:?}");
        assert!(
            (transparent[0] - CLEAR[0]).abs() < 0.05
                && (transparent[1] - CLEAR[1]).abs() < 0.05
                && (transparent[2] - CLEAR[2]).abs() < 0.05,
            "a fully transparent PNG texel, even inside the node's bounding box, should show \
             background through it (real alpha-channel sampling, not just node-rect clipping): \
             {transparent:?} vs clear {CLEAR:?}"
        );

        let far_outside = pixel_at(0, H - 1);
        assert!(
            (far_outside[0] - CLEAR[0]).abs() < 1e-3
                && (far_outside[1] - CLEAR[1]).abs() < 1e-3
                && (far_outside[2] - CLEAR[2]).abs() < 1e-3,
            "outside both icon nodes entirely should be untouched background: {far_outside:?}"
        );

        // The fallback must fill the node, not sample icon shape.
        let flat = pixel_at(ICON_B_POS.0 as u32 + tx, ICON_B_POS.1 as u32 + ty);
        println!("icon_extract unregistered-asset flat-fill pixel = {flat:?}");
        assert!(
            flat[1] > 0.5 && flat[1] > flat[0] + 0.2 && flat[1] > flat[2] + 0.2,
            "unregistered asset should paint a flat, fully-covering green tint rect \
             (untextured ZII fallback): {flat:?}"
        );
        let b_outside = pixel_at(
            ICON_B_POS.0 as u32 + ICON_SIZE as u32 + 4,
            ICON_B_POS.1 as u32,
        );
        assert!(
            (b_outside[0] - CLEAR[0]).abs() < 1e-3 && (b_outside[1] - CLEAR[1]).abs() < 1e-3,
            "just past icon B's right edge should be untouched background: {b_outside:?}"
        );

        println!("icon_extract GPU check OK frame={}", ctx.frame);
        ctx.request_exit();
    }
}

const GIVE_UP_FRAME: u64 = 120;

impl RenderScene for IconExtractCheckScene {
    fn draw(&mut self, ctx: &mut FrameCtx) {
        // Consume the one-frame queue every frame.
        let ready = self.ingest_icons(ctx);
        if ready {
            self.check(ctx);
        } else {
            assert!(
                ctx.frame < GIVE_UP_FRAME,
                "chevron-down (logical slot 1) never finished loading/uploading by frame {} — \
                 either the embedded asset path is wrong (see ui_bridge::icons' module doc) or the \
                 seam never crossed/uploaded it in time",
                ctx.frame
            );
        }
    }

    fn teardown(&mut self, gpu: &Gpu) {
        if let Some(pass) = self.pass.take() {
            pass.free(gpu);
        }
        // SAFETY: textures must outlive their registered heap slots.
        std::mem::take(&mut self.bridge).free(gpu);
        if let Some(heap) = self.heap.take() {
            heap.free(gpu);
        }
        gpu.free(self.readback);
        gpu.texture_free_and_destroy(self.target);
    }
}

/// Runs without libtest because winit requires the main thread.
fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "ui-bridge icon extraction".into(),
                // Fixed scale keeps physical probe coordinates stable.
                resolution:
                    bevy::window::WindowResolution::new(W, H).with_scale_factor_override(1.0),
                visible: false,
                ..Default::default()
            }),
            ..Default::default()
        }))
        .add_plugins(widgets::FeathersCorePlugin)
        .add_plugins(UiBridgePlugin)
        .add_plugins(
            AshaRenderPlugin::new(IconExtractCheckScene::new)
                .extract_ui()
                .extract_icons(),
        )
        .add_systems(Startup, spawn_scene)
        .run();
}
