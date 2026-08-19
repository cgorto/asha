//! Integration test for shaped text extraction and clipped rendering.
//!
//! Verifies glyph extraction, painter ordering, scissoring, and GPU coverage.

use bevy::asset::Assets;
use bevy::math::Rect;
use bevy::prelude::*;
use bevy::text::{Font as BevyFont, FontSize, TextColor, TextFont};
use bevy::ui::{Overflow, PositionType, Val};
use bevy::window::{Window, WindowPlugin};

use gpu::{
    Gpu, HazardFlags, LoadOp, Memory, OwnedTexture, Queue, Stage, StoreOp, TextureDesc,
    TextureFormat, UsageFlags,
};
use render::{AshaRenderPlugin, FrameCtx, RenderScene};
use text::{TextGlyphInstance, TextPass, TextPassTarget};
use ui_bridge::{AshaRenderPluginExt, TextRunBatch, UiBatch, UiBridge, UiBridgePlugin};

const W: u32 = 240;
const H: u32 = 140;
const CLEAR: [f32; 4] = [0.02, 0.02, 0.02, 1.0];

const CONTAINER_POS: (f32, f32) = (10.0, 20.0);
const CONTAINER_SIZE: (f32, f32) = (50.0, 40.0);
const FONT_PX: f32 = 28.0;

const MARKER_POS: (f32, f32) = (150.0, 20.0);
const MARKER_SIZE: (f32, f32) = (40.0, 40.0);

const FIRA_SANS_REGULAR: &[u8] =
    include_bytes!("../../widgets/src/assets/fonts/FiraSans-Regular.ttf");

fn container_rect() -> Rect {
    Rect::new(
        CONTAINER_POS.0,
        CONTAINER_POS.1,
        CONTAINER_POS.0 + CONTAINER_SIZE.0,
        CONTAINER_POS.1 + CONTAINER_SIZE.1,
    )
}

fn spawn_scene(mut commands: Commands, mut fonts: ResMut<Assets<BevyFont>>) {
    let font = fonts.add(BevyFont::from_bytes(FIRA_SANS_REGULAR.to_vec()));

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
                    left: Val::Px(CONTAINER_POS.0),
                    top: Val::Px(CONTAINER_POS.1),
                    width: Val::Px(CONTAINER_SIZE.0),
                    height: Val::Px(CONTAINER_SIZE.1),
                    overflow: Overflow::clip(),
                    ..Default::default()
                },
                BackgroundColor(Color::srgba(0.2, 0.2, 0.25, 1.0)),
            ))
            .with_children(|container| {
                container.spawn((
                    Text::new("Ashá"),
                    TextFont {
                        font: font.into(),
                        font_size: FontSize::Px(FONT_PX),
                        ..Default::default()
                    },
                    TextColor(Color::WHITE),
                ));
            });

            root.spawn((
                Node {
                    position_type: PositionType::Absolute,
                    left: Val::Px(MARKER_POS.0),
                    top: Val::Px(MARKER_POS.1),
                    width: Val::Px(MARKER_SIZE.0),
                    height: Val::Px(MARKER_SIZE.1),
                    ..Default::default()
                },
                BackgroundColor(Color::srgba(0.8, 0.2, 0.2, 1.0)),
            ));
        });
}

struct TextExtractCheckScene {
    bridge: UiBridge,
    pass: Option<TextPass>,
    target: OwnedTexture,
    readback: gpu::Ptr<[f32; 4]>,
}

impl TextExtractCheckScene {
    fn new(gpu: &Gpu) -> Self {
        let pass = TextPass::new(gpu);
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
            target,
            readback,
        }
    }

    fn check(&mut self, ctx: &mut FrameCtx) {
        let glyph_instances = ctx.extracted_host_mut::<TextGlyphInstance>().to_vec();
        let text_batches = ctx.extracted_host::<TextRunBatch>().to_vec();
        let ui_batches = ctx.extracted_host::<UiBatch>().to_vec();

        assert!(
            !text_batches.is_empty(),
            "expected at least one TextRunBatch"
        );
        assert!(
            !glyph_instances.is_empty(),
            "expected at least one glyph instance ('A', 's', 'h', composite 'á')"
        );

        let run = &text_batches[0];
        assert!(
            run.instance_range.len() >= 3,
            "\"Ashá\" should shape to at least 3 glyphs (composite 'á' may or may not fuse with \
             a preceding one depending on shaping), got {}",
            run.instance_range.len()
        );

        // Shaping should advance left-to-right within the container.
        let run_instances = &glyph_instances[run.instance_range.clone()];
        let mut prev_x = f32::NEG_INFINITY;
        for inst in run_instances {
            assert!(
                inst.pen_doc[0] + 0.01 >= prev_x,
                "glyph pen x should be non-decreasing across a left-to-right run: {:?}",
                run_instances.iter().map(|i| i.pen_doc).collect::<Vec<_>>()
            );
            prev_x = inst.pen_doc[0];
            assert!(
                inst.pen_doc[1] > CONTAINER_POS.1
                    && inst.pen_doc[1] < CONTAINER_POS.1 + CONTAINER_SIZE.1 + FONT_PX,
                "glyph baseline y {} should land within/near the container's vertical band",
                inst.pen_doc[1]
            );
        }

        // The text batch must carry the container clip.
        let want_clip = container_rect();
        let got_clip = run
            .clip
            .expect("text node is inside an overflow:clip container");
        let clip_close = (got_clip.min.x - want_clip.min.x).abs() < 1.0
            && (got_clip.min.y - want_clip.min.y).abs() < 1.0
            && (got_clip.max.x - want_clip.max.x).abs() < 1.0
            && (got_clip.max.y - want_clip.max.y).abs() < 1.0;
        assert!(
            clip_close,
            "text batch clip {got_clip:?} should match the container's screen rect {want_clip:?}"
        );

        // Painter order must place text between container and marker.
        assert!(
            ui_batches.len() >= 2,
            "expected the quad batch to split around the interleaved text run, got {} UiBatch(es): {ui_batches:?}",
            ui_batches.len()
        );
        let marker_batch_order = ui_batches
            .iter()
            .max_by_key(|b| b.order)
            .expect("at least one UiBatch")
            .order;
        assert!(
            marker_batch_order > run.order,
            "the marker's UiBatch (order {marker_batch_order}) should paint AFTER the text run \
             (order {}) — it was spawned later and should out-rank it in stack order",
            run.order
        );
        let container_batch_order = ui_batches
            .iter()
            .min_by_key(|b| b.order)
            .expect("at least one UiBatch")
            .order;
        assert!(
            container_batch_order < run.order,
            "the container's own background (order {container_batch_order}) should paint \
             BEFORE the text run (order {})",
            run.order
        );

        println!(
            "text_extract golden OK: {} glyph(s), clip={got_clip:?}, order {container_batch_order} < {} < {marker_batch_order}",
            run_instances.len(),
            run.order
        );

        self.bridge.ingest_text(ctx, [W, H]);
        let batches = self.bridge.text_batches().to_vec();
        assert_eq!(batches.len(), text_batches.len());

        let gpu = ctx.gpu;
        let cb = gpu.commands_begin(Queue::Main);
        self.pass.as_ref().unwrap().record_batches(
            gpu,
            cb,
            TextPassTarget {
                load_op: LoadOp::Clear,
                store_op: StoreOp::Store,
                clear_color: CLEAR,
                ..TextPassTarget::overlay(self.target.texture, [W, H])
            },
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
        let coverage_at = |x: u32, y: u32| -> f32 {
            let p = pixels[(y * W + x) as usize];
            // Coverage is brightness above the clear color.
            (p[0] - CLEAR[0]).max(p[1] - CLEAR[1]).max(p[2] - CLEAR[2])
        };

        // A point near the first glyph should be covered.
        let first_pen = run_instances[0].pen_doc;
        let inside_x = (first_pen[0] as u32 + 3).min(W - 1);
        let inside_y = (first_pen[1] as u32)
            .saturating_sub(FONT_PX as u32 / 3)
            .min(H - 1);
        let inside_cov = coverage_at(inside_x, inside_y);

        // Overflow beyond the container must be clipped.
        let outside_x = (want_clip.max.x as u32 + 10).min(W - 1);
        let outside_y = inside_y;
        let outside_cov = coverage_at(outside_x, outside_y);

        println!(
            "text_extract pixels: inside({inside_x},{inside_y})={inside_cov:.3} \
             outside({outside_x},{outside_y})={outside_cov:.3}"
        );
        assert!(
            inside_cov > 0.05,
            "expected visible glyph coverage inside the clip rect near the first glyph, got {inside_cov}"
        );
        assert!(
            outside_cov < 0.02,
            "expected ZERO coverage outside the scissored container (proves the per-batch \
             scissor clipped the overflowing glyph), got {outside_cov}"
        );

        println!("text_extract GPU check OK frame={}", ctx.frame);
        ctx.request_exit();
    }
}

impl RenderScene for TextExtractCheckScene {
    fn draw(&mut self, ctx: &mut FrameCtx) {
        // Allow font loading, shaping, and clipping to settle.
        if ctx.frame == 10 {
            self.check(ctx);
        }
    }

    fn teardown(&mut self, gpu: &Gpu) {
        if let Some(pass) = self.pass.take() {
            pass.free(gpu);
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
                title: "ui-bridge text extraction".into(),
                // Fixed scale keeps physical probe coordinates stable.
                resolution:
                    bevy::window::WindowResolution::new(W, H).with_scale_factor_override(1.0),
                visible: false,
                ..Default::default()
            }),
            ..Default::default()
        }))
        .add_plugins(UiBridgePlugin)
        .add_plugins(
            AshaRenderPlugin::new(TextExtractCheckScene::new)
                .extract_ui()
                .extract_text(),
        )
        .add_systems(Startup, spawn_scene)
        .run();
}
