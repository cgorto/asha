//! Integration test for UI extraction and GPU rendering.
//!
//! Verifies byte-preserved streams, per-batch offsets, scissors, and shadows.

use bevy::math::{IRect, IVec2};
use bevy::prelude::*;
use bevy::window::{Window, WindowPlugin};

use abi_ui::{UiShadowVertex, UiVertex};
use gpu::{
    Gpu, HazardFlags, Memory, OwnedTexture, Queue, SamplerDesc, Stage, TextureDesc, TextureFormat,
    UsageFlags,
};
use render::{AshaRenderPlugin, FrameCtx, RenderScene};
use ui_bridge::{AshaRenderPluginExt, UiBridge, UiPaintList};

const SIZE: u32 = 64;
const RED: [f32; 4] = [1.0, 0.0, 0.0, 1.0];
const BLUE: [f32; 4] = [0.0, 0.0, 1.0, 1.0];
const BG: [f32; 4] = [0.02, 0.02, 0.02, 1.0];

// Separate target isolates shadow pixel assertions.
const SHADOW_SIZE: u32 = 32;
const SHADOW_COLOR: [f32; 4] = [0.2, 0.8, 0.3, 1.0];

/// Builds TL, TR, BR, BL vertices for a solid quad.
fn solid_quad(rect: [f32; 4], color: [f32; 4]) -> [UiVertex; 4] {
    let [x, y, w, h] = rect;
    let positions = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]];
    let points = [
        [-w / 2.0, -h / 2.0],
        [w / 2.0, -h / 2.0],
        [w / 2.0, h / 2.0],
        [-w / 2.0, h / 2.0],
    ];
    core::array::from_fn(|i| UiVertex {
        pos: positions[i],
        uv: [0.0, 0.0],
        color,
        color2: [0.0; 4],
        radius: [0.0; 4],
        border: [0.0; 4],
        size: [w, h],
        point: points[i],
        flags: 0,
        tex_slot: 0,
    })
}

/// Builds red and blue halves with a batch scissor.
fn fixture_vertices() -> Vec<UiVertex> {
    let half = SIZE as f32 / 2.0;
    let mut v = Vec::with_capacity(8);
    v.extend_from_slice(&solid_quad([0.0, 0.0, half, SIZE as f32], RED));
    v.extend_from_slice(&solid_quad([half, 0.0, half, SIZE as f32], BLUE));
    v
}

/// Builds a shadow whose center has near-full coverage.
fn fixture_shadow_vertices() -> [UiShadowVertex; 4] {
    let size = [SHADOW_SIZE as f32 * 4.0, SHADOW_SIZE as f32 * 4.0];
    let blur = 1.0f32;
    let bounds = [size[0] + 6.0 * blur, size[1] + 6.0 * blur];
    let center = [SHADOW_SIZE as f32 / 2.0, SHADOW_SIZE as f32 / 2.0];
    let corners = [[-0.5f32, -0.5f32], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]];
    core::array::from_fn(|i| {
        let [cx, cy] = corners[i];
        UiShadowVertex {
            pos: [center[0] + cx * bounds[0], center[1] + cy * bounds[1]],
            uv: [cx + 0.5, cy + 0.5],
            color: SHADOW_COLOR,
            size,
            radius: [0.0; 4],
            blur,
            bounds,
        }
    })
}

fn fixture_shadow_batches() -> Vec<ui_bridge::UiShadowBatch> {
    vec![ui_bridge::UiShadowBatch {
        quad_range: 0..1,
        scissor: None,
        order: 0,
    }]
}

fn fixture_batches() -> Vec<ui_bridge::UiBatch> {
    let half = SIZE as i32 / 2;
    vec![
        ui_bridge::UiBatch {
            quad_range: 0..1,
            scissor: None,
            order: 0,
        },
        ui_bridge::UiBatch {
            quad_range: 1..2,
            scissor: Some(IRect {
                min: IVec2::new(half, 0),
                max: IVec2::new(SIZE as i32, SIZE as i32),
            }),
            order: 1,
        },
    ]
}

/// Render-thread fixture for extraction and pixel assertions.
struct UiExtractCheckScene {
    expected_vertices: Vec<UiVertex>,
    expected_batches: Vec<ui_bridge::UiBatch>,
    expected_shadow_vertices: [UiShadowVertex; 4],
    expected_shadow_batches: Vec<ui_bridge::UiShadowBatch>,
    bridge: UiBridge,
    pass: Option<ui::UiPass>,
    heap: Option<gpu::HeapSlots>,
    target: OwnedTexture,
    readback: gpu::Ptr<[f32; 4]>,
    shadow_target: OwnedTexture,
    shadow_readback: gpu::Ptr<[f32; 4]>,
}

impl UiExtractCheckScene {
    fn new(gpu: &Gpu) -> Self {
        let pass = ui::UiPass::new(gpu);
        let mut heap = gpu.heap_slots_create(2, 2, 2);
        // RuntimeArray descriptors must remain valid for untextured draws.
        let _sampler = heap.add_sampler(gpu, gpu.sampler_descriptor(SamplerDesc::default()));
        let target = gpu.texture_alloc_and_create(
            TextureDesc {
                dimensions: [SIZE, SIZE, 1],
                format: TextureFormat::Rgba32Float,
                usage: UsageFlags::COLOR_ATTACHMENT | UsageFlags::TRANSFER_SRC,
                ..Default::default()
            },
            Queue::Main,
            None,
        );
        let readback = gpu.alloc_slice::<[f32; 4]>((SIZE * SIZE) as u64, Memory::Readback);

        let shadow_target = gpu.texture_alloc_and_create(
            TextureDesc {
                dimensions: [SHADOW_SIZE, SHADOW_SIZE, 1],
                format: TextureFormat::Rgba32Float,
                usage: UsageFlags::COLOR_ATTACHMENT | UsageFlags::TRANSFER_SRC,
                ..Default::default()
            },
            Queue::Main,
            None,
        );
        let shadow_readback =
            gpu.alloc_slice::<[f32; 4]>((SHADOW_SIZE * SHADOW_SIZE) as u64, Memory::Readback);

        Self {
            expected_vertices: fixture_vertices(),
            expected_batches: fixture_batches(),
            expected_shadow_vertices: fixture_shadow_vertices(),
            expected_shadow_batches: fixture_shadow_batches(),
            bridge: UiBridge::new(),
            pass: Some(pass),
            heap: Some(heap),
            target,
            readback,
            shadow_target,
            shadow_readback,
        }
    }

    fn check(&mut self, ctx: &mut FrameCtx) {
        // Verify extracted vertex bytes.
        let extracted = ctx.extracted_host_mut::<UiVertex>();
        assert_eq!(
            extracted.len(),
            self.expected_vertices.len(),
            "extracted UiVertex count diverged from UiPaintList::vertices"
        );
        assert_eq!(
            bytemuck::cast_slice::<UiVertex, u8>(extracted),
            bytemuck::cast_slice::<UiVertex, u8>(&self.expected_vertices),
            "extracted UiVertex bytes diverged from UiPaintList::vertices"
        );

        // Verify extracted batch descriptors.
        let extracted_batches = ctx.extracted_host::<ui_bridge::UiBatch>();
        assert_eq!(
            extracted_batches,
            self.expected_batches.as_slice(),
            "extracted UiBatch descriptors diverged from UiPaintList::batches"
        );

        // Verify one draw record per batch.
        self.bridge.ingest(ctx, [SIZE, SIZE]);
        let batches = self.bridge.batches().to_vec();
        assert_eq!(
            batches.len(),
            self.expected_batches.len(),
            "batch count diverged"
        );
        for (got, want) in batches.iter().zip(&self.expected_batches) {
            assert!(
                !got.draw.is_null(),
                "batch has quads but UiDraw pointer is null"
            );
            assert_eq!(
                got.quad_count,
                u32::try_from(want.quad_range.len()).unwrap(),
                "batch quad_count diverged"
            );
            let want_scissor = want.scissor.map(|r| ui::UiScissor {
                offset: [r.min.x, r.min.y],
                extent: [(r.max.x - r.min.x) as u32, (r.max.y - r.min.y) as u32],
            });
            assert_eq!(
                got.scissor, want_scissor,
                "batch scissor conversion diverged"
            );
        }

        // Draw both batches and verify their physical offsets.
        let gpu = ctx.gpu;
        let cb = gpu.commands_begin(Queue::Main);
        self.heap.as_ref().unwrap().bind(gpu, cb);
        self.pass.as_ref().unwrap().record(
            gpu,
            cb,
            ui::UiPassTarget::clear(self.target.texture, [SIZE, SIZE], BG),
            &batches,
        );
        gpu.cmd_barrier(cb, Stage::All, Stage::Transfer, HazardFlags::empty());
        gpu.cmd_copy_texture_to_buffer(cb, self.target.texture, self.readback.cast());
        gpu.queue_submit(Queue::Main, &[cb]);
        gpu.queue_wait_idle(Queue::Main);

        let mut pixels = vec![[0.0f32; 4]; (SIZE * SIZE) as usize];
        // SAFETY: readback size matches the completed texture copy.
        unsafe {
            std::ptr::copy_nonoverlapping(self.readback.cpu, pixels.as_mut_ptr(), pixels.len());
        }
        let pixel_at = |x: u32, y: u32| pixels[(y * SIZE + x) as usize];

        let left = pixel_at(SIZE / 4, SIZE / 2);
        assert!(
            (left[0] - RED[0]).abs() < 1e-3 && (left[2] - RED[2]).abs() < 1e-3,
            "left half (batch 0) should be red: {left:?}"
        );
        let right = pixel_at(SIZE * 3 / 4, SIZE / 2);
        assert!(
            (right[2] - BLUE[2]).abs() < 1e-3 && (right[0] - BLUE[0]).abs() < 1e-3,
            "right half (batch 1) should be blue — if UiBridge mis-offset batch 1's \
             vertices (e.g. reused batch 0's), this would read background instead: {right:?}"
        );

        // Verify shadow descriptors, pointers, and GPU output.
        let extracted_shadow = ctx.extracted_host_mut::<UiShadowVertex>();
        assert_eq!(
            extracted_shadow.len(),
            self.expected_shadow_vertices.len(),
            "extracted UiShadowVertex count diverged from UiPaintList::shadow_vertices"
        );
        assert_eq!(
            bytemuck::cast_slice::<UiShadowVertex, u8>(extracted_shadow),
            bytemuck::cast_slice::<UiShadowVertex, u8>(&self.expected_shadow_vertices),
            "extracted UiShadowVertex bytes diverged from UiPaintList::shadow_vertices"
        );

        let extracted_shadow_batches = ctx.extracted_host::<ui_bridge::UiShadowBatch>();
        assert_eq!(
            extracted_shadow_batches,
            self.expected_shadow_batches.as_slice(),
            "extracted UiShadowBatch descriptors diverged from UiPaintList::shadow_batches"
        );

        let shadow_batches = self.bridge.shadow_batches().to_vec();
        assert_eq!(
            shadow_batches.len(),
            self.expected_shadow_batches.len(),
            "shadow batch count diverged"
        );
        assert!(
            !shadow_batches[0].draw.is_null(),
            "shadow batch has quads but UiShadowDraw pointer is null"
        );
        assert_eq!(shadow_batches[0].quad_count, 1);

        let shadow_cb = gpu.commands_begin(Queue::Main);
        self.heap.as_ref().unwrap().bind(gpu, shadow_cb);
        self.pass.as_ref().unwrap().record_shadows(
            gpu,
            shadow_cb,
            ui::UiPassTarget::clear(self.shadow_target.texture, [SHADOW_SIZE, SHADOW_SIZE], BG),
            &shadow_batches,
        );
        gpu.cmd_barrier(shadow_cb, Stage::All, Stage::Transfer, HazardFlags::empty());
        gpu.cmd_copy_texture_to_buffer(
            shadow_cb,
            self.shadow_target.texture,
            self.shadow_readback.cast(),
        );
        gpu.queue_submit(Queue::Main, &[shadow_cb]);
        gpu.queue_wait_idle(Queue::Main);

        let mut shadow_pixels = vec![[0.0f32; 4]; (SHADOW_SIZE * SHADOW_SIZE) as usize];
        // SAFETY: shadow readback matches the completed texture copy.
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.shadow_readback.cpu,
                shadow_pixels.as_mut_ptr(),
                shadow_pixels.len(),
            );
        }
        let shadow_center =
            shadow_pixels[(SHADOW_SIZE / 2 * SHADOW_SIZE + SHADOW_SIZE / 2) as usize];
        println!("ui_extract shadow center pixel = {shadow_center:?}");
        assert!(
            (shadow_center[0] - SHADOW_COLOR[0]).abs() < 0.05
                && (shadow_center[1] - SHADOW_COLOR[1]).abs() < 0.05
                && (shadow_center[2] - SHADOW_COLOR[2]).abs() < 0.05
                && shadow_center[3] > 0.95,
            "shadow target's center (deep in the shadow's blurred interior) should read \
             ~SHADOW_COLOR at ~full coverage: {shadow_center:?} vs {SHADOW_COLOR:?}"
        );

        println!("ui_extract check OK frame={}", ctx.frame);
        ctx.request_exit();
    }
}

impl RenderScene for UiExtractCheckScene {
    fn draw(&mut self, ctx: &mut FrameCtx) {
        // Frame one contains the first extracted resources.
        if ctx.frame == 1 {
            self.check(ctx);
        }
    }

    fn teardown(&mut self, gpu: &Gpu) {
        if let Some(pass) = self.pass.take() {
            pass.free(gpu);
        }
        if let Some(heap) = self.heap.take() {
            heap.free(gpu);
        }
        gpu.free(self.readback);
        gpu.texture_free_and_destroy(self.target);
        gpu.free(self.shadow_readback);
        gpu.texture_free_and_destroy(self.shadow_target);
    }
}

fn spawn_fixture(mut commands: Commands) {
    commands.insert_resource(UiPaintList {
        vertices: fixture_vertices(),
        batches: fixture_batches(),
        quad_count: 2,
        // Shadow streams are extracted with the UI streams.
        shadow_vertices: fixture_shadow_vertices().to_vec(),
        shadow_batches: fixture_shadow_batches(),
        shadow_quad_count: 1,
    });
}

/// Runs without libtest because winit requires the main thread.
fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "ui-bridge UI extraction".into(),
                // Fixed scale keeps physical probe coordinates stable.
                resolution:
                    bevy::window::WindowResolution::new(SIZE, SIZE).with_scale_factor_override(1.0),
                visible: false,
                ..Default::default()
            }),
            ..Default::default()
        }))
        .add_plugins(AshaRenderPlugin::new(UiExtractCheckScene::new).extract_ui())
        .add_systems(Startup, spawn_fixture)
        .run();
}
