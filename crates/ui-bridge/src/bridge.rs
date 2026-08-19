//! Render-thread half of the UI paint crossing.
//!
//! Converts extracted paint streams into per-batch GPU draw records.

use std::collections::HashMap;

use abi_ui::{UI_FLAG_TEXTURED, UiDraw, UiShadowDraw, UiShadowVertex, UiVertex, ui_flag};
use gpu::{
    Gpu, HeapSlots, Memory, OwnedTexture, Queue, SampledSlot, SamplerDesc, SamplerSlot,
    TextureDesc, TextureFormat, TextureViewDesc, UsageFlags,
};
use render::FrameCtx;
use text::{
    TextBandHeader, TextBatch, TextBatchDesc, TextCamera, TextCurve, TextDraw, TextGlyphDescriptor,
    TextGlyphInstance, TextScissor,
};

use crate::icons::IconUploadPayload;
use crate::paint::TextRunBatch;
use crate::paint::UiBatch as PaintBatch;
use crate::paint::UiShadowBatch as PaintShadowBatch;

/// Per-scene render-thread UI state and retained frame scratch.
///
/// Call [`Self::ingest`] before recording the UI pass.
/// Icon resources are lazy and freed by [`Self::free`].
#[derive(Default)]
pub struct UiBridge {
    /// Current UI draw batches; valid until the next ingest.
    batches: Vec<ui::UiBatch>,
    /// Retained host-lane batch scratch.
    paint_batches_scratch: Vec<PaintBatch>,
    /// Current box-shadow batches; valid until the next ingest.
    shadow_batches: Vec<ui::UiShadowBatch>,
    /// Retained shadow-batch scratch.
    paint_shadow_batches_scratch: Vec<PaintShadowBatch>,
    /// Current text batches; populated by [`Self::ingest_text`].
    text_batches: Vec<TextBatchDesc>,
    /// Retained text-batch scratch.
    text_batches_scratch: Vec<TextRunBatch>,
    /// Maps logical icon slots to bindless sampled slots.
    icon_slots: HashMap<u32, SampledSlot>,
    /// Icon textures owned and freed by this bridge.
    icon_textures: Vec<OwnedTexture>,
    /// Lazily-created sampler shared by textured UI draws.
    icon_sampler: Option<SamplerSlot>,
    /// Retained icon-upload scratch.
    icon_uploads_scratch: Vec<IconUploadPayload>,
}

impl UiBridge {
    pub fn new() -> Self {
        Self::default()
    }

    /// Builds frame draw records from extracted UI streams.
    ///
    /// `target_size` is the physical `[width, height]` of the target. The
    /// generated view uses a top-left origin with +Y down:
    /// `[2 / width, 2 / height, -1, -1]`.
    /// Call once per frame before `UiPass::record`; missing `extract_ui`
    /// wiring panics.
    pub fn ingest(&mut self, ctx: &mut FrameCtx, target_size: [u32; 2]) {
        assert!(
            target_size[0] > 0 && target_size[1] > 0,
            "UiBridge::ingest: target size must be non-zero"
        );

        let (vertices, vertex_count) = ctx.extracted::<UiVertex>();

        // Resolve logical texture slots; unresolved icons become untextured.
        for vertex in ctx.extracted_host_mut::<UiVertex>() {
            if !ui_flag(vertex.flags, UI_FLAG_TEXTURED) {
                continue;
            }
            match self.icon_slots.get(&vertex.tex_slot) {
                Some(slot) => vertex.tex_slot = slot.index(),
                None => {
                    vertex.flags &= !UI_FLAG_TEXTURED;
                    vertex.tex_slot = 0;
                }
            }
        }

        self.paint_batches_scratch.clear();
        self.paint_batches_scratch
            .extend_from_slice(ctx.extracted_host::<PaintBatch>());

        let view = [
            2.0 / target_size[0] as f32,
            2.0 / target_size[1] as f32,
            -1.0,
            -1.0,
        ];

        self.batches.clear();
        for i in 0..self.paint_batches_scratch.len() {
            let quad_range = self.paint_batches_scratch[i].quad_range.clone();
            let scissor = self.paint_batches_scratch[i].scissor;
            let quad_count = u32::try_from(quad_range.len()).expect("quad range fits u32");
            if quad_count == 0 {
                continue;
            }
            assert!(
                (quad_range.end as u64) * 4 <= vertex_count as u64,
                "UiPaintList batch quad range {quad_range:?} exceeds the extracted \
                 vertex stream ({vertex_count} vertices)"
            );

            // UiDraw has no base-quad field; offset each vertex pointer.
            let batch_vertices = vertices.offset((quad_range.start * 4) as i64);
            let draw = ctx.frame_alloc(UiDraw {
                vertices: batch_vertices,
                view,
                quad_count,
                sampler_slot: self.icon_sampler.map_or(0, SamplerSlot::index),
            });

            self.batches.push(ui::UiBatch {
                draw,
                quad_count,
                scissor: scissor.map(|rect| ui::UiScissor {
                    offset: [rect.min.x, rect.min.y],
                    extent: [
                        (rect.max.x - rect.min.x).max(0) as u32,
                        (rect.max.y - rect.min.y).max(0) as u32,
                    ],
                }),
            });
        }

        // Convert shadow batches through their separate ABI stream.
        let (shadow_vertices, shadow_vertex_count) = ctx.extracted::<UiShadowVertex>();

        self.paint_shadow_batches_scratch.clear();
        self.paint_shadow_batches_scratch
            .extend_from_slice(ctx.extracted_host::<PaintShadowBatch>());

        self.shadow_batches.clear();
        for i in 0..self.paint_shadow_batches_scratch.len() {
            let quad_range = self.paint_shadow_batches_scratch[i].quad_range.clone();
            let scissor = self.paint_shadow_batches_scratch[i].scissor;
            let quad_count = u32::try_from(quad_range.len()).expect("quad range fits u32");
            if quad_count == 0 {
                continue;
            }
            assert!(
                (quad_range.end as u64) * 4 <= shadow_vertex_count as u64,
                "UiPaintList shadow batch quad range {quad_range:?} exceeds the extracted \
                 shadow vertex stream ({shadow_vertex_count} vertices)"
            );

            let batch_vertices = shadow_vertices.offset((quad_range.start * 4) as i64);
            let draw = ctx.frame_alloc(UiShadowDraw {
                vertices: batch_vertices,
                view,
                quad_count,
            });

            self.shadow_batches.push(ui::UiShadowBatch {
                draw,
                quad_count,
                scissor: scissor.map(|rect| ui::UiScissor {
                    offset: [rect.min.x, rect.min.y],
                    extent: [
                        (rect.max.x - rect.min.x).max(0) as u32,
                        (rect.max.y - rect.min.y).max(0) as u32,
                    ],
                }),
            });
        }
    }

    /// Current UI batches, valid until the next ingest.
    pub fn batches(&self) -> &[ui::UiBatch] {
        &self.batches
    }

    /// Current shadow batches, valid until the next ingest.
    /// Their indices match the extracted paint shadow descriptors; use those
    /// descriptors' `order` values as the parallel stream when interleaving.
    pub fn shadow_batches(&self) -> &[ui::UiShadowBatch] {
        &self.shadow_batches
    }

    /// Uploads pending icons and registers them in the caller-owned heap.
    ///
    /// Requires `AshaRenderPluginExt::extract_icons` and therefore
    /// `UiBridgePlugin`; missing extraction wiring panics. The main-thread
    /// queue exposes each newly loaded payload for one extracted frame, then
    /// clears it on the next `PostUpdate`. Call once per frame before
    /// [`Self::ingest`].
    pub fn ingest_icons(&mut self, gpu: &Gpu, heap: &mut HeapSlots, ctx: &mut FrameCtx) {
        self.icon_uploads_scratch.clear();
        self.icon_uploads_scratch
            .extend_from_slice(ctx.extracted_host::<IconUploadPayload>());

        if self.icon_uploads_scratch.is_empty() {
            return;
        }

        if self.icon_sampler.is_none() {
            self.icon_sampler =
                Some(heap.add_sampler(gpu, gpu.sampler_descriptor(SamplerDesc::default())));
        }

        for i in 0..self.icon_uploads_scratch.len() {
            let payload = &self.icon_uploads_scratch[i];
            if self.icon_slots.contains_key(&payload.logical_slot) {
                // Ignore duplicate payloads.
                continue;
            }

            let texture = gpu.texture_alloc_and_create(
                TextureDesc {
                    dimensions: [payload.width, payload.height, 1],
                    format: TextureFormat::Rgba8Unorm,
                    usage: UsageFlags::SAMPLED | UsageFlags::TRANSFER_DST,
                    ..Default::default()
                },
                Queue::Main,
                None,
            );

            let staging = gpu.alloc_slice::<u8>(payload.pixels.len() as u64, Memory::Default);
            // SAFETY: allocation matches the RGBA8 payload length.
            // The heap lacks sRGB views, so upload linearized channels.
            unsafe {
                for (px, chunk) in payload.pixels.chunks_exact(4).enumerate() {
                    let dst = staging.cpu.add(px * 4);
                    *dst = srgb_to_linear(chunk[0]);
                    *dst.add(1) = srgb_to_linear(chunk[1]);
                    *dst.add(2) = srgb_to_linear(chunk[2]);
                    *dst.add(3) = chunk[3];
                }
            }

            // Synchronous upload keeps staging lifetime local.
            let cb = gpu.commands_begin(Queue::Main);
            gpu.cmd_copy_to_texture(cb, texture.texture, staging);
            gpu.queue_submit(Queue::Main, &[cb]);
            gpu.queue_wait_idle(Queue::Main);
            gpu.free(staging);

            let slot = heap.add_sampled(
                gpu,
                gpu.texture_view_descriptor(texture.texture, TextureViewDesc::default()),
            );
            self.icon_slots.insert(payload.logical_slot, slot);
            self.icon_textures.push(texture);
        }
    }

    /// Returns whether an icon slot has uploaded.
    pub fn icon_ready(&self, logical_slot: u32) -> bool {
        self.icon_slots.contains_key(&logical_slot)
    }

    /// Frees icon textures owned by this bridge.
    pub fn free(self, gpu: &Gpu) {
        for texture in self.icon_textures {
            gpu.texture_free_and_destroy(texture);
        }
    }

    /// Builds text draw records from extracted glyph buffers.
    ///
    /// Call once per frame before `TextPass::record_batches`.
    /// Panics if text extraction was not configured.
    pub fn ingest_text(&mut self, ctx: &mut FrameCtx, target_size: [u32; 2]) {
        assert!(
            target_size[0] > 0 && target_size[1] > 0,
            "UiBridge::ingest_text: target size must be non-zero"
        );

        let (instances, instance_count) = ctx.extracted::<TextGlyphInstance>();
        let (descriptors, _) = ctx.extracted::<TextGlyphDescriptor>();
        let (curves, _) = ctx.extracted::<TextCurve>();
        let (bands, _) = ctx.extracted::<TextBandHeader>();
        let (band_curve_indices, _) = ctx.extracted::<u32>();

        self.text_batches_scratch.clear();
        self.text_batches_scratch
            .extend_from_slice(ctx.extracted_host::<TextRunBatch>());

        // Match the UI camera: top-left origin, positive y downward.
        let xform = [
            2.0 / target_size[0] as f32,
            2.0 / target_size[1] as f32,
            -1.0,
            -1.0,
        ];

        self.text_batches.clear();
        for run in &self.text_batches_scratch {
            let range = run.instance_range.clone();
            let glyph_count = u32::try_from(range.len()).expect("instance range fits u32");
            if glyph_count == 0 {
                continue;
            }
            assert!(
                (range.end as u64) <= instance_count as u64,
                "TextPaintList batch instance range {range:?} exceeds the extracted \
                 TextGlyphInstance stream ({instance_count} instances)"
            );

            let batch_instances = instances.offset(range.start as i64);
            let draw = ctx.frame_alloc(TextDraw {
                instances: batch_instances,
                descriptors,
                curves,
                bands,
                band_curve_indices,
                camera: TextCamera {
                    xform,
                    zoom: 1.0,
                    font_px_per_em: run.font_px_per_em,
                    _pad0: [0.0; 2],
                },
                glyph_count,
                flags: 0,
                _pad0: [0; 2],
            });

            let scissor = run.clip.map(|rect| TextScissor {
                offset: [rect.min.x as i32, rect.min.y as i32],
                extent: [
                    (rect.max.x - rect.min.x).max(0.0) as u32,
                    (rect.max.y - rect.min.y).max(0.0) as u32,
                ],
            });

            self.text_batches.push(TextBatchDesc {
                batch: TextBatch { draw, glyph_count },
                scissor,
            });
        }
    }

    /// Current text batches, valid until the next text ingest.
    pub fn text_batches(&self) -> &[TextBatchDesc] {
        &self.text_batches
    }
}

/// Converts one sRGB byte to linear space for upload.
fn srgb_to_linear(byte: u8) -> u8 {
    let c = byte as f32 / 255.0;
    let linear = if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    };
    (linear * 255.0 + 0.5) as u8
}

/// Adds generic UI extraction lanes to the render plugin.
pub trait AshaRenderPluginExt {
    /// Extracts UI vertices, batches, and shadow streams each frame.
    /// Requires [`crate::UiBridgePlugin`].
    fn extract_ui(self) -> Self;

    /// Extracts text instances, batches, and glyph buffers each frame.
    /// Requires [`crate::UiBridgePlugin`].
    fn extract_text(self) -> Self;

    /// Extracts pending icon uploads each frame.
    /// Requires [`crate::UiBridgePlugin`].
    fn extract_icons(self) -> Self;
}

impl AshaRenderPluginExt for render::AshaRenderPlugin {
    fn extract_ui(self) -> Self {
        self.extract_resource_slice::<crate::UiPaintList, UiVertex>(|list| &list.vertices)
            .extract_resource_host_clone::<crate::UiPaintList, PaintBatch>(|list| &list.batches)
            // Shadows share the UI extraction lane.
            .extract_resource_slice::<crate::UiPaintList, UiShadowVertex>(|list| {
                &list.shadow_vertices
            })
            .extract_resource_host_clone::<crate::UiPaintList, PaintShadowBatch>(|list| {
                &list.shadow_batches
            })
    }

    fn extract_text(self) -> Self {
        self.extract_resource_slice::<crate::TextPaintList, TextGlyphInstance>(|list| {
            &list.instances
        })
        .extract_resource_host_clone::<crate::TextPaintList, TextRunBatch>(|list| &list.batches)
        .extract_resource_slice::<crate::GlyphOutlineProvider, TextGlyphDescriptor>(|provider| {
            provider.descriptors()
        })
        .extract_resource_slice::<crate::GlyphOutlineProvider, TextCurve>(|provider| {
            provider.curves()
        })
        .extract_resource_slice::<crate::GlyphOutlineProvider, TextBandHeader>(|provider| {
            provider.bands()
        })
        .extract_resource_slice::<crate::GlyphOutlineProvider, u32>(|provider| {
            provider.band_curve_indices()
        })
    }

    fn extract_icons(self) -> Self {
        self.extract_resource_host_clone::<crate::icons::IconUploadQueue, IconUploadPayload>(
            |queue| queue.pending(),
        )
    }
}
