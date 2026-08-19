//! Render-side Slug text pass using the `abi_ui::TextDraw` ABI.
//!
//! CPU shaping/cache preparation lives in this crate; shader logic lives in
//! `shaders/lib`. Overlay targets preserve prior UI or text passes.

use abi_ui::TextDraw;
use gpu::{
    BlendFactor, BlendOp, BlendState, CommandBuffer, CompareOp, DepthFlags, DepthState, Gpu,
    GpuPtr, LoadOp, Memory, RenderAttachment, RenderPassDesc, ShaderTypeGraphics, Stage, StoreOp,
};

const QUAD_INDICES: [u32; 6] = [0, 1, 2, 2, 1, 3];
const COLOR_WRITE_RGBA: u8 = 0xF;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TextScissor {
    pub offset: [i32; 2],
    pub extent: [u32; 2],
}

#[derive(Debug, Clone, Copy)]
pub struct TextPassTarget {
    pub texture: gpu::Texture,
    pub size: [u32; 2],
    pub load_op: LoadOp,
    pub store_op: StoreOp,
    pub clear_color: [f32; 4],
    pub scissor: Option<TextScissor>,
}

impl TextPassTarget {
    /// Draw text over an existing color target with `LoadOp::Load`.
    /// Use when text is interleaved with other overlay passes.
    pub fn overlay(texture: gpu::Texture, size: [u32; 2]) -> Self {
        Self {
            texture,
            size,
            load_op: LoadOp::Load,
            store_op: StoreOp::Store,
            clear_color: [0.0; 4],
            scissor: None,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TextBatch {
    /// GPU address of a buffer-backed [`TextDraw`].
    pub draw: GpuPtr<TextDraw>,
    /// Must match `TextDraw::glyph_count`.
    pub glyph_count: u32,
}

/// A text batch with an optional per-batch scissor rectangle.
#[derive(Debug, Clone, Copy)]
pub struct TextBatchDesc {
    pub batch: TextBatch,
    /// Falls back to the target scissor, then the full target.
    pub scissor: Option<TextScissor>,
}

pub struct TextPass {
    vert_shader: gpu::Shader,
    cover_frag_shader: gpu::Shader,
    blend_frag_shader: gpu::Shader,
    quad_indices: gpu::Ptr<u32>,
}

impl TextPass {
    pub fn new(gpu: &Gpu) -> Self {
        let quad_indices = gpu.alloc_slice::<u32>(QUAD_INDICES.len() as u64, Memory::Default);
        unsafe {
            for (i, index) in QUAD_INDICES.iter().copied().enumerate() {
                *quad_indices.cpu.add(i) = index;
            }
        }

        Self {
            vert_shader: gpu.shader_create(
                &asha_assets::load_spv("text_vert"),
                ShaderTypeGraphics::Vertex,
                "text_vert",
            ),
            cover_frag_shader: gpu.shader_create(
                &asha_assets::load_spv("text_cover_frag"),
                ShaderTypeGraphics::Fragment,
                "text_cover_frag",
            ),
            blend_frag_shader: gpu.shader_create(
                &asha_assets::load_spv("text_blend_frag"),
                ShaderTypeGraphics::Fragment,
                "text_blend_frag",
            ),
            quad_indices,
        }
    }

    pub fn free(self, gpu: &Gpu) {
        gpu.shader_destroy(self.vert_shader);
        gpu.shader_destroy(self.cover_frag_shader);
        gpu.shader_destroy(self.blend_frag_shader);
        gpu.free(self.quad_indices);
    }

    pub fn record(&self, gpu: &Gpu, cb: CommandBuffer, target: TextPassTarget, batch: TextBatch) {
        if batch.glyph_count == 0 {
            return;
        }
        assert!(
            !batch.draw.is_null(),
            "text batch has glyphs but TextDraw pointer is null"
        );
        assert!(
            target.size[0] > 0 && target.size[1] > 0,
            "text target size must be non-zero"
        );

        gpu.cmd_begin_render_pass(
            cb,
            RenderPassDesc {
                render_area_size: target.size,
                color_attachments: &[RenderAttachment {
                    texture: target.texture,
                    load_op: target.load_op,
                    store_op: target.store_op,
                    clear_color: target.clear_color,
                    ..Default::default()
                }],
                ..Default::default()
            },
        );
        if let Some(scissor) = target.scissor {
            gpu.cmd_set_scissor(cb, scissor.offset, scissor.extent);
        }

        // Text overlays require depth and culling disabled.
        gpu.cmd_set_depth_state(
            cb,
            DepthState {
                mode: DepthFlags::empty(),
                compare: CompareOp::Always,
                ..Default::default()
            },
        );
        gpu.cmd_set_cull_mode(cb, false);

        gpu.cmd_set_shaders(cb, self.vert_shader, self.cover_frag_shader);
        gpu.cmd_set_blend_state(cb, cover_blend_state());
        self.draw_quads(gpu, cb, batch);

        gpu.cmd_set_shaders(cb, self.vert_shader, self.blend_frag_shader);
        gpu.cmd_set_blend_state(cb, additive_blend_state());
        self.draw_quads(gpu, cb, batch);

        gpu.cmd_end_render_pass(cb);
        gpu.cmd_barrier(
            cb,
            Stage::RasterColorOut,
            Stage::FragmentShader,
            gpu::HazardFlags::COLOR_ATTACHMENT,
        );
    }

    /// Records batches in order, applying each batch's scissor.
    ///
    /// Cover and blend draws remain adjacent for overlapping scissors;
    /// callers use an overlay target when chaining this pass with UI passes.
    pub fn record_batches(
        &self,
        gpu: &Gpu,
        cb: CommandBuffer,
        target: TextPassTarget,
        batches: &[TextBatchDesc],
    ) {
        assert!(
            target.size[0] > 0 && target.size[1] > 0,
            "text target size must be non-zero"
        );
        if batches.iter().all(|desc| desc.batch.glyph_count == 0) {
            return;
        }

        gpu.cmd_begin_render_pass(
            cb,
            RenderPassDesc {
                render_area_size: target.size,
                color_attachments: &[RenderAttachment {
                    texture: target.texture,
                    load_op: target.load_op,
                    store_op: target.store_op,
                    clear_color: target.clear_color,
                    ..Default::default()
                }],
                ..Default::default()
            },
        );

        gpu.cmd_set_depth_state(
            cb,
            DepthState {
                mode: DepthFlags::empty(),
                compare: CompareOp::Always,
                ..Default::default()
            },
        );
        gpu.cmd_set_cull_mode(cb, false);

        for desc in batches {
            if desc.batch.glyph_count == 0 {
                continue;
            }
            assert!(
                !desc.batch.draw.is_null(),
                "text batch has glyphs but TextDraw pointer is null"
            );
            match desc.scissor.or(target.scissor) {
                Some(scissor) => gpu.cmd_set_scissor(cb, scissor.offset, scissor.extent),
                None => gpu.cmd_set_scissor(cb, [0, 0], target.size),
            }

            gpu.cmd_set_shaders(cb, self.vert_shader, self.cover_frag_shader);
            gpu.cmd_set_blend_state(cb, cover_blend_state());
            self.draw_quads(gpu, cb, desc.batch);

            gpu.cmd_set_shaders(cb, self.vert_shader, self.blend_frag_shader);
            gpu.cmd_set_blend_state(cb, additive_blend_state());
            self.draw_quads(gpu, cb, desc.batch);
        }

        gpu.cmd_end_render_pass(cb);
        gpu.cmd_barrier(
            cb,
            Stage::RasterColorOut,
            Stage::FragmentShader,
            gpu::HazardFlags::COLOR_ATTACHMENT,
        );
    }

    fn draw_quads(&self, gpu: &Gpu, cb: CommandBuffer, batch: TextBatch) {
        gpu.cmd_draw_indexed_instanced(
            cb,
            batch.draw.cast(),
            batch.draw.cast(),
            self.quad_indices.cast(),
            QUAD_INDICES.len() as u32,
            batch.glyph_count,
        );
    }
}

fn cover_blend_state() -> BlendState {
    BlendState {
        enable: true,
        color_op: BlendOp::Add,
        src_color_factor: BlendFactor::Zero,
        dst_color_factor: BlendFactor::OneMinusSrcColor,
        alpha_op: BlendOp::Add,
        src_alpha_factor: BlendFactor::Zero,
        dst_alpha_factor: BlendFactor::OneMinusSrcColor,
        color_write_mask: COLOR_WRITE_RGBA,
    }
}

fn additive_blend_state() -> BlendState {
    BlendState {
        enable: true,
        color_op: BlendOp::Add,
        src_color_factor: BlendFactor::One,
        dst_color_factor: BlendFactor::One,
        alpha_op: BlendOp::Add,
        src_alpha_factor: BlendFactor::One,
        dst_alpha_factor: BlendFactor::One,
        color_write_mask: COLOR_WRITE_RGBA,
    }
}
