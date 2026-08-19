//! Render-side UI pass for SDF shapes, gradients, borders, and icons.
//!
//! `ui-bridge` builds `UiVertex` streams; `abi_ui` defines shading and ABI.
//! The shader vertex-pulls through `UiDraw`; identity indices emulate
//! non-indexed draws because `gpu` exposes indexed drawing only.

use abi_ui::{UiDraw, UiShadowDraw};
use gpu::{
    BlendFactor, BlendOp, BlendState, CommandBuffer, CompareOp, DepthFlags, DepthState, Gpu,
    GpuPtr, LoadOp, Memory, RenderAttachment, RenderPassDesc, ShaderTypeGraphics, Stage, StoreOp,
};

const COLOR_WRITE_RGBA: u8 = 0xF;

/// Identity index buffer capacity, in quads. 65536 quads (393216 indices,
/// ~1.5 MiB) is a generous ceiling for a UI overlay in one batch; a batch
/// past it is almost certainly a bug, not a legitimate scene, so `record`
/// asserts rather than growing mid-frame.
const MAX_QUADS: u32 = 1 << 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UiScissor {
    pub offset: [i32; 2],
    pub extent: [u32; 2],
}

#[derive(Debug, Clone, Copy)]
pub struct UiPassTarget {
    pub texture: gpu::Texture,
    pub size: [u32; 2],
    pub load_op: LoadOp,
    pub store_op: StoreOp,
    pub clear_color: [f32; 4],
}

impl UiPassTarget {
    /// Draw UI over an existing color target, preserving prior pass output.
    /// Use this for each interleaved UI, shadow, or text pass.
    pub fn overlay(texture: gpu::Texture, size: [u32; 2]) -> Self {
        Self {
            texture,
            size,
            load_op: LoadOp::Load,
            store_op: StoreOp::Store,
            clear_color: [0.0; 4],
        }
    }

    /// Clear the target to `clear_color` before drawing UI over it.
    pub fn clear(texture: gpu::Texture, size: [u32; 2], clear_color: [f32; 4]) -> Self {
        Self {
            texture,
            size,
            load_op: LoadOp::Clear,
            store_op: StoreOp::Store,
            clear_color,
        }
    }
}

/// One draw within the pass's single render pass. `draw` is the GPU address
/// of a buffer-backed [`UiDraw`] (vertices, view, quad_count, sampler_slot —
/// both push-constant slots receive the same pointer, per the graphics push
/// ABI). `scissor` restricts the draw to a sub-rect of the target; `None`
/// draws over the whole target (the default `cmd_begin_render_pass` sets).
#[derive(Debug, Clone, Copy)]
pub struct UiBatch {
    /// GPU address of a buffer-backed [`UiDraw`].
    pub draw: GpuPtr<UiDraw>,
    /// CPU-known quad count for the draw call. Must match
    /// `UiDraw::quad_count`; the shader also receives that field and treats
    /// any vertex past it as degenerate (see `ui_vert`).
    pub quad_count: u32,
    pub scissor: Option<UiScissor>,
}

/// Box-shadow draw data for the shadow pipeline.
///
/// The bridge keeps shadow order in a parallel `ui_bridge` descriptor stream.
/// Hosts merge that stream with quad/text orders and record one batch at a
/// time. Each inter-pass target uses [`UiPassTarget::overlay`] (`LoadOp::Load`);
/// `UiPass::end_pass` supplies the color-attachment barrier between passes.
#[derive(Debug, Clone, Copy)]
pub struct UiShadowBatch {
    /// GPU address of a buffer-backed [`UiShadowDraw`].
    pub draw: GpuPtr<UiShadowDraw>,
    /// CPU-known quad count for the draw call. Must match
    /// `UiShadowDraw::quad_count` (see `ui_shadow_vert`).
    pub quad_count: u32,
    pub scissor: Option<UiScissor>,
}

pub struct UiPass {
    vert_shader: gpu::Shader,
    frag_shader: gpu::Shader,
    /// Box-shadow shaders sharing the pass's fixed GPU state.
    shadow_vert_shader: gpu::Shader,
    shadow_frag_shader: gpu::Shader,
    /// Identity indices emulate non-indexed `gl_VertexIndex` values.
    /// Both pipelines use `quad = vertex_index / 6`.
    identity_indices: gpu::Ptr<u32>,
}

impl UiPass {
    pub fn new(gpu: &Gpu) -> Self {
        let index_count = (MAX_QUADS as u64) * 6;
        let identity_indices = gpu.alloc_slice::<u32>(index_count, Memory::Default);
        // SAFETY: fresh host-visible allocation sized for `index_count`.
        unsafe {
            for i in 0..index_count {
                *identity_indices.cpu.add(i as usize) = i as u32;
            }
        }

        Self {
            vert_shader: gpu.shader_create(
                &asha_assets::load_spv("ui_vert"),
                ShaderTypeGraphics::Vertex,
                "ui_vert",
            ),
            frag_shader: gpu.shader_create(
                &asha_assets::load_spv("ui_frag"),
                ShaderTypeGraphics::Fragment,
                "ui_frag",
            ),
            shadow_vert_shader: gpu.shader_create(
                &asha_assets::load_spv("ui_shadow_vert"),
                ShaderTypeGraphics::Vertex,
                "ui_shadow_vert",
            ),
            shadow_frag_shader: gpu.shader_create(
                &asha_assets::load_spv("ui_shadow_frag"),
                ShaderTypeGraphics::Fragment,
                "ui_shadow_frag",
            ),
            identity_indices,
        }
    }

    pub fn free(self, gpu: &Gpu) {
        gpu.shader_destroy(self.vert_shader);
        gpu.shader_destroy(self.frag_shader);
        gpu.shader_destroy(self.shadow_vert_shader);
        gpu.shader_destroy(self.shadow_frag_shader);
        gpu.free(self.identity_indices);
    }

    /// Begins a UI pass with fixed depth, culling, and blend state.
    fn begin_pass(&self, gpu: &Gpu, cb: CommandBuffer, target: &UiPassTarget) {
        assert!(
            target.size[0] > 0 && target.size[1] > 0,
            "ui target size must be non-zero"
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

        // UI winding is CCW; explicitly configure overlay state.
        gpu.cmd_set_depth_state(
            cb,
            DepthState {
                mode: DepthFlags::empty(),
                compare: CompareOp::Always,
                ..Default::default()
            },
        );
        gpu.cmd_set_cull_mode(cb, true);
        gpu.cmd_set_front_face(cb, false);
        gpu.cmd_set_blend_state(cb, straight_alpha_blend_state());
    }

    /// Ends the pass and emits the inter-pass color-attachment barrier.
    fn end_pass(&self, gpu: &Gpu, cb: CommandBuffer) {
        gpu.cmd_end_render_pass(cb);
        gpu.cmd_barrier(
            cb,
            Stage::RasterColorOut,
            Stage::FragmentShader,
            gpu::HazardFlags::COLOR_ATTACHMENT,
        );
    }

    /// Record one render pass over `target`, drawing every batch in order
    /// (painter's order is pre-sorted CPU-side — there is no depth test).
    /// The caller must have already bound the bindless descriptor heaps
    /// (`HeapSlots::bind` / `cmd_set_desc_heap`) on `cb` this frame: the
    /// fragment shader's `RuntimeArray` bindings must be valid even for
    /// untextured batches.
    pub fn record(&self, gpu: &Gpu, cb: CommandBuffer, target: UiPassTarget, batches: &[UiBatch]) {
        self.begin_pass(gpu, cb, &target);
        gpu.cmd_set_shaders(cb, self.vert_shader, self.frag_shader);

        for batch in batches {
            if batch.quad_count == 0 {
                continue;
            }
            assert!(
                !batch.draw.is_null(),
                "ui batch has quads but UiDraw pointer is null"
            );
            assert!(
                batch.quad_count <= MAX_QUADS,
                "ui batch of {} quads exceeds the identity index buffer capacity ({MAX_QUADS} quads)",
                batch.quad_count
            );

            match batch.scissor {
                Some(scissor) => gpu.cmd_set_scissor(cb, scissor.offset, scissor.extent),
                None => gpu.cmd_set_scissor(cb, [0, 0], target.size),
            }

            gpu.cmd_draw_indexed_instanced(
                cb,
                batch.draw.cast(),
                batch.draw.cast(),
                self.identity_indices.cast(),
                batch.quad_count * 6,
                1,
            );
        }

        self.end_pass(gpu, cb);
    }

    /// Records shadow batches using the shadow pipeline.
    pub fn record_shadows(
        &self,
        gpu: &Gpu,
        cb: CommandBuffer,
        target: UiPassTarget,
        batches: &[UiShadowBatch],
    ) {
        self.begin_pass(gpu, cb, &target);
        gpu.cmd_set_shaders(cb, self.shadow_vert_shader, self.shadow_frag_shader);

        for batch in batches {
            if batch.quad_count == 0 {
                continue;
            }
            assert!(
                !batch.draw.is_null(),
                "ui shadow batch has quads but UiShadowDraw pointer is null"
            );
            assert!(
                batch.quad_count <= MAX_QUADS,
                "ui shadow batch of {} quads exceeds the identity index buffer capacity ({MAX_QUADS} quads)",
                batch.quad_count
            );

            match batch.scissor {
                Some(scissor) => gpu.cmd_set_scissor(cb, scissor.offset, scissor.extent),
                None => gpu.cmd_set_scissor(cb, [0, 0], target.size),
            }

            gpu.cmd_draw_indexed_instanced(
                cb,
                batch.draw.cast(),
                batch.draw.cast(),
                self.identity_indices.cast(),
                batch.quad_count * 6,
                1,
            );
        }

        self.end_pass(gpu, cb);
    }
}

/// Straight-alpha Porter–Duff over compositing.
///
/// Alpha uses `One` as its source factor to avoid squaring coverage.
fn straight_alpha_blend_state() -> BlendState {
    BlendState {
        enable: true,
        color_op: BlendOp::Add,
        src_color_factor: BlendFactor::SrcAlpha,
        dst_color_factor: BlendFactor::OneMinusSrcAlpha,
        alpha_op: BlendOp::Add,
        src_alpha_factor: BlendFactor::One,
        dst_alpha_factor: BlendFactor::OneMinusSrcAlpha,
        color_write_mask: COLOR_WRITE_RGBA,
    }
}
