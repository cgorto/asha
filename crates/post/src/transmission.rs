//! Display-referred transmission effects and final output dithering.
//!
//! Apply this pass after the display transform. Continuous noise remains
//! available at rest; impulse effects require the gated shock input. When this
//! pass runs it owns the single final dither; when skipped, the tonemap must
//! dither its float output instead.

use abi_core::GpuPtr;
use abi_post::TransmissionData;
use gpu::{
    CommandBuffer, Gpu, HazardFlags, LoadOp, RenderAttachment, RenderPassDesc, SampledSlot,
    SamplerSlot, ShaderTypeGraphics, Stage, StoreOp, Texture,
};

use gpu::pass::{FrameAlloc, Pass};

#[derive(Clone, Copy, Debug, Default)]
pub struct TransmissionSettings {
    /// Baseline snow amplitude in display units.
    pub snow_base: f32,
    /// Additional snow proportional to acceleration.
    pub snow_accel: f32,
    /// Camera acceleration magnitude, units/s².
    pub accel: f32,
    /// Depth of the slow quality wander, as a fraction of the base.
    pub snow_wander: f32,
    /// Noise-wander phase in host-defined units.
    pub wander_t: f32,
    /// Chroma smear width in UV; 0 disables the extra taps.
    pub chroma_width: f32,
    /// Taps along the smear.
    pub chroma_taps: u32,
    /// Output quantization step; use 1/255 for 8-bit output.
    pub dither_step: f32,
    /// Per-frame noise salt.
    pub frame: u32,

    /// Gated shock value in `[0, 1]` from [`abi_post::shock_gate`].
    pub drive: f32,
    pub tear_bands: f32,
    pub tear_offset: f32,
    pub dropout_rows: f32,
    pub dropout_length: f32,
    pub dropout_gain: f32,
    /// Vertical roll in UV; use a higher event threshold.
    pub roll: f32,
    pub seam_width: f32,
    pub seam_gain: f32,
    pub chroma_desync: f32,
    pub snow_shock: f32,
}

impl TransmissionSettings {
    /// Whether the pass would otherwise be a copy. Dithering is excluded:
    /// if this pass is skipped, the upstream tonemap must dither instead.
    pub fn is_identity(&self) -> bool {
        self.snow_base == 0.0
            && self.snow_accel == 0.0
            && (self.chroma_taps <= 1 || self.chroma_width == 0.0)
            && self.dither_step == 0.0
            && self.drive == 0.0
            && self.roll == 0.0
    }
}

pub struct TransmissionPass {
    fullscreen_vert: gpu::Shader,
    frag: gpu::Shader,
    indices: gpu::Ptr<u32>,
}

impl TransmissionPass {
    /// No target of its own: this pass writes whatever the caller hands it,
    /// which is the backbuffer. That also means nothing to rebuild on resize.
    pub fn new(gpu: &Gpu) -> Self {
        Self {
            fullscreen_vert: gpu.shader_create(
                &asha_assets::load_spv("fullscreen_vert"),
                ShaderTypeGraphics::Vertex,
                "fullscreen_vert",
            ),
            frag: gpu.shader_create(
                &asha_assets::load_spv("transmission_frag"),
                ShaderTypeGraphics::Fragment,
                "transmission_frag",
            ),
            indices: gpu.fullscreen_triangle_indices(),
        }
    }

    /// Record into `target` from `input_slot`. Contract: the input is
    /// sampleable when this records.
    #[allow(clippy::too_many_arguments)]
    pub fn record(
        &self,
        gpu: &Gpu,
        cb: CommandBuffer,
        fa: &mut impl FrameAlloc,
        target: Texture,
        input_slot: SampledSlot,
        sampler: SamplerSlot,
        settings: &TransmissionSettings,
    ) {
        let data = fa.frame_alloc(TransmissionData {
            input_texture_id: input_slot.index(),
            sampler_id: sampler.index(),
            resolution: [target.dimensions[0] as f32, target.dimensions[1] as f32],
            frame: settings.frame,
            snow_base: settings.snow_base,
            snow_accel: settings.snow_accel,
            accel: settings.accel,
            snow_wander: settings.snow_wander,
            wander_t: settings.wander_t,
            chroma_width: settings.chroma_width,
            chroma_taps: settings.chroma_taps,
            dither_step: settings.dither_step,
            drive: settings.drive,
            tear_bands: settings.tear_bands,
            tear_offset: settings.tear_offset,
            dropout_rows: settings.dropout_rows,
            dropout_length: settings.dropout_length,
            dropout_gain: settings.dropout_gain,
            roll: settings.roll,
            seam_width: settings.seam_width,
            seam_gain: settings.seam_gain,
            chroma_desync: settings.chroma_desync,
            snow_shock: settings.snow_shock,
        });
        gpu.cmd_begin_render_pass(
            cb,
            RenderPassDesc {
                color_attachments: &[RenderAttachment {
                    texture: target,
                    load_op: LoadOp::DontCare,
                    store_op: StoreOp::Store,
                    clear_color: [0.0; 4],
                    ..Default::default()
                }],
                ..Default::default()
            },
        );
        gpu.cmd_set_shaders(cb, self.fullscreen_vert, self.frag);
        gpu.cmd_draw_indexed_instanced(cb, GpuPtr::null(), data.cast(), self.indices.cast(), 3, 1);
        gpu.cmd_end_render_pass(cb);
        gpu.cmd_barrier(
            cb,
            Stage::RasterColorOut,
            Stage::FragmentShader,
            HazardFlags::COLOR_ATTACHMENT,
        );
    }
}

impl Pass for TransmissionPass {
    const NAME: &'static str = "transmission";

    fn free(self, gpu: &Gpu) {
        gpu.shader_destroy(self.fullscreen_vert);
        gpu.shader_destroy(self.frag);
        gpu.free(self.indices);
    }
}
