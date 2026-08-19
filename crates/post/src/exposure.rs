//! Measures log-average luminance for caller-controlled auto exposure.
//!
//! A sparse grid feeds a 1×1 target and delayed readback. Temporal response
//! remains the caller's responsibility.

use abi_core::GpuPtr;
use abi_post::ExposureProbeData;
use gpu::{
    CommandBuffer, Gpu, HazardFlags, HeapSlots, LoadOp, Memory, OwnedTexture, Queue,
    RenderAttachment, RenderPassDesc, SampledSlot, SamplerSlot, ShaderTypeGraphics, Stage, StoreOp,
    TextureDesc, TextureFormat, TextureViewDesc, UsageFlags,
};

use gpu::pass::{FrameAlloc, Pass};

pub struct ExposurePass {
    target: OwnedTexture,
    slot: SampledSlot,
    readback: gpu::Ptr<f32>,
    /// Whether the readback contains a recorded measurement.
    primed: bool,
    fullscreen_vert: gpu::Shader,
    frag: gpu::Shader,
    indices: gpu::Ptr<u32>,
}

impl ExposurePass {
    /// Creates the resolution-independent 1×1 probe target.
    pub fn new(gpu: &Gpu, heap: &mut HeapSlots) -> Self {
        let target = gpu.texture_alloc_and_create(
            TextureDesc {
                dimensions: [1, 1, 1],
                format: TextureFormat::R32Float,
                usage: UsageFlags::COLOR_ATTACHMENT
                    | UsageFlags::SAMPLED
                    | UsageFlags::TRANSFER_SRC,
                ..Default::default()
            },
            Queue::Main,
            None,
        );
        let slot = heap.alloc_sampled();
        heap.write_sampled(
            gpu,
            slot,
            gpu.texture_view_descriptor(target.texture, TextureViewDesc::default()),
        );
        Self {
            target,
            slot,
            readback: gpu.alloc_slice(1, Memory::Readback),
            primed: false,
            fullscreen_vert: gpu.shader_create(
                &asha_assets::load_spv("fullscreen_vert"),
                ShaderTypeGraphics::Vertex,
                "fullscreen_vert",
            ),
            frag: gpu.shader_create(
                &asha_assets::load_spv("exposure_probe_frag"),
                ShaderTypeGraphics::Fragment,
                "exposure_probe_frag",
            ),
            indices: gpu.fullscreen_triangle_indices(),
        }
    }

    /// The previous frame's log-average luminance, or `None` before the first
    /// measurement has landed.
    pub fn log_average(&self) -> Option<f32> {
        if !self.primed {
            return None;
        }
        // SAFETY: the caller's retired frame wrote one f32 here.
        let value = unsafe { *self.readback.cpu };
        // Reject non-finite measurements.
        value.is_finite().then_some(value)
    }

    /// Measure `hdr_slot` and stage the result for readback. Contract: the
    /// input is sampleable when this records.
    pub fn record(
        &mut self,
        gpu: &Gpu,
        cb: CommandBuffer,
        fa: &mut impl FrameAlloc,
        hdr_slot: SampledSlot,
        sampler: SamplerSlot,
    ) {
        let data = fa.frame_alloc(ExposureProbeData {
            input_texture_id: hdr_slot.index(),
            sampler_id: sampler.index(),
        });
        gpu.cmd_begin_render_pass(
            cb,
            RenderPassDesc {
                color_attachments: &[RenderAttachment {
                    texture: self.target.texture,
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
            Stage::Transfer,
            HazardFlags::empty(),
        );
        gpu.cmd_copy_texture_to_buffer(cb, self.target.texture, self.readback.cast());
        self.primed = true;
    }

    pub fn slot(&self) -> SampledSlot {
        self.slot
    }
}

impl Pass for ExposurePass {
    const NAME: &'static str = "exposure";

    fn free(self, gpu: &Gpu) {
        gpu.shader_destroy(self.fullscreen_vert);
        gpu.shader_destroy(self.frag);
        gpu.free(self.indices);
        gpu.free(self.readback);
        gpu.texture_free_and_destroy(self.target);
    }
}
