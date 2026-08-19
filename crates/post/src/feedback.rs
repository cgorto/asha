//! Double-buffered HDR feedback with decay and motion reprojection.
//!
//! Each frame combines fresh input with decayed, bilinearly sampled history.
//! Ping-pong textures keep the source separate from the render target.
//! The color-output-to-fragment barrier orders frame-to-frame sampling.

use abi_core::GpuPtr;
use abi_core::glam::UVec2;
use abi_post::{FeedbackCamera, FeedbackData};
use gpu::{
    CommandBuffer, Gpu, HazardFlags, HeapSlots, LoadOp, OwnedTexture, Queue, RenderAttachment,
    RenderPassDesc, SampledSlot, SamplerSlot, ShaderTypeGraphics, Stage, StoreOp, TextureDesc,
    TextureFormat, TextureViewDesc, UsageFlags,
};

use gpu::pass::{FrameAlloc, Pass};

pub struct FeedbackPass {
    size: UVec2,
    targets: [OwnedTexture; 2],
    slots: [SampledSlot; 2],
    /// Ping-pong index for the next record.
    write: usize,
    /// False until a record has defined the history texture's contents —
    /// the first record after (re)build must not sample undefined memory.
    primed: bool,
    fullscreen_vert: gpu::Shader,
    frag: gpu::Shader,
    indices: gpu::Ptr<u32>,
}

impl FeedbackPass {
    /// Rebuild resources when `size` changes.
    ///
    /// Waits before freeing resources and resets the accumulator.
    /// Returns whether resources were rebuilt.
    pub fn ensure(this: &mut Option<Self>, gpu: &Gpu, heap: &mut HeapSlots, size: UVec2) -> bool {
        assert!(size.x > 0 && size.y > 0);
        if this.as_ref().is_some_and(|f| f.size == size) {
            return false;
        }
        let slots = match this.take() {
            Some(old) => {
                gpu.queue_wait_idle(Queue::Main);
                let slots = old.slots;
                old.free(gpu);
                slots
            }
            None => [heap.alloc_sampled(), heap.alloc_sampled()],
        };

        let target = || {
            gpu.texture_alloc_and_create(
                TextureDesc {
                    dimensions: [size.x, size.y, 1],
                    format: TextureFormat::Rgba16Float,
                    // Supports readback.
                    usage: UsageFlags::COLOR_ATTACHMENT
                        | UsageFlags::SAMPLED
                        | UsageFlags::TRANSFER_SRC,
                    ..Default::default()
                },
                Queue::Main,
                None,
            )
        };
        let targets = [target(), target()];
        for (slot, tex) in slots.iter().zip(&targets) {
            heap.write_sampled(
                gpu,
                *slot,
                gpu.texture_view_descriptor(tex.texture, TextureViewDesc::default()),
            );
        }

        let indices = gpu.fullscreen_triangle_indices();
        *this = Some(Self {
            size,
            targets,
            slots,
            write: 0,
            primed: false,
            fullscreen_vert: gpu.shader_create(
                &asha_assets::load_spv("fullscreen_vert"),
                ShaderTypeGraphics::Vertex,
                "fullscreen_vert",
            ),
            frag: gpu.shader_create(
                &asha_assets::load_spv("feedback_frag"),
                ShaderTypeGraphics::Fragment,
                "feedback_frag",
            ),
            indices,
        });
        true
    }

    /// Clear history before resuming after a recording gap.
    pub fn reset(&mut self) {
        self.primed = false;
    }

    /// Record one accumulation step. Contract: `input_slot` is sampleable
    /// when this records; ends with the returned slot (this frame's
    /// accumulator — feed it to the consumer in the input's place)
    /// sampleable. `decay` = exp(−rate·dt) and `floor` = floor/s·dt,
    /// dt-corrected by the host (`abi_post::feedback_combine`).
    #[allow(clippy::too_many_arguments)]
    pub fn record(
        &mut self,
        gpu: &Gpu,
        cb: CommandBuffer,
        fa: &mut impl FrameAlloc,
        input_slot: SampledSlot,
        sampler: SamplerSlot,
        decay: f32,
        floor: f32,
        flow: f32,
        curr: FeedbackCamera,
        prev: FeedbackCamera,
    ) -> SampledSlot {
        let write = self.write;
        let data = fa.frame_alloc(FeedbackData {
            input_texture_id: input_slot.index(),
            history_texture_id: self.slots[1 - write].index(),
            sampler_id: sampler.index(),
            sample_history: self.primed as u32,
            decay,
            floor,
            flow,
            _pad: 0,
            curr,
            prev,
        });
        gpu.cmd_begin_render_pass(
            cb,
            RenderPassDesc {
                color_attachments: &[RenderAttachment {
                    texture: self.targets[write].texture,
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
        self.primed = true;
        self.write = 1 - write;
        self.slots[write]
    }

    /// The most recently written accumulator — for readback/dump paths;
    /// shader access goes through [`Self::record`]'s returned slot.
    pub fn latest_texture(&self) -> gpu::Texture {
        assert!(self.primed, "no record has written an accumulator yet");
        self.targets[1 - self.write].texture
    }
}

impl Pass for FeedbackPass {
    const NAME: &'static str = "feedback";

    fn free(self, gpu: &Gpu) {
        for tex in self.targets {
            gpu.texture_free_and_destroy(tex);
        }
        gpu.shader_destroy(self.fullscreen_vert);
        gpu.shader_destroy(self.frag);
        gpu.free(self.indices);
    }
}
