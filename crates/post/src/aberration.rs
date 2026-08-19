//! Lateral chromatic aberration over a sampled HDR source.
//!
//! Red and blue channels shift radially around screen center. The pass is
//! stateless; callers may skip it when strength is zero.

use abi_core::GpuPtr;
use abi_core::glam::UVec2;
use abi_post::AberrationData;
use gpu::{
    CommandBuffer, Gpu, HazardFlags, HeapSlots, LoadOp, OwnedTexture, Queue, RenderAttachment,
    RenderPassDesc, SampledSlot, SamplerSlot, ShaderTypeGraphics, Stage, StoreOp, TextureDesc,
    TextureFormat, TextureViewDesc, UsageFlags,
};

use gpu::pass::{FrameAlloc, Pass};

pub struct AberrationPass {
    size: UVec2,
    target: OwnedTexture,
    slot: SampledSlot,
    fullscreen_vert: gpu::Shader,
    frag: gpu::Shader,
    indices: gpu::Ptr<u32>,
}

impl AberrationPass {
    /// Rebuild resources when `size` changes.
    ///
    /// Waits for the queue before freeing resources and preserves the slot.
    /// Returns whether resources were rebuilt.
    pub fn ensure(this: &mut Option<Self>, gpu: &Gpu, heap: &mut HeapSlots, size: UVec2) -> bool {
        assert!(size.x > 0 && size.y > 0);
        if this.as_ref().is_some_and(|a| a.size == size) {
            return false;
        }
        let slot = match this.take() {
            Some(old) => {
                gpu.queue_wait_idle(Queue::Main);
                let slot = old.slot;
                old.free(gpu);
                slot
            }
            None => heap.alloc_sampled(),
        };

        let target = gpu.texture_alloc_and_create(
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
        );
        heap.write_sampled(
            gpu,
            slot,
            gpu.texture_view_descriptor(target.texture, TextureViewDesc::default()),
        );

        let indices = gpu.fullscreen_triangle_indices();
        *this = Some(Self {
            size,
            target,
            slot,
            fullscreen_vert: gpu.shader_create(
                &asha_assets::load_spv("fullscreen_vert"),
                ShaderTypeGraphics::Vertex,
                "fullscreen_vert",
            ),
            frag: gpu.shader_create(
                &asha_assets::load_spv("aberration_frag"),
                ShaderTypeGraphics::Fragment,
                "aberration_frag",
            ),
            indices,
        });
        true
    }

    /// Record the fringe over `input_slot`. Contract: the input is
    /// sampleable when this records; ends with the returned slot (the
    /// fringed image — feed it to consumers in the input's place)
    /// sampleable. `strength` is the red channel's UV shift at the image
    /// corner (`abi_post::ca_offset`; negative flips the fringe).
    pub fn record(
        &self,
        gpu: &Gpu,
        cb: CommandBuffer,
        fa: &mut impl FrameAlloc,
        input_slot: SampledSlot,
        sampler: SamplerSlot,
        strength: f32,
    ) -> SampledSlot {
        let data = fa.frame_alloc(AberrationData {
            input_texture_id: input_slot.index(),
            sampler_id: sampler.index(),
            strength,
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
            Stage::FragmentShader,
            HazardFlags::COLOR_ATTACHMENT,
        );
        self.slot
    }

    /// The fringed target — for readback/dump paths; shader access goes
    /// through [`Self::record`]'s returned slot.
    pub fn texture(&self) -> gpu::Texture {
        self.target.texture
    }
}

impl Pass for AberrationPass {
    const NAME: &'static str = "aberration";

    fn free(self, gpu: &Gpu) {
        gpu.texture_free_and_destroy(self.target);
        gpu.shader_destroy(self.fullscreen_vert);
        gpu.shader_destroy(self.frag);
        gpu.free(self.indices);
    }
}
