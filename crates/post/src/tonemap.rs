//! HDR-to-swapchain display transform using the Tony McMapface LUT.

use abi_core::GpuPtr;
use abi_post::{TONY_LUT_HEIGHT, TONY_LUT_WIDTH, TonemapTonyData};
use gpu::pass::{FrameAlloc, Pass};
use gpu::{
    CommandBuffer, Gpu, LoadOp, Memory, OwnedTexture, Queue, RenderAttachment, RenderPassDesc,
    SampledSlot, SamplerSlot, ShaderTypeGraphics, StoreOp, Texture, TextureDesc, TextureFormat,
    TextureViewDesc, UsageFlags,
};

/// Display transform resources and its LUT staging buffer.
///
/// Upload with [`Self::upload`], then release staging after completion.
pub struct TonemapPass {
    lut: OwnedTexture,
    lut_slot: SampledSlot,
    fullscreen_vert: gpu::Shader,
    tony_frag: gpu::Shader,
    indices: gpu::Ptr<u32>,
    /// Staging remains alive until the upload completes.
    staging: gpu::Ptr<u8>,
}

impl TonemapPass {
    pub fn new(gpu: &Gpu, heap: &mut gpu::HeapSlots) -> Self {
        let lut_bytes = std::fs::read(asha_assets::asset_path(
            "luts/tony_mc_mapface_2304x48_rgba16f.bin",
        ))
        .expect("tony LUT asset");
        let lut = gpu.texture_alloc_and_create(
            TextureDesc {
                dimensions: [TONY_LUT_WIDTH, TONY_LUT_HEIGHT, 1],
                format: TextureFormat::Rgba16Float,
                usage: UsageFlags::SAMPLED | UsageFlags::TRANSFER_DST,
                ..Default::default()
            },
            Queue::Main,
            None,
        );
        let staging = gpu.alloc_slice::<u8>(lut_bytes.len() as u64, Memory::Default);
        // SAFETY: allocation matches the source length.
        unsafe {
            std::ptr::copy_nonoverlapping(lut_bytes.as_ptr(), staging.cpu, lut_bytes.len());
        }
        let indices = gpu.fullscreen_triangle_indices();
        Self {
            lut_slot: heap.add_sampled(
                gpu,
                gpu.texture_view_descriptor(lut.texture, TextureViewDesc::default()),
            ),
            lut,
            fullscreen_vert: gpu.shader_create(
                &asha_assets::load_spv("fullscreen_vert"),
                ShaderTypeGraphics::Vertex,
                "fullscreen_vert",
            ),
            tony_frag: gpu.shader_create(
                &asha_assets::load_spv("tony_frag"),
                ShaderTypeGraphics::Fragment,
                "tony_frag",
            ),
            indices,
            staging,
        }
    }

    /// Record the LUT upload into the host's one-time setup submit.
    pub fn upload(&self, gpu: &Gpu, cb: CommandBuffer) {
        assert!(!self.staging.is_null(), "upload records once");
        gpu.cmd_copy_to_texture(cb, self.lut.texture, self.staging);
    }

    /// Release the staging copy. Contract: the `upload` submit has
    /// completed (the host's setup wait-idle).
    pub fn upload_finish(&mut self, gpu: &Gpu) {
        assert!(
            !self.staging.is_null(),
            "upload_finish follows upload, once"
        );
        gpu.free(self.staging);
        self.staging = gpu::Ptr::null();
    }

    /// Tonemap `hdr_slot` into `backbuffer`.
    /// (LoadOp::DontCare — every pixel is written). `sampler` filters both
    /// the HDR source and the LUT; `exposure` is the scene-referred
    /// multiplier applied to the HDR before the tony transform (1.0 =
    /// neutral); `dither_strength` is in output-quant units (1/255 for an
    /// 8-bit swapchain; 0 disables). `bloom` is the bloom chain's final
    /// level and its intensity, composited into the scene before exposure;
    /// None writes the slot-0 sentinel (no bloom).
    #[allow(clippy::too_many_arguments)]
    pub fn record(
        &self,
        gpu: &Gpu,
        cb: CommandBuffer,
        fa: &mut impl FrameAlloc,
        backbuffer: Texture,
        hdr_slot: SampledSlot,
        sampler: SamplerSlot,
        exposure: f32,
        dither_strength: f32,
        bloom: Option<(SampledSlot, f32)>,
    ) {
        let (bloom_texture_id, bloom_intensity) =
            bloom.map_or((0, 0.0), |(slot, intensity)| (slot.index(), intensity));
        let data = fa.frame_alloc(TonemapTonyData {
            hdr_texture_id: hdr_slot.index(),
            hdr_sampler_id: sampler.index(),
            lut_texture_id: self.lut_slot.index(),
            lut_sampler_id: sampler.index(),
            dither_strength,
            exposure,
            bloom_texture_id,
            bloom_intensity,
        });
        gpu.cmd_begin_render_pass(
            cb,
            RenderPassDesc {
                color_attachments: &[RenderAttachment {
                    texture: backbuffer,
                    load_op: LoadOp::DontCare,
                    store_op: StoreOp::Store,
                    clear_color: [0.0; 4],
                    ..Default::default()
                }],
                ..Default::default()
            },
        );
        gpu.cmd_set_shaders(cb, self.fullscreen_vert, self.tony_frag);
        gpu.cmd_draw_indexed_instanced(cb, GpuPtr::null(), data.cast(), self.indices.cast(), 3, 1);
        gpu.cmd_end_render_pass(cb);
    }
}

impl Pass for TonemapPass {
    const NAME: &'static str = "tony";

    fn free(self, gpu: &Gpu) {
        gpu.texture_free_and_destroy(self.lut);
        gpu.shader_destroy(self.fullscreen_vert);
        gpu.shader_destroy(self.tony_frag);
        gpu.free(self.indices);
        if !self.staging.is_null() {
            gpu.free(self.staging);
        }
    }
}
