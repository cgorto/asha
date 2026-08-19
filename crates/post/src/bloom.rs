//! Progressive bloom pyramid with thresholded downsampling and tent upsampling.
//!
//! Each level uses a color attachment. The final level feeds tonemapping.

use abi_core::GpuPtr;
use abi_core::glam::UVec2;
use abi_post::{BloomDownsampleData, BloomUpsampleData};
use gpu::{
    CommandBuffer, Gpu, HazardFlags, HeapSlots, LoadOp, OwnedTexture, Queue, RenderAttachment,
    RenderPassDesc, SampledSlot, SamplerSlot, ShaderTypeGraphics, Stage, StoreOp, TextureDesc,
    TextureFormat, TextureViewDesc, UsageFlags,
};

use gpu::pass::{FrameAlloc, Pass};

/// Maximum number of bloom pyramid levels.
pub const MAX_BLOOM_MIPS: usize = 12;

/// Returns the useful pyramid depth for `size`.
fn mip_count(size: UVec2) -> u32 {
    let base = 31 - size.min_element().max(1).leading_zeros();
    if base <= 3 {
        1
    } else {
        (base - 3).min(MAX_BLOOM_MIPS as u32)
    }
}

/// Returns pyramid level `i` dimensions.
fn mip_size(size: UVec2, i: u32) -> UVec2 {
    UVec2::new((size.x >> (i + 1)).max(1), (size.y >> (i + 1)).max(1))
}

pub struct BloomPass {
    size: UVec2,
    mip_count: u32,
    downsamples: Vec<OwnedTexture>,
    upsamples: Vec<OwnedTexture>,
    down_slots: Vec<SampledSlot>,
    up_slots: Vec<SampledSlot>,
    fullscreen_vert: gpu::Shader,
    down_frag: gpu::Shader,
    up_frag: gpu::Shader,
    indices: gpu::Ptr<u32>,
}

impl BloomPass {
    /// Rebuild resources when `size` changes.
    ///
    /// Waits for the queue before freeing resources and preserves the slots.
    /// Returns whether resources were rebuilt.
    pub fn ensure(this: &mut Option<Self>, gpu: &Gpu, heap: &mut HeapSlots, size: UVec2) -> bool {
        assert!(size.x > 0 && size.y > 0);
        if this.as_ref().is_some_and(|b| b.size == size) {
            return false;
        }
        let (down_slots, up_slots) = match this.take() {
            Some(old) => {
                gpu.queue_wait_idle(Queue::Main);
                let slots = (old.down_slots.clone(), old.up_slots.clone());
                old.free(gpu);
                slots
            }
            None => (
                (0..MAX_BLOOM_MIPS).map(|_| heap.alloc_sampled()).collect(),
                (0..MAX_BLOOM_MIPS).map(|_| heap.alloc_sampled()).collect(),
            ),
        };

        let mips = mip_count(size);
        let level = |i: u32| {
            let dims = mip_size(size, i);
            gpu.texture_alloc_and_create(
                TextureDesc {
                    dimensions: [dims.x, dims.y, 1],
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
        let downsamples: Vec<OwnedTexture> = (0..mips).map(level).collect();
        let upsamples: Vec<OwnedTexture> = (0..mips).map(level).collect();
        let write = |slots: &Vec<SampledSlot>, texs: &Vec<OwnedTexture>| {
            for (slot, tex) in slots.iter().zip(texs) {
                heap.write_sampled(
                    gpu,
                    *slot,
                    gpu.texture_view_descriptor(tex.texture, TextureViewDesc::default()),
                );
            }
        };
        write(&down_slots, &downsamples);
        write(&up_slots, &upsamples);

        let indices = gpu.fullscreen_triangle_indices();
        *this = Some(Self {
            size,
            mip_count: mips,
            downsamples,
            upsamples,
            down_slots,
            up_slots,
            fullscreen_vert: gpu.shader_create(
                &asha_assets::load_spv("fullscreen_vert"),
                ShaderTypeGraphics::Vertex,
                "fullscreen_vert",
            ),
            down_frag: gpu.shader_create(
                &asha_assets::load_spv("bloom_downsample"),
                ShaderTypeGraphics::Fragment,
                "bloom_downsample",
            ),
            up_frag: gpu.shader_create(
                &asha_assets::load_spv("bloom_upsample"),
                ShaderTypeGraphics::Fragment,
                "bloom_upsample",
            ),
            indices,
        });
        true
    }

    /// The level the tonemap composites: the full up chain's output — or
    /// the lone downsample when the screen is too small to build one.
    pub fn final_slot(&self) -> SampledSlot {
        if self.mip_count > 1 {
            self.up_slots[0]
        } else {
            self.down_slots[0]
        }
    }

    /// The final level's dimensions (level 0 = half resolution) — for
    /// sizing downstream consumers of [`Self::final_slot`].
    pub fn final_size(&self) -> UVec2 {
        mip_size(self.size, 0)
    }

    /// The final level's texture — for readback/dump paths; shader access
    /// goes through [`Self::final_slot`].
    pub fn final_texture(&self) -> gpu::Texture {
        if self.mip_count > 1 {
            self.upsamples[0].texture
        } else {
            self.downsamples[0].texture
        }
    }

    /// Record the chain. `hdr_slot` must be sampleable.
    ///
    /// Ends with the final level sampleable by tonemapping.
    ///
    /// Threshold/knee apply on the FIRST downsample only — re-thresholding
    /// each mip would clip the fading edges and narrow the bloom.
    /// `scale` stretches the kernel independently per axis.
    #[allow(clippy::too_many_arguments)]
    pub fn record(
        &self,
        gpu: &Gpu,
        cb: CommandBuffer,
        fa: &mut impl FrameAlloc,
        hdr_slot: SampledSlot,
        sampler: SamplerSlot,
        threshold: f32,
        knee: f32,
        blend_factor: f32,
        scale: [f32; 2],
    ) {
        let draw = |data: GpuPtr<u8>, target: &OwnedTexture, frag: gpu::Shader| {
            gpu.cmd_begin_render_pass(
                cb,
                RenderPassDesc {
                    color_attachments: &[RenderAttachment {
                        texture: target.texture,
                        load_op: LoadOp::DontCare,
                        store_op: StoreOp::Store,
                        clear_color: [0.0; 4],
                        ..Default::default()
                    }],
                    ..Default::default()
                },
            );
            gpu.cmd_set_shaders(cb, self.fullscreen_vert, frag);
            gpu.cmd_draw_indexed_instanced(cb, GpuPtr::null(), data, self.indices.cast(), 3, 1);
            gpu.cmd_end_render_pass(cb);
            gpu.cmd_barrier(
                cb,
                Stage::RasterColorOut,
                Stage::FragmentShader,
                HazardFlags::COLOR_ATTACHMENT,
            );
        };

        for i in 0..self.mip_count {
            let src_size = if i == 0 {
                self.size
            } else {
                mip_size(self.size, i - 1)
            };
            let src_slot = if i == 0 {
                hdr_slot
            } else {
                self.down_slots[(i - 1) as usize]
            };
            let data = fa.frame_alloc(BloomDownsampleData {
                src_texture_id: src_slot.index(),
                src_sampler_id: sampler.index(),
                pixel_size: [1.0 / src_size.x as f32, 1.0 / src_size.y as f32],
                use_anti_flicker: (i == 0) as u32,
                bloom_threshold: if i == 0 { threshold } else { 0.0 },
                bloom_knee: if i == 0 { knee } else { 0.0 },
                bloom_scale: scale,
            });
            draw(data.cast(), &self.downsamples[i as usize], self.down_frag);
        }

        // Upsample from the smallest level.
        for j in 0..self.mip_count.saturating_sub(1) {
            let i = (self.mip_count - 2 - j) as usize;
            let previous = if j == 0 {
                self.down_slots[(self.mip_count - 1) as usize]
            } else {
                self.up_slots[i + 1]
            };
            let dst_size = mip_size(self.size, i as u32);
            let data = fa.frame_alloc(BloomUpsampleData {
                downsample_texture_id: self.down_slots[i].index(),
                previous_texture_id: previous.index(),
                sampler_id: sampler.index(),
                blend_factor,
                pixel_size: [1.0 / dst_size.x as f32, 1.0 / dst_size.y as f32],
                bloom_scale: scale,
            });
            draw(data.cast(), &self.upsamples[i], self.up_frag);
        }
    }
}

impl Pass for BloomPass {
    const NAME: &'static str = "bloom";

    fn free(self, gpu: &Gpu) {
        for tex in self.downsamples {
            gpu.texture_free_and_destroy(tex);
        }
        for tex in self.upsamples {
            gpu.texture_free_and_destroy(tex);
        }
        gpu.shader_destroy(self.fullscreen_vert);
        gpu.shader_destroy(self.down_frag);
        gpu.shader_destroy(self.up_frag);
        gpu.free(self.indices);
    }
}
