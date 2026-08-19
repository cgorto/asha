//! Display-space jump-flood outline pass.
//!
//! Geometry supplies the silhouette mask; this pass owns its processing.

use abi_core::GpuPtr;
use abi_core::glam::UVec2;
use abi_post::{
    OUTLINE_GROUP_CAPACITY, OutlineCompositeData, OutlineGroup, OutlineJfaFloodData,
    OutlineJfaInitData,
};
use gpu::pass::{FrameAlloc, Pass};
use gpu::{
    BlendFactor, BlendOp, BlendState, CommandBuffer, Gpu, HazardFlags, HeapSlots, LoadOp,
    OwnedTexture, Queue, RenderAttachment, RenderPassDesc, SampledSlot, SamplerSlot,
    ShaderTypeGraphics, Stage, StorageSlot, StoreOp, Texture, TextureDesc, TextureFormat,
    TextureViewDesc, UsageFlags,
};

/// Screen-sized outline resources and distance-field stages.
///
/// The exposed mask accepts silhouettes from any producer.
pub struct OutlinePass {
    size: UVec2,
    mask: OwnedTexture,
    mask_slot: SampledSlot,
    jfa_a: OwnedTexture,
    jfa_a_slot: SampledSlot,
    jfa_a_rw: StorageSlot,
    jfa_b: OwnedTexture,
    jfa_b_slot: SampledSlot,
    jfa_b_rw: StorageSlot,
    init_shader: gpu::Shader,
    flood_shader: gpu::Shader,
    fullscreen_vert: gpu::Shader,
    composite_frag: gpu::Shader,
    indices: gpu::Ptr<u32>,
}

impl OutlinePass {
    /// Create resources and reserve stable bindless slots.
    ///
    /// Resizing rewrites descriptors without changing the dispatch ABI.
    pub fn new(gpu: &Gpu, heap: &mut HeapSlots, size: UVec2) -> Self {
        assert!(size.x > 0 && size.y > 0);
        let mask_slot = heap.alloc_sampled();
        let jfa_a_slot = heap.alloc_sampled();
        let jfa_a_rw = heap.alloc_storage();
        let jfa_b_slot = heap.alloc_sampled();
        let jfa_b_rw = heap.alloc_storage();
        let (mask, jfa_a, jfa_b) = Self::create_textures(
            gpu, heap, size, mask_slot, jfa_a_slot, jfa_a_rw, jfa_b_slot, jfa_b_rw,
        );

        let indices = gpu.fullscreen_triangle_indices();

        Self {
            size,
            mask,
            mask_slot,
            jfa_a,
            jfa_a_slot,
            jfa_a_rw,
            jfa_b,
            jfa_b_slot,
            jfa_b_rw,
            init_shader: gpu.shader_create_compute(
                &asha_assets::load_spv("outline_jfa_init"),
                8,
                8,
                1,
                "outline_jfa_init",
            ),
            flood_shader: gpu.shader_create_compute(
                &asha_assets::load_spv("outline_jfa_flood"),
                8,
                8,
                1,
                "outline_jfa_flood",
            ),
            fullscreen_vert: gpu.shader_create(
                &asha_assets::load_spv("fullscreen_vert"),
                ShaderTypeGraphics::Vertex,
                "fullscreen_vert",
            ),
            composite_frag: gpu.shader_create(
                &asha_assets::load_spv("outline_composite"),
                ShaderTypeGraphics::Fragment,
                "outline_composite",
            ),
            indices,
        }
    }

    fn create_textures(
        gpu: &Gpu,
        heap: &HeapSlots,
        size: UVec2,
        mask_slot: SampledSlot,
        jfa_a_slot: SampledSlot,
        jfa_a_rw: StorageSlot,
        jfa_b_slot: SampledSlot,
        jfa_b_rw: StorageSlot,
    ) -> (OwnedTexture, OwnedTexture, OwnedTexture) {
        let mask = gpu.texture_alloc_and_create(
            TextureDesc {
                dimensions: [size.x, size.y, 1],
                format: TextureFormat::R8Unorm,
                usage: UsageFlags::COLOR_ATTACHMENT | UsageFlags::SAMPLED,
                ..Default::default()
            },
            Queue::Main,
            None,
        );
        let jfa = || {
            gpu.texture_alloc_and_create(
                TextureDesc {
                    dimensions: [size.x, size.y, 1],
                    format: TextureFormat::Rgba16Float,
                    usage: UsageFlags::SAMPLED | UsageFlags::STORAGE,
                    ..Default::default()
                },
                Queue::Main,
                None,
            )
        };
        let jfa_a = jfa();
        let jfa_b = jfa();

        let sampled =
            |texture: Texture| gpu.texture_view_descriptor(texture, TextureViewDesc::default());
        let storage =
            |texture: Texture| gpu.texture_rw_view_descriptor(texture, TextureViewDesc::default());
        heap.write_sampled(gpu, mask_slot, sampled(mask.texture));
        heap.write_sampled(gpu, jfa_a_slot, sampled(jfa_a.texture));
        heap.write_storage(gpu, jfa_a_rw, storage(jfa_a.texture));
        heap.write_sampled(gpu, jfa_b_slot, sampled(jfa_b.texture));
        heap.write_storage(gpu, jfa_b_rw, storage(jfa_b.texture));

        (mask, jfa_a, jfa_b)
    }

    /// Rebuild screen-sized images after the main queue is idle.
    ///
    /// Retains slot indices and overwrites descriptors in place.
    pub fn resize(&mut self, gpu: &Gpu, heap: &HeapSlots, size: UVec2) {
        assert!(size.x > 0 && size.y > 0);
        if self.size == size {
            return;
        }

        gpu.queue_wait_idle(Queue::Main);
        let (mask, jfa_a, jfa_b) = Self::create_textures(
            gpu,
            heap,
            size,
            self.mask_slot,
            self.jfa_a_slot,
            self.jfa_a_rw,
            self.jfa_b_slot,
            self.jfa_b_rw,
        );
        let old_mask = core::mem::replace(&mut self.mask, mask);
        let old_jfa_a = core::mem::replace(&mut self.jfa_a, jfa_a);
        let old_jfa_b = core::mem::replace(&mut self.jfa_b, jfa_b);
        gpu.texture_free_and_destroy(old_mask);
        gpu.texture_free_and_destroy(old_jfa_a);
        gpu.texture_free_and_destroy(old_jfa_b);
        self.size = size;
    }

    /// The R8 silhouette target. The external mesh silhouette pass clears
    /// and renders this as a color attachment before [`Self::record`].
    pub fn mask_texture(&self) -> Texture {
        self.mask.texture
    }

    fn initial_step(size: UVec2, max_radius: f32) -> i32 {
        let extent = size.max_element();
        assert!(extent > 0);
        assert!(max_radius.is_finite() && max_radius >= 0.0);
        // Retain one step for 1×N regions.
        let mut step = 1u32;
        while step.saturating_mul(2) < extent {
            step *= 2;
        }
        // Limit flooding to the largest requested outline radius.
        let radius = (max_radius + 1.0).ceil() as u32;
        let mut clamped = 1u32;
        while clamped < radius {
            clamped *= 2;
        }
        step.min(clamped) as i32
    }

    /// Record the JFA init, full-screen ping-pong flood, and alpha-blended
    /// display composite. Contract: the caller has just rendered the
    /// silhouette mask into [`Self::mask_texture`] as a color attachment.
    pub fn record(
        &self,
        gpu: &Gpu,
        cb: CommandBuffer,
        fa: &mut impl FrameAlloc,
        heap: &HeapSlots,
        display: Texture,
        sampler: SamplerSlot,
        groups: &[OutlineGroup],
    ) {
        assert_eq!(
            display.dimensions[0], self.size.x,
            "display width must match outline mask"
        );
        assert_eq!(
            display.dimensions[1], self.size.y,
            "display height must match outline mask"
        );
        assert!(
            !groups.is_empty(),
            "outline composite requires at least one group"
        );
        assert!(
            groups.len() <= OUTLINE_GROUP_CAPACITY as usize,
            "outline group count exceeds fixed GPU table"
        );

        heap.bind(gpu, cb);

        // Order mask writes before JFA reads.
        gpu.cmd_barrier(
            cb,
            Stage::RasterColorOut,
            Stage::Compute,
            HazardFlags::COLOR_ATTACHMENT,
        );
        gpu.cmd_set_compute_shader(cb, self.init_shader);
        let init = fa.frame_alloc(OutlineJfaInitData {
            mask_texture_id: self.mask_slot.index(),
            output_a_id: self.jfa_a_rw.index(),
            output_b_id: self.jfa_b_rw.index(),
            size: self.size.to_array(),
            ..Default::default()
        });
        gpu.cmd_dispatch(
            cb,
            init,
            self.size.x.div_ceil(8),
            self.size.y.div_ceil(8),
            1,
        );

        // Order initialization before the first flood.
        gpu.cmd_barrier(
            cb,
            Stage::Compute,
            Stage::Compute,
            HazardFlags::SHADER_IMAGE,
        );
        gpu.cmd_set_compute_shader(cb, self.flood_shader);

        let max_width = groups.iter().fold(0.0f32, |w, g| w.max(g.width));
        let region_offset = [0, 0];
        let region_size = self.size.to_array();
        let mut step = Self::initial_step(self.size, max_width);
        let mut input_slot = self.jfa_a_slot;
        let mut output_slot = self.jfa_b_rw;
        let final_slot = loop {
            let flood = fa.frame_alloc(OutlineJfaFloodData {
                input_texture_id: input_slot.index(),
                output_texture_id: output_slot.index(),
                step_size: step,
                size: self.size.to_array(),
                region_offset,
                region_size,
                ..Default::default()
            });
            gpu.cmd_dispatch(
                cb,
                flood,
                region_size[0].div_ceil(8),
                region_size[1].div_ceil(8),
                1,
            );

            if step == 1 {
                break if output_slot == self.jfa_a_rw {
                    self.jfa_a_slot
                } else {
                    self.jfa_b_slot
                };
            }

            // Order each flood before the next.
            gpu.cmd_barrier(
                cb,
                Stage::Compute,
                Stage::Compute,
                HazardFlags::SHADER_IMAGE,
            );
            step /= 2;
            if input_slot == self.jfa_a_slot {
                input_slot = self.jfa_b_slot;
                output_slot = self.jfa_a_rw;
            } else {
                input_slot = self.jfa_a_slot;
                output_slot = self.jfa_b_rw;
            }
        };

        // Order the final flood before compositing.
        gpu.cmd_barrier(
            cb,
            Stage::Compute,
            Stage::FragmentShader,
            HazardFlags::SHADER_IMAGE,
        );

        let mut group_table = [OutlineGroup::default(); OUTLINE_GROUP_CAPACITY as usize];
        group_table[..groups.len()].copy_from_slice(groups);
        let composite = fa.frame_alloc(OutlineCompositeData {
            jfa_texture_id: final_slot.index(),
            mask_texture_id: self.mask_slot.index(),
            sampler_id: sampler.index(),
            group_count: groups.len() as u32,
            screen_size: self.size.to_array(),
            region_min: region_offset,
            region_max: region_size,
            groups: group_table,
        });
        gpu.cmd_begin_render_pass(
            cb,
            RenderPassDesc {
                color_attachments: &[RenderAttachment {
                    texture: display,
                    load_op: LoadOp::Load,
                    store_op: StoreOp::Store,
                    clear_color: [0.0; 4],
                    ..Default::default()
                }],
                ..Default::default()
            },
        );
        gpu.cmd_set_shaders(cb, self.fullscreen_vert, self.composite_frag);
        gpu.cmd_set_blend_state(
            cb,
            BlendState {
                enable: true,
                color_op: BlendOp::Add,
                src_color_factor: BlendFactor::SrcAlpha,
                dst_color_factor: BlendFactor::OneMinusSrcAlpha,
                alpha_op: BlendOp::Add,
                src_alpha_factor: BlendFactor::One,
                dst_alpha_factor: BlendFactor::OneMinusSrcAlpha,
                color_write_mask: 0x0f,
            },
        );
        gpu.cmd_draw_indexed_instanced(
            cb,
            GpuPtr::null(),
            composite.cast(),
            self.indices.cast(),
            3,
            1,
        );
        gpu.cmd_end_render_pass(cb);
    }
}

impl Pass for OutlinePass {
    const NAME: &'static str = "outline";

    fn free(self, gpu: &Gpu) {
        gpu.texture_free_and_destroy(self.mask);
        gpu.texture_free_and_destroy(self.jfa_a);
        gpu.texture_free_and_destroy(self.jfa_b);
        gpu.shader_destroy(self.init_shader);
        gpu.shader_destroy(self.flood_shader);
        gpu.shader_destroy(self.fullscreen_vert);
        gpu.shader_destroy(self.composite_frag);
        gpu.free(self.indices);
    }
}
