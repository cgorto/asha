//! Optical projection remap with chromatic aberration and vignette.
//!
//! The effects share one radial resample to limit filtering loss. Apply this
//! pass after tonemapping; scene-space reconstruction must precede the warp.

use abi_core::GpuPtr;
use abi_core::glam::UVec2;
use abi_post::LensData;
use gpu::{
    CommandBuffer, Gpu, HazardFlags, HeapSlots, LoadOp, OwnedTexture, Queue, RenderAttachment,
    RenderPassDesc, SampledSlot, SamplerSlot, ShaderTypeGraphics, Stage, StoreOp, TextureDesc,
    TextureFormat, TextureViewDesc, UsageFlags,
};

use gpu::pass::{FrameAlloc, Pass};

/// Lens parameters grouped for a single pass configuration.
#[derive(Clone, Copy, Debug)]
pub struct LensSettings {
    /// Tangent of the source render's vertical half-FOV.
    pub tan_half_fov_src: f32,
    /// Presented-to-rendered field ratio. One preserves the rendered field;
    /// below one zooms in, while above one requests unrendered rays and may
    /// produce out-of-range UVs for the black surround.
    pub field_scale: f32,
    /// 0 = rectilinear, 1 = cylindrical (world verticals stay straight).
    pub cylindrical: f32,
    /// 0 = off, 1 = equidistant fisheye (everything bends).
    pub fisheye: f32,
    /// Red-channel UV shift at the image corner.
    pub ca_strength: f32,
    /// Spectral taps along the CA displacement; 0 disables. Even counts
    /// integrate to neutral white.
    pub ca_taps: u32,
    /// Blend toward the physical cos⁴ corner falloff.
    pub vignette: f32,
    /// Exponent on that falloff. 1.0 is the physical law.
    pub vignette_power: f32,
    /// Vignette-mask offset in normalized `p`-space.
    pub shake: [f32; 2],
}

impl Default for LensSettings {
    fn default() -> Self {
        Self {
            tan_half_fov_src: 1.0,
            field_scale: 1.0,
            cylindrical: 0.0,
            fisheye: 0.0,
            ca_strength: 0.0,
            ca_taps: 0,
            vignette: 0.0,
            vignette_power: 1.0,
            shake: [0.0; 2],
        }
    }
}

impl LensSettings {
    /// True when the lens would be an identity resample: rectilinear, the
    /// presented field equal to the rendered one, no fringe, no falloff.
    /// The host should skip the record entirely in that case.
    pub fn is_identity(&self) -> bool {
        self.cylindrical == 0.0
            && self.fisheye == 0.0
            && self.vignette == 0.0
            && (self.ca_taps == 0 || self.ca_strength == 0.0)
            && (self.field_scale - 1.0).abs() < 1e-6
            && self.shake == [0.0; 2]
    }
}

pub struct LensPass {
    size: UVec2,
    target: OwnedTexture,
    slot: SampledSlot,
    fullscreen_vert: gpu::Shader,
    frag: gpu::Shader,
    indices: gpu::Ptr<u32>,
}

impl LensPass {
    /// Rebuild resources when `size` changes.
    ///
    /// Waits for the queue before freeing resources and preserves the slot.
    /// Returns whether resources were rebuilt.
    pub fn ensure(this: &mut Option<Self>, gpu: &Gpu, heap: &mut HeapSlots, size: UVec2) -> bool {
        assert!(size.x > 0 && size.y > 0);
        if this.as_ref().is_some_and(|l| l.size == size) {
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
                &asha_assets::load_spv("lens_frag"),
                ShaderTypeGraphics::Fragment,
                "lens_frag",
            ),
            indices,
        });
        true
    }

    /// Record the lens over `input_slot`. Contract: the input is sampleable
    /// when this records; ends with the returned slot (the remapped image —
    /// feed it to consumers in the input's place) sampleable.
    pub fn record(
        &self,
        gpu: &Gpu,
        cb: CommandBuffer,
        fa: &mut impl FrameAlloc,
        input_slot: SampledSlot,
        sampler: SamplerSlot,
        settings: &LensSettings,
    ) -> SampledSlot {
        let data = fa.frame_alloc(LensData {
            input_texture_id: input_slot.index(),
            sampler_id: sampler.index(),
            tan_half_fov_src: settings.tan_half_fov_src,
            field_scale: settings.field_scale,
            aspect: self.size.x as f32 / self.size.y as f32,
            cylindrical: settings.cylindrical,
            fisheye: settings.fisheye,
            ca_strength: settings.ca_strength,
            ca_taps: settings.ca_taps,
            vignette: settings.vignette,
            vignette_power: settings.vignette_power,
            shake: settings.shake,
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

    /// The remapped target — for readback/dump paths; shader access goes
    /// through [`Self::record`]'s returned slot.
    pub fn texture(&self) -> gpu::Texture {
        self.target.texture
    }
}

impl Pass for LensPass {
    const NAME: &'static str = "lens";

    fn free(self, gpu: &Gpu) {
        gpu.texture_free_and_destroy(self.target);
        gpu.shader_destroy(self.fullscreen_vert);
        gpu.shader_destroy(self.frag);
        gpu.free(self.indices);
    }
}
