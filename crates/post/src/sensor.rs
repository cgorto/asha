//! Sensor-stage effects applied after the lens and before tonemapping.
//!
//! Includes rolling shutter, sharpening, and digital read noise. Bloom should
//! sample the clean lens output so noise does not enter the bloom chain.

use abi_core::GpuPtr;
use abi_core::glam::UVec2;
use abi_post::SensorData;
use gpu::{
    CommandBuffer, Gpu, HazardFlags, HeapSlots, LoadOp, OwnedTexture, Queue, RenderAttachment,
    RenderPassDesc, SampledSlot, SamplerSlot, ShaderTypeGraphics, Stage, StoreOp, TextureDesc,
    TextureFormat, TextureViewDesc, UsageFlags,
};

use gpu::pass::{FrameAlloc, Pass};

#[derive(Clone, Copy, Debug, Default)]
pub struct SensorSettings {
    /// Monochrome read-noise amplitude, scene-referred.
    pub grain_luma: f32,
    /// Chromatic read-noise amplitude.
    pub grain_chroma: f32,
    /// Exponent concentrating noise into shadow.
    pub grain_shadow_bias: f32,
    /// Non-animating component: hot pixels and column offsets.
    pub grain_fixed: f32,
    /// Unsharp amount; 0 skips the four extra taps.
    pub sharpen: f32,
    /// Rolling-shutter response scale (dimensionless; feeds the saturation).
    pub shutter: f32,
    /// Maximum rolling-shutter shear in UV.
    pub shutter_max: f32,
    /// Grain-cell count across the frame height.
    pub grain_cells: f32,
    /// Camera yaw rate, radians/second.
    pub yaw_rate: f32,
    /// Per-frame animation salt; change it every frame.
    pub frame: u32,
}

impl SensorSettings {
    /// True when the pass would be a pure copy; the host should skip it.
    pub fn is_identity(&self) -> bool {
        self.grain_luma == 0.0
            && self.grain_chroma == 0.0
            && self.grain_fixed == 0.0
            && self.sharpen == 0.0
            && (self.shutter == 0.0 || self.shutter_max == 0.0 || self.yaw_rate == 0.0)
    }
}

pub struct SensorPass {
    size: UVec2,
    target: OwnedTexture,
    slot: SampledSlot,
    fullscreen_vert: gpu::Shader,
    frag: gpu::Shader,
    indices: gpu::Ptr<u32>,
}

impl SensorPass {
    /// Rebuild resources when `size` changes.
    ///
    /// Waits for the queue before freeing resources and preserves the slot.
    /// Returns whether resources were rebuilt.
    pub fn ensure(this: &mut Option<Self>, gpu: &Gpu, heap: &mut HeapSlots, size: UVec2) -> bool {
        assert!(size.x > 0 && size.y > 0);
        if this.as_ref().is_some_and(|s| s.size == size) {
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
                &asha_assets::load_spv("sensor_frag"),
                ShaderTypeGraphics::Fragment,
                "sensor_frag",
            ),
            indices,
        });
        true
    }

    /// Record the sensor stage over `input_slot`. Contract: the input is
    /// sampleable when this records; ends with the returned slot sampleable.
    pub fn record(
        &self,
        gpu: &Gpu,
        cb: CommandBuffer,
        fa: &mut impl FrameAlloc,
        input_slot: SampledSlot,
        sampler: SamplerSlot,
        settings: &SensorSettings,
    ) -> SampledSlot {
        let data = fa.frame_alloc(SensorData {
            input_texture_id: input_slot.index(),
            sampler_id: sampler.index(),
            resolution: [self.size.x as f32, self.size.y as f32],
            frame: settings.frame,
            grain_luma: settings.grain_luma,
            grain_chroma: settings.grain_chroma,
            grain_shadow_bias: settings.grain_shadow_bias,
            grain_fixed: settings.grain_fixed,
            sharpen: settings.sharpen,
            shutter: settings.shutter,
            shutter_max: settings.shutter_max,
            yaw_rate: settings.yaw_rate,
            grain_cells: settings.grain_cells,
            aspect: self.size.x as f32 / self.size.y as f32,
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

    /// The sensor target — for readback/dump paths; shader access goes
    /// through [`Self::record`]'s returned slot.
    pub fn texture(&self) -> gpu::Texture {
        self.target.texture
    }
}

impl Pass for SensorPass {
    const NAME: &'static str = "sensor";

    fn free(self, gpu: &Gpu) {
        gpu.texture_free_and_destroy(self.target);
        gpu.shader_destroy(self.fullscreen_vert);
        gpu.shader_destroy(self.frag);
        gpu.free(self.indices);
    }
}
