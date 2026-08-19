use abi_core::GraphicsPush;
use abi_post::{EXPOSURE_PROBE_TAPS, ExposureProbeData, exposure_log_luma, exposure_probe_uv};
use glam::{Vec2, Vec4};
use spirv_std::image::Image2d;
use spirv_std::spirv;
use spirv_std::{RuntimeArray, Sampler};

/// Reduces a sparse HDR frame sample grid to one log-average luminance.
/// The geometric mean suits multiplicative exposure and is not dominated by
/// one bright highlight. A fixed sparse grid is noisy per frame, so the
/// consumer settles slowly and quantizes; sub-step noise is not visible.
/// This cannot reuse bloom's smallest mip: bloom thresholds on its first
/// downsample, measuring highlight energy and becoming zero in a dim scene,
/// exactly where an exposure meter still needs a signal.
#[spirv(fragment)]
pub fn exposure_probe_frag(
    #[spirv(push_constant)] push: &GraphicsPush,
    #[spirv(descriptor_set = 0, binding = 0)] textures: &RuntimeArray<Image2d>,
    #[spirv(descriptor_set = 2, binding = 0)] samplers: &RuntimeArray<Sampler>,
    _uv: Vec2,
    out_color: &mut Vec4,
) {
    let data = push.frag::<ExposureProbeData>();
    let sampler = *unsafe { samplers.index(data.sampler_id as usize) };
    let image = unsafe { textures.index(data.input_texture_id as usize) };

    let mut sum = 0.0f32;
    let mut i = 0u32;
    while i < EXPOSURE_PROBE_TAPS {
        let uv = exposure_probe_uv(i);
        let s: Vec4 = image.sample_by_lod(sampler, uv, 0.0);
        sum += exposure_log_luma(s.truncate());
        i += 1;
    }

    let mean_log = sum / EXPOSURE_PROBE_TAPS as f32;
    *out_color = Vec4::new(mean_log, 0.0, 0.0, 1.0);
}
