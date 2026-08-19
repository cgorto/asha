use abi_core::GraphicsPush;
use abi_post::{
    LUMA_709, SHARPEN_TAPS, SensorData, blue_noise, grain_cell, noise_split, rolling_shutter_uv,
    sensor_noise, sensor_shadow_weight, sharpen_combine,
};
use glam::{UVec2, Vec2, Vec3, Vec4};
use spirv_std::image::Image2d;
use spirv_std::spirv;
use spirv_std::{RuntimeArray, Sampler};

/// Applies rolling-shutter shear, sharpening, and HDR sensor noise.
/// Geometry and combination are defined in `abi_post`.
#[spirv(fragment)]
pub fn sensor_frag(
    #[spirv(push_constant)] push: &GraphicsPush,
    #[spirv(descriptor_set = 0, binding = 0)] textures: &RuntimeArray<Image2d>,
    #[spirv(descriptor_set = 2, binding = 0)] samplers: &RuntimeArray<Sampler>,
    uv: Vec2,
    out_color: &mut Vec4,
) {
    let data = push.frag::<SensorData>();
    let sampler = *unsafe { samplers.index(data.sampler_id as usize) };
    let image = unsafe { textures.index(data.input_texture_id as usize) };
    let res = Vec2::new(data.resolution[0], data.resolution[1]);
    let texel = Vec2::ONE / res.max(Vec2::ONE);

    // Sample all sensor effects from the sheared position.
    let src = rolling_shutter_uv(uv, data.yaw_rate, data.shutter, data.shutter_max);

    let center: Vec4 = image.sample_by_lod(sampler, src, 0.0);
    let mut color = center.truncate();

    if data.sharpen != 0.0 {
        let mut sum = Vec3::ZERO;
        let mut i = 0usize;
        while i < 4 {
            let tap: Vec4 = image.sample_by_lod(sampler, src + SHARPEN_TAPS[i] * texel, 0.0);
            sum += tap.truncate();
            i += 1;
        }
        color = sharpen_combine(color, sum, data.sharpen);
    }

    // Index read noise on a resolution-independent sensor grid.
    let cell = grain_cell(uv, data.aspect, data.grain_cells);
    let cell_u = UVec2::new(cell.x as u32, cell.y as u32);
    let luma = color.dot(LUMA_709);
    let weight = sensor_shadow_weight(luma, data.grain_shadow_bias);

    // Read noise is white and independently re-rolled each frame.
    let (mono, chroma) = noise_split(sensor_noise(cell_u, data.frame));
    color += Vec3::splat(mono * data.grain_luma * weight);
    color += chroma * (data.grain_chroma * weight);

    // Fixed-pattern noise uses static blue-noise coordinates.
    if data.grain_fixed != 0.0 {
        let fixed = blue_noise(cell_u) - 0.5;
        color += Vec3::splat(fixed * data.grain_fixed);
    }

    *out_color = color.max(Vec3::ZERO).extend(1.0);
}
