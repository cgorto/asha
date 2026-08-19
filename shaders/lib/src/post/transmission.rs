use abi_core::GraphicsPush;
use abi_post::{
    LUMA_709, TransmissionData, chroma_recombine, chroma_tap_offset, dither_tri, dropout,
    roll_seam, roll_uv, signal_wander, snow, tear_band, transmission_radial,
};
use glam::{UVec2, Vec2, Vec3, Vec4};
use spirv_std::image::Image2d;
use spirv_std::spirv;
use spirv_std::{RuntimeArray, Sampler};

/// Applies continuous and gated transmission artifacts after display encoding.
/// Radial scaling preserves the optical axis; roll uses its own gate.
/// Final quantization dither is applied here.
#[spirv(fragment)]
pub fn transmission_frag(
    #[spirv(push_constant)] push: &GraphicsPush,
    #[spirv(descriptor_set = 0, binding = 0)] textures: &RuntimeArray<Image2d>,
    #[spirv(descriptor_set = 2, binding = 0)] samplers: &RuntimeArray<Sampler>,
    uv: Vec2,
    out_color: &mut Vec4,
) {
    let data = push.frag::<TransmissionData>();
    let sampler = *unsafe { samplers.index(data.sampler_id as usize) };
    let image = unsafe { textures.index(data.input_texture_id as usize) };
    let res = Vec2::new(data.resolution[0], data.resolution[1]);

    let radial = transmission_radial(uv);

    // Compute radial displacement before sampling.
    let tear = tear_band(
        uv.y,
        radial,
        data.drive,
        data.frame,
        data.tear_bands,
        data.tear_offset,
    );
    let torn = Vec2::new(uv.x + tear.offset, tear.hold_y);
    // Apply roll after other displacement.
    let src = roll_uv(torn, data.roll);

    let center: Vec4 = image.sample_by_lod(sampler, src, 0.0);
    let mut color = center.truncate();

    // Average trailing chroma taps while preserving center-pixel luma.
    let desync = data.chroma_desync * data.drive * radial;
    if data.chroma_taps > 1 && (data.chroma_width != 0.0 || desync != 0.0) {
        let sharp_luma = color.dot(LUMA_709);
        let mut sum = Vec3::ZERO;
        let mut i = 0u32;
        while i < data.chroma_taps {
            let dx = chroma_tap_offset(i, data.chroma_taps, data.chroma_width) + desync;
            let tap: Vec4 = image.sample_by_lod(sampler, Vec2::new(src.x + dx, src.y), 0.0);
            sum += tap.truncate();
            i += 1;
        }
        let smeared = sum / data.chroma_taps as f32;
        color = chroma_recombine(smeared, sharp_luma);
    }

    // Dropouts and seams add signal; they do not displace the image.
    color += Vec3::splat(
        dropout(
            uv,
            data.drive,
            data.frame,
            data.dropout_rows,
            data.dropout_length,
            data.dropout_gain,
        ) * radial,
    );
    color += Vec3::splat(roll_seam(uv.y, data.roll, data.seam_width) * data.seam_gain);

    // Add radially weighted snow from link quality and motion.
    let wander = 1.0 + (signal_wander(data.wander_t) - 0.5) * 2.0 * data.snow_wander;
    let amount =
        (data.snow_base * wander + data.snow_accel * data.accel + data.snow_shock * data.drive)
            * (0.35 + 0.65 * radial);
    if amount != 0.0 {
        let px = UVec2::new((uv.x * res.x) as u32, (uv.y * res.y) as u32);
        color += Vec3::splat(snow(px, data.frame) * amount);
    }

    // Apply the sole final-output quantization dither.
    if data.dither_step != 0.0 {
        let px = UVec2::new((uv.x * res.x) as u32, (uv.y * res.y) as u32);
        color += Vec3::splat(dither_tri(px) * data.dither_step);
    }

    *out_color = color.max(Vec3::ZERO).extend(1.0);
}
