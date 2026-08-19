use abi_core::GraphicsPush;
use abi_post::{
    BLOOM_COORDS, BLOOM_TAPS, BloomDownsampleData, BloomUpsampleData, TENT_COORDS, TENT_TAPS,
    TonemapTonyData, bloom_average_partial, bloom_tent_sum, bloom_upsample_blend,
    bloom_weighted_sum, safe_hdr, soft_threshold, tony_encode, tony_taps,
};
use glam::{Vec2, Vec3, Vec4};
use spirv_std::image::Image2d;
use spirv_std::spirv;
use spirv_std::{RuntimeArray, Sampler};

/// Generates the fullscreen triangle from three vertex indices.
/// Winding is CCW in +Y-down NDC; UV origin is top-left.
#[spirv(vertex)]
pub fn fullscreen_vert(
    #[spirv(vertex_index)] vert_id: i32,
    #[spirv(position)] out_pos: &mut Vec4,
    out_uv: &mut Vec2,
) {
    let uv = Vec2::new((vert_id & 2) as f32, ((vert_id & 1) << 1) as f32);
    *out_uv = uv;
    *out_pos = Vec4::new(uv.x * 2.0 - 1.0, uv.y * 2.0 - 1.0, 0.0, 1.0);
}

/// Applies the Tony McMapface display transform and final presentation.
/// R/G use bilinear strip taps; B uses manual interpolation.
#[spirv(fragment)]
pub fn tony_frag(
    #[spirv(push_constant)] push: &GraphicsPush,
    #[spirv(descriptor_set = 0, binding = 0)] textures: &RuntimeArray<Image2d>,
    #[spirv(descriptor_set = 2, binding = 0)] samplers: &RuntimeArray<Sampler>,
    #[spirv(frag_coord)] frag_coord: Vec4,
    uv: Vec2,
    out_color: &mut Vec4,
) {
    let data = push.frag::<TonemapTonyData>();
    let hdr_image = unsafe { textures.index(data.hdr_texture_id as usize) };
    let hdr_sampler = *unsafe { samplers.index(data.hdr_sampler_id as usize) };
    let hdr: Vec4 = hdr_image.sample_by_lod(hdr_sampler, uv, 0.0);

    // Bloom is scene light and precedes exposure; slot zero disables it.
    let mut scene = hdr.truncate();
    if data.bloom_texture_id != 0 {
        let bloom_image = unsafe { textures.index(data.bloom_texture_id as usize) };
        let bloom: Vec4 = bloom_image.sample_by_lod(hdr_sampler, uv, 0.0);
        scene += bloom.truncate() * data.bloom_intensity;
    }

    // Apply scene-referred exposure before display encoding.
    let exposed = scene * data.exposure;
    let taps = tony_taps(tony_encode(exposed));
    let lut = unsafe { textures.index(data.lut_texture_id as usize) };
    let lut_sampler = *unsafe { samplers.index(data.lut_sampler_id as usize) };
    let low: Vec4 = lut.sample_by_lod(lut_sampler, taps.uv_low, 0.0);
    let high: Vec4 = lut.sample_by_lod(lut_sampler, taps.uv_high, 0.0);
    let mapped = low.truncate().lerp(high.truncate(), taps.b_frac);
    // Encode sRGB, then dither in quantizer space before UNORM storage.
    let mut encoded = abi_post::srgb_encode(mapped);
    encoded += Vec3::splat(
        data.dither_strength * abi_post::dither_tri(frag_coord.truncate().truncate().as_uvec2()),
    );
    *out_color = encoded.extend(1.0);
}

/// Bloom downsample (Froyok / CoD:AW 13-tap). Samples through the set-2
/// sampler heap; `abi_post` owns the tap pattern, Karis average, weighted
/// sum, and soft threshold, while this entry only samples and combines.
#[spirv(fragment)]
pub fn bloom_downsample(
    #[spirv(push_constant)] push: &GraphicsPush,
    #[spirv(descriptor_set = 0, binding = 0)] textures: &RuntimeArray<Image2d>,
    #[spirv(descriptor_set = 2, binding = 0)] samplers: &RuntimeArray<Sampler>,
    uv: Vec2,
    out_color: &mut Vec4,
) {
    let data = push.frag::<BloomDownsampleData>();
    let pixel_size = Vec2::from_array(data.pixel_size) * Vec2::from_array(data.bloom_scale);
    let image = unsafe { textures.index(data.src_texture_id as usize) };
    let sampler = *unsafe { samplers.index(data.src_sampler_id as usize) };

    let mut samples = [Vec3::ZERO; BLOOM_TAPS];
    let mut i = 0;
    while i < BLOOM_TAPS {
        let tap_uv = uv + BLOOM_COORDS[i] * pixel_size;
        samples[i] = image.sample_by_lod(sampler, tap_uv, 0.0).truncate();
        i += 1;
    }

    let mut color = if data.use_anti_flicker != 0 {
        bloom_average_partial(&samples) // Karis-weighted first pass.
    } else {
        bloom_weighted_sum(&samples)
    };
    if data.bloom_threshold > 0.0 {
        color = soft_threshold(color, data.bloom_threshold, data.bloom_knee);
    }
    *out_color = safe_hdr(color).extend(1.0);
}

/// Bloom upsample (Froyok 9-tap tent): blends the current-resolution
/// downsample with the tent-filtered smaller level. The tent spreads at
/// destination resolution, while downsample spreads at source resolution;
/// that intentional asymmetry is the filter.
#[spirv(fragment)]
pub fn bloom_upsample(
    #[spirv(push_constant)] push: &GraphicsPush,
    #[spirv(descriptor_set = 0, binding = 0)] textures: &RuntimeArray<Image2d>,
    #[spirv(descriptor_set = 2, binding = 0)] samplers: &RuntimeArray<Sampler>,
    uv: Vec2,
    out_color: &mut Vec4,
) {
    let data = push.frag::<BloomUpsampleData>();
    let pixel_size = Vec2::from_array(data.pixel_size) * Vec2::from_array(data.bloom_scale);
    let sampler = *unsafe { samplers.index(data.sampler_id as usize) };

    let current_image = unsafe { textures.index(data.downsample_texture_id as usize) };
    let current: Vec4 = current_image.sample_by_lod(sampler, uv, 0.0);

    let previous_image = unsafe { textures.index(data.previous_texture_id as usize) };
    let mut taps = [Vec3::ZERO; TENT_TAPS];
    let mut i = 0;
    while i < TENT_TAPS {
        let tap_uv = uv + TENT_COORDS[i] * pixel_size;
        taps[i] = previous_image
            .sample_by_lod(sampler, tap_uv, 0.0)
            .truncate();
        i += 1;
    }

    *out_color = bloom_upsample_blend(current.truncate(), bloom_tent_sum(&taps), data.blend_factor)
        .extend(1.0);
}
