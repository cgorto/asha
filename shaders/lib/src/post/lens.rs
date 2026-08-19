use abi_core::GraphicsPush;
use abi_post::{
    LensData, ca_offset, ca_spectral_weight, ca_tap_offset, lens_ray_angle, lens_source_uv,
    lens_vignette,
};
use glam::{Vec2, Vec3, Vec4};
use spirv_std::image::Image2d;
use spirv_std::spirv;
use spirv_std::{RuntimeArray, Sampler};

/// Applies radial projection remapping, chromatic aberration, and cos⁴ vignette.
/// One resample avoids compounding radial warp softening.
/// Geometry is defined in `abi_post`.
#[spirv(fragment)]
pub fn lens_frag(
    #[spirv(push_constant)] push: &GraphicsPush,
    #[spirv(descriptor_set = 0, binding = 0)] textures: &RuntimeArray<Image2d>,
    #[spirv(descriptor_set = 2, binding = 0)] samplers: &RuntimeArray<Sampler>,
    uv: Vec2,
    out_color: &mut Vec4,
) {
    let data = push.frag::<LensData>();
    let sampler = *unsafe { samplers.index(data.sampler_id as usize) };
    let image = unsafe { textures.index(data.input_texture_id as usize) };

    // Preserve separate horizontal and vertical fields for lens mapping.
    let p = Vec2::new((uv.x - 0.5) * 2.0, (0.5 - uv.y) * 2.0);
    let src = lens_source_uv(
        p,
        data.aspect,
        data.field_scale,
        data.tan_half_fov_src,
        data.cylindrical,
        data.fisheye,
    );

    // Unrendered fisheye rays remain dark rather than clamping edge pixels.
    if src.x < 0.0 || src.x > 1.0 || src.y < 0.0 || src.y > 1.0 {
        *out_color = Vec4::new(0.0, 0.0, 0.0, 1.0);
        return;
    }

    // Apply vignette shake without moving the sampled optical axis.
    let mask_p = p - Vec2::new(data.shake[0], data.shake[1]);
    let theta = lens_ray_angle(mask_p, data.aspect, data.tan_half_fov_src, data.field_scale);
    let vignette = lens_vignette(theta, data.vignette, data.vignette_power);

    // Apply chromatic displacement radially to the remapped source UV.
    let mut color = Vec3::ZERO;
    if data.ca_taps == 0 || data.ca_strength == 0.0 {
        let s: Vec4 = image.sample_by_lod(sampler, src, 0.0);
        color = s.truncate();
    } else {
        let ca = ca_offset(src, data.ca_strength);
        let mut weight_sum = Vec3::ZERO;
        let mut i = 0u32;
        while i < data.ca_taps {
            let w = ca_spectral_weight(i, data.ca_taps);
            let tap = src + ca * ca_tap_offset(i, data.ca_taps);
            let s: Vec4 = image.sample_by_lod(sampler, tap, 0.0);
            color += s.truncate() * w;
            weight_sum += w;
            i += 1;
        }
        // Normalize partial tap sets to avoid chromatic tint.
        color /= weight_sum.max(Vec3::splat(1e-4));
    }

    *out_color = (color * vignette).extend(1.0);
}
