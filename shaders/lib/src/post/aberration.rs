use abi_core::GraphicsPush;
use abi_post::{AberrationData, ca_offset};
use glam::{Vec2, Vec4};
use spirv_std::image::Image2d;
use spirv_std::spirv;
use spirv_std::{RuntimeArray, Sampler};

/// Samples radial chromatic aberration with green centered.
/// Offsets are defined by `abi_post::ca_offset`.
#[spirv(fragment)]
pub fn aberration_frag(
    #[spirv(push_constant)] push: &GraphicsPush,
    #[spirv(descriptor_set = 0, binding = 0)] textures: &RuntimeArray<Image2d>,
    #[spirv(descriptor_set = 2, binding = 0)] samplers: &RuntimeArray<Sampler>,
    uv: Vec2,
    out_color: &mut Vec4,
) {
    let data = push.frag::<AberrationData>();
    let sampler = *unsafe { samplers.index(data.sampler_id as usize) };
    let image = unsafe { textures.index(data.input_texture_id as usize) };
    let ca = ca_offset(uv, data.strength);
    let r: Vec4 = image.sample_by_lod(sampler, uv + ca, 0.0);
    let g: Vec4 = image.sample_by_lod(sampler, uv, 0.0);
    let b: Vec4 = image.sample_by_lod(sampler, uv - ca, 0.0);
    *out_color = Vec4::new(r.x, g.y, b.z, 1.0);
}
