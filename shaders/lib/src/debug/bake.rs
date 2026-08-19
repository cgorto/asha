use abi_bake::bake_eval;
use abi_core::GraphicsPush;
use glam::{Vec2, Vec4};
use spirv_std::spirv;

/// Bakes a debug texture from push-constant parameters.
/// Generators in `abi_bake::bake_eval` remain host-evaluable.
#[spirv(fragment)]
pub fn bake_frag(#[spirv(push_constant)] push: &GraphicsPush, uv: Vec2, out_color: &mut Vec4) {
    let data = push.frag::<abi_bake::BakeData>();
    *out_color = bake_eval(
        data.kind,
        Vec4::from_array(data.color_a),
        Vec4::from_array(data.color_b),
        Vec4::from_array(data.params),
        uv,
    );
}
