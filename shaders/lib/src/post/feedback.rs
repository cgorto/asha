use abi_core::GraphicsPush;
use abi_post::{FeedbackData, feedback_combine, feedback_flow_uv};
use glam::{Vec2, Vec3, Vec4};
use spirv_std::image::Image2d;
use spirv_std::spirv;
use spirv_std::{RuntimeArray, Sampler};

/// Max-combines fresh input with decayed, camera-flowed feedback history.
/// Ping-pong textures alternate between sampling and writing.
#[spirv(fragment)]
pub fn feedback_frag(
    #[spirv(push_constant)] push: &GraphicsPush,
    #[spirv(descriptor_set = 0, binding = 0)] textures: &RuntimeArray<Image2d>,
    #[spirv(descriptor_set = 2, binding = 0)] samplers: &RuntimeArray<Sampler>,
    uv: Vec2,
    out_color: &mut Vec4,
) {
    let data = push.frag::<FeedbackData>();
    let sampler = *unsafe { samplers.index(data.sampler_id as usize) };
    let input_image = unsafe { textures.index(data.input_texture_id as usize) };
    let input: Vec4 = input_image.sample_by_lod(sampler, uv, 0.0);

    // Skip undefined history after rebuild; NaNs could propagate.
    let mut history = Vec3::ZERO;
    if data.sample_history != 0 {
        let history_uv = feedback_flow_uv(&data.curr, &data.prev, uv, data.flow);
        let history_image = unsafe { textures.index(data.history_texture_id as usize) };
        history = history_image
            .sample_by_lod(sampler, history_uv, 0.0)
            .truncate();
    }
    *out_color = feedback_combine(input.truncate(), history, data.decay, data.floor).extend(1.0);
}
