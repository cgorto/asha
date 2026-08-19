//! Jump-flood outline stages after mesh silhouette rendering.

use crate::core::util::smoothstep;
use abi_core::{GpuPtr, GraphicsPush};
use abi_post::{
    OUTLINE_GROUP_CAPACITY, OutlineCompositeData, OutlineJfaFloodData, OutlineJfaInitData,
};
use glam::{IVec2, UVec2, UVec3, Vec2, Vec4};
use spirv_std::image::Image2d;
use spirv_std::num_traits::Float;
use spirv_std::spirv;
use spirv_std::{Image, RuntimeArray};

/// Converts the silhouette mask to JFA seeds and clears both ping-pong textures.
#[spirv(compute(threads(8, 8)))]
pub fn outline_jfa_init(
    #[spirv(push_constant)] data_ptr: &GpuPtr<OutlineJfaInitData>,
    #[spirv(descriptor_set = 0, binding = 0)] textures: &RuntimeArray<Image2d>,
    #[spirv(descriptor_set = 1, binding = 0)] textures_rw: &RuntimeArray<
        Image!(2D, format = rgba16f, sampled = false),
    >,
    #[spirv(global_invocation_id)] gid: UVec3,
) {
    let data = &**data_ptr;
    if gid.x >= data.size[0] || gid.y >= data.size[1] {
        return;
    }

    let coord = gid.truncate();
    let mask = unsafe { textures.index(data.mask_texture_id as usize) }
        .fetch_with_lod(coord, 0)
        .x;
    let seed = if mask > 0.0 {
        Vec4::new(coord.x as f32 + 0.5, coord.y as f32 + 0.5, mask, 0.0)
    } else {
        Vec4::new(-1.0, -1.0, 0.0, 0.0)
    };

    unsafe {
        textures_rw
            .index(data.output_a_id as usize)
            .write(coord, seed);
        textures_rw
            .index(data.output_b_id as usize)
            .write(coord, seed);
    }
}

/// Performs one 3×3 jump-flood step while preserving seed groups.
#[spirv(compute(threads(8, 8)))]
pub fn outline_jfa_flood(
    #[spirv(push_constant)] data_ptr: &GpuPtr<OutlineJfaFloodData>,
    #[spirv(descriptor_set = 0, binding = 0)] textures: &RuntimeArray<Image2d>,
    #[spirv(descriptor_set = 1, binding = 0)] textures_rw: &RuntimeArray<
        Image!(2D, format = rgba16f, sampled = false),
    >,
    #[spirv(global_invocation_id)] gid: UVec3,
) {
    let data = &**data_ptr;
    if gid.x >= data.region_size[0] || gid.y >= data.region_size[1] {
        return;
    }

    let coord = gid.truncate().as_ivec2() + UVec2::from_array(data.region_offset).as_ivec2();
    let pixel_center = coord.as_vec2() + Vec2::splat(0.5);
    let mut best_dist_sq = 1.0e20;
    let mut best_seed = Vec4::new(-1.0, -1.0, 0.0, 0.0);

    let mut dy = -1i32;
    while dy <= 1 {
        let mut dx = -1i32;
        while dx <= 1 {
            let sample_coord = coord + IVec2::new(dx, dy) * data.step_size;
            if sample_coord.x >= 0
                && sample_coord.y >= 0
                && (sample_coord.x as u32) < data.size[0]
                && (sample_coord.y as u32) < data.size[1]
            {
                let seed = unsafe { textures.index(data.input_texture_id as usize) }
                    .fetch_with_lod(sample_coord.as_uvec2(), 0);
                if seed.x >= 0.0 {
                    let delta = pixel_center - Vec2::new(seed.x, seed.y);
                    let dist_sq = delta.dot(delta);
                    if dist_sq < best_dist_sq {
                        best_dist_sq = dist_sq;
                        best_seed = seed;
                    }
                }
            }
            dx += 1;
        }
        dy += 1;
    }

    unsafe {
        textures_rw
            .index(data.output_texture_id as usize)
            .write(coord.as_uvec2(), best_seed);
    }
}

/// Composites the final JFA field in display space.
#[spirv(fragment)]
pub fn outline_composite(
    #[spirv(push_constant)] push: &GraphicsPush,
    #[spirv(descriptor_set = 0, binding = 0)] textures: &RuntimeArray<Image2d>,
    #[spirv(frag_coord)] frag_coord: Vec4,
    out_color: &mut Vec4,
) {
    let data = push.frag::<OutlineCompositeData>();
    let coord = frag_coord.truncate().truncate().as_uvec2();

    if coord.x < data.region_min[0]
        || coord.y < data.region_min[1]
        || coord.x >= data.region_max[0]
        || coord.y >= data.region_max[1]
    {
        spirv_std::arch::kill();
    }

    let mask = unsafe { textures.index(data.mask_texture_id as usize) }
        .fetch_with_lod(coord, 0)
        .x;
    if mask > 0.0 {
        spirv_std::arch::kill();
    }

    let seed = unsafe { textures.index(data.jfa_texture_id as usize) }.fetch_with_lod(coord, 0);
    if seed.x < 0.0 {
        spirv_std::arch::kill();
    }

    let group_id = (seed.z * 255.0 + 0.5) as u32;
    if group_id == 0 || group_id > data.group_count || group_id > OUTLINE_GROUP_CAPACITY {
        spirv_std::arch::kill();
    }

    let group = data.groups[(group_id - 1) as usize];
    let delta = coord.as_vec2() + Vec2::splat(0.5) - Vec2::new(seed.x, seed.y);
    let distance = delta.dot(delta).sqrt();
    if distance > group.width {
        spirv_std::arch::kill();
    }

    let alpha = group.color[3] * smoothstep(group.width, group.width - 1.0, distance);
    *out_color = Vec4::new(group.color[0], group.color[1], group.color[2], alpha);
}
