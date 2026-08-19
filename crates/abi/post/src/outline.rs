//! GPU ABI for the display-space jump-flood outline pass.
//!
//! JFA texels store `[seed_x, seed_y, normalized_group, unused]`. A seed
//! position of `[-1, -1]` is the empty sentinel; group zero is likewise
//! invalid. The R8 silhouette mask preserves group identity as `group / 255`.

use crate::gpu_data;

pub const OUTLINE_GROUP_CAPACITY: u32 = 8;

/// One authored outline style. Width is in display pixels.
#[gpu_data]
pub struct OutlineGroup {
    pub color: [f32; 4],
    pub width: f32,
    pub _pad0: [f32; 3],
}

const _: () = assert!(core::mem::size_of::<OutlineGroup>() == 32);
const _: () = assert!(core::mem::align_of::<OutlineGroup>() == 4);
const _: () = assert!(core::mem::offset_of!(OutlineGroup, color) == 0);
const _: () = assert!(core::mem::offset_of!(OutlineGroup, width) == 16);
const _: () = assert!(core::mem::offset_of!(OutlineGroup, _pad0) == 20);

/// Dispatch data for `outline_jfa_init`: convert the silhouette mask to
/// seeds and initialize both ping-pong textures.
#[gpu_data]
pub struct OutlineJfaInitData {
    pub mask_texture_id: u32,
    pub output_a_id: u32,
    pub output_b_id: u32,
    pub _pad0: u32,
    pub size: [u32; 2],
    pub _pad1: [u32; 2],
}

const _: () = assert!(core::mem::size_of::<OutlineJfaInitData>() == 32);
const _: () = assert!(core::mem::align_of::<OutlineJfaInitData>() == 4);
const _: () = assert!(core::mem::offset_of!(OutlineJfaInitData, mask_texture_id) == 0);
const _: () = assert!(core::mem::offset_of!(OutlineJfaInitData, output_a_id) == 4);
const _: () = assert!(core::mem::offset_of!(OutlineJfaInitData, output_b_id) == 8);
const _: () = assert!(core::mem::offset_of!(OutlineJfaInitData, _pad0) == 12);
const _: () = assert!(core::mem::offset_of!(OutlineJfaInitData, size) == 16);
const _: () = assert!(core::mem::offset_of!(OutlineJfaInitData, _pad1) == 24);

/// Dispatch data for one `outline_jfa_flood` ping-pong step.
#[gpu_data]
pub struct OutlineJfaFloodData {
    pub input_texture_id: u32,
    pub output_texture_id: u32,
    pub step_size: i32,
    pub _pad0: u32,
    pub size: [u32; 2],
    pub region_offset: [u32; 2],
    pub region_size: [u32; 2],
    pub _pad1: [u32; 2],
}

const _: () = assert!(core::mem::size_of::<OutlineJfaFloodData>() == 48);
const _: () = assert!(core::mem::align_of::<OutlineJfaFloodData>() == 4);
const _: () = assert!(core::mem::offset_of!(OutlineJfaFloodData, input_texture_id) == 0);
const _: () = assert!(core::mem::offset_of!(OutlineJfaFloodData, output_texture_id) == 4);
const _: () = assert!(core::mem::offset_of!(OutlineJfaFloodData, step_size) == 8);
const _: () = assert!(core::mem::offset_of!(OutlineJfaFloodData, _pad0) == 12);
const _: () = assert!(core::mem::offset_of!(OutlineJfaFloodData, size) == 16);
const _: () = assert!(core::mem::offset_of!(OutlineJfaFloodData, region_offset) == 24);
const _: () = assert!(core::mem::offset_of!(OutlineJfaFloodData, region_size) == 32);
const _: () = assert!(core::mem::offset_of!(OutlineJfaFloodData, _pad1) == 40);

/// Fragment data for `outline_composite`. `groups` is a fixed eight-entry
/// ABI table; `group_count` authorizes only its leading entries.
#[gpu_data]
pub struct OutlineCompositeData {
    pub jfa_texture_id: u32,
    pub mask_texture_id: u32,
    pub sampler_id: u32,
    pub group_count: u32,
    pub screen_size: [u32; 2],
    pub region_min: [u32; 2],
    pub region_max: [u32; 2],
    pub groups: [OutlineGroup; 8],
}

const _: () = assert!(core::mem::size_of::<OutlineCompositeData>() == 296);
const _: () = assert!(core::mem::align_of::<OutlineCompositeData>() == 4);
const _: () = assert!(core::mem::offset_of!(OutlineCompositeData, jfa_texture_id) == 0);
const _: () = assert!(core::mem::offset_of!(OutlineCompositeData, mask_texture_id) == 4);
const _: () = assert!(core::mem::offset_of!(OutlineCompositeData, sampler_id) == 8);
const _: () = assert!(core::mem::offset_of!(OutlineCompositeData, group_count) == 12);
const _: () = assert!(core::mem::offset_of!(OutlineCompositeData, screen_size) == 16);
const _: () = assert!(core::mem::offset_of!(OutlineCompositeData, region_min) == 24);
const _: () = assert!(core::mem::offset_of!(OutlineCompositeData, region_max) == 32);
const _: () = assert!(core::mem::offset_of!(OutlineCompositeData, groups) == 40);
