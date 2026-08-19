//! The camera basis for ray generation, and its derived quantities.

use crate::gpu_data;
use glam::{UVec2, Vec3};

/// Hit-distance miss sentinel. In an f32 hit-t buffer it simply stays
/// enormous — consumers compare with `>=`, never equality.
pub const HIT_T_MISS: f32 = 1.0e9;

/// The camera basis + output geometry for one ray-generating pass:
/// volumetrics, the sky, and any shader that turns a pixel into a world ray.
#[gpu_data]
pub struct View {
    pub camera_position: [f32; 3],
    pub tan_half_fov: f32,
    pub camera_forward: [f32; 3],
    pub aspect: f32,
    pub camera_right: [f32; 3],
    /// Near plane for reverse-Z hardware depth (`view_z = near / depth`).
    pub depth_near_plane: f32,
    pub camera_up: [f32; 3],
    pub _pad: u32,
    pub output_size: [u32; 2],
    pub _pad2: [u32; 2],
}

const _: () = assert!(core::mem::size_of::<View>() == 80);

/// Pinhole ray through the pixel center. Pixel rows grow downward while
/// `camera_up` points up, hence the ndc.y flip.
pub fn ray_direction(view: &View, pixel: UVec2) -> Vec3 {
    let size = UVec2::from_array(view.output_size).as_vec2();
    let uv = (pixel.as_vec2() + 0.5) / size;
    let ndc_x = uv.x * 2.0 - 1.0;
    let ndc_y = -(uv.y * 2.0 - 1.0);
    let dir = Vec3::from_array(view.camera_forward)
        + Vec3::from_array(view.camera_right) * (ndc_x * view.tan_half_fov * view.aspect)
        + Vec3::from_array(view.camera_up) * (ndc_y * view.tan_half_fov);
    dir.normalize()
}

/// Ray-t → reverse-Z hardware depth (right-handed, infinite reverse-Z:
/// consumers linearize with `view_z = near / depth`).
pub fn hardware_depth(view: &View, ray_dir: Vec3, hit_t: f32) -> f32 {
    if hit_t >= HIT_T_MISS {
        return 0.0;
    }
    let view_z = hit_t
        * Vec3::from_array(view.camera_forward)
            .dot(ray_dir)
            .max(1.0e-6);
    (view.depth_near_plane / view_z).clamp(0.0, 1.0)
}
