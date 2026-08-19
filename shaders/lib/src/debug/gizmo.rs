//! Instanced gizmo entry points using `abi_gizmo` shape math.
//! Vertex pulling derives instance and vertex from `vertex_index`.
//! X-ray mismatches collapse to degenerate vertices.

use abi_core::GraphicsPush;
use abi_gizmo::{
    GIZMO_FLAG_XRAY, GIZMO_SHAPE_ARROW, GIZMO_SHAPE_BOX_EDGES, GIZMO_SHAPE_DISC,
    GIZMO_SHAPE_HANDLE, GIZMO_SHAPE_LATTICE_CELL, GIZMO_SHAPE_PATCH, GIZMO_SHAPE_RING,
    GIZMO_VERTS_PER_INSTANCE, GizmoDraw, gizmo_arrow_distance, gizmo_arrow_half_width,
    gizmo_box_edge_distance, gizmo_cube_vertex, gizmo_disc_extent, gizmo_fill_coverage,
    gizmo_grid_distance, gizmo_patch_distance, gizmo_patch_extent, gizmo_quad_corner,
    gizmo_stroke_coverage,
};
use glam::{Mat4, Vec2, Vec3, Vec4, Vec4Swizzles};
use spirv_std::arch::Derivative;
use spirv_std::num_traits::Float;
use spirv_std::spirv;

fn safe_normalize(v: Vec3, fallback: Vec3) -> Vec3 {
    let len2 = v.length_squared();
    if len2 > 1.0e-12 {
        v * (1.0 / len2.sqrt())
    } else {
        fallback
    }
}

/// Computes an eye-facing lateral basis around an axis.
fn billboard_lateral(axis: Vec3, to_eye: Vec3) -> Vec3 {
    let lateral = to_eye.cross(axis);
    let fallback = if axis.y.abs() < 0.9 { Vec3::Y } else { Vec3::X };
    safe_normalize(lateral, safe_normalize(axis.cross(fallback), Vec3::X))
}

#[spirv(vertex)]
pub fn gizmo_vert(
    #[spirv(push_constant)] push: &GraphicsPush,
    #[spirv(vertex_index)] vert_id: i32,
    #[spirv(position)] out_pos: &mut Vec4,
    out_shape_coord: &mut Vec3,
    #[spirv(flat)] out_color: &mut Vec4,
    #[spirv(flat)] out_params: &mut Vec4,
    #[spirv(flat)] out_shape: &mut u32,
) {
    let data = push.vert::<GizmoDraw>();
    let vid = vert_id as u32;
    let index = vid / GIZMO_VERTS_PER_INSTANCE;
    let k = vid % GIZMO_VERTS_PER_INSTANCE;

    *out_pos = Vec4::ZERO;
    *out_shape_coord = Vec3::ZERO;
    *out_color = Vec4::ZERO;
    *out_params = Vec4::ZERO;
    *out_shape = 0;
    if index >= data.instance_count {
        return;
    }
    let instance = data.instances[index];
    let xray = if instance.flags & GIZMO_FLAG_XRAY != 0 {
        1u32
    } else {
        0u32
    };
    if xray != data.xray_pass {
        return;
    }

    let model = Mat4::from_cols_array_2d(&instance.transform);
    let view_proj = Mat4::from_cols_array_2d(&data.view_proj);
    let origin = model.w_axis.xyz();
    let eye = Vec3::from_array(data.camera_position);
    let params = Vec4::from_array(instance.params);
    let shape = instance.shape;

    let (clip, coord) = if shape == GIZMO_SHAPE_BOX_EDGES || shape == GIZMO_SHAPE_LATTICE_CELL {
        let local = gizmo_cube_vertex(k);
        (view_proj * (model * local.extend(1.0)), local)
    } else if shape == GIZMO_SHAPE_HANDLE {
        // Expand handles in clip space for constant pixel size.
        let q = if k < 6 {
            gizmo_quad_corner(k)
        } else {
            Vec2::ZERO
        };
        let half = params.x.max(1.0);
        let mut clip = view_proj * origin.extend(1.0);
        let screen = Vec2::from_array(data.screen_size).max(Vec2::ONE);
        clip.x += q.x * half * 2.0 * clip.w / screen.x;
        clip.y += q.y * half * 2.0 * clip.w / screen.y;
        (clip, (q * half).extend(0.0))
    } else if shape == GIZMO_SHAPE_ARROW {
        let q = if k < 6 {
            gizmo_quad_corner(k)
        } else {
            Vec2::ZERO
        };
        let axis_vec = model.y_axis.xyz();
        let length = axis_vec.length();
        let axis = safe_normalize(axis_vec, Vec3::Y);
        let head_half = if params.y > 0.0 {
            params.y
        } else {
            length * 0.09
        };
        let shaft_half = if params.x > 0.0 {
            params.x
        } else {
            head_half * 0.34
        };
        let half_w = gizmo_arrow_half_width(shaft_half, head_half);
        let lateral = billboard_lateral(axis, safe_normalize(eye - origin, Vec3::Z));
        // Extend the axial span for antialiasing and arrow tips.
        let margin = half_w;
        let axial = (q.y * 0.5 + 0.5) * (length + 2.0 * margin) - margin;
        let world = origin + axis * axial + lateral * (q.x * half_w);
        (
            view_proj * world.extend(1.0),
            Vec3::new(q.x * half_w, axial, length),
        )
    } else if shape == GIZMO_SHAPE_PATCH {
        // Use patch extents so lattice cells meet edge-to-edge.
        let q = if k < 6 {
            gizmo_quad_corner(k)
        } else {
            Vec2::ZERO
        };
        let extent = gizmo_patch_extent(Vec2::new(params.x, params.y));
        let local = Vec3::new(q.x * extent.x, 0.0, q.y * extent.y);
        (
            view_proj * (model * local.extend(1.0)),
            Vec3::new(local.x, 0.0, local.z),
        )
    } else {
        // Rings and discs use local XZ quads.
        let q = if k < 6 {
            gizmo_quad_corner(k)
        } else {
            Vec2::ZERO
        };
        let extent = gizmo_disc_extent(params.x, 0.0);
        let local = Vec3::new(q.x * extent, 0.0, q.y * extent);
        (
            view_proj * (model * local.extend(1.0)),
            Vec3::new(local.x, 0.0, local.z),
        )
    };

    *out_pos = clip;
    *out_shape_coord = coord;
    *out_color = Vec4::from_array(instance.color);
    *out_params = params;
    *out_shape = shape;
}

#[spirv(fragment)]
pub fn gizmo_frag(
    shape_coord: Vec3,
    #[spirv(flat)] color: Vec4,
    #[spirv(flat)] params: Vec4,
    #[spirv(flat)] shape: u32,
    out_color: &mut Vec4,
) {
    // Differentiate the SDF distance for scale-independent strokes.
    let coverage = if shape == GIZMO_SHAPE_BOX_EDGES || shape == GIZMO_SHAPE_LATTICE_CELL {
        let mut distance = gizmo_box_edge_distance(shape_coord);
        if shape == GIZMO_SHAPE_LATTICE_CELL {
            distance = distance.min(gizmo_grid_distance(shape_coord, params.y));
        }
        let width = if params.x > 0.0 { params.x } else { 1.5 };
        gizmo_stroke_coverage(distance, distance.fwidth().abs(), width)
    } else if shape == GIZMO_SHAPE_RING || shape == GIZMO_SHAPE_DISC {
        let radius = Vec2::new(shape_coord.x, shape_coord.z).length();
        let distance = radius - params.x;
        let fw = distance.fwidth().abs();
        if shape == GIZMO_SHAPE_DISC {
            gizmo_fill_coverage(distance, fw)
        } else {
            let width = if params.y > 0.0 { params.y } else { 1.5 };
            gizmo_stroke_coverage(distance, fw, width)
        }
    } else if shape == GIZMO_SHAPE_PATCH {
        let distance = gizmo_patch_distance(
            Vec2::new(shape_coord.x, shape_coord.z),
            Vec2::new(params.x, params.y),
            params.z,
        );
        gizmo_fill_coverage(distance, distance.fwidth().abs())
    } else if shape == GIZMO_SHAPE_ARROW {
        let length = shape_coord.z;
        let head_len = if params.z > 0.0 {
            params.z
        } else {
            length * 0.3
        };
        let head_half = if params.y > 0.0 {
            params.y
        } else {
            length * 0.09
        };
        let shaft_half = if params.x > 0.0 {
            params.x
        } else {
            head_half * 0.34
        };
        let distance = gizmo_arrow_distance(
            Vec2::new(shape_coord.x, shape_coord.y),
            length,
            shaft_half,
            head_len,
            head_half,
        );
        gizmo_fill_coverage(distance, distance.fwidth().abs())
    } else {
        // Handle: a filled square dot in pixel space, with `params.y`
        // pixels of corner rounding when asked for.
        let p = Vec2::new(shape_coord.x, shape_coord.y);
        let half = params.x.max(1.0);
        let radius = params.y.clamp(0.0, half);
        let d = p.abs() - Vec2::splat(half - radius);
        let distance = d.max(Vec2::ZERO).length() + d.x.max(d.y).min(0.0) - radius;
        gizmo_fill_coverage(distance, distance.fwidth().abs())
    };

    if coverage <= 0.0 {
        spirv_std::arch::kill();
    }
    *out_color = Vec4::new(color.x, color.y, color.z, color.w * coverage);
}
