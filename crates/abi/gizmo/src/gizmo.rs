//! Gizmo GPU contracts and the analytic SDF math both machines share.
//!
//! One instanced pipeline draws every editor gizmo: box edges, rings, discs,
//! arrows, vertex handles, lattice cells. Each instance expands to a fixed
//! budget of [`GIZMO_VERTS_PER_INSTANCE`] vertices (a unit cube's triangle
//! list, or a 6-vertex quad with the tail degenerate), and the fragment
//! stage evaluates the shape's analytic SDF over an interpolated shape-space
//! coordinate. Stroke width is **screen-constant**: the distance field is
//! evaluated in shape space and divided by its own screen-space derivative
//! (`fwidth`), so a 2 px ring edge is 2 px at any zoom or distance.
//!
//! ## Geometry contract
//!
//! Vertex pulling, no vertex input state: draws are
//! `GIZMO_VERTS_PER_INSTANCE * instance_count` vertices of triangle list
//! reached through an identity index buffer (the `ui::UiPass` idiom — `gpu`
//! has no non-indexed draw entry point). The vertex shader derives
//! `instance = vertex_index / GIZMO_VERTS_PER_INSTANCE` and
//! `k = vertex_index % GIZMO_VERTS_PER_INSTANCE`. Culling is disabled, so
//! winding is irrelevant and a box's far faces still shade their edges.
//!
//! ## Shape space, per shape
//!
//! `GizmoInstance::transform` is a column-major local→world `Mat4`.
//!
//! - **BoxEdges / LatticeCell**: local space is the unit cube
//!   `[-0.5, 0.5]³`; the transform's scale IS the box size. The fragment
//!   receives the local position and inks the cube's twelve edges
//!   ([`gizmo_box_edge_distance`]); `LatticeCell` additionally inks a grid
//!   of spacing `params[1]` on each face ([`gizmo_grid_distance`]).
//!   `params[0]` is the stroke width in pixels.
//! - **Ring / Disc**: a quad in the local **XZ** plane (y = 0), radius
//!   `params[0]`. Ring strokes `params[1]` pixels; disc fills.
//! - **Patch**: a filled rectangle in the local **XZ** plane (y = 0), half
//!   extents `params[0]` by `params[1]`. The one shape that **tiles**:
//!   patches on a lattice meet edge to edge with no gap and no overlap,
//!   which a disc cannot do. `params[2]` rounds the corners.
//! - **Arrow**: the transform's translation is the tail and its **Y axis**
//!   is the (unnormalized) tail→tip vector — length included. The quad
//!   billboards around that axis toward the camera, and the fragment
//!   evaluates a 2D shaft-plus-head SDF in `(lateral, axial)` world units.
//! - **Handle**: the transform's translation is the point; the quad is
//!   expanded in CLIP space by `params[0]` PIXELS, so shape space is
//!   already pixels and a handle is screen-constant in size as well as in
//!   stroke.
//!
//! Colors are straight-alpha and display-referred: the pass draws AFTER
//! tonemap, because gizmos are UI-adjacent furniture and must not be eaten
//! by bloom.

use crate::{GpuPtr, gpu_data};
use glam::{Vec2, Vec3};

// Scalar `abs`/`fract`/`sqrt` are std-only; on the GPU they come from the
// num_traits::Float shim spirv-std re-exports (the `abi-ui` idiom).
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

pub const GIZMO_SHAPE_BOX_EDGES: u32 = 0;
pub const GIZMO_SHAPE_RING: u32 = 1;
pub const GIZMO_SHAPE_DISC: u32 = 2;
pub const GIZMO_SHAPE_ARROW: u32 = 3;
pub const GIZMO_SHAPE_HANDLE: u32 = 4;
pub const GIZMO_SHAPE_LATTICE_CELL: u32 = 5;
pub const GIZMO_SHAPE_PATCH: u32 = 6;

/// Ignore the scene depth buffer: the instance draws on top of everything.
/// The host records the two depth behaviors as two passes over the same
/// buffer, selected by [`GizmoDraw::xray_pass`] — no CPU-side sort.
pub const GIZMO_FLAG_XRAY: u32 = 1 << 0;

/// Vertices per instance: a unit cube's 36-vertex triangle list is the
/// widest shape; quad shapes use the first 6 and emit degenerate vertices
/// for the rest. Fixed so `vertex_index` alone locates the instance.
pub const GIZMO_VERTS_PER_INSTANCE: u32 = 36;

/// Instance ceiling for one frame. A few thousand handles is already a very
/// busy editor frame; the host asserts rather than growing mid-frame.
pub const GIZMO_CAPACITY: u32 = 4096;

/// One gizmo instance. Shape-specific meaning of `params` is documented per
/// shape in this module's header.
#[gpu_data]
pub struct GizmoInstance {
    /// Column-major local→world, `glam::Mat4::to_cols_array_2d` (the
    /// `CamMat` convention).
    pub transform: [[f32; 4]; 4],
    /// Straight-alpha, display-referred RGBA.
    pub color: [f32; 4],
    pub params: [f32; 4],
    pub shape: u32,
    pub flags: u32,
    pub _pad: [u32; 2],
}

const _: () = assert!(core::mem::size_of::<GizmoInstance>() == 112);
const _: () = assert!(core::mem::align_of::<GizmoInstance>() == 4);
const _: () = assert!(core::mem::offset_of!(GizmoInstance, transform) == 0);
const _: () = assert!(core::mem::offset_of!(GizmoInstance, color) == 64);
const _: () = assert!(core::mem::offset_of!(GizmoInstance, params) == 80);
const _: () = assert!(core::mem::offset_of!(GizmoInstance, shape) == 96);
const _: () = assert!(core::mem::offset_of!(GizmoInstance, flags) == 100);

/// Draw data for one gizmo pass. Both push-constant slots receive the same
/// pointer (the vertex stage needs instances + view, the fragment stage
/// needs nothing but rides along).
#[gpu_data]
pub struct GizmoDraw {
    pub instances: GpuPtr<GizmoInstance>,
    pub instance_count: u32,
    /// 0 draws instances WITHOUT [`GIZMO_FLAG_XRAY`], 1 draws only the
    /// x-ray ones. Instances that do not match collapse to degenerate
    /// vertices — two cheap vertex passes instead of a host-side sort.
    pub xray_pass: u32,
    pub view_proj: [[f32; 4]; 4],
    pub camera_position: [f32; 3],
    pub _pad0: u32,
    /// Physical pixels; the handle shape's clip-space expansion needs it.
    pub screen_size: [f32; 2],
    pub _pad1: [u32; 2],
}

const _: () = assert!(core::mem::size_of::<GizmoDraw>() == 112);
const _: () = assert!(core::mem::align_of::<GizmoDraw>() == 4);
const _: () = assert!(core::mem::offset_of!(GizmoDraw, instances) == 0);
const _: () = assert!(core::mem::offset_of!(GizmoDraw, instance_count) == 8);
const _: () = assert!(core::mem::offset_of!(GizmoDraw, xray_pass) == 12);
const _: () = assert!(core::mem::offset_of!(GizmoDraw, view_proj) == 16);
const _: () = assert!(core::mem::offset_of!(GizmoDraw, camera_position) == 80);
const _: () = assert!(core::mem::offset_of!(GizmoDraw, screen_size) == 96);

/// Quad corner `k` in `[-1, 1]²`, triangle-list order (tl, bl, br), (tl, br,
/// tr) — the `abi_ui` pattern. `k >= 6` is caller error.
pub fn gizmo_quad_corner(k: u32) -> Vec2 {
    let corner = match k {
        0 | 3 => 0u32, // tl
        1 => 3,        // bl
        2 | 4 => 2,    // br
        _ => 1,        // tr
    };
    match corner {
        0 => Vec2::new(-1.0, 1.0),
        1 => Vec2::new(1.0, 1.0),
        2 => Vec2::new(1.0, -1.0),
        _ => Vec2::new(-1.0, -1.0),
    }
}

/// Vertex `k` of the unit cube's 36-vertex triangle list, in `[-0.5, 0.5]³`.
/// Six faces of two triangles; culling is off, so winding is not load-
/// bearing — coverage of the cube's surface is.
pub fn gizmo_cube_vertex(k: u32) -> Vec3 {
    let face = k / 6;
    let q = gizmo_quad_corner(k % 6) * 0.5;
    match face {
        0 => Vec3::new(-0.5, q.y, q.x),
        1 => Vec3::new(0.5, q.y, q.x),
        2 => Vec3::new(q.x, -0.5, q.y),
        3 => Vec3::new(q.x, 0.5, q.y),
        4 => Vec3::new(q.x, q.y, -0.5),
        _ => Vec3::new(q.x, q.y, 0.5),
    }
}

/// The median of three — the second-smallest face distance of a point on
/// the unit cube's surface, which IS its distance to the nearest edge.
fn median3(x: f32, y: f32, z: f32) -> f32 {
    x.min(y).max(y.min(z).max(x.min(z)))
}

/// Distance from a unit-cube surface point to the nearest cube EDGE, in
/// local units. One face distance is ~0 (the face the point sits on); the
/// next-smallest is the edge distance.
pub fn gizmo_box_edge_distance(local: Vec3) -> f32 {
    let q = Vec3::splat(0.5) - local.abs();
    median3(q.x, q.y, q.z)
}

/// Distance to the nearest interior lattice plane, measured IN the face the
/// point sits on: the face-normal axis is excluded, or its own plane would
/// ink the whole face. `spacing <= 0` returns a large distance (no grid).
pub fn gizmo_grid_distance(local: Vec3, spacing: f32) -> f32 {
    if spacing <= 0.0 {
        return f32::MAX;
    }
    let q = Vec3::splat(0.5) - local.abs();
    let axis = |v: f32| {
        let t = (v + 0.5) / spacing;
        let f = t - t.floor();
        f.min(1.0 - f) * spacing
    };
    let (dx, dy, dz) = (axis(local.x), axis(local.y), axis(local.z));
    // Drop the axis whose face we are on (smallest face distance).
    if q.x <= q.y && q.x <= q.z {
        dy.min(dz)
    } else if q.y <= q.z {
        dx.min(dz)
    } else {
        dx.min(dy)
    }
}

/// Coverage of a `width_px`-wide stroke centered on the zero set of a
/// distance field, given that field's screen-space derivative. This is the
/// screen-constant law: distance and derivative are both in shape space, so
/// their ratio is pixels, whatever the transform's scale or the zoom.
pub fn gizmo_stroke_coverage(distance: f32, derivative: f32, width_px: f32) -> f32 {
    let fw = derivative.max(1.0e-8);
    let half = 0.5 * width_px.max(0.0) * fw;
    ((half - distance.abs()) / fw + 0.5).clamp(0.0, 1.0)
}

/// Coverage of the interior of a distance field, antialiased over one pixel.
pub fn gizmo_fill_coverage(distance: f32, derivative: f32) -> f32 {
    let fw = derivative.max(1.0e-8);
    (0.5 - distance / fw).clamp(0.0, 1.0)
}

/// Signed distance to a rounded rectangle of half extents `half`,
/// centered at the origin — the patch shape's field.
///
/// <https://iquilezles.org/articles/distfunctions2d/>
pub fn gizmo_patch_distance(p: Vec2, half: Vec2, radius: f32) -> f32 {
    let r = radius.clamp(0.0, half.x.min(half.y));
    sd_box2(p, (half - Vec2::splat(r)).max(Vec2::ZERO)) - r
}

/// Half extent of the patch's quad, per axis. A hair over the shape so the
/// antialias band at the edge has room to live inside the geometry.
pub fn gizmo_patch_extent(half: Vec2) -> Vec2 {
    half.max(Vec2::ZERO) + Vec2::splat(1.0e-3)
}

/// iq's 2D box SDF, centered at the origin.
fn sd_box2(p: Vec2, half: Vec2) -> f32 {
    let d = p.abs() - half;
    d.max(Vec2::ZERO).length() + d.x.max(d.y).min(0.0)
}

/// iq's 2D triangle SDF (exact, signed).
fn sd_triangle2(p: Vec2, a: Vec2, b: Vec2, c: Vec2) -> f32 {
    let e0 = b - a;
    let e1 = c - b;
    let e2 = a - c;
    let v0 = p - a;
    let v1 = p - b;
    let v2 = p - c;
    let pq0 = v0 - e0 * (v0.dot(e0) / e0.dot(e0)).clamp(0.0, 1.0);
    let pq1 = v1 - e1 * (v1.dot(e1) / e1.dot(e1)).clamp(0.0, 1.0);
    let pq2 = v2 - e2 * (v2.dot(e2) / e2.dot(e2)).clamp(0.0, 1.0);
    let s = (e0.x * e2.y - e0.y * e2.x).signum();
    let d0 = Vec2::new(pq0.dot(pq0), s * (v0.x * e0.y - v0.y * e0.x));
    let d1 = Vec2::new(pq1.dot(pq1), s * (v1.x * e1.y - v1.y * e1.x));
    let d2 = Vec2::new(pq2.dot(pq2), s * (v2.x * e2.y - v2.y * e2.x));
    let d = d0.min(d1).min(d2);
    -d.x.sqrt() * d.y.signum()
}

/// Shaft-plus-head arrow, in `(lateral, axial)` world units: the shaft runs
/// from axial 0 to `length - head_len` at half width `shaft_half`, the head
/// tapers from `head_half` to a point at axial `length`. Negative inside.
pub fn gizmo_arrow_distance(
    p: Vec2,
    length: f32,
    shaft_half: f32,
    head_len: f32,
    head_half: f32,
) -> f32 {
    let head_len = head_len.clamp(0.0, length);
    let base = length - head_len;
    let shaft = sd_box2(
        p - Vec2::new(0.0, base * 0.5),
        Vec2::new(shaft_half, base * 0.5),
    );
    if head_len <= 0.0 {
        return shaft;
    }
    let head = sd_triangle2(
        p,
        Vec2::new(-head_half, base),
        Vec2::new(head_half, base),
        Vec2::new(0.0, length),
    );
    shaft.min(head)
}

/// The lateral half-extent an arrow's billboard quad must cover, plus the
/// margin the antialias band needs. Host and vertex stage agree through it.
pub fn gizmo_arrow_half_width(shaft_half: f32, head_half: f32) -> f32 {
    shaft_half.max(head_half) * 1.25 + 1.0e-4
}

/// The half-extent a ring/disc quad must cover for radius `radius`: the
/// stroke and its antialias band live outside the radius, and 25% is a
/// generous, scale-free margin (`extra` adds world units on top).
pub fn gizmo_disc_extent(radius: f32, extra: f32) -> f32 {
    radius * 1.25 + extra.max(0.0) + 1.0e-4
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1.0e-5;

    #[test]
    fn cube_vertices_cover_every_face_and_stay_on_the_unit_cube() {
        let mut faces = [0u32; 6];
        for k in 0..GIZMO_VERTS_PER_INSTANCE {
            let v = gizmo_cube_vertex(k);
            assert!(v.abs().max_element() - 0.5 < EPS);
            // Exactly one coordinate is pinned to a face plane.
            let pinned = [v.x, v.y, v.z]
                .iter()
                .filter(|c| (c.abs() - 0.5).abs() < EPS)
                .count();
            assert!(pinned >= 1);
            faces[(k / 6) as usize] += 1;
        }
        assert_eq!(faces, [6; 6]);
    }

    #[test]
    fn quad_corners_are_two_triangles_of_one_square() {
        let corners: Vec<Vec2> = (0..6).map(gizmo_quad_corner).collect();
        for c in &corners {
            assert!((c.x.abs() - 1.0).abs() < EPS && (c.y.abs() - 1.0).abs() < EPS);
        }
        // Both triangles have nonzero, opposite-consistent area.
        let area = |a: Vec2, b: Vec2, c: Vec2| (b - a).perp_dot(c - a);
        let t0 = area(corners[0], corners[1], corners[2]);
        let t1 = area(corners[3], corners[4], corners[5]);
        assert!(t0.abs() > EPS && t1.abs() > EPS);
        assert!(t0.signum() == t1.signum());
    }

    /// **The patch tiles.** This is the whole reason the shape exists:
    /// patches of one unit, one unit apart, meet exactly — no gap where
    /// the ground shows through and no overlap where a wash doubles.
    /// A disc can do neither, whatever its radius.
    #[test]
    fn patches_on_a_lattice_meet_exactly() {
        let half = Vec2::splat(0.5);
        // The shared edge of two neighbours, from both sides: on the
        // boundary of each, so coverage hands over with nothing between.
        assert!(gizmo_patch_distance(Vec2::new(0.5, 0.0), half, 0.0).abs() < EPS);
        assert!(gizmo_patch_distance(Vec2::new(-0.5, 0.0), half, 0.0).abs() < EPS);
        // A corner belongs to four patches, and to each of them equally.
        assert!(gizmo_patch_distance(Vec2::new(0.5, 0.5), half, 0.0).abs() < EPS);
        // Inside is negative (filled), outside positive (clipped).
        assert!(gizmo_patch_distance(Vec2::ZERO, half, 0.0) < 0.0);
        assert!(gizmo_patch_distance(Vec2::new(0.6, 0.0), half, 0.0) > 0.0);
        // The quad contains the shape, or the edge would be cut off
        // before the antialias band could draw it.
        let extent = gizmo_patch_extent(half);
        assert!(extent.x >= half.x && extent.y >= half.y);
    }

    /// Rounding pulls the corners in and leaves the edge midpoints where
    /// they were, so a rounded patch still tiles along its flats.
    #[test]
    fn patch_rounding_only_moves_the_corners() {
        let half = Vec2::splat(0.5);
        assert!(gizmo_patch_distance(Vec2::new(0.5, 0.0), half, 0.2).abs() < EPS);
        assert!(gizmo_patch_distance(Vec2::new(0.5, 0.5), half, 0.2) > 0.0);
        // Rounding beyond the half extent clamps rather than inverting.
        assert!(gizmo_patch_distance(Vec2::ZERO, half, 10.0) < 0.0);
    }

    #[test]
    fn box_edge_distance_is_zero_on_edges_and_grows_inward() {
        // Middle of the +x face: farthest from every edge (0.5 away).
        assert!((gizmo_box_edge_distance(Vec3::new(0.5, 0.0, 0.0)) - 0.5).abs() < EPS);
        // On an edge of the cube.
        assert!(gizmo_box_edge_distance(Vec3::new(0.5, 0.5, 0.0)).abs() < EPS);
        // A corner is on three edges at once.
        assert!(gizmo_box_edge_distance(Vec3::new(0.5, 0.5, 0.5)).abs() < EPS);
        // A quarter of the way across the face.
        assert!((gizmo_box_edge_distance(Vec3::new(0.5, 0.25, 0.0)) - 0.25).abs() < EPS);
    }

    #[test]
    fn grid_distance_ignores_the_face_it_lies_on() {
        // Spacing 0.25 on the +x face: the face's own plane (x = 0.5) would
        // read as a grid line everywhere if the normal axis were counted.
        let d = gizmo_grid_distance(Vec3::new(0.5, 0.125, 0.125), 0.25);
        assert!((d - 0.125).abs() < EPS, "{d}");
        // Directly on an interior lattice plane (y = 0.0 is a multiple).
        assert!(gizmo_grid_distance(Vec3::new(0.5, 0.0, 0.125), 0.25).abs() < EPS);
        // No spacing, no grid.
        assert_eq!(gizmo_grid_distance(Vec3::new(0.5, 0.1, 0.1), 0.0), f32::MAX);
    }

    #[test]
    fn stroke_coverage_is_screen_constant() {
        // Same pixel geometry, two wildly different world scales: coverage
        // Scale invariance is the purpose of dividing by fwidth.
        for scale in [1.0f32, 1000.0] {
            let fw = 0.01 * scale;
            assert!((gizmo_stroke_coverage(0.0, fw, 2.0) - 1.0).abs() < EPS);
            // On the stroke's own edge (1 px from center): the antialias
            // band's midpoint, half covered.
            let c = gizmo_stroke_coverage(fw, fw, 2.0);
            assert!((c - 0.5).abs() < EPS, "{c}");
            // Half a pixel further out: the band has ended.
            let c = gizmo_stroke_coverage(1.5 * fw, fw, 2.0);
            assert!(c.abs() < EPS, "{c}");
        }
    }

    #[test]
    fn fill_coverage_saturates_inside_and_out() {
        assert!((gizmo_fill_coverage(-1.0, 0.01) - 1.0).abs() < EPS);
        assert!(gizmo_fill_coverage(1.0, 0.01).abs() < EPS);
        assert!((gizmo_fill_coverage(0.0, 0.01) - 0.5).abs() < EPS);
    }

    #[test]
    fn arrow_distance_is_inside_the_shaft_and_the_head() {
        let (length, shaft, head_len, head) = (1.0, 0.05, 0.3, 0.15);
        let d = |x, y| gizmo_arrow_distance(Vec2::new(x, y), length, shaft, head_len, head);
        assert!(d(0.0, 0.35) < 0.0, "shaft interior");
        assert!(d(0.0, 0.8) < 0.0, "head interior");
        assert!(d(0.0, length + 0.05) > 0.0, "past the tip");
        assert!(d(0.0, -0.05) > 0.0, "behind the tail");
        assert!(d(0.12, 0.35) > 0.0, "beside the shaft");
        assert!(d(0.12, 0.72) < 0.0, "inside the head's flare");
        // The tip is on the boundary.
        assert!(d(0.0, length).abs() < 1.0e-3);
    }

    #[test]
    fn arrow_with_no_head_is_a_bare_shaft() {
        let d = gizmo_arrow_distance(Vec2::new(0.0, 0.5), 1.0, 0.05, 0.0, 0.2);
        assert!(d < 0.0);
    }

    #[test]
    fn quad_extents_cover_their_shapes() {
        assert!(gizmo_disc_extent(2.0, 0.0) > 2.0);
        assert!(gizmo_arrow_half_width(0.05, 0.15) > 0.15);
    }
}
