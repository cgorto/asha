//! Exact mesh-shadow segment math and GPU ABI.
//!
//! Receiver-to-light queries use `P(t) = origin + direction * t` with an
//! explicit open interval. Affine instance transforms preserve this parameter.

use crate::{GpuPtr, gpu_data};
use glam::{Mat4, Vec3};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

/// obvhs 0.3.2's CWBVH8 node encoded as twenty little-endian words.
///
/// The host packs fields explicitly rather than relying on the dependency's
/// Rust layout. The shader extracts the original byte fields from these words
/// without requiring 8-bit storage-buffer support.
#[gpu_data]
pub struct CwbvhNode {
    pub words: [u32; 20],
}

const _: () = assert!(core::mem::size_of::<CwbvhNode>() == 80);
const _: () = assert!(core::mem::align_of::<CwbvhNode>() == 4);

/// One immutable mesh BLAS. Nodes and primitive IDs use obvhs-local indices;
/// positions and the original triangle-list indices reuse MeshScene storage.
#[gpu_data]
pub struct ShadowBlas {
    pub nodes: GpuPtr<CwbvhNode>,
    pub primitive_ids: GpuPtr<u32>,
    pub positions: GpuPtr<[f32; 4]>,
    pub indices: GpuPtr<u32>,
    pub node_count: u32,
    pub primitive_count: u32,
    pub _pad0: [u32; 2],
}

const _: () = assert!(core::mem::size_of::<ShadowBlas>() == 48);
const _: () = assert!(core::mem::align_of::<ShadowBlas>() == 4);
const _: () = assert!(core::mem::offset_of!(ShadowBlas, primitive_ids) == 8);
const _: () = assert!(core::mem::offset_of!(ShadowBlas, positions) == 16);
const _: () = assert!(core::mem::offset_of!(ShadowBlas, indices) == 24);
const _: () = assert!(core::mem::offset_of!(ShadowBlas, node_count) == 32);

/// A finite ray segment with an open accepted interval `(t_min, t_max)`.
/// All-zero is the inactive ZII value.
#[gpu_data]
pub struct ShadowSegment {
    pub origin: [f32; 3],
    pub t_min: f32,
    pub direction: [f32; 3],
    pub t_max: f32,
}

const _: () = assert!(core::mem::size_of::<ShadowSegment>() == 32);
const _: () = assert!(core::mem::align_of::<ShadowSegment>() == 4);
const _: () = assert!(core::mem::offset_of!(ShadowSegment, t_min) == 12);
const _: () = assert!(core::mem::offset_of!(ShadowSegment, direction) == 16);
const _: () = assert!(core::mem::offset_of!(ShadowSegment, t_max) == 28);

impl ShadowSegment {
    /// Construct a receiver-to-light segment. Biases are world distances,
    /// converted to the segment's dimensionless parameter. Invalid or empty
    /// inputs return the inactive ZII value; recording hosts validate their
    /// authored bias values separately and fail loudly there.
    pub fn between(receiver: Vec3, light: Vec3, origin_bias: f32, destination_bias: f32) -> Self {
        let direction = light - receiver;
        let distance2 = direction.length_squared();
        if !receiver.is_finite()
            || !light.is_finite()
            || !origin_bias.is_finite()
            || !destination_bias.is_finite()
            || origin_bias < 0.0
            || destination_bias < 0.0
            || !distance2.is_finite()
            || distance2 <= 0.0
        {
            return Self::default();
        }

        let distance = distance2.sqrt();
        if distance <= origin_bias + destination_bias {
            return Self::default();
        }

        Self {
            origin: receiver.to_array(),
            t_min: origin_bias / distance,
            direction: direction.to_array(),
            t_max: 1.0 - destination_bias / distance,
        }
    }

    pub fn is_active(&self) -> bool {
        let origin = Vec3::from_array(self.origin);
        let direction = Vec3::from_array(self.direction);
        origin.is_finite()
            && direction.is_finite()
            && direction.length_squared() > 0.0
            && self.t_min.is_finite()
            && self.t_max.is_finite()
            && self.t_min >= 0.0
            && self.t_max <= 1.0
            && self.t_min < self.t_max
    }

    /// Transform into an instance's local space without changing the segment
    /// interval. This is the load-bearing non-uniform-scale law.
    pub fn transformed(&self, world_to_local: Mat4) -> Self {
        if !self.is_active() || !world_to_local.is_finite() {
            return Self::default();
        }
        let origin = world_to_local.transform_point3(Vec3::from_array(self.origin));
        let direction = world_to_local.transform_vector3(Vec3::from_array(self.direction));
        if !origin.is_finite() || !direction.is_finite() || direction.length_squared() <= 0.0 {
            return Self::default();
        }
        Self {
            origin: origin.to_array(),
            t_min: self.t_min,
            direction: direction.to_array(),
            t_max: self.t_max,
        }
    }
}

/// Two-sided Möller-Trumbore intersection over the open segment interval.
pub fn shadow_segment_triangle_t(segment: &ShadowSegment, v0: Vec3, v1: Vec3, v2: Vec3) -> f32 {
    if !segment.is_active() || !v0.is_finite() || !v1.is_finite() || !v2.is_finite() {
        return f32::INFINITY;
    }

    let origin = Vec3::from_array(segment.origin);
    let direction = Vec3::from_array(segment.direction);
    let edge1 = v1 - v0;
    let edge2 = v2 - v0;
    let p = direction.cross(edge2);
    let det = edge1.dot(p);
    if det == 0.0 || !det.is_finite() {
        return f32::INFINITY;
    }

    let inv_det = det.recip();
    let from_v0 = origin - v0;
    let u = from_v0.dot(p) * inv_det;
    if u < 0.0 || u > 1.0 {
        return f32::INFINITY;
    }
    let q = from_v0.cross(edge1);
    let v = direction.dot(q) * inv_det;
    if v < 0.0 || u + v > 1.0 {
        return f32::INFINITY;
    }
    let t = edge2.dot(q) * inv_det;
    if t.is_finite() && t > segment.t_min && t < segment.t_max {
        t
    } else {
        f32::INFINITY
    }
}

/// Stable 32-byte upload representation for the instance hierarchy.
/// `leaf != 0` selects an instance index; otherwise `child_or_instance` and
/// `child_or_instance + 1` are the two child nodes.
#[gpu_data]
pub struct ShadowTlasNode {
    pub min: [f32; 3],
    pub child_or_instance: u32,
    pub max: [f32; 3],
    pub leaf: u32,
}

const _: () = assert!(core::mem::size_of::<ShadowTlasNode>() == 32);
const _: () = assert!(core::mem::align_of::<ShadowTlasNode>() == 4);
const _: () = assert!(core::mem::offset_of!(ShadowTlasNode, child_or_instance) == 12);
const _: () = assert!(core::mem::offset_of!(ShadowTlasNode, max) == 16);
const _: () = assert!(core::mem::offset_of!(ShadowTlasNode, leaf) == 28);
/// One eligible mesh instance in the exact-shadow TLAS. The matrix is
/// pre-inverted on the host so candidate traversal never performs a matrix
/// inverse per ray.
#[gpu_data]
pub struct ShadowTlasInstance {
    pub world_to_local: Mat4,
    pub blas_index: u32,
    pub instance_id: u32,
    /// Bit 0: the source instance carried `MESH_FLAG_DYNAMIC` this frame.
    pub flags: u32,
    pub _pad0: u32,
}

/// `ShadowTlasInstance::flags` bit 0.
pub const SHADOW_TLAS_INSTANCE_DYNAMIC: u32 = 1;

const _: () = assert!(core::mem::size_of::<ShadowTlasInstance>() == 80);
const _: () = assert!(core::mem::align_of::<ShadowTlasInstance>() == 16);
const _: () = assert!(core::mem::offset_of!(ShadowTlasInstance, blas_index) == 64);

/// One frame's exact mesh-shadow world. Empty node/instance pointers are the
/// valid no-occluder representation.
#[gpu_data]
pub struct ShadowWorld {
    pub nodes: GpuPtr<ShadowTlasNode>,
    pub instances: GpuPtr<ShadowTlasInstance>,
    pub blases: GpuPtr<ShadowBlas>,
    pub node_count: u32,
    pub instance_count: u32,
    pub _pad0: [u32; 2],
}

const _: () = assert!(core::mem::size_of::<ShadowWorld>() == 40);
const _: () = assert!(core::mem::align_of::<ShadowWorld>() == 4);
const _: () = assert!(core::mem::offset_of!(ShadowWorld, node_count) == 24);

pub const SHADOW_STATE_INACTIVE: u32 = 0;
pub const SHADOW_STATE_VISIBLE: u32 = 1;
pub const SHADOW_STATE_OCCLUDED: u32 = 2;
pub const SHADOW_STATE_FAILED: u32 = 3;
pub const SHADOW_STATE_UNRESOLVED: u32 = 4;

pub const SHADOW_QUERY_VISIBLE: u32 = 0;
pub const SHADOW_QUERY_OCCLUDED: u32 = 1;
pub const SHADOW_QUERY_FAILED: u32 = 2;
pub const SHADOW_INVALID_PRIMITIVE: u32 = u32::MAX;

/// Per-query traversal diagnostics.
#[gpu_data]
pub struct ShadowQueryResult {
    pub status: u32,
    pub primitive_id: u32,
    pub hit_t: f32,
    pub node_tests: u32,
    pub triangle_tests: u32,
    pub max_stack_depth: u32,
}

const _: () = assert!(core::mem::size_of::<ShadowQueryResult>() == 24);
const _: () = assert!(core::mem::align_of::<ShadowQueryResult>() == 4);

/// Standalone query-batch data for traversal validation.
#[gpu_data]
pub struct ShadowBlasQueryData {
    pub blas: ShadowBlas,
    pub queries: GpuPtr<ShadowSegment>,
    pub results: GpuPtr<ShadowQueryResult>,
    pub query_count: u32,
    pub _pad0: [u32; 3],
}

const _: () = assert!(core::mem::size_of::<ShadowBlasQueryData>() == 80);
const _: () = assert!(core::mem::align_of::<ShadowBlasQueryData>() == 4);
const _: () = assert!(core::mem::offset_of!(ShadowBlasQueryData, queries) == 48);
const _: () = assert!(core::mem::offset_of!(ShadowBlasQueryData, results) == 56);
const _: () = assert!(core::mem::offset_of!(ShadowBlasQueryData, query_count) == 64);
/// Diagnostics from one world-space TLAS→BLAS any-hit query.
#[gpu_data]
pub struct ShadowWorldQueryResult {
    pub status: u32,
    pub instance_id: u32,
    pub primitive_id: u32,
    pub hit_t: f32,
    pub tlas_node_tests: u32,
    pub blas_node_tests: u32,
    pub triangle_tests: u32,
    pub max_stack_depth: u32,
}

const _: () = assert!(core::mem::size_of::<ShadowWorldQueryResult>() == 32);
const _: () = assert!(core::mem::align_of::<ShadowWorldQueryResult>() == 4);

#[gpu_data]
pub struct ShadowWorldQueryData {
    pub world: ShadowWorld,
    pub queries: GpuPtr<ShadowSegment>,
    pub results: GpuPtr<ShadowWorldQueryResult>,
    pub query_count: u32,
    pub _pad0: [u32; 3],
}

const _: () = assert!(core::mem::size_of::<ShadowWorldQueryData>() == 72);
const _: () = assert!(core::mem::align_of::<ShadowWorldQueryData>() == 4);
const _: () = assert!(core::mem::offset_of!(ShadowWorldQueryData, queries) == 40);
const _: () = assert!(core::mem::offset_of!(ShadowWorldQueryData, query_count) == 56);

/// Independent host-only f64 reference for parity tests.
#[cfg(not(target_arch = "spirv"))]
pub fn shadow_segment_triangle_oracle(
    segment: &ShadowSegment,
    v0: Vec3,
    v1: Vec3,
    v2: Vec3,
) -> bool {
    if !segment.is_active() {
        return false;
    }
    let cv = |v: Vec3| [v.x as f64, v.y as f64, v.z as f64];
    let sub = |a: [f64; 3], b: [f64; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let cross = |a: [f64; 3], b: [f64; 3]| {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    };

    let origin = [
        segment.origin[0] as f64,
        segment.origin[1] as f64,
        segment.origin[2] as f64,
    ];
    let direction = [
        segment.direction[0] as f64,
        segment.direction[1] as f64,
        segment.direction[2] as f64,
    ];
    let (v0, v1, v2) = (cv(v0), cv(v1), cv(v2));
    let edge1 = sub(v1, v0);
    let edge2 = sub(v2, v0);
    let p = cross(direction, edge2);
    let det = dot(edge1, p);
    if det == 0.0 || !det.is_finite() {
        return false;
    }
    let inv_det = det.recip();
    let from_v0 = sub(origin, v0);
    let u = dot(from_v0, p) * inv_det;
    if !(0.0..=1.0).contains(&u) {
        return false;
    }
    let q = cross(from_v0, edge1);
    let v = dot(direction, q) * inv_det;
    if v < 0.0 || u + v > 1.0 {
        return false;
    }
    let t = dot(edge2, q) * inv_det;
    t.is_finite() && t > segment.t_min as f64 && t < segment.t_max as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{Mat4, Quat};

    const V0: Vec3 = Vec3::new(-1.0, -1.0, 0.0);
    const V1: Vec3 = Vec3::new(1.0, -1.0, 0.0);
    const V2: Vec3 = Vec3::new(0.0, 1.0, 0.0);

    #[test]
    fn segment_biases_define_an_open_interval() {
        let segment = ShadowSegment::between(Vec3::new(0.0, 0.0, 1.0), Vec3::NEG_Z, 0.25, 0.5);
        assert!(segment.is_active());
        assert!((segment.t_min - 0.125).abs() < 1.0e-6);
        assert!((segment.t_max - 0.75).abs() < 1.0e-6);

        assert!(!ShadowSegment::between(Vec3::ZERO, Vec3::X, 0.5, 0.5).is_active());
        assert!(!ShadowSegment::between(Vec3::ZERO, Vec3::X, -1.0, 0.0).is_active());
        assert!(!ShadowSegment::between(Vec3::ZERO, Vec3::ZERO, 0.0, 0.0).is_active());
    }

    #[test]
    fn affine_transform_preserves_hit_parameter() {
        let world = Mat4::from_scale_rotation_translation(
            Vec3::new(2.0, 0.5, 3.0),
            Quat::from_rotation_y(0.4),
            Vec3::new(4.0, -2.0, 1.0),
        );
        let local = ShadowSegment::between(Vec3::new(0.0, 0.0, 1.0), Vec3::NEG_Z, 0.0, 0.0);
        let world_segment = ShadowSegment {
            origin: world
                .transform_point3(Vec3::from_array(local.origin))
                .to_array(),
            direction: world
                .transform_vector3(Vec3::from_array(local.direction))
                .to_array(),
            ..local
        };
        let recovered = world_segment.transformed(world.inverse());
        let local_t = shadow_segment_triangle_t(&local, V0, V1, V2);
        let recovered_t = shadow_segment_triangle_t(&recovered, V0, V1, V2);
        assert!((local_t - 0.5).abs() < 1.0e-6);
        assert!((recovered_t - local_t).abs() < 2.0e-6);
    }

    #[test]
    fn f32_port_agrees_with_independent_f64_oracle() {
        let cases = [
            ShadowSegment::between(Vec3::new(0.0, 0.0, 1.0), Vec3::NEG_Z, 0.0, 0.0),
            ShadowSegment::between(
                Vec3::new(2.0, 0.0, 1.0),
                Vec3::new(2.0, 0.0, -1.0),
                0.0,
                0.0,
            ),
            ShadowSegment::between(Vec3::new(0.0, 0.0, 1.0), Vec3::new(0.0, 0.0, 0.5), 0.0, 0.0),
            ShadowSegment::between(Vec3::new(0.0, 0.0, -1.0), Vec3::Z, 0.0, 0.0),
        ];
        for segment in cases {
            assert_eq!(
                shadow_segment_triangle_t(&segment, V0, V1, V2).is_finite(),
                shadow_segment_triangle_oracle(&segment, V0, V1, V2)
            );
        }
    }
}
