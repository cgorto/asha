//! Vertex deformation math shared by host tests and shaders: skinning
//! first, then the free-form lattice stack.
//!
//! ## Skinning is DUAL-QUATERNION, and palettes are RIGID
//!
//! [`crate::DualQuat`] is the palette entry, and the blend is
//! dual-quaternion linear blending (Kavan et al.): weighted-sum the four
//! entries after flipping each into slot 0's hemisphere, then renormalize.
//! The result is always a unit rigid transform, so a vertex between two
//! joints rides an interpolated screw motion instead of an averaged matrix.
//! Averaging matrices can collapse a bent limb's cross-section; dual-quaternion
//! blending preserves a rigid interpolated frame instead.
//!
//! Two contracts fall out, and both are enforced rather than assumed:
//!
//! - **Palettes carry no scale, shear, or reflection.** A dual quaternion
//!   cannot represent them, so [`crate::DualQuat::from_mat4`] refuses
//!   them at the call site instead of silently changing the transform.
//! - **Slot 0 is the blend pivot** and must hold the heaviest influence:
//!   the blended norm's floor is exactly `weights[0]`. Build weight rows
//!   through [`crate::JointWeights::canonical`]; glTF does not require
//!   sorted attributes.
//!
//! ## Lattice deformation
//!
//! Callers construct lattice transforms from a model-space OBB:
//! `lattice_to_model = T(center) * axes * S(2 * half_extents) * T(-0.5)`;
//! `model_to_lattice = inverse(lattice_to_model)`.

use crate::{DualQuat, JointWeights};
use crate::{GpuPtr, gpu_data};
use glam::{Mat3, Mat4, Vec3, Vec4};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

pub const MAX_DEFORMERS: usize = 4;
pub const MAX_LATTICE_POINTS: usize = 64;

/// Control offsets are inline, not frame-arena pointers: extracted component
/// bytes cannot carry main-thread GPU addresses.
#[repr(C)]
#[derive(Copy, Clone)]
#[cfg_attr(not(target_arch = "spirv"), derive(Debug))]
pub struct LatticeDeformer {
    /// Model space → lattice space; interior vertices map to `[0,1]³`.
    pub model_to_lattice: Mat4,
    /// Model-space transform; typically `inverse(model_to_lattice)`.
    pub lattice_to_model: Mat4,
    /// Control points per axis. Each axis is 2 (linear basis) or 4 (cubic Bernstein).
    /// Basis order per axis = resolution - 1. Quadratic is deliberately unsupported.
    pub resolution: [u32; 3],
    /// Falloff band width as a fraction of lattice extent, measured inward from the
    /// `[0,1]³` boundary. Zero clips hard at the boundary.
    pub falloff: f32,
    /// `resolution.x*y*z` entries with `[f32; 4]` stride, indexed
    /// k*res.y*res.x + j*res.x + i. Offsets are in LATTICE space, deltas from rest.
    /// All-zero = identity (ZII).
    pub offsets: [[f32; 4]; MAX_LATTICE_POINTS],
}

impl LatticeDeformer {
    #[cfg(not(target_arch = "spirv"))]
    pub const fn zeroed() -> Self {
        Self {
            model_to_lattice: Mat4::ZERO,
            lattice_to_model: Mat4::ZERO,
            resolution: [0; 3],
            falloff: 0.0,
            offsets: [[0.0; 4]; MAX_LATTICE_POINTS],
        }
    }

    #[cfg(target_arch = "spirv")]
    pub fn zeroed() -> Self {
        Self {
            model_to_lattice: Mat4::ZERO,
            lattice_to_model: Mat4::ZERO,
            resolution: [0; 3],
            falloff: 0.0,
            offsets: [[0.0; 4]; MAX_LATTICE_POINTS],
        }
    }
}

impl Default for LatticeDeformer {
    fn default() -> Self {
        Self::zeroed()
    }
}

#[cfg(not(target_arch = "spirv"))]
// SAFETY: `LatticeDeformer` is `#[repr(C)]`; all fields are plain float/u32
// GPU data with all bit patterns valid.
unsafe impl bytemuck::Zeroable for LatticeDeformer {}
#[cfg(not(target_arch = "spirv"))]
// SAFETY: see the `Zeroable` impl; there is no padding-sensitive reference
// or pointer validity in this inline-offset layout.
unsafe impl bytemuck::Pod for LatticeDeformer {}

const _: () = assert!(core::mem::size_of::<LatticeDeformer>() == 1168);
const _: () = assert!(core::mem::align_of::<LatticeDeformer>() == 16);
const _: () = assert!(core::mem::offset_of!(LatticeDeformer, lattice_to_model) == 64);
const _: () = assert!(core::mem::offset_of!(LatticeDeformer, resolution) == 128);
const _: () = assert!(core::mem::offset_of!(LatticeDeformer, falloff) == 140);
const _: () = assert!(core::mem::offset_of!(LatticeDeformer, offsets) == 144);

#[gpu_data(component)]
pub struct DeformerStack {
    pub count: u32,
    pub _pad0: [u32; 3],
    pub lattices: [LatticeDeformer; MAX_DEFORMERS],
}

impl DeformerStack {
    #[cfg(not(target_arch = "spirv"))]
    pub const fn zeroed() -> Self {
        Self {
            count: 0,
            _pad0: [0; 3],
            lattices: [LatticeDeformer::zeroed(); MAX_DEFORMERS],
        }
    }

    #[cfg(target_arch = "spirv")]
    pub fn zeroed() -> Self {
        Self {
            count: 0,
            _pad0: [0; 3],
            lattices: [LatticeDeformer::zeroed(); MAX_DEFORMERS],
        }
    }
}

const _: () = assert!(core::mem::size_of::<DeformerStack>() == 4688);
const _: () = assert!(core::mem::align_of::<DeformerStack>() == 16);
const _: () = assert!(core::mem::offset_of!(DeformerStack, _pad0) == 4);
const _: () = assert!(core::mem::offset_of!(DeformerStack, lattices) == 16);

pub trait OffsetSource {
    fn supports_offsets(&self, count: usize) -> bool;
    fn offset_at(&self, index: usize) -> [f32; 4];
}

#[cfg(not(target_arch = "spirv"))]
impl OffsetSource for &[[f32; 4]] {
    fn supports_offsets(&self, count: usize) -> bool {
        self.len() >= count
    }

    fn offset_at(&self, index: usize) -> [f32; 4] {
        self[index]
    }
}

#[cfg(not(target_arch = "spirv"))]
impl<const N: usize> OffsetSource for &[[f32; 4]; N] {
    fn supports_offsets(&self, count: usize) -> bool {
        N >= count
    }

    fn offset_at(&self, index: usize) -> [f32; 4] {
        self[index]
    }
}

pub trait DeformerSource {
    fn deformer_count(&self) -> u32;
    fn deformer_lattice(&self, lattice_index: usize) -> &LatticeDeformer;
    fn deformer_offsets_available(&self, lattice_index: usize, point_count: usize) -> bool;
    fn deformer_offset_at(&self, lattice_index: usize, offset_index: usize) -> [f32; 4];
}

#[cfg(target_arch = "spirv")]
impl DeformerSource for GpuPtr<DeformerStack> {
    fn deformer_count(&self) -> u32 {
        let ptr = *self;
        if ptr.is_null() { 0 } else { ptr.count }
    }

    fn deformer_lattice(&self, lattice_index: usize) -> &LatticeDeformer {
        &(*self).lattices[lattice_index]
    }

    fn deformer_offsets_available(&self, _lattice_index: usize, point_count: usize) -> bool {
        point_count <= MAX_LATTICE_POINTS
    }

    fn deformer_offset_at(&self, lattice_index: usize, offset_index: usize) -> [f32; 4] {
        (*self).lattices[lattice_index].offsets[offset_index]
    }
}

#[cfg(not(target_arch = "spirv"))]
impl DeformerSource for GpuPtr<DeformerStack> {
    fn deformer_count(&self) -> u32 {
        assert!(
            (*self).is_null(),
            "host GpuPtr<DeformerStack> is opaque; use &DeformerStack for CPU evaluation"
        );
        0
    }

    fn deformer_lattice(&self, _lattice_index: usize) -> &LatticeDeformer {
        panic!("host GpuPtr<DeformerStack> is opaque; use &DeformerStack for CPU evaluation")
    }

    fn deformer_offsets_available(&self, _lattice_index: usize, _point_count: usize) -> bool {
        false
    }

    fn deformer_offset_at(&self, _lattice_index: usize, _offset_index: usize) -> [f32; 4] {
        [0.0; 4]
    }
}

#[cfg(not(target_arch = "spirv"))]
impl DeformerSource for &DeformerStack {
    fn deformer_count(&self) -> u32 {
        self.count
    }

    fn deformer_lattice(&self, lattice_index: usize) -> &LatticeDeformer {
        &self.lattices[lattice_index]
    }

    fn deformer_offsets_available(&self, _lattice_index: usize, point_count: usize) -> bool {
        point_count <= MAX_LATTICE_POINTS
    }

    fn deformer_offset_at(&self, lattice_index: usize, offset_index: usize) -> [f32; 4] {
        self.lattices[lattice_index].offsets[offset_index]
    }
}

/// Narrow read-only seam for a joint palette and its vertex-parallel weights.
/// Shader callers use paired `GpuPtr`s; host tests use paired ordinary slices.
pub trait SkinningSource {
    fn skinning_active(&self) -> bool;
    fn joint_transform(&self, joint_index: usize) -> &DualQuat;
    fn joint_weights(&self, vertex_index: usize) -> &JointWeights;
}

#[cfg(target_arch = "spirv")]
impl SkinningSource for (GpuPtr<DualQuat>, GpuPtr<JointWeights>) {
    fn skinning_active(&self) -> bool {
        let transforms_null = self.0.is_null();
        let weights_null = self.1.is_null();
        assert!(transforms_null == weights_null);
        !transforms_null
    }

    fn joint_transform(&self, joint_index: usize) -> &DualQuat {
        &self.0[joint_index]
    }

    fn joint_weights(&self, vertex_index: usize) -> &JointWeights {
        &self.1[vertex_index]
    }
}

#[cfg(not(target_arch = "spirv"))]
impl SkinningSource for (GpuPtr<DualQuat>, GpuPtr<JointWeights>) {
    fn skinning_active(&self) -> bool {
        let transforms_null = self.0.is_null();
        let weights_null = self.1.is_null();
        assert!(
            transforms_null == weights_null,
            "joint transform and weight pointers must be null or non-null together"
        );
        assert!(
            transforms_null,
            "host GPU pointers are opaque; use joint transform and weight slices"
        );
        false
    }

    fn joint_transform(&self, _joint_index: usize) -> &DualQuat {
        panic!("host GPU pointers are opaque; use joint transform and weight slices")
    }

    fn joint_weights(&self, _vertex_index: usize) -> &JointWeights {
        panic!("host GPU pointers are opaque; use joint transform and weight slices")
    }
}

#[cfg(not(target_arch = "spirv"))]
impl SkinningSource for (&[DualQuat], &[JointWeights]) {
    fn skinning_active(&self) -> bool {
        let transforms_empty = self.0.is_empty();
        let weights_empty = self.1.is_empty();
        assert!(
            transforms_empty == weights_empty,
            "joint transform and weight slices must be empty or non-empty together"
        );
        !transforms_empty
    }

    fn joint_transform(&self, joint_index: usize) -> &DualQuat {
        &self.0[joint_index]
    }

    fn joint_weights(&self, vertex_index: usize) -> &JointWeights {
        &self.1[vertex_index]
    }
}

struct StackOffsets<'a, S: DeformerSource> {
    source: &'a S,
    lattice_index: usize,
}

impl<S: DeformerSource> OffsetSource for StackOffsets<'_, S> {
    fn supports_offsets(&self, count: usize) -> bool {
        self.source
            .deformer_offsets_available(self.lattice_index, count)
    }

    fn offset_at(&self, index: usize) -> [f32; 4] {
        self.source.deformer_offset_at(self.lattice_index, index)
    }
}

#[derive(Copy, Clone)]
struct LatticeSample {
    q: Vec3,
    weight: f32,
    delta: Vec3,
    gradient: [Vec3; 3],
    active: bool,
}

pub fn lattice_point_count(resolution: [u32; 3]) -> usize {
    let valid = supported_axis(resolution[0])
        && supported_axis(resolution[1])
        && supported_axis(resolution[2]);
    debug_assert!(
        valid || resolution == [0, 0, 0],
        "lattice resolution axes must be 2 or 4; quadratic is not supported"
    );
    if !valid {
        return 0;
    }
    let count = (resolution[0] * resolution[1] * resolution[2]) as usize;
    debug_assert!(count <= MAX_LATTICE_POINTS);
    count
}

pub fn lattice_basis(resolution: u32, t: f32) -> [f32; 4] {
    debug_assert!(
        supported_axis(resolution),
        "lattice basis resolution must be 2 or 4; quadratic is not supported"
    );
    if resolution == 2 {
        [1.0 - t, t, 0.0, 0.0]
    } else if resolution == 4 {
        let u = 1.0 - t;
        let tt = t * t;
        let uu = u * u;
        [u * uu, 3.0 * t * uu, 3.0 * tt * u, tt * t]
    } else {
        [0.0; 4]
    }
}

pub fn lattice_basis_gradient(resolution: u32, t: f32) -> [f32; 4] {
    debug_assert!(
        supported_axis(resolution),
        "lattice basis resolution must be 2 or 4; quadratic is not supported"
    );
    if resolution == 2 {
        [-1.0, 1.0, 0.0, 0.0]
    } else if resolution == 4 {
        let u = 1.0 - t;
        let tt = t * t;
        let uu = u * u;
        [
            -3.0 * uu,
            3.0 * uu - 6.0 * t * u,
            6.0 * t * u - 3.0 * tt,
            3.0 * tt,
        ]
    } else {
        [0.0; 4]
    }
}

pub fn lattice_apply<O: OffsetSource>(
    lat: &LatticeDeformer,
    offsets: O,
    p: Vec3,
    n: Vec3,
) -> (Vec3, Vec3) {
    let sample = lattice_sample(lat, &offsets, p);
    if !sample.active {
        return (p, n);
    }

    let gradient_zero = sample.gradient[0] == Vec3::ZERO
        && sample.gradient[1] == Vec3::ZERO
        && sample.gradient[2] == Vec3::ZERO;
    if sample.delta == Vec3::ZERO && gradient_zero {
        return (p, n);
    }

    let p_deformed = lattice_apply_sampled_position(lat, &sample, p);

    if gradient_zero {
        return (p_deformed, n);
    }

    let j = lattice_jacobian_from_sample(lat, &sample);
    // Use `J * n`; extreme nonuniform scales slightly affect normals.
    let n_deformed = j * n;
    let n_len_sq = n_deformed.length_squared();
    let n_deformed = if n_len_sq > 1.0e-20 {
        n_deformed.normalize()
    } else {
        n
    };
    (p_deformed, n_deformed)
}

pub fn lattice_apply_position<O: OffsetSource>(lat: &LatticeDeformer, offsets: O, p: Vec3) -> Vec3 {
    let sample = lattice_sample(lat, &offsets, p);
    lattice_apply_sampled_position(lat, &sample, p)
}

fn lattice_apply_sampled_position(lat: &LatticeDeformer, sample: &LatticeSample, p: Vec3) -> Vec3 {
    if !sample.active || sample.delta == Vec3::ZERO {
        return p;
    }
    let q_deformed = sample.q + sample.delta * sample.weight;
    (lat.lattice_to_model * q_deformed.extend(1.0)).truncate()
}

pub fn lattice_jacobian<O: OffsetSource>(lat: &LatticeDeformer, offsets: O, p: Vec3) -> Mat3 {
    let sample = lattice_sample(lat, &offsets, p);
    if !sample.active
        || (sample.gradient[0] == Vec3::ZERO
            && sample.gradient[1] == Vec3::ZERO
            && sample.gradient[2] == Vec3::ZERO)
    {
        return Mat3::IDENTITY;
    }
    lattice_jacobian_from_sample(lat, &sample)
}

pub fn deform_apply<S: DeformerSource>(stack: S, p: Vec3, n: Vec3) -> (Vec3, Vec3) {
    let raw_count = stack.deformer_count();
    debug_assert!(raw_count <= MAX_DEFORMERS as u32);
    let count = min_usize(raw_count as usize, MAX_DEFORMERS);
    if count == 0 {
        return (p, n);
    }

    let mut p_deformed = p;
    let mut n_deformed = n;
    let mut i = 0;
    while i < count {
        let lat = stack.deformer_lattice(i);
        let offsets = StackOffsets {
            source: &stack,
            lattice_index: i,
        };
        let applied = lattice_apply(lat, offsets, p_deformed, n_deformed);
        p_deformed = applied.0;
        n_deformed = applied.1;
        i += 1;
    }
    (p_deformed, n_deformed)
}

pub fn deform_apply_position<S: DeformerSource>(stack: S, p: Vec3) -> Vec3 {
    let raw_count = stack.deformer_count();
    debug_assert!(raw_count <= MAX_DEFORMERS as u32);
    let count = min_usize(raw_count as usize, MAX_DEFORMERS);
    if count == 0 {
        return p;
    }

    let mut p_deformed = p;
    let mut i = 0;
    while i < count {
        let lat = stack.deformer_lattice(i);
        let offsets = StackOffsets {
            source: &stack,
            lattice_index: i,
        };
        p_deformed = lattice_apply_position(lat, offsets, p_deformed);
        i += 1;
    }
    p_deformed
}

/// Skinning and deformation stay behind this wrapper so mesh vertex paths do
/// not need to know which stages are active. Skinning always runs first.
pub fn evaluate_vertex<M, W>(
    joint_transforms: M,
    joint_weights: W,
    deformer_slot: u32,
    deformers: GpuPtr<DeformerStack>,
    vertex_index: u32,
    position: Vec3,
    normal: Vec3,
) -> (Vec3, Vec3)
where
    (M, W): SkinningSource,
{
    let skinning = (joint_transforms, joint_weights);
    let (position, normal) = if skinning.skinning_active() {
        let weights = skinning.joint_weights(vertex_index as usize);
        let skin_transform = blend_skin_dual_quat(&skinning, weights);
        (
            skin_position(&skin_transform, position),
            skin_normal(&skin_transform, normal),
        )
    } else {
        (position, normal)
    };

    if deformer_slot == 0 || deformers.is_null() {
        return (position, normal);
    }
    deform_apply(
        deformers.offset((deformer_slot - 1) as i64),
        position,
        normal,
    )
}

/// Position-only twin for depth prepass. It shares `skin_position` and
/// `lattice_apply_sampled_position` with `evaluate_vertex`; Equal depth depends
/// on bit-identical clip positions.
pub fn evaluate_vertex_position<M, W>(
    joint_transforms: M,
    joint_weights: W,
    deformer_slot: u32,
    deformers: GpuPtr<DeformerStack>,
    vertex_index: u32,
    position: Vec3,
) -> Vec3
where
    (M, W): SkinningSource,
{
    let skinning = (joint_transforms, joint_weights);
    let position = if skinning.skinning_active() {
        let weights = skinning.joint_weights(vertex_index as usize);
        let skin_transform = blend_skin_dual_quat(&skinning, weights);
        skin_position(&skin_transform, position)
    } else {
        position
    };

    if deformer_slot == 0 || deformers.is_null() {
        return position;
    }
    deform_apply_position(deformers.offset((deformer_slot - 1) as i64), position)
}

/// Dual-quaternion linear blend (Kavan et al.): weighted-sum the four
/// palette entries with each quaternion sign-flipped into slot 0's
/// hemisphere, then normalize by the real part. Slot 0 is the pivot by
/// CONTRACT — it must carry the vertex's leading influence (the host's
/// skin-cull derivation asserts `weights[0] > 0`). Palette reads are
/// unconditional for every slot, including zero-weight slots; the host must
/// therefore prove all four indices in range before publishing the pointer.
///
/// The result is a unit rigid transform, so weight scale cancels and there
/// is no linear-blend matrix collapse at deep flexion. Near-antipodal
/// co-influencing joints (a relative rotation approaching 360°) would
/// degenerate the sum; the hot path clamps the norm and the host's bounds
/// derivation refuses such palettes loudly.
#[inline]
fn blend_skin_dual_quat<S: SkinningSource + ?Sized>(
    skinning: &S,
    weights: &JointWeights,
) -> DualQuat {
    let q0 = skinning.joint_transform(weights.joint_indices[0] as usize);
    let q1 = skinning.joint_transform(weights.joint_indices[1] as usize);
    let q2 = skinning.joint_transform(weights.joint_indices[2] as usize);
    let q3 = skinning.joint_transform(weights.joint_indices[3] as usize);
    let pivot = Vec4::from_array(q0.real);
    let w1 = hemisphere_weight(weights.weights[1], Vec4::from_array(q1.real), pivot);
    let w2 = hemisphere_weight(weights.weights[2], Vec4::from_array(q2.real), pivot);
    let w3 = hemisphere_weight(weights.weights[3], Vec4::from_array(q3.real), pivot);
    let real = pivot * weights.weights[0]
        + Vec4::from_array(q1.real) * w1
        + Vec4::from_array(q2.real) * w2
        + Vec4::from_array(q3.real) * w3;
    let dual = Vec4::from_array(q0.dual) * weights.weights[0]
        + Vec4::from_array(q1.dual) * w1
        + Vec4::from_array(q2.dual) * w2
        + Vec4::from_array(q3.dual) * w3;
    let inverse_norm = 1.0 / real.length().max(1.0e-6);
    DualQuat {
        real: (real * inverse_norm).to_array(),
        dual: (dual * inverse_norm).to_array(),
    }
}

#[inline]
fn hemisphere_weight(weight: f32, real: Vec4, pivot: Vec4) -> f32 {
    if real.dot(pivot) < 0.0 {
        -weight
    } else {
        weight
    }
}

/// Shared by full and position-only paths: Equal depth depends on this exact
/// arithmetic staying in one function.
#[inline]
fn skin_position(skin_transform: &DualQuat, position: Vec3) -> Vec3 {
    skin_transform.transform_point3(position)
}

#[inline]
fn skin_normal(skin_transform: &DualQuat, normal: Vec3) -> Vec3 {
    let skinned = skin_transform.rotate_vector3(normal);
    let length_squared = skinned.length_squared();
    if length_squared > 1.0e-20 && length_squared < f32::INFINITY {
        skinned / length_squared.sqrt()
    } else {
        Vec3::Y
    }
}

#[cfg(not(target_arch = "spirv"))]
pub fn max_offset_magnitude<O: OffsetSource>(offsets: O, count: usize) -> f32 {
    debug_assert!(count <= MAX_LATTICE_POINTS);
    let count = min_usize(count, MAX_LATTICE_POINTS);
    if !offsets.supports_offsets(count) {
        return 0.0;
    }

    let mut max_len_sq = 0.0;
    let mut i = 0;
    while i < count {
        let raw = offsets.offset_at(i);
        let o = Vec3::new(raw[0], raw[1], raw[2]);
        let len_sq = o.length_squared();
        if len_sq > max_len_sq {
            max_len_sq = len_sq;
        }
        i += 1;
    }
    max_len_sq.sqrt()
}

#[cfg(not(target_arch = "spirv"))]
pub fn max_linear_scale(m: &Mat4) -> f32 {
    crate::max_world_scale(m)
}

fn lattice_sample<O: OffsetSource + ?Sized>(
    lat: &LatticeDeformer,
    offsets: &O,
    p: Vec3,
) -> LatticeSample {
    let point_count = lattice_point_count(lat.resolution);
    if point_count == 0 || !offsets.supports_offsets(point_count) {
        return inactive_sample();
    }

    let q = (lat.model_to_lattice * p.extend(1.0)).truncate();
    let d = inside_distance(q);
    if d < 0.0 {
        return inactive_sample();
    }

    let weight = falloff_weight(d, lat.falloff);
    if weight == 0.0 {
        return inactive_sample();
    }

    let q_eval = Vec3::new(clamp01(q.x), clamp01(q.y), clamp01(q.z));
    let bx = lattice_basis(lat.resolution[0], q_eval.x);
    let by = lattice_basis(lat.resolution[1], q_eval.y);
    let bz = lattice_basis(lat.resolution[2], q_eval.z);
    let gx = lattice_basis_gradient(lat.resolution[0], q_eval.x);
    let gy = lattice_basis_gradient(lat.resolution[1], q_eval.y);
    let gz = lattice_basis_gradient(lat.resolution[2], q_eval.z);
    let res = [
        lat.resolution[0] as usize,
        lat.resolution[1] as usize,
        lat.resolution[2] as usize,
    ];

    LatticeSample {
        q,
        weight,
        delta: blend_offsets(offsets, res, &bx, &by, &bz),
        gradient: [
            blend_offsets(offsets, res, &gx, &by, &bz),
            blend_offsets(offsets, res, &bx, &gy, &bz),
            blend_offsets(offsets, res, &bx, &by, &gz),
        ],
        active: true,
    }
}

fn blend_offsets<O: OffsetSource + ?Sized>(
    offsets: &O,
    resolution: [usize; 3],
    wx: &[f32; 4],
    wy: &[f32; 4],
    wz: &[f32; 4],
) -> Vec3 {
    let rx = resolution[0];
    let ry = resolution[1];
    let rz = resolution[2];

    let mut rows = [Vec3::ZERO; 16];
    let mut k = 0;
    while k < rz {
        let mut j = 0;
        while j < ry {
            let mut sum = Vec3::ZERO;
            let mut i = 0;
            while i < rx {
                let raw = offsets.offset_at(k * ry * rx + j * rx + i);
                sum += Vec3::new(raw[0], raw[1], raw[2]) * wx[i];
                i += 1;
            }
            rows[k * 4 + j] = sum;
            j += 1;
        }
        k += 1;
    }

    let mut cols = [Vec3::ZERO; 4];
    k = 0;
    while k < rz {
        let mut sum = Vec3::ZERO;
        let mut j = 0;
        while j < ry {
            sum += rows[k * 4 + j] * wy[j];
            j += 1;
        }
        cols[k] = sum;
        k += 1;
    }

    let mut sum = Vec3::ZERO;
    k = 0;
    while k < rz {
        sum += cols[k] * wz[k];
        k += 1;
    }
    sum
}

fn lattice_jacobian_from_sample(lat: &LatticeDeformer, sample: &LatticeSample) -> Mat3 {
    // Falloff is treated as locally constant; the gradient-of-falloff term is
    // deliberately dropped.
    let j_lattice = Mat3::from_cols(
        Vec3::X + sample.gradient[0] * sample.weight,
        Vec3::Y + sample.gradient[1] * sample.weight,
        Vec3::Z + sample.gradient[2] * sample.weight,
    );
    linear_part(&lat.lattice_to_model) * j_lattice * linear_part(&lat.model_to_lattice)
}

fn linear_part(m: &Mat4) -> Mat3 {
    Mat3::from_cols(
        (*m * Vec4::new(1.0, 0.0, 0.0, 0.0)).truncate(),
        (*m * Vec4::new(0.0, 1.0, 0.0, 0.0)).truncate(),
        (*m * Vec4::new(0.0, 0.0, 1.0, 0.0)).truncate(),
    )
}

fn inactive_sample() -> LatticeSample {
    LatticeSample {
        q: Vec3::ZERO,
        weight: 0.0,
        delta: Vec3::ZERO,
        gradient: [Vec3::ZERO; 3],
        active: false,
    }
}

fn inside_distance(q: Vec3) -> f32 {
    q.x.min(1.0 - q.x)
        .min(q.y.min(1.0 - q.y))
        .min(q.z.min(1.0 - q.z))
}

fn falloff_weight(d: f32, falloff: f32) -> f32 {
    if d < 0.0 {
        0.0
    } else if falloff <= 0.0 {
        1.0
    } else if d <= 0.0 {
        0.0
    } else {
        let t = clamp01(d / falloff);
        t * t * (3.0 - 2.0 * t)
    }
}

fn supported_axis(resolution: u32) -> bool {
    resolution == 2 || resolution == 4
}

fn clamp01(v: f32) -> f32 {
    v.max(0.0).min(1.0)
}

fn min_usize(a: usize, b: usize) -> usize {
    if a < b { a } else { b }
}
