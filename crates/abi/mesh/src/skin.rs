//! Skinning vocabulary: the vertex-parallel weights and the unit dual
//! quaternion palette entries that blend them.

use abi_core::gpu_data;
use glam::Vec3;
#[cfg(not(target_arch = "spirv"))]
use glam::{Mat4, Vec4};

/// Skinning payload lives in the stream table so the mesh ABI does not move.
///
/// SLOT 0 IS THE BLEND PIVOT. Dual-quaternion skinning flips every other
/// influence into slot 0's hemisphere, and the norm floor of the blended
/// sum is exactly `weights[0]` — so slot 0 must be positive, and the
/// heaviest influence makes the blend best conditioned. Build rows through
/// [`Self::canonical`] rather than trusting an exporter's attribute order:
/// glTF does not require weights to be sorted.
#[gpu_data]
pub struct JointWeights {
    pub joint_indices: [u32; 4],
    pub weights: [f32; 4],
}

impl JointWeights {
    /// Reorder four influences by DESCENDING weight, making slot 0 the
    /// heaviest — the pivot law above. Weights must be finite.
    #[cfg(not(target_arch = "spirv"))]
    pub fn canonical(joint_indices: [u32; 4], weights: [f32; 4]) -> Self {
        let mut influences: [(f32, u32); 4] =
            core::array::from_fn(|influence| (weights[influence], joint_indices[influence]));
        influences.sort_unstable_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .expect("joint weights must be finite to canonicalize")
        });
        Self {
            joint_indices: influences.map(|(_, joint)| joint),
            weights: influences.map(|(weight, _)| weight),
        }
    }
}

const _: () = assert!(core::mem::size_of::<JointWeights>() == 32);
const _: () = assert!(core::mem::align_of::<JointWeights>() == 4);
const _: () = assert!(core::mem::offset_of!(JointWeights, weights) == 16);

/// Rigid joint transform as a unit dual quaternion: `real` is the rotation
/// `(x, y, z, w)`, `dual` is the translation moment `0.5 · t ⊗ real`.
///
/// This is the palette representation for skinning — dual quaternions blend
/// without the linear-blend collapse (the pancake elbow) and preserve joint
/// volume, at the deliberate cost that palettes are RIGID ONLY: scale and
/// shear are unrepresentable, and [`Self::from_mat4`] refuses them loudly.
/// The zero value is degenerate like a zero matrix palette; palettes are
/// written from [`Self::IDENTITY`] or a live pose before they are read.
#[gpu_data]
pub struct DualQuat {
    /// Rotation quaternion `(x, y, z, w)`.
    pub real: [f32; 4],
    /// Translation moment `0.5 · (t.x, t.y, t.z, 0) ⊗ real`, `(x, y, z, w)`.
    pub dual: [f32; 4],
}

impl DualQuat {
    pub const IDENTITY: Self = Self {
        real: [0.0, 0.0, 0.0, 1.0],
        dual: [0.0; 4],
    };

    /// Pack a finite RIGID glam matrix (rotation + translation, positive
    /// determinant). Scale, shear, and reflection are contract violations:
    /// a dual quaternion cannot carry them, so accepting one would silently
    /// change the transform.
    #[cfg(not(target_arch = "spirv"))]
    pub fn from_mat4(matrix: Mat4) -> Self {
        assert!(matrix.is_finite(), "joint transform must be finite");
        assert!(
            matrix.row(3) == Vec4::W,
            "DualQuat requires an affine Mat4 with row 3 equal to [0, 0, 0, 1]"
        );
        let linear = glam::Mat3::from_mat4(matrix);
        let gram = linear.transpose() * linear;
        let orthonormal_error = (gram - glam::Mat3::IDENTITY)
            .to_cols_array()
            .into_iter()
            .fold(0.0f32, |max, value| max.max(value.abs()));
        assert!(
            orthonormal_error <= 1.0e-3,
            "DualQuat requires a rigid transform (orthonormality error {orthonormal_error})"
        );
        assert!(
            linear.determinant() > 0.0,
            "DualQuat requires a rotation, not a reflection"
        );
        let rotation = glam::Quat::from_mat3(&linear).normalize();
        Self::from_rotation_translation(rotation, matrix.w_axis.truncate())
    }

    /// Pack a unit rotation quaternion and a translation.
    #[cfg(not(target_arch = "spirv"))]
    pub fn from_rotation_translation(rotation: glam::Quat, translation: Vec3) -> Self {
        assert!(
            rotation.is_finite() && translation.is_finite(),
            "joint transform must be finite"
        );
        assert!(
            (rotation.length_squared() - 1.0).abs() <= 1.0e-3,
            "DualQuat rotation must be a unit quaternion"
        );
        let r = Vec4::new(rotation.x, rotation.y, rotation.z, rotation.w);
        let t = translation;
        // dual = 0.5 · (t, 0) ⊗ real, expanded by hand: quaternion product
        // (v₁, w₁) ⊗ (v₂, w₂) = (w₁v₂ + w₂v₁ + v₁×v₂, w₁w₂ − v₁·v₂).
        Self {
            real: r.to_array(),
            dual: [
                0.5 * (t.x * r.w + t.y * r.z - t.z * r.y),
                0.5 * (-t.x * r.z + t.y * r.w + t.z * r.x),
                0.5 * (t.x * r.y - t.y * r.x + t.z * r.w),
                -0.5 * (t.x * r.x + t.y * r.y + t.z * r.z),
            ],
        }
    }

    /// The translation this transform applies: `2 · dual ⊗ conj(real)`.
    #[inline]
    pub fn translation(&self) -> Vec3 {
        let r = Vec3::new(self.real[0], self.real[1], self.real[2]);
        let d = Vec3::new(self.dual[0], self.dual[1], self.dual[2]);
        2.0 * (self.real[3] * d - self.dual[3] * r + r.cross(d))
    }

    /// Rotate a vector by the real part (assumed unit).
    #[inline]
    pub fn rotate_vector3(&self, vector: Vec3) -> Vec3 {
        #[cfg(not(target_arch = "spirv"))]
        assert!(vector.is_finite(), "vector must be finite");
        let axis = Vec3::new(self.real[0], self.real[1], self.real[2]);
        vector + 2.0 * axis.cross(axis.cross(vector) + self.real[3] * vector)
    }

    /// Transform a point: rotate, then translate.
    #[inline]
    pub fn transform_point3(&self, point: Vec3) -> Vec3 {
        self.rotate_vector3(point) + self.translation()
    }
}

const _: () = assert!(core::mem::size_of::<DualQuat>() == 32);
const _: () = assert!(core::mem::align_of::<DualQuat>() == 4);
const _: () = assert!(core::mem::offset_of!(DualQuat, dual) == 16);
