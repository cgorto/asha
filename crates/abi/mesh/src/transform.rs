//! Host-side transform hierarchy math.
//!
//! glTF authors each node as either a matrix or TRS. TRS decomposition
//! cannot represent inherited shear from a non-uniformly scaled parent
//! under a rotated child, so asha composes hierarchy matrices in `Mat4`
//! and never decomposes mid-chain.

use crate::DrawTransform;
use glam::{Mat4, Quat, Vec3};

/// Degeneracy epsilon for transform determinants (and quaternion length²).
/// `normal_matrix` asserts determinants above it; hosts that shrink things
/// toward nothing must hide anything at or below it rather than submit it —
/// exact-zero tests are not enough, a decay's last frame can sample scales
/// that are tiny but nonzero.
pub const DET_EPS: f32 = 1.0e-8;

/// Local TRS authored by an asset.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Trs {
    pub translation: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
}

impl Trs {
    pub const IDENTITY: Self = Self {
        translation: Vec3::ZERO,
        rotation: Quat::IDENTITY,
        scale: Vec3::ONE,
    };

    pub fn to_matrix(&self) -> Mat4 {
        let rotation_len2 = self.rotation.length_squared();
        assert!(
            self.translation.is_finite(),
            "TRS translation must be finite"
        );
        assert!(
            rotation_len2.is_finite() && rotation_len2 > DET_EPS,
            "TRS rotation quaternion must be finite and non-zero"
        );
        assert!(
            self.scale.is_finite() && self.scale.abs().min_element() > DET_EPS,
            "TRS scale must be finite and non-zero"
        );
        Mat4::from_translation(self.translation)
            * Mat4::from_quat(self.rotation)
            * Mat4::from_scale(self.scale)
    }
}

impl Default for Trs {
    fn default() -> Self {
        // Host-authored transforms intentionally deviate from all-zero ZII:
        // a zero quaternion or zero scale is never meaningful.
        Self::IDENTITY
    }
}

/// glTF node local transform as authored; a node is either matrix-form or TRS.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum LocalTransform {
    Matrix(Mat4),
    Trs(Trs),
}

impl LocalTransform {
    pub fn matrix(&self) -> Mat4 {
        match self {
            Self::Matrix(matrix) => *matrix,
            Self::Trs(trs) => trs.to_matrix(),
        }
    }
}

pub fn compose(parent_world: Mat4, local: &LocalTransform) -> Mat4 {
    parent_world * local.matrix()
}

pub fn normal_matrix(world: Mat4) -> Mat4 {
    let det = world.determinant();
    assert!(
        det.is_finite() && det.abs() > DET_EPS,
        "world transform determinant must be finite and non-zero, got {det:e}"
    );
    assert!(
        det > 0.0,
        "mirrored world transform flips triangle winding; unsupported"
    );
    world.inverse().transpose()
}

pub fn world_transform(world: Mat4) -> DrawTransform {
    DrawTransform {
        model_to_world: world,
        model_to_world_normal: normal_matrix(world),
    }
}
