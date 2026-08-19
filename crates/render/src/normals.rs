//! Cold-path normal repair for meshes with split flat-exported vertices.
//!
//! Positions are welded by quantized coordinates, then face normals are
//! gathered by crease angle and area-weighted before writing the result.

use bevy::math::Vec3;
use bevy::mesh::{Indices, Mesh, PrimitiveTopology, VertexAttributeValues};

/// Absolute position quantization used for welding coincident vertices.
const WELD_EPSILON: f32 = 1.0e-5;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SoftenError(String);

impl std::fmt::Display for SoftenError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for SoftenError {}

/// Averages nearby face normals across the requested crease angle.
///
/// Returns the number of vertex normals that changed. Existing normals provide
/// each vertex's reference direction; unnormalized face normals supply area
/// weighting, while degenerate faces contribute nothing.
pub fn soften_normals(mesh: &mut Mesh, crease_degrees: f32) -> Result<usize, SoftenError> {
    if mesh.primitive_topology() != PrimitiveTopology::TriangleList {
        return Err(SoftenError("mesh is not a TriangleList".into()));
    }
    let positions = match mesh.attribute(Mesh::ATTRIBUTE_POSITION) {
        Some(VertexAttributeValues::Float32x3(values)) => values.clone(),
        _ => return Err(SoftenError("mesh has no Float32x3 POSITION".into())),
    };
    let normals = match mesh.attribute(Mesh::ATTRIBUTE_NORMAL) {
        Some(VertexAttributeValues::Float32x3(values)) => values.clone(),
        _ => return Err(SoftenError("mesh has no Float32x3 NORMAL".into())),
    };
    let indices: Vec<u32> = match mesh.indices() {
        Some(Indices::U32(values)) => values.clone(),
        Some(Indices::U16(values)) => values.iter().map(|&i| i as u32).collect(),
        None => return Err(SoftenError("mesh is not indexed".into())),
    };
    if positions.len() != normals.len() {
        return Err(SoftenError("POSITION and NORMAL counts differ".into()));
    }
    if indices.len() % 3 != 0 {
        return Err(SoftenError("index count is not a multiple of three".into()));
    }

    let mut face_weighted = Vec::with_capacity(indices.len() / 3);
    for triangle in indices.chunks_exact(3) {
        let corner = |slot: usize| -> Result<Vec3, SoftenError> {
            positions
                .get(triangle[slot] as usize)
                .map(|p| Vec3::from_array(*p))
                .ok_or_else(|| SoftenError("index exceeds vertex count".into()))
        };
        let (a, b, c) = (corner(0)?, corner(1)?, corner(2)?);
        face_weighted.push((b - a).cross(c - a));
    }

    let mut buckets: std::collections::HashMap<[i64; 3], Vec<u32>> = Default::default();
    for (face, triangle) in indices.chunks_exact(3).enumerate() {
        for &corner in triangle {
            buckets
                .entry(weld_key(&positions[corner as usize]))
                .or_default()
                .push(face as u32);
        }
    }

    let threshold = crease_degrees.to_radians().cos();
    let mut out = normals.clone();
    let mut moved = 0usize;
    for (vertex, normal) in normals.iter().enumerate() {
        let reference = Vec3::from_array(*normal);
        let Some(faces) = buckets.get(&weld_key(&positions[vertex])) else {
            continue;
        };
        let mut sum = Vec3::ZERO;
        for &face in faces {
            let weighted = face_weighted[face as usize];
            let Some(direction) = weighted.try_normalize() else {
                continue; // Degenerate triangle; it has no opinion.
            };
            if direction.dot(reference) >= threshold {
                sum += weighted;
            }
        }
        if let Some(softened) = sum.try_normalize() {
            if softened.dot(reference) < 0.99999 {
                moved += 1;
            }
            out[vertex] = softened.to_array();
        }
    }

    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, out);
    Ok(moved)
}

fn weld_key(position: &[f32; 3]) -> [i64; 3] {
    position.map(|v| (v / WELD_EPSILON).round() as i64)
}

#[cfg(test)]
mod tests {
    use bevy::asset::RenderAssetUsages;

    use super::*;

    /// Builds two split-normal triangles meeting at `angle` degrees.
    fn hinge(angle_degrees: f32) -> Mesh {
        let half = (angle_degrees.to_radians()) * 0.5;
        let (s, c) = (half.sin(), half.cos());
        let left = [-c, s, 0.0];
        let right = [c, s, 0.0];
        let positions = vec![
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            left,
            [0.0, 0.0, 0.0],
            right,
            [0.0, 0.0, 1.0],
        ];
        let face = |a: [f32; 3], b: [f32; 3], c: [f32; 3]| {
            let (a, b, c) = (Vec3::from(a), Vec3::from(b), Vec3::from(c));
            (b - a).cross(c - a).normalize().to_array()
        };
        let ln = face(positions[0], positions[1], positions[2]);
        let rn = face(positions[3], positions[4], positions[5]);
        let mut mesh = Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::default(),
        );
        mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
        mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, vec![ln, ln, ln, rn, rn, rn]);
        mesh.insert_indices(Indices::U32(vec![0, 1, 2, 3, 4, 5]));
        mesh
    }

    fn normals_of(mesh: &Mesh) -> Vec<[f32; 3]> {
        match mesh.attribute(Mesh::ATTRIBUTE_NORMAL) {
            Some(VertexAttributeValues::Float32x3(values)) => values.clone(),
            _ => panic!("no normals"),
        }
    }

    #[test]
    fn a_shallow_hinge_is_smoothed() {
        let mut mesh = hinge(20.0);
        let before = normals_of(&mesh);
        let moved = soften_normals(&mut mesh, 45.0).expect("soften");
        let after = normals_of(&mesh);
        assert!(
            moved >= 4,
            "expected the hinge vertices to move, got {moved}"
        );
        let shared_left = Vec3::from(after[0]);
        let shared_right = Vec3::from(after[3]);
        assert!(
            shared_left.dot(shared_right) > 0.9999,
            "welded vertices should agree after smoothing: {shared_left:?} vs {shared_right:?}"
        );
        assert!(
            Vec3::from(before[0]).dot(shared_left) < 0.9999,
            "the normal should actually have changed"
        );
    }

    /// A sharp hinge remains unchanged beyond the crease angle.
    #[test]
    fn a_sharp_hinge_is_left_alone() {
        let mut mesh = hinge(100.0);
        let before = normals_of(&mesh);
        soften_normals(&mut mesh, 45.0).expect("soften");
        let after = normals_of(&mesh);
        for (b, a) in before.iter().zip(&after) {
            assert!(
                Vec3::from(*b).dot(Vec3::from(*a)) > 0.9999,
                "a 100 degree crease must survive a 45 degree threshold"
            );
        }
    }

    #[test]
    fn already_smooth_normals_are_a_no_op() {
        let mut mesh = hinge(20.0);
        soften_normals(&mut mesh, 45.0).expect("first");
        let moved = soften_normals(&mut mesh, 45.0).expect("second");
        assert_eq!(moved, 0, "smoothing is idempotent");
    }
}
