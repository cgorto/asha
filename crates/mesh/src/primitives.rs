//! Procedural mesh buffers for cold-path scene setup.

use std::collections::HashMap;
use std::f32::consts::PI;

use crate::MeshDesc;

#[derive(Debug, Clone, Default)]
pub struct MeshBuffers {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub uvs: Vec<[f32; 2]>,
    pub indices: Vec<u32>,
    pub tangents: Option<Vec<[f32; 4]>>,
    /// Optional vertex-parallel RGBA tint; `None` is the untinted default.
    pub colors: Option<Vec<[f32; 4]>>,
}

impl MeshBuffers {
    pub fn desc(&self) -> MeshDesc<'_> {
        MeshDesc {
            positions: &self.positions,
            normals: &self.normals,
            uvs: &self.uvs,
            indices: &self.indices,
            tangents: self.tangents.as_deref(),
            joint_weights: None,
            colors: self.colors.as_deref(),
        }
    }
}

impl<'a> From<&'a MeshBuffers> for MeshDesc<'a> {
    fn from(value: &'a MeshBuffers) -> Self {
        value.desc()
    }
}

pub fn cube(half_extent: f32) -> MeshBuffers {
    assert!(
        half_extent.is_finite() && half_extent > 0.0,
        "cube half_extent must be finite and positive"
    );

    let h = half_extent;
    let mut mesh = MeshBuffers::default();
    push_face(
        &mut mesh,
        [0.0, 0.0, 1.0],
        [[-h, -h, h], [h, -h, h], [h, h, h], [-h, h, h]],
    );
    push_face(
        &mut mesh,
        [0.0, 0.0, -1.0],
        [[h, -h, -h], [-h, -h, -h], [-h, h, -h], [h, h, -h]],
    );
    push_face(
        &mut mesh,
        [1.0, 0.0, 0.0],
        [[h, -h, h], [h, -h, -h], [h, h, -h], [h, h, h]],
    );
    push_face(
        &mut mesh,
        [-1.0, 0.0, 0.0],
        [[-h, -h, -h], [-h, -h, h], [-h, h, h], [-h, h, -h]],
    );
    push_face(
        &mut mesh,
        [0.0, 1.0, 0.0],
        [[-h, h, h], [h, h, h], [h, h, -h], [-h, h, -h]],
    );
    push_face(
        &mut mesh,
        [0.0, -1.0, 0.0],
        [[-h, -h, -h], [h, -h, -h], [h, -h, h], [-h, -h, h]],
    );
    mesh
}

/// Creates an XY-plane quad with +Z normals and [0, 1]² UVs.
pub fn quad(half_extent: f32) -> MeshBuffers {
    assert!(
        half_extent.is_finite() && half_extent > 0.0,
        "quad half_extent must be finite and positive"
    );

    let h = half_extent;
    let mut mesh = MeshBuffers::default();
    push_face(
        &mut mesh,
        [0.0, 0.0, 1.0],
        [[-h, -h, 0.0], [h, -h, 0.0], [h, h, 0.0], [-h, h, 0.0]],
    );
    mesh
}

pub fn icosphere(radius: f32, subdivisions: u32) -> MeshBuffers {
    assert!(
        radius.is_finite() && radius > 0.0,
        "icosphere radius must be finite and positive"
    );
    assert!(subdivisions <= 4, "icosphere subdivisions must be <= 4");

    let t = (1.0 + 5.0_f32.sqrt()) * 0.5;
    let mut positions = vec![
        normalize_radius([-1.0, t, 0.0], radius),
        normalize_radius([1.0, t, 0.0], radius),
        normalize_radius([-1.0, -t, 0.0], radius),
        normalize_radius([1.0, -t, 0.0], radius),
        normalize_radius([0.0, -1.0, t], radius),
        normalize_radius([0.0, 1.0, t], radius),
        normalize_radius([0.0, -1.0, -t], radius),
        normalize_radius([0.0, 1.0, -t], radius),
        normalize_radius([t, 0.0, -1.0], radius),
        normalize_radius([t, 0.0, 1.0], radius),
        normalize_radius([-t, 0.0, -1.0], radius),
        normalize_radius([-t, 0.0, 1.0], radius),
    ];
    let mut faces = vec![
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ];

    for _ in 0..subdivisions {
        let mut midpoint_cache = HashMap::<(u32, u32), u32>::new();
        let mut next_faces = Vec::with_capacity(faces.len() * 4);
        for [a, b, c] in faces {
            let ab = midpoint(&mut positions, &mut midpoint_cache, a, b, radius);
            let bc = midpoint(&mut positions, &mut midpoint_cache, b, c, radius);
            let ca = midpoint(&mut positions, &mut midpoint_cache, c, a, radius);
            next_faces.push([a, ab, ca]);
            next_faces.push([b, bc, ab]);
            next_faces.push([c, ca, bc]);
            next_faces.push([ab, bc, ca]);
        }
        faces = next_faces;
    }

    let normals = positions
        .iter()
        .map(|&p| normalize_radius(p, 1.0))
        .collect::<Vec<_>>();
    let uvs = normals.iter().map(|&n| spherical_uv(n)).collect::<Vec<_>>();
    let mut indices = Vec::with_capacity(faces.len() * 3);
    for [a, b, c] in faces {
        indices.extend_from_slice(&[a, b, c]);
    }

    MeshBuffers {
        positions,
        normals,
        uvs,
        indices,
        tangents: None,
        colors: None,
    }
}

/// Creates a Y-up capsule with hemispherical caps and a cylindrical wall.
/// `rings` controls each hemisphere; `segments` controls the circumference.
/// Equator rings are shared and pole bands use triangle fans.
pub fn capsule(radius: f32, half_height: f32, segments: u32, rings: u32) -> MeshBuffers {
    assert!(
        radius.is_finite() && radius > 0.0,
        "capsule radius must be finite and positive"
    );
    assert!(
        half_height.is_finite() && half_height >= 0.0,
        "capsule half_height must be finite and non-negative"
    );
    assert!(segments >= 3, "capsule needs at least three segments");
    assert!(rings >= 1, "capsule needs at least one ring per hemisphere");

    let mut mesh = MeshBuffers::default();
    let total_arc = PI * radius + 2.0 * half_height;
    let row_count = 2 * rings + 2;

    // Rows run from the top pole through the wall to the bottom pole.
    for j in 0..row_count {
        let (theta, center_y, arc) = if j <= rings {
            let theta = (PI * 0.5) * j as f32 / rings as f32;
            (theta, half_height, radius * theta)
        } else {
            let step = (j - rings - 1) as f32 / rings as f32;
            let theta = PI * 0.5 * (1.0 + step);
            (
                theta,
                -half_height,
                radius * PI * 0.5 + 2.0 * half_height + radius * (theta - PI * 0.5),
            )
        };
        let (ring_radius, y_dir) = (theta.sin(), theta.cos());
        for i in 0..=segments {
            let a = 2.0 * PI * i as f32 / segments as f32;
            let normal = [ring_radius * a.cos(), y_dir, ring_radius * a.sin()];
            mesh.positions.push([
                normal[0] * radius,
                center_y + normal[1] * radius,
                normal[2] * radius,
            ]);
            mesh.normals.push(normal);
            mesh.uvs.push([i as f32 / segments as f32, arc / total_arc]);
        }
    }

    let stride = segments + 1;
    for j in 0..row_count as u32 - 1 {
        for i in 0..segments {
            let v = j * stride + i;
            // Pole rows emit triangle fans instead of degenerate quads.
            if j != 0 {
                mesh.indices.extend_from_slice(&[v, v + 1, v + stride + 1]);
            }
            if j != row_count as u32 - 2 {
                mesh.indices
                    .extend_from_slice(&[v, v + stride + 1, v + stride]);
            }
        }
    }
    mesh
}

fn push_face(mesh: &mut MeshBuffers, normal: [f32; 3], corners: [[f32; 3]; 4]) {
    let base = mesh.positions.len() as u32;
    mesh.positions.extend_from_slice(&corners);
    mesh.normals.extend_from_slice(&[normal; 4]);
    mesh.uvs
        .extend_from_slice(&[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]);
    // Model-space CCW remains front-facing after projection parity mirrors.
    mesh.indices
        .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
}

fn midpoint(
    positions: &mut Vec<[f32; 3]>,
    cache: &mut HashMap<(u32, u32), u32>,
    a: u32,
    b: u32,
    radius: f32,
) -> u32 {
    let key = if a < b { (a, b) } else { (b, a) };
    if let Some(&index) = cache.get(&key) {
        return index;
    }

    let pa = positions[a as usize];
    let pb = positions[b as usize];
    let p = normalize_radius(
        [
            (pa[0] + pb[0]) * 0.5,
            (pa[1] + pb[1]) * 0.5,
            (pa[2] + pb[2]) * 0.5,
        ],
        radius,
    );
    let index = positions.len() as u32;
    positions.push(p);
    cache.insert(key, index);
    index
}

fn normalize_radius(p: [f32; 3], radius: f32) -> [f32; 3] {
    let len = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
    assert!(
        len.is_finite() && len > 0.0,
        "sphere position must be non-zero"
    );
    [
        p[0] / len * radius,
        p[1] / len * radius,
        p[2] / len * radius,
    ]
}

/// Creates a textured sphere from six independently parameterized cube faces.
/// Face-local UVs avoid poles and wrap seams, but jump at face boundaries.
/// `segments` is each face's grid resolution: `6 * segments²` quads.
pub fn cube_sphere(radius: f32, segments: u32) -> MeshBuffers {
    assert!(
        radius.is_finite() && radius > 0.0,
        "cube_sphere radius must be finite and positive"
    );
    assert!(segments >= 1, "cube_sphere needs at least one segment");

    // Right-handed face bases preserve outward CCW winding.
    const FACES: [([f32; 3], [f32; 3], [f32; 3]); 6] = [
        ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]),
        ([-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]),
        ([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]),
        ([0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]),
        ([0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
        ([0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]),
    ];

    let mut mesh = MeshBuffers::default();
    let n = segments as usize;
    for (forward, right, up) in FACES {
        let base = mesh.positions.len() as u32;
        for j in 0..=n {
            for i in 0..=n {
                let (s, t) = (i as f32 / n as f32, j as f32 / n as f32);
                // Tangent spacing produces near-uniform angular cells.
                let (a, b) = (angular(s), angular(t));
                let dir = normalize_radius(
                    [
                        forward[0] + right[0] * a + up[0] * b,
                        forward[1] + right[1] * a + up[1] * b,
                        forward[2] + right[2] * a + up[2] * b,
                    ],
                    1.0,
                );
                mesh.positions
                    .push([dir[0] * radius, dir[1] * radius, dir[2] * radius]);
                mesh.normals.push(dir);
                mesh.uvs.push([s, t]);
            }
        }
        let stride = (n + 1) as u32;
        for j in 0..n as u32 {
            for i in 0..n as u32 {
                let v = base + j * stride + i;
                mesh.indices.extend_from_slice(&[
                    v,
                    v + 1,
                    v + stride + 1,
                    v,
                    v + stride + 1,
                    v + stride,
                ]);
            }
        }
    }
    mesh
}

/// Maps a grid parameter to an angle-spaced cube-face coordinate.
fn angular(s: f32) -> f32 {
    ((s - 0.5) * (PI * 0.5)).tan()
}

fn spherical_uv(n: [f32; 3]) -> [f32; 2] {
    let u = 0.5 + n[2].atan2(n[0]) / (2.0 * PI);
    let v = 0.5 - n[1].asin() / PI;
    [u, v]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cube_counts_and_aabb() {
        let mesh = cube(2.0);
        assert_eq!(mesh.positions.len(), 24);
        assert_eq!(mesh.normals.len(), 24);
        assert_eq!(mesh.uvs.len(), 24);
        assert_eq!(mesh.indices.len(), 36);
        for axis in 0..3 {
            let min = mesh
                .positions
                .iter()
                .map(|p| p[axis])
                .fold(f32::INFINITY, f32::min);
            let max = mesh
                .positions
                .iter()
                .map(|p| p[axis])
                .fold(f32::NEG_INFINITY, f32::max);
            assert_eq!(min, -2.0);
            assert_eq!(max, 2.0);
        }
    }

    #[test]
    fn quad_is_mesh_path_geometry() {
        let mesh = quad(2.0);
        assert_eq!(
            mesh.positions.as_slice(),
            &[
                [-2.0, -2.0, 0.0],
                [2.0, -2.0, 0.0],
                [2.0, 2.0, 0.0],
                [-2.0, 2.0, 0.0],
            ]
        );
        assert_eq!(mesh.normals.as_slice(), &[[0.0, 0.0, 1.0]; 4]);
        assert_eq!(
            mesh.uvs.as_slice(),
            &[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
        );
        assert_eq!(mesh.indices.as_slice(), &[0, 1, 2, 0, 2, 3]);
        assert_vulkan_winding(&mesh);
    }

    #[test]
    fn icosphere_counts() {
        for subdivisions in 0..=4 {
            let mesh = icosphere(1.0, subdivisions);
            let tri_count = 20usize * 4usize.pow(subdivisions);
            let vertex_count = 10usize * 4usize.pow(subdivisions) + 2;
            assert_eq!(mesh.positions.len(), vertex_count);
            assert_eq!(mesh.normals.len(), vertex_count);
            assert_eq!(mesh.uvs.len(), vertex_count);
            assert_eq!(mesh.indices.len(), tri_count * 3);
        }
    }

    #[test]
    fn normals_and_sphere_radius() {
        let mesh = icosphere(3.0, 3);
        for (position, normal) in mesh.positions.iter().zip(&mesh.normals) {
            assert!((len(*position) - 3.0).abs() < 1.0e-5);
            assert!((len(*normal) - 1.0).abs() < 1.0e-5);
        }

        let cube = cube(1.0);
        for normal in cube.normals {
            assert!((len(normal) - 1.0).abs() < 1.0e-6);
        }
    }

    #[test]
    fn capsule_counts_and_surface() {
        let (radius, half_height) = (0.5, 0.6);
        for (segments, rings) in [(3u32, 1u32), (12, 4), (24, 8)] {
            let mesh = capsule(radius, half_height, segments, rings);
            let rows = 2 * rings as usize + 2;
            assert_eq!(mesh.positions.len(), rows * (segments as usize + 1));
            assert_eq!(mesh.normals.len(), mesh.positions.len());
            assert_eq!(mesh.uvs.len(), mesh.positions.len());
            // Two pole fans plus cylindrical quad bands.
            assert_eq!(mesh.indices.len(), 4 * (rings * segments) as usize * 3);
            for (position, normal) in mesh.positions.iter().zip(&mesh.normals) {
                assert!((len(*normal) - 1.0).abs() < 1.0e-5);
                // Verify distance from the capsule's central axis segment.
                let clamped_y = position[1].clamp(-half_height, half_height);
                let offset = [position[0], position[1] - clamped_y, position[2]];
                assert!((len(offset) - radius).abs() < 1.0e-5);
            }
            let top = mesh.positions.iter().map(|p| p[1]).fold(f32::MIN, f32::max);
            let bottom = mesh.positions.iter().map(|p| p[1]).fold(f32::MAX, f32::min);
            assert!((top - (half_height + radius)).abs() < 1.0e-6);
            assert!((bottom + (half_height + radius)).abs() < 1.0e-6);
        }
    }

    #[test]
    fn vulkan_winding_is_consistent() {
        let cube = cube(1.0);
        assert_vulkan_winding(&cube);
        let sphere = icosphere(1.0, 2);
        assert_vulkan_winding(&sphere);
        assert_vulkan_winding(&capsule(0.5, 0.6, 12, 4));
        // Verify each face basis preserves outward winding.
        assert_vulkan_winding(&cube_sphere(1.0, 3));
    }

    #[test]
    fn cube_sphere_counts_and_radius() {
        for segments in [1u32, 2, 8] {
            let mesh = cube_sphere(2.0, segments);
            let n = segments as usize;
            assert_eq!(mesh.positions.len(), 6 * (n + 1) * (n + 1));
            assert_eq!(mesh.normals.len(), mesh.positions.len());
            assert_eq!(mesh.uvs.len(), mesh.positions.len());
            assert_eq!(mesh.indices.len(), 6 * n * n * 6);
            for position in &mesh.positions {
                assert!((len(*position) - 2.0).abs() < 1.0e-5);
            }
        }
    }

    /// Verifies that tangent spacing keeps cube-sphere cells near-uniform.
    #[test]
    fn cube_sphere_cells_are_near_uniform() {
        let mesh = cube_sphere(1.0, 8);
        let (mut min, mut max) = (f32::INFINITY, 0.0f32);
        for triangle in mesh.indices.chunks_exact(3) {
            let [a, b, c] = [0, 1, 2].map(|i| mesh.positions[triangle[i] as usize]);
            // Triangle area approximates solid angle on the unit sphere.
            let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
            let area = len(cross(u, v));
            min = min.min(area);
            max = max.max(area);
        }
        assert!(
            max / min < 1.6,
            "cube sphere cell area spread {:.2}x is too wide",
            max / min
        );
    }

    fn assert_vulkan_winding(mesh: &MeshBuffers) {
        for triangle in mesh.indices.chunks_exact(3) {
            let a = mesh.positions[triangle[0] as usize];
            let b = mesh.positions[triangle[1] as usize];
            let c = mesh.positions[triangle[2] as usize];
            let na = mesh.normals[triangle[0] as usize];
            let nb = mesh.normals[triangle[1] as usize];
            let nc = mesh.normals[triangle[2] as usize];
            let outward = [
                na[0] + nb[0] + nc[0],
                na[1] + nb[1] + nc[1],
                na[2] + nb[2] + nc[2],
            ];
            // Geometric and declared normals must agree.
            assert!(
                dot(cross(sub(b, a), sub(c, a)), outward) > 0.0,
                "triangle winding is not CCW-from-outside"
            );
        }
    }

    fn len(v: [f32; 3]) -> f32 {
        dot(v, v).sqrt()
    }

    fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
        [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
    }

    fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    }

    fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    }
}
