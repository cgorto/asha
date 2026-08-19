//! Surface shading math: the slim sun/ambient model, point-light
//! contributions, ramp and rim terms, and the light-field lookup.

#[cfg(target_arch = "spirv")]
use abi_core::GpuPtr;
use abi_core::gpu_data;
use abi_mesh::MeshShadeLighting;
use glam::Vec3;
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

/// Lights with nonpositive intensity or radius contribute nothing.
#[gpu_data]
pub struct PointLight {
    pub position: [f32; 3],
    pub radius: f32,
    pub color: [f32; 3],
    pub intensity: f32,
}

const _: () = assert!(core::mem::size_of::<PointLight>() == 32);
const _: () = assert!(core::mem::align_of::<PointLight>() == 4);
const _: () = assert!(core::mem::offset_of!(PointLight, position) == 0);
const _: () = assert!(core::mem::offset_of!(PointLight, radius) == 12);
const _: () = assert!(core::mem::offset_of!(PointLight, color) == 16);
const _: () = assert!(core::mem::offset_of!(PointLight, intensity) == 28);

pub fn mesh_shade_slim(
    normal_world: Vec3,
    base_color: [f32; 3],
    lighting: &MeshShadeLighting,
) -> [f32; 3] {
    let n = if normal_world.length_squared() > 1.0e-8 {
        normal_world.normalize()
    } else {
        Vec3::Y
    };
    let sun_direction = Vec3::from_array(lighting.sun_direction);
    let sun_tint = Vec3::from_array(lighting.sun_tint);
    let sky = Vec3::from_array(lighting.sky_ambient);
    let ground = Vec3::from_array(lighting.ground_ambient);
    let hemi_t = (n.y * 0.5 + 0.5).clamp(0.0, 1.0);
    let ambient = ground.lerp(sky, hemi_t);
    let sun = sun_tint * n.dot(sun_direction).max(0.0);
    let lit = Vec3::from_array(base_color) * (ambient + sun);
    lit.to_array()
}

/// Grid-space nearest-cell lookup. The floor operation is the host's
/// `world_to_cell` law: a point on x = cell_size belongs to cell 1, never 0.
fn light_field_index(dims: [u32; 2], cell_size: f32, position_world: Vec3) -> i32 {
    if dims[0] == 0
        || dims[1] == 0
        || dims[0] > i32::MAX as u32
        || dims[1] > i32::MAX as u32
        || cell_size <= 0.0
    {
        return -1;
    }
    let x = (position_world.x / cell_size).floor() as i32;
    let z = (position_world.z / cell_size).floor() as i32;
    if x < 0 || z < 0 || x >= dims[0] as i32 || z >= dims[1] as i32 {
        return -1;
    }
    if z > i32::MAX / dims[0] as i32 {
        return -1;
    }
    z * dims[0] as i32 + x
}

/// Sample a grid-domain light field on the CPU. `None` models the shader's
/// null pointer: zero initialization preserves the ungated path.
#[cfg(not(target_arch = "spirv"))]
pub fn light_field_sample(
    cells: Option<&[f32]>,
    dims: [u32; 2],
    cell_size: f32,
    position_world: Vec3,
) -> f32 {
    let Some(cells) = cells else {
        return 1.0;
    };
    let index = light_field_index(dims, cell_size, position_world);
    if index < 0 {
        return 0.0;
    }
    cells
        .get(index as usize)
        .copied()
        .unwrap_or(0.0)
        .clamp(0.0, 1.0)
}

/// Sample a grid-domain light field in shader code. A null pointer is the
/// zero-initialization contract and intentionally returns the neutral sample.
#[cfg(target_arch = "spirv")]
pub fn light_field_sample(
    cells: GpuPtr<f32>,
    dims: [u32; 2],
    cell_size: f32,
    position_world: Vec3,
) -> f32 {
    if cells.is_null() {
        return 1.0;
    }
    let index = light_field_index(dims, cell_size, position_world);
    if index < 0 {
        return 0.0;
    }
    cells[index as u32].clamp(0.0, 1.0)
}

/// The live A/B keeps zero as the neutral (ungated) path and one as the full
/// territory gate. Clamp defensively so malformed host data cannot amplify.
pub fn light_field_gate(sample: f32, gate: f32) -> f32 {
    1.0 + (sample.clamp(0.0, 1.0) - 1.0) * gate.clamp(0.0, 1.0)
}

/// Incident radiance from one point light at a world-space point, before any
/// surface cosine or volumetric phase term. Uses the exact same finite-radius
/// window and inverse-square-softened attenuation as surface lighting.
pub fn point_light_radiance(position_world: Vec3, light: &PointLight) -> Vec3 {
    if light.intensity <= 0.0 || light.radius <= 0.0 {
        return Vec3::ZERO;
    }
    let dist = (Vec3::from_array(light.position) - position_world)
        .length()
        .max(1.0e-4);
    let radius_ratio = dist / light.radius;
    let radius_ratio2 = radius_ratio * radius_ratio;
    let window = (1.0 - radius_ratio2 * radius_ratio2).clamp(0.0, 1.0);
    let atten = window * window / (dist * dist + 1.0);
    Vec3::from_array(light.color) * light.intensity * atten
}

/// Diffuse contribution of one point light, pre-albedo. `wrap_w = 0` is
/// exactly Lambert; the `1 / (1 + w)` factor keeps integrated energy fixed
/// as the wrap widens (the dial changes shape, not brightness).
pub fn point_light_contribution(
    normal: Vec3,
    position_world: Vec3,
    light: &PointLight,
    wrap_w: f32,
) -> Vec3 {
    // Reject zeroed lights and zero radius.
    if light.intensity <= 0.0 || light.radius <= 0.0 {
        return Vec3::ZERO;
    }
    let to_light = Vec3::from_array(light.position) - position_world;
    let dist = to_light.length().max(1.0e-4);
    let l = to_light / dist;
    let w = wrap_w.max(0.0);
    let wrap = ((normal.dot(l) + w) / (1.0 + w)).clamp(0.0, 1.0) / (1.0 + w);
    let radius_ratio = dist / light.radius;
    let radius_ratio2 = radius_ratio * radius_ratio;
    let window = (1.0 - radius_ratio2 * radius_ratio2).clamp(0.0, 1.0);
    let atten = window * window / (dist * dist + 1.0);
    Vec3::from_array(light.color) * light.intensity * wrap * atten
}

/// No-texture ramp path preserving identity-ramp arithmetic.
pub fn point_light_identity_ramp_contribution(
    normal: Vec3,
    position_world: Vec3,
    light: &PointLight,
    wrap_w: f32,
) -> Vec3 {
    point_light_contribution(normal, position_world, light, wrap_w)
}

/// Non-identity ramp input and light factors outside texture lookup.
pub fn point_light_ramp_terms(
    normal: Vec3,
    position_world: Vec3,
    light: &PointLight,
    wrap_w: f32,
    visibility: f32,
) -> (f32, Vec3) {
    if light.intensity <= 0.0 || light.radius <= 0.0 {
        return (0.0, Vec3::ZERO);
    }
    let to_light = Vec3::from_array(light.position) - position_world;
    let dist = to_light.length().max(1.0e-4);
    let l = to_light / dist;
    let w = wrap_w.max(0.0);
    let shape = ((normal.dot(l) + w) / (1.0 + w)).clamp(0.0, 1.0);
    let radius_ratio = dist / light.radius;
    let radius_ratio2 = radius_ratio * radius_ratio;
    let window = (1.0 - radius_ratio2 * radius_ratio2).clamp(0.0, 1.0);
    let atten = window * window / (dist * dist + 1.0);
    (
        shape * visibility,
        Vec3::from_array(light.color) * light.intensity * (1.0 / (1.0 + w)) * atten,
    )
}

/// Hemisphere ambient evaluated along an already-normalized surface normal.
pub fn mesh_ambient_along_normal(normal: Vec3, lighting: &MeshShadeLighting) -> Vec3 {
    let sky = Vec3::from_array(lighting.sky_ambient);
    let ground = Vec3::from_array(lighting.ground_ambient);
    let hemi_t = (normal.y * 0.5 + 0.5).clamp(0.0, 1.0);
    ground.lerp(sky, hemi_t)
}

/// Territory-gated additive ambient rim.
pub fn mesh_rim_contribution(
    normal: Vec3,
    position_world: Vec3,
    eye: Vec3,
    lighting: &MeshShadeLighting,
    visibility: f32,
    rim_power: f32,
    rim_boost: f32,
) -> Vec3 {
    if rim_boost <= 0.0 {
        return Vec3::ZERO;
    }
    let to_eye = eye - position_world;
    let view = if to_eye.length_squared() > 1.0e-8 {
        to_eye.normalize()
    } else {
        normal
    };
    let sheen = (1.0 - normal.dot(view).clamp(0.0, 1.0)).powf(rim_power.max(0.0));
    mesh_ambient_along_normal(normal, lighting) * sheen * visibility * rim_boost
}

/// CPU twin for the identity-ramp point-light sum.
#[cfg(not(target_arch = "spirv"))]
pub fn mesh_point_lights_identity(
    normal_world: Vec3,
    position_world: Vec3,
    diffuse: [f32; 3],
    wrap_w: f32,
    lights: &[PointLight],
    visibility: f32,
) -> [f32; 3] {
    let normal = if normal_world.length_squared() > 1.0e-8 {
        normal_world.normalize()
    } else {
        Vec3::Y
    };
    let albedo = Vec3::from_array(diffuse);
    let mut direct = Vec3::ZERO;
    for light in lights {
        direct += albedo
            * point_light_identity_ramp_contribution(normal, position_world, light, wrap_w)
            * visibility;
    }
    direct.to_array()
}

/// CPU twin for the identity-ramp shading endpoint.
#[cfg(not(target_arch = "spirv"))]
#[allow(clippy::too_many_arguments)] // Matches the fragment input seam.
pub fn mesh_shade_l3_identity(
    normal_world: Vec3,
    position_world: Vec3,
    eye: Vec3,
    base_color: [f32; 3],
    lighting: &MeshShadeLighting,
    lights: &[PointLight],
    visibility: f32,
    rim_power: f32,
    rim_boost: f32,
) -> [f32; 3] {
    let normal = if normal_world.length_squared() > 1.0e-8 {
        normal_world.normalize()
    } else {
        Vec3::Y
    };
    let mut rgb = Vec3::from_array(mesh_shade_slim(normal_world, base_color, lighting));
    rgb += Vec3::from_array(mesh_point_lights_identity(
        normal_world,
        position_world,
        base_color,
        lighting.wrap_w,
        lights,
        visibility,
    ));
    rgb += mesh_rim_contribution(
        normal,
        position_world,
        eye,
        lighting,
        visibility,
        rim_power,
        rim_boost,
    );
    rgb.to_array()
}

#[cfg(test)]
mod point_light_tests {
    use super::*;

    fn assert_vec3_close(actual: Vec3, expected: Vec3) {
        assert!(
            actual.abs_diff_eq(expected, 1.0e-6),
            "expected {expected:?}, got {actual:?}"
        );
    }

    #[test]
    fn point_light_zii_is_dead() {
        assert_eq!(
            point_light_contribution(Vec3::Z, Vec3::ZERO, &PointLight::default(), 0.0),
            Vec3::ZERO
        );
    }

    #[test]
    fn point_light_radiance_is_surface_law_before_cosine() {
        let light = PointLight {
            position: [0.0, 0.0, 4.0],
            radius: 8.0,
            color: [1.0, 0.5, 0.25],
            intensity: 2.0,
        };
        assert_eq!(
            point_light_radiance(Vec3::ZERO, &PointLight::default()),
            Vec3::ZERO
        );
        assert_vec3_close(
            point_light_radiance(Vec3::ZERO, &light),
            point_light_contribution(Vec3::Z, Vec3::ZERO, &light, 0.0),
        );
    }

    #[test]
    fn point_light_wrap_zero_is_lambert() {
        let light = PointLight {
            position: [0.0, 0.0, 4.0],
            radius: 8.0,
            color: [1.0, 0.5, 0.25],
            intensity: 2.0,
        };
        let window = 1.0 - 0.5f32.powi(4);
        let atten = window * window / 17.0;
        for normal in [
            Vec3::Z,
            Vec3::new(0.0, 3.0_f32.sqrt() * 0.5, 0.5),
            Vec3::NEG_Z,
        ] {
            let expected = Vec3::from_array(light.color)
                * light.intensity
                * normal.dot(Vec3::Z).max(0.0)
                * atten;
            assert_vec3_close(
                point_light_contribution(normal, Vec3::ZERO, &light, 0.0),
                expected,
            );
        }
    }

    #[test]
    fn l3_identity_ramp_is_exactly_l2() {
        let lights = [
            PointLight {
                position: [3.0, 2.0, -1.0],
                radius: 7.0,
                color: [1.0, 0.7, 0.3],
                intensity: 1.25,
            },
            PointLight {
                position: [-2.5, 5.0, 4.0],
                radius: 3.5,
                color: [0.2, 0.8, 1.0],
                intensity: 0.6,
            },
        ];
        let normals = [
            Vec3::X,
            Vec3::Y,
            Vec3::Z,
            Vec3::new(-0.3, 0.7, 0.2).normalize(),
        ];
        let positions = [
            Vec3::ZERO,
            Vec3::new(0.25, -0.5, 2.0),
            Vec3::new(-4.0, 1.0, 0.75),
        ];
        let wraps = [0.0, 0.125, 0.5, 1.75];
        let visibility = [0.0, 0.125, 0.5, 1.0];
        let lighting = MeshShadeLighting {
            sun_direction: [0.3, 0.8, 0.2],
            wrap_w: 0.0,
            sun_tint: [0.2, 0.3, 0.4],
            _pad1: 0,
            sky_ambient: [0.1, 0.2, 0.3],
            _pad2: 0.0,
            ground_ambient: [0.02, 0.04, 0.08],
            _pad3: 0.0,
        };
        let albedo = [0.6, 0.4, 0.8];

        for normal in normals {
            for position in positions {
                for wrap in wraps {
                    for vis in visibility {
                        let mut l2 = Vec3::from_array(mesh_shade_slim(normal, albedo, &lighting));
                        for light in &lights {
                            l2 += Vec3::from_array(albedo)
                                * point_light_contribution(normal, position, light, wrap)
                                * vis;
                        }
                        let eye = Vec3::new(2.0, 3.0, 4.0);
                        let l3 = Vec3::from_array(mesh_shade_l3_identity(
                            normal,
                            position,
                            eye,
                            albedo,
                            &MeshShadeLighting {
                                wrap_w: wrap,
                                ..lighting
                            },
                            &lights,
                            vis,
                            4.0,
                            0.0,
                        ));
                        assert_eq!(
                            l3, l2,
                            "normal={normal:?} position={position:?} wrap={wrap} vis={vis}"
                        );
                        assert_eq!(
                            mesh_rim_contribution(normal, position, eye, &lighting, vis, 4.0, 0.0,),
                            Vec3::ZERO,
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn light_field_absent_is_neutral() {
        assert_eq!(
            light_field_sample(None, [0, 0], 0.0, Vec3::new(-4.0, 0.0, -4.0)),
            1.0,
            "the CPU caller's absent field mirrors a null GPU pointer"
        );
    }

    #[test]
    fn light_field_out_of_bounds_is_dark() {
        let cells = [0.25, 0.5, 0.75, 1.0];
        for position in [
            Vec3::new(-0.001, 0.0, 0.0),
            Vec3::new(8.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, -0.001),
            Vec3::new(0.0, 0.0, 8.0),
        ] {
            assert_eq!(
                light_field_sample(Some(&cells), [2, 2], 4.0, position),
                0.0,
                "{position:?} must follow Grid::light_at's out-of-bounds law"
            );
        }
    }

    #[test]
    fn light_field_quantizes_world_boundaries_with_floor() {
        let cells = [0.25, 0.5, 0.75, 1.0];
        assert_eq!(
            light_field_sample(Some(&cells), [2, 2], 4.0, Vec3::new(3.999, 0.0, 0.0)),
            0.25
        );
        assert_eq!(
            light_field_sample(Some(&cells), [2, 2], 4.0, Vec3::new(4.0, 0.0, 0.0)),
            0.5,
            "x = CELL exactly belongs to the next cell"
        );
        assert_eq!(
            light_field_sample(Some(&cells), [2, 2], 4.0, Vec3::new(4.0, 0.0, 4.0)),
            1.0
        );
    }

    #[test]
    fn light_field_gate_lerps_toward_neutral() {
        assert_eq!(light_field_gate(0.25, 0.0), 1.0);
        assert_eq!(light_field_gate(0.25, 1.0), 0.25);
        assert_eq!(light_field_gate(0.25, 0.5), 0.625);
    }
}
