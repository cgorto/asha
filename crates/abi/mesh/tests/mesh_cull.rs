use abi_core::View;
use abi_core::glam::{Mat4, Vec3, Vec4};
use abi_mesh::{
    MeshBounds, Meshlet, aabb_in_frustum, cull_world_aabb, extract_frustum_planes, max_world_scale,
    mesh_world_to_clip, meshlet_backfacing_to_camera, sphere_inside_planes,
};

/// Builds the canonical forward-facing test camera.
fn base_view() -> View {
    let forward = Vec3::Z;
    let right = forward.cross(Vec3::Y).normalize();
    let up = right.cross(forward);
    View {
        camera_position: [0.0, 0.0, 0.0],
        tan_half_fov: 0.55,
        camera_forward: forward.to_array(),
        aspect: 16.0 / 9.0,
        camera_right: right.to_array(),
        depth_near_plane: 0.1,
        camera_up: up.to_array(),
        _pad: 0,
        output_size: [1280, 720],
        _pad2: [0; 2],
    }
}

/// Builds an offset, rotated test camera.
fn offset_view() -> View {
    let forward = Vec3::new(0.3, -0.2, 1.0).normalize();
    let right = forward.cross(Vec3::Y).normalize();
    let up = right.cross(forward);
    View {
        camera_position: [12.0, 8.0, -30.0],
        camera_forward: forward.to_array(),
        camera_right: right.to_array(),
        camera_up: up.to_array(),
        ..base_view()
    }
}

fn plane_distance(plane: &[f32; 4], p: Vec3) -> f32 {
    Vec3::new(plane[0], plane[1], plane[2]).dot(p) + plane[3]
}

fn point_in_planes(planes: &[[f32; 4]; 6], p: Vec3) -> bool {
    planes.iter().all(|plane| plane_distance(plane, p) >= 0.0)
}

/// Verifies extracted planes against direct clip-volume classification.
#[test]
fn frustum_planes_match_clip_volume() {
    for view in [base_view(), offset_view()] {
        let world_to_clip = mesh_world_to_clip(&view);
        let planes = extract_frustum_planes(&world_to_clip);

        let mut checked = 0u32;
        for ix in 0..11 {
            for iy in 0..11 {
                for iz in 0..11 {
                    let p = Vec3::new(
                        -50.0 + 10.0 * ix as f32,
                        -50.0 + 10.0 * iy as f32,
                        -50.0 + 10.0 * iz as f32,
                    );
                    // Skip razor-edge points: sign agreement there is a
                    // float-rounding coin toss, not a correctness signal.
                    if planes
                        .iter()
                        .any(|plane| plane_distance(plane, p).abs() <= 1.0e-3)
                    {
                        continue;
                    }
                    let clip = world_to_clip * p.extend(1.0);
                    let contained = clip.w > 0.0
                        && clip.x.abs() <= clip.w
                        && clip.y.abs() <= clip.w
                        && 0.0 <= clip.z
                        && clip.z <= clip.w;
                    assert_eq!(
                        point_in_planes(&planes, p),
                        contained,
                        "point {p} disagrees (clip {clip})"
                    );
                    checked += 1;
                }
            }
        }
        assert!(checked > 1000, "lattice degenerated to {checked} points");

        let camera = Vec3::from_array(view.camera_position);
        let forward = Vec3::from_array(view.camera_forward);
        assert!(
            !point_in_planes(&planes, camera - forward * 10.0),
            "behind the camera must be outside"
        );
        assert!(
            point_in_planes(&planes, camera + forward * 10.0),
            "the view target must be inside"
        );
    }
}

#[test]
fn aabb_in_frustum_classifies_boxes() {
    // Finite Vulkan projection exercises all plane rows.
    let proj = Mat4::perspective_rh(1.0, 16.0 / 9.0, 0.1, 100.0);
    let planes = extract_frustum_planes(&proj);

    // A box straddling the left plane remains visible.
    assert!(aabb_in_frustum(
        Vec3::new(-1000.0, -1.0, -11.0),
        Vec3::new(1.0, 1.0, -9.0),
        &planes,
    ));
    // Fully beyond the far plane: not visible.
    assert!(!aabb_in_frustum(
        Vec3::new(-1.0, -1.0, -210.0),
        Vec3::new(1.0, 1.0, -200.0),
        &planes,
    ));
    // A giant AABB containing the whole frustum: visible.
    assert!(aabb_in_frustum(
        Vec3::splat(-1000.0),
        Vec3::splat(1000.0),
        &planes,
    ));

    // The engine's infinite projection agrees.
    let planes = extract_frustum_planes(&mesh_world_to_clip(&base_view()));
    assert!(aabb_in_frustum(
        Vec3::new(-1000.0, -1.0, 9.0),
        Vec3::new(1.0, 1.0, 11.0),
        &planes,
    ));
    assert!(!aabb_in_frustum(
        Vec3::new(-1.0, -1.0, -20.0),
        Vec3::new(1.0, 1.0, -10.0),
        &planes,
    ));
}

#[test]
fn sphere_inside_planes_classifies_spheres() {
    // Finite Vulkan projection supplies six real planes.
    let proj = Mat4::perspective_rh(1.0, 16.0 / 9.0, 0.1, 100.0);
    let planes = extract_frustum_planes(&proj);

    // Fully inside with a small radius.
    assert!(sphere_inside_planes(
        Vec3::new(0.0, 0.0, -10.0),
        1.0,
        &planes
    ));
    // A sphere crossing the left plane remains visible.
    assert!(sphere_inside_planes(
        Vec3::new(-100.0, 0.0, -10.0),
        200.0,
        &planes
    ));
    // Test complete exclusion by every plane family.
    assert!(!sphere_inside_planes(
        Vec3::new(-1000.0, 0.0, -10.0),
        1.0,
        &planes
    ));
    assert!(!sphere_inside_planes(
        Vec3::new(1000.0, 0.0, -10.0),
        1.0,
        &planes
    ));
    assert!(!sphere_inside_planes(
        Vec3::new(0.0, -1000.0, -10.0),
        1.0,
        &planes
    ));
    assert!(!sphere_inside_planes(
        Vec3::new(0.0, 1000.0, -10.0),
        1.0,
        &planes
    ));
    assert!(!sphere_inside_planes(
        Vec3::new(0.0, 0.0, 5.0),
        1.0,
        &planes
    ));
    assert!(!sphere_inside_planes(
        Vec3::new(0.0, 0.0, -500.0),
        1.0,
        &planes
    ));

    // The engine's infinite projection agrees.
    let planes = extract_frustum_planes(&mesh_world_to_clip(&base_view()));
    assert!(sphere_inside_planes(
        Vec3::new(0.0, 0.0, 10.0),
        1.0,
        &planes
    ));
    assert!(!sphere_inside_planes(
        Vec3::new(0.0, 0.0, -10.0),
        1.0,
        &planes
    ));
}

#[test]
fn max_world_scale_bounds_trs_and_shear() {
    assert_eq!(max_world_scale(&Mat4::IDENTITY), 1.0);
    assert!((max_world_scale(&Mat4::from_scale(Vec3::splat(2.5))) - 2.5).abs() < 1.0e-6);
    // Orthogonal nonuniform transforms preserve tight bounds.
    let m = Mat4::from_rotation_y(0.7) * Mat4::from_scale(Vec3::new(0.5, 3.0, 1.25));
    assert!((max_world_scale(&m) - 3.0).abs() < 1.0e-5);

    // Gershgorin bounds cover inherited shear conservatively.
    let shear = Mat4::from_cols(
        Vec4::new(1.0, 0.0, 0.0, 0.0),
        Vec4::new(1.0, 1.0, 0.0, 0.0),
        Vec4::Z,
        Vec4::W,
    );
    let most_stretched = Vec3::new(0.525_731_1, 0.850_650_8, 0.0);
    let actual = shear.transform_vector3(most_stretched).length();
    let bound = max_world_scale(&shear);
    assert!(actual > 2.0f32.sqrt());
    assert!(
        bound >= actual,
        "shear stretch {actual} exceeds bound {bound}"
    );
    assert!((bound - 3.0f32.sqrt()).abs() < 1.0e-6);
}

#[test]
fn meshlet_backface_cone_sense() {
    const EPSILON: f32 = 1.0e-4;
    let camera = Vec3::ZERO;
    let center = Vec3::new(0.0, 0.0, 10.0); // view_to_meshlet = +Z
    let meshlet = |axis: [f32; 3], cutoff: f32| Meshlet {
        cone_axis: axis,
        cone_cutoff: cutoff,
        ..Meshlet::default()
    };

    // Cone facing the camera (axis back along -Z): NOT culled.
    assert!(!meshlet_backfacing_to_camera(
        &meshlet([0.0, 0.0, -1.0], 0.5),
        &Mat4::IDENTITY,
        camera,
        center,
        EPSILON,
    ));
    // Cone facing away (axis along +Z, with the view direction): culled.
    assert!(meshlet_backfacing_to_camera(
        &meshlet([0.0, 0.0, 1.0], 0.5),
        &Mat4::IDENTITY,
        camera,
        center,
        EPSILON,
    ));
    // A 180-degree yaw in the normal matrix flips the verdict — the world
    // transform is really applied to the axis.
    assert!(!meshlet_backfacing_to_camera(
        &meshlet([0.0, 0.0, 1.0], 0.5),
        &Mat4::from_rotation_y(std::f32::consts::PI),
        camera,
        center,
        EPSILON,
    ));
    // The epsilon tie-break: `dot >= cutoff + epsilon` CULLS.
    // Perpendicular axis gives dot = 0; cutoff = -epsilon puts the
    // threshold exactly at 0 (culled), cutoff = 0 puts it above (kept).
    assert!(meshlet_backfacing_to_camera(
        &meshlet([1.0, 0.0, 0.0], -EPSILON),
        &Mat4::IDENTITY,
        camera,
        center,
        EPSILON,
    ));
    assert!(!meshlet_backfacing_to_camera(
        &meshlet([1.0, 0.0, 0.0], 0.0),
        &Mat4::IDENTITY,
        camera,
        center,
        EPSILON,
    ));
    // Degenerate axis (length < 1e-6) never culls, even with cutoff -1.
    assert!(!meshlet_backfacing_to_camera(
        &meshlet([0.0, 0.0, 0.0], -1.0),
        &Mat4::IDENTITY,
        camera,
        center,
        EPSILON,
    ));
}

#[test]
fn cull_world_aabb_identity_and_rotation() {
    let bounds = MeshBounds {
        aabb_min: [-1.0, -2.0, -3.0],
        aabb_max: [4.0, 5.0, 6.0],
        ..MeshBounds::default()
    };
    let (world_min, world_max) = cull_world_aabb(&bounds, &Mat4::IDENTITY, 0.0);
    assert_eq!(world_min.to_array(), bounds.aabb_min);
    assert_eq!(world_max.to_array(), bounds.aabb_max);

    // A unit-half-extent cube rotated 45 degrees around Y spreads to sqrt(2)
    // in x and z; y is untouched.
    let cube = MeshBounds {
        aabb_min: [-1.0, -1.0, -1.0],
        aabb_max: [1.0, 1.0, 1.0],
        ..MeshBounds::default()
    };
    let rot = Mat4::from_rotation_y(45.0f32.to_radians());
    let (world_min, world_max) = cull_world_aabb(&cube, &rot, 0.0);
    let r = 2.0f32.sqrt();
    let expect_min = [-r, -1.0, -r];
    let expect_max = [r, 1.0, r];
    for i in 0..3 {
        assert!(
            (world_min.to_array()[i] - expect_min[i]).abs() < 1.0e-6,
            "min axis {i}: {} != {}",
            world_min.to_array()[i],
            expect_min[i]
        );
        assert!(
            (world_max.to_array()[i] - expect_max[i]).abs() < 1.0e-6,
            "max axis {i}: {} != {}",
            world_max.to_array()[i],
            expect_max[i]
        );
    }

    let (world_min, world_max) = cull_world_aabb(&cube, &Mat4::IDENTITY, 2.5);
    assert_eq!(world_min.to_array(), [-3.5, -3.5, -3.5]);
    assert_eq!(world_max.to_array(), [3.5, 3.5, 3.5]);
}

#[test]
fn dilated_aabb_can_survive_frustum() {
    let planes = extract_frustum_planes(&mesh_world_to_clip(&base_view()));
    let bounds = MeshBounds {
        aabb_min: [-1.0, -1.0, -1.0],
        aabb_max: [1.0, 1.0, 1.0],
        ..MeshBounds::default()
    };
    let world = Mat4::from_translation(Vec3::new(0.0, 0.0, -2.0));
    let (rest_min, rest_max) = cull_world_aabb(&bounds, &world, 0.0);
    assert!(!aabb_in_frustum(rest_min, rest_max, &planes));

    let (dilated_min, dilated_max) = cull_world_aabb(&bounds, &world, 3.5);
    assert!(aabb_in_frustum(dilated_min, dilated_max, &planes));
}
