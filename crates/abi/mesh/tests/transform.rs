use abi_core::glam::{Mat4, Quat, Vec3};
use abi_mesh::{LocalTransform, Trs, compose, normal_matrix, world_transform};

const EPS: f32 = 1.0e-5;

fn assert_vec3_near(a: Vec3, b: Vec3) {
    let delta = (a - b).abs();
    assert!(
        delta.max_element() < EPS,
        "left={a:?} right={b:?} delta={delta:?}"
    );
}

fn assert_mat4_near(a: Mat4, b: Mat4) {
    let a_cols = a.to_cols_array();
    let b_cols = b.to_cols_array();
    for (i, (a, b)) in a_cols.into_iter().zip(b_cols).enumerate() {
        assert!((a - b).abs() < EPS, "col-array[{i}] left={a} right={b}");
    }
}

#[test]
fn matrix_form_local_matches_equivalent_trs() {
    let parent_trs = Trs {
        translation: Vec3::new(3.0, -2.0, 5.0),
        rotation: Quat::from_rotation_y(0.37),
        scale: Vec3::new(1.25, 2.0, 0.75),
    };
    let parent = parent_trs.to_matrix();
    let local_trs = Trs {
        translation: Vec3::new(-1.0, 0.5, 2.0),
        rotation: Quat::from_rotation_x(-0.9),
        scale: Vec3::new(0.5, 1.5, 2.25),
    };
    let local_matrix = local_trs.to_matrix();

    let from_trs = compose(parent, &LocalTransform::Trs(local_trs));
    let from_matrix = compose(parent, &LocalTransform::Matrix(local_matrix));
    assert_mat4_near(from_trs, from_matrix);

    let point = Vec3::new(0.3, -1.2, 4.0);
    let expected = parent.transform_point3(local_matrix.transform_point3(point));
    assert_vec3_near(from_trs.transform_point3(point), expected);
    assert_vec3_near(from_matrix.transform_point3(point), expected);
}

#[test]
fn three_deep_chain_keeps_inherited_shear() {
    let root = LocalTransform::Trs(Trs {
        translation: Vec3::new(2.0, -1.0, 0.5),
        rotation: Quat::from_rotation_z(0.2),
        scale: Vec3::new(2.0, 0.5, 1.5),
    });
    let child = LocalTransform::Trs(Trs {
        translation: Vec3::new(-0.25, 3.0, 1.0),
        rotation: Quat::from_rotation_y(1.1),
        scale: Vec3::ONE,
    });
    let grandchild = LocalTransform::Trs(Trs {
        translation: Vec3::new(0.7, -0.4, 2.0),
        rotation: Quat::from_rotation_x(-0.45),
        scale: Vec3::new(1.0, 0.75, 1.25),
    });

    let root_world = compose(Mat4::IDENTITY, &root);
    let child_world = compose(root_world, &child);
    let grandchild_world = compose(child_world, &grandchild);

    let point = Vec3::new(1.2, -0.8, 0.35);
    let step = root.matrix().transform_point3(
        child
            .matrix()
            .transform_point3(grandchild.matrix().transform_point3(point)),
    );
    assert_vec3_near(grandchild_world.transform_point3(point), step);
}

#[test]
fn normal_matrix_keeps_nonuniform_surface_perpendicular() {
    let world = Mat4::from_translation(Vec3::new(10.0, -4.0, 2.0))
        * Mat4::from_rotation_z(0.8)
        * Mat4::from_rotation_y(-0.3)
        * Mat4::from_scale(Vec3::new(2.0, 0.5, 3.0));
    let tangent = Vec3::X;
    let normal = Vec3::Y;

    let tangent_world = world.transform_vector3(tangent).normalize();
    let normal_world = normal_matrix(world).transform_vector3(normal).normalize();
    assert!(
        tangent_world.dot(normal_world).abs() < EPS,
        "tangent={tangent_world:?} normal={normal_world:?}"
    );
}

#[test]
fn normal_matrix_uniform_scale_matches_rotation_after_normalize() {
    let rotation = Quat::from_rotation_x(0.4) * Quat::from_rotation_z(-0.8);
    let rotation_matrix = Mat4::from_quat(rotation);
    let world = Mat4::from_translation(Vec3::new(-3.0, 8.0, 1.0))
        * rotation_matrix
        * Mat4::from_scale(Vec3::splat(4.0));
    let normal = normal_matrix(world);

    for vector in [
        Vec3::X,
        Vec3::Y,
        Vec3::Z,
        Vec3::new(1.0, -2.0, 0.5).normalize(),
    ] {
        let got = normal.transform_vector3(vector).normalize();
        let expected = rotation_matrix.transform_vector3(vector).normalize();
        assert_vec3_near(got, expected);
    }
}

#[test]
#[should_panic]
fn zero_scale_world_panics() {
    let world = Mat4::from_scale(Vec3::new(1.0, 0.0, 1.0));
    let _ = world_transform(world);
}

#[test]
#[should_panic]
fn mirrored_world_panics() {
    let world = Mat4::from_scale(Vec3::new(-1.0, 1.0, 1.0));
    let _ = normal_matrix(world);
}
