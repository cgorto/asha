use abi_core::GpuPtr;
use abi_core::glam::{Mat3, Mat4, Vec3, Vec4};
use abi_mesh::{
    DeformerStack, LatticeDeformer, MAX_DEFORMERS, deform_apply, evaluate_vertex,
    evaluate_vertex_position, lattice_apply, lattice_basis, lattice_jacobian, lattice_point_count,
};
use abi_mesh::{DualQuat, JointWeights};

const EPS: f32 = 1.0e-5;

fn lattice(resolution: [u32; 3], falloff: f32) -> LatticeDeformer {
    LatticeDeformer {
        model_to_lattice: Mat4::IDENTITY,
        lattice_to_model: Mat4::IDENTITY,
        resolution,
        falloff,
        offsets: [[0.0; 4]; abi_mesh::MAX_LATTICE_POINTS],
    }
}

fn zero_offsets(resolution: [u32; 3]) -> Vec<[f32; 4]> {
    vec![[0.0; 4]; lattice_point_count(resolution)]
}

fn assert_vec3_exact(got: Vec3, expected: Vec3) {
    assert_eq!(got.to_array(), expected.to_array());
}

fn assert_vec3_near(got: Vec3, expected: Vec3, eps: f32) {
    let delta = (got - expected).abs();
    assert!(
        delta.max_element() <= eps,
        "got={got:?} expected={expected:?} delta={delta:?}"
    );
}

fn index(resolution: [u32; 3], i: usize, j: usize, k: usize) -> usize {
    let rx = resolution[0] as usize;
    let ry = resolution[1] as usize;
    k * ry * rx + j * rx + i
}

struct Lcg {
    state: u32,
}

impl Lcg {
    fn new(seed: u32) -> Self {
        Self { state: seed }
    }

    fn unit(&mut self) -> f32 {
        self.state = self
            .state
            .wrapping_mul(1_664_525)
            .wrapping_add(1_013_904_223);
        ((self.state >> 8) as f32) * (1.0 / 16_777_216.0)
    }

    fn signed(&mut self, scale: f32) -> f32 {
        (self.unit() * 2.0 - 1.0) * scale
    }
}

fn random_offsets(resolution: [u32; 3], rng: &mut Lcg, scale: f32) -> Vec<[f32; 4]> {
    let mut offsets = zero_offsets(resolution);
    for offset in &mut offsets {
        *offset = [rng.signed(scale), rng.signed(scale), rng.signed(scale), 0.0];
    }
    offsets
}

fn naive_delta(resolution: [u32; 3], offsets: &[[f32; 4]], q: Vec3) -> Vec3 {
    let bx = lattice_basis(resolution[0], q.x);
    let by = lattice_basis(resolution[1], q.y);
    let bz = lattice_basis(resolution[2], q.z);
    let rx = resolution[0] as usize;
    let ry = resolution[1] as usize;
    let rz = resolution[2] as usize;

    let mut delta = Vec3::ZERO;
    for k in 0..rz {
        for j in 0..ry {
            for i in 0..rx {
                let raw = offsets[index(resolution, i, j, k)];
                let o = Vec3::new(raw[0], raw[1], raw[2]);
                delta += o * bx[i] * by[j] * bz[k];
            }
        }
    }
    delta
}

#[test]
fn dual_quat_round_trips_every_rigid_transform() {
    let mut rng = Lcg::new(0x51D3_C0DE);
    for _ in 0..32 {
        let rotation = glam::Quat::from_euler(
            glam::EulerRot::XYZ,
            rng.signed(3.1),
            rng.signed(3.1),
            rng.signed(3.1),
        );
        let translation = Vec3::new(rng.signed(5.0), rng.signed(5.0), rng.signed(5.0));
        let matrix = Mat4::from_rotation_translation(rotation, translation);
        let packed = DualQuat::from_mat4(matrix);

        assert_vec3_near(packed.translation(), translation, 1.0e-5);
        for point in [
            Vec3::ZERO,
            Vec3::new(1.0, -2.0, 3.5),
            Vec3::new(-4.0, 0.25, 7.0),
        ] {
            assert_vec3_near(
                packed.transform_point3(point),
                matrix.transform_point3(point),
                1.0e-4,
            );
        }
        let vector = Vec3::new(0.3, -0.9, 0.4).normalize();
        assert_vec3_near(
            packed.rotate_vector3(vector),
            matrix.transform_vector3(vector),
            1.0e-5,
        );
    }
}

#[test]
fn dual_quat_identity_is_the_neutral_transform() {
    let point = Vec3::new(1.5, -2.5, 3.5);
    assert_vec3_exact(DualQuat::IDENTITY.transform_point3(point), point);
    assert_vec3_exact(DualQuat::IDENTITY.translation(), Vec3::ZERO);
    // Zero is not a valid dual-quaternion palette entry.
    assert_ne!(DualQuat::default().real, DualQuat::IDENTITY.real);
}

#[test]
fn dual_quat_refuses_scale_shear_and_reflection() {
    for (name, matrix) in [
        ("scale", Mat4::from_scale(Vec3::new(1.5, 1.0, 1.0))),
        (
            "shear",
            Mat4::from_cols(
                Vec4::new(1.0, 0.0, 0.0, 0.0),
                Vec4::new(0.25, 1.0, 0.0, 0.0),
                Vec4::new(0.0, 0.0, 1.0, 0.0),
                Vec4::W,
            ),
        ),
        ("reflection", Mat4::from_scale(Vec3::new(-1.0, 1.0, 1.0))),
    ] {
        let panic = std::panic::catch_unwind(move || DualQuat::from_mat4(matrix))
            .expect_err("a non-rigid joint transform must panic");
        let message = panic
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| panic.downcast_ref::<&str>().map(|s| (*s).to_owned()))
            .unwrap_or_default();
        assert!(
            message.contains("rigid") || message.contains("reflection"),
            "{name} must be refused as non-rigid, got {message:?}"
        );
    }
}

#[test]
fn identity_offsets_are_bit_exact_inside_and_outside() {
    for resolution in [[2, 2, 2], [4, 4, 4], [2, 4, 2]] {
        let lat = lattice(resolution, 0.2);
        let offsets = zero_offsets(resolution);
        for p in [Vec3::new(0.25, 0.5, 0.75), Vec3::new(-0.1, 0.5, 0.5)] {
            let n = Vec3::new(0.25, -0.5, 0.75);
            let got = lattice_apply(&lat, offsets.as_slice(), p, n);
            assert_vec3_exact(got.0, p);
            assert_vec3_exact(got.1, n);
        }
    }
}

#[test]
fn cubic_corner_pull_interpolates_endpoints() {
    let resolution = [4, 4, 4];
    let mut offsets = zero_offsets(resolution);
    let pull = Vec3::new(0.25, -0.125, 0.5);
    offsets[index(resolution, 3, 3, 3)] = [pull.x, pull.y, pull.z, 0.0];

    let lat = lattice(resolution, 0.0);
    let n = Vec3::Z;
    let pulled = lattice_apply(&lat, offsets.as_slice(), Vec3::ONE, n);
    assert_vec3_exact(pulled.0, Vec3::ONE + pull);

    let opposite = lattice_apply(&lat, offsets.as_slice(), Vec3::ZERO, n);
    assert_vec3_exact(opposite.0, Vec3::ZERO);
}

#[test]
fn separable_matches_naive_triple_sum() {
    let mut rng = Lcg::new(0xA51A_FFD1);
    for resolution in [[2, 2, 2], [4, 4, 4], [2, 4, 2]] {
        let lat = lattice(resolution, 0.0);
        let offsets = random_offsets(resolution, &mut rng, 0.2);
        for _ in 0..20 {
            let p = Vec3::new(
                0.05 + rng.unit() * 0.9,
                0.05 + rng.unit() * 0.9,
                0.05 + rng.unit() * 0.9,
            );
            let got = lattice_apply(&lat, offsets.as_slice(), p, Vec3::Y).0;
            let expected = p + naive_delta(resolution, offsets.as_slice(), p);
            assert_vec3_near(got, expected, EPS);
        }
    }
}

#[test]
fn analytical_jacobian_matches_finite_difference() {
    let mut rng = Lcg::new(0xD0E0_FF1D);
    for resolution in [[2, 2, 2], [4, 4, 4]] {
        let lat = lattice(resolution, 0.0);
        let offsets = random_offsets(resolution, &mut rng, 0.15);
        for _ in 0..12 {
            let p = Vec3::new(
                0.2 + rng.unit() * 0.6,
                0.2 + rng.unit() * 0.6,
                0.2 + rng.unit() * 0.6,
            );
            let fd = finite_difference_jacobian(&lat, offsets.as_slice(), p);
            let j = lattice_jacobian(&lat, offsets.as_slice(), p);
            for axis in [Vec3::X, Vec3::Y, Vec3::Z] {
                assert_vec3_near(j * axis, fd * axis, 1.0e-3);
            }
        }
    }
}

fn finite_difference_jacobian(lat: &LatticeDeformer, offsets: &[[f32; 4]], p: Vec3) -> Mat3 {
    let h = 1.0e-3;
    let dx = central_difference(lat, offsets, p, Vec3::X, h);
    let dy = central_difference(lat, offsets, p, Vec3::Y, h);
    let dz = central_difference(lat, offsets, p, Vec3::Z, h);
    Mat3::from_cols(dx, dy, dz)
}

fn central_difference(
    lat: &LatticeDeformer,
    offsets: &[[f32; 4]],
    p: Vec3,
    axis: Vec3,
    h: f32,
) -> Vec3 {
    let p0 = lattice_apply(lat, offsets, p - axis * h, Vec3::Y).0;
    let p1 = lattice_apply(lat, offsets, p + axis * h, Vec3::Y).0;
    (p1 - p0) * (0.5 / h)
}

#[test]
fn falloff_clips_hard_and_smooths_inside_band() {
    let resolution = [2, 2, 2];
    let offsets = vec![[0.25, 0.0, 0.0, 0.0]; lattice_point_count(resolution)];

    let smooth = lattice(resolution, 0.2);
    let outside = Vec3::new(-0.01, 0.5, 0.5);
    let got = lattice_apply(&smooth, offsets.as_slice(), outside, Vec3::Y);
    assert_vec3_exact(got.0, outside);
    assert_vec3_exact(got.1, Vec3::Y);

    let hard = lattice(resolution, 0.0);
    let boundary = Vec3::new(0.0, 0.5, 0.5);
    let got = lattice_apply(&hard, offsets.as_slice(), boundary, Vec3::Y);
    assert_vec3_exact(got.0, boundary + Vec3::new(0.25, 0.0, 0.0));
    let just_outside = Vec3::new(-0.0001, 0.5, 0.5);
    assert_vec3_exact(
        lattice_apply(&hard, offsets.as_slice(), just_outside, Vec3::Y).0,
        just_outside,
    );

    let mut previous = lattice_apply(&smooth, offsets.as_slice(), outside, Vec3::Y).0;
    for step in 0..54 {
        let x = -0.01 + (step as f32 + 1.0) * 0.005;
        let p = Vec3::new(x, 0.5, 0.5);
        let current = lattice_apply(&smooth, offsets.as_slice(), p, Vec3::Y).0;
        let jump = (current - previous).length();
        assert!(jump < 0.03, "falloff jump at x={x}: {jump}");
        previous = current;
    }
}

#[test]
fn stack_applies_lattices_in_list_order() {
    let resolution = [2, 2, 2];
    let lat_a = lattice(resolution, 0.0);
    let lat_b = lattice(resolution, 0.0);
    let mut offsets_a = zero_offsets(resolution);
    let mut offsets_b = zero_offsets(resolution);

    for k in 0..2 {
        for i in 0..2 {
            offsets_a[index(resolution, i, 1, k)] = [1.0, 0.0, 0.0, 0.0];
        }
    }
    for offset in &mut offsets_b {
        *offset = [0.0, 0.25, 0.0, 0.0];
    }

    let mut stack = DeformerStack::zeroed();
    stack.count = 2;
    stack.lattices[0] = lat_a;
    stack.lattices[1] = lat_b;
    stack.lattices[0].offsets[..offsets_a.len()].copy_from_slice(&offsets_a);
    stack.lattices[1].offsets[..offsets_b.len()].copy_from_slice(&offsets_b);

    let p = Vec3::new(0.25, 0.25, 0.5);
    let got = deform_apply(&stack, p, Vec3::Z);
    assert_vec3_near(got.0, Vec3::new(0.5, 0.5, 0.5), EPS);

    let mut reversed = DeformerStack::zeroed();
    reversed.count = 2;
    reversed.lattices[0] = lat_b;
    reversed.lattices[1] = lat_a;
    reversed.lattices[0].offsets[..offsets_b.len()].copy_from_slice(&offsets_b);
    reversed.lattices[1].offsets[..offsets_a.len()].copy_from_slice(&offsets_a);
    let reversed_got = deform_apply(&reversed, p, Vec3::Z);
    assert_vec3_near(reversed_got.0, Vec3::new(0.75, 0.5, 0.5), EPS);
}

#[test]
fn evaluate_vertex_null_inputs_are_identity() {
    let p = Vec3::new(0.4, -0.2, 1.5);
    let n = Vec3::new(0.0, 0.0, 1.0);
    let got = evaluate_vertex(
        GpuPtr::<DualQuat>::null(),
        GpuPtr::<JointWeights>::null(),
        0,
        GpuPtr::<DeformerStack>::null(),
        17,
        p,
        n,
    );
    assert_vec3_exact(got.0, p);
    assert_vec3_exact(got.1, n);
}

#[test]
#[should_panic(expected = "joint transform and weight pointers must be null or non-null together")]
fn evaluate_vertex_rejects_unpaired_joint_pointers() {
    let _ = evaluate_vertex(
        GpuPtr::<DualQuat>::from_addr(16),
        GpuPtr::<JointWeights>::null(),
        0,
        GpuPtr::<DeformerStack>::null(),
        0,
        Vec3::ZERO,
        Vec3::Y,
    );
}

#[test]
fn bind_pose_identity_is_unchanged() {
    let transforms = [DualQuat::IDENTITY; 4];
    let weights = [JointWeights {
        joint_indices: [0, 1, 2, 3],
        weights: [0.25; 4],
    }];
    let p = Vec3::new(0.5, -1.25, 3.0);
    let n = Vec3::Z;
    let got = evaluate_vertex(
        transforms.as_slice(),
        weights.as_slice(),
        0,
        GpuPtr::<DeformerStack>::null(),
        0,
        p,
        n,
    );
    assert_vec3_exact(got.0, p);
    assert_vec3_exact(got.1, n);
}

/// Pure translations make dual-quaternion and linear blends equivalent.
#[test]
fn translation_only_blend_is_the_weighted_mean() {
    let transforms = [
        DualQuat::from_mat4(Mat4::from_translation(Vec3::new(2.0, 0.0, 0.0))),
        DualQuat::from_mat4(Mat4::from_translation(Vec3::new(0.0, 4.0, 0.0))),
    ];
    let weights = [JointWeights {
        joint_indices: [0, 1, 0, 1],
        weights: [0.25, 0.75, 0.0, 0.0],
    }];
    let got = evaluate_vertex(
        transforms.as_slice(),
        weights.as_slice(),
        0,
        GpuPtr::<DeformerStack>::null(),
        0,
        Vec3::new(1.0, 2.0, 3.0),
        Vec3::Z,
    );
    assert_vec3_near(got.0, Vec3::new(1.5, 5.0, 3.0), EPS);
    assert_vec3_near(got.1, Vec3::Z, EPS);
}

/// Dual-quaternion blending preserves radius during deep joint flexion.
#[test]
fn deep_flexion_blend_preserves_radius_where_linear_blend_collapses() {
    for degrees in [60.0f32, 90.0, 110.0, 140.0] {
        let angle = degrees.to_radians();
        let bent = Mat4::from_rotation_z(angle);
        let transforms = [
            DualQuat::from_mat4(Mat4::IDENTITY),
            DualQuat::from_mat4(bent),
        ];
        let weights = [JointWeights {
            joint_indices: [0, 1, 0, 1],
            weights: [0.5, 0.5, 0.0, 0.0],
        }];
        // A ring vertex one unit out along +x from the joint.
        let rest = Vec3::X;
        let skinned = evaluate_vertex_position(
            transforms.as_slice(),
            weights.as_slice(),
            0,
            GpuPtr::<DeformerStack>::null(),
            0,
            rest,
        );

        // Dual quaternions: the exact half-angle rotation, radius intact.
        assert_vec3_near(
            skinned,
            Mat4::from_rotation_z(angle * 0.5).transform_point3(rest),
            1.0e-5,
        );
        assert!(
            (skinned.length() - 1.0).abs() < 1.0e-5,
            "{degrees}°: dual-quaternion radius {} must stay 1",
            skinned.length()
        );

        // Linear blending provides the collapse reference.
        let linear_blend = 0.5 * rest + 0.5 * bent.transform_point3(rest);
        let expected_collapse = (angle * 0.5).cos();
        assert!(
            (linear_blend.length() - expected_collapse).abs() < 1.0e-5,
            "{degrees}°: linear-blend oracle must collapse to cos(angle/2)"
        );
        assert!(
            skinned.length() > linear_blend.length() + 0.01,
            "{degrees}°: dual quaternions must beat the linear blend's collapse"
        );
    }
}

/// Hemisphere alignment makes antipodal quaternion signs equivalent.
#[test]
fn hemisphere_alignment_makes_the_blend_sign_invariant() {
    let bend = Mat4::from_rotation_z(1.9);
    let aligned = [
        DualQuat::from_mat4(Mat4::IDENTITY),
        DualQuat::from_mat4(bend),
    ];
    let mut flipped = aligned;
    for value in flipped[1].real.iter_mut().chain(flipped[1].dual.iter_mut()) {
        *value = -*value;
    }
    let weights = [JointWeights {
        joint_indices: [0, 1, 0, 1],
        weights: [0.4, 0.6, 0.0, 0.0],
    }];
    let position = Vec3::new(1.0, 0.25, -0.5);
    let evaluate = |palette: &[DualQuat]| {
        evaluate_vertex_position(
            palette,
            weights.as_slice(),
            0,
            GpuPtr::<DeformerStack>::null(),
            0,
            position,
        )
    };
    assert_vec3_near(evaluate(&flipped), evaluate(&aligned), 1.0e-5);
}

/// The pivot weight bounds the blend norm away from zero.
#[test]
fn blend_norm_never_collapses_even_for_antipodal_palettes() {
    let transforms = [
        DualQuat::IDENTITY,
        DualQuat::from_rotation_translation(
            glam::Quat::from_rotation_z(std::f32::consts::PI * 0.999),
            Vec3::new(0.5, -0.25, 0.75),
        ),
    ];
    let weights = [JointWeights {
        joint_indices: [0, 1, 0, 1],
        weights: [0.5, 0.5, 0.0, 0.0],
    }];
    let normal = Vec3::new(0.3, -0.9, 0.4).normalize();
    let got = evaluate_vertex(
        transforms.as_slice(),
        weights.as_slice(),
        0,
        GpuPtr::<DeformerStack>::null(),
        0,
        Vec3::new(1.0, 0.5, -0.25),
        normal,
    );
    assert!(got.0.is_finite() && got.1.is_finite());
    assert!(
        (got.1.length() - 1.0).abs() < 1.0e-5,
        "a rigid blend must leave the normal unit, got {}",
        got.1.length()
    );
}

#[test]
fn four_joint_blend_is_rigid_and_matches_its_own_normalized_sum() {
    let transforms = [
        DualQuat::from_mat4(Mat4::from_rotation_translation(
            glam::Quat::from_rotation_x(0.3),
            Vec3::new(1.0, 2.0, 3.0),
        )),
        DualQuat::from_mat4(Mat4::from_rotation_translation(
            glam::Quat::from_rotation_y(-0.5),
            Vec3::new(4.0, 5.0, 6.0),
        )),
        DualQuat::from_mat4(Mat4::from_rotation_translation(
            glam::Quat::from_rotation_z(0.8),
            Vec3::new(7.0, 8.0, 9.0),
        )),
        DualQuat::from_mat4(Mat4::from_rotation_translation(
            glam::Quat::from_euler(glam::EulerRot::XYZ, 0.2, 0.4, -0.6),
            Vec3::new(10.0, 11.0, 12.0),
        )),
    ];
    let weights = [JointWeights {
        joint_indices: [0, 1, 2, 3],
        weights: [0.125, 0.25, 0.125, 0.5],
    }];
    let position = Vec3::new(2.0, -1.0, 0.5);
    let normal = Vec3::new(1.0, 1.0, 0.0).normalize();
    let got = evaluate_vertex(
        transforms.as_slice(),
        weights.as_slice(),
        0,
        GpuPtr::<DeformerStack>::null(),
        0,
        position,
        normal,
    );

    // Rigid transforms preserve distances and unit normals.
    let other = Vec3::new(-1.0, 3.0, 2.25);
    let got_other = evaluate_vertex_position(
        transforms.as_slice(),
        weights.as_slice(),
        0,
        GpuPtr::<DeformerStack>::null(),
        0,
        other,
    );
    assert!(
        ((got.0 - got_other).length() - (position - other).length()).abs() < 1.0e-4,
        "the blended transform must be an isometry"
    );
    assert!((got.1.length() - 1.0).abs() <= EPS);
    // The position-only twin is the same arithmetic, bit for bit.
    assert_vec3_exact(
        got.0,
        evaluate_vertex_position(
            transforms.as_slice(),
            weights.as_slice(),
            0,
            GpuPtr::<DeformerStack>::null(),
            0,
            position,
        ),
    );
}

#[test]
fn full_and_position_only_skinning_are_bit_exact() {
    let transforms = [
        DualQuat::from_mat4(Mat4::from_rotation_translation(
            glam::Quat::from_euler(glam::EulerRot::XYZ, 0.7, -0.2, 1.1),
            Vec3::new(1.0, -2.0, 0.5),
        )),
        DualQuat::from_mat4(Mat4::from_rotation_translation(
            glam::Quat::from_euler(glam::EulerRot::XYZ, -0.4, 0.9, 0.3),
            Vec3::new(-3.0, 1.0, 2.0),
        )),
    ];
    let weights = [JointWeights {
        joint_indices: [0, 1, 0, 1],
        weights: [0.13, 0.27, 0.19, 0.41],
    }];
    let p = Vec3::new(0.37, -1.25, 2.5);
    let full = evaluate_vertex(
        transforms.as_slice(),
        weights.as_slice(),
        0,
        GpuPtr::<DeformerStack>::null(),
        0,
        p,
        Vec3::new(0.2, 0.9, -0.3).normalize(),
    );
    let position_only = evaluate_vertex_position(
        transforms.as_slice(),
        weights.as_slice(),
        0,
        GpuPtr::<DeformerStack>::null(),
        0,
        p,
    );
    assert_vec3_exact(full.0, position_only);
}

#[test]
fn skinned_normal_is_rotated_and_stays_unit() {
    let rotation = glam::Quat::from_euler(glam::EulerRot::XYZ, 0.4, -1.1, 0.25);
    let transforms = [DualQuat::from_rotation_translation(
        rotation,
        Vec3::new(50.0, -30.0, 10.0),
    )];
    let weights = [JointWeights {
        joint_indices: [0; 4],
        weights: [1.0, 0.0, 0.0, 0.0],
    }];
    let normal = Vec3::new(1.0, 1.0, 0.0).normalize();
    let got = evaluate_vertex(
        transforms.as_slice(),
        weights.as_slice(),
        0,
        GpuPtr::<DeformerStack>::null(),
        0,
        Vec3::ZERO,
        normal,
    )
    .1;
    // Translation cannot reach a normal, and a rotation needs no
    // inverse-transpose: the rotated normal IS the correct one.
    assert_vec3_near(got, rotation * normal, EPS);
    assert!((got.length() - 1.0).abs() <= EPS);
}

#[test]
fn stack_view_empty_is_identity() {
    let stack = DeformerStack::zeroed();
    let p = Vec3::new(0.2, 0.3, 0.4);
    let n = Vec3::Y;
    let got = deform_apply(&stack, p, n);
    assert_vec3_exact(got.0, p);
    assert_vec3_exact(got.1, n);
    assert_eq!(MAX_DEFORMERS, 4);
}
