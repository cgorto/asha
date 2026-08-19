use abi_core::glam::{UVec2, Vec3};
use abi_core::{View, hardware_depth, ray_direction};
use abi_light::mesh_shade_slim;
use abi_mesh::{MeshShadeLighting, mesh_world_to_clip};

fn test_view() -> View {
    let forward = Vec3::new(0.3, -0.2, 1.0).normalize();
    let right = forward.cross(Vec3::Y).normalize();
    let up = right.cross(forward);
    View {
        camera_position: [12.0, 8.0, -30.0],
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

#[test]
fn mesh_world_to_clip_matches_voxel_hardware_depth() {
    let view = test_view();
    let m = mesh_world_to_clip(&view);
    let origin = Vec3::from_array(view.camera_position);
    let pixels = [
        UVec2::new(0, 0),
        UVec2::new(1279, 0),
        UVec2::new(0, 719),
        UVec2::new(1279, 719),
        UVec2::new(640, 360),
        UVec2::new(377, 211),
    ];
    let depths = [0.25, 1.0, 7.0, 31.0, 250.0];

    for pixel in pixels {
        let ray = ray_direction(&view, pixel);
        let uv = (pixel.as_vec2() + 0.5) / UVec2::from_array(view.output_size).as_vec2();
        let ndc_x = uv.x * 2.0 - 1.0;
        let ndc_y = uv.y * 2.0 - 1.0;
        for t in depths {
            // `hardware_depth` uses eye-axis distance: for a ray-distance
            // sample it divides by the forward/ray cosine.
            let world = origin + ray * t;
            let clip = m * world.extend(1.0);
            let got = clip.z / clip.w;
            let expected = hardware_depth(&view, ray, t);
            assert!(
                (got - expected).abs() < 2.0e-5,
                "pixel {pixel} t {t}: mesh depth {got} != voxel {expected}"
            );
            let got_x = clip.x / clip.w;
            let got_y = clip.y / clip.w;
            assert!(
                (got_x - ndc_x).abs() < 5.0e-5,
                "pixel {pixel} t {t}: clip x {got_x} != {ndc_x}"
            );
            assert!(
                (got_y - ndc_y).abs() < 5.0e-5,
                "pixel {pixel} t {t}: clip y {got_y} != {ndc_y}"
            );
        }
    }
}

#[test]
fn mesh_slim_shading_uses_sun_and_hemisphere_ambient() {
    let lighting = MeshShadeLighting {
        sun_direction: Vec3::Y.to_array(),
        sun_tint: [2.0, 1.0, 0.5],
        sky_ambient: [0.4, 0.5, 0.6],
        ground_ambient: [0.1, 0.2, 0.3],
        ..MeshShadeLighting::zeroed()
    };

    let up = mesh_shade_slim(Vec3::Y, [0.5, 0.25, 0.1], &lighting);
    assert_vec3_close(up, [1.2, 0.375, 0.11]);

    let side = mesh_shade_slim(Vec3::X, [1.0, 1.0, 1.0], &lighting);
    assert_vec3_close(side, [0.25, 0.35, 0.45]);

    let down = mesh_shade_slim(Vec3::NEG_Y, [2.0, 3.0, 4.0], &lighting);
    assert_vec3_close(down, [0.2, 0.6, 1.2]);
}

fn assert_vec3_close(got: [f32; 3], expected: [f32; 3]) {
    for (i, (got, expected)) in got.into_iter().zip(expected).enumerate() {
        assert!(
            (got - expected).abs() < 1.0e-6,
            "channel {i}: {got} != {expected}"
        );
    }
}
