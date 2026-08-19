//! Headless GPU proofs for fills, borders, gradients, clipping, and scissors.
//!
//! `abi_ui` CPU shading provides the test oracle.

mod common;

use abi_core::glam::{Vec2, Vec4};
use abi_ui::{
    UI_FLAG_BORDER_LEFT, UI_FLAG_BORDER_TOP, UI_FLAG_GRADIENT, UI_FLAG_GRADIENT_SPACE_HSLA,
    ui_fragment_shade,
};
use common::{
    composite, gpu_test_lock, one_batch, pixel, quad, quad_raw, render_ui, upload_quads, view_for,
};
use gpu::Gpu;
use ui::{UiBatch, UiPass, UiScissor};

const BG: [f32; 4] = [0.05, 0.06, 0.08, 1.0];

/// Verifies rounded fill corners, centers, and antialiasing.
#[test]
fn rounded_rect_solid_fill_corner_center_and_aa_band() {
    let _guard = gpu_test_lock();
    let gpu = Gpu::new(true).expect("vulkan init");
    let pass = UiPass::new(&gpu);

    const SIZE: u32 = 100;
    let size = [SIZE, SIZE];
    let fill = Vec4::new(0.9, 0.2, 0.3, 1.0);
    let radius = Vec4::splat(20.0);

    let v = quad(
        [0.0, 0.0, SIZE as f32, SIZE as f32],
        fill,
        Vec4::ZERO,
        radius,
        Vec4::ZERO,
        0,
        [[0.0; 2]; 4],
    );
    let uploaded = upload_quads(&gpu, &[v], view_for(size));
    let batch = one_batch(uploaded.draw, 1, None);
    let readback = render_ui(&gpu, &pass, size, BG, &batch);

    let bg = Vec4::from_array(BG);

    let center = pixel(&readback, SIZE, 50, 50);
    assert!(
        (center - fill).abs().max_element() < 1e-4,
        "center should be pure fill: {center:?}"
    );

    let corner = pixel(&readback, SIZE, 2, 2);
    assert!(
        (corner - bg).abs().max_element() < 1e-4,
        "corner should be background: {corner:?}"
    );

    // Find a partial-coverage arc pixel using the CPU oracle.
    let node_center = Vec2::splat(SIZE as f32 / 2.0);
    let mut found = false;
    for x in 0..30u32 {
        for y in 0..30u32 {
            let px = Vec2::new(x as f32 + 0.5, y as f32 + 0.5);
            let point = px - node_center;
            let shaded = ui_fragment_shade(
                fill,
                Vec4::ZERO,
                Vec2::ZERO,
                point,
                Vec2::splat(SIZE as f32),
                radius,
                Vec4::ZERO,
                0,
            );
            if shaded.w > 0.05 && shaded.w < 0.95 {
                found = true;
                let want = composite(shaded, bg);
                let got = pixel(&readback, SIZE, x, y);
                assert!(
                    (got - want).abs().max_element() < 5e-3,
                    "AA pixel ({x},{y}): gpu {got:?} vs cpu {want:?}"
                );
            }
        }
    }
    assert!(found, "no AA-band pixel found near the corner arc");

    uploaded.free(&gpu);
    pass.free(&gpu);
}

/// Verifies independent left and top border edges.
#[test]
fn per_edge_borders_probe_left_top_and_transparent_interior() {
    let _guard = gpu_test_lock();
    let gpu = Gpu::new(true).expect("vulkan init");
    let pass = UiPass::new(&gpu);

    const SIZE: u32 = 100;
    let size = [SIZE, SIZE];
    let border = Vec4::splat(10.0);
    let color_a = Vec4::new(0.8, 0.1, 0.1, 1.0);
    let color_b = Vec4::new(0.1, 0.2, 0.9, 1.0);

    let rect = [0.0, 0.0, SIZE as f32, SIZE as f32];
    let left = quad(
        rect,
        color_a,
        Vec4::ZERO,
        Vec4::ZERO,
        border,
        UI_FLAG_BORDER_LEFT,
        [[0.0; 2]; 4],
    );
    let top = quad(
        rect,
        color_b,
        Vec4::ZERO,
        Vec4::ZERO,
        border,
        UI_FLAG_BORDER_TOP,
        [[0.0; 2]; 4],
    );

    let uploaded = upload_quads(&gpu, &[left, top], view_for(size));
    let batch = one_batch(uploaded.draw, 2, None);
    let readback = render_ui(&gpu, &pass, size, BG, &batch);

    let bg = Vec4::from_array(BG);

    let left_mid = pixel(&readback, SIZE, 5, 50);
    assert!(
        (left_mid - composite(color_a, bg)).abs().max_element() < 5e-3,
        "left-mid should be color A: {left_mid:?}"
    );

    let top_mid = pixel(&readback, SIZE, 50, 5);
    assert!(
        (top_mid - composite(color_b, bg)).abs().max_element() < 5e-3,
        "top-mid should be color B: {top_mid:?}"
    );

    let interior = pixel(&readback, SIZE, 50, 50);
    assert!(
        (interior - bg).abs().max_element() < 1e-4,
        "interior should stay transparent (background shows through): {interior:?}"
    );

    uploaded.free(&gpu);
    pass.free(&gpu);
}

/// Verifies sRGB and HSL gradient interpolation on the GPU.
#[test]
fn linear_gradient_matches_oracle_and_hsl_space_differs_from_srgb() {
    let _guard = gpu_test_lock();
    let gpu = Gpu::new(true).expect("vulkan init");
    let pass = UiPass::new(&gpu);

    const W: u32 = 100;
    const H: u32 = 20;
    let size = [W, H];
    let rect = [0.0, 0.0, W as f32, H as f32];
    // UV x carries the gradient parameter.
    let uv = [[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 0.0]];

    let srgb_start = Vec4::new(1.0, 0.0, 0.0, 1.0);
    let srgb_end = Vec4::new(0.0, 0.0, 1.0, 1.0);
    let hsl_start = Vec4::new(0.0, 1.0, 0.5, 1.0);
    let hsl_end = Vec4::new(240.0 / 360.0, 1.0, 0.5, 1.0);

    let srgb_quad = quad(
        rect,
        srgb_start,
        srgb_end,
        Vec4::ZERO,
        Vec4::ZERO,
        UI_FLAG_GRADIENT,
        uv,
    );
    let hsl_quad = quad(
        rect,
        hsl_start,
        hsl_end,
        Vec4::ZERO,
        Vec4::ZERO,
        UI_FLAG_GRADIENT | UI_FLAG_GRADIENT_SPACE_HSLA,
        uv,
    );

    let srgb_up = upload_quads(&gpu, &[srgb_quad], view_for(size));
    let srgb_readback = render_ui(&gpu, &pass, size, BG, &one_batch(srgb_up.draw, 1, None));

    let hsl_up = upload_quads(&gpu, &[hsl_quad], view_for(size));
    let hsl_readback = render_ui(&gpu, &pass, size, BG, &one_batch(hsl_up.draw, 1, None));

    let bg = Vec4::from_array(BG);
    let (probe_x, probe_y) = (W / 2, H / 2);
    let probe_uv = (probe_x as f32 + 0.5) / W as f32;
    let probe_point = Vec2::new(
        probe_x as f32 + 0.5 - W as f32 / 2.0,
        probe_y as f32 + 0.5 - H as f32 / 2.0,
    );
    let node_size = Vec2::new(W as f32, H as f32);

    let srgb_expected = ui_fragment_shade(
        srgb_start,
        srgb_end,
        Vec2::new(probe_uv, 0.0),
        probe_point,
        node_size,
        Vec4::ZERO,
        Vec4::ZERO,
        UI_FLAG_GRADIENT,
    );
    let hsl_expected = ui_fragment_shade(
        hsl_start,
        hsl_end,
        Vec2::new(probe_uv, 0.0),
        probe_point,
        node_size,
        Vec4::ZERO,
        Vec4::ZERO,
        UI_FLAG_GRADIENT | UI_FLAG_GRADIENT_SPACE_HSLA,
    );

    let srgb_got = pixel(&srgb_readback, W, probe_x, probe_y);
    let srgb_want = composite(srgb_expected, bg);
    assert!(
        (srgb_got - srgb_want).abs().max_element() < 5e-3,
        "sRGB-space midpoint: gpu {srgb_got:?} vs cpu {srgb_want:?}"
    );

    let hsl_got = pixel(&hsl_readback, W, probe_x, probe_y);
    let hsl_want = composite(hsl_expected, bg);
    assert!(
        (hsl_got - hsl_want).abs().max_element() < 5e-3,
        "HSL-space midpoint: gpu {hsl_got:?} vs cpu {hsl_want:?}"
    );

    // The two color spaces must produce distinct midpoint colors.
    assert!(
        (hsl_expected - srgb_expected).length() > 0.05,
        "HSL and sRGB gradient midpoints should differ: {hsl_expected:?} vs {srgb_expected:?}"
    );
    // Confirm both paths through GPU output.
    assert!(
        (hsl_got - srgb_got).length() > 0.05,
        "GPU HSL and sRGB midpoints should differ: {hsl_got:?} vs {srgb_got:?}"
    );

    srgb_up.free(&gpu);
    hsl_up.free(&gpu);
    pass.free(&gpu);
}

/// Verifies geometric clipping updates positions, points, and UVs.
#[test]
fn corner_displaced_clip_paints_nothing_outside_rect() {
    let _guard = gpu_test_lock();
    let gpu = Gpu::new(true).expect("vulkan init");
    let pass = UiPass::new(&gpu);

    const SIZE: u32 = 100;
    let size = [SIZE, SIZE];
    let fill = Vec4::new(0.2, 0.7, 0.3, 1.0);

    let center = [50.0f32, 50.0f32];
    let logical: [[f32; 2]; 4] = [[10.0, 10.0], [90.0, 10.0], [90.0, 90.0], [10.0, 90.0]];
    let clip_min = [30.0f32, 30.0f32];
    let clip_max = [70.0f32, 70.0f32];

    let clamp2 = |p: [f32; 2]| {
        [
            p[0].clamp(clip_min[0], clip_max[0]),
            p[1].clamp(clip_min[1], clip_max[1]),
        ]
    };
    let positions: [[f32; 2]; 4] = core::array::from_fn(|i| clamp2(logical[i]));
    // The SDF point follows the displaced position.
    let point: [[f32; 2]; 4] =
        core::array::from_fn(|i| [positions[i][0] - center[0], positions[i][1] - center[1]]);
    // UVs track displacement in logical coordinates.
    let uv: [[f32; 2]; 4] = core::array::from_fn(|i| {
        [
            (positions[i][0] - 10.0) / 80.0,
            (positions[i][1] - 10.0) / 80.0,
        ]
    });

    let v = quad_raw(
        positions,
        uv,
        point,
        [80.0, 80.0],
        fill,
        Vec4::ZERO,
        Vec4::ZERO,
        Vec4::ZERO,
        0,
    );
    let uploaded = upload_quads(&gpu, &[v], view_for(size));
    let batch = one_batch(uploaded.draw, 1, None);
    let readback = render_ui(&gpu, &pass, size, BG, &batch);

    let bg = Vec4::from_array(BG);

    let outside_corner = pixel(&readback, SIZE, 12, 12);
    assert!(
        (outside_corner - bg).abs().max_element() < 1e-4,
        "outside the clip rect (original corner) should be untouched: {outside_corner:?}"
    );
    let outside_edge = pixel(&readback, SIZE, 75, 50);
    assert!(
        (outside_edge - bg).abs().max_element() < 1e-4,
        "just outside the clip rect edge should be untouched: {outside_edge:?}"
    );

    let inside = pixel(&readback, SIZE, 50, 50);
    assert!(
        (inside - fill).abs().max_element() < 1e-4,
        "inside the clip rect should be filled: {inside:?}"
    );
    let inside_near_edge = pixel(&readback, SIZE, 65, 50);
    assert!(
        (inside_near_edge - fill).abs().max_element() < 1e-4,
        "inside the clip rect near its edge should be filled: {inside_near_edge:?}"
    );

    uploaded.free(&gpu);
    pass.free(&gpu);
}

/// Verifies per-batch scissor compositing.
#[test]
fn second_batch_scissor_leaves_first_batch_pixels_outside_it_untouched() {
    let _guard = gpu_test_lock();
    let gpu = Gpu::new(true).expect("vulkan init");
    let pass = UiPass::new(&gpu);

    const SIZE: u32 = 100;
    let size = [SIZE, SIZE];
    let color_a = Vec4::new(0.8, 0.3, 0.1, 1.0);
    let color_b = Vec4::new(0.1, 0.4, 0.9, 1.0);

    let full = |color: Vec4| {
        quad(
            [0.0, 0.0, SIZE as f32, SIZE as f32],
            color,
            Vec4::ZERO,
            Vec4::ZERO,
            Vec4::ZERO,
            0,
            [[0.0; 2]; 4],
        )
    };
    let batch1_up = upload_quads(&gpu, &[full(color_a)], view_for(size));
    let batch2_up = upload_quads(&gpu, &[full(color_b)], view_for(size));

    let scissor = UiScissor {
        offset: [60, 0],
        extent: [40, SIZE],
    };
    let batches = [
        UiBatch {
            draw: batch1_up.draw.gpu,
            quad_count: 1,
            scissor: None,
        },
        UiBatch {
            draw: batch2_up.draw.gpu,
            quad_count: 1,
            scissor: Some(scissor),
        },
    ];

    let readback = render_ui(&gpu, &pass, size, BG, &batches);

    let outside = pixel(&readback, SIZE, 30, 50);
    assert!(
        (outside - color_a).abs().max_element() < 1e-4,
        "outside the scissor should show batch 1: {outside:?}"
    );
    let inside = pixel(&readback, SIZE, 80, 50);
    assert!(
        (inside - color_b).abs().max_element() < 1e-4,
        "inside the scissor should show batch 2: {inside:?}"
    );
    let edge_in = pixel(&readback, SIZE, 61, 50);
    assert!(
        (edge_in - color_b).abs().max_element() < 1e-4,
        "just inside the scissor edge should show batch 2: {edge_in:?}"
    );
    let edge_out = pixel(&readback, SIZE, 59, 50);
    assert!(
        (edge_out - color_a).abs().max_element() < 1e-4,
        "just outside the scissor edge should show batch 1: {edge_out:?}"
    );

    batch1_up.free(&gpu);
    batch2_up.free(&gpu);
    pass.free(&gpu);
}
