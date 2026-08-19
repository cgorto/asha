//! Linear-gradient geometry and stop-position math from
//! `bevy_ui_render::gradient` (Bevy 0.19).
//!
//! This keeps CSS/bevy UI geometry and automatic-stop interpolation, but
//! assumes explicit stops are already ordered instead of doing upstream's
//! defensive re-sort.

use bevy::math::Vec2;
use bevy::ui::ColorStop;
use bevy_color::Color;

/// Returns gradient-line length for a CSS angle and node size.
///
/// Follows `bevy_ui_render::gradient::compute_gradient_line_length`.
/// Angle zero points up; angles increase clockwise.
pub(crate) fn compute_gradient_line_length(angle: f32, size: Vec2) -> f32 {
    let center = 0.5 * size;
    let v = Vec2::new(angle.sin(), -angle.cos());

    let (pos_corner, neg_corner) = if v.x >= 0.0 && v.y <= 0.0 {
        (Vec2::new(size.x, 0.0), Vec2::new(0.0, size.y))
    } else if v.x >= 0.0 && v.y > 0.0 {
        (size, Vec2::ZERO)
    } else if v.x < 0.0 && v.y <= 0.0 {
        (Vec2::ZERO, size)
    } else {
        (Vec2::new(0.0, size.y), Vec2::new(size.x, 0.0))
    };

    let t_pos = (pos_corner - center).dot(v);
    let t_neg = (neg_corner - center).dot(v);
    (t_pos - t_neg).abs()
}

/// Resolves stop positions and interpolates `Val::Auto` stops, following
/// `bevy_ui_render::gradient::{compute_color_stops, interpolate_color_stops}`.
/// Explicit stops must be non-decreasing; unlike upstream, they are not
/// defensively re-sorted.
pub(crate) fn resolve_gradient_stops(
    stops: &[ColorStop],
    scale_factor: f32,
    length: f32,
    target_size: Vec2,
    out: &mut Vec<(Color, f32, f32)>,
) {
    out.clear();
    out.extend(stops.iter().map(|stop| {
        let pos = stop
            .point
            .resolve(scale_factor, length, target_size)
            .unwrap_or(f32::NAN);
        (stop.color, pos, stop.hint)
    }));
    if out.is_empty() {
        return;
    }

    let min = out
        .iter()
        .map(|(_, p, _)| *p)
        .find(|p| !p.is_nan())
        .unwrap_or(0.0)
        .min(0.0);
    let max = out
        .iter()
        .rev()
        .map(|(_, p, _)| *p)
        .find(|p| !p.is_nan())
        .unwrap_or(length)
        .max(length);

    let last = out.len() - 1;
    if out[0].1.is_nan() {
        out[0].1 = min;
    }
    if out[last].1.is_nan() {
        out[last].1 = max;
    }

    // Interpolate each contiguous run of automatic stops.
    let mut i = 1;
    while i < last {
        if out[i].1.is_nan() {
            let start = i;
            let mut end = i + 1;
            while end < last && out[end].1.is_nan() {
                end += 1;
            }
            let start_point = out[start - 1].1;
            let end_point = out[end].1;
            let steps = end - start;
            let step = (end_point - start_point) / (steps + 1) as f32;
            for j in 0..steps {
                out[i + j].1 = start_point + step * (j + 1) as f32;
            }
            i = end;
        }
        i += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::ui::Val;

    #[test]
    fn line_length_matches_diagonal_for_45_degrees() {
        let size = Vec2::new(100.0, 50.0);
        // Zero angle spans the height.
        assert!((compute_gradient_line_length(0.0, size) - 50.0).abs() < 1e-4);
        // A quarter turn spans the width.
        assert!(
            (compute_gradient_line_length(std::f32::consts::FRAC_PI_2, size) - 100.0).abs() < 1e-4
        );
    }

    #[test]
    fn implicit_stops_interpolate_evenly() {
        let stops = vec![
            ColorStop {
                color: Color::WHITE,
                point: Val::Percent(0.0),
                hint: 0.5,
            },
            ColorStop {
                color: Color::WHITE,
                point: Val::Auto,
                hint: 0.5,
            },
            ColorStop {
                color: Color::WHITE,
                point: Val::Auto,
                hint: 0.5,
            },
            ColorStop {
                color: Color::WHITE,
                point: Val::Percent(100.0),
                hint: 0.5,
            },
        ];
        let mut out = Vec::new();
        resolve_gradient_stops(&stops, 1.0, 90.0, Vec2::new(90.0, 90.0), &mut out);
        let positions: Vec<f32> = out.iter().map(|(_, p, _)| *p).collect();
        assert!((positions[0] - 0.0).abs() < 1e-4);
        assert!((positions[1] - 30.0).abs() < 1e-4);
        assert!((positions[2] - 60.0).abs() < 1e-4);
        assert!((positions[3] - 90.0).abs() < 1e-4);
    }
}
