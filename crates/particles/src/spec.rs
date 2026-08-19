//! Authoring surface: curves, gradients, shapes, and emitter/effect specs.

use abi_core::glam::{Mat4, UVec2, Vec3};
use abi_particles::ShapeGpu;

pub const MAX_EFFECT_EMITTERS: usize = 8;
pub const PRIMITIVE_COUNT: u32 = 6;

/// Camera data shared by culling and vertex pulling.
#[derive(Clone, Copy, Debug)]
pub struct ParticleView {
    pub view_proj: Mat4,
    pub camera_right: Vec3,
    pub camera_up: Vec3,
    pub camera_forward: Vec3,
    pub screen_size: UVec2,
}

/// Particle primitive types; each uses one indirect draw.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u32)]
pub enum Primitive {
    #[default]
    Quad = 0,
    Disc = 1,
    Cube = 2,
    Icosphere = 3,
    Cone = 4,
    Prism = 5,
}

impl Primitive {
    pub(crate) const fn index_count(self) -> u32 {
        match self {
            Self::Quad => 6,
            Self::Disc => 36,
            Self::Cube => 36,
            Self::Icosphere => 240,
            Self::Cone => 48,
            Self::Prism => 24,
        }
    }

    pub(crate) const fn radius(self) -> f32 {
        match self {
            Self::Quad => 0.707_106_77,
            Self::Disc => 0.5,
            Self::Cube => 0.866_025_4,
            Self::Icosphere => 0.5,
            Self::Cone | Self::Prism => 0.866_025_4,
        }
    }
}

/// Orientation policy for a particle primitive.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u32)]
pub enum Alignment {
    #[default]
    Billboard = 0,
    Mesh3d = 1,
    YToVelocity = 2,
    BillboardYToVelocity = 3,
    BillboardY = 4,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u32)]
pub enum CurveMode {
    #[default]
    Linear,
    Hold,
    Power,
}

/// Curve point; segment mode and tension belong to the right endpoint.
#[derive(Clone, Copy, Debug, Default)]
pub struct CurvePoint {
    pub position: f32,
    pub value: f32,
    pub tension: f32,
    pub mode: CurveMode,
}

/// Up to eight points; clamped values map into the configured range.
#[derive(Clone, Copy, Debug)]
pub struct Curve {
    pub points: [CurvePoint; 8],
    pub count: u8,
    pub range_min: f32,
    pub range_max: f32,
}

impl Default for Curve {
    fn default() -> Self {
        Self {
            points: [CurvePoint::default(); 8],
            count: 0,
            range_min: 0.0,
            range_max: 1.0,
        }
    }
}

impl Curve {
    pub fn constant(value: f32) -> Self {
        Self::from_points(&[CurvePoint {
            value,
            ..Default::default()
        }])
    }

    pub fn linear(from: f32, to: f32) -> Self {
        Self::from_points(&[
            CurvePoint {
                value: from,
                ..Default::default()
            },
            CurvePoint {
                position: 1.0,
                value: to,
                ..Default::default()
            },
        ])
    }

    pub fn ease_in(from: f32, to: f32, tension: f32) -> Self {
        assert_tension(tension);
        let mut curve = Self::linear(from, to);
        curve.points[1].mode = CurveMode::Power;
        curve.points[1].tension = tension;
        curve
    }

    pub fn ease_out(from: f32, to: f32, tension: f32) -> Self {
        let mut curve = Self::ease_in(from, to, tension);
        curve.points[1].tension = -tension;
        curve
    }

    pub fn fade_in_out(peak: f32) -> Self {
        Self::from_points(&[
            CurvePoint::default(),
            CurvePoint {
                position: 0.2,
                value: peak,
                ..Default::default()
            },
            CurvePoint {
                position: 0.8,
                value: peak,
                ..Default::default()
            },
            CurvePoint {
                position: 1.0,
                ..Default::default()
            },
        ])
    }

    pub fn from_points(points: &[CurvePoint]) -> Self {
        assert!(
            !points.is_empty() && points.len() <= 8,
            "curves hold 1..=8 points"
        );
        assert!(
            points
                .windows(2)
                .all(|pair| pair[0].position <= pair[1].position),
            "curve points must be sorted"
        );
        assert!(points.iter().all(|point| {
            point.position.is_finite() && point.value.is_finite() && point.tension.is_finite()
        }));
        for point in points {
            if matches!(point.mode, CurveMode::Power) {
                assert_tension(point.tension);
            }
        }
        let mut curve = Self::default();
        curve.points[..points.len()].copy_from_slice(points);
        curve.count = points.len() as u8;
        curve
    }

    /// Evaluates authored values before configured range mapping.
    pub fn evaluate(self, t: f32) -> f32 {
        let count = self.count as usize;
        if count == 0 {
            return 0.0;
        }
        if count == 1 {
            return self.points[0].value;
        }
        let t = t.clamp(0.0, 1.0);
        if t <= self.points[0].position {
            return self.points[0].value;
        }
        let last = self.points[count - 1];
        if t >= last.position {
            return last.value;
        }
        for right_index in 1..count {
            let right = self.points[right_index];
            if t > right.position {
                continue;
            }
            let left = self.points[right_index - 1];
            let u = (t - left.position) / (right.position - left.position).max(0.0001);
            // Right-endpoint parameters support both ramp directions.
            let slope_sign = (right.value - left.value).signum();
            let blend = match right.mode {
                CurveMode::Linear => u,
                CurveMode::Hold => 0.0,
                CurveMode::Power => power_ease(u, right.tension * slope_sign),
            };
            return left.value + (right.value - left.value) * blend;
        }
        last.value
    }

    /// Clamps authored values, then maps them into the configured range.
    pub(crate) fn mapped(self, t: f32) -> f32 {
        self.range_min + self.evaluate(t).clamp(0.0, 1.0) * (self.range_max - self.range_min)
    }
}

/// Validates power-curve tension in the strict authored range `(-1, 1)`.
fn assert_tension(tension: f32) {
    assert!(
        tension.is_finite() && tension.abs() < 1.0,
        "curve tension {tension} must be finite with absolute value below 1"
    );
}

fn power_ease(t: f32, tension: f32) -> f32 {
    let exponent = 1.0 / (1.0 - tension.abs() * 0.999);
    if tension >= 0.0 {
        t.powf(exponent)
    } else {
        1.0 - (1.0 - t).powf(exponent)
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct GradientStop {
    pub time: f32,
    pub color: [f32; 4],
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u32)]
pub enum GradientInterpolation {
    #[default]
    Linear,
    Steps,
    Smoothstep,
}

/// Up to five HDR RGBA stops keyed on [0, 1]. Empty gradients are white.
#[derive(Clone, Copy, Debug)]
pub struct Gradient {
    pub stops: [GradientStop; 5],
    pub count: u8,
    pub interpolation: GradientInterpolation,
}

impl Default for Gradient {
    fn default() -> Self {
        Self {
            stops: [GradientStop::default(); 5],
            count: 0,
            interpolation: GradientInterpolation::Linear,
        }
    }
}

impl Gradient {
    pub fn two_stop(a: [f32; 4], b: [f32; 4]) -> Self {
        Self::from_stops(
            &[
                GradientStop {
                    time: 0.0,
                    color: a,
                },
                GradientStop {
                    time: 1.0,
                    color: b,
                },
            ],
            GradientInterpolation::Linear,
        )
    }

    pub fn from_stops(stops: &[GradientStop], interpolation: GradientInterpolation) -> Self {
        assert!(stops.len() <= 5, "gradients hold at most five stops");
        assert!(stops.iter().all(|stop| {
            (0.0..=1.0).contains(&stop.time)
                && stop.time.is_finite()
                && stop.color.iter().all(|value| value.is_finite())
        }));
        assert!(stops.windows(2).all(|pair| pair[0].time <= pair[1].time));
        let mut gradient = Self {
            interpolation,
            ..Default::default()
        };
        gradient.stops[..stops.len()].copy_from_slice(stops);
        gradient.count = stops.len() as u8;
        gradient
    }

    pub fn evaluate(self, t: f32) -> [f32; 4] {
        let count = self.count as usize;
        if count == 0 {
            return [1.0; 4];
        }
        if count == 1 {
            return self.stops[0].color;
        }
        let t = t.clamp(0.0, 1.0);
        if t <= self.stops[0].time {
            return self.stops[0].color;
        }
        if t >= self.stops[count - 1].time {
            return self.stops[count - 1].color;
        }
        for index in 1..count {
            let right = self.stops[index];
            if t > right.time {
                continue;
            }
            let left = self.stops[index - 1];
            let local_t = (t - left.time) / (right.time - left.time).max(0.0001);
            let blend = match self.interpolation {
                GradientInterpolation::Linear => local_t,
                GradientInterpolation::Steps => 0.0,
                GradientInterpolation::Smoothstep => local_t * local_t * (3.0 - 2.0 * local_t),
            };
            return lerp4(left.color, right.color, blend);
        }
        self.stops[count - 1].color
    }
}

fn lerp4(a: [f32; 4], b: [f32; 4], t: f32) -> [f32; 4] {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
        a[3] + (b[3] - a[3]) * t,
    ]
}

/// Host shape specification packed into [`ShapeGpu`].
#[derive(Clone, Copy, Debug, Default)]
pub enum Shape {
    #[default]
    Point,
    Sphere {
        radius: f32,
    },
    SphereSurface {
        radius: f32,
    },
    Box {
        half_extents: [f32; 3],
    },
    Cone {
        direction: [f32; 3],
        half_angle: f32,
        base_radius: f32,
    },
    Ring {
        inner: f32,
        outer: f32,
        height: f32,
    },
    /// Jagged polyline from local origin to `(0, 1, 0)`.
    /// Interior offsets hash the spawn seed; endpoints remain unjittered.
    /// `amplitude` is perpendicular displacement as a length fraction.
    Bolt {
        segments: u32,
        amplitude: f32,
    },
}

impl Shape {
    pub const fn point() -> Self {
        Self::Point
    }
    pub const fn sphere(radius: f32) -> Self {
        Self::Sphere { radius }
    }
    pub const fn sphere_surface(radius: f32) -> Self {
        Self::SphereSurface { radius }
    }
    pub const fn box_(half_extents: [f32; 3]) -> Self {
        Self::Box { half_extents }
    }
    pub const fn cone(direction: [f32; 3], half_angle: f32, base_radius: f32) -> Self {
        Self::Cone {
            direction,
            half_angle,
            base_radius,
        }
    }
    pub const fn ring(inner: f32, outer: f32, height: f32) -> Self {
        Self::Ring {
            inner,
            outer,
            height,
        }
    }
    pub const fn bolt(segments: u32, amplitude: f32) -> Self {
        Self::Bolt {
            segments,
            amplitude,
        }
    }

    pub(crate) fn pack(self) -> ShapeGpu {
        let mut gpu = ShapeGpu::default();
        match self {
            Self::Point => {}
            Self::Sphere { radius } => {
                assert!(radius.is_finite() && radius >= 0.0);
                gpu.shape_type = 1;
                gpu.params[0] = radius;
            }
            Self::SphereSurface { radius } => {
                assert!(radius.is_finite() && radius >= 0.0);
                gpu.shape_type = 2;
                gpu.params[0] = radius;
            }
            Self::Box { half_extents } => {
                assert!(
                    half_extents
                        .iter()
                        .all(|value| value.is_finite() && *value >= 0.0)
                );
                gpu.shape_type = 3;
                gpu.params[..3].copy_from_slice(&half_extents);
            }
            Self::Cone {
                direction,
                half_angle,
                base_radius,
            } => {
                assert!(direction.iter().all(|value| value.is_finite()));
                assert!(half_angle.is_finite() && half_angle >= 0.0);
                assert!(base_radius.is_finite() && base_radius >= 0.0);
                gpu.shape_type = 4;
                gpu.params[..3].copy_from_slice(&direction);
                gpu.params[3] = half_angle;
                gpu.params[4] = base_radius;
            }
            Self::Ring {
                inner,
                outer,
                height,
            } => {
                assert!(inner.is_finite() && outer.is_finite() && height.is_finite());
                assert!(inner >= 0.0 && outer >= inner && height >= 0.0);
                gpu.shape_type = 5;
                gpu.params[..3].copy_from_slice(&[inner, outer, height]);
            }
            Self::Bolt {
                segments,
                amplitude,
            } => {
                assert!(segments >= 1, "a bolt is at least one segment");
                assert!(amplitude.is_finite() && amplitude >= 0.0);
                gpu.shape_type = 6;
                gpu.params[0] = segments as f32;
                gpu.params[1] = amplitude;
            }
        }
        gpu
    }
}

/// Sparse emitter authoring resolved to visible runtime defaults.
#[derive(Clone, Copy, Debug, Default)]
pub struct EmitterSpec {
    pub lifetime_range: [f32; 2],
    pub one_shot: bool,
    pub explosiveness: f32,
    pub delay: f32,
    pub spawn_rate: f32,
    pub max_particles: u32,
    pub fixed_fps: u32,
    pub shape: Shape,
    pub offset: [f32; 3],
    pub emission_scale: [f32; 3],
    pub local_coords: bool,
    /// Prevents spawn tint from modifying this emitter's palette.
    pub ignore_tint: bool,
    pub direction: [f32; 3],
    pub spread_deg: f32,
    pub flatness: f32,
    pub speed_range: [f32; 2],
    pub gravity: [f32; 3],
    pub drag: f32,
    pub primitive: Primitive,
    pub alignment: Alignment,
    pub mesh_scale: [f32; 3],
    pub initial_rotation_deg: [f32; 3],
    pub scale_range: [f32; 2],
    pub angle_range_deg: [f32; 2],
    pub angular_velocity_range: [f32; 2],
    pub color_initial: Gradient,
    pub color_ramp: Gradient,
    pub hue_variation: [f32; 2],
    pub scale_curve: Curve,
    pub alpha_curve: Curve,
    pub emission_curve: Curve,
    pub damping_curve: Curve,
}

pub struct EffectSpec {
    pub name: &'static str,
    pub emitters: Vec<EmitterSpec>,
}

impl EffectSpec {
    pub fn new(name: &'static str, emitters: Vec<EmitterSpec>) -> Self {
        assert!(!name.is_empty());
        assert!(
            emitters.len() <= MAX_EFFECT_EMITTERS,
            "effects hold at most eight emitters"
        );
        Self { name, emitters }
    }
}

#[cfg(test)]
mod tests {
    use core::mem::size_of;

    use super::*;

    #[test]
    fn curve_constructors_follow_power_math() {
        let curve = Curve::ease_out(0.0, 1.0, 0.5);
        assert_eq!(curve.evaluate(0.0), 0.0);
        let exponent = 1.0 / (1.0 - 0.5 * 0.999);
        let expected = 1.0 - 0.5_f32.powf(exponent);
        assert!((curve.evaluate(0.5) - expected).abs() < 0.000_01);
        assert_eq!(curve.evaluate(1.0), 1.0);
        assert_eq!(Curve::fade_in_out(2.0).evaluate(0.5), 2.0);
    }

    #[test]
    fn gradient_modes_interpolate() {
        let gradient = Gradient::two_stop([0.0, 0.0, 0.0, 1.0], [2.0, 1.0, 0.0, 0.5]);
        assert_eq!(gradient.evaluate(0.5), [1.0, 0.5, 0.0, 0.75]);
        let steps = Gradient {
            interpolation: GradientInterpolation::Steps,
            ..gradient
        };
        assert_eq!(steps.evaluate(0.5), [0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn shape_packing_is_fixed_48_byte_abi() {
        let shape = Shape::ring(0.2, 0.5, 0.1).pack();
        assert_eq!(shape.shape_type, 5);
        assert_eq!(&shape.params[..3], &[0.2, 0.5, 0.1]);
        assert_eq!(size_of::<ShapeGpu>(), 48);
    }
}
