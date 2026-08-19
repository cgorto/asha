//! Octahedral codec for unit directions: 2 floats per normal, equal-area-ish,
//! identical math on both machines so encoded normals round-trip exactly.

use glam::{Vec2, Vec3};

/// Copy the sign of `y` onto `x` without branching.
fn mulsign(x: f32, y: f32) -> f32 {
    f32::from_bits((y.to_bits() & 0x8000_0000) ^ x.to_bits())
}

/// Octahedral encode of a unit direction into [−1, 1]².
pub fn oct_encode(dir: Vec3) -> Vec2 {
    let s = dir.truncate() / (dir.x.abs() + dir.y.abs() + dir.z.abs());
    if dir.z < 0.0 {
        Vec2::new(mulsign(1.0 - s.y.abs(), s.x), mulsign(1.0 - s.x.abs(), s.y))
    } else {
        s
    }
}

/// Octahedral decode back to a unit direction.
pub fn oct_decode(p: Vec2) -> Vec3 {
    let z = 1.0 - p.x.abs() - p.y.abs();
    let fold = (-z).clamp(0.0, 1.0);
    let n = Vec3::new(p.x - mulsign(fold, p.x), p.y - mulsign(fold, p.y), z);
    n.normalize()
}
