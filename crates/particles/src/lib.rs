//! Resolved particle specs, stable emitter slots, and binned primitive draws.

mod draw;
mod sim;
mod spec;
mod verify;

pub use draw::ParticleDrawPass;
pub use sim::{
    CURVE_ROWS, CURVE_SAMPLES, EffectHandle, MAX_EMITTERS, MAX_MATERIALS, MAX_PARTICLES,
    MAX_SHAPES, ParticleSimPass,
};
pub use spec::{
    Alignment, Curve, CurveMode, CurvePoint, EffectSpec, EmitterSpec, Gradient,
    GradientInterpolation, GradientStop, MAX_EFFECT_EMITTERS, PRIMITIVE_COUNT, ParticleView,
    Primitive, Shape,
};
pub use verify::{
    CUBE_SPAWNS, ParticleVerifySnapshot, QUAD_SPAWNS, VERIFY_RING, assert_particle_states,
    assert_verify_step, assert_visible_indices,
};
