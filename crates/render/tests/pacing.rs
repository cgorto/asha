//! Cross-checks fixed-step counts, catch-up caps, and pause semantics.
//! The Bevy pacing graft is compared with the reference frame clock.

use std::time::Duration;

use bevy::prelude::*;
use bevy::time::TimeUpdateStrategy;
use render::PacingPlugin;

const FIXED_HZ: f64 = 60.0;
const FIXED_DT: f32 = 1.0 / 60.0;
const STEPS_MAX: u32 = 4;

#[derive(Resource, Default)]
struct StepCount(u32);

fn count_steps(mut count: ResMut<StepCount>) {
    count.0 += 1;
}

fn pump(app: &mut App, dt: f64) -> u32 {
    app.world_mut()
        .insert_resource(TimeUpdateStrategy::ManualDuration(Duration::from_secs_f64(
            dt,
        )));
    let before = app.world().resource::<StepCount>().0;
    app.update();
    app.world().resource::<StepCount>().0 - before
}

#[test]
fn graft_matches_fixed_steps_due() {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins)
        .add_plugins(PacingPlugin {
            fixed_hz: FIXED_HZ,
            steps_max: STEPS_MAX,
        })
        .init_resource::<StepCount>()
        .add_systems(FixedUpdate, count_steps);

    assert_eq!(pump(&mut app, 0.0), 0);

    let frames = [
        0.014, 0.02, 0.037, 0.052, 0.11, 0.9, 0.014, 0.007, 0.3, 0.02, 0.014,
    ];

    let mut accumulator = 0.0f32;
    for (i, &dt) in frames.iter().enumerate() {
        let ran = pump(&mut app, dt);

        accumulator += dt as f32;
        let due = app::fixed_steps_due(accumulator, FIXED_DT, STEPS_MAX);
        accumulator = due.leftover;

        assert_eq!(ran, due.steps, "step count diverged at frame {i} (dt={dt})");
        let alpha = app.world().resource::<Time<Fixed>>().overstep_fraction();
        assert!(
            (alpha - due.alpha).abs() < 1e-3,
            "alpha diverged at frame {i} (dt={dt}): graft {alpha}, reference {}",
            due.alpha,
        );
    }
}

#[test]
fn pause_zeroes_the_accumulator() {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins)
        .add_plugins(PacingPlugin {
            fixed_hz: FIXED_HZ,
            steps_max: STEPS_MAX,
        })
        .init_resource::<StepCount>()
        .add_systems(FixedUpdate, count_steps);
    pump(&mut app, 0.0);

    assert_eq!(pump(&mut app, 0.014), 0);
    app.world_mut().resource_mut::<Time<Virtual>>().pause();
    assert_eq!(pump(&mut app, 0.25), 0, "paused frames must not step");
    app.world_mut().resource_mut::<Time<Virtual>>().unpause();
    assert_eq!(
        pump(&mut app, 0.014),
        0,
        "banked pre-pause time must be gone"
    );
    assert_eq!(pump(&mut app, 0.014), 1, "normal stepping resumes");
}
