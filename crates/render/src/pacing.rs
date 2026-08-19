//! Fixed-step pacing with bounded catch-up and explicit pause behavior.
//!
//! Virtual delta is capped at `steps_max * timestep`. Before Bevy's fixed
//! driver runs, excess accumulated time is discarded, so a long stall costs at
//! most `steps_max` steps and cannot create a death spiral. Pausing discards
//! the accumulator, preventing banked time from replaying on resume.

use std::time::Duration;

use bevy::app::RunFixedMainLoopSystems;
use bevy::prelude::*;

pub struct PacingPlugin {
    pub fixed_hz: f64,
    pub steps_max: u32,
}

/// Defaults to 120 Hz with eight catch-up steps.
impl Default for PacingPlugin {
    fn default() -> Self {
        Self {
            fixed_hz: 120.0,
            steps_max: 8,
        }
    }
}

#[derive(Resource, Clone, Copy)]
struct Pacing {
    steps_max: u32,
}

impl Plugin for PacingPlugin {
    fn build(&self, app: &mut App) {
        assert!(self.fixed_hz > 0.0 && self.steps_max > 0);
        let timestep = Duration::from_secs_f64(1.0 / self.fixed_hz);
        app.insert_resource(Time::<Fixed>::from_duration(timestep))
            .insert_resource(Pacing {
                steps_max: self.steps_max,
            })
            .add_systems(Startup, clamp_virtual_delta)
            .add_systems(
                bevy::app::RunFixedMainLoop,
                residual_drop.before(RunFixedMainLoopSystems::FixedMainLoop),
            );
    }
}

fn clamp_virtual_delta(
    mut virt: ResMut<Time<Virtual>>,
    fixed: Res<Time<Fixed>>,
    pacing: Res<Pacing>,
) {
    virt.set_max_delta(fixed.timestep() * pacing.steps_max);
}

/// Drops excess accumulated time before Bevy's fixed driver runs.
fn residual_drop(mut fixed: ResMut<Time<Fixed>>, virt: Res<Time<Virtual>>, pacing: Res<Pacing>) {
    if virt.is_paused() {
        let all = fixed.overstep();
        fixed.discard_overstep(all);
        return;
    }
    let cap = fixed.timestep() * pacing.steps_max;
    let pending = fixed.overstep() + virt.delta();
    if pending > cap {
        fixed.discard_overstep(pending - cap);
    }
}
