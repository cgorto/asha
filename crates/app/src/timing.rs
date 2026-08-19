//! Fixed-timestep pacing utilities.

/// Result of one frame's fixed-timestep accounting.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FixedSteps {
    /// Number of fixed updates due this frame.
    pub steps: u32,
    /// Remaining time after consuming fixed updates.
    pub leftover: f32,
    /// Render interpolation factor in `[0, 1)`.
    pub alpha: f32,
}

/// Calculate fixed updates and interpolation for an accumulator.
pub fn fixed_steps_due(accumulator: f32, fixed_dt: f32, steps_max: u32) -> FixedSteps {
    assert!(fixed_dt > 0.0); // Required divisor and step size.
    assert!(steps_max > 0); // At least one update must be allowed.
    assert!(accumulator >= 0.0); // Accumulated time cannot be negative.

    let steps = ((accumulator / fixed_dt) as u32).min(steps_max);
    let leftover = if steps < steps_max {
        accumulator - steps as f32 * fixed_dt
    } else {
        0.0
    };
    let alpha = leftover / fixed_dt;

    debug_assert!(steps <= steps_max);
    debug_assert!(leftover >= 0.0);
    debug_assert!((0.0..1.0).contains(&alpha));
    FixedSteps {
        steps,
        leftover,
        alpha,
    }
}

/// Measures frame deltas and tracks fixed-step time.
#[derive(Debug, Default)]
pub struct FrameClock {
    prev: Option<std::time::Instant>,
    accumulator: f32,
}

impl FrameClock {
    pub fn new() -> Self {
        Self::default()
    }

    /// Return elapsed seconds since the previous tick.
    pub fn tick(&mut self) -> f32 {
        let now = std::time::Instant::now();
        let raw = match self.prev {
            Some(prev) => now.duration_since(prev).as_secs_f32(),
            None => 1.0 / 60.0,
        };
        self.prev = Some(now);
        raw
    }

    /// Reset timing after a long non-rendering interruption.
    pub fn reset(&mut self) {
        self.prev = None;
        self.accumulator = 0.0;
    }

    /// Advance fixed-step time by a frame delta.
    /// Paused frames discard accumulated time.
    pub fn advance(
        &mut self,
        game_dt: f32,
        paused: bool,
        fixed_dt: f32,
        steps_max: u32,
    ) -> FixedSteps {
        if paused {
            self.accumulator = 0.0;
            return FixedSteps {
                steps: 0,
                leftover: 0.0,
                alpha: 0.0,
            };
        }
        self.accumulator += game_dt;
        let due = fixed_steps_due(self.accumulator, fixed_dt, steps_max);
        self.accumulator = due.leftover;
        due
    }
}

/// Clamp a raw delta to the configured maximum.
pub fn dt_clamped(raw_dt: f32, dt_max: f32) -> f32 {
    raw_dt.min(dt_max)
}

#[cfg(test)]
mod tests {
    use super::*;

    const DT: f32 = 1.0 / 120.0;

    #[test]
    fn zero_accumulator_owes_nothing() {
        let r = fixed_steps_due(0.0, DT, 8);
        assert_eq!(
            r,
            FixedSteps {
                steps: 0,
                leftover: 0.0,
                alpha: 0.0
            }
        );
    }

    #[test]
    fn one_step_exact() {
        let r = fixed_steps_due(DT, DT, 8);
        assert_eq!(r.steps, 1);
        assert!(r.leftover < DT);
    }

    #[test]
    fn partial_step_carries_alpha() {
        let r = fixed_steps_due(DT * 0.5, DT, 8);
        assert_eq!(r.steps, 0);
        assert!((r.alpha - 0.5).abs() < 1e-5);
    }

    #[test]
    fn cap_drops_residual() {
        // Cap updates and discard excess time.
        let r = fixed_steps_due(10.0, DT, 8);
        assert_eq!(r.steps, 8);
        assert_eq!(r.leftover, 0.0);
        assert_eq!(r.alpha, 0.0);
    }

    #[test]
    fn below_cap_keeps_remainder() {
        let r = fixed_steps_due(DT * 3.25, DT, 8);
        assert_eq!(r.steps, 3);
        assert!((r.alpha - 0.25).abs() < 1e-4);
    }

    #[test]
    fn paused_frames_zero_the_accumulator() {
        let mut clock = FrameClock::new();
        let r = clock.advance(0.1, true, DT, 8);
        assert_eq!(r.steps, 0);
        let r = clock.advance(DT, false, DT, 8);
        assert_eq!(r.steps, 1); // Paused time must not carry over.
    }
}
