//! The step MultiOptPy takes along the forces of a chain under the name FIRE.
//!
//! It is not the FIRE of Bitzek et al.: no velocity is carried from one iteration to the
//! next (MultiOptPy's "velocity" is the force itself). With the forces `F` of this
//! iteration and `F_prev` of the last one, and `P = sum F_prev . F` over the chain:
//!
//! - first iteration, or `P <= 0`: `dt <- dt / 2`, `a <- 0.1`, the count of good steps is
//!   reset, and `v = dt F`;
//! - `P > 0`: after more than five good steps in a row `dt <- min(1.1 dt, 1)` and
//!   `a <- 1.1 a`; `v = (1 + dt) F`;
//!
//! the displacement is `dt (v + F_prev)` (`dt v` in the first iteration). So a step is a
//! multiple of the force plus a share of the last force, with a time step that halves
//! whenever the forces turn against the last ones. The mixing parameter `a` has no effect
//! (the "velocity" it would turn towards the force is the force).
//!
//! Left to itself the time step grows by a tenth per iteration until the steps overshoot,
//! is halved, and grows again: the forces of a chain that is almost relaxed then rise by
//! orders of magnitude every few dozen iterations. [`ChainFire::observe_curvature`] stops
//! this. With steady forces a step is `dt (2 + dt) F`, so an error along a direction of
//! curvature `k` is multiplied by `1 - dt (2 + dt) k` per iteration; the largest curvature
//! met along the last steps, `k = max_i -(F_i - F_prev,i) . d_i / |d_i|^2`, bounds the time
//! step by `dt (2 + dt) k <= s` (`s`: [`ChainFireSettings::curvature_safety`]). The force
//! of an image also changes when its neighbours move (its direction along the chain
//! turns), so an image that hardly moved says nothing about the curvature: the maximum
//! is taken over the images that moved at least half as far as the one that moved
//! farthest, together with the quotient of the whole chain,
//! `-sum_i (F_i - F_prev,i) . d_i / sum_i |d_i|^2`. MultiOptPy has no such bound; it is
//! not applied under `Compatibility::MultiOptPy`.
//!
//! MultiOptPy: `Optimizer/fire_neb.py:FIREOptimizer`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Col;

use crate::math::linalg::col_dot;

/// Settings of [`ChainFire`] (MultiOptPy's values).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChainFireSettings {
    /// Time step at the start.
    pub dt: f64,
    /// Largest time step.
    pub dt_max: f64,
    /// Factor on the time step after enough good steps.
    pub increase: f64,
    /// Factor on the time step after a step against the last forces.
    pub decrease: f64,
    /// Good steps in a row before the time step grows.
    pub patience: usize,
    /// `s` of the bound `dt (2 + dt) k <= s` on the time step for the largest curvature
    /// `k` that was observed (1: the stiffest direction is relaxed in one step; 2 is the
    /// limit of stability).
    pub curvature_safety: f64,
}

impl Default for ChainFireSettings {
    fn default() -> Self {
        Self {
            dt: 0.5,
            dt_max: 1.0,
            increase: 1.10,
            decrease: 0.5,
            patience: 5,
            curvature_safety: 1.0,
        }
    }
}

/// Share of the largest curvature observed so far that is kept from one iteration to the
/// next: the bound loosens slowly where the chain has left a stiff region.
const CURVATURE_MEMORY: f64 = 0.9;
/// Share of the longest step of the chain that the step of an image must reach for the
/// image to count in the curvature.
const MOVED_SHARE: f64 = 0.5;

/// The state of the step rule: the time step and the count of good steps.
#[derive(Debug, Clone, PartialEq)]
pub struct ChainFire {
    settings: ChainFireSettings,
    dt: f64,
    good_steps: usize,
    /// Largest curvature along the last steps (decaying), zero until one is observed.
    curvature: f64,
}

impl ChainFire {
    /// A rule at its first step.
    pub fn new(settings: ChainFireSettings) -> Self {
        Self {
            settings,
            dt: settings.dt,
            good_steps: 0,
            curvature: 0.0,
        }
    }

    /// The largest time step the observed curvature allows (the largest of the settings
    /// while none was observed).
    pub fn time_step_bound(&self) -> f64 {
        if self.curvature > 0.0 {
            let bound = (1.0 + self.settings.curvature_safety / self.curvature).sqrt() - 1.0;
            bound.min(self.settings.dt_max)
        } else {
            self.settings.dt_max
        }
    }

    /// Take note of the curvature along the last steps: `forces` now, `previous` forces,
    /// and the `steps` taken between them, image by image. The time step of the following
    /// displacements stays within [`Self::time_step_bound`].
    pub fn observe_curvature(
        &mut self,
        forces: &[Col<f64>],
        previous: &[Col<f64>],
        steps: &[Col<f64>],
    ) {
        let moved: Vec<f64> = steps.iter().map(|step| col_dot(step, step)).collect();
        let farthest = moved.iter().fold(0.0_f64, |m, v| m.max(*v));
        let (mut largest, mut drop, mut total) = (0.0_f64, 0.0, 0.0);
        for (((force, before), step), &length2) in
            forces.iter().zip(previous).zip(steps).zip(&moved)
        {
            let along = col_dot(before, step) - col_dot(force, step);
            drop += along;
            total += length2;
            if length2 > 1e-24 && length2 >= MOVED_SHARE * MOVED_SHARE * farthest {
                largest = largest.max(along / length2);
            }
        }
        if total > 1e-24 {
            largest = largest.max(drop / total);
        }
        self.curvature = largest.max(CURVATURE_MEMORY * self.curvature);
    }

    /// Current time step.
    pub fn time_step(&self) -> f64 {
        self.dt
    }

    /// Displacements along `forces`; `previous` are the forces of the last iteration
    /// (`None` in the first one, and after the chain was rebuilt).
    pub fn displacements(
        &mut self,
        forces: &[Col<f64>],
        previous: Option<&[Col<f64>]>,
    ) -> Vec<Col<f64>> {
        let previous = previous.filter(|p| p.len() == forces.len() && p.len() > 1);
        let overlap = previous.map_or(0.0, |p| {
            p.iter()
                .zip(forces)
                .map(|(a, b)| col_dot(a, b))
                .sum::<f64>()
        });
        let keep = if overlap > 0.0 {
            if self.good_steps > self.settings.patience {
                self.dt = (self.dt * self.settings.increase).min(self.settings.dt_max);
            }
            self.good_steps += 1;
            1.0
        } else {
            self.dt *= self.settings.decrease;
            self.good_steps = 0;
            0.0
        };
        self.dt = self.dt.min(self.time_step_bound());
        let dt = self.dt;
        forces
            .iter()
            .enumerate()
            .map(|(i, force)| {
                let velocity = (keep + dt) * force;
                match previous {
                    Some(p) => dt * (velocity + &p[i]),
                    None => dt * velocity,
                }
            })
            .collect()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn col(values: &[f64]) -> Col<f64> {
        Col::from_fn(values.len(), |k| values[k])
    }

    #[test]
    fn the_time_step_halves_against_the_last_forces_and_grows_after_good_steps() {
        let forces = vec![col(&[1.0, 0.0]), col(&[0.0, 2.0])];
        let mut rule = ChainFire::new(ChainFireSettings::default());
        // First step: dt = 0.25, displacement dt^2 F.
        let first = rule.displacements(&forces, None);
        assert_eq!(rule.time_step(), 0.25);
        assert_eq!(first[1], col(&[0.0, 0.125]));
        // Along the last forces: v = (1 + dt) F, displacement dt (v + F_prev).
        let second = rule.displacements(&forces, Some(&forces));
        assert_eq!(rule.time_step(), 0.25);
        assert!((second[0][0] - 0.25 * (1.25 + 1.0)).abs() < 1e-15);
        // Six good steps are needed before the time step grows.
        for _ in 0..5 {
            rule.displacements(&forces, Some(&forces));
        }
        assert_eq!(rule.time_step(), 0.25);
        rule.displacements(&forces, Some(&forces));
        assert!((rule.time_step() - 0.275).abs() < 1e-15);
        // Against the last forces: half the time step, and the last forces still count.
        let turned: Vec<Col<f64>> = forces.iter().map(|f| -f).collect();
        let back = rule.displacements(&turned, Some(&forces));
        assert!((rule.time_step() - 0.1375).abs() < 1e-15);
        let dt = rule.time_step();
        assert!((back[0][0] - dt * (-dt + 1.0)).abs() < 1e-15);
        // A curvature of 4 along the last steps: dt (2 + dt) 4 <= 1, dt <= sqrt(1.25) - 1.
        let steps = vec![col(&[0.1, 0.0]), col(&[0.0, 0.1])];
        let softer: Vec<Col<f64>> = forces
            .iter()
            .zip(&steps)
            .map(|(f, d)| f - 4.0 * d)
            .collect();
        let mut bounded = ChainFire::new(ChainFireSettings::default());
        assert_eq!(bounded.time_step_bound(), 1.0);
        bounded.observe_curvature(&softer, &forces, &steps);
        let bound = 1.25_f64.sqrt() - 1.0;
        assert!((bounded.time_step_bound() - bound).abs() < 1e-15);
        bounded.displacements(&softer, Some(&forces));
        assert!((bounded.time_step() - bound).abs() < 1e-15);
        // Flatter steps later loosen the bound slowly; steps of no length say nothing.
        bounded.observe_curvature(&forces, &forces, &steps);
        let looser = (1.0_f64 + 1.0 / 3.6).sqrt() - 1.0;
        assert!((bounded.time_step_bound() - looser).abs() < 1e-15);
        bounded.observe_curvature(&softer, &forces, &[col(&[0.0; 2]), col(&[0.0; 2])]);
        assert!(bounded.time_step_bound() > looser);
        // An image that hardly moved while its force changed (its neighbours moved) does
        // not count: the curvature is that of the image that moved, 4.
        let mut uneven = ChainFire::new(ChainFireSettings::default());
        let steps = vec![col(&[0.1, 0.0]), col(&[0.0, 1e-9])];
        let after = vec![&forces[0] - 4.0 * &steps[0], &forces[1] - col(&[0.0, 1e-3])];
        uneven.observe_curvature(&after, &forces, &steps);
        assert!((uneven.time_step_bound() - bound).abs() < 1e-6);
        // A history of another length is no history.
        let alone = rule.displacements(&forces, Some(&forces[..1]));
        assert!((alone[0][0] - rule.time_step().powi(2)).abs() < 1e-15);
    }
}
