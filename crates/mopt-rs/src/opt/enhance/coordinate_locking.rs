//! Coordinate locking: the step components of selected coordinates are set to zero, so that
//! the optimizer works on a subset of the coordinates at a time.
//!
//! This is block coordinate descent with a greedy choice of the block: the coordinates with
//! the largest gradient components move, those with small ones wait (the Gauss-Southwell rule;
//! Nutini et al., Proc. ICML 2015, "Coordinate descent converges faster with the
//! Gauss-Southwell rule than random selection"). Modes, as in MultiOptPy:
//! - adaptive (the default): half of the coordinates, those with the lowest gradient activity,
//!   start locked. Every second iteration a coordinate is locked when its gradient component is
//!   below the median of the last three gradients divided by 2 (0.05 until three gradients are
//!   known) and unlocked when it is above, but not within three iterations of its last change;
//! - scheduled: the locked fraction goes from 0.5 to 0 over ten iterations (gradually, in five
//!   stages, or oscillating), locking the least and unlocking the most active coordinates;
//! - manual: a mask given by the caller.
//!
//! Coordinates whose gradients are correlated (above 0.7 over the last five iterations) can be
//! locked and unlocked together (`use_groups`). The method needs no sign convention, and the
//! port follows MultiOptPy's rules step by step.
//!
//! MultiOptPy: `Optimizer/coordinate_locking.py:CoordinateLocking`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::collections::VecDeque;

use faer::Col;

use super::StepEnhancer;
use crate::error::{MoptError, MoptResult};
use crate::opt::step::StepInput;

/// How the locked fraction changes in the scheduled mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LockSchedule {
    /// Linear in the iteration number.
    Gradual,
    /// In five equal stages.
    Stepwise,
    /// Linear envelope modulated by two cosine periods.
    Oscillating,
}

/// How the locked coordinates are chosen.
#[derive(Debug, Clone, PartialEq)]
pub enum LockMode {
    /// By the size of the gradient components, with hysteresis.
    Adaptive,
    /// A locked fraction following a schedule.
    Scheduled(LockSchedule),
    /// A fixed mask (`true` = locked).
    Manual(Vec<bool>),
}

/// Which quantity decides in the adaptive mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LockCriterion {
    /// Small gradient component.
    Gradient,
    /// Small step component.
    Displacement,
    /// Both.
    Both,
}

/// Parameters of coordinate locking.
#[derive(Debug, Clone, PartialEq)]
pub struct CoordinateLockingParams {
    /// Mode.
    pub mode: LockMode,
    /// Fraction locked at the start.
    pub lock_fraction_initial: f64,
    /// Fraction locked at the end of the schedule.
    pub lock_fraction_final: f64,
    /// Length of the schedule in iterations.
    pub lock_transition_steps: usize,
    /// Criterion of the adaptive mode.
    pub criterion: LockCriterion,
    /// Gradient threshold until three gradients are known.
    pub gradient_threshold: f64,
    /// Step threshold of the displacement criterion.
    pub displacement_threshold: f64,
    /// The adaptive gradient threshold is the median divided by this factor.
    pub gradient_threshold_factor: f64,
    /// Adaptive updates happen every this many iterations.
    pub update_frequency: usize,
    /// Iterations a coordinate keeps its state after a change.
    pub lock_inertia: usize,
    /// Largest fraction locked by a schedule.
    pub lock_max_fraction: f64,
    /// Lock and unlock correlated coordinates together.
    pub use_groups: bool,
    /// Correlation above which coordinates form a group.
    pub group_coupling_threshold: f64,
}

impl Default for CoordinateLockingParams {
    fn default() -> Self {
        Self {
            mode: LockMode::Adaptive,
            lock_fraction_initial: 0.5,
            lock_fraction_final: 0.0,
            lock_transition_steps: 10,
            criterion: LockCriterion::Gradient,
            gradient_threshold: 0.05,
            displacement_threshold: 0.03,
            gradient_threshold_factor: 2.0,
            update_frequency: 2,
            lock_inertia: 3,
            lock_max_fraction: 0.8,
            use_groups: false,
            group_coupling_threshold: 0.7,
        }
    }
}

/// Pearson correlation matrix of the columns of `rows` (one row per observation); undefined
/// entries (constant columns) are zero. Used to group coordinates by their gradient history.
pub fn column_correlations(rows: &[Col<f64>]) -> Vec<Vec<f64>> {
    let n = rows.first().map_or(0, Col::nrows);
    let m = rows.len() as f64;
    let mean: Vec<f64> = (0..n)
        .map(|i| rows.iter().map(|r| r[i]).sum::<f64>() / m)
        .collect();
    let dev: Vec<f64> = (0..n)
        .map(|i| libm::sqrt(rows.iter().map(|r| (r[i] - mean[i]).powi(2)).sum::<f64>()))
        .collect();
    (0..n)
        .map(|i| {
            (0..n)
                .map(|j| {
                    let cov: f64 = rows
                        .iter()
                        .map(|r| (r[i] - mean[i]) * (r[j] - mean[j]))
                        .sum();
                    let c = cov / (dev[i] * dev[j]);
                    if c.is_finite() {
                        c
                    } else {
                        0.0
                    }
                })
                .collect()
        })
        .collect()
}

/// Groups of coordinates whose correlation with the first member exceeds `threshold`.
pub fn correlated_groups(correlation: &[Vec<f64>], threshold: f64) -> Vec<Vec<usize>> {
    let n = correlation.len();
    let mut visited = vec![false; n];
    let mut groups = Vec::new();
    for i in 0..n {
        if visited[i] {
            continue;
        }
        let mut group = vec![i];
        group.extend(
            (0..n).filter(|&j| j != i && !visited[j] && correlation[i][j].abs() > threshold),
        );
        if group.len() > 1 {
            for &j in &group {
                visited[j] = true;
            }
            groups.push(group);
        }
    }
    groups
}

/// Coordinate-locking step enhancer.
#[derive(Debug, Clone)]
pub struct CoordinateLocking {
    params: CoordinateLockingParams,
    mask: Option<Vec<bool>>,
    /// Iteration of the last change of each coordinate.
    changed: Vec<usize>,
    gradients: VecDeque<Col<f64>>,
    fraction: f64,
    groups: Vec<Vec<usize>>,
    iteration: usize,
}

impl CoordinateLocking {
    /// New enhancer.
    pub fn new(params: CoordinateLockingParams) -> Self {
        let fraction = params.lock_fraction_initial;
        Self {
            params,
            mask: None,
            changed: Vec::new(),
            gradients: VecDeque::new(),
            fraction,
            groups: Vec::new(),
            iteration: 0,
        }
    }

    /// Current mask (`true` = locked), once the first step has been seen.
    pub fn mask(&self) -> Option<&[bool]> {
        self.mask.as_deref()
    }

    /// Locked fraction of the schedule at an iteration.
    pub fn scheduled_fraction(&self, schedule: LockSchedule, iteration: usize) -> f64 {
        let p = &self.params;
        let (a, b) = (p.lock_fraction_initial, p.lock_fraction_final);
        let steps = p.lock_transition_steps.max(1) as f64;
        let it = iteration as f64;
        match schedule {
            LockSchedule::Gradual => a + (it / steps).min(1.0) * (b - a),
            LockSchedule::Stepwise => {
                let stage = libm::floor(it / (steps / 5.0));
                a + (stage / 5.0).min(1.0) * (b - a)
            }
            LockSchedule::Oscillating => {
                if it > steps {
                    b
                } else {
                    let progress = it / steps;
                    let wave = 0.5 + 0.5 * libm::cos(4.0 * std::f64::consts::PI * progress);
                    b + wave * (1.0 - progress) * (a - b)
                }
            }
        }
    }

    /// Sum of the absolute gradient components over the last five gradients.
    fn activity(&self, n: usize) -> Vec<f64> {
        let mut act = vec![0.0; n];
        for g in self.gradients.iter().rev().take(5) {
            for i in 0..n {
                act[i] += g[i].abs();
            }
        }
        act
    }

    /// Lock the least active or unlock the most active coordinates to reach the fraction.
    fn apply_fraction(&mut self, mask: &mut [bool]) {
        let n = mask.len();
        let limit = self.fraction.min(self.params.lock_max_fraction).max(0.0);
        let target = (n as f64 * limit) as usize;
        let locked = mask.iter().filter(|m| **m).count();
        let act = self.activity(n);
        if target > locked {
            let mut free: Vec<usize> = (0..n).filter(|&i| !mask[i]).collect();
            free.sort_by(|&a, &b| act[a].total_cmp(&act[b]));
            for &i in free.iter().take(target - locked) {
                mask[i] = true;
                self.changed[i] = self.iteration;
            }
        } else if target < locked {
            let mut held: Vec<usize> = (0..n).filter(|&i| mask[i]).collect();
            held.sort_by(|&a, &b| act[b].total_cmp(&act[a]));
            for &i in held.iter().take(locked - target) {
                mask[i] = false;
                self.changed[i] = self.iteration;
            }
        }
    }

    fn adaptive_update(&mut self, mask: &mut [bool], gradient: &Col<f64>, step: &Col<f64>) {
        let p = &self.params;
        if !self.iteration.is_multiple_of(p.update_frequency.max(1)) {
            return;
        }
        let threshold = if self.gradients.len() > 2 {
            let mut all: Vec<f64> = self
                .gradients
                .iter()
                .rev()
                .take(3)
                .flat_map(|g| g.iter().map(|v| v.abs()))
                .collect();
            all.sort_by(f64::total_cmp);
            let k = all.len();
            let median = if k.is_multiple_of(2) {
                0.5 * (all[k / 2 - 1] + all[k / 2])
            } else {
                all[k / 2]
            };
            median / p.gradient_threshold_factor
        } else {
            p.gradient_threshold
        };
        for i in 0..mask.len() {
            let low_gradient = gradient[i].abs() < threshold;
            let low_move = step[i].abs() < p.displacement_threshold;
            let quiet = match p.criterion {
                LockCriterion::Gradient => low_gradient,
                LockCriterion::Displacement => low_move,
                LockCriterion::Both => low_gradient && low_move,
            };
            if quiet != mask[i] && self.iteration - self.changed[i] > p.lock_inertia {
                mask[i] = quiet;
                self.changed[i] = self.iteration;
            }
        }
    }

    fn group_update(&mut self, mask: &mut [bool]) {
        if self.groups.is_empty() && self.gradients.len() >= 5 {
            let rows: Vec<Col<f64>> = self.gradients.iter().rev().take(5).rev().cloned().collect();
            self.groups = correlated_groups(
                &column_correlations(&rows),
                self.params.group_coupling_threshold,
            );
        }
        for group in &self.groups {
            let locked = group.iter().filter(|&&i| mask[i]).count();
            if 2 * locked > group.len() {
                group.iter().for_each(|&i| mask[i] = true);
            }
            let free = group.iter().filter(|&&i| !mask[i]).count();
            if 2 * free > group.len() {
                group.iter().for_each(|&i| mask[i] = false);
            }
        }
    }
}

impl Default for CoordinateLocking {
    fn default() -> Self {
        Self::new(CoordinateLockingParams::default())
    }
}

impl StepEnhancer for CoordinateLocking {
    fn name(&self) -> &str {
        "coordinate_locking"
    }

    fn enhance(&mut self, input: &StepInput<'_>, step: Col<f64>) -> MoptResult<Col<f64>> {
        let n = step.nrows();
        self.gradients.push_back(input.gradient.clone());
        if self.gradients.len() > 10 {
            self.gradients.pop_front();
        }
        let mode = self.params.mode.clone();
        let mut mask = match self.mask.take() {
            Some(m) if m.len() == n => m,
            _ => {
                self.changed = vec![0; n];
                self.fraction = self.params.lock_fraction_initial;
                let mut m = vec![false; n];
                match &mode {
                    LockMode::Manual(given) => {
                        if given.len() != n {
                            return Err(MoptError::config(format!(
                                "the lock mask has {} entries, the step {n}",
                                given.len()
                            )));
                        }
                        m.clone_from(given);
                    }
                    _ => self.apply_fraction(&mut m),
                }
                m
            }
        };
        match mode {
            LockMode::Scheduled(schedule) => {
                self.fraction = self.scheduled_fraction(schedule, self.iteration);
                self.apply_fraction(&mut mask);
            }
            LockMode::Adaptive => self.adaptive_update(&mut mask, input.gradient, &step),
            LockMode::Manual(_) => {}
        }
        if self.params.use_groups {
            self.group_update(&mut mask);
        }
        let out = Col::from_fn(n, |i| if mask[i] { 0.0 } else { step[i] });
        self.mask = Some(mask);
        self.iteration += 1;
        Ok(out)
    }

    fn reset(&mut self) {
        *self = Self::new(self.params.clone());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::opt::stepper::plain_input;
    use faer::Mat;

    fn run(cl: &mut CoordinateLocking, g: &[f64], it: usize) -> Vec<bool> {
        let n = g.len();
        let x = Col::zeros(n);
        let h = Mat::<f64>::identity(n, n);
        let g = Col::from_fn(n, |i| g[i]);
        let s = Col::from_fn(n, |_| 1.0);
        let out = cl
            .enhance(&plain_input(it, &x, 0.0, &g, &h, None, 0), s)
            .unwrap();
        (0..n).map(|i| out[i] == 0.0).collect()
    }

    #[test]
    fn the_least_active_half_starts_locked_and_large_gradients_unlock_after_the_inertia() {
        let mut cl = CoordinateLocking::default();
        let g = [0.5, 0.01, 0.3, 0.02];
        assert_eq!(run(&mut cl, &g, 0), vec![false, true, false, true]);
        // Coordinate 1 becomes the most active one; nothing changes before iteration 4.
        let g2 = [0.5, 0.9, 0.3, 0.02];
        for it in 1..4 {
            assert_eq!(
                run(&mut cl, &g2, it),
                vec![false, true, false, true],
                "{it}"
            );
        }
        // Median of the last three gradients: (0.3 + 0.5) / 2 = 0.4, threshold 0.2.
        assert_eq!(run(&mut cl, &g2, 4), vec![false, false, false, true]);
        // A coordinate falling below the threshold is locked at the next allowed update.
        let g3 = [0.5, 0.9, 0.05, 0.02];
        assert_eq!(run(&mut cl, &g3, 5), vec![false, false, false, true]);
        assert_eq!(run(&mut cl, &g3, 6), vec![false, false, true, true]);
    }

    #[test]
    fn schedules_manual_masks_and_groups() {
        let cl = CoordinateLocking::default();
        assert!((cl.scheduled_fraction(LockSchedule::Gradual, 5) - 0.25).abs() < 1e-15);
        assert_eq!(cl.scheduled_fraction(LockSchedule::Gradual, 50), 0.0);
        assert!((cl.scheduled_fraction(LockSchedule::Stepwise, 5) - 0.3).abs() < 1e-15);
        assert!((cl.scheduled_fraction(LockSchedule::Oscillating, 0) - 0.5).abs() < 1e-15);
        assert_eq!(cl.scheduled_fraction(LockSchedule::Oscillating, 11), 0.0);
        let mut sched = CoordinateLocking::new(CoordinateLockingParams {
            mode: LockMode::Scheduled(LockSchedule::Gradual),
            ..CoordinateLockingParams::default()
        });
        let g = [0.5, 0.01, 0.3, 0.02];
        assert_eq!(run(&mut sched, &g, 0).iter().filter(|m| **m).count(), 2);
        for it in 1..10 {
            run(&mut sched, &g, it);
        }
        assert_eq!(run(&mut sched, &g, 10).iter().filter(|m| **m).count(), 0);
        let mut manual = CoordinateLocking::new(CoordinateLockingParams {
            mode: LockMode::Manual(vec![true, false, false, true]),
            ..CoordinateLockingParams::default()
        });
        assert_eq!(run(&mut manual, &g, 0), vec![true, false, false, true]);
        let mut wrong = CoordinateLocking::new(CoordinateLockingParams {
            mode: LockMode::Manual(vec![true]),
            ..CoordinateLockingParams::default()
        });
        let x = Col::zeros(2);
        let h = Mat::<f64>::identity(2, 2);
        assert!(wrong
            .enhance(&plain_input(0, &x, 0.0, &x, &h, None, 0), Col::zeros(2))
            .is_err());
        // Coordinates 0 and 1 move together, 2 against them, 3 is constant.
        let rows: Vec<Col<f64>> = [1.0, 2.0, 4.0, 3.0]
            .iter()
            .map(|t| Col::from_fn(4, |i| [*t, 2.0 * t + 1.0, -t, 5.0][i]))
            .collect();
        let corr = column_correlations(&rows);
        assert!((corr[0][1] - 1.0).abs() < 1e-12 && (corr[0][2] + 1.0).abs() < 1e-12);
        assert_eq!(corr[0][3], 0.0);
        assert_eq!(correlated_groups(&corr, 0.7), vec![vec![0, 1, 2]]);
    }
}
