//! The step control of a walk: steps that failed are taken again, and a side goes on to
//! its minimum.
//!
//! Three things MultiOptPy's integrators do not have, for the integrators that use them
//! (the measurements behind them are in `docs/source/irc.md`, "Steps that fail, and where
//! a side ends"):
//!
//! - A step that failed is taken again (`Walk::arrive_or_retry`). Down a path of
//!   steepest descent the energy falls, and the gradient at the end of a step does not
//!   point back along it. A point of higher energy than the one before it is off the
//!   path (met with LQA where the surface is flat and bends: CH3NC to CH3CN rose by 1.2
//!   kcal/mol at one point), and so is a point at which the gradient points back along
//!   the step: the step went past the floor of the valley, and the next one would turn
//!   back. Such a point is dropped, and the step is taken from the last point again at
//!   half its length (with exact Hessians along the path, first at the same length with
//!   the Hessian of that point); the length comes back by doubling after every point
//!   that is kept. A step that raises the energy is halved up to ten times, then the
//!   side ends at the last point; a step that only overshot is halved three times, then
//!   its point is kept.
//! - Zigzag. A step that turns back by more than a right angle against the step before
//!   it halves the length of the next steps; the length comes back as above.
//! - The final approach (`Walk::approach`). MultiOptPy ends a side at the first point
//!   whose gradient is below the thresholds, which lies before the minimum (an
//!   optimization of such an end point with tight thresholds took 20 to 80 iterations
//!   for HCN, CH3NC, and allyl bromide), and on a flat surface far before it: with a
//!   barrier of 1e-4 Hartree the gradient is below the thresholds all the way. Here the
//!   side goes on once the gradient is below the thresholds and the path lies in the bowl
//!   of the minimum (the model has no direction of negative curvature left and less than
//!   a tenth of the energy to lose that the side has lost since its start), until the
//!   gradient is below the tight thresholds of an optimization. Its steps are those of LQA (the
//!   steepest-descent path of the quadratic model, at most a step of the path long, which
//!   ends in the minimum of the model when that is nearer) with the magnitude of the
//!   curvature along every mode, so that they go down whatever the sign the model has
//!   there. A step to the minimum of the model in one go (Newton) reached the same end
//!   points but left sharper turns in the path.
//!
//! With `-irc_opt bend=SHARE` the step length is limited by the bend of the valley as
//! well (`Walk::step_length`).
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::{Col, Mat};

use super::convergence::convergence_check;
use super::integrator::{Arrival, BranchAction, Walk};
use super::lqa::lqa_step_in_modes;
use super::types::{dot, norm, BranchEnd, PointRequest, PointRole, PointValue};
use crate::error::MoptResult;
use crate::math::linalg::symmetric_eigen;

/// What became of the point that was evaluated (`Walk::arrive_or_retry`).
pub(crate) enum Arrived {
    /// It is the next point of the path.
    Point(Arrival),
    /// Its energy rose: it is dropped, and the step is to be taken again from the point in
    /// here (the last point of the path) with [`Walk::step_length`].
    Again(Arrival),
    /// Nothing to do: the walk asked for what it needs, or ended the side.
    Wait,
}

/// What the final approach says at a point of the path (`Walk::approach`).
pub(crate) enum Approach {
    /// Not there yet: take a step of the path.
    Path,
    /// Go to this point next.
    Step(Col<f64>),
    /// The side has arrived (the walk ended it).
    Arrived,
}

/// Rise of the energy from one point to the next that is not rounding.
const ENERGY_RISE: f64 = 1e-10;
/// Times a step is halved before the side ends where it is.
const MOST_HALVINGS: usize = 10;
/// Curvature of the model above minus this: no direction of negative curvature.
const FLAT_MODE: f64 = 1e-8;
/// The path is in the bowl of its minimum when the energy the model has left to lose is
/// below this share of the energy the side has lost.
const NEAR_THE_MINIMUM: f64 = 0.1;
/// Times in a row a step is taken again because it overshot (then its point is kept).
const MOST_OVERSHOOTS: usize = 3;
/// Angle (degrees) at a point between the points before and after it below which the
/// path has turned back (a straight path has 180).
const ZIGZAG_ANGLE: f64 = 90.0;
/// Smallest factor on the step length.
const SHORTEST_SCALE: f64 = 1.0 / 32.0;

impl Walk {
    /// The length of the next step of the path: the step size, shorter after steps that
    /// raised the energy.
    pub fn step_length(&self) -> f64 {
        let length = self.settings.step_size * self.scale;
        match self.bend_limit {
            Some(limit) if !self.approaching => length.min(limit),
            _ => length,
        }
    }

    /// The longest step from the point of `arrival` that does not lose more than the
    /// share `eta` of the energy it gains by cutting the bend of the valley. A step of
    /// the length `s` along the tangent leaves a path of curvature `kappa` by
    /// `kappa s^2 / 2`, which costs `s^4 kappa.H.kappa / 8` (the curvature vector couples
    /// to the modes across the path, each with its force constant: the terms of the
    /// reaction path Hamiltonian), and gains `|g| s`.
    fn bend_limit(&self, arrival: &Arrival, eta: f64) -> Option<f64> {
        let kappa = self.history.curvature()?;
        let hessian = self.model.total(&self.space, &arrival.value);
        let n = kappa.nrows();
        let mut stiffness = 0.0;
        for i in 0..n {
            for j in 0..n {
                stiffness += kappa[i] * hessian[(i, j)] * kappa[j];
            }
        }
        if stiffness <= 0.0 {
            return None;
        }
        let limit = (8.0 * eta * norm(&arrival.g) / stiffness).cbrt();
        Some(limit.max(SHORTEST_SCALE * self.settings.step_size))
    }

    /// [`Self::arrive`] for integrators that take a step again when it raised the energy
    /// (see the module documentation).
    pub fn arrive_or_retry(&mut self, value: &PointValue) -> MoptResult<Arrived> {
        self.check(value)?;
        self.retrying = self.settings.retry_on_energy_rise;
        if self.refreshing {
            // The exact Hessian at the last point of the path: the step from there again.
            self.refreshing = false;
            self.model.observe(&self.space, &self.x, value)?;
            return Ok(match self.accepted.as_mut() {
                Some(last) => {
                    last.value = value.clone();
                    Arrived::Again(last.clone())
                }
                None => Arrived::Wait,
            });
        }
        let rose = self.settings.retry_on_energy_rise
            && self
                .accepted
                .as_ref()
                .is_some_and(|last| value.energy > last.value.energy + ENERGY_RISE);
        // The gradient at the end of the step points back along it: the step went past
        // the floor of the valley (the energy may still be lower), and the next one would
        // turn back.
        let overshot = !rose
            && self.settings.retry_on_energy_rise
            && self.settings.retry_on_overshoot
            && !self.approaching
            && self.halvings < MOST_OVERSHOOTS
            && self.accepted.as_ref().is_some_and(|last| {
                let chord = self.space.to_mw(&self.x) - &last.q;
                dot(&self.space.gradient_to_mw(&value.gradient), &chord) > 0.0
            });
        if !(rose || overshot) {
            let arrival = self.arrive(value)?;
            self.halvings = 0;
            self.refreshed = false;
            // A step that turned back against the one before it: the path zigzags, and
            // the next steps are shorter. Otherwise the length comes back.
            let turned_back = arrival
                .record
                .bending_angle
                .is_some_and(|angle| angle < ZIGZAG_ANGLE);
            if !self.settings.retry_on_energy_rise {
                // MultiOptPy: every step has the step size.
            } else if turned_back && !self.approaching {
                if self.scale > SHORTEST_SCALE {
                    self.counts.shortened += 1;
                }
                self.scale = (0.5 * self.scale).max(SHORTEST_SCALE);
            } else {
                self.scale = (2.0 * self.scale).min(1.0);
            }
            self.bend_limit = match self.settings.bend_share {
                Some(eta) => self.bend_limit(&arrival, eta),
                None => None,
            };
            self.accepted = Some(arrival.clone());
            return Ok(Arrived::Point(arrival));
        }
        let Some(last) = self.accepted.clone() else {
            return Ok(Arrived::Wait);
        };
        // With a gradient below the thresholds the surface rises ahead because a minimum
        // is there: the side goes on as a minimization.
        if rose
            && self.settings.final_approach
            && self.approaches
            && self.in_a_bowl
            && self.forces_arrived(&last.value)
        {
            self.approaching = true;
        }
        // Back to the last point of the path.
        if rose {
            self.counts.dropped += 1;
        } else {
            self.counts.overshot += 1;
        }
        self.x = last.record.x.clone();
        self.step = last.record.step;
        if self.settings.exact_hessian_interval.is_some()
            && !self.refreshed
            && last.value.hessian.is_none()
        {
            self.refreshed = true;
            self.refreshing = true;
            self.counts.refreshed += 1;
            self.next = BranchAction::Evaluate(PointRequest {
                x: self.x.clone(),
                need_hessian: true,
                role: PointRole::Auxiliary,
                step: self.step,
            });
            return Ok(Arrived::Wait);
        }
        if self.halvings >= MOST_HALVINGS {
            // The energy rises whatever the step: a minimum lies within the last one.
            self.next = BranchAction::Finished(BranchEnd::EnergyRise);
            return Ok(Arrived::Wait);
        }
        self.halvings += 1;
        self.scale *= 0.5;
        Ok(Arrived::Again(last))
    }

    /// Whether the gradient of `value` is below the thresholds of the arrival.
    pub(super) fn forces_arrived(&self, value: &PointValue) -> bool {
        let s = &self.settings;
        self.step > s.min_points && convergence_check(&value.gradient, s.max_force, s.rms_force)
    }

    /// The final approach at the point of `arrival` (see the module documentation).
    pub fn approach(&mut self, arrival: &Arrival) -> MoptResult<Approach> {
        if !self.settings.final_approach {
            return Ok(Approach::Path);
        }
        let hessian = self.model.total(&self.space, &arrival.value);
        let (values, vectors) = symmetric_eigen(&hessian)?;
        let n = values.nrows();
        if !self.approaching {
            // A stationary point that the path ran into, with its direction of negative
            // curvature across the path: down a symmetric ridge into the transition state at
            // its end, past a valley-ridge inflection (it came along a direction of positive
            // curvature, where next to the transition state it starts from the negative
            // curvature lies along the path). The gradient is below the tight thresholds, so
            // every step would be zero: the side has arrived there (it is not a minimum; the
            // caller tells by its Hessian). Not before `min_points` points, and only once the
            // side has lost energy.
            let s = &self.settings;
            let lost = self
                .reference_energy
                .is_some_and(|start| start > arrival.value.energy);
            let ridge_end = self.history.direction().is_some_and(|t| {
                let ht = &hessian * &t;
                let along = dot(&t, &ht);
                let across = Mat::from_fn(n, n, |i, j| {
                    hessian[(i, j)] - t[i] * ht[j] - ht[i] * t[j] + along * t[i] * t[j]
                });
                along > FLAT_MODE
                    && symmetric_eigen(&across)
                        .is_ok_and(|(v, _)| v.iter().any(|x| *x < -FLAT_MODE))
            });
            if self.step > s.min_points
                && lost
                && ridge_end
                && convergence_check(
                    &arrival.value.gradient,
                    s.approach_max_force,
                    s.approach_rms_force,
                )
            {
                self.stop(BranchEnd::Converged);
                return Ok(Approach::Arrived);
            }
            // A small gradient alone does not say that a minimum is near: on a flat
            // surface it is small all the way from the transition state, at which it is
            // below any threshold. The approach begins where the path lies in the bowl
            // of the minimum: the model has no direction of negative curvature left, and
            // down to its minimum it has less energy to lose than a tenth of what the
            // side has lost since its start (next to a transition state the two are
            // equal: `g^2 / (2 |lambda|)` and `|lambda| x^2 / 2` with `g = lambda x`).
            // A step of the path that raised the energy at a point in the bowl begins
            // it as well.
            let left: f64 = (0..n)
                .filter(|&k| values[k].abs() > FLAT_MODE)
                .map(|k| {
                    let along: f64 = (0..n).map(|i| vectors[(i, k)] * arrival.g[i]).sum();
                    0.5 * along * along / values[k].abs()
                })
                .sum();
            let lost = self
                .reference_energy
                .map_or(0.0, |start| start - arrival.value.energy);
            self.in_a_bowl =
                left < NEAR_THE_MINIMUM * lost && (0..n).all(|k| values[k] > -FLAT_MODE);
            if !(self.forces_arrived(&arrival.value) && self.in_a_bowl) {
                return Ok(Approach::Path);
            }
            self.approaching = true;
        }
        let s = &self.settings;
        let gradient = &arrival.value.gradient;
        if convergence_check(gradient, s.approach_max_force, s.approach_rms_force)
            || self.counts.approach >= s.approach_steps
            || self.is_last()
        {
            self.stop(BranchEnd::Converged);
            return Ok(Approach::Arrived);
        }
        // Towards the minimum of the quadratic model as a step of LQA does it (the
        // steepest-descent path of the model, at most a step of the path long, which ends
        // in the minimum of the model when that is nearer), with the magnitude of the
        // curvature along every mode: downhill whatever the sign the model has there.
        let magnitudes = Col::from_fn(n, |k| values[k].abs());
        let limit = self.step_length();
        let mut step = lqa_step_in_modes(&magnitudes, &vectors, &arrival.g, limit, 1e-8).step;
        let length = norm(&step);
        if length == 0.0 {
            self.stop(BranchEnd::Converged);
            return Ok(Approach::Arrived);
        }
        // No longer than a step of the path, and halved with every time it raised the
        // energy (a step that is shorter than the limit is not shortened by the limit).
        let wanted = length * 0.5_f64.powi(self.halvings as i32);
        let taken = wanted.min(self.step_length());
        step *= taken / length;
        self.counts.approach += 1;
        Ok(Approach::Step(self.space.displaced(&self.x, &step)))
    }
}
