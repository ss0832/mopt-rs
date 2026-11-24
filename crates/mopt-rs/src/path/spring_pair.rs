//! The spring pair method (SPM): a pair of points joined by a spring finds a saddle point of
//! order 1 from a minimum, using the minimum energy path as its compass.
//!
//! Both points drift under the force across the line of the pair and the spring force,
//! `r_i <- r_i + a1 F_perp(r_i) + a2 F_s(r_i)` with `F_s(r1) = (d - l)(r2 - r1)` and
//! `F_s(r2) = -F_s(r1)` (`d = |r2 - r1|`, `l` the natural length), until the force across the
//! line is small (or after a number of steps); the pair has then settled on the minimum
//! energy path along its tangent. Both points then climb along the line of the pair,
//! `r_i <- r_i - a3 F_par(r_i)`. The cycles end when the force at one of the points meets the
//! thresholds while the curvature along the line (from the forces at the two points) is
//! negative: that point is near the saddle, and the line of the pair approximates its
//! unstable mode. (Without the curvature the minimum the pair starts from would count.)
//!
//! G. Cui, K. Jiang, Phys. Rev. E 2024, 110, 064123 (arXiv:2407.04373), Algorithm 1. The
//! second point starts at `r1 + xi v`; `v` points towards a second structure when one is given
//! (the paper names the final state as a good choice). The paper leaves the end of a drift
//! open ("relaxed"); here it ends below `epsilon_1`, or below the largest force along the
//! line when that is smaller, but not below half the force thresholds of the end (with
//! `epsilon_1` alone a pair near the saddle keeps a force across the line that never meets
//! them). Points that run together keep the line they had. The steps are limited to a
//! largest length (a safeguard for stiff surfaces that the paper does not need), and for
//! structures on surfaces without a preferred orientation the second point is turned onto
//! the first after every step (MultiOptPy does so as well), so that the spring measures the
//! shape.
//!
//! MultiOptPy: `OtherMethod/spring_pair_method.py:SpringPairMethod`. MultiOptPy takes a
//! Hooke spring `10 (d - l)` along the unit vector, adds momentum and an adaptive step to the
//! drift, starts the second point from a random displacement of every atom (ignoring a second
//! structure), climbs by half the parallel gradient, ends on the smaller gradient norm after at
//! least six cycles, and writes the average of the two points to the current directory.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Col;

use super::ieip::aligned_pair;
use super::points::{check_values, gradient_of, PointRequest, PointValue, SearchAction};
use crate::error::{MoptError, MoptResult};
use crate::math::linalg::{col_dot, col_max_abs, col_norm};
use crate::opt::convergence::ConvergenceCriteria;

/// Settings of the spring pair method (molecules: Bohr and Hartree).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpringPairSettings {
    /// Natural length `l` of the spring.
    pub spring_length: f64,
    /// Distance `xi` of the second point from the first at the start.
    pub offset: f64,
    /// `a1`: step per unit of the force across the line.
    pub drift_rate: f64,
    /// `a2`: step per unit of the spring force.
    pub spring_rate: f64,
    /// `a3`: step per unit of the force along the line (climbing).
    pub climb_rate: f64,
    /// The drift ends when the largest component of the force across the line is below this
    /// at both points (`epsilon_1`).
    pub drift_tolerance: f64,
    /// Drift steps at most per cycle (`n_d`).
    pub drift_steps: usize,
    /// Cycles (climbs) at most.
    pub max_cycles: usize,
    /// Largest length of a step of one point.
    pub max_step: f64,
    /// The force thresholds of the end (`epsilon_2`; maximum and RMS component).
    pub criteria: ConvergenceCriteria,
    /// Turn the second point onto the first after every step.
    pub align: bool,
}

/// The rates are measured on HCN -> HNC and CH3CN -> CH3NC with GFN0-xTB (molecules in Bohr
/// and Hartree; docs/source/reaction_paths.md, "ieipmain"): the drift is a gradient step, stable
/// below `2 / lambda_max` (about 0.87 for the stiffest Cartesian Hessian eigenvalue, 2.3, of
/// these molecules), so 0.5 leaves room for stiffer bonds; the climb grows by
/// `1 + a3 lambda` per cycle along a soft line, so a larger rate climbs faster, and 5 stays
/// stable at the saddle for curvatures down to -0.4.
impl Default for SpringPairSettings {
    fn default() -> Self {
        Self {
            spring_length: 0.1,
            offset: 0.1,
            drift_rate: 0.5,
            spring_rate: 1.0,
            climb_rate: 5.0,
            drift_tolerance: 2e-3,
            drift_steps: 200,
            max_cycles: 999,
            max_step: 0.1,
            criteria: ConvergenceCriteria::NORMAL,
            align: true,
        }
    }
}

/// Why a spring pair search ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpringPairEnd {
    /// The force at one of the points meets the thresholds.
    Converged,
    /// The cycles are used up.
    CycleLimit,
}

/// What one `tell` did.
#[derive(Debug, Clone)]
pub struct SpringPairReport {
    /// Cycle (climbs so far).
    pub cycle: usize,
    /// Drift step within the cycle (`None`: the climb).
    pub drift_step: Option<usize>,
    /// The points that were evaluated.
    pub points: [Col<f64>; 2],
    /// Their energies.
    pub energies: [f64; 2],
    /// Largest component of the gradient at each.
    pub max_force: [f64; 2],
    /// Largest component of the force across the line at each.
    pub perpendicular: [f64; 2],
    /// Length of the spring.
    pub length: f64,
    /// Why the search ended, if it did.
    pub end: Option<SpringPairEnd>,
}

/// Ask/tell state machine of the spring pair method.
pub struct SpringPair {
    settings: SpringPairSettings,
    points: [Col<f64>; 2],
    /// The unit vector from the first point to the second at the last evaluation.
    line: Col<f64>,
    frozen: Vec<usize>,
    cycle: usize,
    drift: usize,
    best: Option<(usize, Col<f64>, f64)>,
    end: Option<SpringPairEnd>,
    step_map: Option<super::points::StepMap>,
}

fn rms(v: &Col<f64>, free: usize) -> f64 {
    libm::sqrt(col_dot(v, v) / free.max(1) as f64)
}

impl SpringPair {
    /// A pair from the minimum `start` with the second point `offset` along `direction`.
    pub fn new(
        settings: SpringPairSettings,
        start: &Col<f64>,
        direction: &Col<f64>,
    ) -> MoptResult<Self> {
        let n = start.nrows();
        let size = col_norm(direction);
        if n == 0 || direction.nrows() != n || !(size.is_finite() && size > 0.0) {
            return Err(MoptError::config(
                "spring pair: the direction must be a nonzero vector of the size of the start",
            ));
        }
        let rates = [
            settings.offset,
            settings.drift_rate,
            settings.climb_rate,
            settings.max_step,
            settings.drift_tolerance,
        ];
        if rates.iter().any(|v| !(v.is_finite() && *v > 0.0))
            || settings.spring_rate < 0.0
            || settings.spring_length < 0.0
            || settings.drift_steps == 0
            || settings.max_cycles == 0
        {
            return Err(MoptError::config(
                "spring pair: the rates, the offset, the step, and the counts must be positive",
            ));
        }
        if settings.align && !n.is_multiple_of(3) {
            return Err(MoptError::config(
                "spring pair: only structures can be turned onto each other",
            ));
        }
        let second = start + direction * (settings.offset / size);
        let mut pair = Self {
            settings,
            line: direction * (1.0 / size),
            points: [start.clone(), second],
            frozen: Vec::new(),
            cycle: 0,
            drift: 0,
            best: None,
            end: None,
            step_map: None,
        };
        pair.align();
        Ok(pair)
    }

    /// Take the steps of the two points through `map`, which turns the step from `x` to `y`
    /// into the point it reaches (the point put back onto constraints); the second point is
    /// put there at once.
    ///
    /// MultiOptPy: no counterpart (new in mopt-rs).
    pub fn with_step_map(mut self, map: super::points::StepMap) -> MoptResult<Self> {
        self.points[1] = map(&self.points[0], &self.points[1])?;
        if col_norm(&(&self.points[1] - &self.points[0])) == 0.0 {
            return Err(MoptError::config(
                "spring pair: the direction leaves the constraints only",
            ));
        }
        self.step_map = Some(map);
        Ok(self)
    }

    /// Both points moved by `steps`.
    fn take(&mut self, steps: [&Col<f64>; 2]) -> MoptResult<()> {
        for (point, step) in self.points.iter_mut().zip(steps) {
            let to = &*point + step;
            *point = match &self.step_map {
                Some(map) => map(point, &to)?,
                None => to,
            };
        }
        Ok(())
    }

    /// Keep these variables where they are (fixed atoms; the points are then not turned).
    pub fn with_frozen(mut self, variables: Vec<usize>) -> MoptResult<Self> {
        let n = self.points[0].nrows();
        if variables.iter().any(|&k| k >= n) {
            return Err(MoptError::config(
                "spring pair: a frozen variable is out of range",
            ));
        }
        for &k in &variables {
            self.points[1][k] = self.points[0][k];
        }
        if col_norm(&(&self.points[1] - &self.points[0])) == 0.0 {
            return Err(MoptError::config(
                "spring pair: the direction moves fixed atoms only",
            ));
        }
        self.frozen = variables;
        self.settings.align = false;
        Ok(self)
    }

    fn align(&mut self) {
        if self.settings.align {
            let (second, first) = aligned_pair(&self.points[1], &self.points[0]);
            self.points = [first, second];
        }
    }

    fn masked(&self, mut v: Col<f64>) -> Col<f64> {
        for &k in &self.frozen {
            v[k] = 0.0;
        }
        v
    }

    fn limited(&self, v: Col<f64>) -> Col<f64> {
        let v = self.masked(v);
        let length = col_norm(&v);
        if length > self.settings.max_step {
            v * (self.settings.max_step / length)
        } else {
            v
        }
    }

    /// The two points (after the end: the last ones).
    pub fn points(&self) -> [&Col<f64>; 2] {
        [&self.points[0], &self.points[1]]
    }

    /// Climbs so far.
    pub fn cycles(&self) -> usize {
        self.cycle
    }

    /// The point with the smaller largest force at the end (the estimate of the saddle):
    /// its index and coordinates.
    pub fn saddle_estimate(&self) -> Option<(usize, &Col<f64>)> {
        self.best.as_ref().map(|(k, x, _)| (*k, x))
    }

    /// Why the search ended, once it has.
    pub fn end(&self) -> Option<SpringPairEnd> {
        self.end
    }

    /// Next action.
    pub fn ask(&self) -> SearchAction {
        if self.end.is_some() {
            return SearchAction::Finished;
        }
        SearchAction::Evaluate(vec![
            PointRequest::gradient(self.points[0].clone()),
            PointRequest::gradient(self.points[1].clone()),
        ])
    }

    /// The values at the two points of [`Self::ask`].
    pub fn tell(&mut self, values: Vec<PointValue>) -> MoptResult<SpringPairReport> {
        if self.end.is_some() {
            return Err(MoptError::config(
                "tell() called after the spring pair search ended",
            ));
        }
        let n = self.points[0].nrows();
        let asked = [
            PointRequest::gradient(self.points[0].clone()),
            PointRequest::gradient(self.points[1].clone()),
        ];
        check_values("spring pair", &asked, &values, n)?;
        let g = [
            self.masked(gradient_of("spring pair", &values[0])?),
            self.masked(gradient_of("spring pair", &values[1])?),
        ];
        let between = &self.points[1] - &self.points[0];
        let d = col_norm(&between);
        // Points that have run together keep the line they had.
        let v = if d > 1e-12 {
            &between * (1.0 / d)
        } else {
            self.line.clone()
        };
        self.line = v.clone();
        let along = [col_dot(&g[0], &v), col_dot(&g[1], &v)];
        // Forces across the line: -(g - (g . v) v).
        let across = [
            (&g[0] - &v * along[0]) * -1.0,
            (&g[1] - &v * along[1]) * -1.0,
        ];
        let free = n - self.frozen.len();
        let max_force = [col_max_abs(&g[0]), col_max_abs(&g[1])];
        let perpendicular = [col_max_abs(&across[0]), col_max_abs(&across[1])];
        let report = |s: &Self, drift_step: Option<usize>| SpringPairReport {
            cycle: s.cycle,
            drift_step,
            points: [values_point(&asked[0]), values_point(&asked[1])],
            energies: [values[0].energy, values[1].energy],
            max_force,
            perpendicular,
            length: d,
            end: s.end,
        };
        // The drift ends below epsilon_1, or below the force along the line when that is
        // smaller (near the saddle), down to half the force thresholds of the end (so that a
        // point that has settled can meet them).
        let largest_along = along.iter().fold(0.0_f64, |m, a| m.max(a.abs()));
        let c = &self.settings.criteria;
        let floor = 0.5 * c.max_force.value().min(c.rms_force.value());
        let tolerance = self.settings.drift_tolerance.min(largest_along).max(floor);
        let settled = perpendicular.iter().all(|f| *f < tolerance);
        if !settled && self.drift < self.settings.drift_steps {
            // Drift.
            let spring = &between * (d - self.settings.spring_length);
            let s = &self.settings;
            let step1 = self.limited(&across[0] * s.drift_rate + &spring * s.spring_rate);
            let step2 = self.limited(&across[1] * s.drift_rate - &spring * s.spring_rate);
            let r = report(self, Some(self.drift));
            self.take([&step1, &step2])?;
            self.drift += 1;
            self.align();
            return Ok(r);
        }
        // The pair has settled: is one of the points at the saddle? The curvature along the
        // line (from the two points) must be negative there: at the minimum the pair starts
        // from, the force meets the thresholds as well.
        let c = &self.settings.criteria;
        let meets =
            |k: usize| max_force[k] < c.max_force.value() && rms(&g[k], free) < c.rms_force.value();
        let curvature = if d > 1e-12 {
            (along[1] - along[0]) / d
        } else {
            0.0
        };
        let k = if max_force[0] <= max_force[1] { 0 } else { 1 };
        self.best = Some((k, self.points[k].clone(), max_force[k]));
        if meets(k) && curvature < 0.0 {
            self.end = Some(SpringPairEnd::Converged);
            return Ok(report(self, None));
        }
        // Climb along the line.
        let s = &self.settings;
        let climb1 = self.limited(&v * (s.climb_rate * along[0]));
        let climb2 = self.limited(&v * (s.climb_rate * along[1]));
        let r = report(self, None);
        self.take([&climb1, &climb2])?;
        self.align();
        self.cycle += 1;
        self.drift = 0;
        if self.cycle >= self.settings.max_cycles {
            self.end = Some(SpringPairEnd::CycleLimit);
        }
        Ok(SpringPairReport { end: self.end, ..r })
    }
}

fn values_point(p: &PointRequest) -> Col<f64> {
    p.x.clone()
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    /// `E = (x^2 - 1)^2 + 2 (y - 0.2 x)^2`.
    fn surface(p: &Col<f64>) -> PointValue {
        let (x, y) = (p[0], p[1]);
        let w = y - 0.2 * x;
        PointValue {
            energy: (x * x - 1.0).powi(2) + 2.0 * w * w,
            gradient: Some(Col::from_fn(2, |i| {
                [4.0 * x * (x * x - 1.0) - 0.8 * w, 4.0 * w][i]
            })),
            hessian: None,
        }
    }

    #[test]
    fn climbs_from_a_minimum_to_the_saddle() {
        let settings = SpringPairSettings {
            align: false,
            drift_rate: 0.05,
            spring_rate: 0.25,
            climb_rate: 0.05,
            spring_length: 0.01,
            offset: 0.3,
            drift_tolerance: 1e-2,
            max_step: 0.05,
            ..SpringPairSettings::default()
        };
        let start = Col::from_fn(2, |i| [1.0, 0.2][i]);
        // Roughly towards the other minimum (the final state, as the paper suggests).
        let direction = Col::from_fn(2, |i| [-1.0, -0.1][i]);
        let mut s = SpringPair::new(settings, &start, &direction).unwrap();
        let mut evaluations = 0;
        while let SearchAction::Evaluate(points) = s.ask() {
            evaluations += 2;
            let values = points.iter().map(|p| surface(&p.x)).collect();
            s.tell(values).unwrap();
            assert!(evaluations < 200_000);
        }
        assert_eq!(s.end(), Some(SpringPairEnd::Converged));
        let (_, x) = s.saddle_estimate().unwrap();
        assert!(col_norm(x) < 1e-3, "{x:?}");
    }
}
