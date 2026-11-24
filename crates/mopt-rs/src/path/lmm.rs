//! The local minimax method (LMM) for a saddle point of order 1 next to a minimum
//! (experimental in mopt-rs).
//!
//! With the minimum `A` as the base (the support space `L = {0}`), the peak selection of a
//! unit direction `v` is `p(v) = A + t(v) v`, `t(v)` the local maximum of the energy along the
//! ray from `A` (where `g . v = 0`); the saddle is a local minimum of `E(p(v))` over the unit
//! sphere. The iteration descends from `v_k` with the gradient `g_k` at the peak,
//! `v_k(a) = (v_k - a g_k) / |v_k - a g_k|`, the step `a` by the normalized Armijo rule
//! `E(p(v_k(a))) <= E(p(v_k)) - sigma a t_k |g_k|^2` over `a = lambda rho^m` (sigma = 1e-4,
//! rho = 0.2), and ends when the gradient at the peak meets the force thresholds.
//!
//! Y. Li, J. Zhou, SIAM J. Sci. Comput. 2001, 23, 840 (the method); the normalized Armijo rule
//! and the trial step of Barzilai and Borwein `a = (s . y) / (y . y)` (with `s = v_k - v_k-1`,
//! `y = g_k - g_k-1`, clipped to `[1e-6, 10]`; the projected form) as arXiv:2109.01865 (a
//! nonmonotone local minimax method with such steps) restates and extends them; its
//! nonmonotone search is not taken over. Here (mopt-rs): the peak along a ray is found
//! by bracketing the sign change of `g . v` and the secant method on it, from the peak of the
//! direction before; rigid-body motions are taken out of the directions; the start is the
//! direction towards a second structure. The method needs the mountain pass to be reachable
//! along straight rays from `A` (in the coordinates of the points; `-coord` changes them).
//!
//! Hard constraints `c(x) = t` ([`super::string_nodes::PointGeometry::manifold`]; new in
//! mopt-rs, not in the papers): the base `A` lies on their manifold `M`, the directions `v`
//! are unit vectors of `T_A M` (the step `v - a g` and its normalization projected there),
//! and the ray becomes the curve `p(v, t) = R(A + t v)` on `M` (`R` the restore of the
//! constraints). Its slope is `dE/dt = g . dR/dt`; with the gradient `g` projected into
//! `T_p M` (as the surfaces give it) and `dR/dt = P_M v + O(t kappa)` (`kappa` the curvature
//! of `M`), the test of the peak `g . v = 0` is that of the curve to first order, and the
//! peak is the maximum of the energy along it on `M`. The saddle found is that of the
//! energy on `M`: its gradient on `M` vanishes.
//!
//! [`LocalMinimax`] is an ask/tell state machine over points of `n` variables.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Col;

use super::points::{check_values, gradient_of, PointRequest, PointValue, SearchAction};
use super::string_nodes::{rms, unit, PointGeometry};
use crate::error::{MoptError, MoptResult};
use crate::math::linalg::{col_dot, col_max_abs};
use crate::opt::convergence::ConvergenceCriteria;

/// Settings of the local minimax method (molecules: Bohr and Hartree).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LmmSettings {
    /// Outer iterations (directions) at most.
    pub max_iterations: usize,
    /// `lambda`: the first trial step of the Armijo rule (without a step of Barzilai and
    /// Borwein).
    pub first_step: f64,
    /// `sigma` of the Armijo rule.
    pub sigma: f64,
    /// `rho` of the Armijo rule.
    pub rho: f64,
    /// Trials of the Armijo rule at most per iteration.
    pub max_trials: usize,
    /// Use the step of Barzilai and Borwein as the first trial.
    pub barzilai_borwein: bool,
    /// The first distance along the first ray.
    pub first_distance: f64,
    /// Largest step along a ray while the maximum is bracketed.
    pub ray_step: f64,
    /// The peak is found when `|g . v|` is below this.
    pub ray_tolerance: f64,
    /// Evaluations at most to find a peak.
    pub ray_evaluations: usize,
    /// The force thresholds at the peak.
    pub criteria: ConvergenceCriteria,
}

impl Default for LmmSettings {
    fn default() -> Self {
        Self {
            max_iterations: 200,
            first_step: 0.1,
            sigma: 1e-4,
            rho: 0.2,
            max_trials: 8,
            barzilai_borwein: true,
            first_distance: 1.0,
            ray_step: 0.3,
            ray_tolerance: 1e-4,
            ray_evaluations: 30,
            criteria: ConvergenceCriteria::NORMAL,
        }
    }
}

/// Why the method ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LmmEnd {
    /// The gradient at the peak meets the force thresholds.
    Converged,
    /// The iterations are used up.
    IterationLimit,
    /// A ray has no maximum within the evaluations (the energy keeps rising or the ray
    /// leaves the basin without a pass).
    NoPeak,
}

/// A peak: the distance along the ray, the point, its energy and gradient.
#[derive(Debug, Clone)]
pub struct Peak {
    /// Distance from the base along the direction.
    pub t: f64,
    /// The point.
    pub x: Col<f64>,
    /// Its energy.
    pub energy: f64,
    /// Its gradient.
    pub gradient: Col<f64>,
}

/// What one `tell` did.
#[derive(Debug, Clone)]
pub struct LmmReport {
    /// Outer iteration.
    pub iteration: usize,
    /// The point evaluated and `g . v` there.
    pub x: Col<f64>,
    /// Its energy.
    pub energy: f64,
    /// The peak found, if one was.
    pub peak: Option<Peak>,
    /// Why the method ended, if it did.
    pub end: Option<LmmEnd>,
}

#[derive(Debug, Clone)]
struct RaySearch {
    v: Col<f64>,
    /// The largest distance known with `g . v > 0` and the smallest with `g . v < 0`.
    rising: Option<(f64, f64)>,
    falling: Option<(f64, f64)>,
    t: f64,
    evaluations: usize,
    /// The trial step of the Armijo rule this search answers (`None`: the first direction).
    step: Option<f64>,
}

/// Ask/tell state machine of the local minimax method.
pub struct LocalMinimax {
    settings: LmmSettings,
    geometry: PointGeometry,
    base: Col<f64>,
    v: Col<f64>,
    peak: Option<Peak>,
    previous: Option<(Col<f64>, Col<f64>)>,
    search: RaySearch,
    iteration: usize,
    trials: usize,
    end: Option<LmmEnd>,
    free: usize,
    /// The point asked for.
    x: Col<f64>,
}

impl LocalMinimax {
    /// From the minimum `base` along `direction` (towards a second structure, or any).
    pub fn new(
        settings: LmmSettings,
        base: &Col<f64>,
        direction: &Col<f64>,
        geometry: PointGeometry,
    ) -> MoptResult<Self> {
        let n = base.nrows();
        let positive = [
            settings.first_step,
            settings.sigma,
            settings.rho,
            settings.first_distance,
            settings.ray_step,
            settings.ray_tolerance,
        ];
        if n == 0
            || direction.nrows() != n
            || positive.iter().any(|v| !(v.is_finite() && *v > 0.0))
            || settings.rho >= 1.0
            || settings.max_iterations == 0
            || settings.ray_evaluations < 2
        {
            return Err(MoptError::config(
                "local minimax: a direction of the size of the base, positive steps and \
                 tolerances, rho below 1",
            ));
        }
        let v = unit(&geometry.project(base, direction))
            .ok_or_else(|| MoptError::config("local minimax: the direction is zero"))?;
        let free = n - geometry.frozen.len().min(n);
        let t = settings.first_distance;
        let mut s = Self {
            x: base.clone(),
            search: RaySearch {
                v: v.clone(),
                rising: None,
                falling: None,
                t,
                evaluations: 0,
                step: None,
            },
            settings,
            geometry,
            base: base.clone(),
            v,
            peak: None,
            previous: None,
            iteration: 0,
            trials: 0,
            end: None,
            free,
        };
        s.place()?;
        Ok(s)
    }

    /// The peak of the current direction (after the end: the estimate of the saddle).
    pub fn peak(&self) -> Option<&Peak> {
        self.peak.as_ref()
    }

    /// Why the method ended, once it has.
    pub fn end(&self) -> Option<LmmEnd> {
        self.end
    }

    /// Outer iterations so far.
    pub fn iterations(&self) -> usize {
        self.iteration
    }

    /// The point of the ray search, `A + t v`, put onto the hard constraints (the curve
    /// `R(A + t v)` of the module documentation).
    fn place(&mut self) -> MoptResult<()> {
        self.x = self
            .geometry
            .settle(&(&self.base + &self.search.v * self.search.t))?;
        Ok(())
    }

    fn point(&self) -> Col<f64> {
        self.x.clone()
    }

    /// Next action.
    pub fn ask(&self) -> SearchAction {
        if self.end.is_some() {
            return SearchAction::Finished;
        }
        SearchAction::Evaluate(vec![PointRequest::gradient(self.point())])
    }

    /// The value at the point of [`Self::ask`].
    pub fn tell(&mut self, values: Vec<PointValue>) -> MoptResult<LmmReport> {
        if self.end.is_some() {
            return Err(MoptError::config(
                "tell() called after the local minimax method ended",
            ));
        }
        let x = self.point();
        check_values(
            "local minimax",
            &[PointRequest::gradient(x.clone())],
            &values,
            x.nrows(),
        )?;
        let g = self
            .geometry
            .project(&x, &gradient_of("local minimax", &values[0])?);
        let energy = values[0].energy;
        let mut report = LmmReport {
            iteration: self.iteration,
            x: x.clone(),
            energy,
            peak: None,
            end: None,
        };
        let found = self.ray_step(&x, energy, g)?;
        if let Some(peak) = found {
            report.peak = Some(peak.clone());
            self.accept_or_retry(peak)?;
        }
        if self.end.is_none() {
            self.place()?;
        }
        report.end = self.end;
        Ok(report)
    }

    /// One evaluation of the search along the ray; the peak when it is found.
    fn ray_step(&mut self, x: &Col<f64>, energy: f64, g: Col<f64>) -> MoptResult<Option<Peak>> {
        let s = &mut self.search;
        s.evaluations += 1;
        let h = col_dot(&g, &s.v);
        if h.abs() < self.settings.ray_tolerance {
            return Ok(Some(Peak {
                t: s.t,
                x: x.clone(),
                energy,
                gradient: g,
            }));
        }
        if h > 0.0 {
            s.rising = Some(
                s.rising
                    .map_or((s.t, h), |r| if s.t > r.0 { (s.t, h) } else { r }),
            );
        } else {
            s.falling = Some(
                s.falling
                    .map_or((s.t, h), |f| if s.t < f.0 { (s.t, h) } else { f }),
            );
        }
        if s.evaluations >= self.settings.ray_evaluations {
            self.end = Some(LmmEnd::NoPeak);
            return Ok(None);
        }
        s.t = match (s.rising, s.falling) {
            // The secant on g . v inside the bracket, kept away from its ends.
            (Some((a, ha)), Some((b, hb))) if a < b => {
                let t = a + (b - a) * ha / (ha - hb);
                t.clamp(a + 0.05 * (b - a), b - 0.05 * (b - a))
            }
            // Still rising: further out (or back out past a falling point found inside).
            (Some((a, _)), _) => a + self.settings.ray_step,
            // Falling from the start: further in, but not through the base.
            (None, Some((b, _))) => {
                if b <= 1e-6 {
                    self.end = Some(LmmEnd::NoPeak);
                    return Ok(None);
                }
                (b - self.settings.ray_step).max(0.5 * b)
            }
            (None, None) => s.t,
        };
        Ok(None)
    }

    fn new_search(&mut self, v: Col<f64>, step: Option<f64>, t: f64) {
        self.search = RaySearch {
            v,
            rising: None,
            falling: None,
            t,
            evaluations: 0,
            step,
        };
    }

    /// The direction after the step `a` from the current one.
    fn stepped(&self, a: f64) -> MoptResult<Col<f64>> {
        let peak = self
            .peak
            .as_ref()
            .ok_or_else(|| MoptError::config("local minimax: no peak yet"))?;
        let w = &self.v - &peak.gradient * a;
        unit(&self.geometry.project(&self.base, &w))
            .ok_or_else(|| MoptError::calc("local minimax: the step cancels the direction"))
    }

    fn converged(&self, peak: &Peak) -> bool {
        let c = &self.settings.criteria;
        col_max_abs(&peak.gradient) < c.max_force.value()
            && rms(&peak.gradient, self.free) < c.rms_force.value()
    }

    /// The Armijo rule on the peak found for a trial step, or the first peak.
    fn accept_or_retry(&mut self, peak: Peak) -> MoptResult<()> {
        let Some(a) = self.search.step else {
            // The peak of the first direction.
            return self.accept(peak);
        };
        let current = self
            .peak
            .as_ref()
            .ok_or_else(|| MoptError::config("local minimax: no peak yet"))?;
        let g2 = col_dot(&current.gradient, &current.gradient);
        if peak.energy <= current.energy - self.settings.sigma * a * current.t * g2
            || self.trials >= self.settings.max_trials
        {
            let v_new = self.search.v.clone();
            self.previous = Some((self.v.clone(), current.gradient.clone()));
            self.v = v_new;
            return self.accept(peak);
        }
        self.trials += 1;
        let a = a * self.settings.rho;
        let t = current.t;
        let v = self.stepped(a)?;
        self.new_search(v, Some(a), t);
        Ok(())
    }

    fn accept(&mut self, peak: Peak) -> MoptResult<()> {
        let done = self.converged(&peak);
        let t = peak.t;
        self.peak = Some(peak);
        self.iteration += 1;
        self.trials = 0;
        if done {
            self.end = Some(LmmEnd::Converged);
            return Ok(());
        }
        if self.iteration >= self.settings.max_iterations {
            self.end = Some(LmmEnd::IterationLimit);
            return Ok(());
        }
        let a = self.trial_step();
        let v = self.stepped(a)?;
        self.new_search(v, Some(a), t);
        Ok(())
    }

    /// The first trial step: that of Barzilai and Borwein on the projected differences, or
    /// `first_step`.
    fn trial_step(&self) -> f64 {
        let fallback = self.settings.first_step;
        if !self.settings.barzilai_borwein {
            return fallback;
        }
        let (Some((v_old, g_old)), Some(peak)) = (&self.previous, &self.peak) else {
            return fallback;
        };
        let s = &self.v - v_old;
        let project = |u: &Col<f64>| u - &self.v * col_dot(u, &self.v);
        let y = project(&peak.gradient) - project(g_old);
        let sy = col_dot(&s, &y);
        if sy > 0.0 {
            (sy / col_dot(&y, &y)).clamp(1e-6, 10.0)
        } else {
            fallback
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::math::linalg::col_norm;

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
    fn the_peak_of_the_best_ray_is_the_saddle() {
        let base = Col::from_fn(2, |k| [-1.0, -0.2][k]);
        let direction = Col::from_fn(2, |k| [2.0, 0.8][k]);
        let settings = LmmSettings {
            ray_step: 0.2,
            first_distance: 0.5,
            ..LmmSettings::default()
        };
        let mut m =
            LocalMinimax::new(settings, &base, &direction, PointGeometry::default()).unwrap();
        let mut evaluations = 0;
        while let SearchAction::Evaluate(points) = m.ask() {
            evaluations += 1;
            m.tell(points.iter().map(|p| surface(&p.x)).collect())
                .unwrap();
            assert!(evaluations < 5000);
        }
        assert_eq!(m.end(), Some(LmmEnd::Converged));
        assert!(
            col_norm(&m.peak().unwrap().x) < 1e-3,
            "{:?}",
            m.peak().unwrap().x
        );
    }
}
