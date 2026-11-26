//! Restricted-step image RFO (RS-I-RFO) for minima (`saddle_order = 0`) and saddle points.
//!
//! Banerjee et al., J. Phys. Chem. 1985, 89, 52; Heyden et al., J. Chem. Phys. 2005, 123,
//! 224101; Besalu and Bofill, Theor. Chem. Acc. 1998, 100, 265. Ported from MultiOptPy's
//! `Optimizer/rsirfo.py` (itself based on pysisyphus' `RSIRFOptimizer`).
//!
//! Per step: project rigid-body modes out of the gradient and Hessian (Cartesian coordinates
//! only), adjust the internal trust radius from the ratio of actual to predicted energy change,
//! build the image Hessian by reflecting the `saddle_order` lowest modes, drop near-zero modes,
//! and take the restricted-step RFO step.
//!
//! MultiOptPy: `Optimizer/rsirfo.py:RSIRFO`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::collections::VecDeque;

use faer::{Col, Mat};

use super::mode_following::ModeFollowing;
use super::rfo_core::{rs_step, solve_rfo, RfoParams};
use super::{StepGenerator, StepInput};
use crate::error::MoptResult;
use crate::math::linalg::{col_dot, col_norm, symmetric_eigen, symmetrize};
use crate::math::projection::{project_out_basis_mat, project_out_basis_vec};

/// Internal trust-radius policy parameters.
#[derive(Debug, Clone, Copy)]
pub struct RsirfoTrust {
    /// Initial radius.
    pub initial: f64,
    /// Lower bound.
    pub min: f64,
    /// Upper bound.
    pub max: f64,
    /// The curvature-aware update is used only below this gradient norm.
    pub adaptive_gradient_threshold: f64,
    /// Upper bound on the curvature factor.
    pub max_curvature_factor: f64,
    /// Extra damping along negative curvature in saddle searches.
    pub negative_curvature_safety: f64,
}

impl RsirfoTrust {
    /// MultiOptPy defaults: radius 0.5 (minimization) or 0.1 (saddle search), minimum 0.01.
    pub fn for_saddle_order(saddle_order: usize, max: Option<f64>, min: Option<f64>) -> Self {
        let default = if saddle_order == 0 { 0.5 } else { 0.1 };
        Self {
            initial: default,
            min: min.unwrap_or(0.01),
            max: max.unwrap_or(default),
            adaptive_gradient_threshold: 1e-2,
            max_curvature_factor: 2.5,
            negative_curvature_safety: 0.8,
        }
    }
}

/// RS-I-RFO step generator.
#[derive(Debug, Clone)]
pub struct Rsirfo {
    rfo: RfoParams,
    trust: RsirfoTrust,
    radius: f64,
    small_eigval: f64,
    /// Reflect with `P = I - v v^T` instead of `I - 2 v v^T` (NEB climbing images).
    pub neb_mode: bool,
    predicted: VecDeque<f64>,
    actual: VecDeque<f64>,
    prev_energy: Option<f64>,
    /// Steps proposed so far.
    count: usize,
    /// What the last step found, for the debug level of the run log.
    debug: Vec<String>,
}

/// Above this ratio of the largest to the smallest curvature the Hessian is called
/// ill-conditioned in the debug messages (MultiOptPy's `condition_number_threshold`).
const ILL_CONDITIONED: f64 = 1e8;

impl Rsirfo {
    /// New generator with the given trust-radius policy.
    pub fn new(trust: RsirfoTrust) -> Self {
        Self {
            rfo: RfoParams::default(),
            trust,
            radius: trust.initial,
            small_eigval: 1e-6,
            neb_mode: false,
            predicted: VecDeque::new(),
            actual: VecDeque::new(),
            prev_energy: None,
            count: 0,
            debug: Vec::new(),
        }
    }

    /// Current internal trust radius.
    pub fn trust_radius(&self) -> f64 {
        self.radius
    }

    /// Forget the energies of earlier steps (keeping the trust radius), so that the next call
    /// does not rate a step taken by another method.
    pub fn forget_last_step(&mut self) {
        self.predicted.clear();
        self.actual.clear();
        self.prev_energy = None;
    }

    fn push_bounded(q: &mut VecDeque<f64>, v: f64) {
        if q.len() >= 3 {
            q.pop_front();
        }
        q.push_back(v);
    }

    fn adjust_conventional(&mut self, ratio: f64) {
        if ratio > 0.75 {
            self.radius = (self.radius * 1.2).min(self.trust.max);
        } else if ratio < 0.25 {
            self.radius = (self.radius * 0.5).max(self.trust.min);
        }
    }

    fn adjust_adaptive(&mut self, ratio: f64, min_eig: f64, saddle_order: usize) {
        let t = self.trust;
        let mut curvature = if min_eig.abs() > 1e-6 {
            t.max_curvature_factor.min(1.0 / min_eig.abs().max(0.1))
        } else {
            1.5
        };
        if saddle_order > 0 && min_eig < -1e-6 {
            curvature *= t.negative_curvature_safety;
        }
        self.radius = if ratio > 0.75 {
            (self.radius * (1.5 * curvature).min(t.max_curvature_factor)).min(t.max)
        } else if ratio > 0.5 {
            (self.radius * (1.1 * curvature).min(1.5)).min(t.max)
        } else if ratio > 0.25 {
            if curvature > 1.2 {
                (self.radius * 1.05).min(t.max)
            } else {
                self.radius
            }
        } else if ratio > 0.1 {
            (self.radius * 0.5).max(t.min)
        } else {
            (self.radius * 0.25).max(t.min)
        };
        self.radius = self.radius.clamp(t.min, t.max);
    }

    fn adjust_trust_radius(
        &mut self,
        actual: f64,
        predicted: f64,
        min_eig: f64,
        g_norm: f64,
        saddle_order: usize,
    ) {
        if predicted.abs() < 1e-10 {
            return;
        }
        let ratio = actual / predicted;
        if g_norm < self.trust.adaptive_gradient_threshold {
            self.adjust_adaptive(ratio, min_eig, saddle_order);
        } else {
            self.adjust_conventional(ratio);
        }
    }

    /// Image projector reflecting the `order` lowest non-zero modes.
    fn image_projector(&self, eigvals: &Col<f64>, eigvecs: &Mat<f64>, order: usize) -> Mat<f64> {
        let n = eigvals.nrows();
        let mut p = Mat::<f64>::identity(n, n);
        let factor = if self.neb_mode { 1.0 } else { 2.0 };
        let mut found = 0;
        for i in 0..n {
            if found >= order {
                break;
            }
            if eigvals[i].abs() > 1e-10 {
                for r in 0..n {
                    for c in 0..n {
                        p[(r, c)] -= factor * eigvecs[(r, i)] * eigvecs[(c, i)];
                    }
                }
                found += 1;
            }
        }
        if found < order {
            Mat::identity(n, n)
        } else {
            p
        }
    }

    /// Image projector reflecting the given modes.
    fn reflector(&self, eigvecs: &Mat<f64>, modes: &[usize]) -> Mat<f64> {
        let n = eigvecs.nrows();
        let factor = if self.neb_mode { 1.0 } else { 2.0 };
        let mut p = Mat::<f64>::identity(n, n);
        for &i in modes {
            for r in 0..n {
                for c in 0..n {
                    p[(r, c)] -= factor * eigvecs[(r, i)] * eigvecs[(c, i)];
                }
            }
        }
        p
    }

    /// [`StepGenerator::propose`] with the modes to reflect chosen by a mode tracker instead of
    /// by their position in the spectrum (`None`: the lowest `saddle_order` modes). The tracker
    /// sees the modes with non-zero eigenvalues only, so rigid-body modes are never followed.
    pub fn propose_following(
        &mut self,
        input: &StepInput<'_>,
        follower: Option<&mut ModeFollowing>,
    ) -> MoptResult<Col<f64>> {
        let g_norm = col_norm(input.gradient);
        let gradient = match input.rigid_body_basis {
            Some(d) => project_out_basis_vec(d, input.gradient),
            None => input.gradient.clone(),
        };
        let mut h = match input.bias_hessian {
            Some(b) => input.hessian + b,
            None => input.hessian.clone(),
        };
        if let Some(d) = input.rigid_body_basis {
            h = project_out_basis_mat(d, &h);
        }
        symmetrize(&mut h);
        let n = h.nrows();
        let (eigvals, eigvecs) = match symmetric_eigen(&h) {
            Ok(e) if e.0.iter().all(|v| v.is_finite()) => e,
            _ => (Col::from_fn(n, |_| 1.0), Mat::identity(n, n)),
        };
        let targets: Option<Vec<usize>> = match follower {
            Some(f) if input.saddle_order > 0 => {
                let live: Vec<usize> = (0..n).filter(|&k| eigvals[k].abs() > 1e-10).collect();
                let sub = Mat::from_fn(n, live.len(), |r, c| eigvecs[(r, live[c])]);
                let picked = if f.is_initialized() {
                    f.matched_indices(&sub, Some(&gradient))?
                } else {
                    f.set_references(&sub)
                };
                Some(picked.into_iter().map(|c| live[c]).collect())
            }
            _ => None,
        };

        if let Some(prev) = self.prev_energy {
            let actual = input.energy.value() - prev;
            Self::push_bounded(&mut self.actual, actual);
            if let Some(&pred) = self.predicted.back() {
                // Curvature of the followed mode, or the lowest one.
                let first = targets.as_ref().and_then(|t| t.first().copied());
                let min_eig = match first {
                    Some(t) => eigvals[t],
                    None if n > 0 => eigvals[0],
                    None => 0.0,
                };
                self.adjust_trust_radius(actual, pred, min_eig, g_norm, input.saddle_order);
            }
        }

        let p = match &targets {
            Some(t) => self.reflector(&eigvecs, t),
            None => self.image_projector(&eigvals, &eigvecs, input.saddle_order),
        };
        let mut h_star = &p * &h;
        symmetrize(&mut h_star);
        let g_star = &p * &gradient;
        let (ev_star, vec_star) = match symmetric_eigen(&h_star) {
            Ok(e) if e.0.iter().all(|v| v.is_finite()) => e,
            _ => (Col::from_fn(n, |_| 1.0), Mat::identity(n, n)),
        };
        let keep: Vec<usize> = (0..n)
            .filter(|&k| ev_star[k].abs() >= self.small_eigval)
            .collect();
        let lam: Vec<f64> = keep.iter().map(|&k| ev_star[k]).collect();
        let gt: Vec<f64> = keep
            .iter()
            .map(|&k| (0..n).map(|r| vec_star[(r, k)] * g_star[r]).sum())
            .collect();
        // MultiOptPy's messages of this step.
        let rule = "=".repeat(50);
        self.debug = vec![
            rule.clone(),
            format!("RS-I-RFO Iteration {}", self.count),
            rule,
            format!("Gradient norm: {g_norm:.6}"),
        ];
        self.count += 1;
        let sizes: Vec<f64> = eigvals
            .iter()
            .map(|v| v.abs())
            .filter(|v| *v > 1e-10)
            .collect();
        let largest = sizes.iter().copied().fold(0.0, f64::max);
        let smallest = sizes.iter().copied().fold(f64::INFINITY, f64::min);
        self.debug.push(if sizes.is_empty() {
            "Condition number of Hessian: N/A, Ill-conditioned: True".to_string()
        } else {
            let condition = largest / smallest;
            format!(
                "Condition number of Hessian: {condition:.2}, Ill-conditioned: {}",
                if condition > ILL_CONDITIONED {
                    "True"
                } else {
                    "False"
                }
            )
        });
        self.debug.push(format!(
            "Found {} negative eigenvalues (target for saddle order: {})",
            eigvals.iter().filter(|v| **v < -1e-10).count(),
            input.saddle_order
        ));
        let free = col_norm(&solve_rfo(&lam, &gt, 1.0).0);
        self.debug.push(format!(
            "Initial step with alpha=1.000000 has norm={free:.6}"
        ));
        if free > self.radius {
            self.debug.push(
                "Initial step exceeds trust radius, optimizing alpha to match radius..."
                    .to_string(),
            );
        }
        let step_t = rs_step(&lam, &gt, self.radius, &self.rfo);
        let mut step: Col<f64> = Col::zeros(n);
        for (j, &k) in keep.iter().enumerate() {
            for r in 0..n {
                step[r] += vec_star[(r, k)] * step_t[j];
            }
        }
        if !step.iter().all(|v| v.is_finite()) {
            step = -&gradient;
            let sn = col_norm(&step);
            if sn > self.radius {
                step *= self.radius / sn;
            }
        }

        let predicted = col_dot(&gradient, &step) + 0.5 * col_dot(&step, &(&h * &step));
        self.debug
            .push(format!("Final norm(step)={:.6}", col_norm(&step)));
        self.debug
            .push(format!("Predicted energy change: {predicted:.6}"));
        Self::push_bounded(&mut self.predicted, predicted);
        self.prev_energy = Some(input.energy.value());
        Ok(step)
    }
}

impl Default for Rsirfo {
    fn default() -> Self {
        Self::new(RsirfoTrust::for_saddle_order(0, None, None))
    }
}

impl StepGenerator for Rsirfo {
    fn name(&self) -> &str {
        "rsirfo"
    }

    fn uses_hessian(&self) -> bool {
        true
    }

    fn propose(&mut self, input: &StepInput<'_>) -> MoptResult<Col<f64>> {
        self.propose_following(input, None)
    }

    fn take_debug_notes(&mut self) -> Vec<String> {
        std::mem::take(&mut self.debug)
    }

    fn reset(&mut self) {
        *self = Self::new(self.trust);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input<'a>(x: &'a Col<f64>, g: &'a Col<f64>, h: &'a Mat<f64>, order: usize) -> StepInput<'a> {
        StepInput {
            iteration: 0,
            cartesian: x,
            energy: crate::domain::Quantity::zero(),
            gradient: g,
            raw_gradient: g,
            previous: None,
            hessian: h,
            bias_hessian: None,
            rigid_body_basis: None,
            project_updates: true,
            saddle_order: order,
            system: None,
            cartesian_coordinates: true,
            frame: None,
            cartesian_gradient: None,
        }
    }

    #[test]
    fn minimization_step_on_quadratic_is_newton_like() {
        let x = Col::zeros(2);
        let g = Col::from_fn(2, |i| [0.01, -0.02][i]);
        let h = Mat::from_fn(2, 2, |i, j| if i == j { [1.0, 2.0][i] } else { 0.0 });
        let mut opt = Rsirfo::default();
        let s = opt.propose(&input(&x, &g, &h, 0)).unwrap();
        assert!((s[0] + 0.01).abs() < 1e-3 && (s[1] - 0.01).abs() < 1e-3);
    }

    #[test]
    fn a_minimization_step_never_climbs_its_model() {
        // Random Hessians with up to two negative curvatures, gradients whose components span
        // many orders of magnitude, and trust radii from 0.05 to 0.5 Bohr: the step of a
        // minimization lowers its quadratic model.
        use crate::core_infra::rng::Rng;
        use crate::math::linalg::orthonormalize_columns;
        let mut rng = Rng::from_seed(7);
        let mut worst: Option<(f64, String)> = None;
        let mut longest: Option<(f64, String)> = None;
        let (mut rises, mut long) = (0, 0);
        for case in 0..1500 {
            let n = 3 + rng.gen_range_usize(22);
            let cols: Vec<Col<f64>> = (0..n)
                .map(|_| Col::from_fn(n, |_| rng.standard_normal()))
                .collect();
            let q = orthonormalize_columns(n, &cols, 1e-10);
            if q.ncols() < n {
                continue;
            }
            let negatives = rng.gen_range_usize(3);
            let eig: Vec<f64> = (0..n)
                .map(|k| {
                    let size = libm::pow(10.0, rng.uniform(-4.0, 0.3));
                    if k < negatives {
                        -size
                    } else {
                        size
                    }
                })
                .collect();
            let h = Mat::from_fn(n, n, |r, c| {
                (0..n).map(|k| q[(r, k)] * eig[k] * q[(c, k)]).sum()
            });
            let comps: Vec<f64> = (0..n)
                .map(|_| {
                    let sign = if rng.next_f64() < 0.5 { -1.0 } else { 1.0 };
                    sign * libm::pow(10.0, rng.uniform(-9.0, -1.5))
                })
                .collect();
            let g = Col::from_fn(n, |r| (0..n).map(|k| q[(r, k)] * comps[k]).sum());
            let radius = [0.05, 0.1, 0.15625, 0.3125, 0.5][case % 5];
            let mut opt = Rsirfo::new(RsirfoTrust {
                initial: radius,
                ..RsirfoTrust::for_saddle_order(0, None, None)
            });
            let x = Col::zeros(n);
            let s = opt.propose(&input(&x, &g, &h, 0)).unwrap();
            let model = col_dot(&g, &s) + 0.5 * col_dot(&s, &(&h * &s));
            // The restricted step lands within the tolerance of its search on the radius.
            let over = col_norm(&s) / radius;
            if col_norm(&s) > radius + RfoParams::default().step_norm_tolerance + 1e-12 {
                long += 1;
                if longest.as_ref().is_none_or(|(w, _)| over > *w) {
                    longest = Some((
                        over,
                        format!("case {case}: n {n}, eig {eig:?}, g {comps:?}, radius {radius}"),
                    ));
                }
            }
            if model > 0.0 {
                rises += 1;
                if worst.as_ref().is_none_or(|(w, _)| model > *w) {
                    worst = Some((
                        model,
                        format!("case {case}: n {n}, eig {eig:?}, g {comps:?}, radius {radius}"),
                    ));
                }
            }
        }
        assert!(
            worst.is_none() && longest.is_none(),
            "{rises} step(s) climb the model, worst {worst:?}; {long} step(s) longer than the \
             radius, longest {longest:?}"
        );
    }

    #[test]
    fn saddle_step_goes_uphill_along_negative_mode() {
        let x = Col::zeros(2);
        let g = Col::from_fn(2, |i| [0.02, 0.02][i]);
        let h = Mat::from_fn(2, 2, |i, j| if i == j { [-1.0, 1.0][i] } else { 0.0 });
        let mut opt = Rsirfo::new(RsirfoTrust::for_saddle_order(1, None, None));
        let s = opt.propose(&input(&x, &g, &h, 1)).unwrap();
        // Along mode 0 (negative curvature) the step follows +g (uphill); along mode 1, -g.
        assert!(s[0] > 0.0 && s[1] < 0.0);
        assert!(col_norm(&s) <= 0.1 + 1e-9);
    }
}
