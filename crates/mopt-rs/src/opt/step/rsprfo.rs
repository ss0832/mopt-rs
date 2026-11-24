//! Restricted-step partitioned RFO with eigenvalue shifting and mode following
//! (MultiOptPy's `rsprfo_<updater>` methods).
//!
//! Banerjee, Adams, Simons, and Shepard, J. Phys. Chem. 1985, 89, 52 (P-RFO); Baker, J. Comput.
//! Chem. 1986, 7, 385 (mode following); Besalu and Bofill, Theor. Chem. Acc. 1998, 100, 265
//! (restricted step). Per iteration:
//! 1. rate the last step by the ratio of the actual to the predicted energy change: below 0.25
//!    the radius becomes a quarter of the step length, above 0.75 it doubles if the step reached
//!    the boundary;
//! 2. diagonalize the Hessian (rigid-body modes projected out and skipped);
//! 3. choose the modes to maximize: the mode with the largest overlap with the mode followed
//!    last (if above 0.5; otherwise, among the modes overlapping by more than 0.3, the one with
//!    the largest weight `overlap^2` times 1 for negative and 0.1 for positive curvature;
//!    otherwise the lowest mode), completed by the lowest remaining modes;
//! 4. shift the eigenvalues to the wanted inertia: maximized modes to at most `-0.001`, the
//!    others to `0.001` when below `1e-6` (for minimizations all eigenvalues are shifted
//!    uniformly so that the lowest is `0.001`);
//! 5. take the P-RFO step restricted to the radius ([`super::prfo::partitioned_rfo_step`]); for
//!    gradients below `0.001` the radius is tied to the gradient norm.
//!
//! Differences from MultiOptPy's `EnhancedRSPRFO` (see `docs/source/migration_optimizers.md`): the
//! restricted step is found by bisection on the scale of the augmented Hessian instead of a
//! Newton iteration with an approximate derivative, rigid-body modes are projected out and
//! skipped instead of being shifted to `0.001`, the last step is rated with the prediction made
//! when it was proposed, and the cap of the step at 50 times the gradient norm is dropped.
//!
//! MultiOptPy: `Optimizer/rsprfo.py:EnhancedRSPRFO`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Col;

use super::prfo::partitioned_rfo_step;
use super::{StepGenerator, StepInput};
use crate::error::MoptResult;
use crate::math::linalg::{col_dot, col_norm, symmetric_eigen, symmetrize};
use crate::math::projection::{project_out_basis_mat, project_out_basis_vec};

/// Modes with `|lambda|` below this are null modes (rigid-body motion) and skipped.
const NULL_MODE: f64 = 1e-8;

/// Parameters of the `rsprfo_*` methods.
#[derive(Debug, Clone, Copy)]
pub struct RsprfoParams {
    /// Initial trust radius.
    pub trust_initial: f64,
    /// Smallest trust radius.
    pub trust_min: f64,
    /// Largest trust radius.
    pub trust_max: f64,
    /// Lowest eigenvalue allowed for minimized modes.
    pub min_positive_eigval: f64,
    /// Highest eigenvalue allowed for maximized modes.
    pub min_negative_eigval: f64,
    /// Gradient norm below which the radius is tied to the gradient.
    pub gradient_threshold: f64,
    /// Radius at the gradient threshold, as a fraction of the largest radius.
    pub gradient_trust_coupling: f64,
    /// Overlap above which the previous mode is considered found again.
    pub overlap_threshold: f64,
    /// Overlap above which a mode is a candidate when the previous mode has mixed.
    pub mixing_threshold: f64,
}

impl RsprfoParams {
    /// MultiOptPy's defaults: radius 0.5 (minimization) or 0.1 (saddle search), minimum 0.01.
    pub fn for_saddle_order(saddle_order: usize, max: Option<f64>, min: Option<f64>) -> Self {
        let default = if saddle_order == 0 { 0.5 } else { 0.1 };
        let trust_max = max.unwrap_or(default);
        let trust_min = min.unwrap_or(0.01).min(trust_max);
        Self {
            trust_initial: default.clamp(trust_min, trust_max),
            trust_min,
            trust_max,
            min_positive_eigval: 1e-3,
            min_negative_eigval: -1e-3,
            gradient_threshold: 1e-3,
            gradient_trust_coupling: 0.5,
            overlap_threshold: 0.5,
            mixing_threshold: 0.3,
        }
    }
}

/// Eigenvalues shifted to the inertia of a saddle point of order `n_max` (the first `n_max`
/// entries are the maximized modes) or, for `n_max = 0`, of a minimum.
pub fn shifted_eigenvalues(eigvals: &[f64], n_max: usize, p: &RsprfoParams) -> Vec<f64> {
    if n_max == 0 {
        let lowest = eigvals.iter().copied().fold(f64::INFINITY, f64::min);
        let shift = (p.min_positive_eigval - lowest).max(0.0);
        return eigvals.iter().map(|l| l + shift).collect();
    }
    eigvals
        .iter()
        .enumerate()
        .map(|(i, &l)| {
            if i < n_max {
                l.min(p.min_negative_eigval)
            } else if l < 1e-6 {
                p.min_positive_eigval
            } else {
                l
            }
        })
        .collect()
}

/// `rsprfo_*` step generator.
#[derive(Debug, Clone)]
pub struct Rsprfo {
    params: RsprfoParams,
    radius: f64,
    /// Eigenvector of the mode followed last.
    ts_mode: Option<Col<f64>>,
    /// Energy at the last point and the change predicted for the step from it.
    last: Option<(f64, f64)>,
}

impl Rsprfo {
    /// New generator.
    pub fn new(params: RsprfoParams) -> Self {
        Self {
            params,
            radius: params.trust_initial,
            ts_mode: None,
            last: None,
        }
    }

    /// Current trust radius.
    pub fn trust_radius(&self) -> f64 {
        self.radius
    }

    /// Radius used for a gradient of norm `g_norm`: near convergence it shrinks in proportion
    /// to the gradient (never below the minimum, never above the current radius).
    pub fn effective_radius(&self, g_norm: f64) -> f64 {
        let p = &self.params;
        if g_norm < p.gradient_threshold {
            (p.gradient_trust_coupling * g_norm / p.gradient_threshold * p.trust_max)
                .max(p.trust_min)
                .min(self.radius)
        } else {
            self.radius
        }
    }

    /// Positions (in `0..overlaps.len()`, ascending eigenvalues) of the modes to maximize.
    fn select_modes(
        &mut self,
        eigvals: &[f64],
        overlaps: Option<Vec<f64>>,
        order: usize,
    ) -> Vec<usize> {
        let m = eigvals.len();
        let order = order.min(m);
        if order == 0 {
            return Vec::new();
        }
        let lead = match overlaps {
            None => 0,
            Some(ov) => {
                let best = (0..m).max_by(|&a, &b| ov[a].total_cmp(&ov[b])).unwrap_or(0);
                if ov[best] > self.params.overlap_threshold {
                    best
                } else {
                    // The mode has mixed with others: prefer strong overlap and negative
                    // curvature.
                    (0..m)
                        .filter(|&i| ov[i] > self.params.mixing_threshold)
                        .max_by(|&a, &b| {
                            let w =
                                |i: usize| ov[i] * ov[i] * if eigvals[i] < 0.0 { 1.0 } else { 0.1 };
                            w(a).total_cmp(&w(b))
                        })
                        .unwrap_or(0)
                }
            }
        };
        let mut modes = vec![lead];
        modes.extend((0..m).filter(|&i| i != lead).take(order - 1));
        modes
    }
}

impl StepGenerator for Rsprfo {
    fn name(&self) -> &str {
        "rsprfo"
    }

    fn uses_hessian(&self) -> bool {
        true
    }

    fn propose(&mut self, input: &StepInput<'_>) -> MoptResult<Col<f64>> {
        let p = self.params;
        if let (Some((e_prev, predicted)), Some(prev)) = (self.last.take(), input.previous) {
            let actual = input.energy.value() - e_prev;
            let ratio = if predicted.abs() < 1e-14 {
                if actual.abs() < 1e-14 {
                    1.0
                } else {
                    0.0
                }
            } else {
                actual / predicted
            };
            let len = col_norm(&prev.step);
            if ratio.is_nan() || ratio < 0.25 {
                self.radius = (0.25 * len).max(p.trust_min);
            } else if ratio > 0.75 && len >= 0.95 * self.radius {
                self.radius = (2.0 * self.radius).min(p.trust_max);
            }
            self.radius = self.radius.clamp(p.trust_min, p.trust_max);
        }
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
        let (vals, vecs) = symmetric_eigen(&h)?;
        let live: Vec<usize> = (0..n).filter(|&k| vals[k].abs() >= NULL_MODE).collect();
        let lam: Vec<f64> = live.iter().map(|&k| vals[k]).collect();
        let column = |k: usize| Col::from_fn(n, |r| vecs[(r, k)]);
        let overlaps = self.ts_mode.as_ref().map(|t| {
            live.iter()
                .map(|&k| col_dot(&column(k), t).abs())
                .collect::<Vec<f64>>()
        });
        let maxed = self.select_modes(&lam, overlaps, input.saddle_order);
        if let Some(&lead) = maxed.first() {
            self.ts_mode = Some(column(live[lead]));
        }
        // Maximized modes first, the others in ascending order.
        let order: Vec<usize> = maxed
            .iter()
            .copied()
            .chain((0..live.len()).filter(|i| !maxed.contains(i)))
            .collect();
        let raw: Vec<f64> = order.iter().map(|&i| lam[i]).collect();
        let shifted = shifted_eigenvalues(&raw, maxed.len(), &p);
        let gt: Vec<f64> = order
            .iter()
            .map(|&i| col_dot(&column(live[i]), &gradient))
            .collect();
        let radius = self.effective_radius(col_norm(&gradient));
        let st = partitioned_rfo_step(&shifted, &gt, maxed.len(), radius);
        let mut step: Col<f64> = Col::zeros(n);
        let mut predicted = 0.0;
        for (j, &i) in order.iter().enumerate() {
            let c = st[j];
            predicted += gt[j] * c + 0.5 * shifted[j] * c * c;
            for r in 0..n {
                step[r] += c * vecs[(r, live[i])];
            }
        }
        self.last = Some((input.energy.value(), predicted));
        Ok(step)
    }

    fn reset(&mut self) {
        *self = Self::new(self.params);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::opt::stepper::plain_input;
    use faer::Mat;

    fn diag(d: &[f64]) -> Mat<f64> {
        Mat::from_fn(d.len(), d.len(), |i, j| if i == j { d[i] } else { 0.0 })
    }

    #[test]
    fn eigenvalues_are_shifted_to_the_wanted_inertia() {
        let p = RsprfoParams::for_saddle_order(1, None, None);
        // Saddle: the maximized mode becomes negative, other non-positive modes positive.
        assert_eq!(
            shifted_eigenvalues(&[0.2, -0.3, 1e-9, 0.5], 1, &p),
            vec![-1e-3, 1e-3, 1e-3, 0.5]
        );
        assert_eq!(shifted_eigenvalues(&[-0.4, 0.5], 1, &p), vec![-0.4, 0.5]);
        // Minimum: uniform shift so that the lowest eigenvalue is 0.001.
        let s = shifted_eigenvalues(&[-0.2, 0.5], 0, &p);
        assert!((s[0] - 1e-3).abs() < 1e-15 && (s[1] - 0.701).abs() < 1e-15);
        assert_eq!(shifted_eigenvalues(&[0.3, 0.5], 0, &p), vec![0.3, 0.5]);
    }

    #[test]
    fn step_goes_toward_the_stationary_point_and_respects_the_radius() {
        let x = Col::zeros(3);
        // Saddle at +(0.02 / 0.5, ...): uphill along mode 0, downhill along the others.
        let h = diag(&[-0.5, 1.0, 2.0]);
        let g = Col::from_fn(3, |i| [0.02, 0.02, -0.04][i]);
        let mut opt = Rsprfo::new(RsprfoParams::for_saddle_order(1, None, None));
        let s = opt
            .propose(&plain_input(0, &x, 0.0, &g, &h, None, 1))
            .unwrap();
        assert!(s[0] > 0.0 && s[1] < 0.0 && s[2] > 0.0, "{s:?}");
        // The Newton step (0.04, -0.02, 0.02) lies inside the radius of 0.1.
        assert!((s[0] - 0.04).abs() < 1e-6 && (s[1] + 0.02).abs() < 1e-6);
        // A large gradient: the step has the length of the trust radius.
        let big = &g * 100.0;
        let mut opt = Rsprfo::new(RsprfoParams::for_saddle_order(1, None, None));
        let s = opt
            .propose(&plain_input(0, &x, 0.0, &big, &h, None, 1))
            .unwrap();
        assert!((col_norm(&s) - 0.1).abs() < 1e-9);
        // Minimization on a positive-definite Hessian is a Newton step downhill.
        let h = diag(&[0.5, 1.0, 2.0]);
        let mut opt = Rsprfo::new(RsprfoParams::for_saddle_order(0, None, None));
        let s = opt
            .propose(&plain_input(0, &x, 0.0, &g, &h, None, 0))
            .unwrap();
        assert!(col_dot(&s, &g) < 0.0);
        assert!((s[0] + 0.04).abs() < 1e-4, "{s:?}");
    }

    #[test]
    fn the_followed_mode_survives_a_change_of_order_and_the_radius_follows_the_gradient() {
        let x = Col::zeros(2);
        let g = Col::from_fn(2, |i| [0.02, 0.01][i]);
        let mut opt = Rsprfo::new(RsprfoParams::for_saddle_order(1, None, None));
        let s = opt
            .propose(&plain_input(0, &x, 0.0, &g, &diag(&[-0.5, 1.0]), None, 1))
            .unwrap();
        assert!(s[0] > 0.0 && s[1] < 0.0);
        // y becomes the lowest mode; x is still maximized.
        let s = opt
            .propose(&plain_input(1, &x, 0.0, &g, &diag(&[-0.5, -1.0]), None, 1))
            .unwrap();
        assert!(s[0] > 0.0 && s[1] < 0.0, "{s:?}");
        // Tiny gradients shrink the radius: 0.5 * (1e-4 / 1e-3) * 0.1 = 0.005 -> minimum 0.01.
        assert_eq!(opt.effective_radius(1e-4), 0.01);
        assert_eq!(opt.effective_radius(1e-2), opt.trust_radius());
        assert!((opt.effective_radius(8e-4) - 0.04).abs() < 1e-15);
    }
}
