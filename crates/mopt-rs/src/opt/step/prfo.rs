//! Restricted-step partitioned rational function optimization (RS-P-RFO) for saddle points.
//!
//! Banerjee, Adams, Simons, Shepard, J. Phys. Chem. 1985, 89, 52 (P-RFO); restricted-step form
//! and trust-radius control as described by Wu and Wang, J. Chem. Theory Comput. 2025 (CCQN),
//! eqn (11)-(17) and (19), which follow Sella (Hermes et al., J. Chem. Theory Comput. 2019, 15,
//! 6536; 2022, 18, 6974).
//!
//! The Hessian is diagonalized; the `mu` lowest modes are maximized and the others minimized by
//! two rational function problems sharing the scale `a` (eqn (16), (17)). The step length grows
//! monotonically with `a`, so `a` is found by bisection such that `||s|| = Delta` unless the
//! Newton step already lies inside the trust region. The trust radius follows eqn (19): with
//! `rho = predicted / actual` energy change, it grows by `sigma_inc` when
//! `1/rho_inc <= rho <= rho_inc`, shrinks by `sigma_dec` when `rho <= 1/rho_dec` or
//! `rho >= rho_dec`, and is kept otherwise.
//!
//! MultiOptPy's `rsprfo_*` methods (`EnhancedRSPRFO`) are a different algorithm and are ported
//! separately.
//!
//! MultiOptPy: no counterpart (new in mopt-rs); MultiOptPy's `Optimizer/rsprfo.py:EnhancedRSPRFO`
//!  is a different algorithm.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Col;

use super::{StepGenerator, StepInput};
use crate::error::{MoptError, MoptResult};
use crate::math::linalg::{col_dot, col_norm, symmetric_eigen, symmetrize};
use crate::math::projection::{project_out_basis_mat, project_out_basis_vec};

/// Modes with `|lambda|` below this are treated as null modes (rigid-body motion) and skipped.
const NULL_MODE: f64 = 1e-6;

/// Extreme eigenvalue of the bordered matrix `[[diag(lam), h], [h^T, 0]]`: the smallest
/// (`largest = false`) or the largest. Solved from the secular equation
/// `nu = sum_i h_i^2 / (nu - lam_i)` by bisection inside its bracketing interval.
fn bordered_extreme(lam: &[f64], h: &[f64], largest: bool) -> f64 {
    let h2: Vec<f64> = h.iter().map(|v| v * v).collect();
    let h_norm = h2.iter().sum::<f64>().sqrt();
    let active = lam
        .iter()
        .zip(&h2)
        .filter(|(_, w)| **w > 0.0)
        .map(|(l, _)| *l);
    if h_norm == 0.0 {
        return 0.0;
    }
    // phi(nu) = nu - sum h^2 / (nu - lam) is increasing between poles.
    let phi = |nu: f64| {
        nu - lam
            .iter()
            .zip(&h2)
            .filter(|(_, w)| **w > 0.0)
            .map(|(l, w)| w / (nu - l))
            .sum::<f64>()
    };
    let (mut lo, mut hi) = if largest {
        let base = active.fold(0.0_f64, f64::max);
        (base, base + h_norm)
    } else {
        let base = active.fold(0.0_f64, f64::min);
        (base - h_norm, base)
    };
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if mid <= lo || mid >= hi {
            break;
        }
        if phi(mid) < 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Step components `-g_i / (lam_i - nu)` (zero where `g_i` vanishes; infinite when the shift
/// coincides with the eigenvalue to machine precision, so that such a scale is rejected).
fn shifted_newton(lam: &[f64], g: &[f64], nu: f64) -> Vec<f64> {
    lam.iter()
        .zip(g)
        .map(|(l, gi)| {
            let d = l - nu;
            if *gi == 0.0 {
                0.0
            } else if d == 0.0 {
                f64::INFINITY
            } else {
                -gi / d
            }
        })
        .collect()
}

/// P-RFO step for scale `a` (eqn (14)-(17)): the first `n_max` modes are maximized.
fn prfo_step_at(eigvals: &[f64], g: &[f64], n_max: usize, a: f64) -> Vec<f64> {
    let (lam_max, lam_min) = eigvals.split_at(n_max);
    let (g_max, g_min) = g.split_at(n_max);
    let h_max: Vec<f64> = g_max.iter().map(|v| v / a).collect();
    let h_min: Vec<f64> = g_min.iter().map(|v| v / a).collect();
    let nu_max = bordered_extreme(lam_max, &h_max, true);
    let nu_min = bordered_extreme(lam_min, &h_min, false);
    let mut s = shifted_newton(lam_max, g_max, nu_max);
    s.extend(shifted_newton(lam_min, g_min, nu_min));
    s
}

fn norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Restricted-step P-RFO step in the Hessian eigenbasis (`eigvals` ascending, `g` the gradient
/// in that basis), maximizing along the first `n_max` modes, with `||s|| <= radius`.
pub fn partitioned_rfo_step(eigvals: &[f64], g: &[f64], n_max: usize, radius: f64) -> Col<f64> {
    let n_max = n_max.min(eigvals.len());
    let (a_lo, a_hi) = (1e-12_f64, 1e12_f64);
    let far = prfo_step_at(eigvals, g, n_max, a_hi);
    let s = if norm(&far).is_finite() && norm(&far) <= radius {
        far
    } else {
        // ||s(a)|| increases with a: bisection on log a.
        let (mut lo, mut hi) = (a_lo.ln(), a_hi.ln());
        let mut best = prfo_step_at(eigvals, g, n_max, a_lo);
        for _ in 0..200 {
            let mid = 0.5 * (lo + hi);
            let s = prfo_step_at(eigvals, g, n_max, mid.exp());
            let n = norm(&s);
            if n.is_finite() && n <= radius {
                lo = mid;
                best = s;
            } else {
                hi = mid;
            }
            if hi - lo < 1e-12 {
                break;
            }
        }
        // Land exactly on the boundary.
        let n = norm(&best);
        if n > 0.0 {
            best.iter().map(|v| v * radius / n).collect()
        } else {
            best
        }
    };
    Col::from_fn(s.len(), |i| s[i])
}

/// Trust-radius policy of eqn (19).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PrfoTrust {
    /// Initial radius `Delta_0` (Bohr).
    pub initial: f64,
    /// Lower bound `delta_min`.
    pub min: f64,
    /// Upper bound `delta_max`.
    pub max: f64,
    /// `rho_inc > 1`.
    pub rho_inc: f64,
    /// `rho_dec > rho_inc`.
    pub rho_dec: f64,
    /// `sigma_inc > 1`.
    pub sigma_inc: f64,
    /// `sigma_dec < 1`.
    pub sigma_dec: f64,
}

impl Default for PrfoTrust {
    /// Ratio thresholds and factors of Sella's saddle-point settings, on which CCQN builds; the
    /// radii are in Bohr.
    fn default() -> Self {
        Self {
            initial: 0.1,
            min: 0.005,
            max: 0.3,
            rho_inc: 1.035,
            rho_dec: 5.0,
            sigma_inc: 1.15,
            sigma_dec: 0.65,
        }
    }
}

impl PrfoTrust {
    /// Check `1 < rho_inc < rho_dec`, `sigma_inc > 1`, `0 < sigma_dec < 1`, and the radii.
    pub fn validate(&self) -> MoptResult<()> {
        let ok = 1.0 < self.rho_inc
            && self.rho_inc < self.rho_dec
            && self.sigma_inc > 1.0
            && 0.0 < self.sigma_dec
            && self.sigma_dec < 1.0
            && 0.0 < self.min
            && self.min <= self.initial
            && self.initial <= self.max;
        if ok {
            Ok(())
        } else {
            Err(MoptError::config(format!(
                "invalid RS-P-RFO trust settings: {self:?}"
            )))
        }
    }

    /// Radius after a step with ratio `rho = predicted / actual` (eqn (19)).
    pub fn next_radius(&self, radius: f64, rho: f64) -> f64 {
        if (1.0 / self.rho_inc..=self.rho_inc).contains(&rho) {
            (self.sigma_inc * radius).min(self.max)
        } else if rho <= 1.0 / self.rho_dec || rho >= self.rho_dec || !rho.is_finite() {
            (self.sigma_dec * radius).max(self.min)
        } else {
            radius
        }
    }
}

/// RS-P-RFO step generator.
#[derive(Debug, Clone)]
pub struct RestrictedPrfo {
    trust: PrfoTrust,
    radius: f64,
    /// Energy and predicted change of this generator's last step.
    last: Option<(f64, f64)>,
}

impl RestrictedPrfo {
    /// Generator with the given trust policy.
    pub fn new(trust: PrfoTrust) -> MoptResult<Self> {
        trust.validate()?;
        Ok(Self {
            trust,
            radius: trust.initial,
            last: None,
        })
    }

    /// Current trust radius.
    pub fn trust_radius(&self) -> f64 {
        self.radius
    }

    /// Forget the last step, so that the next call does not rate a step taken by another
    /// method.
    pub fn forget_last_step(&mut self) {
        self.last = None;
    }
}

impl StepGenerator for RestrictedPrfo {
    fn name(&self) -> &str {
        "rs_prfo"
    }

    fn uses_hessian(&self) -> bool {
        true
    }

    fn propose(&mut self, input: &StepInput<'_>) -> MoptResult<Col<f64>> {
        if let Some((e_prev, predicted)) = self.last {
            let actual = input.energy.value() - e_prev;
            if predicted != 0.0 || actual != 0.0 {
                self.radius = self.trust.next_radius(self.radius, predicted / actual);
            }
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
        let (vals, vecs) = symmetric_eigen(&h)?;
        let n = h.nrows();
        let keep: Vec<usize> = (0..n).filter(|&k| vals[k].abs() >= NULL_MODE).collect();
        let lam: Vec<f64> = keep.iter().map(|&k| vals[k]).collect();
        let gt: Vec<f64> = keep
            .iter()
            .map(|&k| (0..n).map(|r| vecs[(r, k)] * gradient[r]).sum())
            .collect();
        let st = partitioned_rfo_step(&lam, &gt, input.saddle_order, self.radius);
        let mut step = Col::zeros(n);
        for (j, &k) in keep.iter().enumerate() {
            for r in 0..n {
                step[r] += vecs[(r, k)] * st[j];
            }
        }
        let predicted = col_dot(&gradient, &step) + 0.5 * col_dot(&step, &(&h * &step));
        self.last = Some((input.energy.value(), predicted));
        if !step.iter().all(|v| v.is_finite())
            || col_norm(&step) == 0.0 && col_norm(&gradient) > 0.0
        {
            return Err(MoptError::linalg("RS-P-RFO produced no usable step"));
        }
        Ok(step)
    }

    fn reset(&mut self) {
        self.radius = self.trust.initial;
        self.last = None;
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn newton_step_inside_the_trust_region() {
        // Saddle point within reach: P-RFO reduces to -B^-1 g.
        let lam = [-0.5, 1.0, 2.0];
        let g = [0.01, 0.02, -0.04];
        let s = partitioned_rfo_step(&lam, &g, 1, 1.0);
        for i in 0..3 {
            assert!((s[i] + g[i] / lam[i]).abs() < 1e-8, "{i}: {}", s[i]);
        }
    }

    #[test]
    fn restricted_step_lies_on_the_boundary_and_has_saddle_signs() {
        // Positive-definite model (potential well): uphill along mode 0, downhill elsewhere.
        let lam = [0.2, 1.0, 2.0];
        let g = [0.3, 0.2, -0.4];
        let s = partitioned_rfo_step(&lam, &g, 1, 0.1);
        assert!((col_norm(&s) - 0.1).abs() < 1e-10);
        assert!(s[0] * g[0] > 0.0, "maximized mode goes uphill");
        assert!(s[1] * g[1] < 0.0 && s[2] * g[2] < 0.0);
    }

    #[test]
    fn step_norm_grows_with_a() {
        let lam = [-0.3, 0.5, 1.5];
        let g = [0.2, -0.1, 0.3];
        let norms: Vec<f64> = [1e-2, 1e-1, 1.0, 10.0, 100.0]
            .iter()
            .map(|&a| norm(&prfo_step_at(&lam, &g, 1, a)))
            .collect();
        assert!(norms.windows(2).all(|w| w[1] > w[0]), "{norms:?}");
    }

    #[test]
    fn trust_policy_of_eqn_19() {
        let t = PrfoTrust::default();
        assert!((t.next_radius(0.1, 1.0) - 0.115).abs() < 1e-12);
        assert!((t.next_radius(0.1, 10.0) - 0.065).abs() < 1e-12);
        assert!((t.next_radius(0.1, 0.1) - 0.065).abs() < 1e-12);
        assert!((t.next_radius(0.1, 2.0) - 0.1).abs() < 1e-12);
    }
}
