//! Trust-region image minimization (TRIM; Helgaker, Chem. Phys. Lett. 1991, 182, 503).
//!
//! When the proposed step is longer than the trust radius, it is replaced by the minimizer of
//! the quadratic model of the image function inside the trust region. In the eigenbasis of the
//! Hessian the image function has the eigenvalues and gradient components of the `saddle_order`
//! lowest modes reversed (`l'_i = -l_i`, `g'_i = -g_i`), and the step is
//! `z_i = -g'_i / (l'_i - mu)` with the level shift `mu <= min(0, l'_min)` such that
//! `||z|| = Delta` (`mu = 0` when the Newton step of the image function fits). The norm grows
//! monotonically with `mu` below the lowest eigenvalue, so the shift is bracketed and found by
//! Brent's method. Steps inside the radius pass unchanged; rigid-body modes are left out.
//!
//! MultiOptPy starts an unsafeguarded Newton iteration for `mu` at zero, which can end between
//! two eigenvalues (a step that is not a minimizer of the model), reflects rigid-body modes when
//! the Hessian has no negative eigenvalue, and falls back to scaling the Newton step.
//!
//! MultiOptPy: `Optimizer/trim.py:TRIM`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Col;

use super::StepEnhancer;
use crate::error::MoptResult;
use crate::math::linalg::{col_norm, symmetric_eigen, symmetrize};
use crate::math::projection::{project_out_basis_mat, project_out_basis_vec};
use crate::math::roots::brentq;
use crate::opt::step::StepInput;

/// Modes with `|lambda|` below this are null modes.
const NULL_MODE: f64 = 1e-8;

/// Level-shifted Newton step `z_i = -g_i / (l_i - mu)` of length at most `radius`, minimizing
/// `g.z + sum_i l_i z_i^2 / 2` in the ball; returns the step and `mu <= min(0, min_i l_i)`.
pub fn level_shifted_step(eigvals: &[f64], g: &[f64], radius: f64) -> (Vec<f64>, f64) {
    let step_at = |mu: f64| -> Vec<f64> {
        eigvals
            .iter()
            .zip(g)
            .map(|(l, gi)| if *gi == 0.0 { 0.0 } else { -gi / (l - mu) })
            .collect()
    };
    let norm = |z: &[f64]| libm::sqrt(z.iter().map(|v| v * v).sum::<f64>());
    let g_norm = norm(g);
    // Lowest eigenvalue with a gradient component (the first pole below which all terms are
    // positive).
    let lowest = eigvals
        .iter()
        .zip(g)
        .filter(|(_, gi)| **gi != 0.0)
        .map(|(l, _)| *l)
        .fold(f64::INFINITY, f64::min);
    if g_norm == 0.0 || !lowest.is_finite() {
        return (vec![0.0; g.len()], 0.0);
    }
    if lowest > 0.0 {
        let z = step_at(0.0);
        if norm(&z) <= radius {
            return (z, 0.0);
        }
    }
    let hi = lowest.min(0.0);
    // ||z(mu)|| <= ||g|| / (lowest - mu): at most `radius` at this shift.
    let lo = lowest - g_norm / radius * (1.0 + 1e-12) - 1e-300;
    let f = |mu: f64| {
        let n = norm(&step_at(mu));
        (if n.is_finite() { 1.0 / n } else { 0.0 }) - 1.0 / radius
    };
    let mu = if f(hi) >= 0.0 {
        hi
    } else {
        brentq(f, lo, hi, 1e-14 * lo.abs().max(1.0), 1e-14, 200).unwrap_or(lo)
    };
    let mut z = step_at(mu);
    let len = norm(&z);
    if len.is_finite() && len > radius {
        for v in &mut z {
            *v *= radius / len;
        }
    }
    (z, mu)
}

/// TRIM step enhancer.
#[derive(Debug, Clone, Copy)]
pub struct Trim {
    /// Trust radius `Delta`.
    pub radius: f64,
}

impl Trim {
    /// TRIM with the given trust radius.
    pub fn new(radius: f64) -> Self {
        Self { radius }
    }
}

impl StepEnhancer for Trim {
    fn name(&self) -> &str {
        "trim"
    }

    fn uses_bias_hessian(&self) -> bool {
        true
    }

    fn enhance(&mut self, input: &StepInput<'_>, step: Col<f64>) -> MoptResult<Col<f64>> {
        if col_norm(&step) < self.radius {
            return Ok(step);
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
        let Ok((vals, vecs)) = symmetric_eigen(&h) else {
            return Ok(step);
        };
        let live: Vec<usize> = (0..n)
            .filter(|&k| vals[k].is_finite() && vals[k].abs() >= NULL_MODE)
            .collect();
        // Image function: the lowest `saddle_order` modes are reversed.
        let sign = |j: usize| if j < input.saddle_order { -1.0 } else { 1.0 };
        let lam: Vec<f64> = live
            .iter()
            .enumerate()
            .map(|(j, &k)| sign(j) * vals[k])
            .collect();
        let gt: Vec<f64> = live
            .iter()
            .enumerate()
            .map(|(j, &k)| sign(j) * (0..n).map(|r| vecs[(r, k)] * gradient[r]).sum::<f64>())
            .collect();
        let (z, _) = level_shifted_step(&lam, &gt, self.radius);
        let mut out: Col<f64> = Col::zeros(n);
        for (j, &k) in live.iter().enumerate() {
            for r in 0..n {
                out[r] += z[j] * vecs[(r, k)];
            }
        }
        let len = col_norm(&out);
        if !len.is_finite() || len < 1e-10 {
            return Ok(step);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::opt::stepper::plain_input;
    use faer::Mat;

    #[test]
    fn level_shift_puts_the_step_on_the_boundary() {
        let lam = [-0.5, 0.3, 2.0];
        let g = [0.2, -0.1, 0.4];
        for radius in [0.05, 0.3, 2.0] {
            let (z, mu) = level_shifted_step(&lam, &g, radius);
            let len = z.iter().map(|v| v * v).sum::<f64>().sqrt();
            assert!((len - radius).abs() < 1e-9, "{radius}: {len}");
            // Below every eigenvalue: all curvatures of the shifted model are positive.
            assert!(mu < -0.5);
            for i in 0..3 {
                assert!((z[i] * (lam[i] - mu) + g[i]).abs() < 1e-10);
            }
        }
        // Positive definite and inside the radius: the Newton step, no shift.
        let (z, mu) = level_shifted_step(&[1.0, 2.0], &[0.1, -0.2], 1.0);
        assert_eq!(mu, 0.0);
        assert!((z[0] + 0.1).abs() < 1e-15 && (z[1] - 0.1).abs() < 1e-15);
        assert_eq!(level_shifted_step(&[1.0], &[0.0], 1.0).0, vec![0.0]);
    }

    #[test]
    fn long_steps_are_replaced_by_the_image_minimizer_on_the_trust_sphere() {
        let x = Col::zeros(2);
        let h = Mat::from_fn(2, 2, |i, j| if i == j { [-0.5, 1.0][i] } else { 0.0 });
        let g = Col::from_fn(2, |i| [0.3, 0.4][i]);
        let mut trim = Trim::new(0.1);
        let input = plain_input(0, &x, 0.0, &g, &h, None, 1);
        // Short steps pass unchanged.
        let short = Col::from_fn(2, |i| [0.01, 0.02][i]);
        assert_eq!(trim.enhance(&input, short.clone()).unwrap(), short);
        // A long step: on the sphere, uphill along the reversed mode, downhill along the other.
        let long = Col::from_fn(2, |i| [5.0, -3.0][i]);
        let s = trim.enhance(&input, long).unwrap();
        assert!((col_norm(&s) - 0.1).abs() < 1e-9);
        assert!(s[0] > 0.0 && s[1] < 0.0, "{s:?}");
        // Minimization (order 0) with negative curvature: still downhill on the sphere.
        let input = plain_input(0, &x, 0.0, &g, &h, None, 0);
        let s = trim
            .enhance(&input, Col::from_fn(2, |i| [5.0, -3.0][i]))
            .unwrap();
        assert!((col_norm(&s) - 0.1).abs() < 1e-9 && s[0] < 0.0 && s[1] < 0.0);
    }
}
