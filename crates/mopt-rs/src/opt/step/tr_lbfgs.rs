//! Limited-memory BFGS in a trust region.
//!
//! The L-BFGS matrix in compact form (Byrd, Nocedal, and Schnabel, Math. Program. 1994, 63,
//! 129) is `B = gamma I + Psi M Psi^T` with `Psi = [gamma S, Y]` and
//! `M = -[[gamma S^T S, L], [L^T, -D]]^-1`. Its eigenvalues are `gamma + theta_i` on the range
//! of `Psi` (from an orthonormal basis of that range) and `gamma` on the complement, so the
//! trust-region subproblem `min g.p + p.B p / 2, ||p|| <= Delta` is solved exactly from the
//! secular equation `||p(sigma)|| = Delta`, `p(sigma) = -(B + sigma I)^-1 g` (Burdakov, Gong,
//! Zikrin, and Yuan, Math. Program. Comput. 2017, 9, 101; Brust, Erway, and Marcia, Comput.
//! Optim. Appl. 2017, 66, 245). The radius follows the ratio of the actual to the predicted
//! reduction: quartered below 0.25, doubled above 0.75 when the step reached the boundary.
//! Pairs violating the curvature condition are corrected by Powell damping (Powell 1978) with
//! the current `B`.
//!
//! Differences from MultiOptPy (see `docs/source/migration_optimizers.md`): the newest secant pair is used
//! for the step of the same iteration (MultiOptPy adds it after computing the step), the
//! predicted reduction is that of the model and gradient that produced the step (MultiOptPy
//! takes the new gradient), and the damping uses `B s` instead of `s`.
//!
//! MultiOptPy: `Optimizer/tr_lbfgs.py:TRLBFGS`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::collections::VecDeque;

use faer::{Col, Mat};

use super::{StepGenerator, StepInput};
use crate::error::MoptResult;
use crate::math::linalg::{col_dot, col_norm, inverse_small, symmetric_eigen, symmetrize};
use crate::math::roots::brentq;

/// TR-L-BFGS parameters.
#[derive(Debug, Clone, Copy)]
pub struct TrLbfgsParams {
    /// Number of stored secant pairs.
    pub memory: usize,
    /// Largest trust radius.
    pub radius_max: f64,
    /// Smallest trust radius.
    pub radius_min: f64,
    /// Initial trust radius.
    pub radius_initial: f64,
    /// Powell damping threshold `theta` (`s.y >= theta s.B s` after damping).
    pub powell_theta: f64,
    /// Relative curvature threshold below which a pair is damped.
    pub curvature_eps: f64,
}

impl Default for TrLbfgsParams {
    fn default() -> Self {
        Self {
            memory: 30,
            radius_max: 0.5,
            radius_min: 0.01,
            radius_initial: 0.375,
            powell_theta: 0.2,
            curvature_eps: 1e-8,
        }
    }
}

/// Spectral form of the L-BFGS matrix: `B = gamma (I - P P^T) + P diag(lambda) P^T`.
#[derive(Debug, Clone)]
pub struct LbfgsSpectrum {
    /// Eigenvalue on the complement of the range of `P`.
    pub gamma: f64,
    /// Orthonormal eigenvectors (columns).
    pub vectors: Mat<f64>,
    /// Eigenvalues belonging to the columns of `vectors`.
    pub values: Vec<f64>,
}

impl LbfgsSpectrum {
    /// `B = gamma I` (no secant pairs).
    pub fn scaled_identity(n: usize, gamma: f64) -> Self {
        Self {
            gamma,
            vectors: Mat::zeros(n, 0),
            values: Vec::new(),
        }
    }

    /// Spectrum of the compact L-BFGS matrix of the pairs `(s_j, y_j)` (oldest first) with
    /// `B_0 = gamma I`; `None` when the middle matrix is singular.
    pub fn from_pairs(pairs: &[(Col<f64>, Col<f64>)], gamma: f64) -> Option<Self> {
        let k = pairs.len();
        let n = pairs.first()?.0.nrows();
        let psi = Mat::from_fn(n, 2 * k, |r, c| {
            if c < k {
                gamma * pairs[c].0[r]
            } else {
                pairs[c - k].1[r]
            }
        });
        let sy = Mat::from_fn(k, k, |i, j| col_dot(&pairs[i].0, &pairs[j].1));
        let w = Mat::from_fn(2 * k, 2 * k, |i, j| match (i < k, j < k) {
            (true, true) => gamma * col_dot(&pairs[i].0, &pairs[j].0),
            (true, false) if i > j - k => sy[(i, j - k)],
            (false, true) if j > i - k => sy[(j, i - k)],
            (false, false) if i == j => -sy[(i - k, i - k)],
            _ => 0.0,
        });
        let m = -inverse_small(&w)?;
        // Orthonormal basis of range(Psi) from the Gram matrix.
        let mut gram = psi.transpose() * &psi;
        symmetrize(&mut gram);
        let (sig, u) = symmetric_eigen(&gram).ok()?;
        let smax = sig.iter().fold(0.0_f64, |a, b| a.max(*b));
        let keep: Vec<usize> = (0..2 * k).filter(|&i| sig[i] > 1e-12 * smax).collect();
        let r = keep.len();
        // Columns of U scaled by sqrt(sigma): T = (sqrt(S) U^T) M (U sqrt(S)).
        let us = Mat::from_fn(2 * k, r, |i, c| u[(i, keep[c])] * libm::sqrt(sig[keep[c]]));
        let mut t = us.transpose() * &m * &us;
        symmetrize(&mut t);
        let (theta, v) = symmetric_eigen(&t).ok()?;
        // Q = Psi U S^-1/2, P = Q V.
        let ui = Mat::from_fn(2 * k, r, |i, c| u[(i, keep[c])] / libm::sqrt(sig[keep[c]]));
        let vectors = &psi * &ui * &v;
        let values: Vec<f64> = theta.iter().map(|t| gamma + t).collect();
        (values.iter().all(|x| x.is_finite())).then_some(Self {
            gamma,
            vectors,
            values,
        })
    }

    /// Components of `v` along the eigenvectors, and the remainder in the complement.
    fn split(&self, v: &Col<f64>) -> (Vec<f64>, Col<f64>) {
        let r = self.values.len();
        let comps: Vec<f64> = (0..r)
            .map(|c| (0..v.nrows()).map(|i| self.vectors[(i, c)] * v[i]).sum())
            .collect();
        let mut rest = v.clone();
        for (c, a) in comps.iter().enumerate() {
            for i in 0..v.nrows() {
                rest[i] -= a * self.vectors[(i, c)];
            }
        }
        (comps, rest)
    }

    /// `B v`.
    pub fn apply(&self, v: &Col<f64>) -> Col<f64> {
        let (comps, rest) = self.split(v);
        let mut out = rest * self.gamma;
        for (c, a) in comps.iter().enumerate() {
            for i in 0..v.nrows() {
                out[i] += self.values[c] * a * self.vectors[(i, c)];
            }
        }
        out
    }

    /// Minimizer of `g.p + p.B p / 2` in the ball `||p|| <= radius`, with the level shift
    /// `sigma >= 0` (`(B + sigma I) p = -g`).
    pub fn trust_region_step(&self, g: &Col<f64>, radius: f64) -> (Col<f64>, f64) {
        let (comps, rest) = self.split(g);
        let rest_norm = col_norm(&rest);
        let build = |sigma: f64| -> Col<f64> {
            let mut p = &rest * (-1.0 / (self.gamma + sigma));
            for (c, a) in comps.iter().enumerate() {
                let f = -a / (self.values[c] + sigma);
                for i in 0..g.nrows() {
                    p[i] += f * self.vectors[(i, c)];
                }
            }
            p
        };
        let norm_at = |sigma: f64| -> f64 {
            let mut sum = (rest_norm / (self.gamma + sigma)).powi(2);
            for (c, a) in comps.iter().enumerate() {
                sum += (a / (self.values[c] + sigma)).powi(2);
            }
            libm::sqrt(sum)
        };
        // Lowest eigenvalue that carries a gradient component.
        let lowest = self
            .values
            .iter()
            .copied()
            .chain((rest_norm > 0.0 || self.values.is_empty()).then_some(self.gamma))
            .fold(f64::INFINITY, f64::min);
        let g_norm = col_norm(g);
        if g_norm == 0.0 {
            return (Col::zeros(g.nrows()), 0.0);
        }
        if lowest > 0.0 && norm_at(0.0) <= radius {
            return (build(0.0), 0.0);
        }
        // ||p(sigma)|| <= ||g|| / (lowest + sigma): the root lies below g_norm / radius - lowest.
        let lo = (-lowest).max(0.0);
        let hi = (g_norm / radius - lowest).max(lo) * (1.0 + 1e-12) + 1e-300;
        let f = |sigma: f64| {
            let nrm = norm_at(sigma);
            // 1 / ||p|| is nearly linear in sigma; at a pole it is 0.
            (if nrm.is_finite() { 1.0 / nrm } else { 0.0 }) - 1.0 / radius
        };
        let sigma = if f(lo) >= 0.0 {
            lo
        } else {
            brentq(f, lo, hi, 1e-14 * hi.max(1.0), 1e-14, 200).unwrap_or(hi)
        };
        let mut p = build(sigma);
        let len = col_norm(&p);
        if len > radius {
            p *= radius / len;
        }
        (p, sigma)
    }
}

/// TR-L-BFGS step generator.
pub struct TrLbfgs {
    params: TrLbfgsParams,
    pairs: VecDeque<(Col<f64>, Col<f64>)>,
    gamma: f64,
    radius: f64,
    spectrum: Option<LbfgsSpectrum>,
    /// Energy at the last point and the reduction the model predicted for the step from it.
    last: Option<(f64, f64)>,
}

impl TrLbfgs {
    /// New generator.
    pub fn new(params: TrLbfgsParams) -> Self {
        Self {
            params,
            pairs: VecDeque::new(),
            gamma: 1.0,
            radius: params
                .radius_initial
                .clamp(params.radius_min, params.radius_max),
            spectrum: None,
            last: None,
        }
    }

    /// Current trust radius.
    pub fn trust_radius(&self) -> f64 {
        self.radius
    }

    /// Number of stored secant pairs.
    pub fn stored_pairs(&self) -> usize {
        self.pairs.len()
    }

    fn add_pair(&mut self, s: &Col<f64>, y: &Col<f64>) {
        let (sn, yn) = (col_norm(s), col_norm(y));
        if sn < 1e-14 || yn < 1e-14 {
            return;
        }
        let mut y = y.clone();
        let mut sy = col_dot(s, &y);
        if sy <= self.params.curvature_eps * sn * yn {
            // Powell damping toward B s keeps the matrix positive definite.
            let bs = match &self.spectrum {
                Some(sp) => sp.apply(s),
                None => s * self.gamma,
            };
            let sbs = col_dot(s, &bs);
            let theta = self.params.powell_theta;
            if sbs <= 0.0 || sbs - sy <= 0.0 {
                return;
            }
            let r = (1.0 - theta) * sbs / (sbs - sy);
            y = &y * r + &bs * (1.0 - r);
            sy = col_dot(s, &y);
            if sy <= 0.0 {
                return;
            }
        }
        let gamma = col_dot(&y, &y) / sy;
        if gamma.is_finite() && gamma > 0.0 {
            self.gamma = gamma;
        }
        self.pairs.push_back((s.clone(), y));
        while self.pairs.len() > self.params.memory.max(1) {
            self.pairs.pop_front();
        }
    }
}

impl Default for TrLbfgs {
    fn default() -> Self {
        Self::new(TrLbfgsParams::default())
    }
}

impl StepGenerator for TrLbfgs {
    fn name(&self) -> &str {
        "tr_lbfgs"
    }

    fn propose(&mut self, input: &StepInput<'_>) -> MoptResult<Col<f64>> {
        let p = self.params;
        let g = input.gradient;
        let n = g.nrows();
        if let Some(prev) = input.previous {
            // Rate the last step with the model that produced it.
            if let Some((e_prev, predicted)) = self.last.take() {
                let actual = e_prev - input.energy.value();
                let ratio = if predicted.abs() < 1e-300 {
                    0.0
                } else {
                    actual / predicted
                };
                let len = col_norm(&prev.step);
                if ratio.is_nan() || ratio < 0.25 {
                    self.radius = (0.25 * self.radius).max(p.radius_min);
                } else if ratio > 0.75 && len >= 0.99 * self.radius {
                    self.radius = (2.0 * self.radius).min(p.radius_max);
                }
            }
            self.add_pair(&prev.step, &(input.raw_gradient - &prev.raw_gradient));
        }
        let pairs: Vec<(Col<f64>, Col<f64>)> = self.pairs.iter().cloned().collect();
        let spectrum = match LbfgsSpectrum::from_pairs(&pairs, self.gamma) {
            Some(sp) => sp,
            None => {
                // Degenerate history: start again from the scaled identity.
                self.pairs.clear();
                LbfgsSpectrum::scaled_identity(n, self.gamma)
            }
        };
        let (step, _) = spectrum.trust_region_step(g, self.radius);
        let predicted = -(col_dot(g, &step) + 0.5 * col_dot(&step, &spectrum.apply(&step)));
        self.last = Some((input.energy.value(), predicted));
        self.spectrum = Some(spectrum);
        Ok(step)
    }

    fn reset(&mut self) {
        *self = Self::new(self.params);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::opt::step::PreviousPoint;
    use crate::opt::stepper::plain_input;

    fn pairs_of(a: &Mat<f64>, steps: &[[f64; 4]]) -> Vec<(Col<f64>, Col<f64>)> {
        steps
            .iter()
            .map(|s| {
                let s = Col::from_fn(4, |i| s[i]);
                let y = a * &s;
                (s, y)
            })
            .collect()
    }

    fn spd() -> Mat<f64> {
        Mat::from_fn(4, 4, |i, j| {
            if i == j {
                [1.0, 2.0, 3.0, 5.0][i]
            } else {
                0.3 / (1.0 + (i + j) as f64)
            }
        })
    }

    #[test]
    fn spectral_form_satisfies_the_secant_equations() {
        let a = spd();
        let pairs = pairs_of(
            &a,
            &[
                [1.0, 0.0, 0.2, 0.0],
                [0.1, 1.0, 0.0, -0.3],
                [0.0, 0.2, 1.0, 0.5],
            ],
        );
        let sp = LbfgsSpectrum::from_pairs(&pairs, 2.0).unwrap();
        // The newest pair is reproduced exactly (BFGS), and B is positive definite.
        let (s, y) = &pairs[2];
        assert!(col_norm(&(&sp.apply(s) - y)) < 1e-10);
        assert!(sp.values.iter().all(|v| *v > 0.0));
        // Orthonormal eigenvectors.
        let gram = sp.vectors.transpose() * &sp.vectors;
        for i in 0..gram.nrows() {
            for j in 0..gram.ncols() {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((gram[(i, j)] - want).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn trust_region_step_satisfies_the_secular_condition() {
        let a = spd();
        let pairs = pairs_of(&a, &[[1.0, 0.0, 0.2, 0.0], [0.1, 1.0, 0.0, -0.3]]);
        let sp = LbfgsSpectrum::from_pairs(&pairs, 2.0).unwrap();
        let g = Col::from_fn(4, |i| [0.5, -1.0, 0.3, 0.8][i]);
        // Inside the region: the quasi-Newton step B p = -g, sigma = 0.
        let (p, sigma) = sp.trust_region_step(&g, 10.0);
        assert_eq!(sigma, 0.0);
        assert!(col_norm(&(&sp.apply(&p) + &g)) < 1e-10);
        // On the boundary: ||p|| = Delta and (B + sigma I) p = -g with sigma > 0.
        for radius in [0.3, 0.05, 1e-4] {
            let (p, sigma) = sp.trust_region_step(&g, radius);
            assert!(
                (col_norm(&p) - radius).abs() < 1e-9 * radius.max(1e-3),
                "{radius}"
            );
            assert!(sigma > 0.0);
            let res = &sp.apply(&p) + &p * sigma + &g;
            assert!(col_norm(&res) < 1e-8, "{radius}: {}", col_norm(&res));
        }
        // No pairs: steepest descent scaled by 1 / gamma, capped.
        let id = LbfgsSpectrum::scaled_identity(4, 4.0);
        let (p, _) = id.trust_region_step(&g, 10.0);
        assert!(col_norm(&(&p + &g * 0.25)) < 1e-14);
    }

    #[test]
    fn quadratic_is_minimized_and_the_radius_adapts() {
        let a = spd();
        let h = Mat::<f64>::identity(4, 4);
        let mut opt = TrLbfgs::default();
        let mut x = Col::from_fn(4, |i| [3.0, -2.0, 1.0, 2.5][i]);
        let mut prev: Option<PreviousPoint> = None;
        let mut radii = Vec::new();
        for it in 0..40 {
            let g = &a * &x;
            let e = 0.5 * col_dot(&x, &g);
            let s = opt
                .propose(&plain_input(it, &x, e, &g, &h, prev.as_ref(), 0))
                .unwrap();
            assert!(col_norm(&s) <= opt.trust_radius() * (1.0 + 1e-9));
            radii.push(opt.trust_radius());
            prev = Some(PreviousPoint {
                energy: crate::domain::Quantity::new(e),
                gradient: g.clone(),
                raw_gradient: g,
                step: s.clone(),
            });
            x += &s;
        }
        assert!(col_norm(&(&a * &x)) < 1e-9, "{}", col_norm(&(&a * &x)));
        assert!(
            radii.iter().any(|r| *r > 0.375),
            "the radius grew: {radii:?}"
        );
    }

    #[test]
    fn negative_curvature_pairs_are_damped_not_stored_raw() {
        let mut opt = TrLbfgs::default();
        let s = Col::from_fn(2, |i| [1.0, 0.0][i]);
        let y = Col::from_fn(2, |i| [-0.5, 0.1][i]);
        opt.add_pair(&s, &y);
        assert_eq!(opt.stored_pairs(), 1);
        let (ps, py) = &opt.pairs[0];
        // Damped pair: s.y = theta s.B s with B = I.
        assert!((col_dot(ps, py) - 0.2).abs() < 1e-12);
    }
}
