//! Coordination numbers of GFN0-xTB, as in xtb and gfn0-rs (`src/coordination.rs`):
//!
//! - the erf CN `CN_i = sum_j erfc-count(r_ij; 4/3 (R_i + R_j))` with the 7.5 steepness, capped
//!   smoothly at 8 (`CN' = ln(1 + e^8) - ln(1 + e^(8 - CN))`), for H0, EEQ, and SRB;
//! - the electronegativity-weighted covalent CN of D4 (`4.10451 exp(-(|EN_i - EN_j| +
//!   19.08857)^2 / (2 11.28174^2))` times the same count);
//!
//! both with xtb's D3 covalent radii and a smooth C-infinity cutoff between 35 and 40 Bohr.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use super::params::ElementParams;
use crate::math::stable::{logistic, softplus};

const SWITCH_ON: f64 = 35.0;
const CUTOFF: f64 = 40.0;
const CN_MAX: f64 = 8.0;

/// The C-infinity switching function `sigma(-(1/u - 1/t))` (`t = (r - on) / (off - on)`) and its
/// radial derivative.
pub fn smooth_cutoff(r: f64, on: f64, off: f64) -> (f64, f64) {
    if r <= on {
        return (1.0, 0.0);
    }
    if r >= off {
        return (0.0, 0.0);
    }
    let width = off - on;
    let t = (r - on) / width;
    let u = 1.0 - t;
    let x = -1.0 / t + 1.0 / u;
    let f = logistic(-x);
    let x1 = 1.0 / (t * t) + 1.0 / (u * u);
    (f, -f * (1.0 - f) * x1 / width)
}

/// Coordination numbers with what their gradients need.
#[derive(Debug, Clone)]
pub struct Coordination {
    /// Capped erf CN.
    pub erf: Vec<f64>,
    /// D4 covalent CN.
    pub covalent: Vec<f64>,
    /// `d CN'_i / d CN_i` of the cap.
    cap_slope: Vec<f64>,
    /// Pairs `(i, j, unit (x_i - x_j) / r, d count / d r, EN weight)`.
    pairs: Vec<(usize, usize, [f64; 3], f64, f64)>,
}

fn en_weight(a: &ElementParams, b: &ElementParams) -> f64 {
    let d = (a.pauling_en - b.pauling_en).abs();
    4.10451 * (-((d + 19.08857) * (d + 19.08857)) / (2.0 * 11.28174 * 11.28174)).exp()
}

impl Coordination {
    /// The coordination numbers of atoms `el` at `x` (Bohr).
    pub fn new(el: &[ElementParams], x: &[[f64; 3]]) -> Self {
        let n = x.len();
        let mut raw = vec![0.0; n];
        let mut covalent = vec![0.0; n];
        let mut pairs = Vec::new();
        for i in 0..n {
            for j in 0..i {
                let d = [x[i][0] - x[j][0], x[i][1] - x[j][1], x[i][2] - x[j][2]];
                let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                if r >= CUTOFF || r <= 1.0e-12 {
                    continue;
                }
                let (cut, dcut) = smooth_cutoff(r, SWITCH_ON, CUTOFF);
                let rc = (el[i].covalent_radius + el[j].covalent_radius) * (4.0 / 3.0);
                let arg = -7.5 * (r - rc) / rc;
                let count = 0.5 * (1.0 + libm::erf(arg));
                let dcount = -7.5 / (std::f64::consts::PI.sqrt() * rc) * (-(arg * arg)).exp();
                let base = count * cut;
                let dbase = dcount * cut + count * dcut;
                let den = en_weight(&el[i], &el[j]);
                raw[i] += base;
                raw[j] += base;
                covalent[i] += den * base;
                covalent[j] += den * base;
                pairs.push((i, j, d.map(|v| v / r), dbase, den));
            }
        }
        let cap = softplus(CN_MAX);
        let erf = raw.iter().map(|&c| cap - softplus(CN_MAX - c)).collect();
        let cap_slope = raw.iter().map(|&c| logistic(CN_MAX - c)).collect();
        Self {
            erf,
            covalent,
            cap_slope,
            pairs,
        }
    }

    /// `sum_i w_erf_i dCN'_i/dx + w_cov_i dCN^cov_i/dx` added to `grad` (3N).
    pub fn contract(&self, w_erf: &[f64], w_cov: &[f64], grad: &mut [f64]) {
        for &(i, j, u, dbase, den) in &self.pairs {
            let s = (w_erf[i] * self.cap_slope[i]
                + w_erf[j] * self.cap_slope[j]
                + (w_cov[i] + w_cov[j]) * den)
                * dbase;
            for k in 0..3 {
                grad[3 * i + k] += s * u[k];
                grad[3 * j + k] -= s * u[k];
            }
        }
    }
}
