//! MultiOptPy's short-range correction of model Hessians (the `sr` modifier,
//! `ModelHessian/shortrange.py`): the curvature of a range-separated Coulomb interaction
//! `q_i q_j c_x s (1 - erf(omega r)) / r` (`omega = 0.2`, `c_x = 0.78`, `s = 0.5`) between
//! non-bonded pairs within 15 Bohr, with charges `0.2 (<EN> - EN_i)` from Pauling
//! electronegativities, projected and added to the model Hessian. MultiOptPy introduces this
//! correction itself, so it serves the published and the `legacy_` models alike.
//!
//! MultiOptPy: `ModelHessian/shortrange.py:ShortRangeCorrectionHessian`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::f64::consts::PI;

use faer::Mat;

use super::common::{coords, norm, project_tr_rot, sub, Connectivity};
use crate::domain::{Element, MolecularSystem};

const OMEGA: f64 = 0.2;
const CX_SR: f64 = 0.78;
const SCALE: f64 = 0.5;
const CUTOFF: f64 = 15.0;

/// MultiOptPy's Pauling electronegativities (H..Kr; 2.0 for other elements).
fn electronegativity(el: Element) -> f64 {
    const EN: [f64; 37] = [
        2.0, 2.2, 0.0, 0.98, 1.57, 2.04, 2.55, 3.04, 3.44, 3.98, 0.0, 0.93, 1.31, 1.61, 1.9, 2.19,
        2.58, 3.16, 0.0, 0.82, 1.0, 1.36, 1.54, 1.63, 1.66, 1.55, 1.83, 1.88, 1.91, 1.9, 1.65,
        1.81, 2.01, 2.18, 2.55, 2.96, 0.0,
    ];
    let z = el.atomic_number() as usize;
    if (1..EN.len()).contains(&z) {
        EN[z]
    } else {
        2.0
    }
}

fn first_derivative(r: f64) -> f64 {
    if r < 1e-10 {
        return -2.0 * OMEGA.powi(3) / (3.0 * PI.sqrt());
    }
    let e = libm::erf(OMEGA * r);
    2.0 * OMEGA * (-(OMEGA * r).powi(2)).exp() / (PI.sqrt() * r) + (e - 1.0) / (r * r)
}

fn second_derivative(r: f64) -> f64 {
    if r < 1e-10 {
        return 0.0;
    }
    let e = libm::erf(OMEGA * r);
    let ef = (-(OMEGA * r).powi(2)).exp() / PI.sqrt();
    2.0 * (2.0 * e - 1.0) / r.powi(3) + 4.0 * OMEGA * ef / (r * r) + 2.0 * OMEGA.powi(3) * ef
}

/// The projected correction of `ShortRangeCorrectionHessian.apply_correction`.
pub fn short_range_correction(sys: &MolecularSystem) -> Mat<f64> {
    let x = coords(sys);
    let el = &sys.elements;
    let n = x.len();
    let bc = Connectivity::new(el, &x);
    let en: Vec<f64> = el.iter().map(|&e| electronegativity(e)).collect();
    let avg = en.iter().sum::<f64>() / n.max(1) as f64;
    let q: Vec<f64> = en.iter().map(|e| 0.2 * (avg - e)).collect();
    let mut h = Mat::<f64>::zeros(3 * n, 3 * n);
    for i in 0..n {
        for j in i + 1..n {
            if bc.bonded[i][j] || bc.bonded[j][i] {
                continue;
            }
            let rv = sub(x[j], x[i]);
            let r = norm(rv);
            if r > CUTOFF {
                continue;
            }
            let u = [rv[0] / r, rv[1] / r, rv[2] / r];
            let qf = q[i] * q[j] * CX_SR * SCALE;
            let (d2, d1) = (second_derivative(r), first_derivative(r));
            for a in 0..3 {
                for b in 0..3 {
                    let pr = u[a] * u[b];
                    let id = if a == b { 1.0 } else { 0.0 };
                    let v = qf * (d2 * pr + d1 / r * (id - pr));
                    h[(3 * i + a, 3 * i + b)] += v;
                    h[(3 * j + a, 3 * j + b)] += v;
                    h[(3 * i + a, 3 * j + b)] -= v;
                    h[(3 * j + a, 3 * i + b)] -= v;
                }
            }
        }
    }
    project_tr_rot(&h, &x)
}

/// `ShortRangeCorrectionHessian.main`: `(H + C + (H + C)^T) / 2` with NaN set to zero.
pub fn apply_short_range(h: &Mat<f64>, sys: &MolecularSystem) -> Mat<f64> {
    let c = short_range_correction(sys);
    let s = h + &c;
    let n = s.nrows();
    let mut out = Mat::<f64>::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let v = 0.5 * (s[(i, j)] + s[(j, i)]);
            out[(i, j)] = if v.is_nan() { 0.0 } else { v };
        }
    }
    out
}
