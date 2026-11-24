//! Secant Hessian update formulas.
//!
//! Each function returns the correction `dB` for the current approximation `B`, displacement
//! `s = x_{k+1} - x_k`, and gradient change `y = g_{k+1} - g_k`. Denominator guards follow
//! MultiOptPy (`|denominator| < 1e-10` drops the corresponding term).
//!
//! References: BFGS / SR1 / PSB (Nocedal & Wright); FSB and Bofill (Bofill, J. Chem. Phys.
//! 1999, 111, 10806); MSP (THEOCHEM 2002, 591, 35); CFD variants (J. Chem. Theory Comput. 2013,
//! 9, 54); double damping (arXiv:2006.08877); flowchart (Theor. Chem. Acc. 2016, 135, 84);
//! TS-BFGS (Bofill, Int. J. Quantum Chem. 2003, 94, 324; Anglada and Bofill, J. Comput. Chem.
//! 2005, 26, 1366; in the form of Wu and Wang, J. Chem. Theory Comput. 2025, eqn (5) and (6)).
//!
//! MultiOptPy: `Optimizer/hessian_update.py:ModelHessianUpdate`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::{Col, Mat};

use crate::math::linalg::{col_dot, symmetric_eigen};

/// Denominator threshold below which an update term is dropped.
pub const DENOM_THRESHOLD: f64 = 1e-10;

fn outer(a: &Col<f64>, b: &Col<f64>) -> Mat<f64> {
    Mat::from_fn(a.nrows(), b.nrows(), |i, j| a[i] * b[j])
}

/// BFGS correction `y y^T / (y^T s) - B s s^T B / (s^T B s)`.
pub fn bfgs(b: &Mat<f64>, s: &Col<f64>, y: &Col<f64>) -> Mat<f64> {
    let n = s.nrows();
    let mut out = Mat::zeros(n, n);
    let ys = col_dot(s, y);
    if ys.abs() >= DENOM_THRESHOLD {
        out += outer(y, y) * (1.0 / ys);
    }
    let bs = b * s;
    let sbs = col_dot(s, &bs);
    if sbs.abs() >= DENOM_THRESHOLD {
        out -= outer(&bs, &bs) * (1.0 / sbs);
    }
    out
}

/// Symmetric-rank-one correction `A A^T / (A^T s)` for a given residual `A`.
pub fn sr1_from_residual(a: &Col<f64>, s: &Col<f64>) -> Mat<f64> {
    let n = s.nrows();
    let denom = col_dot(a, s);
    if denom.abs() >= DENOM_THRESHOLD {
        outer(a, a) * (1.0 / denom)
    } else {
        Mat::zeros(n, n)
    }
}

/// Powell-symmetric-Broyden correction.
pub fn psb(b: &Mat<f64>, s: &Col<f64>, y: &Col<f64>) -> Mat<f64> {
    let n = s.nrows();
    let r = y - b * s;
    let ss = col_dot(s, s);
    if ss.abs() < DENOM_THRESHOLD {
        return Mat::zeros(n, n);
    }
    let rs = col_dot(&r, s);
    outer(s, s) * (-rs / (ss * ss)) + (outer(&r, s) + outer(s, &r)) * (1.0 / ss)
}

/// Bofill weight `phi^2 = (A^T s)^2 / ((A^T A)(s^T s))` (0 when the denominator vanishes).
pub fn bofill_weight(a: &Col<f64>, s: &Col<f64>) -> f64 {
    let num = col_dot(a, s).powi(2);
    let den = col_dot(a, a) * col_dot(s, s);
    if den.abs() >= DENOM_THRESHOLD {
        num / den
    } else {
        0.0
    }
}

/// Residual `A = scale * (y - B s)` (`scale = 2` for the CFD variants).
fn residual(b: &Mat<f64>, s: &Col<f64>, y: &Col<f64>, scale: f64) -> Col<f64> {
    (y - b * s) * scale
}

/// SR1 correction.
pub fn sr1(b: &Mat<f64>, s: &Col<f64>, y: &Col<f64>) -> Mat<f64> {
    sr1_from_residual(&residual(b, s, y, 1.0), s)
}

/// FSB (BFGS/SR1 mixed with `sqrt(phi^2)`); `scale = 2` gives the CFD-FSB variant.
pub fn fsb(b: &Mat<f64>, s: &Col<f64>, y: &Col<f64>, scale: f64) -> Mat<f64> {
    let a = residual(b, s, y, scale);
    let phi = libm::sqrt(bofill_weight(&a, s));
    bfgs(b, s, y) * (1.0 - phi) + sr1_from_residual(&a, s) * phi
}

/// Bofill (PSB/SR1 mixed with `phi^2`); `scale = 2` gives the CFD-Bofill variant.
pub fn bofill(b: &Mat<f64>, s: &Col<f64>, y: &Col<f64>, scale: f64) -> Mat<f64> {
    let a = residual(b, s, y, scale);
    let w = bofill_weight(&a, s);
    psb(b, s, y) * (1.0 - w) + sr1_from_residual(&a, s) * w
}

/// Perturbed CFD-Bofill: CFD-Bofill plus the projection of the correction onto the
/// complement of `s`, symmetrized as in MultiOptPy.
pub fn pcfd_bofill(b: &Mat<f64>, s: &Col<f64>, y: &Col<f64>) -> Mat<f64> {
    let d = bofill(b, s, y, 2.0);
    let n = s.nrows();
    let ss = col_dot(s, s);
    if ss < DENOM_THRESHOLD {
        return d;
    }
    // Orthogonal projector onto the complement of s: Q = I - s s^T / (s^T s).
    // sum_{ij} (v_j^T D v_i)(v_i v_j^T + v_j v_i^T) over an orthonormal basis {v} of the
    // complement equals 2 Q D Q for symmetric D.
    let q = Mat::from_fn(n, n, |i, j| f64::from(u8::from(i == j)) - s[i] * s[j] / ss);
    let qdq = &q * &d * &q;
    &d + qdq * 2.0
}

/// Murtagh-Sargent-Powell mixture of SR1 and PSB.
pub fn msp(b: &Mat<f64>, s: &Col<f64>, y: &Col<f64>) -> Mat<f64> {
    let a = residual(b, s, y, 1.0);
    let denom = libm::sqrt(col_dot(&a, &a)) * libm::sqrt(col_dot(s, s));
    let cos = if denom >= DENOM_THRESHOLD {
        (col_dot(s, &a) / denom).clamp(-1.0, 1.0)
    } else {
        0.0
    };
    let phi = 1.0 - cos * cos;
    psb(b, s, y) * phi + sr1_from_residual(&a, s) * (1.0 - phi)
}

/// Powell damping of `y` with `B = I` (step 2 of double damping, `mu2 = 0.2`).
pub fn double_damping(s: &Col<f64>, y: &Col<f64>, mu2: f64) -> Col<f64> {
    let sy = col_dot(s, y);
    let ss = col_dot(s, s);
    if sy >= mu2 * ss {
        return y.clone();
    }
    let denom = ss - sy;
    let theta = if denom.abs() < DENOM_THRESHOLD {
        0.1
    } else {
        ((1.0 - mu2) * ss / denom).clamp(0.0, 1.0)
    };
    y * theta + s * (1.0 - theta)
}

/// TS-BFGS correction: the minimal correction in the metric `M` that satisfies the secant
/// condition, `dB = j u^T + u j^T - (j^T s) u u^T` with `j = y - B s`, `u = M s / (s^T M s)`, and
/// `M = y y^T + (s^T |B| s) |B|`, where `|B|` has the absolute eigenvalues of `B`. Unlike BFGS
/// it does not force positive definiteness, so negative curvature is learned.
pub fn ts_bfgs(b: &Mat<f64>, s: &Col<f64>, y: &Col<f64>) -> Mat<f64> {
    let n = s.nrows();
    let abs_b = match symmetric_eigen(b) {
        Ok((vals, vecs)) => {
            let mut m = Mat::<f64>::zeros(n, n);
            for k in 0..n {
                let w = vals[k].abs();
                for i in 0..n {
                    for j in 0..n {
                        m[(i, j)] += w * vecs[(i, k)] * vecs[(j, k)];
                    }
                }
            }
            m
        }
        Err(_) => return Mat::zeros(n, n),
    };
    let abs_b_s = &abs_b * s;
    let m_s = y * col_dot(y, s) + &abs_b_s * col_dot(s, &abs_b_s);
    let s_m_s = col_dot(s, &m_s);
    if s_m_s.abs() < DENOM_THRESHOLD {
        return Mat::zeros(n, n);
    }
    let u = m_s * (1.0 / s_m_s);
    let j = y - b * s;
    let js = col_dot(&j, s);
    outer(&j, &u) + outer(&u, &j) - outer(&u, &u) * js
}

/// Flowchart selection between SR1, BFGS, and FSB.
///
/// MultiOptPy evaluates `z = y - B y` (the paper uses `y - B s`); kept for behavioral parity.
pub fn flowchart(b: &Mat<f64>, s: &Col<f64>, y: &Col<f64>) -> Mat<f64> {
    let z = y - b * y;
    let s_norm = libm::sqrt(col_dot(s, s));
    let guard = |d: f64| {
        if d.abs() < DENOM_THRESHOLD {
            d + DENOM_THRESHOLD
        } else {
            d
        }
    };
    let zs = col_dot(&z, s) / guard(s_norm * libm::sqrt(col_dot(&z, &z)));
    let ys = col_dot(y, s) / guard(s_norm * libm::sqrt(col_dot(y, y)));
    if zs < -0.1 {
        sr1(b, s, y)
    } else if ys > 0.1 {
        bfgs(b, s, y)
    } else {
        fsb(b, s, y, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> (Mat<f64>, Col<f64>, Col<f64>, Mat<f64>) {
        let b = Mat::from_fn(3, 3, |i, j| if i == j { 1.0 + i as f64 } else { 0.1 });
        let s = Col::from_fn(3, |i| [0.3, -0.2, 0.1][i]);
        let a_true = Mat::from_fn(3, 3, |i, j| {
            [[3.0, 0.5, 0.0], [0.5, 2.0, 0.3], [0.0, 0.3, 1.5]][i][j]
        });
        let y = &a_true * &s;
        (b, s, y, a_true)
    }

    fn secant_residual(d: Mat<f64>, b: &Mat<f64>, s: &Col<f64>, y: &Col<f64>) -> f64 {
        let bn = b + d;
        let r = &bn * s - y;
        libm::sqrt(col_dot(&r, &r))
    }

    #[test]
    fn secant_condition_holds() {
        let (b, s, y, _) = setup();
        for (name, d) in [
            ("bfgs", bfgs(&b, &s, &y)),
            ("sr1", sr1(&b, &s, &y)),
            ("psb", psb(&b, &s, &y)),
            ("fsb", fsb(&b, &s, &y, 1.0)),
            ("bofill", bofill(&b, &s, &y, 1.0)),
            ("msp", msp(&b, &s, &y)),
            ("ts_bfgs", ts_bfgs(&b, &s, &y)),
        ] {
            assert!(secant_residual(d.clone(), &b, &s, &y) < 1e-12, "{name}");
            assert!((0..3).all(|i| (0..3).all(|j| (d[(i, j)] - d[(j, i)]).abs() < 1e-14)));
        }
    }

    #[test]
    fn ts_bfgs_learns_negative_curvature() {
        // B positive definite, true curvature along s negative.
        let b = Mat::<f64>::identity(2, 2);
        let s = Col::from_fn(2, |i| [1.0, 0.0][i]);
        let y = Col::from_fn(2, |i| [-0.5, 0.0][i]);
        let bn = &b + ts_bfgs(&b, &s, &y);
        assert!((bn[(0, 0)] + 0.5).abs() < 1e-12);
        assert!((bn[(1, 1)] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn damping_keeps_curvature_positive() {
        let s = Col::from_fn(2, |i| [1.0, 0.0][i]);
        let y = Col::from_fn(2, |i| [-0.5, 0.2][i]);
        let yt = double_damping(&s, &y, 0.2);
        assert!(col_dot(&s, &yt) >= 0.2 - 1e-12);
        let y_ok = Col::from_fn(2, |i| [2.0, 0.0][i]);
        assert_eq!(double_damping(&s, &y_ok, 0.2), y_ok);
    }

    #[test]
    fn pcfd_projection_matches_explicit_basis_sum() {
        let (b, s, y, _) = setup();
        let d = bofill(&b, &s, &y, 2.0);
        // Explicit orthonormal basis of the complement of s.
        let e = [
            Col::from_fn(3, |i| [0.2, 0.3, 0.0][i]),
            Col::from_fn(3, |i| [0.0, 0.1, 0.2][i]),
        ];
        let ss = col_dot(&s, &s);
        let mut basis: Vec<Col<f64>> = Vec::new();
        for v in e {
            let mut w = &v - &s * (col_dot(&s, &v) / ss);
            for u in &basis {
                w = &w - u * col_dot(u, &w);
            }
            let n = libm::sqrt(col_dot(&w, &w));
            basis.push(w * (1.0 / n));
        }
        let mut explicit = d.clone();
        for vi in &basis {
            for vj in &basis {
                let sc = col_dot(vj, &(&d * vi));
                explicit += (outer(vi, vj) + outer(vj, vi)) * sc;
            }
        }
        let fast = pcfd_bofill(&b, &s, &y);
        for i in 0..3 {
            for j in 0..3 {
                assert!((fast[(i, j)] - explicit[(i, j)]).abs() < 1e-12);
            }
        }
    }
}
