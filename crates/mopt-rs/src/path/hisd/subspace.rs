//! The linear algebra of a tracked subspace: orthonormalization with the products of the
//! Hessian carried along, the Rayleigh-Ritz problem, principal angles, and the Euler steps
//! of the directions of HiSD, of the Oja flow, and of GGAD.
//!
//! Every function works on directions in the metric of the search (see [`super::frame`]).
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::{Col, Mat};

use super::settings::DirectionUpdate;
use crate::error::MoptResult;
use crate::math::linalg::{col_dot, col_norm, symmetric_eigen};

/// A direction left with less than this fraction of its length after the earlier ones
/// are taken out is dependent on them.
pub const DEPENDENCE: f64 = 1e-8;

/// Orthonormalize `v` against `basis` (orthonormal) by two passes of Gram-Schmidt; `None`
/// when it is dependent on them.
pub fn orthonormal_to(basis: &[Col<f64>], v: &Col<f64>) -> Option<Col<f64>> {
    let length = col_norm(v);
    if !(length.is_finite() && length > 0.0) {
        return None;
    }
    let mut out = v.clone();
    for _ in 0..2 {
        for b in basis {
            let c = col_dot(b, &out);
            out -= b * c;
        }
    }
    let left = col_norm(&out);
    (left > DEPENDENCE * length).then(|| out * (1.0 / left))
}

/// Modified Gram-Schmidt of `vectors` in order; the dependent ones are dropped.
pub fn gram_schmidt(vectors: &[Col<f64>]) -> Vec<Col<f64>> {
    let mut basis: Vec<Col<f64>> = Vec::with_capacity(vectors.len());
    for v in vectors {
        if let Some(b) = orthonormal_to(&basis, v) {
            basis.push(b);
        }
    }
    basis
}

/// A direction and the product of the Hessian with it.
pub type Pair = (Col<f64>, Col<f64>);

/// Gram-Schmidt of the directions of `pairs` in order, with the same linear operations on
/// the products (so that each product stays the product of its direction); dependent
/// directions are dropped.
pub fn orthonormal_pairs(pairs: &[Pair]) -> Vec<Pair> {
    let mut out: Vec<Pair> = Vec::with_capacity(pairs.len());
    for (v, hv) in pairs {
        let length = col_norm(v);
        if !(length.is_finite() && length > 0.0) {
            continue;
        }
        let (mut a, mut b) = (v.clone(), hv.clone());
        for _ in 0..2 {
            for (q, hq) in &out {
                let c = col_dot(q, &a);
                a -= q * c;
                b -= hq * c;
            }
        }
        let left = col_norm(&a);
        if left > DEPENDENCE * length {
            out.push((a * (1.0 / left), b * (1.0 / left)));
        }
    }
    out
}

/// Ritz values (ascending), Ritz vectors, and their products.
#[derive(Debug, Clone)]
pub struct Ritz {
    /// Ritz values.
    pub values: Vec<f64>,
    /// Ritz vectors (orthonormal).
    pub vectors: Vec<Col<f64>>,
    /// Products of the Hessian with the Ritz vectors.
    pub products: Vec<Col<f64>>,
}

/// The Rayleigh-Ritz problem on the span of orthonormal `pairs`.
pub fn rayleigh_ritz(pairs: &[Pair]) -> MoptResult<Ritz> {
    let m = pairs.len();
    let mut a = Mat::<f64>::from_fn(m, m, |i, j| col_dot(&pairs[i].0, &pairs[j].1));
    for i in 0..m {
        for j in (i + 1)..m {
            let s = 0.5 * (a[(i, j)] + a[(j, i)]);
            a[(i, j)] = s;
            a[(j, i)] = s;
        }
    }
    let (values, y) = symmetric_eigen(&a)?;
    let n = pairs.first().map_or(0, |p| p.0.nrows());
    let combine = |j: usize, second: bool| {
        let mut out = Col::<f64>::zeros(n);
        for (i, (v, hv)) in pairs.iter().enumerate() {
            out += (if second { hv } else { v }) * y[(i, j)];
        }
        out
    };
    Ok(Ritz {
        values: (0..m).map(|j| values[j]).collect(),
        vectors: (0..m).map(|j| combine(j, false)).collect(),
        products: (0..m).map(|j| combine(j, true)).collect(),
    })
}

/// The sine of the largest principal angle between the spans of two orthonormal sets of
/// the same size, `||P_a - P_b||_2`.
pub fn largest_principal_sine(a: &[Col<f64>], b: &[Col<f64>]) -> f64 {
    let k = a.len().min(b.len());
    if k == 0 {
        return 0.0;
    }
    let m = Mat::<f64>::from_fn(k, k, |i, j| col_dot(&a[i], &b[j]));
    let mtm = m.transpose() * &m;
    let smallest = symmetric_eigen(&mtm).map_or(0.0, |(s, _)| s[0]);
    libm::sqrt((1.0 - smallest).clamp(0.0, 1.0))
}

/// The directions of change `d_i` of the Euler updates at the directions `v` with the
/// products `u = H v`:
///
/// - HiSD: `d_i = -u_i + (v_i . u_i) v_i + 2 sum_(j<i) (v_j . u_i) v_j`;
/// - Oja: `d_i = -u_i + sum_j (v_j . u_i) v_j`;
/// - GGAD: `d_i = -u_i + (v_i . u_i) v_i + sum_(j<i) (v_j . u_i) v_j`, the deflated
///   Rayleigh quotient flow `-(I - v_i v_i^T) D_i H D_i v_i` for orthonormal directions
///   (`D_i v_i = v_i`).
///
/// Other updates give no Euler directions (empty).
pub fn euler_directions(update: DirectionUpdate, v: &[Col<f64>], u: &[Col<f64>]) -> Vec<Col<f64>> {
    let lower = match update {
        DirectionUpdate::HisdEuler => 2.0,
        DirectionUpdate::Ggad => 1.0,
        DirectionUpdate::OjaQr => 1.0,
        DirectionUpdate::Lobpcg | DirectionUpdate::Eigenvectors => return Vec::new(),
    };
    (0..v.len())
        .map(|i| {
            let mut d = -&u[i] + &v[i] * col_dot(&v[i], &u[i]);
            for (j, vj) in v.iter().enumerate() {
                if j == i {
                    continue;
                }
                let weight = if j < i {
                    lower
                } else if update == DirectionUpdate::OjaQr {
                    1.0
                } else {
                    0.0
                };
                if weight != 0.0 {
                    d += vj * (weight * col_dot(vj, &u[i]));
                }
            }
            d
        })
        .collect()
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn col(values: &[f64]) -> Col<f64> {
        Col::from_fn(values.len(), |k| values[k])
    }

    #[test]
    fn ritz_pairs_of_a_diagonal_matrix_and_principal_angles() {
        let h = |v: &Col<f64>| Col::from_fn(3, |k| [3.0, -1.0, 2.0][k] * v[k]);
        let basis = gram_schmidt(&[col(&[1.0, 1.0, 0.0]), col(&[0.0, 1.0, 1.0])]);
        let pairs: Vec<Pair> = basis.iter().map(|v| (v.clone(), h(v))).collect();
        let r = rayleigh_ritz(&orthonormal_pairs(&pairs)).unwrap();
        assert_eq!(r.values.len(), 2);
        assert!(r.values[0] < r.values[1]);
        for (v, hv) in r.vectors.iter().zip(&r.products) {
            assert!(col_norm(&(h(v) - hv)) < 1e-12);
        }
        let x = [col(&[1.0, 0.0, 0.0])];
        let y = [col(&[0.6, 0.8, 0.0])];
        assert!((largest_principal_sine(&x, &y) - 0.8).abs() < 1e-12);
        assert!(largest_principal_sine(&x, &x) < 1e-7);
        assert!(orthonormal_to(&x, &col(&[2.0, 0.0, 0.0])).is_none());
    }

    #[test]
    fn the_euler_directions_vanish_at_eigenvectors_and_keep_orthogonality_in_hisd() {
        let h = Mat::<f64>::from_fn(3, 3, |i, j| {
            [[2.0, 0.5, 0.1], [0.5, -1.0, 0.3], [0.1, 0.3, -2.0]][i][j]
        });
        let (_, vectors) = symmetric_eigen(&h).unwrap();
        let v: Vec<Col<f64>> = (0..2).map(|j| vectors.col(j).to_owned()).collect();
        let u: Vec<Col<f64>> = v.iter().map(|x| &h * x).collect();
        for update in [
            DirectionUpdate::HisdEuler,
            DirectionUpdate::OjaQr,
            DirectionUpdate::Ggad,
        ] {
            for d in euler_directions(update, &v, &u) {
                assert!(col_norm(&d) < 1e-12, "{update:?}");
            }
        }
        // Away from eigenvectors the HiSD flow keeps v_1 . v_2 = 0 to first order, GGAD
        // does not (hence its Schmidt orthogonalization).
        let v = gram_schmidt(&[col(&[1.0, 0.2, 0.1]), col(&[0.1, 1.0, -0.4])]);
        let u: Vec<Col<f64>> = v.iter().map(|x| &h * x).collect();
        let drift = |update| {
            let d = euler_directions(update, &v, &u);
            col_dot(&d[0], &v[1]) + col_dot(&v[0], &d[1])
        };
        assert!(drift(DirectionUpdate::HisdEuler).abs() < 1e-12);
        assert!(drift(DirectionUpdate::OjaQr).abs() < 1e-12);
        assert!(drift(DirectionUpdate::Ggad).abs() > 1e-3);
    }
}
