//! Small dense solvers shared by the DIIS family: least-squares solutions of small linear
//! systems and the minimum of a quadratic function on the probability simplex.
//!
//! `simplex_minimum` finds the global minimum of `f(c) = l.c + c.Q c / 2` over `c_i >= 0`,
//! `sum_i c_i = 1` by enumerating the faces of the simplex: on every face the stationary point
//! of `f` under the sum constraint solves a linear (Karush-Kuhn-Tucker) system, and the global
//! minimum is the best feasible one of these points. `Q` may be indefinite. The cost grows as
//! `2^n`; the DIIS methods use at most ten points.
//!
//! MultiOptPy: no counterpart (MultiOptPy calls SciPy's SLSQP for the EDIIS and ADIIS
//! coefficients, from the uniform start, which finds a local minimum only).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::{Col, Mat};

use crate::error::{MoptError, MoptResult};
use crate::math::linalg::{symmetric_eigen, symmetrize};

/// Minimum-norm least-squares solution of `A x = b` for a small matrix `A` (any shape), through
/// the eigen-decomposition of `A^T A`; directions with eigenvalues below `1e-13` times the
/// largest are left out.
pub fn least_squares(a: &Mat<f64>, b: &[f64]) -> MoptResult<Vec<f64>> {
    let (m, n) = (a.nrows(), a.ncols());
    if b.len() != m {
        return Err(MoptError::linalg("least squares: sizes do not match"));
    }
    if n == 0 {
        return Ok(Vec::new());
    }
    let mut ata = a.transpose() * a;
    symmetrize(&mut ata);
    let projected = Col::from_fn(n, |j| (0..m).map(|i| a[(i, j)] * b[i]).sum::<f64>());
    let (vals, vecs) = symmetric_eigen(&ata)?;
    let top = vals.iter().fold(0.0_f64, |x, y| x.max(*y));
    let mut x = vec![0.0; n];
    for k in 0..n {
        if vals[k] > 1e-13 * top && vals[k] > 0.0 {
            let proj: f64 = (0..n).map(|r| vecs[(r, k)] * projected[r]).sum();
            for r in 0..n {
                x[r] += vecs[(r, k)] * proj / vals[k];
            }
        }
    }
    if x.iter().all(|v| v.is_finite()) {
        Ok(x)
    } else {
        Err(MoptError::linalg("least squares: non-finite solution"))
    }
}

/// Global minimizer of `l.c + c.Q c / 2` on the simplex `c >= 0`, `sum c = 1` (`Q` symmetric,
/// `n <= 16`).
pub fn simplex_minimum(q: &Mat<f64>, l: &[f64]) -> MoptResult<Vec<f64>> {
    let n = l.len();
    if n == 0 || n > 16 || q.nrows() != n || q.ncols() != n {
        return Err(MoptError::config(
            "simplex minimum needs between 1 and 16 points and a matching matrix",
        ));
    }
    let value = |c: &[f64]| -> f64 {
        let mut v = 0.0;
        for i in 0..n {
            v += l[i] * c[i];
            for j in 0..n {
                v += 0.5 * c[i] * q[(i, j)] * c[j];
            }
        }
        v
    };
    let mut best: Option<(f64, Vec<f64>)> = None;
    for mask in 1_u32..(1_u32 << n) {
        let support: Vec<usize> = (0..n).filter(|i| mask & (1 << i) != 0).collect();
        let k = support.len();
        let c_face: Vec<f64> = if k == 1 {
            vec![1.0]
        } else {
            // [Q_SS 1; 1^T 0] [c; mu] = [-l_S; 1]
            let kkt = Mat::from_fn(k + 1, k + 1, |r, c| match (r < k, c < k) {
                (true, true) => q[(support[r], support[c])],
                (false, false) => 0.0,
                _ => 1.0,
            });
            let mut rhs: Vec<f64> = support.iter().map(|&i| -l[i]).collect();
            rhs.push(1.0);
            match least_squares(&kkt, &rhs) {
                Ok(sol) => sol[..k].to_vec(),
                Err(_) => continue,
            }
        };
        let sum: f64 = c_face.iter().sum();
        if (sum - 1.0).abs() > 1e-8 || c_face.iter().any(|v| *v < -1e-10 || !v.is_finite()) {
            continue;
        }
        let mut c = vec![0.0; n];
        for (j, &i) in support.iter().enumerate() {
            c[i] = c_face[j].max(0.0) / sum;
        }
        let v = value(&c);
        if v.is_finite() && best.as_ref().is_none_or(|(b, _)| v < *b) {
            best = Some((v, c));
        }
    }
    best.map(|(_, c)| c)
        .ok_or_else(|| MoptError::linalg("simplex minimum: no finite value"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn least_squares_handles_square_tall_and_singular_systems() {
        let a = Mat::from_fn(2, 2, |i, j| [[2.0, 1.0], [1.0, 3.0]][i][j]);
        let x = least_squares(&a, &[3.0, 5.0]).unwrap();
        assert!((x[0] - 0.8).abs() < 1e-12 && (x[1] - 1.4).abs() < 1e-12);
        // Rank one: the minimum-norm solution.
        let s = Mat::from_fn(2, 2, |_, _| 1.0);
        let x = least_squares(&s, &[2.0, 2.0]).unwrap();
        assert!((x[0] - 1.0).abs() < 1e-12 && (x[1] - 1.0).abs() < 1e-12);
        let t = Mat::from_fn(3, 1, |i, _| [1.0, 1.0, 1.0][i]);
        let x = least_squares(&t, &[1.0, 2.0, 3.0]).unwrap();
        assert!((x[0] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn simplex_minimum_is_global_for_convex_and_concave_forms() {
        // Convex: interior minimum of |c - (0.2, 0.3, 0.5)|^2.
        let q = Mat::<f64>::identity(3, 3) * 2.0;
        let c = simplex_minimum(&q, &[-0.4, -0.6, -1.0]).unwrap();
        for (a, b) in c.iter().zip([0.2, 0.3, 0.5]) {
            assert!((a - b).abs() < 1e-10, "{c:?}");
        }
        // Convex with the unconstrained minimum outside: a face.
        let c = simplex_minimum(&q, &[0.0, 5.0, 0.0]).unwrap();
        assert!((c[0] - 0.5).abs() < 1e-10 && c[1].abs() < 1e-12 && (c[2] - 0.5).abs() < 1e-10);
        // Concave: the best vertex, not the interior stationary point (a maximum).
        let q = Mat::<f64>::identity(3, 3) * -2.0;
        let c = simplex_minimum(&q, &[0.0, -0.1, 0.0]).unwrap();
        assert_eq!(c, vec![0.0, 1.0, 0.0]);
        // Brute-force comparison on a grid for an indefinite form.
        let q = Mat::from_fn(3, 3, |i, j| {
            [[1.0, -2.0, 0.5], [-2.0, 0.3, 0.1], [0.5, 0.1, -1.0]][i][j]
        });
        let l = [0.2, -0.1, 0.3];
        let c = simplex_minimum(&q, &l).unwrap();
        let f = |c: &[f64]| {
            (0..3)
                .map(|i| l[i] * c[i] + (0..3).map(|j| 0.5 * c[i] * q[(i, j)] * c[j]).sum::<f64>())
                .sum::<f64>()
        };
        let mut grid_best = f64::INFINITY;
        for a in 0..=200 {
            for b in 0..=(200 - a) {
                let p = [
                    a as f64 / 200.0,
                    b as f64 / 200.0,
                    (200 - a - b) as f64 / 200.0,
                ];
                grid_best = grid_best.min(f(&p));
            }
        }
        assert!(f(&c) <= grid_best + 1e-12, "{} vs {grid_best}", f(&c));
        assert!(simplex_minimum(&q, &[]).is_err());
    }
}
