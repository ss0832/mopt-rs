//! The Hessian of O1NumHess from derivatives along a few directions: a local fit followed by
//! a low-rank correction.
//!
//! With the directions `D` (orthonormal columns) and the gradient derivatives `G = H D`:
//!
//! 1. Local part: the symmetric `H` with elements only where `distance < dmax + ddmax` that
//!    minimizes `|G - H D|^2 + |W o H|^2`, `W_ij^2 = lambda max(0, distance_ij - dmax)^(2 beta)`.
//!    The normal equations `M o (sym(H D D^T) + W^2 o H) = M o sym(G D^T)` (`M`: the mask) are
//!    symmetric positive semidefinite in the Frobenius inner product and are solved by
//!    conjugate gradients from zero, which gives the smallest solution where the directions
//!    leave elements undetermined. (The reference implementation stores the lower triangle
//!    without weights, which makes the same equations nonsymmetric, and uses GMRES.)
//! 2. Low-rank part: `H <- H + sym((G - H D) D^T)` until `G` is reproduced, which adds the
//!    long-range elements the local part leaves out.
//!
//! The reference implementation has code that scales the columns of `G` and `D` before the
//! second step, but it reads the norms from an array of zeros and so never scales; the columns
//! are not scaled here either.
//!
//! MultiOptPy: `ModelHessian/o1numhess.py:O1NumHessCalculator._solve_odlr_problem_corrected`,
//!  `_lr_loop_strict` (MultiOptPy adds momentum to the low-rank loop; this file follows the
//!  publication and its reference implementation).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Mat;

use super::O1Settings;
use crate::math::linalg::symmetrize;

/// Absolute residual at which the local fit stops (the reference's `atol`).
const LOCAL_ATOL: f64 = 1e-14;

/// A fitted Hessian.
#[derive(Debug, Clone)]
pub(crate) struct Fit {
    /// The Hessian.
    pub hessian: Mat<f64>,
    /// `|G - H D|` at the end.
    pub residual: f64,
    /// Whether the local fit reached its tolerance.
    pub local_converged: bool,
    /// Whether the low-rank correction reproduced `G` (it stops early when a symmetric
    /// Hessian cannot reproduce it).
    pub low_rank_converged: bool,
}

fn frobenius_dot(a: &Mat<f64>, b: &Mat<f64>) -> f64 {
    let mut s = 0.0;
    for j in 0..a.ncols() {
        for i in 0..a.nrows() {
            s += a[(i, j)] * b[(i, j)];
        }
    }
    s
}

fn frobenius_norm(a: &Mat<f64>) -> f64 {
    frobenius_dot(a, a).sqrt()
}

/// `a <- a + f b`.
fn add_scaled(a: &mut Mat<f64>, f: f64, b: &Mat<f64>) {
    for j in 0..a.ncols() {
        for i in 0..a.nrows() {
            a[(i, j)] += f * b[(i, j)];
        }
    }
}

/// The regularization weights `W^2` inside the mask, `None` outside.
fn weights(distance: &Mat<f64>, s: &O1Settings) -> Vec<Option<f64>> {
    let n = distance.nrows();
    let mut w = Vec::with_capacity(n * n);
    for j in 0..n {
        for i in 0..n {
            let d = distance[(i, j)];
            w.push(
                (d < s.dmax + s.ddmax).then(|| s.lambda * (d - s.dmax).max(0.0).powf(2.0 * s.beta)),
            );
        }
    }
    w
}

/// `M o (sym(a) + W^2 o h)` in place of `a`.
fn mask_and_weigh(a: &mut Mat<f64>, h: Option<&Mat<f64>>, w: &[Option<f64>]) {
    symmetrize(a);
    let n = a.nrows();
    for j in 0..n {
        for i in 0..n {
            a[(i, j)] = match (w[j * n + i], h) {
                (Some(w2), Some(h)) => a[(i, j)] + w2 * h[(i, j)],
                (Some(_), None) => a[(i, j)],
                (None, _) => 0.0,
            };
        }
    }
}

/// The local part of the Hessian and whether the iteration converged.
fn local_fit(distance: &Mat<f64>, d: &Mat<f64>, g: &Mat<f64>, s: &O1Settings) -> (Mat<f64>, bool) {
    let n = distance.nrows();
    let w = weights(distance, s);
    let apply = |h: &Mat<f64>| {
        let mut a = (h * d) * d.transpose();
        mask_and_weigh(&mut a, Some(h), &w);
        a
    };
    let mut rhs = g * d.transpose();
    mask_and_weigh(&mut rhs, None, &w);
    let tolerance = LOCAL_ATOL.max(s.local_tolerance * frobenius_norm(&rhs));
    let mut h = Mat::<f64>::zeros(n, n);
    let mut r = rhs;
    let mut rr = frobenius_dot(&r, &r);
    if rr.sqrt() <= tolerance {
        return (h, true);
    }
    let mut p = r.clone();
    for _ in 0..s.local_iterations {
        let ap = apply(&p);
        let curvature = frobenius_dot(&p, &ap);
        if curvature <= 0.0 || !curvature.is_finite() {
            // Nothing left that the equations determine.
            return (h, false);
        }
        let alpha = rr / curvature;
        add_scaled(&mut h, alpha, &p);
        add_scaled(&mut r, -alpha, &ap);
        let rr_new = frobenius_dot(&r, &r);
        if rr_new.sqrt() <= tolerance {
            return (h, true);
        }
        let beta = rr_new / rr;
        rr = rr_new;
        for j in 0..n {
            for i in 0..n {
                p[(i, j)] = r[(i, j)] + beta * p[(i, j)];
            }
        }
    }
    (h, false)
}

/// Add symmetric low-rank corrections to `h` until `g = h d`; the residual and whether it
/// fell below the threshold.
fn low_rank_correction(
    h: &mut Mat<f64>,
    d: &Mat<f64>,
    g: &Mat<f64>,
    s: &O1Settings,
) -> (f64, bool) {
    let norm_g = frobenius_norm(g);
    let mut damping = 1.0;
    let mut previous = f64::INFINITY;
    let mut error = f64::INFINITY;
    for _ in 0..s.low_rank_iterations {
        let residual = g - &*h * d;
        error = frobenius_norm(&residual);
        if error < s.low_rank_threshold {
            return (error, true);
        }
        if (error - previous).abs() < s.low_rank_threshold * previous {
            // A symmetric Hessian cannot reproduce these derivatives any better.
            return (error, false);
        }
        if error > previous && error > norm_g {
            damping *= 0.5;
        }
        let mut correction = &residual * d.transpose();
        symmetrize(&mut correction);
        add_scaled(h, damping, &correction);
        previous = error;
    }
    (error, false)
}

/// The Hessian from the directions `d` and the derivatives `g` along them.
pub(crate) fn fit(distance: &Mat<f64>, d: &Mat<f64>, g: &Mat<f64>, s: &O1Settings) -> Fit {
    let (mut hessian, local_converged) = local_fit(distance, d, g, s);
    let (residual, low_rank_converged) = low_rank_correction(&mut hessian, d, g, s);
    symmetrize(&mut hessian);
    Fit {
        hessian,
        residual,
        local_converged,
        low_rank_converged,
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn a_full_set_of_directions_gives_the_hessian_itself() {
        let n = 6;
        let exact = Mat::from_fn(n, n, |i, j| {
            1.0 / (1.0 + (i as f64 - j as f64).abs()) + if i == j { 2.0 } else { 0.0 }
        });
        let distance = Mat::from_fn(n, n, |i, j| (i as f64 - j as f64).abs());
        let d = Mat::<f64>::identity(n, n);
        let g = &exact * &d;
        let out = fit(&distance, &d, &g, &O1Settings::default());
        assert!(out.low_rank_converged && out.residual < 1e-8);
        for i in 0..n {
            for j in 0..n {
                assert!((out.hessian[(i, j)] - exact[(i, j)]).abs() < 1e-7);
            }
        }
    }
}
