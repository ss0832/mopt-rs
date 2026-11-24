//! Modifiers of model Hessians (MultiOptPy's `ts` and `clip`).
//!
//! MultiOptPy: `ModelHessian/tshess.py:TransitionStateHessian` and
//!  `ModelHessian/approx_hessian.py:smooth_eigval`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Mat;

use crate::error::MoptResult;
use crate::math::linalg::{scaled_columns_product, symmetric_eigen, symmetrize};

/// First-order saddle Hessian from a minimum Hessian: the lowest non-zero mode `v` is reflected,
/// `H <- (I - 2 v v^T) H` (symmetrized). A Hessian that already has a negative eigenvalue below
/// `-1e-8` is returned unchanged.
pub fn ts_reflect(h: &Mat<f64>) -> MoptResult<Mat<f64>> {
    let (w, v) = symmetric_eigen(h)?;
    if w.iter().any(|&x| x < -1e-8) {
        return Ok(h.clone());
    }
    let n = h.nrows();
    let Some(k) = (0..n).find(|&k| w[k].abs() >= 1e-8) else {
        return Ok(h.clone());
    };
    let mut p = Mat::<f64>::identity(n, n);
    for i in 0..n {
        for j in 0..n {
            p[(i, j)] -= 2.0 * v[(i, k)] * v[(j, k)];
        }
    }
    let mut out = &p * h;
    symmetrize(&mut out);
    Ok(out)
}

/// MultiOptPy's smooth clipping of large eigenvalues: `|w| >= 1` becomes
/// `sign(w) (2 - |w|^-alpha)` (bounded by 2 in magnitude).
pub fn clip_eigenvalues(h: &Mat<f64>, alpha: f64) -> MoptResult<Mat<f64>> {
    let (w, v) = symmetric_eigen(h)?;
    let n = h.nrows();
    let w2: Vec<f64> = w
        .iter()
        .map(|&x| {
            if x.abs() >= 1.0 {
                x.signum() * (2.0 - x.abs().powf(-alpha))
            } else {
                x
            }
        })
        .collect();
    let all: Vec<usize> = (0..n).collect();
    Ok(scaled_columns_product(&v, &v, &all, &w2))
}
