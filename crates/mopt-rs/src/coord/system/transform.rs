//! Shared Cartesian <-> generalized-coordinate transformations.
//!
//! With `G = B B^T` and its pseudo-inverse `G^+`:
//! - gradient: `g_q = G^+ B g_x`
//! - Hessian: `H_q = G^+ B H_x B^T G^+` (the `dB/dx . g_q` curvature term is neglected, as is
//!   customary away from stationary points)
//! - first-order step: `dx = B^T G^+ dq`
//!
//! Finite internal steps are back-transformed iteratively until the displacement measured by the
//! coordinate system matches the requested one.
//!
//! MultiOptPy: `Coordinate/redundant_coordinate.py` (`calc_G_mat`, `calc_inv_G_mat`,
//!  `calc_inv_B_mat`, `calc_int_grad_from_pBmat`, `calc_cart_grad_from_pBmat`,
//!  `calc_cart_hess_from_pBmat_for_non_stationary_point`).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::{Col, Mat};

use super::CoordinateSystem;
use crate::domain::MolecularSystem;
use crate::error::MoptResult;
use crate::math::linalg::{col_norm, scaled_columns_product, symmetric_eigen, symmetrize};

/// Relative eigenvalue threshold for the pseudo-inverse of `G`.
const PINV_REL_TOL: f64 = 1e-10;

/// Symmetric pseudo-inverse of a positive semi-definite matrix.
pub fn pseudo_inverse_psd(g: &Mat<f64>) -> MoptResult<Mat<f64>> {
    let (w, v) = symmetric_eigen(g)?;
    let wmax = w.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
    let cutoff = PINV_REL_TOL * wmax.max(1e-300);
    let kept: Vec<usize> = (0..g.nrows()).filter(|&k| w[k] > cutoff).collect();
    let inv: Vec<f64> = kept.iter().map(|&k| 1.0 / w[k]).collect();
    Ok(scaled_columns_product(&v, &v, &kept, &inv))
}

/// Transformation data of a coordinate system evaluated at one geometry.
#[derive(Debug, Clone)]
pub struct CoordinateFrame {
    /// `None` for Cartesian coordinates (identity transformation).
    b: Option<Mat<f64>>,
    g_pinv: Option<Mat<f64>>,
}

impl CoordinateFrame {
    /// Evaluate the frame of `coords` at `sys`.
    pub fn new(coords: &dyn CoordinateSystem, sys: &MolecularSystem) -> MoptResult<Self> {
        if coords.is_cartesian() {
            return Ok(Self {
                b: None,
                g_pinv: None,
            });
        }
        let b = coords.wilson_b(sys)?;
        let g = &b * b.transpose();
        let g_pinv = pseudo_inverse_psd(&g)?;
        Ok(Self {
            b: Some(b),
            g_pinv: Some(g_pinv),
        })
    }

    /// Cartesian gradient to generalized gradient.
    pub fn gradient_to_q(&self, g_x: &Col<f64>) -> Col<f64> {
        match (&self.b, &self.g_pinv) {
            (Some(b), Some(gi)) => gi * (b * g_x),
            _ => g_x.clone(),
        }
    }

    /// Cartesian Hessian to generalized Hessian.
    pub fn hessian_to_q(&self, h_x: &Mat<f64>) -> Mat<f64> {
        match (&self.b, &self.g_pinv) {
            (Some(b), Some(gi)) => {
                let a = gi * b;
                let mut h = &a * h_x * a.transpose();
                symmetrize(&mut h);
                h
            }
            _ => h_x.clone(),
        }
    }

    /// Generalized Hessian to Cartesian Hessian, `B^T H_q B` (without the term of the
    /// gradient and the second derivatives of the coordinates).
    pub fn hessian_to_cartesian(&self, h_q: &Mat<f64>) -> Mat<f64> {
        match &self.b {
            Some(b) => {
                let mut h = b.transpose() * h_q * b;
                symmetrize(&mut h);
                h
            }
            None => h_q.clone(),
        }
    }

    /// First-order generalized displacement `dq = B dx` for a Cartesian displacement.
    pub fn displacement_to_q(&self, dx: &Col<f64>) -> Col<f64> {
        match &self.b {
            Some(b) => b * dx,
            None => dx.clone(),
        }
    }

    /// First-order Cartesian displacement for a generalized step.
    pub fn step_to_cartesian_linear(&self, dq: &Col<f64>) -> Col<f64> {
        match (&self.b, &self.g_pinv) {
            (Some(b), Some(gi)) => b.transpose() * (gi * dq),
            _ => dq.clone(),
        }
    }
}

/// Options of the iterative back-transformation.
#[derive(Debug, Clone, Copy)]
pub struct BackTransformOptions {
    /// Maximum micro-iterations.
    pub max_iter: usize,
    /// Convergence threshold on `||dq_target - dq_achieved||`.
    pub q_tol: f64,
    /// Convergence threshold on the Cartesian update norm.
    pub x_tol: f64,
}

impl Default for BackTransformOptions {
    fn default() -> Self {
        Self {
            max_iter: 25,
            q_tol: 1e-6,
            x_tol: 1e-10,
        }
    }
}

/// Cartesian displacement realizing the generalized step `dq` from geometry `start`.
///
/// Iterates `x_{n+1} = x_n + B(x_n)^T G(x_n)^+ (dq - (q(x_n) - q(x_0)))`. If the residual grows,
/// the best iterate so far is kept; if no iterate improves on the first-order step, the
/// first-order step is returned. An iterate at which the coordinates cannot be evaluated (for
/// example a fragment turned beyond the range of its rotation coordinates) ends the iteration
/// in the same way.
pub fn back_transform(
    coords: &dyn CoordinateSystem,
    start: &MolecularSystem,
    dq: &Col<f64>,
    opts: &BackTransformOptions,
) -> MoptResult<Col<f64>> {
    let frame0 = CoordinateFrame::new(coords, start)?;
    let first_order = frame0.step_to_cartesian_linear(dq);
    if coords.is_cartesian() {
        return Ok(first_order);
    }
    let x0 = start.coords_flat_bohr();
    let mut x = &x0 + &first_order;
    let mut best: Option<(f64, Col<f64>)> = None;
    for _ in 0..opts.max_iter {
        let current = start.with_flat_coords_bohr(&x)?;
        let Ok(achieved) = coords.displacement(&current, start) else {
            break;
        };
        let residual = dq - &achieved;
        let r = col_norm(&residual);
        if !r.is_finite() {
            break;
        }
        if best.as_ref().is_none_or(|(rb, _)| r < *rb) {
            best = Some((r, &x - &x0));
        } else {
            break;
        }
        if r < opts.q_tol {
            break;
        }
        let Ok(frame) = CoordinateFrame::new(coords, &current) else {
            break;
        };
        let dx = frame.step_to_cartesian_linear(&residual);
        if col_norm(&dx) < opts.x_tol {
            break;
        }
        x += dx;
    }
    Ok(best.map_or(first_order, |(_, dx)| dx))
}
