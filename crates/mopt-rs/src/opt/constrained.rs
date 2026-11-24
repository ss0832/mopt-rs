//! Constraint projection in the optimizer's coordinates.
//!
//! The Cartesian constraint gradients `J` (see [`crate::pes::constraint`]) are carried into the
//! optimizer's coordinates like gradients, `a_q = (B B^T)^+ B a_x`, and an orthonormal basis `Q`
//! of their span is removed from the gradient, the step, and the Hessian. The projected Hessian
//! used for the step is `P (H - C) P + s Q Q^T` with the constraint curvature `C` and a stiffness
//! `s = 100 max(max_i |H_ii|, 0.5)` along the constrained directions (MultiOptPy's hard
//! stiffness), so that step generators neither move along nor pick those directions.
//!
//! MultiOptPy: `Constraint/constraint_condition.py:ProjectOutConstrain` and
//!  `optimization.py:ConstraintManager`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::{Col, Mat};

use crate::coord::system::CoordinateFrame;
use crate::error::MoptResult;
use crate::math::linalg::{symmetric_eigen, symmetrize};
use crate::pes::constraint::ConstraintFrame;

/// MultiOptPy's `hard_mult`.
const STIFFNESS: f64 = 100.0;

/// Projector onto the complement of the constraint directions in `q`.
#[derive(Debug, Clone)]
pub struct ConstraintProjection {
    /// Orthonormal basis of the constraint directions (columns).
    q: Mat<f64>,
}

impl ConstraintProjection {
    /// Projection for the constraints of `cf` in the coordinates of `frame`.
    pub fn new(frame: &CoordinateFrame, cf: &ConstraintFrame) -> MoptResult<Self> {
        let m = cf.jacobian.nrows();
        let cols: Vec<Col<f64>> = (0..m)
            .map(|k| {
                let a = Col::from_fn(cf.jacobian.ncols(), |i| cf.jacobian[(k, i)]);
                frame.gradient_to_q(&a)
            })
            .collect();
        let nq = cols.first().map_or(0, Col::nrows);
        // Orthonormal basis of the span from the eigenvectors of A A^T (rank-revealing).
        let a = Mat::from_fn(nq, m, |i, k| cols[k][i]);
        let g = &a * a.transpose();
        let (w, v) = symmetric_eigen(&g)?;
        let wmax = w.iter().fold(0.0_f64, |s, x| s.max(x.abs()));
        let keep: Vec<usize> = (0..nq)
            .filter(|&k| w[k] > 1e-12 * wmax.max(1e-300))
            .collect();
        let q = Mat::from_fn(nq, keep.len(), |i, c| v[(i, keep[c])]);
        Ok(Self { q })
    }

    /// Number of independent constraint directions.
    pub fn rank(&self) -> usize {
        self.q.ncols()
    }

    /// `v - Q Q^T v`.
    pub fn vector(&self, v: &Col<f64>) -> Col<f64> {
        if self.q.ncols() == 0 {
            return v.clone();
        }
        v - &self.q * (self.q.transpose() * v)
    }

    /// `P M P`.
    pub fn matrix(&self, m: &Mat<f64>) -> Mat<f64> {
        if self.q.ncols() == 0 {
            return m.clone();
        }
        let n = m.nrows();
        let p = Mat::<f64>::identity(n, n) - &self.q * self.q.transpose();
        let mut out = &p * m * &p;
        symmetrize(&mut out);
        out
    }

    /// Hessian for the step: `P (H - C) P + s Q Q^T`.
    pub fn hessian(&self, h: &Mat<f64>, curvature_q: &Mat<f64>) -> Mat<f64> {
        let mut out = self.matrix(&(h - curvature_q));
        if self.q.ncols() > 0 {
            let diag = (0..h.nrows()).fold(0.0_f64, |a, i| a.max(h[(i, i)].abs()));
            let s = STIFFNESS * diag.max(0.5);
            out += &self.q * self.q.transpose() * faer::Scale(s);
        }
        out
    }
}
