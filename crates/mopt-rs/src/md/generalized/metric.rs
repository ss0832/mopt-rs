//! The metric of generalized coordinates at one structure: the Wilson matrix `B = dq/dx`,
//! the inverse kinetic metric `G = B M^-1 B^T` (so that `T = 1/2 p^T G p` for the momenta
//! `p` conjugate to `q`), its eigenvectors for `G p`, `G^-1 v`, and `G^(1/2) p`, and the
//! transformation of a Cartesian gradient to `q`.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::{Col, Mat};

use crate::coord::system::CoordinateSystem;
use crate::domain::MolecularSystem;
use crate::error::{MoptError, MoptResult};
use crate::math::linalg::symmetric_eigen;

/// `B`, `G`, and what is derived from them at one structure.
#[derive(Debug, Clone)]
pub(super) struct Metric {
    /// `B = dq/dx` (`n x 3N`).
    pub b: Mat<f64>,
    /// Eigenvalues of `G`.
    values: Col<f64>,
    /// Eigenvectors of `G` (columns).
    vectors: Mat<f64>,
    /// `(B B^T)^-1`.
    bbt_inverse: Mat<f64>,
}

fn spectral(
    vectors: &Mat<f64>,
    values: &Col<f64>,
    f: impl Fn(f64) -> f64,
    v: &Col<f64>,
) -> Col<f64> {
    let n = values.nrows();
    let mut out = Col::<f64>::zeros(n);
    for k in 0..n {
        let along: f64 = (0..n).map(|i| vectors[(i, k)] * v[i]).sum::<f64>() * f(values[k]);
        for i in 0..n {
            out[i] += vectors[(i, k)] * along;
        }
    }
    out
}

impl Metric {
    /// The metric of `coords` at `sys` for the inverse masses `inverse_masses` (one per
    /// Cartesian component, atomic units).
    pub fn at(
        coords: &dyn CoordinateSystem,
        sys: &MolecularSystem,
        inverse_masses: &[f64],
    ) -> MoptResult<Self> {
        let b = coords.wilson_b(sys)?;
        if b.ncols() != inverse_masses.len() {
            return Err(MoptError::coord(format!(
                "the Wilson matrix has {} columns for {} Cartesian coordinates",
                b.ncols(),
                inverse_masses.len()
            )));
        }
        let n = b.nrows();
        let g = Mat::from_fn(n, n, |i, j| {
            (0..b.ncols())
                .map(|c| b[(i, c)] * inverse_masses[c] * b[(j, c)])
                .sum::<f64>()
        });
        let (values, vectors) = symmetric_eigen(&g)?;
        let bbt = &b * b.transpose();
        let (bv, bw) = symmetric_eigen(&bbt)?;
        let top = bv.iter().fold(0.0_f64, |a, v| a.max(*v));
        let bbt_inverse = Mat::from_fn(n, n, |i, j| {
            (0..n)
                .filter(|&k| bv[k] > 1e-12 * top.max(1e-300))
                .map(|k| bw[(i, k)] * bw[(j, k)] / bv[k])
                .sum::<f64>()
        });
        Ok(Self {
            b,
            values,
            vectors,
            bbt_inverse,
        })
    }

    /// Number of coordinates.
    pub fn dim(&self) -> usize {
        self.b.nrows()
    }

    /// Smallest over largest eigenvalue of `G` (0 for redundant coordinates).
    pub fn conditioning(&self) -> f64 {
        let top = self.values.iter().fold(0.0_f64, |a, v| a.max(*v));
        let low = self.values.iter().fold(f64::INFINITY, |a, v| a.min(*v));
        if top > 0.0 {
            low.max(0.0) / top
        } else {
            0.0
        }
    }

    /// `G p` (the velocities `dq/dt`).
    pub fn times(&self, p: &Col<f64>) -> Col<f64> {
        spectral(&self.vectors, &self.values, |l| l, p)
    }

    /// `G^-1 v`.
    pub fn inverse_times(&self, v: &Col<f64>) -> Col<f64> {
        spectral(&self.vectors, &self.values, |l| 1.0 / l, v)
    }

    /// `G^(1/2) p`: momenta with unit masses (`T = 1/2 |u|^2`).
    pub fn root_times(&self, p: &Col<f64>) -> Col<f64> {
        spectral(&self.vectors, &self.values, libm::sqrt, p)
    }

    /// `G^(-1/2) u`.
    pub fn inverse_root_times(&self, u: &Col<f64>) -> Col<f64> {
        spectral(&self.vectors, &self.values, |l| 1.0 / libm::sqrt(l), u)
    }

    /// The kinetic energy `1/2 p^T G p`.
    pub fn kinetic(&self, p: &Col<f64>) -> f64 {
        let gp = self.times(p);
        0.5 * p.iter().zip(gp.iter()).map(|(a, b)| a * b).sum::<f64>()
    }

    /// The Cartesian velocities `M^-1 B^T p` (with no rigid motion for internal coordinates).
    pub fn cartesian_velocity(&self, p: &Col<f64>, inverse_masses: &[f64]) -> Col<f64> {
        let bt_p = self.b.transpose() * p;
        Col::from_fn(bt_p.nrows(), |c| bt_p[c] * inverse_masses[c])
    }

    /// A Cartesian gradient as a gradient in `q`: `(B B^T)^-1 B g_x`.
    pub fn gradient_to_q(&self, g_x: &Col<f64>) -> Col<f64> {
        &self.bbt_inverse * (&self.b * g_x)
    }
}
