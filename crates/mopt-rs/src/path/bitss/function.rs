//! The BITSS function of two images: its coefficients, value, gradient, and Hessian.
//!
//! MultiOptPy: `ModelFunction/binary_image_ts_search_model_function.py:BITSSModelFunction` (`calc_energy`, `calc_grad`, `calc_hess`).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::{Col, Mat};

use crate::error::{MoptError, MoptResult};
use crate::math::linalg::col_norm;

/// The barrier estimate is not taken below this (it divides `alpha`).
const BARRIER_FLOOR: f64 = 1e-10;

/// The coefficients of the two penalty terms and the barrier estimate they were set from.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Coefficients {
    /// `E_B`.
    pub barrier: f64,
    /// Coefficient of `(E1 - E2)^2`.
    pub kappa_e: f64,
    /// Coefficient of `(d - d_i)^2`.
    pub kappa_d: f64,
}

impl Coefficients {
    /// Eqs. (4) and (5) of the paper for the separation `target` asked for, the barrier
    /// estimate `barrier`, and the sizes `projected` of the two gradients along the line
    /// between the images.
    pub fn new(alpha: f64, beta: f64, target: f64, barrier: f64, projected: [f64; 2]) -> Self {
        let barrier = barrier.max(BARRIER_FLOOR);
        let slope = projected[0].hypot(projected[1]);
        let by_gradient = slope / (2.0 * std::f64::consts::SQRT_2 * beta * target);
        let by_barrier = barrier / (beta * target * target);
        Self {
            barrier,
            kappa_e: alpha / (2.0 * barrier),
            kappa_d: by_gradient.max(by_barrier),
        }
    }
}

/// Distance of two points.
pub fn separation(first: &Col<f64>, second: &Col<f64>) -> f64 {
    col_norm(&(first - second))
}

/// Unit vector from the second point to the first and their distance.
pub(super) fn line(first: &Col<f64>, second: &Col<f64>) -> MoptResult<(Col<f64>, f64)> {
    if first.nrows() != second.nrows() {
        return Err(MoptError::config(format!(
            "BITSS: the images have {} and {} variables",
            first.nrows(),
            second.nrows()
        )));
    }
    let r = first - second;
    let d = col_norm(&r);
    if !(d.is_finite() && d > 1e-12) {
        return Err(MoptError::calc("BITSS: the two images coincide"));
    }
    Ok((r * (1.0 / d), d))
}

/// What goes into the second derivatives of the BITSS function.
pub(super) struct Curvature<'a> {
    /// Factors of the two blocks of the surface (`1 +- 2 kappa_e (E1 - E2)`).
    pub(super) factors: [f64; 2],
    /// `(d - d_i) / d`.
    pub(super) ratio: f64,
    pub(super) unit: &'a Col<f64>,
    pub(super) gradients: [&'a Col<f64>; 2],
    pub(super) blocks: [&'a Mat<f64>; 2],
}

pub(super) fn assemble(c: &Coefficients, parts: &Curvature<'_>) -> Mat<f64> {
    let n = parts.unit.nrows();
    let [g1, g2] = parts.gradients;
    let u = parts.unit;
    // Second derivatives of kappa_d (d - d_i)^2 within one image.
    let distance = |i: usize, j: usize| {
        let along = u[i] * u[j];
        let unit = if i == j { 1.0 } else { 0.0 };
        2.0 * c.kappa_d * (along + parts.ratio * (unit - along))
    };
    Mat::from_fn(2 * n, 2 * n, |row, col| {
        let (a, i) = (row / n, row % n);
        let (b, j) = (col / n, col % n);
        match (a, b) {
            (0, 0) => {
                parts.factors[0] * parts.blocks[0][(i, j)]
                    + 2.0 * c.kappa_e * g1[i] * g1[j]
                    + distance(i, j)
            }
            (1, 1) => {
                parts.factors[1] * parts.blocks[1][(i, j)]
                    + 2.0 * c.kappa_e * g2[i] * g2[j]
                    + distance(i, j)
            }
            (0, _) => -2.0 * c.kappa_e * g1[i] * g2[j] - distance(i, j),
            _ => -2.0 * c.kappa_e * g2[i] * g1[j] - distance(i, j),
        }
    })
}

/// The BITSS function and its gradient in the `2n` variables of the two images (those of
/// the first image, then those of the second), for the separation `target` asked for.
pub fn penalized(
    c: &Coefficients,
    target: f64,
    images: [&Col<f64>; 2],
    energies: [f64; 2],
    gradients: [&Col<f64>; 2],
) -> MoptResult<(f64, Col<f64>)> {
    let (unit, d) = line(images[0], images[1])?;
    let n = unit.nrows();
    if gradients.iter().any(|g| g.nrows() != n) {
        return Err(MoptError::config(
            "BITSS: a gradient does not have the length of its image",
        ));
    }
    let gap = energies[0] - energies[1];
    let stretch = d - target;
    let value = energies[0] + energies[1] + c.kappa_e * gap * gap + c.kappa_d * stretch * stretch;
    let pull = 2.0 * c.kappa_d * stretch;
    let factors = [1.0 + 2.0 * c.kappa_e * gap, 1.0 - 2.0 * c.kappa_e * gap];
    let gradient = Col::from_fn(2 * n, |k| {
        if k < n {
            factors[0] * gradients[0][k] + pull * unit[k]
        } else {
            factors[1] * gradients[1][k - n] - pull * unit[k - n]
        }
    });
    Ok((value, gradient))
}

/// The Hessian of the BITSS function in the `2n` variables of the two images, from the
/// Hessians of the surface at them (the coefficients are constants).
pub fn penalized_hessian(
    c: &Coefficients,
    target: f64,
    images: [&Col<f64>; 2],
    energies: [f64; 2],
    gradients: [&Col<f64>; 2],
    hessians: [&Mat<f64>; 2],
) -> MoptResult<Mat<f64>> {
    let (unit, d) = line(images[0], images[1])?;
    let n = unit.nrows();
    if gradients.iter().any(|g| g.nrows() != n)
        || hessians.iter().any(|h| h.nrows() != n || h.ncols() != n)
    {
        return Err(MoptError::config(
            "BITSS: a gradient or a Hessian does not have the size of its image",
        ));
    }
    let gap = energies[0] - energies[1];
    Ok(assemble(
        c,
        &Curvature {
            factors: [1.0 + 2.0 * c.kappa_e * gap, 1.0 - 2.0 * c.kappa_e * gap],
            ratio: (d - target) / d,
            unit: &unit,
            gradients,
            blocks: hessians,
        },
    ))
}
