//! Check of the branching plane when a search for a conical intersection has converged.
//!
//! Gradient projection with branching plane updating (Maeda, Ohno, Morokuma, J. Chem.
//! Theory Comput. 2010, 6, 1538) has no derivative coupling and estimates the second
//! direction of the branching plane from the steps of the search, starting from the mean
//! gradient. The estimate is right when the difference gradient has turned through the
//! plane on the way. It is wrong when it has not: in a search that keeps the symmetry of
//! its start, the coupling direction is not totally symmetric and never shows up in the
//! gradients, and the search takes a totally symmetric direction out of its gradient
//! instead, so that it stops where the energy still falls along the seam.
//!
//! The check needs gradients only. Around a conical intersection the difference of the
//! gradients of the two states, `d(q) = g_1 - g_2`, depends on the direction from the
//! intersection and not on the distance: along a displacement `v` it is
//! `2 (g (g.v) + h (h.v)) / |...|`, a vector in the branching plane. So
//!
//! ```text
//! w = (d(x + delta v) - d(x - delta v)) / (2 delta)
//! ```
//!
//! lies in the branching plane for any `v` with a component along the coupling direction
//! `h`, and its part orthogonal to the difference gradient is the second direction. (It is
//! the Hessian difference applied to `v`, whose large part next to the cusp has the
//! branching plane as its range.) The product is repeated once with the direction found.
//! With that direction the mean gradient within the seam is tested against the convergence
//! criteria of the search; if it fails, the search takes the direction as its branching
//! plane and goes on.
//!
//! Cost: four evaluations of both states per check (two displacements, twice).
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Col;

use crate::math::linalg::{col_dot, col_norm};

/// Settings of the check.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CrossingCheckSettings {
    /// Displacement of the two structures of a product (Bohr).
    pub displacement: f64,
    /// Products per check: the first along a direction without symmetry, the others along
    /// the direction found before.
    pub products: usize,
    /// Largest number of times the search is sent on with a corrected branching plane.
    pub max_restarts: usize,
}

impl Default for CrossingCheckSettings {
    fn default() -> Self {
        Self {
            displacement: DEFAULT_DISPLACEMENT_BOHR,
            products: 2,
            max_restarts: 5,
        }
    }
}

/// Default displacement; see `docs/source/migration_surfaces.md` ("Check of the branching plane") for
/// the measurement (the direction found does not depend on it between 0.002 and 0.05 Bohr).
pub const DEFAULT_DISPLACEMENT_BOHR: f64 = 0.01;

/// `v - (x . v) x`, normalized; `None` if nothing is left.
pub(crate) fn orthogonal_unit(v: &Col<f64>, x: &Col<f64>) -> Option<Col<f64>> {
    let along = col_dot(v, x);
    let rest = Col::from_fn(v.nrows(), |i| v[i] - along * x[i]);
    let length = col_norm(&rest);
    (length > 1e-10 && length.is_finite()).then(|| Col::from_fn(v.nrows(), |i| rest[i] / length))
}

/// A direction of `n` coordinates without any symmetry, for the first product.
pub(crate) fn seed(n: usize) -> Col<f64> {
    Col::from_fn(n, |i| ((i * 37 + 11) % 17) as f64 / 17.0 - 0.5)
}

/// The mean gradient without its components along the unit vectors `x` and `y` (orthogonal
/// to each other).
pub(crate) fn within_seam(mean_gradient: &Col<f64>, x: &Col<f64>, y: &Col<f64>) -> Col<f64> {
    let (a, b) = (col_dot(mean_gradient, x), col_dot(mean_gradient, y));
    Col::from_fn(mean_gradient.nrows(), |i| {
        mean_gradient[i] - a * x[i] - b * y[i]
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn the_difference_of_the_gradient_difference_lies_in_the_branching_plane() {
        // A cone E_1,2 = -+ sqrt((g.q)^2 + (h.q)^2) in five dimensions.
        let g = Col::from_fn(5, |i| [0.3, 0.1, 0.0, -0.2, 0.05][i]);
        let h = Col::from_fn(5, |i| [0.0, 0.2, 0.1, 0.1, -0.3][i]);
        let difference = |q: &Col<f64>| -> Col<f64> {
            let (u, v) = (col_dot(&g, q), col_dot(&h, q));
            let root = (u * u + v * v).sqrt();
            Col::from_fn(5, |i| -2.0 * (g[i] * u + h[i] * v) / root)
        };
        // At a point next to the tip, the difference gradient is x.
        let at = Col::from_fn(5, |i| [1e-4, -2e-4, 5e-5, 1e-4, 0.0][i]);
        let d0 = difference(&at);
        let x = Col::from_fn(5, |i| d0[i] / col_norm(&d0));
        let delta = 0.01;
        let v = orthogonal_unit(&seed(5), &x).unwrap();
        let plus = difference(&Col::from_fn(5, |i| at[i] + delta * v[i]));
        let minus = difference(&Col::from_fn(5, |i| at[i] - delta * v[i]));
        let w = Col::from_fn(5, |i| (plus[i] - minus[i]) / (2.0 * delta));
        let y = orthogonal_unit(&w, &x).unwrap();
        // x and y span the plane of g and h: both g and h are combinations of them.
        for vector in [&g, &h] {
            let rest = within_seam(vector, &x, &y);
            assert!(
                col_norm(&rest) < 1e-3 * col_norm(vector),
                "{}",
                col_norm(&rest)
            );
        }
        // A gradient along another direction is kept.
        let other = Col::from_fn(5, |i| f64::from(u8::from(i == 2)));
        assert!(col_norm(&within_seam(&other, &x, &y)) > 0.5);
        assert!(orthogonal_unit(&x, &x).is_none());
    }
}
