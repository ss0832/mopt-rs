//! The rotation of a dimer: the curvature along its axis and the rotation to the lowest
//! curvature by a trial rotation and a Fourier fit of the curvature, with the gradient at the
//! end point of the rotated dimer extrapolated (no evaluation).
//!
//! With the gradient `g0` at the centre `x`, `g1` at `x + dR N`, the curvature along `N` is
//! `C = (g1 - g0) . N / dR` and the rotational force (per length) `F = -(g1 - g0)_perp / dR`.
//! A trial rotation by `phi_t` in the plane of `N` and `Theta = F / |F|` gives `C(phi_t)`; with
//! `b1 = dC/dphi(0) / 2 = -|F|`, `a1 = (C(0) - C(phi_t) + b1 sin 2 phi_t) / (1 - cos 2 phi_t)`
//! and `a0 = 2 (C(0) - a1)`, the curvature is `C(phi) = a0 / 2 + a1 cos 2 phi + b1 sin 2 phi`
//! and its minimum lies at `phi = atan(b1 / a1) / 2` (or `pi/2` further). The gradient at the
//! end point after the rotation is
//! `g1(phi) = sin(phi_t - phi) / sin phi_t g1(0) + sin phi / sin phi_t g1(phi_t)
//! + (1 - cos phi - sin phi tan(phi_t / 2)) g0`.
//!
//! G. Henkelman, H. Jonsson, J. Chem. Phys. 1999, 111, 7010 (the dimer); A. Heyden, A. T.
//! Bell, F. J. Keil, J. Chem. Phys. 2005, 123, 224101 (one end point, the curvature from the
//! Fourier fit); J. Kastner, P. Sherwood, J. Chem. Phys. 2008, 128, 014106 (the extrapolation
//! of the gradient).
//!
//! MultiOptPy: `OtherMethod/dimer.py:DimerMethod.dimer_rotate` (which rotates with forces of
//! the wrong sign; see the module documentation of the dimer).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Col;

use crate::math::linalg::{col_dot, col_norm};

/// The curvature along `axis` and the rotational force from the gradients at the centre and
/// at the end point `separation` along `axis`.
pub fn curvature_and_force(
    g0: &Col<f64>,
    g1: &Col<f64>,
    axis: &Col<f64>,
    separation: f64,
) -> (f64, Col<f64>) {
    let d = g1 - g0;
    let along = col_dot(&d, axis);
    let force = (&d - axis * along) * (-1.0 / separation);
    (along / separation, force)
}

/// The result of one rotation.
#[derive(Debug, Clone)]
pub struct Rotation {
    /// The new axis (unit length).
    pub axis: Col<f64>,
    /// The curvature along it (from the fit).
    pub curvature: f64,
    /// The angle of the rotation (rad).
    pub angle: f64,
    /// The extrapolated gradient at the new end point.
    pub end_gradient: Col<f64>,
}

/// The trial axis: `axis` turned by `trial` towards `direction` (unit, perpendicular).
pub fn trial_axis(axis: &Col<f64>, direction: &Col<f64>, trial: f64) -> Col<f64> {
    let t = axis * libm::cos(trial) + direction * libm::sin(trial);
    let n = col_norm(&t);
    t * (1.0 / n)
}

/// The rotation to the lowest curvature in the plane of `axis` and `direction` from the
/// curvature `c0` and end gradient `g1` along `axis`, the size `force` of the rotational force,
/// and the curvature `c_trial` and end gradient `g_trial` along the trial axis turned by
/// `trial`.
#[allow(clippy::too_many_arguments)]
pub fn rotate(
    axis: &Col<f64>,
    direction: &Col<f64>,
    g0: &Col<f64>,
    c0: f64,
    g1: &Col<f64>,
    force: f64,
    c_trial: f64,
    g_trial: &Col<f64>,
    trial: f64,
) -> Rotation {
    let b1 = -force;
    let a1 = (c0 - c_trial + b1 * libm::sin(2.0 * trial)) / (1.0 - libm::cos(2.0 * trial));
    let a0 = 2.0 * (c0 - a1);
    let curvature = |phi: f64| 0.5 * a0 + a1 * libm::cos(2.0 * phi) + b1 * libm::sin(2.0 * phi);
    let mut phi = if a1 == 0.0 {
        0.25 * std::f64::consts::PI
    } else {
        0.5 * libm::atan(b1 / a1)
    };
    if curvature(phi) > c0 {
        phi += 0.5 * std::f64::consts::PI;
    }
    let new_axis = trial_axis(axis, direction, phi);
    let s = libm::sin(trial);
    let end_gradient = g1 * (libm::sin(trial - phi) / s)
        + g_trial * (libm::sin(phi) / s)
        + g0 * (1.0 - libm::cos(phi) - libm::sin(phi) * libm::tan(0.5 * trial));
    Rotation {
        axis: new_axis,
        curvature: curvature(phi),
        angle: phi,
        end_gradient,
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use faer::Mat;

    #[test]
    fn one_rotation_finds_the_lowest_mode_of_a_quadratic() {
        // E = x^T A x / 2: gradients are exact, so a single rotation in the plane of the
        // lowest mode reaches it.
        let a = Mat::from_fn(3, 3, |i, j| {
            [[1.0, 0.0, 0.0], [0.0, -0.5, 0.0], [0.0, 0.0, 2.0]][i][j]
        });
        let x = Col::from_fn(3, |i| [0.1, 0.2, -0.1][i]);
        let g = |p: &Col<f64>| &a * p;
        let dr = 1e-3;
        let axis = Col::from_fn(3, |i| [libm::cos(0.4), libm::sin(0.4), 0.0][i]);
        let g0 = g(&x);
        let g1 = g(&(&x + &axis * dr));
        let (c0, f) = curvature_and_force(&g0, &g1, &axis, dr);
        assert!((c0 - (libm::cos(0.4).powi(2) - 0.5 * libm::sin(0.4).powi(2))).abs() < 1e-9);
        let size = col_norm(&f);
        let direction = &f * (1.0 / size);
        let trial = std::f64::consts::PI / 32.0;
        let t = trial_axis(&axis, &direction, trial);
        let gt = g(&(&x + &t * dr));
        let (ct, _) = curvature_and_force(&g0, &gt, &t, dr);
        let r = rotate(&axis, &direction, &g0, c0, &g1, size, ct, &gt, trial);
        assert!((r.curvature + 0.5).abs() < 1e-8, "{}", r.curvature);
        assert!((r.axis[1].abs() - 1.0).abs() < 1e-8, "{:?}", r.axis);
        let exact = g(&(&x + &r.axis * dr));
        assert!(col_norm(&(&exact - &r.end_gradient)) < 1e-10);
    }
}
