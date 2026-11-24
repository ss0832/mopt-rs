//! The objective of the direct MaxFlux method and its gradient.
//!
//! The flux along a path is largest where
//!
//! ```text
//! I[x] = integral |dx/dt| exp(beta E(x(t))) dt = L integral_0^1 exp(beta E(tau)) dtau
//! ```
//!
//! is smallest (Koda, Saito, J. Chem. Theory Comput. 2024, 20, 2798: the MaxFlux
//! functional at a low temperature, `beta` large; `L` the length of the path, `tau` the
//! share of the length). The objective is `K = ln(I) / beta`, an energy that tends to the
//! highest energy of the path as `beta` grows. No derivative of the energy is in it; its
//! gradient needs the gradients of the energy only. With the energies `E_i` and gradients
//! `g_i` at the points `tau_i` (parameters `t_i`) and the weights `w_i` of the
//! trapezoidal rule in `tau`,
//!
//! ```text
//! K = (1 / beta) [ ln L + ln sum_i w_i exp(beta E_i) ]
//! dK/dc_b = (dL/dc_b) / (beta L) + sum_i p_i [ B_b(t_i) g_i + (g_i . dx/dt_i) dt_i/dc_b ]
//! dt_i/dc_b = -(ds(t_i)/dc_b - tau_i dL/dc_b) / |dx/dt_i|
//! ```
//!
//! with `p_i` the share of point `i` in the sum (the points stay at their shares of the
//! length when the path moves). The sum is taken with its largest term divided out, so
//! that `beta E` of any size neither overflows nor vanishes.
//!
//! PyDMF writes the same integral in the parameter of its spline, keeps that parameter
//! proportional to the length with bounds on the lengths of 16 pieces of the path
//! (`dmf.py:VariationalPathOpt`, `eps_vel`), and takes the speed at a point from the
//! lengths of the pieces next to it. Here the length is integrated along the whole spline
//! and the points are placed by their shares of it, which needs no bounds: with few
//! variables (a surface of two) the bounds cannot all be met, and a speed taken at the
//! points alone lets the path wander between them.
//!
//! MultiOptPy: `MEP/pathopt_dmf_force.py:CaluculationDMF._get_action_and_gradient` (the
//! sum in the parameter of the spline, with the speed at the points only, without the
//! division by its largest term, and with `beta = 10` applied to energies in Hartree,
//! where the method means 10 per eV).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Col;

use super::arc::ArcLength;
use super::spline::SplinePath;
use crate::error::{MoptError, MoptResult};
use crate::math::linalg::{col_dot, col_norm};

/// The weights of the trapezoidal rule on the points `tau` (ascending).
pub fn trapezoid_weights(tau: &[f64]) -> Vec<f64> {
    let n = tau.len();
    (0..n)
        .map(|i| {
            let before = if i > 0 { tau[i] - tau[i - 1] } else { 0.0 };
            let after = if i + 1 < n { tau[i + 1] - tau[i] } else { 0.0 };
            0.5 * (before + after)
        })
        .collect()
}

/// The objective at a path.
#[derive(Debug, Clone)]
pub struct Action {
    /// `K = ln(I) / beta` (Hartree, on the scale of the energies handed in).
    pub value: f64,
    /// The gradient of `K` with respect to the inner control points (the first and the
    /// last control point do not move).
    pub gradient: Vec<Col<f64>>,
    /// The share of every point in the sum (they add up to one).
    pub shares: Vec<f64>,
    /// `dE/dtau` at every point.
    pub slopes: Vec<f64>,
    /// The length of the path.
    pub length: f64,
}

/// The objective for `path` with the energies and their gradients at the points that
/// are at the shares `tau` of its length (at the parameters `t`; weights `weights`);
/// `beta` in 1/Hartree.
#[allow(clippy::too_many_arguments)]
pub fn action(
    path: &SplinePath,
    arc: &ArcLength<'_>,
    tau: &[f64],
    t: &[f64],
    weights: &[f64],
    energies: &[f64],
    gradients: &[Col<f64>],
    beta: f64,
) -> MoptResult<Action> {
    let points = tau.len();
    if points < 2
        || t.len() != points
        || weights.len() != points
        || energies.len() != points
        || gradients.len() != points
    {
        return Err(MoptError::config(
            "the MaxFlux objective needs an energy and a gradient at every point (two at least)",
        ));
    }
    if beta.is_nan() || beta <= 0.0 {
        return Err(MoptError::config(
            "the MaxFlux objective needs a positive beta",
        ));
    }
    let length = arc.length();
    // ln(w_i) + beta E_i, with the largest one taken out of the sum.
    let exponents: Vec<f64> = (0..points)
        .map(|i| {
            if weights[i] > 0.0 {
                weights[i].ln() + beta * energies[i]
            } else {
                f64::NEG_INFINITY
            }
        })
        .collect();
    let largest = exponents.iter().fold(f64::NEG_INFINITY, |m, e| m.max(*e));
    if !largest.is_finite() {
        return Err(MoptError::calc(
            "the MaxFlux objective is not defined: an energy is not finite",
        ));
    }
    let terms: Vec<f64> = exponents.iter().map(|e| (e - largest).exp()).collect();
    let sum: f64 = terms.iter().sum();
    let value = (length.ln() + largest + sum.ln()) / beta;
    let shares: Vec<f64> = terms.iter().map(|term| term / sum).collect();

    let of_length = arc.derivative(1.0);
    let mut gradient: Vec<Col<f64>> = of_length
        .iter()
        .map(|d| d * (1.0 / (beta * length)))
        .collect();
    let mut slopes = Vec::with_capacity(points);
    for i in 0..points {
        let velocity = path.velocity(t[i]);
        let speed = col_norm(&velocity);
        let along = col_dot(&gradients[i], &velocity);
        slopes.push(if speed > 0.0 {
            length * along / speed
        } else {
            0.0
        });
        let inner = tau[i] > 0.0 && tau[i] < 1.0;
        if shares[i] == 0.0 || !inner || speed <= 0.0 {
            continue;
        }
        let values = path.basis.values(t[i]);
        let of_arc = arc.derivative(t[i]);
        for (b, of_point) in gradient.iter_mut().enumerate() {
            let value = values[b + 1];
            if value != 0.0 {
                *of_point += &gradients[i] * (shares[i] * value);
            }
            // The point slides along the path to keep its share of the length.
            let slide = (&of_arc[b] - &of_length[b] * tau[i]) * (-1.0 / speed);
            *of_point += slide * (shares[i] * along);
        }
    }
    Ok(Action {
        value,
        gradient,
        shares,
        slopes,
        length,
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn col(values: &[f64]) -> Col<f64> {
        Col::from_fn(values.len(), |k| values[k])
    }

    /// A surface with a ridge between two basins.
    fn surface(x: &Col<f64>) -> (f64, Col<f64>) {
        let (a, b) = (x[0], x[1]);
        let energy = 0.02 * (a * a - 1.0).powi(2) + 0.05 * (b - 0.3 * a * a).powi(2);
        let gradient = col(&[
            0.08 * a * (a * a - 1.0) - 0.06 * a * (b - 0.3 * a * a),
            0.1 * (b - 0.3 * a * a),
        ]);
        (energy, gradient)
    }

    fn objective(path: &SplinePath, tau: &[f64], beta: f64) -> Action {
        let arc = ArcLength::new(path).unwrap();
        let t: Vec<f64> = tau.iter().map(|&s| arc.parameter(s)).collect();
        let weights = trapezoid_weights(tau);
        let (energies, gradients): (Vec<f64>, Vec<Col<f64>>) =
            t.iter().map(|&ti| surface(&path.at(ti))).unzip();
        action(path, &arc, tau, &t, &weights, &energies, &gradients, beta).unwrap()
    }

    #[test]
    fn the_gradient_is_the_derivative_of_the_objective() {
        let chain = [col(&[-1.0, 0.3]), col(&[0.1, 0.4]), col(&[1.0, 0.3])];
        let mut path = SplinePath::through(&chain, 3, 4).unwrap();
        // An uneven pace, so that the points are not where their parameters say.
        path.control[2][0] += 0.2;
        let tau = [0.0, 0.2, 0.45, 0.6, 0.8, 1.0];
        for (found, wanted) in trapezoid_weights(&tau)
            .iter()
            .zip([0.1, 0.225, 0.2, 0.175, 0.2, 0.1])
        {
            assert!((found - wanted).abs() < 1e-15, "{found} {wanted}");
        }
        for beta in [50.0, 272.0, 5000.0] {
            let found = objective(&path, &tau, beta);
            assert!((found.shares.iter().sum::<f64>() - 1.0).abs() < 1e-14);
            for b in 0..5 {
                for k in 0..2 {
                    let mut up = path.clone();
                    up.control[b + 1][k] += 1e-6;
                    let mut down = path.clone();
                    down.control[b + 1][k] -= 1e-6;
                    let numeric = (objective(&up, &tau, beta).value
                        - objective(&down, &tau, beta).value)
                        / 2e-6;
                    assert!(
                        (numeric - found.gradient[b][k]).abs() < 1e-8,
                        "beta {beta}, control point {b}, {k}: {numeric} {}",
                        found.gradient[b][k]
                    );
                }
            }
        }
        // A large beta: the objective is the highest energy of the points, and nothing
        // overflows with energies of any size.
        let high = objective(&path, &tau, 1e6);
        let arc = ArcLength::new(&path).unwrap();
        let top = tau
            .iter()
            .map(|&s| surface(&path.at(arc.parameter(s))).0)
            .fold(f64::NEG_INFINITY, f64::max);
        assert!((high.value - top).abs() < 1e-4, "{} {top}", high.value);
        let t: Vec<f64> = tau.iter().map(|&s| arc.parameter(s)).collect();
        let gradients: Vec<Col<f64>> = tau.iter().map(|_| col(&[0.0, 0.0])).collect();
        let weights = trapezoid_weights(&tau);
        let shifted = action(
            &path, &arc, &tau, &t, &weights, &[1e4; 6], &gradients, 272.0,
        )
        .unwrap();
        assert!(shifted.value.is_finite() && (shifted.value - 1e4).abs() < 0.1);
    }
}
