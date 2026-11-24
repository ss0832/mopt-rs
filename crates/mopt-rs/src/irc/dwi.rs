//! Distance-weighted interpolant (DWI) of the surface between two points, from the energy,
//! gradient, and Hessian at each: `E(x) = w1(x) T1(x) + w2(x) T2(x)` with the second-order
//! Taylor expansions `T_i` about the points and the weights
//! `w1 = |x - x2|^2 / (|x - x1|^2 + |x - x2|^2)`, `w2 = 1 - w1`.
//!
//! Reference: H. P. Hratchian and H. B. Schlegel, J. Chem. Phys. 2004, 120, 9918 (section
//! II D).
//!
//! MultiOptPy: `IRC/hpc.py:DWISurface` and `IRC/hpc.py:corrector_step`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::{Col, Mat};

use super::types::{dot, norm};

/// A point with the surface expanded to second order about it.
#[derive(Debug, Clone)]
pub struct TaylorPoint {
    /// The point.
    pub x: Col<f64>,
    /// Energy there.
    pub energy: f64,
    /// Gradient there.
    pub gradient: Col<f64>,
    /// Hessian there.
    pub hessian: Mat<f64>,
}

impl TaylorPoint {
    fn energy_at(&self, x: &Col<f64>) -> f64 {
        let dx = x - &self.x;
        self.energy + dot(&self.gradient, &dx) + 0.5 * dot(&dx, &(&self.hessian * &dx))
    }

    fn gradient_at(&self, x: &Col<f64>) -> Col<f64> {
        &self.gradient + &self.hessian * (x - &self.x)
    }
}

/// The interpolant between two points.
#[derive(Debug, Clone)]
pub struct DwiSurface {
    first: TaylorPoint,
    second: TaylorPoint,
}

impl DwiSurface {
    /// The surface through `first` and `second`.
    pub fn new(first: TaylorPoint, second: TaylorPoint) -> Self {
        Self { first, second }
    }

    /// The weights of the two expansions at `x` (one half each where the points coincide
    /// with `x`, as in MultiOptPy).
    pub fn weights(&self, x: &Col<f64>) -> (f64, f64) {
        let d1 = x - &self.first.x;
        let d2 = x - &self.second.x;
        let (n1, n2) = (dot(&d1, &d1), dot(&d2, &d2));
        let d = n1 + n2;
        if d < 1e-12 {
            (0.5, 0.5)
        } else {
            (n2 / d, n1 / d)
        }
    }

    /// Energy of the interpolant at `x`.
    pub fn energy(&self, x: &Col<f64>) -> f64 {
        let (w1, w2) = self.weights(x);
        w1 * self.first.energy_at(x) + w2 * self.second.energy_at(x)
    }

    /// Gradient of the interpolant at `x`.
    pub fn gradient(&self, x: &Col<f64>) -> Col<f64> {
        let d1 = x - &self.first.x;
        let d2 = x - &self.second.x;
        let (n1, n2) = (dot(&d1, &d1), dot(&d2, &d2));
        let d = n1 + n2;
        let (w1, w2) = self.weights(x);
        let (t1, t2) = (self.first.energy_at(x), self.second.energy_at(x));
        let mut g = self.first.gradient_at(x) * w1 + self.second.gradient_at(x) * w2;
        if d >= 1e-12 {
            // grad w1 = (grad n2 * d - n2 * grad d) / d^2, grad w2 = -grad w1.
            let n = x.nrows();
            let weight_gradient = Col::from_fn(n, |i| {
                (2.0 * d2[i] * d - n2 * 2.0 * (d1[i] + d2[i])) / (d * d)
            });
            g += weight_gradient * (t1 - t2);
        }
        g
    }

    /// The point reached from `start` on the steepest-descent path of the interpolant
    /// after the length `length`, in Euler steps of `length / steps` (it stops where the
    /// gradient norm falls below `1e-9`, as in MultiOptPy).
    ///
    /// A step that would raise the energy is halved instead of taken, so that the descent
    /// ends in a minimum of the interpolant when one lies closer than `length`; MultiOptPy
    /// goes on stepping back and forth across it, and its paths do not arrive
    /// (`docs/source/migration_paths.md`). Away from a minimum the two are the same.
    pub fn descend(&self, start: &Col<f64>, length: f64, steps: usize) -> Col<f64> {
        let mut h = length / steps.max(1) as f64;
        let smallest = 1e-10 * h;
        let mut remaining = length;
        let mut x = start.clone();
        let mut energy = self.energy(&x);
        // Every pass takes a step or halves it: the number of passes is bounded.
        for _ in 0..(steps.max(1) + 64) {
            if remaining <= smallest || h <= smallest {
                break;
            }
            let g = self.gradient(&x);
            let size = norm(&g);
            if size < 1e-9 {
                break;
            }
            let taken = h.min(remaining);
            let trial = &x - &g * (taken / size);
            let trial_energy = self.energy(&trial);
            if trial_energy <= energy {
                x = trial;
                energy = trial_energy;
                remaining -= taken;
            } else {
                h *= 0.5;
            }
        }
        x
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn point(x: [f64; 2]) -> TaylorPoint {
        // E = x0^3 / 3 + x0 x1 + x1^2: exact values of a cubic surface.
        let (a, b) = (x[0], x[1]);
        TaylorPoint {
            x: Col::from_fn(2, |i| x[i]),
            energy: a * a * a / 3.0 + a * b + b * b,
            gradient: Col::from_fn(2, |i| [a * a + b, a + 2.0 * b][i]),
            hessian: Mat::from_fn(2, 2, |i, j| [[2.0 * a, 1.0], [1.0, 2.0]][i][j]),
        }
    }

    #[test]
    fn interpolates_values_and_gradients_at_both_points() {
        let (p1, p2) = (point([0.3, -0.2]), point([0.5, 0.1]));
        let s = DwiSurface::new(p1.clone(), p2.clone());
        for p in [&p1, &p2] {
            assert!((s.energy(&p.x) - p.energy).abs() < 1e-14);
            assert!(norm(&(s.gradient(&p.x) - &p.gradient)) < 1e-14);
        }
        // The gradient is the derivative of the energy in between as well.
        let x = Col::from_fn(2, |i| [0.41, -0.07][i]);
        let g = s.gradient(&x);
        for k in 0..2 {
            let h = 1e-6;
            let mut up = x.clone();
            let mut down = x.clone();
            up[k] += h;
            down[k] -= h;
            let numeric = (s.energy(&up) - s.energy(&down)) / (2.0 * h);
            assert!((g[k] - numeric).abs() < 1e-8);
        }
    }

    #[test]
    fn descent_has_the_length_asked_for() {
        let s = DwiSurface::new(point([0.3, -0.2]), point([0.5, 0.1]));
        let start = Col::from_fn(2, |i| [0.5, 0.1][i]);
        let end = s.descend(&start, 0.1, 100);
        let chord = norm(&(&end - &start));
        assert!(chord <= 0.1 + 1e-12 && chord > 0.09);
        assert!(s.energy(&end) < s.energy(&start));
    }
}
