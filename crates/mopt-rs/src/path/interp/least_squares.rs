//! Nonlinear least squares without a Jacobian matrix.
//!
//! [`levenberg_marquardt`] minimizes `sum_i rho(r_i(x)^2) / 2` for residuals `r(x)` whose
//! Jacobian `J` is only known through its products `J v` and `J^T u`
//! ([`Linearization`]). A step solves `(J^T W J + lambda) d = -J^T W r` with conjugate
//! gradients (`W`: the weights of the loss, one for plain least squares), so the memory
//! is that of a few vectors and the work that of some products, whatever the size of
//! `J`. `lambda` follows the gain of the steps (Nielsen's rule: Madsen, Nielsen, Tingleff,
//! Methods for Non-Linear Least Squares Problems, 2004).
//!
//! The loss is the plain square or `soft_l1`, `rho(z) = 2 (sqrt(1 + z) - 1)`, which grows
//! like the absolute value for large residuals (the loss of SciPy's `least_squares` of
//! that name, here with the weights `rho'(z)` and without its second-order term).
//!
//! MultiOptPy: no counterpart (it calls `scipy.optimize.least_squares` with a dense
//! Jacobian).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

/// Residuals at a point and the products with their Jacobian there.
pub trait Linearization {
    /// The residuals `r`.
    fn residuals(&self) -> &[f64];
    /// `J v`.
    fn forward(&self, v: &[f64]) -> Vec<f64>;
    /// `J^T u`.
    fn backward(&self, u: &[f64]) -> Vec<f64>;
}

/// How a least-squares problem is solved.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LeastSquaresSettings {
    /// Most steps (accepted or not).
    pub max_iterations: usize,
    /// The largest component of the gradient at which the solution is found.
    pub gradient_tolerance: f64,
    /// The relative decrease of the function below which the solution is found.
    pub function_tolerance: f64,
    /// Use the loss `soft_l1` instead of the plain square.
    pub soft_l1: bool,
    /// Most products with `J^T W J` in the solution of one step.
    pub inner_iterations: usize,
}

impl Default for LeastSquaresSettings {
    fn default() -> Self {
        Self {
            max_iterations: 50,
            gradient_tolerance: 1e-8,
            function_tolerance: 1e-8,
            soft_l1: false,
            inner_iterations: 200,
        }
    }
}

/// What [`levenberg_marquardt`] found.
#[derive(Debug, Clone)]
pub struct LeastSquaresSolution {
    /// The point.
    pub x: Vec<f64>,
    /// The function there (`sum rho(r^2) / 2`).
    pub cost: f64,
    /// The largest component of its gradient.
    pub gradient: f64,
    /// Steps tried.
    pub iterations: usize,
    /// Whether a tolerance was met.
    pub converged: bool,
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// The function of the residuals `r` and the weights of the loss.
fn loss(r: &[f64], soft_l1: bool) -> (f64, Vec<f64>) {
    if !soft_l1 {
        return (0.5 * dot(r, r), vec![1.0; r.len()]);
    }
    let mut cost = 0.0;
    let weights = r
        .iter()
        .map(|value| {
            let root = (1.0 + value * value).sqrt();
            // 2 (sqrt(1 + z) - 1) = 2 z / (sqrt(1 + z) + 1): no cancellation for small z.
            cost += value * value / (root + 1.0);
            1.0 / root
        })
        .collect();
    (cost, weights)
}

/// `d` with `(J^T W J + lambda) d = -g` by conjugate gradients, to a residual of
/// `tolerance` times `|g|`.
fn step<L: Linearization>(
    at: &L,
    weights: &[f64],
    g: &[f64],
    lambda: f64,
    tolerance: f64,
    most: usize,
) -> Vec<f64> {
    let apply = |v: &[f64]| -> Vec<f64> {
        let mut through = at.forward(v);
        for (value, weight) in through.iter_mut().zip(weights) {
            *value *= weight;
        }
        let mut out = at.backward(&through);
        for (o, value) in out.iter_mut().zip(v) {
            *o += lambda * value;
        }
        out
    };
    let n = g.len();
    let mut d = vec![0.0; n];
    let mut residual: Vec<f64> = g.iter().map(|v| -v).collect();
    let mut direction = residual.clone();
    let mut size = dot(&residual, &residual);
    let target = tolerance * tolerance * size;
    for _ in 0..most.max(1) {
        if size <= target || size == 0.0 {
            break;
        }
        let image = apply(&direction);
        let curvature = dot(&direction, &image);
        if curvature <= 0.0 || !curvature.is_finite() {
            break;
        }
        let along = size / curvature;
        for k in 0..n {
            d[k] += along * direction[k];
            residual[k] -= along * image[k];
        }
        let next = dot(&residual, &residual);
        let ratio = next / size;
        size = next;
        for k in 0..n {
            direction[k] = residual[k] + ratio * direction[k];
        }
    }
    d
}

/// The minimum of `sum rho(r(x)^2) / 2` near `x0`; `at(x)` gives the residuals at `x` and
/// the products with their Jacobian.
pub fn levenberg_marquardt<L: Linearization>(
    at: impl Fn(&[f64]) -> L,
    x0: Vec<f64>,
    settings: &LeastSquaresSettings,
) -> LeastSquaresSolution {
    let mut x = x0;
    let mut here = at(&x);
    let (mut cost, mut weights) = loss(here.residuals(), settings.soft_l1);
    let gradient_of = |point: &L, weights: &[f64]| -> Vec<f64> {
        let weighted: Vec<f64> = point
            .residuals()
            .iter()
            .zip(weights)
            .map(|(r, w)| r * w)
            .collect();
        point.backward(&weighted)
    };
    let largest = |g: &[f64]| g.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let mut g = gradient_of(&here, &weights);
    // The damping starts at the size of the curvature along the gradient.
    let mut lambda = {
        let mut through = here.forward(&g);
        for (value, weight) in through.iter_mut().zip(&weights) {
            *value *= weight;
        }
        let gg = dot(&g, &g);
        if gg > 0.0 {
            1e-3 * dot(&through, &here.forward(&g)) / gg
        } else {
            1e-3
        }
    }
    .max(1e-12);
    let mut growth = 2.0;
    let mut out = LeastSquaresSolution {
        x: x.clone(),
        cost,
        gradient: largest(&g),
        iterations: 0,
        converged: false,
    };
    for iteration in 0..settings.max_iterations {
        out.iterations = iteration;
        if largest(&g) <= settings.gradient_tolerance {
            out.converged = true;
            break;
        }
        let d = step(&here, &weights, &g, lambda, 1e-2, settings.inner_iterations);
        let trial: Vec<f64> = x.iter().zip(&d).map(|(a, b)| a + b).collect();
        let there = at(&trial);
        let (trial_cost, trial_weights) = loss(there.residuals(), settings.soft_l1);
        // The decrease the quadratic model promises: d . (lambda d - g) / 2.
        let promised = 0.5
            * d.iter()
                .zip(&g)
                .map(|(step, slope)| step * (lambda * step - slope))
                .sum::<f64>();
        let gain = if promised > 0.0 && trial_cost.is_finite() {
            (cost - trial_cost) / promised
        } else {
            -1.0
        };
        out.iterations = iteration + 1;
        if gain > 0.0 {
            let decrease = cost - trial_cost;
            x = trial;
            here = there;
            weights = trial_weights;
            g = gradient_of(&here, &weights);
            lambda *= (1.0 - (2.0 * gain - 1.0).powi(3)).max(1.0 / 3.0);
            growth = 2.0;
            let small = decrease <= settings.function_tolerance * cost.max(f64::MIN_POSITIVE);
            cost = trial_cost;
            if small {
                out.converged = true;
                break;
            }
        } else {
            lambda *= growth;
            growth *= 2.0;
            if !lambda.is_finite() {
                break;
            }
        }
    }
    if largest(&g) <= settings.gradient_tolerance {
        out.converged = true;
    }
    out.x = x;
    out.cost = cost;
    out.gradient = largest(&g);
    out
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    /// Rosenbrock's function as two residuals, `10 (y - x^2)` and `1 - x`.
    struct Rosenbrock {
        x: f64,
        r: [f64; 2],
    }

    impl Linearization for Rosenbrock {
        fn residuals(&self) -> &[f64] {
            &self.r
        }
        fn forward(&self, v: &[f64]) -> Vec<f64> {
            vec![10.0 * (v[1] - 2.0 * self.x * v[0]), -v[0]]
        }
        fn backward(&self, u: &[f64]) -> Vec<f64> {
            vec![-20.0 * self.x * u[0] - u[1], 10.0 * u[0]]
        }
    }

    fn rosenbrock(x: &[f64]) -> Rosenbrock {
        Rosenbrock {
            x: x[0],
            r: [10.0 * (x[1] - x[0] * x[0]), 1.0 - x[0]],
        }
    }

    #[test]
    fn the_valley_of_rosenbrock_is_followed_to_its_minimum() {
        let settings = LeastSquaresSettings {
            max_iterations: 200,
            ..LeastSquaresSettings::default()
        };
        let found = levenberg_marquardt(rosenbrock, vec![-1.2, 1.0], &settings);
        assert!(found.converged, "{found:?}");
        assert!((found.x[0] - 1.0).abs() < 1e-6 && (found.x[1] - 1.0).abs() < 1e-6);
        assert!(found.cost < 1e-12 && found.iterations < 100);
        // The same minimum with the loss that grows more slowly.
        let soft = LeastSquaresSettings {
            soft_l1: true,
            ..settings
        };
        let found = levenberg_marquardt(rosenbrock, vec![-1.2, 1.0], &soft);
        assert!((found.x[0] - 1.0).abs() < 1e-5 && (found.x[1] - 1.0).abs() < 1e-5);
    }

    /// A line through points, one of them far off: residuals `a t + b - y`.
    struct Line {
        t: Vec<f64>,
        r: Vec<f64>,
    }

    impl Linearization for Line {
        fn residuals(&self) -> &[f64] {
            &self.r
        }
        fn forward(&self, v: &[f64]) -> Vec<f64> {
            self.t.iter().map(|t| v[0] * t + v[1]).collect()
        }
        fn backward(&self, u: &[f64]) -> Vec<f64> {
            vec![dot(u, &self.t), u.iter().sum()]
        }
    }

    #[test]
    fn the_soft_loss_gives_less_weight_to_a_point_far_off() {
        let t: Vec<f64> = (0..9).map(f64::from).collect();
        let mut y: Vec<f64> = t.iter().map(|t| 2.0 * t + 1.0).collect();
        y[4] += 50.0;
        let line = |x: &[f64]| Line {
            t: t.clone(),
            r: t.iter().zip(&y).map(|(t, y)| x[0] * t + x[1] - y).collect(),
        };
        let plain = levenberg_marquardt(line, vec![0.0, 0.0], &LeastSquaresSettings::default());
        // A linear problem: the least-squares line, lifted by the point far off.
        assert!(plain.converged);
        assert!((plain.x[0] - 2.0).abs() < 1e-6 && (plain.x[1] - (1.0 + 50.0 / 9.0)).abs() < 1e-5);
        let soft = levenberg_marquardt(
            line,
            vec![0.0, 0.0],
            &LeastSquaresSettings {
                soft_l1: true,
                max_iterations: 200,
                ..LeastSquaresSettings::default()
            },
        );
        assert!(
            (soft.x[1] - 1.0).abs() < 0.5 * (plain.x[1] - 1.0).abs(),
            "{soft:?}"
        );
    }
}
