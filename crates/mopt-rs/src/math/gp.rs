//! Gaussian-process regression of a function from its values and gradients.
//!
//! Squared-exponential kernel `k(x, x') = exp(-|x - x'|^2 / (2 l^2))` with a constant prior
//! mean; the observations are the values `E_a` and the gradients `g_a` at the training points,
//! whose covariances are the derivatives of the kernel (Rasmussen and Williams, Gaussian
//! Processes for Machine Learning, 2006, section 9.4; Denzel and Kastner, J. Chem. Phys. 2018,
//! 148, 094114). With `d = x_a - x_b` and `e = k(x_a, x_b)`:
//! ```text
//! cov(E_a, E_b) = e                         cov(E_a, g_b,j) = e d_j / l^2
//! cov(g_a,i, E_b) = -e d_i / l^2            cov(g_a,i, g_b,j) = e (delta_ij / l^2 - d_i d_j / l^4)
//! ```
//! A relative noise `s` regularizes the diagonal (`s^2` for values, `s^2 / l^2` for gradient
//! components), which makes the amplitude of the kernel irrelevant for the mean.
//!
//! Memory and time. Written out, the covariance matrix of `N` points in `D` dimensions has
//! `(N (D + 1))^2` elements: 450 MB for 25 points of a 100-atom molecule, and several copies
//! of it during the factorization. It is never formed here. All differences `d` lie in the
//! space spanned by the points relative to the first one, of dimension `R <= N - 1`. Gradient
//! components along a direction `u` orthogonal to that space have `u . d = 0`, so they are
//! uncorrelated with the values and with every other direction, and their covariance between
//! the points is `e / l^2` for every such `u`. The matrix is therefore block diagonal in a
//! rotated basis: one block of size `N (R + 1)` for the values and the gradient components in
//! the space of the points, and the same `N x N` block `(e_ab + s^2 delta_ab) / l^2` for each
//! of the `D - R` orthogonal directions. This is exact (compare J. de Roos, A. Gessner, P.
//! Hennig, "High-dimensional Gaussian process inference with derivatives", ICML 2021); the
//! fit takes `O(N D)` memory beside two small factors, and `O(N^2 D + (N R)^3)` time instead
//! of `O((N D)^3)`. The posterior mean and its gradient cost `O(N D)` per point.
//!
//! MultiOptPy: `Optimizer/gpr_step.py:GaussianProcessRegression` and the kernel code of
//! `Optimizer/gpmin.py:GPmin` (both fit values only).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::{Col, Mat};

use crate::error::{MoptError, MoptResult};
use crate::math::linalg::{col_dot, col_norm};

/// A difference between points shorter than this fraction of the length scale (after the
/// directions already found are removed) adds no direction to the space of the points.
const SAME_DIRECTION: f64 = 1e-10;

/// One training point: position, value, gradient.
#[derive(Debug, Clone)]
pub struct Observation {
    /// Position.
    pub x: Col<f64>,
    /// Function value.
    pub value: f64,
    /// Gradient.
    pub gradient: Col<f64>,
}

/// Cholesky factor `L` (`A = L L^T`, rows of the lower triangle) of a symmetric
/// positive-definite matrix, or `None`.
fn cholesky(a: &Mat<f64>) -> Option<Vec<Vec<f64>>> {
    let n = a.nrows();
    let mut l: Vec<Vec<f64>> = Vec::with_capacity(n);
    for i in 0..n {
        let mut row = Vec::with_capacity(i + 1);
        for (j, lj) in l.iter().enumerate() {
            let dot: f64 = row.iter().zip(lj).map(|(x, y): (&f64, &f64)| x * y).sum();
            row.push((a[(i, j)] - dot) / lj[j]);
        }
        let d = a[(i, i)] - row.iter().map(|x| x * x).sum::<f64>();
        if !d.is_finite() || d <= 0.0 {
            return None;
        }
        row.push(libm::sqrt(d));
        l.push(row);
    }
    Some(l)
}

/// Solve `L L^T x = b`.
fn cholesky_solve(l: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    let mut y = b.to_vec();
    for i in 0..n {
        let dot: f64 = l[i][..i].iter().zip(&y[..i]).map(|(a, c)| a * c).sum();
        y[i] = (y[i] - dot) / l[i][i];
    }
    for i in (0..n).rev() {
        y[i] /= l[i][i];
        let yi = y[i];
        for (k, lik) in l[i][..i].iter().enumerate() {
            y[k] -= lik * yi;
        }
    }
    y
}

/// Covariance matrix of the values and gradients at `points` (all of one dimension `dim`),
/// without noise: size `N (dim + 1)`, per point the value followed by the gradient.
fn covariance(points: &[Vec<f64>], dim: usize, l2: f64) -> Mat<f64> {
    let n_pts = points.len();
    let size = n_pts * (dim + 1);
    let mut k = Mat::<f64>::zeros(size, size);
    for a in 0..n_pts {
        for b in 0..=a {
            let d: Vec<f64> = (0..dim).map(|j| points[a][j] - points[b][j]).collect();
            let r2: f64 = d.iter().map(|v| v * v).sum();
            let e = libm::exp(-0.5 * r2 / l2);
            let (ra, rb) = (a * (dim + 1), b * (dim + 1));
            k[(ra, rb)] = e;
            for j in 0..dim {
                k[(ra, rb + 1 + j)] = e * d[j] / l2;
                k[(ra + 1 + j, rb)] = -e * d[j] / l2;
                for i in 0..dim {
                    let delta = if i == j { 1.0 / l2 } else { 0.0 };
                    k[(ra + 1 + i, rb + 1 + j)] = e * (delta - d[i] * d[j] / (l2 * l2));
                }
            }
        }
    }
    // Mirror the lower block triangle.
    for r in 0..size {
        for c in (r + 1)..size {
            k[(r, c)] = k[(c, r)];
        }
    }
    k
}

/// `k` with the noise on its diagonal: `noise^2` for values, `noise^2 / l^2` for gradients.
fn with_noise(k: &Mat<f64>, n_pts: usize, dim: usize, noise: f64, l2: f64) -> Mat<f64> {
    let mut reg = k.clone();
    for a in 0..n_pts {
        let r = a * (dim + 1);
        reg[(r, r)] += noise * noise;
        for j in 0..dim {
            reg[(r + 1 + j, r + 1 + j)] += noise * noise / l2;
        }
    }
    reg
}

/// Gaussian process trained on values and gradients.
#[derive(Debug, Clone)]
pub struct GradientGp {
    length_scale: f64,
    prior_mean: f64,
    points: Vec<Col<f64>>,
    /// Weights: per point one for the value and `D` for the gradient.
    weights: Vec<f64>,
    noise: f64,
    /// Orthonormal basis of the space of the points relative to the first one.
    basis: Vec<Col<f64>>,
    /// The points in that basis.
    reduced: Vec<Vec<f64>>,
    /// Factor of the covariance of the values and of the gradients within the basis.
    factor: Vec<Vec<f64>>,
    /// Factor of `e_ab + s^2 delta_ab`: `l^2` times the covariance of the gradient components
    /// along any direction orthogonal to the basis.
    factor_orthogonal: Vec<Vec<f64>>,
}

impl GradientGp {
    /// Fit to `observations` with kernel length `length_scale`, relative noise `noise`, and the
    /// constant prior mean `prior_mean`. If the covariance matrix is not positive definite in
    /// floating point (nearly coincident points), the noise is raised tenfold (from at least
    /// `1e-8`) up to six times.
    pub fn fit(
        observations: &[Observation],
        length_scale: f64,
        noise: f64,
        prior_mean: f64,
    ) -> MoptResult<Self> {
        let n_pts = observations.len();
        let dim = observations
            .first()
            .map(|o| o.x.nrows())
            .ok_or_else(|| MoptError::config("a Gaussian process needs at least one point"))?;
        let ok =
            length_scale.is_finite() && length_scale > 0.0 && noise.is_finite() && noise >= 0.0;
        if !ok
            || observations
                .iter()
                .any(|o| o.x.nrows() != dim || o.gradient.nrows() != dim)
        {
            return Err(MoptError::config(
                "Gaussian process: positive length scale, non-negative noise, and points of one \
                 dimension are required",
            ));
        }
        let l2 = length_scale * length_scale;
        // The space of the points: Gram-Schmidt (twice) of the differences to the first one.
        let origin = &observations[0].x;
        let mut basis: Vec<Col<f64>> = Vec::new();
        for o in &observations[1..] {
            let mut v = &o.x - origin;
            for _pass in 0..2 {
                for b in &basis {
                    let p = col_dot(b, &v);
                    for i in 0..dim {
                        v[i] -= p * b[i];
                    }
                }
            }
            let left = col_norm(&v);
            if left > SAME_DIRECTION * length_scale && basis.len() < dim {
                basis.push(Col::from_fn(dim, |i| v[i] / left));
            }
        }
        let rank = basis.len();
        let project = |v: &Col<f64>| -> Vec<f64> { basis.iter().map(|b| col_dot(b, v)).collect() };
        let reduced: Vec<Vec<f64>> = observations
            .iter()
            .map(|o| project(&(&o.x - origin)))
            .collect();
        // Values and gradient components within the space of the points.
        let mut inner = Vec::with_capacity(n_pts * (rank + 1));
        // Gradient components orthogonal to it, per point.
        let mut outer: Vec<Col<f64>> = Vec::with_capacity(n_pts);
        for o in observations {
            let along = project(&o.gradient);
            let mut rest = o.gradient.clone();
            for (b, c) in basis.iter().zip(&along) {
                for i in 0..dim {
                    rest[i] -= c * b[i];
                }
            }
            inner.push(o.value - prior_mean);
            inner.extend(along);
            outer.push(rest);
        }
        let k = covariance(&reduced, rank, l2);
        let kernel = Mat::from_fn(n_pts, n_pts, |a, b| k[(a * (rank + 1), b * (rank + 1))]);
        let mut noise_now = noise.max(1e-8);
        for _ in 0..7 {
            let factors = cholesky(&with_noise(&k, n_pts, rank, noise_now, l2)).zip(cholesky(
                &Mat::from_fn(n_pts, n_pts, |a, b| {
                    kernel[(a, b)] + if a == b { noise_now * noise_now } else { 0.0 }
                }),
            ));
            if let Some((factor, factor_orthogonal)) = factors {
                let inner_weights = cholesky_solve(&factor, &inner);
                // Orthogonal components: l^2 (e + s^2)^-1 applied coordinate by coordinate.
                let mut weights = vec![0.0; n_pts * (dim + 1)];
                let mut column = vec![0.0; n_pts];
                for i in 0..dim {
                    for (a, rest) in outer.iter().enumerate() {
                        column[a] = rest[i];
                    }
                    let solved = cholesky_solve(&factor_orthogonal, &column);
                    for a in 0..n_pts {
                        weights[a * (dim + 1) + 1 + i] = l2 * solved[a];
                    }
                }
                for a in 0..n_pts {
                    let w = &inner_weights[a * (rank + 1)..(a + 1) * (rank + 1)];
                    weights[a * (dim + 1)] = w[0];
                    for (b, c) in basis.iter().zip(&w[1..]) {
                        for i in 0..dim {
                            weights[a * (dim + 1) + 1 + i] += c * b[i];
                        }
                    }
                }
                if weights.iter().all(|w| w.is_finite()) {
                    return Ok(Self {
                        length_scale,
                        prior_mean,
                        points: observations.iter().map(|o| o.x.clone()).collect(),
                        weights,
                        noise: noise_now,
                        basis,
                        reduced,
                        factor,
                        factor_orthogonal,
                    });
                }
            }
            noise_now *= 10.0;
        }
        Err(MoptError::linalg(
            "Gaussian process: the covariance matrix is not positive definite",
        ))
    }

    /// Relative noise actually used (raised when the factorization required it).
    pub fn noise(&self) -> f64 {
        self.noise
    }

    /// Number of training points.
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Whether there are no training points (never true for a fitted process).
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Dimension of the space spanned by the training points relative to the first one (the
    /// size of the factorized block is `len() * (rank() + 1)`).
    pub fn rank(&self) -> usize {
        self.basis.len()
    }

    /// Posterior mean and its gradient at `x`.
    pub fn predict(&self, x: &Col<f64>) -> (f64, Col<f64>) {
        let dim = x.nrows();
        let l2 = self.length_scale * self.length_scale;
        let mut mean = self.prior_mean;
        let mut grad: Col<f64> = Col::zeros(dim);
        for (a, p) in self.points.iter().enumerate() {
            let w = &self.weights[a * (dim + 1)..(a + 1) * (dim + 1)];
            let d = x - p;
            let r2: f64 = d.iter().map(|v| v * v).sum();
            let e = libm::exp(-0.5 * r2 / l2);
            let wd: f64 = (0..dim).map(|j| w[1 + j] * d[j]).sum();
            mean += e * (w[0] + wd / l2);
            for i in 0..dim {
                grad[i] += e * (-w[0] * d[i] / l2 + w[1 + i] / l2 - d[i] * wd / (l2 * l2));
            }
        }
        (mean, grad)
    }

    /// Hessian of the posterior mean at `x`. With `d = x - x_a`, `e` the kernel and
    /// `phi = w_0 + w . d / l^2` for every training point:
    /// `H_ij = e / l^2 (d_i d_j phi / l^2 - (d_i w_j + d_j w_i) / l^2 - delta_ij phi)`.
    pub fn hessian(&self, x: &Col<f64>) -> Mat<f64> {
        let dim = x.nrows();
        let l2 = self.length_scale * self.length_scale;
        let mut h = Mat::<f64>::zeros(dim, dim);
        for (a, p) in self.points.iter().enumerate() {
            let w = &self.weights[a * (dim + 1)..(a + 1) * (dim + 1)];
            let d = x - p;
            let r2: f64 = d.iter().map(|v| v * v).sum();
            let e = libm::exp(-0.5 * r2 / l2);
            let wd: f64 = (0..dim).map(|j| w[1 + j] * d[j]).sum();
            let phi = w[0] + wd / l2;
            for i in 0..dim {
                for j in 0..dim {
                    let delta = if i == j { phi } else { 0.0 };
                    h[(i, j)] += e / l2
                        * (d[i] * d[j] * phi / l2
                            - (d[i] * w[1 + j] + d[j] * w[1 + i]) / l2
                            - delta);
                }
            }
        }
        h
    }

    /// Posterior variance of the value at `x`, in units of the kernel amplitude (between 0 at
    /// the training points and 1 far from them).
    pub fn variance(&self, x: &Col<f64>) -> f64 {
        let l2 = self.length_scale * self.length_scale;
        let rank = self.basis.len();
        // `x` relative to the first point: within the space of the points, and the rest.
        let from_origin = x - &self.points[0];
        let within: Vec<f64> = self
            .basis
            .iter()
            .map(|b| col_dot(b, &from_origin))
            .collect();
        let beyond2 = (col_dot(&from_origin, &from_origin)
            - within.iter().map(|c| c * c).sum::<f64>())
        .max(0.0);
        let mut inner = Vec::with_capacity(self.points.len() * (rank + 1));
        let mut outer = Vec::with_capacity(self.points.len());
        for (p, reduced) in self.points.iter().zip(&self.reduced) {
            let d = x - p;
            let e = libm::exp(-0.5 * col_dot(&d, &d) / l2);
            inner.push(e);
            inner.extend((0..rank).map(|k| e * (within[k] - reduced[k]) / l2));
            // The gradient component along the one orthogonal direction that `x` singles out.
            outer.push(e * libm::sqrt(beyond2) / l2);
        }
        let solved = cholesky_solve(&self.factor, &inner);
        let mut explained: f64 = inner.iter().zip(&solved).map(|(a, b)| a * b).sum();
        let solved = cholesky_solve(&self.factor_orthogonal, &outer);
        explained += l2 * outer.iter().zip(&solved).map(|(a, b)| a * b).sum::<f64>();
        (1.0 - explained).clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f(x: &Col<f64>) -> (f64, Col<f64>) {
        // A smooth test function: sin(x0) cos(x1) + 0.1 x0^2.
        let v = x[0].sin() * x[1].cos() + 0.1 * x[0] * x[0];
        let g = Col::from_fn(2, |i| {
            if i == 0 {
                x[0].cos() * x[1].cos() + 0.2 * x[0]
            } else {
                -x[0].sin() * x[1].sin()
            }
        });
        (v, g)
    }

    fn observe(points: &[[f64; 2]]) -> Vec<Observation> {
        points
            .iter()
            .map(|p| {
                let x = Col::from_fn(2, |i| p[i]);
                let (value, gradient) = f(&x);
                Observation { x, value, gradient }
            })
            .collect()
    }

    #[test]
    fn training_values_and_gradients_are_reproduced() {
        let obs = observe(&[[0.0, 0.0], [0.5, 0.2], [-0.3, 0.6], [0.4, -0.5]]);
        let gp = GradientGp::fit(&obs, 0.8, 1e-6, 0.0).unwrap();
        for o in &obs {
            let (m, g) = gp.predict(&o.x);
            assert!((m - o.value).abs() < 1e-6, "{m} vs {}", o.value);
            for i in 0..2 {
                assert!((g[i] - o.gradient[i]).abs() < 1e-5);
            }
            assert!(gp.variance(&o.x) < 1e-6);
        }
        // Far from the data: the prior mean and the full variance.
        let far = Col::from_fn(2, |_| 50.0);
        let gp = GradientGp::fit(&obs, 0.8, 1e-6, 3.0).unwrap();
        let (m, g) = gp.predict(&far);
        assert!((m - 3.0).abs() < 1e-12 && g[0].abs() < 1e-12);
        assert!((gp.variance(&far) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn predicted_gradient_is_the_derivative_of_the_predicted_mean() {
        let obs = observe(&[[0.0, 0.0], [0.5, 0.2], [-0.3, 0.6]]);
        let gp = GradientGp::fit(&obs, 0.7, 1e-3, 0.2).unwrap();
        let x = Col::from_fn(2, |i| [0.21, 0.33][i]);
        let (m, g) = gp.predict(&x);
        // Between the points the interpolation is close to the function.
        assert!((m - f(&x).0).abs() < 0.05, "{m} vs {}", f(&x).0);
        let h = 1e-6;
        let hessian = gp.hessian(&x);
        for i in 0..2 {
            let mut xp = x.clone();
            xp[i] += h;
            let mut xm = x.clone();
            xm[i] -= h;
            let fd = (gp.predict(&xp).0 - gp.predict(&xm).0) / (2.0 * h);
            assert!((fd - g[i]).abs() < 1e-7, "{i}: {fd} vs {}", g[i]);
            // The Hessian is the derivative of the predicted gradient.
            let (gp_plus, gp_minus) = (gp.predict(&xp).1, gp.predict(&xm).1);
            for j in 0..2 {
                let fd = (gp_plus[j] - gp_minus[j]) / (2.0 * h);
                assert!((fd - hessian[(j, i)]).abs() < 1e-6, "{j}{i}: {fd}");
            }
        }
    }

    #[test]
    fn coincident_points_raise_the_noise_instead_of_failing() {
        let obs = observe(&[[0.1, 0.2], [0.1, 0.2], [0.1, 0.2]]);
        let gp = GradientGp::fit(&obs, 0.5, 0.0, 0.0).unwrap();
        assert!(gp.noise() > 0.0);
        let (m, _) = gp.predict(&obs[0].x);
        assert!((m - obs[0].value).abs() < 1e-3);
        assert!(GradientGp::fit(&[], 0.5, 0.0, 0.0).is_err());
        assert!(GradientGp::fit(&obs, -1.0, 0.0, 0.0).is_err());
    }

    /// The fit with the full covariance matrix written out: weights and variance at `x`.
    fn written_out(obs: &[Observation], l: f64, noise: f64, x: &Col<f64>) -> (Vec<f64>, f64) {
        let dim = x.nrows();
        let l2 = l * l;
        let points: Vec<Vec<f64>> = obs.iter().map(|o| o.x.iter().copied().collect()).collect();
        let k = with_noise(&covariance(&points, dim, l2), obs.len(), dim, noise, l2);
        let factor = cholesky(&k).unwrap();
        let mut y = Vec::new();
        let mut cross = Vec::new();
        for o in obs {
            y.push(o.value);
            y.extend(o.gradient.iter().copied());
            let d = x - &o.x;
            let e = libm::exp(-0.5 * col_dot(&d, &d) / l2);
            cross.push(e);
            cross.extend(d.iter().map(|v| e * v / l2));
        }
        let solved = cholesky_solve(&factor, &cross);
        let explained: f64 = cross.iter().zip(&solved).map(|(a, b)| a * b).sum();
        (cholesky_solve(&factor, &y), 1.0 - explained)
    }

    #[test]
    fn the_block_form_is_the_full_covariance_matrix() {
        // Four points in seven dimensions: the points span three, four directions are
        // orthogonal to them.
        let dim = 7;
        let obs: Vec<Observation> = (0..4)
            .map(|a| {
                let x = Col::from_fn(dim, |i| 0.4 * (((a * a + 1) * (i + 1)) as f64 * 0.37).sin());
                let gradient = Col::from_fn(dim, |i| ((7 * a + 2 * i) as f64 * 0.9).cos());
                Observation {
                    x,
                    value: 0.3 * a as f64 - 0.5,
                    gradient,
                }
            })
            .collect();
        let gp = GradientGp::fit(&obs, 0.9, 1e-3, 0.0).unwrap();
        assert_eq!(gp.rank(), 3);
        let x = Col::from_fn(dim, |i| 0.2 * (i as f64 * 1.3).cos());
        let (weights, variance) = written_out(&obs, 0.9, 1e-3, &x);
        for (a, b) in gp.weights.iter().zip(&weights) {
            assert!((a - b).abs() < 1e-9 * (1.0 + b.abs()), "{a} vs {b}");
        }
        assert!((gp.variance(&x) - variance).abs() < 1e-10);
        // Points on a line in many dimensions: a block of 3 x 2 instead of 3 x 201.
        let dim = 200;
        let obs: Vec<Observation> = (0..3)
            .map(|a| Observation {
                x: Col::from_fn(dim, |i| a as f64 * 0.1 * (i as f64 * 0.37).sin()),
                value: a as f64,
                gradient: Col::from_fn(dim, |i| (i as f64 + a as f64).cos()),
            })
            .collect();
        let gp = GradientGp::fit(&obs, 0.8, 1e-4, 0.0).unwrap();
        assert_eq!((gp.rank(), gp.factor.len()), (1, 6));
        let x = Col::from_fn(dim, |i| 0.05 * (i as f64 * 0.11).cos());
        let (weights, variance) = written_out(&obs, 0.8, 1e-4, &x);
        for (a, b) in gp.weights.iter().zip(&weights) {
            assert!((a - b).abs() < 1e-8 * (1.0 + b.abs()), "{a} vs {b}");
        }
        assert!((gp.variance(&x) - variance).abs() < 1e-9);
    }
}
