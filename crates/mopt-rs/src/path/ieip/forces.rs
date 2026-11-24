//! The displacements of one iEIP iteration, as MultiOptPy composes them.
//!
//! With `N` the unit vector from the image lower in energy to the higher one, `L` the
//! separation of the images, and `Lt = max(0.9 L, 0.02)` the separation asked for, each image
//! moves by
//!
//! - minus the gradient across the line of the images, `g - (g . N) N`;
//! - the energy difference along the line, `N |E1 - E2| / L` (the first image along `N`,
//!   the second against it);
//! - towards the other image, `(x2 - x1) (L - Lt) / L`;
//! - minus its whole gradient;
//! - away from where it started,
//!
//! each made a displacement of at most 0.04 Bohr (the one away from the start of exactly
//! 0.04 Bohr). From the second iteration on, the attraction of the images and the push away
//! from the start are weighted after DS-AFIR (S. Maeda, Y. Harabuchi, M. Takagi, K. Saita, K.
//! Suzuki, T. Ichino, Y. Sumiya, K. Sugiyama, Y. Ono, J. Comput. Chem. 2018, 39, 233, eqs.
//! 6-10): with `Z = |x - x0| / |x - x'| + cos(x - x0, x - x')`, `Y = Z / (Z + 1) + 1/2`
//! (`1/2` for `Z <= 0`; MultiOptPy's form: the paper has `Z / (Z + 1)` and 0), `u = Y (x - x')
//! / |x - x'| - (1 - Y) (x - x0) / |x - x0|` and `X = beta / |u| - (g . u) / |u|^2`, the
//! attraction is multiplied by `X Y` and the push by `X (1 - Y)`. The sum is the input of an
//! AdaBelief step (Zhuang et al., arXiv:2010.07468) without bias correction.
//!
//! MultiOptPy: `OtherMethod/elastic_image_pair.py:ElasticImagePair` (`iteration`,
//! `displacement`, `displacement_prime`, `perpendicular_force`, `delta_energy_force`,
//! `close_target_force`, `norm_dist_2imgs`, `dist_2imgs`, `target_dist_2imgs`). The weights of
//! DS-AFIR divide by `|x - x0|`, which is zero for an image that has not moved (MultiOptPy then
//! goes on with NaN); here such a direction counts as zero.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Col;

use super::settings::{
    ADABELIEF_BETA_M, ADABELIEF_BETA_V, ADABELIEF_EPSILON, ADABELIEF_RATE, DISPLACEMENT_LIMIT,
    SEPARATION_FLOOR_BASE, SEPARATION_SHARE,
};
use crate::math::linalg::{col_dot, col_norm};

/// Added to lengths that are divided by (MultiOptPy's `1e-10`).
const TINY: f64 = 1e-10;

/// `force` as a displacement of at most [`DISPLACEMENT_LIMIT`] (MultiOptPy `displacement`;
/// the operations in its order).
pub fn limited(force: &Col<f64>) -> Col<f64> {
    let n = col_norm(force);
    let length = n.min(DISPLACEMENT_LIMIT);
    Col::from_fn(force.nrows(), |i| force[i] / (n + TINY) * length)
}

/// `force` as a displacement of [`DISPLACEMENT_LIMIT`] (MultiOptPy `displacement_prime`).
pub fn fixed_length(force: &Col<f64>) -> Col<f64> {
    let n = col_norm(force);
    Col::from_fn(force.nrows(), |i| {
        force[i] / (n + TINY) * DISPLACEMENT_LIMIT
    })
}

/// `v / s` component by component.
fn divided(v: &Col<f64>, s: f64) -> Col<f64> {
    Col::from_fn(v.nrows(), |i| v[i] / s)
}

/// The surface at one image: the calculator alone and with the bias potentials.
#[derive(Debug, Clone)]
pub struct ImageValue {
    /// Energy of the calculator (Hartree).
    pub energy: f64,
    /// Its gradient.
    pub gradient: Col<f64>,
    /// Energy with the bias potentials.
    pub biased_energy: f64,
    /// Gradient with the bias potentials.
    pub biased_gradient: Col<f64>,
}

impl ImageValue {
    /// The same values with and without bias potentials.
    pub fn plain(energy: f64, gradient: Col<f64>) -> Self {
        Self {
            energy,
            biased_energy: energy,
            biased_gradient: gradient.clone(),
            gradient,
        }
    }
}

/// The parts of the displacements of one iteration, for the report.
#[derive(Debug, Clone)]
pub struct Parts {
    /// Gradient across the line of the images, per image.
    pub perpendicular: [Col<f64>; 2],
    /// The energy difference along the line (the same for both images).
    pub energy_difference: Col<f64>,
    /// The attraction of the images (before the DS-AFIR weights).
    pub attraction: Col<f64>,
    /// Sum of the displacements, per image (the input of AdaBelief).
    pub total: [Col<f64>; 2],
    /// The DS-AFIR weights `X` and `Y` per image (from the second iteration on).
    pub weights: Option<[(f64, f64); 2]>,
}

/// Unit vector of `v` (zero for a zero vector).
fn unit(v: &Col<f64>) -> Col<f64> {
    let n = col_norm(v);
    if n > 0.0 {
        divided(v, n)
    } else {
        v * 0.0
    }
}

/// The DS-AFIR weights `(X, Y)` of image `x` with the other image `other`, its start `x0`,
/// and its gradient `g` (of the calculator alone, as in MultiOptPy).
fn weights(x: &Col<f64>, other: &Col<f64>, x0: &Col<f64>, g: &Col<f64>, beta: f64) -> (f64, f64) {
    let to_other = x - other;
    let from_start = x - x0;
    let (a, b) = (col_norm(&from_start), col_norm(&to_other));
    let z = if a > 0.0 && b > 0.0 {
        a / b + col_dot(&from_start, &to_other) / (a * b)
    } else if b > 0.0 {
        0.0
    } else {
        // Coinciding images: only the push away from the start is left.
        f64::INFINITY
    };
    let y = if z > 0.0 {
        if z.is_finite() {
            z / (z + 1.0) + 0.5
        } else {
            1.5
        }
    } else {
        0.5
    };
    let u = unit(&to_other) * y - unit(&from_start) * (1.0 - y);
    let un = col_norm(&u);
    if un == 0.0 {
        return (0.0, y);
    }
    (beta / un - col_dot(g, &u) / (un * un), y)
}

/// `(v * a) * b` component by component (MultiOptPy's `v * a * b`).
fn scaled_twice(v: &Col<f64>, a: f64, b: f64) -> Col<f64> {
    Col::from_fn(v.nrows(), |i| v[i] * a * b)
}

/// The displacements of iteration `iteration` for the images `x` (as evaluated) with the
/// surface `values`, the starting images `start`, and `beta`.
pub fn displacements(
    iteration: usize,
    x: [&Col<f64>; 2],
    values: [&ImageValue; 2],
    start: [&Col<f64>; 2],
    beta: f64,
) -> Parts {
    let [x1, x2] = x;
    let [v1, v2] = values;
    // From the image lower in energy (without bias potentials) to the higher one.
    let (low, high) = if v2.energy > v1.energy {
        (x1, x2)
    } else {
        (x2, x1)
    };
    let separation = col_norm(&(high - low)) + TINY;
    let n = divided(&(high - low), separation);
    let target = (SEPARATION_SHARE * separation).max(SEPARATION_FLOOR_BASE - 0.01);
    let across = |g: &Col<f64>| g - &n * col_dot(g, &n);
    let perpendicular = [across(&v1.biased_gradient), across(&v2.biased_gradient)];
    let gap = (v1.biased_energy - v2.biased_energy).abs();
    let energy_difference = Col::from_fn(n.nrows(), |i| n[i] * gap / separation);
    let between = x2 - x1;
    let attraction = Col::from_fn(between.nrows(), |i| {
        between[i] * (separation - target) / separation
    });
    let along = limited(&energy_difference);
    let pull = limited(&attraction);
    let (pull1, pull2, push1, push2, w) = if iteration == 0 {
        let zero = x1 * 0.0;
        (pull.clone(), pull, zero.clone(), zero, None)
    } else {
        let (x_1, y_1) = weights(x1, x2, start[0], &v1.gradient, beta);
        let (x_2, y_2) = weights(x2, x1, start[1], &v2.gradient, beta);
        let push1 = fixed_length(&(x1 - start[0])) * (x_1 * (1.0 - y_1));
        let push2 = fixed_length(&(x2 - start[1])) * (x_2 * (1.0 - y_2));
        (
            scaled_twice(&pull, x_1, y_1),
            scaled_twice(&pull, x_2, y_2),
            push1,
            push2,
            Some([(x_1, y_1), (x_2, y_2)]),
        )
    };
    // The sums in MultiOptPy's order.
    let total1 = &along - limited(&perpendicular[0]) + pull1 - limited(&v1.biased_gradient) + push1;
    let total2 =
        limited(&perpendicular[1]) * -1.0 - &along - pull2 - limited(&v2.biased_gradient) + push2;
    Parts {
        perpendicular,
        energy_difference,
        attraction,
        total: [total1, total2],
        weights: w,
    }
}

/// The moments of AdaBelief of one image.
#[derive(Debug, Clone)]
pub struct Moments {
    /// Mean of the displacements.
    pub mean: Col<f64>,
    /// Variance of the displacements about their mean.
    pub variance: Col<f64>,
}

impl Moments {
    /// Zero moments for `n` variables.
    pub fn zeros(n: usize) -> Self {
        Self {
            mean: Col::zeros(n),
            variance: Col::zeros(n),
        }
    }

    /// Take in the displacement `d` and return the step (MultiOptPy's AdaBelief without bias
    /// correction).
    pub fn step(&mut self, d: &Col<f64>) -> Col<f64> {
        let n = d.nrows();
        for i in 0..n {
            self.mean[i] = ADABELIEF_BETA_M * self.mean[i] + (1.0 - ADABELIEF_BETA_M) * d[i];
            let deviation = d[i] - self.mean[i];
            self.variance[i] = ADABELIEF_BETA_V * self.variance[i]
                + (1.0 - ADABELIEF_BETA_V) * deviation * deviation;
        }
        Col::from_fn(n, |i| {
            ADABELIEF_RATE * self.mean[i] / (libm::sqrt(self.variance[i]) + ADABELIEF_EPSILON)
        })
    }
}

/// MultiOptPy's `kabsch_algorithm` on two structures of flat Cartesian coordinates: both are
/// centred on their (unweighted) centroids and the first is turned onto the second.
pub fn aligned_pair(p: &Col<f64>, q: &Col<f64>) -> (Col<f64>, Col<f64>) {
    let atoms = p.nrows() / 3;
    if atoms == 0 || q.nrows() != p.nrows() {
        return (p.clone(), q.clone());
    }
    let centroid = |v: &Col<f64>| {
        let mut c = [0.0; 3];
        for i in 0..atoms {
            for k in 0..3 {
                c[k] += v[3 * i + k];
            }
        }
        c.map(|s| s / atoms as f64)
    };
    let (cp, cq) = (centroid(p), centroid(q));
    let pc = Col::from_fn(p.nrows(), |j| p[j] - cp[j % 3]);
    let qc = Col::from_fn(q.nrows(), |j| q[j] - cq[j % 3]);
    let mut cov = faer::Mat::<f64>::zeros(3, 3);
    for i in 0..atoms {
        for r in 0..3 {
            for c in 0..3 {
                cov[(r, c)] += pc[3 * i + r] * qc[3 * i + c];
            }
        }
    }
    let rot = crate::coord::alignment::proper_rotation_from_covariance(&cov);
    let turned = Col::from_fn(p.nrows(), |j| {
        let (i, r) = (j / 3, j % 3);
        (0..3).map(|c| rot[(r, c)] * pc[3 * i + c]).sum()
    });
    (turned, qc)
}
