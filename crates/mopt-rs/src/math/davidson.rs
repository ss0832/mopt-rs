//! The lowest eigenpair of a symmetric operator that is given by its action on a vector
//! (Davidson's method).
//!
//! The matrix is never formed: memory is the vectors of the search space (at most
//! `max_subspace` of them and their images), which is what makes it usable where the
//! matrix would not fit, such as the Hessian of all images of a chain at once.
//!
//! Each iteration applies the operator once. The lowest eigenpair `(theta, x)` of the
//! operator within the span of the search vectors is found by diagonalizing the small
//! projected matrix; its residual `r = A x - theta x` is divided by `diagonal - theta`
//! where an approximate diagonal is given (Davidson's correction; without one the
//! residual itself is the new direction, which is a Lanczos-like search) and added to the
//! span after orthogonalization. When the span is full it is collapsed to its lowest
//! eigenvectors.
//!
//! Reference: E. R. Davidson, J. Comput. Phys. 1975, 17, 87.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::{Col, Mat};

use super::linalg::{col_dot, col_norm, symmetric_eigen};
use crate::error::{MoptError, MoptResult};

/// Settings of [`lowest_eigenpair`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DavidsonSettings {
    /// Applications of the operator at most.
    pub max_iterations: usize,
    /// Norm of the residual at which the pair is accepted.
    pub tolerance: f64,
    /// Vectors of the search space before it is collapsed (at least 3).
    pub max_subspace: usize,
}

impl Default for DavidsonSettings {
    fn default() -> Self {
        Self {
            max_iterations: 200,
            tolerance: 1e-8,
            max_subspace: 20,
        }
    }
}

/// The lowest eigenpair as far as it was found.
#[derive(Debug, Clone)]
pub struct LowestPair {
    /// The eigenvalue (the Rayleigh quotient of `vector`).
    pub value: f64,
    /// The eigenvector, of unit length.
    pub vector: Col<f64>,
    /// Norm of `A x - theta x`.
    pub residual: f64,
    /// Applications of the operator.
    pub applications: usize,
    /// Whether the residual is below the tolerance.
    pub converged: bool,
}

/// Smallest magnitude of `diagonal - theta` in the correction.
const SMALLEST_DENOMINATOR: f64 = 1e-4;
/// Eigenvectors of the search space that are kept when it is collapsed.
const KEPT_AT_COLLAPSE: usize = 5;
/// A new direction shorter than this after orthogonalization adds nothing to the span.
const NOTHING_NEW: f64 = 1e-10;

/// `v` without its components along the columns of `basis`, twice (the second pass removes
/// what rounding left of the first).
fn orthogonalized(mut v: Col<f64>, basis: &[Col<f64>]) -> Col<f64> {
    for _ in 0..2 {
        for b in basis {
            let overlap = col_dot(b, &v);
            v -= b * overlap;
        }
    }
    v
}

/// The lowest eigenpair of the symmetric operator `apply` on vectors of the length of
/// `start`, from the guess `start`. `diagonal` is an approximation of the diagonal of the
/// operator for the correction (`None`: none).
pub fn lowest_eigenpair(
    apply: &mut dyn FnMut(&Col<f64>) -> MoptResult<Col<f64>>,
    diagonal: Option<&Col<f64>>,
    start: &Col<f64>,
    settings: &DavidsonSettings,
) -> MoptResult<LowestPair> {
    if diagonal.is_some_and(|d| d.nrows() != start.nrows()) {
        return Err(MoptError::linalg(
            "Davidson: the diagonal has another length than the vectors",
        ));
    }
    let mut correct = |residual: &Col<f64>, theta: f64| -> Col<f64> {
        match diagonal {
            Some(d) => Col::from_fn(residual.nrows(), |r| {
                let denominator = d[r] - theta;
                let safe = if denominator.abs() < SMALLEST_DENOMINATOR {
                    SMALLEST_DENOMINATOR.copysign(if denominator == 0.0 {
                        1.0
                    } else {
                        denominator
                    })
                } else {
                    denominator
                };
                -residual[r] / safe
            }),
            None => residual.clone(),
        }
    };
    lowest_eigenpair_preconditioned(apply, &mut correct, start, settings)
}

/// [`lowest_eigenpair`] with the correction of the residual given by `correct(residual,
/// theta)`: an approximation of `(A - theta)^-1 residual` (an approximate inverse of the
/// operator, such as that of a quasi-Newton method for a Hessian, turns the search into a
/// preconditioned inverse iteration that finds the lowest pair in few applications).
pub fn lowest_eigenpair_preconditioned(
    apply: &mut dyn FnMut(&Col<f64>) -> MoptResult<Col<f64>>,
    correct: &mut dyn FnMut(&Col<f64>, f64) -> Col<f64>,
    start: &Col<f64>,
    settings: &DavidsonSettings,
) -> MoptResult<LowestPair> {
    let n = start.nrows();
    let length = col_norm(start);
    if n == 0 || !length.is_finite() || length == 0.0 {
        return Err(MoptError::linalg(
            "Davidson: the starting vector is empty, zero, or not finite",
        ));
    }
    let most = settings.max_subspace.max(3);
    let mut basis: Vec<Col<f64>> = vec![start * (1.0 / length)];
    let mut images: Vec<Col<f64>> = Vec::new();
    let mut applications = 0;
    loop {
        // The image of the newest search vector.
        let newest = &basis[basis.len() - 1];
        let image = apply(newest)?;
        applications += 1;
        if image.nrows() != n || !image.iter().all(|v| v.is_finite()) {
            return Err(MoptError::linalg(
                "Davidson: the operator returned a vector of another length or not finite",
            ));
        }
        images.push(image);
        let k = basis.len();
        let mut small = Mat::<f64>::zeros(k, k);
        for i in 0..k {
            for j in 0..=i {
                // Symmetric by construction of the operator; the mean removes rounding.
                let value = 0.5 * (col_dot(&basis[i], &images[j]) + col_dot(&basis[j], &images[i]));
                small[(i, j)] = value;
                small[(j, i)] = value;
            }
        }
        let (values, vectors) = symmetric_eigen(&small)?;
        let theta = values[0];
        let mut x = Col::<f64>::zeros(n);
        let mut ax = Col::<f64>::zeros(n);
        for i in 0..k {
            x += &basis[i] * vectors[(i, 0)];
            ax += &images[i] * vectors[(i, 0)];
        }
        let residual = &ax - &x * theta;
        let norm = col_norm(&residual);
        let converged = norm <= settings.tolerance;
        let pair = LowestPair {
            value: theta,
            vector: x.clone(),
            residual: norm,
            applications,
            converged,
        };
        if converged || applications >= settings.max_iterations || k >= n {
            return Ok(pair);
        }
        // The new direction.
        let correction = correct(&residual, theta);
        if correction.nrows() != n || !correction.iter().all(|v| v.is_finite()) {
            return Err(MoptError::linalg(
                "Davidson: the correction has another length or is not finite",
            ));
        }
        if k >= most {
            // Collapse the span to its lowest eigenvectors (they are orthonormal: the
            // search vectors are, and so are the eigenvectors of the small matrix).
            let kept = KEPT_AT_COLLAPSE.min(k - 1);
            let combine = |vectors_of: &[Col<f64>], column: usize| {
                let mut sum = Col::<f64>::zeros(n);
                for (i, v) in vectors_of.iter().enumerate() {
                    sum += v * vectors[(i, column)];
                }
                sum
            };
            let new_basis: Vec<Col<f64>> = (0..kept).map(|c| combine(&basis, c)).collect();
            let new_images: Vec<Col<f64>> = (0..kept).map(|c| combine(&images, c)).collect();
            basis = new_basis;
            images = new_images;
        }
        let mut added = orthogonalized(correction, &basis);
        let mut new_length = col_norm(&added);
        if new_length < NOTHING_NEW * norm.max(1.0) {
            // The correction lies in the span: try the residual itself.
            added = orthogonalized(residual, &basis);
            new_length = col_norm(&added);
        }
        if !(new_length.is_finite()) || new_length < NOTHING_NEW * norm.max(f64::MIN_POSITIVE) {
            // Nothing can be added: what was found is what there is.
            return Ok(pair);
        }
        basis.push(added * (1.0 / new_length));
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    /// A symmetric matrix with a spread diagonal and couplings between all elements.
    fn matrix(n: usize) -> Mat<f64> {
        Mat::from_fn(n, n, |i, j| {
            let (a, b) = (i.min(j) as f64, i.max(j) as f64);
            if i == j {
                0.5 + 0.37 * a - 0.9 * (a * 0.7).sin()
            } else {
                0.05 * ((a + 1.0) * (b + 2.0)).sin() / (1.0 + (b - a))
            }
        })
    }

    #[test]
    fn the_lowest_pair_of_a_matrix_is_found_from_its_action_alone() {
        let n = 120;
        let a = matrix(n);
        let (values, vectors) = symmetric_eigen(&a).unwrap();
        let diagonal = Col::from_fn(n, |i| a[(i, i)]);
        let start = Col::from_fn(n, |i| 1.0 / (1.0 + i as f64));
        let mut count = 0;
        let mut apply = |v: &Col<f64>| {
            count += 1;
            Ok(&a * v)
        };
        for (preconditioner, most) in [(Some(&diagonal), 60), (None, 200)] {
            let pair = lowest_eigenpair(
                &mut apply,
                preconditioner,
                &start,
                &DavidsonSettings {
                    tolerance: 1e-9,
                    max_iterations: 400,
                    ..DavidsonSettings::default()
                },
            )
            .unwrap();
            assert!(pair.converged, "residual {}", pair.residual);
            assert!((pair.value - values[0]).abs() < 1e-10, "{}", pair.value);
            let overlap: f64 = (0..n).map(|i| pair.vector[i] * vectors[(i, 0)]).sum();
            assert!((overlap.abs() - 1.0).abs() < 1e-8);
            // Far fewer applications than the dimension with the diagonal.
            assert!(pair.applications <= most, "{}", pair.applications);
            println!("applications {}", pair.applications);
        }
        assert!(count > 0);
    }

    #[test]
    fn an_approximate_inverse_as_preconditioner_finds_the_lowest_pair_in_few_applications() {
        // Stiff and soft modes, as in an environment of bonds and weak contacts.
        let n = 80;
        let (_, q) = symmetric_eigen(&matrix(n)).unwrap();
        let spectrum = |k: usize| {
            if k.is_multiple_of(4) {
                1e-3 * (1.0 + k as f64)
            } else {
                0.5 + 0.01 * k as f64
            }
        };
        let a = Mat::from_fn(n, n, |i, j| {
            (0..n)
                .map(|k| q[(i, k)] * spectrum(k) * q[(j, k)])
                .sum::<f64>()
        });
        // An inverse that is off by up to 30 % in every mode.
        let near = Mat::from_fn(n, n, |i, j| {
            (0..n)
                .map(|k| q[(i, k)] * q[(j, k)] / (spectrum(k) * (1.0 + 0.3 * (k as f64).sin())))
                .sum::<f64>()
        });
        let start = Col::from_fn(n, |i| 1.0 / (1.0 + i as f64));
        let settings = DavidsonSettings {
            tolerance: 1e-9,
            max_iterations: 60,
            ..DavidsonSettings::default()
        };
        let mut apply = |v: &Col<f64>| Ok(&a * v);
        let mut correct = |r: &Col<f64>, _theta: f64| &near * r;
        let guided =
            lowest_eigenpair_preconditioned(&mut apply, &mut correct, &start, &settings).unwrap();
        assert!(guided.converged, "residual {}", guided.residual);
        assert!(
            (guided.value - spectrum(0)).abs() < 1e-12,
            "{}",
            guided.value
        );
        // The plain search is far from it after as many applications.
        let plain = lowest_eigenpair(
            &mut apply,
            None,
            &start,
            &DavidsonSettings {
                max_iterations: guided.applications,
                ..settings
            },
        )
        .unwrap();
        assert!(
            plain.residual > 100.0 * guided.residual,
            "{} {}",
            plain.residual,
            guided.residual
        );
        // A correction of another length is refused.
        let mut wrong = |_: &Col<f64>, _: f64| Col::<f64>::zeros(3);
        assert!(
            lowest_eigenpair_preconditioned(&mut apply, &mut wrong, &start, &settings).is_err()
        );
    }

    #[test]
    fn a_small_span_is_collapsed_and_bad_input_is_refused() {
        let n = 40;
        let a = matrix(n);
        let (values, _) = symmetric_eigen(&a).unwrap();
        let start = Col::from_fn(n, |_| 1.0);
        let mut apply = |v: &Col<f64>| Ok(&a * v);
        let pair = lowest_eigenpair(
            &mut apply,
            None,
            &start,
            &DavidsonSettings {
                max_subspace: 4,
                max_iterations: 2000,
                tolerance: 1e-8,
            },
        )
        .unwrap();
        assert!(pair.converged && (pair.value - values[0]).abs() < 1e-9);
        // The iteration limit ends the search with what was found.
        let short = lowest_eigenpair(
            &mut apply,
            None,
            &start,
            &DavidsonSettings {
                max_iterations: 3,
                ..DavidsonSettings::default()
            },
        )
        .unwrap();
        assert!(!short.converged && short.applications == 3 && short.value >= values[0]);
        assert!(lowest_eigenpair(
            &mut apply,
            None,
            &Col::zeros(n),
            &DavidsonSettings::default()
        )
        .is_err());
        let wrong = Col::zeros(3);
        assert!(lowest_eigenpair(
            &mut apply,
            Some(&wrong),
            &start,
            &DavidsonSettings::default()
        )
        .is_err());
    }
}
