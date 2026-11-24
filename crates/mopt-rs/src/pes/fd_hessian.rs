//! Central-difference Hessians of an analytic gradient, with the displaced structures
//! evaluated in parallel.
//!
//! `H[:, k] = (g(x + h e_k) - g(x - h e_k)) / (2 h)` for every coordinate `k`, symmetrized
//! `(H + H^T) / 2`. Every column is computed on its own (two gradients) and written to its own
//! place, so the result is the same bits for any number of threads and any order in which the
//! columns finish; with `parallel` the columns run in the ambient rayon pool (inside the pool
//! of a mapper job or of a chain of images, never in a pool of their own). A caller passes
//! `parallel = false` when a gradient depends on what was evaluated before it (a calculator
//! whose [`crate::pes::Calculator::evaluates_in_order`] is true, or one that keeps a start of
//! its iterations from the last structure): the columns then run one after the other, in the
//! order of the coordinates; [`serial_central_difference_hessian`] is that order for a gradient
//! that keeps state between its calls (`FnMut`). These are the central-difference Hessians of
//! the crate (the engines, the cell of `-vc`, the reaction coordinate of confsearch; also at the
//! former path `pes::native::fd_hessian`).
//!
//! Engines use central differences only where they offer them as an alternative to their
//! analytic Hessians (`-calc_opt hessian=numerical`) or have no analytic second derivatives
//! (the D4 dispersion, excited states).
//!
//! MultiOptPy: no counterpart (new in mopt-rs; MultiOptPy's numerical Hessians are serial loops
//! in `Calculator/*_calculation_tools.py`).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::{Col, Mat};
use rayon::prelude::*;

use crate::domain::MolecularSystem;
use crate::error::{MoptError, MoptResult};

/// The central-difference Hessian of `gradient` at `x0` with the displacement `step`
/// (see the module documentation).
pub fn central_difference_hessian<G>(
    x0: &Col<f64>,
    step: f64,
    parallel: bool,
    gradient: G,
) -> MoptResult<Mat<f64>>
where
    G: Fn(&Col<f64>) -> MoptResult<Col<f64>> + Sync,
{
    check_step(step)?;
    let columns: Vec<Vec<f64>> = if parallel {
        (0..x0.nrows())
            .into_par_iter()
            .map(|k| column(x0, step, k, &gradient))
            .collect::<MoptResult<_>>()?
    } else {
        (0..x0.nrows())
            .map(|k| column(x0, step, k, &gradient))
            .collect::<MoptResult<_>>()?
    };
    Ok(symmetric(&columns))
}

/// [`central_difference_hessian`] with the columns one after the other, in the order of the
/// coordinates, for a gradient that keeps state between its calls (`FnMut`: a calculator that
/// starts from what it evaluated last). The same bits as the parallel form.
pub fn serial_central_difference_hessian<G>(
    x0: &Col<f64>,
    step: f64,
    mut gradient: G,
) -> MoptResult<Mat<f64>>
where
    G: FnMut(&Col<f64>) -> MoptResult<Col<f64>>,
{
    check_step(step)?;
    let columns: Vec<Vec<f64>> = (0..x0.nrows())
        .map(|k| column(x0, step, k, &mut gradient))
        .collect::<MoptResult<_>>()?;
    Ok(symmetric(&columns))
}

fn check_step(step: f64) -> MoptResult<()> {
    if step > 0.0 && step.is_finite() {
        Ok(())
    } else {
        Err(MoptError::config(format!(
            "central-difference Hessian: the displacement must be positive, got {step}"
        )))
    }
}

/// Column `k`: `(g(x0 + h e_k) - g(x0 - h e_k)) / (2 h)`, the plus side evaluated first.
fn column<G>(x0: &Col<f64>, step: f64, k: usize, mut gradient: G) -> MoptResult<Vec<f64>>
where
    G: FnMut(&Col<f64>) -> MoptResult<Col<f64>>,
{
    let dim = x0.nrows();
    let mut xp = x0.clone();
    xp[k] += step;
    let mut xm = x0.clone();
    xm[k] -= step;
    let gp = gradient(&xp)?;
    let gm = gradient(&xm)?;
    if gp.nrows() != dim || gm.nrows() != dim {
        return Err(MoptError::calc(format!(
            "central-difference Hessian: a gradient of {} or {} components for {dim} \
             coordinates",
            gp.nrows(),
            gm.nrows()
        )));
    }
    Ok((0..dim).map(|i| (gp[i] - gm[i]) / (2.0 * step)).collect())
}

/// `(H + H^T) / 2` of the columns `columns`.
fn symmetric(columns: &[Vec<f64>]) -> Mat<f64> {
    let dim = columns.len();
    Mat::from_fn(dim, dim, |i, j| 0.5 * (columns[j][i] + columns[i][j]))
}

/// `f(0), ..., f(count - 1)` in the ambient rayon pool, in order, with at most as many running
/// at once as fit in half the free memory ([`crate::core_infra::memory::budget_bytes`]) when
/// each holds `bytes_per_task` bytes of intermediates: the items are taken in batches of that
/// many (at least one), each batch in parallel. Every item is computed on its own, so the
/// result does not depend on the batches or the number of threads.
pub fn parallel_within_memory<T, F>(count: usize, bytes_per_task: f64, f: F) -> Vec<T>
where
    T: Send,
    F: Fn(usize) -> T + Sync,
{
    let (budget, _) = crate::core_infra::memory::budget_bytes(None);
    let batch = if bytes_per_task > 0.0 {
        ((budget / bytes_per_task).floor() as usize).clamp(1, count.max(1))
    } else {
        count.max(1)
    };
    let mut out = Vec::with_capacity(count);
    let mut start = 0;
    while start < count {
        let end = (start + batch).min(count);
        let part: Vec<T> = (start..end).into_par_iter().map(&f).collect();
        out.extend(part);
        start = end;
    }
    out
}

/// [`central_difference_hessian`] of a gradient of the structure: `gradient` gets `sys` with
/// its flat Cartesian coordinates (Bohr) displaced.
pub fn structure_hessian<G>(
    sys: &MolecularSystem,
    step: f64,
    parallel: bool,
    gradient: G,
) -> MoptResult<Mat<f64>>
where
    G: Fn(&MolecularSystem) -> MoptResult<Col<f64>> + Sync,
{
    central_difference_hessian(&sys.coords_flat_bohr(), step, parallel, |x| {
        gradient(&sys.with_flat_coords_bohr(x)?)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `E = sum_i a_i x_i^2 / 2 + b x_0 x_1^2`: the gradient and its exact Hessian.
    fn gradient(x: &Col<f64>) -> MoptResult<Col<f64>> {
        let b = 0.3;
        Ok(Col::from_fn(x.nrows(), |i| {
            let a = 1.0 + i as f64;
            a * x[i]
                + match i {
                    0 => b * x[1] * x[1],
                    1 => 2.0 * b * x[0] * x[1],
                    _ => 0.0,
                }
        }))
    }

    #[test]
    fn columns_in_parallel_give_the_serial_bits() -> MoptResult<()> {
        let x = Col::from_fn(7, |i| 0.1 * i as f64 - 0.2);
        let serial = central_difference_hessian(&x, 1e-3, false, gradient)?;
        for threads in [1, 2, 4, 8] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .map_err(|e| MoptError::config(e.to_string()))?;
            let h = pool.install(|| central_difference_hessian(&x, 1e-3, true, gradient))?;
            for i in 0..7 {
                for j in 0..7 {
                    assert_eq!(h[(i, j)].to_bits(), serial[(i, j)].to_bits());
                }
            }
        }
        // The exact Hessian of the quadratic-and-cubic test function.
        let b = 0.3;
        assert!((serial[(0, 1)] - 2.0 * b * x[1]).abs() < 1e-9);
        assert!((serial[(1, 1)] - (2.0 + 2.0 * b * x[0])).abs() < 1e-9);
        assert!((serial[(3, 3)] - 4.0).abs() < 1e-9);
        Ok(())
    }

    #[test]
    fn the_serial_form_of_a_gradient_with_state_gives_the_same_bits() -> MoptResult<()> {
        let x = Col::from_fn(6, |i| 0.3 - 0.15 * i as f64);
        let parallel = central_difference_hessian(&x, 2e-3, true, gradient)?;
        let mut order = Vec::new();
        let serial = serial_central_difference_hessian(&x, 2e-3, |y| {
            order.push(y.clone());
            gradient(y)
        })?;
        for i in 0..6 {
            for j in 0..6 {
                assert_eq!(parallel[(i, j)].to_bits(), serial[(i, j)].to_bits());
            }
        }
        // Coordinate by coordinate, the plus side first.
        assert_eq!(order.len(), 12);
        for (k, pair) in order.chunks(2).enumerate() {
            assert!(pair[0][k] > x[k] && pair[1][k] < x[k]);
        }
        assert!(serial_central_difference_hessian(&x, -1.0, gradient).is_err());
        Ok(())
    }

    #[test]
    fn a_displacement_that_is_not_positive_is_refused() {
        let x = Col::zeros(2);
        assert!(central_difference_hessian(&x, 0.0, true, gradient).is_err());
        assert!(central_difference_hessian(&x, f64::NAN, true, gradient).is_err());
    }
}
