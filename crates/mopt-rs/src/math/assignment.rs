//! Linear sum assignment (the Hungarian method).
//!
//! Kuhn, Nav. Res. Logist. Q. 1955, 2, 83; Munkres, J. Soc. Ind. Appl. Math. 1957, 5, 32; in
//! the shortest-augmenting-path form with dual potentials (Jonker and Volgenant, Computing
//! 1987, 38, 325), `O(n^2 m)` for `n` rows and `m >= n` columns.
//!
//! MultiOptPy: no counterpart (MultiOptPy calls `scipy.optimize.linear_sum_assignment`).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Mat;

use crate::error::{MoptError, MoptResult};

/// Assign every row to a distinct column so that the sum of `cost[(row, column)]` is minimal.
/// Returns the column of each row; needs at least as many columns as rows and finite costs.
pub fn linear_sum_assignment(cost: &Mat<f64>) -> MoptResult<Vec<usize>> {
    let (n, m) = (cost.nrows(), cost.ncols());
    if n > m {
        return Err(MoptError::config(format!(
            "assignment needs at least as many columns as rows, got {n} x {m}"
        )));
    }
    if (0..n).any(|i| (0..m).any(|j| !cost[(i, j)].is_finite())) {
        return Err(MoptError::linalg("assignment costs must be finite"));
    }
    // Row potentials u, column potentials v; column 0 is a virtual start column.
    let mut u = vec![0.0; n + 1];
    let mut v = vec![0.0; m + 1];
    let mut row_of = vec![0_usize; m + 1];
    let mut way = vec![0_usize; m + 1];
    for i in 1..=n {
        row_of[0] = i;
        let mut j0 = 0;
        let mut min_to = vec![f64::INFINITY; m + 1];
        let mut used = vec![false; m + 1];
        loop {
            used[j0] = true;
            let i0 = row_of[j0];
            let mut delta = f64::INFINITY;
            let mut j1 = 0;
            for j in 1..=m {
                if !used[j] {
                    let reduced = cost[(i0 - 1, j - 1)] - u[i0] - v[j];
                    if reduced < min_to[j] {
                        min_to[j] = reduced;
                        way[j] = j0;
                    }
                    if min_to[j] < delta {
                        delta = min_to[j];
                        j1 = j;
                    }
                }
            }
            for j in 0..=m {
                if used[j] {
                    u[row_of[j]] += delta;
                    v[j] -= delta;
                } else {
                    min_to[j] -= delta;
                }
            }
            j0 = j1;
            if row_of[j0] == 0 {
                break;
            }
        }
        // Augment along the path back to the virtual column.
        loop {
            let j1 = way[j0];
            row_of[j0] = row_of[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }
    let mut columns = vec![0; n];
    for j in 1..=m {
        if row_of[j] != 0 {
            columns[row_of[j] - 1] = j - 1;
        }
    }
    Ok(columns)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn total(cost: &Mat<f64>, cols: &[usize]) -> f64 {
        cols.iter().enumerate().map(|(i, &j)| cost[(i, j)]).sum()
    }

    /// Minimum over all injective assignments by enumeration.
    fn brute(cost: &Mat<f64>, row: usize, used: &mut Vec<bool>) -> f64 {
        if row == cost.nrows() {
            return 0.0;
        }
        let mut best = f64::INFINITY;
        for j in 0..cost.ncols() {
            if !used[j] {
                used[j] = true;
                best = best.min(cost[(row, j)] + brute(cost, row + 1, used));
                used[j] = false;
            }
        }
        best
    }

    #[test]
    fn optimal_for_square_and_rectangular_costs() {
        let mut rng = crate::core_infra::rng::Rng::from_seed(7);
        for (n, m) in [(1, 1), (3, 3), (4, 6), (5, 5), (2, 7)] {
            let cost = Mat::from_fn(n, m, |_, _| rng.uniform(-1.0, 1.0));
            let cols = linear_sum_assignment(&cost).unwrap();
            let mut seen = vec![false; m];
            for &j in &cols {
                assert!(!seen[j], "column used twice");
                seen[j] = true;
            }
            let best = brute(&cost, 0, &mut vec![false; m]);
            assert!((total(&cost, &cols) - best).abs() < 1e-12, "{n}x{m}");
        }
        // The greedy choice (row 0 takes its best column) is not optimal here.
        let cost = Mat::from_fn(2, 2, |i, j| [[1.0, 2.0], [1.1, 10.0]][i][j]);
        assert_eq!(linear_sum_assignment(&cost).unwrap(), vec![1, 0]);
        assert!(linear_sum_assignment(&Mat::zeros(3, 2)).is_err());
        let nan = Mat::from_fn(1, 1, |_, _| f64::NAN);
        assert!(linear_sum_assignment(&nan).is_err());
    }
}
