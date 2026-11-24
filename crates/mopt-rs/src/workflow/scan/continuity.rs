//! Jumps between branches of a relaxed map: neighbouring grid points whose optimized
//! structures differ much more than neighbours usually do.
//!
//! A relaxed scan follows one valley as long as each point starts from a neighbour, but a
//! valley can end (the constrained minimum disappears) or a point can fall into another one.
//! Two neighbours on different branches then differ by far more than the step of the grid
//! moves the atoms. A pair is a jump when the distance of its structures (mass-weighted after
//! superposition, as the maps of `-cmds`; Euclidean for plain variables) exceeds `factor` times
//! the median over all neighbouring pairs; the higher point of a jump is optimized again from
//! the lower one, and the lower result is kept (the scan keeps the lowest branch it found).
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use super::grid::ScanGrid;

/// A pair of neighbouring points on different branches.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Jump {
    /// One point.
    pub a: usize,
    /// The other.
    pub b: usize,
    /// Distance of their structures.
    pub distance: f64,
    /// Median distance of neighbouring points.
    pub median: f64,
}

/// The jumps of `grid` with the distances `distance(a, b)` of the structures of neighbouring
/// points (`None` where one is missing): pairs above `factor` times the median.
pub fn jumps(
    grid: &ScanGrid,
    distance: impl Fn(usize, usize) -> Option<f64>,
    factor: f64,
) -> Vec<Jump> {
    let mut pairs = Vec::new();
    for p in 0..grid.len() {
        for q in grid.neighbours(p) {
            if q > p {
                if let Some(d) = distance(p, q).filter(|d| d.is_finite()) {
                    pairs.push((p, q, d));
                }
            }
        }
    }
    if pairs.len() < 2 {
        return Vec::new();
    }
    let mut sorted: Vec<f64> = pairs.iter().map(|p| p.2).collect();
    sorted.sort_by(f64::total_cmp);
    let median = sorted[sorted.len() / 2];
    if median <= 0.0 {
        return Vec::new();
    }
    pairs
        .into_iter()
        .filter(|&(_, _, d)| d > factor * median)
        .map(|(a, b, distance)| Jump {
            a,
            b,
            distance,
            median,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;
    use crate::workflow::scan::grid::{GridOrder, ScanAxis};

    #[test]
    fn a_step_much_longer_than_the_others_is_a_jump() {
        let axis = |n| ScanAxis::linspace("q", "", 0.0, 1.0, n).unwrap();
        let grid = ScanGrid::product(vec![axis(4), axis(3)], GridOrder::Snake).unwrap();
        // Structures as numbers: a step of 1 everywhere, a break between columns 1 and 2 of
        // the last row.
        let position = |p: usize| {
            let idx = grid.indices(p);
            idx[0] as f64
                + 10.0 * idx[1] as f64
                + if idx[0] >= 2 && idx[1] == 2 {
                    50.0
                } else {
                    0.0
                }
        };
        let found = jumps(&grid, |a, b| Some((position(a) - position(b)).abs()), 3.0);
        assert!(!found.is_empty());
        assert!(found
            .iter()
            .any(|j| grid.indices(j.a) == vec![1, 2] && grid.indices(j.b) == vec![2, 2]));
        assert!(jumps(&grid, |_, _| None, 3.0).is_empty());
    }
}
