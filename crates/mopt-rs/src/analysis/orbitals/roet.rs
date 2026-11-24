//! Orbital energies along a reaction path: which orbitals change most, and how fast.
//!
//! The reactive orbital energy theory of Tsuneda and coworkers (Tsuneda, Singh, J. Comput.
//! Chem. 2014, doi:10.1002/jcc.23599; Tsuneda, Singh, Chattaraj, Phys. Chem. Chem. Phys.
//! 2018, 20, 14211) reads a reaction from the energies of the orbitals along its path: the reactive
//! orbitals are the occupied and the empty orbital whose energies vary most. As Tsuneda
//! and Taketsugu state it (arXiv:2412.11361), the occupied reactive orbital is the most
//! stabilized occupied orbital, found as the one with the largest relative change
//! `2 (e_prod - e_reac) / |e_prod + e_reac|` between reactant and product, the orbitals
//! being followed along the path so that the same orbital is compared. That source gives
//! no separate formula for the empty one; here it is the empty orbital with the largest
//! relative change in magnitude.
//!
//! The functions here work on energies of orbitals that have already been followed from
//! point to point ([`super::tracking`]).
//!
//! MultiOptPy: `WFAnalyzer/MO_analysis.py:NROAnalysis.run` (first and second derivatives of
//! the orbital energies by differences of orbitals in the order they come).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use crate::error::{MoptError, MoptResult};

/// `2 (last - first) / |last + first|`: the change of an orbital energy relative to its
/// mean size (0 when both vanish).
pub fn relative_change(first: f64, last: f64) -> f64 {
    let sum = (last + first).abs();
    if sum > 0.0 {
        2.0 * (last - first) / sum
    } else {
        0.0
    }
}

/// The orbitals that change most between the two ends of a path.
#[derive(Debug, Clone, PartialEq)]
pub struct ReactiveOrbitals {
    /// Relative change of every orbital ([`relative_change`]).
    pub changes: Vec<f64>,
    /// The occupied orbital that is stabilized most (the most negative change); `None` if
    /// no occupied orbital goes down.
    pub occupied: Option<usize>,
    /// The empty orbital with the largest change in magnitude; `None` if none changes.
    pub unoccupied: Option<usize>,
}

/// Relative changes smaller than this are no change (the two ends are the same structure
/// or mirror images of each other).
pub const NO_CHANGE: f64 = 1e-8;

/// The reactive orbitals from the energies of the followed orbitals at the first and the
/// last point of a path; orbitals filled to more than half of the highest occupation count
/// as occupied.
pub fn reactive_orbitals(
    first: &[f64],
    last: &[f64],
    occupations: &[f64],
) -> MoptResult<ReactiveOrbitals> {
    let n = first.len();
    if last.len() != n || occupations.len() != n {
        return Err(MoptError::config(
            "energies and occupations of the two ends differ in number",
        ));
    }
    let changes: Vec<f64> = first
        .iter()
        .zip(last)
        .map(|(a, b)| relative_change(*a, *b))
        .collect();
    let highest = occupations.iter().copied().fold(0.0, f64::max);
    let filled = |k: usize| highest > 0.0 && occupations[k] > 0.5 * highest;
    let occupied = (0..n)
        .filter(|&k| filled(k) && changes[k] < -NO_CHANGE)
        .min_by(|&a, &b| changes[a].total_cmp(&changes[b]));
    let unoccupied = (0..n)
        .filter(|&k| !filled(k) && changes[k].abs() > NO_CHANGE)
        .max_by(|&a, &b| changes[a].abs().total_cmp(&changes[b].abs()));
    Ok(ReactiveOrbitals {
        changes,
        occupied,
        unoccupied,
    })
}

/// First and second derivatives of `values` given at the increasing positions `s`
/// (three-point formulas for unequal spacing; at the ends the same parabola through the
/// outermost three points). With two points the first derivative is their slope and the
/// second is zero.
pub fn derivatives(s: &[f64], values: &[f64]) -> MoptResult<(Vec<f64>, Vec<f64>)> {
    let n = s.len();
    if values.len() != n || n < 2 {
        return Err(MoptError::config(
            "derivatives along a path need two points at least, with a value each",
        ));
    }
    if s.iter().any(|v| !v.is_finite()) || s.windows(2).any(|w| w[1] <= w[0]) {
        return Err(MoptError::config(
            "the positions along the path must increase",
        ));
    }
    if n == 2 {
        let slope = (values[1] - values[0]) / (s[1] - s[0]);
        return Ok((vec![slope; 2], vec![0.0; 2]));
    }
    let mut first = vec![0.0; n];
    let mut second = vec![0.0; n];
    for k in 0..n {
        // The parabola through the three points around k (shifted inwards at the ends).
        let c = k.clamp(1, n - 2);
        let (h1, h2) = (s[c] - s[c - 1], s[c + 1] - s[c]);
        let (below, here, above) = (values[c - 1], values[c], values[c + 1]);
        let curvature =
            2.0 * (below / (h1 * (h1 + h2)) - here / (h1 * h2) + above / (h2 * (h1 + h2)));
        let slope_at_c = -h2 * below / (h1 * (h1 + h2))
            + (h2 - h1) * here / (h1 * h2)
            + h1 * above / (h2 * (h1 + h2));
        second[k] = curvature;
        first[k] = slope_at_c + curvature * (s[k] - s[c]);
    }
    Ok((first, second))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_orbitals_that_change_most() {
        // Occupied: -0.5 -> -0.6 (stabilized), -0.4 -> -0.38; empty: 0.1 -> 0.25, 0.3 -> 0.29.
        let first = [-0.5, -0.4, 0.1, 0.3];
        let last = [-0.6, -0.38, 0.25, 0.29];
        let r = reactive_orbitals(&first, &last, &[2.0, 2.0, 0.0, 0.0]).unwrap();
        assert!((r.changes[0] + 2.0 * 0.1 / 1.1).abs() < 1e-15);
        assert_eq!(r.occupied, Some(0));
        assert_eq!(r.unoccupied, Some(2));
        // One spin: occupations up to 1; the highest occupied one here goes up, the other
        // down.
        let r =
            reactive_orbitals(&[-0.5, -0.3, 0.2], &[-0.45, -0.35, 0.2], &[1.0, 1.0, 0.0]).unwrap();
        assert_eq!(r.occupied, Some(1));
        assert_eq!(r.unoccupied, None);
        assert_eq!(relative_change(0.0, 0.0), 0.0);
        assert!(reactive_orbitals(&[0.0], &[0.0, 1.0], &[1.0]).is_err());
    }

    #[test]
    fn derivatives_of_a_parabola_on_unequal_steps() {
        let s = [0.0, 0.3, 0.4, 1.0, 1.7];
        let f = |x: f64| 2.0 - 3.0 * x + 0.5 * x * x;
        let values: Vec<f64> = s.iter().map(|x| f(*x)).collect();
        let (first, second) = derivatives(&s, &values).unwrap();
        for (k, x) in s.iter().enumerate() {
            assert!((first[k] - (-3.0 + x)).abs() < 1e-12, "{k}");
            assert!((second[k] - 1.0).abs() < 1e-12, "{k}");
        }
        let (first, second) = derivatives(&[0.0, 2.0], &[1.0, 5.0]).unwrap();
        assert_eq!((first, second), (vec![2.0, 2.0], vec![0.0, 0.0]));
        assert!(derivatives(&[0.0], &[1.0]).is_err());
        assert!(derivatives(&[0.0, 0.0, 1.0], &[1.0, 2.0, 3.0]).is_err());
    }
}
