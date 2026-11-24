//! Distances between all pairs of atoms and the functions of them that the interpolations
//! minimize.
//!
//! - [`idpp_value`]: the image dependent pair potential of Smidstrup, Pedersen, Stokbro, and
//!   Jonsson (J. Chem. Phys. 2014, 140, 214106), `S = sum_{i<j} w(d_ij) (d_ij - t_ij)^2`
//!   with `w(d) = d^-4` and target distances `t` interpolated between the first and the
//!   last structure.
//! - [`lst_value`]: the function of the linear synchronous transit of Halgren and Lipscomb
//!   (Chem. Phys. Lett. 1977, 49, 225) as later papers quote it (the original was not at
//!   hand): the same sum with the weight `t_ij^-4` of the interpolated distance, plus
//!   `1e-6` times the squared distance from the linearly interpolated structure, which
//!   fixes the position and the orientation.
//!
//! In a periodic cell the distances are those of the nearest periodic images.
//!
//! MultiOptPy: `Potential/idpp.py:IDPP.calc_obj_func`, `calc_obj_func_1st_deriv`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Col;

use crate::domain::PeriodicCell;

/// Distances below this (Bohr) are left out of the sums: two atoms at one point have no
/// direction to part along.
const NO_DISTANCE: f64 = 1e-8;

/// The vector from atom `j` to atom `i` of `x` (three coordinates per atom).
fn between(x: &Col<f64>, i: usize, j: usize, cell: Option<&PeriodicCell>) -> [f64; 3] {
    let at = |a: usize| [x[3 * a], x[3 * a + 1], x[3 * a + 2]];
    match cell {
        Some(cell) => cell.mic_delta(at(i), at(j)),
        None => {
            let (p, q) = (at(i), at(j));
            [p[0] - q[0], p[1] - q[1], p[2] - q[2]]
        }
    }
}

fn length(v: &[f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

/// The distances of all pairs `i < j` of the atoms of `x`, row by row.
pub fn pair_distances(x: &Col<f64>, cell: Option<&PeriodicCell>) -> Vec<f64> {
    let atoms = x.nrows() / 3;
    let mut distances = Vec::with_capacity(atoms * atoms.saturating_sub(1) / 2);
    for i in 0..atoms {
        for j in i + 1..atoms {
            distances.push(length(&between(x, i, j, cell)));
        }
    }
    distances
}

/// The distances `first + share (last - first)`.
pub fn interpolated_distances(first: &[f64], last: &[f64], share: f64) -> Vec<f64> {
    first
        .iter()
        .zip(last)
        .map(|(a, b)| a + share * (b - a))
        .collect()
}

/// A sum over the pairs of `x` and its gradient: `term(d, t)` gives the value of a pair at
/// distance `d` with target `t` and its derivative with respect to `d`.
fn over_pairs(
    x: &Col<f64>,
    targets: &[f64],
    cell: Option<&PeriodicCell>,
    term: impl Fn(f64, f64) -> (f64, f64),
) -> (f64, Col<f64>) {
    let atoms = x.nrows() / 3;
    let mut value = 0.0;
    let mut gradient = Col::<f64>::zeros(x.nrows());
    let mut pair = 0;
    for i in 0..atoms {
        for j in i + 1..atoms {
            let v = between(x, i, j, cell);
            let d = length(&v);
            let target = targets.get(pair).copied().unwrap_or(d);
            pair += 1;
            if d < NO_DISTANCE {
                continue;
            }
            let (of_pair, slope) = term(d, target);
            value += of_pair;
            for k in 0..3 {
                let along = slope * v[k] / d;
                gradient[3 * i + k] += along;
                gradient[3 * j + k] -= along;
            }
        }
    }
    (value, gradient)
}

/// The image dependent pair potential of the structure `x` for the target distances
/// `targets` (in the order of [`pair_distances`]) and its gradient.
pub fn idpp_value(x: &Col<f64>, targets: &[f64], cell: Option<&PeriodicCell>) -> (f64, Col<f64>) {
    over_pairs(x, targets, cell, |d, t| {
        let weight = d.powi(-4);
        let off = d - t;
        (
            weight * off * off,
            2.0 * weight * off - 4.0 * weight / d * off * off,
        )
    })
}

/// The function of the linear synchronous transit for the structure `x`: target distances
/// `targets`, the linearly interpolated structure `line`, and `cartesian_weight` of the
/// distance from it. Returns the value and its gradient.
pub fn lst_value(
    x: &Col<f64>,
    targets: &[f64],
    line: &Col<f64>,
    cartesian_weight: f64,
    cell: Option<&PeriodicCell>,
) -> (f64, Col<f64>) {
    let (mut value, mut gradient) = over_pairs(x, targets, cell, |d, t| {
        if t < NO_DISTANCE {
            return (0.0, 0.0);
        }
        let weight = t.powi(-4);
        let off = d - t;
        (weight * off * off, 2.0 * weight * off)
    });
    for k in 0..x.nrows() {
        let off = x[k] - line[k];
        value += cartesian_weight * off * off;
        gradient[k] += 2.0 * cartesian_weight * off;
    }
    (value, gradient)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::domain::{Bohr, Vector3};

    fn col(values: &[f64]) -> Col<f64> {
        Col::from_fn(values.len(), |k| values[k])
    }

    fn numeric(f: impl Fn(&Col<f64>) -> f64, x: &Col<f64>) -> Col<f64> {
        Col::from_fn(x.nrows(), |k| {
            let (mut up, mut down) = (x.clone(), x.clone());
            up[k] += 1e-5;
            down[k] -= 1e-5;
            (f(&up) - f(&down)) / 2e-5
        })
    }

    #[test]
    fn the_gradients_are_those_of_the_values() {
        let x = col(&[0.1, 0.0, 0.2, 2.1, 0.3, -0.1, 1.0, 1.9, 0.4, -0.7, 1.2, 2.2]);
        let targets = [2.0, 2.3, 2.6, 2.1, 3.0, 2.4];
        let line = col(&[0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 1.0, 2.0, 0.0, -0.5, 1.0, 2.0]);
        let cell = PeriodicCell::from_lattice_vectors(
            Vector3::<Bohr>::new([5.0, 0.0, 0.0]),
            Vector3::<Bohr>::new([0.5, 5.5, 0.0]),
            Vector3::<Bohr>::new([0.0, 0.3, 6.0]),
            [true, true, true],
        )
        .unwrap();
        for cell in [None, Some(&cell)] {
            let (value, gradient) = idpp_value(&x, &targets, cell);
            assert!(value > 0.0);
            let expected = numeric(|p| idpp_value(p, &targets, cell).0, &x);
            let (_, lst) = lst_value(&x, &targets, &line, 1e-3, cell);
            let lst_expected = numeric(|p| lst_value(p, &targets, &line, 1e-3, cell).0, &x);
            for k in 0..x.nrows() {
                assert!((gradient[k] - expected[k]).abs() < 1e-8, "{k}");
                assert!((lst[k] - lst_expected[k]).abs() < 1e-8, "{k}");
            }
        }
        // Zero where every distance has its target; the same after a shift by a lattice
        // vector in the cell.
        let own = pair_distances(&x, None);
        assert_eq!(own.len(), 6);
        assert!(idpp_value(&x, &own, None).0.abs() < 1e-30);
        let mut shifted = x.clone();
        shifted[0] += 5.0;
        let in_cell = pair_distances(&x, Some(&cell));
        let after = pair_distances(&shifted, Some(&cell));
        for (a, b) in in_cell.iter().zip(&after) {
            assert!((a - b).abs() < 1e-12);
        }
        let half = interpolated_distances(&[1.0, 2.0], &[3.0, 2.0], 0.25);
        assert!((half[0] - 1.5).abs() < 1e-15 && (half[1] - 2.0).abs() < 1e-15);
    }
}
