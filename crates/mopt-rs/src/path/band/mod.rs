//! Chains of images between two basins: the forces that relax a chain towards a reaction
//! path, the steps taken along them, and what keeps the images in order.
//!
//! What MultiOptPy calls NEB is not the nudged elastic band of the literature. Its default
//! has no springs: every image moves down the part of its gradient that is left after the
//! component along the chain is removed, and the images are spread along the chain again
//! by redistribution every few iterations, as in a string method. The parts here are what
//! that code does, written as formulas; where the code is inconsistent (a sign, a unit),
//! the consistent form is the default and the form of MultiOptPy is kept under
//! [`Compatibility::MultiOptPy`], which the comparisons with its reference data use.
//!
//! A chain is a list of points of the same length `n`: the `3N` Cartesian coordinates of a
//! structure in Bohr ([`ChainKind::Atoms`]: some operations work atom by atom, and the
//! position and orientation of a structure do not count), or any `n` variables
//! ([`ChainKind::Variables`], for example the two variables of a model surface).
//!
//! MultiOptPy: `neb.py`, `MEP/pathopt_bneb_force.py`, `Optimizer/fire_neb.py`,
//! `Optimizer/trust_radius_neb.py`, `Optimizer/gradientdescent_neb.py`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

pub mod align;
pub mod diis;
pub mod fire;
pub mod force;
pub mod global;
pub mod images;
pub mod lup;
pub mod machine;
pub mod peaks;
pub mod redistribute;
pub mod rfo;
pub mod ridge;
pub mod ridge_fixed;
pub mod springs;
pub mod surrogate;
pub mod tangents;
pub mod trust;

use faer::Col;

use crate::error::{MoptError, MoptResult};

/// Number of coordinates of an atom.
pub const BLOCK_ATOM: usize = 3;

/// Which form of a formula is used where MultiOptPy's is inconsistent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Compatibility {
    /// The consistent form (see the documentation of each part for what differs).
    #[default]
    Consistent,
    /// What MultiOptPy 1.21.2 computes, for comparisons with its results.
    MultiOptPy,
}

/// What the points of a chain are.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ChainKind {
    /// Cartesian coordinates of atoms, three per atom.
    #[default]
    Atoms,
    /// Plain variables.
    Variables,
}

impl ChainKind {
    /// Size of the blocks of points of `n` components: an atom, or the whole point (the
    /// blocks of MultiOptPy's projection of the gradient, and what the number of
    /// components must be a multiple of).
    pub fn block(self, n: usize) -> usize {
        match self {
            Self::Atoms => BLOCK_ATOM,
            Self::Variables => n,
        }
    }
}

/// The images of a chain must have the same number of components, a multiple of `block`.
pub fn check_chain(chain: &[Col<f64>], block: usize) -> MoptResult<usize> {
    let n = chain.first().map_or(0, Col::nrows);
    if chain.len() < 2 {
        return Err(MoptError::config("a chain needs at least two images"));
    }
    if block == 0 || n == 0 || !n.is_multiple_of(block) {
        return Err(MoptError::config(format!(
            "images of {n} components cannot be split into blocks of {block}"
        )));
    }
    if chain.iter().any(|image| image.nrows() != n) {
        return Err(MoptError::config(
            "the images of a chain differ in their number of components",
        ));
    }
    Ok(n)
}

/// Centroid of the blocks of `point` (the centre of the atoms).
pub(crate) fn centroid(point: &Col<f64>, block: usize) -> Vec<f64> {
    let count = (point.nrows() / block).max(1) as f64;
    let mut c = vec![0.0; block];
    for (k, value) in point.iter().enumerate() {
        c[k % block] += value / count;
    }
    c
}

/// Distance between two points after each is moved to its own centroid.
fn centred_distance(a: &Col<f64>, b: &Col<f64>, block: usize) -> f64 {
    let (ca, cb) = (centroid(a, block), centroid(b, block));
    a.iter()
        .zip(b.iter())
        .enumerate()
        .map(|(k, (x, y))| ((x - ca[k % block]) - (y - cb[k % block])).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Length of the chain up to each image: the distances between neighbours summed from
/// the first image. For atoms each image is taken at its own centroid (a shift of a whole
/// structure adds no length).
pub fn path_lengths(chain: &[Col<f64>], kind: ChainKind) -> Vec<f64> {
    let mut lengths = Vec::with_capacity(chain.len());
    let mut total = 0.0;
    lengths.push(total);
    for pair in chain.windows(2) {
        total += match kind {
            ChainKind::Atoms => centred_distance(&pair[0], &pair[1], BLOCK_ATOM),
            ChainKind::Variables => {
                let d = &pair[1] - &pair[0];
                crate::math::linalg::col_norm(&d)
            }
        };
        lengths.push(total);
    }
    lengths
}

/// Inner images whose value is above both neighbours.
pub fn strict_maxima(values: &[f64]) -> Vec<usize> {
    (1..values.len().saturating_sub(1))
        .filter(|&i| values[i] > values[i - 1] && values[i] > values[i + 1])
        .collect()
}

/// Inner images whose value is below both neighbours.
pub fn strict_minima(values: &[f64]) -> Vec<usize> {
    (1..values.len().saturating_sub(1))
        .filter(|&i| values[i] < values[i - 1] && values[i] < values[i + 1])
        .collect()
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn lengths_ignore_a_shift_of_an_image_and_extrema_are_strict() {
        let point = |shift: f64, stretch: f64| {
            Col::from_fn(6, |k| shift + if k == 3 { stretch } else { 0.0 })
        };
        // Two atoms along x; the second image is the first one stretched and moved away.
        let chain = vec![point(0.0, 1.0), point(5.0, 2.0), point(-3.0, 4.0)];
        let lengths = path_lengths(&chain, ChainKind::Atoms);
        // Stretching by d moves each atom by d / 2 against the centroid.
        let step = |d: f64| (2.0 * (0.5 * d).powi(2)).sqrt();
        assert!((lengths[1] - step(1.0)).abs() < 1e-14);
        assert!((lengths[2] - step(1.0) - step(2.0)).abs() < 1e-14);
        // Plain variables: the distance as it is.
        let plain = path_lengths(&chain, ChainKind::Variables);
        let expected = (6.0 * 25.0 + 2.0 * 5.0 + 1.0_f64).sqrt();
        assert!((plain[1] - expected).abs() < 1e-13);
        assert_eq!(ChainKind::Atoms.block(6), 3);
        assert_eq!(ChainKind::Variables.block(2), 2);
        assert_eq!(strict_maxima(&[0.0, 2.0, 2.0, 1.0, 3.0, 0.5]), vec![4]);
        assert_eq!(strict_minima(&[0.0, 2.0, 2.0, 1.0, 3.0, 0.5]), vec![3]);
        assert!(check_chain(&chain[..1], 3).is_err());
        assert!(check_chain(&chain, 4).is_err());
        assert_eq!(check_chain(&chain, 3).unwrap(), 6);
    }
}
