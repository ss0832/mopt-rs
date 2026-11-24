//! The excited states of a structure in a form that can be compared between structures.
//!
//! `-es N` counts the states of a structure by their energies. Along an optimization, a
//! path, or a chain of images two states can change places, and the state that is counted
//! as number `N` is then another one. To stay with one state, the states of two
//! structures are compared by their amplitudes: the single excitations of the second
//! structure are brought into the orbitals of the first by the overlaps of the orbitals,
//! `X' -> O_occ X' O_virt^T`, and the overlap of state `k` of the first with state `l` of
//! the second is the scalar product of the amplitudes of `k` with the carried ones of `l`.
//!
//! The orbitals are given in an orthonormal basis that belongs to the atoms (Loewdin
//! orthogonalized atomic orbitals), so that the scalar product of two orbitals of
//! neighbouring structures needs no overlap between the atomic orbitals of the two
//! structures; the change of the basis functions with the positions is left out, as it is
//! in the differences that give the gradients of the excitation energies of the built-in
//! GFN0-xTB (`pes::native::gfn0::excited`).
//!
//! This covers states of single excitations on one closed-shell reference (Tamm-Dancoff
//! and configuration interaction with singles). [`crate::pes::followed`] uses it to stay
//! with a state.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Mat;

use crate::error::{MoptError, MoptResult};

/// The lowest excited states of a structure: orbitals, excitation energies, amplitudes.
#[derive(Debug, Clone)]
pub struct StateCharacters {
    /// The orbitals in an orthonormal basis that belongs to the atoms (columns; the
    /// occupied ones first).
    pub orbitals: Mat<f64>,
    /// Number of doubly occupied orbitals.
    pub occupied: usize,
    /// Excitation energies (Hartree), ascending.
    pub energies: Vec<f64>,
    /// Amplitudes: column `k` is state `k` over the excitations `(i, a)` at `i * virtual +
    /// a`, of length one.
    pub amplitudes: Mat<f64>,
}

/// The amplitudes `amplitudes` (columns over `(i, a)` at `i * virtuals + a`) of a structure
/// in the orbitals of another: `O_occ X O_virt^T`, where `overlap[(p, q)]` is the scalar
/// product of orbital `p` of the structure they are brought to with orbital `q` of theirs.
pub fn carry_amplitudes(
    overlap: &Mat<f64>,
    occupied: usize,
    virtuals: usize,
    amplitudes: &Mat<f64>,
) -> Mat<f64> {
    let count = amplitudes.ncols();
    let mut out = Mat::<f64>::zeros(occupied * virtuals, count);
    let mut middle = vec![0.0; occupied * virtuals];
    for k in 0..count {
        // O_occ X.
        for i in 0..occupied {
            for b in 0..virtuals {
                middle[i * virtuals + b] = (0..occupied)
                    .map(|j| overlap[(i, j)] * amplitudes[(j * virtuals + b, k)])
                    .sum();
            }
        }
        // (O_occ X) O_virt^T.
        for i in 0..occupied {
            for a in 0..virtuals {
                out[(i * virtuals + a, k)] = (0..virtuals)
                    .map(|b| middle[i * virtuals + b] * overlap[(occupied + a, occupied + b)])
                    .sum();
            }
        }
    }
    out
}

impl StateCharacters {
    /// The states with their sizes checked.
    pub fn new(
        orbitals: Mat<f64>,
        occupied: usize,
        energies: Vec<f64>,
        amplitudes: Mat<f64>,
    ) -> MoptResult<Self> {
        let size = orbitals.ncols();
        if orbitals.nrows() != size || occupied == 0 || occupied >= size {
            return Err(MoptError::config(format!(
                "excited states: {} x {size} orbitals with {occupied} occupied ones",
                orbitals.nrows()
            )));
        }
        let excitations = occupied * (size - occupied);
        if amplitudes.nrows() != excitations || amplitudes.ncols() != energies.len() {
            return Err(MoptError::config(format!(
                "excited states: {} x {} amplitudes for {excitations} excitations and {} states",
                amplitudes.nrows(),
                amplitudes.ncols(),
                energies.len()
            )));
        }
        Ok(Self {
            orbitals,
            occupied,
            energies,
            amplitudes,
        })
    }

    /// Number of empty orbitals.
    pub fn virtuals(&self) -> usize {
        self.orbitals.ncols() - self.occupied
    }

    /// Number of states.
    pub fn count(&self) -> usize {
        self.energies.len()
    }

    /// The lowest `count` states (all of them if there are fewer).
    #[must_use]
    pub fn lowest(mut self, count: usize) -> Self {
        if count < self.energies.len() {
            self.energies.truncate(count);
            self.amplitudes = self.amplitudes.subcols(0, count).to_owned();
        }
        self
    }

    /// The squared overlaps of the states of this structure (rows) with the states of
    /// `other` (columns), and the share of the occupied orbitals of this structure that is
    /// still occupied in `other` (one: all of it).
    pub fn squared_overlaps(&self, other: &Self) -> MoptResult<(Mat<f64>, f64)> {
        let (products, kept) = self.overlaps(other)?;
        Ok((
            Mat::from_fn(products.nrows(), products.ncols(), |k, l| {
                products[(k, l)] * products[(k, l)]
            }),
            kept,
        ))
    }

    /// The overlaps of the states of this structure (rows) with the states of `other`
    /// (columns), with their signs, and the share of the occupied orbitals of this
    /// structure that is still occupied in `other`. The sign of a state is that of its
    /// amplitudes as the calculator handed them out.
    pub fn overlaps(&self, other: &Self) -> MoptResult<(Mat<f64>, f64)> {
        if self.orbitals.nrows() != other.orbitals.nrows()
            || self.orbitals.ncols() != other.orbitals.ncols()
            || self.occupied != other.occupied
        {
            return Err(MoptError::config(
                "excited states: the two structures have other orbitals (another molecule, \
                 charge, or basis)",
            ));
        }
        let overlap = self.orbitals.transpose() * &other.orbitals;
        let occupied = self.occupied;
        let kept = (0..occupied)
            .map(|i| {
                (0..occupied)
                    .map(|j| overlap[(i, j)] * overlap[(i, j)])
                    .sum::<f64>()
            })
            .sum::<f64>()
            / occupied as f64;
        let carried = carry_amplitudes(&overlap, occupied, self.virtuals(), &other.amplitudes);
        Ok((self.amplitudes.transpose() * &carried, kept))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Three orbitals (one occupied), rotated by `angle` between the two empty ones, with
    /// the two states the two excitations in the order `first`, `second`.
    fn states(angle: f64, order: [usize; 2]) -> StateCharacters {
        let (c, s) = (angle.cos(), angle.sin());
        let orbitals = Mat::from_fn(3, 3, |m, p| match (m, p) {
            (0, 0) => 1.0,
            (1, 1) | (2, 2) => c,
            (2, 1) => s,
            (1, 2) => -s,
            _ => 0.0,
        });
        let amplitudes = Mat::from_fn(2, 2, |p, k| if order[k] == p { 1.0 } else { 0.0 });
        StateCharacters::new(orbitals, 1, vec![0.1, 0.2], amplitudes).unwrap()
    }

    #[test]
    fn states_are_found_again_after_they_change_places() {
        let here = states(0.0, [0, 1]);
        // The same structure: every state overlaps with itself alone.
        let (same, kept) = here.squared_overlaps(&here).unwrap();
        assert!((kept - 1.0).abs() < 1e-14);
        assert!((same[(0, 0)] - 1.0).abs() < 1e-14 && same[(0, 1)].abs() < 1e-14);
        // The empty orbitals turned a little and the states in the other order: state 0
        // here is state 1 there.
        let there = states(0.1, [1, 0]);
        let (found, kept) = here.squared_overlaps(&there).unwrap();
        assert!((kept - 1.0).abs() < 1e-14);
        assert!(found[(0, 1)] > 0.98 && found[(0, 0)] < 0.02, "{found:?}");
        assert!(found[(1, 0)] > 0.98 && found[(1, 1)] < 0.02, "{found:?}");
        // The lowest state alone, and what does not fit together.
        assert_eq!(there.clone().lowest(1).count(), 1);
        let other = StateCharacters::new(
            Mat::<f64>::identity(4, 4),
            1,
            vec![0.1],
            Mat::from_fn(3, 1, |p, _| if p == 0 { 1.0 } else { 0.0 }),
        )
        .unwrap();
        assert!(here.squared_overlaps(&other).is_err());
        assert!(
            StateCharacters::new(Mat::<f64>::identity(3, 3), 3, vec![], Mat::zeros(0, 0)).is_err()
        );
    }
}

/// An excited state with what the couplings of an electron transfer need from it: its
/// excitation energy (Hartree), its transition dipole from the ground state (e Bohr), and the
/// change of the electrons on each atom from the ground state to it (Mulliken populations of
/// the unrelaxed difference density; `kinetics::rate::couplings::charge_transfer`); where the
/// calculator gives them, the change of the dipole (generalized Mulliken-Hush) and the
/// populations of the transition density (fragment charge difference).
#[derive(Debug, Clone, PartialEq)]
pub struct Transition {
    /// Excitation energy (Hartree).
    pub excitation: f64,
    /// Transition dipole `<0| r |k>` (e Bohr).
    pub transition_dipole: [f64; 3],
    /// Electrons gained by each atom (negative: lost).
    pub population_change: Vec<f64>,
    /// The dipole of the state less that of the ground state (e Bohr, of the unrelaxed
    /// difference density), when the calculator gives it.
    pub dipole_change: Option<[f64; 3]>,
    /// The Mulliken populations of the transition density on each atom (they add up to zero:
    /// the states are orthogonal), when the calculator gives them.
    pub transition_population: Option<Vec<f64>>,
}
