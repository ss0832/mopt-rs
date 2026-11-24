//! Molecular orbitals as a calculator hands them out: coefficients in its atomic orbitals,
//! energies, occupations, and the overlap of the atomic orbitals.
//!
//! MultiOptPy: no counterpart (new in mopt-rs). `WFAnalyzer/MO_analysis.py` expects
//! `orbital_coefficients`, `overlap_matrix`, and `orbital_energies` on a calculator, which
//! none of MultiOptPy's calculators provides.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Mat;

use crate::error::{MoptError, MoptResult};

/// The orbitals of one spin, or the spatial orbitals of a closed shell.
#[derive(Debug, Clone, PartialEq)]
pub struct OrbitalSet {
    /// Coefficients in the atomic orbitals: column `k` is orbital `k`.
    pub coefficients: Mat<f64>,
    /// Orbital energies (Hartree).
    pub energies: Vec<f64>,
    /// Occupations (0 to 1 for one spin, 0 to 2 for the orbitals of a closed shell).
    pub occupations: Vec<f64>,
}

impl OrbitalSet {
    /// Checked construction: one energy and one occupation per column, all finite.
    pub fn new(
        coefficients: Mat<f64>,
        energies: Vec<f64>,
        occupations: Vec<f64>,
    ) -> MoptResult<Self> {
        let n = coefficients.ncols();
        if energies.len() != n || occupations.len() != n {
            return Err(MoptError::config(format!(
                "{n} orbitals with {} energies and {} occupations",
                energies.len(),
                occupations.len()
            )));
        }
        let finite = energies.iter().chain(&occupations).all(|v| v.is_finite())
            && (0..n).all(|k| (0..coefficients.nrows()).all(|r| coefficients[(r, k)].is_finite()));
        if !finite {
            return Err(MoptError::calc("an orbital is not finite"));
        }
        Ok(Self {
            coefficients,
            energies,
            occupations,
        })
    }

    /// Number of orbitals.
    pub fn len(&self) -> usize {
        self.energies.len()
    }

    /// Whether there are no orbitals.
    pub fn is_empty(&self) -> bool {
        self.energies.is_empty()
    }

    /// Number of atomic orbitals.
    pub fn n_basis(&self) -> usize {
        self.coefficients.nrows()
    }
}

/// The orbitals of a structure.
#[derive(Debug, Clone, PartialEq)]
pub struct Orbitals {
    /// Overlap of the atomic orbitals.
    pub overlap: Mat<f64>,
    /// The orbitals of the alpha spin, or the spatial orbitals of a closed shell (then
    /// `beta` is `None` and the occupations go up to 2).
    pub alpha: OrbitalSet,
    /// The orbitals of the beta spin of an open shell.
    pub beta: Option<OrbitalSet>,
    /// The atom (0-based) of every atomic orbital, when the calculator says (the projection of
    /// the orbitals of one structure on those of another needs it).
    pub atom_of_ao: Option<Vec<usize>>,
}

impl Orbitals {
    /// Checked construction: the sets and the overlap are in the same atomic orbitals.
    pub fn new(overlap: Mat<f64>, alpha: OrbitalSet, beta: Option<OrbitalSet>) -> MoptResult<Self> {
        let n = overlap.nrows();
        let sets = std::iter::once(&alpha).chain(beta.as_ref());
        if overlap.ncols() != n || sets.clone().any(|s| s.n_basis() != n) {
            return Err(MoptError::config(
                "orbitals and overlap matrix differ in the number of atomic orbitals",
            ));
        }
        Ok(Self {
            overlap,
            alpha,
            beta,
            atom_of_ao: None,
        })
    }

    /// With the atom of every atomic orbital (checked: one per atomic orbital).
    pub fn with_atoms(mut self, atom_of_ao: Vec<usize>) -> MoptResult<Self> {
        if atom_of_ao.len() != self.overlap.nrows() {
            return Err(MoptError::config(format!(
                "{} atoms of atomic orbitals for {} atomic orbitals",
                atom_of_ao.len(),
                self.overlap.nrows()
            )));
        }
        self.atom_of_ao = Some(atom_of_ao);
        Ok(self)
    }

    /// The sets with their names (`alpha` and `beta`, or `closed shell`).
    pub fn sets(&self) -> Vec<(&'static str, &OrbitalSet)> {
        match &self.beta {
            Some(beta) => vec![("alpha", &self.alpha), ("beta", beta)],
            None => vec![("closed shell", &self.alpha)],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes_are_checked() {
        let c = Mat::<f64>::identity(3, 2);
        assert!(OrbitalSet::new(c.clone(), vec![0.0, 1.0], vec![2.0, 0.0]).is_ok());
        assert!(OrbitalSet::new(c.clone(), vec![0.0], vec![2.0, 0.0]).is_err());
        assert!(OrbitalSet::new(c.clone(), vec![0.0, f64::NAN], vec![2.0, 0.0]).is_err());
        let set = OrbitalSet::new(c, vec![0.0, 1.0], vec![2.0, 0.0]).unwrap();
        assert_eq!((set.len(), set.n_basis(), set.is_empty()), (2, 3, false));
        let closed = Orbitals::new(Mat::identity(3, 3), set.clone(), None).unwrap();
        assert_eq!(closed.sets()[0].0, "closed shell");
        let open = Orbitals::new(Mat::identity(3, 3), set.clone(), Some(set.clone())).unwrap();
        assert_eq!(open.sets().len(), 2);
        assert!(Orbitals::new(Mat::identity(4, 4), set, None).is_err());
    }
}
