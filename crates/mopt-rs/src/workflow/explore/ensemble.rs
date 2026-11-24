//! An ensemble of minima: duplicates removed, conformers told from rotamers, Boltzmann
//! populations, and the conformational entropy.
//!
//! A new structure is compared with every member whose energy is within `energy` of its own:
//!
//! 1. RMSD in the same numbering below `rmsd`: a **duplicate** (the same structure again);
//! 2. relative difference of a rotational constant above `rotational`: a different conformer;
//! 3. RMSD below `rmsd` after the best renumbering of equivalent atoms (and, with `mirror`, of
//!    the mirror image; [`crate::coord::invariant_rmsd`]): a **rotamer** of the member (the
//!    same conformer with permuted atoms, such as a turned methyl group), counted with it;
//! 4. otherwise a different conformer.
//!
//! The thresholds and the order of the tests are those of CREGEN in the iMTD-GC paper
//! (Pracht, Bohle, Grimme, Phys. Chem. Chem. Phys. 2020, 22, 7169: energy, RMSD, rotational
//! constants; 0.05 kcal/mol, 0.125 Angstrom, 1 %), with the rotamers decided by the
//! renumbering itself instead of by the rotational constants alone. Molecules in a periodic
//! cell are compared made whole ([`crate::coord::graph::whole_molecules`]).
//!
//! With `mirror`, a chiral conformer stands for itself and its mirror image (degeneracy 2) in
//! the populations `p_i = g_i exp(-(E_i - E_min) / kT) / Z` and in the conformational entropy
//! `S = -R sum_i p_i ln(p_i / g_i)`.
//!
//! MultiOptPy: `Entrypoints/conformation_search.py:is_identical` and `check_identical`
//!  (energies within 1e-4 Hartree and sorted distance matrices within 0.1 Angstrom; see
//!  `docs/source/migration_workflows.md`), `calc_boltzmann_distribution`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use crate::coord::graph::{whole_molecules, MolecularGraph, MULTIOPTPY_BOND_SCALE};
use crate::coord::invariant_rmsd::{direct_rmsd, invariant_rmsd, is_chiral, RmsdOptions};
use crate::domain::{AtomId, MolecularSystem, Units};
use crate::error::{MoptError, MoptResult};
use crate::vib::rotational_constants_ghz;

/// Thresholds of the comparison.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CompareSettings {
    /// Largest energy difference of the same conformer (Hartree).
    pub energy: f64,
    /// Largest RMSD of the same conformer (Bohr).
    pub rmsd: f64,
    /// Largest relative difference of a rotational constant of the same conformer.
    pub rotational: f64,
    /// Renumber equivalent atoms (rotamers).
    pub permutations: bool,
    /// Mirror images are the same conformer.
    pub mirror: bool,
    /// Scale of the covalent radii of the bonds of the comparison.
    pub bond_scale: f64,
}

impl Default for CompareSettings {
    /// CREGEN's thresholds: 0.05 kcal/mol, 0.125 Angstrom, 1 %.
    fn default() -> Self {
        Self {
            energy: 0.05 * Units::KCAL_MOL_TO_HARTREE,
            rmsd: 0.125 * Units::ANGSTROM_TO_BOHR,
            rotational: 0.01,
            permutations: true,
            mirror: true,
            bond_scale: MULTIOPTPY_BOND_SCALE,
        }
    }
}

/// How a structure compares with the ensemble.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Verdict {
    /// A new conformer (the index it gets).
    New(usize),
    /// The same structure as member `of` (RMSD in Bohr).
    Duplicate {
        /// The member.
        of: usize,
        /// RMSD in the same numbering (Bohr).
        rmsd: f64,
    },
    /// A rotamer of member `of`: the same conformer with permuted (or mirrored) atoms.
    Rotamer {
        /// The member.
        of: usize,
        /// RMSD after the renumbering (Bohr).
        rmsd: f64,
        /// Whether the mirror image matched.
        mirrored: bool,
    },
}

/// A member of the ensemble.
#[derive(Debug, Clone)]
pub struct Member {
    /// The structure as it was found.
    pub system: MolecularSystem,
    /// Its energy (Hartree).
    pub energy: f64,
    /// Its bond graph.
    pub graph: MolecularGraph,
    /// Rotational constants (GHz, largest first).
    pub rotational: [f64; 3],
    /// How it was found.
    pub origin: String,
    /// Rotamers found of it (structures matched after a renumbering).
    pub rotamers: Vec<MolecularSystem>,
    /// Whether its mirror image is a different structure.
    pub chiral: bool,
    /// Trials started from it.
    pub visits: usize,
    comparable: MolecularSystem,
    classes: Vec<usize>,
}

impl Member {
    /// Number of copies of it in the populations: 2 for a chiral conformer that stands for its
    /// mirror image, 1 otherwise.
    pub fn degeneracy(&self, mirror: bool) -> f64 {
        if mirror && self.chiral {
            2.0
        } else {
            1.0
        }
    }
}

/// Populations and thermodynamic functions of an ensemble.
#[derive(Debug, Clone, PartialEq)]
pub struct EnsembleThermo {
    /// Temperature (K).
    pub temperature: f64,
    /// Population of every member (in the order of the members).
    pub populations: Vec<f64>,
    /// Average energy above the lowest member (Hartree).
    pub average_energy: f64,
    /// Conformational entropy (J / (mol K)).
    pub entropy: f64,
    /// `-kT ln Z` with the energies above the lowest member (Hartree): `<E> - T S`.
    pub free_energy: f64,
}

/// Members found so far.
#[derive(Debug, Clone)]
pub struct ConformerEnsemble {
    settings: CompareSettings,
    members: Vec<Member>,
    atoms: Option<Vec<usize>>,
}

fn relative_difference(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a.iter()
        .zip(b)
        .filter(|(x, y)| x.is_finite() && y.is_finite() && **x > 1e-9 && **y > 1e-9)
        .map(|(x, y)| (x - y).abs() / y.max(*x))
        .fold(0.0, f64::max)
}

impl ConformerEnsemble {
    /// An empty ensemble with `settings`.
    pub fn new(settings: CompareSettings) -> Self {
        Self {
            settings,
            members: Vec::new(),
            atoms: None,
        }
    }

    /// Compare only these atoms (from 0), e.g. a solute among solvent molecules.
    #[must_use]
    pub fn comparing(mut self, atoms: Vec<usize>) -> Self {
        self.atoms = (!atoms.is_empty()).then_some(atoms);
        self
    }

    /// The thresholds.
    pub fn settings(&self) -> &CompareSettings {
        &self.settings
    }

    /// The members in the order they were found.
    pub fn members(&self) -> &[Member] {
        &self.members
    }

    /// Member `i` to change its visits.
    pub fn member_mut(&mut self, i: usize) -> Option<&mut Member> {
        self.members.get_mut(i)
    }

    /// Number of members.
    pub fn len(&self) -> usize {
        self.members.len()
    }

    /// Whether there is no member.
    pub fn is_empty(&self) -> bool {
        self.members.is_empty()
    }

    /// The structure, graph, rotational constants, and classes of the comparison.
    fn prepare(
        &self,
        sys: &MolecularSystem,
    ) -> MoptResult<(MolecularGraph, MolecularSystem, [f64; 3], Vec<usize>)> {
        let graph = MolecularGraph::from_system(sys, self.settings.bond_scale);
        let whole = whole_molecules(sys, &graph)?;
        let mut cmp = whole;
        cmp.cell = None;
        let classes_all = graph.classes();
        let (cmp, classes) = match &self.atoms {
            Some(atoms) => {
                let ids: Vec<AtomId> = atoms
                    .iter()
                    .map(|&a| AtomId::new(a).check(sys.n_atoms()))
                    .collect::<MoptResult<_>>()?;
                (
                    cmp.subsystem(&ids, cmp.spin_charge)?,
                    atoms.iter().map(|&a| classes_all[a]).collect(),
                )
            }
            None => (cmp, classes_all),
        };
        let rotational = rotational_constants_ghz(&cmp)?;
        Ok((graph, cmp, rotational, classes))
    }

    /// How `sys` with `energy` compares with the members.
    pub fn classify(&self, sys: &MolecularSystem, energy: f64) -> MoptResult<Verdict> {
        let (_, cmp, rotational, classes) = self.prepare(sys)?;
        self.compare(&cmp, energy, &rotational, &classes)
    }

    fn compare(
        &self,
        cmp: &MolecularSystem,
        energy: f64,
        rotational: &[f64; 3],
        classes: &[usize],
    ) -> MoptResult<Verdict> {
        let s = &self.settings;
        let options = RmsdOptions {
            permutations: true,
            mirror: s.mirror,
            rotation: true,
        };
        for (i, m) in self.members.iter().enumerate() {
            if (m.energy - energy).abs() >= s.energy || m.comparable.n_atoms() != cmp.n_atoms() {
                continue;
            }
            let direct = direct_rmsd(cmp, &m.comparable)?;
            if direct < s.rmsd {
                return Ok(Verdict::Duplicate {
                    of: i,
                    rmsd: direct,
                });
            }
            if relative_difference(rotational, &m.rotational) >= s.rotational {
                continue;
            }
            let mut sa = classes.to_vec();
            let mut sb = m.classes.clone();
            sa.sort_unstable();
            sb.sort_unstable();
            if !s.permutations || sa != sb {
                continue;
            }
            let found = invariant_rmsd(cmp, &m.comparable, classes, &m.classes, &options)?;
            if found.rmsd < s.rmsd {
                return Ok(Verdict::Rotamer {
                    of: i,
                    rmsd: found.rmsd,
                    mirrored: found.mirrored,
                });
            }
        }
        Ok(Verdict::New(self.members.len()))
    }

    /// Compare `sys` with the members and keep it as a new member or as a rotamer of one.
    pub fn add(&mut self, sys: &MolecularSystem, energy: f64, origin: &str) -> MoptResult<Verdict> {
        if !energy.is_finite() {
            return Err(MoptError::calc("a minimum without a finite energy"));
        }
        let (graph, cmp, rotational, classes) = self.prepare(sys)?;
        let verdict = self.compare(&cmp, energy, &rotational, &classes)?;
        match verdict {
            Verdict::New(_) => {
                let chiral = is_chiral(&cmp, &classes, self.settings.rmsd)?;
                self.members.push(Member {
                    system: sys.clone(),
                    energy,
                    graph,
                    rotational,
                    origin: origin.to_string(),
                    rotamers: Vec::new(),
                    chiral,
                    visits: 0,
                    comparable: cmp,
                    classes,
                });
            }
            Verdict::Rotamer { of, .. } => self.members[of].rotamers.push(sys.clone()),
            Verdict::Duplicate { .. } => {}
        }
        Ok(verdict)
    }

    /// RMSD (Bohr) of members `i` and `j` after the best superposition and, where they have
    /// the same classes of atoms, the best renumbering (and mirror image, with `mirror`).
    pub fn rmsd_between(&self, i: usize, j: usize) -> MoptResult<f64> {
        let (Some(a), Some(b)) = (self.members.get(i), self.members.get(j)) else {
            return Err(MoptError::config("no such member of the ensemble"));
        };
        let mut sa = a.classes.clone();
        let mut sb = b.classes.clone();
        sa.sort_unstable();
        sb.sort_unstable();
        if sa != sb || !self.settings.permutations {
            return direct_rmsd(&a.comparable, &b.comparable);
        }
        let options = RmsdOptions {
            permutations: true,
            mirror: self.settings.mirror,
            rotation: true,
        };
        Ok(invariant_rmsd(
            &a.comparable,
            &b.comparable,
            &a.classes,
            &b.classes,
            &options,
        )?
        .rmsd)
    }

    /// Index of the member of lowest energy.
    pub fn lowest(&self) -> Option<usize> {
        (0..self.members.len())
            .min_by(|&a, &b| self.members[a].energy.total_cmp(&self.members[b].energy))
    }

    /// Indices of the members by energy, lowest first.
    pub fn by_energy(&self) -> Vec<usize> {
        let mut order: Vec<usize> = (0..self.members.len()).collect();
        order.sort_by(|&a, &b| self.members[a].energy.total_cmp(&self.members[b].energy));
        order
    }

    /// Indices of the members within `window` (Hartree) of the lowest, lowest first.
    pub fn within(&self, window: f64) -> Vec<usize> {
        let Some(low) = self.lowest().map(|i| self.members[i].energy) else {
            return Vec::new();
        };
        self.by_energy()
            .into_iter()
            .filter(|&i| self.members[i].energy - low <= window)
            .collect()
    }

    /// Populations, average energy, conformational entropy, and free energy at `temperature`.
    pub fn thermo(&self, temperature: f64) -> MoptResult<EnsembleThermo> {
        let Some(low) = self.lowest().map(|i| self.members[i].energy) else {
            return Err(MoptError::config("an empty ensemble has no populations"));
        };
        if !(temperature.is_finite() && temperature > 0.0) {
            return Err(MoptError::config("the temperature must be positive"));
        }
        let kt = Units::BOLTZMANN_HARTREE_PER_K * temperature;
        let mirror = self.settings.mirror;
        // log(g_i) - (E_i - E_min) / kT, then the log-sum-exp.
        let logs: Vec<f64> = self
            .members
            .iter()
            .map(|m| m.degeneracy(mirror).ln() - (m.energy - low) / kt)
            .collect();
        let top = logs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let log_z = top + logs.iter().map(|l| (l - top).exp()).sum::<f64>().ln();
        let populations: Vec<f64> = logs.iter().map(|l| (l - log_z).exp()).collect();
        let average_energy = self
            .members
            .iter()
            .zip(&populations)
            .map(|(m, p)| p * (m.energy - low))
            .sum::<f64>();
        let entropy_over_r: f64 = self
            .members
            .iter()
            .zip(&populations)
            .filter(|(_, p)| **p > 0.0)
            .map(|(m, p)| -p * (p / m.degeneracy(mirror)).ln())
            .sum();
        // `+ 0.0` turns the negative zeros of a single member into zeros.
        Ok(EnsembleThermo {
            temperature,
            populations,
            average_energy: average_energy + 0.0,
            entropy: entropy_over_r * Units::GAS_CONSTANT_J_MOL_K + 0.0,
            free_energy: -kt * log_z + 0.0,
        })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::coord::invariant_rmsd::renumbered;
    use crate::domain::{Position, SpinCharge};
    use crate::io::xyz::parse_xyz_str;

    const ETHANE: &str = "8\n\nC 0.000 0.000 0.765\nC 0.000 0.000 -0.765\n\
H 1.018 0.000 1.162\nH -0.509 0.882 1.162\nH -0.509 -0.882 1.162\n\
H -1.018 0.000 -1.162\nH 0.509 -0.882 -1.162\nH 0.509 0.882 -1.162\n";

    fn ethane() -> MolecularSystem {
        parse_xyz_str(ETHANE, SpinCharge::default(), None).unwrap()
    }

    #[test]
    fn duplicates_rotamers_and_new_conformers_are_told_apart() {
        let mut e = ConformerEnsemble::new(CompareSettings::default());
        let a = ethane();
        assert_eq!(e.add(&a, -1.0, "input").unwrap(), Verdict::New(0));
        // The same structure moved: a duplicate.
        let mut moved = a.clone();
        for at in a.atoms() {
            let p = a.position(at).value();
            moved.set_position(at, Position::new([p[1] + 1.0, -p[0], p[2]]));
        }
        assert!(matches!(
            e.add(&moved, -1.0, "x").unwrap(),
            Verdict::Duplicate { of: 0, .. }
        ));
        // A methyl turned by 120 degrees: the same conformer with permuted hydrogens.
        let turned = renumbered(&a, &[0, 1, 2, 3, 4, 6, 7, 5]).unwrap();
        let v = e.add(&turned, -1.0 + 1e-6, "y").unwrap();
        assert!(matches!(v, Verdict::Rotamer { of: 0, .. }), "{v:?}");
        assert_eq!(e.members()[0].rotamers.len(), 1);
        // Another energy: a new conformer.
        assert_eq!(e.add(&a, -0.99, "z").unwrap(), Verdict::New(1));
        assert_eq!(e.lowest(), Some(0));
        assert_eq!(e.within(0.001), vec![0]);
        assert!(!e.members()[0].chiral);
    }

    #[test]
    fn populations_and_entropy_follow_the_degeneracies() {
        let mut e = ConformerEnsemble::new(CompareSettings::default());
        let a = ethane();
        e.add(&a, 0.0, "input").unwrap();
        e.add(&a, 1.0e-3, "second").unwrap();
        let t = e.thermo(298.15).unwrap();
        let kt = Units::BOLTZMANN_HARTREE_PER_K * 298.15;
        let w = (-1.0e-3 / kt).exp();
        assert!((t.populations[1] - w / (1.0 + w)).abs() < 1e-12);
        let p = t.populations.clone();
        let s = -(p[0] * p[0].ln() + p[1] * p[1].ln()) * Units::GAS_CONSTANT_J_MOL_K;
        assert!((t.entropy - s).abs() < 1e-9);
        assert!(
            (t.free_energy
                - (t.average_energy
                    - 298.15 * s / Units::GAS_CONSTANT_J_MOL_K * Units::BOLTZMANN_HARTREE_PER_K))
                .abs()
                < 1e-12
        );
        // A chiral conformer counts twice when mirror images are merged.
        e.members[1].chiral = true;
        let t2 = e.thermo(298.15).unwrap();
        assert!((t2.populations[1] - 2.0 * w / (1.0 + 2.0 * w)).abs() < 1e-12);
        assert!(ConformerEnsemble::new(CompareSettings::default())
            .thermo(300.0)
            .is_err());
    }
}
