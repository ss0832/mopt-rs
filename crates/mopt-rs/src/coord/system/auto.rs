//! Automatic choice of the coordinate system (`-coord auto`), made from the structure and made
//! again when the structure changes its character during a run.
//!
//! | Structure | Coordinates | Why (measured, `docs/source/coordinate_systems.md`) |
//! |---|---|---|
//! | no bonds at all (a cluster of atoms) | Cartesian | internal coordinates gain nothing (the same iterations) |
//! | a fragment bonded to its own periodic image (a crystal, a slab) | Cartesian | such a network has no orientation; internal coordinates tie at best and fail for small cells |
//! | dense bonding (more than [`MAX_MEAN_COORDINATION`] bonds per atom on average: metals, close-packed clusters) | Cartesian | the primitive set grows with the cube of the coordination, for no fewer iterations |
//! | more than [`MAX_INTERNAL_ATOMS`] atoms | Cartesian | the transformations cost `O((3N)^3)` per step: 3 s per iteration at 194 atoms |
//! | anything else: a molecule, a complex, a solute with solvent | `tric-cart` | fewest iterations in total over molecules and complexes, with and without a model Hessian, and never more than 4 % above the best system of a kind |
//!
//! During a run the choice is made again, and the coordinates are set up anew
//! ([`CoordinateSystem::needs_reinitialization`]), when the chosen system asks for it itself,
//! when a fragment has come apart or fragments have joined, or when the structure has moved
//! into another row of the table. Bonds are judged with two scales of the covalent radii so
//! that a distance near the limit does not switch the coordinates back and forth: fragments
//! join when they are bonded by the ordinary scale and count as separated only beyond
//! [`SEPARATION_FACTOR`] times that scale, and the kind of coordinates is changed only when
//! both scales agree that the current one no longer fits.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::{Col, Mat};

use super::tric::has_periodic_network;
use super::tric::DEFAULT_CARTESIAN_WEIGHT;
use super::{Cartesian, CoordinateSystem, Tric};
use crate::coord::connectivity::BondConnectivity;
use crate::coord::internal::PrimitiveOptions;
use crate::domain::MolecularSystem;
use crate::error::MoptResult;

/// Above this number of atoms the choice is Cartesian coordinates.
pub const MAX_INTERNAL_ATOMS: usize = 300;
/// Above this average number of bonds per atom the choice is Cartesian coordinates.
pub const MAX_MEAN_COORDINATION: f64 = 5.0;
/// Fragments count as separated when they are unbonded with the bond scale times this factor.
pub const SEPARATION_FACTOR: f64 = 1.25;

/// The systems `auto` chooses from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AutoChoice {
    /// Cartesian coordinates.
    Cartesian,
    /// Translation-rotation internal coordinates stacked with Cartesian coordinates.
    TricCart,
}

impl AutoChoice {
    /// Name of the chosen system in the registry.
    pub fn name(self) -> &'static str {
        match self {
            Self::Cartesian => "cart",
            Self::TricCart => "tric-cart",
        }
    }
}

/// The choice for `sys` with bonds judged by `bond_scale`.
pub fn choose(sys: &MolecularSystem, bond_scale: f64) -> AutoChoice {
    choose_with(sys, &BondConnectivity::from_system(sys, bond_scale))
}

fn choose_with(sys: &MolecularSystem, conn: &BondConnectivity) -> AutoChoice {
    let n = sys.n_atoms();
    let mean_coordination = 2.0 * conn.bonds.len() as f64 / n.max(1) as f64;
    if conn.bonds.is_empty()
        || n > MAX_INTERNAL_ATOMS
        || mean_coordination > MAX_MEAN_COORDINATION
        || has_periodic_network(sys, conn)
    {
        AutoChoice::Cartesian
    } else {
        AutoChoice::TricCart
    }
}

/// Fragment index of every atom.
fn fragment_of(conn: &BondConnectivity) -> Vec<usize> {
    let mut of = vec![0; conn.n_atoms];
    for (k, atoms) in conn.fragments.iter().enumerate() {
        for a in atoms {
            of[a.index()] = k;
        }
    }
    of
}

/// Coordinate system chosen from the structure.
pub struct Auto {
    /// Options for the internal coordinates (`bond_scale` also decides the choice).
    pub options: PrimitiveOptions,
    inner: Box<dyn CoordinateSystem>,
    choice: Option<AutoChoice>,
    /// Fragment of every atom when the coordinates were set up.
    fragment_of: Vec<usize>,
    label: String,
}

impl Default for Auto {
    fn default() -> Self {
        Self {
            options: PrimitiveOptions::default(),
            inner: Box::new(Cartesian),
            choice: None,
            fragment_of: Vec::new(),
            label: "auto".to_string(),
        }
    }
}

impl Auto {
    /// The system in use (`None` before [`CoordinateSystem::initialize`]).
    pub fn choice(&self) -> Option<AutoChoice> {
        self.choice
    }

    fn loose_scale(&self) -> f64 {
        self.options.bond_scale * SEPARATION_FACTOR
    }
}

impl CoordinateSystem for Auto {
    /// `auto`, or `auto (tric-cart)` with the system in use.
    fn name(&self) -> &str {
        &self.label
    }

    fn initialize(&mut self, sys: &MolecularSystem) -> MoptResult<()> {
        let conn = BondConnectivity::from_system(sys, self.options.bond_scale);
        let choice = choose_with(sys, &conn);
        let mut inner: Box<dyn CoordinateSystem> = match choice {
            AutoChoice::Cartesian => Box::new(Cartesian),
            AutoChoice::TricCart => {
                let mut tric = Tric::with_cartesian(DEFAULT_CARTESIAN_WEIGHT)?;
                tric.options = self.options.clone();
                Box::new(tric)
            }
        };
        inner.initialize(sys)?;
        self.inner = inner;
        self.choice = Some(choice);
        self.fragment_of = fragment_of(&conn);
        self.label = format!("auto ({})", choice.name());
        Ok(())
    }

    fn dim(&self, sys: &MolecularSystem) -> usize {
        self.inner.dim(sys)
    }

    fn values(&self, sys: &MolecularSystem) -> MoptResult<Col<f64>> {
        self.inner.values(sys)
    }

    fn labels(&self) -> Option<Vec<String>> {
        self.inner.labels()
    }

    fn wilson_b(&self, sys: &MolecularSystem) -> MoptResult<Mat<f64>> {
        self.inner.wilson_b(sys)
    }

    fn displacement(&self, to: &MolecularSystem, from: &MolecularSystem) -> MoptResult<Col<f64>> {
        self.inner.displacement(to, from)
    }

    fn rigid_body_basis(&self, sys: &MolecularSystem) -> Option<Mat<f64>> {
        self.inner.rigid_body_basis(sys)
    }

    fn is_cartesian(&self) -> bool {
        self.inner.is_cartesian()
    }

    fn needs_reinitialization(&self, sys: &MolecularSystem) -> bool {
        let Some(current) = self.choice else {
            return true;
        };
        if self.inner.needs_reinitialization(sys) {
            return true;
        }
        let tight = BondConnectivity::from_system(sys, self.options.bond_scale);
        let loose = BondConnectivity::from_system(sys, self.loose_scale());
        // Another kind of coordinates only when neither scale supports the current one.
        if choose_with(sys, &tight) != current && choose_with(sys, &loose) != current {
            return true;
        }
        if current == AutoChoice::Cartesian || self.fragment_of.len() != sys.n_atoms() {
            return false;
        }
        // The same kind with other fragments: two of them are bonded now (ordinary scale), or
        // atoms of one are apart (also by the loose scale).
        let joined = tight
            .bonds
            .iter()
            .any(|[i, j]| self.fragment_of[i.index()] != self.fragment_of[j.index()]);
        let loose_of = fragment_of(&loose);
        let parted = (0..sys.n_atoms()).any(|i| {
            (0..i).any(|j| self.fragment_of[i] == self.fragment_of[j] && loose_of[i] != loose_of[j])
        });
        joined || parted
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::domain::{positions, Angstrom, Element, PeriodicCell, Quantity, SpinCharge};

    fn system(rows: &[(&str, [f64; 3])], cell: Option<PeriodicCell>) -> MolecularSystem {
        MolecularSystem::new(
            rows.iter()
                .map(|r| Element::parse_token(r.0).unwrap())
                .collect(),
            &positions::<Angstrom>(&rows.iter().map(|r| r.1).collect::<Vec<_>>()),
            SpinCharge::default(),
            cell,
        )
        .unwrap()
    }

    fn water(shift: [f64; 3]) -> Vec<(&'static str, [f64; 3])> {
        [
            ("O", [0.0, 0.0, 0.117]),
            ("H", [0.0, 0.757, -0.469]),
            ("H", [0.0, -0.757, -0.469]),
        ]
        .into_iter()
        .map(|(s, p)| (s, [p[0] + shift[0], p[1] + shift[1], p[2] + shift[2]]))
        .collect()
    }

    fn chosen(sys: &MolecularSystem) -> AutoChoice {
        let mut auto = Auto::default();
        assert_eq!(auto.name(), "auto");
        auto.initialize(sys).unwrap();
        assert!(!auto.needs_reinitialization(sys));
        assert_eq!(
            auto.name(),
            format!("auto ({})", auto.choice().unwrap().name())
        );
        auto.choice().unwrap()
    }

    #[test]
    fn the_choice_follows_the_structure() {
        // One molecule.
        assert_eq!(
            chosen(&system(&water([0.0; 3]), None)),
            AutoChoice::TricCart
        );
        // Two molecules.
        let mut dimer = water([0.0; 3]);
        dimer.extend(water([3.0, 0.2, 0.3]));
        assert_eq!(chosen(&system(&dimer, None)), AutoChoice::TricCart);
        // A molecule and an atom.
        let mut with_atom = water([0.0; 3]);
        with_atom.push(("Ar", [3.6, 0.0, 0.0]));
        assert_eq!(chosen(&system(&with_atom, None)), AutoChoice::TricCart);
        // Atoms without bonds.
        let atoms: Vec<_> = (0..5).map(|k| ("Ar", [3.8 * k as f64, 0.0, 0.0])).collect();
        assert_eq!(chosen(&system(&atoms, None)), AutoChoice::Cartesian);
        // A close-packed copper cluster: center and twelve neighbours (an icosahedron).
        let phi = 0.5 * (1.0 + 5.0_f64.sqrt());
        let s = 2.55 / (1.0 + phi * phi).sqrt();
        let mut cu = vec![("Cu", [0.0; 3])];
        for (a, b) in [(s, s * phi), (s, -s * phi), (-s, s * phi), (-s, -s * phi)] {
            cu.extend([
                ("Cu", [0.0, a, b]),
                ("Cu", [a, b, 0.0]),
                ("Cu", [b, 0.0, a]),
            ]);
        }
        assert_eq!(chosen(&system(&cu, None)), AutoChoice::Cartesian);
        // A chain bonded to its own periodic image.
        let side = Quantity::<Angstrom>::new(6.0);
        let cell = PeriodicCell::orthorhombic(side, side, side).unwrap();
        let chain: Vec<_> = (0..8).map(|k| ("H", [0.75 * k as f64, 0.5, 0.5])).collect();
        assert_eq!(chosen(&system(&chain, Some(cell))), AutoChoice::Cartesian);
        // More atoms than internal coordinates are used for.
        let many: Vec<_> = (0..MAX_INTERNAL_ATOMS + 1)
            .map(|k| ("C", [1.4 * k as f64, 0.3 * (k % 2) as f64, 0.0]))
            .collect();
        assert_eq!(choose(&system(&many, None), 1.3), AutoChoice::Cartesian);
    }

    #[test]
    fn the_choice_is_made_again_when_fragments_part_or_join_with_hysteresis() {
        let h2 = |d: f64| system(&[("H", [0.0; 3]), ("H", [d, 0.0, 0.0])], None);
        let bonded = |sys: &MolecularSystem, scale: f64| {
            !BondConnectivity::from_system(sys, scale).bonds.is_empty()
        };
        // Distances on both sides of the two limits.
        let (near, between, far) = (0.74, 0.90, 1.30);
        assert!(bonded(&h2(near), 1.3) && !bonded(&h2(between), 1.3));
        assert!(bonded(&h2(between), 1.3 * SEPARATION_FACTOR));
        assert!(!bonded(&h2(far), 1.3 * SEPARATION_FACTOR));

        let mut auto = Auto::default();
        auto.initialize(&h2(near)).unwrap();
        assert_eq!(auto.choice(), Some(AutoChoice::TricCart));
        // Stretched past the ordinary limit only: kept.
        assert!(!auto.needs_reinitialization(&h2(between)));
        // Clearly apart: two atoms, Cartesian coordinates.
        assert!(auto.needs_reinitialization(&h2(far)));
        auto.initialize(&h2(far)).unwrap();
        assert_eq!(auto.choice(), Some(AutoChoice::Cartesian));
        // Coming back: kept until the atoms are bonded by the ordinary limit.
        assert!(!auto.needs_reinitialization(&h2(between)));
        assert!(auto.needs_reinitialization(&h2(near)));

        // Two waters joining through a hydrogen that comes within bonding distance, and a
        // water losing a hydrogen.
        let mut dimer = water([0.0; 3]);
        dimer.extend(water([3.0, 0.2, 0.3]));
        let mut auto = Auto::default();
        auto.initialize(&system(&dimer, None)).unwrap();
        assert_eq!(auto.choice(), Some(AutoChoice::TricCart));
        let mut joined = dimer.clone();
        joined[4].1 = [0.6, 0.75, 0.3];
        assert!(auto.needs_reinitialization(&system(&joined, None)));
        let mut auto = Auto::default();
        auto.initialize(&system(&water([0.0; 3]), None)).unwrap();
        let mut parted = water([0.0; 3]);
        parted[1].1 = [0.0, 2.6, -1.4];
        let parted = system(&parted, None);
        assert!(auto.needs_reinitialization(&parted));
        auto.initialize(&parted).unwrap();
        assert_eq!(auto.choice(), Some(AutoChoice::TricCart));
    }
}
