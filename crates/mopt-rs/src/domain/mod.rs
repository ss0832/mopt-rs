//! Core chemistry domain types: unit-tagged quantities, atom ids, elements, periodic cells,
//! molecules, evaluation I/O, and trajectories.
//!
//! MultiOptPy: `Parameters/atomic_number.py`, `Parameters/unit_values.py`, `Utils/pbc.py`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

pub mod atom_id;
pub mod element;
pub mod eval;
pub mod molecule;
pub mod orbitals;
pub mod pbc;
pub mod quantity;
pub mod trajectory;
pub mod units;

pub use atom_id::AtomId;
pub use element::Element;
pub use eval::{EvalOutput, EvalRequest, OrbitalData, StrainHessian};
pub use molecule::{MolecularSystem, SpinCharge};
pub use orbitals::{OrbitalSet, Orbitals};
pub use pbc::PeriodicCell;
pub use quantity::{
    positions, Angstrom, Bohr, Degree, ElectronVolt, Hartree, HartreePerBohr, KcalPerMol, KjPerMol,
    Position, Quantity, Radian, Vector3,
};
pub use trajectory::{Trajectory, TrajectoryFrame};
pub use units::Units;
