//! Coordinates: internal-coordinate primitives, bond connectivity, structure alignment,
//! hyperspherical coordinates, and the coordinate systems used by optimizers.
//!
//! MultiOptPy: `Coordinate/`, `Utils/bond_connectivity.py`, `Utils/calc_tools.py`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

pub mod alignment;
pub mod connectivity;
// Bond graphs: classes of equivalent atoms, isomorphism, molecules in a cell.
pub mod graph;
pub mod hyperspherical;
pub mod internal;
// RMSD independent of the numbering of equivalent atoms.
pub mod invariant_rmsd;
pub mod primitives;
pub mod system;
// Z-matrices along the bond graph.
pub mod zmatrix;
