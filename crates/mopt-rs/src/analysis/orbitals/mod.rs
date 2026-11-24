//! Analyses of molecular orbitals along a displacement of the nuclei.
//!
//! - [`tracking`]: which orbital of a neighbouring structure is which (order, rotation among
//!   degenerate ones, sign); what the two analyses below rest on.
//! - [`nro`]: natural reaction orbitals, the pairs of an occupied and an empty orbital
//!   through which the electrons respond to the displacement, and their singular values.
//! - [`roet`]: the orbitals whose energies change most along a path (reactive orbitals) and
//!   the derivatives of orbital energies along it.
//!
//! The orbitals themselves come from a calculator ([`crate::pes::Calculator::orbitals`]).
//!
//! MultiOptPy: `WFAnalyzer/MO_analysis.py:NROAnalysis`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

pub mod nro;
pub mod roet;
pub mod tracking;
