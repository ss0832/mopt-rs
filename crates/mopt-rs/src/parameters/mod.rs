//! Physical, covalent, van der Waals, and dispersion (`D2`, `D3(BJ)`, `D4`) atomic parameter
//! tables in internal atomic units (Bohr / Hartree), and MultiOptPy's tables for its legacy
//! models (`multioptpy_tables`, `multioptpy_gfn`, in MultiOptPy's units).
//!
//! Provenance and citations for all parameter sets are documented in
//! `third_party/DISPERSION_AND_PARAMETERS.md`.
//!
//! MultiOptPy: `Parameters/` (`parameter.py` and the tables next to it).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

pub mod covalent;
// Generated tables (tools/params); kept as generated.
#[rustfmt::skip]
pub mod d3_reference;
#[rustfmt::skip]
pub mod d4_reference;
pub mod dispersion;
pub mod forcefield;
#[rustfmt::skip]
pub mod gfn0_xtb;
#[rustfmt::skip]
pub mod gfnff;
pub mod isotope_masses;
pub mod model_hessian;
#[rustfmt::skip]
pub mod multioptpy_gfn;
#[rustfmt::skip]
pub mod multioptpy_tables;
pub mod vdw;

pub use covalent::{
    covalent_radius_bohr, double_covalent_radius_bohr, sum_covalent_radii_bohr,
    triple_covalent_radius_bohr,
};
pub use dispersion::{d2_c6_au, d2_vdw_radius_bohr, D2Params, D3Params, D4Params};
pub use forcefield::{
    dreiding_vdw_distance_bohr, dreiding_vdw_well_depth_hartree, gnb_beta_bohr, gnb_c6_au,
    gnb_radius_bohr, gnb_s_bohr, gnb_vdw_distance_bohr, gnb_vdw_well_depth_hartree,
    uff_effective_charge, uff_vdw_distance_bohr, uff_vdw_well_depth_hartree,
};
