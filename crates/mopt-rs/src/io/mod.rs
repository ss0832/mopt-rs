//! Multi-format molecular structure, trajectory, CSV profile, and `software_path.conf` I/O.
//!
//! MultiOptPy: `fileio.py`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

pub mod chem_formats;
pub mod csv_io;
pub mod hessian_file;
pub mod npy;
pub mod software_conf;
pub mod xyz;

pub use chem_formats::{
    parse_gamess_inp, parse_gjf_str, parse_mol2_str, parse_mol_str, parse_orca_inp,
    parse_qchem_inp, read_molecular_file, write_gjf_str, write_molecular_file, GjfOniomAtomMeta,
    ParsedGjf,
};
pub use csv_io::{read_csv_numeric_columns, write_csv_table};
pub use npy::{read_npy_matrix, write_npy_matrix};
pub use software_conf::SoftwarePathConf;
pub use xyz::{
    format_xyz_system, format_xyz_trajectory, parse_xyz_str, parse_xyz_trajectory_str,
    read_xyz_file, read_xyz_trajectory_file, write_xyz_file, write_xyz_trajectory_file,
};
