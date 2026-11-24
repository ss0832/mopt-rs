//! The vibrational analysis of `confsearch -freq` with constraints: the modes on the
//! constraint surface, as `optmain -freq` takes them (the Hessian of the Lagrangian with the
//! constraint normals projected), so that a held bond is no imaginary mode and the analysis has
//! 3N - 6 - m modes.
//!
//! MultiOptPy: `Entrypoints/conformation_search.py` (no vibrational analysis of the conformers
//! there; new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used)]

use mopt_rs::cli::confsearch::{config_from_cli, run_with};
use mopt_rs::pes::registry::CalculatorRegistry;

/// Ar3 with the bond 1-2 stretched to 5.45 Angstrom (1.6 sigma, beyond the inflection point
/// of the Lennard-Jones pair, where its curvature is negative) and the third atom at the
/// minimum of the pair potential from both.
const AR3: &str = "3\nAr3, one bond stretched\nAr 0.0 0.0 0.0\nAr 5.448 0.0 0.0\n\
Ar 2.724 2.678 0.0\n";

/// The wavenumbers of the table `normal_modes.txt` (the rows `Freq [cm^-1]`).
fn wavenumbers(table: &str) -> Vec<f64> {
    table
        .lines()
        .filter_map(|l| l.strip_prefix("Freq [cm^-1]"))
        .flat_map(|rest| rest.split_whitespace().map(|w| w.parse::<f64>().unwrap()))
        .collect()
}

#[test]
fn a_held_bond_is_no_imaginary_mode_of_the_conformer() {
    let dir = std::env::temp_dir().join(format!("mopt_cs_freq_held_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let file = dir.join("ar3.xyz");
    std::fs::write(&file, AR3).unwrap();
    let file = file.to_string_lossy().into_owned();
    let config = config_from_cli(&[
        &file,
        "-os",
        "lj",
        "-ms",
        "0",
        "-pc",
        "bond",
        "1,2",
        "-freq",
        "-cs_opt",
        "seed=1",
        "nudge=off",
        "mode=minima",
    ])
    .unwrap();
    let out = run_with(&config, &dir, None, &CalculatorRegistry::native()).unwrap();
    let analysis = out.directory.join("vibrations").join("EQ0");
    let table = std::fs::read_to_string(analysis.join("normal_modes.txt")).unwrap();
    let w = wavenumbers(&table);
    // 3N - 6 - m: three atoms, one held bond.
    assert_eq!(w.len(), 9 - 6 - 1, "{table}");
    assert!(w.iter().all(|x| *x > 0.0), "{w:?}");
    let log = std::fs::read_to_string(out.directory.join("mopt.log")).unwrap();
    assert!(
        log.contains("Vibrational analysis of EQ0: 0 imaginary frequencies"),
        "{log}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}
