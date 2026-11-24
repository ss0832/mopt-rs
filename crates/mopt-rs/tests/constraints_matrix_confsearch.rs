//! The constraints of `confsearch` in every kind of run: {frozen atoms, a distance, a centroid
//! distance of two fragments (`fbond`), an angle, a rigid group (`-gfix`), a user-defined
//! function} x {a molecule, a cluster in a periodic cell, ONIOM microiterations}. The
//! candidates are made by kicks of the atoms (`method=kick`), which move every atom: every
//! conformer the command writes holds the constraints at the values of the input, and the
//! frozen atoms stay where they are in the input.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used, clippy::panic)]

mod matrix_support;

use matrix_support::{check, frames, kinds, read, scratch, write, FROM_FILES, FROZEN_IN_FILES};
use mopt_rs::cli::confsearch::{config_from_cli, run_with};
use mopt_rs::pes::registry::CalculatorRegistry;

/// n-butane, all anti.
const BUTANE: &str = "14\nn-butane all-anti\nC 0.000000 0.000000 0.000000\n\
C 1.268427 0.855565 0.000000\nC 2.536855 0.000000 0.000000\nC 3.805282 0.855565 0.000000\n\
H -0.875864 0.648816 0.000000\nH -0.013866 -0.629150 0.889988\nH -0.013866 -0.629150 -0.889988\n\
H 1.268427 1.484886 0.889975\nH 1.268427 1.484886 -0.889975\nH 2.536855 -0.629321 -0.889975\n\
H 2.536855 -0.629321 0.889975\nH 4.681147 0.206750 0.000000\nH 3.819149 1.484715 0.889988\n\
H 3.819149 1.484715 -0.889988\n";

/// Ar7 in a cubic box of 30 Angstrom.
const AR7: &str = "7\nLattice=\"30.0 0.0 0.0 0.0 30.0 0.0 0.0 0.0 30.0\" pbc=\"T T T\"\n\
Ar 3.2175 0.0000 0.0000\nAr 0.9943 3.0601 0.0495\nAr -2.6030 1.8912 0.1155\n\
Ar -2.6030 -1.8912 0.1815\nAr 0.9943 -3.0601 0.2475\nAr 0.0000 0.0000 3.4650\n\
Ar 0.3300 0.0000 -3.3000\n";

const ETHANOL: &str = "9\nethanol\nC 0.000 0.000 0.000\nC 1.520 0.020 0.000\n\
O 2.020 1.330 0.050\nH 2.990 1.330 0.000\nH -0.390 1.030 0.000\nH -0.390 -0.510 0.890\n\
H -0.390 -0.510 -0.890\nH 1.910 -0.510 0.890\nH 1.910 -0.510 -0.890\n";

#[allow(clippy::too_many_arguments)]
fn context(
    name: &str,
    text: &str,
    calc: &[&str],
    atoms: [usize; 4],
    fragments: (&[usize], &[usize]),
    radius: &str,
) {
    let dir = scratch("csmatrix", name);
    let file = write(&dir, "start.xyz", text);
    let input = read(&file);
    for (kind, options) in kinds(atoms, fragments) {
        let cell = format!("{name} x {kind}");
        let mut args: Vec<String> = vec![file.to_string_lossy().into_owned()];
        args.extend(calc.iter().map(|s| s.to_string()));
        args.extend(
            [
                "-ms",
                "3",
                "-cs_opt",
                "method=kick",
                radius,
                "seed=3",
                "verify=off",
                "nudge=off",
                "mode=minima",
            ]
            .iter()
            .map(|s| s.to_string()),
        );
        args.extend(options.iter().cloned());
        let argv: Vec<&str> = args.iter().map(String::as_str).collect();
        let config = config_from_cli(&argv).unwrap();
        let out = run_with(&config, &dir, None, &CalculatorRegistry::native())
            .unwrap_or_else(|e| panic!("{cell}: {e}"));
        let written = frames(&out.conformers);
        assert_eq!(written.len(), out.count, "{cell}");
        check(
            &cell,
            &written,
            &config.optmain.settings.constraints,
            &input,
            FROM_FILES,
            FROZEN_IN_FILES,
            false,
        );
        // Every EQ file as well.
        let mut eq = Vec::new();
        for k in 0..out.count {
            let path = out.directory.join(format!("start_EQ{k}.xyz"));
            if path.is_file() {
                eq.push(read(&path));
            }
        }
        check(
            &cell,
            &eq,
            &config.optmain.settings.constraints,
            &input,
            FROM_FILES,
            FROZEN_IN_FILES,
            false,
        );
        if kind == "fbond" {
            // The fragments deform while their centroids keep their distance.
            let d0 = matrix_support::distances(&input, fragments.0);
            let moved = written.iter().any(|s| {
                matrix_support::distances(s, fragments.0)
                    .iter()
                    .zip(&d0)
                    .any(|(a, b)| (a - b).abs() > 1e-3)
            });
            assert!(moved, "{cell}: the fragment did not deform");
        }
        let log = std::fs::read_to_string(out.directory.join("mopt.log")).unwrap();
        assert!(log.contains("off the constraints"), "{cell}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn constraints_of_conformers_of_a_molecule() {
    context(
        "molecule",
        BUTANE,
        &["-os", "gfnff"],
        [1, 2, 3, 4],
        (&[1, 5, 6, 7], &[4, 12, 13, 14]),
        "move_radius=0.4",
    );
}

#[test]
fn constraints_of_conformers_in_a_periodic_cell() {
    context(
        "periodic",
        AR7,
        &["-os", "lj", "-ns", "1000", "-lcc"],
        [1, 2, 3, 4],
        (&[1, 2, 3], &[4, 5, 6]),
        "move_radius=0.7",
    );
}

#[test]
fn constraints_of_conformers_with_oniom_microiterations() {
    // Atoms 2, 3, 4 (C, O, H) are in the model region of the core.
    context(
        "oniom",
        ETHANOL,
        &["-os", "gfnff", "-oniom", "2-4,8,9", "auto", "gfnff"],
        [2, 3, 4, 8],
        (&[2, 8], &[3, 4]),
        "move_radius=0.3",
    );
}
