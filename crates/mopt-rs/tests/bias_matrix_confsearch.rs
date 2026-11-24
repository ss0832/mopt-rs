//! The bias potentials of `confsearch` in every kind of run: {AFIR (`-ma`), a restraint
//! (`-kp`), an absolute-position bias (`-ubias` with `x()`, `y()`, `z()`)} x {a molecule, a
//! cluster in a periodic cell, ONIOM microiterations}. For every conformer written, the bias
//! energy recomputed from its structure equals the biased less the bare energy of
//! `conformers.csv`; the conformers are ranked by the biased energy; the absolute-position bias
//! leaves its atom at the point it pulls to (the optimizations do not remove rigid-body
//! motion). A bias with state (metadynamics) gives the same conformers with 1 and 4 threads.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used, clippy::panic)]

mod matrix_support;

use matrix_support::{bias_energy, biases, column, csv, frames, read, scratch, write};
use mopt_rs::cli::confsearch::{config_from_cli, run_with, ConfsearchResult};
use mopt_rs::pes::registry::CalculatorRegistry;

const BUTANE: &str = "14\nn-butane all-anti\nC 0.000000 0.000000 0.000000\n\
C 1.268427 0.855565 0.000000\nC 2.536855 0.000000 0.000000\nC 3.805282 0.855565 0.000000\n\
H -0.875864 0.648816 0.000000\nH -0.013866 -0.629150 0.889988\nH -0.013866 -0.629150 -0.889988\n\
H 1.268427 1.484886 0.889975\nH 1.268427 1.484886 -0.889975\nH 2.536855 -0.629321 -0.889975\n\
H 2.536855 -0.629321 0.889975\nH 4.681147 0.206750 0.000000\nH 3.819149 1.484715 0.889988\n\
H 3.819149 1.484715 -0.889988\n";

const AR7: &str = "7\nLattice=\"30.0 0.0 0.0 0.0 30.0 0.0 0.0 0.0 30.0\" pbc=\"T T T\"\n\
Ar 3.2175 0.0000 0.0000\nAr 0.9943 3.0601 0.0495\nAr -2.6030 1.8912 0.1155\n\
Ar -2.6030 -1.8912 0.1815\nAr 0.9943 -3.0601 0.2475\nAr 0.0000 0.0000 3.4650\n\
Ar 0.3300 0.0000 -3.3000\n";

const ETHANOL: &str = "9\nethanol\nC 0.000 0.000 0.000\nC 1.520 0.020 0.000\n\
O 2.020 1.330 0.050\nH 2.990 1.330 0.000\nH -0.390 1.030 0.000\nH -0.390 -0.510 0.890\n\
H -0.390 -0.510 -0.890\nH 1.910 -0.510 0.890\nH 1.910 -0.510 -0.890\n";

fn search(file: &str, calc: &[&str], extra: &[String], dir: &std::path::Path) -> ConfsearchResult {
    let mut args: Vec<String> = vec![file.to_string()];
    args.extend(calc.iter().map(|s| s.to_string()));
    for s in [
        "-ms",
        "3",
        "-cs_opt",
        "method=kick",
        "move_radius=0.4",
        "seed=3",
        "verify=off",
        "nudge=off",
        "mode=minima",
    ] {
        args.push(s.to_string());
    }
    args.extend(extra.iter().cloned());
    let argv: Vec<&str> = args.iter().map(String::as_str).collect();
    let config = config_from_cli(&argv).unwrap();
    run_with(&config, dir, None, &CalculatorRegistry::native()).unwrap()
}

fn context(name: &str, text: &str, calc: &[&str], atoms: (usize, usize)) {
    let dir = scratch("csbias", name);
    let file = write(&dir, "start.xyz", text);
    let input = read(&file);
    let file = file.to_string_lossy().into_owned();
    for (kind, flag, arguments, goal) in biases(&input, atoms.0, atoms.1) {
        let cell = format!("{name} x {kind}");
        let mut extra = vec![flag.clone()];
        extra.extend(arguments.iter().cloned());
        let out = search(&file, calc, &extra, &dir);
        let written = frames(&out.conformers);
        let (header, rows) = csv(&out.directory.join("conformers.csv"));
        assert_eq!(rows.len(), written.len(), "{cell}");
        let (eq, biased, bare) = (
            column(&header, "eq"),
            column(&header, "energy_hartree"),
            column(&header, "bare_energy_hartree"),
        );
        let mut last = f64::NEG_INFINITY;
        for (row, sys) in rows.iter().zip(&written) {
            let (e, e0): (f64, f64) = (row[biased].parse().unwrap(), row[bare].parse().unwrap());
            // Ranked by the biased energy.
            assert!(e >= last, "{cell}: ranking {e} after {last}");
            last = e;
            let b = bias_energy(&flag, &arguments, sys);
            assert!(
                (b - (e - e0)).abs() < 1e-7 * (1.0 + b.abs()),
                "{cell} EQ{}: bias {b}, biased - bare {}",
                row[eq],
                e - e0
            );
            if let Some(goal) = goal {
                // The optimizations keep the lab frame: the atom is where the bias pulls it.
                matrix_support::at_point(&cell, sys, atoms.0, goal, 5e-3);
            }
        }
        assert!(
            (out.energies[0] - rows[0][biased].parse::<f64>().unwrap()).abs() < 1e-9,
            "{cell}"
        );
        let text = std::fs::read_to_string(out.directory.join("ensemble.txt")).unwrap();
        assert!(
            text.contains("ranked by: the energy with the bias"),
            "{cell}"
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn biases_of_conformers_of_a_molecule() {
    context("molecule", BUTANE, &["-os", "gfnff"], (1, 4));
}

#[test]
fn biases_of_conformers_in_a_periodic_cell() {
    context(
        "periodic",
        AR7,
        &["-os", "lj", "-ns", "1000", "-lcc"],
        (1, 4),
    );
}

#[test]
fn biases_of_conformers_with_oniom_microiterations() {
    context(
        "oniom",
        ETHANOL,
        &["-os", "gfnff", "-oniom", "2-4,8,9", "auto", "gfnff"],
        (3, 4),
    );
}

#[test]
fn a_bias_with_state_gives_the_same_conformers_with_any_number_of_threads() {
    let dir = scratch("csbias", "threads");
    let file = write(&dir, "start.xyz", BUTANE)
        .to_string_lossy()
        .into_owned();
    // Small deposits, so that the optimizations still converge while the state grows.
    let metad: Vec<String> = ["-metad", "bond", "0.01", "0.05", "1,4"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let with = |threads: &str| {
        let mut extra = metad.clone();
        extra.extend(["-core".to_string(), threads.to_string()]);
        search(&file, &["-os", "gfnff"], &extra, &dir)
    };
    let (one, four) = (with("1"), with("4"));
    assert_eq!(one.energies, four.energies);
    let (a, b) = (frames(&one.conformers), frames(&four.conformers));
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(&b) {
        assert_eq!(x.coords_flat_bohr(), y.coords_flat_bohr());
    }
    let _ = std::fs::remove_dir_all(&dir);
}
