//! The bias potentials of `relaxedscan` in every kind of run: {AFIR (`-ma`), a restraint
//! (`-kp`), an absolute-position bias (`-ubias` with `x()`, `y()`, `z()`)} x {a molecule, a
//! periodic fixed cell, ONIOM}. At every point the bias energy recomputed from its structure
//! equals the biased less the bare energy, and `scan_points.csv` names both relative energies
//! (each against the lowest point of its own surface). A bias with state (metadynamics) is
//! refused with the reason: it would start again at every point.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used, clippy::panic)]

mod matrix_support;

use matrix_support::{bias_energy, biases, column, csv, read, scratch, write};
use mopt_rs::cli::relaxedscan::{config_from_cli, run_with};
use mopt_rs::domain::Units;
use mopt_rs::pes::registry::CalculatorRegistry;

const WATER: &str = "3\nwater\nO 0.000 0.000 0.117\nH 0.000 0.757 -0.469\nH 0.000 -0.757 -0.469\n";

const AR7: &str = "7\nLattice=\"30.0 0.0 0.0 0.0 30.0 0.0 0.0 0.0 30.0\" pbc=\"T T T\"\n\
Ar 1.9500 0.0000 0.0000\nAr 0.6026 1.8546 0.0300\nAr -1.5776 1.1462 0.0700\n\
Ar -1.5776 -1.1462 0.1100\nAr 0.6026 -1.8546 0.1500\nAr 0.0000 0.0000 2.1000\n\
Ar 0.2000 0.0000 -2.0000\n";

const ETHANOL: &str = "9\nethanol\nC 0.000 0.000 0.000\nC 1.520 0.020 0.000\n\
O 2.020 1.330 0.050\nH 2.990 1.330 0.000\nH -0.390 1.030 0.000\nH -0.390 -0.510 0.890\n\
H -0.390 -0.510 -0.890\nH 1.910 -0.510 0.890\nH 1.910 -0.510 -0.890\n";

fn args(file: &str, calc: &[&str], scan: [&str; 3], extra: &[String]) -> Vec<String> {
    let mut a: Vec<String> = vec![file.to_string()];
    a.extend(calc.iter().map(|s| s.to_string()));
    a.push("-scan".into());
    a.extend(scan.iter().map(|s| s.to_string()));
    for s in [
        "-nsample",
        "3",
        "-scan_opt",
        "figures=off",
        "html=off",
        "candidates=off",
    ] {
        a.push(s.to_string());
    }
    a.extend(extra.iter().cloned());
    a
}

fn context(name: &str, text: &str, calc: &[&str], scan: [&str; 3], atoms: (usize, usize)) {
    let dir = scratch("rsbias", name);
    let file = write(&dir, "start.xyz", text);
    let input = read(&file);
    let file = file.to_string_lossy().into_owned();
    for (kind, flag, arguments, _) in biases(&input, atoms.0, atoms.1) {
        let cell = format!("{name} x {kind}");
        let mut extra = vec![flag.clone()];
        extra.extend(arguments.iter().cloned());
        let argv = args(&file, calc, scan, &extra);
        let argv: Vec<&str> = argv.iter().map(String::as_str).collect();
        let mut config = config_from_cli(&argv).unwrap();
        config.restart_dir = Some(dir.join(kind));
        let out = run_with(&config, &dir, None, &CalculatorRegistry::native())
            .unwrap_or_else(|e| panic!("{cell}: {e}"));
        for p in out.points.iter().filter(|p| !p.skipped) {
            let s = read(&dir.join(kind).join(&p.file));
            let b = bias_energy(&flag, &arguments, &s);
            assert!(
                (b - (p.biased - p.energy)).abs() < 1e-7 * (1.0 + b.abs()),
                "{cell} point {}: bias {b}, biased - bare {}",
                p.point,
                p.biased - p.energy
            );
        }
        let (header, rows) = csv(&dir.join(kind).join("scan_points.csv"));
        let (e, eb, rel, relb) = (
            column(&header, "energy_hartree"),
            column(&header, "biased_energy_hartree"),
            column(&header, "relative_bare_kcal_mol"),
            column(&header, "relative_biased_kcal_mol"),
        );
        let value = |row: &Vec<String>, k: usize| row[k].parse::<f64>().unwrap();
        let low = rows
            .iter()
            .map(|r| value(r, e))
            .fold(f64::INFINITY, f64::min);
        let lowb = rows
            .iter()
            .map(|r| value(r, eb))
            .fold(f64::INFINITY, f64::min);
        for row in &rows {
            let kcal = Units::HARTREE_TO_KCAL_MOL;
            assert!(
                (value(row, rel) - (value(row, e) - low) * kcal).abs() < 1e-5,
                "{cell}"
            );
            assert!(
                (value(row, relb) - (value(row, eb) - lowb) * kcal).abs() < 1e-5,
                "{cell}"
            );
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn biases_in_a_scan_of_a_molecule() {
    context(
        "molecule",
        WATER,
        &["-os", "gfnff"],
        ["bond", "1,2", "0.90,1.10"],
        (3, 1),
    );
}

#[test]
fn biases_in_a_scan_in_a_periodic_cell() {
    context(
        "periodic",
        AR7,
        &["-os", "lj"],
        ["bond", "6,7", "4.00,4.20"],
        (1, 2),
    );
}

#[test]
fn biases_in_a_scan_with_oniom() {
    context(
        "oniom",
        ETHANOL,
        &["-os", "gfnff", "-oniom", "2-4,8,9", "auto", "gfnff"],
        ["bond", "2,8", "1.08,1.14"],
        (3, 4),
    );
}

#[test]
fn a_bias_with_state_is_refused_with_the_reason() {
    let dir = scratch("rsbias", "metad");
    let file = write(&dir, "start.xyz", WATER)
        .to_string_lossy()
        .into_owned();
    let extra: Vec<String> = ["-metad", "bond", "2.0", "0.2", "1,3"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let argv = args(
        &file,
        &["-os", "gfnff"],
        ["bond", "1,2", "0.90,1.10"],
        &extra,
    );
    let argv: Vec<&str> = argv.iter().map(String::as_str).collect();
    let mut config = config_from_cli(&argv).unwrap();
    config.restart_dir = Some(dir.join("scan"));
    let err = run_with(&config, &dir, None, &CalculatorRegistry::native())
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("relaxedscan") && err.contains("keeps a state") && err.contains("start again"),
        "{err}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}
