//! The bias potentials of `orientsearch` in every kind of run: {AFIR (`-ma`), a restraint
//! (`-kp`), an absolute-position bias (`-ubias` with `x()`, `y()`, `z()`)} x {a molecular
//! complex, a complex in a periodic cell, ONIOM}. For every minimum the bias energy recomputed
//! from its structure equals its biased less its bare energy (in memory and in
//! `orientsearch_minima.csv`); the minima are ranked and weighted by the biased energy, the
//! surface the samples were minimized on (as `confsearch` ranks); the absolute-position bias
//! leaves its atom at the point it pulls to.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used, clippy::panic)]

mod matrix_support;

use matrix_support::{bias_energy, biases, column, csv, read, scratch, write};
use mopt_rs::cli::orientsearch::{config_from_cli, run_with};
use mopt_rs::pes::registry::CalculatorRegistry;

const WATERS: &str = "6\nwaters\nO 0.0 0.0 0.1173\nH 0.0 0.7572 -0.4692\nH 0.0 -0.7572 -0.4692\n\
O 4.0 0.0 0.1173\nH 4.0 0.7572 -0.4692\nH 4.0 -0.7572 -0.4692\n";

const AR6: &str = "6\nLattice=\"20.0 0.0 0.0 0.0 20.0 0.0 0.0 0.0 20.0\" pbc=\"T T T\"\n\
Ar 5.0 5.0 5.0\nAr 8.8 5.0 5.0\nAr 6.9 8.3 5.0\n\
Ar 5.0 5.0 10.0\nAr 8.8 5.0 10.0\nAr 6.9 8.3 10.2\n";

fn context(name: &str, text: &str, calc: &[&str]) {
    let dir = scratch("osbias", name);
    let file = write(&dir, "start.xyz", text);
    let input = read(&file);
    for (kind, flag, arguments, goal) in biases(&input, 1, 4) {
        let cell = format!("{name} x {kind}");
        let mut args: Vec<String> = vec![file.to_string_lossy().into_owned()];
        args.extend(calc.iter().map(|s| s.to_string()));
        for s in [
            "-nsample",
            "3",
            "-part",
            "1-3",
            "-orient_opt",
            "seed=5",
            "criteria=normal",
        ] {
            args.push(s.to_string());
        }
        args.push(flag.clone());
        args.extend(arguments.iter().cloned());
        let argv: Vec<&str> = args.iter().map(String::as_str).collect();
        let config = config_from_cli(&argv).unwrap();
        let out = run_with(&config, &dir, None, &CalculatorRegistry::native())
            .unwrap_or_else(|e| panic!("{cell}: {e}"));
        assert!(!out.minima.is_empty(), "{cell}");
        let mut last = f64::NEG_INFINITY;
        for m in &out.minima {
            assert!(
                m.energy >= last,
                "{cell}: ranking {} after {last}",
                m.energy
            );
            last = m.energy;
            let b = bias_energy(&flag, &arguments, &m.system);
            assert!(
                (b - (m.energy - m.bare_energy)).abs() < 1e-9 * (1.0 + b.abs()),
                "{cell}: bias {b}, biased - bare {}",
                m.energy - m.bare_energy
            );
            if let Some(goal) = goal {
                matrix_support::at_point(&cell, &m.system, 1, goal, 5e-3);
            }
        }
        // The table says the same.
        let (header, rows) = csv(&out.directory.join("orientsearch_minima.csv"));
        let (e, e0) = (
            column(&header, "energy_hartree"),
            column(&header, "bare_energy_hartree"),
        );
        assert_eq!(rows.len(), out.minima.len(), "{cell}");
        for (row, m) in rows.iter().zip(&out.minima) {
            let (a, b): (f64, f64) = (row[e].parse().unwrap(), row[e0].trim().parse().unwrap());
            assert!(
                (a - m.energy).abs() < 1e-9 && (b - m.bare_energy).abs() < 1e-9,
                "{cell}"
            );
        }
        // The weights are those of the biased energies.
        if out.minima.len() > 1 {
            let kt = 298.15 * 3.166_811_563e-6;
            let ratio = out.minima[1].weight.unwrap() / out.minima[0].weight.unwrap();
            let expected = (-(out.minima[1].energy - out.minima[0].energy) / kt).exp();
            assert!(
                (ratio - expected).abs() < 1e-4 * (1.0 + expected),
                "{cell}: {ratio} vs {expected}"
            );
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn biases_of_a_molecular_complex() {
    context("molecule", WATERS, &["-os", "gfnff"]);
}

#[test]
fn biases_of_a_complex_in_a_periodic_cell() {
    context("periodic", AR6, &["-os", "lj"]);
}

#[test]
fn biases_of_a_complex_with_oniom() {
    context(
        "oniom",
        WATERS,
        &["-os", "gfnff", "-oniom", "1-3", "none", "gfnff"],
    );
}
