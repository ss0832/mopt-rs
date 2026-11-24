//! The constraints of `orientsearch` in every kind of run: {frozen atoms, a distance between
//! two parts, the distance of the centroids of the parts (`fbond`), an angle across the parts,
//! a rigid group across the parts (`-gfix`), a user-defined function} x {a molecular complex,
//! a complex in a periodic cell, ONIOM}. Every sample is placed at random, but every structure
//! the command writes or returns (the start of every sample, its optimized structure, the
//! minima) holds the constraints at the values of the input; the minima in memory to the
//! rounding floor, and a part with frozen atoms stays where it is in the input.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used, clippy::panic)]

mod matrix_support;

use matrix_support::{check, kinds, read, scratch, write, FROM_FILES, FROZEN_IN_FILES};
use mopt_rs::cli::orientsearch::{config_from_cli, run_with};
use mopt_rs::pes::registry::CalculatorRegistry;

/// Two water molecules 4 Angstrom apart.
const WATERS: &str = "6\nwaters\nO 0.0 0.0 0.1173\nH 0.0 0.7572 -0.4692\nH 0.0 -0.7572 -0.4692\n\
O 4.0 0.0 0.1173\nH 4.0 0.7572 -0.4692\nH 4.0 -0.7572 -0.4692\n";

/// Two argon triangles in a periodic box.
const AR6: &str = "6\nLattice=\"20.0 0.0 0.0 0.0 20.0 0.0 0.0 0.0 20.0\" pbc=\"T T T\"\n\
Ar 5.0 5.0 5.0\nAr 8.8 5.0 5.0\nAr 6.9 8.3 5.0\n\
Ar 5.0 5.0 10.0\nAr 8.8 5.0 10.0\nAr 6.9 8.3 10.2\n";

/// The bound on a residual of a structure in memory: the rounding floor of the restore.
/// (The guarantee every hard constraint is held to: `pes::constraint::GUARANTEE`.)
const IN_MEMORY: f64 = 1e-10;

fn context(name: &str, text: &str, calc: &[&str]) {
    let dir = scratch("osmatrix", name);
    let file = write(&dir, "start.xyz", text);
    let input = read(&file);
    for (kind, options) in kinds([1, 2, 4, 5], (&[1, 2, 3], &[4, 5, 6])) {
        let cell = format!("{name} x {kind}");
        let mut args: Vec<String> = vec![file.to_string_lossy().into_owned()];
        args.extend(calc.iter().map(|s| s.to_string()));
        for s in [
            "-nsample",
            "2",
            "-part",
            "1-3",
            "-orient_opt",
            "seed=5",
            "criteria=normal",
            "interaction=off",
        ] {
            args.push(s.to_string());
        }
        args.extend(options.iter().cloned());
        let argv: Vec<&str> = args.iter().map(String::as_str).collect();
        let config = config_from_cli(&argv).unwrap();
        if name == "oniom" && kind == "fbond" {
            // The two centroids name every atom: the core is the whole complex, and the
            // microiterations have no environment to relax (refused, with the reason).
            let err = run_with(&config, &dir, None, &CalculatorRegistry::native())
                .unwrap_err()
                .to_string();
            assert!(err.contains("no environment to relax"), "{cell}: {err}");
            continue;
        }
        let out = run_with(&config, &dir, None, &CalculatorRegistry::native())
            .unwrap_or_else(|e| panic!("{cell}: {e}"));
        let constraints = &config.optmain.settings.constraints;
        assert!(!out.minima.is_empty(), "{cell}");
        // The minima in memory.
        let minima: Vec<_> = out.minima.iter().map(|m| m.system.clone()).collect();
        check(&cell, &minima, constraints, &input, IN_MEMORY, 0.0, false);
        // Every file: the starts, the optimized samples, the minima.
        let mut written = Vec::new();
        for k in 0..out.samples.len() {
            written.push(read(&out.directory.join(format!("start_Sample_{k}.xyz"))));
            let opt = out
                .directory
                .join(format!("start_Sample_{k}_OPT"))
                .join(format!("start_Sample_{k}_optimized.xyz"));
            if opt.is_file() {
                written.push(read(&opt));
            }
        }
        for m in &out.minima {
            written.push(read(&m.file));
        }
        check(
            &cell,
            &written,
            constraints,
            &input,
            FROM_FILES,
            FROZEN_IN_FILES,
            false,
        );
        let log = std::fs::read_to_string(out.directory.join("mopt.log")).unwrap();
        assert!(log.contains("Constraints: resolved at the input"), "{cell}");
        if kind == "frozen" {
            // The part with the frozen atom is not placed: every start has it where it is in
            // the input (the optimization then relaxes its other atoms).
            assert!(log.contains("Part(s) 2 hold atoms"), "{cell}: {log}");
            for k in 0..out.samples.len() {
                let s = read(&out.directory.join(format!("start_Sample_{k}.xyz")));
                for a in 3..6 {
                    let id = mopt_rs::domain::AtomId::new(a);
                    let (p, q) = (s.position(id).value(), input.position(id).value());
                    for c in 0..3 {
                        assert!(
                            (p[c] - q[c]).abs() <= FROZEN_IN_FILES,
                            "{cell}: {p:?} {q:?}"
                        );
                    }
                }
            }
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn constraints_of_a_molecular_complex() {
    context("molecule", WATERS, &["-os", "gfnff"]);
}

#[test]
fn constraints_of_a_complex_in_a_periodic_cell() {
    context("periodic", AR6, &["-os", "lj"]);
}

#[test]
fn constraints_of_a_complex_with_oniom() {
    context(
        "oniom",
        WATERS,
        &["-os", "gfnff", "-oniom", "1-3", "none", "gfnff"],
    );
}
