//! The bias potentials of `optmain` in every kind of run: {AFIR (`-ma`), a restraint
//! (`-kp`), an absolute-position bias (`-ubias` with `x()`, `y()`, `z()`)} x {a molecule, a
//! periodic fixed cell, a periodic variable cell (`-vc`), ONIOM microiterations}. In every
//! cell the bias energy recomputed from the final structure equals the biased energy less
//! the energy the run reports; where the surface is the calculator's, the forces of the
//! calculator and of the bias balance at the optimum; the absolute-position bias leaves its
//! atom at the point it pulls to (nothing else exerts a net force).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::path::PathBuf;

use mopt_rs::cli::bias::parse_bias_option;
use mopt_rs::cli::optmain::{config_from_cli, run_with};
use mopt_rs::domain::{AtomId, EvalRequest, MolecularSystem, SpinCharge};
use mopt_rs::io::xyz::read_xyz_file;
use mopt_rs::pes::registry::{CalculatorOptions, CalculatorRegistry};
use mopt_rs::pes::TermContext;

const STAGGERED: &str = "8\nethane\nC 0.000 0.000 0.000\nC 1.550 0.020 -0.010\n\
H -0.380 1.040 0.020\nH -0.400 -0.500 0.880\nH -0.370 -0.520 -0.900\n\
H 1.930 -1.020 0.030\nH 1.910 0.500 0.900\nH 1.940 0.530 -0.870\n";

const ETHANOL: &str = "9\nethanol\nC 0.000 0.000 0.000\nC 1.520 0.020 0.000\n\
O 2.020 1.330 0.050\nH 2.990 1.330 0.000\nH -0.390 1.030 0.000\nH -0.390 -0.510 0.890\n\
H -0.390 -0.510 -0.890\nH 1.910 -0.510 0.890\nH 1.910 -0.510 -0.890\n";

/// A 2 x 2 x 2 fcc argon crystal (32 atoms), every atom shifted a little.
fn fcc() -> String {
    let a = 5.40;
    let mut rows = Vec::new();
    for i in 0..2 {
        for j in 0..2 {
            for k in 0..2 {
                for f in [
                    [0.0, 0.0, 0.0],
                    [0.5, 0.5, 0.0],
                    [0.5, 0.0, 0.5],
                    [0.0, 0.5, 0.5],
                ] {
                    let n = rows.len() as f64;
                    let shift = [
                        0.07 * (1.3 * n).sin(),
                        0.06 * (0.7 * n).cos(),
                        0.05 * (2.1 * n).sin(),
                    ];
                    let cell = [f64::from(i), f64::from(j), f64::from(k)];
                    rows.push(std::array::from_fn::<f64, 3, _>(|c| {
                        a * (cell[c] + f[c]) + shift[c]
                    }));
                }
            }
        }
    }
    let mut text = format!(
        "{}\nLattice=\"{l} 0.0 0.0 0.0 {l} 0.0 0.0 0.0 {l}\" pbc=\"T T T\"\n",
        rows.len(),
        l = 2.0 * a
    );
    for r in &rows {
        text.push_str(&format!("Ar {:.10} {:.10} {:.10}\n", r[0], r[1], r[2]));
    }
    text
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("mopt_bmatrix_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// The bias options for the atoms `a`, `b`, the absolute bias on `a` pulling it 0.5 Bohr
/// away from where it is in `input`: (name, flag, arguments).
fn biases(input: &MolecularSystem, a: usize, b: usize) -> Vec<(&'static str, String, Vec<String>)> {
    let p = input.position(AtomId::new(a - 1)).value();
    let goal = [p[0] + 0.5, p[1] - 0.3, p[2] + 0.2];
    vec![
        (
            "afir",
            "-ma".into(),
            vec!["50".into(), format!("{a}"), format!("{b}")],
        ),
        (
            "restraint",
            "-kp".into(),
            vec!["0.05".into(), "1.70".into(), format!("{a},{b}")],
        ),
        (
            "absolute",
            "-ubias".into(),
            vec![format!(
                "0.2 * ((x({a}) - {:.10})^2 + (y({a}) - {:.10})^2 + (z({a}) - {:.10})^2)",
                goal[0], goal[1], goal[2]
            )],
        ),
    ]
}

/// One context: the structure in `text`, the calculator `calc`, and `extra` options.
fn context(name: &str, text: &str, calc: &str, extra: &[&str], atoms: (usize, usize)) {
    let dir = scratch(name);
    let file = dir.join("start.xyz");
    std::fs::write(&file, text).unwrap();
    let input = read_xyz_file(&file, SpinCharge::default(), None).unwrap();
    let registry = CalculatorRegistry::native();
    for (kind, flag, arguments) in biases(&input, atoms.0, atoms.1) {
        let cell = format!("{name} x {kind}");
        let mut args: Vec<&str> = vec![file.to_str().unwrap(), "-os", calc, "-opt", "rsirfo_bfgs"];
        args.extend_from_slice(extra);
        args.push(&flag);
        args.extend(arguments.iter().map(String::as_str));
        let config = config_from_cli(&args).unwrap();
        let out = run_with(&config, &dir, None, &registry).unwrap().remove(0);
        assert!(out.converged, "{cell}");
        let summary = out.summary.as_ref().unwrap();
        let end = &summary.system;
        // The bias energy of the final structure.
        let mut bias = 0.0;
        let mut bias_gradient = faer::Col::<f64>::zeros(end.dof());
        let ctx = TermContext::at_iteration(0, end);
        for spec in parse_bias_option(&flag, &arguments).unwrap() {
            let term = spec.build(end).unwrap();
            let e = term.evaluate(end, &ctx, false).unwrap();
            bias += e.energy.value();
            bias_gradient += &e.gradient_bohr;
        }
        let reported = summary.biased_energy.value() - summary.energy.value();
        assert!(
            (bias - reported).abs() < 1e-9 * (1.0 + bias.abs()),
            "{cell}: bias {bias}, reported {reported}"
        );
        // Force balance where the run's surface is the calculator's (with ONIOM the core
        // moves on the relaxed surface).
        if !extra.contains(&"-oniom") {
            let calculator = registry
                .create(calc, &CalculatorOptions::default(), end)
                .unwrap();
            let g = calculator
                .evaluate(end, &EvalRequest::ENERGY_GRAD)
                .unwrap()
                .gradient_bohr;
            let total = &g + &bias_gradient;
            let largest = total.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
            assert!(largest < 1e-3, "{cell}: largest force {largest}");
        }
        if kind == "absolute" {
            // Nothing else exerts a net force: the atom is where the bias pulls it.
            let p = input.position(AtomId::new(atoms.0 - 1)).value();
            let goal = [p[0] + 0.5, p[1] - 0.3, p[2] + 0.2];
            let q = end.position(AtomId::new(atoms.0 - 1)).value();
            for c in 0..3 {
                assert!(
                    (q[c] - goal[c]).abs() < 5e-3,
                    "{cell}: atom at {q:?}, pulled to {goal:?} (Bohr)"
                );
            }
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn biases_on_a_molecule() {
    context("molecule", STAGGERED, "gfnff", &[], (1, 2));
}

#[test]
fn biases_in_a_fixed_periodic_cell() {
    context("fixed_cell", &fcc(), "lj", &["-ns", "400"], (1, 2));
}

#[test]
fn biases_in_a_variable_cell() {
    context(
        "variable_cell",
        &fcc(),
        "lj",
        &["-vc", "-ns", "400"],
        (1, 2),
    );
}

#[test]
fn biases_with_oniom_microiterations() {
    context(
        "oniom",
        ETHANOL,
        "gfnff",
        &["-oniom", "2-4,8,9", "auto", "gfnff"],
        (3, 4),
    );
}
