//! The bias potentials of `mdmain` in every kind of run: {AFIR (`-ma`), a restraint (`-kp`),
//! an absolute-position bias (`-ubias` with `x()`, `y()`, `z()`)} x {an isolated cluster, a
//! periodic fixed cell, ONIOM (LJ:LJ)}. In every frame of an NVE run (written at every step)
//! the bias energy recomputed from the structure equals the potential energy less the one
//! without the bias (`md_properties.csv`); the conserved quantity of NVE with these static
//! biases is conserved; with the absolute-position bias no rigid motion is taken as conserved
//! (the surface sees the position in space). Metadynamics, which deposits Gaussians at the
//! accepted steps, gives the same run on one thread and on four, and the log names the terms
//! that make the conserved quantity drift.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::path::{Path, PathBuf};

use mopt_rs::cli::bias::parse_bias_option;
use mopt_rs::cli::mdmain::{config_from_cli, run_with, MdmainResult};
use mopt_rs::domain::{AtomId, MolecularSystem, SpinCharge};
use mopt_rs::io::xyz::{read_xyz_file, read_xyz_trajectory_file};
use mopt_rs::pes::registry::CalculatorRegistry;
use mopt_rs::pes::TermContext;

const CLUSTER: &str = "5\ncluster\nAr 0.000 0.000 0.000\nAr 3.800 0.000 0.000\n\
Ar 1.900 3.291 0.000\nAr 1.900 1.097 3.100\nAr -2.900 1.500 0.400\n";

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
    let dir = std::env::temp_dir().join(format!("mopt_bmd_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn biases(input: &MolecularSystem, a: usize, b: usize) -> Vec<(&'static str, String, Vec<String>)> {
    let p = input.position(AtomId::new(a - 1)).value();
    let goal = [p[0] + 0.3, p[1] - 0.2, p[2] + 0.1];
    vec![
        (
            "afir",
            "-ma".into(),
            vec!["20".into(), format!("{a}"), format!("{b}")],
        ),
        (
            "restraint",
            "-kp".into(),
            vec!["0.01".into(), "3.70".into(), format!("{a},{b}")],
        ),
        (
            "absolute",
            "-ubias".into(),
            vec![format!(
                "0.005 * ((x({a}) - {:.10})^2 + (y({a}) - {:.10})^2 + (z({a}) - {:.10})^2)",
                goal[0], goal[1], goal[2]
            )],
        ),
    ]
}

/// The columns of `md_properties.csv` by name.
fn properties(dir: &Path) -> Vec<std::collections::HashMap<String, f64>> {
    let text = std::fs::read_to_string(dir.join("md_properties.csv")).unwrap();
    let mut lines = text.lines();
    let names: Vec<String> = lines
        .next()
        .unwrap()
        .split(',')
        .map(|s| s.trim().to_string())
        .collect();
    lines
        .map(|l| {
            names
                .iter()
                .cloned()
                .zip(l.split(',').map(|v| v.trim().parse().unwrap_or(f64::NAN)))
                .collect()
        })
        .collect()
}

fn run(dir: &Path, args: &[String]) -> MdmainResult {
    let argv: Vec<&str> = args.iter().map(String::as_str).collect();
    let config = config_from_cli(&argv).unwrap();
    run_with(&config, dir, None, &CalculatorRegistry::native())
        .unwrap()
        .remove(0)
}

const STEPS: usize = 40;

fn context(name: &str, text: &str, extra: &[&str], atoms: (usize, usize)) {
    let dir = scratch(name);
    let file = dir.join("start.xyz");
    std::fs::write(&file, text).unwrap();
    let input = read_xyz_file(&file, SpinCharge::default(), None).unwrap();
    for (kind, flag, arguments) in biases(&input, atoms.0, atoms.1) {
        let cell = format!("{name} x {kind}");
        let mut args: Vec<String> = vec![
            file.to_string_lossy().into_owned(),
            "-os".into(),
            "lj".into(),
            "-time".into(),
            STEPS.to_string(),
            "-temp".into(),
            "30".into(),
            "-mt".into(),
            "nve".into(),
        ];
        args.extend(extra.iter().map(|s| s.to_string()));
        args.extend(["-md_opt", "write_every=1", "seed=5"].map(String::from));
        args.push(flag.clone());
        args.extend(arguments.iter().cloned());
        let out = run(&dir, &args);
        let rows = properties(&out.directory);
        let frames: Vec<MolecularSystem> =
            read_xyz_trajectory_file(&out.trajectory, SpinCharge::default(), None)
                .unwrap()
                .frames
                .into_iter()
                .map(|f| f.system)
                .collect();
        assert_eq!(rows.len(), frames.len(), "{cell}");
        assert!(rows.len() > STEPS, "{cell}: {} rows", rows.len());
        for (k, (row, sys)) in rows.iter().zip(&frames).enumerate() {
            let mut bias = 0.0;
            for spec in parse_bias_option(&flag, &arguments).unwrap() {
                let term = spec.build(sys).unwrap();
                bias += term
                    .evaluate(sys, &TermContext::at_iteration(0, sys), false)
                    .unwrap()
                    .energy
                    .value();
            }
            let reported = row["potential_energy"] - row["potential_energy_without_bias"];
            assert!(
                (bias - reported).abs() < 1e-8 * (1.0 + bias.abs()),
                "{cell} step {k}: bias {bias}, reported {reported}"
            );
        }
        // NVE on a static surface: the conserved quantity stays within the error of velocity
        // Verlet at 0.5 fs (about 1e-6 Hartree for the 32 atoms of the cell at 30 K) and does
        // not grow beyond it (a bias that changed with the time would move it by its work, of
        // the order of the bias energy, 1e-3 Hartree and more here).
        let first = rows[0]["conserved_quantity"];
        let drift = rows
            .iter()
            .map(|r| (r["conserved_quantity"] - first).abs())
            .fold(0.0_f64, f64::max);
        assert!(drift < 5e-6, "{cell}: drift {drift:.3e}");
        let log = std::fs::read_to_string(out.directory.join("mopt.log")).unwrap();
        assert!(!log.contains("The conserved quantity drifts"), "{cell}");
        if kind == "absolute" {
            assert!(
                log.contains("Conserved motions of the whole system: none"),
                "{cell}: rigid motions conserved on a surface that sees the position"
            );
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn bias_potentials_of_an_isolated_cluster() {
    context("isolated", CLUSTER, &[], (4, 1));
}

#[test]
fn bias_potentials_in_a_periodic_cell() {
    // The absolute bias on the atom at the corner of the cell: `x()` of `-ubias` in a cell
    // is the coordinate of the minimum image from the origin, and a goal far from the
    // corner would put the pull across half the cell.
    context("periodic", &fcc(), &[], (1, 32));
}

#[test]
fn bias_potentials_with_oniom() {
    context("oniom", CLUSTER, &["-oniom", "1-4", "none", "lj"], (4, 1));
}

/// Metadynamics deposits at the accepted steps: the same run on one thread and on four, and
/// the log names it among the terms that make the conserved quantity drift.
#[test]
fn metadynamics_gives_the_same_run_on_one_thread_and_on_four() {
    let dir = scratch("threads");
    let file = dir.join("start.xyz");
    std::fs::write(&file, CLUSTER).unwrap();
    let mut found = Vec::new();
    for threads in ["1", "4"] {
        let args: Vec<String> = [
            file.to_str().unwrap(),
            "-os",
            "lj",
            "-time",
            "30",
            "-temp",
            "60",
            "-core",
            threads,
            "-md_opt",
            "seed=9",
            "-metad",
            "bond",
            "2.0",
            "0.2",
            "1,4",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let out = run(&dir, &args);
        let log = std::fs::read_to_string(out.directory.join("mopt.log")).unwrap();
        assert!(
            log.contains("The conserved quantity drifts") && log.contains("metadynamics"),
            "the log does not name the metadynamics"
        );
        found.push(std::fs::read_to_string(out.directory.join("md_properties.csv")).unwrap());
        std::thread::sleep(std::time::Duration::from_millis(1100));
    }
    assert_eq!(found[0], found[1]);
    let _ = std::fs::remove_dir_all(&dir);
}
