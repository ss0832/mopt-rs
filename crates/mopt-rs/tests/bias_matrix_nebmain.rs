//! The bias potentials of `nebmain` in every kind of run: {AFIR (`-ma`), a restraint (`-kp`),
//! an absolute-position bias (`-ubias` with `x()`, `y()`, `z()`)} x {an isolated cluster, a
//! periodic fixed cell, ONIOM microiterations (LJ:LJ)}. For every image of every iteration
//! the bias energy recomputed from the structure in the files equals the energy with the
//! bias potentials less the one without them (`energy_plot.csv` and
//! `energy_plot_without_bias.csv`); with the absolute-position bias the fixed first image
//! keeps its place in the laboratory (the chain is neither centred nor turned). A bias with
//! a warm start (`-smp`) gives the same chain with one thread and with four; terms of the
//! time or of accumulated state are refused with the reason.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::path::{Path, PathBuf};

use mopt_rs::cli::bias::parse_bias_option;
use mopt_rs::cli::nebmain::{config_from_cli, run_with, NebmainResult};
use mopt_rs::domain::{AtomId, MolecularSystem, SpinCharge};
use mopt_rs::io::xyz::read_xyz_trajectory_file;
use mopt_rs::pes::registry::CalculatorRegistry;
use mopt_rs::pes::TermContext;

/// Five argon atoms at the two ends of the chain (atom 4 moves over the edge 1-2).
fn cluster() -> String {
    let fixed = "Ar 0.000 0.000 0.000\nAr 3.800 0.000 0.000\nAr 1.900 3.291 0.000\n";
    format!(
        "5\nA\n{fixed}Ar 1.900 1.097 3.100\nAr -2.900 1.500 0.400\n\
         5\nB\n{fixed}Ar 1.900 -2.400 2.300\nAr -2.900 1.500 0.400\n"
    )
}

/// The fcc argon crystal of 32 atoms with its last atom 0.4 Angstrom to either side.
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
    let mut text = String::new();
    for side in [0.4, -0.4] {
        text.push_str(&format!(
            "{}\nLattice=\"{l} 0.0 0.0 0.0 {l} 0.0 0.0 0.0 {l}\" pbc=\"T T T\"\n",
            rows.len(),
            l = 2.0 * a
        ));
        for (k, r) in rows.iter().enumerate() {
            let dx = if k + 1 == rows.len() { side } else { 0.0 };
            text.push_str(&format!("Ar {:.10} {:.10} {:.10}\n", r[0] + dx, r[1], r[2]));
        }
    }
    text
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("mopt_bneb_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// The biases on atoms `a` and `b`; the absolute one pulls `a` towards a point beside its
/// place in `input`: (name, flag, arguments).
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
            vec!["0.05".into(), "3.70".into(), format!("{a},{b}")],
        ),
        (
            "absolute",
            "-ubias".into(),
            vec![format!(
                "0.02 * ((x({a}) - {:.10})^2 + (y({a}) - {:.10})^2 + (z({a}) - {:.10})^2)",
                goal[0], goal[1], goal[2]
            )],
        ),
    ]
}

fn table(path: &Path) -> Vec<Vec<f64>> {
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(|l| l.split(',').map(|v| v.trim().parse().unwrap()).collect())
        .collect()
}

/// The chains of every iteration, in order.
fn chains(out: &NebmainResult, name: &str) -> Vec<Vec<MolecularSystem>> {
    (0..out.iterations)
        .map(|k| {
            let path = out
                .directory
                .join(format!("path_ITR_{k}_{name}"))
                .join(format!("{name}_path.xyz"));
            read_xyz_trajectory_file(&path, SpinCharge::default(), None)
                .unwrap()
                .frames
                .into_iter()
                .map(|f| f.system)
                .collect()
        })
        .collect()
}

fn run(dir: &Path, args: &[String]) -> NebmainResult {
    let argv: Vec<&str> = args.iter().map(String::as_str).collect();
    let config = config_from_cli(&argv).unwrap();
    run_with(&config, dir, None, &CalculatorRegistry::native()).unwrap()
}

fn context(name: &str, text: &str, extra: &[&str], atoms: (usize, usize)) {
    let dir = scratch(name);
    let file = dir.join("ends.xyz");
    std::fs::write(&file, text).unwrap();
    let ends: Vec<MolecularSystem> = read_xyz_trajectory_file(&file, SpinCharge::default(), None)
        .unwrap()
        .frames
        .into_iter()
        .map(|f| f.system)
        .collect();
    let input = &ends[0];
    // Every bias with the default chain, and the first one with the locally updated planes.
    let mut cells: Vec<(String, &'static str, String, Vec<String>)> =
        biases(input, atoms.0, atoms.1)
            .into_iter()
            .map(|(kind, flag, arguments)| (kind.to_string(), "", flag, arguments))
            .collect();
    if let Some((kind, _, flag, arguments)) = cells.first().cloned() {
        cells.push((format!("{kind} (lup)"), "-lup", flag, arguments));
    }
    for (kind, chain_method, flag, arguments) in cells {
        let cell = format!("{name} x {kind}");
        let mut args: Vec<String> = vec![
            file.to_string_lossy().into_owned(),
            "-os".into(),
            "lj".into(),
            "-ns".into(),
            "8".into(),
            "-fe".into(),
            "3".into(),
            "-neb_opt".into(),
            "path=linear".into(),
            "figures=off".into(),
        ];
        args.extend(extra.iter().map(|s| s.to_string()));
        if !chain_method.is_empty() {
            args.push(chain_method.to_string());
        }
        args.push(flag.clone());
        args.extend(arguments.iter().cloned());
        let out = run(&dir, &args);
        let biased = table(&out.directory.join("energy_plot.csv"));
        let bare = table(&out.directory.join("energy_plot_without_bias.csv"));
        let all = chains(&out, "ends");
        assert_eq!(biased.len(), all.len(), "{cell}");
        assert_eq!(bare.len(), all.len(), "{cell}");
        for (k, chain) in all.iter().enumerate() {
            assert_eq!(chain.len(), biased[k].len(), "{cell} iteration {k}");
            for (i, sys) in chain.iter().enumerate() {
                let mut bias = 0.0;
                for spec in parse_bias_option(&flag, &arguments).unwrap() {
                    let term = spec.build(sys).unwrap();
                    let e = term
                        .evaluate(sys, &TermContext::at_iteration(k, sys), false)
                        .unwrap();
                    bias += e.energy.value();
                }
                let reported = biased[k][i] - bare[k][i];
                // The files carry 12 decimals of Angstrom.
                assert!(
                    (bias - reported).abs() < 1e-8 * (1.0 + bias.abs()),
                    "{cell} iteration {k} image {i}: bias {bias}, reported {reported}"
                );
            }
            if kind == "absolute" {
                // The fixed first image is where the input put it: nothing centred or
                // turned the chain.
                let (p, q) = (chain[0].coords_flat_bohr(), input.coords_flat_bohr());
                // With ONIOM the environment (atom 5) is relaxed around the core (1-4).
                let moved = if extra.contains(&"-oniom") {
                    12
                } else {
                    p.nrows()
                };
                for c in 0..moved {
                    assert!(
                        (p[c] - q[c]).abs() < 2e-12,
                        "{cell} iteration {k}: component {c}"
                    );
                }
            }
        }
        if kind == "absolute" {
            let log = std::fs::read_to_string(out.directory.join("mopt.log")).unwrap();
            assert!(log.contains("Chain alignment disabled"), "{cell}");
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn bias_potentials_of_an_isolated_chain() {
    context("isolated", &cluster(), &[], (4, 1));
}

#[test]
fn bias_potentials_of_a_chain_in_a_periodic_cell() {
    context("periodic", &fcc(), &["-modelhess", "none"], (32, 1));
}

#[test]
fn bias_potentials_of_a_chain_with_oniom_microiterations() {
    context(
        "oniom",
        &cluster(),
        &["-oniom", "1-4", "none", "lj"],
        (4, 1),
    );
}

/// A bias with a warm start (the spacer model of `-smp`, auxiliary particles minimized at
/// every evaluation) gives the same chain whatever the threads: every image has its own copy.
#[test]
fn a_warm_started_bias_gives_the_same_chain_on_one_thread_and_on_four() {
    let dir = scratch("threads");
    let file = dir.join("ends.xyz");
    std::fs::write(&file, cluster()).unwrap();
    let mut found = Vec::new();
    for threads in ["1", "4"] {
        let args: Vec<String> = [
            file.to_str().unwrap(),
            "-os",
            "lj",
            "-ns",
            "5",
            "-core",
            threads,
            "-neb_opt",
            "path=linear",
            "figures=off",
            "-smp",
            "0.5",
            "3.0",
            "1.2",
            "2",
            "1-5",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let out = run(&dir, &args);
        found.push((
            std::fs::read_to_string(out.directory.join("energy_plot.csv")).unwrap(),
            std::fs::read_to_string(out.directory.join("energy_plot_without_bias.csv")).unwrap(),
        ));
        // Another second for the next run directory.
        std::thread::sleep(std::time::Duration::from_millis(1100));
    }
    assert_eq!(found[0], found[1]);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn terms_of_the_time_or_of_accumulated_state_are_refused() {
    let dir = scratch("refused");
    let file = dir.join("ends.xyz");
    std::fs::write(&file, cluster()).unwrap();
    let file = file.to_string_lossy().into_owned();
    for (extra, expected) in [
        (vec!["-metad", "bond", "1.0", "0.2", "1,2"], "keeps state"),
        (
            vec!["-nrp", "4", "8", "0.1", "0.1", "1", "1"],
            "simulated time",
        ),
        (vec!["-ubias", "0.001 * t * dist(1,2)"], "simulated time"),
    ] {
        let mut args = vec![file.as_str(), "-os", "lj", "-ns", "3"];
        args.extend(extra.iter().copied());
        let config = config_from_cli(&args).unwrap();
        let err = run_with(&config, &dir, None, &CalculatorRegistry::native())
            .unwrap_err()
            .to_string();
        assert!(err.contains(expected), "{extra:?}: {err}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}
