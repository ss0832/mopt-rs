//! The bias potentials of `ieipmain` (iEIP) in every kind of run: {AFIR (`-ma`), a restraint
//! (`-kp`), an absolute-position bias (`-ubias` with `x()`, `y()`, `z()`)} x {an isolated
//! cluster, a periodic fixed cell, ONIOM microiterations (LJ:LJ)}. For both images of every
//! iteration the bias energy recomputed from the structures of the trajectory equals the
//! energy with the bias potentials less the one without them (`ieip_energies.csv`), and so
//! does the bias at the estimate of the transition state (its file and the log); with the
//! absolute-position bias the structures are not laid onto each other. A warm-started bias
//! gives the same search on one thread and on four.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::path::{Path, PathBuf};

use mopt_rs::cli::bias::parse_bias_option;
use mopt_rs::cli::ieipmain::{config_from_cli, run_with, IeipmainResult};
use mopt_rs::domain::{AtomId, MolecularSystem, SpinCharge};
use mopt_rs::io::xyz::{read_xyz_file, read_xyz_trajectory_file};
use mopt_rs::pes::registry::CalculatorRegistry;
use mopt_rs::pes::TermContext;

const FIXED: &str = "Ar 0.000 0.000 0.000\nAr 3.800 0.000 0.000\nAr 1.900 3.291 0.000\n";

fn cluster(second: bool) -> String {
    let moving = if second {
        "Ar 1.900 -2.400 2.300\n"
    } else {
        "Ar 1.900 1.097 3.100\n"
    };
    format!("5\nend\n{FIXED}{moving}Ar -2.900 1.500 0.400\n")
}

fn fcc(second: bool) -> String {
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
    let side = if second { -0.4 } else { 0.4 };
    let mut text = format!(
        "{}\nLattice=\"{l} 0.0 0.0 0.0 {l} 0.0 0.0 0.0 {l}\" pbc=\"T T T\"\n",
        rows.len(),
        l = 2.0 * a
    );
    for (k, r) in rows.iter().enumerate() {
        let dx = if k + 1 == rows.len() { side } else { 0.0 };
        text.push_str(&format!("Ar {:.10} {:.10} {:.10}\n", r[0] + dx, r[1], r[2]));
    }
    text
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("mopt_bieip_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

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

fn bias_at(flag: &str, arguments: &[String], sys: &MolecularSystem) -> f64 {
    parse_bias_option(flag, arguments)
        .unwrap()
        .iter()
        .map(|spec| {
            spec.build(sys)
                .unwrap()
                .evaluate(sys, &TermContext::at_iteration(0, sys), false)
                .unwrap()
                .energy
                .value()
        })
        .sum()
}

fn rows(path: &Path) -> Vec<Vec<f64>> {
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .skip(1)
        .map(|l| l.split(',').map(|v| v.trim().parse().unwrap()).collect())
        .collect()
}

fn run(dir: &Path, args: &[String]) -> IeipmainResult {
    let argv: Vec<&str> = args.iter().map(String::as_str).collect();
    let config = config_from_cli(&argv).unwrap();
    run_with(&config, dir, None, &CalculatorRegistry::native()).unwrap()
}

fn context(name: &str, first: &str, second: &str, extra: &[&str], atoms: (usize, usize)) {
    let dir = scratch(name);
    let (a, b) = (dir.join("a.xyz"), dir.join("b.xyz"));
    std::fs::write(&a, first).unwrap();
    std::fs::write(&b, second).unwrap();
    let input = read_xyz_file(&a, SpinCharge::default(), None).unwrap();
    for (kind, flag, arguments) in biases(&input, atoms.0, atoms.1) {
        let cell = format!("{name} x {kind}");
        let mut args: Vec<String> = vec![
            a.to_string_lossy().into_owned(),
            b.to_string_lossy().into_owned(),
            "-os".into(),
            "lj".into(),
            "-ns".into(),
            "12".into(),
            "-ieip_opt".into(),
            "figures=off".into(),
        ];
        args.extend(extra.iter().map(|s| s.to_string()));
        args.push(flag.clone());
        args.extend(arguments.iter().cloned());
        let out = run(&dir, &args);
        // The rows of the iterations after the first are the images of the frames of the
        // trajectory: the first image in order, then the second backwards.
        let table = rows(&out.directory.join("ieip_energies.csv"));
        let traj: Vec<MolecularSystem> = read_xyz_trajectory_file(
            out.directory.join("a_traj.xyz"),
            SpinCharge::default(),
            input.cell.as_ref(),
        )
        .unwrap()
        .frames
        .into_iter()
        .map(|f| f.system)
        .collect();
        let steps = traj.len() / 2;
        assert!(
            steps >= 2 && table.len() >= steps,
            "{cell}: {} rows, {} frames",
            table.len(),
            traj.len()
        );
        for k in 1..steps {
            let (one, two) = (&traj[k - 1], &traj[traj.len() - k]);
            for (image, sys) in [(0, one), (1, two)] {
                let bias = bias_at(&flag, &arguments, sys);
                let reported = table[k][3 + image] - table[k][1 + image];
                assert!(
                    (bias - reported).abs() < 1e-8 * (1.0 + bias.abs()),
                    "{cell} iteration {k} image {}: bias {bias}, reported {reported}",
                    image + 1
                );
            }
        }
        // The estimate: its file has the energy with the bias potentials, the log both.
        let estimate = &out.transition_states[0];
        let sys = read_xyz_file(estimate, SpinCharge::default(), input.cell.as_ref()).unwrap();
        let log = std::fs::read_to_string(out.directory.join("mopt.log")).unwrap();
        let line = log
            .lines()
            .find(|l| l.contains("Energy at the estimate of the transition state"))
            .unwrap();
        let numbers: Vec<f64> = line
            .split(|c: char| c == ':' || c == '(' || c == ')' || c.is_whitespace())
            .filter_map(|t| t.parse().ok())
            .collect();
        let (bare, biased) = (numbers[0], numbers[numbers.len() - 1]);
        let bias = bias_at(&flag, &arguments, &sys);
        assert!(
            (bias - (biased - bare)).abs() < 1e-7 * (1.0 + bias.abs()),
            "{cell}: estimate bias {bias}, reported {}",
            biased - bare
        );
        if kind == "absolute" {
            assert!(!log.contains("laid onto each other"), "{cell}");
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn bias_potentials_on_an_isolated_cluster() {
    context("isolated", &cluster(false), &cluster(true), &[], (4, 1));
}

#[test]
fn bias_potentials_in_a_periodic_cell() {
    context("periodic", &fcc(false), &fcc(true), &[], (32, 1));
}

#[test]
fn bias_potentials_with_oniom_microiterations() {
    context(
        "oniom",
        &cluster(false),
        &cluster(true),
        &["-oniom", "1-4", "none", "lj"],
        (4, 1),
    );
}

#[test]
fn a_warm_started_bias_gives_the_same_search_on_one_thread_and_on_four() {
    let dir = scratch("threads");
    let (a, b) = (dir.join("a.xyz"), dir.join("b.xyz"));
    std::fs::write(&a, cluster(false)).unwrap();
    std::fs::write(&b, cluster(true)).unwrap();
    let mut found = Vec::new();
    for threads in ["1", "4"] {
        let args: Vec<String> = [
            a.to_str().unwrap(),
            b.to_str().unwrap(),
            "-os",
            "lj",
            "-ns",
            "6",
            "-core",
            threads,
            "-ieip_opt",
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
        found.push(std::fs::read_to_string(out.directory.join("ieip_energies.csv")).unwrap());
        std::thread::sleep(std::time::Duration::from_millis(1100));
    }
    assert_eq!(found[0], found[1]);
    let _ = std::fs::remove_dir_all(&dir);
}
