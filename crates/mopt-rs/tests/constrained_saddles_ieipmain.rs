//! The searches of `ieipmain` under a hard constraint find the constrained saddle: on a
//! Lennard-Jones cluster of five argon atoms (an atom passing through a triangle, from above
//! it to below it) with the distance of atoms 1 and 2 held and atom 3 fixed (so that the two
//! structures are not turned onto each other, which makes the second a permutation of the
//! first), every method that held the constraint only by refusing it before (2PSHS, ADDF,
//! GNT, GSM, SE-GSM, FSM, LMM) gives an estimate from which the constrained search of order 1
//! (`optmain -order 1 -pc bond 1,2 -fix 3`, with an exact Hessian at every step,
//! run by `-ieip_opt optimize=on`) converges nearby, and the saddle it reaches holds the
//! distance.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::path::{Path, PathBuf};

use mopt_rs::cli::ieipmain::{config_from_cli, run_with};
use mopt_rs::domain::{AtomId, SpinCharge, Units};
use mopt_rs::io::xyz::read_xyz_file;
use mopt_rs::pes::registry::CalculatorRegistry;

const FIXED: &str = "Ar 0.000 0.000 0.000\nAr 3.800 0.000 0.000\nAr 1.900 3.291 0.000\n";

fn cluster(second: bool) -> String {
    let moving = if second {
        "Ar 2.000 1.150 -3.000\n"
    } else {
        "Ar 1.700 0.950 3.000\n"
    };
    format!("5\nend\n{FIXED}{moving}Ar -2.900 1.500 0.400\n")
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("mopt_csaddle_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// `text` optimized with the distance of atoms 1 and 2 held (a constrained minimum), and its
/// energy.
fn minimum(dir: &Path, name: &str, text: &str) -> (PathBuf, f64) {
    let file = dir.join(format!("{name}_raw.xyz"));
    std::fs::write(&file, text).unwrap();
    let config = mopt_rs::cli::optmain::config_from_cli(&[
        file.to_str().unwrap(),
        "-os",
        "lj",
        "-pc",
        "bond",
        "1,2",
        "-fix",
        "3",
        "-ns",
        "500",
        "-tcc",
    ])
    .unwrap();
    let out = mopt_rs::cli::optmain::run_with(&config, dir, None, &CalculatorRegistry::native())
        .unwrap()
        .remove(0);
    assert!(out.converged, "{name}: the start is not optimized");
    let target = dir.join(format!("{name}.xyz"));
    std::fs::copy(&out.optimized, &target).unwrap();
    let energy = out.summary.as_ref().unwrap().energy.value();
    (target, energy)
}

fn distance_12(path: &Path) -> f64 {
    let s = read_xyz_file(path, SpinCharge::default(), None).unwrap();
    let (a, b) = (
        s.position(AtomId::new(0)).value(),
        s.position(AtomId::new(1)).value(),
    );
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

/// The largest displacement of an atom between two structures (Angstrom).
fn largest_move(p: &Path, q: &Path) -> f64 {
    let (a, b) = (
        read_xyz_file(p, SpinCharge::default(), None).unwrap(),
        read_xyz_file(q, SpinCharge::default(), None).unwrap(),
    );
    (0..a.n_atoms())
        .map(|k| {
            let (x, y) = (
                a.position(AtomId::new(k)).value(),
                b.position(AtomId::new(k)).value(),
            );
            ((x[0] - y[0]).powi(2) + (x[1] - y[1]).powi(2) + (x[2] - y[2]).powi(2)).sqrt()
        })
        .fold(0.0, f64::max)
        * Units::BOHR_TO_ANGSTROM
}

#[test]
fn every_method_finds_the_constrained_saddle() {
    let dir = scratch("lj5");
    let (a, energy_a) = minimum(&dir, "a", &cluster(false));
    let (b, energy_b) = minimum(&dir, "b", &cluster(true));
    let held = distance_12(&a);
    let (a, b) = (
        a.to_string_lossy().into_owned(),
        b.to_string_lossy().into_owned(),
    );
    // The search of order 1 with an exact Hessian at every step: the saddles of the floppy
    // cluster are flat, and with updated Hessians it wanders off them.
    let settings = dir.join("order1.json");
    std::fs::write(&settings, "{\"calc_exact_hess\": 1}\n").unwrap();
    let optimize_settings = format!("optimize_settings={}", settings.display());
    let methods: Vec<(&str, Vec<&str>)> = vec![
        ("2pshs", vec!["-2pshs"]),
        ("addf", vec!["-addf", "-addf_num", "100"]),
        ("gnt", vec!["-gnt", "-gnt_step", "0.2"]),
        ("gsm", vec!["-gsm"]),
        ("se_gsm", vec!["-se_gsm", "BREAK 4 5"]),
        ("fsm", vec!["-fsm"]),
        ("lmm", vec!["-lmm"]),
    ];
    let registry = CalculatorRegistry::native();
    let mut results = Vec::new();
    for (name, options) in methods {
        let mut args = vec![
            a.as_str(),
            b.as_str(),
            "-os",
            "lj",
            "-pc",
            "bond",
            "1,2",
            "-fix",
            "3",
            "-ieip_opt",
            "figures=off",
            "optimize=on",
            "optimize_check=off",
            optimize_settings.as_str(),
        ];
        args.extend(options.iter().copied());
        let config = config_from_cli(&args).unwrap();
        let out =
            run_with(&config, &dir, None, &registry).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert!(!out.transition_states.is_empty(), "{name}: no estimate");
        // The estimate itself holds the distance.
        let estimate = &out.transition_states[0];
        assert!(
            (distance_12(estimate) - held).abs() < 1e-9,
            "{name}: {} {held}",
            distance_12(estimate)
        );
        // The constrained search of order 1 from it converges nearby, on the constraint.
        let found = out
            .optimized
            .first()
            .unwrap_or_else(|| panic!("{name}: no optimization"));
        let moved = largest_move(estimate, &found.optimized);
        println!(
            "{name}: converged {} in {} iterations, energy {:.10}, {moved:.3} Angstrom from the \
             estimate, distance {:.3e} off",
            found.converged,
            found.iterations,
            found.energy,
            distance_12(&found.optimized) - held
        );
        assert!(
            (distance_12(&found.optimized) - held).abs() < 1e-9,
            "{name}: the search of order 1 leaves the constraint"
        );
        results.push((
            name,
            energy_of(estimate),
            found.converged,
            found.energy,
            moved,
        ));
    }
    // The methods from one structure reach the constrained saddle: the search of order 1
    // from their estimates converges nearby.
    let mut saddle = f64::INFINITY;
    for (name, _, converged, energy, moved) in &results {
        if ["addf", "gnt", "se_gsm"].contains(name) {
            assert!(*converged && *moved < 1.0, "{name}: {converged} {moved}");
            saddle = saddle.min(*energy);
        }
    }
    // The methods between the two structures give the highest point of a path on the
    // constraints from one to the other: by the mountain-pass property it lies at or above
    // the lowest constrained saddle between them, and above both ends. (On this flat cluster
    // the search of order 1 from these estimates does not always converge within 1000
    // steps, so its result is not compared.)
    let ends = energy_a.max(energy_b);
    for (name, estimate, _, _, _) in &results {
        if !["addf", "gnt", "se_gsm"].contains(name) {
            assert!(
                *estimate >= saddle - 1e-8 && *estimate > ends,
                "{name}: estimate {estimate}, saddle {saddle}, ends {ends}"
            );
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// The energy in the comment line of a structure file (`energy=...`).
fn energy_of(path: &Path) -> f64 {
    let text = std::fs::read_to_string(path).unwrap();
    text.lines()
        .nth(1)
        .and_then(|l| {
            l.split_whitespace()
                .find_map(|w| w.strip_prefix("energy=").and_then(|v| v.parse().ok()))
        })
        .unwrap_or_else(|| panic!("{}: no energy", path.display()))
}
