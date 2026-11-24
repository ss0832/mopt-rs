//! `optmain -bias_derivatives`: the derivatives in the constants of a bias option and the
//! response of the optimized structure agree with differences of whole optimizations at the
//! constants moved up and down (two argon atoms with the built-in `lj` and a `-kp` spring that
//! pulls them from the minimum of the pair).
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::path::{Path, PathBuf};

use mopt_rs::cli::bias::parse_bias_option;
use mopt_rs::cli::optmain;
use mopt_rs::domain::{AtomId, EvalRequest, MolecularSystem, Units};
use mopt_rs::pes::native::LennardJones;
use mopt_rs::pes::registry::CalculatorRegistry;
use mopt_rs::pes::term::TermContext;
use mopt_rs::pes::Calculator;

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("mopt_biasder_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// The optimized structure of the pair with the spring `-kp K R0 1,2`, and the run directory.
fn optimize(base: &Path, k: f64, r0: f64, derivatives: bool) -> (MolecularSystem, PathBuf) {
    let input = base.join("ar2.xyz");
    std::fs::write(&input, "2\n0 1\nAr 0.0 0.0 0.0\nAr 3.9 0.0 0.0\n").unwrap();
    let k = k.to_string();
    let r0 = r0.to_string();
    let input = input.to_string_lossy().into_owned();
    let mut args = vec![
        input.as_str(),
        "-os",
        "lj",
        "-kp",
        k.as_str(),
        r0.as_str(),
        "1,2",
        "-tcc",
        "-fc",
        "1",
        "-ns",
        "200",
    ];
    if derivatives {
        args.push("-bias_derivatives");
    }
    let parsed = mopt_rs::cli::option::parser("optmain", mopt_rs::cli::spec::OPTMAIN)
        .parse(args)
        .unwrap();
    let config = optmain::OptmainConfig::from_args(&parsed).unwrap();
    let out = optmain::run_with(&config, base, None, &CalculatorRegistry::native()).unwrap();
    assert!(out[0].converged);
    (
        out[0].summary.clone().unwrap().system,
        out[0].directory.clone(),
    )
}

fn energies(sys: &MolecularSystem, k: f64, r0: f64) -> (f64, f64) {
    let e0 = LennardJones
        .evaluate(sys, &EvalRequest::ENERGY_ONLY)
        .unwrap()
        .energy
        .value();
    let tokens = [k.to_string(), r0.to_string(), "1,2".to_string()];
    let spec = parse_bias_option("-kp", &tokens).unwrap();
    let ctx = TermContext::at_iteration(0, sys);
    let bias: f64 = spec
        .iter()
        .map(|s| {
            s.build(sys)
                .unwrap()
                .evaluate(sys, &ctx, false)
                .unwrap()
                .energy
                .value()
        })
        .sum();
    (e0, e0 + bias)
}

fn distance(sys: &MolecularSystem) -> f64 {
    sys.distance(AtomId::new(0), AtomId::new(1)).value() * Units::BOHR_TO_ANGSTROM
}

/// The rows of `bias_derivatives.csv` by their option argument.
fn rows(dir: &Path) -> Vec<Vec<String>> {
    let text = std::fs::read_to_string(dir.join("bias_derivatives.csv")).unwrap();
    text.lines()
        .skip(1)
        .map(|l| {
            // The first column is quoted and has no commas.
            l.split(',')
                .map(|s| s.trim_matches('"').to_string())
                .collect()
        })
        .collect()
}

#[test]
fn the_derivatives_agree_with_differences_of_optimizations() {
    let base = scratch("ar2");
    let (k, r0) = (0.002, 4.6);
    let (sys, dir) = optimize(&base, k, r0, true);
    let table = rows(&dir);
    // Two constants: the force constant and the distance; the atoms are none.
    assert_eq!(table.len(), 2, "{table:?}");
    assert_eq!(table[0][2], "1");
    assert_eq!(table[1][2], "2");
    let log = std::fs::read_to_string(dir.join("mopt.log")).unwrap();
    assert!(log.contains("Derivatives in the constants of the bias options"));
    // dE/dp of both constants at every iteration; the last row is that of the converged
    // structure.
    let per_iteration =
        std::fs::read_to_string(dir.join("bias_derivatives_iterations.csv")).unwrap();
    let lines: Vec<&str> = per_iteration.lines().collect();
    assert!(lines.len() >= 3, "{per_iteration}");
    assert_eq!(lines[0].matches("dE/dp").count(), 2);
    let last: Vec<f64> = lines[lines.len() - 1]
        .split(',')
        .skip(1)
        .map(|v| v.parse().unwrap())
        .collect();
    let reported: f64 = table[1][6].parse().unwrap();
    assert!(
        (last[1] - reported).abs() <= 1e-6 * reported.abs(),
        "{last:?}"
    );
    for (row, (step, moved)) in table.iter().zip([
        (2e-5, (|h: f64| (0.002 + h, 4.6)) as fn(f64) -> (f64, f64)),
        (2e-3, |h: f64| (0.002, 4.6 + h)),
    ]) {
        let (ku, ru) = moved(step);
        let (kd, rd) = moved(-step);
        let (up, _) = optimize(&base, ku, ru, false);
        let (down, _) = optimize(&base, kd, rd, false);
        let (e0_up, e_up) = energies(&up, ku, ru);
        let (e0_down, e_down) = energies(&down, kd, rd);
        let de = (e_up - e_down) / (2.0 * step);
        let de0 = (e0_up - e0_down) / (2.0 * step);
        let reported_de: f64 = row[6].parse().unwrap();
        let reported_de0: f64 = row[8].parse().unwrap();
        eprintln!(
            "{}: dE/dp {reported_de:.6e} against {de:.6e}; dE0/dp {reported_de0:.6e} against \
             {de0:.6e}; r* {:.5} A",
            row[0],
            distance(&sys)
        );
        assert!(
            (reported_de - de).abs() <= 1e-3 * de.abs() + 1e-9,
            "{}: dE/dp {reported_de} against {de}",
            row[0]
        );
        assert!(
            (reported_de0 - de0).abs() <= 2e-3 * de0.abs() + 1e-9,
            "{}: dE0/dp {reported_de0} against {de0}",
            row[0]
        );
        // The atoms move apart symmetrically: each by half the change of the distance.
        let shift: f64 = row[10].parse().unwrap();
        let dr = (distance(&up) - distance(&down)) / (2.0 * step);
        assert!(
            (shift - 0.5 * dr.abs()).abs() <= 2e-3 * dr.abs() + 1e-9,
            "{}: shift {shift} against {}",
            row[0],
            0.5 * dr.abs()
        );
    }
}
