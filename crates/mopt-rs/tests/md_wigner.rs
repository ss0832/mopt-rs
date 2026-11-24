//! `mdmain -md_opt velocities=wigner`: trajectories of water (GFN-FF) started from the
//! vibrational ground state hold the zero-point energy on average.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used)]

use std::path::PathBuf;

use mopt_rs::cli::mdmain::{config_from_cli, run_with};
use mopt_rs::cli::{freqmain, optmain};
use mopt_rs::domain::Units;
use mopt_rs::domain::{EvalRequest, SpinCharge};
use mopt_rs::io::xyz::read_xyz_file;
use mopt_rs::pes::registry::{CalculatorOptions, CalculatorRegistry};

const WATER: &str = "3\n\nO 0.0 0.0 0.12\nH 0.0 0.80 -0.47\nH 0.0 -0.70 -0.50\n";
/// Hartree per cm^-1.
const HARTREE_PER_WAVENUMBER: f64 = Units::WAVENUMBER_TO_HARTREE;

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("mopt_mdwig_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn trajectories_from_the_ground_state_hold_the_zero_point_energy() {
    let base = scratch("zpe");
    let start = base.join("water.xyz");
    std::fs::write(&start, WATER).unwrap();
    let s = start.to_string_lossy().into_owned();
    let cfg =
        optmain::config_from_cli(&[&s, "-os", "gfnff", "-opt", "rsirfo_bfgs", "-tcc"]).unwrap();
    let optimized = optmain::run(&cfg, &base, None).unwrap();
    let xyz = optimized[0].optimized.to_string_lossy().into_owned();
    let cfg = freqmain::config_from_cli(&[&xyz, "-os", "gfnff"]).unwrap();
    let freq = freqmain::run_with(&cfg, &base, None, &CalculatorRegistry::native()).unwrap();
    let zpe: f64 = std::fs::read_to_string(freq[0].directory.join("normal_modes.txt"))
        .unwrap()
        .lines()
        .filter(|l| l.starts_with("Freq [cm^-1]"))
        .flat_map(|l| {
            l.split_whitespace()
                .skip(2)
                .map(|v| v.parse::<f64>().unwrap())
                .collect::<Vec<_>>()
        })
        .map(|nu| 0.5 * nu * HARTREE_PER_WAVENUMBER)
        .sum();
    let config = config_from_cli(&[
        &xyz,
        "-os",
        "gfnff",
        "-mt",
        "nve",
        "-time",
        "1",
        "-ts",
        "0.25fs",
        "-temp",
        "300",
        "-traj",
        "40",
        "-md_opt",
        "seed=11",
        "velocities=wigner",
        "wigner_temperature=0",
    ])
    .unwrap();
    let runs = run_with(&config, &base, None, &CalculatorRegistry::native()).unwrap();
    assert_eq!(runs.len(), 40);
    let first_row = |dir: &std::path::Path| -> (f64, f64) {
        let text = std::fs::read_to_string(dir.join("md_properties.csv")).unwrap();
        let row: Vec<f64> = text
            .lines()
            .nth(1)
            .unwrap()
            .split(',')
            .map(|v| v.parse().unwrap_or(f64::NAN))
            .collect();
        (row[3], row[4])
    };
    let log = std::fs::read_to_string(runs[0].directory.join("mopt.log")).unwrap();
    assert!(log.contains("Wigner sampling at 0 K (3 modes"), "{log}");
    // The energy above the minimum: potential (the sampled structure) and kinetic.
    let sys = read_xyz_file(&optimized[0].optimized, SpinCharge::default(), None).unwrap();
    let emin = CalculatorRegistry::native()
        .create("gfnff", &CalculatorOptions::default(), &sys)
        .unwrap()
        .evaluate(&sys, &EvalRequest::ENERGY_ONLY)
        .unwrap()
        .energy
        .value();
    let energies: Vec<f64> = runs
        .iter()
        .map(|r| {
            let (pot, kin) = first_row(&r.directory);
            pot - emin + kin
        })
        .collect();
    let mean = energies.iter().sum::<f64>() / energies.len() as f64;
    eprintln!("water, 40 samples at 0 K: mean vibrational energy {mean:.6} Hartree, ZPE {zpe:.6}");
    // Three modes of exponentially distributed energy: the mean of 40 samples is within
    // about 9 % of the zero-point energy (one standard error); anharmonicity adds a little.
    assert!((mean / zpe - 1.0).abs() < 0.3, "{mean} against {zpe}");
    let _ = std::fs::remove_dir_all(&base);
}
