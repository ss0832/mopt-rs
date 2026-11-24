//! The energy `optmain` reports with ONIOM microiterations: the bare energy of its result
//! (and of its files) is the ONIOM energy of the final whole structure, computed directly
//! with the ONIOM calculator, and the bare energy plus the bias is the biased energy; with
//! and without a bias potential, on Lennard-Jones layers alone.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used, clippy::panic)]

mod matrix_support;

use std::path::PathBuf;

use mopt_rs::cli::optmain;
use mopt_rs::domain::{EvalRequest, SpinCharge};
use mopt_rs::io::xyz::read_xyz_file;
use mopt_rs::pes::registry::CalculatorRegistry;

const AR7: &str = "7\nAr7\nAr 3.187637 0.000000 -0.082285\nAr 1.038255 3.074234 -0.032286\n\
Ar -2.569000 1.900000 0.017714\nAr -2.645818 -1.900000 0.067714\nAr 0.878417 -3.074234 0.117714\n\
Ar -0.044787 0.000000 2.007710\nAr 0.155214 0.000000 -2.096278\n";

fn scratch(name: &str) -> PathBuf {
    let dir =
        std::env::temp_dir().join(format!("mopt_oniom_summary_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// The last energy of `energy_profile.csv` in `dir` (Hartree), when the file has one.
fn last_profile_energy(dir: &std::path::Path) -> Option<f64> {
    let text = std::fs::read_to_string(dir.join("energy_profile.csv")).ok()?;
    let line = text.lines().rev().find(|l| !l.trim().is_empty())?;
    // `iter,bias,effective,raw`.
    line.split(',').nth(3)?.trim().parse().ok()
}

#[test]
fn the_bare_energy_with_microiterations_is_the_oniom_energy_of_the_final_structure() {
    let registry = CalculatorRegistry::native();
    // Converged, and stopped by the iteration limit (the last evaluated structure is the
    // result, not the step proposed after it).
    for (name, bias, steps) in [
        ("restraint", vec!["-kp", "0.002", "3.90", "1,2"], "200"),
        ("plain", Vec::<&str>::new(), "200"),
        ("short_restraint", vec!["-kp", "0.002", "3.90", "1,2"], "2"),
        ("short", Vec::<&str>::new(), "2"),
    ] {
        let dir = scratch(name);
        let file = dir.join("ar7.xyz");
        std::fs::write(&file, AR7).unwrap();
        let file = file.to_string_lossy().into_owned();
        let mut args = vec![
            file.as_str(),
            "-os",
            "lj",
            "-oniom",
            "1-4",
            "none",
            "lj",
            "-ns",
            steps,
        ];
        args.extend(bias.iter().copied());
        let config = optmain::config_from_cli(&args).unwrap();
        let out = optmain::run_with(&config, &dir, None, &registry).unwrap();
        let summary = out[0].summary.as_ref().unwrap();
        let whole = &summary.system;
        assert_eq!(whole.n_atoms(), 7, "{name}");
        // The ONIOM energy of the final whole structure, as the run ended with it.
        let calc = registry
            .create_oniom(&config.calculator, &config.calculator_options, whole)
            .unwrap()
            .at_given_structures();
        let direct = calc
            .evaluate(whole, &EvalRequest::ENERGY_GRAD)
            .unwrap()
            .energy
            .value();
        let bare = summary.energy.value();
        let biased = summary.biased_energy.value();
        assert!(
            (bare - direct).abs() < 1e-10,
            "{name}: bare {bare} against the ONIOM energy {direct} of the final structure"
        );
        // The bias at the final structure.
        let b = match bias.split_first() {
            None => 0.0,
            Some((flag, rest)) => {
                let rest: Vec<String> = rest.iter().map(|s| s.to_string()).collect();
                matrix_support::bias_energy(flag, &rest, whole)
            }
        };
        assert!(
            (biased - bare - b).abs() < 1e-9,
            "{name}: biased {biased} - bare {bare} against the bias {b}"
        );
        // The optimized file holds the same structure.
        let optimized = read_xyz_file(&out[0].optimized, SpinCharge::default(), None).unwrap();
        let from_file = calc
            .evaluate(&optimized, &EvalRequest::ENERGY_GRAD)
            .unwrap()
            .energy
            .value();
        assert!(
            (from_file - direct).abs() < 1e-9,
            "{name}: {from_file} {direct}"
        );
        if let Some(e) = last_profile_energy(&out[0].directory) {
            assert!(
                (e - direct).abs() < 1e-9,
                "{name}: profile {e} against {direct}"
            );
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
