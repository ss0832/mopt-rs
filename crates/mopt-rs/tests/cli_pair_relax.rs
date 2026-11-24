//! The relaxation of the path of a two-point method afterwards (`-ieip_opt method_relax=`),
//! a chain of `nebmain` (`lup`: the locally updated planes) whose maxima become the
//! estimates: after a method other than DS-AFIR, and after DS-AFIR through the generic keys
//! and through DS-AFIR's keys of earlier versions (`dsafir_lup=N`), which give the same run.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used)]

use std::path::{Path, PathBuf};

use mopt_rs::cli::ieipmain::{config_from_cli, run_with, IeipmainResult};
use mopt_rs::pes::native::muller_brown::MB_SADDLES;
use mopt_rs::pes::registry::CalculatorRegistry;

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("mopt_pair_relax_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn file(dir: &Path, name: &str, text: &str) -> String {
    let p = dir.join(name);
    std::fs::write(&p, text).unwrap();
    p.to_string_lossy().into_owned()
}

fn run(dir: &Path, arguments: &[&str]) -> IeipmainResult {
    let config = config_from_cli(arguments).unwrap();
    run_with(&config, dir, None, &CalculatorRegistry::native()).unwrap()
}

/// The two variables of a file of variables.
fn variables(path: &Path) -> Vec<f64> {
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .filter_map(|l| l.split_whitespace().nth(1)?.parse().ok())
        .collect()
}

fn distance_to_saddle(x: &[f64]) -> f64 {
    MB_SADDLES
        .iter()
        .map(|s| ((x[0] - s[0]).powi(2) + (x[1] - s[1]).powi(2)).sqrt())
        .fold(f64::INFINITY, f64::min)
}

/// The path of the freezing string between the minima A and B of the Muller-Brown surface
/// relaxed by the locally updated planes: the estimate is a maximum of the relaxed chain,
/// near the saddle between them, and the relaxed path is written.
#[test]
fn the_path_of_another_method_is_relaxed_by_the_planes() {
    let dir = scratch("fsm");
    let a = file(&dir, "a.txt", "x -0.558\ny 1.442\n");
    let b = file(&dir, "b.txt", "x -0.050\ny 0.467\n");
    let plain = run(
        &dir,
        &[&a, &b, "-os", "mb2d", "-fsm", "-ieip_opt", "figures=off"],
    );
    let relaxed = run(
        &dir,
        &[
            &a,
            &b,
            "-os",
            "mb2d",
            "-fsm",
            "-ieip_opt",
            "figures=off",
            "method_relax=lup",
            "method_relax_iterations=60",
            "method_relax_nodes=11",
        ],
    );
    let before = distance_to_saddle(&variables(&plain.transition_states[0]));
    let after = distance_to_saddle(&variables(&relaxed.transition_states[0]));
    assert!(after < 0.05, "{after}");
    assert!(after <= before + 1e-3, "{after} {before}");
    let log = std::fs::read_to_string(relaxed.directory.join("mopt.log")).unwrap();
    assert!(
        log.contains("Relaxation of the path (method_relax=lup)"),
        "{log}"
    );
    assert!(relaxed.directory.join("relaxed_path.xyz").is_file() || log.contains("maxima"));
    // A method without a path between the structures refuses the step by name.
    let config = config_from_cli(&[
        &a,
        &b,
        "-os",
        "mb2d",
        "-use_dimer",
        "-ieip_opt",
        "method_relax=lup",
    ])
    .unwrap();
    let err = run_with(&config, &dir, None, &CalculatorRegistry::native())
        .unwrap_err()
        .to_string();
    assert!(err.contains("method_relax"), "{err}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// DS-AFIR followed by the LUP relaxation: `dsafir_lup=N` (DS-AFIR's keys of earlier
/// versions) is the same step as `method_relax=lup`, and the two runs agree.
#[test]
fn the_alias_of_dsafir_is_the_generic_relaxation() {
    let dir = scratch("dsafir");
    let a = file(&dir, "a.txt", "x -0.558\ny 1.442\n");
    let b = file(&dir, "b.txt", "x -0.050\ny 0.467\n");
    let common = [
        a.as_str(),
        b.as_str(),
        "-os",
        "mb2d",
        "-dsafir",
        "-ieip_opt",
        "figures=off",
    ];
    let mut alias = common.to_vec();
    alias.extend([
        "dsafir_lup=5",
        "dsafir_lup_nodes=9",
        "dsafir_lup_max_step=0.05",
    ]);
    let mut generic = common.to_vec();
    generic.extend([
        "method_relax=lup",
        "method_relax_iterations=5",
        "method_relax_nodes=9",
        "method_relax_max_step=0.05",
    ]);
    let x = run(&dir, &alias);
    let y = run(&dir, &generic);
    let read = |p: &Path| std::fs::read_to_string(p).unwrap();
    assert_eq!(x.transition_states.len(), y.transition_states.len());
    for (p, q) in x.transition_states.iter().zip(&y.transition_states) {
        assert_eq!(read(p), read(q));
    }
    let log = read(&x.directory.join("mopt.log"));
    assert!(
        log.contains("dsafir lup=5: the relaxation of the path after the method"),
        "{log}"
    );
    // Both at once are refused.
    let mut both = alias.clone();
    both.push("method_relax=lup");
    let config = config_from_cli(&both).unwrap();
    assert!(run_with(&config, &dir, None, &CalculatorRegistry::native()).is_err());
    let _ = std::fs::remove_dir_all(&dir);
}
