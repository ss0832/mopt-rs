//! `nebmain` with a Gaussian-process model of the surface (`-neb_opt gp=on`).
//!
//! MultiOptPy: `MEP/pathopt_gpneb_force.py` (unfinished there).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used)]

use std::path::{Path, PathBuf};

use mopt_rs::cli::nebmain::{config_from_cli, run_with, NebmainResult};
use mopt_rs::domain::Units;
use mopt_rs::path::band::machine::ChainEnd;
use mopt_rs::pes::registry::CalculatorRegistry;

const KCAL: f64 = Units::HARTREE_TO_KCAL_MOL;

/// Three rough structures from HCN over a bridged one to HNC.
const HCN: &str = "3\nHCN\nC 0.000 0.000 0.000\nN 1.156 0.000 0.000\nH -1.066 0.000 0.000\n\
3\nbridged\nC 0.000 0.000 0.000\nN 1.190 0.000 0.000\nH 0.368 1.132 0.000\n\
3\nHNC\nC 0.000 0.000 0.000\nN 1.170 0.000 0.000\nH 2.165 0.000 0.000\n";

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("mopt_neb_gp_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn file(dir: &Path, name: &str, text: &str) -> String {
    let p = dir.join(name);
    std::fs::write(&p, text).unwrap();
    p.to_string_lossy().into_owned()
}

/// The gradient calculations of a run, from its log.
fn gradients(result: &NebmainResult) -> usize {
    let log = std::fs::read_to_string(result.directory.join("mopt.log")).unwrap();
    let line = log
        .lines()
        .find(|line| line.contains("Evaluations (this run)"))
        .unwrap();
    line.split_whitespace()
        .find_map(|word| word.parse::<usize>().ok())
        .unwrap()
}

#[test]
fn the_model_answers_for_most_images_and_the_chain_is_the_same() {
    let dir = scratch("hcn");
    let input = file(&dir, "hcn.xyz", HCN);
    let registry = CalculatorRegistry::native();
    let run = |more: &[&str]| {
        let mut arguments = vec![input.as_str(), "-os", "gfn0xtb", "-ns", "150", "-neb_opt"];
        arguments.extend_from_slice(more);
        arguments.push("figures=off");
        let config = config_from_cli(&arguments).unwrap();
        run_with(&config, &dir, None, &registry).unwrap()
    };
    let plain = run(&[]);
    let modelled = run(&["gp=on"]);
    assert_eq!(plain.end, Some(ChainEnd::Converged));
    assert_eq!(modelled.end, Some(ChainEnd::Converged));
    let barrier = |result: &NebmainResult| {
        let top = result
            .energies
            .iter()
            .fold(f64::NEG_INFINITY, |m, e| m.max(*e));
        KCAL * (top - result.energies[0])
    };
    println!(
        "gradients {} against {}, iterations {} against {}, barriers {:.3} and {:.3}",
        gradients(&modelled),
        gradients(&plain),
        modelled.iterations,
        plain.iterations,
        barrier(&modelled),
        barrier(&plain)
    );
    // The last chain is calculated and relaxed: the same barrier. For this molecule the
    // model is never good enough against the forces, and nothing is saved.
    assert!(
        (barrier(&modelled) - 68.285).abs() < 0.1,
        "{}",
        barrier(&modelled)
    );
    assert!(gradients(&modelled) <= gradients(&plain));
    let log = std::fs::read_to_string(modelled.directory.join("mopt.log")).unwrap();
    assert!(log.contains("Gaussian process (-neb_opt gp=on):"), "{log}");
    assert!(log.contains("answered from the model"), "{log}");

    // The settings.
    let config = config_from_cli(&[
        &input,
        "-os",
        "gfn0xtb",
        "-neb_opt",
        "gp=on",
        "gp_uncertainty=0.01",
        "gp_length=0.5",
        "gp_points=8",
        "gp_memory=10",
    ])
    .unwrap();
    let model = config.surrogate.unwrap();
    assert_eq!(
        (model.uncertainty, model.neighbours, model.memory),
        (0.01, 8, 10)
    );
    assert_eq!(config.surrogate_length, Some(0.5));
    assert_eq!(config.surrogate_features, None);
    assert!(config_from_cli(&[&input, "-os", "gfn0xtb"])
        .unwrap()
        .surrogate
        .is_none());
    for wrong in ["gp=yes", "gp_uncertainty=2", "gp_length=0"] {
        assert!(config_from_cli(&[&input, "-os", "gfn0xtb", "-neb_opt", wrong]).is_err());
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn plain_variables_with_the_model() {
    let dir = scratch("mb");
    let start = file(&dir, "start.txt", "x -0.558\ny 1.442\n");
    let end = file(&dir, "end.txt", "x 0.623\ny 0.028\n");
    let registry = CalculatorRegistry::native();
    let run = |more: &[&str]| {
        let mut arguments = vec![
            start.as_str(),
            end.as_str(),
            "-os",
            "mb2d",
            "-p",
            "15",
            "-fe",
            "3",
            "-ns",
            "300",
            "-neb_opt",
            "tolerance=1e-4",
            "figures=off",
        ];
        arguments.extend_from_slice(more);
        let config = config_from_cli(&arguments).unwrap();
        run_with(&config, &dir, None, &registry).unwrap()
    };
    let plain = run(&[]);
    // The surface is in units in which the default length scale is too long.
    let result = run(&["gp=on", "gp_length=0.25", "gp_error=0.3"]);
    assert_eq!(result.end, Some(ChainEnd::Converged));
    // An image on each of the two saddles of the path, as without the model.
    let top = |r: &NebmainResult| r.energies.iter().fold(f64::NEG_INFINITY, |m, e| m.max(*e));
    println!(
        "top {} against {}, {} iterations against {}",
        top(&result),
        top(&plain),
        result.iterations,
        plain.iterations
    );
    assert_eq!(result.candidates.clone().unwrap().images.len(), 2);
    assert!(
        (top(&result) - top(&plain)).abs() < 1e-3 * top(&plain).abs().max(1.0),
        "{} {}",
        top(&result),
        top(&plain)
    );
    let log = std::fs::read_to_string(result.directory.join("mopt.log")).unwrap();
    assert!(log.contains("answered from the model"), "{log}");
    let _ = std::fs::remove_dir_all(&dir);
}
