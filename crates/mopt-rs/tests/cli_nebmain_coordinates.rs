//! `nebmain -coord`: chains of structures relaxed in internal coordinates.
//!
//! MultiOptPy: no counterpart (its chains are in Cartesian coordinates).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used)]

use std::path::{Path, PathBuf};

use mopt_rs::cli::nebmain::{config_from_cli, run_with, NebmainResult};
use mopt_rs::domain::SpinCharge;
use mopt_rs::domain::Units;
use mopt_rs::io::xyz::read_xyz_trajectory_file;
use mopt_rs::path::band::machine::ChainEnd;
use mopt_rs::pes::registry::CalculatorRegistry;

const KCAL: f64 = Units::HARTREE_TO_KCAL_MOL;

/// Three rough structures from HCN over a bridged one to HNC.
const HCN: &str = "3\nHCN\nC 0.000 0.000 0.000\nN 1.156 0.000 0.000\nH -1.066 0.000 0.000\n\
3\nbridged\nC 0.000 0.000 0.000\nN 1.190 0.000 0.000\nH 0.368 1.132 0.000\n\
3\nHNC\nC 0.000 0.000 0.000\nN 1.170 0.000 0.000\nH 2.165 0.000 0.000\n";

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("mopt_neb_coord_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn file(dir: &Path, name: &str, text: &str) -> String {
    let p = dir.join(name);
    std::fs::write(&p, text).unwrap();
    p.to_string_lossy().into_owned()
}

fn barrier(result: &NebmainResult) -> f64 {
    let top = result
        .energies
        .iter()
        .fold(f64::NEG_INFINITY, |m, e| m.max(*e));
    KCAL * (top - result.energies[0])
}

#[test]
fn the_chain_in_internal_coordinates_finds_the_same_path() {
    let dir = scratch("hcn");
    let input = file(&dir, "hcn.xyz", HCN);
    let registry = CalculatorRegistry::native();
    let run = |more: &[&str]| {
        let mut arguments = vec![input.as_str(), "-os", "gfn0xtb", "-ns", "150"];
        arguments.extend_from_slice(more);
        arguments.extend_from_slice(&["-neb_opt", "figures=off"]);
        let config = config_from_cli(&arguments).unwrap();
        run_with(&config, &dir, None, &registry).unwrap()
    };
    let cartesian = run(&[]);
    assert_eq!(cartesian.end, Some(ChainEnd::Converged));
    for coordinates in ["tric-cart", "tric", "dic"] {
        let result = run(&["-coord", coordinates]);
        let log = std::fs::read_to_string(result.directory.join("mopt.log")).unwrap();
        println!(
            "{coordinates}: {} iterations, barrier {:.3} (Cartesian: {}, {:.3})",
            result.iterations,
            barrier(&result),
            cartesian.iterations,
            barrier(&cartesian)
        );
        assert_eq!(
            result.end,
            Some(ChainEnd::Converged),
            "{coordinates}: {log}"
        );
        // HCN and HNC are linear: the orientation of such a fragment is not defined, the
        // translation-rotation coordinates cannot be set up, and the delocalized internal
        // coordinates take their place.
        if coordinates == "dic" {
            assert!(!log.contains("cannot be set up"), "{log}");
        } else {
            assert!(
                log.contains(&format!(
                    "Coordinates (-coord {coordinates}) cannot be set up"
                )),
                "{log}"
            );
        }
        assert!(
            log.contains("Coordinates (-coord dic): the chain is relaxed in 3"),
            "{log}"
        );
        // Every iteration has its line and the table of the images, on the screen and
        // in the log.
        assert_eq!(
            log.matches("  image   E - E(first)").count(),
            result.iterations,
            "{log}"
        );
        // The same transition state and the same ends.
        assert!(
            (barrier(&result) - 68.285).abs() < 0.1,
            "{coordinates}: {}",
            barrier(&result)
        );
        let last = result.energies.len() - 1;
        let reaction = KCAL * (result.energies[last] - result.energies[0]);
        assert!((reaction - 15.72).abs() < 0.05, "{coordinates}: {reaction}");
        // The files have structures of the three atoms, not vectors of coordinates.
        let candidates = result.candidates.clone().unwrap();
        let path = read_xyz_trajectory_file(&candidates.path, SpinCharge::default(), None).unwrap();
        assert_eq!(path.frames.len(), result.energies.len());
        assert!(path.frames.iter().all(|frame| frame.system.n_atoms() == 3));
    }

    // With the model of -neb_opt gp=on, which then works in these coordinates.
    let modelled = run(&["-coord", "dic", "-neb_opt", "gp=on"]);
    assert_eq!(modelled.end, Some(ChainEnd::Converged));
    assert!(
        (barrier(&modelled) - 68.285).abs() < 0.1,
        "{}",
        barrier(&modelled)
    );

    // What cannot be: redundant coordinates, atoms held in place, plain variables.
    let refused = |arguments: &[&str]| {
        let config = config_from_cli(arguments).unwrap();
        run_with(&config, &dir, None, &registry).is_err()
    };
    assert!(refused(&[&input, "-os", "gfn0xtb", "-coord", "ric"]));
    assert!(refused(&[
        &input, "-os", "gfn0xtb", "-coord", "dic", "-fix", "1"
    ]));
    let start = file(&dir, "start.txt", "x -0.558\ny 1.442\n");
    let end = file(&dir, "end.txt", "x 0.623\ny 0.028\n");
    assert!(refused(&[&start, &end, "-os", "mb2d", "-coord", "dic"]));
    let _ = std::fs::remove_dir_all(&dir);
}
