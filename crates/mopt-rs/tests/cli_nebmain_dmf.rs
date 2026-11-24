//! `nebmain -dmf`: the path by the direct MaxFlux method.
//!
//! MultiOptPy: `neb.py:NEB` with `-dmf` (`MEP/pathopt_dmf_force.py`; the method is written
//! anew from its authors' definitions).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used)]

use std::path::{Path, PathBuf};

use mopt_rs::cli::nebmain::{config_from_cli, run_with};
use mopt_rs::domain::SpinCharge;
use mopt_rs::domain::Units;
use mopt_rs::io::xyz::{read_xyz_file, read_xyz_trajectory_file};
use mopt_rs::path::band::machine::ChainEnd;
use mopt_rs::pes::registry::CalculatorRegistry;

/// Three rough structures from HCN over a bridged one to HNC.
const HCN: &str = "3\nHCN\nC 0.000 0.000 0.000\nN 1.156 0.000 0.000\nH -1.066 0.000 0.000\n\
3\nbridged\nC 0.000 0.000 0.000\nN 1.190 0.000 0.000\nH 0.368 1.132 0.000\n\
3\nHNC\nC 0.000 0.000 0.000\nN 1.170 0.000 0.000\nH 2.165 0.000 0.000\n";

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("mopt_neb_dmf_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn file(dir: &Path, name: &str, text: &str) -> String {
    let p = dir.join(name);
    std::fs::write(&p, text).unwrap();
    p.to_string_lossy().into_owned()
}

/// The number after `prefix` in the line of the log that has it.
fn number_after(log: &str, prefix: &str) -> f64 {
    let line = log.lines().find(|l| l.contains(prefix)).unwrap();
    let rest = &line[line.find(prefix).unwrap() + prefix.len()..];
    rest.split_whitespace().next().unwrap().parse().unwrap()
}

#[test]
fn the_path_on_the_muller_brown_surface_goes_over_the_saddle() {
    let dir = scratch("mb");
    let start = file(&dir, "start.txt", "x -0.558224\ny 1.441726\n");
    let end = file(&dir, "end.txt", "x -0.050011\ny 0.466694\n");
    let config = config_from_cli(&[&start, &end, "-os", "mb2d", "-dmf"]).unwrap();
    let registry = CalculatorRegistry::native();
    let result = run_with(&config, &dir, None, &registry).unwrap();
    let log = std::fs::read_to_string(result.directory.join("mopt.log")).unwrap();
    assert_eq!(result.end, Some(ChainEnd::Converged), "{log}");
    // A line and a table of the points for every iteration.
    assert_eq!(log.matches("DMF ITR.").count(), result.iterations, "{log}");
    assert_eq!(log.matches("  point   share of").count(), result.iterations);
    // The estimate of the transition state is the saddle between the two minima.
    let guess = std::fs::read_to_string(result.directory.join("ts_guess.csv")).unwrap();
    let values: Vec<f64> = guess
        .lines()
        .nth(1)
        .unwrap()
        .split(',')
        .map(|v| v.parse().unwrap())
        .collect();
    let (x, y) = (values[1], values[2]);
    println!(
        "{} iterations, estimate ({x:.4}, {y:.4})",
        result.iterations
    );
    assert!(
        ((x + 0.822_002).powi(2) + (y - 0.624_313).powi(2)).sqrt() < 0.05,
        "{x} {y}"
    );
    // One estimate per iteration, the path, the points of the last iteration, and the
    // figures, among them the path on the map of the surface.
    let highest = std::fs::read_to_string(result.directory.join("dmf_highest.csv")).unwrap();
    assert_eq!(highest.lines().count(), result.iterations + 1);
    let path = std::fs::read_to_string(result.directory.join("dmf_path.csv")).unwrap();
    assert_eq!(path.lines().count(), 42);
    assert!(result.candidates.unwrap().path.is_file());
    for figure in ["energy_profile", "convergence", "chain_on_surface"] {
        assert!(
            result.directory.join(format!("{figure}.png")).is_file(),
            "{figure}"
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn hcn_by_the_direct_max_flux_method() {
    let dir = scratch("hcn");
    let input = file(&dir, "hcn.xyz", HCN);
    let registry = CalculatorRegistry::native();
    let config =
        config_from_cli(&[&input, "-os", "gfn0xtb", "-dmf", "-neb_opt", "figures=off"]).unwrap();
    let result = run_with(&config, &dir, None, &registry).unwrap();
    let log = std::fs::read_to_string(result.directory.join("mopt.log")).unwrap();
    assert_eq!(result.end, Some(ChainEnd::Converged), "{log}");
    // The top of the path is the transition state (68.285 kcal/mol above HCN, from
    // the optimization of the saddle).
    let barrier = number_after(
        &log,
        "Estimate of the transition state: the top of the path,",
    );
    println!("{} iterations, barrier {barrier:.3}", result.iterations);
    assert!((barrier - 68.285).abs() < 0.5, "{barrier}");
    let guess = read_xyz_file(
        result.directory.join("hcn_ts_guess.xyz"),
        SpinCharge::default(),
        None,
    )
    .unwrap();
    assert_eq!(guess.n_atoms(), 3);
    let highest = read_xyz_trajectory_file(
        result.directory.join("hcn_dmf_highest.xyz"),
        SpinCharge::default(),
        None,
    )
    .unwrap();
    assert_eq!(highest.frames.len(), result.iterations);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn what_the_direct_max_flux_method_does_not_take() {
    let dir = scratch("refused");
    let input = file(&dir, "hcn.xyz", HCN);
    let registry = CalculatorRegistry::native();
    // Its settings without -dmf, settings it does not have, and values that cannot be.
    assert!(config_from_cli(&[&input, "-os", "gfn0xtb", "-neb_opt", "dmf_points=3"]).is_err());
    for wrong in [
        "dmf_size=3",
        "dmf_points=0",
        "dmf_beta=-1",
        "dmf_moves=maybe",
    ] {
        assert!(
            config_from_cli(&[&input, "-os", "gfn0xtb", "-dmf", "-neb_opt", wrong]).is_err(),
            "{wrong}"
        );
    }
    let given = config_from_cli(&[
        &input,
        "-os",
        "gfn0xtb",
        "-dmf",
        "-ns",
        "7",
        "-neb_opt",
        "dmf_points=5",
        "dmf_guards=0",
        "dmf_moves=off",
        "dmf_beta=20",
    ])
    .unwrap();
    let dmf = given.dmf.unwrap();
    assert_eq!((dmf.points, dmf.guards, dmf.max_iterations), (5, 0, 7));
    assert!(dmf.moves.is_none());
    assert!((dmf.beta - 20.0 * Units::HARTREE_TO_EV).abs() < 1e-9);
    // What belongs to chains of images.
    for more in [
        &["-cineb", "3"][..],
        &["-coord", "dic"],
        &["-ewbneb"],
        &["-aneb", "2", "3"],
    ] {
        let mut arguments = vec![input.as_str(), "-os", "gfn0xtb", "-dmf"];
        arguments.extend_from_slice(more);
        let config = config_from_cli(&arguments).unwrap();
        assert!(
            run_with(&config, &dir, None, &registry).is_err(),
            "{more:?}"
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}
