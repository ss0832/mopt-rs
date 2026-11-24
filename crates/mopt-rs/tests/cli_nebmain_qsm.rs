//! `nebmain -qsm` and `-qsmv2` (MultiOptPy's quadratic string forms on the chain of
//! mopt-rs): what the two options set, HCN to HNC with the built-in GFN0-xTB calculator, and
//! what is refused.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used)]

use std::path::{Path, PathBuf};

use mopt_rs::cli::nebmain::{config_from_cli, run_with, NebmainResult};
use mopt_rs::domain::Units;
use mopt_rs::path::band::machine::{ChainEnd, ChainTangent};
use mopt_rs::pes::registry::CalculatorRegistry;

const KCAL: f64 = Units::HARTREE_TO_KCAL_MOL;
/// The saddle of HCN to HNC above HCN with GFN0-xTB (kcal/mol).
const BARRIER_KCAL: f64 = 68.285;

const HCN: &str = "3\nHCN\nC 0.000 0.000 0.000\nN 1.156 0.000 0.000\nH -1.066 0.000 0.000\n\
3\nbridged\nC 0.000 0.000 0.000\nN 1.190 0.000 0.000\nH 0.368 1.132 0.000\n\
3\nHNC\nC 0.000 0.000 0.000\nN 1.170 0.000 0.000\nH 2.165 0.000 0.000\n";

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("mopt_nebmain_qsm_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn path_file(dir: &Path) -> String {
    let p = dir.join("hcn.xyz");
    std::fs::write(&p, HCN).unwrap();
    p.to_string_lossy().into_owned()
}

fn run(base: &Path, arguments: &[&str]) -> NebmainResult {
    let config = config_from_cli(arguments).unwrap();
    run_with(&config, base, None, &CalculatorRegistry::native()).unwrap()
}

fn barrier(result: &NebmainResult) -> f64 {
    let top = result.energies.iter().copied().fold(f64::MIN, f64::max);
    KCAL * (top - result.energies[0])
}

#[test]
fn the_two_options_choose_the_direction_of_the_chain() {
    let dir = scratch("settings");
    let input = path_file(&dir);
    let config = |more: &[&str]| {
        let mut arguments = vec![input.as_str(), "-os", "gfn0xtb"];
        arguments.extend_from_slice(more);
        config_from_cli(&arguments)
    };
    let plain = config(&[]).unwrap();
    assert!(!plain.per_atom);
    assert_eq!(plain.chain.tangent, ChainTangent::HigherNeighbour);
    // -qsm: atom by atom; -qsmv2: carried from the highest image, for whole structures.
    let first = config(&["-qsm"]).unwrap();
    assert!(first.per_atom);
    assert_eq!(first.chain.tangent, ChainTangent::HigherNeighbour);
    let second = config(&["-qsmv2"]).unwrap();
    assert!(!second.per_atom);
    assert_eq!(second.chain.tangent, ChainTangent::CarriedFromTop);
    assert_eq!(
        config(&["-neb_opt", "tangent=top"]).unwrap().chain.tangent,
        ChainTangent::CarriedFromTop
    );
    // Everything else stays as it is without them.
    assert_eq!(second.chain.rfo.is_some(), plain.chain.rfo.is_some());
    assert_eq!(
        second.chain.predicted_maxima_every,
        plain.chain.predicted_maxima_every
    );
    // What does not go together.
    for wrong in [
        vec!["-qsm", "-qsmv2"],
        vec!["-qsmv2", "-neb_opt", "projection=atom"],
        vec!["-qsm", "-neb_opt", "projection=whole"],
        vec!["-qsm", "-neb_opt", "tangent=top"],
        vec!["-qsmv2", "-neb_opt", "form=multioptpy"],
        vec!["-qsm", "-neb_opt", "form=multioptpy"],
        vec!["-neb_opt", "tangent=sideways"],
    ] {
        assert!(config(&wrong).is_err(), "{wrong:?}");
    }
}

#[test]
fn hcn_to_hnc_with_the_two_options() {
    let dir = scratch("hcn");
    let input = path_file(&dir);
    let chain = |more: &[&str]| {
        let mut arguments = vec![
            input.as_str(),
            "-os",
            "gfn0xtb",
            "-ns",
            "40",
            "-neb_opt",
            "tolerance=1e-4",
        ];
        arguments.extend_from_slice(more);
        run(&dir, &arguments)
    };
    // The direction towards the higher neighbour, for comparison: the chain comes to rest.
    let higher = chain(&[]);
    assert_eq!(
        higher.end,
        Some(ChainEnd::Converged),
        "{}",
        higher.iterations
    );
    assert!((barrier(&higher) - BARRIER_KCAL).abs() < 0.05);

    // -qsm (atom by atom): comes to rest as well, in more iterations.
    let by_atom = chain(&["-qsm"]);
    assert_eq!(
        by_atom.end,
        Some(ChainEnd::Converged),
        "{}",
        by_atom.iterations
    );
    assert!(
        (barrier(&by_atom) - BARRIER_KCAL).abs() < 0.05,
        "{}",
        barrier(&by_atom)
    );
    assert!(by_atom.iterations >= higher.iterations);

    // -qsmv2 (carried from the highest image): the highest image is at the saddle and the
    // profile has its one maximum, but the chain does not come to rest within the
    // tolerance: the image next to an end keeps a force of about 1e-3 Hartree/Bohr, since
    // the carried tangent is not the direction of the chain there (docs/source/migration_paths.md).
    let carried = chain(&["-qsmv2"]);
    assert!(
        (barrier(&carried) - BARRIER_KCAL).abs() < 0.05,
        "{}",
        barrier(&carried)
    );
    assert_eq!(carried.end, Some(ChainEnd::IterationLimit));
    let top = carried
        .energies
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(k, _)| k)
        .unwrap();
    assert!(carried.energies[..=top].windows(2).all(|w| w[0] < w[1]));
    assert!(carried.energies[top..].windows(2).all(|w| w[0] > w[1]));
    assert!(carried.energies != higher.energies);
}
