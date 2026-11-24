//! `nebmain -ewbneb`: a chain with energy-weighted springs.
//!
//! MultiOptPy: `MEP/pathopt_ewbneb_force.py:CaluculationEWBNEB`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used)]

use std::path::{Path, PathBuf};

use mopt_rs::cli::nebmain::{config_from_cli, run_with};
use mopt_rs::domain::SpinCharge;
use mopt_rs::domain::Units;
use mopt_rs::io::xyz::read_xyz_trajectory_file;
use mopt_rs::path::band::machine::ChainEnd;
use mopt_rs::path::band::springs::EnergyWeightedSprings;
use mopt_rs::path::band::{path_lengths, ChainKind};
use mopt_rs::pes::registry::CalculatorRegistry;

const KCAL: f64 = Units::HARTREE_TO_KCAL_MOL;

/// Three rough structures from HCN over a bridged one to HNC.
const HCN: &str = "3\nHCN\nC 0.000 0.000 0.000\nN 1.156 0.000 0.000\nH -1.066 0.000 0.000\n\
3\nbridged\nC 0.000 0.000 0.000\nN 1.190 0.000 0.000\nH 0.368 1.132 0.000\n\
3\nHNC\nC 0.000 0.000 0.000\nN 1.170 0.000 0.000\nH 2.165 0.000 0.000\n";

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("mopt_neb_springs_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn file(dir: &Path, name: &str, text: &str) -> String {
    let p = dir.join(name);
    std::fs::write(&p, text).unwrap();
    p.to_string_lossy().into_owned()
}

#[test]
fn springs_keep_the_images_apart_and_closer_at_the_top() {
    let dir = scratch("hcn");
    let input = file(&dir, "hcn.xyz", HCN);
    let registry = CalculatorRegistry::native();

    // The settings: springs and no redistribution with -ewbneb, unless one is asked for;
    // MultiOptPy's constants with its form.
    let plain = config_from_cli(&[&input, "-os", "gfn0xtb"]).unwrap();
    assert!(plain.chain.springs.is_none() && plain.chain.predicted_maxima_every.is_some());
    let springs = config_from_cli(&[&input, "-os", "gfn0xtb", "-ewbneb"]).unwrap();
    assert_eq!(
        springs.chain.springs,
        Some(EnergyWeightedSprings::default())
    );
    assert_eq!(springs.chain.predicted_maxima_every, None);
    let both = config_from_cli(&[&input, "-os", "gfn0xtb", "-ewbneb", "-adpred", "2"]).unwrap();
    assert_eq!(both.chain.predicted_maxima_every, Some(2));
    let old = config_from_cli(&[
        &input,
        "-os",
        "gfn0xtb",
        "-ewbneb",
        "-neb_opt",
        "form=multioptpy",
    ])
    .unwrap();
    assert_eq!(
        old.chain.springs,
        Some(EnergyWeightedSprings::of_multioptpy())
    );
    let given = config_from_cli(&[
        &input,
        "-os",
        "gfn0xtb",
        "-ewbneb",
        "-neb_opt",
        "spring_upper=0.2",
        "spring_lower=0.02",
    ])
    .unwrap();
    assert_eq!(
        given.chain.springs,
        Some(EnergyWeightedSprings {
            upper: 0.2,
            lower: 0.02
        })
    );
    assert!(config_from_cli(&[&input, "-os", "gfn0xtb", "-neb_opt", "spring_upper=0.2"]).is_err());

    // The chain with springs and an image that climbs: the transition state, and the
    // images next to it nearer than those at the ends.
    let config = config_from_cli(&[
        &input,
        "-os",
        "gfn0xtb",
        "-ewbneb",
        "-cineb",
        "3",
        "-ns",
        "200",
        "-neb_opt",
        "figures=off",
    ])
    .unwrap();
    let result = run_with(&config, &dir, None, &registry).unwrap();
    let log = std::fs::read_to_string(result.directory.join("mopt.log")).unwrap();
    assert_eq!(result.end, Some(ChainEnd::Converged), "{log}");
    let top = result
        .energies
        .iter()
        .fold(f64::NEG_INFINITY, |m, e| m.max(*e));
    let barrier = KCAL * (top - result.energies[0]);
    println!("{} iterations, barrier {barrier:.3}", result.iterations);
    assert!((barrier - 68.285).abs() < 0.05, "{barrier}");
    let candidates = result.candidates.clone().unwrap();
    let path = read_xyz_trajectory_file(&candidates.path, SpinCharge::default(), None).unwrap();
    let chain: Vec<_> = path
        .frames
        .iter()
        .map(|frame| frame.system.coords_flat_bohr())
        .collect();
    let lengths = path_lengths(&chain, ChainKind::Atoms);
    let gaps: Vec<f64> = lengths.windows(2).map(|pair| pair[1] - pair[0]).collect();
    let highest = candidates.images[0];
    println!("gaps {gaps:?}, highest image {highest}");
    let at_the_top = gaps[highest - 1].max(gaps[highest]);
    let at_the_ends = gaps[0].min(gaps[gaps.len() - 1]);
    assert!(at_the_top < at_the_ends, "{gaps:?}");
    let _ = std::fs::remove_dir_all(&dir);
}
