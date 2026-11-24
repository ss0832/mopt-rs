//! `orientsearch` from the command line: the water dimer with the built-in GFN-FF, MultiOptPy's
//! `-part`, a restart from the journals and the STOP file, `-coord`, ONIOM (GFN0-xTB in
//! GFN-FF), and a molecule on a periodic slab (EMT).
//!
//! MultiOptPy: `Entrypoints/orientation_search.py:run_orientsearch`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::path::{Path, PathBuf};

use mopt_rs::cli::orientsearch::{config_from_cli, run_with};
use mopt_rs::domain::{AtomId, SpinCharge, Units};
use mopt_rs::io::xyz::read_xyz_file;
use mopt_rs::pes::registry::CalculatorRegistry;

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("mopt_orient_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

const WATER: [[f64; 3]; 3] = [
    [0.0, 0.0, 0.1173],
    [0.0, 0.7572, -0.4692],
    [0.0, -0.7572, -0.4692],
];

/// `n` water molecules 4 Angstrom apart along x.
fn waters(dir: &Path, n: usize) -> String {
    let mut text = format!("{}\nwaters\n", 3 * n);
    for k in 0..n {
        for (s, p) in ["O", "H", "H"].iter().zip(WATER) {
            text.push_str(&format!(
                "{s} {:.6} {:.6} {:.6}\n",
                p[0] + 4.0 * k as f64,
                p[1],
                p[2]
            ));
        }
    }
    let path = dir.join(format!("w{n}.xyz"));
    std::fs::write(&path, text).unwrap();
    path.to_string_lossy().into_owned()
}

fn args<'a>(input: &'a str, extra: &[&'a str]) -> Vec<&'a str> {
    let mut v = vec![input];
    v.extend_from_slice(extra);
    v
}

/// Shortest O-H distance between different molecules of a water cluster (Angstrom).
fn hydrogen_bond(file: &Path, n: usize) -> f64 {
    let sys = read_xyz_file(file, SpinCharge::default(), None).unwrap();
    let mut best = f64::INFINITY;
    for a in 0..n {
        for b in 0..n {
            if a != b {
                for h in [1, 2] {
                    let d = sys
                        .distance(AtomId::new(3 * a), AtomId::new(3 * b + h))
                        .value();
                    best = best.min(d * Units::BOHR_TO_ANGSTROM);
                }
            }
        }
    }
    best
}

#[test]
fn the_water_dimer_with_gfnff_finds_its_hydrogen_bond() {
    let dir = scratch("dimer");
    let input = waters(&dir, 2);
    let registry = CalculatorRegistry::native();
    let config = config_from_cli(&args(
        &input,
        &["-os", "gfnff", "-nsample", "4", "-orient_opt", "seed=7"],
    ))
    .unwrap();
    assert!(config.seed_given && config.search.samples == 4);
    let result = run_with(&config, &dir, None, &registry).unwrap();
    assert!(!result.interrupted && result.samples.len() == 4);
    assert!(result.samples.iter().all(|s| s.converged));
    let lowest = &result.minima[0];
    assert_eq!(lowest.rank, 1);
    let hb = hydrogen_bond(&lowest.file, 2);
    assert!((1.8..2.1).contains(&hb), "{hb}");
    // GFN-FF binds the dimer by 5 to 8 kcal/mol (the parts alone at their geometries in it).
    let e_int = lowest.interaction_kcal_mol.unwrap();
    assert!((-8.0..-5.0).contains(&e_int), "{e_int}");
    let weights: f64 = result.minima.iter().map(|m| m.weight.unwrap()).sum();
    assert!((weights - 1.0).abs() < 1e-9);
    assert_eq!(result.minima.iter().map(|m| m.hits).sum::<usize>(), 4);
    for name in [
        "orientsearch_minima.csv",
        "orientsearch_samples.csv",
        "w2_minima.xyz",
        "mopt.log",
        "settings.json",
        "w2_Sample_0.xyz",
    ] {
        assert!(result.directory.join(name).exists(), "{name}");
    }
    assert!(result
        .directory
        .join("w2_Sample_3_OPT")
        .join("w2_Sample_3_optimized.xyz")
        .exists());
    let log = std::fs::read_to_string(result.directory.join("mopt.log")).unwrap();
    assert!(log.contains("Number of Parts: 2"), "{log}");
    assert!(log.contains("Evaluations (orientsearch):"), "{log}");
    // The samples converge with the tight criteria unless the command line says otherwise.
    assert!(log.contains("with the tight criteria (-tcc)"), "{log}");
    let sample = std::fs::read_to_string(
        result
            .directory
            .join("w2_Sample_0_OPT")
            .join("settings.json"),
    )
    .unwrap();
    assert!(sample.contains("tight_convergence_criteria"), "{sample}");
    assert!(result.evaluations.gradients > 0);
    let normal = config_from_cli(&args(
        &input,
        &["-os", "gfnff", "-orient_opt", "criteria=normal"],
    ))
    .unwrap();
    assert!(!normal.parsed.get_bool("tight_convergence_criteria", false));
}

#[test]
fn multioptpy_parts_restart_and_stop() {
    let dir = scratch("restart");
    let input = waters(&dir, 3);
    let registry = CalculatorRegistry::native();
    // MultiOptPy's -part: the listed atoms are one part, the others one more.
    let config = config_from_cli(&args(
        &input,
        &[
            "-os",
            "gfnff",
            "-nsample",
            "2",
            "-part",
            "1-3",
            "-coord",
            "tric",
            "-orient_opt",
            "seed=3",
            "criteria=normal",
            "interaction=off",
        ],
    ))
    .unwrap();
    assert_eq!(config.part_lists, vec!["1-3".to_string()]);
    // A STOP file before the first sample: nothing is done, the run can be continued.
    let run_dir = dir.join("run");
    std::fs::create_dir_all(&run_dir).unwrap();
    std::fs::write(run_dir.join("STOP"), "").unwrap();
    let mut first = config.clone();
    first.restart_dir = Some(run_dir.clone());
    let stopped = run_with(&first, &dir, None, &registry).unwrap();
    assert!(stopped.interrupted && stopped.samples.is_empty());
    assert!(!run_dir.join("STOP").exists());
    let log = std::fs::read_to_string(run_dir.join("mopt.log")).unwrap();
    assert!(log.contains("Number of Parts: 2 (-part)"), "{log}");
    // Continued: both samples run.
    let done = run_with(&first, &dir, None, &registry).unwrap();
    assert!(!done.interrupted && done.samples.len() == 2);
    assert!(done.evaluations.gradients > 0);
    // Again from the same directory: every evaluation comes from the journals.
    let again = run_with(&first, &dir, None, &registry).unwrap();
    assert_eq!(again.evaluations.gradients, 0, "{:?}", again.evaluations);
    assert_eq!(again.minima.len(), done.minima.len());
    for (a, b) in again.minima.iter().zip(&done.minima) {
        assert!((a.energy - b.energy).abs() < 1e-12);
    }
    // The settings of the run reproduce it (the seed is in them).
    let settings = std::fs::read_to_string(run_dir.join("settings.json")).unwrap();
    assert!(settings.contains("seed=3"), "{settings}");
}

#[test]
fn oniom_gfn0xtb_in_gfnff_water() {
    let dir = scratch("oniom");
    let input = waters(&dir, 2);
    let registry = CalculatorRegistry::native();
    let config = config_from_cli(&args(
        &input,
        &[
            "-os",
            "gfn0xtb",
            "-oniom",
            "1-3",
            "none",
            "gfnff",
            "-nsample",
            "2",
            "-orient_opt",
            "seed=4",
            "criteria=normal",
        ],
    ))
    .unwrap();
    let result = run_with(&config, &dir, None, &registry).unwrap();
    assert!(!result.minima.is_empty());
    assert!(result.minima[0].interaction_kcal_mol.is_none());
    let log = std::fs::read_to_string(result.directory.join("mopt.log")).unwrap();
    assert!(
        log.contains("with -oniom the layers are given for the atoms of the whole complex"),
        "{log}"
    );
    let hb = hydrogen_bond(&result.minima[0].file, 2);
    assert!(hb < 2.4, "{hb}");
}

/// A copper slab of two layers of 3 x 3 atoms in a cell with 20 Angstrom along z, and a CO
/// molecule far above it.
fn slab(dir: &Path) -> String {
    let a = 2.55;
    let mut text = String::from("20\n");
    text.push_str(&format!(
        "Lattice=\"{l} 0.0 0.0 0.0 {l} 0.0 0.0 0.0 20.0\" pbc=\"T T T\"\n",
        l = 3.0 * a
    ));
    for layer in 0..2 {
        for i in 0..3 {
            for j in 0..3 {
                let s = if layer == 1 { 0.5 * a } else { 0.0 };
                text.push_str(&format!(
                    "Cu {:.6} {:.6} {:.6}\n",
                    a * f64::from(i) + s,
                    a * f64::from(j) + s,
                    1.8 * f64::from(layer)
                ));
            }
        }
    }
    text.push_str("C 1.0 1.0 9.0\nO 1.0 1.0 10.13\n");
    let path = dir.join("slab.xyz");
    std::fs::write(&path, text).unwrap();
    path.to_string_lossy().into_owned()
}

#[test]
fn a_molecule_comes_down_onto_a_periodic_slab() {
    let dir = scratch("slab");
    let input = slab(&dir);
    let registry = CalculatorRegistry::native();
    // The slab is bonded to its periodic images: it stays where it is while the samples are
    // drawn, and -fix keeps it there in the optimizations.
    let config = config_from_cli(&args(
        &input,
        &[
            "-os",
            "emt",
            "-nsample",
            "2",
            "-fix",
            "1-18",
            "-orient_opt",
            "seed=1",
            "criteria=normal",
        ],
    ))
    .unwrap();
    let result = run_with(&config, &dir, None, &registry).unwrap();
    let log = std::fs::read_to_string(result.directory.join("mopt.log")).unwrap();
    assert!(
        log.contains("bonded to its periodic images: stays"),
        "{log}"
    );
    assert!(log.contains("come down onto its free side"), "{log}");
    assert!(!result.minima.is_empty(), "{log}");
    let start = read_xyz_file(&input, SpinCharge::default(), None).unwrap();
    for m in &result.minima {
        let z_top = 1.8 * Units::ANGSTROM_TO_BOHR;
        for i in 18..20 {
            assert!(m.system.position(AtomId::new(i)).value()[2] > z_top);
        }
        // The frozen slab atoms are where they are in the input: the same numbers.
        for i in 0..18 {
            assert_eq!(
                m.system.position(AtomId::new(i)).value(),
                start.position(AtomId::new(i)).value(),
                "slab atom {}",
                i + 1
            );
        }
    }
    // Schemes that put centres at a distance are refused in a cell, with the reason.
    let refused = config_from_cli(&args(
        &input,
        &["-os", "emt", "-orient_opt", "scheme=multioptpy"],
    ))
    .unwrap();
    let e = run_with(&refused, &dir, None, &registry)
        .unwrap_err()
        .to_string();
    assert!(e.contains("periodic cell"), "{e}");
}

#[test]
fn a_bias_potential_holds_the_parts_in_every_sample() {
    let dir = scratch("bias");
    let input = waters(&dir, 2);
    let registry = CalculatorRegistry::native();
    // A stiff spring holds the two oxygen atoms 3.5 Angstrom apart.
    let config = config_from_cli(&args(
        &input,
        &[
            "-os",
            "gfnff",
            "-nsample",
            "2",
            "-kp",
            "1.0",
            "3.5",
            "1,4",
            "-orient_opt",
            "seed=2",
            "criteria=normal",
        ],
    ))
    .unwrap();
    let result = run_with(&config, &dir, None, &registry).unwrap();
    assert!(!result.minima.is_empty());
    for m in &result.minima {
        let d = m.system.distance(AtomId::new(0), AtomId::new(3)).value() * Units::BOHR_TO_ANGSTROM;
        assert!((d - 3.5).abs() < 0.05, "{d}");
    }
}

#[test]
fn settings_that_make_no_sense_are_refused() {
    let dir = scratch("refused");
    let one = dir.join("one.xyz");
    std::fs::write(
        &one,
        "3\nwater\nO 0 0 0.1173\nH 0 0.7572 -0.4692\nH 0 -0.7572 -0.4692\n",
    )
    .unwrap();
    let registry = CalculatorRegistry::native();
    let config = config_from_cli(&[one.to_str().unwrap(), "-os", "gfnff"]).unwrap();
    let e = run_with(&config, &dir, None, &registry)
        .unwrap_err()
        .to_string();
    assert!(e.contains("at least two parts"), "{e}");
    let two = waters(&dir, 2);
    assert!(config_from_cli(&[two.as_str(), "-os", "mb2d"]).is_err());
    assert!(
        config_from_cli(&[two.as_str(), "-os", "gfnff", "-orient_opt", "scheme=other"]).is_err()
    );
    assert!(config_from_cli(&[two.as_str(), two.as_str(), "-os", "gfnff"]).is_err());
}
