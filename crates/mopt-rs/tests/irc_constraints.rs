//! The IRC within constraints: HCN <-> HNC with the built-in GFN0-xTB calculator, with the
//! carbon atom fixed (`-fix`), with the C-N distance held (`-pc bond`), and after a
//! constrained transition-state optimization (`optmain -irc` keeps its constraints).
//!
//! MultiOptPy: no counterpart (new in mopt-rs; MultiOptPy's IRC ignores constraints).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use mopt_rs::cli::ircmain::{self, IrcmainResult};
use mopt_rs::cli::optmain;
use mopt_rs::domain::{MolecularSystem, SpinCharge};
use mopt_rs::io::xyz::{read_xyz_file, read_xyz_trajectory_file};
use mopt_rs::irc::{BranchEnd, StartKind};
use mopt_rs::pes::registry::CalculatorRegistry;

/// A guess of the transition state: H bridging C and N.
const GUESS: &str = "3\n\nC 0.000 0.000 0.000\nN 1.190 0.000 0.000\nH 0.368 1.132 0.000\n";

fn scratch(name: &str) -> PathBuf {
    let base = std::env::temp_dir().join(format!("mopt_irc_kept_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&base);
    std::fs::create_dir_all(&base).unwrap();
    base
}

fn read(path: &Path) -> String {
    std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// The optimized transition state (one optimization for all tests).
fn transition_state() -> &'static PathBuf {
    static TS: OnceLock<PathBuf> = OnceLock::new();
    TS.get_or_init(|| {
        let base = scratch("ts");
        let guess = base.join("hcn.xyz");
        std::fs::write(&guess, GUESS).unwrap();
        let config = optmain::config_from_cli(&[
            guess.to_str().unwrap(),
            "-os",
            "gfn0xtb",
            "-opt",
            "rsirfo_bofill",
            "-order",
            "1",
            "-fc",
            "1",
            "-tcc",
        ])
        .unwrap();
        let out = optmain::run_with(&config, &base, None, &CalculatorRegistry::native()).unwrap();
        assert!(out[0].converged);
        out[0].optimized.clone()
    })
}

fn irc(base: &Path, arguments: &[&str]) -> IrcmainResult {
    let ts = base.join("ts.xyz");
    std::fs::copy(transition_state(), &ts).unwrap();
    let mut args = vec![ts.to_str().unwrap(), "-os", "gfn0xtb", "-irc_step", "0.1"];
    args.extend_from_slice(arguments);
    let config = ircmain::config_from_cli(&args).unwrap();
    ircmain::run_with(&config, base, None, &CalculatorRegistry::native())
        .unwrap()
        .remove(0)
}

fn distance(sys: &MolecularSystem, i: usize, j: usize) -> f64 {
    let x = sys.coords_flat_bohr();
    (0..3)
        .map(|k| (x[3 * i + k] - x[3 * j + k]).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// H-C and H-N distances (Bohr) of the end points, H on C first.
fn ends(run: &IrcmainResult) -> Vec<(f64, f64)> {
    let mut found: Vec<(f64, f64)> = run
        .irc
        .endpoints
        .iter()
        .map(|p| {
            let sys = read_xyz_file(p, SpinCharge::default(), None).unwrap();
            (distance(&sys, 2, 0), distance(&sys, 2, 1))
        })
        .collect();
    found.sort_by(|a, b| a.0.total_cmp(&b.0));
    found
}

fn path(run: &IrcmainResult) -> Vec<MolecularSystem> {
    read_xyz_trajectory_file(
        run.irc.combined_path.as_ref().unwrap(),
        SpinCharge::default(),
        None,
    )
    .unwrap()
    .frames
    .into_iter()
    .map(|f| f.system)
    .collect()
}

#[test]
fn the_path_keeps_a_fixed_atom_and_a_held_bond_and_reaches_both_minima() {
    let ts = read_xyz_file(transition_state(), SpinCharge::default(), None).unwrap();
    for (name, arguments) in [
        ("fixed", vec!["-fix", "1"]),
        ("bond", vec!["-pc", "bond", "1,2"]),
    ] {
        let base = scratch(name);
        let run = irc(&base, &arguments);
        assert_eq!(run.irc.kind, Some(StartKind::TransitionState), "{name}");
        let end = run.irc.end.as_ref().unwrap();
        assert!(
            end.branches.iter().all(|b| b.end == BranchEnd::Converged),
            "{name}: {:?}",
            end.branches
        );
        let log = read(&run.irc.directory.join("mopt.log"));
        assert!(log.contains("constraint(s) kept along the path"), "{name}");
        let structures = path(&run);
        assert!(structures.len() > 20, "{name}: {}", structures.len());
        for sys in &structures {
            match name {
                // The carbon atom does not move (the files have Angstrom with 12 digits).
                "fixed" => {
                    let (a, b) = (sys.coords_flat_bohr(), ts.coords_flat_bohr());
                    for k in 0..3 {
                        assert!((a[k] - b[k]).abs() < 1e-8, "{name}: {} {}", a[k], b[k]);
                    }
                }
                // The C-N distance is that of the transition state.
                _ => assert!(
                    (distance(sys, 0, 1) - distance(&ts, 0, 1)).abs() < 1e-7,
                    "{name}: {} {}",
                    distance(sys, 0, 1),
                    distance(&ts, 0, 1)
                ),
            }
        }
        // H on C (HCN) at one end, on N (HNC) at the other.
        let found = ends(&run);
        assert!(found[0].0 < 2.4 && found[0].1 > 3.7, "{name}: {found:?}");
        assert!(found[1].1 < 2.3 && found[1].0 > 3.7, "{name}: {found:?}");
        let _ = std::fs::remove_dir_all(&base);
    }
}

#[test]
fn optmain_irc_keeps_the_constraints_of_the_optimization() {
    let base = scratch("optmain");
    let guess = base.join("hcn.xyz");
    std::fs::write(&guess, GUESS).unwrap();
    let config = optmain::config_from_cli(&[
        guess.to_str().unwrap(),
        "-os",
        "gfn0xtb",
        "-opt",
        "rsirfo_bofill",
        "-order",
        "1",
        "-fc",
        "1",
        "-fix",
        "1",
        "-irc",
        "0.1",
        "300",
        "lqa",
    ])
    .unwrap();
    let out = optmain::run_with(&config, &base, None, &CalculatorRegistry::native()).unwrap();
    assert!(out[0].converged);
    let log = read(&out[0].directory.join("mopt.log"));
    assert!(log.contains("constraint(s) kept along the path"));
    // The carbon atom of the guess stays where it was through the optimization and the
    // path.
    let guess = read_xyz_file(&guess, SpinCharge::default(), None).unwrap();
    let combined = out[0].directory.join("irc_combined_path.xyz");
    let frames = read_xyz_trajectory_file(&combined, SpinCharge::default(), None)
        .unwrap()
        .frames;
    assert!(frames.len() > 20);
    for sys in frames.into_iter().map(|f| f.system) {
        let (a, b) = (sys.coords_flat_bohr(), guess.coords_flat_bohr());
        for k in 0..3 {
            assert!((a[k] - b[k]).abs() < 1e-8, "{} {}", a[k], b[k]);
        }
    }
    let _ = std::fs::remove_dir_all(&base);
}

#[test]
fn a_projection_for_crossings_is_refused() {
    let ts = transition_state().to_str().unwrap().to_string();
    for name in ["opt_meci", "opt_mesx"] {
        assert!(ircmain::config_from_cli(&[&ts, "-os", "gfn0xtb", "-mf", name]).is_err());
    }
    assert!(ircmain::config_from_cli(&[&ts, "-os", "gfn0xtb", "-mf", "seam"]).is_ok());
}
