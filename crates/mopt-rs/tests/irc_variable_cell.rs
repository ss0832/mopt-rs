//! The IRC of the atoms and the cell together (`-vc`): a vacancy of copper hops to the
//! next site with the built-in EAM potential; the saddle point of the atoms and the cell
//! from `optmain -vc -order 1`, then the IRC (`optmain -irc`, and `ircmain -vc` from the
//! saddle).
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::path::{Path, PathBuf};

use mopt_rs::cli::{ircmain, optmain};
use mopt_rs::domain::Units;
use mopt_rs::domain::{EvalRequest, MolecularSystem, SpinCharge};
use mopt_rs::io::xyz::read_xyz_file;
use mopt_rs::irc::{BranchEnd, StartKind};
use mopt_rs::pes::registry::{CalculatorOptions, CalculatorRegistry};

/// Lattice constant of copper (Angstrom).
const A: f64 = 3.615;

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("mopt_irc_vc_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// 2 x 2 x 2 cells of copper without the atom at the origin; the atom of the site
/// (A/2, A/2, 0) is the first and sits at `hopping`.
fn copper_with_a_vacancy(hopping: [f64; 3]) -> String {
    let mut sites = vec![hopping];
    for i in 0..2 {
        for j in 0..2 {
            for k in 0..2 {
                for f in [
                    [0.0, 0.0, 0.0],
                    [0.5, 0.5, 0.0],
                    [0.5, 0.0, 0.5],
                    [0.0, 0.5, 0.5],
                ] {
                    let cell = [f64::from(i), f64::from(j), f64::from(k)];
                    let empty = cell == [0.0; 3] && f == [0.0, 0.0, 0.0];
                    let hops = cell == [0.0; 3] && f == [0.5, 0.5, 0.0];
                    if !empty && !hops {
                        sites.push(std::array::from_fn(|c| A * (cell[c] + f[c])));
                    }
                }
            }
        }
    }
    let mut text = format!(
        "{}\nLattice=\"{l} 0.0 0.0 0.0 {l} 0.0 0.0 0.0 {l}\" pbc=\"T T T\"\n",
        sites.len(),
        l = 2.0 * A
    );
    for s in &sites {
        text.push_str(&format!("Cu {:.10} {:.10} {:.10}\n", s[0], s[1], s[2]));
    }
    text
}

fn energy(sys: &MolecularSystem) -> f64 {
    let registry = CalculatorRegistry::native();
    let calc = registry
        .create("eam", &CalculatorOptions::default(), sys)
        .unwrap();
    calc.evaluate(sys, &EvalRequest::ENERGY_ONLY)
        .unwrap()
        .energy
        .value()
}

fn read(path: &Path) -> MolecularSystem {
    read_xyz_file(path, SpinCharge::default(), None).unwrap()
}

#[test]
fn the_vacancy_hops_with_the_atoms_and_the_cell_together() {
    let dir = scratch("hop");
    // The hopping atom halfway between its site and the empty one: the saddle by symmetry.
    let guess = dir.join("guess.xyz");
    std::fs::write(&guess, copper_with_a_vacancy([0.25 * A, 0.25 * A, 0.0])).unwrap();
    let config = optmain::config_from_cli(&[
        guess.to_str().unwrap(),
        "-os",
        "eam",
        "-vc",
        "-opt",
        "rsirfo_bofill",
        "-order",
        "1",
        "-fc",
        "1000",
        "-irc",
        "0.3",
        "300",
        "lqa",
    ])
    .unwrap();
    let out = optmain::run_with(&config, &dir, None, &CalculatorRegistry::native()).unwrap();
    assert!(out[0].converged);
    let summary = out[0].summary.as_ref().unwrap();
    let saddle = energy(&summary.system);
    let log = std::fs::read_to_string(out[0].directory.join("mopt.log")).unwrap();
    assert!(log.contains("IRC of the atoms and the cell (-vc)"), "{log}");
    assert!(log.contains("Execute IRC"), "{log}");
    // Both sides end at minima (a copper atom is heavy: the mass-weighted path is long).
    assert_eq!(
        log.matches("Convergence reached. (IRC)").count(),
        2,
        "{log}"
    );
    // Two ends with their cells, below the saddle by the barrier of the hop, alike by
    // symmetry.
    assert_eq!(summary.irc_endpoints.len(), 2);
    let ends: Vec<MolecularSystem> = summary.irc_endpoints.iter().map(|p| read(p)).collect();
    let barriers: Vec<f64> = ends
        .iter()
        .map(|s| (saddle - energy(s)) * Units::HARTREE_TO_EV)
        .collect();
    println!("barriers (eV) {barriers:?}");
    for b in &barriers {
        assert!(*b > 0.3 && *b < 1.0, "{barriers:?}");
    }
    assert!((barriers[0] - barriers[1]).abs() < 2e-3, "{barriers:?}");
    // The lengths of the lattice vectors: the saddle of the atoms and the cell has a
    // distorted cell (the hop squeezes the plane it is in), the minima at the ends are
    // cubic again, and alike.
    let lengths = |sys: &MolecularSystem| -> [f64; 3] {
        let cell = sys.active_cell().unwrap();
        std::array::from_fn(|k| {
            let v = cell.lattice_vector(k).value();
            (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
        })
    };
    let at_saddle = lengths(&summary.system);
    println!("lattice at the saddle {at_saddle:?}");
    for end in &ends {
        let l = lengths(end);
        println!("lattice at an end {l:?}");
        assert!(
            (l[0] - l[1]).abs() < 1e-3 && (l[0] - l[2]).abs() < 1e-3,
            "{l:?}"
        );
        assert!(
            (0..3).any(|k| (l[k] - at_saddle[k]).abs() > 0.05),
            "{l:?} {at_saddle:?}"
        );
    }
    let (l0, l1) = (lengths(&ends[0]), lengths(&ends[1]));
    assert!(
        (0..3).all(|k| (l0[k] - l1[k]).abs() < 1e-4),
        "{l0:?} {l1:?}"
    );

    // ircmain -vc from the saddle: the same path.
    let ts = dir.join("ts.xyz");
    std::fs::copy(&out[0].optimized, &ts).unwrap();
    let config = ircmain::config_from_cli(&[
        ts.to_str().unwrap(),
        "-os",
        "eam",
        "-vc",
        "-irc_step",
        "0.3",
    ])
    .unwrap();
    let run = ircmain::run_with(&config, &dir, None, &CalculatorRegistry::native())
        .unwrap()
        .remove(0);
    assert_eq!(run.irc.kind, Some(StartKind::TransitionState));
    let end = run.irc.end.as_ref().unwrap();
    assert!(
        end.branches.iter().all(|b| b.end == BranchEnd::Converged),
        "{:?}",
        end.branches
    );
    assert_eq!(run.irc.endpoints.len(), 2);
    assert!(run
        .irc
        .directory
        .join("irc_forward_structures_cell.xyz")
        .is_file());
    // Plain variables, -mf, and constraints do not go with -vc.
    assert!(ircmain::config_from_cli(&[
        ts.to_str().unwrap(),
        "-os",
        "eam",
        "-vc",
        "-vc_afir",
        "100",
        "a"
    ])
    .is_err());
    let _ = std::fs::remove_dir_all(&dir);
}
