//! `run_autots` in a periodic cell with a search between two structures as step 1: the hop
//! of a vacancy in copper (31 atoms in 2 x 2 x 2 cells, the built-in EAM potential), from
//! the two ends to the transition state and, by its IRC, back to the two ends.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::{Path, PathBuf};

use mopt_rs::cli::autots::{run_with, AutotsConfig};
use mopt_rs::cli::option::parser;
use mopt_rs::cli::spec::RUN_AUTOTS;
use mopt_rs::domain::Units;
use mopt_rs::domain::{EvalRequest, SpinCharge};
use mopt_rs::io::xyz::{read_xyz_file, read_xyz_trajectory_file};
use mopt_rs::pes::registry::{CalculatorOptions, CalculatorRegistry};

const EV: f64 = Units::HARTREE_TO_EV;
/// Lattice constant of copper (Angstrom).
const A: f64 = 3.615;
/// The barrier of the hop with this potential and cell (eV), from the chain of
/// `tests/cli_nebmain_periodic.rs` and the transition-state optimization after it.
const BARRIER_EV: f64 = 0.5281;

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("mopt_autots_pbc_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// The 31 atoms of 2 x 2 x 2 cells of copper without the atom at the origin; the atom of
/// the site (A/2, A/2, 0) is the first and is put at `hopping`.
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
                        sites.push([
                            A * (cell[0] + f[0]),
                            A * (cell[1] + f[1]),
                            A * (cell[2] + f[2]),
                        ]);
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

/// The vacancy before and after the hop of the first atom into it.
fn two_ends() -> String {
    copper_with_a_vacancy([0.5 * A, 0.5 * A, 0.0]) + &copper_with_a_vacancy([0.0, 0.0, 0.0])
}

fn config(d: &Path, json: &str, more: &[&str]) -> AutotsConfig {
    let cfg = d.join("config.json");
    std::fs::write(&cfg, json).unwrap();
    let file = d.join("cu.xyz");
    std::fs::write(&file, two_ends()).unwrap();
    let mut all = vec![
        file.to_string_lossy().into_owned(),
        "-cfg".to_string(),
        cfg.to_string_lossy().into_owned(),
    ];
    all.extend(more.iter().map(|s| (*s).to_string()));
    let refs: Vec<&str> = all.iter().map(String::as_str).collect();
    AutotsConfig::from_args(&parser("run_autots", RUN_AUTOTS).parse(&refs).unwrap()).unwrap()
}

fn energy(path: &Path) -> f64 {
    let sys = read_xyz_file(path, SpinCharge::default(), None).unwrap();
    CalculatorRegistry::native()
        .create("eam", &CalculatorOptions::default(), &sys)
        .unwrap()
        .evaluate(&sys, &EvalRequest::ENERGY_ONLY)
        .unwrap()
        .energy
        .value()
}

#[test]
fn searches_between_two_structures_in_a_cell_give_a_first_path() {
    let d = scratch("pairs");
    let mut said = Vec::new();
    for method in [
        "gsm", "fsm", "ds-afir", "ieip", "2pshs", "gnt", "dimer", "spm", "lmm",
    ] {
        let _ = std::fs::remove_dir_all(d.join("work"));
        let c = config(
            &d,
            &format!(
                r#"{{
                  "work_dir": "work",
                  "step1_method": "{method}",
                  "step1_settings": {{"othersoft": "eam", "ieip_options": ["figures=off"]}},
                  "workflow": [{{"step": "step1"}}]
                }}"#
            ),
            &[],
        );
        match run_with(&c, &d, None, &CalculatorRegistry::native()) {
            Ok(result) => {
                let path = result.afir_path.clone().unwrap();
                let read = read_xyz_trajectory_file(&path, SpinCharge::default(), None).unwrap();
                let cells = read
                    .frames
                    .iter()
                    .filter(|f| f.system.cell.is_some())
                    .count();
                said.push(format!(
                    "PATH {method}: {} points, {cells} with the cell",
                    read.frames.len()
                ));
            }
            Err(error) => said.push(format!("FAILED {method}: {error}")),
        }
    }
    for line in &said {
        println!("{line}");
    }
    assert!(
        said.iter().all(|line| line.starts_with("PATH")),
        "{said:#?}"
    );
    // Every point of every path is a structure in the cell: the next step reads it as one.
    for line in &said {
        let counts: Vec<&str> = line
            .split_whitespace()
            .filter(|word| word.parse::<usize>().is_ok())
            .collect();
        assert!(counts.len() == 2 && counts[0] == counts[1], "{line}");
    }
}

#[test]
fn a_growing_string_in_a_cell_to_the_transition_state_and_the_ends_of_its_path() {
    let d = scratch("gsm");
    let c = config(
        &d,
        r#"{
          "work_dir": "work",
          "top_n_candidates": 1,
          "step1_method": "gsm",
          "step1_settings": {"othersoft": "eam", "ieip_options": ["figures=off"]},
          "step2_settings": {"othersoft": "eam", "NSTEP": 30, "neb_options": ["figures=off"]},
          "step3_settings": {"othersoft": "eam", "calc_exact_hess": 5,
                             "tight_convergence_criteria": true},
          "step4_settings": {"othersoft": "eam", "calc_exact_hess": 10,
                             "tight_convergence_criteria": true,
                             "intrinsic_reaction_coordinates": ["0.1", "300", "lqa"]}
        }"#,
        &["--run_step4"],
    );
    let result = run_with(&c, &d, None, &CalculatorRegistry::native()).unwrap();
    let states: Vec<_> = result.transition_states.iter().collect();
    assert_eq!(states.len(), 1, "{states:?}");
    assert!(states[0].converged && states[0].imaginary_modes == Some(1));
    // Against the relaxed vacancy before the hop.
    let reactant = d.join("reactant.xyz");
    std::fs::write(&reactant, copper_with_a_vacancy([0.5 * A, 0.5 * A, 0.0])).unwrap();
    let relaxed = mopt_rs::cli::optmain::run(
        &mopt_rs::cli::optmain::config_from_cli(&[
            reactant.to_string_lossy().as_ref(),
            "-os",
            "eam",
            "-tcc",
        ])
        .unwrap(),
        &d,
        None,
    )
    .unwrap();
    let minimum = energy(&relaxed[0].optimized);
    let barrier = (energy(&states[0].file) - minimum) * EV;
    println!("BARRIER {barrier:.4} eV");
    assert!((barrier - BARRIER_EV).abs() < 0.01, "{barrier}");
    // Step 4: the path from the transition state ends, on both sides, in the vacancy next
    // to an atom that has hopped or has not: the same minimum, read back with its cell.
    assert_eq!(states[0].ends.len(), 2, "{:?}", states[0].ends);
    for (file, written) in &states[0].ends {
        let end = energy(file);
        println!(
            "END {:.6} eV above the relaxed vacancy",
            (end - minimum) * EV
        );
        assert!((end - written).abs() < 1e-8, "{end} {written}");
        assert!(
            (end - minimum).abs() * EV < 1e-3,
            "{}",
            (end - minimum) * EV
        );
    }
}
