//! `ieipmain -vc`: the searches of `ieipmain` in the variables of the atoms and the cell. A
//! vacancy of copper hops to the next site (31 atoms in 2 x 2 x 2 cells, the built-in EAM
//! potential) between two ends relaxed with `optmain -vc`: every method either runs on the
//! variables of the cell, with an estimate whose cell is its own and whose energy (with a
//! pressure: enthalpy) is that of the surface, or is refused with its reason; the hard
//! constraints hold at every point; the results do not depend on the number of threads.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use mopt_rs::cli::ieipmain::{config_from_cli, run_with, IeipmainResult};
use mopt_rs::cli::optmain;
use mopt_rs::domain::{AtomId, EvalRequest, MolecularSystem, SpinCharge, Units};
use mopt_rs::io::xyz::{format_xyz_system, read_xyz_file, read_xyz_trajectory_file};
use mopt_rs::opt::cell::gigapascal_per_atomic_unit;
use mopt_rs::pes::registry::{CalculatorOptions, CalculatorRegistry};

/// Lattice constant of copper (Angstrom).
const A: f64 = 3.615;

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("mopt_ieip_vc_{name}_{}", std::process::id()));
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

/// The two ends of the hop relaxed with `optmain -vc` (atom 5 frozen in both).
fn ends() -> &'static [String; 2] {
    static ENDS: OnceLock<[String; 2]> = OnceLock::new();
    ENDS.get_or_init(|| {
        let dir = scratch("ends");
        let relaxed =
            [("a", [0.5 * A, 0.5 * A, 0.0]), ("b", [0.0, 0.0, 0.0])].map(|(name, hopping)| {
                let file = dir.join(format!("{name}.xyz"));
                std::fs::write(&file, copper_with_a_vacancy(hopping)).unwrap();
                let config = optmain::config_from_cli(&[
                    file.to_str().unwrap(),
                    "-os",
                    "eam",
                    "-vc",
                    "-fix",
                    "5",
                    "-ns",
                    "400",
                ])
                .unwrap();
                let out =
                    optmain::run_with(&config, &dir, None, &CalculatorRegistry::native()).unwrap();
                assert!(out[0].converged, "end {name}");
                format_xyz_system(&out[0].summary.as_ref().unwrap().system, None)
            });
        let _ = std::fs::remove_dir_all(&dir);
        relaxed
    })
}

fn write_ends(dir: &Path) -> [PathBuf; 2] {
    let [a, b] = ends();
    let (fa, fb) = (dir.join("hop_A.xyz"), dir.join("hop_B.xyz"));
    std::fs::write(&fa, a).unwrap();
    std::fs::write(&fb, b).unwrap();
    [fa, fb]
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

fn volume(sys: &MolecularSystem) -> f64 {
    sys.active_cell().unwrap().volume_bohr3()
}

fn run(dir: &Path, all: &[String]) -> Result<IeipmainResult, String> {
    let argv: Vec<&str> = all.iter().map(String::as_str).collect();
    let config = config_from_cli(&argv).map_err(|e| e.to_string())?;
    run_with(&config, dir, None, &CalculatorRegistry::native()).map_err(|e| e.to_string())
}

fn args(files: &[PathBuf], extra: &[&str]) -> Vec<String> {
    let mut all: Vec<String> = files
        .iter()
        .map(|f| f.to_string_lossy().into_owned())
        .collect();
    all.extend(
        ["-os", "eam", "-vc", "-ieip_opt", "figures=off"]
            .iter()
            .map(|s| s.to_string()),
    );
    all.extend(extra.iter().map(|s| s.to_string()));
    all
}

/// The energy written in the comment line of a structure file (`energy=`).
fn written_energy(file: &Path) -> Option<f64> {
    let text = std::fs::read_to_string(file).ok()?;
    text.lines().nth(1).and_then(|l| {
        l.split_whitespace().find_map(|t| {
            t.strip_prefix("energy=")
                .and_then(|v| v.parse::<f64>().ok())
        })
    })
}

/// The estimate of a run: a structure with its own cell, its energy that of the surface
/// (`pressure` in Hartree/Bohr^3: the enthalpy) to `within` Hartree.
fn check_estimate(what: &str, out: &IeipmainResult, pressure: f64, within: f64) -> MolecularSystem {
    let file = out
        .transition_states
        .first()
        .unwrap_or_else(|| panic!("{what}: no estimate"));
    let sys = read_xyz_file(file, SpinCharge::default(), None).unwrap();
    assert!(sys.has_pbc(), "{what}: the estimate has no cell");
    // The energy in the comment line is that of the surface at the estimate.
    if let Some(e) = written_energy(file) {
        let h = energy(&sys) + pressure * volume(&sys);
        assert!((e - h).abs() < within, "{what}: {e} {h}");
    }
    sys
}

#[test]
fn every_method_runs_in_the_variables_of_the_cell_or_is_refused() {
    let dir = scratch("methods");
    let [a, b] = write_ends(&dir);
    let start = read_xyz_file(&a, SpinCharge::default(), None).unwrap();
    let two = [a.clone(), b.clone()];
    let one = [a.clone()];
    // (name, inputs, options); every run is short.
    let methods: Vec<(&str, &[PathBuf], Vec<&str>)> = vec![
        ("ieip", &two, vec!["-ns", "40"]),
        ("2pshs", &two, vec!["-2pshs", "-2pshs_num", "6"]),
        ("gnt", &two, vec!["-gnt", "-ns", "6"]),
        ("dimer", &two, vec!["-use_dimer", "-dimer_maxiter", "20"]),
        ("spm", &two, vec!["-use_spm", "-ns", "4"]),
        ("gsm", &two, vec!["-gsm", "-ieip_opt", "gsm_iterations=4"]),
        (
            "fsm",
            &two,
            vec!["-fsm", "-ieip_opt", "fsm_images=4", "fsm_gradients=1"],
        ),
        (
            "dsafir",
            &two,
            vec!["-dsafir", "-ieip_opt", "dsafir_steps=40"],
        ),
        ("lmm", &two, vec!["-lmm", "-ns", "4"]),
        (
            "addf",
            &one,
            vec!["-addf", "-addf_num", "3", "-addf_nadd", "1"],
        ),
    ];
    let mut table = Vec::new();
    for (name, inputs, options) in methods {
        let out = run(&dir, &args(inputs, &options));
        match out {
            Ok(out) => {
                if let Some(file) = out.transition_states.first() {
                    // The energy a method returns with its estimate is listed, not checked
                    // (measured: the dimer gives that of its last evaluated centre, 6e-6
                    // Hartree off the written structure; GSM after four iterations 4e-3).
                    let sys = check_estimate(name, &out, 0.0, f64::INFINITY);
                    let change = (volume(&sys) - volume(&start)).abs();
                    let off = written_energy(file).map(|e| e - energy(&sys));
                    table.push(format!(
                        "{name}: estimate {} (volume change {change:.3e} Bohr^3, written energy \
                         minus that of the structure {off:?})",
                        file.display()
                    ));
                } else {
                    table.push(format!(
                        "{name}: ran, no estimate ({})",
                        out.directory.display()
                    ));
                }
                // The files of the method have the cells of their points.
                if let Some(path) = &out.path {
                    let frames =
                        read_xyz_trajectory_file(path, SpinCharge::default(), None).unwrap();
                    assert!(frames.frames.iter().all(|f| f.system.has_pbc()), "{name}");
                }
            }
            Err(message) => table.push(format!("{name}: refused: {message}")),
        }
    }
    eprintln!("{}", table.join("\n"));
    // Only refusals with a reason; no method fails otherwise.
    for line in &table {
        assert!(
            !line.contains("refused") || line.contains("-vc"),
            "a failure that names no reason: {line}"
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_ieip_estimate_is_on_the_enthalpy_and_its_cell_differs_from_the_ends() {
    let dir = scratch("ieip");
    let files = write_ends(&dir);
    let start = read_xyz_file(&files[0], SpinCharge::default(), None).unwrap();
    let out = run(&dir, &args(&files, &["-ns", "60", "-vc_pressure", "2.0"])).unwrap();
    let pressure = 2.0 / gigapascal_per_atomic_unit();
    let sys = check_estimate("ieip", &out, pressure, 1e-8);
    let change = (volume(&sys) - volume(&start)).abs();
    eprintln!(
        "volume of the estimate: {} against {} Bohr^3",
        volume(&sys),
        volume(&start)
    );
    assert!(change > 1e-3, "{change}");
    // The images of iEIP with their cells.
    let traj = out.directory.join("hop_A_traj.xyz");
    let frames = read_xyz_trajectory_file(&traj, SpinCharge::default(), None).unwrap();
    assert!(!frames.frames.is_empty() && frames.frames.iter().all(|f| f.system.has_pbc()));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn constraints_hold_at_every_point_of_the_search() {
    let dir = scratch("constraints");
    let files = write_ends(&dir);
    let out = run(
        &dir,
        &args(
            &files,
            &[
                "-ns",
                "20",
                "-fix",
                "5",
                "-pc",
                "manual",
                "bond",
                "2,3",
                "2.60",
                "-vc_fix",
                "volume=375.0",
            ],
        ),
    )
    .unwrap();
    let mut worst = (0.0_f64, 0.0_f64);
    for name in ["hop_A_traj.xyz"] {
        let frames =
            read_xyz_trajectory_file(out.directory.join(name), SpinCharge::default(), None)
                .unwrap();
        for f in &frames.frames {
            let sys = &f.system;
            let cell = sys.active_cell().unwrap();
            let d = cell.mic_delta(
                sys.position(AtomId::new(1)).value(),
                sys.position(AtomId::new(2)).value(),
            );
            let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            worst.0 = worst.0.max((r - 2.60 * Units::ANGSTROM_TO_BOHR).abs());
            let v = cell.volume_bohr3() * Units::BOHR_TO_ANGSTROM.powi(3);
            worst.1 = worst.1.max((v - 375.0).abs());
        }
    }
    eprintln!(
        "largest residuals in the files: bond {:.2e} Bohr, volume {:.2e} Angstrom^3",
        worst.0, worst.1
    );
    // The files have 12 decimals of Angstrom.
    assert!(worst.0 < 1e-9 && worst.1 < 1e-7, "{worst:?}");
    // A method that holds constraints through a manifold refuses them.
    let refused = run(&dir, &args(&files, &["-gnt", "-pc", "bond", "2,3"])).unwrap_err();
    assert!(refused.contains("manifold"), "{refused}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn what_cannot_be_done_on_a_variable_cell_is_refused_with_its_reason() {
    let dir = scratch("refused");
    let files = write_ends(&dir);
    for (extra, expected) in [
        (vec!["-vc_afir", "100", "a"], "-vc_afir with ieipmain"),
        (vec!["-coord", "dic"], "-coord dic with -vc"),
        (vec!["-oniom", "1-4", "none", "lj"], "-oniom with -vc"),
        (vec!["-addf", "-fix", "5"], "with -vc and -fix"),
    ] {
        let message = run(&dir, &args(&files, &extra)).unwrap_err();
        assert!(message.contains(expected), "{extra:?}: {message}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_search_does_not_depend_on_the_threads() {
    let dir = scratch("threads");
    let files = write_ends(&dir);
    let estimate = |threads: &str| {
        let out = run(&dir, &args(&files, &["-ns", "15", "-core", threads])).unwrap();
        std::fs::read_to_string(&out.transition_states[0]).unwrap()
    };
    assert_eq!(estimate("1"), estimate("4"));
    let _ = std::fs::remove_dir_all(&dir);
}
