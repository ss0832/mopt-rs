//! `nebmain -vc`: the chain of the atoms and the cell. A vacancy of copper hops to the next
//! site (31 atoms in 2 x 2 x 2 cells, the built-in EAM potential) between two ends relaxed
//! with `optmain -vc`: every image has a cell of its own, the cells of the images on the way
//! differ from those of the ends, the ends stay the minima they were, with a pressure the
//! profile is the enthalpy `E + P V`, the hard constraints hold on every image of every
//! iteration, every method of the chain runs on the cell, what cannot be done is refused with
//! its reason, the results do not depend on the number of threads, and a restart from the
//! journal gives the uninterrupted run.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use mopt_rs::cli::nebmain::{config_from_cli, run_with, run_with_observers, NebmainResult};
use mopt_rs::cli::optmain;
use mopt_rs::domain::{AtomId, EvalRequest, MolecularSystem, SpinCharge, Units};
use mopt_rs::io::xyz::{format_xyz_system, read_xyz_trajectory_file};
use mopt_rs::opt::cell::{gigapascal_per_atomic_unit, VariableCell, DEFAULT_LATTICE_WEIGHT};
use mopt_rs::path::band::machine::{ChainRelaxation, ChainReport};
use mopt_rs::pes::registry::{CalculatorOptions, CalculatorRegistry};
use mopt_rs::workflow::chain::Flow;

/// Lattice constant of copper (Angstrom).
const A: f64 = 3.615;

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("mopt_neb_vc_{name}_{}", std::process::id()));
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

/// The two ends of the hop, each relaxed with `optmain -vc` (atom 5 frozen in both, so that
/// it has the same fractional coordinates in the two ends), as one file of two frames.
fn ends() -> &'static str {
    static ENDS: OnceLock<String> = OnceLock::new();
    ENDS.get_or_init(|| {
        let dir = scratch("ends");
        let mut text = String::new();
        for (name, hopping) in [("a", [0.5 * A, 0.5 * A, 0.0]), ("b", [0.0, 0.0, 0.0])] {
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
            text.push_str(&format_xyz_system(
                &out[0].summary.as_ref().unwrap().system,
                None,
            ));
        }
        let _ = std::fs::remove_dir_all(&dir);
        text
    })
}

fn write_ends(dir: &Path) -> PathBuf {
    let file = dir.join("hop.xyz");
    std::fs::write(&file, ends()).unwrap();
    file
}

fn read_frames(path: &Path) -> Vec<MolecularSystem> {
    read_xyz_trajectory_file(path, SpinCharge::default(), None)
        .unwrap()
        .frames
        .into_iter()
        .map(|f| f.system)
        .collect()
}

fn volume(sys: &MolecularSystem) -> f64 {
    sys.active_cell().unwrap().volume_bohr3()
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

fn args(file: &Path, extra: &[&str]) -> Vec<String> {
    let mut all: Vec<String> = [
        file.to_str().unwrap(),
        "-os",
        "eam",
        "-vc",
        "-nd",
        "0",
        "-p",
        "5",
        "-neb_opt",
        "figures=off",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    all.extend(extra.iter().map(|s| s.to_string()));
    all
}

fn run(dir: &Path, all: &[String]) -> NebmainResult {
    let argv: Vec<&str> = all.iter().map(String::as_str).collect();
    let config = config_from_cli(&argv).unwrap();
    run_with(&config, dir, None, &CalculatorRegistry::native()).unwrap()
}

fn refused(dir: &Path, all: &[String]) -> String {
    let argv: Vec<&str> = all.iter().map(String::as_str).collect();
    match config_from_cli(&argv) {
        Err(e) => e.to_string(),
        Ok(config) => run_with(&config, dir, None, &CalculatorRegistry::native())
            .map(|_| "ran".to_string())
            .unwrap_or_else(|e| e.to_string()),
    }
}

/// The structures of the last chain of a run (with their cells).
fn last_chain(out: &NebmainResult) -> Vec<MolecularSystem> {
    read_frames(&out.candidates.as_ref().unwrap().path)
}

/// The largest change of a Cartesian coordinate or a lattice component (Bohr).
fn moved(a: &MolecularSystem, b: &MolecularSystem) -> f64 {
    let x = a.coords_flat_bohr() - b.coords_flat_bohr();
    let mut largest = x.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let (ca, cb) = (a.active_cell().unwrap(), b.active_cell().unwrap());
    for r in 0..3 {
        for c in 0..3 {
            largest = largest.max((ca.h_bohr[r][c] - cb.h_bohr[r][c]).abs());
        }
    }
    largest
}

#[test]
fn the_cells_of_the_images_change_along_the_hop_and_the_ends_stay_minima() {
    let dir = scratch("hop");
    let file = write_ends(&dir);
    let input = read_frames(&file);
    // Atom 5 is held as in the optimizations of the ends.
    let out = run(
        &dir,
        &args(&file, &["-modelhess", "none", "-ns", "15", "-fix", "5"]),
    );
    assert_eq!(out.iterations, 15);
    let chain = last_chain(&out);
    assert_eq!(chain.len(), 7);
    let ends = (volume(&input[0]), volume(&input[1]));
    let inner = chain[1..chain.len() - 1]
        .iter()
        .map(|s| (volume(s) - ends.0).abs())
        .fold(0.0_f64, f64::max);
    eprintln!(
        "ends {:.6} {:.6} Bohr^3, largest change of an inner image {inner:.6e}",
        ends.0, ends.1
    );
    // The two ends are the same crystal: the same volume.
    assert!((ends.0 - ends.1).abs() < 1e-3 * ends.0, "{ends:?}");
    // On the way the cell responds to the hopping atom.
    assert!(inner > 1e-3, "{inner}");
    // The ends were minima and stay where they were (force tolerance of the optimization).
    for (end, image) in [(&input[0], &chain[0]), (&input[1], &chain[6])] {
        let d = moved(end, image);
        eprintln!("an end moved by {d:.3e} Bohr");
        assert!(d < 2e-3, "an end moved by {d}");
    }
    // A barrier between the ends.
    let top = out
        .energies
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(top - out.energies[0] > 0.01, "{:?}", out.energies);
    // The lattice of every image of every iteration is in its file.
    for k in 0..out.iterations {
        let path = out
            .directory
            .join(format!("path_ITR_{k}_hop"))
            .join("hop_path.xyz");
        let frames = read_frames(&path);
        assert_eq!(frames.len(), 7);
        assert!(frames.iter().all(MolecularSystem::has_pbc));
    }
    let volumes = std::fs::read_to_string(out.directory.join("cell_volume.csv")).unwrap();
    assert_eq!(volumes.lines().count(), out.iterations);
    let log = std::fs::read_to_string(out.directory.join("mopt.log")).unwrap();
    assert!(log.contains("Variable cell (-vc)"), "{log}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn with_a_pressure_the_profile_is_the_enthalpy() {
    let dir = scratch("pressure");
    let file = write_ends(&dir);
    let gpa = 2.0;
    let out = run(
        &dir,
        &args(
            &file,
            &["-modelhess", "none", "-ns", "4", "-vc_pressure", "2.0"],
        ),
    );
    let pressure = gpa / gigapascal_per_atomic_unit();
    let chain = last_chain(&out);
    for (k, image) in chain.iter().enumerate() {
        let e = energy(image);
        let h = e + pressure * volume(image);
        assert!(
            (h - out.energies[k]).abs() < 1e-8,
            "image {k}: {h} {}",
            out.energies[k]
        );
        assert!(pressure * volume(image) > 1e-3);
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn hard_constraints_hold_on_every_image_of_every_iteration() {
    let dir = scratch("constraints");
    let file = write_ends(&dir);
    let input = read_frames(&file);
    let all = args(
        &file,
        &[
            "-modelhess",
            "none",
            "-ns",
            "6",
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
    );
    let argv: Vec<&str> = all.iter().map(String::as_str).collect();
    let config = config_from_cli(&argv).unwrap();
    let mut seen: Vec<Vec<faer::Col<f64>>> = Vec::new();
    let mut watch = |report: &ChainReport, machine: &mut ChainRelaxation| {
        seen.push(report.chain.clone());
        seen.push(machine.chain().to_vec());
        Ok(Flow::Continue)
    };
    let out = run_with_observers(
        &config,
        &dir,
        None,
        &CalculatorRegistry::native(),
        &mut [&mut watch],
    )
    .unwrap();
    assert_eq!(out.iterations, 6);
    let map = VariableCell::new(&input[0], DEFAULT_LATTICE_WEIGHT, 0.0).unwrap();
    let frozen = map.variables(&input[0]).unwrap();
    let (mut bond, mut vol) = (0.0_f64, 0.0_f64);
    for chain in &seen {
        for x in chain {
            let sys = map.system(x).unwrap();
            let cell = sys.active_cell().unwrap();
            let (p, q) = (
                sys.position(AtomId::new(1)).value(),
                sys.position(AtomId::new(2)).value(),
            );
            let d = cell.mic_delta(p, q);
            let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            bond = bond.max((r - 2.60 * Units::ANGSTROM_TO_BOHR).abs());
            vol = vol.max(
                (cell.volume_bohr3() * Units::BOHR_TO_ANGSTROM.powi(3) - 375.0).abs()
                    / Units::BOHR_TO_ANGSTROM.powi(3),
            );
            // Frozen: the quasi-Cartesian coordinates (fractional ones in the first cell) of
            // the first end, to the rounding of the files of the ends.
            for k in 12..15 {
                assert!((x[k] - frozen[k]).abs() < 1e-10, "frozen atom 5");
            }
        }
    }
    eprintln!("largest residuals: bond {bond:.2e} Bohr, volume {vol:.2e} Bohr^3");
    assert!(bond <= 1e-10 && vol <= 1e-10, "{bond} {vol}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn every_method_of_the_chain_runs_on_the_cell() {
    let dir = scratch("methods");
    let file = write_ends(&dir);
    let input = read_frames(&file);
    let methods: Vec<Vec<&str>> = vec![
        // The default steps with the Hessians of the enthalpy at the start.
        vec!["-ns", "2"],
        vec!["-modelhess", "none", "-ns", "3", "-cineb", "1"],
        vec!["-modelhess", "none", "-ns", "3", "-sd", "0"],
        vec!["-modelhess", "none", "-ns", "3", "-qsm"],
        vec!["-modelhess", "none", "-ns", "3", "-qsmv2"],
        vec!["-modelhess", "none", "-ns", "3", "-lup"],
        vec!["-modelhess", "none", "-ns", "3", "-ewbneb"],
        vec!["-modelhess", "none", "-ns", "3", "-aneb", "1", "2"],
        vec!["-modelhess", "none", "-ns", "3", "-ad", "1"],
        vec!["-modelhess", "none", "-ns", "3", "-neb_opt", "gp=on"],
        vec!["-ns", "2", "-gqnt"],
        vec![
            "-modelhess",
            "none",
            "-ns",
            "3",
            "-neb_opt",
            "optimize=minima",
            "optimize_check=off",
        ],
    ];
    for method in methods {
        let out = run(&dir, &args(&file, &method));
        assert!(out.iterations > 0, "{method:?}");
        let chain = last_chain(&out);
        let inner = chain[1..chain.len() - 1]
            .iter()
            .map(|s| (volume(s) - volume(&input[0])).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            inner > 1e-6,
            "{method:?}: the cell did not change ({inner})"
        );
        if method.contains(&"optimize=minima") {
            // The optimizations after the chain are those of the atoms and the cell.
            assert!(!out.optimized.is_empty());
            for o in &out.optimized {
                let log = std::fs::read_to_string(o.directory.join("mopt.log")).unwrap();
                assert!(log.contains("Variable cell"), "{method:?}");
            }
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn what_a_chain_of_the_cell_cannot_do_is_refused_with_its_reason() {
    let dir = scratch("refused");
    let file = write_ends(&dir);
    for (extra, expected) in [
        (vec!["-dmf"], "-dmf with -vc"),
        (vec!["-coord", "dic"], "-coord dic with -vc"),
        (vec!["-vc_afir", "100", "a"], "-vc_afir with nebmain"),
        (vec!["-neb_opt", "path=idpp"], "path=idpp with -vc"),
        (vec!["-oniom", "1-4", "none", "lj"], "-oniom with -vc"),
        (vec!["-pc", "x", "3"], "-vc with this constraint"),
        (
            vec!["-vc_fix", "a_vector,b_vector,c_vector"],
            "fix it completely",
        ),
    ] {
        let mut all = args(&file, &["-modelhess", "none", "-ns", "1"]);
        all.extend(extra.iter().map(|s| s.to_string()));
        let message = refused(&dir, &all);
        assert!(message.contains(expected), "{extra:?}: {message}");
    }
    // A structure without a cell, and plain variables.
    let molecule = dir.join("molecule.xyz");
    std::fs::write(
        &molecule,
        "2\nA\nAr 0 0 0\nAr 3.8 0 0\n2\nB\nAr 0 0 0\nAr 4.2 0 0\n",
    )
    .unwrap();
    let message = refused(
        &dir,
        &[molecule.to_str().unwrap(), "-os", "lj", "-vc"]
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>(),
    );
    assert!(message.contains("periodic cell"), "{message}");
    let (a, b) = (dir.join("a.txt"), dir.join("b.txt"));
    std::fs::write(&a, "x -0.558\ny 1.442\n").unwrap();
    std::fs::write(&b, "x 0.623\ny 0.028\n").unwrap();
    let message = refused(
        &dir,
        &[
            a.to_str().unwrap(),
            b.to_str().unwrap(),
            "-os",
            "mb2d",
            "-vc",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>(),
    );
    assert!(
        message.contains("plain variables have no cell"),
        "{message}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_chain_of_the_cell_does_not_depend_on_the_threads_and_restarts_from_its_journal() {
    let dir = scratch("threads");
    let file = write_ends(&dir);
    let tables = |out: &NebmainResult| {
        (
            std::fs::read_to_string(out.directory.join("energy_plot.csv")).unwrap(),
            std::fs::read_to_string(out.directory.join("cell_volume.csv")).unwrap(),
        )
    };
    let one = run(
        &dir,
        &args(&file, &["-modelhess", "none", "-ns", "6", "-core", "1"]),
    );
    let four = run(
        &dir,
        &args(&file, &["-modelhess", "none", "-ns", "6", "-core", "4"]),
    );
    assert_eq!(tables(&one), tables(&four));

    // Stopped after the third iteration and continued in its run directory.
    let all = args(&file, &["-modelhess", "none", "-ns", "6", "-core", "1"]);
    let argv: Vec<&str> = all.iter().map(String::as_str).collect();
    let config = config_from_cli(&argv).unwrap();
    let mut stop = |report: &ChainReport, _: &mut ChainRelaxation| {
        Ok(if report.iteration == 2 {
            Flow::Stop
        } else {
            Flow::Continue
        })
    };
    let registry = CalculatorRegistry::native();
    let part = run_with_observers(&config, &dir, None, &registry, &mut [&mut stop]).unwrap();
    assert_eq!(part.iterations, 3);
    let mut continued = config_from_cli(&argv).unwrap();
    continued.restart_dir = Some(part.directory.clone());
    let rest = run_with(&continued, &dir, None, &registry).unwrap();
    assert_eq!(rest.iterations, 6);
    assert_eq!(tables(&rest), tables(&one));
    let log = std::fs::read_to_string(rest.directory.join("mopt.log")).unwrap();
    assert!(log.contains("evaluations taken from"), "{log}");
    let _ = std::fs::remove_dir_all(&dir);
}
