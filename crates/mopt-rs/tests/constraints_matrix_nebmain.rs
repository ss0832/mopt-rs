//! The constraints of `nebmain` in every kind of run: {frozen atoms, a distance, an angle, a
//! user-defined function, `-gfix`} x {an isolated cluster, a periodic fixed cell, ONIOM
//! microiterations (LJ:LJ)} x {the default band, climbing images, optimizations after the
//! chain}. Every cell holds a value assertion on every frame the run writes (all images of
//! every iteration, the files of the optimizations after the chain): `|c(x) - c(input)| <=
//! 1e-8` Bohr or rad, frozen atoms where they were in the input. The constrained atoms are
//! the same in the two end structures, so that every image of the first chain (straight
//! lines) has the values of the input.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::path::{Path, PathBuf};

use mopt_rs::cli::constraints::{parse_projection_constraints, user_constraint};
use mopt_rs::cli::nebmain::{config_from_cli, run_with, run_with_observers, NebmainResult};
use mopt_rs::domain::{AtomId, MolecularSystem, SpinCharge};
use mopt_rs::io::xyz::{read_xyz_file, read_xyz_trajectory_file};
use mopt_rs::path::band::machine::{ChainRelaxation, ChainReport};
use mopt_rs::pes::constraint::{Constraint, ConstraintKind, ConstraintSet};
use mopt_rs::pes::registry::CalculatorRegistry;
use mopt_rs::workflow::chain::Flow;

/// Five argon atoms: a triangle (atoms 1-3), an atom that moves over its edge (4), and one
/// beside it (5), at the two ends of the chain.
fn cluster() -> String {
    let fixed = "Ar 0.000 0.000 0.000\nAr 3.800 0.000 0.000\nAr 1.900 3.291 0.000\n";
    format!(
        "5\nA\n{fixed}Ar 1.900 1.097 3.100\nAr -2.900 1.500 0.400\n\
         5\nB\n{fixed}Ar 1.900 -2.400 2.300\nAr -2.900 1.500 0.400\n"
    )
}

/// A 2 x 2 x 2 fcc argon crystal (32 atoms), every atom shifted a little; the last atom
/// 0.4 Angstrom to one side or the other at the two ends.
fn fcc() -> String {
    let a = 5.40;
    let mut rows = Vec::new();
    for i in 0..2 {
        for j in 0..2 {
            for k in 0..2 {
                for f in [
                    [0.0, 0.0, 0.0],
                    [0.5, 0.5, 0.0],
                    [0.5, 0.0, 0.5],
                    [0.0, 0.5, 0.5],
                ] {
                    let n = rows.len() as f64;
                    let shift = [
                        0.07 * (1.3 * n).sin(),
                        0.06 * (0.7 * n).cos(),
                        0.05 * (2.1 * n).sin(),
                    ];
                    let cell = [f64::from(i), f64::from(j), f64::from(k)];
                    rows.push(std::array::from_fn::<f64, 3, _>(|c| {
                        a * (cell[c] + f[c]) + shift[c]
                    }));
                }
            }
        }
    }
    let mut text = String::new();
    for side in [0.4, -0.4] {
        text.push_str(&format!(
            "{}\nLattice=\"{l} 0.0 0.0 0.0 {l} 0.0 0.0 0.0 {l}\" pbc=\"T T T\"\n",
            rows.len(),
            l = 2.0 * a
        ));
        for (k, r) in rows.iter().enumerate() {
            let dx = if k + 1 == rows.len() { side } else { 0.0 };
            text.push_str(&format!("Ar {:.10} {:.10} {:.10}\n", r[0] + dx, r[1], r[2]));
        }
    }
    text
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("mopt_cneb_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// The kinds of constraints on atoms 1-3: (name, options).
fn kinds() -> Vec<(&'static str, Vec<String>)> {
    let s = |v: &[&str]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>();
    vec![
        ("frozen", s(&["-fix", "3"])),
        ("distance", s(&["-pc", "bond", "1,2"])),
        ("angle", s(&["-pc", "angle", "3,1,2"])),
        ("user", s(&["-uc", "dist(1,3) - dist(2,3)"])),
        ("gfix", s(&["-gfix", "1,2,3"])),
    ]
}

/// The constraints of `options` resolved at `input`, and the frozen atoms.
fn held(options: &[String], input: &MolecularSystem) -> (ConstraintSet, Vec<usize>) {
    let atoms = |t: &str| -> Vec<usize> { t.split(',').map(|a| a.parse().unwrap()).collect() };
    let (c, frozen): (Vec<Constraint>, Vec<usize>) = match options[0].as_str() {
        "-fix" => {
            let atom: usize = options[1].parse().unwrap();
            (
                vec![Constraint::at_start(ConstraintKind::Frozen(vec![
                    AtomId::new(atom - 1),
                ]))],
                vec![atom - 1],
            )
        }
        "-pc" => (
            parse_projection_constraints(&options[1..]).unwrap(),
            Vec::new(),
        ),
        "-gfix" => {
            let a = atoms(&options[1]);
            let mut c = Vec::new();
            for i in 0..a.len() {
                for j in i + 1..a.len() {
                    c.push(Constraint::at_start(ConstraintKind::Distance(
                        vec![AtomId::new(a[i] - 1)],
                        vec![AtomId::new(a[j] - 1)],
                    )));
                }
            }
            (c, Vec::new())
        }
        _ => (vec![user_constraint(&options[1]).unwrap()], Vec::new()),
    };
    (ConstraintSet::new(&c, input, None).unwrap(), frozen)
}

/// Every structure of every xyz file under `dir` (frames of `atoms` atoms).
fn all_frames(dir: &Path, atoms: usize) -> Vec<(PathBuf, MolecularSystem)> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for entry in std::fs::read_dir(&d).unwrap().flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|e| e == "xyz") {
                let Ok(traj) = read_xyz_trajectory_file(&path, SpinCharge::default(), None) else {
                    continue;
                };
                for f in traj.frames {
                    if f.system.n_atoms() == atoms {
                        out.push((path.clone(), f.system));
                    }
                }
            }
        }
    }
    out
}

/// The bound on a residual recomputed in memory: the rounding floor of evaluating a
/// distance or an angle of a few Bohr (about 1e-15) and of the Newton steps that restore
/// it, with a margin.
/// (The guarantee every hard constraint is held to: `pes::constraint::GUARANTEE`.)
const IN_MEMORY: f64 = 1e-10;
/// The bound on a residual recomputed from a file: 12 decimals of Angstrom are written
/// (rounding of up to 9.4e-13 Bohr per coordinate), which moves a distance or an angle of a
/// few Bohr by up to about 3e-12; a drift above that would show.
/// (The guarantee every hard constraint is held to, from the files as well.)
const FROM_FILES: f64 = 1e-10;
/// The bound on a frozen coordinate read from a file: the rounding of the 12 decimals.
const FROZEN_IN_FILES: f64 = 1e-12;

/// Every frame of `frames` holds the constraint of `options` at its value in `input`:
/// residuals below `bound`, frozen atoms within `frozen_bound` of the input (0: the same
/// numbers).
fn check_frames_within(
    cell: &str,
    frames: &[(PathBuf, MolecularSystem)],
    options: &[String],
    input: &MolecularSystem,
    bound: f64,
    frozen_bound: f64,
) {
    assert!(!frames.is_empty(), "{cell}: no frames");
    let (set, frozen) = held(options, input);
    for (path, sys) in frames {
        if frozen.is_empty() {
            let r = set.residuals(sys).unwrap();
            assert!(
                r.iter().all(|v| v.abs() <= bound),
                "{cell} {}: {r:?}",
                path.display()
            );
        }
        for &atom in &frozen {
            let id = AtomId::new(atom);
            let (p, q) = (sys.position(id).value(), input.position(id).value());
            for c in 0..3 {
                if frozen_bound == 0.0 {
                    assert_eq!(p[c].to_bits(), q[c].to_bits(), "{cell} {}", path.display());
                } else {
                    assert!(
                        (p[c] - q[c]).abs() <= frozen_bound,
                        "{cell} {}: {p:?} {q:?}",
                        path.display()
                    );
                }
            }
        }
    }
}

/// [`check_frames_within`] for frames read from files.
fn check_frames(
    cell: &str,
    frames: &[(PathBuf, MolecularSystem)],
    options: &[String],
    input: &MolecularSystem,
) {
    check_frames_within(cell, frames, options, input, FROM_FILES, FROZEN_IN_FILES);
}

/// The structures of the first `atoms` atoms of `input` at the points of `chain` (the core
/// of ONIOM microiterations is the first atoms here).
fn in_memory(
    input: &MolecularSystem,
    chains: &[Vec<faer::Col<f64>>],
) -> Vec<(PathBuf, MolecularSystem)> {
    let mut out = Vec::new();
    for (k, chain) in chains.iter().enumerate() {
        for (i, x) in chain.iter().enumerate() {
            let n = x.nrows() / 3;
            let positions: Vec<_> = (0..n).map(|a| input.position(AtomId::new(a))).collect();
            let template = MolecularSystem::new(
                input.elements[..n].to_vec(),
                &positions,
                input.spin_charge,
                input.cell.clone(),
            )
            .unwrap();
            out.push((
                PathBuf::from(format!("iteration {k} image {i} (in memory)")),
                template.with_flat_coords_bohr(x).unwrap(),
            ));
        }
    }
    out
}

/// The run, and every chain it evaluated and every chain it went on with, in memory.
fn run(dir: &Path, args: &[String]) -> (NebmainResult, Vec<Vec<faer::Col<f64>>>) {
    let argv: Vec<&str> = args.iter().map(String::as_str).collect();
    let config = config_from_cli(&argv).unwrap();
    let mut seen: Vec<Vec<faer::Col<f64>>> = Vec::new();
    let mut watch = |report: &ChainReport, machine: &mut ChainRelaxation| {
        seen.push(report.chain.clone());
        seen.push(machine.chain().to_vec());
        Ok(Flow::Continue)
    };
    let out = run_with_observers(
        &config,
        dir,
        None,
        &CalculatorRegistry::native(),
        &mut [&mut watch],
    )
    .unwrap();
    (out, seen)
}

/// The variants of the chain: (name, options, settings of `-neb_opt`).
fn variants() -> Vec<(&'static str, Vec<&'static str>, Vec<&'static str>)> {
    vec![
        ("band", vec![], vec![]),
        ("climbing", vec!["-cineb", "2"], vec![]),
        // The locally updated planes.
        ("lup", vec!["-lup"], vec![]),
        (
            "followup",
            vec![],
            vec!["optimize=minima", "optimize_check=off"],
        ),
    ]
}

fn context(name: &str, text: &str, extra: &[&str]) {
    let dir = scratch(name);
    let file = dir.join("ends.xyz");
    std::fs::write(&file, text).unwrap();
    let input = read_xyz_trajectory_file(&file, SpinCharge::default(), None)
        .unwrap()
        .frames
        .remove(0)
        .system;
    for (variant, options_of_variant, settings) in variants() {
        for (kind, options) in kinds() {
            let cell = format!("{name} x {variant} x {kind}");
            let mut args: Vec<String> = vec![
                file.to_string_lossy().into_owned(),
                "-os".into(),
                "lj".into(),
                "-ns".into(),
                "8".into(),
                "-neb_opt".into(),
                "path=linear".into(),
                "figures=off".into(),
            ];
            args.extend(settings.iter().map(|s| s.to_string()));
            args.extend(extra.iter().map(|s| s.to_string()));
            args.extend(options_of_variant.iter().map(|s| s.to_string()));
            args.extend(options.iter().cloned());
            let (out, chains) = run(&dir, &args);
            assert!(out.iterations > 0, "{cell}");
            // Every image of every iteration, at the rounding floor, frozen atoms the same
            // numbers as in the input.
            assert_eq!(chains.len(), 2 * out.iterations, "{cell}");
            check_frames_within(
                &cell,
                &in_memory(&input, &chains),
                &options,
                &input,
                IN_MEMORY,
                0.0,
            );
            let frames = all_frames(&out.directory, input.n_atoms());
            // Every image of every iteration at least.
            assert!(
                frames.len() > 3 * out.iterations,
                "{cell}: {} frames",
                frames.len()
            );
            check_frames(&cell, &frames, &options, &input);
            if variant == "followup" {
                assert!(!out.optimized.is_empty(), "{cell}: no optimizations");
                let optimized: Vec<(PathBuf, MolecularSystem)> = out
                    .optimized
                    .iter()
                    .map(|o| {
                        (
                            o.optimized.clone(),
                            read_xyz_file(&o.optimized, SpinCharge::default(), input.cell.as_ref())
                                .unwrap(),
                        )
                    })
                    .collect();
                check_frames(&cell, &optimized, &options, &input);
                // The frames of the optimizations are among those checked.
                assert!(
                    frames
                        .iter()
                        .any(|(p, _)| p.starts_with(&out.optimized[0].directory)),
                    "{cell}"
                );
            }
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn constraints_of_an_isolated_chain() {
    context("isolated", &cluster(), &[]);
}

#[test]
fn constraints_of_a_chain_in_a_periodic_cell() {
    context("periodic", &fcc(), &["-modelhess", "none"]);
}

#[test]
fn constraints_of_a_chain_with_oniom_microiterations() {
    context("oniom", &cluster(), &["-oniom", "1-4", "none", "lj"]);
}

#[test]
fn refused_combinations_name_their_reason() {
    let dir = scratch("refused");
    let file = dir.join("ends.xyz");
    std::fs::write(&file, cluster()).unwrap();
    let file = file.to_string_lossy().into_owned();
    for (extra, expected) in [
        (vec!["-pc", "eigvec", "0"], "eigvec"),
        (vec!["-dmf", "-pc", "bond", "1,2"], "direct MaxFlux"),
        (
            vec!["-coord", "dic", "-pc", "bond", "1,2"],
            "internal coordinates",
        ),
    ] {
        let mut args = vec![file.as_str(), "-os", "lj", "-ns", "3"];
        args.extend(extra.iter().copied());
        let config = config_from_cli(&args).unwrap();
        let err = run_with(&config, &dir, None, &CalculatorRegistry::native())
            .unwrap_err()
            .to_string();
        assert!(err.contains(expected), "{extra:?}: {err}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn an_adaptive_chain_keeps_the_constraints_of_its_added_images() {
    let dir = scratch("aneb");
    let file = dir.join("ends.xyz");
    std::fs::write(&file, cluster()).unwrap();
    let input = read_xyz_trajectory_file(&file, SpinCharge::default(), None)
        .unwrap()
        .frames
        .remove(0)
        .system;
    let options: Vec<String> = ["-pc", "bond", "1,2"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let mut args: Vec<String> = [
        file.to_str().unwrap(),
        "-os",
        "lj",
        "-ns",
        "8",
        "-aneb",
        "1",
        "2",
        "-neb_opt",
        "path=linear",
        "figures=off",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    args.extend(options.iter().cloned());
    let (out, chains) = run(&dir, &args);
    check_frames_within(
        "aneb",
        &in_memory(&input, &chains),
        &options,
        &input,
        IN_MEMORY,
        0.0,
    );
    let frames = all_frames(&out.directory, input.n_atoms());
    check_frames("aneb", &frames, &options, &input);
    let log = std::fs::read_to_string(out.directory.join("mopt.log")).unwrap();
    assert!(
        log.contains("were added around the maxima"),
        "no images were added"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// A force on a frozen atom is not a force of the chain: a chain whose frozen atom is pulled
/// by its neighbours reaches the tolerance (it did not, before the forces of the chain left
/// the fixed components out).
#[test]
fn a_force_on_a_frozen_atom_does_not_keep_the_chain_from_converging() {
    let dir = scratch("frozen_force");
    let file = dir.join("ends.xyz");
    std::fs::write(&file, cluster()).unwrap();
    let args: Vec<String> = [
        file.to_str().unwrap(),
        "-os",
        "lj",
        "-ns",
        "300",
        "-fix",
        "1",
        "-neb_opt",
        "path=linear",
        "figures=off",
        "tolerance=1e-3",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    let (out, _) = run(&dir, &args);
    assert_eq!(
        out.end,
        Some(mopt_rs::path::band::machine::ChainEnd::Converged),
        "{} iterations",
        out.iterations
    );
    // The largest force of the images left out the frozen atom, which carries one.
    let table = std::fs::read_to_string(out.directory.join("perp_max_gradient.csv")).unwrap();
    let last: Vec<f64> = table
        .lines()
        .last()
        .unwrap()
        .split(',')
        .map(|v| v.parse().unwrap())
        .collect();
    assert!(last.iter().all(|f| *f < 1e-3), "{last:?}");
    let _ = std::fs::remove_dir_all(&dir);
}
