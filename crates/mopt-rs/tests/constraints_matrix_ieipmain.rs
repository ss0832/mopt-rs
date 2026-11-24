//! The constraints of `ieipmain` for every method: {frozen atoms, a distance, the distance of
//! the centroids of two fragments, an angle, a dihedral, a user-defined function, `-gfix`} x
//! {iEIP, the dimer, the spring pair, DS-AFIR, 2PSHS, ADDF, GNT, GSM, SE-GSM, FSM, LMM} x {an
//! isolated cluster, a periodic fixed cell, ONIOM microiterations (LJ:LJ)}. Every method holds
//! every constraint: every frame of every file it writes is on it (`|c(x) - c(input)| <=
//! 1e-10` Bohr or rad, the precision of the 12 decimals of Angstrom of the files; frozen
//! atoms where they were in the input to those decimals). The optimization after the search
//! (`-ieip_opt optimize=on`) holds the constraints as well.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::path::{Path, PathBuf};

use mopt_rs::cli::constraints::{parse_projection_constraints, user_constraint};
use mopt_rs::cli::ieipmain::{config_from_cli, run_with};
use mopt_rs::domain::{AtomId, MolecularSystem, SpinCharge};
use mopt_rs::io::xyz::{read_xyz_file, read_xyz_trajectory_file};
use mopt_rs::pes::constraint::{Constraint, ConstraintKind, ConstraintSet};
use mopt_rs::pes::registry::CalculatorRegistry;

const FIXED: &str = "Ar 0.000 0.000 0.000\nAr 3.800 0.000 0.000\nAr 1.900 3.291 0.000\n";

/// Five argon atoms: a triangle (atoms 1-3), an atom over it or under it (4), one beside it (5).
fn cluster(second: bool) -> String {
    let moving = if second {
        "Ar 2.000 1.150 -3.000\n"
    } else {
        "Ar 1.700 0.950 3.000\n"
    };
    format!("5\nend\n{FIXED}{moving}Ar -2.900 1.500 0.400\n")
}

/// A 2 x 2 x 2 fcc argon crystal with a vacancy at its last site (31 atoms), every atom
/// shifted a little; `second`: the last atom has hopped into the vacancy.
fn fcc(second: bool) -> String {
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
    let vacancy = rows.pop().unwrap_or([0.0; 3]);
    if second {
        let last = rows.len() - 1;
        rows[last] = vacancy;
    }
    let mut text = format!(
        "{}\nLattice=\"{l} 0.0 0.0 0.0 {l} 0.0 0.0 0.0 {l}\" pbc=\"T T T\"\n",
        rows.len(),
        l = 2.0 * a
    );
    for r in &rows {
        text.push_str(&format!("Ar {:.10} {:.10} {:.10}\n", r[0], r[1], r[2]));
    }
    text
}

/// `text` optimized with atoms 1-3 fixed (on `extra`): the methods that start from a minimum
/// (2PSHS, ADDF) need one on the surface they search, which the other methods do not mind.
fn optimized(dir: &Path, name: &str, text: &str, extra: &[&str]) -> String {
    let file = dir.join(format!("{name}_raw.xyz"));
    std::fs::write(&file, text).unwrap();
    let mut args = vec![
        file.to_str().unwrap(),
        "-os",
        "lj",
        "-fix",
        "1-3",
        "-ns",
        "500",
        "-tcc",
    ];
    args.extend_from_slice(extra);
    let config = mopt_rs::cli::optmain::config_from_cli(&args).unwrap();
    let out = mopt_rs::cli::optmain::run_with(&config, dir, None, &CalculatorRegistry::native())
        .unwrap()
        .remove(0);
    assert!(out.converged, "{name}: the start is not optimized");
    std::fs::read_to_string(&out.optimized).unwrap()
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("mopt_cieip_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn kinds() -> Vec<(&'static str, Vec<String>)> {
    kinds_with_dihedral("3,1,2,5")
}

/// The kinds with the dihedral on `dihedral`.
fn kinds_with_dihedral(dihedral: &str) -> Vec<(&'static str, Vec<String>)> {
    let s = |v: &[&str]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>();
    vec![
        // Three atoms, so that no rigid motion is left on the surface: 2PSHS and ADDF start
        // from a minimum without zero modes.
        ("frozen", s(&["-fix", "1-3"])),
        ("distance", s(&["-pc", "bond", "1,2"])),
        // The distance between the centroids of atoms 1-2 and 3-4, whose atoms stay free.
        ("fbond", s(&["-pc", "fbond", "1,2,3,4,2"])),
        ("angle", s(&["-pc", "angle", "3,1,2"])),
        // Atom 5 beside the triangle in both structures: a dihedral with atom 4 changes sign
        // between the structures, and the second one put onto the value of the first is far
        // off. Under ONIOM atom 5 is the environment (a constrained atom would join the
        // core and leave none), and the dihedral is that of atom 4.
        ("dihedral", s(&["-pc", "dihedral", dihedral])),
        ("user", s(&["-uc", "dist(1,3) - dist(2,3)"])),
        ("gfix", s(&["-gfix", "1,2,3"])),
    ]
}

/// The methods with their options; every one holds every kind of constraint. `isolated`: the
/// driving coordinate of SE-GSM is the distance of atoms 4 and 5 (atom 5 is the environment
/// under ONIOM, and in the crystal the growth along it ends at once); else that of 1 and 4.
fn methods(isolated: bool) -> Vec<(&'static str, Vec<&'static str>)> {
    let driving = if isolated { "BREAK 4 5" } else { "BREAK 1 4" };
    vec![
        ("ieip", vec!["-ns", "15"]),
        ("dimer", vec!["-use_dimer", "-dimer_maxiter", "6"]),
        ("spm", vec!["-use_spm", "-ns", "4"]),
        ("dsafir", vec!["-dsafir", "-ns", "15"]),
        // Small spheres: with -gfix the second structure turned onto the first (the atom over
        // the triangle reached by turning the whole cluster) lies 1e-3 from it in the scaled
        // coordinates, within the default step.
        (
            "2pshs",
            vec!["-2pshs", "-2pshs_num", "3", "-2pshs_step", "0.002"],
        ),
        ("addf", vec!["-addf", "-addf_num", "3"]),
        ("gnt", vec!["-gnt", "-ns", "4"]),
        ("gsm", vec!["-gsm", "-ns", "4"]),
        ("se_gsm", vec!["-se_gsm", driving, "-ns", "4"]),
        ("fsm", vec!["-fsm", "-ns", "4"]),
        ("lmm", vec!["-lmm", "-ns", "4"]),
    ]
}

/// The constraints of `options` resolved at `input`, and the frozen atoms.
fn held(options: &[String], input: &MolecularSystem) -> (ConstraintSet, Vec<usize>) {
    let atoms = |t: &str| -> Vec<usize> { t.split(',').map(|a| a.parse().unwrap()).collect() };
    let (c, frozen): (Vec<Constraint>, Vec<usize>) = match options[0].as_str() {
        "-fix" => {
            let (first, last) = options[1].split_once('-').unwrap();
            let list: Vec<usize> =
                (first.parse::<usize>().unwrap() - 1..last.parse::<usize>().unwrap()).collect();
            (
                vec![Constraint::at_start(ConstraintKind::Frozen(
                    list.iter().map(|&a| AtomId::new(a)).collect(),
                ))],
                list,
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
            } else if path.extension().is_some_and(|e| e == "xyz")
                // Directions of modes, not structures.
                && !path.to_string_lossy().contains("eigenmodes")
            {
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

/// The bound on a residual recomputed from the files: the searches write 12 decimals of
/// Angstrom (rounding of up to 9.4e-13 Bohr per coordinate, which moves a distance or an
/// angle of a few Bohr by up to about 3e-12); the points are on the constraints to the
/// rounding floor of the restoration, and a drift above this bound would show.
/// (The guarantee every hard constraint is held to, from the files as well.)
const FROM_FILES: f64 = 1e-10;
/// The bound on a frozen coordinate read from a file: the rounding of its 12 decimals.
const FROZEN_IN_FILES: f64 = 1e-12;

fn check_frames(
    cell: &str,
    frames: &[(PathBuf, MolecularSystem)],
    options: &[String],
    input: &MolecularSystem,
) {
    assert!(!frames.is_empty(), "{cell}: no frames");
    let (set, frozen) = held(options, input);
    for (path, sys) in frames {
        if frozen.is_empty() {
            let r = set.residuals(sys).unwrap();
            assert!(
                r.iter().all(|v| v.abs() <= FROM_FILES),
                "{cell} {}: {r:?}",
                path.display()
            );
        }
        for &atom in &frozen {
            let id = AtomId::new(atom);
            let (p, q) = (sys.position(id).value(), input.position(id).value());
            for c in 0..3 {
                assert!(
                    (p[c] - q[c]).abs() <= FROZEN_IN_FILES,
                    "{cell} {}: {p:?} {q:?}",
                    path.display()
                );
            }
        }
    }
}

fn context(name: &str, first: &str, second: &str, extra: &[&str]) {
    let dir = scratch(name);
    let starts = scratch(&format!("{name}_starts"));
    let (a, b) = (dir.join("a.xyz"), dir.join("b.xyz"));
    std::fs::write(&a, optimized(&starts, "a", first, extra)).unwrap();
    std::fs::write(&b, optimized(&starts, "b", second, extra)).unwrap();
    let _ = std::fs::remove_dir_all(&starts);
    let input = read_xyz_file(&a, SpinCharge::default(), None).unwrap();
    let registry = CalculatorRegistry::native();
    let kinds = if extra.contains(&"-oniom") {
        kinds_with_dihedral("3,1,2,4")
    } else {
        kinds()
    };
    for (method, method_options) in methods(name == "isolated") {
        for (kind, options) in kinds.clone() {
            let cell = format!("{name} x {method} x {kind}");
            let mut args: Vec<String> = vec![
                a.to_string_lossy().into_owned(),
                b.to_string_lossy().into_owned(),
                "-os".into(),
                "lj".into(),
                "-ieip_opt".into(),
                "figures=off".into(),
            ];
            args.extend(extra.iter().map(|s| s.to_string()));
            args.extend(method_options.iter().map(|s| s.to_string()));
            args.extend(options.iter().cloned());
            let argv: Vec<&str> = args.iter().map(String::as_str).collect();
            let config = config_from_cli(&argv).unwrap();
            let found = run_with(&config, &dir, None, &registry);
            let out = found.unwrap_or_else(|e| panic!("{cell}: {e}"));
            let frames = all_frames(&out.directory, input.n_atoms());
            check_frames(&cell, &frames, &options, &input);
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn constraints_of_the_methods_on_an_isolated_cluster() {
    context("isolated", &cluster(false), &cluster(true), &[]);
}

#[test]
fn constraints_of_the_methods_in_a_periodic_cell() {
    context("periodic", &fcc(false), &fcc(true), &[]);
}

#[test]
fn constraints_of_the_methods_with_oniom_microiterations() {
    context(
        "oniom",
        &cluster(false),
        &cluster(true),
        &["-oniom", "1-4", "none", "lj"],
    );
}

/// The optimization after the search holds the constraints of the search (`-gfix` among
/// them), at the values of the first structure.
#[test]
fn the_optimization_after_the_search_holds_the_constraints() {
    let dir = scratch("followup");
    let (a, b) = (dir.join("a.xyz"), dir.join("b.xyz"));
    std::fs::write(&a, cluster(false)).unwrap();
    std::fs::write(&b, cluster(true)).unwrap();
    let input = read_xyz_file(&a, SpinCharge::default(), None).unwrap();
    for (kind, options) in kinds() {
        let mut args: Vec<String> = [
            a.to_str().unwrap(),
            b.to_str().unwrap(),
            "-os",
            "lj",
            "-ns",
            "15",
            "-ieip_opt",
            "figures=off",
            "optimize=on",
            "optimize_check=off",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        args.extend(options.iter().cloned());
        let argv: Vec<&str> = args.iter().map(String::as_str).collect();
        let config = config_from_cli(&argv).unwrap();
        let out = run_with(&config, &dir, None, &CalculatorRegistry::native()).unwrap();
        assert!(!out.optimized.is_empty(), "{kind}: no optimization");
        let frames = all_frames(&out.optimized[0].directory, input.n_atoms());
        assert!(frames.len() > 1, "{kind}: {} frames", frames.len());
        check_frames(&format!("followup x {kind}"), &frames, &options, &input);
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// Constraints that need a Hessian, and terms that have no meaning without accepted
/// evaluations, are refused with the reason.
#[test]
fn refused_combinations_name_their_reason() {
    let dir = scratch("refused");
    let (a, b) = (dir.join("a.xyz"), dir.join("b.xyz"));
    std::fs::write(&a, cluster(false)).unwrap();
    std::fs::write(&b, cluster(true)).unwrap();
    let (a, b) = (
        a.to_string_lossy().into_owned(),
        b.to_string_lossy().into_owned(),
    );
    for (extra, expected) in [
        (vec!["-pc", "eigvec", "0"], "eigvec"),
        (vec!["-metad", "bond", "1.0", "0.2", "1,2"], "keeps state"),
        (
            vec!["-nrp", "4", "8", "0.1", "0.1", "1", "1"],
            "simulated time",
        ),
        // A driving coordinate of SE-GSM that the constraints hold cannot drive it.
        (
            vec!["-se_gsm", "BREAK 1 2", "-pc", "bond", "1,2"],
            "is held by the constraints",
        ),
    ] {
        let mut args = vec![a.as_str(), b.as_str(), "-os", "lj", "-ns", "3"];
        args.extend(extra.iter().copied());
        let config = config_from_cli(&args).unwrap();
        let err = run_with(&config, &dir, None, &CalculatorRegistry::native())
            .unwrap_err()
            .to_string();
        assert!(err.contains(expected), "{extra:?}: {err}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}
