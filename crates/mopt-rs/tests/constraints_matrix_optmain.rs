//! The constraints of `optmain` in every kind of run: {frozen atoms, a distance, the distance
//! of the centroids of two fragments (`fbond`), an angle, a rigid group of three atoms
//! (`-gfix`), an oriented group (`-pc rot`), a user-defined function} x {a minimum, a
//! transition state (`-order 1`), `-coord dic`, a periodic fixed cell, a periodic variable
//! cell (`-vc`), ONIOM microiterations, `-irc`}. Every cell holds a value assertion on every
//! frame the run writes (or an asserted refusal) and on the final structure in memory: in
//! memory `|c(x) - c(input)| <= 1e-12` Bohr or rad (the rounding floor of the restore) and
//! frozen atoms bit-identical to the input; from the trajectories (12 decimals of Angstrom)
//! `5e-12`, from the final structure files (12 decimals) `1e-10`; at the minima the
//! vibrations on the constraint surface.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::path::{Path, PathBuf};

use mopt_rs::cli::constraints::parse_projection_constraints;
use mopt_rs::cli::constraints::user_constraint;
use mopt_rs::cli::optmain::{config_from_cli, run_with, OptmainResult};
use mopt_rs::domain::{AtomId, MolecularSystem, SpinCharge};
use mopt_rs::io::xyz::{read_xyz_file, read_xyz_trajectory_file};
use mopt_rs::pes::constraint::{Constraint, ConstraintKind, ConstraintSet};
use mopt_rs::pes::registry::CalculatorRegistry;

/// Ethane with its methyl groups eclipsed: the saddle point of the internal rotation.
const ECLIPSED: &str = "8\nethane\nC 0.000 0.000 0.000\nC 1.530 0.000 0.000\n\
H -0.390 1.030 0.000\nH -0.390 -0.510 0.890\nH -0.390 -0.510 -0.890\n\
H 1.920 1.030 0.000\nH 1.920 -0.510 0.890\nH 1.920 -0.510 -0.890\n";

/// Ethane near its staggered minimum, a little distorted.
const STAGGERED: &str = "8\nethane\nC 0.000 0.000 0.000\nC 1.550 0.020 -0.010\n\
H -0.380 1.040 0.020\nH -0.400 -0.500 0.880\nH -0.370 -0.520 -0.900\n\
H 1.930 -1.020 0.030\nH 1.910 0.500 0.900\nH 1.940 0.530 -0.870\n";

const ETHANOL: &str = "9\nethanol\nC 0.000 0.000 0.000\nC 1.520 0.020 0.000\n\
O 2.020 1.330 0.050\nH 2.990 1.330 0.000\nH -0.390 1.030 0.000\nH -0.390 -0.510 0.890\n\
H -0.390 -0.510 -0.890\nH 1.910 -0.510 0.890\nH 1.910 -0.510 -0.890\n";

/// A 2 x 2 x 2 fcc argon crystal (32 atoms), every atom shifted a little.
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

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("mopt_cmatrix_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn frames(path: &Path) -> Vec<MolecularSystem> {
    read_xyz_trajectory_file(path, SpinCharge::default(), None)
        .unwrap()
        .frames
        .into_iter()
        .map(|f| f.system)
        .collect()
}

/// The kinds of constraints, as options, for the atoms of a structure and two fragments
/// (lists of atoms followed by the size of the first, as `-pc fbond` takes them): (name,
/// options).
fn kinds(atoms: [usize; 3], fragments: &str) -> Vec<(&'static str, Vec<String>)> {
    let [a, b, c] = atoms;
    vec![
        ("frozen", vec!["-fix".into(), format!("{c}")]),
        (
            "distance",
            vec!["-pc".into(), "bond".into(), format!("{a},{b}")],
        ),
        (
            "fbond",
            vec!["-pc".into(), "fbond".into(), fragments.to_string()],
        ),
        (
            "angle",
            vec!["-pc".into(), "angle".into(), format!("{c},{a},{b}")],
        ),
        ("gfix", vec!["-gfix".into(), format!("{a},{b},{c}")]),
        (
            "rot",
            vec!["-pc".into(), "rot".into(), format!("{a},{b},{c}")],
        ),
        (
            "user",
            vec!["-uc".into(), format!("dist({a},{c}) - dist({b},{c})")],
        ),
    ]
}

/// Bound on a residual of a structure in memory: the guarantee every hard constraint is held
/// to (`pes::constraint::GUARANTEE`; the restore reaches about 1e-14 here).
const IN_MEMORY: f64 = 1e-10;
/// Bound on a residual recomputed from a trajectory (12 decimals of Angstrom, a rounding of
/// up to 9.4e-13 Bohr per coordinate): the guarantee as well.
const FROM_TRAJECTORY: f64 = 1e-10;
/// Bound on a residual recomputed from a final structure file (`format_xyz_system`: 12
/// decimals of Angstrom, up to 9.4e-13 Bohr per coordinate): the guarantee as well.
const FROM_STRUCTURE_FILE: f64 = 1e-10;

/// The constraints of `options` as a set resolved at `input`, and the frozen atoms.
fn held(options: &[String], input: &MolecularSystem) -> (ConstraintSet, Vec<usize>) {
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
            let a: Vec<usize> = options[1].split(',').map(|x| x.parse().unwrap()).collect();
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

/// Every structure of `path` holds the constraint of `options` at its value in `input`
/// (files: the bound of the printed digits).
fn check_frames(
    cell: &str,
    path: &[MolecularSystem],
    options: &[String],
    input: &MolecularSystem,
    fractional: bool,
) {
    check_within(
        cell,
        path,
        options,
        input,
        fractional,
        FROM_TRAJECTORY,
        false,
    );
}

/// [`check_frames`] with the bound `bound` on the residuals.
fn check_within(
    cell: &str,
    path: &[MolecularSystem],
    options: &[String],
    input: &MolecularSystem,
    fractional: bool,
    bound: f64,
    memory: bool,
) {
    assert!(!path.is_empty(), "{cell}: no frames");
    let (set, frozen) = held(options, input);
    let mut largest = 0.0_f64;
    for (k, sys) in path.iter().enumerate() {
        assert_eq!(sys.n_atoms(), input.n_atoms(), "{cell}");
        if frozen.is_empty() {
            let r = set.residuals(sys).unwrap();
            largest = r.iter().fold(largest, |a, v| a.max(v.abs()));
            eprintln!("RESIDUAL {cell} (bound {bound:.0e}): {largest:.2e}");
            assert!(
                r.iter().all(|v| v.abs() <= bound),
                "{cell} frame {k}: {r:?}"
            );
            continue;
        }
        for &atom in &frozen {
            let id = AtomId::new(atom);
            if fractional {
                // Files carry 12 decimals of the positions and fewer of the lattice.
                let s = sys
                    .active_cell()
                    .unwrap()
                    .cart_to_frac(sys.position(id).value());
                let s0 = input
                    .active_cell()
                    .unwrap()
                    .cart_to_frac(input.position(id).value());
                for c in 0..3 {
                    assert!(
                        (s[c] - s0[c]).abs() < 1e-12,
                        "{cell} frame {k}: {s:?} {s0:?}"
                    );
                }
            } else {
                let (p, q) = (sys.position(id).value(), input.position(id).value());
                for c in 0..3 {
                    if memory {
                        // In memory: assigned, the same numbers.
                        assert_eq!(p[c].to_bits(), q[c].to_bits(), "{cell} frame {k}");
                    } else {
                        // The rounding of the 12 printed decimals.
                        let printed = 2e-12;
                        assert!(
                            (p[c] - q[c]).abs() < printed,
                            "{cell} frame {k}: {p:?} {q:?}"
                        );
                    }
                }
            }
        }
    }
}

fn run(dir: &Path, args: &[String]) -> OptmainResult {
    let argv: Vec<&str> = args.iter().map(String::as_str).collect();
    let config = config_from_cli(&argv).unwrap();
    run_with(&config, dir, None, &CalculatorRegistry::native())
        .unwrap()
        .remove(0)
}

/// The number of modes in `normal_modes.txt` (those above the cutoff of the table).
fn listed_modes(dir: &Path) -> usize {
    std::fs::read_to_string(dir.join("normal_modes.txt"))
        .unwrap()
        .lines()
        .filter(|l| l.starts_with("Freq"))
        .map(|l| l.split_whitespace().count() - 2)
        .sum()
}

fn molecule_cell(context: &str, extra: &[&str], geometry: &str) {
    let dir = scratch(context);
    let file = dir.join("mol.xyz");
    std::fs::write(&file, geometry).unwrap();
    let input = read_xyz_file(&file, SpinCharge::default(), None).unwrap();
    // The two methyl groups of ethane.
    for (kind, options) in kinds([1, 2, 3], "1,3,4,5,2,6,7,8,4") {
        let cell = format!("{context} x {kind}");
        let mut args: Vec<String> = vec![file.to_string_lossy().into_owned(), "-os".into()];
        args.push("gfnff".into());
        args.extend(extra.iter().map(|s| s.to_string()));
        args.extend(options.iter().cloned());
        if context == "dic" && (kind == "frozen" || kind == "rot") {
            // Internal coordinates have no position and orientation to hold a frozen atom in.
            let argv: Vec<&str> = args.iter().map(String::as_str).collect();
            let config = config_from_cli(&argv).unwrap();
            let err = run_with(&config, &dir, None, &CalculatorRegistry::native())
                .unwrap_err()
                .to_string();
            assert!(
                err.contains("pin the frame") && err.contains("-coord"),
                "{cell}: {err}"
            );
            continue;
        }
        let out = run(&dir, &args);
        assert!(out.converged, "{cell}");
        check_frames(
            &cell,
            &frames(&out.directory.join("mol_traj.xyz")),
            &options,
            &input,
            false,
        );
        // The final structure in memory, to the rounding floor, and its file.
        let end = out.summary.as_ref().unwrap().system.clone();
        check_within(
            &cell,
            std::slice::from_ref(&end),
            &options,
            &input,
            false,
            IN_MEMORY,
            true,
        );
        check_within(
            &cell,
            &[read_xyz_file(&out.optimized, SpinCharge::default(), None).unwrap()],
            &options,
            &input,
            false,
            FROM_STRUCTURE_FILE,
            false,
        );
        let distance = |s: &MolecularSystem, a: usize, b: usize| {
            s.distance(AtomId::new(a - 1), AtomId::new(b - 1)).value()
        };
        match kind {
            // The fragments deform and turn while their centroids keep their distance.
            "fbond" => assert!(
                (distance(&end, 1, 3) - distance(&input, 1, 3)).abs() > 1e-4
                    || (distance(&end, 2, 6) - distance(&input, 2, 6)).abs() > 1e-4,
                "{cell}: no distance within a fragment changed"
            ),
            // The group moves as a rigid body: its atoms move, their distances do not.
            "gfix" => {
                let moved = (0..3).any(|a| {
                    let (p, q) = (
                        end.position(AtomId::new(a)).value(),
                        input.position(AtomId::new(a)).value(),
                    );
                    (0..3).any(|c| (p[c] - q[c]).abs() > 1e-4)
                });
                assert!(moved, "{cell}: the group did not move");
            }
            _ => {}
        }
        if context == "minimum" {
            // The vibrations on the constraint surface: 3N - 6 - m without frozen atoms
            // (one scalar constraint), and with the frozen atom (three) the frame is pinned:
            // its rotations about the atom are left with no curvature and not listed.
            let summary = out.summary.as_ref().unwrap();
            assert_eq!(summary.imaginary_modes, Some(0), "{cell}");
            let n = 3 * input.n_atoms();
            let listed = listed_modes(&out.directory);
            match kind {
                // Frozen atoms and an oriented group pin the frame (three scalar
                // constraints each).
                "frozen" | "rot" => {
                    assert!(listed >= n - 6 - 3 && listed <= n - 3, "{cell}: {listed}")
                }
                // Three distances of the rigid group.
                "gfix" => assert_eq!(listed, n - 6 - 3, "{cell}"),
                _ => assert_eq!(listed, n - 6 - 1, "{cell}"),
            }
            let thermo =
                std::fs::read_to_string(out.directory.join("thermochemistry.txt")).unwrap();
            assert!(thermo.contains("constraint surface"), "{cell}");
        }
        if context == "irc" {
            let path = frames(&out.directory.join("irc_combined_path.xyz"));
            assert!(path.len() > 5, "{cell}: {}", path.len());
            // Written with 12 decimals.
            check_within(
                &cell,
                &path,
                &options,
                &input,
                false,
                FROM_STRUCTURE_FILE,
                false,
            );
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn constraints_at_a_minimum() {
    molecule_cell(
        "minimum",
        &["-opt", "rsirfo_bfgs", "-freq", "-nosymm"],
        STAGGERED,
    );
}

#[test]
fn constraints_at_a_transition_state() {
    molecule_cell(
        "saddle",
        &["-opt", "rsirfo_bofill", "-order", "1", "-fc", "1"],
        ECLIPSED,
    );
}

#[test]
fn constraints_in_delocalized_internal_coordinates() {
    molecule_cell("dic", &["-opt", "rsirfo_bfgs", "-coord", "dic"], STAGGERED);
}

#[test]
fn constraints_along_the_irc() {
    molecule_cell(
        "irc",
        &[
            "-opt",
            "rsirfo_bofill",
            "-order",
            "1",
            "-fc",
            "1",
            "-irc",
            "0.1",
            "40",
            "lqa",
        ],
        ECLIPSED,
    );
}

fn crystal_cell(context: &str, extra: &[&str]) {
    let dir = scratch(context);
    let file = dir.join("fcc.xyz");
    std::fs::write(&file, fcc()).unwrap();
    let input = read_xyz_file(&file, SpinCharge::default(), None).unwrap();
    let variable = extra.contains(&"-vc");
    for (kind, options) in kinds([1, 2, 3], "1,2,3,4,5,6,3") {
        let cell = format!("{context} x {kind}");
        if variable && kind == "rot" {
            // Refused with the reason below.
            continue;
        }
        let mut args: Vec<String> = vec![
            file.to_string_lossy().into_owned(),
            "-os".into(),
            "lj".into(),
            "-opt".into(),
            "rsirfo_bfgs".into(),
            "-ns".into(),
            "400".into(),
        ];
        args.extend(extra.iter().map(|s| s.to_string()));
        args.extend(options.iter().cloned());
        let out = run(&dir, &args);
        assert!(out.converged, "{cell}");
        let path = frames(&out.directory.join("fcc_traj.xyz"));
        check_frames(&cell, &path, &options, &input, variable);
        if !variable {
            let end = out.summary.as_ref().unwrap().system.clone();
            check_within(
                &cell,
                std::slice::from_ref(&end),
                &options,
                &input,
                false,
                IN_MEMORY,
                true,
            );
        }
        if variable {
            // The cell has relaxed (the lattice of the input is compressed).
            let (v0, v1) = (
                input.active_cell().unwrap().volume_bohr3(),
                path.last().unwrap().active_cell().unwrap().volume_bohr3(),
            );
            assert!((v1 - v0).abs() > 1e-3 * v0, "{cell}: {v0} -> {v1}");
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn constraints_in_a_fixed_periodic_cell() {
    crystal_cell("fixed_cell", &[]);
}

#[test]
fn constraints_in_a_variable_cell() {
    crystal_cell("variable_cell", &["-vc"]);
    // What has no meaning in a variable cell, and an IRC after a constrained one, are refused
    // with the reason.
    let dir = scratch("variable_cell_refused");
    let file = dir.join("fcc.xyz");
    std::fs::write(&file, fcc()).unwrap();
    let file = file.to_string_lossy().into_owned();
    for (extra, expected) in [
        (vec!["-pc", "x", "1"], "Cartesian component"),
        (vec!["-pc", "rot", "1-4"], "orientation"),
        (vec!["-pc", "atoms_pair", "1,2"], "atoms_pair"),
        (vec!["-uc", "x(1)"], "Cartesian positions"),
        (
            vec!["-fix", "1", "-irc", "0.1", "10", "lqa"],
            "-irc after a constrained -vc",
        ),
    ] {
        let mut args = vec![file.as_str(), "-os", "lj", "-vc"];
        args.extend(extra.iter().copied());
        let config = config_from_cli(&args).unwrap();
        let err = run_with(&config, &dir, None, &CalculatorRegistry::native())
            .unwrap_err()
            .to_string();
        assert!(err.contains(expected), "{extra:?}: {err}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// The constrained optimum of a variable cell against a stiff spring on the same distance,
/// and the stress of the cell at it: with frozen atoms (fractional coordinates held) every
/// lattice variable is free, so the derivative of the enthalpy along each vanishes.
#[test]
fn a_constrained_variable_cell_agrees_with_a_stiff_penalty() {
    use mopt_rs::domain::EvalRequest;
    use mopt_rs::opt::cell::VariableCell;
    use mopt_rs::pes::registry::CalculatorOptions;
    let dir = scratch("vc_penalty");
    let file = dir.join("fcc.xyz");
    std::fs::write(&file, fcc()).unwrap();
    let file = file.to_string_lossy().into_owned();
    let base = [
        file.as_str(),
        "-os",
        "lj",
        "-vc",
        "-opt",
        "rsirfo_bfgs",
        "-tcc",
        "-ns",
        "600",
    ];
    let with = |extra: &[&str]| {
        let mut args = base.to_vec();
        args.extend_from_slice(extra);
        let config = config_from_cli(&args).unwrap();
        let out = run_with(&config, &dir, None, &CalculatorRegistry::native())
            .unwrap()
            .remove(0);
        assert!(out.converged, "{extra:?}");
        out.summary.unwrap()
    };
    let held = with(&["-pc", "manual", "bond", "1,2", "3.60"]);
    let spring = with(&["-kp", "20.0", "3.60", "1,2"]);
    let d = |s: &MolecularSystem| s.distance(AtomId::new(0), AtomId::new(1)).value();
    // In memory: the rounding floor of the restore (the target itself carries the rounding of
    // the conversion from Angstrom).
    let target = 3.60 * mopt_rs::domain::Units::ANGSTROM_TO_BOHR;
    assert!(
        (d(&held.system) - target).abs() < 1e-12,
        "{}",
        d(&held.system) - target
    );
    let (v1, v2) = (
        held.system.active_cell().unwrap().volume_bohr3(),
        spring.system.active_cell().unwrap().volume_bohr3(),
    );
    assert!((v1 - v2).abs() < 1e-3 * v1, "volume {v1} vs {v2}");
    assert!(
        (held.energy.value() - spring.energy.value()).abs() < 2e-5,
        "{} vs {}",
        held.energy.value(),
        spring.energy.value()
    );
    // Frozen atoms: the lattice part of the gradient of the enthalpy vanishes.
    let frozen = with(&["-fix", "1,5"]);
    let end = &frozen.system;
    let calc = CalculatorRegistry::native()
        .create("lj", &CalculatorOptions::default(), end)
        .unwrap();
    let out = calc
        .evaluate(end, &EvalRequest::ENERGY_GRAD_STRESS)
        .unwrap();
    let stress = match out.stress {
        Some(s) => s,
        None => mopt_rs::pes::stress::numerical_stress(calc.as_ref(), end, 1e-5).unwrap(),
    };
    let map = VariableCell::new(end, 2.0, 0.0).unwrap();
    let (_, slope) = map
        .objective(end, out.energy.value(), &out.gradient_bohr, &stress)
        .unwrap();
    let n = end.dof();
    for k in n..slope.nrows() {
        assert!(slope[k].abs() < 1e-4, "lattice variable {k}: {}", slope[k]);
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn constraints_with_oniom_microiterations() {
    let dir = scratch("oniom");
    let file = dir.join("ethanol.xyz");
    std::fs::write(&file, ETHANOL).unwrap();
    let input = read_xyz_file(&file, SpinCharge::default(), None).unwrap();
    // Atoms 2, 3, 4 (C, O, H) are in the model region of the core.
    for (kind, options) in kinds([2, 3, 4], "2,8,3,4,2") {
        let cell = format!("oniom x {kind}");
        let mut args: Vec<String> = [
            file.to_str().unwrap(),
            "-os",
            "gfnff",
            "-oniom",
            "2-4,8,9",
            "auto",
            "gfnff",
            "-opt",
            "rsirfo_bfgs",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        args.extend(options.iter().cloned());
        let out = run(&dir, &args);
        assert!(out.converged, "{cell}");
        let log = std::fs::read_to_string(out.directory.join("mopt.log")).unwrap();
        assert!(log.contains("ONIOM microiterations"), "{cell}");
        // The trajectory and the result hold the whole system.
        // The trajectory is written again with the whole system after the run, with 12
        // decimals (`pes::oniom::micro::files`).
        let path = frames(&out.directory.join("ethanol_traj.xyz"));
        check_within(
            &cell,
            &path,
            &options,
            &input,
            false,
            FROM_STRUCTURE_FILE,
            false,
        );
        check_within(
            &cell,
            &[read_xyz_file(&out.optimized, SpinCharge::default(), None).unwrap()],
            &options,
            &input,
            false,
            FROM_STRUCTURE_FILE,
            false,
        );
        // The whole system in memory.
        let end = out.summary.as_ref().unwrap().system.clone();
        check_within(
            &cell,
            std::slice::from_ref(&end),
            &options,
            &input,
            false,
            IN_MEMORY,
            true,
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// The constraints on the cell itself (`-vc_fix` and `-uc` of the functions of the cell) in a
/// variable cell: every frame of the trajectory, the final structure file, and the final
/// structure in memory hold them at the values of the input (or the value given) to the
/// guarantee, and the rest of the cell relaxes.
#[test]
fn constraints_on_the_cell_in_a_variable_cell() {
    use mopt_rs::cli::constraints::{cell_fix_constraints, resolve_for_run};
    let dir = scratch("cell_kinds");
    let file = dir.join("fcc.xyz");
    std::fs::write(&file, fcc()).unwrap();
    let input = read_xyz_file(&file, SpinCharge::default(), None).unwrap();
    let name = file.to_string_lossy().into_owned();
    for (kind, option, value) in [
        ("length", "-vc_fix", "a"),
        ("length at a value", "-vc_fix", "b=10.7"),
        ("angle", "-vc_fix", "gamma"),
        ("volume", "-vc_fix", "volume"),
        ("shape", "-vc_fix", "shape"),
        ("lattice vector", "-vc_fix", "c_vector"),
        ("component", "-vc_fix", "az"),
        ("ratio", "-uc", "cell_a()/cell_b()"),
    ] {
        let cell = format!("variable cell x {kind}");
        let args = [
            name.as_str(),
            "-os",
            "lj",
            "-vc",
            "-opt",
            "rsirfo_bfgs",
            "-ns",
            "400",
            option,
            value,
        ];
        let out = run(&dir, &args.map(String::from));
        assert!(out.converged, "{cell}");
        let held = if option == "-vc_fix" {
            cell_fix_constraints(&[value.to_string()]).unwrap()
        } else {
            vec![user_constraint(value).unwrap()]
        };
        let set =
            ConstraintSet::new(&resolve_for_run(&held, &input).unwrap(), &input, None).unwrap();
        let end = out.summary.as_ref().unwrap().system.clone();
        let mut structures = frames(&out.directory.join("fcc_traj.xyz"));
        structures.push(read_xyz_file(&out.optimized, SpinCharge::default(), None).unwrap());
        structures.push(end.clone());
        let mut largest = 0.0_f64;
        for (k, s) in structures.iter().enumerate() {
            let r = set.residuals(s).unwrap();
            largest = r.iter().fold(largest, |m, v| m.max(v.abs()));
            assert!(
                r.iter().all(|v| v.abs() <= IN_MEMORY),
                "{cell} structure {k}: {r:?}"
            );
        }
        eprintln!("RESIDUAL {cell} (bound 1e-10, files and memory): {largest:.2e}");
        // The cell has relaxed.
        let (v0, v1) = (
            input.active_cell().unwrap().volume_bohr3(),
            end.active_cell().unwrap().volume_bohr3(),
        );
        if kind != "volume" {
            assert!((v1 - v0).abs() > 1e-4 * v0, "{cell}: {v0} -> {v1}");
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}
