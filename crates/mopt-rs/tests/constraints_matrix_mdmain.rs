//! The constraints of `mdmain` in every kind of run: {frozen atoms, a distance, an angle, a
//! user-defined function, `-gfix`, `-cc`} x {an isolated cluster, a periodic fixed cell, ONIOM
//! (LJ:LJ)} x {NVE, a thermostat, a Wigner start}. Every frame of the trajectory (written at
//! every step) and every step in memory holds the constraint at its value in the input
//! structure, `|c(x) - c(input)| <= 1e-10` Bohr or rad in memory (SHAKE iterates to the
//! rounding floor) and 1e-10 from the files (12 decimals of Angstrom), frozen atoms the same
//! numbers as in the input (also from a Wigner sample, whose frozen atoms are not displaced);
//! and the temperature counts the degrees of freedom the constraints and the conserved
//! motions leave.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::path::{Path, PathBuf};

use mopt_rs::cli::constraints::{parse_projection_constraints, user_constraint};
use mopt_rs::cli::mdmain::{config_from_cli, run_with_observers};
use mopt_rs::domain::{AtomId, MolecularSystem, SpinCharge, Units};
use mopt_rs::io::xyz::{read_xyz_file, read_xyz_trajectory_file};
use mopt_rs::md::MdStepReport;
use mopt_rs::pes::constraint::{Constraint, ConstraintKind, ConstraintSet};
use mopt_rs::pes::registry::CalculatorRegistry;
use mopt_rs::workflow::md::MdObserver;

/// Five argon atoms near a minimum of their cluster (a triangle, one atom over it, one
/// beside it).
const CLUSTER: &str = "5\ncluster\nAr 0.000 0.000 0.000\nAr 3.800 0.000 0.000\n\
Ar 1.900 3.291 0.000\nAr 1.900 1.097 3.100\nAr -2.900 1.500 0.400\n";

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
    let dir = std::env::temp_dir().join(format!("mopt_cmd_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// The kinds of constraints on atoms 1-3: (name, options, scalar constraints, frozen atoms).
fn kinds(input: &MolecularSystem) -> Vec<(&'static str, Vec<String>, usize, usize)> {
    let s = |v: &[&str]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>();
    let d12 = input.distance(AtomId::new(0), AtomId::new(1)).value() * Units::BOHR_TO_ANGSTROM;
    vec![
        ("frozen", s(&["-fix", "3"]), 0, 1),
        ("distance", s(&["-pc", "bond", "1,2"]), 1, 0),
        ("angle", s(&["-pc", "angle", "3,1,2"]), 1, 0),
        ("user", s(&["-uc", "dist(1,3) - dist(2,3)"]), 1, 0),
        ("gfix", s(&["-gfix", "1,2,3"]), 3, 0),
        ("cc", vec!["-cc".into(), format!("{d12:.10},1,2")], 1, 0),
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
        // `-cc VALUE,1,2`: the value given (the distance of the input to 10 decimals).
        "-cc" => {
            let value: f64 = options[1].split(',').next().unwrap().parse().unwrap();
            (
                vec![Constraint::with_target(
                    ConstraintKind::Distance(vec![AtomId::new(0)], vec![AtomId::new(1)]),
                    value * Units::ANGSTROM_TO_BOHR,
                )],
                Vec::new(),
            )
        }
        _ => (vec![user_constraint(&options[1]).unwrap()], Vec::new()),
    };
    (ConstraintSet::new(&c, input, None).unwrap(), frozen)
}

/// Every structure of every xyz file under `dir` (frames of `atoms` atoms).
fn all_frames(dir: &Path, atoms: usize) -> Vec<(PathBuf, MolecularSystem)> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(dir).unwrap().flatten() {
        let path = entry.path();
        if path.extension().is_some_and(|e| e == "xyz")
            && !path.to_string_lossy().contains("velocit")
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
    out
}

/// The bound on a residual recomputed in memory: the rounding floor of evaluating a
/// distance or an angle of a few Bohr (about 1e-15) and of the SHAKE iterations, which go
/// on to that floor, with a margin.
/// (The guarantee every hard constraint is held to: `pes::constraint::GUARANTEE`.)
const IN_MEMORY: f64 = 1e-10;
/// The bound on a residual recomputed from the trajectory: 12 decimals of Angstrom are
/// written (rounding of up to 9.4e-13 Bohr per coordinate, a distance or an angle of a few
/// Bohr moves by up to about 3e-12); a drift above that would show.
/// (The guarantee every hard constraint is held to, from the files as well.)
const FROM_FILES: f64 = 1e-10;
/// The bound on a frozen coordinate read from the trajectory: the rounding of its decimals.
const FROZEN_IN_FILES: f64 = 1e-12;

/// The positions of every step, in memory.
#[derive(Default)]
struct Steps(Vec<faer::Col<f64>>);

impl MdObserver for Steps {
    fn on_step(&mut self, report: &MdStepReport) -> mopt_rs::error::MoptResult<()> {
        self.0.push(report.x.clone());
        Ok(())
    }
}

fn check_frames(
    cell: &str,
    frames: &[(PathBuf, MolecularSystem)],
    options: &[String],
    input: &MolecularSystem,
    steps: usize,
) {
    check_frames_within(
        cell,
        frames,
        options,
        input,
        steps,
        FROM_FILES,
        FROZEN_IN_FILES,
    );
}

/// Every frame holds the constraint: residuals below ound, frozen atoms within
/// rozen_bound of the input (0: the same numbers).
fn check_frames_within(
    cell: &str,
    frames: &[(PathBuf, MolecularSystem)],
    options: &[String],
    input: &MolecularSystem,
    steps: usize,
    bound: f64,
    frozen_bound: f64,
) {
    assert!(frames.len() > steps, "{cell}: {} frames", frames.len());
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

/// The degrees of freedom and the conserved motions the log reports.
fn reported(dir: &Path) -> (f64, String) {
    let log = std::fs::read_to_string(dir.join("mopt.log")).unwrap();
    let dof = log
        .lines()
        .find_map(|l| l.split("Degrees of freedom of the temperature: ").nth(1))
        .unwrap()
        .trim()
        .parse()
        .unwrap();
    let rigid = log
        .lines()
        .find_map(|l| l.split("Conserved motions of the whole system: ").nth(1))
        .unwrap()
        .split(';')
        .next()
        .unwrap()
        .to_string();
    (dof, rigid)
}

const STEPS: usize = 30;

fn variants() -> Vec<(&'static str, Vec<&'static str>)> {
    vec![
        ("nve", vec!["-mt", "nve"]),
        ("thermostat", vec!["-mt", "nosehooverchain"]),
        (
            "wigner",
            vec![
                "-mt",
                "nve",
                "-md_opt",
                "velocities=wigner",
                "write_every=1",
            ],
        ),
    ]
}

fn context(name: &str, text: &str, extra: &[&str]) {
    let dir = scratch(name);
    let file = dir.join("start.xyz");
    std::fs::write(&file, text).unwrap();
    let input = read_xyz_file(&file, SpinCharge::default(), None).unwrap();
    let registry = CalculatorRegistry::native();
    let n = input.n_atoms();
    for (variant, variant_options) in variants() {
        for (kind, options, scalars, frozen) in kinds(&input) {
            let cell = format!("{name} x {variant} x {kind}");
            let mut args: Vec<String> = vec![
                file.to_string_lossy().into_owned(),
                "-os".into(),
                "lj".into(),
                "-time".into(),
                STEPS.to_string(),
                "-temp".into(),
                "40".into(),
            ];
            args.extend(extra.iter().map(|s| s.to_string()));
            args.extend(variant_options.iter().map(|s| s.to_string()));
            if !variant_options.contains(&"-md_opt") {
                args.extend([
                    "-md_opt".to_string(),
                    "write_every=1".to_string(),
                    "seed=3".to_string(),
                ]);
            } else {
                args.push("seed=3".into());
            }
            args.extend(options.iter().cloned());
            let argv: Vec<&str> = args.iter().map(String::as_str).collect();
            let config = config_from_cli(&argv).unwrap();
            let mut steps = Steps::default();
            let out = run_with_observers(&config, &dir, None, &registry, &mut [&mut steps])
                .unwrap_or_else(|e| panic!("{cell}: {e}"))
                .remove(0);
            // Every step in memory, at the rounding floor; frozen atoms the same numbers.
            let memory: Vec<(PathBuf, MolecularSystem)> = steps
                .0
                .iter()
                .enumerate()
                .map(|(k, x)| {
                    (
                        PathBuf::from(format!("step {k}")),
                        input.with_flat_coords_bohr(x).unwrap(),
                    )
                })
                .collect();
            check_frames_within(&cell, &memory, &options, &input, STEPS, IN_MEMORY, 0.0);
            check_frames(
                &cell,
                &all_frames(&out.directory, n),
                &options,
                &input,
                STEPS,
            );
            // The temperature counts 3N less the held atoms, the constraints, and the
            // conserved motions.
            let (dof, rigid) = reported(&out.directory);
            let motions = match rigid.as_str() {
                "none" => 0,
                "translation" => 3,
                "translation and rotation" => 6,
                other => panic!("{cell}: {other}"),
            };
            if frozen > 0 {
                assert_eq!(motions, 0, "{cell}: held atoms pin the structure");
            }
            if input.has_pbc() {
                assert!(motions <= 3, "{cell}: a cell has no free rotations");
            }
            if frozen == 0 && variant != "thermostat" && extra.is_empty() {
                // NVE on a surface blind to rigid motions (Lennard-Jones, distances and
                // angles held): those motions are conserved and not counted.
                assert_eq!(motions, if input.has_pbc() { 3 } else { 6 }, "{cell}");
            }
            let want = (3 * n - 3 * frozen - scalars - motions) as f64;
            assert_eq!(dof, want, "{cell}: {rigid}");
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn constraints_of_an_isolated_cluster() {
    context("isolated", CLUSTER, &[]);
}

#[test]
fn constraints_in_a_periodic_cell() {
    context("periodic", &fcc(), &[]);
}

#[test]
fn constraints_with_oniom() {
    context("oniom", CLUSTER, &["-oniom", "1-4", "none", "lj"]);
}

/// The frozen atom of a Wigner start stays where it is in the input, and a value held at
/// the start is that of the input, not of the sample (before, both came from the sampled
/// structure).
#[test]
fn a_wigner_start_holds_the_values_of_the_input() {
    let dir = scratch("wigner");
    let file = dir.join("start.xyz");
    std::fs::write(&file, CLUSTER).unwrap();
    let input = read_xyz_file(&file, SpinCharge::default(), None).unwrap();
    for options in [vec!["-fix", "3"], vec!["-pc", "bond", "1,4"]] {
        let mut args = vec![
            file.to_str().unwrap(),
            "-os",
            "lj",
            "-time",
            "5",
            "-temp",
            "300",
            "-mt",
            "nve",
            "-md_opt",
            "velocities=wigner",
            "write_every=1",
            "seed=11",
        ];
        args.extend(options.iter().copied());
        let config = config_from_cli(&args).unwrap();
        let mut steps = Steps::default();
        let out = run_with_observers(
            &config,
            &dir,
            None,
            &CalculatorRegistry::native(),
            &mut [&mut steps],
        )
        .unwrap()
        .remove(0);
        let options: Vec<String> = options.iter().map(|s| s.to_string()).collect();
        let memory: Vec<(PathBuf, MolecularSystem)> = steps
            .0
            .iter()
            .map(|x| {
                (
                    PathBuf::from("in memory"),
                    input.with_flat_coords_bohr(x).unwrap(),
                )
            })
            .collect();
        check_frames_within("wigner", &memory, &options, &input, 5, IN_MEMORY, 0.0);
        check_frames(
            "wigner",
            &all_frames(&out.directory, 5),
            &options,
            &input,
            5,
        );
        let log = std::fs::read_to_string(out.directory.join("mopt.log")).unwrap();
        assert!(log.contains("constraint direction(s)"), "{log}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}
