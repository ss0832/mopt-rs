//! Constraints and bias potentials in the optimizer and in `optmain`: one test per defect that
//! was found by reading the code (rigid-body motion with absolute-position terms, constraints
//! carried to the IRC and the vibrational analysis, `-vc` with constraints, `-gfix`, the bias
//! Hessian in the trust radius, ramped biases in the IRC, the check of the branching plane with
//! constraints, `-tcc` with `-lcc`, and the displacement of the first iteration).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::path::{Path, PathBuf};
use std::sync::Arc;

use mopt_rs::cli::optmain::{config_from_cli, run_with};
use mopt_rs::domain::{
    positions, Angstrom, AtomId, Bohr, Element, EvalOutput, EvalRequest, MolecularSystem, Quantity,
    SpinCharge,
};
use mopt_rs::error::MoptResult;
use mopt_rs::io::xyz::{read_xyz_file, read_xyz_trajectory_file};
use mopt_rs::math::ad::{eval_value_grad_hessian, Scalar};
use mopt_rs::opt::convergence::ConvergenceCriteria;
use mopt_rs::opt::state_machine::{
    GeometryOptimizer, IterationReport, OptimizerSettings, Termination,
};
use mopt_rs::pes::bias::{AdBias, KeepDistance, Ramp, VoidPoint};
use mopt_rs::pes::combine::{model_function, two_states, TaskSpec};
use mopt_rs::pes::constraint::{Constraint, ConstraintKind, ConstraintSet};
use mopt_rs::pes::native::LennardJones;
use mopt_rs::pes::registry::{CalculatorOptions, CalculatorRegistry};
use mopt_rs::pes::{Calculator, PesTerm};
use mopt_rs::workflow::{run_optimization, Observer};

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("mopt_w1a_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn write(dir: &Path, name: &str, text: &str) -> String {
    let path = dir.join(name);
    std::fs::write(&path, text).unwrap();
    path.to_string_lossy().into_owned()
}

fn frames(path: &Path) -> Vec<MolecularSystem> {
    read_xyz_trajectory_file(path, SpinCharge::default(), None)
        .unwrap()
        .frames
        .into_iter()
        .map(|f| f.system)
        .collect()
}

fn distance(sys: &MolecularSystem, i: usize, j: usize) -> f64 {
    sys.distance(AtomId::new(i), AtomId::new(j)).value()
}

const AR7: &str = "7
argon cluster
Ar   1.9500   0.0000   0.0000
Ar   0.6026   1.8546   0.0300
Ar  -1.5776   1.1462   0.0700
Ar  -1.5776  -1.1462   0.1100
Ar   0.6026  -1.8546   0.1500
Ar   0.0000   0.0000   2.1000
Ar   0.2000   0.0000  -2.0000
";

/// A guess of the transition state of HCN <-> HNC: H bridging C and N.
const HCN_GUESS: &str = "3\n\nC 0.000 0.000 0.000\nN 1.190 0.000 0.000\nH 0.368 1.132 0.000\n";

fn argon(rows: &[[f64; 3]]) -> MolecularSystem {
    MolecularSystem::new(
        vec![Element::parse_token("Ar").unwrap(); rows.len()],
        &positions::<Angstrom>(rows),
        SpinCharge::default(),
        None,
    )
    .unwrap()
}

/// Collects the notes of every iteration.
#[derive(Default)]
struct Notes(Vec<Vec<String>>);

impl Observer for Notes {
    fn on_iteration(&mut self, report: &IterationReport) -> MoptResult<()> {
        self.0.push(report.notes.clone());
        Ok(())
    }
}

// ---------------------------------------------------------------------------------------
// Rigid-body invariance decided from the calculator and the terms.

#[test]
fn an_absolute_position_bias_keeps_the_structure_in_the_lab_frame() {
    let dir = scratch("absolute");
    let xyz = write(&dir, "ar7.xyz", AR7);
    // The centroid of the cluster held at (30, 0, 0) Bohr: the cluster has to move there as
    // a whole. An optimizer that recentres the structure at every step keeps the centroid at
    // the origin.
    let config = config_from_cli(&[
        &xyz,
        "-os",
        "lj",
        "-opt",
        "rsirfo_bfgs",
        "-ns",
        "600",
        "-ubias",
        "0.5 * ((x([1-7]) - 30)^2 + y([1-7])^2 + z([1-7])^2)",
    ])
    .unwrap();
    let out = run_with(&config, &dir, None, &CalculatorRegistry::native()).unwrap();
    assert!(out[0].converged);
    let end = read_xyz_file(&out[0].optimized, SpinCharge::default(), None).unwrap();
    let x = end.coords_flat_bohr();
    let p: [f64; 3] = std::array::from_fn(|c| (0..7).map(|a| x[3 * a + c]).sum::<f64>() / 7.0);
    assert!(
        (p[0] - 30.0).abs() < 2e-3 && p[1].abs() < 2e-3 && p[2].abs() < 2e-3,
        "centroid at {p:?} Bohr"
    );
    // The cluster is whole: every atom has a neighbour within 5 Angstrom.
    for i in 0..7 {
        let nearest = (0..7)
            .filter(|&j| j != i)
            .map(|j| distance(&end, i, j))
            .fold(f64::INFINITY, f64::min);
        assert!(nearest < 5.0 / 0.529_177, "atom {i}: {nearest} Bohr");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn an_optimizer_that_removes_rigid_motion_refuses_a_frame_dependent_term() {
    let start = argon(&[[0.0, 0.0, 0.0], [3.9, 0.0, 0.0], [1.9, 3.3, 0.0]]);
    let mut opt = GeometryOptimizer::new(OptimizerSettings::default(), start.clone()).unwrap();
    let wall: Arc<dyn PesTerm> = Arc::new(AdBias::new(VoidPoint {
        atom: AtomId::new(0),
        point: [0.0; 3],
        k: 0.1,
        r0: 1.0,
        order: 2.0,
    }));
    let err = opt.add_term("vpp_0", wall).unwrap_err().to_string();
    assert!(err.contains("void_point") || err.contains("vpp"), "{err}");
    // A distance restraint is welcome.
    let mut opt = GeometryOptimizer::new(OptimizerSettings::default(), start).unwrap();
    let spring: Arc<dyn PesTerm> = Arc::new(AdBias::new(KeepDistance {
        a: vec![AtomId::new(0)],
        b: vec![AtomId::new(1)],
        k: Ramp::constant(0.1),
        r0: Ramp::constant(7.5),
    }));
    assert!(opt.add_term("kp_0", spring).is_ok());
}

// ---------------------------------------------------------------------------------------
// The constraints of the input carried to the IRC.

#[test]
fn the_irc_after_an_eigenvector_constrained_search_keeps_the_input_direction() {
    let dir = scratch("eigvec_irc");
    let guess = write(&dir, "hcn.xyz", HCN_GUESS);
    let config = config_from_cli(&[
        &guess,
        "-os",
        "gfn0xtb",
        "-opt",
        "rsirfo_bofill",
        "-order",
        "1",
        "-fc",
        "1",
        "-pc",
        "eigvec",
        "2",
        "-irc",
        "0.1",
        "100",
        "lqa",
    ])
    .unwrap();
    let registry = CalculatorRegistry::native();
    let out = run_with(&config, &dir, None, &registry).unwrap();
    assert!(out[0].converged);
    // The direction is that of the Hessian of the input structure.
    let input = read_xyz_file(Path::new(&guess), SpinCharge::default(), None).unwrap();
    let calc = registry
        .create("gfn0xtb", &CalculatorOptions::default(), &input)
        .unwrap();
    let hessian = calc
        .evaluate(&input, &EvalRequest::ENERGY_GRAD_HESS)
        .unwrap()
        .hessian_bohr
        .unwrap();
    let resolved = Constraint::at_start(ConstraintKind::EigenVector(2))
        .resolve(&input, Some(&hessian))
        .unwrap();
    let set = ConstraintSet::new(&[resolved], &input, None).unwrap();
    let path = frames(&out[0].directory.join("irc_combined_path.xyz"));
    assert!(path.len() > 5, "{}", path.len());
    for sys in &path {
        let r = set.residuals(sys).unwrap()[0];
        // The IRC path is written with 12 decimals of Angstrom: the guarantee.
        assert!(r.abs() <= 1e-10, "v . x - t = {r}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

// ---------------------------------------------------------------------------------------
// Vibrations at a constrained stationary point.

#[test]
fn the_vibrations_of_a_constrained_minimum_are_those_of_the_constraint_surface() {
    let dir = scratch("freq_constrained");
    let water = write(
        &dir,
        "water.xyz",
        "3\n\nO 0.000 0.000 0.117\nH 0.000 0.757 -0.467\nH 0.000 -0.757 -0.467\n",
    );
    // An O-H bond held far beyond the inflection of its curve: along the bond the bare
    // surface curves down.
    let config = config_from_cli(&[
        &water,
        "-os",
        "gfn0xtb",
        "-opt",
        "rsirfo_bfgs",
        "-pc",
        "manual",
        "bond",
        "1,2",
        "1.8",
        "-freq",
        "-nosymm",
    ])
    .unwrap();
    let out = run_with(&config, &dir, None, &CalculatorRegistry::native()).unwrap();
    assert!(out[0].converged);
    let summary = out[0].summary.as_ref().unwrap();
    assert_eq!(summary.imaginary_modes, Some(0));
    let log = std::fs::read_to_string(out[0].directory.join("mopt.log")).unwrap();
    assert!(log.contains("constraint surface"), "{log}");
    let _ = std::fs::remove_dir_all(&dir);
}

// ---------------------------------------------------------------------------------------
// -gfix for optmain.

#[test]
fn gfix_holds_the_distances_between_its_atoms_in_optmain() {
    let dir = scratch("gfix");
    let xyz = write(&dir, "ar7.xyz", AR7);
    let input = read_xyz_file(Path::new(&xyz), SpinCharge::default(), None).unwrap();
    let config =
        config_from_cli(&[&xyz, "-os", "lj", "-opt", "rsirfo_bfgs", "-gfix", "1,2,3"]).unwrap();
    let out = run_with(&config, &dir, None, &CalculatorRegistry::native()).unwrap();
    assert!(out[0].converged);
    for sys in frames(&out[0].directory.join("ar7_traj.xyz")) {
        for (i, j) in [(0, 1), (0, 2), (1, 2)] {
            let d = distance(&sys, i, j) - distance(&input, i, j);
            // The guarantee, from the trajectory (12 decimals of Angstrom).
            assert!(d.abs() <= 1e-10, "{i}-{j}: {d}");
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}

// ---------------------------------------------------------------------------------------
// Bias potentials in the optimizer.

#[test]
fn the_trust_radius_predicts_with_the_hessian_of_the_bias_potentials() {
    let start = argon(&[[0.0, 0.0, 0.0], [3.6, 0.0, 0.0], [1.7, 3.1, 0.2]]);
    let settings = OptimizerSettings {
        methods: vec!["rsirfo_bfgs".to_string()],
        exact_hessian_interval: Some(1),
        ..OptimizerSettings::default()
    };
    let mut opt = GeometryOptimizer::new(settings, start).unwrap();
    // A stiff spring: the bias dominates the curvature along the bond.
    opt.add_term(
        "kp_0",
        Arc::new(AdBias::new(KeepDistance {
            a: vec![AtomId::new(0)],
            b: vec![AtomId::new(1)],
            k: Ramp::constant(0.5),
            r0: Ramp::constant(8.5),
        })),
    )
    .unwrap();
    let mut notes = Notes::default();
    let out = run_optimization(&mut opt, &LennardJones, &mut [&mut notes]).unwrap();
    assert_eq!(out.termination, Some(Termination::Converged));
    let ratios: Vec<f64> = notes
        .0
        .iter()
        .flatten()
        .filter_map(|n| n.strip_prefix("Energy ratio (actual/predicted): "))
        .map(|v| v.trim().parse::<f64>().unwrap())
        .collect();
    assert!(!ratios.is_empty());
    println!("ratios of the actual to the predicted change: {ratios:?}");
    // The first prediction, far from the minimum (later ones divide rounding by rounding).
    assert!((ratios[0] - 1.0).abs() < 0.1, "{ratios:?}");
}

/// Time per iteration of FIRE (no Hessian) on an argon cluster with several bias potentials
/// (printed; the Hessians of the terms are not computed for it).
#[test]
fn time_per_iteration_of_a_gradient_only_run_with_biases() {
    let rows: Vec<[f64; 3]> = (0..27)
        .map(|k| {
            let (i, j, l) = (k % 3, (k / 3) % 3, k / 9);
            let w = 0.09 * (k as f64).sin();
            [
                3.8 * i as f64 + w,
                3.8 * j as f64 - 0.5 * w,
                3.8 * l as f64 + 0.3 * w,
            ]
        })
        .collect();
    let start = argon(&rows);
    let settings = OptimizerSettings {
        methods: vec!["fire".to_string()],
        max_iterations: 200,
        ..OptimizerSettings::default()
    };
    let mut opt = GeometryOptimizer::new(settings, start.clone()).unwrap();
    let options: [(&str, &[&str]); 5] = [
        ("-kp", &["0.01", "3.5", "1,2", "0.01", "3.9", "4,8"]),
        ("-ka", &["0.01", "100", "1,2,5"]),
        ("-kda", &["0.01", "30", "2,1,4,5"]),
        ("-ma", &["50", "1-9", "19-27"]),
        ("-ubias", &["0.001 * (dist([1-9],[19-27]) - 7)^2"]),
    ];
    let mut count = 0;
    for (flag, args) in options {
        let tokens: Vec<String> = args.iter().map(|s| s.to_string()).collect();
        for (k, spec) in mopt_rs::cli::bias::parse_bias_option(flag, &tokens)
            .unwrap()
            .iter()
            .enumerate()
        {
            opt.add_term(&format!("{flag}_{k}"), spec.build(&start).unwrap())
                .unwrap();
            count += 1;
        }
    }
    let clock = std::time::Instant::now();
    let out = run_optimization(&mut opt, &LennardJones, &mut []).unwrap();
    println!(
        "FIRE, 27 argon atoms, {count} bias terms: {:.3} ms per iteration ({} iterations)",
        1e3 * clock.elapsed().as_secs_f64() / out.iterations as f64,
        out.iterations
    );
}

#[test]
fn the_irc_runs_on_the_surface_of_the_last_iteration_of_a_ramped_bias() {
    let dir = scratch("ramp_irc");
    let guess = write(&dir, "hcn.xyz", HCN_GUESS);
    let config = config_from_cli(&[
        &guess,
        "-os",
        "gfn0xtb",
        "-opt",
        "rsirfo_bofill",
        "-order",
        "1",
        "-fc",
        "1",
        "-kp",
        "0.01,0.3",
        "1.25",
        "1,2",
        "-irc",
        "0.1",
        "60",
        "lqa",
    ])
    .unwrap();
    let out = run_with(&config, &dir, None, &CalculatorRegistry::native()).unwrap();
    assert!(out[0].converged);
    let summary = out[0].summary.as_ref().unwrap();
    assert!(summary.iterations > 3);
    let table = std::fs::read_to_string(out[0].directory.join("irc_combined_data.csv")).unwrap();
    let start = table
        .lines()
        .find(|l| l.starts_with("0,"))
        .unwrap()
        .split(',')
        .nth(2)
        .unwrap()
        .parse::<f64>()
        .unwrap();
    let end = summary.biased_energy.value();
    assert!(
        (start - end).abs() < 1e-9,
        "IRC start {start}, end of the optimization {end}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_time_dependent_term_is_refused_in_the_irc() {
    let dir = scratch("time_irc");
    let guess = write(&dir, "hcn.xyz", HCN_GUESS);
    let config = config_from_cli(&[
        &guess,
        "-os",
        "gfn0xtb",
        "-order",
        "1",
        "-fc",
        "1",
        "-ubias",
        "0.0001 * t * dist(1,2)",
        "-irc",
        "0.1",
        "60",
        "lqa",
    ])
    .unwrap();
    let err = run_with(&config, &dir, None, &CalculatorRegistry::native())
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("IRC") && err.contains("TimeDependent"),
        "{err}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

// ---------------------------------------------------------------------------------------
// The check of the branching plane with a constraint.

/// One atom on two harmonic diabatic surfaces coupled by `lambda z`; the adiabatic states
/// meet at a conical intersection where `V1 = V2` and `z = 0`.
struct Cone;

impl Cone {
    fn energy<T: Scalar>(state: usize, r: &[T]) -> T {
        let sq = |p: [f64; 3]| {
            (0..3).fold(T::zero(), |acc, i| {
                let d = r[i] - T::from_f64(p[i]);
                acc + d * d
            })
        };
        let v1 = T::from_f64(0.5) * sq([0.0, 0.0, 0.3]);
        let v2 = T::from_f64(0.5) * sq([1.0, 0.0, 0.3]) + T::from_f64(0.02);
        let mean = (v1 + v2) * T::from_f64(0.5);
        let half_gap = (v1 - v2) * T::from_f64(0.5);
        let v12 = r[2] * T::from_f64(0.1);
        let root = (half_gap * half_gap + v12 * v12 + T::from_f64(1e-16)).sqrt();
        if state == 0 {
            mean - root
        } else {
            mean + root
        }
    }
}

impl Calculator for Cone {
    fn name(&self) -> &str {
        "cone"
    }

    fn evaluate(&self, sys: &MolecularSystem, req: &EvalRequest) -> MoptResult<EvalOutput> {
        let x: Vec<f64> = sys.coords_flat_bohr().iter().copied().collect();
        let (e, g, h) = eval_value_grad_hessian(&x, None, |r| Self::energy(req.excited_state, r));
        Ok(EvalOutput {
            energy: Quantity::new(e),
            gradient_bohr: g,
            hessian_bohr: req.need_hessian.then_some(h),
            orbitals: None,
            stress: None,
            strain_hessian: None,
        })
    }

    fn is_rigid_body_invariant(&self) -> bool {
        false
    }

    fn supports_excited_states(&self) -> bool {
        true
    }
}

#[test]
fn the_check_of_the_branching_plane_tests_the_gradient_within_the_constraints() {
    let start = MolecularSystem::new(
        vec![Element::H],
        &positions::<Bohr>(&[[0.3, 0.2, 0.15]]),
        SpinCharge::default(),
        None,
    )
    .unwrap();
    // y held at 0.2: the lowest point of the seam with it is (0.52, 0.2, 0), where the mean
    // gradient within the seam is the constraint force along y.
    let settings = OptimizerSettings {
        methods: vec!["rsirfo_bfgs".to_string()],
        max_iterations: 500,
        criteria: ConvergenceCriteria::TIGHT,
        rigid_body_invariant: false,
        dissociation_distance: None,
        constraints: vec![Constraint::with_target(
            ConstraintKind::Coordinate(AtomId::new(0), 1),
            0.2,
        )],
        ..OptimizerSettings::default()
    };
    let mut opt = GeometryOptimizer::new(settings, start).unwrap();
    opt.set_surface(
        two_states(
            model_function("opt_meci", None, None).unwrap(),
            TaskSpec::whole("State2").excited_state(1),
        )
        .unwrap(),
    )
    .unwrap();
    let mut notes = Notes::default();
    let out = run_optimization(&mut opt, &Cone, &mut [&mut notes]).unwrap();
    assert_eq!(out.termination, Some(Termination::Converged));
    let restarts = notes
        .0
        .iter()
        .flatten()
        .filter(|n| n.contains("the search goes on with this branching plane"))
        .count();
    assert_eq!(restarts, 0, "{:?}", notes.0.last());
    let p = out.system.position(AtomId::new(0)).value();
    assert!(
        (p[0] - 0.52).abs() < 1e-3 && (p[1] - 0.2).abs() < 1e-10 && p[2].abs() < 1e-3,
        "{p:?}"
    );
}

// ---------------------------------------------------------------------------------------
// Small items.

#[test]
fn tight_and_loose_criteria_together_are_refused() {
    let dir = scratch("tcc_lcc");
    let xyz = write(&dir, "ar7.xyz", AR7);
    let err = config_from_cli(&[&xyz, "-os", "lj", "-tcc", "-lcc"])
        .err()
        .map(|e| e.to_string())
        .unwrap_or_default();
    assert!(err.contains("-tcc") && err.contains("-lcc"), "{err}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_first_iteration_is_not_converged_by_a_step_that_the_constraints_undo() {
    // An argon trimer at its minimum (taken from a short optimization).
    let guess = argon(&[[0.0, 0.0, 0.0], [3.7, 0.1, 0.0], [1.8, 3.4, 0.1]]);
    let mut free = GeometryOptimizer::new(
        OptimizerSettings {
            methods: vec!["rsirfo_bfgs".to_string()],
            criteria: ConvergenceCriteria::TIGHT,
            ..OptimizerSettings::default()
        },
        guess,
    )
    .unwrap();
    let minimum = run_optimization(&mut free, &LennardJones, &mut [])
        .unwrap()
        .system;
    // At the minimum the gradient vanishes, but the bond 1-2 is to be held 0.5 Bohr longer;
    // the eigenvector constraint makes the constraints wait for the first Hessian.
    let target = distance(&minimum, 0, 1) + 0.5;
    let settings = OptimizerSettings {
        methods: vec!["rsirfo_bfgs".to_string()],
        exact_hessian_interval: Some(1),
        constraints: vec![
            Constraint::at_start(ConstraintKind::EigenVector(0)),
            Constraint::with_target(
                ConstraintKind::Distance(vec![AtomId::new(0)], vec![AtomId::new(1)]),
                target,
            ),
        ],
        ..OptimizerSettings::default()
    };
    let mut opt = GeometryOptimizer::new(settings, minimum).unwrap();
    let out = run_optimization(&mut opt, &LennardJones, &mut []).unwrap();
    assert_eq!(out.termination, Some(Termination::Converged));
    let d = distance(&out.system, 0, 1);
    // In memory: the guarantee every hard constraint is held to.
    assert!((d - target).abs() <= 1e-10, "{d} vs {target}");
    assert!(out.iterations > 1, "{}", out.iterations);
}

// ---------------------------------------------------------------------------------------
// -vc with constraints.

/// A 2 x 2 x 2 fcc argon crystal (32 atoms) with every atom shifted a little.
fn fcc_argon() -> String {
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

fn fractional(sys: &MolecularSystem, atom: usize) -> [f64; 3] {
    sys.active_cell()
        .unwrap()
        .cart_to_frac(sys.position(AtomId::new(atom)).value())
}

#[test]
fn a_variable_cell_keeps_frozen_atoms_at_their_fractional_coordinates() {
    let dir = scratch("vc_fix");
    let xyz = write(&dir, "fcc.xyz", &fcc_argon());
    let input = read_xyz_file(Path::new(&xyz), SpinCharge::default(), None).unwrap();
    let config = config_from_cli(&[
        &xyz,
        "-os",
        "lj",
        "-vc",
        "-opt",
        "rsirfo_bfgs",
        "-ns",
        "300",
        "-fix",
        "1,7",
    ])
    .unwrap();
    let out = run_with(&config, &dir, None, &CalculatorRegistry::native()).unwrap();
    // The files carry 12 decimals of the positions and 15 of the lattice: the fractional
    // coordinates read back agree to 1e-12 (the restore holds them to rounding, see the unit
    // tests of `opt::cell_constraints`).
    let path = frames(&out[0].directory.join("fcc_traj.xyz"));
    assert!(path.len() > 3);
    for sys in &path {
        for atom in [0, 6] {
            let (s, s0) = (fractional(sys, atom), fractional(&input, atom));
            for c in 0..3 {
                assert!((s[c] - s0[c]).abs() < 1e-12, "atom {atom}: {s:?} vs {s0:?}");
            }
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}
