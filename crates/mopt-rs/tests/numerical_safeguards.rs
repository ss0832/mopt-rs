//! Safeguards of the optimizers that differ from MultiOptPy (see `docs/source/migration_safeguards.md`,
//! "Numerical safeguards"): saddle searches learn negative curvature, steps that are not
//! finite are refused, and a model Hessian that is not finite does not replace the Hessian in
//! use.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Arc;

use faer::{Col, Mat};
use mopt_rs::domain::{
    positions, Angstrom, AtomId, Element, EvalRequest, MolecularSystem, SpinCharge, Units,
};
use mopt_rs::error::MoptResult;
use mopt_rs::math::linalg::symmetric_eigen;
use mopt_rs::math::projection::project_out_tr_rot_hessian;
use mopt_rs::model_hessian::ModelHessian;
use mopt_rs::opt::mode_kill::ModeKillSettings;
use mopt_rs::opt::registry::MethodRegistry;
use mopt_rs::opt::state_machine::{
    GeometryOptimizer, IterationReport, OptimizerSettings, Termination,
};
use mopt_rs::opt::step::{StepGenerator, StepInput};
use mopt_rs::opt::update::{apply_update, UpdaterRegistry};
use mopt_rs::opt::vector::{VectorOptimizer, VectorSettings};
use mopt_rs::pes::native::muller_brown::MB_SADDLES;
use mopt_rs::pes::native::{Gfn0Xtb, LennardJones};
use mopt_rs::pes::objective::{MullerBrown2D, Objective};
use mopt_rs::pes::Calculator;
use mopt_rs::workflow::{run_optimization, run_vector_optimization, Observer};

fn col(v: &[f64]) -> Col<f64> {
    Col::from_fn(v.len(), |i| v[i])
}

/// A saddle search whose only exact Hessian is the first one has to learn the curvature of the
/// mode it climbs from its own steps, for which `s . y <= 0`. MultiOptPy drops those pairs;
/// with them dropped, Bofill and MSP need 18 iterations from the first point.
#[test]
fn saddle_searches_learn_negative_curvature_from_their_steps() {
    let mb = MullerBrown2D::default();
    for (start, target) in [([-0.7, 0.5], MB_SADDLES[0]), ([0.3, 0.2], MB_SADDLES[1])] {
        for method in ["rsirfo_bofill", "rsirfo_msp"] {
            let s = VectorSettings {
                methods: vec![method.to_string()],
                saddle_order: 1,
                max_iterations: 300,
                exact_hessian_interval: Some(1000),
                ..VectorSettings::default()
            };
            let mut opt = VectorOptimizer::new(s, col(&start)).unwrap();
            let out = run_vector_optimization(&mut opt, &mb, &mut |_| Ok(())).unwrap();
            assert_eq!(out.termination, Some(Termination::Converged), "{method}");
            let d = ((out.x[0] - target[0]).powi(2) + (out.x[1] - target[1]).powi(2)).sqrt();
            assert!(d < 2e-2, "{method} from {start:?}: {:?}", out.x);
            assert!(
                out.iterations <= 12,
                "{method} from {start:?}: {}",
                out.iterations
            );
            // One negative curvature in the learned Hessian.
            let h = opt.hessian();
            let det = h[(0, 0)] * h[(1, 1)] - h[(0, 1)] * h[(1, 0)];
            assert!(det < 0.0, "{method}: {h:?}");
        }
    }
}

/// The block Bofill updates take those pairs as well: from eight starts around the two
/// Muller-Brown saddle points, with the Hessian of the start only, `block_bofill` takes 55
/// iterations in total with them and 62 without, `block_bofill_weighted` 46 and 82.
#[test]
fn block_bofill_updates_learn_negative_curvature_too() {
    let mb = MullerBrown2D::default();
    let starts = [
        ([-0.7, 0.5], MB_SADDLES[0]),
        ([-0.9, 0.7], MB_SADDLES[0]),
        ([-0.75, 0.7], MB_SADDLES[0]),
        ([-0.9, 0.55], MB_SADDLES[0]),
        ([0.3, 0.2], MB_SADDLES[1]),
        ([0.1, 0.4], MB_SADDLES[1]),
        ([0.3, 0.35], MB_SADDLES[1]),
        ([0.15, 0.2], MB_SADDLES[1]),
    ];
    for (method, limit) in [
        ("rsirfo_block_bofill", 60),
        ("rsirfo_block_bofill_weighted", 52),
    ] {
        let mut total = 0;
        for (start, target) in starts {
            let s = VectorSettings {
                methods: vec![method.to_string()],
                saddle_order: 1,
                max_iterations: 300,
                exact_hessian_interval: Some(1000),
                ..VectorSettings::default()
            };
            let mut opt = VectorOptimizer::new(s, col(&start)).unwrap();
            let out = run_vector_optimization(&mut opt, &mb, &mut |_| Ok(())).unwrap();
            assert_eq!(out.termination, Some(Termination::Converged), "{method}");
            let d = ((out.x[0] - target[0]).powi(2) + (out.x[1] - target[1]).powi(2)).sqrt();
            assert!(d < 2e-2, "{method} from {start:?}: {:?}", out.x);
            total += out.iterations;
        }
        println!("{method}: {total} iterations from eight starts");
        assert!(total <= limit, "{method}: {total}");
    }
}

/// The rule itself: saddle searches feed pairs with `s . y < 0` to the formulas without a BFGS
/// term; a formula that can represent the pair reproduces it; BFGS never takes it.
#[test]
fn negative_curvature_pairs_and_the_update_formulas() {
    let (s, y) = (col(&[0.1, 0.0]), col(&[-0.02, 0.01]));
    let reg = UpdaterRegistry::default();
    for (method, saddle) in [
        ("rsirfo_bofill", true),
        ("rsirfo_msp", true),
        ("rsirfo_sr1", true),
        ("rsirfo_psb", true),
        ("rsirfo_ts_bfgs", true),
        ("rsirfo_fsb", false),
        ("rsirfo_bfgs", false),
        ("rsirfo_block_bofill", true),
        ("rsirfo_block_bofill_weighted", true),
        ("rsirfo_block_cfd_bofill", true),
        ("rsirfo_block_fsb", false),
        ("rsirfo_block_bfgs", false),
    ] {
        assert_eq!(
            reg.resolve(method).unwrap().learns_saddle_curvature(),
            saddle,
            "{method}"
        );
    }
    for (method, allow, expect) in [
        ("rsirfo_bofill", true, true),
        ("rsirfo_msp", true, true),
        ("rsirfo_bofill", false, false),
        ("rsirfo_bfgs", true, false),
    ] {
        let mut up = reg.resolve(method).unwrap();
        let mut h = Mat::<f64>::identity(2, 2);
        let applied = apply_update(up.as_mut(), &mut h, &s, &y, allow).unwrap();
        assert_eq!(applied, expect, "{method} allow={allow}");
        if applied {
            // Secant condition: the new Hessian reproduces the negative curvature.
            let r = &h * &s - &y;
            assert!(r.norm_max() < 1e-12, "{method}");
        }
    }
}

/// A hand-built guess of the Diels-Alder transition state of butadiene and ethylene (forming
/// bonds 2.20 Angstrom), with the built-in GFN0-xTB.
fn diels_alder_guess() -> MolecularSystem {
    let rows: [(&str, [f64; 3]); 16] = [
        ("C", [-1.42, 1.18, 0.0]),
        ("C", [-0.705, 0.0, 0.0]),
        ("C", [0.705, 0.0, 0.0]),
        ("C", [1.42, 1.18, 0.0]),
        ("C", [-0.695, 2.08, -1.8727]),
        ("C", [0.695, 2.08, -1.8727]),
        ("H", [-1.25, -0.944, 0.0]),
        ("H", [1.25, -0.944, 0.0]),
        ("H", [-2.50, 1.16, 0.15]),
        ("H", [-0.933, 2.066, 0.405]),
        ("H", [2.50, 1.16, 0.15]),
        ("H", [0.933, 2.066, 0.405]),
        ("H", [-1.24, 3.00, -2.0727]),
        ("H", [-1.24, 1.16, -2.0727]),
        ("H", [1.24, 3.00, -2.0727]),
        ("H", [1.24, 1.16, -2.0727]),
    ];
    MolecularSystem::new(
        rows.iter()
            .map(|r| Element::parse_token(r.0).unwrap())
            .collect(),
        &positions::<Angstrom>(&rows.map(|r| r.1)),
        SpinCharge::default(),
        None,
    )
    .unwrap()
}

/// A molecular saddle search with the rule: RS-I-RFO with the Bofill update and an exact
/// Hessian every 20 iterations reaches the concerted, symmetric Diels-Alder transition state
/// (one negative curvature, both forming bonds 2.125 Angstrom). Over four guesses and three
/// Hessian intervals, Bofill and MSP reach it in every run with and without the pairs (318 and
/// 325 iterations in total with them, 324 and 330 without).
#[test]
fn diels_alder_transition_state_with_updates_from_negative_curvature_pairs() {
    for method in ["rsirfo_bofill", "rsirfo_block_bofill"] {
        let iterations = diels_alder_transition_state(method);
        println!("Diels-Alder transition state: {method} {iterations} iterations");
        assert!(iterations <= 30, "{method}: {iterations}");
    }
}

/// The same transition state with `gpmin`: a Gaussian process on top of the quadratic model
/// of the last exact Hessian (every 20 iterations) takes the place of the Hessian updates.
#[test]
fn diels_alder_transition_state_with_gpmin() {
    let iterations = diels_alder_transition_state("gpmin");
    println!("Diels-Alder transition state: gpmin {iterations} iterations");
    assert!(iterations <= 30, "{iterations}");
}

/// Search the Diels-Alder transition state with `method` from the guess and check it;
/// the number of iterations.
fn diels_alder_transition_state(method: &str) -> usize {
    let calc = Gfn0Xtb::default();
    let settings = OptimizerSettings {
        methods: vec![method.to_string()],
        saddle_order: 1,
        max_iterations: 150,
        exact_hessian_interval: Some(20),
        ..OptimizerSettings::default()
    };
    let mut opt = GeometryOptimizer::new(settings, diels_alder_guess()).unwrap();
    let out = run_optimization(&mut opt, &calc, &mut []).unwrap();
    assert_eq!(out.termination, Some(Termination::Converged));
    let ts = opt.current();
    for (a, b) in [(0, 4), (3, 5)] {
        let d = ts.distance(AtomId::new(a), AtomId::new(b)).value() * Units::BOHR_TO_ANGSTROM;
        assert!((d - 2.125).abs() < 0.01, "forming bond {a}-{b}: {d}");
    }
    let hessian = calc
        .evaluate(ts, &EvalRequest::ENERGY_GRAD_HESS)
        .unwrap()
        .hessian_bohr
        .unwrap();
    let (w, _) = symmetric_eigen(&project_out_tr_rot_hessian(ts, &hessian, false)).unwrap();
    assert_eq!(
        w.iter().filter(|v| **v < -1e-4).count(),
        1,
        "lowest {}",
        w[0]
    );
    out.iterations
}

struct NotANumber;

impl StepGenerator for NotANumber {
    fn name(&self) -> &str {
        "broken"
    }

    fn propose(&mut self, input: &StepInput<'_>) -> MoptResult<Col<f64>> {
        Ok(Col::from_fn(input.gradient.nrows(), |i| {
            if i == 0 {
                f64::NAN
            } else {
                -input.gradient[i]
            }
        }))
    }

    fn reset(&mut self) {}
}

fn registry_with_broken_method() -> MethodRegistry {
    let mut reg = MethodRegistry::default();
    reg.register_generator(
        "broken",
        false,
        Arc::new(|_, _| Ok(Box::new(NotANumber) as Box<dyn StepGenerator>)),
    );
    reg
}

fn ar3() -> MolecularSystem {
    MolecularSystem::new(
        vec![Element::parse_token("Ar").unwrap(); 3],
        &positions::<Angstrom>(&[[0.0, 0.0, 0.0], [3.9, 0.0, 0.0], [1.8, 3.4, 0.2]]),
        SpinCharge::default(),
        None,
    )
    .unwrap()
}

/// A step that is not finite is an error naming the method, for plain variables and for
/// molecules; the structure is left as it was.
#[test]
fn steps_that_are_not_finite_are_refused() {
    let s = VectorSettings {
        methods: vec!["broken".to_string()],
        ..VectorSettings::default()
    };
    let mut opt =
        VectorOptimizer::with_registry(s, col(&[-0.4, 1.3]), &registry_with_broken_method())
            .unwrap();
    let v = MullerBrown2D::default()
        .evaluate(&col(&[-0.4, 1.3]), false)
        .unwrap();
    let err = opt.tell(v.value, v.gradient, None).unwrap_err().to_string();
    assert!(err.contains("'broken' is not finite"), "{err}");
    assert_eq!(opt.current()[0], -0.4);

    let settings = OptimizerSettings {
        methods: vec!["broken".to_string()],
        ..OptimizerSettings::default()
    };
    let mut opt = GeometryOptimizer::with_registries(
        settings,
        ar3(),
        &registry_with_broken_method(),
        &mopt_rs::coord::system::CoordinateRegistry::default(),
    )
    .unwrap();
    let err = run_optimization(&mut opt, &LennardJones, &mut [])
        .unwrap_err()
        .to_string();
    assert!(err.contains("'broken' is not finite"), "{err}");
    assert!(opt
        .current()
        .coords_flat_bohr()
        .iter()
        .all(|v| v.is_finite()));
}

struct BrokenModel;

impl ModelHessian for BrokenModel {
    fn name(&self) -> &str {
        "broken_model"
    }

    fn hessian(&self, sys: &MolecularSystem, _gradient: &Col<f64>) -> MoptResult<Mat<f64>> {
        let n = 3 * sys.n_atoms();
        Ok(Mat::from_fn(
            n,
            n,
            |i, j| if i == j { f64::NAN } else { 0.0 },
        ))
    }
}

struct Notes(Vec<String>);

impl Observer for Notes {
    fn on_iteration(&mut self, r: &IterationReport) -> MoptResult<()> {
        self.0.extend(r.notes.iter().cloned());
        Ok(())
    }
}

/// A model Hessian that is not finite is reported and not used: the optimization goes on with
/// the Hessian it has and converges.
#[test]
fn model_hessians_that_are_not_finite_are_not_used() {
    let settings = OptimizerSettings {
        methods: vec!["rsirfo_bfgs".to_string()],
        max_iterations: 300,
        ..OptimizerSettings::default()
    };
    let mut opt = GeometryOptimizer::new(settings, ar3()).unwrap();
    opt.set_model_hessian(Some(Box::new(BrokenModel)));
    let mut notes = Notes(Vec::new());
    let out = run_optimization(&mut opt, &LennardJones, &mut [&mut notes]).unwrap();
    assert_eq!(out.termination, Some(Termination::Converged));
    assert!(
        notes
            .0
            .iter()
            .any(|n| n.contains("'broken_model' is not finite")),
        "{:?}",
        notes.0
    );
    let h = opt.hessian();
    assert!((0..h.nrows()).all(|i| (0..h.ncols()).all(|j| h[(i, j)].is_finite())));
}

/// Mode kill for a molecule: ethylene and peroxyacetic acid arranged symmetrically (the
/// ethylene bond perpendicular to the plane of the acid). A search for the transition state of
/// the epoxidation converges, with GFN0-xTB, to a saddle point of order 2 that keeps the
/// symmetry (both O-C distances 1.86 Angstrom); with `-modekill` it leaves that point and
/// reaches the transition state, in which the oxygen is closer to one carbon.
#[test]
fn mode_kill_finds_the_epoxidation_transition_state_from_a_symmetric_guess() {
    let along = [0.985, 0.174, 0.0];
    let across = [-0.174, 0.985, 0.0];
    let inner = [0.675, -1.169, 0.0];
    let terminal = [inner[0] + 1.95 * along[0], inner[1] + 1.95 * along[1], 0.0];
    let reach = (1.95_f64 * 1.95 - 0.68 * 0.68).sqrt();
    let middle = [
        terminal[0] + reach * along[0],
        terminal[1] + reach * along[1],
        0.0,
    ];
    let mut rows: Vec<(&str, [f64; 3])> = vec![
        ("C", [-1.50, 0.0, 0.0]),
        ("H", [-1.87, 1.03, 0.0]),
        ("H", [-1.87, -0.51, 0.89]),
        ("H", [-1.87, -0.51, -0.89]),
        ("C", [0.0, 0.0, 0.0]),
        ("O", [0.605, 1.048, 0.0]),
        ("O", inner),
        ("O", terminal),
        ("H", [terminal[0], terminal[1] + 0.98, 0.0]),
    ];
    for side in [1.0, -1.0] {
        let c = [middle[0], middle[1], side * 0.68];
        rows.push(("C", c));
        for hand in [1.0, -1.0] {
            rows.push((
                "H",
                [
                    c[0] + hand * 0.925 * across[0] + 0.10 * along[0],
                    c[1] + hand * 0.925 * across[1] + 0.10 * along[1],
                    c[2] + side * 0.567,
                ],
            ));
        }
    }
    let start = MolecularSystem::new(
        rows.iter()
            .map(|r| Element::parse_token(r.0).unwrap())
            .collect(),
        &positions::<Angstrom>(&rows.iter().map(|r| r.1).collect::<Vec<_>>()),
        SpinCharge::default(),
        None,
    )
    .unwrap();
    let calc = Gfn0Xtb::default();
    let search = |kill: bool| {
        let settings = OptimizerSettings {
            methods: vec!["rsirfo_bofill".to_string()],
            saddle_order: 1,
            max_iterations: 200,
            exact_hessian_interval: Some(20),
            mode_kill: kill.then(ModeKillSettings::default),
            ..OptimizerSettings::default()
        };
        let mut opt = GeometryOptimizer::new(settings, start.clone()).unwrap();
        let out = run_optimization(&mut opt, &calc, &mut []).unwrap();
        assert_eq!(out.termination, Some(Termination::Converged));
        let end = opt.current().clone();
        let hessian = calc
            .evaluate(&end, &EvalRequest::ENERGY_GRAD_HESS)
            .unwrap()
            .hessian_bohr
            .unwrap();
        let (w, _) = symmetric_eigen(&project_out_tr_rot_hessian(&end, &hessian, false)).unwrap();
        let negative = w.iter().filter(|v| **v < -1e-4).count();
        let oc: Vec<f64> = [9, 12]
            .iter()
            .map(|&c| {
                end.distance(AtomId::new(7), AtomId::new(c)).value() * Units::BOHR_TO_ANGSTROM
            })
            .collect();
        (negative, oc, out.iterations)
    };
    let (negative, oc, _) = search(false);
    assert_eq!(negative, 2, "O-C {oc:?}");
    assert!((oc[0] - oc[1]).abs() < 1e-2, "{oc:?}");
    let (negative, oc, iterations) = search(true);
    println!("epoxidation with mode kill: {iterations} iterations, O-C {oc:?}");
    assert_eq!(negative, 1, "O-C {oc:?}");
    assert!((oc[0] - oc[1]).abs() > 0.15, "{oc:?}");
}
