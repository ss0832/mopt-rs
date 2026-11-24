//! End-to-end optimizations on model surfaces with known stationary points.
//!
//! This file is part of mopt-rs.

#![allow(clippy::unwrap_used)]

use mopt_rs::domain::Units;
use mopt_rs::domain::{positions, Angstrom, AtomId, Bohr};
use mopt_rs::domain::{Element, MolecularSystem, SpinCharge};
use mopt_rs::opt::convergence::ConvergenceCriteria;
use mopt_rs::opt::state_machine::{GeometryOptimizer, OptimizerSettings, Termination};
use mopt_rs::opt::stationary::{Minimum, TransitionState};
use mopt_rs::pes::native::muller_brown::{MB_MINIMA, MB_SADDLES};
use mopt_rs::pes::native::{LennardJones, MullerBrown};
use mopt_rs::pes::Calculator;
use mopt_rs::workflow::run_optimization;

fn single_atom(x: f64, y: f64) -> MolecularSystem {
    MolecularSystem::new(
        vec![Element::H],
        &positions::<Bohr>(&[[x, y, 0.0]]),
        SpinCharge::default(),
        None,
    )
    .unwrap()
}

fn mb_settings(methods: &[&str], order: usize, fc: Option<usize>) -> OptimizerSettings {
    OptimizerSettings {
        methods: methods.iter().map(|s| s.to_string()).collect(),
        saddle_order: order,
        max_iterations: 300,
        exact_hessian_interval: fc,
        rigid_body_invariant: false,
        dissociation_distance: None,
        ..OptimizerSettings::default()
    }
}

fn run(
    settings: OptimizerSettings,
    start: MolecularSystem,
    calc: &dyn Calculator,
) -> (Termination, MolecularSystem, usize) {
    let mut opt = GeometryOptimizer::new(settings, start).unwrap();
    let out = run_optimization(&mut opt, calc, &mut []).unwrap();
    (out.termination.unwrap(), out.system, out.iterations)
}

fn distance_xy(sys: &MolecularSystem, target: [f64; 2]) -> f64 {
    let p = sys.position(AtomId::new(0)).value();
    ((p[0] - target[0]).powi(2) + (p[1] - target[1]).powi(2)).sqrt()
}

#[test]
fn rsirfo_finds_muller_brown_minima() {
    let calc = MullerBrown::default();
    for (start, target) in [
        ((-0.4, 1.3), MB_MINIMA[0]),
        ((0.5, 0.1), MB_MINIMA[1]),
        ((0.0, 0.5), MB_MINIMA[2]),
    ] {
        for method in [
            "rsirfo_bfgs",
            "rsirfo_block_bfgs",
            "rsirfo_fsb",
            "rsirfo_block_fsb",
        ] {
            let (t, sys, n) = run(
                mb_settings(&[method], 0, None),
                single_atom(start.0, start.1),
                &calc,
            );
            assert_eq!(t, Termination::Converged, "{method} from {start:?}");
            assert!(
                distance_xy(&sys, target) < 2e-2,
                "{method}: {:?} after {n}",
                sys.position(AtomId::new(0)).value()
            );
        }
    }
}

#[test]
fn rsirfo_finds_muller_brown_saddles_with_exact_hessian() {
    let calc = MullerBrown::default();
    for (start, target) in [((-0.8, 0.6), MB_SADDLES[0]), ((0.2, 0.3), MB_SADDLES[1])] {
        for method in ["rsirfo_bofill", "rsirfo_block_bofill"] {
            let (t, sys, _) = run(
                mb_settings(&[method], 1, Some(1)),
                single_atom(start.0, start.1),
                &calc,
            );
            assert_eq!(t, Termination::Converged, "{method}");
            assert!(
                distance_xy(&sys, target) < 2e-2,
                "{method}: {:?}",
                sys.position(AtomId::new(0)).value()
            );
        }
    }
}

fn argon_cluster() -> MolecularSystem {
    let ar = Element::parse_token("Ar").unwrap();
    // Distorted pentagonal bipyramid (LJ7 global minimum basin).
    let a = 3.8;
    let pos: Vec<[f64; 3]> = (0..5)
        .map(|k| {
            let t = k as f64 * 2.0 * std::f64::consts::PI / 5.0;
            let r = a / (2.0 * (std::f64::consts::PI / 5.0).sin());
            [
                r * t.cos() + 0.1 * (k as f64).sin(),
                r * t.sin(),
                0.05 * k as f64,
            ]
        })
        .chain([[0.0, 0.0, 2.1], [0.2, 0.0, -2.0]])
        .collect();
    MolecularSystem::new(
        vec![ar; 7],
        &positions::<Angstrom>(&pos),
        SpinCharge::default(),
        None,
    )
    .unwrap()
}

#[test]
fn lennard_jones_cluster_relaxes_with_several_methods() {
    let calc = LennardJones;
    // UFF argon: epsilon = D_i = 0.237 kcal/mol; the LJ7 global minimum is -16.505 epsilon.
    let eps = 0.237 * Units::KCAL_MOL_TO_HARTREE;
    for (method, coords) in [
        ("FIRELARS", "cart"),
        ("lbfgs", "cart"),
        ("rsirfo_bfgs", "cart"),
        ("rsirfo_block_fsb", "cart"),
        ("rsirfo_bfgs", "dic"),
        ("rsirfo_block_bfgs", "ric"),
    ] {
        let settings = OptimizerSettings {
            methods: vec![method.to_string()],
            max_iterations: 400,
            coordinates: coords.to_string(),
            // The identity starting Hessian is ~1000x stiffer than argon; Hessian-based methods
            // start from the exact Hessian (`-fc` with a large interval).
            exact_hessian_interval: method.starts_with("rsirfo").then_some(100_000),
            // Argon forces are far below the default thresholds (3e-4 Hartree/Bohr).
            criteria: ConvergenceCriteria::new(1e-7, 5e-8, 1e-3, 5e-4),
            ..OptimizerSettings::default()
        };
        let mut opt = GeometryOptimizer::new(settings, argon_cluster()).unwrap();
        let out = run_optimization(&mut opt, &calc, &mut []).unwrap();
        assert_eq!(
            out.termination,
            Some(Termination::Converged),
            "{method}/{coords}"
        );
        assert!(
            (out.energy.value() / eps + 16.505).abs() < 1e-3,
            "{method}/{coords}: E = {} eps",
            out.energy.value() / eps
        );
    }
}

#[test]
fn stationary_points_are_typed_by_saddle_order() {
    let calc = MullerBrown::default();
    let mut opt = GeometryOptimizer::new(
        mb_settings(&["rsirfo_bofill"], 1, Some(1)),
        single_atom(-0.8, 0.6),
    )
    .unwrap();
    run_optimization(&mut opt, &calc, &mut []).unwrap();
    let ts = opt.stationary_point::<TransitionState>().unwrap();
    assert!(distance_xy(ts.system(), MB_SADDLES[0]) < 2e-2);
    // A transition-state run does not yield a minimum.
    assert!(opt.stationary_point::<Minimum>().is_err());

    let unfinished = GeometryOptimizer::new(
        mb_settings(&["rsirfo_bfgs"], 0, None),
        single_atom(0.5, 0.1),
    )
    .unwrap();
    assert!(unfinished.stationary_point::<Minimum>().is_err());
}
