//! CCQN transition-state searches started inside potential wells (Wu and Wang 2025).
//!
//! This file is part of mopt-rs.

#![allow(clippy::unwrap_used)]

use mopt_rs::domain::Units;
use mopt_rs::domain::{
    positions, Angstrom, AtomId, Bohr, Element, EvalRequest, MolecularSystem, SpinCharge,
};
use mopt_rs::math::linalg::symmetric_eigen;
use mopt_rs::math::projection::{build_tr_rot_basis, project_out_basis_mat};
use mopt_rs::opt::convergence::ConvergenceCriteria;
use mopt_rs::opt::state_machine::{GeometryOptimizer, OptimizerSettings, Termination};
use mopt_rs::opt::stationary::TransitionState;
use mopt_rs::opt::step::ccqn::{CcqnAxis, CcqnSettings};
use mopt_rs::pes::native::muller_brown::{MB_MINIMA, MB_SADDLES};
use mopt_rs::pes::native::{LennardJones, MullerBrown};
use mopt_rs::pes::Calculator;
use mopt_rs::workflow::run_optimization;

fn atom(p: [f64; 2]) -> MolecularSystem {
    MolecularSystem::new(
        vec![Element::H],
        &positions::<Bohr>(&[[p[0], p[1], 0.0]]),
        SpinCharge::default(),
        None,
    )
    .unwrap()
}

/// `saddle + alpha (minimum - saddle)`, the initial guesses of the paper's benchmark.
fn guess(saddle: [f64; 2], minimum: [f64; 2], alpha: f64) -> [f64; 2] {
    [
        saddle[0] + alpha * (minimum[0] - saddle[0]),
        saddle[1] + alpha * (minimum[1] - saddle[1]),
    ]
}

/// Run `method` from `start`; `fc` is the exact-Hessian interval (`-fc`).
fn search(
    method: &str,
    start: [f64; 2],
    axis: CcqnAxis,
    fc: Option<usize>,
) -> (Termination, [f64; 3], usize) {
    let settings = OptimizerSettings {
        methods: vec![method.to_string()],
        saddle_order: 1,
        max_iterations: 300,
        exact_hessian_interval: fc,
        rigid_body_invariant: false,
        dissociation_distance: None,
        ccqn: Some(mb_settings(axis)),
        ..OptimizerSettings::default()
    };
    let mut opt = GeometryOptimizer::new(settings, atom(start)).unwrap();
    let out = run_optimization(&mut opt, &MullerBrown::default(), &mut []).unwrap();
    let t = out.termination.unwrap();
    if t == Termination::Converged {
        let ts = opt.stationary_point::<TransitionState>().unwrap();
        assert_eq!(ts.system().n_atoms(), 1);
    }
    (
        t,
        out.system.position(AtomId::new(0)).value(),
        out.iterations,
    )
}

/// The paper's settings with a narrower cone (cos(phi) = 0.9): the straight-line midpoint is a
/// coarse axis on the curved Muller-Brown valleys (with the paper's 0.5, 30 of 96 starting
/// points fail; with 0.8-0.9, 92-96 of 96 succeed; a narrower cone is the paper's own remedy,
/// SI section 2.2).
///
/// The step sizes are half the paper's (uphill steps and saddle-phase radii of 0.05 to 0.1
/// Angstrom instead of 0.1 to 0.2): the paper's are made for molecules, and on Muller-Brown,
/// whose whole surface spans about 3 by 2 units, steps of 0.19 to 0.38 units can carry the
/// search out of the region of negative curvature and back in every step. From inside the
/// well of minimum A (alpha 0.6) RS-I-RFO with block Bofill and an exact Hessian every 5 steps
/// then alternated between uphill steps (0.19) and saddle steps at the largest radius (0.38) all
/// over the surface: scaling the start by 1 +- 1e-12, 1e-10 or 1e-8 decided between convergence
/// in 127 to 211 iterations and 300 iterations without it (3 of 7 failed; 165 of 168 runs of all
/// methods, starts and perturbations converged). With half the steps all 168 converge, in 8.3
/// iterations on average with the exact Hessian every 5 steps and 7.4 without it (a quarter:
/// also 168, 10.2 and 8.9).
fn mb_settings(axis: CcqnAxis) -> CcqnSettings {
    let mut c = CcqnSettings::new(axis);
    c.cos_phi = 0.9;
    c.uphill_radius *= 0.5;
    c.trust.initial *= 0.5;
    c.trust.min *= 0.5;
    c.trust.max *= 0.5;
    c
}

fn dist(p: [f64; 3], q: [f64; 2]) -> f64 {
    ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2)).sqrt()
}

/// Both saddle-region steps (RS-P-RFO, RS-I-RFO) with TS-BFGS and with block Bofill.
const METHODS: [&str; 4] = [
    "ccqn",
    "ccqn_rsirfo",
    "ccqn_block_bofill",
    "ccqn_rsirfo_block_bofill",
];

/// `(saddle, minimum of the start side, product minimum)` for both Muller-Brown saddles.
const CASES: [([f64; 2], [f64; 2], [f64; 2]); 2] = [
    (MB_SADDLES[1], MB_MINIMA[2], MB_MINIMA[1]),
    (MB_SADDLES[0], MB_MINIMA[0], MB_MINIMA[2]),
];

fn check_all(fc: Option<usize>) {
    for (saddle, from, to) in CASES {
        for alpha in [0.1, 0.3, 0.6] {
            let start = guess(saddle, from, alpha);
            for method in METHODS {
                // Axis toward the midpoint of the straight line to the product (eqn (18)).
                let axis = CcqnAxis::linear_midpoint(&atom(start), &atom(to)).unwrap();
                let (t, p, n) = search(method, start, axis, fc);
                let tag = format!("{method}, fc {fc:?}, saddle {saddle:?}, alpha {alpha}");
                assert_eq!(t, Termination::Converged, "{tag}: {p:?} after {n}");
                assert!(dist(p, saddle) < 2e-2, "{tag}: {p:?} after {n}");
            }
        }
    }
}

#[test]
fn ccqn_reaches_the_saddle_from_inside_the_wells_with_exact_hessian_every_5_steps() {
    check_all(Some(5));
}

#[test]
fn ccqn_reaches_the_saddle_from_an_identity_hessian() {
    // The paper's setting: scaled identity start, quasi-Newton updates only.
    check_all(None);
}

#[test]
fn ccqn_succeeds_where_rsirfo_block_bofill_alone_fails() {
    // Deep in the well of minimum A with updated Hessians only: RS-I-RFO with block Bofill does
    // not find the saddle, CCQN with the same updater does.
    let (saddle, from, to) = CASES[1];
    let start = guess(saddle, from, 0.6);
    let axis = || CcqnAxis::linear_midpoint(&atom(start), &atom(to)).unwrap();
    let (t, p, _) = search("rsirfo_block_bofill", start, axis(), None);
    assert!(t != Termination::Converged || dist(p, saddle) > 2e-2);
    for method in ["ccqn_block_bofill", "ccqn_rsirfo_block_bofill"] {
        let (t, p, n) = search(method, start, axis(), None);
        assert_eq!(t, Termination::Converged, "{method}: {p:?} after {n}");
        assert!(dist(p, saddle) < 2e-2, "{method}: {p:?} after {n}");
    }
}

#[test]
fn ccqn_with_a_fixed_direction() {
    // Saddle between minima A and C, starting on the A side, axis from A toward C.
    let (saddle, from, to) = CASES[1];
    let start = guess(saddle, from, 0.3);
    let d = [to[0] - from[0], to[1] - from[1], 0.0];
    for method in METHODS {
        let axis = CcqnAxis::Direction(faer::Col::from_fn(3, |i| d[i]));
        let (t, p, n) = search(method, start, axis, Some(5));
        assert_eq!(t, Termination::Converged, "{method}: {p:?} after {n}");
        assert!(dist(p, saddle) < 2e-2, "{method}: {p:?} after {n}");
    }
}

fn argon_cluster() -> MolecularSystem {
    let ar = Element::parse_token("Ar").unwrap();
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

/// Number of negative eigenvalues of the exact Hessian without rigid-body modes, and the
/// largest gradient component.
fn curvature(sys: &MolecularSystem) -> (usize, f64) {
    let out = LennardJones
        .evaluate(sys, &EvalRequest::ENERGY_GRAD_HESS)
        .unwrap();
    let d = build_tr_rot_basis(sys, false);
    let h = project_out_basis_mat(&d, &out.hessian_bohr.unwrap());
    let (ev, _) = symmetric_eigen(&h).unwrap();
    let gmax = out
        .gradient_bohr
        .iter()
        .fold(0.0_f64, |m, v| m.max(v.abs()));
    (ev.iter().filter(|v| **v < -1e-6).count(), gmax)
}

#[test]
fn ccqn_finds_a_first_order_saddle_of_lj7_in_cartesian_and_dic_coordinates() {
    let eps = 0.237 * Units::KCAL_MOL_TO_HARTREE;
    let criteria = ConvergenceCriteria::new(eps * 1e-3, eps * 0.7e-3, 1e-3, 0.7e-3);
    let mut relax = GeometryOptimizer::new(
        OptimizerSettings {
            methods: vec!["rsirfo_bfgs".into()],
            criteria,
            max_iterations: 500,
            ..OptimizerSettings::default()
        },
        argon_cluster(),
    )
    .unwrap();
    let minimum = run_optimization(&mut relax, &LennardJones, &mut [])
        .unwrap()
        .system;
    assert_eq!(curvature(&minimum).0, 0);
    let bond = vec![(AtomId::new(0), AtomId::new(1))];
    for coords in ["cart", "dic"] {
        for method in ["ccqn", "ccqn_rsirfo", "ccqn_block_bofill"] {
            let mut c = CcqnSettings::new(CcqnAxis::ReactiveBonds(bond.clone()));
            c.cos_phi = 0.5;
            let settings = OptimizerSettings {
                methods: vec![method.into()],
                saddle_order: 1,
                criteria,
                max_iterations: 300,
                coordinates: coords.into(),
                exact_hessian_interval: Some(5),
                dissociation_distance: None,
                ccqn: Some(c),
                ..OptimizerSettings::default()
            };
            let mut opt = GeometryOptimizer::new(settings, minimum.clone()).unwrap();
            let out = run_optimization(&mut opt, &LennardJones, &mut []).unwrap();
            let (negatives, gmax) = curvature(&out.system);
            let tag = format!("{method} in {coords}: {} iterations", out.iterations);
            assert_eq!(out.termination, Some(Termination::Converged), "{tag}");
            assert_eq!(negatives, 1, "{tag}");
            assert!(gmax < 1e-6, "{tag}: {gmax}");
            // The saddle 1.909 epsilon above the LJ7 minimum (-16.5054 epsilon).
            assert!((out.energy.value() / eps + 14.5969).abs() < 1e-3, "{tag}");
        }
    }
}
