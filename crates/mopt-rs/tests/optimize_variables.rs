//! Optimization of plain variables (no atoms, any number): the two-variable Muller-Brown
//! surface (its three minima and two saddle points with several methods, and the same
//! trajectory as the one-atom Muller-Brown model), a five-variable quadratic form, and
//! expression objectives.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used, clippy::expect_used)]

use faer::{Col, Mat};
use mopt_rs::domain::{positions, AtomId, Bohr, Element, MolecularSystem, SpinCharge};
use mopt_rs::opt::mode_kill::ModeKillSettings;
use mopt_rs::opt::state_machine::{GeometryOptimizer, OptimizerSettings, Termination};
use mopt_rs::opt::step::ccqn::{CcqnAxis, CcqnSettings};
use mopt_rs::opt::vector::{VectorOptimizer, VectorSettings};
use mopt_rs::pes::native::muller_brown::{MB_MINIMA, MB_SADDLES};
use mopt_rs::pes::native::MullerBrown;
use mopt_rs::pes::objective::{MullerBrown2D, Objective, ObjectiveRegistry, Quadratic};
use mopt_rs::pes::registry::CalculatorOptions;
use mopt_rs::workflow::{run_optimization, run_vector_optimization};

fn col(v: &[f64]) -> Col<f64> {
    Col::from_fn(v.len(), |i| v[i])
}

fn settings(method: &str, order: usize, fc: Option<usize>) -> VectorSettings {
    VectorSettings {
        methods: vec![method.to_string()],
        saddle_order: order,
        max_iterations: 300,
        exact_hessian_interval: fc,
        ..VectorSettings::default()
    }
}

fn solve(
    s: VectorSettings,
    obj: &dyn Objective,
    start: &[f64],
) -> (Option<Termination>, Col<f64>, usize, Vec<Col<f64>>) {
    let mut opt = VectorOptimizer::new(s, col(start)).unwrap();
    let mut path = Vec::new();
    let out = run_vector_optimization(&mut opt, obj, &mut |r| {
        path.push(r.x.clone());
        Ok(())
    })
    .unwrap();
    (out.termination, out.x, out.iterations, path)
}

fn dist(x: &Col<f64>, p: [f64; 2]) -> f64 {
    ((x[0] - p[0]).powi(2) + (x[1] - p[1]).powi(2)).sqrt()
}

#[test]
fn muller_brown_minima_with_several_methods() {
    let mb = MullerBrown2D::default();
    for (start, target) in [
        ([-0.4, 1.3], MB_MINIMA[0]),
        ([0.5, 0.1], MB_MINIMA[1]),
        ([0.0, 0.5], MB_MINIMA[2]),
    ] {
        for method in [
            "rsirfo_bfgs",
            "rsirfo_block_fsb",
            "FIRE",
            "lbfgs",
            "FIRELARS",
        ] {
            let (t, x, n, _) = solve(settings(method, 0, None), &mb, &start);
            assert_eq!(t, Some(Termination::Converged), "{method} from {start:?}");
            assert!(dist(&x, target) < 2e-2, "{method}: {x:?} after {n}");
        }
    }
}

#[test]
fn muller_brown_saddles_with_rsirfo_and_ccqn() {
    let mb = MullerBrown2D::default();
    for (start, target) in [([-0.8, 0.6], MB_SADDLES[0]), ([0.2, 0.3], MB_SADDLES[1])] {
        for method in ["rsirfo_bofill", "rsirfo_block_bofill", "rsirfo_fsb"] {
            let (t, x, n, _) = solve(settings(method, 1, Some(1)), &mb, &start);
            assert_eq!(t, Some(Termination::Converged), "{method} from {start:?}");
            assert!(dist(&x, target) < 2e-2, "{method}: {x:?} after {n}");
        }
    }
    // CCQN with a fixed axis from minimum C toward B (no atoms needed).
    let (from, to) = (MB_MINIMA[2], MB_MINIMA[1]);
    let start = [
        MB_SADDLES[1][0] + 0.3 * (from[0] - MB_SADDLES[1][0]),
        MB_SADDLES[1][1] + 0.3 * (from[1] - MB_SADDLES[1][1]),
    ];
    for method in ["ccqn", "ccqn_rsirfo"] {
        let mut c = CcqnSettings::new(CcqnAxis::Direction(col(&[
            to[0] - from[0],
            to[1] - from[1],
        ])));
        c.cos_phi = 0.9;
        let s = VectorSettings {
            ccqn: Some(c),
            ..settings(method, 1, Some(5))
        };
        let (t, x, n, _) = solve(s, &mb, &start);
        assert_eq!(t, Some(Termination::Converged), "{method}");
        assert!(dist(&x, MB_SADDLES[1]) < 2e-2, "{method}: {x:?} after {n}");
    }
    // An axis that needs atoms is refused with a clear message.
    let s = VectorSettings {
        ccqn: Some(CcqnSettings::new(CcqnAxis::ReactiveBonds(vec![(
            AtomId::new(0),
            AtomId::new(1),
        )]))),
        ..settings("ccqn", 1, None)
    };
    let mut opt = VectorOptimizer::new(s, col(&start)).unwrap();
    let v = mb.evaluate(&col(&start), false).unwrap();
    let err = opt.tell(v.value, v.gradient, None).unwrap_err().to_string();
    assert!(err.contains("needs atoms"), "{err}");
}

#[test]
fn same_trajectory_as_the_one_atom_muller_brown_model() {
    let mb = MullerBrown2D::default();
    let start = [-0.4, 1.3];
    for method in ["rsirfo_bfgs", "FIRE", "lbfgs"] {
        let (_, _, _, path) = solve(settings(method, 0, None), &mb, &start);
        let s = OptimizerSettings {
            methods: vec![method.to_string()],
            max_iterations: 300,
            rigid_body_invariant: false,
            dissociation_distance: None,
            ..OptimizerSettings::default()
        };
        let atom = MolecularSystem::new(
            vec![Element::H],
            &positions::<Bohr>(&[[start[0], start[1], 0.0]]),
            SpinCharge::default(),
            None,
        )
        .unwrap();
        let mut opt = GeometryOptimizer::new(s, atom).unwrap();
        let mut atom_path = Vec::new();
        struct Track<'a>(&'a mut Vec<[f64; 3]>);
        impl mopt_rs::workflow::Observer for Track<'_> {
            fn on_iteration(
                &mut self,
                r: &mopt_rs::opt::state_machine::IterationReport,
            ) -> mopt_rs::error::MoptResult<()> {
                self.0.push(r.system.position(AtomId::new(0)).value());
                Ok(())
            }
        }
        run_optimization(
            &mut opt,
            &MullerBrown::default(),
            &mut [&mut Track(&mut atom_path)],
        )
        .unwrap();
        assert_eq!(path.len(), atom_path.len(), "{method}");
        for (a, b) in path.iter().zip(&atom_path) {
            assert!(
                (a[0] - b[0]).abs() < 1e-10 && (a[1] - b[1]).abs() < 1e-10,
                "{method}"
            );
        }
    }
}

#[test]
fn five_variable_quadratic_form() {
    let n = 5;
    let a = Mat::from_fn(n, n, |i, j| {
        if i == j {
            1.0 + i as f64
        } else {
            0.1 / (1.0 + (i + j) as f64)
        }
    });
    let center = col(&[0.3, -0.2, 0.5, 1.0, -0.7]);
    let q = Quadratic::new(a, center.clone(), -1.0).unwrap();
    for method in ["rsirfo_bfgs", "lbfgs", "FIRE"] {
        let (t, x, it, _) = solve(settings(method, 0, None), &q, &[0.0; 5]);
        assert_eq!(t, Some(Termination::Converged), "{method}");
        let err = (0..n).map(|i| (x[i] - center[i]).abs()).fold(0.0, f64::max);
        assert!(err < 1e-3, "{method}: {err} after {it}");
    }
    // A saddle of order 1 of an indefinite form.
    let b = Mat::from_fn(3, 3, |i, j| if i == j { [-1.0, 2.0, 3.0][i] } else { 0.0 });
    let q = Quadratic::new(b, col(&[0.1, 0.2, 0.3]), 0.0).unwrap();
    let (t, x, _, _) = solve(settings("rsirfo_bofill", 1, Some(1)), &q, &[0.0, 0.0, 0.0]);
    assert_eq!(t, Some(Termination::Converged));
    assert!(
        (x[0] - 0.1).abs() < 1e-4 && (x[2] - 0.3).abs() < 1e-4,
        "{x:?}"
    );
}

#[test]
fn expression_objective_from_the_registry() {
    let reg = ObjectiveRegistry::builtin();
    let obj = reg
        .create(
            "expr:(q1 - 1)^2 + 2*(q2 + 0.5)^2 + 0.1*q1*q2 + (q3 - q4)^2 + q4^2",
            &CalculatorOptions::default(),
        )
        .unwrap();
    let (t, x, _, _) = solve(settings("rsirfo_bfgs", 0, None), obj.as_ref(), &[0.0; 4]);
    assert_eq!(t, Some(Termination::Converged));
    let g = obj.evaluate(&x, false).unwrap().gradient;
    assert!(g.iter().all(|v| v.abs() < 1e-3), "{g:?}");
    assert!(x[3].abs() < 1e-3 && x[2].abs() < 1e-3);
    // Wrong dimension.
    let mb = reg.create("mb2d", &CalculatorOptions::default()).unwrap();
    let mut opt = VectorOptimizer::new(settings("FIRE", 0, None), col(&[0.0; 3])).unwrap();
    assert!(run_vector_optimization(&mut opt, mb.as_ref(), &mut |_| Ok(())).is_err());
}

/// Mode kill: by symmetry a search for a first-order saddle ends at a saddle of order 2 (the
/// gradient along `q2` vanishes all the way), and a minimization that starts on a saddle point
/// never moves. With the option the Hessian is checked at convergence and the surplus modes
/// are left.
#[test]
fn mode_kill_leaves_stationary_points_of_too_high_an_order() {
    let reg = ObjectiveRegistry::builtin();
    // Curvatures -1, -0.6, 2 at the origin; first-order saddles at q2 = +-sqrt(0.6).
    let obj = reg
        .create(
            "expr:-0.5*q1^2 - 0.3*q2^2 + 0.25*q2^4 + q3^2",
            &CalculatorOptions::default(),
        )
        .unwrap();
    let run = |order: usize, kill: bool, start: &[f64]| {
        let s = VectorSettings {
            mode_kill: kill.then(ModeKillSettings::default),
            ..settings("rsirfo_bofill", order, Some(1000))
        };
        let mut opt = VectorOptimizer::new(s, col(start)).unwrap();
        let mut notes = Vec::new();
        let out = run_vector_optimization(&mut opt, obj.as_ref(), &mut |r| {
            notes.extend(r.notes.clone());
            Ok(())
        })
        .unwrap();
        assert_eq!(out.termination, Some(Termination::Converged));
        (out.x, notes)
    };
    let start = [0.3, 0.0, 0.4];
    let (x, notes) = run(1, false, &start);
    assert!(x[1].abs() < 1e-8 && x[0].abs() < 1e-3, "{x:?}");
    assert!(notes.iter().all(|n| !n.contains("Mode kill")));
    let (x, notes) = run(1, true, &start);
    assert!((x[1].abs() - 0.6_f64.sqrt()).abs() < 1e-3, "{x:?}");
    assert!(x[0].abs() < 1e-3 && x[2].abs() < 1e-3, "{x:?}");
    assert!(
        notes.iter().any(|n| n.contains("2 imaginary mode(s)")),
        "{notes:?}"
    );
    // The Hessian for the check was asked for at the converged point.
    assert!(notes.iter().any(|n| n.contains("requested here")));
    // A point of the order asked for is accepted.
    let (again, notes) = run(1, true, &[x[0], x[1], x[2]]);
    assert!((again[1] - x[1]).abs() < 1e-3);
    assert!(notes.iter().all(|n| !n.contains("leaving it")));
    // A minimization that starts exactly on the saddle of a double well.
    let well = reg
        .create("expr:(q1^2 - 1)^2 + q2^2", &CalculatorOptions::default())
        .unwrap();
    let minimize = |kill: bool| {
        let s = VectorSettings {
            mode_kill: kill.then(ModeKillSettings::default),
            ..settings("rsirfo_bfgs", 0, None)
        };
        let mut opt = VectorOptimizer::new(s, col(&[0.0, 0.0])).unwrap();
        let out = run_vector_optimization(&mut opt, well.as_ref(), &mut |_| Ok(())).unwrap();
        assert_eq!(out.termination, Some(Termination::Converged));
        out.x
    };
    assert!(minimize(false)[0].abs() < 1e-12);
    assert!((minimize(true)[0].abs() - 1.0).abs() < 1e-3);
}
