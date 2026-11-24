//! The optimizers ported from MultiOptPy's `Optimizer/` directory, each on the two-variable
//! Muller-Brown surface, on a five-variable quadratic form, and on the Lennard-Jones Ar7
//! cluster (rigid-body motion included).
//!
//! MultiOptPy: `Optimizer/` (tests of the ported classes; no counterpart file).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use faer::{Col, Mat};
use mopt_rs::domain::{positions, Angstrom, Element, MolecularSystem, SpinCharge};
use mopt_rs::opt::convergence::ConvergenceCriteria;
use mopt_rs::opt::state_machine::{Action, GeometryOptimizer, OptimizerSettings, Termination};
use mopt_rs::opt::vector::{VectorOptimizer, VectorSettings};
use mopt_rs::pes::native::muller_brown::{MB_MINIMA, MB_SADDLES};
use mopt_rs::pes::native::LennardJones;
use mopt_rs::pes::objective::{MullerBrown2D, Objective, Quadratic};
use mopt_rs::pes::Calculator;
use mopt_rs::workflow::run_vector_optimization;

fn col(v: &[f64]) -> Col<f64> {
    Col::from_fn(v.len(), |i| v[i])
}

struct Outcome {
    termination: Option<Termination>,
    x: Col<f64>,
    iterations: usize,
}

fn solve(
    method: &str,
    order: usize,
    fc: Option<usize>,
    max_iterations: usize,
    obj: &dyn Objective,
    start: &[f64],
) -> Outcome {
    let s = VectorSettings {
        methods: vec![method.to_string()],
        saddle_order: order,
        max_iterations,
        exact_hessian_interval: fc,
        ..VectorSettings::default()
    };
    let mut opt = VectorOptimizer::new(s, col(start)).unwrap();
    let out = run_vector_optimization(&mut opt, obj, &mut |_| Ok(())).unwrap();
    Outcome {
        termination: out.termination,
        x: out.x,
        iterations: out.iterations,
    }
}

fn dist(x: &Col<f64>, p: [f64; 2]) -> f64 {
    ((x[0] - p[0]).powi(2) + (x[1] - p[1]).powi(2)).sqrt()
}

/// `E = (x - c)^T A (x - c) / 2 - 1` with eigenvalues between `low` and about `high`.
fn quadratic(low: f64, high: f64) -> (Quadratic, Col<f64>) {
    let n = 5;
    let a = Mat::from_fn(n, n, |i, j| {
        if i == j {
            low + (high - low) * i as f64 / 4.0
        } else {
            0.02 * low / (1.0 + (i + j) as f64)
        }
    });
    let center = col(&[0.3, -0.2, 0.5, 1.0, -0.7]);
    (Quadratic::new(a, center.clone(), -1.0).unwrap(), center)
}

fn ar7() -> MolecularSystem {
    let pos = [
        [1.9500, 0.0000, 0.0000],
        [0.6026, 1.8546, 0.0300],
        [-1.5776, 1.1462, 0.0700],
        [-1.5776, -1.1462, 0.1100],
        [0.6026, -1.8546, 0.1500],
        [0.0000, 0.0000, 2.1000],
        [0.2000, 0.0000, -2.0000],
    ];
    MolecularSystem::new(
        vec![Element::parse_token("Ar").unwrap(); 7],
        &positions::<Angstrom>(&pos),
        SpinCharge::default(),
        None,
    )
    .unwrap()
}

struct Ar7Run {
    termination: Termination,
    first_energy: f64,
    energy: f64,
    max_gradient: f64,
    iterations: usize,
}

/// Optimize Ar7 on the Lennard-Jones surface (rigid-body motion is removed by the optimizer).
fn solve_ar7(method: &str, max_iterations: usize, tight: bool) -> Ar7Run {
    let settings = OptimizerSettings {
        methods: vec![method.to_string()],
        max_iterations,
        criteria: if tight {
            ConvergenceCriteria::TIGHT
        } else {
            ConvergenceCriteria::NORMAL
        },
        ..OptimizerSettings::default()
    };
    let mut opt = GeometryOptimizer::new(settings, ar7()).unwrap();
    let mut run = Ar7Run {
        termination: Termination::MaxIterations,
        first_energy: f64::NAN,
        energy: f64::NAN,
        max_gradient: f64::NAN,
        iterations: 0,
    };
    loop {
        match opt.ask() {
            Action::Finished(t) => {
                run.termination = t;
                run.iterations = opt.iterations();
                return run;
            }
            Action::Evaluate(tasks) => {
                let out = LennardJones
                    .evaluate(&tasks[0].system, &tasks[0].request)
                    .unwrap();
                run.energy = out.energy.value();
                if run.first_energy.is_nan() {
                    run.first_energy = run.energy;
                }
                run.max_gradient = out
                    .gradient_bohr
                    .iter()
                    .fold(0.0_f64, |m, v| m.max(v.abs()));
                opt.tell_one(out).unwrap();
            }
        }
    }
}

/// Energy of the Ar7 minimum reached from the test geometry.
fn ar7_reference_energy() -> f64 {
    let run = solve_ar7("rsirfo_bfgs", 500, true);
    assert_eq!(run.termination, Termination::Converged);
    run.energy
}

/// What a minimizer is checked against.
#[derive(Clone, Copy)]
struct Case {
    /// Iteration limit.
    limit: usize,
    /// Scale of the Muller-Brown surface (the published surface times this factor).
    mb_scale: f64,
    /// Lowest and highest curvature of the quadratic form.
    quad: (f64, f64),
    /// Whether Ar7 is run with the tight criteria and compared with the minimum energy.
    tight: bool,
}

/// The method converges to a Muller-Brown minimum, to the minimum of the quadratic form, and on
/// Ar7. Returns the failures.
fn check_minimizer(method: &str, case: Case, e_ref: f64) -> Vec<String> {
    let mut failures = Vec::new();
    let mb = MullerBrown2D {
        energy_scale: case.mb_scale,
    };
    let out = solve(method, 0, None, case.limit, &mb, &[-0.4, 1.3]);
    if out.termination != Some(Termination::Converged) || dist(&out.x, MB_MINIMA[0]) > 2e-2 {
        failures.push(format!(
            "{method} on Muller-Brown: {:?} at ({:.4}, {:.4}) after {}",
            out.termination, out.x[0], out.x[1], out.iterations
        ));
    }
    let mb_iterations = out.iterations;
    let (q, center) = quadratic(case.quad.0, case.quad.1);
    // Not from the origin: LARS scales the step with the norm of the point.
    let out = solve(method, 0, None, case.limit, &q, &[-0.5, 0.4, 0.1, 0.2, 0.3]);
    let err = (0..5)
        .map(|i| (out.x[i] - center[i]).abs())
        .fold(0.0, f64::max);
    if out.termination != Some(Termination::Converged) || err > 5e-3 {
        failures.push(format!(
            "{method} on the quadratic form: {:?}, error {err:.2e} after {}",
            out.termination, out.iterations
        ));
    }
    let quad_iterations = out.iterations;
    let run = solve_ar7(method, case.limit, case.tight);
    let energy_ok = if case.tight {
        (run.energy - e_ref).abs() < 3e-6
    } else {
        run.energy < run.first_energy
    };
    if run.termination != Termination::Converged || !energy_ok {
        failures.push(format!(
            "{method} on Ar7: {:?}, E = {:.8} (minimum {e_ref:.8}, start {:.8}), max gradient \
             {:.2e} after {}",
            run.termination, run.energy, run.first_energy, run.max_gradient, run.iterations
        ));
    }
    eprintln!(
        "{method}: Muller-Brown {mb_iterations}, quadratic {quad_iterations}, Ar7 {} iterations",
        run.iterations
    );
    failures
}

fn check_all(methods: &[&str], case: Case) {
    let e_ref = ar7_reference_energy();
    let failures: Vec<String> = methods
        .iter()
        .flat_map(|m| check_minimizer(m, case, e_ref))
        .collect();
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn fixed_step_and_inertial_methods_minimize() {
    // Steepest descent with a fixed factor of 1 needs curvatures below 2: the Muller-Brown
    // surface is scaled so that its stiffest curvature at the minimum is 0.7.
    check_all(
        &["fire2", "abcfire"],
        Case {
            limit: 3000,
            mb_scale: 1e-3,
            quad: (1.0, 5.0),
            tight: true,
        },
    );
    check_all(
        &["gradientdescent", "steepest_descent"],
        Case {
            limit: 3000,
            mb_scale: 2e-4,
            quad: (0.3, 1.2),
            tight: false,
        },
    );
}

#[test]
fn mass_weighted_steepest_descent_follows_the_path_downhill() {
    // Without atoms the masses are 1: the same as plain steepest descent.
    let mb = MullerBrown2D { energy_scale: 2e-4 };
    let a = solve("mwgradientdescent", 0, None, 3000, &mb, &[-0.4, 1.3]);
    let b = solve("gradientdescent", 0, None, 3000, &mb, &[-0.4, 1.3]);
    assert_eq!(a.termination, Some(Termination::Converged));
    assert_eq!(a.iterations, b.iterations);
    assert!(dist(&a.x, MB_MINIMA[0]) < 2e-2);
    // Argon (39.9 amu): steps of -g / m, an Euler integration of the steepest-descent path in
    // mass-weighted coordinates. The energy falls; the minimum is far beyond a few hundred steps.
    let run = solve_ar7("mwgradientdescent", 400, false);
    let plain = solve_ar7("gradientdescent", 400, false);
    assert!(run.energy < run.first_energy);
    assert!(
        run.energy > plain.energy,
        "shorter steps than plain descent"
    );
}

#[test]
fn adam_family_minimizes() {
    check_all(
        &[
            "adam",
            "adamlars",
            "adamlookahead",
            "adamlookaheadlars",
            "ranger",
            "rangerlars",
            "radam",
            "adabelief",
            "eve",
            "prodigy",
        ],
        Case {
            limit: 8000,
            mb_scale: 1e-3,
            quad: (0.3, 1.2),
            tight: false,
        },
    );
}

#[test]
fn conjugate_gradients_minimize() {
    check_all(
        &["cg", "cg_pr", "cg_fr", "cg_hs", "cg_dy"],
        Case {
            limit: 1500,
            mb_scale: 1e-3,
            quad: (1.0, 5.0),
            tight: true,
        },
    );
}

#[test]
fn hessian_based_and_trust_region_methods_minimize() {
    check_all(
        &[
            "newton_bfgs",
            "mnewton_bfgs",
            "tr_lbfgs",
            "rsprfo_bfgs",
            "rsprfo_block_fsb",
            "crsirfo_bfgs",
            "mf_rsirfo_bfgs",
            "rsirfo_bfgs_trim",
            "rsprfo_fsb_trim",
        ],
        Case {
            limit: 1500,
            mb_scale: 1e-3,
            quad: (1.0, 5.0),
            tight: true,
        },
    );
}

#[test]
fn diis_family_and_step_modifiers_minimize() {
    check_all(
        &[
            "rsirfo_bfgs_gdiis",
            "rsirfo_bfgs_c2diis",
            "rsirfo_bfgs_kdiis",
            "rsirfo_bfgs_ediis",
            "rsirfo_bfgs_adiis",
            "rsirfo_bfgs_gediis",
            "lbfgs_gdiis",
            "lbfgs_kdiis",
            "lbfgs_gediis",
            "rsirfo_bfgs_geodesic_step",
        ],
        Case {
            limit: 1500,
            mb_scale: 1e-3,
            quad: (1.0, 5.0),
            tight: true,
        },
    );
    check_all(
        &[
            "rsirfo_bfgs_coordinate_locking",
            "lbfgs_coordinate_locking",
            // Factors up to 5 overshoot with Newton-type steps: first-order methods only.
            "fire_component_wise_scaling",
            "fire2_component_wise_scaling",
            "fire_coordinate_locking",
            "lbfgs_perturbation",
            "fire_perturbation",
        ],
        Case {
            limit: 3000,
            mb_scale: 1e-3,
            quad: (1.0, 5.0),
            tight: false,
        },
    );
}

#[test]
fn line_search_and_learned_step_modifiers_minimize() {
    check_all(
        &[
            "rsirfo_bfgs_linesearch",
            "lbfgs_linesearch",
            "gradientdescent_linesearch",
            "rsirfo_bfgs_gpr_step",
            "lbfgs_gpr_step",
        ],
        Case {
            limit: 1500,
            mb_scale: 1e-3,
            quad: (1.0, 5.0),
            tight: true,
        },
    );
    check_all(
        &[
            "rsirfo_bfgs_gan_step",
            "lbfgs_gan_step",
            "rsirfo_bfgs_rl_step",
            "lbfgs_rl_step",
            "fire_rl_step",
        ],
        Case {
            limit: 3000,
            mb_scale: 1e-3,
            quad: (1.0, 5.0),
            tight: false,
        },
    );
}

#[test]
fn crsirfo_is_rsirfo_under_the_optimizers_constraints() {
    // Constraints are held by the optimizer for every method: without any, the constrained
    // RS-I-RFO name gives the trajectory of RS-I-RFO.
    let a = solve_ar7("crsirfo_bfgs", 500, true);
    let b = solve_ar7("rsirfo_bfgs", 500, true);
    assert_eq!(a.termination, Termination::Converged);
    assert_eq!(a.iterations, b.iterations);
    assert_eq!(a.energy, b.energy);
}

#[test]
fn gpmin_minimizes() {
    check_all(
        &["gpmin"],
        Case {
            limit: 300,
            mb_scale: 1e-3,
            quad: (1.0, 5.0),
            tight: false,
        },
    );
}

/// Saddle searches on the Muller-Brown surface (exact Hessian every `fc` iterations) and on an
/// indefinite quadratic form.
fn check_saddle_method(method: &str, fc: usize) -> Vec<String> {
    let mut failures = Vec::new();
    let mb = MullerBrown2D::default();
    for (start, target) in [([-0.8, 0.6], MB_SADDLES[0]), ([0.2, 0.3], MB_SADDLES[1])] {
        let out = solve(method, 1, Some(fc), 300, &mb, &start);
        if out.termination != Some(Termination::Converged) || dist(&out.x, target) > 2e-2 {
            failures.push(format!(
                "{method} from {start:?}: {:?} at ({:.4}, {:.4}) after {}",
                out.termination, out.x[0], out.x[1], out.iterations
            ));
        }
    }
    let b = Mat::from_fn(3, 3, |i, j| if i == j { [-1.0, 2.0, 3.0][i] } else { 0.05 });
    let center = col(&[0.1, 0.2, 0.3]);
    let q = Quadratic::new(b, center.clone(), 0.0).unwrap();
    let out = solve(method, 1, Some(fc), 300, &q, &[0.0, 0.0, 0.0]);
    let err = (0..3)
        .map(|i| (out.x[i] - center[i]).abs())
        .fold(0.0, f64::max);
    if out.termination != Some(Termination::Converged) || err > 1e-3 {
        failures.push(format!(
            "{method} on the indefinite form: {:?}, error {err:.2e} after {}",
            out.termination, out.iterations
        ));
    }
    failures
}

#[test]
fn gpmin_finds_saddle_points() {
    // With an exact Hessian at every iteration, and with one every 20 iterations (the
    // surface fitted to the gradients has to supply the curvature in between).
    let failures: Vec<String> = [1, 20]
        .iter()
        .flat_map(|&fc| check_saddle_method("gpmin", fc))
        .collect();
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
    // Iterations with a Hessian every 20 iterations, beside RS-I-RFO with block Bofill updates.
    let mb = MullerBrown2D::default();
    let count = |method: &str, start: [f64; 2]| solve(method, 1, Some(20), 300, &mb, &start);
    let (a, b) = (count("gpmin", [-0.8, 0.6]), count("gpmin", [0.2, 0.3]));
    let (c, d) = (
        count("rsirfo_block_bofill", [-0.8, 0.6]),
        count("rsirfo_block_bofill", [0.2, 0.3]),
    );
    println!(
        "Muller-Brown saddles: gpmin {} and {}, rsirfo_block_bofill {} and {} iterations",
        a.iterations, b.iterations, c.iterations, d.iterations
    );
    assert!(a.iterations <= c.iterations + 5 && b.iterations <= d.iterations + 5);
}

/// A saddle of order 2 of an anharmonic function of three variables (at 0.5, -0.3, 0.4),
/// with the Hessian given once at the start or at every iteration: GPMin beside RS-I-RFO with
/// the Bofill update from one step and from all stored steps (block Bofill).
#[test]
fn gpmin_and_block_bofill_at_a_saddle_of_order_two() {
    let reg = mopt_rs::pes::objective::ObjectiveRegistry::builtin();
    let obj = reg
        .create(
            "expr:-0.5*(q1-0.5)^2 - 0.3*(q2+0.3)^2 + (q3-0.4)^2 + 0.3*(q1-0.5)^2*(q3-0.4) \
             + 0.2*(q2+0.3)^3 + 0.1*(q1-0.5)^4",
            &mopt_rs::pes::registry::CalculatorOptions::default(),
        )
        .unwrap();
    let starts = [[0.15, -0.05, 0.75], [-0.3, 0.3, 1.0], [1.2, -0.9, -0.2]];
    let target = [0.5, -0.3, 0.4];
    let mut rows = Vec::new();
    for fc in [1000, 1] {
        for method in ["gpmin", "rsirfo_bofill", "rsirfo_block_bofill"] {
            let cells: Vec<String> = starts
                .iter()
                .map(|s| {
                    let out = solve(method, 2, Some(fc), 200, obj.as_ref(), s);
                    let off = (0..3)
                        .map(|i| (out.x[i] - target[i]).abs())
                        .fold(0.0, f64::max);
                    let there = out.termination == Some(Termination::Converged) && off < 1e-2;
                    if method == "gpmin" {
                        assert!(
                            there,
                            "gpmin from {s:?}: {:?}, off by {off}",
                            out.termination
                        );
                    }
                    format!("{}{}", out.iterations, if there { "" } else { "*" })
                })
                .collect();
            rows.push(format!("Hessian every {fc}: {method} {}", cells.join(", ")));
        }
    }
    println!("saddle of order 2 (* = not there):\n{}", rows.join("\n"));
}

#[test]
fn saddle_points_with_the_ported_hessian_methods() {
    let failures: Vec<String> = [
        "rsprfo_bofill",
        "rsprfo_block_bofill",
        "rsprfo_fsb",
        "mf_rsirfo_bofill",
        "smf_rsirfo_bofill",
        "mwmf_rsirfo_bofill",
        "mwsmf_rsirfo_fsb",
        "mf_rsirfo_bofill:0:ema0.5:grad0.2",
        "crsirfo_bofill",
        "newton_bofill",
        "rsirfo_bofill_trim",
        "rsprfo_bofill_trim",
        "rsirfo_bofill_gdiis",
        "rsirfo_bofill_c2diis",
        "rsirfo_bofill_kdiis",
        "rsirfo_bofill_gediis",
    ]
    .iter()
    .flat_map(|m| check_saddle_method(m, 1))
    .collect();
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}
