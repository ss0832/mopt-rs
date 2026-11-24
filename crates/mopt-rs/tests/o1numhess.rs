//! O1NumHess: Hessians from a number of gradients that does not grow with the system, against
//! exact Hessians (a banded function, a Lennard-Jones cluster, alkanes with GFN-FF), the second
//! round along negative modes, and `-modelhess o1numhess` in optimizations.
//!
//! MultiOptPy: `ModelHessian/o1numhess.py` has no tests.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used, clippy::expect_used)]

use faer::{Col, Mat};
use mopt_rs::cli::optmain::{config_from_cli, run};
use mopt_rs::domain::{Angstrom, Element, EvalRequest, MolecularSystem, SpinCharge, Vector3};
use mopt_rs::math::linalg::symmetric_eigen;
use mopt_rs::opt::state_machine::{GeometryOptimizer, OptimizerSettings, Termination};
use mopt_rs::pes::native::{Gfnff, LennardJones};
use mopt_rs::pes::o1numhess::{
    hessian_from_gradients, wrap_if_requested, Invariance, O1Action, O1NumHess,
    O1NumHessCalculator, O1Problem, O1Settings,
};
use mopt_rs::pes::Calculator;
use mopt_rs::workflow::run_optimization;

type Row = (&'static str, [f64; 3]);

fn system(rows: &[Row]) -> MolecularSystem {
    let el: Vec<Element> = rows
        .iter()
        .map(|(s, _)| Element::parse_token(s).unwrap())
        .collect();
    let pos: Vec<_> = rows
        .iter()
        .map(|(_, p)| Vector3::<Angstrom>::new(*p))
        .collect();
    MolecularSystem::new(el, &pos, SpinCharge::default(), None).unwrap()
}

/// Deterministic displacements of amplitude `amp` (Angstrom).
fn rattle(mut rows: Vec<Row>, amp: f64) -> Vec<Row> {
    for (i, r) in rows.iter_mut().enumerate() {
        for a in 0..3 {
            r.1[a] += amp * ((7 * i + 3 * a) as f64 * 1.37).sin();
        }
    }
    rows
}

/// All-trans alkane with `n` carbons; the second value lists the atoms of every CHx group.
fn alkane(n: usize) -> (Vec<Row>, Vec<(usize, usize)>) {
    let (dx, dy) = (1.277, 0.861);
    let mut rows = Vec::new();
    let mut groups = Vec::new();
    for k in 0..n {
        let first = rows.len();
        let up = k % 2 == 1;
        let c = [dx * k as f64, if up { dy } else { 0.0 }, 0.0];
        let out = if up { 0.63 } else { -0.63 };
        let end = if up { -0.609 } else { 0.609 };
        rows.push(("C", c));
        rows.push(("H", [c[0], c[1] + out, 0.89]));
        rows.push(("H", [c[0], c[1] + out, -0.89]));
        if k == 0 {
            rows.push(("H", [c[0] - 0.904, c[1] + end, 0.0]));
        }
        if k == n - 1 {
            rows.push(("H", [c[0] + 0.904, c[1] + end, 0.0]));
        }
        groups.push((first, rows.len()));
    }
    (rows, groups)
}

fn rotate(p: [f64; 3], axis: [f64; 3], angle: f64) -> [f64; 3] {
    let n = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
    let k = [axis[0] / n, axis[1] / n, axis[2] / n];
    let (s, c) = angle.sin_cos();
    let kxp = [
        k[1] * p[2] - k[2] * p[1],
        k[2] * p[0] - k[0] * p[2],
        k[0] * p[1] - k[1] * p[0],
    ];
    let kp = k[0] * p[0] + k[1] * p[1] + k[2] * p[2];
    std::array::from_fn(|a| p[a] * c + kxp[a] * s + k[a] * kp * (1.0 - c))
}

/// The alkane with the part after carbon `k` turned by `angle` about the bond `k - (k+1)`.
fn twisted(mut rows: Vec<Row>, groups: &[(usize, usize)], k: usize, angle: f64) -> Vec<Row> {
    let (a, b) = (rows[groups[k].0].1, rows[groups[k + 1].0].1);
    let axis = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    for g in &groups[k + 1..] {
        for r in &mut rows[g.0..g.1] {
            let rel = [r.1[0] - b[0], r.1[1] - b[1], r.1[2] - b[2]];
            let q = rotate(rel, axis, angle);
            r.1 = [q[0] + b[0], q[1] + b[1], q[2] + b[2]];
        }
    }
    rows
}

fn gradient(calc: &dyn Calculator, sys: &MolecularSystem) -> Col<f64> {
    calc.evaluate(sys, &EvalRequest::ENERGY_GRAD)
        .unwrap()
        .gradient_bohr
}

/// Largest element of `a - b` and of `b`.
fn difference(a: &Mat<f64>, b: &Mat<f64>) -> (f64, f64) {
    let mut worst = 0.0_f64;
    let mut scale = 0.0_f64;
    for i in 0..a.nrows() {
        for j in 0..a.ncols() {
            worst = worst.max((a[(i, j)] - b[(i, j)]).abs());
            scale = scale.max(b[(i, j)].abs());
        }
    }
    (worst, scale)
}

/// Largest difference between the sorted eigenvalues of `a` and `b`.
fn spectrum_difference(a: &Mat<f64>, b: &Mat<f64>) -> f64 {
    let (wa, _) = symmetric_eigen(a).unwrap();
    let (wb, _) = symmetric_eigen(b).unwrap();
    (0..wa.nrows()).fold(0.0_f64, |m, i| m.max((wa[i] - wb[i]).abs()))
}

#[test]
fn a_banded_hessian_takes_as_many_gradients_as_the_band_is_wide() {
    // f = sum_i a_i x_i^2 / 2 + b (x_i - x_{i+1})^2 / 2 + c x_i x_{i+2}: a quadratic function
    // whose Hessian has two side bands.
    let n = 40;
    let exact = Mat::from_fn(n, n, |i, j| match i.abs_diff(j) {
        0 => {
            let ends = usize::from(i > 0) + usize::from(i + 1 < n);
            1.0 + 0.05 * i as f64 + 0.7 * ends as f64
        }
        1 => -0.7,
        2 => 0.2,
        _ => 0.0,
    });
    let x = Col::from_fn(n, |i| 0.3 * (i as f64 * 0.9).sin());
    let grad = |x: &Col<f64>| &exact * x;
    let settings = O1Settings {
        dmax: 2.5,
        ..O1Settings::default()
    };
    let mut job = O1NumHess::new(O1Problem::banded(x.clone(), grad(&x)), settings).unwrap();
    let mut rounds = 0;
    while let O1Action::Evaluate(points) = job.ask() {
        let gradients: Vec<Col<f64>> = points.iter().map(&grad).collect();
        job.tell(&gradients).unwrap();
        rounds += 1;
    }
    // Five neighbours each (two on either side and the variable itself): five gradients.
    assert_eq!((rounds, job.gradient_count()), (1, 5));
    assert_eq!(job.directions().ncols(), 5);
    let (worst, _) = difference(job.hessian().unwrap(), &exact);
    assert!(worst < 1e-8, "{worst}");
    assert!(job.residual() < 1e-8);
    // tell() with the wrong number of gradients, and after the end, is refused.
    assert!(job.tell(&[]).is_err());
    let fresh = O1NumHess::new(O1Problem::banded(x.clone(), grad(&x)), settings).unwrap();
    assert!(fresh.clone().tell(std::slice::from_ref(&x)).is_err());
    assert!(fresh.hessian().is_none());
}

/// An anharmonic function with couplings to the next two variables: gradient and Hessian.
fn anharmonic(x: &Col<f64>) -> (Col<f64>, Mat<f64>) {
    let n = x.nrows();
    let mut g = Col::<f64>::zeros(n);
    let mut h = Mat::<f64>::zeros(n, n);
    for i in 0..n {
        let a = 1.0 + 0.05 * i as f64;
        g[i] += a * x[i] + x[i].powi(3);
        h[(i, i)] += a + 3.0 * x[i].powi(2);
        for (step, k, c) in [(1, 0.7, 0.4), (2, -0.2, 0.3)] {
            if i + step < n {
                let j = i + step;
                let d = x[i] - x[j];
                // k d^2 / 2 + c d^3 / 3 + cos(d).
                let first = k * d + c * d * d - d.sin();
                let second = k + 2.0 * c * d - d.cos();
                g[i] += first;
                g[j] -= first;
                h[(i, i)] += second;
                h[(j, j)] += second;
                h[(i, j)] -= second;
                h[(j, i)] -= second;
            }
        }
    }
    (g, h)
}

#[test]
fn an_anharmonic_banded_function_is_as_accurate_as_finite_differences() {
    // Five gradients whatever the number of variables. With 40 variables some neighbourhoods
    // are not spanned completely by the five directions; the error stays at the level of full
    // differences all the same.
    for (n, delta) in [(60, 5e-3), (40, 1e-5), (25, 1e-5)] {
        let x = Col::from_fn(n, |i| {
            0.4 * (i as f64 * 0.83).sin() + 0.1 * (i as f64 * 0.31).cos()
        });
        let (g0, exact) = anharmonic(&x);
        let settings = O1Settings {
            delta,
            dmax: 2.5,
            follow_negative_modes: false,
            ..O1Settings::default()
        };
        let problem = O1Problem::banded(x.clone(), g0.clone());
        let mut job = O1NumHess::new(problem, settings).unwrap();
        while let O1Action::Evaluate(points) = job.ask() {
            let gradients: Vec<Col<f64>> = points.iter().map(|p| anharmonic(p).0).collect();
            job.tell(&gradients).unwrap();
        }
        assert_eq!(job.gradient_count(), 5);
        // Full single-sided differences with the same step: n gradients.
        let mut full = Mat::<f64>::zeros(n, n);
        for j in 0..n {
            let mut xp = x.clone();
            xp[j] += delta;
            let gp = anharmonic(&xp).0;
            for i in 0..n {
                full[(i, j)] = (gp[i] - g0[i]) / delta;
            }
        }
        let (o1, _) = difference(job.hessian().unwrap(), &exact);
        let (fd, _) = difference(&full, &exact);
        assert!(o1 < 2.5 * fd, "n = {n}: {o1} against {fd}");
        // Points on both sides: ten gradients and a much smaller error.
        let central = O1Settings {
            central: true,
            ..settings
        };
        let problem = O1Problem::banded(x.clone(), g0.clone());
        let mut job = O1NumHess::new(problem, central).unwrap();
        while let O1Action::Evaluate(points) = job.ask() {
            let gradients: Vec<Col<f64>> = points.iter().map(|p| anharmonic(p).0).collect();
            job.tell(&gradients).unwrap();
        }
        assert_eq!(job.gradient_count(), 10);
        let (both, _) = difference(job.hessian().unwrap(), &exact);
        assert!(both < 0.05 * fd.max(1e-6), "n = {n}: {both} against {fd}");
    }
}

#[test]
fn a_small_molecule_gets_the_full_finite_difference_hessian() {
    // All atoms of ethane are neighbours: the directions are a complete basis, and six of
    // them (translations and rotations) cost nothing.
    let (rows, _) = alkane(2);
    let sys = system(&rattle(rows, 0.03));
    let calc = Gfnff::starting_from(&sys);
    let reference = calc.hessian(&sys).unwrap();
    let out = hessian_from_gradients(
        &sys,
        &gradient(&calc, &sys),
        Invariance::of(&calc, &sys),
        O1Settings {
            follow_negative_modes: false,
            ..O1Settings::default()
        },
        |s| Ok(gradient(&calc, s)),
    )
    .unwrap();
    // 24 coordinates - 6 known + 1 (the breathing mode is taken on both sides).
    assert_eq!(out.gradients, 19);
    let (worst, scale) = difference(&out.hessian, &reference);
    // Single-sided differences with 0.005 Bohr: errors of a few thousandths.
    assert!(worst < 0.01 * scale, "{worst} of {scale}");
}

#[test]
fn an_alkane_takes_far_fewer_gradients_than_coordinates() {
    let (rows, _) = alkane(14);
    let sys = system(&rattle(rows, 0.03));
    let calc = Gfnff::starting_from(&sys);
    let reference = calc.hessian(&sys).unwrap();
    let wrapped = O1NumHessCalculator::new(Box::new(calc), O1Settings::default());
    let out = wrapped
        .evaluate(&sys, &EvalRequest::ENERGY_GRAD_HESS)
        .unwrap();
    let hessian = out.hessian_bohr.unwrap();
    let notes = wrapped.take_warnings();
    assert!(
        notes[0].starts_with("O1NumHess: Hessian of 132 coordinates from "),
        "{notes:?}"
    );
    let gradients: usize = notes[0].split_whitespace().nth(6).unwrap().parse().unwrap();
    assert!(gradients < 132 / 2, "{gradients}");
    let (worst, scale) = difference(&hessian, &reference);
    assert!(worst < 0.01 * scale, "{worst} of {scale}");
    assert!(spectrum_difference(&hessian, &reference) < 0.02 * scale);
    // Requests without a Hessian go straight to the calculator.
    let plain = wrapped.evaluate(&sys, &EvalRequest::ENERGY_GRAD).unwrap();
    assert!(plain.hessian_bohr.is_none());
    assert_eq!(plain.energy, out.energy);
    assert!(wrapped.take_warnings().is_empty());
    assert!(wrapped.description().ends_with("Hessians by O1NumHess"));
}

#[test]
fn a_lennard_jones_cluster_against_the_analytic_hessian() {
    // 3 x 2 x 2 face-centred cubic cells of argon, slightly displaced: 48 atoms.
    let a = 5.26;
    let base = [
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
    ];
    let mut rows: Vec<Row> = Vec::new();
    for i in 0..3 {
        for j in 0..2 {
            for k in 0..2 {
                for b in base {
                    rows.push((
                        "Ar",
                        [
                            a * (i as f64 + b[0]),
                            a * (j as f64 + b[1]),
                            a * (k as f64 + b[2]),
                        ],
                    ));
                }
            }
        }
    }
    let sys = system(&rattle(rows, 0.05));
    let exact = LennardJones
        .evaluate(&sys, &EvalRequest::ENERGY_GRAD_HESS)
        .unwrap();
    let reference = exact.hessian_bohr.unwrap();
    let out = hessian_from_gradients(
        &sys,
        &exact.gradient_bohr,
        Invariance::of(&LennardJones, &sys),
        O1Settings::default(),
        |s| Ok(gradient(&LennardJones, s)),
    )
    .unwrap();
    assert!(out.gradients < sys.dof() / 3, "{}", out.gradients);
    let (worst, scale) = difference(&out.hessian, &reference);
    assert!(worst < 0.02 * scale, "{worst} of {scale}");
    assert!(spectrum_difference(&out.hessian, &reference) < 0.02 * scale);
}

#[test]
fn negative_modes_are_differentiated_in_a_second_round() {
    // Decane with the central bond turned to the eclipsed form: the torsion has negative
    // curvature there.
    let (rows, groups) = alkane(10);
    let sys = system(&twisted(rows, &groups, 4, 60.0_f64.to_radians()));
    let calc = Gfnff::starting_from(&sys);
    let reference = calc.hessian(&sys).unwrap();
    let (exact_values, _) = symmetric_eigen(&reference).unwrap();
    assert!(exact_values[0] < -1e-4, "{}", exact_values[0]);
    let g0 = gradient(&calc, &sys);
    let run = |follow: bool| {
        hessian_from_gradients(
            &sys,
            &g0,
            Invariance::of(&calc, &sys),
            O1Settings {
                follow_negative_modes: follow,
                ..O1Settings::default()
            },
            |s| Ok(gradient(&calc, s)),
        )
        .unwrap()
    };
    let (once, twice) = (run(false), run(true));
    assert!(twice.gradients > once.gradients);
    let (values, _) = symmetric_eigen(&twice.hessian).unwrap();
    let negative = |w: &Col<f64>| (0..w.nrows()).filter(|&i| w[i] < -1e-4).count();
    assert_eq!(negative(&values), negative(&exact_values));
    // The curvature of the negative mode is measured, not fitted.
    assert!(
        (values[0] - exact_values[0]).abs() < 0.05 * exact_values[0].abs(),
        "{} against {}",
        values[0],
        exact_values[0]
    );
}

#[test]
fn optimizations_ask_for_o1numhess_every_mfc_iterations() {
    let (rows, _) = alkane(8);
    let start = system(&rattle(rows, 0.06));
    let run = |model: Option<&str>| {
        let settings = OptimizerSettings {
            methods: vec!["rsirfo_bfgs".to_string()],
            model_hessian: model.map(str::to_string),
            model_hessian_interval: 10,
            ..OptimizerSettings::default()
        };
        let calc: Box<dyn Calculator> = Box::new(Gfnff::starting_from(&start));
        let calc = wrap_if_requested(calc, model, false);
        let mut opt = GeometryOptimizer::new(settings, start.clone()).unwrap();
        let out = run_optimization(&mut opt, calc.as_ref(), &mut []).unwrap();
        assert_eq!(out.termination, Some(Termination::Converged));
        (opt.iterations(), calc.take_warnings().len())
    };
    let (plain, _) = run(None);
    let (with_hessian, _) = run(Some("o1numhess"));
    assert!(
        with_hessian < plain,
        "{with_hessian} iterations with O1NumHess, {plain} without"
    );
    // -fc takes precedence: the calculator is left as it is.
    let calc: Box<dyn Calculator> = Box::new(LennardJones);
    let kept = wrap_if_requested(calc, Some("o1numhess"), true);
    assert_eq!(kept.description(), LennardJones.description());
}

#[test]
fn optmain_takes_modelhess_o1numhess() {
    let dir = std::env::temp_dir().join(format!("mopt_o1numhess_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let (rows, _) = alkane(6);
    let mut xyz = format!("{}\nhexane\n", rows.len());
    for (s, p) in rattle(rows, 0.05) {
        xyz.push_str(&format!("{s} {:.6} {:.6} {:.6}\n", p[0], p[1], p[2]));
    }
    let input = dir.join("hexane.xyz");
    std::fs::write(&input, xyz).unwrap();
    let input = input.to_string_lossy().into_owned();
    let args = [
        input.as_str(),
        "-os",
        "gfnff",
        "-opt",
        "rsirfo_bfgs",
        "-modelhess",
        "o1numhess",
        "-mfc",
        "8",
    ];
    let res = run(&config_from_cli(&args).unwrap(), &dir, None).unwrap();
    assert!(res[0].converged);
    let log = std::fs::read_to_string(res[0].directory.join("mopt.log")).unwrap();
    assert!(log.contains("Hessians by O1NumHess"), "{log}");
    assert!(
        log.contains("O1NumHess: Hessian of 60 coordinates from "),
        "{log}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}
