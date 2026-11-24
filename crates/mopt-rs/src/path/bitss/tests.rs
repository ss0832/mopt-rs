//! Tests of the BITSS search.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::{Col, Mat};

use super::*;
use crate::opt::vector::VectorSettings;
use crate::pes::objective::{MullerBrown2D, Objective};

/// A surface of three variables with two wells for the derivative checks.
fn surface(x: &Col<f64>) -> (f64, Col<f64>, Mat<f64>) {
    let (a, b, c) = (x[0], x[1], x[2]);
    let e = (a * a - 1.0).powi(2) + 0.7 * b * b + 0.3 * c * c + 0.2 * a * b + 0.1 * b * c * a;
    let g = Col::from_fn(3, |k| match k {
        0 => 4.0 * a * (a * a - 1.0) + 0.2 * b + 0.1 * b * c,
        1 => 1.4 * b + 0.2 * a + 0.1 * c * a,
        _ => 0.6 * c + 0.1 * b * a,
    });
    let h = Mat::from_fn(3, 3, |i, j| match (i.min(j), i.max(j)) {
        (0, 0) => 12.0 * a * a - 4.0,
        (0, 1) => 0.2 + 0.1 * c,
        (0, 2) => 0.1 * b,
        (1, 1) => 1.4,
        (1, 2) => 0.1 * a,
        _ => 0.6,
    });
    (e, g, h)
}

fn function_at(c: &Coefficients, target: f64, x: &Col<f64>) -> (f64, Col<f64>, Mat<f64>) {
    let first = Col::from_fn(3, |k| x[k]);
    let second = Col::from_fn(3, |k| x[3 + k]);
    let (e1, g1, h1) = surface(&first);
    let (e2, g2, h2) = surface(&second);
    let (f, g) = penalized(c, target, [&first, &second], [e1, e2], [&g1, &g2]).unwrap();
    let h = penalized_hessian(
        c,
        target,
        [&first, &second],
        [e1, e2],
        [&g1, &g2],
        [&h1, &h2],
    )
    .unwrap();
    (f, g, h)
}

#[test]
fn the_coefficients_are_those_of_the_paper() {
    let c = Coefficients::new(10.0, 0.1, 2.0, 0.04, [0.03, 0.04]);
    assert!((c.kappa_e - 10.0 / 0.08).abs() < 1e-12);
    // By the gradients: 0.05 / (2 sqrt(2) 0.1 2) = 0.0884; by the barrier: 0.04 / 0.4.
    assert!((c.kappa_d - 0.1).abs() < 1e-12);
    let steep = Coefficients::new(10.0, 0.1, 2.0, 0.04, [0.3, 0.4]);
    assert!((steep.kappa_d - 0.5 / (2.0 * 2.0_f64.sqrt() * 0.2)).abs() < 1e-12);
    // No barrier on the line: the floor keeps the energy coefficient finite.
    let flat = Coefficients::new(10.0, 0.1, 2.0, -0.2, [0.0, 0.0]);
    assert!(flat.kappa_e.is_finite() && flat.kappa_d > 0.0);
}

#[test]
fn gradient_and_hessian_agree_with_differences() {
    let c = Coefficients {
        barrier: 0.5,
        kappa_e: 1.3,
        kappa_d: 0.8,
    };
    let target = 1.1;
    let x = Col::from_fn(6, |k| [-0.9, 0.2, -0.1, 0.8, -0.3, 0.25][k]);
    let (_, g, h) = function_at(&c, target, &x);
    let step = 1e-5;
    for k in 0..6 {
        let moved = |s: f64| Col::from_fn(6, |i| if i == k { x[i] + s } else { x[i] });
        let (fp, gp, _) = function_at(&c, target, &moved(step));
        let (fm, gm, _) = function_at(&c, target, &moved(-step));
        assert!(
            ((fp - fm) / (2.0 * step) - g[k]).abs() < 1e-8,
            "gradient {k}"
        );
        for i in 0..6 {
            let numeric = (gp[i] - gm[i]) / (2.0 * step);
            assert!((numeric - h[(i, k)]).abs() < 1e-7, "Hessian {i} {k}");
        }
    }
}

fn value_of(surface: &MullerBrown2D, point: &BitssPoint) -> PointValue {
    let v = surface.evaluate(&point.x, point.hessian).unwrap();
    PointValue {
        energy: v.value,
        gradient: point.gradient.then_some(v.gradient),
        hessian: v.hessian,
    }
}

/// Run a search on the Muller-Brown surface; the end, the evaluations of gradients and
/// of energies alone, and the search itself.
fn search(
    settings: BitssSettings,
    first: [f64; 2],
    second: [f64; 2],
) -> (BitssEnd, usize, usize, Bitss) {
    let surface = MullerBrown2D::default();
    let a = Col::from_fn(2, |k| first[k]);
    let b = Col::from_fn(2, |k| second[k]);
    let mut bitss = Bitss::new(settings, &a, &b).unwrap();
    let (mut gradients, mut energies) = (0, 0);
    loop {
        match bitss.ask() {
            BitssAction::Finished(end) => return (end, gradients, energies, bitss),
            BitssAction::Evaluate(points) => {
                gradients += points.iter().filter(|p| p.gradient).count();
                energies += points.iter().filter(|p| !p.gradient).count();
                let values = points.iter().map(|p| value_of(&surface, p)).collect();
                bitss.tell(values).unwrap();
            }
        }
    }
}

fn settings_with(method: &str) -> BitssSettings {
    BitssSettings {
        minimizer: VectorSettings {
            methods: vec![method.to_string()],
            max_iterations: 3000,
            ..VectorSettings::default()
        },
        ..BitssSettings::default()
    }
}

// Stationary points of the Muller-Brown surface.
const MIN_A: [f64; 2] = [-0.558_223_6, 1.441_725_8];
const MIN_B: [f64; 2] = [0.623_499_4, 0.028_037_8];
const MIN_C: [f64; 2] = [-0.050_010_8, 0.466_694_1];
const SADDLE_AC: [f64; 2] = [-0.822_001_6, 0.624_312_8];
const SADDLE_BC: [f64; 2] = [0.212_486_6, 0.292_988_3];

fn distance(x: &Col<f64>, p: [f64; 2]) -> f64 {
    (x[0] - p[0]).hypot(x[1] - p[1])
}

#[test]
fn the_images_meet_at_the_saddle_between_two_neighbouring_minima() {
    for method in ["lbfgs", "rsirfo_bfgs", "fire"] {
        let (end, .., bitss) = search(settings_with(method), MIN_B, MIN_C);
        assert!(end.converged(), "{method}: {end:?}");
        let ts = bitss.transition_state_estimate();
        assert!(
            distance(&ts, SADDLE_BC) < 5e-3,
            "{method}: midpoint {:?}",
            (ts[0], ts[1])
        );
    }
}

#[test]
fn across_an_intermediate_the_higher_saddle_is_found() {
    // From the deepest minimum to the one two basins away: the paper's claim is the
    // highest transition state on the way.
    let (end, .., bitss) = search(settings_with("lbfgs"), MIN_A, MIN_B);
    assert!(end.converged(), "{end:?}");
    let ts = bitss.transition_state_estimate();
    assert!(
        distance(&ts, SADDLE_AC) < 5e-3,
        "midpoint {:?}",
        (ts[0], ts[1])
    );
}

#[test]
fn a_pair_in_one_basin_is_not_taken_for_a_transition_state() {
    // Both images in the basin of the minimum C: they meet at the minimum.
    let (end, .., bitss) = search(settings_with("lbfgs"), [-0.15, 0.55], [0.05, 0.40]);
    assert_eq!(end, BitssEnd::NoBarrier);
    assert!(!end.converged());
    assert!(distance(&bitss.transition_state_estimate(), MIN_C) < 0.05);
}

#[test]
fn exact_hessians_are_asked_for_and_used() {
    let mut settings = settings_with("rsirfo_bfgs");
    settings.minimizer.exact_hessian_interval = Some(5);
    let (end, .., bitss) = search(settings, MIN_B, MIN_C);
    assert!(end.converged(), "{end:?}");
    assert!(distance(&bitss.transition_state_estimate(), SADDLE_BC) < 5e-3);
}

#[test]
fn a_frozen_variable_stays_and_the_separation_shrinks_by_stages() {
    let surface = MullerBrown2D::default();
    let a = Col::from_fn(2, |k| [0.6, 0.1][k]);
    let b = Col::from_fn(2, |k| [0.0, 0.1][k]);
    let mut bitss = Bitss::new(settings_with("lbfgs"), &a, &b)
        .unwrap()
        .with_frozen(&[1])
        .unwrap();
    let mut targets = Vec::new();
    for _ in 0..400 {
        let BitssAction::Evaluate(points) = bitss.ask() else {
            break;
        };
        let values = points.iter().map(|p| value_of(&surface, p)).collect();
        let report = bitss.tell(values).unwrap();
        if report.new_stage && report.end.is_none() {
            targets.push(report.target_separation);
        }
        assert!((report.images[0][1] - 0.1).abs() < 1e-14);
        assert!((report.images[1][1] - 0.1).abs() < 1e-14);
    }
    assert!(targets.len() >= 2, "{targets:?}");
    assert!((targets[0] - 0.3).abs() < 1e-12);
    for pair in targets.windows(2) {
        assert!((pair[1] - 0.5 * pair[0]).abs() < 1e-12);
    }
}

#[test]
fn wrong_input_is_refused() {
    let a = Col::from_fn(2, |k| [0.0, 0.0][k]);
    let b = Col::from_fn(2, |k| [1.0, 0.0][k]);
    assert!(Bitss::new(BitssSettings::default(), &a, &a).is_err());
    let bad = BitssSettings {
        reduction: 1.0,
        ..BitssSettings::default()
    };
    assert!(Bitss::new(bad, &a, &b).is_err());
    let mut saddle = BitssSettings::default();
    saddle.minimizer.saddle_order = 1;
    assert!(Bitss::new(saddle, &a, &b).is_err());
    // The images differ in a variable that is to stay.
    assert!(Bitss::new(BitssSettings::default(), &a, &b)
        .unwrap()
        .with_frozen(&[0])
        .is_err());
    let mut bitss = Bitss::new(BitssSettings::default(), &a, &b).unwrap();
    assert!(bitss.tell(Vec::new()).is_err());
}

/// Counts of evaluations for the documentation: `MOPT_FULL_MEASUREMENTS=1 cargo test -p
/// mopt-rs --lib bitss_measurement -- --nocapture` prints the whole table. Every run
/// checks what the table showed about the limit of the step for the quasi-Newton methods.
#[test]
fn bitss_measurement() {
    let full = std::env::var_os("MOPT_FULL_MEASUREMENTS").is_some();
    let cases = [
        ("B-C", MIN_B, MIN_C, SADDLE_BC),
        ("A-C", MIN_A, MIN_C, SADDLE_AC),
        ("A-B", MIN_A, MIN_B, SADDLE_AC),
    ];
    let methods: &[&str] = if full {
        &["lbfgs", "rsirfo_bfgs", "rsirfo_fsb", "fire", "adabelief"]
    } else {
        &["lbfgs", "rsirfo_bfgs", "rsirfo_fsb"]
    };
    // Prints the rows of one setting and returns how many searches found their saddle.
    let run = |label: &str, change: &dyn Fn(&mut BitssSettings)| -> usize {
        let mut found = 0;
        for method in methods {
            let mut line = format!("{label:24} {method:12}");
            for (_, first, second, saddle) in cases {
                let mut settings = settings_with(method);
                change(&mut settings);
                let (end, gradients, energies, bitss) = search(settings, first, second);
                let error = distance(&bitss.transition_state_estimate(), saddle);
                let mark = if !end.converged() {
                    "limit"
                } else if error < 5e-3 {
                    found += 1;
                    "ok"
                } else {
                    "OTHER"
                };
                line.push_str(&format!(
                    " | {mark:5} st {:2} it {:4} g {:4} e {:3}",
                    bitss.stage(),
                    bitss.iterations(),
                    gradients,
                    energies
                ));
            }
            println!("{line}");
        }
        found
    };
    println!("columns: B-C | A-C | A-B");
    // Without a limit of the step some quasi-Newton searches end at another saddle or
    // do not end; with the step limited to half the separation all of them find theirs.
    let unlimited = run("step share None", &|s| s.step_share = None);
    let limited = run("step share Some(0.5)", &|s| s.step_share = Some(0.5));
    assert!(unlimited < limited, "{unlimited} {limited}");
    assert_eq!(limited, if full { 12 } else { 9 });
    if !full {
        return;
    }
    for share in [Some(1.0), Some(0.25), Some(0.1)] {
        run(&format!("step share {share:?}"), &|s| s.step_share = share);
    }
    for points in [1, 3, 5] {
        run(&format!("points {points}"), &|s| {
            s.interpolation_points = points
        });
    }
    for beta in [0.1, 0.02] {
        run(&format!("beta {beta}"), &|s| s.beta = beta);
    }
    for reduction in [0.25, 0.5, 0.75] {
        run(&format!("reduction {reduction}"), &|s| {
            s.reduction = reduction
        });
    }
    for interval in [20, 100] {
        run(&format!("interval {interval}"), &|s| {
            s.coefficient_interval = interval
        });
    }
}
