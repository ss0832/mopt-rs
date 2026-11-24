//! Tests of the IRC state machine on the Muller-Brown surface of two variables.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used)]

use faer::Col;

use super::*;
use crate::pes::native::muller_brown::{MB_MINIMA, MB_SADDLES};
use crate::pes::objective::{MullerBrown2D, Objective};

fn value_at(x: &Col<f64>, hessian: bool) -> PointValue {
    let v = MullerBrown2D::default().evaluate(x, hessian).unwrap();
    PointValue::plain(v.value, v.gradient, v.hessian)
}

struct Run {
    end: IrcEnd,
    events: Vec<IrcEvent>,
    evaluations: usize,
}

fn run(settings: IrcSettings, start: [f64; 2]) -> Run {
    let x = Col::from_fn(2, |i| start[i]);
    let mut irc = Irc::new(settings, IrcSpace::variables(2), x).unwrap();
    let mut events = Vec::new();
    loop {
        let action = irc.ask().unwrap();
        events.extend(irc.drain_events());
        match action {
            IrcAction::Finished(end) => {
                return Run {
                    end,
                    events,
                    evaluations: irc.evaluations(),
                }
            }
            IrcAction::Evaluate(request) => {
                irc.tell(value_at(&request.point.x, request.point.need_hessian))
                    .unwrap();
                events.extend(irc.drain_events());
            }
        }
    }
}

fn nearest_minimum(x: &Col<f64>) -> (usize, f64) {
    let mut best = (0, f64::INFINITY);
    for (k, m) in MB_MINIMA.iter().enumerate() {
        let d = ((x[0] - m[0]).powi(2) + (x[1] - m[1]).powi(2)).sqrt();
        if d < best.1 {
            best = (k, d);
        }
    }
    best
}

fn settings(method: &str, step: f64) -> IrcSettings {
    IrcSettings {
        step_size: step,
        max_steps: 600,
        method: method.to_string(),
        ..IrcSettings::default()
    }
}

/// The minima each saddle of `MB_SADDLES` connects (numbers in `MB_MINIMA`).
const CONNECTED: [[usize; 2]; 2] = [[0, 2], [1, 2]];

#[test]
fn every_integrator_reaches_both_minima_from_both_saddles() {
    // (method, step size, distance from the minimum at the end)
    let cases = [
        ("lqa", 0.05, 2e-3),
        ("hpc", 0.05, 5e-3),
        ("euler", 0.01, 0.03),
        ("rk4", 0.03, 0.06),
        ("gs2", 0.05, 0.08),
        ("dvv", 0.05, 0.15),
    ];
    for (method, step, tolerance) in cases {
        for (saddle, connected) in MB_SADDLES.iter().zip(CONNECTED) {
            let result = run(settings(method, step), *saddle);
            assert_eq!(result.end.kind, StartKind::TransitionState, "{method}");
            assert_eq!(result.end.branches.len(), 2);
            let mut reached: Vec<usize> = Vec::new();
            for side in &result.end.branches {
                let (minimum, distance) = nearest_minimum(&side.x);
                assert!(
                    distance < tolerance,
                    "{method} {saddle:?} {:?}: {distance} from minimum {minimum} after {} points ({:?})",
                    side.branch,
                    side.points,
                    side.end
                );
                // RK4 and DVV neither arrive nor oscillate at a minimum (as in MultiOptPy):
                // the four gradients of an RK4 step cancel there, the velocity of DVV is
                // rescaled to the same magnitude after every step, and both go on around
                // the minimum until the steps are used up.
                if method != "rk4" && method != "dvv" {
                    assert_ne!(side.end, BranchEnd::MaxSteps, "{method} {saddle:?}");
                }
                reached.push(minimum);
            }
            reached.sort_unstable();
            assert_eq!(reached, connected.to_vec(), "{method} {saddle:?}");
        }
    }
}

#[test]
fn the_energy_falls_along_an_lqa_path_and_the_records_are_numbered() {
    let result = run(settings("lqa", 0.05), MB_SADDLES[0]);
    let mut last: Option<(Branch, usize, f64)> = None;
    let mut starts = 0;
    for event in &result.events {
        match event {
            IrcEvent::Start(report) => {
                starts += 1;
                assert_eq!(report.imaginary, 1);
                assert!(report.transition_eigenvalue.unwrap() < 0.0);
                assert!(report.stationary);
            }
            IrcEvent::Step { branch, record } => {
                if let Some((b, step, energy)) = last {
                    if b == *branch {
                        assert_eq!(record.step, step + 1);
                        assert!(record.biased_energy < energy + 1e-12);
                    } else {
                        assert_eq!(record.step, 1);
                    }
                }
                last = Some((*branch, record.step, record.biased_energy));
            }
            IrcEvent::BranchFinished(summary) => assert_eq!(summary.end, BranchEnd::Converged),
            // The first point of each side is tested by default; here it is within the
            // limits and stays.
            IrcEvent::FirstStep(report) => assert!(!report.corrected),
        }
    }
    assert_eq!(starts, 1);
}

#[test]
fn a_structure_off_the_saddle_gives_a_meta_irc_to_the_minimum_below() {
    let result = run(settings("lqa", 0.05), [-0.7, 0.9]);
    assert_eq!(result.end.kind, StartKind::MetaIrc);
    assert_eq!(result.end.branches.len(), 1);
    let side = &result.end.branches[0];
    assert_eq!(side.branch, Branch::Single);
    let (minimum, distance) = nearest_minimum(&side.x);
    assert_eq!(minimum, 0);
    assert!(distance < 2e-3, "{distance}");
}

#[test]
fn one_side_only_and_the_step_limit() {
    let mut s = settings("lqa", 0.05);
    s.sides = Sides::Backward;
    s.max_steps = 5;
    let result = run(s, MB_SADDLES[1]);
    assert_eq!(result.end.branches.len(), 1);
    let side = &result.end.branches[0];
    assert_eq!(side.branch, Branch::Backward);
    // MultiOptPy's loop `range(1, max_step)`: four points.
    assert_eq!((side.end, side.points), (BranchEnd::MaxSteps, 4));
    // The starting structure and four points.
    assert_eq!(result.evaluations, 5);
    let mut s = settings("euler", 0.01);
    s.sides = Sides::Forward;
    s.max_steps = 5;
    let side = &run(s, MB_SADDLES[1]).end.branches[0];
    // `range(1, max_step + 1)`: five points.
    assert_eq!((side.end, side.points), (BranchEnd::MaxSteps, 5));
}

#[test]
fn exact_hessians_at_an_interval_are_asked_for() {
    let mut s = settings("lqa", 0.05);
    s.exact_hessian_interval = Some(3);
    s.sides = Sides::Forward;
    let x = Col::from_fn(2, |i| MB_SADDLES[0][i]);
    let mut irc = Irc::new(s, IrcSpace::variables(2), x).unwrap();
    let mut with_hessian = Vec::new();
    loop {
        match irc.ask().unwrap() {
            IrcAction::Finished(_) => break,
            IrcAction::Evaluate(request) => {
                if request.point.need_hessian {
                    with_hessian.push(request.point.step);
                }
                irc.tell(value_at(&request.point.x, request.point.need_hessian))
                    .unwrap();
            }
        }
    }
    assert_eq!(&with_hessian[..4], &[0, 3, 6, 9]);
    // Without an interval (MultiOptPy's `-fc -1`) the Hessian of the start is the only
    // exact one; MultiOptPy's `iter % -1 == 0` took the one of the start at every step.
    let x = Col::from_fn(2, |i| MB_SADDLES[0][i]);
    let mut irc = Irc::new(settings("lqa", 0.05), IrcSpace::variables(2), x).unwrap();
    let mut asked = 0;
    while let IrcAction::Evaluate(request) = irc.ask().unwrap() {
        asked += usize::from(request.point.need_hessian);
        irc.tell(value_at(&request.point.x, request.point.need_hessian))
            .unwrap();
    }
    assert_eq!(asked, 1);
    // A value without the Hessian that was asked for is refused at the start.
    let x = Col::from_fn(2, |i| MB_SADDLES[0][i]);
    let mut irc = Irc::new(settings("lqa", 0.05), IrcSpace::variables(2), x.clone()).unwrap();
    let _ = irc.ask().unwrap();
    assert!(irc.tell(value_at(&x, false)).is_err());
    // Unknown methods and bad settings are refused.
    assert!(Irc::new(settings("lqaa", 0.05), IrcSpace::variables(2), x.clone()).is_err());
    assert!(Irc::new(settings("lqa", -1.0), IrcSpace::variables(2), x).is_err());
}

#[test]
fn dvv_evaluates_every_point_once_and_euler_goes_past_its_second_point() {
    for method in ["dvv", "euler"] {
        let mut s = settings(method, 0.01);
        s.max_steps = 30;
        s.sides = Sides::Forward;
        let result = run(s, MB_SADDLES[0]);
        let side = &result.end.branches[0];
        assert!(side.points >= 29, "{method}: {}", side.points);
        // The start and one evaluation for every point (MultiOptPy's DVV evaluates its
        // first point twice; its Euler stops with an error at the second).
        assert_eq!(result.evaluations, side.points + 1, "{method}");
        let mut last: Option<Col<f64>> = None;
        for event in &result.events {
            if let IrcEvent::Step { record, .. } = event {
                if let Some(previous) = &last {
                    assert!(previous != &record.x, "{method}: a point twice");
                }
                last = Some(record.x.clone());
            }
        }
    }
}

#[test]
fn the_curvature_of_a_circular_valley_is_the_inverse_of_its_radius() {
    // E = k (r - R)^2 / 2 + c * angle: a valley along the circle of radius R = 2 with a
    // slope along it. Euler in small steps follows the circle; the curvature it reports
    // is 1 / R, with its component along the mode across the valley.
    let (k, radius, c) = (2.0, 2.0, 0.05);
    let at = |x: &Col<f64>, hessian: bool| {
        let r = (x[0] * x[0] + x[1] * x[1]).sqrt();
        let angle = x[1].atan2(x[0]);
        let radial = [x[0] / r, x[1] / r];
        let around = [-radial[1], radial[0]];
        let g = Col::from_fn(2, |i| k * (r - radius) * radial[i] + c / r * around[i]);
        // Close to the floor of the valley the Hessian is k along the radius.
        let h = hessian.then(|| faer::Mat::from_fn(2, 2, |i, j| k * radial[i] * radial[j]));
        PointValue::plain(0.5 * k * (r - radius).powi(2) + c * angle, g, h)
    };
    let mut s = settings("euler", 0.01);
    s.max_steps = 60;
    s.exact_hessian_interval = Some(1);
    let x = Col::from_fn(2, |i| [radius, 0.0][i]);
    let mut irc = Irc::new(s, IrcSpace::variables(2), x).unwrap();
    let mut curvatures = Vec::new();
    while let IrcAction::Evaluate(request) = irc.ask().unwrap() {
        irc.tell(at(&request.point.x, request.point.need_hessian))
            .unwrap();
        for event in irc.drain_events() {
            if let IrcEvent::Step { record, .. } = event {
                curvatures.extend(record.curvature);
            }
        }
    }
    assert_eq!(irc.kind(), Some(StartKind::MetaIrc));
    assert!(curvatures.len() > 40);
    for curvature in &curvatures[20..] {
        assert!(
            (curvature.scalar - 0.5).abs() < 0.05,
            "{}",
            curvature.scalar
        );
        assert_eq!(curvature.couplings.len(), 1);
        assert!((curvature.couplings[0].abs() - 0.5).abs() < 0.05);
    }
}

#[test]
fn modekill_leaves_a_maximum_along_its_second_mode() {
    // E = -0.5 a x^2 - 0.5 b y^2 + x^4 + y^4: a maximum at the origin with two imaginary
    // modes; killing the second one leaves a point with the first one only.
    struct Hill;
    impl Hill {
        fn at(x: &Col<f64>, hessian: bool) -> PointValue {
            let (a, b) = (0.2, 0.1);
            let (u, v) = (x[0], x[1]);
            let e = -0.5 * a * u * u - 0.5 * b * v * v + u.powi(4) + v.powi(4);
            let g = Col::from_fn(2, |i| {
                [-a * u + 4.0 * u.powi(3), -b * v + 4.0 * v.powi(3)][i]
            });
            let h = hessian.then(|| {
                faer::Mat::from_fn(2, 2, |i, j| {
                    if i == j {
                        [-a + 12.0 * u * u, -b + 12.0 * v * v][i]
                    } else {
                        0.0
                    }
                })
            });
            PointValue::plain(e, g, h)
        }
    }
    let mut s = settings("modekill", 0.02);
    s.max_steps = 200;
    // Masses that give wavenumbers far below the threshold of -5 cm^-1.
    let space = IrcSpace::with_masses(&[1.0, 1.0]).unwrap();
    let mut irc = Irc::new(s, space, Col::from_fn(2, |i| [0.0, 1e-3][i])).unwrap();
    let mut rows = 0;
    let end = loop {
        match irc.ask().unwrap() {
            IrcAction::Finished(end) => break end,
            IrcAction::Evaluate(request) => {
                irc.tell(Hill::at(&request.point.x, request.point.need_hessian))
                    .unwrap();
                for event in irc.drain_events() {
                    if let IrcEvent::Step { record, .. } = event {
                        rows += 1;
                        assert_eq!(record.extras[0].0, "Imaginary Frequencies");
                    }
                }
            }
        }
    };
    assert_eq!(end.kind, StartKind::ModeKill);
    let side = &end.branches[0];
    assert_eq!(side.end, BranchEnd::ModesRemoved, "{rows} rows");
    // The second mode (y) turned positive: beyond y = sqrt(b / 12); x stayed on the ridge.
    assert!(side.x[1].abs() > (0.1_f64 / 12.0).sqrt(), "{:?}", side.x);
    assert!(side.x[0].abs() < 1e-8);
}
