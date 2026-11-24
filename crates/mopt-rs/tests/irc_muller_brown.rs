//! The IRC on the Muller-Brown surface (two plain variables, unit masses): how close every
//! integrator stays to the path, and the test and correction of the first step.
//!
//! The reference path is integrated with RK4 in steps of 0.002. Two tests print
//! the tables of `docs/source/irc.md` (`cargo test -p mopt-rs --test irc_muller_brown --
//! --nocapture`).
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used, clippy::expect_used)]

use faer::Col;

use mopt_rs::irc::{
    Branch, BranchEnd, FirstStepReport, Irc, IrcEnd, IrcEvent, IrcSettings, IrcSpace, Sides,
};
use mopt_rs::pes::native::muller_brown::{MB_MINIMA, MB_SADDLES};
use mopt_rs::pes::objective::MullerBrown2D;
use mopt_rs::workflow::irc::{run_irc, FnObserver, ObjectiveSurface};

type Point = [f64; 2];

struct Path {
    end: IrcEnd,
    points: Vec<(Branch, Point)>,
    first_steps: Vec<FirstStepReport>,
    evaluations: usize,
}

impl Path {
    fn side(&self, branch: Branch) -> Vec<Point> {
        self.points
            .iter()
            .filter(|(b, _)| *b == branch)
            .map(|(_, p)| *p)
            .collect()
    }
}

fn follow(settings: IrcSettings, start: Point) -> Path {
    let objective = MullerBrown2D::default();
    let surface = ObjectiveSurface(&objective);
    let x = Col::from_fn(2, |i| start[i]);
    let mut irc = Irc::new(settings, IrcSpace::variables(2), x).unwrap();
    let mut points = Vec::new();
    let mut first_steps = Vec::new();
    let mut observer = FnObserver::new(|event: &IrcEvent| {
        match event {
            IrcEvent::Step { branch, record } => {
                points.push((*branch, [record.x[0], record.x[1]]));
            }
            IrcEvent::FirstStep(report) => first_steps.push(report.clone()),
            _ => {}
        }
        Ok(true)
    });
    let end = run_irc(&mut irc, &surface, &mut [&mut observer])
        .unwrap()
        .unwrap();
    Path {
        end,
        points,
        first_steps,
        evaluations: irc.evaluations(),
    }
}

fn settings(method: &str, step: f64) -> IrcSettings {
    IrcSettings {
        step_size: step,
        max_steps: 600,
        method: method.to_string(),
        // DVV goes on around a minimum at a constant speed; its path is taken up to the
        // first point at which the energy rises.
        stop_on_energy_rise: method == "dvv",
        ..IrcSettings::default()
    }
}

/// The reference path of a side: the saddle, then RK4 points 0.002 apart.
fn reference(saddle: Point, side: Sides) -> Vec<Point> {
    let mut s = settings("rk4", 0.002);
    s.max_steps = 1500;
    s.sides = side;
    s.initial_displacement = Some(1e-3);
    let path = follow(s, saddle);
    let mut points = vec![saddle];
    points.extend(path.points.iter().map(|(_, p)| *p));
    points
}

fn distance(a: Point, b: Point) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
}

/// Distance of `p` from the polyline `line`.
fn off_path(p: Point, line: &[Point]) -> f64 {
    line.windows(2)
        .map(|w| {
            let (a, b) = (w[0], w[1]);
            let ab = [b[0] - a[0], b[1] - a[1]];
            let len2 = ab[0] * ab[0] + ab[1] * ab[1];
            let t = if len2 == 0.0 {
                0.0
            } else {
                (((p[0] - a[0]) * ab[0] + (p[1] - a[1]) * ab[1]) / len2).clamp(0.0, 1.0)
            };
            distance(p, [a[0] + t * ab[0], a[1] + t * ab[1]])
        })
        .fold(f64::INFINITY, f64::min)
}

fn nearest_minimum(p: Point) -> (usize, f64) {
    let mut best = (0, f64::INFINITY);
    for (k, m) in MB_MINIMA.iter().enumerate() {
        let d = distance(p, *m);
        if d < best.1 {
            best = (k, d);
        }
    }
    best
}

/// (method, step size, largest accepted distance of a path point from the reference path)
/// (the limits are about 1.5 times the largest distance measured)
const CASES: [(&str, f64, f64); 6] = [
    ("lqa", 0.05, 1.2e-2),
    ("hpc", 0.05, 1.7e-2),
    ("gs2", 0.05, 1e-3),
    ("rk4", 0.03, 1e-4),
    ("euler", 0.01, 5e-3),
    ("dvv", 0.05, 5e-2),
];

/// Largest distance of the points of `path` from the reference paths of the two sides.
/// Points closer than 0.15 to a minimum are left out: the reference path ends short of the
/// minima (RK4 does not arrive), and every integrator ends its path there in its own way.
fn largest_deviation(path: &Path, forward: &[Point], backward: &[Point]) -> f64 {
    path.points
        .iter()
        .filter(|(_, p)| nearest_minimum(*p).1 > 0.15)
        .map(|(branch, p)| {
            let line = if *branch == Branch::Forward {
                forward
            } else {
                backward
            };
            off_path(*p, line)
        })
        .fold(0.0, f64::max)
}

#[test]
fn every_integrator_stays_on_the_path() {
    for saddle in MB_SADDLES {
        let forward = reference(saddle, Sides::Forward);
        let backward = reference(saddle, Sides::Backward);
        for (method, step, limit) in CASES {
            let path = follow(settings(method, step), saddle);
            let deviation = largest_deviation(&path, &forward, &backward);
            assert!(
                deviation < limit,
                "{method} from {saddle:?}: {deviation} off the path"
            );
            for side in &path.end.branches {
                let (_, d) = nearest_minimum([side.x[0], side.x[1]]);
                assert!(d < 0.2, "{method} {saddle:?}: {d} from a minimum");
            }
        }
    }
}

/// The table of the documentation (`cargo test -p mopt-rs --test irc_muller_brown
/// accuracy_and_cost -- --nocapture`): every integrator at several steps from both saddles.
#[test]
fn accuracy_and_cost_of_the_integrators() {
    println!(
        "| method | step | saddle | points | evaluations | largest distance from the path | ends |"
    );
    println!("|---|---|---|---|---|---|---|");
    for (k, saddle) in MB_SADDLES.iter().enumerate() {
        let forward = reference(*saddle, Sides::Forward);
        let backward = reference(*saddle, Sides::Backward);
        let mut cases: Vec<(&str, f64)> = CASES.iter().map(|c| (c.0, c.1)).collect();
        cases.extend([
            ("lqa", 0.02),
            ("lqa", 0.1),
            ("lqa", 0.2),
            ("lqa", 0.5),
            ("gs2", 0.1),
            ("gs2", 0.2),
            ("hpc", 0.1),
            ("hpc", 0.2),
        ]);
        for (method, step) in cases {
            let path = follow(settings(method, step), *saddle);
            report(method, &step.to_string(), k, &path, &forward, &backward);
        }
        // LQA with the exact Hessian at every point, GS2 with other ends of its
        // minimizations.
        for step in [0.02, 0.05, 0.1] {
            let mut s = settings("lqa", step);
            s.exact_hessian_interval = Some(1);
            let path = follow(s, *saddle);
            report(
                "lqa, exact Hessians",
                &step.to_string(),
                k,
                &path,
                &forward,
                &backward,
            );
        }
        for tolerance in [1e-3, 1e-4, 1e-5] {
            let mut s = settings("gs2", 0.1);
            s.sphere.gradient_tolerance = tolerance;
            let path = follow(s, *saddle);
            let label = format!("gs2, sphere_gradient={tolerance:e}");
            report(&label, "0.1", k, &path, &forward, &backward);
        }
        for share in [1e-2, 1e-3, 1e-4] {
            let mut s = settings("gs2", 0.1);
            s.sphere.step_share = share;
            let path = follow(s, *saddle);
            let label = format!("gs2, sphere_step={share:e}");
            report(&label, "0.1", k, &path, &forward, &backward);
        }
    }
}

fn report(method: &str, step: &str, k: usize, path: &Path, forward: &[Point], backward: &[Point]) {
    {
        {
            let ends: Vec<String> = path
                .end
                .branches
                .iter()
                .map(|b| {
                    let (m, d) = nearest_minimum([b.x[0], b.x[1]]);
                    format!("{:?} at {d:.1e} from minimum {m}", b.end)
                })
                .collect();
            println!(
                "| {method} | {step} | {} | {} | {} | {:.2e} | {} |",
                k + 1,
                path.points.len(),
                path.evaluations,
                largest_deviation(path, forward, backward),
                ends.join("; ")
            );
        }
    }
}

fn checked(length: f64, check: bool) -> IrcSettings {
    let mut s = settings("lqa", 0.05);
    s.initial_displacement = Some(length);
    s.first_step.check = check;
    s
}

#[test]
fn a_long_first_step_is_put_back_on_the_path() {
    // 0.3 along the transition vector of the second saddle: the valley bends away, the
    // straight point lies on its wall.
    let saddle = MB_SADDLES[1];
    let lines = [
        (Branch::Forward, reference(saddle, Sides::Forward)),
        (Branch::Backward, reference(saddle, Sides::Backward)),
    ];
    let plain = follow(checked(0.3, false), saddle);
    let corrected = follow(checked(0.3, true), saddle);
    assert!(plain.first_steps.is_empty());
    assert_eq!(corrected.first_steps.len(), 2);
    for (branch, line) in &lines {
        let straight = off_path(plain.side(*branch)[0], line);
        let repaired = off_path(corrected.side(*branch)[0], line);
        let report = corrected
            .first_steps
            .iter()
            .find(|r| r.branch == *branch)
            .unwrap();
        // Both sides are outside the limits of the test and are corrected.
        assert!(report.corrected && report.converged && report.iterations >= 2);
        assert!(report.test.angle_deg > 10.0 && report.test.rho < 0.75);
        assert!(report.energy_gain < 0.0);
        // On the sphere around the saddle, and closer to the path.
        let first = corrected.side(*branch)[0];
        assert!((distance(first, saddle) - 0.3).abs() < 1e-9);
        assert!(
            repaired < 0.6 * straight,
            "{branch:?}: {straight} -> {repaired}"
        );
        if *branch == Branch::Forward {
            // The straight point climbed the wall of the valley (no energy gained, the
            // gradient points back); the corrected one is on its floor.
            assert!(report.test.angle_deg > 90.0 && report.test.rho.abs() < 0.05);
            assert!(
                straight > 0.1 && repaired < 2e-3,
                "{straight} -> {repaired}"
            );
        }
    }
    // With the correction both sides still end in the minima the saddle connects.
    let mut reached: Vec<usize> = corrected
        .end
        .branches
        .iter()
        .map(|b| {
            assert_eq!(b.end, BranchEnd::Converged);
            nearest_minimum([b.x[0], b.x[1]]).0
        })
        .collect();
    reached.sort_unstable();
    assert_eq!(reached, vec![1, 2]);
    // A short first step passes the test and is not touched.
    let short = follow(checked(0.005, true), saddle);
    assert!(short
        .first_steps
        .iter()
        .all(|r| !r.corrected && r.iterations == 0));
    assert_eq!(
        short.side(Branch::Forward)[0],
        follow(checked(0.005, false), saddle).side(Branch::Forward)[0]
    );
}

/// The table of the documentation (`cargo test -p mopt-rs --test irc_muller_brown
/// first_step_numbers -- --nocapture`): the test of the first step over its length.
#[test]
fn first_step_numbers_over_the_length() {
    println!("| saddle | side | length | angle (deg) | rho | off the path | corrected | evaluations | off the path after |");
    println!("|---|---|---|---|---|---|---|---|---|");
    for (k, saddle) in MB_SADDLES.iter().enumerate() {
        let lines = [
            (Branch::Forward, reference(*saddle, Sides::Forward)),
            (Branch::Backward, reference(*saddle, Sides::Backward)),
        ];
        for length in [0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4] {
            let mut s = checked(length, true);
            s.max_steps = 3;
            let plain = {
                let mut p = checked(length, false);
                p.max_steps = 3;
                follow(p, *saddle)
            };
            let path = follow(s, *saddle);
            for (branch, line) in &lines {
                let r = path
                    .first_steps
                    .iter()
                    .find(|r| r.branch == *branch)
                    .unwrap();
                println!(
                    "| {} | {} | {length} | {:.2} | {:.3} | {:.2e} | {} | {} | {:.2e} |",
                    k + 1,
                    branch.label(),
                    r.test.angle_deg,
                    r.test.rho,
                    off_path(plain.side(*branch)[0], line),
                    if r.corrected { "yes" } else { "no" },
                    r.iterations,
                    off_path(path.side(*branch)[0], line),
                );
            }
        }
    }
}
