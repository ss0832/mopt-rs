//! The IRC where it is hard: a valley that is flat along the path, stiff across it, and
//! bent. Steps of a fixed length go from wall to wall there (zigzag), and a Hessian that
//! was updated many times no longer describes the surface. The energy must fall from
//! point to point all the same, and each side must end in its minimum.
//!
//! `E(x, y) = a (x^2 - 1)^2 + K (y - c x^2)^2 / 2` with `a = 1e-4`, `K = 1`, `c = 0.5`: a
//! saddle at the origin, `a` above the minima at `(-1, c)` and `(1, c)`; the curvature
//! across the valley is a thousand times that along it.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used)]

use faer::{Col, Mat};

use mopt_rs::error::MoptResult;
use mopt_rs::irc::{Branch, BranchEnd, Irc, IrcEvent, IrcSettings, IrcSpace, StepCounts};
use mopt_rs::pes::objective::{Objective, ObjectiveValue};
use mopt_rs::workflow::irc::{run_irc, FnObserver, ObjectiveSurface};

const A: f64 = 1e-4;
const K: f64 = 1.0;
const C: f64 = 0.5;

struct FlatValley;

impl Objective for FlatValley {
    fn name(&self) -> &str {
        "flat_valley"
    }

    fn dimension(&self) -> Option<usize> {
        Some(2)
    }

    fn evaluate(&self, p: &Col<f64>, need_hessian: bool) -> MoptResult<ObjectiveValue> {
        let (x, y) = (p[0], p[1]);
        let off = y - C * x * x;
        let gradient = [4.0 * A * x * (x * x - 1.0) - 2.0 * C * K * x * off, K * off];
        let hessian = need_hessian.then(|| {
            let xx = 4.0 * A * (3.0 * x * x - 1.0) - 2.0 * C * K * off + 4.0 * C * C * K * x * x;
            let values = [[xx, -2.0 * C * K * x], [-2.0 * C * K * x, K]];
            Mat::from_fn(2, 2, |r, c| values[r][c])
        });
        Ok(ObjectiveValue {
            value: A * (x * x - 1.0).powi(2) + 0.5 * K * off * off,
            gradient: Col::from_fn(2, |k| gradient[k]),
            hessian,
        })
    }
}

/// One side of a path: its points with their energies, how it ended, and what the step
/// control did.
struct Side {
    points: Vec<([f64; 2], f64)>,
    end: Option<BranchEnd>,
    counts: StepCounts,
}

fn follow(settings: IrcSettings) -> (Side, Side, usize) {
    let surface = ObjectiveSurface(&FlatValley);
    let start = Col::from_fn(2, |_| 0.0);
    let mut irc = Irc::new(settings, IrcSpace::variables(2), start).unwrap();
    let side = || Side {
        points: Vec::new(),
        end: None,
        counts: StepCounts::default(),
    };
    let mut sides = [side(), side()];
    let index = |branch: Branch| usize::from(branch != Branch::Forward);
    let mut observer = FnObserver::new(|event: &IrcEvent| {
        match event {
            IrcEvent::Step { branch, record } => sides[index(*branch)]
                .points
                .push(([record.x[0], record.x[1]], record.biased_energy)),
            IrcEvent::BranchFinished(summary) => {
                sides[index(summary.branch)].end = Some(summary.end);
                sides[index(summary.branch)].counts = summary.counts;
            }
            _ => {}
        }
        Ok(true)
    });
    run_irc(&mut irc, &surface, &mut [&mut observer])
        .unwrap()
        .unwrap();
    let evaluations = irc.evaluations();
    let [forward, backward] = sides;
    (forward, backward, evaluations)
}

fn settings(method: &str) -> IrcSettings {
    IrcSettings {
        step_size: 0.1,
        max_steps: 600,
        method: method.to_string(),
        ..IrcSettings::default()
    }
}

/// Points at which the energy is above that of the point before (by more than the
/// 1e-10 that counts as rounding).
fn rises(side: &Side) -> usize {
    side.points
        .windows(2)
        .filter(|pair| pair[1].1 > pair[0].1 + 1e-10)
        .count()
}

/// The steepest-descent path from `start` to where `|x|` passes 0.999: the gradient flow
/// in time (with the length of the path as the variable explicit steps are unstable in a
/// stiff valley), fourth order, a point every 2e-4.
fn true_path(start: [f64; 2]) -> Vec<[f64; 2]> {
    let dt = 0.05 / K;
    let flow = |p: [f64; 2]| {
        let v = FlatValley
            .evaluate(&Col::from_fn(2, |k| p[k]), false)
            .unwrap();
        [-v.gradient[0], -v.gradient[1]]
    };
    let towards = |p: [f64; 2], f: [f64; 2], h: f64| [p[0] + h * f[0], p[1] + h * f[1]];
    let far = |a: [f64; 2], b: [f64; 2]| ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt();
    let mut p = start;
    let mut path = vec![p];
    for _ in 0..5_000_000 {
        let k1 = flow(p);
        let k2 = flow(towards(p, k1, 0.5 * dt));
        let k3 = flow(towards(p, k2, 0.5 * dt));
        let k4 = flow(towards(p, k3, dt));
        for k in 0..2 {
            p[k] += dt / 6.0 * (k1[k] + 2.0 * k2[k] + 2.0 * k3[k] + k4[k]);
        }
        if far(p, path[path.len() - 1]) > 2e-4 {
            path.push(p);
        }
        if p[0].abs() > 0.999 {
            break;
        }
    }
    path
}

/// Largest distance of the points of `side` (those with `|x| < 0.99`) from `path`.
fn off_the_path(side: &Side, path: &[[f64; 2]]) -> f64 {
    side.points
        .iter()
        .filter(|(p, _)| p[0].abs() < 0.99)
        .map(|(p, _)| {
            path.iter()
                .map(|r| ((r[0] - p[0]).powi(2) + (r[1] - p[1]).powi(2)).sqrt())
                .fold(f64::INFINITY, f64::min)
        })
        .fold(0.0, f64::max)
}

/// Distance of the last point from the minimum it should be in.
fn off_the_minimum(side: &Side) -> f64 {
    let (last, _) = side.points[side.points.len() - 1];
    ((last[0].abs() - 1.0).powi(2) + (last[1] - C).powi(2)).sqrt()
}

#[test]
fn the_energy_falls_and_each_side_ends_in_its_minimum() {
    let (forward, backward, evaluations) = follow(settings("lqa"));
    for side in [&forward, &backward] {
        assert_eq!(side.end, Some(BranchEnd::Converged));
        // No point above the one before it.
        assert_eq!(rises(side), 0);
        // The last point is the minimum: within 0.02 along the flat valley (where a
        // gradient of 1.5e-5 is that far away), with nothing left of the barrier of 1e-4.
        assert!(off_the_minimum(side) < 2e-2, "{}", off_the_minimum(side));
        let (last, energy) = side.points[side.points.len() - 1];
        assert!(energy.abs() < 2e-7, "{energy:e}");
        let value = FlatValley
            .evaluate(&Col::from_fn(2, |k| last[k]), false)
            .unwrap();
        assert!(value.gradient.iter().all(|g| g.abs() < 1.5e-5));
        // Steps were taken again, and the walk says so.
        assert!(side.counts.dropped > 0 && side.counts.approach > 0);
    }
    // The points lie along the steepest-descent path from the first one: within 0.004 at
    // this step of 0.1 (a step that is not shortened lags behind the bending floor by
    // s^2 / 2 = 0.005).
    let path = true_path(forward.points[0].0);
    let off = off_the_path(&forward, &path);
    assert!(off < 6e-3, "{off}");
    // The two sides went to the two minima.
    let x = |side: &Side| side.points[side.points.len() - 1].0[0];
    assert!(x(&forward) * x(&backward) < 0.0);
    assert!(evaluations < 400, "{evaluations} evaluations");
}

#[test]
fn straight_steps_do_not_arrive_at_the_saddle() {
    // Euler's straight steps are stable in this valley below a length of 2 |g| / K, about
    // 1e-4: at the step size every step raises the energy and is halved several times,
    // and the side creeps. What is asked of it: no point of higher energy, and no arrival
    // next to the transition state, where the gradient is below every threshold.
    let mut euler = settings("euler");
    euler.max_steps = 60;
    let (forward, backward, _) = follow(euler);
    for side in [&forward, &backward] {
        assert_eq!(rises(side), 0);
        assert_eq!(side.end, Some(BranchEnd::MaxSteps));
        assert!(side.counts.dropped > side.points.len());
    }
}

#[test]
fn without_the_new_parts_a_side_ends_before_its_minimum() {
    // MultiOptPy's way: the first point as it is, points of higher energy kept, the side
    // over at the first point below the thresholds.
    let mut old = settings("lqa");
    old.first_step.check = false;
    old.retry_on_energy_rise = false;
    old.final_approach = false;
    let (forward, backward, _) = follow(old);
    assert_eq!(forward.counts, StepCounts::default());
    let (new_forward, new_backward, _) = follow(settings("lqa"));
    let worst = |a: &Side, b: &Side| off_the_minimum(a).max(off_the_minimum(b));
    let (before, after) = (
        worst(&forward, &backward),
        worst(&new_forward, &new_backward),
    );
    assert!(before > 5.0 * after, "{before} against {after}");
    let energy = |a: &Side, b: &Side| {
        let last = |s: &Side| s.points[s.points.len() - 1].1;
        last(a).max(last(b))
    };
    assert!(
        energy(&forward, &backward) > 10.0 * energy(&new_forward, &new_backward).max(1e-12),
        "{:e} against {:e}",
        energy(&forward, &backward),
        energy(&new_forward, &new_backward)
    );
}

#[test]
fn with_exact_hessians_the_bend_limit_keeps_the_points_on_the_path() {
    // The step limit from the curvature of the path and the stiffness across it
    // (`-irc_opt bend=0.25`): with the Hessian of every point the steps stay short enough
    // not to cut the bend, and the points lie closer to the path (5.6e-4 against 3.5e-3).
    let exact = |options: &[&str]| {
        let options: Vec<String> = options.iter().map(|s| s.to_string()).collect();
        let mut s = settings("lqa").with_options(&options).unwrap();
        s.exact_hessian_interval = Some(1);
        follow(s)
    };
    let (plain, _, _) = exact(&[]);
    let (limited, _, _) = exact(&["bend=0.25"]);
    let path = true_path(plain.points[0].0);
    let (before, after) = (off_the_path(&plain, &path), off_the_path(&limited, &path));
    assert!(
        after < 1.5e-3 && before > 2.0 * after,
        "{before} against {after}"
    );
    assert!(limited.counts.dropped <= plain.counts.dropped);
    for side in [&plain, &limited] {
        assert_eq!(side.end, Some(BranchEnd::Converged));
        assert_eq!(rises(side), 0);
    }
}
