//! Tests of DS-AFIR on a surface of two variables.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used)]

use super::*;

/// `E = (x^2 - 1)^2 + 2 (y - 0.2 x)^2`.
fn surface(p: &Col<f64>) -> PointValue {
    let (x, y) = (p[0], p[1]);
    let w = y - 0.2 * x;
    PointValue {
        energy: (x * x - 1.0).powi(2) + 2.0 * w * w,
        gradient: Some(Col::from_fn(2, |i| {
            [4.0 * x * (x * x - 1.0) - 0.8 * w, 4.0 * w][i]
        })),
        hessian: Some(Mat::from_fn(2, 2, |i, j| {
            [[12.0 * x * x - 4.0 + 0.16, -0.8], [-0.8, 4.0]][i][j]
        })),
    }
}

fn run(settings: DsAfirSettings) -> (DoubleSphereAfir, usize) {
    let a = Col::from_fn(2, |k| [-1.0, -0.2][k]);
    let b = Col::from_fn(2, |k| [1.0, 0.2][k]);
    let mut s = DoubleSphereAfir::new(settings, &a, &b, PointGeometry::default()).unwrap();
    let mut evaluations = 0;
    while let SearchAction::Evaluate(points) = s.ask() {
        evaluations += points.len();
        // The Hessian only where it was asked for.
        let values = points
            .iter()
            .map(|p| {
                let mut v = surface(&p.x);
                if !p.hessian {
                    v.hessian = None;
                }
                v
            })
            .collect();
        s.tell(values).unwrap();
        assert!(evaluations < 10_000);
    }
    (s, evaluations)
}

/// Steps of fixed length straight down the AFIR function.
fn small() -> DsAfirSettings {
    DsAfirSettings {
        integrator: DsIntegrator::Euler,
        ..small_lqa()
    }
}

/// The steps of the paper: the local quadratic approximation.
fn small_lqa() -> DsAfirSettings {
    DsAfirSettings {
        force: 1.0,
        step: 0.02,
        separation: 0.05,
        ..DsAfirSettings::paper()
    }
}

#[test]
fn the_ends_meet_at_the_ridge_with_lqa_steps() {
    // Exact Hessians at every tenth point of an end, at the first point only, and with
    // the curvature of the distance terms; the second term of the function left out.
    let variants = [
        small_lqa(),
        DsAfirSettings {
            hessian_every: 0,
            ..small_lqa()
        },
        DsAfirSettings {
            afir_curvature: true,
            ..small_lqa()
        },
        DsAfirSettings {
            attraction_only: true,
            ..small_lqa()
        },
    ];
    for settings in variants {
        let (s, evaluations) = run(settings.clone());
        assert_eq!(s.end(), Some(DsAfirEnd::Met), "{settings:?}");
        let (x, _) = s.transition_state_estimate().unwrap();
        assert!(col_norm(&x) < 0.1, "{x:?} {settings:?}");
        assert!(evaluations < 400, "{evaluations} {settings:?}");
    }
}

#[test]
fn hessians_are_asked_for_at_the_first_point_and_every_nth_of_an_end() {
    let a = Col::from_fn(2, |k| [-1.0, -0.2][k]);
    let b = Col::from_fn(2, |k| [1.0, 0.2][k]);
    let count = |settings: DsAfirSettings| {
        let mut s = DoubleSphereAfir::new(settings, &a, &b, PointGeometry::default()).unwrap();
        let (mut hessians, mut points_of_end) = (0, [0_usize; 2]);
        let mut asked_at = Vec::new();
        while let SearchAction::Evaluate(points) = s.ask() {
            let values: Vec<PointValue> = points
                .iter()
                .map(|p| {
                    let mut v = surface(&p.x);
                    if p.hessian {
                        hessians += 1;
                    } else {
                        v.hessian = None;
                    }
                    v
                })
                .collect();
            let first = points_of_end == [0, 0];
            let r = s.tell(values).unwrap();
            if first {
                points_of_end = [1, 1];
                asked_at.push((0, points[0].hessian && points[1].hessian));
            } else if let Some(side) = r.moved {
                points_of_end[side] += 1;
                asked_at.push((points_of_end[side] - 1, points[0].hessian));
            }
        }
        (hessians, asked_at)
    };
    // Every third point of an end: the points 0, 3, 6, ... of each.
    let (hessians, asked_at) = count(DsAfirSettings {
        hessian_every: 3,
        ..small_lqa()
    });
    assert!(hessians > 4);
    for (k, asked) in &asked_at {
        assert_eq!(*asked, k % 3 == 0, "point {k}");
    }
    // The first points only.
    let (hessians, _) = count(DsAfirSettings {
        hessian_every: 0,
        ..small_lqa()
    });
    assert_eq!(hessians, 2);
    // No Hessian for steps straight down the function.
    assert_eq!(count(small()).0, 0);
}

#[test]
fn the_ends_meet_at_the_ridge() {
    let (s, _) = run(small());
    assert_eq!(s.end(), Some(DsAfirEnd::Met));
    let (x, _) = s.transition_state_estimate().unwrap();
    assert!(col_norm(&x) < 0.05, "{x:?}");
}

#[test]
fn the_ends_relax_across_the_drive_between_its_steps() {
    let (s, evaluations) = run(DsAfirSettings {
        relax_steps: 5,
        relax_tolerance: 1e-3,
        relax_steps_settings: NodeStepSettings {
            first_scale: 0.1,
            max_step: 0.05,
            ..NodeStepSettings::default()
        },
        ..small()
    });
    assert_eq!(s.end(), Some(DsAfirEnd::Met));
    let (x, _) = s.transition_state_estimate().unwrap();
    assert!(col_norm(&x) < 0.05, "{x:?}");
    assert!(evaluations > 0);
}
