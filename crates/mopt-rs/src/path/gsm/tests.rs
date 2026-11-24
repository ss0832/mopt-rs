//! Tests of the growing string on a surface of two variables.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used)]

use super::*;

/// `E = (x^2 - 1)^2 + 2 (y - 0.2 x)^2`: minima at `(-1, -0.2)` and `(1, 0.2)`, the saddle at
/// the origin.
fn surface(p: &Col<f64>) -> PointValue {
    let (x, y) = (p[0], p[1]);
    let w = y - 0.2 * x;
    PointValue {
        energy: (x * x - 1.0).powi(2) + 2.0 * w * w,
        gradient: Some(Col::from_fn(2, |i| {
            [4.0 * x * (x * x - 1.0) - 0.8 * w, 4.0 * w][i]
        })),
        hessian: None,
    }
}

fn settings() -> GsmSettings {
    GsmSettings {
        steps: NodeStepSettings {
            max_step: 0.05,
            first_scale: 0.1,
            ..NodeStepSettings::default()
        },
        drive_step: 0.25,
        ..GsmSettings::default()
    }
}

fn run(mut s: GrowingString) -> (GrowingString, usize) {
    let mut evaluations = 0;
    while let SearchAction::Evaluate(points) = s.ask() {
        evaluations += points.len();
        let values = points.iter().map(|p| surface(&p.x)).collect();
        s.tell(values).unwrap();
        assert!(evaluations < 100_000);
    }
    (s, evaluations)
}

fn col(v: &[f64]) -> Col<f64> {
    Col::from_fn(v.len(), |k| v[k])
}

#[test]
fn the_double_ended_string_climbs_to_the_saddle() {
    let s = GrowingString::double_ended(
        settings(),
        &col(&[-1.0, -0.2]),
        &col(&[1.0, 0.2]),
        PointGeometry::default(),
    )
    .unwrap();
    let (s, _) = run(s);
    assert_eq!(s.end(), Some(GsmEnd::Converged));
    let (x, _) = s.transition_state_estimate().unwrap();
    assert!(col_norm(&x) < 1e-3, "{x:?}");
    assert_eq!(s.nodes().len(), 9);
}

#[test]
fn the_single_ended_string_follows_its_driving_coordinate() {
    let driving = Driving {
        coordinates: vec![DrivingCoordinate::Variable(0, 1.0)],
        elements: None,
    };
    let s = GrowingString::single_ended(
        settings(),
        &col(&[-1.0, -0.2]),
        driving,
        PointGeometry::default(),
    )
    .unwrap();
    let (s, _) = run(s);
    assert_eq!(s.end(), Some(GsmEnd::Converged));
    let (x, _) = s.transition_state_estimate().unwrap();
    assert!(col_norm(&x) < 1e-3, "{x:?}");
}

#[test]
fn settings_and_inputs_are_checked() {
    let bad = GsmSettings {
        nodes: 2,
        ..GsmSettings::default()
    };
    assert!(
        GrowingString::double_ended(bad, &col(&[0.0]), &col(&[1.0]), PointGeometry::default())
            .is_err()
    );
    let none = Driving {
        coordinates: Vec::new(),
        elements: None,
    };
    assert!(GrowingString::single_ended(
        GsmSettings::default(),
        &col(&[0.0]),
        none,
        PointGeometry::default()
    )
    .is_err());
}
