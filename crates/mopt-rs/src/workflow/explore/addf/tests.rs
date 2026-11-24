//! Tests of the ADDF state machine on the Muller-Brown surface.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used)]

use faer::Col;

use super::*;
use crate::pes::objective::{Objective, ObjectiveRegistry};
use crate::pes::registry::CalculatorOptions;

fn objective() -> Box<dyn Objective> {
    ObjectiveRegistry::builtin()
        .create("mb2d", &CalculatorOptions::default())
        .unwrap()
}

fn run(settings: AddfSettings, start: [f64; 2]) -> (Addf, usize) {
    let mb = objective();
    let x0 = Col::from_fn(2, |i| start[i]);
    let mut addf = Addf::new(settings, &x0, None).unwrap();
    let mut evaluations = 0;
    while let SearchAction::Evaluate(points) = addf.ask() {
        let values = points
            .iter()
            .map(|p| {
                let v = mb.evaluate(&p.x, p.hessian).unwrap();
                PointValue {
                    energy: v.value,
                    gradient: Some(v.gradient),
                    hessian: v.hessian,
                }
            })
            .collect::<Vec<_>>();
        evaluations += values.len();
        addf.tell(values).unwrap();
        assert!(evaluations < 20_000);
    }
    (addf, evaluations)
}

/// The deepest minimum of the surface (accurate to 1e-6).
const MINIMUM_A: [f64; 2] = [-0.558_224, 1.441_726];
/// Its two saddles.
const SADDLE_1: [f64; 2] = [-0.822_002, 0.624_313];

#[test]
fn from_the_deepest_minimum_of_muller_brown_to_its_saddle() {
    let settings = AddfSettings {
        initial_length: 0.01,
        step_length: 0.02,
        paths: 0,
        gradient_tolerance: 1e-2,
        ..AddfSettings::default()
    };
    let (addf, evaluations) = run(settings, MINIMUM_A);
    assert!(addf.end().is_some());
    let ts = addf.transition_states();
    assert!(!ts.is_empty(), "no transition state");
    let near = ts.iter().any(|(_, p)| {
        let d = libm::hypot(p.x[0] - SADDLE_1[0], p.x[1] - SADDLE_1[1]);
        d < 0.06
    });
    assert!(
        near,
        "{:?}",
        ts.iter().map(|(_, p)| p.x.clone()).collect::<Vec<_>>()
    );
    // Along every path the ADD is negative beyond the first sphere and the radius grows.
    for path in addf.paths() {
        assert!(path.points.windows(2).all(|w| w[1].radius > w[0].radius));
    }
    assert!(evaluations > 10);
}

#[test]
fn a_point_that_is_not_a_minimum_is_refused() {
    let mb = objective();
    let x0 = Col::from_fn(2, |i| SADDLE_1[i] + 0.05 * i as f64);
    let mut addf = Addf::new(AddfSettings::default(), &x0, None).unwrap();
    let points = match addf.ask() {
        SearchAction::Evaluate(points) => points,
        SearchAction::Finished => Vec::new(),
    };
    assert_eq!(points.len(), 1);
    let v = mb.evaluate(&points[0].x, true).unwrap();
    let value = PointValue {
        energy: v.value,
        gradient: Some(v.gradient),
        hessian: v.hessian,
    };
    assert!(addf.tell(vec![value]).is_err());
}
