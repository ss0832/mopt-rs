//! Tests of the dimer method on surfaces of plain variables.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used)]

use super::*;

/// `E = (x^2 - 1)^2 + 10 (y - 0.2 x)^2 + 10 z^2`: the saddle at the origin; at the
/// minima the softest mode runs along the valley towards it.
fn surface(p: &Col<f64>) -> PointValue {
    let (x, y, z) = (p[0], p[1], p[2]);
    let w = y - 0.2 * x;
    PointValue {
        energy: (x * x - 1.0).powi(2) + 10.0 * w * w + 10.0 * z * z,
        gradient: Some(Col::from_fn(3, |i| {
            [4.0 * x * (x * x - 1.0) - 4.0 * w, 20.0 * w, 20.0 * z][i]
        })),
        hessian: None,
    }
}

#[test]
fn from_a_minimum_to_the_saddle() {
    let start = Col::from_fn(3, |i| [1.0, 0.2, 0.05][i]);
    let axis = Col::from_fn(3, |i| [-1.0, 0.3, 0.2][i]);
    let mut d = Dimer::new(DimerSettings::default(), &start, &axis).unwrap();
    let mut evaluations = 0;
    while let SearchAction::Evaluate(points) = d.ask() {
        evaluations += points.len();
        let values = points.iter().map(|p| surface(&p.x)).collect();
        d.tell(values).unwrap();
    }
    assert_eq!(d.end(), Some(DimerEnd::Converged));
    assert!(col_norm(d.center()) < 1e-3, "{:?}", d.center());
    assert!(d.curvature() < 0.0);
    assert!((d.axis()[0].abs() - 0.98).abs() < 0.02, "{:?}", d.axis());
    assert!(evaluations < 400, "{evaluations}");
}
