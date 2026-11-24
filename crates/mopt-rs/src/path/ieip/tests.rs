//! Tests of the iEIP state machine on surfaces of plain variables.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used)]

use faer::Col;

use super::*;

/// `E = (x^2 - 1)^2 + (y - 0.2 x)^2`: minima at `(+-1, +-0.2)`, the saddle at the origin.
fn double_well(p: &Col<f64>) -> ImageValue {
    let (x, y) = (p[0], p[1]);
    let w = y - 0.2 * x;
    let e = (x * x - 1.0).powi(2) + w * w;
    let g = Col::from_fn(2, |i| {
        if i == 0 {
            4.0 * x * (x * x - 1.0) - 0.4 * w
        } else {
            2.0 * w
        }
    });
    ImageValue::plain(e, g)
}

fn point(x: f64, y: f64) -> Col<f64> {
    Col::from_fn(2, |i| if i == 0 { x } else { y })
}

fn run(settings: IeipSettings) -> (ElasticImagePair, usize) {
    let mut pair = ElasticImagePair::new(settings, &point(-1.0, -0.2), &point(1.0, 0.2)).unwrap();
    let mut evaluations = 0;
    while let IeipAction::Evaluate([a, b]) = pair.ask() {
        pair.tell([double_well(&a), double_well(&b)]).unwrap();
        evaluations += 1;
        assert!(evaluations < 10_000);
    }
    (pair, evaluations)
}

#[test]
fn the_pair_meets_on_the_ridge_of_a_double_well() {
    let settings = IeipSettings {
        align: false,
        ..IeipSettings::default()
    };
    let (mut pair, evaluations) = run(settings);
    assert_eq!(pair.end(), Some(IeipEnd::Converged));
    assert!(pair.separation() < SEPARATION_TOLERANCE);
    let ts = pair.transition_state_estimate();
    assert!(ts[0].abs() < 0.1 && ts[1].abs() < 0.1, "{ts:?}");
    assert_eq!(evaluations, pair.iteration());
    assert!(matches!(
        pair.ask(),
        IeipAction::Finished(IeipEnd::Converged)
    ));
    assert!(pair.tell([double_well(&ts), double_well(&ts)]).is_err());
}

#[test]
fn relaxation_needs_structures() {
    // The coordinates of the relaxation are those of atoms about where an image was.
    let settings = IeipSettings {
        align: false,
        micro: MicroSettings {
            steps: 3,
            ..MicroSettings::default()
        },
        ..IeipSettings::default()
    };
    assert!(ElasticImagePair::new(settings, &point(-1.0, -0.2), &point(1.0, 0.2)).is_err());
}

#[test]
fn the_first_displacements_by_hand() {
    // One variable: the images at 0 and 1, the second higher.
    let x = [Col::from_fn(1, |_| 0.0), Col::from_fn(1, |_| 1.0)];
    let values = [
        ImageValue::plain(0.0, Col::from_fn(1, |_| 0.5)),
        ImageValue::plain(1.0, Col::from_fn(1, |_| -0.5)),
    ];
    let parts = forces::displacements(
        0,
        [&x[0], &x[1]],
        [&values[0], &values[1]],
        [&x[0], &x[1]],
        1.0,
    );
    // Across the line there is nothing; along it the energy difference and the attraction
    // move the first image up by 0.04 each and its own gradient down by 0.04.
    assert!(parts.perpendicular[0][0].abs() < 1e-9);
    assert!(
        (parts.total[0][0] - 0.04).abs() < 1e-9,
        "{:?}",
        parts.total[0]
    );
    assert!(
        (parts.total[1][0] + 0.04).abs() < 1e-9,
        "{:?}",
        parts.total[1]
    );
    // AdaBelief turns the first displacement into a step of the same sign.
    let mut m = Moments::zeros(1);
    let step = m.step(&parts.total[0]);
    assert!(step[0] > 0.0 && step[0] < 0.04);
}

#[test]
fn laid_onto_each_other() {
    // A structure turned by 30 degrees about z and shifted.
    let atoms = [
        [0.0, 0.0, 0.0],
        [1.2, 0.1, 0.0],
        [0.3, 1.1, 0.4],
        [-0.5, 0.2, 0.9],
    ];
    let (c, s) = (libm::cos(0.5), libm::sin(0.5));
    let p = Col::from_fn(12, |j| atoms[j / 3][j % 3]);
    let q = Col::from_fn(12, |j| {
        let [x, y, z] = atoms[j / 3];
        [c * x - s * y + 1.0, s * x + c * y - 2.0, z + 0.5][j % 3]
    });
    let (a, b) = aligned_pair(&p, &q);
    for j in 0..12 {
        assert!((a[j] - b[j]).abs() < 1e-10, "{j}");
    }
    let centroid = (0..4).map(|i| b[3 * i]).sum::<f64>();
    assert!(centroid.abs() < 1e-12);
}

#[test]
fn bad_input_is_refused() {
    let a = point(0.0, 0.0);
    assert!(ElasticImagePair::new(IeipSettings::default(), &a, &a).is_err());
    let settings = IeipSettings {
        align: false,
        ..IeipSettings::default()
    };
    assert!(ElasticImagePair::new(settings.clone(), &a, &a).is_err());
    assert!(ElasticImagePair::new(settings.clone(), &a, &Col::zeros(3)).is_err());
    let bad = IeipSettings {
        beta: -1.0,
        ..settings.clone()
    };
    assert!(ElasticImagePair::new(bad, &a, &point(1.0, 0.0)).is_err());
    let mut pair = ElasticImagePair::new(settings, &a, &point(1.0, 0.0)).unwrap();
    let wrong = ImageValue::plain(0.0, Col::zeros(3));
    assert!(pair.tell([wrong.clone(), wrong]).is_err());
}
