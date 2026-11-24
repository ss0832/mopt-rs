//! Tests of the placement of the images along a chain (`redistribute.rs`).
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use super::*;

fn col(values: &[f64]) -> Col<f64> {
    Col::from_fn(values.len(), |k| values[k])
}

fn along_x(positions: &[f64]) -> Vec<Col<f64>> {
    positions.iter().map(|&x| col(&[x, 0.0])).collect()
}

#[test]
fn images_are_spread_evenly_or_at_a_given_distance() {
    let chain = along_x(&[0.0, 0.1, 0.2, 3.0]);
    let even = equal_spacing(&chain, ChainKind::Variables, Curve::Polyline);
    for (found, expected) in even.iter().zip([0.0, 1.0, 2.0, 3.0]) {
        assert!((found[0] - expected).abs() < 1e-14);
    }
    // A bent chain: the new images lie on its segments.
    let bent = vec![col(&[0.0, 0.0]), col(&[1.0, 0.0]), col(&[1.0, 3.0])];
    let even = equal_spacing(&bent, ChainKind::Variables, Curve::Polyline);
    assert!((even[1][0] - 1.0).abs() < 1e-14 && (even[1][1] - 1.0).abs() < 1e-14);
    // Between anchors: the anchors stay, the images between them are spread evenly.
    let uneven = along_x(&[0.0, 0.1, 1.0, 1.1, 1.2, 4.0]);
    let kept = equal_spacing_between(&uneven, ChainKind::Variables, &[2, 0, 9], Curve::Polyline);
    let found: Vec<f64> = kept.iter().map(|p| p[0]).collect();
    for (a, b) in found.iter().zip([0.0, 0.5, 1.0, 2.0, 3.0, 4.0]) {
        assert!((a - b).abs() < 1e-14, "{found:?}");
    }
    assert_eq!(
        equal_spacing_between(&uneven, ChainKind::Variables, &[], Curve::Polyline),
        equal_spacing(&uneven, ChainKind::Variables, Curve::Polyline)
    );
    // A point as a chain stays as it is.
    let still = along_x(&[1.0, 1.0, 1.0]);
    assert_eq!(
        equal_spacing(&still, ChainKind::Variables, Curve::Polyline),
        still
    );

    // On the smooth curve: images on a quarter circle of radius 2, unevenly placed,
    // stay within 4e-3 of the circle (the straight lines cut 0.06 off it), and images
    // that are evenly placed already stay exactly where they are.
    let on_circle = |angles: &[f64]| -> Vec<Col<f64>> {
        angles
            .iter()
            .map(|a| col(&[2.0 * a.cos(), 2.0 * a.sin()]))
            .collect()
    };
    let off = |points: &[Col<f64>]| {
        points
            .iter()
            .map(|p| ((p[0] * p[0] + p[1] * p[1]).sqrt() - 2.0).abs())
            .fold(0.0_f64, f64::max)
    };
    let arc = on_circle(&[0.0, 0.1, 0.5, 0.7, 1.3, 1.5]);
    let smooth = equal_spacing(&arc, ChainKind::Variables, Curve::Smooth);
    let straight = equal_spacing(&arc, ChainKind::Variables, Curve::Polyline);
    assert!(off(&smooth) < 4e-3, "{}", off(&smooth));
    assert!(off(&straight) > 0.04, "{}", off(&straight));
    let even = on_circle(&[0.0, 0.3, 0.6, 0.9, 1.2, 1.5]);
    let kept = equal_spacing(&even, ChainKind::Variables, Curve::Smooth);
    for (a, b) in kept.iter().zip(&even) {
        assert!((a - b).iter().all(|d| d.abs() < 1e-14));
    }
    assert_eq!(Curve::of(Compatibility::Consistent), Curve::Smooth);
    assert_eq!(Curve::of(Compatibility::MultiOptPy), Curve::Polyline);
    // Two images, or images on one point, have a curve too.
    let two = along_x(&[0.0, 1.0]);
    assert_eq!(CubicThrough::new(&two, &[0.0, 1.0]).point(0.25)[0], 0.25);
    let one = along_x(&[1.0, 1.0, 1.0]);
    assert_eq!(CubicThrough::new(&one, &[0.0, 0.0, 0.0]).point(0.0)[0], 1.0);

    // Every 0.8 along a chain of length 3: 0, 0.8, 1.6, 2.4, and the end.
    let spaced = by_spacing(&chain, 0.8, ChainKind::Variables);
    let found: Vec<f64> = spaced.iter().map(|p| p[0]).collect();
    for (a, b) in found.iter().zip([0.0, 0.8, 1.6, 2.4, 3.0]) {
        assert!((a - b).abs() < 1e-14);
    }
    assert_eq!(found.len(), 5);
    // A spacing that divides the length: the end is not added twice.
    let near_end = by_spacing(&chain, 0.99999, ChainKind::Variables);
    assert_eq!(near_end.first(), chain.first());
    assert_eq!(near_end.last(), chain.last());
    assert_eq!(by_spacing(&chain, 1.0, ChainKind::Variables).len(), 4);
    assert_eq!(by_spacing(&still, 0.5, ChainKind::Variables).len(), 1);
}

#[test]
fn the_image_at_a_maximum_moves_to_where_the_maximum_is() {
    // E(x) = -(x - 1.3)^2 along x, images at 0, 1, 2, 3, 4: the maximum of the profile
    // is at the image at 1, the maximum of the energy at 1.3.
    let energy = |x: f64| -(x - 1.3) * (x - 1.3);
    let slope = |x: f64| -2.0 * (x - 1.3);
    let at = [0.0, 1.0, 2.0, 3.0, 4.0];
    let chain = along_x(&at);
    let energies: Vec<f64> = at.iter().map(|&x| energy(x)).collect();
    let gradients: Vec<Col<f64>> = at.iter().map(|&x| col(&[slope(x), 0.7])).collect();
    let moved = towards_predicted_maxima(&chain, &energies, &gradients, Compatibility::Consistent);
    assert!((moved[1][0] - 1.3).abs() < 1e-9, "{}", moved[1][0]);
    // The others are spread evenly between the anchors: 1.3 ... 4 in three steps.
    assert!((moved[2][0] - 2.2).abs() < 1e-9 && (moved[3][0] - 3.1).abs() < 1e-9);
    assert_eq!(moved[0], chain[0]);
    assert_eq!(moved[4], chain[4]);
    // A profile without an inner maximum: spread evenly between the ends.
    let rising: Vec<f64> = at.to_vec();
    let uneven = along_x(&[0.0, 0.5, 1.0, 1.5, 4.0]);
    let moved = towards_predicted_maxima(&uneven, &rising, &gradients, Compatibility::Consistent);
    for (found, expected) in moved.iter().zip(at) {
        assert!((found[0] - expected).abs() < 1e-13);
    }
    // Gathered: the image at the maximum is where it was put (1.3), and its
    // neighbours are nearer to it than with even spreading (0.65 and 2.2); far from
    // the maximum the images are farther apart; the order is kept.
    let gathering = Gathering {
        weight: 2.0,
        width: 0.5,
    };
    let gathered = gathered_at_predicted_maxima(
        &chain,
        &energies,
        &gradients,
        Compatibility::Consistent,
        Some(gathering),
        None,
    );
    let even = towards_predicted_maxima(&chain, &energies, &gradients, Compatibility::Consistent);
    assert!((gathered[1][0] - 1.3).abs() < 1e-9);
    assert!(gathered[2][0] < even[2][0] - 0.1, "{}", gathered[2][0]);
    assert!(gathered[3][0] < even[3][0]);
    assert!(gathered[3][0] - gathered[2][0] > gathered[2][0] - gathered[1][0]);
    assert!(gathered.windows(2).all(|w| w[0][0] < w[1][0]));
    assert_eq!(gathered[4], chain[4]);
    // No weight, no width, or no maximum: spread evenly.
    for none in [
        Gathering {
            weight: 0.0,
            width: 0.5,
        },
        Gathering {
            weight: 2.0,
            width: 0.0,
        },
    ] {
        let same = gathered_at_predicted_maxima(
            &chain,
            &energies,
            &gradients,
            Compatibility::Consistent,
            Some(none),
            None,
        );
        for (a, b) in same.iter().zip(&even) {
            assert!((a - b).iter().all(|d| d.abs() < 1e-12));
        }
    }
    // Equal amounts of the density: two maxima at 2 and 8 of a chain from 0 to 10, six
    // intervals between them; the middle image is in the middle, and the images next
    // to the maxima are closer to them than a sixth of the way.
    let between = positions_between((2.0, 8.0), 6, Some((&[2.0, 8.0], 1.0, 3.0)));
    assert_eq!(between.len(), 5);
    assert!((between[2] - 5.0).abs() < 1e-9);
    assert!(between[0] - 2.0 < 0.6 && 8.0 - between[4] < 0.6);
    assert!((between[0] - 2.0 - (8.0 - between[4])).abs() < 1e-9);
    assert_eq!(positions_between((2.0, 8.0), 3, None), vec![4.0, 6.0]);

    // The form of MultiOptPy: lengths in Angstrom against slopes per Bohr, which puts
    // the maximum elsewhere.
    let legacy = towards_predicted_maxima(&chain, &energies, &gradients, Compatibility::MultiOptPy);
    assert!((legacy[1][0] - 1.3).abs() > 1e-3);
}

#[test]
fn maxima_of_fitted_polynomials() {
    // Through three points of E(s) = 1 - 2 (s - 0.2)^2: the maximum at 0.2.
    let e = |s: f64| 1.0 - 2.0 * (s - 0.2) * (s - 0.2);
    let g = |s: f64| -4.0 * (s - 0.2);
    let s = [-0.7, 0.0, 0.9];
    let found = fitted_maximum(s, s.map(e), s.map(g), None).unwrap();
    assert!((found - 0.2).abs() < 1e-10);
    let with_curvature = fitted_maximum(s, s.map(e), s.map(g), Some([-4.0; 3])).unwrap();
    assert!((with_curvature - 0.2).abs() < 1e-9);
    // A minimum is not a maximum; coinciding points give nothing.
    assert!(fitted_maximum(s, s.map(|x| -e(x)), s.map(|x| -g(x)), None).is_none());
    assert!(fitted_maximum([0.0; 3], [0.0; 3], [0.0; 3], None).is_none());
}

#[test]
fn images_are_added_on_both_sides_of_a_maximum() {
    // A quarter circle of five images; the middle one is the maximum.
    let chain: Vec<Col<f64>> = (0..5)
        .map(|i| {
            let angle = std::f64::consts::FRAC_PI_2 * i as f64 / 4.0;
            Col::from_fn(2, |k| if k == 0 { angle.cos() } else { angle.sin() })
        })
        .collect();
    let (straight, sources) = with_images_around(&chain, &[2], 2, Curve::Polyline);
    assert_eq!(straight.len(), 9);
    assert_eq!(
        sources,
        vec![
            ImageSource::Old(0),
            ImageSource::Old(1),
            ImageSource::New { like: 1 },
            ImageSource::New { like: 2 },
            ImageSource::Old(2),
            ImageSource::New { like: 2 },
            ImageSource::New { like: 3 },
            ImageSource::Old(3),
            ImageSource::Old(4),
        ]
    );
    // MultiOptPy's points: thirds of the straight line between the images.
    let third = &chain[1] + (&chain[2] - &chain[1]) * (1.0 / 3.0);
    assert!(col_norm(&(&straight[2] - &third)) < 1e-14);
    assert_eq!(straight[4], chain[2]);
    // On the curve through the images the new points lie on the circle within 5e-4;
    // the straight lines cut it by more than 3e-3.
    let (curved, _) = with_images_around(&chain, &[2], 2, Curve::Smooth);
    let off = |points: &[Col<f64>]| {
        points
            .iter()
            .map(|p| (col_norm(p) - 1.0).abs())
            .fold(0.0_f64, f64::max)
    };
    assert!(off(&curved) < 1e-3, "{}", off(&curved));
    assert!(off(&straight) > 3e-3);
    // No maximum, an end, or nothing to add: the chain as it is.
    for (maxima, added) in [(vec![], 2), (vec![0, 4], 2), (vec![2], 0)] {
        let (same, sources) = with_images_around(&chain, &maxima, added, Curve::Smooth);
        assert_eq!(same.len(), 5);
        assert!(sources
            .iter()
            .enumerate()
            .all(|(i, s)| *s == ImageSource::Old(i)));
    }
}

#[test]
fn images_on_the_bezier_curve_of_a_chain() {
    let point = |x: f64, y: f64| Col::from_fn(2, |k| if k == 0 { x } else { y });
    // Control points at equal distances on a line: the curve is the line, the parameter
    // the share of its length.
    let line: Vec<Col<f64>> = (0..5).map(|k| point(f64::from(k), 0.0)).collect();
    let on_it = bezier_by_spacing(&line, 1.5, ChainKind::Variables);
    let firsts: Vec<f64> = on_it.iter().map(|p| p[0]).collect();
    assert_eq!(firsts.len(), 4);
    for (found, expected) in firsts.iter().zip([0.0, 1.5, 3.0, 4.0]) {
        assert!((found - expected).abs() < 1e-12, "{firsts:?}");
    }
    // A corner: the ends are kept, the curve cuts the corner, and a distance that divides
    // the length has the end once.
    let corner = [point(0.0, 0.0), point(1.0, 0.0), point(1.0, 1.0)];
    let cut = bezier_by_spacing(&corner, 1.0, ChainKind::Variables);
    assert_eq!(cut.len(), 3);
    assert!(col_norm(&(&cut[0] - &corner[0])) < 1e-14);
    assert!(col_norm(&(&cut[2] - &corner[2])) < 1e-14);
    assert!((cut[1][0] - 0.75).abs() < 1e-14 && (cut[1][1] - 0.25).abs() < 1e-14);
    // Many control points: no overflow (the binomial coefficients of 400 points are not
    // numbers of this size).
    let long: Vec<Col<f64>> = (0..400).map(|k| point(f64::from(k), 0.0)).collect();
    let on_long = bezier_by_spacing(&long, 100.0, ChainKind::Variables);
    assert!(on_long.iter().all(|p| p[0].is_finite()));
    assert!((on_long[1][0] - 100.0).abs() < 1e-9, "{}", on_long[1][0]);
    // No length or no distance: the chain as it is.
    assert_eq!(
        bezier_by_spacing(&corner, 0.0, ChainKind::Variables).len(),
        3
    );
}
