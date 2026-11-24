//! Tests of the first images between two structures.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Col;

use super::*;
use crate::domain::{Bohr, Vector3};

fn col(values: &[f64]) -> Col<f64> {
    Col::from_fn(values.len(), |k| values[k])
}

fn distance(x: &Col<f64>, i: usize, j: usize) -> f64 {
    (0..3)
        .map(|k| (x[3 * i + k] - x[3 * j + k]).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// The dihedral angle of the atoms 0-1-2-3 in degrees.
fn dihedral(x: &Col<f64>) -> f64 {
    let p = |a: usize| [x[3 * a], x[3 * a + 1], x[3 * a + 2]];
    let sub = |a: [f64; 3], b: [f64; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    let cross = |a: [f64; 3], b: [f64; 3]| {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    };
    let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let (b1, b2, b3) = (sub(p(1), p(0)), sub(p(2), p(1)), sub(p(3), p(2)));
    let (n1, n2) = (cross(b1, b2), cross(b2, b3));
    let size = dot(b2, b2).sqrt();
    let y = dot(cross(n1, n2), b2) / size;
    y.atan2(dot(n1, n2)).to_degrees()
}

/// Four atoms in a chain; the last one turned about the middle bond by `degrees`.
fn rotor(degrees: f64) -> Col<f64> {
    let (c, s) = (degrees.to_radians().cos(), degrees.to_radians().sin());
    col(&[
        -0.6,
        2.0,
        0.0, // bound to the second
        0.0,
        0.0,
        0.0,
        2.8,
        0.0,
        0.0, // the bond that is turned about
        3.4,
        2.0 * c,
        2.0 * s,
    ])
}

const ALL: [FirstPath; 5] = [
    FirstPath::Linear,
    FirstPath::Lst,
    FirstPath::Idpp,
    FirstPath::SequentialIdpp,
    FirstPath::Geodesic,
];

#[test]
fn a_group_that_turns_keeps_its_bond() {
    let (first, last) = (rotor(0.0), rotor(120.0));
    let bond = distance(&first, 2, 3);
    let frame = PathFrame::default();
    for method in ALL {
        let made =
            images_between(method, &first, &last, 3, &frame, &PathSettings::default()).unwrap();
        assert_eq!(made.images.len(), 3, "{}", method.name());
        let shortest = made
            .images
            .iter()
            .map(|x| distance(x, 2, 3))
            .fold(f64::INFINITY, f64::min);
        let angles: Vec<f64> = made.images.iter().map(dihedral).collect();
        if method == FirstPath::Linear {
            // The straight line cuts the corner: the bond is half as long half way.
            assert!(shortest < 0.6 * bond, "{shortest}");
            assert_eq!(made.iterations, 0);
            continue;
        }
        assert!(
            made.converged,
            "{}: {} iterations",
            method.name(),
            made.iterations
        );
        assert!(
            (shortest - bond).abs() < 0.05 * bond,
            "{}: {shortest} of {bond}",
            method.name()
        );
        // The images follow each other on the way round.
        assert!(
            angles[0] > 5.0 && angles[0] < angles[1] && angles[1] < angles[2] && angles[2] < 115.0,
            "{}: {angles:?}",
            method.name()
        );
    }
}

#[test]
fn an_atom_goes_round_a_bond_and_not_through_it() {
    // Two atoms on the x axis and a third that changes sides, a little off the plane.
    let first = col(&[-1.2, 0.0, 0.0, 1.2, 0.0, 0.0, 0.0, -3.0, 0.0]);
    let last = col(&[-1.2, 0.0, 0.0, 1.2, 0.0, 0.0, 0.0, 3.0, 0.4]);
    let frame = PathFrame::default();
    for method in ALL {
        let made =
            images_between(method, &first, &last, 5, &frame, &PathSettings::default()).unwrap();
        let nearest = made
            .images
            .iter()
            .map(|x| distance(x, 0, 2).min(distance(x, 1, 2)))
            .fold(f64::INFINITY, f64::min);
        if method == FirstPath::Linear {
            assert!(nearest < 1.3, "{nearest}");
        } else {
            assert!(nearest > 2.4, "{}: {nearest}", method.name());
        }
    }
}

#[test]
fn cells_fixed_coordinates_and_names() {
    // Two atoms that are neighbours through the wall of the cell and move apart.
    let cell = PeriodicCell::from_lattice_vectors(
        Vector3::<Bohr>::new([10.0, 0.0, 0.0]),
        Vector3::<Bohr>::new([0.0, 10.0, 0.0]),
        Vector3::<Bohr>::new([0.0, 0.0, 10.0]),
        [true, true, true],
    )
    .unwrap();
    let first = col(&[0.5, 5.0, 5.0, 9.0, 5.0, 5.0, 5.0, 5.0, 5.0]);
    let last = col(&[0.5, 5.0, 5.0, 9.0, 7.5, 5.0, 5.0, 5.0, 5.0]);
    let frame = PathFrame {
        elements: Vec::new(),
        cell: Some(cell.clone()),
        fixed_components: vec![0, 1, 2, 6, 7, 8],
    };
    for method in ALL {
        let made =
            images_between(method, &first, &last, 2, &frame, &PathSettings::default()).unwrap();
        for image in &made.images {
            for &k in &frame.fixed_components {
                assert!(
                    (image[k] - first[k]).abs() < 1e-12,
                    "{}: {k}",
                    method.name()
                );
            }
            // The distance through the wall is between those of the ends.
            let d = pairs::pair_distances(image, Some(&cell))[0];
            assert!(d > 1.49 && d < 2.93, "{}: {d}", method.name());
        }
    }

    assert_eq!(
        FirstPath::from_name("S-IDPP").unwrap(),
        FirstPath::SequentialIdpp
    );
    assert_eq!(
        FirstPath::from_name("geodesic").unwrap(),
        FirstPath::Geodesic
    );
    for name in FirstPath::NAMES {
        assert_eq!(FirstPath::from_name(name).unwrap().name(), name);
    }
    assert!(FirstPath::from_name("spline").is_err());
    // Structures of different sizes, and a fixed coordinate that is none.
    assert!(images_between(
        FirstPath::Idpp,
        &first,
        &col(&[0.0; 6]),
        1,
        &PathFrame::default(),
        &PathSettings::default()
    )
    .is_err());
    let wrong = PathFrame {
        fixed_components: vec![9],
        ..PathFrame::default()
    };
    assert!(images_between(
        FirstPath::Lst,
        &first,
        &last,
        1,
        &wrong,
        &PathSettings::default()
    )
    .is_err());
    // No images asked for: none, by every method.
    for method in ALL {
        let none = images_between(
            method,
            &first,
            &last,
            0,
            &PathFrame::default(),
            &PathSettings::default(),
        );
        assert!(none.unwrap().images.is_empty());
    }
}

#[test]
fn three_atoms_on_a_line_change_their_order_without_meeting() {
    // HCN to HNC, both on the x axis: on the straight line the hydrogen passes through
    // the carbon and the nitrogen.
    let first = col(&[0.0, 0.0, 0.0, 2.18, 0.0, 0.0, -2.01, 0.0, 0.0]);
    let last = col(&[0.0, 0.0, 0.0, 2.18, 0.0, 0.0, 4.06, 0.0, 0.0]);
    let frame = PathFrame::default();
    for method in ALL {
        let made =
            images_between(method, &first, &last, 7, &frame, &PathSettings::default()).unwrap();
        let nearest = made
            .images
            .iter()
            .map(|x| distance(x, 0, 2).min(distance(x, 1, 2)))
            .fold(f64::INFINITY, f64::min);
        // The bond of carbon and nitrogen stays about as long as it is.
        let bond = made
            .images
            .iter()
            .map(|x| (distance(x, 0, 1) - 2.18).abs())
            .fold(0.0, f64::max);
        if method == FirstPath::Linear {
            assert!(nearest < 0.6, "{nearest}");
            continue;
        }
        assert!(made.converged, "{}", method.name());
        assert!(nearest > 1.5, "{}: {nearest}", method.name());
        // From the straight line, on which the hydrogen lies in the other atoms, the plain
        // relaxation ends on a path that pulls carbon and nitrogen apart (what the
        // sequential variant was made for); the others keep the bond.
        if method != FirstPath::Idpp {
            assert!(bond < 0.3, "{}: {bond}", method.name());
        }
    }
}

/// `side^3` atoms on a cubic lattice, 4.7 Bohr apart; in the second structure the atoms
/// of one corner are turned by a quarter about the z axis through their middle.
fn lattice_with_a_corner_turned(side: usize) -> (Col<f64>, Col<f64>) {
    let mut first = Vec::new();
    let mut last = Vec::new();
    let centre = 4.7 * 0.5;
    for i in 0..side {
        for j in 0..side {
            for k in 0..side {
                let p = [4.7 * i as f64, 4.7 * j as f64, 4.7 * k as f64];
                first.extend(p);
                if i < 2 && j < 2 && k < 2 {
                    let (x, y) = (p[0] - centre, p[1] - centre);
                    last.extend([centre - y, centre + x, p[2]]);
                } else {
                    last.extend(p);
                }
            }
        }
    }
    (col(&first), col(&last))
}

/// The time of every method: for 64 atoms on every run, where each must converge; for 216
/// and 512 atoms with `MOPT_FULL_MEASUREMENTS` set (`cargo test -p mopt-rs --lib
/// the_time_of_the_methods -- --nocapture`; minutes in a build without optimization).
#[test]
fn the_time_of_the_methods_for_many_atoms() {
    let sides: &[usize] = if std::env::var_os("MOPT_FULL_MEASUREMENTS").is_some() {
        &[4, 6, 8]
    } else {
        &[4]
    };
    for &side in sides {
        let (first, last) = lattice_with_a_corner_turned(side);
        for method in ALL {
            let clock = std::time::Instant::now();
            let made = images_between(
                method,
                &first,
                &last,
                8,
                &PathFrame::default(),
                &PathSettings::default(),
            )
            .unwrap();
            println!(
                "{} atoms, {}: {:.2} s, {} iterations, converged {}",
                side * side * side,
                method.name(),
                clock.elapsed().as_secs_f64(),
                made.iterations,
                made.converged
            );
            assert!(
                made.converged,
                "{} atoms, {}",
                side * side * side,
                method.name()
            );
        }
    }
}

fn elements(symbols: &[&str]) -> Vec<crate::domain::Element> {
    symbols
        .iter()
        .map(|s| crate::domain::Element::parse_token(s).unwrap())
        .collect()
}

#[test]
fn the_flat_bottom_network_of_hcn_and_hnc() {
    use super::fbenm::FlatBottomNetwork;
    let first = col(&[0.0, 0.0, 0.0, 2.18, 0.0, 0.0, -2.01, 0.0, 0.0]);
    let last = col(&[0.0, 0.0, 0.0, 2.18, 0.0, 0.0, 4.06, 0.0, 0.0]);
    let frame = PathFrame {
        elements: elements(&["C", "N", "H"]),
        ..PathFrame::default()
    };
    let model = FlatBottomNetwork::new(&[&first, &last], &frame, true).unwrap();
    // The bond of carbon and nitrogen keeps its length; the hydrogen is bonded to the
    // carbon in one structure and two bonds from it in the other: anything between.
    let (low, high) = model.limits(0, 1);
    assert!((low - 2.18).abs() < 1e-12 && (high - 2.18).abs() < 1e-12);
    let (low, high) = model.limits(0, 2);
    assert!((low - 2.01).abs() < 1e-12 && (high - 4.06).abs() < 1e-12);
    // The hydrogen changes its partner: one correlated pair of pairs.
    assert_eq!(model.correlated_terms(), 1);
    assert_eq!(
        FlatBottomNetwork::new(&[&first, &last], &frame, false)
            .unwrap()
            .correlated_terms(),
        0
    );
    // Nothing at the two structures, something where the hydrogen is far from both
    // partners or in the carbon; the gradient is that of the value.
    assert!(model.value(&first).0.abs() < 1e-20 && model.value(&last).0.abs() < 1e-20);
    for x in [
        col(&[0.0, 0.0, 0.0, 2.18, 0.0, 0.0, 1.1, 4.4, 0.3]),
        col(&[0.1, 0.0, 0.0, 2.5, 0.2, 0.0, 0.4, 0.9, 0.1]),
    ] {
        let (value, gradient) = model.value(&x);
        assert!(value > 0.01, "{value}");
        for k in 0..9 {
            let (mut up, mut down) = (x.clone(), x.clone());
            up[k] += 1e-5;
            down[k] -= 1e-5;
            let numeric = (model.value(&up).0 - model.value(&down).0) / 2e-5;
            assert!(
                (gradient[k] - numeric).abs() < 1e-6 * (1.0 + numeric.abs()),
                "{k}"
            );
        }
    }
    // No elements, no model.
    assert!(FlatBottomNetwork::new(&[&first, &last], &PathFrame::default(), true).is_err());

    // The images: the hydrogen goes round, the bond stays, with and without the
    // correlated term.
    for method in [FirstPath::Fbenm, FirstPath::Cfbenm] {
        let made =
            images_between(method, &first, &last, 7, &frame, &PathSettings::default()).unwrap();
        let nearest = made
            .images
            .iter()
            .map(|x| distance(x, 0, 2).min(distance(x, 1, 2)))
            .fold(f64::INFINITY, f64::min);
        let bond = made
            .images
            .iter()
            .map(|x| (distance(x, 0, 1) - 2.18).abs())
            .fold(0.0, f64::max);
        println!(
            "{}: nearest {nearest:.3}, bond off {bond:.3}, {} iterations, converged {}",
            method.name(),
            made.iterations,
            made.converged
        );
        assert!(made.converged, "{}", method.name());
        assert!(
            nearest > 1.5 && bond < 0.3,
            "{}: {nearest} {bond}",
            method.name()
        );
        // Every image is in the flat bottom or near it.
        let worst = made
            .images
            .iter()
            .map(|x| model.value(x).0)
            .fold(0.0, f64::max);
        if method == FirstPath::Cfbenm {
            assert!(worst < 0.05, "{worst}");
        }
    }
}

#[test]
fn four_atoms_in_a_plane_keep_their_distances() {
    use super::fbenm::FlatBottomNetwork;
    // Four carbon atoms in a row and in a plane (the outer two 3 bonds apart).
    let flat = |shift: f64| {
        col(&[
            -3.6,
            1.6 + shift,
            0.0,
            -1.3,
            0.0,
            0.0,
            1.3,
            0.0,
            0.0,
            3.6,
            1.6,
            0.0,
        ])
    };
    let frame = PathFrame {
        elements: elements(&["C", "C", "C", "C"]),
        ..PathFrame::default()
    };
    let (first, last) = (flat(0.0), flat(0.0));
    let model = FlatBottomNetwork::new(&[&first, &last], &frame, true).unwrap();
    let outer = distance(&first, 0, 3);
    let (low, high) = model.limits(0, 3);
    assert!((low - outer).abs() < 1e-12 && (high - outer).abs() < 1e-12);
    // Turned out of the plane in the second structure: the outer two may come as near as
    // their van der Waals radii and go far.
    let mut twisted = flat(0.0);
    twisted[10] = 0.0;
    twisted[11] = 1.6;
    let model = FlatBottomNetwork::new(&[&first, &twisted], &frame, true).unwrap();
    let (low, high) = model.limits(0, 3);
    let contact = 2.0 * 1.77 * crate::domain::Units::ANGSTROM_TO_BOHR;
    assert!(
        (low - contact.min(distance(&twisted, 0, 3))).abs() < 1e-9,
        "{low}"
    );
    assert!(high > 2.0 * outer - 1e-9, "{high}");
    assert_eq!(model.correlated_terms(), 0);
}
