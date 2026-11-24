//! Tests of the ReSPer coordinates and their pre-images.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used)]

use std::sync::Arc;

use faer::Col;

use super::*;
use crate::analysis::resper::preimage::{aligned_references, preimage, PreimageSettings};
use crate::domain::{Element, SpinCharge};

fn point(x: f64, y: f64) -> Col<f64> {
    Col::from_fn(2, |i| [x, y][i])
}

/// Points along a bent curve in the plane.
fn curve() -> Vec<Col<f64>> {
    (0..9)
        .map(|k| {
            let s = k as f64 / 8.0;
            point(2.0 * s - 1.0, 0.8 * (2.0 * s - 1.0).powi(2))
        })
        .collect()
}

fn structure(bohr: &[[f64; 3]]) -> MolecularSystem {
    let elements = vec![Element::O, Element::H, Element::H, Element::C][..bohr.len()].to_vec();
    let flat = Col::from_fn(3 * bohr.len(), |k| bohr[k / 3][k % 3]);
    MolecularSystem::from_flat_bohr(elements, &flat, SpinCharge::default(), None).unwrap()
}

/// A bent four-atom molecule whose last atom swings out of the plane step by step.
fn swing() -> Vec<MolecularSystem> {
    (0..7)
        .map(|k| {
            let phi = 0.25 * k as f64;
            structure(&[
                [0.0, 0.0, 0.0],
                [1.8, 0.0, 0.0],
                [-0.5, 1.7, 0.0],
                [2.4, 1.6 * phi.cos(), 1.6 * phi.sin() + 0.1 * k as f64],
            ])
        })
        .collect()
}

#[test]
fn gower_gives_the_references_their_coordinates_back() {
    let map = ResperMap::from_points(curve()).unwrap();
    assert_eq!(map.components(), 2);
    for (i, p) in curve().iter().enumerate() {
        let (y, _) = map.project_point(p).unwrap();
        for (k, value) in y.iter().enumerate() {
            assert!(
                (value - map.reference_coordinate(i, k)).abs() < 1e-10,
                "{i} {k}"
            );
        }
    }
    // Points of the plane are placed isometrically: their distances to the references in
    // the map are the true ones.
    let p = point(0.3, -0.4);
    let (y, _) = map.project_point(&p).unwrap();
    for (i, r) in curve().iter().enumerate() {
        let true_distance = (&p - r).norm_l2();
        let mapped = ((y[0] - map.reference_coordinate(i, 0)).powi(2)
            + (y[1] - map.reference_coordinate(i, 1)).powi(2))
        .sqrt();
        assert!((true_distance - mapped).abs() < 1e-10, "{i}");
    }
    let structures = ResperMap::from_structures(swing(), &[], true).unwrap();
    for (i, s) in swing().iter().enumerate() {
        let (y, _) = structures.project_structure(s).unwrap();
        for (k, value) in y.iter().enumerate() {
            assert!(
                (value - structures.reference_coordinate(i, k)).abs() < 1e-8,
                "{i} {k}: {value} {}",
                structures.reference_coordinate(i, k)
            );
        }
    }
    assert!(ResperMap::from_points(curve()[..2].to_vec()).is_err());
}

#[test]
fn the_gradients_are_those_of_the_coordinates() {
    let map = Arc::new(ResperMap::from_structures(swing(), &[], true).unwrap());
    // A structure between the references, turned and moved.
    let base = swing()[3].coords_flat_bohr();
    let x = Col::from_fn(base.nrows(), |k| {
        base[k] + 0.05 * ((k * 7 % 5) as f64 - 2.0) + if k % 3 == 1 { 0.4 } else { 0.0 }
    });
    let sys = swing()[0].with_flat_coords_bohr(&x).unwrap();
    let (_, gradients) = map.project_structure(&sys).unwrap();
    let h = 1e-5;
    for (k, gradient) in gradients.iter().enumerate() {
        for j in 0..x.nrows() {
            let value = |sign: f64| {
                let mut moved = x.clone();
                moved[j] += sign * h;
                map.project_structure(&sys.with_flat_coords_bohr(&moved).unwrap())
                    .unwrap()
                    .0[k]
            };
            let numeric = (value(1.0) - value(-1.0)) / (2.0 * h);
            assert!(
                (numeric - gradient[j]).abs() < 1e-5 * (1.0 + numeric.abs()),
                "PCo{} d/dx{j}: {numeric} {}",
                k + 1,
                gradient[j]
            );
        }
    }
    // As a surface term, with second derivatives that are symmetric.
    let term = ResperCoordinate::new(Arc::clone(&map), 0)
        .unwrap()
        .with_curvature(true);
    let out = term
        .evaluate(&sys, &TermContext::at_iteration(0, &sys), true)
        .unwrap();
    let hess = out.hessian_bohr.unwrap();
    assert!((0..12).all(|i| (0..12).all(|j| (hess[(i, j)] - hess[(j, i)]).abs() < 1e-9)));
    assert!(ResperCoordinate::new(Arc::clone(&map), 5).is_err());
    // Plain variables.
    let points = Arc::new(ResperMap::from_points(curve()).unwrap());
    let variable = ResperVariable::new(Arc::clone(&points), 1).unwrap();
    let x = point(0.2, 0.5);
    let (_, g) = variable.evaluate(&x).unwrap();
    for j in 0..2 {
        let mut a = x.clone();
        let mut b = x.clone();
        a[j] += h;
        b[j] -= h;
        let numeric =
            (variable.evaluate(&a).unwrap().0 - variable.evaluate(&b).unwrap().0) / (2.0 * h);
        assert!((numeric - g[j]).abs() < 1e-7);
    }
    assert!(ResperVariable::new(map, 0).is_err());
}

#[test]
fn a_preimage_lands_on_its_point() {
    let map = ResperMap::from_points(curve()).unwrap();
    let aligned = aligned_references(&map).unwrap();
    // Halfway between two references along the first coordinate.
    let t = 0.5 * (map.reference_coordinate(3, 0) + map.reference_coordinate(4, 0));
    let off = PreimageSettings {
        correct: false,
        ..PreimageSettings::default()
    };
    let raw = preimage(&map, &aligned, &[0], &[t], &off).unwrap();
    assert!(!raw.extrapolated);
    assert_eq!(raw.weights.len(), 2);
    let fixed = preimage(&map, &aligned, &[0], &[t], &PreimageSettings::default()).unwrap();
    assert!(fixed.remaining < 1e-8, "{}", fixed.remaining);
    assert!(fixed.correction <= raw.error + 1e-9 || raw.error < 1e-9);
    // In both coordinates, near the curve and far from it.
    let near = [
        map.reference_coordinate(5, 0),
        map.reference_coordinate(5, 1),
    ];
    let p = preimage(&map, &aligned, &[0, 1], &near, &PreimageSettings::default()).unwrap();
    assert!(p.remaining < 1e-8 && !p.extrapolated && p.nearest < 1e-9);
    let far = [map.reference_coordinate(8, 0) + 5.0, 3.0];
    assert!(
        preimage(&map, &aligned, &[0, 1], &far, &off)
            .unwrap()
            .extrapolated
    );
    // Structures: the interpolated structure of a point between two references lands near it.
    let structures = ResperMap::from_structures(swing(), &[], true).unwrap();
    let aligned = aligned_references(&structures).unwrap();
    let t = 0.5 * (structures.reference_coordinate(2, 0) + structures.reference_coordinate(3, 0));
    let s = preimage(&structures, &aligned, &[0], &[t], &off).unwrap();
    let spacing =
        (structures.reference_coordinate(2, 0) - structures.reference_coordinate(3, 0)).abs();
    assert!(s.error < 0.2 * spacing, "{} of {spacing}", s.error);
    let c = preimage(
        &structures,
        &aligned,
        &[0],
        &[t],
        &PreimageSettings::default(),
    )
    .unwrap();
    assert!(c.remaining < 1e-8);
    assert!(preimage(&structures, &aligned, &[3], &[t], &off).is_err());
}
