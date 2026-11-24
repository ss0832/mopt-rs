//! Tests of the placements of an orientation search.
//!
//! MultiOptPy: `Entrypoints/orientation_search.py:make_random_orientation_xyz_file`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used)]

use super::*;
use crate::domain::quantity::shorthand::bohr;
use crate::domain::{positions, Angstrom, Element, PeriodicCell, SpinCharge};
use crate::workflow::explore::identity::DEFAULT_BOND_SCALE;

const WATER: [[f64; 3]; 3] = [
    [0.0, 0.0, 0.117],
    [0.0, 0.757, -0.469],
    [0.0, -0.757, -0.469],
];

fn waters(n: usize) -> MolecularSystem {
    let mut pos = Vec::new();
    for k in 0..n {
        pos.extend(WATER.iter().map(|p| [p[0] + 4.0 * k as f64, p[1], p[2]]));
    }
    MolecularSystem::new(
        [Element::O, Element::H, Element::H].repeat(n),
        &positions::<Angstrom>(&pos),
        SpinCharge::default(),
        None,
    )
    .unwrap()
}

fn sampler(sys: &MolecularSystem, scheme: Scheme) -> Sampler {
    let parts = ComplexParts::molecules(sys, &[], DEFAULT_BOND_SCALE).unwrap();
    let settings = PlacementSettings {
        scheme,
        clash: 0.0,
        ..PlacementSettings::default()
    };
    Sampler::new(sys, &parts, settings, DEFAULT_BOND_SCALE).unwrap()
}

fn centre(sys: &MolecularSystem, atoms: &[usize]) -> [f64; 3] {
    centroid(
        &atoms
            .iter()
            .map(|&i| sys.position(AtomId::new(i)).value())
            .collect::<Vec<_>>(),
    )
}

fn distance(a: [f64; 3], b: [f64; 3]) -> f64 {
    norm([a[0] - b[0], a[1] - b[1], a[2] - b[2]])
}

#[test]
fn the_rotation_is_multioptpys_row_vector_product() {
    // MultiOptPy: geom @ Rx(a), then @ Ry(b), then @ Rz(c), with its matrices.
    let (a, b, c): (f64, f64, f64) = (0.7, 2.1, 4.4);
    let rx = [
        [1.0, 0.0, 0.0],
        [0.0, a.cos(), -a.sin()],
        [0.0, a.sin(), a.cos()],
    ];
    let ry = [
        [b.cos(), 0.0, b.sin()],
        [0.0, 1.0, 0.0],
        [-b.sin(), 0.0, b.cos()],
    ];
    let rz = [
        [c.cos(), -c.sin(), 0.0],
        [c.sin(), c.cos(), 0.0],
        [0.0, 0.0, 1.0],
    ];
    let row = |v: [f64; 3], m: [[f64; 3]; 3]| -> [f64; 3] {
        std::array::from_fn(|k| (0..3).map(|i| v[i] * m[i][k]).sum())
    };
    let v = [0.3, -1.2, 0.8];
    let theirs = row(row(row(v, rx), ry), rz);
    let ours = apply(&multioptpy_rotation(a, b, c), v);
    assert!(distance(theirs, ours) < 1e-14);
}

#[test]
fn euler_angles_are_not_uniform_rotations() {
    // For rotations uniform over SO(3) every element has the mean square 1/3; MultiOptPy's
    // three independent uniform angles give R_zz = cos(a) cos(b) and a mean square of 1/4.
    let mut rng = Rng::from_seed(3);
    let n = 40_000;
    let (mut euler, mut haar) = (0.0, 0.0);
    for _ in 0..n {
        let r = multioptpy_rotation(
            rng.uniform(0.0, std::f64::consts::TAU),
            rng.uniform(0.0, std::f64::consts::TAU),
            rng.uniform(0.0, std::f64::consts::TAU),
        );
        euler += r[2][2] * r[2][2] / n as f64;
        let u = rng.rotation_matrix_3d();
        haar += u[2][2] * u[2][2] / n as f64;
    }
    assert!((euler - 0.25).abs() < 0.01, "{euler}");
    assert!((haar - 1.0 / 3.0).abs() < 0.01, "{haar}");
}

#[test]
fn multioptpy_polygon_has_neighbours_at_the_drawn_distance() {
    for n in 2..=6 {
        let sys = waters(n);
        let s = sampler(&sys, Scheme::MultiOptPy);
        let parts = ComplexParts::molecules(&sys, &[], DEFAULT_BOND_SCALE).unwrap();
        let d = s.settings().distance.value();
        let mut rng = Rng::from_seed(n as u64);
        let mut beyond = 0;
        for _ in 0..5 {
            let (x, redraws) = s.sample(&mut rng).unwrap();
            assert_eq!(redraws, 0);
            let c: Vec<[f64; 3]> = parts.atoms.iter().map(|a| centre(&x, a)).collect();
            // All centres on one circle about the origin in the xy plane.
            let r0 = norm(c[0]);
            assert!(c
                .iter()
                .all(|p| (norm(*p) - r0).abs() < 1e-9 && p[2].abs() < 1e-9));
            // The closest neighbours are as far apart as the distance drawn from [d, 1.1 d].
            let mut closest = f64::INFINITY;
            for i in 0..n {
                for j in 0..i {
                    closest = closest.min(distance(c[i], c[j]));
                }
            }
            assert!(
                closest >= d - 1e-9 && closest <= 1.1 * d + 1e-9,
                "{n}: {closest} {d}"
            );
            // MultiOptPy's radius 0.5 d tan((n - 2) pi / n) would put four parts at about
            // 1e16 Angstrom and five or six at 1.81 d or 0.87 d from their neighbours.
            if n == 2 {
                assert!(
                    (c[0][0] + 0.5 * closest).abs() < 1e-9
                        && (c[1][0] - 0.5 * closest).abs() < 1e-9
                );
            }
            if closest > 1.001 * d {
                beyond += 1;
            }
        }
        // The distance is drawn for two parts as well (MultiOptPy keeps d there).
        assert!(beyond > 0, "{n}");
    }
}

#[test]
fn the_contact_scheme_puts_parts_at_van_der_waals_contact() {
    for n in [2, 3, 5] {
        let sys = waters(n);
        let s = sampler(&sys, Scheme::Contact);
        let parts = ComplexParts::molecules(&sys, &[], DEFAULT_BOND_SCALE).unwrap();
        let [lo, hi] = s.settings().contact;
        let mut rng = Rng::from_seed(11 + n as u64);
        for _ in 0..10 {
            let (x, _) = s.sample(&mut rng).unwrap();
            let groups: Vec<Option<Vec<[f64; 3]>>> = parts
                .atoms
                .iter()
                .map(|a| {
                    Some(
                        a.iter()
                            .map(|&i| x.position(AtomId::new(i)).value())
                            .collect(),
                    )
                })
                .collect();
            let closest = s.closest(&groups);
            assert!(closest >= lo - 1e-6, "{n}: {closest}");
            // Every part touches another one within the contact range.
            for k in 0..n {
                let mut placed = Placed::new(None);
                for (m, g) in groups.iter().enumerate() {
                    if m != k {
                        placed.extend(g.as_ref().unwrap(), &s.radii[m]);
                    }
                }
                let r = placed.contact_ratio(groups[k].as_ref().unwrap(), &s.radii[k]);
                assert!(r <= hi + 1e-6, "{n}: part {k} at {r}");
            }
            // The molecules keep their shape.
            for a in &parts.atoms {
                let d = x.distance(AtomId::new(a[0]), AtomId::new(a[1])).value();
                assert!((d - sys.distance(AtomId::new(0), AtomId::new(1)).value()).abs() < 1e-9);
            }
        }
    }
}

#[test]
fn clashing_draws_are_drawn_again() {
    let sys = waters(2);
    let parts = ComplexParts::molecules(&sys, &[], DEFAULT_BOND_SCALE).unwrap();
    let settings = PlacementSettings {
        scheme: Scheme::Uniform,
        distance: Quantity::new(1.6 * Units::ANGSTROM_TO_BOHR),
        clash: 0.6,
        ..PlacementSettings::default()
    };
    let s = Sampler::new(&sys, &parts, settings, DEFAULT_BOND_SCALE).unwrap();
    let mut rng = Rng::from_seed(5);
    let mut redrawn = 0;
    for _ in 0..20 {
        let (x, redraws) = s.sample(&mut rng).unwrap();
        redrawn += redraws;
        let groups: Vec<Option<Vec<[f64; 3]>>> = parts
            .atoms
            .iter()
            .map(|a| {
                Some(
                    a.iter()
                        .map(|&i| x.position(AtomId::new(i)).value())
                        .collect(),
                )
            })
            .collect();
        if redraws + 1 < settings.attempts {
            assert!(s.closest(&groups) >= 0.6);
        }
    }
    assert!(redrawn > 0);
}

/// A copper slab of two layers (3 x 3 atoms each, 2.55 Angstrom apart) in a cell with 20
/// Angstrom along z, and a CO molecule above it.
fn slab_with_co() -> MolecularSystem {
    let a = 2.55;
    let mut pos = Vec::new();
    let mut el = Vec::new();
    for layer in 0..2 {
        for i in 0..3 {
            for j in 0..3 {
                let shift = if layer == 1 { 0.5 * a } else { 0.0 };
                pos.push([
                    a * i as f64 + shift,
                    a * j as f64 + shift,
                    1.8 * layer as f64,
                ]);
                el.push(Element::parse_token("Cu").unwrap());
            }
        }
    }
    pos.extend([[1.0, 1.0, 6.0], [1.0, 1.0, 7.13]]);
    el.extend([Element::C, Element::O]);
    let cell = PeriodicCell::orthorhombic(
        bohr(3.0 * a * 1.889_726_1),
        bohr(3.0 * a * 1.889_726_1),
        bohr(20.0 * 1.889_726_1),
    )
    .unwrap();
    MolecularSystem::new(
        el,
        &positions::<Angstrom>(&pos),
        SpinCharge::default(),
        Some(cell),
    )
    .unwrap()
}

#[test]
fn on_a_slab_the_molecule_comes_down_onto_the_free_side() {
    let sys = slab_with_co();
    let parts = ComplexParts::from_lists(&sys, &["1-18".into()], &[], DEFAULT_BOND_SCALE).unwrap();
    assert_eq!(parts.periodic, vec![true, false]);
    let s = Sampler::new(
        &sys,
        &parts,
        PlacementSettings::default(),
        DEFAULT_BOND_SCALE,
    )
    .unwrap();
    let slab = s.slab().unwrap();
    assert_eq!(slab.axis, 2);
    assert!((slab.top - 1.8 * 1.889_726_1).abs() < 1e-6);
    let mut rng = Rng::from_seed(2);
    for _ in 0..10 {
        let (x, _) = s.sample(&mut rng).unwrap();
        // The slab stays; the molecule is above it at contact.
        for i in 0..18 {
            assert_eq!(
                x.position(AtomId::new(i)).value(),
                sys.position(AtomId::new(i)).value()
            );
        }
        let mut placed = Placed::new(sys.active_cell());
        placed.extend(
            &(0..18)
                .map(|i| x.position(AtomId::new(i)).value())
                .collect::<Vec<_>>(),
            &s.radii[0],
        );
        let co: Vec<[f64; 3]> = (18..20)
            .map(|i| x.position(AtomId::new(i)).value())
            .collect();
        let r = placed.contact_ratio(&co, &s.radii[1]);
        let [lo, hi] = s.settings().contact;
        assert!(r >= lo - 1e-6 && r <= hi + 1e-6, "{r}");
        assert!(co.iter().all(|p| p[2] > slab.top), "{co:?}");
    }
    // The schemes with distances between centres are refused in a cell, as are fixed parts.
    for scheme in [Scheme::MultiOptPy, Scheme::Uniform] {
        let settings = PlacementSettings {
            scheme,
            ..PlacementSettings::default()
        };
        let e = Sampler::new(&sys, &parts, settings, DEFAULT_BOND_SCALE).unwrap_err();
        assert!(e.to_string().contains("periodic cell"), "{e}");
    }
    let w = waters(2);
    let fixed = ComplexParts::molecules(&w, &[1], DEFAULT_BOND_SCALE).unwrap();
    let settings = PlacementSettings {
        scheme: Scheme::MultiOptPy,
        ..PlacementSettings::default()
    };
    assert!(Sampler::new(&w, &fixed, settings, DEFAULT_BOND_SCALE).is_err());
}

#[test]
fn a_perturbed_minimum_keeps_the_other_parts_and_the_shapes() {
    let sys = waters(3);
    let s = sampler(&sys, Scheme::Contact);
    let mut rng = Rng::from_seed(9);
    let (start, _) = s.sample(&mut rng).unwrap();
    for _ in 0..10 {
        let x = s
            .perturb(&start, &Perturbation::default(), &mut rng)
            .unwrap();
        let moved_parts: Vec<usize> = (0..3)
            .filter(|&k| {
                (0..3).any(|m| {
                    x.position(AtomId::new(3 * k + m)).value()
                        != start.position(AtomId::new(3 * k + m)).value()
                })
            })
            .collect();
        assert!(moved_parts.len() <= 1, "{moved_parts:?}");
        for k in 0..3 {
            let d = |s: &MolecularSystem| {
                s.distance(AtomId::new(3 * k), AtomId::new(3 * k + 1))
                    .value()
            };
            assert!((d(&x) - d(&start)).abs() < 1e-9);
        }
        let parts = ComplexParts::molecules(&sys, &[], DEFAULT_BOND_SCALE).unwrap();
        let groups: Vec<Option<Vec<[f64; 3]>>> = parts
            .atoms
            .iter()
            .map(|a| {
                Some(
                    a.iter()
                        .map(|&i| x.position(AtomId::new(i)).value())
                        .collect(),
                )
            })
            .collect();
        assert!(s.closest(&groups) >= s.settings().contact[0] - 1e-6);
    }
}
