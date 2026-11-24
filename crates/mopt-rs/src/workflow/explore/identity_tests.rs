//! Tests of the comparison of structures with exchangeable atoms and molecules.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used)]

use super::*;
use crate::domain::quantity::shorthand::bohr;
use crate::domain::{positions, Angstrom, Element, SpinCharge};

const WATER: [[f64; 3]; 3] = [
    [0.0, 0.0, 0.117],
    [0.0, 0.757, -0.469],
    [0.0, -0.757, -0.469],
];

fn rotation(a: f64, b: f64, c: f64) -> [[f64; 3]; 3] {
    let rz = |t: f64| {
        [
            [t.cos(), -t.sin(), 0.0],
            [t.sin(), t.cos(), 0.0],
            [0.0, 0.0, 1.0],
        ]
    };
    let rx = |t: f64| {
        [
            [1.0, 0.0, 0.0],
            [0.0, t.cos(), -t.sin()],
            [0.0, t.sin(), t.cos()],
        ]
    };
    mat_mul(&mat_mul(&rz(a), &rx(b)), &rz(c))
}

fn system(elements: Vec<Element>, pos: &[[f64; 3]], cell: Option<PeriodicCell>) -> MolecularSystem {
    MolecularSystem::new(
        elements,
        &positions::<Angstrom>(pos),
        SpinCharge::default(),
        cell,
    )
    .unwrap()
}

/// Two waters: the second one turned by `turn` and put at `offset`.
fn dimer(turn: [[f64; 3]; 3], offset: [f64; 3]) -> Vec<[f64; 3]> {
    let mut pos: Vec<[f64; 3]> = WATER.to_vec();
    for p in WATER {
        let q = apply(&turn, p);
        pos.push([q[0] + offset[0], q[1] + offset[1], q[2] + offset[2]]);
    }
    pos
}

fn waters(n: usize) -> Vec<Element> {
    [Element::O, Element::H, Element::H].repeat(n)
}

/// `pos` moved rigidly, with the atoms in the order of `order` (new atom k is old atom
/// `order[k]`).
fn moved(pos: &[[f64; 3]], r: [[f64; 3]; 3], shift: [f64; 3], order: &[usize]) -> Vec<[f64; 3]> {
    order
        .iter()
        .map(|&k| {
            let q = apply(&r, pos[k]);
            [q[0] + shift[0], q[1] + shift[1], q[2] + shift[2]]
        })
        .collect()
}

#[test]
fn identical_molecules_and_equivalent_atoms_exchange_places() {
    let pos = dimer(rotation(0.4, 1.1, -0.3), [2.1, 1.7, 0.6]);
    let a = system(waters(2), &pos, None);
    // The two molecules swapped, the hydrogen atoms of the first swapped, all turned.
    let order = [3, 5, 4, 0, 2, 1];
    let b = system(
        waters(2),
        &moved(&pos, rotation(1.3, 0.7, 2.2), [5.0, -1.0, 2.0], &order),
        None,
    );
    let matcher = StructureMatcher::default();
    let found = matcher.best_match(&a, &b).unwrap().unwrap();
    assert!(found.rmsd.value() < 1e-8, "{:?}", found.rmsd);
    // The matching undoes the exchange.
    for (i, &j) in found.permutation.iter().enumerate() {
        assert_eq!(order[j], i, "{:?}", found.permutation);
    }
    assert!(matcher
        .same(&a, Quantity::new(-1.0), &b, Quantity::new(-1.0))
        .unwrap());
    // Beyond the energy tolerance the same geometry is another minimum.
    assert!(!matcher
        .same(&a, Quantity::new(-1.0), &b, Quantity::new(-1.01))
        .unwrap());
}

#[test]
fn a_mirror_image_is_the_same_only_when_asked() {
    // A water dimer in a general orientation has no mirror plane.
    let pos = dimer(rotation(0.9, 0.5, 1.7), [2.4, 1.1, 0.8]);
    let a = system(waters(2), &pos, None);
    let mirrored: Vec<[f64; 3]> = pos.iter().map(|p| [-p[0], p[1], p[2]]).collect();
    let b = system(waters(2), &mirrored, None);
    let with = StructureMatcher::default()
        .best_match(&a, &b)
        .unwrap()
        .unwrap();
    assert!(with.rmsd.value() < 1e-8 && with.mirrored);
    let without = StructureMatcher {
        mirror_images: false,
        ..StructureMatcher::default()
    };
    let m = without.best_match(&a, &b).unwrap().unwrap();
    assert!(m.rmsd.value() > 0.1, "{:?}", m.rmsd);
}

#[test]
fn other_bonds_or_another_arrangement_are_other_minima() {
    let matcher = StructureMatcher::default();
    let a = system(
        waters(2),
        &dimer(rotation(0.0, 0.0, 0.0), [2.9, 0.0, 0.0]),
        None,
    );
    // A proton moved from one water to the other: H3O+ and OH-.
    let mut ion = dimer(rotation(0.0, 0.0, 0.0), [2.9, 0.0, 0.0]);
    ion[1] = [2.0, 0.3, 0.2];
    let b = system(waters(2), &ion, None);
    assert!(matcher.best_match(&a, &b).unwrap().is_none());
    // The second water turned into the perpendicular plane: the same bonds, another
    // arrangement (turning it over instead would give the same dimer by a twofold rotation).
    let quarter = rotation(std::f64::consts::FRAC_PI_2, 0.0, 0.0);
    let c = system(waters(2), &dimer(quarter, [2.9, 0.0, 0.0]), None);
    let m = matcher.best_match(&a, &c).unwrap().unwrap();
    assert!(
        m.rmsd.value() > matcher.rmsd_tolerance.value(),
        "{:?}",
        m.rmsd
    );
    let graph = BondGraph::of(&a, DEFAULT_BOND_SCALE);
    assert_eq!(graph.molecules, vec![vec![0, 1, 2], vec![3, 4, 5]]);
    assert_eq!(graph.molecule_colors[0], graph.molecule_colors[1]);
    assert_eq!(graph.colors[1], graph.colors[2]);
    assert_ne!(graph.colors[0], graph.colors[1]);
    assert!(graph.keeps_bonds(&graph, &[0, 2, 1, 3, 4, 5]));
    assert!(!graph.keeps_bonds(&graph, &[0, 4, 2, 3, 1, 5]));
}

#[test]
fn a_stacked_benzene_dimer_matches_itself_turned_about_its_axis() {
    // Two benzene rings 3.8 Angstrom apart, the upper one turned by 60 degrees: the same
    // structure, which needs the turns about the degenerate axis.
    let ring = |z: f64, phase: f64| -> Vec<[f64; 3]> {
        let mut pos = Vec::new();
        for k in 0..6 {
            let t = phase + std::f64::consts::TAU * k as f64 / 6.0;
            pos.push([1.39 * t.cos(), 1.39 * t.sin(), z]);
        }
        for k in 0..6 {
            let t = phase + std::f64::consts::TAU * k as f64 / 6.0;
            pos.push([2.47 * t.cos(), 2.47 * t.sin(), z]);
        }
        pos
    };
    let elements: Vec<Element> = [vec![Element::C; 6], vec![Element::H; 6]]
        .concat()
        .repeat(2);
    let a_pos = [ring(0.0, 0.0), ring(3.8, 0.3)].concat();
    let b_pos = [
        ring(0.0, std::f64::consts::FRAC_PI_3),
        ring(3.8, 0.3 + std::f64::consts::FRAC_PI_3),
    ]
    .concat();
    let turned = moved(
        &b_pos,
        rotation(0.8, 0.0, 0.0),
        [0.0; 3],
        &(0..24).collect::<Vec<_>>(),
    );
    let a = system(elements.clone(), &a_pos, None);
    let b = system(elements, &turned, None);
    let m = StructureMatcher::default()
        .best_match(&a, &b)
        .unwrap()
        .unwrap();
    assert!(m.rmsd.value() < 1e-6, "{:?}", m.rmsd);
}

#[test]
fn a_vacancy_that_hopped_to_the_next_site_is_the_same_crystal() {
    // Copper in 2 x 2 x 2 face-centred cubic cells without the atom at the origin, and the same
    // with the atom at (a/2, a/2, 0) moved into the vacancy: the crystal moved by that lattice
    // translation of the sublattice.
    let a = 3.615;
    let cell = PeriodicCell::orthorhombic(
        Quantity::<Angstrom>::new(2.0 * a),
        Quantity::<Angstrom>::new(2.0 * a),
        Quantity::<Angstrom>::new(2.0 * a),
    )
    .unwrap();
    let mut sites = Vec::new();
    for i in 0..2 {
        for j in 0..2 {
            for k in 0..2 {
                for f in [
                    [0.0, 0.0, 0.0],
                    [0.5, 0.5, 0.0],
                    [0.5, 0.0, 0.5],
                    [0.0, 0.5, 0.5],
                ] {
                    sites.push([
                        a * (f64::from(i) + f[0]),
                        a * (f64::from(j) + f[1]),
                        a * (f64::from(k) + f[2]),
                    ]);
                }
            }
        }
    }
    let copper = Element::parse_token("Cu").unwrap();
    let with_vacancy = |empty: usize| -> MolecularSystem {
        let pos: Vec<[f64; 3]> = sites
            .iter()
            .enumerate()
            .filter(|(k, _)| *k != empty)
            .map(|(_, p)| *p)
            .collect();
        system(vec![copper; pos.len()], &pos, Some(cell.clone()))
    };
    let (before, after) = (with_vacancy(0), with_vacancy(1));
    let m = StructureMatcher::default()
        .best_match(&before, &after)
        .unwrap()
        .unwrap();
    assert!(m.rmsd.value() < 1e-8, "{:?}", m.rmsd);
}

#[test]
fn the_lattice_has_its_point_operations_and_turned_crystals_are_one() {
    let cubic = PeriodicCell::orthorhombic(bohr(10.0), bohr(10.0), bohr(10.0)).unwrap();
    // The 24 rotations of the cube, the 48 operations with the reflections.
    assert_eq!(lattice_operations(&cubic, false).len(), 24);
    assert_eq!(lattice_operations(&cubic, true).len(), 48);
    let box_cell = PeriodicCell::orthorhombic(bohr(10.0), bohr(11.0), bohr(12.0)).unwrap();
    assert_eq!(lattice_operations(&box_cell, false).len(), 4);
    // An argon atom off the centre of a cubic cell and the same turned by 90 degrees about z
    // (with a partner far away, so that the pair has an orientation).
    let ar = Element::from_atomic_number(18).unwrap();
    let a = system(
        vec![ar; 2],
        &[[1.0, 0.5, 0.2], [3.0, 0.5, 0.2]],
        Some(cubic.clone()),
    );
    let b = system(
        vec![ar; 2],
        &[[-0.5, 1.0, 0.2], [-0.5, 3.0, 0.2]],
        Some(cubic.clone()),
    );
    let m = StructureMatcher::default()
        .best_match(&a, &b)
        .unwrap()
        .unwrap();
    assert!(m.rmsd.value() < 1e-8, "{:?}", m.rmsd);
}

#[test]
fn structures_in_two_cells_are_compared_at_their_fractional_coordinates() {
    let pos = dimer(rotation(0.2, 0.4, 0.1), [2.9, 0.2, 0.1]);
    let cell = |edge: f64| PeriodicCell::orthorhombic(bohr(edge), bohr(edge), bohr(edge)).unwrap();
    let a = system(waters(2), &pos, Some(cell(15.0)));
    // The same arrangement in a cell 0.1 percent longer (the atoms strained with it): the same
    // structure.
    let strained = |factor: f64| {
        let moved: Vec<[f64; 3]> = pos.iter().map(|p| p.map(|c| c * factor)).collect();
        system(waters(2), &moved, Some(cell(15.0 * factor)))
    };
    let matcher = StructureMatcher::default();
    let m = matcher.best_match(&a, &strained(1.001)).unwrap().unwrap();
    assert!(m.rmsd.value() < 1e-8, "{:?}", m.rmsd);
    // A cell 5 percent longer (0.75 Bohr) is another structure, however the atoms lie.
    assert!(matcher.best_match(&a, &strained(1.05)).unwrap().is_none());
}

#[test]
fn two_optimized_cells_of_a_nearly_cubic_lattice_are_compared_in_the_same_basis() {
    // Two argon atoms in a cell that an optimization left cubic to 1e-4 Bohr, and the same
    // structure from another optimization (the cell and the atoms within 1e-4 Bohr): every
    // operation of the cubic lattice fits the two cells about as well as the identity, and
    // the structures are one.
    let cell = |a: f64, b: f64, c: f64| {
        let v = |x: [f64; 3]| crate::domain::Vector3::<crate::domain::Bohr>::new(x);
        PeriodicCell::from_lattice_vectors(
            v([a, 3.6e-12, 1.5e-11]),
            v([3.1e-12, b, -1.1e-19]),
            v([1.3e-11, -1.1e-19, c]),
            [true, true, true],
        )
        .unwrap()
    };
    let to_angstrom = |p: [f64; 3]| p.map(|c| c * Units::BOHR_TO_ANGSTROM);
    let argon = vec![Element::from_atomic_number(18).unwrap(); 2];
    let a = system(
        argon.clone(),
        &[
            to_angstrom([0.59679, 0.82715, 1.03394]),
            to_angstrom([2.83252, 0.82715, 1.03394]),
        ],
        Some(cell(5.471425, 5.471379, 5.471379)),
    );
    let b = system(
        argon,
        &[
            to_angstrom([0.59680, 0.82717, 1.03396]),
            to_angstrom([2.83252, 0.82717, 1.03396]),
        ],
        Some(cell(5.471450, 5.471471, 5.471471)),
    );
    let m = StructureMatcher::default()
        .best_match(&a, &b)
        .unwrap()
        .unwrap();
    assert!(m.rmsd.value() < 1e-3, "{:?}", m.rmsd);
}

#[test]
fn a_crystal_turned_as_a_whole_with_its_vectors_reordered_is_the_same_structure() {
    // Two waters in an orthorhombic cell; the same crystal turned as a whole (cell and atoms)
    // with its lattice vectors named in another order (c, a, b): one structure. Strained by 5
    // percent as well, it is another.
    let pos = dimer(rotation(0.2, 0.4, 0.1), [2.9, 0.2, 0.1]);
    let edges = [15.0, 16.5, 18.0];
    let cell_a =
        PeriodicCell::orthorhombic(bohr(edges[0]), bohr(edges[1]), bohr(edges[2])).unwrap();
    let a = system(waters(2), &pos, Some(cell_a.clone()));
    let turn = rotation(0.3, -0.5, 0.7);
    let apply = |p: [f64; 3]| -> [f64; 3] {
        std::array::from_fn(|r| (0..3).map(|c| turn[r][c] * p[c]).sum())
    };
    let turned = |factor: f64| {
        let vectors: Vec<[f64; 3]> = [2, 0, 1]
            .iter()
            .map(|&k| apply(cell_a.lattice_vector(k).value().map(|c| c * factor)))
            .collect();
        let cell = PeriodicCell::from_lattice_vectors(
            crate::domain::Vector3::<crate::domain::Bohr>::new(vectors[0]),
            crate::domain::Vector3::<crate::domain::Bohr>::new(vectors[1]),
            crate::domain::Vector3::<crate::domain::Bohr>::new(vectors[2]),
            [true, true, true],
        )
        .unwrap();
        let moved: Vec<[f64; 3]> = pos
            .iter()
            .map(|p| {
                let bohr_p = p.map(|c| c * Units::ANGSTROM_TO_BOHR * factor);
                apply(bohr_p).map(|c| c * Units::BOHR_TO_ANGSTROM)
            })
            .collect();
        system(waters(2), &moved, Some(cell))
    };
    let matcher = StructureMatcher::default();
    let m = matcher.best_match(&a, &turned(1.0)).unwrap().unwrap();
    assert!(m.rmsd.value() < 1e-6, "{:?}", m.rmsd);
    assert!(!m.mirrored);
    assert!(matcher.best_match(&a, &turned(1.05)).unwrap().is_none());
}

#[test]
fn in_a_cell_images_and_a_drift_do_not_count() {
    let cell = PeriodicCell::orthorhombic(bohr(15.0), bohr(15.0), bohr(15.0)).unwrap();
    let pos = dimer(rotation(0.2, 0.4, 0.1), [2.9, 0.2, 0.1]);
    let a = system(waters(2), &pos, Some(cell.clone()));
    // The second molecule moved by a lattice vector, everything drifted, molecules swapped.
    let lattice = 15.0 * Units::BOHR_TO_ANGSTROM;
    let mut shifted: Vec<[f64; 3]> = pos
        .iter()
        .map(|p| [p[0] + 0.3, p[1] - 0.2, p[2] + 0.1])
        .collect();
    for p in &mut shifted[3..] {
        p[0] += lattice;
    }
    let order = [3, 4, 5, 0, 1, 2];
    let b_pos: Vec<[f64; 3]> = order.iter().map(|&k| shifted[k]).collect();
    let b = system(waters(2), &b_pos, Some(cell));
    let m = StructureMatcher::default()
        .best_match(&a, &b)
        .unwrap()
        .unwrap();
    assert!(m.rmsd.value() < 1e-8, "{:?}", m.rmsd);
    assert_eq!(m.permutation, vec![3, 4, 5, 0, 1, 2]);
}
