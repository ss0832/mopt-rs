//! The periodic GFN-FF of mopt-rs keeps the bonds of its topology however far the molecules move
//! from where it was set up: the benzene crystal of `tests/golden/gfnff_pbc_reference.json`,
//! a little deformed, and the same moved by half a lattice vector (its atoms then on both sides
//! of the halfway point) have the same energy and gradient with the topology of the start (each
//! atom at its own nearest image would have put some atoms of a molecule a lattice vector away
//! from their neighbours).
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used, clippy::expect_used)]

use faer::Col;
use mopt_rs::core_infra::json::JsonValue;
use mopt_rs::domain::{Bohr, Element, MolecularSystem, PeriodicCell, SpinCharge, Vector3};
use mopt_rs::pes::native::gfnff::GfnffTopology;

fn floats(v: &JsonValue) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}

/// The benzene crystal of the golden data (lattice rows are the lattice vectors).
fn benzene() -> MolecularSystem {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/golden/gfnff_pbc_reference.json"
    );
    let doc = JsonValue::parse(&std::fs::read_to_string(path).unwrap()).unwrap();
    let case = doc
        .get("cases")
        .unwrap()
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c.get("name").and_then(JsonValue::as_str) == Some("benzene_crystal"))
        .unwrap()
        .clone();
    let el: Vec<Element> = case
        .get("symbols")
        .unwrap()
        .as_string_vec()
        .unwrap()
        .iter()
        .map(|s| Element::parse_token(s).unwrap())
        .collect();
    let x = floats(case.get("positions_bohr").unwrap());
    let l: Vec<Vec<f64>> = case
        .get("lattice_bohr")
        .unwrap()
        .as_array()
        .unwrap()
        .iter()
        .map(floats)
        .collect();
    let v = |k: usize| Vector3::<Bohr>::new([l[k][0], l[k][1], l[k][2]]);
    let cell = PeriodicCell::from_lattice_vectors(v(0), v(1), v(2), [true; 3]).unwrap();
    MolecularSystem::from_flat_bohr(
        el,
        &Col::from_fn(x.len(), |i| x[i]),
        SpinCharge::default(),
        Some(cell),
    )
    .unwrap()
}

#[test]
fn molecules_that_moved_by_half_a_cell_keep_their_bonds() {
    let sys = benzene();
    let topology = GfnffTopology::new(&sys).unwrap();
    // Every other atom a little up and the rest a little down along the third lattice vector
    // (+-0.001 of it), and the same moved by half of that vector, the atoms then on both sides
    // of the halfway point from where the topology was set up: by the periodicity the same
    // energy and gradient.
    let c = sys.active_cell().unwrap().lattice_vector(2).value();
    let x = sys.coords_flat_bohr();
    let moved = |t: f64| {
        let y = Col::from_fn(x.nrows(), |k| {
            let d = if (k / 3) % 2 == 0 { -0.001 } else { 0.001 };
            x[k] + (t + d) * c[k % 3]
        });
        topology
            .compute(&sys.with_flat_coords_bohr(&y).unwrap())
            .unwrap()
    };
    let (near, far) = (moved(0.0), moved(0.5));
    assert!(
        (near.energy.total - far.energy.total).abs() < 1e-9,
        "{} against {} half a cell away",
        near.energy.total,
        far.energy.total
    );
    let largest = (0..x.nrows())
        .map(|k| (near.gradient[k] - far.gradient[k]).abs())
        .fold(0.0_f64, f64::max);
    assert!(largest < 1e-8, "{largest}");
}
