//! GFN1-xTB of mopt-rs against tblite (`tests/golden/gfn1_xtb.json`, made by
//! `tools/golden/gfn1_xtb.py`): energies, gradients, and Mulliken charges of 119 systems with
//! every element from H to Rn, closed and open shells (a doublet, a triplet, and
//! high-spin aqua complexes of V(III), Fe(II), Fe(III)), anions and cations, fractional
//! occupations (a rectangle of four hydrogen atoms, the d shells of the aqua complexes),
//! halogen bonds (with N, O, and P as acceptors), and transition-metal carbonyls, oxides,
//! halides, ammine complexes, ferrocene, a Ti-Ru hydride (whose Fock-matrix extrapolation
//! settles on another solution unless the charges are also mixed by Broyden's method) and
//! Cu2Cl2. The first 18 records come from tblite 0.4.0, the next 19 from tblite 0.8.0, which
//! agrees with 0.4.0 to 1e-14 Hartree for the light elements, and the 26 with the
//! fourth-period elements and Ru from tblite 0.4.0, as are the 56 with every element from Rb to
//! Rn other than Ru and I (all fifteen lanthanides, 4d and 5d complexes, organometallics, noble
//! gases, an At...N halogen bond, SrCl2, whose extrapolated Fock matrices settle on Sr at
//! -2.1 e unless the charges are also mixed by Broyden's method); a record with Br or I
//! from 0.8.0 (`"compare": "classical"`) is compared in its repulsion, halogen-bond and
//! dispersion energies only, since 0.8.0 moved the electronic energies of Br and I by up to
//! 2e-5 Hartree.
//!
//! tblite ends its search of the Fermi level at an electron count within about the square
//! root of the machine epsilon (`electron_count_error` of a record: 1.1e-8 electrons for
//! high-spin Fe(II) with its large smearing), and its energy moves by that times the Fermi
//! level; the energies are compared within that much more (the Fermi level is below 1
//! Hartree in magnitude).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used, clippy::expect_used)]

use faer::Col;
use mopt_rs::core_infra::json::JsonValue;
use mopt_rs::domain::{Element, MolecularSystem, SpinCharge};
use mopt_rs::pes::native::Gfn1Xtb;

fn floats(v: &JsonValue) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}

#[test]
fn gfn1_xtb_reproduces_tblite() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/golden/gfn1_xtb.json"
    );
    let doc = JsonValue::parse(&std::fs::read_to_string(path).unwrap()).unwrap();
    let mut failures = Vec::new();
    let mut worst = (0.0_f64, 0.0_f64, 0.0_f64);
    for case in doc.get("systems").unwrap().as_array().unwrap() {
        let name = case.get("name").unwrap().as_str().unwrap();
        let elements: Vec<Element> = case
            .get("symbols")
            .unwrap()
            .as_string_vec()
            .unwrap()
            .iter()
            .map(|s| Element::parse_token(s).unwrap())
            .collect();
        let x = floats(case.get("positions_bohr").unwrap());
        let uhf = case.get("uhf").unwrap().as_f64().unwrap() as u32;
        let spin = SpinCharge {
            charge: case.get("charge").unwrap().as_f64().unwrap() as i32,
            multiplicity: uhf + 1,
        };
        let sys =
            MolecularSystem::from_flat_bohr(elements, &Col::from_fn(x.len(), |i| x[i]), spin, None)
                .unwrap();
        let calc = Gfn1Xtb::default();
        let r = calc.compute(&sys).unwrap();
        if case.get("compare").and_then(|v| v.as_str()) == Some("classical") {
            let parts = case.get("energy_decomposition").unwrap();
            let part = |key: &str| parts.get(key).unwrap().as_f64().unwrap();
            let terms = [
                ("repulsion", r.energy.repulsion),
                ("halogen", r.energy.halogen_bond),
                ("dispersion", r.energy.dispersion),
            ];
            for (key, ours) in terms {
                // tblite prints these to 14 significant digits.
                if (ours - part(key)).abs() > 1e-12 + 1e-12 * ours.abs() {
                    failures.push(format!(
                        "{name}: {key} {ours:.14e}, tblite {:.14e}",
                        part(key)
                    ));
                }
            }
            assert!(r.energy.halogen_bond.abs() > 1e-5 || name != "PH3_BrCF3");
            continue;
        }
        let e_ref = case.get("energy").unwrap().as_f64().unwrap();
        let g_ref = floats(case.get("gradient").unwrap());
        let q_ref = floats(case.get("charges").unwrap());
        let de = (r.energy.total - e_ref).abs();
        let count_error = case
            .get("electron_count_error")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0)
            .abs();
        let dg = (0..g_ref.len())
            .map(|k| (r.gradient[k] - g_ref[k]).abs())
            .fold(0.0, f64::max);
        let dq = (0..q_ref.len())
            .map(|k| (r.charges[k] - q_ref[k]).abs())
            .fold(0.0, f64::max);
        worst = (worst.0.max(de), worst.1.max(dg), worst.2.max(dq));
        println!(
            "{name:20} |dE| {de:.2e} |dg| {dg:.2e} |dq| {dq:.2e} iterations {}",
            r.iterations
        );
        // tblite's effective self energies differ from E/27.21138505 by up to 5e-11 Hartree
        // (its own unit handling, which is not public): about 1e-9 Hartree in the energy.
        if de > 2.0e-9 + count_error || dg > 1.0e-9 || dq > 1.0e-8 {
            failures.push(format!(
                "{name}: |dE| = {de:.3e} (E = {:.10}, tblite {e_ref:.10}), max |dg| = {dg:.3e}, \
                 max |dq| = {dq:.3e} ({:?})",
                r.energy.total, r.energy
            ));
        }
    }
    println!(
        "worst |dE| {:.2e} Hartree, |dg| {:.2e} Hartree/Bohr, |dq| {:.2e} e",
        worst.0, worst.1, worst.2
    );
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
