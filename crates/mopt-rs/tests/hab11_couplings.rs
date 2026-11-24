//! The transfer integrals of the projection method (`kinetics::rate::transfer_integral`) with
//! the built-in GFN1-xTB and GFN0-xTB on the HAB11 set: the hole-transfer couplings of ten
//! cofacial pi dimers at 3.5, 4.0, 4.5, and 5.0 Angstrom (Kubas et al., J. Chem. Phys. 2014,
//! 140, 104105; the data of `tools/golden/hab11.py`).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used, clippy::expect_used)]

use faer::Col;
use mopt_rs::core_infra::json::JsonValue;
use mopt_rs::domain::{AtomId, Element, EvalRequest, MolecularSystem, SpinCharge, Units};
use mopt_rs::kinetics::rate::transfer_integral::{fragment_transfer_integral, Carrier};
use mopt_rs::pes::native::{Gfn0Xtb, Gfn1Xtb};
use mopt_rs::pes::Calculator;

struct Dimer {
    name: String,
    distance: f64,
    reference_mev: f64,
    system: MolecularSystem,
    molecules: [Vec<usize>; 2],
}

fn dimers() -> Vec<Dimer> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/golden/hab11.json");
    let doc = JsonValue::parse(&std::fs::read_to_string(path).unwrap()).unwrap();
    doc.get("dimers")
        .unwrap()
        .as_array()
        .unwrap()
        .iter()
        .map(|d| {
            let elements: Vec<Element> = d
                .get("symbols")
                .unwrap()
                .as_string_vec()
                .unwrap()
                .iter()
                .map(|s| Element::parse_token(s).unwrap())
                .collect();
            let flat: Vec<f64> = d
                .get("positions_angstrom")
                .unwrap()
                .as_array()
                .unwrap()
                .iter()
                .flat_map(|p| {
                    p.as_array()
                        .unwrap()
                        .iter()
                        .map(|x| x.as_f64().unwrap() * Units::ANGSTROM_TO_BOHR)
                        .collect::<Vec<f64>>()
                })
                .collect();
            let system = MolecularSystem::from_flat_bohr(
                elements,
                &Col::from_fn(flat.len(), |i| flat[i]),
                SpinCharge::new(0, 1),
                None,
            )
            .unwrap();
            let molecule = |k: usize| -> Vec<usize> {
                d.get("molecules").unwrap().as_array().unwrap()[k]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_f64().unwrap() as usize)
                    .collect()
            };
            Dimer {
                name: d.get("name").unwrap().as_str().unwrap().to_string(),
                distance: d.get("distance_angstrom").unwrap().as_f64().unwrap(),
                reference_mev: d.get("reference_mev").unwrap().as_f64().unwrap(),
                system,
                molecules: [molecule(0), molecule(1)],
            }
        })
        .collect()
}

/// |J_eff| (meV) of the hole transfer of `d` with `calc`.
fn coupling_mev(calc: &dyn Calculator, d: &Dimer) -> f64 {
    let orbitals = |sys: &MolecularSystem| calc.orbitals(sys).unwrap().expect("orbitals");
    let pair = orbitals(&d.system);
    let parts: Vec<_> = d
        .molecules
        .iter()
        .map(|atoms| {
            let ids: Vec<AtomId> = atoms.iter().map(|&a| AtomId::new(a)).collect();
            orbitals(&d.system.subsystem(&ids, SpinCharge::new(0, 1)).unwrap())
        })
        .collect();
    let ti = fragment_transfer_integral(
        &pair,
        &parts[0],
        &parts[1],
        &d.molecules[0],
        &d.molecules[1],
        Carrier::Hole,
    )
    .unwrap();
    ti.effective.abs() * Units::HARTREE_TO_EV * 1000.0
}

/// The energy (Hartree) of `d` (`atoms` `None`) or of the molecule of `atoms`, in `sc`.
fn energy(calc: &dyn Calculator, d: &Dimer, atoms: Option<&[usize]>, sc: SpinCharge) -> f64 {
    let sys = match atoms {
        None => {
            let mut s = d.system.clone();
            s.spin_charge = sc;
            s
        }
        Some(a) => {
            let ids: Vec<AtomId> = a.iter().map(|&k| AtomId::new(k)).collect();
            d.system.subsystem(&ids, sc).unwrap()
        }
    };
    calc.evaluate(&sys, &EvalRequest::ENERGY_GRAD)
        .unwrap()
        .energy
        .value()
}

/// EDEEL's coupling (meV) from the diabatic states `v11`, `v22` and the adiabatic dimer cation
/// `cation`: `V12 = sqrt((V11 - E) (V22 - E))` (at the crossing `V11 - E`, the form of
/// `kinetics::rate::electron_transfer::edeel_coupling`).
fn two_state_mev(v11: f64, v22: f64, cation: f64) -> f64 {
    ((v11 - cation) * (v22 - cation)).max(0.0).sqrt() * Units::HARTREE_TO_EV * 1000.0
}

/// EDEEL as published, for the electron that moves from the neutral molecule B to the cation
/// A: the complex without it is the dication, `V11 = E(A+ B+) - E(B+) + E(B)`,
/// `V22 = E(A+ B+) - E(A+) + E(A)`, the dication in the multiplicity `dication`.
fn edeel_electron_mev(calc: &dyn Calculator, d: &Dimer, dication: u32) -> f64 {
    let (a, b) = (&d.molecules[0][..], &d.molecules[1][..]);
    let both = energy(calc, d, None, SpinCharge::new(2, dication));
    let v11 = both - energy(calc, d, Some(b), SpinCharge::new(1, 2))
        + energy(calc, d, Some(b), SpinCharge::new(0, 1));
    let v22 = both - energy(calc, d, Some(a), SpinCharge::new(1, 2))
        + energy(calc, d, Some(a), SpinCharge::new(0, 1));
    two_state_mev(v11, v22, energy(calc, d, None, SpinCharge::new(1, 2)))
}

/// The same form for the hole: the complex without it is the neutral dimer,
/// `V11 = E(A B) - E(A) + E(A+)`, `V22 = E(A B) - E(B) + E(B+)`.
fn edeel_hole_mev(calc: &dyn Calculator, d: &Dimer) -> f64 {
    let (a, b) = (&d.molecules[0][..], &d.molecules[1][..]);
    let neutral = energy(calc, d, None, SpinCharge::new(0, 1));
    let v11 = neutral - energy(calc, d, Some(a), SpinCharge::new(0, 1))
        + energy(calc, d, Some(a), SpinCharge::new(1, 2));
    let v22 = neutral - energy(calc, d, Some(b), SpinCharge::new(0, 1))
        + energy(calc, d, Some(b), SpinCharge::new(1, 2));
    two_state_mev(v11, v22, energy(calc, d, None, SpinCharge::new(1, 2)))
}

/// The slope `a` of `ln |J| = c - a d` (1/Angstrom) by least squares.
fn decay(points: &[(f64, f64)]) -> f64 {
    let n = points.len() as f64;
    let mx = points.iter().map(|p| p.0).sum::<f64>() / n;
    let my = points.iter().map(|p| p.1.ln()).sum::<f64>() / n;
    let sxy: f64 = points.iter().map(|p| (p.0 - mx) * (p.1.ln() - my)).sum();
    let sxx: f64 = points.iter().map(|p| (p.0 - mx).powi(2)).sum();
    -sxy / sxx
}

const DISTANCES: [f64; 4] = [3.5, 4.0, 4.5, 5.0];

/// The names of the molecules of `rows`, each once.
fn names<'a>(rows: &[(&'a Dimer, f64)]) -> Vec<&'a str> {
    let mut out: Vec<&str> = rows.iter().map(|r| r.0.name.as_str()).collect();
    out.sort_unstable();
    out.dedup();
    out
}

/// The mean ratio to the reference at each distance, and the mean decay of `ln |J|` per
/// Angstrom of the couplings and of the references, of `rows` (`(dimer, coupling)`).
fn summary(rows: &[(&Dimer, f64)], label: &str) -> ([f64; 4], f64, f64) {
    let mut ratios = [0.0; 4];
    for (k, distance) in DISTANCES.iter().enumerate() {
        let at: Vec<&(&Dimer, f64)> = rows
            .iter()
            .filter(|r| (r.0.distance - distance).abs() < 1e-9)
            .collect();
        let n = at.len() as f64;
        let mue = at
            .iter()
            .map(|r| (r.1 - r.0.reference_mev).abs())
            .sum::<f64>()
            / n;
        ratios[k] = at.iter().map(|r| r.1 / r.0.reference_mev).sum::<f64>() / n;
        println!(
            "STAT {label} {distance:.1}: MUE {mue:.1} meV, mean ratio {:.3}",
            ratios[k]
        );
    }
    let molecules = names(rows);
    let (mut ours, mut theirs) = (0.0, 0.0);
    for name in &molecules {
        let of: Vec<&(&Dimer, f64)> = rows.iter().filter(|r| r.0.name == *name).collect();
        ours += decay(&of.iter().map(|r| (r.0.distance, r.1)).collect::<Vec<_>>());
        theirs += decay(
            &of.iter()
                .map(|r| (r.0.distance, r.0.reference_mev))
                .collect::<Vec<_>>(),
        );
    }
    let n = molecules.len() as f64;
    let mue = rows
        .iter()
        .map(|r| (r.1 - r.0.reference_mev).abs())
        .sum::<f64>()
        / rows.len() as f64;
    println!(
        "STAT {label} all: MUE {mue:.1} meV, decay {:.3} against {:.3} per Angstrom",
        ours / n,
        theirs / n
    );
    (ratios, ours / n, theirs / n)
}

/// Whether the couplings of each molecule of `rows` fall with the distance.
fn falls_with_distance(rows: &[(&Dimer, f64)]) -> bool {
    names(rows).iter().all(|name| {
        let mut of: Vec<&(&Dimer, f64)> = rows.iter().filter(|r| r.0.name == *name).collect();
        of.sort_by(|x, y| x.0.distance.total_cmp(&y.0.distance));
        of.windows(2).all(|w| w[1].1 < w[0].1)
    })
}

#[test]
fn hab11_hole_couplings_of_the_projection_method() {
    // Measured (docs/source/electron_transfer.md): the minimal bases of GFN1 and GFN0 give a
    // quarter and a sixth of the reference at 3.5 Angstrom and decay about twice as fast (2.66
    // and 3.40 against 1.43 per Angstrom); the bounds hold those to about 10 %.
    let set = dimers();
    assert_eq!(set.len(), 40);
    let gfn1 = Gfn1Xtb::default();
    let gfn0 = Gfn0Xtb::default();
    let cases: [(&str, &dyn Calculator, [f64; 2], [f64; 2]); 2] = [
        ("gfn1", &gfn1, [0.23, 0.29], [2.4, 2.9]),
        ("gfn0", &gfn0, [0.16, 0.20], [3.1, 3.7]),
    ];
    for (label, calc, first, slope) in cases {
        let rows: Vec<(&Dimer, f64)> = set.iter().map(|d| (d, coupling_mev(calc, d))).collect();
        for (d, j) in &rows {
            println!(
                "HAB11 {label} {:16} {:.1} {:8.1} {:8.1}",
                d.name, d.distance, d.reference_mev, j
            );
            assert!(
                *j > 0.0 && *j < d.reference_mev,
                "{label} {} {}",
                d.name,
                d.distance
            );
        }
        assert!(falls_with_distance(&rows), "{label}");
        let (ratios, ours, theirs) = summary(&rows, label);
        assert!(
            ratios[0] > first[0] && ratios[0] < first[1],
            "{label}: {ratios:?}"
        );
        assert!(
            ratios.windows(2).all(|w| w[1] < w[0]),
            "{label}: {ratios:?}"
        );
        assert!(ours > slope[0] && ours < slope[1], "{label}: decay {ours}");
        assert!(
            (theirs - 1.431).abs() < 1e-3,
            "the reference decay {theirs}"
        );
    }
}

#[test]
fn hab11_couplings_of_edeel_measure_the_delocalization_error() {
    // EDEEL's V12 = V11 - E needs the adiabatic energy E of the dimer cation, which GFN1 (like
    // any method with a delocalization error) puts too low for a hole spread over both
    // molecules. Measured on three of the molecules (docs/source/electron_transfer.md): with the
    // dication as the complex without the moving electron, V12 holds the Coulomb repulsion of
    // the two cations (above 3 eV); with the neutral dimer as the complex without the hole,
    // it is about 1 eV and grows with the distance, where the references fall from 520 to
    // 40 meV.
    let gfn1 = Gfn1Xtb::default();
    let set: Vec<Dimer> = dimers()
        .into_iter()
        .filter(|d| ["ethylene", "thiophene", "phenol"].contains(&d.name.as_str()))
        .collect();
    assert_eq!(set.len(), 12);
    let electron: Vec<(&Dimer, f64)> = set
        .iter()
        .map(|d| (d, edeel_electron_mev(&gfn1, d, 3)))
        .collect();
    let hole: Vec<(&Dimer, f64)> = set.iter().map(|d| (d, edeel_hole_mev(&gfn1, d))).collect();
    for ((d, e), (_, h)) in electron.iter().zip(&hole) {
        println!(
            "EDEEL {:16} {:.1} reference {:8.1} electron {:8.1} hole {:8.1}",
            d.name, d.distance, d.reference_mev, e, h
        );
        // e^2 / R alone is 2.9 eV at 5 Angstrom.
        assert!(*e > 3000.0, "{} {}: {e}", d.name, d.distance);
        assert!(
            *h > 800.0 && *h > d.reference_mev,
            "{} {}: {h}",
            d.name,
            d.distance
        );
    }
    summary(&electron, "edeel electron");
    let (ratios, _, _) = summary(&hole, "edeel hole");
    // The hole form grows with the distance: its ratio to the reference grows many times.
    assert!(!falls_with_distance(&hole));
    assert!(ratios[3] > 5.0 * ratios[0], "{ratios:?}");
}
