//! Tests of the modified embedded-atom method.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used)]

use faer::Col;

use super::super::neighbors::neighbor_list;
use super::parameters::TABLE;
use super::*;
use crate::domain::{positions, Angstrom, PeriodicCell, SpinCharge, Vector3};

/// eV / Angstrom^3 in GPa.
const GPA: f64 = Units::GPA_PER_EV_ANGSTROM3;

fn element(symbol: &str) -> Element {
    Element::parse_token(symbol).unwrap()
}

/// A crystal of `symbol` with the nearest-neighbour distance `d` (Angstrom), deformed by
/// the strain `1 + strain`: `fcc` (four atoms), `bcc` (two), `dia` (eight), or `hcp` with
/// the ideal axial ratio (two).
fn strained(symbol: &str, lattice: &str, d: f64, strain: [[f64; 3]; 3]) -> MolecularSystem {
    let cubic = |a: f64| [[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]];
    let face = [
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
    ];
    let (vectors, fractions): ([[f64; 3]; 3], Vec<[f64; 3]>) = match lattice {
        "fcc" => (cubic(d * 2.0_f64.sqrt()), face.to_vec()),
        "bcc" => (
            cubic(2.0 * d / 3.0_f64.sqrt()),
            vec![[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        ),
        "dia" => {
            let mut all = face.to_vec();
            all.extend(face.iter().map(|f| [f[0] + 0.25, f[1] + 0.25, f[2] + 0.25]));
            (cubic(4.0 * d / 3.0_f64.sqrt()), all)
        }
        _ => (
            [
                [d, 0.0, 0.0],
                [-0.5 * d, 0.5 * 3.0_f64.sqrt() * d, 0.0],
                [0.0, 0.0, d * (8.0_f64 / 3.0).sqrt()],
            ],
            vec![[0.0, 0.0, 0.0], [1.0 / 3.0, 2.0 / 3.0, 0.5]],
        ),
    };
    let deform = |v: [f64; 3]| -> [f64; 3] {
        std::array::from_fn(|p| v[p] + (0..3).map(|q| strain[p][q] * v[q]).sum::<f64>())
    };
    let vectors = vectors.map(deform);
    let cell = PeriodicCell::from_lattice_vectors(
        Vector3::<Angstrom>::new(vectors[0]),
        Vector3::<Angstrom>::new(vectors[1]),
        Vector3::<Angstrom>::new(vectors[2]),
        [true, true, true],
    )
    .unwrap();
    let points: Vec<[f64; 3]> = fractions
        .iter()
        .map(|f| {
            std::array::from_fn(|c| {
                f[0] * vectors[0][c] + f[1] * vectors[1][c] + f[2] * vectors[2][c]
            })
        })
        .collect();
    MolecularSystem::new(
        vec![element(symbol); points.len()],
        &positions::<Angstrom>(&points),
        SpinCharge::default(),
        Some(cell),
    )
    .unwrap()
}

fn crystal(symbol: &str, lattice: &str, d: f64) -> MolecularSystem {
    strained(symbol, lattice, d, [[0.0; 3]; 3])
}

/// Energy in eV.
fn energy_with(calc: &Meam, sys: &MolecularSystem) -> f64 {
    calc.evaluate(sys, &EvalRequest::ENERGY_ONLY)
        .unwrap()
        .energy
        .value()
        / Units::EV_TO_HARTREE
}

fn energy_per_atom(sys: &MolecularSystem) -> f64 {
    energy_with(&Meam::library(&sys.elements).unwrap(), sys) / sys.n_atoms() as f64
}

fn library_entry(symbol: &str) -> (&'static str, MeamElement) {
    let lattice = TABLE.iter().find(|(s, ..)| *s == symbol).unwrap().1;
    (lattice, MeamElement::library(element(symbol)).unwrap())
}

#[test]
fn the_reference_structures_have_the_cohesive_energies_of_the_library() {
    assert_eq!(TABLE.len(), ELEMENTS.len());
    for symbol in ELEMENTS {
        let (lattice, p) = library_entry(symbol);
        let at = |scale: f64| energy_per_atom(&crystal(symbol, lattice, p.re * scale));
        assert!(
            (at(1.0) + p.ec).abs() < 1e-10,
            "{symbol}: {} eV/atom against -{}",
            at(1.0),
            p.ec
        );
        // The universal equation of state around its minimum at r_e.
        for scale in [0.9, 0.99, 1.01, 1.05] {
            let a = p.alpha * (scale - 1.0);
            let expected = -p.ec * (1.0 + a) * (-a).exp();
            assert!(
                (at(scale) - expected).abs() < 1e-10,
                "{symbol} at {scale}: {} against {expected}",
                at(scale)
            );
        }
    }
}

/// Elastic constants `C11`, `C12`, `C44` (GPa) of a cubic crystal at its equilibrium from
/// second differences of the energy, each from the strain `+e` and from `-e`.
fn elastic_constants(symbol: &str, lattice: &str, d: f64, calc: &Meam) -> [[f64; 2]; 3] {
    let e = 2e-3;
    let base = crystal(symbol, lattice, d);
    let volume = base.active_cell().unwrap().volume_bohr3() * Units::BOHR_TO_ANGSTROM.powi(3);
    let e0 = energy_with(calc, &base);
    let curvature = |pattern: [[f64; 3]; 3]| -> [f64; 2] {
        [1.0, -1.0].map(|sign| {
            let strain = pattern.map(|row| row.map(|x| x * sign * e));
            let energy = energy_with(calc, &strained(symbol, lattice, d, strain));
            2.0 * (energy - e0) / (volume * e * e) * GPA
        })
    };
    let uniaxial = curvature([[1.0, 0.0, 0.0], [0.0; 3], [0.0; 3]]);
    let biaxial = curvature([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0; 3]]);
    let shear = curvature([[0.0, 0.5, 0.0], [0.5, 0.0, 0.0], [0.0; 3]]);
    [
        uniaxial,
        [0, 1].map(|s| 0.5 * biaxial[s] - uniaxial[s]),
        shear,
    ]
}

#[test]
fn elastic_constants_of_cubic_metals_are_those_the_parameters_were_fitted_to() {
    // Measured shear constants at room temperature (GPa): (C11 - C12) / 2 and C44.
    let measured = [
        ("Cu", 23.5, 75.4),
        ("Ag", 15.3, 46.1),
        ("Au", 14.5, 42.0),
        ("Ni", 49.6, 124.7),
        ("Pt", 48.0, 76.5),
        ("Al", 23.0, 28.5),
        ("Pb", 3.6, 14.9),
        ("Fe", 47.5, 117.0),
        ("Mo", 142.0, 110.0),
        ("W", 159.0, 160.6),
        ("Cr", 141.0, 100.8),
        ("Ta", 52.9, 82.5),
        ("Nb", 56.0, 28.7),
        ("V", 54.7, 42.6),
        ("K", 0.28, 1.88),
    ];
    for (symbol, shear, c44) in measured {
        let (lattice, p) = library_entry(symbol);
        let calc = Meam::library(&[element(symbol)]).unwrap();
        let c =
            elastic_constants(symbol, lattice, p.re, &calc).map(|pair| 0.5 * (pair[0] + pair[1]));
        // The bulk modulus is that of the equation of state, alpha^2 E_c / (9 Omega).
        let volume = if lattice == "fcc" {
            p.re.powi(3) / 2.0_f64.sqrt()
        } else {
            4.0 * p.re.powi(3) / (3.0 * 3.0_f64.sqrt())
        };
        let bulk = p.alpha * p.alpha * p.ec / (9.0 * volume) * GPA;
        let found = (c[0] + 2.0 * c[1]) / 3.0;
        assert!(
            (found / bulk - 1.0).abs() < 5e-3,
            "{symbol}: B {found} against {bulk}"
        );
        let found = 0.5 * (c[0] - c[1]);
        assert!(
            (found / shear - 1.0).abs() < 0.05,
            "{symbol}: C' {found} against {shear}"
        );
        assert!(
            (c[2] / c44 - 1.0).abs() < 0.05,
            "{symbol}: C44 {} against {c44}",
            c[2]
        );
    }
}

/// The table of the elastic constants of the cubic metals of the library (`cargo test -p
/// mopt-rs --lib table_of_elastic_constants -- --nocapture`).
#[test]
fn table_of_elastic_constants() {
    for symbol in ELEMENTS {
        let (lattice, p) = library_entry(symbol);
        if lattice == "hcp" {
            continue;
        }
        let calc = Meam::library(&[element(symbol)]).unwrap();
        let c = elastic_constants(symbol, lattice, p.re, &calc);
        assert!(c.iter().flatten().all(|v| v.is_finite()), "{symbol}: {c:?}");
        println!(
            "{symbol:2} {lattice} cutoff {:.3}  C11 {:8.2} {:8.2}  C12 {:8.2} {:8.2}  C44 {:8.2} {:8.2}",
            calc.cutoff_angstrom(),
            c[0][0],
            c[0][1],
            c[1][0],
            c[1][1],
            c[2][0],
            c[2][1]
        );
    }
    // The cutoff of potassium.
    let (lattice, p) = library_entry("K");
    for factor in [1.1, 1.17, 1.25, 1.5, 2.0] {
        let calc = Meam::library(&[element("K")])
            .unwrap()
            .with_cutoff(factor * p.re)
            .unwrap();
        let c = elastic_constants("K", lattice, p.re, &calc);
        let e = energy_with(&calc, &crystal("K", lattice, p.re)) / 2.0;
        println!(
            "K cutoff {factor} r_e: E {e:.6}  C11 {:.3} {:.3}  C12 {:.3} {:.3}  C44 {:.3} {:.3}",
            c[0][0], c[0][1], c[1][0], c[1][1], c[2][0], c[2][1]
        );
    }
}

/// Differences of the energy and of the gradient (five points: the cutoff function is
/// steep) against the analytic gradient and Hessian.
fn check_derivatives(calc: &Meam, sys: &MolecularSystem) {
    let out = calc.evaluate(sys, &EvalRequest::ENERGY_GRAD_HESS).unwrap();
    let hessian = out.hessian_bohr.unwrap();
    let plain = calc.evaluate(sys, &EvalRequest::ENERGY_GRAD).unwrap();
    let x0 = sys.coords_flat_bohr();
    let n = x0.nrows();
    let h = 2e-4;
    let scale = |value: f64| value.abs().max(1.0);
    for k in 0..n {
        // The gradient is the same with and without the Hessian.
        assert!((plain.gradient_bohr[k] - out.gradient_bohr[k]).abs() < 1e-13);
        let at = |steps: f64| {
            let moved = Col::from_fn(n, |i| x0[i] + if i == k { steps * h } else { 0.0 });
            calc.evaluate(
                &sys.with_flat_coords_bohr(&moved).unwrap(),
                &EvalRequest::ENERGY_GRAD,
            )
            .unwrap()
        };
        let (far_minus, minus, plus, far_plus) = (at(-2.0), at(-1.0), at(1.0), at(2.0));
        let five = |f: &dyn Fn(&EvalOutput) -> f64| {
            (f(&far_minus) - 8.0 * f(&minus) + 8.0 * f(&plus) - f(&far_plus)) / (12.0 * h)
        };
        let slope = five(&|o| o.energy.value());
        assert!(
            (slope - out.gradient_bohr[k]).abs() < 1e-9 * scale(slope),
            "gradient {k}: {slope} against {}",
            out.gradient_bohr[k]
        );
        for i in 0..n {
            let second = five(&|o| o.gradient_bohr[i]);
            assert!(
                (second - hessian[(i, k)]).abs() < 1e-8 * scale(second),
                "Hessian ({i}, {k}): {second} against {}",
                hessian[(i, k)]
            );
            assert!((hessian[(i, k)] - hessian[(k, i)]).abs() < 1e-12 * scale(second));
        }
    }
}

/// Pairs within the cutoff by their screening: (not screened, screened in part, screened
/// completely).
fn screening_census(calc: &Meam, sys: &MolecularSystem) -> (usize, usize, usize) {
    let reach = calc.cutoff * calc.ellipse_bound().sqrt();
    let nl = neighbor_list(sys, Quantity::<Bohr>::new(reach / Units::BOHR_TO_ANGSTROM));
    let mut census = (0, 0, 0);
    for nbs in &nl {
        let sites: Vec<[f64; 3]> = nbs
            .iter()
            .map(|nb| nb.vector.map(|c| c * Units::BOHR_TO_ANGSTROM))
            .collect();
        for (j, nb) in nbs.iter().enumerate() {
            if nb.distance * Units::BOHR_TO_ANGSTROM >= calc.cutoff {
                continue;
            }
            match calc.screening(&sites, j) {
                None => census.2 += 1,
                Some(k) if k.is_empty() => census.0 += 1,
                Some(_) => census.1 += 1,
            }
        }
    }
    census
}

/// The atoms moved off their sites by up to `size` Bohr.
fn shaken(sys: &MolecularSystem, size: f64) -> MolecularSystem {
    let x = sys.coords_flat_bohr();
    let moved = Col::from_fn(x.nrows(), |i| {
        x[i] + size * (((i * 7 + 3) % 5) as f64 - 2.0) / 2.0
    });
    sys.with_flat_coords_bohr(&moved).unwrap()
}

fn cluster(symbols: &[&str], points: &[[f64; 3]]) -> MolecularSystem {
    MolecularSystem::new(
        symbols.iter().map(|s| element(s)).collect(),
        &positions::<Angstrom>(points),
        SpinCharge::default(),
        None,
    )
    .unwrap()
}

#[test]
fn gradient_and_hessian_are_exact_for_a_cluster_of_several_elements() {
    // Seven atoms of five elements (square-root and exponential forms of G, fcc, bcc, and
    // diamond references), with pairs screened in part and completely.
    let sys = cluster(
        &["Cu", "Ni", "Fe", "Si", "Al", "Cu", "Ni"],
        &[
            [0.0, 0.0, 0.0],
            [2.5, 0.1, 0.0],
            [1.2, 2.2, 0.2],
            [1.3, 0.8, 2.1],
            [3.6, 2.1, 1.4],
            [-1.1, 1.9, 2.0],
            [2.3, -1.6, 2.2],
        ],
    );
    let calc = Meam::library(&sys.elements).unwrap();
    let census = screening_census(&calc, &sys);
    assert!(census.0 > 0 && census.1 > 0 && census.2 > 0, "{census:?}");
    check_derivatives(&calc, &sys);
    // The same with a given reference for one pair and without the nuclear repulsion.
    let given = calc
        .clone()
        .with_pair(
            element("Ni"),
            element("Al"),
            MeamPair {
                lattice: Lattice::B2,
                ec: 4.5,
                re: 2.5,
                alpha: 4.8,
            },
        )
        .unwrap()
        .with_nuclear_repulsion(false);
    assert!((energy_with(&given, &sys) - energy_with(&calc, &sys)).abs() > 1e-3);
    check_derivatives(&given, &sys);
}

#[test]
fn gradient_and_hessian_are_exact_in_periodic_cells() {
    // fcc with an impurity, bcc (second neighbours at the limit of the screening), diamond
    // with the exponential form, and hcp, all with the atoms off their sites.
    let mut alloy = crystal("Cu", "fcc", 2.56);
    alloy.elements[1] = element("Au");
    for (sys, size) in [
        (alloy, 0.15),
        (crystal("Fe", "bcc", 2.48), 0.12),
        (crystal("Si", "dia", 2.35), 0.1),
        (crystal("Mg", "hcp", 3.2), 0.15),
    ] {
        let sys = shaken(&sys, size);
        let calc = Meam::library(&sys.elements).unwrap();
        check_derivatives(&calc, &sys);
    }
    // Iron pressed along one axis: the second neighbours along it, screened completely in
    // the cubic crystal (C = C_min), are screened in part (C = 2 + 4 * 0.05).
    let pressed = strained("Fe", "bcc", 2.48, [[-0.05, 0.0, 0.0], [0.0; 3], [0.0; 3]]);
    let iron = shaken(&pressed, 0.01);
    let calc = Meam::library(&iron.elements).unwrap();
    let cubic = screening_census(&calc, &crystal("Fe", "bcc", 2.48));
    assert_eq!(cubic, (16, 0, 12));
    let census = screening_census(&calc, &iron);
    assert_eq!((census.0, census.1), (16, 4), "{census:?}");
    check_derivatives(&calc, &iron);
    // The same with the infinitely smooth step in place of the cutoff function: other
    // energies where pairs are screened in part, the same in the cubic crystal.
    let smooth = calc.clone().with_smooth_cutoff(true);
    check_derivatives(&smooth, &iron);
    assert!((energy_with(&smooth, &iron) - energy_with(&calc, &iron)).abs() > 1e-6);
    let cubic = crystal("Fe", "bcc", 2.48);
    assert!((energy_with(&smooth, &cubic) - energy_with(&calc, &cubic)).abs() < 1e-12);
    assert!(smooth.description().contains("infinitely smooth"));
}

#[test]
fn close_atoms_feel_the_repulsion_of_the_nuclei() {
    let copper = MeamElement::library(element("Cu")).unwrap();
    let calc = Meam::library(&[element("Cu")]).unwrap();
    let pair = |r: f64| -> f64 { calc.pair_potential(0, 0, r) };
    // alpha (r / r_e - 1) = -3 and -1.
    let inner = copper.re * (1.0 - 3.0 / copper.alpha);
    let outer = copper.re * (1.0 - 1.0 / copper.alpha);
    let nuclear = |r: f64| -> f64 { functions::zbl(29.0, 29.0, r) };
    assert!((pair(inner * 0.9) - nuclear(inner * 0.9)).abs() < 1e-12);
    assert!((pair(inner + 1e-9) - nuclear(inner)).abs() < 1e-5);
    let plain = calc.clone().with_nuclear_repulsion(false);
    let unblended = |r: f64| -> f64 { plain.pair_potential(0, 0, r) };
    assert!((pair(outer - 1e-9) - unblended(outer)).abs() < 1e-6);
    assert_eq!(pair(outer + 0.3), unblended(outer + 0.3));
    // Two atoms pressed together: derivatives through the blend.
    let close = cluster(
        &["Cu", "Cu"],
        &[[0.0, 0.0, 0.0], [0.5 * (inner + outer), 0.0, 0.0]],
    );
    check_derivatives(&calc, &close);
    assert!(energy_with(&calc, &close) > 0.0);
}

#[test]
fn settings_are_checked_and_unknown_elements_refused() {
    let atom = cluster(&["W"], &[[0.0, 0.0, 0.0]]);
    assert_eq!(energy_per_atom(&atom), 0.0);
    assert!(Meam::library(&[element("Cu"), element("Zn")]).is_err());
    let copper = Meam::library(&[element("Cu")]).unwrap();
    let silver = cluster(&["Ag", "Ag"], &[[0.0, 0.0, 0.0], [2.9, 0.0, 0.0]]);
    assert!(copper.evaluate(&silver, &EvalRequest::ENERGY_GRAD).is_err());
    assert!(copper.clone().with_cutoff(-1.0).is_err());
    assert!(copper.clone().with_screening(2.8, 2.0).is_err());
    assert!(copper
        .clone()
        .with_pair(
            element("Cu"),
            element("Cu"),
            MeamPair {
                lattice: Lattice::B1,
                ec: 1.0,
                re: 2.0,
                alpha: 5.0
            }
        )
        .is_err());
    let mut twice = vec![MeamElement::library(element("Cu")).unwrap(); 2];
    assert!(Meam::new(twice.clone()).is_err());
    twice.truncate(1);
    twice[0].ibar = 2;
    assert!(Meam::new(twice).is_err());
    // The cutoff follows the largest atom.
    assert_eq!(copper.cutoff_angstrom(), DEFAULT_CUTOFF_ANGSTROM);
    let potassium = Meam::library(&[element("K")]).unwrap();
    let re = MeamElement::library(element("K")).unwrap().re;
    assert!((potassium.cutoff_angstrom() - CUTOFF_PER_DISTANCE * re).abs() < 1e-12);
    // A wider screening window lets second neighbours in: the energy changes.
    let open = copper.clone().with_screening(0.0, 2.8).unwrap();
    let cell = crystal("Cu", "fcc", 2.56);
    assert!((energy_with(&open, &cell) - energy_with(&copper, &cell)).abs() > 1e-3);
    assert!(copper.description().contains("Cu") && !copper.description().contains("not fitted"));
    let both = Meam::library(&[element("Cu"), element("Ag")]).unwrap();
    assert!(both.description().contains("Ag-Cu") || both.description().contains("Cu-Ag"));
}
