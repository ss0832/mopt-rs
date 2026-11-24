//! The rigid-body motions of a vibrational analysis: one linearity test for the projection,
//! the rotor of the thermochemistry, the number of vibrations and the point group (nearly
//! linear CO2 and HCN bent by 1e-9 to 1e-1 rad); the translations and the rotation about the
//! axis of partially periodic structures (a Lennard-Jones slab and wire); the vibrational terms
//! of the thermochemistry from 1 K to 5000 K.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used)]

use faer::Mat;
use mopt_rs::domain::{
    positions, Angstrom, Bohr, Element, EvalRequest, MolecularSystem, PeriodicCell, SpinCharge,
    Units, Vector3,
};
use mopt_rs::math::projection::build_tr_rot_basis;
use mopt_rs::pes::native::TruncatedLennardJones;
use mopt_rs::pes::Calculator;
use mopt_rs::vib::{self, FrequencySettings, NormalModes, RotorType, ThermoSettings};

type V3 = [f64; 3];

fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn norm(a: V3) -> f64 {
    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
}

/// `d (d / |d|) / d d`: `(1 - e e^T) / |d|`.
fn unit_derivative(d: V3) -> [V3; 3] {
    let r = norm(d);
    let e = [d[0] / r, d[1] / r, d[2] / r];
    std::array::from_fn(|i| {
        std::array::from_fn(|j| (f64::from(u8::from(i == j)) - e[i] * e[j]) / r)
    })
}

/// A valence model Hessian (Hartree/Bohr^2) of `sys`: springs `k e e^T` along the bonds and,
/// for every bend `(a, b, c)`, `k J^T J` of the vector `u = e_ba + e_bc` (zero at a linear
/// angle, so the bend is stiff in both directions across the line). It is exactly invariant
/// under translations, and under rotations at a linear structure.
fn valence_hessian(
    sys: &MolecularSystem,
    bonds: &[(usize, usize, f64)],
    bends: &[(usize, usize, usize, f64)],
) -> Mat<f64> {
    let x = sys.coords_flat_bohr();
    let p = |i: usize| -> V3 { [x[3 * i], x[3 * i + 1], x[3 * i + 2]] };
    let n = sys.dof();
    let mut h = Mat::<f64>::zeros(n, n);
    for &(a, b, k) in bonds {
        let d = sub(p(a), p(b));
        let r = norm(d);
        let e = [d[0] / r, d[1] / r, d[2] / r];
        for i in 0..3 {
            for j in 0..3 {
                let v = k * e[i] * e[j];
                h[(3 * a + i, 3 * a + j)] += v;
                h[(3 * b + i, 3 * b + j)] += v;
                h[(3 * a + i, 3 * b + j)] -= v;
                h[(3 * b + i, 3 * a + j)] -= v;
            }
        }
    }
    for &(a, b, c, k) in bends {
        let (da, dc) = (
            unit_derivative(sub(p(a), p(b))),
            unit_derivative(sub(p(c), p(b))),
        );
        // J: 3 x 3N, the rows of u.
        let mut jac = Mat::<f64>::zeros(3, n);
        for i in 0..3 {
            for j in 0..3 {
                jac[(i, 3 * a + j)] += da[i][j];
                jac[(i, 3 * c + j)] += dc[i][j];
                jac[(i, 3 * b + j)] -= da[i][j] + dc[i][j];
            }
        }
        for r in 0..n {
            for s in 0..n {
                h[(r, s)] += k * (0..3).map(|i| jac[(i, r)] * jac[(i, s)]).sum::<f64>();
            }
        }
    }
    h
}

/// The axis `(1, 2, 3) / sqrt(14)` and two directions across it, for a structure that lies
/// along no Cartesian axis.
fn oblique(p: V3) -> V3 {
    let w = [1.0 / 14f64.sqrt(), 2.0 / 14f64.sqrt(), 3.0 / 14f64.sqrt()];
    let u0 = [1.0, 0.0, 0.0];
    let along = w[0];
    let u1 = sub(u0, [along * w[0], along * w[1], along * w[2]]);
    let l = norm(u1);
    let u = [u1[0] / l, u1[1] / l, u1[2] / l];
    let v = [
        w[1] * u[2] - w[2] * u[1],
        w[2] * u[0] - w[0] * u[2],
        w[0] * u[1] - w[1] * u[0],
    ];
    // p = (across, across, along) in the frame (u, v, w).
    std::array::from_fn(|c| p[0] * u[c] + p[1] * v[c] + p[2] * w[c])
}

fn system(elements: Vec<Element>, xyz: &[V3]) -> MolecularSystem {
    MolecularSystem::new(
        elements,
        &positions::<Angstrom>(xyz),
        SpinCharge::default(),
        None,
    )
    .unwrap()
}

/// CO2 bent by `alpha` at the carbon (O-C 1.16 Angstrom).
fn co2(alpha: f64, turn: bool) -> MolecularSystem {
    let l = 1.16;
    let (s, c) = ((0.5 * alpha).sin(), (0.5 * alpha).cos());
    let place = |p: V3| if turn { oblique(p) } else { p };
    system(
        vec![Element::O, Element::C, Element::O],
        &[
            place([l * s, 0.0, -l * c]),
            place([0.0; 3]),
            place([l * s, 0.0, l * c]),
        ],
    )
}

/// HCN bent by `alpha` at the carbon (H-C 1.065, C-N 1.153 Angstrom).
fn hcn(alpha: f64, turn: bool) -> MolecularSystem {
    let place = |p: V3| if turn { oblique(p) } else { p };
    system(
        vec![Element::H, Element::C, Element::N],
        &[
            place([1.065 * alpha.sin(), 0.0, -1.065 * alpha.cos()]),
            place([0.0; 3]),
            place([0.0, 0.0, 1.153]),
        ],
    )
}

/// What the parts of a vibrational analysis say about a structure.
#[derive(Debug, PartialEq)]
struct Verdict {
    projected: usize,
    rotor: RotorType,
    linear_group: bool,
    vibrations: usize,
}

fn verdict(sys: &MolecularSystem, hessian: &Mat<f64>) -> (Verdict, f64, String) {
    let analysis = vib::analyze(sys, hessian, 0.0, &FrequencySettings::default()).unwrap();
    let label = analysis.point_group.label.clone();
    (
        Verdict {
            projected: build_tr_rot_basis(sys, true).ncols(),
            rotor: analysis.thermo.rotor,
            linear_group: label.contains('\u{221e}'),
            vibrations: analysis.thermo.vibrations,
        },
        analysis.thermo.zero_point_energy,
        label,
    )
}

#[test]
fn projection_rotor_vibrations_and_point_group_agree_on_linearity() {
    let angles = [
        0.0, 1e-9, 1e-8, 1e-7, 3e-7, 1e-6, 1e-5, 1e-4, 1e-3, 2e-3, 5e-3, 1e-2, 3e-2, 5e-2, 1e-1,
    ];
    let mut failures = Vec::new();
    for (name, make, bonds, bends) in [
        (
            "CO2",
            co2 as fn(f64, bool) -> MolecularSystem,
            vec![(0, 1, 0.8), (1, 2, 0.8)],
            vec![(0, 1, 2, 0.15)],
        ),
        (
            "HCN",
            hcn as fn(f64, bool) -> MolecularSystem,
            vec![(0, 1, 0.4), (1, 2, 1.0)],
            vec![(0, 1, 2, 0.05)],
        ),
    ] {
        for turn in [false, true] {
            let mut linear_zpe = None;
            for &alpha in &angles {
                let sys = make(alpha, turn);
                let h = valence_hessian(&sys, &bonds, &bends);
                let (found, zpe, label) = verdict(&sys, &h);
                let linear = found.rotor == RotorType::Linear;
                linear_zpe.get_or_insert(zpe);
                let lost = (linear_zpe.unwrap_or(zpe) - zpe) * Units::HARTREE_TO_KCAL_MOL;
                eprintln!(
                    "{name} turned={turn} alpha={alpha:e}: projected {} rotor {:?} group {label} \
                     vibrations {} zpe change {lost:.3} kcal/mol",
                    found.projected, found.rotor, found.vibrations
                );
                let n = sys.dof();
                let agree = (found.projected == 5) == linear
                    && found.linear_group == linear
                    && found.vibrations == n - found.projected
                    && found.projected == if linear { 5 } else { 6 };
                if !agree {
                    failures.push(format!("{name} turned={turn} alpha={alpha:e}: {found:?}"));
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "disagreements:\n{}",
        failures.join("\n")
    );
    // Exactly linear and clearly bent molecules keep their usual counts.
    let (straight, _, label) = verdict(
        &co2(0.0, false),
        &valence_hessian(
            &co2(0.0, false),
            &[(0, 1, 0.8), (1, 2, 0.8)],
            &[(0, 1, 2, 0.15)],
        ),
    );
    assert_eq!((straight.projected, straight.vibrations), (5, 4));
    assert_eq!(label, "D\u{221e}h");
    let bent = co2(0.3, false);
    let (bent, _, label) = verdict(
        &bent,
        &valence_hessian(&bent, &[(0, 1, 0.8), (1, 2, 0.8)], &[(0, 1, 2, 0.15)]),
    );
    assert_eq!((bent.projected, bent.vibrations), (6, 3));
    assert_eq!(label, "C2v");
}

/// A cell periodic along the axes `periodic` with the edges `edges` (Angstrom, orthorhombic).
fn cell(edges: V3, periodic: [bool; 3]) -> PeriodicCell {
    PeriodicCell::from_lattice_vectors(
        Vector3::<Angstrom>::new([edges[0], 0.0, 0.0]),
        Vector3::<Angstrom>::new([0.0, edges[1], 0.0]),
        Vector3::<Angstrom>::new([0.0, 0.0, edges[2]]),
        periodic,
    )
    .unwrap()
}

/// The transverse modes (the ones left by the projection) of `sys` with the truncated
/// Lennard-Jones Hessian.
fn periodic_modes(sys: &MolecularSystem) -> (usize, NormalModes) {
    let calc =
        TruncatedLennardJones::new(mopt_rs::domain::Quantity::<Bohr>::new(15.0), None).unwrap();
    let out = calc.evaluate(sys, &EvalRequest::ENERGY_GRAD_HESS).unwrap();
    let hessian = out.hessian_bohr.unwrap();
    (
        build_tr_rot_basis(sys, true).ncols(),
        vib::transverse_modes(sys, &hessian, true, &[]).unwrap(),
    )
}

#[test]
fn slabs_and_wires_lose_all_translations_and_a_wire_its_rotation() {
    // An argon slab: a square layer (3.8 Angstrom) with a second one over its hollows,
    // periodic in x and y; 2 atoms per cell.
    let d = 3.8;
    let slab = MolecularSystem::new(
        vec![Element::parse_token("Ar").unwrap(); 2],
        &positions::<Angstrom>(&[[0.0, 0.0, 5.0], [0.5 * d, 0.5 * d, 5.0 + d / 2f64.sqrt()]]),
        SpinCharge::default(),
        Some(cell([d, d, 30.0], [true, true, false])),
    )
    .unwrap();
    let (projected, modes) = periodic_modes(&slab);
    eprintln!("slab: projected {projected}, modes {:?}", modes.wavenumbers);
    // Three translations (none of the rotations: they turn the lattice).
    assert_eq!(projected, 3);
    assert_eq!(modes.wavenumbers.len(), slab.dof() - 3);
    assert!(
        modes.wavenumbers.iter().all(|w| w.abs() > 1.0),
        "{:?}",
        modes.wavenumbers
    );
    // An argon wire along x: three atoms per cell off the axis (not in one plane).
    let wire = MolecularSystem::new(
        vec![Element::parse_token("Ar").unwrap(); 3],
        &positions::<Angstrom>(&[
            [0.0, 10.0, 10.0],
            [0.33 * 7.0, 13.2, 10.0],
            [0.66 * 7.0, 11.0, 12.6],
        ]),
        SpinCharge::default(),
        Some(cell([7.0, 30.0, 30.0], [true, false, false])),
    )
    .unwrap();
    let (projected, modes) = periodic_modes(&wire);
    eprintln!("wire: projected {projected}, modes {:?}", modes.wavenumbers);
    // Three translations and the rotation about the axis of the wire.
    assert_eq!(projected, 4);
    assert_eq!(modes.wavenumbers.len(), wire.dof() - 4);
    assert!(
        modes.wavenumbers.iter().all(|w| w.abs() > 1.0),
        "{:?}",
        modes.wavenumbers
    );
    // A crystal keeps its three translations.
    let mut crystal = slab.clone();
    crystal.cell = Some(cell([d, d, d * 2f64.sqrt()], [true; 3]));
    assert_eq!(build_tr_rot_basis(&crystal, true).ncols(), 3);
}

#[test]
fn the_vibrational_heat_capacity_stays_finite_at_low_temperatures() {
    // A mode of 4000 cm^-1 at 8 K: theta / T = 719, where (x / expm1 x)^2 e^x is 0 * inf.
    let water = system(
        vec![Element::O, Element::H, Element::H],
        &[[0.0, 0.0, 0.12], [0.0, 0.76, -0.47], [0.0, -0.76, -0.47]],
    );
    let wavenumbers = [1600.0, 3700.0, 4000.0];
    let to_theta = Units::HC_OVER_KB_CM_K;
    let modes = NormalModes {
        wavenumbers: wavenumbers.to_vec(),
        curvatures: vec![1.0; 3],
        displacements: Vec::new(),
        reduced_masses: Vec::new(),
        force_constants: Vec::new(),
        temperatures: wavenumbers.iter().map(|w| w * to_theta).collect(),
    };
    for temperature in [1.0, 8.0, 298.15, 5000.0] {
        let settings = ThermoSettings {
            temperature,
            ..ThermoSettings::default()
        };
        let out = vib::thermochemistry(&water, &modes, -76.0, &settings).unwrap();
        // The exact heat capacity of the three oscillators (units of k): x^2 e^-x / (1 -
        // e^-x)^2, zero in doubles below 8 K for the stiff ones.
        let exact: f64 = wavenumbers
            .iter()
            .map(|w| {
                let x = w * to_theta / temperature;
                let gap = -(-x).exp_m1();
                if x > 700.0 {
                    (2.0 * x.ln() - x - 2.0 * gap.ln()).exp()
                } else {
                    x * x * (-x).exp() / (gap * gap)
                }
            })
            .sum();
        let k = Units::BOLTZMANN_HARTREE_PER_K;
        assert!(
            out.cv.vibrational.is_finite()
                && (out.cv.vibrational / k - exact).abs() <= 1e-12 * exact.max(1e-300),
            "{temperature} K: {} against {exact}",
            out.cv.vibrational / k
        );
        assert!(out.entropy.vibrational.is_finite() && out.gibbs.total.is_finite());
    }
}
