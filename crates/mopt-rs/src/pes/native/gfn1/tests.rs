//! Tests of GFN1-xTB: the gradient against central differences of the energy (closed shell,
//! open shell, halogen bond, the elements to Ar), the start of the charges (SAD, the last
//! structure), the physical solution of an anion where tblite runs away, the refused
//! elements.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used, clippy::expect_used)]

use faer::Col;

use super::*;
use crate::domain::{Element, SpinCharge};

fn system(symbols: &[&str], x: &[f64], charge: i32, multiplicity: u32) -> MolecularSystem {
    let el: Vec<Element> = symbols
        .iter()
        .map(|s| Element::parse_token(s).unwrap())
        .collect();
    MolecularSystem::from_flat_bohr(
        el,
        &Col::from_fn(x.len(), |i| x[i]),
        SpinCharge {
            charge,
            multiplicity,
        },
        None,
    )
    .unwrap()
}

fn check_gradient(sys: &MolecularSystem) {
    let calc = Gfn1Xtb::default();
    let r = calc.compute(sys).unwrap();
    let x0 = sys.coords_flat_bohr();
    let h = 1.0e-4;
    let mut worst = 0.0_f64;
    for k in 0..x0.nrows() {
        let mut xp = x0.clone();
        xp[k] += h;
        let mut xm = x0.clone();
        xm[k] -= h;
        let ep = calc
            .compute(&sys.with_flat_coords_bohr(&xp).unwrap())
            .unwrap()
            .energy
            .total;
        let em = calc
            .compute(&sys.with_flat_coords_bohr(&xm).unwrap())
            .unwrap()
            .energy
            .total;
        let fd = (ep - em) / (2.0 * h);
        worst = worst.max((fd - r.gradient[k]).abs());
    }
    assert!(worst < 2.0e-7, "max |analytic - difference| = {worst:.3e}");
}

#[test]
fn gradient_of_a_closed_shell_matches_differences() {
    // Formaldehyde with a thiol (S d shells), slightly distorted.
    let sys = system(
        &["C", "O", "H", "H", "S", "H"],
        &[
            0.0, 0.0, 0.0, 0.0, 0.1, 2.28, 1.77, 0.0, -1.05, -1.75, 0.2, -1.1, 0.3, 4.9, -3.0, 1.2,
            5.6, -1.0,
        ],
        0,
        1,
    );
    check_gradient(&sys);
}

#[test]
fn gradient_of_an_open_shell_and_a_halogen_bond_matches_differences() {
    // NH2 radical.
    let nh2 = system(
        &["N", "H", "H"],
        &[0.0, 0.0, 0.2, 0.0, 1.5, -0.9, 0.1, -1.6, -0.8],
        0,
        2,
    );
    check_gradient(&nh2);
    // Ammonia and bromine: a halogen bond (N...Br-Br).
    let xb = system(
        &["N", "H", "H", "H", "Br", "Br"],
        &[
            0.0, 0.0, 0.0, 1.0, 1.5, -0.6, 0.8, -1.6, -0.5, -1.8, 0.1, -0.7, 0.1, 0.2, 5.4, 0.0,
            -0.1, 9.8,
        ],
        0,
        1,
    );
    check_gradient(&xb);
}

#[test]
fn both_starts_give_the_same_energy() {
    let sys = system(
        &["O", "H", "H"],
        &[0.0, 0.0, 0.2, 0.0, 1.45, -0.9, 0.1, -1.5, -0.85],
        0,
        1,
    );
    // Always from SAD: the result does not depend on what was computed before.
    let calc = Gfn1Xtb {
        guess: SccGuess::Sad,
        ..Gfn1Xtb::default()
    };
    let first = calc.compute(&sys).unwrap();
    let again = calc.compute(&sys).unwrap();
    assert_eq!(first.energy.total.to_bits(), again.energy.total.to_bits());
    assert_eq!(first.iterations, again.iterations);
    // The default starts from the charges of the last structure: the same energy, fewer
    // iterations.
    let calc = Gfn1Xtb::default();
    assert_eq!(calc.guess, SccGuess::Previous);
    let cold = calc.compute(&sys).unwrap();
    let warm = calc.compute(&sys).unwrap();
    assert!((cold.energy.total - first.energy.total).abs() < 1.0e-11);
    assert!((cold.energy.total - warm.energy.total).abs() < 1.0e-11);
    assert!(
        warm.iterations < cold.iterations,
        "{} vs {}",
        warm.iterations,
        cold.iterations
    );
}

#[test]
fn the_anion_bf4_stays_on_the_physical_solution() {
    // tblite 0.8.0 runs away from the same start to B -5.8 e and F +1.2 e, 1.66 Hartree lower
    // (the third-order term q^3 has no lower bound); the extrapolation from SAD keeps boron
    // positive and the fluorines negative.
    let r = 1.403 / 0.529_177_210_903;
    let c = r / 3f64.sqrt();
    let sys = system(
        &["B", "F", "F", "F", "F"],
        &[0.0, 0.0, 0.0, c, c, c, -c, -c, c, -c, c, -c, c, -c, -c],
        -1,
        1,
    );
    let out = Gfn1Xtb::default().compute(&sys).unwrap();
    assert!(out.charges[0] > 0.0, "{:?}", out.charges);
    assert!(
        out.charges[1..].iter().all(|q| *q < 0.0),
        "{:?}",
        out.charges
    );
    assert!((out.charges.iter().sum::<f64>() + 1.0).abs() < 1e-8);
    assert!(out.iterations < 60, "{}", out.iterations);
}

#[test]
fn gradients_of_the_elements_to_argon_match_differences() {
    // LiF, BeH2 bent, ammonia borane fragments, Na+ water, AlCl, SiH2 with P and Mg: every
    // new element in a distorted molecule.
    let lif = system(&["Li", "F"], &[0.0, 0.0, 0.0, 0.2, 0.1, 3.0], 0, 1);
    check_gradient(&lif);
    let bnh = system(
        &["B", "N", "H", "H", "H", "H"],
        &[
            0.0, 0.0, 0.0, 0.1, 0.0, 3.1, 1.9, 0.3, -0.8, -1.1, 1.7, -0.7, 1.7, 0.4, 3.9, -0.9,
            -1.5, 3.8,
        ],
        0,
        1,
    );
    check_gradient(&bnh);
    let na = system(
        &["Na", "O", "H", "H"],
        &[
            0.0, 0.0, -4.3, 0.0, 0.1, 0.0, 1.45, 0.0, 1.0, -1.4, 0.2, 1.1,
        ],
        1,
        1,
    );
    check_gradient(&na);
    let mix = system(
        &["Si", "P", "H", "H", "Mg", "Cl", "Al", "Cl"],
        &[
            0.0, 0.0, 0.0, 0.2, 0.1, 4.2, 2.6, 0.3, -0.9, -1.4, 2.2, -0.8, -4.5, 0.4, 0.3, -8.3,
            0.2, 0.5, 0.3, -6.0, 3.5, 0.6, -9.9, 3.0,
        ],
        0,
        1,
    );
    check_gradient(&mix);
}

/// The analytic Hessian against central differences of the analytic gradient.
fn check_hessian(sys: &MolecularSystem, charges: Option<&PointCharges>) {
    check_hessian_with_step(sys, charges, 5e-4);
}

/// As [`check_hessian`], with the step of the differences (their error grows as its square).
fn check_hessian_with_step(sys: &MolecularSystem, charges: Option<&PointCharges>, step: f64) {
    let calc = Gfn1Xtb {
        charge_tolerance: 1e-12,
        hessian_step_bohr: step,
        ..Gfn1Xtb::default()
    };
    let analytic = calc.analytic_hessian_in(sys, charges).unwrap();
    let numerical = calc.numerical_hessian_in(sys, charges).unwrap();
    let dim = analytic.nrows();
    let mut worst = 0.0_f64;
    let mut at = (0, 0);
    for i in 0..dim {
        for j in 0..dim {
            let d = (analytic[(i, j)] - numerical[(i, j)]).abs();
            if d > worst {
                worst = d;
                at = (i, j);
            }
        }
    }
    if worst >= 2e-6 {
        let fine = Gfn1Xtb {
            hessian_step_bohr: 0.2 * step,
            ..calc.clone()
        };
        let finer = fine.numerical_hessian_in(sys, charges).unwrap();
        panic!(
            "max |analytic - differences| = {worst:.3e} at {at:?}: {} vs {} (step {step:e}), {} \
             (step {:e})",
            analytic[at],
            numerical[at],
            finer[at],
            0.2 * step
        );
    }
}

#[test]
fn the_analytic_hessian_of_closed_shells_matches_differences() {
    // Water, and formaldehyde with a thiol (S d shells).
    check_hessian(
        &system(
            &["O", "H", "H"],
            &[0.0, 0.0, 0.2, 0.0, 1.45, -0.9, 0.1, -1.5, -0.85],
            0,
            1,
        ),
        None,
    );
    check_hessian(
        &system(
            &["C", "O", "H", "H", "S", "H"],
            &[
                0.0, 0.0, 0.0, 0.0, 0.1, 2.28, 1.77, 0.0, -1.05, -1.75, 0.2, -1.1, 0.3, 4.9, -3.0,
                1.2, 5.6, -1.0,
            ],
            0,
            1,
        ),
        None,
    );
}

#[test]
fn the_analytic_hessian_of_transition_metals_matches_differences() {
    // Titanium hydride (Sc to Cr shift their d levels the other way with the coordination
    // number), nickel hydride (the Ni-H pair scaling, a doublet), and a titanium-ruthenium
    // hydride (the scaling of a 3d-4d pair), all distorted.
    check_hessian(
        &system(
            &["Ti", "H", "H", "H", "H"],
            &[
                0.0, 0.0, 0.1, 1.9, 1.85, 1.75, -1.8, -1.95, 1.8, -1.85, 1.9, -1.7, 1.95, -1.8,
                -1.9,
            ],
            0,
            1,
        ),
        None,
    );
    check_hessian(
        &system(&["Ni", "H"], &[0.0, 0.0, 0.0, 0.2, 0.1, 2.75], 0, 2),
        None,
    );
    check_hessian(
        &system(
            &["Ti", "Ru", "H", "H", "H", "H"],
            &[
                0.0, 0.0, 0.0, 0.1, 0.2, 4.8, 3.1, 0.0, -0.8, -3.0, 0.4, -0.9, 0.2, 3.05, 5.7,
                -0.3, -3.1, 5.6,
            ],
            0,
            1,
        ),
        None,
    );
}

#[test]
fn the_analytic_hessian_of_an_open_shell_matches_differences() {
    check_hessian(
        &system(
            &["N", "H", "H"],
            &[0.0, 0.0, 0.2, 0.0, 1.5, -0.9, 0.1, -1.6, -0.8],
            0,
            2,
        ),
        None,
    );
}

#[test]
fn the_analytic_hessian_with_fractional_occupations_matches_differences() {
    // Four hydrogen atoms in a rectangle: fractional occupations at 300 K. The occupations
    // change fast with the geometry, so the differences need a small step: at 5e-4 Bohr they
    // are off by 1e-5, at 1e-4 by 4e-7 (the square of the step).
    check_hessian_with_step(
        &system(
            &["H", "H", "H", "H"],
            &[
                0.0, 0.0, 0.0, 2.27, 0.0, 0.0, 2.27, 2.36, 0.06, 0.0, 2.27, -0.04,
            ],
            0,
            1,
        ),
        None,
        1e-4,
    );
}

#[test]
fn the_analytic_hessian_with_a_halogen_bond_matches_differences() {
    // Ammonia and bromine.
    check_hessian(
        &system(
            &["N", "H", "H", "H", "Br", "Br"],
            &[
                0.0, 0.0, 0.0, 1.0, 1.5, -0.6, 0.8, -1.6, -0.5, -1.8, 0.1, -0.7, 0.1, 0.2, 5.4,
                0.0, -0.1, 9.8,
            ],
            0,
            1,
        ),
        None,
    );
}

#[test]
fn the_analytic_hessian_with_point_charges_and_new_elements_matches() {
    let water = system(
        &["O", "H", "H"],
        &[0.0, 0.0, 0.2, 0.0, 1.43, -0.9, 0.0, -1.43, -0.9],
        0,
        1,
    );
    let charges = PointCharges::new(
        vec![[0.4, 0.3, 4.6], [3.9, -0.8, -1.1]],
        vec![0.42, -0.83],
        vec![Some(Element::H), None],
    )
    .unwrap();
    check_hessian(&water, Some(&charges));
    // Na+ and water; boron trifluoride, slightly bent.
    check_hessian(
        &system(
            &["Na", "O", "H", "H"],
            &[
                0.0, 0.0, -4.3, 0.0, 0.1, 0.0, 1.45, 0.0, 1.0, -1.4, 0.2, 1.1,
            ],
            1,
            1,
        ),
        None,
    );
    check_hessian(
        &system(
            &["B", "F", "F", "F"],
            &[
                0.0, 0.0, 0.1, 2.47, 0.0, 0.0, -1.24, 2.14, 0.05, -1.22, -2.15, -0.1,
            ],
            0,
            1,
        ),
        None,
    );
}

#[test]
fn elements_without_parameters_are_refused() {
    let sys = system(&["Fr", "H"], &[0.0, 0.0, 0.0, 0.0, 0.0, 4.2], 0, 1);
    let err = Gfn1Xtb::default().compute(&sys).unwrap_err().to_string();
    assert!(
        err.contains(" Fr") && err.contains(params::ELEMENT_RANGE),
        "{err}"
    );
}

#[test]
fn every_element_from_h_to_rn_has_parameters() {
    assert_eq!(params::ELEMENTS.len(), 86);
    for z in 1..=86u8 {
        let el = Element::from_atomic_number(z).unwrap();
        let p = params::element(el).unwrap();
        assert_eq!(p.z, z);
        assert!(p.shells.iter().all(|s| s.l <= 2), "{}", el.symbol());
        // The reference occupations of GFN1-xTB give the valence electrons of the neutral
        // atom (the lanthanides with 4f in the core: three).
        assert!(p.valence_electrons() > 0.0, "{}", el.symbol());
    }
    for z in [0u8, 87, 103, 118] {
        let err = params::element(Element::from_atomic_number(z).unwrap()).unwrap_err();
        assert!(err.to_string().contains(params::ELEMENT_RANGE), "{err}");
    }
    assert_eq!(element_symbols().len(), 86);
    // The pair parameters of gfn1.f90 for hydrogen with Re and Pt, and the halogen bond of
    // astatine.
    assert_eq!(params::pair_scaling(75, 1), 0.8);
    assert_eq!(params::pair_scaling(1, 78), 0.8);
    assert_eq!(params::pair_scaling(1, 77), 1.0);
    assert!(params::XB_HALOGENS.contains(&85));
}

#[test]
fn point_charges_act_on_the_shells_and_both_gradients_agree_with_differences() {
    // Water near two point charges (one of an element, one of none).
    let sys = system(
        &["O", "H", "H"],
        &[0.0, 0.0, 0.2, 0.0, 1.43, -0.9, 0.0, -1.43, -0.9],
        0,
        1,
    );
    let charges = PointCharges::new(
        vec![[0.4, 0.3, 4.6], [3.9, -0.8, -1.1]],
        vec![0.42, -0.83],
        vec![Some(Element::H), None],
    )
    .unwrap();
    let calc = Gfn1Xtb::default();
    let r = calc.compute_in(&sys, Some(&charges)).unwrap();
    let plain = calc.compute(&sys).unwrap();
    assert!((r.energy.total - plain.energy.total).abs() > 1e-4);
    assert!(r.energy.embedding.abs() > 1e-4);
    let energy =
        |s: &MolecularSystem, pc: &PointCharges| calc.compute_in(s, Some(pc)).unwrap().energy.total;
    let h = 1.0e-4;
    let x0 = sys.coords_flat_bohr();
    for k in 0..x0.nrows() {
        let mut xp = x0.clone();
        xp[k] += h;
        let mut xm = x0.clone();
        xm[k] -= h;
        let fd = (energy(&sys.with_flat_coords_bohr(&xp).unwrap(), &charges)
            - energy(&sys.with_flat_coords_bohr(&xm).unwrap(), &charges))
            / (2.0 * h);
        assert!(
            (fd - r.gradient[k]).abs() < 1e-6,
            "{k}: {fd} {}",
            r.gradient[k]
        );
    }
    for c in 0..2 {
        for k in 0..3 {
            let mut up = charges.clone();
            up.positions_bohr[c][k] += h;
            let mut down = charges.clone();
            down.positions_bohr[c][k] -= h;
            let fd = (energy(&sys, &up) - energy(&sys, &down)) / (2.0 * h);
            let g = r.charge_gradient[c][k];
            assert!((fd - g).abs() < 1e-6, "charge {c} {k}: {fd} {g}");
        }
    }
    // Far away, a charge acts on the total charge only (zero here): no interaction.
    let far = PointCharges::new(vec![[0.0, 0.0, 1.0e6]], vec![1.0], vec![None]).unwrap();
    let r_far = calc.compute_in(&sys, Some(&far)).unwrap();
    assert!((r_far.energy.total - plain.energy.total).abs() < 1e-6);
}

/// Formaldehyde (Bohr).
fn formaldehyde() -> MolecularSystem {
    system(
        &["C", "O", "H", "H"],
        &[
            0.0, 0.0, -1.0, 0.0, 0.0, 1.28, 0.0, 1.78, -2.1, 0.0, -1.78, -2.1,
        ],
        0,
        1,
    )
}

#[test]
fn excited_states_of_the_tda_and_of_stda_with_their_gradients() {
    let tda = Gfn1Xtb::default();
    let stda = Gfn1Xtb {
        excited: ExcitedStates::Stda,
        ..Gfn1Xtb::default()
    };
    let sys = formaldehyde();
    let energies = tda.excitation_energies(&sys).unwrap();
    // All single excitations, ascending, the lowest a few eV above the ground state.
    assert!(energies.windows(2).all(|w| w[0] <= w[1]));
    assert!(energies[0] > 0.02 && energies[0] < 0.5, "{}", energies[0]);
    // The kernel couples the excitations, and the two kernels differ. The lowest one, n to
    // pi*, has no transition charges by symmetry (an in-plane times an out-of-plane orbital),
    // so both leave it at the orbital energy difference; others differ.
    let simplified = stda.excitation_energies(&sys).unwrap();
    assert_eq!(simplified.len(), energies.len());
    let largest = energies
        .iter()
        .zip(&simplified)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(largest > 1e-3, "{largest}");
    let request = |root: usize| EvalRequest {
        excited_state: root,
        ..EvalRequest::ENERGY_GRAD
    };
    let ground = tda.compute(&sys).unwrap().energy.total;
    let total = |at: &MolecularSystem| -> f64 {
        tda.compute(at).unwrap().energy.total + tda.excitation_energies(at).unwrap()[0]
    };
    // The gradient of the lowest state is the slope of its energy along a direction that
    // keeps no symmetry.
    let direction: Vec<f64> = (0..12).map(|k| ((k * 7 + 3) % 5) as f64 - 2.0).collect();
    let length = direction.iter().map(|d| d * d).sum::<f64>().sqrt();
    let x0 = sys.coords_flat_bohr();
    let moved = |s: f64| {
        sys.with_flat_coords_bohr(&Col::from_fn(12, |i| x0[i] + s * direction[i] / length))
            .unwrap()
    };
    let out = tda.evaluate(&sys, &request(1)).unwrap();
    assert!((out.energy.value() - ground - energies[0]).abs() < 1e-10);
    let h = 2.0e-3;
    let numeric = (total(&moved(h)) - total(&moved(-h))) / (2.0 * h);
    let analytic: f64 = (0..12)
        .map(|i| out.gradient_bohr[i] * direction[i] / length)
        .sum();
    assert!(
        (numeric - analytic).abs() < 2e-5,
        "{numeric} against {analytic}"
    );
    // Refused: a state that does not exist, an open shell.
    assert!(tda.evaluate(&sys, &request(energies.len() + 1)).is_err());
    let cation = system(
        &["C", "O", "H", "H"],
        &[
            0.0, 0.0, -1.0, 0.0, 0.0, 1.28, 0.0, 1.78, -2.1, 0.0, -1.78, -2.1,
        ],
        1,
        2,
    );
    let e = tda.evaluate(&cation, &request(1)).unwrap_err().to_string();
    assert!(e.contains("closed-shell"), "{e}");
    assert!(tda.supports_excited_states());
}

/// Formaldehyde and ethylene without symmetry (Bohr).
fn distorted() -> Vec<MolecularSystem> {
    vec![
        system(
            &["C", "O", "H", "H"],
            &[
                0.05, 0.02, -1.0, -0.03, 0.04, 1.28, 0.1, 1.78, -2.15, -0.05, -1.70, -2.05,
            ],
            0,
            1,
        ),
        system(
            &["C", "C", "H", "H", "H", "H"],
            &[
                0.0, 0.0, 0.0, 2.52, 0.1, 0.05, -1.0, 1.75, 0.1, -1.05, -1.72, -0.05, 3.55, 1.8,
                0.1, 3.5, -1.65, 0.2,
            ],
            0,
            1,
        ),
    ]
}

#[test]
fn the_analytic_gradient_of_excited_states_matches_central_differences() {
    for excited in [ExcitedStates::Tda, ExcitedStates::Stda] {
        let analytic = Gfn1Xtb {
            excited,
            guess: SccGuess::Sad,
            ..Gfn1Xtb::default()
        };
        let numerical = Gfn1Xtb {
            analytic_excited: false,
            ..analytic.clone()
        };
        for sys in distorted() {
            for root in 1..=3 {
                let (ea, ga) = analytic.excited(&sys, root).unwrap();
                let (en, gn) = numerical.excited(&sys, root).unwrap();
                assert!((ea - en).abs() < 1e-9, "{excited:?} state {root}");
                let worst = (0..ga.nrows())
                    .map(|k| (ga[k] - gn[k]).abs())
                    .fold(0.0_f64, f64::max);
                assert!(
                    worst < 5e-6,
                    "{excited:?} state {root}: {worst:e}\n{ga:?}\n{gn:?}"
                );
            }
        }
    }
}

/// Molecules with elements of the fifth and sixth periods, distorted (Bohr): CsF, an
/// open-shell gadolinium hydride (a lanthanide), PtHCl (a 5d metal with the Pt-H pair
/// parameter), XeF2, an astatine hydride with ammonia (an At...N halogen bond), the SnH3
/// radical, and an open-shell silver-gold hydride (a 4d-5d pair).
fn heavy_element_systems() -> Vec<MolecularSystem> {
    vec![
        system(&["Cs", "F"], &[0.0, 0.0, 0.0, 0.2, 0.1, 4.45], 0, 1),
        system(
            &["Gd", "H", "H"],
            &[0.0, 0.0, 0.0, 3.85, 0.1, 0.2, -1.8, 3.4, -0.1],
            0,
            2,
        ),
        system(
            &["Pt", "H", "Cl"],
            &[0.0, 0.0, 0.0, 2.95, 0.15, 0.0, -4.2, 0.9, 0.2],
            0,
            1,
        ),
        system(
            &["Xe", "F", "F"],
            &[0.0, 0.0, 0.0, 3.72, 0.1, 0.05, -3.76, 0.3, -0.1],
            0,
            1,
        ),
        system(
            &["H", "At", "N", "H", "H", "H"],
            &[
                0.0, 0.0, -3.3, 0.1, 0.05, 0.0, -0.1, 0.2, 5.7, 1.75, 0.1, 6.4, -0.9, 1.6, 6.3,
                -0.85, -1.5, 6.5,
            ],
            0,
            1,
        ),
        system(
            &["Sn", "H", "H", "H"],
            &[
                0.0, 0.0, 0.0, 3.1, 0.1, -1.1, -1.6, 2.75, -1.2, -1.5, -2.8, -1.05,
            ],
            0,
            2,
        ),
        system(
            &["Ag", "Au", "H"],
            &[0.0, 0.0, 0.0, 4.85, 0.1, 0.0, 7.7, 0.6, 0.3],
            0,
            2,
        ),
    ]
}

#[test]
fn gradients_with_heavy_elements_match_differences() {
    let systems = heavy_element_systems();
    // The astatine hydride with ammonia has a halogen bond.
    let xb = Gfn1Xtb::default().compute(&systems[4]).unwrap();
    assert!(xb.energy.halogen_bond.abs() > 1e-6, "{:?}", xb.energy);
    for sys in &systems {
        check_gradient(sys);
    }
}

#[test]
fn the_analytic_hessian_with_heavy_elements_matches_differences() {
    for sys in &heavy_element_systems() {
        check_hessian(sys, None);
    }
}

#[test]
fn excited_states_of_a_heavy_element_molecule_have_analytic_gradients() {
    // Mercury dichloride, bent: a closed shell with a 5d metal.
    let sys = system(
        &["Hg", "Cl", "Cl"],
        &[0.0, 0.0, 0.0, 4.25, 0.1, 0.05, -4.2, 0.6, -0.1],
        0,
        1,
    );
    for excited in [ExcitedStates::Tda, ExcitedStates::Stda] {
        let analytic = Gfn1Xtb {
            excited,
            guess: SccGuess::Sad,
            ..Gfn1Xtb::default()
        };
        let numerical = Gfn1Xtb {
            analytic_excited: false,
            ..analytic.clone()
        };
        for root in 1..=3 {
            let (ea, ga) = analytic.excited(&sys, root).unwrap();
            let (en, gn) = numerical.excited(&sys, root).unwrap();
            assert!((ea - en).abs() < 1e-9, "{excited:?} state {root}");
            let worst = (0..ga.nrows())
                .map(|k| (ga[k] - gn[k]).abs())
                .fold(0.0_f64, f64::max);
            assert!(
                worst < 5e-6,
                "{excited:?} state {root}: {worst:e}\n{ga:?}\n{gn:?}"
            );
        }
    }
}

#[test]
fn transitions_of_closed_and_open_shells() {
    let tda = Gfn1Xtb::default();
    let sys = distorted().remove(0);
    let singlets = tda.excitation_energies(&sys).unwrap();
    let scc = tda.scc(&sys, None).unwrap();
    let eps = &scc.elec.eigenvalues;
    let occupied = scc.elec.spin_occupations[0]
        .iter()
        .filter(|&&f| f > 0.5)
        .count();
    let mut gaps: Vec<f64> = (0..occupied)
        .flat_map(|i| (occupied..eps.len()).map(move |a| eps[a] - eps[i]))
        .collect();
    gaps.sort_by(f64::total_cmp);
    let count = 12;
    let found = tda.transitions_of(&sys, count).unwrap();
    assert_eq!(found.len(), count);
    let near = |e: f64, list: &[f64]| {
        list.iter()
            .map(|v| (v - e).abs())
            .fold(f64::INFINITY, f64::min)
    };
    // A closed shell: the states of both spins are its singlets (the closed-shell TDA) and the
    // triplets (Ms = 0, no kernel: the orbital energy differences, without a dipole).
    for t in &found {
        let (to_singlet, to_triplet) = (near(t.excitation, &singlets), near(t.excitation, &gaps));
        assert!(to_singlet < 1e-7 || to_triplet < 1e-7, "{}", t.excitation);
        if to_singlet > 1e-5 {
            let mu: f64 = t
                .transition_dipole
                .iter()
                .map(|v| v * v)
                .sum::<f64>()
                .sqrt();
            assert!(mu < 1e-6, "a triplet with a dipole {mu}");
            // ... nor a transition density: its two spins cancel.
            let tp = t.transition_population.as_ref().unwrap();
            assert!(tp.iter().all(|q| q.abs() < 1e-6), "{tp:?}");
        }
        // The electrons move between the atoms, none are made.
        let moved: f64 = t.population_change.iter().sum();
        assert!(moved.abs() < 1e-9, "{moved}");
        // The transition density has no electrons in all (the states are orthogonal), and the
        // change of the dipole is there.
        let tp: f64 = t.transition_population.as_ref().unwrap().iter().sum();
        assert!(tp.abs() < 1e-9, "{tp}");
        assert!(t.dipole_change.unwrap().iter().all(|v| v.is_finite()));
    }
    let highest = found[count - 1].excitation;
    for &e in singlets.iter().take_while(|&&e| e < highest - 1e-6) {
        assert!(
            found.iter().any(|t| (t.excitation - e).abs() < 1e-7),
            "singlet {e} missing"
        );
    }
    // A singlet with a dipole: the n to pi* state of the distorted molecule is weak, a pi to
    // pi* one is not.
    assert!(found
        .iter()
        .any(|t| t.transition_dipole.iter().map(|v| v * v).sum::<f64>() > 1e-2));
    // An open shell: the cation, a doublet.
    let mut cation = sys.clone();
    cation.spin_charge = SpinCharge {
        charge: 1,
        multiplicity: 2,
    };
    let states = tda.transitions_of(&cation, 6).unwrap();
    assert_eq!(states.len(), 6);
    assert!(states
        .windows(2)
        .all(|w| w[0].excitation <= w[1].excitation));
    assert!(states[0].excitation > 0.0);
    for t in &states {
        assert!(t.population_change.iter().sum::<f64>().abs() < 1e-9);
    }
    // Through the calculator: the TDA gives them, the sTDA does not.
    assert!(tda.transitions(&cation, 3).unwrap().is_some());
    let stda = Gfn1Xtb {
        excited: ExcitedStates::Stda,
        ..Gfn1Xtb::default()
    };
    assert!(stda.transitions(&cation, 3).unwrap().is_none());
}
