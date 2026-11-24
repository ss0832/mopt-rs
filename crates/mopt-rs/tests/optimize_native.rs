//! Geometry optimizations with the native EMT and Tersoff calculators, isolated and periodic.
//!
//! This file is part of mopt-rs.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use mopt_rs::domain::quantity::shorthand::angstrom;
use mopt_rs::domain::{
    positions, Angstrom, Element, EvalRequest, MolecularSystem, PeriodicCell, SpinCharge,
};
use mopt_rs::opt::convergence::ConvergenceCriteria;
use mopt_rs::opt::state_machine::{GeometryOptimizer, OptimizerSettings, Termination};
use mopt_rs::pes::native::{Emt, Tersoff};
use mopt_rs::pes::Calculator;
use mopt_rs::workflow::run_optimization;

fn rattle(rows: &[[f64; 3]], amp: f64) -> Vec<[f64; 3]> {
    rows.iter()
        .enumerate()
        .map(|(i, r)| {
            let s = |k: usize| amp * ((7 * i + 3 * k) as f64 * 1.37).sin();
            [r[0] + s(0), r[1] + s(1), r[2] + s(2)]
        })
        .collect()
}

fn fcc_conventional(a: f64) -> Vec<[f64; 3]> {
    vec![
        [0.0, 0.0, 0.0],
        [0.0, 0.5 * a, 0.5 * a],
        [0.5 * a, 0.0, 0.5 * a],
        [0.5 * a, 0.5 * a, 0.0],
    ]
}

fn diamond_conventional(a: f64) -> Vec<[f64; 3]> {
    let mut out = fcc_conventional(a);
    let shifted: Vec<[f64; 3]> = out
        .iter()
        .map(|p| [p[0] + 0.25 * a, p[1] + 0.25 * a, p[2] + 0.25 * a])
        .collect();
    out.extend(shifted);
    out
}

fn energy(calc: &dyn Calculator, sys: &MolecularSystem) -> f64 {
    calc.evaluate(sys, &EvalRequest::ENERGY_GRAD)
        .unwrap()
        .energy
        .value()
}

fn optimize(
    calc: &dyn Calculator,
    start: MolecularSystem,
    method: &str,
    fc: Option<usize>,
) -> (Termination, MolecularSystem, f64) {
    let settings = OptimizerSettings {
        methods: vec![method.to_string()],
        max_iterations: 500,
        exact_hessian_interval: fc,
        criteria: ConvergenceCriteria::TIGHT,
        dissociation_distance: None,
        ..OptimizerSettings::default()
    };
    let mut opt = GeometryOptimizer::new(settings, start).unwrap();
    let out = run_optimization(&mut opt, calc, &mut []).unwrap();
    (out.termination.unwrap(), out.system, out.energy.value())
}

fn periodic(el: Element, rows: &[[f64; 3]], a: f64) -> MolecularSystem {
    let cell = PeriodicCell::orthorhombic(angstrom(a), angstrom(a), angstrom(a)).unwrap();
    MolecularSystem::new(
        vec![el; rows.len()],
        &positions::<Angstrom>(rows),
        SpinCharge::default(),
        Some(cell),
    )
    .unwrap()
}

#[test]
fn periodic_crystals_relax_to_the_perfect_lattice() {
    let cu = Element::parse_token("Cu").unwrap();
    let si = Element::parse_token("Si").unwrap();
    let cases: Vec<(Box<dyn Calculator>, Element, Vec<[f64; 3]>, f64)> = vec![
        (Box::new(Emt::new()), cu, fcc_conventional(3.61), 3.61),
        (
            Box::new(Tersoff::tersoff_1989(&[si]).unwrap()),
            si,
            diamond_conventional(5.43),
            5.43,
        ),
    ];
    for (calc, el, rows, a) in cases {
        let perfect = energy(calc.as_ref(), &periodic(el, &rows, a));
        for (method, fc) in [
            ("rsirfo_bfgs", Some(1000)),
            ("rsirfo_block_fsb", Some(1000)),
            ("FIRELARS", None),
            ("lbfgs", None),
        ] {
            let start = periodic(el, &rattle(&rows, 0.12), a);
            let e0 = energy(calc.as_ref(), &start);
            let (t, _, e) = optimize(calc.as_ref(), start, method, fc);
            eprintln!(
                "{} {method}: E0 = {e0:.8}, E = {e:.10}, perfect = {perfect:.10} ({t:?})",
                calc.name()
            );
            assert_eq!(t, Termination::Converged, "{} {method}", calc.name());
            assert!(
                (e - perfect).abs() < 1e-7,
                "{} {method}: {e} vs perfect {perfect}",
                calc.name()
            );
        }
    }
}

#[test]
fn clusters_relax_with_cartesian_and_internal_coordinates() {
    let cu = Element::parse_token("Cu").unwrap();
    let si = Element::parse_token("Si").unwrap();
    let cu7 = MolecularSystem::new(
        vec![cu; 7],
        &positions::<Angstrom>(&rattle(
            &[
                [0.0, 0.0, 0.0],
                [2.5, 0.0, 0.0],
                [1.25, 2.17, 0.0],
                [1.25, 0.72, 2.04],
                [-1.25, 2.17, 0.0],
                [3.75, 2.17, 0.0],
                [1.25, 0.72, -2.04],
            ],
            0.1,
        )),
        SpinCharge::default(),
        None,
    )
    .unwrap();
    let si5 = MolecularSystem::new(
        vec![si; 5],
        &positions::<Angstrom>(&[
            [0.0, 0.0, 0.0],
            [2.3, 0.1, 0.0],
            [1.1, 2.0, 0.2],
            [1.2, 0.7, 1.9],
            [-0.9, 1.5, 1.4],
        ]),
        SpinCharge::default(),
        None,
    )
    .unwrap();
    let tersoff = Tersoff::tersoff_1989(&[si]).unwrap();
    let emt = Emt::new();
    let cases: Vec<(&dyn Calculator, &MolecularSystem)> = vec![(&emt, &cu7), (&tersoff, &si5)];
    for (calc, start) in cases {
        let run = |coords: &str, method: &str, from: &MolecularSystem| {
            let settings = OptimizerSettings {
                methods: vec![method.to_string()],
                coordinates: coords.to_string(),
                max_iterations: 800,
                exact_hessian_interval: method.starts_with("rsirfo").then_some(1000),
                criteria: ConvergenceCriteria::TIGHT,
                ..OptimizerSettings::default()
            };
            let mut opt = GeometryOptimizer::new(settings, from.clone()).unwrap();
            let out = run_optimization(&mut opt, calc, &mut []).unwrap();
            eprintln!(
                "{} {coords}/{method}: E0 = {:.8} -> {:.10} in {} ({:?})",
                calc.name(),
                energy(calc, from),
                out.energy.value(),
                out.iterations,
                out.termination
            );
            assert_eq!(
                out.termination,
                Some(Termination::Converged),
                "{coords}/{method}"
            );
            assert!(out.energy.value() < energy(calc, from));
            out
        };
        // Every method relaxes the strained start (possibly into different minima).
        let reference = run("cart", "rsirfo_bfgs", start);
        run("dic", "rsirfo_bfgs", start);
        run("cart", "FIRELARS", start);
        // From a small perturbation of a minimum, Cartesian and delocalized-internal
        // quasi-Newton runs return to it.
        let rows: Vec<[f64; 3]> = reference
            .system
            .positions()
            .iter()
            .map(|p| p.to::<Angstrom>().value())
            .collect();
        let perturbed = MolecularSystem::new(
            start.elements.clone(),
            &positions::<Angstrom>(&rattle(&rows, 0.05)),
            SpinCharge::default(),
            None,
        )
        .unwrap();
        for coords in ["cart", "dic", "ric"] {
            let out = run(coords, "rsirfo_bfgs", &perturbed);
            assert!(
                (out.energy.value() - reference.energy.value()).abs() < 1e-7,
                "{coords}"
            );
        }
    }
}
