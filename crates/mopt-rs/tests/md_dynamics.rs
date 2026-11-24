//! Molecular dynamics on the built-in calculators: energy conservation against the time step
//! (Lennard-Jones, EMT in a periodic cell, GFN-FF), canonical temperatures, constant
//! pressure, SHAKE/RATTLE, and the nanoreactor's periodic compression.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Arc;

use mopt_rs::domain::{
    positions, Angstrom, AtomId, Element, MolecularSystem, PeriodicCell, Quantity, SpinCharge,
    Units,
};
use mopt_rs::md::barostat::gigapascal_per_atomic_unit;
use mopt_rs::md::settings::ConstraintSolver;
use mopt_rs::md::{
    MdSettings, MdSpace, MdStepReport, MolecularDynamics, RigidMotion, ThermostatSettings,
};
use mopt_rs::opt::state_machine::{GeometryOptimizer, OptimizerSettings};
use mopt_rs::pes::bias::{AdBias, NanoReactor};
use mopt_rs::pes::calculator::Calculator;
use mopt_rs::pes::constraint::{Constraint, ConstraintKind};
use mopt_rs::pes::registry::{CalculatorOptions, CalculatorRegistry};
use mopt_rs::pes::PesTerm;
use mopt_rs::workflow::md::{run_md, FnMdObserver, MoleculeMdSurface};
use mopt_rs::workflow::run_optimization;

fn structure(symbols: &[&str], rows: &[[f64; 3]], cell: Option<PeriodicCell>) -> MolecularSystem {
    let elements = symbols
        .iter()
        .map(|s| Element::parse_token(s).unwrap())
        .collect();
    MolecularSystem::new(
        elements,
        &positions::<Angstrom>(rows),
        SpinCharge::default(),
        cell,
    )
    .unwrap()
}

fn calculator(name: &str, sys: &MolecularSystem) -> Box<dyn Calculator> {
    CalculatorRegistry::native()
        .create(name, &CalculatorOptions::default(), sys)
        .unwrap()
}

fn relaxed(calc: &dyn Calculator, sys: &MolecularSystem) -> MolecularSystem {
    let mut opt = GeometryOptimizer::new(
        OptimizerSettings {
            max_iterations: 300,
            ..OptimizerSettings::default()
        },
        sys.clone(),
    )
    .unwrap();
    run_optimization(&mut opt, calc, &mut []).unwrap().system
}

fn dynamics(
    calc: &dyn Calculator,
    start: &MolecularSystem,
    terms: Vec<(String, Arc<dyn PesTerm>)>,
    settings: MdSettings,
) -> Vec<MdStepReport> {
    let space = MdSpace::atoms(&start.masses_amu()).unwrap();
    let mut md = MolecularDynamics::new(
        settings,
        space,
        start.coords_flat_bohr(),
        Some(start.clone()),
    )
    .unwrap();
    let surface = MoleculeMdSurface::with_terms(calc, start.clone(), terms).unwrap();
    let mut reports = Vec::new();
    let mut keep = FnMdObserver::new(|r: &MdStepReport| {
        reports.push(r.clone());
        Ok(true)
    });
    run_md(&mut md, &surface, &mut [&mut keep]).unwrap();
    reports
}

fn nve(dt_fs: f64, total_fs: f64, temperature: f64, rigid: RigidMotion) -> MdSettings {
    MdSettings {
        time_step: dt_fs * Units::FS_TO_AU_TIME,
        steps: (total_fs / dt_fs) as usize,
        temperature,
        seed: 7,
        rigid,
        ..MdSettings::default()
    }
}

/// Largest deviation of the conserved quantity from its start.
fn deviation(reports: &[MdStepReport]) -> f64 {
    let c0 = reports[0].conserved;
    reports
        .iter()
        .map(|r| (r.conserved - c0).abs())
        .fold(0.0, f64::max)
}

fn ethanol() -> MolecularSystem {
    structure(
        &["C", "C", "O", "H", "H", "H", "H", "H", "H"],
        &[
            [0.000, 0.000, 0.000],
            [1.515, 0.000, 0.000],
            [2.030, 1.320, 0.000],
            [2.990, 1.280, 0.000],
            [-0.390, 1.020, 0.000],
            [-0.390, -0.510, 0.880],
            [-0.390, -0.510, -0.880],
            [1.900, -0.520, 0.880],
            [1.900, -0.520, -0.880],
        ],
        None,
    )
}

fn copper(a: f64) -> MolecularSystem {
    let mut rows = Vec::new();
    for i in 0..2 {
        for j in 0..2 {
            for k in 0..2 {
                for f in [
                    [0.0, 0.0, 0.0],
                    [0.5, 0.5, 0.0],
                    [0.5, 0.0, 0.5],
                    [0.0, 0.5, 0.5],
                ] {
                    rows.push([
                        (i as f64 + f[0]) * a,
                        (j as f64 + f[1]) * a,
                        (k as f64 + f[2]) * a,
                    ]);
                }
            }
        }
    }
    let l = Quantity::<Angstrom>::new(2.0 * a);
    structure(
        &["Cu"; 32],
        &rows,
        Some(PeriodicCell::orthorhombic(l, l, l).unwrap()),
    )
}

#[test]
fn the_energy_error_of_velocity_verlet_falls_with_the_square_of_the_time_step() {
    let mol = ethanol();
    let ff = calculator("gfnff", &mol);
    let mol = relaxed(ff.as_ref(), &mol);
    let full = RigidMotion::TranslationRotation;
    let coarse = deviation(&dynamics(
        ff.as_ref(),
        &mol,
        Vec::new(),
        nve(0.5, 300.0, 300.0, full),
    ));
    let fine = deviation(&dynamics(
        ff.as_ref(),
        &mol,
        Vec::new(),
        nve(0.25, 300.0, 300.0, full),
    ));
    // Below 1 % of k_B T per atom at 0.5 fs, the default.
    let kt = Units::BOLTZMANN_HARTREE_PER_K * 300.0;
    assert!(coarse < 0.01 * kt * 9.0, "{coarse}");
    assert!((coarse / fine).log2() > 1.6, "{coarse} {fine}");
    // Copper in its cell with EMT, 2 fs.
    let cu = copper(3.60);
    let emt = calculator("emt", &cu);
    let reports = dynamics(
        emt.as_ref(),
        &cu,
        Vec::new(),
        nve(2.0, 400.0, 300.0, RigidMotion::Translation),
    );
    assert!(deviation(&reports) < 0.01 * kt * 32.0);
    assert_eq!(reports[0].dof, 93.0);
    assert!(reports[0].pressure.is_some() && reports[0].volume.is_some());
}

#[test]
fn bonds_to_hydrogen_stay_at_their_lengths_with_shake_and_rattle() {
    let mol = ethanol();
    let ff = calculator("gfnff", &mol);
    let mol = relaxed(ff.as_ref(), &mol);
    let bonds: Vec<Constraint> = [(0, 4), (0, 5), (0, 6), (1, 7), (1, 8), (2, 3)]
        .iter()
        .map(|&(a, b)| {
            Constraint::at_start(ConstraintKind::Distance(
                vec![AtomId::new(a)],
                vec![AtomId::new(b)],
            ))
        })
        .collect();
    let settings = MdSettings {
        constraints: bonds,
        ..nve(1.0, 400.0, 300.0, RigidMotion::TranslationRotation)
    };
    let reports = dynamics(ff.as_ref(), &mol, Vec::new(), settings);
    assert_eq!(reports[0].dof, 27.0 - 6.0 - 6.0);
    for r in &reports {
        assert!(r.constraint_residual.unwrap() < 1e-9);
    }
    let kt = Units::BOLTZMANN_HARTREE_PER_K * 300.0;
    assert!(
        deviation(&reports) < 0.02 * kt * 9.0,
        "{}",
        deviation(&reports)
    );
}

#[test]
fn lincs_holds_the_bonds_to_hydrogen_like_rattle() {
    let mol = ethanol();
    let ff = calculator("gfnff", &mol);
    let mol = relaxed(ff.as_ref(), &mol);
    let bonds: Vec<Constraint> = [(0, 4), (0, 5), (0, 6), (1, 7), (1, 8), (2, 3)]
        .iter()
        .map(|&(a, b)| {
            Constraint::at_start(ConstraintKind::Distance(
                vec![AtomId::new(a)],
                vec![AtomId::new(b)],
            ))
        })
        .collect();
    let run = |solver| {
        let settings = MdSettings {
            constraints: bonds.clone(),
            constraint_solver: solver,
            ..nve(1.0, 400.0, 300.0, RigidMotion::TranslationRotation)
        };
        dynamics(ff.as_ref(), &mol, Vec::new(), settings)
    };
    let rattle = run(ConstraintSolver::Shake);
    let lincs = run(ConstraintSolver::Lincs {
        order: 4,
        iterations: 1,
    });
    let worst = |r: &[MdStepReport]| {
        r.iter()
            .filter_map(|x| x.constraint_residual)
            .fold(0.0_f64, f64::max)
    };
    eprintln!(
        "X-H bonds, 1 fs, 400 fs: residual {:.2e} (RATTLE), {:.2e} (LINCS); deviation {:.3e}, \
         {:.3e} Hartree",
        worst(&rattle),
        worst(&lincs),
        deviation(&rattle),
        deviation(&lincs)
    );
    assert_eq!(lincs[0].dof, rattle[0].dof);
    assert!(worst(&lincs) < 1e-5);
    let kt = Units::BOLTZMANN_HARTREE_PER_K * 300.0;
    assert!(deviation(&lincs) < 0.02 * kt * 9.0);
    // Other constraints than bonds are refused for LINCS.
    let angle = vec![Constraint::at_start(ConstraintKind::Angle([
        AtomId::new(4),
        AtomId::new(0),
        AtomId::new(5),
    ]))];
    let settings = MdSettings {
        constraints: angle,
        constraint_solver: ConstraintSolver::Lincs {
            order: 4,
            iterations: 1,
        },
        ..nve(1.0, 10.0, 300.0, RigidMotion::TranslationRotation)
    };
    let space = MdSpace::atoms(&mol.masses_amu()).unwrap();
    assert!(
        MolecularDynamics::new(settings, space, mol.coords_flat_bohr(), Some(mol.clone())).is_err()
    );
}

#[test]
fn copper_at_constant_pressure_keeps_the_pressure_and_its_volume_fluctuates() {
    let cu = copper(3.60);
    let emt = calculator("emt", &cu);
    let mut settings = MdSettings {
        time_step: 4.0 * Units::FS_TO_AU_TIME,
        steps: 4000,
        temperature: 300.0,
        thermostat: "nose_hoover_chain".to_string(),
        barostat: Some("mtk".to_string()),
        pressure: 0.0,
        seed: 3,
        rigid: RigidMotion::Translation,
        ..MdSettings::default()
    };
    settings.barostat_settings.tau = 500.0 * Units::FS_TO_AU_TIME;
    let reports = dynamics(emt.as_ref(), &cu, Vec::new(), settings);
    let tail = &reports[1000..];
    let n = tail.len() as f64;
    let p =
        tail.iter().map(|r| r.pressure.unwrap()).sum::<f64>() / n * gigapascal_per_atomic_unit();
    assert!(p.abs() < 0.3, "mean pressure {p} GPa");
    let v: Vec<f64> = tail.iter().map(|r| r.volume.unwrap()).collect();
    let mean = v.iter().sum::<f64>() / n;
    let spread = (v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n).sqrt();
    assert!(spread > 1e-4 * mean, "the volume moves: {spread}");
    // The conserved quantity of the Martyna-Tobias-Klein equations stays.
    let kt = Units::BOLTZMANN_HARTREE_PER_K * 300.0;
    assert!(
        deviation(&reports) < 0.05 * kt * 32.0,
        "{}",
        deviation(&reports)
    );
}

#[test]
fn copper_in_a_flexible_cell_keeps_every_component_of_the_pressure() {
    let cu = copper(3.60);
    let masses: Vec<f64> = cu
        .masses_amu()
        .iter()
        .map(|m| m * Units::AMU_TO_AU_MASS)
        .collect();
    let emt = calculator("emt", &cu);
    let kt = Units::BOLTZMANN_HARTREE_PER_K * 300.0;
    // The bound on every mean component: about three standard errors over these 12 ps, from
    // the 80 ps runs of the documentation of mdmain (the stochastic cell rescaling follows the
    // pressure only on average and scatters most).
    for (barostat, thermostat, bound) in [
        ("mtk_aniso", "nose_hoover_chain", 0.3),
        ("crescale_aniso", "bussi", 1.3),
    ] {
        let mut settings = MdSettings {
            time_step: 4.0 * Units::FS_TO_AU_TIME,
            steps: 4000,
            temperature: 300.0,
            thermostat: thermostat.to_string(),
            barostat: Some(barostat.to_string()),
            pressure: 0.0,
            seed: 3,
            rigid: RigidMotion::Translation,
            ..MdSettings::default()
        };
        settings.barostat_settings.tau = 500.0 * Units::FS_TO_AU_TIME;
        let reports = dynamics(emt.as_ref(), &cu, Vec::new(), settings);
        let tail = &reports[1000..];
        let n = tail.len() as f64;
        // The mean of every component of the pressure tensor K / V - sigma is the target.
        let mut mean = [[0.0; 3]; 3];
        for r in tail {
            let (s, v) = (r.stress.unwrap(), r.volume.unwrap());
            for a in 0..3 {
                for b in 0..3 {
                    let k: f64 = masses
                        .iter()
                        .enumerate()
                        .map(|(i, m)| m * r.v[3 * i + a] * r.v[3 * i + b])
                        .sum();
                    mean[a][b] += (k / v - s[a][b]) / n * gigapascal_per_atomic_unit();
                }
            }
        }
        for row in &mean {
            for p in row {
                assert!(
                    p.abs() < bound,
                    "{barostat}: mean pressure tensor {mean:?} GPa"
                );
            }
        }
        // The lengths and the angles of the cell move.
        let lengths = |r: &MdStepReport| -> [f64; 3] {
            let c = r.cell.as_ref().unwrap();
            std::array::from_fn(|k| {
                let v = c.lattice_vector(k).value();
                (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
            })
        };
        let ratios: Vec<f64> = tail
            .iter()
            .map(|r| {
                let l = lengths(r);
                l[0] / l[1]
            })
            .collect();
        let spread = ratios.iter().map(|x| (x - 1.0).abs()).fold(0.0, f64::max);
        assert!(spread > 1e-4, "{barostat}: the shape moves: {spread}");
        let skew = tail
            .iter()
            .map(|r| {
                let c = r.cell.as_ref().unwrap();
                let (a, b) = (c.lattice_vector(0).value(), c.lattice_vector(1).value());
                let dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
                (dot / (lengths(r)[0] * lengths(r)[1])).abs()
            })
            .fold(0.0, f64::max);
        assert!(
            skew > 1e-5 && skew < 0.1,
            "{barostat}: the angles move: {skew}"
        );
        if barostat == "mtk_aniso" {
            // The conserved quantity of the Martyna-Tobias-Klein equations stays.
            assert!(
                deviation(&reports) < 0.05 * kt * 32.0,
                "{}",
                deviation(&reports)
            );
        }
    }
}

#[test]
fn a_cluster_at_a_canonical_temperature() {
    let phi = 0.5 * (1.0 + 5f64.sqrt());
    let mut rows = vec![[0.0, 0.0, 0.0]];
    for (a, b) in [(1.0, phi), (1.0, -phi), (-1.0, phi), (-1.0, -phi)] {
        rows.push([0.0, a, b]);
        rows.push([a, b, 0.0]);
        rows.push([b, 0.0, a]);
    }
    let r = (1.0 + phi * phi).sqrt();
    let rows: Vec<[f64; 3]> = rows.iter().map(|p| p.map(|c| 3.7 * c / r)).collect();
    let ar = structure(&["Ar"; 13], &rows, None);
    let lj = calculator("lj", &ar);
    let ar = relaxed(lj.as_ref(), &ar);
    let settings = MdSettings {
        time_step: 10.0 * Units::FS_TO_AU_TIME,
        steps: 30_000,
        temperature: 30.0,
        thermostat: "bussi".to_string(),
        thermostat_settings: ThermostatSettings::default(),
        seed: 12,
        rigid: RigidMotion::TranslationRotation,
        ..MdSettings::default()
    };
    let reports = dynamics(lj.as_ref(), &ar, Vec::new(), settings);
    let t: Vec<f64> = reports[5000..].iter().map(|r| r.temperature).collect();
    let n = t.len() as f64;
    let mean = t.iter().sum::<f64>() / n;
    let var = t.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    let canonical = 2.0 * 30.0 * 30.0 / reports[0].dof;
    assert!((mean / 30.0 - 1.0).abs() < 0.02, "{mean}");
    assert!((var / canonical - 1.0).abs() < 0.15, "{var} {canonical}");
}

#[test]
fn the_nanoreactor_compresses_on_its_period() {
    // Eight argon atoms spread over 9 Angstrom; the narrow sphere (3 Angstrom) holds for
    // the second half of every 200 fs.
    let rows: Vec<[f64; 3]> = (0..8)
        .map(|k| {
            let s = |b: usize| if (k >> b) & 1 == 1 { 4.5 } else { -4.5 };
            [s(0), s(1), s(2)]
        })
        .collect();
    let ar = structure(&["Ar"; 8], &rows, None);
    let fs = Units::FS_TO_AU_TIME;
    let k = Units::KCAL_MOL_TO_HARTREE / (Units::ANGSTROM_TO_BOHR * Units::ANGSTROM_TO_BOHR);
    let wall = NanoReactor {
        atoms: ar.atoms().map(|a| (a, 39.948)).collect(),
        center: [0.0; 3],
        r1: 14.0 * Units::ANGSTROM_TO_BOHR,
        k1: 1.0 * k,
        r2: 3.0 * Units::ANGSTROM_TO_BOHR,
        k2: 0.5 * k,
        tau: 100.0 * fs,
        period: 200.0 * fs,
    };
    let terms: Vec<(String, Arc<dyn PesTerm>)> = vec![(
        "nano_reactor_potential_0".to_string(),
        Arc::new(AdBias::new(wall)),
    )];
    let lj = calculator("lj", &ar);
    let settings = MdSettings {
        time_step: 2.0 * fs,
        steps: 200,
        temperature: 50.0,
        thermostat: "langevin".to_string(),
        integrator: "baoab".to_string(),
        seed: 1,
        ..MdSettings::default()
    };
    let reports = dynamics(lj.as_ref(), &ar, terms, settings);
    let radius = |r: &MdStepReport| -> f64 {
        (0..8)
            .map(|i| (0..3).map(|c| r.x[3 * i + c].powi(2)).sum::<f64>().sqrt())
            .fold(0.0, f64::max)
    };
    // No wall energy in the wide phase, a wall in the narrow one, and the atoms pushed in.
    assert!(reports[..50].iter().all(|r| r.energy == r.raw_energy));
    assert!(reports[51..100].iter().all(|r| r.energy > r.raw_energy));
    assert!(radius(&reports[100]) < radius(&reports[50]));
}
