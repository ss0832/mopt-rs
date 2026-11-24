//! Measurements behind the defaults of `mdmain`: energy conservation of NVE runs against
//! the time step, the temperature distributions of the thermostats, the volume of NPT runs
//! with the barostats, and the drift of SHAKE/RATTLE constraints. The tables go to
//! `docs/source/molecular_dynamics.md`.
//!
//! ```text
//! cargo run --release -p mopt-rs --example md_measurements -- nve|thermostats|npt|shake|all
//! ```
//!
//! Systems: an Ar13 cluster (`lj`), 32 copper atoms in a periodic cell (`emt`), ethanol
//! (`gfnff`), and the Muller-Brown surface of two variables with unit masses (`mb2d`).
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Col;
use mopt_rs::domain::{
    positions, Angstrom, Element, MolecularSystem, PeriodicCell, SpinCharge, Units,
};
use mopt_rs::error::{MoptError, MoptResult};
use mopt_rs::md::barostat::gigapascal_per_atomic_unit;
use mopt_rs::md::settings::ConstraintSolver;
use mopt_rs::md::{
    InitialVelocities, MdSettings, MdSpace, MdStepReport, MolecularDynamics, RigidMotion,
    ThermostatSettings,
};
use mopt_rs::opt::state_machine::{GeometryOptimizer, OptimizerSettings};
use mopt_rs::pes::calculator::Calculator;
use mopt_rs::pes::constraint::{Constraint, ConstraintKind};
use mopt_rs::pes::objective::MullerBrown2D;
use mopt_rs::pes::registry::{CalculatorOptions, CalculatorRegistry};
use mopt_rs::workflow::md::{run_md, FnMdObserver, MoleculeMdSurface, ObjectiveMdSurface};
use mopt_rs::workflow::run_optimization;

fn element(symbol: &str) -> MoptResult<Element> {
    Element::parse_token(symbol)
}

fn structure(
    symbols: &[&str],
    rows: &[[f64; 3]],
    cell: Option<PeriodicCell>,
) -> MoptResult<MolecularSystem> {
    let elements = symbols
        .iter()
        .map(|s| element(s))
        .collect::<MoptResult<Vec<_>>>()?;
    MolecularSystem::new(
        elements,
        &positions::<Angstrom>(rows),
        SpinCharge::default(),
        cell,
    )
}

fn calculator(name: &str, sys: &MolecularSystem) -> MoptResult<Box<dyn Calculator>> {
    CalculatorRegistry::native().create(name, &CalculatorOptions::default(), sys)
}

/// `sys` relaxed on `calc` (rigid-body motion kept out).
fn relaxed(calc: &dyn Calculator, sys: &MolecularSystem) -> MoptResult<MolecularSystem> {
    let settings = OptimizerSettings {
        max_iterations: 300,
        rigid_body_invariant: !sys.has_pbc(),
        ..OptimizerSettings::default()
    };
    let mut opt = GeometryOptimizer::new(settings, sys.clone())?;
    let outcome = run_optimization(&mut opt, calc, &mut [])?;
    Ok(outcome.system)
}

fn ar13() -> MoptResult<MolecularSystem> {
    let phi = 0.5 * (1.0 + 5f64.sqrt());
    let mut rows = vec![[0.0, 0.0, 0.0]];
    for (a, b) in [(1.0, phi), (1.0, -phi), (-1.0, phi), (-1.0, -phi)] {
        rows.push([0.0, a, b]);
        rows.push([a, b, 0.0]);
        rows.push([b, 0.0, a]);
    }
    let r = (1.0 + phi * phi).sqrt();
    let rows: Vec<[f64; 3]> = rows.iter().map(|p| p.map(|c| 3.7 * c / r)).collect();
    structure(&["Ar"; 13], &rows, None)
}

fn copper(a: f64) -> MoptResult<MolecularSystem> {
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
    let l = mopt_rs::domain::Quantity::<Angstrom>::new(2.0 * a);
    let cell = PeriodicCell::orthorhombic(l, l, l)?;
    structure(&["Cu"; 32], &rows, Some(cell))
}

fn ethanol() -> MoptResult<MolecularSystem> {
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

/// Run `settings` on `calc` from `start`; the reports of all steps.
fn dynamics(
    calc: &dyn Calculator,
    start: &MolecularSystem,
    settings: MdSettings,
) -> MoptResult<Vec<MdStepReport>> {
    let space = MdSpace::atoms(&start.masses_amu())?;
    let mut md = MolecularDynamics::new(
        settings,
        space,
        start.coords_flat_bohr(),
        Some(start.clone()),
    )?;
    let surface = MoleculeMdSurface::with_terms(calc, start.clone(), Vec::new())?;
    let mut reports = Vec::new();
    let mut keep = FnMdObserver::new(|r: &MdStepReport| {
        reports.push(r.clone());
        Ok(true)
    });
    run_md(&mut md, &surface, &mut [&mut keep])?;
    Ok(reports)
}

/// Largest deviation of the conserved quantity and its drift (linear fit), both per atom
/// in meV (per variable for plain variables), and the mean temperature.
fn conservation(reports: &[MdStepReport], particles: f64) -> (f64, f64, f64) {
    let ev = Units::HARTREE_TO_EV * 1000.0 / particles;
    let c0 = reports[0].conserved;
    let worst = reports
        .iter()
        .map(|r| (r.conserved - c0).abs())
        .fold(0.0_f64, f64::max)
        * ev;
    let n = reports.len() as f64;
    let (tm, cm) = reports.iter().fold((0.0, 0.0), |(a, b), r| {
        (a + r.time_fs() / n, b + r.conserved / n)
    });
    let (num, den) = reports.iter().fold((0.0, 0.0), |(a, b), r| {
        let dt = r.time_fs() - tm;
        (a + dt * (r.conserved - cm), b + dt * dt)
    });
    let slope = if den > 0.0 {
        num / den * 1000.0 * ev
    } else {
        0.0
    };
    let t = reports[1..].iter().map(|r| r.temperature).sum::<f64>() / (n - 1.0).max(1.0);
    (worst, slope, t)
}

fn nve_row(
    name: &str,
    calc: &dyn Calculator,
    start: &MolecularSystem,
    temperature: f64,
    dt_fs: f64,
    total_fs: f64,
) -> MoptResult<()> {
    let settings = MdSettings {
        time_step: dt_fs * Units::FS_TO_AU_TIME,
        steps: (total_fs / dt_fs).round() as usize,
        temperature,
        seed: 7,
        rigid: if start.has_pbc() {
            RigidMotion::Translation
        } else {
            RigidMotion::TranslationRotation
        },
        ..MdSettings::default()
    };
    let reports = dynamics(calc, start, settings)?;
    let (worst, slope, t) = conservation(&reports, start.n_atoms() as f64);
    println!(
        "| {name} | {dt_fs} | {:.4} | {:.4} | {:.1} |",
        worst, slope, t
    );
    Ok(())
}

fn nve() -> MoptResult<()> {
    println!("| system | time step (fs) | largest deviation (meV/atom) | drift (meV/atom/ps) | mean T (K) |");
    println!("|---|---|---|---|---|");
    let ar = ar13()?;
    let lj = calculator("lj", &ar)?;
    let ar = relaxed(lj.as_ref(), &ar)?;
    for dt in [2.0, 5.0, 10.0, 20.0, 40.0] {
        nve_row("Ar13, lj, 30 K", lj.as_ref(), &ar, 30.0, dt, 20_000.0)?;
    }
    let cu = copper(3.60)?;
    let emt = calculator("emt", &cu)?;
    for dt in [0.5, 1.0, 2.0, 4.0, 8.0] {
        nve_row(
            "Cu32 cell, emt, 300 K",
            emt.as_ref(),
            &cu,
            300.0,
            dt,
            4000.0,
        )?;
    }
    let mol = ethanol()?;
    let ff = calculator("gfnff", &mol)?;
    let mol = relaxed(ff.as_ref(), &mol)?;
    for dt in [0.1, 0.25, 0.5, 1.0, 2.0] {
        nve_row(
            "ethanol, gfnff, 300 K",
            ff.as_ref(),
            &mol,
            300.0,
            dt,
            2000.0,
        )?;
    }
    mb_nve()
}

fn mb_nve() -> MoptResult<()> {
    let objective = MullerBrown2D::default();
    println!("\n| Muller-Brown, unit masses, kT = 1e-3 | time step | largest deviation | drift per unit time |");
    println!("|---|---|---|---|");
    for dt in [0.01, 0.05, 0.1, 0.2, 0.4] {
        let settings = MdSettings {
            time_step: dt,
            steps: (200.0 / dt) as usize,
            temperature: 1e-3 / Units::BOLTZMANN_HARTREE_PER_K,
            seed: 3,
            ..MdSettings::default()
        };
        let x0 = Col::from_fn(2, |i| [-0.558_224, 1.441_726][i]);
        let mut md = MolecularDynamics::new(settings, MdSpace::variables(2)?, x0, None)?;
        let mut reports = Vec::new();
        let mut keep = FnMdObserver::new(|r: &MdStepReport| {
            reports.push((r.time, r.conserved));
            Ok(true)
        });
        run_md(&mut md, &ObjectiveMdSurface(&objective), &mut [&mut keep])?;
        let c0 = reports[0].1;
        let worst = reports
            .iter()
            .map(|(_, c)| (c - c0).abs())
            .fold(0.0_f64, f64::max);
        let last = reports
            .last()
            .map_or(0.0, |(t, c)| (c - c0) / t.max(1e-300));
        println!("| | {dt} | {worst:.3e} | {last:.3e} |");
    }
    Ok(())
}

fn temperature_stats(reports: &[MdStepReport], skip: usize) -> (f64, f64, f64) {
    let t: Vec<f64> = reports[skip..].iter().map(|r| r.temperature).collect();
    let n = t.len() as f64;
    let mean = t.iter().sum::<f64>() / n;
    let var = t.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    (mean, var, reports[skip].dof)
}

/// Time until the running mean of the temperature over 50 fs is within 5 % of the target
/// after a start at a tenth of it.
fn equilibration(reports: &[MdStepReport], target: f64, dt_fs: f64) -> f64 {
    let window = ((50.0 / dt_fs) as usize).max(1);
    for k in window..reports.len() {
        let mean = reports[k - window..k]
            .iter()
            .map(|r| r.temperature)
            .sum::<f64>()
            / window as f64;
        if (mean / target - 1.0).abs() < 0.05 {
            return reports[k].time_fs();
        }
    }
    f64::NAN
}

#[allow(clippy::too_many_arguments)]
fn thermostat_row(
    name: &str,
    calc: &dyn Calculator,
    start: &MolecularSystem,
    thermostat: &str,
    tau: f64,
    dt: f64,
    steps: usize,
    target: f64,
) -> MoptResult<()> {
    let integrator = if thermostat == "langevin" {
        "baoab"
    } else {
        "velocity_verlet"
    };
    let settings = MdSettings {
        time_step: dt * Units::FS_TO_AU_TIME,
        steps,
        temperature: 0.1 * target,
        schedule: vec![(0, target)],
        thermostat: thermostat.to_string(),
        integrator: integrator.to_string(),
        thermostat_settings: ThermostatSettings {
            tau: tau * Units::FS_TO_AU_TIME,
            ..ThermostatSettings::default()
        },
        seed: 21,
        rigid: RigidMotion::TranslationRotation,
        velocities: InitialVelocities::Maxwell { exact: true },
        ..MdSettings::default()
    };
    let reports = match dynamics(calc, start, settings) {
        Ok(reports) => reports,
        Err(e) => {
            println!("| {name} | {thermostat} | {tau} | failed: {e} | | |");
            return Ok(());
        }
    };
    let (mean, var, dof) = temperature_stats(&reports, steps / 5);
    let canonical = 2.0 * target * target / dof;
    println!(
        "| {name} | {thermostat} | {tau} | {:.1} | {:.3} | {:.0} |",
        mean,
        var / canonical,
        equilibration(&reports, target, dt)
    );
    Ok(())
}

fn thermostats() -> MoptResult<()> {
    println!(
        "| system | thermostat | tau (fs) | mean T (K) | var(T) / canonical | time to 5 % (fs) |"
    );
    println!("|---|---|---|---|---|---|");
    let ar = ar13()?;
    let lj = calculator("lj", &ar)?;
    let ar = relaxed(lj.as_ref(), &ar)?;
    for thermostat in [
        "berendsen",
        "bussi",
        "langevin",
        "andersen",
        "nose_hoover",
        "nose_hoover_chain",
    ] {
        for tau in [20.0, 100.0, 500.0] {
            thermostat_row(
                "Ar13, lj, 30 K",
                lj.as_ref(),
                &ar,
                thermostat,
                tau,
                10.0,
                60_000,
                30.0,
            )?;
        }
    }
    let mol = ethanol()?;
    let ff = calculator("gfnff", &mol)?;
    let mol = relaxed(ff.as_ref(), &mol)?;
    for thermostat in [
        "berendsen",
        "bussi",
        "langevin",
        "andersen",
        "nose_hoover",
        "nose_hoover_chain",
    ] {
        for tau in [20.0, 100.0, 500.0] {
            thermostat_row(
                "ethanol, gfnff, 300 K",
                ff.as_ref(),
                &mol,
                thermostat,
                tau,
                0.5,
                60_000,
                300.0,
            )?;
        }
    }
    Ok(())
}

fn npt() -> MoptResult<()> {
    let cu = copper(3.60)?;
    let emt = calculator("emt", &cu)?;
    // The bulk modulus of the model from the pressure at two volumes.
    let pressure_at = |a: f64| -> MoptResult<(f64, f64)> {
        let sys = copper(a)?;
        let out = emt.evaluate(&sys, &mopt_rs::domain::EvalRequest::ENERGY_GRAD_STRESS)?;
        let s = out.stress.ok_or_else(|| MoptError::calc("no stress"))?;
        let v = sys.active_cell().map_or(0.0, PeriodicCell::volume_bohr3);
        Ok((-(s[0][0] + s[1][1] + s[2][2]) / 3.0, v))
    };
    let ((p1, v1), (p2, v2)) = (pressure_at(3.58)?, pressure_at(3.62)?);
    let bulk = -0.5 * (v1 + v2) * (p2 - p1) / (v2 - v1);
    println!(
        "EMT Cu: bulk modulus {:.1} GPa from the pressure at two volumes",
        bulk * gigapascal_per_atomic_unit()
    );
    println!("| barostat | tau_p (fs) | compressibility (1/GPa) | seed | mean V (ang.^3) | var(V) / (kT V / B) | mean P (GPa) | largest mean P_ab (GPa) |");
    println!("|---|---|---|---|---|---|---|---|");
    let compressibility = 1.0 / (bulk * gigapascal_per_atomic_unit());
    let only = std::env::args().nth(2);
    let masses: Vec<f64> = cu
        .masses_amu()
        .iter()
        .map(|m| m * Units::AMU_TO_AU_MASS)
        .collect();
    for (barostat, thermostat) in [
        ("berendsen", "bussi"),
        ("crescale", "bussi"),
        ("mtk", "nose_hoover_chain"),
        ("berendsen_aniso", "bussi"),
        ("crescale_aniso", "bussi"),
        ("mtk_aniso", "nose_hoover_chain"),
    ] {
        if only.as_deref().is_some_and(|o| o != barostat) {
            continue;
        }
        for (tau_p, beta, seed) in [
            (500.0, compressibility, 5),
            (500.0, compressibility, 6),
            (1000.0, compressibility, 5),
            (1000.0, 0.01, 5),
        ] {
            let mut settings = MdSettings {
                time_step: 4.0 * Units::FS_TO_AU_TIME,
                steps: 25_000,
                temperature: 300.0,
                thermostat: thermostat.to_string(),
                barostat: Some(barostat.to_string()),
                seed,
                rigid: RigidMotion::Translation,
                pressure: 0.0,
                ..MdSettings::default()
            };
            settings.barostat_settings.tau = tau_p * Units::FS_TO_AU_TIME;
            settings.barostat_settings.compressibility = beta * gigapascal_per_atomic_unit();
            let reports = dynamics(emt.as_ref(), &cu, settings)?;
            let tail = &reports[5000..];
            let n = tail.len() as f64;
            let volumes: Vec<f64> = tail.iter().filter_map(|r| r.volume).collect();
            let mean = volumes.iter().sum::<f64>() / n;
            let var = volumes.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
            let kt = Units::BOLTZMANN_HARTREE_PER_K * 300.0;
            let expected = kt * mean / bulk;
            let p = tail.iter().filter_map(|r| r.pressure).sum::<f64>() / n;
            // Standard error of the mean pressure from 20 blocks.
            let blocks: Vec<f64> = tail
                .chunks(tail.len() / 20)
                .map(|b| b.iter().filter_map(|r| r.pressure).sum::<f64>() / b.len() as f64)
                .collect();
            let m = blocks.len() as f64;
            let bm = blocks.iter().sum::<f64>() / m;
            let error =
                (blocks.iter().map(|b| (b - bm).powi(2)).sum::<f64>() / (m * (m - 1.0))).sqrt();
            let gpa = gigapascal_per_atomic_unit();
            // The largest component of the mean pressure tensor K / V - sigma (the target
            // is zero).
            let mut tensor = [[0.0; 3]; 3];
            for r in tail {
                let (Some(s), Some(v)) = (r.stress, r.volume) else {
                    continue;
                };
                for (a, row) in tensor.iter_mut().enumerate() {
                    for (b, x) in row.iter_mut().enumerate() {
                        let k: f64 = masses
                            .iter()
                            .enumerate()
                            .map(|(i, m)| m * r.v[3 * i + a] * r.v[3 * i + b])
                            .sum();
                        *x += (k / v - s[a][b]) / n;
                    }
                }
            }
            let largest = tensor.iter().flatten().fold(0.0_f64, |m, x| m.max(x.abs()));
            println!(
                "| {barostat} | {tau_p} | {beta:.4} | {seed} | {:.2} | {:.3} | {:.3} +- {:.3} | {:.3} |",
                mean * Units::BOHR_TO_ANGSTROM.powi(3),
                var / expected,
                p * gpa,
                error * gpa,
                largest * gpa
            );
        }
    }
    Ok(())
}

fn shake() -> MoptResult<()> {
    let mol = ethanol()?;
    let ff = calculator("gfnff", &mol)?;
    let mol = relaxed(ff.as_ref(), &mol)?;
    // The bonds to hydrogen held.
    let bonds: Vec<Constraint> = [(0, 4), (0, 5), (0, 6), (1, 7), (1, 8), (2, 3)]
        .iter()
        .map(|&(a, b)| {
            Constraint::at_start(ConstraintKind::Distance(
                vec![mopt_rs::domain::AtomId::new(a)],
                vec![mopt_rs::domain::AtomId::new(b)],
            ))
        })
        .collect();
    println!("| ethanol, gfnff, X-H bonds held | solver | time step (fs) | largest residual (Bohr) | largest deviation (meV/atom) | drift (meV/atom/ps) | time per step (us) |");
    println!("|---|---|---|---|---|---|---|");
    let solvers = [
        ("SHAKE/RATTLE", ConstraintSolver::Shake),
        (
            "LINCS (order 4, 1 correction)",
            ConstraintSolver::Lincs {
                order: 4,
                iterations: 1,
            },
        ),
        (
            "LINCS (order 8, 2 corrections)",
            ConstraintSolver::Lincs {
                order: 8,
                iterations: 2,
            },
        ),
    ];
    for (name, solver) in solvers {
        for dt in [0.5, 1.0, 2.0] {
            let settings = MdSettings {
                time_step: dt * Units::FS_TO_AU_TIME,
                steps: (2000.0 / dt) as usize,
                temperature: 300.0,
                seed: 13,
                rigid: RigidMotion::TranslationRotation,
                constraints: bonds.clone(),
                constraint_solver: solver,
                ..MdSettings::default()
            };
            let steps = settings.steps as f64;
            let clock = std::time::Instant::now();
            let reports = dynamics(ff.as_ref(), &mol, settings)?;
            let micro = clock.elapsed().as_secs_f64() * 1e6 / steps;
            let residual = reports
                .iter()
                .filter_map(|r| r.constraint_residual)
                .fold(0.0_f64, f64::max);
            let (worst, slope, _) = conservation(&reports, mol.n_atoms() as f64);
            println!("| | {name} | {dt} | {residual:.2e} | {worst:.4} | {slope:.4} | {micro:.0} |");
        }
    }
    Ok(())
}

fn main() -> MoptResult<()> {
    let what = std::env::args().nth(1).unwrap_or_else(|| "all".to_string());
    match what.as_str() {
        "nve" => nve(),
        "thermostats" => thermostats(),
        "npt" => npt(),
        "shake" => shake(),
        "all" => {
            nve()?;
            thermostats()?;
            npt()?;
            shake()
        }
        other => Err(MoptError::config(format!(
            "unknown measurement '{other}' (nve, thermostats, npt, shake, all)"
        ))),
    }
}
