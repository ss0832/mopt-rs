//! Measurements of molecular dynamics in generalized coordinates (`mdmain -coord`): the
//! conservation of the energy against Cartesian dynamics at the same time step, the cost,
//! and the distributions of a bond length and an angle at constant temperature. The tables
//! go to `docs/source/molecular_dynamics.md`.
//!
//! ```text
//! cargo run --release -p mopt-rs --example md_coordinates -- nve|nvt|all
//! ```
//!
//! Systems: water, hydrogen cyanide (linear), and ethanol with GFN-FF, each relaxed first.
//! Coordinates: Cartesian (velocity Verlet with the rigid motions taken out), delocalized
//! internal coordinates (`dic`), and spherical coordinates of every atom about a point
//! outside the molecule (a full curvilinear set defined here).
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::f64::consts::PI;
use std::time::Instant;

use faer::{Col, Mat};
use mopt_rs::analysis::trajectory::statistics;
use mopt_rs::coord::system::{CoordinateRegistry, CoordinateSystem};
use mopt_rs::domain::{positions, Angstrom, Element, MolecularSystem, SpinCharge, Units};
use mopt_rs::error::{MoptError, MoptResult};
use mopt_rs::md::generalized::GeneralizedDynamics;
use mopt_rs::md::{MdMachine, MdSettings, MdSpace, MdStepReport, MolecularDynamics, RigidMotion};
use mopt_rs::opt::state_machine::{GeometryOptimizer, OptimizerSettings};
use mopt_rs::pes::calculator::Calculator;
use mopt_rs::pes::registry::{CalculatorOptions, CalculatorRegistry};
use mopt_rs::workflow::md::{run_md, FnMdObserver, MoleculeMdSurface};
use mopt_rs::workflow::run_optimization;

/// Spherical coordinates `(r, theta, phi)` of every atom about a fixed point: `3N`
/// coordinates that cover all motions, curvilinear everywhere.
struct AtomSpherical {
    origin: [f64; 3],
}

impl AtomSpherical {
    fn of(&self, p: [f64; 3]) -> ([f64; 3], [[f64; 3]; 3]) {
        let d = [
            p[0] - self.origin[0],
            p[1] - self.origin[1],
            p[2] - self.origin[2],
        ];
        let rho2 = d[0] * d[0] + d[1] * d[1];
        let r2 = rho2 + d[2] * d[2];
        let (r, rho) = (r2.sqrt(), rho2.sqrt());
        let values = [r, (d[2] / r).acos(), d[1].atan2(d[0])];
        let rows = [
            [d[0] / r, d[1] / r, d[2] / r],
            [
                d[0] * d[2] / (r2 * rho),
                d[1] * d[2] / (r2 * rho),
                -rho / r2,
            ],
            [-d[1] / rho2, d[0] / rho2, 0.0],
        ];
        (values, rows)
    }
}

impl CoordinateSystem for AtomSpherical {
    fn name(&self) -> &str {
        "spherical_atoms"
    }
    fn initialize(&mut self, _sys: &MolecularSystem) -> MoptResult<()> {
        Ok(())
    }
    fn dim(&self, sys: &MolecularSystem) -> usize {
        3 * sys.n_atoms()
    }
    fn values(&self, sys: &MolecularSystem) -> MoptResult<Col<f64>> {
        let x = sys.coords_flat_bohr();
        Ok(Col::from_fn(x.nrows(), |k| {
            let i = k / 3;
            self.of([x[3 * i], x[3 * i + 1], x[3 * i + 2]]).0[k % 3]
        }))
    }
    fn wilson_b(&self, sys: &MolecularSystem) -> MoptResult<Mat<f64>> {
        let x = sys.coords_flat_bohr();
        let n = x.nrows();
        let mut b = Mat::zeros(n, n);
        for i in 0..n / 3 {
            let (_, rows) = self.of([x[3 * i], x[3 * i + 1], x[3 * i + 2]]);
            for (a, row) in rows.iter().enumerate() {
                for c in 0..3 {
                    b[(3 * i + a, 3 * i + c)] = row[c];
                }
            }
        }
        Ok(b)
    }
    fn displacement(&self, to: &MolecularSystem, from: &MolecularSystem) -> MoptResult<Col<f64>> {
        let d = self.values(to)? - self.values(from)?;
        Ok(Col::from_fn(d.nrows(), |k| {
            if k % 3 == 2 {
                (d[k] + PI).rem_euclid(2.0 * PI) - PI
            } else {
                d[k]
            }
        }))
    }
}

fn structure(symbols: &[&str], rows: &[[f64; 3]]) -> MoptResult<MolecularSystem> {
    let elements = symbols
        .iter()
        .map(|s| Element::parse_token(s))
        .collect::<MoptResult<Vec<_>>>()?;
    MolecularSystem::new(
        elements,
        &positions::<Angstrom>(rows),
        SpinCharge::default(),
        None,
    )
}

fn calculator(sys: &MolecularSystem) -> MoptResult<Box<dyn Calculator>> {
    CalculatorRegistry::native().create("gfnff", &CalculatorOptions::default(), sys)
}

fn relaxed(calc: &dyn Calculator, sys: &MolecularSystem) -> MoptResult<MolecularSystem> {
    let settings = OptimizerSettings {
        max_iterations: 300,
        rigid_body_invariant: true,
        ..OptimizerSettings::default()
    };
    let mut opt = GeometryOptimizer::new(settings, sys.clone())?;
    Ok(run_optimization(&mut opt, calc, &mut [])?.system)
}

fn molecules() -> MoptResult<Vec<(&'static str, MolecularSystem)>> {
    Ok(vec![
        (
            "water",
            structure(
                &["O", "H", "H"],
                &[[0.0, 0.0, 0.12], [0.0, 0.76, -0.47], [0.0, -0.76, -0.47]],
            )?,
        ),
        (
            "HCN",
            structure(
                &["H", "C", "N"],
                &[[0.0, 0.0, -1.07], [0.0, 0.0, 0.0], [0.0, 0.0, 1.16]],
            )?,
        ),
        (
            "ethanol",
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
            )?,
        ),
    ])
}

/// The coordinates of a case (`None`: Cartesian dynamics).
fn coordinates(name: &str, sys: &MolecularSystem) -> MoptResult<Option<Box<dyn CoordinateSystem>>> {
    match name {
        "cart" => Ok(None),
        "spherical_atoms" => {
            // A point off every axis of the molecule, 20 Bohr away.
            let c = sys.centroid().value();
            Ok(Some(Box::new(AtomSpherical {
                origin: [c[0] - 11.0, c[1] - 13.0, c[2] - 9.0],
            })))
        }
        other => CoordinateRegistry::default().create(other).map(Some),
    }
}

/// A run; its reports, the wall time per step (microseconds), and the evaluations of the
/// Wilson matrix per step.
fn run(
    calc: &dyn Calculator,
    start: &MolecularSystem,
    coords: Option<Box<dyn CoordinateSystem>>,
    settings: MdSettings,
) -> MoptResult<(Vec<MdStepReport>, f64, f64)> {
    let surface = MoleculeMdSurface::with_terms(calc, start.clone(), Vec::new())?;
    let mut reports = Vec::new();
    let steps = settings.steps as f64;
    let clock = Instant::now();
    let mut b_per_step = 0.0;
    {
        let mut keep = FnMdObserver::new(|r: &MdStepReport| {
            reports.push(r.clone());
            Ok(true)
        });
        match coords {
            None => {
                let space = MdSpace::atoms(&start.masses_amu())?;
                let mut md = MolecularDynamics::new(
                    settings,
                    space,
                    start.coords_flat_bohr(),
                    Some(start.clone()),
                )?;
                run_md(&mut md, &surface, &mut [&mut keep])?;
            }
            Some(c) => {
                let mut md = GeneralizedDynamics::new(settings, c, start.clone(), true)?;
                run_md(&mut md, &surface, &mut [&mut keep])?;
                b_per_step = md.wilson_evaluations() as f64 / steps;
                debug_assert!(md.finished());
            }
        }
    }
    let micro = clock.elapsed().as_secs_f64() * 1e6 / steps;
    Ok((reports, micro, b_per_step))
}

fn settings(dt_fs: f64, total_fs: f64, thermostat: &str) -> MdSettings {
    MdSettings {
        time_step: dt_fs * Units::FS_TO_AU_TIME,
        steps: (total_fs / dt_fs).round() as usize,
        temperature: 300.0,
        seed: 7,
        thermostat: thermostat.to_string(),
        rigid: RigidMotion::TranslationRotation,
        ..MdSettings::default()
    }
}

fn worst_deviation(reports: &[MdStepReport], atoms: f64) -> f64 {
    let c0 = reports[0].conserved;
    reports
        .iter()
        .map(|r| (r.conserved - c0).abs())
        .fold(0.0_f64, f64::max)
        * Units::HARTREE_TO_EV
        * 1000.0
        / atoms
}

fn nve() -> MoptResult<()> {
    println!(
        "| molecule | coordinates | time step (fs) | largest deviation (meV/atom) | time per step \
         (us) | B per step |"
    );
    println!("|---|---|---|---|---|---|");
    for (name, sys) in molecules()? {
        let calc = calculator(&sys)?;
        let start = relaxed(calc.as_ref(), &sys)?;
        for coords in ["cart", "dic", "tric", "ric", "spherical_atoms"] {
            for dt in [0.25, 0.5, 1.0] {
                let c = coordinates(coords, &start)?;
                match run(calc.as_ref(), &start, c, settings(dt, 2000.0, "none")) {
                    Ok((reports, micro, b)) => println!(
                        "| {name} | {coords} | {dt} | {:.4} | {micro:.0} | {b:.1} |",
                        worst_deviation(&reports, start.n_atoms() as f64)
                    ),
                    Err(e) => println!("| {name} | {coords} | {dt} | failed: {e} | | |"),
                }
            }
        }
    }
    Ok(())
}

fn distance(r: &MdStepReport, i: usize, j: usize) -> f64 {
    let d: f64 = (0..3)
        .map(|c| (r.x[3 * i + c] - r.x[3 * j + c]).powi(2))
        .sum();
    d.sqrt() * Units::BOHR_TO_ANGSTROM
}

fn angle(r: &MdStepReport, i: usize, j: usize, k: usize) -> f64 {
    let a: Vec<f64> = (0..3).map(|c| r.x[3 * i + c] - r.x[3 * j + c]).collect();
    let b: Vec<f64> = (0..3).map(|c| r.x[3 * k + c] - r.x[3 * j + c]).collect();
    let dot: f64 = a.iter().zip(&b).map(|(p, q)| p * q).sum();
    let na: f64 = a.iter().map(|p| p * p).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|p| p * p).sum::<f64>().sqrt();
    (dot / (na * nb)).clamp(-1.0, 1.0).acos() * Units::RAD_TO_DEG
}

fn summary(values: &[f64]) -> MoptResult<String> {
    let s = statistics(values)?;
    Ok(format!(
        "{:.5} +- {:.5} (sd {:.4})",
        s.mean, s.error, s.deviation
    ))
}

fn nvt() -> MoptResult<()> {
    println!(
        "| coordinates | thermostat | r(O-H) (ang.) | angle H-O-H (deg.) | T (K) | time per step (us) |"
    );
    println!("|---|---|---|---|---|---|");
    let Some((_, sys)) = molecules()?.into_iter().next() else {
        return Err(MoptError::config("no molecule"));
    };
    let calc = calculator(&sys)?;
    let start = relaxed(calc.as_ref(), &sys)?;
    for thermostat in ["bussi", "langevin"] {
        for coords in ["cart", "dic", "spherical_atoms"] {
            let c = coordinates(coords, &start)?;
            let mut s = settings(0.5, 40000.0, thermostat);
            if thermostat == "langevin" && coords == "cart" {
                s.rigid = RigidMotion::None;
            }
            let (reports, micro, _) = run(calc.as_ref(), &start, c, s)?;
            // The first 4 ps are left out.
            let kept = &reports[8000..];
            let r: Vec<f64> = kept.iter().map(|r| distance(r, 0, 1)).collect();
            let a: Vec<f64> = kept.iter().map(|r| angle(r, 1, 0, 2)).collect();
            let t: Vec<f64> = kept.iter().map(|r| r.temperature).collect();
            println!(
                "| {coords} | {thermostat} | {} | {} | {} | {micro:.0} |",
                summary(&r)?,
                summary(&a)?,
                summary(&t)?
            );
        }
    }
    Ok(())
}

fn main() -> MoptResult<()> {
    let what = std::env::args().nth(1).unwrap_or_else(|| "all".to_string());
    match what.as_str() {
        "nve" => nve(),
        "nvt" => nvt(),
        "all" => {
            nve()?;
            nvt()
        }
        other => Err(MoptError::config(format!(
            "unknown measurement '{other}' (nve, nvt, all)"
        ))),
    }
}
