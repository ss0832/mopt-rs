//! Transport coefficients of the Lennard-Jones fluid at `rho* = 0.6`, `T* = 1.35` with the
//! cutoff `2.5 sigma`: self-diffusion (Einstein and Green-Kubo), thermal conductivity
//! (Green-Kubo of the heat current and Muller-Plathe's reverse non-equilibrium method), and
//! shear viscosity (Green-Kubo), in SI and reduced units. The thermal conductivity there is
//! about 3.4 in reduced units in the simulations of Evans (Phys. Rev. A 1986, 34, 1449,
//! Fig. 1). The tables go to `docs/source/molecular_dynamics.md`.
//!
//! ```text
//! cargo run --release -p mopt-rs --example md_transport -- STEPS [ATOMS_PER_EDGE]
//! ```
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::path::{Path, PathBuf};
use std::time::Instant;

use mopt_rs::core_infra::log::{Level, Logger};
use mopt_rs::domain::{positions, Angstrom, Element, MolecularSystem, PeriodicCell, Quantity};
use mopt_rs::domain::{SpinCharge, Units};
use mopt_rs::error::{MoptError, MoptResult};
use mopt_rs::md::settings::Exchange;
use mopt_rs::md::{InitialVelocities, MdSettings, MdSpace, MdStepReport, MolecularDynamics};
use mopt_rs::md::{RigidMotion, ThermostatSettings};
use mopt_rs::pes::calculator::Calculator;
use mopt_rs::pes::native::TruncatedLennardJones;
use mopt_rs::workflow::md::{run_md, FnMdObserver, MdObserver, MoleculeMdSurface};
use mopt_rs::workflow::md_analysis::{analyze_files, Analysis, AnalysisSettings, RunFiles};
use mopt_rs::workflow::md_files::{MdFiles, MdLayout};
use mopt_rs::workflow::md_flux::HeatFlux;

/// UFF argon of the built-in `lj`: sigma (Angstrom), epsilon (K).
const SIGMA: f64 = 3.817 / 1.122_462_048_309_373;
const EPSILON_K: f64 = 0.237 / Units::BOLTZMANN_KCAL_MOL_K;
const MASS_AMU: f64 = 39.948;

fn fluid(edge: usize, density: f64) -> MoptResult<MolecularSystem> {
    let n = edge * edge * edge;
    let side = (n as f64 / density).cbrt() * SIGMA;
    let a = side / edge as f64;
    let mut rows = Vec::with_capacity(n);
    for i in 0..edge {
        for j in 0..edge {
            for k in 0..edge {
                rows.push([
                    (i as f64 + 0.25) * a,
                    (j as f64 + 0.25) * a,
                    (k as f64 + 0.25) * a,
                ]);
            }
        }
    }
    let l = Quantity::<Angstrom>::new(side);
    MolecularSystem::new(
        vec![Element::parse_token("Ar")?; n],
        &positions::<Angstrom>(&rows),
        SpinCharge::default(),
        Some(PeriodicCell::orthorhombic(l, l, l)?),
    )
}

/// A run of `settings` from `start`; the last report. With `dir`, the files of the run
/// (and the heat current) are written there.
fn run(
    calc: &dyn Calculator,
    start: &MolecularSystem,
    settings: MdSettings,
    dir: Option<&Path>,
) -> MoptResult<MdStepReport> {
    let space = MdSpace::atoms(&start.masses_amu())?;
    let mut md = MolecularDynamics::new(
        settings,
        space,
        start.coords_flat_bohr(),
        Some(start.clone()),
    )?;
    let surface = MoleculeMdSurface::with_terms(calc, start.clone(), Vec::new())?;
    let mut last = None;
    {
        let mut keep = FnMdObserver::new(|r: &MdStepReport| {
            last = Some(r.clone());
            Ok(true)
        });
        match dir {
            Some(d) => {
                let layout = MdLayout::Atoms(Box::new(start.clone()));
                let mut files = MdFiles::new(d, "lj", layout, 10)?;
                let mut flux = HeatFlux::new(calc, start, d, "lj")?;
                let mut observers: Vec<&mut dyn MdObserver> =
                    vec![&mut files, &mut flux, &mut keep];
                run_md(&mut md, &surface, &mut observers)?;
            }
            None => {
                run_md(&mut md, &surface, &mut [&mut keep])?;
            }
        }
    }
    last.ok_or_else(|| MoptError::config("the run made no step"))
}

fn main() -> MoptResult<()> {
    let mut args = std::env::args().skip(1);
    let steps: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(20000);
    let edge: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(8);
    let temperature = 1.35 * EPSILON_K;
    let start = fluid(edge, 0.6)?;
    let cutoff = Quantity::<Angstrom>::new(2.5 * SIGMA);
    let calc = TruncatedLennardJones::new(cutoff.to(), Some(Quantity::<Angstrom>::new(0.3).to()))?;
    let base = std::env::temp_dir().join(format!("mopt_transport_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&base);
    let dt = 10.0 * Units::FS_TO_AU_TIME;
    let clock = Instant::now();
    // Equilibration at T* = 1.35 (Bussi, 1 ps), then the runs in NVE from its last state.
    let equilibrated = run(
        &calc,
        &start,
        MdSettings {
            time_step: dt,
            steps: 3000,
            temperature,
            thermostat: "bussi".into(),
            thermostat_settings: ThermostatSettings {
                tau: 1000.0 * Units::FS_TO_AU_TIME,
                ..ThermostatSettings::default()
            },
            rigid: RigidMotion::Translation,
            seed: 3,
            ..MdSettings::default()
        },
        None,
    )?;
    let mut state = start.with_flat_coords_bohr(&equilibrated.x)?;
    state.cell.clone_from(&equilibrated.cell);
    let nve = |extra: Option<Exchange>| MdSettings {
        time_step: dt,
        steps,
        temperature,
        velocities: InitialVelocities::Given(equilibrated.v.clone()),
        rigid: RigidMotion::Translation,
        stress: true,
        exchange: extra,
        seed: 4,
        ..MdSettings::default()
    };
    let analyze = |dir: &Path, skip: f64| -> MoptResult<()> {
        let files = RunFiles::locate(&dir.join("lj_traj.xyz"))?;
        let settings = AnalysisSettings {
            analyses: vec![Analysis::Statistics, Analysis::Transport],
            skip_fs: skip,
            ..AnalysisSettings::default()
        };
        let log = Logger::console(Level::Info);
        analyze_files(&files, &settings, &dir.join("trajectory_analysis"), &log)?;
        Ok(())
    };
    let gk: PathBuf = base.join("green_kubo");
    run(&calc, &state, nve(None), Some(&gk))?;
    analyze(&gk, 0.0)?;
    let nemd: PathBuf = base.join("reverse");
    run(
        &calc,
        &state,
        nve(Some(Exchange {
            every: 100,
            slabs: 20,
            axis: 2,
        })),
        Some(&nemd),
    )?;
    analyze(&nemd, 0.2 * steps as f64 * 10.0)?;
    // Reduced units.
    let kb = Units::BOLTZMANN_J_PER_K;
    let sigma_m = SIGMA * 1e-10;
    let eps_j = EPSILON_K * kb;
    let mass = MASS_AMU * Units::AMU_TO_KG;
    let tau_s = sigma_m * (mass / eps_j).sqrt();
    println!(
        "reduced units: kappa* = kappa [W/(m K)] x {:.6}, eta* = eta [Pa s] x {:.6}, D* = D \
         [ang^2/ps] x {:.6}",
        sigma_m * sigma_m / kb * (mass / eps_j).sqrt(),
        sigma_m * sigma_m / (mass * eps_j).sqrt(),
        1e-8 * tau_s / (sigma_m * sigma_m) * 1.0,
    );
    println!(
        "{} atoms, {steps} steps of 10 fs (dt* = {:.5}), {:.0} s; files in {}",
        start.n_atoms(),
        10e-15 / tau_s,
        clock.elapsed().as_secs_f64(),
        base.display()
    );
    Ok(())
}
