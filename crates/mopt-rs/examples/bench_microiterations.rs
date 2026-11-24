//! The relaxations of the environment of ONIOM microiterations, replayed along the core
//! structures of a run: each setting relaxes the environment at the same sequence of core
//! structures (those of an optimization trajectory), so that the cost of the relaxations
//! (gradients, time) is compared without the outer optimization taking other steps.
//!
//! ```text
//! cargo run --release -p mopt-rs --example bench_microiterations -- \
//!     TRAJECTORY.xyz "CALCULATOR OPTIONS" [hessian_every=N] SETTING ...
//! ```
//!
//! `TRAJECTORY.xyz` holds full structures (the trajectory of an `optmain` run with
//! microiterations); the environment of the first frame is the start. `CALCULATOR OPTIONS`
//! are those of `optmain` with `-oniom` (for example `-os gfn1xtb -oniom 1-14 none gfnff`).
//! Each `SETTING` is a list of `-oniom_opt` settings separated by commas (`micro_predict=on,
//! micro_coord=fragments`), or `default`. `hessian_every=N` asks for the Hessian of the core
//! at every Nth frame, as `-fc N` does (0, the default: never). The first setting is the
//! reference of the energies and gradients of the others.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::time::Instant;

use mopt_rs::cli::calculator::{calculator_name, calculator_options};
use mopt_rs::cli::oniom_micro::wrap;
use mopt_rs::cli::option::parser;
use mopt_rs::cli::spec::OPTMAIN;
use mopt_rs::domain::{EvalRequest, SpinCharge};
use mopt_rs::error::{MoptError, MoptResult};
use mopt_rs::io::xyz::read_xyz_trajectory_file;
use mopt_rs::math::linalg::col_max_abs;
use mopt_rs::pes::registry::CalculatorRegistry;

/// What one setting did along the trajectory.
struct Outcome {
    name: String,
    seconds: f64,
    gradients: usize,
    energies: Vec<f64>,
    gradients_of_core: Vec<faer::Col<f64>>,
    counts: mopt_rs::pes::oniom::micro::MicroCounts,
}

fn run(
    trajectory: &str,
    frames: &[mopt_rs::domain::MolecularSystem],
    calculator: &[String],
    setting: &str,
    hessian_every: usize,
) -> MoptResult<Outcome> {
    let mut argv = vec![trajectory.to_string()];
    argv.extend(calculator.iter().cloned());
    if setting != "default" {
        argv.push("-oniom_opt".to_string());
        argv.extend(setting.split(',').map(|s| s.trim().to_string()));
    }
    let p = parser("optmain", OPTMAIN).parse(&argv)?;
    let options = calculator_options(&p)?;
    let registry = CalculatorRegistry::native();
    let first = frames
        .first()
        .ok_or_else(|| MoptError::config("the trajectory has no frames"))?
        .clone();
    let wrapped = wrap(&options, None, first, &calculator_name(&p)?, &registry)?;
    let relaxed = wrapped
        .micro
        .ok_or_else(|| MoptError::config("the calculator options give no microiterations"))?;
    if let Some(note) = &wrapped.note {
        println!("{setting}: {note}");
    }
    let start = Instant::now();
    let mut energies = Vec::new();
    let mut gradients_of_core = Vec::new();
    for (k, frame) in frames.iter().enumerate() {
        let core = relaxed.core_of(frame)?;
        let request = if hessian_every > 0 && k % hessian_every == 0 {
            EvalRequest::ENERGY_GRAD_HESS
        } else {
            EvalRequest::ENERGY_GRAD
        };
        let out =
            mopt_rs::pes::calculator::Calculator::evaluate(relaxed.as_ref(), &core, &request)?;
        energies.push(out.energy.value());
        gradients_of_core.push(out.gradient_bohr);
    }
    Ok(Outcome {
        name: setting.to_string(),
        seconds: start.elapsed().as_secs_f64(),
        gradients: registry.counts().gradients,
        energies,
        gradients_of_core,
        counts: relaxed.counts(),
    })
}

/// `models STRUCTURE.xyz FIXED`: the time of the model Hessians a preconditioner of the
/// environment is made from, on the atoms that `FIXED` (atom numbers from 1, `1-88,144-769`)
/// leaves free: `lindh2007d3` as `-modelhess` builds it, its parts (the Lindh terms and the
/// D3 Hessian), and `lindh2007d3` with the free atoms and their fixed neighbours.
fn models(path: &str, fixed: &str) -> MoptResult<()> {
    use mopt_rs::model_hessian::{resolve, Dispersion, InternalModel, ModelHessian};
    let frames = read_xyz_trajectory_file(path, SpinCharge::default(), None)?;
    let sys = frames
        .frames
        .into_iter()
        .next()
        .ok_or_else(|| MoptError::config("the file has no structure"))?
        .system;
    let mut held: Vec<usize> =
        mopt_rs::core_infra::arg_parser::parse_atom_indices_zero_based(fixed)?;
    held.sort_unstable();
    let free: Vec<mopt_rs::domain::AtomId> = (0..sys.n_atoms())
        .filter(|a| held.binary_search(a).is_err())
        .map(mopt_rs::domain::AtomId::new)
        .collect();
    let sub = sys.subsystem(&free, sys.spin_charge)?;
    let zero = faer::Col::zeros(sub.dof());
    let time = |what: &str, f: &dyn Fn() -> MoptResult<()>| -> MoptResult<()> {
        let start = Instant::now();
        f()?;
        println!(
            "{what:<60} {:>9.2} s ({} atoms)",
            start.elapsed().as_secs_f64(),
            sub.n_atoms()
        );
        Ok(())
    };
    time("lindh2007d3 (-modelhess)", &|| {
        resolve("lindh2007d3")?.hessian(&sub, &zero).map(|_| ())
    })?;
    time("lindh2007 terms without dispersion", &|| {
        InternalModel::lindh_2007(Dispersion::None)
            .hessian(&sub, &zero)
            .map(|_| ())
    })?;
    time("D3(BJ) Hessian", &|| {
        mopt_rs::pes::dispersion::D3Bj::PBE
            .evaluate(&sub, true)
            .map(|_| ())
    })?;
    Ok(())
}

fn main() -> MoptResult<()> {
    let usage = "usage: bench_microiterations TRAJECTORY.xyz \"CALCULATOR OPTIONS\" \
                 [hessian_every=N] SETTING ...\n       bench_microiterations models STRUCTURE.xyz \
                 FIXED";
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    if arguments.first().map(String::as_str) == Some("models") {
        return match (arguments.get(1), arguments.get(2)) {
            (Some(path), Some(fixed)) => models(path, fixed),
            _ => Err(MoptError::config(usage)),
        };
    }
    let (trajectory, calculator) = match (arguments.first(), arguments.get(1)) {
        (Some(t), Some(c)) => (
            t.clone(),
            c.split_whitespace().map(String::from).collect::<Vec<_>>(),
        ),
        _ => return Err(MoptError::config(usage)),
    };
    let mut hessian_every = 0;
    let mut settings = Vec::new();
    for a in &arguments[2..] {
        match a.strip_prefix("hessian_every=") {
            Some(n) => {
                hessian_every = n
                    .parse()
                    .map_err(|_| MoptError::config(format!("hessian_every={n}: not a number")))?;
            }
            None => settings.push(a.clone()),
        }
    }
    if settings.is_empty() {
        return Err(MoptError::config(usage));
    }
    let frames: Vec<_> = read_xyz_trajectory_file(&trajectory, SpinCharge::default(), None)?
        .frames
        .into_iter()
        .map(|f| f.system)
        .collect();
    println!(
        "{} frames of {} atoms, Hessian every {} frame(s)",
        frames.len(),
        frames.first().map_or(0, |f| f.n_atoms()),
        hessian_every
    );
    let mut outcomes: Vec<Outcome> = Vec::new();
    for setting in &settings {
        let o = run(&trajectory, &frames, &calculator, setting, hessian_every)?;
        let (de, dg) = match outcomes.first() {
            Some(reference) => (
                o.energies
                    .iter()
                    .zip(&reference.energies)
                    .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs())),
                o.gradients_of_core
                    .iter()
                    .zip(&reference.gradients_of_core)
                    .fold(0.0_f64, |m, (a, b)| m.max(col_max_abs(&(a - b)))),
            ),
            None => (0.0, 0.0),
        };
        let c = o.counts;
        println!(
            "{:<48} {:>8.1} s {:>9} gradients {:>8} inner iterations {:>4} predicted {:>4} loose \
             {:>3} killed {:>3} unconverged  |dE*| {:.1e}  |dg| {:.1e}",
            o.name,
            o.seconds,
            o.gradients,
            c.inner_iterations,
            c.predicted,
            c.loose,
            c.killed_modes,
            c.unconverged,
            de,
            dg
        );
        // Where the gradients of the relaxations went.
        let checks = 2 * c.curvature_products;
        let searches = c
            .environment_gradients
            .saturating_sub(checks + c.inner_iterations + c.relaxations);
        println!(
            "{:<48} environment {} = iterations {} + first points {} + curvature {} ({} checks, \
             {} at the limit) + rest {} (trial points of line searches, predictions); {} \
             fallback(s) to FIRE with {} iteration(s); {} preconditioner build(s)",
            "",
            c.environment_gradients,
            c.inner_iterations,
            c.relaxations,
            checks,
            c.curvature_checks,
            c.unconverged_checks,
            searches,
            c.fallbacks,
            c.fallback_iterations,
            c.preconditioner_builds
        );
        outcomes.push(o);
    }
    Ok(())
}
