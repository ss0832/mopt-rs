//! `mdmain` on an objective of plain variables (`-os mb2d`, `-os expr:...` with a
//! variables file; unit masses), and the lines that start and end every run.
//!
//! MultiOptPy: no counterpart (new in mopt-rs; MultiOptPy's dynamics acts on atoms).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::path::Path;

use super::mdmain::{open_run, settings_of, stem_of, MdmainConfig, MdmainResult};
use super::variables::read_variables_file;
use crate::core_infra::log::{python_float, Level, Logger};
use crate::domain::Units;
use crate::error::{MoptError, MoptResult};
use crate::md::barostat::gigapascal_per_atomic_unit;
use crate::md::{MdMachine, MdSpace, MolecularDynamics, RigidMotion};
use crate::pes::objective::{check_dimension, ObjectiveRegistry};
use crate::workflow::md::{run_md, MdObserver, ObjectiveMdSurface};
use crate::workflow::md_files::{MdFiles, MdLayout};
use crate::workflow::md_log::MdLog;
use crate::workflow::observers::StopFile;

/// The lines that start every run: the parts, the settings, the references.
pub(super) fn announce(md: &MolecularDynamics, log: &Logger) {
    let s = md.settings();
    let (integrator, thermostat, barostat) = md.part_names();
    log.info(&format!(
        "Molecular dynamics: {} steps of {} a.u. ({} fs), integrator {integrator}, thermostat \
         {thermostat}{}, temperature {} K, seed {}",
        s.steps,
        python_float(s.time_step),
        python_float(s.time_step * Units::AU_TIME_TO_FS),
        if thermostat == "none" {
            String::new()
        } else {
            format!(
                " (coupling time {} fs)",
                python_float(s.thermostat_settings.tau * Units::AU_TIME_TO_FS)
            )
        },
        python_float(s.temperature),
        s.seed
    ));
    // Measured: a Nose-Hoover thermostat with a coupling time of two time steps broke down
    // (docs/source/molecular_dynamics.md).
    if thermostat.starts_with("nose_hoover") && s.thermostat_settings.tau < 10.0 * s.time_step {
        log.warn(&format!(
            "The coupling time of the thermostat ({} fs) is shorter than ten time steps; \
             Nose-Hoover thermostats can become unstable (see the documentation of mdmain).",
            python_float(s.thermostat_settings.tau * Units::AU_TIME_TO_FS)
        ));
    }
    if let Some(b) = barostat {
        log.info(&format!(
            "Constant pressure: barostat {b}, pressure {} GPa ({} kPa), coupling time {} fs",
            python_float(s.pressure * gigapascal_per_atomic_unit()),
            python_float(s.pressure * gigapascal_per_atomic_unit() * 1e6),
            python_float(s.barostat_settings.tau * Units::AU_TIME_TO_FS)
        ));
    }
    if !s.schedule.is_empty() {
        let changes: Vec<String> = s
            .schedule
            .iter()
            .map(|(step, t)| format!("{} K from step {step}", python_float(*t)))
            .collect();
        log.info(&format!(
            "Temperature changes (-ct): {}",
            changes.join(", ")
        ));
    }
    log.info(&format!(
        "Conserved motions of the whole system: {}; constraints held by SHAKE/RATTLE: {}",
        md.rigid().name(),
        md.constraint_count()
    ));
    for paper in md.references() {
        log.info(&format!("Reference: {paper}"));
    }
}

pub(super) fn finish(
    config: &MdmainConfig,
    md: &dyn MdMachine,
    interrupted: bool,
    dir: &Path,
    files: &MdFiles,
    log: &Logger,
) -> MoptResult<()> {
    log.info(&format!(
        "Degrees of freedom of the temperature: {}",
        md.dof()
    ));
    if md.frozen_modes() > 0 {
        log.info(&format!("Frozen modes: {}", md.frozen_modes()));
    }
    if interrupted {
        log.info(&format!(
            "Continue with: mdmain --restart {}",
            dir.display()
        ));
    }
    log.info(&format!(
        "Trajectory file path: {}",
        files.trajectory_path().display()
    ));
    log.info(&format!(
        "Last structure file path: {}",
        files.final_path().display()
    ));
    if config.inputs.len() > 1 || config.trajectories > 1 {
        log.info("Complete...");
    }
    Ok(())
}

pub(super) fn run_objective(
    config: &MdmainConfig,
    input: &Path,
    k: usize,
    base: &Path,
    console: Option<Level>,
    objectives: &ObjectiveRegistry,
    extra: &mut [&mut dyn MdObserver],
) -> MoptResult<MdmainResult> {
    let atoms_only = !config.biases.is_empty()
        || config.excited_state > 0
        || !config.model_function.is_empty()
        || config.cell.is_some()
        || !config.md.constraints.is_empty()
        || config.md.barostat.is_some();
    if atoms_only {
        return Err(MoptError::parse(
            "bias potentials, -es, -mf, -pbc, constraints, and barostats need atoms; an \
             objective of plain variables takes thermostats and frozen modes",
        ));
    }
    let evaluations_before = objectives.counts();
    let objective = objectives.create(&config.calculator, &config.calculator_options)?;
    let vars = read_variables_file(input)?;
    let n = vars.values.len();
    check_dimension(objective.as_ref(), n)?;
    let names = vars
        .names
        .clone()
        .unwrap_or_else(|| objective.variable_names(n));
    let (dir, log) = open_run(config, input, k, base, console)?;
    log.info(&format!(
        "Mode: molecular dynamics of {n} plain variables ({}), unit masses",
        names.join(", ")
    ));
    log.info(&format!("Objective: {}", objective.description()));
    if config.restart_dir.is_some() {
        log.info("Restart: the run is repeated from the start (objectives are cheap).");
    }
    let mut settings = settings_of(config, k);
    settings.rigid = RigidMotion::None;
    let x0 = faer::Col::from_fn(n, |i| vars.values[i]);
    let mut md = MolecularDynamics::new(settings, MdSpace::variables(n)?, x0, None)?;
    announce(&md, &log);
    let stem = stem_of(input);
    let mut files = MdFiles::new(
        &dir,
        &stem,
        MdLayout::Variables(names.clone()),
        config.run.write_every,
    )?;
    let mut printed = MdLog::new(log.clone(), config.run.print_every, md.settings().steps)
        .with_state(config.run.structure_every, MdLayout::Variables(names));
    let mut stop = StopFile::new(&dir, log.clone());
    let end = {
        let mut observers: Vec<&mut dyn MdObserver> = vec![&mut printed, &mut files, &mut stop];
        for o in extra.iter_mut() {
            observers.push(&mut **o);
        }
        run_md(
            &mut md,
            &ObjectiveMdSurface(objective.as_ref()),
            &mut observers,
        )?
    };
    finish(config, &md, end.is_none(), &dir, &files, &log)?;
    super::mdmain_analysis::analyze_variables(config, &dir, &log)?;
    log.info(
        &objectives
            .counts()
            .since(&evaluations_before)
            .line("this input"),
    );
    log.flush();
    Ok(MdmainResult {
        directory: dir,
        trajectory: files.trajectory_path(),
        last: files.final_path(),
        interrupted: end.is_none(),
        steps: md.evaluations().saturating_sub(1),
    })
}
