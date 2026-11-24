//! The IRC calculation of the commands: `optmain -irc STEP_SIZE MAX_STEP METHOD` (after the
//! optimization) and `ircmain` (on a given structure) run the same state machine
//! ([`crate::irc::Irc`]) with the same driver and files.
//!
//! MultiOptPy: `optimization.py:Optimize._run_single_job` (the `IRC(...)` block) and
//! `irc.py:IRC.run`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::path::{Path, PathBuf};
use std::sync::Arc;

use faer::Col;

use crate::core_infra::arg_parser::ParsedArgs;
use crate::core_infra::log::{python_float, Logger};
use crate::domain::MolecularSystem;
use crate::error::{MoptError, MoptResult};
use crate::irc::{
    most_abundant_isotope_masses, Irc, IrcEnd, IrcSettings, IrcSpace, Masses, StartKind,
};
use crate::pes::calculator::Calculator;
use crate::pes::combine::PesExpr;
use crate::pes::constraint::ConstraintSet;
use crate::pes::objective::Objective;
use crate::pes::term::{History, PesTerm};
use crate::workflow::irc::{run_irc, IrcObserver, IrcSurface, MoleculeSurface, ObjectiveSurface};
use crate::workflow::irc_cell::{CellStructures, CellSurface};
use crate::workflow::irc_output::{IrcOutput, PathLayout};
use crate::workflow::observers::StopFile;

/// What an IRC calculation of a command left behind.
#[derive(Debug, Clone)]
pub struct IrcRun {
    /// The directory of its files.
    pub directory: PathBuf,
    /// How it ended (`None`: interrupted).
    pub end: Option<IrcEnd>,
    /// What was done from the starting structure (known once it was evaluated).
    pub kind: Option<StartKind>,
    /// Whether a `STOP` file interrupted it.
    pub interrupted: bool,
    /// The files of the end points (MultiOptPy's `terminal_struct_paths`).
    pub endpoints: Vec<PathBuf>,
    /// `irc_combined_path.xyz`, the path as `freqmain -path` reads it (structures from a
    /// transition state).
    pub combined_path: Option<PathBuf>,
    /// Evaluations of the surface.
    pub evaluations: usize,
}

/// The settings of `-irc STEP_SIZE MAX_STEP METHOD`, `-irc_opt`, and `-fc`; `None` without
/// `-irc`.
pub fn settings_from_args(p: &ParsedArgs) -> MoptResult<Option<IrcSettings>> {
    let settings = match p.get_list("intrinsic_reaction_coordinates") {
        [] => return Ok(None),
        [step, max, method] => IrcSettings::from_multioptpy(step, max, method)?,
        other => {
            return Err(MoptError::parse(format!(
                "-irc takes STEP_SIZE MAX_STEP METHOD (e.g. -irc 0.5 300 lqa), got {} value(s)",
                other.len()
            )))
        }
    };
    with_common_options(settings, p).map(Some)
}

/// The surface of `-mf` for an IRC (`None` without `-mf`): a function of the states, not
/// a projection for crossings (`opt_mesx`, `opt_meci`), whose gradient is not that of an
/// energy.
pub fn irc_states(spec: &[String], root: usize) -> MoptResult<Option<PesExpr>> {
    let Some(name) = spec.first() else {
        return Ok(None);
    };
    let lower = name.to_ascii_lowercase();
    if matches!(
        lower.as_str(),
        "opt_mesx" | "opt_mesx_2" | "mesx" | "opt_meci" | "meci"
    ) {
        return Err(MoptError::unsupported(format!(
            "-mf {name} projects gradients to find a crossing; its gradient is not that of an energy and has no path of steepest descent. An IRC follows a function of the states (-mf seam, conical, avoiding, ...) or one state (-es)"
        )));
    }
    super::optmain::state_surface(spec, root)
}

/// `-fc` and `-irc_opt` on top of `settings`.
pub fn with_common_options(mut settings: IrcSettings, p: &ParsedArgs) -> MoptResult<IrcSettings> {
    let fc = p.get_i64("calc_exact_hess", -1)?;
    settings.exact_hessian_interval = usize::try_from(fc).ok().filter(|&n| n > 0);
    settings.with_options(p.get_list("irc_options"))
}

fn references(method: &str) -> &'static [&'static str] {
    match method {
        "lqa" => &[
            "M. Page and J. W. McIver, J. Chem. Phys. 1988, 88, 922",
            "M. Page, C. Doubleday, and J. W. McIver, J. Chem. Phys. 1990, 93, 5634",
        ],
        "hpc" => &[
            "H. P. Hratchian and H. B. Schlegel, J. Chem. Phys. 2004, 120, 9918",
            "H. P. Hratchian and H. B. Schlegel, J. Chem. Theory Comput. 2005, 1, 61",
        ],
        "dvv" => &["H. P. Hratchian and H. B. Schlegel, J. Phys. Chem. A 2002, 106, 165"],
        "gs2" => &[
            "C. Gonzalez and H. B. Schlegel, J. Chem. Phys. 1989, 90, 2154",
            "C. Gonzalez and H. B. Schlegel, J. Phys. Chem. 1990, 94, 5523",
        ],
        "rk4" => &[
            "K. K. Baldridge, M. S. Gordon, R. Steckler, and D. G. Truhlar, J. Phys. \
                    Chem. 1989, 93, 5107",
        ],
        _ => &[],
    }
}

fn announce(irc: &Irc, log: &Logger) {
    let s = irc.settings();
    log.info(&format!(
        "IRC: method {}{}, step size {} (mass-weighted), MAX_STEP {}, exact Hessian {}",
        s.method,
        if s.method_options.is_empty() {
            String::new()
        } else {
            format!(":{}", s.method_options.join(":"))
        },
        python_float(s.step_size),
        s.max_steps,
        match s.exact_hessian_interval {
            Some(n) => format!("every {n} point(s)"),
            None => "at the starting structure only".to_string(),
        }
    ));
    for paper in references(&s.method) {
        log.info(&format!("Reference: {paper}"));
    }
    if s.first_step.check {
        log.info(&format!(
            "IRC: the first point of each side is tested (angle <= {} degrees, {} <= rho <= {}) \
             and corrected on a hypersphere around the transition state",
            python_float(s.first_step.max_angle_deg),
            python_float(s.first_step.rho_min),
            python_float(s.first_step.rho_max)
        ));
    }
}

fn drive(
    mut irc: Irc,
    surface: &dyn IrcSurface,
    mut output: IrcOutput,
    extra: &mut [&mut dyn IrcObserver],
    dir: &Path,
    log: &Logger,
) -> MoptResult<IrcRun> {
    announce(&irc, log);
    let mut stop = StopFile::new(dir, log.clone());
    let end = {
        let mut observers: Vec<&mut dyn IrcObserver> = vec![&mut output, &mut stop];
        for observer in extra.iter_mut() {
            observers.push(&mut **observer);
        }
        run_irc(&mut irc, surface, &mut observers)?
    };
    if end.is_some() {
        output.finish()?;
    }
    log.flush();
    Ok(IrcRun {
        directory: dir.to_path_buf(),
        kind: irc.kind(),
        interrupted: end.is_none(),
        endpoints: output.endpoints().to_vec(),
        combined_path: output.combined_path(),
        evaluations: irc.evaluations(),
        end,
    })
}

/// What the IRC of a structure follows besides the calculator: bias potentials, the
/// electronic state or the states combined (`-mf`), and constraints.
#[derive(Default)]
pub struct IrcSurfaceParts {
    /// Bias potentials with their labels.
    pub terms: Vec<(String, Arc<dyn PesTerm>)>,
    /// Electronic state (0: the ground state).
    pub excited_state: usize,
    /// The states combined into the surface (`-mf`; [`irc_states`]).
    pub states: Option<PesExpr>,
    /// Constraints the path keeps (`-fix`, `-pc`, `-gfix`, `-uc`).
    pub constraints: Option<ConstraintSet>,
    /// The iteration the terms see along the path: that of the last evaluation of the
    /// optimization the path starts from (`GeometryOptimizer::last_surface_iteration`),
    /// 0 for a given structure.
    pub iteration: usize,
}

/// Refuse the terms whose value at a structure depends on more than the structure and the
/// iteration ([`History::TimeDependent`], [`History::Stateful`]): a path of steepest descent
/// follows one fixed surface.
///
/// MultiOptPy: no counterpart (new in mopt-rs).
pub fn check_irc_terms(
    terms: &[(String, Arc<dyn PesTerm>)],
    states: Option<&PesExpr>,
) -> MoptResult<()> {
    let mut all: Vec<(String, Arc<dyn PesTerm>)> = terms.to_vec();
    if let Some(states) = states {
        all.extend(
            states
                .terms()?
                .into_iter()
                .map(|t| (t.name().to_string(), t)),
        );
    }
    for (label, term) in &all {
        let history = term.history();
        if matches!(history, History::TimeDependent | History::Stateful) {
            return Err(MoptError::unsupported(format!(
                "IRC: the term '{label}' ({history:?}) changes with the time or with what it \
                 has seen; a path of steepest descent follows one fixed surface (a term with a \
                 ramp is held at the iteration where the optimization ended)"
            )));
        }
    }
    Ok(())
}

/// The machine of the IRC from the structure `start` on the surface of `calc` with `parts`
/// (the space with its masses and the constraints of `parts`), and its space. Shared by
/// `ircmain` ([`run_structure`]) and the commands that run an IRC with observers of their own
/// ([`run_structure_observed`]).
fn structure_machine(
    settings: &IrcSettings,
    calc: &dyn Calculator,
    parts: &mut IrcSurfaceParts,
    start: &MolecularSystem,
    log: &Logger,
) -> MoptResult<(Irc, IrcSpace)> {
    check_irc_terms(&parts.terms, parts.states.as_ref())?;
    // Rigid-body motion is removed only where the calculator and every term do not see it.
    let state_terms = match &parts.states {
        Some(states) => states.terms()?,
        None => Vec::new(),
    };
    let rigid = crate::pes::term::frame_invariant(
        calc.is_rigid_body_invariant(),
        parts
            .terms
            .iter()
            .map(|(_, t)| t.as_ref())
            .chain(state_terms.iter().map(|t| t.as_ref())),
    );
    let space = match settings.masses {
        Masses::Standard => IrcSpace::molecule(start, rigid),
        Masses::MostAbundantIsotope => {
            log.info("IRC: masses of the most abundant isotopes (as MultiOptPy)");
            IrcSpace::molecule_with_masses(start, &most_abundant_isotope_masses(start)?, rigid)?
        }
    };
    let space = match parts.constraints.take() {
        Some(set) if !set.is_empty() => {
            log.info(&format!(
                "IRC: {} constraint(s) kept along the path (the path of steepest descent within them, in mass-weighted coordinates)",
                set.len()
            ));
            space.with_constraints(set)?
        }
        _ => space,
    };
    if parts.states.is_some() {
        log.info("IRC: on the surface that combines the states (-mf)");
    }
    let irc = Irc::new(settings.clone(), space.clone(), start.coords_flat_bohr())?;
    Ok((irc, space))
}

/// The IRC from the structure `start` on the surface of `calc` with `parts`; the files go
/// to `dir`, `stem` names the files of the end points. The observers `extra` see every
/// event after the files and the log were written (further processing after every step)
/// and can end the run.
#[allow(clippy::too_many_arguments)]
pub fn run_structure(
    settings: &IrcSettings,
    calc: &dyn Calculator,
    mut parts: IrcSurfaceParts,
    start: &MolecularSystem,
    stem: &str,
    dir: &Path,
    log: &Logger,
    extra: &mut [&mut dyn IrcObserver],
) -> MoptResult<IrcRun> {
    let (irc, space) = structure_machine(settings, calc, &mut parts, start, log)?;
    let output = IrcOutput::new(
        dir,
        stem,
        PathLayout::Atoms(Box::new(start.clone())),
        space,
        irc.start().clone(),
        &irc.settings().method,
        log.clone(),
    );
    let surface = MoleculeSurface {
        calc,
        terms: &parts.terms,
        start,
        excited_state: parts.excited_state,
        log: Some(log),
        states: parts.states.map(std::sync::Mutex::new),
        iteration: parts.iteration,
    };
    drive(irc, &surface, output, extra, dir, log)
}

/// The IRC of [`run_structure`] without its files: the same machine and surface, seen by
/// `observers` alone (a command that keeps the path itself, such as `run_mapper`). From a
/// structure that is no transition state the machine runs a meta-IRC, as in `ircmain`.
/// Returns how it ended (`None`: an observer stopped it).
pub fn run_structure_observed(
    settings: &IrcSettings,
    calc: &dyn Calculator,
    mut parts: IrcSurfaceParts,
    start: &MolecularSystem,
    log: &Logger,
    observers: &mut [&mut dyn IrcObserver],
) -> MoptResult<Option<IrcEnd>> {
    let (mut irc, _) = structure_machine(settings, calc, &mut parts, start, log)?;
    let surface = MoleculeSurface {
        calc,
        terms: &parts.terms,
        start,
        excited_state: parts.excited_state,
        log: Some(log),
        states: parts.states.map(std::sync::Mutex::new),
        iteration: parts.iteration,
    };
    run_irc(&mut irc, &surface, observers)
}

/// The IRC of the atoms and the cell of the periodic structure `start` together (`-vc`):
/// in the variables of the variable-cell optimization with the weight `weight` of the
/// lattice variables, on the enthalpy at the pressure `pressure` (Hartree/Bohr^3) of `calc`
/// with the bias potentials `terms` (seen at `iteration`). The files of the variables are those of an IRC of
/// plain variables; the structures with their cells are written beside them
/// (`irc_<side>_structures_cell.xyz`, `<stem>_irc_endpoint_<n>.xyz`, the end points of the
/// run). `extra` as in [`run_structure`].
#[allow(clippy::too_many_arguments)]
pub fn run_cell(
    settings: &IrcSettings,
    calc: &dyn Calculator,
    terms: &[(String, Arc<dyn PesTerm>)],
    start: &MolecularSystem,
    excited_state: usize,
    iteration: usize,
    weight: f64,
    pressure: f64,
    stem: &str,
    dir: &Path,
    log: &Logger,
    extra: &mut [&mut dyn IrcObserver],
) -> MoptResult<IrcRun> {
    check_irc_terms(terms, None)?;
    let map = crate::opt::cell::VariableCell::new(start, weight, pressure)?;
    let masses = map.masses()?;
    let names = map.names();
    let n = start.dof();
    log.info(&format!(
        "IRC of the atoms and the cell (-vc): {} variables (quasi-Cartesian coordinates of the \
         atoms and {} scaled lattice variables, weight {}), the enthalpy at {} GPa; masses: \
         those of the atoms, and for the lattice variables {} amu (the kinetic metric of a \
         change of the lattice with the atoms following)",
        map.dimension(),
        map.dimension() - n,
        python_float(weight),
        python_float(pressure * crate::opt::cell::gigapascal_per_atomic_unit()),
        masses[n..]
            .iter()
            .map(|m| format!("{m:.1}"))
            .collect::<Vec<_>>()
            .join(", ")
    ));
    let space = IrcSpace::with_masses(&masses)?;
    let x = map.variables(start)?;
    let irc = Irc::new(settings.clone(), space.clone(), x.clone())?;
    let output = IrcOutput::new(
        dir,
        stem,
        PathLayout::Variables(names),
        space,
        x,
        &irc.settings().method,
        log.clone(),
    );
    let surface = CellSurface {
        calc,
        terms,
        map: &map,
        start,
        excited_state,
        states: None,
        log: Some(log),
        iteration,
    };
    let mut structures = CellStructures::new(&map, dir, stem);
    let mut run = {
        let mut observers: Vec<&mut dyn IrcObserver> = vec![&mut structures];
        for observer in extra.iter_mut() {
            observers.push(&mut **observer);
        }
        drive(irc, &surface, output, &mut observers, dir, log)?
    };
    // The structures, not the variables, are the end points of the run.
    run.endpoints = structures.endpoints().to_vec();
    for path in &run.endpoints {
        log.info(&format!("IRC end point with its cell: {}", path.display()));
    }
    Ok(run)
}

/// The IRC of the atoms and the cell of the periodic structure `start` as [`run_cell`] does
/// it, in the variables `map` (made from `start`), with `observers` and no files of its own:
/// what the commands that keep the path themselves run ([`run_structure_observed`] of a
/// variable cell; the mapper with `-vc`). With `states`, on the two states combined by a
/// model function ([`crate::workflow::irc_cell::CellStates`]) in place of `excited_state`.
#[allow(clippy::too_many_arguments)]
pub fn run_cell_observed(
    settings: &IrcSettings,
    map: &crate::opt::cell::VariableCell,
    calc: &dyn Calculator,
    terms: &[(String, Arc<dyn PesTerm>)],
    start: &MolecularSystem,
    excited_state: usize,
    states: Option<&crate::workflow::irc_cell::CellStates>,
    log: &Logger,
    observers: &mut [&mut dyn IrcObserver],
) -> MoptResult<Option<IrcEnd>> {
    let space = IrcSpace::with_masses(&map.masses()?)?;
    let mut irc = Irc::new(settings.clone(), space, map.variables(start)?)?;
    let surface = CellSurface {
        calc,
        terms,
        map,
        start,
        excited_state,
        states,
        log: Some(log),
        iteration: 0,
    };
    run_irc(&mut irc, &surface, observers)
}

/// The IRC from the point `values` of the plain variables `names` on `objective` (unit
/// masses); `extra` as in [`run_structure`].
#[allow(clippy::too_many_arguments)]
pub fn run_variables(
    settings: &IrcSettings,
    objective: &dyn Objective,
    names: &[String],
    values: &[f64],
    stem: &str,
    dir: &Path,
    log: &Logger,
    extra: &mut [&mut dyn IrcObserver],
) -> MoptResult<IrcRun> {
    let space = IrcSpace::variables(values.len());
    let x = Col::from_fn(values.len(), |i| values[i]);
    let irc = Irc::new(settings.clone(), space.clone(), x.clone())?;
    let output = IrcOutput::new(
        dir,
        stem,
        PathLayout::Variables(names.to_vec()),
        space,
        x,
        &irc.settings().method,
        log.clone(),
    );
    drive(irc, &ObjectiveSurface(objective), output, extra, dir, log)
}
