//! The optimizations, dynamics, Hessians, and IRCs of `confsearch`, on the surface of the
//! command (the calculator of the registry, `-mf`, `-es`, the bias potentials, the
//! constraints, `-coord`, and the other settings of `optmain`).
//!
//! Every job uses the one calculator of the run (wrapped by the evaluation journal), so that
//! a restart replays the evaluations. A `STOP` file (or MultiOptPy's `end.txt`) in the run
//! directory stops every job that runs and every later one.
//!
//! MultiOptPy: the calls of `optimization.Optimize(args).run()` in
//!  `Entrypoints/conformation_search.py:run_confsearch`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

use faer::Mat;

use super::confsearch_settings::ConfsearchConfig;
use super::irc::{irc_states, run_structure, IrcSurfaceParts};
use super::optmain::state_surface;
use crate::core_infra::log::{Level, Logger};
use crate::domain::{AtomId, MolecularSystem, SpinCharge};
use crate::error::{MoptError, MoptResult};
use crate::irc::IrcSettings;
use crate::md::{MdSettings, MdSpace, MolecularDynamics, RigidMotion};
use crate::md::{MdStepReport, ThermostatSettings};
use crate::opt::convergence::ConvergenceCriteria;
use crate::opt::state_machine::{GeometryOptimizer, IterationReport, Termination};
use crate::opt::step::ccqn::CcqnAxis;
use crate::pes::calculator::Calculator;
use crate::pes::combine::PesExpr;
use crate::pes::constraint::{Constraint, ConstraintKind, ConstraintSet};
use crate::pes::term::{fork_term, History, PesTerm};
use crate::workflow::confsearch::{
    DynamicsJob, Level as JobLevel, RelaxJob, Relaxed, Sampled, SearchJobs, OFF_CONSTRAINTS,
};
use crate::workflow::driver::CalculatorSet;
use crate::workflow::md::{run_md, MdObserver, MoleculeMdSurface};
use crate::workflow::observers::{ShapeGuard, STOP_FILE};
use crate::workflow::{run_optimization, Observer};

/// MultiOptPy's stop file of `confsearch`.
pub const MULTIOPTPY_STOP_FILE: &str = "end.txt";

/// Standard deviation (Bohr) of the seeded displacement of every coordinate of a converged
/// conformer before it is optimized once more (`-cs_opt nudge=on`).
pub const NUDGE_BOHR: f64 = 0.02;
/// Energy (Hartree) by which the displaced optimization must end lower to replace the first.
pub const NUDGE_GAIN: f64 = 1e-6;

/// A stop signal shared by every job of the run.
#[derive(Debug, Clone)]
pub struct SharedStop {
    dir: PathBuf,
    flag: Arc<AtomicBool>,
    log: Logger,
    /// The stop of the run that the search is a part of ([`Self::following`]).
    outer: Option<Outer>,
}

/// The stop signal of a run around a search.
#[derive(Clone)]
struct Outer(Arc<dyn Fn() -> bool + Send + Sync>);

impl std::fmt::Debug for Outer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Outer")
    }
}

impl SharedStop {
    /// Watch `dir` for `STOP` and `end.txt`.
    pub fn new(dir: &Path, log: Logger) -> Self {
        Self {
            dir: dir.to_path_buf(),
            flag: Arc::new(AtomicBool::new(false)),
            log,
            outer: None,
        }
    }

    /// As [`Self::new`], and stopped as well as soon as `outer` says so: the stop signal of a
    /// search that is one job of a larger run (the conformer search of a transition state in
    /// the mapper, which stops with the run and at the time limit of its job).
    pub fn following(
        dir: &Path,
        log: Logger,
        outer: impl Fn() -> bool + Send + Sync + 'static,
    ) -> Self {
        Self {
            outer: Some(Outer(Arc::new(outer))),
            ..Self::new(dir, log)
        }
    }

    /// Whether the run is to stop (looks for the files; `STOP` is removed, so that a restart
    /// goes on, and `end.txt` is left as MultiOptPy leaves it).
    pub fn check(&self) -> bool {
        if self.flag.load(Ordering::SeqCst) {
            return true;
        }
        if self.outer.as_ref().is_some_and(|o| (o.0)()) {
            self.flag.store(true, Ordering::SeqCst);
            return true;
        }
        let stop = self.dir.join(STOP_FILE);
        if stop.exists() {
            let _ = std::fs::remove_file(&stop);
            self.log.info(&format!(
                "{} found: the run is interrupted.",
                stop.display()
            ));
            self.flag.store(true, Ordering::SeqCst);
        }
        if self.dir.join(MULTIOPTPY_STOP_FILE).exists() {
            self.flag.store(true, Ordering::SeqCst);
        }
        self.flag.load(Ordering::SeqCst)
    }

    /// Whether a stop was seen.
    pub fn stopped(&self) -> bool {
        self.flag.load(Ordering::SeqCst)
    }
}

impl Observer for SharedStop {
    fn on_iteration(&mut self, _report: &IterationReport) -> MoptResult<()> {
        Ok(())
    }

    fn should_stop(&mut self, _current: &MolecularSystem) -> bool {
        self.check()
    }
}

/// Keeps a frame every `every` steps of a dynamics, and stops it with the run.
struct Frames {
    template: MolecularSystem,
    every: usize,
    frames: Vec<MolecularSystem>,
    stop: SharedStop,
}

impl MdObserver for Frames {
    fn on_step(&mut self, report: &MdStepReport) -> MoptResult<()> {
        if report.step.is_multiple_of(self.every.max(1)) {
            let mut sys = self.template.with_flat_coords_bohr(&report.x)?;
            if report.cell.is_some() {
                sys.cell.clone_from(&report.cell);
            }
            self.frames.push(sys);
        }
        Ok(())
    }

    fn should_stop(&mut self) -> bool {
        self.stop.check()
    }
}

/// The settings of the command that name atoms, in the numbering of the structures the jobs
/// move (the core with ONIOM microiterations, where the bias potentials are evaluated at the
/// core in the whole system; the whole system otherwise).
pub struct JobAtoms {
    /// The constraints of the command.
    pub constraints: Vec<Constraint>,
    /// Its bias potentials, with their labels.
    pub terms: Vec<(String, Arc<dyn PesTerm>)>,
    /// The fragments of `-gfix`.
    pub fragments: Vec<Vec<AtomId>>,
    /// The reactive bonds of CCQN.
    pub ccqn_bonds: Vec<(AtomId, AtomId)>,
    /// The product of CCQN.
    pub ccqn_product: Option<MolecularSystem>,
}

/// The jobs of a run.
pub struct CommandJobs<'a> {
    config: &'a ConfsearchConfig,
    atoms: JobAtoms,
    calc: &'a dyn Calculator,
    dir: PathBuf,
    log: Logger,
    stop: SharedStop,
    irc: IrcSettings,
    ircs: AtomicUsize,
    save_irc_files: bool,
}

impl<'a> CommandJobs<'a> {
    /// Jobs of `config` with the calculator `calc` (journaled), writing IRCs under `dir`.
    pub fn new(
        config: &'a ConfsearchConfig,
        atoms: JobAtoms,
        calc: &'a dyn Calculator,
        dir: &Path,
        log: Logger,
        stop: SharedStop,
    ) -> MoptResult<Self> {
        let irc = match &config.optmain.irc {
            Some(s) => s.clone(),
            None => IrcSettings::from_multioptpy("0.1", "300", "lqa")?,
        };
        Ok(Self {
            config,
            atoms,
            calc,
            dir: dir.to_path_buf(),
            log,
            stop,
            irc,
            ircs: AtomicUsize::new(0),
            save_irc_files: true,
        })
    }

    /// Keep only the endpoints in memory for a mapper run with stage output disabled.
    pub(super) fn with_irc_files(mut self, enabled: bool) -> Self {
        self.save_irc_files = enabled;
        self
    }

    /// The bias potentials of the command, with their labels: a copy of each for every job
    /// ([`fork_term`]), so that a term with state (metadynamics deposits, the warm start of an
    /// inner problem) starts every job from the same state and the energies of the jobs do
    /// not depend on the order or the number of threads they run in.
    fn biases(&self) -> MoptResult<Vec<(String, Arc<dyn PesTerm>)>> {
        self.atoms
            .terms
            .iter()
            .map(|(label, t)| Ok((label.clone(), fork_term(t)?)))
            .collect()
    }

    /// `start` restored onto the constraints of the command and of `job` (the constraints of
    /// the command were resolved at its input: frozen atoms go back to their places and
    /// every value to that of the input, whatever the move made of the candidate); an error
    /// starting with [`OFF_CONSTRAINTS`] when that fails.
    fn on_constraints(
        &self,
        start: &MolecularSystem,
        job: &RelaxJob,
    ) -> MoptResult<MolecularSystem> {
        let mut all = self.atoms.constraints.clone();
        all.extend(job.constraints.iter().cloned());
        if all.is_empty() || all.iter().any(Constraint::is_pending) {
            return Ok(start.clone());
        }
        ConstraintSet::new(&all, start, None)
            .and_then(|set| set.restore(start))
            .map_err(|e| MoptError::conv(format!("{OFF_CONSTRAINTS}: {e}")))
    }

    /// An optimizer of the command for `start` with the extras of `job`.
    fn optimizer(&self, start: &MolecularSystem, job: &RelaxJob) -> MoptResult<GeometryOptimizer> {
        let o = &self.config.optmain;
        let mut settings = o.settings.clone();
        settings.rigid_body_invariant = crate::pes::term::frame_invariant(
            self.calc.is_rigid_body_invariant(),
            self.atoms
                .terms
                .iter()
                .chain(&job.terms)
                .map(|(_, t)| t.as_ref()),
        );
        if job.level == JobLevel::Loose {
            settings.criteria = ConvergenceCriteria::LOOSE;
        }
        if job.saddle_order != settings.saddle_order {
            settings.saddle_order = job.saddle_order;
            if !self.config.methods_given {
                settings.methods = crate::opt::registry::default_methods(job.saddle_order);
            }
        }
        settings.constraints.clone_from(&self.atoms.constraints);
        settings.constraints.extend(job.constraints.iter().cloned());
        settings.fragments.clone_from(&self.atoms.fragments);
        if let Some(c) = settings.ccqn.as_mut() {
            match &self.atoms.ccqn_product {
                Some(product) => c.axis = CcqnAxis::linear_midpoint(start, product)?,
                None => c.axis = CcqnAxis::ReactiveBonds(self.atoms.ccqn_bonds.clone()),
            }
        }
        let mut opt = GeometryOptimizer::new(settings, start.clone())?;
        if let Some(surface) = state_surface(&o.model_function, o.excited_state)? {
            opt.set_surface(surface)?;
        }
        for (label, term) in self.biases()? {
            opt.add_term(&label, term)?;
        }
        for (label, term) in &job.terms {
            opt.add_term(label, Arc::clone(term))?;
        }
        Ok(opt)
    }

    /// The Hessian of the surface of the command at `sys`.
    /// The vibrational problem of `sys` as `optmain -freq` takes it
    /// (`optmain::frequency_problem`), with whether the rigid-body motions are projected (the
    /// optimizer removes them only where the frame of the surface has them: no frozen atoms,
    /// no bias fixed in space).
    pub(super) fn frequency_problem(
        &self,
        sys: &MolecularSystem,
    ) -> MoptResult<(super::optmain::FrequencyProblem, bool)> {
        let mut opt = self.optimizer(sys, &RelaxJob::minimize("Hessian", JobLevel::Final))?;
        let tasks = opt.hessian_tasks(sys)?;
        let results = CalculatorSet::single(self.calc).evaluate(&tasks)?;
        let project = opt.removes_rigid_body_motion();
        let problem = super::optmain::frequency_problem(&mut opt, self.calc, sys, results)?;
        Ok((problem, project))
    }

    fn surface_hessian(&self, sys: &MolecularSystem) -> MoptResult<Mat<f64>> {
        let mut opt = self.optimizer(sys, &RelaxJob::minimize("Hessian", JobLevel::Final))?;
        let tasks = opt.hessian_tasks(sys)?;
        let results = CalculatorSet::single(self.calc).evaluate(&tasks)?;
        Ok(opt.surface_hessian(sys, results)?.2)
    }

    /// One optimization from `start`.
    fn optimize(&self, start: &MolecularSystem, job: &RelaxJob) -> MoptResult<Relaxed> {
        let start = &self.on_constraints(start, job)?;
        let mut opt = self.optimizer(start, job)?;
        let mut stop = self.stop.clone();
        let mut shape = ShapeGuard::new(
            self.config.optmain.shape_conditions.clone(),
            self.log.clone(),
        );
        let outcome = {
            let mut observers: Vec<&mut dyn Observer> = vec![&mut stop, &mut shape];
            run_optimization(&mut opt, self.calc, &mut observers)?
        };
        let termination = outcome.termination;
        self.log.debug(&format!(
            "{}: {} iterations, E = {:.10}, {}",
            job.label,
            outcome.iterations,
            outcome.biased_energy.value(),
            match termination {
                Some(Termination::Converged) => "converged",
                Some(Termination::MaxIterations | Termination::SeamCheckFailed) => "not converged",
                Some(Termination::Dissociated) => "dissociated",
                None => "stopped",
            }
        ));
        Ok(Relaxed {
            energy: if job.terms.is_empty() {
                outcome.biased_energy.value()
            } else {
                outcome.energy.value()
            },
            bare_energy: outcome.energy.value(),
            system: outcome.system,
            converged: termination == Some(Termination::Converged),
            dissociated: termination == Some(Termination::Dissociated),
            iterations: outcome.iterations,
            interrupted: termination.is_none() && self.stop.stopped(),
        })
    }

    /// Whether the dynamics may take the rigid motions out (as `mdmain` decides).
    fn rigid(&self, sys: &MolecularSystem) -> RigidMotion {
        let invariant = self.calc.is_rigid_body_invariant()
            && self.atoms.terms.is_empty()
            && !crate::pes::constraint::breaks_rigid_body(&self.atoms.constraints);
        match (invariant, sys.has_pbc()) {
            (false, _) => RigidMotion::None,
            (true, true) => RigidMotion::Translation,
            (true, false) => RigidMotion::TranslationRotation,
        }
    }
}

impl SearchJobs for CommandJobs<'_> {
    fn relax(&self, start: &MolecularSystem, job: &RelaxJob) -> MoptResult<Relaxed> {
        if self.stop.check() {
            return Ok(Relaxed {
                system: start.clone(),
                energy: f64::NAN,
                bare_energy: f64::NAN,
                converged: false,
                dissociated: false,
                iterations: 0,
                interrupted: true,
            });
        }
        let first = self.optimize(start, job)?;
        let nudge = self.config.nudge
            && first.converged
            && job.level == JobLevel::Final
            && job.saddle_order == 0
            && job.terms.is_empty()
            && job.constraints.is_empty()
            && self.atoms.constraints.is_empty();
        if !nudge {
            return Ok(first);
        }
        // A minimum by symmetry only (an eclipsed methyl group: a saddle point whose
        // gradient vanishes) does not survive a small displacement; a minimum does.
        let mut x = first.system.coords_flat_bohr();
        let mut seed = 0x9e37_79b9_7f4a_7c15_u64;
        for v in x.iter() {
            seed = (seed ^ v.to_bits()).wrapping_mul(0x0100_0000_01b3);
        }
        let mut rng = crate::core_infra::rng::Rng::from_seed(seed);
        for v in x.iter_mut() {
            *v += NUDGE_BOHR * rng.standard_normal();
        }
        let nudged = first.system.with_flat_coords_bohr(&x)?;
        let label = format!("{} (displaced)", job.label);
        let second = self.optimize(
            &nudged,
            &RelaxJob {
                label,
                ..job.clone()
            },
        )?;
        if second.interrupted {
            return Ok(second);
        }
        if second.converged && second.energy < first.energy - NUDGE_GAIN {
            self.log.debug(&format!(
                "{}: the displaced optimization went {:.4} kcal/mol lower (a saddle point by \
                 symmetry was left)",
                job.label,
                (first.energy - second.energy) * crate::domain::Units::HARTREE_TO_KCAL_MOL
            ));
            return Ok(Relaxed {
                iterations: first.iterations + second.iterations,
                ..second
            });
        }
        Ok(first)
    }

    fn dynamics(&self, start: &MolecularSystem, job: &DynamicsJob) -> MoptResult<Sampled> {
        if self.stop.check() {
            return Ok(Sampled {
                frames: vec![start.clone()],
                interrupted: true,
            });
        }
        let o = &self.config.optmain;
        let mut masses = start.masses_amu();
        if let Some(h) = job.hydrogen_mass {
            for (m, e) in masses.iter_mut().zip(&start.elements) {
                if e.atomic_number() == 1 {
                    *m = h;
                }
            }
        }
        let mut constraints = self.atoms.constraints.clone();
        constraints.extend(job.constraints.iter().cloned());
        let settings = MdSettings {
            time_step: job.time_step,
            steps: job.steps,
            temperature: job.temperature,
            thermostat: "bussi".to_string(),
            thermostat_settings: ThermostatSettings::default(),
            seed: job.seed,
            rigid: self.rigid(start),
            constraints,
            freeze: job.freeze.clone(),
            ..MdSettings::default()
        };
        let mut md = MolecularDynamics::new(
            settings,
            MdSpace::atoms(&masses)?,
            start.coords_flat_bohr(),
            Some(start.clone()),
        )?;
        let mut terms = self.biases()?;
        terms.extend(job.terms.iter().cloned());
        let mut expr =
            state_surface(&o.model_function, o.excited_state)?.unwrap_or_else(PesExpr::single);
        for (label, term) in &terms {
            expr = expr.plus_term(label, Arc::clone(term));
        }
        let surface =
            MoleculeMdSurface::new(CalculatorSet::single(self.calc), expr, start.clone(), terms)?;
        let mut frames = Frames {
            template: start.clone(),
            every: job.frame_every,
            frames: Vec::new(),
            stop: self.stop.clone(),
        };
        let end = run_md(&mut md, &surface, &mut [&mut frames])?;
        self.log.debug(&format!(
            "{}: {} steps, {} frames",
            job.label,
            md.evaluations(),
            frames.frames.len()
        ));
        Ok(Sampled {
            frames: frames.frames,
            interrupted: end.is_none(),
        })
    }

    fn hessian(&self, sys: &MolecularSystem) -> MoptResult<Mat<f64>> {
        self.surface_hessian(sys)
    }

    /// As `optmain -freq` (`optmain::frequency_problem`): the rigid-body motions projected
    /// where the optimizer removes them, and with constraints the modes on the constraint
    /// surface (the Hessian of the Lagrangian, the constraint normals and the rigid motions
    /// that keep the constraints projected), so that zero modes are never counted as
    /// imaginary.
    fn modes(&self, sys: &MolecularSystem) -> MoptResult<crate::vib::NormalModes> {
        let (problem, project) = self.frequency_problem(sys)?;
        crate::vib::normal_modes_without(sys, &problem.hessian, project, &problem.normals)
    }

    fn reaction_ends(
        &self,
        ts: &MolecularSystem,
        label: &str,
    ) -> MoptResult<Option<[(MolecularSystem, f64); 2]>> {
        let o = &self.config.optmain;
        let k = self.ircs.fetch_add(1, Ordering::SeqCst);
        let dir = self.dir.join("ts_irc").join(format!("irc_{k}"));
        let log = Logger::silent();
        if self.save_irc_files {
            std::fs::create_dir_all(&dir)?;
            log.add_file(dir.join("mopt.log"), Level::Debug)?;
        }
        log.info(&format!("IRC of {label}"));
        let constraints = if self.atoms.constraints.is_empty() {
            None
        } else {
            Some(ConstraintSet::new(&self.atoms.constraints, ts, None)?)
        };
        let parts = IrcSurfaceParts {
            terms: self.biases()?,
            excited_state: o.excited_state,
            states: irc_states(&o.model_function, o.excited_state)?,
            constraints,
            iteration: 0,
        };
        let settings = self.irc.clone().accepting_converged(
            o.settings.criteria.max_force.value(),
            o.settings.criteria.rms_force.value(),
        );
        let points: Vec<MolecularSystem> = if self.save_irc_files {
            let run = run_structure(&settings, self.calc, parts, ts, "ts", &dir, &log, &mut [])?;
            log.flush();
            if run.interrupted {
                return Ok(None);
            }
            run.endpoints
                .iter()
                .map(|path| crate::io::xyz::read_xyz_file(path, ts.spin_charge, ts.cell.as_ref()))
                .collect::<MoptResult<_>>()?
        } else {
            let mut stop = crate::workflow::irc::FnObserver::new(|_| Ok(!self.stop.check()));
            let Some(run) = super::irc::run_structure_observed(
                &settings,
                self.calc,
                parts,
                ts,
                &log,
                &mut [&mut stop],
            )?
            else {
                return Ok(None);
            };
            run.branches
                .iter()
                .map(|side| ts.with_flat_coords_bohr(&side.x))
                .collect::<MoptResult<_>>()?
        };
        if points.len() != 2 {
            return Ok(None);
        }
        let mut ends = Vec::new();
        for end in points {
            let relaxed = self.relax(
                &end,
                &RelaxJob::minimize(format!("{label}: IRC end"), JobLevel::Final),
            )?;
            if !relaxed.converged {
                return Ok(None);
            }
            ends.push((relaxed.system, relaxed.energy));
        }
        let second = ends.pop();
        let first = ends.pop();
        match (first, second) {
            (Some(a), Some(b)) => Ok(Some([a, b])),
            _ => Err(MoptError::calc("the IRC gave fewer than two ends")),
        }
    }

    fn stop_requested(&self) -> bool {
        self.stop.check()
    }

    fn frozen_atoms(&self) -> Vec<usize> {
        let mut atoms: Vec<usize> = self
            .atoms
            .constraints
            .iter()
            .filter_map(|c| match &c.kind {
                ConstraintKind::Frozen(a) => Some(a.iter().map(|x| x.index()).collect::<Vec<_>>()),
                _ => None,
            })
            .flatten()
            .collect();
        atoms.sort_unstable();
        atoms.dedup();
        atoms
    }
}

/// Refuses bias potentials that make no sense for a ranking of minima: a term that changes
/// with the simulated time or with the iteration of an optimization gives every minimum a
/// bias that depends on how long its optimization ran, not on where it is.
///
/// MultiOptPy: no counterpart (new in mopt-rs).
pub fn refuse_unrankable_terms(
    command: &str,
    terms: &[(String, Arc<dyn PesTerm>)],
) -> MoptResult<()> {
    for (label, t) in terms {
        let why = match t.history() {
            History::TimeDependent => "changes with the simulated time",
            History::IterationRamp => "changes with the iteration of an optimization (a ramp)",
            History::Static | History::Stateful => continue,
        };
        return Err(MoptError::unsupported(format!(
            "{command}: the bias potential {label} ('{}') {why}; {command} ranks minima by their \
             energy with the bias potentials, and every minimum would carry a bias that depends \
             on how long its optimization ran (give a constant bias, or use optmain)",
            t.name()
        )));
    }
    Ok(())
}

/// The spin state of the input (`-elec`, `-spin`, `-unpaired`).
pub fn input_spin(config: &ConfsearchConfig) -> MoptResult<SpinCharge> {
    config.optmain.state.of_file(&config.input)
}
