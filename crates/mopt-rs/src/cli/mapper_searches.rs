//! The searches and the dynamics of `run_mapper` ([`MapperJobs::pair_search`],
//! [`MapperJobs::dynamics`]) on the one surface of the run, with the calculator of the job:
//! a method of the registry of two-point methods (ADDF from a minimum, DS-AFIR or a chain
//! between two minima, ...), and molecular dynamics with bias potentials (the nanoreactor).
//!
//! A search writes its files and its full log (every point with its structure) to the
//! directory of its job, `jobs/job_000003/` (the log `<label>.log`); the log of the job gets
//! one line of what it found. Every structure it evaluates goes to the trajectory of the job,
//! as do the kept frames of the dynamics. Both stop with the run: a stop during a search
//! ends it at the next evaluation, and the job is run again by a restart, its evaluations
//! read back from the journal.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use faer::Col;

use super::ieipmain_methods::{fragment_test, fragment_test_on, fragment_test_with};
use super::mapper_jobs::{CommandJobs, MapperStop};
use super::optmain::state_surface;
use crate::core_infra::log::{Level, Logger};
use crate::domain::MolecularSystem;
use crate::error::{MoptError, MoptResult, ResultExt};
use crate::md::{
    InitialVelocities, MdSettings, MdSpace, MdStepReport, MolecularDynamics, RigidMotion,
    ThermostatSettings,
};
use crate::path::points::PointRequest;
use crate::path::string_nodes::PointGeometry;
use crate::pes::calculator::Calculator;
use crate::pes::combine::PesExpr;
use crate::pes::objective::Objective;
use crate::workflow::chain::ChainObserver;
use crate::workflow::chain_files::{ChainFiles, ChainPoints};
use crate::workflow::driver::CalculatorSet;
use crate::workflow::mapper::observers::job_files_dir;
use crate::workflow::mapper::{
    Configuration, DynamicsOutcome, DynamicsRequest, FrameSink, MapperJobs, PairSearchOutcome,
    PairSearchRequest, PathPoint, Stationary,
};
use crate::workflow::md::{run_md, MdObserver, MoleculeMdSurface};
use crate::workflow::pair_search::{
    ImageState, PairInput, PairMethodRegistry, PairSettings, SearchRun, SearchSurface,
    StructureSurface, SurfaceValue, VariableSurface,
};

/// The surface of a search, which stops with the run and hands every evaluated structure to
/// the trajectory of the job.
struct Watched<'s, 'f> {
    inner: &'s dyn SearchSurface,
    stop: &'s MapperStop,
    stopped: AtomicBool,
    frames: Mutex<&'f mut dyn FrameSink>,
    label: &'s str,
    count: AtomicUsize,
    /// Current NEB iteration, for the stage trajectory of its evaluations.
    band: Option<AtomicUsize>,
    /// A point of the plain variables of the search (`None`: structures).
    variables: Option<Configuration>,
}

impl SearchSurface for Watched<'_, '_> {
    fn description(&self) -> String {
        self.inner.description()
    }

    fn evaluate(
        &self,
        x: &Col<f64>,
        image: usize,
        request: &PointRequest,
    ) -> MoptResult<SurfaceValue> {
        if self.stop.check() {
            self.stopped.store(true, Ordering::SeqCst);
            return Err(MoptError::calc("the run was stopped"));
        }
        let value = self.inner.evaluate(x, image, request)?;
        let point = match (self.inner.structure(x, image), &self.variables) {
            (Some(sys), _) => Some(Configuration::Structure(sys)),
            (None, Some(template)) => Some(template.with_flat(x)?),
            (None, None) => None,
        };
        if let Some(point) = point {
            let k = self.count.fetch_add(1, Ordering::Relaxed) + 1;
            let mut frames = self
                .frames
                .lock()
                .map_err(|_| MoptError::calc("the trajectory of the job is not available"))?;
            let label = match &self.band {
                Some(iteration) => format!(
                    "{} band {:04} point {k}",
                    self.label,
                    iteration.load(Ordering::Relaxed)
                ),
                None => format!("{} point {k}", self.label),
            };
            frames.push(&point, value.energy, &label)?;
        }
        Ok(value)
    }

    fn structure(&self, x: &Col<f64>, image: usize) -> Option<MolecularSystem> {
        self.inner.structure(x, image)
    }

    fn take_notes(&self) -> Vec<String> {
        self.inner.take_notes()
    }

    fn begin_iteration(&self, iteration: usize) {
        if let Some(band) = &self.band {
            band.store(iteration, Ordering::Relaxed);
        }
        self.inner.begin_iteration(iteration);
    }

    fn two_surfaces(&self) -> bool {
        self.inner.two_surfaces()
    }

    fn in_order(&self) -> bool {
        self.inner.in_order()
    }

    fn model_hessian(
        &self,
        x: &Col<f64>,
        image: usize,
        gradient: &Col<f64>,
        model: &dyn crate::model_hessian::ModelHessian,
    ) -> MoptResult<Option<faer::Mat<f64>>> {
        self.inner.model_hessian(x, image, gradient, model)
    }
}

/// A file name of `label` (letters, digits, `-` and `_`).
fn file_stem(label: &str) -> String {
    let s: String = label
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect();
    if s.is_empty() {
        "search".to_string()
    } else {
        s
    }
}

/// A step of the dynamics that moves an atom farther than this (Bohr) has blown up: more
/// than ten times what a hydrogen atom moves in a step of 0.5 fs at 2000 K.
pub const BLOW_UP_JUMP_BOHR: f64 = 1.0;

/// Atoms farther than this (Bohr) from the centre of a structure mean that it has blown
/// apart: no structure a search starts from has them.
pub const FAR_APART_BOHR: f64 = 200.0;

/// Refuse `structures` with coordinates that are not finite or atoms that have flown apart.
fn check_structures(label: &str, structures: &[MolecularSystem]) -> MoptResult<()> {
    for (k, s) in structures.iter().enumerate() {
        let x = s.coords_flat_bohr();
        let n = s.n_atoms().max(1);
        let c: [f64; 3] =
            std::array::from_fn(|j| (0..n).map(|i| x[3 * i + j]).sum::<f64>() / n as f64);
        let far = (0..s.n_atoms())
            .map(|i| {
                (0..3)
                    .map(|j| (x[3 * i + j] - c[j]).powi(2))
                    .sum::<f64>()
                    .sqrt()
            })
            .fold(0.0, f64::max);
        if !far.is_finite() || far > FAR_APART_BOHR {
            return Err(MoptError::conv(format!(
                "{label}: structure {} has an atom {:.3e} Angstrom from its centre (it has \
                 blown apart); not searched",
                k + 1,
                far * crate::domain::Units::BOHR_TO_ANGSTROM
            )));
        }
    }
    Ok(())
}

/// The kept frames of the dynamics, to the outcome and the trajectory of the job; a step
/// that blows up ends the dynamics.
struct Kept<'a> {
    template: &'a MolecularSystem,
    every: usize,
    frames: Vec<PathPoint>,
    sink: &'a mut dyn FrameSink,
    label: &'a str,
    stop: MapperStop,
    last: Option<Col<f64>>,
    blew_up: Option<usize>,
    /// The last step: positions, velocities, and the energies without and with the bias.
    end: Option<(Col<f64>, Col<f64>, f64, f64)>,
}

impl MdObserver for Kept<'_> {
    fn on_step(&mut self, report: &MdStepReport) -> MoptResult<()> {
        if self.blew_up.is_some() {
            return Ok(());
        }
        if let Some(last) = &self.last {
            let jump = (0..report.x.nrows() / 3)
                .map(|i| {
                    (0..3)
                        .map(|j| (report.x[3 * i + j] - last[3 * i + j]).powi(2))
                        .sum::<f64>()
                        .sqrt()
                })
                .fold(0.0, f64::max);
            if !jump.is_finite() || jump > BLOW_UP_JUMP_BOHR {
                self.blew_up = Some(report.step);
                return Ok(());
            }
        }
        self.last = Some(report.x.clone());
        self.end = Some((
            report.x.clone(),
            report.v.clone(),
            report.raw_energy,
            report.energy,
        ));
        if !report.step.is_multiple_of(self.every.max(1)) {
            return Ok(());
        }
        let mut sys = self.template.with_flat_coords_bohr(&report.x)?;
        if report.cell.is_some() {
            sys.cell.clone_from(&report.cell);
        }
        let point = Configuration::Structure(sys);
        self.sink.push(
            &point,
            report.raw_energy,
            &format!("{} step {}", self.label, report.step),
        )?;
        self.frames.push(PathPoint {
            system: point,
            energy: report.raw_energy,
            biased_energy: report.energy,
        });
        Ok(())
    }

    fn should_stop(&mut self) -> bool {
        self.blew_up.is_some() || self.stop.check()
    }
}

impl CommandJobs<'_> {
    /// [`crate::workflow::mapper::MapperJobs::pair_search`] with `calc`.
    pub(super) fn pair_search_with(
        &self,
        calc: &dyn Calculator,
        request: &PairSearchRequest,
        frames: &mut dyn FrameSink,
    ) -> MoptResult<PairSearchOutcome> {
        let stop = self.stop_signal();
        if stop.check() {
            return Ok(PairSearchOutcome::stopped());
        }
        let structures = request
            .structures
            .iter()
            .map(|p| p.require_structure(&request.label).cloned())
            .collect::<MoptResult<Vec<_>>>()?;
        check_structures(&request.label, &structures)?;
        let template = structures
            .first()
            .ok_or_else(|| MoptError::config(format!("{}: no structure", request.label)))?
            .clone();
        let o = &self.steps().minimum;
        // The hard constraints of the run: the gradients of the images projected onto them,
        // the fixed atoms fixed, and every point put back onto the others.
        let held = Self::path_constraints(self.steps(), &template)?;
        let images: Vec<ImageState> = structures
            .iter()
            .map(|s| ImageState {
                template: s.clone(),
                excited_state: o.excited_state,
                constraints: held.set.clone(),
            })
            .collect();
        let mut surface = StructureSurface::new(calc, images)?;
        if !o.model_function.is_empty() {
            if let Some(expr) = state_surface(&o.model_function, o.excited_state)? {
                surface = surface.with_states(expr);
            }
        }
        let input = PairInput {
            points: structures
                .iter()
                .map(MolecularSystem::coords_flat_bohr)
                .collect(),
            geometry: PointGeometry::of_structure(
                Some(&template),
                calc.is_rigid_body_invariant() && !held.pins_frame,
                held.frozen.clone(),
            )
            .with_manifold(held.manifold.clone()),
            structures: true,
            elements: Some(template.elements.clone()),
            template: Some(template.clone()),
            fragments: Some(
                match (&request.dissociation_atoms, request.dissociation_scale) {
                    (Some(atoms), scale) => fragment_test_on(
                        template.clone(),
                        scale.unwrap_or(super::ieipmain_methods::DISSOCIATION_SCALE),
                        atoms.clone(),
                    ),
                    (None, Some(scale)) => fragment_test_with(template.clone(), None, scale),
                    (None, None) => fragment_test(template.clone(), None),
                },
            ),
            ..PairInput::default()
        };
        self.run_search(
            &surface,
            &input,
            request,
            frames,
            &Configuration::Structure(template),
            None,
        )
    }

    /// [`crate::workflow::mapper::MapperJobs::pair_search`] on the objective of plain
    /// variables `objective`.
    pub(super) fn pair_search_variables(
        &self,
        objective: &dyn Objective,
        request: &PairSearchRequest,
        frames: &mut dyn FrameSink,
    ) -> MoptResult<PairSearchOutcome> {
        let stop = self.stop_signal();
        if stop.check() {
            return Ok(PairSearchOutcome::stopped());
        }
        let points = request
            .structures
            .iter()
            .map(|p| {
                p.point().cloned().ok_or_else(|| {
                    MoptError::config(format!(
                        "{}: a structure in a search of plain variables",
                        request.label
                    ))
                })
            })
            .collect::<MoptResult<Vec<_>>>()?;
        let first = points
            .first()
            .ok_or_else(|| MoptError::config(format!("{}: no point", request.label)))?
            .clone();
        if points
            .iter()
            .any(|p| p.values.iter().any(|v| !v.is_finite()))
        {
            return Err(MoptError::conv(format!(
                "{}: a point with values that are not finite; not searched",
                request.label
            )));
        }
        let surface = VariableSurface(objective);
        let input = PairInput {
            points: points.iter().map(|p| p.values.clone()).collect(),
            structures: false,
            ..PairInput::default()
        };
        self.run_search(
            &surface,
            &input,
            request,
            frames,
            &Configuration::Variables(first),
            None,
        )
    }

    /// Run the search of `request` on `surface` from `input`; its points are those of
    /// `template` (a structure, or a point of plain variables).
    pub(super) fn run_search(
        &self,
        surface: &dyn SearchSurface,
        input: &PairInput,
        request: &PairSearchRequest,
        frames: &mut dyn FrameSink,
        template: &Configuration,
        cell: Option<&crate::opt::cell::VariableCell>,
    ) -> MoptResult<PairSearchOutcome> {
        let stop = self.stop_signal();
        let watched = Watched {
            inner: surface,
            stop: &stop,
            stopped: AtomicBool::new(false),
            frames: Mutex::new(frames),
            label: &request.label,
            count: AtomicUsize::new(0),
            band: request
                .method
                .eq_ignore_ascii_case("neb")
                .then(|| AtomicUsize::new(0)),
            variables: template.is_variables().then(|| template.clone()),
        };
        let dir = stop.dir().join(job_files_dir(request.job));
        // The full log of the search (every point with its structure) goes to its own file.
        let stem = file_stem(&request.label);
        let log = Logger::silent();
        if self.steps().save_job_steps {
            std::fs::create_dir_all(&dir)?;
            log.add_file(dir.join(format!("{stem}.log")), Level::Debug)?;
        }
        let run = SearchRun {
            save_steps: self.steps().save_job_steps,
            dir: &dir,
            stems: [stem.clone(), stem.clone()],
            log: &log,
            figures: false,
            commands: &super::pair_settings::COMMAND_PARSERS,
        };
        let settings = PairSettings::new(request.settings.clone());
        let mut files = if self.steps().save_job_steps && request.method.eq_ignore_ascii_case("neb")
        {
            // A replay or another chain with the same label keeps the earlier attempts.
            let mut attempt = 1;
            let path = loop {
                let path = dir.join(format!("{stem}_{attempt:03}"));
                match std::fs::create_dir(&path) {
                    Ok(()) => break path,
                    Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => attempt += 1,
                    Err(e) => return Err(e.into()),
                }
            };
            let files = match (cell, template) {
                (Some(map), _) => ChainFiles::variable_cell(&path, "neb", map)?,
                (_, Configuration::Structure(s)) => {
                    ChainFiles::new(path, "neb", ChainPoints::Structures(Box::new(s.clone())))?
                }
                (_, Configuration::Variables(p)) => {
                    ChainFiles::new(path, "neb", ChainPoints::Variables((*p.names).clone()))?
                }
            };
            Some(files.with_atoms(self.trajectory_atoms()))
        } else {
            None
        };
        let result = PairMethodRegistry::default().run_observed(
            &request.method,
            input,
            &settings,
            &watched,
            &run,
            files.as_mut().map(|f| f as &mut dyn ChainObserver),
        );
        let r = match result {
            Ok(r) => r,
            Err(_) if watched.stopped.load(Ordering::SeqCst) => {
                return Ok(PairSearchOutcome::stopped());
            }
            // What a method refuses about its start (ADDF from a structure whose Hessian has a
            // negative eigenvalue: a node of the network that is not quite a minimum) is an
            // outcome of the job, which fails alone; the settings were checked before the run.
            Err(MoptError::Config(message)) => {
                return Err(MoptError::conv(format!("{}: {message}", request.label)));
            }
            Err(e) => return Err(e.context(request.label.clone())),
        };
        let mut estimates = Vec::with_capacity(r.transition_states.len());
        for (x, energy) in &r.transition_states {
            estimates.push(Stationary {
                index_wavenumbers: None,
                system: template.with_flat(x)?,
                energy: *energy,
                wavenumbers: None,
                dissociation: None,
            });
        }
        let mut path = Vec::new();
        for (x, energy) in r.path.iter().flatten() {
            path.push(PathPoint {
                system: template.with_flat(x)?,
                energy: *energy,
                biased_energy: *energy,
            });
        }
        self.note(&format!(
            "{}: {} ({} transition-state estimate(s); {} gradients, {} Hessians, {} energies)",
            request.label,
            r.end,
            estimates.len(),
            r.counts.gradients,
            r.counts.hessians,
            r.counts.energies
        ));
        Ok(PairSearchOutcome {
            interrupted: r.interrupted,
            converged: r.converged,
            end: r.end,
            estimates,
            path,
        })
    }

    /// [`crate::workflow::mapper::MapperJobs::dynamics`] with `calc`.
    pub(super) fn dynamics_with(
        &self,
        calc: &dyn Calculator,
        start: &MolecularSystem,
        request: &DynamicsRequest,
        frames: &mut dyn FrameSink,
    ) -> MoptResult<DynamicsOutcome> {
        let stop = self.stop_signal();
        if stop.check() {
            return Ok(DynamicsOutcome {
                interrupted: true,
                frames: Vec::new(),
                blew_up: None,
                end: None,
            });
        }
        let o = &self.steps().minimum;
        // Bias potentials (walls about a fixed centre) break the invariance under translation
        // and rotation.
        let rigid = match (
            calc.is_rigid_body_invariant() && request.terms.is_empty(),
            start.has_pbc(),
        ) {
            (false, _) => RigidMotion::None,
            (true, true) => RigidMotion::Translation,
            (true, false) => RigidMotion::TranslationRotation,
        };
        let settings = MdSettings {
            time_step: request.time_step,
            steps: request.steps,
            temperature: request.temperature,
            thermostat: "bussi".to_string(),
            thermostat_settings: ThermostatSettings::default(),
            seed: request.seed,
            rigid,
            velocities: match &request.velocities {
                Some(v) => InitialVelocities::Given(v.clone()),
                None => MdSettings::default().velocities,
            },
            ..MdSettings::default()
        };
        let mut md = MolecularDynamics::new(
            settings,
            MdSpace::atoms(&start.masses_amu())?,
            start.coords_flat_bohr(),
            Some(start.clone()),
        )?;
        let mut expr =
            state_surface(&o.model_function, o.excited_state)?.unwrap_or_else(PesExpr::single);
        for (label, term) in &request.terms {
            expr = expr.plus_term(label, Arc::clone(term));
        }
        let surface = MoleculeMdSurface::new(
            CalculatorSet::single(calc),
            expr,
            start.clone(),
            request.terms.clone(),
        )?
        .telling(self.job_log().clone());
        let mut kept = Kept {
            template: start,
            every: request.frame_every,
            frames: Vec::new(),
            sink: frames,
            label: &request.label,
            stop: stop.clone(),
            last: None,
            blew_up: None,
            end: None,
        };
        let end = run_md(&mut md, &surface, &mut [&mut kept]).context(request.label.clone())?;
        self.job_log().debug(&format!(
            "{}: {} steps, {} frames kept",
            request.label,
            md.evaluations(),
            kept.frames.len()
        ));
        if let Some(step) = kept.blew_up {
            self.job_log().warn(&format!(
                "{}: the dynamics blew up at step {step} (an atom moved more than \
                 {BLOW_UP_JUMP_BOHR} Bohr in one step); it ended there",
                request.label
            ));
        }
        let last = match (kept.end, kept.blew_up) {
            (Some((x, v, energy, biased_energy)), None) => {
                let mut sys = start.with_flat_coords_bohr(&x)?;
                sys.cell.clone_from(&start.cell);
                Some(crate::workflow::mapper::DynamicsEnd {
                    point: Configuration::Structure(sys),
                    velocities: v,
                    energy,
                    biased_energy,
                })
            }
            _ => None,
        };
        Ok(DynamicsOutcome {
            interrupted: end.is_none() && kept.blew_up.is_none() && stop.stopped(),
            frames: kept.frames,
            blew_up: kept.blew_up,
            end: last,
        })
    }
}

/// The force thresholds of the high-index saddle dynamics of the mapper as a share of those
/// of the transition-state search: well below them, so that a search that starts beside a
/// stationary point (a transition state pushed along a soft mode) does not end at once there,
/// on a surface whose forces are small (argon: epsilon / sigma is about 6e-5 Hartree/Bohr).
pub const HISD_CRITERIA_SHARE: f64 = 0.01;

/// The hand-off to the optimization at the order sought: the reflected subspace turning by
/// less than this (the sine of the largest principal angle between iterations), the value of
/// `docs/source/high_index_saddles.md`, at an RMS gradient below the scaled RMS threshold.
pub const HISD_HAND_OFF_TURN: f64 = 0.1;

/// High-index saddle dynamics of order `order` from `start` with `directions` on `surface`
/// (structures: mass-weighted, the rigid motions left out), with the force thresholds
/// [`HISD_CRITERIA_SHARE`] of `criteria` (those of the transition-state search), handed over at
/// that RMS force with [`HISD_HAND_OFF_TURN`]; every point goes to `frames`.
pub(super) fn run_hisd(
    surface: &dyn crate::workflow::pair_search::surface::SearchSurface,
    start: &Configuration,
    order: usize,
    directions: &[faer::Col<f64>],
    label: &str,
    frames: &mut dyn FrameSink,
    stop: &super::mapper_jobs::MapperStop,
    criteria: &crate::opt::convergence::ConvergenceCriteria,
    frame: crate::path::hisd::SearchFrame,
) -> MoptResult<crate::workflow::mapper::HisdOutcome> {
    use crate::path::hisd::{HandOff, HighIndexSaddleSearch, HisdEnd, HisdSettings};
    use crate::path::points::{PointValue, SearchAction};
    let mut settings = HisdSettings::hiosd(order);
    settings.criteria = crate::opt::convergence::ConvergenceCriteria::new(
        criteria.max_force.value() * HISD_CRITERIA_SHARE,
        criteria.rms_force.value() * HISD_CRITERIA_SHARE,
        criteria.max_displacement.value(),
        criteria.rms_displacement.value(),
    );
    // The hand-off at the RMS force of the scaled criteria: the dynamics climbs until its
    // gradient is well below what the refinement would call converged.
    settings.hand_off = Some(HandOff {
        rms_gradient: criteria.rms_force.value() * HISD_CRITERIA_SHARE,
        projector_change: HISD_HAND_OFF_TURN,
    });
    let x0 = start.flat();
    let mut search = HighIndexSaddleSearch::new(settings, &x0, directions)?.with_frame(frame)?;
    let mut interrupted = false;
    while let SearchAction::Evaluate(points) = search.ask() {
        if stop.check() {
            interrupted = true;
            break;
        }
        let mut values = Vec::with_capacity(points.len());
        for (k, request) in points.iter().enumerate() {
            let v = surface.evaluate(&request.x, 0, request)?;
            if k == 0 {
                frames.push(
                    &start.with_flat(&request.x)?,
                    v.energy,
                    &format!("{label} iteration {}", search.iterations()),
                )?;
            }
            values.push(PointValue {
                energy: v.energy,
                gradient: request.gradient.then_some(v.gradient),
                hessian: v.hessian,
            });
        }
        search.tell(values)?;
    }
    let end = search.end();
    Ok(crate::workflow::mapper::HisdOutcome {
        point: start.with_flat(search.point())?,
        energy: search.energy(),
        settled: matches!(end, Some(HisdEnd::Converged | HisdEnd::HandOff)),
        end: end.map_or_else(
            || "stopped".to_string(),
            |e| {
                let curvatures: Vec<String> = search
                    .curvatures()
                    .iter()
                    .map(|c| format!("{c:.3e}"))
                    .collect();
                format!(
                    "{e:?} after {} iterations, tracked curvatures [{}]",
                    search.iterations(),
                    curvatures.join(", ")
                )
            },
        ),
        directions: search.point_directions(),
        interrupted,
    })
}
