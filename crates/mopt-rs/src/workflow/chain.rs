//! Driving the relaxation of a chain: the images are evaluated on a surface, and after
//! every iteration the observers are called, which may write files, draw figures, stop the
//! run, or change the chain (any processing a caller wants between two iterations).
//!
//! The surface of a chain of structures ([`CalculatorSurface`]) is a calculator, to which
//! bias potentials are added as in `optmain`; its Hessians are those of the calculator or
//! of a model Hessian; and constraints take their directions out of the gradient and the
//! Hessian of every image, each image holding the values it started with
//! ([`ChainConstraints`]; [`KeepConstraints`] puts the images back onto them after every
//! iteration). Anything that is a calculator can be the surface: a combined surface of
//! several states or an ONIOM surface as well.
//!
//! The bias potentials of a chain: every image evaluates its own copy of every term
//! ([`fork_term`]), so that a term that keeps a warm start for an inner problem (the
//! auxiliary variables of `-smp`, `-aerp`, `-aerpv2`) keeps one per image and the values
//! do not depend on the order or the threads the images are evaluated in. A term that
//! changes with the simulated time or keeps state that changes its value (metadynamics) has
//! no meaning on a chain, whose surface is never told which evaluation is accepted; such
//! terms are refused ([`check_chain_terms`]). A term ramped over the iterations follows the
//! iteration of the chain relaxation.
//!
//! MultiOptPy: `neb.py:NEB.execute` (the loop; MultiOptPy has no hooks).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use faer::Col;
use rayon::prelude::*;

use super::driver::CalculatorSet;
use crate::domain::{EvalRequest, MolecularSystem};
use crate::error::{MoptError, MoptResult, ResultExt};
use crate::model_hessian::ModelHessian;
use crate::path::band::images::{FailedImage, ImageFailure, RETRIED_SHARES};
use crate::path::band::machine::{ChainAction, ChainEnd, ChainRelaxation, ChainReport, ImageValue};
use crate::pes::calculator::evaluate_checked;
use crate::pes::combine::PesExpr;
use crate::pes::objective::Objective;
use crate::pes::term::{check_term_output, fork_term, History, PesTerm, TermContext};
use crate::pes::Calculator;

#[path = "chain_constraints.rs"]
mod chain_constraints;
pub(crate) use chain_constraints::follow_resolved;
pub use chain_constraints::{project_image, ChainConstraints, SharedChainConstraints};

/// Labelled terms of a surface.
pub type Terms = Vec<(String, Arc<dyn PesTerm>)>;

/// Refuse the terms that have no meaning on a surface whose evaluations are never told
/// which one is accepted (a chain, the pairs and points of `ieipmain`): a term that changes
/// with the simulated time ([`History::TimeDependent`]), and a term that keeps state that
/// changes its value ([`History::Stateful`]: metadynamics, which deposits nothing without
/// accepted evaluations) unless `warm_start(label)` says that its state is only the warm
/// start of an inner problem (the auxiliary variables of `-smp`, `-aerp`, `-aerpv2`), which
/// [`fork_term`] then copies for every image. `what` names the run in the message.
///
/// MultiOptPy: no counterpart (new in mopt-rs).
pub fn check_chain_terms(
    terms: &[(String, Arc<dyn PesTerm>)],
    what: &str,
    warm_start: &dyn Fn(&str) -> bool,
) -> MoptResult<()> {
    for (label, term) in terms {
        match term.history() {
            History::Static | History::IterationRamp => {}
            History::TimeDependent => {
                return Err(MoptError::unsupported(format!(
                    "{what}: the bias potential '{label}' changes with the simulated time; \
                     the structures of {what} are not points of a trajectory in time (it is \
                     for molecular dynamics, mdmain)"
                )))
            }
            History::Stateful if warm_start(label) => {
                fork_term(term).context(label.clone())?;
            }
            History::Stateful => {
                return Err(MoptError::unsupported(format!(
                    "{what}: the bias potential '{label}' keeps state that changes its value \
                     (deposits of accepted structures, as metadynamics); {what} is never told \
                     which evaluation is accepted, so the term would do nothing or depend on \
                     the order of the evaluations (it is for molecular dynamics, mdmain)"
                )))
            }
        }
    }
    Ok(())
}

/// The labels of the bias potentials whose state is the warm start of an inner problem
/// (`-smp`, `-aerp`, `-aerpv2`: auxiliary variables minimized at every evaluation), by the
/// option key their labels begin with.
pub const WARM_START_KEYS: &[&str] = &[
    "spacer_model_potential",
    "asymmetric_ellipsoidal_repulsive_potential",
];

/// Whether `label` (an option key and a number) is one of [`WARM_START_KEYS`].
pub fn warm_start_label(label: &str) -> bool {
    WARM_START_KEYS.iter().any(|key| label.starts_with(key))
}

/// Where the images of a chain get their energies and gradients.
pub trait ChainSurface {
    /// What the surface is, for the log.
    fn description(&self) -> String;

    /// Energy and gradient of image `index` at `point`; with `need_hessian` also a Hessian
    /// of the image (a model Hessian or an exact one) if the surface has one.
    fn evaluate(
        &self,
        point: &Col<f64>,
        index: usize,
        need_hessian: bool,
    ) -> MoptResult<ImageValue>;

    /// The values of the images of one iteration (`images`: the number of each image and
    /// its point), in their order. Here one after the other; a surface whose images are
    /// calculations that do not depend on each other does them at the same time.
    fn evaluate_all(
        &self,
        images: &[(usize, &Col<f64>)],
        need_hessian: bool,
    ) -> Vec<MoptResult<ImageValue>> {
        images
            .iter()
            .map(|(index, point)| self.evaluate(point, *index, need_hessian))
            .collect()
    }

    /// Told before the images of `iteration` are evaluated (surfaces that change with
    /// the iteration, such as bias potentials whose strength is raised step by step).
    fn begin_iteration(&self, _iteration: usize) {}

    /// Told that the images `dropped` (sorted numbers in the chain as it was) were taken
    /// out of the chain: what the surface keeps image by image follows.
    fn images_dropped(&self, _dropped: &[usize]) {}

    /// Told that images were added to the chain (the images have other numbers now).
    fn images_added(&self) {}

    /// Whether all values of the iteration that was evaluated last were calculated on the
    /// surface itself. A surface that answers some images from a model says no then, and
    /// a chain does not count as relaxed on such values.
    fn values_are_calculated(&self) -> bool {
        true
    }

    /// Asks that all values of the next iteration be calculated (`check`: because a chain
    /// looked relaxed with values that were not).
    fn calculate_next(&self, _check: bool) {}

    /// Told the largest force of the images that move, after every iteration.
    fn told(&self, _largest_force: f64) {}

    /// What was said while the images of the iteration were evaluated (warnings of the
    /// calculator, the output of a program, `<S**2>`), handed over once.
    fn take_notes(&self) -> Vec<String> {
        Vec::new()
    }

    /// The energies without the bias potentials of the images evaluated since this was
    /// called last, by the number of the image (empty: the surface has no bias potentials).
    fn take_bare_energies(&self) -> Vec<(usize, f64)> {
        Vec::new()
    }
}

/// A function of plain variables as the surface of a chain.
pub struct ObjectiveSurface<'a>(pub &'a dyn Objective);

impl ChainSurface for ObjectiveSurface<'_> {
    fn description(&self) -> String {
        self.0.description()
    }

    fn evaluate(
        &self,
        point: &Col<f64>,
        _index: usize,
        need_hessian: bool,
    ) -> MoptResult<ImageValue> {
        let found = self.0.evaluate(point, need_hessian)?;
        Ok(ImageValue {
            energy: found.value,
            gradient: found.gradient,
            hessian: found.hessian,
        })
    }
}

/// Makes the surface of the states of one image ([`CalculatorSurface::with_states`]).
pub type StateFactory = Arc<dyn Fn() -> MoptResult<PesExpr> + Send + Sync>;

/// A calculator as the surface of a chain of structures: every image is `template` with
/// the coordinates of the image (Bohr).
pub struct CalculatorSurface<'a> {
    calculator: &'a dyn Calculator,
    template: MolecularSystem,
    terms: Vec<(String, Arc<dyn PesTerm>)>,
    /// The copies of the terms of every image ([`fork_term`]); they follow the images when
    /// some are taken out, and start anew when images are added.
    image_terms: Mutex<Vec<Terms>>,
    model: Option<Box<dyn ModelHessian>>,
    /// The constraints of the images (shared with the observer that restores them, which
    /// keeps them in step with the images).
    constraints: Option<SharedChainConstraints>,
    /// The energies without the bias potentials since they were taken last.
    bare: Mutex<Vec<(usize, f64)>>,
    excited_state: usize,
    /// Makes the surface that combines electronic states for an image (`-mf`).
    state_factory: Option<StateFactory>,
    /// The surfaces of the images made so far (they keep what their combiners learn
    /// from iteration to iteration, such as the branching plane of a projection).
    states: Mutex<Vec<PesExpr>>,
    /// What the calculator said since the notes were taken last.
    notes: Mutex<Vec<String>>,
    iteration: AtomicUsize,
}

impl<'a> CalculatorSurface<'a> {
    /// The surface of `calculator` for structures with the atoms, charge, spin, and cell
    /// of `template`.
    pub fn new(calculator: &'a dyn Calculator, template: MolecularSystem) -> Self {
        Self {
            calculator,
            template,
            terms: Vec::new(),
            image_terms: Mutex::new(Vec::new()),
            model: None,
            constraints: None,
            bare: Mutex::new(Vec::new()),
            excited_state: 0,
            state_factory: None,
            states: Mutex::new(Vec::new()),
            notes: Mutex::new(Vec::new()),
            iteration: AtomicUsize::new(0),
        }
    }

    /// Bias potentials (with their labels) added to the energy, the gradient, and the
    /// Hessian; every image evaluates its own copy of each ([`fork_term`]: a term that
    /// keeps state and cannot be copied is an error when an image is evaluated, see
    /// [`check_chain_terms`]).
    pub fn with_terms(mut self, terms: Vec<(String, Arc<dyn PesTerm>)>) -> Self {
        self.terms = terms;
        self
    }

    /// The Hessians of the images from this model instead of from the calculator.
    pub fn with_model_hessian(mut self, model: Box<dyn ModelHessian>) -> Self {
        self.model = Some(model);
        self
    }

    /// The constraints of the images: their directions are taken out of the gradient of
    /// every image, and its Hessian is projected as the optimizer projects it
    /// ([`project_image`]). The same constraints go to the [`KeepConstraints`] of the run,
    /// which keeps them in step with the images.
    pub fn with_constraints(mut self, constraints: SharedChainConstraints) -> Self {
        self.constraints = Some(constraints);
        self
    }

    /// The copies of the terms of image `index` (made at its first evaluation).
    fn terms_of(&self, index: usize) -> MoptResult<Terms> {
        if self.terms.is_empty() {
            return Ok(Vec::new());
        }
        let mut all = self
            .image_terms
            .lock()
            .map_err(|_| MoptError::calc("the bias potentials of the images are not available"))?;
        while all.len() <= index {
            let copies = self
                .terms
                .iter()
                .map(|(label, term)| Ok((label.clone(), fork_term(term).context(label.clone())?)))
                .collect::<MoptResult<Terms>>()?;
            all.push(copies);
        }
        Ok(all[index].clone())
    }

    /// The electronic state (0: the ground state).
    pub fn with_excited_state(mut self, state: usize) -> Self {
        self.excited_state = state;
        self
    }

    /// The chain is relaxed on a surface that combines electronic states (a model
    /// function of two states, a projection for crossings: `-mf`), one made by `factory`
    /// for every image. The states are those of the surface; the state of
    /// [`Self::with_excited_state`] is not used then.
    pub fn with_states(mut self, factory: StateFactory) -> Self {
        self.state_factory = Some(factory);
        self
    }

    /// The structure of an image.
    pub fn system(&self, point: &Col<f64>) -> MoptResult<MolecularSystem> {
        if point.nrows() != self.template.dof() {
            return Err(MoptError::config(format!(
                "an image of {} coordinates for {} atoms",
                point.nrows(),
                self.template.n_atoms()
            )));
        }
        self.template.with_flat_coords_bohr(point)
    }
}

impl CalculatorSurface<'_> {
    /// What the calculator said about `out`, the value of `image` (in the state `state`).
    fn said(
        &self,
        image: usize,
        state: Option<&str>,
        out: &crate::domain::EvalOutput,
    ) -> Vec<String> {
        let of = match state {
            Some(state) => format!("image {image} ({state})"),
            None => format!("image {image}"),
        };
        crate::pes::calculator::said(self.calculator, &of, out)
    }

    fn keep_notes(&self, said: Vec<String>) {
        if let Ok(mut notes) = self.notes.lock() {
            notes.extend(said);
        }
    }

    fn keep_bare(&self, index: usize, bare: Option<f64>) {
        if let (Some(energy), Ok(mut all)) = (bare, self.bare.lock()) {
            all.push((index, energy));
        }
    }

    /// Whether the images of an iteration are calculations that do not depend on each
    /// other or on their order: no calculator that asks for its structures in order
    /// (followed states, an environment that is relaxed from the nearest one before), and
    /// no surfaces of several states, which learn from image to image.
    fn images_are_independent(&self) -> bool {
        !self.calculator.evaluates_in_order() && self.state_factory.is_none()
    }
}

impl ChainSurface for CalculatorSurface<'_> {
    fn description(&self) -> String {
        self.calculator.description()
    }

    fn begin_iteration(&self, iteration: usize) {
        self.iteration.store(iteration, Ordering::Relaxed);
    }

    fn images_dropped(&self, dropped: &[usize]) {
        // The constraints follow the images in the observer that restores them
        // (`KeepConstraints`); here what the surface keeps image by image.
        if let Ok(mut states) = self.states.lock() {
            for &image in dropped.iter().rev() {
                if image < states.len() {
                    states.remove(image);
                }
            }
        }
        if let Ok(mut terms) = self.image_terms.lock() {
            for &image in dropped.iter().rev() {
                if image < terms.len() {
                    terms.remove(image);
                }
            }
        }
    }

    fn images_added(&self) {
        // The numbers of the images changed: the surfaces of the states and the copies of
        // the terms start anew.
        if let Ok(mut states) = self.states.lock() {
            states.clear();
        }
        if let Ok(mut terms) = self.image_terms.lock() {
            terms.clear();
        }
    }

    fn take_notes(&self) -> Vec<String> {
        self.notes
            .lock()
            .map(|mut notes| std::mem::take(&mut *notes))
            .unwrap_or_default()
    }

    fn take_bare_energies(&self) -> Vec<(usize, f64)> {
        self.bare
            .lock()
            .map(|mut bare| std::mem::take(&mut *bare))
            .unwrap_or_default()
    }

    fn evaluate(
        &self,
        point: &Col<f64>,
        index: usize,
        need_hessian: bool,
    ) -> MoptResult<ImageValue> {
        let (value, said, bare) = self.value_of(point, index, need_hessian)?;
        self.keep_notes(said);
        self.keep_bare(index, bare);
        Ok(value)
    }

    fn evaluate_all(
        &self,
        images: &[(usize, &Col<f64>)],
        need_hessian: bool,
    ) -> Vec<MoptResult<ImageValue>> {
        let one = |(index, point): &(usize, &Col<f64>)| self.value_of(point, *index, need_hessian);
        let found: Vec<MoptResult<Found>> = if images.len() > 1 && self.images_are_independent() {
            images.par_iter().map(one).collect()
        } else {
            images.iter().map(one).collect()
        };
        // The notes in the order of the images, whichever was finished first.
        found
            .into_iter()
            .zip(images)
            .map(|(result, (index, _))| {
                result.map(|(value, said, bare)| {
                    self.keep_notes(said);
                    self.keep_bare(*index, bare);
                    value
                })
            })
            .collect()
    }
}

/// The value of an image, what the calculator said about it, and its energy without the
/// bias potentials (when there are any).
type Found = (ImageValue, Vec<String>, Option<f64>);

impl CalculatorSurface<'_> {
    /// The value of image `index` at `point` and what the calculator said about it.
    fn value_of(&self, point: &Col<f64>, index: usize, need_hessian: bool) -> MoptResult<Found> {
        let mut said = Vec::new();
        let sys = self.system(point)?;
        let base = if need_hessian && self.model.is_none() {
            EvalRequest::ENERGY_GRAD_HESS
        } else {
            EvalRequest::ENERGY_GRAD
        };
        let request = EvalRequest {
            excited_state: self.excited_state,
            ..base
        };
        let out = match &self.state_factory {
            None => {
                let out = evaluate_checked(self.calculator, &sys, &request)?;
                said.extend(self.said(index, None, &out));
                out
            }
            Some(factory) => {
                let mut states = self.states.lock().map_err(|_| {
                    MoptError::calc("the surfaces of the states of the chain are not available")
                })?;
                while states.len() <= index {
                    states.push(factory()?);
                }
                // The states are those of the surface; the tasks carry them.
                let tasks = states[index].tasks(&sys, &base)?;
                let results = CalculatorSet::single(self.calculator).evaluate(&tasks)?;
                for (task, out) in tasks.iter().zip(&results) {
                    said.extend(self.said(index, Some(&task.label), out));
                }
                let ctx = TermContext::at_iteration(
                    self.iteration.load(Ordering::Relaxed),
                    &self.template,
                );
                // The Hessians of the terms of the states only when the chain uses them.
                states[index]
                    .evaluate_as_requested(&sys, &base, results, &ctx)?
                    .total
            }
        };
        let bare_energy = out.energy.value();
        let mut energy = bare_energy;
        let mut gradient = out.gradient_bohr;
        let mut hessian = match (&self.model, need_hessian) {
            (_, false) => None,
            (Some(model), true) => Some(model.hessian(&sys, &gradient)?),
            (None, true) => Some(out.hessian_bohr.ok_or_else(|| {
                MoptError::calc(format!(
                    "calculator '{}' returned no Hessian",
                    self.calculator.name()
                ))
            })?),
        };
        let terms = self.terms_of(index)?;
        if !terms.is_empty() {
            let ctx =
                TermContext::at_iteration(self.iteration.load(Ordering::Relaxed), &self.template);
            for (label, term) in &terms {
                let bias = term
                    .evaluate(&sys, &ctx, need_hessian)
                    .context(label.clone())?;
                check_term_output(label, &bias)?;
                energy += bias.energy.value();
                gradient += &bias.gradient_bohr;
                if let Some(total) = hessian.as_mut() {
                    let of_bias = bias.hessian_bohr.as_ref().ok_or_else(|| {
                        MoptError::calc(format!("bias potential '{label}' returned no Hessian"))
                    })?;
                    *total += of_bias;
                }
            }
        }
        if let Some(shared) = &self.constraints {
            // The set of the image out of the lock: the projection evaluates the constraints.
            let set = shared
                .lock()
                .map_err(|_| MoptError::calc("the constraints of the chain are not available"))?
                .set(index)
                .cloned();
            if let Some(set) = set {
                (gradient, hessian) = project_image(&set, &sys, &gradient, hessian)
                    .context(format!("the constraints of image {index}"))?;
            }
        }
        Ok((
            ImageValue {
                energy,
                gradient,
                hessian,
            },
            said,
            (!terms.is_empty()).then_some(bare_energy),
        ))
    }
}

/// Observer that puts the structures of the chain back onto their constraints after
/// every iteration, and keeps the constraints of the images in step with them: the
/// constraints of images that were taken out go with them, and where the images were put
/// elsewhere along the chain (redistributed, added, or a chain of another length set by an
/// observer) the images hold the values interpolated at their fractions of the length of the
/// chain ([`ChainConstraints::follow`]).
pub struct KeepConstraints<C: HeldByImages = ChainConstraints> {
    constraints: Arc<Mutex<C>>,
    /// The chain as it was restored last (what the constraints belong to).
    last: Option<Vec<Col<f64>>>,
}

/// The constraints of every image of a chain, as [`KeepConstraints`] keeps them in step with
/// the images: those of structures ([`ChainConstraints`]) or those of the points of another
/// vector of variables (the atoms and the cell of `nebmain -vc`).
///
/// MultiOptPy: no counterpart (new in mopt-rs).
pub trait HeldByImages: Send {
    /// Images.
    fn images(&self) -> usize;
    /// The images `dropped` (sorted numbers) taken out.
    fn drop_images(&mut self, dropped: &[usize]);
    /// The constraints of the images of `after`, a chain made from `before`.
    fn follow(&mut self, before: &[Col<f64>], after: &[Col<f64>]) -> MoptResult<()>;
    /// `chain` put back onto the constraints of its images.
    fn restore(&self, chain: &[Col<f64>]) -> MoptResult<Vec<Col<f64>>>;
}

impl HeldByImages for ChainConstraints {
    fn images(&self) -> usize {
        self.len()
    }
    fn drop_images(&mut self, dropped: &[usize]) {
        ChainConstraints::drop_images(self, dropped);
    }
    fn follow(&mut self, before: &[Col<f64>], after: &[Col<f64>]) -> MoptResult<()> {
        ChainConstraints::follow(self, before, after)
    }
    fn restore(&self, chain: &[Col<f64>]) -> MoptResult<Vec<Col<f64>>> {
        ChainConstraints::restore(self, chain)
    }
}

impl<C: HeldByImages> KeepConstraints<C> {
    /// The observer of `constraints` (the same as those of the surface).
    pub fn new(constraints: Arc<Mutex<C>>) -> Self {
        Self {
            constraints,
            last: None,
        }
    }

    /// `chain` (the first chain) put onto the constraints of its images: explicit values
    /// (`-pc manual`, `-uc "... = VALUE"`) are held from the first evaluation on.
    pub fn restore_first(&mut self, chain: &[Col<f64>]) -> MoptResult<Vec<Col<f64>>> {
        let restored = self
            .constraints
            .lock()
            .map_err(|_| MoptError::calc("the constraints of the chain are not available"))?
            .restore(chain)?;
        self.last = Some(restored.clone());
        Ok(restored)
    }
}

impl<C: HeldByImages> ChainObserver for KeepConstraints<C> {
    fn after_iteration(
        &mut self,
        report: &ChainReport,
        machine: &mut ChainRelaxation,
    ) -> MoptResult<Flow> {
        let mut constraints = self
            .constraints
            .lock()
            .map_err(|_| MoptError::calc("the constraints of the chain are not available"))?;
        // A chain set by an observer after the last restoration (another number of
        // images): the constraints follow it from the chain they belonged to.
        if let Some(last) = &self.last {
            if last.len() == constraints.images()
                && report.chain.len() + count_dropped(report) != last.len()
            {
                constraints.follow(last, &report.chain)?;
            }
        }
        // The constraints of images that were taken out in this iteration go with them.
        let mut gone: Vec<usize> = report
            .failures
            .iter()
            .filter(|failure| failure.dropped)
            .map(|failure| failure.image)
            .collect();
        gone.sort_unstable();
        if constraints.images() == report.chain.len() + gone.len() {
            constraints.drop_images(&gone);
        }
        let next = machine.chain().to_vec();
        if report.redistributed || report.added > 0 || next.len() != constraints.images() {
            constraints.follow(&report.chain, &next)?;
        }
        let restored = constraints.restore(&next)?;
        drop(constraints);
        self.last = Some(restored.clone());
        machine.set_chain(restored)?;
        Ok(Flow::Continue)
    }
}

/// The images taken out of the chain in the iteration of `report`.
fn count_dropped(report: &ChainReport) -> usize {
    report.failures.iter().filter(|f| f.dropped).count()
}

/// What an observer wants after an iteration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Flow {
    /// Go on.
    Continue,
    /// End the run here.
    Stop,
}

/// Called after every iteration of a chain relaxation.
pub trait ChainObserver {
    /// `report` is what the iteration did; `machine` holds the chain of the next
    /// iteration, which may be replaced ([`ChainRelaxation::set_chain`]).
    fn after_iteration(
        &mut self,
        report: &ChainReport,
        machine: &mut ChainRelaxation,
    ) -> MoptResult<Flow>;
}

impl<F> ChainObserver for F
where
    F: FnMut(&ChainReport, &mut ChainRelaxation) -> MoptResult<Flow>,
{
    fn after_iteration(
        &mut self,
        report: &ChainReport,
        machine: &mut ChainRelaxation,
    ) -> MoptResult<Flow> {
        self(report, machine)
    }
}

/// How a relaxation went.
#[derive(Debug, Clone)]
pub struct ChainOutcome {
    /// The report of the last iteration (`None` if there was none).
    pub last: Option<ChainReport>,
    /// Iterations that were run.
    pub iterations: usize,
    /// Why the state machine ended (`None` when an observer stopped the run).
    pub end: Option<ChainEnd>,
}

/// Run `machine` on `surface` until it ends or an observer stops it.
pub fn relax_chain(
    machine: &mut ChainRelaxation,
    surface: &dyn ChainSurface,
    observers: &mut [&mut dyn ChainObserver],
) -> MoptResult<ChainOutcome> {
    let mut outcome = ChainOutcome {
        last: None,
        iterations: 0,
        end: None,
    };
    loop {
        let request = match machine.ask() {
            ChainAction::Evaluate(request) => request,
            ChainAction::Finished(end) => {
                outcome.end = Some(end);
                return Ok(outcome);
            }
        };
        if machine.is_last_iteration() {
            // What is reported at the end is calculated.
            surface.calculate_next(false);
        }
        surface.begin_iteration(machine.iteration());
        let policy = machine.on_failure();
        let mut values = Vec::with_capacity(request.images.len());
        let mut failures = Vec::new();
        let numbered: Vec<(usize, &Col<f64>)> = request.images.iter().enumerate().collect();
        let found = surface.evaluate_all(&numbered, request.need_hessian);
        for (index, result) in found.into_iter().enumerate() {
            let message = match result {
                Ok(value) => {
                    values.push(Some(value));
                    continue;
                }
                // A calculation that failed; anything else ends the run.
                Err(MoptError::Calculator(message)) if policy != FailedImage::Stop => message,
                Err(other) => return Err(other),
            };
            let mut retaken = None;
            if policy == FailedImage::Retry {
                // The step of the image again, shorter (see `path::band::images`).
                let mut attempt = 0;
                while let Some(nearer) = machine.retried_point(index, attempt) {
                    match surface.evaluate(&nearer, index, request.need_hessian) {
                        Ok(value) => {
                            machine.replace_image(index, nearer)?;
                            retaken = Some((RETRIED_SHARES[attempt], value));
                            break;
                        }
                        Err(MoptError::Calculator(_)) => attempt += 1,
                        Err(other) => return Err(other),
                    }
                }
            }
            failures.push(ImageFailure {
                image: index,
                message,
                retaken_at: retaken.as_ref().map(|(share, _)| *share),
                dropped: retaken.is_none(),
            });
            values.push(retaken.map(|(_, value)| value));
        }
        let dropped: Vec<usize> = failures
            .iter()
            .filter(|failure| failure.dropped)
            .map(|failure| failure.image)
            .collect();
        if !dropped.is_empty() {
            // Too few images left: the run ends, and says why the first of them failed.
            machine
                .drop_images(&dropped)
                .map_err(|e| match failures.first() {
                    Some(first) => MoptError::calc(format!(
                        "{e}; image {} failed with: {}",
                        first.image, first.message
                    )),
                    None => e,
                })?;
            surface.images_dropped(&dropped);
        }
        // The energies without the bias potentials, of the images that are left.
        let mut bare = vec![f64::NAN; values.len()];
        let mut any = false;
        for (index, energy) in surface.take_bare_energies() {
            if let Some(slot) = bare.get_mut(index) {
                *slot = energy;
                any = true;
            }
        }
        let bare: Vec<f64> = bare
            .into_iter()
            .enumerate()
            .filter(|(index, _)| !dropped.contains(index))
            .map(|(_, energy)| energy)
            .collect();
        let mut report = machine.tell(values.into_iter().flatten().collect())?;
        if any {
            report.bare_energies = bare;
        }
        report.failures = failures;
        report.notes = surface.take_notes();
        surface.told(report.largest_force);
        if machine.end() == Some(ChainEnd::Converged) && !surface.values_are_calculated() {
            // Relaxed on values that were not calculated: the same chain once more, all
            // of it calculated.
            machine.reopen();
            surface.calculate_next(true);
        }
        if report.added > 0 {
            surface.images_added();
        }
        outcome.iterations += 1;
        let mut stop = false;
        for observer in observers.iter_mut() {
            stop |= observer.after_iteration(&report, machine)? == Flow::Stop;
        }
        outcome.last = Some(report);
        if stop {
            return Ok(outcome);
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::path::band::machine::ChainSettings;
    use crate::path::band::trust::FixedEnds;
    use crate::path::band::ChainKind;
    use crate::pes::objective::MullerBrown2D;

    fn col(values: &[f64]) -> Col<f64> {
        Col::from_fn(values.len(), |k| values[k])
    }

    fn straight(from: [f64; 2], to: [f64; 2], images: usize) -> Vec<Col<f64>> {
        (0..images)
            .map(|k| {
                let t = k as f64 / (images - 1) as f64;
                col(&[
                    from[0] + t * (to[0] - from[0]),
                    from[1] + t * (to[1] - from[1]),
                ])
            })
            .collect()
    }

    #[test]
    fn observers_see_every_iteration_and_may_stop_or_change_the_chain() {
        let surface = MullerBrown2D::default();
        let settings = ChainSettings {
            kind: ChainKind::Variables,
            max_iterations: 30,
            ends: FixedEnds::from_code(3),
            ..ChainSettings::default()
        };
        let start = straight([-0.558, 1.442], [-0.050, 0.467], 7);
        let mut machine = ChainRelaxation::new(settings.clone(), start.clone()).unwrap();
        let mut seen = Vec::new();
        let mut record = |report: &ChainReport, _: &mut ChainRelaxation| {
            seen.push(report.iteration);
            Ok(Flow::Continue)
        };
        let outcome = relax_chain(
            &mut machine,
            &ObjectiveSurface(&surface),
            &mut [&mut record],
        )
        .unwrap();
        assert_eq!(outcome.iterations, 30);
        assert_eq!(outcome.end, Some(ChainEnd::IterationLimit));
        assert_eq!(seen, (0..30).collect::<Vec<_>>());
        // The highest image came down from the straight line.
        let last = outcome.last.unwrap();
        let top = |energies: &[f64]| energies.iter().fold(f64::NEG_INFINITY, |m, e| m.max(*e));
        let first_top = top(&start
            .iter()
            .map(|p| surface.evaluate(p, false).unwrap().value)
            .collect::<Vec<_>>());
        assert!(top(&last.energies) < first_top - 1e-3);

        // An observer that stops the run after the third iteration, and one that puts
        // more images into the chain after the first.
        let mut machine = ChainRelaxation::new(settings, start).unwrap();
        let mut stop = |report: &ChainReport, _: &mut ChainRelaxation| {
            Ok(if report.iteration == 2 {
                Flow::Stop
            } else {
                Flow::Continue
            })
        };
        let mut refine = |report: &ChainReport, machine: &mut ChainRelaxation| {
            if report.iteration == 0 {
                let chain = machine.chain().to_vec();
                let mut longer = Vec::new();
                for pair in chain.windows(2) {
                    longer.push(pair[0].clone());
                    longer.push(0.5 * (&pair[0] + &pair[1]));
                }
                longer.push(chain[chain.len() - 1].clone());
                machine.set_chain(longer)?;
            }
            Ok(Flow::Continue)
        };
        let outcome = relax_chain(
            &mut machine,
            &ObjectiveSurface(&surface),
            &mut [&mut refine, &mut stop],
        )
        .unwrap();
        assert_eq!(outcome.iterations, 3);
        assert_eq!(outcome.end, None);
        assert_eq!(outcome.last.unwrap().chain.len(), 13);
    }

    /// The Muller-Brown surface, on which the evaluation of one image fails in one
    /// iteration: where the image is farther than `reach` from the point it was last
    /// evaluated at (`None`: wherever it is).
    struct Failing {
        surface: MullerBrown2D,
        iteration: AtomicUsize,
        image: usize,
        at: usize,
        reach: Option<f64>,
        last: Mutex<Option<Col<f64>>>,
        calls: AtomicUsize,
    }

    impl ChainSurface for Failing {
        fn description(&self) -> String {
            "a surface that fails".to_string()
        }

        fn begin_iteration(&self, iteration: usize) {
            self.iteration.store(iteration, Ordering::Relaxed);
        }

        fn evaluate(
            &self,
            point: &Col<f64>,
            index: usize,
            need_hessian: bool,
        ) -> MoptResult<ImageValue> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            let now = self.iteration.load(Ordering::Relaxed);
            let mut last = self.last.lock().unwrap();
            if index == self.image && now == self.at {
                let far = match (&*last, self.reach) {
                    (Some(before), Some(reach)) => {
                        crate::math::linalg::col_norm(&(point - before)) > reach
                    }
                    _ => true,
                };
                if far {
                    return Err(MoptError::calc("no convergence of the wavefunction"));
                }
            }
            if index == self.image {
                *last = Some(point.clone());
            }
            ObjectiveSurface(&self.surface).evaluate(point, index, need_hessian)
        }
    }

    fn failing(image: usize, at: usize, reach: Option<f64>) -> Failing {
        Failing {
            surface: MullerBrown2D::default(),
            iteration: AtomicUsize::new(0),
            image,
            at,
            reach,
            last: Mutex::new(None),
            calls: AtomicUsize::new(0),
        }
    }

    #[test]
    fn an_image_whose_calculation_fails_is_put_nearer_or_taken_out() {
        let settings = ChainSettings {
            kind: ChainKind::Variables,
            max_iterations: 12,
            ends: FixedEnds::from_code(3),
            ..ChainSettings::default()
        };
        let start = straight([-0.558, 1.442], [-0.050, 0.467], 7);
        type Seen = (Vec<ImageFailure>, Vec<(usize, usize)>, Vec<Vec<Col<f64>>>);
        let run = |settings: ChainSettings, surface: &Failing| {
            let mut machine = ChainRelaxation::new(settings, start.clone()).unwrap();
            let mut seen: Seen = (Vec::new(), Vec::new(), Vec::new());
            let mut watch = |report: &ChainReport, machine: &mut ChainRelaxation| {
                seen.0.extend(report.failures.iter().cloned());
                seen.1.push((report.energies.len(), machine.chain().len()));
                seen.2.push(report.chain.clone());
                Ok(Flow::Continue)
            };
            let outcome = relax_chain(&mut machine, surface, &mut [&mut watch]);
            (outcome, seen)
        };
        // The step of image 3 from iteration 3 to iteration 4, without any failure.
        let (_, (_, _, chains)) = run(settings.clone(), &failing(99, 99, None));
        let step = crate::math::linalg::col_norm(&(&chains[4][3] - &chains[3][3]));
        assert!(step > 0.0);

        // The calculation of image 3 fails in iteration 4 unless the image is within a
        // third of its step: it goes through at a quarter of the step.
        let surface = failing(3, 4, Some(step / 3.0));
        let (outcome, (failures, lengths, _)) = run(settings.clone(), &surface);
        assert_eq!(outcome.unwrap().iterations, 12);
        assert_eq!(failures.len(), 1);
        assert_eq!(
            (
                failures[0].image,
                failures[0].retaken_at,
                failures[0].dropped
            ),
            (3, Some(0.25), false)
        );
        assert!(failures[0].message.contains("no convergence"));
        assert!(lengths.iter().all(|l| *l == (7, 7)));
        // 12 iterations of 7 images, and the two points that were tried again.
        assert_eq!(surface.calls.load(Ordering::Relaxed), 12 * 7 + 2);

        // It fails wherever the image is put, also where it was before: the image is
        // taken out, and the relaxation goes on with six.
        let (outcome, (failures, lengths, _)) = run(settings.clone(), &failing(3, 4, None));
        assert_eq!(outcome.unwrap().iterations, 12);
        assert_eq!((failures[0].retaken_at, failures[0].dropped), (None, true));
        assert_eq!(lengths[3], (7, 7));
        assert_eq!(lengths[4], (6, 6));
        assert_eq!(lengths[11], (6, 6));

        // Taken out at once (MultiOptPy), or the end of the run.
        let surface = failing(3, 4, Some(step / 3.0));
        let drop = ChainSettings {
            on_failure: FailedImage::Drop,
            ..settings.clone()
        };
        let (outcome, (failures, lengths, _)) = run(drop, &surface);
        assert!(outcome.is_ok());
        assert_eq!((failures[0].retaken_at, failures[0].dropped), (None, true));
        assert_eq!(lengths[4], (6, 6));
        assert_eq!(surface.calls.load(Ordering::Relaxed), 4 * 7 + 7 + 7 * 6);
        let stop = ChainSettings {
            on_failure: FailedImage::Stop,
            ..settings
        };
        let (outcome, (failures, _, _)) = run(stop, &failing(3, 4, None));
        assert!(matches!(outcome, Err(MoptError::Calculator(_))));
        assert!(failures.is_empty());
    }

    #[test]
    fn what_the_calculator_says_is_in_the_report_of_the_iteration() {
        use crate::domain::{EvalOutput, EvalRequest, OrbitalData, SpinCharge};
        use crate::io::xyz::parse_xyz_str;
        use crate::pes::calculator::Calculator;
        use crate::pes::native::LennardJones;

        /// A calculator with a warning, a spin, and an output of its own.
        struct Talking;

        impl Calculator for Talking {
            fn name(&self) -> &str {
                "talking"
            }

            fn evaluate(&self, sys: &MolecularSystem, req: &EvalRequest) -> MoptResult<EvalOutput> {
                let mut out = LennardJones.evaluate(sys, req)?;
                out.orbitals = Some(OrbitalData {
                    spin_square: Some(0.7512),
                    program_output: Some("cycle 1\ncycle 2\nconverged\n".to_string()),
                    ..OrbitalData::default()
                });
                Ok(out)
            }

            fn take_warnings(&self) -> Vec<String> {
                vec!["the wavefunction was started anew".to_string()]
            }
        }

        let template = parse_xyz_str(
            "3\n\nAr 0.0 0.0 0.0\nAr 3.9 0.0 0.0\nAr 1.9 3.4 0.0\n",
            SpinCharge::default(),
            None,
        )
        .unwrap();
        let first = template.coords_flat_bohr();
        let mut last = first.clone();
        last[7] += 1.0;
        let middle = Col::from_fn(9, |k| 0.5 * (first[k] + last[k]));
        let surface = CalculatorSurface::new(&Talking, template);
        let mut machine = ChainRelaxation::new(
            ChainSettings {
                max_iterations: 2,
                ..ChainSettings::default()
            },
            vec![first, middle, last],
        )
        .unwrap();
        let outcome = relax_chain(&mut machine, &surface, &mut []).unwrap();
        let notes = &outcome.last.unwrap().notes;
        // Three things for each of the three images, and none left over from the
        // iteration before.
        assert_eq!(notes.len(), 9, "{notes:?}");
        assert_eq!(
            notes[0],
            "Warning: image 0: the wavefunction was started anew"
        );
        assert_eq!(notes[1], "image 0: <S**2> = 0.751200");
        assert_eq!(
            notes[2],
            "image 0: output of the program\ncycle 1\ncycle 2\nconverged"
        );
        assert!(notes[8].starts_with("image 2: output of the program"));
        assert!(surface.take_notes().is_empty());
    }

    #[test]
    fn the_images_of_an_iteration_are_calculated_at_the_same_time() {
        use crate::domain::{EvalOutput, EvalRequest, SpinCharge};
        use crate::io::xyz::parse_xyz_str;
        use crate::pes::calculator::Calculator;
        use crate::pes::native::LennardJones;

        /// A calculator that takes a moment and counts how many calls are inside it.
        struct Slow {
            inside: AtomicUsize,
            most: AtomicUsize,
            in_order: bool,
        }

        impl Slow {
            fn new(in_order: bool) -> Self {
                Self {
                    inside: AtomicUsize::new(0),
                    most: AtomicUsize::new(0),
                    in_order,
                }
            }
        }

        impl Calculator for Slow {
            fn name(&self) -> &str {
                "slow"
            }

            fn evaluate(&self, sys: &MolecularSystem, req: &EvalRequest) -> MoptResult<EvalOutput> {
                let now = self.inside.fetch_add(1, Ordering::SeqCst) + 1;
                self.most.fetch_max(now, Ordering::SeqCst);
                std::thread::sleep(std::time::Duration::from_millis(40));
                let out = LennardJones.evaluate(sys, req);
                self.inside.fetch_sub(1, Ordering::SeqCst);
                out
            }

            fn evaluates_in_order(&self) -> bool {
                self.in_order
            }
        }

        let template = parse_xyz_str(
            "3\n\nAr 0.0 0.0 0.0\nAr 3.9 0.0 0.0\nAr 1.9 3.4 0.0\n",
            SpinCharge::default(),
            None,
        )
        .unwrap();
        let first = template.coords_flat_bohr();
        let points: Vec<Col<f64>> = (0..6)
            .map(|k| {
                let mut x = first.clone();
                x[7] += 0.2 * f64::from(k);
                x
            })
            .collect();
        let numbered: Vec<(usize, &Col<f64>)> = points.iter().enumerate().collect();

        let free = Slow::new(false);
        let together =
            CalculatorSurface::new(&free, template.clone()).evaluate_all(&numbered, false);
        let ordered = Slow::new(true);
        let one_by_one = CalculatorSurface::new(&ordered, template).evaluate_all(&numbered, false);
        assert_eq!(together.len(), 6);
        for (a, b) in together.iter().zip(&one_by_one) {
            let (a, b) = (a.as_ref().unwrap(), b.as_ref().unwrap());
            assert_eq!(a.energy, b.energy);
            assert!((0..9).all(|k| a.gradient[k] == b.gradient[k]));
        }
        // A calculator that asks for its structures in order gets one at a time; the
        // other one several at once, where the machine has the threads for it.
        assert_eq!(ordered.most.load(Ordering::SeqCst), 1);
        if rayon::current_num_threads() > 1 {
            assert!(free.most.load(Ordering::SeqCst) > 1);
        }
    }
}
