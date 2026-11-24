//! The surfaces of the searches of `ieipmain`: structures on a calculator with bias potentials
//! and constraints (one electronic state per image, or one function of several states for
//! every point: `-mf`), or an objective of plain variables.
//!
//! An evaluation gives the energy and gradient of the calculator alone and with the bias
//! potentials (iEIP uses both), the Hessian with them when it is asked for, and keeps what the
//! calculator said (warnings, `<S**2>`, the output of a program) for the log.
//!
//! MultiOptPy: the `single_point` and `BiasPotentialCalculation.main` calls of
//! `OtherMethod/*.py`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::sync::{Arc, Mutex};

use faer::{Col, Mat};
use rayon::prelude::*;

use crate::domain::{EvalRequest, MolecularSystem};
use crate::error::{MoptError, MoptResult, ResultExt};
use crate::model_hessian::ModelHessian;
use crate::path::ieip::ImageValue;
use crate::path::points::{PointRequest, PointValue};
use crate::pes::calculator::{evaluate_checked, said, Calculator};
use crate::pes::combine::PesExpr;
use crate::pes::constraint::ConstraintSet;
use crate::pes::objective::Objective;
use crate::pes::term::{check_term_output, PesTerm, TermContext};
use crate::workflow::chain::project_image;

/// The surface at a point.
#[derive(Debug, Clone)]
pub struct SurfaceValue {
    /// Energy of the calculator alone.
    pub energy: f64,
    /// Its gradient (zero when it was not asked for).
    pub gradient: Col<f64>,
    /// Energy with the bias potentials.
    pub biased_energy: f64,
    /// Gradient with them (constraints projected out).
    pub biased_gradient: Col<f64>,
    /// Hessian with them, when it was asked for.
    pub hessian: Option<Mat<f64>>,
}

impl SurfaceValue {
    /// The value as the searches of points take it (with the bias potentials).
    pub fn point_value(&self, request: &PointRequest) -> PointValue {
        PointValue {
            energy: self.biased_energy,
            gradient: request.gradient.then(|| self.biased_gradient.clone()),
            hessian: if request.hessian {
                self.hessian.clone()
            } else {
                None
            },
        }
    }

    /// The value as iEIP takes it.
    pub fn image_value(&self) -> ImageValue {
        ImageValue {
            energy: self.energy,
            gradient: self.gradient.clone(),
            biased_energy: self.biased_energy,
            biased_gradient: self.biased_gradient.clone(),
        }
    }
}

/// A surface for the searches of `ieipmain`.
pub trait SearchSurface: Sync {
    /// What it is, for the log.
    fn description(&self) -> String;

    /// The surface at `x` in the state of image `image` (0 or 1).
    fn evaluate(
        &self,
        x: &Col<f64>,
        image: usize,
        request: &PointRequest,
    ) -> MoptResult<SurfaceValue>;

    /// The structure of `x` in the state of image `image` (`None` for plain variables).
    fn structure(&self, x: &Col<f64>, image: usize) -> Option<MolecularSystem>;

    /// What was said since this was called last.
    fn take_notes(&self) -> Vec<String> {
        Vec::new()
    }

    /// Told before the points of an iteration are evaluated.
    fn begin_iteration(&self, _iteration: usize) {}

    /// Whether the two images are on two surfaces (another charge, multiplicity, or
    /// electronic state).
    fn two_surfaces(&self) -> bool {
        false
    }

    /// Whether the points of a batch must be evaluated one after the other in their order
    /// ([`Calculator::evaluates_in_order`]).
    fn in_order(&self) -> bool {
        false
    }

    /// The Hessian at `x` (image `image`) with `model` in the place of the calculator's, the
    /// bias potentials exact on top; `gradient` is the gradient of the calculator there.
    /// `None` when the surface has no structures in Cartesian coordinates (the default).
    fn model_hessian(
        &self,
        x: &Col<f64>,
        image: usize,
        gradient: &Col<f64>,
        model: &dyn ModelHessian,
    ) -> MoptResult<Option<Mat<f64>>> {
        let _ = (x, image, gradient, model);
        Ok(None)
    }
}

/// The copy of `term` that one evaluation uses: its own [`PesTerm::fork`] when it gives one
/// (a warm start that does not leak from one evaluation into another), the term itself
/// otherwise (`ieipmain` refuses the terms whose state matters and that cannot be copied,
/// `workflow::chain::check_chain_terms`).
fn own_copy(term: &Arc<dyn PesTerm>) -> Arc<dyn PesTerm> {
    term.fork().unwrap_or_else(|| Arc::clone(term))
}

/// Evaluate `points` (all in the state of `image`, or image `k` for the `k`-th point with
/// `per_point`), in parallel when there are several and the surface does not ask for
/// their order.
pub fn evaluate_all(
    surface: &dyn SearchSurface,
    points: &[PointRequest],
    image_of: &(dyn Fn(usize) -> usize + Sync),
) -> MoptResult<Vec<SurfaceValue>> {
    if points.len() > 1 && !surface.in_order() {
        points
            .par_iter()
            .enumerate()
            .map(|(k, p)| surface.evaluate(&p.x, image_of(k), p))
            .collect()
    } else {
        points
            .iter()
            .enumerate()
            .map(|(k, p)| surface.evaluate(&p.x, image_of(k), p))
            .collect()
    }
}

/// One image of structures: its atoms, charge, spin, cell, and electronic state.
#[derive(Debug, Clone)]
pub struct ImageState {
    /// The structure that carries the atoms, charge, spin, and cell.
    pub template: MolecularSystem,
    /// Electronic state (0: the ground state).
    pub excited_state: usize,
    /// Constraints (`-fix`, `-pc`, `-gfix`, `-uc`) whose directions are taken out of the
    /// gradient and the Hessian; the methods put their points back onto them.
    pub constraints: Option<ConstraintSet>,
}

/// Structures on a calculator with bias potentials.
pub struct StructureSurface<'a> {
    calculator: &'a dyn Calculator,
    images: Vec<ImageState>,
    terms: Vec<(String, Arc<dyn PesTerm>)>,
    notes: Mutex<Vec<String>>,
    iteration: std::sync::atomic::AtomicUsize,
    /// The function of electronic states that every point is evaluated on (`-mf`), instead
    /// of the state of its image.
    states: Option<Mutex<PesExpr>>,
}

impl<'a> StructureSurface<'a> {
    /// The surface of `calculator` for the images `images` (one or two).
    pub fn new(calculator: &'a dyn Calculator, images: Vec<ImageState>) -> MoptResult<Self> {
        if images.is_empty() {
            return Err(MoptError::config("a surface needs at least one image"));
        }
        Ok(Self {
            calculator,
            images,
            terms: Vec::new(),
            notes: Mutex::new(Vec::new()),
            iteration: std::sync::atomic::AtomicUsize::new(0),
            states: None,
        })
    }

    /// Every point on `surface`, a function of several electronic states (a model function
    /// of two states: `-mf`), in the charge and multiplicity of the first image; the states
    /// are those of the function, and the states of the images are not used.
    pub fn with_states(mut self, surface: PesExpr) -> Self {
        self.states = Some(Mutex::new(surface));
        self
    }

    /// The energies of the electronic states at `x` by their labels, when the surface is a
    /// function of several (empty otherwise).
    pub fn state_energies(&self, x: &Col<f64>) -> MoptResult<Vec<(String, f64)>> {
        let Some(states) = &self.states else {
            return Ok(Vec::new());
        };
        let sys = self.state(0).template.with_flat_coords_bohr(x)?;
        let surface = states
            .lock()
            .map_err(|_| MoptError::calc("the surface of the states is not available"))?;
        let tasks = surface.tasks(&sys, &EvalRequest::ENERGY_ONLY)?;
        let results =
            crate::workflow::driver::CalculatorSet::single(self.calculator).evaluate(&tasks)?;
        Ok(tasks
            .iter()
            .zip(&results)
            .map(|(task, out)| (task.label.clone(), out.energy.value()))
            .collect())
    }

    /// Bias potentials (with their labels).
    pub fn with_terms(mut self, terms: Vec<(String, Arc<dyn PesTerm>)>) -> Self {
        self.terms = terms;
        self
    }

    fn state(&self, image: usize) -> &ImageState {
        &self.images[image.min(self.images.len() - 1)]
    }
}

impl SearchSurface for StructureSurface<'_> {
    fn description(&self) -> String {
        self.calculator.description()
    }

    fn structure(&self, x: &Col<f64>, image: usize) -> Option<MolecularSystem> {
        self.state(image).template.with_flat_coords_bohr(x).ok()
    }

    fn begin_iteration(&self, iteration: usize) {
        self.iteration
            .store(iteration, std::sync::atomic::Ordering::Relaxed);
    }

    fn in_order(&self) -> bool {
        self.calculator.evaluates_in_order()
    }

    fn two_surfaces(&self) -> bool {
        if self.states.is_some() {
            // One function of the states for every point.
            return false;
        }
        match self.images.as_slice() {
            [a, b, ..] => {
                a.template.spin_charge != b.template.spin_charge
                    || a.excited_state != b.excited_state
            }
            _ => false,
        }
    }

    fn take_notes(&self) -> Vec<String> {
        self.notes
            .lock()
            .map(|mut n| std::mem::take(&mut *n))
            .unwrap_or_default()
    }

    fn model_hessian(
        &self,
        x: &Col<f64>,
        image: usize,
        gradient: &Col<f64>,
        model: &dyn ModelHessian,
    ) -> MoptResult<Option<Mat<f64>>> {
        let state = self.state(if self.states.is_some() { 0 } else { image });
        let sys = state.template.with_flat_coords_bohr(x)?;
        let mut hessian = model.hessian(&sys, gradient)?;
        let iteration = self.iteration.load(std::sync::atomic::Ordering::Relaxed);
        let ctx = TermContext::at_iteration(iteration, &state.template);
        let mut biased = gradient.clone();
        for (label, term) in &self.terms {
            let term = own_copy(term);
            let bias = term.evaluate(&sys, &ctx, true).context(label.clone())?;
            check_term_output(label, &bias)?;
            let of_bias = bias.hessian_bohr.as_ref().ok_or_else(|| {
                MoptError::calc(format!("bias potential '{label}' returned no Hessian"))
            })?;
            hessian += of_bias;
            biased += &bias.gradient_bohr;
        }
        if let Some(set) = &state.constraints {
            let (_, projected) = project_image(set, &sys, &biased, Some(hessian))?;
            return Ok(projected);
        }
        Ok(Some(hessian))
    }

    fn evaluate(
        &self,
        x: &Col<f64>,
        image: usize,
        request: &PointRequest,
    ) -> MoptResult<SurfaceValue> {
        let state = self.state(if self.states.is_some() { 0 } else { image });
        let sys = state.template.with_flat_coords_bohr(x)?;
        let base = match (request.gradient || request.hessian, request.hessian) {
            (_, true) => EvalRequest::ENERGY_GRAD_HESS,
            (true, false) => EvalRequest::ENERGY_GRAD,
            (false, false) => EvalRequest::ENERGY_ONLY,
        };
        let eval = EvalRequest {
            excited_state: state.excited_state,
            ..base
        };
        let out = match &self.states {
            None => {
                let out = evaluate_checked(self.calculator, &sys, &eval)?;
                if let Ok(mut notes) = self.notes.lock() {
                    notes.extend(said(self.calculator, &format!("image {}", image + 1), &out));
                }
                out
            }
            Some(states) => {
                // One evaluation at a time: the surface keeps what it needs between its
                // tasks and their results.
                let mut surface = states
                    .lock()
                    .map_err(|_| MoptError::calc("the surface of the states is not available"))?;
                let tasks = surface.tasks(&sys, &base)?;
                let results = crate::workflow::driver::CalculatorSet::single(self.calculator)
                    .evaluate(&tasks)?;
                if let Ok(mut notes) = self.notes.lock() {
                    for (task, out) in tasks.iter().zip(&results) {
                        notes.extend(said(self.calculator, &task.label, out));
                    }
                }
                let iteration = self.iteration.load(std::sync::atomic::Ordering::Relaxed);
                let ctx = TermContext::at_iteration(iteration, &state.template);
                // The Hessians of the terms of the states only when they are asked for.
                surface
                    .evaluate_as_requested(&sys, &base, results, &ctx)?
                    .total
            }
        };
        let n = x.nrows();
        let gradient = if eval.need_gradient {
            out.gradient_bohr.clone()
        } else {
            Col::zeros(n)
        };
        let mut biased_energy = out.energy.value();
        let mut biased_gradient = gradient.clone();
        let mut hessian = if request.hessian {
            Some(out.hessian_bohr.clone().ok_or_else(|| {
                MoptError::calc(format!(
                    "calculator '{}' returned no Hessian",
                    self.calculator.name()
                ))
            })?)
        } else {
            None
        };
        let iteration = self.iteration.load(std::sync::atomic::Ordering::Relaxed);
        let ctx = TermContext::at_iteration(iteration, &state.template);
        for (label, term) in &self.terms {
            // A copy of the term for this evaluation where it gives one: a warm start of an
            // inner problem starts from the term as it was built, so that the value does not
            // depend on which points were evaluated before or at the same time.
            let term = own_copy(term);
            let bias = term
                .evaluate(&sys, &ctx, request.hessian)
                .context(label.clone())?;
            check_term_output(label, &bias)?;
            biased_energy += bias.energy.value();
            if eval.need_gradient {
                biased_gradient += &bias.gradient_bohr;
            }
            if let Some(h) = hessian.as_mut() {
                let of_bias = bias.hessian_bohr.as_ref().ok_or_else(|| {
                    MoptError::calc(format!("bias potential '{label}' returned no Hessian"))
                })?;
                *h += of_bias;
            }
        }
        if let (Some(set), true) = (&state.constraints, eval.need_gradient) {
            // The constraint directions out of the gradient, and the Hessian projected as
            // the optimizer projects it: methods that follow modes of the Hessian (ADDF, the
            // dimer, DS-AFIR) see the constraints as stiff directions.
            (biased_gradient, hessian) = project_image(set, &sys, &biased_gradient, hessian)?;
        }
        Ok(SurfaceValue {
            energy: out.energy.value(),
            gradient,
            biased_energy,
            biased_gradient,
            hessian,
        })
    }
}

/// An objective of plain variables.
pub struct VariableSurface<'a>(pub &'a dyn Objective);

impl SearchSurface for VariableSurface<'_> {
    fn description(&self) -> String {
        self.0.description()
    }

    fn structure(&self, _x: &Col<f64>, _image: usize) -> Option<MolecularSystem> {
        None
    }

    fn evaluate(
        &self,
        x: &Col<f64>,
        _image: usize,
        request: &PointRequest,
    ) -> MoptResult<SurfaceValue> {
        let v = self.0.evaluate(x, request.hessian)?;
        Ok(SurfaceValue {
            energy: v.value,
            biased_energy: v.value,
            biased_gradient: v.gradient.clone(),
            gradient: v.gradient,
            hessian: v.hessian,
        })
    }
}
