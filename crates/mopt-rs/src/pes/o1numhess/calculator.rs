//! A calculator whose Hessians come from O1NumHess, and the `-modelhess o1numhess` hook.
//!
//! [`O1NumHessCalculator`] passes every request on to the calculator it wraps, except that a
//! Hessian request is answered with the energy and gradient of the wrapped calculator and a
//! Hessian built from a few more of its gradients. Any calculator that returns gradients
//! therefore has Hessians, at a cost that does not grow with the size of the system.
//!
//! MultiOptPy: `ModelHessian/o1numhess.py:O1NumHessCalculator._get_gradient`,
//!  `compute_hessian`, and the `o1numhess` branches of `Calculator/*_calculation_tools.py`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::sync::Mutex;

use faer::{Col, Mat};

use super::{molecular_problem, Invariance, O1Action, O1NumHess, O1Settings};
use crate::domain::{EvalOutput, EvalRequest, MolecularSystem, Orbitals};
use crate::error::{MoptError, MoptResult};
use crate::pes::Calculator;

/// The Hessian of O1NumHess with what it took.
#[derive(Debug, Clone)]
pub struct O1Hessian {
    /// Cartesian Hessian (Hartree/Bohr^2).
    pub hessian: Mat<f64>,
    /// Gradients evaluated (besides the one at the structure itself).
    pub gradients: usize,
    /// `|G - H D|`: how well the Hessian reproduces the gradient derivatives.
    pub residual: f64,
    /// Remarks for the log.
    pub notes: Vec<String>,
}

/// The O1NumHess Hessian of `sys`, whose gradient is `gradient`, from the gradients that
/// `evaluate` returns for displaced structures.
pub fn hessian_from_gradients(
    sys: &MolecularSystem,
    gradient: &Col<f64>,
    invariance: Invariance,
    settings: O1Settings,
    mut evaluate: impl FnMut(&MolecularSystem) -> MoptResult<Col<f64>>,
) -> MoptResult<O1Hessian> {
    run_rounds(sys, gradient, invariance, settings, |points| {
        points
            .iter()
            .map(|x| evaluate(&sys.with_flat_coords_bohr(x)?))
            .collect()
    })
}

/// [`hessian_from_gradients`] with the displaced structures of each round evaluated in the
/// ambient rayon pool, for an `evaluate` that does not depend on what was evaluated before.
/// The gradients are handed back in the order of the points, so the Hessian is the same bits
/// for any number of threads (new; MultiOptPy evaluates them one after the other).
pub fn hessian_from_gradients_parallel(
    sys: &MolecularSystem,
    gradient: &Col<f64>,
    invariance: Invariance,
    settings: O1Settings,
    evaluate: impl Fn(&MolecularSystem) -> MoptResult<Col<f64>> + Sync,
) -> MoptResult<O1Hessian> {
    use rayon::prelude::*;
    run_rounds(sys, gradient, invariance, settings, |points| {
        points
            .par_iter()
            .map(|x| evaluate(&sys.with_flat_coords_bohr(x)?))
            .collect()
    })
}

/// The rounds of O1NumHess, each with the gradients of its points from `round`.
fn run_rounds(
    sys: &MolecularSystem,
    gradient: &Col<f64>,
    invariance: Invariance,
    settings: O1Settings,
    mut round: impl FnMut(&[Col<f64>]) -> MoptResult<Vec<Col<f64>>>,
) -> MoptResult<O1Hessian> {
    let mut job = O1NumHess::new(molecular_problem(sys, gradient, invariance)?, settings)?;
    while let O1Action::Evaluate(points) = job.ask() {
        let gradients = round(&points)?;
        job.tell(&gradients)?;
    }
    let hessian = job
        .hessian()
        .cloned()
        .ok_or_else(|| MoptError::calc("O1NumHess finished without a Hessian"))?;
    let mut notes = vec![format!(
        "O1NumHess: Hessian of {} coordinates from {} gradients (residual {:.1e}).",
        sys.dof(),
        job.gradient_count(),
        job.residual()
    )];
    notes.extend_from_slice(job.notes());
    Ok(O1Hessian {
        hessian,
        gradients: job.gradient_count(),
        residual: job.residual(),
        notes,
    })
}

/// A calculator that answers Hessian requests with O1NumHess (see the module documentation).
pub struct O1NumHessCalculator {
    inner: Box<dyn Calculator>,
    settings: O1Settings,
    notes: Mutex<Vec<String>>,
}

impl O1NumHessCalculator {
    /// Wrap `inner`.
    pub fn new(inner: Box<dyn Calculator>, settings: O1Settings) -> Self {
        Self {
            inner,
            settings,
            notes: Mutex::new(Vec::new()),
        }
    }

    /// The wrapped calculator.
    pub fn inner(&self) -> &dyn Calculator {
        self.inner.as_ref()
    }
}

impl Calculator for O1NumHessCalculator {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn citations(&self) -> Vec<String> {
        self.inner.citations()
    }

    fn description(&self) -> String {
        format!("{}; Hessians by O1NumHess", self.inner.description())
    }

    fn evaluate(&self, sys: &MolecularSystem, req: &EvalRequest) -> MoptResult<EvalOutput> {
        if !req.need_hessian {
            return self.inner.evaluate(sys, req);
        }
        let plain = EvalRequest {
            need_gradient: true,
            need_hessian: false,
            ..*req
        };
        let mut out = self.inner.evaluate(sys, &plain)?;
        let invariance = Invariance::of(self.inner.as_ref(), sys);
        let evaluate =
            |displaced: &MolecularSystem| Ok(self.inner.evaluate(displaced, &plain)?.gradient_bohr);
        let result = if self.inner.evaluates_in_order() {
            hessian_from_gradients(sys, &out.gradient_bohr, invariance, self.settings, evaluate)?
        } else {
            hessian_from_gradients_parallel(
                sys,
                &out.gradient_bohr,
                invariance,
                self.settings,
                evaluate,
            )?
        };
        if let Ok(mut notes) = self.notes.lock() {
            notes.extend(result.notes);
        }
        out.hessian_bohr = Some(result.hessian);
        Ok(out)
    }

    fn supports_pbc(&self) -> bool {
        self.inner.supports_pbc()
    }

    fn is_rigid_body_invariant(&self) -> bool {
        self.inner.is_rigid_body_invariant()
    }

    fn evaluates_in_order(&self) -> bool {
        self.inner.evaluates_in_order()
    }

    fn state_characters(
        &self,
        sys: &MolecularSystem,
        count: usize,
    ) -> MoptResult<Option<crate::pes::states::StateCharacters>> {
        self.inner.state_characters(sys, count)
    }

    fn transitions(
        &self,
        sys: &MolecularSystem,
        count: usize,
    ) -> MoptResult<Option<Vec<crate::pes::states::Transition>>> {
        self.inner.transitions(sys, count)
    }

    fn followed_state(&self, sys: &MolecularSystem, label: usize) -> Option<usize> {
        self.inner.followed_state(sys, label)
    }

    fn recall_state(&self, sys: &MolecularSystem, label: usize, root: usize) {
        self.inner.recall_state(sys, label, root);
    }

    fn relaxed_environment(&self, sys: &MolecularSystem) -> Option<Vec<f64>> {
        self.inner.relaxed_environment(sys)
    }

    fn recall_environment(&self, sys: &MolecularSystem, environment: &[f64]) {
        self.inner.recall_environment(sys, environment);
    }

    fn supports_excited_states(&self) -> bool {
        self.inner.supports_excited_states()
    }

    fn excitation_energies(&self, sys: &MolecularSystem) -> MoptResult<Option<Vec<f64>>> {
        self.inner.excitation_energies(sys)
    }

    fn orbitals(&self, sys: &MolecularSystem) -> MoptResult<Option<Orbitals>> {
        self.inner.orbitals(sys)
    }

    fn uses_electrons(&self) -> bool {
        self.inner.uses_electrons()
    }

    fn take_warnings(&self) -> Vec<String> {
        let mut all = self.inner.take_warnings();
        if let Ok(mut notes) = self.notes.lock() {
            all.append(&mut notes);
        }
        all
    }
}

/// Whether the `-modelhess` name asks for O1NumHess.
pub fn is_o1numhess(model_hessian: &str) -> bool {
    model_hessian.to_ascii_lowercase().contains("o1numhess")
}

/// `calculator` wrapped in [`O1NumHessCalculator`] when `-modelhess` names O1NumHess and no
/// exact Hessians are requested (`-fc`, which takes precedence as in MultiOptPy); the
/// optimizer then asks for a Hessian every `-mfc` iterations.
pub fn wrap_if_requested(
    calculator: Box<dyn Calculator>,
    model_hessian: Option<&str>,
    exact_hessians: bool,
) -> Box<dyn Calculator> {
    match model_hessian {
        Some(name) if is_o1numhess(name) && !exact_hessians => {
            Box::new(O1NumHessCalculator::new(calculator, O1Settings::default()))
        }
        _ => calculator,
    }
}
