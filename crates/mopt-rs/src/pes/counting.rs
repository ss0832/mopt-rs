//! Counts of the evaluations that calculators and objectives made.
//!
//! Every calculator a [`crate::pes::registry::CalculatorRegistry`] creates, and every
//! objective of an [`crate::pes::objective::ObjectiveRegistry`], is wrapped so that its
//! evaluations are counted: those that returned a gradient, those of them that returned a
//! Hessian as well, and those of the energy alone. The commands write the counts of each
//! input and their total at the end of a run.
//!
//! The relaxations of an ONIOM environment (microiterations) are counted with them: how many
//! relaxations, and how many inner iterations they took; the gradients of the environment are
//! evaluations of the calculator of the lowest level and are among the gradients.
//!
//! What is counted is what the calculator was asked for. A Hessian that a calculator makes
//! from differences of its own gradients counts as one Hessian; the gradients that
//! O1NumHess asks a calculator for count as gradients. Evaluations that a restarted run
//! reads from its journal are not made again and are not counted.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use faer::Col;

use super::calculator::Calculator;
use super::objective::{Objective, ObjectiveValue};
use crate::domain::{EvalOutput, EvalRequest, MolecularSystem, Orbitals};
use crate::error::MoptResult;

/// The counters that the wrapped calculators of one registry share.
#[derive(Debug, Default)]
pub struct EvaluationCounters {
    energies: AtomicUsize,
    gradients: AtomicUsize,
    hessians: AtomicUsize,
    micro_relaxations: AtomicUsize,
    micro_iterations: AtomicUsize,
}

impl EvaluationCounters {
    /// One evaluation that succeeded.
    fn record(&self, gradient: bool, hessian: bool) {
        if hessian {
            self.hessians.fetch_add(1, Ordering::Relaxed);
        }
        if gradient || hessian {
            self.gradients.fetch_add(1, Ordering::Relaxed);
        } else {
            self.energies.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// One Hessian made by a part of a calculator outside its wrapper (the analytic Hessians
    /// of the model systems of ONIOM microiterations).
    pub fn record_hessian(&self) {
        self.record(true, true);
    }

    /// One relaxation of an environment (microiterations) with `iterations` inner iterations.
    pub fn record_micro(&self, iterations: usize) {
        self.micro_relaxations.fetch_add(1, Ordering::Relaxed);
        self.micro_iterations
            .fetch_add(iterations, Ordering::Relaxed);
    }

    /// The counts now.
    pub fn counts(&self) -> EvaluationCounts {
        EvaluationCounts {
            energies: self.energies.load(Ordering::Relaxed),
            gradients: self.gradients.load(Ordering::Relaxed),
            hessians: self.hessians.load(Ordering::Relaxed),
            micro_relaxations: self.micro_relaxations.load(Ordering::Relaxed),
            micro_iterations: self.micro_iterations.load(Ordering::Relaxed),
        }
    }
}

/// Numbers of evaluations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct EvaluationCounts {
    /// Evaluations of the energy alone.
    pub energies: usize,
    /// Evaluations with the gradient (those with the Hessian are among them).
    pub gradients: usize,
    /// Evaluations with the Hessian.
    pub hessians: usize,
    /// Relaxations of an ONIOM environment (microiterations).
    pub micro_relaxations: usize,
    /// Their inner iterations, all together.
    pub micro_iterations: usize,
}

impl EvaluationCounts {
    /// The evaluations made since `earlier` was taken.
    pub fn since(&self, earlier: &Self) -> Self {
        Self {
            energies: self.energies.saturating_sub(earlier.energies),
            gradients: self.gradients.saturating_sub(earlier.gradients),
            hessians: self.hessians.saturating_sub(earlier.hessians),
            micro_relaxations: self
                .micro_relaxations
                .saturating_sub(earlier.micro_relaxations),
            micro_iterations: self
                .micro_iterations
                .saturating_sub(earlier.micro_iterations),
        }
    }

    /// The sum of two counts.
    pub fn plus(&self, other: &Self) -> Self {
        Self {
            energies: self.energies + other.energies,
            gradients: self.gradients + other.gradients,
            hessians: self.hessians + other.hessians,
            micro_relaxations: self.micro_relaxations + other.micro_relaxations,
            micro_iterations: self.micro_iterations + other.micro_iterations,
        }
    }

    /// One line for the log; `scope` says what was counted (`this input`, `in total`).
    pub fn line(&self, scope: &str) -> String {
        let mut text = format!(
            "Evaluations ({scope}): {} gradient calculation(s), {} exact Hessian calculation(s)",
            self.gradients, self.hessians
        );
        if self.energies > 0 {
            text.push_str(&format!(
                ", {} calculation(s) of the energy alone",
                self.energies
            ));
        }
        if self.micro_relaxations > 0 {
            text.push_str(&format!(
                ", {} microiteration(s) in {} relaxation(s) of the environment",
                self.micro_iterations, self.micro_relaxations
            ));
        }
        text.push('.');
        text
    }
}

/// A calculator whose evaluations are counted.
pub struct CountingCalculator {
    inner: Box<dyn Calculator>,
    counters: Arc<EvaluationCounters>,
    /// Said once, with the warnings of the first evaluation.
    note: std::sync::Mutex<Option<String>>,
}

impl CountingCalculator {
    /// `inner`, counted in `counters`.
    pub fn new(inner: Box<dyn Calculator>, counters: Arc<EvaluationCounters>) -> Self {
        Self {
            inner,
            counters,
            note: std::sync::Mutex::new(None),
        }
    }

    /// The same with `note` said once, among the warnings after the first evaluation (an
    /// option that the calculator does not take, for example).
    pub fn with_note(self, note: String) -> Self {
        Self {
            note: std::sync::Mutex::new(Some(note)),
            ..self
        }
    }
}

impl Calculator for CountingCalculator {
    fn analytic_hessian(&self) -> bool {
        self.inner.analytic_hessian()
    }

    fn name(&self) -> &str {
        self.inner.name()
    }

    fn description(&self) -> String {
        self.inner.description()
    }

    fn citations(&self) -> Vec<String> {
        self.inner.citations()
    }

    fn evaluate(&self, sys: &MolecularSystem, req: &EvalRequest) -> MoptResult<EvalOutput> {
        let out = self.inner.evaluate(sys, req)?;
        self.counters.record(req.need_gradient, req.need_hessian);
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

    fn supports_excited_states(&self) -> bool {
        self.inner.supports_excited_states()
    }

    fn excitation_energies(&self, sys: &MolecularSystem) -> MoptResult<Option<Vec<f64>>> {
        self.inner.excitation_energies(sys)
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

    fn orbitals(&self, sys: &MolecularSystem) -> MoptResult<Option<Orbitals>> {
        self.inner.orbitals(sys)
    }

    fn supports_point_charges(&self) -> bool {
        self.inner.supports_point_charges()
    }

    fn evaluate_in_charges(
        &self,
        sys: &MolecularSystem,
        req: &EvalRequest,
        charges: &crate::pes::point_charges::PointCharges,
    ) -> MoptResult<crate::pes::point_charges::EmbeddedOutput> {
        let out = self.inner.evaluate_in_charges(sys, req, charges)?;
        self.counters.record(req.need_gradient, req.need_hessian);
        Ok(out)
    }

    fn embedding_charges(&self, sys: &MolecularSystem) -> MoptResult<Option<Vec<f64>>> {
        self.inner.embedding_charges(sys)
    }

    fn uses_electrons(&self) -> bool {
        self.inner.uses_electrons()
    }

    fn checks_stability(&self) -> bool {
        self.inner.checks_stability()
    }

    fn take_warnings(&self) -> Vec<String> {
        let mut out: Vec<String> = self
            .note
            .lock()
            .ok()
            .and_then(|mut note| note.take())
            .into_iter()
            .collect();
        out.extend(self.inner.take_warnings());
        out
    }

    fn site_terms(
        &self,
        sys: &MolecularSystem,
    ) -> Option<MoptResult<super::calculator::SiteTerms>> {
        self.inner.site_terms(sys)
    }
}

/// An objective whose evaluations are counted (every one returns its gradient).
pub struct CountingObjective {
    inner: Box<dyn Objective>,
    counters: Arc<EvaluationCounters>,
}

impl CountingObjective {
    /// `inner`, counted in `counters`.
    pub fn new(inner: Box<dyn Objective>, counters: Arc<EvaluationCounters>) -> Self {
        Self { inner, counters }
    }
}

impl Objective for CountingObjective {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn description(&self) -> String {
        self.inner.description()
    }

    fn dimension(&self) -> Option<usize> {
        self.inner.dimension()
    }

    fn variable_names(&self, n: usize) -> Vec<String> {
        self.inner.variable_names(n)
    }

    fn evaluate(&self, x: &Col<f64>, need_hessian: bool) -> MoptResult<ObjectiveValue> {
        let value = self.inner.evaluate(x, need_hessian)?;
        self.counters.record(true, need_hessian);
        Ok(value)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::domain::{Element, SpinCharge};
    use crate::pes::objective::ObjectiveRegistry;
    use crate::pes::registry::{CalculatorOptions, CalculatorRegistry};

    #[test]
    fn evaluations_of_created_calculators_are_counted_by_kind() {
        let registry = CalculatorRegistry::native();
        let argon = Element::parse_token("Ar").unwrap();
        let flat = Col::from_fn(6, |k| [0.0, 0.0, 0.0, 7.2, 0.0, 0.0][k]);
        let sys =
            MolecularSystem::from_flat_bohr(vec![argon, argon], &flat, SpinCharge::default(), None)
                .unwrap();
        let calc = registry
            .create("lj", &CalculatorOptions::default(), &sys)
            .unwrap();
        assert_eq!(registry.counts(), EvaluationCounts::default());
        calc.evaluate(&sys, &EvalRequest::ENERGY_ONLY).unwrap();
        calc.evaluate(&sys, &EvalRequest::ENERGY_GRAD).unwrap();
        calc.evaluate(&sys, &EvalRequest::ENERGY_GRAD).unwrap();
        let before = registry.counts();
        calc.evaluate(&sys, &EvalRequest::ENERGY_GRAD_HESS).unwrap();
        // A second calculator of the same registry, and a clone of the registry, count
        // into the same numbers.
        let other = registry
            .clone()
            .create("lj", &CalculatorOptions::default(), &sys)
            .unwrap();
        other.evaluate(&sys, &EvalRequest::ENERGY_GRAD).unwrap();
        let counts = registry.counts();
        assert_eq!(
            counts,
            EvaluationCounts {
                energies: 1,
                gradients: 4,
                hessians: 1,
                ..EvaluationCounts::default()
            }
        );
        assert_eq!(
            counts.since(&before),
            EvaluationCounts {
                energies: 0,
                gradients: 2,
                hessians: 1,
                ..EvaluationCounts::default()
            }
        );
        assert_eq!(calc.name(), "lj");
        assert_eq!(
            counts.line("in total"),
            "Evaluations (in total): 4 gradient calculation(s), 1 exact Hessian \
             calculation(s), 1 calculation(s) of the energy alone."
        );
    }

    #[test]
    fn a_calculator_without_an_scf_says_once_that_it_tests_nothing() {
        let registry = CalculatorRegistry::native();
        let argon = Element::parse_token("Ar").unwrap();
        let flat = Col::from_fn(6, |k| [0.0, 0.0, 0.0, 7.2, 0.0, 0.0][k]);
        let sys =
            MolecularSystem::from_flat_bohr(vec![argon, argon], &flat, SpinCharge::default(), None)
                .unwrap();
        let options = CalculatorOptions {
            stability: 2,
            ..CalculatorOptions::default()
        };
        let calc = registry.create("lj", &options, &sys).unwrap();
        assert!(!calc.checks_stability());
        calc.evaluate(&sys, &EvalRequest::ENERGY_GRAD).unwrap();
        let said = calc.take_warnings();
        assert_eq!(said.len(), 1);
        assert!(said[0].contains("-stability 2: lj has no SCF solution to test"));
        calc.evaluate(&sys, &EvalRequest::ENERGY_GRAD).unwrap();
        assert!(calc.take_warnings().is_empty());
        // Without the option nothing is said.
        let quiet = registry
            .create("lj", &CalculatorOptions::default(), &sys)
            .unwrap();
        assert!(quiet.take_warnings().is_empty());
    }

    #[test]
    fn evaluations_of_objectives_are_counted() {
        let objectives = ObjectiveRegistry::builtin();
        let mb = objectives
            .create("mb2d", &CalculatorOptions::default())
            .unwrap();
        let x = Col::from_fn(2, |k| [0.1, 0.2][k]);
        mb.evaluate(&x, false).unwrap();
        mb.evaluate(&x, true).unwrap();
        assert_eq!(
            objectives.counts(),
            EvaluationCounts {
                energies: 0,
                gradients: 2,
                hessians: 1,
                ..EvaluationCounts::default()
            }
        );
        assert_eq!(mb.dimension(), Some(2));
    }
}
