//! A calculator with static bias potentials added: [`BiasedCalculator`].
//!
//! Wherever a plain [`Calculator`] is expected (a step of a workflow, a single point, a
//! frequency calculation), the calculator and in-process terms can be handed over as one:
//! energy, gradient, Hessian, and stress are those of the calculator with the terms added, and
//! [`BiasedCalculator::evaluate_parts`] also reports the bare energy of the calculator, as
//! [`crate::pes::combine::SurfaceEvaluation`] carries the raw and the biased surface.
//!
//! Only terms whose value depends on the structure alone ([`History::Static`]) are accepted:
//! a calculator is called without the iteration or the simulated time, and in any order, so a
//! ramp, a time dependence, or accumulated state would silently be evaluated wrongly. Such
//! terms belong in a [`crate::pes::combine::PesExpr`] driven by an optimizer or a dynamics.
//!
//! MultiOptPy: no counterpart (new in mopt-rs); MultiOptPy adds its bias potentials inside the
//! optimization loop (`optimization.py`, `BiasPotentialCalculation`).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::sync::Arc;

use crate::domain::{EvalOutput, EvalRequest, Hartree, MolecularSystem, Orbitals, Quantity};
use crate::error::{MoptError, MoptResult, ResultExt};
use crate::pes::calculator::{evaluate_checked, validate_output, Calculator};
use crate::pes::stress::{term_stress, DEFAULT_STRAIN_STEP};
use crate::pes::term::{check_term_output, frame_invariant, History, PesTerm, TermContext};

/// The result of [`BiasedCalculator::evaluate_parts`].
#[derive(Debug, Clone)]
pub struct BiasedEvaluation {
    /// The calculator with the terms (what [`Calculator::evaluate`] returns).
    pub total: EvalOutput,
    /// The calculator alone.
    pub raw: EvalOutput,
    /// The energies of the terms by label, in their order.
    pub components: Vec<(String, Quantity<Hartree>)>,
}

impl BiasedEvaluation {
    /// The energy without the terms (`JobSummary::energy` of the commands).
    pub fn energy(&self) -> Quantity<Hartree> {
        self.raw.energy
    }

    /// The energy with the terms (`JobSummary::biased_energy` of the commands).
    pub fn biased_energy(&self) -> Quantity<Hartree> {
        self.total.energy
    }
}

/// A calculator with static in-process terms added to its surface.
pub struct BiasedCalculator {
    inner: Arc<dyn Calculator>,
    terms: Vec<(String, Arc<dyn PesTerm>)>,
}

impl std::fmt::Debug for BiasedCalculator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BiasedCalculator")
            .field("inner", &self.inner.name())
            .field(
                "terms",
                &self
                    .terms
                    .iter()
                    .map(|(l, _)| l.as_str())
                    .collect::<Vec<_>>(),
            )
            .finish()
    }
}

impl BiasedCalculator {
    /// `inner` with the terms `terms` (label, term) added. Refuses a term whose
    /// [`PesTerm::history`] is not [`History::Static`], naming it, and repeated labels.
    pub fn new(
        inner: Arc<dyn Calculator>,
        terms: Vec<(String, Arc<dyn PesTerm>)>,
    ) -> MoptResult<Self> {
        for (k, (label, term)) in terms.iter().enumerate() {
            let history = term.history();
            if history != History::Static {
                return Err(MoptError::config(format!(
                    "bias '{label}' ({}) cannot be added to the calculator '{}': its value \
                     depends on more than the structure ({history:?}), and a calculator is \
                     evaluated without the iteration, the time, or the order of the \
                     evaluations; add it to the surface of the optimizer instead",
                    term.name(),
                    inner.name()
                )));
            }
            if terms[..k].iter().any(|(l, _)| l == label) {
                return Err(MoptError::config(format!(
                    "bias label '{label}' is used twice on the calculator '{}'",
                    inner.name()
                )));
            }
        }
        Ok(Self { inner, terms })
    }

    /// The calculator without the terms.
    pub fn inner(&self) -> &Arc<dyn Calculator> {
        &self.inner
    }

    /// The terms (label, term).
    pub fn terms(&self) -> &[(String, Arc<dyn PesTerm>)] {
        &self.terms
    }

    /// Whether the stress of the terms is analytic (every term gives it); the calculator's
    /// own stress is whatever it returns.
    pub fn analytic_term_stress(&self) -> bool {
        self.terms.iter().all(|(_, t)| t.analytic_stress())
    }

    /// The calculator alone and with the terms at `sys`. The stress is the calculator's plus
    /// that of every term (analytic where the term gives it, central differences of its
    /// energy along the strain otherwise) when the calculator returns one, and `None`
    /// otherwise, so that a caller that differentiates the whole calculator numerically
    /// includes the terms. A strain Hessian is kept only when every term gives one.
    pub fn evaluate_parts(
        &self,
        sys: &MolecularSystem,
        req: &EvalRequest,
    ) -> MoptResult<BiasedEvaluation> {
        let raw = evaluate_checked(self.inner.as_ref(), sys, req)?;
        let ctx = TermContext::at_iteration(0, sys);
        let mut total = raw.clone();
        let mut components = Vec::with_capacity(self.terms.len());
        for (label, term) in &self.terms {
            let out = term
                .evaluate(sys, &ctx, req.need_hessian)
                .context(format!("bias '{label}'"))?;
            validate_output(&out, sys, &asked(req))
                .context(format!("bias '{label}' ({})", term.name()))?;
            check_term_output(label, &out)?;
            components.push((label.clone(), out.energy));
            total.energy = Quantity::new(total.energy.value() + out.energy.value());
            total.gradient_bohr += &out.gradient_bohr;
            total.hessian_bohr = match (total.hessian_bohr.take(), &out.hessian_bohr) {
                (Some(a), Some(b)) => Some(a + b),
                (h, _) => h,
            };
            if let Some(s) = total.stress {
                let ts = term_stress(term.as_ref(), &out, sys, &ctx, DEFAULT_STRAIN_STEP)
                    .context(format!("the stress of bias '{label}'"))?;
                total.stress = Some(std::array::from_fn(|a| {
                    std::array::from_fn(|b| s[a][b] + ts[a][b])
                }));
            }
            total.strain_hessian = match (total.strain_hessian.take(), &out.strain_hessian) {
                (Some(mut a), Some(b)) => {
                    for (ra, rb) in a.strain.iter_mut().zip(&b.strain) {
                        for (x, y) in ra.iter_mut().zip(rb) {
                            *x += y;
                        }
                    }
                    a.mixed = &a.mixed + &b.mixed;
                    Some(a)
                }
                _ => None,
            };
        }
        Ok(BiasedEvaluation {
            total,
            raw,
            components,
        })
    }
}

/// What a term must return for `req`.
fn asked(req: &EvalRequest) -> EvalRequest {
    if req.need_hessian {
        EvalRequest::ENERGY_GRAD_HESS
    } else {
        EvalRequest::ENERGY_GRAD
    }
}

impl Calculator for BiasedCalculator {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn description(&self) -> String {
        let labels: Vec<&str> = self.terms.iter().map(|(l, _)| l.as_str()).collect();
        if labels.is_empty() {
            self.inner.description()
        } else {
            format!("{} with {}", self.inner.description(), labels.join(", "))
        }
    }

    fn citations(&self) -> Vec<String> {
        self.inner.citations()
    }

    fn evaluate(&self, sys: &MolecularSystem, req: &EvalRequest) -> MoptResult<EvalOutput> {
        Ok(self.evaluate_parts(sys, req)?.total)
    }

    fn supports_pbc(&self) -> bool {
        self.inner.supports_pbc()
    }

    fn is_rigid_body_invariant(&self) -> bool {
        frame_invariant(
            self.inner.is_rigid_body_invariant(),
            self.terms.iter().map(|(_, t)| t.as_ref()),
        )
    }

    fn analytic_hessian(&self) -> bool {
        self.inner.analytic_hessian() && self.terms.iter().all(|(_, t)| t.analytic_hessian())
    }

    fn supports_excited_states(&self) -> bool {
        self.inner.supports_excited_states()
    }

    fn excitation_energies(&self, sys: &MolecularSystem) -> MoptResult<Option<Vec<f64>>> {
        self.inner.excitation_energies(sys)
    }

    fn evaluates_in_order(&self) -> bool {
        self.inner.evaluates_in_order()
    }

    fn orbitals(&self, sys: &MolecularSystem) -> MoptResult<Option<Orbitals>> {
        self.inner.orbitals(sys)
    }

    fn uses_electrons(&self) -> bool {
        self.inner.uses_electrons()
    }

    fn checks_stability(&self) -> bool {
        self.inner.checks_stability()
    }

    fn take_warnings(&self) -> Vec<String> {
        self.inner.take_warnings()
    }
}
