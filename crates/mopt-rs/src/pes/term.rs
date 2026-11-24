//! In-process surface terms (extension point).
//!
//! A [`PesTerm`] is any energy contribution evaluated inside mopt-rs, without a round trip
//! through the optimizer's ask/tell interface: bias potentials, restraints, and native
//! calculators used as components of a combined surface. Terms appear as leaves of a
//! [`PesExpr`](crate::pes::combine::PesExpr), so a bias can be added to the whole surface, to one
//! state before a crossing penalty, or to one image of a two-image method alike.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::sync::Arc;

use faer::{Col, Mat};

use crate::domain::{AtomId, EvalOutput, EvalRequest, MolecularSystem, Quantity};
use crate::error::MoptResult;
use crate::pes::calculator::{evaluate_checked, Calculator};

/// Context passed to terms.
#[derive(Debug, Clone, Copy)]
pub struct TermContext<'a> {
    /// Optimizer iteration (some biases change with the iteration).
    pub iteration: usize,
    /// Initial structure of the run.
    pub initial: &'a MolecularSystem,
    /// Simulated time (atomic units) for time-dependent terms in molecular dynamics; `None` in
    /// optimizations, where such terms use the iteration instead.
    pub time_au: Option<f64>,
}

impl<'a> TermContext<'a> {
    /// Context of optimizer iteration `iteration`.
    pub fn at_iteration(iteration: usize, initial: &'a MolecularSystem) -> Self {
        Self::new(iteration, initial, None)
    }

    /// Context of iteration `iteration` at the simulated time `time_au` (`None` outside
    /// dynamics). Callers build contexts with the constructors, so that a further field
    /// changes this function only.
    pub fn new(iteration: usize, initial: &'a MolecularSystem, time_au: Option<f64>) -> Self {
        Self {
            iteration,
            initial,
            time_au,
        }
    }
}

/// Whether the value of a term depends on where the structure sits in space.
///
/// "Invariant" refers to rigid motions of the whole structure. In a periodic cell the cell
/// stays where it is, so a rigid rotation of the atoms is not a symmetry of the system; there
/// only the translations are meaningful, and both [`Frame::Invariant`] and
/// [`Frame::TranslationOnly`] say that a rigid translation of all atoms leaves the value
/// unchanged.
///
/// The variants are ordered from the least to the most dependent; the frame of a sum of terms
/// is the largest of their frames.
///
/// MultiOptPy: no counterpart (new in mopt-rs).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Frame {
    /// Unchanged by any rigid translation and rotation (functions of distances, angles,
    /// dihedrals, superposed RMSDs).
    Invariant,
    /// Unchanged by rigid translations, changed by rotations (fixed directions in space).
    TranslationOnly,
    /// Depends on the absolute position of the structure (fixed points, Cartesian
    /// coordinates, walls in space). The safe answer for terms that do not know.
    Absolute,
}

impl Frame {
    /// The frame of a sum of a term in `self` and one in `other`.
    #[must_use]
    pub fn and(self, other: Self) -> Self {
        self.max(other)
    }
}

/// Whether the value of a term at a structure depends on more than that structure.
///
/// Ordered from the least to the most history-dependent; the history of a sum of terms is the
/// largest of theirs.
///
/// MultiOptPy: no counterpart (new in mopt-rs).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum History {
    /// A function of the structure alone: the same value whenever and however often it is
    /// evaluated. A term may keep a warm start for an inner problem and still be static, as
    /// long as that start cannot change the value.
    Static,
    /// Changes with the optimizer iteration (`TermContext::iteration`), e.g. MultiOptPy's
    /// parameter ramps over 300 iterations; restarting the iteration count restarts the ramp.
    IterationRamp,
    /// Changes with the simulated time (`TermContext::time_au`; outside dynamics usually the
    /// iteration), e.g. the oscillating walls of the nanoreactor.
    TimeDependent,
    /// Keeps state that the value depends on, accumulated over evaluations or commits
    /// (metadynamics deposits, the previous solution of an inner minimization with several
    /// minima). The safe answer for terms that do not know.
    Stateful,
}

impl History {
    /// The history of a sum of a term with `self` and one with `other`.
    #[must_use]
    pub fn and(self, other: Self) -> Self {
        self.max(other)
    }
}

/// An energy contribution evaluated in-process.
pub trait PesTerm: Send + Sync {
    /// Canonical name.
    fn name(&self) -> &str;

    /// Energy, Cartesian gradient, and (when `need_hessian`) Hessian at `sys`.
    fn evaluate(
        &self,
        sys: &MolecularSystem,
        ctx: &TermContext<'_>,
        need_hessian: bool,
    ) -> MoptResult<EvalOutput>;

    /// Record `sys` as an accepted evaluation (called once per surface evaluation, after
    /// [`Self::evaluate`]); terms with history, such as metadynamics, update it here.
    fn commit(&self, _sys: &MolecularSystem, _ctx: &TermContext<'_>) {}

    /// The atoms the term depends on, when it says (`None`: possibly every atom). A term
    /// on the atoms of the core of ONIOM microiterations alone can go with them.
    fn atoms(&self) -> Option<Vec<AtomId>> {
        None
    }

    /// Whether the value depends on the absolute position or orientation of the structure.
    /// The default, [`Frame::Absolute`], is the safe answer for a term that does not say (a
    /// user callback): optimizers then keep rigid-body motions instead of removing them.
    fn frame(&self) -> Frame {
        Frame::Absolute
    }

    /// Whether the value depends on more than the structure (iteration, time, accumulated
    /// state). The default, [`History::Stateful`], is the safe answer for a term that does not
    /// say.
    fn history(&self) -> History {
        History::Stateful
    }

    /// A copy of the term with its own, unshared state, for another job or image: the state
    /// it has now is copied into new storage, so that [`Self::commit`] on one copy does not
    /// change the other. `None` (the default) means that the term has no state of its own
    /// worth copying, so the same term can be shared; [`fork_term`] then shares it unless
    /// [`Self::history`] is [`History::Stateful`].
    fn fork(&self) -> Option<Arc<dyn PesTerm>> {
        None
    }

    /// Whether the Hessian of [`Self::evaluate`] is exact (analytic, or by automatic
    /// differentiation) rather than made of differences. The default is `false`.
    fn analytic_hessian(&self) -> bool {
        false
    }

    /// Whether [`Self::evaluate`] gives the stress (`EvalOutput::stress`) in a periodic cell
    /// analytically; otherwise the stress is taken from differences of the energy along the
    /// strain (`pes::stress::term_stress`). The default is `false`.
    fn analytic_stress(&self) -> bool {
        false
    }
}

/// A copy of `term` for another job or image: its own [`PesTerm::fork`] when it gives one,
/// the term itself (shared) when it has no state that the value depends on, and an error for a
/// stateful term that cannot be copied (sharing it would mix the histories of the jobs).
///
/// MultiOptPy: no counterpart (new in mopt-rs).
pub fn fork_term(term: &Arc<dyn PesTerm>) -> MoptResult<Arc<dyn PesTerm>> {
    if let Some(copy) = term.fork() {
        return Ok(copy);
    }
    if term.history() == History::Stateful {
        return Err(crate::error::MoptError::config(format!(
            "term '{}' keeps state but cannot be copied for another job (it implements no \
             PesTerm::fork)",
            term.name()
        )));
    }
    Ok(Arc::clone(term))
}

/// Whether a surface made of a calculator (rigid-body invariant or not, as
/// `Calculator::is_rigid_body_invariant` says) and the in-process `terms` is invariant under
/// rigid translations and rotations, so that optimizers may remove those motions from their
/// steps and recentre the structure. This is the one place that decides it: every term must be
/// [`Frame::Invariant`].
///
/// MultiOptPy: no counterpart (new in mopt-rs).
pub fn frame_invariant<'a>(
    calculator_is_invariant: bool,
    terms: impl IntoIterator<Item = &'a dyn PesTerm>,
) -> bool {
    calculator_is_invariant && terms.into_iter().all(|t| t.frame() == Frame::Invariant)
}

/// Check that the output of the term `name` is finite: energy, gradient, Hessian, stress, and
/// strain Hessian where present. Every surface that evaluates terms can call this to refuse a
/// broken value with the name of the term instead of passing NaN on.
///
/// MultiOptPy: no counterpart (new in mopt-rs).
pub fn check_term_output(name: &str, out: &EvalOutput) -> MoptResult<()> {
    let bad = |what: &str| {
        Err(crate::error::MoptError::calc(format!(
            "term '{name}': the {what} is not finite"
        )))
    };
    if !out.energy.value().is_finite() {
        return bad("energy");
    }
    if out.gradient_bohr.iter().any(|v| !v.is_finite()) {
        return bad("gradient");
    }
    if let Some(h) = &out.hessian_bohr {
        if (0..h.nrows()).any(|i| (0..h.ncols()).any(|j| !h[(i, j)].is_finite())) {
            return bad("Hessian");
        }
    }
    if let Some(s) = &out.stress {
        if s.iter().flatten().any(|v| !v.is_finite()) {
            return bad("stress");
        }
    }
    if let Some(sh) = &out.strain_hessian {
        let mixed = &sh.mixed;
        if sh.strain.iter().flatten().any(|v| !v.is_finite())
            || (0..mixed.nrows()).any(|i| (0..mixed.ncols()).any(|j| !mixed[(i, j)].is_finite()))
        {
            return bad("strain Hessian");
        }
    }
    Ok(())
}

/// Any [`Calculator`] used as an in-process term.
pub struct CalculatorTerm {
    calc: Arc<dyn Calculator>,
}

impl CalculatorTerm {
    /// Wrap `calc`.
    pub fn new(calc: Arc<dyn Calculator>) -> Self {
        Self { calc }
    }
}

impl PesTerm for CalculatorTerm {
    fn name(&self) -> &str {
        self.calc.name()
    }

    fn evaluate(
        &self,
        sys: &MolecularSystem,
        _ctx: &TermContext<'_>,
        need_hessian: bool,
    ) -> MoptResult<EvalOutput> {
        let req = if need_hessian {
            EvalRequest::ENERGY_GRAD_HESS
        } else {
            EvalRequest::ENERGY_GRAD
        };
        evaluate_checked(self.calc.as_ref(), sys, &req)
    }

    fn frame(&self) -> Frame {
        if self.calc.is_rigid_body_invariant() {
            Frame::Invariant
        } else {
            Frame::Absolute
        }
    }

    /// A calculator whose results depend on the order of its evaluations (one that follows
    /// its electronic states) is stateful; others are functions of the structure.
    fn history(&self) -> History {
        if self.calc.evaluates_in_order() {
            History::Stateful
        } else {
            History::Static
        }
    }

    fn analytic_hessian(&self) -> bool {
        self.calc.analytic_hessian()
    }
}

/// A constant added to the energy, without derivatives: an energy shift method. Hatanaka and
/// Morokuma (J. Chem. Theory Comput. 2014, 10, 4184) added the 4f excitation energies of
/// lanthanide complexes to their ground state this way; the solvent activation energy shift
/// (SAES) of EDEEL adds the energy that takes the implicit solvent to a common orientation
/// (`kinetics::rate::solvent_activation::SolventActivation`).
#[derive(Debug, Clone, PartialEq)]
pub struct EnergyShift {
    name: String,
    value: f64,
}

impl EnergyShift {
    /// The shift `value` (Hartree), named `name` in reports.
    pub fn new(name: &str, value: f64) -> Self {
        Self {
            name: name.to_string(),
            value,
        }
    }

    /// The shift (Hartree).
    pub fn value(&self) -> f64 {
        self.value
    }
}

impl PesTerm for EnergyShift {
    fn name(&self) -> &str {
        &self.name
    }

    fn evaluate(
        &self,
        sys: &MolecularSystem,
        _ctx: &TermContext<'_>,
        need_hessian: bool,
    ) -> MoptResult<EvalOutput> {
        let n = sys.dof();
        let mut out = EvalOutput::energy_gradient(Quantity::new(self.value), Col::zeros(n));
        if need_hessian {
            out.hessian_bohr = Some(Mat::zeros(n, n));
        }
        Ok(out)
    }

    fn atoms(&self) -> Option<Vec<AtomId>> {
        Some(Vec::new())
    }

    fn frame(&self) -> Frame {
        Frame::Invariant
    }

    fn history(&self) -> History {
        History::Static
    }

    fn analytic_hessian(&self) -> bool {
        true
    }
}
