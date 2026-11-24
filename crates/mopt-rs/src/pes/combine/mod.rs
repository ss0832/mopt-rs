//! Combined potential energy surfaces as expression trees (extension point).
//!
//! A [`PesExpr`] describes the surface an optimizer works on:
//! - [`PesExpr::Task`]: a single-point evaluation by a calculator, performed by the caller of the
//!   optimizer's ask/tell interface (Rust or Python calculators), possibly on a subsystem, with
//!   its own charge / multiplicity / electronic state;
//! - [`PesExpr::Term`]: an in-process [`PesTerm`] (bias potentials, restraints, native terms),
//!   possibly on a subsystem;
//! - [`PesExpr::Linear`]: a linear combination (e.g. EDEEL's diabatic
//!   `V11 = E_complex - E_donor,ox + E_donor,red`, or "surface + bias");
//! - [`PesExpr::Combine`]: a nonlinear [`StateCombiner`] of sub-surfaces (crossing-seam
//!   penalties, gradient projection for MECI / MESX, user-defined functions).
//!
//! Bias potentials and state combiners are therefore one mechanism: a bias is a term leaf, so it
//! can be added to the whole surface, to one state before a crossing penalty, or to a fragment.
//!
//! Leaves are identified by their labels. A label used several times denotes one evaluation
//! (EDEEL's complex enters both diabatic states but is computed once); reusing a label for a
//! different evaluation is an error. Gradients and Hessians of subsystem leaves are scattered
//! onto the full structure. A subsystem task can cap the bonds cut at its border with hydrogen
//! atoms ([`Caps`], the link atoms of `pes::oniom::layout`): the fragments of a molecule of
//! donor, bridge, and acceptor in EDEEL; their derivatives go back through the positions of
//! the caps.
//!
//! The optimizer needs the surface split into the part that comes from calculators (whose
//! Hessian is modelled and quasi-Newton-updated) and the in-process part (whose Hessian is
//! exact). [`PesExpr::evaluate`] therefore returns the full surface, the surface with every term
//! removed ("raw"), and the Hessian of the difference computed with the calculator Hessians set
//! to zero. For "surface + bias" this is exactly MultiOptPy's split into raw and bias parts;
//! below a nonlinear combiner it is the part of the difference that is known without
//! calculator Hessians.
//!
//! MultiOptPy: `optimization.py` (`ModelFunctionHandler`, `EDEELHandler`) and `ModelFunction/`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

pub mod functions;
pub mod presets;
pub mod projection;

use std::collections::HashMap;
use std::sync::Arc;

use faer::{Col, Mat};

pub use functions::{
    AvoidingModel, ConicalPenalty, EnergyFunction, FunctionCombiner, LegacyAvoidingModel,
    QuadraticPenalty, WeightedSum, AVOIDING_ALPHA,
};
pub use presets::{
    edeel, edeel_states, edeel_with, model_function, two_states, two_states_of, EdeelFragment,
    EdeelLayout, EdeelStates, EdeelTarget,
};
pub use projection::{GradientProjection, ProjectionKind};

use crate::domain::{
    AtomId, EvalOutput, EvalRequest, Hartree, MolecularSystem, Quantity, SpinCharge,
};
use crate::error::{MoptError, MoptResult, ResultExt};
use crate::pes::calculator::validate_output;
use crate::pes::oniom::layout::{crossing_bonds, Link, LinkPlacement, Region};
use crate::pes::oniom::BOND_SCALE;
use crate::pes::term::{PesTerm, TermContext};

/// Label of the calculator used when only one calculator is involved.
pub const DEFAULT_CALCULATOR: &str = "default";

/// Hydrogen caps of a subsystem: every bond from an atom of the subsystem to an atom outside
/// it is replaced by a hydrogen on the line of the bond, placed as the link atoms of ONIOM
/// (`pes::oniom::layout`).
#[derive(Debug, Clone, PartialEq)]
pub struct Caps {
    /// The cut bonds, the atom of the subsystem first.
    pub links: Vec<Link>,
    /// Where the hydrogens sit.
    pub placement: LinkPlacement,
}

impl Caps {
    /// The caps of the bonds of `sys` from `atoms` to the other atoms (closer than
    /// `pes::oniom::BOND_SCALE` times the sum of the covalent radii), placed by `placement`.
    pub fn of_bonds(sys: &MolecularSystem, atoms: &[AtomId], placement: LinkPlacement) -> Self {
        let links = crossing_bonds(sys, atoms, BOND_SCALE)
            .into_iter()
            .map(|(inner, outer)| Link::new(sys, inner, outer, placement))
            .collect();
        Self { links, placement }
    }
}

/// A single-point evaluation performed through ask/tell.
#[derive(Debug, Clone, PartialEq)]
pub struct TaskSpec {
    /// Name used in logs and output directories; identifies the evaluation.
    pub label: String,
    /// Label of the calculator that evaluates the task.
    pub calculator: String,
    /// Atoms of the subsystem (all atoms when `None`).
    pub atoms: Option<Vec<AtomId>>,
    /// Charge and multiplicity (those of the structure when `None`).
    pub spin_charge: Option<SpinCharge>,
    /// Electronic state root (`0` = ground state).
    pub excited_state: usize,
    /// Hydrogen caps of the cut bonds of the subsystem (`None`: the atoms alone).
    pub caps: Option<Caps>,
}

impl TaskSpec {
    /// Whole structure, default calculator, ground state.
    pub fn whole(label: &str) -> Self {
        Self {
            label: label.to_string(),
            calculator: DEFAULT_CALCULATOR.to_string(),
            atoms: None,
            spin_charge: None,
            excited_state: 0,
            caps: None,
        }
    }

    /// Cap the bonds cut at the border of the subsystem ([`Self::atoms`]) with hydrogens.
    #[must_use]
    pub fn caps(mut self, caps: Caps) -> Self {
        self.caps = Some(caps);
        self
    }

    /// The subsystem with its caps as an ONIOM region, in the state `sc`.
    fn region(&self, sc: SpinCharge) -> MoptResult<Option<Region>> {
        let Some(caps) = &self.caps else {
            return Ok(None);
        };
        let atoms = self.atoms.clone().ok_or_else(|| {
            MoptError::config(format!(
                "task '{}': caps need the atoms of a subsystem",
                self.label
            ))
        })?;
        Ok(Some(Region {
            atoms,
            links: caps.links.clone(),
            spin_charge: sc,
            placement: caps.placement,
            follows: None,
        }))
    }

    /// The structure this task evaluates at `sys`.
    fn system(&self, sys: &MolecularSystem) -> MoptResult<MolecularSystem> {
        let sc = self.spin_charge.unwrap_or(sys.spin_charge);
        match self.region(sc)? {
            Some(region) => region.model(sys),
            None => leaf_system(sys, self.atoms.as_ref(), sc),
        }
    }

    /// Its result `out` on the whole structure `sys`.
    fn scatter(&self, sys: &MolecularSystem, out: EvalOutput) -> MoptResult<Value> {
        let Some(region) = self.region(sys.spin_charge)? else {
            return Ok(scatter(sys, self.atoms.as_ref(), out));
        };
        let gradient = region.gradient_to_real(sys, &out.gradient_bohr)?;
        let hessian = match &out.hessian_bohr {
            Some(h) => Some(region.hessian_to_real(sys, &out.gradient_bohr, h)?),
            None => None,
        };
        Ok(Value {
            energy: out.energy.value(),
            gradient,
            hessian,
        })
    }

    /// Use calculator `label`.
    #[must_use]
    pub fn on(mut self, calculator: &str) -> Self {
        self.calculator = calculator.to_string();
        self
    }

    /// Restrict to the listed atoms.
    #[must_use]
    pub fn atoms(mut self, atoms: Vec<AtomId>) -> Self {
        self.atoms = Some(atoms);
        self
    }

    /// Use the given charge and multiplicity.
    #[must_use]
    pub fn spin_charge(mut self, sc: SpinCharge) -> Self {
        self.spin_charge = Some(sc);
        self
    }

    /// Evaluate electronic state `root`.
    #[must_use]
    pub fn excited_state(mut self, root: usize) -> Self {
        self.excited_state = root;
        self
    }
}

/// A concrete evaluation for the caller of ask/tell.
#[derive(Debug, Clone)]
pub struct EvalTask {
    /// Task label.
    pub label: String,
    /// Calculator label.
    pub calculator: String,
    /// Structure (whole or subsystem).
    pub system: MolecularSystem,
    /// Requested derivatives (including the electronic state root).
    pub request: EvalRequest,
}

/// Energies, gradients (full structure, Hartree/Bohr), and optionally Hessians of the inputs of a
/// combiner.
#[derive(Debug, Clone)]
pub struct StateValues {
    /// Energies (Hartree).
    pub energies: Vec<f64>,
    /// Gradients.
    pub gradients: Vec<Col<f64>>,
    /// Hessians, when every input has one.
    pub hessians: Option<Vec<Mat<f64>>>,
}

/// Effective surface produced by a combiner (Hartree, Bohr).
#[derive(Debug, Clone)]
pub struct Combined {
    /// Energy.
    pub energy: f64,
    /// Gradient (or pseudo-gradient for projection methods).
    pub gradient: Col<f64>,
    /// Hessian, if it can be formed.
    pub hessian: Option<Mat<f64>>,
}

/// Nonlinear combination of sub-surfaces.
pub trait StateCombiner: Send {
    /// Canonical name.
    fn name(&self) -> &str;

    /// Number of inputs.
    fn n_states(&self) -> usize;

    /// Combine the inputs at `sys`. Must not change internal state: it is called several times
    /// per optimizer iteration.
    fn combine(&self, sys: &MolecularSystem, states: &StateValues) -> MoptResult<Combined>;

    /// Record the accepted inputs of an iteration (for combiners with history).
    fn commit(&mut self, sys: &MolecularSystem, states: &StateValues) {
        let _ = (sys, states);
    }

    /// For a combiner that searches a crossing of its two inputs: unit vectors spanning the
    /// branching plane at these inputs, the direction of the difference gradient first.
    /// `None` for other combiners.
    fn branching_plane(&self, states: &StateValues) -> Option<Vec<Col<f64>>> {
        let _ = states;
        None
    }

    /// Replace the branching plane a crossing search has built up from its steps by
    /// `directions` (as [`Self::branching_plane`] gives them). Other combiners ignore it.
    fn set_branching_plane(&mut self, directions: &[Col<f64>]) {
        let _ = directions;
    }
}

/// The two states of a crossing search with its branching plane.
#[derive(Debug, Clone)]
pub struct CrossingStates {
    /// Names of the two states.
    pub labels: Vec<String>,
    /// Energies, gradients, and Hessians of the two states on the whole structure; terms
    /// added to the whole surface (bias potentials) are included in both.
    pub states: StateValues,
    /// Unit vectors spanning the branching plane as the combiner sees it: along the
    /// difference gradient, then (for a conical intersection) the coupling direction.
    pub plane: Vec<Col<f64>>,
}

/// Expression for a (combined) potential energy surface.
pub enum PesExpr {
    /// Calculator evaluation through ask/tell.
    Task(TaskSpec),
    /// In-process term.
    Term {
        /// Name used in reports; identifies the evaluation.
        label: String,
        /// The term.
        term: Arc<dyn PesTerm>,
        /// Atoms of the subsystem (all atoms when `None`).
        atoms: Option<Vec<AtomId>>,
    },
    /// `sum_i c_i S_i`.
    Linear(Vec<(f64, PesExpr)>),
    /// `F(S_1, ..., S_n)`.
    Combine {
        /// Combiner.
        combiner: Box<dyn StateCombiner>,
        /// Names of the inputs used in reports.
        labels: Vec<String>,
        /// Inputs.
        inputs: Vec<PesExpr>,
    },
}

impl std::fmt::Debug for PesExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Task(t) => f.debug_tuple("Task").field(t).finish(),
            Self::Term { label, atoms, .. } => f
                .debug_struct("Term")
                .field("label", label)
                .field("atoms", atoms)
                .finish(),
            Self::Linear(v) => f.debug_tuple("Linear").field(v).finish(),
            Self::Combine {
                combiner,
                labels,
                inputs,
            } => f
                .debug_struct("Combine")
                .field("combiner", &combiner.name())
                .field("labels", labels)
                .field("inputs", inputs)
                .finish(),
        }
    }
}

impl Default for PesExpr {
    fn default() -> Self {
        Self::single()
    }
}

/// Result of evaluating a surface expression.
#[derive(Debug, Clone)]
pub struct SurfaceEvaluation {
    /// Full surface (calculators and in-process terms).
    pub total: EvalOutput,
    /// Surface with every in-process term removed.
    pub raw: EvalOutput,
    /// Hessian of `total - raw` with calculator Hessians set to zero (`None` without terms).
    pub term_hessian: Option<Mat<f64>>,
    /// Energies of the leaves and of the combiner inputs, in tree order.
    pub components: Vec<(String, Quantity<Hartree>)>,
}

#[derive(Debug, Clone)]
struct Value {
    energy: f64,
    gradient: Col<f64>,
    hessian: Option<Mat<f64>>,
}

impl Value {
    fn zero(n: usize, with_hessian: bool) -> Self {
        Self {
            energy: 0.0,
            gradient: Col::zeros(n),
            hessian: with_hessian.then(|| Mat::zeros(n, n)),
        }
    }

    fn scaled(&self, c: f64) -> Self {
        Self {
            energy: c * self.energy,
            gradient: &self.gradient * c,
            hessian: self.hessian.as_ref().map(|h| h * c),
        }
    }

    fn add_scaled(&mut self, c: f64, other: &Self) {
        self.energy += c * other.energy;
        self.gradient += &other.gradient * c;
        self.hessian = match (self.hessian.take(), other.hessian.as_ref()) {
            (Some(a), Some(b)) => Some(a + b * c),
            _ => None,
        };
    }

    fn into_output(self) -> EvalOutput {
        EvalOutput {
            energy: Quantity::new(self.energy),
            gradient_bohr: self.gradient,
            hessian_bohr: self.hessian,
            orbitals: None,
            stress: None,
            strain_hessian: None,
        }
    }
}

/// How leaves are treated in one pass over the tree.
#[derive(Debug, Clone, Copy)]
struct Pass {
    include_terms: bool,
    zero_task_hessians: bool,
}

/// Unique term leaf: label, term, and subsystem atoms.
type TermLeaf<'a> = (&'a str, &'a Arc<dyn PesTerm>, Option<&'a Vec<AtomId>>);

/// Unique leaves of an expression, in first-appearance order.
struct Leaves<'a> {
    tasks: Vec<&'a TaskSpec>,
    terms: Vec<TermLeaf<'a>>,
}

/// Leaf values on the full structure, keyed by label.
struct Cache {
    tasks: HashMap<String, Value>,
    terms: HashMap<String, Value>,
}

struct Walk<'a> {
    sys: &'a MolecularSystem,
    cache: &'a Cache,
    need_hessian: bool,
    pass: Pass,
    record: bool,
    components: Vec<(String, Quantity<Hartree>)>,
    states: Vec<StateValues>,
}

fn scatter(sys: &MolecularSystem, atoms: Option<&Vec<AtomId>>, out: EvalOutput) -> Value {
    let Some(atoms) = atoms else {
        return Value {
            energy: out.energy.value(),
            gradient: out.gradient_bohr,
            hessian: out.hessian_bohr,
        };
    };
    let n = sys.dof();
    let mut g = Col::zeros(n);
    for (slot, a) in atoms.iter().enumerate() {
        for c in 0..3 {
            g[a.dof(c)] += out.gradient_bohr[3 * slot + c];
        }
    }
    let hessian = out.hessian_bohr.as_ref().map(|h| {
        let mut f = Mat::zeros(n, n);
        for (sa, a) in atoms.iter().enumerate() {
            for (sb, b) in atoms.iter().enumerate() {
                for p in 0..3 {
                    for q in 0..3 {
                        f[(a.dof(p), b.dof(q))] += h[(3 * sa + p, 3 * sb + q)];
                    }
                }
            }
        }
        f
    });
    Value {
        energy: out.energy.value(),
        gradient: g,
        hessian,
    }
}

fn leaf_system(
    sys: &MolecularSystem,
    atoms: Option<&Vec<AtomId>>,
    sc: SpinCharge,
) -> MoptResult<MolecularSystem> {
    match atoms {
        Some(a) => sys.subsystem(a, sc),
        None => {
            let mut s = sys.clone();
            s.spin_charge = sc;
            Ok(s)
        }
    }
}

impl PesExpr {
    /// The ordinary case: the default calculator on the whole structure.
    pub fn single() -> Self {
        Self::Task(TaskSpec::whole("pes"))
    }

    /// A term on the whole structure.
    pub fn term(label: &str, term: Arc<dyn PesTerm>) -> Self {
        Self::Term {
            label: label.to_string(),
            term,
            atoms: None,
        }
    }

    /// `self + term` (e.g. a bias potential on the whole structure).
    #[must_use]
    pub fn plus_term(self, label: &str, term: Arc<dyn PesTerm>) -> Self {
        let leaf = Self::term(label, term);
        match self {
            Self::Linear(mut v) => {
                v.push((1.0, leaf));
                Self::Linear(v)
            }
            other => Self::Linear(vec![(1.0, other), (1.0, leaf)]),
        }
    }

    /// `F(inputs)` with checked arity and labels.
    pub fn combine(
        combiner: Box<dyn StateCombiner>,
        labels: Vec<String>,
        inputs: Vec<Self>,
    ) -> MoptResult<Self> {
        let expr = Self::Combine {
            combiner,
            labels,
            inputs,
        };
        expr.validate()?;
        Ok(expr)
    }

    /// Whether the expression is the default calculator on the whole structure, without terms.
    pub fn is_single(&self) -> bool {
        matches!(self, Self::Task(t) if t.atoms.is_none()
            && t.caps.is_none()
            && t.spin_charge.is_none()
            && t.excited_state == 0
            && t.calculator == DEFAULT_CALCULATOR)
    }

    /// Whether the expression contains in-process terms.
    pub fn has_terms(&self) -> bool {
        match self {
            Self::Task(_) => false,
            Self::Term { .. } => true,
            Self::Linear(v) => v.iter().any(|(_, e)| e.has_terms()),
            Self::Combine { inputs, .. } => inputs.iter().any(Self::has_terms),
        }
    }

    /// Check combiner arities and the consistency of leaf labels.
    pub fn validate(&self) -> MoptResult<()> {
        self.validate_shape()?;
        self.leaves().map(|_| ())
    }

    fn validate_shape(&self) -> MoptResult<()> {
        match self {
            Self::Task(_) | Self::Term { .. } => Ok(()),
            Self::Linear(v) => {
                if v.is_empty() {
                    return Err(MoptError::config("empty linear combination of surfaces"));
                }
                v.iter().try_for_each(|(_, e)| e.validate_shape())
            }
            Self::Combine {
                combiner,
                labels,
                inputs,
            } => {
                if combiner.n_states() != inputs.len() || labels.len() != inputs.len() {
                    return Err(MoptError::config(format!(
                        "combiner '{}' expects {} inputs, got {} inputs and {} labels",
                        combiner.name(),
                        combiner.n_states(),
                        inputs.len(),
                        labels.len()
                    )));
                }
                inputs.iter().try_for_each(Self::validate_shape)
            }
        }
    }

    fn leaves(&self) -> MoptResult<Leaves<'_>> {
        let mut leaves = Leaves {
            tasks: Vec::new(),
            terms: Vec::new(),
        };
        self.collect_leaves(&mut leaves)?;
        Ok(leaves)
    }

    fn collect_leaves<'a>(&'a self, out: &mut Leaves<'a>) -> MoptResult<()> {
        match self {
            Self::Task(t) => {
                if out.terms.iter().any(|(l, _, _)| *l == t.label) {
                    return Err(MoptError::config(format!(
                        "label '{}' is used for a task and a term",
                        t.label
                    )));
                }
                match out.tasks.iter().find(|s| s.label == t.label) {
                    Some(s) if *s != t => Err(MoptError::config(format!(
                        "task label '{}' is used for different evaluations",
                        t.label
                    ))),
                    Some(_) => Ok(()),
                    None => {
                        out.tasks.push(t);
                        Ok(())
                    }
                }
            }
            Self::Term { label, term, atoms } => {
                if out.tasks.iter().any(|s| s.label == *label) {
                    return Err(MoptError::config(format!(
                        "label '{label}' is used for a task and a term"
                    )));
                }
                match out.terms.iter().find(|(l, _, _)| l == label) {
                    Some((_, t, a)) if !Arc::ptr_eq(t, term) || *a != atoms.as_ref() => {
                        Err(MoptError::config(format!(
                            "term label '{label}' is used for different terms"
                        )))
                    }
                    Some(_) => Ok(()),
                    None => {
                        out.terms.push((label, term, atoms.as_ref()));
                        Ok(())
                    }
                }
            }
            Self::Linear(v) => v.iter().try_for_each(|(_, e)| e.collect_leaves(out)),
            Self::Combine { inputs, .. } => inputs.iter().try_for_each(|e| e.collect_leaves(out)),
        }
    }

    /// Calculator evaluations needed at `sys` (one per task label, in first-appearance order).
    pub fn tasks(&self, sys: &MolecularSystem, request: &EvalRequest) -> MoptResult<Vec<EvalTask>> {
        self.leaves()?
            .tasks
            .into_iter()
            .map(|t| {
                let system = t.system(sys).context(format!("task '{}'", t.label))?;
                Ok(EvalTask {
                    label: t.label.clone(),
                    calculator: t.calculator.clone(),
                    system,
                    request: EvalRequest {
                        excited_state: t.excited_state,
                        ..*request
                    },
                })
            })
            .collect()
    }

    /// The in-process terms of the expression (one per label, in first-appearance order).
    pub fn terms(&self) -> MoptResult<Vec<Arc<dyn PesTerm>>> {
        Ok(self
            .leaves()?
            .terms
            .into_iter()
            .map(|(_, t, _)| Arc::clone(t))
            .collect())
    }

    /// Whether the surface is invariant under rigid motions, for calculators that are
    /// (`calculator_is_invariant`) or not: see [`crate::pes::term::frame_invariant`].
    pub fn frame_invariant(&self, calculator_is_invariant: bool) -> MoptResult<bool> {
        let terms = self.terms()?;
        Ok(crate::pes::term::frame_invariant(
            calculator_is_invariant,
            terms.iter().map(|t| t.as_ref()),
        ))
    }

    /// Evaluate the surface at `sys` from the results of [`Self::tasks`] (same order).
    ///
    /// Combiners and terms with history record this evaluation as accepted. The terms are
    /// always evaluated with their Hessians, so that [`SurfaceEvaluation::term_hessian`] is
    /// there even for a request without a Hessian (an optimizer adds the exact Hessian of its
    /// bias potentials to its quasi-Newton Hessian at every iteration); callers that do not
    /// use it can save that work with [`Self::evaluate_as_requested`].
    pub fn evaluate(
        &mut self,
        sys: &MolecularSystem,
        request: &EvalRequest,
        results: Vec<EvalOutput>,
        term_ctx: &TermContext<'_>,
    ) -> MoptResult<SurfaceEvaluation> {
        self.evaluate_with(sys, request, results, term_ctx, true)
    }

    /// [`Self::evaluate`] with the Hessians of the terms only when `request` asks for a
    /// Hessian: for a request without one, [`SurfaceEvaluation::term_hessian`] is `None`
    /// and everything else is the same.
    pub fn evaluate_as_requested(
        &mut self,
        sys: &MolecularSystem,
        request: &EvalRequest,
        results: Vec<EvalOutput>,
        term_ctx: &TermContext<'_>,
    ) -> MoptResult<SurfaceEvaluation> {
        self.evaluate_with(sys, request, results, term_ctx, request.need_hessian)
    }

    fn evaluate_with(
        &mut self,
        sys: &MolecularSystem,
        request: &EvalRequest,
        results: Vec<EvalOutput>,
        term_ctx: &TermContext<'_>,
        term_hessians: bool,
    ) -> MoptResult<SurfaceEvaluation> {
        let cache = self.build_cache(sys, request, results, term_ctx, term_hessians)?;
        let mut components: Vec<(String, Quantity<Hartree>)> = Vec::new();
        {
            let leaves = self.leaves()?;
            for t in &leaves.tasks {
                if let Some(v) = cache.tasks.get(&t.label) {
                    components.push((t.label.clone(), Quantity::new(v.energy)));
                }
            }
            for (label, _, _) in &leaves.terms {
                if let Some(v) = cache.terms.get(*label) {
                    components.push(((*label).to_string(), Quantity::new(v.energy)));
                }
            }
        }

        let mut walk = Walk {
            sys,
            cache: &cache,
            need_hessian: request.need_hessian,
            pass: Pass {
                include_terms: true,
                zero_task_hessians: false,
            },
            record: true,
            components: Vec::new(),
            states: Vec::new(),
        };
        let total = self.eval(&mut walk)?;
        let states = std::mem::take(&mut walk.states);
        components.append(&mut walk.components);
        walk.record = false;

        let (raw, term_hessian) = if self.has_terms() {
            let mut run = |include_terms, zero_task_hessians| {
                walk.pass = Pass {
                    include_terms,
                    zero_task_hessians,
                };
                self.eval(&mut walk)
            };
            let raw = run(false, false)?;
            let with = run(true, true)?;
            let without = run(false, true)?;
            let th = match (with.hessian, without.hessian) {
                (Some(a), Some(b)) => Some(a - b),
                _ => None,
            };
            (raw, th)
        } else {
            (total.clone(), None)
        };

        let mut states = states.into_iter();
        self.commit(sys, &mut states);
        for (_, term, atoms) in self.leaves()?.terms {
            let sub = leaf_system(sys, atoms, sys.spin_charge)?;
            term.commit(&sub, term_ctx);
        }
        Ok(SurfaceEvaluation {
            total: total.into_output(),
            raw: raw.into_output(),
            term_hessian,
            components,
        })
    }

    /// The leaf values at `sys`; the terms with their Hessians when `term_hessians`.
    fn build_cache(
        &self,
        sys: &MolecularSystem,
        request: &EvalRequest,
        results: Vec<EvalOutput>,
        term_ctx: &TermContext<'_>,
        term_hessians: bool,
    ) -> MoptResult<Cache> {
        let tasks = self.tasks(sys, request)?;
        if results.len() != tasks.len() {
            return Err(MoptError::calc(format!(
                "expected {} task results, got {}",
                tasks.len(),
                results.len()
            )));
        }
        let leaves = self.leaves()?;
        let mut cache = Cache {
            tasks: HashMap::new(),
            terms: HashMap::new(),
        };
        for ((task, spec), out) in tasks.iter().zip(&leaves.tasks).zip(results) {
            validate_output(&out, &task.system, &task.request)
                .context(format!("task '{}'", task.label))?;
            let value = spec
                .scatter(sys, out)
                .context(format!("task '{}'", task.label))?;
            cache.tasks.insert(task.label.clone(), value);
        }
        for (label, term, atoms) in leaves.terms {
            let sub = leaf_system(sys, atoms, sys.spin_charge)?;
            let out = term
                .evaluate(&sub, term_ctx, term_hessians)
                .context(format!("term '{label}'"))?;
            let asked = if term_hessians {
                EvalRequest::ENERGY_GRAD_HESS
            } else {
                EvalRequest::ENERGY_GRAD
            };
            validate_output(&out, &sub, &asked)
                .context(format!("term '{label}' ({})", term.name()))?;
            cache
                .terms
                .insert(label.to_string(), scatter(sys, atoms, out));
        }
        Ok(cache)
    }

    fn eval(&self, w: &mut Walk<'_>) -> MoptResult<Value> {
        let n = w.sys.dof();
        match self {
            Self::Task(t) => {
                let mut v =
                    w.cache.tasks.get(&t.label).cloned().ok_or_else(|| {
                        MoptError::calc(format!("no result for task '{}'", t.label))
                    })?;
                if w.pass.zero_task_hessians {
                    v.hessian = Some(Mat::zeros(n, n));
                }
                Ok(v)
            }
            Self::Term { label, .. } => {
                if !w.pass.include_terms {
                    return Ok(Value::zero(n, w.need_hessian || w.pass.zero_task_hessians));
                }
                w.cache
                    .terms
                    .get(label)
                    .cloned()
                    .ok_or_else(|| MoptError::calc(format!("no value for term '{label}'")))
            }
            Self::Linear(items) => {
                let mut acc: Option<Value> = None;
                for (c, e) in items {
                    let v = e.eval(w)?;
                    match acc.as_mut() {
                        None => acc = Some(v.scaled(*c)),
                        Some(a) => a.add_scaled(*c, &v),
                    }
                }
                acc.ok_or_else(|| MoptError::config("empty linear combination of surfaces"))
            }
            Self::Combine {
                combiner,
                labels,
                inputs,
            } => {
                let vals = inputs
                    .iter()
                    .map(|e| e.eval(w))
                    .collect::<MoptResult<Vec<_>>>()?;
                // A combination is not linear in the energies of its inputs. In a pass without
                // the terms its derivatives are still taken at the energies with them, where
                // the structure is on the combination: a constant term in one input (the shift
                // of one state) would otherwise move the point of the combination, and the
                // gradient without terms would be that of another function (a crossing search
                // away from the crossing). Only the energy is the one without the terms.
                let with_terms = if !w.pass.include_terms && inputs.iter().any(Self::has_terms) {
                    let (pass, record) = (w.pass, w.record);
                    w.pass.include_terms = true;
                    w.record = false;
                    let energies = inputs
                        .iter()
                        .map(|e| e.eval(w).map(|v| v.energy))
                        .collect::<MoptResult<Vec<f64>>>();
                    w.pass = pass;
                    w.record = record;
                    Some(energies?)
                } else {
                    None
                };
                let hessians = vals.iter().map(|v| v.hessian.clone()).collect();
                let without_terms: Vec<f64> = vals.iter().map(|v| v.energy).collect();
                let mut states = StateValues {
                    energies: with_terms.unwrap_or_else(|| without_terms.clone()),
                    gradients: vals.into_iter().map(|v| v.gradient).collect(),
                    hessians,
                };
                let mut c = combiner
                    .combine(w.sys, &states)
                    .context(format!("combiner '{}'", combiner.name()))?;
                if states.energies != without_terms {
                    states.energies = without_terms;
                    states.hessians = None;
                    c.energy = combiner
                        .combine(w.sys, &states)
                        .context(format!("combiner '{}'", combiner.name()))?
                        .energy;
                }
                if c.gradient.nrows() != n
                    || c.hessian
                        .as_ref()
                        .is_some_and(|h| h.nrows() != n || h.ncols() != n)
                {
                    return Err(MoptError::calc(format!(
                        "combiner '{}' returned arrays of the wrong size",
                        combiner.name()
                    )));
                }
                if w.record {
                    for (l, e) in labels.iter().zip(&states.energies) {
                        w.components.push((l.clone(), Quantity::new(*e)));
                    }
                    w.states.push(states);
                }
                Ok(Value {
                    energy: c.energy,
                    gradient: c.gradient,
                    hessian: c.hessian,
                })
            }
        }
    }

    /// If the surface is a crossing search of two states (gradient projection), alone or
    /// with terms added to it: the two states with their Hessians and the branching plane at
    /// `sys`, from the results of [`Self::tasks`] with Hessians. `None` for other surfaces.
    pub fn crossing(
        &self,
        sys: &MolecularSystem,
        results: Vec<EvalOutput>,
        term_ctx: &TermContext<'_>,
    ) -> MoptResult<Option<CrossingStates>> {
        let found = self.crossing_states(sys, &EvalRequest::ENERGY_GRAD_HESS, results, term_ctx)?;
        if found.as_ref().is_some_and(|c| c.states.hessians.is_none()) {
            return Err(MoptError::calc(
                "the states of the crossing came without Hessians",
            ));
        }
        Ok(found)
    }

    /// Whether the surface is a combination of two states, alone or with terms added to it
    /// (what [`Self::crossing_states`] looks at).
    pub fn is_two_state(&self) -> bool {
        let two = |e: &Self| matches!(e, Self::Combine { inputs, .. } if inputs.len() == 2);
        match self {
            Self::Combine { .. } => two(self),
            Self::Linear(items) => items.iter().filter(|(_, e)| two(e)).count() == 1,
            _ => false,
        }
    }

    /// Hand `directions` to the combiner of a crossing search as its branching plane (see
    /// [`StateCombiner::set_branching_plane`]).
    pub fn set_branching_plane(&mut self, directions: &[Col<f64>]) {
        match self {
            Self::Combine { combiner, .. } => combiner.set_branching_plane(directions),
            Self::Linear(items) => {
                for (_, expr) in items {
                    if matches!(expr, Self::Combine { .. }) {
                        expr.set_branching_plane(directions);
                    }
                }
            }
            _ => {}
        }
    }

    /// [`Self::crossing`] for the derivatives of `request`: without Hessians when it asks
    /// for none.
    pub fn crossing_states(
        &self,
        sys: &MolecularSystem,
        request: &EvalRequest,
        results: Vec<EvalOutput>,
        term_ctx: &TermContext<'_>,
    ) -> MoptResult<Option<CrossingStates>> {
        let is_combine = |e: &Self| matches!(e, Self::Combine { .. });
        let (node, added): (&Self, Vec<&(f64, Self)>) = match self {
            Self::Combine { .. } => (self, Vec::new()),
            Self::Linear(items) => {
                let found: Vec<&(f64, Self)> =
                    items.iter().filter(|(_, e)| is_combine(e)).collect();
                match found[..] {
                    [(weight, node)] if *weight == 1.0 => {
                        (node, items.iter().filter(|(_, e)| !is_combine(e)).collect())
                    }
                    _ => return Ok(None),
                }
            }
            _ => return Ok(None),
        };
        let Self::Combine {
            combiner,
            labels,
            inputs,
        } = node
        else {
            return Ok(None);
        };
        if inputs.len() != 2 {
            return Ok(None);
        }
        let cache = self.build_cache(sys, request, results, term_ctx, true)?;
        let mut walk = Walk {
            sys,
            cache: &cache,
            need_hessian: request.need_hessian,
            pass: Pass {
                include_terms: true,
                zero_task_hessians: false,
            },
            record: false,
            components: Vec::new(),
            states: Vec::new(),
        };
        let mut values = inputs
            .iter()
            .map(|e| e.eval(&mut walk))
            .collect::<MoptResult<Vec<Value>>>()?;
        for (weight, expr) in added {
            let extra = expr.eval(&mut walk)?;
            for value in &mut values {
                value.add_scaled(*weight, &extra);
            }
        }
        let hessians: Option<Vec<Mat<f64>>> = values.iter().map(|v| v.hessian.clone()).collect();
        let states = StateValues {
            energies: values.iter().map(|v| v.energy).collect(),
            gradients: values.into_iter().map(|v| v.gradient).collect(),
            hessians,
        };
        let Some(plane) = combiner.branching_plane(&states) else {
            return Ok(None);
        };
        Ok(Some(CrossingStates {
            labels: labels.clone(),
            states,
            plane,
        }))
    }

    /// A model Hessian `model` of the structure (one for every state) through the combiner of
    /// a surface of two states (alone or with terms added to it): the Hessian the combiner
    /// makes of two states that both have it, with their energies and gradients from the
    /// results of [`Self::tasks`] at `sys`; for a crossing search the projected `P H P` with
    /// the curvature of the gap along the difference gradient, for a function of the two
    /// states its chain rule. `None` for other surfaces, whose model Hessian is the model.
    pub fn model_hessian_of_states(
        &self,
        sys: &MolecularSystem,
        results: Vec<EvalOutput>,
        term_ctx: &TermContext<'_>,
        model: &Mat<f64>,
    ) -> MoptResult<Option<Mat<f64>>> {
        let is_combine = |e: &Self| matches!(e, Self::Combine { .. });
        let node = match self {
            Self::Combine { .. } => self,
            Self::Linear(items) => {
                let found: Vec<&(f64, Self)> =
                    items.iter().filter(|(_, e)| is_combine(e)).collect();
                match found[..] {
                    [(weight, node)] if *weight == 1.0 => node,
                    _ => return Ok(None),
                }
            }
            _ => return Ok(None),
        };
        let Self::Combine {
            combiner, inputs, ..
        } = node
        else {
            return Ok(None);
        };
        if inputs.len() != 2 {
            return Ok(None);
        }
        let request = EvalRequest::ENERGY_GRAD;
        // The Hessians of the states are replaced by the model below, so those of the terms
        // are not needed.
        let cache = self.build_cache(sys, &request, results, term_ctx, false)?;
        let mut walk = Walk {
            sys,
            cache: &cache,
            need_hessian: false,
            // The energies with the terms of the inputs (the shifts of a crossing of shifted
            // states), on which a function of the states depends.
            pass: Pass {
                include_terms: true,
                zero_task_hessians: false,
            },
            record: false,
            components: Vec::new(),
            states: Vec::new(),
        };
        let values = inputs
            .iter()
            .map(|e| e.eval(&mut walk))
            .collect::<MoptResult<Vec<Value>>>()?;
        let states = StateValues {
            energies: values.iter().map(|v| v.energy).collect(),
            gradients: values.into_iter().map(|v| v.gradient).collect(),
            hessians: Some(vec![model.clone(), model.clone()]),
        };
        Ok(combiner.combine(sys, &states)?.hessian)
    }

    /// Hand the recorded inputs to the combiners, in the post-order of [`Self::eval`].
    fn commit(&mut self, sys: &MolecularSystem, states: &mut impl Iterator<Item = StateValues>) {
        match self {
            Self::Task(_) | Self::Term { .. } => {}
            Self::Linear(v) => v.iter_mut().for_each(|(_, e)| e.commit(sys, states)),
            Self::Combine {
                combiner, inputs, ..
            } => {
                inputs.iter_mut().for_each(|e| e.commit(sys, states));
                if let Some(s) = states.next() {
                    combiner.commit(sys, &s);
                }
            }
        }
    }
}
