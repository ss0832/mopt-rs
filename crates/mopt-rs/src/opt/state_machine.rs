//! Ask/tell geometry optimizer.
//!
//! The optimizer never calls a calculator. [`GeometryOptimizer::ask`] returns the evaluations
//! needed at the current structure (one for an ordinary run; several for combined surfaces such
//! as crossing-point searches or EDEEL); [`GeometryOptimizer::tell`] consumes the results and
//! performs one iteration of MultiOptPy's `Optimize` loop: surface assembly (bias potentials and
//! state combination, see [`PesExpr`]), Hessian bookkeeping, trust radius, step generation and
//! enhancement, method blending, back-transformation, rigid-body alignment, convergence, and
//! dissociation checks. The same state machine is driven by Rust calculators
//! (`workflow::driver`) and by Python calculators (through the Python bindings, `mopt_rs`).
//!
//! MultiOptPy: one iteration of `optimization.py:Optimize` with
//!  `optimizer.py:CalculateMoveVector.calc_move_vector`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::sync::Arc;

use crate::domain::{Angstrom, AtomId, Bohr, Hartree, Quantity};
use faer::{Col, Mat};

use super::constrained::ConstraintProjection;
use super::convergence::{rms_all, ConvergenceCriteria, ConvergenceReport};
use super::crossing_check::{self, CrossingCheckSettings};
use super::mode_kill::{self, ModeKillSettings, ModeKillVerdict};
use super::registry::{MethodContext, MethodRegistry, ResolvedMethod};
use super::stationary::{StationaryKind, StationaryPoint};
use super::step::ccqn::CcqnSettings;
use super::step::{ensure_finite_step, PreviousPoint, StepInput};
use super::trust::{OuterTrustRadius, TrustSettings, TrustUpdateInput};
use super::update::apply_update;
use crate::coord::alignment::align_unweighted_centered;
use crate::coord::connectivity::BondConnectivity;
use crate::coord::system::transform::BackTransformOptions;
use crate::coord::system::{back_transform, CoordinateFrame, CoordinateRegistry, CoordinateSystem};
use crate::domain::{EvalOutput, EvalRequest, MolecularSystem};
use crate::error::{MoptError, MoptResult, ResultExt};
use crate::math::linalg::{col_dot, col_norm, symmetric_eigen};
use crate::math::projection::{
    build_tr_rot_basis, project_out_basis_mat, project_out_tr_rot_hessian,
};
use crate::model_hessian::ModelHessian;
use crate::pes::combine::{CrossingStates, EvalTask, PesExpr};
use crate::pes::constraint::{breaks_rigid_body, Constraint, ConstraintSet};
use crate::pes::term::{Frame, PesTerm, TermContext};

/// Optimizer settings (the `optmain` options handled by the state machine).
#[derive(Debug, Clone)]
pub struct OptimizerSettings {
    /// `-opt`: one method, or two blended by the RMS force. Empty: the method for the
    /// saddle order ([`super::registry::default_methods`]).
    pub methods: Vec<String>,
    /// `-order`: number of negative-curvature directions (0 = minimization).
    pub saddle_order: usize,
    /// `-ns`: maximum number of iterations (energy evaluations).
    pub max_iterations: usize,
    /// `-tcc` / `-lcc`.
    pub criteria: ConvergenceCriteria,
    /// `-tr` / `-mintr`.
    pub trust: TrustSettings,
    /// `-fc N`: request the exact Hessian every `N` iterations.
    pub exact_hessian_interval: Option<usize>,
    /// `-coord`: coordinate system name.
    pub coordinates: String,
    /// Remove rigid-body motion (false for model surfaces in a fixed frame).
    pub rigid_body_invariant: bool,
    /// `-dc`: stop when fragments separate by more than this distance.
    pub dissociation_distance: Option<Quantity<Bohr>>,
    /// CCQN reaction-coordinate axis and parameters (for `ccqn*` methods).
    pub ccqn: Option<CcqnSettings>,
    /// `-fix`, `-pc`: geometric constraints, held exactly (see [`crate::pes::constraint`]).
    pub constraints: Vec<Constraint>,
    /// `-modelhess`: model Hessian replacing the current Hessian (see
    /// [`crate::model_hessian::resolve`]); used when no exact Hessians are requested.
    pub model_hessian: Option<String>,
    /// `-mfc`: model Hessian every this many iterations.
    pub model_hessian_interval: usize,
    /// `-modekill`: at convergence, leave a stationary point with more negative curvatures
    /// than `saddle_order` (see [`crate::opt::mode_kill`]); off by default.
    pub mode_kill: Option<ModeKillSettings>,
    /// When a search for a conical intersection has converged, check its branching plane
    /// with gradients at displaced structures and go on if the point is not stationary on
    /// the seam (see [`crate::opt::crossing_check`]); on by default.
    pub crossing_check: Option<CrossingCheckSettings>,
    /// On a gradient projection of two states (`-mf opt_mesx`, `-mf opt_meci`): judge a
    /// step for the trust radius by the function the projected gradient belongs to instead of
    /// the mean energy of the two states (`-projection_trust merit`; `SeamPoint` in this
    /// module has the function).
    pub projection_trust: bool,
    /// `-d`: step factor of the methods that have one (`None`: their own).
    pub step_factor: Option<f64>,
    /// `-of`: groups of atoms that get the mean gradient of the group, so that each group
    /// moves as a whole with methods that do not use a Hessian.
    pub fragments: Vec<Vec<AtomId>>,
}

impl Default for OptimizerSettings {
    fn default() -> Self {
        Self {
            methods: Vec::new(),
            saddle_order: 0,
            max_iterations: 1000,
            criteria: ConvergenceCriteria::NORMAL,
            trust: TrustSettings::default(),
            exact_hessian_interval: None,
            coordinates: "cart".to_string(),
            rigid_body_invariant: true,
            dissociation_distance: Some(Quantity::<Angstrom>::new(10.0).to()),
            ccqn: None,
            constraints: Vec::new(),
            model_hessian: None,
            model_hessian_interval: 50,
            mode_kill: None,
            crossing_check: Some(CrossingCheckSettings::default()),
            projection_trust: false,
            step_factor: None,
            fragments: Vec::new(),
        }
    }
}

/// The two states of a gradient projection at one structure, as an iteration reports them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CrossingReport {
    /// The energy of the first state minus that of the second (Hartree).
    pub gap: f64,
    /// The norm of the difference of the gradients of the two states (Hartree/Bohr).
    pub gradient_difference: f64,
}

/// What a step on a gradient projection of two states is judged by.
///
/// The projected gradient `G = 2 (E_1 - E_2) x + P (g_1 + g_2) / 2` is not the gradient of the
/// mean energy that the search reports: the first term closes the gap, and the second leaves
/// out the branching plane. So the ratio of the change of the mean energy to the prediction
/// of the quadratic model of `G` says nothing about the model (measured on a model of two
/// crossing surfaces: ratios of -98, 284, -48 in consecutive steps, and a trust radius that
/// shrinks at every step). `G` is, with `|g_1 - g_2|` and the plane held, the gradient of
///
/// `Phi = (E_1 + E_2) / 2 within the seam + (E_1 - E_2)^2 / |g_1 - g_2|`,
///
/// whose curvature along `x` is the `2 |g_1 - g_2|` of the Hessian of the projection. Its
/// decrease over a step `s` from the point kept here is
///
/// `(f - f') + sum_u (g . u) (u . s) + ((E_1 - E_2)^2 - (E_1' - E_2')^2) / |g_1 - g_2|`,
///
/// with the mean energy `f`, the mean gradient `g`, and the unit vectors `u` of the branching
/// plane at the point (the second term takes the change of `f` along the plane out, to first
/// order). The trust radius takes this in the place of `f - f'` (composite-step trust regions
/// for equality constraints judge a step by such a merit function: Byrd, Schnabel, Shultz,
/// SIAM J. Numer. Anal. 1987, 24, 1152).
#[derive(Debug, Clone)]
struct SeamPoint {
    /// Flat Cartesian coordinates (Bohr).
    x: Col<f64>,
    /// `E_1 - E_2`.
    gap: f64,
    /// `|g_1 - g_2|`.
    difference: f64,
    /// `(E_1 + E_2) / 2`.
    mean_energy: f64,
    /// `(g_1 + g_2) / 2`.
    mean_gradient: Col<f64>,
    /// Unit vectors of the branching plane.
    plane: Vec<Col<f64>>,
}

impl SeamPoint {
    /// The decrease of the function of the projected gradient from this point to `next`.
    fn decrease_to(&self, next: &Self) -> f64 {
        let n = self.x.nrows();
        if next.x.nrows() != n {
            return self.mean_energy - next.mean_energy;
        }
        let s = Col::from_fn(n, |i| next.x[i] - self.x[i]);
        let along: f64 = self
            .plane
            .iter()
            .filter(|u| u.nrows() == n)
            .map(|u| col_dot(&self.mean_gradient, u) * col_dot(u, &s))
            .sum();
        let closed = if self.difference > 1e-10 {
            (self.gap * self.gap - next.gap * next.gap) / self.difference
        } else {
            0.0
        };
        (self.mean_energy - next.mean_energy) + along + closed
    }
}

/// Give every atom of a group the mean gradient of the group (MultiOptPy
/// `calc_fragment_grads`).
fn average_over_fragments(gradient: &mut Col<f64>, fragments: &[Vec<AtomId>]) {
    for fragment in fragments {
        if fragment.is_empty() {
            continue;
        }
        for c in 0..3 {
            let mean =
                fragment.iter().map(|a| gradient[a.dof(c)]).sum::<f64>() / fragment.len() as f64;
            for a in fragment {
                gradient[a.dof(c)] = mean;
            }
        }
    }
}

/// Next action requested by the optimizer.
#[derive(Debug, Clone)]
pub enum Action {
    /// Perform these evaluations and pass the results to `tell` in the same order.
    Evaluate(Vec<EvalTask>),
    /// The optimization has terminated.
    Finished(Termination),
}

/// Why the optimization stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Termination {
    /// All convergence criteria were met.
    Converged,
    /// The iteration limit was reached.
    MaxIterations,
    /// Fragments separated beyond the dissociation distance.
    Dissociated,
    /// A search for a conical intersection met its criteria, but the check of its branching
    /// plane ([`crate::opt::crossing_check`]) found it not stationary on the seam every time
    /// the search was sent on with a corrected plane: the structure is not a converged
    /// crossing point.
    SeamCheckFailed,
}

/// `<S^2>` of one evaluation.
#[derive(Debug, Clone, PartialEq)]
pub struct SpinSquare {
    /// Label of the evaluation.
    pub label: String,
    /// `<S^2>` of the wavefunction.
    pub value: f64,
    /// `S (S + 1)` of the multiplicity asked for.
    pub exact: f64,
}

/// Values of the coordinates an optimization runs in (other than Cartesian ones).
#[derive(Debug, Clone, PartialEq)]
pub struct CoordinateValues {
    /// Name of the coordinate system.
    pub system: String,
    /// Names of the coordinates where each is one internal coordinate.
    pub labels: Option<Vec<String>>,
    /// Their values in atomic units (Bohr, radian).
    pub values: Vec<f64>,
}

/// Summary of one iteration.
#[derive(Debug, Clone)]
pub struct IterationReport {
    /// Iteration index.
    pub iteration: usize,
    /// Structure that was evaluated.
    pub system: MolecularSystem,
    /// Energy of the surface without in-process terms (bias potentials).
    pub energy: Quantity<Hartree>,
    /// Energy of the full surface.
    pub biased_energy: Quantity<Hartree>,
    /// Cartesian gradient of the full surface.
    pub gradient: Col<f64>,
    /// Cartesian gradient of the surface without in-process terms.
    pub raw_gradient: Col<f64>,
    /// Energies of the evaluations, terms, and combined states that make up the surface.
    pub components: Vec<(String, Quantity<Hartree>)>,
    /// Convergence test.
    pub convergence: ConvergenceReport,
    /// Norm of the Cartesian step taken.
    pub step_norm: Quantity<Bohr>,
    /// Outer trust radius used to cap the step.
    pub trust_radius: Quantity<Bohr>,
    /// Notes from method resolution and blending.
    pub notes: Vec<String>,
    /// What the methods did inside their steps (debug level of the run log).
    pub debug_notes: Vec<String>,
    /// What the programs behind the calculators printed during the evaluations (label of
    /// the evaluation, text).
    pub program_outputs: Vec<(String, String)>,
    /// The structure in the coordinates of the optimization, unless they are Cartesian.
    pub coordinates: Option<CoordinateValues>,
    /// `<S^2>` of the evaluations whose calculators tell it.
    pub spin_squares: Vec<SpinSquare>,
    /// Eigenvalues of the Hessian of the full surface (rigid motions projected out and
    /// left out), in an iteration in which it was evaluated.
    pub hessian_eigenvalues: Option<Vec<f64>>,
    /// On a gradient projection of two states: their gap and the norm of the difference of
    /// their gradients at the structure.
    pub crossing: Option<CrossingReport>,
}

/// `h` with its block of rigid-body motions replaced by `scale` times the unit matrix:
/// `P h P + scale D D^T` with the orthonormal rigid-body basis `D` and `P = 1 - D D^T`.
///
/// A model Hessian of the bonded curvature (Lindh, Swart, Fischer, Schlegel) has none along
/// the translations and rotations. On a surface that is not invariant under them (an external
/// field or a tether in the calculator, or a calculator that says so; not constraints that
/// pin the frame, which have their own projection) the gradient has
/// components there, the step generators drop modes of zero curvature as rigid-body ones, and
/// the updates never learn the curvature, as no step goes there: the run stalls with a constant
/// gradient. Those directions get the curvature of a start without a model instead.
///
/// MultiOptPy: no counterpart (new in mopt-rs; MultiOptPy takes the model as it is).
fn with_rigid_body_curvature(sys: &MolecularSystem, h: &Mat<f64>, scale: f64) -> Mat<f64> {
    let d = build_tr_rot_basis(sys, false);
    let mut out = project_out_basis_mat(&d, h);
    out += (&d * d.transpose()) * scale;
    out
}

/// Energies and gradients of one evaluated structure, of the calculator alone (`raw_*`) and
/// of the whole surface.
#[derive(Debug, Clone)]
struct LastEvaluation {
    raw_energy: Quantity<Hartree>,
    raw_gradient: Col<f64>,
    energy: Quantity<Hartree>,
    gradient: Col<f64>,
    /// The iteration whose surface it was.
    iteration: usize,
}

struct Previous {
    system: MolecularSystem,
    point: PreviousPoint,
    cartesian_step_norm: Quantity<Bohr>,
    /// The exact Hessian of the in-process terms at that point (optimizer coordinates, with
    /// the constraint projection), where it was computed: the quadratic model of the step
    /// is that of the whole surface, the quasi-Newton Hessian of the calculator plus this.
    bias_hessian: Option<Mat<f64>>,
}

/// Ask/tell geometry optimizer.
pub struct GeometryOptimizer {
    settings: OptimizerSettings,
    methods: Vec<ResolvedMethod>,
    coords: Box<dyn CoordinateSystem>,
    surface: PesExpr,
    /// Evaluations requested at the current structure.
    pending: Vec<EvalTask>,
    initial: MolecularSystem,
    current: MolecularSystem,
    iteration: usize,
    hessian: Mat<f64>,
    previous: Option<Previous>,
    trust: OuterTrustRadius,
    termination: Option<Termination>,
    pending_notes: Vec<String>,
    last_hessian_update: Option<String>,
    /// In a saddle-point search: the number of negative curvatures of the last exact or model
    /// Hessian, which the quasi-Newton updates after it may not exceed (with the saddle order)
    /// ([crate::opt::update::keeps_inertia]).
    negatives_at_reset: Option<usize>,
    /// The last evaluated structure: the energy and gradient of the calculator alone and of
    /// the whole surface (with the in-process terms), as consistent pairs.
    last_eval: Option<LastEvaluation>,
    /// Resolved constraints (`None` until resolved; eigenvector constraints wait for the first
    /// Hessian).
    constraints: Option<ConstraintSet>,
    /// `-modelhess`.
    model: Option<Box<dyn ModelHessian>>,
    /// The Hessian is requested at the current structure for the mode-kill check.
    verify_hessian: bool,
    /// Displacements made by the mode kill so far.
    mode_kill_rounds: usize,
    hisd_mode_kill: bool,
    /// The check of the branching plane under way (see [`crate::opt::crossing_check`]).
    probe: Option<Probe>,
    /// Times the search was sent on with a corrected branching plane.
    crossing_restarts: usize,
    /// The two states at the structure of the iteration before, on a gradient projection.
    seam: Option<SeamPoint>,
}

/// A check of the branching plane of a converged search for a conical intersection.
struct Probe {
    /// Report of the converged iteration.
    report: IterationReport,
    /// Unit vector along the difference gradient.
    x: Col<f64>,
    /// Mean gradient of the two states on the whole surface.
    mean_gradient: Col<f64>,
    /// Second direction of the branching plane as the search has it.
    given: Col<f64>,
    /// Direction of the displacements of the product under way.
    direction: Col<f64>,
    /// Products done.
    done: usize,
}

/// What follows a converged iteration.
enum AfterConvergence {
    Finish,
    /// Ask for the Hessian at the same structure.
    CheckHessian,
    /// Go on from this structure (mode kill).
    Continue(Box<MolecularSystem>),
    /// The mode kill of a checked Hessian ([`ModeKillSettings::checked_hessian`]) used up its
    /// displacements and the structure still has more imaginary modes than asked for: not a
    /// stationary point of the order asked for, so the run does not end as converged.
    WrongOrder,
}

impl GeometryOptimizer {
    /// Create an optimizer with the default method and coordinate registries.
    pub fn new(settings: OptimizerSettings, initial: MolecularSystem) -> MoptResult<Self> {
        Self::with_registries(
            settings,
            initial,
            &MethodRegistry::default(),
            &CoordinateRegistry::default(),
        )
    }

    /// Create an optimizer resolving names in the given registries.
    pub fn with_registries(
        settings: OptimizerSettings,
        initial: MolecularSystem,
        methods: &MethodRegistry,
        coordinates: &CoordinateRegistry,
    ) -> MoptResult<Self> {
        let mut settings = settings;
        if settings.methods.is_empty() {
            settings.methods = super::registry::default_methods(settings.saddle_order);
        }
        if settings.methods.len() > 2 {
            return Err(MoptError::config("-opt takes one or two methods"));
        }
        if settings.max_iterations == 0 {
            return Err(MoptError::config("the iteration limit must be positive"));
        }
        let mut settings = settings;
        // Frozen atoms, fixed coordinates, and oriented groups pin the frame: no re-centering
        // or rigid-body projection (MultiOptPy's `allactive_flag`).
        if breaks_rigid_body(&settings.constraints) {
            settings.rigid_body_invariant = false;
        }
        // O1NumHess is not a model: it is a Hessian from the calculator's gradients, asked for
        // like an exact Hessian every `-mfc` iterations (see `pes::o1numhess`).
        if settings
            .model_hessian
            .as_deref()
            .is_some_and(crate::pes::o1numhess::is_o1numhess)
        {
            settings.model_hessian = None;
            settings
                .exact_hessian_interval
                .get_or_insert(settings.model_hessian_interval.max(1));
        }
        let ctx = MethodContext {
            saddle_order: settings.saddle_order,
            trust: settings.trust,
            ccqn: settings.ccqn.clone().map(Arc::new),
        };
        let mut resolved: Vec<ResolvedMethod> = settings
            .methods
            .iter()
            .map(|m| methods.resolve(m, &ctx))
            .collect::<MoptResult<_>>()?;
        if let Some(delta) = settings.step_factor {
            if !(delta.is_finite() && delta > 0.0) {
                return Err(MoptError::config(format!(
                    "the step factor (-d) must be a positive number, got {delta}"
                )));
            }
            for m in &mut resolved {
                if !m.generator.set_step_factor(delta) {
                    m.warnings.push(format!(
                        "-d: {} has no step factor; the value is not used",
                        m.method
                    ));
                }
            }
        }
        for fragment in &settings.fragments {
            if let Some(a) = fragment.iter().find(|a| a.index() >= initial.n_atoms()) {
                return Err(MoptError::config(format!(
                    "-of: atom {} of a fragment does not exist ({} atoms)",
                    a.index() + 1,
                    initial.n_atoms()
                )));
            }
        }
        if !settings.fragments.is_empty() && resolved.iter().any(|m| m.generator.uses_hessian()) {
            if let Some(m) = resolved.first_mut() {
                m.warnings.push(
                    "-of: the fragments keep their shape only with methods that do not use a \
                     Hessian (as in MultiOptPy)"
                        .to_string(),
                );
            }
        }
        let pending_notes = resolved.iter().flat_map(|m| m.warnings.clone()).collect();
        // Every new geometry is re-centered at the origin (MultiOptPy's Kabsch step). Centering
        // the start as well keeps that translation out of the first secant pair.
        let initial = if settings.rigid_body_invariant && !initial.has_pbc() {
            align_unweighted_centered(&initial, &initial)?
        } else {
            initial
        };
        // Resolve the constraints now (those resolved before, at the input of a command, come
        // back unchanged) unless one needs the first Hessian, and start on them.
        settings.constraints = settings
            .constraints
            .iter()
            .map(|c| c.resolve(&initial, None))
            .collect::<MoptResult<_>>()
            .context("constraints at the start")?;
        let needs_hessian = settings.constraints.iter().any(Constraint::is_pending);
        let constraints = if settings.constraints.is_empty() || needs_hessian {
            None
        } else {
            Some(ConstraintSet::new(&settings.constraints, &initial, None)?)
        };
        let initial = match &constraints {
            Some(c) => c.restore(&initial).context("constraints at the start")?,
            None => initial,
        };
        let model = settings
            .model_hessian
            .as_deref()
            .map(crate::model_hessian::resolve)
            .transpose()?;
        let mut coords = coordinates.create(&settings.coordinates)?;
        coords
            .initialize(&initial)
            .context("coordinate system setup")?;
        // Constraints that pin the frame hold Cartesian values: coordinates without the
        // position and orientation of the structure (internal coordinates) cannot keep them
        // (the projection of a frozen atom in them takes away the internal motions of the
        // atom, and the run stalls with a gradient left).
        let coordinate_count = coords.dim(&initial);
        if breaks_rigid_body(&settings.constraints) && coordinate_count < initial.dof() {
            return Err(MoptError::unsupported(format!(
                "-coord {}: {coordinate_count} coordinates for {} Cartesian ones, without the \
                 position and orientation of the structure; constraints that pin the frame (-fix, \
                 -pc x|y|z|rot|eigvec|atoms_pair, -uc with x(), y(), z()) need coordinates that \
                 have them (cart, or tric)",
                coords.name(),
                initial.dof()
            )));
        }
        let mut pending_notes: Vec<String> = pending_notes;
        if let Some(c) = &constraints {
            pending_notes.extend(c.notes().iter().cloned());
        }
        if !settings.coordinates.eq_ignore_ascii_case(coords.name()) {
            pending_notes.push(format!("Coordinate system: {}", coords.name()));
        }
        // MultiOptPy starts from an identity Cartesian Hessian.
        let frame = CoordinateFrame::new(coords.as_ref(), &initial)?;
        // ... unless the method asks for another scale (e.g. CCQN's 70 eV/Angstrom^2).
        let scale = resolved
            .iter()
            .find_map(|m| m.generator.initial_hessian_scale())
            .unwrap_or(1.0);
        let hessian =
            frame.hessian_to_q(&(Mat::<f64>::identity(initial.dof(), initial.dof()) * scale));
        let uses_hessian = resolved.iter().any(|m| m.generator.uses_hessian());
        let adaptive = uses_hessian && settings.exact_hessian_interval.is_some();
        let trust = OuterTrustRadius::new(settings.trust, settings.saddle_order, adaptive);
        let mut opt = Self {
            settings,
            methods: resolved,
            coords,
            surface: PesExpr::single(),
            pending: Vec::new(),
            current: initial.clone(),
            initial,
            iteration: 0,
            hessian,
            previous: None,
            trust,
            termination: None,
            pending_notes,
            last_hessian_update: None,
            negatives_at_reset: None,
            last_eval: None,
            constraints,
            model,
            verify_hessian: false,
            mode_kill_rounds: 0,
            hisd_mode_kill: false,
            probe: None,
            crossing_restarts: 0,
            seam: None,
        };
        opt.refresh_pending()?;
        Ok(opt)
    }

    /// Replace the surface (default: the calculator on the whole structure).
    pub fn set_surface(&mut self, surface: PesExpr) -> MoptResult<()> {
        if self.iteration > 0 {
            return Err(MoptError::config(
                "the surface cannot be changed after the first iteration",
            ));
        }
        surface.validate()?;
        // An optimizer that removes rigid-body motions (recentering, superposition, projection
        // of the rigid modes) is only right on a surface that does not see them; a term that
        // depends on where the structure is must have been accounted for when the settings
        // were made (`pes::term::frame_invariant`).
        if self.settings.rigid_body_invariant {
            let moving: Vec<String> = surface
                .terms()?
                .iter()
                .filter(|t| t.frame() != Frame::Invariant)
                .map(|t| format!("'{}' ({:?})", t.name(), t.frame()))
                .collect();
            if !moving.is_empty() {
                return Err(MoptError::config(format!(
                    "the term(s) {} depend on the position or orientation of the structure, but \
                     the optimizer removes rigid-body motions; decide the setting \
                     rigid_body_invariant with pes::term::frame_invariant over the calculator \
                     and every term",
                    moving.join(", ")
                )));
            }
        }
        self.surface = surface;
        self.refresh_pending()
    }

    /// Whether the optimizer removes rigid-body motions (recentres and superposes the
    /// structures, projects the rigid modes): the setting, switched off by constraints that
    /// pin the frame.
    pub fn removes_rigid_body_motion(&self) -> bool {
        self.settings.rigid_body_invariant
    }

    /// Select the post-displacement method for an explicit standalone mode kill.
    pub(crate) fn enable_hisd_mode_kill(&mut self) {
        self.hisd_mode_kill = self.settings.mode_kill.is_some();
    }

    /// Follow `directions` (Cartesian displacements, one per mode of the saddle order) from
    /// the start instead of the lowest modes of the first Hessian: the mode closest to them is
    /// taken uphill, and then tracked by overlap. The method must follow modes
    /// (`mf_rsirfo_*`), and the coordinates must be Cartesian.
    pub fn follow_directions(&mut self, directions: &[Col<f64>]) -> MoptResult<()> {
        if !self.coords.is_cartesian() {
            return Err(MoptError::unsupported(
                "a direction to follow (-follow) is given in Cartesian coordinates; it is not \
                 available with -coord",
            ));
        }
        let n = 3 * self.current.n_atoms();
        if let Some(d) = directions.iter().find(|d| d.nrows() != n) {
            return Err(MoptError::config(format!(
                "a direction to follow has {} components for {n} coordinates",
                d.nrows()
            )));
        }
        let mut seeded = false;
        for m in &mut self.methods {
            seeded |= m.generator.seed_followed_modes(directions);
        }
        if seeded {
            Ok(())
        } else {
            Err(MoptError::config(
                "a direction to follow (-follow) needs a method that follows modes (mf_rsirfo_*)",
            ))
        }
    }

    /// Use `model` as the model Hessian (instead of the one named in the settings), e.g. a
    /// model of your own; `None` removes it.
    pub fn set_model_hessian(&mut self, model: Option<Box<dyn ModelHessian>>) {
        self.model = model;
    }

    /// Add an in-process term (e.g. a bias potential) to the whole surface.
    pub fn add_term(&mut self, label: &str, term: Arc<dyn PesTerm>) -> MoptResult<()> {
        let surface = std::mem::take(&mut self.surface).plus_term(label, term);
        self.set_surface(surface)
    }

    /// The surface being optimized.
    pub fn surface(&self) -> &PesExpr {
        &self.surface
    }

    /// The evaluations that give the surface with its Hessian at `sys` (for analyses after
    /// the optimization); pass their results to [`Self::surface_hessian`].
    pub fn hessian_tasks(&self, sys: &MolecularSystem) -> MoptResult<Vec<EvalTask>> {
        self.surface.tasks(sys, &EvalRequest::ENERGY_GRAD_HESS)
    }

    /// [`Self::hessian_tasks`] for a check of stability ([`EvalRequest::for_stability`]): a
    /// surface whose Hessian is an approximation for the steps computes it to convergence,
    /// or gives none.
    pub fn hessian_check_tasks(&self, sys: &MolecularSystem) -> MoptResult<Vec<EvalTask>> {
        self.surface
            .tasks(sys, &EvalRequest::ENERGY_GRAD_HESS_CHECK)
    }

    /// Energy, Cartesian gradient, and Cartesian Hessian of the whole surface (calculators
    /// and in-process terms such as bias potentials) at `sys`, from the results of
    /// [`Self::hessian_tasks`].
    pub fn surface_hessian(
        &mut self,
        sys: &MolecularSystem,
        results: Vec<EvalOutput>,
    ) -> MoptResult<(Quantity<Hartree>, Col<f64>, Mat<f64>)> {
        let ctx = TermContext::new(self.analysis_iteration(), &self.initial, None);
        let surf = self
            .surface
            .evaluate(sys, &EvalRequest::ENERGY_GRAD_HESS, results, &ctx)?;
        let mut hessian = surf
            .raw
            .hessian_bohr
            .ok_or_else(|| MoptError::calc("the surface returned no Hessian"))?;
        if let Some(terms) = surf.term_hessian {
            hessian += terms;
        }
        Ok((surf.total.energy, surf.total.gradient_bohr, hessian))
    }

    /// If the surface is a crossing search of two states: the states with their Hessians
    /// and the branching plane at `sys`, from the results of [`Self::hessian_tasks`]. The
    /// branching plane of a conical intersection is the one of the last iteration.
    pub fn crossing(
        &self,
        sys: &MolecularSystem,
        results: Vec<EvalOutput>,
    ) -> MoptResult<Option<CrossingStates>> {
        let ctx = TermContext::new(self.analysis_iteration(), &self.initial, None);
        self.surface.crossing(sys, results, &ctx)
    }

    /// [`Self::crossing`] for the derivatives of `request` (without Hessians when it asks
    /// for none), from the results of the tasks of the surface at `sys` for that request.
    pub fn crossing_states(
        &self,
        sys: &MolecularSystem,
        request: &EvalRequest,
        results: Vec<EvalOutput>,
    ) -> MoptResult<Option<CrossingStates>> {
        let ctx = TermContext::new(self.analysis_iteration(), &self.initial, None);
        self.surface.crossing_states(sys, request, results, &ctx)
    }

    /// Starting structure.
    pub fn initial(&self) -> &MolecularSystem {
        &self.initial
    }

    /// The constraints of the run with every value fixed ([`Constraint::resolve`]): at the
    /// start of the optimizer, or as they were handed in already resolved; eigenvector
    /// constraints once the first Hessian has resolved them (pending before). Follow-up stages
    /// (an IRC, a vibrational analysis, further optimizations) take these, so that they hold
    /// the values of the input and not those of their own starting structures.
    pub fn resolved_constraints(&self) -> &[Constraint] {
        &self.settings.constraints
    }

    /// The constraints as the optimizer holds them (`None` before they are resolved, and
    /// without constraints).
    pub fn constraint_set(&self) -> Option<&ConstraintSet> {
        self.constraints.as_ref()
    }

    /// The iteration whose surface the last evaluated structure was evaluated on (the
    /// iteration that terms with a ramp saw), `None` before the first evaluation: the surface
    /// on which a converged structure is stationary, for the analyses and paths that follow.
    pub fn last_surface_iteration(&self) -> Option<usize> {
        self.last_eval.as_ref().map(|l| l.iteration)
    }

    /// The iteration the terms see in the analyses after the run ([`Self::surface_hessian`],
    /// [`Self::crossing`]): that of the last evaluation, or of the start before any.
    fn analysis_iteration(&self) -> usize {
        self.last_surface_iteration().unwrap_or(self.iteration)
    }

    fn refresh_pending(&mut self) -> MoptResult<()> {
        self.pending = self.surface.tasks(&self.current, &self.request())?;
        Ok(())
    }

    /// Current structure (the last evaluated one after termination).
    pub fn current(&self) -> &MolecularSystem {
        &self.current
    }

    /// Iterations performed so far.
    pub fn iterations(&self) -> usize {
        self.iteration
    }

    /// The converged stationary point, if the run converged with the saddle order of `K`.
    pub fn stationary_point<K: StationaryKind>(&self) -> MoptResult<StationaryPoint<K>> {
        if self.termination != Some(Termination::Converged) {
            return Err(MoptError::conv(format!(
                "no {}: the optimization has not converged ({:?})",
                K::NAME,
                self.termination
            )));
        }
        if self.settings.saddle_order != K::SADDLE_ORDER {
            return Err(MoptError::config(format!(
                "a {} needs saddle order {}, but the run used {}",
                K::NAME,
                K::SADDLE_ORDER,
                self.settings.saddle_order
            )));
        }
        let last = self
            .last_eval
            .clone()
            .ok_or_else(|| MoptError::conv("no evaluation recorded"))?;
        Ok(StationaryPoint::new(
            self.current.clone(),
            last.raw_energy,
            last.energy,
            last.gradient,
            last.raw_gradient,
            self.iteration,
        ))
    }

    /// Termination reason, once finished.
    pub fn termination(&self) -> Option<Termination> {
        self.termination
    }

    /// Hessian in the optimizer's coordinates.
    pub fn hessian(&self) -> &Mat<f64> {
        &self.hessian
    }

    pub(crate) fn updated_hessian(&self) -> Option<(&str, &Mat<f64>)> {
        self.last_hessian_update
            .as_deref()
            .map(|method| (method, &self.hessian))
    }

    /// Whether the optimizer's coordinates are Cartesian (then [`Self::hessian`] is the
    /// Cartesian Hessian).
    pub fn is_cartesian(&self) -> bool {
        self.coords.is_cartesian()
    }

    /// Next action.
    pub fn ask(&self) -> Action {
        match self.termination {
            Some(t) => Action::Finished(t),
            None => Action::Evaluate(self.pending.clone()),
        }
    }

    /// Derivatives requested in the current iteration.
    fn request(&self) -> EvalRequest {
        let need_hessian = self.verify_hessian
            || self
                .settings
                .exact_hessian_interval
                .is_some_and(|n| n > 0 && self.iteration.is_multiple_of(n));
        if self.verify_hessian {
            // The Hessian of the mode kill is that of a check of stability.
            EvalRequest::ENERGY_GRAD_HESS_CHECK
        } else if need_hessian {
            EvalRequest::ENERGY_GRAD_HESS
        } else {
            EvalRequest::ENERGY_GRAD
        }
    }

    /// Consume the results of the evaluations returned by [`Self::ask`], in the same order.
    pub fn tell(&mut self, results: Vec<EvalOutput>) -> MoptResult<IterationReport> {
        self.last_hessian_update = None;
        if self.termination.is_some() {
            return Err(MoptError::config(
                "tell() called after the optimization finished",
            ));
        }
        let it = self.iteration;
        if self.probe.is_some() {
            return self
                .probe_step(results)
                .map_err(|e| e.context(format!("check of the branching plane, iteration {it}")));
        }
        self.step(results)
            .map_err(|e| e.context(format!("optimizer iteration {it}")))
    }

    /// The two structures of a product of the check: the current one displaced along
    /// `direction` and against it.
    fn probe_structures(&self, direction: &Col<f64>) -> MoptResult<[MolecularSystem; 2]> {
        let delta = self
            .settings
            .crossing_check
            .map_or(crossing_check::DEFAULT_DISPLACEMENT_BOHR, |c| {
                c.displacement
            });
        let x = self.current.coords_flat_bohr();
        let moved = |sign: f64| {
            self.current
                .with_flat_coords_bohr(&Col::from_fn(x.nrows(), |i| {
                    x[i] + sign * delta * direction[i]
                }))
        };
        Ok([moved(1.0)?, moved(-1.0)?])
    }

    /// The evaluations of a product of the check.
    fn probe_tasks(&self, direction: &Col<f64>) -> MoptResult<Vec<EvalTask>> {
        let [plus, minus] = self.probe_structures(direction)?;
        let mut tasks = self.surface.tasks(&plus, &EvalRequest::ENERGY_GRAD)?;
        tasks.extend(self.surface.tasks(&minus, &EvalRequest::ENERGY_GRAD)?);
        Ok(tasks)
    }

    /// A direction for the first product: without symmetry, orthogonal to `x`, and without
    /// rigid-body motion.
    fn probe_seed(&self, x: &Col<f64>) -> MoptResult<Option<Col<f64>>> {
        let mut seed = crossing_check::seed(x.nrows());
        if self.settings.rigid_body_invariant && !self.current.has_pbc() {
            let rigid = build_tr_rot_basis(&self.current, false);
            seed = crate::math::projection::project_out_basis_vec(&rigid, &seed);
        }
        let seed = self.within_constraints(&seed)?;
        Ok(crossing_check::orthogonal_unit(
            &seed,
            &self.constrained_unit(x)?,
        ))
    }

    /// The unit vector `x` taken into the directions the constraints allow and normalized
    /// again (`x` itself without constraints, or when nothing of it is left).
    fn constrained_unit(&self, x: &Col<f64>) -> MoptResult<Col<f64>> {
        if self
            .constraints
            .as_ref()
            .is_none_or(ConstraintSet::is_empty)
        {
            return Ok(x.clone());
        }
        let p = self.within_constraints(x)?;
        let length = col_norm(&p);
        Ok(if length > 1e-10 {
            Col::from_fn(p.nrows(), |i| p[i] / length)
        } else {
            x.clone()
        })
    }

    /// `v` (Cartesian) without its components along the gradients of the constraints at the
    /// current structure: a direction the constrained search can take, or a gradient as the
    /// convergence test sees it (`v` itself without constraints).
    fn within_constraints(&self, v: &Col<f64>) -> MoptResult<Col<f64>> {
        match &self.constraints {
            Some(c) if !c.is_empty() => Ok(c.frame(&self.current, v)?.project_gradient(v)),
            _ => Ok(v.clone()),
        }
    }

    /// Consume the results of a product of the check of the branching plane.
    fn probe_step(&mut self, results: Vec<EvalOutput>) -> MoptResult<IterationReport> {
        let Some(mut probe) = self.probe.take() else {
            return Err(MoptError::config(
                "no check of the branching plane under way",
            ));
        };
        let check = self.settings.crossing_check.unwrap_or_default();
        let half = results.len() / 2;
        let mut results = results;
        let second = results.split_off(half);
        let [plus, minus] = self.probe_structures(&probe.direction)?;
        let ctx = TermContext::new(self.iteration, &self.initial, None);
        let request = EvalRequest::ENERGY_GRAD;
        let difference =
            |sys: &MolecularSystem, results: Vec<EvalOutput>| -> MoptResult<Col<f64>> {
                let states = self
                    .surface
                    .crossing_states(sys, &request, results, &ctx)?
                    .ok_or_else(|| MoptError::calc("the surface is not a crossing of two states"))?
                    .states;
                Ok(&states.gradients[0] - &states.gradients[1])
            };
        let (d_plus, d_minus) = (difference(&plus, results)?, difference(&minus, second)?);
        let n = d_plus.nrows();
        let w = Col::from_fn(n, |i| (d_plus[i] - d_minus[i]) / (2.0 * check.displacement));
        probe.done += 1;
        let mut report = probe.report.clone();
        report.iteration = self.iteration;
        self.iteration += 1;
        let out_of_iterations = self.iteration >= self.settings.max_iterations;
        // Only directions the constrained search can take: the product along a direction the
        // constraints forbid says nothing about the seam the search moves on.
        let w = self.within_constraints(&w)?;
        let x_free = self.constrained_unit(&probe.x)?;
        let Some(y) = crossing_check::orthogonal_unit(&w, &x_free) else {
            report.notes = vec![
                "Check of the branching plane: the difference of the two gradients does not \
                 change with the structure; the converged structure is accepted as it is."
                    .to_string(),
            ];
            self.termination = Some(Termination::Converged);
            return Ok(report);
        };
        if probe.done < check.products && !out_of_iterations {
            report.notes = vec![format!(
                "Check of the branching plane: product {} of {}.",
                probe.done, check.products
            )];
            self.pending = self.probe_tasks(&y)?;
            probe.direction = y;
            self.probe = Some(probe);
            return Ok(report);
        }
        let overlap = col_dot(&y, &probe.given).abs();
        // The gradient the convergence of the search is judged by: without the constraint
        // forces (frozen atoms, held distances), as in the iterations.
        let rest = crossing_check::within_seam(
            &self.within_constraints(&probe.mean_gradient)?,
            &x_free,
            &y,
        );
        let test = self
            .settings
            .criteria
            .check(&rest, &Col::zeros(rest.nrows()));
        let summary = format!(
            "Check of the branching plane: its second direction from gradients at displaced \
             structures overlaps {overlap:.4} with that of the search; RMS of the mean gradient \
             within the seam {:.3e} Hartree/Bohr",
            test.rms_force.value()
        );
        if test.converged {
            report.notes = vec![format!("{summary}: converged.")];
            self.termination = Some(Termination::Converged);
        } else if self.crossing_restarts >= check.max_restarts || out_of_iterations {
            report.notes = vec![format!(
                "{summary}: the structure is not stationary on the seam, and the search stops \
                 after {} corrections of the branching plane.",
                self.crossing_restarts
            )];
            self.termination = Some(if out_of_iterations {
                Termination::MaxIterations
            } else {
                Termination::SeamCheckFailed
            });
        } else {
            report.notes = vec![format!(
                "{summary}: the structure is not stationary on the seam; the search goes on \
                 with this branching plane."
            )];
            self.surface.set_branching_plane(&[probe.x, y]);
            self.crossing_restarts += 1;
            self.previous = None;
            for m in &mut self.methods {
                m.generator.reset();
                if let Some(up) = m.updater.as_mut() {
                    up.reset();
                }
                for e in &mut m.enhancers {
                    e.reset();
                }
            }
            self.refresh_pending()?;
        }
        Ok(report)
    }

    /// [`Self::tell`] for a surface with a single evaluation.
    pub fn tell_one(&mut self, eval: EvalOutput) -> MoptResult<IterationReport> {
        if self.pending.len() != 1 {
            return Err(MoptError::config(format!(
                "the surface needs {} evaluations per iteration; use tell()",
                self.pending.len()
            )));
        }
        self.tell(vec![eval])
    }

    fn step(&mut self, results: Vec<EvalOutput>) -> MoptResult<IterationReport> {
        let mut notes = std::mem::take(&mut self.pending_notes);

        // Assemble the surface: calculators, bias potentials, and state combination.
        let ctx = TermContext::new(self.iteration, &self.initial, None);
        let request = self.request();
        let spin_squares: Vec<SpinSquare> = self
            .pending
            .iter()
            .zip(&results)
            .filter_map(|(task, out)| {
                let value = out.orbitals.as_ref()?.spin_square?;
                let s = 0.5 * (f64::from(task.system.spin_charge.multiplicity) - 1.0);
                Some(SpinSquare {
                    label: task.label.clone(),
                    value,
                    exact: s * (s + 1.0),
                })
            })
            .collect();
        let program_outputs: Vec<(String, String)> = self
            .pending
            .iter()
            .zip(&results)
            .filter_map(|(task, out)| {
                let text = out.orbitals.as_ref()?.program_output.clone()?;
                Some((task.label.clone(), text))
            })
            .collect();
        let verifying = std::mem::take(&mut self.verify_hessian);
        // Kept for the check of the branching plane of a crossing search.
        let for_crossing = (self.settings.crossing_check.is_some() && self.surface.is_two_state())
            .then(|| results.clone());
        // The energies and gradients of the evaluations (no Hessians), for the two states of
        // a gradient projection at this structure.
        let light: Option<Vec<EvalOutput>> = self.surface.is_two_state().then(|| {
            results
                .iter()
                .map(|r| EvalOutput::energy_gradient(r.energy, r.gradient_bohr.clone()))
                .collect()
        });
        // Kept for a model Hessian of a surface of two states, which goes through its
        // combiner like an exact one (it needs the energies and gradients only).
        let model_due = self.settings.exact_hessian_interval.is_none()
            && self.model.is_some()
            && self
                .iteration
                .is_multiple_of(self.settings.model_hessian_interval.max(1));
        let for_model = if model_due { light.clone() } else { None };
        // The Hessians of the terms only where something reads them: a method or enhancer of
        // the step (the methods with a Hessian, GPMin, TRIM), or an iteration that asks for a
        // Hessian anyway (the exact Hessian, the mode kill). Gradient-only methods do without.
        let term_hessians = self.methods.iter().any(|m| {
            m.generator.uses_bias_hessian() || m.enhancers.iter().any(|e| e.uses_bias_hessian())
        });
        let mut surf = if term_hessians {
            self.surface
                .evaluate(&self.current, &request, results, &ctx)?
        } else {
            self.surface
                .evaluate_as_requested(&self.current, &request, results, &ctx)?
        };
        let seam_now: Option<SeamPoint> = match light {
            Some(light) => self
                .surface
                .crossing_states(&self.current, &EvalRequest::ENERGY_GRAD, light, &ctx)?
                .and_then(
                    |c| match (&c.states.energies[..], &c.states.gradients[..]) {
                        ([e1, e2], [g1, g2]) => Some(SeamPoint {
                            x: self.current.coords_flat_bohr(),
                            gap: e1 - e2,
                            difference: col_norm(&(g1 - g2)),
                            mean_energy: 0.5 * (e1 + e2),
                            mean_gradient: Col::from_fn(g1.nrows(), |i| 0.5 * (g1[i] + g2[i])),
                            plane: c.plane,
                        }),
                        _ => None,
                    },
                ),
            None => None,
        };
        let crossing_report = seam_now.as_ref().map(|s| CrossingReport {
            gap: s.gap,
            gradient_difference: s.difference,
        });
        if !self.settings.fragments.is_empty() {
            average_over_fragments(&mut surf.total.gradient_bohr, &self.settings.fragments);
            average_over_fragments(&mut surf.raw.gradient_bohr, &self.settings.fragments);
        }
        let exact_now = surf.raw.hessian_bohr.is_some();
        let hessian_eigenvalues = match surf.total.hessian_bohr.as_ref() {
            Some(h) if exact_now => {
                let projected = if self.settings.rigid_body_invariant {
                    project_out_tr_rot_hessian(&self.current, h, false)
                } else {
                    h.clone()
                };
                let (values, _) = symmetric_eigen(&projected)?;
                Some(
                    values
                        .iter()
                        .copied()
                        .filter(|v| v.abs() > 1e-10)
                        .collect::<Vec<f64>>(),
                )
            }
            _ => None,
        };
        let raw_energy = surf.raw.energy;
        let biased_energy = surf.total.energy;
        let biased_grad = surf.total.gradient_bohr;

        // Coordinates whose definition is about to break down at this structure are set up
        // anew here, before anything is expressed in them.
        if self.coords.needs_reinitialization(&self.current) {
            self.reinitialize_coordinates(&mut notes)?;
        }

        // Transform to the optimizer's coordinates.
        let frame = CoordinateFrame::new(self.coords.as_ref(), &self.current)?;
        if let Some(h) = &surf.raw.hessian_bohr {
            self.hessian = frame.hessian_to_q(h);
        }
        // Model Hessian (MultiOptPy: every `-mfc` iterations when `-fc` is not used).
        let model_now = self.settings.exact_hessian_interval.is_none()
            && surf.raw.hessian_bohr.is_none()
            && self
                .iteration
                .is_multiple_of(self.settings.model_hessian_interval.max(1));
        let model_now = match (&self.model, model_now) {
            (Some(model), true) => {
                let mut h = model
                    .hessian(&self.current, &surf.raw.gradient_bohr)
                    .context(format!("model Hessian '{}'", model.name()))?;
                // Only where the surface itself is not invariant: constraints that pin the frame
                // (frozen atoms, fixed coordinates) have their own projection, and the bonds to
                // the frozen atoms give the model its curvature there.
                if !self.settings.rigid_body_invariant
                    && !breaks_rigid_body(&self.settings.constraints)
                    && !self.current.has_pbc()
                {
                    let scale = self
                        .methods
                        .iter()
                        .find_map(|m| m.generator.initial_hessian_scale())
                        .unwrap_or(1.0);
                    h = with_rigid_body_curvature(&self.current, &h, scale);
                }
                if let Some(results) = for_model {
                    let here = TermContext::new(self.iteration, &self.initial, None);
                    if let Some(through) = self
                        .surface
                        .model_hessian_of_states(&self.current, results, &here, &h)
                        .context("the model Hessian of the two states")?
                    {
                        h = through;
                    }
                }
                if self.settings.rigid_body_invariant && !self.current.has_pbc() {
                    h = project_out_tr_rot_hessian(&self.current, &h, false);
                }
                // A model can break down at a degenerate structure (MultiOptPy takes the
                // Hessian as it is, after which RS-I-RFO falls back to steepest descent for the
                // rest of the run): keep the Hessian in use and say so.
                if (0..h.nrows()).all(|i| (0..h.ncols()).all(|j| h[(i, j)].is_finite())) {
                    self.hessian = frame.hessian_to_q(&h);
                    true
                } else {
                    notes.push(format!(
                        "Model Hessian '{}' is not finite at this structure; the current \
                         Hessian is kept.",
                        model.name()
                    ));
                    false
                }
            }
            _ => false,
        };
        if self.constraints.is_none() && !self.settings.constraints.is_empty() {
            // The eigenvector constraints resolve at the first Hessian; the resolved set is
            // what follow-up stages receive ([`Self::resolved_constraints`]).
            let h = surf.raw.hessian_bohr.as_ref();
            self.settings.constraints = self
                .settings
                .constraints
                .iter()
                .map(|c| c.resolve(&self.current, h))
                .collect::<MoptResult<_>>()
                .context("constraints")?;
            let set = ConstraintSet::new(&self.settings.constraints, &self.current, h)
                .context("constraints")?;
            notes.extend(set.notes().iter().cloned());
            self.constraints = Some(set);
        }
        let cframe = self
            .constraints
            .as_ref()
            .map(|c| c.frame(&self.current, &biased_grad))
            .transpose()?;
        let projection = cframe
            .as_ref()
            .map(|cf| ConstraintProjection::new(&frame, cf))
            .transpose()?;
        // The gradient without the constraint forces decides convergence.
        let free_grad = match &cframe {
            Some(cf) => cf.project_gradient(&biased_grad),
            None => biased_grad.clone(),
        };
        let mut g_q = frame.gradient_to_q(&biased_grad);
        if let Some(p) = &projection {
            g_q = p.vector(&g_q);
        }
        let raw_q = frame.gradient_to_q(&surf.raw.gradient_bohr);
        let bias_hessian_q = surf.term_hessian.as_ref().map(|h| match &projection {
            Some(p) => p.matrix(&frame.hessian_to_q(h)),
            None => frame.hessian_to_q(h),
        });

        // Outer trust radius (uses the Hessian before this iteration's update).
        // On a gradient projection the step is judged by the function of the projected
        // gradient (`projection_trust`).
        let merit = match (&self.seam, &seam_now) {
            (Some(before), Some(now)) if self.settings.projection_trust => {
                Some(before.decrease_to(now))
            }
            _ => None,
        };
        // The prediction is that of the model the step was taken with: the energies and
        // gradients include the bias potentials, so the Hessian does as well.
        let model_hessian = self
            .previous
            .as_ref()
            .and_then(|p| p.bias_hessian.as_ref())
            .filter(|b| b.nrows() == self.hessian.nrows() && b.ncols() == self.hessian.ncols())
            .map(|b| &self.hessian + b);
        let trust_input = self.previous.as_ref().map(|p| TrustUpdateInput {
            energy: biased_energy,
            prev_energy: p.point.energy,
            prev_gradient: &p.point.gradient,
            prev_step: &p.point.step,
            prev_cartesian_step_norm: p.cartesian_step_norm,
            hessian: model_hessian.as_ref().unwrap_or(&self.hessian),
            decrease: merit,
        });
        self.trust.update(trust_input);
        self.seam = seam_now;
        notes.extend(self.trust.take_notes());

        // The inertia of a new exact or model Hessian bounds the updates after it. Where the
        // surface does not change under rigid motions, their directions are left out of the
        // count: their curvatures are zero up to the noise of the Hessian (an exact Hessian of
        // HCN had one of them at -1e-5), and counted they let the guard below skip updates of
        // a search whose inertia is fine.
        let inertia_basis = (self.coords.is_cartesian() && self.settings.rigid_body_invariant)
            .then(|| build_tr_rot_basis(&self.current, false));
        let without_rigid = |h: &Mat<f64>| match &inertia_basis {
            Some(b) if b.nrows() == h.nrows() => project_out_basis_mat(b, h),
            _ => h.clone(),
        };
        if self.settings.saddle_order > 0 && (surf.raw.hessian_bohr.is_some() || model_now) {
            self.negatives_at_reset =
                crate::opt::update::negative_curvatures(&without_rigid(&self.hessian)).ok();
        }
        // Quasi-Newton update with the unbiased gradient.
        let mut update_debug: Vec<String> = Vec::new();
        if let Some(prev) = &self.previous {
            let s = self.coords.displacement(&self.current, &prev.system)?;
            let y = &raw_q - &prev.point.raw_gradient;
            let learns = self
                .methods
                .iter()
                .any(|m| m.generator.learns_negative_curvature());
            let saddle = self.settings.saddle_order > 0;
            if let Some(up) = self.methods.iter_mut().find_map(|m| m.updater.as_mut()) {
                // Secant pairs from before an exact Hessian are already contained in it; a block
                // updater must not impose them again (only the newest pair is applied, as in
                // MultiOptPy).
                if surf.raw.hessian_bohr.is_some() || model_now {
                    up.reset();
                }
                // Saddle searches learn negative curvature from the steps along the
                // transition vector (see `apply_update`).
                let negative = learns || (saddle && up.learns_saddle_curvature());
                // The secant mismatch the update acts on, for the debug level of the log (it is
                // to be compared with the noise of the gradients).
                if self.hessian.nrows() == s.nrows() && s.nrows() == y.nrows() {
                    let mismatch = &y - &self.hessian * &s;
                    update_debug.push(format!(
                        "Hessian update: |y - H s| = {:.3e}, |s| = {:.3e}, |y| = {:.3e}",
                        col_norm(&mismatch),
                        col_norm(&s),
                        col_norm(&y)
                    ));
                }
                // In a saddle search an update may not give the Hessian more negative curvatures
                // than the saddle order or the last exact Hessian had: near convergence the steps
                // are short and the change of a gradient with noise (a relaxed environment) is
                // read as curvature, which turned soft modes negative and sent the step along
                // them (docs/source/oniom.md).
                let before = saddle.then(|| self.hessian.clone());
                if apply_update(up.as_mut(), &mut self.hessian, &s, &y, negative)? {
                    self.last_hessian_update = Some(up.name().to_owned());
                    if let Some(before) = before {
                        let allowed = self
                            .settings
                            .saddle_order
                            .max(self.negatives_at_reset.unwrap_or(0));
                        if let Some(count) = crate::opt::update::keeps_inertia(
                            &without_rigid(&self.hessian),
                            allowed,
                        )? {
                            update_debug.push(format!(
                                "Hessian update skipped: it gave {count} negative curvature(s), \
                                 more than the {allowed} of the last exact Hessian and the \
                                 saddle order"
                            ));
                            self.hessian = before;
                            self.last_hessian_update = None;
                        }
                    }
                }
            }
        }

        let rigid = inertia_basis;
        let x = self.current.coords_flat_bohr();
        let step_hessian = match (&projection, &cframe) {
            (Some(p), Some(cf)) => p.hessian(&self.hessian, &frame.hessian_to_q(&cf.curvature)),
            _ => self.hessian.clone(),
        };
        let input = StepInput {
            iteration: self.iteration,
            cartesian: &x,
            energy: biased_energy,
            gradient: &g_q,
            raw_gradient: &raw_q,
            previous: self.previous.as_ref().map(|p| &p.point),
            hessian: &step_hessian,
            bias_hessian: bias_hessian_q.as_ref(),
            rigid_body_basis: rigid.as_ref(),
            saddle_order: self.settings.saddle_order,
            system: Some(&self.current),
            cartesian_coordinates: self.coords.is_cartesian(),
            frame: Some(&frame),
            cartesian_gradient: Some(&free_grad),
        };
        let mut proposals = Vec::with_capacity(self.methods.len());
        for m in &mut self.methods {
            let mut dq = m.generator.propose(&input).context(m.method.clone())?;
            for e in &mut m.enhancers {
                dq = e.enhance(&input, dq).context(e.name().to_string())?;
            }
            proposals.push(dq);
        }
        let mut dq = blend(&proposals, &free_grad, &self.methods, &mut notes);
        if let Some(p) = &projection {
            dq = p.vector(&dq);
        }

        let dx = back_transform(
            self.coords.as_ref(),
            &self.current,
            &dq,
            &BackTransformOptions::default(),
        )?;
        let self_limited = self
            .methods
            .iter()
            .all(|m| m.generator.controls_step_length());
        let dx = if self_limited && self.settings.trust.max.is_none() {
            dx
        } else {
            self.trust.cap(dx)
        };
        ensure_finite_step(&dx, &self.settings.methods)?;
        let step_norm = Quantity::<Bohr>::new(col_norm(&dx));
        let next = self.placed_on_constraints(&(&x + &dx))?;
        if let Some(c) = &self.constraints {
            notes.extend(c.lenient_notes());
        }
        let displacement = if self.iteration == 0 {
            dx.clone()
        } else {
            next.coords_flat_bohr() - &x
        };
        let mut convergence = self.settings.criteria.check(&free_grad, &displacement);
        // The check of a converged structure by a checked Hessian evaluates the same structure
        // again: it met every criterion at the iteration before, and this one judges its order.
        // Its forces must still meet theirs (the environment of the relaxed surface is relaxed
        // tighter for the check); the step that the exact Hessian proposes from it is not a
        // criterion again, since along the soft modes of a relaxed surface (curvatures of 3e-4
        // Hartree/Bohr^2 next to a gradient noise of 1e-5) it runs to several times the
        // displacement threshold however close the structure is (chorismate mutase: 4.7e-3
        // Bohr at a force of 1.2e-5, and the search wandered for 20 iterations after it).
        // The same holds for every check of the mode kill: on the relaxed surface of ONIOM
        // microiterations a minimization of the mapper (vinyl alcohol in two waters) ran its
        // 400 iterations between convergence and a check whose tighter environment changed
        // the proposed step, each time judged again on the displacement.
        let verified_by_forces = verifying
            && self.settings.mode_kill.is_some()
            && !convergence.converged
            && convergence.max_force < self.settings.criteria.max_force
            && convergence.rms_force < self.settings.criteria.rms_force;
        if verified_by_forces {
            convergence.converged = true;
            notes.push(format!(
                "The structure converged at the iteration before; its check meets the criteria of \
                 the forces (the step of {:.2e} Bohr that its exact Hessian proposes is not judged \
                 again).",
                convergence.max_displacement.value()
            ));
        }

        // Mode kill: a converged structure must not have more imaginary modes than asked.
        let mut after = AfterConvergence::Finish;
        if let Some(mk) = self.settings.mode_kill.filter(|_| convergence.converged) {
            let order = self.settings.saddle_order;
            // A surface whose Hessians for a check of stability differ from those for the
            // steps is judged by a checked one (`ModeKillSettings::checked_hessian`).
            let judged = exact_now && (verifying || !mk.checked_hessian);
            if judged {
                // The one criterion of an imaginary mode (`opt::mode_kill`): the mass-weighted
                // Hessian with the masses of the atoms. In Cartesian coordinates that of the
                // optimizer's surface; in other coordinates without constraints the Cartesian
                // Hessian of the surface, the step then in Cartesian coordinates; where
                // constraints or a projection shape the space of other coordinates, the
                // Hessian in them with unit masses.
                let masses: Vec<f64> = self
                    .current
                    .masses_amu()
                    .iter()
                    .flat_map(|&m| [m, m, m])
                    .collect();
                let cartesian_total = (!self.coords.is_cartesian()
                    && self.constraints.is_none()
                    && projection.is_none())
                .then(|| surf.total.hessian_bohr.clone())
                .flatten();
                let symmetries = self
                    .settings
                    .rigid_body_invariant
                    .then(|| build_tr_rot_basis(&self.current, false));
                let (h, g, weights, basis, in_cartesian) = match &cartesian_total {
                    Some(hx) => (
                        hx.clone(),
                        biased_grad.clone(),
                        Some(masses.as_slice()),
                        symmetries.as_ref(),
                        true,
                    ),
                    None => {
                        let h = match &bias_hessian_q {
                            Some(b) => &step_hessian + b,
                            None => step_hessian.clone(),
                        };
                        let cartesian = self.coords.is_cartesian();
                        (
                            h,
                            g_q.clone(),
                            cartesian.then_some(masses.as_slice()),
                            if cartesian { rigid.as_ref() } else { None },
                            false,
                        )
                    }
                };
                let (verdict, retained) =
                    mode_kill::examine_retaining(&h, &g, order, &mk, weights, basis)?;
                match verdict {
                    ModeKillVerdict::Accept { negative } => {
                        if negative < order {
                            notes.push(format!(
                                "Mode kill: the converged structure has {negative} imaginary \
                                 mode(s), fewer than the order {order} asked for."
                            ));
                        }
                    }
                    ModeKillVerdict::Displace {
                        step,
                        negative,
                        killed,
                        left,
                    } if self.mode_kill_rounds < mk.max_rounds => {
                        let dx = if in_cartesian {
                            step
                        } else {
                            back_transform(
                                self.coords.as_ref(),
                                &self.current,
                                &step,
                                &BackTransformOptions::default(),
                            )?
                        };
                        let moved = self.placed_on_constraints(&(&x + &dx))?;
                        notes.push(format!(
                            "Mode kill: the converged structure has {negative} imaginary \
                             mode(s), order {order} was asked for; leaving it along {killed} \
                             mode(s) ({}) and going on.",
                            crate::opt::mode_kill::describe_left(&left)
                        ));
                        if self.hisd_mode_kill {
                            let retained = if in_cartesian {
                                retained
                                    .iter()
                                    .map(|d| frame.displacement_to_q(d))
                                    .collect()
                            } else {
                                retained
                            };
                            self.methods = vec![ResolvedMethod::mode_kill_hisd(&retained)?];
                            notes
                                .push(format!("Mode kill: continuing with HiSD at order {order}."));
                        }
                        after = AfterConvergence::Continue(Box::new(moved));
                    }
                    ModeKillVerdict::Displace { negative, .. } if self.hisd_mode_kill => {
                        return Err(MoptError::conv(format!(
                            "HiSD mode kill exhausted {} displacements with {negative} imaginary mode(s) remaining",
                            self.mode_kill_rounds
                        )));
                    }
                    ModeKillVerdict::Displace { negative, left, .. } => {
                        notes.push(format!(
                            "Mode kill: stopped after {} displacements; {negative} imaginary \
                             mode(s) remain ({}; order {order} was asked for).{}",
                            self.mode_kill_rounds,
                            crate::opt::mode_kill::describe_left(&left),
                            if mk.checked_hessian {
                                " The structure is not a stationary point of the order asked \
                                 for: the run ends without convergence."
                            } else {
                                ""
                            }
                        ));
                        if mk.checked_hessian {
                            after = AfterConvergence::WrongOrder;
                        }
                    }
                }
            } else if verifying {
                if self.hisd_mode_kill {
                    return Err(MoptError::conv(
                        "HiSD mode kill could not verify the final Hessian",
                    ));
                }
                notes.push(
                    "Mode kill: the surface returned no Hessian; the structure is accepted \
                     unchecked."
                        .to_string(),
                );
            } else {
                notes.push(
                    "Converged; the Hessian is requested here for the mode-kill check.".to_string(),
                );
                after = AfterConvergence::CheckHessian;
            }
        }

        // A converged search for a conical intersection: check its branching plane first.
        let mut probe_start = None;
        if let (true, AfterConvergence::Finish, Some(results)) =
            (convergence.converged, &after, for_crossing)
        {
            let ctx = TermContext::new(self.iteration, &self.initial, None);
            if let Some(crossing) =
                self.surface
                    .crossing_states(&self.current, &request, results, &ctx)?
            {
                if let [x, given] = &crossing.plane[..] {
                    let states = &crossing.states;
                    let mean = Col::from_fn(x.nrows(), |i| {
                        0.5 * (states.gradients[0][i] + states.gradients[1][i])
                    });
                    if let Some(direction) = self.probe_seed(x)? {
                        probe_start = Some((x.clone(), given.clone(), mean, direction));
                    }
                }
            }
        }

        self.last_eval = Some(LastEvaluation {
            raw_energy,
            raw_gradient: surf.raw.gradient_bohr.clone(),
            energy: biased_energy,
            gradient: biased_grad.clone(),
            iteration: self.iteration,
        });
        let coordinates = if self.coords.is_cartesian() {
            None
        } else {
            Some(CoordinateValues {
                system: self.coords.name().to_string(),
                labels: self.coords.labels(),
                values: self.coords.values(&self.current)?.iter().copied().collect(),
            })
        };
        let mut debug_notes: Vec<String> = self
            .methods
            .iter_mut()
            .flat_map(|m| m.generator.take_debug_notes())
            .collect();
        debug_notes.extend(update_debug);
        let report = IterationReport {
            iteration: self.iteration,
            system: self.current.clone(),
            energy: raw_energy,
            biased_energy,
            gradient: biased_grad.clone(),
            raw_gradient: surf.raw.gradient_bohr.clone(),
            components: surf.components,
            convergence,
            step_norm,
            trust_radius: self.trust.radius(),
            notes,
            debug_notes,
            program_outputs,
            coordinates,
            spin_squares,
            hessian_eigenvalues,
            crossing: crossing_report,
        };

        if convergence.converged {
            match after {
                AfterConvergence::Finish => match probe_start {
                    Some((x, given, mean_gradient, direction))
                        if self.iteration + 1 < self.settings.max_iterations =>
                    {
                        self.probe = Some(Probe {
                            report: report.clone(),
                            x,
                            mean_gradient,
                            given,
                            direction,
                            done: 0,
                        });
                    }
                    _ => self.termination = Some(Termination::Converged),
                },
                AfterConvergence::CheckHessian => {
                    self.verify_hessian = true;
                    self.previous = None;
                }
                // No variant of its own (the callers match the terminations exhaustively): the
                // run stopped short of a stationary point of the order asked for, as at the
                // limit of iterations; the note of the iteration says why.
                AfterConvergence::WrongOrder => {
                    self.termination = Some(Termination::MaxIterations);
                }
                AfterConvergence::Continue(moved) => {
                    self.current = *moved;
                    self.previous = None;
                    self.mode_kill_rounds += 1;
                    for m in &mut self.methods {
                        m.generator.reset();
                        if let Some(up) = m.updater.as_mut() {
                            up.reset();
                        }
                        for e in &mut m.enhancers {
                            e.reset();
                        }
                    }
                }
            }
            if self.termination.is_none() && self.iteration + 1 >= self.settings.max_iterations {
                self.termination = Some(Termination::MaxIterations);
            }
        } else if self.is_dissociated(&next) {
            self.termination = Some(Termination::Dissociated);
        } else {
            // A step that ends where the coordinates are no longer defined leaves no secant
            // pair; the next iteration sets the coordinates up anew.
            self.previous = match self.coords.displacement(&next, &self.current) {
                Ok(s_q) => Some(Previous {
                    system: self.current.clone(),
                    point: PreviousPoint {
                        energy: biased_energy,
                        gradient: g_q,
                        raw_gradient: raw_q,
                        step: s_q,
                    },
                    cartesian_step_norm: step_norm,
                    bias_hessian: bias_hessian_q,
                }),
                Err(_) if self.coords.needs_reinitialization(&next) => None,
                Err(e) => return Err(e),
            };
            self.current = next;
            if self.iteration + 1 >= self.settings.max_iterations {
                self.termination = Some(Termination::MaxIterations);
            }
        }
        self.iteration += 1;
        if self.termination.is_none() {
            match &self.probe {
                Some(probe) => self.pending = self.probe_tasks(&probe.direction)?,
                None => self.refresh_pending()?,
            }
        }
        let mut report = report;
        if self.termination.is_some() {
            // The constraints held only with a warning, once more at the end of the run.
            if let Some(summary) = self.constraints.as_ref().and_then(|c| c.lenient_summary()) {
                report.notes.push(summary);
            }
        }
        Ok(report)
    }

    /// Set the coordinate system up anew at the current structure. The Hessian is carried
    /// over through Cartesian coordinates (the initial one when the old coordinates cannot be
    /// evaluated here); what refers to the old coordinates (the previous point, the histories
    /// of the methods and of the Hessian update) is dropped.
    ///
    /// MultiOptPy: no counterpart (new in mopt-rs).
    fn reinitialize_coordinates(&mut self, notes: &mut Vec<String>) -> MoptResult<()> {
        let cartesian_hessian = CoordinateFrame::new(self.coords.as_ref(), &self.current)
            .ok()
            .map(|frame| frame.hessian_to_cartesian(&self.hessian));
        self.coords
            .initialize(&self.current)
            .context("coordinate system setup")?;
        let frame = CoordinateFrame::new(self.coords.as_ref(), &self.current)?;
        self.hessian = match cartesian_hessian {
            Some(h) => frame.hessian_to_q(&h),
            None => {
                let scale = self
                    .methods
                    .iter()
                    .find_map(|m| m.generator.initial_hessian_scale())
                    .unwrap_or(1.0);
                let n = self.coords.dim(&self.current);
                Mat::<f64>::identity(n, n) * scale
            }
        };
        self.previous = None;
        for m in &mut self.methods {
            m.generator.reset();
            if let Some(up) = m.updater.as_mut() {
                up.reset();
            }
            for e in &mut m.enhancers {
                e.reset();
            }
        }
        notes.push(format!(
            "The coordinates were set up anew at this structure: {}.",
            self.coords.name()
        ));
        Ok(())
    }

    /// The structure at the Cartesian coordinates `x` (after the step, its trust-radius cap
    /// and back-transformation) as the next iterate: superposed on the current one when the
    /// optimizer removes rigid-body motions, and then restored onto the constraints. The
    /// restore is the last thing that changes the structure, so that the constraints hold to
    /// their rounding floor in every structure the optimizer hands out (a superposition after
    /// it would leave them off by the rounding of the rotation; the constraints of an
    /// optimizer that superposes are invariant to it, and the Newton steps of the restore
    /// move no centroid and turn nothing to first order).
    ///
    /// MultiOptPy: no counterpart (new in mopt-rs; MultiOptPy projects after superposing).
    fn placed_on_constraints(&self, x: &Col<f64>) -> MoptResult<MolecularSystem> {
        let moved = self.current.with_flat_coords_bohr(x)?;
        let moved = if self.settings.rigid_body_invariant && !moved.has_pbc() {
            align_unweighted_centered(&moved, &self.current)?
        } else {
            moved
        };
        match &self.constraints {
            Some(c) => c.restore(&moved),
            None => Ok(moved),
        }
    }

    fn is_dissociated(&self, sys: &MolecularSystem) -> bool {
        let Some(limit) = self.settings.dissociation_distance.map(Quantity::value) else {
            return false;
        };
        let conn = BondConnectivity::from_system(sys, 1.2);
        if conn.fragments.len() < 2 {
            return false;
        }
        let mut min_dist = f64::INFINITY;
        for (i, fa) in conn.fragments.iter().enumerate() {
            for fb in &conn.fragments[i + 1..] {
                for &a in fa {
                    for &b in fb {
                        min_dist = min_dist.min(sys.distance(a, b).value());
                    }
                }
            }
        }
        min_dist > limit
    }
}

/// MultiOptPy's switching between two methods: the first above an RMS force of 0.05, the second
/// below 0.005, and a logistic blend in between.
pub(crate) fn blend(
    proposals: &[Col<f64>],
    gradient: &Col<f64>,
    methods: &[ResolvedMethod],
    notes: &mut Vec<String>,
) -> Col<f64> {
    if proposals.len() == 1 {
        return proposals[0].clone();
    }
    let (hi, lo, steepness, offset) = (0.05, 0.005, 10.0, 0.5);
    let rms = rms_all(gradient);
    if rms > hi {
        notes.push(format!("Switching to {}", methods[0].method));
        proposals[0].clone()
    } else if rms > lo {
        let xj = (rms - lo) / (hi - lo);
        let f = 1.0 / (1.0 + libm::exp(-steepness * (xj - offset)));
        notes.push(format!(
            "Weighted switching: {f:.3} {} {}",
            methods[0].method, methods[1].method
        ));
        &proposals[0] * f + &proposals[1] * (1.0 - f)
    } else {
        notes.push(format!("Switching to {}", methods[1].method));
        proposals[1].clone()
    }
}
