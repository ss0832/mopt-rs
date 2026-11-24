//! Microiterations: relaxation of "environment" variables at fixed outer coordinates.
//!
//! Several surfaces are defined by an inner minimization: ONIOM microiterations relax the
//! low-level environment while the model region is held fixed, and some bias potentials carry
//! auxiliary variables that are minimized at every evaluation (the rotation angles of the
//! asymmetric ellipsoidal potentials, the solvent particles of the spacer model). All of them
//! use this module:
//!
//! - [`relax`] minimizes a [`MicroProblem`] with a damped Newton method (quasi-Newton when the
//!   problem has no Hessian), followed by **mode kill**: negative-curvature modes left at the
//!   end are followed downhill and the environment is relaxed again, so the inner loop ends at a
//!   minimum, not at a saddle point of the cheap surface.
//! - The minimizer is one of [`InnerMethod`]: the limited-memory BFGS (the default), a
//!   truncated Newton method on Hessian-vector products, or a method of the optimizer
//!   registry (`-opt` names) with a dense Hessian. The methods of the registry, and FIRE where
//!   a line search of the limited-memory BFGS fails, are run by the optimizers, a higher layer
//!   (`docs/source/architecture.md`): the code that builds the settings of a relaxation hands
//!   them in ([`MicroSettings::methods`], `crate::opt::micro::REGISTRY_METHODS`).
//! - [`relax_in`] runs the limited-memory BFGS in other coordinates than the Cartesian ones
//!   ([`InnerCoordinates`], for example internal coordinates of each fragment of an ONIOM
//!   environment).
//! - [`effective_hessian`] gives the Hessian of the relaxed surface
//!   `E*(x) = min_y E(x, y)` in the outer variables, `H_xx - H_xy H_yy^+ H_yx`; its gradient is
//!   `dE/dx` at the relaxed `y` (the inner gradient vanishes there).
//!
//! MultiOptPy: the microiterations of `optimization.py:ONIOMHandler.compute` and of
//!  `Potential/asym_elllipsoidal_potential.py` / `spacer_model_potential.py`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

mod internal;
mod memory;
mod newton_cg;
mod solver;

use faer::{Col, Mat};

use crate::error::{MoptError, MoptResult};
use crate::math::linalg::{symmetric_eigen, symmetrize};

pub use memory::MicroMemory;
pub use solver::{relax, relax_in, relax_with};

/// Energy and derivatives of the environment at one point.
#[derive(Debug, Clone)]
pub struct MicroEval {
    /// Energy (Hartree).
    pub energy: f64,
    /// Gradient with respect to the environment variables.
    pub gradient: Col<f64>,
    /// Hessian with respect to the environment variables, when requested and available.
    pub hessian: Option<Mat<f64>>,
}

/// A minimization over environment variables `y` at fixed outer coordinates.
pub trait MicroProblem {
    /// Number of environment variables.
    fn dim(&self) -> usize;

    /// Energy, gradient, and (when `need_hessian` and [`Self::has_hessian`]) Hessian at `y`.
    fn evaluate(&self, y: &[f64], need_hessian: bool) -> MoptResult<MicroEval>;

    /// Whether [`Self::evaluate`] returns Hessians; otherwise the solver updates its own and
    /// mode kill uses finite differences of the gradient.
    fn has_hessian(&self) -> bool {
        true
    }

    /// `P^-1 v` with a preconditioner of the problem (an approximate inverse Hessian that is
    /// positive definite), or `None` without one: the initial matrix of the limited-memory
    /// BFGS and the correction of the search for the lowest curvature.
    fn precondition(&self, _v: &Col<f64>) -> Option<Col<f64>> {
        None
    }

    /// Told what the relaxation does while it runs (every accepted step, the end of a
    /// descent, the check of the curvature); a problem that reports its relaxations writes
    /// what it wants of it.
    fn progress(&self, _event: &MicroProgress) {}
}

/// What a relaxation tells while it runs ([`MicroProblem::progress`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MicroProgress {
    /// An accepted step.
    Iteration {
        /// Inner iteration of the descent (from 1).
        iteration: usize,
        /// Energy (Hartree).
        energy: f64,
        /// Largest gradient component.
        max_gradient: f64,
        /// Root mean square of the gradient.
        rms_gradient: f64,
    },
    /// A line search found no step that lowers the energy after `iteration` inner
    /// iterations; the relaxation goes on with `method`: the limited-memory BFGS with a line
    /// search on the gradient alone, then [`FALLBACK_METHOD`].
    Fallback {
        /// Inner iterations so far.
        iteration: usize,
        /// What the relaxation goes on with.
        method: &'static str,
    },
    /// The method of the fallback stopped with an error after `iteration` inner iterations;
    /// the relaxation ends where it was before it, short of its thresholds.
    FallbackFailed {
        /// Inner iterations before the fallback.
        iteration: usize,
        /// The method.
        method: &'static str,
        /// Why it stopped.
        reason: &'static str,
    },
    /// A descent ended.
    Ended {
        /// Its inner iterations.
        iterations: usize,
        /// Largest gradient component at its end.
        max_gradient: f64,
        /// Whether the thresholds were met.
        converged: bool,
    },
    /// The lowest curvature at the end of a descent (the test for a saddle point).
    Curvature {
        /// The lowest curvature (Hartree/Bohr^2).
        lowest: f64,
        /// Whether it counts as negative (the mode is followed downhill then).
        negative: bool,
        /// Hessian-vector products the check took (two gradients each).
        products: usize,
        /// Norm of the residual of the mode (Hartree/Bohr^2).
        residual: f64,
        /// Whether the residual met the tolerance of the search.
        converged: bool,
    },
    /// The descents from both sides of a negative-curvature mode ended.
    Left {
        /// Whether one of them reached a lower energy (otherwise both relaxed back).
        lower: bool,
    },
}

/// Largest component and root mean square of the gradient `g`.
pub(crate) fn gradient_norms(g: &Col<f64>) -> (f64, f64) {
    let n = g.nrows();
    if n == 0 {
        return (0.0, 0.0);
    }
    (
        crate::math::linalg::col_max_abs(g),
        (g.iter().map(|v| v * v).sum::<f64>() / n as f64).sqrt(),
    )
}

/// Tell `problem` an accepted step.
pub(crate) fn tell_step(problem: &dyn MicroProblem, iteration: usize, energy: f64, g: &Col<f64>) {
    let (max_gradient, rms_gradient) = gradient_norms(g);
    problem.progress(&MicroProgress::Iteration {
        iteration,
        energy,
        max_gradient,
        rms_gradient,
    });
}

/// Mode kill after the inner relaxation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModeKill {
    /// Whether to remove negative-curvature modes.
    pub enabled: bool,
    /// Eigenvalues below `-threshold` count as negative curvature.
    pub threshold: f64,
    /// Displacement along the mode before relaxing again.
    pub displacement: f64,
    /// Maximum number of killed modes per relaxation.
    pub max_kills: usize,
    /// The search for the lowest curvature starts from the mode of the last check and
    /// takes the inverse Hessian of the limited-memory BFGS of the relaxation as its
    /// preconditioner (`micro_curvature=guided`); `false`: from a fixed start without one
    /// (`micro_curvature=plain`).
    pub guided: bool,
}

impl Default for ModeKill {
    fn default() -> Self {
        Self {
            enabled: true,
            threshold: 1e-6,
            displacement: 0.1,
            max_kills: 10,
            guided: true,
        }
    }
}

/// Coordinates other than the Cartesian ones in which a relaxation takes its steps: values
/// `q(y)`, the gradient in `q`, and steps in `q` taken back to `y`. The limited-memory BFGS
/// of [`relax_in`] runs in them; the convergence test stays on the gradient in `y`.
pub trait InnerCoordinates {
    /// Number of coordinates.
    fn dim(&self) -> usize;

    /// Set up the frame of the transformations at `y` (before [`Self::gradient_to_q`]).
    fn prepare(&mut self, y: &[f64]) -> MoptResult<()>;

    /// The gradient in `q` of the gradient `g` in `y`, at the prepared point.
    fn gradient_to_q(&self, g: &Col<f64>) -> Col<f64>;

    /// The point `y` of the step `dq` from `y` (iterative back-transformation).
    fn step(&self, y: &[f64], dq: &Col<f64>) -> MoptResult<Vec<f64>>;

    /// `q(to) - q(from)`.
    fn displacement(&self, to: &[f64], from: &[f64]) -> MoptResult<Col<f64>>;

    /// Set the coordinates up anew at `y` where they are about to break down there; `true`
    /// when they were (the generation then changes).
    fn renew_if_needed(&mut self, y: &[f64]) -> MoptResult<bool>;

    /// Changes each time the coordinates are set up anew, so that limited-memory BFGS pairs
    /// of older coordinates are not used.
    fn generation(&self) -> usize;
}

/// The minimizer of a relaxation.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum InnerMethod {
    /// Limited-memory BFGS (damped Newton when the problem has Hessians); the pairs can be
    /// kept from one relaxation to the next ([`relax_with`]).
    #[default]
    Lbfgs,
    /// Truncated Newton: each step solves `H p = -g` by preconditioned conjugate gradients on
    /// Hessian-vector products from differences of the gradient, to a relative residual that
    /// shrinks with the gradient (Dembo, Eisenstat, Steihaug, SIAM J. Numer. Anal. 1982, 19,
    /// 400; Nash, J. Comput. Appl. Math. 2000, 124, 45). For large environments whose
    /// Hessian is badly conditioned.
    TruncatedNewton,
    /// A method of the optimizer registry by its `-opt` name (RS-I-RFO, FIRE, ...), with a
    /// dense Hessian of the environment (small environments only).
    Method(String),
}

impl InnerMethod {
    /// The method of a name: `lbfgs`, `tn` (or `truncated_newton`, `newton_cg`), else a name
    /// of the optimizer registry.
    pub fn parse(text: &str) -> Self {
        match text.trim().to_ascii_lowercase().replace('-', "_").as_str() {
            "lbfgs" | "l_bfgs" => Self::Lbfgs,
            "tn" | "truncated_newton" | "newton_cg" => Self::TruncatedNewton,
            _ => Self::Method(text.trim().to_string()),
        }
    }

    /// The name, as [`Self::parse`] reads it.
    pub fn name(&self) -> &str {
        match self {
            Self::Lbfgs => "lbfgs",
            Self::TruncatedNewton => "tn",
            Self::Method(m) => m,
        }
    }
}

/// A relaxation by a method of the optimizer registry (its `-opt` name) from `y`, without
/// mode kill. The methods are those of the optimizers, a layer above the surfaces, so this
/// layer takes the function from the code that builds the settings ([`MicroSettings::methods`]).
pub type MethodRelaxationFn =
    fn(&dyn MicroProblem, Vec<f64>, &MicroSettings, &str) -> MoptResult<MicroResult>;

/// The relaxations by the methods of the optimizer registry, handed in with the settings.
#[derive(Clone, Copy)]
pub struct MethodRelaxation(pub MethodRelaxationFn);

impl std::fmt::Debug for MethodRelaxation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("MethodRelaxation")
    }
}

impl PartialEq for MethodRelaxation {
    /// Settings are the same whatever code runs the methods: the code is one service, not a
    /// setting (and the addresses of functions are not comparable).
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}

/// Settings of [`relax`].
#[derive(Debug, Clone, PartialEq)]
pub struct MicroSettings {
    /// Maximum inner iterations per relaxation.
    pub max_iterations: usize,
    /// Convergence threshold on the largest gradient component.
    pub gradient_tolerance: f64,
    /// Convergence threshold on the root mean square of the gradient (infinite: only the
    /// largest component decides).
    pub rms_tolerance: f64,
    /// Largest step (Euclidean norm).
    pub max_step: f64,
    /// Mode kill.
    pub mode_kill: ModeKill,
    /// Pairs of the limited-memory BFGS of a problem without Hessian.
    pub memory: usize,
    /// The minimizer.
    pub method: InnerMethod,
    /// The methods of the optimizer registry: [`InnerMethod::Method`], and FIRE after a line
    /// search of the limited-memory BFGS found no step (`None`: the relaxation then ends where
    /// it is, and a named method is refused).
    pub methods: Option<MethodRelaxation>,
}

impl Default for MicroSettings {
    fn default() -> Self {
        Self {
            max_iterations: 500,
            gradient_tolerance: 1e-10,
            rms_tolerance: f64::INFINITY,
            max_step: 0.5,
            mode_kill: ModeKill::default(),
            memory: 30,
            method: InnerMethod::Lbfgs,
            methods: None,
        }
    }
}

/// Outcome of [`relax`].
#[derive(Debug, Clone)]
pub struct MicroResult {
    /// Relaxed environment variables.
    pub y: Vec<f64>,
    /// Energy at `y`.
    pub energy: f64,
    /// Largest gradient component at `y`.
    pub max_gradient: f64,
    /// Inner iterations used (all relaxations together).
    pub iterations: usize,
    /// Negative-curvature modes removed by mode kill.
    pub killed_modes: usize,
    /// Whether the gradient threshold was met.
    pub converged: bool,
    /// Whether the relaxation stopped because the energy no longer decreased (a floor of the
    /// surface above the threshold: a kink of the energy, or a gradient at the precision of
    /// the calculator).
    pub stalled: bool,
    /// Whether a line search found no step that lowers the energy (the caller goes on with
    /// [`FALLBACK_METHOD`]).
    pub line_search_failed: bool,
    /// Times the relaxation went on with [`FALLBACK_METHOD`] after a line search failed.
    pub fallbacks: usize,
}

/// Accepted steps after which a relaxation that lowered neither its energy (beyond the
/// round-off) nor its largest gradient component (to 0.9 of the lowest so far) stops short of
/// the threshold.
pub const STALL_STEPS: usize = 100;

/// Shortest accepted step of a line search, as a share of the proposed one: a shorter one
/// lowers the energy by an amount at the round-off and counts as a failure.
pub const SMALLEST_STEP: f64 = 1e-8;

/// The method a relaxation goes on with after a line search failed (the FIRE of the
/// optimizer registry, which moves by the forces alone and compares no energies). A line
/// search fails where the surface does not follow its gradient: at a discontinuity (a
/// neighbour list of a force field that changes, a kink) the gradient says downhill and no
/// step along it lowers the energy; a quasi-Newton method with a line search then stops or
/// lowers the energy by amounts at the round-off without end.
pub const FALLBACK_METHOD: &str = "fire";

/// Evaluations of the fallback after which a relaxation whose largest gradient component
/// did not fall to 0.9 of the lowest so far stops short of the thresholds.
pub const FALLBACK_STALL_STEPS: usize = 5000;

/// Whether the gradient `g` meets the thresholds of `s` (largest component and root mean
/// square).
pub(crate) fn is_converged(g: &Col<f64>, s: &MicroSettings) -> bool {
    if g.nrows() == 0 {
        return true;
    }
    let (max, rms) = gradient_norms(g);
    max < s.gradient_tolerance && rms < s.rms_tolerance
}

/// The stall test of the minimizers: the lowest energy and the lowest largest gradient
/// component so far, and the steps since either improved. Near a tight threshold the energy
/// changes by less than its round-off while the gradient still falls, so both are watched.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Stall {
    energy: f64,
    gradient: f64,
    since: usize,
}

impl Stall {
    pub(crate) fn new(energy: f64, gradient: f64) -> Self {
        Self {
            energy,
            gradient,
            since: 0,
        }
    }

    /// Record an accepted step; `true` when the relaxation has stalled.
    pub(crate) fn stalled_after(&mut self, energy: f64, gradient: f64) -> bool {
        let lower = energy < self.energy - 1e-14 * self.energy.abs().max(1.0);
        let flatter = gradient < 0.9 * self.gradient;
        if lower {
            self.energy = energy;
        }
        if flatter {
            self.gradient = gradient;
        }
        if lower || flatter {
            self.since = 0;
        } else {
            self.since += 1;
        }
        self.since >= STALL_STEPS
    }
}

/// Hessian of `E*(x) = min_y E(x, y)` with respect to `x` from the full Hessian over `(x, y)`
/// (the first `n_outer` rows and columns are `x`): `H_xx - H_xy H_yy^+ H_yx`.
///
/// Flat environment directions (eigenvalues of `H_yy` below `1e-10` of the largest in
/// magnitude, e.g. variables the energy does not depend on) are left out of the inverse.
pub fn effective_hessian(full: &Mat<f64>, n_outer: usize) -> MoptResult<Mat<f64>> {
    let n = full.nrows();
    if full.ncols() != n || n_outer > n {
        return Err(MoptError::linalg(format!(
            "effective_hessian: {}x{} matrix with {n_outer} outer variables",
            full.nrows(),
            full.ncols()
        )));
    }
    let m = n - n_outer;
    let hxx = Mat::from_fn(n_outer, n_outer, |i, j| full[(i, j)]);
    if m == 0 {
        return Ok(hxx);
    }
    let hxy = Mat::from_fn(n_outer, m, |i, j| full[(i, n_outer + j)]);
    let hyy = Mat::from_fn(m, m, |i, j| full[(n_outer + i, n_outer + j)]);
    let (w, v) = symmetric_eigen(&hyy)?;
    let wmax = w.iter().fold(0.0_f64, |a, x| a.max(x.abs()));
    let cut = 1e-10 * wmax;
    let mut inv = Mat::<f64>::zeros(m, m);
    for k in (0..m).filter(|&k| w[k].abs() > cut) {
        for i in 0..m {
            for j in 0..m {
                inv[(i, j)] += v[(i, k)] * v[(j, k)] / w[k];
            }
        }
    }
    let mut h = hxx - &hxy * &inv * hxy.transpose();
    symmetrize(&mut h);
    Ok(h)
}
