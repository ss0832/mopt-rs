//! Ask/tell optimizer over a plain vector of `n` variables.
//!
//! The same step generators, Hessian updaters, step enhancers, outer trust radius, method
//! blending, and convergence test as [`super::state_machine::GeometryOptimizer`], without what
//! belongs to atoms (coordinate systems, rigid-body motion, bias potentials, dissociation).
//! Equality constraints of the variables ([`super::vector_constraints`]) are held exactly
//! when given ([`VectorOptimizer::set_constraints`]). The objective is any function `x in R^n -> (E, dE/dx, d2E/dx2)`
//! ([`crate::pes::objective::Objective`]), for example the two-variable Muller-Brown surface;
//! `n` need not be a multiple of three. Thresholds and trust radii are in the units of the
//! variables and of the objective (they are compared with the same numbers as for molecules).
//!
//! MultiOptPy: no counterpart (new in mopt-rs; MultiOptPy optimizes atomic coordinates only).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::sync::Arc;

use faer::{Col, Mat};

use super::convergence::{ConvergenceCriteria, ConvergenceReport};
use super::mode_kill::{self, ModeKillSettings, ModeKillVerdict};
use super::registry::{MethodContext, MethodRegistry, ResolvedMethod};
use super::state_machine::{blend, Termination};
use super::step::ccqn::CcqnSettings;
use super::step::{ensure_finite_step, PreviousPoint, StepInput};
use super::trust::{OuterTrustRadius, TrustSettings, TrustUpdateInput};
use super::update::{apply_update, project_out_rigid};
use super::vector_constraints::VectorConstraintSet;
use crate::domain::Quantity;
use crate::error::{MoptError, MoptResult, ResultExt};
use crate::math::linalg::col_norm;

/// Settings of a vector optimization (the subset of `optmain`'s that applies).
#[derive(Debug, Clone)]
pub struct VectorSettings {
    /// One method, or two blended by the RMS gradient (as `-opt`). Empty: the method
    /// for the saddle order ([`super::registry::default_methods`]).
    pub methods: Vec<String>,
    /// Number of negative-curvature directions (0 = minimization).
    pub saddle_order: usize,
    /// Maximum number of iterations.
    pub max_iterations: usize,
    /// Convergence thresholds.
    pub criteria: ConvergenceCriteria,
    /// Trust radius.
    pub trust: TrustSettings,
    /// Request the exact Hessian every this many iterations.
    pub exact_hessian_interval: Option<usize>,
    /// CCQN axis and parameters (`ccqn*` methods; the axis must be a direction of length `n`).
    pub ccqn: Option<CcqnSettings>,
    /// At convergence, leave a stationary point with more negative curvatures than
    /// `saddle_order` (see [`crate::opt::mode_kill`]); off by default.
    pub mode_kill: Option<ModeKillSettings>,
    /// Take the rigid motions of [`VectorOptimizer::set_rigid_modes`] out of the kept Hessian
    /// after every update (`-update_proj`; on by default). An exact Hessian is projected when
    /// it is given either way.
    pub project_updates: bool,
}

impl Default for VectorSettings {
    fn default() -> Self {
        Self {
            methods: Vec::new(),
            saddle_order: 0,
            max_iterations: 1000,
            criteria: ConvergenceCriteria::NORMAL,
            trust: TrustSettings::default(),
            exact_hessian_interval: None,
            ccqn: None,
            mode_kill: None,
            project_updates: true,
        }
    }
}

/// Next action of a [`VectorOptimizer`].
#[derive(Debug, Clone)]
pub enum VectorAction {
    /// Evaluate the objective at `x` (with the Hessian when `need_hessian`).
    Evaluate {
        /// Point.
        x: Col<f64>,
        /// Whether the Hessian is requested.
        need_hessian: bool,
    },
    /// The optimization has terminated.
    Finished(Termination),
}

/// Summary of one iteration.
#[derive(Debug, Clone)]
pub struct VectorReport {
    /// Iteration index.
    pub iteration: usize,
    /// Point that was evaluated.
    pub x: Col<f64>,
    /// Objective value.
    pub energy: f64,
    /// Gradient.
    pub gradient: Col<f64>,
    /// Convergence test.
    pub convergence: ConvergenceReport,
    /// Norm of the step taken.
    pub step_norm: f64,
    /// Outer trust radius.
    pub trust_radius: f64,
    /// Notes from method resolution and blending.
    pub notes: Vec<String>,
}

struct Previous {
    point: PreviousPoint,
    step_norm: f64,
}

/// Ask/tell optimizer of `n` variables.
pub struct VectorOptimizer {
    settings: VectorSettings,
    methods: Vec<ResolvedMethod>,
    x: Col<f64>,
    iteration: usize,
    hessian: Mat<f64>,
    /// Scale of the unit matrix a Hessian starts from (that of the methods, 1 without one).
    initial_scale: f64,
    previous: Option<Previous>,
    trust: OuterTrustRadius,
    termination: Option<Termination>,
    pending_notes: Vec<String>,
    last_hessian_update: Option<String>,
    last: Option<(f64, Col<f64>)>,
    /// The Hessian is requested at the current point for the mode-kill check.
    verify_hessian: bool,
    /// Displacements made by the mode kill so far.
    mode_kill_rounds: usize,
    hisd_mode_kill: bool,
    mode_kill_masses: Option<Vec<f64>>,
    /// Equality constraints held exactly (none by default).
    constraints: Option<VectorConstraintSet>,
    /// The rigid motions of the variables at a point (none by default: plain variables have
    /// none).
    rigid_modes: Option<RigidModes>,
}

/// Orthonormal vectors along which the objective does not change at a point: the rigid motions
/// of variables that describe a structure (the atoms and the cell of a variable-cell
/// optimization). See [`VectorOptimizer::set_rigid_modes`].
pub type RigidModes = Box<dyn Fn(&Col<f64>) -> MoptResult<Vec<Col<f64>>> + Send + Sync>;

impl VectorOptimizer {
    /// Start at `x0` with the default method registry.
    pub fn new(settings: VectorSettings, x0: Col<f64>) -> MoptResult<Self> {
        Self::with_registry(settings, x0, &MethodRegistry::default())
    }

    /// Start at `x0`, resolving method names in `methods`.
    pub fn with_registry(
        settings: VectorSettings,
        x0: Col<f64>,
        methods: &MethodRegistry,
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
        if x0.nrows() == 0 {
            return Err(MoptError::config("no variables to optimize"));
        }
        if x0.iter().any(|v| !v.is_finite()) {
            return Err(MoptError::config("the starting point is not finite"));
        }
        let ctx = MethodContext {
            saddle_order: settings.saddle_order,
            trust: settings.trust,
            ccqn: settings.ccqn.clone().map(Arc::new),
        };
        let resolved: Vec<ResolvedMethod> = settings
            .methods
            .iter()
            .map(|m| methods.resolve(m, &ctx))
            .collect::<MoptResult<_>>()?;
        let pending_notes = resolved.iter().flat_map(|m| m.warnings.clone()).collect();
        let scale = resolved
            .iter()
            .find_map(|m| m.generator.initial_hessian_scale())
            .unwrap_or(1.0);
        let n = x0.nrows();
        let uses_hessian = resolved.iter().any(|m| m.generator.uses_hessian());
        // A Hessian is kept where something reads it: a method that steps with it, an
        // updater, an enhancer, mode kill, exact Hessians. Methods that move by the forces
        // alone (FIRE, the gradient methods) on many variables then need no `n x n` matrix.
        let keeps_hessian = uses_hessian
            || resolved
                .iter()
                .any(|m| m.updater.is_some() || !m.enhancers.is_empty())
            || settings.mode_kill.is_some()
            || settings.exact_hessian_interval.is_some();
        let hessian = if keeps_hessian {
            Mat::<f64>::identity(n, n) * scale
        } else {
            Mat::<f64>::zeros(0, 0)
        };
        let adaptive = uses_hessian && settings.exact_hessian_interval.is_some();
        let trust = OuterTrustRadius::new(settings.trust, settings.saddle_order, adaptive);
        Ok(Self {
            settings,
            methods: resolved,
            x: x0,
            iteration: 0,
            hessian,
            initial_scale: scale,
            previous: None,
            trust,
            termination: None,
            pending_notes,
            last_hessian_update: None,
            last: None,
            verify_hessian: false,
            mode_kill_rounds: 0,
            hisd_mode_kill: false,
            mode_kill_masses: None,
            constraints: None,
            rigid_modes: None,
        })
    }

    /// Take the rigid motions `modes` out of the Hessian that is kept, after every update and
    /// when an exact Hessian is given (as the optimizer of structures does with the
    /// translations and rotations): `modes` gives them at a point as orthonormal vectors.
    /// Without this call nothing is taken out, which is right for plain variables.
    pub fn set_rigid_modes(&mut self, modes: RigidModes) {
        self.rigid_modes = Some(modes);
    }

    /// `P H P` on the kept Hessian with the rigid motions at the current point.
    fn project_out_rigid_modes(&mut self) -> MoptResult<()> {
        let Some(modes) = &self.rigid_modes else {
            return Ok(());
        };
        let modes = modes(&self.x)?;
        let n = self.x.nrows();
        if modes.iter().any(|m| m.nrows() != n) {
            return Err(MoptError::config(format!(
                "a rigid motion does not have {n} components"
            )));
        }
        let basis = Mat::from_fn(n, modes.len(), |i, j| modes[j][i]);
        project_out_rigid(&mut self.hessian, Some(&basis));
        Ok(())
    }

    /// Follow `directions` (one per mode of the saddle order) from the start instead of the
    /// lowest modes of the first Hessian, as [`super::state_machine::GeometryOptimizer::
    /// follow_directions`] does; the method must follow modes (`mf_rsirfo_*`).
    pub fn follow_directions(&mut self, directions: &[Col<f64>]) -> MoptResult<()> {
        let n = self.x.nrows();
        if let Some(d) = directions.iter().find(|d| d.nrows() != n) {
            return Err(MoptError::config(format!(
                "a direction to follow has {} components for {n} variables",
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
                "a direction to follow needs a method that follows modes (mf_rsirfo_*)",
            ))
        }
    }

    /// Hold `constraints` exactly from now on (before the first iteration): the starting
    /// point is moved onto them, and every step stays on them.
    pub fn set_constraints(&mut self, constraints: VectorConstraintSet) -> MoptResult<()> {
        if self.iteration > 0 {
            return Err(MoptError::config(
                "constraints are set before the first iteration",
            ));
        }
        self.x = constraints
            .restore(&self.x)
            .map_err(|e| e.context("constraints at the starting point"))?;
        self.constraints = (!constraints.is_empty()).then_some(constraints);
        // The constraints project the Hessian of a step: one is kept from here on.
        let n = self.x.nrows();
        if self.constraints.is_some() && self.hessian.nrows() != n {
            self.hessian = Mat::<f64>::identity(n, n);
        }
        Ok(())
    }

    /// Number of variables.
    pub fn dimension(&self) -> usize {
        self.x.nrows()
    }

    /// Current point (the last evaluated one after termination).
    pub fn current(&self) -> &Col<f64> {
        &self.x
    }

    /// Iterations performed so far.
    pub fn iterations(&self) -> usize {
        self.iteration
    }

    /// Termination reason, once finished.
    pub fn termination(&self) -> Option<Termination> {
        self.termination
    }

    /// Select the post-displacement method for an explicit standalone mode kill.
    pub(crate) fn configure_mode_kill(&mut self, downhill: bool, masses: Option<Vec<f64>>) {
        self.mode_kill_masses = masses;
        self.hisd_mode_kill = downhill && self.settings.mode_kill.is_some();
    }

    /// Scale of the unit matrix the Hessian of the methods starts from (also when none is
    /// kept: a caller that sets a Hessian of its own builds on it).
    pub fn initial_hessian_scale(&self) -> f64 {
        self.initial_scale
    }

    /// Whether a Hessian is kept (not for methods that move by the forces alone, without
    /// updater, enhancer, mode kill, exact Hessians, or constraints).
    pub fn keeps_hessian(&self) -> bool {
        self.hessian.nrows() == self.x.nrows()
    }

    /// Current (model, exact, or updated) Hessian; empty when none is kept
    /// ([`Self::keeps_hessian`]).
    pub fn hessian(&self) -> &Mat<f64> {
        &self.hessian
    }

    pub(crate) fn updated_hessian(&self) -> Option<(&str, &Mat<f64>)> {
        self.last_hessian_update
            .as_deref()
            .map(|method| (method, &self.hessian))
    }

    /// Replace the current Hessian by `hessian` (a model of the objective's curvature to
    /// start from, in place of the scaled unit matrix).
    pub fn set_hessian(&mut self, hessian: Mat<f64>) -> MoptResult<()> {
        let n = self.dimension();
        if hessian.nrows() != n || hessian.ncols() != n {
            return Err(MoptError::config(format!(
                "Hessian is {}x{}, expected {n}x{n}",
                hessian.nrows(),
                hessian.ncols()
            )));
        }
        if (0..n).any(|j| (0..n).any(|i| !hessian[(i, j)].is_finite())) {
            return Err(MoptError::config("the Hessian is not finite"));
        }
        self.hessian = hessian;
        Ok(())
    }

    /// Objective value and gradient of the last evaluated point.
    pub fn last_evaluation(&self) -> Option<&(f64, Col<f64>)> {
        self.last.as_ref()
    }

    fn need_hessian(&self) -> bool {
        self.verify_hessian
            || self
                .settings
                .exact_hessian_interval
                .is_some_and(|n| n > 0 && self.iteration.is_multiple_of(n))
    }

    /// Next action.
    pub fn ask(&self) -> VectorAction {
        match self.termination {
            Some(t) => VectorAction::Finished(t),
            None => VectorAction::Evaluate {
                x: self.x.clone(),
                need_hessian: self.need_hessian(),
            },
        }
    }

    /// Consume the objective at the point of [`Self::ask`].
    pub fn tell(
        &mut self,
        energy: f64,
        gradient: Col<f64>,
        hessian: Option<Mat<f64>>,
    ) -> MoptResult<VectorReport> {
        if self.termination.is_some() {
            return Err(MoptError::config(
                "tell() called after the optimization finished",
            ));
        }
        let n = self.dimension();
        if gradient.nrows() != n {
            return Err(MoptError::config(format!(
                "gradient has {} components, expected {n}",
                gradient.nrows()
            )));
        }
        if let Some(h) = &hessian {
            if h.nrows() != n || h.ncols() != n {
                return Err(MoptError::config(format!(
                    "Hessian is {}x{}, expected {n}x{n}",
                    h.nrows(),
                    h.ncols()
                )));
            }
        }
        if !energy.is_finite() || gradient.iter().any(|v| !v.is_finite()) {
            return Err(MoptError::calc("the objective is not finite at this point"));
        }
        let it = self.iteration;
        self.step(energy, gradient, hessian)
            .map_err(|e| e.context(format!("optimizer iteration {it}")))
    }

    fn step(
        &mut self,
        energy: f64,
        gradient: Col<f64>,
        hessian: Option<Mat<f64>>,
    ) -> MoptResult<VectorReport> {
        let mut notes = std::mem::take(&mut self.pending_notes);
        self.last_hessian_update = None;
        let e = Quantity::new(energy);
        let exact = hessian.is_some();
        let verifying = std::mem::take(&mut self.verify_hessian);
        if let Some(h) = hessian {
            self.hessian = h;
            self.project_out_rigid_modes()?;
        }
        // With constraints: the gradient without the constraint forces moves the point, is
        // learned from, and decides convergence; the objective's gradient is reported.
        let frame = self
            .constraints
            .as_ref()
            .map(|c| c.frame(&self.x))
            .transpose()?;
        let gradient = match &frame {
            Some(f) => f.project(&gradient),
            None => gradient,
        };
        let trust_input = self.previous.as_ref().map(|p| TrustUpdateInput {
            energy: e,
            prev_energy: p.point.energy,
            prev_gradient: &p.point.gradient,
            prev_step: &p.point.step,
            prev_cartesian_step_norm: Quantity::new(p.step_norm),
            hessian: &self.hessian,
            decrease: None,
        });
        self.trust.update(trust_input);
        let mut learned = false;
        if let Some(prev) = &self.previous {
            let y = &gradient - &prev.point.raw_gradient;
            let learns = self
                .methods
                .iter()
                .any(|m| m.generator.learns_negative_curvature());
            let saddle = self.settings.saddle_order > 0;
            if let Some(up) = self.methods.iter_mut().find_map(|m| m.updater.as_mut()) {
                if exact {
                    up.reset();
                }
                let negative = learns || (saddle && up.learns_saddle_curvature());
                if apply_update(
                    up.as_mut(),
                    &mut self.hessian,
                    &prev.point.step,
                    &y,
                    negative,
                )? {
                    self.last_hessian_update = Some(up.name().to_owned());
                }
                learned = true;
            }
        }
        // An update puts curvature along the rigid motions of the newer point; a Hessian that
        // was not updated belongs to the point before and is brought to this one as well.
        if learned && self.settings.project_updates {
            self.project_out_rigid_modes()?;
        }
        let step_hessian = match &frame {
            Some(f) => f.step_hessian(&self.hessian),
            None => self.hessian.clone(),
        };
        let input = StepInput {
            iteration: self.iteration,
            cartesian: &self.x,
            energy: e,
            gradient: &gradient,
            raw_gradient: &gradient,
            previous: self.previous.as_ref().map(|p| &p.point),
            hessian: &step_hessian,
            bias_hessian: None,
            rigid_body_basis: None,
            project_updates: self.settings.project_updates,
            saddle_order: self.settings.saddle_order,
            system: None,
            cartesian_coordinates: true,
            frame: None,
            cartesian_gradient: Some(&gradient),
        };
        let mut proposals = Vec::with_capacity(self.methods.len());
        for m in &mut self.methods {
            let mut dq = m.generator.propose(&input).context(m.method.clone())?;
            for enh in &mut m.enhancers {
                dq = enh.enhance(&input, dq).context(enh.name().to_string())?;
            }
            proposals.push(dq);
        }
        let mut dq = blend(&proposals, &gradient, &self.methods, &mut notes);
        if let Some(f) = &frame {
            dq = f.project(&dq);
        }
        let self_limited = self
            .methods
            .iter()
            .all(|m| m.generator.controls_step_length());
        let dx = if self_limited && self.settings.trust.max.is_none() {
            dq
        } else {
            self.trust.cap(dq)
        };
        ensure_finite_step(&dx, &self.settings.methods)?;
        // The next point, back on the constraints; the step is what it moves.
        let (next, dx) = match &self.constraints {
            Some(c) => {
                let next = c.restore(&(&self.x + &dx))?;
                let moved = &next - &self.x;
                (next, moved)
            }
            None => (&self.x + &dx, dx),
        };
        let step_norm = col_norm(&dx);
        let convergence = self.settings.criteria.check(&gradient, &dx);
        // Mode kill: a converged point must not have more negative curvatures than asked.
        let mut check_hessian = false;
        let mut leave: Option<Col<f64>> = None;
        if let Some(mk) = self.settings.mode_kill.filter(|_| convergence.converged) {
            let order = self.settings.saddle_order;
            if exact {
                // Plain variables use unit masses; a variable cell supplies its atom/strain masses.
                let (verdict, retained) = mode_kill::examine_retaining(
                    &step_hessian,
                    &gradient,
                    order,
                    &mk,
                    self.mode_kill_masses.as_deref(),
                    None,
                )?;
                match verdict {
                    ModeKillVerdict::Accept { negative } => {
                        if negative < order {
                            notes.push(format!(
                                "Mode kill: the converged point has {negative} imaginary \
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
                        notes.push(format!(
                            "Mode kill: the converged point has {negative} imaginary mode(s), \
                             order {order} was asked for; leaving it along {killed} mode(s) \
                             ({}) and going on.",
                            crate::opt::mode_kill::describe_left(&left)
                        ));
                        if self.hisd_mode_kill {
                            self.methods = vec![ResolvedMethod::mode_kill_hisd(&retained)?];
                            notes
                                .push(format!("Mode kill: continuing with HiSD at order {order}."));
                        }
                        leave = Some(step);
                    }
                    ModeKillVerdict::Displace { negative, .. } if self.hisd_mode_kill => {
                        return Err(MoptError::conv(format!(
                            "HiSD mode kill exhausted {} displacements with {negative} imaginary mode(s) remaining",
                            self.mode_kill_rounds
                        )));
                    }
                    ModeKillVerdict::Displace { negative, .. } => notes.push(format!(
                        "Mode kill: stopped after {} displacements; {negative} imaginary \
                         mode(s) remain (order {order} was asked for).",
                        self.mode_kill_rounds
                    )),
                }
            } else if verifying {
                if self.hisd_mode_kill {
                    return Err(MoptError::conv(
                        "HiSD mode kill could not verify the final Hessian",
                    ));
                }
                notes.push(
                    "Mode kill: the objective returned no Hessian; the point is accepted \
                     unchecked."
                        .to_string(),
                );
            } else {
                notes.push(
                    "Converged; the Hessian is requested here for the mode-kill check.".to_string(),
                );
                check_hessian = true;
            }
        }
        self.last = Some((energy, gradient.clone()));
        let report = VectorReport {
            iteration: self.iteration,
            x: self.x.clone(),
            energy,
            gradient: gradient.clone(),
            convergence,
            step_norm,
            trust_radius: self.trust.radius().value(),
            notes,
        };
        if convergence.converged {
            if let Some(step) = leave {
                self.x = match &self.constraints {
                    Some(c) => c.restore(&(&self.x + &step))?,
                    None => &self.x + &step,
                };
                self.previous = None;
                self.mode_kill_rounds += 1;
                for m in &mut self.methods {
                    m.generator.reset();
                    if let Some(up) = m.updater.as_mut() {
                        up.reset();
                    }
                    for enh in &mut m.enhancers {
                        enh.reset();
                    }
                }
            } else if check_hessian {
                self.verify_hessian = true;
                self.previous = None;
            } else {
                self.termination = Some(Termination::Converged);
            }
            if self.termination.is_none() && self.iteration + 1 >= self.settings.max_iterations {
                self.termination = Some(Termination::MaxIterations);
            }
        } else {
            self.previous = Some(Previous {
                point: PreviousPoint {
                    energy: e,
                    gradient: gradient.clone(),
                    raw_gradient: gradient,
                    step: dx.clone(),
                },
                step_norm,
            });
            self.x = next;
            if self.iteration + 1 >= self.settings.max_iterations {
                self.termination = Some(Termination::MaxIterations);
            }
        }
        self.iteration += 1;
        Ok(report)
    }
}
