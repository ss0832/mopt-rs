//! Molecular dynamics in the generalized coordinates `q` of a non-redundant coordinate
//! system, as an ask/tell state machine like [`super::MolecularDynamics`].
//!
//! The Hamiltonian is `H(q, p) = 1/2 p^T G(q) p + V(q)` with the inverse kinetic metric
//! `G = B M^-1 B^T` of the Wilson matrix `B = dq/dx`, the exact kinetic energy of the atoms
//! in these coordinates. It is integrated with the generalized leapfrog (the Stormer-Verlet
//! method for a Hamiltonian that is not separable; Hairer, Lubich, and Wanner, Geometric
//! Numerical Integration, 2nd ed., Springer 2006, ch. VI.3, eq. 3.4), which is symplectic,
//! time-reversible, and of second order:
//!
//! ```text
//! p' = p_n - h/2 [dT/dq(q_n, p') + dV/dq(q_n)]            (implicit in p')
//! q_{n+1} = q_n + h/2 [G(q_n) + G(q_{n+1})] p'           (implicit in q_{n+1})
//! p_{n+1} = p' - h/2 [dT/dq(q_{n+1}, p') + dV/dq(q_{n+1})]
//! ```
//!
//! with one evaluation of the surface per step, at `x(q_{n+1})`. `dT/dq` at fixed `p` is
//! `(B B^T)^-1 B dT/dx` with `dT/dx = sum_l p_l (d^2 q_l / dx^2) w` (`w = M^-1 B^T p`, the
//! Cartesian velocity), the derivative of `B(x)^T p` along `w`, taken by central differences
//! of `B` (two evaluations of `B` per use; the second derivatives of the coordinates are
//! not needed). The structure of each `q` is found by the iterative back-transformation of
//! `coord::system::transform` from the one before, to 1e-12.
//!
//! Coordinates that cover all `3N` Cartesian motions (Cartesian, spherical coordinates of
//! the atoms) give the dynamics of the atoms in other variables. Internal coordinates that
//! cover exactly the `3N - 6` (`3N - 5`) internal motions (delocalized internal
//! coordinates) give the dynamics at zero linear and angular momentum: their metric is the
//! one of the velocities without rigid motion, so that this is the dynamics of
//! [`super::MolecularDynamics`] with the rigid motions taken out, on a surface that does
//! not depend on the position and orientation in space. Redundant coordinates have a
//! singular `G` and are refused. Thermostats act on `u = G^(1/2) p`, whose kinetic energy is
//! that of unit masses, so that every thermostat of the registry samples
//! `p ~ N(0, k_B T G^-1)`; the degrees of freedom are the coordinates (less the conserved
//! rigid motions of a full set).
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

mod metric;

use faer::Col;

use metric::Metric;

use super::report::{MdAction, MdEnd, MdRequest, MdStepReport, MdValue};
use super::settings::{InitialVelocities, MdSettings};
use super::space::{temperature_of, MdSpace, RigidMotion};
use super::thermostat::{Thermostat, ThermostatContext, ThermostatRegistry};
use super::MdMachine;
use crate::coord::system::transform::{back_transform, BackTransformOptions};
use crate::coord::system::CoordinateSystem;
use crate::core_infra::rng::Rng;
use crate::domain::{MolecularSystem, Units};
use crate::error::{MoptError, MoptResult, ResultExt};
use crate::math::linalg::col_norm;
use crate::math::projection::build_tr_rot_basis;

/// Cartesian displacement of the central differences of `B` (Bohr).
const DIFFERENCE_STEP: f64 = 1e-5;
/// Relative tolerance of the fixed-point iterations of a step.
const FIXED_POINT_TOLERANCE: f64 = 1e-12;
/// Most iterations of a fixed point.
const FIXED_POINT_ITERATIONS: usize = 100;
/// Largest residual of a back-transformation that is accepted (in the units of `q`).
const BACK_TRANSFORM_RESIDUAL: f64 = 1e-9;
/// `G` with a smallest-to-largest eigenvalue ratio below this is singular (redundant).
const REDUNDANCY_RATIO: f64 = 1e-10;

/// What the coordinates cover.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Span {
    /// All `3N` Cartesian motions.
    Full,
    /// Exactly the internal motions (`3N` less this many rigid motions).
    Internal(usize),
}

/// Ask/tell dynamics in generalized coordinates.
pub struct GeneralizedDynamics {
    settings: MdSettings,
    coords: Box<dyn CoordinateSystem>,
    template: MolecularSystem,
    inverse_masses: Vec<f64>,
    space: MdSpace,
    unit_space: MdSpace,
    thermostat: Box<dyn Thermostat>,
    span: Span,
    rigid: RigidMotion,
    rng: Rng,
    x: Col<f64>,
    p: Col<f64>,
    metric: Metric,
    force_q: Col<f64>,
    half: Col<f64>,
    energy: f64,
    raw_energy: f64,
    gradient: Col<f64>,
    raw_gradient: Col<f64>,
    step: usize,
    started: bool,
    finished: bool,
    dof: f64,
    work: f64,
    target: f64,
    evaluations: usize,
    iterations: (usize, usize),
    b_evaluations: usize,
    dipole: Option<[f64; 3]>,
}

impl GeneralizedDynamics {
    /// Dynamics of `start` in `coords` with the settings (thermostat, temperature, time step,
    /// steps, seed, velocities, conserved motions) of `settings`. `invariant`: whether the
    /// surface does not change under rigid motions (internal coordinates need it).
    pub fn new(
        settings: MdSettings,
        mut coords: Box<dyn CoordinateSystem>,
        start: MolecularSystem,
        invariant: bool,
    ) -> MoptResult<Self> {
        settings.validate()?;
        if settings.hopping.is_some() {
            return Err(MoptError::unsupported(
                "surface hopping changes the velocities along a direction of the mass-weighted \
                 Cartesian coordinates; it runs in Cartesian coordinates (-md_opt \
                 integrate=cartesian with -coord)",
            ));
        }
        refuse_unsupported(&settings, &start)?;
        coords.initialize(&start).context("the coordinates")?;
        let space = MdSpace::atoms(&start.masses_amu())?;
        let inverse_masses = space.inverse_masses().to_vec();
        let metric = Metric::at(coords.as_ref(), &start, &inverse_masses)?;
        let span = span_of(coords.as_ref(), &start, &metric)?;
        if matches!(span, Span::Internal(_)) && !invariant {
            return Err(MoptError::unsupported(
                "internal coordinates carry no rigid motion, and this surface depends on the \
                 position or orientation in space (bias potentials, an external field): the \
                 dynamics has to run in Cartesian coordinates",
            ));
        }
        let thermostat = ThermostatRegistry::builtin()
            .create(&settings.thermostat, &settings.thermostat_settings)?;
        let rigid = match span {
            Span::Internal(_) => RigidMotion::TranslationRotation,
            Span::Full if !invariant || !thermostat.conserves_momentum() => RigidMotion::None,
            Span::Full => settings.rigid,
        };
        let n = metric.dim();
        Ok(Self {
            rng: Rng::from_seed(settings.seed),
            target: settings.target_at(0),
            unit_space: MdSpace::variables(n)?,
            x: start.coords_flat_bohr(),
            p: Col::zeros(n),
            force_q: Col::zeros(n),
            half: Col::zeros(n),
            gradient: Col::zeros(3 * start.n_atoms()),
            raw_gradient: Col::zeros(3 * start.n_atoms()),
            settings,
            coords,
            template: start,
            inverse_masses,
            space,
            thermostat,
            span,
            rigid,
            metric,
            energy: f64::NAN,
            raw_energy: f64::NAN,
            step: 0,
            started: false,
            finished: false,
            dof: 0.0,
            work: 0.0,
            evaluations: 0,
            iterations: (0, 0),
            b_evaluations: 0,
            dipole: None,
        })
    }

    /// The settings.
    pub fn settings(&self) -> &MdSettings {
        &self.settings
    }

    /// What the coordinates cover.
    pub fn span(&self) -> Span {
        self.span
    }

    /// Number of coordinates.
    pub fn dim(&self) -> usize {
        self.metric.dim()
    }

    /// The conserved rigid motions.
    pub fn rigid(&self) -> RigidMotion {
        self.rigid
    }

    /// Name of the coordinate system.
    pub fn coordinates(&self) -> &str {
        self.coords.name()
    }

    /// Degrees of freedom of the temperature (known after the first evaluation).
    pub fn dof(&self) -> f64 {
        self.dof
    }

    /// Evaluations told so far.
    pub fn evaluations(&self) -> usize {
        self.evaluations
    }

    /// Evaluations of the Wilson matrix so far (the cost of the coordinates).
    pub fn wilson_evaluations(&self) -> usize {
        self.b_evaluations
    }

    /// Names of the integrator and the thermostat.
    pub fn part_names(&self) -> (String, String) {
        (
            "generalized_leapfrog".to_string(),
            self.thermostat.name().to_string(),
        )
    }

    /// The papers of the integrator and the thermostat.
    pub fn references(&self) -> Vec<&'static str> {
        let leapfrog = "E. Hairer, C. Lubich, G. Wanner, Geometric Numerical Integration, 2nd \
                        ed. (Springer, 2006), ch. VI.3 (the generalized leapfrog)";
        std::iter::once(leapfrog)
            .chain(self.thermostat.reference())
            .collect()
    }

    /// The generalized momenta.
    pub fn momenta(&self) -> &Col<f64> {
        &self.p
    }

    fn structure(&self, x: &Col<f64>) -> MoptResult<MolecularSystem> {
        self.template.with_flat_coords_bohr(x)
    }

    fn metric_at(&mut self, sys: &MolecularSystem) -> MoptResult<Metric> {
        self.b_evaluations += 1;
        Metric::at(self.coords.as_ref(), sys, &self.inverse_masses)
    }

    /// `dT/dq` at `x` (metric `m`) for the momenta `p`.
    fn kinetic_gradient(&mut self, x: &Col<f64>, m: &Metric, p: &Col<f64>) -> MoptResult<Col<f64>> {
        let w = m.cartesian_velocity(p, &self.inverse_masses);
        let norm = col_norm(&w);
        if norm == 0.0 {
            return Ok(Col::zeros(p.nrows()));
        }
        let e = DIFFERENCE_STEP / norm;
        let plus = self.structure(&(x + &w * e))?;
        let minus = self.structure(&(x - &w * e))?;
        self.b_evaluations += 2;
        let bp = self.coords.wilson_b(&plus)?;
        let bm = self.coords.wilson_b(&minus)?;
        let dx = (bp - bm).transpose() * p * (0.5 / e);
        Ok(m.gradient_to_q(&dx))
    }

    /// The thermostat for the time `h`, on `u = G^(1/2) p`.
    fn thermostat_stage(&mut self, h: f64) -> MoptResult<()> {
        let mut u = self.metric.root_times(&self.p);
        let before = 0.5 * u.iter().map(|v| v * v).sum::<f64>();
        let mut ctx = ThermostatContext {
            space: &self.unit_space,
            dof: self.dof,
            temperature: self.target,
            rng: &mut self.rng,
        };
        self.thermostat.apply(h, &mut u, &mut ctx)?;
        if self.thermostat.bath_energy(self.dof, self.target).is_none() {
            self.work += before - 0.5 * u.iter().map(|v| v * v).sum::<f64>();
        }
        self.p = self.metric.inverse_root_times(&u);
        Ok(())
    }

    /// `p' = p_n - h/2 [dT/dq(q_n, p') + dV/dq(q_n)]` by fixed-point iteration.
    fn implicit_kick(&mut self, h: f64) -> MoptResult<Col<f64>> {
        let (x, m, p0) = (self.x.clone(), self.metric.clone(), self.p.clone());
        let mut p = p0.clone();
        for k in 1..=FIXED_POINT_ITERATIONS {
            let t = self.kinetic_gradient(&x, &m, &p)?;
            let next = &p0 + (&self.force_q - &t) * (0.5 * h);
            let change = col_norm(&(&next - &p));
            p = next;
            if change <= FIXED_POINT_TOLERANCE * (1.0 + col_norm(&p)) {
                self.iterations.0 = k;
                return Ok(p);
            }
        }
        Err(MoptError::conv(
            "the implicit half kick of the generalized leapfrog does not converge; a smaller \
             time step is needed",
        ))
    }

    /// `q_{n+1} = q_n + h/2 [G(q_n) + G(q_{n+1})] p'`: the new structure and its metric.
    fn implicit_drift(&mut self, h: f64, p: &Col<f64>) -> MoptResult<(Col<f64>, Metric)> {
        let start = self.structure(&self.x)?;
        let a = self.metric.times(p);
        let mut dq = &a * h;
        let opts = BackTransformOptions {
            max_iter: 100,
            q_tol: 1e-12,
            x_tol: 1e-15,
        };
        for k in 1..=FIXED_POINT_ITERATIONS {
            let x1 = if self.coords.is_cartesian() {
                &self.x + &dq
            } else {
                &self.x + back_transform(self.coords.as_ref(), &start, &dq, &opts)?
            };
            let sys1 = self.structure(&x1)?;
            let achieved = self.coords.displacement(&sys1, &start)?;
            let residual = col_norm(&(&achieved - &dq));
            if residual > BACK_TRANSFORM_RESIDUAL {
                return Err(MoptError::conv(format!(
                    "the structure of the step is not found (residual {residual:.2e} in the \
                     coordinates); a smaller time step is needed"
                )));
            }
            let m1 = self.metric_at(&sys1)?;
            let next = (&a + m1.times(p)) * (0.5 * h);
            let change = col_norm(&(&next - &dq));
            if change <= FIXED_POINT_TOLERANCE * (1.0 + col_norm(&dq)) {
                self.iterations.1 = k;
                return Ok((x1, m1));
            }
            dq = next;
        }
        Err(MoptError::conv(
            "the implicit drift of the generalized leapfrog does not converge; a smaller time \
             step is needed",
        ))
    }

    fn start(&mut self) -> MoptResult<()> {
        let maxwell = matches!(self.settings.velocities, InitialVelocities::Maxwell { .. });
        let mut v = match &self.settings.velocities {
            InitialVelocities::Maxwell { .. } => self
                .space
                .maxwell_boltzmann(self.settings.temperature, &mut self.rng),
            InitialVelocities::Zero => Col::zeros(self.x.nrows()),
            InitialVelocities::Given(v) if v.nrows() == self.x.nrows() => v.clone(),
            InitialVelocities::Given(_) => {
                return Err(MoptError::config("the starting velocities do not match"))
            }
        };
        if maxwell && self.span == Span::Full {
            self.space
                .remove_rigid_motion(&self.x, &mut v, self.rigid)?;
        }
        // p = G^-1 B v: the momenta of the velocities (their rigid part drops out of
        // internal coordinates).
        self.p = self.metric.inverse_times(&(&self.metric.b * &v));
        self.dof = match self.span {
            Span::Internal(_) => self.metric.dim() as f64,
            Span::Full => {
                self.metric.dim() as f64 - self.space.rigid_count(&self.x, self.rigid) as f64
            }
        };
        if let InitialVelocities::Maxwell { exact: true } = self.settings.velocities {
            let kinetic = self.metric.kinetic(&self.p);
            let wanted =
                0.5 * self.dof * Units::BOLTZMANN_HARTREE_PER_K * self.settings.temperature;
            if kinetic > 0.0 {
                self.p = &self.p * libm::sqrt(wanted / kinetic);
            }
        }
        self.started = true;
        Ok(())
    }

    fn accept(&mut self, value: MdValue) -> MoptResult<()> {
        let n3 = self.x.nrows();
        if value.gradient.nrows() != n3 {
            return Err(MoptError::calc(format!(
                "the gradient has {} components for {n3} coordinates",
                value.gradient.nrows()
            )));
        }
        if !value.energy.is_finite() || value.gradient.iter().any(|g| !g.is_finite()) {
            return Err(MoptError::calc("the energy or the gradient is not finite"));
        }
        self.force_q = -self.metric.gradient_to_q(&value.gradient);
        self.dipole = value.dipole;
        self.energy = value.energy;
        self.raw_energy = value.raw_energy;
        self.raw_gradient = value.raw_gradient.unwrap_or_else(|| value.gradient.clone());
        self.gradient = value.gradient;
        self.evaluations += 1;
        Ok(())
    }

    fn receive(&mut self, value: MdValue) -> MoptResult<MdStepReport> {
        self.accept(value)?;
        let h = self.settings.time_step;
        if self.started {
            // p_{n+1} = p' - h/2 [dT/dq(q_{n+1}, p') + dV/dq(q_{n+1})].
            let (x, m, half) = (self.x.clone(), self.metric.clone(), self.half.clone());
            let t = self.kinetic_gradient(&x, &m, &half)?;
            self.p = &half + (&self.force_q - &t) * (0.5 * h);
            self.thermostat_stage(0.5 * h)?;
        } else {
            self.start()?;
        }
        let report = self.report();
        if self.step >= self.settings.steps {
            self.finished = true;
        } else {
            self.target = self.settings.target_at(self.step);
            self.thermostat_stage(0.5 * h)?;
            let half = self.implicit_kick(h)?;
            let (x1, m1) = self.implicit_drift(h, &half)?;
            self.half = half;
            self.x = x1;
            self.metric = m1;
            self.step += 1;
        }
        Ok(report)
    }

    fn report(&self) -> MdStepReport {
        let kinetic = self.metric.kinetic(&self.p);
        let bath = self
            .thermostat
            .bath_energy(self.dof, self.target)
            .unwrap_or(self.work);
        let mut notes = self.thermostat.notes();
        if self.step > 0 {
            notes.push(format!(
                "generalized leapfrog: {} and {} fixed-point iterations",
                self.iterations.0, self.iterations.1
            ));
        }
        MdStepReport {
            step: self.step,
            time: self.step as f64 * self.settings.time_step,
            x: self.x.clone(),
            v: self
                .metric
                .cartesian_velocity(&self.p, &self.inverse_masses),
            cell: None,
            energy: self.energy,
            raw_energy: self.raw_energy,
            gradient: self.gradient.clone(),
            raw_gradient: self.raw_gradient.clone(),
            kinetic,
            temperature: temperature_of(kinetic, self.dof),
            target_temperature: self.target,
            dof: self.dof,
            conserved: self.energy + kinetic + bath,
            pressure: None,
            volume: None,
            stress: None,
            constraint_residual: None,
            freeze: None,
            notes,
            dipole: self.dipole,
            heat_exchanged: 0.0,
            electronic_state: None,
            state_energies: Vec::new(),
            hops: Vec::new(),
        }
    }
}

impl MdMachine for GeneralizedDynamics {
    fn ask(&self) -> MdAction {
        if self.finished {
            return MdAction::Finished(MdEnd::Completed);
        }
        MdAction::Evaluate(MdRequest {
            x: self.x.clone(),
            cell: None,
            need_hessian: false,
            need_stress: false,
            step: self.step,
            time: self.step as f64 * self.settings.time_step,
            states: None,
        })
    }

    fn tell(&mut self, value: MdValue) -> MoptResult<MdStepReport> {
        if self.finished {
            return Err(MoptError::config("tell() after the dynamics ended"));
        }
        let step = self.step;
        self.receive(value).context(format!("MD step {step}"))
    }

    fn finished(&self) -> bool {
        self.finished
    }

    fn evaluations(&self) -> usize {
        self.evaluations
    }

    fn dof(&self) -> f64 {
        self.dof
    }
}

/// What dynamics in generalized coordinates does not take.
fn refuse_unsupported(settings: &MdSettings, start: &MolecularSystem) -> MoptResult<()> {
    let reason = if start.active_cell().is_some() {
        Some("a periodic cell")
    } else if settings.barostat.is_some() {
        Some("a barostat")
    } else if !settings.constraints.is_empty() {
        Some("constraints (SHAKE and RATTLE act on Cartesian coordinates)")
    } else if settings.freeze.is_some() {
        Some("frozen modes")
    } else if !matches!(
        settings.integrator.as_str(),
        "velocity_verlet" | "vv" | "verlet" | "velocityverlet"
    ) {
        Some("an integrator other than velocity Verlet (the generalized leapfrog takes its place)")
    } else {
        None
    };
    match reason {
        Some(r) => Err(MoptError::unsupported(format!(
            "dynamics in generalized coordinates does not take {r}; run it in Cartesian \
             coordinates (the default of mdmain without -md_opt integrate=coordinates)"
        ))),
        None => Ok(()),
    }
}

/// Whether `coords` at `sys` cover all motions or exactly the internal ones; an error for
/// redundant coordinates and for anything else.
fn span_of(coords: &dyn CoordinateSystem, sys: &MolecularSystem, m: &Metric) -> MoptResult<Span> {
    let n3 = 3 * sys.n_atoms();
    let n = m.dim();
    if m.conditioning() < REDUNDANCY_RATIO {
        return Err(MoptError::unsupported(format!(
            "the coordinates '{}' are redundant or singular here ({n} coordinates whose \
             metric G = B M^-1 B^T is singular): there are no momenta conjugate to them and \
             the dynamics cannot run in them; run it in Cartesian coordinates (the default of \
             mdmain without -md_opt integrate=coordinates)",
            coords.name()
        )));
    }
    if n == n3 {
        return Ok(Span::Full);
    }
    let rigid = build_tr_rot_basis(sys, false);
    let br = &m.b * &rigid;
    let size = |a: &faer::Mat<f64>| {
        libm::sqrt(
            (0..a.nrows())
                .flat_map(|i| (0..a.ncols()).map(move |j| (i, j)))
                .map(|(i, j)| a[(i, j)] * a[(i, j)])
                .sum::<f64>(),
        )
    };
    let leak = size(&br) / size(&m.b).max(1e-300);
    if n + rigid.ncols() == n3 && leak < 1e-6 {
        return Ok(Span::Internal(rigid.ncols()));
    }
    Err(MoptError::unsupported(format!(
        "the coordinates '{}' ({n} of them for {n3} Cartesian coordinates) cover neither all \
         motions nor exactly the internal ones: the dynamics cannot run in them",
        coords.name()
    )))
}
