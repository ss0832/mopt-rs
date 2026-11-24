//! The ask/tell state machine of molecular dynamics.
//!
//! [`MolecularDynamics::ask`] names the positions (and cell) whose energy, gradient, and,
//! when needed, Hessian or stress are wanted; [`MolecularDynamics::tell`] takes them, ends
//! the step with the stages of the integrator after the forces, reports it, and begins the
//! next step up to the positions of the next evaluation. The machine never calls a
//! calculator and draws its random numbers from one seeded generator, so that the same
//! settings and the same evaluations give the same run bit for bit (a restart replays the
//! evaluations of the journal).
//!
//! MultiOptPy: `moleculardynamics.py:MD.md` and `MD.exec_md`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::{Col, Mat};

use super::barostat::{Barostat, BarostatRegistry};
use super::constraints::Rattle;
use super::freeze::{FreezeUpdate, ModeFreezing};
use super::hopping::{HopRecord, StateValues, SurfaceHopping};
use super::integrator::{Integrator, IntegratorRegistry, Stage};
use super::report::{MdAction, MdEnd, MdRequest, MdStepReport, MdValue};
use super::settings::{InitialVelocities, MdSettings};
use super::space::{MdSpace, RigidMotion};
use super::thermostat::{Thermostat, ThermostatRegistry};
use crate::core_infra::rng::Rng;
use crate::domain::{MolecularSystem, PeriodicCell};
use crate::error::{MoptError, MoptResult, ResultExt};
use crate::math::projection::build_tr_rot_basis;
use crate::pes::constraint::{Constraint, ConstraintSet};
use crate::pes::stress::Stress;

mod hops;
mod setup;
mod stages;

use setup::{check_barostat, conserved_motion, split_constraints};

/// The parts of a run that can be replaced.
pub struct MdParts {
    /// The splitting of a step.
    pub integrator: Box<dyn Integrator>,
    /// The thermostat.
    pub thermostat: Box<dyn Thermostat>,
    /// The barostat (`None`: a fixed cell).
    pub barostat: Option<Box<dyn Barostat>>,
}

impl MdParts {
    /// The parts named in `settings`, from the built-in registries.
    pub fn from_settings(settings: &MdSettings) -> MoptResult<Self> {
        Ok(Self {
            integrator: IntegratorRegistry::builtin().create(&settings.integrator)?,
            thermostat: ThermostatRegistry::builtin()
                .create(&settings.thermostat, &settings.thermostat_settings)?,
            barostat: settings
                .barostat
                .as_deref()
                .map(|name| BarostatRegistry::builtin().create(name, &settings.barostat_settings))
                .transpose()?,
        })
    }
}

/// Ask/tell molecular dynamics.
pub struct MolecularDynamics {
    settings: MdSettings,
    space: MdSpace,
    template: Option<MolecularSystem>,
    parts: MdParts,
    before: Vec<Stage>,
    after: Vec<Stage>,
    rattle: Rattle,
    deferred: Vec<Constraint>,
    freezing: Option<ModeFreezing>,
    rng: Rng,
    x: Col<f64>,
    v: Col<f64>,
    cell: Option<PeriodicCell>,
    acceleration: Col<f64>,
    energy: f64,
    raw_energy: f64,
    gradient: Col<f64>,
    raw_gradient: Col<f64>,
    stress: Option<Stress>,
    step: usize,
    started: bool,
    finished: bool,
    dof: f64,
    work: f64,
    target: f64,
    rigid: RigidMotion,
    freeze_event: Option<FreezeUpdate>,
    evaluations: usize,
    dipole: Option<[f64; 3]>,
    exchanged: f64,
    hopping: Option<SurfaceHopping>,
    states: Option<StateValues>,
    hop_records: Vec<HopRecord>,
    state_energies: Vec<(usize, f64)>,
}

impl MolecularDynamics {
    /// Dynamics of the variables of `space` from `x0` with the parts named in `settings`;
    /// `template` gives the atoms (elements, charge, cell) when the variables are the
    /// coordinates of a structure.
    pub fn new(
        settings: MdSettings,
        space: MdSpace,
        x0: Col<f64>,
        template: Option<MolecularSystem>,
    ) -> MoptResult<Self> {
        let parts = MdParts::from_settings(&settings)?;
        Self::with_parts(settings, space, x0, template, parts)
    }

    /// [`Self::new`] with parts of one's own.
    pub fn with_parts(
        settings: MdSettings,
        mut space: MdSpace,
        x0: Col<f64>,
        template: Option<MolecularSystem>,
        parts: MdParts,
    ) -> MoptResult<Self> {
        settings.validate()?;
        super::integrator::check(parts.integrator.as_ref())?;
        if x0.nrows() != space.dimension() {
            return Err(MoptError::config(format!(
                "{} starting values for {} variables",
                x0.nrows(),
                space.dimension()
            )));
        }
        if let Some(t) = &template {
            if t.dof() != space.dimension() || !space.is_atomic() {
                return Err(MoptError::config(
                    "the structure does not match the variables of the dynamics",
                ));
            }
        }
        let (others, deferred) = split_constraints(&settings.constraints, &mut space)?;
        let start = template
            .as_ref()
            .map(|t| t.with_flat_coords_bohr(&x0))
            .transpose()?;
        let mut sets = Vec::new();
        if !others.is_empty() {
            let sys = start.as_ref().ok_or_else(|| {
                MoptError::config("constraints other than frozen modes need atoms")
            })?;
            sets.push(ConstraintSet::new(&others, sys, None).context("constraints")?);
        }
        let mut rattle = Rattle::new(sets, template.clone())?;
        rattle.set_tolerance(settings.shake_tolerance);
        if let crate::md::settings::ConstraintSolver::Lincs { order, iterations } =
            settings.constraint_solver
        {
            let bonds = start
                .as_ref()
                .and_then(|sys| super::lincs::Lincs::bonds_of(&others, sys))
                .filter(|b| !b.is_empty())
                .ok_or_else(|| {
                    MoptError::unsupported(
                        "LINCS holds bond lengths (distances of two atoms) only; other \
                         constraints need SHAKE and RATTLE (-md_opt constraint_solver=shake)",
                    )
                })?;
            rattle.use_lincs(super::lincs::Lincs::new(bonds, order, iterations)?);
        }
        let mut x = x0;
        rattle
            .restore(&mut x, space.inverse_masses())
            .context("constraints at the start")?;
        let cell = template.as_ref().and_then(|t| t.active_cell().cloned());
        check_barostat(&parts, cell.as_ref(), &rattle, &space, &settings)?;
        setup::check_exchange(&parts, cell.as_ref(), &rattle, &settings)?;
        let rigid = conserved_motion(&settings, &space, &rattle, &parts, cell.as_ref());
        let mut freezing = settings.freeze.clone().map(ModeFreezing::new).transpose()?;
        if let Some(f) = freezing.as_mut() {
            if f.start_from_vectors(space.inverse_masses())? {
                rattle.set_linear(f.rows(&x, space.inverse_masses()));
            }
        }
        let hopping = settings
            .hopping
            .clone()
            .map(|h| SurfaceHopping::new(h, settings.seed))
            .transpose()?;
        let n = space.dimension();
        let before = parts.integrator.before_forces();
        let after = parts.integrator.after_forces();
        Ok(Self {
            rng: Rng::from_seed(settings.seed),
            target: settings.target_at(0),
            settings,
            space,
            template,
            parts,
            before,
            after,
            rattle,
            deferred,
            freezing,
            x,
            v: Col::zeros(n),
            cell,
            acceleration: Col::zeros(n),
            energy: f64::NAN,
            raw_energy: f64::NAN,
            gradient: Col::zeros(n),
            raw_gradient: Col::zeros(n),
            stress: None,
            step: 0,
            started: false,
            finished: false,
            dof: 0.0,
            work: 0.0,
            rigid,
            freeze_event: None,
            evaluations: 0,
            dipole: None,
            exchanged: 0.0,
            hopping,
            states: None,
            hop_records: Vec::new(),
            state_energies: Vec::new(),
        })
    }

    /// The settings.
    pub fn settings(&self) -> &MdSettings {
        &self.settings
    }

    /// The variables.
    pub fn space(&self) -> &MdSpace {
        &self.space
    }

    /// Degrees of freedom of the temperature (known after the first evaluation).
    pub fn dof(&self) -> f64 {
        self.dof
    }

    /// The motions of the whole system taken as conserved.
    pub fn rigid(&self) -> RigidMotion {
        self.rigid
    }

    /// Number of scalar constraints held by RATTLE (frozen modes included).
    pub fn constraint_count(&self) -> usize {
        self.rattle.len()
    }

    /// Number of frozen modes.
    pub fn frozen_modes(&self) -> usize {
        self.freezing.as_ref().map_or(0, |f| f.modes().len())
    }

    /// Names of the integrator, thermostat, and barostat.
    pub fn part_names(&self) -> (String, String, Option<String>) {
        (
            self.parts.integrator.name().to_string(),
            self.parts.thermostat.name().to_string(),
            self.parts.barostat.as_ref().map(|b| b.name().to_string()),
        )
    }

    /// The papers of the integrator, thermostat, and barostat.
    pub fn references(&self) -> Vec<&'static str> {
        [
            self.parts.integrator.reference(),
            self.parts.thermostat.reference(),
            self.parts.barostat.as_ref().and_then(|b| b.reference()),
        ]
        .into_iter()
        .flatten()
        .collect()
    }

    /// Evaluations told so far.
    pub fn evaluations(&self) -> usize {
        self.evaluations
    }

    /// Whether the run has ended.
    pub fn finished(&self) -> bool {
        self.finished
    }

    /// Current positions (those of the next evaluation while the run goes on).
    pub fn positions(&self) -> &Col<f64> {
        &self.x
    }

    /// Current velocities.
    pub fn velocities(&self) -> &Col<f64> {
        &self.v
    }

    /// The next action.
    pub fn ask(&self) -> MdAction {
        if self.finished {
            return MdAction::Finished(MdEnd::Completed);
        }
        let need_hessian = (!self.started && !self.deferred.is_empty())
            || self
                .freezing
                .as_ref()
                .is_some_and(|f| f.needs_hessian(self.step));
        MdAction::Evaluate(MdRequest {
            x: self.x.clone(),
            cell: self.cell.clone(),
            need_hessian,
            need_stress: self.parts.barostat.is_some() || self.settings.stress,
            step: self.step,
            time: self.step as f64 * self.settings.time_step,
            states: self.hopping.as_ref().map(SurfaceHopping::request),
        })
    }

    /// Surface hopping, when the run has it.
    pub fn hopping(&self) -> Option<&SurfaceHopping> {
        self.hopping.as_ref()
    }

    /// The surface at the positions of [`Self::ask`]; returns the report of the step that
    /// it completes.
    pub fn tell(&mut self, value: MdValue) -> MoptResult<MdStepReport> {
        if self.finished {
            return Err(MoptError::config("tell() after the dynamics ended"));
        }
        let step = self.step;
        self.receive(value).context(format!("MD step {step}"))
    }

    fn receive(&mut self, value: MdValue) -> MoptResult<MdStepReport> {
        let hessian = self.accept(value)?;
        if let Some(h) = hessian {
            self.use_hessian(&h)?;
        } else if let MdAction::Evaluate(r) = self.ask() {
            if r.need_hessian {
                return Err(MoptError::calc("the Hessian was asked for and not told"));
            }
        }
        if self.started {
            for stage in self.after.clone() {
                self.execute(stage)?;
            }
            if let Some(e) = self.settings.exchange {
                if self.step.is_multiple_of(e.every.max(1)) {
                    self.exchange(e)?;
                }
            }
        } else {
            self.start()?;
        }
        self.consider_hop()?;
        let report = self.report()?;
        if self.step >= self.settings.steps {
            self.finished = true;
        } else {
            self.target = self.settings.target_at(self.step);
            for stage in self.before.clone() {
                self.execute(stage)?;
            }
            self.step += 1;
        }
        Ok(report)
    }

    fn accept(&mut self, value: MdValue) -> MoptResult<Option<Mat<f64>>> {
        let n = self.space.dimension();
        let finite = |c: &Col<f64>| c.iter().all(|v| v.is_finite());
        if value.gradient.nrows() != n
            || value.raw_gradient.as_ref().is_some_and(|g| g.nrows() != n)
        {
            return Err(MoptError::calc(format!(
                "the gradient has {} components for {n} variables",
                value.gradient.nrows()
            )));
        }
        if !value.energy.is_finite() || !finite(&value.gradient) {
            return Err(MoptError::calc("the energy or the gradient is not finite"));
        }
        if self.parts.barostat.is_some() && value.stress.is_none() {
            return Err(MoptError::calc(
                "the barostat needs the stress, and none was told",
            ));
        }
        let w = self.space.inverse_masses();
        self.acceleration = Col::from_fn(n, |i| -value.gradient[i] * w[i]);
        self.energy = value.energy;
        self.raw_energy = value.raw_energy;
        self.raw_gradient = value.raw_gradient.unwrap_or_else(|| value.gradient.clone());
        self.gradient = value.gradient;
        self.stress = value.stress;
        self.dipole = value.dipole;
        if let Some(h) = &self.hopping {
            let states = value.states.ok_or_else(|| {
                MoptError::calc("surface hopping asked for the electronic states; none were told")
            })?;
            h.check(&states, n)?;
            self.state_energies = states
                .states
                .iter()
                .copied()
                .zip(states.energies.iter().copied())
                .collect();
            self.states = Some(states);
        }
        self.evaluations += 1;
        Ok(value.hessian)
    }

    fn use_hessian(&mut self, hessian: &Mat<f64>) -> MoptResult<()> {
        let n = self.space.dimension();
        if hessian.nrows() != n || hessian.ncols() != n {
            return Err(MoptError::calc(format!(
                "the Hessian is {}x{} for {n} variables",
                hessian.nrows(),
                hessian.ncols()
            )));
        }
        if !self.started && !self.deferred.is_empty() {
            let sys = self.structure()?;
            let set = ConstraintSet::new(&self.deferred, &sys, Some(hessian))
                .context("eigenvector constraints")?;
            self.rattle.add_set(set)?;
            self.deferred.clear();
        }
        let step = self.step;
        let rigid = self.rigid_basis()?;
        let w = self.space.inverse_masses().to_vec();
        if let Some(f) = self.freezing.as_mut().filter(|f| f.needs_hessian(step)) {
            f.update(step, hessian, &w, rigid.as_ref())
                .context("frozen modes")?;
            self.freeze_event = f.history().last().cloned();
            self.rattle.set_linear(f.rows(&self.x, &w));
            if self.started {
                self.rattle.project_velocities(&self.x, &mut self.v, &w)?;
            }
        }
        Ok(())
    }

    /// The structure at the current positions.
    fn structure(&self) -> MoptResult<MolecularSystem> {
        let t = self
            .template
            .as_ref()
            .ok_or_else(|| MoptError::config("this needs a structure of atoms"))?;
        let mut sys = t.with_flat_coords_bohr(&self.x)?;
        if let Some(cell) = &self.cell {
            sys.cell = Some(cell.clone());
        }
        Ok(sys)
    }

    /// Mass-weighted rigid-body modes for the analysis of the Hessian (none when atoms are
    /// held or the surface depends on the position in space).
    fn rigid_basis(&self) -> MoptResult<Option<Mat<f64>>> {
        if self.settings.rigid == RigidMotion::None
            || self.space.has_held()
            || self.template.is_none()
        {
            return Ok(None);
        }
        // The masses of the space are those of the elements; the basis uses the same ones.
        let sys = self.structure()?;
        let basis = build_tr_rot_basis(&sys, true);
        if self.settings.rigid == RigidMotion::Translation && sys.active_cell().is_none() {
            // The translations come first.
            let k = basis.ncols().min(3);
            return Ok(Some(Mat::from_fn(basis.nrows(), k, |i, j| basis[(i, j)])));
        }
        Ok(Some(basis))
    }

    fn start(&mut self) -> MoptResult<()> {
        let w = self.space.inverse_masses().to_vec();
        // The conserved motions are not counted in the degrees of freedom: drawn or given
        // velocities lose their part along them, so that the count and the velocities agree.
        let drawn_or_given = matches!(
            self.settings.velocities,
            InitialVelocities::Maxwell { .. } | InitialVelocities::Given(_)
        );
        self.v = match &self.settings.velocities {
            InitialVelocities::Maxwell { .. } => self
                .space
                .maxwell_boltzmann(self.settings.temperature, &mut self.rng),
            InitialVelocities::Zero => Col::zeros(self.space.dimension()),
            InitialVelocities::Given(v) => {
                if v.nrows() != self.space.dimension() {
                    return Err(MoptError::config("the starting velocities do not match"));
                }
                Col::from_fn(v.nrows(), |i| if w[i] > 0.0 { v[i] } else { 0.0 })
            }
        };
        if drawn_or_given {
            self.space
                .remove_rigid_motion(&self.x, &mut self.v, self.rigid)?;
        }
        self.rattle.project_velocities(&self.x, &mut self.v, &w)?;
        let rank = self.rattle.rank(&self.x, &w)?;
        let rigid = self.space.rigid_count(&self.x, self.rigid);
        self.dof = self.space.mobile() as f64 - rank as f64 - rigid as f64;
        if self.dof < 1.0 && self.parts.thermostat.name() != "none" {
            return Err(MoptError::config(format!(
                "no degrees of freedom are left for the thermostat ({} variables move, {rank} \
                 constraints, {rigid} conserved motions)",
                self.space.mobile()
            )));
        }
        if let InitialVelocities::Maxwell { exact: true } = self.settings.velocities {
            let kinetic = self.space.kinetic_energy(&self.v);
            let wanted = 0.5
                * self.dof.max(0.0)
                * crate::domain::Units::BOLTZMANN_HARTREE_PER_K
                * self.settings.temperature;
            if kinetic > 0.0 {
                let s = libm::sqrt(wanted / kinetic);
                for x in self.v.iter_mut() {
                    *x *= s;
                }
            }
        }
        self.started = true;
        Ok(())
    }
}
