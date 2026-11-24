//! Settings of a molecular dynamics run.
//!
//! MultiOptPy: the attributes of `moleculardynamics.py:MD` and `MD/thermostat.py:Thermostat`
//! (`timestep`, `NSTEP`, `initial_temperature`, `change_temperature`, `mdtype`,
//! `initial_pressure`).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Col;

use super::barostat::{gigapascal_per_atomic_unit, BarostatSettings};
use super::constraints::DEFAULT_TOLERANCE;
use super::freeze::FreezeSettings;
use super::space::RigidMotion;
use super::thermostat::ThermostatSettings;
use crate::domain::Units;
use crate::error::{MoptError, MoptResult};
use crate::pes::constraint::Constraint;

/// Time step of atoms without `-ts` (fs; measured, see `docs/source/molecular_dynamics.md`).
pub const DEFAULT_TIME_STEP_FS: f64 = 0.5;
/// Time step of plain variables (unit masses) without `-ts`, in the time unit of the
/// variables (MultiOptPy's default value; measured on the Muller-Brown surface).
pub const DEFAULT_VARIABLE_TIME_STEP: f64 = 0.1;

/// How the velocities start.
#[derive(Debug, Clone, PartialEq)]
pub enum InitialVelocities {
    /// Drawn from the Maxwell-Boltzmann distribution at the starting temperature; with
    /// `exact`, scaled to that temperature (after the conserved motions and the
    /// constraints are taken out).
    Maxwell {
        /// Scale to the starting temperature exactly.
        exact: bool,
    },
    /// All zero.
    Zero,
    /// Given (Bohr per atomic time unit).
    Given(Col<f64>),
}

/// Settings of a run.
#[derive(Debug, Clone)]
pub struct MdSettings {
    /// Time step (atomic units).
    pub time_step: f64,
    /// Number of steps.
    pub steps: usize,
    /// Starting and target temperature (K).
    pub temperature: f64,
    /// Changes of the target temperature: from step `n` on, `T` (K).
    pub schedule: Vec<(usize, f64)>,
    /// Integrator name ([`super::integrator::IntegratorRegistry`]).
    pub integrator: String,
    /// Thermostat name ([`super::thermostat::ThermostatRegistry`]).
    pub thermostat: String,
    /// Settings of the thermostat.
    pub thermostat_settings: ThermostatSettings,
    /// Barostat name ([`super::barostat::BarostatRegistry`]); `None`: a fixed cell.
    pub barostat: Option<String>,
    /// Settings of the barostat.
    pub barostat_settings: BarostatSettings,
    /// Target pressure (Hartree / Bohr^3).
    pub pressure: f64,
    /// Seed of the random numbers (initial velocities, stochastic thermostats).
    pub seed: u64,
    /// How the velocities start.
    pub velocities: InitialVelocities,
    /// Motions of the whole system that are conserved (taken out of the starting
    /// velocities and of the degrees of freedom).
    pub rigid: RigidMotion,
    /// Constraints (`-fix` atoms are held by zero inverse masses, the others by RATTLE).
    pub constraints: Vec<Constraint>,
    /// Mode freezing.
    pub freeze: Option<FreezeSettings>,
    /// Largest residual of the constraints after SHAKE.
    pub shake_tolerance: f64,
    /// Ask for the stress at every step (for the pressure tensor of the shear viscosity),
    /// not only for a barostat.
    pub stress: bool,
    /// The velocity exchanges of a reverse non-equilibrium run (thermal conductivity).
    pub exchange: Option<Exchange>,
    /// The solver of the constraints.
    pub constraint_solver: ConstraintSolver,
    /// Trajectory surface hopping among electronic states (`None`: one surface).
    pub hopping: Option<super::hopping::HoppingSettings>,
}

/// The solver of holonomic constraints.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstraintSolver {
    /// SHAKE and RATTLE (any constraint).
    Shake,
    /// LINCS (bond lengths only; Hess et al., J. Comput. Chem. 1997, 18, 1463): the order
    /// of the expansion and the number of corrections for the rotation of the bonds.
    Lincs {
        /// Order of the expansion of the inverse (4).
        order: usize,
        /// Corrections for the rotation of the bonds (1).
        iterations: usize,
    },
}

/// The velocity exchanges of Muller-Plathe's reverse non-equilibrium method (J. Chem.
/// Phys. 1997, 106, 6082): every `every` steps the atom of the cold slab (the first of
/// `slabs` along `axis`) with the largest kinetic energy and the atom of the hot slab (the
/// middle one) with the smallest, of the same element, exchange their velocities.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Exchange {
    /// Steps between exchanges.
    pub every: usize,
    /// Number of slabs (even).
    pub slabs: usize,
    /// Axis of the slabs (0, 1, 2 for a, b, c).
    pub axis: usize,
}

/// Target pressure of `-press` without a value: MultiOptPy's 101.3 kPa.
pub const DEFAULT_PRESSURE_KPA: f64 = 101.3;

/// `kPa` in atomic units of pressure.
pub fn kilopascal_to_atomic(kpa: f64) -> f64 {
    kpa * 1e-6 / gigapascal_per_atomic_unit()
}

impl Default for MdSettings {
    fn default() -> Self {
        Self {
            time_step: DEFAULT_TIME_STEP_FS * Units::FS_TO_AU_TIME,
            steps: 1000,
            temperature: 298.15,
            schedule: Vec::new(),
            integrator: "velocity_verlet".to_string(),
            thermostat: "none".to_string(),
            thermostat_settings: ThermostatSettings::default(),
            barostat: None,
            barostat_settings: BarostatSettings::default(),
            pressure: kilopascal_to_atomic(DEFAULT_PRESSURE_KPA),
            seed: 1,
            velocities: InitialVelocities::Maxwell { exact: true },
            rigid: RigidMotion::None,
            constraints: Vec::new(),
            freeze: None,
            shake_tolerance: DEFAULT_TOLERANCE,
            stress: false,
            exchange: None,
            constraint_solver: ConstraintSolver::Shake,
            hopping: None,
        }
    }
}

impl MdSettings {
    /// Check the values.
    pub fn validate(&self) -> MoptResult<()> {
        if !(self.time_step.is_finite() && self.time_step > 0.0) {
            return Err(MoptError::config(format!(
                "the time step must be positive, got {}",
                self.time_step
            )));
        }
        if !(self.temperature.is_finite() && self.temperature >= 0.0) {
            return Err(MoptError::config(format!(
                "the temperature must be zero or positive, got {}",
                self.temperature
            )));
        }
        if let Some((step, t)) = self
            .schedule
            .iter()
            .find(|(_, t)| !(t.is_finite() && *t >= 0.0))
        {
            return Err(MoptError::config(format!(
                "temperature change at step {step}: {t} K"
            )));
        }
        if !self.pressure.is_finite() {
            return Err(MoptError::config("the pressure must be finite"));
        }
        if !(self.shake_tolerance.is_finite() && self.shake_tolerance > 0.0) {
            return Err(MoptError::config("the SHAKE tolerance must be positive"));
        }
        if let Some(h) = &self.hopping {
            h.validate()?;
        }
        Ok(())
    }

    /// The target temperature at `step`.
    pub fn target_at(&self, step: usize) -> f64 {
        self.schedule
            .iter()
            .filter(|(from, _)| *from <= step)
            .max_by_key(|(from, _)| *from)
            .map_or(self.temperature, |(_, t)| *t)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn the_schedule_changes_the_target_from_its_step_on() {
        let s = MdSettings {
            temperature: 300.0,
            schedule: vec![(1000, 500.0), (5000, 1000.0)],
            ..MdSettings::default()
        };
        assert_eq!(s.target_at(0), 300.0);
        assert_eq!(s.target_at(999), 300.0);
        assert_eq!(s.target_at(1000), 500.0);
        assert_eq!(s.target_at(7000), 1000.0);
        assert!(s.validate().is_ok());
        // 101.3 kPa is 3.443e-9 atomic units.
        assert!((kilopascal_to_atomic(101.3) - 3.443e-9).abs() < 1e-11);
        let bad = MdSettings {
            time_step: 0.0,
            ..MdSettings::default()
        };
        assert!(bad.validate().is_err());
    }
}
