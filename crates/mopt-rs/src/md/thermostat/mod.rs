//! Thermostats (extension point).
//!
//! A [`Thermostat`] acts on the velocities for a share of the time step, at the places the
//! integrator gives it (both ends of a velocity Verlet step, the middle of a BAOAB step).
//! Built in: none (NVE), Berendsen's weak coupling, the canonical velocity rescaling of
//! Bussi, Donadio, and Parrinello, Langevin dynamics, Andersen's collisions, and Nose-Hoover
//! chains (a single Nose-Hoover thermostat is a chain of one). Thermostats that have
//! variables of their own report their energy; for the others the driver accounts the work
//! they did on the velocities, so that every run has an effective energy that the
//! integration must conserve.
//!
//! MultiOptPy: `MD/thermostat.py:Thermostat` (`Nose_Hoover_thermostat`,
//! `Nose_Hoover_chain_thermostat`, `Langevin_thermostat`).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

pub mod langevin;
pub mod nose_hoover;
pub mod rescale;

use faer::Col;

pub use langevin::{Andersen, Langevin};
pub use nose_hoover::{NoseHooverChain, ThermostatChain};
pub use rescale::{Berendsen, Bussi};

use super::space::MdSpace;
use crate::core_infra::rng::Rng;
use crate::domain::Units;
use crate::error::{MoptError, MoptResult};

/// What a thermostat sees when it acts.
pub struct ThermostatContext<'a> {
    /// The variables and their masses.
    pub space: &'a MdSpace,
    /// Degrees of freedom of the temperature (constraints and conserved motions taken out).
    pub dof: f64,
    /// Target temperature (K).
    pub temperature: f64,
    /// The random numbers of the run.
    pub rng: &'a mut Rng,
}

impl ThermostatContext<'_> {
    /// `k_B T` of the target (Hartree).
    pub fn kt(&self) -> f64 {
        Units::BOLTZMANN_HARTREE_PER_K * self.temperature
    }
}

/// A thermostat.
pub trait Thermostat: Send {
    /// Registered name.
    fn name(&self) -> &str;

    /// Act on the velocities `v` for the time `dt` (atomic units).
    fn apply(
        &mut self,
        dt: f64,
        v: &mut Col<f64>,
        ctx: &mut ThermostatContext<'_>,
    ) -> MoptResult<()>;

    /// The energy of the thermostat's own variables (Nose-Hoover chains) at the target
    /// temperature; `None` for thermostats without them, whose work on the velocities the
    /// driver accounts instead.
    fn bath_energy(&self, dof: f64, temperature: f64) -> Option<f64> {
        let _ = (dof, temperature);
        None
    }

    /// Whether the total momentum stays as it is (false for thermostats that act on every
    /// particle on its own with random forces).
    fn conserves_momentum(&self) -> bool {
        true
    }

    /// Whether the thermostat draws random numbers.
    fn is_stochastic(&self) -> bool {
        false
    }

    /// The state of its variables for the log.
    fn notes(&self) -> Vec<String> {
        Vec::new()
    }

    /// The paper of the method.
    fn reference(&self) -> Option<&'static str> {
        None
    }
}

/// No thermostat: the dynamics of the surface alone (NVE).
#[derive(Debug, Clone, Copy, Default)]
pub struct NoThermostat;

impl Thermostat for NoThermostat {
    fn name(&self) -> &str {
        "none"
    }

    fn apply(
        &mut self,
        _dt: f64,
        _v: &mut Col<f64>,
        _ctx: &mut ThermostatContext<'_>,
    ) -> MoptResult<()> {
        Ok(())
    }
}

/// Settings of the built-in thermostats.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ThermostatSettings {
    /// Coupling time (atomic units): the relaxation time of Berendsen and of the velocity
    /// rescaling, the period of the Nose-Hoover chains, and the inverse friction of
    /// Langevin and the inverse collision frequency of Andersen unless they are given.
    pub tau: f64,
    /// Langevin friction (inverse atomic time); `None`: `1 / tau`.
    pub friction: Option<f64>,
    /// Andersen collision frequency per particle (inverse atomic time); `None`: `1 / tau`.
    pub collision: Option<f64>,
    /// Links of a Nose-Hoover chain.
    pub chain: usize,
    /// Suzuki-Yoshida order of the chain integration (1, 3, or 5).
    pub yoshida: usize,
    /// Repetitions of the chain integration per half step.
    pub substeps: usize,
}

/// Coupling time of the thermostats without `tau` (fs; measured, see
/// `docs/source/molecular_dynamics.md`).
pub const DEFAULT_TAU_FS: f64 = 100.0;

impl Default for ThermostatSettings {
    fn default() -> Self {
        Self {
            tau: DEFAULT_TAU_FS * Units::FS_TO_AU_TIME,
            friction: None,
            collision: None,
            chain: 3,
            yoshida: 3,
            substeps: 1,
        }
    }
}

type ThermostatFactory =
    Box<dyn Fn(&ThermostatSettings) -> MoptResult<Box<dyn Thermostat>> + Send + Sync>;

/// Thermostats by name.
pub struct ThermostatRegistry {
    entries: Vec<(String, ThermostatFactory)>,
}

impl ThermostatRegistry {
    /// `none`, `berendsen`, `bussi` (`csvr`, `vrescale`), `langevin`, `andersen`,
    /// `nose_hoover` (a chain of one), `nose_hoover_chain`.
    pub fn builtin() -> Self {
        let mut r = Self {
            entries: Vec::new(),
        };
        r.register("none", |_| Ok(Box::new(NoThermostat)));
        r.register("berendsen", |s| Ok(Box::new(Berendsen::new(s.tau)?)));
        r.register("bussi", |s| Ok(Box::new(Bussi::new(s.tau)?)));
        r.register("langevin", |s| {
            Ok(Box::new(Langevin::new(s.friction.unwrap_or(1.0 / s.tau))?))
        });
        r.register("andersen", |s| {
            Ok(Box::new(Andersen::new(s.collision.unwrap_or(1.0 / s.tau))?))
        });
        r.register("nose_hoover", |s| {
            Ok(Box::new(NoseHooverChain::new(
                s.tau, 1, s.yoshida, s.substeps,
            )?))
        });
        r.register("nose_hoover_chain", |s| {
            Ok(Box::new(NoseHooverChain::new(
                s.tau, s.chain, s.yoshida, s.substeps,
            )?))
        });
        r
    }

    /// Add (or replace) a thermostat.
    pub fn register(
        &mut self,
        name: &str,
        make: impl Fn(&ThermostatSettings) -> MoptResult<Box<dyn Thermostat>> + Send + Sync + 'static,
    ) {
        self.entries.retain(|(n, _)| n != name);
        self.entries.push((name.to_string(), Box::new(make)));
    }

    /// Names.
    pub fn names(&self) -> Vec<String> {
        self.entries.iter().map(|(n, _)| n.clone()).collect()
    }

    /// The canonical name of `name` (aliases folded).
    pub fn canonical(name: &str) -> String {
        match name.trim().to_ascii_lowercase().replace('-', "_").as_str() {
            "csvr" | "vrescale" | "v_rescale" => "bussi".to_string(),
            "nosehoover" | "nh" => "nose_hoover".to_string(),
            "nosehooverchain" | "nhc" => "nose_hoover_chain".to_string(),
            "nve" | "off" => "none".to_string(),
            other => other.to_string(),
        }
    }

    /// The thermostat `name` with `settings`.
    pub fn create(
        &self,
        name: &str,
        settings: &ThermostatSettings,
    ) -> MoptResult<Box<dyn Thermostat>> {
        let key = Self::canonical(name);
        let make = self
            .entries
            .iter()
            .find(|(n, _)| *n == key)
            .map(|(_, m)| m)
            .ok_or_else(|| {
                MoptError::config(format!(
                    "unknown thermostat '{name}' (known: {})",
                    self.names().join(", ")
                ))
            })?;
        make(settings)
    }
}

impl Default for ThermostatRegistry {
    fn default() -> Self {
        Self::builtin()
    }
}

/// Check a coupling time or rate.
pub(crate) fn positive(value: f64, what: &str) -> MoptResult<f64> {
    if value.is_finite() && value > 0.0 {
        Ok(value)
    } else {
        Err(MoptError::config(format!(
            "{what} must be positive and finite, got {value}"
        )))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn every_built_in_thermostat_can_be_made() {
        let r = ThermostatRegistry::builtin();
        let s = ThermostatSettings::default();
        for name in [
            "none",
            "nve",
            "berendsen",
            "bussi",
            "csvr",
            "langevin",
            "andersen",
            "nosehoover",
            "nose_hoover_chain",
            "NHC",
        ] {
            assert!(r.create(name, &s).is_ok(), "{name}");
        }
        assert!(r.create("gauss", &s).is_err());
        let bad = ThermostatSettings { tau: -1.0, ..s };
        assert!(r.create("berendsen", &bad).is_err());
    }
}
