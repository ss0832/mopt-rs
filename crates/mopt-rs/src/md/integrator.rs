//! Integrators as splittings of one time step (extension point).
//!
//! A step is a sequence of stages around one evaluation of the forces: kicks (the
//! velocities follow the forces), drifts (the positions follow the velocities), and the
//! actions of the thermostat and the barostat, each for a share of the time step. An
//! [`Integrator`] names the stages before the evaluation of the forces at the new positions
//! and after it. Velocity Verlet with the thermostat and barostat at both ends is the
//! symmetric Trotter splitting of Martyna, Tuckerman, Tobias, and Klein; BAOAB puts the
//! thermostat in the middle of the drift (Leimkuhler and Matthews). Any other splitting
//! with one force evaluation per step is a new implementation of the trait.
//!
//! MultiOptPy: `MD/thermostat.py:Thermostat` (`Velocity_Verlet`, `Nose_Hoover_thermostat`,
//! `Langevin_thermostat`: one fixed sequence each).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use crate::error::{MoptError, MoptResult};

/// One stage of a step, with its share of the time step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Stage {
    /// The velocities follow the forces.
    Kick(f64),
    /// The positions follow the velocities (and the cell its barostat).
    Drift(f64),
    /// The thermostat acts on the velocities.
    Thermostat(f64),
    /// The barostat acts on the cell and its own variables.
    Barostat(f64),
}

/// A splitting of a time step around one evaluation of the forces.
pub trait Integrator: Send + Sync {
    /// Registered name.
    fn name(&self) -> &str;

    /// Stages from the positions and forces of the last evaluation to the positions of the
    /// next one.
    fn before_forces(&self) -> Vec<Stage>;

    /// Stages after the forces at the new positions are known.
    fn after_forces(&self) -> Vec<Stage>;

    /// The paper of the method, for the log.
    fn reference(&self) -> Option<&'static str> {
        None
    }
}

/// Velocity Verlet with the thermostat and the barostat at both ends of the step (with a
/// Langevin thermostat this is OBABO).
#[derive(Debug, Clone, Copy, Default)]
pub struct VelocityVerlet;

impl Integrator for VelocityVerlet {
    fn name(&self) -> &str {
        "velocity_verlet"
    }

    fn before_forces(&self) -> Vec<Stage> {
        vec![
            Stage::Thermostat(0.5),
            Stage::Barostat(0.5),
            Stage::Kick(0.5),
            Stage::Drift(1.0),
        ]
    }

    fn after_forces(&self) -> Vec<Stage> {
        vec![
            Stage::Kick(0.5),
            Stage::Barostat(0.5),
            Stage::Thermostat(0.5),
        ]
    }

    fn reference(&self) -> Option<&'static str> {
        Some(
            "W. C. Swope, H. C. Andersen, P. H. Berens, K. R. Wilson, J. Chem. Phys. 1982, 76, \
             637; G. J. Martyna, M. E. Tuckerman, D. J. Tobias, M. L. Klein, Mol. Phys. 1996, \
             87, 1117 (Trotter splitting)",
        )
    }
}

/// BAOAB: the thermostat in the middle of the drift (Leimkuhler and Matthews), the
/// splitting with the smallest error of configurational averages for Langevin dynamics.
#[derive(Debug, Clone, Copy, Default)]
pub struct Baoab;

impl Integrator for Baoab {
    fn name(&self) -> &str {
        "baoab"
    }

    fn before_forces(&self) -> Vec<Stage> {
        vec![
            Stage::Barostat(0.5),
            Stage::Kick(0.5),
            Stage::Drift(0.5),
            Stage::Thermostat(1.0),
            Stage::Drift(0.5),
        ]
    }

    fn after_forces(&self) -> Vec<Stage> {
        vec![Stage::Kick(0.5), Stage::Barostat(0.5)]
    }

    fn reference(&self) -> Option<&'static str> {
        Some("B. Leimkuhler and C. Matthews, Appl. Math. Res. Express 2013, 34; J. Chem. Phys. 2013, 138, 174102")
    }
}

/// Check that the stages of `integrator` add up to one step: kicks, drifts, thermostat,
/// and barostat each once over the step, with at least one drift before the forces.
pub fn check(integrator: &dyn Integrator) -> MoptResult<()> {
    let before = integrator.before_forces();
    let after = integrator.after_forces();
    let mut sums = [0.0_f64; 4];
    for stage in before.iter().chain(&after) {
        let (k, share) = match *stage {
            Stage::Kick(s) => (0, s),
            Stage::Drift(s) => (1, s),
            Stage::Thermostat(s) => (2, s),
            Stage::Barostat(s) => (3, s),
        };
        if !(share.is_finite() && share > 0.0) {
            return Err(MoptError::config(format!(
                "integrator '{}': every stage needs a positive share of the step",
                integrator.name()
            )));
        }
        sums[k] += share;
    }
    let drift_after = after.iter().any(|s| matches!(s, Stage::Drift(_)));
    let names = ["kicks", "drifts", "thermostat stages", "barostat stages"];
    for (k, total) in sums.iter().enumerate() {
        if (total - 1.0).abs() > 1e-12 {
            return Err(MoptError::config(format!(
                "integrator '{}': the {} add up to {total} steps, not one",
                integrator.name(),
                names[k]
            )));
        }
    }
    if drift_after {
        return Err(MoptError::config(format!(
            "integrator '{}': the positions may move only before the forces are evaluated",
            integrator.name()
        )));
    }
    Ok(())
}

/// Integrators by name.
pub struct IntegratorRegistry {
    entries: Vec<(String, Box<dyn Fn() -> Box<dyn Integrator> + Send + Sync>)>,
}

impl IntegratorRegistry {
    /// `velocity_verlet` (also `vv`, `verlet`) and `baoab`.
    pub fn builtin() -> Self {
        let mut r = Self {
            entries: Vec::new(),
        };
        r.register("velocity_verlet", || Box::new(VelocityVerlet));
        r.register("baoab", || Box::new(Baoab));
        r
    }

    /// Add (or replace) an integrator.
    pub fn register(
        &mut self,
        name: &str,
        make: impl Fn() -> Box<dyn Integrator> + Send + Sync + 'static,
    ) {
        self.entries.retain(|(n, _)| n != name);
        self.entries.push((name.to_string(), Box::new(make)));
    }

    /// Names.
    pub fn names(&self) -> Vec<String> {
        self.entries.iter().map(|(n, _)| n.clone()).collect()
    }

    /// The integrator `name`.
    pub fn create(&self, name: &str) -> MoptResult<Box<dyn Integrator>> {
        let key = match name.trim().to_ascii_lowercase().as_str() {
            "vv" | "verlet" | "velocityverlet" => "velocity_verlet".to_string(),
            other => other.to_string(),
        };
        let integrator = self
            .entries
            .iter()
            .find(|(n, _)| *n == key)
            .map(|(_, make)| make())
            .ok_or_else(|| {
                MoptError::config(format!(
                    "unknown integrator '{name}' (known: {})",
                    self.names().join(", ")
                ))
            })?;
        check(integrator.as_ref())?;
        Ok(integrator)
    }
}

impl Default for IntegratorRegistry {
    fn default() -> Self {
        Self::builtin()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    struct Lopsided;

    impl Integrator for Lopsided {
        fn name(&self) -> &str {
            "lopsided"
        }
        fn before_forces(&self) -> Vec<Stage> {
            vec![Stage::Kick(1.0), Stage::Drift(1.0)]
        }
        fn after_forces(&self) -> Vec<Stage> {
            vec![
                Stage::Kick(0.5),
                Stage::Thermostat(1.0),
                Stage::Barostat(1.0),
            ]
        }
    }

    #[test]
    fn the_built_in_splittings_are_complete_and_others_are_checked() {
        let r = IntegratorRegistry::builtin();
        for name in ["velocity_verlet", "vv", "baoab"] {
            assert!(r.create(name).is_ok(), "{name}");
        }
        assert!(r.create("leapfrog").is_err());
        let e = check(&Lopsided).unwrap_err().to_string();
        assert!(e.contains("kicks"), "{e}");
    }
}
