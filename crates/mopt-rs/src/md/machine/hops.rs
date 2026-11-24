//! The hops of surface hopping in [`super::MolecularDynamics`]: after a step is complete,
//! the candidates at a gap minimum of the step before are decided
//! ([`crate::md::hopping::SurfaceHopping::observe`]), and an accepted hop changes the
//! velocities along the hopping direction, the active state, and the forces.
//!
//! The direction is taken in the space the velocities may move in: the conserved motions of
//! the whole system (`rigid`) are taken out of it and it is projected onto the constraints,
//! so that the changed velocities keep them, and then scaled to mass-weighted length one.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Col;

use super::MolecularDynamics;
use crate::error::{MoptError, MoptResult};
use crate::md::hopping::direction::{adjust_velocities, normalized, reverse_along};
use crate::md::hopping::{FrustratedHop, HopOutcome};

impl MolecularDynamics {
    /// Decide on a hop after the step is complete, and make it.
    pub(super) fn consider_hop(&mut self) -> MoptResult<()> {
        let Some(hopping) = self.hopping.as_mut() else {
            return Ok(());
        };
        let values = self.states.take().ok_or_else(|| {
            MoptError::calc("surface hopping: the states of the step are missing")
        })?;
        let w = self.space.inverse_masses().to_vec();
        let m = self.space.masses().to_vec();
        let time = self.step as f64 * self.settings.time_step;
        let mut records = Vec::new();
        let plan = hopping.observe(
            self.step,
            time,
            &self.v,
            values.clone(),
            &w,
            &m,
            &mut records,
        );
        self.hop_records = records;
        let Some(mut plan) = plan else {
            return Ok(());
        };
        let to = plan.to;
        let (Some(energy), Some(gradient)) = (values.energy(to), values.gradient(to).cloned())
        else {
            return Err(MoptError::calc(format!(
                "surface hopping: no gradient of state {to} at the step of the hop"
            )));
        };
        let mut u = plan.direction.clone();
        self.space
            .remove_rigid_motion(&self.x, &mut u, self.rigid)?;
        self.rattle.project_velocities(&self.x, &mut u, &w)?;
        let frustrated = self.settings.hopping.as_ref().map(|h| h.frustrated);
        let kinetic_before = self.space.kinetic_energy(&self.v);
        let made = match normalized(&u, &m) {
            Some(u) => {
                let made = adjust_velocities(&mut self.v, &u, &m, energy - self.energy).is_some();
                if !made && frustrated == Some(FrustratedHop::Reverse) {
                    reverse_along(&mut self.v, &u, &m);
                }
                made
            }
            None => false,
        };
        if made {
            plan.record.potential_change = energy - self.energy;
            plan.record.kinetic_change = self.space.kinetic_energy(&self.v) - kinetic_before;
            // The bias potentials are the same on every state.
            let bias = self.energy - self.raw_energy;
            let bias_gradient =
                Col::<f64>::from_fn(w.len(), |i| self.gradient[i] - self.raw_gradient[i]);
            self.energy = energy;
            self.raw_energy = energy - bias;
            self.raw_gradient = Col::from_fn(w.len(), |i| gradient[i] - bias_gradient[i]);
            self.acceleration = Col::from_fn(w.len(), |i| -gradient[i] * w[i]);
            self.gradient = gradient;
            if let Some(h) = self.hopping.as_mut() {
                h.hopped(to);
            }
        } else {
            plan.record.outcome = HopOutcome::Frustrated;
        }
        self.hop_records.push(plan.record);
        Ok(())
    }
}
