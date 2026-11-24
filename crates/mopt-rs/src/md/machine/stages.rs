//! The stages of a step of [`super::MolecularDynamics`] (kicks, drifts, thermostat,
//! barostat) and the report of a step.
//!
//! MultiOptPy: `MD/thermostat.py:Thermostat` (the updates of the momenta and positions).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use super::MolecularDynamics;
use crate::domain::PeriodicCell;
use crate::error::{MoptError, MoptResult};
use crate::md::integrator::Stage;
use crate::md::report::MdStepReport;
use crate::md::space::temperature_of;
use crate::md::thermostat::ThermostatContext;

impl MolecularDynamics {
    pub(super) fn execute(&mut self, stage: Stage) -> MoptResult<()> {
        let dt = self.settings.time_step;
        let w = self.space.inverse_masses().to_vec();
        match stage {
            Stage::Kick(share) => {
                let h = share * dt;
                match &self.parts.barostat {
                    Some(b) => b.kick(h, &mut self.v, &self.acceleration),
                    None => {
                        for (v, a) in self.v.iter_mut().zip(self.acceleration.iter()) {
                            *v += h * a;
                        }
                    }
                }
                self.rattle.project_velocities(&self.x, &mut self.v, &w)
            }
            Stage::Drift(share) => {
                let h = share * dt;
                let old = self.x.clone();
                match (&mut self.parts.barostat, &mut self.cell) {
                    (Some(b), Some(cell)) => b.drift(h, &mut self.x, &mut self.v, cell)?,
                    _ => {
                        for (x, v) in self.x.iter_mut().zip(self.v.iter()) {
                            *x += h * v;
                        }
                    }
                }
                self.rattle.shake(&old, &mut self.x, &mut self.v, h, &w)?;
                Ok(())
            }
            Stage::Thermostat(share) => self.thermostat(share * dt, &w),
            Stage::Barostat(share) => self.barostat(share * dt),
        }
    }

    fn thermostat(&mut self, h: f64, w: &[f64]) -> MoptResult<()> {
        let before = self.space.kinetic_energy(&self.v);
        let mut ctx = ThermostatContext {
            space: &self.space,
            dof: self.dof,
            temperature: self.target,
            rng: &mut self.rng,
        };
        self.parts.thermostat.apply(h, &mut self.v, &mut ctx)?;
        if self.parts.thermostat.is_stochastic() {
            self.rattle.project_velocities(&self.x, &mut self.v, w)?;
        }
        if self
            .parts
            .thermostat
            .bath_energy(self.dof, self.target)
            .is_none()
        {
            self.work += before - self.space.kinetic_energy(&self.v);
        }
        Ok(())
    }

    fn barostat(&mut self, h: f64) -> MoptResult<()> {
        let Some(b) = self.parts.barostat.as_mut() else {
            return Ok(());
        };
        let (Some(cell), Some(stress)) = (&self.cell, self.stress) else {
            return Err(MoptError::config(
                "a barostat needs a periodic cell and the stress",
            ));
        };
        let mut ctx = crate::md::barostat::BarostatContext {
            space: &self.space,
            dof: self.dof,
            temperature: self.target,
            pressure: self.settings.pressure,
            volume: cell.volume_bohr3(),
            kinetic: self.space.kinetic_tensor(&self.v),
            stress,
            rng: &mut self.rng,
        };
        b.apply(h, &mut self.v, &mut ctx)
    }

    pub(super) fn report(&mut self) -> MoptResult<MdStepReport> {
        let kinetic = self.space.kinetic_energy(&self.v);
        let bath = self
            .parts
            .thermostat
            .bath_energy(self.dof, self.target)
            .unwrap_or(self.work);
        let volume = self.cell.as_ref().map(PeriodicCell::volume_bohr3);
        let baro = match (&self.parts.barostat, volume) {
            (Some(b), Some(v)) => b.energy(v, self.settings.pressure, self.target),
            _ => 0.0,
        };
        let full = self
            .cell
            .as_ref()
            .is_some_and(|c| c.periodic_axes.iter().all(|p| *p));
        let pressure = match (self.stress, volume, full) {
            (Some(s), Some(v), true) => {
                let p = crate::md::barostat::pressure_tensor(
                    &self.space.kinetic_tensor(&self.v),
                    &s,
                    v,
                );
                Some((p[0][0] + p[1][1] + p[2][2]) / 3.0)
            }
            _ => None,
        };
        let mut notes = self.parts.thermostat.notes();
        if let Some(b) = &self.parts.barostat {
            notes.extend(b.notes());
        }
        Ok(MdStepReport {
            step: self.step,
            time: self.step as f64 * self.settings.time_step,
            x: self.x.clone(),
            v: self.v.clone(),
            cell: self.cell.clone(),
            energy: self.energy,
            raw_energy: self.raw_energy,
            gradient: self.gradient.clone(),
            raw_gradient: self.raw_gradient.clone(),
            kinetic,
            temperature: temperature_of(kinetic, self.dof),
            target_temperature: self.target,
            dof: self.dof,
            conserved: self.energy + kinetic + bath + baro,
            pressure,
            volume,
            stress: self.stress,
            constraint_residual: if self.rattle.is_empty() {
                None
            } else {
                Some(self.rattle.max_residual(&self.x)?)
            },
            freeze: self.freeze_event.take(),
            notes,
            dipole: self.dipole,
            heat_exchanged: self.exchanged,
            electronic_state: self.hopping.as_ref().map(|h| h.active()),
            state_energies: self.state_energies.clone(),
            hops: std::mem::take(&mut self.hop_records),
        })
    }

    /// Muller-Plathe's exchange: the atom of the cold slab with the largest kinetic energy
    /// and the atom of the hot slab with the smallest, of the same mass, swap velocities when
    /// that moves energy from the cold to the hot slab.
    pub(super) fn exchange(&mut self, e: crate::md::settings::Exchange) -> MoptResult<()> {
        let Some(cell) = self.cell.as_ref() else {
            return Err(MoptError::config("velocity exchanges need a cell"));
        };
        let m = self.space.masses();
        let w = self.space.inverse_masses();
        let mut cold: Option<(usize, f64)> = None;
        let mut hot: Option<(usize, f64)> = None;
        for i in 0..self.space.particles() {
            if w[3 * i] == 0.0 {
                continue;
            }
            let s =
                cell.cart_to_frac([self.x[3 * i], self.x[3 * i + 1], self.x[3 * i + 2]])[e.axis];
            let slab = ((s - libm::floor(s)) * e.slabs as f64).floor() as usize % e.slabs;
            let ke = 0.5 * m[3 * i] * (0..3).map(|c| self.v[3 * i + c].powi(2)).sum::<f64>();
            if slab == 0 && cold.is_none_or(|(_, k)| ke > k) {
                cold = Some((i, ke));
            }
            if slab == e.slabs / 2 && hot.is_none_or(|(_, k)| ke < k) {
                hot = Some((i, ke));
            }
        }
        if let (Some((c, kc)), Some((h, kh))) = (cold, hot) {
            if kc > kh && (m[3 * c] - m[3 * h]).abs() <= 1e-12 * m[3 * c] {
                for k in 0..3 {
                    let t = self.v[3 * c + k];
                    self.v[3 * c + k] = self.v[3 * h + k];
                    self.v[3 * h + k] = t;
                }
                self.exchanged += kc - kh;
            }
        }
        Ok(())
    }
}
