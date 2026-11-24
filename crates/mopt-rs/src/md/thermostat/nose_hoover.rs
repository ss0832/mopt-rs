//! Nose-Hoover chains, integrated as in Martyna, Tuckerman, Tobias, and Klein (Trotter
//! factorization with Suzuki-Yoshida weights), for the particles and for the barostat of
//! an isothermal-isobaric run.
//!
//! The thermostat masses are `Q_1 = N_f k_B T tau^2` and `Q_j = k_B T tau^2` (Martyna,
//! Klein, Tuckerman, J. Chem. Phys. 1992, 97, 2635), so that `tau` is the period of the
//! motion of the chain.
//!
//! MultiOptPy: `MD/thermostat.py:Thermostat.Nose_Hoover_thermostat`,
//! `Thermostat._propagate_nhc_zeta`, `Thermostat.Nose_Hoover_chain_thermostat`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Col;

use super::{positive, Thermostat, ThermostatContext};
use crate::error::{MoptError, MoptResult};

/// The weights of the Suzuki-Yoshida factorization of order `order` (1, 3, or 5).
pub fn yoshida_weights(order: usize) -> MoptResult<Vec<f64>> {
    match order {
        1 => Ok(vec![1.0]),
        3 => {
            let w = 1.0 / (2.0 - libm::cbrt(2.0));
            Ok(vec![w, 1.0 - 2.0 * w, w])
        }
        5 => {
            let w = 1.0 / (4.0 - libm::pow(4.0, 1.0 / 3.0));
            Ok(vec![w, w, 1.0 - 4.0 * w, w, w])
        }
        other => Err(MoptError::config(format!(
            "the Suzuki-Yoshida order of a Nose-Hoover chain is 1, 3, or 5, got {other}"
        ))),
    }
}

/// A Nose-Hoover chain coupled to a set of degrees of freedom.
#[derive(Debug, Clone)]
pub struct ThermostatChain {
    positions: Vec<f64>,
    velocities: Vec<f64>,
    weights: Vec<f64>,
    substeps: usize,
    tau: f64,
}

impl ThermostatChain {
    /// A chain of `links` with period `tau` (atomic units).
    pub fn new(links: usize, tau: f64, yoshida: usize, substeps: usize) -> MoptResult<Self> {
        if links == 0 || substeps == 0 {
            return Err(MoptError::config(
                "a Nose-Hoover chain needs at least one link and one substep",
            ));
        }
        Ok(Self {
            positions: vec![0.0; links],
            velocities: vec![0.0; links],
            weights: yoshida_weights(yoshida)?,
            substeps,
            tau: positive(tau, "the period of a Nose-Hoover chain")?,
        })
    }

    /// The masses of the links for `dof` coupled degrees of freedom at `kt`.
    fn masses(&self, dof: f64, kt: f64) -> Vec<f64> {
        let base = kt * self.tau * self.tau;
        (0..self.velocities.len())
            .map(|j| if j == 0 { dof.max(1.0) * base } else { base })
            .collect()
    }

    /// The force on link `j` with twice the kinetic energy `k2` of the coupled variables.
    fn force(&self, j: usize, k2: f64, dof: f64, kt: f64, q: &[f64]) -> f64 {
        if j == 0 {
            (k2 - dof * kt) / q[0]
        } else {
            let v = self.velocities[j - 1];
            (q[j - 1] * v * v - kt) / q[j]
        }
    }

    /// Advance the chain by `dt` for coupled variables with twice the kinetic energy `k2`
    /// (`dof` of them, at `kt`); returns the factor by which their velocities are scaled.
    pub fn propagate(&mut self, dt: f64, k2: f64, dof: f64, kt: f64) -> f64 {
        if kt <= 0.0 || dof <= 0.0 {
            return 1.0;
        }
        let q = self.masses(dof, kt);
        let last = self.velocities.len() - 1;
        let mut k2 = k2;
        let mut scale = 1.0;
        let weights = self.weights.clone();
        for _ in 0..self.substeps {
            for w in &weights {
                let h = w * dt / self.substeps as f64;
                self.velocities[last] += 0.5 * h * self.force(last, k2, dof, kt, &q);
                for j in (0..last).rev() {
                    let damp = libm::exp(-0.25 * h * self.velocities[j + 1]);
                    self.velocities[j] *= damp;
                    self.velocities[j] += 0.5 * h * self.force(j, k2, dof, kt, &q);
                    self.velocities[j] *= damp;
                }
                let s = libm::exp(-h * self.velocities[0]);
                scale *= s;
                k2 *= s * s;
                for (x, v) in self.positions.iter_mut().zip(&self.velocities) {
                    *x += h * v;
                }
                for j in 0..last {
                    let damp = libm::exp(-0.25 * h * self.velocities[j + 1]);
                    self.velocities[j] *= damp;
                    self.velocities[j] += 0.5 * h * self.force(j, k2, dof, kt, &q);
                    self.velocities[j] *= damp;
                }
                self.velocities[last] += 0.5 * h * self.force(last, k2, dof, kt, &q);
            }
        }
        scale
    }

    /// The energy of the chain: `sum Q_j v_j^2 / 2 + N_f k_B T x_1 + k_B T sum_{j>1} x_j`.
    pub fn energy(&self, dof: f64, kt: f64) -> f64 {
        let q = self.masses(dof, kt);
        let kinetic: f64 = q
            .iter()
            .zip(&self.velocities)
            .map(|(q, v)| 0.5 * q * v * v)
            .sum();
        let potential: f64 = self
            .positions
            .iter()
            .enumerate()
            .map(|(j, x)| if j == 0 { dof * kt * x } else { kt * x })
            .sum();
        kinetic + potential
    }

    /// The velocities of the links (inverse atomic time).
    pub fn velocities(&self) -> &[f64] {
        &self.velocities
    }
}

/// A Nose-Hoover chain on the velocities of the particles (one link: a Nose-Hoover
/// thermostat).
#[derive(Debug, Clone)]
pub struct NoseHooverChain {
    chain: ThermostatChain,
}

impl NoseHooverChain {
    /// A chain of `links` with period `tau` (atomic units).
    pub fn new(tau: f64, links: usize, yoshida: usize, substeps: usize) -> MoptResult<Self> {
        Ok(Self {
            chain: ThermostatChain::new(links, tau, yoshida, substeps)?,
        })
    }
}

impl Thermostat for NoseHooverChain {
    fn name(&self) -> &str {
        if self.chain.velocities.len() == 1 {
            "nose_hoover"
        } else {
            "nose_hoover_chain"
        }
    }

    fn apply(
        &mut self,
        dt: f64,
        v: &mut Col<f64>,
        ctx: &mut ThermostatContext<'_>,
    ) -> MoptResult<()> {
        let k2 = 2.0 * ctx.space.kinetic_energy(v);
        let scale = self.chain.propagate(dt, k2, ctx.dof, ctx.kt());
        for x in v.iter_mut() {
            *x *= scale;
        }
        Ok(())
    }

    fn bath_energy(&self, dof: f64, temperature: f64) -> Option<f64> {
        let kt = crate::domain::Units::BOLTZMANN_HARTREE_PER_K * temperature;
        Some(self.chain.energy(dof, kt))
    }

    fn notes(&self) -> Vec<String> {
        let values: Vec<String> = self
            .chain
            .velocities()
            .iter()
            .map(|v| format!("{v:.6e}"))
            .collect();
        vec![format!(
            "Nose-Hoover chain velocities (1/a.u.): {}",
            values.join(" ")
        )]
    }

    fn reference(&self) -> Option<&'static str> {
        Some(
            "G. J. Martyna, M. L. Klein, M. Tuckerman, J. Chem. Phys. 1992, 97, 2635; G. J. \
             Martyna, M. E. Tuckerman, D. J. Tobias, M. L. Klein, Mol. Phys. 1996, 87, 1117",
        )
    }
}
