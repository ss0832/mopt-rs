//! `eyring`: `k = (k_B T / h) exp(-(G_ts - G_min) / RT)` with one energy per stationary point:
//! the electronic energy (MultiOptPy's default) or the Gibbs free energy given in the network
//! file (`--energy-type free`).
//!
//! MultiOptPy: `Utils/rcmc.py:RCMCQueue.pop` (`k_uv = kB_T_h * exp(-(E_TS_u - E_u) / RT_ha)`).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use super::RateTheory;
use crate::error::MoptResult;
use crate::kinetics::conditions::{Conditions, EnergyKind};
use crate::kinetics::network::{Bottleneck, KineticState};

/// The Eyring rate theory with the energies of the file.
#[derive(Debug, Clone, Copy, Default)]
pub struct Eyring;

impl RateTheory for Eyring {
    fn name(&self) -> &str {
        "eyring"
    }

    fn state_energy(&self, state: &KineticState, c: &Conditions) -> MoptResult<Option<f64>> {
        Ok(match c.energy {
            EnergyKind::Electronic => Some(state.energy),
            EnergyKind::Free => state.free_energy,
        })
    }

    fn bottleneck_energy(&self, b: &Bottleneck, c: &Conditions) -> MoptResult<Option<f64>> {
        Ok(match c.energy {
            EnergyKind::Electronic => Some(b.energy),
            EnergyKind::Free => b.free_energy,
        })
    }

    fn standard_state(&self, c: &Conditions) -> Option<f64> {
        match c.energy {
            EnergyKind::Electronic => None,
            // The free energies of the file are taken at the standard state of the conditions.
            EnergyKind::Free => Some(c.free_energy_concentration()),
        }
    }

    fn describe(&self, c: &Conditions) -> String {
        match c.energy {
            EnergyKind::Electronic => "electronic".to_string(),
            EnergyKind::Free => "free (Gibbs, from the network file)".to_string(),
        }
    }
}
