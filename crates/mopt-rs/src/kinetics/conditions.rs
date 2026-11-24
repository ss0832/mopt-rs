//! The conditions of a kinetic model: temperature, the standard state, which energies are used,
//! and how saddle points below a minimum are treated.
//!
//! The constants are those of MultiOptPy's `rcmc.py` (`k_B` in J/K and in Hartree/K, `h`),
//! which are those of [`crate::domain::Units`], so that the Eyring rate constants agree with
//! MultiOptPy's to rounding.
//!
//! MultiOptPy: `Utils/rcmc.py` (`K_B_J_K`, `H_J_S`, `K_B_HARTREE`, `--energy-type`).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use crate::domain::Units;
use crate::error::{MoptError, MoptResult};

/// Which energy of a stationary point enters the rates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnergyKind {
    /// The electronic energy (MultiOptPy's default).
    Electronic,
    /// The Gibbs free energy given in the network file.
    Free,
}

impl EnergyKind {
    /// `electronic` or `free`.
    pub fn parse(text: &str) -> MoptResult<Self> {
        match text.to_ascii_lowercase().as_str() {
            "electronic" | "e" | "scf" => Ok(Self::Electronic),
            "free" | "gibbs" | "g" => Ok(Self::Free),
            other => Err(MoptError::parse(format!(
                "unknown energy type '{other}' (electronic, free)"
            ))),
        }
    }

    /// Its name.
    pub fn name(self) -> &'static str {
        match self {
            Self::Electronic => "electronic",
            Self::Free => "free",
        }
    }
}

/// The treatment of a saddle point that lies below one of its minima (possible with free
/// energies or mixed levels of theory).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Clamp {
    /// `max(E_ts, E_a, E_b)` for both directions: detailed balance is kept.
    Consistent,
    /// MultiOptPy's `max(E_ts, E_a)` and `max(E_ts, E_b)` separately for the two directions,
    /// which breaks detailed balance when the saddle point lies below the higher minimum.
    Legacy,
    /// No clamp: a rate constant may exceed `k_B T / h`.
    None,
}

impl Clamp {
    /// `consistent`, `legacy`, or `none`.
    pub fn parse(text: &str) -> MoptResult<Self> {
        match text.to_ascii_lowercase().as_str() {
            "consistent" => Ok(Self::Consistent),
            "legacy" | "multioptpy" => Ok(Self::Legacy),
            "none" | "off" => Ok(Self::None),
            other => Err(MoptError::parse(format!(
                "unknown clamp '{other}' (consistent, legacy, none)"
            ))),
        }
    }

    /// Its name.
    pub fn name(self) -> &'static str {
        match self {
            Self::Consistent => "consistent",
            Self::Legacy => "legacy",
            Self::None => "none",
        }
    }
}

/// The conditions of a kinetic model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Conditions {
    /// Temperature (K).
    pub temperature: f64,
    /// Pressure of the standard state of the gas (Pa), for partition functions.
    pub pressure: f64,
    /// Concentration of the standard state in solution (mol/L) instead of the gas: that of
    /// the translational entropy of [`crate::vib::ThermoSettings::concentration`] (`-conc`),
    /// and then also `c°` of the bimolecular steps.
    pub concentration: Option<f64>,
    /// The energies used.
    pub energy: EnergyKind,
    /// The treatment of saddle points below a minimum.
    pub clamp: Clamp,
    /// The viscosity of the solvent (Pa s), for the association of barrierless steps by
    /// diffusion (Collins and Kimball); `None`: the gas (capture on contact alone).
    pub viscosity: Option<f64>,
}

impl Default for Conditions {
    fn default() -> Self {
        Self {
            temperature: 300.0,
            pressure: 101_325.0,
            concentration: None,
            energy: EnergyKind::Electronic,
            clamp: Clamp::Consistent,
            viscosity: None,
        }
    }
}

impl Conditions {
    /// `k_B T` in Hartree.
    pub fn kt(&self) -> f64 {
        Units::BOLTZMANN_HARTREE_PER_K * self.temperature
    }

    /// `ln(k_B T / h)` with `k_B T / h` in 1/s.
    pub fn ln_kt_over_h(&self) -> f64 {
        (Units::BOLTZMANN_J_PER_K * self.temperature / Units::PLANCK_J_S).ln()
    }

    /// The concentration (mol/L) of the standard state of the free energies, as
    /// [`crate::vib::thermochemistry`] takes it (`-conc` and `-press` of `freqmain`): the
    /// concentration given for a solution, or that of the ideal gas at the pressure of the
    /// conditions, `p / RT` (0.0409 mol/L at 1 atm and 298.15 K).
    pub fn free_energy_concentration(&self) -> f64 {
        self.concentration.unwrap_or_else(|| {
            self.pressure / (Units::GAS_CONSTANT_J_MOL_K * self.temperature) / 1000.0
        })
    }

    /// The standard concentration `c°` (mol/L) of the bimolecular steps: the concentration of
    /// the free energies when one is given (`-conc`), otherwise 1 mol/L. A pseudo-first-order
    /// rate constant is `k [S] / c°` in these units.
    pub fn standard_concentration(&self) -> f64 {
        self.concentration.unwrap_or(1.0)
    }

    /// The conversion (Hartree per molecule) of a free energy from the standard state of
    /// `from` (mol/L, e.g. [`Conditions::free_energy_concentration`]) to `c°`:
    /// `k_B T ln(c° / c)`. From the ideal gas at `p` it is `RT ln(c° R T / p)` per mole,
    /// `RT ln 24.47` = 1.894 kcal/mol at 298.15 K and 1 atm (`RT ln 24.79` at 1 bar).
    pub fn standard_state_shift(&self, from: f64) -> f64 {
        self.kt() * (self.standard_concentration() / from).ln()
    }

    /// Check the values.
    pub fn check(&self) -> MoptResult<()> {
        let positive = |x: f64| x.is_finite() && x > 0.0;
        if !positive(self.temperature) {
            return Err(MoptError::config(format!(
                "the temperature must be positive (got {} K)",
                self.temperature
            )));
        }
        if !positive(self.pressure) || self.concentration.is_some_and(|c| !positive(c)) {
            return Err(MoptError::config(
                "the pressure and the concentration of the standard state must be positive",
            ));
        }
        if self.viscosity.is_some_and(|eta| !positive(eta)) {
            return Err(MoptError::config("the viscosity must be positive (Pa s)"));
        }
        Ok(())
    }
}
