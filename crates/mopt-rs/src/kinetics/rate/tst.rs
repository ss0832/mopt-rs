//! `tst`: transition state theory with free energies from partition functions,
//! `k = (k_B T / h) exp(-(G_ts - G_min) / RT)` with `G = E + G_trans + G_rot + G_vib + G_elec`
//! of the ideal gas ([`crate::vib::thermochemistry`]): rigid rotors with the rotational
//! symmetry number of each structure (found from its point group, or given, or 1), harmonic
//! oscillators with the quasi-harmonic treatment of low frequencies (Grimme's interpolation of
//! the entropy with Head-Gordon's thermal energy by default, or Truhlar's raised frequencies,
//! or none), and the standard state of the gas at the pressure of the conditions or of a
//! solution at a concentration. The imaginary frequency of a saddle point does not enter.
//!
//! For the unimolecular steps of a network the standard state cancels (every state has the
//! same translational partition function); it is kept for networks with association steps.
//! The symmetry numbers give the reaction-path degeneracy `sigma_min / sigma_ts` when
//! symmetry-equivalent saddle points are one bottleneck, as in the networks of the mapper
//! (Fernandez-Ramos et al., Theor. Chem. Acc. 2007, 118, 813).
//!
//! A state or bottleneck without frequencies or without a structure is left out and listed.
//!
//! MultiOptPy: no counterpart (new in mopt-rs); the free energies of MultiOptPy's network are
//! read by `eyring --energy-type free`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use super::RateTheory;
use crate::domain::MolecularSystem;
use crate::error::MoptResult;
use crate::kinetics::conditions::Conditions;
use crate::kinetics::network::{Bottleneck, KineticState};
use crate::vib::{point_group, thermochemistry, NormalModes, QuasiHarmonic, ThermoSettings};

/// Tolerance (Bohr) of the point-group analysis, that of `freqmain`.
pub const SYMMETRY_TOLERANCE: f64 = 1e-2;

/// How the rotational symmetry numbers are found.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymmetryChoice {
    /// From the point group of each structure (where none is given).
    Auto,
    /// 1 for every structure (where none is given).
    None,
}

impl SymmetryChoice {
    /// `auto` or `none`.
    pub fn parse(text: &str) -> MoptResult<Self> {
        match text.to_ascii_lowercase().as_str() {
            "auto" | "on" => Ok(Self::Auto),
            "none" | "off" | "1" => Ok(Self::None),
            other => Err(crate::error::MoptError::parse(format!(
                "unknown symmetry choice '{other}' (auto, none)"
            ))),
        }
    }
}

/// The settings of `tst`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QrrhoSettings {
    /// Treatment of low frequencies (`grimme`, as `freqmain`).
    pub method: QuasiHarmonic,
    /// Its cutoff (cm^-1), 100 as Grimme and `freqmain`.
    pub cutoff: f64,
    /// The symmetry numbers.
    pub symmetry: SymmetryChoice,
}

impl Default for QrrhoSettings {
    fn default() -> Self {
        Self {
            method: QuasiHarmonic::Grimme,
            cutoff: 100.0,
            symmetry: SymmetryChoice::Auto,
        }
    }
}

/// TST with partition functions.
#[derive(Debug, Clone, Copy)]
pub struct Tst {
    settings: QrrhoSettings,
}

impl Tst {
    /// With `settings`.
    pub fn new(settings: QrrhoSettings) -> Self {
        Self { settings }
    }

    /// The Gibbs free energy (Hartree) of a structure with `frequencies` and the electronic
    /// energy `energy`.
    pub fn gibbs(
        &self,
        system: &MolecularSystem,
        frequencies: &[f64],
        energy: f64,
        multiplicity: u32,
        symmetry_number: Option<u32>,
        c: &Conditions,
    ) -> MoptResult<f64> {
        let sigma = match (symmetry_number, self.settings.symmetry) {
            (Some(s), _) => s,
            (None, SymmetryChoice::Auto) => point_group(system, SYMMETRY_TOLERANCE).symmetry_number,
            (None, SymmetryChoice::None) => 1,
        };
        let settings = ThermoSettings {
            temperature: c.temperature,
            pressure: c.pressure,
            quasi_harmonic: self.settings.method,
            cutoff: self.settings.cutoff,
            symmetry_number: sigma.max(1),
            multiplicity: multiplicity.max(1),
            concentration: c.concentration,
            ..ThermoSettings::default()
        };
        let modes = NormalModes {
            wavenumbers: frequencies.to_vec(),
            curvatures: Vec::new(),
            displacements: Vec::new(),
            reduced_masses: Vec::new(),
            force_constants: Vec::new(),
            temperatures: Vec::new(),
        };
        Ok(thermochemistry(system, &modes, energy, &settings)?
            .gibbs
            .total)
    }
}

impl RateTheory for Tst {
    fn name(&self) -> &str {
        "tst"
    }

    fn state_energy(&self, s: &KineticState, c: &Conditions) -> MoptResult<Option<f64>> {
        match (&s.system, &s.frequencies) {
            (Some(sys), Some(f)) => self
                .gibbs(sys, f, s.energy, s.multiplicity, s.symmetry_number, c)
                .map(Some),
            _ => Ok(None),
        }
    }

    fn bottleneck_energy(&self, b: &Bottleneck, c: &Conditions) -> MoptResult<Option<f64>> {
        match (&b.system, &b.frequencies) {
            (Some(sys), Some(f)) => self
                .gibbs(sys, f, b.energy, b.multiplicity, b.symmetry_number, c)
                .map(Some),
            _ => Ok(None),
        }
    }

    fn describe(&self, _c: &Conditions) -> String {
        let method = match self.settings.method {
            QuasiHarmonic::Grimme => "Grimme's quasi-RRHO",
            QuasiHarmonic::Truhlar => "Truhlar's raised frequencies",
            QuasiHarmonic::None => "RRHO",
        };
        format!(
            "free energy from frequencies and structure ({method}, cutoff {} cm^-1, symmetry \
             numbers {})",
            self.settings.cutoff,
            match self.settings.symmetry {
                SymmetryChoice::Auto => "from the point groups",
                SymmetryChoice::None => "1",
            }
        )
    }
}
