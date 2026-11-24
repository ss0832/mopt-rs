//! The masses a vibrational analysis weights with: the standard atomic weights (the default,
//! those of the molecular dynamics) or the masses of the most abundant isotopes (those of
//! MultiOptPy, and of the IRC option `masses=isotope`). The choice is an explicit input of the
//! analysis, so that the frequencies along a path come from the masses the path was traced
//! with; the reports name it.
//!
//! The isotope masses are those of `irc::most_abundant_isotope_masses` (one table for the
//! crate, `parameters::isotope_masses`); elements beyond its end are refused, not silently
//! given their standard weight. Masses of other isotopes (deuterium, 13C) are not tabulated in
//! the crate and cannot be chosen.
//!
//! MultiOptPy: `Parameters/parameter.py:atomic_mass` (the most abundant isotopes, which both
//! its IRC and its vibrational analysis, `Thermo/normal_mode_analyzer.py:
//! MolecularVibrations._get_atomic_mass`, weight with); the standard weights, the default of
//! mopt-rs, are new.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use crate::domain::MolecularSystem;
use crate::error::{MoptError, MoptResult};

/// Which masses a vibrational analysis uses (`-freq_masses`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MassChoice {
    /// The standard atomic weights (`domain::Element::mass_amu`): the default.
    #[default]
    Standard,
    /// The masses of the most abundant isotopes.
    Isotope,
}

impl MassChoice {
    /// `standard` or `isotope` (the words of the IRC option `masses`).
    pub fn parse(name: &str) -> MoptResult<Self> {
        match name.trim().to_ascii_lowercase().as_str() {
            "standard" => Ok(Self::Standard),
            "isotope" => Ok(Self::Isotope),
            other => Err(MoptError::config(format!(
                "-freq_masses is standard or isotope, got '{other}'"
            ))),
        }
    }

    /// The name, as [`Self::parse`] takes it.
    pub fn name(self) -> &'static str {
        match self {
            Self::Standard => "standard",
            Self::Isotope => "isotope",
        }
    }

    /// A line for the header of a report.
    pub fn describe(self) -> &'static str {
        match self {
            Self::Standard => "Masses: standard atomic weights",
            Self::Isotope => "Masses: most abundant isotopes",
        }
    }

    /// The masses of the atoms of `sys` (amu).
    pub fn masses(self, sys: &MolecularSystem) -> MoptResult<Vec<f64>> {
        match self {
            Self::Standard => Ok(sys.masses_amu()),
            Self::Isotope => crate::irc::most_abundant_isotope_masses(sys),
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::domain::{positions, Angstrom, Element, SpinCharge};

    #[test]
    fn the_two_tables_and_their_names() {
        let sys = MolecularSystem::new(
            vec![Element::O, Element::H],
            &positions::<Angstrom>(&[[0.0; 3], [0.0, 0.0, 0.97]]),
            SpinCharge::default(),
            None,
        )
        .unwrap();
        let standard = MassChoice::Standard.masses(&sys).unwrap();
        let isotope = MassChoice::Isotope.masses(&sys).unwrap();
        assert_eq!(standard, sys.masses_amu());
        // 16O and 1H against 15.999 and 1.008.
        assert!((isotope[0] - 15.994_914_619_57).abs() < 1e-9);
        assert!(isotope[1] < standard[1] && isotope[0] < standard[0]);
        assert_eq!(MassChoice::parse(" Isotope ").unwrap(), MassChoice::Isotope);
        assert_eq!(MassChoice::parse("standard").unwrap().name(), "standard");
        assert!(MassChoice::parse("deuterium").is_err());
        assert_eq!(MassChoice::default(), MassChoice::Standard);
    }
}
