//! Vibrational analysis and thermochemistry from a Hessian: normal modes and harmonic
//! frequencies ([`normal_modes`]), and the thermodynamic functions of the ideal gas in the
//! rigid-rotor harmonic-oscillator approximation with quasi-harmonic corrections for low
//! frequencies ([`thermochemistry`]). Along a reaction path: the elements of the
//! reaction-path Hamiltonian ([`transverse_profile`]), the unified reaction valley approach
//! ([`internal_components`], [`reaction_phases`]), and variational transition state theory
//! with tunneling ([`free_energy_profile`], [`tunneling`]). For chosen atoms: the partial
//! and the effective Hessian ([`partial_hessian`]); for internal coordinates: their local
//! modes ([`local_modes`]). At a crossing of two states: the vibrations within the seam
//! ([`seam_analysis`]). In a periodic cell: phonons at any wave vector ([`phonon`]).
//!
//! The physical constants are the exact SI values and CODATA 2018, as in MultiOptPy's
//! `Parameters/unit_values.py`, so that frequencies and thermodynamic functions agree with
//! MultiOptPy to rounding where MultiOptPy is right.
//!
//! MultiOptPy: `Thermo/normal_mode_analyzer.py`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use crate::domain::Units;

mod local;
mod masses;
mod normal_modes;
mod partial;
mod path;
pub mod phonon;
mod qff;
mod report;
mod seam;
mod symmetry;
mod thermo;
mod urva;
mod vpt2;
mod vscf;
mod vtst;

pub use local::{local_modes, LocalMode, LocalModes};
pub use masses::MassChoice;
pub use normal_modes::{
    imaginary_wavenumbers, is_imaginary_curvature, normal_modes, normal_modes_with_masses,
    normal_modes_without, transverse_modes, transverse_modes_with_masses, Direction, NormalModes,
};
pub use partial::{
    partial_hessian, relaxed_stretch, PartialHessian, PartialKind, DEFAULT_FREE_DIRECTION,
};
pub use path::{
    gradient_alignment, gradient_alignment_with_masses, path_tangents, path_tangents_with_masses,
    transverse_profile, transverse_profile_with_masses, Bifurcation, PathPoint, PathProfile,
};
pub use qff::{ForceField, Qff, QffSettings, DEFAULT_STEP};
pub use report::{
    analyze, analyze_without, animations, normal_modes_text, thermochemistry_text,
    FrequencySettings, VibrationalAnalysis,
};
pub use seam::{seam_analysis, SeamAnalysis, SecondDirection};
pub use symmetry::{point_group, PointGroup};
pub use thermo::{
    rotational_constants_ghz, rotational_constants_with_masses, rotor_type, thermochemistry,
    EnthalpyTreatment, Parts, QuasiHarmonic, RotorMoment, RotorType, ThermoSettings,
    Thermochemistry,
};
pub use urva::{
    internal_components, path_primitives, reaction_phases, InternalComponents, ReactionPhase,
};
pub use vpt2::{vpt2, FermiResonance, Vpt2, Vpt2Settings};
pub use vscf::{vscf, Vscf, VscfSettings, DEFAULT_BASIS};
pub use vtst::{
    effective_mass_ratio, free_energy_profile, transmission, tunneling, variational_maximum,
    FreeEnergyProfile, Tunneling,
};

/// Boltzmann constant in Hartree/K.
pub fn boltzmann_hartree() -> f64 {
    Units::BOLTZMANN_HARTREE_PER_K
}

/// Frequency in Hz of a mass-weighted curvature of 1 Hartree / (Bohr^2 amu).
pub(crate) fn curvature_to_hz() -> f64 {
    Units::MW_CURVATURE_TO_SI.sqrt() / (2.0 * std::f64::consts::PI)
}
