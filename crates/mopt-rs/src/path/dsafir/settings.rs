//! The settings of DS-AFIR: the method as the paper states it with the constants of the
//! manual of its authors' program ([`DsAfirSettings::paper`]), and the default of mopt-rs.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use crate::domain::Units;
use crate::path::string_nodes::NodeStepSettings;

/// `delta` of the paper's procedure as its authors' program documents its default (`0.03`,
/// atomic units: Hartree per Bohr); the paper itself gives values for the Muller-Brown
/// surface only.
pub const PAPER_FORCE: f64 = 0.03;

/// How an end takes its drive steps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DsIntegrator {
    /// The local quadratic approximation, the paper's integrator ([`super::lqa`]).
    Lqa,
    /// Steps of the fixed length straight down `grad F`.
    Euler,
}

/// Settings of DS-AFIR (molecules: Bohr and Hartree).
#[derive(Debug, Clone)]
pub struct DsAfirSettings {
    /// `delta`: the slope of the AFIR function along `u`.
    pub force: f64,
    /// Length of a step of an end (the arc length of an LQA step).
    pub step: f64,
    /// How a drive step is taken.
    pub integrator: DsIntegrator,
    /// With the LQA steps: the Hessian of the energy is exact at every this many points of
    /// an end (0: at its first point only) and updated in between.
    pub hessian_every: usize,
    /// With the LQA steps: add the second derivatives of the two distance terms of `F` to
    /// the Hessian (mopt-rs).
    pub afir_curvature: bool,
    /// Leave the second term of `F` out throughout (`Y = 1`: the end is drawn towards the
    /// other and never pushed from its reference; mopt-rs).
    pub attraction_only: bool,
    /// The ends have met when they are closer than this.
    pub separation: f64,
    /// Drive steps of the ends at most.
    pub max_steps: usize,
    /// Lay the other end onto the moving one before every step (mopt-rs).
    pub align: bool,
    /// Move the ends in turn instead of the lower one (mopt-rs).
    pub alternate: bool,
    /// L-BFGS steps down the AFIR function, at most `step` long, instead of Euler steps of
    /// the length `step` (mopt-rs).
    pub quasi_newton: bool,
    /// Steps of the moving end across its drive direction after each drive step (mopt-rs;
    /// 0: none).
    pub relax_steps: usize,
    /// The relaxation across the drive ends below this RMS of the gradient across it.
    pub relax_tolerance: f64,
    /// The steps of that relaxation.
    pub relax_steps_settings: NodeStepSettings,
}

impl DsAfirSettings {
    /// The method as the paper states it, with the constants the authors' manual documents
    /// (the path of the two ends; its relaxation is a step after the method,
    /// `method_relax`).
    pub fn paper() -> Self {
        let tenth = 0.1 * Units::ANGSTROM_TO_BOHR;
        Self {
            force: PAPER_FORCE,
            step: tenth,
            integrator: DsIntegrator::Lqa,
            hessian_every: 10,
            afir_curvature: false,
            attraction_only: false,
            separation: 0.12 * Units::ANGSTROM_TO_BOHR,
            max_steps: 2000,
            align: false,
            alternate: false,
            quasi_newton: false,
            relax_steps: 0,
            relax_tolerance: 1e-3,
            relax_steps_settings: NodeStepSettings {
                max_step: tenth,
                ..NodeStepSettings::default()
            },
        }
    }
}

/// The default of mopt-rs: the paper's procedure with the other end turned onto the moving
/// one before every step. With it the ends met on all four reactions measured, in 23 to 212
/// gradients; without it they met on two (`docs/source/reaction_paths.md`, "DS-AFIR: the paper and
/// the additions").
impl Default for DsAfirSettings {
    fn default() -> Self {
        Self {
            align: true,
            ..Self::paper()
        }
    }
}
