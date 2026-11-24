//! Settings of the improved elastic image pair method, with MultiOptPy's values.
//!
//! MultiOptPy: `ieip.py:IEIPConfig` and the constants of
//! `OtherMethod/elastic_image_pair.py:ElasticImagePair.microiteration`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use crate::error::{MoptError, MoptResult};

/// Longest displacement made of one force (Bohr; MultiOptPy `displacement_limit`).
pub const DISPLACEMENT_LIMIT: f64 = 0.04;
/// MultiOptPy's `L_covergence` (Bohr); the separation asked for never falls below
/// `L_covergence - 0.01`.
pub const SEPARATION_FLOOR_BASE: f64 = 0.03;
/// The pair has converged when the images are closer than this after a step (Bohr;
/// MultiOptPy `img_distance_convage_criterion`).
pub const SEPARATION_TOLERANCE: f64 = 0.15;
/// Share of the separation the attraction of the images asks for in one iteration.
pub const SEPARATION_SHARE: f64 = 0.9;
/// Factor of `beta` whenever the images moved apart in an iteration.
pub const BETA_GROWTH: f64 = 1.02;
/// AdaBelief: decay of the mean of the displacements.
pub const ADABELIEF_BETA_M: f64 = 0.9;
/// AdaBelief: decay of the variance of the displacements about their mean.
pub const ADABELIEF_BETA_V: f64 = 0.999;
/// AdaBelief: learning rate.
pub const ADABELIEF_RATE: f64 = 0.01;
/// AdaBelief: added to the square root of the variance.
pub const ADABELIEF_EPSILON: f64 = 1e-8;

/// Settings of the relaxation of the images on hyperspheres (`-mi`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MicroSettings {
    /// Relaxation steps per iteration (`-mi`; 0: none).
    pub steps: usize,
    /// First trust radius of a relaxation (0.015).
    pub initial_trust: f64,
    /// Smallest and largest trust radius (0.01 and 0.1).
    pub min_trust: f64,
    /// Largest trust radius.
    pub max_trust: f64,
    /// Weight of the length and of the angles in the length of a step (1.0 and 0.85).
    pub radial_weight: f64,
    /// Weight of the angles in the length of a step.
    pub angular_weight: f64,
    /// A relaxation ends when the length of the force in hyperspherical coordinates of both
    /// images is below this (0.01).
    pub force_tolerance: f64,
    /// Pairs of the L-BFGS memory (10).
    pub memory: usize,
}

impl Default for MicroSettings {
    fn default() -> Self {
        Self {
            steps: 0,
            initial_trust: 0.015,
            min_trust: 0.01,
            max_trust: 0.1,
            radial_weight: 1.0,
            angular_weight: 0.85,
            force_tolerance: 0.01,
            memory: 10,
        }
    }
}

/// Settings of an iEIP run.
#[derive(Debug, Clone)]
pub struct IeipSettings {
    /// Largest number of iterations (`-ns`; MultiOptPy's default 999).
    pub max_iterations: usize,
    /// `beta` of the attraction of the images (`-beta`; 1.0).
    pub beta: f64,
    /// Relaxation of the images on hyperspheres around where they were (`-mi`).
    pub micro: MicroSettings,
    /// The coordinates of that relaxation (needed when `micro.steps > 0`; MultiOptPy's are
    /// [`super::micro::MicroCoordinates::polar`]).
    pub micro_coordinates: Option<super::micro::MicroCoordinates>,
    /// Turn the first image onto the second (both centred) at the start and after every
    /// step, as MultiOptPy does (only for surfaces that do not change with rigid motions).
    pub align: bool,
    /// Variables that keep their values (fixed atoms: their three coordinates).
    pub frozen: Vec<usize>,
}

impl Default for IeipSettings {
    fn default() -> Self {
        Self {
            max_iterations: 999,
            beta: 1.0,
            micro: MicroSettings::default(),
            micro_coordinates: None,
            align: true,
            frozen: Vec::new(),
        }
    }
}

impl IeipSettings {
    /// Check the settings.
    pub fn validate(&self) -> MoptResult<()> {
        if self.max_iterations == 0 {
            return Err(MoptError::config(
                "iEIP: the number of iterations must be positive",
            ));
        }
        if !(self.beta.is_finite() && self.beta > 0.0) {
            return Err(MoptError::config(format!(
                "iEIP: beta must be positive, got {}",
                self.beta
            )));
        }
        let m = &self.micro;
        let positive = [m.initial_trust, m.min_trust, m.max_trust, m.force_tolerance];
        if positive.iter().any(|v| !(v.is_finite() && *v > 0.0)) || m.min_trust > m.max_trust {
            return Err(MoptError::config(
                "iEIP: the trust radii and the tolerance of the relaxation must be positive",
            ));
        }
        if m.steps > 0 && self.micro_coordinates.is_none() {
            return Err(MoptError::unsupported(
                "iEIP: the relaxation of the images (-mi) needs structures (its coordinates \
                 are those of the atoms about where an image was)",
            ));
        }
        Ok(())
    }
}
