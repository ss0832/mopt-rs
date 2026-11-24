//! Settings, paths, and events of the anharmonic downward distortion following.
//!
//! MultiOptPy: `OtherMethod/addf.py:ADDFlikeMethod` (`addf_config`, the `ADDthetaClass`
//! records).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::{Col, Mat};

use crate::domain::Units;
use crate::error::{MoptError, MoptResult};
use crate::irc::SphereSettings;

/// Settings of an ADDF run (lengths in Bohr for structures, in the unit of the variables
/// otherwise).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AddfSettings {
    /// Displacement along the stiffest mode on the first sphere, where the ADDs are
    /// detected (0.03 Angstrom).
    pub initial_length: f64,
    /// Displacement along the stiffest mode by which the radius grows per sphere
    /// (`-addf_step`; 0.1 Angstrom).
    pub step_length: f64,
    /// Spheres at most (`-addf_num`).
    pub max_spheres: usize,
    /// Paths followed: the largest ADDs of the first sphere (`-addf_nadd`; 0: all).
    pub paths: usize,
    /// The minimization on a sphere.
    pub sphere: SphereSettings,
    /// Two maximal ADD points closer in angle than this (rad) are the same.
    pub duplicate_angle: f64,
    /// Largest component of the gradient at the starting minimum.
    pub gradient_tolerance: f64,
}

impl Default for AddfSettings {
    fn default() -> Self {
        Self {
            initial_length: 0.03 * Units::ANGSTROM_TO_BOHR,
            step_length: 0.1 * Units::ANGSTROM_TO_BOHR,
            max_spheres: 300,
            paths: 5,
            sphere: SphereSettings::default(),
            duplicate_angle: 0.01,
            gradient_tolerance: 1e-3,
        }
    }
}

impl AddfSettings {
    /// Check the settings.
    pub fn validate(&self) -> MoptResult<()> {
        let positive = [self.initial_length, self.step_length, self.duplicate_angle];
        if positive.iter().any(|v| !(v.is_finite() && *v > 0.0)) || self.max_spheres == 0 {
            return Err(MoptError::config(
                "ADDF: the lengths, the angle, and the number of spheres must be positive",
            ));
        }
        if self.gradient_tolerance.is_nan() || self.gradient_tolerance <= 0.0 {
            return Err(MoptError::config(
                "ADDF: the gradient tolerance must be positive",
            ));
        }
        Ok(())
    }
}

/// The maximal ADD point of a path on one sphere.
#[derive(Debug, Clone)]
pub struct AddfPoint {
    /// Sphere (0: the sphere on which the ADDs were detected).
    pub sphere: usize,
    /// Radius in scaled coordinates.
    pub radius: f64,
    /// Scaled coordinates.
    pub q: Col<f64>,
    /// The point.
    pub x: Col<f64>,
    /// Energy (without the elimination functions).
    pub energy: f64,
    /// The ADD, `E - E0 - R^2 / 2`.
    pub add: f64,
    /// The ADD with the elimination functions of the larger ADDs of the sphere.
    pub add_ioe: f64,
    /// Evaluations on the sphere.
    pub evaluations: usize,
}

/// What became of a path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PathState {
    /// It is followed.
    Active,
    /// Its energy passed a maximum; the point of this index of the path is the estimate of
    /// the transition state.
    TransitionState(usize),
    /// It ran into the path of this number on a sphere.
    Merged(usize),
    /// Its structure fell apart into fragments (a dissociation channel).
    Dissociated,
    /// The surface is not defined where it went (an energy or a gradient that is not finite,
    /// such as the overflow of a growing exponential of a model surface far from its region).
    Undefined,
    /// The spheres were used up.
    SphereLimit,
}

/// One followed ADD.
#[derive(Debug, Clone)]
pub struct AddfPath {
    /// Number of the path (by its ADD on the first sphere, largest first).
    pub id: usize,
    /// The mode and sign it started from.
    pub mode: (usize, f64),
    /// Its points, sphere by sphere.
    pub points: Vec<AddfPoint>,
    /// What became of it.
    pub state: PathState,
    /// The Hessian of the minimizations on the spheres (scaled coordinates), carried along.
    pub(crate) hessian: Mat<f64>,
}

impl AddfPath {
    /// The last point.
    pub fn last(&self) -> Option<&AddfPoint> {
        self.points.last()
    }

    /// The estimate of the transition state, once the path has passed one.
    pub fn transition_state(&self) -> Option<&AddfPoint> {
        match self.state {
            PathState::TransitionState(k) => self.points.get(k),
            _ => None,
        }
    }
}

/// An ADD direction of the first sphere.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Candidate {
    /// Mode (by eigenvalue, lowest first).
    pub mode: usize,
    /// Direction along it (+1 or -1).
    pub sign: f64,
    /// Eigenvalue of the mode.
    pub eigenvalue: f64,
    /// The ADD there.
    pub add: f64,
}

/// What a `tell` found out.
#[derive(Debug, Clone)]
pub enum AddfEvent {
    /// The minimum and its scaled coordinates: the eigenvalues of the modes, the radii of
    /// the first sphere and of the steps.
    Started {
        /// Energy of the minimum.
        energy: f64,
        /// Eigenvalues of the modes, ascending.
        eigenvalues: Vec<f64>,
        /// Radius of the first sphere.
        first_radius: f64,
        /// Growth of the radius per sphere.
        radius_step: f64,
    },
    /// The ADDs along every mode on the first sphere, and the paths that are followed.
    Scanned {
        /// All directions, by ADD (largest first).
        candidates: Vec<Candidate>,
        /// How many of them are followed.
        followed: usize,
    },
    /// A path found its maximal ADD point on a sphere.
    Point {
        /// Path number.
        path: usize,
        /// The point.
        point: AddfPoint,
    },
    /// A path ran into another one.
    Merged {
        /// Path number.
        path: usize,
        /// The path it ran into.
        into: usize,
    },
    /// The structure of a path fell apart (a dissociation channel).
    Dissociated {
        /// Path number.
        path: usize,
    },
    /// A path reached a point where the surface is not defined (its energy or gradient is not
    /// finite); it is followed no further.
    Undefined {
        /// Path number.
        path: usize,
    },
    /// A path passed a maximum of its energy.
    TransitionState {
        /// Path number.
        path: usize,
        /// The highest point.
        point: AddfPoint,
    },
    /// A sphere is done.
    SphereDone {
        /// Sphere.
        sphere: usize,
        /// Paths still followed.
        active: usize,
    },
}

/// Why an ADDF run ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AddfEnd {
    /// Every path passed a transition state or ran into another one.
    AllPathsEnded,
    /// The spheres are used up.
    SphereLimit,
}
