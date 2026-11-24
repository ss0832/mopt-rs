//! Images between two structures made without a calculation: the first chain of a
//! relaxation.
//!
//! A straight line between two structures passes through places no path does: atoms run
//! into each other, and a group that turns shrinks on the way. The methods here put the
//! images where the distances between the atoms change evenly instead.
//!
//! | Name | Images |
//! |---|---|
//! | `linear` | on the straight line (what MultiOptPy's `-p` does) |
//! | `lst` | linear synchronous transit: each image fits the interpolated distances by least squares ([`lst`]) |
//! | `idpp` | image dependent pair potential: the straight line relaxed across the path on a function of the interpolated distances ([`idpp`]) |
//! | `sidpp` | sequential IDPP: the images are put one after the other from both ends, each time relaxed ([`idpp`]) |
//! | `geodesic` | the shortest path in scaled interatomic distances ([`geodesic`]) |
//! | `fbenm`, `cfbenm` | the straight line relaxed on the flat-bottom elastic network model, without and with its correlated term ([`fbenm`]) |
//!
//! The names differ between programs: what some call LST is the straight line here.
//!
//! MultiOptPy: `Potential/idpp.py:IDPP` (`-idpp`), `Interpolation/geodesic_interpolation.py`
//! (`-adg`), `Potential/idpp.py:CFB_ENM` (`-cfbenm`); the linear synchronous transit and
//! the sequential IDPP have no counterpart.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

pub mod fbenm;
pub mod geodesic;
pub mod idpp;
pub mod least_squares;
pub mod lst;
pub mod pairs;

use faer::Col;

use crate::domain::{Element, PeriodicCell};
use crate::error::{MoptError, MoptResult};

/// How the images between two structures are made.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FirstPath {
    /// On the straight line.
    #[default]
    Linear,
    /// Linear synchronous transit.
    Lst,
    /// Image dependent pair potential.
    Idpp,
    /// Sequential image dependent pair potential.
    SequentialIdpp,
    /// Geodesic in scaled interatomic distances.
    Geodesic,
    /// Flat-bottom elastic network model.
    Fbenm,
    /// Correlated flat-bottom elastic network model.
    Cfbenm,
}

impl FirstPath {
    /// The names, in the order of the table of the module documentation.
    pub const NAMES: [&'static str; 7] = [
        "linear", "lst", "idpp", "sidpp", "geodesic", "fbenm", "cfbenm",
    ];

    /// The method called `name` (`s-idpp` and `sequential_idpp` are `sidpp`).
    pub fn from_name(name: &str) -> MoptResult<Self> {
        match name.to_ascii_lowercase().replace(['-', '_'], "").as_str() {
            "linear" => Ok(Self::Linear),
            "lst" => Ok(Self::Lst),
            "idpp" => Ok(Self::Idpp),
            "sidpp" | "sequentialidpp" => Ok(Self::SequentialIdpp),
            "geodesic" => Ok(Self::Geodesic),
            "fbenm" => Ok(Self::Fbenm),
            "cfbenm" => Ok(Self::Cfbenm),
            _ => Err(MoptError::parse(format!(
                "no way of making the first images called '{name}' ({})",
                Self::NAMES.join(", ")
            ))),
        }
    }

    /// Its name.
    pub fn name(self) -> &'static str {
        match self {
            Self::Linear => "linear",
            Self::Lst => "lst",
            Self::Idpp => "idpp",
            Self::SequentialIdpp => "sidpp",
            Self::Geodesic => "geodesic",
            Self::Fbenm => "fbenm",
            Self::Cfbenm => "cfbenm",
        }
    }
}

/// What the structures of a path have in common besides their atoms.
#[derive(Debug, Clone, Default)]
pub struct PathFrame {
    /// The elements of the atoms (the geodesic uses their radii; empty: every pair counts
    /// as it is. The flat-bottom elastic network needs them).
    pub elements: Vec<Element>,
    /// The periodic cell: distances are those of the nearest images.
    pub cell: Option<PeriodicCell>,
    /// Coordinates that keep their values (the three of every fixed atom).
    pub fixed_components: Vec<usize>,
}

/// How long and how far the images are relaxed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PathSettings {
    /// Most iterations of one relaxation.
    pub max_iterations: usize,
    /// Largest component of the force across the path at which a relaxation on the image
    /// dependent pair potential ends (1/Bohr^3).
    pub force_tolerance: f64,
}

impl Default for PathSettings {
    fn default() -> Self {
        Self {
            max_iterations: 500,
            force_tolerance: idpp::FORCE_TOLERANCE,
        }
    }
}

/// The images between two structures and how they were found.
#[derive(Debug, Clone)]
pub struct Interpolated {
    /// The images between the two structures, in order (without the structures).
    pub images: Vec<Col<f64>>,
    /// Iterations of all relaxations together (0 for the straight line).
    pub iterations: usize,
    /// Whether every relaxation ended at its tolerance.
    pub converged: bool,
}

/// `count` images between the structures `first` and `last` (three coordinates per atom,
/// Bohr), made by `method`.
pub fn images_between(
    method: FirstPath,
    first: &Col<f64>,
    last: &Col<f64>,
    count: usize,
    frame: &PathFrame,
    settings: &PathSettings,
) -> MoptResult<Interpolated> {
    if first.nrows() != last.nrows() || !first.nrows().is_multiple_of(3) {
        return Err(MoptError::config(format!(
            "two structures of {} and {} coordinates",
            first.nrows(),
            last.nrows()
        )));
    }
    if let Some(&k) = frame.fixed_components.iter().find(|k| **k >= first.nrows()) {
        return Err(MoptError::config(format!(
            "fixed coordinate {k} of {} coordinates",
            first.nrows()
        )));
    }
    match method {
        FirstPath::Linear => Ok(Interpolated {
            images: on_the_line(first, last, count),
            iterations: 0,
            converged: true,
        }),
        FirstPath::Lst => lst::lst_images(first, last, count, frame, settings),
        FirstPath::Idpp => idpp::idpp_images(first, last, count, frame, settings),
        FirstPath::SequentialIdpp => {
            idpp::sequential_idpp_images(first, last, count, frame, settings)
        }
        FirstPath::Geodesic => geodesic::geodesic_images(first, last, count, frame, settings),
        FirstPath::Fbenm => fbenm::fbenm_images(first, last, count, frame, settings, false),
        FirstPath::Cfbenm => fbenm::fbenm_images(first, last, count, frame, settings, true),
    }
}

/// Largest shift of a coordinate by [`nudged`] (Bohr; 0.01 Angstrom).
pub const NUDGE: f64 = 0.018_897_261;

/// `x` with every coordinate that is not fixed shifted a little by a fixed pattern.
///
/// The methods start from structures on a straight line. Where that line has a symmetry
/// that the path must break (an atom that passes straight through a bond, three atoms on
/// one line), the forces across it are zero and the images would stay; the shift is small
/// against any distance between atoms and the same for the same input.
pub fn nudged(x: &Col<f64>, frame: &PathFrame) -> Col<f64> {
    Col::from_fn(x.nrows(), |k| {
        if frame.fixed_components.contains(&k) {
            x[k]
        } else {
            x[k] + NUDGE * (0.618_033_988_75 * (k + 1) as f64).fract()
        }
    })
}

/// `count` images on the straight line between `first` and `last`.
pub fn on_the_line(first: &Col<f64>, last: &Col<f64>, count: usize) -> Vec<Col<f64>> {
    (1..=count)
        .map(|k| {
            let share = k as f64 / (count + 1) as f64;
            first + share * (last - first)
        })
        .collect()
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests;
