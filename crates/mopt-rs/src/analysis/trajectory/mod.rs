//! Analysis of a molecular dynamics trajectory (of `mdmain` or of any other program): the
//! statistics of a time series with block averages and their errors, the radial
//! distribution function of every pair of elements (minimum image in a cell), the mean
//! squared displacement and the diffusion coefficient, the velocity autocorrelation
//! function and the vibrational density of states, the RMSD to the first frame and the
//! RMSF of every atom, distances, angles, and dihedral angles over time, and the projection
//! on the principal components of the motion ([`crate::analysis::path_map`]).
//!
//! Everything here works on structures in memory ([`MdTrajectory`]); reading the files of a
//! run and writing the results is `workflow::md_analysis`.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

pub mod dynamics;
pub mod rdf;
pub mod statistics;
pub mod structure;
#[cfg(test)]
mod tests;

pub use dynamics::{
    mean_squared_displacement, peaks, velocity_autocorrelation, vibrational_density_of_states,
    Diffusion, Msd, MsdSettings, Vacf, Vdos, VdosSettings,
};
pub use rdf::{radial_distribution, Rdf, RdfPair, RdfSettings};
pub use statistics::{statistics, BlockLevel, Statistics};
pub use structure::{fluctuations, geometry_series, trajectory_pca, Fluctuations, Measure};

use faer::Col;

use crate::domain::{AtomId, Element, MolecularSystem, Trajectory, Units};
use crate::error::{MoptError, MoptResult};

/// Velocities of one frame: Bohr per atomic unit of time, one row per atom.
pub type FrameVelocities = Vec<[f64; 3]>;

/// Frames of a trajectory with their times (and, for the correlation functions of the
/// velocities, the velocities of the atoms).
#[derive(Debug, Clone)]
pub struct MdTrajectory {
    systems: Vec<MolecularSystem>,
    times: Vec<f64>,
    velocities: Option<Vec<FrameVelocities>>,
}

/// The time of a frame from its comment (`time=<fs>` as `mdmain` writes it).
pub fn time_in_comment(comment: &str) -> Option<f64> {
    comment
        .split_whitespace()
        .find_map(|t| t.strip_prefix("time="))
        .and_then(|v| v.trim_end_matches([',', ';']).parse::<f64>().ok())
}

impl MdTrajectory {
    /// Frames `systems` (the same atoms in the same order) at `times_fs` (fs, increasing).
    pub fn new(systems: Vec<MolecularSystem>, times_fs: Vec<f64>) -> MoptResult<Self> {
        let first = systems
            .first()
            .ok_or_else(|| MoptError::config("a trajectory needs at least one frame"))?;
        if systems.iter().any(|s| s.elements != first.elements) {
            return Err(MoptError::config(
                "the frames of a trajectory must have the same atoms in the same order",
            ));
        }
        if times_fs.len() != systems.len() {
            return Err(MoptError::config(format!(
                "{} frames but {} times",
                systems.len(),
                times_fs.len()
            )));
        }
        if times_fs.iter().any(|t| !t.is_finite()) || times_fs.windows(2).any(|w| w[1] <= w[0]) {
            return Err(MoptError::config(
                "the times of the frames must be finite and increasing",
            ));
        }
        Ok(Self {
            systems,
            times: times_fs,
            velocities: None,
        })
    }

    /// The frames of `trajectory` at the times in their comments (`time=<fs>`), or, when a
    /// comment has none, at `interval_fs` times the frame index.
    pub fn from_trajectory(trajectory: &Trajectory, interval_fs: f64) -> MoptResult<Self> {
        let all: Option<Vec<f64>> = trajectory
            .frames
            .iter()
            .map(|f| time_in_comment(&f.system.comment))
            .collect();
        let times = all.unwrap_or_else(|| {
            (0..trajectory.len())
                .map(|k| k as f64 * interval_fs)
                .collect()
        });
        let systems = trajectory.frames.iter().map(|f| f.system.clone()).collect();
        Self::new(systems, times)
    }

    /// The same frames with the velocities of the atoms (Bohr per atomic unit of time), one
    /// vector of `3 n` per frame.
    pub fn with_velocities(mut self, velocities: &[Col<f64>]) -> MoptResult<Self> {
        if velocities.len() != self.systems.len() {
            return Err(MoptError::config(format!(
                "{} frames but velocities of {}",
                self.systems.len(),
                velocities.len()
            )));
        }
        let n = self.n_atoms();
        let mut rows = Vec::with_capacity(velocities.len());
        for v in velocities {
            if v.nrows() != 3 * n || v.iter().any(|x| !x.is_finite()) {
                return Err(MoptError::config(format!(
                    "the velocities of a frame must be {} finite numbers",
                    3 * n
                )));
            }
            rows.push(
                (0..n)
                    .map(|i| [v[3 * i], v[3 * i + 1], v[3 * i + 2]])
                    .collect(),
            );
        }
        self.velocities = Some(rows);
        Ok(self)
    }

    /// Number of frames.
    pub fn len(&self) -> usize {
        self.systems.len()
    }

    /// Whether there are no frames (never: a trajectory has at least one).
    pub fn is_empty(&self) -> bool {
        self.systems.is_empty()
    }

    /// Number of atoms.
    pub fn n_atoms(&self) -> usize {
        self.systems.first().map_or(0, MolecularSystem::n_atoms)
    }

    /// The structures.
    pub fn systems(&self) -> &[MolecularSystem] {
        &self.systems
    }

    /// The times (fs).
    pub fn times(&self) -> &[f64] {
        &self.times
    }

    /// The velocities, when given.
    pub fn velocities(&self) -> Option<&[FrameVelocities]> {
        self.velocities.as_deref()
    }

    /// The elements of the atoms.
    pub fn elements(&self) -> &[Element] {
        self.systems.first().map_or(&[], |s| s.elements.as_slice())
    }

    /// The masses of the atoms (amu).
    pub fn masses(&self) -> Vec<f64> {
        self.elements().iter().map(|e| e.mass_amu()).collect()
    }

    /// The elements in the order they first appear, with the atoms of each.
    pub fn species(&self) -> Vec<(Element, Vec<usize>)> {
        let mut out: Vec<(Element, Vec<usize>)> = Vec::new();
        for (i, e) in self.elements().iter().enumerate() {
            match out.iter_mut().find(|(f, _)| f == e) {
                Some((_, atoms)) => atoms.push(i),
                None => out.push((*e, vec![i])),
            }
        }
        out
    }

    /// Whether the frames have a cell periodic along every axis.
    pub fn is_periodic(&self) -> bool {
        self.systems.iter().all(|s| {
            s.active_cell()
                .is_some_and(|c| c.periodic_axes.iter().all(|p| *p))
        })
    }

    /// The frames from time `from_fs` on (all frames when none is that late is an error).
    pub fn from_time(&self, from_fs: f64) -> MoptResult<Self> {
        let start = self.times.iter().position(|t| *t >= from_fs - 1e-9);
        let Some(start) = start else {
            return Err(MoptError::config(format!(
                "no frame at or after {from_fs} fs (the last is at {} fs)",
                self.times.last().copied().unwrap_or(0.0)
            )));
        };
        Ok(Self {
            systems: self.systems[start..].to_vec(),
            times: self.times[start..].to_vec(),
            velocities: self.velocities.as_ref().map(|v| v[start..].to_vec()),
        })
    }

    /// The interval of the frames (fs), when they are evenly spaced (to a part in 10^6);
    /// the correlation functions need that.
    pub fn interval(&self) -> MoptResult<f64> {
        if self.len() < 2 {
            return Err(MoptError::config(
                "a time correlation needs at least two frames",
            ));
        }
        let dt = (self.times[self.len() - 1] - self.times[0]) / (self.len() - 1) as f64;
        let even = self
            .times
            .windows(2)
            .all(|w| ((w[1] - w[0]) - dt).abs() <= 1e-6 * dt.max(1e-12) + 1e-9);
        if !even {
            return Err(MoptError::config(
                "the frames are not evenly spaced in time",
            ));
        }
        Ok(dt)
    }

    /// The frames without the last one when it is closer to the one before than the others
    /// are to theirs (a run writes its last step whether or not it falls on the interval of
    /// the frames); whether it was dropped.
    pub fn without_short_last_interval(self) -> (Self, bool) {
        let n = self.len();
        if n < 3 {
            return (self, false);
        }
        let before = self.times[n - 2] - self.times[n - 3];
        let last = self.times[n - 1] - self.times[n - 2];
        if (last - before).abs() <= 1e-6 * before.max(1e-12) + 1e-9 {
            return (self, false);
        }
        let mut out = self;
        out.systems.pop();
        out.times.pop();
        if let Some(v) = out.velocities.as_mut() {
            v.pop();
        }
        (out, true)
    }

    /// The position of atom `i` in frame `t` (Bohr).
    pub(crate) fn position(&self, t: usize, i: usize) -> [f64; 3] {
        self.systems[t].position(AtomId::new(i)).value()
    }

    /// `r_j - r_i` in frame `t` (Bohr), by the minimum image in a cell.
    pub(crate) fn delta(&self, t: usize, i: usize, j: usize) -> [f64; 3] {
        let (p, q) = (self.position(t, i), self.position(t, j));
        match self.systems[t].active_cell() {
            Some(cell) => cell.mic_delta(q, p),
            None => std::array::from_fn(|c| q[c] - p[c]),
        }
    }

    /// The positions with the jumps through the faces of the cell taken out: each atom moves
    /// from one frame to the next by the nearest image of its displacement (the frames must
    /// be close enough for that: less than half the cell apart). Without a cell, the
    /// positions as they are. Bohr, `[frame][atom]`.
    pub fn unwrapped(&self) -> Vec<Vec<[f64; 3]>> {
        let n = self.n_atoms();
        let mut out: Vec<Vec<[f64; 3]>> = Vec::with_capacity(self.len());
        out.push((0..n).map(|i| self.position(0, i)).collect());
        for t in 1..self.len() {
            let cell = self.systems[t].active_cell();
            let row: Vec<[f64; 3]> = (0..n)
                .map(|i| {
                    let (now, before) = (self.position(t, i), self.position(t - 1, i));
                    let step = match cell {
                        Some(c) => c.mic_delta(now, before),
                        None => std::array::from_fn(|k| now[k] - before[k]),
                    };
                    let last = out[t - 1][i];
                    std::array::from_fn(|k| last[k] + step[k])
                })
                .collect();
            out.push(row);
        }
        out
    }

    /// [`Self::unwrapped`] with the displacement of the centre of mass of all atoms from
    /// the first frame subtracted in every frame.
    pub fn unwrapped_without_drift(&self) -> Vec<Vec<[f64; 3]>> {
        let masses = self.masses();
        let total: f64 = masses.iter().sum();
        let mut u = self.unwrapped();
        let centre = |row: &[[f64; 3]]| -> [f64; 3] {
            std::array::from_fn(|c| {
                row.iter().zip(&masses).map(|(p, m)| m * p[c]).sum::<f64>() / total
            })
        };
        let start = centre(&u[0]);
        for row in &mut u {
            let now = centre(row);
            for p in row.iter_mut() {
                for c in 0..3 {
                    p[c] -= now[c] - start[c];
                }
            }
        }
        u
    }
}

/// Bohr to Angstrom.
pub(crate) const B2A: f64 = Units::BOHR_TO_ANGSTROM;
