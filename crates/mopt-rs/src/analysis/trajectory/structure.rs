//! The structure along a trajectory: RMSD to the first frame, RMSF of every atom,
//! distances, angles, and dihedral angles of chosen atoms, and the principal components of
//! the motion.
//!
//! Without a cell, every frame is superposed on the first one (mass-weighted Kabsch
//! superposition, [`kabsch_align_bohr`]) and the RMSD is the root mean square of the
//! displacements of the atoms (not weighted). In a cell, rotations are not free: the
//! positions are taken with the jumps through the faces taken out and the drift of the
//! centre of mass subtracted. The RMSF of an atom is the root mean square of its
//! displacements from its mean position in the superposed frames. The principal components
//! are those of the mass-weighted coordinates of the superposed frames
//! ([`principal_components`] of `analysis::path_map`).
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Mat;
use rayon::prelude::*;

use super::{MdTrajectory, B2A};
use crate::analysis::path_map::{principal_components, PrincipalComponents};
use crate::coord::alignment::kabsch_align_bohr;
use crate::domain::{AtomId, Units};
use crate::error::{MoptError, MoptResult};

/// RMSD and RMSF.
#[derive(Debug, Clone, PartialEq)]
pub struct Fluctuations {
    /// RMSD of every frame to the first (Angstrom).
    pub rmsd: Vec<f64>,
    /// RMSF of every atom (Angstrom).
    pub rmsf: Vec<f64>,
}

/// The frames superposed on the first (Bohr, `[frame][atom]`).
fn superposed(traj: &MdTrajectory) -> MoptResult<Vec<Vec<[f64; 3]>>> {
    let n = traj.n_atoms();
    if traj.systems().iter().any(|s| s.active_cell().is_some()) {
        return Ok(traj.unwrapped_without_drift());
    }
    let first = &traj.systems()[0];
    traj.systems()
        .par_iter()
        .map(|s| {
            let (aligned, _) = kabsch_align_bohr(s, first, true)?;
            Ok((0..n)
                .map(|i| aligned.position(AtomId::new(i)).value())
                .collect())
        })
        .collect()
}

fn squared(a: [f64; 3], b: [f64; 3]) -> f64 {
    (0..3).map(|c| (a[c] - b[c]).powi(2)).sum()
}

/// RMSD to the first frame and RMSF of every atom.
pub fn fluctuations(traj: &MdTrajectory) -> MoptResult<Fluctuations> {
    let u = superposed(traj)?;
    let n = traj.n_atoms();
    let frames = u.len() as f64;
    let rmsd = u
        .iter()
        .map(|row| {
            let sum: f64 = row.iter().zip(&u[0]).map(|(p, q)| squared(*p, *q)).sum();
            libm::sqrt(sum / n as f64) * B2A
        })
        .collect();
    let rmsf = (0..n)
        .map(|i| {
            let mean: [f64; 3] =
                std::array::from_fn(|c| u.iter().map(|row| row[i][c]).sum::<f64>() / frames);
            let sum: f64 = u.iter().map(|row| squared(row[i], mean)).sum();
            libm::sqrt(sum / frames) * B2A
        })
        .collect();
    Ok(Fluctuations { rmsd, rmsf })
}

/// The principal components of the mass-weighted coordinates (Angstrom amu^(1/2)) of the
/// superposed frames (at least two), at most `components` of them.
pub fn trajectory_pca(traj: &MdTrajectory, components: usize) -> MoptResult<PrincipalComponents> {
    let u = superposed(traj)?;
    let roots: Vec<f64> = traj.masses().iter().map(|m| libm::sqrt(*m) * B2A).collect();
    let data = Mat::from_fn(u.len(), 3 * traj.n_atoms(), |t, v| {
        u[t][v / 3][v % 3] * roots[v / 3]
    });
    principal_components(&data, false, components.max(1))
}

/// A quantity of chosen atoms (0-based indices).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Measure {
    /// Distance of two atoms.
    Distance(usize, usize),
    /// Angle at the second of three atoms.
    Angle(usize, usize, usize),
    /// Dihedral angle of four atoms (from -180 to 180 degrees).
    Dihedral(usize, usize, usize, usize),
}

impl Measure {
    /// The measure of 2, 3, or 4 atoms (0-based).
    pub fn of(atoms: &[usize]) -> MoptResult<Self> {
        match *atoms {
            [i, j] => Ok(Self::Distance(i, j)),
            [i, j, k] => Ok(Self::Angle(i, j, k)),
            [i, j, k, l] => Ok(Self::Dihedral(i, j, k, l)),
            _ => Err(MoptError::config(format!(
                "a distance, angle, or dihedral angle needs 2, 3, or 4 atoms, not {}",
                atoms.len()
            ))),
        }
    }

    /// The atoms (0-based).
    pub fn atoms(&self) -> Vec<usize> {
        match *self {
            Self::Distance(i, j) => vec![i, j],
            Self::Angle(i, j, k) => vec![i, j, k],
            Self::Dihedral(i, j, k, l) => vec![i, j, k, l],
        }
    }

    /// Name with the 1-based atoms and the unit, as in a CSV header.
    pub fn name(&self) -> String {
        let atoms: Vec<String> = self.atoms().iter().map(|i| (i + 1).to_string()).collect();
        let (kind, unit) = match self {
            Self::Distance(..) => ("distance", "ang."),
            Self::Angle(..) => ("angle", "deg."),
            Self::Dihedral(..) => ("dihedral", "deg."),
        };
        format!("{kind} {} ({unit})", atoms.join("-"))
    }
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn value(traj: &MdTrajectory, t: usize, m: &Measure) -> f64 {
    match *m {
        Measure::Distance(i, j) => libm::sqrt(dot(traj.delta(t, i, j), traj.delta(t, i, j))) * B2A,
        Measure::Angle(i, j, k) => {
            let (a, b) = (traj.delta(t, j, i), traj.delta(t, j, k));
            let c = dot(a, b) / libm::sqrt(dot(a, a) * dot(b, b));
            libm::acos(c.clamp(-1.0, 1.0)) * Units::RAD_TO_DEG
        }
        Measure::Dihedral(i, j, k, l) => {
            let (b1, b2, b3) = (
                traj.delta(t, i, j),
                traj.delta(t, j, k),
                traj.delta(t, k, l),
            );
            let (n1, n2) = (cross(b1, b2), cross(b2, b3));
            let y = libm::sqrt(dot(b2, b2)) * dot(b1, n2);
            libm::atan2(y, dot(n1, n2)) * Units::RAD_TO_DEG
        }
    }
}

/// The values of `measures` in every frame (Angstrom and degrees), `[measure][frame]`.
pub fn geometry_series(traj: &MdTrajectory, measures: &[Measure]) -> MoptResult<Vec<Vec<f64>>> {
    let n = traj.n_atoms();
    for m in measures {
        let atoms = m.atoms();
        if let Some(bad) = atoms.iter().find(|i| **i >= n) {
            return Err(MoptError::config(format!(
                "atom {} of {} is not in the structure of {n} atoms",
                bad + 1,
                m.name()
            )));
        }
        let mut sorted = atoms.clone();
        sorted.sort_unstable();
        sorted.dedup();
        if sorted.len() != atoms.len() {
            return Err(MoptError::config(format!(
                "{} names an atom twice",
                m.name()
            )));
        }
    }
    Ok(measures
        .iter()
        .map(|m| (0..traj.len()).map(|t| value(traj, t, m)).collect())
        .collect())
}
