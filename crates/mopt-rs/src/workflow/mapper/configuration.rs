//! What the nodes of a network are points of: structures of atoms (with their cell, charge, and
//! spin), or points of an objective of plain variables (`-os mb2d`, `-os rosenbrock`, ...; any
//! number of variables, nothing treated as an atom).
//!
//! The parts of the mapper that need atoms (AFIR pushes between pairs of atoms, bond graphs,
//! dissociation channels, the nanoreactor) ask for the structure and refuse plain variables
//! with the reason; everything else (optimizations, IRC, the two-point searches, the identity
//! of nodes, the files, the kinetics) works on both.
//!
//! A point of plain variables is written as a frame of the same shape as an xyz frame: the
//! number of variables, a comment line, and one `name value` line per variable (the lines of a
//! variables file), so that a trajectory of them is a file of such frames (`.txt`).
//!
//! MultiOptPy: no counterpart (new in mopt-rs; MultiOptPy's mapper explores molecules only).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::sync::Arc;

use faer::Col;

use crate::domain::{MolecularSystem, Quantity};
use crate::error::{MoptError, MoptResult};
use crate::io::xyz::format_xyz_system;
use crate::workflow::explore::identity::StructureMatcher;

/// The values of the variables of an objective, with their names.
#[derive(Debug, Clone)]
pub struct VariablePoint {
    /// The values.
    pub values: Col<f64>,
    /// The names (shared by every point of a network).
    pub names: Arc<Vec<String>>,
}

/// A point of the surface of a network.
#[derive(Debug, Clone)]
pub enum Configuration {
    /// A structure (Bohr).
    Structure(MolecularSystem),
    /// The values of the variables of an objective.
    Variables(VariablePoint),
}

impl From<MolecularSystem> for Configuration {
    fn from(system: MolecularSystem) -> Self {
        Self::Structure(system)
    }
}

/// The extension of the frame files of a network (`true`: of plain variables).
pub fn frame_extension(variables: bool) -> &'static str {
    if variables {
        "txt"
    } else {
        "xyz"
    }
}

impl Configuration {
    /// A point of the variables `names` at `values`.
    pub fn variables(values: Col<f64>, names: Arc<Vec<String>>) -> MoptResult<Self> {
        if values.nrows() != names.len() {
            return Err(MoptError::config(format!(
                "{} values for {} variables",
                values.nrows(),
                names.len()
            )));
        }
        Ok(Self::Variables(VariablePoint { values, names }))
    }

    /// The structure, when it is one.
    pub fn structure(&self) -> Option<&MolecularSystem> {
        match self {
            Self::Structure(s) => Some(s),
            Self::Variables(_) => None,
        }
    }

    /// The structure, or the error that `what` needs atoms.
    pub fn require_structure(&self, what: &str) -> MoptResult<&MolecularSystem> {
        self.structure().ok_or_else(|| {
            MoptError::unsupported(format!(
                "{what} needs atoms; the points of this network are plain variables"
            ))
        })
    }

    /// The variables, when it is a point of them.
    pub fn point(&self) -> Option<&VariablePoint> {
        match self {
            Self::Structure(_) => None,
            Self::Variables(p) => Some(p),
        }
    }

    /// Whether it is a point of plain variables.
    pub fn is_variables(&self) -> bool {
        matches!(self, Self::Variables(_))
    }

    /// Its coordinates as one vector (Bohr for a structure).
    pub fn flat(&self) -> Col<f64> {
        match self {
            Self::Structure(s) => s.coords_flat_bohr(),
            Self::Variables(p) => p.values.clone(),
        }
    }

    /// Displace a structure or a variable point along a flat coordinate direction.
    pub(crate) fn displaced_along(&self, direction: &Col<f64>, distance: f64) -> MoptResult<Self> {
        let x = self.flat();
        if x.nrows() != direction.nrows() {
            return Err(MoptError::config(
                "mode dimension does not match the optimization coordinates",
            ));
        }
        self.with_flat(&(x + direction * distance))
    }

    /// The same kind of point at the coordinates `x`.
    pub fn with_flat(&self, x: &Col<f64>) -> MoptResult<Self> {
        match self {
            Self::Structure(s) => Ok(Self::Structure(s.with_flat_coords_bohr(x)?)),
            Self::Variables(p) => Self::variables(x.clone(), Arc::clone(&p.names)),
        }
    }

    /// The number of coordinates (`3N` for a structure).
    pub fn dimension(&self) -> usize {
        match self {
            Self::Structure(s) => s.dof(),
            Self::Variables(p) => p.values.nrows(),
        }
    }

    /// Whether this point (energy `energy`) and `other` (energy `other_energy`) are the same
    /// by `matcher`: structures after the best matching and superposition of their atoms,
    /// points of plain variables when their energies are within its energy tolerance and their
    /// Euclidean distance within its `rmsd_tolerance`, read in the units of the variables
    /// (the mapper sets it from `variable_tolerance` for such a network).
    pub fn same_as(
        &self,
        energy: f64,
        other: &Self,
        other_energy: f64,
        matcher: &StructureMatcher,
    ) -> MoptResult<bool> {
        match (self, other) {
            (Self::Structure(a), Self::Structure(b)) => {
                matcher.same(a, Quantity::new(energy), b, Quantity::new(other_energy))
            }
            _ => Ok(
                (energy - other_energy).abs() <= matcher.energy_tolerance.value()
                    && self
                        .variable_distance(other)
                        .is_some_and(|d| d <= matcher.rmsd_tolerance.value()),
            ),
        }
    }

    /// The Euclidean distance between two points of plain variables (`None` for structures,
    /// which are compared after the best matching of their atoms).
    pub fn variable_distance(&self, other: &Self) -> Option<f64> {
        match (self, other) {
            (Self::Variables(a), Self::Variables(b)) if a.values.nrows() == b.values.nrows() => {
                Some(
                    a.values
                        .iter()
                        .zip(b.values.iter())
                        .map(|(p, q)| (p - q) * (p - q))
                        .sum::<f64>()
                        .sqrt(),
                )
            }
            _ => None,
        }
    }

    /// The point as one frame with `comment`: an xyz frame (Angstrom) for a structure; the
    /// number of variables, the comment, and `name value` lines for plain variables.
    pub fn frame(&self, comment: &str) -> String {
        match self {
            Self::Structure(s) => {
                let mut sys = s.clone();
                sys.comment = comment.to_string();
                format_xyz_system(&sys, None)
            }
            Self::Variables(p) => {
                let mut out = format!("{}\n{}\n", p.values.nrows(), comment.replace('\n', " "));
                for (name, v) in p.names.iter().zip(p.values.iter()) {
                    out.push_str(&format!("{name} {v:.12}\n"));
                }
                out
            }
        }
    }
}

/// The points of the frames of `text` (as [`Configuration::frame`] writes those of plain
/// variables) with their comments.
pub fn variable_frames(text: &str) -> MoptResult<Vec<(String, VariablePoint)>> {
    let mut lines = text.lines();
    let mut out = Vec::new();
    let mut names: Option<Arc<Vec<String>>> = None;
    while let Some(first) = lines.next() {
        if first.trim().is_empty() {
            continue;
        }
        let n: usize = first
            .trim()
            .parse()
            .map_err(|_| MoptError::parse(format!("a frame of variables starts with '{first}'")))?;
        let comment = lines.next().unwrap_or_default().to_string();
        let mut these = Vec::with_capacity(n);
        let mut values = Vec::with_capacity(n);
        for _ in 0..n {
            let line = lines
                .next()
                .ok_or_else(|| MoptError::parse("a frame of variables ends early"))?;
            let mut parts = line.split_whitespace();
            let (Some(name), Some(v)) = (parts.next(), parts.next()) else {
                return Err(MoptError::parse(format!("'{line}' is not 'name value'")));
            };
            these.push(name.to_string());
            values.push(
                v.parse::<f64>()
                    .map_err(|_| MoptError::parse(format!("'{v}' is not a number")))?,
            );
        }
        let shared = match &names {
            Some(known) if **known == these => Arc::clone(known),
            _ => {
                let fresh = Arc::new(these);
                names = Some(Arc::clone(&fresh));
                fresh
            }
        };
        out.push((
            comment,
            VariablePoint {
                values: Col::from_fn(values.len(), |i| values[i]),
                names: shared,
            },
        ));
    }
    Ok(out)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn variables_keep_their_dimension_and_have_no_atoms() {
        let names = Arc::new(vec!["x".to_string(), "y".to_string()]);
        let p = Configuration::variables(Col::from_fn(2, |i| [0.5, -1.0][i]), Arc::clone(&names))
            .unwrap();
        assert!(p.is_variables() && p.structure().is_none());
        assert!(p.require_structure("an AFIR push").is_err());
        assert_eq!(p.dimension(), 2);
        let q = p.with_flat(&Col::from_fn(2, |i| [0.5, 2.0][i])).unwrap();
        assert!((p.variable_distance(&q).unwrap() - 3.0).abs() < 1e-15);
        assert!(p.with_flat(&Col::zeros(3)).is_err());
        assert!(Configuration::variables(Col::zeros(3), names).is_err());
    }

    #[test]
    fn frames_of_variables_read_back() {
        let names = Arc::new(vec!["x".to_string(), "y".to_string()]);
        let p = Configuration::variables(Col::from_fn(2, |i| [0.5, -1.25][i]), names).unwrap();
        let text = format!("{}{}", p.frame("TS energy=-0.1"), p.frame("EQ energy=-0.2"));
        let frames = variable_frames(&text).unwrap();
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[1].0, "EQ energy=-0.2");
        assert_eq!(frames[0].1.values[1], -1.25);
        assert!(Arc::ptr_eq(&frames[0].1.names, &frames[1].1.names));
        assert!(variable_frames("2\nc\nx 1\n").is_err());
    }
}
