//! ReSPer coordinates: the principal coordinates of a set of reference structures (a reaction
//! path, an IRC, the chain of a band, an MD trajectory) by classical multidimensional scaling,
//! extended to any structure, with their gradients, so that they serve as collective
//! variables of a scan.
//!
//! The references are mapped as by `optmain -cmds` ([`super::path_map`]): the distance of two
//! structures is `sqrt(sum_a m_a |r_a - r'_a|^2)` after their mass-weighted superposition
//! (Angstrom amu^(1/2); unit masses without mass weighting), over all atoms or chosen ones;
//! for plain variables it is the Euclidean distance of the points. With
//! `B = -J D2 J / 2` diagonalized and the coordinates `X = V Lambda^(1/2)` of the
//! references, a new structure with the squared distances `d_i^2` to the references has the
//! coordinates (Gower's formula for adding a point)
//!
//! ```text
//! y_k = sum_i X_ik (B_ii - d_i^2) / (2 lambda_k),
//! ```
//!
//! which gives every reference its own coordinates back. The gradient follows from that of
//! the squared distances: at the optimal superposition (envelope theorem) the derivative of
//! `d_i^2` by the position `R_a` of atom `a` is `2 m_a (R_a - R'_a)`, with `R'` the
//! reference superposed on the structure, so
//! `dy_k/dR = -sum_i X_ik d(d_i^2)/dR / (2 lambda_k)`. Second derivatives are by central
//! differences of the gradient when asked for ([`ResperCoordinate::with_curvature`]).
//!
//! References: Y. Tsutsumi, Y. Ono, T. Taketsugu, Chem. Commun. 2021, 57, 11734 (the
//! reaction space projector, ReSPer); J. C. Gower, Biometrika 1968, 55, 582 (adding a point
//! to a vector diagram).
//!
//! MultiOptPy: no counterpart (new in mopt-rs; MultiOptPy's `PESAnalyzer/cmds_analysis.py`
//! maps the structures of a run and has no coordinates of new structures).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::sync::Arc;

use faer::{Col, Mat};
use rayon::prelude::*;

use super::path_map::classical_mds;
use crate::coord::alignment::kabsch_align_bohr;
use crate::domain::{AtomId, EvalOutput, MolecularSystem, Quantity, Units};
use crate::error::{MoptError, MoptResult};
use crate::opt::vector_constraints::VectorConstraint;
use crate::pes::{PesTerm, TermContext};

/// Coordinates of the map (the first two principal coordinates, or one where the references
/// span only one dimension).
pub const MAP_COMPONENTS: usize = 2;
/// Displacement (Bohr, or the unit of the variables) of the central differences of the
/// second derivatives.
const CURVATURE_STEP: f64 = 1e-4;

/// What the references are.
#[derive(Debug, Clone)]
enum References {
    /// Structures of the same atoms: the whole structures and the chosen atoms.
    Structures {
        frames: Vec<MolecularSystem>,
        atoms: Vec<AtomId>,
        mass_weighted: bool,
    },
    /// Points of plain variables.
    Points(Vec<Col<f64>>),
}

/// The map of a set of references and the coordinates of new structures on it.
#[derive(Debug, Clone)]
pub struct ResperMap {
    references: References,
    /// Coordinates of the references (row: reference, column: coordinate).
    coordinates: Mat<f64>,
    /// Eigenvalues of the coordinates kept.
    eigenvalues: Vec<f64>,
    /// The diagonal of `B` (squared distances of the references from the centre).
    diagonal: Vec<f64>,
    /// Share of each coordinate in the sum of the positive eigenvalues.
    contributions: Vec<f64>,
    /// MultiOptPy's "dimensional reproducibility" of the distances.
    reproducibility: f64,
}

/// Squared distances to the references and their gradients (one row per reference).
struct Distances {
    squared: Vec<f64>,
    gradients: Vec<Col<f64>>,
}

impl ResperMap {
    /// The map of the structures `frames` (at least three, all of the same atoms). `atoms`
    /// are the atoms the distances are taken over (all when empty); `mass_weighted` uses
    /// the masses in the superposition and the distance (unit masses otherwise).
    pub fn from_structures(
        frames: Vec<MolecularSystem>,
        atoms: &[AtomId],
        mass_weighted: bool,
    ) -> MoptResult<Self> {
        let first = frames
            .first()
            .ok_or_else(|| MoptError::config("ReSPer: no reference structures"))?;
        if frames.iter().any(|f| f.elements != first.elements) {
            return Err(MoptError::config(
                "ReSPer: the reference structures must have the same atoms in the same order",
            ));
        }
        let n = first.n_atoms();
        let mut atoms: Vec<AtomId> = if atoms.is_empty() {
            first.atoms().collect()
        } else {
            atoms.to_vec()
        };
        atoms.sort_unstable();
        atoms.dedup();
        for a in &atoms {
            a.check(n)?;
        }
        if atoms.len() < 2 {
            return Err(MoptError::config(
                "ReSPer: the distances need at least two atoms",
            ));
        }
        let references = References::Structures {
            frames,
            atoms,
            mass_weighted,
        };
        Self::build(references)
    }

    /// The atoms of the distances of a map of structures.
    pub fn atoms(&self) -> Option<&[AtomId]> {
        match &self.references {
            References::Structures { atoms, .. } => Some(atoms),
            References::Points(_) => None,
        }
    }

    /// The map of the points `points` of plain variables (at least three, all of one
    /// length).
    pub fn from_points(points: Vec<Col<f64>>) -> MoptResult<Self> {
        let n = points.first().map_or(0, Col::nrows);
        if n == 0 || points.iter().any(|p| p.nrows() != n) {
            return Err(MoptError::config(
                "ReSPer: the reference points must have the same number of variables",
            ));
        }
        if points.iter().any(|p| p.iter().any(|v| !v.is_finite())) {
            return Err(MoptError::config("ReSPer: a reference point is not finite"));
        }
        Self::build(References::Points(points))
    }

    fn build(references: References) -> MoptResult<Self> {
        let count = match &references {
            References::Structures { frames, .. } => frames.len(),
            References::Points(points) => points.len(),
        };
        if count < 3 {
            return Err(MoptError::config(format!(
                "ReSPer: a map needs at least three reference structures, got {count}"
            )));
        }
        let mut map = Self {
            references,
            coordinates: Mat::zeros(0, 0),
            eigenvalues: Vec::new(),
            diagonal: Vec::new(),
            contributions: Vec::new(),
            reproducibility: 0.0,
        };
        let pairs: Vec<(usize, usize)> = (0..count)
            .flat_map(|i| (i + 1..count).map(move |j| (i, j)))
            .collect();
        let values: Vec<f64> = pairs
            .par_iter()
            .map(|&(i, j)| map.reference_distance(i, j))
            .collect::<MoptResult<_>>()?;
        let mut d = Mat::<f64>::zeros(count, count);
        for (&(i, j), value) in pairs.iter().zip(values) {
            d[(i, j)] = value;
            d[(j, i)] = value;
        }
        let embedding = classical_mds(&d, MAP_COMPONENTS)?;
        // Coordinates of zero variance (references on a line) are left out.
        let largest = embedding.eigenvalues.first().copied().unwrap_or(0.0);
        let kept = embedding
            .eigenvalues
            .iter()
            .take(MAP_COMPONENTS)
            .take_while(|v| **v > 1e-10 * largest.max(f64::MIN_POSITIVE))
            .count();
        if kept == 0 {
            return Err(MoptError::config(
                "ReSPer: the reference structures all coincide",
            ));
        }
        let square = |i: usize, j: usize| d[(i, j)] * d[(i, j)];
        let row_mean: Vec<f64> = (0..count)
            .map(|i| (0..count).map(|j| square(i, j)).sum::<f64>() / count as f64)
            .collect();
        let mean = row_mean.iter().sum::<f64>() / count as f64;
        map.diagonal = row_mean.iter().map(|r| r - 0.5 * mean).collect();
        map.coordinates = Mat::from_fn(count, kept, |i, k| embedding.coordinates[(i, k)]);
        map.eigenvalues = embedding.eigenvalues[..kept].to_vec();
        map.contributions = embedding.contributions[..kept].to_vec();
        map.reproducibility = embedding.reproducibility;
        Ok(map)
    }

    /// The distance of references `i` and `j`.
    fn reference_distance(&self, i: usize, j: usize) -> MoptResult<f64> {
        match &self.references {
            References::Structures {
                frames,
                atoms,
                mass_weighted,
            } => {
                let a = frames[i].subsystem(atoms, frames[i].spin_charge)?;
                let b = frames[j].subsystem(atoms, frames[j].spin_charge)?;
                Ok(squared_distance(&a, &b, *mass_weighted)?.0.max(0.0).sqrt())
            }
            References::Points(points) => Ok((&points[i] - &points[j]).norm_l2()),
        }
    }

    /// Number of references.
    pub fn len(&self) -> usize {
        self.coordinates.nrows()
    }

    /// Whether there are no references (never for a map that was built).
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Number of coordinates of the map (two, or one for references on a line).
    pub fn components(&self) -> usize {
        self.coordinates.ncols()
    }

    /// Coordinate `k` of reference `i`.
    pub fn reference_coordinate(&self, i: usize, k: usize) -> f64 {
        self.coordinates[(i, k)]
    }

    /// Eigenvalues of the coordinates.
    pub fn eigenvalues(&self) -> &[f64] {
        &self.eigenvalues
    }

    /// Share of each coordinate in the sum of the positive eigenvalues.
    pub fn contributions(&self) -> &[f64] {
        &self.contributions
    }

    /// How well points in a Euclidean space reproduce the distances (1: exactly).
    pub fn reproducibility(&self) -> f64 {
        self.reproducibility
    }

    /// Whether the references are structures (otherwise points of plain variables).
    pub fn of_structures(&self) -> bool {
        matches!(self.references, References::Structures { .. })
    }

    /// The reference structures (empty for points).
    pub fn reference_structures(&self) -> &[MolecularSystem] {
        match &self.references {
            References::Structures { frames, .. } => frames,
            References::Points(_) => &[],
        }
    }

    /// The reference points (empty for structures).
    pub fn reference_points(&self) -> &[Col<f64>] {
        match &self.references {
            References::Points(points) => points,
            References::Structures { .. } => &[],
        }
    }

    /// The weights of the variables in the metric of the map: the mass of the atom of each
    /// Cartesian component (zero for atoms outside the distances), or ones.
    pub fn metric_weights(&self, dof: usize) -> Vec<f64> {
        match &self.references {
            References::Structures {
                frames,
                atoms,
                mass_weighted,
            } => {
                let mut w = vec![0.0; dof];
                for a in atoms {
                    let m = if *mass_weighted {
                        frames[0].element(*a).mass_amu()
                    } else {
                        1.0
                    };
                    for c in 0..3 {
                        if let Some(slot) = w.get_mut(a.dof(c)) {
                            *slot = m;
                        }
                    }
                }
                w
            }
            References::Points(_) => vec![1.0; dof],
        }
    }

    fn distances_to_structure(&self, sys: &MolecularSystem) -> MoptResult<Distances> {
        let References::Structures {
            frames,
            atoms,
            mass_weighted,
        } = &self.references
        else {
            return Err(MoptError::config(
                "ReSPer: a map of plain variables has no coordinates of structures",
            ));
        };
        if sys.elements != frames[0].elements {
            return Err(MoptError::config(format!(
                "ReSPer: a structure of {} atoms for references of {} (or of other elements)",
                sys.n_atoms(),
                frames[0].n_atoms()
            )));
        }
        let here = sys.subsystem(atoms, sys.spin_charge)?;
        let results: Vec<(f64, Col<f64>)> = frames
            .par_iter()
            .map(|frame| {
                let reference = frame.subsystem(atoms, frame.spin_charge)?;
                squared_distance(&here, &reference, *mass_weighted)
            })
            .collect::<MoptResult<_>>()?;
        let dof = sys.dof();
        let mut squared = Vec::with_capacity(results.len());
        let mut gradients = Vec::with_capacity(results.len());
        for (d2, g_sub) in results {
            let mut g = Col::zeros(dof);
            for (k, a) in atoms.iter().enumerate() {
                for c in 0..3 {
                    g[a.dof(c)] = g_sub[3 * k + c];
                }
            }
            squared.push(d2);
            gradients.push(g);
        }
        Ok(Distances { squared, gradients })
    }

    fn distances_to_point(&self, x: &Col<f64>) -> MoptResult<Distances> {
        let References::Points(points) = &self.references else {
            return Err(MoptError::config(
                "ReSPer: a map of structures has no coordinates of plain variables",
            ));
        };
        if x.nrows() != points[0].nrows() {
            return Err(MoptError::config(format!(
                "ReSPer: {} variables for references of {}",
                x.nrows(),
                points[0].nrows()
            )));
        }
        let mut squared = Vec::with_capacity(points.len());
        let mut gradients = Vec::with_capacity(points.len());
        for p in points {
            let d = x - p;
            squared.push(d.iter().map(|v| v * v).sum());
            gradients.push(d * faer::Scale(2.0));
        }
        Ok(Distances { squared, gradients })
    }

    /// Gower's formula for the coordinates and their gradients from the distances.
    fn project_distances(&self, d: &Distances, dof: usize) -> (Vec<f64>, Vec<Col<f64>>) {
        let n = self.len();
        let mut values = Vec::with_capacity(self.components());
        let mut gradients = Vec::with_capacity(self.components());
        for k in 0..self.components() {
            let scale = 1.0 / (2.0 * self.eigenvalues[k]);
            let mut y = 0.0;
            let mut g = Col::<f64>::zeros(dof);
            for i in 0..n {
                let x = self.coordinates[(i, k)];
                y += x * (self.diagonal[i] - d.squared[i]);
                g -= &d.gradients[i] * faer::Scale(x);
            }
            values.push(y * scale);
            gradients.push(g * faer::Scale(scale));
        }
        (values, gradients)
    }

    /// The coordinates of the structure `sys` (of the atoms of the references) and their
    /// gradients (Cartesian, per Bohr).
    pub fn project_structure(
        &self,
        sys: &MolecularSystem,
    ) -> MoptResult<(Vec<f64>, Vec<Col<f64>>)> {
        let d = self.distances_to_structure(sys)?;
        Ok(self.project_distances(&d, sys.dof()))
    }

    /// The coordinates of the point `x` of plain variables and their gradients.
    pub fn project_point(&self, x: &Col<f64>) -> MoptResult<(Vec<f64>, Vec<Col<f64>>)> {
        let d = self.distances_to_point(x)?;
        Ok(self.project_distances(&d, x.nrows()))
    }

    /// The coordinates and gradients of the flat vector `x`: the positions (Bohr) of a
    /// structure like the references, or a point of plain variables.
    pub fn project_flat(&self, x: &Col<f64>) -> MoptResult<(Vec<f64>, Vec<Col<f64>>)> {
        match &self.references {
            References::Structures { frames, .. } => {
                self.project_structure(&frames[0].with_flat_coords_bohr(x)?)
            }
            References::Points(_) => self.project_point(x),
        }
    }
}

/// `d^2 = sum_a m_a |r_a - r'_a|^2` (Angstrom^2 amu) of `here` and `reference` after the
/// superposition of the reference on `here`, and its gradient by the positions of `here`
/// (Bohr).
fn squared_distance(
    here: &MolecularSystem,
    reference: &MolecularSystem,
    mass_weighted: bool,
) -> MoptResult<(f64, Col<f64>)> {
    let (aligned, _) = kabsch_align_bohr(reference, here, mass_weighted)?;
    let a2 = Units::BOHR_TO_ANGSTROM * Units::BOHR_TO_ANGSTROM;
    let mut sum = 0.0;
    let mut g = Col::zeros(here.dof());
    for atom in here.atoms() {
        let m = if mass_weighted {
            here.element(atom).mass_amu()
        } else {
            1.0
        };
        let p = here.position(atom).value();
        let q = aligned.position(atom).value();
        for c in 0..3 {
            let r = p[c] - q[c];
            sum += m * r * r;
            g[atom.dof(c)] = 2.0 * a2 * m * r;
        }
    }
    Ok((a2 * sum, g))
}

/// One ReSPer coordinate of structures as a surface term: its value is the "energy" and its
/// gradient the gradient, so that a constraint can hold it ([`crate::pes::constraint`]).
pub struct ResperCoordinate {
    map: Arc<ResperMap>,
    component: usize,
    curvature: bool,
    name: String,
}

impl ResperCoordinate {
    /// Coordinate `component` (0-based) of `map`.
    pub fn new(map: Arc<ResperMap>, component: usize) -> MoptResult<Self> {
        if component >= map.components() {
            return Err(MoptError::config(format!(
                "ReSPer: the map has {} coordinate(s); there is no PCo{}",
                map.components(),
                component + 1
            )));
        }
        Ok(Self {
            name: format!("ReSPer PCo{}", component + 1),
            map,
            component,
            curvature: false,
        })
    }

    /// Give the second derivatives when they are asked for (central differences of the
    /// gradient: two projections per Cartesian component).
    #[must_use]
    pub fn with_curvature(mut self, curvature: bool) -> Self {
        self.curvature = curvature;
        self
    }
}

impl PesTerm for ResperCoordinate {
    fn name(&self) -> &str {
        &self.name
    }

    /// The atoms of the distances of the map (those of structures; a map of plain variables
    /// has none).
    fn atoms(&self) -> Option<Vec<AtomId>> {
        self.map.atoms().map(<[AtomId]>::to_vec)
    }

    fn evaluate(
        &self,
        sys: &MolecularSystem,
        _ctx: &TermContext<'_>,
        need_hessian: bool,
    ) -> MoptResult<EvalOutput> {
        let (values, gradients) = self.map.project_structure(sys)?;
        let mut out = EvalOutput::energy_gradient(
            Quantity::new(values[self.component]),
            gradients[self.component].clone(),
        );
        if need_hessian && self.curvature {
            let x = sys.coords_flat_bohr();
            let n = x.nrows();
            let mut h = Mat::<f64>::zeros(n, n);
            for j in 0..n {
                let mut columns = Vec::with_capacity(2);
                for sign in [1.0, -1.0] {
                    let mut moved = x.clone();
                    moved[j] += sign * CURVATURE_STEP;
                    let shifted = sys.with_flat_coords_bohr(&moved)?;
                    columns.push(self.map.project_structure(&shifted)?.1[self.component].clone());
                }
                for i in 0..n {
                    h[(i, j)] = (columns[0][i] - columns[1][i]) / (2.0 * CURVATURE_STEP);
                }
            }
            out.hessian_bohr = Some(Mat::from_fn(n, n, |i, j| 0.5 * (h[(i, j)] + h[(j, i)])));
        }
        Ok(out)
    }

    /// The coordinate is a function of the distances to the references after their
    /// superposition on the structure.
    fn frame(&self) -> crate::pes::term::Frame {
        crate::pes::term::Frame::Invariant
    }

    fn history(&self) -> crate::pes::term::History {
        crate::pes::term::History::Static
    }
}

/// One ReSPer coordinate of plain variables as a constraint of their optimization.
pub struct ResperVariable {
    map: Arc<ResperMap>,
    component: usize,
}

impl ResperVariable {
    /// Coordinate `component` (0-based) of the map of points `map`.
    pub fn new(map: Arc<ResperMap>, component: usize) -> MoptResult<Self> {
        if component >= map.components() || map.of_structures() {
            return Err(MoptError::config(format!(
                "ReSPer: no PCo{} of plain variables on this map",
                component + 1
            )));
        }
        Ok(Self { map, component })
    }
}

impl VectorConstraint for ResperVariable {
    fn name(&self) -> String {
        format!("ReSPer PCo{}", self.component + 1)
    }

    // Computed through a superposition with a noise of about 1e-9: a residual above the
    // guarantee is warned about, not an error.
    fn warns_only(&self) -> bool {
        true
    }

    fn evaluate(&self, x: &Col<f64>) -> MoptResult<(f64, Col<f64>)> {
        let (values, mut gradients) = self.map.project_point(x)?;
        Ok((
            values[self.component],
            gradients.swap_remove(self.component),
        ))
    }
}

pub mod preimage;

#[cfg(test)]
mod tests;
