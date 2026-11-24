//! The metric and the motions left out of a saddle search: masses, rigid-body motions, and
//! fixed variables.
//!
//! A search works in the coordinates `q = M^(1/2) x` (mass-weighted Cartesians for
//! structures with `mass_weighted`; `q = x` otherwise). Gradients become `M^(-1/2) g`,
//! Hessians `M^(-1/2) H M^(-1/2)`, and a direction `v_x` of the points the direction
//! `M^(1/2) v_x`. The number of negative curvatures of a stationary point is the same in
//! both (the two Hessians are congruent), the paths of the dynamics are not. Every
//! direction, gradient and product of the Hessian is projected with `Q = I - D D^T`, `D` the
//! orthonormal rigid-body motions at the point (in the mass-weighted metric when the search
//! is), and the fixed variables are set to zero.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::{Col, Mat};

use crate::domain::MolecularSystem;
use crate::error::{MoptError, MoptResult};
use crate::math::linalg::{col_norm, symmetric_eigen, symmetrize};
use crate::math::projection::{build_tr_rot_basis, project_out_basis_vec};

/// Principal moments (of the rotation vectors in the metric of the search) below this
/// fraction of the largest belong to a nearly linear structure: that rotation is not left
/// out ([`SearchFrame::removed`]).
pub const LINEAR_MOMENT: f64 = 1e-3;

/// Masses, rigid-body motions and fixed variables of a search.
#[derive(Debug, Clone, Default)]
pub struct SearchFrame {
    /// `sqrt(m)` per coordinate (mass-weighted searches).
    weights: Option<Col<f64>>,
    /// The structure whose coordinates the points are.
    template: Option<MolecularSystem>,
    /// Cell coordinates whose rigid motions are projected at each new point.
    cell: Option<crate::opt::cell::VariableCell>,
    /// Whether the rigid-body motions are left out.
    rigid: bool,
    /// Variables that keep their values.
    frozen: Vec<usize>,
}

impl SearchFrame {
    /// Plain variables: no masses, nothing left out.
    pub fn plain() -> Self {
        Self::default()
    }

    /// The coordinates of `template` (Bohr): the rigid-body motions are left out with
    /// `rigid` (an isolated system on a surface without a preferred orientation), and the
    /// metric is the mass-weighted one with `mass_weighted`.
    pub fn structure(template: MolecularSystem, rigid: bool, mass_weighted: bool) -> Self {
        let weights = mass_weighted.then(|| {
            Col::from_fn(template.dof(), |k| {
                libm::sqrt(template.elements[k / 3].mass_amu().max(1e-6))
            })
        });
        Self {
            weights,
            template: Some(template),
            cell: None,
            rigid,
            frozen: Vec::new(),
        }
    }

    pub(crate) fn variable_cell(map: crate::opt::cell::VariableCell) -> MoptResult<Self> {
        let masses = map.masses()?;
        Ok(Self {
            weights: Some(Col::from_fn(masses.len(), |i| masses[i].sqrt())),
            cell: Some(map),
            ..Self::default()
        })
    }

    pub(crate) fn removed_at(&self, x: &Col<f64>) -> MoptResult<Option<Mat<f64>>> {
        if let Some(map) = &self.cell {
            let motions: Vec<_> = map
                .rigid_modes(x)?
                .iter()
                .map(|d| self.direction_to_q(d))
                .collect();
            return Ok(Some(crate::math::linalg::orthonormalize_columns(
                x.nrows(),
                &motions,
                1e-10,
            )));
        }
        Ok(self.removed(x))
    }

    /// Keep these variables where they are.
    pub fn with_frozen(mut self, frozen: Vec<usize>) -> Self {
        self.frozen = frozen;
        self
    }

    /// The fixed variables (the frame is used up).
    pub fn frozen_variables(self) -> Vec<usize> {
        self.frozen
    }

    /// Number of fixed variables.
    pub fn frozen_count(&self) -> usize {
        self.frozen.len()
    }

    /// Check that points of `n` variables fit.
    pub fn check(&self, n: usize) -> MoptResult<()> {
        if let Some(t) = &self.template {
            if t.dof() != n {
                return Err(MoptError::config(format!(
                    "a structure of {} coordinates for points of {n}",
                    t.dof()
                )));
            }
        }
        if self.cell.as_ref().is_some_and(|c| c.dimension() != n) {
            return Err(MoptError::config("HiSD cell and point dimensions differ"));
        }
        if self.frozen.iter().any(|&k| k >= n) {
            return Err(MoptError::config("a fixed variable is out of range"));
        }
        Ok(())
    }

    /// Whether the points are coordinates of atoms.
    pub fn is_structure(&self) -> bool {
        self.template.is_some()
    }

    /// Whether the metric is mass-weighted.
    pub fn is_mass_weighted(&self) -> bool {
        self.weights.is_some()
    }

    /// The structure at the point `x`, when the points are coordinates of atoms.
    pub fn structure_at(&self, x: &Col<f64>) -> Option<MolecularSystem> {
        self.template
            .as_ref()
            .and_then(|t| t.with_flat_coords_bohr(x).ok())
    }

    /// The orthonormal rigid-body motions at `x` in the metric of the search (`None`: none
    /// are left out).
    ///
    /// Rotations about principal axes whose moment is below [`LINEAR_MOMENT`] times the
    /// largest are not left out: a nearly linear structure is treated as a linear one.
    /// Otherwise the rotation about the axis of a slightly bent chain, normalized, is a
    /// bending motion of the chain, and leaving it out removes a direction of the search
    /// (the linear chain of seven Lennard-Jones atoms, of index 10, has ten bending modes).
    /// In a periodic cell the translations along the periodic directions are left out
    /// ([`build_tr_rot_basis`]).
    pub fn removed(&self, x: &Col<f64>) -> Option<Mat<f64>> {
        if !self.rigid {
            return None;
        }
        let sys = self.structure_at(x)?;
        let mass_weighted = self.weights.is_some();
        if sys.active_cell().is_some() {
            let d = build_tr_rot_basis(&sys, mass_weighted);
            return (d.ncols() > 0).then_some(d);
        }
        let n = sys.n_atoms();
        let dof = 3 * n;
        let w: Vec<f64> = (0..n)
            .map(|i| match &self.weights {
                Some(w) => w[3 * i],
                None => 1.0,
            })
            .collect();
        let com = sys.center_of_mass().value();
        let mut columns: Vec<Col<f64>> = (0..3)
            .map(|c| Col::from_fn(dof, |k| if k % 3 == c { w[k / 3] } else { 0.0 }))
            .collect();
        let rotations: Vec<Col<f64>> = (0..3)
            .map(|axis| {
                Col::from_fn(dof, |k| {
                    let (i, c) = (k / 3, k % 3);
                    let r = [
                        x[3 * i] - com[0],
                        x[3 * i + 1] - com[1],
                        x[3 * i + 2] - com[2],
                    ];
                    // (e_axis x r)_c
                    let cross = match axis {
                        0 => [0.0, -r[2], r[1]],
                        1 => [r[2], 0.0, -r[0]],
                        _ => [-r[1], r[0], 0.0],
                    };
                    w[i] * cross[c]
                })
            })
            .collect();
        let gram = Mat::<f64>::from_fn(3, 3, |a, b| {
            rotations[a]
                .iter()
                .zip(rotations[b].iter())
                .map(|(p, q)| p * q)
                .sum()
        });
        if let Ok((moments, axes)) = symmetric_eigen(&gram) {
            let largest = moments[2].max(0.0);
            for k in 0..3 {
                if moments[k] > LINEAR_MOMENT * largest && moments[k] > 0.0 {
                    let mut r = Col::<f64>::zeros(dof);
                    for a in 0..3 {
                        r += &rotations[a] * axes[(a, k)];
                    }
                    columns.push(r);
                }
            }
        }
        let d = crate::math::linalg::orthonormalize_columns(dof, &columns, 1e-10);
        (d.ncols() > 0).then_some(d)
    }

    /// `v` (in the metric of the search) without rigid-body motions and fixed variables.
    pub fn mask(&self, removed: Option<&Mat<f64>>, v: &Col<f64>) -> Col<f64> {
        let mut out = match removed {
            Some(d) => project_out_basis_vec(d, v),
            None => v.clone(),
        };
        for &k in &self.frozen {
            out[k] = 0.0;
        }
        out
    }

    /// A gradient of the points in the metric of the search, `M^(-1/2) g`.
    pub fn gradient_to_q(&self, g: &Col<f64>) -> Col<f64> {
        match &self.weights {
            Some(w) => Col::from_fn(g.nrows(), |k| g[k] / w[k]),
            None => g.clone(),
        }
    }

    /// A gradient in the metric of the search back to the points, `M^(1/2) g_q`.
    pub fn gradient_to_x(&self, g: &Col<f64>) -> Col<f64> {
        match &self.weights {
            Some(w) => Col::from_fn(g.nrows(), |k| g[k] * w[k]),
            None => g.clone(),
        }
    }

    /// A direction (or displacement) of the points in the metric of the search,
    /// `M^(1/2) v`.
    pub fn direction_to_q(&self, v: &Col<f64>) -> Col<f64> {
        self.gradient_to_x(v)
    }

    /// A direction (or displacement) in the metric of the search back to the points,
    /// `M^(-1/2) v_q`.
    pub fn direction_to_x(&self, v: &Col<f64>) -> Col<f64> {
        self.gradient_to_q(v)
    }

    /// The largest displacement of an atom in `dx` (structures), or its length (plain
    /// variables).
    pub fn largest_displacement(&self, dx: &Col<f64>) -> f64 {
        if self.template.is_none() {
            return col_norm(dx);
        }
        (0..dx.nrows() / 3)
            .map(|a| libm::sqrt(dx[3 * a].powi(2) + dx[3 * a + 1].powi(2) + dx[3 * a + 2].powi(2)))
            .fold(0.0, f64::max)
    }

    /// The Hessian of the points in the metric of the search, projected with `Q` and the
    /// fixed variables taken out: `Q M^(-1/2) H M^(-1/2) Q` (rows and columns of fixed
    /// variables zero).
    pub fn hessian_to_q(&self, removed: Option<&Mat<f64>>, h: &Mat<f64>) -> Mat<f64> {
        let n = h.nrows();
        let mut out = match &self.weights {
            Some(w) => Mat::from_fn(n, n, |i, j| h[(i, j)] / (w[i] * w[j])),
            None => h.clone(),
        };
        if let Some(d) = removed {
            let m = &out - d * (d.transpose() * &out);
            out = &m - (&m * d) * d.transpose();
        }
        for &k in &self.frozen {
            for j in 0..n {
                out[(k, j)] = 0.0;
                out[(j, k)] = 0.0;
            }
        }
        symmetrize(&mut out);
        out
    }

    /// The eigenvalues and eigenvectors of a Hessian from [`Self::hessian_to_q`] without
    /// the motions left out (rigid-body motions, fixed variables), ascending.
    pub fn eigen(
        &self,
        removed: Option<&Mat<f64>>,
        h_q: &Mat<f64>,
    ) -> MoptResult<(Vec<f64>, Vec<Col<f64>>)> {
        let n = h_q.nrows();
        // The motions left out are moved above every curvature, then dropped.
        let scale = (0..n).map(|i| h_q[(i, i)].abs()).fold(1.0, f64::max);
        let shift = 10.0 * scale * n as f64 + 1.0;
        let mut shifted = h_q.clone();
        if let Some(d) = removed {
            shifted += (d * d.transpose()) * shift;
        }
        for &k in &self.frozen {
            shifted[(k, k)] += shift;
        }
        let (values, vectors) = symmetric_eigen(&shifted)?;
        let dropped = removed.map_or(0, Mat::ncols) + self.frozen.len();
        let kept = n.saturating_sub(dropped);
        Ok((
            (0..kept).map(|i| values[i]).collect(),
            (0..kept).map(|i| vectors.col(i).to_owned()).collect(),
        ))
    }

    /// The curvatures (eigenvalues in the metric of the search, ascending) of the Hessian
    /// `h` of the points at `x`, without the motions left out: the number of negative ones
    /// is the index of a stationary point.
    pub fn curvatures(&self, x: &Col<f64>, h: &Mat<f64>) -> MoptResult<Vec<f64>> {
        let removed = self.removed(x);
        let h_q = self.hessian_to_q(removed.as_ref(), h);
        Ok(self.eigen(removed.as_ref(), &h_q)?.0)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::domain::{positions, Bohr, Element, SpinCharge};

    #[test]
    fn rigid_motions_and_masses_are_left_out_of_the_curvatures() {
        let sys = MolecularSystem::new(
            vec![Element::parse_token("Ar").unwrap(); 2],
            &positions::<Bohr>(&[[0.0, 0.0, 0.0], [7.0, 0.0, 0.0]]),
            SpinCharge::default(),
            None,
        )
        .unwrap();
        // A stretch of curvature 2 between the atoms (along x), nothing else.
        let mut h = Mat::<f64>::zeros(6, 6);
        h[(0, 0)] = 1.0;
        h[(3, 3)] = 1.0;
        h[(0, 3)] = -1.0;
        h[(3, 0)] = -1.0;
        let x = sys.coords_flat_bohr();
        let frame = SearchFrame::structure(sys.clone(), true, false);
        let c = frame.curvatures(&x, &h).unwrap();
        // 6 - 5 rigid motions of a linear molecule = 1 vibration.
        assert_eq!(c.len(), 1);
        assert!((c[0] - 2.0).abs() < 1e-10, "{c:?}");
        let weighted = SearchFrame::structure(sys, true, true);
        let c = weighted.curvatures(&x, &h).unwrap();
        let m = Element::parse_token("Ar").unwrap().mass_amu();
        assert!((c[0] - 2.0 / m).abs() < 1e-10, "{c:?}");
        let v = Col::from_fn(6, |k| [1.0, 2.0, 0.0, 1.0, 2.0, 0.0][k]);
        // A translation is removed entirely.
        let r = frame.removed(&x);
        assert!(col_norm(&frame.mask(r.as_ref(), &v)) < 1e-12);
        assert!((frame.largest_displacement(&v) - 5.0_f64.sqrt()).abs() < 1e-12);
    }
}
