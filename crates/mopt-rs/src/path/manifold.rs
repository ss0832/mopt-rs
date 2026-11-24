//! The hard constraints of the searches of points (`-pc`, `-gfix`, `-uc`, with `-fix`) as a
//! manifold of the points: every point a search evaluates or keeps lies on
//! `M = {x : c(x) = t}` (to the floor of [`ConstraintSet::restore`], at most 1e-10), the
//! directions it moves along lie in the tangent space `T_x M = ker J(x)` (`J` the constraint
//! gradients as rows), and the gradient it is told is already without its part along the rows
//! of `J` (the surfaces of `ieipmain` project it).
//!
//! How each family of methods stays on `M` (derived in their own modules):
//!
//! - string methods (the growing and the freezing string): tangents, steps, and new nodes
//!   are taken into `T_x M` ([`super::string_nodes::PointGeometry::project`]), and every node
//!   that moved is put back onto `M` ([`super::string_nodes::PointGeometry::settle`]);
//! - the local minimax method: its rays from the minimum `A` become the curves
//!   `t -> R(A + t v)` (`R` the restore, `v` in `T_A M`), whose slope `g . v` with the
//!   projected gradient is that along the curve to first order;
//! - the growing Newton trajectory: its corrector minimizes on `M` intersected with the plane
//!   perpendicular to the direction ([`PointManifold::vector_constraints`]);
//! - the scaled hyperspheres (2PSHS, ADDF): the normal coordinates are those of the projected
//!   Hessian of the Lagrangian at the constrained minimum, the normals of `M` left out, and a
//!   point of the sphere is the point of `M` with those coordinates
//!   ([`super::shs::ScaledCoordinates::on_manifold`]).
//!
//! MultiOptPy: no counterpart (new in mopt-rs; MultiOptPy's methods drop such constraints).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::sync::Arc;

use faer::{Col, Mat};

use crate::domain::MolecularSystem;
use crate::error::{MoptError, MoptResult};
use crate::math::linalg::{col_dot, col_norm};
use crate::opt::vector_constraints::{FixedVariable, VectorConstraint, VectorConstraintSet};
use crate::pes::constraint::ConstraintSet;

/// Rows of the Jacobian shorter than this share of the longest one, after the rows before
/// them are taken out, add no direction to the normals (dependent constraints).
const DEPENDENT_SHARE: f64 = 1e-10;

/// The constraints of a run on the points of its structures.
#[derive(Debug, Clone)]
pub struct PointManifold {
    set: ConstraintSet,
    template: MolecularSystem,
    /// Rows that hold one variable at a value (frozen atoms, Cartesian components): the
    /// variable, per row (`None` for the other rows).
    fixed: Vec<Option<usize>>,
}

impl PointManifold {
    /// The constraints `set` on points that place the atoms of `template` (which lies on
    /// them).
    pub fn new(set: ConstraintSet, template: MolecularSystem) -> MoptResult<Self> {
        let x = template.coords_flat_bohr();
        let j = set.frame(&template, &Col::zeros(x.nrows()))?.jacobian;
        // A row that is one unit component is a variable held at a value (the restore
        // assigns those, bit-identical to their targets).
        let fixed = (0..j.nrows())
            .map(|k| {
                let nonzero: Vec<usize> = (0..j.ncols()).filter(|&i| j[(k, i)] != 0.0).collect();
                match nonzero.as_slice() {
                    [i] if j[(k, *i)] == 1.0 => Some(*i),
                    _ => None,
                }
            })
            .collect();
        Ok(Self {
            set,
            template,
            fixed,
        })
    }

    /// Number of scalar constraints.
    pub fn len(&self) -> usize {
        self.set.len()
    }

    /// Whether there are none.
    pub fn is_empty(&self) -> bool {
        self.set.is_empty()
    }

    /// The variables held at a value by a row of their own.
    pub fn fixed_variables(&self) -> Vec<usize> {
        self.fixed.iter().flatten().copied().collect()
    }

    fn system(&self, x: &Col<f64>) -> MoptResult<MolecularSystem> {
        self.template.with_flat_coords_bohr(x)
    }

    /// `x` put back onto the constraints ([`ConstraintSet::restore`]).
    pub fn restore(&self, x: &Col<f64>) -> MoptResult<Col<f64>> {
        Ok(self.set.restore(&self.system(x)?)?.coords_flat_bohr())
    }

    /// Values minus targets at `x`.
    pub fn residuals(&self, x: &Col<f64>) -> MoptResult<Col<f64>> {
        self.set.residuals(&self.system(x)?)
    }

    /// The constraint gradients at `x` (rows).
    pub fn jacobian(&self, x: &Col<f64>) -> MoptResult<Mat<f64>> {
        Ok(self
            .set
            .frame(&self.system(x)?, &Col::zeros(x.nrows()))?
            .jacobian)
    }

    /// An orthonormal basis of the normals of the constraints at `x` (columns: the span of
    /// the rows of the Jacobian, by modified Gram-Schmidt; dependent rows add nothing).
    pub fn normals(&self, x: &Col<f64>) -> MoptResult<Mat<f64>> {
        let j = self.jacobian(x)?;
        let n = j.ncols();
        let longest = (0..j.nrows())
            .map(|k| libm::sqrt((0..n).map(|i| j[(k, i)] * j[(k, i)]).sum::<f64>()))
            .fold(0.0_f64, f64::max);
        let mut basis: Vec<Col<f64>> = Vec::new();
        for k in 0..j.nrows() {
            let mut v = Col::from_fn(n, |i| j[(k, i)]);
            for b in &basis {
                let d = col_dot(b, &v);
                v -= b * d;
            }
            let size = col_norm(&v);
            if size > DEPENDENT_SHARE * longest && size > 0.0 {
                basis.push(v * (1.0 / size));
            }
        }
        Ok(Mat::from_fn(n, basis.len(), |i, c| basis[c][i]))
    }

    /// `v` at `x` without its part along the normals of the constraints (in `T_x M`).
    pub fn project(&self, x: &Col<f64>, v: &Col<f64>) -> MoptResult<Col<f64>> {
        Ok(without(&self.normals(x)?, v))
    }

    /// The constraints as constraints of an optimizer of plain variables
    /// ([`crate::opt::vector::VectorOptimizer::set_constraints`]): the variables held by a
    /// row of their own as fixed variables, every other row as a function of the point.
    pub fn vector_constraints(self: &Arc<Self>) -> MoptResult<VectorConstraintSet> {
        let mut set = VectorConstraintSet::new();
        let x = self.template.coords_flat_bohr();
        for (row, fixed) in self.fixed.iter().enumerate() {
            match fixed {
                Some(index) => set.push(Arc::new(FixedVariable { index: *index }), x[*index])?,
                None => set.push(
                    Arc::new(ManifoldRow {
                        manifold: Arc::clone(self),
                        row,
                    }),
                    0.0,
                )?,
            }
        }
        Ok(set)
    }
}

/// `v` without its part along the orthonormal columns of `basis`.
pub fn without(basis: &Mat<f64>, v: &Col<f64>) -> Col<f64> {
    if basis.ncols() == 0 {
        return v.clone();
    }
    v - basis * (basis.transpose() * v)
}

/// One row of a [`PointManifold`] as a function of the point: its residual (value minus
/// target, wrapped into half a period) and its gradient.
struct ManifoldRow {
    manifold: Arc<PointManifold>,
    row: usize,
}

impl VectorConstraint for ManifoldRow {
    fn name(&self) -> String {
        format!("constraint {}", self.row + 1)
    }

    fn evaluate(&self, x: &Col<f64>) -> MoptResult<(f64, Col<f64>)> {
        let r = self.manifold.residuals(x)?;
        let j = self.manifold.jacobian(x)?;
        if self.row >= r.nrows() {
            return Err(MoptError::config(format!(
                "constraint {}: there are {} constraints",
                self.row + 1,
                r.nrows()
            )));
        }
        Ok((r[self.row], Col::from_fn(j.ncols(), |i| j[(self.row, i)])))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::domain::{positions, AtomId, Bohr, Element, SpinCharge};
    use crate::pes::constraint::{Constraint, ConstraintKind};

    fn argon(points: &[[f64; 3]]) -> MolecularSystem {
        MolecularSystem::new(
            vec![Element::parse_token("Ar").unwrap(); points.len()],
            &positions::<Bohr>(points),
            SpinCharge::default(),
            None,
        )
        .unwrap()
    }

    #[test]
    fn points_are_restored_and_directions_projected() {
        let s = argon(&[[0.0, 0.0, 0.0], [7.2, 0.0, 0.0], [3.6, 6.2, 0.0]]);
        let c = vec![
            Constraint::at_start(ConstraintKind::Distance(
                vec![AtomId::new(0)],
                vec![AtomId::new(1)],
            )),
            Constraint::at_start(ConstraintKind::Frozen(vec![AtomId::new(2)])),
        ];
        let set = ConstraintSet::new(&c, &s, None).unwrap();
        let m = Arc::new(PointManifold::new(set, s.clone()).unwrap());
        assert_eq!(m.fixed_variables(), vec![6, 7, 8]);
        let mut x = s.coords_flat_bohr();
        x[0] += 0.3;
        x[7] += 0.2;
        let on = m.restore(&x).unwrap();
        assert!(m.residuals(&on).unwrap().iter().all(|r| r.abs() < 1e-10));
        assert_eq!(on[7], s.coords_flat_bohr()[7]);
        let v = Col::from_fn(9, |i| (i as f64 + 1.0) * 0.1);
        let p = m.project(&on, &v).unwrap();
        let j = m.jacobian(&on).unwrap();
        for k in 0..j.nrows() {
            let d: f64 = (0..9).map(|i| j[(k, i)] * p[i]).sum();
            assert!(d.abs() < 1e-12);
        }
        // The same constraints for an optimizer of plain variables.
        let vc = m.vector_constraints().unwrap();
        assert_eq!(vc.len(), 4);
        let back = vc.restore(&x).unwrap();
        assert!(m.residuals(&back).unwrap().iter().all(|r| r.abs() < 1e-10));
        assert_eq!(back[7], s.coords_flat_bohr()[7]);
    }
}
