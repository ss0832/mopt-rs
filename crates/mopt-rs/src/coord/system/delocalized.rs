//! Delocalized internal coordinates (Baker, Kessi, Delley, J. Chem. Phys. 1996, 105, 192).
//!
//! The active space `U` (eigenvectors of `G = B_prim B_prim^T` with non-zero eigenvalues) is
//! computed at the starting geometry and kept fixed, so `q = U^T q_prim` and
//! `B = U^T B_prim`. Keeping `U` fixed makes the coordinate definition, and therefore the
//! quasi-Newton Hessian, consistent across iterations.
//!
//! This replaces MultiOptPy's unregistered `DIC_RSIRFO` prototype (stretches only, no bends or
//! torsions); see `docs/source/migration_optimizers.md`. When the primitives no longer suit a structure
//! (an angle has become linear) or the active space has lost rank there, the optimizer sets
//! the coordinates up anew ([`CoordinateSystem::needs_reinitialization`]).
//!
//! MultiOptPy: `Optimizer/dic_rsirfo.py` (replaced, not ported).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::{Col, Mat};

use super::CoordinateSystem;
use crate::coord::internal::{PrimitiveOptions, PrimitiveSet};
use crate::domain::MolecularSystem;
use crate::error::{MoptError, MoptResult};
use crate::math::linalg::symmetric_eigen;

/// Relative threshold on eigenvalues of `G` defining the active space.
const ACTIVE_REL_TOL: f64 = 1e-8;

/// Active space of a primitive set with Wilson matrix `b` (`m x 3N`): the eigenvectors of
/// `G = B B^T` with non-zero eigenvalues, as columns (`m x k`).
pub(crate) fn active_space(b: &Mat<f64>) -> MoptResult<Mat<f64>> {
    let g = b * b.transpose();
    let (w, v) = symmetric_eigen(&g)?;
    let wmax = w.iter().fold(0.0_f64, |m, x| m.max(*x));
    let active: Vec<usize> = (0..w.nrows())
        .filter(|&k| w[k] > ACTIVE_REL_TOL * wmax)
        .collect();
    if active.is_empty() {
        return Err(MoptError::coord("no active delocalized coordinates"));
    }
    Ok(Mat::from_fn(v.nrows(), active.len(), |r, c| {
        v[(r, active[c])]
    }))
}

/// Whether the fixed active space `u` no longer spans independent coordinates at a structure
/// with the primitive Wilson matrix `b`: an eigenvalue of `(U^T B)(U^T B)^T` has dropped to
/// zero (relative to the largest), so that some motion of the atoms is no longer described.
pub(crate) fn lost_rank(u: &Mat<f64>, b: &Mat<f64>) -> bool {
    let bd = u.transpose() * b;
    let g = &bd * bd.transpose();
    match symmetric_eigen(&g) {
        Ok((w, _)) => {
            let wmax = w.iter().fold(0.0_f64, |m, x| m.max(*x));
            w.iter().any(|x| *x <= ACTIVE_REL_TOL * wmax)
        }
        Err(_) => true,
    }
}

/// Delocalized internal coordinates.
#[derive(Debug, Clone, Default)]
pub struct DelocalizedInternal {
    /// Options used to generate the primitives.
    pub options: PrimitiveOptions,
    primitives: PrimitiveSet,
    /// Active-space basis `U` (`m x k`).
    u: Option<Mat<f64>>,
}

impl DelocalizedInternal {
    /// Delocalized internal coordinates with `options` for the primitives.
    pub fn with_options(options: PrimitiveOptions) -> Self {
        Self {
            options,
            ..Self::default()
        }
    }

    /// The primitive set.
    pub fn primitives(&self) -> &PrimitiveSet {
        &self.primitives
    }

    fn basis(&self) -> MoptResult<&Mat<f64>> {
        self.u
            .as_ref()
            .ok_or_else(|| MoptError::coord("delocalized internal coordinates are not initialized"))
    }
}

impl CoordinateSystem for DelocalizedInternal {
    fn name(&self) -> &str {
        "dic"
    }

    fn initialize(&mut self, sys: &MolecularSystem) -> MoptResult<()> {
        self.primitives = PrimitiveSet::generate(sys, &self.options)?;
        self.u = Some(active_space(&self.primitives.wilson_b(sys))?);
        Ok(())
    }

    fn dim(&self, _sys: &MolecularSystem) -> usize {
        self.u.as_ref().map_or(0, |u| u.ncols())
    }

    fn values(&self, sys: &MolecularSystem) -> MoptResult<Col<f64>> {
        Ok(self.basis()?.transpose() * self.primitives.values(sys))
    }

    fn wilson_b(&self, sys: &MolecularSystem) -> MoptResult<Mat<f64>> {
        Ok(self.basis()?.transpose() * self.primitives.wilson_b(sys))
    }

    fn displacement(&self, to: &MolecularSystem, from: &MolecularSystem) -> MoptResult<Col<f64>> {
        Ok(self.basis()?.transpose() * self.primitives.difference(to, from))
    }

    /// New primitives and a new active space where the primitives no longer suit the structure
    /// ([`PrimitiveSet::needs_regeneration`]) or the active space has lost rank.
    fn needs_reinitialization(&self, sys: &MolecularSystem) -> bool {
        self.primitives.needs_regeneration(sys)
            || self
                .u
                .as_ref()
                .is_some_and(|u| lost_rank(u, &self.primitives.wilson_b(sys)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coord::system::{back_transform, transform::BackTransformOptions, CoordinateFrame};
    use crate::domain::{positions, Angstrom};
    use crate::domain::{Element, SpinCharge};
    use crate::math::linalg::col_norm;

    fn h2o2() -> MolecularSystem {
        MolecularSystem::new(
            vec![Element::H, Element::O, Element::O, Element::H],
            &positions::<Angstrom>(&[
                [0.9, 0.3, 0.6],
                [0.0, 0.7, 0.0],
                [0.0, -0.7, 0.0],
                [-0.8, -0.9, 0.5],
            ]),
            SpinCharge::default(),
            None,
        )
        .unwrap()
    }

    #[test]
    fn dic_dimension_is_3n_minus_6() {
        let sys = h2o2();
        let mut dic = DelocalizedInternal::default();
        dic.initialize(&sys).unwrap();
        assert_eq!(dic.dim(&sys), 6);
    }

    #[test]
    fn back_transformation_reaches_target_displacement() {
        let sys = h2o2();
        let mut dic = DelocalizedInternal::default();
        dic.initialize(&sys).unwrap();
        let dq = Col::from_fn(dic.dim(&sys), |k| 0.05 * ((k as f64) * 0.7).cos());
        let dx = back_transform(&dic, &sys, &dq, &BackTransformOptions::default()).unwrap();
        let moved = sys
            .with_flat_coords_bohr(&(sys.coords_flat_bohr() + &dx))
            .unwrap();
        let achieved = dic.displacement(&moved, &sys).unwrap();
        assert!(col_norm(&(&achieved - &dq)) < 1e-6);
    }

    #[test]
    fn gradient_transformation_is_consistent_with_energy_change() {
        // For E(x) = c . x, the first-order energy change along a small internal step equals
        // g_q . dq.
        let sys = h2o2();
        let mut dic = DelocalizedInternal::default();
        dic.initialize(&sys).unwrap();
        let c = Col::from_fn(sys.dof(), |k| ((k as f64) * 1.3).sin());
        let frame = CoordinateFrame::new(&dic, &sys).unwrap();
        let gq = frame.gradient_to_q(&c);
        let dq = Col::from_fn(dic.dim(&sys), |k| 1e-5 * (1.0 + k as f64));
        let dx = frame.step_to_cartesian_linear(&dq);
        let de_x: f64 = c.iter().zip(dx.iter()).map(|(a, b)| a * b).sum();
        let de_q: f64 = gq.iter().zip(dq.iter()).map(|(a, b)| a * b).sum();
        // c has components outside the internal space (rigid-body); only the internal part
        // is captured, which is what dx spans.
        assert!((de_x - de_q).abs() < 1e-12);
    }
}
