//! Cartesian coordinates (`q = x`, `B = I`).
//!
//! MultiOptPy: Cartesian steps of `optimizer.py:CalculateMoveVector`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::{Col, Mat};

use super::CoordinateSystem;
use crate::domain::MolecularSystem;
use crate::error::{MoptError, MoptResult};
use crate::math::projection::build_tr_rot_basis;

/// Cartesian coordinates in Bohr.
#[derive(Debug, Clone, Copy, Default)]
pub struct Cartesian;

impl CoordinateSystem for Cartesian {
    fn name(&self) -> &str {
        "cart"
    }

    fn initialize(&mut self, _sys: &MolecularSystem) -> MoptResult<()> {
        Ok(())
    }

    fn dim(&self, sys: &MolecularSystem) -> usize {
        sys.dof()
    }

    fn values(&self, sys: &MolecularSystem) -> MoptResult<Col<f64>> {
        Ok(sys.coords_flat_bohr())
    }

    fn wilson_b(&self, sys: &MolecularSystem) -> MoptResult<Mat<f64>> {
        Ok(Mat::identity(sys.dof(), sys.dof()))
    }

    fn displacement(&self, to: &MolecularSystem, from: &MolecularSystem) -> MoptResult<Col<f64>> {
        if to.n_atoms() != from.n_atoms() {
            return Err(MoptError::coord(
                "displacement between systems of different size",
            ));
        }
        // Optimization trajectories are kept unwrapped, so the plain difference is used.
        Ok(to.coords_flat_bohr() - from.coords_flat_bohr())
    }

    fn rigid_body_basis(&self, sys: &MolecularSystem) -> Option<Mat<f64>> {
        Some(build_tr_rot_basis(sys, false))
    }

    fn is_cartesian(&self) -> bool {
        true
    }
}
