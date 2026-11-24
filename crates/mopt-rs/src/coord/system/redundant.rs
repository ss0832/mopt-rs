//! Redundant internal coordinates (all generated primitives).
//!
//! MultiOptPy: `Coordinate/redundant_coordinate.py:RedundantInternalCoordinates` (`B_matrix`,
//!  `G_matrix`, `cartgrad2RICgrad`, `RICgrad2cartgrad`, `RIChess2carthess`).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::{Col, Mat};

use super::CoordinateSystem;
use crate::coord::internal::{PrimitiveOptions, PrimitiveSet};
use crate::domain::MolecularSystem;
use crate::error::{MoptError, MoptResult};

/// Redundant internal coordinates.
#[derive(Debug, Clone, Default)]
pub struct RedundantInternal {
    /// Options used to generate the primitives.
    pub options: PrimitiveOptions,
    primitives: PrimitiveSet,
    /// The set was given by the caller and is never generated anew.
    fixed: bool,
}

impl RedundantInternal {
    /// Use an explicit primitive set instead of automatic generation.
    pub fn with_primitives(primitives: PrimitiveSet) -> Self {
        Self {
            options: PrimitiveOptions::default(),
            primitives,
            fixed: true,
        }
    }

    /// The primitive set.
    pub fn primitives(&self) -> &PrimitiveSet {
        &self.primitives
    }

    fn ensure_ready(&self) -> MoptResult<()> {
        if self.primitives.is_empty() {
            Err(MoptError::coord(
                "redundant internal coordinates are not initialized",
            ))
        } else {
            Ok(())
        }
    }
}

impl CoordinateSystem for RedundantInternal {
    fn name(&self) -> &str {
        "ric"
    }

    fn labels(&self) -> Option<Vec<String>> {
        Some(
            self.primitives
                .primitives
                .iter()
                .map(|p| p.label())
                .collect(),
        )
    }

    fn initialize(&mut self, sys: &MolecularSystem) -> MoptResult<()> {
        if !self.fixed {
            self.primitives = PrimitiveSet::generate(sys, &self.options)?;
        }
        self.ensure_ready()
    }

    fn dim(&self, _sys: &MolecularSystem) -> usize {
        self.primitives.len()
    }

    fn values(&self, sys: &MolecularSystem) -> MoptResult<Col<f64>> {
        self.ensure_ready()?;
        Ok(self.primitives.values(sys))
    }

    fn wilson_b(&self, sys: &MolecularSystem) -> MoptResult<Mat<f64>> {
        self.ensure_ready()?;
        Ok(self.primitives.wilson_b(sys))
    }

    fn displacement(&self, to: &MolecularSystem, from: &MolecularSystem) -> MoptResult<Col<f64>> {
        self.ensure_ready()?;
        Ok(self.primitives.difference(to, from))
    }

    /// A generated set is generated anew where it no longer suits the structure
    /// ([`PrimitiveSet::needs_regeneration`]); a set given by the caller is kept.
    fn needs_reinitialization(&self, sys: &MolecularSystem) -> bool {
        !self.fixed && self.primitives.needs_regeneration(sys)
    }
}
