//! Ordered structure sequences (optimization history, reaction paths, MD trajectories).
//!
//! MultiOptPy: `fileio.py:traj2list` and the trajectory files of `optimization.py:Optimize`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use crate::domain::quantity::{Hartree, Quantity};
use faer::Col;

use crate::domain::molecule::MolecularSystem;

/// Single frame of a trajectory.
#[derive(Debug, Clone)]
pub struct TrajectoryFrame {
    /// Step index.
    pub step: isize,
    /// Structure.
    pub system: MolecularSystem,
    /// Unbiased energy.
    pub energy: Option<Quantity<Hartree>>,
    /// Bias-including energy.
    pub biased_energy: Option<Quantity<Hartree>>,
    /// Gradient in `Hartree / Bohr`.
    pub gradient_bohr: Option<Col<f64>>,
}

/// Ordered sequence of structures.
#[derive(Debug, Clone, Default)]
pub struct Trajectory {
    /// Frames in order.
    pub frames: Vec<TrajectoryFrame>,
}

impl Trajectory {
    /// Create an empty trajectory.
    pub fn new() -> Self {
        Self { frames: Vec::new() }
    }

    /// Number of frames.
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    /// Return `true` if there are no frames.
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// Append a frame.
    pub fn push(&mut self, frame: TrajectoryFrame) {
        self.frames.push(frame);
    }
}
