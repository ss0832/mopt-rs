//! Evaluations of an earlier run offered to the calculators of a new one: the interface between
//! the calculator registry and the journals of a workflow that is restarted from its directory
//! (`workflow::journal::ReplayPool` gives them). A restarted workflow runs its commands again
//! from the start; every evaluation that the run before made is taken from there instead of
//! computed, so the commands reach the point where the run stopped without recomputing what
//! was done, and go on from there.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use crate::domain::{EvalOutput, EvalRequest, MolecularSystem};

/// An earlier evaluation: its result, the followed state it was for, and the environment the
/// calculator had relaxed (ONIOM microiterations).
#[derive(Debug, Clone)]
pub struct Replayed {
    /// The result.
    pub output: EvalOutput,
    /// The index of the state by energy that a followed state was (`pes::followed`).
    pub state_root: Option<usize>,
    /// The relaxed environment (flat, Bohr).
    pub environment: Option<Vec<f64>>,
}

/// A source of earlier evaluations.
pub trait ReplaySource: Send + Sync {
    /// The evaluation of `sys` for `req` by the calculator described by `identity`, recorded
    /// under `label`, taken from the source (each evaluation answers once), or `None`.
    fn take(
        &self,
        label: &str,
        identity: &str,
        sys: &MolecularSystem,
        req: &EvalRequest,
    ) -> Option<Replayed>;

    /// (evaluations offered, evaluations taken so far).
    fn counts(&self) -> (usize, usize);
}
