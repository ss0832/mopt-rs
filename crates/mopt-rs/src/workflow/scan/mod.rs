//! The parts of a scan (`relaxedscan`) that do no calculation: the points and their order
//! ([`grid`]), the structures of rigid scans ([`rigid`]), the stationary points of the scanned
//! surface ([`candidates`]), jumps between branches of a map ([`continuity`]), the state and
//! tables of the points ([`output`]), and the figures ([`figures`]). The command
//! (`cli::relaxedscan`) runs the points with `optmain`.
//!
//! MultiOptPy: `Entrypoints/relaxed_scan.py:run_relaxedscan`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

pub mod candidates;
pub mod continuity;
pub mod figures;
pub mod grid;
pub mod output;
pub mod rigid;

pub use candidates::{stationary_points, stationary_points_periodic, Candidate, CandidateKind};
pub use grid::{GridKind, GridOrder, ScanAxis, ScanGrid};
pub use output::{ScanPoint, ScanState};
