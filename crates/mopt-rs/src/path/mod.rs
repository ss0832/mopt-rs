//! Searches that work on more than one structure at a time: pairs and chains of images
//! between two basins.
//!
//! The state machines here ask for energies and gradients of points and never evaluate
//! anything themselves, like the optimizers of [`crate::opt`]; a point is a plain vector of
//! `n` variables (the `3N` Cartesian coordinates of a structure, or any other variables).
//!
//! MultiOptPy: `ModelFunction/binary_image_ts_search_model_function.py` (BITSS), `neb.py`
//! (chains of images).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

pub mod band;
pub mod bitss;
pub mod dmf;
pub mod interp;
// Two-structure and single-ended searches of `ieipmain`.
pub mod dimer;
pub mod dsafir;
pub mod fsm;
pub mod gsm;
pub mod hisd;
pub mod ieip;
pub mod lmm;
pub mod manifold;
pub mod newton_trajectory;
pub mod points;
pub mod shs;
pub mod spring_pair;
pub mod string_nodes;
pub mod twopshs;

/// The step length of Barzilai and Borwein, `s.y / y.y` (`s`, `y`: the changes of the point and
/// of the gradient), or `fallback` where it is not positive (`s.y <= 0` or `y = 0`).
///
/// MultiOptPy: no counterpart (new in mopt-rs; one rule for the steps of HiSD, of the ridge
/// band, and of the local minimax method, which clips it).
pub fn barzilai_borwein(sy: f64, yy: f64, fallback: f64) -> f64 {
    if sy > 0.0 && yy > 0.0 {
        sy / yy
    } else {
        fallback
    }
}
