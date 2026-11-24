//! Excited states along the frames of a path: their energies and the overlaps of the
//! states of neighbouring frames, which is what a diabatization takes
//! ([`crate::analysis::diabatic`]).
//!
//! The states are named by their numbers at the first frame. At every next frame they are
//! the states (among the lowest ones) that hold most of the states of the frame before,
//! so that another state that passes through them in energy is not taken for one of them.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Mat;

use super::orbital_analysis::arc_lengths;
use crate::domain::{EvalRequest, MolecularSystem};
use crate::error::{MoptError, MoptResult, ResultExt};
use crate::pes::calculator::{evaluate_checked, Calculator};

/// How many states above the highest one that is followed are looked at for the states of
/// the next frame (the window of [`crate::pes::followed`]).
const WINDOW: usize = 6;

/// Excited states along the frames of a path.
#[derive(Debug, Clone, PartialEq)]
pub struct StateSet {
    /// Positions of the frames along the path (amu^(1/2) Bohr).
    pub arc_lengths: Vec<f64>,
    /// Energies of the ground state (Hartree).
    pub ground: Vec<f64>,
    /// For every frame, the energies of the states (Hartree, with the ground state), in
    /// ascending order.
    pub energies: Vec<Vec<f64>>,
    /// For every frame, the numbers of the states among its excited states (1: the
    /// lowest), in ascending order.
    pub numbers: Vec<Vec<usize>>,
    /// For every two neighbouring frames, the overlaps of the states of the first (rows)
    /// with the states of the second (columns).
    pub overlaps: Vec<Mat<f64>>,
}

/// The excited states `states` (numbers at the first frame, 1: the lowest; two at least,
/// each once) along `frames`, with the states `calc` hands out
/// ([`Calculator::state_characters`]).
pub fn states_along(
    calc: &dyn Calculator,
    frames: &[MolecularSystem],
    states: &[usize],
) -> MoptResult<StateSet> {
    let mut set: Vec<usize> = states.to_vec();
    set.sort_unstable();
    set.dedup();
    if set.len() < 2 || set.len() != states.len() || set[0] == 0 {
        return Err(MoptError::config(format!(
            "two or more different excited states are needed (1: the lowest), got {states:?}"
        )));
    }
    if frames.len() < 2 {
        return Err(MoptError::config(
            "states are followed along the frames of a path (two frames at least)",
        ));
    }
    let count = set.len();
    // From here on the numbers count from 0.
    for state in &mut set {
        *state -= 1;
    }
    let mut out = StateSet {
        arc_lengths: arc_lengths(frames)?,
        ground: Vec::with_capacity(frames.len()),
        energies: Vec::with_capacity(frames.len()),
        numbers: Vec::with_capacity(frames.len()),
        overlaps: Vec::with_capacity(frames.len() - 1),
    };
    let mut before = None;
    for (k, sys) in frames.iter().enumerate() {
        let of_frame = || format!("frame {k} of the path");
        let ground = evaluate_checked(calc, sys, &EvalRequest::ENERGY_ONLY)
            .context(of_frame())?
            .energy
            .value();
        let highest = set[count - 1];
        let found = calc
            .state_characters(sys, highest + 1 + WINDOW)
            .context(of_frame())?
            .ok_or_else(|| {
                MoptError::unsupported(format!(
                    "calculator '{}' does not hand out its excited states in a form that can \
                     be compared between structures",
                    calc.name()
                ))
            })?;
        if found.count() <= highest {
            return Err(MoptError::calc(format!(
                "{}: {} excited states, state {} is not among them",
                of_frame(),
                found.count(),
                highest + 1
            )));
        }
        if let Some(last) = &before {
            let (overlap, _) =
                crate::pes::states::StateCharacters::overlaps(last, &found).context(of_frame())?;
            // The states here that hold most of the states of the frame before.
            let weight = |c: usize| set.iter().map(|&r| overlap[(r, c)].powi(2)).sum::<f64>();
            let mut columns: Vec<usize> = (0..overlap.ncols()).collect();
            columns.sort_by(|&a, &b| weight(b).total_cmp(&weight(a)));
            if columns.len() < count {
                return Err(MoptError::calc(format!(
                    "{}: {} excited states for the {count} that are followed",
                    of_frame(),
                    columns.len()
                )));
            }
            let mut next: Vec<usize> = columns[..count].to_vec();
            next.sort_unstable();
            out.overlaps.push(Mat::from_fn(count, count, |i, j| {
                overlap[(set[i], next[j])]
            }));
            set = next;
        }
        out.ground.push(ground);
        out.energies
            .push(set.iter().map(|&s| ground + found.energies[s]).collect());
        out.numbers.push(set.iter().map(|&s| s + 1).collect());
        before = Some(found);
    }
    Ok(out)
}
