//! `ieipmain -mf NAME` with one of the methods of mopt-rs (`-dsafir`, `-gsm`, `-se_gsm`,
//! `-fsm`, `-lmm`): the method runs on the function `NAME` of two electronic states, the
//! state of the first structure and the state of the second.
//!
//! A function such as `seam` is low where the two states are close, so the points of
//! interest are the low points of the path: every point lower than both its neighbours is
//! written as a candidate for a crossing (`<stem>_crossing_guess.xyz`, `_2`, ...), with the
//! energies of the two states there. The search for the crossing from a candidate is
//! `optmain -mf opt_meci|opt_mesx`, which `-ieip_opt optimize=on` runs.
//!
//! MultiOptPy: no counterpart (new in mopt-rs; MultiOptPy's `ieipmain -mf` runs an
//! optimization on the function, which stays the meaning of `-mf` without these methods).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::path::{Path, PathBuf};

use faer::Col;

use super::ieipmain_followup::CrossingStates;
use super::ieipmain_methods::MethodOutcome;
use super::ieipmain_settings::IeipmainConfig;
use crate::core_infra::log::Logger;
use crate::domain::units::Units;
use crate::domain::{MolecularSystem, Quantity};
use crate::error::{MoptError, MoptResult};
use crate::io::xyz::write_xyz_file;
use crate::workflow::pair_search::StructureSurface;

/// The two states of the function: those of the first and of the last structure.
fn states_of(config: &IeipmainConfig, structures: &[MolecularSystem]) -> CrossingStates {
    let first = structures[0].spin_charge;
    let second = structures[structures.len() - 1].spin_charge;
    CrossingStates {
        first: (first.charge, first.multiplicity, config.excited_states[0]),
        second: (second.charge, second.multiplicity, config.excited_states[1]),
    }
}

/// `surface` with every point on the function of the states that `-mf` names next to a
/// method of mopt-rs; unchanged without one.
pub fn on_function<'a>(
    config: &IeipmainConfig,
    structures: &[MolecularSystem],
    surface: StructureSurface<'a>,
    log: &Logger,
) -> MoptResult<StructureSurface<'a>> {
    let Some(name) = &config.surface_function else {
        return Ok(surface);
    };
    let (charge, multiplicity, state) = states_of(config, structures).second;
    let spec = vec![
        name.clone(),
        charge.to_string(),
        multiplicity.to_string(),
        state.to_string(),
    ];
    // A function of the states, not a projection for crossings (refused as for an IRC).
    let function = super::irc::irc_states(&spec, config.excited_states[0])?
        .ok_or_else(|| MoptError::config("-mf: no function of the states"))?;
    log.info(&format!(
        "Mode: Model Function ({name}): the search runs on this function of the state of the \
         first structure and of the state of the second (charge {charge}, multiplicity \
         {multiplicity}, state {state})."
    ));
    Ok(surface.with_states(function))
}

/// The energies of the two states at `x` and how far apart they are, for the log.
fn states_at(surface: &StructureSurface<'_>, x: &Col<f64>) -> MoptResult<String> {
    let energies = surface.state_energies(x)?;
    Ok(match energies.as_slice() {
        [(a, ea), (b, eb), ..] => format!(
            "{a} {ea:.10}, {b} {eb:.10}, {:.3} kcal/mol apart",
            (eb - ea).abs() * Units::HARTREE_TO_KCAL_MOL
        ),
        _ => String::new(),
    })
}

/// What a search on a function of two states found: the energies of the states at its
/// estimates go to the log, and the points of the path that are lower than both their
/// neighbours are written as candidates for a crossing. Returns their files and the two
/// states, or nothing when the path has no such point.
pub fn crossing_candidates(
    config: &IeipmainConfig,
    structures: &[MolecularSystem],
    surface: &StructureSurface<'_>,
    outcome: &MethodOutcome,
    stem: &str,
    dir: &Path,
    log: &Logger,
) -> MoptResult<Option<(Vec<PathBuf>, CrossingStates)>> {
    let template = &structures[0];
    for (k, (x, _)) in outcome.transition_states.iter().enumerate() {
        log.info(&format!(
            "Estimate {} (the highest point of the function along the path): {}.",
            k + 1,
            states_at(surface, x)?
        ));
    }
    let path = outcome.path.as_deref().unwrap_or(&[]);
    let mut candidates = Vec::new();
    for (i, three) in path.windows(3).enumerate() {
        let (x, value) = &three[1];
        if !(*value < three[0].1 && *value < three[2].1) {
            continue;
        }
        let suffix = if candidates.is_empty() {
            String::new()
        } else {
            format!("_{}", candidates.len() + 1)
        };
        let file = dir.join(format!("{stem}_crossing_guess{suffix}.xyz"));
        write_xyz_file(
            &file,
            &template.with_flat_coords_bohr(x)?,
            Some(Quantity::new(*value)),
        )?;
        log.info(&format!(
            "Point {} of the path is a minimum of the function along it, a candidate for a \
             crossing: {} ({}).",
            i + 1,
            states_at(surface, x)?,
            file.display()
        ));
        candidates.push(file);
    }
    if candidates.is_empty() {
        log.info(
            "The function has no minimum along the path between its ends: no candidate for a \
             crossing.",
        );
        return Ok(None);
    }
    let states = states_of(config, structures);
    log.info(&format!(
        "A search for the crossing from a candidate: optmain FILE {} (-ieip_opt optimize=on \
         runs it).",
        states.options()
    ));
    Ok(Some((candidates, states)))
}
