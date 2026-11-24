//! `mdmain -coord NAME`: the coordinate system of the dynamics.
//!
//! By default the dynamics runs in Cartesian coordinates and the coordinates serve for the
//! output (`<stem>_coordinates.csv`): in the measured runs (water, hydrogen cyanide, ethanol
//! with GFN-FF) the dynamics in generalized coordinates conserved the energy exactly as well
//! as velocity Verlet at the same time step, at 10 to 60 times the cost per step, and it
//! broke down for ethanol at 1 fs (see the documentation of mdmain). With
//! `-md_opt integrate=coordinates` a non-redundant system (`dic`, `tric`, or a registered one
//! such as spherical coordinates) is integrated in its own coordinates with the generalized
//! leapfrog of [`crate::md::generalized`]; redundant systems (the primitive sets `ric` and
//! `tric-p` of a molecule with more primitives than motions; decided from the eigenvalues of
//! the kinetic metric at the start), constraints, frozen modes, a barostat, a periodic cell,
//! or bias potentials on internal coordinates are then refused with the reason.
//!
//! MultiOptPy: no counterpart (new in mopt-rs; MultiOptPy's dynamics is Cartesian).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::path::Path;

use super::mdmain::MdmainConfig;
use super::mdmain_settings::Integration;
use crate::coord::system::{CoordinateRegistry, CoordinateSystem};
use crate::core_infra::log::{python_float, Logger};
use crate::domain::{MolecularSystem, Units};
use crate::error::MoptResult;
use crate::md::generalized::{GeneralizedDynamics, Span};
use crate::md::MdSettings;
use crate::workflow::md_coordinates::MdCoordinates;

fn create(name: &str) -> MoptResult<Box<dyn CoordinateSystem>> {
    CoordinateRegistry::default().create(name)
}

/// Whether `-coord` names Cartesian coordinates (the default).
pub fn is_cartesian(name: &str) -> bool {
    matches!(
        name.trim().to_ascii_lowercase().as_str(),
        "" | "cart" | "cartesian"
    )
}

/// The dynamics in the coordinates of `-coord`, when they are to be integrated in;
/// `None`: the dynamics runs in Cartesian coordinates (the reason is in the log).
pub(super) fn machine(
    config: &MdmainConfig,
    settings: &MdSettings,
    start: &MolecularSystem,
    invariant: bool,
    log: &Logger,
) -> MoptResult<Option<GeneralizedDynamics>> {
    let name = config.coordinates.as_str();
    if is_cartesian(name) {
        if config.run.integrate == Integration::Coordinates {
            log.info("-md_opt integrate=coordinates with Cartesian coordinates: velocity Verlet.");
        }
        return Ok(None);
    }
    if config.run.integrate == Integration::Cartesian {
        log.info(&format!(
            "Coordinates '{name}': the dynamics runs in Cartesian coordinates (the default; \
             -md_opt integrate=coordinates runs it in them); the coordinates are written along \
             the run."
        ));
        return Ok(None);
    }
    GeneralizedDynamics::new(settings.clone(), create(name)?, start.clone(), invariant).map(Some)
}

/// The observer that writes the coordinates of `-coord` at the frames (none for
/// Cartesian coordinates).
pub(super) fn output(
    config: &MdmainConfig,
    start: &MolecularSystem,
    dir: &Path,
    stem: &str,
) -> MoptResult<Option<MdCoordinates>> {
    if is_cartesian(&config.coordinates) {
        return Ok(None);
    }
    MdCoordinates::new(
        create(&config.coordinates)?,
        start,
        dir,
        stem,
        config.run.write_every,
    )
    .map(Some)
}

/// The lines that start a run in generalized coordinates.
pub(super) fn announce(md: &GeneralizedDynamics, log: &Logger) {
    let s = md.settings();
    let (integrator, thermostat) = md.part_names();
    let span = match md.span() {
        Span::Full => "all Cartesian motions".to_string(),
        Span::Internal(k) => format!("the internal motions (without the {k} rigid motions)"),
    };
    log.info(&format!(
        "Molecular dynamics in the coordinates '{}' ({} of them, covering {span}): {} steps \
         of {} a.u. ({} fs), integrator {integrator}, thermostat {thermostat}{}, temperature \
         {} K, seed {}",
        md.coordinates(),
        md.dim(),
        s.steps,
        python_float(s.time_step),
        python_float(s.time_step * Units::AU_TIME_TO_FS),
        if thermostat == "none" {
            String::new()
        } else {
            format!(
                " (coupling time {} fs)",
                python_float(s.thermostat_settings.tau * Units::AU_TIME_TO_FS)
            )
        },
        python_float(s.temperature),
        s.seed
    ));
    log.info(&format!(
        "Kinetic energy 1/2 p^T G p with G = B M^-1 B^T; conserved motions: {}",
        md.rigid().name()
    ));
    for paper in md.references() {
        log.info(&format!("Reference: {paper}"));
    }
}
