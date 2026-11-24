//! Drivers of the searches of `ieipmain`: the improved elastic image pair method, the dimer
//! method, the spring pair method, the two-point scaled hypersphere search, the growing
//! Newton trajectory, and the anharmonic downward distortion following. Each runs its state
//! machine on a [`SearchSurface`] (structures on a calculator, or an objective of plain
//! variables), writes MultiOptPy's files and a figure that is drawn anew after every
//! iteration, reports to the log, watches the `STOP` file, and calls a hook after every
//! iteration.
//!
//! MultiOptPy: `ieip.py:iEIP.optimize` and the methods of `OtherMethod/`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

pub mod addf_run;
pub mod files;
pub mod ieip_run;
pub mod internal;
pub mod point_runs;
pub mod registry;
pub mod registry_chains;
pub mod registry_experimental;
pub mod registry_methods;
pub mod registry_single;
pub mod registry_strings;
pub mod relax_path;
pub mod string_runs;
pub mod surface;

use std::path::Path;

pub use addf_run::{run_addf, AddfOutcome};
pub use ieip_run::{run_ieip, IeipOutcome};
pub use point_runs::{
    run_dimer, run_newton_trajectory, run_spring_pair, run_two_point_shs, PointOutcome,
};
pub use registry::{
    CountingSurface, Ends, PairCounts, PairInput, PairMethod, PairMethodRegistry, PairResult,
    PairSettings,
};
pub use registry_chains::{ChainCommandSettings, CommandSettings};
pub use string_runs::{run_dsafir, run_fsm, run_gsm, run_lmm};
pub use surface::{
    evaluate_all, ImageState, SearchSurface, StructureSurface, SurfaceValue, VariableSurface,
};

use crate::core_infra::log::Logger;

/// Evaluated images, kept separately until the two inward walks form a path.
#[derive(Default)]
pub(super) struct ImagePaths([Vec<(faer::Col<f64>, f64)>; 2]);

impl ImagePaths {
    pub(super) fn push(&mut self, images: &[faer::Col<f64>; 2], energies: [f64; 2]) {
        for ((side, x), energy) in self.0.iter_mut().zip(images).zip(energies) {
            if side.last().is_none_or(|(previous, _)| previous != x) {
                side.push((x.clone(), energy));
            }
        }
    }

    pub(super) fn join(self) -> Vec<(faer::Col<f64>, f64)> {
        let [mut first, second] = self.0;
        first.extend(second.into_iter().rev());
        first
    }
}

/// Where a search writes and logs.
pub struct SearchRun<'a> {
    /// The run directory.
    pub dir: &'a Path,
    /// The stems of the input files of the two images (MultiOptPy names its files after
    /// them).
    pub stems: [String; 2],
    /// The log (screen and `mopt.log`).
    pub log: &'a Logger,
    /// Save intermediate trajectories, tables and plots of the search.
    pub save_steps: bool,
    /// Draw the figures after every iteration.
    pub figures: bool,
    /// The settings of the members that are those of a command (the chain relaxation of
    /// `nebmain`, BITSS of `optmain -mf bitss`), read from their keys by the command layer
    /// (`cli::pair_settings::COMMAND_PARSERS`).
    pub commands: &'a dyn CommandSettings,
}

/// The resolved method parameters in a search directory.
pub const METHOD_SETTINGS_FILE: &str = "method_settings.txt";

impl SearchRun<'_> {
    fn record_settings(
        &self,
        method: &str,
        settings: &impl std::fmt::Debug,
    ) -> crate::error::MoptResult<()> {
        if self.save_steps {
            std::fs::write(
                self.dir.join(METHOD_SETTINGS_FILE),
                format!("{method}\n{settings:#?}\n"),
            )?;
        }
        Ok(())
    }
}

/// The points of an iteration into the log (the screen and `mopt.log`), each under its
/// label: structures in Cartesian coordinates (Angstrom) as `optmain` shows the structure of
/// an iteration, plain variables as their values. Each entry is the label, the point, and
/// the image whose state it is in.
pub fn log_structures(
    surface: &dyn SearchSurface,
    log: &Logger,
    points: &[(String, &faer::Col<f64>, usize)],
) {
    use std::fmt::Write as _;
    let mut text = String::new();
    for (label, x, image) in points {
        match surface.structure(x, *image) {
            Some(sys) => {
                let _ = write!(
                    text,
                    "{label}\n{}",
                    crate::workflow::observers::geometry_block(&sys)
                );
            }
            None => {
                let values: Vec<String> = x.iter().map(|v| format!("{v:.10}")).collect();
                let _ = writeln!(text, "{label}: {}", values.join(" "));
            }
        }
    }
    if !text.is_empty() {
        log.info(text.trim_end_matches('\n'));
    }
}

/// What the calculator said, into the log.
pub fn log_notes(surface: &dyn SearchSurface, log: &Logger) {
    for note in surface.take_notes() {
        if let Some(rest) = note.strip_prefix("Warning:") {
            // The log puts its own "Warning:" in front.
            log.warn(rest.trim_start());
        } else {
            log.info(&note);
        }
    }
}
