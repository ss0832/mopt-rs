//! Path peaks: the highest point of a path whose guesses gave no verified transition state
//! (`path_peaks=failed`, the default; `all`: of every path; `off`: none), saved as a record
//! `PP` with the minima on its two sides and the path between them. A path peak is saved and is not in the network (user decision, 2026-11-19): it
//! is no node and no edge, the report does not draw it, the kinetics does not use it, and it
//! joins nothing.
//!
//! The ends come from the nearest minima of the energy along the path on both sides of the
//! peak (from the peak downwards as long as the energy falls), each minimized like the end of
//! an IRC and tested for separation. A peak whose two sides fall to one minimum is not kept.
//! The rules are in `docs/source/mapper_design.md` ("Path peaks").
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use super::afir::local_maxima;
use super::configuration::frame_extension;
use super::dissociation::dissociation_of;
use super::method::{Job, JobContext, JobOutcome, OptimizeRequest, PathPoint, Stage, Stationary};
use super::network::Level;
use super::observers::PATHS_DIR;
use super::Configuration;
use crate::error::MoptResult;

/// Which path peaks a run saves (`path_peaks`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PeakMode {
    /// None.
    Off,
    /// Those of the paths that gave no transition state (the default).
    #[default]
    Failed,
    /// That of every path.
    All,
}

impl PeakMode {
    /// The mode of the setting `text` (`failed`, `all`, `off`; `on` is `failed`).
    pub fn parse(text: &str) -> MoptResult<Self> {
        match text.trim().to_ascii_lowercase().as_str() {
            "failed" | "on" | "true" | "yes" | "1" => Ok(Self::Failed),
            "all" => Ok(Self::All),
            "off" | "none" | "false" | "no" | "0" => Ok(Self::Off),
            other => Err(crate::error::MoptError::parse(format!(
                "path_peaks={other}: failed (the peak of a path that gave no transition state, \
                 the default), all, or off"
            ))),
        }
    }

    /// Its name in the settings.
    pub fn name(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Failed => "failed",
            Self::All => "all",
        }
    }
}

/// What a path peak rests on.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PeakBasis {
    /// The method of the path (`neb of the AFIR path`, `AFIR path`, `gsm`, ...).
    pub method: String,
    /// The points of the path.
    pub points: usize,
    /// Whether the path method ended on its criteria.
    pub converged: bool,
    /// The transition-state searches from the maxima of the path that were tried and gave no
    /// transition state.
    pub searches: usize,
}

/// A path peak a job found.
#[derive(Debug, Clone)]
pub struct FoundPeak {
    /// The peak (no wavenumbers: it is no stationary point).
    pub point: Stationary,
    /// The minima (or dissociation channels) on its two sides, the side of the start of the
    /// path first; `None` where the minimization did not converge.
    pub ends: [Option<Stationary>; 2],
    /// What it rests on.
    pub basis: PeakBasis,
    /// The points of the path between the two sides, the peak among them.
    pub path: Vec<PathPoint>,
    /// The minimization of each side, from the path to the minimum.
    pub settles: [Vec<PathPoint>; 2],
}

/// A path peak as the network file keeps it: a record beside the network, no node of it.
#[derive(Debug, Clone)]
pub struct PathPeak {
    /// Its number (index into [`super::Network::path_peaks`]; `PP0`, `PP1`, ...).
    pub id: usize,
    /// The level it was computed at.
    pub level: Level,
    /// The peak: a structure, or a point of plain variables.
    pub system: Configuration,
    /// Its energy (Hartree).
    pub energy: f64,
    /// The minima (or channels) on its two sides (nodes of the network), where they are known.
    pub ends: [Option<usize>; 2],
    /// `E(peak) - E(end)` for both ends where they are known (Hartree).
    pub barriers: [Option<f64>; 2],
    /// What it rests on.
    pub basis: PeakBasis,
    /// The job that found it.
    pub job: Option<usize>,
}

impl PathPeak {
    /// `PP3`.
    pub fn label(&self) -> String {
        format!("PP{}", self.id)
    }
}

/// The structure of the path peak `PP<id>`, relative to the run directory.
pub fn peak_structure_file(id: usize, variables: bool) -> String {
    format!(
        "{}/PP{id:06}.{}",
        super::store::STRUCTURES_DIR,
        frame_extension(variables)
    )
}

/// The path of the path peak `PP<id>` between its two ends, relative to the run directory.
pub fn peak_path_file(id: usize, variables: bool) -> String {
    format!("{PATHS_DIR}/PP{id:06}_path.{}", frame_extension(variables))
}

/// The highest inner maximum of the energy along `path`.
pub fn peak_index(path: &[PathPoint]) -> Option<usize> {
    local_maxima(path).first().copied()
}

/// The point of `path` where the energy stops falling from the point `peak` towards the start
/// (`forward` false) or the end of the path: the nearest minimum of the path energy on that
/// side, or the end of the path.
pub fn side_minimum(path: &[PathPoint], peak: usize, forward: bool) -> usize {
    let mut at = peak.min(path.len().saturating_sub(1));
    loop {
        let next = if forward {
            (at + 1 < path.len()).then_some(at + 1)
        } else {
            at.checked_sub(1)
        };
        match next {
            Some(n) if path[n].energy <= path[at].energy => at = n,
            _ => return at,
        }
    }
}

/// Keep the highest maximum of `path` as a path peak in `outcome` (see the module text), where
/// the run keeps the peak of such a path: `found` says whether the job got a transition state
/// from the path (a new one or a known one), and `basis` what the path is. Returns `false`
/// when the run was stopped.
pub fn keep_path_peak(
    job: &Job,
    ctx: &mut JobContext<'_>,
    path: &[PathPoint],
    found: bool,
    basis: PeakBasis,
    outcome: &mut JobOutcome,
) -> MoptResult<bool> {
    match ctx.jobs.path_peaks() {
        PeakMode::Off => return Ok(true),
        PeakMode::Failed if found => return Ok(true),
        PeakMode::Failed | PeakMode::All => {}
    }
    let Some(at) = peak_index(path) else {
        return Ok(true);
    };
    let sides = [side_minimum(path, at, false), side_minimum(path, at, true)];
    let mut ends: [Option<Stationary>; 2] = [None, None];
    let mut settles: [Vec<PathPoint>; 2] = [Vec::new(), Vec::new()];
    for (s, name) in ["backward", "forward"].into_iter().enumerate() {
        let end = ctx.jobs.optimize(
            &path[sides[s]].system,
            &OptimizeRequest {
                label: format!("path peak end ({name})"),
                stage: Stage::Minimum,
                terms: Vec::new(),
            },
            ctx.frames,
        )?;
        if end.interrupted {
            return Ok(false);
        }
        if !end.converged {
            outcome.warnings.push(format!(
                "the minimization of the {name} side of a path peak did not converge; that end \
                 of the peak is not known"
            ));
            continue;
        }
        let molecules = dissociation_of(ctx, &end.system, false, &mut outcome.warnings)?;
        if ctx.jobs.stop_requested() {
            return Ok(false);
        }
        let apart = molecules.as_ref().is_some_and(|d| d.separated);
        settles[s] = end.path;
        ends[s] = Some(Stationary {
            index_wavenumbers: if apart { None } else { end.index_wavenumbers },
            energy: match &molecules {
                Some(d) if apart => d.energy,
                _ => end.energy,
            },
            system: end.system,
            wavenumbers: if apart { None } else { end.wavenumbers },
            dissociation: molecules,
        });
    }
    match &ends {
        [None, None] => {
            ctx.jobs.note(&format!(
                "Job {}: neither side of the highest point of the path was minimized; no path \
                 peak is kept.",
                job.id
            ));
            return Ok(true);
        }
        [Some(a), Some(b)]
            if a.system
                .same_as(a.energy, &b.system, b.energy, ctx.matcher)? =>
        {
            ctx.jobs.note(&format!(
                "Job {}: both sides of the highest point of the path fall to the same minimum; \
                 no path peak is kept.",
                job.id
            ));
            return Ok(true);
        }
        _ => {}
    }
    let peak = &path[at];
    ctx.jobs.note(&format!(
        "Job {}: {} ({}); its highest point (point {} of {}, E = {:.10} Hartree) is saved as a \
         path peak, which is no verified transition state and no part of the network.",
        job.id,
        if found {
            "the peak of the path"
        } else {
            "no transition state from the path"
        },
        basis.method,
        at + 1,
        path.len(),
        peak.energy
    ));
    outcome.peaks.push(FoundPeak {
        point: Stationary {
            index_wavenumbers: None,
            system: peak.system.clone(),
            energy: peak.energy,
            wavenumbers: None,
            dissociation: None,
        },
        ends,
        basis,
        path: path[sides[0]..=sides[1]].to_vec(),
        settles,
    });
    Ok(true)
}

/// Follow the `guesses` taken from `path` ([`super::follow::follow_guesses`]) and keep the
/// peak of the path where the run keeps it: with `path_peaks=failed` when none of the guesses
/// led to a transition state. `method` names the path, `converged` says whether its method
/// ended on its criteria. Returns `false` when the run was stopped.
pub fn follow_with_peak(
    job: &Job,
    ctx: &mut JobContext<'_>,
    guesses: &[super::follow::Guess],
    path: &[PathPoint],
    method: &str,
    converged: bool,
    outcome: &mut JobOutcome,
) -> MoptResult<bool> {
    let before = (outcome.reactions.len(), outcome.known.len());
    if !super::follow::follow_guesses(job, ctx, guesses, outcome)? {
        return Ok(false);
    }
    let found = (outcome.reactions.len(), outcome.known.len()) != before;
    keep_path_peak(
        job,
        ctx,
        path,
        found,
        PeakBasis {
            method: method.to_string(),
            points: path.len(),
            converged,
            searches: guesses.len(),
        },
        outcome,
    )
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::workflow::mapper::configuration::VariablePoint;
    use crate::workflow::mapper::Configuration;

    fn path(energies: &[f64]) -> Vec<PathPoint> {
        energies
            .iter()
            .enumerate()
            .map(|(k, &e)| PathPoint {
                system: Configuration::Variables(VariablePoint {
                    names: std::sync::Arc::new(vec!["x".to_string()]),
                    values: faer::Col::from_fn(1, |_| k as f64),
                }),
                energy: e,
                biased_energy: e,
            })
            .collect()
    }

    #[test]
    fn the_peak_is_the_highest_inner_maximum_and_its_sides_the_nearest_minima() {
        // A shallow maximum, a minimum, the highest maximum, a shoulder, the end.
        let p = path(&[0.0, 0.4, 0.2, 1.0, 0.6, 0.6, 0.3, 0.5]);
        assert_eq!(peak_index(&p), Some(3));
        assert_eq!(side_minimum(&p, 3, false), 2);
        assert_eq!(side_minimum(&p, 3, true), 6);
        // A path that only rises or only falls has no peak.
        assert_eq!(peak_index(&path(&[0.0, 0.1, 0.2])), None);
        assert_eq!(peak_index(&path(&[0.3, 0.1, 0.0])), None);
        // The sides of a peak next to the ends of the path are those ends.
        let q = path(&[0.0, 1.0, 0.5]);
        assert_eq!(peak_index(&q), Some(1));
        assert_eq!(
            (side_minimum(&q, 1, false), side_minimum(&q, 1, true)),
            (0, 2)
        );
    }
}
