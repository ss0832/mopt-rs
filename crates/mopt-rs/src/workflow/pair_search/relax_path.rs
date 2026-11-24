//! The relaxation of the path of a two-point method, a follow-up step of every method that
//! makes a path (`method_relax`, off by default): images spread evenly along the path of
//! the method, its ends fixed, relaxed by a chain method of `nebmain` (`lup`: the locally
//! updated planes, `nebmain -lup`; `neb`: `nebmain`'s default chain) with `nebmain`'s
//! settings, the same code path as a chain of `nebmain` (the settings read by the command
//! layer, [`super::CommandSettings::chain`]; the relaxation
//! [`crate::workflow::chain::relax_chain`]). The maxima of the relaxed chain replace the
//! estimates of the method, and the follow-up runs start from them. Points on hard
//! constraints are put back onto them after every iteration, as `nebmain` does.
//!
//! The keys (the vocabulary of the mapper's relaxation of the AFIR path,
//! `method_relax`/`method_relax_KEY`): `method_relax` (`off`, `lup`, `neb`),
//! `method_relax_iterations` (5), `method_relax_nodes` (0: by the spacing),
//! `method_relax_spacing` (0.5 Angstrom), `method_relax_max_step` (the longest
//! quasi-Newton step of an image, Bohr; the chain's own without it), and
//! `method_relax_options` (further options of `nebmain`). DS-AFIR's keys `lup=N`,
//! `lup_nodes`, `lup_spacing`, `lup_max_step` of earlier versions name this step with the
//! LUP chain ([`dsafir_alias`]).
//!
//! The procedure of the AFIR papers relaxes the AFIR path by the locally updated planes
//! (S. Maeda et al., J. Comput. Chem. 2018, 39, 233; R. Elber, I. Ulitsky, J. Chem. Phys.
//! 1990, 92, 1510); here any path of any two-point method can be relaxed so.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Col;

use super::registry::{PairInput, PairSettings};
use super::registry_chains::{chain_model, ChainOnSearch};
use super::surface::SearchSurface;
use super::SearchRun;
use crate::domain::Units;
use crate::error::{MoptError, MoptResult};
use crate::path::band::machine::{ChainEnd, ChainRelaxation, ChainReport};
use crate::path::band::trust::FixedEnds;
use crate::path::band::ChainKind;
use crate::path::string_nodes::{point_at_length, segment_lengths};
use crate::workflow::chain::{relax_chain, ChainObserver, Flow};

/// The keys of the relaxation step, with their meaning (every method of the registry takes
/// them).
pub const RELAX_KEYS: &[(&str, &str)] = &[
    (
        "method_relax",
        "relax the path of the method afterwards by a chain of nebmain: lup (nebmain -lup), \
         neb (nebmain's default chain), or off (the default)",
    ),
    (
        "method_relax_iterations",
        "iterations of that relaxation (5)",
    ),
    (
        "method_relax_nodes",
        "images of the relaxed path with its ends (0: by method_relax_spacing)",
    ),
    (
        "method_relax_spacing",
        "spacing of the images of the relaxed path (0.5 Angstrom)",
    ),
    (
        "method_relax_max_step",
        "longest quasi-Newton step of an image of the relaxation, in Bohr (the chain's own)",
    ),
    (
        "method_relax_options",
        "further options of nebmain for the relaxation, as on its command line",
    ),
];

/// DS-AFIR's keys of earlier versions that name the relaxation step with the LUP chain.
pub const DSAFIR_ALIAS_KEYS: &[(&str, &str)] = &[
    (
        "lup",
        "method_relax=lup with method_relax_iterations=N (0: none)",
    ),
    ("lup_nodes", "method_relax_nodes"),
    ("lup_spacing", "method_relax_spacing"),
    ("lup_max_step", "method_relax_max_step"),
];

/// The relaxation step of a path.
#[derive(Debug, Clone, PartialEq)]
pub struct PathRelaxation {
    /// The chain method: `lup` or `neb`.
    pub method: String,
    /// Iterations at most.
    pub iterations: usize,
    /// Images with the ends (0: by `spacing`).
    pub nodes: usize,
    /// Spacing of the images along the path (Bohr, or the unit of the variables).
    pub spacing: f64,
    /// Longest quasi-Newton step of an image (`None`: the chain's own).
    pub max_step: Option<f64>,
    /// Further options of `nebmain`.
    pub options: Vec<String>,
}

impl PathRelaxation {
    /// The relaxation of the keys `method_relax*` of `settings` (`None` without it or with
    /// `off`); unknown keys of that prefix are refused.
    pub fn from_settings(settings: &PairSettings, input: &PairInput) -> MoptResult<Option<Self>> {
        for key in settings.keys() {
            if key.starts_with("method_relax") && !RELAX_KEYS.iter().any(|(k, _)| *k == key) {
                let known: Vec<&str> = RELAX_KEYS.iter().map(|(k, _)| *k).collect();
                return Err(MoptError::parse(format!(
                    "{key}: no such setting of the relaxation of the path ({})",
                    known.join(", ")
                )));
            }
        }
        let method = settings
            .text("method_relax")
            .map(|m| m.to_ascii_lowercase());
        let method = match method.as_deref() {
            None | Some("off") => {
                // The other keys are read (so that they are not refused) but do nothing.
                for (key, _) in &RELAX_KEYS[1..] {
                    let _ = settings.text(key);
                }
                return Ok(None);
            }
            Some(m @ ("lup" | "neb")) => m.to_string(),
            Some("dmf") => {
                return Err(MoptError::unsupported(
                    "method_relax=dmf: the direct MaxFlux method replaces a path rather than \
                     relaxing its images; run it on the path with the registry method dmf \
                     (lup or neb here)",
                ))
            }
            Some(other) => {
                return Err(MoptError::parse(format!(
                    "method_relax: lup, neb, or off, got '{other}'"
                )))
            }
        };
        Ok(Some(Self {
            method,
            iterations: settings.usize("method_relax_iterations", 5)?.max(1),
            nodes: settings.usize("method_relax_nodes", 0)?,
            spacing: match settings.text("method_relax_spacing") {
                Some(_) => input.length(settings.f64("method_relax_spacing", 0.5)?),
                None => input.length(0.5),
            },
            max_step: settings
                .text("method_relax_max_step")
                .map(|_| settings.f64("method_relax_max_step", 0.0))
                .transpose()?,
            options: settings
                .text("method_relax_options")
                .map(|o| o.split_whitespace().map(str::to_string).collect())
                .unwrap_or_default(),
        }))
    }

    /// The `nebmain` options of the chain method with the further options.
    fn chain_options(&self) -> String {
        let mut words: Vec<String> = match self.method.as_str() {
            "lup" => vec!["-lup".to_string()],
            _ => Vec::new(),
        };
        words.extend(self.options.iter().cloned());
        words.join(" ")
    }
}

/// DS-AFIR's keys of earlier versions (`lup=N`, `lup_nodes`, `lup_spacing`, `lup_max_step`)
/// as the relaxation step with the LUP chain (`None` without `lup` or with `lup=0`); the
/// log says so.
pub fn dsafir_alias(
    settings: &PairSettings,
    input: &PairInput,
    run: &SearchRun,
) -> MoptResult<Option<PathRelaxation>> {
    let given = ["lup_nodes", "lup_spacing", "lup_max_step"]
        .iter()
        .any(|k| settings.text(k).is_some());
    let iterations = settings.usize("lup", 0)?;
    if iterations == 0 {
        if given {
            run.log.warn(
                "dsafir: lup_nodes, lup_spacing, and lup_max_step set the relaxation of the \
                 path, which lup=N (or method_relax=lup) turns on; without it they do nothing.",
            );
        }
        return Ok(None);
    }
    let relaxation = PathRelaxation {
        method: "lup".to_string(),
        iterations,
        nodes: settings.usize("lup_nodes", 0)?,
        spacing: match settings.text("lup_spacing") {
            Some(_) => input.length(settings.f64("lup_spacing", 0.5)?),
            None => input.length(0.5),
        },
        // The step of the relaxation of earlier versions: 0.1 Angstrom.
        max_step: Some(settings.f64("lup_max_step", 0.1 * Units::ANGSTROM_TO_BOHR)?),
        options: Vec::new(),
    };
    run.log.info(&format!(
        "dsafir lup={iterations}: the relaxation of the path after the method with the LUP \
         chain (method_relax=lup, method_relax_iterations={iterations}{}{}{}).",
        settings
            .text("lup_nodes")
            .map_or(String::new(), |v| format!(", method_relax_nodes={v}")),
        settings
            .text("lup_spacing")
            .map_or(String::new(), |v| format!(", method_relax_spacing={v}")),
        settings
            .text("lup_max_step")
            .map_or(String::new(), |v| format!(", method_relax_max_step={v}")),
    ));
    Ok(Some(relaxation))
}

/// The relaxed path (images with their energies) and why the relaxation ended.
#[derive(Debug, Clone)]
pub struct RelaxedPath {
    /// The images of the last iteration with their energies.
    pub path: Vec<(Col<f64>, f64)>,
    /// The maxima of the profile among the inner images (the highest inner image if there
    /// is none), the estimates of transition states.
    pub maxima: Vec<(Col<f64>, f64)>,
    /// Why the relaxation ended.
    pub end: Option<ChainEnd>,
    /// Iterations run.
    pub iterations: usize,
}

/// Refuse the relaxation where it cannot run: on two surfaces (the path of two structures
/// in different states has no single energy profile).
pub fn check_relaxation(surface: &dyn SearchSurface) -> MoptResult<()> {
    if surface.two_surfaces() {
        return Err(MoptError::unsupported(
            "method_relax: the relaxation of the path needs one surface; the two structures \
             are in different states",
        ));
    }
    Ok(())
}

/// Relax `path` (the path of a two-point method between its ends) as a chain of `nebmain`.
pub fn relax_path(
    relaxation: &PathRelaxation,
    path: &[(Col<f64>, f64)],
    input: &PairInput,
    surface: &dyn SearchSurface,
    run: &SearchRun,
) -> MoptResult<RelaxedPath> {
    check_relaxation(surface)?;
    let points: Vec<Col<f64>> = path.iter().map(|(x, _)| x.clone()).collect();
    let total: f64 = segment_lengths(&points).iter().sum();
    if !(total.is_finite() && total > 0.0) || points.len() < 2 {
        return Err(MoptError::calc(
            "method_relax: the path of the method has no length",
        ));
    }
    let count = if relaxation.nodes >= 3 {
        relaxation.nodes
    } else {
        ((total / relaxation.spacing).round() as usize + 1).max(3)
    };
    // Images evenly apart along the path, the ends where they are, on the constraints.
    let mut images = Vec::with_capacity(count);
    for k in 0..count {
        let (x, _) = point_at_length(&points, total * k as f64 / (count - 1) as f64)
            .ok_or_else(|| MoptError::calc("method_relax: the path has no length"))?;
        images.push(settle(input, &x)?);
    }
    let options = relaxation.chain_options();
    let pairs = PairSettings::new(vec![
        ("options".to_string(), options.clone()),
        ("iterations".to_string(), relaxation.iterations.to_string()),
    ]);
    let config = run.commands.chain(&pairs, run.dir, false)?;
    let mut chain = config.chain.clone();
    chain.kind = if input.structures {
        ChainKind::Atoms
    } else {
        ChainKind::Variables
    };
    chain.max_iterations = relaxation.iterations;
    chain.ends = FixedEnds {
        first: true,
        last: true,
    };
    chain.align &= input.structures && input.geometry.aligns();
    chain.fixed_components = input.geometry.frozen.clone();
    if let (Some(rfo), Some(step)) = (chain.rfo.as_mut(), relaxation.max_step) {
        rfo.inner_step = step;
    }
    run.log.info(&format!(
        "Relaxation of the path (method_relax={}): a chain of nebmain{}, {count} images, at \
         most {} iterations, the ends fixed.",
        relaxation.method,
        if options.is_empty() {
            String::new()
        } else {
            format!(" ({options})")
        },
        chain.max_iterations
    ));
    let mut m = ChainRelaxation::new(chain, images)?;
    if let (Some(t), true) = (&input.template, input.structures) {
        m = m.with_structure(t.clone())?;
    }
    let mut adapter = ChainOnSearch::new(surface);
    if input.structures {
        if let Some(model) = chain_model(&config)? {
            adapter = adapter.with_model_hessian(model);
        }
    }
    let mut keep = KeepOnConstraints { input };
    let mut log = |report: &ChainReport, _: &mut ChainRelaxation| -> MoptResult<Flow> {
        run.log.info(&format!(
            "Relaxation of the path, iteration {}: largest force {:.3e}",
            report.iteration, report.largest_force
        ));
        Ok(Flow::Continue)
    };
    let mut observers: Vec<&mut dyn ChainObserver> = vec![&mut keep, &mut log];
    let o = relax_chain(&mut m, &adapter, &mut observers)?;
    let last = o
        .last
        .ok_or_else(|| MoptError::calc("method_relax: the relaxation made no iteration"))?;
    let path: Vec<(Col<f64>, f64)> = last
        .chain
        .iter()
        .cloned()
        .zip(last.energies.iter().copied())
        .collect();
    let mut maxima: Vec<(Col<f64>, f64)> = last
        .maxima
        .iter()
        .filter_map(|&k| path.get(k).cloned())
        .collect();
    if maxima.is_empty() {
        let n = path.len();
        if let Some(top) = path
            .iter()
            .enumerate()
            .filter(|(k, _)| *k > 0 && *k + 1 < n)
            .max_by(|a, b| a.1 .1.total_cmp(&b.1 .1))
        {
            maxima.push(top.1.clone());
        }
    }
    run.log.info(&format!(
        "Relaxation of the path: {} after {} iterations; {} maxima of the relaxed chain are the \
         estimates.",
        match o.end {
            Some(ChainEnd::Converged) => "converged",
            Some(ChainEnd::IterationLimit) => "the iterations are used up",
            None => "stopped",
        },
        o.iterations,
        maxima.len()
    ));
    Ok(RelaxedPath {
        path,
        maxima,
        end: o.end,
        iterations: o.iterations,
    })
}

/// `x` on the hard constraints of the points (itself without any).
pub(super) fn settle(input: &PairInput, x: &Col<f64>) -> MoptResult<Col<f64>> {
    let x = input.geometry.settle(x)?;
    match &input.step_map {
        Some(map) if input.geometry.manifold.is_none() => map(&x, &x),
        _ => Ok(x),
    }
}

/// The images put back onto the hard constraints after every iteration.
pub(super) struct KeepOnConstraints<'a> {
    pub(super) input: &'a PairInput,
}

impl ChainObserver for KeepOnConstraints<'_> {
    fn after_iteration(
        &mut self,
        _report: &ChainReport,
        machine: &mut ChainRelaxation,
    ) -> MoptResult<Flow> {
        if self.input.geometry.manifold.is_none() && self.input.step_map.is_none() {
            return Ok(Flow::Continue);
        }
        let chain = machine
            .chain()
            .iter()
            .map(|x| settle(self.input, x))
            .collect::<MoptResult<Vec<_>>>()?;
        machine.set_chain(chain)?;
        Ok(Flow::Continue)
    }
}
