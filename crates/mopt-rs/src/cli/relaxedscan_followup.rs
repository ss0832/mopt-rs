//! The optimizations of the transition-state candidates of a scan (`-scan_opt
//! ts_optimize=on`): `optmain -order 1` from the structure of every candidate, with the
//! settings of the scan (the calculator, the surface, the user's constraints; not the scanned
//! coordinates), the Hessian of the calculator at the first point, and a vibrational analysis
//! that counts the imaginary frequencies. Candidates that end at the same structure are one
//! transition state (the later ones are marked as duplicates). The results are in
//! `ts_optimizations.csv` of the scan, the runs in `ts_optimizations/candidate_<n>/`.
//!
//! MultiOptPy: no counterpart (new in mopt-rs; as `nebmain -neb_opt optimize=saddles` does
//! after a chain).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::PathBuf;

use super::optmain::{self, OptmainConfig};
use super::relaxedscan::{distance, Scan, Surface};
use super::relaxedscan_outputs::candidate_file;
use super::relaxedscan_points::{read_structure, Structure};
use super::settings::{parse_settings, INPUT_KEY};
use super::spec::OPTMAIN;
use crate::core_infra::json::JsonValue;
use crate::domain::Units;
use crate::error::{MoptError, MoptResult, ResultExt};
use crate::pes::objective::ObjectiveRegistry;
use crate::pes::registry::CalculatorRegistry;
use crate::workflow::observers::STOP_FILE;
use crate::workflow::scan::{Candidate, CandidateKind, ScanState};

/// `-fc` of an optimization that gets the Hessian of the calculator at its first point only.
const AT_THE_START_ONLY: &str = "1000000";
/// Two optimized candidates closer than this (Angstrom amu^0.5, or the unit of the
/// variables) with energies within [`SAME_ENERGY`] are one transition state.
const SAME_PLACE: f64 = 0.01;
/// Hartree.
const SAME_ENERGY: f64 = 1e-6;

/// A transition-state candidate after its optimization.
#[derive(Debug, Clone)]
pub struct OptimizedCandidate {
    /// Its number among the candidates (1-based, as its file).
    pub candidate: usize,
    /// The grid point it came from.
    pub point: usize,
    /// Whether the optimization converged.
    pub converged: bool,
    /// Iterations.
    pub iterations: usize,
    /// Energy of the optimized structure with the bias potentials (Hartree).
    pub energy: f64,
    /// Imaginary frequencies of the optimized structure (structures).
    pub imaginary_modes: Option<usize>,
    /// The earlier candidate that ended at the same structure.
    pub duplicate_of: Option<usize>,
    /// The optimized structure.
    pub file: PathBuf,
}

/// The optmain settings of the optimization of candidate file `input`.
fn settings_for(
    scan: &Scan<'_>,
    input: &std::path::Path,
    plain: bool,
) -> MoptResult<OptmainConfig> {
    let text = |v: &str| JsonValue::String(v.to_string());
    let document = JsonValue::parse(&scan.config.settings_document)?;
    let mut settings = BTreeMap::new();
    if let Some(given) = document.as_object() {
        for (key, value) in given {
            let shared = OPTMAIN.iter().any(|o| o.spec.key == key);
            if shared && !["saddle_order", "initial_hessian", INPUT_KEY].contains(&key.as_str()) {
                settings.insert(key.clone(), value.clone());
            }
        }
    }
    settings.insert(
        INPUT_KEY.to_string(),
        JsonValue::Array(vec![text(&input.to_string_lossy())]),
    );
    settings.insert("saddle_order".into(), text("1"));
    if plain {
        settings.insert("calc_exact_hess".into(), text("1"));
    } else {
        settings.insert("frequency_analysis".into(), JsonValue::Bool(true));
        settings
            .entry("calc_exact_hess".to_string())
            .or_insert_with(|| text(AT_THE_START_ONLY));
    }
    let parsed = parse_settings(&JsonValue::Object(settings).to_string_compact(), OPTMAIN)?;
    OptmainConfig::from_args(&parsed)
}

/// Optimize every transition-state candidate of `candidates` to a saddle point.
pub(crate) fn optimize_candidates(
    scan: &Scan<'_>,
    candidates: &[Candidate],
    state: &ScanState,
    registry: &CalculatorRegistry,
    objectives: &ObjectiveRegistry,
) -> MoptResult<Vec<OptimizedCandidate>> {
    let saddles: Vec<&Candidate> = candidates
        .iter()
        .filter(|c| c.kind == CandidateKind::Saddle)
        .collect();
    if saddles.is_empty() {
        scan.log
            .info("No transition-state candidates to optimize (ts_optimize=on).");
        return Ok(Vec::new());
    }
    let plain = matches!(scan.surface, Surface::Plain { .. });
    let base = std::path::absolute(scan.dir.join("ts_optimizations"))?;
    let lowest = state
        .done
        .iter()
        .filter(|p| p.biased.is_finite())
        .map(|p| p.biased)
        .fold(f64::INFINITY, f64::min);
    let mut done: Vec<OptimizedCandidate> = Vec::new();
    let mut found: Vec<Structure> = Vec::new();
    for (k, c) in saddles.iter().enumerate() {
        let n = k + 1;
        let input = std::path::absolute(scan.dir.join(candidate_file(scan, n)))?;
        let mut config = settings_for(scan, &input, plain)?;
        let dir = base.join(format!("candidate_{n}"));
        config.restart_dir = Some(dir.clone());
        config.stop_file = Some(scan.dir.join(STOP_FILE));
        scan.log.info(&format!(
            "Transition-state candidate {n} (point {}): optmain -order 1 ({}).",
            c.point + 1,
            config.settings.methods.join(" ")
        ));
        let runs = optmain::run_with(&config, &base, None, registry)
            .context(format!("the optimization of candidate {n}"))?;
        let run = runs
            .into_iter()
            .next()
            .ok_or_else(|| MoptError::conv("the optimization gave no result"))?;
        let structure = read_structure(&run.optimized, &scan.surface.start())?;
        let (energy, iterations, imaginary) = match (&run.summary, &scan.surface, &structure) {
            (Some(s), _, _) => (s.biased_energy.value(), s.iterations, s.imaginary_modes),
            (None, Surface::Plain { .. }, Structure::Variables(x)) => {
                let objective = objectives.create_uncounted(
                    &scan.config.optmain.calculator,
                    &scan.config.optmain.calculator_options,
                )?;
                (objective.evaluate(x, false)?.value, 0, None)
            }
            _ => (f64::NAN, 0, None),
        };
        let duplicate_of = found
            .iter()
            .zip(&done)
            .find(|(s, o)| {
                o.duplicate_of.is_none()
                    && (o.energy - energy).abs() < SAME_ENERGY
                    && distance(s, &structure).is_some_and(|d| d < SAME_PLACE)
            })
            .map(|(_, o)| o.candidate);
        scan.log.info(&format!(
            "Candidate {n}: {} after {iterations} iteration(s), {:.4} kcal/mol above the lowest \
             point of the scan{}{}.",
            if run.converged {
                "converged"
            } else {
                "not converged"
            },
            (energy - lowest) * Units::HARTREE_TO_KCAL_MOL,
            imaginary.map_or_else(String::new, |m| format!(
                ", {m} imaginary frequenc{}",
                if m == 1 { "y" } else { "ies" }
            )),
            duplicate_of.map_or_else(String::new, |d| format!(", the same as candidate {d}"))
        ));
        if run.converged && imaginary.is_some_and(|m| m != 1) {
            scan.log.warn(&format!(
                "Candidate {n}: the optimization ended at a structure with {} imaginary \
                 frequencies, not a transition state.",
                imaginary.unwrap_or(0)
            ));
        }
        found.push(structure);
        done.push(OptimizedCandidate {
            candidate: n,
            point: c.point,
            converged: run.converged,
            iterations,
            energy,
            imaginary_modes: imaginary,
            duplicate_of,
            file: run.optimized.clone(),
        });
        if run.interrupted {
            break;
        }
    }
    let mut table = String::from(
        "candidate,point,converged,iterations,energy_hartree,relative_kcal_mol,imaginary_modes,\
         duplicate_of,file\n",
    );
    for o in &done {
        let _ = writeln!(
            table,
            "{},{},{},{},{:.12},{:.6},{},{},{}",
            o.candidate,
            o.point + 1,
            o.converged,
            o.iterations,
            o.energy,
            (o.energy - lowest) * Units::HARTREE_TO_KCAL_MOL,
            o.imaginary_modes
                .map_or_else(String::new, |m| m.to_string()),
            o.duplicate_of.map_or_else(String::new, |d| d.to_string()),
            o.file
                .strip_prefix(std::path::absolute(&scan.dir)?)
                .unwrap_or(&o.file)
                .display()
        );
    }
    std::fs::write(scan.dir.join("ts_optimizations.csv"), table)?;
    Ok(done)
}
