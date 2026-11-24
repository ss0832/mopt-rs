//! One point of a scan: the constrained optimization of a structure with `optmain` (every
//! option of optmain applies: the calculator, `-mf`, bias potentials, the user's constraints,
//! `-coord`, ONIOM with its microiterations, `-vc`), a single point (`-rigid`: one evaluation
//! of the same surface), or the same for an objective of plain variables (the constrained
//! optimizer of plain variables, [`crate::opt::vector_constraints`]).
//!
//! Each point has its own directory under the scan: the files of its optmain run (log,
//! trajectory, `evaluations.journal`, which a restart of the scan replays) and its start
//! structure. The scanned coordinates are added to the constraints of the run and, where they
//! can be written as `-pc manual` or `-uc`, to its `settings.json`. A `STOP` file in the
//! directory of the scan stops the point at once.
//!
//! MultiOptPy: the call `multioptpy.optimization.Optimize(args).run()` with
//!  `args.projection_constrain` set to the scan values in `Entrypoints/relaxed_scan.py`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use faer::Col;

use super::constraints::projection_constraint;
use super::optmain::{self, OptmainConfig};
use super::variables::{format_variables, vector_settings};
use crate::core_infra::json::JsonValue;
use crate::core_infra::log::{Level, Logger};
use crate::domain::{MolecularSystem, Units};
use crate::error::{MoptError, MoptResult, ResultExt};
use crate::io::xyz::{read_xyz_file, write_xyz_file};
use crate::opt::convergence::ConvergenceCriteria;
use crate::opt::state_machine::Termination;
use crate::opt::vector::VectorOptimizer;
use crate::opt::vector_constraints::VectorConstraintSet;
use crate::pes::constraint::Constraint;
use crate::pes::objective::Objective;
use crate::pes::registry::CalculatorRegistry;
use crate::workflow::observers::StopFile;
use crate::workflow::run_vector_optimization_until;

/// The structure of a point (one kind per scan, so the size of the variants does not matter).
#[derive(Debug, Clone)]
#[allow(clippy::large_enum_variant)]
pub enum Structure {
    /// Atoms.
    Atoms(MolecularSystem),
    /// Plain variables.
    Variables(Col<f64>),
}

/// How a point went.
#[derive(Debug, Clone)]
pub struct PointOutcome {
    /// Energy without bias potentials (Hartree).
    pub energy: f64,
    /// Energy with them.
    pub biased: f64,
    /// Whether the optimization converged (true for single points).
    pub converged: bool,
    /// Iterations.
    pub iterations: usize,
    /// Stopped by a `STOP` file.
    pub interrupted: bool,
    /// The last evaluated structure (the optimized one when it converged).
    pub structure: Structure,
}

/// The scanned coordinates of a point as constraints and as the tokens of its settings.
#[derive(Debug, Clone, Default)]
pub struct PointConstraints {
    /// Constraints added to those of the run.
    pub constraints: Vec<Constraint>,
    /// `(option, tokens)` of `-pc manual` and `-uc` that say the same in the settings.
    pub tokens: Vec<(&'static str, Vec<String>)>,
    /// The Hessian the optimization starts from (the last one of the neighbour it starts
    /// from), as `-ih`.
    pub first_hessian: Option<PathBuf>,
    /// Where the optimization leaves its last Hessian (for the points that start from it).
    pub keep_hessian: Option<PathBuf>,
}

/// File of the last Hessian of the optimization of a point, in its directory.
pub const FINAL_HESSIAN: &str = "final_hessian.npy";

/// What every point of a scan of structures shares.
pub struct MoleculePoints<'a> {
    /// The optmain settings of every point.
    pub base: &'a OptmainConfig,
    /// `-tcc`, `-lcc`.
    pub criteria_flags: (bool, bool),
    /// Optimize (false: one evaluation).
    pub relax: bool,
    /// The `STOP` file of the scan.
    pub stop_file: PathBuf,
    /// The screen output of the optimizations (none: only their logs).
    pub console: Option<Level>,
    /// The calculators.
    pub registry: &'a CalculatorRegistry,
}

/// The user's `-pc` tokens of the automatic form (`NAME ATOMS ...`) in the manual form
/// (`manual NAME ATOMS VALUE ...`) with the values the run holds: those of `resolved` (the
/// constraints resolved at the input of the scan); `None` when one of them has no value to
/// write (`rot`, `eigvec`, `atoms_pair`) or is not among `resolved`.
///
/// MultiOptPy: no counterpart (new in mopt-rs).
fn manual_form(automatic: &[String], resolved: &[Constraint]) -> Option<Vec<String>> {
    if !automatic.len().is_multiple_of(2) {
        return None;
    }
    let mut out = Vec::with_capacity(2 * automatic.len());
    for pair in automatic.chunks(2) {
        let (name, atoms) = (&pair[0], &pair[1]);
        let kind = projection_constraint(name, atoms, None).ok()?.kind;
        let target = resolved.iter().find(|c| c.kind == kind)?.target?;
        let value = match name.to_ascii_lowercase().as_str() {
            "bond" | "fbond" | "x" | "y" | "z" => target * Units::BOHR_TO_ANGSTROM,
            "angle" | "dihedral" => target * Units::RAD_TO_DEG,
            _ => return None,
        };
        // Groups of four, as `-pc manual` reads them.
        out.extend([
            "manual".to_string(),
            name.clone(),
            atoms.clone(),
            value.to_string(),
        ]);
    }
    Some(out)
}

/// A scanned coordinate of `-pc manual` (`manual NAME ATOMS VALUE`) as a `-uc` token with the
/// same value, for a run whose `-pc` cannot be written in the manual form; `None` for a kind
/// the expressions do not have (`fbond`).
fn as_user_constraint(manual: &[String]) -> Option<String> {
    let [_, name, atoms, value] = manual else {
        return None;
    };
    let ids: Vec<&str> = atoms.split(',').map(str::trim).collect();
    let list = ids.join(",");
    Some(match (name.to_ascii_lowercase().as_str(), ids.len()) {
        ("bond", 2) => format!("dist({list}) = {value}*ang"),
        ("angle", 3) => format!("angle({list}) = {value}*deg"),
        ("dihedral", 4) => format!("dihedral({list}) = {value}*deg; period = 2*pi"),
        (axis @ ("x" | "y" | "z"), 1) => format!("{axis}({list}) = {value}*ang"),
        _ => return None,
    })
}

/// The settings document `document` with `tokens` appended to their options, `NSTEP` set for a
/// single point. Where the user's `-pc` has the automatic form, it is written in the manual
/// form with the values the run holds (`resolved`, resolved at the input of the scan), so that
/// the scanned coordinates can follow as `-pc manual`; where it cannot be (`rot`, `eigvec`,
/// `atoms_pair`), the scanned coordinates are written as `-uc`, and those the expressions do
/// not have (`fbond`) under `scan_projection_constrain`.
fn point_document(
    document: &str,
    tokens: &[(&'static str, Vec<String>)],
    single: bool,
    resolved: &[Constraint],
) -> MoptResult<String> {
    let mut doc = JsonValue::parse(document)?;
    let Some(map) = doc.as_object_mut() else {
        return Ok(document.to_string());
    };
    let as_strings = |v: Option<&JsonValue>| -> Vec<JsonValue> {
        match v {
            Some(JsonValue::Array(items)) => items.clone(),
            None | Some(JsonValue::Null) => Vec::new(),
            Some(other) => vec![other.clone()],
        }
    };
    let append = |map: &mut std::collections::BTreeMap<String, JsonValue>,
                  key: &str,
                  values: Vec<String>| {
        let mut items = as_strings(map.get(key));
        items.extend(values.into_iter().map(JsonValue::String));
        map.insert(key.to_string(), JsonValue::Array(items));
    };
    for (key, values) in tokens {
        let current = as_strings(map.get(*key));
        let manual = current
            .first()
            .and_then(JsonValue::as_str)
            .is_none_or(|s| s.eq_ignore_ascii_case("manual"));
        if *key != "projection_constrain" || manual {
            append(map, key, values.clone());
            continue;
        }
        let automatic: Vec<String> = current
            .iter()
            .filter_map(|v| v.as_str().map(str::to_string))
            .collect();
        match manual_form(&automatic, resolved) {
            Some(written) => {
                map.insert(
                    (*key).to_string(),
                    JsonValue::Array(written.into_iter().map(JsonValue::String).collect()),
                );
                append(map, key, values.clone());
            }
            None => match as_user_constraint(values) {
                Some(uc) => append(map, "user_constraint", vec![uc]),
                None => append(map, "scan_projection_constrain", values.clone()),
            },
        }
    }
    if single {
        map.insert("NSTEP".into(), JsonValue::String("1".into()));
        for key in [
            "calc_exact_hess",
            "frequency_analysis",
            "force_frequency_analysis",
        ] {
            map.remove(key);
        }
    }
    Ok(doc.to_string_pretty())
}

impl MoleculePoints<'_> {
    /// The point in `dir` from `start` (whole structure), holding `scan` (relaxed scans) or
    /// with `start` as it is (single points). `stem` names its files.
    pub fn run(
        &self,
        start: &MolecularSystem,
        scan: &PointConstraints,
        dir: &Path,
        stem: &str,
    ) -> MoptResult<PointOutcome> {
        std::fs::create_dir_all(dir)?;
        let input = dir.join(format!("{stem}.xyz"));
        let mut first = start.clone();
        first.comment = format!("start of {stem}");
        write_xyz_file(&input, &first, None)?;
        let mut config = self.base.clone();
        config.inputs = vec![input.clone()];
        config.restart_dir = Some(dir.to_path_buf());
        config.stop_file = Some(self.stop_file.clone());
        let single = !self.relax;
        config.settings_document = point_document(
            &config.settings_document,
            &scan.tokens,
            single,
            &self.base.settings.constraints,
        )?;
        if single {
            config.settings.max_iterations = 1;
            config.settings.exact_hessian_interval = None;
            config.frequency_analysis = false;
            config.force_frequency_analysis = false;
        } else {
            config
                .settings
                .constraints
                .extend(scan.constraints.iter().cloned());
            let (tight, loose) = self.criteria_flags;
            config.settings.criteria = ConvergenceCriteria::from_flags(tight, loose, true);
            config.initial_hessian.clone_from(&scan.first_hessian);
            config.final_hessian.clone_from(&scan.keep_hessian);
            if let Some(h) = &scan.first_hessian {
                let mut doc = JsonValue::parse(&config.settings_document)?;
                if let Some(map) = doc.as_object_mut() {
                    map.insert(
                        "initial_hessian".into(),
                        JsonValue::String(h.to_string_lossy().into_owned()),
                    );
                }
                config.settings_document = doc.to_string_pretty();
            }
        }
        let runs = optmain::run_with(&config, dir, self.console, self.registry)?;
        let run = runs
            .into_iter()
            .next()
            .ok_or_else(|| MoptError::conv("the optimization of the point gave no result"))?;
        let summary = run
            .summary
            .ok_or_else(|| MoptError::conv("the optimization of the point gave no structure"))?;
        // The last evaluated structure (whole, with ONIOM microiterations), whose energy the
        // summary has.
        let structure = if run.optimized.is_file() {
            read_atoms(&run.optimized, start)?
        } else {
            summary.system.clone()
        };
        Ok(PointOutcome {
            energy: summary.energy.value(),
            biased: summary.biased_energy.value(),
            converged: run.converged || single,
            iterations: summary.iterations,
            interrupted: run.interrupted,
            structure: Structure::Atoms(structure),
        })
    }
}

/// What every point of a scan of plain variables shares.
pub struct VariablePoints<'a> {
    /// The optmain settings of every point.
    pub base: &'a OptmainConfig,
    /// The objective.
    pub objective: &'a dyn Objective,
    /// Names of the variables.
    pub names: &'a [String],
    /// Whether the files of the variables are JSON.
    pub json: bool,
    /// Optimize (false: one evaluation).
    pub relax: bool,
    /// The `STOP` file of the scan.
    pub stop_file: PathBuf,
}

impl VariablePoints<'_> {
    /// The point in `dir` from `start`, holding `constraints` (relaxed scans) or at `start`
    /// itself (single points).
    pub fn run(
        &self,
        start: &Col<f64>,
        constraints: VectorConstraintSet,
        dir: &Path,
        stem: &str,
    ) -> MoptResult<PointOutcome> {
        std::fs::create_dir_all(dir)?;
        let extension = if self.json { "json" } else { "txt" };
        let values = |x: &Col<f64>| -> Vec<f64> { x.iter().copied().collect() };
        if !self.relax {
            let v = self.objective.evaluate(start, false)?;
            std::fs::write(
                dir.join(format!("{stem}.{extension}")),
                format_variables(self.names, &values(start), self.json, v.value),
            )?;
            return Ok(PointOutcome {
                energy: v.value,
                biased: v.value,
                converged: true,
                iterations: 1,
                interrupted: false,
                structure: Structure::Variables(start.clone()),
            });
        }
        let settings = vector_settings(self.base, &values(start))?;
        let mut opt = VectorOptimizer::new(settings, start.clone())?;
        // A copy that shares the record of the restores (warnings of ReSPer coordinates).
        let held = constraints.clone();
        opt.set_constraints(constraints)?;
        let log = Logger::silent();
        log.add_file(dir.join("mopt.log"), Level::Debug)?;
        let mut stop = StopFile::new(dir, log.clone()).also_watching(Some(self.stop_file.clone()));
        let mut table = String::from("iteration,value,max_gradient,rms_gradient");
        for n in self.names {
            let _ = write!(table, ",{n}");
        }
        table.push('\n');
        let outcome = run_vector_optimization_until(
            &mut opt,
            self.objective,
            &mut |r| {
                let _ = write!(
                    table,
                    "{},{:.12e},{:.6e},{:.6e}",
                    r.iteration,
                    r.energy,
                    r.convergence.max_force.value(),
                    r.convergence.rms_force.value()
                );
                for v in r.x.iter() {
                    let _ = write!(table, ",{v:.12e}");
                }
                table.push('\n');
                log.info(&format!(
                    "# ITR. {}  value {:.12}  max gradient {:.3e}",
                    r.iteration,
                    r.energy,
                    r.convergence.max_force.value()
                ));
                Ok(())
            },
            &mut || stop.check(),
        )?;
        for line in held.lenient_notes() {
            log.warn(&line);
        }
        if let Some(summary) = held.lenient_summary() {
            log.warn(&summary);
        }
        std::fs::write(dir.join(format!("{stem}_traj.csv")), table)?;
        std::fs::write(
            dir.join(format!("{stem}_optimized.{extension}")),
            format_variables(self.names, &values(&outcome.x), self.json, outcome.value),
        )?;
        log.flush();
        Ok(PointOutcome {
            energy: outcome.value,
            biased: outcome.value,
            converged: outcome.termination == Some(Termination::Converged),
            iterations: outcome.iterations,
            interrupted: stop.stopped(),
            structure: Structure::Variables(outcome.x),
        })
    }
}

/// A structure like `template` from `path`: its own cell where the file has one (a cell
/// optimized with `-vc`), else that of `template`.
fn read_atoms(path: &Path, template: &MolecularSystem) -> MoptResult<MolecularSystem> {
    let mut s = read_xyz_file(path, template.spin_charge, None).context(path.display())?;
    s.spin_charge = template.spin_charge;
    if s.cell.is_none() {
        s.cell.clone_from(&template.cell);
    }
    if s.elements != template.elements {
        return Err(MoptError::config(format!(
            "{}: not a structure of the atoms of the scan",
            path.display()
        )));
    }
    Ok(s)
}

/// The structure of a point read back from its file in the directory of the scan.
pub fn read_structure(path: &Path, template: &Structure) -> MoptResult<Structure> {
    match template {
        Structure::Atoms(t) => Ok(Structure::Atoms(read_atoms(path, t)?)),
        Structure::Variables(_) => {
            let v = super::variables::read_variables_file(path)?;
            Ok(Structure::Variables(Col::from_fn(v.values.len(), |i| {
                v.values[i]
            })))
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    #[test]
    fn the_settings_of_a_point_say_what_it_holds() {
        let doc = r#"{"INPUT": ["a.xyz"], "projection_constrain": ["manual", "bond", "3,4", "1.2"], "calc_exact_hess": "5"}"#;
        let tokens = vec![
            (
                "projection_constrain",
                vec![
                    "manual".to_string(),
                    "bond".into(),
                    "1,2".into(),
                    "1.5".into(),
                ],
            ),
            ("user_constraint", vec!["dist(1,2) = 2".to_string()]),
        ];
        let out = JsonValue::parse(&point_document(doc, &tokens, true, &[]).unwrap()).unwrap();
        let pc = out
            .get("projection_constrain")
            .unwrap()
            .as_string_vec()
            .unwrap();
        assert_eq!(pc.len(), 8);
        assert_eq!(out.get("NSTEP").unwrap().as_str(), Some("1"));
        assert!(out.get("calc_exact_hess").is_none());
        assert_eq!(
            out.get("user_constraint").unwrap().as_string_vec().unwrap(),
            vec!["dist(1,2) = 2"]
        );
        // The user's -pc of the automatic form is written in the manual form with the value
        // resolved at the input, and the scanned coordinate follows it (before, the scanned
        // coordinate was left out of the settings of such a run).
        let resolved = [Constraint::with_target(
            crate::pes::constraint::ConstraintKind::Distance(
                vec![crate::domain::AtomId::new(2)],
                vec![crate::domain::AtomId::new(3)],
            ),
            1.25 * Units::ANGSTROM_TO_BOHR,
        )];
        let auto = r#"{"projection_constrain": ["bond", "3,4"]}"#;
        let out =
            JsonValue::parse(&point_document(auto, &tokens, false, &resolved).unwrap()).unwrap();
        let pc = out
            .get("projection_constrain")
            .unwrap()
            .as_string_vec()
            .unwrap();
        assert_eq!(pc.len(), 8, "{pc:?}");
        let held = super::super::constraints::parse_projection_constraints(&pc).unwrap();
        assert!((held[0].target.unwrap() - resolved[0].target.unwrap()).abs() < 1e-12);
        assert!((held[1].target.unwrap() - 1.5 * Units::ANGSTROM_TO_BOHR).abs() < 1e-12);
        // A -pc that has no value to write (rot): the scanned coordinate goes to -uc.
        let rot = r#"{"projection_constrain": ["rot", "1-3"]}"#;
        let out = JsonValue::parse(&point_document(rot, &tokens, false, &[]).unwrap()).unwrap();
        assert_eq!(
            out.get("projection_constrain")
                .unwrap()
                .as_string_vec()
                .unwrap(),
            vec!["rot", "1-3"]
        );
        let uc = out.get("user_constraint").unwrap().as_string_vec().unwrap();
        assert_eq!(uc, vec!["dist(1,2) = 1.5*ang", "dist(1,2) = 2"]);
        for u in &uc {
            super::super::constraints::user_constraint(u).unwrap();
        }
    }
}
