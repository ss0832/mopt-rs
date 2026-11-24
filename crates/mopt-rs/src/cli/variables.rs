//! `optmain` on plain variables: `-os mb2d`, `-os expr:...` (any objective of
//! [`ObjectiveRegistry`]) with a variables file as the input.
//!
//! The variables file is text (one variable per line, `value` or `name value`, `#` starts a
//! comment) or JSON (`[1.0, 2.0]` or `{"variables": [...], "names": [...]}`). The run directory
//! holds `mopt.log`, `input.txt`, `settings.json`, the trajectory `<stem>_traj.csv` (iteration,
//! value, convergence measures, variables, gradient), the plots `value_plot_.png` and
//! `plot_gradient_.png`, and the optimized variables in the format of the input. Options that
//! need atoms (bias potentials, constraints, coordinate systems, model Hessians, ...) are
//! refused. Numbers (thresholds, trust radii, `-ccqn_step`) are in the units of the variables
//! and of the objective.
//!
//! MultiOptPy: no counterpart (new in mopt-rs; MultiOptPy optimizes atomic coordinates only).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use faer::Col;

use super::optmain::{run_directory, OptmainConfig, OptmainResult};
use crate::core_infra::clock::DateTime;
use crate::core_infra::json::JsonValue;
use crate::core_infra::log::{Level, Logger};
use crate::core_infra::viz::figure::save_single_line_plot;
use crate::error::{MoptError, MoptResult, ResultExt};
use crate::opt::state_machine::Termination;
use crate::opt::step::ccqn::{CcqnAxis, CcqnSettings};
use crate::opt::vector::{VectorOptimizer, VectorReport, VectorSettings};
use crate::pes::objective::{check_dimension, ObjectiveRegistry};
use crate::workflow::observers::StopFile;
use crate::workflow::run_vector_optimization_until;

/// The contents of a variables file.
#[derive(Debug, Clone, PartialEq)]
pub struct Variables {
    /// Names (`None`: the file gives none; the objective's names are used).
    pub names: Option<Vec<String>>,
    /// Values.
    pub values: Vec<f64>,
    /// Whether the file is JSON (the optimized variables are written in the same format).
    pub json: bool,
}

fn number(token: &str, line: usize) -> MoptResult<f64> {
    let v: f64 = token.parse().map_err(|_| {
        MoptError::parse(format!(
            "variables file, line {line}: '{token}' is not a number"
        ))
    })?;
    if v.is_finite() {
        Ok(v)
    } else {
        Err(MoptError::parse(format!(
            "variables file, line {line}: '{token}' is not finite"
        )))
    }
}

fn parse_json(text: &str) -> MoptResult<Variables> {
    let doc = JsonValue::parse(text).context("variables file")?;
    let floats = |v: &JsonValue| -> MoptResult<Vec<f64>> {
        v.as_array()
            .ok_or_else(|| MoptError::parse("variables file: 'variables' must be a list"))?
            .iter()
            .map(|x| {
                x.as_f64()
                    .filter(|f| f.is_finite())
                    .ok_or_else(|| MoptError::parse("variables file: the values must be numbers"))
            })
            .collect()
    };
    if doc.as_array().is_some() {
        return Ok(Variables {
            names: None,
            values: floats(&doc)?,
            json: true,
        });
    }
    let values = floats(doc.get("variables").ok_or_else(|| {
        MoptError::parse("variables file: a JSON object needs the key 'variables'")
    })?)?;
    let names = match doc.get("names") {
        None => None,
        Some(n) => Some(
            n.as_string_vec()
                .ok_or_else(|| MoptError::parse("variables file: 'names' must be strings"))?,
        ),
    };
    Ok(Variables {
        names,
        values,
        json: true,
    })
}

/// Parse the text of a variables file.
pub fn parse_variables(text: &str) -> MoptResult<Variables> {
    let json = text.trim_start().starts_with(['{', '[']);
    let vars = if json {
        parse_json(text)?
    } else {
        let mut names = Vec::new();
        let mut values = Vec::new();
        for (k, raw) in text.lines().enumerate() {
            let line = raw.split('#').next().unwrap_or("").trim();
            let tokens: Vec<&str> = line.split_whitespace().collect();
            match tokens.as_slice() {
                [] => {}
                [v] => {
                    names.push(None);
                    values.push(number(v, k + 1)?);
                }
                [name, v] => {
                    names.push(Some((*name).to_string()));
                    values.push(number(v, k + 1)?);
                }
                _ => {
                    return Err(MoptError::parse(format!(
                        "variables file, line {}: expected `value` or `name value`, got '{line}'",
                        k + 1
                    )))
                }
            }
        }
        let named = names.iter().filter(|n| n.is_some()).count();
        let names = match named {
            0 => None,
            n if n == values.len() => Some(names.into_iter().flatten().collect()),
            _ => {
                return Err(MoptError::parse(
                    "variables file: name every variable or none of them",
                ))
            }
        };
        Variables {
            names,
            values,
            json: false,
        }
    };
    if vars.values.is_empty() {
        return Err(MoptError::parse("variables file: no variables"));
    }
    if let Some(names) = &vars.names {
        if names.len() != vars.values.len() {
            return Err(MoptError::parse(format!(
                "variables file: {} names for {} variables",
                names.len(),
                vars.values.len()
            )));
        }
        let mut seen = std::collections::BTreeSet::new();
        for n in names {
            if !seen.insert(n.as_str()) {
                return Err(MoptError::parse(format!(
                    "variables file: the name '{n}' is given twice"
                )));
            }
        }
    }
    Ok(vars)
}

/// Read a variables file.
pub fn read_variables_file(path: &Path) -> MoptResult<Variables> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| MoptError::parse(format!("cannot read {}: {e}", path.display())))?;
    parse_variables(&text).context(path.display())
}

/// A variables file with `names` and `values` (JSON when `json`); `value` is recorded as a
/// comment (text) or under the key `value` (JSON).
pub fn format_variables(names: &[String], values: &[f64], json: bool, value: f64) -> String {
    if json {
        let mut o = BTreeMap::new();
        o.insert(
            "names".to_string(),
            JsonValue::Array(names.iter().cloned().map(JsonValue::String).collect()),
        );
        o.insert(
            "variables".to_string(),
            JsonValue::Array(values.iter().copied().map(JsonValue::Number).collect()),
        );
        o.insert("value".to_string(), JsonValue::Number(value));
        return format!("{}\n", JsonValue::Object(o).to_string_pretty());
    }
    let width = names.iter().map(String::len).max().unwrap_or(0);
    let mut s = format!("# value = {value:.12e}\n");
    for (n, v) in names.iter().zip(values) {
        let _ = writeln!(s, "{n:<width$}  {v:.12e}");
    }
    s
}

fn refuse(what: &str) -> MoptError {
    MoptError::parse(format!(
        "{what} needs atoms; it cannot be used with an objective of plain variables"
    ))
}

/// The optimizer settings of a run on plain variables (`start` is the starting point);
/// options that need atoms are refused.
pub fn vector_settings(config: &OptmainConfig, start: &[f64]) -> MoptResult<VectorSettings> {
    let s = &config.settings;
    if !config.biases.is_empty() {
        return Err(refuse("a bias potential (-ma, -kp, ...)"));
    }
    if !config.model_function.is_empty() {
        return Err(refuse("-mf"));
    }
    if !s.constraints.is_empty() {
        return Err(refuse("a constraint (-fix, -pc, ...)"));
    }
    if !config.shape_conditions.is_empty() {
        return Err(refuse("-sc"));
    }
    if !config.ccqn_bonds.is_empty() {
        return Err(MoptError::parse(
            "-ccqn_bonds needs atoms; give -ccqn_product VARIABLES_FILE (the axis points from \
             the start to that point)",
        ));
    }
    if s.model_hessian.is_some() {
        return Err(refuse("-modelhess"));
    }
    if !matches!(
        s.coordinates.to_ascii_lowercase().as_str(),
        "cart" | "cartesian"
    ) {
        return Err(refuse("-coord"));
    }
    if config.state.charge.is_some() || config.state.multiplicity.is_some() {
        return Err(refuse("-elec / -spin"));
    }
    let ccqn = match (&s.ccqn, &config.ccqn_product) {
        (None, _) => None,
        (Some(_), None) => {
            return Err(MoptError::parse(
                "CCQN on plain variables needs -ccqn_product VARIABLES_FILE",
            ))
        }
        (Some(given), Some(product)) => {
            let p = read_variables_file(product)?;
            if p.values.len() != start.len() {
                return Err(MoptError::parse(format!(
                    "-ccqn_product has {} variables, the input {}",
                    p.values.len(),
                    start.len()
                )));
            }
            let axis = Col::from_fn(start.len(), |i| p.values[i] - start[i]);
            if axis.iter().all(|v| *v == 0.0) {
                return Err(MoptError::parse(
                    "-ccqn_product is the starting point itself",
                ));
            }
            let mut c = CcqnSettings::new(CcqnAxis::Direction(axis));
            c.cos_phi = given.cos_phi;
            c.cone_solver = given.cone_solver;
            if let Some(step) = config.ccqn_step {
                c.uphill_radius = step;
                c.trust.initial = step;
                c.trust.min = c.trust.min.min(step);
                c.trust.max = c.trust.max.max(step);
            }
            Some(c)
        }
    };
    Ok(VectorSettings {
        methods: s.methods.clone(),
        saddle_order: s.saddle_order,
        max_iterations: s.max_iterations,
        criteria: s.criteria,
        trust: s.trust,
        exact_hessian_interval: s.exact_hessian_interval,
        ccqn,
        mode_kill: s.mode_kill,
    })
}

/// Files of a run on plain variables.
struct Trajectory {
    csv: String,
    values: Vec<f64>,
    rms_gradients: Vec<f64>,
}

impl Trajectory {
    fn new(names: &[String]) -> Self {
        let mut csv = String::from(
            "iteration,value,max_gradient,rms_gradient,max_step,rms_step,trust_radius",
        );
        for n in names {
            let _ = write!(csv, ",{n}");
        }
        for n in names {
            let _ = write!(csv, ",d/d{n}");
        }
        csv.push('\n');
        Self {
            csv,
            values: Vec::new(),
            rms_gradients: Vec::new(),
        }
    }

    fn push(&mut self, r: &VectorReport) {
        let c = &r.convergence;
        let _ = write!(
            self.csv,
            "{},{:.12e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e}",
            r.iteration,
            r.energy,
            c.max_force.value(),
            c.rms_force.value(),
            c.max_displacement.value(),
            c.rms_displacement.value(),
            r.trust_radius
        );
        for v in r.x.iter() {
            let _ = write!(self.csv, ",{v:.12e}");
        }
        for g in r.gradient.iter() {
            let _ = write!(self.csv, ",{g:.6e}");
        }
        self.csv.push('\n');
        self.values.push(r.energy);
        self.rms_gradients.push(c.rms_force.value());
    }

    fn save(&self, dir: &Path, csv: &Path) -> MoptResult<()> {
        std::fs::write(csv, &self.csv)?;
        let xs: Vec<f64> = (0..self.values.len()).map(|i| i as f64).collect();
        save_single_line_plot(
            dir.join("value_plot_.png"),
            "objective",
            "ITR. ",
            "value",
            &xs,
            &self.values,
        )?;
        save_single_line_plot(
            dir.join("plot_gradient_.png"),
            "gradient",
            "ITR. ",
            "gradient (RMS)",
            &xs,
            &self.rms_gradients,
        )
    }
}

fn iteration_text(
    r: &VectorReport,
    names: &[String],
    settings: &VectorSettings,
    previous: Option<f64>,
) -> String {
    let c = &r.convergence;
    let mut s = String::new();
    let _ = writeln!(s, "# ITR. {}", r.iteration);
    for note in &r.notes {
        let _ = writeln!(s, "{note}");
    }
    if names.len() <= 12 {
        let list: Vec<String> = names
            .iter()
            .zip(r.x.iter())
            .map(|(n, v)| format!("{n} = {v:.10}"))
            .collect();
        let _ = writeln!(s, "variables              : {}", list.join(", "));
    }
    let _ = writeln!(
        s,
        "                         Value                     Threshold "
    );
    let _ = writeln!(s, "VALUE                  : {:>15.12} ", r.energy);
    let row = |name: &str, v: f64, t: f64| {
        format!("{name:<23}: {v:>15.12}                 {t:>15.12} \n")
    };
    let k = &settings.criteria;
    s.push_str(&row(
        "Maximum  Gradient",
        c.max_force.value(),
        k.max_force.value(),
    ));
    s.push_str(&row(
        "RMS      Gradient",
        c.rms_force.value(),
        k.rms_force.value(),
    ));
    s.push_str(&row(
        "Maximum  Step",
        c.max_displacement.value(),
        c.max_displacement_threshold.value(),
    ));
    s.push_str(&row(
        "RMS      Step",
        c.rms_displacement.value(),
        c.rms_displacement_threshold.value(),
    ));
    let _ = write!(
        s,
        "VALUE SHIFT            : {:>15.12} ",
        r.energy - previous.unwrap_or(r.energy)
    );
    s
}

fn optimized_path(dir: &Path, input: &Path, stem: &str) -> PathBuf {
    let ext = input
        .extension()
        .map_or_else(|| "txt".to_string(), |e| e.to_string_lossy().into_owned());
    dir.join(format!("{stem}_optimized.{ext}"))
}

/// Run `optmain` for one variables file with the objective named by `config.calculator`.
pub fn run_one(
    config: &OptmainConfig,
    input: &Path,
    base: &Path,
    console: Option<Level>,
    objectives: &ObjectiveRegistry,
) -> MoptResult<OptmainResult> {
    let evaluations_before = objectives.counts();
    let objective = objectives.create(&config.calculator, &config.calculator_options)?;
    let vars = read_variables_file(input)?;
    let n = vars.values.len();
    check_dimension(objective.as_ref(), n)?;
    let names = vars
        .names
        .clone()
        .unwrap_or_else(|| objective.variable_names(n));
    let settings = vector_settings(config, &vars.values)?;

    let dir = match &config.restart_dir {
        Some(d) => d.clone(),
        None => run_directory(base, input, &DateTime::now_utc()),
    };
    std::fs::create_dir_all(&dir)?;
    let log = console.map_or_else(Logger::silent, Logger::console);
    log.add_file(dir.join("mopt.log"), Level::Debug)?;
    log.info(&format!(
        "mopt-rs {} optmain: {}",
        crate::VERSION,
        input.display()
    ));
    log.info(&format!(
        "Mode: optimization of {n} plain variables ({})",
        names.join(", ")
    ));
    log.info(&format!("Objective: {}", objective.description()));
    if config.restart_dir.is_some() {
        log.info("Restart: the run is repeated from the starting point (objectives are cheap).");
    }
    std::fs::write(dir.join("input.txt"), format!("{config:#?}\n"))?;
    std::fs::write(
        dir.join("settings.json"),
        format!("{}\n", config.settings_document),
    )?;

    let stem = input
        .file_stem()
        .map_or_else(|| "input".to_string(), |s| s.to_string_lossy().into_owned());
    let csv = dir.join(format!("{stem}_traj.csv"));
    let optimized = optimized_path(&dir, input, &stem);
    let start = Col::from_fn(n, |i| vars.values[i]);
    let mut opt = VectorOptimizer::new(settings.clone(), start)?;
    opt.configure_mode_kill(config.hisd_mode_kill, None);
    let mut trajectory = Trajectory::new(&names);
    let mut stop = StopFile::new(&dir, log.clone());
    let mut previous = None;
    let outcome = run_vector_optimization_until(
        &mut opt,
        objective.as_ref(),
        &mut |r| {
            log.info(&iteration_text(r, &names, &settings, previous));
            previous = Some(r.energy);
            trajectory.push(r);
            Ok(())
        },
        &mut || stop.check(),
    )?;
    log.info(match outcome.termination {
        Some(Termination::Converged) => {
            "\n=====================================================\nconverged!!!\n====================================================="
        }
        Some(Termination::MaxIterations | Termination::SeamCheckFailed) => {
            "Reached maximum number of iterations. This is not converged."
        }
        Some(Termination::Dissociated) | None => {
            "The optimization was stopped by a stop condition."
        }
    });
    trajectory.save(&dir, &csv)?;
    let x: Vec<f64> = outcome.x.iter().copied().collect();
    std::fs::write(
        &optimized,
        format_variables(&names, &x, vars.json, outcome.value),
    )?;
    if stop.stopped() {
        log.info(&format!(
            "Continue with: optmain --restart {}",
            dir.display()
        ));
    }
    // `-irc`: the path from the optimized point.
    let mut interrupted = stop.stopped();
    if let (Some(irc), false) = (&config.irc, stop.stopped()) {
        // A point the optimization converged to is a stationary point for the IRC.
        let irc = if outcome.termination == Some(Termination::Converged) {
            irc.clone().accepting_converged(
                settings.criteria.max_force.value(),
                settings.criteria.rms_force.value(),
            )
        } else {
            irc.clone()
        };
        let run = super::irc::run_variables(
            &irc,
            objective.as_ref(),
            &names,
            &x,
            &stem,
            &dir,
            &log,
            &mut [],
        )?;
        interrupted = run.interrupted;
    }
    log.info(&format!(
        "Optimized variables file path: {}",
        optimized.display()
    ));
    log.info(&format!("Trajectory file path: {}", csv.display()));
    log.info(
        &objectives
            .counts()
            .since(&evaluations_before)
            .line("this input"),
    );
    log.flush();
    Ok(OptmainResult {
        directory: dir,
        optimized,
        converged: outcome.termination == Some(Termination::Converged),
        interrupted,
        summary: None,
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn variables_files_in_text_and_json() {
        let v = parse_variables("# start\n-0.4\n1.3  # second\n\n").unwrap();
        assert_eq!((v.names, v.values, v.json), (None, vec![-0.4, 1.3], false));
        let v = parse_variables("x -0.4\ny 1.3e0\n").unwrap();
        assert_eq!(v.names, Some(vec!["x".to_string(), "y".to_string()]));
        let v = parse_variables("[1, 2.5, -3]").unwrap();
        assert_eq!((v.values, v.json), (vec![1.0, 2.5, -3.0], true));
        let v = parse_variables(r#"{"variables": [1, 2], "names": ["a", "b"]}"#).unwrap();
        assert_eq!(v.names, Some(vec!["a".to_string(), "b".to_string()]));
        for bad in [
            "",
            "x 1\n2\n",
            "x 1 2\n",
            "abc\n",
            "x 1\nx 2\n",
            "nan\n",
            r#"{"names": ["a"]}"#,
            r#"{"variables": [1, 2], "names": ["a"]}"#,
            r#"{"variables": [1, "b"]}"#,
        ] {
            assert!(parse_variables(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn written_variables_are_read_back() {
        let names = vec!["x".to_string(), "long_name".to_string()];
        let values = [-0.558_223_634_633, 1.441_725_841_804];
        for json in [false, true] {
            let text = format_variables(&names, &values, json, -0.1467);
            let v = parse_variables(&text).unwrap();
            assert_eq!(v.names.as_deref(), Some(names.as_slice()));
            assert_eq!(v.json, json);
            for (a, b) in v.values.iter().zip(values) {
                assert!((a - b).abs() < 1e-12);
            }
        }
    }
}
