//! The constants of the bias options of a command line and the derivatives in them
//! (`optmain -bias_derivatives`; the numbers are those of [`crate::pes::bias_response`]).
//!
//! A constant is a number among the arguments of a bias option that the option still reads
//! when it is moved a little: `-kp 0.5 1.5 1,2` has the force constant 0.5 and the distance
//! 1.5; an argument with commas (the limits of a well, the two ends of a ramp) has a constant
//! per field. Atom numbers read only as whole numbers and are no constants, nor are words.
//! A derivative is per unit of the option as it is written (kJ/mol for the AFIR collision
//! energy, Angstrom for distances, degrees for angles, atomic units for spring constants;
//! MultiOptPy's units of each option). The numbers inside the expression of `-ubias` are not
//! found (the expression is one argument).
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::fmt::Write as _;
use std::path::Path;
use std::sync::Arc;

use faer::{Col, Mat};

use super::bias::BiasRegistry;
use crate::core_infra::log::Logger;
use crate::domain::{MolecularSystem, Units};
use crate::error::{MoptError, MoptResult};
use crate::pes::bias_response::{
    constant_derivative, default_step, stationary_response, ConstantDerivative, StationaryResponse,
};
use crate::pes::term::{PesTerm, TermContext};

/// A constant of a bias option.
#[derive(Debug, Clone, PartialEq)]
pub struct BiasConstant {
    /// The option (`-kp`).
    pub flag: String,
    /// Its argument (from 0).
    pub argument: usize,
    /// The field of the argument (from 0; arguments without commas have one).
    pub field: usize,
    /// The fields of the argument.
    pub fields: usize,
    /// Its value as it is written.
    pub value: f64,
}

impl BiasConstant {
    /// `-kp argument 2` or `-wp argument 2, field 3` (from 1).
    pub fn name(&self) -> String {
        if self.fields > 1 {
            format!(
                "{} argument {}, field {}",
                self.flag,
                self.argument + 1,
                self.field + 1
            )
        } else {
            format!("{} argument {}", self.flag, self.argument + 1)
        }
    }
}

/// The byte ranges of the comma-separated fields of `token`.
fn field_ranges(token: &str) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut start = 0;
    for (i, c) in token.char_indices() {
        if c == ',' {
            out.push((start, i));
            start = i + 1;
        }
    }
    out.push((start, token.len()));
    out
}

/// `tokens` with field `range` of argument `argument` set to `value`.
fn replaced(tokens: &[String], argument: usize, range: (usize, usize), value: f64) -> Vec<String> {
    let mut out = tokens.to_vec();
    let t = &tokens[argument];
    out[argument] = format!("{}{value}{}", &t[..range.0], &t[range.1..]);
    out
}

/// The constants of option `flag` with its arguments `tokens`.
pub fn bias_constants(registry: &BiasRegistry, flag: &str, tokens: &[String]) -> Vec<BiasConstant> {
    let mut out = Vec::new();
    for (argument, token) in tokens.iter().enumerate() {
        let ranges = field_ranges(token);
        for (field, &range) in ranges.iter().enumerate() {
            let Ok(value) = token[range.0..range.1].trim().parse::<f64>() else {
                continue;
            };
            if !value.is_finite() {
                continue;
            }
            let h = default_step(value);
            let reads = [value + h, value - h].iter().all(|&v| {
                registry
                    .parse_option(flag, &replaced(tokens, argument, range, v))
                    .is_ok()
            });
            if reads {
                out.push(BiasConstant {
                    flag: flag.to_string(),
                    argument,
                    field,
                    fields: ranges.len(),
                    value,
                });
            }
        }
    }
    out
}

/// The terms of option `flag` with `tokens` for `sys`.
pub fn option_terms(
    registry: &BiasRegistry,
    flag: &str,
    tokens: &[String],
    sys: &MolecularSystem,
) -> MoptResult<Vec<Arc<dyn PesTerm>>> {
    registry
        .parse_option(flag, tokens)?
        .iter()
        .map(|s| s.build(sys))
        .collect()
}

/// The derivatives in one constant, and the response of the stationary point where it was
/// computed.
#[derive(Debug, Clone)]
pub struct ConstantResult {
    /// The constant.
    pub constant: BiasConstant,
    /// The derivatives of the bias energy and its gradient.
    pub derivative: ConstantDerivative,
    /// How the stationary point moves (`None` when it was not computed).
    pub response: Option<StationaryResponse>,
}

/// The Hessian of the whole surface at a stationary point, the gradient there without the
/// bias, and the directions taken out (rigid motions, constraints).
pub struct StationaryData<'a> {
    /// Hessian of the whole surface (Hartree / Bohr^2).
    pub hessian: &'a Mat<f64>,
    /// Gradient of the surface without the bias.
    pub unbiased_gradient: &'a Col<f64>,
    /// Orthonormal directions taken out.
    pub removed: Option<&'a Mat<f64>>,
}

/// The derivatives in every constant of `options` (flags with their arguments) at `sys`, with
/// the response of the stationary point when `stationary` is given.
pub fn constant_results(
    registry: &BiasRegistry,
    options: &[(String, Vec<String>)],
    sys: &MolecularSystem,
    ctx: &TermContext<'_>,
    stationary: Option<&StationaryData<'_>>,
) -> MoptResult<Vec<ConstantResult>> {
    let mut out = Vec::new();
    for (flag, tokens) in options {
        for constant in bias_constants(registry, flag, tokens) {
            let range = field_ranges(&tokens[constant.argument])[constant.field];
            let build = |v: f64| {
                option_terms(
                    registry,
                    flag,
                    &replaced(tokens, constant.argument, range, v),
                    sys,
                )
            };
            let derivative = constant_derivative(
                &build,
                constant.value,
                default_step(constant.value),
                sys,
                ctx,
            )
            .map_err(|e| e.context(constant.name()))?;
            let response = match stationary {
                Some(s) => Some(stationary_response(
                    s.hessian,
                    &derivative,
                    s.unbiased_gradient,
                    s.removed,
                )?),
                None => None,
            };
            out.push(ConstantResult {
                constant,
                derivative,
                response,
            });
        }
    }
    Ok(out)
}

/// The largest displacement of an atom in `dx` (Bohr).
fn largest_shift(dx: &Col<f64>) -> f64 {
    (0..dx.nrows() / 3)
        .map(|i| (dx[3 * i].powi(2) + dx[3 * i + 1].powi(2) + dx[3 * i + 2].powi(2)).sqrt())
        .fold(0.0, f64::max)
}

/// Write `results` to `bias_derivatives.csv` in `dir` and to the log.
pub fn write_constant_results(
    results: &[ConstantResult],
    dir: &Path,
    log: &Logger,
) -> MoptResult<()> {
    if results.is_empty() {
        log.info("Bias derivatives: the bias options have no constants.");
        return Ok(());
    }
    let kcal = Units::HARTREE_TO_KCAL_MOL;
    let mut csv = String::from(
        "constant,option,argument,field,value,step,dE_bias_dp_hartree,dE_bias_dp_kcal_mol,\
         dE_unbiased_dp_hartree,dE_unbiased_dp_kcal_mol,largest_atom_shift_angstrom,\
         rigid_or_constraints_removed,smallest_curvature\n",
    );
    let mut text = String::from(
        "Derivatives in the constants of the bias options (per unit of the option as written; \
         the stationary point moves by dx/dp = -H^+ dg/dp):",
    );
    for r in results {
        let c = &r.constant;
        let d = &r.derivative;
        let (de0, shift, removed, curvature) = match &r.response {
            Some(s) => (
                Some(s.d_unbiased_energy),
                Some(largest_shift(&s.d_structure) * Units::BOHR_TO_ANGSTROM),
                Some(s.rigid_removed),
                Some(s.smallest_curvature),
            ),
            None => (None, None, None, None),
        };
        let opt = |v: Option<f64>| v.map_or_else(String::new, |v| format!("{v:.10e}"));
        let _ = writeln!(
            csv,
            "\"{}\",{},{},{},{},{},{:.10e},{:.10e},{},{},{},{},{}",
            c.name(),
            c.flag,
            c.argument + 1,
            c.field + 1,
            c.value,
            d.step,
            d.d_energy,
            d.d_energy * kcal,
            opt(de0),
            opt(de0.map(|v| v * kcal)),
            opt(shift),
            removed.map_or_else(String::new, |b| b.to_string()),
            opt(curvature)
        );
        let _ = write!(
            text,
            "\n  {} = {}: dE(bias)/dp {:+.6e} kcal/mol",
            c.name(),
            c.value,
            d.d_energy * kcal
        );
        if let (Some(de0), Some(shift)) = (de0, shift) {
            let _ = write!(
                text,
                "; the optimized structure: dE(without bias)/dp {:+.6e} kcal/mol, largest atom \
                 shift {shift:.4e} Angstrom per unit",
                de0 * kcal
            );
        }
    }
    log.info(&text);
    std::fs::write(dir.join("bias_derivatives.csv"), csv)
        .map_err(|e| MoptError::config(format!("cannot write bias_derivatives.csv: {e}")))?;
    Ok(())
}

/// The file of the derivatives of the bias energy at every iteration.
pub const ITERATIONS_FILE: &str = "bias_derivatives_iterations.csv";

/// `dE_bias/dp` of every constant at every iteration of an optimization, appended to
/// [`ITERATIONS_FILE`] (MultiOptPy writes `dE_bias_pot/d_param` of every bias at every
/// iteration).
pub struct IterationDerivatives {
    registry: BiasRegistry,
    options: Vec<(String, Vec<String>)>,
    constants: Vec<BiasConstant>,
    start: MolecularSystem,
    path: std::path::PathBuf,
}

impl IterationDerivatives {
    /// For the bias `options` of a run from `start` in the run directory `dir` (the file is
    /// written anew).
    pub fn new(
        options: &[(String, Vec<String>)],
        start: &MolecularSystem,
        dir: &Path,
    ) -> MoptResult<Self> {
        let registry = BiasRegistry::new();
        let constants: Vec<BiasConstant> = options
            .iter()
            .flat_map(|(flag, tokens)| bias_constants(&registry, flag, tokens))
            .collect();
        let path = dir.join(ITERATIONS_FILE);
        let mut head = String::from("iteration");
        for c in &constants {
            let _ = write!(head, ",\"dE/dp {} (Hartree per unit)\"", c.name());
        }
        head.push('\n');
        std::fs::write(&path, head)?;
        Ok(Self {
            registry,
            options: options.to_vec(),
            constants,
            start: start.clone(),
            path,
        })
    }
}

impl crate::workflow::Observer for IterationDerivatives {
    fn on_iteration(
        &mut self,
        report: &crate::opt::state_machine::IterationReport,
    ) -> MoptResult<()> {
        use std::io::Write as _;
        if self.constants.is_empty() {
            return Ok(());
        }
        let sys = &report.system;
        let ctx = TermContext::at_iteration(report.iteration, &self.start);
        let mut line = report.iteration.to_string();
        for c in &self.constants {
            let Some((_, tokens)) = self.options.iter().find(|(f, _)| *f == c.flag) else {
                continue;
            };
            let range = field_ranges(&tokens[c.argument])[c.field];
            let build = |v: f64| {
                option_terms(
                    &self.registry,
                    &c.flag,
                    &replaced(tokens, c.argument, range, v),
                    sys,
                )
            };
            let d = constant_derivative(&build, c.value, default_step(c.value), sys, &ctx)?;
            let _ = write!(line, ",{:.10e}", d.d_energy);
        }
        line.push('\n');
        std::fs::OpenOptions::new()
            .append(true)
            .open(&self.path)?
            .write_all(line.as_bytes())?;
        Ok(())
    }
}

/// Whether the surface with the Hessian `hessian` does not depend on the rigid motions `r` of
/// the structure (their curvature vanishes against that of the vibrations).
fn rigid_free(hessian: &Mat<f64>, r: &Mat<f64>) -> bool {
    let along = r.transpose() * hessian * r;
    let largest = (0..hessian.nrows())
        .flat_map(|i| (0..hessian.ncols()).map(move |j| (i, j)))
        .fold(0.0_f64, |m, (i, j)| m.max(hessian[(i, j)].abs()));
    let rigid = (0..along.nrows())
        .flat_map(|i| (0..along.ncols()).map(move |j| (i, j)))
        .fold(0.0_f64, |m, (i, j)| m.max(along[(i, j)].abs()));
    rigid <= 1e-5 * largest.max(f64::MIN_POSITIVE)
}

/// `optmain -bias_derivatives`: at the converged structure `sys` of `opt` (started from
/// `start`, `iterations` iterations), the Hessian of the whole surface, and the derivatives in
/// every constant of `options` with the response of the stationary point (constraints and,
/// where the surface does not depend on them, the rigid motions taken out), to
/// `bias_derivatives.csv` in `dir` and the log.
#[allow(clippy::too_many_arguments)]
pub fn write_bias_derivatives(
    opt: &mut crate::opt::state_machine::GeometryOptimizer,
    calc: &dyn crate::pes::calculator::Calculator,
    sys: &MolecularSystem,
    start: &MolecularSystem,
    iterations: usize,
    options: &[(String, Vec<String>)],
    constraints: &[crate::pes::constraint::Constraint],
    dir: &Path,
    log: &Logger,
) -> MoptResult<()> {
    if options.is_empty() {
        log.info("-bias_derivatives: no bias option was given.");
        return Ok(());
    }
    log.info("Derivatives in the constants of the bias options (Hessian of the whole surface)...");
    let registry = BiasRegistry::new();
    let ctx = TermContext::at_iteration(iterations, start);
    let tasks = opt.hessian_tasks(sys)?;
    let results = crate::workflow::driver::CalculatorSet::single(calc).evaluate(&tasks)?;
    let (_, gradient, mut hessian) = opt.surface_hessian(sys, results)?;
    // The gradient without the bias: that of the whole surface less that of the options.
    let mut terms = Vec::new();
    for (flag, tokens) in options {
        terms.extend(option_terms(&registry, flag, tokens, sys)?);
    }
    let (_, bias_gradient) = crate::pes::bias_response::terms_value(&terms, sys, &ctx)?;
    let unbiased = &gradient - &bias_gradient;
    let mut columns: Vec<Col<f64>> = Vec::new();
    let mut rigid_allowed = true;
    if !constraints.is_empty() {
        let set = crate::pes::constraint::ConstraintSet::new(constraints, start, None)?;
        let frame = set.frame(sys, &gradient)?;
        for k in 0..frame.jacobian.nrows() {
            columns.push(frame.jacobian.row(k).transpose().to_owned());
        }
        // The Hessian of the Lagrangian in the space the constraints leave.
        hessian = &hessian - &frame.curvature;
        rigid_allowed = !set.breaks_rigid_body();
    }
    if rigid_allowed && calc.is_rigid_body_invariant() && !sys.has_pbc() {
        let r = crate::math::projection::build_tr_rot_basis(sys, false);
        if rigid_free(&hessian, &r) {
            columns.extend((0..r.ncols()).map(|k| r.col(k).to_owned()));
        }
    }
    let removed = if columns.is_empty() {
        None
    } else {
        let m = Mat::from_fn(sys.n_atoms() * 3, columns.len(), |i, j| columns[j][i]);
        crate::pes::bias_response::orthonormal_columns(&m)
    };
    let data = StationaryData {
        hessian: &hessian,
        unbiased_gradient: &unbiased,
        removed: removed.as_ref(),
    };
    let results = constant_results(&registry, options, sys, &ctx, Some(&data))?;
    write_constant_results(&results, dir, log)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn tokens(t: &[&str]) -> Vec<String> {
        t.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn constants_are_the_numbers_an_option_still_reads_when_moved() {
        let r = BiasRegistry::new();
        // -kp k r0 atoms: the atoms are no constants.
        let c = bias_constants(&r, "-kp", &tokens(&["0.5", "1.5", "1,2"]));
        assert_eq!(
            c.iter().map(|c| (c.argument, c.value)).collect::<Vec<_>>(),
            vec![(0, 0.5), (1, 1.5)]
        );
        // -ma gamma atoms atoms.
        let c = bias_constants(&r, "-ma", &tokens(&["100", "1", "2"]));
        assert_eq!(c.len(), 1);
        assert_eq!((c[0].argument, c[0].value), (0, 100.0));
        assert_eq!(c[0].name(), "-ma argument 1");
        // A ramp has two fields.
        let c = bias_constants(&r, "-kp", &tokens(&["0.5,1.0", "1.5", "1,2"]));
        assert_eq!(c.len(), 3);
        assert_eq!(c[0].name(), "-kp argument 1, field 1");
    }

    #[test]
    fn a_moved_field_keeps_the_rest_of_its_argument() {
        let t = tokens(&["0.5,1.0", "1.5"]);
        let r = field_ranges(&t[0]);
        assert_eq!(r, vec![(0, 3), (4, 7)]);
        assert_eq!(replaced(&t, 0, r[1], 2.25), tokens(&["0.5,2.25", "1.5"]));
    }
}
