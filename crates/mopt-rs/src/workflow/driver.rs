//! Drive an ask/tell optimizer with Rust calculators and report progress to observers.
//!
//! MultiOptPy: the iteration loop of `optimization.py:Optimize._run_single_job`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use rayon::prelude::*;

use faer::{Col, Mat};

use crate::domain::{EvalOutput, Hartree, MolecularSystem, Quantity};
use crate::error::{MoptError, MoptResult};
use crate::opt::state_machine::{Action, GeometryOptimizer, IterationReport, Termination};
use crate::opt::vector::{VectorAction, VectorOptimizer, VectorReport};
use crate::pes::calculator::{evaluate_checked, Calculator};
use crate::pes::combine::{EvalTask, DEFAULT_CALCULATOR};
use crate::pes::objective::{check_dimension, Objective};

/// Receives optimizer progress (logging, files, plots). Observers perform all I/O.
pub trait Observer {
    /// Called after every iteration.
    fn on_iteration(&mut self, report: &IterationReport) -> MoptResult<()>;

    /// Report an iteration together with its applied quasi-Newton update, when present.
    fn on_iteration_with_hessian(
        &mut self,
        report: &IterationReport,
        _update: Option<(&str, &Mat<f64>)>,
    ) -> MoptResult<()> {
        self.on_iteration(report)
    }

    /// Called with each warning of a calculator (after the evaluations that raised it).
    fn on_warning(&mut self, message: &str) -> MoptResult<()> {
        let _ = message;
        Ok(())
    }

    /// Called once when the optimization ends.
    fn on_finish(&mut self, outcome: &Outcome) -> MoptResult<()> {
        let _ = outcome;
        Ok(())
    }

    /// Return `true` to stop early (e.g. an `end.txt` file or a shape condition).
    fn should_stop(&mut self, current: &MolecularSystem) -> bool {
        let _ = current;
        false
    }
}

/// Final result of a driven optimization.
#[derive(Debug, Clone)]
pub struct Outcome {
    /// Termination reason, or `None` if an observer stopped the run.
    pub termination: Option<Termination>,
    /// Final structure (the last evaluated one).
    pub system: MolecularSystem,
    /// Energy of the final structure without in-process terms (bias potentials).
    pub energy: Quantity<Hartree>,
    /// Energy of the full surface at the final structure.
    pub biased_energy: Quantity<Hartree>,
    /// Number of optimizer iterations.
    pub iterations: usize,
}

/// Calculators by label, for surfaces whose evaluations use different calculators.
#[derive(Clone, Default)]
pub struct CalculatorSet<'a> {
    entries: Vec<(String, &'a dyn Calculator)>,
}

impl<'a> CalculatorSet<'a> {
    /// `calc` under the default label.
    pub fn single(calc: &'a dyn Calculator) -> Self {
        Self::default().with(DEFAULT_CALCULATOR, calc)
    }

    /// Add (or replace) the calculator for `label`.
    #[must_use]
    pub fn with(mut self, label: &str, calc: &'a dyn Calculator) -> Self {
        self.entries.retain(|(l, _)| l != label);
        self.entries.push((label.to_string(), calc));
        self
    }

    /// Calculator for `label`.
    pub fn get(&self, label: &str) -> MoptResult<&'a dyn Calculator> {
        self.entries
            .iter()
            .find(|(l, _)| l == label)
            .map(|(_, c)| *c)
            .ok_or_else(|| MoptError::config(format!("no calculator registered as '{label}'")))
    }

    /// Warnings of all calculators (each calculator once).
    pub fn take_warnings(&self) -> Vec<String> {
        let mut seen: Vec<*const ()> = Vec::new();
        let mut out = Vec::new();
        for (_, c) in &self.entries {
            let p = (*c as *const dyn Calculator).cast::<()>();
            if seen.contains(&p) {
                continue;
            }
            seen.push(p);
            out.extend(c.take_warnings());
        }
        out
    }

    /// Perform `tasks` (in parallel when there are several, one after the other in their
    /// order where a calculator asks for that) and return the results in order.
    pub fn evaluate(&self, tasks: &[EvalTask]) -> MoptResult<Vec<EvalOutput>> {
        let run = |t: &EvalTask| -> MoptResult<EvalOutput> {
            evaluate_checked(self.get(&t.calculator)?, &t.system, &t.request)
                .map_err(|e| e.context(format!("task '{}'", t.label)))
        };
        let in_order = self.entries.iter().any(|(_, c)| c.evaluates_in_order());
        if tasks.len() > 1 && !in_order {
            tasks.par_iter().map(run).collect()
        } else {
            tasks.iter().map(run).collect()
        }
    }
}

/// Run `opt` to termination with a single calculator.
pub fn run_optimization(
    opt: &mut GeometryOptimizer,
    calc: &dyn Calculator,
    observers: &mut [&mut dyn Observer],
) -> MoptResult<Outcome> {
    run_optimization_with(opt, &CalculatorSet::single(calc), observers)
}

/// Run `opt` to termination, taking calculators from `calcs` by label.
pub fn run_optimization_with(
    opt: &mut GeometryOptimizer,
    calcs: &CalculatorSet<'_>,
    observers: &mut [&mut dyn Observer],
) -> MoptResult<Outcome> {
    let mut last: Option<IterationReport> = None;
    let mut stopped = false;
    loop {
        if observers.iter_mut().any(|o| o.should_stop(opt.current())) {
            stopped = true;
            break;
        }
        match opt.ask() {
            Action::Finished(_) => break,
            Action::Evaluate(tasks) => {
                let results = calcs.evaluate(&tasks)?;
                for w in calcs.take_warnings() {
                    for o in observers.iter_mut() {
                        o.on_warning(&w)?;
                    }
                }
                let report = opt.tell(results)?;
                for o in observers.iter_mut() {
                    o.on_iteration_with_hessian(&report, opt.updated_hessian())?;
                }
                last = Some(report);
            }
        }
    }
    // The result is the last evaluated structure with its energies, converged or not: after
    // an iteration that did not converge the optimizer holds the step it proposes next, a
    // structure that was never evaluated, whose energy the report does not have.
    let (energy, biased_energy, system) = match &last {
        Some(r) => (r.energy, r.biased_energy, r.system.clone()),
        None => (
            Quantity::new(f64::NAN),
            Quantity::new(f64::NAN),
            opt.current().clone(),
        ),
    };
    let outcome = Outcome {
        termination: if stopped { None } else { opt.termination() },
        system,
        energy,
        biased_energy,
        iterations: opt.iterations(),
    };
    for o in observers.iter_mut() {
        o.on_finish(&outcome)?;
    }
    Ok(outcome)
}

/// Final result of a driven vector optimization.
#[derive(Debug, Clone)]
pub struct VectorOutcome {
    /// Termination reason.
    pub termination: Option<Termination>,
    /// Final point (the last evaluated one).
    pub x: Col<f64>,
    /// Objective value at `x`.
    pub value: f64,
    /// Gradient at `x`.
    pub gradient: Col<f64>,
    /// Number of iterations.
    pub iterations: usize,
}

/// Run `opt` on `objective` to termination; `on_iteration` sees every report (logging, files).
pub fn run_vector_optimization(
    opt: &mut VectorOptimizer,
    objective: &dyn Objective,
    on_iteration: &mut dyn FnMut(&VectorReport) -> MoptResult<()>,
) -> MoptResult<VectorOutcome> {
    run_vector_optimization_until(opt, objective, on_iteration, &mut || false)
}

/// [`run_vector_optimization`] that ends early when `should_stop` (asked after every
/// iteration) returns true; the outcome then has no termination reason.
pub fn run_vector_optimization_until(
    opt: &mut VectorOptimizer,
    objective: &dyn Objective,
    on_iteration: &mut dyn FnMut(&VectorReport) -> MoptResult<()>,
    should_stop: &mut dyn FnMut() -> bool,
) -> MoptResult<VectorOutcome> {
    check_dimension(objective, opt.dimension())?;
    let mut last: Option<VectorReport> = None;
    while let VectorAction::Evaluate { x, need_hessian } = opt.ask() {
        let v = objective
            .evaluate(&x, need_hessian)
            .map_err(|e| e.context(format!("objective '{}'", objective.name())))?;
        let report = opt.tell(v.value, v.gradient, v.hessian)?;
        on_iteration(&report)?;
        last = Some(report);
        if opt.termination().is_none() && should_stop() {
            break;
        }
    }
    let r = last.ok_or_else(|| MoptError::conv("no iteration was performed"))?;
    Ok(VectorOutcome {
        termination: opt.termination(),
        x: r.x,
        value: r.energy,
        gradient: r.gradient,
        iterations: opt.iterations(),
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use std::sync::Mutex;

    use super::*;
    use crate::domain::{Element, EvalRequest, SpinCharge};

    /// Records the order of its evaluations; asks for them in their order or not.
    struct Recording {
        in_order: bool,
        seen: Mutex<Vec<f64>>,
    }

    impl Calculator for Recording {
        fn name(&self) -> &str {
            "recording"
        }

        fn evaluate(&self, sys: &MolecularSystem, _: &EvalRequest) -> MoptResult<EvalOutput> {
            let d = sys.coords_flat_bohr()[3];
            self.seen.lock().unwrap().push(d);
            Ok(EvalOutput::energy_gradient(Quantity::new(d), Col::zeros(6)))
        }

        fn evaluates_in_order(&self) -> bool {
            self.in_order
        }
    }

    #[test]
    fn a_calculator_that_asks_for_it_gets_its_batch_in_order() {
        let h = Element::parse_token("H").unwrap();
        let tasks: Vec<EvalTask> = (0..24)
            .map(|k| {
                let flat = Col::from_fn(6, |i| if i == 3 { 1.0 + k as f64 } else { 0.0 });
                EvalTask {
                    label: format!("image {k}"),
                    calculator: DEFAULT_CALCULATOR.to_string(),
                    system: MolecularSystem::from_flat_bohr(
                        vec![h, h],
                        &flat,
                        SpinCharge::default(),
                        None,
                    )
                    .unwrap(),
                    request: EvalRequest::ENERGY_ONLY,
                }
            })
            .collect();
        let expected: Vec<f64> = (0..24).map(|k| 1.0 + k as f64).collect();
        for in_order in [true, false] {
            let calc = Recording {
                in_order,
                seen: Mutex::new(Vec::new()),
            };
            let out = CalculatorSet::single(&calc).evaluate(&tasks).unwrap();
            // The results are those of the tasks in either case.
            let energies: Vec<f64> = out.iter().map(|o| o.energy.value()).collect();
            assert_eq!(energies, expected);
            let mut seen = calc.seen.lock().unwrap().clone();
            if in_order {
                assert_eq!(seen, expected, "one after the other");
            }
            seen.sort_by(f64::total_cmp);
            assert_eq!(seen, expected);
        }
    }
}
