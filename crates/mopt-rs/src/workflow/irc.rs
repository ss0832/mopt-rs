//! Driver of the IRC state machine: evaluates the points it asks for on a surface (a
//! calculator with bias potentials, a surface that combines electronic states as `-mf`
//! does, or an objective of plain variables) and hands what happens to an observer.
//!
//! MultiOptPy: `irc.py:IRC` (the calls of `QM_interface.single_point` and
//! `BiasPotentialCalculation.main` in `saddle_check` and in the integrators of `IRC/`).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::sync::{Arc, Mutex};

use faer::Mat;

use super::driver::CalculatorSet;
use crate::core_infra::log::Logger;
use crate::domain::{EvalRequest, MolecularSystem};
use crate::error::{MoptError, MoptResult, ResultExt};
use crate::irc::{Irc, IrcAction, IrcEnd, IrcEvent, IrcRequest, PointValue};
use crate::pes::calculator::{evaluate_checked, said, Calculator};
use crate::pes::combine::PesExpr;
use crate::pes::objective::Objective;
use crate::pes::term::{check_term_output, PesTerm, TermContext};

/// A surface on which a path is followed.
pub trait IrcSurface {
    /// The surface at the point of `request`.
    fn evaluate(&self, request: &IrcRequest) -> MoptResult<PointValue>;
}

/// The surface of a calculator with bias potentials for the atoms of a structure.
pub struct MoleculeSurface<'a> {
    /// The calculator.
    pub calc: &'a dyn Calculator,
    /// Bias potentials with their labels.
    pub terms: &'a [(String, Arc<dyn PesTerm>)],
    /// The starting structure (elements, charge, multiplicity, cell).
    pub start: &'a MolecularSystem,
    /// Electronic state (0: the ground state).
    pub excited_state: usize,
    /// Where the warnings of the calculator, `<S**2>`, and the output of the program go
    /// after every evaluation; `None`: nowhere.
    pub log: Option<&'a Logger>,
    /// The states combined into the surface that is followed (`-mf`); `None`: the state
    /// of `excited_state`.
    pub states: Option<Mutex<PesExpr>>,
    /// The iteration the in-process terms (and the terms of `states`) see at every point of
    /// the path: the surface of the iteration of the optimization that found the start, so
    /// that a ramped bias does not start over (0 for a path from a given structure).
    pub iteration: usize,
}

impl MoleculeSurface<'_> {
    /// The calculator's output at `sys`: of the state, or of the states combined.
    fn output(
        &self,
        sys: &MolecularSystem,
        base: &EvalRequest,
        of: &str,
        step: usize,
    ) -> MoptResult<crate::domain::EvalOutput> {
        let Some(states) = &self.states else {
            let eval = EvalRequest {
                excited_state: self.excited_state,
                ..*base
            };
            let out = evaluate_checked(self.calc, sys, &eval)?;
            if let Some(log) = self.log {
                for line in said(self.calc, of, &out) {
                    log.info(&line);
                }
            }
            return Ok(out);
        };
        let mut states = states
            .lock()
            .map_err(|_| MoptError::calc("the surface of the states is not available"))?;
        // The states are those of the surface; the tasks carry them.
        let tasks = states.tasks(sys, base)?;
        let results = CalculatorSet::single(self.calc).evaluate(&tasks)?;
        if let Some(log) = self.log {
            for (task, out) in tasks.iter().zip(&results) {
                for line in said(self.calc, &format!("{of} {}", task.label), out) {
                    log.info(&line);
                }
            }
        }
        let ctx = TermContext::new(step, self.start, None);
        Ok(states.evaluate(sys, base, results, &ctx)?.total)
    }
}

impl IrcSurface for MoleculeSurface<'_> {
    fn evaluate(&self, request: &IrcRequest) -> MoptResult<PointValue> {
        let point = &request.point;
        let sys = self.start.with_flat_coords_bohr(&point.x)?;
        // The Hessian at the start decides whether it is a transition state: a check of
        // stability.
        let base = match (point.need_hessian, point.role) {
            (true, crate::irc::PointRole::TransitionState) => EvalRequest::ENERGY_GRAD_HESS_CHECK,
            (true, _) => EvalRequest::ENERGY_GRAD_HESS,
            (false, _) => EvalRequest::ENERGY_GRAD,
        };
        let of = format!("IRC point {} ({:?})", point.step, point.role);
        let mut out = self.output(&sys, &base, &of, self.iteration)?;
        if base.for_stability && out.hessian_bohr.is_none() {
            // The surface could not give the Hessian of the check to convergence: the start
            // goes on with the Hessian of the steps, and its order is not confirmed.
            if let Some(log) = self.log {
                log.warn(
                    "IRC: the Hessian at the start could not be had to convergence; the order \
                     of the transition state is not confirmed, and the direction is taken from \
                     the Hessian of the steps",
                );
            }
            out = self.output(&sys, &EvalRequest::ENERGY_GRAD_HESS, &of, self.iteration)?;
        }
        let hessian = if point.need_hessian {
            Some(out.hessian_bohr.clone().ok_or_else(|| {
                MoptError::calc(format!(
                    "calculator '{}' returned no Hessian",
                    self.calc.name()
                ))
            })?)
        } else {
            None
        };
        let mut energy = out.energy.value();
        let mut gradient = out.gradient_bohr.clone();
        let mut bias_hessian: Option<Mat<f64>> = None;
        if !self.terms.is_empty() {
            let n = gradient.nrows();
            let mut total = Mat::<f64>::zeros(n, n);
            let ctx = TermContext::new(self.iteration, self.start, None);
            for (label, term) in self.terms {
                let bias = term.evaluate(&sys, &ctx, true).context(label.clone())?;
                check_term_output(label, &bias)?;
                energy += bias.energy.value();
                gradient += &bias.gradient_bohr;
                let of_bias = bias.hessian_bohr.as_ref().ok_or_else(|| {
                    MoptError::calc(format!("bias potential '{label}' returned no Hessian"))
                })?;
                total += of_bias;
            }
            bias_hessian = Some(total);
        }
        Ok(PointValue {
            energy,
            gradient,
            raw_energy: out.energy.value(),
            raw_gradient: out.gradient_bohr,
            hessian,
            bias_hessian,
        })
    }
}

/// An objective of plain variables as a surface.
pub struct ObjectiveSurface<'a>(pub &'a dyn Objective);

impl IrcSurface for ObjectiveSurface<'_> {
    fn evaluate(&self, request: &IrcRequest) -> MoptResult<PointValue> {
        let value = self
            .0
            .evaluate(&request.point.x, request.point.need_hessian)?;
        if request.point.need_hessian && value.hessian.is_none() {
            return Err(MoptError::calc(format!(
                "objective '{}' returned no Hessian",
                self.0.name()
            )));
        }
        Ok(PointValue::plain(
            value.value,
            value.gradient,
            value.hessian,
        ))
    }
}

/// Sees what happens in an IRC calculation, like [`super::Observer`] for optimizations:
/// the files and the log are observers, and any further processing after a step of the
/// path (an analysis, a check that ends the run) is one more.
pub trait IrcObserver: Send {
    /// Something happened: the starting structure was examined, a first point was tested,
    /// a point of the path was completed ([`IrcEvent::Step`]), or a side ended.
    fn on_event(&mut self, event: &IrcEvent) -> MoptResult<()>;

    /// Asked after every evaluation; `true` interrupts the run cleanly.
    fn should_stop(&mut self) -> bool {
        false
    }
}

/// A closure as an observer: it sees every event, and the run is interrupted when it
/// returns `false`.
pub struct FnObserver<F: FnMut(&IrcEvent) -> MoptResult<bool> + Send> {
    function: F,
    stop: bool,
}

impl<F: FnMut(&IrcEvent) -> MoptResult<bool> + Send> FnObserver<F> {
    /// Observe with `function`.
    pub fn new(function: F) -> Self {
        Self {
            function,
            stop: false,
        }
    }
}

impl<F: FnMut(&IrcEvent) -> MoptResult<bool> + Send> IrcObserver for FnObserver<F> {
    fn on_event(&mut self, event: &IrcEvent) -> MoptResult<()> {
        if !(self.function)(event)? {
            self.stop = true;
        }
        Ok(())
    }

    fn should_stop(&mut self) -> bool {
        self.stop
    }
}

impl IrcObserver for super::observers::StopFile {
    fn on_event(&mut self, _event: &IrcEvent) -> MoptResult<()> {
        Ok(())
    }

    fn should_stop(&mut self) -> bool {
        self.check()
    }
}

fn notify(irc: &mut Irc, observers: &mut [&mut dyn IrcObserver]) -> MoptResult<()> {
    for event in irc.drain_events() {
        for observer in observers.iter_mut() {
            observer.on_event(&event)?;
        }
    }
    Ok(())
}

/// Run `irc` on `surface` to its end. The observers see every event; after every
/// evaluation they are asked whether to stop, and `None` is returned when one says so (the
/// run is interrupted).
pub fn run_irc(
    irc: &mut Irc,
    surface: &dyn IrcSurface,
    observers: &mut [&mut dyn IrcObserver],
) -> MoptResult<Option<IrcEnd>> {
    loop {
        let action = irc.ask()?;
        notify(irc, observers)?;
        match action {
            IrcAction::Finished(end) => return Ok(Some(end)),
            IrcAction::Evaluate(request) => {
                let value = surface.evaluate(&request).context(format!(
                    "IRC point {} ({:?})",
                    request.point.step, request.point.role
                ))?;
                irc.tell(value)?;
                notify(irc, observers)?;
                // Every observer is asked, so that each sees the question.
                let mut stop = false;
                for observer in observers.iter_mut() {
                    stop |= observer.should_stop();
                }
                if stop {
                    return Ok(None);
                }
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::irc::{BranchEnd, IrcSettings, IrcSpace};
    use crate::pes::native::muller_brown::MB_SADDLES;
    use crate::pes::objective::MullerBrown2D;
    use faer::Col;

    #[test]
    fn an_objective_is_followed_and_a_stop_interrupts() {
        let objective = MullerBrown2D::default();
        let surface = ObjectiveSurface(&objective);
        let settings = IrcSettings {
            step_size: 0.05,
            ..IrcSettings::default()
        };
        let x = Col::from_fn(2, |i| MB_SADDLES[0][i]);
        let mut irc = Irc::new(settings.clone(), IrcSpace::variables(2), x.clone()).unwrap();
        let mut steps = 0;
        let mut counter = FnObserver::new(|event: &IrcEvent| {
            if matches!(event, IrcEvent::Step { .. }) {
                steps += 1;
            }
            Ok(true)
        });
        let end = run_irc(&mut irc, &surface, &mut [&mut counter])
            .unwrap()
            .unwrap();
        assert!(end.branches.iter().all(|b| b.end == BranchEnd::Converged));
        assert_eq!(steps, end.branches.iter().map(|b| b.points).sum::<usize>());
        // An observer ends the run after the third point of the path.
        let mut irc = Irc::new(settings, IrcSpace::variables(2), x).unwrap();
        let mut seen = 0;
        let mut third = FnObserver::new(|event: &IrcEvent| {
            if matches!(event, IrcEvent::Step { .. }) {
                seen += 1;
            }
            Ok(seen < 3)
        });
        let stopped = run_irc(&mut irc, &surface, &mut [&mut third]).unwrap();
        assert!(stopped.is_none());
        // The starting structure and three points.
        assert_eq!(irc.evaluations(), 4);
    }

    #[test]
    fn what_the_calculator_says_at_a_point_goes_to_the_log() {
        use crate::core_infra::log::{Level, MemorySink};
        use crate::domain::{EvalOutput, OrbitalData, SpinCharge};
        use crate::io::xyz::parse_xyz_str;
        use crate::irc::{PointRequest, PointRole};
        use crate::pes::native::LennardJones;

        /// A calculator with a warning and a spin.
        struct Talking;

        impl Calculator for Talking {
            fn name(&self) -> &str {
                "talking"
            }

            fn evaluate(&self, sys: &MolecularSystem, req: &EvalRequest) -> MoptResult<EvalOutput> {
                let mut out = LennardJones.evaluate(sys, req)?;
                out.orbitals = Some(OrbitalData {
                    spin_square: Some(0.7512),
                    ..OrbitalData::default()
                });
                Ok(out)
            }

            fn take_warnings(&self) -> Vec<String> {
                vec!["the wavefunction was started anew".to_string()]
            }
        }

        let start = parse_xyz_str(
            "2\n\nAr 0.0 0.0 0.0\nAr 3.9 0.0 0.0\n",
            SpinCharge::default(),
            None,
        )
        .unwrap();
        let sink = MemorySink::new();
        let log = Logger::silent();
        log.add_sink(Level::Info, Box::new(sink.clone()));
        let request = IrcRequest {
            point: PointRequest {
                x: start.coords_flat_bohr(),
                need_hessian: false,
                role: PointRole::Path,
                step: 7,
            },
            branch: None,
        };
        let told = MoleculeSurface {
            calc: &Talking,
            terms: &[],
            start: &start,
            excited_state: 0,
            log: Some(&log),
            states: None,
            iteration: 0,
        };
        let value = told.evaluate(&request).unwrap();
        let lines: Vec<String> = sink.lines().into_iter().map(|(_, line)| line).collect();
        assert_eq!(
            lines,
            [
                "Warning: IRC point 7 (Path): the wavefunction was started anew",
                "IRC point 7 (Path): <S**2> = 0.751200",
            ]
        );
        // Without a log the surface is the same and nothing is written.
        let silent = MoleculeSurface {
            calc: &Talking,
            terms: &[],
            start: &start,
            excited_state: 0,
            log: None,
            states: None,
            iteration: 0,
        };
        assert_eq!(silent.evaluate(&request).unwrap().energy, value.energy);
        assert_eq!(sink.lines().len(), 2);
    }
}
