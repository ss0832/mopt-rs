//! Driver of the molecular dynamics state machine: evaluates the positions it asks for on a
//! surface (calculators with the surface expression of the run, or an objective of plain
//! variables) and hands every step to observers (files, log, a `STOP` file, hooks).
//!
//! The simulated time of each request reaches the in-process terms of the surface
//! ([`TermContext::time_au`]), so that time-dependent bias potentials (the nanoreactor's
//! periodic compression, expressions of `t`) follow the dynamics.
//!
//! MultiOptPy: `moleculardynamics.py:MD.md` (the calls of `single_point` and
//! `BiasPotentialCalculation.main` in the loop over the steps).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::sync::{Arc, Mutex};

use faer::Col;

use crate::core_infra::log::Logger;
use crate::domain::{EvalRequest, MolecularSystem};
use crate::error::{MoptError, MoptResult, ResultExt};
use crate::md::hopping::{StateValues, StatesRequest};
use crate::md::{MdAction, MdEnd, MdMachine, MdRequest, MdStepReport, MdValue};
use crate::pes::combine::{EvalTask, PesExpr, DEFAULT_CALCULATOR};
use crate::pes::objective::Objective;
use crate::pes::stress::{term_stress, DEFAULT_STRAIN_STEP};
use crate::pes::term::{PesTerm, TermContext};
use crate::workflow::driver::CalculatorSet;

/// A surface on which the dynamics runs.
pub trait MdSurface {
    /// The surface at the positions of `request`.
    fn evaluate(&self, request: &MdRequest) -> MoptResult<MdValue>;
}

/// Calculators (by label) with the surface expression of a run (bias potentials, combined
/// electronic states) for the atoms of a structure.
///
/// For surface hopping ([`MdRequest::states`]) the surface of the active state is the
/// expression given for it ([`Self::with_states`]); the other states are evaluated by the
/// calculator of the default label, for their energies, and for their gradients where they
/// are neighbours of the active state closer to it than [`StatesRequest::gradient_gap`]
/// (at once for those that were close at the step before, [`StatesRequest::expected`]). The
/// bias potentials of the whole surface are the same on every state: their energy and
/// gradient (the active surface without its calculator part) are added to the others.
pub struct MoleculeMdSurface<'a> {
    calcs: CalculatorSet<'a>,
    surface: Mutex<PesExpr>,
    state_surfaces: Vec<(usize, Mutex<PesExpr>)>,
    template: MolecularSystem,
    initial: MolecularSystem,
    terms: Vec<(String, Arc<dyn PesTerm>)>,
    log: Option<Logger>,
}

impl<'a> MoleculeMdSurface<'a> {
    /// The surface `surface` (whose calculator tasks `calcs` evaluate) for structures like
    /// `template` that start at `initial`; `terms` are the in-process terms added to the
    /// whole surface, whose share of the stress is taken by differences of their energy.
    pub fn new(
        calcs: CalculatorSet<'a>,
        surface: PesExpr,
        template: MolecularSystem,
        terms: Vec<(String, Arc<dyn PesTerm>)>,
    ) -> MoptResult<Self> {
        surface.validate()?;
        Ok(Self {
            calcs,
            surface: Mutex::new(surface),
            state_surfaces: Vec::new(),
            initial: template.clone(),
            template,
            terms,
            log: None,
        })
    }

    /// The calculator `calc` with the bias potentials `terms` on the whole surface.
    pub fn with_terms(
        calc: &'a dyn crate::pes::calculator::Calculator,
        template: MolecularSystem,
        terms: Vec<(String, Arc<dyn PesTerm>)>,
    ) -> MoptResult<Self> {
        let mut surface = PesExpr::single();
        for (label, term) in &terms {
            surface = surface.plus_term(label, Arc::clone(term));
        }
        Self::new(CalculatorSet::single(calc), surface, template, terms)
    }

    /// The surfaces of the electronic states of surface hopping: `(state, expression)`,
    /// the expression of each with its calculator task on that state and the bias
    /// potentials of the run.
    pub fn with_states(mut self, surfaces: Vec<(usize, PesExpr)>) -> MoptResult<Self> {
        let mut out = Vec::new();
        for (state, surface) in surfaces {
            surface.validate()?;
            out.push((state, Mutex::new(surface)));
        }
        self.state_surfaces = out;
        Ok(self)
    }

    /// Write what the calculators say (warnings, `<S**2>`, the output of the program) to
    /// `log` after every evaluation.
    #[must_use]
    pub fn telling(mut self, log: Logger) -> Self {
        self.log = Some(log);
        self
    }

    fn tell_log(&self, step: usize, outputs: &[crate::domain::EvalOutput]) {
        let Some(log) = &self.log else {
            return;
        };
        let of = format!("MD step {step}");
        for warning in self.calcs.take_warnings() {
            log.info(&format!("Warning: {of}: {warning}"));
        }
        for out in outputs {
            if let Some(o) = &out.orbitals {
                if let Some(spin) = o.spin_square {
                    log.info(&format!("{of}: <S**2> = {spin:.6}"));
                }
                if let Some(text) = &o.program_output {
                    log.info(&format!(
                        "{of}: output of the program\n{}",
                        text.trim_end_matches('\n')
                    ));
                }
            }
        }
    }
}

impl MdSurface for MoleculeMdSurface<'_> {
    fn evaluate(&self, request: &MdRequest) -> MoptResult<MdValue> {
        let mut sys = self.template.with_flat_coords_bohr(&request.x)?;
        if request.cell.is_some() {
            sys.cell.clone_from(&request.cell);
        }
        let base = if request.need_hessian {
            EvalRequest::ENERGY_GRAD_HESS
        } else {
            EvalRequest::ENERGY_GRAD
        };
        let req = EvalRequest {
            need_stress: request.need_stress,
            ..base
        };
        let chosen = match &request.states {
            Some(states) => self
                .state_surfaces
                .iter()
                .find(|(s, _)| *s == states.active)
                .map(|(_, m)| m)
                .ok_or_else(|| {
                    MoptError::config(format!(
                        "surface hopping: no surface was set up for state {}",
                        states.active
                    ))
                })?,
            None => &self.surface,
        };
        let mut surface = chosen
            .lock()
            .map_err(|_| MoptError::calc("the surface of the dynamics is poisoned"))?;
        let tasks = surface.tasks(&sys, &req)?;
        let results = self.calcs.evaluate(&tasks)?;
        self.tell_log(request.step, &results);
        let ctx = TermContext::new(request.step, &self.initial, Some(request.time));
        let stress = if request.need_stress {
            Some(self.stress(&sys, &results, &ctx)?)
        } else {
            match results.as_slice() {
                [only] if self.terms.is_empty() => only.stress,
                _ => None,
            }
        };
        // The dipole of the atomic charges of one calculator (isolated structures).
        let dipole = match results.as_slice() {
            [only] if sys.active_cell().is_none() => only
                .orbitals
                .as_ref()
                .and_then(|o| o.atomic_charges.as_ref())
                .filter(|q| q.len() == sys.n_atoms())
                .map(|q| {
                    let x = sys.coords_flat_bohr();
                    std::array::from_fn(|c| {
                        q.iter().enumerate().map(|(i, qi)| qi * x[3 * i + c]).sum()
                    })
                }),
            _ => None,
        };
        // The Hessians of the terms only when the step asks for one (the dynamics uses no
        // other); the evaluation is accepted, so terms with history (metadynamics) record it.
        let surf = surface.evaluate_as_requested(&sys, &req, results, &ctx)?;
        drop(surface);
        let states = match &request.states {
            Some(states) => Some(self.other_states(
                &sys,
                states,
                (surf.total.energy.value(), surf.raw.energy.value()),
                (&surf.total.gradient_bohr, &surf.raw.gradient_bohr),
            )?),
            None => None,
        };
        Ok(MdValue {
            states,
            dipole,
            energy: surf.total.energy.value(),
            gradient: surf.total.gradient_bohr,
            raw_energy: surf.raw.energy.value(),
            raw_gradient: Some(surf.raw.gradient_bohr),
            hessian: if request.need_hessian {
                Some(
                    surf.total
                        .hessian_bohr
                        .ok_or_else(|| MoptError::calc("the surface returned no Hessian"))?,
                )
            } else {
                None
            },
            stress,
        })
    }
}

impl MoleculeMdSurface<'_> {
    /// The states of surface hopping besides the active one, whose whole and raw energies and
    /// gradients are `active` and `active_gradients`.
    fn other_states(
        &self,
        sys: &MolecularSystem,
        request: &StatesRequest,
        active: (f64, f64),
        active_gradients: (&Col<f64>, &Col<f64>),
    ) -> MoptResult<StateValues> {
        let bias = active.0 - active.1;
        let n = sys.dof();
        let bias_gradient =
            Col::<f64>::from_fn(n, |i| active_gradients.0[i] - active_gradients.1[i]);
        let task = |state: usize, gradient: bool| EvalTask {
            label: format!("state {state}"),
            calculator: DEFAULT_CALCULATOR.to_string(),
            system: sys.clone(),
            request: EvalRequest {
                need_gradient: gradient,
                excited_state: state,
                ..EvalRequest::ENERGY_ONLY
            },
        };
        let others: Vec<usize> = request
            .states
            .iter()
            .copied()
            .filter(|&s| s != request.active)
            .collect();
        let tasks: Vec<EvalTask> = others
            .iter()
            .map(|&s| task(s, request.expected.contains(&s)))
            .collect();
        let found = self.calcs.evaluate(&tasks)?;
        let mut energies = Vec::with_capacity(request.states.len());
        let mut gradients: Vec<Option<Col<f64>>> = Vec::with_capacity(request.states.len());
        for &state in &request.states {
            if state == request.active {
                energies.push(active.0);
                gradients.push(Some(active_gradients.0.clone()));
            } else {
                let k = others.iter().position(|&s| s == state).unwrap_or(0);
                energies.push(found[k].energy.value() + bias);
                gradients.push(
                    request
                        .expected
                        .contains(&state)
                        .then(|| Col::from_fn(n, |i| found[k].gradient_bohr[i] + bias_gradient[i])),
                );
            }
        }
        // The gradients of the neighbours close to the active state that were not asked for
        // at once.
        let close: Vec<usize> = request
            .neighbours()
            .into_iter()
            .filter(|&s| {
                request
                    .states
                    .iter()
                    .position(|&t| t == s)
                    .is_some_and(|k| {
                        gradients[k].is_none()
                            && (energies[k] - active.0).abs() < request.gradient_gap
                    })
            })
            .collect();
        if !close.is_empty() {
            let tasks: Vec<EvalTask> = close.iter().map(|&s| task(s, true)).collect();
            let found = self.calcs.evaluate(&tasks)?;
            for (state, out) in close.iter().zip(found) {
                if let Some(k) = request.states.iter().position(|t| t == state) {
                    energies[k] = out.energy.value() + bias;
                    gradients[k] =
                        Some(Col::from_fn(n, |i| out.gradient_bohr[i] + bias_gradient[i]));
                }
            }
        }
        Ok(StateValues {
            states: request.states.clone(),
            energies,
            gradients,
        })
    }

    /// The stress of a surface of one calculator with in-process terms.
    fn stress(
        &self,
        sys: &MolecularSystem,
        results: &[crate::domain::EvalOutput],
        ctx: &TermContext<'_>,
    ) -> MoptResult<crate::pes::stress::Stress> {
        let [only] = results else {
            return Err(MoptError::unsupported(
                "the stress of a surface of several evaluations (-mf) is not available",
            ));
        };
        let mut stress = only.stress.ok_or_else(|| {
            MoptError::calc("the calculator gave no stress (needed by the barostat)")
        })?;
        for (label, term) in &self.terms {
            // The stress the term gives (analytically), differences of its energy otherwise.
            let out = term.evaluate(sys, ctx, false).context(label.clone())?;
            let share = term_stress(term.as_ref(), &out, sys, ctx, DEFAULT_STRAIN_STEP)
                .context(label.clone())?;
            for (row, add) in stress.iter_mut().zip(&share) {
                for (value, part) in row.iter_mut().zip(add) {
                    *value += part;
                }
            }
        }
        Ok(stress)
    }
}

/// An objective of plain variables as a surface.
pub struct ObjectiveMdSurface<'a>(pub &'a dyn Objective);

impl MdSurface for ObjectiveMdSurface<'_> {
    fn evaluate(&self, request: &MdRequest) -> MoptResult<MdValue> {
        let v = self.0.evaluate(&request.x, request.need_hessian)?;
        if request.need_hessian && v.hessian.is_none() {
            return Err(MoptError::calc(format!(
                "objective '{}' returned no Hessian",
                self.0.name()
            )));
        }
        let mut value = MdValue::plain(v.value, v.gradient);
        value.hessian = v.hessian;
        Ok(value)
    }
}

/// Sees every step of a run: the files and the log are observers, and so is any further
/// processing after a step (Rust hooks; Python callbacks through the Python driver).
pub trait MdObserver: Send {
    /// A step was completed (step 0: the start).
    fn on_step(&mut self, report: &MdStepReport) -> MoptResult<()>;

    /// The run ended (`None`: it was stopped).
    fn on_finish(&mut self, end: Option<MdEnd>) -> MoptResult<()> {
        let _ = end;
        Ok(())
    }

    /// Asked after every step; `true` interrupts the run cleanly.
    fn should_stop(&mut self) -> bool {
        false
    }
}

/// A closure as an observer: it sees every step, and the run stops when it returns `false`.
pub struct FnMdObserver<F: FnMut(&MdStepReport) -> MoptResult<bool> + Send> {
    function: F,
    stop: bool,
}

impl<F: FnMut(&MdStepReport) -> MoptResult<bool> + Send> FnMdObserver<F> {
    /// Observe with `function`.
    pub fn new(function: F) -> Self {
        Self {
            function,
            stop: false,
        }
    }
}

impl<F: FnMut(&MdStepReport) -> MoptResult<bool> + Send> MdObserver for FnMdObserver<F> {
    fn on_step(&mut self, report: &MdStepReport) -> MoptResult<()> {
        if !(self.function)(report)? {
            self.stop = true;
        }
        Ok(())
    }

    fn should_stop(&mut self) -> bool {
        self.stop
    }
}

impl MdObserver for super::observers::StopFile {
    fn on_step(&mut self, _report: &MdStepReport) -> MoptResult<()> {
        Ok(())
    }

    fn should_stop(&mut self) -> bool {
        self.check()
    }
}

/// Run `md` on `surface` to its end. The observers see every step; after every step they
/// are asked whether to stop, and `None` is returned when one says so.
pub fn run_md(
    md: &mut dyn MdMachine,
    surface: &dyn MdSurface,
    observers: &mut [&mut dyn MdObserver],
) -> MoptResult<Option<MdEnd>> {
    loop {
        match md.ask() {
            MdAction::Finished(end) => {
                for o in observers.iter_mut() {
                    o.on_finish(Some(end))?;
                }
                return Ok(Some(end));
            }
            MdAction::Evaluate(request) => {
                let value = surface
                    .evaluate(&request)
                    .context(format!("MD step {}", request.step))?;
                let report = md.tell(value)?;
                for o in observers.iter_mut() {
                    o.on_step(&report)?;
                }
                let mut stop = false;
                for o in observers.iter_mut() {
                    stop |= o.should_stop();
                }
                if stop && !md.finished() {
                    for o in observers.iter_mut() {
                        o.on_finish(None)?;
                    }
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
    use crate::domain::SpinCharge;
    use crate::io::xyz::parse_xyz_str;
    use crate::md::{MdSettings, MdSpace, MolecularDynamics, ThermostatSettings};
    use crate::pes::bias::{AdBias, NanoReactor};
    use crate::pes::native::LennardJones;

    #[test]
    fn the_time_of_the_dynamics_reaches_the_nanoreactor_and_a_stop_ends_the_run() {
        let sys = parse_xyz_str(
            "2\n\nAr 0.0 0.0 0.0\nAr 9.0 0.0 0.0\n",
            SpinCharge::default(),
            None,
        )
        .unwrap();
        // The narrow sphere (radius 2 Bohr) holds from 100 a.u. on: the second atom is
        // pulled only then.
        let wall = NanoReactor {
            atoms: sys.atoms().map(|a| (a, 39.95)).collect(),
            center: crate::pes::bias::nanoreactor::MULTIOPTPY_CENTER,
            r1: 50.0,
            k1: 1e-3,
            r2: 2.0,
            k2: 1e-3,
            tau: 100.0,
            period: 1e9,
        };
        let terms: Vec<(String, Arc<dyn PesTerm>)> =
            vec![("nrp".to_string(), Arc::new(AdBias::new(wall)))];
        let surface = MoleculeMdSurface::with_terms(&LennardJones, sys.clone(), terms).unwrap();
        let settings = MdSettings {
            time_step: 20.0,
            steps: 20,
            temperature: 0.0,
            thermostat_settings: ThermostatSettings::default(),
            ..MdSettings::default()
        };
        let space = MdSpace::atoms(&[39.95, 39.95]).unwrap();
        let mut md =
            MolecularDynamics::new(settings, space, sys.coords_flat_bohr(), Some(sys)).unwrap();
        let mut seen = Vec::new();
        let mut watch = FnMdObserver::new(|r: &MdStepReport| {
            seen.push((r.time, r.energy - r.raw_energy));
            Ok(r.step < 10)
        });
        let end = run_md(&mut md, &surface, &mut [&mut watch]).unwrap();
        assert!(end.is_none());
        assert_eq!(seen.len(), 11);
        assert!(seen
            .iter()
            .filter(|(t, _)| *t < 100.0)
            .all(|(_, b)| *b == 0.0));
        assert!(seen
            .iter()
            .filter(|(t, _)| *t > 100.0)
            .all(|(_, b)| *b > 0.0));
    }
}
