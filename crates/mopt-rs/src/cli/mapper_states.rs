//! The calculations of the mapper between the electronic states of a run (`states`): the
//! minimum-energy crossing point of the state of a job and another state, and a minimization on
//! another state (the descents of a crossing point).
//!
//! The crossing point is optimized with the settings of the transition-state search
//! (`step3_settings`) at order 0 on the surface of the gradient projection (`-mf opt_mesx` for
//! states of different spin or charge, `-mf opt_meci` with the branching-plane updating for two
//! roots of one spin; `pes::combine::projection`), with the method and the exact Hessians of
//! the crossing searches unless the step names its own
//! ([`crate::cli::mapper_settings::CROSSING_METHOD`]). At its end both states are evaluated:
//! the point is a crossing point when the optimization converged and the gap is at most
//! `crossing_gap` ([`crate::cli::mapper_settings::DEFAULT_CROSSING_GAP`]); with a larger gap,
//! or a conical intersection whose check of the branching plane ran out of corrections, the
//! search has found no crossing point, which is said in a warning, and no meta-IRC runs from
//! its end. An accepted point is checked within the seam (`crossing_seam_check`, on by
//! default; `vib::seam_analysis`, as `freqmain` does it): the number of its imaginary modes
//! there.
//!
//! The AFIR paths towards crossing points (`crossing_afir`) run on a model function of the two
//! states whose minima lie near their crossing points: for states of different spin or charge
//! the seam model function `(E_1 + E_2) / 2 + (E_1 - E_2)^2 / alpha` (Sato et al., J. Phys.
//! Chem. A 2009, 113, 1704; `-mf seam`), for two roots of one spin the Avoiding model function
//! of Maeda, Ohno, and Morokuma (`-mf avoiding`, or `-mf avoiding_lower` for the lower adiabatic
//! state; `pes::combine::AvoidingModel`).
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::sync::Arc;

use super::CommandJobs;
use crate::cli::optmain::state_surface;
use crate::cli::variable_cell::cell_states;
use crate::domain::{EvalRequest, MolecularSystem, Quantity, SpinCharge};
use crate::error::{MoptError, MoptResult};
use crate::math::linalg::col_norm;
use crate::opt::cell::gigapascal_per_atomic_unit;
use crate::opt::state_machine::{GeometryOptimizer, Termination};
use crate::pes::calculator::Calculator;
use crate::pes::combine::{model_function, two_states_of, PesExpr, StateCombiner, TaskSpec};
use crate::vib::{seam_analysis, SecondDirection};
use crate::workflow::irc_cell::CellStates;
use crate::workflow::mapper::network::{CrossingKind, SeamCheck};
use crate::workflow::mapper::{
    crossing_kind, surface_name, Configuration, CrossingPath, CrossingSearch, ElectronicState,
    FrameSink, MetaIrc, OptimizeRequest, Optimized, Stage,
};
use crate::workflow::CalculatorSet;

/// The gradient projection of `-mf` for the crossing points of `kind`.
fn projection_name(kind: CrossingKind) -> &'static str {
    match kind {
        CrossingKind::Mecp => "opt_mesx",
        CrossingKind::Meci => "opt_meci",
    }
}

/// The trust radius `max` (Bohr) of the settings, at most `cap`.
fn capped(max: Option<Quantity<crate::domain::Bohr>>, cap: f64) -> Quantity<crate::domain::Bohr> {
    Quantity::new(max.map_or(cap, |m| m.value().min(cap)))
}

/// The number of imaginary modes within the seam at the end `end` of the crossing search
/// `opt` (its surface of two states), by the one criterion
/// (`opt::mode_kill::IMAGINARY_CURVATURE`): the analysis of `freqmain` at a crossing, the
/// Hessians of both states and, for a conical intersection, the second direction of the
/// branching plane from the gradients at displaced structures
/// ([`crate::cli::freqmain::probed_plane`]; from the Hessians where that gives none).
fn seam_modes(
    calc: &dyn Calculator,
    opt: &GeometryOptimizer,
    end: &MolecularSystem,
) -> MoptResult<usize> {
    let tasks = opt.hessian_tasks(end)?;
    let results = CalculatorSet::single(calc).evaluate(&tasks)?;
    let crossing = opt
        .crossing(end, results)?
        .ok_or_else(|| MoptError::calc("the surface is not a crossing of two states"))?;
    // The rigid-body motions are projected as the search projected them.
    let rigid = calc.is_rigid_body_invariant();
    let probed = crate::cli::freqmain::probed_plane(opt, calc, end, &crossing, rigid)?;
    let states = &crossing.states;
    let hessians = states
        .hessians
        .as_ref()
        .ok_or_else(|| MoptError::calc("the states of the crossing came without Hessians"))?;
    let (plane, second) = match &probed {
        Some(plane) => (&plane[..], SecondDirection::Given),
        None => (&crossing.plane[..], SecondDirection::FromHessians),
    };
    let seam = seam_analysis(
        end,
        [states.energies[0], states.energies[1]],
        [&states.gradients[0], &states.gradients[1]],
        [&hessians[0], &hessians[1]],
        plane,
        second,
        rigid,
    )?;
    Ok(seam.modes.imaginary_modes())
}

/// `sys` with the charge and multiplicity of `state`.
fn on_state(sys: &MolecularSystem, state: &ElectronicState) -> MoptResult<MolecularSystem> {
    let mut out = sys.clone();
    out.spin_charge = SpinCharge::checked(i64::from(state.charge), i64::from(state.multiplicity))?;
    Ok(out)
}

impl CommandJobs<'_> {
    /// The state of index `i` of the run.
    fn run_state(&self, i: usize) -> MoptResult<ElectronicState> {
        self.run_steps.states.get(i).copied().ok_or_else(|| {
            MoptError::config(format!(
                "state {} of a run of {} states",
                i + 1,
                self.run_steps.states.len()
            ))
        })
    }

    /// [`crate::workflow::mapper::MapperJobs::energy`] of a structure with `calc`.
    pub(super) fn energy_with(
        &self,
        calc: &dyn Calculator,
        sys: &MolecularSystem,
        on: Option<usize>,
    ) -> MoptResult<f64> {
        let (at, excited_state) = match on {
            Some(state) => {
                let target = self.run_state(state)?;
                (on_state(sys, &target)?, target.excited_state)
            }
            None => (sys.clone(), self.steps.minimum.excited_state),
        };
        let request = EvalRequest {
            excited_state,
            ..EvalRequest::ENERGY_ONLY
        };
        Ok(calc.evaluate(&at, &request)?.energy.value())
    }

    /// [`crate::workflow::mapper::MapperJobs::crossing`] of a structure with `calc`.
    pub(super) fn crossing_with(
        &self,
        calc: &dyn Calculator,
        sys: &MolecularSystem,
        other: usize,
        label: &str,
        frames: &mut dyn FrameSink,
    ) -> MoptResult<CrossingSearch> {
        let (a, b) = (self.run_state(self.state)?, self.run_state(other)?);
        let name = match crossing_kind(&a, &b) {
            CrossingKind::Mecp => "opt_mesx",
            CrossingKind::Meci => "opt_meci",
        };
        let spec = vec![
            name.to_string(),
            b.charge.to_string(),
            b.multiplicity.to_string(),
            b.excited_state.to_string(),
        ];
        let o = &self.steps.saddle;
        let models = self.run_steps.crossing_models;
        let mut settings = o.settings.clone();
        settings.saddle_order = 0;
        settings.mode_kill = None;
        settings.rigid_body_invariant = calc.is_rigid_body_invariant();
        settings.trust.max = Some(capped(settings.trust.max, models.max_step));
        // A crossing point is a minimum of the projected surface: the method and the exact
        // Hessians of the crossing searches, unless step3_settings name either (then the
        // search takes the settings of the step as they are).
        if !models.own_method && !models.own_exact_hessians {
            settings.methods = vec![crate::cli::mapper_settings::CROSSING_METHOD.to_string()];
            settings.exact_hessian_interval = crate::cli::mapper_settings::CROSSING_EXACT_HESSIANS;
        }
        let start = on_state(sys, &a)?;
        let mut searched: Option<GeometryOptimizer> = None;
        let mut seam_check_failed = false;
        // On a variable cell the atoms and the cell move together, on the gradient projection
        // of the enthalpies of the two states in the variables of the cell; the mean enthalpy
        // is the energy of the point, as the enthalpy is that of the minima. The stop at a
        // parallel region is one of the optimizer of structures and does not run there.
        let (end, stationary, iterations, parallel, interrupted, pv) =
            match super::cell::variable_cell(self.steps) {
                Some(cell) => {
                    let mut on_cell = o.clone();
                    on_cell.settings = settings;
                    let run = self.cell_run_with(
                        calc,
                        &start,
                        &[],
                        &on_cell,
                        cell_states(&spec, a.excited_state)?,
                        0,
                        &[],
                        label,
                        frames,
                        cell.coordinates(&start)?,
                    )?;
                    let volume = run.system.active_cell().map_or(0.0, |c| c.volume_bohr3());
                    let pv = cell.pressure_gpa / gigapascal_per_atomic_unit() * volume;
                    (
                        run.system,
                        run.converged,
                        run.iterations,
                        None,
                        run.interrupted,
                        pv,
                    )
                }
                None => {
                    let mut opt = GeometryOptimizer::new(settings, start)?;
                    let surface = state_surface(&spec, a.excited_state)?.ok_or_else(|| {
                        MoptError::config(format!("{label}: no surface of the two states"))
                    })?;
                    opt.set_surface(surface)?;
                    let (outcome, parallel) = self.run_watched(
                        &mut opt,
                        calc,
                        label,
                        frames,
                        self.run_steps
                            .crossing_models
                            .parallel
                            .map(|p| (p, self.run_steps.crossing_models.gap_limit)),
                    )?;
                    let interrupted =
                        outcome.termination.is_none() && parallel.is_none() && self.stop.stopped();
                    seam_check_failed = outcome.termination == Some(Termination::SeamCheckFailed);
                    searched = Some(opt);
                    (
                        outcome.system,
                        outcome.termination == Some(Termination::Converged),
                        outcome.iterations,
                        parallel,
                        interrupted,
                        0.0,
                    )
                }
            };
        if interrupted {
            return Ok(CrossingSearch {
                point: Configuration::Structure(end),
                energy: f64::NAN,
                gap: f64::NAN,
                gradient_difference: None,
                converged: false,
                stationary: false,
                gap_limit: models.gap_limit,
                parallel: None,
                interrupted: true,
                seam_check_failed: false,
                seam: None,
            });
        }
        // Both states at the end.
        let end_a = on_state(&end, &a)?;
        let end_b = on_state(&end, &b)?;
        let va = calc.evaluate(
            &end_a,
            &EvalRequest {
                excited_state: a.excited_state,
                ..EvalRequest::ENERGY_GRAD
            },
        )?;
        let vb = calc.evaluate(
            &end_b,
            &EvalRequest {
                excited_state: b.excited_state,
                ..EvalRequest::ENERGY_GRAD
            },
        )?;
        let (ea, eb) = (va.energy.value(), vb.energy.value());
        let gap = ea - eb;
        let difference = col_norm(&(&va.gradient_bohr - &vb.gradient_bohr));
        // The two states at the end, before anything runs down from it: further apart than
        // `crossing_gap`, the end is no crossing point.
        // The gap is held against `crossing_gap` for both kinds of crossing point: a minimum
        // on a seam of two spin states and a conical intersection are both points where the
        // two states are degenerate. The convergence of the projected gradient does not make
        // the gap small enough by itself: its component `2 (E_1 - E_2)` along the difference
        // gradient bounds the gap only by `sqrt(3N)` times the force threshold over two, which
        // grows with the size of the system, so the gap is a condition of its own.
        let gap_limit = models.gap_limit;
        let converged = stationary && gap.abs() <= gap_limit && parallel.is_none();
        self.log.debug(&format!(
            "{label}: {} iterations, E({}) = {ea:.10}, E({}) = {eb:.10} Hartree, gap {gap:.3e}, \
             |g1 - g2| {difference:.3e} Hartree/Bohr, {}",
            iterations,
            surface_name(self.state),
            surface_name(other),
            if converged {
                "converged"
            } else if seam_check_failed {
                "not converged: the check of the branching plane ran out of corrections"
            } else if stationary {
                "converged, but the gap is too large for a crossing point"
            } else {
                "not converged"
            }
        ));
        // The check to second order within the seam of an accepted point.
        let seam = converged.then(|| match (&searched, models.seam_check) {
            (_, false) => SeamCheck::NotChecked("switched off, crossing_seam_check=off".into()),
            (None, true) => SeamCheck::NotChecked(
                "a variable cell: the analysis within the seam is that of the atoms in a fixed \
                 cell"
                    .into(),
            ),
            (Some(opt), true) => match seam_modes(calc, opt, &end_a) {
                Ok(n) => SeamCheck::Checked(n),
                Err(e) => SeamCheck::NotChecked(format!("the analysis failed: {e}")),
            },
        });
        Ok(CrossingSearch {
            point: Configuration::Structure(end_a),
            energy: 0.5 * (ea + eb) + pv,
            gap,
            gradient_difference: Some(difference),
            converged,
            stationary,
            gap_limit,
            parallel,
            interrupted: false,
            seam_check_failed,
            seam,
        })
    }

    /// [`crate::workflow::mapper::MapperJobs::crossing_path`] of a structure with `calc`: the
    /// settings of the stage of `request`, the step at most `crossing_step`. On the gradient
    /// projection the bias potentials of `request` stand with the energies of both states
    /// (`E_k + bias` in the place of `E_k`), so that they cancel in the gap and act within the
    /// seam space; on a model function they are added to it. The energy of every point of the
    /// path is the function whose local minima are the guesses of crossing points: the model
    /// function itself, and for a path on the gradient projection the seam model function
    /// `(E_1 + E_2) / 2 + (E_1 - E_2)^2 / alpha` of the two states there (it only chooses the
    /// guesses and has no part in the path; Harabuchi, Taketsugu, Maeda, Chem. Phys. Lett.
    /// 2017, 674, 141, eq. 7).
    pub(super) fn crossing_path_with(
        &self,
        calc: &dyn Calculator,
        sys: &MolecularSystem,
        other: usize,
        path: CrossingPath,
        request: &OptimizeRequest,
        frames: &mut dyn FrameSink,
    ) -> MoptResult<Optimized> {
        let (a, b) = (self.run_state(self.state)?, self.run_state(other)?);
        let models = self.run_steps.crossing_models;
        let kind = crossing_kind(&a, &b);
        let second = SpinCharge::checked(i64::from(b.charge), i64::from(b.multiplicity))?;
        let state1 = TaskSpec::whole("State1").excited_state(a.excited_state);
        let state2 = TaskSpec::whole("State2")
            .spin_charge(second)
            .excited_state(b.excited_state);
        let model = || -> MoptResult<Box<dyn StateCombiner>> {
            match kind {
                CrossingKind::Mecp => model_function("seam", Some(models.smf_alpha), None),
                CrossingKind::Meci if models.amf_upper => {
                    model_function("avoiding", Some(models.amf_alpha), None)
                }
                CrossingKind::Meci => {
                    model_function("avoiding_lower", Some(models.amf_alpha), None)
                }
            }
        };
        if let Some(cell) = super::cell::variable_cell(self.steps) {
            return self.crossing_path_on_cell(
                calc,
                &on_state(sys, &a)?,
                path,
                match path {
                    CrossingPath::Gp => model_function(projection_name(kind), None, None)?,
                    CrossingPath::Model => model()?,
                },
                (state1, state2),
                request,
                frames,
                cell,
            );
        }
        let surface = match path {
            CrossingPath::Gp => {
                let combiner = model_function(projection_name(kind), None, None)?;
                let (mut first, mut last) = (PesExpr::Task(state1), PesExpr::Task(state2));
                for (label, term) in &request.terms {
                    first = first.plus_term(&format!("{label}_State1"), Arc::clone(term));
                    last = last.plus_term(&format!("{label}_State2"), Arc::clone(term));
                }
                PesExpr::combine(
                    combiner,
                    vec!["State1".to_string(), "State2".to_string()],
                    vec![first, last],
                )?
            }
            CrossingPath::Model => two_states_of(model()?, state1, state2)?,
        };
        let o = self.steps.config(request.stage);
        let mut settings = o.settings.clone();
        settings.mode_kill = None;
        settings.saddle_order = 0;
        settings.rigid_body_invariant = crate::pes::term::frame_invariant(
            calc.is_rigid_body_invariant(),
            request.terms.iter().map(|(_, t)| t.as_ref()),
        );
        settings.trust.max = Some(capped(settings.trust.max, models.max_step));
        let mut opt = GeometryOptimizer::new(settings, on_state(sys, &a)?)?;
        opt.set_surface(surface)?;
        if path == CrossingPath::Model {
            for (label, term) in &request.terms {
                opt.add_term(label, Arc::clone(term))?;
            }
        }
        let (outcome, mut points, states) =
            self.run_recorded_states(&mut opt, calc, &request.label, frames)?;
        if path == CrossingPath::Gp {
            for (point, state) in points.iter_mut().zip(&states) {
                if let Some([e1, e2]) = state {
                    let gap = e1 - e2;
                    point.energy = 0.5 * (e1 + e2) + gap * gap / models.smf_alpha;
                }
            }
        }
        self.log.debug(&format!(
            "{}: {} iterations, F = {:.10} Hartree, {}",
            request.label,
            outcome.iterations,
            outcome.energy.value(),
            super::ended(outcome.termination)
        ));
        Ok(Optimized {
            index_wavenumbers: None,
            energy: outcome.energy.value(),
            system: Configuration::Structure(outcome.system),
            converged: outcome.termination == Some(Termination::Converged),
            interrupted: outcome.termination.is_none() && self.stop.stopped(),
            path: points,
            wavenumbers: None,
        })
    }

    /// [`Self::crossing_path_with`] on a variable cell: the atoms and the cell together, on
    /// the two states `states` (from `start`, in the first) combined by `combiner` in the
    /// variables of the cell. On the gradient projection the bias potentials of `request`
    /// stand with the enthalpies of both states ([`CellStates::with_terms_in_states`]), on a
    /// model function they are added to it; the energy of every point of a path on the
    /// gradient projection is the seam model function of the enthalpies of the two states
    /// there, as in a fixed cell.
    #[allow(clippy::too_many_arguments)]
    fn crossing_path_on_cell(
        &self,
        calc: &dyn Calculator,
        start: &MolecularSystem,
        path: CrossingPath,
        combiner: Box<dyn StateCombiner>,
        states: (TaskSpec, TaskSpec),
        request: &OptimizeRequest,
        frames: &mut dyn FrameSink,
        cell: &crate::cli::variable_cell::VariableCellSettings,
    ) -> MoptResult<Optimized> {
        let models = self.run_steps.crossing_models;
        let combined =
            CellStates::of(Some(two_states_of(combiner, states.0, states.1)?))?.map(|s| {
                if path == CrossingPath::Gp {
                    s.with_terms_in_states()
                } else {
                    s
                }
            });
        let o = self.steps.config(request.stage);
        let mut on_cell = o.clone();
        on_cell.settings.mode_kill = None;
        on_cell.settings.saddle_order = 0;
        on_cell.settings.trust.max = Some(capped(on_cell.settings.trust.max, models.max_step));
        let run = self.cell_run_with(
            calc,
            start,
            &request.terms,
            &on_cell,
            combined,
            0,
            &[],
            &request.label,
            frames,
            cell.coordinates(start)?,
        )?;
        let mut points = run.path;
        if path == CrossingPath::Gp {
            for (point, state) in points.iter_mut().zip(&run.state_enthalpies) {
                if let [h1, h2] = state[..] {
                    let gap = h1 - h2;
                    point.energy = 0.5 * (h1 + h2) + gap * gap / models.smf_alpha;
                }
            }
        }
        self.log.debug(&format!(
            "{}: {} iterations on the variable cell, F = {:.10} Hartree, {}",
            request.label,
            run.iterations,
            run.enthalpy,
            if run.converged {
                "converged"
            } else if run.interrupted {
                "stopped"
            } else {
                "not converged"
            }
        ));
        Ok(Optimized {
            index_wavenumbers: None,
            energy: run.enthalpy,
            system: Configuration::Structure(run.system),
            converged: run.converged,
            interrupted: run.interrupted,
            path: points,
            wavenumbers: None,
        })
    }

    /// [`crate::workflow::mapper::MapperJobs::meta_irc_in`] of a structure with `calc`: the
    /// IRC of the run (the machine of `ircmain`) on the state `state` from `sys`, which is no
    /// stationary point of it, so that one side runs down from it.
    pub(super) fn meta_irc_with(
        &self,
        calc: &dyn Calculator,
        sys: &MolecularSystem,
        state: usize,
        label: &str,
        frames: &mut dyn FrameSink,
    ) -> MoptResult<MetaIrc> {
        let target = self.run_state(state)?;
        let steps = self.run_steps.on_states.get(state).ok_or_else(|| {
            MoptError::config(format!("{label}: no steps on {}", surface_name(state)))
        })?;
        let start = on_state(sys, &target)?;
        let outcome = self.irc_in(calc, &start, frames, steps, true)?;
        self.log.debug(&format!(
            "{label}: {}",
            match outcome.branches.first() {
                Some(b) => format!("{} points, ended {}", b.side.energies.len(), b.side.end),
                None => "no path".to_string(),
            }
        ));
        Ok(MetaIrc {
            interrupted: outcome.interrupted,
            branch: outcome.branches.into_iter().next(),
        })
    }

    /// [`crate::workflow::mapper::MapperJobs::minimize_in`] of a structure with `calc`: the
    /// minimization of the run on the state `state`, the structure given its charge and
    /// multiplicity.
    pub(super) fn minimize_with(
        &self,
        calc: &dyn Calculator,
        sys: &MolecularSystem,
        state: usize,
        label: &str,
        frames: &mut dyn FrameSink,
    ) -> MoptResult<Optimized> {
        let target = self.run_state(state)?;
        let steps = self.run_steps.on_states.get(state).ok_or_else(|| {
            MoptError::config(format!("{label}: no steps on {}", surface_name(state)))
        })?;
        let start = on_state(sys, &target)?;
        self.optimize_in(
            calc,
            &start,
            &OptimizeRequest {
                label: label.to_string(),
                stage: Stage::Minimum,
                terms: Vec::new(),
            },
            frames,
            steps,
        )
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::domain::{Element, EvalOutput};
    use crate::math::ad::{eval_value_grad_hessian, HyperDual, Scalar};
    use crate::opt::convergence::ConvergenceCriteria;
    use crate::opt::state_machine::OptimizerSettings;
    use crate::pes::combine::two_states;
    use crate::workflow::run_optimization;

    /// One atom on two harmonic wells by the multiplicity (singlet around the origin, triplet
    /// around (1, 0, 0) and 0.02 Hartree higher), which cross on the plane `x = 0.52`, and a
    /// term both states share, `2 (y^2 - 0.25)^2`: along the seam the mean energy
    /// `y^2 / 2 + 2 (y^2 - 0.25)^2 + ...` has a maximum at `y = 0` and minima at
    /// `y = +-sqrt(1/8)`.
    struct SeamHill;

    impl Calculator for SeamHill {
        fn name(&self) -> &str {
            "seam_hill"
        }

        fn evaluate(
            &self,
            sys: &MolecularSystem,
            req: &EvalRequest,
        ) -> MoptResult<crate::domain::EvalOutput> {
            let triplet = sys.spin_charge.multiplicity == 3;
            let x: Vec<f64> = sys.coords_flat_bohr().iter().copied().collect();
            let (e, g, h) = eval_value_grad_hessian(&x, None, |r: &[HyperDual]| {
                let (at, shift) = if triplet { (1.0, 0.02) } else { (0.0, 0.0) };
                let dx = r[0] - HyperDual::from_f64(at);
                let well = HyperDual::from_f64(0.5) * (dx * dx + r[1] * r[1] + r[2] * r[2]);
                let hill = r[1] * r[1] - HyperDual::from_f64(0.25);
                well + HyperDual::from_f64(shift) + HyperDual::from_f64(2.0) * hill * hill
            });
            Ok(EvalOutput {
                energy: Quantity::new(e),
                gradient_bohr: g,
                hessian_bohr: req.need_hessian.then_some(h),
                orbitals: None,
                stress: None,
                strain_hessian: None,
            })
        }

        fn is_rigid_body_invariant(&self) -> bool {
            false
        }
    }

    /// The MECP search from `start` and the count of imaginary modes within the seam at its end.
    fn checked_from(start: [f64; 3]) -> (MolecularSystem, usize) {
        let sys = MolecularSystem::from_flat_bohr(
            vec![Element::H],
            &faer::Col::from_fn(3, |i| start[i]),
            SpinCharge::default(),
            None,
        )
        .unwrap();
        let settings = OptimizerSettings {
            methods: vec![crate::cli::mapper_settings::CROSSING_METHOD.to_string()],
            criteria: ConvergenceCriteria::TIGHT,
            rigid_body_invariant: false,
            dissociation_distance: None,
            ..OptimizerSettings::default()
        };
        let mut opt = GeometryOptimizer::new(settings, sys).unwrap();
        opt.set_surface(
            two_states(
                model_function("opt_mesx", None, None).unwrap(),
                TaskSpec::whole("State2").spin_charge(SpinCharge::new(0, 3)),
            )
            .unwrap(),
        )
        .unwrap();
        let out = run_optimization(&mut opt, &SeamHill, &mut []).unwrap();
        assert_eq!(out.termination, Some(Termination::Converged));
        let n = seam_modes(&SeamHill, &opt, &out.system).unwrap();
        (out.system, n)
    }

    #[test]
    fn the_check_within_the_seam_tells_a_minimum_from_a_saddle_point_on_it() {
        // From a start at y = 0 every gradient has no y component: the search ends at the
        // maximum of the seam along y, a saddle point on the seam with one imaginary mode.
        let (end, n) = checked_from([0.3, 0.0, 0.1]);
        let p = end.coords_flat_bohr();
        assert!((p[0] - 0.52).abs() < 1e-4 && p[1].abs() < 1e-9, "{p:?}");
        assert_eq!(n, 1);
        // From y = 0.4 it reaches the minimum of the seam at y = sqrt(1/8): none.
        let (end, n) = checked_from([0.3, 0.4, 0.1]);
        let p = end.coords_flat_bohr();
        assert!(
            (p[0] - 0.52).abs() < 1e-4 && (p[1] - 0.125_f64.sqrt()).abs() < 1e-3,
            "{p:?}"
        );
        assert_eq!(n, 0);
    }
}
