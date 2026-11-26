//! The calculations of `run_mapper` on a variable cell (`variable_cell` in the settings of
//! the steps, `-vc` of `optmain`): the atoms and the cell move together under the external
//! pressure, on the enthalpy `E + P V` in the variables of the variable-cell optimization
//! (`opt::cell`: quasi-Cartesian coordinates of the atoms and scaled lattice vectors). The
//! enthalpy with the bias potentials of a request is an objective of plain variables
//! ([`CellObjective`]), so that the optimizer of plain variables, its transition-state
//! searches, and the two-point searches run on it as they are; the IRC is that of `ircmain
//! -vc`, mass-weighted with the masses of the variables. The structures of the network keep
//! their cells, and the energies of its nodes are enthalpies.
//!
//! - Minima, AFIR paths, transition states: optimizations of the variables made from the
//!   structure of the request. The push of an AFIR path is a bias potential of the objective,
//!   and its share of the stress moves the cell (PBC-AFIR on a variable cell). A minimum is
//!   checked with the Hessian of the variables (the curvatures of the atoms and the cell);
//!   a negative curvature is left once by the step of the mode kill and the structure
//!   optimized again.
//! - Vibrations (the kinetics): those of the atoms in the cell of the structure, from the
//!   block of the atoms of that Hessian at the structure itself, where the variables of the
//!   atoms are their Cartesian coordinates.
//! - The modes along which the mapper displaces a structure: of the atoms in its cell (that
//!   same block); the number of negative curvatures is that of the atoms and the cell.
//! - Two-point searches (the relaxation of an AFIR path, the missing connections): on the
//!   variables of the first structure, so that the cell is a part of every image, as in the
//!   solid-state NEB.
//!
//! References: M. Gubler, M. Krummenacher, H. Huber, S. Goedecker, J. Comput. Phys. X 2023,
//! 17, 100131 (the variables); D. Sheppard, P. Xiao, W. Chemelewski, D. D. Johnson,
//! G. Henkelman, J. Chem. Phys. 2012, 136, 074103 (the cell in the images of a chain).
//!
//! MultiOptPy: no counterpart (new in mopt-rs; MultiOptPy's mapper explores molecules only).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::sync::Arc;

use faer::{Col, Mat};

use super::{irc_outcome, say_not_stationary, CommandJobs, IrcCollector, StopWatch};
use crate::cli::mapper_minima::{mapper_mode_kill, vibrations};
use crate::cli::mapper_settings::StepConfigs;
use crate::cli::optmain::OptmainConfig;
use crate::cli::variable_cell::{cell_states, VariableCellSettings};
use crate::domain::MolecularSystem;
use crate::error::{MoptError, MoptResult};
use crate::opt::cell::VariableCell;
use crate::opt::mode_kill::{examine_retaining, ModeKillVerdict};
use crate::opt::state_machine::Termination;
use crate::opt::vector::{VectorOptimizer, VectorSettings};
use crate::pes::calculator::Calculator;
use crate::pes::objective::Objective;
use crate::pes::term::PesTerm;
use crate::workflow::irc_cell::{CellObjective, CellStates};
use crate::workflow::mapper::{
    Configuration, FrameSink, IrcOutcome, OptimizeRequest, Optimized, OrderedSaddle,
    PairSearchOutcome, PairSearchRequest, PathPoint, PointMode, PointModes, Stage,
};
use crate::workflow::pair_search::{PairInput, VariableSurface};
use crate::workflow::run_vector_optimization_until;

/// The variable cell of the steps `steps` (`None`: the cell is fixed).
pub(in crate::cli) fn variable_cell(steps: &StepConfigs) -> Option<&VariableCellSettings> {
    steps.config(Stage::Minimum).variable_cell.as_ref()
}

/// The structure of a point of the variables of `map` (a structure stays as it is).
fn structure_of(map: &VariableCell, point: &Configuration) -> MoptResult<Configuration> {
    match point {
        Configuration::Variables(p) => Ok(Configuration::Structure(map.system(&p.values)?)),
        Configuration::Structure(_) => Ok(point.clone()),
    }
}

/// The frames of points of the variables of a cell, written as their structures.
struct CellFrames<'a> {
    map: &'a VariableCell,
    inner: &'a mut dyn FrameSink,
}

impl FrameSink for CellFrames<'_> {
    fn push(&mut self, system: &Configuration, energy: f64, label: &str) -> MoptResult<()> {
        let s = structure_of(self.map, system)?;
        self.inner.push(&s, energy, label)
    }
}

/// Where an optimization on a variable cell ended.
pub(in crate::cli) struct CellRun {
    pub(in crate::cli) system: MolecularSystem,
    /// The enthalpy there without the bias potentials.
    pub(in crate::cli) enthalpy: f64,
    pub(in crate::cli) converged: bool,
    pub(in crate::cli) interrupted: bool,
    pub(in crate::cli) path: Vec<PathPoint>,
    /// The enthalpy of every state without the bias potentials at the points of `path`.
    pub(in crate::cli) state_enthalpies: Vec<Vec<f64>>,
    /// The iterations it took.
    pub(in crate::cli) iterations: usize,
}

/// What the Hessian of the variables of a cell says at a structure.
struct CellCheck {
    /// The negative curvatures (of the atoms and the cell together).
    negative: usize,
    /// Modes left by the index check, with coupled atom and strain directions.
    surplus: Vec<PointMode>,
    /// The step of the mode kill out of the surplus ones (in the variables of `map`).
    step: Option<Col<f64>>,
    /// Coupled unstable directions that the order-constrained optimization must follow.
    retained: Vec<Col<f64>>,
    /// The vibrations of the atoms in the cell (cm^-1).
    wavenumbers: Vec<f64>,
    /// The Hessian in the coupled optimization coordinates.
    hessian: Mat<f64>,
    /// The gradient in the same coordinates.
    gradient: Col<f64>,
    /// The variables (the structure is their reference).
    map: VariableCell,
}

impl CommandJobs<'_> {
    /// Optimize `start` on the variable cell `cell` with the settings `o` to a saddle point of
    /// order `order` (0: a minimum), with the bias potentials `terms`, following `follow`
    /// (directions of the atoms) where given.
    #[allow(clippy::too_many_arguments)]
    fn cell_run(
        &self,
        calc: &dyn Calculator,
        start: &MolecularSystem,
        terms: &[(String, Arc<dyn PesTerm>)],
        o: &OptmainConfig,
        order: usize,
        follow: &[Col<f64>],
        label: &str,
        frames: &mut dyn FrameSink,
        cell: &VariableCellSettings,
    ) -> MoptResult<CellRun> {
        let states = cell_states(&o.model_function, o.excited_state)?;
        self.cell_run_with(
            calc,
            start,
            terms,
            o,
            states,
            order,
            follow,
            label,
            frames,
            cell.coordinates(start)?,
        )
    }

    /// [`Self::cell_run`] on the surface of `states` (two states combined by a model function,
    /// with the bias potentials as [`CellStates`] places them; `None`: the electronic state of
    /// `o`) in place of the states of `o`.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::cli) fn cell_run_with(
        &self,
        calc: &dyn Calculator,
        start: &MolecularSystem,
        terms: &[(String, Arc<dyn PesTerm>)],
        o: &OptmainConfig,
        states: Option<CellStates>,
        order: usize,
        follow: &[Col<f64>],
        label: &str,
        frames: &mut dyn FrameSink,
        map: VariableCell,
    ) -> MoptResult<CellRun> {
        let objective = CellObjective::new(
            calc,
            terms.to_vec(),
            map.clone(),
            start.clone(),
            o.excited_state,
        )
        .with_states(states);
        let s = &o.settings;
        let settings = VectorSettings {
            methods: if follow.is_empty() {
                s.methods.clone()
            } else {
                vec![super::FOLLOWING_METHOD.to_string()]
            },
            saddle_order: order,
            max_iterations: s.max_iterations,
            criteria: s.criteria,
            trust: s.trust,
            exact_hessian_interval: s.exact_hessian_interval,
            ccqn: None,
            mode_kill: None,
            project_updates: s.project_updates,
        };
        let mut opt = VectorOptimizer::new(settings, map.variables(start)?)?;
        // The rigid motions of the whole structure, those taken out of its exact Hessians
        // ([`crate::workflow::irc_cell::cell_hessian`]), out of the updated ones as well.
        let modes = map.clone();
        opt.set_rigid_modes(Box::new(move |x| modes.rigid_modes(x)));
        if !follow.is_empty() {
            // Atomic directions hold the cell fixed. Coupled directions already span the map.
            let padded: Vec<Col<f64>> = follow
                .iter()
                .map(|d| Col::from_fn(map.dimension(), |i| if i < d.nrows() { d[i] } else { 0.0 }))
                .collect();
            opt.follow_directions(&padded)?;
        }
        let mut path = Vec::new();
        let mut state_enthalpies = Vec::new();
        let stop = self.stop.clone();
        let outcome = run_vector_optimization_until(
            &mut opt,
            &objective,
            &mut |r| {
                let enthalpy = objective.unbiased_at(&r.x).unwrap_or(r.energy);
                state_enthalpies.push(objective.state_enthalpies_at(&r.x).unwrap_or_default());
                let point = Configuration::Structure(map.system(&r.x)?);
                frames.push(
                    &point,
                    enthalpy,
                    &format!("{label} iteration {}", r.iteration),
                )?;
                path.push(PathPoint {
                    system: point,
                    energy: enthalpy,
                    biased_energy: r.energy,
                });
                Ok(())
            },
            &mut || stop.check(),
        )?;
        let converged = outcome.termination == Some(Termination::Converged);
        let enthalpy = objective.unbiased_at(&outcome.x).unwrap_or(outcome.value);
        self.log.debug(&format!(
            "{label}: {} iterations on the variable cell, H = {enthalpy:.10} Hartree, {}",
            outcome.iterations,
            if converged {
                "converged"
            } else if outcome.termination.is_none() {
                "stopped"
            } else {
                "not converged"
            }
        ));
        Ok(CellRun {
            system: map.system(&outcome.x)?,
            enthalpy,
            converged,
            interrupted: outcome.termination.is_none() && self.stop.stopped(),
            path,
            state_enthalpies,
            iterations: outcome.iterations,
        })
    }

    /// The Hessian of the variables of a cell made from `sys` on the surface of the states of
    /// `o` (its electronic state, or the two states of its model function), and what it says
    /// about saddle points of order `order`.
    fn cell_check(
        &self,
        calc: &dyn Calculator,
        sys: &MolecularSystem,
        cell: &VariableCellSettings,
        o: &OptmainConfig,
        order: usize,
    ) -> MoptResult<CellCheck> {
        let map = cell.coordinates(sys)?;
        let objective =
            CellObjective::new(calc, Vec::new(), map.clone(), sys.clone(), o.excited_state)
                .with_states(cell_states(&o.model_function, o.excited_state)?);
        let x = map.variables(sys)?;
        let v = objective.evaluate(&x, true)?;
        let hessian = v
            .hessian
            .ok_or_else(|| MoptError::calc("the variable cell gave no Hessian"))?;
        let mut mode_kill = mapper_mode_kill(o.settings.mode_kill);
        mode_kill.threshold =
            mapper_mode_kill(self.config(Stage::Minimum).settings.mode_kill).threshold;
        // The one criterion of an imaginary mode, with the masses of the variables of the
        // atoms and the cell (those of the IRC on the cell); the rigid motions are out of the
        // Hessian already.
        let masses = map.masses()?;
        let (verdict, retained) = examine_retaining(
            &hessian,
            &v.gradient,
            order,
            &mode_kill,
            Some(&masses),
            None,
        )?;
        let (negative, step, surplus) = match verdict {
            ModeKillVerdict::Accept { negative } => (negative, None, Vec::new()),
            ModeKillVerdict::Displace {
                step,
                negative,
                left,
                ..
            } => (
                negative,
                Some(step),
                crate::cli::mapper_minima::index_modes(left),
            ),
        };
        let n = sys.dof();
        let atoms = Mat::from_fn(n, n, |i, j| hessian[(i, j)]);
        let wavenumbers = vibrations(sys, &atoms, calc.is_rigid_body_invariant())?;
        Ok(CellCheck {
            negative,
            step,
            retained,
            surplus,
            wavenumbers,
            hessian,
            gradient: v.gradient,
            map,
        })
    }

    /// [`crate::workflow::mapper::MapperJobs::optimize`] of a structure on the variable cell
    /// `cell` (`steps`: the steps of the state).
    pub(super) fn optimize_cell(
        &self,
        calc: &dyn Calculator,
        start: &MolecularSystem,
        request: &OptimizeRequest,
        frames: &mut dyn FrameSink,
        steps: &StepConfigs,
        cell: &VariableCellSettings,
    ) -> MoptResult<Optimized> {
        let o = steps.config(request.stage);
        let verify = matches!(request.stage, Stage::Minimum | Stage::IrcEnd) && steps.verify_minima;
        let mut run = self.cell_run(
            calc,
            start,
            &request.terms,
            o,
            o.settings.saddle_order,
            &[],
            &request.label,
            frames,
            cell,
        )?;
        let mut converged = run.converged;
        let mut wavenumbers = None;
        let mut index_wavenumbers = None;
        if verify && converged {
            let mut check = self.cell_check(calc, &run.system, cell, o, 0)?;
            // Apart in its cell: a dissociation channel, kept with its imaginary modes and not
            // displaced along them.
            if let Some(why) = self.apart_in_cell(&run.system) {
                if check.negative > 0 {
                    self.log.info(&format!(
                        "{}: {} imaginary mode(s) of the atoms and the cell remain, but {why}: \
                         kept as a dissociation channel",
                        request.label, check.negative
                    ));
                }
                return Ok(Optimized {
                    index_wavenumbers: Some(
                        check.surplus.iter().filter_map(|m| m.wavenumber).collect(),
                    ),
                    energy: run.enthalpy,
                    system: Configuration::Structure(run.system),
                    converged,
                    interrupted: run.interrupted,
                    path: run.path,
                    wavenumbers: Some(check.wavenumbers),
                });
            }
            if let Some(step) = check.step.take().filter(|_| o.settings.mode_kill.is_some()) {
                // Leave the negative curvature once along the step of the mode kill.
                let x = check.map.variables(&run.system)? + &step;
                let displaced = check.map.system(&x)?;
                let label = format!("{} after the mode kill", request.label);
                let mut path = std::mem::take(&mut run.path);
                let mut state_enthalpies = std::mem::take(&mut run.state_enthalpies);
                let iterations = run.iterations;
                let again =
                    self.cell_run(calc, &displaced, &[], o, 0, &[], &label, frames, cell)?;
                path.extend(again.path.iter().cloned());
                state_enthalpies.extend(again.state_enthalpies.iter().cloned());
                run = CellRun {
                    path,
                    state_enthalpies,
                    iterations: iterations + again.iterations,
                    ..again
                };
                converged = run.converged;
                if converged {
                    check = self.cell_check(calc, &run.system, cell, o, 0)?;
                }
            }
            if converged {
                if check.negative == 0 || request.stage == Stage::IrcEnd {
                    index_wavenumbers =
                        Some(check.surplus.iter().filter_map(|m| m.wavenumber).collect());
                    wavenumbers = Some(check.wavenumbers);
                } else {
                    converged = false;
                    self.log.warn(&format!(
                        "{}: the converged structure is not a minimum of the enthalpy: {} \
                         negative curvature(s) of the atoms and the cell remain",
                        request.label, check.negative
                    ));
                }
            }
        }
        Ok(Optimized {
            index_wavenumbers,
            energy: run.enthalpy,
            system: Configuration::Structure(run.system),
            converged,
            interrupted: run.interrupted,
            path: run.path,
            wavenumbers,
        })
    }

    /// Leave excess negative curvature in the same atom/strain space used to measure the index.
    pub(super) fn escape_cell_saddle(
        &self,
        calc: &dyn Calculator,
        start: &MolecularSystem,
        cell: &VariableCellSettings,
        label: &str,
        frames: &mut dyn FrameSink,
    ) -> MoptResult<Option<OrderedSaddle>> {
        let o = self.config(Stage::Saddle);
        let check = self.cell_check(calc, start, cell, o, 1)?;
        let Some(step) = check.step else {
            return Ok(None);
        };
        let origin = check.map.variables(start)?;
        let settings = mapper_mode_kill(o.settings.mode_kill);
        let objective = CellObjective::new(
            calc,
            Vec::new(),
            check.map.clone(),
            start.clone(),
            o.excited_state,
        )
        .with_states(cell_states(&o.model_function, o.excited_state)?);
        let frame = crate::path::hisd::SearchFrame::variable_cell(check.map.clone())?;
        for attempt in 0..settings.max_rounds {
            if self.stop.check() {
                break;
            }
            let swap = check.negative == 2 && attempt % 4 >= 2;
            let branches = if check.negative == 2 { 4 } else { 2 };
            let scale = 2.0_f64.powi((attempt / branches) as i32)
                * if attempt % 2 == 0 { 1.0 } else { -1.0 };
            let (displacement, directions) = if swap {
                (
                    &check.retained[0] * (scale * settings.displacement),
                    std::slice::from_ref(&step),
                )
            } else {
                (&step * scale, check.retained.as_slice())
            };
            let tag = format!("{label} attempt {}", attempt + 1);
            let seed = match check.map.system(&(&origin + &displacement)) {
                Ok(seed) => seed,
                Err(error) => {
                    self.log
                        .warn(&format!("{tag}: invalid displaced cell: {error}"));
                    continue;
                }
            };
            let magnitude = crate::math::linalg::col_norm(&displacement);
            self.log.info(&format!("{tag}: leaving index {} with coupled atom/cell displacement {magnitude:.6} (weighted cell coordinates)", check.negative));
            let point =
                Configuration::variables(check.map.variables(&seed)?, Arc::new(check.map.names()))?;
            let dynamics = super::super::mapper_searches::run_hisd(
                &VariableSurface(&objective),
                &point,
                1,
                directions,
                &format!("{tag} HiSD"),
                &mut CellFrames {
                    map: &check.map,
                    inner: frames,
                },
                &self.stop,
                &o.settings.criteria,
                frame.clone(),
            )?;
            self.log.info(&format!("{tag}: HiSD {}", dynamics.end));
            if dynamics.interrupted {
                return Ok(None);
            }
            if !dynamics.settled {
                continue;
            }
            let run = self.cell_run_with(
                calc,
                &check.map.system(&dynamics.point.flat())?,
                &[],
                o,
                cell_states(&o.model_function, o.excited_state)?,
                1,
                &dynamics.directions,
                &tag,
                frames,
                check.map.clone(),
            )?;
            if run.interrupted {
                return Ok(None);
            }
            if !run.converged {
                continue;
            }
            let delta = check.map.variables(&run.system)? - &origin;
            let max = crate::math::linalg::col_max_abs(&delta);
            let rms = crate::math::linalg::col_norm(&delta) / (delta.nrows() as f64).sqrt();
            let modes = self.cell_modes(calc, &run.system, false, cell)?;
            let returned = max <= o.settings.criteria.max_displacement.value()
                && rms <= o.settings.criteria.rms_displacement.value();
            self.log.info(&format!("{tag}: verified index {}, displacement from initial saddle (max {max:.6}, RMS {rms:.6}); {}",
                modes.negative, if returned { "returned within geometry tolerance" } else { "distinct geometry" }));
            if !returned && modes.negative == 1 {
                return Ok(Some(OrderedSaddle {
                    optimized: Optimized {
                        index_wavenumbers: None,
                        energy: run.enthalpy,
                        system: Configuration::Structure(run.system),
                        converged: true,
                        interrupted: false,
                        path: run.path,
                        wavenumbers: None,
                    },
                    modes: Some(modes),
                }));
            }
        }
        Ok(None)
    }

    /// [`crate::workflow::mapper::MapperJobs::saddle`] of a structure on the variable cell.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn saddle_cell(
        &self,
        calc: &dyn Calculator,
        start: &MolecularSystem,
        order: usize,
        follow: &[Col<f64>],
        label: &str,
        frames: &mut dyn FrameSink,
        cell: &VariableCellSettings,
    ) -> MoptResult<OrderedSaddle> {
        let o = self.config(Stage::Saddle);
        let run = self.cell_run(calc, start, &[], o, order, follow, label, frames, cell)?;
        let modes = if run.converged {
            Some(self.cell_modes(calc, &run.system, false, cell)?)
        } else {
            None
        };
        Ok(OrderedSaddle {
            optimized: Optimized {
                index_wavenumbers: None,
                energy: run.enthalpy,
                system: Configuration::Structure(run.system),
                converged: run.converged,
                interrupted: run.interrupted,
                path: run.path,
                wavenumbers: None,
            },
            modes,
        })
    }

    /// The modes of a structure on the variable cell: those of its atoms in its cell (the
    /// directions the mapper displaces it along), the number of negative curvatures of the
    /// atoms and the cell together, and the vibrations of the atoms.
    pub(super) fn cell_modes(
        &self,
        calc: &dyn Calculator,
        sys: &MolecularSystem,
        across: bool,
        cell: &VariableCellSettings,
    ) -> MoptResult<PointModes> {
        let o = self.config(Stage::Minimum);
        let check = self.cell_check(calc, sys, cell, o, 0)?;
        let masses = check.map.masses()?;
        let inverse: Vec<_> = masses.iter().map(|m| 1.0 / m.sqrt()).collect();
        let n = masses.len();
        let weighted = Mat::from_fn(n, n, |i, j| check.hessian[(i, j)] * inverse[i] * inverse[j]);
        let gradient = Col::from_fn(n, |i| check.gradient[i] * inverse[i]);
        let threshold = mapper_mode_kill(o.settings.mode_kill).threshold;
        let mut list = crate::cli::mapper_minima::coordinate_modes(&weighted, &gradient, across)?;
        list.retain(|m| m.curvature.abs() > threshold);
        for mode in &mut list {
            let direction = Col::from_fn(n, |i| mode.displacement[i] * inverse[i]);
            let norm = crate::math::linalg::col_norm(&direction);
            mode.displacement = direction * (1.0 / norm);
        }
        Ok(PointModes {
            unstable: check.surplus,
            modes: list,
            negative: check.negative,
            wavenumbers: Some(check.wavenumbers),
        })
    }

    /// Follow modes in the coordinate and mass frame of a variable-cell enthalpy surface.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn cell_hisd(
        &self,
        calc: &dyn Calculator,
        start: &MolecularSystem,
        cell: &VariableCellSettings,
        order: usize,
        directions: &[Col<f64>],
        label: &str,
        frames: &mut dyn FrameSink,
    ) -> MoptResult<crate::workflow::mapper::HisdOutcome> {
        let map = cell.coordinates(start)?;
        let o = self.config(Stage::Saddle);
        let objective = CellObjective::new(
            calc,
            Vec::new(),
            map.clone(),
            start.clone(),
            o.excited_state,
        )
        .with_states(cell_states(&o.model_function, o.excited_state)?);
        let point = Configuration::variables(map.variables(start)?, Arc::new(map.names()))?;
        let mut run = super::super::mapper_searches::run_hisd(
            &VariableSurface(&objective),
            &point,
            order,
            directions,
            label,
            &mut CellFrames {
                map: &map,
                inner: frames,
            },
            &self.stop,
            &o.settings.criteria,
            crate::path::hisd::SearchFrame::variable_cell(map.clone())?,
        )?;
        run.point = structure_of(&map, &run.point)?;
        Ok(run)
    }

    /// The enthalpy `E + P V` of a structure in its cell.
    pub(super) fn cell_enthalpy(
        &self,
        calc: &dyn Calculator,
        sys: &MolecularSystem,
        cell: &VariableCellSettings,
    ) -> MoptResult<f64> {
        let map = cell.coordinates(sys)?;
        let o = self.config(Stage::Minimum);
        let objective =
            CellObjective::new(calc, Vec::new(), map.clone(), sys.clone(), o.excited_state)
                .with_states(cell_states(&o.model_function, o.excited_state)?);
        Ok(objective.value(&map.variables(sys)?)?.enthalpy)
    }

    /// The IRC of the atoms and the cell from `ts` (`steps`: the steps of the state; `meta`: a
    /// start that is no transition state is followed down as a meta-IRC).
    pub(super) fn irc_cell(
        &self,
        calc: &dyn Calculator,
        ts: &MolecularSystem,
        frames: &mut dyn FrameSink,
        steps: &StepConfigs,
        meta: bool,
        cell: &VariableCellSettings,
    ) -> MoptResult<IrcOutcome> {
        let settings = Self::irc_settings(steps);
        let map = cell.coordinates(ts)?;
        let start = Configuration::Structure(ts.clone());
        let mut collector = IrcCollector {
            start: &start,
            frames,
            sides: Vec::new(),
            kind: None,
            imaginary: 0,
            wavenumber: None,
            rigid: calc.is_rigid_body_invariant(),
            wavenumbers: None,
            bifurcation_threshold_cm1: steps.bifurcation_threshold_cm1,
            clear: Vec::new(),
            refused: false,
            meta,
            start_gradient: None,
            cell: Some(&map),
        };
        let mut watch = StopWatch(self.stop.clone());
        let states = cell_states(&steps.minimum.model_function, steps.minimum.excited_state)?;
        let end = crate::cli::irc::run_cell_observed(
            &settings,
            &map,
            calc,
            &[],
            ts,
            steps.minimum.excited_state,
            states.as_ref(),
            &self.log,
            &mut [&mut collector, &mut watch],
        )?;
        say_not_stationary(&collector, &settings, &self.log);
        Ok(irc_outcome(collector, end.is_none()))
    }

    /// [`crate::workflow::mapper::MapperJobs::pair_search`] on the variable cell: the search
    /// runs on the variables made from the first structure, every image with its cell.
    pub(super) fn pair_search_cell(
        &self,
        calc: &dyn Calculator,
        request: &PairSearchRequest,
        frames: &mut dyn FrameSink,
        cell: &VariableCellSettings,
    ) -> MoptResult<PairSearchOutcome> {
        let stop = self.stop.clone();
        if stop.check() {
            return Ok(PairSearchOutcome::stopped());
        }
        let structures = request
            .structures
            .iter()
            .map(|p| p.require_structure(&request.label).cloned())
            .collect::<MoptResult<Vec<_>>>()?;
        let first = structures
            .first()
            .ok_or_else(|| MoptError::config(format!("{}: no structure", request.label)))?;
        let map = cell.coordinates(first)?;
        let o = self.config(Stage::Minimum);
        let objective = CellObjective::new(
            calc,
            Vec::new(),
            map.clone(),
            first.clone(),
            o.excited_state,
        )
        .with_states(cell_states(&o.model_function, o.excited_state)?);
        let points = structures
            .iter()
            .map(|s| map.variables(s))
            .collect::<MoptResult<Vec<_>>>()?;
        let template = Configuration::variables(points[0].clone(), Arc::new(map.names()))?;
        let input = PairInput {
            points,
            structures: false,
            ..PairInput::default()
        };
        let mut cell_frames = CellFrames {
            map: &map,
            inner: frames,
        };
        let mut outcome = self.run_search(
            &VariableSurface(&objective),
            &input,
            request,
            &mut cell_frames,
            &template,
            Some(&map),
        )?;
        for e in &mut outcome.estimates {
            e.system = structure_of(&map, &e.system)?;
        }
        for p in &mut outcome.path {
            p.system = structure_of(&map, &p.system)?;
        }
        Ok(outcome)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn points_of_the_variables_are_written_as_structures() {
        let sys = crate::io::xyz::parse_xyz_str(
            "2\nLattice=\"3.0 0.0 0.0 0.0 3.0 0.0 0.0 0.0 3.0\"\nAr 0.0 0.0 0.0\nAr 1.5 1.5 1.5\n",
            crate::domain::SpinCharge::default(),
            None,
        )
        .unwrap();
        let cell = VariableCellSettings {
            pressure_gpa: 0.0,
            weight: 1.0,
            forces: Vec::new(),
        };
        let map = cell.coordinates(&sys).unwrap();
        let mut x = map.variables(&sys).unwrap();
        // The atoms of the reference are at their Cartesian coordinates.
        let flat = sys.coords_flat_bohr();
        for i in 0..flat.nrows() {
            assert!((x[i] - flat[i]).abs() < 1e-12);
        }
        // A longer lattice: the structure of the point has the longer cell.
        let n = sys.dof();
        x[n] *= 1.1;
        let point = Configuration::variables(x, Arc::new(map.names())).unwrap();
        let back = structure_of(&map, &point).unwrap();
        let s = back.structure().unwrap();
        let before = sys.active_cell().unwrap().volume_bohr3();
        let after = s.active_cell().unwrap().volume_bohr3();
        assert!(after > before * 1.05, "{before} {after}");
    }
}
