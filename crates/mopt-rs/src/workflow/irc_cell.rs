//! The IRC of the atoms and the cell together (`ircmain -vc`, `optmain -vc -irc`): the path
//! of steepest descent of the enthalpy in the variables of the variable-cell optimization
//! (`opt::cell`: quasi-Cartesian coordinates of the atoms and scaled lattice vectors),
//! mass-weighted with the masses of [`crate::opt::cell::VariableCell::masses`].
//!
//! The surface ([`CellSurface`]) is the enthalpy `E + P V` with the bias potentials and its
//! gradient in the variables, from the energy, the gradient, and the stress of the
//! calculator (the stress by differences where the calculator has none); Hessians come from
//! differences of that gradient, with the translations and rotations of the whole structure
//! projected out. [`CellStructures`] writes the structures of the path with their cells
//! (extended XYZ), next to the files of the variables that the IRC writes.
//!
//! The same surface as an objective of plain variables ([`CellObjective`]) is what the
//! optimizers, the transition-state searches, and the two-point searches of a variable cell
//! run on (the mapper with `-vc`); both take their points from [`cell_value`] and their
//! Hessians from [`cell_hessian`].
//!
//! Two electronic states combined by a model function (`-mf`: crossing points, conical
//! intersections, penalty functions) are combined in the variables of the cell
//! ([`CellStates`]): the enthalpy of every state and its gradient in the variables, from the
//! energy, the gradient, and the stress of that state, are what the combiner takes, as it
//! takes the energies and Cartesian gradients of the states in a fixed cell.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::path::{Path, PathBuf};
use std::sync::Arc;

use faer::{Col, Mat};

use super::irc::{IrcObserver, IrcSurface};
use crate::core_infra::log::Logger;
use crate::domain::{EvalOutput, EvalRequest, MolecularSystem, Quantity};
use crate::error::{MoptError, MoptResult, ResultExt};
use crate::io::xyz::format_xyz_system;
use crate::irc::types::Branch;
use crate::irc::{IrcEvent, IrcRequest, PointValue};
use crate::math::linalg::col_norm;
use crate::opt::cell::{numerical_hessian, CellSymmetries, VariableCell, DEFAULT_HESSIAN_STEP};
use crate::pes::calculator::{evaluate_checked, said, Calculator};
use crate::pes::combine::{PesExpr, StateCombiner, StateValues, TaskSpec};
use crate::pes::objective::{Objective, ObjectiveValue};
use crate::pes::stress::{numerical_stress_of_state, term_stress, Stress, DEFAULT_STRAIN_STEP};
use crate::pes::term::{PesTerm, TermContext};

/// Two electronic states of a variable cell combined by a model function (`-mf`).
///
/// The combiner takes the enthalpy `E_i + P V` of every state and its gradient in the
/// variables of the cell (from the energy, the gradient, and the stress of that state), as it
/// takes the energies and the Cartesian gradients of the states in a fixed cell: a function of
/// the energies weighs the stresses of the states by its derivatives as it weighs their
/// gradients, and a gradient projection projects the components of the lattice with those of
/// the atoms (its difference gradient and its branching plane span all the variables). The
/// pressure term is the same in both states, so the gap is that of the energies, and the
/// mean is the mean enthalpy. Bias potentials are added on top of the combined surface, as in
/// a fixed cell, or with the energy of every state ([`Self::with_terms_in_states`]).
pub struct CellStates {
    labels: Vec<String>,
    states: Vec<TaskSpec>,
    combiner: std::sync::Mutex<Box<dyn StateCombiner>>,
    terms_in_states: bool,
}

impl CellStates {
    /// The states of `surface` (`cli::optmain::state_surface`): `None` for no surface or one
    /// state (the electronic state of [`CellParts::excited_state`] then); refused for a
    /// surface that is not a combination of states of the whole structure.
    pub fn of(surface: Option<PesExpr>) -> MoptResult<Option<Self>> {
        match surface {
            None | Some(PesExpr::Task(_)) => Ok(None),
            Some(PesExpr::Combine {
                combiner,
                labels,
                inputs,
            }) => {
                let states = inputs
                    .into_iter()
                    .map(|input| match input {
                        PesExpr::Task(spec) => Ok(spec),
                        _ => Err(MoptError::unsupported(
                            "a variable cell combines states of the whole structure (-mf), not \
                             other surfaces",
                        )),
                    })
                    .collect::<MoptResult<Vec<TaskSpec>>>()?;
                Ok(Some(Self {
                    labels,
                    states,
                    combiner: std::sync::Mutex::new(combiner),
                    terms_in_states: false,
                }))
            }
            Some(_) => Err(MoptError::unsupported(
                "a variable cell takes one electronic state or two combined by -mf",
            )),
        }
    }

    /// The bias potentials stand with the energy of every state (`E_k + bias` in place of
    /// `E_k`, their stress with that of the state) instead of on top of the combined surface:
    /// on a gradient projection they then cancel in the gap and act within the seam, as on the
    /// AFIR paths of the mapper towards crossing points.
    #[must_use]
    pub fn with_terms_in_states(mut self) -> Self {
        self.terms_in_states = true;
        self
    }

    /// The name of the model function.
    pub fn name(&self) -> String {
        self.combiner
            .lock()
            .map_or_else(|_| "model function".to_string(), |c| c.name().to_string())
    }

    /// The labels of the states.
    pub fn labels(&self) -> &[String] {
        &self.labels
    }

    /// The evaluations of the states at `sys`, with their stresses: (label, structure in the
    /// charge and multiplicity of the state, request with its electronic state).
    fn evaluations(
        &self,
        sys: &MolecularSystem,
    ) -> MoptResult<Vec<(String, MolecularSystem, EvalRequest)>> {
        let mut out = Vec::with_capacity(self.states.len());
        for spec in &self.states {
            for task in PesExpr::Task(spec.clone()).tasks(sys, &EvalRequest::ENERGY_GRAD_STRESS)? {
                out.push((task.label, task.system, task.request));
            }
        }
        Ok(out)
    }

    fn combine(
        &self,
        sys: &MolecularSystem,
        values: &StateValues,
    ) -> MoptResult<crate::pes::combine::Combined> {
        let combiner = self
            .combiner
            .lock()
            .map_err(|_| MoptError::calc("the states of the variable cell are not available"))?;
        combiner.combine(sys, values)
    }

    /// Record the states of an accepted point (`values`, as [`CellValue::states`] gives
    /// them), for a combiner with a history (the branching plane of a conical intersection).
    pub fn commit(&self, sys: &MolecularSystem, values: &StateValues) {
        if let Ok(mut combiner) = self.combiner.lock() {
            combiner.commit(sys, values);
        }
    }
}

/// The enthalpy of the atoms and the cell of a structure as a surface of the variables of
/// `map`.
pub struct CellSurface<'a> {
    /// The calculator.
    pub calc: &'a dyn Calculator,
    /// Bias potentials with their labels.
    pub terms: &'a [(String, Arc<dyn PesTerm>)],
    /// The variables.
    pub map: &'a VariableCell,
    /// The starting structure.
    pub start: &'a MolecularSystem,
    /// Electronic state (0: the ground state).
    pub excited_state: usize,
    /// Two states combined by a model function (`-mf`), in place of `excited_state`.
    pub states: Option<&'a CellStates>,
    /// Where the calculator's messages go.
    pub log: Option<&'a Logger>,
    /// The iteration the bias potentials see at every point of the path: the surface of the
    /// iteration of the optimization that found the start (a ramp does not start over).
    pub iteration: usize,
}

impl CellSurface<'_> {
    fn parts(&self) -> CellParts<'_> {
        CellParts {
            calc: self.calc,
            terms: self.terms,
            map: self.map,
            start: self.start,
            excited_state: self.excited_state,
            states: self.states,
            log: self.log,
        }
    }

    /// The energy of the calculator, the enthalpy with the bias potentials, and its
    /// gradient in the variables at `x`; the states of a point of the path are recorded
    /// ([`CellStates::commit`]).
    fn point(&self, x: &Col<f64>, step: usize, of: &str) -> MoptResult<(f64, f64, Col<f64>)> {
        let v = cell_value(&self.parts(), x, step, of)?;
        if let (Some(states), Some(values)) = (self.states, &v.states) {
            states.commit(&v.system, values);
        }
        Ok((v.energy, v.biased, v.slope))
    }
}

/// What the surface of a variable cell is made of: the calculator, the bias potentials, the
/// variables, the starting structure, the electronic state, and where the calculator's
/// messages go.
pub struct CellParts<'a> {
    /// The calculator.
    pub calc: &'a dyn Calculator,
    /// Bias potentials with their labels.
    pub terms: &'a [(String, Arc<dyn PesTerm>)],
    /// The variables.
    pub map: &'a VariableCell,
    /// The starting structure (of the bias potentials).
    pub start: &'a MolecularSystem,
    /// Electronic state (0: the ground state).
    pub excited_state: usize,
    /// Two states combined by a model function (`-mf`), in place of `excited_state`.
    pub states: Option<&'a CellStates>,
    /// Where the calculator's messages go.
    pub log: Option<&'a Logger>,
}

/// The output of the calculator for one state at a point of a variable cell.
#[derive(Debug, Clone)]
pub struct StateOutput {
    /// The label of the state (`pes` for the one state of a surface).
    pub label: String,
    /// The multiplicity it was evaluated in.
    pub multiplicity: u32,
    /// What the calculator gave.
    pub output: EvalOutput,
}

/// A point of the variables of a variable cell.
#[derive(Debug, Clone)]
pub struct CellValue {
    /// The structure with its cell.
    pub system: MolecularSystem,
    /// The energy of the calculator (with two states: the combined enthalpy less `P V`, the
    /// mean energy for a gradient projection).
    pub energy: f64,
    /// The enthalpy `E + P V` without the bias potentials (with two states: combined).
    pub enthalpy: f64,
    /// The enthalpy with the bias potentials.
    pub biased: f64,
    /// The gradient of the biased enthalpy in the variables.
    pub slope: Col<f64>,
    /// The Cartesian gradient of the calculator (with two states: the mean of theirs).
    pub raw_gradient: Col<f64>,
    /// The Cartesian gradient with the bias potentials.
    pub gradient: Col<f64>,
    /// The stress with the bias potentials (with two states: the mean of theirs, for the
    /// log; the cell moves on the combined gradient in the variables).
    pub stress: Stress,
    /// The energies of the parts: the calculator (`pes`, or every state by its label) and the
    /// bias potentials by theirs.
    pub components: Vec<(String, f64)>,
    /// The outputs of the calculator: one (`pes`), or one per state by its label.
    pub outputs: Vec<StateOutput>,
    /// With two states: their enthalpies and gradients in the variables as the combiner took
    /// them ([`CellStates::commit`]).
    pub states: Option<StateValues>,
    /// With two states: the gap of their energies `E_1 - E_2` (Hartree) and the norm of the
    /// difference of their Cartesian gradients (Hartree/Bohr).
    pub crossing: Option<(f64, f64)>,
    /// The enthalpy `E + P V` of every state without the bias potentials (one for a surface of
    /// one state).
    pub state_enthalpies: Vec<f64>,
}

/// The surface of `parts` at the variables `x` (`step`: the iteration the bias potentials
/// see; `of`: what the point is, for the calculator's messages).
pub fn cell_value(
    parts: &CellParts<'_>,
    x: &Col<f64>,
    step: usize,
    of: &str,
) -> MoptResult<CellValue> {
    let sys = parts.map.system(x)?;
    // The states: one (the electronic state of the parts) or the two of the model function,
    // each with its charge, multiplicity, and electronic state, and its stress.
    let evaluations = match parts.states {
        Some(states) => states.evaluations(&sys)?,
        None => vec![(
            "pes".to_string(),
            sys.clone(),
            EvalRequest {
                excited_state: parts.excited_state,
                ..EvalRequest::ENERGY_GRAD_STRESS
            },
        )],
    };
    let mut outputs: Vec<StateOutput> = Vec::with_capacity(evaluations.len());
    let mut stresses: Vec<Stress> = Vec::with_capacity(evaluations.len());
    for (label, system, request) in evaluations {
        let out = evaluate_checked(parts.calc, &system, &request)?;
        if let Some(log) = parts.log {
            let what = if parts.states.is_some() {
                format!("{of} {label}")
            } else {
                of.to_string()
            };
            for line in said(parts.calc, &what, &out) {
                log.info(&line);
            }
        }
        stresses.push(match out.stress {
            Some(stress) => stress,
            None => numerical_stress_of_state(
                parts.calc,
                &system,
                request.excited_state,
                DEFAULT_STRAIN_STEP,
            )?,
        });
        outputs.push(StateOutput {
            label,
            multiplicity: system.spin_charge.multiplicity,
            output: out,
        });
    }
    let ctx = TermContext::new(step, parts.start, None);
    let volume = sys.active_cell().map_or(0.0, |c| c.volume_bohr3());
    let pv = parts.map.pressure() * volume;
    let Some(states) = parts.states else {
        let (Some(StateOutput { output: out, .. }), Some(&first)) =
            (outputs.first(), stresses.first())
        else {
            return Err(MoptError::calc("the variable cell evaluated no state"));
        };
        let mut stress = first;
        let mut total = out.energy.value();
        let mut gradient = out.gradient_bohr.clone();
        let mut components = vec![("pes".to_string(), out.energy.value())];
        for (label, term) in parts.terms {
            let bias = term.evaluate(&sys, &ctx, false).context(label.clone())?;
            total += bias.energy.value();
            gradient += &bias.gradient_bohr;
            let share = term_stress(term.as_ref(), &bias, &sys, &ctx, DEFAULT_STRAIN_STEP)?;
            for (row, add) in stress.iter_mut().zip(&share) {
                for (value, part) in row.iter_mut().zip(add) {
                    *value += part;
                }
            }
            components.push((label.clone(), bias.energy.value()));
        }
        let (biased, slope) = parts.map.objective(&sys, total, &gradient, &stress)?;
        let energy = out.energy.value();
        let raw_gradient = out.gradient_bohr.clone();
        return Ok(CellValue {
            energy,
            enthalpy: energy + pv,
            biased,
            slope,
            raw_gradient,
            gradient,
            stress,
            components,
            outputs,
            states: None,
            crossing: None,
            state_enthalpies: vec![energy + pv],
            system: sys,
        });
    };
    // The bias potentials: their energy, Cartesian gradient, and stress.
    let n = sys.dof();
    let mut bias_energy = 0.0;
    let mut bias_gradient = Col::<f64>::zeros(n);
    let mut bias_stress: Stress = [[0.0; 3]; 3];
    let mut components: Vec<(String, f64)> = outputs
        .iter()
        .map(|state| (state.label.clone(), state.output.energy.value()))
        .collect();
    for (label, term) in parts.terms {
        let bias = term.evaluate(&sys, &ctx, false).context(label.clone())?;
        bias_energy += bias.energy.value();
        bias_gradient += &bias.gradient_bohr;
        let share = term_stress(term.as_ref(), &bias, &sys, &ctx, DEFAULT_STRAIN_STEP)?;
        for (row, add) in bias_stress.iter_mut().zip(&share) {
            for (value, part) in row.iter_mut().zip(add) {
                *value += part;
            }
        }
        components.push((label.clone(), bias.energy.value()));
    }
    // Two states: their enthalpies and gradients in the variables (with the bias potentials
    // where they stand with the states), combined.
    let inside = states.terms_in_states && !parts.terms.is_empty();
    let mut enthalpies = Vec::with_capacity(outputs.len());
    let mut slopes = Vec::with_capacity(outputs.len());
    for (state, stress) in outputs.iter().zip(&stresses) {
        let out = &state.output;
        let (h, s) = if inside {
            let mut with_bias = *stress;
            for (row, add) in with_bias.iter_mut().zip(&bias_stress) {
                for (value, part) in row.iter_mut().zip(add) {
                    *value += part;
                }
            }
            parts.map.objective(
                &sys,
                out.energy.value() + bias_energy,
                &(&out.gradient_bohr + &bias_gradient),
                &with_bias,
            )?
        } else {
            parts
                .map
                .objective(&sys, out.energy.value(), &out.gradient_bohr, stress)?
        };
        enthalpies.push(h);
        slopes.push(s);
    }
    let values = StateValues {
        energies: enthalpies,
        gradients: slopes,
        hessians: None,
    };
    let combined = states.combine(&sys, &values)?;
    // The bias potentials on top of the combined surface: their gradient in the variables
    // without the pressure term (which the enthalpies of the states hold).
    let mut slope = combined.gradient.clone();
    if !parts.terms.is_empty() && !inside {
        let (_, with_bias) =
            parts
                .map
                .objective(&sys, bias_energy, &bias_gradient, &bias_stress)?;
        let (_, pressure_alone) = parts
            .map
            .objective(&sys, 0.0, &Col::zeros(n), &[[0.0; 3]; 3])?;
        slope += &with_bias;
        slope -= &pressure_alone;
    }
    let count = outputs.len() as f64;
    let mut gradient_sum = Col::<f64>::zeros(n);
    let mut stress: Stress = [[0.0; 3]; 3];
    for (state, s) in outputs.iter().zip(&stresses) {
        gradient_sum += &state.output.gradient_bohr;
        for (row, add) in stress.iter_mut().zip(s) {
            for (value, part) in row.iter_mut().zip(add) {
                *value += part;
            }
        }
    }
    let raw_gradient = gradient_sum * faer::Scale(1.0 / count);
    for (row, add) in stress.iter_mut().zip(&bias_stress) {
        for (value, part) in row.iter_mut().zip(add) {
            *value = *value / count + part;
        }
    }
    let crossing = match &outputs[..] {
        [first, second] => Some((
            first.output.energy.value() - second.output.energy.value(),
            col_norm(&(&first.output.gradient_bohr - &second.output.gradient_bohr)),
        )),
        _ => None,
    };
    let state_enthalpies = outputs
        .iter()
        .map(|state| state.output.energy.value() + pv)
        .collect();
    Ok(CellValue {
        energy: combined.energy - pv,
        enthalpy: combined.energy,
        biased: if inside {
            combined.energy
        } else {
            combined.energy + bias_energy
        },
        slope,
        gradient: &raw_gradient + &bias_gradient,
        raw_gradient,
        stress,
        components,
        outputs,
        states: Some(values),
        crossing,
        state_enthalpies,
        system: sys,
    })
}

/// The analytic second derivatives of the surface of a variable cell at a structure: the
/// Cartesian Hessian of the atoms (with the bias potentials) and, where the calculator and
/// every bias potential give them, the second derivatives along the strain with the
/// Cartesian gradient they go with.
#[derive(Debug, Clone)]
pub struct AnalyticCell {
    /// The Cartesian Hessian.
    pub hessian: Mat<f64>,
    /// The second derivatives along the strain and the Cartesian gradient.
    pub strain: Option<(crate::domain::StrainHessian, Col<f64>)>,
}

impl AnalyticCell {
    /// Add the second derivatives along the strain of a function of the lattice alone (a
    /// force on the cell).
    pub fn add_lattice(&mut self, strain: &[[f64; 9]; 9]) {
        if let Some((sh, _)) = self.strain.as_mut() {
            for (row, add) in sh.strain.iter_mut().zip(strain) {
                for (value, part) in row.iter_mut().zip(add) {
                    *value += part;
                }
            }
        }
    }
}

/// The Hessian in the variables of `map` at `x`, with the translations and rotations of the
/// whole structure projected out (they have no curvature, and away from a stationary point
/// their differences are not zero). With `analytic`, the block of the atoms is its Cartesian
/// Hessian taken to the variables of the atoms ([`VariableCell::atoms_map`]), and the columns
/// of the lattice variables come from its second derivatives along the strain
/// ([`VariableCell::lattice_columns`]) where it has them, from central differences of
/// `gradient_at` otherwise; every column comes from differences without it.
pub fn cell_hessian(
    map: &VariableCell,
    x: &Col<f64>,
    analytic: Option<&AnalyticCell>,
    gradient_at: impl FnMut(&Col<f64>) -> MoptResult<Col<f64>>,
) -> MoptResult<Mat<f64>> {
    cell_hessian_with(map, x, analytic, CellSymmetries::ALL, gradient_at)
}

/// [`cell_hessian`] with only the rigid motions of `symmetries` projected out: with frozen
/// atoms the translations of the atoms are no symmetry of the constrained problem (the
/// rotations of the lattice, which carry the atoms with their fractional coordinates, still
/// are), and on a surface fixed in space neither is.
pub fn cell_hessian_with(
    map: &VariableCell,
    x: &Col<f64>,
    analytic: Option<&AnalyticCell>,
    symmetries: CellSymmetries,
    mut gradient_at: impl FnMut(&Col<f64>) -> MoptResult<Col<f64>>,
) -> MoptResult<Mat<f64>> {
    let dim = x.nrows();
    let raw = match analytic {
        None => numerical_hessian(x, DEFAULT_HESSIAN_STEP, gradient_at)?,
        Some(analytic) => {
            let h = &analytic.hessian;
            let sys = map.system(x)?;
            let cell = sys.active_cell().ok_or_else(|| {
                crate::error::MoptError::config("the structure has no periodic cell")
            })?;
            let m = map.atoms_map(cell)?;
            let n = h.nrows();
            if n != sys.dof() || h.ncols() != n {
                return Err(crate::error::MoptError::calc(
                    "the Cartesian Hessian does not fit the structure",
                ));
            }
            let mut raw = Mat::zeros(dim, dim);
            for i in 0..n / 3 {
                for j in 0..n / 3 {
                    for a in 0..3 {
                        for c in 0..3 {
                            let mut v = 0.0;
                            for b in 0..3 {
                                for d in 0..3 {
                                    v += m[a][b] * h[(3 * i + b, 3 * j + d)] * m[c][d];
                                }
                            }
                            raw[(3 * i + a, 3 * j + c)] = v;
                        }
                    }
                }
            }
            match &analytic.strain {
                Some((strain, gradient)) => {
                    let (rows, block) = map.lattice_columns(&sys, gradient, strain)?;
                    for l in n..dim {
                        for i in 0..n {
                            raw[(i, l)] = rows[(i, l - n)];
                            raw[(l, i)] = rows[(i, l - n)];
                        }
                        for k in n..dim {
                            raw[(k, l)] = block[(k - n, l - n)];
                        }
                    }
                }
                None => {
                    for l in n..dim {
                        let moved = |sign: f64| {
                            Col::from_fn(dim, |i| {
                                x[i] + if i == l {
                                    sign * DEFAULT_HESSIAN_STEP
                                } else {
                                    0.0
                                }
                            })
                        };
                        let (plus, minus) = (gradient_at(&moved(1.0))?, gradient_at(&moved(-1.0))?);
                        for i in 0..dim {
                            let v = (plus[i] - minus[i]) / (2.0 * DEFAULT_HESSIAN_STEP);
                            raw[(i, l)] = v;
                            if i < n {
                                raw[(l, i)] = v;
                            }
                        }
                    }
                }
            }
            // The block of the lattice variables made symmetric.
            for l in n..dim {
                for k in l + 1..dim {
                    let v = 0.5 * (raw[(l, k)] + raw[(k, l)]);
                    raw[(l, k)] = v;
                    raw[(k, l)] = v;
                }
            }
            raw
        }
    };
    let modes = map.rigid_modes_with(x, symmetries)?;
    let projector = Mat::from_fn(dim, dim, |i, j| {
        let kept = if i == j { 1.0 } else { 0.0 };
        kept - modes.iter().map(|m| m[i] * m[j]).sum::<f64>()
    });
    Ok(&projector * &raw * &projector)
}

/// The analytic second derivatives of `calc` with the bias potentials `terms` at `sys`
/// ([`AnalyticCell`]), where `calc` has an analytic Hessian and every bias potential gives its
/// own (`None` otherwise: the Hessian of the variables of a cell then comes from differences
/// in every column); the second derivatives along the strain where all of them give theirs
/// (the columns of the lattice come from differences otherwise).
pub fn analytic_cell(
    calc: &dyn Calculator,
    terms: &[(String, Arc<dyn PesTerm>)],
    sys: &MolecularSystem,
    excited_state: usize,
    ctx: &TermContext<'_>,
) -> MoptResult<Option<AnalyticCell>> {
    if !calc.analytic_hessian() {
        return Ok(None);
    }
    let request = EvalRequest {
        excited_state,
        need_stress: true,
        ..EvalRequest::ENERGY_GRAD_HESS
    };
    let out = evaluate_checked(calc, sys, &request)?;
    let Some(mut hessian) = out.hessian_bohr else {
        return Ok(None);
    };
    let mut gradient = out.gradient_bohr;
    let mut strain = out.strain_hessian;
    for (label, term) in terms {
        let bias = term.evaluate(sys, ctx, true).context(label.clone())?;
        match bias.hessian_bohr {
            Some(h) if h.nrows() == hessian.nrows() => hessian = &hessian + &h,
            _ => return Ok(None),
        }
        gradient += &bias.gradient_bohr;
        strain = match (strain, &bias.strain_hessian) {
            (Some(mut s), Some(b)) => {
                s.add(b);
                Some(s)
            }
            _ => None,
        };
    }
    Ok(Some(AnalyticCell {
        hessian,
        strain: strain.map(|s| (s, gradient)),
    }))
}

/// The Hessian of the surface of `parts` in the variables of its map at `x`
/// ([`cell_hessian`], with the analytic Hessian of the atoms where there is one).
pub fn cell_hessian_of(
    parts: &CellParts<'_>,
    x: &Col<f64>,
    step: usize,
    of: &str,
) -> MoptResult<Mat<f64>> {
    let sys = parts.map.system(x)?;
    let ctx = TermContext::new(step, parts.start, None);
    // Two states combined: the combined gradient has no analytic derivative here (a gradient
    // projection is no gradient of a function), so every column comes from its differences.
    let analytic = match parts.states {
        Some(_) => None,
        None => analytic_cell(parts.calc, parts.terms, &sys, parts.excited_state, &ctx)?,
    };
    let displaced = format!("{of}, displaced");
    cell_hessian(parts.map, x, analytic.as_ref(), |y| {
        Ok(cell_value(parts, y, step, &displaced)?.slope)
    })
}

/// The enthalpy of the atoms and the cell of structures like `start`, with bias potentials, as
/// an objective of the variables of `map`: what the optimizers of plain variables, the
/// transition-state searches, and the two-point searches run on for a variable cell (the
/// mapper with `-vc`). Its Hessians are those of [`cell_hessian`]. The enthalpy without the
/// bias potentials of the last point it evaluated is kept ([`Self::unbiased_at`]).
pub struct CellObjective<'a> {
    /// The calculator.
    pub calc: &'a dyn Calculator,
    /// Bias potentials with their labels.
    pub terms: Vec<(String, Arc<dyn PesTerm>)>,
    /// The variables.
    pub map: VariableCell,
    /// The starting structure (of the bias potentials).
    pub start: MolecularSystem,
    /// Electronic state (0: the ground state).
    pub excited_state: usize,
    /// Two states combined by a model function (`-mf`), in place of `excited_state`.
    pub states: Option<CellStates>,
    last: std::sync::Mutex<Option<(Col<f64>, f64, Vec<f64>)>>,
}

impl<'a> CellObjective<'a> {
    /// The enthalpy of `calc` with `terms` in the variables `map` of structures like `start`.
    pub fn new(
        calc: &'a dyn Calculator,
        terms: Vec<(String, Arc<dyn PesTerm>)>,
        map: VariableCell,
        start: MolecularSystem,
        excited_state: usize,
    ) -> Self {
        Self {
            calc,
            terms,
            map,
            start,
            excited_state,
            states: None,
            last: std::sync::Mutex::new(None),
        }
    }

    /// The same objective on two states combined by a model function ([`CellStates`];
    /// `None` keeps the one state of `excited_state`).
    #[must_use]
    pub fn with_states(mut self, states: Option<CellStates>) -> Self {
        self.states = states;
        self
    }

    fn parts(&self) -> CellParts<'_> {
        CellParts {
            calc: self.calc,
            terms: &self.terms,
            map: &self.map,
            start: &self.start,
            excited_state: self.excited_state,
            states: self.states.as_ref(),
            log: None,
        }
    }

    /// The enthalpy without the bias potentials at `x`, if `x` is the last point evaluated
    /// (with bias potentials that stand with the states, [`CellStates::with_terms_in_states`],
    /// the combined enthalpy with them).
    pub fn unbiased_at(&self, x: &Col<f64>) -> Option<f64> {
        let last = self.last.lock().ok()?;
        let (at, enthalpy, _) = last.as_ref()?;
        (at.nrows() == x.nrows() && at.iter().zip(x.iter()).all(|(a, b)| a == b))
            .then_some(*enthalpy)
    }

    /// The enthalpy of every state without the bias potentials at `x`
    /// ([`CellValue::state_enthalpies`]), if `x` is the last point evaluated.
    pub fn state_enthalpies_at(&self, x: &Col<f64>) -> Option<Vec<f64>> {
        let last = self.last.lock().ok()?;
        let (at, _, states) = last.as_ref()?;
        (at.nrows() == x.nrows() && at.iter().zip(x.iter()).all(|(a, b)| a == b))
            .then(|| states.clone())
    }

    /// The point at `x` ([`cell_value`]).
    pub fn value(&self, x: &Col<f64>) -> MoptResult<CellValue> {
        cell_value(&self.parts(), x, 0, "variable cell")
    }
}

impl Objective for CellObjective<'_> {
    fn name(&self) -> &str {
        "variable cell"
    }

    fn description(&self) -> String {
        format!(
            "the enthalpy of the atoms and the cell ({} variables) of {}",
            self.map.dimension(),
            self.calc.name()
        )
    }

    fn dimension(&self) -> Option<usize> {
        Some(self.map.dimension())
    }

    fn variable_names(&self, _n: usize) -> Vec<String> {
        self.map.names()
    }

    fn evaluate(&self, x: &Col<f64>, need_hessian: bool) -> MoptResult<ObjectiveValue> {
        let v = self.value(x)?;
        if let Ok(mut last) = self.last.lock() {
            *last = Some((x.clone(), v.enthalpy, v.state_enthalpies.clone()));
        }
        // The states of a point the optimizer takes (not of the differences of a Hessian).
        if let (Some(states), Some(values)) = (&self.states, &v.states) {
            states.commit(&v.system, values);
        }
        let hessian = if need_hessian {
            Some(cell_hessian_of(&self.parts(), x, 0, "variable cell")?)
        } else {
            None
        };
        Ok(ObjectiveValue {
            value: v.biased,
            gradient: v.slope,
            hessian,
        })
    }
}

impl IrcSurface for CellSurface<'_> {
    fn evaluate(&self, request: &IrcRequest) -> MoptResult<PointValue> {
        let point = &request.point;
        let of = format!("IRC point {} ({:?})", point.step, point.role);
        let (energy, enthalpy, slope) = self.point(&point.x, self.iteration, &of)?;
        let hessian = if point.need_hessian {
            Some(cell_hessian_of(
                &self.parts(),
                &point.x,
                self.iteration,
                &of,
            )?)
        } else {
            None
        };
        // The enthalpy is the surface; the energy of the calculator is shown beside it.
        Ok(PointValue {
            energy: enthalpy,
            gradient: slope.clone(),
            raw_energy: energy,
            raw_gradient: slope,
            hessian,
            bias_hessian: None,
        })
    }
}

/// Writes the structures of the path with their cells: `irc_<side>_structures_cell.xyz`
/// for every point and `<stem>_irc_endpoint_<n>.xyz` for the end of every side.
pub struct CellStructures<'a> {
    map: &'a VariableCell,
    dir: PathBuf,
    stem: String,
    endpoints: Vec<PathBuf>,
}

impl<'a> CellStructures<'a> {
    /// The structures of the variables of `map`, into `dir`.
    pub fn new(map: &'a VariableCell, dir: &Path, stem: &str) -> Self {
        Self {
            map,
            dir: dir.to_path_buf(),
            stem: stem.to_string(),
            endpoints: Vec::new(),
        }
    }

    /// The structure files of the ends of the sides, in the order the sides ended.
    pub fn endpoints(&self) -> &[PathBuf] {
        &self.endpoints
    }

    fn side(branch: Branch) -> &'static str {
        match branch {
            Branch::Forward => "forward",
            Branch::Backward => "backward",
            Branch::Single => "single",
        }
    }

    fn frame(&self, x: &Col<f64>, energy: f64) -> MoptResult<String> {
        let sys = self.map.system(x)?;
        Ok(format_xyz_system(&sys, Some(Quantity::new(energy))))
    }
}

impl IrcObserver for CellStructures<'_> {
    fn on_event(&mut self, event: &IrcEvent) -> MoptResult<()> {
        match event {
            IrcEvent::Step { branch, record } => {
                let path = self
                    .dir
                    .join(format!("irc_{}_structures_cell.xyz", Self::side(*branch)));
                let text = self.frame(&record.x, record.biased_energy)?;
                use std::io::Write as _;
                let mut file = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&path)?;
                file.write_all(text.as_bytes())?;
            }
            IrcEvent::BranchFinished(summary) => {
                let number = match summary.branch {
                    Branch::Forward | Branch::Single => 1,
                    Branch::Backward => 2,
                };
                let sys = self.map.system(&summary.x)?;
                let path = self
                    .dir
                    .join(format!("{}_irc_endpoint_{number}.xyz", self.stem));
                std::fs::write(&path, format_xyz_system(&sys, None))?;
                self.endpoints.push(path);
            }
            IrcEvent::Start(_) | IrcEvent::FirstStep(_) => {}
        }
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::domain::{positions, Angstrom, Element, PeriodicCell, SpinCharge, Vector3};
    use crate::pes::native::Emt;

    /// Four copper atoms in a sheared cell, off their sites.
    fn copper() -> MolecularSystem {
        let edge = 3.5;
        let cell = PeriodicCell::from_lattice_vectors(
            Vector3::<Angstrom>::new([edge, 0.0, 0.0]),
            Vector3::<Angstrom>::new([0.05 * edge, 1.04 * edge, 0.0]),
            Vector3::<Angstrom>::new([-0.03 * edge, 0.06 * edge, 0.97 * edge]),
            [true, true, true],
        )
        .unwrap();
        let h = edge / 2.0;
        MolecularSystem::new(
            vec![Element::parse_token("Cu").unwrap(); 4],
            &positions::<Angstrom>(&[
                [0.05, 0.02, -0.03],
                [h + 0.04, h - 0.06, 0.02],
                [h - 0.03, 0.05, h + 0.07],
                [-0.02, h + 0.03, h - 0.05],
            ]),
            SpinCharge::default(),
            Some(cell),
        )
        .unwrap()
    }

    #[test]
    fn the_analytic_block_of_the_atoms_gives_the_hessian_of_the_differences() {
        let sys = copper();
        let calc = Emt::default();
        assert!(calc.analytic_hessian());
        // Under 2 GPa, at a point away from the reference (the cell and the atoms moved), so
        // that the map of the atoms is not the identity.
        let map = VariableCell::new(
            &sys,
            1.0,
            2.0 / crate::opt::cell::gigapascal_per_atomic_unit(),
        )
        .unwrap();
        let mut x = map.variables(&sys).unwrap();
        let n = sys.dof();
        for (k, v) in [
            (n, 0.03),
            (n + 4, -0.02),
            (n + 7, 0.01),
            (2, 0.04),
            (7, -0.03),
        ] {
            x[k] += v;
        }
        let terms: Vec<(String, Arc<dyn PesTerm>)> = Vec::new();
        let parts = CellParts {
            calc: &calc,
            terms: &terms,
            map: &map,
            start: &sys,
            excited_state: 0,
            states: None,
            log: None,
        };
        let analytic = cell_hessian_of(&parts, &x, 0, "test").unwrap();
        let numeric = cell_hessian(&map, &x, None, |y| {
            Ok(cell_value(&parts, y, 0, "test")?.slope)
        })
        .unwrap();
        let scale = (0..x.nrows())
            .map(|i| numeric[(i, i)].abs())
            .fold(0.0_f64, f64::max);
        for i in 0..x.nrows() {
            for j in 0..x.nrows() {
                assert!(
                    (analytic[(i, j)] - numeric[(i, j)]).abs() < 1e-5 * scale,
                    "({i}, {j}): {} {}",
                    analytic[(i, j)],
                    numeric[(i, j)]
                );
            }
        }
    }
}
