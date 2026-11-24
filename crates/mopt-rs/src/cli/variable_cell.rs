//! `optmain -vc`: optimization of the atoms and the cell together (`opt::cell`).
//!
//! The surface is that of `workflow::irc_cell` (the enthalpy with the bias potentials, and
//! its gradient in the variables of the cell): one electronic state (`-es`), or two combined
//! by the model function of `-mf` (crossing points, conical intersections, penalty
//! functions) in the variables of the cell ([`CellStates`]: the enthalpies of the states and
//! their gradients in the variables, from their energies, gradients, and stresses), with the
//! artificial forces on the cell (`-vc_afir`) on top.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::cell::Cell;
use std::path::Path;
use std::sync::Arc;

use super::optmain::{JobSummary, OptmainConfig, OptmainResult};
use crate::core_infra::log::{python_float, Logger};
use crate::opt::vector_constraints::VectorConstraintFrame;
use faer::{Col, Mat};

use crate::domain::{MolecularSystem, Quantity};
use crate::error::{MoptError, MoptResult};
use crate::math::linalg::symmetric_eigen;
use crate::opt::cell::{
    gigapascal_per_atomic_unit, CellDistance, CellForce, CellSymmetries, VariableCell,
};
use crate::opt::cell_constraints::{cell_constraints, pins_rotations, pins_translations};
use crate::opt::state_machine::{CrossingReport, IterationReport, SpinSquare, Termination};
use crate::opt::vector::{VectorAction, VectorOptimizer, VectorSettings};
use crate::pes::bias::Ramp;
use crate::pes::calculator::Calculator;
use crate::pes::constraint::Constraint;
use crate::pes::stress::Stress;
use crate::pes::term::{PesTerm, TermContext};
use crate::vib;
use crate::workflow::driver::Outcome;
use crate::workflow::irc_cell::{
    analytic_cell, cell_hessian_with, cell_value, CellParts, CellStates, CellValue,
};
use crate::workflow::observers::{geometry_block, ProgressLog, RunFiles, StopFile};
use crate::workflow::Observer;

/// Settings of `-vc`.
#[derive(Debug, Clone)]
pub struct VariableCellSettings {
    /// External pressure in GPa (`-vc_pressure`).
    pub pressure_gpa: f64,
    /// Weight `w` of the lattice variables (`-vc_weight`).
    pub weight: f64,
    /// Artificial forces on the cell (`-vc_afir`).
    pub forces: Vec<CellForce>,
}

impl VariableCellSettings {
    /// `-vc` with `-vc_pressure`, `-vc_weight`, and `-vc_afir` of `p` (`None` without `-vc`),
    /// as `optmain` reads them; for the commands that take the options of `optmain -vc`.
    pub fn from_args(p: &crate::core_infra::arg_parser::ParsedArgs) -> MoptResult<Option<Self>> {
        if !p.get_bool("variable_cell", false) {
            return Ok(None);
        }
        Ok(Some(Self {
            pressure_gpa: p.get_f64("variable_cell_pressure", 0.0)?,
            weight: p.get_f64(
                "variable_cell_weight",
                crate::opt::cell::DEFAULT_LATTICE_WEIGHT,
            )?,
            forces: parse_cell_forces(p.get_list("variable_cell_afir"))?,
        }))
    }

    /// Coordinates and enthalpy convention of a cell referenced to this structure.
    pub fn coordinates(&self, sys: &MolecularSystem) -> MoptResult<VariableCell> {
        VariableCell::new(
            sys,
            self.weight,
            self.pressure_gpa / gigapascal_per_atomic_unit(),
        )
    }
}

/// The two states of `-mf` for a variable cell, combined in the variables of the cell
/// ([`CellStates`]); `None` without `-mf` (`excited_state` is then the one state of the
/// surface). `-mf bitss` is a search with two structures and refused, as in a fixed cell.
pub fn cell_states(
    model_function: &[String],
    excited_state: usize,
) -> MoptResult<Option<CellStates>> {
    if model_function.is_empty() {
        return Ok(None);
    }
    CellStates::of(super::optmain::state_surface(
        model_function,
        excited_state,
    )?)
}

/// `-vc_afir GAMMA DISTANCES ...`: pairs of a collision energy (kJ/mol, with a ramp as for
/// `-ma`) and a list of distances of the cell (`a,b,c,ab,bc,ca`).
pub fn parse_cell_forces(tokens: &[String]) -> MoptResult<Vec<CellForce>> {
    if !tokens.len().is_multiple_of(2) {
        return Err(MoptError::parse(
            "-vc_afir expects pairs of a collision energy and a list of distances (a,b,c,ab,bc,ca)",
        ));
    }
    tokens
        .chunks(2)
        .map(|pair| {
            Ok(CellForce {
                gamma: Ramp::parse(&pair[0])
                    .map_err(|e| MoptError::parse(format!("-vc_afir: {e}")))?,
                distances: pair[1]
                    .split(',')
                    .map(CellDistance::parse)
                    .collect::<MoptResult<Vec<_>>>()?,
            })
        })
        .collect()
}

/// The volume and stress in atomic units (`mean`: the stress is the mean of two states).
fn cell_lines(sys: &MolecularSystem, stress: &Stress, mean: bool) -> Vec<String> {
    let mut lines = Vec::new();
    let Some(cell) = sys.active_cell() else {
        return lines;
    };
    lines.push(format!(
        "{:<23}: {:>15.6} ",
        "VOLUME (unit a.u.)",
        cell.volume_bohr3()
    ));
    lines.push(if mean {
        "STRESS, mean of the two states (unit a.u.; xx yy zz yz xz xy):".to_string()
    } else {
        "STRESS (unit a.u.; xx yy zz yz xz xy):".to_string()
    });
    lines.push(
        [
            stress[0][0],
            stress[1][1],
            stress[2][2],
            stress[1][2],
            stress[0][2],
            stress[0][1],
        ]
        .iter()
        .map(|v| format!("{v:15.12}"))
        .collect::<Vec<String>>()
        .join(" "),
    );
    if cell.periodic_axes.iter().all(|p| *p) {
        let pressure = -(stress[0][0] + stress[1][1] + stress[2][2]) / 3.0;
        lines.push(format!(
            "{:<23}: {pressure:>15.12}  ({} GPa)",
            "PRESSURE",
            python_float(pressure * gigapascal_per_atomic_unit())
        ));
    } else {
        // A slab or a wire: volume and stress refer to the stored cell with its vectors
        // along the directions that are not periodic; their product is what counts.
        lines.push(
            "(volume and stress refer to the whole stored cell; a slab or wire has no pressure)"
                .to_string(),
        );
    }
    lines
}

/// The Hessian `h` on the surface of the constraints of `frame`: `P h P` with `P` the
/// projector onto the directions that keep them, so that the held directions have no
/// curvature (the Hessian of the Lagrangian when `h` carries the curvature of the
/// constraints).
pub fn surface_hessian(frame: &VectorConstraintFrame, h: &Mat<f64>) -> Mat<f64> {
    let n = h.nrows();
    if frame.basis.ncols() == 0 {
        return h.clone();
    }
    let p = Mat::<f64>::identity(n, n) - &frame.basis * frame.basis.transpose();
    let out = &p * h * &p;
    Mat::from_fn(n, n, |i, j| 0.5 * (out[(i, j)] + out[(j, i)]))
}

/// One set of variables evaluated.
struct Point {
    /// The surface of the states with the bias potentials ([`cell_value`]).
    value: CellValue,
    /// Energies of the calculator (or of every state), the bias potentials, and the forces on
    /// the cell.
    parts: Vec<(String, f64)>,
    /// The function that is minimized (with the forces on the cell).
    function: f64,
    /// Its gradient in the variables.
    slope: Col<f64>,
}

/// Optimize the atoms and the cell of `start` with `calc`, logging to `log` and writing the
/// files of a run into `dir`.
#[allow(clippy::too_many_arguments)]
pub(super) fn run(
    config: &OptmainConfig,
    cell: &VariableCellSettings,
    terms: &[(String, Arc<dyn PesTerm>)],
    constraints: &[Constraint],
    input: &Path,
    start: &MolecularSystem,
    calc: &dyn Calculator,
    dir: &Path,
    log: &Logger,
) -> MoptResult<OptmainResult> {
    let s = &config.settings;
    let map = cell.coordinates(start)?;
    if !constraints.is_empty() && config.irc.is_some() {
        return Err(MoptError::unsupported(
            "-irc after a constrained -vc: the IRC of the atoms and the cell follows the \
             surface without constraints",
        ));
    }
    // The constraints in the variables of the cell (frozen atoms keep their fractional
    // coordinates; distances, angles, dihedrals, and -uc functions of the structure).
    let held = cell_constraints(&map, constraints, start)?;
    let pressure = map.pressure();
    let states = cell_states(&config.model_function, config.excited_state)?;
    if states.is_some() {
        if !held.is_empty()
            && config
                .model_function
                .first()
                .is_some_and(|m| super::optmain::is_gradient_projection(m))
        {
            // Measured: -mf opt_mesx with the length of a held stopped as converged with a
            // gap of 2.4e-3 Hartree (test variable_cell_constraints).
            return Err(MoptError::unsupported(
                "-mf with a gradient projection (opt_mesx, opt_meci) and constraints under -vc: \
                 the projection of the two states is formed in all the variables of the cell \
                 before the constraints are projected out, so that its parts can cancel away \
                 from the seam; use a penalty function (-mf seam, conical, avoiding) with the \
                 constraints",
            ));
        }
        if s.projection_trust {
            return Err(MoptError::unsupported(
                "-projection_trust merit with -vc: the optimizer of the atoms and the cell judges \
                 its steps by the function it minimizes (-projection_trust energy)",
            ));
        }
        if config.irc.is_some() {
            return Err(MoptError::unsupported(
                "-irc after -vc with -mf: the IRC of the atoms and the cell follows one state",
            ));
        }
    }
    let settings = VectorSettings {
        methods: s.methods.clone(),
        saddle_order: s.saddle_order,
        max_iterations: s.max_iterations,
        criteria: s.criteria,
        trust: s.trust,
        exact_hessian_interval: s.exact_hessian_interval,
        ccqn: None,
        mode_kill: s.mode_kill,
    };
    let mut opt = VectorOptimizer::new(settings, map.variables(start)?)?;
    // The constraints as the optimizer holds them, for the curvature they add to exact
    // Hessians (the Hessian of the Lagrangian).
    let curvature_of = (!held.is_empty()).then(|| held.clone());
    if !held.is_empty() {
        log.info(&format!(
            "Variable cell: {} constraint(s) held in the variables of the atoms and the cell",
            held.len()
        ));
        for note in held.notes(&map.variables(start)?)? {
            log.info(&note);
        }
        opt.set_constraints(held)?;
    }
    // The rigid motions of the Hessian: the translations of the atoms only when nothing pins
    // them (frozen atoms keep their fractional coordinates), the rotations of the lattice
    // only when no constraint holds a Cartesian component of a lattice vector, every symmetry
    // only on a surface that does not see where the structure is.
    let invariant = crate::pes::term::frame_invariant(
        calc.is_rigid_body_invariant(),
        terms.iter().map(|(_, t)| t.as_ref()),
    );
    let symmetries = CellSymmetries {
        translations: !pins_translations(constraints) && invariant,
        rotations: !pins_rotations(constraints) && invariant,
    };
    if s.mode_kill.is_some() {
        opt.configure_mode_kill(config.hisd_mode_kill, Some(map.masses()?));
    }
    log.info(&format!(
        "Variable cell: atoms in quasi-Cartesian coordinates and scaled lattice vectors \
         (weight {}) as one set of {} variables; external pressure {} GPa; the function \
         minimized is the energy with the bias potentials, the forces on the cell, and P V \
         (shown as BIAS ENERGY)",
        python_float(cell.weight),
        map.dimension(),
        python_float(cell.pressure_gpa)
    ));
    log.info("Reference: M. Gubler, M. Krummenacher, H. Huber, S. Goedecker, J. Comput. Phys. X 2023, 17, 100131 (variables of the variable-cell optimization)");
    match &states {
        Some(states) => log.info(&format!(
            "Mode: Model Function Optimization ({}) in a variable cell: the enthalpies of the \
             states ({}) and their gradients in the variables, from their energies, gradients, \
             and stresses, are combined as the energies and gradients of the states in a fixed \
             cell; the gap is that of the energies (P V is common to both). The check of the \
             branching plane at convergence is one of a fixed cell and is not run here.",
            states.name(),
            states.labels().join(", ")
        )),
        None if config.excited_state > 0 => {
            log.info(&format!("Electronic state: {}", config.excited_state));
        }
        None => {}
    }
    log.info(&format!("{}\n", geometry_block(start)));
    let stem = input
        .file_stem()
        .map_or_else(|| "input".to_string(), |s| s.to_string_lossy().into_owned());
    let mut progress = ProgressLog::new(log.clone(), s.criteria);
    let mut files = RunFiles::new(dir, &stem)?;
    let mut stop = StopFile::new(dir, log.clone());
    let mut last: Option<(MolecularSystem, f64, f64)> = None;
    if let Some(force) = cell.forces.iter().find(|f| !map.takes(f)) {
        return Err(MoptError::config(format!(
            "-vc_afir: a distance of {:?} belongs to a lattice vector of a direction that is not \
             periodic",
            force.distances
        )));
    }
    // The surface of the states with the bias potentials at a set of variables
    // ([`cell_value`]), with the energies of the forces on the cell, the function that is
    // minimized, and its gradient in the variables.
    let iteration = Cell::new(0_usize);
    let parts = CellParts {
        calc,
        terms,
        map: &map,
        start,
        excited_state: config.excited_state,
        states: states.as_ref(),
        log: None,
    };
    let point = |x: &Col<f64>| -> MoptResult<Point> {
        let value = cell_value(&parts, x, iteration.get(), "variable cell")?;
        let mut function = value.biased;
        let mut slope = value.slope.clone();
        let mut energies = value.components.clone();
        // Forces on the cell itself.
        if let Some(lattice) = value.system.active_cell() {
            let vectors: [[f64; 3]; 3] = std::array::from_fn(|k| lattice.lattice_vector(k).value());
            for (k, force) in cell.forces.iter().enumerate() {
                let (energy, of_lattice) = force.evaluate(&vectors, iteration.get());
                function += energy;
                map.add_lattice_gradient(&mut slope, &of_lattice);
                energies.push((format!("cell_AFIR_{k}"), energy));
            }
        }
        Ok(Point {
            value,
            parts: energies,
            function,
            slope,
        })
    };
    let termination: Option<Termination> = loop {
        let (x, need_hessian) = match opt.ask() {
            VectorAction::Finished(t) => break Some(t),
            VectorAction::Evaluate { x, need_hessian } => (x, need_hessian),
        };
        iteration.set(opt.iterations());
        let Point {
            value,
            parts: energies,
            function: enthalpy,
            slope: gradient,
        } = point(&x)?;
        let sys = value.system.clone();
        let energy = value.energy;
        // The Hessian in the variables of the atoms and the cell: the block of the atoms
        // from the analytic Hessian of the calculator and the bias potentials where they
        // have one, the columns of the lattice variables from their second derivatives along
        // the strain where they give them, from differences of the gradient otherwise (every
        // column without an analytic Hessian); the translations and rotations of the whole
        // structure taken out (away from a stationary point their curvatures are not zero,
        // and a search for a saddle point must not follow them). The forces on the cell act
        // on the lattice alone.
        let hessian = if need_hessian {
            let ctx = TermContext::new(iteration.get(), start, None);
            // Two states combined: every column from differences of the combined gradient.
            let mut analytic = match &states {
                Some(_) => None,
                None => analytic_cell(calc, terms, &sys, config.excited_state, &ctx)?,
            };
            if let (Some(a), Some(lattice)) = (analytic.as_mut(), sys.active_cell()) {
                let vectors: [[f64; 3]; 3] =
                    std::array::from_fn(|k| lattice.lattice_vector(k).value());
                for force in &cell.forces {
                    a.add_lattice(&force.strain_hessian(&vectors, ctx.iteration));
                }
            }
            let mut h = cell_hessian_with(&map, &x, analytic.as_ref(), symmetries, |y| {
                Ok(point(y)?.slope)
            })?;
            // The curvature of the constraints that give theirs (the functions of the cell):
            // the Hessian of the Lagrangian, `H - sum_k lambda_k d2c_k`.
            // The rigid motions are taken out of it as out of the Hessian: along a straight
            // line the length of a turned vector grows, so the curvature of a length has a
            // part along the linear rotations, which are no motion of the surface.
            if let Some(c) = &curvature_of {
                if let Some(k) = c.curvature(&x, &gradient)? {
                    let dim = x.nrows();
                    let modes = map.rigid_modes_with(&x, symmetries)?;
                    let p = Mat::from_fn(dim, dim, |i, j| {
                        let kept = if i == j { 1.0 } else { 0.0 };
                        kept - modes.iter().map(|m| m[i] * m[j]).sum::<f64>()
                    });
                    h = &h - &(&p * &k * &p);
                }
            }
            Some(h)
        } else {
            None
        };
        // On the constraint surface: the held directions are taken out with the rigid
        // motions (their zero eigenvalues are not listed), as for the constraints of atoms.
        let eigenvalues = match &hessian {
            Some(h) => {
                let on_surface = match &curvature_of {
                    Some(c) => surface_hessian(&c.frame(&x)?, h),
                    None => h.clone(),
                };
                Some(
                    symmetric_eigen(&on_surface)?
                        .0
                        .iter()
                        .copied()
                        .filter(|v| v.abs() > 1e-10)
                        .collect::<Vec<f64>>(),
                )
            }
            None => None,
        };
        let r = opt.tell(enthalpy, gradient, hessian)?;
        // The states of the point the optimizer took, for a combiner with a history.
        if let (Some(states), Some(values)) = (&states, &value.states) {
            states.commit(&value.system, values);
        }
        let mut notes = r.notes.clone();
        notes.extend(cell_lines(&sys, &value.stress, value.states.is_some()));
        if let Some((gap, difference)) = value.crossing {
            notes.push(format!(
                "STATES: {}; gap E1 - E2 {gap:+.10} Hartree, |g1 - g2| {difference:.6e} \
                 Hartree/Bohr",
                value
                    .outputs
                    .iter()
                    .map(|o| format!("{} {:.10}", o.label, o.output.energy.value()))
                    .collect::<Vec<String>>()
                    .join(", ")
            ));
        }
        let report = IterationReport {
            crossing: value.crossing.map(|(gap, difference)| CrossingReport {
                gap,
                gradient_difference: difference,
            }),
            iteration: r.iteration,
            system: sys.clone(),
            energy: Quantity::new(energy),
            biased_energy: Quantity::new(enthalpy),
            gradient: value.gradient.clone(),
            raw_gradient: value.raw_gradient.clone(),
            components: energies
                .iter()
                .map(|(label, value)| (label.clone(), Quantity::new(*value)))
                .collect(),
            convergence: r.convergence,
            step_norm: Quantity::new(r.step_norm),
            trust_radius: Quantity::new(r.trust_radius),
            notes,
            debug_notes: Vec::new(),
            program_outputs: value
                .outputs
                .iter()
                .filter_map(|o| {
                    let text = o.output.orbitals.as_ref()?.program_output.clone()?;
                    Some((o.label.clone(), text))
                })
                .collect(),
            coordinates: None,
            spin_squares: value
                .outputs
                .iter()
                .filter_map(|o| {
                    let measured = o.output.orbitals.as_ref()?.spin_square?;
                    let spin = 0.5 * (f64::from(o.multiplicity) - 1.0);
                    Some(SpinSquare {
                        label: o.label.clone(),
                        value: measured,
                        exact: spin * (spin + 1.0),
                    })
                })
                .collect(),
            hessian_eigenvalues: eigenvalues,
        };
        progress.on_iteration_with_hessian(&report, opt.updated_hessian())?;
        files.on_iteration(&report)?;
        let ctx = TermContext::new(r.iteration, start, None);
        for (_, term) in terms {
            term.commit(&sys, &ctx);
        }
        last = Some((sys.clone(), energy, enthalpy));
        if stop.should_stop(&sys) {
            break None;
        }
    };
    let (system, energy, enthalpy) =
        last.ok_or_else(|| MoptError::config("the optimization ended before an evaluation"))?;
    let outcome = Outcome {
        termination,
        system: system.clone(),
        energy: Quantity::new(energy),
        biased_energy: Quantity::new(enthalpy),
        iterations: opt.iterations(),
    };
    progress.on_finish(&outcome)?;
    files.on_finish(&outcome)?;
    let converged = termination == Some(Termination::Converged);
    let mut symmetry = None;
    if converged {
        let label = match config.frequency.symmetry_tolerance {
            Some(tolerance) => vib::point_group(&system, tolerance).label,
            None => "C1".to_string(),
        };
        std::fs::write(
            dir.join("symmetry.txt"),
            format!("Symmetry of final structure: {label}"),
        )?;
        log.info(&format!("Symmetry: {label}"));
        symmetry = Some(label);
    }
    // -freq: the vibrations of the optimized atoms in the optimized cell, the cell held.
    let mut imaginary_modes = None;
    if (config.frequency_analysis && converged) || config.force_frequency_analysis {
        let wavenumbers = held_cell_frequencies(config, terms, &system, calc, dir, log)?;
        imaginary_modes = wavenumbers.map(|w| vib::imaginary_wavenumbers(&w));
        if !converged {
            log.warn(
                "Vibrational analysis was performed, but the optimization did not converge. \
                 The result of thermochemistry is unreliable.",
            );
        }
    }
    // MultiOptPy: the IRC from the last structure of the optimization; here of the atoms
    // and the cell together.
    let mut irc_endpoints = Vec::new();
    if let (Some(irc), false) = (&config.irc, stop.stopped()) {
        if !cell.forces.is_empty() {
            return Err(MoptError::unsupported(
                "-irc after -vc_afir: the forces on the cell push it in a search; the IRC \
                 follows the surface itself",
            ));
        }
        let irc = if converged {
            irc.clone()
                .accepting_converged(s.criteria.max_force.value(), s.criteria.rms_force.value())
        } else {
            irc.clone()
        };
        let run = super::irc::run_cell(
            &irc,
            calc,
            terms,
            &system,
            config.excited_state,
            iteration.get(),
            cell.weight,
            pressure,
            &stem,
            dir,
            log,
            &mut [],
        )?;
        if run.interrupted {
            log.info(&format!(
                "Continue with: optmain --restart {}",
                dir.display()
            ));
        }
        irc_endpoints = run.endpoints;
    }
    log.info(&format!(
        "Trial of geometry optimization ({}) was completed.",
        input.display()
    ));
    if config.inputs.last().is_some_and(|last| last == input) {
        log.info("All calculations were completed.");
    }
    log.info(&format!(
        "Optimized structure file path: {}",
        files.optimized_path().display()
    ));
    log.info(&format!(
        "Trajectory file path: {}",
        files.trajectory_path().display()
    ));
    log.flush();
    Ok(OptmainResult {
        directory: dir.to_path_buf(),
        optimized: files.optimized_path(),
        converged,
        interrupted: stop.stopped(),
        summary: Some(JobSummary {
            input: input.to_path_buf(),
            trajectory: files.trajectory_path(),
            system,
            energy: Quantity::new(energy),
            biased_energy: Quantity::new(enthalpy),
            symmetry,
            iterations: outcome.iterations,
            irc_endpoints,
            imaginary_modes,
        }),
    })
}

/// `-freq` after `-vc`: the vibrational analysis of the optimized structure `system` in its
/// optimized cell with the cell held, the analysis of a fixed cell (`optmain -freq`): the
/// Hessian of the atoms on the surface of the calculator and the bias potentials, the modes at
/// the zone centre with the rigid translations projected out, and with constraints of the
/// atoms the vibrations on their surface. The lattice variables are no part of it (their
/// curvature is that of the strain, not a vibration), nor the artificial forces on the cell,
/// which act on the lattice alone. Returns the wavenumbers, `None` at a crossing of two
/// states (the vibrations within the seam are written instead).
///
/// MultiOptPy: no counterpart (new in mopt-rs).
fn held_cell_frequencies(
    config: &OptmainConfig,
    terms: &[(String, Arc<dyn PesTerm>)],
    system: &MolecularSystem,
    calc: &dyn Calculator,
    dir: &Path,
    log: &Logger,
) -> MoptResult<Option<Vec<f64>>> {
    log.info(
        "Vibrational analysis after the variable-cell optimization: the atoms of the optimized \
         structure in the optimized cell, the cell held (the analysis of a fixed cell; the \
         lattice variables and the forces on the cell are not part of it)",
    );
    // The constraints of the atoms at the optimized structure: frozen atoms where they ended
    // (they kept their fractional coordinates, so their Cartesian positions moved with the
    // cell), explicit targets as given. The functions of the cell alone hold with the cell.
    let of_atoms: Vec<Constraint> = config
        .settings
        .constraints
        .iter()
        .filter(|c| match &c.kind {
            crate::pes::constraint::ConstraintKind::Function(f) => {
                !f.term.atoms().is_some_and(|a| a.is_empty())
            }
            _ => true,
        })
        .cloned()
        .collect();
    let dropped = config.settings.constraints.len() - of_atoms.len();
    if dropped > 0 {
        log.info(&format!(
            "{dropped} constraint(s) of the cell alone hold with the held cell and are not part \
             of the analysis"
        ));
    }
    let resolved = super::constraints::resolve_for_run(&of_atoms, system)
        .map_err(|e| MoptError::unsupported(format!("-freq after -vc: {e}")))?;
    let settings = crate::opt::state_machine::OptimizerSettings {
        constraints: resolved,
        rigid_body_invariant: crate::pes::term::frame_invariant(
            calc.is_rigid_body_invariant(),
            terms.iter().map(|(_, t)| t.as_ref()),
        ),
        ..crate::opt::state_machine::OptimizerSettings::default()
    };
    let mut opt = crate::opt::state_machine::GeometryOptimizer::new(settings, system.clone())
        .map_err(|e| {
            MoptError::unsupported(format!(
                "-freq after -vc: the analysis in the held cell cannot take the constraints or \
                 the structure ({e})"
            ))
        })?;
    if let Some(surface) =
        super::optmain::state_surface(&config.model_function, config.excited_state)?
    {
        opt.set_surface(surface)?;
    }
    for (label, term) in terms {
        opt.add_term(label, Arc::clone(term))?;
    }
    let wavenumbers = super::optmain::write_frequency_analysis(
        &mut opt,
        calc,
        system,
        &config.frequency,
        config.project_gradient,
        dir,
        log,
    )?;
    // In the log, and beside the table of the modes (whose layout is MultiOptPy's).
    let note = "The cell was held at its optimized lattice vectors: the vibrations in \
                normal_modes.txt are those of the atoms in that cell (zone centre).";
    log.info(note);
    std::fs::write(dir.join("frequency_cell.txt"), format!("{note}\n"))?;
    Ok(wavenumbers)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;
    use crate::opt::vector_constraints::{FixedVariable, VectorConstraintSet};

    #[test]
    fn the_held_directions_have_no_curvature_on_the_constraint_surface() {
        let mut set = VectorConstraintSet::new();
        set.push(Arc::new(FixedVariable { index: 1 }), 0.0).unwrap();
        let x = Col::from_fn(3, |i| [0.1, 0.0, -0.2][i]);
        let h = Mat::from_fn(3, 3, |i, j| if i == j { [2.0, 5.0, 3.0][i] } else { 0.1 });
        let on = surface_hessian(&set.frame(&x).unwrap(), &h);
        let (w, _) = symmetric_eigen(&on).unwrap();
        let zero = w.iter().filter(|v| v.abs() < 1e-12).count();
        assert_eq!(zero, 1, "{w:?}");
        for k in 0..3 {
            assert!(on[(1, k)].abs() < 1e-15 && on[(k, 1)].abs() < 1e-15);
        }
    }
}
