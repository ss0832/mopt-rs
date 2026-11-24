//! What a scan scans and how one point of it is prepared and run: the structure or point of
//! plain variables of the input (`Surface`), the parts every point needs (`Scan`: the grid,
//! the ReSPer references, the constraints of a point, the pre-images, the structures of rigid
//! points), and `run_point`, which chooses the start of a point, runs it
//! ([`super::relaxedscan_points`]), and records its structure.
//!
//! MultiOptPy: the loop of `Entrypoints/relaxed_scan.py:run_relaxedscan` (the input of each
//!  point: the optimized structure of the one before, or the input with `-fo`).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use faer::Col;

use super::relaxedscan_points::{
    MoleculePoints, PointConstraints, PointOutcome, Structure, VariablePoints, FINAL_HESSIAN,
};
use super::relaxedscan_resper::{structure_constraint, variable_constraint, ResperReferences};
use super::relaxedscan_settings::{Outside, RelaxedscanConfig, ScanTarget, TargetKind};
use super::variables::read_variables_file;
use crate::analysis::path_map::mass_weighted_distance;
use crate::analysis::resper::preimage::{preimage, Preimage};
use crate::coord::connectivity::BondConnectivity;
use crate::core_infra::log::{Level, Logger};
use crate::error::{MoptError, MoptResult, ResultExt};
use crate::io::xyz::{format_xyz_system, read_xyz_file};
use crate::opt::vector_constraints::{FixedVariable, VectorConstraintSet};
use crate::pes::objective::{Objective, ObjectiveRegistry};
use crate::pes::registry::CalculatorRegistry;
use crate::workflow::observers::STOP_FILE;
use crate::workflow::scan::rigid::{bond_graph, set_rigidly};
use crate::workflow::scan::{GridKind, ScanGrid, ScanPoint, ScanState};

/// What is scanned: a structure or the variables of an objective (one per scan).
#[allow(clippy::large_enum_variant)]
pub(crate) enum Surface {
    /// Atoms, with the bond graph of the input (for rigid moves).
    Atoms {
        /// The input structure.
        start: crate::domain::MolecularSystem,
        /// Its bond graph.
        bonds: BondConnectivity,
    },
    /// Plain variables.
    Plain {
        /// The objective.
        objective: Box<dyn Objective>,
        /// The names of the variables.
        names: Vec<String>,
        /// The input point.
        start: Col<f64>,
        /// Whether the variables file is JSON.
        json: bool,
    },
}

impl Surface {
    pub(crate) fn start(&self) -> Structure {
        match self {
            Self::Atoms { start, .. } => Structure::Atoms(start.clone()),
            Self::Plain { start, .. } => Structure::Variables(start.clone()),
        }
    }

    pub(crate) fn extension(&self) -> &'static str {
        match self {
            Self::Atoms { .. } => "xyz",
            Self::Plain { json: true, .. } => "json",
            Self::Plain { .. } => "txt",
        }
    }
}

/// The text of the file of a structure with `comment`.
pub(crate) fn structure_text(
    surface: &Surface,
    structure: &Structure,
    comment: &str,
    energy: f64,
) -> String {
    match (surface, structure) {
        (_, Structure::Atoms(s)) => {
            let mut s = s.clone();
            s.comment = comment.to_string();
            format_xyz_system(&s, None)
        }
        (Surface::Plain { names, json, .. }, Structure::Variables(x)) => {
            let values: Vec<f64> = x.iter().copied().collect();
            super::variables::format_variables(names, &values, *json, energy)
        }
        (Surface::Atoms { .. }, Structure::Variables(_)) => String::new(),
    }
}

/// The distance of two structures (Angstrom amu^0.5 after superposition; Euclidean for
/// variables).
pub(crate) fn distance(a: &Structure, b: &Structure) -> Option<f64> {
    match (a, b) {
        (Structure::Atoms(a), Structure::Atoms(b)) => mass_weighted_distance(a, b).ok(),
        (Structure::Variables(a), Structure::Variables(b)) if a.nrows() == b.nrows() => {
            Some((a - b).norm_l2())
        }
        _ => None,
    }
}

/// The scan's parts that every point needs.
pub(crate) struct Scan<'a> {
    pub(crate) config: &'a RelaxedscanConfig,
    pub(crate) surface: Surface,
    pub(crate) grid: ScanGrid,
    pub(crate) resper: Option<ResperReferences>,
    /// Index of each variable target (plain variables).
    pub(crate) variables: Vec<Option<usize>>,
    pub(crate) dir: PathBuf,
    pub(crate) stem: String,
    pub(crate) log: Logger,
}

impl Scan<'_> {
    pub(crate) fn targets(&self) -> &[ScanTarget] {
        &self.config.scan.targets
    }

    /// The pre-image of the ReSPer coordinates of `point`, if it has any.
    fn preimage_of(&self, point: usize) -> MoptResult<Option<Preimage>> {
        let Some(refs) = &self.resper else {
            return Ok(None);
        };
        let values = self.grid.values(point);
        let (mut components, mut target) = (Vec::new(), Vec::new());
        for (t, v) in self.targets().iter().zip(&values) {
            if let TargetKind::Resper(k) = t.kind {
                components.push(k);
                target.push(*v);
            }
        }
        let settings = self
            .config
            .scan
            .resper
            .as_ref()
            .map(|r| r.preimage)
            .unwrap_or_default();
        preimage(&refs.map, &refs.aligned, &components, &target, &settings).map(Some)
    }

    /// The constraints of `point` for a scan of structures.
    fn molecule_constraints(&self, point: usize) -> MoptResult<PointConstraints> {
        let mut out = PointConstraints::default();
        let curvature = self
            .config
            .scan
            .resper
            .as_ref()
            .is_some_and(|r| r.curvature);
        for (t, v) in self.targets().iter().zip(self.grid.values(point)) {
            match &t.kind {
                TargetKind::Resper(k) => {
                    let refs = self
                        .resper
                        .as_ref()
                        .ok_or_else(|| MoptError::config("ReSPer coordinates need -resper"))?;
                    out.constraints
                        .push(structure_constraint(refs, *k, v, curvature)?);
                }
                _ => {
                    if let Some(c) = t.constraint(v)? {
                        out.constraints.push(c);
                    }
                    if let Some(tokens) = t.option_tokens(v) {
                        out.tokens.push(tokens);
                    }
                }
            }
        }
        Ok(out)
    }

    /// The constraints of `point` for a scan of plain variables.
    fn variable_constraints(&self, point: usize) -> MoptResult<VectorConstraintSet> {
        let mut set = VectorConstraintSet::new();
        for ((t, v), index) in self
            .targets()
            .iter()
            .zip(self.grid.values(point))
            .zip(&self.variables)
        {
            match (&t.kind, index) {
                (TargetKind::Resper(k), _) => {
                    let refs = self
                        .resper
                        .as_ref()
                        .ok_or_else(|| MoptError::config("ReSPer coordinates need -resper"))?;
                    set.push(Arc::new(variable_constraint(refs, *k)?), v)?;
                }
                (_, Some(i)) => set.push(Arc::new(FixedVariable { index: *i }), v)?,
                _ => {}
            }
        }
        Ok(set)
    }

    /// The structure a single point is calculated at: the input (or the pre-image of its
    /// ReSPer coordinates) with the other coordinates set rigidly.
    fn rigid_structure(&self, point: usize, preimage: Option<&Preimage>) -> MoptResult<Structure> {
        match &self.surface {
            Surface::Atoms { start, bonds } => {
                let mut s = start.clone();
                if let Some(p) = preimage {
                    s = s.with_flat_coords_bohr(&p.x)?;
                }
                // The ReSPer coordinates are those of the pre-image; the others are set by
                // rigid moves, but for the parameters of the cell, which the variable-cell
                // optimization of the point brings onto their values from its start.
                let mut geometric = Vec::new();
                for (t, v) in self.targets().iter().zip(self.grid.values(point)) {
                    if !matches!(t.kind, TargetKind::Resper(_) | TargetKind::Cell(_)) {
                        geometric.extend(t.constraint(v)?);
                    }
                }
                if !geometric.is_empty() {
                    s = set_rigidly(&s, bonds, &geometric)?;
                }
                Ok(Structure::Atoms(s))
            }
            Surface::Plain { start, .. } => {
                let mut x = preimage.map_or_else(|| start.clone(), |p| p.x.clone());
                for ((t, v), index) in self
                    .targets()
                    .iter()
                    .zip(self.grid.values(point))
                    .zip(&self.variables)
                {
                    if let (TargetKind::Variable(_), Some(i)) = (&t.kind, index) {
                        x[*i] = v;
                    }
                }
                Ok(Structure::Variables(x))
            }
        }
    }

    pub(crate) fn point_name(&self, point: usize) -> String {
        let width = self.grid.len().to_string().len().max(3);
        format!("point_{:0width$}", point + 1)
    }

    /// The directory of the optimization that gave the result `r` (that of the repair when
    /// the point was optimized again).
    pub(crate) fn point_dir(&self, r: &ScanPoint) -> PathBuf {
        let name = self.point_name(r.point);
        match (r.repaired, r.start_from) {
            (true, Some(f)) => self.dir.join(format!("{name}_from_{}", f + 1)),
            _ => self.dir.join(name),
        }
    }

    /// Which dimensions of the grid are periodic: dihedral angles over one whole turn (the
    /// last value is the first); a joint curve when all its coordinates are.
    pub(crate) fn periodic_dims(&self) -> Vec<bool> {
        let full_turn: Vec<bool> = self
            .targets()
            .iter()
            .map(|t| {
                let dihedral = matches!(
                    &t.kind,
                    TargetKind::Geometric { name, .. } if name.eq_ignore_ascii_case("dihedral")
                );
                dihedral && ((t.end - t.start).abs() - 360.0).abs() < 1e-9
            })
            .collect();
        match self.grid.kind() {
            GridKind::Product => full_turn,
            GridKind::Joint => vec![full_turn.iter().all(|p| *p)],
        }
    }

    /// The pairs of points that are one point of a periodic dimension (index 0 and the last).
    pub(crate) fn periodic_copies(&self) -> Vec<(usize, usize)> {
        let periodic = self.periodic_dims();
        let shape = self.grid.shape();
        let mut pairs = Vec::new();
        for p in 0..self.grid.len() {
            let idx = self.grid.indices(p);
            for (d, &is_periodic) in periodic.iter().enumerate() {
                if is_periodic && idx[d] == 0 && shape[d] > 1 {
                    let mut other = idx.clone();
                    other[d] = shape[d] - 1;
                    pairs.push((p, self.grid.point_at(&other)));
                }
            }
        }
        pairs
    }

    /// Whether every relaxed point starts from the last Hessian of the point it starts from
    /// (`hessian=continue`; only where nothing else gives the first Hessian). With ONIOM
    /// microiterations it is the Hessian of the core, as the optimizer of a point leaves it.
    pub(crate) fn continues_hessians(&self) -> bool {
        let o = &self.config.optmain;
        self.config.scan.relax
            && self.config.scan.continue_hessians
            && o.model_function.is_empty()
            && o.initial_hessian.is_none()
            && o.settings.exact_hessian_interval.is_none()
            && o.settings.model_hessian.is_none()
    }

    /// One line of the log for a point.
    pub(crate) fn describe_point(&self, point: usize) -> String {
        let values: Vec<String> = self
            .grid
            .axes()
            .iter()
            .zip(self.grid.values(point))
            .map(|(a, v)| format!("{} = {v:.4}", a.label))
            .collect();
        values.join(", ")
    }
}

/// The input structure or point of `config`.
pub(crate) fn surface_of(
    config: &RelaxedscanConfig,
    objectives: &ObjectiveRegistry,
) -> MoptResult<Surface> {
    if objectives.contains(&config.optmain.calculator) {
        let objective = objectives.create(
            &config.optmain.calculator,
            &config.optmain.calculator_options,
        )?;
        let vars = read_variables_file(&config.input)?;
        crate::pes::objective::check_dimension(objective.as_ref(), vars.values.len())?;
        let names = vars
            .names
            .clone()
            .unwrap_or_else(|| objective.variable_names(vars.values.len()));
        return Ok(Surface::Plain {
            objective,
            names,
            start: Col::from_fn(vars.values.len(), |i| vars.values[i]),
            json: vars.json,
        });
    }
    let spin = config.optmain.state.of_file(&config.input)?;
    let start = read_xyz_file(&config.input, spin, None).context(config.input.display())?;
    let bonds = bond_graph(&start);
    Ok(Surface::Atoms { start, bonds })
}

/// The index of every target that is a variable (plain variables), with the checks that the
/// kinds of the targets suit the surface.
pub(crate) fn variable_indices(
    targets: &[ScanTarget],
    surface: &Surface,
) -> MoptResult<Vec<Option<usize>>> {
    targets
        .iter()
        .map(|t| match (&t.kind, surface) {
            (TargetKind::Variable(v), Surface::Plain { names, .. }) => {
                let by_number = v
                    .parse::<usize>()
                    .ok()
                    .filter(|k| (1..=names.len()).contains(k));
                by_number
                    .map(|k| k - 1)
                    .or_else(|| names.iter().position(|n| n == v))
                    .map(Some)
                    .ok_or_else(|| {
                        MoptError::config(format!(
                            "-scan var: '{v}' is not a variable (1 to {}, or one of {})",
                            names.len(),
                            names.join(", ")
                        ))
                    })
            }
            (TargetKind::Variable(_), Surface::Atoms { .. }) => Err(MoptError::config(
                "-scan var is for objectives of plain variables (-os mb2d, expr:...)",
            )),
            (
                TargetKind::Geometric { .. } | TargetKind::Expression(_) | TargetKind::Cell(_),
                Surface::Plain { .. },
            ) => Err(MoptError::config(format!(
                "-scan {}: the coordinate needs atoms; an objective of plain variables \
                     scans var or resper",
                t.kind_name()
            ))),
            _ => Ok(None),
        })
        .collect()
}

/// Run grid point `point` (from the structure of `from` when given, else as the scan
/// chooses), record its structure; `None` for a skipped point (it is recorded as such).
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_point(
    scan: &Scan<'_>,
    point: usize,
    from: Option<usize>,
    registry: &CalculatorRegistry,
    console: Option<Level>,
    state: &mut ScanState,
    structures: &mut HashMap<usize, Structure>,
) -> MoptResult<Option<(ScanPoint, PointOutcome)>> {
    let config = scan.config;
    let name = scan.point_name(point);
    let file = format!(
        "{}_RelaxedScan_{}.{}",
        scan.stem,
        point + 1,
        scan.surface.extension()
    );
    let pre = scan.preimage_of(point)?;
    let extrapolated = pre.as_ref().is_some_and(|p| p.extrapolated);
    let outside = config.scan.resper.as_ref().map(|r| r.outside);
    if extrapolated && outside == Some(Outside::Skip) && from.is_none() {
        scan.log.info(&format!(
            "point {}: {} is beyond the references; skipped (resper_outside=skip).",
            point + 1,
            scan.describe_point(point)
        ));
        state.record(ScanPoint {
            point,
            energy: f64::NAN,
            biased: f64::NAN,
            converged: false,
            iterations: 0,
            start_from: None,
            file: String::new(),
            extrapolated,
            skipped: true,
            failed: false,
            preimage_error: pre.as_ref().map(|p| p.error),
            repaired: false,
        });
        return Ok(None);
    }
    let attempt = (|| -> MoptResult<(Option<usize>, PointOutcome)> {
        let done: Vec<usize> = state
            .done
            .iter()
            .filter(|p| !p.skipped && structures.contains_key(&p.point))
            .map(|p| p.point)
            .collect();
        let start_from = match from {
            Some(f) => Some(f),
            None if config.scan.first_only || !config.scan.relax => None,
            None if config
                .scan
                .resper
                .as_ref()
                .is_some_and(|r| r.start_from_preimage) =>
            {
                None
            }
            None => scan.grid.start_of(point, &done),
        };
        let start = match start_from.and_then(|f| structures.get(&f)) {
            Some(s) => s.clone(),
            None if !config.scan.relax => scan.rigid_structure(point, pre.as_ref())?,
            None => match (&pre, &scan.surface) {
                (Some(p), Surface::Atoms { start, .. }) => {
                    Structure::Atoms(start.with_flat_coords_bohr(&p.x)?)
                }
                (Some(p), Surface::Plain { .. }) => Structure::Variables(p.x.clone()),
                (None, _) => scan.surface.start(),
            },
        };
        let dir = match from {
            Some(f) => scan.dir.join(format!("{name}_from_{}", f + 1)),
            None => scan.dir.join(&name),
        };
        let point_stem = format!("{}_{}", scan.stem, name);
        let stop_file = scan.dir.join(STOP_FILE);
        let outcome = match (&scan.surface, &start) {
            (Surface::Atoms { .. }, Structure::Atoms(s)) => {
                let points = MoleculePoints {
                    base: &config.optmain,
                    criteria_flags: config.criteria_flags,
                    relax: config.scan.relax,
                    stop_file,
                    console: console.filter(|_| config.optmain.debug),
                    registry,
                };
                let mut constraints = if config.scan.relax {
                    scan.molecule_constraints(point)?
                } else {
                    PointConstraints::default()
                };
                if scan.continues_hessians() {
                    constraints.keep_hessian = Some(dir.join(FINAL_HESSIAN));
                    constraints.first_hessian = start_from
                        .and_then(|f| state.result(f))
                        .map(|r| scan.point_dir(r).join(FINAL_HESSIAN))
                        .filter(|p| p.is_file());
                }
                points.run(s, &constraints, &dir, &point_stem)?
            }
            (
                Surface::Plain {
                    objective,
                    names,
                    json,
                    ..
                },
                Structure::Variables(x),
            ) => {
                let points = VariablePoints {
                    base: &config.optmain,
                    objective: objective.as_ref(),
                    names,
                    json: *json,
                    relax: config.scan.relax,
                    stop_file,
                };
                points.run(x, scan.variable_constraints(point)?, &dir, &point_stem)?
            }
            _ => {
                return Err(MoptError::config(
                    "a structure of the wrong kind for the scan",
                ))
            }
        };
        Ok((start_from, outcome))
    })();
    let (start_from, outcome) = match attempt {
        Ok(done) => done,
        Err(e) if recoverable(&e) => {
            scan.log.warn(&format!(
                "point {} ({}) failed: {e}. The scan goes on without it.",
                point + 1,
                scan.describe_point(point)
            ));
            if from.is_none() {
                state.record(ScanPoint {
                    point,
                    energy: f64::NAN,
                    biased: f64::NAN,
                    converged: false,
                    iterations: 0,
                    start_from: None,
                    file: String::new(),
                    extrapolated,
                    skipped: true,
                    failed: true,
                    preimage_error: pre.as_ref().map(|p| p.error),
                    repaired: false,
                });
            }
            return Ok(None);
        }
        Err(e) => return Err(e),
    };
    let values: Vec<String> = scan
        .grid
        .axes()
        .iter()
        .zip(scan.grid.values(point))
        .map(|(a, v)| format!("{}={v}", a.label.replace(' ', "")))
        .collect();
    let comment = format!("RelaxedScan_{} {}", point + 1, values.join(" "));
    std::fs::write(
        scan.dir.join(&file),
        structure_text(&scan.surface, &outcome.structure, &comment, outcome.biased),
    )?;
    structures.insert(point, outcome.structure.clone());
    let result = ScanPoint {
        point,
        energy: outcome.energy,
        biased: outcome.biased,
        converged: outcome.converged,
        iterations: outcome.iterations,
        start_from,
        file,
        extrapolated,
        skipped: false,
        failed: false,
        preimage_error: if config.scan.relax {
            None
        } else {
            pre.as_ref().map(|p| p.error)
        },
        repaired: from.is_some(),
    };
    Ok(Some((result, outcome)))
}

/// Whether a failure of one point leaves the scan able to go on: a calculation, a
/// constraint, or a solver that failed there (not a setting or a file).
fn recoverable(e: &MoptError) -> bool {
    matches!(
        e,
        MoptError::Calculator(_)
            | MoptError::Convergence(_)
            | MoptError::Coordinate(_)
            | MoptError::Linalg(_)
    )
}
