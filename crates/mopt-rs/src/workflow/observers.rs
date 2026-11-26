//! Observers of optimization runs: progress logging, MultiOptPy's run-directory files, and
//! shape conditions (`-sc`).
//!
//! MultiOptPy: `optimization.py` (`RunLogger`, `ResultPaths`) and `fileio.py:FileIO`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::fmt::Write as _;
use std::fs::{self, OpenOptions};
use std::io::Write as _;
use std::path::{Path, PathBuf};

use super::driver::{Observer, Outcome};
use super::trajectories::GrowingTrajectory;
use crate::coord::connectivity::{any_violated, ShapeCondition};
use crate::core_infra::log::{python_float, Logger};
use crate::core_infra::viz::color::Rgb;
use crate::core_infra::viz::figure::{save_single_line_plot, PlotFigure, PlotSeries, PlotStyle};
use crate::domain::{MolecularSystem, PeriodicCell, Units};
use crate::error::MoptResult;
use crate::math::linalg::col_rms;
use crate::opt::convergence::ConvergenceCriteria;
use crate::opt::state_machine::{IterationReport, Termination};

/// One row of the run log's Cartesian layout.
pub(crate) fn coordinate_row(label: &str, v: [f64; 3]) -> String {
    crate::io::xyz::multioptpy_row(label, "   ", v)
}

/// Lattice vectors in the same Cartesian layout as the atoms above them.
pub(crate) fn lattice_block(cell: &PeriodicCell) -> String {
    let mut s = String::from("LATTICE VECTORS (unit ang.):\n");
    for (k, label) in ["a", "b", "c"].into_iter().enumerate() {
        let v = cell
            .lattice_vector(k)
            .value()
            .map(|v| v * Units::BOHR_TO_ANGSTROM);
        s.push_str(&coordinate_row(label, v));
    }
    let flags = cell.periodic_axes.map(|axis| if axis { "T" } else { "F" });
    let _ = writeln!(s, "pbc=\"{}\"", flags.join(" "));
    s
}

/// The structure as MultiOptPy prints it: element and Cartesian coordinates in Angstrom, one
/// atom per line, followed for a periodic system by its current cell.
pub fn geometry_block(sys: &MolecularSystem) -> String {
    labelled_geometry_blocks(sys, &[])
}

/// The structures held one after the other in `sys` (the two images of a pair as one
/// structure of 2N atoms), each under its label and as [`geometry_block`] prints it.
pub fn labelled_geometry_blocks(sys: &MolecularSystem, labels: &[String]) -> String {
    let parts = labels.len().max(1);
    let size = sys.n_atoms() / parts;
    let labelled = !labels.is_empty() && size > 0 && size * parts == sys.n_atoms();
    let mut s = String::new();
    for (k, atom) in sys.atoms().enumerate() {
        if labelled && k % size == 0 {
            if k > 0 {
                s.push('\n');
            }
            let _ = writeln!(s, "{}", labels[k / size]);
        }
        let p = sys
            .position(atom)
            .value()
            .map(|c| c * Units::BOHR_TO_ANGSTROM);
        s.push_str(&coordinate_row(sys.element(atom).symbol(), p));
    }
    s.push('\n');
    if let Some(cell) = sys.active_cell() {
        s.push_str(&lattice_block(cell));
    }
    s
}

/// Rule above and below the summary of a step.
const RULE: &str =
    "==================================================================================";

/// Per-iteration console / file output in MultiOptPy's format.
#[derive(Debug)]
pub struct ProgressLog {
    logger: Logger,
    criteria: ConvergenceCriteria,
    previous: Option<(f64, f64)>,
    parts: Vec<String>,
    geometry_atoms: Option<Vec<crate::domain::AtomId>>,
}

impl ProgressLog {
    /// Report to `logger`; `criteria` gives the force thresholds shown.
    pub fn new(logger: Logger, criteria: ConvergenceCriteria) -> Self {
        Self {
            logger,
            criteria,
            previous: None,
            parts: Vec::new(),
            geometry_atoms: None,
        }
    }

    /// Limit Cartesian geometry blocks in the log to these atoms.
    pub fn select_geometry_atoms(&mut self, atoms: Option<Vec<crate::domain::AtomId>>) {
        self.geometry_atoms = atoms;
    }

    /// Format the Cartesian geometry that this progress log displays.
    pub fn geometry(&self, sys: &MolecularSystem) -> MoptResult<String> {
        match &self.geometry_atoms {
            Some(ids) => Ok(geometry_block(&sys.trajectory_atoms(ids)?)),
            None => Ok(labelled_geometry_blocks(sys, &self.parts)),
        }
    }

    /// Show the structure of every iteration as several structures of the same atoms one
    /// after the other, each under its label (MultiOptPy's `State1` and `State2` of the two
    /// images of BITSS, which it holds as one structure of 2N atoms).
    #[must_use]
    pub fn in_parts(mut self, labels: &[&str]) -> Self {
        self.parts = labels.iter().map(|l| (*l).to_string()).collect();
        self
    }
}

impl Observer for ProgressLog {
    fn on_iteration(&mut self, r: &IterationReport) -> MoptResult<()> {
        self.on_iteration_with_hessian(r, None)
    }

    fn on_iteration_with_hessian(
        &mut self,
        r: &IterationReport,
        update: Option<(&str, &faer::Mat<f64>)>,
    ) -> MoptResult<()> {
        let e = r.energy.value();
        let be = r.biased_energy.value();
        // MultiOptPy starts from a previous energy of zero: the shift of the first iteration
        // is the energy itself.
        let (pe, pbe) = self.previous.unwrap_or((0.0, 0.0));
        self.previous = Some((e, be));
        let c = &r.convergence;
        let mut s = String::new();
        let _ = writeln!(s, "# ITR. {}", r.iteration);
        let _ = writeln!(s, "{}", self.geometry(&r.system)?);
        if let Some(c) = &r.coordinates {
            // The same structure in the coordinates of the optimization, in atomic units.
            let _ = writeln!(
                s,
                "COORDINATES ({}, NUMBER OF VALUES: {}, unit a.u.):",
                c.system,
                c.values.len()
            );
            for (n, value) in c.values.iter().enumerate() {
                let label = match &c.labels {
                    Some(labels) if n < labels.len() => labels[n].clone(),
                    _ => format!("q{}", n + 1),
                };
                let _ = writeln!(s, "{label:<23}: {value:>15.12} ");
            }
            s.push_str("\n\n");
        }
        for (_, text) in &r.program_outputs {
            // The output of the program itself, as it printed it.
            let _ = writeln!(s, "{}", text.trim_end_matches('\n'));
        }
        for (label, value) in &r.components {
            let _ = writeln!(s, "{label:<23}: {:>15.12} ", value.value());
        }
        for spin in &r.spin_squares {
            let _ = writeln!(
                s,
                "{:<23}: {:>15.12}  (S(S+1) = {:.4}; {})",
                "<S**2>", spin.value, spin.exact, spin.label
            );
        }
        // The messages of the inner solvers go between the two halves at debug level:
        // always into the log file, onto the screen with -debug.
        if !r.debug_notes.is_empty() {
            self.logger.info(s.trim_end_matches('\n'));
            self.logger.debug(&r.debug_notes.join("\n"));
            s.clear();
        }
        for note in &r.notes {
            let _ = writeln!(s, "{note}");
        }
        // One spectrum per iteration: that of the exact Hessian of this iteration where there
        // is one (the rigid motions taken out), that of the updated Hessian otherwise.
        let exact = r.hessian_eigenvalues.is_some();
        let updated_values = update.and_then(|(method, hessian)| {
            let _ = writeln!(s, "Hessian update: {method}");
            if exact {
                return None;
            }
            match hessian.self_adjoint_eigenvalues(faer::Side::Lower) {
                Ok(mut values) => {
                    values.retain(|v| v.abs() > 1e-10);
                    values.sort_by(f64::total_cmp);
                    Some(values)
                }
                Err(err) => {
                    self.logger
                        .warn(&format!("Updated Hessian eigenvalues unavailable: {err:?}"));
                    None
                }
            }
        });
        let _ = writeln!(s, "{RULE}");
        let _ = writeln!(
            s,
            "trust radii (unit. Bohr):  {}",
            python_float(r.trust_radius.value())
        );
        let _ = writeln!(
            s,
            "step  radii (unit. Bohr):  {}",
            python_float(r.step_norm.value())
        );
        if let Some(values) = r.hessian_eigenvalues.as_ref().or(updated_values.as_ref()) {
            let _ = writeln!(
                s,
                "EIGENVALUES (NORMAL COORDINATE, NUMBER OF VALUES: {}):",
                values.len()
            );
            for row in values.chunks(6) {
                let line: Vec<String> = row.iter().map(|v| format!("{v:12.8}")).collect();
                let _ = writeln!(s, "{}", line.join(" "));
            }
        }
        let _ = writeln!(s, "{RULE}");
        let _ = writeln!(s, "caluculation results (unit a.u.):");
        let _ = writeln!(
            s,
            "                         Value                     Threshold "
        );
        let _ = writeln!(s, "ENERGY                 : {e:>15.12} ");
        let _ = writeln!(s, "BIAS  ENERGY           : {be:>15.12} ");
        let row = |name: &str, v: f64, t: f64| {
            format!("{name:<23}: {v:>15.12}                 {t:>15.12} \n")
        };
        s.push_str(&row(
            "Maximum  Force",
            c.max_force.value(),
            self.criteria.max_force.value(),
        ));
        s.push_str(&row(
            "RMS      Force",
            c.rms_force.value(),
            self.criteria.rms_force.value(),
        ));
        s.push_str(&row(
            "Maximum  Displacement",
            c.max_displacement.value(),
            c.max_displacement_threshold.value(),
        ));
        s.push_str(&row(
            "RMS      Displacement",
            c.rms_displacement.value(),
            c.rms_displacement_threshold.value(),
        ));
        let _ = writeln!(s, "ENERGY SHIFT           : {:>15.12} ", e - pe);
        let _ = write!(s, "BIAS ENERGY SHIFT      : {:>15.12} ", be - pbe);
        self.logger.info(&s);
        Ok(())
    }

    fn on_warning(&mut self, message: &str) -> MoptResult<()> {
        self.logger.warn(message);
        Ok(())
    }

    fn on_finish(&mut self, outcome: &Outcome) -> MoptResult<()> {
        let msg = match outcome.termination {
            Some(Termination::Converged) => {
                "\n=====================================================\nconverged!!!\n=====================================================".to_string()
            }
            Some(Termination::MaxIterations) => {
                "Reached maximum number of iterations. This is not converged.".to_string()
            }
            Some(Termination::Dissociated) => {
                "Dissociation is detected. Optimization stopped.".to_string()
            }
            Some(Termination::SeamCheckFailed) => {
                "The check of the branching plane found the structure not stationary on the \
                 seam after every correction of the plane. This is not converged."
                    .to_string()
            }
            None => "The optimization was stopped by a stop condition.".to_string(),
        };
        self.logger.info(&msg);
        self.logger
            .info(&format!("{}\n", self.geometry(&outcome.system)?));
        Ok(())
    }
}

/// Stops the run cleanly when a file named [`STOP_FILE`] appears in the run directory (the
/// file is removed, so that a restart continues).
#[derive(Debug)]
pub struct StopFile {
    path: PathBuf,
    /// A further file that stops the run (that of a command running this one as a job).
    also: Option<PathBuf>,
    logger: Logger,
    stopped: bool,
}

/// Name of the file that stops a run.
pub const STOP_FILE: &str = "STOP";

impl StopFile {
    /// Watch `dir`.
    pub fn new(dir: &Path, logger: Logger) -> Self {
        Self {
            path: dir.join(STOP_FILE),
            also: None,
            logger,
            stopped: false,
        }
    }

    /// Stop also when the file `also` appears (it is removed as well).
    #[must_use]
    pub fn also_watching(mut self, also: Option<PathBuf>) -> Self {
        self.also = also;
        self
    }

    /// Whether the run was stopped.
    pub fn stopped(&self) -> bool {
        self.stopped
    }

    /// Look for the file now (and remove it); true once the run is to stop.
    pub fn check(&mut self) -> bool {
        for path in std::iter::once(&self.path).chain(self.also.iter()) {
            if path.exists() {
                let _ = std::fs::remove_file(path);
                self.logger.info(&format!(
                    "{} found: the run is interrupted.",
                    path.display()
                ));
                self.stopped = true;
            }
        }
        self.stopped
    }
}

impl Observer for StopFile {
    fn on_iteration(&mut self, _report: &IterationReport) -> MoptResult<()> {
        Ok(())
    }

    fn should_stop(&mut self, _current: &MolecularSystem) -> bool {
        self.check()
    }
}

/// `-gi`: distances, angles, and dihedral angles between chosen atoms at every iteration
/// (`geometry_info.csv`), and for each of the atoms the cosine between the gradient of the
/// bias potentials and that of the surface without them (`plot_orthogonality_<atom>.png`).
///
/// MultiOptPy lists every ordered triple and quadruple of the atoms, so each angle twice
/// and each dihedral angle twice among its 24 orders; here each coordinate is written once
/// (angles `a-b-c` with `a < c`, dihedral angles with the first atom below the last).
#[derive(Debug)]
pub struct GeometryInfo {
    atoms: Vec<crate::domain::AtomId>,
    dir: PathBuf,
    iterations: Vec<f64>,
    cosines: Vec<Vec<f64>>,
}

impl GeometryInfo {
    /// For the atoms `atoms` (nothing is written for fewer than two), files in `dir`.
    pub fn new(atoms: Vec<crate::domain::AtomId>, dir: &Path) -> Self {
        Self {
            cosines: vec![Vec::new(); atoms.len()],
            atoms,
            dir: dir.to_path_buf(),
            iterations: Vec::new(),
        }
    }

    /// Names and values (Angstrom, degrees) of the coordinates of `sys`.
    fn coordinates(&self, sys: &MolecularSystem) -> (Vec<String>, Vec<f64>) {
        let at =
            |a: crate::domain::AtomId| sys.position(a).value().map(|c| c * Units::BOHR_TO_ANGSTROM);
        let sub = |p: [f64; 3], q: [f64; 3]| [p[0] - q[0], p[1] - q[1], p[2] - q[2]];
        let dot = |p: [f64; 3], q: [f64; 3]| p[0] * q[0] + p[1] * q[1] + p[2] * q[2];
        let cross = |p: [f64; 3], q: [f64; 3]| {
            [
                p[1] * q[2] - p[2] * q[1],
                p[2] * q[0] - p[0] * q[2],
                p[0] * q[1] - p[1] * q[0],
            ]
        };
        let number = |a: crate::domain::AtomId| a.index() + 1;
        let (mut names, mut values) = (Vec::new(), Vec::new());
        let n = self.atoms.len();
        for i in 0..n {
            for j in i + 1..n {
                let (a, b) = (self.atoms[i], self.atoms[j]);
                let d = sub(at(a), at(b));
                names.push(format!("Distance ({}-{})  [ang.]", number(a), number(b)));
                values.push(libm::sqrt(dot(d, d)));
            }
        }
        for b in 0..n {
            for a in 0..n {
                for c in a + 1..n {
                    if a == b || c == b {
                        continue;
                    }
                    let (pa, pb, pc) = (self.atoms[a], self.atoms[b], self.atoms[c]);
                    let (u, v) = (sub(at(pa), at(pb)), sub(at(pc), at(pb)));
                    let cosine = dot(u, v) / libm::sqrt(dot(u, u) * dot(v, v));
                    names.push(format!(
                        "Bond_angle ({}-{}-{}) [deg.]",
                        number(pa),
                        number(pb),
                        number(pc)
                    ));
                    values.push(libm::acos(cosine.clamp(-1.0, 1.0)).to_degrees());
                }
            }
        }
        for a in 0..n {
            for b in 0..n {
                for c in 0..n {
                    for d in a + 1..n {
                        if a == b || a == c || b == c || b == d || c == d {
                            continue;
                        }
                        let p: [crate::domain::AtomId; 4] =
                            [self.atoms[a], self.atoms[b], self.atoms[c], self.atoms[d]];
                        let (b1, b2, b3) = (
                            sub(at(p[1]), at(p[0])),
                            sub(at(p[2]), at(p[1])),
                            sub(at(p[3]), at(p[2])),
                        );
                        let (n1, n2) = (cross(b1, b2), cross(b2, b3));
                        let along = libm::sqrt(dot(b2, b2));
                        let angle = libm::atan2(dot(cross(n1, n2), b2) / along, dot(n1, n2));
                        names.push(format!(
                            "Dihedral_angle ({}-{}-{}-{}) [deg.]",
                            number(p[0]),
                            number(p[1]),
                            number(p[2]),
                            number(p[3])
                        ));
                        values.push(angle.to_degrees());
                    }
                }
            }
        }
        (names, values)
    }
}

impl Observer for GeometryInfo {
    fn on_iteration(&mut self, r: &IterationReport) -> MoptResult<()> {
        self.record(r.iteration, &r.system, &r.raw_gradient, &r.gradient)
    }

    fn on_finish(&mut self, _outcome: &Outcome) -> MoptResult<()> {
        self.finish()
    }
}

impl GeometryInfo {
    /// The coordinates of `sys` at iteration (or step) `iteration`, and the cosines between
    /// the gradient of the bias potentials (`gradient - raw_gradient`) and `raw_gradient` on
    /// the atoms.
    pub fn record(
        &mut self,
        iteration: usize,
        sys: &MolecularSystem,
        raw_gradient: &faer::Col<f64>,
        gradient: &faer::Col<f64>,
    ) -> MoptResult<()> {
        if self.atoms.len() < 2 {
            return Ok(());
        }
        let (names, values) = self.coordinates(sys);
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.dir.join("geometry_info.csv"))?;
        if self.iterations.is_empty() {
            writeln!(file, "{}", names.join(","))?;
        }
        let row: Vec<String> = values.iter().map(|v| python_float(*v)).collect();
        writeln!(file, "{}", row.join(","))?;
        // Cosine between the gradient of the bias potentials and that of the surface on
        // each atom; 2 where one of them vanishes (MultiOptPy's mark).
        for (atom, list) in self.atoms.iter().zip(&mut self.cosines) {
            let raw: [f64; 3] = std::array::from_fn(|c| raw_gradient[atom.dof(c)]);
            let bias: [f64; 3] = std::array::from_fn(|c| gradient[atom.dof(c)] - raw[c]);
            let norm = |v: [f64; 3]| libm::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
            let (nb, nr) = (norm(bias), norm(raw));
            list.push(if nb == 0.0 || nr == 0.0 {
                2.0
            } else {
                (bias[0] * raw[0] + bias[1] * raw[1] + bias[2] * raw[2]) / (nb * nr)
            });
        }
        self.iterations.push(iteration as f64);
        Ok(())
    }

    /// The figures of the cosines (`plot_orthogonality_<atom>.png`).
    pub fn finish(&mut self) -> MoptResult<()> {
        if self.atoms.len() < 2 || self.iterations.is_empty() {
            return Ok(());
        }
        for (atom, list) in self.atoms.iter().zip(&self.cosines) {
            let number = atom.index() + 1;
            save_single_line_plot(
                self.dir.join(format!("plot_orthogonality_{number}.png")),
                &number.to_string(),
                "ITR. ",
                "cos",
                &self.iterations,
                list,
            )?;
        }
        Ok(())
    }
}

/// `-negeigval`: stops a saddle-point search whose Hessian at the first iteration has no
/// negative eigenvalue (MultiOptPy `judge_early_stop_due_to_no_negative_eigenvalues`).
#[derive(Debug)]
pub struct NegativeEigenvalueGuard {
    saddle_order: usize,
    dir: PathBuf,
    logger: Logger,
    stopped: bool,
}

impl NegativeEigenvalueGuard {
    /// Watch a search for a saddle point of order `saddle_order`; the note goes to `dir`.
    pub fn new(saddle_order: usize, dir: &Path, logger: Logger) -> Self {
        Self {
            saddle_order,
            dir: dir.to_path_buf(),
            logger,
            stopped: false,
        }
    }

    /// Whether the run was stopped.
    pub fn stopped(&self) -> bool {
        self.stopped
    }
}

impl Observer for NegativeEigenvalueGuard {
    fn on_iteration(&mut self, r: &IterationReport) -> MoptResult<()> {
        const MESSAGE: &str =
            "No negative eigenvalues detected while saddle_order > 0. Stopping optimization.";
        if r.iteration != 0 || self.saddle_order == 0 {
            return Ok(());
        }
        if let Some(values) = &r.hessian_eigenvalues {
            if !values.iter().any(|v| *v < -1e-10) {
                self.logger.info(MESSAGE);
                fs::write(
                    self.dir.join("no_negative_eigenvalues_detected.txt"),
                    MESSAGE,
                )?;
                self.stopped = true;
            }
        }
        Ok(())
    }

    fn should_stop(&mut self, _current: &MolecularSystem) -> bool {
        self.stopped
    }
}

/// Stops the run when a `-sc` shape condition is violated.
#[derive(Debug)]
pub struct ShapeGuard {
    conditions: Vec<ShapeCondition>,
    logger: Logger,
    violated: bool,
}

impl ShapeGuard {
    /// Guard for `conditions`.
    pub fn new(conditions: Vec<ShapeCondition>, logger: Logger) -> Self {
        Self {
            conditions,
            logger,
            violated: false,
        }
    }

    /// Whether a condition was violated.
    pub fn violated(&self) -> bool {
        self.violated
    }
}

impl Observer for ShapeGuard {
    fn on_iteration(&mut self, _report: &IterationReport) -> MoptResult<()> {
        Ok(())
    }

    fn should_stop(&mut self, current: &MolecularSystem) -> bool {
        if self.conditions.is_empty() {
            return false;
        }
        match any_violated(&self.conditions, current) {
            Ok(true) => {
                self.logger
                    .info("Shape conditions are not satisfied. Optimization stopped.");
                self.violated = true;
                true
            }
            Ok(false) => false,
            Err(e) => {
                self.logger.error(&format!("shape condition: {e}"));
                self.violated = true;
                true
            }
        }
    }
}

/// Writes MultiOptPy's run-directory files.
///
/// Per iteration: a frame of `<stem>_traj.xyz` (the evaluated structure, appended, so that
/// the trajectory is there while the run goes on and after it was interrupted; MultiOptPy
/// writes a directory `samples_<stem>_<iter>/` per iteration and the trajectory at the end),
/// and rows of `energy_profile.csv` (`iter,bias,effective,raw`), `gradient_profile.csv`, and
/// `bias_gradient_profile.csv` (RMS gradients), plus `surface_components.csv` for combined
/// surfaces. At the end: `<stem>_optimized.xyz`, `energy_profile_kcalmol.csv`,
/// `approx_TS.txt`, `approx_EQ.txt`, `local_min_grad.txt`, and the plots `energy_plot_.png`,
/// `plot_gradient_.png`, `plot_bias_gradient_.png`.
#[derive(Debug)]
pub struct RunFiles {
    dir: PathBuf,
    stem: String,
    own_trajectory: bool,
    trajectory: Option<GrowingTrajectory>,
    selected_atoms: Option<Vec<crate::domain::AtomId>>,
    full_path: Option<GrowingTrajectory>,
    last: Option<MolecularSystem>,
    energies_kcal: Vec<f64>,
    biased_kcal: Vec<f64>,
    grad_rms: Vec<f64>,
    bias_grad_rms: Vec<f64>,
    component_header: Option<Vec<String>>,
}

fn append(path: &Path, text: &str) -> MoptResult<()> {
    let mut f = OpenOptions::new().create(true).append(true).open(path)?;
    f.write_all(text.as_bytes())?;
    Ok(())
}

/// MultiOptPy's coordinate line: `{el:2}  {x:>17.12f}   {y:>17.12f}   {z:>17.12f}`.
fn coordinate_lines(sys: &MolecularSystem, sep: &str) -> String {
    crate::io::xyz::multioptpy_rows(sys, sep)
}

/// Indices `i` with `v[i]` strictly above (`max`) or below both neighbours (scipy's
/// `argrelextrema` with `order = 1`).
pub fn relative_extrema(v: &[f64], max: bool) -> Vec<usize> {
    if max {
        crate::path::band::strict_maxima(v)
    } else {
        crate::path::band::strict_minima(v)
    }
}

impl RunFiles {
    /// Files for input `stem` in `dir` (created if needed).
    pub fn new(dir: impl Into<PathBuf>, stem: &str) -> MoptResult<Self> {
        let dir = dir.into();
        fs::create_dir_all(&dir)?;
        Ok(Self {
            dir,
            stem: stem.to_string(),
            own_trajectory: true,
            trajectory: None,
            selected_atoms: None,
            full_path: None,
            last: None,
            energies_kcal: Vec::new(),
            biased_kcal: Vec::new(),
            grad_rms: Vec::new(),
            bias_grad_rms: Vec::new(),
            component_header: None,
        })
    }

    /// Limit XYZ frames to selected atoms. A consuming workflow can keep its full path.
    pub fn select_trajectory_atoms(
        &mut self,
        atoms: Vec<crate::domain::AtomId>,
        keep_path: bool,
    ) -> MoptResult<()> {
        self.selected_atoms = Some(atoms);
        if keep_path {
            self.full_path = Some(GrowingTrajectory::start(
                self.dir.join(format!("{}_path.xyz", self.stem)),
            )?);
        }
        Ok(())
    }

    /// Full path for another workflow, or the ordinary trajectory.
    pub fn result_path(&self) -> PathBuf {
        self.full_path
            .as_ref()
            .map_or_else(|| self.trajectory_path(), |t| t.path().to_owned())
    }

    /// Run directory.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Path of `<stem>_optimized.xyz`.
    pub fn optimized_path(&self) -> PathBuf {
        self.dir.join(format!("{}_optimized.xyz", self.stem))
    }

    /// Path of `<stem>_traj.xyz`.
    pub fn trajectory_path(&self) -> PathBuf {
        self.dir.join(format!("{}_traj.xyz", self.stem))
    }

    /// Leave `<stem>_traj.xyz` to the caller (a search with two structures writes one
    /// trajectory per structure and joins them at its end).
    #[must_use]
    pub fn without_trajectory(mut self) -> Self {
        self.own_trajectory = false;
        self
    }

    /// Append the structure of iteration `iter` to the trajectory (started with the first).
    fn write_frame(&mut self, iter: usize, sys: &MolecularSystem) -> MoptResult<()> {
        if !self.own_trajectory {
            return Ok(());
        }
        let frame = |system: &MolecularSystem, label: &str| {
            format!(
                "{}\nFrame {iter}{label}{}\n{}",
                system.n_atoms(),
                lattice_note(system),
                coordinate_lines(system, "  ")
            )
        };
        if let Some(full) = &mut self.full_path {
            full.push(&frame(sys, ""))?;
        }
        let path = self.trajectory_path();
        let trajectory = match &mut self.trajectory {
            Some(t) => t,
            slot => slot.insert(GrowingTrajectory::start(path)?),
        };
        match &self.selected_atoms {
            Some(atoms) => {
                trajectory.push(&frame(&sys.trajectory_atoms(atoms)?, " oniom_layer=high"))
            }
            None => trajectory.push(&frame(sys, "")),
        }
    }

    fn write_extrema(&self, name: &str, v: &[f64], max: bool) -> MoptResult<()> {
        let text: String = relative_extrema(v, max)
            .into_iter()
            .map(|i| format!("{i}\n"))
            .collect();
        fs::write(self.dir.join(format!("{name}.txt")), text)?;
        Ok(())
    }
}

/// The cell of a periodic structure for the comment line of its file (extended XYZ:
/// ` Lattice="..." pbc="..."`), so that the file can be read back with its cell; nothing for
/// a structure without one.
fn lattice_note(sys: &MolecularSystem) -> String {
    sys.active_cell()
        .map_or_else(String::new, |cell| format!(" {}", cell.to_extxyz_header()))
}

impl Observer for RunFiles {
    fn on_iteration(&mut self, r: &IterationReport) -> MoptResult<()> {
        let it = r.iteration;
        self.write_frame(it, &r.system)?;
        let (e, be) = (r.energy.value(), r.biased_energy.value());
        let g_rms = col_rms(&r.raw_gradient);
        let bg_rms = col_rms(&r.gradient);
        let csv = self.dir.join("energy_profile.csv");
        let grad = self.dir.join("gradient_profile.csv");
        let bgrad = self.dir.join("bias_gradient_profile.csv");
        if it == 0 {
            fs::write(&csv, "iter,bias,effective,raw\n")?;
            fs::write(&grad, "gradient (RMS) [hartree/Bohr] \n")?;
            fs::write(&bgrad, "bias gradient (RMS) [hartree/Bohr] \n")?;
        }
        append(&csv, &format!("{it},{be},{be},{e}\n"))?;
        append(&grad, &format!("{g_rms}\n"))?;
        append(&bgrad, &format!("{bg_rms}\n"))?;
        if r.components.len() > 1 {
            let path = self.dir.join("surface_components.csv");
            let names: Vec<String> = r.components.iter().map(|(k, _)| k.clone()).collect();
            if self.component_header.as_ref() != Some(&names) {
                append(&path, &format!("iter,{}\n", names.join(",")))?;
                self.component_header = Some(names);
            }
            let values: Vec<String> = r
                .components
                .iter()
                .map(|(_, v)| v.value().to_string())
                .collect();
            append(&path, &format!("{it},{}\n", values.join(",")))?;
        }
        self.last = Some(r.system.clone());
        self.energies_kcal.push(e * Units::HARTREE_TO_KCAL_MOL);
        self.biased_kcal.push(be * Units::HARTREE_TO_KCAL_MOL);
        self.grad_rms.push(g_rms);
        self.bias_grad_rms.push(bg_rms);
        Ok(())
    }

    fn on_finish(&mut self, _outcome: &Outcome) -> MoptResult<()> {
        if self.own_trajectory && self.trajectory.is_none() {
            // A run without an iteration still leaves its (empty) trajectory.
            self.trajectory = Some(GrowingTrajectory::start(self.trajectory_path())?);
        }
        if let Some(last) = &self.last {
            let text = format!(
                "{}\nOptimizedStructure{}\n{}",
                last.n_atoms(),
                lattice_note(last),
                coordinate_lines(last, "  ")
            );
            fs::write(self.optimized_path(), text)?;
        }
        let first = self.energies_kcal.first().copied().unwrap_or(0.0);
        let mut kcal = String::from("ITER.,energy[kcal/mol]\n");
        for (i, e) in self.energies_kcal.iter().enumerate() {
            let _ = writeln!(kcal, "{i},{}", e - first);
        }
        fs::write(self.dir.join("energy_profile_kcalmol.csv"), kcal)?;
        self.write_extrema("approx_TS", &self.energies_kcal, true)?;
        self.write_extrema("approx_EQ", &self.energies_kcal, false)?;
        self.write_extrema("local_min_grad", &self.grad_rms, false)?;

        let xs: Vec<f64> = (0..self.energies_kcal.len()).map(|i| i as f64).collect();
        let mut fig = PlotFigure::new("Energy Profile", "ITR.", "Electronic Energy [kcal/mol]");
        fig.add_series(PlotSeries {
            label: "Normal Energy".into(),
            xs: xs.clone(),
            ys: self.energies_kcal.clone(),
            color: Rgb::EMERALD,
            style: PlotStyle::LinePoints,
        });
        fig.add_series(PlotSeries {
            label: "Bias Energy".into(),
            xs: xs.clone(),
            ys: self.biased_kcal.clone(),
            color: Rgb::BLUE,
            style: PlotStyle::LinePoints,
        });
        fig.save(self.dir.join("energy_plot_.png"))?;
        save_single_line_plot(
            self.dir.join("plot_gradient_.png"),
            "gradient",
            "ITR. ",
            "gradient (RMS) [a.u.]",
            &xs,
            &self.grad_rms,
        )?;
        save_single_line_plot(
            self.dir.join("plot_bias_gradient_.png"),
            "bias_gradient",
            "ITR. ",
            "bias gradient (RMS) [a.u.]",
            &xs,
            &self.bias_grad_rms,
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extrema_like_scipy() {
        let v = [0.0, 1.0, 0.5, 0.5, 2.0, 1.0, 3.0];
        assert_eq!(relative_extrema(&v, true), vec![1, 4]);
        assert_eq!(relative_extrema(&v, false), vec![5]);
    }
}
