//! The files and the log of an IRC calculation, written from the events of the state
//! machine.
//!
//! As MultiOptPy: `irc_forward/` and `irc_backward/` (a meta-IRC and ModeKill: the run
//! directory itself) with `irc_energies_gradients.csv` and `irc_structures.xyz`
//! (`modekill_*` for ModeKill), `irc_curvature_properties.csv` for Euler and RK4, the plots
//! of the bending angles and of the DVV time steps, and at the end
//! `irc_combined_path.xyz`, `irc_combined_data.csv`, and
//! `<stem>_irc_endpoint_{1,2}.xyz`. New: `irc_profile.csv` (the energies over the signed
//! mass-weighted length of the path, one row per frame of the combined path) and
//! `irc_first_step.csv` (the test of the first points). For plain variables the points are
//! written to CSV files instead of xyz files.
//!
//! The combined path runs from the end of the forward side through the starting structure
//! to the end of the backward side, the combined table the other way round, as in
//! MultiOptPy. MultiOptPy writes the plots of a side into the run directory under a name
//! that starts with the name of the side (`irc_forwardplot_...`, a missing path separator);
//! here they are in the directory of the side.
//!
//! MultiOptPy: `irc.py:IRC` (`combine_xyz_files`, `combine_csv_data`, `write_xyz`) and the
//! file handling of `IRC/lqa.py:LQA` (`create_csv_file`, `save_to_csv`,
//! `save_xyz_structure`) and its siblings.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use faer::Col;

use super::irc_combined::{
    combined_path, combined_table, field, point_text, SideData, StartPoint, TABLE_COLUMNS,
};
use crate::core_infra::log::{python_float, Logger};
use crate::core_infra::viz::figure::save_single_line_plot;
use crate::domain::{MolecularSystem, Units};
use crate::error::MoptResult;
use crate::irc::{
    Branch, BranchSummary, FirstStepReport, IrcEvent, IrcSpace, StartKind, StartReport, StepCounts,
    StepRecord,
};

/// What the points of the path are.
#[derive(Debug, Clone)]
pub enum PathLayout {
    /// Structures (xyz files).
    Atoms(Box<MolecularSystem>),
    /// Plain variables with these names (CSV files).
    Variables(Vec<String>),
}

/// Name of the combined path of a structure.
pub const COMBINED_PATH_XYZ: &str = "irc_combined_path.xyz";

fn append(path: &Path, text: &str) -> MoptResult<()> {
    use std::io::Write as _;
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    file.write_all(text.as_bytes())?;
    Ok(())
}

impl super::irc::IrcObserver for IrcOutput {
    fn on_event(&mut self, event: &IrcEvent) -> MoptResult<()> {
        match event {
            IrcEvent::Start(report) => self.on_start(report),
            IrcEvent::FirstStep(report) => self.on_first_step(report),
            IrcEvent::Step { branch, record } => self.on_step(*branch, record),
            IrcEvent::BranchFinished(summary) => self.on_side_end(summary),
        }
    }
}

/// One side while it runs.
struct Side {
    dir: PathBuf,
    records: Vec<StepRecord>,
    rows: Vec<Vec<String>>,
    angles: Vec<f64>,
    curvature_started: bool,
    table_started: bool,
}

/// Files and log of one IRC calculation.
pub struct IrcOutput {
    dir: PathBuf,
    stem: String,
    layout: PathLayout,
    space: IrcSpace,
    start: Col<f64>,
    method: String,
    log: Logger,
    kind: Option<StartKind>,
    report: Option<StartReport>,
    sides: HashMap<Branch, Side>,
    order: Vec<Branch>,
    first_steps: Vec<FirstStepReport>,
    endpoints: Vec<PathBuf>,
}

impl IrcOutput {
    /// Output of a calculation from the structure `start` into `dir`; `stem` names the
    /// files of the end points.
    pub fn new(
        dir: &Path,
        stem: &str,
        layout: PathLayout,
        space: IrcSpace,
        start: Col<f64>,
        method: &str,
        log: Logger,
    ) -> Self {
        Self {
            dir: dir.to_path_buf(),
            stem: stem.to_string(),
            layout,
            space,
            start,
            method: method.to_string(),
            log,
            kind: None,
            report: None,
            sides: HashMap::new(),
            order: Vec::new(),
            first_steps: Vec::new(),
            endpoints: Vec::new(),
        }
    }

    /// The files of the end points written so far (the forward side first).
    pub fn endpoints(&self) -> &[PathBuf] {
        &self.endpoints
    }

    /// The combined path (a transition state with atoms), once it is written.
    pub fn combined_path(&self) -> Option<PathBuf> {
        let path = self.dir.join(COMBINED_PATH_XYZ);
        path.exists().then_some(path)
    }

    fn modekill(&self) -> bool {
        self.kind == Some(StartKind::ModeKill)
    }

    fn prefix(&self) -> &'static str {
        if self.modekill() {
            "modekill"
        } else {
            "irc"
        }
    }

    fn side_dir(&self, branch: Branch) -> PathBuf {
        match branch {
            Branch::Forward => self.dir.join("irc_forward"),
            Branch::Backward => self.dir.join("irc_backward"),
            Branch::Single => self.dir.clone(),
        }
    }

    fn banner(&self) -> &'static str {
        match self.method.as_str() {
            "lqa" => "Local Quadratic Approximation method",
            "hpc" => "Hessian-based Predictor-Corrector (HPC) method",
            "dvv" => "Damped Velocity Verlet method",
            "euler" => "Euler method for IRC calculation",
            "rk4" => "Runge-Kutta 4th Order method",
            "gs2" => "Gonzalez-Schlegel second-order method",
            "modekill" => "ModeKill: Selective Imaginary Mode Removal",
            _ => "IRC",
        }
    }

    /// The side `branch`, with its files started.
    fn side(&mut self, branch: Branch) -> MoptResult<&mut Side> {
        if !self.sides.contains_key(&branch) {
            let dir = self.side_dir(branch);
            std::fs::create_dir_all(&dir)?;
            let prefix = self.prefix();
            // The files of an earlier run of the same directory (a restart) start anew.
            let points = match &self.layout {
                PathLayout::Atoms(_) => format!("{prefix}_structures.xyz"),
                PathLayout::Variables(names) => {
                    let name = format!("{prefix}_points.csv");
                    std::fs::write(dir.join(&name), format!("Step,{}\n", names.join(",")))?;
                    name
                }
            };
            if matches!(self.layout, PathLayout::Atoms(_)) {
                std::fs::write(dir.join(points), "")?;
            }
            let _ = std::fs::remove_file(dir.join("irc_curvature_properties.csv"));
            match branch {
                Branch::Forward => self.log.info("Forward IRC"),
                Branch::Backward => self.log.info("Backward IRC"),
                Branch::Single => {}
            }
            self.log.info(self.banner());
            self.order.push(branch);
            self.sides.insert(
                branch,
                Side {
                    dir,
                    records: Vec::new(),
                    rows: Vec::new(),
                    angles: Vec::new(),
                    curvature_started: false,
                    table_started: false,
                },
            );
        }
        match self.sides.get_mut(&branch) {
            Some(side) => Ok(side),
            None => Err(crate::error::MoptError::config("IRC output: unknown side")),
        }
    }

    /// The structure as MultiOptPy prints it at every step.
    fn geometry_lines(&self, x: &Col<f64>) -> Option<String> {
        let PathLayout::Atoms(atoms) = &self.layout else {
            return None;
        };
        let mut text = String::from("\n");
        for (i, element) in atoms.elements.iter().enumerate() {
            text.push_str(&super::observers::coordinate_row(
                element.symbol(),
                std::array::from_fn(|k| x[3 * i + k] * Units::BOHR_TO_ANGSTROM),
            ));
        }
        if let Some(cell) = atoms.active_cell() {
            text.push('\n');
            text.push_str(&super::observers::lattice_block(cell));
        }
        text.pop();
        Some(text)
    }

    fn on_start(&mut self, report: &StartReport) -> MoptResult<()> {
        self.kind = Some(report.kind);
        self.report = Some(report.clone());
        let mut lines = format!(
            "EIGENVALUES (MASS-WEIGHTED COORDINATE, NUMBER OF VALUES: {}):",
            report.eigenvalues.len()
        );
        for chunk in report.eigenvalues.chunks(6) {
            let row: Vec<String> = chunk.iter().map(|v| format!("{v:12.8}")).collect();
            let _ = write!(lines, "\n{}", row.join(" "));
        }
        self.log.info(&lines);
        self.log.info(&format!(
            "Number of imaginary eigenvalues:  {}",
            report.imaginary
        ));
        if let Some(nu) = report.transition_wavenumber {
            self.log.info(&format!(
                "Imaginary mode of the transition state: {nu:.2} cm^-1"
            ));
        }
        if report.kind == StartKind::MetaIrc {
            self.log.info(&format!(
                "The starting structure is not a transition state ({} imaginary mode(s), largest \
                 gradient component {:.6}, RMS gradient {:.6}): one path down the gradient.",
                report.imaginary, report.max_gradient, report.rms_gradient
            ));
        }
        self.log.info(report.kind.message());
        self.log.info("IRC carry out...");
        Ok(())
    }

    fn on_first_step(&mut self, report: &FirstStepReport) -> MoptResult<()> {
        self.side(report.branch)?;
        let t = &report.test;
        self.log.info(&format!(
            "First step ({}): length {:.6}, angle between the downhill direction and the \
             transition vector {:.3} degrees, rho = dE_actual / dE_predicted = {:.4} \
             ({:.6e} / {:.6e})",
            report.branch.label(),
            report.length,
            t.angle_deg,
            t.rho,
            t.actual_change,
            t.predicted_change
        ));
        self.log.info(&if report.corrected {
            format!(
                "First step ({}): outside the limits; replaced by the energy minimum on the \
                 hypersphere around the transition state in {} evaluation(s){}: moved by \
                 {:.6}, energy change {:.6e}",
                report.branch.label(),
                report.iterations,
                if report.converged {
                    ""
                } else {
                    " (not converged)"
                },
                report.moved,
                report.energy_gain
            )
        } else {
            format!(
                "First step ({}): within the limits; no correction.",
                report.branch.label()
            )
        });
        self.first_steps.push(report.clone());
        let mut text = "Side,Length,Angle (deg),Rho,Predicted dE,Actual dE,Corrected,\
                        Evaluations,Converged,Moved,Energy Change\n"
            .to_string();
        for r in &self.first_steps {
            let _ = writeln!(
                text,
                "{},{},{},{},{},{},{},{},{},{},{}",
                r.branch.label(),
                python_float(r.length),
                python_float(r.test.angle_deg),
                python_float(r.test.rho),
                python_float(r.test.predicted_change),
                python_float(r.test.actual_change),
                r.corrected,
                r.iterations,
                r.converged,
                python_float(r.moved),
                python_float(r.energy_gain)
            );
        }
        std::fs::write(self.dir.join("irc_first_step.csv"), text)?;
        Ok(())
    }

    fn on_step(&mut self, branch: Branch, record: &StepRecord) -> MoptResult<()> {
        let prefix = self.prefix();
        let comment = if self.modekill() {
            format!("ModeKill Step {}", record.step)
        } else {
            format!("IRC Step {}", record.step)
        };
        let frame = point_text(&self.layout, &record.x, &record.step.to_string(), &comment);
        let points = match &self.layout {
            PathLayout::Atoms(_) => format!("{prefix}_structures.xyz"),
            PathLayout::Variables(_) => format!("{prefix}_points.csv"),
        };
        let geometry = self.geometry_lines(&record.x);
        let log = self.log.clone();
        let side = self.side(branch)?;
        append(&side.dir.join(points), &frame)?;
        let table = side.dir.join(format!("{prefix}_energies_gradients.csv"));
        if !side.table_started {
            let mut header: Vec<String> = TABLE_COLUMNS.iter().map(|c| (*c).to_string()).collect();
            header.extend(record.extras.iter().map(|(name, _)| (*name).to_string()));
            std::fs::write(&table, header.join(",") + "\n")?;
            side.table_started = true;
        }
        let mut row = vec![
            record.step.to_string(),
            python_float(record.energy),
            python_float(record.biased_energy),
            python_float(record.rms_gradient),
            python_float(record.rms_biased_gradient),
        ];
        row.extend(record.extras.iter().map(|(_, value)| value.clone()));
        if record.in_table {
            let fields: Vec<String> = row.iter().map(|f| field(f)).collect();
            append(&table, &(fields.join(",") + "\n"))?;
            side.rows.push(row);
        }
        if let Some(c) = &record.curvature {
            let path = side.dir.join("irc_curvature_properties.csv");
            if !side.curvature_started {
                let names: Vec<String> = (1..=c.couplings.len())
                    .map(|k| format!("Curvature_Coupling_{k}"))
                    .collect();
                let header = std::iter::once("Scalar_Curvature".to_string()).chain(names);
                std::fs::write(&path, header.collect::<Vec<_>>().join(",") + "\n")?;
                side.curvature_started = true;
            }
            let values: Vec<String> = std::iter::once(c.scalar)
                .chain(c.couplings.iter().copied())
                .map(|v| format!("{v:.6}"))
                .collect();
            append(&path, &(values.join(",") + "\n"))?;
        }
        if let Some(angle) = record.bending_angle {
            side.angles.push(angle);
        }
        side.records.push(record.clone());
        log.info(&format!("# STEP:  {}", record.step));
        for note in &record.notes {
            log.info(note);
        }
        if let Some(angle) = record.bending_angle {
            log.info(&format!("Path bending angle:  {}", python_float(angle)));
        }
        if let Some(text) = geometry {
            log.info(&text);
        }
        let mut lines = format!(
            "\nEnergy         :  {}\nBias Energy    :  {}\nRMS B. grad    :  {}",
            python_float(record.energy),
            python_float(record.biased_energy),
            python_float(record.rms_biased_gradient)
        );
        for (name, value) in &record.extras {
            let _ = write!(lines, "\n{name:<15}:  {value}");
        }
        if let Some(c) = &record.curvature {
            let _ = write!(lines, "\nScalar curvature:  {}", python_float(c.scalar));
        }
        log.info(&lines);
        Ok(())
    }

    fn on_side_end(&mut self, summary: &BranchSummary) -> MoptResult<()> {
        let modekill = self.modekill();
        let log = self.log.clone();
        let side = self.side(summary.branch)?;
        if !side.table_started {
            let name = if modekill { "modekill" } else { "irc" };
            std::fs::write(
                side.dir.join(format!("{name}_energies_gradients.csv")),
                TABLE_COLUMNS.join(",") + "\n",
            )?;
            side.table_started = true;
        }
        log.info(summary.end.message());
        let counts = summary.counts;
        if counts != StepCounts::default() {
            log.info(&format!(
                "Step control ({}): {} point(s) of higher energy and {} point(s) past the floor \
                 of the valley dropped and their steps taken again, {} exact Hessian(s) asked \
                 for after such a point, the step length halved {} time(s) after a point that \
                 was kept, {} step(s) of the final approach.",
                summary.branch.label(),
                counts.dropped,
                counts.overshot,
                counts.refreshed,
                counts.shortened,
                counts.approach
            ));
        }
        if !side.angles.is_empty() {
            let xs: Vec<f64> = (0..side.angles.len()).map(|k| k as f64).collect();
            let name = if modekill {
                "plot_ModeKill_bending_0.png"
            } else {
                "plot_IRC_bending_0.png"
            };
            save_single_line_plot(
                side.dir.join(name),
                "0",
                "# STEP",
                "bending angle [degrees]",
                &xs,
                &side.angles,
            )?;
        }
        let time_steps: Vec<f64> = side
            .records
            .iter()
            .filter_map(|r| {
                r.extras
                    .iter()
                    .find(|(name, _)| *name == "Time Step (fs)")
                    .and_then(|(_, value)| value.parse::<f64>().ok())
            })
            .collect();
        if !time_steps.is_empty() {
            let xs: Vec<f64> = (0..time_steps.len()).map(|k| k as f64).collect();
            save_single_line_plot(
                side.dir.join("plot_DVV_timesteps_0.png"),
                "0",
                "# STEP",
                "Time step (fs)",
                &xs,
                &time_steps,
            )?;
        }
        if modekill {
            log.info("ModeKill calculation finished.");
        }
        Ok(())
    }

    /// The file of an end point (`number` 1: forward or single, 2: backward).
    fn write_endpoint(&mut self, number: usize, record: &StepRecord) -> MoptResult<PathBuf> {
        let path = match &self.layout {
            PathLayout::Atoms(atoms) => {
                let path = self
                    .dir
                    .join(format!("{}_irc_endpoint_{number}.xyz", self.stem));
                // A periodic structure keeps its cell (extended xyz), so that what follows
                // (the optimization of the end) sees the same system.
                let cell = atoms
                    .cell
                    .as_ref()
                    .filter(|c| c.is_any_periodic())
                    .map_or_else(String::new, |c| format!(" {}", c.to_extxyz_header()));
                let mut text = format!("{}\nTerminal structure{cell}\n", atoms.n_atoms());
                for (i, element) in atoms.elements.iter().enumerate() {
                    let _ = writeln!(
                        text,
                        "{}  {:.12} {:.12} {:.12}",
                        element.symbol(),
                        record.x[3 * i] * Units::BOHR_TO_ANGSTROM,
                        record.x[3 * i + 1] * Units::BOHR_TO_ANGSTROM,
                        record.x[3 * i + 2] * Units::BOHR_TO_ANGSTROM
                    );
                }
                std::fs::write(&path, text)?;
                path
            }
            PathLayout::Variables(names) => {
                let path = self
                    .dir
                    .join(format!("{}_irc_endpoint_{number}.txt", self.stem));
                let width = names.iter().map(String::len).max().unwrap_or(0);
                let mut text = format!("# value = {:.12e}\n", record.biased_energy);
                for (name, value) in names.iter().zip(record.x.iter()) {
                    let _ = writeln!(text, "{name:<width$}  {value:.12e}");
                }
                std::fs::write(&path, text)?;
                path
            }
        };
        self.endpoints.push(path.clone());
        Ok(path)
    }

    /// Mass-weighted distance of the first point of a side from the starting structure.
    fn first_length(&self, records: &[StepRecord]) -> f64 {
        records.first().map_or(0.0, |r| {
            let d = self.space.to_mw(&(&r.x - &self.start));
            libm::sqrt(d.iter().map(|v| v * v).sum::<f64>()) - r.arc_length
        })
    }

    /// Write what is written at the end of the calculation: the end points, and for a
    /// transition state the combined path, the combined table, and the energy profile.
    pub fn finish(&mut self) -> MoptResult<()> {
        let take = |sides: &HashMap<Branch, Side>,
                    branch: Branch|
         -> (Vec<StepRecord>, Vec<Vec<String>>) {
            sides
                .get(&branch)
                .map_or_else(Default::default, |s| (s.records.clone(), s.rows.clone()))
        };
        match self.kind {
            Some(StartKind::TransitionState) => {}
            Some(StartKind::MetaIrc) => {
                let (records, _) = take(&self.sides, Branch::Single);
                if let Some(last) = records.last() {
                    let path = self.write_endpoint(1, last)?;
                    self.log.info(&format!(
                        "Meta-IRC terminal structure saved to {}",
                        path.display()
                    ));
                }
                self.log.info("IRC calculation is finished.");
                return Ok(());
            }
            _ => {
                self.log.info("IRC calculation is finished.");
                return Ok(());
            }
        }
        let (forward, forward_rows) = take(&self.sides, Branch::Forward);
        let (backward, backward_rows) = take(&self.sides, Branch::Backward);
        if let Some(last) = forward.last() {
            let path = self.write_endpoint(1, last)?;
            self.log.info(&format!(
                "Forward terminal structure saved to {}",
                path.display()
            ));
        }
        if let Some(last) = backward.last() {
            let path = self.write_endpoint(2, last)?;
            self.log.info(&format!(
                "Backward terminal structure saved to {}",
                path.display()
            ));
        }
        let report = self.report.as_ref();
        let start = StartPoint {
            x: &self.start,
            energy: report.map_or(f64::NAN, |r| r.energy),
            biased_energy: report.map_or(f64::NAN, |r| r.biased_energy),
            rms_gradient: report.map_or(f64::NAN, |r| r.rms_raw_gradient),
            rms_biased_gradient: report.map_or(f64::NAN, |r| r.rms_gradient),
        };
        let forward = SideData {
            records: &forward,
            rows: &forward_rows,
            first_length: self.first_length(&forward),
        };
        let backward = SideData {
            records: &backward,
            rows: &backward_rows,
            first_length: self.first_length(&backward),
        };
        let (path_text, profile) = combined_path(&self.layout, &start, &forward, &backward);
        let combined = match &self.layout {
            PathLayout::Atoms(_) => self.dir.join(COMBINED_PATH_XYZ),
            PathLayout::Variables(_) => self.dir.join("irc_combined_path.csv"),
        };
        std::fs::write(&combined, path_text)?;
        std::fs::write(self.dir.join("irc_profile.csv"), profile)?;
        self.log.info(&format!(
            "Combined IRC path saved to {}",
            combined.display()
        ));
        let data = self.dir.join("irc_combined_data.csv");
        std::fs::write(&data, combined_table(&start, &forward, &backward))?;
        self.log
            .info(&format!("Combined IRC data saved to {}", data.display()));
        self.log.info("IRC calculation is finished.");
        Ok(())
    }
}
