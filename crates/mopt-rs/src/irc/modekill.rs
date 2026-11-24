//! ModeKill: remove chosen imaginary modes of a stationary point by stepping downhill along
//! their eigenvectors until they are no longer imaginary.
//!
//! `METHOD` is `modekill` (every imaginary mode below `nu_thresh` except the lowest, which
//! is taken for the reaction coordinate) or `modekill:I,J,...` (the modes with these
//! numbers, counted from 0 in the order of the eigenvalues of the mass-weighted Hessian, as
//! in MultiOptPy). Further options: `nu_thresh=-5` (cm^-1), `update=off` (the exact Hessian
//! at every cycle), `hess=off` (no Hessian of the last structure).
//!
//! Every cycle: the surface at the structure, the Hessian (exact with `-fc`, otherwise
//! updated), its eigenvectors; the modes to remove are followed from cycle to cycle by
//! their overlap with the previous eigenvectors among the modes that are still negative
//! (overlap above 0.3); the step is the step size along each of them, towards the side on
//! which the gradient descends ([`crate::opt::mode_kill::downhill_step`], shared with the
//! mode kill of the optimizers). The run ends when no mode is left to follow, when the
//! gradient is below the thresholds with no imaginary mode left, or after `MAX_STEP`
//! cycles.
//!
//! Differences to MultiOptPy (`docs/source/migration_paths.md`): the Hessian between exact ones
//! is updated with Bofill's formula (MultiOptPy's BFGS update is skipped along negative
//! curvature, and its default `-fc -1` computes the exact Hessian at every cycle by an
//! accident of `%`); the modes are those of the surface with the bias potentials.
//!
//! MultiOptPy: `IRC/modekill.py:ModeKill`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::sync::Arc;

use faer::{Col, Mat};

use super::convergence::convergence_check;
use super::history::{HessianUpdate, PathHistory, SurfaceModel};
use super::integrator::{BranchAction, BranchStart, Integrator};
use super::settings::{IrcSettings, MethodOptions};
use super::space::IrcSpace;
use super::types::{rms, BranchEnd, PointRequest, PointRole, PointValue, StepRecord};
use crate::core_infra::log::python_float;
use crate::error::{MoptError, MoptResult};
use crate::math::linalg::symmetric_eigen;
use crate::opt::mode_kill::downhill_step;

/// MultiOptPy's wavenumber (cm^-1) below which a mode counts as imaginary in its ModeKill
/// (`nu_thresh`). Not the default here: [`default_nu_threshold`].
pub const MULTIOPTPY_NU_THRESHOLD: f64 = -5.0;

/// The wavenumber (cm^-1) below which ModeKill removes a mode by default: that of the one
/// criterion of an imaginary mode ([`crate::opt::mode_kill::IMAGINARY_CURVATURE`], about
/// -0.51 cm^-1), so that a mode the start of an IRC counts as imaginary is one ModeKill removes
/// (MultiOptPy's -5 cm^-1 left modes between -5 and -0.51 cm^-1 that the start then counted;
/// `modekill:nu_thresh=-5` gives MultiOptPy's).
pub fn default_nu_threshold() -> f64 {
    eigenvalue_to_wavenumber(-crate::opt::mode_kill::IMAGINARY_CURVATURE)
}
/// Smallest overlap with a mode of the previous cycle at which a mode is still the same.
pub const OVERLAP_THRESHOLD: f64 = 0.3;

/// Wavenumber of an eigenvalue of the mass-weighted Hessian (negative: imaginary); the one
/// conversion of the crate, [`crate::domain::units::eigenvalue_to_wavenumber`].
pub use crate::domain::units::eigenvalue_to_wavenumber;

/// A list of numbers as Python prints it (`[-512.3, -48.1]`).
fn python_list(values: &[f64]) -> String {
    let items: Vec<String> = values.iter().map(|v| python_float(*v)).collect();
    format!("[{}]", items.join(", "))
}

/// The ModeKill integrator.
pub struct ModeKill {
    space: Arc<IrcSpace>,
    settings: IrcSettings,
    x: Col<f64>,
    cycle: usize,
    model: SurfaceModel,
    /// Numbers of the modes to remove among the current eigenvectors.
    kill: Vec<usize>,
    /// Whether `kill` still has to be checked against the first Hessian.
    unchecked: bool,
    previous_modes: Option<Mat<f64>>,
    nu_threshold: f64,
    final_hessian: bool,
    update: bool,
    history: PathHistory,
    next: BranchAction,
    /// The end that waits for the Hessian of the last structure.
    pending_end: Option<BranchEnd>,
}

fn wavenumbers(values: &Col<f64>) -> Vec<f64> {
    values
        .iter()
        .map(|v| eigenvalue_to_wavenumber(*v))
        .collect()
}

impl ModeKill {
    /// Remove modes of the structure `start.x`.
    pub fn new(start: BranchStart) -> MoptResult<Self> {
        let options = MethodOptions::parse(&start.settings.method_options);
        options.only("modekill", &["nu_thresh", "update", "hess"], true)?;
        let nu_threshold = options.number("modekill", "nu_thresh", default_nu_threshold())?;
        let update = options.switch("modekill", "update", true)?;
        let final_hessian = options.switch("modekill", "hess", true)?;
        let explicit: Vec<usize> = options
            .bare
            .iter()
            .map(|t| {
                t.parse::<usize>().map_err(|_| {
                    MoptError::parse(format!(
                        "modekill: '{t}' is not the number of a mode (modekill:1,2)"
                    ))
                })
            })
            .collect::<MoptResult<_>>()?;
        let (values, _) = symmetric_eigen(&start.hessian)?;
        let nus = wavenumbers(&values);
        // MultiOptPy: every imaginary mode except the first, which is left for the IRC.
        let kill: Vec<usize> = if explicit.is_empty() {
            (0..nus.len())
                .filter(|&k| nus[k] < nu_threshold)
                .skip(1)
                .collect()
        } else {
            explicit.clone()
        };
        let mut this = Self {
            space: start.space,
            x: start.x,
            cycle: 1,
            model: SurfaceModel::new(start.hessian, HessianUpdate::Bofill),
            kill,
            unchecked: !explicit.is_empty(),
            previous_modes: None,
            nu_threshold,
            final_hessian,
            update,
            history: PathHistory::new(),
            next: BranchAction::Finished(BranchEnd::NothingToRemove),
            pending_end: None,
            settings: start.settings,
        };
        if !this.kill.is_empty() {
            this.next = this.cycle_request();
        }
        Ok(this)
    }

    fn cycle_request(&self) -> BranchAction {
        BranchAction::Evaluate(PointRequest {
            x: self.x.clone(),
            need_hessian: !self.update || self.settings.exact_hessian_at(self.cycle),
            role: PointRole::Path,
            step: self.cycle,
        })
    }

    /// The modes to remove among the eigenvectors `vectors` with the eigenvalues `values`,
    /// and what was found, for the log.
    fn follow(
        &mut self,
        values: &Col<f64>,
        vectors: &Mat<f64>,
        nus: &[f64],
        notes: &mut Vec<String>,
    ) -> MoptResult<()> {
        let n = values.nrows();
        let Some(previous) = &self.previous_modes else {
            if self.unchecked {
                self.unchecked = false;
                if self.kill.iter().any(|&k| k >= n) {
                    // MultiOptPy: numbers out of range fall back to every imaginary mode.
                    notes.push(
                        "Warning: Kill indices out of range. Using available negative modes."
                            .to_string(),
                    );
                    self.kill = (0..n).filter(|&k| nus[k] < self.nu_threshold).collect();
                } else if let Some(&k) = self.kill.iter().find(|&&k| nus[k] >= self.nu_threshold) {
                    return Err(MoptError::config(format!(
                        "ModeKill is intended for removal of imaginary frequencies below {} \
                         cm^-1; mode {k} has {:.2} cm^-1. Please choose different modes.",
                        self.nu_threshold, nus[k]
                    )));
                }
            }
            return Ok(());
        };
        let mut found = Vec::new();
        for &old in &self.kill {
            let mut best: Option<(usize, f64)> = None;
            // Among the modes that are still negative.
            for k in (0..n).filter(|&k| values[k] < 0.0) {
                let overlap: f64 = (0..n)
                    .map(|i| previous[(i, old)] * vectors[(i, k)])
                    .sum::<f64>()
                    .abs();
                if best.is_none_or(|(_, o)| overlap > o) {
                    best = Some((k, overlap));
                }
            }
            if let Some((k, overlap)) = best.filter(|(_, o)| *o > OVERLAP_THRESHOLD) {
                notes.push(format!(
                    "Mode {old} tracked to current mode {k} with overlap {overlap:.4}"
                ));
                if !found.contains(&k) {
                    found.push(k);
                }
            }
        }
        self.kill = found;
        Ok(())
    }

    fn finish(&mut self, end: BranchEnd) {
        if self.final_hessian {
            self.pending_end = Some(end);
            self.next = BranchAction::Evaluate(PointRequest {
                x: self.x.clone(),
                need_hessian: true,
                role: PointRole::FinalHessian,
                step: self.cycle + 1,
            });
        } else {
            self.next = BranchAction::Finished(end);
        }
    }

    fn record(&self, step: usize, value: &PointValue, imaginary: &[f64]) -> StepRecord {
        StepRecord {
            step,
            x: self.x.clone(),
            energy: value.raw_energy,
            biased_energy: value.energy,
            rms_gradient: rms(&value.raw_gradient),
            rms_biased_gradient: rms(&value.gradient),
            arc_length: self.history.arc_length(),
            bending_angle: None,
            extras: vec![("Imaginary Frequencies", python_list(imaginary))],
            curvature: None,
            in_table: true,
            notes: Vec::new(),
            curvature_across: None,
        }
    }
}

impl Integrator for ModeKill {
    fn name(&self) -> &'static str {
        "modekill"
    }

    fn ask(&self) -> BranchAction {
        self.next.clone()
    }

    fn tell(&mut self, value: &PointValue) -> MoptResult<Option<StepRecord>> {
        if !value.is_finite() || value.gradient.nrows() != self.space.dimension() {
            return Err(MoptError::calc(format!(
                "ModeKill cycle {}: the surface returned a value that is not finite or of \
                 another size",
                self.cycle
            )));
        }
        self.model.observe(&self.space, &self.x, value)?;
        let (values, vectors) = symmetric_eigen(&self.model.total(&self.space, value))?;
        let nus = wavenumbers(&values);
        let imaginary: Vec<f64> = nus
            .iter()
            .copied()
            .filter(|nu| *nu <= self.nu_threshold)
            .collect();
        if let Some(end) = self.pending_end.take() {
            // The Hessian of the last structure.
            let mut record = self.record(self.cycle + 1, value, &imaginary);
            record.notes.push(format!(
                "Final wavenumbers of imaginary modes (<= {} cm^-1): {} cm^-1",
                python_float(self.nu_threshold),
                python_list(&imaginary)
            ));
            self.next = BranchAction::Finished(end);
            return Ok(Some(record));
        }
        let mut notes = Vec::new();
        self.follow(&values, &vectors, &nus, &mut notes)?;
        let q = self.space.to_mw(&self.x);
        let bending_angle = self.history.push(q, value.energy);
        let mut record = self.record(self.cycle, value, &imaginary);
        record.bending_angle = bending_angle;
        if self.kill.is_empty() {
            // MultiOptPy writes the structure but no row of the table here.
            notes.push("No modes left to track. ModeKill will exit.".to_string());
            record.in_table = false;
            record.notes = notes;
            self.finish(BranchEnd::ModesRemoved);
            return Ok(Some(record));
        }
        for &k in &self.kill {
            notes.push(format!("Mode {k}: {:.2} cm^-1", nus[k]));
        }
        record.notes = notes;
        let g = self.space.gradient_to_mw(&value.gradient);
        let step = downhill_step(&vectors, &self.kill, &g, self.settings.step_size);
        self.previous_modes = Some(vectors);
        self.x = self.space.displaced(&self.x, &step);
        let s = &self.settings;
        if convergence_check(&value.gradient, s.max_force, s.rms_force) && imaginary.is_empty() {
            self.finish(BranchEnd::Converged);
        } else if self.cycle >= s.max_steps {
            self.finish(BranchEnd::MaxSteps);
        } else {
            self.cycle += 1;
            self.next = self.cycle_request();
        }
        Ok(Some(record))
    }

    fn current(&self) -> &Col<f64> {
        &self.x
    }
}
