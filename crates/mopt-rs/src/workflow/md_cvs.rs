//! The collective variables of `-cv` along a run: `<stem>_cv.csv` with their values
//! (atomic units: Bohr, radians) at every step, for the statistics and autocorrelations of
//! the trajectory analysis.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::fmt::Write as _;
use std::fs::{self, OpenOptions};
use std::io::Write as _;
use std::path::{Path, PathBuf};

use super::md::MdObserver;
use crate::domain::MolecularSystem;
use crate::error::MoptResult;
use crate::md::cv::Cv;
use crate::md::MdStepReport;

/// Writes the values of collective variables at every step.
pub struct MdCvs {
    cvs: Vec<(String, Cv)>,
    template: MolecularSystem,
    path: PathBuf,
}

impl MdCvs {
    /// The variables `cvs` (with the names of their columns) of structures like `template`
    /// into `<stem>_cv.csv` in `dir`.
    pub fn new(cvs: Vec<(String, Cv)>, template: &MolecularSystem, dir: &Path, stem: &str) -> Self {
        Self {
            cvs,
            template: template.clone(),
            path: dir.join(format!("{stem}_cv.csv")),
        }
    }

    /// The file.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl MdObserver for MdCvs {
    fn on_step(&mut self, r: &MdStepReport) -> MoptResult<()> {
        if r.step == 0 {
            let names: Vec<&str> = self.cvs.iter().map(|(n, _)| n.as_str()).collect();
            fs::write(&self.path, format!("step,time_fs,{}\n", names.join(",")))?;
        }
        let mut sys = self.template.with_flat_coords_bohr(&r.x)?;
        if r.cell.is_some() {
            sys.cell.clone_from(&r.cell);
        }
        let mut line = format!("{},{}", r.step, r.time_fs());
        for (_, cv) in &self.cvs {
            let _ = write!(line, ",{}", cv.value(&sys)?);
        }
        line.push('\n');
        let mut f = OpenOptions::new()
            .append(true)
            .create(true)
            .open(&self.path)?;
        f.write_all(line.as_bytes())?;
        Ok(())
    }
}
