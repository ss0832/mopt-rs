//! The coordinates of `-coord` along a molecular dynamics run: `<stem>_coordinates.csv`
//! with the values of the chosen coordinate system at every frame of the trajectory (every
//! `write_every` steps and the last step), whether the dynamics runs in these coordinates
//! or in Cartesian ones.
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
use crate::coord::system::CoordinateSystem;
use crate::domain::MolecularSystem;
use crate::error::MoptResult;
use crate::md::{MdEnd, MdStepReport};

/// Writes the values of a coordinate system at the frames of a run.
pub struct MdCoordinates {
    coords: Box<dyn CoordinateSystem>,
    template: MolecularSystem,
    path: PathBuf,
    write_every: usize,
    last: Option<MdStepReport>,
}

impl MdCoordinates {
    /// The values of `coords` (set up at `template`) into `<stem>_coordinates.csv` in `dir`.
    pub fn new(
        mut coords: Box<dyn CoordinateSystem>,
        template: &MolecularSystem,
        dir: &Path,
        stem: &str,
        write_every: usize,
    ) -> MoptResult<Self> {
        coords.initialize(template)?;
        Ok(Self {
            coords,
            template: template.clone(),
            path: dir.join(format!("{stem}_coordinates.csv")),
            write_every: write_every.max(1),
            last: None,
        })
    }

    /// The file.
    pub fn path(&self) -> &Path {
        &self.path
    }

    fn header(&self) -> String {
        let n = self.coords.dim(&self.template);
        let labels = self
            .coords
            .labels()
            .filter(|l| l.len() == n)
            .unwrap_or_else(|| (1..=n).map(|k| format!("q{k}")).collect());
        format!("step,time_fs,{}\n", labels.join(","))
    }

    fn row(&self, r: &MdStepReport) -> MoptResult<String> {
        let mut sys = self.template.with_flat_coords_bohr(&r.x)?;
        if r.cell.is_some() {
            sys.cell.clone_from(&r.cell);
        }
        let q = self.coords.values(&sys)?;
        let mut line = format!("{},{}", r.step, r.time_fs());
        for v in q.iter() {
            let _ = write!(line, ",{v:.12e}");
        }
        line.push('\n');
        Ok(line)
    }

    fn append(&self, text: &str) -> MoptResult<()> {
        let mut f = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?;
        f.write_all(text.as_bytes())?;
        Ok(())
    }
}

impl MdObserver for MdCoordinates {
    fn on_step(&mut self, r: &MdStepReport) -> MoptResult<()> {
        if r.step == 0 {
            fs::write(&self.path, self.header())?;
        }
        if r.step.is_multiple_of(self.write_every) {
            self.append(&self.row(r)?)?;
        }
        self.last = Some(r.clone());
        Ok(())
    }

    fn on_finish(&mut self, _end: Option<MdEnd>) -> MoptResult<()> {
        if let Some(last) = &self.last {
            if !last.step.is_multiple_of(self.write_every) {
                self.append(&self.row(last)?)?;
            }
        }
        Ok(())
    }
}
