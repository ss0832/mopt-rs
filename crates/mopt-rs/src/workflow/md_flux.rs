//! The heat current of every step of a run (`-md_opt heat_flux=on`), for the Green-Kubo
//! thermal conductivity: `J = sum_i (e_i v_i + W_i v_i)` with the energy of every atom
//! `e_i = m_i v_i^2 / 2 + U_i` and its virial `W_i` from the calculator
//! (`Calculator::site_terms`: the built-in pair and embedded-atom potentials `lj` with a
//! cutoff, `eam`, and `emt`, and the bond-order potential `tersoff`), written to
//! `<stem>_heat_flux.csv` (Hartree Bohr per atomic unit of time, with the volume of the cell
//! in Bohr^3).
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::fs::{self, OpenOptions};
use std::io::Write as _;
use std::path::{Path, PathBuf};

use super::md::MdObserver;
use crate::domain::{MolecularSystem, Units};
use crate::error::{MoptError, MoptResult};
use crate::md::MdStepReport;
use crate::pes::calculator::Calculator;

/// Writes the heat current of every step.
pub struct HeatFlux<'a> {
    calc: &'a dyn Calculator,
    template: MolecularSystem,
    masses: Vec<f64>,
    path: PathBuf,
}

impl<'a> HeatFlux<'a> {
    /// The heat current of `calc` on structures like `template` into
    /// `<stem>_heat_flux.csv` in `dir`; refused when the calculator cannot split its energy
    /// over the atoms.
    pub fn new(
        calc: &'a dyn Calculator,
        template: &MolecularSystem,
        dir: &Path,
        stem: &str,
    ) -> MoptResult<Self> {
        match calc.site_terms(template) {
            None => {
                return Err(MoptError::unsupported(format!(
                    "the heat current needs the energies and virials of the atoms, which \
                     {} does not give (the built-in lj with a cutoff, eam, emt, and tersoff do; the \
                     reverse non-equilibrium method of -md_opt exchange= works with any \
                     calculator)",
                    calc.name()
                )))
            }
            Some(Err(e)) => return Err(e),
            Some(Ok(_)) => {}
        }
        Ok(Self {
            calc,
            masses: template
                .masses_amu()
                .iter()
                .map(|m| m * Units::AMU_TO_AU_MASS)
                .collect(),
            template: template.clone(),
            path: dir.join(format!("{stem}_heat_flux.csv")),
        })
    }

    /// The file.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The heat current at the structure and velocities of `r`.
    pub fn current(&self, r: &MdStepReport) -> MoptResult<[f64; 3]> {
        let mut sys = self.template.with_flat_coords_bohr(&r.x)?;
        if r.cell.is_some() {
            sys.cell.clone_from(&r.cell);
        }
        let terms = self
            .calc
            .site_terms(&sys)
            .ok_or_else(|| MoptError::calc("the calculator stopped giving site energies"))??;
        let mut j = [0.0; 3];
        for (i, (u, w)) in terms.energies.iter().zip(&terms.virials).enumerate() {
            let v = [r.v[3 * i], r.v[3 * i + 1], r.v[3 * i + 2]];
            let e = 0.5 * self.masses[i] * (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) + u;
            for a in 0..3 {
                j[a] += e * v[a] + (0..3).map(|b| w[a][b] * v[b]).sum::<f64>();
            }
        }
        Ok(j)
    }
}

impl MdObserver for HeatFlux<'_> {
    fn on_step(&mut self, r: &MdStepReport) -> MoptResult<()> {
        if r.step == 0 {
            fs::write(&self.path, "step,time_fs,J_x,J_y,J_z,volume_bohr3\n")?;
        }
        let j = self.current(r)?;
        let mut f = OpenOptions::new()
            .append(true)
            .create(true)
            .open(&self.path)?;
        f.write_all(
            format!(
                "{},{},{},{},{},{}\n",
                r.step,
                r.time_fs(),
                j[0],
                j[1],
                j[2],
                r.volume.unwrap_or(f64::NAN)
            )
            .as_bytes(),
        )?;
        Ok(())
    }
}
