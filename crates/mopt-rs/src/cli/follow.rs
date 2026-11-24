//! `optmain -follow FILE`: the direction a transition-state search starts uphill along (for
//! example the tangent of a chain at the candidate it starts from), read from an xyz file of
//! displacements.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::fmt::Write as _;
use std::path::Path;

use faer::Col;

use crate::domain::MolecularSystem;
use crate::error::{MoptError, MoptResult};
use crate::pes::oniom::micro::RelaxedEnvironment;

/// The direction of `file` (an xyz file whose coordinates are displacements; their length does
/// not matter) for the structure `start`. With ONIOM microiterations `start` is the core and
/// the file may hold the whole system: its components of the core are taken.
pub fn read_direction(
    file: &Path,
    start: &MolecularSystem,
    micro: Option<&RelaxedEnvironment>,
) -> MoptResult<Col<f64>> {
    let text = std::fs::read_to_string(file)
        .map_err(|e| MoptError::config(format!("-follow {}: {e}", file.display())))?;
    let mut lines = text.lines().filter(|l| !l.trim().is_empty());
    let n: usize = lines
        .next()
        .and_then(|l| l.trim().parse().ok())
        .ok_or_else(|| MoptError::parse(format!("-follow {}: no atom count", file.display())))?;
    let _comment = lines.next();
    let mut values = Vec::with_capacity(3 * n);
    for line in lines.take(n) {
        let f: Vec<&str> = line.split_whitespace().collect();
        if f.len() < 4 {
            return Err(MoptError::parse(format!("-follow: '{line}'")));
        }
        for v in &f[1..4] {
            values.push(
                v.parse::<f64>()
                    .map_err(|_| MoptError::parse(format!("-follow: '{line}'")))?,
            );
        }
    }
    let all = Col::from_fn(values.len(), |i| values[i]);
    if all.nrows() == 3 * start.n_atoms() {
        return Ok(all);
    }
    if let Some(relaxed) = micro {
        let core = &relaxed.split().core;
        if all.nrows()
            == 3 * (core.len() + relaxed.split().environment.len() + relaxed.split().fixed.len())
        {
            return Ok(Col::from_fn(3 * core.len(), |k| {
                all[3 * core[k / 3] + k % 3]
            }));
        }
    }
    Err(MoptError::config(format!(
        "-follow {}: {} atoms for a structure of {}",
        file.display(),
        all.nrows() / 3,
        start.n_atoms()
    )))
}

/// The text of a direction file: `atoms` with the displacements `d` (Bohr, written in
/// Angstrom).
pub fn direction_text(atoms: &MolecularSystem, d: &Col<f64>, comment: &str) -> String {
    let mut out = format!("{}\n{comment}\n", atoms.n_atoms());
    for (i, e) in atoms.elements.iter().enumerate() {
        let _ = writeln!(
            out,
            "{:<3} {:15.10} {:15.10} {:15.10}",
            e.symbol(),
            d[3 * i] * crate::domain::Units::BOHR_TO_ANGSTROM,
            d[3 * i + 1] * crate::domain::Units::BOHR_TO_ANGSTROM,
            d[3 * i + 2] * crate::domain::Units::BOHR_TO_ANGSTROM
        );
    }
    out
}
