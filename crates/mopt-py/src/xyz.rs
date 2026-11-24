//! Structure files from Python: the `.xyz` and extended `.xyz` reader and writer of the
//! library (`mopt_rs::io::xyz`), so that a script reads and writes the files the commands
//! read and write, with the same numbers of decimals (12 for coordinates in Angstrom, 15 for
//! the lattice).
//!
//! MultiOptPy: `fileio.py` (`xyz2list`, `traj2list`, `write_xyz_file`).
//!
//! This file is part of mopt-py (the Python interface of mopt-rs).
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::io::Write as _;
use std::path::PathBuf;

use faer::Col;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use mopt_rs::domain::{Bohr, Element, MolecularSystem, PeriodicCell, SpinCharge, Units, Vector3};
use mopt_rs::error::MoptError;
use mopt_rs::io::xyz::{format_xyz_system, read_xyz_trajectory_file, state_of_xyz};

use super::err;

/// One frame for [`xyz_write`]: symbols, positions (Angstrom, flat), comment line, lattice
/// vectors (Angstrom, the rows a, b, c, flat) and the periodic axes.
type FrameIn = (
    Vec<String>,
    Vec<f64>,
    String,
    Option<Vec<f64>>,
    Option<[bool; 3]>,
);

/// The frames of an `.xyz` or extended `.xyz` file (all of them, or the first with
/// `first_only`), read by the reader of the commands: dicts with `symbols`, `positions_bohr`
/// (flat, as the library holds them), `comment`, `lattice_bohr` (the rows a, b, c, flat, or
/// `None`), `pbc`, `energy` (Hartree, from the comment line where it gives one) and `state`
/// (`(charge, multiplicity)` when the comment line is two whole numbers).
#[pyfunction]
#[pyo3(signature = (path, first_only = false))]
fn xyz_read<'py>(
    py: Python<'py>,
    path: PathBuf,
    first_only: bool,
) -> PyResult<Vec<Bound<'py, PyDict>>> {
    let trajectory = read_xyz_trajectory_file(&path, SpinCharge::default(), None)
        .map_err(|e| err(e.context(format!("{}", path.display()))))?;
    let count = if first_only {
        1
    } else {
        trajectory.frames.len()
    };
    let mut out = Vec::with_capacity(count);
    for frame in trajectory.frames.iter().take(count) {
        let s = &frame.system;
        let d = PyDict::new(py);
        d.set_item(
            "symbols",
            s.elements.iter().map(|e| e.symbol()).collect::<Vec<_>>(),
        )?;
        d.set_item(
            "positions_bohr",
            s.coords_flat_bohr().iter().copied().collect::<Vec<f64>>(),
        )?;
        d.set_item("comment", &s.comment)?;
        let cell = s.cell.as_ref();
        d.set_item(
            "lattice_bohr",
            cell.map(|c| {
                (0..3)
                    .flat_map(|k| c.lattice_vector(k).value())
                    .collect::<Vec<f64>>()
            }),
        )?;
        d.set_item("pbc", cell.map_or([false; 3], |c| c.periodic_axes).to_vec())?;
        d.set_item("energy", frame.energy.map(|e| e.value()))?;
        d.set_item("state", state_of_xyz(&s.comment))?;
        out.push(d);
    }
    Ok(out)
}

/// `comment` without the keys `Lattice="..."`, `pbc="..."`, and `Properties=...` that the
/// writer puts in front of the comment of a periodic frame (a comment read from an extended
/// `.xyz` has them; written again with its lattice, they would appear twice).
fn without_cell_keys(comment: &str) -> String {
    let mut rest = comment;
    let mut kept = Vec::new();
    while let Some(start) = rest.find(|c: char| !c.is_whitespace()) {
        rest = &rest[start..];
        let quoted = ["Lattice=\"", "pbc=\""]
            .iter()
            .find(|key| rest.starts_with(**key))
            .map(|key| key.len());
        let end = match quoted {
            Some(open) => rest[open..]
                .find('"')
                .map_or(rest.len(), |close| open + close + 1),
            None => rest.find(char::is_whitespace).unwrap_or(rest.len()),
        };
        let token = &rest[..end];
        if quoted.is_none() && !token.starts_with("Properties=") {
            kept.push(token);
        }
        rest = &rest[end..];
    }
    kept.join(" ")
}

/// The text of one frame as the library writes it (`format_xyz_system`).
fn frame_text(frame: &FrameIn) -> Result<String, MoptError> {
    let (symbols, positions, comment, lattice, pbc) = frame;
    let elements = symbols
        .iter()
        .map(|s| Element::parse_token(s))
        .collect::<Result<Vec<_>, _>>()?;
    if positions.len() != 3 * elements.len() {
        return Err(MoptError::config(format!(
            "{} coordinates for {} atoms",
            positions.len(),
            elements.len()
        )));
    }
    let cell = match lattice {
        None if pbc.is_some_and(|axes| axes.contains(&true)) => {
            return Err(MoptError::config("periodic axes need a lattice"))
        }
        None => None,
        Some(l) if l.len() != 9 => {
            return Err(MoptError::config(
                "a lattice has 9 numbers (the rows a, b, c)",
            ))
        }
        Some(l) => {
            let a2b = Units::ANGSTROM_TO_BOHR;
            let v = |k: usize| {
                Vector3::<Bohr>::new([l[3 * k] * a2b, l[3 * k + 1] * a2b, l[3 * k + 2] * a2b])
            };
            Some(PeriodicCell::from_lattice_vectors(
                v(0),
                v(1),
                v(2),
                pbc.unwrap_or([true; 3]),
            )?)
        }
    };
    let flat = Col::from_fn(positions.len(), |i| positions[i] * Units::ANGSTROM_TO_BOHR);
    let periodic = cell.as_ref().is_some_and(|c| c.is_any_periodic());
    let mut system = MolecularSystem::from_flat_bohr(elements, &flat, SpinCharge::default(), cell)?;
    system.comment = if periodic {
        without_cell_keys(comment)
    } else {
        comment.clone()
    };
    Ok(format_xyz_system(&system, None))
}

/// Write `frames` (symbols, positions in Angstrom, comment, lattice in Angstrom or `None`,
/// periodic axes or `None`) to `path` with the writer of the commands; `append` adds them
/// at the end of the file instead of replacing it.
#[pyfunction]
#[pyo3(signature = (path, frames, append = false))]
fn xyz_write(path: PathBuf, frames: Vec<FrameIn>, append: bool) -> PyResult<()> {
    let text = frames
        .iter()
        .map(frame_text)
        .collect::<Result<String, _>>()
        .map_err(err)?;
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .append(append)
        .truncate(!append)
        .open(&path)
        .map_err(|e| err(MoptError::from(e).context(format!("{}", path.display()))))?;
    file.write_all(text.as_bytes())
        .map_err(|e| err(MoptError::from(e).context(format!("{}", path.display()))))
}

/// Add the functions of this module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(xyz_read, m)?)?;
    m.add_function(wrap_pyfunction!(xyz_write, m)?)?;
    Ok(())
}
