//! Standard `.xyz` and Extended XYZ (`Lattice="..."`) single-frame and multi-frame trajectory I/O.
//!
//! Fixes MultiOptPy's `traj2list` bug where the final trajectory frame was silently dropped
//! when no trailing blank line was present at EOF.
//!
//! MultiOptPy reads its structure files by the shape of their lines: a line `element x y z`
//! is an atom, a line of two whole numbers is `charge multiplicity`, and everything else
//! separates frames. Its own files therefore need no line with the number of atoms (the
//! files of a directory for `nebmain` start with `0 1`), and the second line of an `.xyz`
//! file may give the state (`-1 1`). Both are read here: a file whose first line is not a
//! number of atoms is read by the shape of its lines, and [`state_of_xyz`] finds the line
//! of the state, which the commands use where the command line gives none.
//!
//! MultiOptPy: `fileio.py` (`xyz2list`, `traj2list`, `write_xyz_file`, `FileIO`).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use crate::domain::{positions, Angstrom, AtomId};
use crate::domain::{Hartree, Quantity};
use std::fs;
use std::path::Path;

use crate::domain::{
    Element, MolecularSystem, PeriodicCell, SpinCharge, Trajectory, TrajectoryFrame, Units,
};
use crate::error::{MoptError, MoptResult};

/// Extract an optional electronic energy (in Hartree) from an XYZ comment string.
fn parse_comment_energy(comment: &str) -> Option<f64> {
    // Support Extended XYZ `energy=-76.1234` or MultiOptPy `ITR. 0\t-76.1234` or standalone float
    for part in comment.split_whitespace() {
        if let Some(val) = part
            .strip_prefix("energy=")
            .or_else(|| part.strip_prefix("Energy="))
            .or_else(|| part.strip_prefix("E="))
        {
            if let Ok(e) = val.trim_end_matches([',', ';']).parse::<f64>() {
                return Some(e);
            }
        }
    }
    let tokens: Vec<&str> = comment.split_whitespace().collect();
    if let Some(last) = tokens.last() {
        if let Ok(e) = last.parse::<f64>() {
            return Some(e);
        }
    }
    None
}

/// `charge multiplicity` if the line consists of two whole numbers (MultiOptPy's line of
/// the state).
fn state_line(line: &str) -> Option<(i64, i64)> {
    let mut tokens = line.split_whitespace();
    let charge = tokens.next()?.parse::<i64>().ok()?;
    let multiplicity = tokens.next()?.parse::<i64>().ok()?;
    tokens.next().is_none().then_some((charge, multiplicity))
}

/// The atom of a line `element x y z` (further columns are ignored), if it is one.
fn atom_line(line: &str) -> Option<(Element, [f64; 3])> {
    let fields: Vec<&str> = line.split_whitespace().collect();
    if fields.len() < 4 || !fields[0].chars().all(|c| c.is_ascii_alphabetic()) {
        return None;
    }
    let element = Element::parse_token(fields[0]).ok()?;
    let mut xyz = [0.0; 3];
    for (value, field) in xyz.iter_mut().zip(&fields[1..4]) {
        *value = field.parse::<f64>().ok()?;
    }
    Some((element, xyz))
}

/// The charge and the multiplicity that a structure file names in a line of two whole
/// numbers (the first such line), or `None`.
pub fn state_of_xyz(content: &str) -> Option<(i64, i64)> {
    content.lines().find_map(state_line)
}

/// [`state_of_xyz`] of a file; `None` as well for a file that is not text.
pub fn state_of_xyz_file(path: impl AsRef<Path>) -> MoptResult<Option<(i64, i64)>> {
    let bytes = fs::read(path.as_ref())?;
    Ok(std::str::from_utf8(&bytes).ok().and_then(state_of_xyz))
}

/// The frames of a file without lines of the number of atoms, as MultiOptPy reads it: the
/// atoms are the lines `element x y z`, and any other line ends a frame.
fn parse_by_the_shape_of_the_lines(
    content: &str,
    default_spin_charge: SpinCharge,
    override_cell: Option<&PeriodicCell>,
) -> MoptResult<Trajectory> {
    let mut frames = Vec::new();
    let mut elements = Vec::new();
    let mut pos_ang = Vec::new();
    let mut close = |elements: &mut Vec<Element>, pos_ang: &mut Vec<[f64; 3]>| -> MoptResult<()> {
        if elements.is_empty() {
            return Ok(());
        }
        let system = MolecularSystem::new(
            std::mem::take(elements),
            &positions::<Angstrom>(pos_ang),
            default_spin_charge,
            override_cell.cloned(),
        )?;
        pos_ang.clear();
        frames.push(TrajectoryFrame {
            step: frames.len() as isize,
            system,
            energy: None,
            biased_energy: None,
            gradient_bohr: None,
        });
        Ok(())
    };
    for line in content.lines() {
        match atom_line(line) {
            Some((element, xyz)) => {
                elements.push(element);
                pos_ang.push(xyz);
            }
            None => close(&mut elements, &mut pos_ang)?,
        }
    }
    close(&mut elements, &mut pos_ang)?;
    if frames.is_empty() {
        return Err(MoptError::parse(
            "the structure file has neither a first line with the number of atoms nor a line \
             `element x y z`",
        ));
    }
    Ok(Trajectory { frames })
}

/// Parse all frames from an `.xyz` / Extended `.xyz` string, or from a file of MultiOptPy
/// without lines of the number of atoms.
pub fn parse_xyz_trajectory_str(
    content: &str,
    default_spin_charge: SpinCharge,
    override_cell: Option<&PeriodicCell>,
) -> MoptResult<Trajectory> {
    let lines: Vec<&str> = content.lines().collect();
    let first = lines
        .iter()
        .map(|line| line.trim())
        .find(|line| !line.is_empty());
    if first.is_some_and(|line| line.parse::<usize>().is_err())
        && lines.iter().any(|line| atom_line(line).is_some())
    {
        return parse_by_the_shape_of_the_lines(content, default_spin_charge, override_cell);
    }
    let mut idx = 0;
    let mut frames = Vec::new();
    let mut step_counter: isize = 0;

    while idx < lines.len() {
        let header = lines[idx].trim();
        if header.is_empty() {
            idx += 1;
            continue;
        }
        let n_atoms = header.parse::<usize>().map_err(|e| {
            MoptError::parse(format!(
                "Invalid XYZ atom count '{}' at line {}: {e}",
                header,
                idx + 1
            ))
        })?;
        if n_atoms == 0 {
            return Err(MoptError::parse(format!(
                "XYZ frame at line {} declares 0 atoms",
                idx + 1
            )));
        }
        if idx + 1 + n_atoms > lines.len() {
            return Err(MoptError::parse(format!(
                "Truncated XYZ frame starting at line {} (expected {} atoms, only {} lines remain)",
                idx + 1,
                n_atoms,
                lines.len().saturating_sub(idx + 2)
            )));
        }

        let comment = lines[idx + 1].trim().to_string();
        let parsed_cell = override_cell
            .cloned()
            .or_else(|| PeriodicCell::parse_extxyz_comment(&comment));
        // A line of the state is not an energy.
        let energy = match state_line(&comment) {
            Some(_) => None,
            None => parse_comment_energy(&comment).map(Quantity::<Hartree>::new),
        };

        let mut elements = Vec::with_capacity(n_atoms);
        let mut pos_ang = Vec::with_capacity(n_atoms);

        for atom_offset in 0..n_atoms {
            let line_no = idx + 2 + atom_offset;
            let fields: Vec<&str> = lines[line_no].split_whitespace().collect();
            if fields.len() < 4 {
                return Err(MoptError::parse(format!(
                    "Malformed XYZ atom line {}: '{}'",
                    line_no + 1,
                    lines[line_no]
                )));
            }
            let el = Element::parse_token(fields[0])?;
            let x = fields[1].parse::<f64>().map_err(|e| {
                MoptError::parse(format!("Invalid X coordinate on line {}: {e}", line_no + 1))
            })?;
            let y = fields[2].parse::<f64>().map_err(|e| {
                MoptError::parse(format!("Invalid Y coordinate on line {}: {e}", line_no + 1))
            })?;
            let z = fields[3].parse::<f64>().map_err(|e| {
                MoptError::parse(format!("Invalid Z coordinate on line {}: {e}", line_no + 1))
            })?;
            elements.push(el);
            pos_ang.push([x, y, z]);
        }

        let mut system = MolecularSystem::new(
            elements,
            &positions::<Angstrom>(&pos_ang),
            default_spin_charge,
            parsed_cell,
        )?;
        system.comment = comment;

        frames.push(TrajectoryFrame {
            step: step_counter,
            system,
            energy,
            biased_energy: None,
            gradient_bohr: None,
        });

        step_counter += 1;
        idx += 2 + n_atoms;
    }

    if frames.is_empty() {
        return Err(MoptError::parse("XYZ input contains no molecular frames"));
    }

    Ok(Trajectory { frames })
}

/// Parse the first (or single) frame from an `.xyz` / Extended `.xyz` string.
pub fn parse_xyz_str(
    content: &str,
    default_spin_charge: SpinCharge,
    override_cell: Option<&PeriodicCell>,
) -> MoptResult<MolecularSystem> {
    let mut traj = parse_xyz_trajectory_str(content, default_spin_charge, override_cell)?;
    Ok(traj.frames.swap_remove(0).system)
}

/// Read a single `MolecularSystem` from an `.xyz` file path.
pub fn read_xyz_file(
    path: impl AsRef<Path>,
    default_spin_charge: SpinCharge,
    override_cell: Option<&PeriodicCell>,
) -> MoptResult<MolecularSystem> {
    let content = fs::read_to_string(path.as_ref())?;
    parse_xyz_str(&content, default_spin_charge, override_cell)
}

/// Read all frames of a multi-structure `.xyz` trajectory file.
pub fn read_xyz_trajectory_file(
    path: impl AsRef<Path>,
    default_spin_charge: SpinCharge,
    override_cell: Option<&PeriodicCell>,
) -> MoptResult<Trajectory> {
    let content = fs::read_to_string(path.as_ref())?;
    parse_xyz_trajectory_str(&content, default_spin_charge, override_cell)
}

/// Format a single `MolecularSystem` into `.xyz` / Extended `.xyz` (in Angstroms).
pub fn format_xyz_system(system: &MolecularSystem, energy: Option<Quantity<Hartree>>) -> String {
    let mut out = String::with_capacity(system.n_atoms() * 64 + 128);
    out.push_str(&format!("{}\n", system.n_atoms()));

    let mut header_parts = Vec::new();
    if let Some(cell) = &system.cell {
        if cell.is_any_periodic() {
            header_parts.push(cell.to_extxyz_header());
            header_parts.push("Properties=species:S:1:pos:R:3".to_string());
        }
    }
    if let Some(e) = energy {
        header_parts.push(format!("energy={:.12}", e.value()));
    } else if !system.comment.is_empty() {
        header_parts.push(system.comment.clone());
    }
    out.push_str(&header_parts.join(" "));
    out.push('\n');

    // Twelve decimals of Angstrom (a rounding of at most 5e-13 Angstrom, 9.4e-13 Bohr, per
    // coordinate), as the trajectories: a value recomputed from the file stays within the
    // 1e-10 every hard constraint is held to (`pes::constraint::GUARANTEE`).
    let b2a = Units::BOHR_TO_ANGSTROM;
    for i in 0..system.n_atoms() {
        let p = system.position(AtomId::new(i)).value();
        out.push_str(&format!(
            "{:<3} {:>17.12} {:>17.12} {:>17.12}\n",
            system.elements[i].symbol(),
            p[0] * b2a,
            p[1] * b2a,
            p[2] * b2a
        ));
    }
    out
}

/// One line of MultiOptPy's coordinate layout: the symbol in two columns, `sep`, and the
/// three values with 12 decimals in 17 columns separated by three spaces (MultiOptPy's
/// `{el:2}<sep>{x:>17.12f}   {y:>17.12f}   {z:>17.12f}`), ending in a newline. The one writer
/// of this layout for the run logs, trajectories, and inputs of the commands.
///
/// MultiOptPy: the coordinate lines of `optimization.py` / `interface.py` (`make_traj_file`).
pub fn multioptpy_row(symbol: &str, sep: &str, v: [f64; 3]) -> String {
    format!(
        "{symbol:<2}{sep}{:>17.12}   {:>17.12}   {:>17.12}\n",
        v[0], v[1], v[2]
    )
}

/// [`multioptpy_row`] for every atom of `sys`, positions in Angstrom.
///
/// MultiOptPy: as [`multioptpy_row`].
pub fn multioptpy_rows(sys: &MolecularSystem, sep: &str) -> String {
    let b2a = Units::BOHR_TO_ANGSTROM;
    let mut s = String::with_capacity(sys.n_atoms() * 64);
    for i in 0..sys.n_atoms() {
        let p = sys.position(AtomId::new(i)).value();
        s.push_str(&multioptpy_row(
            sys.elements[i].symbol(),
            sep,
            [p[0] * b2a, p[1] * b2a, p[2] * b2a],
        ));
    }
    s
}

/// Format an entire `Trajectory` into a concatenated multi-frame `.xyz` string.
pub fn format_xyz_trajectory(traj: &Trajectory) -> String {
    let mut out = String::new();
    for frame in &traj.frames {
        let mut sys = frame.system.clone();
        if sys.comment.is_empty() {
            sys.comment = format!("ITR. {}", frame.step);
        }
        out.push_str(&format_xyz_system(&sys, frame.energy));
    }
    out
}

/// Write a single `MolecularSystem` to an `.xyz` file.
pub fn write_xyz_file(
    path: impl AsRef<Path>,
    system: &MolecularSystem,
    energy: Option<Quantity<Hartree>>,
) -> MoptResult<()> {
    let txt = format_xyz_system(system, energy);
    fs::write(path, txt)?;
    Ok(())
}

/// Write a multi-frame `Trajectory` to an `.xyz` file.
pub fn write_xyz_trajectory_file(path: impl AsRef<Path>, traj: &Trajectory) -> MoptResult<()> {
    let txt = format_xyz_trajectory(traj);
    fs::write(path, txt)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_multi_frame_xyz_without_trailing_newline_and_extxyz_lattice() {
        let sample = "\
3
Lattice=\"10.0 0.0 0.0 0.0 10.0 0.0 0.0 0.0 10.0\" pbc=\"T T T\" energy=-76.40
O  0.000000  0.000000  0.117300
H  0.000000  0.757200 -0.469200
H  0.000000 -0.757200 -0.469200
3
ITR. 1 -76.42
O  0.000000  0.000000  0.120000
H  0.000000  0.760000 -0.470000
H  0.000000 -0.760000 -0.470000";

        let traj = parse_xyz_trajectory_str(sample, SpinCharge::default(), None).unwrap();
        assert_eq!(traj.len(), 2);
        assert!(traj.frames[0].system.has_pbc());
        assert!((traj.frames[0].energy.unwrap().value() - (-76.40)).abs() < 1e-10);
        assert!((traj.frames[1].energy.unwrap().value() - (-76.42)).abs() < 1e-10);
    }

    #[test]
    fn reads_the_files_of_multioptpy_by_the_shape_of_their_lines() {
        let read = |text: &str| parse_xyz_trajectory_str(text, SpinCharge::default(), None);
        // A file of a directory for nebmain: the state, then the atoms, no number of atoms.
        let numbered = "0 1\nC \t  -1.95\t   0.05\t  -0.46\nO \t  -2.87\t  -0.06\t   0.34\n";
        let traj = read(numbered).unwrap();
        assert_eq!(traj.len(), 1);
        assert_eq!(traj.frames[0].system.n_atoms(), 2);
        assert_eq!(state_of_xyz(numbered), Some((0, 1)));
        // Several frames, separated by lines that are not atoms.
        let frames = "0 1\nH 0 0 0\nH 0 0 0.74\n\nH 0 0 0\nH 0 0 0.80\n";
        assert_eq!(read(frames).unwrap().len(), 2);
        // The second line of a standard file may be the state; it is not an energy.
        let anion = "2\n-1 1\nO 0 0 0\nH 0 0 0.96\n";
        assert!(read(anion).unwrap().frames[0].energy.is_none());
        assert_eq!(state_of_xyz(anion), Some((-1, 1)));
        // Other comments are no state, and a file of neither form is refused.
        assert_eq!(
            state_of_xyz("2\nITR. 1 -76.42\nO 0 0 0\nH 0 0 0.96\n"),
            None
        );
        assert_eq!(state_of_xyz("2\n\nO 0 0 0\nH 0 0 0.96\n"), None);
        assert!(read("nothing here\n").is_err());
    }

    #[test]
    fn the_multioptpy_row_is_the_line_of_the_writers_it_replaced() {
        for (symbol, v) in [
            ("H", [0.0, -1.5, 2.25]),
            ("Cl", [-123.456_789_012_345_6, 1e-13, 98_765.432_1]),
            ("Uue", [f64::MIN_POSITIVE, -0.0, 7.0]),
        ] {
            // The six writers before they were one (two spellings of the symbol width).
            for sep in ["  ", "   "] {
                let left = format!(
                    "{:<2}{sep}{:>17.12}   {:>17.12}   {:>17.12}\n",
                    symbol, v[0], v[1], v[2]
                );
                let plain = format!(
                    "{:2}{sep}{:>17.12}   {:>17.12}   {:>17.12}\n",
                    symbol, v[0], v[1], v[2]
                );
                assert_eq!(multioptpy_row(symbol, sep, v), left);
                assert_eq!(left, plain);
            }
        }
    }

    #[test]
    fn a_written_structure_reads_back_within_the_guarantee() {
        use crate::domain::{positions, Bohr, Element};
        let cell = PeriodicCell::from_lattice_vectors(
            crate::domain::Vector3::<Bohr>::new([20.123_456_789_1, 0.0, 0.0]),
            crate::domain::Vector3::new([0.3, 19.987_654_321_1, 0.0]),
            crate::domain::Vector3::new([-0.2, 0.4, 21.111_111_111_1]),
            [true; 3],
        )
        .unwrap();
        let sys = MolecularSystem::new(
            vec![Element::from_atomic_number(18).unwrap(); 2],
            &positions::<Bohr>(&[
                [
                    1.234_567_890_123_4,
                    -7.654_321_098_765_4,
                    3.241_592_653_589_7,
                ],
                [
                    18.765_432_109_876_5,
                    0.123_456_789_012_3,
                    19.876_543_210_987_6,
                ],
            ]),
            SpinCharge::default(),
            Some(cell),
        )
        .unwrap();
        let back =
            parse_xyz_trajectory_str(&format_xyz_system(&sys, None), SpinCharge::default(), None)
                .unwrap()
                .frames
                .remove(0)
                .system;
        // 12 decimals of Angstrom: at most 9.4e-13 Bohr per coordinate.
        let (x, y) = (sys.coords_flat_bohr(), back.coords_flat_bohr());
        for k in 0..6 {
            assert!((x[k] - y[k]).abs() < 1e-12, "{k}: {} {}", x[k], y[k]);
        }
        // The distance through the periodic image, within the guarantee.
        let d = |s: &MolecularSystem| s.distance(AtomId::new(0), AtomId::new(1)).value();
        assert!((d(&sys) - d(&back)).abs() < 1e-11);
        // Files written with fewer decimals read as before.
        let old = "1\nLattice=\"10.00000000 0.0 0.0 0.0 10.0 0.0 0.0 0.0 10.0\"\n\
                   Ar     1.2345678901     0.0000000000     0.0000000000\n";
        let old = parse_xyz_trajectory_str(old, SpinCharge::default(), None).unwrap();
        assert!(old.frames[0].system.has_pbc());
    }
}
