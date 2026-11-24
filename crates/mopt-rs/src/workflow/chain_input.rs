//! The first chain of a relaxation, from what `nebmain` is given.
//!
//! Structures come from a file of several structures (the trajectory of a scan or of an
//! optimization with an artificial force, as in the articles of MultiOptPy's author), from
//! several files, or from a directory with files `<name>_<number>.xyz` (MultiOptPy's
//! "input folder"; in the order of the numbers). They become a chain in MultiOptPy's
//! order:
//!
//! 1. `-p N`: `N` images are put on the straight line between every two structures that
//!    follow each other;
//! 2. `-ad`: the images are spread evenly along the chain;
//! 3. `-nd D`: the chain is replaced by images `D` apart along it.
//!
//! Two errors of MultiOptPy's input are not taken over (`docs/source/migration_paths.md`, "Chain
//! relaxation"): a file of several structures loses its last structure there unless a
//! line that is not an atom follows it, and the first structure enters the chain twice.
//!
//! MultiOptPy: `neb.py:NEB.make_geometry_list`, `fileio.py:traj2list`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::path::{Path, PathBuf};

use faer::Col;

use crate::domain::{MolecularSystem, SpinCharge};
use crate::error::{MoptError, MoptResult, ResultExt};
use crate::io::xyz::read_xyz_trajectory_file;
use crate::path::band::align::{centre_chain_at, free_shifts};
use crate::path::band::path_lengths;
use crate::path::band::redistribute::{by_spacing, equal_spacing, Curve};
use crate::path::band::ChainKind;
use crate::path::interp::{images_between, FirstPath, PathFrame, PathSettings};

/// Most digits of the number in the name of a structure file of an input directory.
const MOST_DIGITS: usize = 6;

/// The files `<name>_<number>.xyz` of `directory`, in MultiOptPy's order: by the number of
/// digits of the number, then by name.
pub fn numbered_structure_files(directory: &Path) -> MoptResult<Vec<PathBuf>> {
    let mut found: Vec<(usize, String, PathBuf)> = Vec::new();
    let entries = std::fs::read_dir(directory)
        .map_err(|e| MoptError::Io(format!("{}: {e}", directory.display())))?;
    for entry in entries {
        let path = entry?.path();
        let is_xyz = path
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("xyz"));
        let Some(stem) = path.file_stem().map(|s| s.to_string_lossy().into_owned()) else {
            continue;
        };
        let digits = stem
            .rsplit_once('_')
            .map(|(_, number)| number)
            .filter(|number| !number.is_empty() && number.bytes().all(|b| b.is_ascii_digit()))
            .map(str::len);
        if let (true, Some(digits @ 1..=MOST_DIGITS)) = (is_xyz, digits) {
            found.push((digits, stem, path));
        }
    }
    found.sort();
    Ok(found.into_iter().map(|(_, _, path)| path).collect())
}

/// Largest difference of two lattice vectors (Bohr) at which two structures are in the
/// same cell (the files have a limited number of digits).
const SAME_CELL_BOHR: f64 = 1e-6;

/// Structures in a periodic cell made ready for a chain. Returns the directions along
/// which a shift of all atoms changes nothing ([`free_shifts`]; `None` without a cell).
///
/// - All structures get the cell of the first one. A structure with another cell is an
///   error (a chain whose cell changes is not implemented); one without a cell takes the
///   cell of the first.
/// - Every atom follows its nearest periodic image from one structure to the next, so
///   that no atom crosses the cell between two images that show the same position. An
///   atom that really moves further than half a cell between two structures needs
///   structures in between.
/// - Every structure is shifted as a whole along the periodic directions so that its
///   centroid is that of the first there: such a shift changes nothing in a periodic
///   cell and would otherwise count as a part of the path (a length, and a component of
///   the tangents that no force has).
pub fn join_in_the_cell(structures: &mut [MolecularSystem]) -> MoptResult<Option<Vec<[f64; 3]>>> {
    join_in_the_cell_centred(structures, true)
}

/// [`join_in_the_cell`], the structures shifted to the centroid of the first only with
/// `centre` (without: on a surface that depends on the position in the cell, such as a bias
/// potential at a fixed point, a shift is not free).
///
/// MultiOptPy: no counterpart (new in mopt-rs).
pub fn join_in_the_cell_centred(
    structures: &mut [MolecularSystem],
    centre: bool,
) -> MoptResult<Option<Vec<[f64; 3]>>> {
    let Some(cell) = structures
        .first()
        .and_then(|first| first.active_cell().cloned())
    else {
        if structures.iter().any(|s| s.active_cell().is_some()) {
            return Err(MoptError::config(
                "the first structure has no periodic cell, a later one has; give the cell \
                 (Lattice=\"...\") in the first structure",
            ));
        }
        return Ok(None);
    };
    for (k, structure) in structures.iter().enumerate().skip(1) {
        if let Some(other) = structure.active_cell() {
            let same = other.periodic_axes == cell.periodic_axes
                && (0..3).all(|r| {
                    (0..3).all(|c| (other.h_bohr[r][c] - cell.h_bohr[r][c]).abs() < SAME_CELL_BOHR)
                });
            if !same {
                return Err(MoptError::unsupported(format!(
                    "structure {} has another cell than the first one; a chain whose cell \
                     changes is not implemented (all images are in the cell of the first \
                     structure)",
                    k + 1
                )));
            }
        }
    }
    let atoms = structures[0].n_atoms();
    let mut chain = vec![structures[0].coords_flat_bohr()];
    for structure in structures.iter().skip(1) {
        let read = structure.coords_flat_bohr();
        let previous = &chain[chain.len() - 1];
        let mut joined = Col::<f64>::zeros(read.nrows());
        for atom in 0..atoms {
            let at = |x: &Col<f64>| [x[3 * atom], x[3 * atom + 1], x[3 * atom + 2]];
            let delta = cell.mic_delta(at(&read), at(previous));
            for k in 0..3 {
                joined[3 * atom + k] = previous[3 * atom + k] + delta[k];
            }
        }
        chain.push(joined);
    }
    let shifts = free_shifts(&cell);
    if centre {
        let c = crate::path::band::centroid(&chain[0], 3);
        centre_chain_at(&mut chain, [c[0], c[1], c[2]], &shifts);
    }
    for (structure, joined) in structures.iter_mut().zip(&chain).skip(1) {
        structure.cell = Some(cell.clone());
        structure.set_flat_coords_bohr(joined)?;
    }
    Ok(Some(shifts))
}

/// The first structure file of `inputs` (the first numbered file of a directory).
pub fn first_structure_file(inputs: &[PathBuf]) -> MoptResult<PathBuf> {
    let first = inputs
        .first()
        .ok_or_else(|| MoptError::config("no input structures"))?;
    if first.is_dir() {
        numbered_structure_files(first)?
            .into_iter()
            .next()
            .ok_or_else(|| {
                MoptError::config(format!(
                    "{}: no files <name>_<number>.xyz in this directory",
                    first.display()
                ))
            })
    } else {
        Ok(first.clone())
    }
}

/// The structures of `inputs` in order: all structures of every file, the numbered files of
/// a directory. All must have the same atoms in the same order.
pub fn read_structures(inputs: &[PathBuf], spin: SpinCharge) -> MoptResult<Vec<MolecularSystem>> {
    let mut files = Vec::new();
    for input in inputs {
        if input.is_dir() {
            let numbered = numbered_structure_files(input)?;
            if numbered.is_empty() {
                return Err(MoptError::config(format!(
                    "{}: no files <name>_<number>.xyz in this directory",
                    input.display()
                )));
            }
            files.extend(numbered);
        } else {
            files.push(input.clone());
        }
    }
    let mut structures: Vec<MolecularSystem> = Vec::new();
    for file in &files {
        let frames = read_xyz_trajectory_file(file, spin, None).context(file.display())?;
        for frame in frames.frames {
            if let Some(first) = structures.first() {
                if frame.system.elements != first.elements {
                    return Err(MoptError::config(format!(
                        "{}: a structure with other atoms than the first one (the structures \
                         of a chain have the same atoms in the same order)",
                        file.display()
                    )));
                }
            }
            structures.push(frame.system);
        }
    }
    if structures.len() < 2 {
        return Err(MoptError::config(format!(
            "a chain needs at least two structures; {} was read",
            structures.len()
        )));
    }
    Ok(structures)
}

/// How the points that were read become the first chain.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FirstChain {
    /// Images put on the straight line between every two points that follow each other
    /// (`-p`).
    pub inserted: usize,
    /// Spread the images evenly along the chain (`-ad`).
    pub equal_spacing: bool,
    /// Replace the chain by images this far apart along it (`-nd`; in the unit of the
    /// points: Bohr for structures).
    pub spacing: Option<f64>,
    /// How the images between two points that follow each other are made
    /// (`-neb_opt path=`; anything but the straight line needs structures).
    pub path: FirstPath,
}

/// The first chain from `points` (see the module documentation), with the images between
/// two points on the straight line.
pub fn first_chain(
    points: &[Col<f64>],
    kind: ChainKind,
    how: &FirstChain,
) -> MoptResult<Vec<Col<f64>>> {
    first_chain_in(
        points,
        kind,
        how,
        &PathFrame::default(),
        &PathSettings::default(),
    )
    .map(|(chain, _)| chain)
}

/// The first chain from `points`, with what was done for the log: the images between two
/// points that follow each other are made by `how.path` in `frame`
/// ([`crate::path::interp`]). There are `-p` of them; without `-p` and with another method
/// than the straight line, as many as the distance of `-nd` asks for between the two
/// points (none between points that are nearer than that, so a file of many structures
/// stays as it is).
pub fn first_chain_in(
    points: &[Col<f64>],
    kind: ChainKind,
    how: &FirstChain,
    frame: &PathFrame,
    settings: &PathSettings,
) -> MoptResult<(Vec<Col<f64>>, Vec<String>)> {
    if points.len() < 2 {
        return Err(MoptError::config("a chain needs at least two points"));
    }
    let mut notes = Vec::new();
    let mut chain = Vec::with_capacity((points.len() - 1) * (how.inserted + 1) + 1);
    if how.path == FirstPath::Linear {
        for pair in points.windows(2) {
            for i in 0..=how.inserted {
                let t = i as f64 / (how.inserted + 1) as f64;
                chain.push(&pair[0] + t * (&pair[1] - &pair[0]));
            }
        }
    } else {
        if kind != ChainKind::Atoms {
            return Err(MoptError::config(format!(
                "-neb_opt path={}: the images between plain variables are on the straight line \
                 (the other ways are made of distances between atoms)",
                how.path.name()
            )));
        }
        let (mut made, mut iterations, mut converged) = (0, 0, true);
        for pair in points.windows(2) {
            let count = match (how.inserted, how.spacing) {
                (0, Some(spacing)) if spacing > 0.0 => {
                    let apart = path_lengths(pair, kind)[1];
                    ((apart / spacing).ceil() as usize).saturating_sub(1)
                }
                (inserted, _) => inserted,
            };
            chain.push(pair[0].clone());
            let between = images_between(how.path, &pair[0], &pair[1], count, frame, settings)?;
            made += between.images.len();
            iterations += between.iterations;
            converged &= between.converged;
            chain.extend(between.images);
        }
        notes.push(format!(
            "First images (-neb_opt path={}): {made} image(s) between the structures in \
             {iterations} iteration(s){}.",
            how.path.name(),
            if converged {
                ""
            } else {
                "; not all of them came to rest within their iterations"
            }
        ));
    }
    chain.push(points[points.len() - 1].clone());
    if how.equal_spacing {
        chain = equal_spacing(&chain, kind, Curve::Polyline);
    }
    if let Some(spacing) = how.spacing {
        if spacing.is_nan() || spacing <= 0.0 {
            return Err(MoptError::config(format!(
                "the distance of the images must be positive, not {spacing}"
            )));
        }
        chain = by_spacing(&chain, spacing, kind);
    }
    if chain.len() < 3 {
        return Err(MoptError::config(format!(
            "the first chain has {} images; a relaxation needs at least three (more points, \
             -p, or a smaller -nd)",
            chain.len()
        )));
    }
    Ok((chain, notes))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn col(values: &[f64]) -> Col<f64> {
        Col::from_fn(values.len(), |k| values[k])
    }

    fn scratch(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("mopt_chain_input_{name}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn water(shift: f64) -> String {
        format!(
            "3\n\nO {:.3} 0.000 0.000\nH {:.3} 0.757 0.586\nH {:.3} -0.757 0.586\n",
            shift, shift, shift
        )
    }

    #[test]
    fn structures_come_from_a_file_several_files_or_a_directory() {
        let dir = scratch("read");
        // A file of three structures without a line after the last atom: all three are
        // read (MultiOptPy loses the last one).
        let trajectory = dir.join("traj.xyz");
        let text = format!("{}{}{}", water(0.0), water(0.5), water(1.0));
        std::fs::write(&trajectory, text.trim_end()).unwrap();
        let read = read_structures(std::slice::from_ref(&trajectory), SpinCharge::default());
        let structures = read.unwrap();
        assert_eq!(structures.len(), 3);
        let x = |k: usize| structures[k].coords_flat_bohr()[0];
        assert!(x(0).abs() < 1e-12 && x(2) > x(1) && x(1) > 0.9);

        // Several files, in the order given.
        let (a, b) = (dir.join("a.xyz"), dir.join("b.xyz"));
        std::fs::write(&a, water(2.0)).unwrap();
        std::fs::write(&b, water(3.0)).unwrap();
        let two = read_structures(&[b.clone(), a.clone()], SpinCharge::default()).unwrap();
        assert!(two[0].coords_flat_bohr()[0] > two[1].coords_flat_bohr()[0]);

        // A directory: the numbered files in the order of their numbers (2 before 10).
        let folder = dir.join("folder");
        std::fs::create_dir_all(&folder).unwrap();
        for (number, shift) in [(10, 3.0), (2, 2.0), (1, 1.0)] {
            std::fs::write(folder.join(format!("w_{number}.xyz")), water(shift)).unwrap();
        }
        std::fs::write(folder.join("notes.txt"), "not a structure").unwrap();
        std::fs::write(folder.join("other.xyz"), water(9.0)).unwrap();
        let numbered = numbered_structure_files(&folder).unwrap();
        let names: Vec<String> = numbered
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        assert_eq!(names, ["w_1.xyz", "w_2.xyz", "w_10.xyz"]);
        let from_folder = read_structures(&[folder], SpinCharge::default()).unwrap();
        assert_eq!(from_folder.len(), 3);

        // One structure is no chain; other atoms are refused; an empty directory too.
        assert!(read_structures(&[a], SpinCharge::default()).is_err());
        let other = dir.join("other.xyz");
        std::fs::write(&other, "1\n\nHe 0.0 0.0 0.0\n").unwrap();
        assert!(read_structures(&[b, other], SpinCharge::default()).is_err());
        let empty = dir.join("empty");
        std::fs::create_dir_all(&empty).unwrap();
        assert!(read_structures(&[empty], SpinCharge::default()).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_first_chain_follows_the_points() {
        let points = vec![col(&[0.0, 0.0]), col(&[3.0, 0.0]), col(&[3.0, 6.0])];
        let plain = FirstChain {
            inserted: 0,
            equal_spacing: false,
            spacing: None,
            path: FirstPath::Linear,
        };
        // As read: the first point once.
        let chain = first_chain(&points, ChainKind::Variables, &plain).unwrap();
        assert_eq!(chain, points);
        // -p 2: two images between every two points.
        let inserted = FirstChain {
            inserted: 2,
            ..plain
        };
        let chain = first_chain(&points, ChainKind::Variables, &inserted).unwrap();
        assert_eq!(chain.len(), 7);
        assert!((chain[1][0] - 1.0).abs() < 1e-14 && (chain[5][1] - 4.0).abs() < 1e-14);
        assert_eq!(chain[3], points[1]);
        // -ad: evenly along the chain of length 9 (the corner is cut by no image).
        let even = FirstChain {
            equal_spacing: true,
            ..inserted
        };
        let chain = first_chain(&points, ChainKind::Variables, &even).unwrap();
        assert!((chain[1][0] - 1.5).abs() < 1e-13 && (chain[2][0] - 3.0).abs() < 1e-13);
        assert!((chain[3][1] - 1.5).abs() < 1e-13);
        // -nd 2: images every 2 along the chain, and the end.
        let spaced = FirstChain {
            spacing: Some(2.0),
            ..plain
        };
        let chain = first_chain(&points, ChainKind::Variables, &spaced).unwrap();
        assert_eq!(chain.len(), 6);
        assert!((chain[1][0] - 2.0).abs() < 1e-13 && (chain[2][1] - 1.0).abs() < 1e-13);
        assert_eq!(chain[5], points[2]);
        // Too few images or points, or a distance that is not positive.
        assert!(first_chain(&points[..2], ChainKind::Variables, &plain).is_err());
        assert!(first_chain(&points[..1], ChainKind::Variables, &plain).is_err());
        let bad = FirstChain {
            spacing: Some(0.0),
            ..plain
        };
        assert!(first_chain(&points, ChainKind::Variables, &bad).is_err());
    }

    #[test]
    fn structures_in_a_cell_are_joined_by_the_nearest_images() {
        use crate::domain::Units;
        use crate::io::xyz::parse_xyz_str;
        let cell = "Lattice=\"10.0 0.0 0.0 0.0 10.0 0.0 0.0 0.0 10.0\" pbc=\"T T T\"";
        let read = |text: &str| parse_xyz_str(text, SpinCharge::default(), None).unwrap();
        let a = Units::ANGSTROM_TO_BOHR;
        // The second structure: the first atom 0.4 further along x, written one cell
        // back; the second atom where it was, written one cell up along y; all atoms then
        // shifted by 0.2 along z.
        let mut structures = vec![
            read(&format!("2\n{cell}\nAr 9.8 1.0 1.0\nAr 5.0 5.0 5.0\n")),
            read(&format!("2\n{cell}\nAr 0.2 1.0 1.2\nAr 5.0 15.0 5.2\n")),
        ];
        assert_eq!(join_in_the_cell(&mut structures).unwrap().unwrap().len(), 3);
        let x = structures[1].coords_flat_bohr();
        // Joined: (10.2, 1, 1.2) and (5, 5, 5.2); the centroid moved by (0.2, 0, 0.2) and
        // is put back.
        let expected = [10.0, 1.0, 1.0, 4.8, 5.0, 5.0];
        for (k, value) in expected.iter().enumerate() {
            assert!((x[k] - value * a).abs() < 1e-9, "{k}: {}", x[k] / a);
        }
        assert!(structures[1].active_cell().is_some());

        // A structure without a cell takes the cell of the first; another cell is refused;
        // without any cell nothing is done.
        let mut bare = vec![
            read(&format!("2\n{cell}\nAr 9.8 1.0 1.0\nAr 5.0 5.0 5.0\n")),
            read("2\n\nAr 9.9 1.0 1.0\nAr 5.1 5.0 5.0\n"),
        ];
        assert!(join_in_the_cell(&mut bare).unwrap().is_some());
        assert!(bare[1].active_cell().is_some());
        let other = cell.replace("10.0 0.0 0.0 0.0 10.0", "10.5 0.0 0.0 0.0 10.0");
        let mut two_cells = vec![
            read(&format!("2\n{cell}\nAr 9.8 1.0 1.0\nAr 5.0 5.0 5.0\n")),
            read(&format!("2\n{other}\nAr 9.8 1.0 1.0\nAr 5.0 5.0 5.0\n")),
        ];
        assert!(join_in_the_cell(&mut two_cells).is_err());
        let mut none = vec![
            read("2\n\nAr 9.8 1.0 1.0\nAr 5.0 5.0 5.0\n"),
            read("2\n\nAr 0.2 1.0 1.2\nAr 5.0 15.0 5.2\n"),
        ];
        assert!(join_in_the_cell(&mut none).unwrap().is_none());
        assert!((none[1].coords_flat_bohr()[4] - 15.0 * a).abs() < 1e-9);
    }
}
