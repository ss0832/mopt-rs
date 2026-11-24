//! The path of a search of `ieipmain` under one name: `<stem>_path.xyz`, the structures
//! from the first input structure to the second.
//!
//! Every method leaves the points it went through, each in its own file and order: the
//! strings (`-gsm`, `-fsm`, `-dsafir`, `-lmm`) their path, iEIP the histories of its two
//! images (`<stem>_traj.xyz`, without the two inputs), 2PSHS its walk from the second
//! structure in (`ADD_0_trajectory.xyz`), the growing Newton trajectory its points
//! (`newton_trajectory.xyz`), the dimer the trajectory of its centre
//! (`dimer_trajectory.xyz`), the spring pair those of its particles
//! (`spm_trajectory.xyz`); the local minimax method, which moves a direction and not a
//! structure, its estimate of the transition state. What follows a search (a relaxation of the path with `nebmain`,
//! step 2 of `run_autots`) needs one form: the same points, ordered from the first
//! structure to the second, with the two input structures as the ends. The files of the
//! methods stay as they are.
//!
//! MultiOptPy: no counterpart (new in mopt-rs; MultiOptPy's `make_traj_file_for_DM` joins
//! the two images of iEIP only).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::path::{Path, PathBuf};

use faer::Col;

use super::ieipmain_methods::MethodOutcome;
use super::ieipmain_settings::IeipMethod;
use crate::domain::{MolecularSystem, SpinCharge};
use crate::error::MoptResult;
use crate::io::xyz::read_xyz_trajectory_file;
use crate::math::linalg::col_norm;
use crate::workflow::pair_search::files::write_frames;

/// Two structures are the same when they are this near (Bohr, norm of the difference).
const SAME_BOHR: f64 = 1.0e-6;

/// The file in which `method` leaves the points it went through, for a first structure
/// with the file stem `stem`; `None` for the methods whose points come with their outcome
/// and for those that make no path between two structures.
fn points_file(method: &IeipMethod, stem: &str) -> Option<String> {
    match method {
        IeipMethod::Ieip => Some(format!("{stem}_traj.xyz")),
        IeipMethod::TwoPshs => Some("ADD_0_trajectory.xyz".to_string()),
        IeipMethod::NewtonTrajectory => Some("newton_trajectory.xyz".to_string()),
        IeipMethod::Dimer => Some("dimer_trajectory.xyz".to_string()),
        IeipMethod::SpringPair => None,
        IeipMethod::ModelFunction(_)
        | IeipMethod::Addf
        | IeipMethod::Gsm
        | IeipMethod::SeGsm(_)
        | IeipMethod::Fsm
        | IeipMethod::DsAfir
        | IeipMethod::Lmm => None,
    }
}

/// `points` from `first` to `last` (when there is a second structure),
/// supplied in path order, with the two structures as ends where the points do not start
/// and end there.
fn from_first_to_last(
    points: Vec<(Col<f64>, Option<f64>)>,
    first: &Col<f64>,
    last: Option<&Col<f64>>,
) -> Vec<(Col<f64>, Option<f64>)> {
    // The producer knows the order of the images. Distances between endpoints
    // in different rigid frames cannot determine that order.
    let mut out = Vec::with_capacity(points.len() + 2);
    if points
        .first()
        .is_none_or(|(x, _)| col_norm(&(x - first)) > SAME_BOHR)
    {
        out.push((first.clone(), None));
    }
    out.extend(points);
    if let Some(last) = last {
        if out
            .last()
            .is_none_or(|(x, _)| col_norm(&(x - last)) > SAME_BOHR)
        {
            out.push((last.clone(), None));
        }
    }
    out
}

/// Write the path of the search as `<stem>_path.xyz` in `dir` and return the file; `None`
/// when the method makes no path between structures (ADDF, a model function) or left no
/// points.
pub fn write_path(
    method: &IeipMethod,
    outcome: &MethodOutcome,
    structures: &[MolecularSystem],
    stem: &str,
    dir: &Path,
) -> MoptResult<Option<PathBuf>> {
    let Some(template) = structures.first() else {
        return Ok(None);
    };
    let first = template.coords_flat_bohr();
    let last = (structures.len() > 1).then(|| structures[structures.len() - 1].coords_flat_bohr());
    write_path_of(
        method,
        outcome,
        (&first, last.as_ref()),
        &|x| template.with_flat_coords_bohr(x),
        &|s| (s.n_atoms() == template.n_atoms()).then(|| s.coords_flat_bohr()),
        stem,
        dir,
    )
}

/// [`write_path`] for points that are not Cartesian coordinates of the atoms (the variables
/// of the atoms and the cell of `ieipmain -vc`): `ends` are the points of the first and the
/// last structure, `structure` makes the structure of a point, and `point` the point of a
/// structure read from the file of a method (`None`: not one of the points).
///
/// MultiOptPy: no counterpart (new in mopt-rs).
pub fn write_path_of(
    method: &IeipMethod,
    outcome: &MethodOutcome,
    ends: (&Col<f64>, Option<&Col<f64>>),
    structure: &dyn Fn(&Col<f64>) -> MoptResult<MolecularSystem>,
    point: &dyn Fn(&MolecularSystem) -> Option<Col<f64>>,
    stem: &str,
    dir: &Path,
) -> MoptResult<Option<PathBuf>> {
    // A path without points is no path.
    let returned = outcome.path.as_ref().filter(|path| !path.is_empty());
    let points: Vec<(Col<f64>, Option<f64>)> = match (returned, points_file(method, stem)) {
        (Some(path), _) => path
            .iter()
            .map(|(x, e)| (x.clone(), e.is_finite().then_some(*e)))
            .collect(),
        (None, Some(name)) => {
            let file = dir.join(name);
            if !file.is_file() {
                return Ok(None);
            }
            read_xyz_trajectory_file(&file, SpinCharge::default(), None)?
                .frames
                .iter()
                .filter_map(|frame| point(&frame.system))
                .map(|x| (x, None))
                .collect()
        }
        // A method that leaves no points of its own (the local minimax method moves a
        // direction, not a structure): its estimates between the two structures.
        (None, None) if !matches!(method, IeipMethod::Addf | IeipMethod::ModelFunction(_)) => {
            outcome
                .transition_states
                .iter()
                .map(|(x, e)| (x.clone(), *e))
                .collect()
        }
        (None, None) => return Ok(None),
    };
    if points.is_empty() {
        return Ok(None);
    }
    let ordered = from_first_to_last(points, ends.0, ends.1);
    let mut frames = Vec::with_capacity(ordered.len());
    for (k, (x, energy)) in ordered.iter().enumerate() {
        let comment = match energy {
            Some(energy) => format!("Path point {k} energy {energy:.10}"),
            None => format!("Path point {k}"),
        };
        frames.push((structure(x)?, comment));
    }
    let file = dir.join(format!("{stem}_path.xyz"));
    write_frames(&file, &frames)?;
    Ok(Some(file))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn point(x: f64) -> Col<f64> {
        Col::from_fn(3, |k| if k == 0 { x } else { 0.0 })
    }

    #[test]
    fn attaching_the_input_ends_preserves_both_branches_and_their_energies() {
        let points = [(1.0, 0.4), (1.8, 0.9), (2.2, 0.8), (3.0, 0.2)]
            .map(|(x, e)| (point(x), Some(e)))
            .to_vec();
        let path = from_first_to_last(points.clone(), &point(0.0), Some(&point(4.0)));
        assert_eq!(&path[1..5], points);
        assert_eq!(path[0], (point(0.0), None));
        assert_eq!(path[5], (point(4.0), None));
        assert_eq!(
            from_first_to_last(path.clone(), &point(0.0), Some(&point(4.0))),
            path
        );
    }

    #[test]
    fn every_method_between_two_structures_has_its_points_somewhere() {
        for method in [
            IeipMethod::Ieip,
            IeipMethod::TwoPshs,
            IeipMethod::NewtonTrajectory,
            IeipMethod::Dimer,
        ] {
            assert!(points_file(&method, "a").is_some(), "{}", method.title());
        }
        assert_eq!(points_file(&IeipMethod::Ieip, "a").unwrap(), "a_traj.xyz");
        // The strings return their points; ADDF goes out from one structure.
        assert!(points_file(&IeipMethod::Gsm, "a").is_none());
        assert!(points_file(&IeipMethod::Addf, "a").is_none());
    }
}
