//! Collective variables from text, one form for every place `mdmain` takes them (`-cv`
//! for the output and the analysis, `-cv_bias` for harmonic restraints, `-umbrella cv=`):
//!
//! | form | variable (atomic units) |
//! |---|---|
//! | `distance:1,2` | distance of two atoms (Bohr) |
//! | `distance:1,2,3/4,5` | distance of the centroids of two groups |
//! | `angle:1,2,3` | angle at the second atom (radians) |
//! | `dihedral:1,2,3,4` | dihedral angle (radians, period `2 pi`) |
//! | `coordination:1,2/3,4[:r0=ANG][:n=6][:m=12]` | `sum (1 - (r/r0)^n) / (1 - (r/r0)^m)` over the pairs of the groups (r0 2.0 Angstrom) |
//! | `path_s:FILE[:lambda=PER_ANG2]`, `path_z:FILE[:...]` | progress along and distance from the path of the frames of an XYZ file (all atoms) |
//! | anything else | an expression of the `-ubias` language, e.g. `dist(1,2) - dist(2,3)` |
//!
//! Atoms are counted from 1. Without `lambda`, the path variables take `2.3 / <d^2>` with
//! `<d^2>` the mean squared distance per atom of neighbouring frames (Branduardi,
//! Gervasio, and Parrinello, J. Chem. Phys. 2007, 126, 054103).
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::path::Path;

use super::cv::Cv;
use crate::domain::{AtomId, SpinCharge, Units};
use crate::error::{MoptError, MoptResult};
use crate::io::xyz::read_xyz_trajectory_file;

fn atoms(list: &str) -> MoptResult<Vec<AtomId>> {
    list.split(',')
        .filter(|t| !t.trim().is_empty())
        .map(|t| match t.trim().parse::<usize>() {
            Ok(i) if i >= 1 => AtomId::from_one_based(i),
            _ => Err(MoptError::config(format!(
                "'{t}' is not an atom number (counted from 1)"
            ))),
        })
        .collect()
}

fn option(parts: &[&str], key: &str) -> MoptResult<Option<f64>> {
    for p in parts {
        if let Some(v) = p.trim().strip_prefix(&format!("{key}=")) {
            return v
                .trim()
                .parse::<f64>()
                .map(Some)
                .map_err(|_| MoptError::config(format!("{key}={v}: not a number")));
        }
    }
    Ok(None)
}

fn exactly(found: Vec<AtomId>, n: usize, kind: &str) -> MoptResult<Vec<AtomId>> {
    if found.len() == n {
        Ok(found)
    } else {
        Err(MoptError::config(format!(
            "{kind} needs {n} atoms, got {}",
            found.len()
        )))
    }
}

fn path(kind: &str, parts: &[&str], progress: bool) -> MoptResult<Cv> {
    let file = parts
        .first()
        .ok_or_else(|| MoptError::config(format!("{kind}:FILE needs the file of the path")))?;
    let traj = read_xyz_trajectory_file(Path::new(file.trim()), SpinCharge::default(), None)?;
    let frames: Vec<Vec<f64>> = traj
        .frames
        .iter()
        .map(|f| f.system.coords_flat_bohr().iter().copied().collect())
        .collect();
    let n = traj.frames.first().map_or(0, |f| f.system.n_atoms());
    let lambda = match option(&parts[1..], "lambda")? {
        Some(l) => l * Units::BOHR_TO_ANGSTROM * Units::BOHR_TO_ANGSTROM,
        None => {
            let d2: Vec<f64> = frames
                .windows(2)
                .map(|w| {
                    w[0].iter()
                        .zip(&w[1])
                        .map(|(a, b)| (a - b).powi(2))
                        .sum::<f64>()
                        / n.max(1) as f64
                })
                .collect();
            let mean = d2.iter().sum::<f64>() / d2.len().max(1) as f64;
            if mean <= 0.0 {
                return Err(MoptError::config("the frames of the path coincide"));
            }
            2.3 / mean
        }
    };
    let all = (0..n).map(AtomId::new).collect();
    Cv::path(all, frames, lambda, progress)
}

/// The collective variable of `spec`.
pub fn parse_cv(spec: &str) -> MoptResult<Cv> {
    let text = spec.trim();
    let Some((kind, rest)) = text.split_once(':') else {
        return Cv::expression(text, None);
    };
    let parts: Vec<&str> = rest.split(':').collect();
    let first = parts.first().copied().unwrap_or("");
    match kind.trim().to_ascii_lowercase().as_str() {
        "distance" => match first.split_once('/') {
            Some((a, b)) => Ok(Cv::distance_of_groups(atoms(a)?, atoms(b)?)),
            None => {
                let a = exactly(atoms(first)?, 2, "a distance")?;
                Ok(Cv::distance(a[0], a[1]))
            }
        },
        "angle" => {
            let a = exactly(atoms(first)?, 3, "an angle")?;
            Ok(Cv::angle(a[0], a[1], a[2]))
        }
        "dihedral" => {
            let a = exactly(atoms(first)?, 4, "a dihedral angle")?;
            Ok(Cv::dihedral(a[0], a[1], a[2], a[3]))
        }
        "coordination" => {
            let (a, b) = first.split_once('/').ok_or_else(|| {
                MoptError::config("coordination:A/B needs two groups, as coordination:1,2/3,4")
            })?;
            let r0 = option(&parts[1..], "r0")?.unwrap_or(2.0) * Units::ANGSTROM_TO_BOHR;
            let n = option(&parts[1..], "n")?.unwrap_or(6.0) as u32;
            let m = option(&parts[1..], "m")?.unwrap_or(12.0) as u32;
            Cv::coordination(atoms(a)?, atoms(b)?, r0, n, m)
        }
        "path_s" => path("path_s", &parts, true),
        "path_z" => path("path_z", &parts, false),
        "expr" => Cv::expression(rest, None),
        // An expression with a colon of its own.
        _ => Cv::expression(text, None),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::domain::{positions, Angstrom, Element, MolecularSystem};

    #[test]
    fn every_form_gives_its_variable() {
        let sys = MolecularSystem::new(
            vec![Element::O, Element::H, Element::H],
            &positions::<Angstrom>(&[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]),
            SpinCharge::default(),
            None,
        )
        .unwrap();
        let d = parse_cv("distance:1,2").unwrap().value(&sys).unwrap();
        assert!((d * Units::BOHR_TO_ANGSTROM - 0.96).abs() < 1e-9);
        let a = parse_cv("angle:2,1,3").unwrap().value(&sys).unwrap();
        assert!((a.to_degrees() - (180.0 - (0.93_f64).atan2(0.24).to_degrees())).abs() < 1e-6);
        let e = parse_cv("dist(1,2)").unwrap().value(&sys).unwrap();
        assert!((e - d).abs() < 1e-12);
        let c = parse_cv("coordination:1/2,3:r0=1.2")
            .unwrap()
            .value(&sys)
            .unwrap();
        assert!(c > 1.0 && c < 2.0, "{c}");
        assert!(parse_cv("angle:1,2").is_err());
        assert!(parse_cv("coordination:1,2").is_err());
    }
}
