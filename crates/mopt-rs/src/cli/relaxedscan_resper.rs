//! The ReSPer references of a scan (`-resper FILE`): the structures of a path or trajectory
//! (xyz frames) or the points of plain variables (a CSV with a column per variable, such as
//! `<stem>_traj.csv` of optmain, or numbers per line), thinned to at most `resper_max`, their
//! map ([`crate::analysis::resper`]), and the scanned coordinates as constraints.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::path::Path;
use std::sync::Arc;

use faer::Col;

use super::relaxedscan_settings::ResperSettings;
use crate::analysis::resper::preimage::aligned_references;
use crate::analysis::resper::{ResperCoordinate, ResperMap, ResperVariable};
use crate::core_infra::log::Logger;
use crate::domain::MolecularSystem;
use crate::error::{MoptError, MoptResult, ResultExt};
use crate::io::xyz::read_xyz_trajectory_file;
use crate::pes::constraint::{Constraint, ConstraintFunction, ConstraintKind};

/// The references of a scan and their map.
pub struct ResperReferences {
    /// The map.
    pub map: Arc<ResperMap>,
    /// The references superposed in their order (for pre-images).
    pub aligned: Vec<Col<f64>>,
    /// Their coordinates on the map (PCo1, PCo2; PCo2 zero for references on a line).
    pub path: (Vec<f64>, Vec<f64>),
    /// Their indices in the file.
    pub frames: Vec<usize>,
}

/// Every `stride`-th of `count` items, then evenly thinned to at most `max` (the first and
/// the last kept).
fn chosen(count: usize, stride: usize, max: usize) -> Vec<usize> {
    let taken: Vec<usize> = (0..count).step_by(stride.max(1)).collect();
    if taken.len() <= max {
        return taken;
    }
    let mut out: Vec<usize> = (0..max)
        .map(|k| taken[k * (taken.len() - 1) / (max - 1).max(1)])
        .collect();
    out.dedup();
    out
}

fn finish(
    map: ResperMap,
    frames: Vec<usize>,
    log: &Logger,
    what: &str,
) -> MoptResult<ResperReferences> {
    let aligned = aligned_references(&map)?;
    let xs = (0..map.len())
        .map(|i| map.reference_coordinate(i, 0))
        .collect();
    let ys = (0..map.len())
        .map(|i| {
            if map.components() > 1 {
                map.reference_coordinate(i, 1)
            } else {
                0.0
            }
        })
        .collect();
    let shares: Vec<String> = map
        .contributions()
        .iter()
        .enumerate()
        .map(|(k, c)| format!("PCo{} {:.1} %", k + 1, 100.0 * c))
        .collect();
    log.info(&format!(
        "ReSPer map of {} {what}: {} (dimensional reproducibility {:.4}).",
        map.len(),
        shares.join(", "),
        map.reproducibility()
    ));
    let ranges: Vec<String> = (0..map.components())
        .map(|k| {
            let values = (0..map.len()).map(|i| map.reference_coordinate(i, k));
            let (lo, hi) = values.fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), v| {
                (l.min(v), h.max(v))
            });
            format!(
                "PCo{} from {lo:.6} to {hi:.6} (first {:.6}, last {:.6})",
                k + 1,
                map.reference_coordinate(0, k),
                map.reference_coordinate(map.len() - 1, k)
            )
        })
        .collect();
    log.info(&format!(
        "The references on the map: {}.",
        ranges.join("; ")
    ));
    Ok(ResperReferences {
        map: Arc::new(map),
        aligned,
        path: (xs, ys),
        frames,
    })
}

/// The references of a scan of structures like `template`.
pub fn load_structures(
    settings: &ResperSettings,
    template: &MolecularSystem,
    log: &Logger,
) -> MoptResult<ResperReferences> {
    let file = &settings.file;
    let all = read_xyz_trajectory_file(file, template.spin_charge, None).context(file.display())?;
    let frames = chosen(all.frames.len(), settings.stride, settings.max);
    let mut structures = Vec::with_capacity(frames.len());
    for &k in &frames {
        let mut s = all.frames[k].system.clone();
        if s.elements != template.elements {
            return Err(MoptError::config(format!(
                "{}: frame {} has other atoms than the input structure",
                file.display(),
                k + 1
            )));
        }
        if s.cell.is_none() {
            s.cell.clone_from(&template.cell);
        }
        s.spin_charge = template.spin_charge;
        structures.push(s);
    }
    if frames.len() < all.frames.len() {
        log.info(&format!(
            "ReSPer references: {} of the {} frames of {} (resper_stride, resper_max).",
            frames.len(),
            all.frames.len(),
            file.display()
        ));
    }
    let map = ResperMap::from_structures(structures, &settings.atoms, settings.mass_weighted)?;
    finish(map, frames, log, "reference structures")
}

/// Points of plain variables from a file: a header with the names of the variables (their
/// columns are taken, as `<stem>_traj.csv` of optmain has them) or numbers only, one point per
/// line (separated by commas or blanks; `#` starts a comment).
pub fn read_points(path: &Path, names: &[String]) -> MoptResult<Vec<Col<f64>>> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| MoptError::parse(format!("cannot read {}: {e}", path.display())))?;
    let rows: Vec<Vec<String>> = text
        .lines()
        .map(|l| l.split('#').next().unwrap_or("").trim())
        .filter(|l| !l.is_empty())
        .map(|l| {
            l.split([',', ' ', '\t'])
                .filter(|t| !t.is_empty())
                .map(str::to_string)
                .collect()
        })
        .collect();
    let Some(first) = rows.first() else {
        return Err(MoptError::parse(format!("{}: no points", path.display())));
    };
    let header = first.iter().any(|t| t.parse::<f64>().is_err());
    let columns: Vec<usize> = if header {
        names
            .iter()
            .map(|n| {
                first.iter().position(|h| h == n).ok_or_else(|| {
                    MoptError::parse(format!(
                        "{}: no column of the variable '{n}' (columns: {})",
                        path.display(),
                        first.join(", ")
                    ))
                })
            })
            .collect::<MoptResult<_>>()?
    } else {
        (0..names.len()).collect()
    };
    let body = if header { &rows[1..] } else { &rows[..] };
    body.iter()
        .enumerate()
        .map(|(k, row)| {
            let values = columns
                .iter()
                .map(|&c| {
                    row.get(c)
                        .and_then(|t| t.parse::<f64>().ok())
                        .ok_or_else(|| {
                            MoptError::parse(format!(
                                "{}: point {} has no number in column {}",
                                path.display(),
                                k + 1,
                                c + 1
                            ))
                        })
                })
                .collect::<MoptResult<Vec<f64>>>()?;
            if !header && row.len() != names.len() {
                return Err(MoptError::parse(format!(
                    "{}: point {} has {} numbers for {} variables",
                    path.display(),
                    k + 1,
                    row.len(),
                    names.len()
                )));
            }
            Ok(Col::from_fn(values.len(), |i| values[i]))
        })
        .collect()
}

/// The references of a scan of plain variables with `names`.
pub fn load_points(
    settings: &ResperSettings,
    names: &[String],
    log: &Logger,
) -> MoptResult<ResperReferences> {
    let all = read_points(&settings.file, names)?;
    let frames = chosen(all.len(), settings.stride, settings.max);
    let points = frames.iter().map(|&k| all[k].clone()).collect();
    let map = ResperMap::from_points(points)?;
    finish(map, frames, log, "reference points")
}

/// The constraint that holds the ReSPer coordinate `component` of structures at `value`.
pub fn structure_constraint(
    references: &ResperReferences,
    component: usize,
    value: f64,
    curvature: bool,
) -> MoptResult<Constraint> {
    let term =
        ResperCoordinate::new(Arc::clone(&references.map), component)?.with_curvature(curvature);
    Ok(Constraint {
        // Computed through a superposition: a residual above the guarantee is a warning.
        kind: ConstraintKind::Function(ConstraintFunction::new(Arc::new(term)).warning_only()),
        target: Some(value),
        anchor: None,
    })
}

/// The constraint of plain variables that holds the ReSPer coordinate `component`.
pub fn variable_constraint(
    references: &ResperReferences,
    component: usize,
) -> MoptResult<ResperVariable> {
    ResperVariable::new(Arc::clone(&references.map), component)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    #[test]
    fn references_are_thinned_and_points_read() {
        assert_eq!(chosen(10, 1, 20), (0..10).collect::<Vec<_>>());
        assert_eq!(chosen(10, 3, 20), vec![0, 3, 6, 9]);
        let few = chosen(1000, 1, 5);
        assert_eq!(few, vec![0, 249, 499, 749, 999]);
        let dir = std::env::temp_dir().join(format!("mopt_resper_points_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let names = vec!["x".to_string(), "y".to_string()];
        let csv = dir.join("traj.csv");
        std::fs::write(
            &csv,
            "iteration,value,x,y,d/dx\n0,1.0,0.5,1.5,0\n1,0.9,0.4,1.4,0\n",
        )
        .unwrap();
        let points = read_points(&csv, &names).unwrap();
        assert_eq!(points.len(), 2);
        assert_eq!((points[1][0], points[1][1]), (0.4, 1.4));
        let plain = dir.join("points.txt");
        std::fs::write(&plain, "# a path\n0.1 0.2\n0.3, 0.4\n").unwrap();
        assert_eq!(read_points(&plain, &names).unwrap().len(), 2);
        std::fs::write(&plain, "0.1 0.2 0.3\n").unwrap();
        assert!(read_points(&plain, &names).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
