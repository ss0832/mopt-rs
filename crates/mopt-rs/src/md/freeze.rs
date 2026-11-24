//! Molecular dynamics that does not move along chosen modes of the Hessian.
//!
//! The Hessian is diagonalized in mass-weighted coordinates `q = M^(1/2) x` (with the
//! rigid-body motions projected out where they are conserved, and held variables left
//! out). The chosen modes `u_k` (negative curvature by default; or below an eigenvalue, by
//! index, or given vectors) are orthonormalized, and the coordinates `u_k . M^(1/2) x` are
//! held at their values as linear constraints of [`super::constraints::Rattle`]: the
//! velocities, the forces (through the multipliers), and the displacements keep no
//! component along the modes (`P = 1 - sum u_k u_k^T` in mass-weighted coordinates), and
//! the frozen modes are taken out of the count of degrees of freedom. When the modes are
//! computed again during the run, each frozen mode is followed to the new mode it overlaps
//! most (an assignment over all of them), so that the same direction stays frozen as it
//! turns.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::{Col, Mat};

use super::constraints::LinearRow;
use crate::error::{MoptError, MoptResult};
use crate::math::assignment::linear_sum_assignment;
use crate::math::linalg::{orthonormalize_columns, symmetric_eigen};

/// Which modes are frozen.
#[derive(Debug, Clone, PartialEq)]
pub enum ModeSelection {
    /// Those of negative curvature.
    Negative,
    /// Those with a mass-weighted eigenvalue below the value (Hartree / (Bohr^2 m_e); for
    /// plain variables the unit of the objective over the square of the variables).
    Below(f64),
    /// These, counted from the lowest eigenvalue (from 0), rigid-body and held motions
    /// skipped.
    Indices(Vec<usize>),
    /// These Cartesian directions (one per mode; mass weighting is applied).
    Vectors(Vec<Col<f64>>),
}

/// Settings of mode freezing.
#[derive(Debug, Clone, PartialEq)]
pub struct FreezeSettings {
    /// Which modes.
    pub selection: ModeSelection,
    /// Compute the modes again every this many steps (`None`: at the start only).
    pub update_interval: Option<usize>,
}

/// One computation of the modes, for the log.
#[derive(Debug, Clone, PartialEq)]
pub struct FreezeUpdate {
    /// Step of the Hessian.
    pub step: usize,
    /// Mass-weighted eigenvalues of the frozen modes.
    pub eigenvalues: Vec<f64>,
    /// Overlap of each frozen mode with the one it replaced (none at the start).
    pub overlaps: Vec<f64>,
}

/// The frozen modes of a run.
#[derive(Debug, Clone)]
pub struct ModeFreezing {
    settings: FreezeSettings,
    modes: Vec<Col<f64>>,
    history: Vec<FreezeUpdate>,
}

/// The mass-weighted Hessian `W^(1/2) H W^(1/2)` (`W` the inverse masses), with the
/// directions of `rigid` (orthonormal mass-weighted columns) projected out.
pub fn mass_weighted_hessian(hessian: &Mat<f64>, w: &[f64], rigid: Option<&Mat<f64>>) -> Mat<f64> {
    let n = hessian.nrows();
    let s: Vec<f64> = w.iter().map(|x| libm::sqrt(*x)).collect();
    let mut h = Mat::from_fn(n, n, |i, j| {
        0.5 * (hessian[(i, j)] + hessian[(j, i)]) * s[i] * s[j]
    });
    if let Some(d) = rigid.filter(|d| d.ncols() > 0) {
        let p = Mat::from_fn(n, n, |i, j| {
            let delta = if i == j { 1.0 } else { 0.0 };
            delta - (0..d.ncols()).map(|k| d[(i, k)] * d[(j, k)]).sum::<f64>()
        });
        h = &p * &h * &p;
    }
    h
}

/// Eigenvalues and eigenvectors of the motions that are neither held nor rigid.
fn vibrations(
    h: &Mat<f64>,
    w: &[f64],
    rigid: Option<&Mat<f64>>,
) -> MoptResult<Vec<(f64, Col<f64>)>> {
    let n = h.nrows();
    let (values, vectors) = symmetric_eigen(h)?;
    let mut out = Vec::new();
    for k in 0..n {
        let u = Col::from_fn(n, |i| vectors[(i, k)]);
        let held: f64 = (0..n).filter(|&i| w[i] == 0.0).map(|i| u[i] * u[i]).sum();
        let along_rigid: f64 = rigid.map_or(0.0, |d| {
            (0..d.ncols())
                .map(|c| (0..n).map(|i| d[(i, c)] * u[i]).sum::<f64>().powi(2))
                .sum()
        });
        if held + along_rigid < 0.5 {
            out.push((values[k], u));
        }
    }
    Ok(out)
}

impl ModeFreezing {
    /// Freezing with `settings`; the modes are chosen with the first Hessian (or from the
    /// given vectors).
    pub fn new(settings: FreezeSettings) -> MoptResult<Self> {
        if settings.update_interval == Some(0) {
            return Err(MoptError::config(
                "the interval of the mode updates must be positive",
            ));
        }
        Ok(Self {
            settings,
            modes: Vec::new(),
            history: Vec::new(),
        })
    }

    /// Whether the Hessian is needed at `step`.
    pub fn needs_hessian(&self, step: usize) -> bool {
        if matches!(self.settings.selection, ModeSelection::Vectors(_)) {
            return false;
        }
        step == 0
            || self
                .settings
                .update_interval
                .is_some_and(|n| step.is_multiple_of(n))
    }

    /// The frozen modes (orthonormal, mass-weighted).
    pub fn modes(&self) -> &[Col<f64>] {
        &self.modes
    }

    /// The computations of the modes so far.
    pub fn history(&self) -> &[FreezeUpdate] {
        &self.history
    }

    /// Choose the modes at the start from given vectors (no Hessian needed).
    pub fn start_from_vectors(&mut self, w: &[f64]) -> MoptResult<bool> {
        let ModeSelection::Vectors(vectors) = &self.settings.selection else {
            return Ok(false);
        };
        let n = w.len();
        let weighted: Vec<Col<f64>> = vectors
            .iter()
            .map(|d| {
                if d.nrows() != n {
                    return Err(MoptError::config(format!(
                        "a frozen direction has {} components for {n} variables",
                        d.nrows()
                    )));
                }
                // u = M^(1/2) d on the variables that move.
                Ok(Col::from_fn(n, |i| {
                    if w[i] > 0.0 {
                        d[i] / libm::sqrt(w[i])
                    } else {
                        0.0
                    }
                }))
            })
            .collect::<MoptResult<_>>()?;
        let basis = orthonormalize_columns(n, &weighted, 1e-10);
        self.modes = (0..basis.ncols())
            .map(|k| Col::from_fn(n, |i| basis[(i, k)]))
            .collect();
        if self.modes.is_empty() {
            return Err(MoptError::config(
                "the frozen directions vanish on the variables that move",
            ));
        }
        self.history.push(FreezeUpdate {
            step: 0,
            eigenvalues: Vec::new(),
            overlaps: Vec::new(),
        });
        Ok(true)
    }

    /// Choose (step 0) or follow (later steps) the modes with the Hessian at `step`.
    pub fn update(
        &mut self,
        step: usize,
        hessian: &Mat<f64>,
        w: &[f64],
        rigid: Option<&Mat<f64>>,
    ) -> MoptResult<()> {
        let h = mass_weighted_hessian(hessian, w, rigid);
        let candidates = vibrations(&h, w, rigid)?;
        if self.modes.is_empty() {
            return self.choose(step, &candidates);
        }
        self.follow(step, &candidates)
    }

    fn choose(&mut self, step: usize, candidates: &[(f64, Col<f64>)]) -> MoptResult<()> {
        let top = candidates.iter().fold(0.0_f64, |m, (v, _)| m.max(v.abs()));
        let picked: Vec<usize> = match &self.settings.selection {
            ModeSelection::Negative => (0..candidates.len())
                .filter(|&k| candidates[k].0 < -1e-8 * top.max(1e-300))
                .collect(),
            ModeSelection::Below(limit) => (0..candidates.len())
                .filter(|&k| candidates[k].0 < *limit)
                .collect(),
            ModeSelection::Indices(list) => {
                for &k in list {
                    if k >= candidates.len() {
                        return Err(MoptError::config(format!(
                            "mode {k} does not exist ({} modes)",
                            candidates.len()
                        )));
                    }
                }
                list.clone()
            }
            ModeSelection::Vectors(_) => Vec::new(),
        };
        if picked.is_empty() {
            return Err(MoptError::config(
                "no mode of the Hessian matches the selection of frozen modes (with the default \
                 selection: the structure has no negative curvature)",
            ));
        }
        self.modes = picked.iter().map(|&k| candidates[k].1.clone()).collect();
        self.history.push(FreezeUpdate {
            step,
            eigenvalues: picked.iter().map(|&k| candidates[k].0).collect(),
            overlaps: Vec::new(),
        });
        Ok(())
    }

    fn follow(&mut self, step: usize, candidates: &[(f64, Col<f64>)]) -> MoptResult<()> {
        let (old, new) = (self.modes.len(), candidates.len());
        if new < old {
            return Err(MoptError::conv(
                "fewer modes than frozen ones at the update",
            ));
        }
        let overlap =
            |a: &Col<f64>, b: &Col<f64>| a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f64>();
        let cost = Mat::from_fn(old, new, |k, l| {
            -overlap(&self.modes[k], &candidates[l].1).abs()
        });
        let assigned = linear_sum_assignment(&cost)?;
        let mut modes = Vec::with_capacity(old);
        let mut values = Vec::with_capacity(old);
        let mut overlaps = Vec::with_capacity(old);
        for (k, &l) in assigned.iter().enumerate() {
            let mut u = candidates[l].1.clone();
            let o = overlap(&self.modes[k], &u);
            if o < 0.0 {
                u = -u;
            }
            modes.push(u);
            values.push(candidates[l].0);
            overlaps.push(o.abs());
        }
        self.modes = modes;
        self.history.push(FreezeUpdate {
            step,
            eigenvalues: values,
            overlaps,
        });
        Ok(())
    }

    /// The linear constraints that hold the coordinates of the frozen modes at their values
    /// at `x` (inverse masses `w`).
    pub fn rows(&self, x: &Col<f64>, w: &[f64]) -> Vec<LinearRow> {
        self.modes
            .iter()
            .map(|u| {
                // r = M^(1/2) u.
                let row = Col::from_fn(u.nrows(), |i| {
                    if w[i] > 0.0 {
                        u[i] / libm::sqrt(w[i])
                    } else {
                        0.0
                    }
                });
                let target = row.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
                LinearRow { row, target }
            })
            .collect()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn the_negative_mode_is_chosen_and_followed_as_it_turns() {
        let w = [1.0, 1.0];
        let h = Mat::from_fn(2, 2, |i, j| [[-1.0, 0.0], [0.0, 2.0]][i][j]);
        let mut f = ModeFreezing::new(FreezeSettings {
            selection: ModeSelection::Negative,
            update_interval: Some(5),
        })
        .unwrap();
        assert!(f.needs_hessian(0) && f.needs_hessian(10) && !f.needs_hessian(3));
        f.update(0, &h, &w, None).unwrap();
        assert_eq!(f.modes().len(), 1);
        assert!((f.modes()[0][0].abs() - 1.0).abs() < 1e-12);
        // The modes turn by 30 degrees and their curvatures exchange order: the frozen one
        // is followed by overlap, not by its eigenvalue.
        let (c, s) = (30f64.to_radians().cos(), 30f64.to_radians().sin());
        let r = Mat::from_fn(2, 2, |i, j| [[c, -s], [s, c]][i][j]);
        let d = Mat::from_fn(2, 2, |i, j| [[3.0, 0.0], [0.0, 1.0]][i][j]);
        let turned = &r * &d * r.transpose();
        f.update(5, &turned, &w, None).unwrap();
        let last = f.history().last().unwrap();
        assert!((last.eigenvalues[0] - 3.0).abs() < 1e-10);
        assert!((last.overlaps[0] - c).abs() < 1e-10);
        let rows = f.rows(&Col::from_fn(2, |k| [0.5, 0.25][k]), &w);
        assert!((rows[0].target.abs() - (0.5 * c + 0.25 * s)).abs() < 1e-10);
        let mut none = ModeFreezing::new(FreezeSettings {
            selection: ModeSelection::Negative,
            update_interval: None,
        })
        .unwrap();
        assert!(none.update(0, &d, &w, None).is_err());
    }
}
