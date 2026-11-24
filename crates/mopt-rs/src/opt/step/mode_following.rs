//! Tracking of Hessian eigenvectors from one iteration to the next (mode following).
//!
//! The modes to follow uphill are identified by their overlap with reference vectors instead of
//! by their position in the spectrum (Baker, J. Comput. Chem. 1986, 7, 385; the maximum overlap
//! idea of Gilbert, Besley, and Gill, J. Phys. Chem. A 2008, 112, 13164). Options, as in
//! MultiOptPy:
//! - overlaps in the Cartesian or in the mass-weighted metric `v1^T M v2`,
//! - static references or references updated by an exponential moving average
//!   `ref <- (1 - a) ref + a v` (`a = 1`: the reference is the last matched mode),
//! - a bias toward modes along the gradient, `score = |overlap| + w |overlap(g, v)|`,
//! - a global assignment of references to modes (Hungarian method) or a greedy one.
//!
//! MultiOptPy: `Optimizer/mode_following.py:ModeFollowing`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::{Col, Mat};

use crate::error::{MoptError, MoptResult};
use crate::math::assignment::linear_sum_assignment;

/// Settings of the mode tracker.
#[derive(Debug, Clone, PartialEq)]
pub struct ModeFollowingSettings {
    /// Index (in ascending eigenvalue order) of the first mode to follow.
    pub target_index: usize,
    /// Update the references with the matched modes.
    pub adaptive: bool,
    /// Weight `a` of the matched mode in the reference update (`1` replaces the reference).
    pub update_rate: f64,
    /// Assign all references at once (Hungarian method) instead of greedily.
    pub hungarian: bool,
    /// Weight of the overlap with the gradient in the matching score.
    pub gradient_weight: f64,
    /// Use the mass-weighted metric (needs atoms; without them all masses are 1).
    pub mass_weighted: bool,
}

impl Default for ModeFollowingSettings {
    fn default() -> Self {
        Self {
            target_index: 0,
            adaptive: true,
            update_rate: 1.0,
            hungarian: true,
            gradient_weight: 0.0,
            mass_weighted: false,
        }
    }
}

impl ModeFollowingSettings {
    /// Settings selected by a method string: the prefixes `mw` (mass-weighted) and `s` (static
    /// references) of `mf_rsirfo`, and the suffixes `:<index>`, `:ema<rate>`, `:grad<weight>`
    /// (for example `mf_rsirfo_bofill:1:ema0.5:grad0.3`).
    pub fn from_method(method: &str) -> MoptResult<Self> {
        let lower = method.to_ascii_lowercase();
        let mut s = Self {
            mass_weighted: lower.contains("mwsmf_rsirfo") || lower.contains("mwmf_rsirfo"),
            adaptive: !lower.contains("smf_rsirfo"),
            ..Self::default()
        };
        s.update_rate = if s.adaptive { 1.0 } else { 0.0 };
        for part in lower.split(':').skip(1).map(str::trim) {
            let bad = || MoptError::config(format!("invalid mode-following option '{part}'"));
            if part.is_empty() {
                continue;
            } else if part.chars().all(|c| c.is_ascii_digit()) {
                s.target_index = part.parse().map_err(|_| bad())?;
            } else if let Some(v) = part.strip_prefix("ema") {
                s.update_rate = v.parse().map_err(|_| bad())?;
                if !s.update_rate.is_finite() {
                    return Err(bad());
                }
                s.adaptive = s.update_rate > 1e-12;
            } else if let Some(v) = part.strip_prefix("grad") {
                s.gradient_weight = v.parse().map_err(|_| bad())?;
                if !s.gradient_weight.is_finite() {
                    return Err(bad());
                }
            } else {
                return Err(bad());
            }
        }
        Ok(s)
    }
}

/// Tracker of `count` modes.
#[derive(Debug, Clone)]
pub struct ModeFollowing {
    settings: ModeFollowingSettings,
    count: usize,
    references: Vec<Col<f64>>,
    indices: Vec<usize>,
    /// Diagonal of the metric (masses per coordinate); `None` for the Cartesian metric.
    weights: Option<Col<f64>>,
}

impl ModeFollowing {
    /// Tracker of `count` modes (the saddle order).
    pub fn new(count: usize, settings: ModeFollowingSettings) -> Self {
        Self {
            settings,
            count,
            references: Vec::new(),
            indices: Vec::new(),
            weights: None,
        }
    }

    /// Settings.
    pub fn settings(&self) -> &ModeFollowingSettings {
        &self.settings
    }

    /// Masses per coordinate for the mass-weighted metric (ignored unless it is selected).
    pub fn set_masses(&mut self, masses: Col<f64>) {
        if self.settings.mass_weighted {
            self.weights = Some(masses);
        }
    }

    /// Whether references have been set.
    pub fn is_initialized(&self) -> bool {
        !self.references.is_empty() || self.count == 0
    }

    /// Indices of the modes matched last.
    pub fn indices(&self) -> &[usize] {
        &self.indices
    }

    fn inner(&self, a: &Col<f64>, b: &Col<f64>) -> f64 {
        match &self.weights {
            Some(w) if w.nrows() == a.nrows() => (0..a.nrows()).map(|i| a[i] * w[i] * b[i]).sum(),
            _ => (0..a.nrows()).map(|i| a[i] * b[i]).sum(),
        }
    }

    /// Normalized overlap in `[-1, 1]` (`0` when a vector vanishes).
    pub fn overlap(&self, a: &Col<f64>, b: &Col<f64>) -> f64 {
        let (na, nb) = (libm::sqrt(self.inner(a, a)), libm::sqrt(self.inner(b, b)));
        if na < 1e-12 || nb < 1e-12 {
            0.0
        } else {
            self.inner(a, b) / (na * nb)
        }
    }

    fn column(m: &Mat<f64>, j: usize) -> Col<f64> {
        Col::from_fn(m.nrows(), |r| m[(r, j)])
    }

    /// Take the modes `target_index .. target_index + count` of `eigvecs` (columns, ascending
    /// eigenvalues) as references and return their indices. Falls back to the lowest modes
    /// when the range exceeds the spectrum.
    pub fn set_references(&mut self, eigvecs: &Mat<f64>) -> Vec<usize> {
        let m = eigvecs.ncols();
        let count = self.count.min(m);
        let start = if self.settings.target_index + count > m {
            0
        } else {
            self.settings.target_index
        };
        self.indices = (start..start + count).collect();
        self.references = self
            .indices
            .iter()
            .map(|&j| Self::column(eigvecs, j))
            .collect();
        self.indices.clone()
    }

    /// Indices of the modes of `eigvecs` matching the references; updates the references when
    /// adaptive. `gradient` enters the score when the gradient weight is positive.
    pub fn matched_indices(
        &mut self,
        eigvecs: &Mat<f64>,
        gradient: Option<&Col<f64>>,
    ) -> MoptResult<Vec<usize>> {
        if !self.is_initialized() {
            return Err(MoptError::config("mode following has no reference modes"));
        }
        let (n_ref, m) = (self.references.len(), eigvecs.ncols());
        if n_ref > m {
            return Err(MoptError::linalg(format!(
                "{n_ref} modes to follow, but only {m} modes are left"
            )));
        }
        let modes: Vec<Col<f64>> = (0..m).map(|j| Self::column(eigvecs, j)).collect();
        let w = self.settings.gradient_weight;
        let bias: Vec<f64> = match gradient {
            Some(g) if w > 0.0 => modes.iter().map(|v| w * self.overlap(g, v).abs()).collect(),
            _ => vec![0.0; m],
        };
        let signed = Mat::from_fn(n_ref, m, |i, j| {
            self.overlap(&self.references[i], &modes[j])
        });
        let score = Mat::from_fn(n_ref, m, |i, j| signed[(i, j)].abs() + bias[j]);
        let matched: Vec<usize> = if self.settings.hungarian {
            linear_sum_assignment(&(-&score))?
        } else {
            let mut used = vec![false; m];
            let mut out = Vec::with_capacity(n_ref);
            for i in 0..n_ref {
                let best = (0..m)
                    .filter(|&j| !used[j])
                    .max_by(|&a, &b| score[(i, a)].total_cmp(&score[(i, b)]));
                let j = best.ok_or_else(|| MoptError::linalg("no mode left to match"))?;
                used[j] = true;
                out.push(j);
            }
            out
        };
        if self.settings.adaptive {
            let a = self.settings.update_rate.clamp(0.0, 1.0);
            for (i, &j) in matched.iter().enumerate() {
                let sign = if signed[(i, j)] >= 0.0 { 1.0 } else { -1.0 };
                let mixed = &self.references[i] * (1.0 - a) + &modes[j] * (a * sign);
                let norm = libm::sqrt(self.inner(&mixed, &mixed));
                if norm >= 1e-12 {
                    self.references[i] = mixed / norm;
                }
            }
        }
        self.indices = matched.clone();
        Ok(matched)
    }

    /// Take `directions` as the references (normalized in the metric of the overlaps), so that
    /// the first match is the mode closest to them instead of the lowest modes.
    pub fn seed(&mut self, directions: &[Col<f64>]) {
        self.references = directions
            .iter()
            .take(self.count.max(1))
            .filter_map(|d| {
                let norm = libm::sqrt(self.inner(d, d));
                (norm >= 1e-12).then(|| d / norm)
            })
            .collect();
        self.indices.clear();
    }

    /// Forget the references.
    pub fn reset(&mut self) {
        self.references.clear();
        self.indices.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rotation(theta: f64) -> Mat<f64> {
        // Columns: three orthonormal modes; modes 0 and 1 rotated by theta in their plane.
        let (c, s) = (theta.cos(), theta.sin());
        Mat::from_fn(3, 3, |r, k| {
            [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]][r][k]
        })
    }

    #[test]
    fn a_mode_is_followed_through_a_reordering_of_the_spectrum() {
        let mut mf = ModeFollowing::new(1, ModeFollowingSettings::default());
        assert_eq!(mf.set_references(&rotation(0.0)), vec![0]);
        // The followed mode rotates slightly and moves to position 2 of the spectrum.
        let r = rotation(0.2);
        let swapped = Mat::from_fn(3, 3, |row, k| r[(row, [2, 1, 0][k])]);
        assert_eq!(mf.matched_indices(&swapped, None).unwrap(), vec![2]);
        // Adaptive: the reference is now the rotated mode (sign-aligned).
        let again = Mat::from_fn(3, 3, |row, k| -r[(row, k)]);
        assert_eq!(mf.matched_indices(&again, None).unwrap(), vec![0]);
        // An out-of-range target falls back to the lowest mode.
        let mut far = ModeFollowing::new(
            1,
            ModeFollowingSettings {
                target_index: 7,
                ..ModeFollowingSettings::default()
            },
        );
        assert_eq!(far.set_references(&rotation(0.0)), vec![0]);
    }

    #[test]
    fn global_assignment_avoids_the_greedy_conflict_and_gradient_bias_decides_ties() {
        // Reference 0 overlaps equally with modes 0 and 1; reference 1 only with mode 0.
        let h = std::f64::consts::FRAC_1_SQRT_2;
        let refs = Mat::from_fn(3, 3, |r, k| {
            [[h, 1.0, 0.0], [h, 0.0, 0.0], [0.0, 0.0, 1.0]][r][k]
        });
        let modes = Mat::<f64>::identity(3, 3);
        let mut settings = ModeFollowingSettings {
            adaptive: false,
            ..ModeFollowingSettings::default()
        };
        let mut mf = ModeFollowing::new(2, settings.clone());
        mf.set_references(&refs);
        assert_eq!(mf.matched_indices(&modes, None).unwrap(), vec![1, 0]);
        settings.hungarian = false;
        let mut greedy = ModeFollowing::new(2, settings.clone());
        greedy.set_references(&refs);
        // Greedy gives reference 0 the first of the tied modes and reference 1 a poor match.
        let g = greedy.matched_indices(&modes, None).unwrap();
        assert_eq!(g.len(), 2);
        assert_ne!(g[0], g[1]);
        // One reference with a tie: the gradient along mode 1 decides.
        settings.gradient_weight = 0.5;
        let mut biased = ModeFollowing::new(1, settings);
        biased.set_references(&refs);
        let grad = Col::from_fn(3, |i| [0.0, 2.0, 0.0][i]);
        assert_eq!(
            biased.matched_indices(&modes, Some(&grad)).unwrap(),
            vec![1]
        );
    }

    #[test]
    fn mass_weighted_overlap_and_method_strings() {
        let mut mf = ModeFollowing::new(
            1,
            ModeFollowingSettings {
                mass_weighted: true,
                ..ModeFollowingSettings::default()
            },
        );
        mf.set_masses(Col::from_fn(2, |i| [1.0, 100.0][i]));
        let a = Col::from_fn(2, |i| [1.0, 0.1][i]);
        let b = Col::from_fn(2, |i| [0.0, 1.0][i]);
        // <a, b>_M = 10, |a|_M = sqrt(2), |b|_M = 10.
        assert!((mf.overlap(&a, &b) - 1.0 / 2.0_f64.sqrt()).abs() < 1e-12);
        let s =
            ModeFollowingSettings::from_method("mwmf_rsirfo_block_fsb:1:ema0.5:grad0.3").unwrap();
        assert!(s.mass_weighted && s.adaptive && s.target_index == 1);
        assert!((s.update_rate - 0.5).abs() < 1e-15 && (s.gradient_weight - 0.3).abs() < 1e-15);
        let s = ModeFollowingSettings::from_method("smf_rsirfo_bofill").unwrap();
        assert!(!s.adaptive && !s.mass_weighted && s.update_rate == 0.0);
        let s = ModeFollowingSettings::from_method("mwsmf_rsirfo_bofill:ema0").unwrap();
        assert!(!s.adaptive && s.mass_weighted);
        assert!(ModeFollowingSettings::from_method("mf_rsirfo_bofill:emax").is_err());
        assert!(ModeFollowingSettings::from_method("mf_rsirfo_bofill:fast").is_err());
    }
}
