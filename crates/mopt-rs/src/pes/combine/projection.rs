//! Gradient projection for minimum-energy crossing points (MESX, MECI).
//!
//! Gradient projection of Bearpark, Robb, and Schlegel (Chem. Phys. Lett. 1994, 223, 269), as in
//! MultiOptPy's `opt_mesx`, `opt_mesx_2`, and `opt_meci`. With the difference gradient direction
//! `x = (g_1 - g_2) / |g_1 - g_2|` and the projector `P` that removes the branching plane, the
//! pseudo-gradient is
//!
//! `G = 2 (E_1 - E_2) x + P (g_1 + g_2) / 2`,
//!
//! and the reported energy is `(E_1 + E_2) / 2`. For a crossing of states of different spin or
//! symmetry (MESX) the branching plane is spanned by `x` alone. For a conical intersection (MECI)
//! it also contains the coupling direction `y`, which is approximated by the branching plane
//! updating (BPU) of Maeda, Ohno, and Morokuma (J. Chem. Theory Comput. 2010, 6, 1538):
//!
//! `y_k ∝ (y_{k-1} . x_k) x_{k-1} - (x_{k-1} . x_k) y_{k-1}`,
//!
//! started from the mean gradient orthogonalized against `x_0`.
//!
//! The Hessian is `P W P + 2 |g_1 - g_2| x x^T`. The second term is the curvature of the gap
//! term (`2 (E_1 - E_2) x` changes by `2 |g_1 - g_2|` per Bohr along `x`); MultiOptPy uses
//! `2 x x^T`, which is consistent only for `|g_1 - g_2| = 1` Hartree/Bohr (see
//! `docs/source/migration_surfaces.md`). `W = (1/2 - l) H_1 + (1/2 + l) H_2` is the Hessian of the
//! Lagrangian of the mean energy on the seam, with the multiplier `l = (mean gradient . x) /
//! |g_1 - g_2|` (as in `vib::seam`): the energy within the seam to second order, the seam
//! bending included. MultiOptPy takes the mean `H_mean = (H_1 + H_2) / 2` (`l = 0`), which
//! is that only at a crossing of equal and opposite slopes; where one state is far from its
//! minimum (`l` near `-1/2` at the equilibrium of the other, the crossing of a solvent shift
//! of `alpha_SA = 1` in `etmain`), `H_mean` takes the downhill curvature of that state for the
//! seam's and the steps leave the minimum on the seam. A model Hessian given to both states is
//! `W` itself.
//!
//! MultiOptPy: `ModelFunction/opt_mesx.py`, `opt_mesx_2.py`, `opt_meci.py`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::{Col, Mat};

use super::{Combined, StateCombiner, StateValues};
use crate::domain::MolecularSystem;
use crate::error::{MoptError, MoptResult};
use crate::math::linalg::{col_dot, col_norm};

/// Norm below which a direction is treated as undefined.
const TINY: f64 = 1e-8;

/// The smallest norm of the difference of the two gradients `|g_1 - g_2|` (Hartree/Bohr) the
/// projection works with: 1e-8, the norm below which it treats a direction as undefined.
///
/// The limit `|g_1 - g_2| -> 0` (two states that run parallel) has no crossing search: the
/// term `2 (E_1 - E_2) x` that closes the gap has no direction, the MECI projection would take
/// the whole mean gradient out with its second direction (built from the mean gradient), so
/// that any start looks stationary, and the MESX projection would go down the mean energy
/// with the gap open. The projection is refused there with an error that says so. Above the
/// limit the multiplier of the seam condition `l = (mean gradient . x) / |g_1 - g_2|` is
/// finite; it grows as `1 / |g_1 - g_2|`, and so does the Hessian `W` built with it, whose
/// steps the trust radius holds.
pub const PARALLEL_LIMIT: f64 = TINY;

/// Branching plane used by the projection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProjectionKind {
    /// Minimum-energy seam of crossing: project out the difference gradient only.
    Mesx,
    /// Minimum-energy conical intersection: project out the difference gradient and the
    /// approximate coupling direction (branching plane updating).
    Meci,
}

/// Gradient projection combiner for two states.
#[derive(Debug, Clone)]
pub struct GradientProjection {
    kind: ProjectionKind,
    /// `(x_{k-1}, y_{k-1})` of the last accepted iteration (MECI only).
    previous: Option<(Col<f64>, Col<f64>)>,
}

/// Branching plane at one geometry.
struct Plane {
    x: Col<f64>,
    gap_curvature: f64,
    y: Option<Col<f64>>,
}

/// `v - (x . v) x`, normalized; `None` if it vanishes.
fn orthonormal_to(v: &Col<f64>, x: &Col<f64>) -> Option<Col<f64>> {
    let w = v - x * col_dot(x, v);
    let n = col_norm(&w);
    (n >= TINY).then(|| w * (1.0 / n))
}

/// Deterministic unit vector orthogonal to `x`: the Cartesian axis with the smallest overlap.
/// (MultiOptPy draws a random vector here.)
fn any_orthonormal(x: &Col<f64>) -> Col<f64> {
    let dim = x.nrows();
    let mut order: Vec<usize> = (0..dim).collect();
    order.sort_by(|&a, &b| x[a].abs().total_cmp(&x[b].abs()));
    for k in order {
        let e = Col::from_fn(dim, |i| if i == k { 1.0 } else { 0.0 });
        if let Some(y) = orthonormal_to(&e, x) {
            return y;
        }
    }
    Col::zeros(dim)
}

/// `P v = v - x (x . v) - y (y . v)`.
fn project(v: &Col<f64>, plane: &Plane) -> Col<f64> {
    let mut out = v - &plane.x * col_dot(&plane.x, v);
    if let Some(y) = &plane.y {
        out -= y * col_dot(y, v);
    }
    out
}

impl GradientProjection {
    /// Combiner for a crossing seam (MultiOptPy `opt_mesx`, `opt_mesx_2`).
    pub fn mesx() -> Self {
        Self {
            kind: ProjectionKind::Mesx,
            previous: None,
        }
    }

    /// Combiner for a conical intersection (MultiOptPy `opt_meci`).
    pub fn meci() -> Self {
        Self {
            kind: ProjectionKind::Meci,
            previous: None,
        }
    }

    /// Branching plane kind.
    pub fn kind(&self) -> ProjectionKind {
        self.kind
    }

    fn plane(&self, s: &StateValues) -> MoptResult<Plane> {
        if s.energies.len() != 2 || s.gradients.len() != 2 {
            return Err(MoptError::config("gradient projection needs two states"));
        }
        let difference = &s.gradients[0] - &s.gradients[1];
        let gap_curvature = col_norm(&difference);
        if gap_curvature.is_nan() || gap_curvature < PARALLEL_LIMIT {
            return Err(MoptError::conv(format!(
                "gradient projection ({}): the difference of the gradients of the two states is \
                 {gap_curvature:.1e} Hartree/Bohr, below the limit of 1e-8: the states run \
                 parallel here, and no direction closes their gap ({:.3e} Hartree); there is \
                 no crossing search from this structure (start from another one, or search on \
                 a model function of the two states: -mf seam, -mf avoiding)",
                self.name(),
                s.energies[0] - s.energies[1]
            )));
        }
        let x = difference * (1.0 / gap_curvature);
        let y = match self.kind {
            ProjectionKind::Mesx => None,
            ProjectionKind::Meci => Some(self.coupling_direction(&x, s)),
        };
        Ok(Plane {
            x,
            gap_curvature: 2.0 * gap_curvature,
            y,
        })
    }

    fn coupling_direction(&self, x: &Col<f64>, s: &StateValues) -> Col<f64> {
        let updated = self.previous.as_ref().and_then(|(xp, yp)| {
            let num = xp * col_dot(yp, x) - yp * col_dot(xp, x);
            let n = col_norm(&num);
            if n >= TINY {
                Some(num * (1.0 / n))
            } else {
                // x barely changed: keep y orthogonalized to the new x.
                orthonormal_to(yp, x)
            }
        });
        updated.unwrap_or_else(|| {
            let mean = (&s.gradients[0] + &s.gradients[1]) * 0.5;
            orthonormal_to(&mean, x).unwrap_or_else(|| any_orthonormal(x))
        })
    }
}

impl StateCombiner for GradientProjection {
    fn name(&self) -> &str {
        match self.kind {
            ProjectionKind::Mesx => "opt_mesx",
            ProjectionKind::Meci => "opt_meci",
        }
    }

    fn n_states(&self) -> usize {
        2
    }

    fn combine(&self, _sys: &MolecularSystem, s: &StateValues) -> MoptResult<Combined> {
        let plane = self.plane(s)?;
        let (e1, e2) = (s.energies[0], s.energies[1]);
        let mean_grad = (&s.gradients[0] + &s.gradients[1]) * 0.5;
        let gradient = &plane.x * (2.0 * (e1 - e2)) + project(&mean_grad, &plane);
        // The multiplier of the seam condition: grad E_m = l (g_1 - g_2) within the plane;
        // `|g_1 - g_2|` is at least `PARALLEL_LIMIT` here (see `plane`).
        let difference = 0.5 * plane.gap_curvature;
        let multiplier = col_dot(&plane.x, &mean_grad) / difference;
        let hessian = s.hessians.as_ref().map(|hs| {
            let mean_h = &hs[0] * (0.5 - multiplier) + &hs[1] * (0.5 + multiplier);
            let dim = mean_h.nrows();
            // P H P, applied column by column on both sides.
            let mut ph = Mat::zeros(dim, dim);
            for j in 0..dim {
                let col = project(&mean_h.col(j).to_owned(), &plane);
                ph.col_mut(j).copy_from(&col);
            }
            let mut php = Mat::zeros(dim, dim);
            for i in 0..dim {
                let row = project(&ph.row(i).transpose().to_owned(), &plane);
                php.row_mut(i).copy_from(row.transpose());
            }
            for i in 0..dim {
                for j in 0..dim {
                    php[(i, j)] += plane.gap_curvature * plane.x[i] * plane.x[j];
                }
            }
            php
        });
        Ok(Combined {
            energy: 0.5 * (e1 + e2),
            gradient,
            hessian,
        })
    }

    fn branching_plane(&self, s: &StateValues) -> Option<Vec<Col<f64>>> {
        let plane = self.plane(s).ok()?;
        let mut directions = vec![plane.x];
        directions.extend(plane.y);
        Some(directions)
    }

    fn set_branching_plane(&mut self, directions: &[Col<f64>]) {
        if let (ProjectionKind::Meci, [x, y]) = (self.kind, directions) {
            self.previous = Some((x.clone(), y.clone()));
        }
    }

    fn commit(&mut self, _sys: &MolecularSystem, s: &StateValues) {
        if self.kind != ProjectionKind::Meci {
            return;
        }
        if let Ok(plane) = self.plane(s) {
            if let Some(y) = plane.y {
                self.previous = Some((plane.x, y));
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn states(e: [f64; 2], g1: [f64; 3], g2: [f64; 3]) -> StateValues {
        StateValues {
            energies: e.to_vec(),
            gradients: vec![Col::from_fn(3, |i| g1[i]), Col::from_fn(3, |i| g2[i])],
            hessians: Some(vec![
                Mat::<f64>::identity(3, 3),
                Mat::<f64>::identity(3, 3) * 3.0,
            ]),
        }
    }

    fn dummy() -> MolecularSystem {
        MolecularSystem::from_flat_bohr(
            vec![crate::domain::Element::H],
            &Col::zeros(3),
            crate::domain::SpinCharge::default(),
            None,
        )
        .unwrap()
    }

    #[test]
    fn mesx_pseudo_gradient() {
        let s = states([-1.0, -1.1], [0.3, 0.1, 0.0], [-0.1, 0.1, 0.0]);
        let c = GradientProjection::mesx().combine(&dummy(), &s).unwrap();
        // x = (1, 0, 0), |dg| = 0.4; mean gradient (0.1, 0.1, 0) projected -> (0, 0.1, 0).
        assert!((c.gradient[0] - 0.2).abs() < 1e-14);
        assert!((c.gradient[1] - 0.1).abs() < 1e-14);
        assert!((c.energy + 1.05).abs() < 1e-14);
        // The Hessian of the Lagrangian within the seam: l = (mean gradient . x) / |dg| = 0.25,
        // W = (1/2 - l) H_1 + (1/2 + l) H_2 = 0.25 I + 0.75 (3 I) = 2.5 I.
        let h = c.hessian.unwrap();
        assert!((h[(0, 0)] - 0.8).abs() < 1e-14);
        assert!((h[(1, 1)] - 2.5).abs() < 1e-14);
        assert!(h[(0, 1)].abs() < 1e-14);
        // Equal Hessians (a model Hessian given to both states) stay as they are.
        let mut same = s.clone();
        same.hessians = Some(vec![Mat::<f64>::identity(3, 3) * 2.0; 2]);
        let h = GradientProjection::mesx()
            .combine(&dummy(), &same)
            .unwrap()
            .hessian
            .unwrap();
        assert!((h[(1, 1)] - 2.0).abs() < 1e-14 && (h[(2, 2)] - 2.0).abs() < 1e-14);
    }

    #[test]
    fn equal_gradients_of_the_two_states_are_refused() {
        // No direction closes the gap: the projection has no pseudo-gradient, and a start
        // there must not look stationary (MECI: everything projected out) or go down the
        // mean energy with the gap open (MESX).
        for gp in [GradientProjection::meci(), GradientProjection::mesx()] {
            let s = states([-1.0, -1.02], [0.3, 0.2, 0.05], [0.3, 0.2, 0.05]);
            let e = gp.combine(&dummy(), &s).err();
            assert!(
                e.as_ref().is_some_and(|e| e.to_string().contains("1e-8")),
                "{}: {e:?}",
                gp.name()
            );
            // Below the limit as well.
            let s = states([-1.0, -1.02], [0.3, 0.2, 0.05], [0.3 + 5e-9, 0.2, 0.05]);
            assert!(gp.combine(&dummy(), &s).is_err(), "{}", gp.name());
        }
        // Just above it the multiplier and the Hessian stay finite.
        let s = states([-1.0, -1.0], [0.3, 0.2, 0.05], [0.3 + 2e-8, 0.2, 0.05]);
        let c = GradientProjection::mesx().combine(&dummy(), &s).unwrap();
        let h = c.hessian.unwrap();
        assert!((0..3).all(|i| (0..3).all(|j| h[(i, j)].is_finite())));
    }

    #[test]
    fn meci_projects_both_branching_directions() {
        let mut gp = GradientProjection::meci();
        let s = states([-1.0, -1.0], [0.3, 0.2, 0.05], [-0.1, 0.2, 0.05]);
        let c = gp.combine(&dummy(), &s).unwrap();
        // Initial y is the mean gradient orthogonalized to x: (0, 0.2, 0.05)/|...|; only the
        // remaining direction keeps a projected mean gradient, which is zero here.
        assert!(c.gradient.iter().all(|v| v.abs() < 1e-14));
        gp.commit(&dummy(), &s);
        // Rotate the difference gradient within the x-z plane: BPU keeps y orthogonal to x.
        let s2 = states([-1.0, -1.01], [0.3, 0.2, 0.15], [-0.1, 0.2, 0.05]);
        let plane = gp.plane(&s2).unwrap();
        let y = plane.y.unwrap();
        assert!(col_dot(&y, &plane.x).abs() < 1e-12);
        assert!((col_norm(&y) - 1.0).abs() < 1e-12);
    }
}
