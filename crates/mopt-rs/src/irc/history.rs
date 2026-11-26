//! What the integrators remember of a side: the last points (oscillation of the energy,
//! bending of the path, its length) and the Hessian of the surface along it.
//!
//! The Hessian is kept for the surface without bias potentials, in mass-weighted
//! coordinates: an exact one replaces it when the calculator was asked for it, otherwise it
//! is updated with the change of the gradient between two evaluated points, and the rigid-body
//! motions are taken out of the update as out of an exact Hessian (`-update_proj`). The Hessian of
//! the bias potentials is exact at every point and added on top. MultiOptPy does the same,
//! with three slips that are not repeated here (`docs/source/migration_paths.md`): its starting
//! Hessian already holds the bias potentials (counted twice), `iter % FC_count` with the
//! default `-1` replaces the Hessian by the one of the transition state at every step, and
//! an exact Hessian is still updated with the step that led to its point.
//!
//! MultiOptPy: `IRC/lqa.py:LQA` (`check_energy_oscillation`, the lists `irc_*`, the Hessian
//! handling of `step` and `run`) and `Utils/calc_tools.py:calc_multi_dim_vec_angle`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::{Col, Mat};

use super::constraints::IrcConstraints;
use super::space::IrcSpace;
use super::types::{dot, norm, PointValue};
use crate::error::MoptResult;
use crate::opt::update::formulas::{bfgs, bofill, fsb};

/// MultiOptPy's smallest `s.y` at which its BFGS update is applied.
pub const BFGS_CURVATURE_THRESHOLD: f64 = 1e-10;

/// The update of the Hessian between two evaluated points.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HessianUpdate {
    /// FSB (MultiOptPy's LQA).
    Fsb,
    /// BFGS when `s.y > 1e-10` (MultiOptPy's HPC, RK4, and ModeKill).
    GuardedBfgs,
    /// Bofill, for regions of negative curvature.
    Bofill,
}

/// The last three points of a side.
#[derive(Debug, Clone)]
pub struct PathHistory {
    energies: Vec<f64>,
    coords: Vec<Col<f64>>,
    reversals: usize,
    arc: f64,
}

/// Angle (degrees) between `a` and `b`; `None` when one of them vanishes.
pub fn angle_deg(a: &Col<f64>, b: &Col<f64>) -> Option<f64> {
    let (na, nb) = (norm(a), norm(b));
    if na == 0.0 || nb == 0.0 {
        return None;
    }
    // MultiOptPy adds 1e-8 to the cosine and gets NaN for parallel vectors; the cosine is
    // limited to [-1, 1] instead.
    let cos = (dot(a, b) / (na * nb)).clamp(-1.0, 1.0);
    Some(libm::acos(cos).to_degrees())
}

impl PathHistory {
    /// No points yet.
    pub fn new() -> Self {
        Self {
            energies: Vec::new(),
            coords: Vec::new(),
            reversals: 0,
            arc: 0.0,
        }
    }

    /// Add the point `q` (mass-weighted) with the energy of the surface that is followed.
    /// Returns the bending angle at the point before it (degrees; from the third point on).
    pub fn push(&mut self, q: Col<f64>, energy: f64) -> Option<f64> {
        if let Some(last) = self.coords.last() {
            self.arc += norm(&(&q - last));
        }
        if self.energies.len() >= 3 {
            self.energies.remove(0);
            self.coords.remove(0);
        }
        self.energies.push(energy);
        self.coords.push(q);
        // MultiOptPy: the energy change reversed its sign between the last two steps.
        let reversed = match self.energies.as_slice() {
            [a, b, c] => (c - b) * (b - a) < 0.0,
            _ => false,
        };
        self.reversals = if reversed { self.reversals + 1 } else { 0 };
        match self.coords.as_slice() {
            [a, b, c] => angle_deg(&(a - b), &(c - b)),
            _ => None,
        }
    }

    /// The curvature vector of the path at the last points: the change of the unit
    /// tangent between the last two steps over the length between their middles (`None`
    /// before the third point).
    pub fn curvature(&self) -> Option<Col<f64>> {
        let [a, b, c] = self.coords.as_slice() else {
            return None;
        };
        let (first, second) = (b - a, c - b);
        let (l1, l2) = (norm(&first), norm(&second));
        if l1 == 0.0 || l2 == 0.0 {
            return None;
        }
        Some((second * (1.0 / l2) - first * (1.0 / l1)) * (2.0 / (l1 + l2)))
    }

    /// Reversals of the energy change in a row up to the last point.
    pub fn reversals(&self) -> usize {
        self.reversals
    }

    /// Whether the energy of the last point is above that of the point before it.
    pub fn rose(&self) -> bool {
        match self.energies.as_slice() {
            [.., a, b] => b > a,
            _ => false,
        }
    }

    /// Mass-weighted length of the path through the points so far.
    pub fn arc_length(&self) -> f64 {
        self.arc
    }

    /// The last point.
    pub fn last(&self) -> Option<&Col<f64>> {
        self.coords.last()
    }

    /// The unit direction in which the path came to its last point: from the last point
    /// before it at another place (steps of length zero are passed over).
    pub fn direction(&self) -> Option<Col<f64>> {
        let last = self.coords.last()?;
        self.coords.iter().rev().skip(1).find_map(|q| {
            let d = last - q;
            let length = norm(&d);
            (length > 1e-12).then(|| d * (1.0 / length))
        })
    }
}

impl Default for PathHistory {
    fn default() -> Self {
        Self::new()
    }
}

/// The Hessian of the surface along a side (mass-weighted, without bias potentials).
#[derive(Debug, Clone)]
pub struct SurfaceModel {
    hessian: Mat<f64>,
    update: HessianUpdate,
    last: Option<(Col<f64>, Col<f64>)>,
}

impl SurfaceModel {
    /// Start from `hessian` (mass-weighted, projected, without bias potentials).
    pub fn new(hessian: Mat<f64>, update: HessianUpdate) -> Self {
        Self {
            hessian,
            update,
            last: None,
        }
    }

    /// Take in the evaluated point `x`: its exact Hessian when it has one, otherwise an
    /// update with the change from the point observed before. Returns whether the Hessian
    /// is exact at this point.
    pub fn observe(
        &mut self,
        space: &IrcSpace,
        x: &Col<f64>,
        value: &PointValue,
    ) -> MoptResult<bool> {
        let q = space.to_mw(x);
        let g = space.gradient_to_mw(&value.raw_gradient);
        let exact = match &value.hessian {
            Some(h) => {
                self.hessian = space.prepared_hessian(h, x)?;
                true
            }
            None => {
                if let Some((q0, g0)) = &self.last {
                    let s = &q - q0;
                    let y = &g - g0;
                    match space.constraints() {
                        // On the surface of constraints: the update with the parts of the
                        // step and of the change of the gradient on it, and the Hessian put
                        // back on it.
                        Some(c) => {
                            let normals = c.normals(x)?;
                            self.apply(
                                &IrcConstraints::along_surface(&s, &normals),
                                &IrcConstraints::along_surface(&y, &normals),
                            );
                            self.hessian =
                                IrcConstraints::on_surface(&self.hessian, &normals, true);
                        }
                        None => self.apply(&s, &y),
                    }
                    // The pair joins two structures and puts curvature along the rigid
                    // motions of this one: they are taken out as out of an exact Hessian (also
                    // where the formula left the Hessian as it was: it belongs to the point
                    // before).
                    self.hessian = space.project_updated_hessian(&self.hessian, x)?;
                }
                false
            }
        };
        self.last = Some((q, g));
        Ok(exact)
    }

    fn apply(&mut self, s: &Col<f64>, y: &Col<f64>) {
        let change = match self.update {
            HessianUpdate::Fsb => fsb(&self.hessian, s, y, 1.0),
            HessianUpdate::Bofill => bofill(&self.hessian, s, y, 1.0),
            HessianUpdate::GuardedBfgs => {
                if dot(s, y) <= BFGS_CURVATURE_THRESHOLD {
                    return;
                }
                bfgs(&self.hessian, s, y)
            }
        };
        let finite =
            (0..change.ncols()).all(|j| (0..change.nrows()).all(|i| change[(i, j)].is_finite()));
        if finite {
            self.hessian += change;
        }
    }

    /// The Hessian without bias potentials.
    pub fn hessian(&self) -> &Mat<f64> {
        &self.hessian
    }

    /// The Hessian of the surface that is followed at the point of `value`: with the
    /// Hessian of the bias potentials there.
    pub fn total(&self, space: &IrcSpace, value: &PointValue) -> Mat<f64> {
        match &value.bias_hessian {
            Some(b) => &self.hessian + space.hessian_to_mw(b),
            None => self.hessian.clone(),
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn col(v: &[f64]) -> Col<f64> {
        Col::from_fn(v.len(), |i| v[i])
    }

    #[test]
    fn oscillation_bending_and_length() {
        let mut h = PathHistory::new();
        assert_eq!(h.push(col(&[0.0, 0.0]), 0.0), None);
        assert_eq!(h.push(col(&[1.0, 0.0]), -1.0), None);
        // Straight on and still downhill.
        let angle = h.push(col(&[2.0, 0.0]), -2.0).unwrap();
        assert!((angle - 180.0).abs() < 1e-9);
        assert_eq!(h.reversals(), 0);
        // A right turn and uphill.
        let angle = h.push(col(&[2.0, 1.0]), -1.5).unwrap();
        assert!((angle - 90.0).abs() < 1e-9);
        assert_eq!(h.reversals(), 1);
        h.push(col(&[2.0, 2.0]), -1.7);
        assert_eq!(h.reversals(), 2);
        h.push(col(&[2.0, 3.0]), -1.9);
        assert_eq!(h.reversals(), 0);
        assert!((h.arc_length() - 5.0).abs() < 1e-12);
        // Parallel vectors: 0 degrees, not NaN.
        assert_eq!(angle_deg(&col(&[1.0, 0.0]), &col(&[2.0, 0.0])), Some(0.0));
    }

    #[test]
    fn exact_hessians_replace_and_secant_pairs_update() {
        let space = IrcSpace::variables(2);
        let h0 = Mat::from_fn(2, 2, |i, j| if i == j { 1.0 } else { 0.0 });
        let truth = Mat::from_fn(2, 2, |i, j| [[2.0, 0.3], [0.3, -1.0]][i][j]);
        let at = |x: &Col<f64>, hessian: bool| {
            let g = &truth * x;
            PointValue::plain(0.5 * dot(x, &g), g, hessian.then(|| truth.clone()))
        };
        for update in [HessianUpdate::Fsb, HessianUpdate::Bofill] {
            let mut model = SurfaceModel::new(h0.clone(), update);
            let a = col(&[0.1, 0.2]);
            let b = col(&[0.3, -0.1]);
            assert!(!model.observe(&space, &a, &at(&a, false)).unwrap());
            assert!(!model.observe(&space, &b, &at(&b, false)).unwrap());
            // The secant condition holds after the update.
            let s = &b - &a;
            let y = &truth * &s;
            let r = model.hessian() * &s - &y;
            assert!(norm(&r) < 1e-12, "{update:?}");
            // An exact Hessian is taken as it is.
            assert!(model.observe(&space, &a, &at(&a, true)).unwrap());
            assert_eq!(model.hessian(), &truth);
        }
        // The Hessian of the bias potentials is added once, at the point it belongs to.
        let model = SurfaceModel::new(truth.clone(), HessianUpdate::Fsb);
        let mut biased = at(&col(&[0.1, 0.2]), false);
        biased.bias_hessian = Some(Mat::from_fn(2, 2, |i, j| if i == j { 0.5 } else { 0.0 }));
        let total = model.total(&space, &biased);
        assert!((total[(0, 0)] - 2.5).abs() < 1e-15 && (total[(1, 1)] + 0.5).abs() < 1e-15);
        assert_eq!(model.hessian(), &truth);
        // BFGS leaves the Hessian alone along negative curvature.
        let mut model = SurfaceModel::new(h0.clone(), HessianUpdate::GuardedBfgs);
        let a = col(&[0.0, 0.1]);
        let b = col(&[0.0, 0.3]);
        model.observe(&space, &a, &at(&a, false)).unwrap();
        model.observe(&space, &b, &at(&b, false)).unwrap();
        assert_eq!(model.hessian(), &h0);
    }
}
