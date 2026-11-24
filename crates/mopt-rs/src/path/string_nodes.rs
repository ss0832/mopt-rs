//! What the string methods share (the growing string, the freezing string, the relaxation
//! of a path after DS-AFIR): the geometry of the points (rigid-body motions and fixed
//! variables taken out of their directions), the tangent of a string at a node, a string
//! spread evenly along its length, and the quasi-Newton steps of one node across its
//! tangent.
//!
//! The steps of a node are L-BFGS steps (Nocedal, Math. Comput. 1980, 35, 773) on the
//! gradient across the tangent, the step then taken across the tangent as well and limited
//! in length and in its largest component. The pairs of steps and gradient differences stay
//! with the node while it moves; a pair without positive curvature is not kept.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::collections::VecDeque;
use std::sync::Arc;

use faer::{Col, Mat};

use super::ieip::aligned_pair;
use super::manifold::PointManifold;
use crate::domain::MolecularSystem;
use crate::error::MoptResult;
use crate::math::linalg::{col_dot, col_max_abs, col_norm};
use crate::math::projection::{build_tr_rot_basis, project_out_basis_vec};

/// What the points of a string are: structures without a preferred orientation (rigid-body
/// motions are left out of every direction), fixed variables, or neither.
#[derive(Debug, Clone, Default)]
pub struct PointGeometry {
    /// The structure whose atoms the points place, when the surface has no preferred
    /// orientation and position.
    pub rigid_body: Option<MolecularSystem>,
    /// Variables that keep their values.
    pub frozen: Vec<usize>,
    /// The cell of periodic structures: a point is laid onto another by the nearest image of
    /// every atom, so that a path between two structures does not cross the cell.
    pub cell: Option<crate::domain::PeriodicCell>,
    /// Hard constraints beyond fixed variables (`-pc`, `-gfix`, `-uc`): directions are taken
    /// into their tangent space ([`Self::project`]) and moved points back onto them
    /// ([`Self::settle`]).
    pub manifold: Option<Arc<PointManifold>>,
}

impl PointGeometry {
    /// The points held on `manifold` as well.
    pub fn with_manifold(mut self, manifold: Option<Arc<PointManifold>>) -> Self {
        self.manifold = manifold;
        self
    }

    /// `x` put back onto the hard constraints (itself without any).
    pub fn settle(&self, x: &Col<f64>) -> MoptResult<Col<f64>> {
        match &self.manifold {
            Some(m) => m.restore(x),
            None => Ok(x.clone()),
        }
    }

    /// Structures without a preferred orientation, if `rigid`.
    pub fn of_structure(
        template: Option<&MolecularSystem>,
        rigid: bool,
        frozen: Vec<usize>,
    ) -> Self {
        Self {
            rigid_body: template
                .filter(|t| rigid && frozen.is_empty() && !t.has_pbc())
                .cloned(),
            frozen,
            cell: template.and_then(MolecularSystem::active_cell).cloned(),
            manifold: None,
        }
    }

    /// Whether points are turned onto each other.
    pub fn aligns(&self) -> bool {
        self.rigid_body.is_some()
    }

    /// `v` at the point `x` without rigid-body motions and fixed variables, and in the
    /// tangent space of the hard constraints. The constraint gradients were evaluated at the
    /// start of the run ([`PointManifold::new`]); should their evaluation fail at `x`, `v`
    /// keeps its part along them, which the next [`Self::settle`] takes out of the point.
    pub fn project(&self, x: &Col<f64>, v: &Col<f64>) -> Col<f64> {
        let out = self.project_plain(x, v);
        match &self.manifold {
            Some(m) => m.project(x, &out).unwrap_or(out),
            None => out,
        }
    }

    /// `v` at the point `x` without rigid-body motions and fixed variables.
    fn project_plain(&self, x: &Col<f64>, v: &Col<f64>) -> Col<f64> {
        let mut out = v.clone();
        if let Some(template) = &self.rigid_body {
            if let Ok(sys) = template.with_flat_coords_bohr(x) {
                out = project_out_basis_vec(&build_tr_rot_basis(&sys, false), &out);
            }
        }
        for &k in &self.frozen {
            if k < out.nrows() {
                out[k] = 0.0;
            }
        }
        out
    }

    /// `target` laid onto `reference` (both centred on the centroid of `reference`), for
    /// structures without a preferred orientation; `target` itself otherwise.
    pub fn laid_onto(&self, target: &Col<f64>, reference: &Col<f64>) -> Col<f64> {
        // Periodic structures: every atom at its image nearest to that of the reference.
        if let Some(cell) = &self.cell {
            let mut out = target.clone();
            for a in 0..target.nrows() / 3 {
                let r = [reference[3 * a], reference[3 * a + 1], reference[3 * a + 2]];
                let t = [target[3 * a], target[3 * a + 1], target[3 * a + 2]];
                let d = cell.mic_delta(t, r);
                for k in 0..3 {
                    out[3 * a + k] = r[k] + d[k];
                }
            }
            return out;
        }
        if !self.aligns() {
            return target.clone();
        }
        let (turned, centred) = aligned_pair(target, reference);
        // Back to the centroid of the reference.
        &turned + (reference - &centred)
    }
}

/// Unit vector of `v` (`None` for a zero vector).
pub fn unit(v: &Col<f64>) -> Option<Col<f64>> {
    let n = col_norm(v);
    (n.is_finite() && n > 0.0).then(|| v * (1.0 / n))
}

/// The tangent at node `k` of a string with the energies `energies`: the upwind tangent of
/// Henkelman and Jonsson (J. Chem. Phys. 2000, 113, 9978), towards the neighbour higher in
/// energy, with the energy-weighted mix at extrema of the profile. The ends take the
/// direction to their neighbour.
pub fn tangent(points: &[Col<f64>], energies: &[f64], k: usize) -> Option<Col<f64>> {
    let n = points.len();
    if n < 2 || k >= n {
        return None;
    }
    if k == 0 {
        return unit(&(&points[1] - &points[0]));
    }
    if k == n - 1 {
        return unit(&(&points[n - 1] - &points[n - 2]));
    }
    let forward = &points[k + 1] - &points[k];
    let backward = &points[k] - &points[k - 1];
    let (e0, e, e1) = (energies[k - 1], energies[k], energies[k + 1]);
    let t = if e1 > e && e > e0 {
        forward
    } else if e1 < e && e < e0 {
        backward
    } else {
        let (big, small) = {
            let a = (e1 - e).abs();
            let b = (e0 - e).abs();
            (a.max(b), a.min(b))
        };
        if e1 > e0 {
            forward * big + backward * small
        } else {
            forward * small + backward * big
        }
    };
    unit(&t)
}

/// Lengths of the segments of a string.
pub fn segment_lengths(points: &[Col<f64>]) -> Vec<f64> {
    points
        .windows(2)
        .map(|w| col_norm(&(&w[1] - &w[0])))
        .collect()
}

/// The point at the length `s` along the polygon `points` (clamped to its ends), with the
/// direction of the segment it lies on.
pub fn point_at_length(points: &[Col<f64>], s: f64) -> Option<(Col<f64>, Col<f64>)> {
    let lengths = segment_lengths(points);
    let mut left = s.max(0.0);
    for (k, &l) in lengths.iter().enumerate() {
        if l <= 0.0 {
            continue;
        }
        if left <= l || k == lengths.len() - 1 {
            let share = (left / l).min(1.0);
            let d = &points[k + 1] - &points[k];
            return Some((&points[k] + &d * share, d * (1.0 / l)));
        }
        left -= l;
    }
    None
}

/// The nodes `first..=last` of `points` put at equal lengths along the polygon through them
/// (the two nodes stay where they are). Returns whether anything moved.
pub fn spread_evenly(points: &mut [Col<f64>], first: usize, last: usize) -> bool {
    if last <= first + 1 || last >= points.len() {
        return false;
    }
    let part: Vec<Col<f64>> = points[first..=last].to_vec();
    let total: f64 = segment_lengths(&part).iter().sum();
    if total.is_nan() || total <= 0.0 {
        return false;
    }
    let count = last - first;
    for j in 1..count {
        if let Some((x, _)) = point_at_length(&part, total * j as f64 / count as f64) {
            points[first + j] = x;
        }
    }
    true
}

/// Settings of the steps of one node.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NodeStepSettings {
    /// Pairs kept.
    pub memory: usize,
    /// Largest length of a step.
    pub max_step: f64,
    /// Largest component of a step.
    pub max_component: f64,
    /// Inverse curvature of the first step (length per unit of gradient).
    pub first_scale: f64,
}

impl Default for NodeStepSettings {
    fn default() -> Self {
        Self {
            memory: 10,
            max_step: 0.1,
            max_component: f64::INFINITY,
            first_scale: 1.0,
        }
    }
}

/// L-BFGS steps of one node.
#[derive(Debug, Clone)]
pub struct NodeSteps {
    settings: NodeStepSettings,
    pairs: VecDeque<(Col<f64>, Col<f64>)>,
    last: Option<(Col<f64>, Col<f64>)>,
}

impl NodeSteps {
    /// No history yet.
    pub fn new(settings: NodeStepSettings) -> Self {
        Self {
            settings,
            pairs: VecDeque::new(),
            last: None,
        }
    }

    /// Forget the history (the function the node minimizes has changed).
    pub fn reset(&mut self) {
        self.pairs.clear();
        self.last = None;
    }

    /// The step from `x` with the (projected) gradient `g`; `across` takes a direction at
    /// `x` to the space the node moves in.
    pub fn step(
        &mut self,
        x: &Col<f64>,
        g: &Col<f64>,
        across: &dyn Fn(&Col<f64>) -> Col<f64>,
    ) -> Col<f64> {
        if let Some((x0, g0)) = self.last.take() {
            let s = x - &x0;
            let y = g - &g0;
            let sy = col_dot(&s, &y);
            if sy > 1e-12 * col_norm(&s) * col_norm(&y) && sy > 0.0 {
                self.pairs.push_back((s, y));
                while self.pairs.len() > self.settings.memory.max(1) {
                    self.pairs.pop_front();
                }
            }
        }
        self.last = Some((x.clone(), g.clone()));
        let mut q = g.clone();
        let mut alphas = Vec::with_capacity(self.pairs.len());
        for (s, y) in self.pairs.iter().rev() {
            let rho = 1.0 / col_dot(y, s);
            let a = rho * col_dot(s, &q);
            q -= y * a;
            alphas.push((a, rho));
        }
        let gamma = match self.pairs.back() {
            Some((s, y)) => col_dot(s, y) / col_dot(y, y),
            None => self.settings.first_scale,
        };
        let mut r = q * gamma;
        for ((s, y), (a, rho)) in self.pairs.iter().zip(alphas.into_iter().rev()) {
            let b = rho * col_dot(y, &r);
            r += s * (a - b);
        }
        let mut p = across(&(r * -1.0));
        if col_dot(&p, g) > 0.0 {
            // Not downhill: the history no longer fits; a gradient step instead.
            self.pairs.clear();
            p = across(&(g * -self.settings.first_scale));
        }
        limited(p, self.settings.max_step, self.settings.max_component)
    }
}

/// `p` limited to the length `max_step` and the largest component `max_component`.
pub fn limited(p: Col<f64>, max_step: f64, max_component: f64) -> Col<f64> {
    let largest = col_max_abs(&p);
    let p = if largest > max_component {
        p * (max_component / largest)
    } else {
        p
    };
    let length = col_norm(&p);
    if length > max_step {
        p * (max_step / length)
    } else {
        p
    }
}

/// `g` without its component along `t`.
pub fn across(g: &Col<f64>, t: &Col<f64>) -> Col<f64> {
    g - t * col_dot(g, t)
}

/// RMS of the components of `v` over `free` variables.
pub fn rms(v: &Col<f64>, free: usize) -> f64 {
    libm::sqrt(col_dot(v, v) / free.max(1) as f64)
}

/// The matrix of the rows `points` (for reports).
pub fn as_rows(points: &[Col<f64>]) -> Mat<f64> {
    let n = points.first().map_or(0, Col::nrows);
    Mat::from_fn(points.len(), n, |i, j| points[i][j])
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn col(v: &[f64]) -> Col<f64> {
        Col::from_fn(v.len(), |k| v[k])
    }

    #[test]
    fn a_string_is_spread_evenly_along_its_polygon() {
        let mut points = vec![
            col(&[0.0, 0.0]),
            col(&[0.1, 0.0]),
            col(&[1.0, 0.0]),
            col(&[1.0, 1.0]),
        ];
        assert!(spread_evenly(&mut points, 0, 3));
        // Equal lengths along the polygon as it was (length 2): at 2/3 and 4/3.
        assert!((points[1][0] - 2.0 / 3.0).abs() < 1e-12 && points[1][1].abs() < 1e-12);
        assert!((points[2][0] - 1.0).abs() < 1e-12 && (points[2][1] - 1.0 / 3.0).abs() < 1e-12);
        let points = vec![col(&[0.0, 0.0]), col(&[1.0, 0.0]), col(&[1.0, 1.0])];
        let (x, d) = point_at_length(&points, 1.5).unwrap();
        assert!((x[0] - 1.0).abs() < 1e-12 && (x[1] - 0.5).abs() < 1e-12);
        assert!((d[1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn the_tangent_points_uphill() {
        let points = vec![col(&[0.0, 0.0]), col(&[1.0, 0.0]), col(&[1.0, 1.0])];
        let t = tangent(&points, &[0.0, 1.0, 2.0], 1).unwrap();
        assert!((t[1] - 1.0).abs() < 1e-12);
        let t = tangent(&points, &[2.0, 1.0, 0.0], 1).unwrap();
        assert!((t[0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn node_steps_minimize_a_quadratic_across_a_tangent() {
        // E = x^2 + 10 y^2 + z^2, moving across t = z.
        let t = col(&[0.0, 0.0, 1.0]);
        let gradient = |x: &Col<f64>| col(&[2.0 * x[0], 20.0 * x[1], 2.0 * x[2]]);
        let mut x = col(&[1.0, 0.5, 0.3]);
        let mut steps = NodeSteps::new(NodeStepSettings {
            first_scale: 0.04,
            max_step: 1.0,
            ..NodeStepSettings::default()
        });
        for _ in 0..30 {
            let g = across(&gradient(&x), &t);
            let p = steps.step(&x, &g, &|v| across(v, &t));
            x += &p;
        }
        assert!(x[0].abs() < 1e-6 && x[1].abs() < 1e-6, "{x:?}");
        assert!((x[2] - 0.3).abs() < 1e-12);
    }
}
