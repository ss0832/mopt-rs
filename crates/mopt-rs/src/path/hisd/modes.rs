//! The lowest curvatures and their directions at a fixed point, from gradients alone: the
//! locally optimal block preconditioned conjugate gradient method (A. V. Knyazev, SIAM J.
//! Sci. Comput. 2001, 23, 517) with products of the Hessian from differences of gradients
//! (the dimer of the high-index saddle dynamics), or the eigenvectors of an exact Hessian.
//!
//! It gives the unstable directions of a saddle point and its softest stable ones, the
//! starts of the searches of [`super::landscape`], when no Hessian is at hand. Each
//! iteration costs one gradient per direction; the products of the conjugate directions
//! follow from the others by linearity.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::{Col, Mat};

use super::frame::SearchFrame;
use super::settings::Curvature;
use super::subspace::{orthonormal_pairs, orthonormal_to, rayleigh_ritz, Pair};
use crate::core_infra::rng::Rng;
use crate::domain::MolecularSystem;
use crate::error::{MoptError, MoptResult};
use crate::math::linalg::{col_dot, col_norm};
use crate::path::points::{
    check_values, gradient_of, hessian_of, PointRequest, PointValue, SearchAction,
};

/// Settings of [`LowestModes`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModeSettings {
    /// Number of lowest curvatures sought.
    pub count: usize,
    /// Products from differences of gradients or an exact Hessian.
    pub curvature: Curvature,
    /// Displacement of the differences (the largest displacement of an atom).
    pub dimer_length: f64,
    /// Iterations at most.
    pub max_iterations: usize,
    /// Converged when every residual `|H v - c v|` is below this fraction of the largest
    /// curvature magnitude found.
    pub tolerance: f64,
    /// Seed of random start directions.
    pub seed: u64,
}

impl Default for ModeSettings {
    fn default() -> Self {
        Self {
            count: 2,
            curvature: Curvature::Forward,
            dimer_length: 1e-3,
            max_iterations: 100,
            tolerance: 1e-3,
            seed: 1,
        }
    }
}

#[derive(Debug, Clone)]
enum Phase {
    /// The gradient at the point (and the Hessian, or the products of the start).
    Start,
    /// Products along the residual directions.
    Residuals(Vec<Col<f64>>),
    Done,
}

/// Ask/tell search for the lowest curvatures at a point.
pub struct LowestModes {
    settings: ModeSettings,
    frame: SearchFrame,
    x: Col<f64>,
    given: Vec<Col<f64>>,
    removed: Option<Mat<f64>>,
    gradient_q: Col<f64>,
    /// Ritz pairs (direction, product) of the current block.
    block: Vec<Pair>,
    /// Conjugate directions and their products.
    conjugate: Vec<Pair>,
    values: Vec<f64>,
    phase: Phase,
    iterations: usize,
    gradients: usize,
    hessians: usize,
    converged: bool,
}

impl LowestModes {
    /// The lowest curvatures at `point`, starting from `start` (any number of directions;
    /// random ones fill up to the count).
    pub fn new(settings: ModeSettings, point: &Col<f64>, start: &[Col<f64>]) -> MoptResult<Self> {
        let n = point.nrows();
        if settings.count == 0 || settings.count > n || settings.max_iterations == 0 {
            return Err(MoptError::config(
                "lowest modes: the count must lie between 1 and the number of variables, and \
                 the iterations be positive",
            ));
        }
        if !(settings.dimer_length > 0.0 && settings.tolerance > 0.0) {
            return Err(MoptError::config(
                "lowest modes: the displacement and the tolerance must be positive",
            ));
        }
        if start.iter().any(|d| d.nrows() != n) {
            return Err(MoptError::config(
                "lowest modes: a start direction differs in size",
            ));
        }
        let mut s = Self {
            settings,
            frame: SearchFrame::plain(),
            x: point.clone(),
            given: start.to_vec(),
            removed: None,
            gradient_q: Col::zeros(n),
            block: Vec::new(),
            conjugate: Vec::new(),
            values: Vec::new(),
            phase: Phase::Start,
            iterations: 0,
            gradients: 0,
            hessians: 0,
            converged: false,
        };
        s.prepare()?;
        Ok(s)
    }

    /// As [`super::HighIndexSaddleSearch::with_structure`].
    pub fn with_structure(
        mut self,
        template: MolecularSystem,
        rigid: bool,
        mass_weighted: bool,
    ) -> MoptResult<Self> {
        let frozen = std::mem::take(&mut self.frame).frozen_variables();
        self.frame = SearchFrame::structure(template, rigid, mass_weighted).with_frozen(frozen);
        self.prepare()?;
        Ok(self)
    }

    /// Keep these variables where they are.
    pub fn with_frozen(mut self, variables: Vec<usize>) -> MoptResult<Self> {
        self.frame = std::mem::take(&mut self.frame).with_frozen(variables);
        self.prepare()?;
        Ok(self)
    }

    fn prepare(&mut self) -> MoptResult<()> {
        let n = self.x.nrows();
        self.frame.check(n)?;
        self.removed = self.frame.removed(&self.x);
        let mut rng = Rng::from_seed(self.settings.seed);
        let mut basis: Vec<Col<f64>> = Vec::new();
        for d in &self.given {
            let d = self
                .frame
                .mask(self.removed.as_ref(), &self.frame.direction_to_q(d));
            if let Some(b) = orthonormal_to(&basis, &d) {
                basis.push(b);
            }
        }
        let mut tries = 0;
        while basis.len() < self.settings.count {
            tries += 1;
            if tries > 20 * self.settings.count + 20 {
                return Err(MoptError::config(
                    "lowest modes: fewer free directions than curvatures sought",
                ));
            }
            let r = Col::from_fn(n, |_| rng.standard_normal());
            let r = self.frame.mask(self.removed.as_ref(), &r);
            if let Some(b) = orthonormal_to(&basis, &r) {
                basis.push(b);
            }
        }
        basis.truncate(self.settings.count);
        self.block = basis.into_iter().map(|v| (v, Col::zeros(n))).collect();
        Ok(())
    }

    fn displacement(&self, v: &Col<f64>) -> (Col<f64>, f64) {
        let d = self.frame.direction_to_x(v);
        let size = self.frame.largest_displacement(&d).max(1e-300);
        let c = self.settings.dimer_length / size;
        (d * c, c)
    }

    fn probes(&self, directions: &[Col<f64>]) -> Vec<PointRequest> {
        let mut out = Vec::new();
        for v in directions {
            let (d, _) = self.displacement(v);
            out.push(PointRequest::gradient(&self.x + &d));
            if self.settings.curvature == Curvature::Central {
                out.push(PointRequest::gradient(&self.x - &d));
            }
        }
        out
    }

    /// Next action.
    pub fn ask(&self) -> SearchAction {
        match &self.phase {
            Phase::Done => SearchAction::Finished,
            Phase::Start if self.settings.curvature == Curvature::Hessian => {
                SearchAction::Evaluate(vec![PointRequest::hessian(self.x.clone())])
            }
            Phase::Start => {
                let directions: Vec<Col<f64>> = self.block.iter().map(|p| p.0.clone()).collect();
                let mut points = vec![PointRequest::gradient(self.x.clone())];
                points.extend(self.probes(&directions));
                SearchAction::Evaluate(points)
            }
            Phase::Residuals(directions) => SearchAction::Evaluate(self.probes(directions)),
        }
    }

    fn products(
        &self,
        directions: &[Col<f64>],
        values: &[PointValue],
    ) -> MoptResult<Vec<Col<f64>>> {
        let per = if self.settings.curvature == Curvature::Central {
            2
        } else {
            1
        };
        let q = |value: &PointValue| -> MoptResult<Col<f64>> {
            Ok(self.frame.mask(
                self.removed.as_ref(),
                &self
                    .frame
                    .gradient_to_q(&gradient_of("lowest modes", value)?),
            ))
        };
        directions
            .iter()
            .enumerate()
            .map(|(j, v)| {
                let (_, c) = self.displacement(v);
                let plus = q(&values[per * j])?;
                Ok(if per == 2 {
                    (plus - q(&values[per * j + 1])?) * (0.5 / c)
                } else {
                    (plus - &self.gradient_q) * (1.0 / c)
                })
            })
            .collect()
    }

    /// The values of the points of [`Self::ask`].
    pub fn tell(&mut self, values: Vec<PointValue>) -> MoptResult<()> {
        let asked = match self.ask() {
            SearchAction::Evaluate(points) => points,
            SearchAction::Finished => {
                return Err(MoptError::config(
                    "tell() called after the mode search ended",
                ))
            }
        };
        check_values("lowest modes", &asked, &values, self.x.nrows())?;
        self.gradients += values.iter().filter(|v| v.gradient.is_some()).count();
        match std::mem::replace(&mut self.phase, Phase::Done) {
            Phase::Start => {
                self.gradient_q = self.frame.mask(
                    self.removed.as_ref(),
                    &self
                        .frame
                        .gradient_to_q(&gradient_of("lowest modes", &values[0])?),
                );
                if self.settings.curvature == Curvature::Hessian {
                    self.hessians += 1;
                    let h = self.frame.hessian_to_q(
                        self.removed.as_ref(),
                        &hessian_of("lowest modes", &values[0])?,
                    );
                    let (values, vectors) = self.frame.eigen(self.removed.as_ref(), &h)?;
                    let m = self.settings.count.min(vectors.len());
                    self.values = values[..m].to_vec();
                    self.block = vectors[..m].iter().map(|v| (v.clone(), &h * v)).collect();
                    self.converged = true;
                    return Ok(());
                }
                let directions: Vec<Col<f64>> = self.block.iter().map(|p| p.0.clone()).collect();
                let products = self.products(&directions, &values[1..])?;
                let pairs: Vec<Pair> = directions.into_iter().zip(products).collect();
                self.iterate(pairs, Vec::new())
            }
            Phase::Residuals(directions) => {
                let products = self.products(&directions, &values)?;
                let fresh: Vec<Pair> = directions.into_iter().zip(products).collect();
                let current = std::mem::take(&mut self.block);
                self.iterate(current, fresh)
            }
            Phase::Done => Err(MoptError::config(
                "tell() called after the mode search ended",
            )),
        }
    }

    /// The Rayleigh-Ritz problem on the block, the new directions and the conjugate ones;
    /// then the residuals of the new block.
    fn iterate(&mut self, block: Vec<Pair>, fresh: Vec<Pair>) -> MoptResult<()> {
        let m = self.settings.count;
        let old: Vec<Pair> = block.clone();
        let mut all = block;
        all.extend(fresh);
        all.extend(self.conjugate.iter().cloned());
        let ritz = rayleigh_ritz(&orthonormal_pairs(&all))?;
        let kept = m.min(ritz.vectors.len());
        let new_block: Vec<Pair> = (0..kept)
            .map(|j| (ritz.vectors[j].clone(), ritz.products[j].clone()))
            .collect();
        // The conjugate directions: the parts of the new block outside the old one.
        self.conjugate = new_block
            .iter()
            .map(|(y, hy)| {
                let (mut p, mut hp) = (y.clone(), hy.clone());
                for (v, hv) in &old {
                    let c = col_dot(v, y);
                    p -= v * c;
                    hp -= hv * c;
                }
                (p, hp)
            })
            .filter(|(p, _)| col_norm(p) > 1e-10)
            .collect();
        self.values = ritz.values[..kept].to_vec();
        self.block = new_block;
        self.iterations += 1;
        let scale = self.values.iter().fold(1e-12_f64, |s, v| s.max(v.abs()));
        let residuals: Vec<Col<f64>> = self
            .block
            .iter()
            .zip(&self.values)
            .map(|((v, hv), c)| self.frame.mask(self.removed.as_ref(), &(hv - v * *c)))
            .collect();
        let worst = residuals.iter().fold(0.0_f64, |w, r| w.max(col_norm(r)));
        if worst <= self.settings.tolerance * scale
            || self.iterations >= self.settings.max_iterations
        {
            self.converged = worst <= self.settings.tolerance * scale;
            self.phase = Phase::Done;
            return Ok(());
        }
        let mut basis: Vec<Col<f64>> = self.block.iter().map(|p| p.0.clone()).collect();
        basis.extend(self.conjugate.iter().map(|p| p.0.clone()));
        let mut directions = Vec::new();
        for r in residuals {
            if let Some(d) = orthonormal_to(&basis, &r) {
                basis.push(d.clone());
                directions.push(d);
            }
        }
        if directions.is_empty() {
            self.converged = true;
            self.phase = Phase::Done;
            return Ok(());
        }
        self.phase = Phase::Residuals(directions);
        Ok(())
    }

    /// The curvatures found (ascending, in the metric of the search).
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// Their directions in the metric of the search (orthonormal).
    pub fn directions(&self) -> Vec<Col<f64>> {
        self.block.iter().map(|p| p.0.clone()).collect()
    }

    /// Their directions in the coordinates of the point, each of unit length.
    pub fn point_directions(&self) -> Vec<Col<f64>> {
        self.block
            .iter()
            .map(|(v, _)| {
                let d = self.frame.direction_to_x(v);
                let length = col_norm(&d);
                if length > 0.0 {
                    d * (1.0 / length)
                } else {
                    d
                }
            })
            .collect()
    }

    /// Whether the residuals met the tolerance.
    pub fn converged(&self) -> bool {
        self.converged
    }

    /// Whether the search has ended.
    pub fn finished(&self) -> bool {
        matches!(self.phase, Phase::Done)
    }

    /// Gradients evaluated (the differences included).
    pub fn gradient_evaluations(&self) -> usize {
        self.gradients
    }

    /// Hessians evaluated.
    pub fn hessian_evaluations(&self) -> usize {
        self.hessians
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn the_lowest_curvatures_of_a_quadratic_from_gradients() {
        let a = Mat::<f64>::from_fn(5, 5, |i, j| {
            if i == j {
                [-3.0, -1.0, 0.5, 2.0, 7.0][i]
            } else {
                0.05 * ((i + 2 * j) % 3) as f64 - 0.05 * ((j + 2 * i) % 3) as f64
                    + 0.02 * (i + j) as f64
            }
        });
        let a = (&a + a.transpose()) * 0.5;
        let value = |x: &Col<f64>| PointValue {
            energy: 0.5 * col_dot(x, &(&a * x)),
            gradient: Some(&a * x),
            hessian: Some(a.clone()),
        };
        let exact = crate::math::linalg::symmetric_eigen(&a).unwrap().0;
        for curvature in [Curvature::Forward, Curvature::Hessian] {
            let settings = ModeSettings {
                count: 3,
                curvature,
                tolerance: 1e-6,
                ..ModeSettings::default()
            };
            let x = Col::from_fn(5, |k| 0.1 * k as f64);
            let mut m = LowestModes::new(settings, &x, &[]).unwrap();
            while let SearchAction::Evaluate(points) = m.ask() {
                m.tell(points.iter().map(|p| value(&p.x)).collect())
                    .unwrap();
            }
            assert!(m.converged());
            for j in 0..3 {
                assert!(
                    (m.values()[j] - exact[j]).abs() < 1e-5,
                    "{curvature:?} {:?}",
                    m.values()
                );
            }
        }
    }
}
