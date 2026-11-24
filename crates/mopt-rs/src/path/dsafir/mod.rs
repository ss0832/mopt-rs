//! The double-sphere artificial force induced reaction method (DS-AFIR): the two given
//! structures approach each other, the lower of the two taking a step down the AFIR
//! function at a time, until they meet; the points they passed are a path between them, whose
//! highest point is the estimate of the transition state.
//!
//! With the moving end `q`, the other end `p`, and the reference `q0` of `q` (where it
//! started, then the latest minimum of the energy along its path), the function is
//! (S. Maeda, Y. Harabuchi, M. Takagi, K. Saita, K. Suzuki, T. Ichino, Y. Sumiya, K. Sugiyama,
//! Y. Ono, J. Comput. Chem. 2018, 39, 233, eqs. 6-10)
//!
//! `F(q) = E(q) + X Y |q - p| - X (1 - Y) |q - q0|`,
//!
//! `Z = |q - q0| / |q - p| + (q - p) . (q - q0) / (|q - p| |q - q0|)`, `Y = Z / (1 + Z)` for
//! `Z > 0` and 0 otherwise, `u = Y (q - p) / |q - p| - (1 - Y) (q - q0) / |q - q0|`, and
//! `X = delta / |u| - g . u / |u|^2`, so that the slope of `F` along `u` is `delta`: the end
//! is drawn towards the other with the force `delta` near the ridge and pushed away from its
//! reference near a minimum (`Y = 1` while `q` is its reference). The path goes down
//! `grad F = g + X u` (`X`, `Y` held over a step); the ends have met when they are closer
//! than `separation` (0.12 Angstrom in the paper). The distance is the plain Cartesian one of
//! the points (the paper: non-mass-weighted coordinates). Rigid-body motions are taken out
//! of `g` and of `u` (`X` is that of the projected `u`, so that the slope along the step is
//! `delta`), as in every method of mopt-rs.
//!
//! The paper integrates that path by the local quadratic approximation ("the steepest
//! descent path is integrated by the LQA method"), which [`DsIntegrator::Lqa`] does
//! ([`lqa`]): each step follows the steepest-descent path of the quadratic model of `F` for
//! the arc length `step` (the manual of the authors' program: 0.1 Angstrom), with the
//! Hessian of the energy exact at every `hessian_every`-th point of an end and updated in
//! between (the manual names an exact Hessian every 10 steps for its IRC; for DS-AFIR it
//! does not say). [`DsIntegrator::Euler`] takes steps of the fixed length straight down
//! `grad F` instead; on molecules those overshoot the stiff modes across the drive, zigzag,
//! and push the ends hundreds of kcal/mol up (what mopt-rs did before it had the LQA steps).
//! `delta` is 0.03 Hartree/Bohr in the manual (the paper gives values for the Muller-Brown
//! surface only), and the second structure is turned onto the first at the start.
//!
//! The paper's own procedure then relaxes the path by the locally updated planes method and
//! optimizes its maxima. The relaxation is not part of DS-AFIR: it is a general relaxation
//! of a path, a step after any two-point method (`method_relax`,
//! [`crate::workflow::pair_search::relax_path`], with the chain of `nebmain -lup`,
//! [`crate::path::band::lup`]). DS-AFIR ends with its own path and the highest point of it
//! (the paper's `DS = Single`).
//!
//! Additions of mopt-rs, each switchable ([`DsAfirSettings`]): laying the other end onto the
//! moving one before every step (`align`; the superposition of the start alone can leave the
//! straight line between the ends running through atoms), leaving the second term of `F`
//! out (`attraction_only`, `Y = 1`), the second derivatives of the two distance terms in the
//! Hessian of the LQA step (`afir_curvature`), moving the two ends in turn (`alternate`),
//! and, from before the LQA steps, L-BFGS steps down the AFIR function (`quasi_newton`; `F`
//! is no potential, so they run past the saddle) and up to `relax_steps` L-BFGS steps of the
//! moving end across its drive direction `u` after each drive step (`relax_steps`).
//!
//! The two ends can be on two surfaces (two electronic states, or two charges or spins of
//! the same atoms): each end is then evaluated on its own surface ([`DoubleSphereAfir::
//! pending_ends`] says which point belongs to which end), the lower one still moves, and
//! the point where the ends meet ([`DoubleSphereAfir::meeting_point`]) is an estimate of a
//! crossing of the two surfaces, to be refined by a search for the crossing. That use is an
//! addition of mopt-rs (the paper's text was not at hand for what its authors do there);
//! the path has no single energy profile then, so it is not relaxed and has no highest
//! point to report.
//!
//! [`DoubleSphereAfir`] is an ask/tell state machine over points of `n` variables.
//!
//! MultiOptPy: no counterpart (new in mopt-rs; the weights of DS-AFIR appear in MultiOptPy's
//! iEIP in another form, see [`super::ieip::forces`]).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

pub mod lqa;
mod settings;
#[cfg(test)]
mod tests;

use faer::{Col, Mat};

use lqa::EndCurvature;
pub use settings::{DsAfirSettings, DsIntegrator, PAPER_FORCE};

use super::points::{check_values, gradient_of, PointRequest, PointValue, SearchAction};
use super::string_nodes::{across, rms, unit, NodeStepSettings, NodeSteps, PointGeometry};
use crate::error::{MoptError, MoptResult};
use crate::math::linalg::{col_dot, col_norm};

/// Why DS-AFIR ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DsAfirEnd {
    /// The ends met.
    Met,
    /// The steps are used up before the ends met.
    StepLimit,
}

/// What one `tell` did.
#[derive(Debug, Clone)]
pub struct DsAfirReport {
    /// Drive steps of the ends so far.
    pub steps: usize,
    /// The end that moved (0: the first, 1: the second; `None`: the start or the relaxation).
    pub moved: Option<usize>,
    /// Energies of the two ends.
    pub energies: [f64; 2],
    /// Distance of the ends.
    pub separation: f64,
    /// `X` and `Y` of the drive step taken (`None` for a relaxation across the drive).
    pub weights: Option<(f64, f64)>,
    /// Why the search ended, if it did.
    pub end: Option<DsAfirEnd>,
}

#[derive(Debug, Clone)]
struct Side {
    path: Vec<(Col<f64>, f64, Col<f64>)>,
    /// The energies of the points the drive steps started from (the reference is the latest
    /// minimum among them).
    settled: Vec<(Col<f64>, f64)>,
    reference: Col<f64>,
}

/// The drive at an end: `u`, `X`, `Y`, and `grad F`.
struct Drive {
    u: Col<f64>,
    x: f64,
    y: f64,
    grad_f: Col<f64>,
    curvature: Mat<f64>,
}

/// Ask/tell state machine of DS-AFIR.
pub struct DoubleSphereAfir {
    settings: DsAfirSettings,
    geometry: PointGeometry,
    sides: Option<[Side; 2]>,
    ends: [Col<f64>; 2],
    pending: Vec<(usize, Col<f64>)>,
    steps: usize,
    turn: usize,
    end: Option<DsAfirEnd>,
    free: usize,
    step_map: Option<super::points::StepMap>,
    quasi_newton: [NodeSteps; 2],
    relaxers: [NodeSteps; 2],
    relaxing: Option<(usize, usize)>,
    curvature: [EndCurvature; 2],
}

impl DoubleSphereAfir {
    /// Take the steps of the ends through other coordinates (the AFIR function and the
    /// distance of the ends stay in the coordinates of the points): `map` turns the step
    /// from `x` to `y` into the point it reaches.
    pub fn with_step_map(mut self, map: super::points::StepMap) -> Self {
        self.step_map = Some(map);
        self
    }

    /// From `first` and `second` (the second laid onto the first for structures without a
    /// preferred orientation).
    pub fn new(
        settings: DsAfirSettings,
        first: &Col<f64>,
        second: &Col<f64>,
        geometry: PointGeometry,
    ) -> MoptResult<Self> {
        let n = first.nrows();
        let positive = [
            settings.force,
            settings.step,
            settings.separation,
            settings.relax_tolerance,
        ];
        if n == 0
            || second.nrows() != n
            || positive.iter().any(|v| !(v.is_finite() && *v > 0.0))
            || settings.max_steps == 0
        {
            return Err(MoptError::config(
                "DS-AFIR: two points of one size, a positive force, step, separation, \
                 tolerance, and count",
            ));
        }
        let second = geometry.laid_onto(second, first);
        let free = n - geometry.frozen.len().min(n);
        let qn = NodeStepSettings {
            max_step: settings.step,
            ..NodeStepSettings::default()
        };
        let relax = settings.relax_steps_settings;
        Ok(Self {
            pending: vec![(0, first.clone()), (1, second.clone())],
            ends: [first.clone(), second],
            settings,
            geometry,
            sides: None,
            steps: 0,
            turn: 0,
            end: None,
            free,
            step_map: None,
            quasi_newton: [NodeSteps::new(qn), NodeSteps::new(qn)],
            relaxers: [NodeSteps::new(relax), NodeSteps::new(relax)],
            relaxing: None,
            curvature: [EndCurvature::default(), EndCurvature::default()],
        })
    }

    /// The path: the points of the first end, then those of the second backwards, with their
    /// energies.
    pub fn path(&self) -> Vec<(Col<f64>, f64)> {
        let Some(sides) = &self.sides else {
            return Vec::new();
        };
        let mut out: Vec<(Col<f64>, f64)> = sides[0]
            .path
            .iter()
            .map(|(x, e, _)| (x.clone(), *e))
            .collect();
        out.extend(sides[1].path.iter().rev().map(|(x, e, _)| (x.clone(), *e)));
        out
    }

    /// The two ends where they are now, with their energies (`None` before they were
    /// evaluated).
    pub fn current_ends(&self) -> Option<[(Col<f64>, f64); 2]> {
        let sides = self.sides.as_ref()?;
        let last = |k: usize| sides[k].path.last().map(|(x, e, _)| (x.clone(), *e));
        Some([last(0)?, last(1)?])
    }

    /// The end each pending point belongs to (0 or 1).
    pub fn pending_ends(&self) -> Vec<usize> {
        self.pending
            .iter()
            .map(|(side, _)| (*side).min(1))
            .collect()
    }

    /// Where the ends are met: the midpoint of the two ends as they are now (the second laid
    /// onto the first for structures without a preferred orientation), with the energies of
    /// the two ends.
    pub fn meeting_point(&self) -> Option<(Col<f64>, [f64; 2])> {
        let [(q, first), (p, second)] = self.current_ends()?;
        let p = self.geometry.laid_onto(&p, &q);
        Some(((&q + &p) * 0.5, [first, second]))
    }

    /// The highest point of the path between its ends.
    pub fn transition_state_estimate(&self) -> Option<(Col<f64>, f64)> {
        let path = self.path();
        let last = path.len().saturating_sub(1);
        path.into_iter()
            .enumerate()
            .filter(|(k, _)| *k > 0 && *k < last)
            .map(|(_, p)| p)
            .max_by(|a, b| a.1.total_cmp(&b.1))
    }

    /// Why the search ended, once it has.
    pub fn end(&self) -> Option<DsAfirEnd> {
        self.end
    }

    /// Next action.
    pub fn ask(&self) -> SearchAction {
        if self.end.is_some() {
            return SearchAction::Finished;
        }
        SearchAction::Evaluate(self.requests())
    }

    /// Whether the ends take LQA steps.
    fn takes_lqa_steps(&self) -> bool {
        self.settings.integrator == DsIntegrator::Lqa && !self.settings.quasi_newton
    }

    /// What is asked for at the pending points: the gradient, and the Hessian where an end
    /// that takes LQA steps is due an exact one.
    fn requests(&self) -> Vec<PointRequest> {
        let ends = self.takes_lqa_steps();
        self.pending
            .iter()
            .map(|(side, x)| {
                if ends
                    && self.relaxing.is_none()
                    && self.curvature[(*side).min(1)].needs_exact(self.settings.hessian_every)
                {
                    PointRequest::hessian(x.clone())
                } else {
                    PointRequest::gradient(x.clone())
                }
            })
            .collect()
    }

    fn current(&self, side: usize) -> MoptResult<&(Col<f64>, f64, Col<f64>)> {
        self.sides
            .as_ref()
            .and_then(|s| s[side].path.last())
            .ok_or_else(|| MoptError::config("DS-AFIR: an end without a point"))
    }

    /// The other end as the moving one sees it.
    fn other_end(&self, moving: usize) -> MoptResult<Col<f64>> {
        let q = &self.current(moving)?.0;
        let p = &self.current(1 - moving)?.0;
        Ok(if self.settings.align {
            self.geometry.laid_onto(p, q)
        } else {
            p.clone()
        })
    }

    fn separation(&self) -> MoptResult<f64> {
        let q = &self.current(0)?.0;
        Ok(col_norm(&(q - &self.other_end(0)?)))
    }

    fn record(
        &mut self,
        pending: Vec<(usize, Col<f64>)>,
        values: Vec<PointValue>,
    ) -> MoptResult<Option<usize>> {
        let mut moved = None;
        let lqa = self.takes_lqa_steps();
        for ((side, x), v) in pending.into_iter().zip(values) {
            let raw = gradient_of("DS-AFIR", &v)?;
            if lqa {
                // The curvature of the energy along the walk of this end.
                self.curvature[side.min(1)].take(&x, &raw, v.hessian.as_ref());
            }
            let g = self.geometry.project(&x, &raw);
            if self.sides.is_none() {
                let start = |r: &Col<f64>| Side {
                    path: Vec::new(),
                    settled: Vec::new(),
                    reference: r.clone(),
                };
                self.sides = Some([start(&self.ends[0]), start(&self.ends[1])]);
            } else {
                moved = Some(side);
            }
            if let Some(sides) = self.sides.as_mut() {
                sides[side].path.push((x, v.energy, g));
            }
        }
        Ok(moved)
    }

    /// The values at the points of [`Self::ask`].
    pub fn tell(&mut self, values: Vec<PointValue>) -> MoptResult<DsAfirReport> {
        if self.end.is_some() {
            return Err(MoptError::config("tell() called after DS-AFIR ended"));
        }
        let asked = self.requests();
        check_values("DS-AFIR", &asked, &values, self.ends[0].nrows())?;
        let pending = std::mem::take(&mut self.pending);
        let moved = self.record(pending, values)?;
        let mut report = DsAfirReport {
            steps: self.steps,
            moved,
            energies: [self.current(0)?.1, self.current(1)?.1],
            separation: self.separation()?,
            weights: None,
            end: None,
        };
        if report.separation < self.settings.separation {
            self.end = Some(DsAfirEnd::Met);
            report.end = self.end;
            return Ok(report);
        }
        if let Some(side) = moved {
            if self.relax_across(side)? {
                return Ok(report);
            }
        }
        self.relaxing = None;
        if self.steps >= self.settings.max_steps {
            self.end = Some(DsAfirEnd::StepLimit);
            report.end = self.end;
            return Ok(report);
        }
        let e = report.energies;
        let side = if self.settings.alternate {
            self.turn
        } else if e[0] <= e[1] {
            0
        } else {
            1
        };
        self.turn = 1 - side;
        self.settle(side)?;
        let (x, weights) = self.drive_step(side)?;
        report.weights = Some(weights);
        self.pending = vec![(side, x)];
        self.steps += 1;
        Ok(report)
    }

    /// The point of `side` a drive step starts from settles; the reference becomes the latest
    /// minimum of the energies of such points.
    fn settle(&mut self, side: usize) -> MoptResult<()> {
        let (x, e, _) = self.current(side)?.clone();
        let Some(sides) = self.sides.as_mut() else {
            return Ok(());
        };
        let s = &mut sides[side];
        s.settled.push((x, e));
        let k = s.settled.len();
        if k >= 3 {
            let (a, b, c) = (s.settled[k - 3].1, s.settled[k - 2].1, s.settled[k - 1].1);
            if b < a && b < c {
                s.reference = s.settled[k - 2].0.clone();
                // F has changed: the history of its steps no longer fits.
                self.quasi_newton[side].reset();
            }
        }
        Ok(())
    }

    /// The drive at end `side` (eqs. 7-10).
    fn drive(&self, side: usize) -> MoptResult<Drive> {
        let (q, _, g) = self.current(side)?.clone();
        let p = self.other_end(side)?;
        let q0 = self
            .sides
            .as_ref()
            .map(|s| s[side].reference.clone())
            .ok_or_else(|| MoptError::config("DS-AFIR: no reference"))?;
        let to_other = &q - &p;
        let from_reference = &q - &q0;
        let (a, b) = (col_norm(&from_reference), col_norm(&to_other));
        if b == 0.0 {
            return Err(MoptError::calc("DS-AFIR: the ends coincide"));
        }
        let y = if a > 0.0 && !self.settings.attraction_only {
            let z = a / b + col_dot(&to_other, &from_reference) / (a * b);
            if z > 0.0 {
                z / (1.0 + z)
            } else {
                0.0
            }
        } else {
            // The second term is left out while the end is at its reference.
            1.0
        };
        let mut u = &to_other * (y / b);
        if a > 0.0 {
            u -= &from_reference * ((1.0 - y) / a);
        }
        // Without rigid-body motions and fixed variables, as `g` is: the slope of `F` along
        // the direction the end can move in is `delta`.
        let u = self.geometry.project(&q, &u);
        let un = col_norm(&u);
        if un == 0.0 {
            return Err(MoptError::calc("DS-AFIR: the two forces cancel"));
        }
        let x = self.settings.force / un - col_dot(&g, &u) / (un * un);
        let grad_f = &g + &u * x;
        // The second derivatives of the two distance terms of `F` (`X`, `Y` held).
        let n = q.nrows();
        let mut curvature = Mat::<f64>::zeros(n, n);
        let mut add = |d: &Col<f64>, length: f64, weight: f64| {
            for i in 0..n {
                for j in 0..n {
                    let delta = if i == j { 1.0 } else { 0.0 };
                    curvature[(i, j)] +=
                        weight * (delta - d[i] * d[j] / (length * length)) / length;
                }
            }
        };
        add(&to_other, b, x * y);
        if a > 0.0 {
            add(&from_reference, a, -x * (1.0 - y));
        }
        Ok(Drive {
            u,
            x,
            y,
            grad_f,
            curvature,
        })
    }

    /// `h` at `x` without the rigid-body motions and the fixed variables (on both sides).
    fn projected(&self, x: &Col<f64>, h: &Mat<f64>) -> Mat<f64> {
        let n = h.nrows();
        let mut half = Mat::<f64>::zeros(n, n);
        for j in 0..n {
            let column = self.geometry.project(x, &Col::from_fn(n, |i| h[(i, j)]));
            for i in 0..n {
                half[(i, j)] = column[i];
            }
        }
        let mut out = Mat::<f64>::zeros(n, n);
        for i in 0..n {
            let row = self.geometry.project(x, &Col::from_fn(n, |j| half[(i, j)]));
            for j in 0..n {
                out[(i, j)] = row[j];
            }
        }
        Mat::from_fn(n, n, |i, j| 0.5 * (out[(i, j)] + out[(j, i)]))
    }

    /// The LQA step of end `side` from `q` with the drive `d`.
    fn lqa_step(&self, side: usize, q: &Col<f64>, d: &Drive) -> MoptResult<Col<f64>> {
        let mut h = self.curvature[side]
            .hessian()
            .ok_or_else(|| {
                MoptError::calc(
                    "DS-AFIR: no Hessian of the energy at an end (integrator=euler takes \
                     steps without one)",
                )
            })?
            .clone();
        if self.settings.afir_curvature {
            h += &d.curvature;
        }
        let h = self.projected(q, &h);
        lqa::lqa_displacement(&h, &d.grad_f, self.settings.step)?
            .ok_or_else(|| MoptError::calc("DS-AFIR: the AFIR function is flat"))
    }

    /// The point a step reaches (through the coordinates of the step map, if any).
    fn reached(&self, from: &Col<f64>, to: Col<f64>) -> MoptResult<Col<f64>> {
        match &self.step_map {
            Some(map) => map(from, &to),
            None => Ok(to),
        }
    }

    /// The next point of end `side` and the weights `(X, Y)` of its drive step.
    fn drive_step(&mut self, side: usize) -> MoptResult<(Col<f64>, (f64, f64))> {
        let q = self.current(side)?.0.clone();
        let d = self.drive(side)?;
        let next = if self.settings.quasi_newton {
            let geometry = self.geometry.clone();
            let at = q.clone();
            let project = move |v: &Col<f64>| geometry.project(&at, v);
            &q + self.quasi_newton[side].step(&q, &d.grad_f, &project)
        } else if self.settings.integrator == DsIntegrator::Lqa {
            &q + self.lqa_step(side, &q, &d)?
        } else {
            let dir = unit(&d.grad_f)
                .ok_or_else(|| MoptError::calc("DS-AFIR: the AFIR function is flat"))?;
            &q - &dir * self.settings.step
        };
        Ok((self.reached(&q, next)?, (d.x, d.y)))
    }

    /// A step of end `side` across its drive direction, if the settings ask for one and the
    /// gradient across it is not small yet; returns whether one was asked for.
    fn relax_across(&mut self, side: usize) -> MoptResult<bool> {
        if self.settings.relax_steps == 0 {
            return Ok(false);
        }
        // The pairs of the relaxations of an end stay from one drive step to the next (the
        // curvature of the energy there changes little).
        let done = match self.relaxing {
            Some((s, c)) if s == side => c,
            _ => 0,
        };
        if done >= self.settings.relax_steps {
            return Ok(false);
        }
        let (q, _, g) = self.current(side)?.clone();
        let Some(t) = unit(&self.geometry.project(&q, &self.drive(side)?.u)) else {
            return Ok(false);
        };
        let g_across = across(&g, &t);
        if rms(&g_across, self.free) < self.settings.relax_tolerance {
            return Ok(false);
        }
        let geometry = self.geometry.clone();
        let at = q.clone();
        let project = move |v: &Col<f64>| geometry.project(&at, &across(v, &t));
        let p = self.relaxers[side].step(&q, &g_across, &project);
        let next = self.reached(&q, &q + p)?;
        self.pending = vec![(side, next)];
        self.relaxing = Some((side, done + 1));
        Ok(true)
    }
}
