//! What the tests of the saddle searches on the seven-atom Lennard-Jones cluster share: the
//! built-in `lj` surface of argon in reduced units, the identity of stationary points (energy
//! and sorted interatomic distances: permutations, rotations and the inversion make no new
//! point), the refinement by Newton steps on the exact Hessian at a given order, and the
//! searches of high-index saddle dynamics run on threads.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(dead_code, clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use faer::{Col, Mat};
use mopt_rs::domain::{positions, Bohr, Element, EvalRequest, MolecularSystem, SpinCharge, Units};
use mopt_rs::opt::convergence::ConvergenceCriteria;
use mopt_rs::path::hisd::landscape::SearchStart;
use mopt_rs::path::hisd::{HandOff, HighIndexSaddleSearch, HisdEnd, HisdSettings, SearchFrame};
use mopt_rs::path::points::{PointValue, SearchAction};
use mopt_rs::pes::native::LennardJones;
use mopt_rs::pes::Calculator;

pub const ATOMS: usize = 7;
pub const N: usize = 3 * ATOMS;

/// The built-in Lennard-Jones surface of argon in reduced units (energies in epsilon,
/// lengths in sigma).
pub struct Reduced {
    pub template: MolecularSystem,
    sigma: f64,
    epsilon: f64,
}

impl Default for Reduced {
    fn default() -> Self {
        Self::new()
    }
}

impl Reduced {
    pub fn new() -> Self {
        let ar = Element::parse_token("Ar").unwrap();
        let points: Vec<[f64; 3]> = (0..ATOMS).map(|i| [i as f64 * 1.5, 0.0, 0.0]).collect();
        let template = MolecularSystem::new(
            vec![ar; ATOMS],
            &positions::<Bohr>(&points),
            SpinCharge::default(),
            None,
        )
        .unwrap();
        // The UFF parameters of the calculator: sigma = x / 2^(1/6), epsilon = D.
        let sigma = 3.817 / libm::pow(2.0, 1.0 / 6.0) * Units::ANGSTROM_TO_BOHR;
        let epsilon = 0.237 / Units::HARTREE_TO_KCAL_MOL;
        Self {
            template,
            sigma,
            epsilon,
        }
    }

    /// Energy, gradient, and (when asked) Hessian at `x` (reduced units).
    pub fn eval(&self, x: &Col<f64>, hessian: bool) -> (f64, Col<f64>, Option<Mat<f64>>) {
        let sys = self
            .template
            .with_flat_coords_bohr(&(x * self.sigma))
            .unwrap();
        let request = if hessian {
            EvalRequest::ENERGY_GRAD_HESS
        } else {
            EvalRequest::ENERGY_GRAD
        };
        let out = LennardJones.evaluate(&sys, &request).unwrap();
        (
            out.energy.value() / self.epsilon,
            out.gradient_bohr * (self.sigma / self.epsilon),
            out.hessian_bohr
                .map(|h| h * (self.sigma * self.sigma / self.epsilon)),
        )
    }

    /// The values a search asks for.
    pub fn values(&self, points: &[mopt_rs::path::points::PointRequest]) -> Vec<PointValue> {
        points
            .iter()
            .map(|p| {
                let (energy, gradient, hessian) = self.eval(&p.x, p.hessian);
                PointValue {
                    energy,
                    gradient: Some(gradient),
                    hessian,
                }
            })
            .collect()
    }

    /// The frame of the searches: rigid-body motions left out (equal masses, so the
    /// mass-weighted metric is the Cartesian one scaled).
    pub fn frame(&self) -> SearchFrame {
        SearchFrame::structure(self.template.clone(), true, false)
    }
}

pub fn distances(x: &Col<f64>) -> Vec<f64> {
    let mut d = Vec::with_capacity(ATOMS * (ATOMS - 1) / 2);
    for i in 0..ATOMS {
        for j in (i + 1)..ATOMS {
            d.push(
                ((0..3)
                    .map(|c| (x[3 * i + c] - x[3 * j + c]).powi(2))
                    .sum::<f64>())
                .sqrt(),
            );
        }
    }
    d.sort_by(f64::total_cmp);
    d
}

/// Whether the cluster is one piece with neighbours within `limit`.
pub fn bound(x: &Col<f64>, limit: f64) -> bool {
    let mut reached = [false; ATOMS];
    reached[0] = true;
    let mut stack = vec![0];
    while let Some(i) = stack.pop() {
        for j in 0..ATOMS {
            let r = ((0..3)
                .map(|c| (x[3 * i + c] - x[3 * j + c]).powi(2))
                .sum::<f64>())
            .sqrt();
            if !reached[j] && r < limit {
                reached[j] = true;
                stack.push(j);
            }
        }
    }
    reached.iter().all(|r| *r)
}

/// A stationary point.
#[derive(Clone)]
pub struct Stationary {
    pub x: Col<f64>,
    pub energy: f64,
    /// Curvatures below -1e-6 (without the rigid-body motions).
    pub index: usize,
    /// Curvatures of magnitude at most 1e-6: the index is not decided.
    pub near_zero: usize,
    pub distances: Vec<f64>,
    pub curvatures: Vec<f64>,
    pub modes: Vec<Col<f64>>,
}

impl Stationary {
    /// The same structure: energy within 1e-6 epsilon, sorted distances within 1e-3 sigma,
    /// and curvatures within 1e-3 of their size (at least 1). Without the curvatures two
    /// distinct points of LJ7 of index 3 fall together; tighter distances split one point
    /// whose position along a soft mode is known only to about 1e-3.
    pub fn same(&self, other: &Self) -> bool {
        (self.energy - other.energy).abs() < 1e-6
            && self
                .distances
                .iter()
                .zip(&other.distances)
                .all(|(a, b)| (a - b).abs() < 1e-3)
            && self.curvatures.len() == other.curvatures.len()
            && self
                .curvatures
                .iter()
                .zip(&other.curvatures)
                .all(|(a, b)| (a - b).abs() < 1e-3 * a.abs().max(1.0))
    }
}

/// Why a search or a refinement failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Failure {
    /// Not converged in the steps allowed (or the search ended without a point).
    Steps,
    /// The cluster fell apart.
    Apart,
}

/// Newton steps on the exact Hessian without the rigid-body motions, to a gradient of
/// 1e-9. With `order = Some(k)` the steps go uphill along the `k` lowest modes and downhill
/// along the rest, with the curvature magnitudes (Newton on the reflected problem: the
/// eigenvector following of a refinement at order `k`); with `None` plain Newton.
pub fn refine(
    lj: &Reduced,
    start: &Col<f64>,
    order: Option<usize>,
    hessians: &mut usize,
) -> Result<Stationary, Failure> {
    let frame = lj.frame();
    let mut x = start.clone();
    for _ in 0..60 {
        let (energy, g, h) = lj.eval(&x, true);
        *hessians += 1;
        let removed = frame.removed(&x);
        let h_q = frame.hessian_to_q(removed.as_ref(), &h.ok_or(Failure::Steps)?);
        let (values, vectors) = frame
            .eigen(removed.as_ref(), &h_q)
            .map_err(|_| Failure::Steps)?;
        let g = frame.mask(removed.as_ref(), &g);
        let largest = g.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        if largest < 1e-9 {
            if !bound(&x, 1.7) {
                return Err(Failure::Apart);
            }
            return Ok(Stationary {
                distances: distances(&x),
                index: values.iter().filter(|v| **v < -1e-6).count(),
                near_zero: values.iter().filter(|v| v.abs() <= 1e-6).count(),
                x,
                energy,
                curvatures: values,
                modes: vectors,
            });
        }
        let mut dx = Col::<f64>::zeros(N);
        for (j, (value, v)) in values.iter().zip(&vectors).enumerate() {
            if value.abs() > 1e-8 {
                let c: f64 = v.iter().zip(g.iter()).map(|(a, b)| a * b).sum();
                let curvature = match order {
                    Some(k) if j < k => -value.abs(),
                    Some(_) => value.abs(),
                    None => *value,
                };
                dx -= v * (c / curvature);
            }
        }
        let size = frame.largest_displacement(&dx);
        if size > 0.1 {
            dx *= 0.1 / size;
        }
        x += dx;
        if !bound(&x, 2.5) {
            return Err(Failure::Apart);
        }
    }
    Err(Failure::Steps)
}

/// The settings of the searches of these tests (reduced units).
pub fn settings(order: usize) -> HisdSettings {
    let mut s = HisdSettings::hiosd(order);
    s.criteria = ConvergenceCriteria::new(1e-6, 1e-6, 1.0, 1.0);
    s.hand_off = Some(HandOff {
        rms_gradient: 1e-2,
        projector_change: 0.1,
    });
    s.step.first = 0.02;
    s.step.max_step = 0.1;
    s.max_iterations = 500;
    s
}

/// What one search found.
pub struct Outcome {
    pub found: Option<Stationary>,
    pub failure: Option<Failure>,
    pub end: Option<HisdEnd>,
    pub gradients: usize,
    pub hessians: usize,
}

/// A minimization by Barzilai-Borwein steps, to a gradient of 1e-4.
pub fn minimize(lj: &Reduced, start: &Col<f64>, gradients: &mut usize) -> Option<Col<f64>> {
    let frame = lj.frame();
    let mut x = start.clone();
    let mut last: Option<(Col<f64>, Col<f64>)> = None;
    for _ in 0..2000 {
        let (_, g, _) = lj.eval(&x, false);
        *gradients += 1;
        if g.iter().fold(0.0_f64, |m, v| m.max(v.abs())) < 1e-4 {
            return Some(x);
        }
        let beta = match &last {
            Some((xb, gb)) => {
                let s = &x - xb;
                let y = &g - gb;
                let sy: f64 = s.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
                let yy: f64 = y.iter().map(|a| a * a).sum();
                if sy > 0.0 && yy > 0.0 {
                    sy / yy
                } else {
                    0.01
                }
            }
            None => 0.01,
        };
        last = Some((x.clone(), g.clone()));
        let mut dx = &g * -beta;
        let size = frame.largest_displacement(&dx);
        if size > 0.1 {
            dx *= 0.1 / size;
        }
        x += dx;
        if !bound(&x, 2.5) {
            return None;
        }
    }
    None
}

/// One search from `start` (a minimization for order 0), then the refinement at its order.
pub fn search(
    lj: &Reduced,
    start: &SearchStart,
    settings: &dyn Fn(usize) -> HisdSettings,
) -> Outcome {
    let mut gradients = 0;
    let mut hessians = 0;
    let failed = |failure, gradients, hessians| Outcome {
        found: None,
        failure: Some(failure),
        end: None,
        gradients,
        hessians,
    };
    if start.order == 0 {
        let result = minimize(lj, &start.point, &mut gradients)
            .ok_or(Failure::Apart)
            .and_then(|x| refine(lj, &x, Some(0), &mut hessians));
        return Outcome {
            failure: result.as_ref().err().copied(),
            found: result.ok(),
            end: None,
            gradients,
            hessians,
        };
    }
    let machine =
        HighIndexSaddleSearch::new(settings(start.order), &start.point, &start.directions)
            .and_then(|m| m.with_structure(lj.template.clone(), true, false));
    let Ok(mut machine) = machine else {
        return failed(Failure::Steps, gradients, hessians);
    };
    while let SearchAction::Evaluate(points) = machine.ask() {
        let values = lj.values(&points);
        let report = machine.tell(values).unwrap();
        if !bound(&report.point, 2.0) {
            return failed(Failure::Apart, machine.gradient_evaluations(), hessians);
        }
    }
    let end = machine.end();
    gradients = machine.gradient_evaluations();
    hessians += machine.hessian_evaluations();
    let result = match end {
        Some(HisdEnd::Converged | HisdEnd::HandOff | HisdEnd::WrongIndex) => {
            refine(lj, machine.point(), Some(start.order), &mut hessians)
        }
        _ => Err(Failure::Steps),
    };
    Outcome {
        failure: result.as_ref().err().copied(),
        found: result.ok(),
        end,
        gradients,
        hessians,
    }
}

/// `work` on every item, on threads; the results in the order of the items.
pub fn parallel<T: Sync, R: Send>(items: &[T], work: &(dyn Fn(&T) -> R + Sync)) -> Vec<R> {
    let threads = std::thread::available_parallelism()
        .map_or(4, |n| n.get())
        .clamp(1, 16);
    let next = AtomicUsize::new(0);
    let results: Mutex<Vec<Option<R>>> = Mutex::new((0..items.len()).map(|_| None).collect());
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= items.len() {
                    break;
                }
                let r = work(&items[i]);
                results.lock().unwrap()[i] = Some(r);
            });
        }
    });
    results
        .into_inner()
        .unwrap()
        .into_iter()
        .map(|o| o.unwrap())
        .collect()
}

/// The pentagonal bipyramid, near the global minimum.
pub fn bipyramid() -> Col<f64> {
    let ring = 0.936;
    let apex = 0.58;
    let mut x = Col::<f64>::zeros(N);
    for i in 0..5 {
        let t = 2.0 * std::f64::consts::PI * i as f64 / 5.0;
        x[3 * i] = ring * libm::cos(t);
        x[3 * i + 1] = ring * libm::sin(t);
    }
    x[17] = apex;
    x[20] = -apex;
    x
}
