//! Dynamics at a saddle point for the kinetics: pairs of trajectories from the dividing surface
//! (branching fractions and the transmission coefficient of `reactive_flux`,
//! [`crate::kinetics::branching`]) and the friction kernel of the reaction coordinate held at
//! the barrier top (`grote_hynes`).
//!
//! **Pairs.** Every sample starts on the dividing surface through the saddle point orthogonal
//! to its imaginary mode (no displacement along it) with thermal displacements and momenta of
//! the other modes and a thermal momentum along the imaginary one: quasi-classical (the Wigner
//! distribution of [`crate::md::wigner`]) for a structure, classical for plain variables (a
//! model surface). It is run with its velocities and with them reversed, without a thermostat,
//! until each run has stayed in one minimum for `commit` checks (every `check_every` steps) or
//! `max_steps` are reached (then it is undecided). The minimum of a structure is the one with
//! its bond graph (several: the smallest RMSD after alignment, [`StructureMatcher`]); that of
//! plain variables is named by a function of the caller (the basins of a model surface).
//!
//! **Friction kernel.** Dynamics with the imaginary mode held (the mode frozen, [`crate::md`]),
//! from thermal samples of the other modes: the force on the mass-weighted reaction coordinate
//! `F(t) = -l . M^(-1/2) g` and its autocorrelation, `zeta(t) = <dF(0) dF(t)> / k_B T` (the
//! friction kernel per unit mass of Grote and Hynes, J. Chem. Phys. 1980, 73, 2715), averaged
//! over the samples and the time origins.
//!
//! The results go into `run_kinetics` through its annotations ([`annotations`]): `outcomes`,
//! `outcome_errors`, `reactive_flux_kappa`, `reactive_flux_error`, and the kernel as
//! `friction_kernel_file`.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::collections::BTreeMap;

use faer::{Col, Mat};

use crate::core_infra::json::JsonValue;
use crate::core_infra::log::Logger;
use crate::core_infra::rng::Rng;
use crate::domain::{EvalRequest, MolecularSystem, Units};
use crate::error::{MoptError, MoptResult};
use crate::kinetics::branching::{Branching, TrajectoryPair};
use crate::md::freeze::{FreezeSettings, ModeSelection};
use crate::md::wigner::{sample, ImaginaryModes, LowModes, WignerSettings};
use crate::md::{InitialVelocities, MdSettings, MdSpace, MdStepReport, MolecularDynamics};
use crate::pes::calculator::Calculator;
use crate::pes::objective::Objective;
use crate::vib::transverse_modes;
use crate::workflow::explore::identity::{BondGraph, StructureMatcher};
use crate::workflow::md::{run_md, FnMdObserver, MdSurface, MoleculeMdSurface, ObjectiveMdSurface};

/// The settings of the runs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SaddleDynamicsSettings {
    /// Pairs (or samples of the friction kernel).
    pub pairs: usize,
    /// Temperature (K).
    pub temperature: f64,
    /// Time step (fs).
    pub time_step_fs: f64,
    /// The longest run (steps).
    pub max_steps: usize,
    /// Steps between two checks of the minimum.
    pub check_every: usize,
    /// Checks in one minimum before a run is committed.
    pub commit: usize,
    /// Seed of the samples.
    pub seed: u64,
}

impl Default for SaddleDynamicsSettings {
    fn default() -> Self {
        Self {
            pairs: 100,
            temperature: 298.15,
            time_step_fs: 0.5,
            max_steps: 2000,
            check_every: 10,
            commit: 5,
            seed: 42,
        }
    }
}

/// A start on the dividing surface: positions, velocities, and the speed along the imaginary
/// mode (positive: along it).
pub type Start = (Col<f64>, Col<f64>, f64);

/// Run the pairs from the starts of `draw` on `surface`, with `machine(x, v)` making the
/// dynamics of one run and `classify` naming the minimum of a position.
pub fn trajectory_pairs(
    surface: &dyn MdSurface,
    machine: &dyn Fn(Col<f64>, Col<f64>) -> MoptResult<MolecularDynamics>,
    draw: &mut dyn FnMut(&mut Rng) -> MoptResult<Start>,
    classify: &(dyn Fn(&Col<f64>) -> MoptResult<Option<usize>> + Sync),
    s: &SaddleDynamicsSettings,
) -> MoptResult<Vec<TrajectoryPair>> {
    if s.pairs == 0 || s.check_every == 0 || s.commit == 0 {
        return Err(MoptError::config(
            "saddle dynamics: pairs, check_every, and commit must be positive",
        ));
    }
    let mut rng = Rng::from_seed(s.seed);
    let mut out = Vec::with_capacity(s.pairs);
    for _ in 0..s.pairs {
        let (x, v, speed) = draw(&mut rng)?;
        let mut ends = [None, None];
        for (k, sign) in [1.0, -1.0].iter().enumerate() {
            let vk = Col::from_fn(v.nrows(), |i| sign * v[i]);
            let mut md = machine(x.clone(), vk)?;
            let mut last: Option<usize> = None;
            let mut streak = 0usize;
            let mut error: Option<MoptError> = None;
            let mut watch = FnMdObserver::new(|r: &MdStepReport| {
                if r.step == 0 || !r.step.is_multiple_of(s.check_every) {
                    return Ok(true);
                }
                let here = match classify(&r.x) {
                    Ok(h) => h,
                    Err(e) => {
                        error = Some(e);
                        return Ok(false);
                    }
                };
                if here.is_some() && here == last {
                    streak += 1;
                } else {
                    streak = usize::from(here.is_some());
                    last = here;
                }
                Ok(streak < s.commit)
            });
            run_md(&mut md, surface, &mut [&mut watch])?;
            if let Some(e) = error {
                return Err(e);
            }
            ends[k] = if streak >= s.commit { last } else { None };
        }
        // The member that moves along the imaginary mode is the forward run.
        let (forward, backward) = if speed >= 0.0 {
            (ends[0], ends[1])
        } else {
            (ends[1], ends[0])
        };
        out.push(TrajectoryPair {
            speed: speed.abs(),
            forward,
            backward,
        });
    }
    Ok(out)
}

/// The settings of a run without a thermostat from `x`, `v`.
fn nve(s: &SaddleDynamicsSettings, v: Col<f64>) -> MdSettings {
    MdSettings {
        time_step: s.time_step_fs * Units::FS_TO_AU_TIME,
        steps: s.max_steps,
        temperature: s.temperature,
        thermostat: "none".to_string(),
        velocities: InitialVelocities::Given(v),
        seed: s.seed,
        ..MdSettings::default()
    }
}

/// The minima of a structure: by bond graph, then by the RMSD after alignment.
pub struct MinimumClassifier {
    template: MolecularSystem,
    minima: Vec<(MolecularSystem, BondGraph)>,
    matcher: StructureMatcher,
}

impl MinimumClassifier {
    /// The minima `minima` of structures like `template`.
    pub fn new(template: &MolecularSystem, minima: &[MolecularSystem]) -> Self {
        let matcher = StructureMatcher::default();
        Self {
            template: template.clone(),
            minima: minima
                .iter()
                .map(|m| (m.clone(), BondGraph::of(m, matcher.bond_scale)))
                .collect(),
            matcher,
        }
    }

    /// The minimum of the positions `x` (Bohr), if one has its bond graph.
    pub fn classify(&self, x: &Col<f64>) -> MoptResult<Option<usize>> {
        let s = MolecularSystem::from_flat_bohr(
            self.template.elements.clone(),
            x,
            self.template.spin_charge,
            self.template.cell.clone(),
        )?;
        let g = BondGraph::of(&s, self.matcher.bond_scale);
        let mut best: Option<(usize, f64)> = None;
        for (k, (m, gm)) in self.minima.iter().enumerate() {
            if !g.same_as(gm) {
                continue;
            }
            if let Some(found) = self.matcher.best_match_with(&s, &g, m, gm)? {
                let r = found.rmsd.value();
                if best.is_none_or(|(_, b)| r < b) {
                    best = Some((k, r));
                }
            }
        }
        Ok(best.map(|(k, _)| k))
    }
}

/// The pairs of the saddle point `ts` of a molecule on `calc`, classified by `minima`.
pub fn molecule_pairs(
    ts: &MolecularSystem,
    calc: &dyn Calculator,
    minima: &[MolecularSystem],
    s: &SaddleDynamicsSettings,
    log: &Logger,
) -> MoptResult<Vec<TrajectoryPair>> {
    let out = calc.evaluate(ts, &EvalRequest::ENERGY_GRAD_HESS)?;
    let hessian = out
        .hessian_bohr
        .ok_or_else(|| MoptError::calc("the trajectories from a saddle point need its Hessian"))?;
    let modes = transverse_modes(ts, &hessian, true, &[])?;
    let imaginary: Vec<usize> = (0..modes.wavenumbers.len())
        .filter(|&k| modes.wavenumbers[k] < 0.0)
        .collect();
    if imaginary.len() != 1 {
        return Err(MoptError::config(format!(
            "the trajectories need a first-order saddle point; this structure has {} imaginary \
             modes",
            imaginary.len()
        )));
    }
    let d = modes.displacements[imaginary[0]].clone();
    let masses = ts.masses_amu();
    let wigner = WignerSettings {
        temperature: s.temperature,
        low_cutoff: 0.0,
        low_modes: LowModes::Wigner,
        imaginary: ImaginaryModes::Boltzmann,
    };
    let x0 = ts.coords_flat_bohr();
    let mut draw = |rng: &mut Rng| -> MoptResult<Start> {
        let w = sample(&x0, &modes, &wigner, rng)?;
        // P along the imaginary mode: sqrt(amu) sum_i m_i d_i v_i (see md::wigner).
        let p: f64 = (0..x0.nrows())
            .map(|i| masses[i / 3] * d[i] * w.v[i])
            .sum::<f64>()
            * Units::AMU_TO_AU_MASS.sqrt();
        Ok((w.x, w.v, p))
    };
    let surface = MoleculeMdSurface::with_terms(calc, ts.clone(), Vec::new())?;
    let space = MdSpace::atoms(&masses)?;
    let machine = |x: Col<f64>, v: Col<f64>| {
        MolecularDynamics::new(nve(s, v), space.clone(), x, Some(ts.clone()))
    };
    let classifier = MinimumClassifier::new(ts, minima);
    let classify = |x: &Col<f64>| classifier.classify(x);
    log.info(&format!(
        "Trajectories from the saddle point: {} pairs at {} K (Wigner sampling of the transverse \
         modes, a Boltzmann momentum along the imaginary mode of {:.1} cm^-1), at most {} steps \
         of {} fs per run",
        s.pairs, s.temperature, modes.wavenumbers[imaginary[0]], s.max_steps, s.time_step_fs
    ));
    trajectory_pairs(&surface, &machine, &mut draw, &classify, s)
}

/// The mass-weighted modes of the Hessian `hessian` of plain variables of unit mass at `x0`:
/// eigenvalues ascending and the eigenvectors.
fn plain_modes(hessian: &Mat<f64>) -> MoptResult<(Col<f64>, Mat<f64>)> {
    crate::math::linalg::symmetric_eigen(hessian)
}

/// The pairs of the saddle point `x0` of an objective of plain variables (unit masses,
/// classical thermal sampling), the basins named by `classify`.
pub fn objective_pairs(
    objective: &dyn Objective,
    x0: &Col<f64>,
    classify: &(dyn Fn(&Col<f64>) -> MoptResult<Option<usize>> + Sync),
    s: &SaddleDynamicsSettings,
) -> MoptResult<Vec<TrajectoryPair>> {
    let value = objective.evaluate(x0, true)?;
    let h = value.hessian.ok_or_else(|| {
        MoptError::calc("the trajectories from a saddle point need the Hessian of the objective")
    })?;
    let (lambda, u) = plain_modes(&h)?;
    let n = x0.nrows();
    let negative: Vec<usize> = (0..n).filter(|&k| lambda[k] < 0.0).collect();
    if negative.len() != 1 {
        return Err(MoptError::config(format!(
            "the trajectories need a first-order saddle point; the objective has {} negative \
             curvatures there",
            negative.len()
        )));
    }
    let kt = Units::BOLTZMANN_HARTREE_PER_K * s.temperature;
    let mut draw = |rng: &mut Rng| -> MoptResult<Start> {
        let mut x = x0.clone();
        let mut v = Col::<f64>::zeros(n);
        let mut speed = 0.0;
        for k in 0..n {
            let p = kt.sqrt() * rng.standard_normal();
            let q = if lambda[k] > 0.0 {
                (kt / lambda[k]).sqrt() * rng.standard_normal()
            } else {
                0.0
            };
            if k == negative[0] {
                speed = p;
            }
            for i in 0..n {
                x[i] += u[(i, k)] * q;
                v[i] += u[(i, k)] * p;
            }
        }
        Ok((x, v, speed))
    };
    let surface = ObjectiveMdSurface(objective);
    let space = MdSpace::variables(n)?;
    let machine =
        |x: Col<f64>, v: Col<f64>| MolecularDynamics::new(nve(s, v), space.clone(), x, None);
    trajectory_pairs(&surface, &machine, &mut draw, classify, s)
}

/// The friction kernel `zeta(t)` (times in s, values in 1/s^2) of the force on the
/// mass-weighted reaction coordinate of the saddle point `ts` on `calc`, held at the barrier
/// top, from `s.pairs` samples of `s.max_steps` steps each (correlation times up to
/// `max_lag` steps).
pub fn friction_kernel(
    ts: &MolecularSystem,
    calc: &dyn Calculator,
    s: &SaddleDynamicsSettings,
    max_lag: usize,
) -> MoptResult<(Vec<f64>, Vec<f64>)> {
    let out = calc.evaluate(ts, &EvalRequest::ENERGY_GRAD_HESS)?;
    let hessian = out.hessian_bohr.ok_or_else(|| {
        MoptError::calc("the friction kernel needs the Hessian at the saddle point")
    })?;
    let modes = transverse_modes(ts, &hessian, true, &[])?;
    let Some(k) = (0..modes.wavenumbers.len()).find(|&k| modes.wavenumbers[k] < 0.0) else {
        return Err(MoptError::config(
            "the friction kernel needs a saddle point (an imaginary mode)",
        ));
    };
    let d = modes.displacements[k].clone();
    let masses = ts.masses_amu();
    let n = d.nrows();
    // The unit mass-weighted vector l = sqrt(m) d (m in amu), and the force on q = l . sqrt(M) x
    // in atomic units: F = -l . M^(-1/2) g with M in electron masses.
    let au = Units::AMU_TO_AU_MASS;
    let force = |g: &Col<f64>| -> f64 {
        -(0..n)
            .map(|i| masses[i / 3].sqrt() * d[i] * g[i] / (masses[i / 3] * au).sqrt())
            .sum::<f64>()
    };
    let wigner = WignerSettings {
        temperature: s.temperature,
        low_cutoff: 0.0,
        low_modes: LowModes::Classical,
        imaginary: ImaginaryModes::Zero,
    };
    let surface = MoleculeMdSurface::with_terms(calc, ts.clone(), Vec::new())?;
    let space = MdSpace::atoms(&masses)?;
    let x0 = ts.coords_flat_bohr();
    let mut rng = Rng::from_seed(s.seed);
    let lag = max_lag.max(1).min(s.max_steps.max(1));
    let mut sums = vec![0.0; lag + 1];
    let mut counts = vec![0usize; lag + 1];
    for _ in 0..s.pairs.max(1) {
        let w = sample(&x0, &modes, &wigner, &mut rng)?;
        let settings = MdSettings {
            freeze: Some(FreezeSettings {
                selection: ModeSelection::Vectors(vec![d.clone()]),
                update_interval: None,
            }),
            ..nve(s, w.v.clone())
        };
        let mut md = MolecularDynamics::new(settings, space.clone(), w.x, Some(ts.clone()))?;
        let mut series = Vec::with_capacity(s.max_steps + 1);
        let mut watch = FnMdObserver::new(|r: &MdStepReport| {
            series.push(force(&r.gradient));
            Ok(true)
        });
        run_md(&mut md, &surface, &mut [&mut watch])?;
        let mean = series.iter().sum::<f64>() / series.len().max(1) as f64;
        let dev: Vec<f64> = series.iter().map(|f| f - mean).collect();
        for tau in 0..=lag.min(dev.len().saturating_sub(1)) {
            for t0 in 0..dev.len() - tau {
                sums[tau] += dev[t0] * dev[t0 + tau];
                counts[tau] += 1;
            }
        }
    }
    let kt = Units::BOLTZMANN_HARTREE_PER_K * s.temperature;
    let dt = s.time_step_fs * Units::FS_TO_AU_TIME;
    let to_s2 = 1.0 / (Units::AU_TIME_TO_SEC * Units::AU_TIME_TO_SEC);
    let mut times = Vec::new();
    let mut values = Vec::new();
    for tau in 0..=lag {
        if counts[tau] == 0 {
            break;
        }
        times.push(tau as f64 * dt * Units::AU_TIME_TO_SEC);
        values.push(sums[tau] / counts[tau] as f64 / kt * to_s2);
    }
    Ok((times, values))
}

/// The annotations of a saddle point from its trajectories, for `run_kinetics --annotations`:
/// the transmission coefficient and the branching fractions from the state the most flux comes
/// from, with their errors. `labels` names the states by the indices of the classification.
pub fn annotations(edge: &str, b: &Branching, labels: &[String]) -> MoptResult<JsonValue> {
    let name = |k: usize| {
        labels
            .get(k)
            .cloned()
            .ok_or_else(|| MoptError::config(format!("no label for the state {k}")))
    };
    let mut entry = BTreeMap::new();
    entry.insert(
        "reactive_flux_kappa".to_string(),
        JsonValue::Number(b.kappa),
    );
    entry.insert(
        "reactive_flux_error".to_string(),
        JsonValue::Number(if b.kappa_error.is_finite() {
            b.kappa_error
        } else {
            0.0
        }),
    );
    // The state the most flux comes from (the outcomes from other states are trajectories
    // that recross between two products, which a step of the network does not hold).
    let mut by_from: BTreeMap<usize, f64> = BTreeMap::new();
    for o in &b.outcomes {
        *by_from.entry(o.from).or_default() += o.fraction;
    }
    if let Some((&first, _)) = by_from.iter().max_by(|a, b| a.1.total_cmp(b.1)) {
        let mut to = BTreeMap::new();
        let mut errors = BTreeMap::new();
        for o in b.outcomes.iter().filter(|o| o.from == first) {
            to.insert(name(o.to)?, JsonValue::Number(o.fraction));
            errors.insert(
                name(o.to)?,
                JsonValue::Number(if o.error.is_finite() { o.error } else { 0.0 }),
            );
        }
        let mut outcomes = BTreeMap::new();
        outcomes.insert("from".to_string(), JsonValue::String(name(first)?));
        outcomes.insert("to".to_string(), JsonValue::Object(to));
        entry.insert("outcomes".to_string(), JsonValue::Object(outcomes));
        entry.insert("outcome_errors".to_string(), JsonValue::Object(errors));
    }
    let mut edges = BTreeMap::new();
    edges.insert(edge.to_string(), JsonValue::Object(entry));
    let mut doc = BTreeMap::new();
    doc.insert("edges".to_string(), JsonValue::Object(edges));
    Ok(JsonValue::Object(doc))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::kinetics::branching::{branching, Weighting};
    use crate::pes::objective::ObjectiveValue;

    /// `V = a (x^2 - 1)^2 ... ` a saddle at the origin whose product valley at x > 0 splits:
    /// `V = -b x^2/2 + c x^4/4 + (k0 - k1 x) y^2 / 2 + d y^4 / 4 + e y`, with `k` changing sign at
    /// `x = k0 / k1` (the valley-ridge inflection), `e` tilting the split.
    struct Bifurcation {
        tilt: f64,
    }

    impl Objective for Bifurcation {
        fn name(&self) -> &str {
            "bifurcation"
        }
        fn dimension(&self) -> Option<usize> {
            Some(2)
        }
        fn evaluate(&self, q: &Col<f64>, need_hessian: bool) -> MoptResult<ObjectiveValue> {
            let (b, c, k0, k1, d, e) = (0.02, 0.02, 0.01, 0.03, 0.02, self.tilt);
            let (x, y) = (q[0], q[1]);
            let k = k0 - k1 * x;
            let value = -b * x * x / 2.0
                + c * x.powi(4) / 4.0
                + k * y * y / 2.0
                + d * y.powi(4) / 4.0
                + e * y * x.max(0.0).powi(2);
            let gx = -b * x + c * x.powi(3) - k1 * y * y / 2.0 + 2.0 * e * y * x.max(0.0);
            let gy = k * y + d * y.powi(3) + e * x.max(0.0).powi(2);
            let hessian = need_hessian.then(|| {
                Mat::from_fn(2, 2, |i, j| match (i, j) {
                    (0, 0) => -b + 3.0 * c * x * x + 2.0 * e * y * if x > 0.0 { 1.0 } else { 0.0 },
                    (1, 1) => k + 3.0 * d * y * y,
                    _ => -k1 * y + 2.0 * e * x.max(0.0),
                })
            });
            Ok(ObjectiveValue {
                value,
                gradient: Col::from_fn(2, |i| if i == 0 { gx } else { gy }),
                hessian,
            })
        }
    }

    #[test]
    fn a_symmetric_bifurcation_splits_evenly() {
        // The basins: the reactant at x < 0, the two products past the valley-ridge inflection
        // (x = k0 / k1 = 1/3) on either side of the ridge y = 0.
        let classify = |q: &Col<f64>| -> MoptResult<Option<usize>> {
            Ok(if q[0] < -0.3 {
                Some(0)
            } else if q[0] > 0.6 && q[1] > 0.3 {
                Some(1)
            } else if q[0] > 0.6 && q[1] < -0.3 {
                Some(2)
            } else {
                None
            })
        };
        let s = SaddleDynamicsSettings {
            pairs: 300,
            temperature: 300.0,
            time_step_fs: 0.02,
            max_steps: 20_000,
            check_every: 5,
            commit: 3,
            seed: 3,
        };
        let zero = Col::<f64>::zeros(2);
        let pairs = objective_pairs(&Bifurcation { tilt: 0.0 }, &zero, &classify, &s).unwrap();
        let b = branching(&pairs, Weighting::Count).unwrap();
        assert!(b.undecided < 15, "{b:?}");
        let share = |b: &Branching, to: usize| {
            b.outcomes
                .iter()
                .find(|o| o.from == 0 && o.to == to)
                .map_or(0.0, |o| o.fraction)
        };
        let (f1, f2) = (share(&b, 1), share(&b, 2));
        let n = (pairs.len() - b.undecided - b.recrossing) as f64;
        let se = (0.25 / n).sqrt();
        let half = f1 / (f1 + f2);
        assert!((half - 0.5).abs() < 3.0 * se, "{f1} {f2} ({se}) {b:?}");
        // The flux goes from the reactant to the products; trajectories that recross into the
        // other product valley carry next to nothing.
        let main: f64 = b
            .outcomes
            .iter()
            .filter(|o| o.from == 0)
            .map(|o| o.fraction)
            .sum();
        assert!(main > 0.95, "{b:?}");
        assert!(b.kappa > 0.3 && b.kappa <= 1.0 + 1e-12, "{}", b.kappa);
        let doc = annotations("TS0", &b, &["EQ0".into(), "EQ1".into(), "EQ2".into()]).unwrap();
        assert!(doc.to_string_pretty().contains("outcomes"));
        // A tilt of the product valleys favours one side.
        let tilted = objective_pairs(&Bifurcation { tilt: 0.004 }, &zero, &classify, &s).unwrap();
        let bt = branching(&tilted, Weighting::Count).unwrap();
        let g2 = share(&bt, 2) / (share(&bt, 1) + share(&bt, 2));
        assert!(g2 > 0.5 + 3.0 * se, "{g2} {bt:?}");
    }
}
