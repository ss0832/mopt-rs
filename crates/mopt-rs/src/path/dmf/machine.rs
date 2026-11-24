//! The direct MaxFlux method as a state machine: it asks for energies and gradients at a
//! few points of a spline path and moves the path until the objective is stationary.
//!
//! One iteration: the energies and gradients at the points, which sit at given shares of
//! the length of the path ([`DmfAction::Evaluate`]; the two ends are asked for once), the
//! objective and its gradient ([`super::action`]), a limited-memory quasi-Newton step,
//! and the points moved towards the top of the energy profile ([`super::profile`]). The
//! objective does not change when the path is run through at another pace, so the
//! control points move across the path only: their gradients and steps are taken without
//! the part along the tangent of the path where each acts most (its Greville abscissa),
//! as the images of a nudged elastic band. When
//! the parameter of the spline runs very unevenly along the path, the spline is fitted
//! anew to the same curve at even pace (and the quasi-Newton model starts again).
//!
//! A step is taken back when the top of the energy profile (the estimate of the
//! transition state, which sees between the points through the slopes of the energy)
//! rises above the lowest one so far by more than a share of the barrier: the path goes
//! back to that one and the step is halved, until the top comes lower. The sum over a
//! few points can be lowered by a path that crosses the ridge between two of them, which
//! lengthens elsewhere so that the points slide off the top; at large `beta` the method
//! minimizes the highest energy of the path, and a rising top is the sign of that (HCN to
//! HNC with GFN0-xTB: the top came to the saddle in six iterations, then climbed 40
//! kcal/mol in the next eighty with every step accepted). From the path with the lowest
//! top the gradient can point that way again, so the run also ends when the top has not
//! come down by 0.1 % of the barrier in eight iterations ([`DmfEnd::Settled`]), with that
//! path.
//!
//! PyDMF hands the problem (objective, bounds on the pace, bounds on rigid motion) to the
//! interior-point program IPOPT. What is taken from it: the limited-memory model starts
//! from a constant curvature (`2.5` for the objective scaled by `0.1`, in eV and
//! Angstrom: 25 eV/Angstrom^2), the run ends when the largest
//! component of the gradient is below a tolerance (IPOPT's dual infeasibility; `0.04`
//! eV/Angstrom for its `tight`), and the points are moved with the same gates on that
//! number. What is not: there are no bounds (see [`super::action`]), and none on rigid
//! motion (the ends are laid over each other before, and a path that turns is longer,
//! which the objective counts).
//!
//! MultiOptPy: `MEP/pathopt_dmf_force.py:CaluculationDMF.calc_force` fits a spline through
//! the images of a chain in every iteration and hands the gradient of the objective back
//! as forces on the images, with points at equal steps and no condition on the pace.
//! That is not ported; this is the method as its authors define it.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Col;

use super::action::{action, trapezoid_weights};
use super::arc::ArcLength;
use super::profile::EnergyProfile;
use super::spline::SplinePath;
use crate::domain::Units;
use crate::error::{MoptError, MoptResult};
use crate::math::linalg::col_dot;
use crate::math::stable::logistic;

/// eV/Angstrom in Hartree/Bohr.
const EV_PER_ANGSTROM: f64 = Units::EV_TO_HARTREE * Units::BOHR_TO_ANGSTROM;

/// How the points are moved towards the top of the energy profile (PyDMF's
/// `params_t_update`; forces in eV/Angstrom, energies in eV there).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PointMoves {
    /// The largest share of the way to its place that a point goes in one iteration.
    pub largest_share: f64,
    /// The largest spacing of the energies of the points below the top (Hartree).
    pub spacing: f64,
    /// The points start to move when the gradient that is left falls to this
    /// (Hartree/Bohr) ...
    pub start: f64,
    /// ... over a range of about one over this (Bohr/Hartree).
    pub start_sharpness: f64,
    /// The moves get smaller when the gradient that is left falls to this (Hartree/Bohr)
    pub settle: f64,
    /// ... over a range of about one over this (Bohr/Hartree) ...
    pub settle_sharpness: f64,
    /// ... by this share in every iteration.
    pub settle_rate: f64,
}

impl Default for PointMoves {
    fn default() -> Self {
        Self {
            largest_share: 0.1,
            spacing: 0.15 * Units::EV_TO_HARTREE,
            start: 1.0 * EV_PER_ANGSTROM,
            start_sharpness: 5.0 / EV_PER_ANGSTROM,
            settle: 0.2 * EV_PER_ANGSTROM,
            settle_sharpness: 5.0 / EV_PER_ANGSTROM,
            settle_rate: 0.02,
        }
    }
}

/// Settings of the direct MaxFlux method.
#[derive(Debug, Clone, PartialEq)]
pub struct DmfSettings {
    /// `beta` of the objective (1/Hartree); PyDMF: 10 per eV.
    pub beta: f64,
    /// Degree of the B-splines.
    pub degree: usize,
    /// Number of segments of the spline.
    pub segments: usize,
    /// Number of points between the ends that move towards the top of the energy
    /// profile.
    pub points: usize,
    /// Number of points between each end and those that move, at equal shares of the
    /// length there: they keep the rest of the path in view (none in PyDMF).
    pub guards: usize,
    /// How the points are moved towards the top of the energy profile; `None`: they stay
    /// at equal shares of the length.
    pub moves: Option<PointMoves>,
    /// The spline is fitted anew when its largest speed is this many times its smallest.
    pub refit_unevenness: f64,
    /// The run ends when the largest component of the gradient that is left is below
    /// this (Hartree/Bohr).
    pub tolerance: f64,
    /// Largest number of iterations.
    pub max_iterations: usize,
    /// Largest change of a coordinate of a control point in one iteration (Bohr).
    pub max_step: f64,
    /// Curvature the quasi-Newton model starts from (Hartree/Bohr^2).
    pub first_curvature: f64,
    /// Number of steps the quasi-Newton model remembers.
    pub memory: usize,
    /// A step is taken back when the top of the path rises above the lowest one so far
    /// by more than this share of the barrier.
    pub rise: f64,
    /// The run ends when the top has not come down by this share of the barrier ...
    pub settle: f64,
    /// ... in this many iterations.
    pub patience: usize,
    /// Variables that do not move (coordinates of atoms held in place).
    pub fixed: Vec<usize>,
}

impl Default for DmfSettings {
    fn default() -> Self {
        Self {
            beta: 10.0 * Units::HARTREE_TO_EV,
            degree: 3,
            segments: 4,
            points: 3,
            guards: 1,
            moves: Some(PointMoves::default()),
            refit_unevenness: 3.0,
            tolerance: 0.04 * EV_PER_ANGSTROM,
            max_iterations: 200,
            max_step: 0.3,
            first_curvature: 25.0 * EV_PER_ANGSTROM / Units::ANGSTROM_TO_BOHR,
            memory: 6,
            rise: 0.01,
            settle: 1e-3,
            patience: 8,
            fixed: Vec::new(),
        }
    }
}

/// A point to evaluate.
#[derive(Debug, Clone)]
pub struct DmfPoint {
    /// Its number among the points of the path (0: the first end).
    pub index: usize,
    /// Its share of the length of the path.
    pub t: f64,
    /// The point.
    pub x: Col<f64>,
}

/// How a run ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DmfEnd {
    /// The gradient that is left is below the tolerance.
    Converged,
    /// The largest number of iterations was reached.
    IterationLimit,
    /// The top of the path no longer came down (by [`DmfSettings::settle`] of the
    /// barrier in [`DmfSettings::patience`] iterations); the run ends with the lowest one.
    Settled,
}

/// What the machine wants next.
#[derive(Debug, Clone)]
pub enum DmfAction {
    /// Energies and gradients at these points, answered in this order with
    /// [`DirectMaxFlux::tell`].
    Evaluate(Vec<DmfPoint>),
    /// The run is over.
    Finished(DmfEnd),
}

/// An iteration.
#[derive(Debug, Clone)]
pub struct DmfReport {
    /// Number of the iteration (from 0).
    pub iteration: usize,
    /// The objective `ln(I) / beta` plus the lower energy of the two ends (Hartree).
    pub objective: f64,
    /// Largest component of the gradient of the objective that changes the path and not
    /// its pace (Hartree/Bohr).
    pub largest_gradient: f64,
    /// The points as shares of the length of the path, the ends among them.
    pub t: Vec<f64>,
    /// Where they are.
    pub points: Vec<Col<f64>>,
    /// The energies at the points (Hartree).
    pub energies: Vec<f64>,
    /// The energy along the path through the points, relative to `reference`.
    pub profile: EnergyProfile,
    /// The lower energy of the two ends (Hartree).
    pub reference: f64,
    /// Where the energy profile is highest (share of the length).
    pub t_max: f64,
    /// The energy there, from the profile (Hartree).
    pub e_max: f64,
    /// The point of the path there: the estimate of the transition state.
    pub highest: Col<f64>,
    /// The length of the path.
    pub length: f64,
    /// How unevenly the parameter of the spline runs (largest speed over smallest).
    pub unevenness: f64,
    /// Whether the spline was fitted anew after the step of this iteration.
    pub refitted: bool,
    /// Whether the step that led here raised the top of the path and was taken back.
    pub taken_back: bool,
}

/// The path with the lowest top so far.
#[derive(Debug, Clone)]
struct Best {
    path: SplinePath,
    t: Vec<f64>,
    gradient: Vec<Col<f64>>,
    report: DmfReport,
}

/// The direct MaxFlux method.
#[derive(Debug, Clone)]
pub struct DirectMaxFlux {
    settings: DmfSettings,
    path: SplinePath,
    /// The points as shares of the length, the ends among them.
    t: Vec<f64>,
    /// Their parameters on the spline as it was asked.
    parameters: Vec<f64>,
    /// Energies and gradients of the two ends.
    ends: Option<[(f64, Col<f64>); 2]>,
    /// The points asked for.
    asked: Vec<usize>,
    iteration: usize,
    /// Control points and projected gradient of the iteration before.
    before: Option<(Vec<Col<f64>>, Vec<Col<f64>>)>,
    /// Steps and changes of the projected gradient, oldest first.
    pairs: Vec<(Vec<Col<f64>>, Vec<Col<f64>>)>,
    move_share: f64,
    end: Option<DmfEnd>,
    report: Option<DmfReport>,
    best: Option<Best>,
    /// Iterations since the top came down by `settle` of the barrier.
    since_lower: usize,
    /// Share of the quasi-Newton step that is taken (halved after a step is taken back).
    step_share: f64,
}

fn dot(a: &[Col<f64>], b: &[Col<f64>]) -> f64 {
    a.iter().zip(b).map(|(x, y)| col_dot(x, y)).sum()
}

/// `vectors` (one per inner control point) without their parts along `tangents`.
fn across(vectors: &[Col<f64>], tangents: &[Col<f64>]) -> Vec<Col<f64>> {
    vectors
        .iter()
        .zip(tangents)
        .map(|(v, t)| v - t * col_dot(v, t))
        .collect()
}

fn largest(a: &[Col<f64>]) -> f64 {
    a.iter()
        .flat_map(|x| x.iter())
        .fold(0.0_f64, |m, v| m.max(v.abs()))
}

impl DirectMaxFlux {
    /// The method for the path through the points `chain` (the ends of the path are its
    /// first and its last point).
    pub fn new(settings: DmfSettings, chain: &[Col<f64>]) -> MoptResult<Self> {
        if settings.points == 0 {
            return Err(MoptError::config(
                "the direct MaxFlux method needs one point between the ends at least",
            ));
        }
        if settings.beta.is_nan() || settings.beta <= 0.0 {
            return Err(MoptError::config(
                "the direct MaxFlux method needs a positive beta",
            ));
        }
        if settings.first_curvature.is_nan() || settings.first_curvature <= 0.0 {
            return Err(MoptError::config(
                "the direct MaxFlux method needs a positive first curvature",
            ));
        }
        let path = SplinePath::through(chain, settings.degree, settings.segments)?;
        ArcLength::new(&path)?;
        let n = path.dimension();
        if let Some(&bad) = settings.fixed.iter().find(|&&k| k >= n) {
            return Err(MoptError::config(format!(
                "fixed variable {bad} of a path with {n} variables"
            )));
        }
        let count = settings.points + 2 * settings.guards + 2;
        let t: Vec<f64> = (0..count).map(|k| k as f64 / (count - 1) as f64).collect();
        let move_share = settings.moves.map_or(0.0, |m| m.largest_share);
        Ok(Self {
            settings,
            path,
            t,
            parameters: Vec::new(),
            ends: None,
            asked: Vec::new(),
            iteration: 0,
            before: None,
            pairs: Vec::new(),
            move_share,
            end: None,
            report: None,
            best: None,
            since_lower: 0,
            step_share: 1.0,
        })
    }

    /// The settings.
    pub fn settings(&self) -> &DmfSettings {
        &self.settings
    }

    /// The path as it is now.
    pub fn path(&self) -> &SplinePath {
        &self.path
    }

    /// How the run ended, or `None` while it runs.
    pub fn end(&self) -> Option<DmfEnd> {
        self.end
    }

    /// The iteration that was completed last.
    pub fn report(&self) -> Option<&DmfReport> {
        self.report.as_ref()
    }

    /// What the run leaves: the last iteration when it converged, else the one with the
    /// lowest top of the path.
    pub fn result(&self) -> Option<&DmfReport> {
        match (self.end, &self.best) {
            (Some(DmfEnd::Converged), _) | (_, None) => self.report.as_ref(),
            (_, Some(best)) => Some(&best.report),
        }
    }

    /// What is to be done next.
    pub fn ask(&mut self) -> MoptResult<DmfAction> {
        if let Some(end) = self.end {
            return Ok(DmfAction::Finished(end));
        }
        if !self.asked.is_empty() {
            return Err(MoptError::config(
                "the direct MaxFlux method was asked again before it was told the points",
            ));
        }
        let last = self.t.len() - 1;
        let arc = ArcLength::new(&self.path)?;
        self.parameters = self.t.iter().map(|&share| arc.parameter(share)).collect();
        self.asked = if self.ends.is_some() {
            (1..last).collect()
        } else {
            (0..=last).collect()
        };
        Ok(DmfAction::Evaluate(
            self.asked
                .iter()
                .map(|&index| DmfPoint {
                    index,
                    t: self.t[index],
                    x: self.path.at(self.parameters[index]),
                })
                .collect(),
        ))
    }

    /// The energies (Hartree) and gradients at the points asked for, in their order.
    pub fn tell(&mut self, values: Vec<(f64, Col<f64>)>) -> MoptResult<()> {
        if self.asked.is_empty() || values.len() != self.asked.len() {
            return Err(MoptError::config(format!(
                "the direct MaxFlux method asked for {} point(s) and was told {}",
                self.asked.len(),
                values.len()
            )));
        }
        let n = self.path.dimension();
        if values
            .iter()
            .any(|(energy, gradient)| !energy.is_finite() || gradient.nrows() != n)
        {
            return Err(MoptError::calc(
                "the direct MaxFlux method was told an energy that is not finite or a gradient \
                 of another length",
            ));
        }
        let mut values = values;
        for (_, gradient) in &mut values {
            for &k in &self.settings.fixed {
                gradient[k] = 0.0;
            }
        }
        let last = self.t.len() - 1;
        if self.ends.is_none() {
            let end = values.pop();
            let start = (!values.is_empty()).then(|| values.remove(0));
            match (start, end) {
                (Some(start), Some(end)) => self.ends = Some([start, end]),
                _ => return Err(MoptError::config("the ends of the path were not told")),
            }
        }
        self.asked.clear();
        let Some(ends) = &self.ends else {
            return Err(MoptError::config("the ends of the path were not told"));
        };
        // Energies from the lower end, so that the numbers of the objective are small.
        let reference = ends[0].0.min(ends[1].0);
        let mut energies = Vec::with_capacity(last + 1);
        let mut gradients = Vec::with_capacity(last + 1);
        energies.push(ends[0].0 - reference);
        gradients.push(ends[0].1.clone());
        for (energy, gradient) in values {
            energies.push(energy - reference);
            gradients.push(gradient);
        }
        energies.push(ends[1].0 - reference);
        gradients.push(ends[1].1.clone());

        let weights = trapezoid_weights(&self.t);
        let arc = ArcLength::new(&self.path)?;
        let found = action(
            &self.path,
            &arc,
            &self.t,
            &self.parameters,
            &weights,
            &energies,
            &gradients,
            self.settings.beta,
        )?;
        let unevenness = arc.unevenness();
        let tangents = self.tangents();
        let gradient = across(&found.gradient, &tangents);
        let largest_gradient = largest(&gradient);
        let profile = EnergyProfile::new(&self.t, &energies, &found.slopes)?;
        self.report = Some(DmfReport {
            iteration: self.iteration,
            objective: found.value + reference,
            largest_gradient,
            t: self.t.clone(),
            points: self.parameters.iter().map(|&t| self.path.at(t)).collect(),
            energies: energies.iter().map(|e| e + reference).collect(),
            profile: profile.clone(),
            reference,
            t_max: profile.t_max,
            e_max: profile.e_max + reference,
            highest: self.path.at(arc.parameter(profile.t_max)),
            length: found.length,
            unevenness,
            refitted: false,
            taken_back: false,
        });
        // A top that rose: back to the lowest one, with a shorter step.
        let barrier = (profile.e_max - energies[0].max(energies[last])).max(0.0);
        let rose = self.best.as_ref().is_some_and(|best| {
            profile.e_max + reference > best.report.e_max + self.settings.rise * barrier
        });
        let lower = self.best.as_ref().is_none_or(|best| {
            profile.e_max + reference < best.report.e_max - self.settings.settle * barrier
        });
        self.since_lower = if lower { 0 } else { self.since_lower + 1 };
        if !rose && self.best.is_some() && self.since_lower >= self.settings.patience {
            self.end = Some(DmfEnd::Settled);
            return Ok(());
        }
        if let (true, Some(best)) = (rose, self.best.clone()) {
            if let Some(report) = self.report.as_mut() {
                report.taken_back = true;
            }
            if self.since_lower >= self.settings.patience {
                self.end = Some(DmfEnd::Settled);
                return Ok(());
            }
            if self.iteration + 1 >= self.settings.max_iterations {
                self.end = Some(DmfEnd::IterationLimit);
                return Ok(());
            }
            self.iteration += 1;
            self.path = best.path;
            self.t = best.t;
            self.pairs.clear();
            self.before = None;
            self.step_share *= 0.5;
            let tangents = self.tangents();
            return self.step(best.gradient, &tangents, None);
        }
        if lower {
            self.step_share = (2.0 * self.step_share).min(1.0);
        }
        if self
            .best
            .as_ref()
            .is_none_or(|best| profile.e_max + reference < best.report.e_max)
        {
            if let Some(report) = &self.report {
                self.best = Some(Best {
                    path: self.path.clone(),
                    t: self.t.clone(),
                    gradient: gradient.clone(),
                    report: report.clone(),
                });
            }
        }
        if largest_gradient < self.settings.tolerance {
            self.end = Some(DmfEnd::Converged);
            return Ok(());
        }
        if self.iteration + 1 >= self.settings.max_iterations {
            self.end = Some(DmfEnd::IterationLimit);
            return Ok(());
        }
        self.iteration += 1;
        self.step(
            gradient,
            &tangents,
            Some((&profile, &energies, largest_gradient)),
        )
    }

    /// The quasi-Newton step from the path as it is with its gradient `gradient`, then
    /// the points moved towards the top of `moves` (profile, energies, largest gradient).
    fn step(
        &mut self,
        gradient: Vec<Col<f64>>,
        tangents: &[Col<f64>],
        moves: Option<(&EnergyProfile, &[f64], f64)>,
    ) -> MoptResult<()> {
        let inner: Vec<Col<f64>> = self.path.control[1..self.path.control.len() - 1].to_vec();
        if let Some((points, old)) = &self.before {
            let step: Vec<Col<f64>> = inner.iter().zip(points).map(|(a, b)| a - b).collect();
            let change: Vec<Col<f64>> = gradient.iter().zip(old).map(|(a, b)| a - b).collect();
            let curvature = dot(&step, &change);
            if curvature > 1e-10 * (dot(&step, &step) * dot(&change, &change)).sqrt() {
                self.pairs.push((step, change));
                if self.pairs.len() > self.settings.memory {
                    self.pairs.remove(0);
                }
            }
        }
        let mut step = across(&self.quasi_newton(&gradient), tangents);
        if dot(&step, &gradient) >= 0.0 {
            // Not downhill: the model is of no use here.
            self.pairs.clear();
            step = gradient
                .iter()
                .map(|g| g * (-1.0 / self.settings.first_curvature))
                .collect();
        }
        let size = largest(&step) * self.step_share;
        let scale = if size > self.settings.max_step {
            self.settings.max_step / size
        } else {
            1.0
        } * self.step_share;
        for part in &mut step {
            *part *= scale;
        }
        self.before = Some((inner, gradient));
        for (control, change) in self.path.control[1..].iter_mut().zip(&step) {
            *control += change;
        }
        self.even_pace()?;
        if let Some((profile, energies, largest_gradient)) = moves {
            self.move_points(profile, energies, largest_gradient)?;
        }
        Ok(())
    }

    /// The unit tangents of the path at the Greville abscissae of the inner control
    /// points.
    fn tangents(&self) -> Vec<Col<f64>> {
        let at = self.path.basis.greville();
        at[1..at.len() - 1]
            .iter()
            .map(|&t| {
                let velocity = self.path.velocity(t);
                let speed = col_dot(&velocity, &velocity).sqrt();
                if speed > 0.0 {
                    velocity * (1.0 / speed)
                } else {
                    velocity
                }
            })
            .collect()
    }

    /// Fits the spline anew to its own curve at even pace when its parameter runs too
    /// unevenly; the quasi-Newton model starts again then.
    fn even_pace(&mut self) -> MoptResult<()> {
        let arc = ArcLength::new(&self.path)?;
        if arc.unevenness() <= self.settings.refit_unevenness {
            return Ok(());
        }
        let samples = 16 * self.settings.segments + 1;
        let curve: Vec<Col<f64>> = (0..samples)
            .map(|k| self.path.at(arc.parameter(k as f64 / (samples - 1) as f64)))
            .collect();
        self.path = SplinePath::through(&curve, self.settings.degree, self.settings.segments)?;
        self.before = None;
        self.pairs.clear();
        if let Some(report) = self.report.as_mut() {
            report.refitted = true;
        }
        Ok(())
    }

    /// `-H g` with the limited-memory inverse `H` (two loops over the remembered steps).
    fn quasi_newton(&self, gradient: &[Col<f64>]) -> Vec<Col<f64>> {
        let mut q: Vec<Col<f64>> = gradient.to_vec();
        let mut factors = Vec::with_capacity(self.pairs.len());
        for (s, y) in self.pairs.iter().rev() {
            let rho = 1.0 / dot(s, y);
            let a = rho * dot(s, &q);
            for (part, of_y) in q.iter_mut().zip(y) {
                *part -= of_y * a;
            }
            factors.push((rho, a));
        }
        for part in &mut q {
            *part *= 1.0 / self.settings.first_curvature;
        }
        for ((s, y), (rho, a)) in self.pairs.iter().zip(factors.iter().rev()) {
            let b = rho * dot(y, &q);
            for (part, of_s) in q.iter_mut().zip(s) {
                *part += of_s * (a - b);
            }
        }
        for part in &mut q {
            *part *= -1.0;
        }
        q
    }

    /// Moves the points between the ends towards the top of the energy profile.
    fn move_points(
        &mut self,
        profile: &EnergyProfile,
        energies: &[f64],
        largest_gradient: f64,
    ) -> MoptResult<()> {
        let Some(moves) = self.settings.moves else {
            return Ok(());
        };
        let last = self.t.len() - 1;
        // 0.5 (1 + tanh(z)) is the logistic function of 2 z.
        let go = logistic(-4.0 * moves.start_sharpness * (largest_gradient - moves.start));
        let settle = 1.0
            - moves.settle_rate
                * logistic(-4.0 * moves.settle_sharpness * (largest_gradient - moves.settle));
        let barrier = profile.e_max - energies[0].max(energies[last]);
        if barrier > 0.0 {
            let count = self.settings.points;
            let guards = self.settings.guards;
            let spacing = (2.0 / (count + 1) as f64 * barrier).min(moves.spacing);
            let share = go * self.move_share;
            let mut moved = self.t.clone();
            for (k, target) in profile.targets(count, spacing)?.into_iter().enumerate() {
                if let Some(target) = target {
                    let at = guards + 1 + k;
                    moved[at] = (1.0 - share) * moved[at] + share * target;
                }
            }
            // The guards at equal shares between the ends and the points that moved.
            let (first, last_moved) = (moved[guards + 1], moved[guards + count]);
            for k in 1..=guards {
                let share = k as f64 / (guards + 1) as f64;
                moved[k] = first * share;
                moved[guards + count + k] = last_moved + (1.0 - last_moved) * share;
            }
            // Points that would meet stay where they are.
            if moved.windows(2).all(|pair| pair[1] - pair[0] > 1e-3) {
                self.t = moved;
            }
        }
        self.move_share *= settle;
        Ok(())
    }
}
