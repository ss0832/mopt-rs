//! Binary-image transition state search (BITSS).
//!
//! Two images, one in each of two basins, are minimized together on
//!
//! `E_BITSS(x1, x2) = E1 + E2 + kappa_e (E1 - E2)^2 + kappa_d (d - d_i)^2`, `d = |x1 - x2|`,
//!
//! while the separation asked for is reduced from stage to stage, `d_i = (1 - f) d_(i-1)`,
//! `d_0` being the separation at the start. The energy term keeps either image from being
//! pulled over the ridge between the basins, so the pair closes in on the lowest point of
//! that ridge: a transition state, with the images on either side of it along the direction
//! of negative curvature.
//!
//! S. J. Avis, J. R. Panter, H. Kusumaatmaja, J. Chem. Phys. 2022, 157, 124107
//! (arXiv:2209.02541), Eqs. (1) to (6):
//!
//! - `kappa_e = alpha / (2 E_B)`;
//! - `kappa_d = max( sqrt(|g1|^2 + |g2|^2) / (2 sqrt(2) beta d_i), E_B / (beta d_i^2) )`, the
//!   gradients projected on the line between the images;
//! - `E_B`, an estimate of the barrier: the highest energy on the straight line between the
//!   images less the mean of their two energies;
//! - the coefficients are set at the start of every minimization and again after a number
//!   of its iterations (100 in the paper); `alpha = 10`, `beta = 0.1`, `f = 0.5`;
//! - the search ends on the separation of the images, on the gradient at their midpoint, or
//!   on the change of the midpoint during a stage.
//!
//! [`Bitss`] is an ask/tell state machine over two points of `n` variables (the `3N`
//! Cartesian coordinates of two structures, or any other variables): it asks for energies
//! and gradients of points and never evaluates anything itself. The minimization of each
//! stage is done by a [`VectorOptimizer`], so every minimizer of `-opt` can be used (the
//! paper uses L-BFGS). The functions [`penalized`] and [`penalized_hessian`] give the BITSS
//! function alone.
//!
//! MultiOptPy: `ModelFunction/binary_image_ts_search_model_function.py:BITSSModelFunction`
//! and the BITSS branch of `optimization.py:ModelFunctionHandler`. MultiOptPy reduces the
//! separation in every iteration (not after a minimization), takes `E_B = |E1 - E2|`, sets
//! the coefficients every 500 iterations, and uses `beta = 0.02`; this module follows the
//! paper. The gradient and the Hessian of the BITSS function are the same in both.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

mod function;
mod report;
mod settings;
#[cfg(test)]
mod tests;

use faer::{Col, Mat};

pub use function::{penalized, penalized_hessian, separation, Coefficients};
pub use report::{
    BitssAction, BitssEnd, BitssPoint, BitssReport, Midpoint, PointRole, PointValue, StepSummary,
};
pub use settings::{
    BitssSettings, DEFAULT_ALPHA, DEFAULT_BETA, DEFAULT_COEFFICIENT_INTERVAL,
    DEFAULT_INTERPOLATION_POINTS, DEFAULT_MAX_STAGES, DEFAULT_MIN_SEPARATION, DEFAULT_REDUCTION,
    DEFAULT_STEP_SHARE,
};

use crate::domain::Quantity;
use crate::error::{MoptError, MoptResult};
use crate::math::linalg::{col_dot, col_max_abs, col_norm};
use crate::opt::state_machine::Termination;
use crate::opt::vector::{VectorAction, VectorOptimizer};
use function::{assemble, line, Curvature};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    /// The line between the images is surveyed and the coefficients are set; at the start
    /// of a stage the end of the search is tested first and the separation is reduced.
    Survey { new_stage: bool },
    /// The minimizer runs.
    Minimize,
}

/// The surface at the two images.
struct Images {
    energies: [f64; 2],
    gradients: [Col<f64>; 2],
    hessians: Option<[Mat<f64>; 2]>,
}

/// Ask/tell state machine of a BITSS search.
pub struct Bitss {
    settings: BitssSettings,
    n: usize,
    frozen: Vec<bool>,
    /// The two images to evaluate next (`2n` variables).
    x: Col<f64>,
    stage: usize,
    target: f64,
    previous_target: f64,
    coefficients: Option<Coefficients>,
    inner: Option<VectorOptimizer>,
    since_update: usize,
    iterations: usize,
    phase: Phase,
    /// The surface at `x`, when the minimizer has just evaluated it.
    known: Option<([f64; 2], [Col<f64>; 2])>,
    stage_start_midpoint: Option<Col<f64>>,
    midpoint: Option<Midpoint>,
    guess: Option<[Mat<f64>; 2]>,
    end: Option<BitssEnd>,
}

impl Bitss {
    pub(crate) fn updated_hessian(&self) -> Option<(&str, &Mat<f64>)> {
        self.inner.as_ref()?.updated_hessian()
    }

    /// A search between the points `first` and `second`.
    pub fn new(settings: BitssSettings, first: &Col<f64>, second: &Col<f64>) -> MoptResult<Self> {
        settings.validate()?;
        let n = first.nrows();
        if n == 0 {
            return Err(MoptError::config("BITSS: no variables"));
        }
        if first.iter().chain(second.iter()).any(|v| !v.is_finite()) {
            return Err(MoptError::config("BITSS: an image is not finite"));
        }
        let (_, d) = line(first, second)?;
        let x = Col::from_fn(2 * n, |k| if k < n { first[k] } else { second[k - n] });
        Ok(Self {
            settings,
            n,
            frozen: vec![false; n],
            x,
            stage: 0,
            target: d,
            previous_target: d,
            coefficients: None,
            inner: None,
            since_update: 0,
            iterations: 0,
            phase: Phase::Survey { new_stage: true },
            known: None,
            stage_start_midpoint: None,
            midpoint: None,
            guess: None,
            end: None,
        })
    }

    /// Keep the listed variables (indices within one image) where they are in both images;
    /// the two images must agree in them.
    pub fn with_frozen(mut self, variables: &[usize]) -> MoptResult<Self> {
        for &k in variables {
            if k >= self.n {
                return Err(MoptError::config(format!(
                    "BITSS: frozen variable {k} of {}",
                    self.n
                )));
            }
            if (self.x[k] - self.x[self.n + k]).abs() > 1e-8 {
                return Err(MoptError::config(format!(
                    "BITSS: the images differ in the frozen variable {k}"
                )));
            }
            self.frozen[k] = true;
        }
        let (first, second) = self.images();
        let (_, d) = line(&first, &second)?;
        self.target = d;
        self.previous_target = d;
        Ok(self)
    }

    /// Curvatures of the surface at the two images to start the minimizations from (model
    /// Hessians, for example), in place of the scaled unit matrix of the minimizer.
    pub fn set_curvature_guess(&mut self, first: Mat<f64>, second: Mat<f64>) -> MoptResult<()> {
        let n = self.n;
        if [&first, &second]
            .iter()
            .any(|h| h.nrows() != n || h.ncols() != n)
        {
            return Err(MoptError::config(format!(
                "BITSS: a curvature guess must be {n}x{n}"
            )));
        }
        self.guess = Some([first, second]);
        Ok(())
    }

    /// Number of variables of one image.
    pub fn dimension(&self) -> usize {
        self.n
    }

    /// Stage (0 before the first reduction of the separation).
    pub fn stage(&self) -> usize {
        self.stage
    }

    /// Separation asked for at this stage.
    pub fn target_separation(&self) -> f64 {
        self.target
    }

    /// Iterations of the minimizer so far.
    pub fn iterations(&self) -> usize {
        self.iterations
    }

    /// Why the search ended, once it has.
    pub fn end(&self) -> Option<BitssEnd> {
        self.end
    }

    /// The coefficients in force.
    pub fn coefficients(&self) -> Option<Coefficients> {
        self.coefficients
    }

    /// The two images (the ones to evaluate next; after the end, the last ones).
    pub fn images(&self) -> (Col<f64>, Col<f64>) {
        let n = self.n;
        (
            Col::from_fn(n, |k| self.x[k]),
            Col::from_fn(n, |k| self.x[n + k]),
        )
    }

    /// The point halfway between the images: the estimate of the transition state.
    pub fn transition_state_estimate(&self) -> Col<f64> {
        let n = self.n;
        Col::from_fn(n, |k| 0.5 * (self.x[k] + self.x[n + k]))
    }

    /// The midpoint as the last survey found it (with its energy; with its gradient when
    /// that survey closed a stage). It lags behind the images during a minimization.
    pub fn midpoint(&self) -> Option<&Midpoint> {
        self.midpoint.as_ref()
    }

    /// Shares of the way from the first image to the second of the points surveyed, and the
    /// index of the midpoint among them.
    fn fractions(&self) -> (Vec<f64>, usize) {
        let m = self.settings.interpolation_points;
        let mut shares: Vec<f64> = (1..=m).map(|k| k as f64 / (m + 1) as f64).collect();
        if m.is_multiple_of(2) {
            shares.insert(m / 2, 0.5);
            (shares, m / 2)
        } else {
            (shares, (m - 1) / 2)
        }
    }

    fn exact_hessians(&self) -> bool {
        self.settings
            .minimizer
            .exact_hessian_interval
            .is_some_and(|n| n > 0)
    }

    fn image_points(&self, hessian: bool) -> Vec<BitssPoint> {
        let (first, second) = self.images();
        vec![
            BitssPoint {
                role: PointRole::First,
                x: first,
                gradient: true,
                hessian,
            },
            BitssPoint {
                role: PointRole::Second,
                x: second,
                gradient: true,
                hessian,
            },
        ]
    }

    /// Next action.
    pub fn ask(&self) -> BitssAction {
        if let Some(end) = self.end {
            return BitssAction::Finished(end);
        }
        match self.phase {
            Phase::Minimize => {
                let hessian = self.inner.as_ref().is_some_and(|inner| {
                    matches!(
                        inner.ask(),
                        VectorAction::Evaluate {
                            need_hessian: true,
                            ..
                        }
                    )
                });
                BitssAction::Evaluate(self.image_points(hessian))
            }
            Phase::Survey { new_stage } => {
                let hessian = self.exact_hessians();
                let mut points = if self.known.is_none() || hessian {
                    self.image_points(hessian)
                } else {
                    Vec::new()
                };
                let (first, second) = self.images();
                let (shares, middle) = self.fractions();
                for (k, share) in shares.into_iter().enumerate() {
                    points.push(BitssPoint {
                        role: PointRole::Between(share),
                        x: Col::from_fn(self.n, |i| first[i] + share * (second[i] - first[i])),
                        gradient: new_stage && self.stage > 0 && k == middle,
                        hessian: false,
                    });
                }
                BitssAction::Evaluate(points)
            }
        }
    }

    /// Consume the values of the points of [`Self::ask`], in its order.
    pub fn tell(&mut self, values: Vec<PointValue>) -> MoptResult<BitssReport> {
        let asked = match self.ask() {
            BitssAction::Evaluate(points) => points,
            BitssAction::Finished(_) => {
                return Err(MoptError::config(
                    "tell() called after the BITSS search ended",
                ))
            }
        };
        if values.len() != asked.len() {
            return Err(MoptError::config(format!(
                "BITSS: {} values for {} points",
                values.len(),
                asked.len()
            )));
        }
        let n = self.n;
        for (point, value) in asked.iter().zip(&values) {
            if !value.energy.is_finite() {
                return Err(MoptError::calc("BITSS: an energy is not finite"));
            }
            let gradient_fits = value
                .gradient
                .as_ref()
                .is_some_and(|g| g.nrows() == n && g.iter().all(|v| v.is_finite()));
            let hessian_fits = value
                .hessian
                .as_ref()
                .is_some_and(|h| h.nrows() == n && h.ncols() == n);
            if (point.gradient && !gradient_fits) || (point.hessian && !hessian_fits) {
                return Err(MoptError::calc(format!(
                    "BITSS: the gradient or the Hessian asked for at {:?} is missing, of the \
                     wrong size, or not finite",
                    point.role
                )));
            }
        }
        let it = self.iterations;
        let stage = self.stage;
        let result = match self.phase {
            Phase::Minimize => self.minimize(values),
            Phase::Survey { new_stage } => self.survey(new_stage, &asked, values),
        };
        result.map_err(|e| e.context(format!("BITSS stage {stage}, iteration {it}")))
    }

    fn take_images(values: &mut std::vec::IntoIter<PointValue>) -> MoptResult<Images> {
        let mut next = || {
            values
                .next()
                .and_then(|v| v.gradient.map(|g| (v.energy, g, v.hessian)))
                .ok_or_else(|| MoptError::calc("BITSS: an image came without its gradient"))
        };
        let (e1, g1, h1) = next()?;
        let (e2, g2, h2) = next()?;
        Ok(Images {
            energies: [e1, e2],
            gradients: [g1, g2],
            hessians: h1.zip(h2).map(|(a, b)| [a, b]),
        })
    }

    fn masked(&self, gradient: &Col<f64>) -> Col<f64> {
        Col::from_fn(gradient.nrows(), |k| {
            if self.frozen[k % self.n] {
                0.0
            } else {
                gradient[k]
            }
        })
    }

    fn survey(
        &mut self,
        new_stage: bool,
        asked: &[BitssPoint],
        values: Vec<PointValue>,
    ) -> MoptResult<BitssReport> {
        let with_images = asked.first().is_some_and(|p| p.role == PointRole::First);
        let mut values = values.into_iter();
        let images = if with_images {
            Self::take_images(&mut values)?
        } else {
            let (energies, gradients) = self
                .known
                .take()
                .ok_or_else(|| MoptError::calc("BITSS: the images were not evaluated"))?;
            Images {
                energies,
                gradients,
                hessians: None,
            }
        };
        let between: Vec<PointValue> = values.collect();
        let (_, middle) = self.fractions();
        let at_middle = between
            .get(middle)
            .ok_or_else(|| MoptError::calc("BITSS: the midpoint was not evaluated"))?;
        let gradient = at_middle.gradient.as_ref().map(|g| self.masked(g));
        let free = self.frozen.iter().filter(|f| !**f).count().max(1);
        let midpoint = Midpoint {
            x: self.transition_state_estimate(),
            energy: at_middle.energy,
            max_force: gradient.as_ref().map(col_max_abs),
            rms_force: gradient
                .as_ref()
                .map(|g| libm::sqrt(col_dot(g, g) / free as f64)),
            gradient,
        };
        let highest = between
            .iter()
            .map(|v| v.energy)
            .chain(images.energies)
            .fold(f64::NEG_INFINITY, f64::max);
        let (first, second) = self.images();
        let (unit, d) = line(&first, &second)?;

        if new_stage {
            if let Some(end) = self.finished(&midpoint, images.energies, d) {
                self.end = Some(end);
                self.midpoint = Some(midpoint.clone());
                // The function of the stage that has just been minimized.
                let c = self
                    .coefficients
                    .ok_or_else(|| MoptError::calc("BITSS: the coefficients are not set"))?;
                let (function, gradient) = penalized(
                    &c,
                    self.target,
                    [&first, &second],
                    images.energies,
                    [&images.gradients[0], &images.gradients[1]],
                )?;
                return Ok(BitssReport {
                    iteration: self.iterations,
                    stage: self.stage,
                    target_separation: self.target,
                    separation: d,
                    images: [first, second],
                    energies: images.energies,
                    gradients: images.gradients,
                    coefficients: c,
                    function,
                    gradient: self.masked(&gradient),
                    step: None,
                    midpoint: Some(midpoint),
                    new_stage: true,
                    end: self.end,
                    notes: Vec::new(),
                });
            }
            self.stage_start_midpoint = Some(midpoint.x.clone());
            self.previous_target = self.target;
            self.target *= 1.0 - self.settings.reduction;
            self.stage += 1;
        }

        let projected = [
            col_dot(&unit, &images.gradients[0]).abs(),
            col_dot(&unit, &images.gradients[1]).abs(),
        ];
        let barrier = highest - 0.5 * (images.energies[0] + images.energies[1]);
        let c = Coefficients::new(
            self.settings.alpha,
            self.settings.beta,
            self.target,
            barrier,
            projected,
        );
        self.coefficients = Some(c);

        let mut minimizer = self.settings.minimizer.clone();
        minimizer.max_iterations = minimizer
            .max_iterations
            .saturating_sub(self.iterations)
            .max(1);
        minimizer.ccqn = None;
        minimizer.mode_kill = None;
        if let Some(share) = self.settings.step_share {
            let longest = share * self.target;
            let radius = minimizer
                .trust
                .max
                .map_or(longest, |max| max.value().min(longest));
            minimizer.trust.max = Some(Quantity::new(radius));
            minimizer.trust.min = Quantity::new(minimizer.trust.min.value().min(radius));
        }
        let mut inner = VectorOptimizer::new(minimizer, self.x.clone())?;
        if images.hessians.is_none() {
            // The curvature the two penalty terms are known to have, on top of the guess
            // for the surface. Only what cannot turn the start negative is taken: the
            // surface blocks unscaled, and the distance term as if stretched.
            let n = self.n;
            let scale = inner.initial_hessian_scale();
            let unit_block = Mat::<f64>::identity(n, n) * scale;
            let blocks = match &self.guess {
                Some([a, b]) => [a, b],
                None => [&unit_block, &unit_block],
            };
            let start = assemble(
                &c,
                &Curvature {
                    factors: [1.0, 1.0],
                    ratio: ((d - self.target) / d).max(0.0),
                    unit: &unit,
                    gradients: [&images.gradients[0], &images.gradients[1]],
                    blocks,
                },
            );
            inner.set_hessian(self.masked_hessian(start))?;
        }
        self.inner = Some(inner);
        self.since_update = 0;
        self.midpoint = Some(midpoint.clone());
        self.phase = Phase::Minimize;
        self.advance(images, d, Some(midpoint), new_stage)
    }

    /// Whether the search ends at the start of a new stage, and why.
    fn finished(
        &self,
        midpoint: &Midpoint,
        energies: [f64; 2],
        separation: f64,
    ) -> Option<BitssEnd> {
        if self.stage == 0 {
            return None;
        }
        // Between two images on either side of a ridge the energy has a maximum.
        let found = if midpoint.energy > energies[0].max(energies[1]) {
            |end: BitssEnd| end
        } else {
            |_: BitssEnd| BitssEnd::NoBarrier
        };
        let criteria = &self.settings.minimizer.criteria;
        let at_rest = midpoint
            .max_force
            .zip(midpoint.rms_force)
            .is_some_and(|(max, rms)| {
                max < criteria.max_force.value() && rms < criteria.rms_force.value()
            });
        if self.settings.stop_on_midpoint_gradient && at_rest {
            return Some(found(BitssEnd::MidpointGradient));
        }
        if separation <= self.settings.min_separation {
            return Some(found(BitssEnd::Separation));
        }
        if let (Some(share), Some(start)) =
            (self.settings.midpoint_shift, &self.stage_start_midpoint)
        {
            let shift = col_norm(&(&midpoint.x - start));
            if shift < share * (self.previous_target - self.target) {
                return Some(BitssEnd::MidpointShift);
            }
        }
        if self.stage >= self.settings.max_stages {
            return Some(BitssEnd::StageLimit);
        }
        if self.iterations >= self.settings.minimizer.max_iterations {
            return Some(BitssEnd::IterationLimit);
        }
        None
    }

    fn minimize(&mut self, values: Vec<PointValue>) -> MoptResult<BitssReport> {
        let images = Self::take_images(&mut values.into_iter())?;
        let (first, second) = self.images();
        let (_, d) = line(&first, &second)?;
        self.advance(images, d, None, false)
    }

    fn masked_hessian(&self, hessian: Mat<f64>) -> Mat<f64> {
        if !self.frozen.iter().any(|f| *f) {
            return hessian;
        }
        let n = self.n;
        Mat::from_fn(2 * n, 2 * n, |i, j| {
            if self.frozen[i % n] || self.frozen[j % n] {
                if i == j {
                    1.0
                } else {
                    0.0
                }
            } else {
                hessian[(i, j)]
            }
        })
    }

    /// Pass the BITSS function at the images to the minimizer and decide what comes next.
    fn advance(
        &mut self,
        images: Images,
        separation: f64,
        midpoint: Option<Midpoint>,
        new_stage: bool,
    ) -> MoptResult<BitssReport> {
        let c = self
            .coefficients
            .ok_or_else(|| MoptError::calc("BITSS: the coefficients are not set"))?;
        let (first, second) = self.images();
        let gradients = [&images.gradients[0], &images.gradients[1]];
        let (function, gradient) = penalized(
            &c,
            self.target,
            [&first, &second],
            images.energies,
            gradients,
        )?;
        let gradient = self.masked(&gradient);
        let hessian = match &images.hessians {
            Some([h1, h2]) => Some(self.masked_hessian(penalized_hessian(
                &c,
                self.target,
                [&first, &second],
                images.energies,
                gradients,
                [h1, h2],
            )?)),
            None => None,
        };
        let inner = self
            .inner
            .as_mut()
            .ok_or_else(|| MoptError::calc("BITSS: no minimization is running"))?;
        let step = inner.tell(function, gradient, hessian)?;
        self.iterations += 1;
        self.since_update += 1;
        let mut stage_converged = false;
        match inner.termination() {
            Some(Termination::Converged) => {
                stage_converged = true;
                self.known = Some((images.energies, images.gradients.clone()));
                self.phase = Phase::Survey { new_stage: true };
            }
            Some(_) => self.end = Some(BitssEnd::IterationLimit),
            None => {
                if let VectorAction::Evaluate { x, .. } = inner.ask() {
                    self.x = x;
                }
                self.known = None;
                self.phase = if self.since_update >= self.settings.coefficient_interval {
                    Phase::Survey { new_stage: false }
                } else {
                    Phase::Minimize
                };
            }
        }
        Ok(BitssReport {
            iteration: self.iterations,
            stage: self.stage,
            target_separation: self.target,
            separation,
            images: [first, second],
            energies: images.energies,
            gradients: images.gradients,
            coefficients: c,
            function,
            gradient: step.gradient,
            step: Some(StepSummary {
                convergence: step.convergence,
                step_norm: step.step_norm,
                trust_radius: step.trust_radius,
                stage_converged,
            }),
            midpoint,
            new_stage,
            end: self.end,
            notes: step.notes,
        })
    }
}
