//! A surface learned from the values that were calculated, which answers for images near
//! them without a calculation.
//!
//! Chains can be relaxed with far fewer calculations when a Gaussian process stands in
//! for the surface between them (Koistinen, Dagbjartsdottir, Asgeirsson, Vehtari, Jonsson,
//! J. Chem. Phys. 2017, 147, 152720; Garrido Torres, Jennings, Hansen, Boes, Bligaard,
//! Phys. Rev. Lett. 2019, 122, 156001). [`Surrogate`] is the part of that idea that does
//! not depend on how the chain is relaxed: it keeps the points that were calculated
//! (position, value, gradient) and predicts value and gradient at a point with the
//! uncertainty of the prediction. Who relaxes the chain decides with the uncertainty
//! whether to believe the prediction or to calculate
//! ([`crate::workflow::chain_surrogate`]), so any forces, steps, and redistributions of
//! the chain work with it.
//!
//! Memory and time are kept small in three ways:
//!
//! - **Local models.** A prediction uses the few points nearest to it (`neighbours`), not
//!   all points of the run: one Gaussian process of at most that many points per
//!   prediction, with the factorization of [`crate::math::gp`] that never forms the
//!   covariance matrix of all coordinates. The time of a prediction does not grow with
//!   the length of the run.
//! - **No near duplicates.** A new point replaces a kept point that is nearer than a small
//!   share of the length scale: a chain that comes to rest would otherwise fill the
//!   model with points that say the same and make it ill conditioned.
//! - **A limit.** [`Surrogate::keep_near`] keeps the points nearest to the images of the
//!   chain, a given number per image, and forgets the others.
//!
//! This is not the algorithm of either publication (which relax the whole chain on the
//! model between calculations and choose one image to calculate): the kernel is the
//! squared exponential with a fixed length scale and gradient observations, the prior
//! mean is the mean of the values of the points used, and the uncertainty is the posterior
//! variance relative to the amplitude of the kernel.
//!
//! How far the model can be believed was measured (HCN, GFN0-xTB; the ignored test of
//! this file): with twelve calculated points within 0.05 to 0.3 Bohr the predicted
//! gradient is off by `1e-3` to `5e-2` Hartree/Bohr, whatever the length scale (0.4 to
//! 1.5 Bohr) and the noise, while the posterior variance says `1e-9` to `1e-3`. The error
//! is that of the surface (its third derivatives over these distances), and the variance
//! does not see it. The settings therefore have a distance beyond which nothing is
//! predicted, and who uses the model measures its error against calculated values.
//!
//! MultiOptPy: `MEP/pathopt_gpneb_force.py` (unfinished there: not called, and it uses
//! names that are not defined).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::{Col, Mat};

use crate::error::MoptResult;
use crate::math::gp::{GradientGp, Observation};
use crate::math::linalg::pseudo_inverse;

/// A new point replaces a kept one nearer than this share of the length scale.
const SAME_POINT: f64 = 0.02;

/// What the model sees of a point.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Features {
    /// The point itself.
    #[default]
    Points,
    /// For a structure (three coordinates per atom): the inverse distances of all pairs
    /// of atoms (Koistinen, Asgeirsson, Vehtari, Jonsson, J. Chem. Theory Comput. 2019,
    /// 15, 6738, use them inside the kernel). They do not change when the structure is
    /// shifted or turned, have no places where they are not defined, and a bond that
    /// stretches is nearer a straight line in them. The gradient of the energy in them
    /// is the one of least length that gives the Cartesian gradient, `J (J^T J)^+ g`
    /// with `J` the derivatives of the inverse distances; a predicted gradient goes back
    /// with `J^T`.
    InverseDistances,
}

/// The inverse distances of all pairs `i < j` of the atoms of `x`, and for each pair the
/// vector `d(1/r)/dx_i` (the derivative for the atom `j` is its negative).
fn inverse_distances(x: &Col<f64>) -> (Col<f64>, Vec<[f64; 3]>) {
    let atoms = x.nrows() / 3;
    let pairs = atoms * atoms.saturating_sub(1) / 2;
    let mut values = Col::<f64>::zeros(pairs);
    let mut slopes = Vec::with_capacity(pairs);
    let mut pair = 0;
    for i in 0..atoms {
        for j in i + 1..atoms {
            let v = [
                x[3 * i] - x[3 * j],
                x[3 * i + 1] - x[3 * j + 1],
                x[3 * i + 2] - x[3 * j + 2],
            ];
            let r = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-12);
            values[pair] = 1.0 / r;
            let scale = -1.0 / (r * r * r);
            slopes.push([scale * v[0], scale * v[1], scale * v[2]]);
            pair += 1;
        }
    }
    (values, slopes)
}

/// `J^T u`: a gradient in the inverse distances as a Cartesian gradient.
fn to_cartesian(slopes: &[[f64; 3]], u: &Col<f64>, coordinates: usize) -> Col<f64> {
    let atoms = coordinates / 3;
    let mut out = Col::<f64>::zeros(coordinates);
    let mut pair = 0;
    for i in 0..atoms {
        for j in i + 1..atoms {
            for k in 0..3 {
                out[3 * i + k] += u[pair] * slopes[pair][k];
                out[3 * j + k] -= u[pair] * slopes[pair][k];
            }
            pair += 1;
        }
    }
    out
}

/// `J (J^T J)^+ g`: the Cartesian gradient `g` as the shortest gradient in the inverse
/// distances that gives it.
fn to_features(slopes: &[[f64; 3]], g: &Col<f64>) -> MoptResult<Col<f64>> {
    let n = g.nrows();
    let atoms = n / 3;
    // J^T J, atom block by atom block.
    let mut metric = Mat::<f64>::zeros(n, n);
    let mut pair = 0;
    for i in 0..atoms {
        for j in i + 1..atoms {
            let s = &slopes[pair];
            for a in 0..3 {
                for b in 0..3 {
                    let value = s[a] * s[b];
                    metric[(3 * i + a, 3 * i + b)] += value;
                    metric[(3 * j + a, 3 * j + b)] += value;
                    metric[(3 * i + a, 3 * j + b)] -= value;
                    metric[(3 * j + a, 3 * i + b)] -= value;
                }
            }
            pair += 1;
        }
    }
    let weights = pseudo_inverse(&metric)? * g;
    let mut out = Col::<f64>::zeros(slopes.len());
    let mut pair = 0;
    for i in 0..atoms {
        for j in i + 1..atoms {
            out[pair] = (0..3)
                .map(|k| slopes[pair][k] * (weights[3 * i + k] - weights[3 * j + k]))
                .sum();
            pair += 1;
        }
    }
    Ok(out)
}

/// How the model is built.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurrogateSettings {
    /// Length scale of the kernel (in the unit of the points: Bohr for structures).
    pub length_scale: f64,
    /// Relative noise of the observations.
    pub noise: f64,
    /// Points used for one prediction (the nearest ones).
    pub neighbours: usize,
    /// Fewest points for a prediction.
    pub fewest: usize,
    /// Largest uncertainty (posterior variance over the amplitude of the kernel, between
    /// 0 and 1) at which a prediction is believed.
    pub uncertainty: f64,
    /// Points kept per image of the chain ([`Surrogate::keep_near`]).
    pub memory: usize,
    /// Largest distance from the nearest calculated point at which a prediction is
    /// believed, in length scales.
    pub reach: f64,
    /// Iterations of a chain answered from the model between two that are calculated.
    pub between: usize,
    /// What the model sees of a point.
    pub features: Features,
    /// The model is used while its measured error of the gradient is below this share of
    /// the largest force of the chain. With 0.1 the model did no harm to the chains of
    /// molecules that were measured and was seldom used; with 0.3 it saved calculations
    /// for some and cost many for others (`docs/source/migration_paths.md`, "Chains with a
    /// Gaussian process").
    pub error_share: f64,
}

impl Default for SurrogateSettings {
    fn default() -> Self {
        Self {
            // 0.4 Angstrom, as in the Gaussian-process optimizer of this library.
            length_scale: 0.4 * crate::domain::Units::ANGSTROM_TO_BOHR,
            noise: 1e-4,
            neighbours: 12,
            fewest: 2,
            uncertainty: 0.02,
            memory: 30,
            reach: 0.5,
            between: 2,
            features: Features::Points,
            error_share: 0.1,
        }
    }
}

/// A prediction at a point.
#[derive(Debug, Clone)]
pub struct Predicted {
    /// The value.
    pub value: f64,
    /// Its gradient.
    pub gradient: Col<f64>,
    /// The uncertainty (0 at a point that was calculated, 1 far from all of them).
    pub uncertainty: f64,
    /// Distance from the nearest calculated point.
    pub nearest: f64,
}

fn distance(a: &Col<f64>, b: &Col<f64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt()
}

/// The points that were calculated, and predictions from them.
#[derive(Debug, Clone)]
pub struct Surrogate {
    settings: SurrogateSettings,
    points: Vec<Observation>,
}

impl Surrogate {
    /// An empty model.
    pub fn new(settings: SurrogateSettings) -> Self {
        Self {
            settings,
            points: Vec::new(),
        }
    }

    /// The settings.
    pub fn settings(&self) -> &SurrogateSettings {
        &self.settings
    }

    /// Whether `found` is within what the settings believe: a small uncertainty, near a
    /// calculated point.
    pub fn believes(&self, found: &Predicted) -> bool {
        found.uncertainty <= self.settings.uncertainty
            && found.nearest <= self.settings.reach * self.settings.length_scale
    }

    /// The number of points kept.
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Whether no point is kept.
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// A calculated point is added (it replaces a kept point at nearly the same place).
    pub fn record(&mut self, x: &Col<f64>, value: f64, gradient: &Col<f64>) -> MoptResult<()> {
        let (seen, gradient) = match self.settings.features {
            Features::Points => (x.clone(), gradient.clone()),
            Features::InverseDistances => {
                let (values, slopes) = inverse_distances(x);
                (values, to_features(&slopes, gradient)?)
            }
        };
        let same = SAME_POINT * self.settings.length_scale;
        self.points.retain(|kept| distance(&kept.x, &seen) >= same);
        self.points.push(Observation {
            x: seen,
            value,
            gradient,
        });
        Ok(())
    }

    /// Value, gradient, and uncertainty at `x` from the nearest points; `None` with fewer
    /// points than the settings ask for.
    pub fn predict(&self, x: &Col<f64>) -> MoptResult<Option<Predicted>> {
        if self.points.len() < self.settings.fewest.max(1) {
            return Ok(None);
        }
        let coordinates = x.nrows();
        let (seen, slopes) = match self.settings.features {
            Features::Points => (x.clone(), Vec::new()),
            Features::InverseDistances => inverse_distances(x),
        };
        let x = &seen;
        let mut order: Vec<(f64, usize)> = self
            .points
            .iter()
            .enumerate()
            .map(|(k, kept)| (distance(&kept.x, x), k))
            .collect();
        order.sort_by(|a, b| a.0.total_cmp(&b.0));
        let used: Vec<Observation> = order
            .iter()
            .take(self.settings.neighbours.max(1))
            .map(|(_, k)| self.points[*k].clone())
            .collect();
        let mean = used.iter().map(|o| o.value).sum::<f64>() / used.len() as f64;
        let model = GradientGp::fit(&used, self.settings.length_scale, self.settings.noise, mean)?;
        let (value, gradient) = model.predict(x);
        let gradient = match self.settings.features {
            Features::Points => gradient,
            Features::InverseDistances => to_cartesian(&slopes, &gradient, coordinates),
        };
        Ok(Some(Predicted {
            value,
            gradient,
            uncertainty: model.variance(x).clamp(0.0, 1.0),
            nearest: order.first().map_or(f64::INFINITY, |(d, _)| *d),
        }))
    }

    /// The points nearest to the images of `chain` are kept, `memory` per image at most;
    /// the others are forgotten.
    pub fn keep_near(&mut self, chain: &[Col<f64>]) {
        let most = self.settings.memory.max(1) * chain.len().max(1);
        if self.points.len() <= most {
            return;
        }
        let seen: Vec<Col<f64>> = chain
            .iter()
            .map(|image| match self.settings.features {
                Features::Points => image.clone(),
                Features::InverseDistances => inverse_distances(image).0,
            })
            .collect();
        let chain = &seen;
        let mut order: Vec<(f64, usize)> = self
            .points
            .iter()
            .enumerate()
            .map(|(k, kept)| {
                let nearest = chain
                    .iter()
                    .map(|image| distance(&kept.x, image))
                    .fold(f64::INFINITY, f64::min);
                (nearest, k)
            })
            .collect();
        order.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut keep: Vec<usize> = order.iter().take(most).map(|(_, k)| *k).collect();
        keep.sort_unstable();
        self.points = keep.into_iter().map(|k| self.points[k].clone()).collect();
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    /// A function of two variables with a curved valley.
    fn surface(x: &Col<f64>) -> (f64, Col<f64>) {
        let (a, b) = (x[0], x[1]);
        let bend = b - 0.3 * a * a;
        let value = 0.05 * a * a + 0.8 * bend * bend + 0.02 * (2.0 * a).sin();
        let gradient = Col::from_fn(2, |k| {
            if k == 0 {
                0.1 * a - 1.6 * bend * 0.6 * a + 0.04 * (2.0 * a).cos()
            } else {
                1.6 * bend
            }
        });
        (value, gradient)
    }

    fn at(a: f64, b: f64) -> Col<f64> {
        Col::from_fn(2, |k| if k == 0 { a } else { b })
    }

    #[test]
    fn predictions_near_the_points_are_good_and_say_how_far_they_are() {
        let mut model = Surrogate::new(SurrogateSettings {
            length_scale: 0.8,
            ..SurrogateSettings::default()
        });
        assert!(model.predict(&at(0.0, 0.0)).unwrap().is_none());
        for k in 0..8 {
            let x = at(-1.0 + 0.3 * f64::from(k), 0.1 * f64::from(k % 3));
            let (value, gradient) = surface(&x);
            model.record(&x, value, &gradient).unwrap();
        }
        assert_eq!(model.len(), 8);
        // At a point that was calculated: the value, the gradient, no uncertainty.
        let x = at(-0.4, 0.2);
        let (value, gradient) = surface(&x);
        let found = model.predict(&x).unwrap().unwrap();
        // (The fit has a little noise and does not pass through the points exactly.)
        assert!(
            (found.value - value).abs() < 1e-4,
            "{}",
            found.value - value
        );
        assert!((found.gradient[0] - gradient[0]).abs() < 1e-3);
        assert!(found.uncertainty < 1e-4, "{}", found.uncertainty);
        // Between the points: close, with a small uncertainty; far away: none of it.
        let x = at(-0.25, 0.12);
        let (value, gradient) = surface(&x);
        let between = model.predict(&x).unwrap().unwrap();
        assert!(
            (between.value - value).abs() < 2e-3,
            "{}",
            between.value - value
        );
        assert!((between.gradient[1] - gradient[1]).abs() < 2e-2);
        assert!(between.uncertainty < 0.02, "{}", between.uncertainty);
        let far = model.predict(&at(4.0, -3.0)).unwrap().unwrap();
        assert!(far.uncertainty > 0.9, "{}", far.uncertainty);

        // A point at nearly the same place replaces the one there.
        let x = at(-0.4 + 1e-4, 0.2);
        let (value, gradient) = surface(&x);
        model.record(&x, value, &gradient).unwrap();
        assert_eq!(model.len(), 8);
        // Only the points near the chain are kept.
        let mut small = Surrogate::new(SurrogateSettings {
            memory: 2,
            ..*model.settings()
        });
        for k in 0..8 {
            let x = at(-1.0 + 0.3 * f64::from(k), 0.0);
            let (value, gradient) = surface(&x);
            small.record(&x, value, &gradient).unwrap();
        }
        small.keep_near(&[at(-1.0, 0.0)]);
        assert_eq!(small.len(), 2);
        assert!(small.predict(&at(-0.9, 0.0)).unwrap().unwrap().uncertainty < 0.05);
        assert!(small.predict(&at(1.1, 0.0)).unwrap().unwrap().uncertainty > 0.5);
    }

    /// The error of predictions for HCN against their uncertainty (the table behind the
    /// way the surrogate is trusted: `cargo test -p mopt-rs --lib how_wrong -- --nocapture`).
    #[test]
    fn how_wrong_predictions_are_for_a_molecule() {
        use crate::domain::{EvalRequest, SpinCharge};
        use crate::io::xyz::parse_xyz_str;
        use crate::pes::registry::{CalculatorOptions, CalculatorRegistry};
        let sys = parse_xyz_str(
            "3\n\nC 0.000 0.000 0.000\nN 1.190 0.000 0.000\nH 0.368 1.132 0.000\n",
            SpinCharge::default(),
            None,
        )
        .unwrap();
        let calculator = CalculatorRegistry::native()
            .create("gfn0xtb", &CalculatorOptions::default(), &sys)
            .unwrap();
        let surface = |x: &Col<f64>| {
            let out = calculator
                .evaluate(
                    &sys.with_flat_coords_bohr(x).unwrap(),
                    &EvalRequest::ENERGY_GRAD,
                )
                .unwrap();
            (out.energy.value(), out.gradient_bohr)
        };
        let x0 = sys.coords_flat_bohr();
        let wave =
            |k: usize, seed: f64| ((0.618_033_988_75 * (k + 1) as f64 + seed).fract() - 0.5) * 2.0;
        let settings = [
            (Features::Points, 0.756),
            (Features::InverseDistances, 0.02),
            (Features::InverseDistances, 0.05),
            (Features::InverseDistances, 0.1),
            (Features::InverseDistances, 0.2),
            (Features::InverseDistances, 0.5),
        ];
        // Predictions whose gradient is off by more than a hundred times what the model
        // says of its uncertainty.
        let mut far_off = 0;
        for (features, length) in settings {
            let noise = 1e-4;
            println!("-- {features:?}, length scale {length}");
            for spread in [0.05, 0.15, 0.3] {
                let mut model = Surrogate::new(SurrogateSettings {
                    length_scale: length,
                    noise,
                    features,
                    ..SurrogateSettings::default()
                });
                // Twelve points around the structure, as a chain that moves leaves them.
                for point in 0..12 {
                    let x = Col::from_fn(9, |k| x0[k] + spread * wave(k, 0.37 * point as f64));
                    let (value, gradient) = surface(&x);
                    model.record(&x, value, &gradient).unwrap();
                }
                for reach in [0.02, 0.05, 0.1, 0.2] {
                    let x = Col::from_fn(9, |k| x0[k] + reach * wave(k, 0.11));
                    let (value, gradient) = surface(&x);
                    let found = model.predict(&x).unwrap().unwrap();
                    let worst = (0..9)
                        .map(|k| (found.gradient[k] - gradient[k]).abs())
                        .fold(0.0, f64::max);
                    println!(
                        "l {length} noise {noise:e} spread {spread} reach {reach}: energy off {:.2e}, gradient off {worst:.2e}, uncertainty {:.2e}",
                        found.value - value,
                        found.uncertainty
                    );
                    assert!(worst.is_finite() && found.uncertainty.is_finite());
                    if worst > 100.0 * found.uncertainty {
                        far_off += 1;
                    }
                }
            }
        }
        // What the measurement showed: the uncertainty of the model does not bound the
        // error of its gradients, which is why the error is measured against calculations.
        assert!(far_off > 0);
    }
}
