//! Thermal perturbation of the step: Gaussian noise of standard deviation
//! `delta sqrt(2 gamma k_B T)` per coordinate is added, as in the random displacement of
//! overdamped Langevin (Brownian) dynamics (Ermak and McCammon, J. Chem. Phys. 1978, 69, 1352),
//! with `delta = 0.06`, the damping coefficient `gamma = 10`, and the CODATA 2018 `k_B`
//! (`Units::BOLTZMANN_HARTREE_PER_K`; MultiOptPy truncates it to 3.16681e-6 Hartree/K). Noise lets
//! a search leave saddle points and shallow minima (Jin et al., Proc. ICML 2017, "How to escape
//! saddle points efficiently").
//!
//! MultiOptPy's function draws at a constant temperature from NumPy's global generator and is
//! not called by any command. As an enhancer of a converging optimization the noise has to
//! vanish, so the temperature is annealed, `T_k = T cooling^k` (`cooling = 0.9`; `1` keeps it
//! constant), and the generator is seeded, which makes runs reproducible.
//!
//! MultiOptPy: `Optimizer/purtubation.py:Perturbation`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Col;

use super::StepEnhancer;
use crate::core_infra::rng::Rng;
use crate::domain::Units;
use crate::error::MoptResult;
use crate::opt::step::StepInput;

/// Thermal perturbation of the step.
#[derive(Debug, Clone)]
pub struct Perturbation {
    /// Scale `delta` of the noise.
    pub delta: f64,
    /// Damping coefficient `gamma`.
    pub damping_coefficient: f64,
    /// Initial temperature in K.
    pub temperature: f64,
    /// Factor applied to the temperature after every step.
    pub cooling: f64,
    seed: u64,
    rng: Rng,
    current: f64,
}

impl Perturbation {
    /// Perturbation at `temperature` (K), annealed by `cooling` per step, with a seeded
    /// generator.
    pub fn new(temperature: f64, cooling: f64, seed: u64) -> Self {
        Self {
            delta: 0.06,
            damping_coefficient: 10.0,
            temperature,
            cooling,
            seed,
            rng: Rng::from_seed(seed),
            current: temperature,
        }
    }

    /// Standard deviation of the noise per coordinate at temperature `t` (zero for `t <= 0`).
    pub fn amplitude(&self, t: f64) -> f64 {
        self.delta
            * libm::sqrt(
                (2.0 * self.damping_coefficient * Units::BOLTZMANN_HARTREE_PER_K * t).max(0.0),
            )
    }

    /// Current temperature.
    pub fn current_temperature(&self) -> f64 {
        self.current
    }
}

impl Default for Perturbation {
    fn default() -> Self {
        Self::new(298.15, 0.9, 0)
    }
}

impl StepEnhancer for Perturbation {
    fn name(&self) -> &str {
        "perturbation"
    }

    fn enhance(&mut self, _input: &StepInput<'_>, step: Col<f64>) -> MoptResult<Col<f64>> {
        let sigma = self.amplitude(self.current);
        self.current *= self.cooling;
        if sigma == 0.0 {
            return Ok(step);
        }
        let n = step.nrows();
        let noise = Col::from_fn(n, |_| sigma * self.rng.standard_normal());
        Ok(step + noise)
    }

    fn reset(&mut self) {
        self.rng = Rng::from_seed(self.seed);
        self.current = self.temperature;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::opt::stepper::plain_input;
    use faer::Mat;

    #[test]
    fn noise_has_the_thermal_variance_is_reproducible_and_anneals() {
        let n = 4000;
        let x = Col::zeros(n);
        let h = Mat::<f64>::zeros(0, 0);
        let input = plain_input(0, &x, 0.0, &x, &h, None, 0);
        let mut p = Perturbation::new(300.0, 1.0, 42);
        let s = p.enhance(&input, Col::zeros(n)).unwrap();
        let var = s.iter().map(|v| v * v).sum::<f64>() / n as f64;
        let want = 0.06_f64.powi(2) * 2.0 * 10.0 * Units::BOLTZMANN_HARTREE_PER_K * 300.0;
        assert!((var / want - 1.0).abs() < 0.1, "{var} vs {want}");
        assert!((s.iter().sum::<f64>() / n as f64).abs() < 3.0 * (want / n as f64).sqrt());
        // Same seed, same noise.
        let mut q = Perturbation::new(300.0, 1.0, 42);
        assert_eq!(q.enhance(&input, Col::zeros(n)).unwrap(), s);
        p.reset();
        assert_eq!(p.enhance(&input, Col::zeros(n)).unwrap(), s);
        // Annealing: the amplitude falls by sqrt(cooling) per step; zero temperature is silent.
        let mut a = Perturbation::default();
        let t0 = a.current_temperature();
        a.enhance(&input, Col::zeros(n)).unwrap();
        assert!((a.current_temperature() - 0.9 * t0).abs() < 1e-12);
        let mut cold = Perturbation::new(0.0, 0.9, 1);
        let step = Col::from_fn(n, |i| i as f64);
        assert_eq!(cold.enhance(&input, step.clone()).unwrap(), step);
    }
}
