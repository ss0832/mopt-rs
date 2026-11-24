//! Component-wise step scaling: every coordinate gets its own positive scale factor, a
//! diagonal preconditioner learned from the history of gradients and steps.
//!
//! Strategies, as in MultiOptPy:
//! - gradient-based: factors proportional to `(|g_i| / mean |g|)^(-1/2)`, so that coordinates
//!   with small gradients take relatively longer steps (the scaling of AdaGrad-type methods;
//!   Duchi, Hazan, and Singer, J. Mach. Learn. Res. 2011, 12, 2121);
//! - history-based: the ratio `sum |dg_i| / sum |dx_i|` over the last iterations estimates the
//!   curvature along each coordinate (a diagonal secant estimate); factors are multiplied by
//!   the inverse square root of its value relative to the mean, limited to `[0.5, 2]`;
//! - adaptive (the default): both of the above;
//! - curvature: factors proportional to `|g_k - 2 g_{k-1} + g_{k-2}|^(-1/2)`.
//!
//! Factors change by a fraction of 0.2 per iteration, stay within `[0.01, 5]`, outliers beyond
//! three standard deviations are pulled halfway to the mean, and the scaled step is damped by
//! 0.9. Correlated coordinates can share their factors (`group_coordinates`). The method needs
//! no sign convention, and the port follows MultiOptPy step by step (including the window of
//! the step history, which lags the gradient window by one iteration once it is full).
//!
//! MultiOptPy: `Optimizer/component_wise_scaling.py:ComponentWiseScaling`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Col;

use super::coordinate_locking::{column_correlations, correlated_groups};
use super::StepEnhancer;
use crate::error::MoptResult;
use crate::opt::step::StepInput;

/// How the factors are adapted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalingStrategy {
    /// Gradient-based and history-based.
    Adaptive,
    /// From the size of the gradient components.
    GradientBased,
    /// From the change of the gradient per step.
    HistoryBased,
    /// From the second difference of the gradient.
    Curvature,
}

/// Parameters of component-wise scaling.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ComponentWiseScalingParams {
    /// Strategy.
    pub strategy: ScalingStrategy,
    /// Smallest factor.
    pub min_scale_factor: f64,
    /// Largest factor.
    pub max_scale_factor: f64,
    /// Fraction by which the factors move toward their targets per iteration.
    pub adaptation_rate: f64,
    /// Exponent of the gradient-based scaling.
    pub gradient_scaling_power: f64,
    /// Number of iterations in the history window.
    pub history_memory: usize,
    /// Scale gradients by their mean before the gradient-based scaling.
    pub normalize_gradients: bool,
    /// Factor applied to the whole scaled step (`None`: no damping).
    pub global_damping: Option<f64>,
    /// Number of standard deviations beyond which a factor is an outlier.
    pub outlier_threshold: f64,
    /// Share factors among coordinates with correlated gradients.
    pub group_coordinates: bool,
    /// Correlation above which coordinates form a group.
    pub group_similarity_threshold: f64,
}

impl Default for ComponentWiseScalingParams {
    fn default() -> Self {
        Self {
            strategy: ScalingStrategy::Adaptive,
            min_scale_factor: 0.01,
            max_scale_factor: 5.0,
            adaptation_rate: 0.2,
            gradient_scaling_power: 0.5,
            history_memory: 5,
            normalize_gradients: true,
            global_damping: Some(0.9),
            outlier_threshold: 3.0,
            group_coordinates: false,
            group_similarity_threshold: 0.8,
        }
    }
}

/// Component-wise scaling step enhancer.
#[derive(Debug, Clone)]
pub struct ComponentWiseScaling {
    params: ComponentWiseScalingParams,
    factors: Vec<f64>,
    gradients: Vec<Col<f64>>,
    steps: Vec<Col<f64>>,
    groups: Vec<Vec<usize>>,
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len().max(1) as f64
}

impl ComponentWiseScaling {
    /// New enhancer.
    pub fn new(params: ComponentWiseScalingParams) -> Self {
        Self {
            params,
            factors: Vec::new(),
            gradients: Vec::new(),
            steps: Vec::new(),
            groups: Vec::new(),
        }
    }

    /// Current scale factors.
    pub fn factors(&self) -> &[f64] {
        &self.factors
    }

    fn blend(&mut self, targets: &[f64]) {
        let (p, r) = (&self.params, self.params.adaptation_rate);
        for (f, t) in self.factors.iter_mut().zip(targets) {
            *f = (1.0 - r) * *f + r * t.clamp(p.min_scale_factor, p.max_scale_factor);
        }
    }

    fn gradient_based(&mut self, g: &Col<f64>) {
        let p = self.params;
        let abs: Vec<f64> = g.iter().map(|v| v.abs().max(1e-10)).collect();
        if p.normalize_gradients {
            let m = mean(&abs);
            if m > 1e-10 {
                let targets: Vec<f64> = abs
                    .iter()
                    .map(|a| libm::pow(a / m, -p.gradient_scaling_power))
                    .collect();
                self.blend(&targets);
            }
        } else {
            let raw: Vec<f64> = abs
                .iter()
                .map(|a| libm::pow(*a, -p.gradient_scaling_power))
                .collect();
            let m = mean(&raw);
            let targets: Vec<f64> = raw.iter().map(|v| v / m).collect();
            self.blend(&targets);
        }
    }

    fn history_based(&mut self) {
        let p = self.params;
        if self.gradients.len() < 2 || self.steps.len() < 2 {
            return;
        }
        let n = self.factors.len();
        let grads =
            &self.gradients[self.gradients.len() - p.history_memory.min(self.gradients.len())..];
        let steps = &self.steps[self.steps.len() - p.history_memory.min(self.steps.len())..];
        let mut change = vec![0.0; n];
        let mut moved = vec![0.0; n];
        for k in 1..grads.len() {
            for i in 0..n {
                change[i] += (grads[k][i] - grads[k - 1][i]).abs();
                moved[i] += steps[k - 1][i].abs();
            }
        }
        // Change of the gradient per unit step: a curvature estimate per coordinate.
        let eff: Vec<f64> = (0..n)
            .map(|i| (change[i] / moved[i].max(1e-10)).max(1e-10))
            .collect();
        let m = mean(&eff);
        if m > 1e-10 {
            for (f, e) in self.factors.iter_mut().zip(&eff) {
                let adjust = (1.0 / libm::sqrt(e / m)).clamp(0.5, 2.0);
                *f = (*f * (1.0 + p.adaptation_rate * (adjust - 1.0)))
                    .clamp(p.min_scale_factor, p.max_scale_factor);
            }
        }
    }

    fn curvature_based(&mut self) {
        let p = self.params;
        let k = self.gradients.len();
        if k < 3 {
            return;
        }
        let (g0, g1, g2) = (
            &self.gradients[k - 1],
            &self.gradients[k - 2],
            &self.gradients[k - 3],
        );
        let mut s: Vec<f64> = (0..self.factors.len())
            .map(|i| 1.0 / libm::sqrt((g0[i] - 2.0 * g1[i] + g2[i]).abs().max(1e-10)))
            .collect();
        let m = mean(&s);
        if m > 1e-10 {
            s.iter_mut().for_each(|v| *v /= m);
        }
        for (f, t) in self.factors.iter_mut().zip(&s) {
            *f = ((1.0 - p.adaptation_rate) * *f + p.adaptation_rate * t.clamp(0.2, 5.0))
                .clamp(p.min_scale_factor, p.max_scale_factor);
        }
    }

    fn damp_outliers(&mut self) {
        let m = mean(&self.factors);
        let var = mean(
            &self
                .factors
                .iter()
                .map(|f| (f - m).powi(2))
                .collect::<Vec<_>>(),
        );
        let std = libm::sqrt(var);
        if std < 1e-10 {
            return;
        }
        for f in &mut self.factors {
            if ((*f - m) / std).abs() > self.params.outlier_threshold {
                *f -= 0.5 * (*f - m);
            }
        }
    }

    fn share_in_groups(&mut self) {
        if self.groups.is_empty() {
            if self.gradients.len() < 3 {
                return;
            }
            let rows = &self.gradients[self.gradients.len().saturating_sub(5)..];
            self.groups = correlated_groups(
                &column_correlations(rows),
                self.params.group_similarity_threshold,
            );
        }
        for group in &self.groups {
            let mut vals: Vec<f64> = group.iter().map(|&i| self.factors[i]).collect();
            vals.sort_by(f64::total_cmp);
            let k = vals.len();
            let median = if k.is_multiple_of(2) {
                0.5 * (vals[k / 2 - 1] + vals[k / 2])
            } else {
                vals[k / 2]
            };
            for &i in group {
                self.factors[i] = 0.7 * median + 0.3 * self.factors[i];
            }
        }
    }
}

impl Default for ComponentWiseScaling {
    fn default() -> Self {
        Self::new(ComponentWiseScalingParams::default())
    }
}

impl StepEnhancer for ComponentWiseScaling {
    fn name(&self) -> &str {
        "component_wise_scaling"
    }

    fn enhance(&mut self, input: &StepInput<'_>, step: Col<f64>) -> MoptResult<Col<f64>> {
        let n = step.nrows();
        let window = 2 * self.params.history_memory.max(1);
        self.gradients.push(input.gradient.clone());
        if self.gradients.len() > window {
            self.gradients.remove(0);
        }
        if self.factors.len() != n {
            self.factors = vec![1.0; n];
            self.steps.clear();
            self.groups.clear();
        }
        match self.params.strategy {
            ScalingStrategy::Adaptive => {
                self.gradient_based(input.gradient);
                self.history_based();
            }
            ScalingStrategy::GradientBased => self.gradient_based(input.gradient),
            ScalingStrategy::HistoryBased => self.history_based(),
            ScalingStrategy::Curvature => self.curvature_based(),
        }
        self.damp_outliers();
        if self.params.group_coordinates {
            self.share_in_groups();
        }
        let damping = self.params.global_damping.unwrap_or(1.0);
        let out = Col::from_fn(n, |i| step[i] * self.factors[i] * damping);
        self.steps.push(out.clone());
        if self.steps.len() > window {
            self.steps.remove(0);
        }
        Ok(out)
    }

    fn reset(&mut self) {
        *self = Self::new(self.params);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::opt::stepper::plain_input;
    use faer::Mat;

    fn apply(cws: &mut ComponentWiseScaling, g: &[f64], s: &[f64]) -> Col<f64> {
        let n = g.len();
        let x = Col::zeros(n);
        let h = Mat::<f64>::identity(n, n);
        let g = Col::from_fn(n, |i| g[i]);
        let s = Col::from_fn(n, |i| s[i]);
        cws.enhance(&plain_input(0, &x, 0.0, &g, &h, None, 0), s)
            .unwrap()
    }

    #[test]
    fn small_gradients_get_larger_factors_and_all_stay_positive_and_bounded() {
        let mut cws = ComponentWiseScaling::default();
        let out = apply(&mut cws, &[1.0, 0.01], &[-1.0, -0.01]);
        // Targets (|g| / 0.505)^(-1/2) = 0.7106, 5 (clipped from 7.1); factors 1 + 0.2 (t - 1).
        let f = cws.factors().to_vec();
        assert!((f[0] - (0.8 + 0.2 * (1.0_f64 / 0.505).powf(-0.5))).abs() < 1e-12);
        assert!((f[1] - 1.8).abs() < 1e-12);
        assert!((out[0] + 0.9 * f[0]).abs() < 1e-15 && (out[1] + 0.009 * f[1]).abs() < 1e-15);
        for _ in 0..200 {
            apply(&mut cws, &[1.0, 1e-9], &[-1.0, -1e-9]);
        }
        assert!(cws.factors().iter().all(|f| (0.01..=5.0).contains(f)));
    }

    #[test]
    fn history_lowers_the_factor_of_the_stiff_coordinate() {
        // Gradient changes per step: 10 for coordinate 0, 0.1 for coordinate 1.
        let mut cws = ComponentWiseScaling::new(ComponentWiseScalingParams {
            strategy: ScalingStrategy::HistoryBased,
            ..ComponentWiseScalingParams::default()
        });
        let mut g = [5.0, 0.5];
        for _ in 0..6 {
            apply(&mut cws, &g, &[-0.1, -0.1]);
            let f = cws.factors().to_vec();
            g = [g[0] - 10.0 * 0.09 * f[0], g[1] - 0.1 * 0.09 * f[1]];
        }
        let f = cws.factors();
        assert!(f[0] < 1.0 && f[1] > 1.0, "{f:?}");
        // Curvature strategy: the second difference of the gradient.
        let mut cws = ComponentWiseScaling::new(ComponentWiseScalingParams {
            strategy: ScalingStrategy::Curvature,
            ..ComponentWiseScalingParams::default()
        });
        for g in [[1.0, 1.0], [2.0, 1.1], [5.0, 1.2]] {
            apply(&mut cws, &g, &[-0.1, -0.1]);
        }
        let f = cws.factors();
        assert!(f[0] < f[1], "{f:?}");
    }
}
