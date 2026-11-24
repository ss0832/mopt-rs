//! Outer trust radius applied to every final step (MultiOptPy `CalculateMoveVector` +
//! `Optimizer/trust_radius.py`).
//!
//! The radius starts at 0.5 Bohr (minimization) or 0.1 Bohr (saddle search), or at the user's
//! maximum. It is only adapted when a Hessian is available (exact Hessian requested or model
//! Hessian enabled) and a Hessian-based step generator is used; the adaptation compares the
//! actual energy decrease with the quadratic-model prediction.
//!
//! MultiOptPy: `Optimizer/trust_radius.py:TrustRadius` and the step capping of
//!  `optimizer.py:CalculateMoveVector`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use crate::core_infra::log::python_float;
use crate::domain::{Bohr, Hartree, Quantity};
use faer::{Col, Mat};

use crate::math::linalg::col_dot;

/// User-facing trust-radius settings.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrustSettings {
    /// User maximum (`-tr`).
    pub max: Option<Quantity<Bohr>>,
    /// User minimum (`-mintr`, default 0.01 Bohr).
    pub min: Quantity<Bohr>,
}

impl Default for TrustSettings {
    fn default() -> Self {
        Self {
            max: None,
            min: Quantity::new(0.01),
        }
    }
}

/// Outer trust radius state.
#[derive(Debug, Clone)]
pub struct OuterTrustRadius {
    radius: f64,
    min: f64,
    max: f64,
    user_max: Option<f64>,
    saddle_order: usize,
    adaptive_enabled: bool,
    first_update_done: bool,
    ratios: Vec<f64>,
    energy_changes: Vec<f64>,
    /// What the last adaptation found and did, for the run log.
    notes: Vec<String>,
}

/// Previous-step data used by the adaptation.
#[derive(Debug, Clone, Copy)]
pub struct TrustUpdateInput<'a> {
    /// Current bias-including energy.
    pub energy: Quantity<Hartree>,
    /// Previous bias-including energy.
    pub prev_energy: Quantity<Hartree>,
    /// Previous gradient (optimizer coordinates).
    pub prev_gradient: &'a Col<f64>,
    /// Previous step (optimizer coordinates).
    pub prev_step: &'a Col<f64>,
    /// Norm of the previous Cartesian step.
    pub prev_cartesian_step_norm: Quantity<Bohr>,
    /// Hessian (optimizer coordinates) used for the prediction.
    pub hessian: &'a Mat<f64>,
    /// The decrease that the quadratic model of `prev_gradient` and `hessian` predicts, where
    /// it is not that of the energy: on a gradient projection of two states the gradient is
    /// not the gradient of the energy that is reported, and the decrease of the function it
    /// does belong to is given here ([`crate::opt::state_machine`], `projection_trust`).
    /// `None`: `prev_energy - energy`.
    pub decrease: Option<f64>,
}

impl OuterTrustRadius {
    /// `adaptive_enabled` corresponds to MultiOptPy's condition that an exact or model Hessian
    /// is in use and a Hessian-based optimizer is selected.
    pub fn new(settings: TrustSettings, saddle_order: usize, adaptive_enabled: bool) -> Self {
        let default = if saddle_order > 0 { 0.1 } else { 0.5 };
        let user_max = settings.max.map(Quantity::value);
        let radius = match user_max {
            None => default,
            Some(m) if saddle_order > 0 => m.min(0.1),
            Some(m) => m,
        };
        Self {
            radius,
            min: settings.min.value(),
            max: user_max.unwrap_or(0.5),
            user_max,
            saddle_order,
            adaptive_enabled,
            first_update_done: false,
            ratios: Vec::new(),
            energy_changes: Vec::new(),
            notes: Vec::new(),
        }
    }

    /// The lines MultiOptPy prints when it adapts the radius (ratio of the actual to the
    /// predicted change of the energy, the factor, the radius, and what was done), since
    /// the last call.
    pub fn take_notes(&mut self) -> Vec<String> {
        std::mem::take(&mut self.notes)
    }

    /// Current radius.
    pub fn radius(&self) -> Quantity<Bohr> {
        Quantity::new(self.radius)
    }

    fn adaptive_factor(&self) -> f64 {
        let recent = &self.ratios[self.ratios.len().saturating_sub(5)..];
        let var = if recent.len() > 1 {
            let mean = recent.iter().sum::<f64>() / recent.len() as f64;
            recent.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / recent.len() as f64
        } else {
            0.0
        };
        let mut base = 2.0 * libm::exp(-var);
        let k = self.energy_changes.len();
        if k >= 2 {
            let last = &self.energy_changes[k.saturating_sub(3)..];
            let mean = last.iter().map(|v| v.abs()).sum::<f64>() / last.len() as f64;
            if last.iter().all(|v| v.abs() < 0.01) && mean < 0.005 {
                base *= 0.8;
            }
        }
        base.clamp(1.1, 3.0)
    }

    /// Adapt the radius at the start of an iteration (no-op on the first call).
    ///
    /// The predicted decrease is `-(g . s + s^T H s / 2)`; MultiOptPy adds the quadratic term
    /// with the wrong sign (documented in `docs/source/migration_optimizers.md`).
    pub fn update(&mut self, data: Option<TrustUpdateInput<'_>>) {
        if !self.adaptive_enabled {
            return;
        }
        if let Some(d) = data {
            if self.first_update_done {
                self.adapt(&d);
            }
            self.first_update_done = true;
        }
        if self.user_max.is_none() && self.saddle_order > 0 {
            self.radius = self.radius.min(0.1);
        }
    }

    fn adapt(&mut self, d: &TrustUpdateInput<'_>) {
        const EPS: f64 = 1e-8;
        let hs = d.hessian * d.prev_step;
        let mut predicted =
            -(col_dot(d.prev_gradient, d.prev_step) + 0.5 * col_dot(d.prev_step, &hs));
        if predicted.abs() < EPS {
            predicted += predicted.signum() * EPS;
            if predicted.abs() < EPS {
                predicted = EPS;
            }
        }
        let decrease = d
            .decrease
            .unwrap_or_else(|| (d.prev_energy - d.energy).value());
        let r = decrease / predicted;
        self.ratios.push(r);
        self.energy_changes.push(decrease);
        let factor = self.adaptive_factor();
        self.notes.push(format!(
            "{} ratio (actual/predicted): {}",
            if d.decrease.is_some() {
                "Merit"
            } else {
                "Energy"
            },
            python_float(r)
        ));
        self.notes
            .push(format!("Adaptive factor: {}", python_float(factor)));
        self.notes.push(format!(
            "Current trust radius: {}",
            python_float(self.radius)
        ));
        if r <= 0.25 || r >= 1.75 {
            self.radius /= factor;
            self.notes.push("Decrease trust radius".to_string());
        } else if (0.75..=1.25).contains(&r) {
            if (d.prev_cartesian_step_norm.value() - self.radius).abs() < EPS {
                self.radius *= libm::sqrt(factor);
                self.notes.push("Increase trust radius".to_string());
            } else {
                self.notes
                    .push("Keep trust radius (good prediction)".to_string());
            }
        } else {
            self.notes
                .push("Keep trust radius (moderate prediction)".to_string());
        }
        self.radius = self.radius.clamp(self.min, self.max);
    }

    /// Scale `step` (Cartesian, Bohr) down to the radius if longer.
    pub fn cap(&self, step: Col<f64>) -> Col<f64> {
        let n = libm::sqrt(col_dot(&step, &step));
        if n > self.radius {
            step * (self.radius / n)
        } else {
            step
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_and_cap() {
        let t = OuterTrustRadius::new(TrustSettings::default(), 0, false);
        assert_eq!(t.radius().value(), 0.5);
        let ts = OuterTrustRadius::new(
            TrustSettings {
                max: Some(Quantity::new(0.3)),
                min: Quantity::new(0.01),
            },
            1,
            false,
        );
        assert_eq!(ts.radius().value(), 0.1);
        let s = t.cap(Col::from_fn(2, |i| [3.0, 4.0][i]));
        assert!((s[0] - 0.3).abs() < 1e-15 && (s[1] - 0.4).abs() < 1e-15);
    }

    #[test]
    fn poor_prediction_shrinks_radius() {
        let mut t = OuterTrustRadius::new(TrustSettings::default(), 0, true);
        let g = Col::from_fn(1, |_| 1.0);
        let s = Col::from_fn(1, |_| -0.1);
        let h = Mat::identity(1, 1);
        let mk = |e: f64| TrustUpdateInput {
            energy: Quantity::new(e),
            prev_energy: Quantity::zero(),
            prev_gradient: &g,
            prev_step: &s,
            prev_cartesian_step_norm: Quantity::new(0.1),
            hessian: &h,
            decrease: None,
        };
        t.update(Some(mk(-0.095)));
        assert_eq!(t.radius().value(), 0.5);
        // Energy went up: ratio < 0 -> shrink by the adaptive factor (2.0).
        t.update(Some(mk(0.01)));
        assert!((t.radius().value() - 0.25).abs() < 1e-12);
    }
}
