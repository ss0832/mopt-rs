//! Step-length control by reinforcement learning: a stochastic policy chooses a scale factor
//! for the proposed step and is trained during the optimization with proximal policy
//! optimization (Schulman et al., arXiv:1707.06347) and generalized advantage estimation
//! (Schulman et al., arXiv:1506.02438).
//!
//! - State: `(1, log10(rms g) / 5, cos(g, g_prev), last log-scale, last reward,
//!   tanh(last energy ratio))`.
//! - Action: the logarithm of the scale, drawn from a normal distribution whose mean is linear
//!   in the state; the scale is limited to `[0.05, 2]`.
//! - Reward of a step: the energy decrease relative to the first-order prediction,
//!   `(E_prev - E) / |g_prev . s|`, plus half the relative decrease of the gradient norm (both
//!   limited to `[-2, 2]`).
//! - Every ten steps the policy and a linear value function are updated by ten epochs of the
//!   clipped surrogate objective (`clip = 0.2`, `gamma = 0.99`, `lambda = 0.95`).
//!
//! The applied scale mixes the policy's with the neutral scale 1,
//! `w scale_policy + (1 - w)`, where the weight `w` starts at 0.1, grows by 5% after every step
//! that lowered the energy (up to 0.5), and shrinks by 5% after three failures in a row, which
//! are followed by two steps at the neutral scale. The generator is seeded, so runs are
//! reproducible; the weights can be read and set to carry a trained policy from one run to
//! the next.
//!
//! MultiOptPy's `RLStepSizeOptimizer` counts a step as successful only if the energy changed by
//! more than 70% of its absolute value, which never happens for total energies, so the policy
//! weight decays to its minimum and every step is multiplied by about 0.5; its rewards divide
//! by the absolute energy, and it reads and writes `./rl_models/step_size_policy.pt` without
//! being asked. Its neutral scale is 0.5, which halves the rate of every base method. See
//! `docs/source/migration_optimizers.md`.
//!
//! MultiOptPy: `Optimizer/rl_step.py:RLStepSizeOptimizer`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Col;

use super::StepEnhancer;
use crate::core_infra::rng::Rng;
use crate::error::{MoptError, MoptResult};
use crate::math::linalg::{col_dot, col_norm, col_rms};
use crate::opt::step::StepInput;

/// Number of state features.
pub const RL_FEATURES: usize = 6;

/// Parameters of the reinforcement-learning step scaling.
#[derive(Debug, Clone, Copy)]
pub struct RlStepParams {
    /// Smallest scale.
    pub min_scale: f64,
    /// Largest scale.
    pub max_scale: f64,
    /// Scale used without the policy.
    pub default_scale: f64,
    /// Initial weight of the policy's scale.
    pub rl_weight: f64,
    /// Smallest weight.
    pub rl_weight_min: f64,
    /// Largest weight.
    pub rl_weight_max: f64,
    /// Learning rate of the policy and value updates.
    pub learning_rate: f64,
    /// Clipping range of the probability ratio.
    pub clip_ratio: f64,
    /// Epochs per update.
    pub epochs: usize,
    /// Steps between updates.
    pub update_interval: usize,
    /// Discount factor.
    pub gamma: f64,
    /// Parameter of the advantage estimation.
    pub gae_lambda: f64,
    /// Failures in a row before the policy is paused.
    pub max_failures: usize,
    /// Steps at the default scale after such failures.
    pub recovery_steps: usize,
    /// Whether the policy is updated.
    pub training: bool,
    /// Seed of the random generator.
    pub seed: u64,
}

impl Default for RlStepParams {
    fn default() -> Self {
        Self {
            min_scale: 0.05,
            max_scale: 2.0,
            default_scale: 1.0,
            rl_weight: 0.1,
            rl_weight_min: 0.01,
            rl_weight_max: 0.5,
            learning_rate: 3e-4,
            clip_ratio: 0.2,
            epochs: 10,
            update_interval: 10,
            gamma: 0.99,
            gae_lambda: 0.95,
            max_failures: 3,
            recovery_steps: 2,
            training: true,
            seed: 0,
        }
    }
}

/// Weights of the policy: mean of the log-scale, logarithm of its standard deviation, and the
/// value function.
#[derive(Debug, Clone, PartialEq)]
pub struct RlPolicy {
    /// Coefficients of the mean of the log-scale.
    pub mean: [f64; RL_FEATURES],
    /// Logarithm of the standard deviation of the log-scale.
    pub log_sigma: f64,
    /// Coefficients of the value function.
    pub value: [f64; RL_FEATURES],
}

impl Default for RlPolicy {
    /// Mean scale 1 for every state, standard deviation 0.2, zero value.
    fn default() -> Self {
        Self {
            mean: [0.0; RL_FEATURES],
            log_sigma: libm::log(0.2),
            value: [0.0; RL_FEATURES],
        }
    }
}

fn dot(a: &[f64; RL_FEATURES], b: &[f64; RL_FEATURES]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

impl RlPolicy {
    /// Log-probability density of the action `a` in the state `s`.
    pub fn log_prob(&self, s: &[f64; RL_FEATURES], a: f64) -> f64 {
        let sigma = libm::exp(self.log_sigma);
        let z = (a - dot(&self.mean, s)) / sigma;
        -0.5 * z * z - self.log_sigma - 0.5 * libm::log(2.0 * std::f64::consts::PI)
    }
}

#[derive(Debug, Clone, Copy)]
struct Transition {
    state: [f64; RL_FEATURES],
    action: f64,
    log_prob: f64,
    value: f64,
    reward: f64,
}

/// Reinforcement-learning step-scaling enhancer.
#[derive(Debug, Clone)]
pub struct RlStep {
    params: RlStepParams,
    policy: RlPolicy,
    rng: Rng,
    weight: f64,
    failures: usize,
    recovery: usize,
    memory: Vec<Transition>,
    /// State, action, log-probability, and value of the step taken last, awaiting its reward.
    pending: Option<([f64; RL_FEATURES], f64, f64, f64)>,
    /// Gradient and energy of the previous point.
    last: Option<(Col<f64>, f64)>,
    last_log_scale: f64,
    last_reward: f64,
    last_ratio: f64,
    updates: usize,
}

impl RlStep {
    /// New enhancer with the neutral policy.
    pub fn new(params: RlStepParams) -> Self {
        Self {
            params,
            policy: RlPolicy::default(),
            rng: Rng::from_seed(params.seed),
            weight: params.rl_weight,
            failures: 0,
            recovery: 0,
            memory: Vec::new(),
            pending: None,
            last: None,
            last_log_scale: 0.0,
            last_reward: 0.0,
            last_ratio: 0.0,
            updates: 0,
        }
    }

    /// Current policy weights (to continue training in a later run).
    pub fn policy(&self) -> &RlPolicy {
        &self.policy
    }

    /// Start from the given policy weights.
    pub fn set_policy(&mut self, policy: RlPolicy) -> MoptResult<()> {
        let finite = policy
            .mean
            .iter()
            .chain(&policy.value)
            .all(|v| v.is_finite())
            && policy.log_sigma.is_finite();
        if !finite {
            return Err(MoptError::config("policy weights must be finite"));
        }
        self.policy = policy;
        Ok(())
    }

    /// Current weight of the policy's scale.
    pub fn rl_weight(&self) -> f64 {
        self.weight
    }

    /// Number of policy updates performed.
    pub fn updates(&self) -> usize {
        self.updates
    }

    /// Reward of a step `s` taken from a point with energy `e_prev` and gradient `g_prev` to a
    /// point with energy `e` and gradient `g`.
    pub fn reward(e_prev: f64, e: f64, g_prev: &Col<f64>, g: &Col<f64>, s: &Col<f64>) -> f64 {
        let predicted = col_dot(g_prev, s).abs();
        let decrease = e_prev - e;
        // decrease / predicted, limited to [-2, 2] without dividing by a vanishing prediction.
        let energy_term = if decrease.abs() >= 2.0 * predicted {
            if decrease == 0.0 {
                0.0
            } else {
                2.0 * decrease.signum()
            }
        } else {
            decrease / predicted
        };
        let (n_prev, n) = (col_norm(g_prev), col_norm(g));
        let gradient_term = if n_prev > 0.0 {
            (0.5 * (n_prev - n) / n_prev).clamp(-2.0, 2.0)
        } else {
            0.0
        };
        energy_term + gradient_term
    }

    fn state(&self, input: &StepInput<'_>) -> [f64; RL_FEATURES] {
        let g = input.gradient;
        let rms = col_rms(g);
        let log_g = if rms > 0.0 {
            (libm::log10(rms) / 5.0).clamp(-3.0, 3.0)
        } else {
            -3.0
        };
        let cosine = match &self.last {
            Some((gp, _)) if gp.nrows() == g.nrows() => {
                let d = col_norm(gp) * col_norm(g);
                if d > 0.0 {
                    col_dot(gp, g) / d
                } else {
                    0.0
                }
            }
            _ => 0.0,
        };
        [
            1.0,
            log_g,
            cosine,
            self.last_log_scale,
            self.last_reward,
            libm::tanh(self.last_ratio),
        ]
    }

    /// Proximal policy optimization on the stored transitions.
    fn update(&mut self, bootstrap: f64) {
        let p = self.params;
        let n = self.memory.len();
        if n == 0 {
            return;
        }
        // Generalized advantage estimation.
        let mut adv = vec![0.0; n];
        let mut running = 0.0;
        for t in (0..n).rev() {
            let next = if t + 1 < n {
                self.memory[t + 1].value
            } else {
                bootstrap
            };
            let delta = self.memory[t].reward + p.gamma * next - self.memory[t].value;
            running = delta + p.gamma * p.gae_lambda * running;
            adv[t] = running;
        }
        let returns: Vec<f64> = (0..n).map(|t| adv[t] + self.memory[t].value).collect();
        let mean = adv.iter().sum::<f64>() / n as f64;
        let std = libm::sqrt(adv.iter().map(|a| (a - mean).powi(2)).sum::<f64>() / n as f64);
        for a in &mut adv {
            *a = (*a - mean) / (std + 1e-8);
        }
        for _ in 0..p.epochs {
            let mut g_mean = [0.0; RL_FEATURES];
            let mut g_sigma = 0.0;
            let mut g_value = [0.0; RL_FEATURES];
            for (t, tr) in self.memory.iter().enumerate() {
                let sigma = libm::exp(self.policy.log_sigma);
                let z = (tr.action - dot(&self.policy.mean, &tr.state)) / sigma;
                let ratio = libm::exp(
                    (self.policy.log_prob(&tr.state, tr.action) - tr.log_prob).clamp(-20.0, 20.0),
                );
                // Gradient of min(ratio A, clip(ratio) A): zero where the clipped term is active.
                let clipped = ratio.clamp(1.0 - p.clip_ratio, 1.0 + p.clip_ratio);
                if ratio * adv[t] <= clipped * adv[t] {
                    let w = ratio * adv[t];
                    for (gm, x) in g_mean.iter_mut().zip(&tr.state) {
                        *gm += w * z / sigma * x;
                    }
                    g_sigma += w * (z * z - 1.0);
                }
                let err = dot(&self.policy.value, &tr.state) - returns[t];
                for (gv, x) in g_value.iter_mut().zip(&tr.state) {
                    *gv += err * x;
                }
            }
            let scale = p.learning_rate / n as f64;
            for k in 0..RL_FEATURES {
                self.policy.mean[k] += scale * g_mean[k];
                self.policy.value[k] -= scale * g_value[k];
            }
            // Standard deviation between 0.01 and 1.
            self.policy.log_sigma =
                (self.policy.log_sigma + scale * g_sigma).clamp(libm::log(0.01), 0.0);
        }
        self.memory.clear();
        self.updates += 1;
    }
}

impl Default for RlStep {
    fn default() -> Self {
        Self::new(RlStepParams::default())
    }
}

impl StepEnhancer for RlStep {
    fn name(&self) -> &str {
        "rl_step"
    }

    fn enhance(&mut self, input: &StepInput<'_>, step: Col<f64>) -> MoptResult<Col<f64>> {
        let p = self.params;
        let e = input.energy.value();
        // Reward the step that led here.
        if let (Some((g_prev, e_prev)), Some(prev)) = (&self.last, input.previous) {
            if g_prev.nrows() == prev.step.nrows() {
                let r = Self::reward(*e_prev, e, g_prev, input.gradient, &prev.step);
                self.last_reward = r.clamp(-4.0, 4.0);
                let predicted = col_dot(g_prev, &prev.step).abs();
                self.last_ratio = if predicted > 0.0 {
                    ((e_prev - e) / predicted).clamp(-10.0, 10.0)
                } else {
                    0.0
                };
                if e < *e_prev {
                    self.failures = 0;
                    self.weight = (self.weight * 1.05).min(p.rl_weight_max);
                } else {
                    self.failures += 1;
                    if self.failures >= p.max_failures {
                        self.recovery = p.recovery_steps;
                        self.weight = (self.weight * 0.95).max(p.rl_weight_min);
                        self.failures = 0;
                    }
                }
                if let Some((state, action, log_prob, value)) = self.pending.take() {
                    self.memory.push(Transition {
                        state,
                        action,
                        log_prob,
                        value,
                        reward: self.last_reward,
                    });
                }
            }
        }
        let state = self.state(input);
        self.last = Some((input.gradient.clone(), e));
        if p.training && self.memory.len() >= p.update_interval.max(1) {
            self.update(dot(&self.policy.value, &state));
        }
        if self.recovery > 0 {
            self.recovery -= 1;
            self.pending = None;
            self.last_log_scale = libm::log(p.default_scale);
            return Ok(step * p.default_scale);
        }
        let sigma = libm::exp(self.policy.log_sigma);
        let action = dot(&self.policy.mean, &state) + sigma * self.rng.standard_normal();
        let log_prob = self.policy.log_prob(&state, action);
        let policy_scale = libm::exp(action.clamp(-20.0, 20.0)).clamp(p.min_scale, p.max_scale);
        let scale = self.weight * policy_scale + (1.0 - self.weight) * p.default_scale;
        self.pending = Some((state, action, log_prob, dot(&self.policy.value, &state)));
        self.last_log_scale = libm::log(scale);
        Ok(step * scale)
    }

    fn reset(&mut self) {
        let policy = self.policy.clone();
        *self = Self::new(self.params);
        // What has been learned is kept.
        self.policy = policy;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::opt::step::PreviousPoint;
    use crate::opt::stepper::plain_input;
    use faer::Mat;

    #[test]
    fn rewards_are_scale_free_and_bounded() {
        let g = Col::from_fn(2, |i| [1.0, 0.0][i]);
        let s = Col::from_fn(2, |i| [-0.5, 0.0][i]);
        let half = Col::from_fn(2, |i| [0.5, 0.0][i]);
        // Quadratic E = x^2 / 2 from x = 1 to 0.5: decrease 0.375, predicted 0.5, |g| halves.
        let r = RlStep::reward(0.5, 0.125, &g, &half, &s);
        assert!((r - (0.75 + 0.25)).abs() < 1e-12);
        // The same in other units of energy.
        let r2 = RlStep::reward(500.0, 125.0, &(&g * 1000.0), &(&half * 1000.0), &s);
        assert!((r2 - r).abs() < 1e-12);
        // An energy increase is punished; a vanishing prediction does not divide by zero.
        assert!(RlStep::reward(0.5, 0.9, &g, &g, &s) < 0.0);
        let zero = Col::zeros(2);
        assert_eq!(RlStep::reward(1.0, 0.0, &zero, &zero, &s), 2.0);
        assert_eq!(RlStep::reward(1.0, 1.0, &zero, &zero, &s), 0.0);
    }

    /// Steepest descent with factor `f` on `E = k x^2 / 2`, scaled by the enhancer.
    fn run(rl: &mut RlStep, k: f64, f: f64, steps: usize) -> (f64, Vec<f64>) {
        let h = Mat::<f64>::identity(1, 1);
        let mut x: Col<f64> = Col::from_fn(1, |_| 1.0);
        let mut prev: Option<PreviousPoint> = None;
        let mut scales = Vec::new();
        for it in 0..steps {
            if x[0].abs() < 1e-9 {
                x[0] = 1.0;
                prev = None;
            }
            let g = &x * k;
            let e = 0.5 * k * x[0] * x[0];
            let proposal = &g * (-f);
            let s = rl
                .enhance(
                    &plain_input(it, &x, e, &g, &h, prev.as_ref(), 0),
                    proposal.clone(),
                )
                .unwrap();
            scales.push(s[0] / proposal[0]);
            prev = Some(PreviousPoint {
                energy: crate::domain::Quantity::new(e),
                gradient: g.clone(),
                raw_gradient: g,
                step: s.clone(),
            });
            x += &s;
        }
        (x[0], scales)
    }

    #[test]
    fn scales_stay_near_one_at_first_and_are_reproducible() {
        let mut rl = RlStep::default();
        let (_, scales) = run(&mut rl, 1.0, 0.5, 30);
        // Weight at most 0.5 on a scale in [0.05, 2]: the applied scale is within [0.5, 1.5].
        assert!(scales.iter().all(|s| (0.5..=1.5).contains(s)), "{scales:?}");
        assert!(rl.rl_weight() > 0.1, "successes raise the weight");
        assert!(rl.updates() >= 2);
        let mut again = RlStep::default();
        assert_eq!(run(&mut again, 1.0, 0.5, 30).1, scales);
        // Failures: three energy increases in a row pause the policy for two steps.
        let mut rl = RlStep::default();
        let (_, scales) = run(&mut rl, 1.0, 2.5, 8);
        assert!(rl.rl_weight() < 0.1);
        assert!(
            scales.iter().filter(|s| **s == 1.0).count() >= 2,
            "{scales:?}"
        );
        assert!(rl
            .set_policy(RlPolicy {
                log_sigma: f64::NAN,
                ..RlPolicy::default()
            })
            .is_err());
    }

    #[test]
    fn the_policy_learns_to_shorten_steps_that_overshoot() {
        // Proposals overshoot (factor 1.8 on curvature 1: the error changes sign and shrinks
        // by 0.8 per step); shorter steps earn higher rewards.
        let mut rl = RlStep::new(RlStepParams {
            learning_rate: 0.05,
            rl_weight: 0.5,
            ..RlStepParams::default()
        });
        run(&mut rl, 1.0, 1.8, 3000);
        assert!(rl.updates() > 100);
        assert!(
            rl.policy().mean[0] < -0.05,
            "mean log-scale {:?}",
            rl.policy().mean
        );
    }
}
