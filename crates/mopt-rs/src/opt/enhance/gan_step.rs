//! Adversarially learned step scaling (a generative adversarial network; Goodfellow et al.,
//! Adv. Neural Inf. Process. Syst. 2014, 27, 2672).
//!
//! A generator network maps the features of one coordinate, the gradient and step components
//! in units of their root mean squares `(g_i / rms g, s_i / rms s)`, to a scale factor in
//! `[min_scale, max_scale]`. A discriminator network is trained to tell the scaled proposals
//! of the generator from the components of steps that lowered the energy (kept in a replay
//! buffer); the generator is trained to make its scaled proposals indistinguishable from
//! those (non-saturating loss), so it learns how long successful steps were relative to the
//! gradients. The applied step is `mix_ratio` times the scaled step plus `1 - mix_ratio` times
//! the proposed one, limited to the norm `step_clip`. Scale factors are positive, so no
//! component changes its direction, and the untrained generator returns exactly 1.
//!
//! Networks: generator 2-16-16-1, discriminator 2-16-1 (`tanh`), Adam with learning rates
//! `2e-4` and `1e-4` and `beta_1 = 0.5`, five training steps per iteration on batches of 16
//! real and 16 generated samples, from a seeded generator (reproducible runs).
//!
//! In MultiOptPy's `GANStep` the generator receives no gradient: its output is detached with
//! `.item()` before it reaches the discriminator, so the generator keeps its random initial
//! weights and the method scales each coordinate by a fixed random function. The discriminator
//! is also given the energy change of the step as an input, which labels the samples by
//! itself, and the absolute coordinates, which depend on the frame. See
//! `docs/source/migration_optimizers.md`.
//!
//! MultiOptPy: `Optimizer/gan_step.py:GANStep`.
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::collections::VecDeque;

use faer::Col;

use super::StepEnhancer;
use crate::core_infra::rng::Rng;
use crate::error::MoptResult;
use crate::math::linalg::{col_norm, col_rms};
use crate::math::mlp::{AdamState, Mlp};
use crate::math::stable::{logistic, softplus};
use crate::opt::step::StepInput;

/// Parameters of the GAN step.
#[derive(Debug, Clone, Copy)]
pub struct GanStepParams {
    /// Smallest scale factor.
    pub min_scale: f64,
    /// Largest scale factor.
    pub max_scale: f64,
    /// Largest norm of the returned step.
    pub step_clip: f64,
    /// Weight of the scaled step in the returned step.
    pub mix_ratio: f64,
    /// Real samples needed before training starts.
    pub min_samples: usize,
    /// Training steps per iteration.
    pub training_steps: usize,
    /// Samples of each kind per training step.
    pub batch: usize,
    /// Capacity of the replay buffers.
    pub buffer: usize,
    /// Learning rate of the generator.
    pub generator_rate: f64,
    /// Learning rate of the discriminator.
    pub discriminator_rate: f64,
    /// Seed of the random generator.
    pub seed: u64,
}

impl Default for GanStepParams {
    fn default() -> Self {
        Self {
            min_scale: 0.2,
            max_scale: 3.0,
            step_clip: 0.5,
            mix_ratio: 0.7,
            min_samples: 10,
            training_steps: 5,
            batch: 16,
            buffer: 1000,
            generator_rate: 2e-4,
            discriminator_rate: 1e-4,
            seed: 0,
        }
    }
}

/// GAN step enhancer.
#[derive(Debug, Clone)]
pub struct GanStep {
    params: GanStepParams,
    generator: Mlp,
    discriminator: Mlp,
    gen_adam: AdamState,
    dis_adam: AdamState,
    rng: Rng,
    /// Features of step components that lowered the energy.
    good: VecDeque<[f64; 2]>,
    /// Features of step components that raised it.
    bad: VecDeque<[f64; 2]>,
    /// Gradient and energy of the previous point.
    last: Option<(Col<f64>, f64)>,
    trained_steps: usize,
}

fn features(gradient: &Col<f64>, step: &Col<f64>) -> Vec<[f64; 2]> {
    let (gr, sr) = (col_rms(gradient), col_rms(step));
    if gr == 0.0 || sr == 0.0 || !gr.is_finite() || !sr.is_finite() {
        return Vec::new();
    }
    (0..step.nrows())
        .map(|i| [gradient[i] / gr, step[i] / sr])
        .collect()
}

impl GanStep {
    /// New enhancer with freshly initialized networks.
    pub fn new(params: GanStepParams) -> Self {
        let mut rng = Rng::from_seed(params.seed);
        let mut generator = Mlp::new(&[2, 16, 16, 1], &mut rng);
        // Scale 1 before any training: logistic(b) = (1 - min) / (max - min).
        let t = ((1.0 - params.min_scale) / (params.max_scale - params.min_scale))
            .clamp(1e-6, 1.0 - 1e-6);
        generator.constant_output(libm::log(t / (1.0 - t)));
        let discriminator = Mlp::new(&[2, 16, 1], &mut rng);
        let gen_adam = generator.adam(params.generator_rate, 0.5, 0.999);
        let dis_adam = discriminator.adam(params.discriminator_rate, 0.5, 0.999);
        Self {
            params,
            generator,
            discriminator,
            gen_adam,
            dis_adam,
            rng,
            good: VecDeque::new(),
            bad: VecDeque::new(),
            last: None,
            trained_steps: 0,
        }
    }

    /// Scale factor of the generator for one feature pair, with its derivative with respect
    /// to the network output.
    fn scale_of(&self, output: f64) -> (f64, f64) {
        let p = &self.params;
        let s = logistic(output);
        let width = p.max_scale - p.min_scale;
        (p.min_scale + width * s, width * s * (1.0 - s))
    }

    /// Scale factors the generator assigns to the components of `step`.
    pub fn scales(&self, gradient: &Col<f64>, step: &Col<f64>) -> Vec<f64> {
        let f = features(gradient, step);
        if f.is_empty() {
            return vec![1.0; step.nrows()];
        }
        f.iter()
            .map(|x| self.scale_of(self.generator.forward(x)[0]).0)
            .collect()
    }

    /// Number of training steps performed so far.
    pub fn trained_steps(&self) -> usize {
        self.trained_steps
    }

    /// Number of samples from steps that lowered and that raised the energy.
    pub fn samples(&self) -> (usize, usize) {
        (self.good.len(), self.bad.len())
    }

    fn sample(&mut self, from_good: bool, fallback: &[[f64; 2]]) -> Option<[f64; 2]> {
        let buf = if from_good { &self.good } else { &self.bad };
        if !buf.is_empty() {
            let k = self.rng.gen_range_usize(buf.len());
            return buf.get(k).copied();
        }
        if fallback.is_empty() {
            return None;
        }
        let k = self.rng.gen_range_usize(fallback.len());
        fallback.get(k).copied()
    }

    /// One training step of both networks; `current` are the features of the present proposal
    /// (the generator's inputs when no unsuccessful step has been seen).
    fn train(&mut self, current: &[[f64; 2]]) {
        let mut d_grad = self.discriminator.zero_gradient();
        let mut g_grad = self.generator.zero_gradient();
        let batch = self.params.batch.max(1);
        for _ in 0..batch {
            // Discriminator on a real sample: label 1, loss softplus(-z).
            if let Some(real) = self.sample(true, &[]) {
                let t = self.discriminator.trace(&real);
                let z = t.output[0];
                self.discriminator
                    .backward(&t, &[logistic(z) - 1.0], &mut d_grad);
            }
            // Generated sample: a proposal scaled by the generator.
            let Some(x) = self.sample(false, current) else {
                continue;
            };
            let gt = self.generator.trace(&x);
            let (scale, d_scale) = self.scale_of(gt.output[0]);
            let fake = [x[0], x[1] * scale];
            let dt = self.discriminator.trace(&fake);
            let z = dt.output[0];
            // Discriminator: label 0, loss softplus(z).
            self.discriminator
                .backward(&dt, &[logistic(z)], &mut d_grad);
            // Generator: non-saturating loss softplus(-z); the gradient reaches the generator
            // through the scaled step component.
            let mut unused = self.discriminator.zero_gradient();
            let d_in = self
                .discriminator
                .backward(&dt, &[logistic(z) - 1.0], &mut unused);
            self.generator
                .backward(&gt, &[d_in[1] * x[1] * d_scale], &mut g_grad);
        }
        self.discriminator.adam_step(&mut self.dis_adam, &d_grad);
        self.generator.adam_step(&mut self.gen_adam, &g_grad);
        self.trained_steps += 1;
    }

    /// Discriminator loss of a real and a generated sample (for monitoring).
    pub fn discriminator_loss(&self, real: [f64; 2], generated: [f64; 2]) -> f64 {
        softplus(-self.discriminator.forward(&real)[0])
            + softplus(self.discriminator.forward(&generated)[0])
    }
}

impl Default for GanStep {
    fn default() -> Self {
        Self::new(GanStepParams::default())
    }
}

impl StepEnhancer for GanStep {
    fn name(&self) -> &str {
        "gan_step"
    }

    fn enhance(&mut self, input: &StepInput<'_>, step: Col<f64>) -> MoptResult<Col<f64>> {
        let p = self.params;
        // Label the step that led here by the energy change it produced.
        if let (Some((g_prev, e_prev)), Some(prev)) = (self.last.take(), input.previous) {
            if g_prev.nrows() == prev.step.nrows() {
                let lowered = input.energy.value() <= e_prev;
                let buf = if lowered {
                    &mut self.good
                } else {
                    &mut self.bad
                };
                for f in features(&g_prev, &prev.step) {
                    buf.push_back(f);
                    if buf.len() > p.buffer.max(1) {
                        buf.pop_front();
                    }
                }
            }
        }
        self.last = Some((input.gradient.clone(), input.energy.value()));
        let current = features(input.gradient, &step);
        if current.is_empty() || self.good.len() < p.min_samples {
            return Ok(step);
        }
        for _ in 0..p.training_steps {
            self.train(&current);
        }
        let scales = self.scales(input.gradient, &step);
        let mut out = Col::from_fn(step.nrows(), |i| {
            let s = if scales[i].is_finite() {
                scales[i]
            } else {
                1.0
            };
            step[i] * (p.mix_ratio * s + (1.0 - p.mix_ratio))
        });
        let len = col_norm(&out);
        if !len.is_finite() {
            return Ok(step);
        }
        if len > p.step_clip {
            out *= p.step_clip / len;
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
    use crate::opt::step::PreviousPoint;
    use crate::opt::stepper::plain_input;
    use faer::Mat;

    #[test]
    fn untrained_generator_is_the_identity_and_scales_stay_in_range() {
        let gan = GanStep::default();
        let g = Col::from_fn(3, |i| [0.2, -0.1, 0.05][i]);
        let s = Col::from_fn(3, |i| [-0.02, 0.01, -0.005][i]);
        for v in gan.scales(&g, &s) {
            assert!((v - 1.0).abs() < 1e-12);
        }
        // Zero gradient or step: no features, unit scales.
        assert_eq!(gan.scales(&Col::zeros(3), &s), vec![1.0; 3]);
        let h = Mat::<f64>::identity(3, 3);
        let x = Col::zeros(3);
        let mut gan = GanStep::default();
        let out = gan
            .enhance(&plain_input(0, &x, 0.0, &g, &h, None, 0), s.clone())
            .unwrap();
        assert_eq!(out, s);
    }

    #[test]
    fn the_generator_learns_the_scale_of_successful_steps() {
        // Successful steps are half as long (relative to the gradient) as the proposals, which
        // all fail: the generator should shrink its proposals.
        let mut gan = GanStep::new(GanStepParams {
            generator_rate: 5e-3,
            discriminator_rate: 5e-3,
            ..GanStepParams::default()
        });
        let mut rng = Rng::from_seed(11);
        for _ in 0..200 {
            let g = rng.normal(0.0, 1.0);
            gan.good.push_back([g, -0.5 * g]);
            gan.bad.push_back([g, -1.5 * g]);
        }
        let current: Vec<[f64; 2]> = (0..8)
            .map(|i| [0.3 * i as f64 - 1.0, 1.5 * (1.0 - 0.3 * i as f64)])
            .collect();
        let before = gan.discriminator_loss([1.0, -0.5], [1.0, -1.5]);
        for _ in 0..600 {
            gan.train(&current);
        }
        assert_eq!(gan.trained_steps(), 600);
        assert!(gan.discriminator_loss([1.0, -0.5], [1.0, -1.5]) < before);
        let g = Col::from_fn(4, |i| [1.0, -1.0, 0.5, -0.5][i]);
        let s = Col::from_fn(4, |i| [-1.5, 1.5, -0.75, 0.75][i]);
        let scales = gan.scales(&g, &s);
        let mean = scales.iter().sum::<f64>() / 4.0;
        assert!(mean < 0.9, "scales {scales:?}");
        assert!(scales.iter().all(|v| (0.2..=3.0).contains(v)));
    }

    #[test]
    fn steps_are_labelled_by_the_energy_change_and_training_starts_with_enough_samples() {
        let h = Mat::<f64>::identity(4, 4);
        let mut gan = GanStep::default();
        let mut x = Col::from_fn(4, |i| 1.0 + i as f64);
        let mut prev: Option<PreviousPoint> = None;
        for it in 0..8 {
            let g = x.clone();
            let e = 0.5 * x.iter().map(|v| v * v).sum::<f64>();
            let s = gan
                .enhance(
                    &plain_input(it, &x, e, &g, &h, prev.as_ref(), 0),
                    &g * (-0.1),
                )
                .unwrap();
            prev = Some(PreviousPoint {
                energy: crate::domain::Quantity::new(e),
                gradient: g.clone(),
                raw_gradient: g,
                step: s.clone(),
            });
            x += &s;
        }
        let (good, bad) = gan.samples();
        assert_eq!((good, bad), (28, 0));
        assert!(gan.trained_steps() > 0);
        // Same seed, same result.
        let a = GanStep::default().generator.forward(&[0.3, 0.4]);
        let b = GanStep::default().generator.forward(&[0.3, 0.4]);
        assert_eq!(a, b);
    }
}
