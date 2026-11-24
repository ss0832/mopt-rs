//! A small fully connected neural network with `tanh` hidden layers and a linear output layer,
//! with reverse-mode gradients and the Adam update (Kingma and Ba, arXiv:1412.6980).
//!
//! Weights are initialized uniformly in `+-sqrt(6 / (n_in + n_out))` (Glorot and Bengio, Proc.
//! AISTATS 2010) from a seeded generator, so that training is reproducible.
//!
//! MultiOptPy: no counterpart (MultiOptPy uses PyTorch modules in `Optimizer/gan_step.py` and
//! `Optimizer/rl_step.py`).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use crate::core_infra::rng::Rng;

#[derive(Debug, Clone)]
struct Layer {
    n_in: usize,
    n_out: usize,
    /// Row-major weights (`n_out` rows) followed by the biases.
    params: Vec<f64>,
}

impl Layer {
    fn apply(&self, x: &[f64]) -> Vec<f64> {
        (0..self.n_out)
            .map(|o| {
                let row = &self.params[o * self.n_in..(o + 1) * self.n_in];
                row.iter().zip(x).map(|(w, v)| w * v).sum::<f64>()
                    + self.params[self.n_out * self.n_in + o]
            })
            .collect()
    }
}

/// Fully connected network.
#[derive(Debug, Clone)]
pub struct Mlp {
    layers: Vec<Layer>,
}

/// Activations of one forward pass, kept for the backward pass.
#[derive(Debug, Clone)]
pub struct Trace {
    /// Input of every layer (the network input first).
    inputs: Vec<Vec<f64>>,
    /// Network output.
    pub output: Vec<f64>,
}

/// Gradient of a loss with respect to all parameters, in the layout of the network.
#[derive(Debug, Clone)]
pub struct MlpGradient(Vec<Vec<f64>>);

/// State of the Adam optimizer for one network.
#[derive(Debug, Clone)]
pub struct AdamState {
    m: Vec<Vec<f64>>,
    v: Vec<Vec<f64>>,
    step: i32,
    /// Learning rate.
    pub learning_rate: f64,
    /// Decay of the first moment.
    pub beta1: f64,
    /// Decay of the second moment.
    pub beta2: f64,
}

impl Mlp {
    /// Network with the layer sizes `sizes` (input, hidden..., output).
    pub fn new(sizes: &[usize], rng: &mut Rng) -> Self {
        let layers = sizes
            .windows(2)
            .map(|w| {
                let (n_in, n_out) = (w[0], w[1]);
                let bound = libm::sqrt(6.0 / (n_in + n_out) as f64);
                let mut params: Vec<f64> = (0..n_in * n_out)
                    .map(|_| rng.uniform(-bound, bound))
                    .collect();
                params.extend(std::iter::repeat_n(0.0, n_out));
                Layer {
                    n_in,
                    n_out,
                    params,
                }
            })
            .collect();
        Self { layers }
    }

    /// Set the weights of the output layer to zero and its biases to `bias`, so that the
    /// untrained network returns `bias` for every input.
    pub fn constant_output(&mut self, bias: f64) {
        if let Some(last) = self.layers.last_mut() {
            let w = last.n_in * last.n_out;
            last.params[..w].iter_mut().for_each(|p| *p = 0.0);
            last.params[w..].iter_mut().for_each(|p| *p = bias);
        }
    }

    /// Output for the input `x`.
    pub fn forward(&self, x: &[f64]) -> Vec<f64> {
        self.trace(x).output
    }

    /// Forward pass keeping what the backward pass needs.
    pub fn trace(&self, x: &[f64]) -> Trace {
        let mut inputs = Vec::with_capacity(self.layers.len());
        let mut a = x.to_vec();
        let last = self.layers.len().saturating_sub(1);
        for (k, layer) in self.layers.iter().enumerate() {
            let z = layer.apply(&a);
            inputs.push(a);
            a = if k < last {
                z.into_iter().map(libm::tanh).collect()
            } else {
                z
            };
        }
        Trace { inputs, output: a }
    }

    /// A zero gradient of the right shape.
    pub fn zero_gradient(&self) -> MlpGradient {
        MlpGradient(
            self.layers
                .iter()
                .map(|l| vec![0.0; l.params.len()])
                .collect(),
        )
    }

    /// Add to `grad` the gradient of a loss whose derivative with respect to the output of the
    /// traced pass is `d_output`; returns the derivative with respect to the input.
    pub fn backward(&self, trace: &Trace, d_output: &[f64], grad: &mut MlpGradient) -> Vec<f64> {
        let mut delta = d_output.to_vec();
        for (k, layer) in self.layers.iter().enumerate().rev() {
            let input = &trace.inputs[k];
            let g = &mut grad.0[k];
            let mut d_in = vec![0.0; layer.n_in];
            for o in 0..layer.n_out {
                let d = delta[o];
                for i in 0..layer.n_in {
                    g[o * layer.n_in + i] += d * input[i];
                    d_in[i] += d * layer.params[o * layer.n_in + i];
                }
                g[layer.n_out * layer.n_in + o] += d;
            }
            // The input of layer k > 0 is tanh of the previous pre-activation.
            delta = if k > 0 {
                d_in.iter()
                    .zip(input)
                    .map(|(d, a)| d * (1.0 - a * a))
                    .collect()
            } else {
                d_in
            };
        }
        delta
    }

    /// Adam optimizer state for this network.
    pub fn adam(&self, learning_rate: f64, beta1: f64, beta2: f64) -> AdamState {
        let zeros: Vec<Vec<f64>> = self
            .layers
            .iter()
            .map(|l| vec![0.0; l.params.len()])
            .collect();
        AdamState {
            m: zeros.clone(),
            v: zeros,
            step: 0,
            learning_rate,
            beta1,
            beta2,
        }
    }

    /// One Adam step along `-grad` (a descent step for the loss whose gradient was collected).
    pub fn adam_step(&mut self, state: &mut AdamState, grad: &MlpGradient) {
        state.step = state.step.saturating_add(1);
        let c1 = 1.0 - state.beta1.powi(state.step);
        let c2 = 1.0 - state.beta2.powi(state.step);
        for (k, layer) in self.layers.iter_mut().enumerate() {
            for (i, p) in layer.params.iter_mut().enumerate() {
                let g = grad.0[k][i];
                if !g.is_finite() {
                    continue;
                }
                let m = &mut state.m[k][i];
                let v = &mut state.v[k][i];
                *m = state.beta1 * *m + (1.0 - state.beta1) * g;
                *v = state.beta2 * *v + (1.0 - state.beta2) * g * g;
                *p -= state.learning_rate * (*m / c1) / (libm::sqrt(*v / c2) + 1e-8);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gradients_match_finite_differences_and_training_reduces_the_loss() {
        let mut rng = Rng::from_seed(3);
        let mut net = Mlp::new(&[2, 5, 4, 1], &mut rng);
        let x = [0.3, -0.7];
        // Loss = output^2 / 2.
        let loss = |n: &Mlp| 0.5 * n.forward(&x)[0].powi(2);
        let trace = net.trace(&x);
        let mut grad = net.zero_gradient();
        let d_in = net.backward(&trace, &[trace.output[0]], &mut grad);
        let h = 1e-6;
        for k in 0..net.layers.len() {
            for i in 0..net.layers[k].params.len() {
                let mut plus = net.clone();
                plus.layers[k].params[i] += h;
                let mut minus = net.clone();
                minus.layers[k].params[i] -= h;
                let fd = (loss(&plus) - loss(&minus)) / (2.0 * h);
                assert!((fd - grad.0[k][i]).abs() < 1e-8, "layer {k}, parameter {i}");
            }
        }
        for i in 0..2 {
            let mut xp = x;
            xp[i] += h;
            let mut xm = x;
            xm[i] -= h;
            let fd =
                (0.5 * net.forward(&xp)[0].powi(2) - 0.5 * net.forward(&xm)[0].powi(2)) / (2.0 * h);
            assert!((fd - d_in[i]).abs() < 1e-8);
        }
        // Fit y = x0 - x1 on a few points.
        let data = [
            ([0.1, 0.2], -0.1),
            ([0.5, -0.3], 0.8),
            ([-0.4, 0.4], -0.8),
            ([0.9, 0.1], 0.8),
        ];
        let total = |n: &Mlp| -> f64 {
            data.iter()
                .map(|(x, y)| (n.forward(x)[0] - y).powi(2))
                .sum()
        };
        let before = total(&net);
        let mut adam = net.adam(0.02, 0.9, 0.999);
        for _ in 0..400 {
            let mut grad = net.zero_gradient();
            for (x, y) in &data {
                let t = net.trace(x);
                net.backward(&t, &[2.0 * (t.output[0] - y)], &mut grad);
            }
            net.adam_step(&mut adam, &grad);
        }
        assert!(
            total(&net) < 1e-3 * before.max(1e-3),
            "{} -> {}",
            before,
            total(&net)
        );
        net.constant_output(0.7);
        assert_eq!(net.forward(&[5.0, -3.0]), vec![0.7]);
    }
}
