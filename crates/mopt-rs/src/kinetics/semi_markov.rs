//! Memory: semi-Markov chains (`semi_markov`) and the contraction of memory kernels
//! (`kinetics_beyond_tst.md`, section 3.6).
//!
//! **Semi-Markov chains.** A state `j` with the mean residence time `tau_j` and the jump
//! probabilities `P[i][j]` has the Laplace-domain kernel `Khat(s) = (P - I) diag(s psi(s) / (1 -
//! psi(s)))`, `psi` the transform of its residence-time density. Expanded at small `s`,
//!
//! ```text
//! Khat(s) = K0 + s K1 + O(s^2),   K0 = (P - I) diag(1 / tau),   K1 = K0 diag(tau (cv^2 - 1) / 2)
//! ```
//!
//! (`cv^2 = <t^2> / tau^2 - 1`, the squared coefficient of variation; `cv = 1` for the
//! exponential of a Markov state, where `K1 = 0`). `K0` is the Markov limit: it has the exact
//! stationary distribution (`pi_j` proportional to the visits times `tau_j`) and, for a Markov
//! chain, the exact dynamics. The model `semi_markov` takes `tau` and `P` from the rate constants
//! of the network (or from a trajectory, `workflow::kinetics`), the shape `k` of a gamma
//! distribution of every residence time from the annotations (`residence_shape`, `cv^2 = 1 / k`),
//! solves the Markov limit with the linear solvers and samples the chain itself with `kmc`; the
//! first-order memory correction `K0 (I - K1)^-1` is reported next to it.
//!
//! **Kernel contraction** (a proposal of the design): for `s p(s) - p(0) = Khat(s) p(s)` with
//! `Khat = K0 + s K1`, eliminating the states `S` in the Laplace domain,
//! `D(s) = Khat_TT + Khat_TS (s - Khat_SS)^-1 Khat_ST`, gives
//!
//! ```text
//! D(0)  = K0_TT + K0_TS C,                 C = -K0_SS^-1 K0_ST,  B = -K0_TS K0_SS^-1
//! D'(0) = K1_TT + K1_TS C + B K1_ST + B K1_SS C - B C
//! G     = D(0) (I - D'(0))^-1
//! ```
//!
//! For a Markov generator (`K1 = 0`) `D'(0) = -B C`, `I - D'(0) = M`, and `G` is closure B of the
//! contraction (checked in the tests).
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::linalg::solvers::Solve;
use faer::Mat;

use super::network::KineticNetwork;
use super::rate::RateSet;
use super::solve::SemiMarkov;
use crate::error::{MoptError, MoptResult};

/// The blocks of a kernel contraction.
#[derive(Debug, Clone)]
pub struct KernelContraction {
    /// The kept states.
    pub kept: Vec<usize>,
    /// `D(0)`: the Markov limit of the contracted kernel.
    pub d0: Mat<f64>,
    /// `D'(0)`: its first-order memory.
    pub d1: Mat<f64>,
    /// `G = D(0) (I - D'(0))^-1`.
    pub g: Mat<f64>,
}

fn block(a: &Mat<f64>, rows: &[usize], cols: &[usize]) -> Mat<f64> {
    Mat::from_fn(rows.len(), cols.len(), |i, j| a[(rows[i], cols[j])])
}

fn inverse(a: &Mat<f64>) -> MoptResult<Mat<f64>> {
    let n = a.nrows();
    let lu = a.partial_piv_lu();
    let inv = lu.solve(Mat::<f64>::identity(n, n).as_ref());
    if inv.col_iter().all(|c| c.iter().all(|v| v.is_finite())) {
        Ok(inv)
    } else {
        Err(MoptError::linalg(
            "kernel contraction: a block of the kernel is singular",
        ))
    }
}

/// The contraction of the kernel `K0 + s K1` with the states `eliminated` taken out.
pub fn kernel_contraction(
    k0: &Mat<f64>,
    k1: &Mat<f64>,
    eliminated: &[usize],
) -> MoptResult<KernelContraction> {
    let n = k0.nrows();
    if k0.ncols() != n || k1.nrows() != n || k1.ncols() != n {
        return Err(MoptError::config(
            "kernel contraction: K0 and K1 must be square and of one size",
        ));
    }
    if eliminated.iter().any(|&s| s >= n) {
        return Err(MoptError::config(
            "kernel contraction: an eliminated state is not a state of the kernel",
        ));
    }
    let kept: Vec<usize> = (0..n).filter(|i| !eliminated.contains(i)).collect();
    let s = eliminated;
    let t = &kept;
    let nt = t.len();
    let (k0tt, k0ts, k0st, k0ss) = (
        block(k0, t, t),
        block(k0, t, s),
        block(k0, s, t),
        block(k0, s, s),
    );
    let (k1tt, k1ts, k1st, k1ss) = (
        block(k1, t, t),
        block(k1, t, s),
        block(k1, s, t),
        block(k1, s, s),
    );
    let (d0, d1) = if s.is_empty() {
        (k0tt, k1tt)
    } else {
        let inv = inverse(&k0ss)?;
        let c = -(&inv * &k0st);
        let b = -(&k0ts * &inv);
        let d0 = &k0tt + &k0ts * &c;
        let bc = &b * &c;
        let d1 = &k1tt + &k1ts * &c + &b * &k1st + &b * &k1ss * &c - &bc;
        (d0, d1)
    };
    let m = Mat::<f64>::identity(nt, nt) - &d1;
    let g = &d0 * &inverse(&m)?;
    Ok(KernelContraction { kept, d0, d1, g })
}

/// The memory kernel `K1 = K0 diag(tau (cv^2 - 1) / 2)` of a semi-Markov chain with the Markov
/// limit `k0` (with its diagonal) and the squared coefficients of variation `cv2`.
pub fn semi_markov_memory(k0: &Mat<f64>, cv2: &[f64]) -> Mat<f64> {
    let n = k0.nrows();
    Mat::from_fn(n, n, |i, j| {
        let out = -k0[(j, j)];
        if out > 0.0 {
            k0[(i, j)] * (cv2[j] - 1.0) / (2.0 * out)
        } else {
            0.0
        }
    })
}

/// The semi-Markov chain of the rate constants `rates` of `net` with the shapes of the gamma
/// distributions of the states (`residence_shape`, 1 by default).
pub fn semi_markov_of(net: &KineticNetwork, rates: &RateSet) -> MoptResult<SemiMarkov> {
    let generator = rates.generator()?;
    let shapes: Vec<f64> = rates
        .states
        .iter()
        .map(|&k| net.states[k].extra.residence_shape.unwrap_or(1.0))
        .collect();
    if shapes.iter().any(|k| !(*k > 0.0 && k.is_finite())) {
        return Err(MoptError::config(
            "semi_markov: the shape of a residence time (residence_shape) must be positive",
        ));
    }
    let k0 = generator.rates().matrix();
    let cv2: Vec<f64> = shapes.iter().map(|k| 1.0 / k).collect();
    let k1 = semi_markov_memory(&k0, &cv2);
    let n = k0.nrows();
    let memory = inverse(&(Mat::<f64>::identity(n, n) - &k1))
        .ok()
        .map(|inv| &k0 * &inv);
    let non_exponential = shapes.iter().filter(|k| (**k - 1.0).abs() > 1e-12).count();
    Ok(SemiMarkov {
        generator,
        shapes,
        memory,
        notes: vec![format!(
            "semi_markov: mean residence times and jumps from the rate constants; {non_exponential} \
             state(s) with gamma-distributed residence times (the linear solvers use the Markov \
             limit, kmc samples the chain)"
        )],
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::core_infra::rng::Rng;
    use crate::kinetics::contract::contract;
    use crate::kinetics::generator::Generator;
    use crate::kinetics::solve::{Evolution, KineticSolver, Kmc, Steady};

    fn dense(k: &[Vec<f64>]) -> Mat<f64> {
        Generator::from_matrix(k).unwrap().rates().matrix()
    }

    #[test]
    fn the_kernel_of_a_fast_block_expands_as_the_contraction_says() {
        // A slow pair with a fast state between them.
        let k = vec![
            vec![0.0, 50.0, 0.0, 0.0],
            vec![1.0, 0.0, 30.0, 0.1],
            vec![0.0, 40.0, 0.0, 0.0],
            vec![0.0, 0.2, 0.0, 0.0],
        ];
        let k0 = dense(&k);
        let zero = Mat::<f64>::zeros(4, 4);
        let kc = kernel_contraction(&k0, &zero, &[1]).unwrap();
        let rates = Generator::from_matrix(&k).unwrap().rates();
        let c = contract(&rates, &[1]).unwrap();
        // D'(0) = -(M - I) and G = closure B.
        let m = c.m_matrix();
        let (gb, _) = c.generator_b();
        for i in 0..3 {
            for j in 0..3 {
                let eye = if i == j { 1.0 } else { 0.0 };
                assert!((kc.d1[(i, j)] + (m[(i, j)] - eye)).abs() < 1e-12);
                assert!((kc.g[(i, j)] - gb[(i, j)]).abs() < 1e-10 * gb[(i, j)].abs().max(1.0));
                assert!((kc.d0[(i, j)] - c.d_matrix()[(i, j)]).abs() < 1e-10);
            }
        }
        // D(s) = D(0) + s D'(0) + O(s^2), from the resolvent at small s.
        let resolvent = |s: f64| {
            let t = [0usize, 2, 3];
            let ktt = block(&k0, &t, &t);
            let kts = block(&k0, &t, &[1]);
            let kst = block(&k0, &[1], &t);
            let kss = block(&k0, &[1], &[1]);
            let a = Mat::<f64>::identity(1, 1) * s - kss;
            ktt + kts * inverse(&a).unwrap() * kst
        };
        for s in [1e-3, 1e-4] {
            let d = resolvent(s);
            for i in 0..3 {
                for j in 0..3 {
                    let lin = kc.d0[(i, j)] + s * kc.d1[(i, j)];
                    assert!((d[(i, j)] - lin).abs() < 1e-5 * s, "{s}");
                }
            }
        }
    }

    #[test]
    fn a_semi_markov_chain_reaches_the_populations_of_its_markov_limit() {
        // Three states in a cycle with gamma residence times of shapes 0.5, 1, 4.
        let k = vec![
            vec![0.0, 0.0, 2.0],
            vec![1.0, 0.0, 0.0],
            vec![0.0, 3.0, 0.0],
        ];
        let generator = Generator::from_matrix(&k).unwrap();
        let ev = Evolution::SemiMarkov(SemiMarkov {
            generator: generator.clone(),
            shapes: vec![0.5, 1.0, 4.0],
            memory: None,
            notes: Vec::new(),
        });
        let mut rng = Rng::from_seed(11);
        let p0 = [1.0, 0.0, 0.0];
        let n = 20_000;
        let sampled = Kmc {
            trajectories: n,
            ..Kmc::default()
        }
        .solve(&ev, &p0, &[40.0], &mut rng)
        .unwrap();
        let exact = Steady
            .solve(&Evolution::Linear(generator), &p0, &[40.0], &mut rng)
            .unwrap();
        for (s, e) in sampled.populations[0].iter().zip(&exact.populations[0]) {
            let se = (e * (1.0 - e) / n as f64).sqrt();
            assert!((s - e).abs() < 3.0 * se, "{s} {e}");
        }
        // The memory kernel vanishes for exponential residence times.
        let k0 = dense(&k);
        let k1 = semi_markov_memory(&k0, &[1.0, 1.0, 1.0]);
        assert!(k1.col_iter().all(|c| c.iter().all(|v| *v == 0.0)));
    }
}
