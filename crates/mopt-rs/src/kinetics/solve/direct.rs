//! The direct solutions: `expm` (`p(t) = exp(t K) p0` by the subtraction-free uniformized
//! scaling and squaring of [`crate::kinetics::expm::expm_metzler`], the reference),
//! `spectral` (`p(t) = Pi^(1/2) U exp(Lambda t) U^T Pi^(-1/2) p0` from the eigenvectors of the
//! symmetrized generator, which needs detailed balance; components below about `1e-16` of the
//! largest are lost to cancellation, so it is a diagnostic, not a reference), and `steady`
//! (the stationary distribution reached from `p0`, from the full subtraction-free
//! elimination; the same at every time).
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Mat;

use super::{
    check_input, conservation, mat_vec, Evolution, KineticSolution, KineticSolver, SolveDiagnostics,
};
use crate::core_infra::rng::Rng;
use crate::error::{MoptError, MoptResult};
use crate::kinetics::contract::stationary;
use crate::kinetics::expm::{exp_and_integral, expm_metzler};
use crate::kinetics::generator::scaled_symmetric_eigen;

fn solution(
    name: &str,
    times: &[f64],
    populations: Vec<Vec<f64>>,
    p0: &[f64],
    notes: Vec<String>,
) -> KineticSolution {
    let diagnostics = times
        .iter()
        .zip(&populations)
        .map(|(&t, p)| {
            let (c, neg) = conservation(p, p0);
            SolveDiagnostics {
                time: t,
                conservation: c,
                most_negative: neg,
                ..SolveDiagnostics::default()
            }
        })
        .collect();
    KineticSolution {
        solver: name.to_string(),
        times: times.to_vec(),
        populations,
        diagnostics,
        superstates: vec![None; times.len()],
        notes,
    }
}

fn dropped_note(dropped: usize) -> Vec<String> {
    if dropped > 0 {
        vec![format!(
            "{dropped} rate constant(s) more than exp(-700) below the largest were dropped"
        )]
    } else {
        Vec::new()
    }
}

/// `exp(t K) p0`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Expm;

impl KineticSolver for Expm {
    fn name(&self) -> &str {
        "expm"
    }

    fn solve(
        &mut self,
        ev: &Evolution,
        p0: &[f64],
        times: &[f64],
        _rng: &mut Rng,
    ) -> MoptResult<KineticSolution> {
        let g = ev.generator()?;
        check_input(g.n, p0, times)?;
        let rates = g.rates();
        let k = rates.matrix();
        let mut out = Vec::with_capacity(times.len());
        for &t in times {
            let e = expm_metzler(&k, t)?;
            out.push(mat_vec(&e, p0));
        }
        Ok(solution(
            "expm",
            times,
            out,
            p0,
            dropped_note(rates.dropped),
        ))
    }

    fn residence(
        &mut self,
        ev: &Evolution,
        p0: &[f64],
        t: f64,
        _rng: &mut Rng,
    ) -> MoptResult<Vec<f64>> {
        let g = ev.generator()?;
        check_input(g.n, p0, &[t])?;
        let k = g.rates().matrix();
        Ok(exp_and_integral(&k, t, p0)?.1)
    }
}

/// The spectral solution of a generator with detailed balance.
#[derive(Debug, Clone, Copy, Default)]
pub struct Spectral;

impl KineticSolver for Spectral {
    fn name(&self) -> &str {
        "spectral"
    }

    fn solve(
        &mut self,
        ev: &Evolution,
        p0: &[f64],
        times: &[f64],
        _rng: &mut Rng,
    ) -> MoptResult<KineticSolution> {
        let g = ev.generator()?;
        check_input(g.n, p0, times)?;
        let Some(ln_pi) = &g.ln_pi else {
            return Err(MoptError::config(
                "the solver spectral needs a generator with detailed balance",
            ));
        };
        let n = g.n;
        let rates = g.rates();
        let k = rates.matrix();
        // Pi^(-1/2) K Pi^(1/2): off-diagonal sqrt(K_ij K_ji).
        let sym = Mat::from_fn(n, n, |i, j| {
            if i == j {
                k[(i, i)]
            } else {
                (k[(i, j)] * k[(j, i)]).sqrt()
            }
        });
        let (lambda, u) = scaled_symmetric_eigen(&sym)?;
        let half: Vec<f64> = ln_pi.iter().map(|l| (0.5 * l).exp()).collect();
        let y: Vec<f64> = (0..n).map(|i| p0[i] / half[i]).collect();
        let coef: Vec<f64> = (0..n)
            .map(|m| (0..n).map(|i| u[(i, m)] * y[i]).sum())
            .collect();
        let mut out = Vec::with_capacity(times.len());
        for &t in times {
            let p = (0..n)
                .map(|i| {
                    half[i]
                        * (0..n)
                            .map(|m| u[(i, m)] * (lambda[m] * t).exp() * coef[m])
                            .sum::<f64>()
                })
                .collect();
            out.push(p);
        }
        Ok(solution(
            "spectral",
            times,
            out,
            p0,
            dropped_note(rates.dropped),
        ))
    }
}

/// The relaxation rates of a generator with detailed balance (`-` the eigenvalues of the
/// symmetrized generator, ascending; the first is 0 for every closed class).
pub fn relaxation_rates(ev: &Evolution) -> MoptResult<Vec<f64>> {
    let g = ev.generator()?;
    if g.ln_pi.is_none() {
        return Err(MoptError::config(
            "the spectrum needs a generator with detailed balance",
        ));
    }
    let n = g.n;
    let k = g.rates().matrix();
    let sym = Mat::from_fn(n, n, |i, j| {
        if i == j {
            -k[(i, i)]
        } else {
            -(k[(i, j)] * k[(j, i)]).sqrt()
        }
    });
    let (lambda, _) = scaled_symmetric_eigen(&sym)?;
    // The zero modes of the closed classes come out as rounding errors of the largest rate.
    let largest = (0..n).map(|m| lambda[m].abs()).fold(0.0, f64::max);
    let floor = 1e-14 * n as f64 * largest;
    Ok((0..n)
        .map(|m| if lambda[m] <= floor { 0.0 } else { lambda[m] })
        .collect())
}

/// The stationary distribution reached from `p0`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Steady;

impl KineticSolver for Steady {
    fn name(&self) -> &str {
        "steady"
    }

    fn solve(
        &mut self,
        ev: &Evolution,
        p0: &[f64],
        times: &[f64],
        _rng: &mut Rng,
    ) -> MoptResult<KineticSolution> {
        let g = ev.generator()?;
        check_input(g.n, p0, times)?;
        let rates = g.rates();
        let p = stationary(&rates, p0)?;
        let mut notes = dropped_note(rates.dropped);
        notes.push("steady: the stationary distribution (t -> infinity) at every time".into());
        Ok(solution("steady", times, vec![p; times.len()], p0, notes))
    }

    fn residence(
        &mut self,
        _ev: &Evolution,
        _p0: &[f64],
        _t: f64,
        _rng: &mut Rng,
    ) -> MoptResult<Vec<f64>> {
        Err(MoptError::config(
            "the solver steady has no time dependence, so it gives no residence times",
        ))
    }
}
