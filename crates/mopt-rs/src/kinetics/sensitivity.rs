//! Degrees of rate control of a population (Campbell, ACS Catal. 2017, 7, 2770; Stegelmann,
//! Andreasen, Campbell, J. Am. Chem. Soc. 2009, 131, 8077) for the generator in flux form:
//!
//! ```text
//! X_e = d ln p_target(t) / d(-G_ts(e) / RT)  = d ln y / d ln F_e       (every bottleneck)
//! X_j = d ln p_target(t) / d(-G_j / RT)      = d ln y / d ln pi_j      (every state; F fixed)
//! ```
//!
//! All derivatives come from one exponential of a block matrix (Van Loan, IEEE Trans. Autom.
//! Control 1978, 23, 395): with `lambda(s) = exp((t - s) K^T) e_target` and `p(s) = exp(s K)
//! p0`, `dy/dtheta = int_0^t lambda^T (dK/dtheta) p ds = sum_ij (dK/dtheta)[i][j] Z[i][j]`
//! where `Z = int_0^t lambda(s) p(s)^T ds` is the upper-right block of
//! `exp([[K^T, e_target p0^T], [0, K^T]] t)`. Two exact identities check it: scaling every
//! rate constant scales time, so `sum_e X_e = t (dy/dt) / y`; shifting every free energy
//! changes nothing, so `sum_e X_e + sum_j X_j = 0`.
//!
//! With the consistent clamp a saddle point below a minimum takes the energy of that minimum,
//! so its flux moves with it.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Mat;

use super::expm::{expm_metzler, van_loan};
use super::rate::RateSet;
use super::solve::mat_vec;
use crate::error::{MoptError, MoptResult};

/// The degrees of rate control of one population.
#[derive(Debug, Clone, PartialEq)]
pub struct RateControl {
    /// The state (index of the generator).
    pub target: usize,
    /// The time.
    pub time: f64,
    /// Its population.
    pub value: f64,
    /// `X_e` of every step of the rate set, in its order.
    pub steps: Vec<f64>,
    /// `X_j` of every state.
    pub states: Vec<f64>,
    /// `t (dy/dt) / y`, which the `X_e` sum to.
    pub time_scaling: f64,
}

/// The degrees of rate control of `p_target(t)` from `p0`.
pub fn rate_control(rates: &RateSet, p0: &[f64], t: f64, target: usize) -> MoptResult<RateControl> {
    if !rates.detailed_balance {
        return Err(MoptError::config(
            "the degrees of rate control need rate constants in flux form (detailed balance); \
             the clamp legacy breaks it",
        ));
    }
    let g = rates.generator()?;
    let n = g.n;
    if target >= n || p0.len() != n {
        return Err(MoptError::config(
            "rate control: the target or the initial population does not fit the model",
        ));
    }
    if !(t > 0.0 && t.is_finite()) {
        return Err(MoptError::config("rate control: the time must be positive"));
    }
    let k = g.rates().matrix();
    let p = mat_vec(&expm_metzler(&k, t)?, p0);
    let y = p[target];
    if y.is_nan() || y <= 0.0 {
        return Err(MoptError::config(format!(
            "rate control: the population of {} at {t} s is zero",
            g.labels[target]
        )));
    }
    let kt = Mat::from_fn(n, n, |i, j| k[(j, i)]);
    let b = Mat::from_fn(n, n, |i, j| if i == target { p0[j] } else { 0.0 });
    let z = van_loan(&kt, &b, &kt, t)?;
    let step_derivative = |s: &crate::kinetics::rate::StepRate| -> f64 {
        let [a, c] = s.states;
        let (k_ca, k_ac) = (s.ln_k[0].exp(), s.ln_k[1].exp());
        k_ca * (z[(c, a)] - z[(a, a)]) + k_ac * (z[(a, c)] - z[(c, c)])
    };
    let steps: Vec<f64> = rates.steps.iter().map(|s| step_derivative(s) / y).collect();
    let mut states = vec![0.0; n];
    for (j, x) in states.iter_mut().enumerate() {
        let mut d = 0.0;
        for i in 0..n {
            if i != j {
                d += k[(i, j)] * (z[(j, j)] - z[(i, j)]);
            }
        }
        *x = d;
    }
    for s in rates.steps.iter().filter(|s| s.clamped) {
        for &j in &s.states {
            if s.energy == rates.state_energies[j] {
                states[j] += step_derivative(s);
                break;
            }
        }
    }
    for x in &mut states {
        *x /= y;
    }
    let rate: f64 = (0..n).map(|j| k[(target, j)] * p[j]).sum();
    Ok(RateControl {
        target,
        time: t,
        value: y,
        steps,
        states,
        time_scaling: t * rate / y,
    })
}
