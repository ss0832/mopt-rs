//! Derivatives of the populations (yields) with respect to the parameters of the rates, by
//! forward-mode dual numbers through the code of the solvers (`kinetics_beyond_tst.md`,
//! section 3.10):
//!
//! - `dp_i(t) / dE_j` for every node `j`: the energy of a minimum (as the rate theory uses it:
//!   electronic, or the free energy) moves its weight `pi_j` and the flux of every saddle point
//!   that the consistent clamp has put at its level; the energy of a saddle point or crossing
//!   point moves its flux. Both directions of every step follow (flux form), so detailed
//!   balance holds for the perturbed rates; one direction alone is never changed.
//! - `dp_i(t) / d ln k_e` for every step `e`: both rate constants of the step scaled together
//!   (`ln F_e` moved, the equilibrium constant kept); `d ln p_i / d ln k_e` is the degree of rate
//!   control of the yield.
//! - `dp_i(t) / dT`: the tangents of `ln k` in `T` are a central difference of the rate
//!   assembly (relative step 1e-5; the rate theories and corrections are not written for dual
//!   numbers), then forward mode through the solver.
//! - `dp_i(t) / d ln [S]` for every species held at a concentration (pseudo-first order): its
//!   chemical potential moves by `k_B T` on the free side of its steps, so the step from that
//!   side scales with `[S]` and its reverse stays; `d ln p_i / d ln [S]` is the apparent
//!   reaction order of the yield. `dp_i / dG°_S` follows as `(1 / k_B T) dp_i / d ln [S]`.
//!
//! Every tangent is that of `ln k` of both directions of a step from the free energies of its
//! two sides and its bottleneck ([`crate::kinetics::rate::StepRate::tangent`]), so networks
//! without a potential (a catalytic cycle that converts species) are differentiated too.
//!
//! The yields at the output times, the final yield (`t -> infinity`: the stationary
//! distribution reached from `p0`, from the full elimination), and the effective rate
//! constants between superstates (the off-diagonal entries of the superstate generator `G`;
//! homogeneous of degree one in the rate constants, so their degrees of rate control sum to 1
//! exactly) are differentiated.
//!
//! One tangent per pass, the passes in parallel: every pass runs the solver itself (its greedy
//! order and cut, taken from the real parts, are the same in all passes), so the derivatives
//! are those of the numbers it reports, at every time and for every state in one pass per
//! parameter. The networks of the mapper have tens to hundreds of parameters, and the passes
//! are independent; an adjoint would need a transposed solver for every target. The adjoint of
//! the exact exponential (`sensitivity::rate_control`) checks the result. Derivatives are
//! provided for `propagate` (closure A), `rcmc`, `expm`, and `steady`.
//!
//! Units: energies in Hartree (`dp/dE` in 1/Hartree), `ln k` without unit, `T` in K.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use rayon::prelude::*;

use super::conditions::Clamp;
use super::contract::{greedy_order, Contraction, Eliminator, GreedyOrder, TimeRule};
use super::expm::expm_metzler_scalar;
use super::generator::LN_FLOOR;
use super::network::KineticNetwork;
use super::rate::RateSet;
use super::solve::propagate::Propagate;
use super::solve::Closure;
use super::{initial_populations, prepare, KineticsParts, KineticsSettings};
use crate::error::{MoptError, MoptResult};
use crate::math::ad::{Dual, Scalar};

/// Relative step of the central difference of the rates in the temperature.
pub const TEMPERATURE_STEP: f64 = 1e-5;

/// Step (in `k_B T`) of the central difference of the corrections in an energy.
pub const ENERGY_STEP: f64 = 1e-4;

/// A parameter of the rates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Parameter {
    /// The energy of a state (index into [`KineticNetwork::states`]).
    StateEnergy(usize),
    /// The energy of a bottleneck (index into [`KineticNetwork::bottlenecks`]).
    BottleneckEnergy(usize),
    /// `ln k` of the step of a bottleneck, both directions together.
    LnK(usize),
    /// The temperature.
    Temperature,
    /// `ln [S]` of a species held at a concentration (index into
    /// [`KineticNetwork::species`]): `d ln p / d ln [S]` is the apparent reaction order.
    LnConcentration(usize),
    /// The standard free energy `G°` of a species (index into [`KineticNetwork::species`]).
    SpeciesEnergy(usize),
}

impl Parameter {
    /// `state_energy`, `bottleneck_energy`, `ln_k`, `temperature`, `ln_concentration`, or
    /// `species_energy`.
    pub fn kind(self) -> &'static str {
        match self {
            Self::StateEnergy(_) => "state_energy",
            Self::BottleneckEnergy(_) => "bottleneck_energy",
            Self::LnK(_) => "ln_k",
            Self::Temperature => "temperature",
            Self::LnConcentration(_) => "ln_concentration",
            Self::SpeciesEnergy(_) => "species_energy",
        }
    }

    /// The unit of `dp / dtheta`.
    pub fn unit(self) -> &'static str {
        match self {
            Self::StateEnergy(_) | Self::BottleneckEnergy(_) | Self::SpeciesEnergy(_) => {
                "1/hartree"
            }
            Self::LnK(_) | Self::LnConcentration(_) => "1",
            Self::Temperature => "1/K",
        }
    }
}

/// A parameter with its label and value.
#[derive(Debug, Clone, PartialEq)]
pub struct ParameterInfo {
    /// The parameter.
    pub parameter: Parameter,
    /// The node or step (`EQ3`, `TS12`, `T`).
    pub label: String,
    /// Its value: the energy (Hartree) the rate theory uses (after the clamp for a
    /// bottleneck), `ln k` (1/s) of the step from its first state to its second, or `T` (K).
    pub value: f64,
}

/// The derivatives of a solution.
#[derive(Debug, Clone, PartialEq)]
pub struct Derivatives {
    /// The solver.
    pub solver: String,
    /// The output times (s).
    pub times: Vec<f64>,
    /// Labels of the states of the model.
    pub labels: Vec<String>,
    /// `populations[k][i]` at `times[k]`.
    pub populations: Vec<Vec<f64>>,
    /// The final populations (`t -> infinity`).
    pub final_populations: Vec<f64>,
    /// The parameters.
    pub parameters: Vec<ParameterInfo>,
    /// `d_populations[m][k][i] = dp_i(times[k]) / dtheta_m`.
    pub d_populations: Vec<Vec<Vec<f64>>>,
    /// `d_final[m][i] = dp_i(inf) / dtheta_m`.
    pub d_final: Vec<Vec<f64>>,
    /// The superstate generator at every time where the solver contracts: the kept states and
    /// `G` (row-major; `G[a][b]`: from superstate `b` to `a`, 1/s).
    pub superstates: Vec<Option<(Vec<usize>, Vec<f64>)>>,
    /// `dG / dtheta_m` at every time where the solver contracts.
    pub d_superstates: Vec<Vec<Option<Vec<f64>>>>,
    /// `k_B T` (Hartree).
    pub kt: f64,
    /// Passes (one per parameter but those of the bottleneck energies, which follow from the
    /// passes of `ln k`).
    pub passes: usize,
}

/// The tangent of the rates: `d ln k` of both directions of every step.
#[derive(Debug, Clone, PartialEq)]
struct Seed {
    ln_k: Vec<[f64; 2]>,
}

/// What every pass needs.
struct Plan<'a> {
    rates: &'a RateSet,
    n: usize,
    p0: Vec<f64>,
    times: &'a [f64],
    solver: Solver,
    order: GreedyOrder,
    prefixes: Vec<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Solver {
    Propagate,
    Rcmc,
    Expm,
    Steady,
}

/// The rate matrix (`w[b * n + a]`: from `a` to `b`) of `rates` with the tangent `seed`, as the
/// floor of [`super::generator::Generator::rates`] drops entries.
fn dual_rates(rates: &RateSet, seed: &Seed) -> Vec<Dual> {
    let n = rates.states.len();
    let top = rates
        .steps
        .iter()
        .flat_map(|s| s.ln_k)
        .fold(f64::NEG_INFINITY, f64::max);
    let mut w = vec![Dual::constant(0.0); n * n];
    for (k, s) in rates.steps.iter().enumerate() {
        let [a, b] = s.states;
        for (d, (ln_k, from, to)) in [(s.ln_k[0], a, b), (s.ln_k[1], b, a)]
            .into_iter()
            .enumerate()
        {
            if ln_k - top < LN_FLOOR || ln_k < -745.0 {
                continue;
            }
            let v = ln_k.exp();
            w[to * n + from] += Dual {
                re: v,
                eps: v * seed.ln_k[k][d],
            };
        }
    }
    w
}

/// The full generator (with its diagonal) of the rates `w`.
fn with_diagonal<T: Scalar>(w: &[T], n: usize) -> Vec<T> {
    let mut k = w.to_vec();
    for j in 0..n {
        let mut out = T::zero();
        for i in 0..n {
            if i != j {
                out += k[i * n + j];
            }
        }
        k[j * n + j] = -out;
    }
    k
}

fn mat_vec<T: Scalar>(m: &[T], x: &[T], n: usize) -> Vec<T> {
    (0..n)
        .map(|i| (0..n).fold(T::zero(), |acc, j| acc + m[i * n + j] * x[j]))
        .collect()
}

/// The result of one pass: populations at the times, final populations, superstate matrices.
type Pass = (Vec<Vec<Dual>>, Vec<Dual>, Vec<Option<Vec<Dual>>>);

/// Snapshots of the dual elimination after `prefixes[k]` steps of `order` (one pass).
fn dual_snapshots(
    w: &[Dual],
    n: usize,
    order: &GreedyOrder,
    prefixes: &[usize],
) -> MoptResult<Vec<Contraction<Dual>>> {
    let mut wanted: Vec<usize> = prefixes.to_vec();
    wanted.sort_unstable();
    wanted.dedup();
    let mut e = Eliminator::from_weights(n, w.to_vec());
    let mut done: Vec<(usize, Contraction<Dual>)> = Vec::with_capacity(wanted.len());
    for &k in &wanted {
        while e.order().len() < k {
            let j = order.steps[e.order().len()].state;
            e.eliminate(j)?;
        }
        done.push((k, e.snapshot()));
    }
    Ok(prefixes
        .iter()
        .filter_map(|k| done.iter().find(|(m, _)| m == k).map(|(_, c)| c.clone()))
        .collect())
}

fn run_pass(plan: &Plan<'_>, seed: &Seed) -> MoptResult<Pass> {
    let n = plan.n;
    let w = dual_rates(plan.rates, seed);
    let p0: Vec<Dual> = plan.p0.iter().map(|&p| Dual::constant(p)).collect();
    let full = plan.order.steps.len();
    let mut prefixes = plan.prefixes.clone();
    prefixes.push(full);
    let snaps = dual_snapshots(&w, n, &plan.order, &prefixes)?;
    let (at_times, last) = snaps.split_at(plan.times.len());
    let final_c = last
        .first()
        .ok_or_else(|| MoptError::config("derivatives: no final contraction"))?;
    let final_p = final_c.unfold_a(&final_c.lump(&p0), n);
    let k_full = with_diagonal(&w, n);
    let mut pops = Vec::with_capacity(plan.times.len());
    let mut supers = Vec::with_capacity(plan.times.len());
    for ((&t, &k), c) in plan.times.iter().zip(&plan.prefixes).zip(at_times) {
        match plan.solver {
            Solver::Steady => {
                pops.push(final_p.clone());
                supers.push(None);
            }
            Solver::Expm => {
                let e = expm_metzler_scalar(&k_full, n, t)?;
                pops.push(mat_vec(&e, &p0, n));
                supers.push(None);
            }
            Solver::Propagate if k == 0 => {
                let e = expm_metzler_scalar(&k_full, n, t)?;
                pops.push(mat_vec(&e, &p0, n));
                supers.push(None);
            }
            Solver::Propagate => {
                let nt = c.kept.len();
                let g = c.generator_a_dense();
                let e = expm_metzler_scalar(&g, nt, t)?;
                let q = mat_vec(&e, &c.lump(&p0), nt);
                pops.push(c.unfold_a(&q, n));
                supers.push(Some(g));
            }
            Solver::Rcmc => {
                pops.push(c.unfold_a(&c.lump(&p0), n));
                supers.push(Some(c.generator_a_dense()));
            }
        }
    }
    Ok((pops, final_p, supers))
}

/// The seeds of the states, the steps, the species held at a concentration, and the
/// temperature. Every step's rate constants depend on the free energies of its own two sides
/// and its bottleneck only, so each seed moves both directions of a step consistently (each
/// step keeps balancing its reverse).
fn seeds(
    net: &KineticNetwork,
    s: &KineticsSettings,
    parts: &KineticsParts,
    rates: &RateSet,
) -> MoptResult<(Vec<Parameter>, Vec<Seed>)> {
    let (n, m) = (rates.states.len(), rates.steps.len());
    let kt = rates.kt;
    let mut params = Vec::new();
    let mut out = Vec::new();
    // The corrections (tunnelling through the barriers, ...) depend on the energies too: their
    // tangent is a central difference of the assembly with the stationary point shifted.
    let corrected = !s.corrections.is_empty();
    let same = |r: &RateSet| {
        r.states == rates.states
            && r.steps.len() == m
            && r.steps
                .iter()
                .zip(&rates.steps)
                .all(|(x, y)| x.bottleneck == y.bottleneck && x.states == y.states)
    };
    let factor_tangent = |shift: &dyn Fn(&mut KineticNetwork, f64)| -> MoptResult<Vec<f64>> {
        let h = ENERGY_STEP * kt;
        let at = |x: f64| -> MoptResult<RateSet> {
            let mut moved = net.clone();
            shift(&mut moved, x);
            Ok(prepare(&moved, s, parts)?.rates)
        };
        let (up, down) = (at(h)?, at(-h)?);
        if !same(&up) || !same(&down) {
            return Err(MoptError::config(
                "derivatives: the model changes its states or steps within the step of an energy",
            ));
        }
        let sum = |st: &super::rate::StepRate| st.flux.factors.iter().map(|(_, f)| f).sum::<f64>();
        Ok(up
            .steps
            .iter()
            .zip(&down.steps)
            .map(|(u, d)| (sum(u) - sum(d)) / (2.0 * h))
            .collect())
    };
    for g in 0..n {
        let mut ln_k: Vec<[f64; 2]> = rates
            .steps
            .iter()
            .map(|st| {
                let d = [
                    f64::from(u8::from(st.states[0] == g)),
                    f64::from(u8::from(st.states[1] == g)),
                ];
                st.tangent(d, 0.0, kt)
            })
            .collect();
        let k = rates.states[g];
        if corrected {
            let df = factor_tangent(&|x: &mut KineticNetwork, h: f64| {
                x.states[k].energy += h;
                if let Some(f) = &mut x.states[k].free_energy {
                    *f += h;
                }
            })?;
            for (t, f) in ln_k.iter_mut().zip(&df) {
                t[0] += f;
                t[1] += f;
            }
        }
        params.push(Parameter::StateEnergy(k));
        out.push(Seed { ln_k });
    }
    if corrected {
        // The bottlenecks get their own passes (their corrections move with them).
        for (e, st) in rates.steps.iter().enumerate() {
            let b = st.bottleneck;
            if net.bottlenecks[b].transfer.is_some() {
                continue;
            }
            let mut ln_k = vec![[0.0; 2]; m];
            ln_k[e] = st.tangent([0.0; 2], 1.0, kt);
            let df = factor_tangent(&|x: &mut KineticNetwork, h: f64| {
                x.bottlenecks[b].energy += h;
                if let Some(f) = &mut x.bottlenecks[b].free_energy {
                    *f += h;
                }
            })?;
            for (t, f) in ln_k.iter_mut().zip(&df) {
                t[0] += f;
                t[1] += f;
            }
            params.push(Parameter::BottleneckEnergy(b));
            out.push(Seed { ln_k });
        }
    }
    for (k, st) in rates.steps.iter().enumerate() {
        let mut ln_k = vec![[0.0; 2]; m];
        ln_k[k] = [1.0, 1.0];
        params.push(Parameter::LnK(st.bottleneck));
        out.push(Seed { ln_k });
    }
    // ln [S]: mu_S moves by k_B T on the free side of the steps of the species.
    let sp = &rates.species;
    if !sp.at_standard {
        for sidx in 0..sp.names.len() {
            if !sp.used[sidx] {
                continue;
            }
            let ln_k = rates
                .steps
                .iter()
                .map(|st| match st.partner {
                    Some((p, free)) if p == sidx => {
                        let mut d = [0.0; 2];
                        d[free] = kt;
                        st.tangent(d, 0.0, kt)
                    }
                    _ => [0.0; 2],
                })
                .collect();
            params.push(Parameter::LnConcentration(sidx));
            out.push(Seed { ln_k });
        }
    }
    // The temperature: a central difference of the assembly.
    let t = s.conditions.temperature;
    let h = TEMPERATURE_STEP * t;
    let at = |temperature: f64| -> MoptResult<RateSet> {
        let mut shifted = s.clone();
        shifted.conditions.temperature = temperature;
        Ok(prepare(net, &shifted, parts)?.rates)
    };
    let (up, down) = (at(t + h)?, at(t - h)?);
    if !same(&up) || !same(&down) {
        return Err(MoptError::config(
            "derivatives: the model changes its states or steps within the step of the \
             temperature",
        ));
    }
    out.push(Seed {
        ln_k: up
            .steps
            .iter()
            .zip(&down.steps)
            .map(|(u, d)| {
                [
                    (u.ln_k[0] - d.ln_k[0]) / (2.0 * h),
                    (u.ln_k[1] - d.ln_k[1]) / (2.0 * h),
                ]
            })
            .collect(),
    });
    params.push(Parameter::Temperature);
    Ok((params, out))
}

/// The derivatives of the populations of `net` from `shares` (`(node id, share)`) at `times`
/// by the solver of `s` (`propagate` with closure A, `rcmc`, `expm`, `steady`).
pub fn derivatives(
    net: &KineticNetwork,
    s: &KineticsSettings,
    parts: &KineticsParts,
    shares: &[(usize, f64)],
    times: &[f64],
) -> MoptResult<Derivatives> {
    let solver = match s.solver.as_str() {
        "propagate" if s.solver_settings.closure == Closure::A => Solver::Propagate,
        "rcmc" => Solver::Rcmc,
        "expm" => Solver::Expm,
        "steady" => Solver::Steady,
        other => {
            return Err(MoptError::unsupported(format!(
                "derivatives are provided for the solvers propagate (closure a), rcmc, expm, \
                 and steady, not for '{other}'{}",
                if other == "propagate" {
                    " with closure b"
                } else {
                    ""
                }
            )))
        }
    };
    if times.iter().any(|t| !(t.is_finite() && *t >= 0.0)) {
        return Err(MoptError::config(
            "derivatives: the output times must not be negative",
        ));
    }
    let prepared = prepare(net, s, parts)?;
    let rates = &prepared.rates;
    if s.conditions.clamp == Clamp::Legacy {
        return Err(MoptError::config(
            "derivatives need the clamp consistent or none; the clamp legacy clamps each \
             direction by itself",
        ));
    }
    if !matches!(prepared.evolution, super::solve::Evolution::Linear(_)) {
        return Err(MoptError::unsupported(
            "derivatives are provided for the linear model canonical (with pseudo-first-order \
             steps), not for mass_action",
        ));
    }
    let n = rates.states.len();
    let p0 = initial_populations(net, rates, shares)?;
    let (params, seeds) = seeds(net, s, parts, rates)?;
    // The discrete choices from the real parts: the greedy order and the cuts.
    let zero = Seed {
        ln_k: vec![[0.0; 2]; rates.steps.len()],
    };
    let w0: Vec<f64> = dual_rates(rates, &zero).iter().map(|d| d.re).collect();
    let real = super::generator::Rates {
        n,
        w: w0,
        dropped: 0,
    };
    let rule = if solver == Solver::Rcmc {
        s.solver_settings.time_rule
    } else {
        TimeRule::Gershgorin
    };
    let order = greedy_order(&real, rule, true)?;
    let prefixes: Vec<usize> = match solver {
        Solver::Propagate => {
            let p = Propagate::new(s.solver_settings.tolerance, Closure::A)?;
            times.iter().map(|&t| p.cut(&order, t)).collect()
        }
        Solver::Rcmc => times.iter().map(|&t| order.steps_until(t)).collect(),
        Solver::Expm | Solver::Steady => vec![0; times.len()],
    };
    let plan = Plan {
        rates,
        n,
        p0,
        times,
        solver,
        order,
        prefixes,
    };
    let passes: Vec<MoptResult<Pass>> =
        seeds.par_iter().map(|seed| run_pass(&plan, seed)).collect();
    let passes: Vec<Pass> = passes.into_iter().collect::<MoptResult<_>>()?;
    let first = passes
        .first()
        .ok_or_else(|| MoptError::config("derivatives: the model has no parameter"))?;
    let populations: Vec<Vec<f64>> = first
        .0
        .iter()
        .map(|p| p.iter().map(|d| d.re).collect())
        .collect();
    let final_populations: Vec<f64> = first.1.iter().map(|d| d.re).collect();
    let superstates: Vec<Option<(Vec<usize>, Vec<f64>)>> = first
        .2
        .iter()
        .zip(&plan.prefixes)
        .map(|(g, &k)| {
            g.as_ref().map(|g| {
                let kept: Vec<usize> = (0..n)
                    .filter(|i| !plan.order.steps[..k].iter().any(|st| st.state == *i))
                    .collect();
                (kept, g.iter().map(|d| d.re).collect())
            })
        })
        .collect();
    let eps_of = |p: &Pass| -> (Vec<Vec<f64>>, Vec<f64>, Vec<Option<Vec<f64>>>) {
        (
            p.0.iter()
                .map(|v| v.iter().map(|d| d.eps).collect())
                .collect(),
            p.1.iter().map(|d| d.eps).collect(),
            p.2.iter()
                .map(|g| g.as_ref().map(|g| g.iter().map(|d| d.eps).collect()))
                .collect(),
        )
    };
    let mut infos = Vec::new();
    let mut d_populations = Vec::new();
    let mut d_final = Vec::new();
    let mut d_superstates = Vec::new();
    for (param, pass) in params.iter().zip(&passes) {
        let (dp, dfin, dg) = eps_of(pass);
        let (label, value) = match *param {
            Parameter::StateEnergy(k) => {
                let g = rates.index_of(k).unwrap_or(0);
                (net.states[k].label.clone(), rates.state_energies[g])
            }
            Parameter::LnK(b) => {
                let st = rates
                    .steps
                    .iter()
                    .position(|x| x.bottleneck == b)
                    .unwrap_or(0);
                (net.bottlenecks[b].label.clone(), rates.steps[st].ln_k[0])
            }
            Parameter::Temperature => ("T".to_string(), s.conditions.temperature),
            Parameter::LnConcentration(sp) => (
                net.species[sp].name.clone(),
                rates.species.ln_ratio[sp] + rates.species.standard_concentration.ln(),
            ),
            Parameter::BottleneckEnergy(b) => (
                net.bottlenecks[b].label.clone(),
                rates
                    .steps
                    .iter()
                    .find(|x| x.bottleneck == b)
                    .map_or(f64::NAN, |x| x.energy),
            ),
            Parameter::SpeciesEnergy(_) => continue,
        };
        infos.push(ParameterInfo {
            parameter: *param,
            label,
            value,
        });
        d_populations.push(dp);
        d_final.push(dfin);
        d_superstates.push(dg);
    }
    // The energies of the bottlenecks: d/dE = -(1/kT) d/d ln F where the clamp is not active.
    let mut extra = Vec::new();
    for st in &rates.steps {
        // An electron transfer has no free energy of its own (it follows from its sides); with
        // corrections the bottlenecks have passes of their own.
        if net.bottlenecks[st.bottleneck].transfer.is_some()
            || infos
                .iter()
                .any(|p| p.parameter == Parameter::BottleneckEnergy(st.bottleneck))
        {
            continue;
        }
        let Some(m) = infos
            .iter()
            .position(|p| p.parameter == Parameter::LnK(st.bottleneck))
        else {
            continue;
        };
        let factor = if st.clamped { 0.0 } else { -1.0 / rates.kt };
        let scale = |v: &Vec<f64>| v.iter().map(|x| factor * x).collect::<Vec<f64>>();
        extra.push((
            ParameterInfo {
                parameter: Parameter::BottleneckEnergy(st.bottleneck),
                label: net.bottlenecks[st.bottleneck].label.clone(),
                value: st.energy,
            },
            d_populations[m].iter().map(scale).collect::<Vec<_>>(),
            scale(&d_final[m]),
            d_superstates[m]
                .iter()
                .map(|g| g.as_ref().map(scale))
                .collect::<Vec<_>>(),
        ));
    }
    // The standard free energies of the species: d/dG° = (1 / kT) d/d ln[S].
    for sidx in 0..rates.species.names.len() {
        let Some(m) = infos
            .iter()
            .position(|p| p.parameter == Parameter::LnConcentration(sidx))
        else {
            continue;
        };
        let factor = 1.0 / rates.kt;
        let scale = |v: &Vec<f64>| v.iter().map(|x| factor * x).collect::<Vec<f64>>();
        extra.push((
            ParameterInfo {
                parameter: Parameter::SpeciesEnergy(sidx),
                label: net.species[sidx].name.clone(),
                value: rates.species.energies[sidx],
            },
            d_populations[m].iter().map(scale).collect::<Vec<_>>(),
            scale(&d_final[m]),
            d_superstates[m]
                .iter()
                .map(|g| g.as_ref().map(scale))
                .collect::<Vec<_>>(),
        ));
    }
    for (info, dp, dfin, dg) in extra {
        infos.push(info);
        d_populations.push(dp);
        d_final.push(dfin);
        d_superstates.push(dg);
    }
    Ok(Derivatives {
        solver: s.solver.clone(),
        times: times.to_vec(),
        labels: rates.labels.clone(),
        populations,
        final_populations,
        parameters: infos,
        d_populations,
        d_final,
        superstates,
        d_superstates,
        kt: rates.kt,
        passes: seeds.len(),
    })
}

impl Derivatives {
    /// The index of the parameter `p`.
    pub fn index_of(&self, p: Parameter) -> Option<usize> {
        self.parameters.iter().position(|x| x.parameter == p)
    }
}
