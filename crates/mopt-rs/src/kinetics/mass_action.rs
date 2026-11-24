//! The model `mass_action` (`kinetics_beyond_tst.md`, stage 3): the states of the network and
//! the species of its bimolecular steps as concentrations (mol/L) under the law of mass action,
//!
//! ```text
//! a + S -> b :  rate = k_2 [a] [S],   k_2 = k(a -> b at c°) / c°   (L/mol/s)
//! b -> a + S :  rate = k(b -> a)                                     (1/s)
//! a -> b     :  rate = k(a -> b)                                     (1/s)
//! ```
//!
//! with the rate constants of the rate assembly at the standard state (every species at `c°`,
//! so that both directions balance at the standard state), and the stiff integration of the
//! non-linear equations by the solver `rosenbrock`: the L-stable two-stage Rosenbrock method
//! ROS2 (Verwer, Spee, Blom and Hundsdorfer, SIAM J. Sci. Comput. 1999, 20, 1456),
//!
//! ```text
//! (I - gamma h J) k1 = f(y)
//! (I - gamma h J) k2 = f(y + h k1) - 2 k1
//! y' = y + 3/2 h k1 + 1/2 h k2,   gamma = 1 + 1/sqrt(2)
//! ```
//!
//! with the embedded first-order solution `y + h k1` for the control of the step size. A linear
//! method keeps every linear invariant: the total of the states of each connected component
//! and of every species, free and bound, are conserved, which the diagnostics report. With
//! steps far beyond the fastest time scale the rounding of `I - gamma h J` (entries of
//! `h k` up to 1e15 and more) would lose them at the 1e-6 level, so one row of every stage is
//! replaced by its invariant `c^T k = 0` (an equivalent system, as `c^T f = 0` for every `y`),
//! which keeps them to rounding. What rounding still limits is the slow evolution next to a
//! fast equilibrium: the net rate is the difference of two large fluxes, so its relative error
//! is about `1e-16` times the ratio of the fast to the slow rate (the linear solvers do not
//! have this limit, as they never form that difference). The solver `rosenbrock` integrates the linear model `canonical` too (`dp/dt = K p`),
//! a check of the matrix exponential.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use super::rate::RateSet;
use super::solve::{check_input, Evolution, KineticSolution, KineticSolver, SolveDiagnostics};
use crate::core_infra::rng::Rng;
use crate::error::{MoptError, MoptResult};

/// The default relative tolerance of `rosenbrock`.
pub const DEFAULT_ODE_TOLERANCE: f64 = 1e-7;

/// The most steps of one integration.
pub const MAX_STEPS: usize = 2_000_000;

/// One reaction of mass action.
#[derive(Debug, Clone, PartialEq)]
pub struct Reaction {
    /// The entities it consumes (indices into [`MassActionSystem::labels`]).
    pub reactants: Vec<usize>,
    /// The entities it forms.
    pub products: Vec<usize>,
    /// `ln k` (`k` in 1/s for one reactant, L/mol/s for two).
    pub ln_k: f64,
    /// The step of the rate set it comes from.
    pub step: usize,
}

/// The equations of mass action of a network.
#[derive(Debug, Clone, PartialEq)]
pub struct MassActionSystem {
    /// The entities: the states of the model, then the species its steps use.
    pub labels: Vec<String>,
    /// How many of them are states.
    pub n_states: usize,
    /// The species of the rate set behind the entities after the states.
    pub species: Vec<usize>,
    /// The reactions.
    pub reactions: Vec<Reaction>,
    /// The initial concentration (mol/L) of the species (their concentrations given, 0 for
    /// one that is only formed).
    pub species_initial: Vec<f64>,
    /// The initial concentration (mol/L) of the states together (the initial populations
    /// times it).
    pub total: f64,
    /// The conserved totals: name and coefficients over the entities.
    pub conserved: Vec<(String, Vec<f64>)>,
}

impl MassActionSystem {
    /// The equations of the rate set `rates` (assembled at the standard state) with the states
    /// together at `total` mol/L.
    pub fn new(rates: &RateSet, total: Option<f64>) -> MoptResult<Self> {
        let total = total.ok_or_else(|| {
            MoptError::config(
                "the model mass_action needs the initial concentration of the states of the \
                 network together (--total-concentration, mol/L)",
            )
        })?;
        if !(total.is_finite() && total > 0.0) {
            return Err(MoptError::config(format!(
                "the total concentration must be positive (got {total} mol/L)"
            )));
        }
        let sp = &rates.species;
        if !sp.at_standard {
            return Err(MoptError::config(
                "mass action needs the rate constants at the standard state of the species",
            ));
        }
        let n = rates.states.len();
        let species: Vec<usize> = (0..sp.names.len()).filter(|&s| sp.used[s]).collect();
        let entity = |s: usize| species.iter().position(|&x| x == s).map(|p| n + p);
        let mut labels = rates.labels.clone();
        labels.extend(species.iter().map(|&s| sp.names[s].clone()));
        let ln_c0 = sp.standard_concentration.ln();
        let mut reactions = Vec::new();
        for (m, st) in rates.steps.iter().enumerate() {
            let [a, b] = st.states;
            match st.partner {
                None => {
                    reactions.push(Reaction {
                        reactants: vec![a],
                        products: vec![b],
                        ln_k: st.ln_k[0],
                        step: m,
                    });
                    reactions.push(Reaction {
                        reactants: vec![b],
                        products: vec![a],
                        ln_k: st.ln_k[1],
                        step: m,
                    });
                }
                Some((s, free)) => {
                    let e = entity(s).ok_or_else(|| {
                        MoptError::config("mass action: a partner species is not used")
                    })?;
                    let (f, bound) = (st.states[free], st.states[1 - free]);
                    reactions.push(Reaction {
                        reactants: vec![f, e],
                        products: vec![bound],
                        ln_k: st.ln_k[free] - ln_c0,
                        step: m,
                    });
                    reactions.push(Reaction {
                        reactants: vec![bound],
                        products: vec![f, e],
                        ln_k: st.ln_k[1 - free],
                        step: m,
                    });
                }
            }
        }
        let species_initial = species
            .iter()
            .map(|&s| sp.concentrations[s].unwrap_or(0.0))
            .collect();
        // The conserved totals: the states of every connected component, and every species
        // free and bound where the network has a potential.
        let dim = labels.len();
        let mut component = vec![usize::MAX; n];
        let mut count = 0;
        for root in 0..n {
            if component[root] != usize::MAX {
                continue;
            }
            component[root] = count;
            let mut stack = vec![root];
            while let Some(x) = stack.pop() {
                for st in &rates.steps {
                    let [a, b] = st.states;
                    for (p, q) in [(a, b), (b, a)] {
                        if p == x && component[q] == usize::MAX {
                            component[q] = count;
                            stack.push(q);
                        }
                    }
                }
            }
            count += 1;
        }
        let mut conserved = Vec::new();
        for k in 0..count {
            let mut row = vec![0.0; dim];
            let mut first = None;
            for g in 0..n {
                if component[g] == k {
                    row[g] = 1.0;
                    first.get_or_insert(g);
                }
            }
            if count == 1 {
                conserved.push(("states".to_string(), row));
            } else {
                let label = first.map_or("-", |g| rates.labels[g].as_str());
                conserved.push((format!("states with {label}"), row));
            }
        }
        if let Some(comp) = &sp.composition {
            for (p, &s) in species.iter().enumerate() {
                let mut row = vec![0.0; dim];
                row[n + p] = 1.0;
                for k in 0..count {
                    let top = (0..n)
                        .filter(|&g| component[g] == k)
                        .map(|g| comp[g][s])
                        .max()
                        .unwrap_or(0);
                    for g in (0..n).filter(|&g| component[g] == k) {
                        row[g] = f64::from(top - comp[g][s]);
                    }
                }
                conserved.push((format!("{} (free and bound)", sp.names[s]), row));
            }
        }
        Ok(Self {
            labels,
            n_states: n,
            species,
            reactions,
            species_initial,
            total,
            conserved,
        })
    }

    /// The number of entities.
    pub fn dim(&self) -> usize {
        self.labels.len()
    }

    /// The initial concentrations from the initial populations `p0` of the states.
    pub fn initial(&self, p0: &[f64]) -> MoptResult<Vec<f64>> {
        check_input(self.n_states, p0, &[])?;
        let mut y: Vec<f64> = p0.iter().map(|p| p * self.total).collect();
        y.extend(&self.species_initial);
        Ok(y)
    }

    /// The largest change of a conserved total from `y0` to `y`, relative to the largest
    /// initial concentration.
    pub fn conservation(&self, y0: &[f64], y: &[f64]) -> f64 {
        let scale = y0
            .iter()
            .copied()
            .fold(0.0, f64::max)
            .max(f64::MIN_POSITIVE);
        self.conserved
            .iter()
            .map(|(_, row)| {
                let dot = |v: &[f64]| row.iter().zip(v).map(|(a, b)| a * b).sum::<f64>();
                (dot(y) - dot(y0)).abs() / scale
            })
            .fold(0.0, f64::max)
    }
}

/// A system of ordinary differential equations `dy/dt = f(y)`.
pub trait Ode {
    /// The dimension.
    fn dim(&self) -> usize;
    /// `f(y)`.
    fn rhs(&self, y: &[f64], f: &mut [f64]);
    /// The Jacobian `J[i * n + k] = df_i / dy_k`.
    fn jacobian(&self, y: &[f64], j: &mut [f64]);
    /// Linear invariants `c^T y` (`c^T f = 0` for every `y`), each with a row of its own where
    /// `c` is not 0: the integrator puts `c^T k = 0` in that row of every stage, an equivalent
    /// system whose conservation does not suffer from the rounding of `h J` with large `h`.
    fn invariants(&self) -> Vec<(usize, Vec<f64>)> {
        Vec::new()
    }
}

impl Ode for MassActionSystem {
    fn dim(&self) -> usize {
        self.labels.len()
    }

    fn rhs(&self, y: &[f64], f: &mut [f64]) {
        f.iter_mut().for_each(|v| *v = 0.0);
        for r in &self.reactions {
            let rate = r.reactants.iter().fold(r.ln_k.exp(), |acc, &i| acc * y[i]);
            for &i in &r.reactants {
                f[i] -= rate;
            }
            for &i in &r.products {
                f[i] += rate;
            }
        }
    }

    fn jacobian(&self, y: &[f64], j: &mut [f64]) {
        let n = self.labels.len();
        j.iter_mut().for_each(|v| *v = 0.0);
        let k_of = |r: &Reaction| r.ln_k.exp();
        for r in &self.reactions {
            let k = k_of(r);
            for (q, &col) in r.reactants.iter().enumerate() {
                // d rate / d y_col: the product of the other reactants.
                let d = r
                    .reactants
                    .iter()
                    .enumerate()
                    .filter(|(p, _)| *p != q)
                    .fold(k, |acc, (_, &i)| acc * y[i]);
                for &i in &r.reactants {
                    j[i * n + col] -= d;
                }
                for &i in &r.products {
                    j[i * n + col] += d;
                }
            }
        }
    }

    fn invariants(&self) -> Vec<(usize, Vec<f64>)> {
        let mut used = vec![false; self.labels.len()];
        let mut out = Vec::new();
        // A state of every component for its total, the free species for its own.
        for (name, row) in &self.conserved {
            let pick = if name.contains("(free and bound)") {
                (self.n_states..row.len()).find(|&i| row[i] != 0.0 && !used[i])
            } else {
                (0..self.n_states).find(|&i| row[i] != 0.0 && !used[i])
            };
            if let Some(i) = pick {
                used[i] = true;
                out.push((i, row.clone()));
            }
        }
        out
    }
}

/// `dp/dt = K p` with a dense `K` (row-major).
struct LinearOde {
    n: usize,
    k: Vec<f64>,
}

impl Ode for LinearOde {
    fn dim(&self) -> usize {
        self.n
    }

    fn rhs(&self, y: &[f64], f: &mut [f64]) {
        for (i, v) in f.iter_mut().enumerate() {
            *v = (0..self.n).map(|k| self.k[i * self.n + k] * y[k]).sum();
        }
    }

    fn jacobian(&self, _y: &[f64], j: &mut [f64]) {
        j.copy_from_slice(&self.k);
    }

    fn invariants(&self) -> Vec<(usize, Vec<f64>)> {
        if self.n == 0 {
            return Vec::new();
        }
        vec![(0, vec![1.0; self.n])]
    }
}

/// Counts of an integration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct OdeStats {
    /// Accepted steps.
    pub steps: usize,
    /// Rejected steps.
    pub rejected: usize,
}

/// LU factors with partial pivoting of the `n x n` matrix `a` (row-major), in place.
fn lu_factor(a: &mut [f64], n: usize, piv: &mut [usize]) -> MoptResult<()> {
    for k in 0..n {
        let mut p = k;
        for i in k + 1..n {
            if a[i * n + k].abs() > a[p * n + k].abs() {
                p = i;
            }
        }
        piv[k] = p;
        if a[p * n + k] == 0.0 || !a[p * n + k].is_finite() {
            return Err(MoptError::config(
                "rosenbrock: the matrix of a stage is singular",
            ));
        }
        if p != k {
            for c in 0..n {
                a.swap(k * n + c, p * n + c);
            }
        }
        let pivot = a[k * n + k];
        for i in k + 1..n {
            let f = a[i * n + k] / pivot;
            a[i * n + k] = f;
            if f != 0.0 {
                for c in k + 1..n {
                    a[i * n + c] -= f * a[k * n + c];
                }
            }
        }
    }
    Ok(())
}

/// Solve with the factors of [`lu_factor`], in place.
fn lu_solve(lu: &[f64], n: usize, piv: &[usize], b: &mut [f64]) {
    for (k, &p) in piv.iter().enumerate().take(n) {
        b.swap(k, p);
    }
    for i in 0..n {
        let mut s = b[i];
        for k in 0..i {
            s -= lu[i * n + k] * b[k];
        }
        b[i] = s;
    }
    for i in (0..n).rev() {
        let mut s = b[i];
        for k in i + 1..n {
            s -= lu[i * n + k] * b[k];
        }
        b[i] = s / lu[i * n + i];
    }
}

/// `y` at `times` (ascending, from `t = 0`) by ROS2 with the relative tolerance `rtol` and the
/// absolute tolerance `atol`.
pub fn ros2(
    ode: &dyn Ode,
    y0: &[f64],
    times: &[f64],
    rtol: f64,
    atol: f64,
) -> MoptResult<(Vec<Vec<f64>>, OdeStats)> {
    let n = ode.dim();
    if y0.len() != n {
        return Err(MoptError::config(
            "rosenbrock: the initial values do not fit",
        ));
    }
    if !(rtol > 0.0 && rtol.is_finite() && atol > 0.0 && atol.is_finite()) {
        return Err(MoptError::config(
            "rosenbrock: the tolerances must be positive",
        ));
    }
    if times.windows(2).any(|w| w[1] < w[0]) || times.iter().any(|t| t.is_nan() || *t < 0.0) {
        return Err(MoptError::config(
            "rosenbrock: the output times must be ascending and not negative",
        ));
    }
    let gamma = 1.0 + std::f64::consts::FRAC_1_SQRT_2;
    let mut stats = OdeStats::default();
    let mut y = y0.to_vec();
    let mut t = 0.0_f64;
    let mut f0 = vec![0.0; n];
    let mut f1 = vec![0.0; n];
    let mut jac = vec![0.0; n * n];
    let mut w = vec![0.0; n * n];
    let mut piv = vec![0; n];
    let mut k1 = vec![0.0; n];
    let mut k2 = vec![0.0; n];
    let mut y1 = vec![0.0; n];
    let mut y_new = vec![0.0; n];
    ode.jacobian(&y, &mut jac);
    let norm = (0..n)
        .map(|i| (0..n).map(|k| jac[i * n + k].abs()).sum::<f64>())
        .fold(0.0, f64::max);
    let t_end = times.iter().copied().fold(0.0, f64::max);
    let mut h = if norm > 0.0 {
        (1e-4 / norm).min(t_end.max(f64::MIN_POSITIVE))
    } else {
        t_end.max(f64::MIN_POSITIVE)
    };
    // `c^T f = 0` identically for an invariant, so `c^T k = 0` in every stage (the computed
    // `c^T f` is only the rounding of the fluxes, which are large at a fast equilibrium).
    let invariants = ode.invariants();
    let replace = |rhs: &mut [f64]| {
        for (row, _) in &invariants {
            rhs[*row] = 0.0;
        }
    };
    let mut out = Vec::with_capacity(times.len());
    for &t_out in times {
        while t < t_out {
            if stats.steps + stats.rejected >= MAX_STEPS {
                return Err(MoptError::config(format!(
                    "rosenbrock: more than {MAX_STEPS} steps before {t_out:e} s (at {t:e} s); \
                     loosen --ode-tolerance"
                )));
            }
            let reach = h >= t_out - t;
            let h_try = if reach { t_out - t } else { h };
            if h_try <= t.abs() * 4.0 * f64::EPSILON {
                return Err(MoptError::config(format!(
                    "rosenbrock: the step size underflows at {t:e} s"
                )));
            }
            ode.rhs(&y, &mut f0);
            ode.jacobian(&y, &mut jac);
            for i in 0..n {
                for k in 0..n {
                    w[i * n + k] = -gamma * h_try * jac[i * n + k];
                }
                w[i * n + i] += 1.0;
            }
            for (row, c) in &invariants {
                w[row * n..(row + 1) * n].copy_from_slice(c);
            }
            lu_factor(&mut w, n, &mut piv)?;
            k1.copy_from_slice(&f0);
            replace(&mut k1);
            lu_solve(&w, n, &piv, &mut k1);
            for i in 0..n {
                y1[i] = y[i] + h_try * k1[i];
            }
            ode.rhs(&y1, &mut f1);
            for i in 0..n {
                k2[i] = f1[i] - 2.0 * k1[i];
            }
            replace(&mut k2);
            lu_solve(&w, n, &piv, &mut k2);
            let mut sum = 0.0;
            for i in 0..n {
                y_new[i] = y[i] + 1.5 * h_try * k1[i] + 0.5 * h_try * k2[i];
                let e = 0.5 * h_try * (k1[i] + k2[i]);
                let sc = atol + rtol * y[i].abs().max(y_new[i].abs());
                sum += (e / sc).powi(2);
            }
            let err = (sum / n.max(1) as f64).sqrt();
            if err.is_finite() && err <= 1.0 {
                stats.steps += 1;
                t = if reach { t_out } else { t + h_try };
                y.copy_from_slice(&y_new);
                let fac = (0.9 / err.max(1e-10).sqrt()).clamp(0.2, 5.0);
                h = (h_try * fac).max(if reach { h } else { 0.0 });
            } else {
                stats.rejected += 1;
                let fac = if err.is_finite() {
                    (0.9 / err.sqrt()).clamp(0.2, 0.9)
                } else {
                    0.2
                };
                h = h_try * fac;
            }
        }
        out.push(y.clone());
    }
    Ok((out, stats))
}

/// The solver `rosenbrock`: ROS2 for the models `mass_action` and `canonical`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rosenbrock {
    /// The relative tolerance.
    pub tolerance: f64,
}

impl KineticSolver for Rosenbrock {
    fn name(&self) -> &str {
        "rosenbrock"
    }

    fn solve(
        &mut self,
        ev: &Evolution,
        p0: &[f64],
        times: &[f64],
        _rng: &mut Rng,
    ) -> MoptResult<KineticSolution> {
        let (ode, y0, system): (Box<dyn Ode>, Vec<f64>, Option<&MassActionSystem>) = match ev {
            Evolution::Linear(_) | Evolution::Resolved(_) | Evolution::SemiMarkov(_) => {
                let g = ev.generator()?;
                check_input(g.n, p0, times)?;
                let m = g.rates().matrix();
                let n = g.n;
                let k = (0..n * n).map(|x| m[(x / n, x % n)]).collect();
                (Box::new(LinearOde { n, k }), p0.to_vec(), None)
            }
            Evolution::MassAction(sys) => {
                check_input(sys.n_states, p0, times)?;
                (Box::new(sys.clone()), sys.initial(p0)?, Some(sys))
            }
        };
        let mut order: Vec<usize> = (0..times.len()).collect();
        order.sort_by(|a, b| times[*a].total_cmp(&times[*b]));
        let sorted: Vec<f64> = order.iter().map(|&k| times[k]).collect();
        let scale = y0
            .iter()
            .copied()
            .fold(0.0, f64::max)
            .max(f64::MIN_POSITIVE);
        let (ys, stats) = ros2(
            ode.as_ref(),
            &y0,
            &sorted,
            self.tolerance,
            self.tolerance * 1e-6 * scale,
        )?;
        let mut populations = vec![Vec::new(); times.len()];
        for (k, y) in order.iter().zip(ys) {
            populations[*k] = y;
        }
        let diagnostics = times
            .iter()
            .zip(&populations)
            .map(|(&t, y)| {
                let conservation = match system {
                    Some(sys) => sys.conservation(&y0, y),
                    None => (y.iter().sum::<f64>() - y0.iter().sum::<f64>()).abs(),
                };
                SolveDiagnostics {
                    time: t,
                    conservation,
                    most_negative: y.iter().copied().fold(0.0, f64::min),
                    ..SolveDiagnostics::default()
                }
            })
            .collect();
        Ok(KineticSolution {
            solver: "rosenbrock".to_string(),
            times: times.to_vec(),
            populations,
            diagnostics,
            superstates: vec![None; times.len()],
            notes: vec![format!(
                "rosenbrock (ROS2): {} steps, {} rejected, relative tolerance {:e}{}",
                stats.steps,
                stats.rejected,
                self.tolerance,
                if system.is_some() {
                    "; concentrations in mol/L"
                } else {
                    ""
                }
            )],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `A + B <-> C` with `k_f` (L/mol/s) and `k_r` (1/s) from `A0`, `B0`.
    struct Binding {
        kf: f64,
        kr: f64,
    }

    impl Ode for Binding {
        fn dim(&self) -> usize {
            3
        }
        fn rhs(&self, y: &[f64], f: &mut [f64]) {
            let r = self.kf * y[0] * y[1] - self.kr * y[2];
            f[0] = -r;
            f[1] = -r;
            f[2] = r;
        }
        fn jacobian(&self, y: &[f64], j: &mut [f64]) {
            let (da, db, dc) = (self.kf * y[1], self.kf * y[0], -self.kr);
            j.copy_from_slice(&[-da, -db, -dc, -da, -db, -dc, da, db, dc]);
        }
    }

    /// The analytic `[C](t)` of `A + B <-> C` from `[C](0) = 0`.
    fn analytic(kf: f64, kr: f64, a0: f64, b0: f64, t: f64) -> f64 {
        // dx/dt = kf (a0 - x)(b0 - x) - kr x = kf (x - r1)(x - r2), r1 < r2.
        let s = a0 + b0 + kr / kf;
        let disc = (s * s - 4.0 * a0 * b0).sqrt();
        let (r1, r2) = ((s - disc) / 2.0, (s + disc) / 2.0);
        let alpha = kf * (r2 - r1);
        let e = (-alpha * t).exp();
        r1 * (1.0 - e) / (1.0 - (r1 / r2) * e)
    }

    #[test]
    fn ros2_follows_the_analytic_binding_curve() {
        let (kf, kr, a0, b0) = (2.0e3, 0.5, 1e-3, 3e-3);
        let times = [1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0];
        let (ys, stats) = ros2(&Binding { kf, kr }, &[a0, b0, 0.0], &times, 1e-8, 1e-16).unwrap();
        for (y, &t) in ys.iter().zip(&times) {
            let c = analytic(kf, kr, a0, b0, t);
            assert!((y[2] - c).abs() < 1e-6 * a0, "t {t}: {} vs {c}", y[2]);
            // Conservation of A + C and B + C to rounding.
            assert!((y[0] + y[2] - a0).abs() < 1e-15, "{}", y[0] + y[2] - a0);
            assert!((y[1] + y[2] - b0).abs() < 1e-15);
        }
        assert!(stats.steps > 10 && stats.steps < 100_000, "{stats:?}");
        // A tighter tolerance gives a smaller error.
        let (loose, _) = ros2(&Binding { kf, kr }, &[a0, b0, 0.0], &[1.0], 1e-4, 1e-16).unwrap();
        let (tight, _) = ros2(&Binding { kf, kr }, &[a0, b0, 0.0], &[1.0], 1e-9, 1e-18).unwrap();
        let c = analytic(kf, kr, a0, b0, 1.0);
        assert!((tight[0][2] - c).abs() < (loose[0][2] - c).abs());
    }

    #[test]
    fn ros2_is_stable_on_a_stiff_linear_chain() {
        // A <-> B <-> C with rates from 1e12 to 1e-2 per second.
        let k = vec![
            -1e12,
            1e9,
            0.0, //
            1e12,
            -1e9 - 1e-2,
            1.0, //
            0.0,
            1e-2,
            -1.0,
        ];
        let ode = LinearOde { n: 3, k };
        let (ys, _) = ros2(&ode, &[1.0, 0.0, 0.0], &[1e-9, 1e3], 1e-9, 1e-18).unwrap();
        let last = &ys[1];
        // Equilibrium: pB / pA = 1e3, pC / pB = 1e-2.
        let z = 1.0 + 1e3 + 10.0;
        assert!((last[0] - 1.0 / z).abs() < 1e-7, "{last:?}");
        assert!((last[2] - 10.0 / z).abs() < 1e-7, "{last:?}");
        let sum = last.iter().sum::<f64>();
        assert!((sum - 1.0).abs() < 1e-12, "{sum}");
    }
}
