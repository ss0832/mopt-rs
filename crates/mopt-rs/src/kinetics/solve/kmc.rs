//! `kmc`: kinetic Monte Carlo by Gillespie's direct method (Gillespie, J. Comput. Phys. 1976,
//! 22, 403; J. Phys. Chem. 1977, 81, 2340; the same rejection-free scheme as Bortz, Kalos,
//! Lebowitz, J. Comput. Phys. 1975, 17, 10). Every trajectory starts in a state drawn from the
//! initial populations, waits in a state `j` an exponential time of rate `a_j = sum_i K[i][j]`
//! and jumps to `i` with probability `K[i][j] / a_j`; the populations at the output times are
//! the shares of the trajectories in every state, with the standard error
//! `sqrt(p (1 - p) / N)` in the diagnostics.
//!
//! A semi-Markov model (`semi_markov`) keeps the jump probabilities of its Markov limit and
//! draws the residence time of a state from its gamma distribution of the same mean (shape
//! `k`; Marsaglia and Tsang, ACM Trans. Math. Softw. 2000, 26, 363), so that the populations
//! of a chain with non-exponential residence times are sampled exactly.
//!
//! It is the reference for small models and semi-Markov tests, not a general solver: the
//! number of events grows with the fastest rate times the time (`kinetics_beyond_tst.md`,
//! section 7), and a run that would need more than `max_events` events is stopped with an
//! error that names the contraction instead.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use super::{
    check_input, conservation, Evolution, KineticSolution, KineticSolver, SolveDiagnostics,
};
use crate::core_infra::rng::Rng;
use crate::error::{MoptError, MoptResult};

/// The default number of trajectories.
pub const DEFAULT_TRAJECTORIES: usize = 10_000;
/// The default largest number of events of a run.
pub const DEFAULT_MAX_EVENTS: u64 = 100_000_000;

/// Kinetic Monte Carlo.
#[derive(Debug, Clone, Copy)]
pub struct Kmc {
    /// Trajectories.
    pub trajectories: usize,
    /// The largest number of events of a run (all trajectories together).
    pub max_events: u64,
}

impl Default for Kmc {
    fn default() -> Self {
        Self {
            trajectories: DEFAULT_TRAJECTORIES,
            max_events: DEFAULT_MAX_EVENTS,
        }
    }
}

/// A draw from the gamma distribution of shape `k` and scale 1 (Marsaglia and Tsang; for
/// `k < 1` boosted by `u^(1/k)`).
pub fn gamma_draw(k: f64, rng: &mut Rng) -> f64 {
    if k < 1.0 {
        let u = rng.next_f64().max(f64::MIN_POSITIVE);
        return gamma_draw(k + 1.0, rng) * u.powf(1.0 / k);
    }
    let d = k - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let x = rng.standard_normal();
        let v = 1.0 + c * x;
        if v <= 0.0 {
            continue;
        }
        let v = v * v * v;
        let u = rng.next_f64();
        if u < 1.0 - 0.0331 * x.powi(4) || u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
            return d * v;
        }
    }
}

/// The jump table of a generator: for every state its total rate and the cumulative rates of
/// its exits.
struct Jumps {
    out: Vec<f64>,
    exits: Vec<Vec<(usize, f64)>>,
}

impl Jumps {
    fn of(rates: &crate::kinetics::generator::Rates) -> Self {
        let n = rates.n;
        let mut out = vec![0.0; n];
        let mut exits = vec![Vec::new(); n];
        for j in 0..n {
            let mut cum = 0.0;
            for i in 0..n {
                let k = rates.get(i, j);
                if i != j && k > 0.0 {
                    cum += k;
                    exits[j].push((i, cum));
                }
            }
            out[j] = cum;
        }
        Self { out, exits }
    }

    fn next(&self, j: usize, rng: &mut Rng) -> usize {
        let target = rng.next_f64() * self.out[j];
        for &(i, cum) in &self.exits[j] {
            if target < cum {
                return i;
            }
        }
        self.exits[j].last().map_or(j, |&(i, _)| i)
    }
}

impl KineticSolver for Kmc {
    fn name(&self) -> &str {
        "kmc"
    }

    fn solve(
        &mut self,
        ev: &Evolution,
        p0: &[f64],
        times: &[f64],
        rng: &mut Rng,
    ) -> MoptResult<KineticSolution> {
        let g = ev.generator()?;
        let n = g.n;
        check_input(n, p0, times)?;
        if self.trajectories == 0 {
            return Err(MoptError::config("kmc needs at least one trajectory"));
        }
        let shapes: Option<Vec<f64>> = match ev {
            Evolution::SemiMarkov(s) => Some(s.shapes.clone()),
            _ => None,
        };
        let jumps = Jumps::of(&g.rates());
        let total: f64 = p0.iter().sum();
        let mut order: Vec<usize> = (0..times.len()).collect();
        order.sort_by(|&a, &b| times[a].total_cmp(&times[b]));
        let mut counts = vec![vec![0usize; n]; times.len()];
        let mut events: u64 = 0;
        for _ in 0..self.trajectories {
            // The start.
            let target = rng.next_f64() * total;
            let mut cum = 0.0;
            let mut state = n - 1;
            for (i, p) in p0.iter().enumerate() {
                cum += p;
                if target < cum {
                    state = i;
                    break;
                }
            }
            let mut t = 0.0;
            let mut next = 0;
            while next < order.len() {
                let a = jumps.out[state];
                if a.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
                    // An absorbing state: it stays.
                    for &k in &order[next..] {
                        counts[k][state] += 1;
                    }
                    break;
                }
                let wait = match &shapes {
                    Some(k) if (k[state] - 1.0).abs() > 1e-12 => {
                        gamma_draw(k[state], rng) / (k[state] * a)
                    }
                    _ => -(1.0 - rng.next_f64()).ln() / a,
                };
                let leave = t + wait;
                while next < order.len() && times[order[next]] < leave {
                    counts[order[next]][state] += 1;
                    next += 1;
                }
                if next >= order.len() {
                    break;
                }
                state = jumps.next(state, rng);
                t = leave;
                events += 1;
                if events > self.max_events {
                    return Err(MoptError::unsupported(format!(
                        "kmc: more than {} events (the fastest rate {:.3e} 1/s times the last \
                         output time {:.3e} s, for {} trajectories): the network is too stiff for \
                         kinetic Monte Carlo; use propagate (the contraction) or expm",
                        self.max_events,
                        jumps.out.iter().copied().fold(0.0, f64::max),
                        times.iter().copied().fold(0.0, f64::max),
                        self.trajectories
                    )));
                }
            }
        }
        let trajectories = self.trajectories as f64;
        let mut populations = Vec::with_capacity(times.len());
        let mut diagnostics = Vec::with_capacity(times.len());
        for (k, &t) in times.iter().enumerate() {
            let p: Vec<f64> = counts[k]
                .iter()
                .map(|c| total * *c as f64 / trajectories)
                .collect();
            let worst = p
                .iter()
                .map(|x| {
                    let s = (x / total).clamp(0.0, 1.0);
                    total * (s * (1.0 - s) / trajectories).sqrt()
                })
                .fold(0.0, f64::max);
            let (cons, neg) = conservation(&p, p0);
            diagnostics.push(SolveDiagnostics {
                time: t,
                conservation: cons,
                most_negative: neg,
                notes: vec![format!(
                    "kmc: {} trajectories, largest standard error of a population {worst:.2e}",
                    self.trajectories
                )],
                ..SolveDiagnostics::default()
            });
            populations.push(p);
        }
        Ok(KineticSolution {
            solver: "kmc".to_string(),
            times: times.to_vec(),
            populations,
            diagnostics,
            superstates: vec![None; times.len()],
            notes: vec![format!(
                "kmc: {events} events in {} trajectories{}",
                self.trajectories,
                if shapes.is_some() {
                    " (residence times of the semi-Markov model)"
                } else {
                    ""
                }
            )],
        })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::kinetics::generator::Generator;
    use crate::kinetics::solve::Expm;

    #[test]
    fn kmc_agrees_with_the_master_equation_within_three_standard_errors() {
        // A <-> B -> C with rates of the same order (not stiff).
        let k = vec![
            vec![0.0, 0.5, 0.0],
            vec![2.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
        ];
        let ev = Evolution::Linear(Generator::from_matrix(&k).unwrap());
        let p0 = [1.0, 0.0, 0.0];
        let times = [0.3, 1.0, 3.0];
        let mut rng = Rng::from_seed(7);
        let mut kmc = Kmc {
            trajectories: 20_000,
            ..Kmc::default()
        };
        let sampled = kmc.solve(&ev, &p0, &times, &mut rng).unwrap();
        let exact = Expm.solve(&ev, &p0, &times, &mut rng).unwrap();
        for (ps, pe) in sampled.populations.iter().zip(&exact.populations) {
            for (s, e) in ps.iter().zip(pe) {
                let se = (e * (1.0 - e) / 20_000.0).sqrt().max(1e-12);
                assert!((s - e).abs() < 3.0 * se + 1e-12, "{s} {e} ({se})");
            }
            assert!((ps.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        }
        // Stiff networks are refused.
        let stiff = vec![vec![0.0, 1e9], vec![1e9, 0.0]];
        let ev = Evolution::Linear(Generator::from_matrix(&stiff).unwrap());
        let mut kmc = Kmc {
            trajectories: 10,
            max_events: 1000,
        };
        assert!(kmc.solve(&ev, &[1.0, 0.0], &[1.0], &mut rng).is_err());
    }

    #[test]
    fn the_gamma_draws_have_their_mean_and_variance() {
        let mut rng = Rng::from_seed(3);
        for k in [0.5_f64, 1.0, 3.0] {
            let n = 200_000;
            let x: Vec<f64> = (0..n).map(|_| gamma_draw(k, &mut rng)).collect();
            let mean = x.iter().sum::<f64>() / n as f64;
            let var = x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;
            assert!(
                (mean - k).abs() < 4.0 * (k / n as f64).sqrt(),
                "{k}: {mean}"
            );
            assert!((var / k - 1.0).abs() < 0.03, "{k}: {var}");
        }
    }
}
