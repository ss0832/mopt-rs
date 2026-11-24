//! `windows`: mass action by pseudo-first-order windows, the linear contraction of RCMC carried
//! over a changing reservoir (`kinetics_beyond_tst.md`, section 3.8, item 1).
//!
//! In the steps of a network at most one state appears on each side and every other partner is
//! a species (a reservoir). With the species held at their concentrations the states obey the
//! linear generator `K(c_res)` (`k' = k_2 [S]` for a step that takes up `S`), which `propagate`
//! (the contraction to superstates and their exact propagation) solves over a window; then the
//! species are moved by the stoichiometry of the fluxes of that window, `delta [S] = sum_r
//! nu_rS k'_r int c_i(r) dt` (the time integrals of the states from the block exponential, so the
//! totals of every species, free and bound, are kept to rounding), and the next window starts.
//! A window is as long as no species changes by more than `tolerance` of its concentration
//! (`--window-tolerance`, 1e-3; relative to the total of the states for a species near zero), as
//! estimated from the rates at its start and checked after it (a window that changed a species
//! by more than twice the tolerance is done again with half the length). The error is of first
//! order in the tolerance (checked against the analytic second-order solution of `A + B -> C`)
//! and vanishes while the reservoir is constant.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use super::{
    check_input, Closure, Evolution, KineticSolution, KineticSolver, Propagate, SolveDiagnostics,
};
use crate::core_infra::rng::Rng;
use crate::error::{MoptError, MoptResult};
use crate::kinetics::generator::{Generator, RateEntry};
use crate::kinetics::mass_action::MassActionSystem;

/// The default largest relative change of a species over one window.
pub const DEFAULT_WINDOW_TOLERANCE: f64 = 1e-3;

/// Pseudo-first-order windows.
#[derive(Debug, Clone, Copy)]
pub struct Windows {
    /// The largest relative change of a species over a window.
    pub tolerance: f64,
    /// The tolerance of the contraction within a window.
    pub contraction: f64,
}

/// The linear generator of the states with the species at `c` (one entry per reaction with a
/// positive rate), and for every reaction its state reactant, its rate constant, and the
/// stoichiometry of the species.
fn window(
    sys: &MassActionSystem,
    c: &[f64],
) -> MoptResult<(Generator, Vec<(usize, f64, Vec<(usize, f64)>)>)> {
    let n = sys.n_states;
    let mut entries = Vec::new();
    let mut reactions = Vec::new();
    for r in &sys.reactions {
        let states: Vec<usize> = r.reactants.iter().copied().filter(|&i| i < n).collect();
        let products: Vec<usize> = r.products.iter().copied().filter(|&i| i < n).collect();
        let (&[from], &[to]) = (states.as_slice(), products.as_slice()) else {
            return Err(MoptError::unsupported(
                "windows: a reaction with other than one state on each side (the steps of a \
                 network have one); use the solver contract",
            ));
        };
        let mut k = r.ln_k.exp();
        for &i in r.reactants.iter().filter(|&&i| i >= n) {
            k *= c[i];
        }
        let mut nu: Vec<(usize, f64)> = Vec::new();
        for &i in r.reactants.iter().filter(|&&i| i >= n) {
            nu.push((i, -1.0));
        }
        for &i in r.products.iter().filter(|&&i| i >= n) {
            nu.push((i, 1.0));
        }
        if k > 0.0 && from != to {
            entries.push(RateEntry {
                from,
                to,
                ln_k: k.ln(),
            });
        }
        reactions.push((from, k, nu));
    }
    let g = Generator::from_directed(sys.labels[..n].to_vec(), &entries)?;
    Ok((g, reactions))
}

impl KineticSolver for Windows {
    fn name(&self) -> &str {
        "windows"
    }

    fn solve(
        &mut self,
        ev: &Evolution,
        p0: &[f64],
        times: &[f64],
        rng: &mut Rng,
    ) -> MoptResult<KineticSolution> {
        let Evolution::MassAction(sys) = ev else {
            return Err(MoptError::unsupported(
                "windows is a solver of the model mass_action (a linear model is one window)",
            ));
        };
        if !(self.tolerance > 0.0 && self.tolerance < 1.0) {
            return Err(MoptError::config(
                "windows: the tolerance must be between 0 and 1",
            ));
        }
        check_input(sys.n_states, p0, times)?;
        let n = sys.n_states;
        let y0 = sys.initial(p0)?;
        let floor = self.tolerance.max(1e-6) * sys.total;
        let mut order: Vec<usize> = (0..times.len()).collect();
        order.sort_by(|&a, &b| times[a].total_cmp(&times[b]));
        let mut y = y0.clone();
        let mut t = 0.0;
        let mut windows = 0usize;
        let mut redone = 0usize;
        let mut out = vec![Vec::new(); times.len()];
        let mut propagate = Propagate::new(self.contraction, Closure::A)?;
        for &k in &order {
            let target = times[k];
            while t < target {
                let (g, reactions) = window(sys, &y)?;
                // The rates of change of the species at the start.
                let mut rate = vec![0.0; y.len()];
                for (from, kr, nu) in &reactions {
                    for &(i, v) in nu {
                        rate[i] += v * kr * y[*from];
                    }
                }
                let mut dt = target - t;
                for i in n..y.len() {
                    if rate[i] != 0.0 {
                        dt = dt.min(self.tolerance * y[i].abs().max(floor) / rate[i].abs());
                    }
                }
                loop {
                    let lin = Evolution::Linear(g.clone());
                    let p: Vec<f64> = y[..n].to_vec();
                    let end = propagate.solve(&lin, &p, &[dt], rng)?;
                    let tau = propagate.residence(&lin, &p, dt, rng)?;
                    let mut next = y.clone();
                    next[..n].copy_from_slice(&end.populations[0]);
                    for (from, kr, nu) in &reactions {
                        for &(i, v) in nu {
                            next[i] += v * kr * tau[*from];
                        }
                    }
                    let worst = (n..y.len())
                        .map(|i| (next[i] - y[i]).abs() / y[i].abs().max(floor))
                        .fold(0.0, f64::max);
                    if worst > 2.0 * self.tolerance && dt > 1e-12 * target {
                        dt *= 0.5;
                        redone += 1;
                        continue;
                    }
                    y = next;
                    t += dt;
                    windows += 1;
                    break;
                }
                if windows > crate::kinetics::mass_action::MAX_STEPS {
                    return Err(MoptError::calc(
                        "windows: too many windows (the reservoir changes too fast; use contract)",
                    ));
                }
            }
            out[k] = y.clone();
        }
        let diagnostics = times
            .iter()
            .zip(&out)
            .map(|(&time, y)| SolveDiagnostics {
                time,
                conservation: sys.conservation(&y0, y),
                most_negative: y.iter().copied().fold(0.0, f64::min),
                ..SolveDiagnostics::default()
            })
            .collect();
        Ok(KineticSolution {
            solver: "windows".to_string(),
            times: times.to_vec(),
            populations: out,
            diagnostics,
            superstates: vec![None; times.len()],
            notes: vec![format!(
                "windows: {windows} pseudo-first-order windows ({redone} halved), the species \
                 changing by at most {:e} in each; concentrations in mol/L",
                self.tolerance
            )],
        })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::kinetics::mass_action::Reaction;

    /// `A + B -> C` (states A, C; species B) with `k` (L/mol/s).
    fn binding(k: f64, a0: f64, b0: f64) -> MassActionSystem {
        MassActionSystem {
            labels: vec!["A".into(), "C".into(), "B".into()],
            n_states: 2,
            species: vec![0],
            reactions: vec![Reaction {
                reactants: vec![0, 2],
                products: vec![1],
                ln_k: k.ln(),
                step: 0,
            }],
            species_initial: vec![b0],
            total: a0,
            conserved: vec![
                ("states".into(), vec![1.0, 1.0, 0.0]),
                ("B (free and bound)".into(), vec![0.0, 1.0, 1.0]),
            ],
        }
    }

    #[test]
    fn windows_converge_to_the_second_order_solution_at_first_order() {
        // A + B -> C with [A]0 = [B]0: [A](t) = a0 / (1 + k a0 t).
        let (k, a0) = (2.0, 1.0);
        let t = 1.0;
        let exact = a0 / (1.0 + k * a0 * t);
        let ev = Evolution::MassAction(binding(k, a0, a0));
        let mut rng = Rng::from_seed(1);
        let mut errors = Vec::new();
        for tol in [1e-2, 1e-3, 1e-4] {
            let mut s = Windows {
                tolerance: tol,
                contraction: 1e-8,
            };
            let sol = s.solve(&ev, &[1.0, 0.0], &[t], &mut rng).unwrap();
            let y = &sol.populations[0];
            errors.push((y[0] - exact).abs());
            // B free and bound, and the states, are kept.
            assert!(
                sol.diagnostics[0].conservation < 1e-12,
                "{}",
                sol.diagnostics[0].conservation
            );
            assert!((y[2] - y[0]).abs() < 1e-12);
        }
        assert!(
            errors[1] < errors[0] / 5.0 && errors[2] < errors[1] / 5.0,
            "{errors:?}"
        );
        assert!(errors[2] < 1e-3, "{errors:?}");
        // B in excess: the reservoir hardly changes and A decays exponentially.
        let b0 = 1e3;
        let ev = Evolution::MassAction(binding(k, 1e-3, b0));
        let mut s = Windows {
            tolerance: 1e-3,
            contraction: 1e-8,
        };
        let sol = s.solve(&ev, &[1.0, 0.0], &[1e-3], &mut rng).unwrap();
        let want = 1e-3 * (-k * b0 * 1e-3_f64).exp();
        assert!(
            (sol.populations[0][0] / want - 1.0).abs() < 1e-5,
            "{} {want}",
            sol.populations[0][0]
        );
    }
}
