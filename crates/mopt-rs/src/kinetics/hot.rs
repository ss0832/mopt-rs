//! Hot (chemically activated) intermediates as additional states: the diagnostic `H` and the
//! two-grain model `two_grain` (`kinetics_beyond_tst.md`, sections 2.3, 3.6, 3.7, 3.11).
//!
//! An intermediate `I` formed from `a` over the saddle point `e` holds the energy of the flux
//! that crossed `e`: its energy above the lowest level of `I` is `E* = E0(e) + eps - E0(I)`,
//! `eps` the mean energy of the canonical flux above the level of the saddle point
//! (`sum_j W_j (j delta) exp(-j delta / kT) / sum_j W_j exp(-j delta / kT)`, from the states
//! counted as for `rrkm`). With the cooling rate `omega` (`--cooling-rate`, 1/s; of the order of
//! 1e11 to 1e12 1/s in solution) the diagnostic is
//!
//! ```text
//! H = max over the other exits x of I of k_x(E*) / omega
//! ```
//!
//! with `k_x(E*)` the RRKM rate constant of the exit at that energy ([`crate::kinetics::rrkm`]):
//! `H << 1` thermal (the canonical model holds), `H` of order one an energy-resolved model is
//! needed.
//!
//! **The two-grain model** adds, for every entrance with `H` above `--hot-threshold` (0.1), a
//! hot grain `I*` that the entrance feeds, a proposal of the design (a Markovian embedding of
//! the memory of `I`). The population of `I` is split so that the model is the canonical one
//! when the cooling is fast:
//!
//! ```text
//! pi*      = F_e / k_e(E*)                  (the weight of the energies at which the entrance arrives)
//! pi_I     -> pi_I - pi*                    (the thermal part)
//! a  <-> I*:  F_e                          (the whole entrance)
//! I* <-> I :  omega pi* (1 - pi* / pi_I)   (cooling and heating, two-level strong collisions)
//! I* <-> b :  k_x(E*) pi*                   (the hot exits)
//! I  <-> b :  F_x - k_x(E*) pi*             (what is left of the thermal exits)
//! ```
//!
//! Lumping `I*` with `I` gives back `pi_I`, `F_e`, and every `F_x`, so `omega -> infinity` is
//! the canonical model exactly; `omega -> 0` leaves the chemically activated branching
//! `k_x(E*) / sum_y k_y(E*)` (checked in the tests). The generator is in flux form (detailed
//! balance). A thermal exit that would become negative keeps a tiny remainder and the note says
//! so. Exits without counted states at their saddle points (crossing points, electron transfers,
//! barrierless steps) are not hot exits.
//!
//! The populations of `I*` are added to those of `I` in the reports.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use super::conditions::Conditions;
use super::generator::Generator;
use super::network::{BottleneckKind, KineticNetwork};
use super::rate::tst::SymmetryChoice;
use super::rate::RateSet;
use super::rrkm::{rate_at_energy, Rrkm, RrkmSettings, StateCount};
use super::solve::Resolved;
use crate::domain::Units;
use crate::error::{MoptError, MoptResult};
use crate::math::stable::{log_diff_exp, log_sum_exp};

/// The settings of `two_grain`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HotSettings {
    /// The cooling rate of a hot intermediate (1/s).
    pub cooling: Option<f64>,
    /// A hot grain is added where `H` exceeds this.
    pub threshold: f64,
}

impl Default for HotSettings {
    fn default() -> Self {
        Self {
            cooling: None,
            threshold: 0.1,
        }
    }
}

/// One entrance into an intermediate and its diagnostic.
#[derive(Debug, Clone, PartialEq)]
pub struct HotEntrance {
    /// The step of the rate set.
    pub step: usize,
    /// The state the intermediate is formed from.
    pub from: usize,
    /// The intermediate.
    pub intermediate: usize,
    /// `E*` above the lowest level of the intermediate (Hartree).
    pub excess: f64,
    /// `k(E*)` back over the entrance (1/s).
    pub back: f64,
    /// `(step, k_x(E*))` of the other exits (1/s).
    pub exits: Vec<(usize, f64)>,
    /// `H`.
    pub h: f64,
    /// Whether a hot grain was added.
    pub grain: bool,
}

/// The mean energy (Hartree) of the canonical flux above the level of a saddle point.
fn mean_flux_energy(ts: &StateCount, kt: f64) -> f64 {
    let w = ts.sum_of_states();
    let mut num = Vec::new();
    let mut den = Vec::new();
    for (j, x) in w.iter().enumerate() {
        if *x > 0.0 && j > 0 {
            let e = j as f64 * ts.cell;
            num.push(x.ln() + e.ln() - e / kt);
            den.push(x.ln() - e / kt);
        } else if *x > 0.0 {
            den.push(x.ln());
        }
    }
    (log_sum_exp(&num) - log_sum_exp(&den)).exp()
}

/// The two-grain model.
#[derive(Debug, Clone, Copy)]
pub struct TwoGrain {
    /// Its settings.
    pub settings: HotSettings,
    /// The counting of states.
    pub rrkm: RrkmSettings,
    /// The symmetry numbers.
    pub symmetry: SymmetryChoice,
}

impl TwoGrain {
    /// The entrances of `net` into its intermediates with their diagnostics.
    pub fn entrances(
        &self,
        net: &KineticNetwork,
        rates: &RateSet,
        c: &Conditions,
        cooling: f64,
    ) -> MoptResult<Vec<HotEntrance>> {
        let theory = Rrkm::new(self.rrkm, self.symmetry)?;
        let kt = c.kt();
        let n = rates.states.len();
        let mut wells: Vec<Option<StateCount>> = Vec::with_capacity(n);
        for &k in &rates.states {
            let s = &net.states[k];
            wells.push(if s.fragments.is_empty() {
                theory.count_state(s, c, None)?
            } else {
                None
            });
        }
        let mut saddles: Vec<Option<StateCount>> = Vec::with_capacity(rates.steps.len());
        for s in &rates.steps {
            let b = &net.bottlenecks[s.bottleneck];
            saddles.push(if b.kind == BottleneckKind::Saddle && !s.clamped {
                // Up to the highest energy at which an exit is asked for.
                theory.count_bottleneck(b, c, None)?
            } else {
                None
            });
        }
        let mut out = Vec::new();
        for (e, step) in rates.steps.iter().enumerate() {
            let Some(ts) = &saddles[e] else {
                continue;
            };
            for side in 0..2 {
                let (from, i) = (step.states[side], step.states[1 - side]);
                let Some(well) = &wells[i] else {
                    continue;
                };
                let e_star = ts.e0 + mean_flux_energy(ts, kt);
                if e_star <= well.e0 {
                    continue;
                }
                // Recount the intermediate up to E* plus a margin when its count is short.
                let needed = ((e_star - well.e0) / well.cell).ceil() as usize + 64;
                let well = if well.counts.len() < needed {
                    match theory.count_state(&net.states[rates.states[i]], c, Some(needed))? {
                        Some(w) => w,
                        None => continue,
                    }
                } else {
                    well.clone()
                };
                let width = 0.5 * kt;
                let at = |x: usize| -> MoptResult<f64> {
                    let Some(t) = &saddles[x] else {
                        return Ok(0.0);
                    };
                    let reach = ((e_star - t.e0) / t.cell).ceil();
                    let t = if reach > t.counts.len() as f64 {
                        let b = &net.bottlenecks[rates.steps[x].bottleneck];
                        match theory.count_bottleneck(b, c, Some(reach as usize + 64))? {
                            Some(full) => full,
                            None => return Ok(0.0),
                        }
                    } else {
                        t.clone()
                    };
                    Ok(rate_at_energy(&t, &well, e_star, width, c))
                };
                let back = at(e)?;
                if back.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
                    continue;
                }
                let mut exits = Vec::new();
                let mut h: f64 = 0.0;
                for (x, other) in rates.steps.iter().enumerate() {
                    if x == e || !other.states.contains(&i) {
                        continue;
                    }
                    let k = at(x)?;
                    if k > 0.0 {
                        h = h.max(k / cooling);
                        exits.push((x, k));
                    }
                }
                out.push(HotEntrance {
                    step: e,
                    from,
                    intermediate: i,
                    excess: e_star - well.e0,
                    back,
                    exits,
                    h,
                    grain: h > self.settings.threshold,
                });
            }
        }
        Ok(out)
    }

    /// The generator of the states of `rates` with the hot grains.
    pub fn resolve(
        &self,
        net: &KineticNetwork,
        rates: &RateSet,
        c: &Conditions,
    ) -> MoptResult<Resolved> {
        let Some(cooling) = self.settings.cooling else {
            return Err(MoptError::config(
                "the model two_grain needs the cooling rate of a hot intermediate \
                 (--cooling-rate, 1/s; of the order of 1e11 to 1e12 in solution)",
            ));
        };
        if !(cooling > 0.0 && cooling.is_finite()) {
            return Err(MoptError::config("--cooling-rate must be positive"));
        }
        if !rates.detailed_balance {
            return Err(MoptError::unsupported(
                "the model two_grain needs detailed balance (flux form); the rate constants of \
                 this network are directed",
            ));
        }
        let entrances = self.entrances(net, rates, c, cooling)?;
        let n = rates.states.len();
        let ln_flux: Vec<f64> = rates
            .steps
            .iter()
            .map(|s| s.ln_k[0] + rates.ln_pi[s.states[0]])
            .collect();
        let mut ln_pi = rates.ln_pi.clone();
        let mut labels = rates.labels.clone();
        let mut owner: Vec<usize> = (0..n).collect();
        // The hot part taken from every thermal exit and from the thermal states.
        let mut taken: Vec<Vec<f64>> = vec![Vec::new(); rates.steps.len()];
        let mut hot_weight: Vec<Vec<f64>> = vec![Vec::new(); n];
        let mut fluxes: Vec<(usize, usize, f64)> = Vec::new();
        let mut replaced = vec![false; rates.steps.len()];
        let mut notes = Vec::new();
        let kcal = Units::HARTREE_TO_KCAL_MOL;
        for h in &entrances {
            let step = &rates.steps[h.step];
            if !h.grain {
                notes.push(format!(
                    "two_grain: {} formed from {} over {} at {:.2} kcal/mol: H = {:.3e} (thermal)",
                    rates.labels[h.intermediate],
                    rates.labels[h.from],
                    net.bottlenecks[step.bottleneck].label,
                    h.excess * kcal,
                    h.h
                ));
                continue;
            }
            if replaced[h.step] {
                // Both sides of one step would be hot: the first keeps the entrance.
                continue;
            }
            replaced[h.step] = true;
            let star = ln_pi.len();
            let ln_star = ln_flux[h.step] - h.back.ln();
            ln_pi.push(ln_star);
            labels.push(format!(
                "{}*({})",
                rates.labels[h.intermediate], net.bottlenecks[step.bottleneck].label
            ));
            owner.push(h.intermediate);
            hot_weight[h.intermediate].push(ln_star);
            fluxes.push((h.from, star, ln_flux[h.step]));
            for &(x, k) in &h.exits {
                let s = &rates.steps[x];
                let other = if s.states[0] == h.intermediate {
                    s.states[1]
                } else {
                    s.states[0]
                };
                fluxes.push((star, other, k.ln() + ln_star));
                taken[x].push(k.ln() + ln_star);
            }
            notes.push(format!(
                "two_grain: {} formed from {} over {} at {:.2} kcal/mol: H = {:.3e}, a hot grain \
                 is added (k back {:.3e} 1/s, {} hot exit(s))",
                rates.labels[h.intermediate],
                rates.labels[h.from],
                net.bottlenecks[step.bottleneck].label,
                h.excess * kcal,
                h.h,
                h.back,
                h.exits.len()
            ));
        }
        // The thermal parts and the cooling.
        let mut clipped = 0usize;
        for i in 0..n {
            if hot_weight[i].is_empty() {
                continue;
            }
            let ln_hot = log_sum_exp(&hot_weight[i]);
            if ln_hot >= ln_pi[i] {
                return Err(MoptError::calc(format!(
                    "two_grain: the hot grains of {} would hold more than its whole population",
                    rates.labels[i]
                )));
            }
            let thermal = log_diff_exp(ln_pi[i], ln_hot);
            let total = ln_pi[i];
            ln_pi[i] = thermal;
            for (star, &o) in owner.iter().enumerate().skip(n) {
                if o == i {
                    // omega pi* (1 - pi* / pi_I) with the whole pi_I.
                    let share = log_diff_exp(0.0, ln_pi[star] - total);
                    fluxes.push((star, i, cooling.ln() + ln_pi[star] + share));
                }
            }
        }
        for (x, s) in rates.steps.iter().enumerate() {
            if replaced[x] {
                continue;
            }
            let ln = if taken[x].is_empty() {
                ln_flux[x]
            } else {
                let hot = log_sum_exp(&taken[x]);
                if hot >= ln_flux[x] - 1e-12 {
                    clipped += 1;
                    ln_flux[x] - 30.0
                } else {
                    log_diff_exp(ln_flux[x], hot)
                }
            };
            fluxes.push((s.states[0], s.states[1], ln));
        }
        if clipped > 0 {
            notes.push(format!(
                "two_grain: {clipped} thermal exit(s) would become negative after their hot part \
                 is taken out (the hot exits carry nearly all of their flux); a remainder of \
                 exp(-30) is kept"
            ));
        }
        let generator = Generator::from_fluxes(labels, ln_pi, &fluxes)?;
        Ok(Resolved {
            generator,
            owner,
            states: n,
            model: "two_grain".to_string(),
            phenomenological: None,
            notes,
        })
    }
}
