//! `egme`: the energy-grained master equation (`kinetics_beyond_tst.md`, sections 2.4 and 3.7).
//!
//! The states are pairs (well, grain): every minimum with frequencies and a structure is
//! divided into grains of width `--egme-grain` (50 cm^-1) on one energy axis that starts at the
//! lowest level of the wells and reaches `--egme-above` (40) `k_B T` above the highest saddle
//! point or well (the share of the thermal populations and fluxes that the grains hold is
//! checked against `rrkm` and reported when it is short by more than 1e-6). The states of the other kind (dissociation channels, minima without
//! frequencies) stay thermal states. The states are counted as the rate theory `rrkm` counts
//! them ([`crate::kinetics::rrkm`]) and every quantity is a sum over the cells of a grain, so
//! that the sums over the grains are the canonical quantities of `rrkm` exactly:
//!
//! ```text
//! weights:     pi[(w,a)] = exp(-(G_trans + G_elec)_w / kT) sum_{cells of w in a} N_j exp(-E_j / kT)
//! reaction:    F[(v,a)][(w,a)] = exp(-(G_trans + G_elec)_ts / kT) (1/h) sum_{cells of ts in a} W_j delta exp(-E_j / kT)
//! collisions:  F[(w,a)][(w,b)] = omega P(a <- b) pi[(w,b)]   (a below b)
//! ```
//!
//! so `K[(v,a)][(w,a)] = F / pi[(w,a)]` is `k_vw(E_a)` of RRKM, and the generator is in flux form
//! (detailed balance by construction). A saddle point between a well and a thermal state takes
//! the products into that state; the corrections of a step (tunnelling, recrossing, shares of an
//! N-furcation) multiply its flux at every energy (an energy-independent transmission); a step
//! without counted states at its bottleneck (a crossing point, an electron transfer, a
//! barrierless step, a saddle point without frequencies) joins the grains of equal energy of two
//! wells, or every grain of a well and a thermal state, with its canonical flux distributed as
//! the Boltzmann distribution of the well; two thermal states keep their canonical step.
//!
//! **Collisions** (Pilling and Robertson, Annu. Rev. Phys. Chem. 2003, 54, 245; Miller and
//! Klippenstein, J. Phys. Chem. A 2006, 110, 10528): `exponential_down`, the downward
//! probabilities `exp(-(E_b - E_a) / alpha)` with `alpha` the mean energy transferred downwards
//! (`--energy-transfer`, cm^-1), the upward ones from detailed balance, normalized from the top
//! grain down so that `sum_a P(a <- b) = 1` (where the upward probabilities of the lowest grains
//! add to more than 1 the downward part alone is normalized, and the note says how many grains);
//! or `strong`, `P(a <- b) = pi_a / Q_w`. The collision frequency `omega` (1/s) is given
//! (`--collision-frequency`) or that of Lennard-Jones collisions with the bath gas,
//! `omega = pi sigma^2 sqrt(8 k_B T / (pi mu)) Omega^(2,2)* [M]`, `[M] = p / k_B T`, with the
//! reduced collision integral of Troe (J. Chem. Phys. 1977, 66, 4758):
//! `Omega^(2,2)* = (0.636 + 0.567 log10(k_B T / epsilon))^-1`, `sigma` and `epsilon` the
//! arithmetic and geometric means of those of the molecule (`--lennard-jones`, or
//! `lennard_jones` of a state in the annotations) and of the bath (`--bath SIGMA:EPS:MASS`).
//!
//! **Phenomenological rate constants** ([`Phenomenological`]) between the species (the wells
//! and the thermal states):
//!
//! - by contraction (a proposal of the design, section 3.7): the grains are eliminated in the
//!   greedy order of RCMC except the last one of every species; the superstate generator of
//!   closure A (and B) is the matrix of the rate constants, well-skipping steps included;
//! - by eigenvalues (Bartis and Widom, J. Chem. Phys. 1974, 60, 3474; Klippenstein and Miller,
//!   J. Phys. Chem. A 2002, 106, 9267): the `n` slowest eigenvalues of the symmetrized generator
//!   (`n` the number of species), their eigenvectors summed over the grains of every species,
//!   `Z`, and `k = Z Lambda Z^-1`; the gap between the slowest relaxation of the energies and
//!   the fastest chemical eigenvalue says whether the species are well defined.
//!
//! Every species also has its high-pressure rate constants (`rrkm`, the rate set) for the
//! fall-off `k / k_inf`.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::collections::BTreeMap;
use std::f64::consts::PI;

use faer::Mat;

use super::conditions::Conditions;
use super::contract::Eliminator;
use super::generator::{scaled_symmetric_eigen, Generator};
use super::network::{BottleneckKind, KineticNetwork};
use super::rate::tst::SymmetryChoice;
use super::rate::RateSet;
use super::rrkm::{ln_inverse_planck, Rrkm, RrkmSettings, StateCount};
use super::solve::Resolved;
use crate::domain::Units;
use crate::error::{MoptError, MoptResult};
use crate::math::stable::log_sum_exp;

/// The model of the energy transferred in a collision.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CollisionModel {
    /// Downward probabilities `exp(-dE / alpha)` with `alpha` (cm^-1).
    ExponentialDown {
        /// The mean energy transferred downwards (cm^-1).
        alpha: f64,
    },
    /// Every collision leaves the Boltzmann distribution of the well.
    Strong,
}

impl CollisionModel {
    /// `exponential_down` (with `alpha`) or `strong`.
    pub fn parse(text: &str, alpha: Option<f64>) -> MoptResult<Self> {
        match text.to_ascii_lowercase().as_str() {
            "strong" => Ok(Self::Strong),
            "exponential_down" | "exp_down" => match alpha {
                Some(a) if a > 0.0 && a.is_finite() => Ok(Self::ExponentialDown { alpha: a }),
                _ => Err(MoptError::config(
                    "the collision model exponential_down needs the mean energy transferred \
                     downwards (--energy-transfer CM1, positive)",
                )),
            },
            other => Err(MoptError::parse(format!(
                "unknown collision model '{other}' (exponential_down, strong)"
            ))),
        }
    }

    /// The name.
    pub fn name(self) -> &'static str {
        match self {
            Self::ExponentialDown { .. } => "exponential_down",
            Self::Strong => "strong",
        }
    }
}

/// A bath gas: Lennard-Jones diameter (Angstrom), well depth (K), mass (amu).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bath {
    /// Diameter (Angstrom).
    pub sigma: f64,
    /// Well depth over `k_B` (K).
    pub epsilon: f64,
    /// Mass (amu).
    pub mass: f64,
}

impl Bath {
    /// `SIGMA:EPS:MASS`.
    pub fn parse(text: &str) -> MoptResult<Self> {
        let v: Vec<f64> = text
            .split(':')
            .map(|x| x.trim().parse::<f64>())
            .collect::<Result<_, _>>()
            .map_err(|_| MoptError::parse(format!("--bath: '{text}' is not SIGMA:EPS:MASS")))?;
        match v.as_slice() {
            [s, e, m] if *s > 0.0 && *e > 0.0 && *m > 0.0 => Ok(Self {
                sigma: *s,
                epsilon: *e,
                mass: *m,
            }),
            _ => Err(MoptError::parse(format!(
                "--bath: '{text}' is not SIGMA:EPS:MASS (Angstrom, K, amu; positive)"
            ))),
        }
    }
}

/// The settings of `egme`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EgmeSettings {
    /// Grain width (cm^-1).
    pub grain: f64,
    /// The grains reach this many `k_B T` above the highest saddle point or well.
    pub above_kt: f64,
    /// The collision model (`None`: refused with what is needed).
    pub collision: Option<CollisionModel>,
    /// The collision frequency (1/s), when it is given.
    pub frequency: Option<f64>,
    /// The bath gas of the Lennard-Jones collision frequency.
    pub bath: Option<Bath>,
    /// The Lennard-Jones diameter (Angstrom) and depth (K) of every well without its own.
    pub lennard_jones: Option<(f64, f64)>,
}

impl Default for EgmeSettings {
    fn default() -> Self {
        Self {
            grain: 50.0,
            above_kt: 40.0,
            collision: None,
            frequency: None,
            bath: None,
            lennard_jones: None,
        }
    }
}

/// Troe's reduced collision integral `Omega^(2,2)*` at `k_B T / epsilon`.
pub fn troe_collision_integral(reduced_temperature: f64) -> f64 {
    1.0 / (0.636 + 0.567 * reduced_temperature.log10())
}

/// The Lennard-Jones collision frequency (1/s) of a molecule (diameter Angstrom, depth K, mass
/// amu) with `bath` at `temperature` (K) and `pressure` (Pa).
pub fn lennard_jones_frequency(
    molecule: (f64, f64),
    mass: f64,
    bath: &Bath,
    temperature: f64,
    pressure: f64,
) -> f64 {
    let sigma = 0.5 * (molecule.0 + bath.sigma) * 1e-10;
    let epsilon = (molecule.1 * bath.epsilon).sqrt();
    let mu = mass * bath.mass / (mass + bath.mass) * Units::AMU_TO_KG;
    let kt = Units::BOLTZMANN_J_PER_K * temperature;
    let speed = (8.0 * kt / (PI * mu)).sqrt();
    let density = pressure / kt;
    PI * sigma * sigma * speed * troe_collision_integral(temperature / epsilon) * density
}

/// The phenomenological rate constants of a grained model.
#[derive(Debug, Clone, PartialEq)]
pub struct Phenomenological {
    /// The species (states of the rate set).
    pub species: Vec<usize>,
    /// The rate constants by contraction, closure A: `k[i][j]` from species `j` to `i` (1/s,
    /// with the diagonal).
    pub contraction_a: Mat<f64>,
    /// The same with closure B.
    pub contraction_b: Mat<f64>,
    /// The rate constants from the chemically significant eigenpairs (`None` when the
    /// eigenvectors are not resolved in double precision, [`Phenomenological::eigen_note`]).
    pub eigen: Option<Mat<f64>>,
    /// Why `eigen` is missing: the components of a slow eigenvector on a species of a far
    /// larger weight are below the precision of the decomposition (their sums do not vanish as
    /// the conservation of the population requires), the limitation of the eigenvalue analysis
    /// at large ranges of the weights (Frankcombe and Smith, Theor. Chem. Acc. 2009, 124, 303).
    pub eigen_note: Option<String>,
    /// The slowest eigenvalues of the generator (1/s, negative or zero), one more than the
    /// species.
    pub eigenvalues: Vec<f64>,
    /// The fastest chemical eigenvalue over the slowest relaxation of the energies
    /// (`|lambda_n| / |lambda_{n-1}|`): the separation of the species.
    pub gap: f64,
    /// The high-pressure (canonical) rate constants of the rate set between the species.
    pub high_pressure: Mat<f64>,
}

/// The energy-grained master equation.
#[derive(Debug, Clone, Copy)]
pub struct Egme {
    /// Its settings.
    pub settings: EgmeSettings,
    /// The counting of states (those of the rate theory `rrkm`).
    pub rrkm: RrkmSettings,
    /// The symmetry numbers.
    pub symmetry: SymmetryChoice,
}

/// One fine state.
struct Fine {
    owner: usize,
    grain: usize,
    ln_pi: f64,
}

/// `ln` of the counted states of `count` per grain: `ln sum N_j f(j) exp(-(E_j - reference) / kT)`
/// over the cells of every grain (`f(j) = N_j` for a well, `W_j delta` for a saddle point).
fn per_grain(
    count: &StateCount,
    values: &[f64],
    e_min: f64,
    grain: f64,
    grains: usize,
    kt: f64,
    reference: f64,
) -> Vec<f64> {
    let mut terms: Vec<Vec<f64>> = vec![Vec::new(); grains];
    for (j, v) in values.iter().enumerate() {
        if (*v).partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
            continue;
        }
        let e = count.e0 + j as f64 * count.cell;
        let g = ((e - e_min) / grain).floor();
        if g < 0.0 {
            continue;
        }
        let g = (g as usize).min(grains - 1);
        terms[g].push(v.ln() - (e - reference) / kt);
    }
    terms.iter().map(|t| log_sum_exp(t)).collect()
}

impl Egme {
    /// The collision frequency of the well `k` of `net`.
    fn frequency(&self, net: &KineticNetwork, k: usize, c: &Conditions) -> MoptResult<f64> {
        if let Some(w) = self.settings.frequency {
            return Ok(w);
        }
        let s = &net.states[k];
        let lj = s.extra.lennard_jones.or(self.settings.lennard_jones);
        match (lj, self.settings.bath, &s.system) {
            (Some(lj), Some(bath), Some(sys)) => Ok(lennard_jones_frequency(
                lj,
                sys.masses_amu().iter().sum(),
                &bath,
                c.temperature,
                c.pressure,
            )),
            _ => Err(MoptError::config(format!(
                "egme: {} has no collision frequency: give --collision-frequency (1/s), or the \
                 bath gas (--bath SIGMA:EPS:MASS) with the Lennard-Jones parameters of the \
                 molecule (--lennard-jones SIGMA:EPS, or lennard_jones in the annotations) and \
                 the pressure",
                s.label
            ))),
        }
    }

    /// The generator of the grains of `net` with the rate constants `rates` (of `rrkm`).
    pub fn resolve(
        &self,
        net: &KineticNetwork,
        rates: &RateSet,
        c: &Conditions,
    ) -> MoptResult<Resolved> {
        if rates.theory != "rrkm" {
            return Err(MoptError::config(format!(
                "the model egme counts the states of the wells as the rate theory rrkm does: give \
                 --rate-theory rrkm (not {})",
                rates.theory
            )));
        }
        if !rates.detailed_balance {
            return Err(MoptError::unsupported(
                "the model egme needs detailed balance (flux form); the rate constants of this \
                 network are directed (MultiOptPy's clamp, or a cycle that converts species)",
            ));
        }
        let Some(collision) = self.settings.collision else {
            return Err(MoptError::config(
                "the model egme needs a collision model: --collision-model strong, or \
                 exponential_down with --energy-transfer CM1",
            ));
        };
        let grain = self.settings.grain * Units::WAVENUMBER_TO_HARTREE;
        if !(grain > 0.0 && self.settings.above_kt >= 5.0) {
            return Err(MoptError::config(
                "egme: the grain must be positive and the grains must reach at least 5 kT \
                 above the highest saddle point",
            ));
        }
        let kt = c.kt();
        let reference = rates.reference;
        let theory = Rrkm::new(self.rrkm, self.symmetry)?;
        let ns = rates.states.len();
        // The wells: states with frequencies and a structure (not channels).
        let mut wells: Vec<Option<StateCount>> = vec![None; ns];
        for (i, &k) in rates.states.iter().enumerate() {
            let s = &net.states[k];
            if s.fragments.is_empty() {
                if let Some(count) = theory.count_state(s, c, Some(1))? {
                    wells[i] = Some(count);
                }
            }
        }
        if wells.iter().all(Option::is_none) {
            return Err(MoptError::config(
                "egme: no state has the frequencies and the structure that the grains need",
            ));
        }
        // The energy axis.
        let e_min = wells
            .iter()
            .flatten()
            .map(|w| w.e0)
            .fold(f64::INFINITY, f64::min);
        let mut e_top = wells
            .iter()
            .flatten()
            .map(|w| w.e0)
            .fold(f64::NEG_INFINITY, f64::max);
        let mut ts_counts: BTreeMap<usize, StateCount> = BTreeMap::new();
        for step in &rates.steps {
            let b = &net.bottlenecks[step.bottleneck];
            if b.kind != BottleneckKind::Saddle || step.clamped {
                continue;
            }
            if let Some(count) = theory.count_bottleneck(b, c, Some(1))? {
                e_top = e_top.max(count.e0);
                ts_counts.insert(step.bottleneck, count);
            }
        }
        e_top += self.settings.above_kt * kt;
        let grains = ((e_top - e_min) / grain).ceil().max(1.0) as usize;
        let cells = |e0: f64| {
            ((e_top - e0) / (self.rrkm.cell * Units::WAVENUMBER_TO_HARTREE))
                .ceil()
                .max(1.0) as usize
                + 1
        };
        // Recount up to the top of the axis.
        for (i, &k) in rates.states.iter().enumerate() {
            if let Some(w) = &wells[i] {
                wells[i] = theory.count_state(&net.states[k], c, Some(cells(w.e0)))?;
            }
        }
        for (key, count) in ts_counts.iter_mut() {
            let n = cells(count.e0);
            if let Some(full) = theory.count_bottleneck(&net.bottlenecks[*key], c, Some(n))? {
                *count = full;
            }
        }
        // The fine states.
        let mut fine: Vec<Fine> = Vec::new();
        let mut index: Vec<Vec<Option<usize>>> = vec![Vec::new(); ns];
        let mut notes = Vec::new();
        for i in 0..ns {
            match &wells[i] {
                Some(w) => {
                    let ln = per_grain(w, &w.counts, e_min, grain, grains, kt, reference);
                    let shift = -(w.outer - 0.0) / kt;
                    index[i] = vec![None; grains];
                    let mut held = Vec::new();
                    for (g, v) in ln.iter().enumerate() {
                        if *v > f64::NEG_INFINITY {
                            index[i][g] = Some(fine.len());
                            fine.push(Fine {
                                owner: i,
                                grain: g,
                                ln_pi: v + shift,
                            });
                            held.push(v + shift);
                        }
                    }
                    let share = (log_sum_exp(&held) - rates.ln_pi[i]).exp();
                    if (share - 1.0).abs() > 1e-6 {
                        notes.push(format!(
                            "egme: the grains of {} hold {share:.6} of its thermal population \
                             (raise --egme-above)",
                            rates.labels[i]
                        ));
                    }
                }
                None => {
                    index[i] = vec![Some(fine.len())];
                    fine.push(Fine {
                        owner: i,
                        grain: usize::MAX,
                        ln_pi: rates.ln_pi[i],
                    });
                }
            }
        }
        let ln_h = ln_inverse_planck(c);
        let mut fluxes: Vec<(usize, usize, f64)> = Vec::new();
        // Collisions.
        let mut unnormalized = 0usize;
        for i in 0..ns {
            if wells[i].is_none() {
                continue;
            }
            let omega = self.frequency(net, rates.states[i], c)?;
            if !(omega > 0.0 && omega.is_finite()) {
                return Err(MoptError::config(format!(
                    "egme: the collision frequency of {} must be positive",
                    rates.labels[i]
                )));
            }
            let members: Vec<usize> = index[i].iter().flatten().copied().collect();
            let energy = |f: usize| e_min + (fine[f].grain as f64 + 0.5) * grain;
            match collision {
                CollisionModel::Strong => {
                    let ln_q =
                        log_sum_exp(&members.iter().map(|&f| fine[f].ln_pi).collect::<Vec<_>>());
                    for (x, &a) in members.iter().enumerate() {
                        for &b in &members[x + 1..] {
                            fluxes.push((a, b, omega.ln() + fine[a].ln_pi + fine[b].ln_pi - ln_q));
                        }
                    }
                }
                CollisionModel::ExponentialDown { alpha } => {
                    let alpha = alpha * Units::WAVENUMBER_TO_HARTREE;
                    let m = members.len();
                    // ln N_b from the top down.
                    let mut ln_norm = vec![0.0; m];
                    for xb in (0..m).rev() {
                        let b = members[xb];
                        let down: Vec<f64> = (0..=xb)
                            .map(|xa| -(energy(b) - energy(members[xa])) / alpha)
                            .collect();
                        let ln_d = log_sum_exp(&down);
                        // U_b = sum_{a above b} P(b <- a) pi_a / pi_b.
                        let up: Vec<f64> = (xb + 1..m)
                            .map(|xa| {
                                let a = members[xa];
                                -(energy(a) - energy(b)) / alpha - ln_norm[xa] + fine[a].ln_pi
                                    - fine[b].ln_pi
                            })
                            .collect();
                        let u = log_sum_exp(&up).exp();
                        ln_norm[xb] = if u < 1.0 - 1e-12 {
                            ln_d - (1.0 - u).ln()
                        } else {
                            unnormalized += 1;
                            ln_d
                        };
                    }
                    for xb in 0..m {
                        let b = members[xb];
                        for &a in &members[..xb] {
                            fluxes.push((
                                a,
                                b,
                                omega.ln() - (energy(b) - energy(a)) / alpha - ln_norm[xb]
                                    + fine[b].ln_pi,
                            ));
                        }
                    }
                }
            }
        }
        if unnormalized > 0 {
            notes.push(format!(
                "egme: at {unnormalized} low grains the upward probabilities of exponential_down \
                 add to more than 1; their downward part alone is normalized (they collide more \
                 often than omega)"
            ));
        }
        // Reactions.
        let mut moved = 0usize;
        let nearest = |i: usize, g: usize, moved: &mut usize| -> Option<usize> {
            let row = &index[i];
            if row.len() == 1 {
                return row[0];
            }
            if let Some(f) = row.get(g).copied().flatten() {
                return Some(f);
            }
            *moved += 1;
            (0..g.min(row.len())).rev().find_map(|x| row[x])
        };
        let mut thermal_steps = 0usize;
        for step in &rates.steps {
            let [a, z] = step.states;
            let ln_f: f64 = step.flux.factors.iter().map(|(_, f)| f).sum();
            match ts_counts.get(&step.bottleneck) {
                Some(ts) if wells[a].is_some() || wells[z].is_some() => {
                    let w = ts.sum_of_states();
                    let values: Vec<f64> = w.iter().map(|x| x * ts.cell).collect();
                    let ln = per_grain(ts, &values, e_min, grain, grains, kt, reference);
                    let share = net.bottlenecks[step.bottleneck]
                        .share
                        .filter(|s| *s > 0.0 && *s < 1.0)
                        .map_or(0.0, f64::ln);
                    let shift = -ts.outer / kt + ln_h + ln_f + share;
                    let held = (log_sum_exp(&ln) + shift - (step.ln_k[0] + rates.ln_pi[a])).exp();
                    if (held - 1.0).abs() > 1e-6 {
                        notes.push(format!(
                            "egme: the grains hold {held:.6} of the canonical flux of {} (raise \
                             --egme-above)",
                            net.bottlenecks[step.bottleneck].label
                        ));
                    }
                    for (g, v) in ln.iter().enumerate() {
                        if *v == f64::NEG_INFINITY {
                            continue;
                        }
                        let (Some(fa), Some(fz)) =
                            (nearest(a, g, &mut moved), nearest(z, g, &mut moved))
                        else {
                            continue;
                        };
                        if fa != fz {
                            fluxes.push((fa, fz, v + shift));
                        }
                    }
                }
                _ => {
                    // The canonical flux, distributed over the grains of the well(s).
                    let ln_flux = step.ln_k[0] + rates.ln_pi[a];
                    match (&wells[a], &wells[z]) {
                        (None, None) => {
                            if let (Some(fa), Some(fz)) = (index[a][0], index[z][0]) {
                                fluxes.push((fa, fz, ln_flux));
                            }
                        }
                        (Some(_), _) | (_, Some(_)) => {
                            thermal_steps += 1;
                            let (side, other) = if wells[a].is_some() { (a, z) } else { (z, a) };
                            for f in index[side].iter().flatten() {
                                let g = fine[*f].grain;
                                let share = fine[*f].ln_pi - rates.ln_pi[side];
                                let Some(fo) = nearest(other, g, &mut moved) else {
                                    continue;
                                };
                                if fo != *f {
                                    fluxes.push((*f, fo, ln_flux + share));
                                }
                            }
                        }
                    }
                }
            }
        }
        if moved > 0 {
            notes.push(format!(
                "egme: {moved} reactive grain(s) fell where a well has no states; their flux went \
                 to the nearest lower grain of that well"
            ));
        }
        if thermal_steps > 0 {
            notes.push(format!(
                "egme: {thermal_steps} step(s) without counted states at the bottleneck (crossing \
                 points, electron transfers, barrierless steps, saddle points without frequencies) \
                 join the grains with their canonical flux (valid near the high-pressure limit)"
            ));
        }
        let labels: Vec<String> = fine
            .iter()
            .map(|f| {
                if f.grain == usize::MAX {
                    rates.labels[f.owner].clone()
                } else {
                    format!(
                        "{}:{:.0}",
                        rates.labels[f.owner],
                        ((e_min + (f.grain as f64 + 0.5) * grain) - e_min)
                            / Units::WAVENUMBER_TO_HARTREE
                    )
                }
            })
            .collect();
        let ln_pi: Vec<f64> = fine.iter().map(|f| f.ln_pi).collect();
        let generator = Generator::from_fluxes(labels, ln_pi, &fluxes)?;
        let owner: Vec<usize> = fine.iter().map(|f| f.owner).collect();
        notes.push(format!(
            "egme: {} grains of {} cm^-1 per well from the lowest well up to {:.0} cm^-1, {} fine \
             states, collisions {}",
            grains,
            self.settings.grain,
            (e_top - e_min) / Units::WAVENUMBER_TO_HARTREE,
            fine.len(),
            collision.name()
        ));
        let mut resolved = Resolved {
            generator,
            owner,
            states: ns,
            model: "egme".to_string(),
            phenomenological: None,
            notes,
        };
        let high = high_pressure(rates);
        resolved.phenomenological = Some(phenomenological(&resolved, high)?);
        Ok(resolved)
    }
}

/// The canonical rate constants between the states of a rate set (1/s, with the diagonal).
pub fn high_pressure(rates: &RateSet) -> Mat<f64> {
    let n = rates.states.len();
    let mut k = Mat::<f64>::zeros(n, n);
    for s in &rates.steps {
        let [a, z] = s.states;
        k[(z, a)] += s.ln_k[0].exp();
        k[(a, z)] += s.ln_k[1].exp();
    }
    for j in 0..n {
        let out: f64 = (0..n).filter(|&i| i != j).map(|i| k[(i, j)]).sum();
        k[(j, j)] = -out;
    }
    k
}

/// The phenomenological rate constants of a resolved model, by contraction and by eigenvalues.
pub fn phenomenological(r: &Resolved, high_pressure: Mat<f64>) -> MoptResult<Phenomenological> {
    let rates = r.generator.rates();
    let n = rates.n;
    let species = r.states;
    // Contraction: the greedy order, the last fine state of every species kept.
    let mut left = vec![0usize; species];
    for &o in &r.owner {
        left[o] += 1;
    }
    let mut e = Eliminator::new(&rates);
    loop {
        let mut best: Option<(usize, f64)> = None;
        for j in e.kept() {
            if left[r.owner[j]] <= 1 {
                continue;
            }
            let o = e.out(j);
            if o > 0.0 && best.is_none_or(|(_, b)| o > b) {
                best = Some((j, o));
            }
        }
        let Some((j, _)) = best else {
            break;
        };
        e.eliminate(j)?;
        left[r.owner[j]] -= 1;
    }
    let contraction = e.snapshot();
    let kept = &contraction.kept;
    // The kept fine state of every species.
    let mut of_species = vec![usize::MAX; species];
    for (x, &f) in kept.iter().enumerate() {
        if of_species[r.owner[f]] == usize::MAX {
            of_species[r.owner[f]] = x;
        }
    }
    if of_species.contains(&usize::MAX) {
        return Err(MoptError::calc(
            "egme: a species has no fine state left after the contraction",
        ));
    }
    let ga = contraction.generator_a();
    let (gb, _) = contraction.generator_b();
    let pick =
        |g: &Mat<f64>| Mat::from_fn(species, species, |i, j| g[(of_species[i], of_species[j])]);
    let contraction_a = pick(&ga);
    let contraction_b = pick(&gb);
    // Eigenvalues of the symmetrized generator.
    let mut eigen = None;
    let mut eigen_note = None;
    let mut eigenvalues = Vec::new();
    let mut gap = f64::NAN;
    if let Some(ln_pi) = &r.generator.ln_pi {
        let k = rates.matrix();
        let s = Mat::from_fn(n, n, |i, j| {
            if i == j {
                k[(i, i)]
            } else {
                k[(i, j)] * (0.5 * (ln_pi[j] - ln_pi[i])).exp()
            }
        });
        let sym = Mat::from_fn(n, n, |i, j| 0.5 * (s[(i, j)] + s[(j, i)]));
        let (values, vectors) = scaled_symmetric_eigen(&sym)?;
        // Slowest first: the largest (closest to zero).
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| values[b].total_cmp(&values[a]));
        let take = species.min(n);
        eigenvalues = order.iter().take(take + 1).map(|&m| values[m]).collect();
        if n > species {
            let fastest = values[order[species - 1]].abs();
            let relax = values[order[species]].abs();
            gap = relax / fastest.max(f64::MIN_POSITIVE);
        }
        // Z[s][m] = sum over the fine states of s of pi^(1/2) u_m.
        let top = ln_pi.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let z = Mat::from_fn(species, take, |sp, m| {
            let col = order[m];
            (0..n)
                .filter(|&f| r.owner[f] == sp)
                .map(|f| (0.5 * (ln_pi[f] - top)).exp() * vectors[(f, col)])
                .sum::<f64>()
        });
        // Every slow mode but the stationary one moves population between the species
        // without changing its total: its column of Z sums to zero.
        let worst = (1..take)
            .map(|m| {
                let sum: f64 = (0..species).map(|s| z[(s, m)]).sum();
                let big = (0..species).map(|s| z[(s, m)].abs()).fold(0.0, f64::max);
                sum.abs() / big.max(f64::MIN_POSITIVE)
            })
            .fold(0.0, f64::max);
        let lu = z.partial_piv_lu();
        let inv = faer::linalg::solvers::Solve::solve(
            &lu,
            Mat::<f64>::identity(species, species).as_ref(),
        );
        if take < species {
            eigen_note = Some("fewer grains than species".to_string());
        } else if worst > 1e-8 {
            eigen_note = Some(format!(
                "the slow eigenvectors are not resolved in double precision (their components                  on the species sum to {worst:.1e} of their largest instead of 0: the weights of                  the species span too many orders); the contraction is the result"
            ));
        } else if inv.col_iter().all(|c| c.iter().all(|v| v.is_finite())) {
            let lambda = Mat::from_fn(
                species,
                species,
                |i, j| {
                    if i == j {
                        eigenvalues[i]
                    } else {
                        0.0
                    }
                },
            );
            eigen = Some(&z * &lambda * &inv);
        } else {
            eigen_note = Some("the slow eigenvectors are linearly dependent".to_string());
        }
    }
    Ok(Phenomenological {
        species: (0..species).collect(),
        contraction_a,
        contraction_b,
        eigen,
        eigen_note,
        eigenvalues,
        gap,
        high_pressure,
    })
}
