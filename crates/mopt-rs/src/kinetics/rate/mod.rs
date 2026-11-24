//! Rate constants in flux form: a [`RateTheory`] gives the free energies of the states and of
//! the bottlenecks and the prefactor, [`RateCorrection`]s multiply the equilibrium flux of a
//! bottleneck (tunnelling, a crossing probability, a given transmission coefficient), and
//! [`RateSet::assemble`] puts them together:
//!
//! ```text
//! ln pi[j] = -(G_j - G_ref) / RT
//! ln F[ab] = ln(prefactor) - (G_ts - G_ref) / RT + sum_c ln f_c(ab)
//! K[b][a]  = F[ab] / pi[a]
//! ```
//!
//! With the consistent clamp `G_ts` is replaced by `max(G_ts, G_a, G_b)` for both directions;
//! with MultiOptPy's clamp each direction is clamped by its own minimum, so the rates are kept
//! as directed entries and the generator claims no detailed balance. Every correction used
//! here is symmetric (the same factor in both directions, referred to the bottleneck), so it
//! belongs in `F`.
//!
//! States or bottlenecks without the energy a theory needs (no free energy in the file, no
//! frequencies) are left out and listed in [`RateSet::notes`], as MultiOptPy's `run_rcmc` drops
//! nodes without a free energy; nothing is guessed.
//!
//! Bimolecular steps (`state + S <-> other state`, [`crate::kinetics::species`]) take the
//! chemical potential of the species at its concentration on their free side, so that the
//! step from that side is pseudo-first order, `k' = k_2 [S]`.
//!
//! Built in: theories `eyring`, `tst`, and `rrkm` (the Boltzmann average of the microcanonical
//! rate constants from directly counted states, [`crate::kinetics::rrkm`]); corrections
//! `wigner`, `bell`, `eckart`, `skodje_truhlar`, `kappa`, `small_curvature` and
//! `zero_curvature` (tunnelling along the path of `freqmain -path`, [`small_curvature`]),
//! `kramers`, `grote_hynes`, `reactive_flux` (recrossing, [`recrossing`]), `collins_kimball`
//! (the diffusion limit of bimolecular steps, [`diffusion`]), `landau_zener`,
//! `zhu_nakamura` (sloped crossings above the crossing energy, [`zhu_nakamura`]; peaked
//! crossings at every energy, tunnelling below the crossing included,
//! [`nonadiabatic_tunnelling`]), `weak_coupling` (the Airy form of the weak-coupling limit,
//! [`weak_coupling`](mod@weak_coupling)), `cvt` (canonical variational TST of the steps with a path,
//! [`variational`]). Electron-transfer edges (`transfer` in the annotations) take their
//! bottleneck from the Marcus parabolas of [`electron_transfer`] instead of a saddle point,
//! non-adiabatic or adiabatic; every step keeps both directions, so saddle points, variational
//! dividing surfaces, electron transfers, and crossing points mix in one network.
//! Two names of the design stay refused with the reason: `marcus` as a correction of every step
//! (an electron transfer is a property of its edge) and `symmetry` (part of `tst`).
//!
//! MultiOptPy: `Utils/rcmc.py:RCMCQueue.pop` (Eyring rate constants of both directions of every
//! edge).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

pub mod couplings;
pub mod crossing;
pub mod diffusion;
pub mod electron_transfer;
pub mod encounter;
pub mod eyring;
pub mod inner_sphere;
pub mod nonadiabatic_tunnelling;
pub mod pcet;
pub mod recrossing;
pub mod small_curvature;
pub mod solvent_activation;
pub mod transfer_integral;
pub mod tst;
pub mod tunneling;
pub mod variational;
pub mod weak_coupling;
pub mod zhu_nakamura;

use std::sync::Arc;

use super::conditions::{Clamp, Conditions};
use super::generator::{Generator, RateEntry};
use super::network::{Bottleneck, KineticNetwork, KineticState};
use super::species::{wertz_correction, Association, Reservoir, SpeciesSettings, StepLink};
use crate::error::{MoptError, MoptResult};
use electron_transfer::transfer_barrier;
use encounter::association_top;

/// The free energy of `s` by `theory`: for a dissociation channel the sum of its fragments,
/// every fragment beyond the first one more free molecule converted to the standard
/// concentration `c°` (none for energies without a standard state).
fn state_free_energy(
    theory: &dyn RateTheory,
    s: &KineticState,
    c: &Conditions,
) -> MoptResult<Option<f64>> {
    if s.fragments.is_empty() {
        return theory.state_energy(s, c);
    }
    let mut sum = 0.0;
    for f in &s.fragments {
        match theory.state_energy(f, c)? {
            Some(g) => sum += g,
            None => return Ok(None),
        }
    }
    if let Some(from) = theory.standard_state(c) {
        sum += (s.fragments.len() - 1) as f64 * c.standard_state_shift(from);
    }
    Ok(Some(sum))
}

pub use crossing::{landau_zener_ln_average, landau_zener_ln_average_sloped, LandauZener};
pub use diffusion::CollinsKimball;
pub use eyring::Eyring;
pub use nonadiabatic_tunnelling::{peaked_model, PeakedAverage, PeakedCrossing, PeakedModel};
pub use recrossing::{grote_hynes_frequency, kramers_ln_kappa, GroteHynes, Kramers, ReactiveFlux};
pub use small_curvature::CurvatureTunnelling;
pub use tst::{QrrhoSettings, Tst};
pub use tunneling::{
    bell_ln_kappa, bell_truncated_ln_kappa, eckart_ln_kappa, eckart_ln_probability,
    skodje_truhlar_ln_kappa, Bell, BellBarrier, BellSettings, BellTruncation, Eckart, Kappa,
    SkodjeTruhlar, Wigner,
};
pub use variational::{cvt_ln_factor, Cvt};
pub use weak_coupling::{weak_coupling, WeakCoupling, WeakCouplingCorrection};
pub use zhu_nakamura::{adiabatic_model, AdiabaticModel, ZhuNakamura, ZhuNakamuraCrossing};

/// A named number that says when a correction applies (`u = h nu / k_B T`, `T / T_c`, ...).
#[derive(Debug, Clone, PartialEq)]
pub struct Diagnostic {
    /// Its name.
    pub name: String,
    /// Its value.
    pub value: f64,
}

impl Diagnostic {
    /// `name = value`.
    pub fn new(name: &str, value: f64) -> Self {
        Self {
            name: name.to_string(),
            value,
        }
    }
}

/// `ln F` of a bottleneck and what it was made of.
#[derive(Debug, Clone, PartialEq)]
pub struct LnFlux {
    /// `ln F` of the rate theory alone, relative to the reference energy.
    pub ln_tst: f64,
    /// The corrections: (name, ln factor).
    pub factors: Vec<(String, f64)>,
    /// Their diagnostics.
    pub diagnostics: Vec<Diagnostic>,
}

impl LnFlux {
    /// `ln F` with the corrections.
    pub fn total(&self) -> f64 {
        self.ln_tst + self.factors.iter().map(|(_, f)| f).sum::<f64>()
    }
}

/// What a correction may need about a step besides the bottleneck: the barriers from both
/// sides as they enter tunnelling (electronic, with the zero-point energies where the
/// frequencies of all three stationary points are known).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StepContext {
    /// `E(bottleneck) - E(state)` for both states of the step (Hartree).
    pub barriers: [f64; 2],
    /// Whether the zero-point energies are included.
    pub zero_point: bool,
}

/// A rate theory: the free energies of the states and the bottlenecks and the prefactor.
pub trait RateTheory: Send + Sync {
    /// Its name in the registry.
    fn name(&self) -> &str;

    /// The free energy (Hartree) of `state` that gives its weight `pi`, or `None` when the
    /// state lacks what is needed (the state is then left out).
    fn state_energy(&self, state: &KineticState, c: &Conditions) -> MoptResult<Option<f64>>;

    /// The free energy (Hartree) of the bottleneck, or `None` when it lacks what is needed.
    fn bottleneck_energy(&self, b: &Bottleneck, c: &Conditions) -> MoptResult<Option<f64>>;

    /// `ln` of the prefactor (1/s): `ln(k_B T / h)` by default.
    fn ln_prefactor(&self, c: &Conditions) -> f64 {
        c.ln_kt_over_h()
    }

    /// The concentration (mol/L) of the standard state of its free energies, which the
    /// bimolecular steps convert to `c°` ([`Conditions::standard_state_shift`]), or `None` for
    /// energies without a standard state (electronic energies). By default that of the
    /// thermochemistry, [`Conditions::free_energy_concentration`].
    fn standard_state(&self, c: &Conditions) -> Option<f64> {
        Some(c.free_energy_concentration())
    }

    /// What the energies are made of, for the outputs.
    fn describe(&self, c: &Conditions) -> String;
}

/// What a correction that is taken after the others knows of its step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AfterOthers {
    /// `ln k` of both directions (1/s) with the rate theory, the clamp, and the other
    /// corrections.
    pub ln_k: [f64; 2],
    /// `ln [S]` (mol/L) of the partner of a bimolecular step, as its rate constants use it.
    pub ln_concentration: Option<f64>,
}

/// A multiplicative correction of the equilibrium flux of a bottleneck.
pub trait RateCorrection: Send + Sync {
    /// Its name in the registry.
    fn name(&self) -> &str;

    /// `ln` of the factor and its diagnostics, or `None` when it does not apply to `b` (a
    /// crossing probability at a saddle point, tunnelling without an imaginary frequency).
    fn ln_factor(
        &self,
        b: &Bottleneck,
        step: &StepContext,
        net: &KineticNetwork,
        c: &Conditions,
    ) -> MoptResult<Option<(f64, Vec<Diagnostic>)>>;

    /// Whether the factor depends on the rate constants that the theory and the other
    /// corrections give the step (the diffusion limit of an association): it is then taken
    /// after them, by [`RateCorrection::ln_factor_after`].
    fn after_others(&self) -> bool {
        false
    }

    /// The factor of a correction taken after the others ([`RateCorrection::after_others`]);
    /// by default [`RateCorrection::ln_factor`].
    fn ln_factor_after(
        &self,
        b: &Bottleneck,
        step: &StepContext,
        net: &KineticNetwork,
        c: &Conditions,
        _after: &AfterOthers,
    ) -> MoptResult<Option<(f64, Vec<Diagnostic>)>> {
        self.ln_factor(b, step, net, c)
    }
}

/// The rate constants of one step.
#[derive(Debug, Clone, PartialEq)]
pub struct StepRate {
    /// The bottleneck (index into [`KineticNetwork::bottlenecks`]).
    pub bottleneck: usize,
    /// Its two states (indices into the states of the generator).
    pub states: [usize; 2],
    /// The flux and its parts.
    pub flux: LnFlux,
    /// `ln k` of both directions: `states[0] -> states[1]`, `states[1] -> states[0]` (1/s).
    pub ln_k: [f64; 2],
    /// The free energy of the bottleneck after the clamp (Hartree).
    pub energy: f64,
    /// Whether the clamp raised it.
    pub clamped: bool,
    /// The free energies (Hartree) of its two sides as the step sees them: those of its
    /// states, with the chemical potential of a free partner on its side.
    pub ends: [f64; 2],
    /// The side whose level the consistent clamp gave the bottleneck (`None`: its own).
    pub level: Option<usize>,
    /// The partner of a bimolecular step: (species, the side of `states` where it is free).
    pub partner: Option<(usize, usize)>,
    /// `d energy / d ends` of a bottleneck whose free energy follows from its sides (an
    /// electron transfer); 0 for a saddle point.
    pub slopes: [f64; 2],
    /// The rate theory of the step (`eyring`, `tst`, `marcus_nonadiabatic`, ...).
    pub rate: String,
}

impl StepRate {
    /// The tangent of `ln k` of both directions when the free energies of the two sides move
    /// by `d_ends` and that of the bottleneck by `d_ts` (consistent clamp or none).
    pub fn tangent(&self, d_ends: [f64; 2], d_ts: f64, kt: f64) -> [f64; 2] {
        let dg = match self.level {
            Some(side) => d_ends[side],
            None => d_ts + self.slopes[0] * d_ends[0] + self.slopes[1] * d_ends[1],
        };
        [-(dg - d_ends[0]) / kt, -(dg - d_ends[1]) / kt]
    }
}

/// The rate constants of a network.
#[derive(Debug, Clone)]
pub struct RateSet {
    /// The network states that are in the model (generator index -> network state index).
    pub states: Vec<usize>,
    /// The free energies of those states (Hartree).
    pub state_energies: Vec<f64>,
    /// The reference energy (the lowest of the states).
    pub reference: f64,
    /// `ln pi` of the states.
    pub ln_pi: Vec<f64>,
    /// The steps.
    pub steps: Vec<StepRate>,
    /// Whether every step is in flux form (detailed balance).
    pub detailed_balance: bool,
    /// What was left out and why.
    pub notes: Vec<String>,
    /// The rate theory.
    pub theory: String,
    /// What its energies are.
    pub description: String,
    /// The corrections.
    pub corrections: Vec<String>,
    /// `k_B T` (Hartree).
    pub kt: f64,
    /// Labels of the states of the generator.
    pub labels: Vec<String>,
    /// The species of the network and their chemical potentials.
    pub species: SpeciesRates,
}

/// The species of a rate assembly.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct SpeciesRates {
    /// Their names.
    pub names: Vec<String>,
    /// `G°` at `c°` (Hartree), with the conversion of the standard state and the treatment of
    /// the association (NaN for a species no step uses).
    pub energies: Vec<f64>,
    /// `ln([S] / c°)` (0 at the standard state).
    pub ln_ratio: Vec<f64>,
    /// The concentrations given (mol/L).
    pub concentrations: Vec<Option<f64>>,
    /// Whether a step of the model uses the species.
    pub used: Vec<bool>,
    /// The treatment of the association of every species.
    pub treatments: Vec<Association>,
    /// The free molecules of every species held by every state, when the network has a
    /// potential (no cycle of steps converts species); `None` otherwise.
    pub composition: Option<Vec<Vec<i32>>>,
    /// `c°` (mol/L).
    pub standard_concentration: f64,
    /// The conversion of the standard state of the free energies to `c°` (Hartree per
    /// molecule).
    pub shift: f64,
    /// Every species at `c°` (the rate constants of mass action).
    pub at_standard: bool,
    /// The Wertz-type correction of the states and bottlenecks (Hartree), when it is used.
    pub wertz: Option<(Vec<f64>, Vec<f64>)>,
}

impl SpeciesRates {
    /// The chemical potential (Hartree) of species `s`.
    pub fn mu(&self, s: usize, kt: f64) -> f64 {
        self.energies[s] + kt * self.ln_ratio[s]
    }

    /// The concentration (mol/L) of species `s` in the rate constants: `[S]` in pseudo-first
    /// order, `c°` at the standard state (a pseudo-first-order rate constant over it is the
    /// second-order one, L/mol/s).
    pub fn concentration(&self, s: usize) -> f64 {
        self.standard_concentration * self.ln_ratio[s].exp()
    }
}

/// Harmonic zero-point energy (Hartree) of the real wavenumbers `w` (cm^-1).
pub fn zero_point_energy(w: &[f64]) -> f64 {
    use crate::domain::Units;
    0.5 * w.iter().filter(|x| **x > 0.0).sum::<f64>()
        * Units::PLANCK_J_S
        * Units::SPEED_OF_LIGHT_CM_S
        / Units::HARTREE_TO_JOULE
}

fn tunnelling_barriers(b: &Bottleneck, net: &KineticNetwork) -> StepContext {
    let ends = [&net.states[b.states[0]], &net.states[b.states[1]]];
    let zpe = |f: &Option<Vec<f64>>| f.as_deref().map(zero_point_energy);
    // The partner of a bimolecular step counts on its free side.
    let partner = b.partner.as_ref().map(|p| {
        let s = &net.species[p.species].data;
        (p.free_side, s.energy, zpe(&s.frequencies))
    });
    let extra = |side: usize, zero_point: bool| -> f64 {
        match partner {
            Some((free, e, z)) if free == side => {
                e + if zero_point { z.unwrap_or(0.0) } else { 0.0 }
            }
            _ => 0.0,
        }
    };
    let partner_zpe = partner.is_none_or(|(_, _, z)| z.is_some());
    match (
        zpe(&b.frequencies),
        zpe(&ends[0].frequencies),
        zpe(&ends[1].frequencies),
    ) {
        (Some(z), Some(z0), Some(z1)) if partner_zpe => StepContext {
            barriers: [
                b.energy + z - ends[0].energy - z0 - extra(0, true),
                b.energy + z - ends[1].energy - z1 - extra(1, true),
            ],
            zero_point: true,
        },
        _ => StepContext {
            barriers: [
                b.energy - ends[0].energy - extra(0, false),
                b.energy - ends[1].energy - extra(1, false),
            ],
            zero_point: false,
        },
    }
}

impl RateSet {
    /// The rate constants of `net` by `theory` with `corrections` under `c` (species, if any,
    /// with the default settings: their concentrations are then needed by
    /// [`RateSet::assemble_with`]).
    pub fn assemble(
        net: &KineticNetwork,
        theory: &dyn RateTheory,
        corrections: &[Arc<dyn RateCorrection>],
        c: &Conditions,
    ) -> MoptResult<Self> {
        Self::assemble_with(net, theory, corrections, c, &SpeciesSettings::default())
    }

    /// The rate constants of `net` by `theory` with `corrections` under `c`, with the
    /// bimolecular steps at the concentrations of `species` (pseudo-first order) or at `c°`.
    ///
    /// Every step gets its rate constants from the free energies of its two sides as it sees
    /// them (a free partner `S` counted with `mu_S` on its side) and of its bottleneck, so that
    /// each step balances its reverse. When the free molecules of every state are consistent
    /// over the network (no cycle converts species), the states have the free energies
    /// `G + sum n mu` and the rates are in flux form; otherwise (a catalytic cycle that turns
    /// `S` into `P`) the network has no potential and the rates are directed.
    pub fn assemble_with(
        net: &KineticNetwork,
        theory: &dyn RateTheory,
        corrections: &[Arc<dyn RateCorrection>],
        c: &Conditions,
        species: &SpeciesSettings,
    ) -> MoptResult<Self> {
        c.check()?;
        net.check()?;
        let kt = c.kt();
        let mut notes = Vec::new();
        let mut index = vec![usize::MAX; net.states.len()];
        let mut states = Vec::new();
        let mut raw = Vec::new();
        for (k, s) in net.states.iter().enumerate() {
            match state_free_energy(theory, s, c)? {
                Some(g) if g.is_finite() => {
                    index[k] = states.len();
                    states.push(k);
                    raw.push(g);
                }
                Some(_) => {
                    return Err(MoptError::config(format!(
                        "{}: the free energy of {} is not finite",
                        theory.name(),
                        s.label
                    )))
                }
                None => notes.push(format!(
                    "{} is left out: it has no {} energy",
                    s.label,
                    theory.describe(c)
                )),
            }
        }
        if states.is_empty() {
            return Err(MoptError::config(format!(
                "no state of the network has the energies of {} ({})",
                theory.name(),
                theory.describe(c)
            )));
        }
        // The bottlenecks of the model.
        let mut candidates: Vec<(usize, [usize; 2], f64)> = Vec::new();
        for (k, b) in net.bottlenecks.iter().enumerate() {
            let [a, z] = b.states;
            if index[a] == usize::MAX || index[z] == usize::MAX {
                notes.push(format!(
                    "{} is left out: a state of it is not in the model",
                    b.label
                ));
                continue;
            }
            if b.transfer.is_some() || b.association.is_some() {
                // The bottleneck of an electron transfer or an association follows from its
                // two sides.
                candidates.push((k, [index[a], index[z]], f64::NAN));
                continue;
            }
            let Some(g_ts) = theory.bottleneck_energy(b, c)? else {
                notes.push(format!(
                    "{} is left out: it has no {} energy",
                    b.label,
                    theory.describe(c)
                ));
                continue;
            };
            // A share of the flux (an N-furcation) multiplies the rates both ways: a free
            // energy -kT ln(share) on top of the bottleneck.
            let g_ts = match b.share {
                Some(w) if w > 0.0 && w < 1.0 => g_ts - c.kt() * w.ln(),
                _ => g_ts,
            };
            candidates.push((k, [index[a], index[z]], g_ts));
        }
        let labels: Vec<String> = states
            .iter()
            .map(|&k| net.states[k].label.clone())
            .collect();
        // The Wertz-type correction of every structure.
        let wertz = match species.association {
            Association::Wertz { fraction } => {
                let wertz_of = |s: &KineticState| {
                    wertz_correction(
                        &s.label,
                        s.system.as_ref(),
                        s.symmetry_number,
                        fraction,
                        species.symmetry,
                        c,
                    )
                };
                let w_states = states
                    .iter()
                    .map(|&k| {
                        let s = &net.states[k];
                        if s.fragments.is_empty() {
                            wertz_of(s)
                        } else {
                            // A channel: every fragment for itself.
                            s.fragments.iter().map(wertz_of).sum()
                        }
                    })
                    .collect::<MoptResult<Vec<f64>>>()?;
                let w_steps = candidates
                    .iter()
                    .map(|&(k, _, _)| {
                        let b = &net.bottlenecks[k];
                        if b.transfer.is_some() || b.association.is_some() {
                            return Ok(0.0);
                        }
                        wertz_correction(
                            &b.label,
                            b.system.as_ref(),
                            b.symmetry_number,
                            fraction,
                            species.symmetry,
                            c,
                        )
                    })
                    .collect::<MoptResult<Vec<f64>>>()?;
                for (e, w) in raw.iter_mut().zip(&w_states) {
                    *e += w;
                }
                for (cand, w) in candidates.iter_mut().zip(&w_steps) {
                    cand.2 += w;
                }
                Some((w_states, w_steps))
            }
            _ => None,
        };
        // The species: their chemical potentials and the free molecules of every state.
        let links: Vec<StepLink> = candidates
            .iter()
            .map(|&(k, st, _)| StepLink {
                states: st,
                partner: net.bottlenecks[k].partner.as_ref().map(|p| {
                    // The side within the step of the state where the species is free.
                    (p.species, p.free_side)
                }),
            })
            .collect();
        let reservoir = Reservoir::of(net, theory, c, species, states.len(), &links)?;
        if reservoir.composition.is_none() {
            notes.push(
                "a cycle of steps converts species (a catalytic cycle): the network has no \
                 potential, the rate constants are directed (each step balances its reverse)"
                    .to_string(),
            );
        }
        let energies: Vec<f64> = match &reservoir.composition {
            Some(n) => raw
                .iter()
                .zip(n)
                .map(|(e, n)| e + reservoir.offset(n))
                .collect(),
            None => raw.clone(),
        };
        let reference = energies.iter().copied().fold(f64::INFINITY, f64::min);
        let ln_pi: Vec<f64> = energies.iter().map(|g| -(g - reference) / kt).collect();
        let ln_pre = theory.ln_prefactor(c);
        let mut steps = Vec::new();
        for (m, &(k, [ia, iz], g_raw)) in candidates.iter().enumerate() {
            let b = &net.bottlenecks[k];
            // The frame of the step: the free molecules of its bound side.
            let frame = match &reservoir.step_composition {
                Some(n) => reservoir.offset(&n[m]),
                None => 0.0,
            };
            let mut ends = [raw[ia] + frame, raw[iz] + frame];
            if let Some((s, free)) = links[m].partner {
                ends[free] += reservoir.mu(s);
            }
            let (g_ts, ln_pre, slopes, rate) = match (&b.transfer, &b.association) {
                (Some(tr), _) => {
                    let t = transfer_barrier(tr, b.coupling, ends, c.temperature, &b.label)?;
                    (t.top, t.ln_prefactor, t.slopes, tr.form.name().to_string())
                }
                (None, Some(step)) => {
                    let (top, slopes) = association_top(step, ends, c, ln_pre, &b.label)?;
                    let name = if c.viscosity.is_some() {
                        "collins_kimball"
                    } else {
                        "capture"
                    };
                    (top, ln_pre, slopes, name.to_string())
                }
                (None, None) => (g_raw + frame, ln_pre, [0.0; 2], theory.name().to_string()),
            };
            let step = tunnelling_barriers(b, net);
            let mut factors = Vec::new();
            let mut diagnostics = Vec::new();
            for corr in corrections.iter().filter(|x| !x.after_others()) {
                if let Some((ln_f, mut d)) = corr.ln_factor(b, &step, net, c)? {
                    if !ln_f.is_finite() {
                        return Err(MoptError::config(format!(
                            "{}: the correction {} is not finite",
                            b.label,
                            corr.name()
                        )));
                    }
                    factors.push((corr.name().to_string(), ln_f));
                    diagnostics.append(&mut d);
                }
            }
            let sum_f: f64 = factors.iter().map(|(_, f)| f).sum();
            let [ga, gz] = ends;
            let (energy, clamped, level, ln_k, ln_tst) = match c.clamp {
                Clamp::Consistent => {
                    let g = g_ts.max(ga).max(gz);
                    let level = if g > g_ts {
                        Some(if ga >= gz { 0 } else { 1 })
                    } else {
                        None
                    };
                    let ln_tst = ln_pre - (g - reference) / kt;
                    (
                        g,
                        g > g_ts,
                        level,
                        [
                            ln_pre - (g - ga) / kt + sum_f,
                            ln_pre - (g - gz) / kt + sum_f,
                        ],
                        ln_tst,
                    )
                }
                Clamp::None => {
                    let ln_tst = ln_pre - (g_ts - reference) / kt;
                    (
                        g_ts,
                        false,
                        None,
                        [
                            ln_pre - (g_ts - ga) / kt + sum_f,
                            ln_pre - (g_ts - gz) / kt + sum_f,
                        ],
                        ln_tst,
                    )
                }
                Clamp::Legacy => {
                    // MultiOptPy: k_uv = (k_B T / h) exp(-(max(E_TS, E_u) - E_u) / RT).
                    let ln_tst = ln_pre - (g_ts - reference) / kt;
                    (
                        g_ts,
                        g_ts < ga.max(gz),
                        None,
                        [
                            ln_pre - (g_ts.max(ga) - ga) / kt + sum_f,
                            ln_pre - (g_ts.max(gz) - gz) / kt + sum_f,
                        ],
                        ln_tst,
                    )
                }
            };
            // The corrections that need the rate constants of the step (the diffusion limit).
            let mut ln_k = ln_k;
            for corr in corrections.iter().filter(|x| x.after_others()) {
                let after = AfterOthers {
                    ln_k,
                    ln_concentration: links[m]
                        .partner
                        .map(|(s, _)| c.standard_concentration().ln() + reservoir.ln_ratio[s]),
                };
                if let Some((ln_f, mut d)) = corr.ln_factor_after(b, &step, net, c, &after)? {
                    if !ln_f.is_finite() {
                        return Err(MoptError::config(format!(
                            "{}: the correction {} is not finite",
                            b.label,
                            corr.name()
                        )));
                    }
                    factors.push((corr.name().to_string(), ln_f));
                    diagnostics.append(&mut d);
                    ln_k = [ln_k[0] + ln_f, ln_k[1] + ln_f];
                }
            }
            steps.push(StepRate {
                bottleneck: k,
                states: [ia, iz],
                flux: LnFlux {
                    ln_tst,
                    factors,
                    diagnostics,
                },
                ln_k,
                energy,
                clamped,
                ends,
                level,
                partner: links[m].partner,
                slopes,
                rate,
            });
        }
        let detailed_balance = reservoir.composition.is_some()
            && (c.clamp != Clamp::Legacy || steps.iter().all(|s| !s.clamped));
        Ok(Self {
            states,
            state_energies: energies,
            reference,
            ln_pi,
            steps,
            detailed_balance,
            notes,
            theory: theory.name().to_string(),
            description: theory.describe(c),
            corrections: corrections.iter().map(|c| c.name().to_string()).collect(),
            kt,
            labels,
            species: SpeciesRates {
                names: net.species.iter().map(|s| s.name.clone()).collect(),
                energies: reservoir.energies,
                ln_ratio: reservoir.ln_ratio,
                concentrations: reservoir.concentrations,
                used: reservoir.used,
                treatments: reservoir.treatments,
                composition: reservoir.composition,
                standard_concentration: c.standard_concentration(),
                shift: reservoir.shift,
                at_standard: species.at_standard,
                wertz,
            },
        })
    }

    /// The generator: in flux form when every step has detailed balance, else directed.
    pub fn generator(&self) -> MoptResult<Generator> {
        if self.detailed_balance {
            let fluxes: Vec<(usize, usize, f64)> = self
                .steps
                .iter()
                .map(|s| {
                    (
                        s.states[0],
                        s.states[1],
                        s.ln_k[0] + self.ln_pi[s.states[0]],
                    )
                })
                .collect();
            Generator::from_fluxes(self.labels.clone(), self.ln_pi.clone(), &fluxes)
        } else {
            let entries: Vec<RateEntry> = self
                .steps
                .iter()
                .flat_map(|s| {
                    [
                        RateEntry {
                            from: s.states[0],
                            to: s.states[1],
                            ln_k: s.ln_k[0],
                        },
                        RateEntry {
                            from: s.states[1],
                            to: s.states[0],
                            ln_k: s.ln_k[1],
                        },
                    ]
                })
                .collect();
            Generator::from_directed(self.labels.clone(), &entries)
        }
    }

    /// The generator index of the network state `k`.
    pub fn index_of(&self, k: usize) -> Option<usize> {
        self.states.iter().position(|&s| s == k)
    }
}

/// Makes a rate theory.
pub type RateTheoryFactory =
    Arc<dyn Fn(&RateSettings) -> MoptResult<Box<dyn RateTheory>> + Send + Sync>;
/// Makes a correction.
pub type RateCorrectionFactory =
    Arc<dyn Fn(&RateSettings) -> MoptResult<Arc<dyn RateCorrection>> + Send + Sync>;

/// What the theories and corrections are made with.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct RateSettings {
    /// The low-frequency treatment and symmetry numbers of `tst`.
    pub qrrho: QrrhoSettings,
    /// The transmission coefficient of `kappa` (for every step without its own).
    pub kappa: Option<f64>,
    /// The truncation and the barrier of `bell`.
    pub bell: BellSettings,
    /// The friction (1/s) of `kramers` and `grote_hynes` for every saddle point without its
    /// own.
    pub friction: Option<f64>,
    /// The counting of states of `rrkm`.
    pub rrkm: crate::kinetics::rrkm::RrkmSettings,
}

/// Names of the design that are refused, with what they need.
const NOT_IMPLEMENTED: &[(&str, &str)] = &[
    ("marcus", "nothing more: an electron transfer is a property of its edge (\"transfer\": \"nonadiabatic\" or \"adiabatic\", \"reorganization_hartree\", and the coupling in the annotations), not a correction of every step"),
    ("symmetry", "nothing: symmetry numbers are part of the rate theories tst and rrkm (--symmetry)"),
];

fn refused(name: &str) -> Option<MoptError> {
    NOT_IMPLEMENTED
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(n, why)| {
            MoptError::unsupported(format!(
                "the rate model '{n}' is not implemented: it needs {why}"
            ))
        })
}

/// Rate theories by name.
pub struct RateTheoryRegistry {
    entries: Vec<(String, RateTheoryFactory)>,
}

impl RateTheoryRegistry {
    /// `eyring`, `tst`, and `rrkm`.
    pub fn builtin() -> Self {
        let mut r = Self {
            entries: Vec::new(),
        };
        r.register(
            "eyring",
            Arc::new(|_: &RateSettings| Ok(Box::new(Eyring) as Box<dyn RateTheory>)),
        );
        r.register(
            "tst",
            Arc::new(|s: &RateSettings| Ok(Box::new(Tst::new(s.qrrho)) as Box<dyn RateTheory>)),
        );
        r.register(
            "rrkm",
            Arc::new(|s: &RateSettings| {
                Ok(
                    Box::new(crate::kinetics::rrkm::Rrkm::new(s.rrkm, s.qrrho.symmetry)?)
                        as Box<dyn RateTheory>,
                )
            }),
        );
        r
    }

    /// Register (or replace) a theory.
    pub fn register(&mut self, name: &str, factory: RateTheoryFactory) {
        let key = name.to_ascii_lowercase();
        self.entries.retain(|(k, _)| *k != key);
        self.entries.push((key, factory));
    }

    /// The registered names.
    pub fn names(&self) -> Vec<&str> {
        self.entries.iter().map(|(k, _)| k.as_str()).collect()
    }

    /// The theory `name`.
    pub fn create(&self, name: &str, s: &RateSettings) -> MoptResult<Box<dyn RateTheory>> {
        let key = name.to_ascii_lowercase();
        match self.entries.iter().find(|(k, _)| *k == key) {
            Some((_, f)) => f(s),
            None => Err(refused(&key).unwrap_or_else(|| {
                MoptError::parse(format!(
                    "unknown rate theory '{name}' (available: {})",
                    self.names().join(", ")
                ))
            })),
        }
    }
}

impl Default for RateTheoryRegistry {
    fn default() -> Self {
        Self::builtin()
    }
}

/// Corrections by name.
pub struct RateCorrectionRegistry {
    entries: Vec<(String, RateCorrectionFactory)>,
}

impl RateCorrectionRegistry {
    /// `wigner`, `bell`, `eckart`, `skodje_truhlar`, `kappa`, `landau_zener`,
    /// `zhu_nakamura`, `weak_coupling`, `cvt`, `kramers`, `grote_hynes`, `reactive_flux`,
    /// `collins_kimball`, `small_curvature`, `zero_curvature`.
    pub fn builtin() -> Self {
        let mut r = Self {
            entries: Vec::new(),
        };
        r.register(
            "wigner",
            Arc::new(|_: &RateSettings| Ok(Arc::new(Wigner) as Arc<dyn RateCorrection>)),
        );
        r.register(
            "eckart",
            Arc::new(|_: &RateSettings| Ok(Arc::new(Eckart) as Arc<dyn RateCorrection>)),
        );
        r.register(
            "bell",
            Arc::new(|s: &RateSettings| {
                Ok(Arc::new(Bell { settings: s.bell }) as Arc<dyn RateCorrection>)
            }),
        );
        for name in ["cvt", "variational"] {
            r.register(
                name,
                Arc::new(|s: &RateSettings| {
                    Ok(Arc::new(Cvt { qrrho: s.qrrho }) as Arc<dyn RateCorrection>)
                }),
            );
        }
        r.register(
            "skodje_truhlar",
            Arc::new(|_: &RateSettings| Ok(Arc::new(SkodjeTruhlar) as Arc<dyn RateCorrection>)),
        );
        r.register(
            "kappa",
            Arc::new(|s: &RateSettings| {
                Ok(Arc::new(Kappa { value: s.kappa }) as Arc<dyn RateCorrection>)
            }),
        );
        r.register(
            "landau_zener",
            Arc::new(|_: &RateSettings| Ok(Arc::new(LandauZener) as Arc<dyn RateCorrection>)),
        );
        r.register(
            "zhu_nakamura",
            Arc::new(|_: &RateSettings| Ok(Arc::new(ZhuNakamura) as Arc<dyn RateCorrection>)),
        );
        r.register(
            "weak_coupling",
            Arc::new(|_: &RateSettings| {
                Ok(Arc::new(WeakCouplingCorrection) as Arc<dyn RateCorrection>)
            }),
        );
        r.register(
            "kramers",
            Arc::new(|s: &RateSettings| {
                Ok(Arc::new(Kramers {
                    friction: s.friction,
                }) as Arc<dyn RateCorrection>)
            }),
        );
        r.register(
            "grote_hynes",
            Arc::new(|s: &RateSettings| {
                Ok(Arc::new(GroteHynes {
                    friction: s.friction,
                }) as Arc<dyn RateCorrection>)
            }),
        );
        r.register(
            "reactive_flux",
            Arc::new(|_: &RateSettings| Ok(Arc::new(ReactiveFlux) as Arc<dyn RateCorrection>)),
        );
        r.register(
            "collins_kimball",
            Arc::new(|_: &RateSettings| Ok(Arc::new(CollinsKimball) as Arc<dyn RateCorrection>)),
        );
        for (name, small) in [("small_curvature", true), ("zero_curvature", false)] {
            r.register(
                name,
                Arc::new(move |_: &RateSettings| {
                    Ok(Arc::new(CurvatureTunnelling {
                        small_curvature: small,
                    }) as Arc<dyn RateCorrection>)
                }),
            );
        }
        r
    }

    /// Register (or replace) a correction.
    pub fn register(&mut self, name: &str, factory: RateCorrectionFactory) {
        let key = name.to_ascii_lowercase();
        self.entries.retain(|(k, _)| *k != key);
        self.entries.push((key, factory));
    }

    /// The registered names.
    pub fn names(&self) -> Vec<&str> {
        self.entries.iter().map(|(k, _)| k.as_str()).collect()
    }

    /// The correction `name`.
    pub fn create(&self, name: &str, s: &RateSettings) -> MoptResult<Arc<dyn RateCorrection>> {
        let key = name.to_ascii_lowercase();
        match self.entries.iter().find(|(k, _)| *k == key) {
            Some((_, f)) => f(s),
            None => Err(refused(&key).unwrap_or_else(|| {
                MoptError::parse(format!(
                    "unknown rate correction '{name}' (available: {})",
                    self.names().join(", ")
                ))
            })),
        }
    }
}

impl Default for RateCorrectionRegistry {
    fn default() -> Self {
        Self::builtin()
    }
}
