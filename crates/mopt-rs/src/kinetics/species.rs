//! Species outside the network and the bimolecular steps that take them up or give them off
//! (`kinetics_beyond_tst.md`, stage 2, pseudo-first order): `state + S <-> other state` with
//! the species `S` held at a concentration `[S]`.
//!
//! Every state of the model holds a number of free molecules of each species relative to the
//! other states of its connected component (the step `a + S -> b` has one more free `S` on the
//! side of `a`), found by a walk over the steps; inconsistent steps are refused. A state with
//! `n` free molecules of `S` has the free energy `G + n mu_S`, a saddle point that of its bound
//! side, with the chemical potential
//!
//! ```text
//! mu_S = G°_S(c°) + k_B T ln([S] / c°)
//! ```
//!
//! so that the step from the free side has the pseudo-first-order rate constant
//! `k' = k_2 [S]` with `k_2 = (k_B T / h) exp(-(G_ts - G_a - G°_S) / RT) / c°` (L/mol/s), and
//! detailed balance holds with the same `mu_S` in both directions.
//!
//! The standard state: the free energies of the rate theory are at that of
//! [`crate::vib::thermochemistry`] (the ideal gas at the pressure of the conditions, or a
//! solution at `-conc`); `c°` is `-conc` when it is given and 1 mol/L otherwise, and every
//! free energy at the pressure `p` is converted with `RT ln(c° R T / p)` per molecule (1.894
//! kcal/mol at 298.15 K and 1 atm, `RT ln 24.47`). Only the number of molecules matters, so
//! this changes the bimolecular steps alone. Electronic energies have no standard state and are
//! not converted.
//!
//! Optional treatments of the association entropy in solution ([`Association`]; none is the
//! default):
//!
//! - `wertz:F`: every structure (state, saddle point, species) loses the fraction `F` of its
//!   translational and rotational entropy at `c°`, `G += F T (S_trans + S_rot)`, so that the
//!   loss on association is scaled by `1 - F` (Wertz-type scaling; D. H. Wertz, J. Am. Chem.
//!   Soc. 1980, 102, 5316, which was not checked here: the fraction is the user's);
//! - `free_volume:M:V` or `free_volume:SOLVENT`: the translational entropy of a species in the
//!   free volume of a solvent of molarity `M` (mol/L) and molecular volume `V` (Angstrom^3),
//!   `v_free = 8 ((1/(M N_A))^(1/3) - V^(1/3))^3` per solvent molecule, as GoodVibes computes
//!   it (`get_free_space`, there marked as deprecated and experimental; the source it cites,
//!   Mammen, Shakhnovich and Whitesides, was not checked here), with GoodVibes's table of
//!   solvents; per free molecule `k_B T ln(1 / phi)` with `phi` the free fraction of the volume;
//! - `effective:C`: an effective concentration `C` (mol/L) in place of `c°` in the conversion
//!   of the standard state, per free molecule `k_B T ln(C / c°)`;
//! - `factor:F`: the association constant and the rate constant of association multiplied by
//!   `F`, per free molecule `k_B T ln F` (the same as `effective` with `C = F c°`).
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::collections::VecDeque;

use super::conditions::Conditions;
use super::network::KineticNetwork;
use super::rate::tst::{SymmetryChoice, SYMMETRY_TOLERANCE};
use super::rate::RateTheory;
use crate::domain::{MolecularSystem, Units};
use crate::error::{MoptError, MoptResult};
use crate::vib::{point_group, thermochemistry, NormalModes, ThermoSettings};

/// The solvents of GoodVibes's free-volume correction: name, molarity (mol/L), molecular
/// volume (Angstrom^3), as `get_free_space` tabulates them.
pub const FREE_VOLUME_SOLVENTS: &[(&str, f64, f64)] = &[
    ("h2o", 55.6, 27.944),
    ("water", 55.6, 27.944),
    ("toluene", 9.4, 149.070),
    ("dmf", 12.9, 77.442),
    ("acoh", 17.4, 86.10),
    ("chloroform", 12.5, 97.0),
];

/// The treatment of the association entropy in solution.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum Association {
    /// The conversion of the standard state only (the default).
    #[default]
    Standard,
    /// Every structure loses `fraction` of its translational and rotational entropy.
    Wertz {
        /// The fraction lost (between 0 and 1).
        fraction: f64,
    },
    /// The translational entropy in the free volume of the solvent.
    FreeVolume {
        /// The molarity of the solvent (mol/L).
        molarity: f64,
        /// The volume of a solvent molecule (Angstrom^3).
        volume: f64,
    },
    /// An effective concentration (mol/L) in place of `c°`.
    Effective(f64),
    /// A factor of the association constant.
    Factor(f64),
}

fn number(text: &str, what: &str) -> MoptResult<f64> {
    let x: f64 = text
        .trim()
        .parse()
        .map_err(|_| MoptError::parse(format!("{what}: '{text}' is not a number")))?;
    if !(x.is_finite() && x > 0.0) {
        return Err(MoptError::parse(format!(
            "{what}: '{text}' must be positive"
        )));
    }
    Ok(x)
}

impl Association {
    /// `none` (or `standard`), `wertz:F`, `free_volume:M:V`, `free_volume:SOLVENT`,
    /// `effective:C`, `factor:F`.
    pub fn parse(text: &str) -> MoptResult<Self> {
        let lower = text.trim().to_ascii_lowercase();
        let mut parts = lower.split(':');
        let head = parts.next().unwrap_or("");
        let rest: Vec<&str> = parts.collect();
        let what = format!("the association treatment '{text}'");
        match (head, rest.as_slice()) {
            ("none" | "standard", []) => Ok(Self::Standard),
            ("wertz", [f]) => {
                let fraction = number(f, &what)?;
                if fraction >= 1.0 {
                    return Err(MoptError::parse(format!(
                        "{what}: the fraction lost must be below 1"
                    )));
                }
                Ok(Self::Wertz { fraction })
            }
            ("wertz", []) => Err(MoptError::parse(
                "the association treatment wertz needs the fraction of the translational and \
                 rotational entropy lost in solution, e.g. wertz:0.5",
            )),
            ("free_volume", [name]) => FREE_VOLUME_SOLVENTS
                .iter()
                .find(|(n, _, _)| n == name)
                .map(|&(_, molarity, volume)| Self::FreeVolume { molarity, volume })
                .ok_or_else(|| {
                    MoptError::parse(format!(
                        "{what}: unknown solvent (give free_volume:MOLARITY:VOLUME or one of \
                         h2o, toluene, dmf, acoh, chloroform)"
                    ))
                }),
            ("free_volume", [m, v]) => {
                let a = Self::FreeVolume {
                    molarity: number(m, &what)?,
                    volume: number(v, &what)?,
                };
                a.free_fraction()?;
                Ok(a)
            }
            ("effective", [c]) => Ok(Self::Effective(number(c, &what)?)),
            ("factor", [f]) => Ok(Self::Factor(number(f, &what)?)),
            _ => Err(MoptError::parse(format!(
                "unknown {what} (none, wertz:F, free_volume:M:V, free_volume:SOLVENT, \
                 effective:C, factor:F)"
            ))),
        }
    }

    /// Its name with its parameters, for the outputs.
    pub fn name(&self) -> String {
        match self {
            Self::Standard => "standard".to_string(),
            Self::Wertz { fraction } => format!("wertz:{fraction}"),
            Self::FreeVolume { molarity, volume } => format!("free_volume:{molarity}:{volume}"),
            Self::Effective(c) => format!("effective:{c}"),
            Self::Factor(f) => format!("factor:{f}"),
        }
    }

    /// The free fraction of the volume of the solvent, `phi = v_free M N_A` with
    /// `v_free = 8 ((1/(M N_A))^(1/3) - V^(1/3))^3` (volumes in L).
    pub fn free_fraction(&self) -> MoptResult<f64> {
        let Self::FreeVolume { molarity, volume } = *self else {
            return Ok(1.0);
        };
        // Angstrom^3 per solvent molecule.
        let cell = 1e27 / (molarity * Units::AVOGADRO);
        let gap = cell.cbrt() - volume.cbrt();
        if gap <= 0.0 {
            return Err(MoptError::config(format!(
                "free_volume: a solvent molecule of {volume} A^3 does not fit in its cell of \
                 {cell:.3} A^3 at {molarity} mol/L"
            )));
        }
        Ok(8.0 * gap.powi(3) * molarity * Units::AVOGADRO * 1e-27)
    }

    /// The free energy (Hartree) added per free molecule of a species with this treatment.
    pub fn species_shift(&self, c: &Conditions) -> MoptResult<f64> {
        Ok(match *self {
            Self::Standard | Self::Wertz { .. } => 0.0,
            Self::FreeVolume { .. } => -c.kt() * self.free_fraction()?.ln(),
            Self::Effective(ce) => c.kt() * (ce / c.standard_concentration()).ln(),
            Self::Factor(f) => c.kt() * f.ln(),
        })
    }
}

/// The species settings of a rate assembly.
#[derive(Debug, Clone, PartialEq)]
pub struct SpeciesSettings {
    /// The concentrations (mol/L) of the species for the pseudo-first-order steps.
    pub concentrations: Vec<(String, f64)>,
    /// The treatment of the association entropy (for every species without its own).
    pub association: Association,
    /// Every species at `c°` (the rate constants of the model `mass_action`).
    pub at_standard: bool,
    /// The symmetry numbers of the entropies of `wertz` (as `tst`).
    pub symmetry: SymmetryChoice,
}

impl Default for SpeciesSettings {
    fn default() -> Self {
        Self {
            concentrations: Vec::new(),
            association: Association::Standard,
            at_standard: false,
            symmetry: SymmetryChoice::Auto,
        }
    }
}

/// `S_trans + S_rot` (Hartree/K) of `system` at the standard state `c°` of `c`.
pub fn trans_rot_entropy(
    system: &MolecularSystem,
    symmetry_number: Option<u32>,
    choice: SymmetryChoice,
    c: &Conditions,
) -> MoptResult<f64> {
    let sigma = match (symmetry_number, choice) {
        (Some(s), _) => s,
        (None, SymmetryChoice::Auto) => point_group(system, SYMMETRY_TOLERANCE).symmetry_number,
        (None, SymmetryChoice::None) => 1,
    };
    let settings = ThermoSettings {
        temperature: c.temperature,
        pressure: c.pressure,
        symmetry_number: sigma.max(1),
        concentration: Some(c.standard_concentration()),
        ..ThermoSettings::default()
    };
    let modes = NormalModes {
        wavenumbers: Vec::new(),
        curvatures: Vec::new(),
        displacements: Vec::new(),
        reduced_masses: Vec::new(),
        force_constants: Vec::new(),
        temperatures: Vec::new(),
    };
    let t = thermochemistry(system, &modes, 0.0, &settings)?;
    Ok(t.entropy.translational + t.entropy.rotational)
}

/// The Wertz-type correction `F T (S_trans + S_rot)` (Hartree) of a structure.
pub fn wertz_correction(
    label: &str,
    system: Option<&MolecularSystem>,
    symmetry_number: Option<u32>,
    fraction: f64,
    choice: SymmetryChoice,
    c: &Conditions,
) -> MoptResult<f64> {
    let system = system.ok_or_else(|| {
        MoptError::config(format!(
            "the association treatment wertz needs the structure of every state, saddle point \
             and species of the model; {label} has none"
        ))
    })?;
    Ok(fraction * c.temperature * trans_rot_entropy(system, symmetry_number, choice, c)?)
}

/// The species of a rate assembly: their chemical potentials and how many free molecules of
/// each the states and saddle points of the model hold.
#[derive(Debug, Clone, PartialEq)]
pub struct Reservoir {
    /// `G°` (Hartree) of every species at `c°`, with the conversion of the standard state and
    /// its treatment (NaN for a species no step of the model uses).
    pub energies: Vec<f64>,
    /// `ln([S] / c°)` of every species (0 at the standard state or when unused).
    pub ln_ratio: Vec<f64>,
    /// The concentrations given (mol/L).
    pub concentrations: Vec<Option<f64>>,
    /// Whether a step of the model takes up or gives off the species.
    pub used: Vec<bool>,
    /// The treatment of every species.
    pub treatments: Vec<Association>,
    /// The free molecules of every species held by every state of the model (`None`: a cycle
    /// of steps converts species, and the network has no potential).
    pub composition: Option<Vec<Vec<i32>>>,
    /// The free molecules of every species held by the saddle point of every step.
    pub step_composition: Option<Vec<Vec<i32>>>,
    /// The conversion of the standard state (Hartree per molecule).
    pub shift: f64,
    /// `k_B T` (Hartree).
    pub kt: f64,
}

/// A step of the model as the composition needs it: its states (generator indices) and its
/// partner (species, the index 0 or 1 of the free side within `states`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StepLink {
    /// The states of the step.
    pub states: [usize; 2],
    /// The partner species and the free side.
    pub partner: Option<(usize, usize)>,
}

impl Reservoir {
    /// The chemical potential (Hartree) of species `s`.
    pub fn mu(&self, s: usize) -> f64 {
        self.energies[s] + self.kt * self.ln_ratio[s]
    }

    /// `sum_S n_S mu_S` (Hartree) of the free molecules `n`.
    pub fn offset(&self, n: &[i32]) -> f64 {
        n.iter()
            .enumerate()
            .filter(|(_, &k)| k != 0)
            .map(|(s, &k)| f64::from(k) * self.mu(s))
            .sum()
    }

    /// The species of `net` for the `n_states` states and the steps `links` of a model.
    pub fn of(
        net: &KineticNetwork,
        theory: &dyn RateTheory,
        c: &Conditions,
        settings: &SpeciesSettings,
        n_states: usize,
        links: &[StepLink],
    ) -> MoptResult<Self> {
        let ns = net.species.len();
        let kt = c.kt();
        let shift = theory
            .standard_state(c)
            .map_or(0.0, |from| c.standard_state_shift(from));
        for (name, value) in &settings.concentrations {
            if net.species_index(name).is_none() {
                return Err(MoptError::config(format!(
                    "a concentration is given for '{name}', which is not a species of the \
                     network (species: {})",
                    if ns == 0 {
                        "none".to_string()
                    } else {
                        net.species
                            .iter()
                            .map(|s| s.name.as_str())
                            .collect::<Vec<_>>()
                            .join(", ")
                    }
                )));
            }
            if !(value.is_finite() && *value > 0.0) {
                return Err(MoptError::config(format!(
                    "the concentration of {name} must be positive (got {value} mol/L); a \
                     species that is only released is for the model mass_action"
                )));
            }
        }
        let mut used = vec![false; ns];
        for l in links {
            if let Some((s, _)) = l.partner {
                used[s] = true;
            }
        }
        let (composition, step_composition) = match compositions(n_states, links, ns) {
            Some((n, ts)) => (Some(n), Some(ts)),
            None => (None, None),
        };
        let mut energies = vec![f64::NAN; ns];
        let mut ln_ratio = vec![0.0; ns];
        let mut concentrations = vec![None; ns];
        let mut treatments = vec![settings.association; ns];
        for (s, sp) in net.species.iter().enumerate() {
            concentrations[s] = settings
                .concentrations
                .iter()
                .rev()
                .find(|(n, _)| *n == sp.name)
                .map(|(_, v)| *v);
            if let Some(own) = sp.association {
                if matches!(own, Association::Wertz { .. }) {
                    return Err(MoptError::config(format!(
                        "species {}: wertz applies to every structure of the model; give it as \
                         the treatment of the settings, not of one species",
                        sp.name
                    )));
                }
                treatments[s] = own;
            }
            if !used[s] {
                continue;
            }
            let g = match theory.state_energy(&sp.data, c)? {
                Some(g) if g.is_finite() => g,
                _ => {
                    return Err(MoptError::config(format!(
                        "species {}: it has no {} energy (give it in the annotations)",
                        sp.name,
                        theory.describe(c)
                    )))
                }
            };
            let wertz = match settings.association {
                Association::Wertz { fraction } => wertz_correction(
                    &sp.name,
                    sp.data.system.as_ref(),
                    sp.data.symmetry_number,
                    fraction,
                    settings.symmetry,
                    c,
                )?,
                _ => 0.0,
            };
            energies[s] = g + shift + treatments[s].species_shift(c)? + wertz;
            if !settings.at_standard {
                let value = concentrations[s].ok_or_else(|| {
                    MoptError::config(format!(
                        "a bimolecular step takes up or gives off {}: give its concentration \
                         (--concentration {}:VALUE, mol/L)",
                        sp.name, sp.name
                    ))
                })?;
                ln_ratio[s] = (value / c.standard_concentration()).ln();
            }
        }
        Ok(Self {
            energies,
            ln_ratio,
            concentrations,
            used,
            treatments,
            composition,
            step_composition,
            shift,
            kt,
        })
    }
}

/// The free molecules of every species held by every state and by the saddle point of every
/// step, relative within each connected component, or `None` when a cycle of steps converts
/// species (a state would hold different numbers along two paths).
pub fn compositions(
    n: usize,
    links: &[StepLink],
    ns: usize,
) -> Option<(Vec<Vec<i32>>, Vec<Vec<i32>>)> {
    let mut adjacent: Vec<Vec<(usize, usize, usize)>> = vec![Vec::new(); n];
    for (m, l) in links.iter().enumerate() {
        adjacent[l.states[0]].push((m, 0, l.states[1]));
        adjacent[l.states[1]].push((m, 1, l.states[0]));
    }
    let mut comp: Vec<Option<Vec<i32>>> = vec![None; n];
    for root in 0..n {
        if comp[root].is_some() {
            continue;
        }
        comp[root] = Some(vec![0; ns]);
        let mut queue = VecDeque::from([root]);
        while let Some(a) = queue.pop_front() {
            let here = comp[a].clone().unwrap_or_else(|| vec![0; ns]);
            for &(m, side, b) in &adjacent[a] {
                let mut there = here.clone();
                if let Some((s, free)) = links[m].partner {
                    // The free side holds one more molecule of the species.
                    if free == side {
                        there[s] -= 1;
                    } else {
                        there[s] += 1;
                    }
                }
                match &comp[b] {
                    None => {
                        comp[b] = Some(there);
                        queue.push_back(b);
                    }
                    Some(known) if *known != there => return None,
                    Some(_) => {}
                }
            }
        }
    }
    let composition: Vec<Vec<i32>> = comp
        .into_iter()
        .map(|x| x.unwrap_or_else(|| vec![0; ns]))
        .collect();
    let steps = links
        .iter()
        .map(|l| match l.partner {
            // The saddle point holds the partner bound: the composition of the bound side.
            Some((_, free)) => composition[l.states[1 - free]].clone(),
            None => composition[l.states[0]].clone(),
        })
        .collect();
    Some((composition, steps))
}

/// `RT ln(c° R T / p)` (kcal/mol) at `t` (K), `p` (Pa), and `c°` (mol/L).
pub fn standard_state_kcal_mol(t: f64, p: f64, c_standard: f64) -> f64 {
    Units::BOLTZMANN_HARTREE_PER_K
        * t
        * (c_standard * 1000.0 * Units::GAS_CONSTANT_J_MOL_K * t / p).ln()
        * Units::HARTREE_TO_KCAL_MOL
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_shift_of_the_standard_state_is_rt_ln_24_46_at_298_k() {
        let c = Conditions {
            temperature: 298.15,
            ..Conditions::default()
        };
        let from = c.free_energy_concentration();
        let shift = c.standard_state_shift(from) * Units::HARTREE_TO_KCAL_MOL;
        let rt = Units::BOLTZMANN_HARTREE_PER_K * 298.15 * Units::HARTREE_TO_KCAL_MOL;
        // 1 atm: c° R T / p = 24.465 (24.46 to the four digits usually quoted).
        assert!((shift - rt * 24.46_f64.ln()).abs() < 5e-4 * rt, "{shift}");
        assert!((shift - 1.8943).abs() < 1e-3, "{shift}");
        assert!((standard_state_kcal_mol(298.15, 101_325.0, 1.0) - shift).abs() < 1e-12);
        // 1 bar: RT ln 24.79.
        let bar = Conditions { pressure: 1e5, ..c };
        let s =
            bar.standard_state_shift(bar.free_energy_concentration()) * Units::HARTREE_TO_KCAL_MOL;
        assert!((s - rt * 24.79_f64.ln()).abs() < 1e-3 * rt, "{s}");
        // A solution at -conc: the free energies are there already.
        let solution = Conditions {
            concentration: Some(1.0),
            ..c
        };
        assert_eq!(
            solution.standard_state_shift(solution.free_energy_concentration()),
            0.0
        );
    }

    #[test]
    fn treatments_parse_and_shift() {
        let c = Conditions {
            temperature: 298.15,
            ..Conditions::default()
        };
        assert_eq!(Association::parse("none").unwrap(), Association::Standard);
        assert_eq!(
            Association::parse("wertz:0.5").unwrap(),
            Association::Wertz { fraction: 0.5 }
        );
        assert!(Association::parse("wertz").is_err());
        assert!(Association::parse("wertz:1.2").is_err());
        let f = Association::parse("factor:10").unwrap();
        let e = Association::parse("effective:10").unwrap();
        assert!((f.species_shift(&c).unwrap() - e.species_shift(&c).unwrap()).abs() < 1e-15);
        assert!((f.species_shift(&c).unwrap() - c.kt() * 10f64.ln()).abs() < 1e-15);
        // Water by GoodVibes's numbers: 0.0838 mL of free volume per L.
        let w = Association::parse("free_volume:h2o").unwrap();
        let phi = w.free_fraction().unwrap();
        assert!((phi * 1e6 - 83.8).abs() < 1.0, "{phi}");
        assert!(w.species_shift(&c).unwrap() > 0.0);
        assert!(Association::parse("free_volume:55.6:40").is_err());
        assert!(Association::parse("free_volume:mercury").is_err());
    }

    #[test]
    fn compositions_follow_the_partner_steps() {
        // E + S <-> ES (E free), ES <-> EP, EP <-> EQ + P (EQ free of P).
        let links = [
            StepLink {
                states: [0, 1],
                partner: Some((0, 0)),
            },
            StepLink {
                states: [1, 2],
                partner: None,
            },
            StepLink {
                states: [2, 3],
                partner: Some((1, 1)),
            },
        ];
        let (n, ts) = compositions(4, &links, 2).unwrap();
        assert_eq!(n[0], vec![0, 0]);
        assert_eq!(n[1], vec![-1, 0]);
        assert_eq!(n[2], vec![-1, 0]);
        assert_eq!(n[3], vec![-1, 1]);
        assert_eq!(ts[0], n[1]);
        assert_eq!(ts[2], n[2]);
        // The catalytic cycle E + S -> ES -> EP -> E + P converts S into P: no potential.
        let cycle = [
            links[0],
            links[1],
            StepLink {
                states: [2, 0],
                partner: Some((1, 1)),
            },
        ];
        assert!(compositions(3, &cycle, 2).is_none());
    }
}
