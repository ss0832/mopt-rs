//! The settings of `orientsearch`: MultiOptPy's `-nsample`, `-part`, `-dist`, and the
//! `-orient_opt KEY=VALUE` settings of mopt-rs.
//!
//! MultiOptPy: `Entrypoints/orientation_search.py:orientation_search` (the parser of
//!  `-nsample`, `-part`, `-dist`).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use crate::core_infra::arg_parser::parse_atom_indices_one_based;
use crate::domain::{Quantity, Units};
use crate::error::{MoptError, MoptResult};
use crate::workflow::explore::orient::{OrientSettings, Scheme};

/// Which energies of the parts alone are computed for every minimum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Interaction {
    /// None.
    Off,
    /// The parts at their geometries in the minimum (one evaluation per part).
    Vertical,
    /// The parts optimized alone once, from the input (one optimization per part).
    Relaxed,
    /// Both.
    Both,
}

impl Interaction {
    /// Whether the vertical interaction energy is computed.
    pub fn vertical(self) -> bool {
        matches!(self, Self::Vertical | Self::Both)
    }

    /// Whether the binding energy against the relaxed parts is computed.
    pub fn relaxed(self) -> bool {
        matches!(self, Self::Relaxed | Self::Both)
    }
}

/// `-orient_opt` with the defaults of mopt-rs.
#[derive(Debug, Clone, PartialEq)]
pub struct OrientOptions {
    /// The search (scheme, contact range, refinement, tolerances, seed).
    pub search: OrientSettings,
    /// Whether `seed=` was given.
    pub seed_given: bool,
    /// `fixed=`: parts (from 1) that keep their place.
    pub fixed: Vec<usize>,
    /// `interaction=`.
    pub interaction: Interaction,
    /// `part_charges=`.
    pub part_charges: Option<Vec<i32>>,
    /// `part_multiplicities=`.
    pub part_multiplicities: Option<Vec<u32>>,
    /// `criteria=`: the convergence criteria of the samples when neither `-tcc` nor `-lcc` is
    /// given.
    pub criteria: Criteria,
}

/// Convergence criteria of the optimizations of the samples.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Criteria {
    /// `optmain -tcc` (the default of `orientsearch`, measured: see running.md).
    Tight,
    /// `optmain`'s default criteria (MultiOptPy's samples).
    Normal,
    /// `optmain -lcc`.
    Loose,
}

impl Default for OrientOptions {
    fn default() -> Self {
        Self {
            search: OrientSettings::default(),
            seed_given: false,
            fixed: Vec::new(),
            interaction: Interaction::Vertical,
            part_charges: None,
            part_multiplicities: None,
            criteria: Criteria::Tight,
        }
    }
}

/// The keys of `-orient_opt`.
pub const ORIENT_KEYS: &str = "scheme, seed, contact, clash, attempts, refine, priority, angle, \
                               shift, rmsd, energy, mirror, bonds, fixed, interaction, \
                               part_charges, part_multiplicities, criteria";

fn number(key: &str, value: &str) -> MoptResult<f64> {
    let v: f64 = value
        .trim()
        .parse()
        .map_err(|_| MoptError::parse(format!("-orient_opt {key}: '{value}' is not a number")))?;
    if v.is_finite() {
        Ok(v)
    } else {
        Err(MoptError::parse(format!(
            "-orient_opt {key}: '{value}' is not finite"
        )))
    }
}

/// A switch of `-orient_opt` (`core_infra::arg_parser::parse_switch`).
fn switch(key: &str, value: &str) -> MoptResult<bool> {
    crate::core_infra::arg_parser::parse_switch(&format!("-orient_opt {key}"), value)
}

fn integers<T: std::str::FromStr>(key: &str, value: &str) -> MoptResult<Vec<T>> {
    value
        .split(',')
        .map(|t| {
            t.trim().parse::<T>().map_err(|_| {
                MoptError::parse(format!("-orient_opt {key}: '{t}' is not a whole number"))
            })
        })
        .collect()
}

/// `-orient_opt KEY=VALUE ...` on top of the defaults.
pub fn parse_orient_options(tokens: &[String]) -> MoptResult<OrientOptions> {
    let mut o = OrientOptions::default();
    for token in tokens {
        let (key, value) = token.split_once('=').ok_or_else(|| {
            MoptError::parse(format!("-orient_opt expects KEY=VALUE, got '{token}'"))
        })?;
        let key = key.trim().to_ascii_lowercase();
        let s = &mut o.search;
        match key.as_str() {
            "scheme" => s.placement.scheme = Scheme::parse(value)?,
            "seed" => {
                s.seed = value.trim().parse().map_err(|_| {
                    MoptError::parse(format!("-orient_opt seed: '{value}' is not a whole number"))
                })?;
                o.seed_given = true;
            }
            "contact" => {
                let range: Vec<f64> = value
                    .split(',')
                    .map(|t| number("contact", t))
                    .collect::<MoptResult<_>>()?;
                s.placement.contact = match range.as_slice() {
                    [v] => [*v, *v],
                    [lo, hi] if *lo > 0.0 && lo <= hi => [*lo, *hi],
                    _ => {
                        return Err(MoptError::parse(format!(
                            "-orient_opt contact: one ratio or LOW,HIGH with 0 < LOW <= HIGH, got \
                             '{value}'"
                        )))
                    }
                };
            }
            "clash" => s.placement.clash = number("clash", value)?.max(0.0),
            "attempts" => {
                s.placement.attempts = integers::<usize>("attempts", value)?
                    .first()
                    .copied()
                    .unwrap_or(1)
                    .max(1);
            }
            "refine" => s.refine = number("refine", value)?,
            "priority" => s.priority = value.trim().to_string(),
            "angle" => s.perturbation.max_angle = number("angle", value)?.to_radians(),
            "shift" => s.perturbation.max_shift = number("shift", value)? * Units::ANGSTROM_TO_BOHR,
            "rmsd" => {
                s.matcher.rmsd_tolerance =
                    Quantity::new(number("rmsd", value)? * Units::ANGSTROM_TO_BOHR);
            }
            "energy" => s.matcher.energy_tolerance = Quantity::new(number("energy", value)?.abs()),
            "mirror" => s.matcher.mirror_images = switch("mirror", value)?,
            "bonds" => s.matcher.bond_scale = number("bonds", value)?,
            "fixed" => {
                o.fixed = parse_atom_indices_one_based(value)
                    .map_err(|e| MoptError::parse(format!("-orient_opt fixed: {e}")))?;
            }
            "interaction" => {
                o.interaction = match value.trim().to_ascii_lowercase().as_str() {
                    "off" | "none" => Interaction::Off,
                    "vertical" => Interaction::Vertical,
                    "relaxed" => Interaction::Relaxed,
                    "both" => Interaction::Both,
                    other => {
                        return Err(MoptError::parse(format!(
                            "-orient_opt interaction: vertical, relaxed, both, or off, not \
                             '{other}'"
                        )))
                    }
                };
            }
            "criteria" => {
                o.criteria = match value.trim().to_ascii_lowercase().as_str() {
                    "tight" => Criteria::Tight,
                    "normal" => Criteria::Normal,
                    "loose" => Criteria::Loose,
                    other => {
                        return Err(MoptError::parse(format!(
                            "-orient_opt criteria: tight, normal, or loose, not '{other}'"
                        )))
                    }
                };
            }
            "part_charges" => o.part_charges = Some(integers("part_charges", value)?),
            "part_multiplicities" => {
                let m: Vec<u32> = integers("part_multiplicities", value)?;
                if m.contains(&0) {
                    return Err(MoptError::parse(
                        "-orient_opt part_multiplicities: multiplicities are 2S + 1 >= 1",
                    ));
                }
                o.part_multiplicities = Some(m);
            }
            other => {
                return Err(MoptError::parse(format!(
                    "-orient_opt: unknown key '{other}' (known: {ORIENT_KEYS})"
                )))
            }
        }
    }
    if o.search.matcher.bond_scale <= 0.5 {
        return Err(MoptError::parse(
            "-orient_opt bonds: the bond scale must be above 0.5",
        ));
    }
    Ok(o)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn tokens(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn options_are_read_and_checked() {
        let o = parse_orient_options(&tokens(&[
            "scheme=uniform",
            "seed=12",
            "contact=0.9,1.3",
            "refine=0.25",
            "priority=boltzmann",
            "rmsd=0.3",
            "energy=2e-4",
            "mirror=false",
            "fixed=1",
            "interaction=both",
            "part_charges=0,1",
            "part_multiplicities=1,2",
            "angle=45",
            "criteria=normal",
        ]))
        .unwrap();
        assert_eq!(o.criteria, Criteria::Normal);
        assert_eq!(OrientOptions::default().criteria, Criteria::Tight);
        assert_eq!(o.search.placement.scheme, Scheme::Uniform);
        assert!(o.seed_given && o.search.seed == 12);
        assert_eq!(o.search.placement.contact, [0.9, 1.3]);
        assert!((o.search.refine - 0.25).abs() < 1e-15);
        assert!(
            (o.search.matcher.rmsd_tolerance.value() - 0.3 * Units::ANGSTROM_TO_BOHR).abs() < 1e-12
        );
        assert!(!o.search.matcher.mirror_images);
        assert_eq!(o.fixed, vec![1]);
        assert_eq!(o.interaction, Interaction::Both);
        assert_eq!(o.part_charges, Some(vec![0, 1]));
        assert!((o.search.perturbation.max_angle - 45f64.to_radians()).abs() < 1e-12);
        let d = parse_orient_options(&[]).unwrap();
        assert!(!d.seed_given && d.interaction.vertical() && !d.interaction.relaxed());
        for bad in [
            "scheme=other",
            "contact=1.2,0.9",
            "nope=1",
            "seed",
            "mirror=maybe",
            "bonds=0.1",
        ] {
            assert!(parse_orient_options(&tokens(&[bad])).is_err(), "{bad}");
        }
    }
}
