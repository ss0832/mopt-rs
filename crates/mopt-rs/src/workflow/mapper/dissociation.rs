//! What the jobs of the mapper do with the molecules of a structure ([`super::channels`]):
//! whether an end of an IRC is a dissociation channel (its molecules apart), and the
//! fragments of a channel or of a complex, each optimized on its own once and taken from the
//! network afterwards.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use super::channels::{
    formula, fragment_key, fragment_states, groups, molecules, subsystem, FragmentRecord,
};
use super::configuration::Configuration;
use super::method::{FrameSink, JobContext, OptimizeRequest, Stage};
use super::network::Level;
use crate::error::MoptResult;

/// The fragments of a structure: a dissociation channel, or the channel of a complex.
#[derive(Debug, Clone)]
pub struct Dissociation {
    /// Whether the structure itself is the channel (its molecules apart); otherwise it is a
    /// complex of them.
    pub separated: bool,
    /// Every fragment: its atoms in the structure and its record (`id` is set when the
    /// driver adds it to the network).
    pub parts: Vec<(Vec<usize>, FragmentRecord)>,
    /// The sum of the energies of the fragments (Hartree).
    pub energy: f64,
    /// The relaxed scan of a complex towards its channel (`confirm_dissociations`).
    pub scan: Option<super::scan::DissociationScan>,
}

/// A sink that keeps no frames: those of the fragments do not go to the trajectory of the job
/// (they have other atoms).
pub struct Discard;

impl FrameSink for Discard {
    fn push(&mut self, _system: &Configuration, _energy: f64, _label: &str) -> MoptResult<()> {
        Ok(())
    }
}

/// The dissociation of `system`: its groups of molecules in contact when there are two or
/// more (a channel), or its two molecules when they are in contact (a bimolecular complex,
/// unless `only_separated`). `None` otherwise, in a periodic cell, and when the calculations
/// do not recognize channels, and for points of plain variables. Fragments the network knows are taken from it. A fragment whose
/// optimization does not converge leaves that assignment of charges and spins out (with a
/// warning in `warnings`); when every assignment is left out the result is `None`. The caller
/// checks whether the run was stopped.
pub fn dissociation_of(
    ctx: &mut JobContext<'_>,
    point: &Configuration,
    only_separated: bool,
    warnings: &mut Vec<String>,
) -> MoptResult<Option<Dissociation>> {
    let (Some(settings), Some(system)) = (ctx.jobs.channels(), point.structure()) else {
        return Ok(None);
    };
    if !settings.enabled || system.cell.is_some() {
        return Ok(None);
    }
    let molecules = molecules(system, settings.bond_scale);
    if molecules.len() < 2 {
        return Ok(None);
    }
    let found = groups(system, &molecules, settings.margin);
    let apart = found.len() >= 2;
    let parts = if apart {
        found
    } else if !only_separated && molecules.len() == 2 {
        molecules
    } else {
        return Ok(None);
    };
    let mut computed: Vec<FragmentRecord> = Vec::new();
    let mut best: Option<Dissociation> = None;
    'states: for states in fragment_states(system, &parts) {
        let mut records = Vec::with_capacity(parts.len());
        for (atoms, &(charge, multiplicity)) in parts.iter().zip(&states) {
            let fragment = subsystem(system, atoms, charge, multiplicity)?;
            let key = fragment_key(&fragment, settings.bond_scale, settings.mirror_images);
            let same = |f: &FragmentRecord| {
                f.key == key && f.charge == charge && f.multiplicity == multiplicity
            };
            let known = ctx
                .network
                .find_fragment(key, charge, multiplicity, Level::Exploration)
                .map(|id| ctx.network.fragments[id].clone())
                .or_else(|| computed.iter().find(|f| same(f)).cloned());
            let record = match known {
                Some(r) => r,
                None => {
                    let name = formula(system, atoms);
                    let label =
                        format!("fragment {name} (charge {charge}, multiplicity {multiplicity})");
                    let r = ctx.jobs.optimize(
                        &Configuration::Structure(fragment),
                        &OptimizeRequest {
                            label: label.clone(),
                            stage: Stage::Minimum,
                            terms: Vec::new(),
                        },
                        &mut Discard,
                    )?;
                    if r.interrupted {
                        return Ok(None);
                    }
                    if !r.converged {
                        warnings.push(format!(
                            "{label} did not converge to a minimum: the channel with it is left \
                             out"
                        ));
                        continue 'states;
                    }
                    let optimized = r.system.require_structure("a fragment")?.clone();
                    let found =
                        fragment_key(&optimized, settings.bond_scale, settings.mirror_images);
                    if found != key {
                        warnings.push(format!(
                            "{label} changed its bonds when optimized on its own; it is kept as \
                             optimized"
                        ));
                    }
                    let record = FragmentRecord {
                        id: 0,
                        key: found,
                        formula: name,
                        charge,
                        multiplicity,
                        level: Level::Exploration,
                        system: optimized,
                        energy: r.energy,
                        frequencies: r.wavenumbers,
                    };
                    computed.push(record.clone());
                    record
                }
            };
            records.push((atoms.clone(), record));
        }
        let energy = records.iter().map(|(_, r)| r.energy).sum::<f64>();
        if best.as_ref().is_none_or(|b| energy < b.energy) {
            best = Some(Dissociation {
                separated: apart,
                parts: records,
                energy,
                scan: None,
            });
        }
    }
    // The scan that confirms the barrierless dissociation of a complex of two molecules.
    if let (Some(d), Some(scan)) = (best.as_mut(), ctx.jobs.dissociation_scan()) {
        if !d.separated && d.parts.len() == 2 {
            let (a, b) = (d.parts[0].0.clone(), d.parts[1].0.clone());
            match super::scan::dissociation_scan(ctx, point, &a, &b, d.energy, &scan, &settings)? {
                super::scan::ScanEnd::Stopped => return Ok(None),
                super::scan::ScanEnd::Known => {}
                super::scan::ScanEnd::Done(s) => d.scan = Some(s),
            }
        }
    }
    Ok(best)
}
