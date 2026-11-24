//! The kinetics of a network file (`run_kinetics`, `run_rcmc`): the network is read (the
//! `reaction_network.json` of `run_mapper` or its database `reaction_network.sqlite`, or
//! MultiOptPy's network file), annotated, turned
//! into rate constants and a generator by the parts of [`crate::kinetics`], solved at the
//! output times, checked, and written to the run directory:
//!
//! | File | Content |
//! |---|---|
//! | `populations.csv` | the population of every state at every output time |
//! | `rates.csv` | every step: its states, the energy of the bottleneck, `ln k` both ways, the factors and their diagnostics |
//! | `rcmc_K_contracted.csv` | MultiOptPy's layout: the superstate matrix of the last time and the populations; the header names the matrix |
//! | `rcmc_superstates.json` | the superstates of every time: members, weights `B`, composition `C`, `1^T M`, the matrix, lifetimes |
//! | `diagnostics.json` | per time: eliminated states, `eps_hat`, `sigma_hat t`, `rho_hat t`, conservation, detailed balance, the comparison of `--validate` |
//! | `populations.svg` | populations against `log10(t)` (several times) |
//! | `sensitivity.csv`, `rate_control.csv` | `--sensitivity`: derivatives of the yields; degrees of rate control of the effective rate constants |
//! | `uncertainty.csv`, `priority.csv`, `species.csv`, `spectrum.csv` | when asked for |
//!
//! MultiOptPy: `Utils/rcmc.py:main` (reads the network file, runs one `pop()`, prints the
//! ranked populations, writes `rcmc_K_contracted.csv`).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use crate::core_infra::json::JsonValue;
use crate::core_infra::log::{python_float, Logger};
use crate::core_infra::rng::Rng;
use crate::core_infra::viz::{PlotFigure, PlotSeries, PlotStyle, Rgb};
use crate::domain::{SpinCharge, Units};
use crate::error::{MoptError, MoptResult};
use crate::io::xyz::parse_xyz_str;
use crate::kinetics::conditions::EnergyKind;
use crate::kinetics::derivative::{derivatives, Derivatives, Parameter};
use crate::kinetics::network::{
    Bottleneck, BottleneckKind, CvtPath, KineticNetwork, KineticState, Partner, Species, StateExtra,
};
use crate::kinetics::observe::{scores, Score};
use crate::kinetics::rate::electron_transfer::{TransferForm, TransferStep};
use crate::kinetics::rate::encounter::{AssociationStep, EncounterPair};
use crate::kinetics::refine::{refinement_tasks, Observable, Uncertainties};
use crate::kinetics::solve::direct::relaxation_rates;
use crate::kinetics::solve::Evolution;
use crate::kinetics::solve::{Expm, KineticSolution};
use crate::kinetics::species::Association;
use crate::kinetics::uncertainty::{sample, PopulationStatistics};
use crate::kinetics::{
    initial_populations, prepare, KineticSolver, KineticsParts, KineticsSettings, Prepared, RateSet,
};
use crate::workflow::diabatic_crossings;
use crate::workflow::explore::identity::{BondGraph, StructureMatcher};
use crate::workflow::kinetics_beyond::{self as beyond, ExtraResults, KineticsExtras};
use crate::workflow::mapper::network::{Level, Network, NodeKind};
use crate::workflow::mapper::network_json::{from_json, NETWORK_FORMAT};
use crate::workflow::mapper::sqlite_store::{read_sqlite, SQLITE_FILE};

/// The network file of a run directory of `run_mapper`.
pub const NETWORK_FILE: &str = "reaction_network.json";

/// What a network file was.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NetworkFormat {
    /// The `reaction_network.json` of mopt-rs.
    Mopt,
    /// MultiOptPy's network file (`nodes`, `edges`).
    MultiOptPy,
}

/// A network as read.
#[derive(Debug, Clone)]
pub struct ReadNetwork {
    /// The kinetic network.
    pub network: KineticNetwork,
    /// Its format.
    pub format: NetworkFormat,
    /// The node the population starts in when none is given (the first input of a mapper
    /// run; node 0 of MultiOptPy).
    pub default_start: Option<usize>,
    /// Jobs started from every state (mopt-rs networks; for `discovery_flux`).
    pub visits: Vec<usize>,
}

/// The kinetic network of the minima of `net` at `level`: every transition-state edge between
/// two different minima is a saddle point, every crossing edge with both ends known a
/// crossing; what is left out is noted.
pub fn kinetic_network(net: &Network, level: Level) -> KineticNetwork {
    let mut out = KineticNetwork {
        source: "mopt-rs reaction network".to_string(),
        ..KineticNetwork::default()
    };
    let mut index = vec![usize::MAX; net.nodes.len()];
    for n in net.minima().filter(|n| n.level == level) {
        index[n.id] = out.states.len();
        out.states.push(KineticState {
            id: n.id,
            label: n.label(),
            energy: n.energy,
            free_energy: n.free_energy,
            frequencies: n.frequencies.clone(),
            system: n.system.structure().cloned(),
            multiplicity: n.state.multiplicity,
            symmetry_number: None,
            fragments: Vec::new(),
            extra: StateExtra::default(),
        });
    }
    // The dissociation channels: the sums of their fragments.
    for n in net
        .of_kind(&NodeKind::Dissociated)
        .filter(|n| n.level == level)
    {
        let fragments: Option<Vec<KineticState>> = n
            .parts
            .iter()
            .map(|p| {
                net.fragments.get(p.fragment).map(|f| KineticState {
                    id: f.id,
                    label: format!("{} (F{})", f.formula, f.id),
                    energy: f.energy,
                    free_energy: None,
                    frequencies: f.frequencies.clone(),
                    system: Some(f.system.clone()),
                    multiplicity: f.multiplicity,
                    symmetry_number: None,
                    fragments: Vec::new(),
                    extra: StateExtra::default(),
                })
            })
            .collect();
        let Some(fragments) = fragments else {
            out.notes.push(format!(
                "{} is left out: a fragment of it is not in the network file",
                n.label()
            ));
            continue;
        };
        index[n.id] = out.states.len();
        out.states.push(KineticState {
            id: n.id,
            label: n.label(),
            energy: n.energy,
            free_energy: None,
            frequencies: None,
            system: n.system.structure().cloned(),
            multiplicity: n.state.multiplicity,
            symmetry_number: None,
            fragments,
            extra: StateExtra::default(),
        });
    }
    // A transition state with an end of its IRC that is not known has no state to send its
    // flux to (docs/source/mapper_design.md, kinetics).
    for e in net.open_ts_edges.iter().filter(|e| e.level == level) {
        if let Some(ts) = net.nodes.get(e.ts) {
            out.notes.push(format!(
                "{} is left out: an end of its IRC is not known (its minimization did not \
                 converge)",
                ts.label()
            ));
        }
    }
    for e in net.ts_edges.iter().filter(|e| e.level == level) {
        let Some(ts) = net.nodes.get(e.ts) else {
            continue;
        };
        let [a, b] = e.ends;
        if a == b {
            out.notes.push(format!(
                "{} (edge {}) joins {} with itself (a degenerate rearrangement): it changes no \
                 population and is left out",
                ts.label(),
                e.id,
                net.nodes[a].label()
            ));
            continue;
        }
        let imaginary = e.imaginary_wavenumber.or_else(|| {
            ts.frequencies
                .as_ref()
                .and_then(|f| f.iter().copied().filter(|w| *w < 0.0).reduce(f64::min))
        });
        let step = |states: [usize; 2], share: Option<f64>| {
            let mut b = Bottleneck::saddle(e.id, &ts.label(), states, ts.energy);
            b.free_energy = ts.free_energy;
            b.frequencies = ts.frequencies.clone();
            b.imaginary_wavenumber = imaginary;
            b.system = ts.system.structure().cloned();
            b.multiplicity = ts.state.multiplicity;
            b.share = share;
            b
        };
        // A transition state at an end (a TS–TS link) or a saddle point of higher order (a side
        // that ran into it): an N-furcation over what lies beyond.
        let is_ts = |id: usize| net.nodes.get(id).is_some_and(|n| n.kind.is_saddle());
        match (is_ts(a), is_ts(b)) {
            (false, false) => {
                let (Some(&ia), Some(&ib)) = (index.get(a), index.get(b)) else {
                    continue;
                };
                if ia == usize::MAX || ib == usize::MAX {
                    out.notes.push(format!(
                        "{} is left out: an end of its IRC is not a minimum of level {}",
                        ts.label(),
                        level.number()
                    ));
                    continue;
                }
                // A confirmed bifurcation of a side (confirm_bifurcations): the flux towards
                // that side divides among its end and the minima across the path.
                let mut split = false;
                for (side, (here, there)) in [(0usize, (ib, ia)), (1, (ia, ib))] {
                    let name = if side == 0 { "backward" } else { "forward" };
                    let Some(b) = e
                        .irc
                        .possible_bifurcations
                        .iter()
                        .find(|b| b.side == name && !b.confirmed_ends.is_empty())
                    else {
                        continue;
                    };
                    let mut products = vec![there];
                    for &p in &b.confirmed_ends {
                        let ip = index.get(p).copied().unwrap_or(usize::MAX);
                        if ip != usize::MAX && ip != here && !products.contains(&ip) {
                            products.push(ip);
                        }
                    }
                    if products.len() < 2 {
                        continue;
                    }
                    split = true;
                    let share = 1.0 / products.len() as f64;
                    out.notes.push(format!(
                        "{}: the {name} side of its IRC bifurcates (confirmed across the path): \
                         its flux goes to {} minima, split equally (assumed)",
                        ts.label(),
                        products.len()
                    ));
                    for ip in products {
                        let states = if side == 0 { [ip, here] } else { [here, ip] };
                        out.bottlenecks.push(step(states, Some(share)));
                    }
                }
                if !split {
                    out.bottlenecks.push(step([ia, ib], None));
                }
            }
            (true, true) => out.notes.push(format!(
                "{} is left out: both sides of its IRC end at transition states (TS–TS links on \
                 both sides are not divided among their products yet)",
                ts.label()
            )),
            (linked_first, _) => {
                // An N-furcation: the rate out of the other end comes from this transition
                // state, and its flux divides among the products the linked transition state
                // leads to (the search stops at every state, so later steps are not lumped).
                let (here, there) = if linked_first { (b, a) } else { (a, b) };
                let ih = index.get(here).copied().unwrap_or(usize::MAX);
                if ih == usize::MAX {
                    out.notes.push(format!(
                        "{} is left out: an end of its IRC is not a minimum of level {}",
                        ts.label(),
                        level.number()
                    ));
                    continue;
                }
                let products: Vec<usize> = net
                    .beyond_link(there, level)
                    .into_iter()
                    .filter(|&p| p != here && index.get(p).is_some_and(|&i| i != usize::MAX))
                    .collect();
                if products.is_empty() {
                    out.notes.push(format!(
                        "{} is left out: the transition state {} its IRC ends at leads to no \
                         state of level {} (other than its start)",
                        ts.label(),
                        net.nodes[there].label(),
                        level.number()
                    ));
                    continue;
                }
                let share = 1.0 / products.len() as f64;
                let names: Vec<String> = products.iter().map(|&p| net.nodes[p].label()).collect();
                out.notes.push(format!(
                    "{}: its IRC runs into the transition state {} (a TS–TS link): its flux goes \
                     to the {} products beyond ({}), split equally (assumed: the network does not \
                     tell the split; only trajectories from {} would)",
                    ts.label(),
                    net.nodes[there].label(),
                    products.len(),
                    names.join(", "),
                    ts.label()
                ));
                for p in products {
                    let ip = index[p];
                    let states = if linked_first { [ip, ih] } else { [ih, ip] };
                    out.bottlenecks.push(step(states, Some(share)));
                }
            }
        }
    }
    for e in net.crossing_edges.iter().filter(|e| e.level == level) {
        let Some(x) = net.nodes.get(e.crossing) else {
            continue;
        };
        let ends = match e.ends {
            [Some(a), Some(b)] if a != b => [index[a], index[b]],
            _ => {
                out.notes.push(format!(
                    "{} is left out: the minima on both of its sides are not known",
                    x.label()
                ));
                continue;
            }
        };
        if ends.contains(&usize::MAX) {
            out.notes.push(format!(
                "{} is left out: a minimum on one of its sides is not of level {}",
                x.label(),
                level.number()
            ));
            continue;
        }
        let mut b = Bottleneck::saddle(e.id, &x.label(), ends, x.energy);
        b.kind = BottleneckKind::Crossing;
        b.free_energy = x.free_energy;
        b.frequencies = x.frequencies.clone();
        b.system = x.system.structure().cloned();
        b.multiplicity = match x.kind {
            NodeKind::Crossing { states, .. } => states[0].multiplicity,
            _ => x.state.multiplicity,
        };
        b.gradient_difference = e.gradient_difference;
        out.bottlenecks.push(b);
    }
    for e in net.dissociation_edges.iter().filter(|e| e.level == level) {
        let (Some(complex), Some(channel)) = (net.nodes.get(e.minimum), net.nodes.get(e.channel))
        else {
            continue;
        };
        let label = format!("{}:{}", complex.label(), channel.label());
        let (Some(&ia), Some(&ib)) = (index.get(e.minimum), index.get(e.channel)) else {
            continue;
        };
        // A dissociation whose scan found a barrier is no capture.
        if let Some(scan) = e.scan.as_ref().filter(|s| s.barrier) {
            out.notes.push(format!(
                "{label} is left out: the scan from the complex towards the channel has a \
                 maximum {:.2} kcal/mol above the channel, so the dissociation is not \
                 barrierless and has no rate without its transition state",
                scan.highest.unwrap_or(f64::NAN) * Units::HARTREE_TO_KCAL_MOL
            ));
            continue;
        }
        if ia == usize::MAX || ib == usize::MAX {
            out.notes.push(format!(
                "{label} is left out: the complex or the channel is not in the model"
            ));
            continue;
        }
        let parts: Vec<_> = channel
            .parts
            .iter()
            .filter_map(|p| net.fragments.get(p.fragment))
            .collect();
        let [a, b] = parts[..] else {
            out.notes.push(format!(
                "{label} is left out: {} fragments (an association of two is the one built)",
                channel.parts.len()
            ));
            continue;
        };
        // The energy shown for the step; its rate comes from the association.
        let mut bottleneck =
            Bottleneck::saddle(e.id, &label, [ia, ib], complex.energy.max(channel.energy));
        bottleneck.kind = BottleneckKind::Barrierless;
        bottleneck.system = channel.system.structure().cloned();
        bottleneck.multiplicity = channel.state.multiplicity;
        bottleneck.association = Some(AssociationStep {
            channel_side: 1,
            pair: EncounterPair::of(&a.system, &b.system),
        });
        out.bottlenecks.push(bottleneck);
    }
    out
}

fn json_f64(m: &BTreeMap<String, JsonValue>, key: &str) -> Option<f64> {
    m.get(key).and_then(JsonValue::as_f64)
}

/// The kinetic network of MultiOptPy's network file (`nodes[].node_id`, `energy_hartree`,
/// `free_energy_hartree`; `edges[].node_id_1`, `node_id_2`, `ts_energy_hartree`,
/// `ts_free_energy_hartree`), as `rcmc.py` reads it: nodes without an energy, self-loops, and
/// edges without the energy of `energy` are left out.
pub fn multioptpy_network(
    doc: &JsonValue,
    energy: EnergyKind,
    source: &str,
) -> MoptResult<KineticNetwork> {
    let mut out = KineticNetwork {
        source: source.to_string(),
        ..KineticNetwork::default()
    };
    let nodes = doc
        .get("nodes")
        .and_then(JsonValue::as_array)
        .ok_or_else(|| MoptError::parse(format!("{source}: no list 'nodes'")))?;
    if nodes.is_empty() {
        return Err(MoptError::parse(format!("{source}: no nodes found")));
    }
    for (k, v) in nodes.iter().enumerate() {
        let m = v
            .as_object()
            .ok_or_else(|| MoptError::parse(format!("{source}: node {k} is not an object")))?;
        let id = m
            .get("node_id")
            .and_then(JsonValue::as_usize)
            .ok_or_else(|| MoptError::parse(format!("{source}: node {k} has no node_id")))?;
        let Some(e) = json_f64(m, "energy_hartree") else {
            out.notes
                .push(format!("EQ{id} is left out: it has no energy_hartree"));
            continue;
        };
        let mut s = KineticState::with_energy(id, &format!("EQ{id}"), e);
        s.free_energy = json_f64(m, "free_energy_hartree");
        out.states.push(s);
    }
    let edges = doc
        .get("edges")
        .and_then(JsonValue::as_array)
        .ok_or_else(|| MoptError::parse(format!("{source}: no list 'edges'")))?;
    for (k, v) in edges.iter().enumerate() {
        let m = v
            .as_object()
            .ok_or_else(|| MoptError::parse(format!("{source}: edge {k} is not an object")))?;
        let id = m.get("edge_id").and_then(JsonValue::as_usize).unwrap_or(k);
        let label = format!("TS{id}");
        let ends = (
            m.get("node_id_1").and_then(JsonValue::as_usize),
            m.get("node_id_2").and_then(JsonValue::as_usize),
        );
        let (Some(a), Some(b)) = ends else {
            return Err(MoptError::parse(format!(
                "{source}: edge {k} has no node_id_1 and node_id_2"
            )));
        };
        let ts_e = json_f64(m, "ts_energy_hartree");
        let ts_g = json_f64(m, "ts_free_energy_hartree");
        let energy_value = match (energy, ts_e, ts_g) {
            (EnergyKind::Electronic, Some(e), _) => e,
            (EnergyKind::Free, Some(e), Some(_)) => e,
            (EnergyKind::Free, None, Some(g)) => g,
            _ => {
                out.notes.push(format!(
                    "{label} is left out: it has no {} energy of its transition state",
                    energy.name()
                ));
                continue;
            }
        };
        if a == b {
            out.notes
                .push(format!("{label} is left out: a self-loop of EQ{a}"));
            continue;
        }
        let (Some(ia), Some(ib)) = (out.index_of(a), out.index_of(b)) else {
            out.notes.push(format!(
                "{label} is left out: EQ{a} or EQ{b} is not a node with an energy"
            ));
            continue;
        };
        let mut bn = Bottleneck::saddle(id, &label, [ia, ib], energy_value);
        bn.free_energy = ts_g;
        out.bottlenecks.push(bn);
    }
    Ok(out)
}

/// Read a network file: the `reaction_network.json` of `run_mapper`, its database
/// `reaction_network.sqlite` (`store=sqlite`, read as far as it goes, with notes of what was
/// left out), or MultiOptPy's network file; for a run directory its `reaction_network.json`,
/// or its database when it has only that.
pub fn read_network(path: &Path, level: Level, energy: EnergyKind) -> MoptResult<ReadNetwork> {
    // A run directory: its network file, or its database when it has only that.
    let file = if path.is_dir() {
        let json = path.join(NETWORK_FILE);
        let sqlite = path.join(SQLITE_FILE);
        if !json.is_file() && sqlite.is_file() {
            sqlite
        } else {
            json
        }
    } else {
        path.to_path_buf()
    };
    let source = file.display().to_string();
    let bytes = std::fs::read(&file).map_err(|e| MoptError::from(e).context(file.display()))?;
    if bytes.starts_with(SQLITE_HEADER) {
        // The database of `store=sqlite`, read as far as it goes; what was left out is noted.
        let (net, _, read_notes) = read_sqlite(&file)?;
        let mut out = mopt_network(&net, level, source);
        out.network.notes.extend(
            read_notes
                .into_iter()
                .map(|n| format!("{}: {n}", file.display())),
        );
        return Ok(out);
    }
    let text = String::from_utf8(bytes)
        .map_err(|_| MoptError::parse(format!("{source}: neither JSON nor an SQLite database")))?;
    let doc = JsonValue::parse(&text).map_err(|e| e.context(file.display()))?;
    if doc.get("format").and_then(JsonValue::as_str) == Some(NETWORK_FORMAT) {
        let (net, _) = from_json(&doc).map_err(|e| e.context(file.display()))?;
        return Ok(mopt_network(&net, level, source));
    }
    if doc.get("nodes").is_some() && doc.get("edges").is_some() {
        let k = multioptpy_network(&doc, energy, &source)?;
        let visits = vec![0; k.states.len()];
        return Ok(ReadNetwork {
            network: k,
            format: NetworkFormat::MultiOptPy,
            default_start: Some(0),
            visits,
        });
    }
    Err(MoptError::parse(format!(
        "{source} is neither a network of mopt-rs (format '{NETWORK_FORMAT}') nor one of \
         MultiOptPy (nodes and edges)"
    )))
}

/// The first bytes of an SQLite database file.
const SQLITE_HEADER: &[u8] = b"SQLite format 3\0";

/// The kinetic network of the mapper network `net` read from `source`.
fn mopt_network(net: &Network, level: Level, source: String) -> ReadNetwork {
    let mut k = kinetic_network(net, level);
    k.source = source;
    let visits = k.states.iter().map(|s| net.visits(s.id)).collect();
    // A network with an entrance channel starts there (the structures of the channel are
    // one node, so the whole population is on it); any other at its first input.
    let default_start = net
        .entrance
        .as_ref()
        .map(|e| e.channel)
        .filter(|id| k.index_of(*id).is_some())
        .or_else(|| {
            net.inputs
                .iter()
                .find_map(|i| i.node)
                .filter(|id| k.index_of(*id).is_some())
        })
        .or_else(|| k.states.first().map(|s| s.id));
    ReadNetwork {
        network: k,
        format: NetworkFormat::Mopt,
        default_start,
        visits,
    }
}

const NODE_KEYS: &[&str] = &[
    "energy_hartree",
    "free_energy_hartree",
    "frequencies_cm1",
    "symmetry_number",
    "multiplicity",
];
const EDGE_KEYS: &[&str] = &[
    "energy_hartree",
    "free_energy_hartree",
    "frequencies_cm1",
    "imaginary_wavenumber_cm1",
    "symmetry_number",
    "multiplicity",
    "kappa",
    "coupling_hartree",
    "coupling_cm1",
    "gradient_difference",
    "slope_product",
    "reduced_mass_amu",
    "diabatic_crossings",
    "diabatic_crossing",
    "partner",
    "partner_free_state",
    "transfer",
    "reorganization_hartree",
    "reorganization_kcal_mol",
    "delta_g0_hartree",
    "transfer_wavenumber_cm1",
    "cvt_profile_file",
    "cvt_temperature",
    "cvt_path",
];
const SPECIES_KEYS: &[&str] = &[
    "energy_hartree",
    "free_energy_hartree",
    "frequencies_cm1",
    "symmetry_number",
    "multiplicity",
    "xyz",
    "association",
];

fn numbers(v: &JsonValue, what: &str) -> MoptResult<Vec<f64>> {
    v.as_array()
        .ok_or_else(|| MoptError::parse(format!("{what} is a list of numbers")))?
        .iter()
        .map(|x| {
            x.as_f64()
                .ok_or_else(|| MoptError::parse(format!("{what} is a list of numbers")))
        })
        .collect()
}

/// Apply the annotations `doc` (`{"nodes": {LABEL or ID: {...}}, "edges": {...},
/// "species": {NAME: {...}}}`) to `net`. A species is added (or updated) by `species`; an edge
/// becomes bimolecular with `"partner": NAME` and `"partner_free_state": LABEL or ID` (the
/// state of the edge where the species is free), an electron transfer with `"transfer":
/// "nonadiabatic"` or `"adiabatic"` and the reorganization energy, and gets a reaction path for
/// `cvt` by `"cvt_profile_file"` or `"cvt_path"`. Files are taken relative to the working
/// directory ([`apply_annotations_in`] for another).
pub fn apply_annotations(net: &mut KineticNetwork, doc: &JsonValue) -> MoptResult<()> {
    apply_annotations_in(net, doc, Path::new("."))
}

/// The profile `free_energy_profile.csv` of `freqmain -path` (arc length, energy, generalized
/// free energy) at `temperature` (K), or at the temperature of `variational_tst.txt` next to it.
pub fn read_cvt_profile(path: &Path, temperature: Option<f64>) -> MoptResult<CvtPath> {
    let text =
        std::fs::read_to_string(path).map_err(|e| MoptError::from(e).context(path.display()))?;
    let mut lines = text.lines().filter(|l| !l.trim().is_empty());
    let header: Vec<&str> = lines
        .next()
        .ok_or_else(|| MoptError::parse(format!("{}: empty", path.display())))?
        .split(',')
        .map(str::trim)
        .collect();
    let col = |name: &str| header.iter().position(|h| *h == name);
    let (Some(s), Some(e), Some(g)) = (
        col("arc_length_sqrt_amu_bohr"),
        col("energy_hartree"),
        col("free_energy_hartree"),
    ) else {
        return Err(MoptError::parse(format!(
            "{}: not a free_energy_profile.csv of freqmain -path (arc_length_sqrt_amu_bohr, \
             energy_hartree, free_energy_hartree)",
            path.display()
        )));
    };
    let mut out = CvtPath {
        source: path.display().to_string(),
        ..CvtPath::default()
    };
    let mut free = Vec::new();
    for line in lines {
        let cells: Vec<&str> = line.split(',').map(str::trim).collect();
        let num = |k: usize| -> MoptResult<f64> {
            cells
                .get(k)
                .and_then(|x| x.parse::<f64>().ok())
                .ok_or_else(|| MoptError::parse(format!("{}: bad line '{line}'", path.display())))
        };
        out.arc_length.push(num(s)?);
        out.energies.push(num(e)?);
        free.push(num(g)?);
    }
    out.free_energies = Some(free);
    out.temperature = match temperature {
        Some(t) => Some(t),
        None => {
            let report = path.with_file_name("variational_tst.txt");
            std::fs::read_to_string(&report).ok().and_then(|r| {
                r.lines().find_map(|l| {
                    l.strip_prefix("Temperature:")
                        .and_then(|x| x.trim().trim_end_matches('K').trim().parse().ok())
                })
            })
        }
    };
    Ok(out)
}

/// [`apply_annotations`] with the files of the annotations taken relative to `base`.
pub fn apply_annotations_in(
    net: &mut KineticNetwork,
    doc: &JsonValue,
    base: &Path,
) -> MoptResult<()> {
    let top = doc
        .as_object()
        .ok_or_else(|| MoptError::parse("the annotations are a JSON object"))?;
    for key in top.keys() {
        if key != "nodes" && key != "edges" && key != "species" {
            return Err(MoptError::parse(format!(
                "annotations: unknown key '{key}' (nodes, edges, species)"
            )));
        }
    }
    let entries = |key: &str| -> MoptResult<Vec<(String, BTreeMap<String, JsonValue>)>> {
        match top.get(key) {
            None | Some(JsonValue::Null) => Ok(Vec::new()),
            Some(JsonValue::Object(m)) => m
                .iter()
                .map(|(k, v)| {
                    v.as_object()
                        .cloned()
                        .map(|o| (k.clone(), o))
                        .ok_or_else(|| {
                            MoptError::parse(format!("annotations: {key}.{k} is not an object"))
                        })
                })
                .collect(),
            Some(_) => Err(MoptError::parse(format!(
                "annotations: '{key}' is an object of labels"
            ))),
        }
    };
    for (name, m) in entries("species")? {
        let k = match net.species_index(&name) {
            Some(k) => k,
            None => {
                net.species.push(Species::new(&name, None, None));
                net.species.len() - 1
            }
        };
        let s = &mut net.species[k];
        for (key, v) in &m {
            let what = format!("annotations: species.{name}.{key}");
            match key.as_str() {
                "energy_hartree" => {
                    s.data.energy = v.as_f64().ok_or_else(|| MoptError::parse(what.clone()))?
                }
                "free_energy_hartree" => s.data.free_energy = v.as_f64(),
                "frequencies_cm1" => s.data.frequencies = Some(numbers(v, &what)?),
                "symmetry_number" => s.data.symmetry_number = v.as_usize().map(|x| x as u32),
                "multiplicity" => {
                    s.data.multiplicity =
                        v.as_usize().ok_or_else(|| MoptError::parse(what.clone()))? as u32
                }
                "xyz" => {
                    let text = v
                        .as_str()
                        .ok_or_else(|| MoptError::parse(format!("{what} is XYZ text")))?;
                    s.data.system = Some(
                        parse_xyz_str(text, SpinCharge::default(), None)
                            .map_err(|e| e.context(&what))?,
                    );
                }
                "association" => {
                    let text = v.as_str().ok_or_else(|| {
                        MoptError::parse(format!("{what} is a treatment such as factor:2"))
                    })?;
                    s.association = Some(Association::parse(text)?);
                }
                other => {
                    return Err(MoptError::parse(format!(
                        "annotations: unknown key '{other}' of a species ({})",
                        SPECIES_KEYS.join(", ")
                    )))
                }
            }
        }
    }
    for (name, m) in entries("nodes")? {
        let Some(k) = net
            .states
            .iter()
            .position(|s| s.label == name || s.id.to_string() == name)
        else {
            return Err(MoptError::config(format!(
                "annotations: no state '{name}' in the network"
            )));
        };
        let s = &mut net.states[k];
        for (key, v) in &m {
            let what = format!("annotations: nodes.{name}.{key}");
            match key.as_str() {
                "energy_hartree" => {
                    s.energy = v.as_f64().ok_or_else(|| MoptError::parse(what.clone()))?
                }
                "free_energy_hartree" => s.free_energy = v.as_f64(),
                "frequencies_cm1" => s.frequencies = Some(numbers(v, &what)?),
                "symmetry_number" => s.symmetry_number = v.as_usize().map(|x| x as u32),
                "multiplicity" => {
                    s.multiplicity =
                        v.as_usize().ok_or_else(|| MoptError::parse(what.clone()))? as u32
                }
                other => {
                    if !beyond::apply_node_key(s, other, v, &what)? {
                        return Err(MoptError::parse(format!(
                            "annotations: unknown key '{other}' of a node ({}, {})",
                            NODE_KEYS.join(", "),
                            beyond::NODE_KEYS.join(", ")
                        )));
                    }
                }
            }
        }
    }
    for (name, m) in entries("edges")? {
        let Some(k) = net
            .bottlenecks
            .iter()
            .position(|b| b.label == name || b.id.to_string() == name)
        else {
            return Err(MoptError::config(format!(
                "annotations: no bottleneck '{name}' in the network"
            )));
        };
        let mut partner: Option<String> = None;
        let mut free_state: Option<String> = None;
        let mut transfer: Option<TransferForm> = None;
        let mut reorganization: Option<f64> = None;
        let mut delta_g0: Option<f64> = None;
        let mut transfer_wavenumber: Option<f64> = None;
        let mut cvt_file: Option<PathBuf> = None;
        let mut cvt_temperature: Option<f64> = None;
        let mut cvt_inline: Option<CvtPath> = None;
        let mut crossings_file: Option<PathBuf> = None;
        let mut crossing_number: Option<usize> = None;
        let mut outcomes: Option<JsonValue> = None;
        let mut outcome_errors: Option<JsonValue> = None;
        let b = &mut net.bottlenecks[k];
        for (key, v) in &m {
            let what = format!("annotations: edges.{name}.{key}");
            let num = || v.as_f64().ok_or_else(|| MoptError::parse(what.clone()));
            let label = || {
                v.as_str()
                    .map(str::to_string)
                    .or_else(|| v.as_usize().map(|x| x.to_string()))
                    .ok_or_else(|| MoptError::parse(format!("{what} is a name")))
            };
            match key.as_str() {
                "energy_hartree" => b.energy = num()?,
                "free_energy_hartree" => b.free_energy = v.as_f64(),
                "frequencies_cm1" => b.frequencies = Some(numbers(v, &what)?),
                "imaginary_wavenumber_cm1" => b.imaginary_wavenumber = Some(-num()?.abs()),
                "symmetry_number" => b.symmetry_number = v.as_usize().map(|x| x as u32),
                "multiplicity" => b.multiplicity = num()? as u32,
                "kappa" => b.kappa = Some(num()?),
                "coupling_hartree" => b.coupling = Some(num()?),
                "coupling_cm1" => {
                    b.coupling = Some(
                        num()? * Units::PLANCK_J_S * Units::SPEED_OF_LIGHT_CM_S
                            / Units::HARTREE_TO_JOULE,
                    )
                }
                "gradient_difference" => b.gradient_difference = Some(num()?),
                "slope_product" => b.slope_product = Some(num()?),
                "reduced_mass_amu" => b.reduced_mass = Some(num()?),
                "diabatic_crossings" => {
                    let p = v
                        .as_str()
                        .ok_or_else(|| MoptError::parse(format!("{what} is a path")))?;
                    crossings_file = Some(base.join(p));
                }
                "diabatic_crossing" => {
                    crossing_number = Some(
                        v.as_usize()
                            .ok_or_else(|| MoptError::parse(format!("{what} is a number")))?,
                    )
                }
                "partner" => partner = Some(label()?),
                "partner_free_state" => free_state = Some(label()?),
                "transfer" => {
                    transfer = Some(TransferForm::parse(v.as_str().ok_or_else(|| {
                        MoptError::parse(format!(
                            "{what} is nonadiabatic, adiabatic, or zhu_nakamura"
                        ))
                    })?)?)
                }
                "reorganization_hartree" => reorganization = Some(num()?),
                "reorganization_kcal_mol" => {
                    reorganization = Some(num()? * Units::KCAL_MOL_TO_HARTREE)
                }
                "delta_g0_hartree" => delta_g0 = Some(num()?),
                "transfer_wavenumber_cm1" => transfer_wavenumber = Some(num()?),
                "cvt_profile_file" => {
                    let p = v
                        .as_str()
                        .ok_or_else(|| MoptError::parse(format!("{what} is a path")))?;
                    cvt_file = Some(base.join(p));
                }
                "cvt_temperature" => cvt_temperature = Some(num()?),
                "cvt_path" => {
                    let m = v
                        .as_object()
                        .ok_or_else(|| MoptError::parse(format!("{what} is an object")))?;
                    let list = |key: &str| -> MoptResult<Option<Vec<f64>>> {
                        m.get(key)
                            .map(|x| numbers(x, &format!("{what}.{key}")))
                            .transpose()
                    };
                    let frequencies = match m.get("frequencies_cm1") {
                        None => None,
                        Some(x) => Some(
                            x.as_array()
                                .ok_or_else(|| {
                                    MoptError::parse(format!(
                                        "{what}.frequencies_cm1 is a list of lists"
                                    ))
                                })?
                                .iter()
                                .map(|row| numbers(row, &format!("{what}.frequencies_cm1")))
                                .collect::<MoptResult<Vec<_>>>()?,
                        ),
                    };
                    cvt_inline = Some(CvtPath {
                        arc_length: list("arc_length")?.unwrap_or_default(),
                        energies: list("energy_hartree")?.unwrap_or_default(),
                        frequencies,
                        free_energies: list("free_energy_hartree")?,
                        temperature: m.get("temperature").and_then(JsonValue::as_f64),
                        source: format!("annotations of {name}"),
                    });
                }
                "outcomes" => outcomes = Some(v.clone()),
                "outcome_errors" => outcome_errors = Some(v.clone()),
                other => {
                    if !beyond::apply_edge_key(b, other, v, &what, base)? {
                        return Err(MoptError::parse(format!(
                            "annotations: unknown key '{other}' of an edge ({}, {})",
                            EDGE_KEYS.join(", "),
                            beyond::EDGE_KEYS.join(", ")
                        )));
                    }
                }
            }
        }
        match (transfer, reorganization) {
            (None, None) => {}
            (Some(form), Some(reorganization)) => {
                net.bottlenecks[k].transfer = Some(TransferStep {
                    reorganization,
                    form,
                    delta_g0,
                    wavenumber: transfer_wavenumber,
                })
            }
            _ => {
                return Err(MoptError::parse(format!(
                    "annotations: edges.{name}: an electron transfer needs 'transfer' (its form) \
                     and the reorganization energy ('reorganization_hartree' or \
                     'reorganization_kcal_mol')"
                )))
            }
        }
        match (crossings_file, crossing_number) {
            (Some(file), number) => {
                // The slopes of the file are along the mass-weighted path (reduced mass 1
                // amu), so they come together; a coupling of the edge itself (a spin-orbit
                // coupling, say) is kept.
                let given = ["gradient_difference", "slope_product", "reduced_mass_amu"];
                if let Some(key) = given.iter().find(|key| m.contains_key(**key)) {
                    return Err(MoptError::parse(format!(
                        "annotations: edges.{name}: {key} and diabatic_crossings together; the \
                         file gives the slopes and the reduced mass along its path"
                    )));
                }
                let r = diabatic_crossings::row(&file, number)?;
                let b = &mut net.bottlenecks[k];
                b.coupling = b.coupling.or(Some(r.coupling));
                b.gradient_difference = Some(r.slope_difference());
                b.slope_product = Some(r.slope_product());
                b.reduced_mass = Some(1.0);
            }
            (None, Some(_)) => {
                return Err(MoptError::parse(format!(
                    "annotations: edges.{name}: diabatic_crossing needs the file \
                     (diabatic_crossings)"
                )))
            }
            (None, None) => {}
        }
        if let Some(file) = cvt_file {
            net.bottlenecks[k].path = Some(read_cvt_profile(&file, cvt_temperature)?);
        } else if let Some(mut inline) = cvt_inline {
            inline.temperature = inline.temperature.or(cvt_temperature);
            net.bottlenecks[k].path = Some(inline);
        }
        match (partner, free_state) {
            (None, None) => {}
            (Some(species), Some(state)) => set_partner(net, k, &species, &state)?,
            _ => {
                return Err(MoptError::parse(format!(
                    "annotations: edges.{name}: a bimolecular edge needs both 'partner' (the \
                     species) and 'partner_free_state' (its state where the species is free)"
                )))
            }
        }
        // The steps of an N-furcation share their saddle point: what belongs to it goes to
        // every one of them.
        let saddle = net.bottlenecks[k].clone();
        for (j, other) in net.bottlenecks.iter_mut().enumerate() {
            if j != k
                && other.share.is_some()
                && other.id == saddle.id
                && other.label == saddle.label
            {
                other.energy = saddle.energy;
                other.free_energy = saddle.free_energy;
                other.frequencies.clone_from(&saddle.frequencies);
                other.imaginary_wavenumber = saddle.imaginary_wavenumber;
                other.symmetry_number = saddle.symmetry_number;
                other.multiplicity = saddle.multiplicity;
                other.kappa = saddle.kappa;
                other.path.clone_from(&saddle.path);
                other.friction.clone_from(&saddle.friction);
                other.reactive_flux = saddle.reactive_flux;
                other.tunnelling_path.clone_from(&saddle.tunnelling_path);
                other.uncertainty = saddle.uncertainty;
            }
        }
        if let Some(o) = &outcomes {
            beyond::apply_outcomes(net, k, o, outcome_errors.as_ref(), &name)?;
        } else if outcome_errors.is_some() {
            return Err(MoptError::parse(format!(
                "annotations: edges.{name}: outcome_errors without outcomes"
            )));
        }
    }
    net.check()
}

/// Make the bottleneck `k` bimolecular: `species` is free in its state `state` (label or id).
pub fn set_partner(
    net: &mut KineticNetwork,
    k: usize,
    species: &str,
    state: &str,
) -> MoptResult<()> {
    let label = net.bottlenecks[k].label.clone();
    let s = net.species_index(species).ok_or_else(|| {
        MoptError::config(format!(
            "{label}: the partner '{species}' is not a species (define it under \"species\" of \
             the annotations or with --species)"
        ))
    })?;
    let ends = net.bottlenecks[k].states;
    let side = (0..2)
        .find(|&d| {
            let st = &net.states[ends[d]];
            st.label == state || st.id.to_string() == state
        })
        .ok_or_else(|| {
            MoptError::config(format!(
                "{label}: '{state}' is not one of its states ({}, {})",
                net.states[ends[0]].label, net.states[ends[1]].label
            ))
        })?;
    net.bottlenecks[k].partner = Some(Partner {
        species: s,
        free_side: side,
    });
    Ok(())
}

/// How the screen output looks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Report {
    /// The report of `run_kinetics`.
    Kinetics,
    /// MultiOptPy's table of `run_rcmc`.
    Rcmc,
}

/// One kinetic analysis of a network file.
#[derive(Debug, Clone)]
pub struct KineticsJob {
    /// The network file or run directory.
    pub network: PathBuf,
    /// The level of a network of mopt-rs.
    pub level: Level,
    /// Further data of states and bottlenecks.
    pub annotations: Option<PathBuf>,
    /// The parts and conditions.
    pub settings: KineticsSettings,
    /// The output times (s), ascending; the last is the horizon of the scores.
    pub times: Vec<f64>,
    /// The initial population as `(node id, share)` (empty: the default start node).
    pub initial: Vec<(usize, f64)>,
    /// `--validate`: compare with the matrix exponential (`Some(None)`: the tolerance
    /// `max(eps_hat, 1e-10)`).
    pub validate: Option<Option<f64>>,
    /// `--sensitivity`: the node ids of the states whose yields are differentiated (empty:
    /// every state).
    pub sensitivity: Option<Vec<usize>>,
    /// `--uncertainty SIGMA` (kcal/mol), the draws, and the seed.
    pub uncertainty: Option<(f64, usize, u64)>,
    /// `--priority`: rank the states for an exploration.
    pub priority: Option<Score>,
    /// Report states with the same bond graph together.
    pub lump_conformers: bool,
    /// The relaxation spectrum.
    pub spectrum: bool,
    /// The run directory (`None`: the screen only).
    pub output: Option<PathBuf>,
    /// The screen output.
    pub report: Report,
    /// The options in effect (`settings.json`).
    pub settings_document: Option<String>,
    /// Annotations from the command line (`--species`, `--partner`), applied after those of
    /// the file.
    pub cli_annotations: Option<JsonValue>,
    /// The analyses beyond the canonical model (`--refine`, `--md-trajectory`, the seed of
    /// `kmc`).
    pub extras: KineticsExtras,
}

/// The comparison with the matrix exponential at one time.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Validation {
    /// The time.
    pub time: f64,
    /// `max_i |p_i - (exp(t K) p0)_i|`.
    pub error: f64,
    /// The tolerance it was compared with.
    pub tolerance: f64,
}

/// What a kinetic analysis gave.
#[derive(Debug, Clone)]
pub struct KineticsRun {
    /// The network.
    pub network: KineticNetwork,
    /// The rate constants and the generator.
    pub prepared: Prepared,
    /// The initial populations (over the states of the model).
    pub initial: Vec<f64>,
    /// The solution.
    pub solution: KineticSolution,
    /// `--validate`.
    pub validation: Vec<Validation>,
    /// `--sensitivity`: the derivatives of the yields.
    pub sensitivity: Option<Derivatives>,
    /// The states (of the model) whose yields `sensitivity.csv` lists.
    pub sensitivity_targets: Vec<usize>,
    /// `--uncertainty`.
    pub uncertainty: Option<PopulationStatistics>,
    /// `--priority`: (state of the model, score), highest first.
    pub ranking: Vec<(usize, f64)>,
    /// `--spectrum`: the relaxation rates (1/s).
    pub spectrum: Vec<f64>,
    /// `--lump-conformers`: the species (states of the model of each).
    pub species: Vec<Vec<usize>>,
    /// The analyses beyond the canonical model.
    pub extra: ExtraResults,
}

/// `x` as Python's `f"{x:.{digits}e}"` (`1.234560e+03`).
pub fn python_exp(x: f64, digits: usize) -> String {
    if !x.is_finite() {
        return python_float(x);
    }
    let text = format!("{x:.digits$e}");
    let (mantissa, exponent) = text.split_once('e').unwrap_or((&text, "0"));
    let (sign, value) = match exponent.strip_prefix('-') {
        Some(v) => ('-', v),
        None => ('+', exponent),
    };
    format!("{mantissa}e{sign}{value:0>2}")
}

fn num(x: f64) -> JsonValue {
    if x.is_finite() {
        JsonValue::Number(x)
    } else {
        JsonValue::Null
    }
}

fn text(s: &str) -> JsonValue {
    JsonValue::String(s.to_string())
}

fn object(entries: Vec<(&str, JsonValue)>) -> JsonValue {
    JsonValue::Object(
        entries
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect(),
    )
}

/// Run `job` with the parts of `parts`, telling `log`.
pub fn run_job(job: &KineticsJob, parts: &KineticsParts, log: &Logger) -> MoptResult<KineticsRun> {
    let read = read_network(&job.network, job.level, job.settings.conditions.energy)?;
    let mut net = read.network;
    if let Some(path) = &job.annotations {
        let content = std::fs::read_to_string(path)
            .map_err(|e| MoptError::from(e).context(path.display()))?;
        let doc = JsonValue::parse(&content).map_err(|e| e.context(path.display()))?;
        let base = path.parent().unwrap_or_else(|| Path::new("."));
        apply_annotations_in(&mut net, &doc, base).map_err(|e| e.context(path.display()))?;
    }
    if let Some(doc) = &job.cli_annotations {
        apply_annotations(&mut net, doc)?;
    }
    if job.times.is_empty() {
        return Err(MoptError::config("run_kinetics: no output time"));
    }
    // The model mass_action: concentrations by the solver rosenbrock.
    let mut settings = job.settings.clone();
    let mut model_notes = Vec::new();
    if settings.model == "auto" {
        auto_model(&net, &mut settings, parts, &mut model_notes)?;
        for n in &model_notes {
            log.info(n);
        }
    }
    if settings.model == "mass_action" {
        match settings.solver.as_str() {
            "rosenbrock" | "contract" | "windows" => {}
            "propagate" => {
                settings.solver = "rosenbrock".to_string();
                model_notes.push(
                    "the model mass_action is integrated by the solver rosenbrock".to_string(),
                );
            }
            other => {
                return Err(MoptError::unsupported(format!(
                    "the model mass_action is non-linear: the solver '{other}' is for the \
                     linear model canonical; use rosenbrock, contract, or windows"
                )))
            }
        }
        if job.validate.is_some()
            || job.sensitivity.is_some()
            || job.uncertainty.is_some()
            || job.priority.is_some()
            || job.spectrum
        {
            return Err(MoptError::unsupported(
                "--validate, --sensitivity, --uncertainty, --priority, and --spectrum are for \
                 the linear model canonical (pseudo-first order); the model mass_action gives \
                 the concentrations and the final yields",
            ));
        }
    }
    let resolved_model = matches!(settings.model.as_str(), "egme" | "two_grain");
    if resolved_model
        && (job.sensitivity.is_some() || job.priority.is_some() || job.extras.refine.is_some())
    {
        return Err(MoptError::unsupported(format!(
            "--sensitivity, --refine, and --priority differentiate or rank the states of the \
             canonical model; the model {} resolves them into grains (its populations, \
             --validate, --uncertainty, and the phenomenological rate constants are reported)",
            settings.model
        )));
    }
    if job.extras.md_trajectory.is_some() && settings.model != "semi_markov" {
        return Err(MoptError::config(
            "--md-trajectory gives the residence times of the model semi_markov (--model \
             semi_markov)",
        ));
    }
    let s = &settings;
    let mut prepared = prepare(&net, s, parts)?;
    let mut extra = ExtraResults::default();
    if let Some(md) = &job.extras.md_trajectory {
        let (chain, report) = beyond::md_semi_markov(&net, &prepared.rates, md, log)?;
        prepared.evolution = Evolution::SemiMarkov(chain);
        extra.msm = Some(report);
    }
    let rates = &prepared.rates;
    let shares = if job.initial.is_empty() {
        let start = read.default_start.ok_or_else(|| {
            MoptError::config("the network has no state to start from; give --start-node")
        })?;
        vec![(start, 1.0)]
    } else {
        job.initial.clone()
    };
    let p0 = initial_populations(&net, rates, &shares)?;
    let mut solver = parts.solvers.create(&s.solver, &s.solver_settings)?;
    let seed = job
        .extras
        .seed
        .unwrap_or_else(|| job.uncertainty.map_or(42, |u| u.2));
    let mut rng = Rng::from_seed(seed);
    let ev = &prepared.evolution;
    // A resolved model is solved on its fine states and reported on the states.
    let fine_p0 = ev.expand(&p0);
    let mut solution = solver.solve(ev, &fine_p0, &job.times, &mut rng)?;
    if let Evolution::Resolved(r) = ev {
        for p in &mut solution.populations {
            *p = r.fold(p);
        }
        if solution.superstates.iter().any(Option::is_some) {
            model_notes.push(format!(
                "the superstates of the {} grains are not written (their phenomenological rate \
                 constants are)",
                r.generator.n
            ));
        }
        solution.superstates = vec![None; solution.times.len()];
    }
    model_notes.append(&mut solution.notes);
    solution.notes = model_notes;
    let n = rates.states.len();
    // --validate: against the matrix exponential.
    let mut validation = Vec::new();
    if let Some(tol) = job.validate {
        let mut reference = Expm.solve(ev, &fine_p0, &job.times, &mut rng)?;
        for p in &mut reference.populations {
            *p = ev.fold(p);
        }
        for (k, &t) in job.times.iter().enumerate() {
            let error = (0..n)
                .map(|i| (solution.populations[k][i] - reference.populations[k][i]).abs())
                .fold(0.0, f64::max);
            // The error of a contraction: the closure (eps_hat) and the initial transient of
            // the eliminated states (exp(-sigma_hat t)).
            let bound = tol.unwrap_or_else(|| {
                let d = &solution.diagnostics[k];
                if d.eliminated > 0 {
                    d.eps_hat
                        .unwrap_or(0.0)
                        .max(d.sigma_t.map_or(0.0, |s| (-s).exp()))
                        .max(1e-10)
                } else {
                    1e-10
                }
            });
            validation.push(Validation {
                time: t,
                error,
                tolerance: bound,
            });
        }
    }
    let t_max = job.times.iter().copied().fold(0.0, f64::max);
    // --sensitivity: derivatives of the yields by forward mode through the solver.
    let mut sensitivity = None;
    let mut sensitivity_targets = Vec::new();
    if let Some(targets) = &job.sensitivity {
        sensitivity_targets = if targets.is_empty() {
            (0..n).collect()
        } else {
            targets
                .iter()
                .map(|id| {
                    net.index_of(*id)
                        .and_then(|k| rates.index_of(k))
                        .ok_or_else(|| {
                            MoptError::config(format!(
                                "--sensitivity: node {id} is not a state of the model"
                            ))
                        })
                })
                .collect::<MoptResult<_>>()?
        };
        let start = std::time::Instant::now();
        let d = derivatives(&net, s, parts, &shares, &job.times)?;
        log.info(&format!(
            "Derivatives: {} parameters in {} forward passes (dual numbers through {}), {:.2} s",
            d.parameters.len(),
            d.passes,
            d.solver,
            start.elapsed().as_secs_f64()
        ));
        sensitivity = Some(d);
    }
    // --uncertainty: draws of the energies.
    let uncertainty = match job.uncertainty {
        Some((sigma, samples, seed)) => Some(sample(
            &net,
            sigma * Units::KCAL_MOL_TO_HARTREE,
            samples,
            seed,
            |perturbed| {
                let p = prepare(perturbed, s, parts)?;
                let start = initial_populations(perturbed, &p.rates, &shares)?;
                let mut solver = parts.solvers.create(&s.solver, &s.solver_settings)?;
                let mut rng = Rng::from_seed(seed);
                let sol = solver.solve(
                    &p.evolution,
                    &p.evolution.expand(&start),
                    &[t_max],
                    &mut rng,
                )?;
                Ok(p.evolution
                    .fold(&sol.populations.into_iter().next().unwrap_or_default()))
            },
        )?),
        None => None,
    };
    // --priority: the ranking of an exploration.
    let mut ranking = Vec::new();
    if let Some(score) = job.priority {
        let visits: Vec<usize> = rates.states.iter().map(|&k| read.visits[k]).collect();
        let v = scores(
            solver.as_mut(),
            ev,
            &p0,
            t_max,
            score,
            Some(&visits),
            &mut rng,
        )?;
        ranking = v.into_iter().enumerate().collect();
        ranking.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    }
    let spectrum = if job.spectrum {
        relaxation_rates(ev)?
    } else {
        Vec::new()
    };
    // --refine: the refinement tasks ranked by sensitivity times uncertainty.
    if let Some(targets) = &job.extras.refine {
        let d = match &sensitivity {
            Some(d) => d.clone(),
            None => derivatives(&net, s, parts, &shares, &job.times)?,
        };
        let last = job.times.len() - 1;
        let states: Vec<usize> = if targets.is_empty() {
            let p = &solution.populations[last];
            vec![(0..n).fold(0, |b, i| if p[i] > p[b] { i } else { b })]
        } else {
            targets
                .iter()
                .map(|id| {
                    net.index_of(*id)
                        .and_then(|k| rates.index_of(k))
                        .ok_or_else(|| {
                            MoptError::config(format!(
                                "--refine: node {id} is not a state of the model"
                            ))
                        })
                })
                .collect::<MoptResult<_>>()?
        };
        let sigma = job
            .uncertainty
            .map(|u| u.0 * Units::KCAL_MOL_TO_HARTREE)
            .or(job.extras.refine_sigma)
            .unwrap_or(Units::KCAL_MOL_TO_HARTREE);
        for i in states {
            let tasks = refinement_tasks(
                &d,
                &net,
                Observable {
                    state: i,
                    time: Some(last),
                },
                &Uncertainties { energy: sigma },
            )?;
            extra.refinement.push((
                format!("{} at {:.3e} s", rates.labels[i], job.times[last]),
                tasks,
            ));
        }
    }
    let species = if job.lump_conformers {
        lump_conformers(&net, &rates.states)?
    } else {
        Vec::new()
    };
    let run = KineticsRun {
        network: net,
        prepared,
        initial: p0,
        solution,
        validation,
        sensitivity,
        sensitivity_targets,
        uncertainty,
        ranking,
        spectrum,
        species,
        extra,
    };
    report(job, &run, log);
    if let Some(dir) = &job.output {
        write_outputs(job, &run, dir, log)?;
    }
    Ok(run)
}

/// `--model auto`: `canonical` when every bimolecular step is pseudo-first order (its partner
/// changes by less than the tolerance, [`crate::kinetics::order`]), `mass_action` (by the
/// solver `contract` unless one is given) otherwise; the judgment goes to the notes.
fn auto_model(
    net: &KineticNetwork,
    settings: &mut KineticsSettings,
    parts: &KineticsParts,
    notes: &mut Vec<String>,
) -> MoptResult<()> {
    use crate::kinetics::order::{judge_orders, DEFAULT_ORDER_TOLERANCE};
    let tol = settings
        .model_settings
        .order_tolerance
        .unwrap_or(DEFAULT_ORDER_TOLERANCE);
    let mass = |settings: &mut KineticsSettings| {
        settings.model = "mass_action".to_string();
        if settings.solver == "propagate" {
            settings.solver = "contract".to_string();
        }
    };
    if net.bottlenecks.iter().all(|b| b.partner.is_none()) {
        settings.model = "canonical".to_string();
        notes.push("auto: every step is unimolecular: the model canonical".to_string());
        return Ok(());
    }
    let Some(total) = settings.model_settings.total_concentration else {
        settings.model = "canonical".to_string();
        notes.push(
            "auto: without --total-concentration the orders cannot be judged; the bimolecular \
             steps are taken as pseudo-first order (the model canonical)"
                .to_string(),
        );
        return Ok(());
    };
    let probe = KineticsSettings {
        model: "canonical".to_string(),
        ..settings.clone()
    };
    match prepare(net, &probe, parts) {
        Ok(p) => {
            let judgment = judge_orders(&p.rates, total, tol);
            for line in judgment.lines(&p.rates) {
                notes.push(format!("auto: {line}"));
            }
            if judgment.linear() {
                settings.model = "canonical".to_string();
                notes.push(format!(
                    "auto: every partner changes by less than {tol:e}: the model canonical \
                     (pseudo-first order)"
                ));
            } else {
                mass(settings);
                notes.push(format!(
                    "auto: a partner changes by more than {tol:e}: the model mass_action (the \
                     solver {})",
                    settings.solver
                ));
            }
        }
        Err(e) => {
            mass(settings);
            notes.push(format!(
                "auto: the steps cannot be pseudo-first order ({e}): the model mass_action (the \
                 solver {})",
                settings.solver
            ));
        }
    }
    Ok(())
}

/// The species of the states of the model: states with the same bond graph (up to a
/// renumbering of the atoms) are one species.
fn lump_conformers(net: &KineticNetwork, states: &[usize]) -> MoptResult<Vec<Vec<usize>>> {
    let scale = StructureMatcher::default().bond_scale;
    let mut graphs: Vec<Option<BondGraph>> = Vec::new();
    for &k in states {
        graphs.push(
            net.states[k]
                .system
                .as_ref()
                .map(|s| BondGraph::of(s, scale)),
        );
    }
    if graphs.iter().any(Option::is_none) {
        return Err(MoptError::config(
            "--lump-conformers needs the structures of the states (networks of mopt-rs)",
        ));
    }
    let mut species: Vec<Vec<usize>> = Vec::new();
    for (i, g) in graphs.iter().enumerate() {
        let found = species.iter().position(
            |members| matches!((&graphs[members[0]], g), (Some(a), Some(b)) if a.same_as(b)),
        );
        match found {
            Some(s) => species[s].push(i),
            None => species.push(vec![i]),
        }
    }
    Ok(species)
}

fn energy_label(job: &KineticsJob) -> String {
    let s = &job.settings;
    match (s.rate_theory.as_str(), s.conditions.energy) {
        ("eyring", EnergyKind::Electronic) => "Electronic energy (SCF)".to_string(),
        ("eyring", EnergyKind::Free) => "Gibbs free energy (G_tot)".to_string(),
        (theory, _) => format!("free energies of the rate theory {theory}"),
    }
}

/// The labels of what the solution holds: the states of the model, and with the model
/// `mass_action` the species after them.
pub fn entity_labels(run: &KineticsRun) -> Vec<String> {
    match &run.prepared.evolution {
        Evolution::MassAction(sys) => sys.labels.clone(),
        Evolution::Linear(_) | Evolution::Resolved(_) | Evolution::SemiMarkov(_) => {
            run.prepared.rates.labels.clone()
        }
    }
}

/// The species and the bimolecular steps of a run, for the screen.
fn species_lines(run: &KineticsRun, s: &KineticsSettings) -> Vec<String> {
    let rates = &run.prepared.rates;
    let sp = &rates.species;
    let net = &run.network;
    let mut out = Vec::new();
    if !sp.used.iter().any(|u| *u) {
        return out;
    }
    let c0 = sp.standard_concentration;
    out.push(format!(
        "Standard state of the bimolecular steps: c° = {c0} mol/L; the free energies are \
         converted by {:+.4} kcal/mol per molecule ({})",
        sp.shift * Units::HARTREE_TO_KCAL_MOL,
        match (s.conditions.concentration, sp.shift == 0.0) {
            (Some(c), _) => format!("already at {c} mol/L by -conc"),
            (None, true) => "electronic energies: no standard state".to_string(),
            (None, false) => format!(
                "RT ln(c° R T / p) from the ideal gas at {} Pa",
                python_float(s.conditions.pressure)
            ),
        }
    ));
    for (k, name) in sp.names.iter().enumerate() {
        if !sp.used[k] {
            continue;
        }
        out.push(format!(
            "Species {name}: {}, G° = {:.8} Hartree at c°, association {}",
            match (sp.at_standard, sp.concentrations[k]) {
                (true, Some(c)) => format!("initial {c:e} mol/L"),
                (true, None) => "initial 0 mol/L".to_string(),
                (false, Some(c)) => format!("held at {c:e} mol/L (pseudo-first order)"),
                (false, None) => "no concentration".to_string(),
            },
            sp.energies[k],
            sp.treatments[k].name()
        ));
    }
    for st in &rates.steps {
        if let Some((p, free)) = st.partner {
            let b = &net.bottlenecks[st.bottleneck];
            let k2 = st.ln_k[free].exp() / sp.concentration(p);
            out.push(format!(
                "Bimolecular {}: {} + {} -> {}, k2 = {:.4e} L/mol/s, k(back) = {:.4e} 1/s",
                b.label,
                rates.labels[st.states[free]],
                sp.names[p],
                rates.labels[st.states[1 - free]],
                k2,
                st.ln_k[1 - free].exp()
            ));
        }
    }
    out
}

/// The screen output (and the log).
fn report(job: &KineticsJob, run: &KineticsRun, log: &Logger) {
    let rates = &run.prepared.rates;
    let net = &run.network;
    let s = &job.settings;
    let t_max = job.times.iter().copied().fold(0.0, f64::max);
    let line = "\u{2550}".repeat(60);
    let start = job
        .initial
        .first()
        .map(|(id, _)| *id)
        .or_else(|| {
            run.initial
                .iter()
                .position(|p| *p > 0.0)
                .map(|i| net.states[rates.states[i]].id)
        })
        .unwrap_or(0);
    let title = match job.report {
        Report::Rcmc => "RCMC standalone analysis",
        Report::Kinetics => "Kinetics of a reaction network",
    };
    log.info(&format!(
        "\n{line}\n  {title}\n  JSON \
         : {}\n  Nodes     : {}  |  Edges: {}\n  T         : {} K\n  \
         Time scale: {} s\n  Start node: EQ{start}\n  Energy    : {}\n{line}",
        job.network.display(),
        net.states.len(),
        net.bottlenecks.len(),
        python_float(s.conditions.temperature),
        python_exp(t_max, 3),
        energy_label(job)
    ));
    if job.report == Report::Kinetics {
        log.info(&format!(
            "Model: {} rates ({}{}), clamp {}, solver {}; {} states and {} steps in the model",
            rates.theory,
            rates.description,
            if rates.corrections.is_empty() {
                String::new()
            } else {
                format!("; corrections {}", rates.corrections.join(", "))
            },
            s.conditions.clamp.name(),
            run.solution.solver,
            rates.states.len(),
            rates.steps.len()
        ));
    }
    for note in net.notes.iter().chain(&rates.notes) {
        log.warn(note);
    }
    for note in &run.solution.notes {
        log.info(note);
    }
    for line in species_lines(run, s) {
        log.info(&line);
    }
    let last = run.solution.populations.last().cloned().unwrap_or_default();
    let mut order: Vec<usize> = (0..last.len()).collect();
    order.sort_by(|a, b| last[*b].total_cmp(&last[*a]).then(a.cmp(b)));
    let labels: Vec<String> = entity_labels(run);
    let mass_action = matches!(run.prepared.evolution, Evolution::MassAction(_));
    let width = labels.iter().map(String::len).max().unwrap_or(4) + 2;
    let mut table = format!(
        "\n{:<width$}  {:>18}  {:>6}\n{}  {}  {}\n",
        "Node",
        if mass_action {
            "Conc. (mol/L)"
        } else {
            "Population"
        },
        "Rank",
        "\u{2500}".repeat(width),
        "\u{2500}".repeat(18),
        "\u{2500}".repeat(6)
    );
    for (rank, &i) in order.iter().enumerate() {
        let marker = if rank == 0 {
            "  <- highest priority"
        } else {
            ""
        };
        let _ = writeln!(
            table,
            "{:<width$}  {:>18}  {:>6}{marker}",
            labels[i],
            python_exp(last[i], 8),
            rank + 1
        );
    }
    log.info(&table);
    if let Some(&top) = order.first() {
        let verb = match job.report {
            Report::Rcmc => "RCMC recommends exploring",
            Report::Kinetics => "Most populated at the last time",
        };
        log.info(&format!(
            "{verb}: {}  (population = {})",
            labels[top],
            python_exp(last[top], 6)
        ));
    }
    if job.times.len() > 1 && job.report == Report::Kinetics {
        log.info(&format!(
            "{} at {} times from {:.3e} s to {:.3e} s are in {}.",
            if mass_action {
                "Concentrations"
            } else {
                "Populations"
            },
            job.times.len(),
            job.times[0],
            t_max,
            if mass_action {
                "concentrations.csv (final yields in yields.csv)"
            } else {
                "populations.csv"
            }
        ));
    }
    if let Some(Some(sup)) = run.solution.superstates.last() {
        let mut text = format!("Superstates at {:.3e} s ({}):", sup.time, sup.matrix_name);
        for m in &sup.members {
            let names: Vec<&str> = m.iter().map(|&i| labels[i].as_str()).collect();
            let _ = write!(text, "\n  {} = [{}]", labels[m[0]], names.join(", "));
        }
        log.info(&text);
    }
    if let Some(d) = run.solution.diagnostics.last() {
        log.info(&format!(
            "Diagnostics at {:.3e} s: {} state(s) eliminated{}{}{}, conservation {:.1e}{}",
            d.time,
            d.eliminated,
            d.eps_hat
                .map_or(String::new(), |e| format!(", eps_hat {e:.2e}")),
            d.sigma_t
                .map_or(String::new(), |x| format!(", sigma_hat t {x:.2e}")),
            d.rho_t
                .map_or(String::new(), |x| format!(", rho_hat t {x:.2e}")),
            d.conservation,
            d.detailed_balance.map_or(String::new(), |r| format!(
                ", detailed-balance residual {r:.1e}"
            ))
        ));
    }
    for v in &run.validation {
        let ok = v.error <= v.tolerance;
        let msg = format!(
            "Validation at {:.3e} s: largest difference to the matrix exponential {:.2e} \
             (tolerance {:.1e}): {}",
            v.time,
            v.error,
            v.tolerance,
            if ok { "passed" } else { "FAILED" }
        );
        if ok {
            log.info(&msg);
        } else {
            log.warn(&msg);
        }
    }
    if let Some(d) = &run.sensitivity {
        // The most populated state at the last time: its degrees of rate control.
        if let (Some(p), Some(&t)) = (d.populations.last(), d.times.last()) {
            let i = (0..p.len()).fold(0, |b, k| if p[k] > p[b] { k } else { b });
            let k_last = d.times.len() - 1;
            let mut pairs: Vec<(String, f64)> = d
                .parameters
                .iter()
                .enumerate()
                .filter(|(_, x)| matches!(x.parameter, Parameter::LnK(_)))
                .map(|(m, x)| (x.label.clone(), d.d_populations[m][k_last][i] / p[i]))
                .collect();
            let sum: f64 = pairs.iter().map(|(_, x)| x).sum();
            pairs.sort_by(|a, b| b.1.abs().total_cmp(&a.1.abs()));
            let top: Vec<String> = pairs
                .iter()
                .take(5)
                .map(|(l, x)| format!("{l} {x:+.4}"))
                .collect();
            let rate: f64 = match run.prepared.evolution.generator() {
                Ok(g) => {
                    let km = g.rates().matrix();
                    (0..p.len()).map(|j| km[(i, j)] * p[j]).sum()
                }
                Err(_) => f64::NAN,
            };
            let dt = d
                .index_of(Parameter::Temperature)
                .map_or(f64::NAN, |m| d.d_populations[m][k_last][i]);
            log.info(&format!(
                "Degrees of rate control of p({}) at {:.3e} s, d ln p / d ln k: {} (sum {:.6}; \
                 t (dp/dt) / p = {:.6}); dp/dT = {:.4e} 1/K",
                labels[i],
                t,
                top.join(", "),
                sum,
                t * rate / p[i],
                dt
            ));
        }
    }
    if let Some(u) = &run.uncertainty {
        let best = (0..u.top.len()).fold(0, |b, i| if u.top[i] > u.top[b] { i } else { b });
        log.info(&format!(
            "Uncertainty ({} draws): {} is the most populated in {:.0} % of them; quantiles in \
             uncertainty.csv",
            u.samples,
            labels.get(best).map_or("-", String::as_str),
            100.0 * u.top.get(best).copied().unwrap_or(0.0)
        ));
    }
    if !run.ranking.is_empty() {
        let top: Vec<String> = run
            .ranking
            .iter()
            .take(10)
            .map(|(i, x)| format!("{} {:.3e}", labels[*i], x))
            .collect();
        log.info(&format!(
            "Exploration priority ({}): {}",
            job.priority.map_or("", Score::name),
            top.join(", ")
        ));
    }
    for line in beyond::report_lines(&run.prepared.evolution, &labels, &run.extra) {
        log.info(&line);
    }
    if let Some(dir) = &job.output {
        if job.report == Report::Rcmc {
            log.info(&format!(
                "\nContracted K-matrix + populations saved to: {}",
                dir.join("rcmc_K_contracted.csv").display()
            ));
        } else {
            log.info(&format!("Outputs: {}", dir.display()));
        }
    }
}

/// The species of `rates` and their bimolecular steps for `diagnostics.json`.
fn species_json(rates: &RateSet) -> JsonValue {
    let sp = &rates.species;
    let species: Vec<JsonValue> = (0..sp.names.len())
        .map(|k| {
            object(vec![
                ("name", text(&sp.names[k])),
                ("used", JsonValue::Bool(sp.used[k])),
                (
                    "concentration_mol_L",
                    sp.concentrations[k].map_or(JsonValue::Null, num),
                ),
                ("standard_free_energy_hartree", num(sp.energies[k])),
                ("ln_concentration_over_standard", num(sp.ln_ratio[k])),
                ("association", text(&sp.treatments[k].name())),
            ])
        })
        .collect();
    let steps: Vec<JsonValue> = rates
        .steps
        .iter()
        .filter_map(|st| {
            let (p, free) = st.partner?;
            Some(object(vec![
                ("step", JsonValue::Number(st.bottleneck as f64)),
                ("partner", text(&sp.names[p])),
                ("free_state", text(&rates.labels[st.states[free]])),
                ("bound_state", text(&rates.labels[st.states[1 - free]])),
                ("association", text(&sp.treatments[p].name())),
                (
                    "k2_L_mol-1_s-1",
                    num(st.ln_k[free].exp() / sp.concentration(p)),
                ),
                ("k_back_s-1", num(st.ln_k[1 - free].exp())),
            ]))
        })
        .collect();
    object(vec![
        (
            "standard_concentration_mol_L",
            num(sp.standard_concentration),
        ),
        (
            "standard_state_shift_kcal_mol_per_molecule",
            num(sp.shift * Units::HARTREE_TO_KCAL_MOL),
        ),
        ("at_standard_state", JsonValue::Bool(sp.at_standard)),
        ("species", JsonValue::Array(species)),
        ("bimolecular_steps", JsonValue::Array(steps)),
    ])
}

/// `rcmc_K_contracted.csv` in MultiOptPy's layout for the last output time.
pub fn contracted_csv(job: &KineticsJob, run: &KineticsRun) -> Option<String> {
    contraction_csv(
        &run.solution,
        &run.prepared.rates.labels,
        job.settings.conditions.temperature,
        0,
    )
}

/// A contraction at the last output time, labelled with the mapper queue iteration.
pub(crate) fn contraction_csv(
    solution: &KineticSolution,
    labels: &[String],
    temperature: f64,
    iteration: usize,
) -> Option<String> {
    let sup = solution.superstates.last()?.as_ref()?;
    let n = sup.kept.len();
    let mut out = String::new();
    let names: Vec<&str> = sup.kept.iter().map(|&i| labels[i].as_str()).collect();
    let header = if solution.solver == "legacy_rcmc" {
        "RCMC contracted super-state rate matrix D [s^-1]".to_string()
    } else {
        sup.matrix_name.clone()
    };
    let _ = writeln!(
        out,
        "# {header}  pop_step={iteration}  T={} K  n_superstates={n}",
        python_float(temperature)
    );
    let members: Vec<String> = sup
        .members
        .iter()
        .zip(&names)
        .map(|(m, name)| {
            let list: Vec<&str> = m.iter().map(|&i| labels[i].as_str()).collect();
            format!("{name}=[{}]", list.join("+"))
        })
        .collect();
    let _ = writeln!(out, "# superstate_members: {}", members.join(",  "));
    let _ = writeln!(out, "node,{}", names.join(","));
    for (i, name) in names.iter().enumerate() {
        let row: Vec<String> = (0..n).map(|j| python_exp(sup.matrix[(i, j)], 6)).collect();
        let _ = writeln!(out, "{name},{}", row.join(","));
    }
    out.push('\n');
    let _ = writeln!(
        out,
        "# RCMC transient population  pop_step={iteration}  T={} K  t={} s",
        python_float(temperature),
        python_float(sup.time)
    );
    out.push_str("node,population\n");
    let last = solution.populations.last()?;
    for (i, l) in labels.iter().enumerate() {
        let _ = writeln!(out, "{l},{}", python_exp(last[i], 8));
    }
    Some(out)
}

/// The normalized derivative: `d ln p / d(-E / RT)` for an energy, `d ln p / d ln k` for a
/// step, `d ln p / d ln T` for the temperature (empty for a population of 0).
fn normalized(d: &Derivatives, m: usize, p: f64, dp: f64) -> String {
    if p.is_nan() || p <= 0.0 {
        return String::new();
    }
    let x = match d.parameters[m].parameter {
        Parameter::StateEnergy(_) | Parameter::BottleneckEnergy(_) => -d.kt * dp / p,
        Parameter::LnK(_) | Parameter::LnConcentration(_) => dp / p,
        Parameter::SpeciesEnergy(_) => -d.kt * dp / p,
        Parameter::Temperature => d.parameters[m].value * dp / p,
    };
    format!("{x:.10e}")
}

/// `sensitivity.csv`: every derivative of the yields of `targets` at every output time and at
/// `t = inf`.
pub fn sensitivity_csv(d: &Derivatives, targets: &[usize]) -> String {
    let mut csv = String::from(
        "time_s,state,population,parameter_kind,parameter,parameter_value,derivative,unit,\
         normalized\n",
    );
    let mut rows = |time: &str, p: &[f64], dp: &dyn Fn(usize) -> Vec<f64>| {
        for m in 0..d.parameters.len() {
            let info = &d.parameters[m];
            let col = dp(m);
            for &i in targets {
                let _ = writeln!(
                    csv,
                    "{time},{},{:.10e},{},{},{:.10e},{:.10e},{},{}",
                    d.labels[i],
                    p[i],
                    info.parameter.kind(),
                    info.label,
                    info.value,
                    col[i],
                    info.parameter.unit(),
                    normalized(d, m, p[i], col[i])
                );
            }
        }
    };
    for (k, t) in d.times.iter().enumerate() {
        rows(&format!("{t:.6e}"), &d.populations[k], &|m| {
            d.d_populations[m][k].clone()
        });
    }
    rows("inf", &d.final_populations, &|m| d.d_final[m].clone());
    csv
}

/// `rate_control.csv`: the degrees of rate control `d ln G_ab / d ln k_e` of the effective rate
/// constants between superstates (each group sums to 1).
pub fn rate_control_csv(d: &Derivatives) -> String {
    let mut csv = String::from("time_s,from,to,k_eff_s-1,step,degree_of_rate_control\n");
    for (k, sup) in d.superstates.iter().enumerate() {
        let Some((kept, g)) = sup else {
            continue;
        };
        let nt = kept.len();
        for a in 0..nt {
            for b in 0..nt {
                let v = g[a * nt + b];
                if a == b || v.is_nan() || v <= 0.0 {
                    continue;
                }
                let mut sum = 0.0;
                for (m, info) in d.parameters.iter().enumerate() {
                    if !matches!(info.parameter, Parameter::LnK(_)) {
                        continue;
                    }
                    let x = d.d_superstates[m][k]
                        .as_ref()
                        .map_or(0.0, |dg| dg[a * nt + b] / v);
                    sum += x;
                    let _ = writeln!(
                        csv,
                        "{:.6e},{},{},{v:.10e},{},{x:.10e}",
                        d.times[k], d.labels[kept[b]], d.labels[kept[a]], info.label
                    );
                }
                let _ = writeln!(
                    csv,
                    "{:.6e},{},{},{v:.10e},(sum),{sum:.10e}",
                    d.times[k], d.labels[kept[b]], d.labels[kept[a]]
                );
            }
        }
    }
    csv
}

fn write_outputs(job: &KineticsJob, run: &KineticsRun, dir: &Path, log: &Logger) -> MoptResult<()> {
    std::fs::create_dir_all(dir).map_err(|e| MoptError::from(e).context(dir.display()))?;
    let write = |name: &str, content: &str| -> MoptResult<()> {
        let path = dir.join(name);
        std::fs::write(&path, content).map_err(|e| MoptError::from(e).context(path.display()))
    };
    let rates = &run.prepared.rates;
    let net = &run.network;
    let labels = &rates.labels;
    let sol = &run.solution;
    if let Some(doc) = &job.settings_document {
        write("settings.json", &format!("{doc}\n"))?;
    }
    // populations.csv, or concentrations.csv and yields.csv of mass action.
    let entities = entity_labels(run);
    let mut csv = format!("time_s,{}\n", entities.join(","));
    for (t, p) in sol.times.iter().zip(&sol.populations) {
        let row: Vec<String> = p.iter().map(|v| format!("{v:.10e}")).collect();
        let _ = writeln!(csv, "{t:.6e},{}", row.join(","));
    }
    if let Evolution::MassAction(sys) = &run.prepared.evolution {
        write("concentrations.csv", &csv)?;
        let start = sys.initial(&run.initial)?;
        let last = sol.populations.last().cloned().unwrap_or_default();
        let mut y =
            String::from("entity,kind,initial_mol_L,final_mol_L,change_mol_L,yield,final_time_s\n");
        let t_last = sol.times.last().copied().unwrap_or(0.0);
        for (i, name) in entities.iter().enumerate() {
            let (a, z) = (start[i], last.get(i).copied().unwrap_or(f64::NAN));
            // The yield of a state: its share of the states together; of a species: what is
            // left of it (or, formed from nothing, its concentration over that of the states).
            let (kind, share) = if i < sys.n_states {
                ("state", z / sys.total)
            } else if a > 0.0 {
                ("species", z / a)
            } else {
                ("species", z / sys.total)
            };
            let _ = writeln!(
                y,
                "{name},{kind},{a:.10e},{z:.10e},{:.10e},{share:.10e},{t_last:.6e}",
                z - a
            );
        }
        write("yields.csv", &y)?;
    } else {
        write("populations.csv", &csv)?;
    }
    // rates.csv
    let mut csv = String::from(
        "bottleneck,kind,state_a,state_b,energy_hartree,clamped,barrier_a_kcal_mol,\
         barrier_b_kcal_mol,ln_k_ab,ln_k_ba,k_ab_s-1,k_ba_s-1,ln_tst,factors,diagnostics,\
         partner,partner_free_state,association,k2_L_mol-1_s-1,step_rate\n",
    );
    for st in &rates.steps {
        let b = &net.bottlenecks[st.bottleneck];
        let [a, z] = st.states;
        let factors: Vec<String> = st
            .flux
            .factors
            .iter()
            .map(|(n, f)| format!("{n}={f:.6e}"))
            .collect();
        let diag: Vec<String> = st
            .flux
            .diagnostics
            .iter()
            .map(|d| format!("{}={:.6e}", d.name, d.value))
            .collect();
        let sp = &rates.species;
        let (partner, free_state, association, k2) = match st.partner {
            Some((p, free)) => (
                sp.names[p].clone(),
                labels[st.states[free]].clone(),
                sp.treatments[p].name(),
                format!("{:.6e}", st.ln_k[free].exp() / sp.concentration(p)),
            ),
            None => Default::default(),
        };
        let _ = writeln!(
            csv,
            "{},{},{},{},{:.10},{},{:.6},{:.6},{:.10e},{:.10e},{:.6e},{:.6e},{:.10e},{},{},{},{},\
             {},{},{}",
            b.label,
            b.kind.name(),
            labels[a],
            labels[z],
            st.energy,
            st.clamped,
            (st.energy - st.ends[0]) * Units::HARTREE_TO_KCAL_MOL,
            (st.energy - st.ends[1]) * Units::HARTREE_TO_KCAL_MOL,
            st.ln_k[0],
            st.ln_k[1],
            st.ln_k[0].exp(),
            st.ln_k[1].exp(),
            st.flux.ln_tst,
            factors.join(";"),
            diag.join(";"),
            partner,
            free_state,
            association,
            k2,
            st.rate
        );
    }
    write("rates.csv", &csv)?;
    if let Some(text) = contracted_csv(job, run) {
        write("rcmc_K_contracted.csv", &text)?;
    }
    // rcmc_superstates.json
    let mut entries = Vec::new();
    for sup in sol.superstates.iter().flatten() {
        let n = sup.kept.len();
        let lifetimes: Vec<JsonValue> = (0..n)
            .map(|i| {
                let d = -sup.matrix[(i, i)];
                num(if d > 0.0 { 1.0 / d } else { f64::INFINITY })
            })
            .collect();
        let superstates: Vec<JsonValue> = sup
            .members
            .iter()
            .enumerate()
            .map(|(a, m)| {
                let mut fields = vec![
                    ("state", text(&labels[sup.kept[a]])),
                    (
                        "members",
                        JsonValue::Array(m.iter().map(|&i| text(&labels[i])).collect()),
                    ),
                    ("lifetime_s", lifetimes[a].clone()),
                ];
                if let Some(c) = &sup.contraction {
                    let (nt, ns) = (c.kept.len(), c.eliminated.len());
                    let w = c.weights[a];
                    let mut composition = BTreeMap::new();
                    composition.insert(labels[c.kept[a]].clone(), num(1.0 / w));
                    let mut entering = BTreeMap::new();
                    for q in 0..ns {
                        let share = c.c[q * nt + a] / w;
                        if share > 1e-12 {
                            composition.insert(labels[c.eliminated[q]].clone(), num(share));
                        }
                        let b = c.b[a * ns + q];
                        if b > 1e-12 {
                            entering.insert(labels[c.eliminated[q]].clone(), num(b));
                        }
                    }
                    fields.push(("weight_1TM", num(w)));
                    fields.push(("composition", JsonValue::Object(composition)));
                    fields.push(("entered_from", JsonValue::Object(entering)));
                }
                object(fields)
            })
            .collect();
        let matrix: Vec<JsonValue> = (0..n)
            .map(|i| JsonValue::Array((0..n).map(|j| num(sup.matrix[(i, j)])).collect()))
            .collect();
        entries.push(object(vec![
            ("time_s", num(sup.time)),
            ("matrix", text(&sup.matrix_name)),
            ("superstates", JsonValue::Array(superstates)),
            ("generator", JsonValue::Array(matrix)),
        ]));
    }
    write(
        "rcmc_superstates.json",
        &(JsonValue::Array(entries).to_string_pretty() + "\n"),
    )?;
    // diagnostics.json
    let per_time: Vec<JsonValue> = sol
        .diagnostics
        .iter()
        .map(|d| {
            let v = run.validation.iter().find(|v| v.time == d.time);
            object(vec![
                ("time_s", num(d.time)),
                ("eliminated", JsonValue::Number(d.eliminated as f64)),
                ("eps_hat", d.eps_hat.map_or(JsonValue::Null, num)),
                ("sigma_hat_t", d.sigma_t.map_or(JsonValue::Null, num)),
                ("rho_hat_t", d.rho_t.map_or(JsonValue::Null, num)),
                ("conservation", num(d.conservation)),
                (
                    "detailed_balance_residual",
                    d.detailed_balance.map_or(JsonValue::Null, num),
                ),
                ("most_negative_population", num(d.most_negative)),
                (
                    "validation",
                    v.map_or(JsonValue::Null, |v| {
                        object(vec![
                            ("max_difference_to_expm", num(v.error)),
                            ("tolerance", num(v.tolerance)),
                            ("passed", JsonValue::Bool(v.error <= v.tolerance)),
                        ])
                    }),
                ),
                (
                    "notes",
                    JsonValue::Array(d.notes.iter().map(|s| text(s)).collect()),
                ),
            ])
        })
        .collect();
    let generator = run.prepared.evolution.generator();
    let doc = object(vec![
        ("solver", text(&sol.solver)),
        ("rate_theory", text(&rates.theory)),
        ("energies", text(&rates.description)),
        (
            "corrections",
            JsonValue::Array(rates.corrections.iter().map(|c| text(c)).collect()),
        ),
        ("temperature_K", num(job.settings.conditions.temperature)),
        ("clamp", text(job.settings.conditions.clamp.name())),
        ("states", JsonValue::Number(rates.states.len() as f64)),
        ("steps", JsonValue::Number(rates.steps.len() as f64)),
        (
            "model",
            // The model that ran (that of --model auto after its judgment).
            text(match &run.prepared.evolution {
                Evolution::MassAction(_) => "mass_action",
                Evolution::Resolved(r) => r.model.as_str(),
                Evolution::SemiMarkov(_) => "semi_markov",
                Evolution::Linear(_) if job.settings.model == "auto" => "canonical",
                Evolution::Linear(_) => job.settings.model.as_str(),
            }),
        ),
        (
            "potential",
            JsonValue::Bool(rates.species.composition.is_some()),
        ),
        ("species", species_json(rates)),
        (
            "detailed_balance_residual_of_K",
            generator
                .ok()
                .and_then(|g| g.detailed_balance_residual())
                .map_or(JsonValue::Null, num),
        ),
        (
            "left_out",
            JsonValue::Array(
                net.notes
                    .iter()
                    .chain(&rates.notes)
                    .map(|s| text(s))
                    .collect(),
            ),
        ),
        (
            "notes",
            JsonValue::Array(sol.notes.iter().map(|s| text(s)).collect()),
        ),
        ("times", JsonValue::Array(per_time)),
    ]);
    write("diagnostics.json", &(doc.to_string_pretty() + "\n"))?;
    if sol.times.len() > 1 {
        let mass = matches!(run.prepared.evolution, Evolution::MassAction(_));
        let (title, axis, file) = if mass {
            (
                "Concentrations",
                "concentration (mol/L)",
                "concentrations.svg",
            )
        } else {
            ("Populations", "population", "populations.svg")
        };
        let top = sol
            .populations
            .iter()
            .flatten()
            .copied()
            .fold(0.0, f64::max);
        let mut fig = PlotFigure::new(title, "log10(t / s)", axis);
        let shown: Vec<usize> = (0..entities.len())
            .filter(|&i| sol.populations.iter().any(|p| p[i] > 1e-3 * top))
            .collect();
        for (k, &i) in shown.iter().enumerate() {
            let xs: Vec<f64> = sol
                .times
                .iter()
                .map(|t| if *t > 0.0 { t.log10() } else { f64::NAN })
                .collect();
            fig.add_series(PlotSeries {
                label: entities[i].clone(),
                xs,
                ys: sol.populations.iter().map(|p| p[i]).collect(),
                color: Rgb::jet(k as f64 / shown.len().max(2).saturating_sub(1) as f64),
                style: PlotStyle::LinePoints,
            });
        }
        fig.save(dir.join(file))?;
    }
    if let Some(d) = &run.sensitivity {
        write(
            "sensitivity.csv",
            &sensitivity_csv(d, &run.sensitivity_targets),
        )?;
        if d.superstates.iter().any(Option::is_some) {
            write("rate_control.csv", &rate_control_csv(d))?;
        }
    }
    if let Some(u) = &run.uncertainty {
        let mut csv = String::from("state,mean,q05,median,q95,most_populated_share\n");
        for (i, l) in labels.iter().enumerate() {
            let _ = writeln!(
                csv,
                "{l},{:.10e},{:.10e},{:.10e},{:.10e},{:.4}",
                u.mean[i], u.q05[i], u.q50[i], u.q95[i], u.top[i]
            );
        }
        write("uncertainty.csv", &csv)?;
    }
    if !run.ranking.is_empty() {
        let mut csv = String::from("rank,state,score\n");
        for (r, (i, x)) in run.ranking.iter().enumerate() {
            let _ = writeln!(csv, "{},{},{x:.10e}", r + 1, labels[*i]);
        }
        write("priority.csv", &csv)?;
    }
    if !run.species.is_empty() {
        let names: Vec<String> = run
            .species
            .iter()
            .map(|m| {
                m.iter()
                    .map(|&i| labels[i].as_str())
                    .collect::<Vec<_>>()
                    .join("+")
            })
            .collect();
        let mut csv = format!("time_s,{}\n", names.join(","));
        for (t, p) in sol.times.iter().zip(&sol.populations) {
            let row: Vec<String> = run
                .species
                .iter()
                .map(|m| format!("{:.10e}", m.iter().map(|&i| p[i]).sum::<f64>()))
                .collect();
            let _ = writeln!(csv, "{t:.6e},{}", row.join(","));
        }
        write("species.csv", &csv)?;
    }
    if !run.spectrum.is_empty() {
        let mut csv = String::from("mode,relaxation_rate_s-1,relaxation_time_s,ratio_to_next\n");
        let r = &run.spectrum;
        for k in 0..r.len() {
            let next = r.get(k + 1).copied();
            let _ = writeln!(
                csv,
                "{k},{:.10e},{:.6e},{}",
                r[k],
                if r[k] > 0.0 {
                    1.0 / r[k]
                } else {
                    f64::INFINITY
                },
                next.filter(|x| *x > 0.0)
                    .map_or(String::new(), |x| format!("{:.6e}", r[k] / x))
            );
        }
        write("spectrum.csv", &csv)?;
    }
    beyond::write_outputs(&run.prepared.evolution, labels, &run.extra, dir)?;
    log.debug(&format!("outputs written to {}", dir.display()));
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::domain::{positions, Angstrom, Element, MolecularSystem, SpinCharge};
    use crate::workflow::mapper::network::{ElectronicState, IrcPath, Node, Origin, TsEdge};

    fn argon(x: f64) -> MolecularSystem {
        MolecularSystem::new(
            vec![Element::from_atomic_number(18).unwrap(); 3],
            &positions::<Angstrom>(&[[0.0, 0.0, 0.0], [3.8, 0.0, 0.0], [x, 3.3, 0.0]]),
            SpinCharge::default(),
            None,
        )
        .unwrap()
    }

    fn node(kind: NodeKind, x: f64, energy: f64) -> Node {
        let sys = argon(x);
        let state = ElectronicState::of(&sys, 0);
        Node::new(
            kind,
            Level::Exploration,
            sys.into(),
            energy,
            state,
            Origin::default(),
        )
    }

    #[test]
    fn the_network_of_the_mapper_becomes_a_kinetic_network() {
        let mut net = Network::new(vec![Element::from_atomic_number(18).unwrap(); 3]);
        let a = net.add_node(node(NodeKind::Minimum, 1.9, -0.002)).unwrap();
        let b = net.add_node(node(NodeKind::Minimum, 6.0, -0.001)).unwrap();
        let t = net
            .add_node(node(NodeKind::TransitionState, 4.0, 0.0005))
            .unwrap();
        let t2 = net
            .add_node(node(NodeKind::TransitionState, 2.5, 0.001))
            .unwrap();
        for (ts, ends) in [(t, [a, b]), (t2, [b, b])] {
            net.add_ts_edge(TsEdge {
                id: 0,
                level: Level::Exploration,
                ts,
                ends,
                barriers: [0.0, 0.0],
                imaginary_wavenumber: Some(-80.0),
                irc: IrcPath::default(),
                job: None,
            })
            .unwrap();
        }
        let k = kinetic_network(&net, Level::Exploration);
        assert_eq!(k.states.len(), 2);
        assert_eq!(k.bottlenecks.len(), 1);
        assert_eq!(k.bottlenecks[0].states, [0, 1]);
        assert_eq!(k.bottlenecks[0].imaginary_wavenumber, Some(-80.0));
        assert!(k.bottlenecks[0].system.is_some() && k.states[1].label == "EQ1");
        assert_eq!(k.notes.len(), 1, "{:?}", k.notes);
        assert!(k.notes[0].contains("degenerate"));
        k.check().unwrap();
    }

    #[test]
    fn an_irc_that_runs_into_another_transition_state_divides_its_flux() {
        let mut net = Network::new(vec![Element::from_atomic_number(18).unwrap(); 3]);
        let r = net.add_node(node(NodeKind::Minimum, 1.9, -0.002)).unwrap();
        let p1 = net.add_node(node(NodeKind::Minimum, 6.0, -0.003)).unwrap();
        let p2 = net.add_node(node(NodeKind::Minimum, 7.0, -0.003)).unwrap();
        let ts1 = net
            .add_node(node(NodeKind::TransitionState, 4.0, 0.002))
            .unwrap();
        let ts2 = net
            .add_node(node(NodeKind::TransitionState, 6.5, 0.0))
            .unwrap();
        // TS1's IRC runs from the reactant into TS2, the saddle between the two products.
        for (ts, ends) in [(ts2, [p1, p2]), (ts1, [r, ts2])] {
            net.add_ts_edge(TsEdge {
                id: 0,
                level: Level::Exploration,
                ts,
                ends,
                barriers: [0.0, 0.0],
                imaginary_wavenumber: Some(-80.0),
                irc: IrcPath::default(),
                job: None,
            })
            .unwrap();
        }
        let mut k = kinetic_network(&net, Level::Exploration);
        assert_eq!(k.states.len(), 3);
        // TS2 between the products, and TS1 from the reactant to each of them with half of
        // its flux.
        assert_eq!(k.bottlenecks.len(), 3, "{:?}", k.notes);
        let label = net.nodes[ts1].label();
        let halves: Vec<&Bottleneck> = k.bottlenecks.iter().filter(|b| b.label == label).collect();
        assert_eq!(halves.len(), 2);
        let (ir, i1, i2) = (
            k.index_of(r).unwrap(),
            k.index_of(p1).unwrap(),
            k.index_of(p2).unwrap(),
        );
        assert_eq!(halves[0].states, [ir, i1]);
        assert_eq!(halves[1].states, [ir, i2]);
        assert!(halves.iter().all(|b| b.share == Some(0.5)));
        let own = k
            .bottlenecks
            .iter()
            .find(|b| b.label == net.nodes[ts2].label())
            .unwrap();
        assert_eq!((own.states, own.share), ([i1, i2], None));
        assert!(
            k.notes
                .iter()
                .any(|n| n.contains("TS–TS link") && n.contains("split equally")),
            "{:?}",
            k.notes
        );
        k.check().unwrap();
        // What an annotation gives TS1 goes to both of its steps.
        let ann = JsonValue::parse(&format!(
            r#"{{"edges": {{"{label}": {{"frequencies_cm1": [-500.0, 100.0, 200.0]}}}}}}"#
        ))
        .unwrap();
        apply_annotations(&mut k, &ann).unwrap();
        assert!(k
            .bottlenecks
            .iter()
            .filter(|b| b.label == label)
            .all(|b| b.frequencies.as_deref() == Some(&[-500.0, 100.0, 200.0][..])));
    }

    #[test]
    fn annotations_and_multioptpy_files() {
        let doc = JsonValue::parse(
            r#"{"nodes": [{"node_id": 0, "energy_hartree": 0.0, "free_energy_hartree": 0.0},
                          {"node_id": 3, "energy_hartree": 0.01, "free_energy_hartree": null},
                          {"node_id": 4, "energy_hartree": null}],
                "edges": [{"edge_id": 7, "node_id_1": 0, "node_id_2": 3, "ts_energy_hartree": 0.02,
                           "ts_free_energy_hartree": 0.019},
                          {"edge_id": 8, "node_id_1": 3, "node_id_2": 3, "ts_energy_hartree": 0.03},
                          {"edge_id": 9, "node_id_1": 0, "node_id_2": 4, "ts_energy_hartree": 0.03}]}"#,
        )
        .unwrap();
        let mut net = multioptpy_network(&doc, EnergyKind::Electronic, "test").unwrap();
        assert_eq!(net.states.len(), 2);
        assert_eq!(net.bottlenecks.len(), 1);
        assert_eq!(net.bottlenecks[0].label, "TS7");
        assert_eq!(net.notes.len(), 3, "{:?}", net.notes);
        let ann = JsonValue::parse(
            r#"{"nodes": {"EQ3": {"symmetry_number": 2, "frequencies_cm1": [100.0, 200.0]}},
                "edges": {"7": {"kappa": 0.5, "coupling_cm1": 100.0}}}"#,
        )
        .unwrap();
        apply_annotations(&mut net, &ann).unwrap();
        assert_eq!(net.states[1].symmetry_number, Some(2));
        assert_eq!(net.bottlenecks[0].kappa, Some(0.5));
        let h12 = net.bottlenecks[0].coupling.unwrap();
        assert!(
            (h12 / (100.0 / Units::HARTREE_TO_WAVENUMBER) - 1.0).abs() < 1e-6,
            "{h12}"
        );
        for bad in [
            r#"{"nodes": {"EQ9": {"multiplicity": 1}}}"#,
            r#"{"edges": {"TS7": {"nope": 1}}}"#,
            r#"{"other": {}}"#,
        ] {
            assert!(apply_annotations(&mut net, &JsonValue::parse(bad).unwrap()).is_err());
        }
        assert_eq!(python_exp(1234.56, 6), "1.234560e+03");
        assert_eq!(python_exp(-0.0, 6), "-0.000000e+00");
        assert_eq!(python_exp(4.5e-10, 8), "4.50000000e-10");
    }
}
