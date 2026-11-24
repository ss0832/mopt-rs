//! The networks of several runs as one (`run_mapper -merge RUN_DIR ...`), without a calculation.
//!
//! The first network is taken as it is. Every node of a later one is matched with the nodes so
//! far by the identity test (kind, level, state, bond graph, RMSD, energy; a dissociation
//! channel by its fragments) and added when it is new; an edge is added when the merged
//! network has none of its kind with the same bottleneck (the transition state of a TS edge,
//! the crossing point of a crossing edge, the saddle point and mode of a descent, the complex
//! and channel of a dissociation); the kinds of fragments of the channels are merged by their
//! key, charge and multiplicity (the record of the network that had the kind first is kept);
//! the jobs are appended with new numbers, and the inputs of every run are inputs of the merged
//! network. [`MergeMap`] says what every node and path file of the later network became, for
//! the files to copy and the table of the merge.
//!
//! That the runs ran on the same surface is the user's word: nothing in a network says more of
//! its surface than the name of its calculator. The rules are in
//! `docs/source/mapper_design.md` ("Networks of other runs").
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use super::network::{
    InputRecord, JobRecord, Level, Network, NetworkWarning, Node, NodeKind, Origin,
};
use super::observers::irc_file;
use super::peak::PathPeak;
use crate::error::{MoptError, MoptResult};
use crate::workflow::explore::identity::StructureMatcher;

/// What the nodes and files of a network became in the merged one.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MergeMap {
    /// The node of the merged network for every node of the other (by its number there).
    pub nodes: Vec<usize>,
    /// Whether that node was added (otherwise the merged network had it).
    pub added: Vec<bool>,
    /// The path files to copy: the file of the other run and its name in the merged one (both
    /// relative to their run directories).
    pub paths: Vec<(String, String)>,
    /// The path peaks of the other network that were added: their numbers there and in the
    /// merged one.
    pub peaks: Vec<(usize, usize)>,
    /// How many edges of each kind were added (TS, crossing, descent, dissociation).
    pub edges: [usize; 4],
    /// What was left out, in words (an edge whose bottleneck the merged network has with
    /// other ends).
    pub notes: Vec<String>,
}

/// Why `other` cannot be merged into `net`, if it cannot.
fn mismatch(net: &Network, other: &Network) -> Option<String> {
    if net.variables != other.variables {
        return Some("another set of variables (or structures against variables)".to_string());
    }
    if net.elements != other.elements {
        return Some("other atoms, or another order of the atoms".to_string());
    }
    if net.whole_elements != other.whole_elements {
        return Some(
            "another whole system of its cores (ONIOM), or cores against whole structures"
                .to_string(),
        );
    }
    if net.states != other.states {
        return Some("other electronic states (states)".to_string());
    }
    None
}

/// Add what `other` (the network of the run `name`) has beyond `net` to it; see the module
/// text.
pub fn merge_into(
    net: &mut Network,
    other: &Network,
    matcher: &StructureMatcher,
    name: &str,
) -> MoptResult<MergeMap> {
    if let Some(why) = mismatch(net, other) {
        return Err(MoptError::config(format!(
            "the network of {name} cannot be merged: {why}"
        )));
    }
    let mut map = MergeMap::default();
    let variables = net.variables.is_some();
    let job_offset = net.jobs.iter().map(|j| j.id + 1).max().unwrap_or(0);
    let input_offset = net.inputs.len();
    // The kinds of fragments of the channels.
    let mut fragment_of: Vec<usize> = Vec::with_capacity(other.fragments.len());
    for f in &other.fragments {
        let same = net.fragments.iter().position(|g| {
            g.key == f.key
                && g.charge == f.charge
                && g.multiplicity == f.multiplicity
                && g.level == f.level
        });
        fragment_of.push(match same {
            Some(id) => id,
            None => {
                let mut record = f.clone();
                record.id = net.fragments.len();
                net.fragments.push(record);
                net.fragments.len() - 1
            }
        });
    }
    let fragments_of = |node: &Node| -> Vec<usize> {
        let mut ids: Vec<usize> = node
            .parts
            .iter()
            .filter_map(|p| fragment_of.get(p.fragment).copied())
            .collect();
        ids.sort_unstable();
        ids
    };
    // The nodes.
    for node in &other.nodes {
        let same = if node.kind == NodeKind::Dissociated {
            let wanted = fragments_of(node);
            net.nodes
                .iter()
                .find(|n| {
                    n.kind == NodeKind::Dissociated && n.level == node.level && {
                        let mut have: Vec<usize> = n.parts.iter().map(|p| p.fragment).collect();
                        have.sort_unstable();
                        have == wanted
                    }
                })
                .map(|n| n.id)
        } else {
            net.find_same_in(
                &node.kind,
                node.level,
                &node.system,
                node.energy,
                matcher,
                (!net.states.is_empty()).then_some(&node.state),
            )?
        };
        match same {
            Some(id) => {
                map.nodes.push(id);
                map.added.push(false);
            }
            None => {
                let mut new = node.clone();
                new.origin = Origin {
                    input: node.origin.input.map(|i| i + input_offset),
                    job: node.origin.job.map(|j| j + job_offset),
                    // Set below, when every node has its place.
                    from_node: None,
                };
                for part in &mut new.parts {
                    if let Some(&f) = fragment_of.get(part.fragment) {
                        part.fragment = f;
                    }
                }
                map.nodes.push(net.add_node(new)?);
                map.added.push(true);
            }
        }
    }
    let place = |id: usize| -> MoptResult<usize> {
        map.nodes.get(id).copied().ok_or_else(|| {
            MoptError::config(format!(
                "the network of {name} names a node {id} it has not"
            ))
        })
    };
    for (old, node) in other.nodes.iter().enumerate() {
        if map.added[old] {
            if let Some(from) = node.origin.from_node {
                net.nodes[map.nodes[old]].origin.from_node = Some(place(from)?);
            }
        }
    }
    // The inputs.
    for input in &other.inputs {
        net.inputs.push(InputRecord {
            file: format!("{} ({name})", input.file),
            frame: input.frame,
            node: input.node.map(&place).transpose()?,
            goal: false,
        });
    }
    // The TS edges; `ts_edge_of` for the jobs.
    let mut ts_edge_of: Vec<Option<usize>> = Vec::with_capacity(other.ts_edges.len());
    for e in &other.ts_edges {
        let ts = place(e.ts)?;
        let ends = [place(e.ends[0])?, place(e.ends[1])?];
        if let Some(have) = net
            .ts_edges
            .iter()
            .find(|x| x.level == e.level && x.ts == ts)
        {
            let same_ends = have.ends == ends || have.ends == [ends[1], ends[0]];
            if !same_ends {
                map.notes.push(format!(
                    "{} of {name} is {} of the merged network, whose IRC ends at other minima \
                     there; the edge the merged network has is kept",
                    other.node(e.ts)?.label(),
                    net.node(ts)?.label()
                ));
            }
            ts_edge_of.push(Some(have.id));
            continue;
        }
        let mut new = e.clone();
        new.ts = ts;
        new.ends = ends;
        new.job = e.job.map(|j| j + job_offset);
        for b in &mut new.irc.possible_bifurcations {
            b.confirmed_ends = b
                .confirmed_ends
                .iter()
                .map(|&id| place(id))
                .collect::<MoptResult<_>>()?;
        }
        if let Some(old) = &e.irc.file {
            let file = irc_file(net.node(ts)?.number, variables);
            map.paths.push((old.clone(), file.clone()));
            new.irc.file = Some(file);
        }
        ts_edge_of.push(Some(net.add_ts_edge(new)?));
        map.edges[0] += 1;
    }
    // The crossing edges.
    for e in &other.crossing_edges {
        let crossing = place(e.crossing)?;
        if net
            .crossing_edges
            .iter()
            .any(|x| x.level == e.level && x.crossing == crossing)
        {
            continue;
        }
        let mut new = e.clone();
        new.crossing = crossing;
        new.ends = [
            e.ends[0].map(&place).transpose()?,
            e.ends[1].map(&place).transpose()?,
        ];
        new.job = e.job.map(|j| j + job_offset);
        if let Some(old) = &e.irc.file {
            let node = net.node(crossing)?;
            let file = format!(
                "paths/{}{:06}_edge{}_path.xyz",
                node.kind.tag(),
                node.number,
                net.crossing_edges.len()
            );
            map.paths.push((old.clone(), file.clone()));
            new.irc.file = Some(file);
        }
        net.add_crossing_edge(new)?;
        map.edges[1] += 1;
    }
    // The transition states with an end of their IRC that is not known, where the merged
    // network has no edge of the same transition state.
    for e in &other.open_ts_edges {
        let ts = place(e.ts)?;
        let have = |x: usize, level: Level| x == ts && level == e.level;
        if net.ts_edges.iter().any(|x| have(x.ts, x.level))
            || net.open_ts_edges.iter().any(|x| have(x.ts, x.level))
        {
            continue;
        }
        let mut new = e.clone();
        new.ts = ts;
        new.ends = [
            e.ends[0].map(&place).transpose()?,
            e.ends[1].map(&place).transpose()?,
        ];
        new.job = e.job.map(|j| j + job_offset);
        for b in &mut new.irc.possible_bifurcations {
            b.confirmed_ends = b
                .confirmed_ends
                .iter()
                .map(|&id| place(id))
                .collect::<MoptResult<_>>()?;
        }
        if let Some(old) = &e.irc.file {
            let file = irc_file(net.node(ts)?.number, variables);
            map.paths.push((old.clone(), file.clone()));
            new.irc.file = Some(file);
        }
        net.add_open_ts_edge(new)?;
    }
    // A transition state that the other network joins to both its ends is no longer one with
    // an unknown end.
    let joined: Vec<(usize, Level)> = net.ts_edges.iter().map(|x| (x.ts, x.level)).collect();
    net.open_ts_edges
        .retain(|x| !joined.contains(&(x.ts, x.level)));
    for (k, x) in net.open_ts_edges.iter_mut().enumerate() {
        x.id = k;
    }
    // The descents of the saddle points of higher order.
    for e in &other.saddle_edges {
        let saddle = place(e.saddle)?;
        if net
            .saddle_edges
            .iter()
            .any(|x| x.level == e.level && x.saddle == saddle && x.mode == e.mode)
        {
            continue;
        }
        let mut new = e.clone();
        new.saddle = saddle;
        new.ends = [
            e.ends[0].map(&place).transpose()?,
            e.ends[1].map(&place).transpose()?,
        ];
        new.job = e.job.map(|j| j + job_offset);
        net.add_saddle_edge(new)?;
        map.edges[2] += 1;
    }
    // The barrierless dissociations.
    for e in &other.dissociation_edges {
        let (minimum, channel) = (place(e.minimum)?, place(e.channel)?);
        if net
            .dissociation_edges
            .iter()
            .any(|x| x.level == e.level && x.minimum == minimum && x.channel == channel)
        {
            continue;
        }
        let mut new = e.clone();
        new.id = net.dissociation_edges.len();
        new.minimum = minimum;
        new.channel = channel;
        new.job = e.job.map(|j| j + job_offset);
        net.dissociation_edges.push(new);
        map.edges[3] += 1;
    }
    // The path peaks (records beside the network): one that is the same point as a recorded
    // one is not recorded again.
    for p in &other.path_peaks {
        let mut known = false;
        for q in &net.path_peaks {
            if q.level == p.level && q.system.same_as(q.energy, &p.system, p.energy, matcher)? {
                known = true;
                break;
            }
        }
        if known {
            continue;
        }
        let id = net.path_peaks.len();
        net.path_peaks.push(PathPeak {
            id,
            ends: [
                p.ends[0].map(&place).transpose()?,
                p.ends[1].map(&place).transpose()?,
            ],
            job: p.job.map(|j| j + job_offset),
            ..p.clone()
        });
        map.peaks.push((p.id, id));
    }
    // The jobs and the warnings. (The trajectories of the jobs stay in their run directories.)
    for j in &other.jobs {
        net.jobs.push(JobRecord {
            id: j.id + job_offset,
            node: place(j.node)?,
            new_nodes: j
                .new_nodes
                .iter()
                .filter(|&&n| map.added.get(n).copied().unwrap_or(false))
                .map(|&n| place(n))
                .collect::<MoptResult<_>>()?,
            new_edges: j
                .new_edges
                .iter()
                .filter_map(|&e| ts_edge_of.get(e).copied().flatten())
                .collect(),
            trajectory: None,
            ..j.clone()
        });
    }
    for w in &other.warnings {
        net.warnings.push(NetworkWarning {
            job: w.job.map(|j| j + job_offset),
            message: w.message.clone(),
        });
    }
    let (a, b) = (&mut net.evaluations, &other.evaluations);
    a.energies += b.energies;
    a.gradients += b.gradients;
    a.hessians += b.hessians;
    a.micro_relaxations += b.micro_relaxations;
    a.micro_iterations += b.micro_iterations;
    Ok(map)
}
