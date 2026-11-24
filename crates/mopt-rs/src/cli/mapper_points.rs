//! `run_mapper -single_points RUN_DIR`: the energies of every node of the network of another
//! run on the surface of this run, at the structures of that network (the layer `2//1`): no
//! optimization and no Hessian. A minimum, a transition state and a saddle point of higher
//! order have one energy on their state; a crossing point the energies of its two states (the
//! node has their mean); a dissociation channel the sum of its fragments, each on its own. The
//! network of the run is the first one with these energies: the same nodes, numbers, labels
//! and frequencies, the barriers of the edges computed again. The first run is only read.
//! The rules are in `docs/source/mapper_design.md`.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::fmt::Write as _;
use std::path::Path;

use super::mapper_remap::FirstNetwork;
use crate::core_infra::json::JsonValue;
use crate::domain::Units;
use crate::error::{MoptError, MoptResult, ResultExt};
use crate::workflow::mapper::network::{Level, Network, Node, NodeKind};
use crate::workflow::mapper::store::write_whole;
use crate::workflow::mapper::{
    Configuration, ExplorationMethod, Job, JobContext, JobOutcome, MapperWorkers, Task,
};

/// The name of the method of such a run in its network file.
pub const POINTS_METHOD: &str = "single_points";

/// The table of the energies of both surfaces, in the run directory.
pub const POINTS_TABLE: &str = "single_points.csv";

/// A method without tasks: a run of single points explores nothing.
pub struct NoTasks;

impl ExplorationMethod for NoTasks {
    fn name(&self) -> &str {
        POINTS_METHOD
    }

    fn tasks(&self, _node: &Node, _network: &Network) -> MoptResult<Vec<Task>> {
        Ok(Vec::new())
    }

    fn run(&self, job: &Job, _context: &mut JobContext<'_>) -> MoptResult<JobOutcome> {
        Err(MoptError::config(format!(
            "{POINTS_METHOD} has no job (task '{}')",
            job.task.key
        )))
    }
}

/// Where the energy of an evaluation goes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Target {
    /// The energy of a node (of the first of its two states for a crossing point).
    Node(usize),
    /// The energy of the second state of a crossing point.
    Second(usize),
    /// The energy of a fragment record.
    Fragment(usize),
}

/// One evaluation.
struct Evaluation {
    target: Target,
    /// What it is, for the log.
    label: String,
    system: Configuration,
    /// The state it is on (its index among the states of the run), or the structure as it is.
    on: Option<usize>,
    /// The followed states at the structure.
    roots: crate::workflow::mapper::FollowedRoots,
}

/// The evaluations of the network `net`: every node and fragment once, a crossing point twice.
fn evaluations(net: &Network) -> MoptResult<Vec<Evaluation>> {
    let index = |state: &crate::workflow::mapper::ElectronicState, what: &str| {
        if net.states.is_empty() {
            return Ok(None);
        }
        net.states
            .iter()
            .position(|s| s == state)
            .map(Some)
            .ok_or_else(|| {
                MoptError::config(format!(
                    "{what} is on a state that is not among the states of its network"
                ))
            })
    };
    let mut out = Vec::new();
    for n in net.nodes.iter().filter(|n| n.level == Level::Exploration) {
        match &n.kind {
            // The sum of its fragments (a channel of a periodic cell has none: its own
            // structure, below).
            NodeKind::Dissociated if !n.parts.is_empty() => {}
            NodeKind::Crossing { states, .. } => {
                for (side, state) in states.iter().enumerate() {
                    let on = index(state, &n.label())?.ok_or_else(|| {
                        MoptError::config(format!(
                            "{} is a crossing point of a network without states",
                            n.label()
                        ))
                    })?;
                    out.push(Evaluation {
                        target: if side == 0 {
                            Target::Node(n.id)
                        } else {
                            Target::Second(n.id)
                        },
                        label: format!("{} on state {}", n.label(), on + 1),
                        system: n.system.clone(),
                        on: Some(on),
                        roots: n.roots.clone(),
                    });
                }
            }
            _ => out.push(Evaluation {
                target: Target::Node(n.id),
                label: n.label(),
                system: n.system.clone(),
                on: index(&n.state, &n.label())?,
                roots: n.roots.clone(),
            }),
        }
    }
    for f in net
        .fragments
        .iter()
        .filter(|f| f.level == Level::Exploration)
    {
        out.push(Evaluation {
            target: Target::Fragment(f.id),
            label: format!(
                "fragment {} (charge {}, multiplicity {})",
                f.formula, f.charge, f.multiplicity
            ),
            system: Configuration::Structure(f.system.clone()),
            on: None,
            roots: Vec::new(),
        });
    }
    Ok(out)
}

/// What the single points of a network gave.
pub struct Points {
    /// The network with the new energies.
    pub network: Network,
    /// How many energies were computed (or taken from the journal).
    pub evaluations: usize,
}

/// The energies of every node of `first` on the surface of `workers`, `at_once` at a time, and
/// the network with them; `single_points.csv` is written to `dir`. `None`: the run was
/// stopped (nothing is written; the evaluations made are in the journal).
pub fn evaluate(
    first: &FirstNetwork,
    dir: &Path,
    workers: &dyn MapperWorkers,
    at_once: usize,
) -> MoptResult<Option<Points>> {
    let old = &first.network;
    let list = evaluations(old)?;
    let jobs_dir = dir.join("jobs");
    std::fs::create_dir_all(&jobs_dir)
        .map_err(|e| MoptError::from(e).context(jobs_dir.display()))?;
    let mut energies: Vec<f64> = Vec::with_capacity(list.len());
    for (chunk_index, chunk) in list.chunks(at_once.max(1)).enumerate() {
        if workers.run().stop_between_jobs() {
            return Ok(None);
        }
        let offset = chunk_index * at_once.max(1);
        let one = |k: usize, e: &Evaluation| -> MoptResult<Option<f64>> {
            let job = Job {
                id: offset + k,
                node: match e.target {
                    Target::Node(id) | Target::Second(id) => id,
                    Target::Fragment(_) => 0,
                },
                task: Task {
                    key: format!("point:{}", offset + k),
                    description: format!("the energy of {}", e.label),
                    parameters: JsonValue::Null,
                },
                start: e.system.clone(),
                state: e.on.unwrap_or(0),
                roots: e.roots.clone(),
            };
            let jobs = workers.job(&job, None)?;
            jobs.energy(&e.system, e.on, &e.label).context(&e.label)
        };
        let results: Vec<MoptResult<Option<f64>>> = if chunk.len() == 1 {
            vec![one(0, &chunk[0])]
        } else {
            std::thread::scope(|scope| {
                let threads: Vec<_> = chunk
                    .iter()
                    .enumerate()
                    .map(|(k, e)| scope.spawn(move || one(k, e)))
                    .collect();
                threads
                    .into_iter()
                    .map(|t| {
                        t.join().unwrap_or_else(|_| {
                            Err(MoptError::config(
                                "a thread of the single points ended without a result",
                            ))
                        })
                    })
                    .collect()
            })
        };
        for r in results {
            match r? {
                Some(e) => energies.push(e),
                None => return Ok(None),
            }
        }
        workers.run().progress(&format!(
            "Single points: {} of {} energies.",
            energies.len(),
            list.len()
        ));
    }

    // The network of the first run with the new energies.
    let mut net = old.clone();
    let mut second: std::collections::HashMap<usize, f64> = std::collections::HashMap::new();
    let mut gaps: std::collections::HashMap<usize, f64> = std::collections::HashMap::new();
    for (e, &energy) in list.iter().zip(&energies) {
        match e.target {
            Target::Node(id) => {
                if let Some(n) = net.nodes.get_mut(id) {
                    if let Some(g) = n.free_energy.as_mut() {
                        // The thermal part is that of the first surface.
                        *g += energy - n.energy;
                    }
                    n.energy = energy;
                }
            }
            Target::Second(id) => {
                second.insert(id, energy);
            }
            Target::Fragment(k) => {
                if let Some(f) = net.fragments.get_mut(k) {
                    f.energy = energy;
                }
            }
        }
    }
    // A crossing point: the mean of its two states, and their gap for the table.
    for (&id, &other) in &second {
        if let Some(n) = net.nodes.get_mut(id) {
            let first_state = n.energy;
            let mean = 0.5 * (first_state + other);
            if let Some(g) = n.free_energy.as_mut() {
                // (It was shifted by the first state above.)
                *g += mean - first_state;
            }
            n.energy = mean;
            gaps.insert(id, first_state - other);
        }
    }
    // A dissociation channel: the sum of its fragments.
    let channels: Vec<(usize, f64)> = net
        .nodes
        .iter()
        .filter(|n| {
            n.kind == NodeKind::Dissociated && n.level == Level::Exploration && !n.parts.is_empty()
        })
        .map(|n| {
            let records: Vec<usize> = n.parts.iter().map(|p| p.fragment).collect();
            (n.id, net.channel_energy(&records))
        })
        .collect();
    for (id, energy) in channels {
        if let Some(n) = net.nodes.get_mut(id) {
            n.energy = energy;
        }
    }
    // The barriers of the edges from the energies of the nodes; the energies along the paths
    // are those of the first surface and are left out.
    let energy_of = |net: &Network, id: usize| net.nodes.get(id).map_or(f64::NAN, |n| n.energy);
    let tops: Vec<[f64; 3]> = net
        .ts_edges
        .iter()
        .map(|e| {
            [
                energy_of(&net, e.ts),
                energy_of(&net, e.ends[0]),
                energy_of(&net, e.ends[1]),
            ]
        })
        .collect();
    for (e, [top, a, b]) in net.ts_edges.iter_mut().zip(tops) {
        e.barriers = [top - a, top - b];
        e.irc.sides.clear();
    }
    let tops: Vec<(f64, [Option<f64>; 2])> = net
        .crossing_edges
        .iter()
        .map(|e| {
            (
                energy_of(&net, e.crossing),
                [
                    e.ends[0].map(|id| energy_of(&net, id)),
                    e.ends[1].map(|id| energy_of(&net, id)),
                ],
            )
        })
        .collect();
    for (e, (top, ends)) in net.crossing_edges.iter_mut().zip(tops) {
        e.barriers = [ends[0].map(|a| top - a), ends[1].map(|b| top - b)];
        e.irc.sides.clear();
    }
    let differences: Vec<f64> = net
        .dissociation_edges
        .iter()
        .map(|e| energy_of(&net, e.channel) - energy_of(&net, e.minimum))
        .collect();
    for (e, d) in net.dissociation_edges.iter_mut().zip(differences) {
        e.energy_difference = d;
        // (The scan of the edge is that of the first surface, like the energies along the
        // paths.)
        e.scan = None;
    }
    // The path peaks are left alone (user decision, 2026-11-19): they are records of the
    // first run on its surface, with no energy here, and stay in its files.
    net.path_peaks.clear();

    // The table: both energies of every node and fragment.
    let kcal = Units::HARTREE_TO_KCAL_MOL;
    let (low1, low2) = (
        old.lowest_minimum().unwrap_or(0.0),
        net.lowest_minimum().unwrap_or(0.0),
    );
    let mut text = String::from(
        "item,label,kind,first_energy_hartree,energy_hartree,first_relative_kcal_mol,\
         relative_kcal_mol,gap_hartree\n",
    );
    for (a, b) in old.nodes.iter().zip(&net.nodes) {
        if a.level != Level::Exploration {
            continue;
        }
        let _ = writeln!(
            text,
            "node:{},{},{},{:.12},{:.12},{:.4},{:.4},{}",
            a.id,
            a.label(),
            a.kind.tag(),
            a.energy,
            b.energy,
            (a.energy - low1) * kcal,
            (b.energy - low2) * kcal,
            gaps.get(&a.id)
                .map_or_else(String::new, |g| format!("{g:.12}")),
        );
    }
    for (a, b) in old.fragments.iter().zip(&net.fragments) {
        let _ = writeln!(
            text,
            "fragment:{},{} (charge {}; multiplicity {}),fragment,{:.12},{:.12},,,",
            a.id, a.formula, a.charge, a.multiplicity, a.energy, b.energy,
        );
    }
    write_whole(&dir.join(POINTS_TABLE), text.as_bytes())?;
    // The path files of the first run beside the network (their structures, and the energies
    // in their titles, are those of the first surface).
    let paths = first.dir.join("paths");
    if paths.is_dir() {
        let to = dir.join("paths");
        std::fs::create_dir_all(&to).map_err(|e| MoptError::from(e).context(to.display()))?;
        for entry in paths
            .read_dir()
            .map_err(|e| MoptError::from(e).context(paths.display()))?
        {
            let entry = entry.map_err(|e| MoptError::from(e).context(paths.display()))?;
            if entry.path().is_file() {
                let target = to.join(entry.file_name());
                std::fs::copy(entry.path(), &target)
                    .map_err(|e| MoptError::from(e).context(target.display()))?;
            }
        }
    }
    Ok(Some(Points {
        network: net,
        evaluations: energies.len(),
    }))
}
