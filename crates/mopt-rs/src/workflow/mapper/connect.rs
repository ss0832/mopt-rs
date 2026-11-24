//! Missing connections (`method=connect`, `goal=FILE`): the known minima as a complete graph,
//! a pair joined by a transition-state edge weighing 0, a pair whose attempts are used up
//! infinitely much, and any other pair `D^2` (`D` the distance after the best superposition);
//! Dijkstra's method gives the path of least weight from the starts to the goal, and every
//! pair of it with a weight above 0 (a gap) is a task of the minimum that runs it: a two-point
//! method of [`crate::workflow::pair_search`] between the two, its maxima followed to
//! transition states and their IRCs as for the other methods. The path is computed again
//! after every job, until the goal is joined to a start (the stop of the goal in
//! [`super::driver`]) or no gap can be tried. The rules are in `docs/source/mapper_design.md`.
//!
//! After J. M. Carr, S. A. Trygubenko, D. J. Wales, J. Chem. Phys. 2005, 122, 234903
//! (arXiv:cond-mat/0504785).
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::Mutex;

use super::configuration::Configuration;
use super::follow::Guess;
use super::method::{ExplorationMethod, Job, JobContext, JobOutcome, PairSearchRequest, Task};
use super::network::{Level, Network, Node, NodeKind};
use crate::coord::invariant_rmsd::{invariant_rmsd, renumbered, RmsdOptions};
use crate::core_infra::json::JsonValue;
use crate::domain::MolecularSystem;
use crate::error::{MoptError, MoptResult};
use crate::workflow::pair_search::{Ends, PairMethodRegistry};

/// The two-point method of a gap when none is given.
pub const DEFAULT_PATH: &str = "neb";

/// Attempts of one pair before its weight is infinite (`n_max`).
pub const DEFAULT_ATTEMPTS: usize = 2;

/// The images of the method on the first attempt when `method_path_images` is not given
/// (those of `neb`); attempt `k` doubles them `k - 1` times.
pub const DEFAULT_IMAGES: usize = 10;

/// The settings of the method (`method_KEY=VALUE`).
#[derive(Debug, Clone, PartialEq)]
pub struct ConnectSettings {
    /// The two-point method of the registry (`method_path`).
    pub path: String,
    /// Its settings (`method_path_KEY`).
    pub path_settings: Vec<(String, String)>,
    /// Attempts of one pair (`method_attempts`).
    pub attempts: usize,
}

impl Default for ConnectSettings {
    fn default() -> Self {
        Self {
            path: DEFAULT_PATH.to_string(),
            path_settings: Vec::new(),
            attempts: DEFAULT_ATTEMPTS,
        }
    }
}

impl ConnectSettings {
    /// From the `method_*` settings (without the prefix); what cannot run is refused.
    pub fn parse(extra: &BTreeMap<String, String>) -> MoptResult<Self> {
        let mut s = Self::default();
        for (k, v) in extra {
            match k.as_str() {
                "path" => s.path = v.trim().to_string(),
                "attempts" => {
                    s.attempts = v
                        .trim()
                        .parse()
                        .ok()
                        .filter(|&n: &usize| n >= 1)
                        .ok_or_else(|| {
                            MoptError::parse(format!(
                                "method_attempts is a whole number of at least 1, not '{v}'"
                            ))
                        })?
                }
                _ => match k.strip_prefix("path_") {
                    Some(key) if !key.is_empty() => {
                        s.path_settings.push((key.to_string(), v.clone()));
                    }
                    _ => {
                        return Err(MoptError::parse(format!(
                            "the mapper method connect: unknown setting method_{k} (keys: \
                             method_path, method_path_KEY, method_attempts)"
                        )))
                    }
                },
            }
        }
        let method = PairMethodRegistry::default().get(&s.path)?;
        if method.ends() == Ends::One {
            return Err(MoptError::parse(format!(
                "method_path: {} starts from one structure; a gap has two minima (a two-point \
                 method: neb, dmf, dsafir, gsm, ...)",
                method.name()
            )));
        }
        Ok(s)
    }
}

/// A gap of the current path: the pair, which end runs it, and the attempt.
#[derive(Debug, Clone, PartialEq)]
struct Gap {
    pair: [usize; 2],
    runner: usize,
    other: usize,
    attempt: usize,
}

/// The plan of a network (the gaps of its path), kept until the network changes.
#[derive(Default)]
struct Cache {
    key: Option<(usize, usize, usize)>,
    gaps: Vec<Gap>,
    /// `D^2` of pairs (structures never change).
    distances: HashMap<[usize; 2], f64>,
}

/// The method.
pub struct Connect {
    settings: ConnectSettings,
    images_key: bool,
    cache: Mutex<Cache>,
}

/// The key of the attempts of a pair: `connect:A:B:` (`A < B`), the attempt after it.
fn pair_key(pair: [usize; 2]) -> String {
    format!("connect:{}:{}:", pair[0], pair[1])
}

fn ordered(a: usize, b: usize) -> [usize; 2] {
    if a < b {
        [a, b]
    } else {
        [b, a]
    }
}

/// The match of `b` onto `a`: the atoms of `b` renumbered within their elements and the
/// structure turned for the least distance (the minimum over the numberings of identical
/// atoms, as Carr, Trygubenko and Wales minimize over permutation-inversion isomers; mirror
/// images are not taken, which a path cannot reach), and its squared distance (Bohr^2).
fn matched(
    a: &MolecularSystem,
    b: &MolecularSystem,
    held: &[usize],
) -> MoptResult<(MolecularSystem, f64)> {
    // The atoms of hard constraints keep their numbers: each is a class of its own.
    let classes = |s: &MolecularSystem| -> Vec<usize> {
        s.elements
            .iter()
            .enumerate()
            .map(|(i, e)| match held.binary_search(&i) {
                Ok(_) => 1000 + i,
                Err(_) => usize::from(e.atomic_number()),
            })
            .collect()
    };
    let m = invariant_rmsd(a, b, &classes(a), &classes(b), &RmsdOptions::default())?;
    let renamed = renumbered(b, &m.permutation)?;
    Ok((renamed, m.rmsd * m.rmsd * a.n_atoms() as f64))
}

/// The squared distance of two nodes after the best superposition and numbering of their
/// identical atoms (Bohr^2), or of two points of plain variables.
fn squared_distance(a: &Node, b: &Node, held: &[usize]) -> MoptResult<f64> {
    match (a.system.structure(), b.system.structure()) {
        (Some(sa), Some(sb)) => Ok(matched(sa, sb, held)?.1),
        _ => {
            let (x, y) = (a.system.flat(), b.system.flat());
            Ok((0..x.nrows().min(y.nrows()))
                .map(|k| (x[k] - y[k]).powi(2))
                .sum())
        }
    }
}

impl Connect {
    /// The name in the registry.
    pub const NAME: &'static str = "connect";

    /// The method with `settings`.
    pub fn new(settings: ConnectSettings) -> MoptResult<Self> {
        let method = PairMethodRegistry::default().get(&settings.path)?;
        let images_key = method.keys().iter().any(|(k, _)| *k == "images");
        Ok(Self {
            settings,
            images_key,
            cache: Mutex::new(Cache::default()),
        })
    }

    /// Attempts of each pair so far, from the finished jobs.
    fn attempts(network: &Network) -> HashMap<[usize; 2], usize> {
        let mut out: HashMap<[usize; 2], usize> = HashMap::new();
        for j in &network.jobs {
            let Some(rest) = j.task.strip_prefix("connect:") else {
                continue;
            };
            let mut parts = rest.split(':');
            let (Some(a), Some(b)) = (
                parts.next().and_then(|x| x.parse().ok()),
                parts.next().and_then(|x| x.parse().ok()),
            ) else {
                continue;
            };
            *out.entry(ordered(a, b)).or_insert(0) += 1;
        }
        out
    }

    /// The gaps of the path of least weight from the starts to the goal of `network`.
    fn plan(&self, network: &Network) -> MoptResult<Vec<Gap>> {
        let key = (
            network.nodes.len(),
            network.ts_edges.len(),
            network.jobs.len(),
        );
        let mut cache = self
            .cache
            .lock()
            .map_err(|_| MoptError::config("the plan of the missing connections is poisoned"))?;
        if cache.key == Some(key) {
            return Ok(cache.gaps.clone());
        }
        let gaps = self.compute(network, &mut cache.distances)?;
        cache.key = Some(key);
        cache.gaps.clone_from(&gaps);
        Ok(gaps)
    }

    fn compute(
        &self,
        network: &Network,
        distances: &mut HashMap<[usize; 2], f64>,
    ) -> MoptResult<Vec<Gap>> {
        let (Some(goal), starts) = (network.goal_node(), network.start_nodes()) else {
            return Ok(Vec::new());
        };
        if starts.is_empty() || network.goal_joined() == Some(true) {
            return Ok(Vec::new());
        }
        let minima: Vec<usize> = network
            .nodes
            .iter()
            .filter(|n| n.kind == NodeKind::Minimum && n.level == Level::Exploration)
            .map(|n| n.id)
            .collect();
        let index: HashMap<usize, usize> =
            minima.iter().enumerate().map(|(k, &id)| (id, k)).collect();
        let joined: BTreeSet<[usize; 2]> = network
            .ts_edges
            .iter()
            .filter(|e| e.level == Level::Exploration)
            .map(|e| ordered(e.ends[0], e.ends[1]))
            .collect();
        let tried = Self::attempts(network);
        let runnable = |id: usize| network.nodes.get(id).is_some_and(|n| n.excluded.is_none());
        let m = minima.len();
        let mut dist = vec![f64::INFINITY; m];
        let mut previous: Vec<Option<usize>> = vec![None; m];
        let mut done = vec![false; m];
        for s in &starts {
            if let Some(&k) = index.get(s) {
                dist[k] = 0.0;
            }
        }
        let Some(&goal_k) = index.get(&goal) else {
            return Ok(Vec::new());
        };
        while let Some(u) = (0..m)
            .filter(|&k| !done[k] && dist[k].is_finite())
            .min_by(|&a, &b| dist[a].total_cmp(&dist[b]))
        {
            done[u] = true;
            if u == goal_k {
                break;
            }
            for v in 0..m {
                if done[v] {
                    continue;
                }
                let pair = ordered(minima[u], minima[v]);
                let w = if joined.contains(&pair) {
                    0.0
                } else if tried.get(&pair).copied().unwrap_or(0) >= self.settings.attempts
                    || !(runnable(minima[u]) || runnable(minima[v]))
                {
                    f64::INFINITY
                } else {
                    match distances.get(&pair) {
                        Some(&d) => d,
                        None => {
                            let d = squared_distance(
                                network.node(pair[0])?,
                                network.node(pair[1])?,
                                &network.held.atoms,
                            )?;
                            distances.insert(pair, d);
                            d
                        }
                    }
                };
                if dist[u] + w < dist[v] {
                    dist[v] = dist[u] + w;
                    previous[v] = Some(u);
                }
            }
        }
        if !dist[goal_k].is_finite() {
            return Ok(Vec::new());
        }
        let mut path = vec![goal_k];
        while let Some(p) = previous[*path.last().unwrap_or(&goal_k)] {
            path.push(p);
        }
        path.reverse();
        let mut gaps = Vec::new();
        for w in path.windows(2) {
            let (a, b) = (minima[w[0]], minima[w[1]]);
            let pair = ordered(a, b);
            if joined.contains(&pair) {
                continue;
            }
            let (runner, other) = if runnable(a) { (a, b) } else { (b, a) };
            gaps.push(Gap {
                pair,
                runner,
                other,
                attempt: tried.get(&pair).copied().unwrap_or(0) + 1,
            });
        }
        Ok(gaps)
    }
}

impl ExplorationMethod for Connect {
    fn name(&self) -> &str {
        Self::NAME
    }

    fn tasks_follow_network(&self) -> bool {
        true
    }

    fn tasks(&self, node: &Node, network: &Network) -> MoptResult<Vec<Task>> {
        if node.kind != NodeKind::Minimum {
            return Ok(Vec::new());
        }
        Ok(self
            .plan(network)?
            .into_iter()
            .filter(|g| g.runner == node.id)
            .map(|g| {
                let label = |id: usize| {
                    network
                        .node(id)
                        .map_or_else(|_| format!("node {id}"), Node::label)
                };
                let mut p = BTreeMap::new();
                p.insert("other".to_string(), JsonValue::Number(g.other as f64));
                p.insert("attempt".to_string(), JsonValue::Number(g.attempt as f64));
                Task {
                    key: format!("{}{}", pair_key(g.pair), g.attempt),
                    description: format!(
                        "missing connection {} - {} (attempt {})",
                        label(g.runner),
                        label(g.other),
                        g.attempt
                    ),
                    parameters: JsonValue::Object(p),
                }
            })
            .collect())
    }

    fn run(&self, job: &Job, ctx: &mut JobContext<'_>) -> MoptResult<JobOutcome> {
        let number = |key: &str| {
            job.task
                .parameters
                .get(key)
                .and_then(JsonValue::as_usize)
                .ok_or_else(|| {
                    MoptError::parse(format!("connect: the task '{}' has no {key}", job.task.key))
                })
        };
        let (other, attempt) = (number("other")?, number("attempt")?);
        // The far end with its identical atoms numbered as in the start, so that the path does
        // not exchange them.
        let end = match (
            job.start.structure(),
            ctx.network.node(other)?.system.structure(),
        ) {
            (Some(a), Some(b)) => {
                Configuration::Structure(matched(a, b, &ctx.network.held.atoms)?.0)
            }
            _ => ctx.network.node(other)?.system.clone(),
        };
        let mut settings = self.settings.path_settings.clone();
        if attempt > 1 && self.images_key {
            let base = settings
                .iter()
                .find(|(k, _)| k == "images")
                .and_then(|(_, v)| v.trim().parse::<usize>().ok())
                .unwrap_or(DEFAULT_IMAGES);
            settings.retain(|(k, _)| k != "images");
            let factor = 1usize << (attempt - 1).min(8);
            settings.push(("images".to_string(), (base * factor).to_string()));
        }
        let mut outcome = JobOutcome::default();
        let search = ctx.jobs.pair_search(
            &PairSearchRequest {
                label: format!("{} of the gap (attempt {attempt})", self.settings.path),
                job: job.id,
                method: self.settings.path.clone(),
                structures: vec![job.start.clone(), end],
                settings,
                dissociation_scale: None,
                dissociation_atoms: None,
            },
            ctx.frames,
        )?;
        if search.interrupted {
            return Ok(JobOutcome::stopped());
        }
        if search.estimates.is_empty() {
            outcome.warnings.push(format!(
                "{} gave no transition-state estimate for the gap ({})",
                self.settings.path, search.end
            ));
            return Ok(outcome);
        }
        let (path, converged) = (search.path, search.converged);
        let mut estimates = search.estimates;
        estimates.sort_by(|a, b| b.energy.total_cmp(&a.energy));
        let guesses: Vec<Guess> = estimates
            .into_iter()
            .enumerate()
            .map(|(m, e)| Guess {
                system: e.system,
                origin: format!("{} maximum {} of the gap", self.settings.path, m + 1),
            })
            .collect();
        if !super::peak::follow_with_peak(
            job,
            ctx,
            &guesses,
            &path,
            &format!("{} of the gap", self.settings.path),
            converged,
            &mut outcome,
        )? {
            return Ok(JobOutcome::stopped());
        }
        Ok(outcome)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn the_settings_are_checked() {
        let mut o = BTreeMap::new();
        assert_eq!(
            ConnectSettings::parse(&o).unwrap(),
            ConnectSettings::default()
        );
        o.insert("attempts".to_string(), "3".to_string());
        o.insert("path_images".to_string(), "12".to_string());
        let s = ConnectSettings::parse(&o).unwrap();
        assert_eq!(s.attempts, 3);
        assert_eq!(
            s.path_settings,
            vec![("images".to_string(), "12".to_string())]
        );
        o.insert("attempts".to_string(), "0".to_string());
        assert!(ConnectSettings::parse(&o).is_err());
        o.clear();
        o.insert("nope".to_string(), "1".to_string());
        assert!(ConnectSettings::parse(&o).is_err());
        o.clear();
        o.insert("path".to_string(), "no_such_method".to_string());
        assert!(ConnectSettings::parse(&o).is_err());
    }
}
