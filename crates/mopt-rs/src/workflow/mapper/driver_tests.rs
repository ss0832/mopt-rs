//! Tests of the driver and the loop of the mapper with a toy method on argon dimers: the
//! distance of the dimer stands for the minimum, so that every step of the bookkeeping can be
//! checked without a calculation.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

#![allow(clippy::unwrap_used, clippy::panic)]

use std::cell::Cell;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use super::*;
use crate::core_infra::json::JsonValue;
use crate::domain::{positions, Angstrom, Element, MolecularSystem, SpinCharge};
use crate::workflow::explore::MaxTrials;
use crate::workflow::mapper::method::{
    FoundReaction, IrcOutcome, OptimizeRequest, Optimized, SaddleEnd, Stationary,
};
use crate::workflow::mapper::network::{mapper_matcher, InputRecord};
use crate::workflow::mapper::network_json::{from_json, to_json};
use crate::workflow::mapper::priority::ExplorationPriorityRegistry;
use crate::workflow::mapper::Configuration;
use crate::workflow::mapper::{
    explore, explore_with, ExplorationEnd, MapperObserver, MapperWorkers, PrioritySettings,
    Schedule, ScheduleMode,
};

fn dimer(d: f64) -> Configuration {
    dimer_structure(d).into()
}

fn dimer_structure(d: f64) -> MolecularSystem {
    MolecularSystem::new(
        vec![
            Element::from_atomic_number(18).unwrap(),
            Element::from_atomic_number(18).unwrap(),
        ],
        &positions::<Angstrom>(&[[0.0, 0.0, 0.0], [d, 0.0, 0.0]]),
        SpinCharge::default(),
        None,
    )
    .unwrap()
}

fn length(point: &Configuration) -> f64 {
    let sys = point.structure().unwrap();
    sys.distance(crate::domain::AtomId::new(0), crate::domain::AtomId::new(1))
        .value()
        * crate::domain::Units::BOHR_TO_ANGSTROM
}

fn energy(d: f64) -> f64 {
    -1.0 / d
}

/// Two tasks per minimum: a step of 1 and of 2 Angstrom outwards, up to 8 Angstrom; the task
/// `fail` of the minimum at 5 Angstrom fails.
struct Toy;

impl ExplorationMethod for Toy {
    fn name(&self) -> &str {
        "toy"
    }

    fn tasks(&self, node: &Node, _network: &Network) -> MoptResult<Vec<Task>> {
        let mut out = Vec::new();
        for step in [1.0, 2.0] {
            let mut p = BTreeMap::new();
            p.insert("step".to_string(), JsonValue::Number(step));
            out.push(Task {
                key: format!("step{step}"),
                description: format!("a step of {step} Angstrom"),
                parameters: JsonValue::Object(p),
            });
        }
        if (length(&node.system) - 5.0).abs() < 1e-6 {
            out.push(Task {
                key: "fail".to_string(),
                description: "fails".to_string(),
                parameters: JsonValue::Null,
            });
        }
        Ok(out)
    }

    fn run(&self, job: &Job, context: &mut JobContext<'_>) -> MoptResult<JobOutcome> {
        let Some(step) = job.task.parameters.get("step").and_then(JsonValue::as_f64) else {
            return Err(MoptError::calc("the toy calculation failed"));
        };
        let d = length(&job.start);
        context.frames.push(&job.start, energy(d), "start")?;
        let to = d + step;
        if to > 8.0 + 1e-9 {
            return Ok(JobOutcome {
                warnings: vec![format!("no reaction beyond {to} Angstrom")],
                ..JobOutcome::default()
            });
        }
        let ts = dimer(d + 0.5 * step);
        Ok(JobOutcome {
            saddles: Vec::new(),
            crossings: Vec::new(),
            peaks: Vec::new(),
            followed: Vec::new(),
            known: Vec::new(),
            interrupted: false,
            reactions: vec![FoundReaction {
                ts: Stationary {
                    index_wavenumbers: None,
                    energy: energy(d) + 0.01 * step,
                    system: ts,
                    wavenumbers: Some(vec![-10.0 * step]),
                    dissociation: None,
                },
                imaginary_wavenumber: Some(-10.0 * step),
                ends: [
                    Some(Stationary {
                        index_wavenumbers: None,
                        system: job.start.clone(),
                        energy: energy(d),
                        wavenumbers: None,
                        dissociation: None,
                    }),
                    Some(Stationary {
                        index_wavenumbers: None,
                        system: dimer(to),
                        energy: energy(to),
                        wavenumbers: Some(vec![100.0 * to]),
                        dissociation: None,
                    }),
                ],
                branches: Vec::new(),
                saddle_ends: [None, None],
                bifurcation_ends: Vec::new(),
            }],
            warnings: Vec::new(),
            wholes: Vec::new(),
        })
    }
}

/// Minimizations that keep the structure; a stop before the job `stop_at`.
struct ToyJobs {
    asked: Cell<usize>,
    stop_at: Option<usize>,
}

impl MapperJobs for ToyJobs {
    fn optimize(
        &self,
        start: &Configuration,
        _request: &OptimizeRequest,
        _frames: &mut dyn FrameSink,
    ) -> MoptResult<Optimized> {
        Ok(Optimized {
            index_wavenumbers: None,
            system: start.clone(),
            energy: energy(length(start)),
            converged: true,
            interrupted: false,
            path: Vec::new(),
            wavenumbers: Some(vec![100.0 * length(start)]),
        })
    }

    fn irc(&self, _ts: &Configuration, _frames: &mut dyn FrameSink) -> MoptResult<IrcOutcome> {
        Err(MoptError::unsupported("no IRC in the toy"))
    }

    fn stop_requested(&self) -> bool {
        false
    }

    fn stop_between_jobs(&self) -> bool {
        let k = self.asked.get();
        self.asked.set(k + 1);
        self.stop_at.is_some_and(|s| k >= s)
    }

    fn note(&self, _message: &str) {}
}

fn driver(network: Network, max: usize) -> MapperDriver {
    driver_with(network, max, Box::new(Toy))
}

fn driver_with(network: Network, max: usize, method: Box<dyn ExplorationMethod>) -> MapperDriver {
    let priority = ExplorationPriorityRegistry::default()
        .create("boltzmann", &PrioritySettings { temperature: 300.0 })
        .unwrap();
    MapperDriver::new(
        network,
        method,
        priority,
        vec![Box::new(MaxTrials { max })],
        DriverSettings {
            seed: 7,
            matcher: mapper_matcher(0.3, 1.0, false),
            excluded: Vec::new(),
            exclude_bond_rearrangement: false,
            max_bond_changes: None,
            max_species_changes: None,
            entrance_inputs: Vec::new(),
            excited_state: 0,
            goal_stop: true,
            goal_bonds: false,
            accept_temperature: None,
            input_states: Vec::new(),
            skip_failed_inputs: false,
            environments: false,
            environment_tolerance: 1.0 * crate::domain::Units::KCAL_MOL_TO_HARTREE,
            cell_contact_margin: None,
            verify_minima: false,
        },
    )
}

/// How the jobs of [`Uneven`] end.
#[derive(Clone, Copy, PartialEq)]
enum Finish {
    /// A few milliseconds by the number of the job, mixed.
    Mixed,
    /// Within every three jobs, the last ends first.
    Reverse,
}

/// The toy method with jobs of different lengths, so that jobs that run at the same time end
/// in another order than they started, and that stop when the run is stopped; with `slow`,
/// the task `slow` of the minimum at 6 Angstrom runs until it is stopped; with `panics`, the
/// task `panic` of the minimum at 7 Angstrom panics. The numbers of the jobs are kept in the
/// order they end.
struct Uneven {
    slow: bool,
    panics: bool,
    finish: Finish,
    ended: Arc<std::sync::Mutex<Vec<usize>>>,
}

impl Uneven {
    fn new(slow: bool) -> Self {
        Self {
            slow,
            panics: false,
            finish: Finish::Mixed,
            ended: Arc::default(),
        }
    }
}

impl ExplorationMethod for Uneven {
    fn name(&self) -> &str {
        "toy"
    }

    fn tasks(&self, node: &Node, network: &Network) -> MoptResult<Vec<Task>> {
        let mut tasks = Toy.tasks(node, network)?;
        let d = length(&node.system);
        for (on, key, at) in [(self.slow, "slow", 6.0), (self.panics, "panic", 7.0)] {
            if on && (d - at).abs() < 1e-6 {
                tasks.push(Task {
                    key: key.to_string(),
                    description: key.to_string(),
                    parameters: JsonValue::Null,
                });
            }
        }
        Ok(tasks)
    }

    fn run(&self, job: &Job, context: &mut JobContext<'_>) -> MoptResult<JobOutcome> {
        if job.task.key == "slow" {
            let start = Instant::now();
            while !context.jobs.stop_requested() {
                if start.elapsed() > Duration::from_secs(20) {
                    return Err(MoptError::calc("the slow task was never stopped"));
                }
                std::thread::sleep(Duration::from_millis(2));
            }
            return Ok(JobOutcome::stopped());
        }
        if job.task.key == "panic" {
            panic!("a stand-in job that panics");
        }
        let ms = match self.finish {
            Finish::Mixed => 5 + ((job.id * 7) % 5) as u64 * 3,
            Finish::Reverse => 3 + (2 - job.id % 3) as u64 * 25,
        };
        std::thread::sleep(Duration::from_millis(ms));
        if context.jobs.stop_requested() {
            return Ok(JobOutcome::stopped());
        }
        let out = Toy.run(job, context);
        self.ended.lock().unwrap().push(job.id);
        out
    }
}

/// The toy calculations for jobs that run at the same time: each job counts the jobs that
/// run with it, and stops at its time limit or when the run is stopped.
struct ToyWorkers {
    run: SyncToyJobs,
    running: AtomicUsize,
    most: AtomicUsize,
    stop: Arc<AtomicBool>,
    /// The run is stopped when this many jobs have started (0: never).
    stop_at_start: usize,
    /// The run is stopped when this many jobs have ended (0: never).
    stop_at_end: usize,
    started: AtomicUsize,
    ended: AtomicUsize,
}

struct SyncToyJobs {
    deadline: Option<Instant>,
    stop: Arc<AtomicBool>,
}

impl SyncToyJobs {
    fn stopped(&self) -> bool {
        self.stop.load(Ordering::SeqCst) || self.deadline.is_some_and(|d| Instant::now() >= d)
    }
}

impl MapperJobs for SyncToyJobs {
    fn optimize(
        &self,
        start: &Configuration,
        request: &OptimizeRequest,
        frames: &mut dyn FrameSink,
    ) -> MoptResult<Optimized> {
        ToyJobs {
            asked: Cell::new(0),
            stop_at: None,
        }
        .optimize(start, request, frames)
    }

    fn irc(&self, _ts: &Configuration, _frames: &mut dyn FrameSink) -> MoptResult<IrcOutcome> {
        Err(MoptError::unsupported("no IRC in the toy"))
    }

    fn stop_requested(&self) -> bool {
        self.stopped()
    }

    fn note(&self, _message: &str) {}
}

impl ToyWorkers {
    fn new(stop_at_start: usize) -> Self {
        Self::stopping(stop_at_start, 0)
    }

    fn stopping(stop_at_start: usize, stop_at_end: usize) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        Self {
            run: SyncToyJobs {
                deadline: None,
                stop: Arc::clone(&stop),
            },
            running: AtomicUsize::new(0),
            most: AtomicUsize::new(0),
            stop,
            stop_at_start,
            stop_at_end,
            started: AtomicUsize::new(0),
            ended: AtomicUsize::new(0),
        }
    }
}

/// The calculations of one toy job, which counts while it lives.
struct CountedJob<'a> {
    inner: SyncToyJobs,
    workers: &'a ToyWorkers,
}

impl Drop for CountedJob<'_> {
    fn drop(&mut self) {
        let w = self.workers;
        w.running.fetch_sub(1, Ordering::SeqCst);
        let ended = w.ended.fetch_add(1, Ordering::SeqCst) + 1;
        if w.stop_at_end > 0 && ended >= w.stop_at_end {
            w.stop.store(true, Ordering::SeqCst);
        }
    }
}

impl MapperJobs for CountedJob<'_> {
    fn optimize(
        &self,
        start: &Configuration,
        request: &OptimizeRequest,
        frames: &mut dyn FrameSink,
    ) -> MoptResult<Optimized> {
        self.inner.optimize(start, request, frames)
    }

    fn irc(&self, ts: &Configuration, frames: &mut dyn FrameSink) -> MoptResult<IrcOutcome> {
        self.inner.irc(ts, frames)
    }

    fn stop_requested(&self) -> bool {
        self.inner.stopped()
    }

    fn note(&self, _message: &str) {}
}

impl MapperWorkers for ToyWorkers {
    fn run(&self) -> &dyn MapperJobs {
        &self.run
    }

    fn job(&self, _job: &Job, deadline: Option<Instant>) -> MoptResult<Box<dyn MapperJobs + '_>> {
        let now = self.running.fetch_add(1, Ordering::SeqCst) + 1;
        self.most.fetch_max(now, Ordering::SeqCst);
        let started = self.started.fetch_add(1, Ordering::SeqCst) + 1;
        if self.stop_at_start > 0 && started >= self.stop_at_start {
            self.stop.store(true, Ordering::SeqCst);
        }
        Ok(Box::new(CountedJob {
            inner: SyncToyJobs {
                deadline,
                stop: Arc::clone(&self.stop),
            },
            workers: self,
        }))
    }
}

fn fresh() -> Network {
    let mut network = Network::new(vec![
        Element::from_atomic_number(18).unwrap(),
        Element::from_atomic_number(18).unwrap(),
    ]);
    network.inputs.push(InputRecord {
        file: "a.xyz".into(),
        frame: 0,
        node: None,
        goal: false,
    });
    network
}

fn text(network: &Network) -> String {
    to_json(network, JsonValue::Null).to_string_pretty()
}

#[test]
fn the_loop_explores_dedupes_and_records_every_job() {
    let mut d = driver(fresh(), 100);
    let jobs = ToyJobs {
        asked: Cell::new(0),
        stop_at: None,
    };
    let end = explore(&mut d, &jobs, &[dimer(4.0)], &mut []).unwrap();
    assert!(!end.interrupted);
    let net = d.network();
    // Minima at 4, 5, 6, 7, 8 Angstrom, each found once.
    let mut lengths: Vec<f64> = net.minima().map(|n| length(&n.system)).collect();
    lengths.sort_by(f64::total_cmp);
    assert_eq!(lengths.len(), 5, "{lengths:?}");
    for (l, want) in lengths.iter().zip([4.0, 5.0, 6.0, 7.0, 8.0]) {
        assert!((l - want).abs() < 1e-6);
    }
    // Every task of every minimum ran once: two per minimum and the failing one.
    assert_eq!(net.jobs.len(), 11);
    assert!(end.reason.unwrap().contains("Every task"));
    let failed: Vec<_> = net
        .jobs
        .iter()
        .filter(|j| matches!(j.status, JobStatus::Failed(_)))
        .collect();
    assert_eq!(failed.len(), 1);
    assert_eq!(failed[0].task, "fail");
    // Transition states of the steps 4->5, 4->6, 5->6, 5->7, 6->7, 6->8, 7->8, each once.
    assert_eq!(net.ts_edges.len(), 7);
    for e in &net.ts_edges {
        let (a, b) = (
            length(&net.nodes[e.ends[0]].system),
            length(&net.nodes[e.ends[1]].system),
        );
        assert!(b > a);
        assert!((e.barriers[0] - (net.nodes[e.ts].energy - energy(a))).abs() < 1e-12);
        // Numbered among the transition states, in the order of the edges.
        assert_eq!(net.nodes[e.ts].number, e.id);
        assert_eq!(
            e.irc.file.as_deref(),
            Some(format!("paths/TS{:06}_irc.xyz", e.id).as_str())
        );
    }
    // Minima and transition states are numbered apart: EQ0, EQ1, ... and TS0, TS1, ...
    let labels: Vec<String> = net.nodes.iter().map(Node::label).collect();
    assert!(labels.contains(&"TS0".to_string()) && labels.contains(&"EQ0".to_string()));
    assert!(net.warnings.iter().any(|w| w.message.contains("beyond")));
    assert!(net.warnings.iter().any(|w| w.message.contains("failed")));
    // The input is the origin of every node.
    assert!(net.nodes.iter().all(|n| n.origin.input == Some(0)));
    // Every minimum carries the wavenumbers of the Hessian that verified it: the input from
    // its optimization, the others from the end of the reaction that found them; every
    // transition state those of the start check of its IRC.
    for n in net.minima() {
        let w = n
            .frequencies
            .as_ref()
            .expect("the wavenumbers of a minimum");
        assert!((w[0] - 100.0 * length(&n.system)).abs() < 1e-6, "{w:?}");
    }
    for n in net
        .nodes
        .iter()
        .filter(|n| n.kind == NodeKind::TransitionState)
    {
        let w = n.frequencies.as_ref().expect("the wavenumbers of a TS");
        assert!(w.len() == 1 && w[0] < 0.0, "{}: {w:?}", n.label());
    }
}

#[test]
fn a_stopped_run_goes_on_from_its_file_to_the_same_network() {
    let jobs = |stop_at| ToyJobs {
        asked: Cell::new(0),
        stop_at,
    };
    let mut whole = driver(fresh(), 100);
    explore(&mut whole, &jobs(None), &[dimer(4.0)], &mut []).unwrap();
    // Stopped before the fifth job (the check before every job, the first one being that of
    // the input), written, read back, continued.
    let mut first = driver(fresh(), 100);
    let end = explore(&mut first, &jobs(Some(5)), &[dimer(4.0)], &mut []).unwrap();
    assert!(end.interrupted && end.reason.is_none());
    assert_eq!(first.network().jobs.len(), 4);
    let (back, _) = from_json(&JsonValue::parse(&text(first.network())).unwrap()).unwrap();
    let mut second = driver(back, 100);
    let end = explore(&mut second, &jobs(None), &[dimer(4.0)], &mut []).unwrap();
    assert!(!end.interrupted);
    assert_eq!(text(second.network()), text(whole.network()));
    // A limit of jobs ends the run, also when it is read back with as many jobs done.
    let mut limited = driver(fresh(), 3);
    let end = explore(&mut limited, &jobs(None), &[dimer(4.0)], &mut []).unwrap();
    assert_eq!(limited.network().jobs.len(), 3);
    assert!(end.reason.unwrap().contains("Max"));
    let (back, _) = from_json(&JsonValue::parse(&text(limited.network())).unwrap()).unwrap();
    let mut again = driver(back, 3);
    assert!(matches!(again.ask().unwrap(), MapperAction::Finished(_)));
}

#[test]
fn excluded_nodes_are_kept_but_not_explored() {
    let mut d = driver(fresh(), 100);
    d.settings.excluded = vec![1];
    let jobs = ToyJobs {
        asked: Cell::new(0),
        stop_at: None,
    };
    explore(&mut d, &jobs, &[dimer(4.0)], &mut []).unwrap();
    let net = d.network();
    assert!(net.nodes[1].excluded.is_some());
    assert!(net.jobs.iter().all(|j| j.node != 1));
    // Two inputs that are the same minimum are one node.
    let mut two = fresh();
    two.inputs.push(InputRecord {
        file: "b.xyz".into(),
        frame: 0,
        node: None,
        goal: false,
    });
    let mut d = driver(two, 1);
    explore(&mut d, &jobs, &[dimer(4.0), dimer(4.0)], &mut []).unwrap();
    assert_eq!(d.network().inputs[0].node, d.network().inputs[1].node);
    // Asking twice without telling is refused.
    let mut d = driver(fresh(), 100);
    d.add_input(0, dimer(4.0), energy(4.0)).unwrap();
    assert!(matches!(d.ask().unwrap(), MapperAction::Run(_)));
    assert!(d.ask().is_err());
}

fn argon_atom() -> MolecularSystem {
    MolecularSystem::new(
        vec![Element::from_atomic_number(18).unwrap()],
        &positions::<Angstrom>(&[[0.0, 0.0, 0.0]]),
        SpinCharge::default(),
        None,
    )
    .unwrap()
}

fn two_argon_atoms(separated: bool, each: f64) -> crate::workflow::mapper::Dissociation {
    let atom = crate::workflow::mapper::FragmentRecord {
        id: 0,
        key: 42,
        formula: "Ar".into(),
        charge: 0,
        multiplicity: 1,
        level: Level::Exploration,
        system: argon_atom(),
        energy: each,
        frequencies: Some(Vec::new()),
    };
    crate::workflow::mapper::Dissociation {
        separated,
        parts: vec![(vec![0], atom.clone()), (vec![1], atom)],
        energy: 2.0 * each,
        scan: None,
    }
}

fn reaction(ts: f64, ends: [Stationary; 2]) -> FoundReaction {
    FoundReaction {
        ts: Stationary {
            index_wavenumbers: None,
            system: dimer(ts),
            energy: energy(ts) + 0.05,
            wavenumbers: None,
            dissociation: None,
        },
        imaginary_wavenumber: Some(-10.0),
        ends: ends.map(Some),
        branches: Vec::new(),
        saddle_ends: [None, None],
        bifurcation_ends: Vec::new(),
    }
}

#[test]
fn ends_apart_are_channels_and_complexes_dissociate_into_them() {
    let mut d = driver(fresh(), 100);
    let start = d.add_input(0, dimer(4.0), energy(4.0)).unwrap();
    let MapperAction::Run(job) = d.ask().unwrap() else {
        panic!("a job was expected");
    };
    let end = |x: f64, molecules: Option<crate::workflow::mapper::Dissociation>| Stationary {
        index_wavenumbers: None,
        system: dimer(x),
        energy: energy(x),
        wavenumbers: None,
        dissociation: molecules,
    };
    // One end apart: a channel (its energy that of the two atoms, each optimized alone); the
    // other end a complex of the same two atoms.
    let outcome = JobOutcome {
        reactions: vec![reaction(
            5.0,
            [
                end(4.0, Some(two_argon_atoms(false, -0.1))),
                end(9.0, Some(two_argon_atoms(true, -0.1))),
            ],
        )],
        ..JobOutcome::default()
    };
    let told = d.tell(&outcome, None).unwrap();
    let net = d.network();
    let channels: Vec<&Node> = net.of_kind(&NodeKind::Dissociated).collect();
    assert_eq!(channels.len(), 1);
    let dc = channels[0];
    assert_eq!(dc.label(), "DC0");
    assert!((dc.energy + 0.2).abs() < 1e-12);
    // One record for both atoms, the same channel from both ends.
    assert_eq!(net.fragments.len(), 1);
    assert_eq!(
        dc.parts.iter().map(|p| p.fragment).collect::<Vec<_>>(),
        [0, 0]
    );
    // The transition state joins the input and the channel, with the barrier from the energy
    // of the channel; the input is a complex with a barrierless edge to the same channel.
    let e = &net.ts_edges[0];
    assert_eq!(e.ends, [start, dc.id]);
    assert!((e.barriers[1] - (energy(5.0) + 0.05 + 0.2)).abs() < 1e-12);
    assert_eq!(net.dissociation_edges.len(), 1);
    let x = &net.dissociation_edges[0];
    assert_eq!((x.minimum, x.channel), (start, dc.id));
    assert!((x.energy_difference - (-0.2 - energy(4.0))).abs() < 1e-12);
    // A channel is not a new minimum, and it is never explored.
    assert!(told.new_minima.is_empty());
    let dc = dc.id;
    assert!(!d.explorable(d.network().node(dc).unwrap()));
    // A lower energy of the same fragment found later lowers the channel and its edges.
    let mut lower = two_argon_atoms(true, -0.15);
    lower.energy = -0.3;
    let MapperAction::Run(_) = d.ask().unwrap() else {
        panic!("a job was expected");
    };
    d.tell(
        &JobOutcome {
            reactions: vec![reaction(6.0, [end(4.0, None), end(10.0, Some(lower))])],
            ..JobOutcome::default()
        },
        None,
    )
    .unwrap();
    let net = d.network();
    assert_eq!(net.of_kind(&NodeKind::Dissociated).count(), 1);
    assert!((net.node(dc).unwrap().energy + 0.3).abs() < 1e-12);
    assert!((net.ts_edges[0].barriers[1] - (energy(5.0) + 0.05 + 0.3)).abs() < 1e-12);
    assert!((net.dissociation_edges[0].energy_difference - (-0.3 - energy(4.0))).abs() < 1e-12);
    // The file keeps the fragments, the parts, and the dissociations.
    let (back, _) = from_json(&to_json(net, JsonValue::Null)).unwrap();
    assert_eq!(back.fragments.len(), 1);
    assert_eq!(back.fragments[0].key, 42);
    assert_eq!(back.nodes[dc].parts, net.nodes[dc].parts);
    assert_eq!(back.nodes[dc].kind, NodeKind::Dissociated);
    assert_eq!(back.dissociation_edges.len(), 1);
    assert_eq!(back.find_channel(Level::Exploration, &[0, 0]), Some(dc));
    let _ = job;
}

#[test]
fn jobs_chosen_together_leave_each_other_out_and_are_told_in_any_order() {
    let mut d = driver(fresh(), 100);
    d.add_input(0, dimer(4.0), energy(4.0)).unwrap();
    // The one minimum has two tasks: a batch of three gets both, each once.
    let MapperBatch::Run(batch) = d.ask_batch(3).unwrap() else {
        panic!("a batch was expected");
    };
    assert_eq!(batch.iter().map(|j| j.id).collect::<Vec<_>>(), vec![0, 1]);
    assert_ne!(batch[0].task.key, batch[1].task.key);
    assert_eq!(d.in_flight().len(), 2);
    assert!(
        d.ask_batch(1).is_err(),
        "nothing to choose while both are in flight"
    );
    assert!(matches!(d.ask_next().unwrap(), NextJob::Wait));
    let jobs = ToyJobs {
        asked: Cell::new(0),
        stop_at: None,
    };
    let mut sink = Vec::<f64>::new();
    struct Energies<'a>(&'a mut Vec<f64>);
    impl FrameSink for Energies<'_> {
        fn push(&mut self, _s: &Configuration, e: f64, _l: &str) -> MoptResult<()> {
            self.0.push(e);
            Ok(())
        }
    }
    let outcomes: Vec<JobOutcome> = batch
        .iter()
        .map(|j| d.run_job(j, &jobs, &mut Energies(&mut sink)).unwrap())
        .collect();
    // The second is told first; the network takes both.
    d.tell_job(1, &outcomes[1], None).unwrap();
    d.tell_job(0, &outcomes[0], None).unwrap();
    assert!(d.tell_job(0, &outcomes[0], None).is_err());
    assert_eq!(d.network().jobs.len(), 2);
    assert_eq!(d.network().minima().count(), 3);
    // A job that was dropped is offered again under its number.
    let MapperAction::Run(next) = d.ask().unwrap() else {
        panic!("a job was expected");
    };
    d.drop_job(next.id);
    let MapperAction::Run(again) = d.ask().unwrap() else {
        panic!("a job was expected");
    };
    assert_eq!((again.id, &again.task.key), (next.id, &next.task.key));
}

/// The network of the toy explored with `schedule` by `workers`.
fn explored(
    schedule: Schedule,
    workers: &ToyWorkers,
    slow: bool,
) -> (MapperDriver, ExplorationEnd) {
    let mut d = driver_with(fresh(), 100, Box::new(Uneven::new(slow)));
    let end = explore_with(&mut d, workers, &schedule, &[dimer(4.0)], &mut []).unwrap();
    (d, end)
}

fn batches_of(batch: usize, workers: usize) -> Schedule {
    Schedule {
        workers,
        batch,
        ..Schedule::default()
    }
}

#[test]
fn a_batch_gives_the_same_network_with_any_number_of_workers() {
    let (one, end) = explored(batches_of(3, 1), &ToyWorkers::new(0), false);
    assert!(!end.interrupted);
    assert_eq!(one.network().jobs.len(), 11);
    let three = ToyWorkers::new(0);
    let (d3, _) = explored(batches_of(3, 3), &three, false);
    assert!(
        three.most.load(Ordering::SeqCst) >= 2,
        "the jobs of a batch ran at the same time"
    );
    let (d2, _) = explored(batches_of(3, 2), &ToyWorkers::new(0), false);
    assert_eq!(text(d3.network()), text(one.network()));
    assert_eq!(text(d2.network()), text(one.network()));
    // Every job has its number in the order of the run.
    let ids: Vec<usize> = d3.network().jobs.iter().map(|j| j.id).collect();
    assert_eq!(ids, (0..11).collect::<Vec<_>>());
    // Rolling: every task is run as well, the jobs kept as they end.
    let (rolling, end) = explored(
        Schedule {
            mode: ScheduleMode::Rolling,
            ..batches_of(1, 3)
        },
        &ToyWorkers::new(0),
        false,
    );
    assert!(!end.interrupted && end.reason.is_some());
    let net = rolling.network();
    assert_eq!(net.jobs.len(), 11);
    let mut lengths: Vec<f64> = net.minima().map(|n| length(&n.system)).collect();
    lengths.sort_by(f64::total_cmp);
    assert_eq!(lengths.len(), 5, "{lengths:?}");
    assert_eq!(net.ts_edges.len(), 7);
}

/// Stops the toy run (the flag of its workers) at a moment of the commit, and keeps the order
/// in which the jobs were told.
struct StopWhen {
    flag: Arc<AtomicBool>,
    after_job: Option<usize>,
    after_batch: Option<usize>,
    told: Vec<usize>,
    batches: Vec<Vec<usize>>,
}

impl StopWhen {
    fn new(workers: &ToyWorkers) -> Self {
        Self {
            flag: Arc::clone(&workers.stop),
            after_job: None,
            after_batch: None,
            told: Vec::new(),
            batches: Vec::new(),
        }
    }
}

impl MapperObserver for StopWhen {
    fn after_job(
        &mut self,
        _network: &Network,
        job: &Job,
        _outcome: &JobOutcome,
        _told: &Told,
    ) -> MoptResult<()> {
        self.told.push(job.id);
        if self.after_job == Some(job.id) {
            self.flag.store(true, Ordering::SeqCst);
        }
        Ok(())
    }

    fn after_batch(&mut self, _network: &Network, jobs: &[usize]) -> MoptResult<()> {
        self.batches.push(jobs.to_vec());
        if self.after_batch.is_some_and(|k| jobs.contains(&k)) {
            self.flag.store(true, Ordering::SeqCst);
        }
        Ok(())
    }
}

/// Continue the toy network `network` (written and read back) with `workers` to its end.
fn continued(
    network: &Network,
    method: Uneven,
    workers: &ToyWorkers,
    schedule: Schedule,
) -> (MapperDriver, ExplorationEnd) {
    let (back, _) = from_json(&JsonValue::parse(&text(network)).unwrap()).unwrap();
    let mut d = driver_with(back, 100, Box::new(method));
    let end = explore_with(&mut d, workers, &schedule, &[dimer(4.0)], &mut []).unwrap();
    (d, end)
}

/// Explore the toy with `workers` and the observer `watch`.
fn watched(
    method: Uneven,
    workers: &ToyWorkers,
    schedule: Schedule,
    watch: &mut StopWhen,
) -> (MapperDriver, ExplorationEnd) {
    let mut d = driver_with(fresh(), 100, Box::new(method));
    let mut observers: [&mut dyn MapperObserver; 1] = [watch];
    let end = explore_with(&mut d, workers, &schedule, &[dimer(4.0)], &mut observers).unwrap();
    (d, end)
}

#[test]
fn a_stop_at_every_phase_of_a_batch_continues_to_the_same_network() {
    // Three jobs at a time in batches of three: the first batch has the two tasks of the
    // input (jobs 0 and 1), the second jobs 2, 3, and 4 (which end after 17, 8, and 14 ms).
    let schedule = batches_of(3, 3);
    let (whole, _) = explored(schedule, &ToyWorkers::new(0), false);
    let goal = text(whole.network());
    let restart = |net: &Network| {
        let (d, end) = continued(net, Uneven::new(false), &ToyWorkers::new(0), schedule);
        assert!(!end.interrupted);
        text(d.network())
    };
    // Before a batch: stopped once the first batch is kept.
    let workers = ToyWorkers::new(0);
    let mut watch = StopWhen::new(&workers);
    watch.after_batch = Some(1);
    let (d, end) = watched(Uneven::new(false), &workers, schedule, &mut watch);
    assert!(end.interrupted);
    assert_eq!(d.network().jobs.len(), 2, "before the second batch");
    assert_eq!(restart(d.network()), goal, "before a batch");
    // With every job of the second batch in flight (stopped when its last one starts).
    let (d, end) = explored(schedule, &ToyWorkers::new(5), false);
    assert!(end.interrupted && end.reason.is_none());
    assert_eq!(d.network().jobs.len(), 2, "nothing of the second batch");
    assert!(d.in_flight().is_empty());
    assert_eq!(restart(d.network()), goal, "all in flight");
    // With one job of the second batch ended (job 3, the third job to end) and two in flight.
    let method = Uneven::new(false);
    let ended = Arc::clone(&method.ended);
    let mut d = driver_with(fresh(), 100, Box::new(method));
    let end = explore_with(
        &mut d,
        &ToyWorkers::stopping(0, 3),
        &schedule,
        &[dimer(4.0)],
        &mut [],
    )
    .unwrap();
    assert!(end.interrupted);
    assert_eq!(d.network().jobs.len(), 2, "nothing of the second batch");
    assert!(
        ended.lock().unwrap().iter().any(|&j| j >= 2),
        "a job of the second batch had ended and is not kept"
    );
    assert_eq!(restart(d.network()), goal, "some ended, some in flight");
    // During the commit of the second batch: the batch is kept whole, and the run stops
    // before the next.
    let workers = ToyWorkers::new(0);
    let mut watch = StopWhen::new(&workers);
    watch.after_job = Some(3);
    let (d, end) = watched(Uneven::new(false), &workers, schedule, &mut watch);
    assert!(end.interrupted);
    assert_eq!(d.network().jobs.len(), 5, "the second batch whole");
    assert_eq!(restart(d.network()), goal, "during the commit");
    // A second stop during the restart, then a restart to the end.
    let (d, _) = explored(schedule, &ToyWorkers::new(5), false);
    let (again, end) = continued(
        d.network(),
        Uneven::new(false),
        &ToyWorkers::stopping(0, 4),
        schedule,
    );
    assert!(end.interrupted);
    assert!(again.network().jobs.len() >= 2);
    assert_eq!(restart(again.network()), goal, "a stop during the restart");
    // One job at a time, stopped within the fourth job: the same as above.
    let one = batches_of(1, 1);
    let (whole_one, _) = explored(one, &ToyWorkers::new(0), false);
    let (d, end) = explored(one, &ToyWorkers::new(4), false);
    assert!(end.interrupted);
    assert_eq!(d.network().jobs.len(), 3);
    let (d, _) = continued(d.network(), Uneven::new(false), &ToyWorkers::new(0), one);
    assert_eq!(text(d.network()), text(whole_one.network()));
}

#[test]
fn jobs_are_told_in_the_order_of_their_numbers_whatever_order_they_end_in() {
    let reverse = || Uneven {
        finish: Finish::Reverse,
        ..Uneven::new(false)
    };
    let method = reverse();
    let ended = Arc::clone(&method.ended);
    let workers = ToyWorkers::new(0);
    let mut watch = StopWhen::new(&workers);
    let (d, _) = watched(method, &workers, batches_of(3, 3), &mut watch);
    let n = d.network().jobs.len();
    assert_eq!(watch.told, (0..n).collect::<Vec<_>>());
    for b in &watch.batches {
        assert!(b.is_sorted_by(|x, y| x < y), "{b:?}");
    }
    let ended = ended.lock().unwrap().clone();
    assert_ne!(
        ended,
        (0..n).collect::<Vec<_>>(),
        "the jobs ended out of order"
    );
    // One worker with the same batches: the same network.
    let mut one = driver_with(fresh(), 100, Box::new(reverse()));
    explore_with(
        &mut one,
        &ToyWorkers::new(0),
        &batches_of(3, 1),
        &[dimer(4.0)],
        &mut [],
    )
    .unwrap();
    assert_eq!(text(d.network()), text(one.network()));
}

#[test]
fn a_job_that_panics_fails_alone() {
    let panicking = || Uneven {
        panics: true,
        ..Uneven::new(false)
    };
    let mut networks = Vec::new();
    for workers in [3, 1] {
        let mut d = driver_with(fresh(), 100, Box::new(panicking()));
        let end = explore_with(
            &mut d,
            &ToyWorkers::new(0),
            &batches_of(3, workers),
            &[dimer(4.0)],
            &mut [],
        )
        .unwrap();
        assert!(!end.interrupted && end.reason.is_some());
        let net = d.network();
        let failed: Vec<&JobRecord> = net
            .jobs
            .iter()
            .filter(|j| matches!(&j.status, JobStatus::Failed(e) if e.contains("panicked")))
            .collect();
        assert_eq!(failed.len(), 1, "{workers}");
        assert_eq!(failed[0].task, "panic");
        // The others ran: every task, the failing one of the minimum at 5 Angstrom too.
        assert_eq!(net.jobs.len(), 12, "{workers}");
        networks.push(text(net));
    }
    assert_eq!(networks[0], networks[1]);
}

#[test]
fn a_stop_while_a_job_reaches_its_time_limit_ends_and_continues() {
    let limited = Schedule {
        time_limit: Some(Duration::from_millis(150)),
        ..batches_of(3, 3)
    };
    let (whole, _) = explored(limited, &ToyWorkers::new(0), true);
    let goal = text(whole.network());
    for at in [100_u64, 150, 200, 250] {
        let workers = ToyWorkers::new(0);
        let flag = Arc::clone(&workers.stop);
        let timer = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(at));
            flag.store(true, Ordering::SeqCst);
        });
        let (d, _) = explored(limited, &workers, true);
        timer.join().unwrap();
        let (d, end) = continued(d.network(), Uneven::new(true), &ToyWorkers::new(0), limited);
        assert!(!end.interrupted);
        assert_eq!(text(d.network()), goal, "stopped after {at} ms");
    }
    // Rolling: stopped, then continued to every task.
    let rolling = Schedule {
        mode: ScheduleMode::Rolling,
        ..limited
    };
    let (d, end) = explored(rolling, &ToyWorkers::new(6), true);
    assert!(end.interrupted);
    assert!(d.in_flight().is_empty());
    let (d, end) = continued(d.network(), Uneven::new(true), &ToyWorkers::new(0), rolling);
    assert!(!end.interrupted);
    assert_eq!(d.network().jobs.len(), 12);
}

#[test]
fn a_job_past_its_time_limit_is_stopped_and_recorded_as_failed() {
    for workers in [1, 2] {
        let limit = Schedule {
            time_limit: Some(Duration::from_millis(60)),
            ..batches_of(workers, workers)
        };
        let (d, end) = explored(limit, &ToyWorkers::new(0), true);
        assert!(!end.interrupted, "{workers}");
        let net = d.network();
        let slow: Vec<&JobRecord> = net.jobs.iter().filter(|j| j.task == "slow").collect();
        assert_eq!(slow.len(), 1, "{workers}");
        assert!(
            matches!(&slow[0].status, JobStatus::Failed(e) if e.contains("worker_timeout_s")),
            "{:?}",
            slow[0].status
        );
        // The other tasks ran to their end.
        assert_eq!(net.jobs.len(), 12, "{workers}");
        assert!(net
            .warnings
            .iter()
            .any(|w| w.message.contains("worker_timeout_s")));
    }
}

#[test]
fn an_irc_side_at_another_transition_state_is_a_ts_ts_link() {
    let mut d = driver(fresh(), 100);
    let start = d.add_input(0, dimer(4.0), energy(4.0)).unwrap();
    let MapperAction::Run(_) = d.ask().unwrap() else {
        panic!("a job was expected");
    };
    let at = |x: f64, e: f64| Stationary {
        index_wavenumbers: None,
        system: dimer(x),
        energy: e,
        wavenumbers: None,
        dissociation: None,
    };
    // TS2 at 6 Angstrom between minima at 5.5 and 6.5 Angstrom; TS1 at 5 Angstrom, above it,
    // between the input and TS2 (its forward side ran down into TS2). The job pushes the
    // reaction of TS2 first.
    let e_ts2 = energy(6.0);
    let ts2 = FoundReaction {
        ts: at(6.0, e_ts2),
        imaginary_wavenumber: Some(-20.0),
        ends: [Some(at(5.5, e_ts2 - 0.05)), Some(at(6.5, e_ts2 - 0.04))],
        branches: Vec::new(),
        saddle_ends: [None, None],
        bifurcation_ends: Vec::new(),
    };
    let ts1 = FoundReaction {
        ts: at(5.0, energy(5.0) + 0.05),
        imaginary_wavenumber: Some(-50.0),
        ends: [Some(at(4.0, energy(4.0))), Some(ts2.ts.clone())],
        branches: Vec::new(),
        saddle_ends: [None, Some(SaddleEnd::Reaction(0))],
        bifurcation_ends: Vec::new(),
    };
    let told = d
        .tell(
            &JobOutcome {
                reactions: vec![ts2, ts1],
                ..JobOutcome::default()
            },
            None,
        )
        .unwrap();
    let net = d.network();
    assert_eq!(net.ts_edges.len(), 2);
    assert_eq!(
        told.new_minima.len(),
        2,
        "the two minima of TS2, not TS2 itself"
    );
    let (e2, e1) = (&net.ts_edges[0], &net.ts_edges[1]);
    assert_eq!(net.nodes[e2.ts].kind, NodeKind::TransitionState);
    // TS1 joins the input and TS2; its barrier from the side of TS2 is the energy between the
    // two transition states.
    assert_eq!(e1.ends, [start, e2.ts]);
    let drop = net.nodes[e1.ts].energy - net.nodes[e2.ts].energy;
    assert!(drop > 0.0 && (e1.barriers[1] - drop).abs() < 1e-12);
    // TS2 leads to its own two minima; the link is in the warnings of the network.
    assert!(e2
        .ends
        .iter()
        .all(|&m| net.nodes[m].kind == NodeKind::Minimum));
    assert!(net
        .warnings
        .iter()
        .any(|w| w.message.contains("TS–TS link")));
    // The file keeps the edge whose end is a transition state.
    let (back, _) = from_json(&to_json(net, JsonValue::Null)).unwrap();
    assert_eq!(back.ts_edges[1].ends, [start, e2.ts]);
    // A reaction whose end is a transition state that gave no reaction is left out.
    let MapperAction::Run(_) = d.ask().unwrap() else {
        panic!("a job was expected");
    };
    let stray = FoundReaction {
        ts: at(7.0, energy(7.0) + 0.05),
        imaginary_wavenumber: Some(-40.0),
        ends: [Some(at(4.0, energy(4.0))), Some(at(7.5, energy(7.5)))],
        branches: Vec::new(),
        saddle_ends: [None, Some(SaddleEnd::Reaction(5))],
        bifurcation_ends: Vec::new(),
    };
    d.tell(
        &JobOutcome {
            reactions: vec![stray],
            ..JobOutcome::default()
        },
        None,
    )
    .unwrap();
    assert_eq!(d.network().ts_edges.len(), 2);
    assert!(d
        .network()
        .warnings
        .iter()
        .any(|w| w.message.contains("left out")));
}

fn toy_jobs() -> ToyJobs {
    ToyJobs {
        asked: Cell::new(0),
        stop_at: None,
    }
}

#[test]
fn minima_whose_bonds_or_species_changed_too_often_are_not_explored() {
    // Two argon atoms 2 Angstrom apart are bonded by the test of the bonds (within 1.2 times
    // their covalent radii), at 3 Angstrom and beyond they are not: the first step of the toy
    // breaks the bond, and no later step changes it again.
    let lengths = |d: &MapperDriver| -> Vec<f64> {
        let mut explored: Vec<f64> = d
            .network()
            .minima()
            .filter(|n| n.excluded.is_none())
            .map(|n| length(&n.system))
            .collect();
        explored.sort_by(f64::total_cmp);
        explored
    };
    let mut free = driver(fresh(), 100);
    explore(&mut free, &toy_jobs(), &[dimer(2.0)], &mut []).unwrap();
    assert_eq!(lengths(&free).len(), 7, "2 to 8 Angstrom");
    for by_species in [false, true] {
        // No change allowed: only the input is explored, and what it reacts to is kept.
        let mut none = driver(fresh(), 100);
        if by_species {
            none.settings.max_species_changes = Some(0);
        } else {
            none.settings.max_bond_changes = Some(0);
        }
        explore(&mut none, &toy_jobs(), &[dimer(2.0)], &mut []).unwrap();
        let net = none.network();
        assert_eq!(lengths(&none).len(), 1, "{by_species}");
        assert_eq!(
            net.minima().count(),
            3,
            "the ends of its two reactions are kept"
        );
        assert_eq!(net.jobs.len(), 2);
        let why = net.nodes[1].excluded.as_deref().unwrap();
        let setting = if by_species {
            "max_species_changes=0"
        } else {
            "max_bond_changes=0"
        };
        assert!(why.contains(setting) && why.contains("1 time(s)"), "{why}");
        // One change allowed: everything of this network is within it.
        let mut one = driver(fresh(), 100);
        if by_species {
            one.settings.max_species_changes = Some(1);
        } else {
            one.settings.max_bond_changes = Some(1);
        }
        explore(&mut one, &toy_jobs(), &[dimer(2.0)], &mut []).unwrap();
        assert_eq!(lengths(&one), lengths(&free));
    }
    // From a dimer that is not bonded nothing changes, and the limit leaves the run alone.
    let mut apart = driver(fresh(), 100);
    apart.settings.max_bond_changes = Some(0);
    explore(&mut apart, &toy_jobs(), &[dimer(4.0)], &mut []).unwrap();
    assert_eq!(lengths(&apart).len(), 5);
}

#[test]
fn path_peaks_are_records_beside_the_network() {
    use crate::workflow::mapper::peak::{FoundPeak, PeakBasis};
    let at = |d: f64, e: f64| Stationary {
        index_wavenumbers: None,
        system: dimer(d),
        energy: e,
        wavenumbers: None,
        dissociation: None,
    };
    let peak = |top: f64, ends: [f64; 2]| FoundPeak {
        point: at(top, -0.2),
        ends: [
            Some(at(ends[0], energy(ends[0]))),
            Some(at(ends[1], energy(ends[1]))),
        ],
        basis: PeakBasis {
            method: "toy path".to_string(),
            points: 9,
            converged: true,
            searches: 2,
        },
        path: Vec::new(),
        settles: [Vec::new(), Vec::new()],
    };
    let mut d = driver(fresh(), 100);
    d.add_input(0, dimer(4.0), energy(4.0)).unwrap();
    let MapperAction::Run(_) = d.ask().unwrap() else {
        panic!("a job was expected");
    };
    let told = d
        .tell(
            &JobOutcome {
                peaks: vec![peak(4.5, [4.0, 5.0])],
                ..JobOutcome::default()
            },
            None,
        )
        .unwrap();
    assert_eq!(told.new_peaks, vec![(0, 0)]);
    let net = d.network();
    // The record: its ends are the start and a new minimum, which is a node like any other.
    assert_eq!(net.path_peaks.len(), 1);
    let p = &net.path_peaks[0];
    assert_eq!((p.label().as_str(), p.ends), ("PP0", [Some(0), Some(1)]));
    assert!((p.barriers[0].unwrap() - (-0.2 - energy(4.0))).abs() < 1e-12);
    assert_eq!(p.job, Some(0));
    assert_eq!(net.minima().count(), 2);
    // No part of the network: no node of its own, no edge.
    assert_eq!(net.nodes.len(), 2);
    assert!(net.ts_edges.is_empty());
    // The file keeps it.
    let (back, _) = from_json(&to_json(net, JsonValue::Null)).unwrap();
    assert_eq!(back.path_peaks.len(), 1);
    let q = &back.path_peaks[0];
    assert_eq!((q.ends, q.job, &q.basis), (p.ends, p.job, &p.basis));
    assert!((q.energy - p.energy).abs() < 1e-15);
    assert!((length(&q.system) - 4.5).abs() < 1e-9);
    // A network without path peaks has no such entry in its file.
    assert!(!text(&fresh()).contains("path_peaks"));
    // The same peak again is not recorded twice, nor one whose two sides are one minimum.
    let MapperAction::Run(_) = d.ask().unwrap() else {
        panic!("a job was expected");
    };
    let told = d
        .tell(
            &JobOutcome {
                peaks: vec![peak(4.5, [4.0, 5.0]), peak(4.2, [4.0, 4.0])],
                ..JobOutcome::default()
            },
            None,
        )
        .unwrap();
    assert!(told.new_peaks.is_empty());
    assert_eq!(d.network().path_peaks.len(), 1);
}

#[test]
fn lines_of_the_user_steer_a_run() {
    use crate::workflow::mapper::steer::STEERED_REASON;
    // Two jobs from the dimer at 4 Angstrom: minima at 4, 5 and 6, and the limit is reached.
    let mut d = driver(fresh(), 100);
    d.set_job_limit(2);
    explore(&mut d, &toy_jobs(), &[dimer(4.0)], &mut []).unwrap();
    assert_eq!(d.network().jobs.len(), 2);
    assert!(d.finished().is_some_and(|r| r.contains("limit")));
    // A line that cannot be read changes nothing.
    assert!(d.steer("faster EQ0").contains("not applied"));
    assert!(d.steer("explore TS0").contains("not applied"));
    assert!(d.steer("# a note").contains("nothing"));
    assert!(d.network().steering.is_empty());
    // The limit of jobs is raised, and the run goes on.
    assert!(d.steer("max_jobs 5").contains('5'));
    assert!(d.finished().is_none());
    // A minimum is skipped and taken back.
    assert!(d.steer("skip EQ1").contains("no more jobs"));
    assert_eq!(
        d.network().nodes[1].excluded.as_deref(),
        Some(STEERED_REASON)
    );
    assert!(d.steer("unskip EQ1").contains("explored again"));
    assert!(d.network().nodes[1].excluded.is_none());
    assert!(d.steer("unskip EQ1").contains("nothing changes"));
    d.steer("skip EQ1");
    let skipped_at = d.network().jobs.len();
    // The open tasks of the minimum that is asked for run next, whatever the priority says.
    assert!(d.steer("explore EQ2").contains("EQ2"));
    let eq2 = d
        .network()
        .nodes
        .iter()
        .position(|n| n.label() == "EQ2")
        .unwrap();
    let open = d.open_tasks(eq2).unwrap().len();
    assert!(open > 0);
    for _ in 0..open {
        let MapperAction::Run(job) = d.ask().unwrap() else {
            panic!("a job was expected");
        };
        assert_eq!(job.node, eq2);
        d.tell(&JobOutcome::default(), None).unwrap();
    }
    // Its tasks are used up: the priority chooses again (the minimum leaves the list at the
    // next choice), and the skipped minimum is left alone up to the end of the run.
    let reason = loop {
        match d.ask().unwrap() {
            MapperAction::Run(job) => {
                assert_ne!(job.node, 1);
                d.tell(&JobOutcome::default(), None).unwrap();
            }
            MapperAction::Finished(reason) => break reason,
        }
    };
    assert!(
        reason.contains("Every task") || reason.contains("limit"),
        "{reason}"
    );
    assert!(d.network().jobs.len() <= 5);
    assert!(d.network().steering.first.is_empty());
    assert!(d.network().jobs[skipped_at..].iter().all(|j| j.node != 1));
    // What was applied is kept, with the jobs told by then, and the file keeps it.
    let applied = &d.network().steering.applied;
    assert_eq!(applied.len(), 6);
    assert_eq!(
        (applied[0].after_jobs, applied[0].line.as_str()),
        (2, "max_jobs 5")
    );
    let (back, _) = from_json(&to_json(d.network(), JsonValue::Null)).unwrap();
    assert_eq!(back.steering, d.network().steering);
    assert_eq!(back.steering.max_jobs, Some(5));
    assert_eq!(back.nodes[1].excluded.as_deref(), Some(STEERED_REASON));
}

#[test]
fn a_transition_state_with_an_unknown_end_is_kept_out_of_the_kinetics() {
    let mut d = driver(fresh(), 100);
    let start = d.add_input(0, dimer(4.0), energy(4.0)).unwrap();
    let MapperAction::Run(_) = d.ask().unwrap() else {
        panic!("a job was expected");
    };
    let end = Stationary {
        index_wavenumbers: None,
        system: dimer(4.0),
        energy: energy(4.0),
        wavenumbers: None,
        dissociation: None,
    };
    // The minimization of the forward end did not converge.
    let mut r = reaction(5.0, [end.clone(), end]);
    r.ends[1] = None;
    let told = d
        .tell(
            &JobOutcome {
                reactions: vec![r],
                ..JobOutcome::default()
            },
            None,
        )
        .unwrap();
    let net = d.network();
    // The transition state is a node with the end that is known, and no edge of the kinetics.
    assert!(net.ts_edges.is_empty());
    assert_eq!(net.open_ts_edges.len(), 1);
    let open = &net.open_ts_edges[0];
    assert_eq!(net.node(open.ts).unwrap().kind, NodeKind::TransitionState);
    assert_eq!(open.ends, [Some(start), None]);
    assert!((open.barriers[0].unwrap() - 0.05 - energy(5.0) + energy(4.0)).abs() < 1e-12);
    assert!(open.barriers[1].is_none());
    assert_eq!(told.new_open_edges, vec![(0, 0)]);
    assert!(told.new_edges.is_empty());
    assert!(net
        .warnings
        .iter()
        .any(|w| w.message.contains("is left out of the kinetics")));
    let kinetic = crate::workflow::kinetics::kinetic_network(net, Level::Exploration);
    assert!(kinetic.bottlenecks.is_empty());
    assert!(kinetic
        .notes
        .iter()
        .any(|n| n.contains("TS0 is left out: an end of its IRC is not known")));
    // The network file keeps it; a file without such transition states has no key for them.
    let doc = to_json(net, JsonValue::Null);
    let (back, _) = from_json(&doc).unwrap();
    assert_eq!(back.open_ts_edges, net.open_ts_edges);
    assert!(!text(&fresh()).contains("open_ts_edges"));
}

/// A transition state at 4.2 Angstrom whose IRC runs to 4.5 (backward) and 3.9 Angstrom
/// (forward). The backward endpoint can fail to converge or have an imaginary mode.
struct EndpointJobs<'a> {
    backward_frequency: Option<&'a [f64]>,
}

impl MapperJobs for EndpointJobs<'_> {
    fn optimize(
        &self,
        start: &Configuration,
        request: &OptimizeRequest,
        _frames: &mut dyn FrameSink,
    ) -> MoptResult<Optimized> {
        // A fragment of a channel (one atom) has no distance.
        let pair = start.structure().is_some_and(|s| s.n_atoms() == 2);
        let d = if pair { length(start) } else { 0.0 };
        let backward =
            request.stage == crate::workflow::mapper::Stage::IrcEnd && (d - 4.5).abs() < 1e-8;
        Ok(Optimized {
            index_wavenumbers: self
                .backward_frequency
                .filter(|_| backward)
                .map(|values| values.iter().copied().filter(|&w| w < 0.0).collect()),
            system: start.clone(),
            energy: if pair { energy(d) } else { 0.0 },
            converged: !backward || self.backward_frequency.is_some(),
            interrupted: false,
            path: Vec::new(),
            // A stable atomic block can coexist with imaginary strain modes.
            wavenumbers: Some(vec![120.0]),
        })
    }

    fn irc(&self, ts: &Configuration, _frames: &mut dyn FrameSink) -> MoptResult<IrcOutcome> {
        let branch = |name: &str, d: f64| crate::workflow::mapper::method::IrcBranch {
            side: crate::workflow::mapper::network::IrcSide {
                name: name.to_string(),
                arc_lengths: vec![0.1, 0.3],
                energies: vec![energy(4.2) - 0.001, energy(d)],
                end: "max_steps".to_string(),
                transverse: Vec::new(),
            },
            frames: vec![dimer(4.2), dimer(d)],
            last: dimer(d),
            settle: Vec::new(),
            possible_bifurcation: None,
        };
        Ok(IrcOutcome {
            interrupted: false,
            transition_state: true,
            imaginary_modes: 1,
            imaginary_wavenumber: Some(-30.0),
            wavenumbers: None,
            branches: if length(ts) > 4.4 {
                vec![branch("backward", 4.8), branch("forward", 5.1)]
            } else {
                vec![branch("backward", 4.5), branch("forward", 3.9)]
            },
        })
    }

    fn stop_requested(&self) -> bool {
        false
    }

    fn note(&self, _message: &str) {}
}

fn follow_test_endpoint(backward_frequency: Option<&[f64]>) -> (MapperDriver, JobOutcome) {
    let mut d = driver(fresh(), 100);
    d.add_input(0, dimer(3.9), energy(3.9)).unwrap();
    let MapperAction::Run(job) = d.ask().unwrap() else {
        panic!("a job was expected");
    };
    let matcher = mapper_matcher(0.125, 0.05, false);
    let mut frames = crate::workflow::mapper::dissociation::Discard;
    let mut outcome = JobOutcome::default();
    {
        let mut ctx = JobContext {
            jobs: &EndpointJobs { backward_frequency },
            network: d.network(),
            matcher: &matcher,
            frames: &mut frames,
            state: None,
        };
        let guesses = [crate::workflow::mapper::follow::Guess {
            system: dimer(4.2),
            origin: "a test".to_string(),
        }];
        assert!(crate::workflow::mapper::follow::follow_guesses(
            &job,
            &mut ctx,
            &guesses,
            &mut outcome
        )
        .unwrap());
    }
    (d, outcome)
}

#[test]
fn irc_endpoint_vibrations_select_equilibria_and_linked_transition_states() {
    for w in [-49.99, -50.0, -50.01] {
        let (mut d, outcome) = follow_test_endpoint(Some(&[120.0, w]));
        d.tell(&outcome, None).unwrap();
        let net = d.network();
        if w > -50.0 {
            assert_eq!(net.ts_edges.len(), 1);
            assert!(net
                .warnings
                .iter()
                .any(|w| w.message.contains("weak imaginary mode")));
            assert!(net.minima().any(|m| m
                .index_wavenumbers
                .as_ref()
                .is_some_and(|modes| modes.contains(&w))));
        } else {
            assert_eq!(net.ts_edges.len(), 2);
            assert!(net.ts_edges.iter().any(|edge| edge
                .ends
                .iter()
                .any(|&id| net.nodes[id].kind == NodeKind::TransitionState)));
        }
    }
}

#[test]
fn multiple_imaginary_modes_never_make_an_equilibrium() {
    for pair in [[-1.0, -2.0], [-0.1, -50.0], [-51.0, -80.0]] {
        let (mut d, outcome) = follow_test_endpoint(Some(&pair));
        d.tell(&outcome, None).unwrap();
        let net = d.network();
        assert!(net.ts_edges.is_empty());
        assert_eq!(net.open_ts_edges.len(), 1);
        assert_eq!(net.minima().count(), 1);
        assert_eq!(net.open_ts_edges[0].ends, [None, net.inputs[0].node]);
        let (restored, _) = from_json(&to_json(net, JsonValue::Null)).unwrap();
        assert_eq!(restored.open_ts_edges, net.open_ts_edges);
        assert!(
            crate::workflow::kinetics::kinetic_network(&restored, Level::Exploration)
                .bottlenecks
                .is_empty()
        );
        assert!(net
            .warnings
            .iter()
            .any(|w| w.message.contains("2 imaginary modes")));
    }
}

#[test]
fn an_unconverged_end_is_unknown_and_the_other_side_is_still_followed() {
    let (mut d, outcome) = follow_test_endpoint(None);
    // The backward end (first) did not converge; the forward end was minimized all the same.
    assert_eq!(outcome.reactions.len(), 1);
    let r = &outcome.reactions[0];
    assert!(r.ends[0].is_none());
    let forward = r.ends[1].as_ref().unwrap();
    assert!((length(&forward.system) - 3.9).abs() < 1e-9);
    assert!(outcome
        .warnings
        .iter()
        .any(|w| w.contains("backward end") && w.contains("that end is unknown")));
    // In the network: the transition state with the forward end, apart from the TS edges.
    d.tell(&outcome, None).unwrap();
    let net = d.network();
    assert!(net.ts_edges.is_empty());
    assert_eq!(net.open_ts_edges.len(), 1);
    assert!(net.open_ts_edges[0].ends[0].is_none());
    assert!(net.open_ts_edges[0].ends[1].is_some());
}

#[test]
fn another_environment_of_a_core_is_listed_and_the_lowest_is_the_node() {
    let mut d = driver(fresh(), 100);
    d.settings_mut().environments = true;
    d.settings_mut().matcher.energy_tolerance = crate::domain::Quantity::new(f64::INFINITY);
    let start = d.add_input(0, dimer(4.0), energy(4.0)).unwrap();
    // The whole structure of a core: the dimer with a third atom far away (an environment).
    let whole = |x: f64| -> Configuration {
        let argon = Element::from_atomic_number(18).unwrap();
        MolecularSystem::new(
            vec![argon; 3],
            &positions::<Angstrom>(&[[0.0, 0.0, 0.0], [x, 0.0, 0.0], [0.0, 9.0, 0.0]]),
            SpinCharge::default(),
            None,
        )
        .unwrap()
        .into()
    };
    let end = |x: f64, e: f64| Stationary {
        index_wavenumbers: None,
        system: dimer(x),
        energy: e,
        wavenumbers: None,
        dissociation: None,
    };
    let tell = |d: &mut MapperDriver, back: f64, e_back: f64| {
        let MapperAction::Run(_) = d.ask().unwrap() else {
            panic!("a job was expected");
        };
        let ends = [end(back, e_back), end(6.0, energy(6.0))];
        let outcome = JobOutcome {
            reactions: vec![reaction(5.0, ends)],
            wholes: vec![(dimer(back), whole(back)), (dimer(6.0), whole(6.0))],
            ..JobOutcome::default()
        };
        d.tell(&outcome, None).unwrap()
    };
    // The same core found again 0.01 Hartree higher: listed on the node.
    let told = tell(&mut d, 4.0, energy(4.0) + 0.01);
    assert!(told.lowered.is_empty());
    let net = d.network();
    assert_eq!(net.nodes[start].environments.len(), 1);
    assert!((net.nodes[start].environments[0].energy - energy(4.0) - 0.01).abs() < 1e-12);
    // The new minimum has its whole structure from the outcome.
    let other = net.minima().find(|n| n.id != start).map(|n| n.id).unwrap();
    assert_eq!(
        net.nodes[other]
            .whole
            .as_ref()
            .and_then(Configuration::structure)
            .map(MolecularSystem::n_atoms),
        Some(3)
    );
    // Found again 0.02 Hartree lower: it becomes the node, the old one is listed.
    d.network_mut().nodes[start].frequencies = Some(vec![40.0]);
    d.network_mut().nodes[start].index_wavenumbers = Some(vec![-1.2]);
    d.network_mut().nodes[start].free_energy = Some(-0.1);
    let told = tell(&mut d, 4.0, energy(4.0) - 0.02);
    assert_eq!(told.lowered, vec![start]);
    let n = &d.network().nodes[start];
    assert!((n.energy - energy(4.0) + 0.02).abs() < 1e-12);
    assert_eq!(n.environments.len(), 2);
    assert!(n.frequencies.is_none());
    assert!(n.index_wavenumbers.is_none());
    assert!(n.free_energy.is_none());
    assert!(n.whole.is_some());
    // The network file keeps the whole structures and the environments.
    let mut net = d.network().clone();
    net.whole_elements = Some(vec![Element::from_atomic_number(18).unwrap(); 3]);
    let (back, _) = from_json(&to_json(&net, JsonValue::Null)).unwrap();
    assert_eq!(
        back.nodes[start].environments,
        net.nodes[start].environments
    );
    assert_eq!(
        back.nodes[start]
            .whole
            .as_ref()
            .map(Configuration::flat)
            .map(|x| x.nrows()),
        Some(9)
    );
    assert_eq!(back.whole_elements, net.whole_elements);
}
