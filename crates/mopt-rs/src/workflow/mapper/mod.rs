//! The reaction-network mapper (`run_mapper`): from one or more minima, jobs that each start
//! from a known minimum and look for reactions out of it (by default an AFIR push between two
//! atoms, the transition states on its path, and their IRCs), until no minimum has a task left
//! or a stop rule ends the search.
//!
//! The parts, each a trait with a named registry so that users can add their own from Rust:
//! - how a node is explored: [`method::ExplorationMethod`] ([`method::MethodRegistry`]; built
//!   in: [`afir::AfirTs`], `afir_ts`; [`addf::AddfMap`], `addf`; [`nanoreactor::Nanoreactor`],
//!   `nanoreactor`), the guesses of every method followed alike ([`follow`]); methods of one's
//!   own that only propose guesses and paths ([`plugin::Proposer`], the methods of Python
//!   classes among them);
//! - dissociation channels: fragments apart at an end of an IRC, and the channels of complexes
//!   ([`channels`], [`dissociation`]);
//! - which node is explored next: [`priority::ExplorationPriority`]
//!   ([`priority::ExplorationPriorityRegistry`]; Boltzmann by default, every rule of
//!   [`crate::workflow::explore::PriorityRegistry`], `kinetics` from the kinetics of the
//!   network, and `rrt` towards a goal structure, [`goal`]);
//! - the gaps between a start and a goal filled by two-point methods ([`connect::Connect`],
//!   `connect`, the missing connections);
//! - the crossing points between the electronic states of a run (`states`, PES1, PES2, ...):
//!   [`crossing::WithCrossings`];
//! - when two structures are the same: [`crate::workflow::explore::identity`];
//! - when the search ends: [`stop::StopRuleRegistry`];
//! - where the network is kept: [`store::NetworkStore`] ([`store::StoreRegistry`]; `files`,
//!   and `sqlite` ([`sqlite_store`]): the network as an SQLite database).
//!
//! The decisions are those of the ask/tell state machine [`driver::MapperDriver`]; the loop
//! [`explore::explore`] runs its jobs with the calculations of a [`method::MapperJobs`] (and
//! [`explore::explore_with`] several at the same time, as a [`schedule::Schedule`] says) and
//! tells the [`observers::MapperObserver`]s (the trajectories, the IRC paths, the stores, the
//! report of the command) what happens. The network is [`network::Network`], written as JSON
//! by [`network_json`].
//!
//! MultiOptPy: `Wrapper/mapper.py` (`ReactionNetworkMapper` and its parts).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

pub mod addf;
pub mod afir;
pub mod channels;
pub mod channels_json;
pub mod configuration;
pub mod connect;
pub mod crossing;
pub mod dissociation;
pub mod driver;
pub mod entrance;
pub mod explore;
pub mod extras_json;
pub mod follow;
pub mod goal;
pub mod merge;
pub mod method;
pub mod nanoreactor;
pub mod network;
pub mod network_json;
pub mod observers;
pub mod peak;
pub mod plugin;
pub mod priority;
pub mod remap;
pub mod saddle_json;
pub mod scan;
pub mod schedule;
pub mod sqlite_store;
pub mod steer;
pub mod stop;
pub mod store;
pub mod ts_conformers;
pub mod upward;

pub use addf::AddfMap;
pub use afir::{AfirSettings, AfirTs};
pub use channels::{ChannelPart, ChannelSettings, DissociationEdge, FragmentRecord};
pub use configuration::Configuration;
pub use connect::{Connect, ConnectSettings};
pub use crossing::{
    crossing_kind, refine_crossing, state_index, surface_name, CandidateFilter, CrossingAfir,
    CrossingPairs, FragmentPush, WithCrossings,
};
pub use dissociation::{dissociation_of, Dissociation};
pub use driver::{DriverSettings, MapperAction, MapperBatch, MapperDriver, NextJob, Told};
pub use entrance::{entrance_structures, Entrance, EntranceSettings};
pub use explore::{explore, explore_with};
pub use follow::{follow_guesses, follow_saddle_point, Guess};
pub use goal::{RrtPriority, RrtSettings};
pub use method::{
    CrossingPath, CrossingSearch, Descent, DescentEnd, DynamicsEnd, DynamicsOutcome,
    DynamicsRequest, ExplorationMethod, FollowedRoots, FoundCrossing, FoundReaction, FoundSaddle,
    FrameSink, HigherSaddleSettings, HisdOutcome, IrcBranch, IrcOutcome, Job, JobContext,
    JobOutcome, MapperJobs, MetaIrc, MethodRegistry, MethodSettings, OptimizeRequest, Optimized,
    OrderedSaddle, PairSearchOutcome, PairSearchRequest, PathPoint, PointMode, PointModes,
    SaddleEnd, Stage, Stationary, Task, TsConformerSearch,
};
pub use nanoreactor::{Nanoreactor, NanoreactorSettings};
pub use network::{
    mapper_matcher, seam_of, BiasEnergy, CrossingEdge, CrossingKind, ElectronicState, HeldAtoms,
    InputRecord, IrcPath, IrcSide, JobRecord, JobStatus, Level, Network, NetworkWarning, Node,
    NodeKind, OpenTsEdge, Origin, PossibleBifurcation, SaddleEdge, SeamCheck, TsEdge,
    DEFAULT_BIFURCATION_THRESHOLD_CM1,
};
pub use observers::{
    ExplorationEnd, MapperObserver, PathFiles, Source, StoreObserver, TrajectoryFiles,
};
pub use peak::{keep_path_peak, FoundPeak, PathPeak, PeakBasis, PeakMode};
pub use priority::{
    Candidate, ExplorationPriority, ExplorationPriorityRegistry, KineticChoice, PrioritySettings,
    SamplingAdapter,
};
pub use remap::{Remap, RemapItem, RemapKind, RemapSettings, RemapSource};
pub use scan::{DissociationScan, ScanSettings};
pub use schedule::{MapperWorkers, Schedule, ScheduleMode};
pub use steer::{SteerCommand, Steering};
pub use stop::StopRuleRegistry;
pub use store::{FileStore, NetworkStore, StoreRegistry};
pub use ts_conformers::{WithTsConformers, TS_CONFORMER_PREFIX};
pub use upward::{Upward, UpwardSettings, WithUpward};
