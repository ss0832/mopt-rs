//! The IRC calculation as an ask/tell state machine: the starting structure with its
//! Hessian, then each side with its integrator.
//!
//! ```text
//! loop {
//!     match irc.ask()? {
//!         IrcAction::Evaluate(request) => irc.tell(surface_at(&request))?,
//!         IrcAction::Finished(end) => break,
//!     }
//!     for event in irc.drain_events() { /* write files, log */ }
//! }
//! ```
//!
//! The machine evaluates nothing and writes nothing. It is deterministic: the same values
//! told in the same order give the same requests, which is what the restart from the
//! journal of evaluations relies on.
//!
//! MultiOptPy: `irc.py:IRC` (`run`, `calc_IRCpath`, `_run_irc`, `_run_meta_irc`,
//! `_run_single_irc`).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::collections::VecDeque;
use std::sync::Arc;

use faer::Col;

use super::first_step::{FirstStep, FirstStepReport, Origin};
use super::integrator::{BranchAction, BranchStart, Integrator};
use super::registry::IntegratorRegistry;
use super::settings::IrcSettings;
use super::space::IrcSpace;
use super::start::{examine, StartKind, StartReport};
use super::types::{
    Branch, BranchEnd, PointRequest, PointRole, PointValue, StepCounts, StepRecord,
};
use crate::error::{MoptError, MoptResult};

/// A point to evaluate, and the side it belongs to.
#[derive(Debug, Clone)]
pub struct IrcRequest {
    /// The point.
    pub point: PointRequest,
    /// The side (`None` for the starting structure).
    pub branch: Option<Branch>,
}

/// How a side ended.
#[derive(Debug, Clone)]
pub struct BranchSummary {
    /// The side.
    pub branch: Branch,
    /// Why it ended.
    pub end: BranchEnd,
    /// Points of the path that were reported.
    pub points: usize,
    /// Its last structure.
    pub x: Col<f64>,
    /// What the step control did along it.
    pub counts: StepCounts,
}

/// The end of the calculation.
#[derive(Debug, Clone)]
pub struct IrcEnd {
    /// What was done from the starting structure.
    pub kind: StartKind,
    /// The sides in the order they were run.
    pub branches: Vec<BranchSummary>,
}

/// Next action of an [`Irc`].
#[derive(Debug, Clone)]
pub enum IrcAction {
    /// Evaluate the surface at a point.
    Evaluate(IrcRequest),
    /// The calculation ended.
    Finished(IrcEnd),
}

/// What happened since the events were last taken.
#[derive(Debug, Clone)]
pub enum IrcEvent {
    /// The starting structure was examined.
    Start(StartReport),
    /// The first point of a side was tested (and corrected).
    FirstStep(FirstStepReport),
    /// A point of the path.
    Step {
        /// The side.
        branch: Branch,
        /// The point.
        record: StepRecord,
    },
    /// A side ended.
    BranchFinished(BranchSummary),
}

struct Running {
    branch: Branch,
    integrator: Box<dyn Integrator>,
    points: usize,
}

enum Phase {
    Start,
    FirstStep(Box<FirstStep>),
    Run(Running),
    Done,
}

/// An IRC calculation.
pub struct Irc {
    settings: IrcSettings,
    space: Arc<IrcSpace>,
    registry: IntegratorRegistry,
    x_start: Col<f64>,
    phase: Phase,
    pending: Option<IrcRequest>,
    events: Vec<IrcEvent>,
    /// A point whose surface is known already (the first point after its test).
    known: Option<(Col<f64>, PointValue)>,
    kind: Option<StartKind>,
    hessian: Option<faer::Mat<f64>>,
    origin: Option<Arc<Origin>>,
    queue: VecDeque<(Branch, Col<f64>)>,
    finished: Vec<BranchSummary>,
    evaluations: usize,
    /// Energy of the surface that is followed at the starting structure.
    start_energy: Option<f64>,
}

impl Irc {
    /// A calculation from the structure `x` (a transition state, or any structure for a
    /// meta-IRC or ModeKill) with the built-in integrators.
    pub fn new(settings: IrcSettings, space: IrcSpace, x: Col<f64>) -> MoptResult<Self> {
        Self::with_registry(settings, space, x, IntegratorRegistry::builtin())
    }

    /// [`Self::new`] with the integrators of `registry`.
    pub fn with_registry(
        settings: IrcSettings,
        space: IrcSpace,
        x: Col<f64>,
        registry: IntegratorRegistry,
    ) -> MoptResult<Self> {
        let mut settings = settings.validated()?;
        settings.method = registry.resolve(&settings.method)?;
        space.check(&x)?;
        if x.nrows() == 0 || x.iter().any(|v| !v.is_finite()) {
            return Err(MoptError::config(
                "IRC: the starting structure is empty or not finite",
            ));
        }
        // Every point of the path has its centre of mass at the origin (where rigid-body
        // motion is projected out); so has the start, so that all structures are in one
        // frame. MultiOptPy leaves the start where the input has it.
        let x = space.placed(&x);
        if let Some(c) = space.constraints() {
            let off = c.largest_residual(&x)?;
            if off > 1e-6 {
                return Err(MoptError::config(format!(
                    "IRC: the starting structure cannot be brought onto the constraints (largest deviation {off:.3e})"
                )));
            }
        }
        Ok(Self {
            settings,
            space: Arc::new(space),
            registry,
            x_start: x,
            phase: Phase::Start,
            pending: None,
            events: Vec::new(),
            known: None,
            kind: None,
            hessian: None,
            origin: None,
            queue: VecDeque::new(),
            finished: Vec::new(),
            evaluations: 0,
            start_energy: None,
        })
    }

    /// The settings in effect (the method by its registered name).
    pub fn settings(&self) -> &IrcSettings {
        &self.settings
    }

    /// The variables and their masses.
    pub fn space(&self) -> &IrcSpace {
        &self.space
    }

    /// The starting structure (with its centre of mass at the origin where rigid-body
    /// motion is projected out).
    pub fn start(&self) -> &Col<f64> {
        &self.x_start
    }

    /// What is done from the starting structure (known once it was evaluated).
    pub fn kind(&self) -> Option<StartKind> {
        self.kind
    }

    /// Values told so far.
    pub fn evaluations(&self) -> usize {
        self.evaluations
    }

    /// The sides that ended so far.
    pub fn finished(&self) -> &[BranchSummary] {
        &self.finished
    }

    /// What happened since the last call.
    pub fn drain_events(&mut self) -> Vec<IrcEvent> {
        std::mem::take(&mut self.events)
    }

    /// The next point to evaluate, or the end.
    pub fn ask(&mut self) -> MoptResult<IrcAction> {
        loop {
            if let Some(request) = &self.pending {
                return Ok(IrcAction::Evaluate(request.clone()));
            }
            let from_integrator = match &self.phase {
                Phase::Start => {
                    self.pending = Some(IrcRequest {
                        point: PointRequest {
                            x: self.x_start.clone(),
                            need_hessian: true,
                            role: PointRole::TransitionState,
                            step: 0,
                        },
                        branch: None,
                    });
                    continue;
                }
                Phase::FirstStep(first) => {
                    self.pending = Some(IrcRequest {
                        point: PointRequest {
                            x: first.point().clone(),
                            need_hessian: false,
                            role: PointRole::FirstStep,
                            step: 0,
                        },
                        branch: self.queue.front().map(|(branch, _)| *branch),
                    });
                    continue;
                }
                Phase::Run(running) => (running.branch, running.integrator.ask()),
                Phase::Done => {
                    return Ok(IrcAction::Finished(IrcEnd {
                        kind: self.kind.unwrap_or(StartKind::MetaIrc),
                        branches: self.finished.clone(),
                    }))
                }
            };
            match from_integrator {
                (branch, BranchAction::Evaluate(point)) => {
                    let known = self.known.take_if(|(x, v)| {
                        *x == point.x && (!point.need_hessian || v.hessian.is_some())
                    });
                    match known {
                        Some((_, value)) => self.feed(&value)?,
                        None => {
                            self.known = None;
                            self.pending = Some(IrcRequest {
                                point,
                                branch: Some(branch),
                            });
                        }
                    }
                }
                (_, BranchAction::Finished(end)) => {
                    self.finish_side(end);
                    self.next_side()?;
                }
            }
        }
    }

    /// The surface at the point last asked for.
    pub fn tell(&mut self, value: PointValue) -> MoptResult<()> {
        let Some(asked) = self.pending.take() else {
            return Err(MoptError::config(
                "IRC: a value was told although no point was asked for",
            ));
        };
        // Within the constraints, the surface is the one on them.
        let value = self.space.constrained(&asked.point.x, value)?;
        self.evaluations += 1;
        match std::mem::replace(&mut self.phase, Phase::Done) {
            Phase::Start => self.begin(&value),
            Phase::FirstStep(mut first) => {
                let branch = match self.queue.front() {
                    Some((branch, _)) => *branch,
                    None => return Err(MoptError::config("IRC: no side is waiting")),
                };
                match first.tell(value)? {
                    None => {
                        self.phase = Phase::FirstStep(first);
                        Ok(())
                    }
                    Some(result) => {
                        self.events.push(IrcEvent::FirstStep(result.report));
                        self.queue.pop_front();
                        self.known = Some((result.x.clone(), result.value));
                        self.run(branch, result.x)
                    }
                }
            }
            Phase::Run(running) => {
                self.phase = Phase::Run(running);
                self.feed(&value)
            }
            Phase::Done => Err(MoptError::config("IRC: the calculation has ended")),
        }
    }

    /// The starting structure was evaluated: what is to be done from it.
    fn begin(&mut self, value: &PointValue) -> MoptResult<()> {
        if !value.is_finite() || value.gradient.nrows() != self.space.dimension() {
            return Err(MoptError::calc(
                "IRC: the surface at the starting structure is not finite or of another size",
            ));
        }
        self.start_energy = Some(value.energy);
        let modekill = self.settings.method == "modekill";
        let start = examine(&self.space, &self.settings, modekill, &self.x_start, value)?;
        self.kind = Some(start.report.kind);
        if let Some(eigenvalue) = start.report.transition_eigenvalue {
            self.origin = Some(Arc::new(Origin {
                q: self.space.to_mw(&self.x_start),
                energy: value.energy,
                gradient: self.space.gradient_to_mw(&value.gradient),
                eigenvalue,
                total_hessian: start.total_hessian,
            }));
        }
        self.hessian = Some(start.hessian);
        self.queue = start.sides.into_iter().collect();
        self.events.push(IrcEvent::Start(start.report));
        self.next_side()
    }

    /// Start the next side that waits, or end.
    fn next_side(&mut self) -> MoptResult<()> {
        let Some((branch, x)) = self.queue.front().cloned() else {
            self.phase = Phase::Done;
            return Ok(());
        };
        match &self.origin {
            Some(origin) if self.settings.first_step.check => {
                self.phase = Phase::FirstStep(Box::new(FirstStep::new(
                    self.space.clone(),
                    self.settings.first_step,
                    branch,
                    origin.clone(),
                    x,
                )));
                Ok(())
            }
            _ => {
                self.queue.pop_front();
                self.run(branch, x)
            }
        }
    }

    /// Run the integrator of `branch` from `x`.
    fn run(&mut self, branch: Branch, x: Col<f64>) -> MoptResult<()> {
        let hessian = self
            .hessian
            .clone()
            .ok_or_else(|| MoptError::config("IRC: the starting structure was not evaluated"))?;
        let integrator = self.registry.create(
            &self.settings.method,
            BranchStart {
                x,
                hessian,
                settings: self.settings.clone(),
                space: self.space.clone(),
                energy: self.start_energy,
            },
        )?;
        self.phase = Phase::Run(Running {
            branch,
            integrator,
            points: 0,
        });
        Ok(())
    }

    /// Hand a value to the integrator that runs.
    fn feed(&mut self, value: &PointValue) -> MoptResult<()> {
        let Phase::Run(running) = &mut self.phase else {
            return Err(MoptError::config("IRC: no side is running"));
        };
        if let Some(record) = running.integrator.tell(value)? {
            running.points += 1;
            self.events.push(IrcEvent::Step {
                branch: running.branch,
                record,
            });
        }
        Ok(())
    }

    fn finish_side(&mut self, end: BranchEnd) {
        if let Phase::Run(running) = &self.phase {
            let summary = BranchSummary {
                branch: running.branch,
                end,
                points: running.points,
                x: running.integrator.current().clone(),
                counts: running.integrator.counts(),
            };
            self.finished.push(summary.clone());
            self.events.push(IrcEvent::BranchFinished(summary));
        }
    }
}
