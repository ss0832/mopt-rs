//! The electronic states of the driver (`states`, PES1, PES2, ...): the state of a node and of
//! an input, and the crossing points a job found with the minima of their descents
//! ([`super::super::crossing`]).
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use super::super::configuration::Configuration;
use super::super::method::{Job, JobOutcome};
use super::super::network::{
    CrossingEdge, ElectronicState, IrcPath, IrcSide, Level, Node, NodeKind, Origin,
};
use super::{MapperDriver, Told};
use crate::error::{MoptError, MoptResult};

impl MapperDriver {
    /// The state of index `on` for a node at `system`: that of the states of the run, or for a
    /// run of one state the charge and multiplicity of the structure with the excited state of
    /// the run.
    pub(super) fn state_for(&self, on: usize, system: &Configuration) -> ElectronicState {
        self.network
            .states
            .get(on)
            .copied()
            .unwrap_or_else(|| ElectronicState::of_point(system, self.settings.excited_state))
    }

    /// The state of the input `input` at `system`: for a run of several states the state it
    /// was given with ([`DriverSettings::input_states`]), or the first of them with its charge
    /// and multiplicity (an input on none of them is refused), else as [`Self::state_for`].
    pub(super) fn input_state(
        &self,
        input: usize,
        system: &Configuration,
    ) -> MoptResult<ElectronicState> {
        let Some(first) = self.network.states.first() else {
            return Ok(self.state_for(0, system));
        };
        if let Some(Some(on)) = self.settings.input_states.get(input) {
            return self.network.states.get(*on).copied().ok_or_else(|| {
                MoptError::config(format!(
                    "input {} is on state {} of a run of {} states",
                    input + 1,
                    on + 1,
                    self.network.states.len()
                ))
            });
        }
        let Some(sys) = system.structure() else {
            return Ok(*first);
        };
        let (charge, multiplicity) = (sys.spin_charge.charge, sys.spin_charge.multiplicity);
        self.network
            .states
            .iter()
            .copied()
            .find(|s| s.charge == charge && s.multiplicity == multiplicity)
            .ok_or_else(|| {
                MoptError::config(format!(
                    "an input of charge {charge} and multiplicity {multiplicity} lies on none of \
                     the states of the run (states)"
                ))
            })
    }

    /// The index among the states of the run of the state of the input `input` at `system`
    /// (the state it was given with, or the first of its charge and multiplicity): the state
    /// it is minimized on. `None` for a run of one state.
    pub fn input_state_index(
        &self,
        input: usize,
        system: &Configuration,
    ) -> MoptResult<Option<usize>> {
        if self.network.states.is_empty() {
            return Ok(None);
        }
        let state = self.input_state(input, system)?;
        Ok(self.network.states.iter().position(|s| *s == state))
    }

    /// The crossing points of `outcome`: each the known crossing point of its kind and states
    /// or a new node (`MECP`, `MECI`), the minima of its descents on each state, and the
    /// crossing edge between them (a known edge with the same ends is not added again).
    pub(super) fn tell_crossings(
        &mut self,
        job: &Job,
        outcome: &JobOutcome,
        told: &mut Told,
        new_nodes: &mut Vec<usize>,
    ) -> MoptResult<()> {
        let matcher = self.settings.matcher;
        for (index, c) in outcome.crossings.iter().enumerate() {
            let state = |i: usize| {
                self.network.states.get(i).copied().ok_or_else(|| {
                    MoptError::config(format!(
                        "job {}: a crossing point with state {} of a run of {} states",
                        job.id,
                        i + 1,
                        self.network.states.len()
                    ))
                })
            };
            let states = [state(c.states[0])?, state(c.states[1])?];
            let kind = NodeKind::Crossing {
                kind: c.kind,
                states,
            };
            let crossing = match self.network.find_same(
                &kind,
                Level::Exploration,
                &c.point.system,
                c.point.energy,
                &matcher,
            )? {
                Some(id) => id,
                None => {
                    let input = self.network.node(job.node)?.origin.input;
                    let id = self.network.add_node(Node::new(
                        kind,
                        Level::Exploration,
                        c.point.system.clone(),
                        c.point.energy,
                        states[0],
                        Origin {
                            input,
                            job: Some(job.id),
                            from_node: Some(job.node),
                        },
                    ))?;
                    new_nodes.push(id);
                    id
                }
            };
            let mut ends = [None, None];
            for (side, end) in c.ends.iter().enumerate() {
                if let Some(end) = end {
                    ends[side] =
                        Some(self.minimum_on(job, end, c.states[side], &mut told.new_minima)?);
                }
            }
            if self
                .network
                .crossing_edges
                .iter()
                .any(|e| e.level == Level::Exploration && e.crossing == crossing && e.ends == ends)
            {
                continue;
            }
            let at = self.network.node(crossing)?.energy;
            let mut barriers = [None, None];
            for (side, end) in ends.iter().enumerate() {
                if let Some(id) = end {
                    barriers[side] = Some(at - self.network.node(*id)?.energy);
                }
            }
            // The meta-IRC of each state that has one, and the file of the structures of the
            // whole path between the two minima.
            let sides: Vec<IrcSide> = c.paths.iter().flatten().map(|b| b.side.clone()).collect();
            let node = self.network.node(crossing)?;
            let file = (!sides.is_empty()).then(|| {
                format!(
                    "paths/{}{:06}_edge{}_path.xyz",
                    node.kind.tag(),
                    node.number,
                    self.network.crossing_edges.len()
                )
            });
            let edge = self.network.add_crossing_edge(CrossingEdge {
                id: 0,
                level: Level::Exploration,
                crossing,
                ends,
                barriers,
                gradient_difference: c.gradient_difference,
                gap: c.gap.is_finite().then_some(c.gap),
                seam: c.seam.clone(),
                irc: IrcPath {
                    sides,
                    file,
                    possible_bifurcations: Vec::new(),
                },
                job: Some(job.id),
            })?;
            told.new_crossings.push((index, edge));
        }
        Ok(())
    }
}
