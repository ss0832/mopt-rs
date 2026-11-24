//! The dissociation channels of the driver: the node of an end of a reaction that is a
//! channel, the channel of a complex and the barrierless edge to it
//! ([`super::super::channels`]).
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use super::super::channels::{ChannelPart, DissociationEdge};
use super::super::dissociation::Dissociation;
use super::super::method::{Job, Stationary};
use super::super::network::{ElectronicState, Level, Node, NodeKind, Origin};
use super::MapperDriver;
use crate::error::MoptResult;
use crate::workflow::mapper::configuration::Configuration;

impl MapperDriver {
    /// The node of the end `end` of a reaction of `job`: a minimum (with the channel of its
    /// molecules when it is a complex), or a dissociation channel. New minima go to
    /// `new_minima`, new channels to `new_nodes`.
    pub(super) fn end_node(
        &mut self,
        job: &Job,
        end: &Stationary,
        new_minima: &mut Vec<usize>,
        new_nodes: &mut Vec<usize>,
    ) -> MoptResult<usize> {
        let origin = Origin {
            input: self.network.node(job.node)?.origin.input,
            job: Some(job.id),
            from_node: Some(job.node),
        };
        match &end.dissociation {
            Some(d) if d.separated => self.channel(origin, &end.system, d, new_nodes),
            Some(d) => {
                let id = self.minimum(job, end, new_minima)?;
                let channel = self.channel(origin, &end.system, d, new_nodes)?;
                self.join_channel(id, channel, Some(job.id), d.scan.as_ref())?;
                Ok(id)
            }
            None => self.minimum(job, end, new_minima),
        }
    }

    /// The channel node of the fragments `d` of `system`: the known one with the same
    /// fragments, or a new node (pushed to `new`). The fragments go to the network, where a
    /// known fragment keeps the lower of the two energies.
    pub fn channel(
        &mut self,
        origin: Origin,
        system: &Configuration,
        d: &Dissociation,
        new: &mut Vec<usize>,
    ) -> MoptResult<usize> {
        let mut parts = Vec::with_capacity(d.parts.len());
        for (atoms, record) in &d.parts {
            parts.push(ChannelPart {
                atoms: atoms.clone(),
                fragment: self.network.add_fragment(record.clone()),
            });
        }
        let records: Vec<usize> = parts.iter().map(|p| p.fragment).collect();
        if let Some(id) = self.network.find_channel(Level::Exploration, &records) {
            return Ok(id);
        }
        let mut node = Node::new(
            NodeKind::Dissociated,
            Level::Exploration,
            system.clone(),
            self.network.channel_energy(&records),
            ElectronicState::of_point(system, self.settings.excited_state),
            origin,
        );
        node.parts = parts;
        let id = self.network.add_node(node)?;
        new.push(id);
        Ok(id)
    }

    /// The barrierless dissociation of the complex `minimum` into `channel` (the known edge if
    /// they are joined already).
    pub fn join_channel(
        &mut self,
        minimum: usize,
        channel: usize,
        job: Option<usize>,
        scan: Option<&super::super::scan::DissociationScan>,
    ) -> MoptResult<usize> {
        let difference = self.network.node(channel)?.energy - self.network.node(minimum)?.energy;
        let edge = self.network.add_dissociation_edge(DissociationEdge {
            id: 0,
            level: Level::Exploration,
            minimum,
            channel,
            energy_difference: difference,
            job,
            scan: scan.cloned(),
        })?;
        // A known edge that was not scanned takes the scan of this one.
        if let (Some(e), Some(s)) = (self.network.dissociation_edges.get_mut(edge), scan) {
            if e.scan.is_none() {
                e.scan = Some(s.clone());
            }
        }
        if let Some(s) = scan.filter(|s| s.barrier) {
            let (complex, to) = (
                self.network.node(minimum)?.label(),
                self.network.node(channel)?.label(),
            );
            let message = format!(
                "{complex}: the scan towards its dissociation channel {to} has a maximum {:.2} \
                 kcal/mol above the channel; the dissociation is not barrierless (the edge is \
                 marked, and the kinetics does not take it for a capture)",
                s.highest.unwrap_or(f64::NAN) * crate::domain::Units::HARTREE_TO_KCAL_MOL
            );
            if !self.network.warnings.iter().any(|w| w.message == message) {
                self.network
                    .warnings
                    .push(super::super::network::NetworkWarning { job, message });
            }
        }
        Ok(edge)
    }
}
