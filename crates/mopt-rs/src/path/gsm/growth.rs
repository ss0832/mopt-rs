//! The growth of a growing string: the nodes added in the gap of DE-GSM and along the
//! driving coordinates of SE-GSM, and the minimization of the last node of SE-GSM.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Col;

use super::{GrowingString, GsmPhase, GsmSettings, Node, StepKind};
use crate::error::{MoptError, MoptResult};
use crate::math::linalg::col_max_abs;
use crate::path::interp::{images_between, FirstPath, PathFrame, PathSettings};
use crate::path::string_nodes::{point_at_length, rms, segment_lengths, unit};

impl GrowingString {
    pub(super) fn start_growth(&mut self) -> MoptResult<()> {
        self.phase = GsmPhase::Growth;
        if self.driving.is_some() {
            self.add_driven()?;
        } else {
            self.add_in_gap(true)?;
            if self.gap.is_some() {
                self.add_in_gap(false)?;
            }
        }
        Ok(())
    }

    /// The node at the distance that leaves equal spacing for the nodes still to come,
    /// from the frontier of the reactant side (`left`) or of the product side.
    pub(super) fn add_in_gap(&mut self, left: bool) -> MoptResult<()> {
        let Some(gap) = self.gap else {
            return Ok(());
        };
        let (a, b) = (self.nodes[gap - 1].x.clone(), self.nodes[gap].x.clone());
        let (from, to) = if left {
            (a.clone(), b.clone())
        } else {
            (b.clone(), a.clone())
        };
        let line = self.gap_path(&from, &to)?;
        let length: f64 = segment_lengths(&line).iter().sum();
        let left_to_add = self.settings.nodes - self.nodes.len();
        let share = length / (left_to_add + 1) as f64;
        let (x, _) = point_at_length(&line, share)
            .ok_or_else(|| MoptError::calc("growing string: the frontiers coincide"))?;
        let node = self.node(x);
        if left {
            self.nodes.insert(gap, node);
            self.gap = Some(gap + 1);
        } else {
            self.nodes.insert(gap, node);
        }
        if self.nodes.len() >= self.settings.nodes {
            self.join();
        }
        Ok(())
    }

    /// The path through the gap from `from` to `to`.
    pub(super) fn gap_path(&self, from: &Col<f64>, to: &Col<f64>) -> MoptResult<Vec<Col<f64>>> {
        let n = from.nrows();
        let structures = self.geometry.rigid_body.is_some() && n.is_multiple_of(3);
        let mut line = vec![from.clone()];
        if self.settings.interpolation != FirstPath::Linear && structures {
            let frame = PathFrame {
                elements: self
                    .geometry
                    .rigid_body
                    .as_ref()
                    .map(|s| s.elements.clone())
                    .unwrap_or_default(),
                ..PathFrame::default()
            };
            let images = images_between(
                self.settings.interpolation,
                from,
                to,
                8,
                &frame,
                &PathSettings::default(),
            )?;
            line.extend(images.images);
        }
        line.push(to.clone());
        Ok(line)
    }

    pub(super) fn join(&mut self) {
        self.gap = None;
        if self.driving.is_none() {
            self.phase = GsmPhase::Optimization;
        }
    }

    pub(super) fn add_driven(&mut self) -> MoptResult<()> {
        let Some(driving) = &self.driving else {
            return Ok(());
        };
        let last = &self.nodes[self.nodes.len() - 1].x;
        let (d, _) = driving.direction(last);
        let d = unit(&self.geometry.project(last, &d)).ok_or_else(|| {
            MoptError::calc("growing string: the driving coordinates give no direction")
        })?;
        let x = last + &d * self.settings.drive_step;
        let node = self.node(x);
        self.nodes.push(node);
        Ok(())
    }

    pub(super) fn grow(&mut self) -> MoptResult<()> {
        let count = self.nodes.len();
        for k in 0..count {
            if !self.is_end(k) {
                self.step_node(k, StepKind::Across)?;
            }
        }
        if self.driving.is_some() {
            return self.grow_driven();
        }
        let Some(gap) = self.gap else {
            self.phase = GsmPhase::Optimization;
            return Ok(());
        };
        let settled =
            |n: &Node, s: &GsmSettings| n.across < s.add_tolerance || n.taken >= s.node_steps;
        let left = gap >= 2 && settled(&self.nodes[gap - 1], &self.settings);
        let right = gap + 1 < self.nodes.len() && settled(&self.nodes[gap], &self.settings);
        if left {
            self.add_in_gap(true)?;
        }
        if right && self.gap.is_some() {
            self.add_in_gap(false)?;
        }
        Ok(())
    }

    pub(super) fn grow_driven(&mut self) -> MoptResult<()> {
        let last = self.nodes.len() - 1;
        let frontier = &self.nodes[last];
        if !(frontier.across < self.settings.add_tolerance
            || frontier.taken >= self.settings.node_steps)
        {
            return Ok(());
        }
        let energies = self.energies();
        let (top, highest) =
            energies[..=last]
                .iter()
                .enumerate()
                .fold(
                    (0, f64::NEG_INFINITY),
                    |m, (k, e)| if *e > m.1 { (k, *e) } else { m },
                );
        let made = self
            .driving
            .as_ref()
            .is_some_and(|d| d.direction(&frontier.x).1);
        let passed = top > 0 && top < last && energies[last] < highest - self.settings.drop;
        if made || passed || self.nodes.len() >= self.settings.nodes {
            if self.nodes.len() < 3 {
                return Err(MoptError::calc(
                    "single-ended growing string: the growth ended before a node between the \
                     ends (larger -gsm_nodes, or other driving coordinates)",
                ));
            }
            self.phase = GsmPhase::LastNode;
            let k = self.nodes.len() - 1;
            self.nodes[k].steps.reset();
            self.nodes[k].taken = 0;
            return Ok(());
        }
        self.add_driven()
    }

    pub(super) fn minimize_last(&mut self) -> MoptResult<()> {
        let k = self.nodes.len() - 1;
        let converged = self.nodes[k].value.as_ref().is_some_and(|(_, g)| {
            col_max_abs(g) < self.settings.criteria.max_force.value()
                && rms(g, self.free) < self.settings.criteria.rms_force.value()
        });
        if converged || self.nodes[k].taken >= self.settings.final_steps {
            self.phase = GsmPhase::Optimization;
            for node in &mut self.nodes {
                node.steps.reset();
            }
            return self.optimize();
        }
        self.step_node(k, StepKind::Free)
    }
}
