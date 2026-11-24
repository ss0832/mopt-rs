//! The growing string method (GSM): a string of nodes grown from the reactant and the
//! product towards each other (double-ended, DE-GSM), or from the reactant alone along
//! driving coordinates (single-ended, SE-GSM, [`driving`]), then optimized as a whole with
//! its highest node climbing to the transition state.
//!
//! Growth: every node is moved across the tangent of the string by quasi-Newton steps
//! ([`super::string_nodes`]); when the frontier node of a side has settled (the RMS of its
//! gradient across the tangent below `add_tolerance`, or after `node_steps` steps) a node is
//! added beyond it. DE-GSM puts the new node into the gap between the two frontiers, at the
//! share of the gap that leaves the same spacing for the nodes still to come, until the
//! string has its `nodes`; SE-GSM steps by `drive_step` along the driving coordinates and
//! stops growing when they are made, when the string has passed over a maximum (the
//! frontier `drop` below the highest node), or at `nodes`; its last node is then minimized
//! (`final_steps` steps) and becomes the product.
//!
//! Optimization: every node takes a step across its tangent (the upwind tangent), the string
//! is spread evenly along its length, and once the largest RMS of the gradient across the
//! tangent is below `climb_tolerance` the highest node climbs (its gradient along the tangent
//! inverted, as the climbing image of Henkelman, Uberuaga, Jonsson, J. Chem. Phys. 2000, 113,
//! 9901); the sides on either side of it are spread separately. The search has converged when
//! the climbing node meets the force thresholds.
//!
//! B. Peters, A. Heyden, A. T. Bell, A. Chakraborty, J. Chem. Phys. 2004, 120, 7877 (the
//! growing string); P. M. Zimmerman, J. Chem. Phys. 2013, 138, 184102 (growth by settled
//! frontiers, a fixed number of nodes), J. Chem. Theory Comput. 2013, 9, 3043 (the climbing
//! node after the string has settled), J. Comput. Chem. 2015, 36, 601 (the single-ended
//! string). Zimmerman's strings live in delocalized internal coordinates; here the nodes are
//! Cartesian (rigid-body motions taken out), as in Peters et al.; the new nodes of DE-GSM are
//! placed on the straight line through the gap, or on the linear synchronous transit
//! between the frontiers (`interpolation`); the exact search for the transition state that
//! Zimmerman ends with is left to `optmain -order 1` from the estimate.
//!
//! Hard constraints `c(x) = t` ([`super::string_nodes::PointGeometry::manifold`]; new in
//! mopt-rs, not in the papers): the string lives on the manifold `M` of the constraints. The
//! perpendicular force of a node is `-(P_M g)_perp`, with `P_M` the projection onto the tangent
//! space `T_x M` (the surface gives `P_M g`) and the tangent of the string taken into `T_x M`
//! before the component along it is removed; the growth direction of SE-GSM (the driving
//! coordinates) is taken into `T_x M` too, and a driving coordinate that the constraints hold
//! has none left there (refused by name before the run). The steps of a node are in
//! `T_x M` and leave `M` at second order; every node that moved, every new node of the gap or
//! the drive, and every node the spreading moved is put back onto `M` (the restore of the
//! constraints) before it is evaluated, so that every evaluated and written node holds the
//! constraints.
//!
//! [`GrowingString`] is an ask/tell state machine over points of `n` variables.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

pub mod driving;
mod growth;
#[cfg(test)]
mod tests;

use faer::Col;

pub use driving::{Driving, DrivingCoordinate};

use super::interp::FirstPath;
use super::points::{check_values, gradient_of, PointRequest, PointValue, SearchAction};
use super::string_nodes::{
    across, rms, spread_evenly, tangent, unit, NodeStepSettings, NodeSteps, PointGeometry,
};
use crate::error::{MoptError, MoptResult};
use crate::math::linalg::{col_dot, col_max_abs, col_norm};
use crate::opt::convergence::ConvergenceCriteria;

/// A driving coordinate whose gradient keeps less than this share of its length in the
/// tangent space of the hard constraints is held by them.
const HELD_SHARE: f64 = 1e-6;

/// Settings of a growing string (molecules: Bohr and Hartree).
#[derive(Debug, Clone)]
pub struct GsmSettings {
    /// Nodes of the string, its ends included (SE-GSM: the most it grows to).
    pub nodes: usize,
    /// RMS of the gradient across the tangent at which a frontier counts as settled.
    pub add_tolerance: f64,
    /// Steps of a frontier after which a node is added in any case.
    pub node_steps: usize,
    /// Largest RMS of the gradient across the tangent at which the highest node starts
    /// climbing.
    pub climb_tolerance: f64,
    /// Iterations at most (each moves every node once).
    pub max_iterations: usize,
    /// The steps of the nodes.
    pub steps: NodeStepSettings,
    /// The force thresholds of the climbing node.
    pub criteria: ConvergenceCriteria,
    /// The interpolation of the gap of DE-GSM (structures: `Linear` or `Lst`).
    pub interpolation: FirstPath,
    /// SE-GSM: the length of a growth step along the driving coordinates.
    pub drive_step: f64,
    /// SE-GSM: the growth ends when the frontier is this far below the highest node.
    pub drop: f64,
    /// SE-GSM: minimization steps of the last node.
    pub final_steps: usize,
}

impl Default for GsmSettings {
    fn default() -> Self {
        Self {
            nodes: 9,
            add_tolerance: 5e-3,
            node_steps: 5,
            climb_tolerance: 2e-3,
            max_iterations: 300,
            steps: NodeStepSettings::default(),
            criteria: ConvergenceCriteria::NORMAL,
            interpolation: FirstPath::Linear,
            drive_step: 0.3,
            drop: 2e-3,
            final_steps: 20,
        }
    }
}

/// What the string is doing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GsmPhase {
    /// The ends are evaluated.
    Ends,
    /// The string grows.
    Growth,
    /// SE-GSM: the last node is minimized.
    LastNode,
    /// The whole string is optimized.
    Optimization,
}

/// Why a growing string ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GsmEnd {
    /// The climbing node meets the force thresholds.
    Converged,
    /// The iterations are used up.
    IterationLimit,
}

/// What one `tell` did.
#[derive(Debug, Clone)]
pub struct GsmReport {
    /// Iteration.
    pub iteration: usize,
    /// The phase the iteration was in.
    pub phase: GsmPhase,
    /// The nodes as they were evaluated (in the order of the string) and their energies.
    pub nodes: Vec<(Col<f64>, f64)>,
    /// RMS of the gradient across the tangent per node (0 at the ends).
    pub across: Vec<f64>,
    /// Index of the first node of the product side while there is a gap.
    pub gap: Option<usize>,
    /// The climbing node.
    pub climbing: Option<usize>,
    /// Why the search ended, if it did.
    pub end: Option<GsmEnd>,
}

#[derive(Debug, Clone)]
struct Node {
    x: Col<f64>,
    value: Option<(f64, Col<f64>)>,
    steps: NodeSteps,
    taken: usize,
    across: f64,
    pending: bool,
}

/// Ask/tell state machine of the growing string.
pub struct GrowingString {
    settings: GsmSettings,
    geometry: PointGeometry,
    driving: Option<Driving>,
    nodes: Vec<Node>,
    gap: Option<usize>,
    climbing: Option<usize>,
    phase: GsmPhase,
    iteration: usize,
    end: Option<GsmEnd>,
    free: usize,
}

impl GrowingString {
    fn check(settings: &GsmSettings, n: usize) -> MoptResult<()> {
        let positive = [
            settings.add_tolerance,
            settings.climb_tolerance,
            settings.drive_step,
            settings.steps.max_step,
        ];
        if n == 0
            || settings.nodes < 3
            || settings.max_iterations == 0
            || positive.iter().any(|v| !(v.is_finite() && *v > 0.0))
        {
            return Err(MoptError::config(
                "growing string: at least 3 nodes, and positive tolerances, steps, and counts",
            ));
        }
        Ok(())
    }

    fn node(&self, x: Col<f64>) -> Node {
        Node {
            x,
            value: None,
            steps: NodeSteps::new(self.settings.steps),
            taken: 0,
            across: f64::INFINITY,
            pending: true,
        }
    }

    /// DE-GSM between `reactant` and `product` (the product is laid onto the reactant for
    /// structures without a preferred orientation).
    pub fn double_ended(
        settings: GsmSettings,
        reactant: &Col<f64>,
        product: &Col<f64>,
        geometry: PointGeometry,
    ) -> MoptResult<Self> {
        let n = reactant.nrows();
        Self::check(&settings, n)?;
        if product.nrows() != n {
            return Err(MoptError::config("growing string: the ends differ in size"));
        }
        let product = geometry.laid_onto(product, reactant);
        let mut s = Self::empty(settings, geometry, None, n);
        s.nodes = vec![s.node(reactant.clone()), s.node(product)];
        s.gap = Some(1);
        Ok(s)
    }

    /// SE-GSM from `reactant` along `driving`.
    pub fn single_ended(
        settings: GsmSettings,
        reactant: &Col<f64>,
        driving: Driving,
        geometry: PointGeometry,
    ) -> MoptResult<Self> {
        let n = reactant.nrows();
        Self::check(&settings, n)?;
        driving.check(n)?;
        if geometry.manifold.is_some() {
            // A driving coordinate must be free to change on the hard constraints: its
            // gradient keeps a part in their tangent space.
            for c in &driving.coordinates {
                let (_, g) = c.value(reactant);
                let left = geometry.project(reactant, &g);
                if col_norm(&left) <= HELD_SHARE * col_norm(&g) {
                    return Err(MoptError::config(format!(
                        "single-ended growing string: the driving coordinate {c:?} is held by \
                         the constraints (-pc, -gfix, -uc, -fix); it cannot drive the growth"
                    )));
                }
            }
        }
        let mut s = Self::empty(settings, geometry, Some(driving), n);
        s.nodes = vec![s.node(reactant.clone())];
        Ok(s)
    }

    fn empty(
        settings: GsmSettings,
        geometry: PointGeometry,
        driving: Option<Driving>,
        n: usize,
    ) -> Self {
        let free = n - geometry.frozen.len().min(n);
        Self {
            settings,
            geometry,
            driving,
            nodes: Vec::new(),
            gap: None,
            climbing: None,
            phase: GsmPhase::Ends,
            iteration: 0,
            end: None,
            free,
        }
    }

    /// The nodes (in the order of the string) with their energies where known.
    pub fn nodes(&self) -> Vec<(Col<f64>, Option<f64>)> {
        self.nodes
            .iter()
            .map(|n| (n.x.clone(), n.value.as_ref().map(|v| v.0)))
            .collect()
    }

    /// The estimate of the transition state: the climbing node, or the highest node between
    /// the ends.
    pub fn transition_state_estimate(&self) -> Option<(Col<f64>, f64)> {
        if let Some((x, e)) = self
            .climbing
            .and_then(|c| self.nodes.get(c))
            .and_then(|n| n.value.as_ref().map(|v| (n.x.clone(), v.0)))
        {
            return Some((x, e));
        }
        let last = self.nodes.len().saturating_sub(1);
        self.nodes
            .iter()
            .enumerate()
            .filter(|(k, _)| *k > 0 && *k < last)
            .filter_map(|(_, n)| n.value.as_ref().map(|v| (n.x.clone(), v.0)))
            .max_by(|a, b| a.1.total_cmp(&b.1))
    }

    /// Why the search ended, once it has.
    pub fn end(&self) -> Option<GsmEnd> {
        self.end
    }

    /// The phase.
    pub fn phase(&self) -> GsmPhase {
        self.phase
    }

    /// Iterations so far.
    pub fn iterations(&self) -> usize {
        self.iteration
    }

    fn pending(&self) -> Vec<usize> {
        (0..self.nodes.len())
            .filter(|&k| self.nodes[k].pending)
            .collect()
    }

    /// Next action.
    pub fn ask(&self) -> SearchAction {
        if self.end.is_some() {
            return SearchAction::Finished;
        }
        SearchAction::Evaluate(
            self.pending()
                .into_iter()
                .map(|k| PointRequest::gradient(self.nodes[k].x.clone()))
                .collect(),
        )
    }

    /// The values at the points of [`Self::ask`].
    pub fn tell(&mut self, values: Vec<PointValue>) -> MoptResult<GsmReport> {
        if self.end.is_some() {
            return Err(MoptError::config(
                "tell() called after the growing string ended",
            ));
        }
        let pending = self.pending();
        let asked: Vec<PointRequest> = pending
            .iter()
            .map(|&k| PointRequest::gradient(self.nodes[k].x.clone()))
            .collect();
        let n = self.nodes[0].x.nrows();
        check_values("growing string", &asked, &values, n)?;
        for (&k, v) in pending.iter().zip(&values) {
            let g = self
                .geometry
                .project(&self.nodes[k].x, &gradient_of("growing string", v)?);
            self.nodes[k].value = Some((v.energy, g));
            self.nodes[k].pending = false;
        }
        let phase = self.phase;
        self.measure_across();
        let report = GsmReport {
            iteration: self.iteration,
            phase,
            nodes: self
                .nodes
                .iter()
                .map(|n| (n.x.clone(), n.value.as_ref().map_or(f64::NAN, |v| v.0)))
                .collect(),
            across: self
                .nodes
                .iter()
                .map(|n| if n.across.is_finite() { n.across } else { 0.0 })
                .collect(),
            gap: self.gap,
            climbing: self.climbing,
            end: None,
        };
        match phase {
            GsmPhase::Ends => self.start_growth()?,
            GsmPhase::Growth => self.grow()?,
            GsmPhase::LastNode => self.minimize_last()?,
            GsmPhase::Optimization => self.optimize()?,
        }
        // Every node that was moved, added, or spread is put back onto the hard constraints
        // before it is evaluated.
        if self.geometry.manifold.is_some() {
            for k in self.pending() {
                self.nodes[k].x = self.geometry.settle(&self.nodes[k].x)?;
            }
        }
        self.iteration += 1;
        if self.end.is_none() && self.iteration >= self.settings.max_iterations {
            self.end = Some(GsmEnd::IterationLimit);
        }
        Ok(GsmReport {
            end: self.end,
            climbing: self.climbing,
            ..report
        })
    }

    fn points(&self) -> Vec<Col<f64>> {
        self.nodes.iter().map(|n| n.x.clone()).collect()
    }

    fn energies(&self) -> Vec<f64> {
        self.nodes
            .iter()
            .map(|n| n.value.as_ref().map_or(f64::NAN, |v| v.0))
            .collect()
    }

    /// Whether node `k` is an end that does not move.
    fn is_end(&self, k: usize) -> bool {
        k == 0 || (self.has_product() && k == self.nodes.len() - 1)
    }

    fn has_product(&self) -> bool {
        self.driving.is_none() || self.phase == GsmPhase::Optimization
    }

    /// The tangent at node `k`.
    fn tangent_at(&self, k: usize) -> Option<Col<f64>> {
        let points = self.points();
        let t = match (&self.driving, self.phase) {
            // The frontier of SE-GSM moves across the driving direction.
            (Some(d), GsmPhase::Growth) if k == self.nodes.len() - 1 => d.direction(&points[k]).0,
            _ => tangent(&points, &self.energies(), k)?,
        };
        unit(&self.geometry.project(&points[k], &t))
    }

    fn measure_across(&mut self) {
        for k in 0..self.nodes.len() {
            let a = match (&self.nodes[k].value, self.tangent_at(k)) {
                (Some((_, g)), Some(t)) if !self.is_end(k) => rms(&across(g, &t), self.free),
                _ => f64::INFINITY,
            };
            self.nodes[k].across = a;
        }
    }

    /// One step of node `k` across its tangent (or climbing, or free).
    fn step_node(&mut self, k: usize, how: StepKind) -> MoptResult<()> {
        let x = self.nodes[k].x.clone();
        let Some((_, g)) = self.nodes[k].value.clone() else {
            return Ok(());
        };
        let t = self.tangent_at(k);
        let geometry = self.geometry.clone();
        let (g_eff, t_across) = match (how, &t) {
            (StepKind::Across, Some(t)) => (across(&g, t), Some(t.clone())),
            (StepKind::Climb, Some(t)) => (&g - t * (2.0 * col_dot(&g, t)), None),
            _ => (g.clone(), None),
        };
        let at = x.clone();
        let project = move |v: &Col<f64>| -> Col<f64> {
            let v = match &t_across {
                Some(t) => across(v, t),
                None => v.clone(),
            };
            geometry.project(&at, &v)
        };
        let p = self.nodes[k].steps.step(&x, &g_eff, &project);
        if col_max_abs(&p) > 0.0 {
            self.nodes[k].x += &p;
            self.nodes[k].pending = true;
        }
        self.nodes[k].taken += 1;
        Ok(())
    }

    fn optimize(&mut self) -> MoptResult<()> {
        self.measure_across();
        let last = self.nodes.len() - 1;
        let largest = (1..last)
            .filter(|k| Some(*k) != self.climbing)
            .map(|k| self.nodes[k].across)
            .fold(0.0_f64, f64::max);
        if self.climbing.is_none() && largest < self.settings.climb_tolerance {
            let energies = self.energies();
            let top = (1..last).max_by(|a, b| energies[*a].total_cmp(&energies[*b]));
            if let Some(c) = top {
                self.climbing = Some(c);
                self.nodes[c].steps.reset();
            }
        }
        if let Some(c) = self.climbing {
            let meets = self.nodes[c].value.as_ref().is_some_and(|(_, g)| {
                col_max_abs(g) < self.settings.criteria.max_force.value()
                    && rms(g, self.free) < self.settings.criteria.rms_force.value()
            });
            if meets && largest < self.settings.climb_tolerance {
                self.end = Some(GsmEnd::Converged);
                return Ok(());
            }
        }
        for k in 1..last {
            let how = if Some(k) == self.climbing {
                StepKind::Climb
            } else {
                StepKind::Across
            };
            self.step_node(k, how)?;
        }
        let mut points = self.points();
        match self.climbing {
            Some(c) => {
                spread_evenly(&mut points, 0, c);
                spread_evenly(&mut points, c, last);
            }
            None => {
                spread_evenly(&mut points, 0, last);
            }
        }
        for (k, x) in points.into_iter().enumerate() {
            if k > 0 && k < last && col_norm(&(&x - &self.nodes[k].x)) > 0.0 {
                self.nodes[k].x = x;
                self.nodes[k].pending = true;
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StepKind {
    Across,
    Climb,
    Free,
}
