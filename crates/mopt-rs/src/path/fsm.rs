//! The freezing string method (FSM): a string grown from the reactant and the product
//! towards each other, a node at a time and the two sides in turn, each new node optimized by
//! a few quasi-Newton steps across the path and then frozen for good; when the two sides
//! meet, the highest node is the estimate of the transition state.
//!
//! The step `s` is the length of the interpolated path between the ends divided by `nodes`.
//! A new node lies at the length `s` from the frontier of its side along the interpolation
//! between the two frontiers (the linear synchronous transit for structures by default,
//! the straight line otherwise); its tangent is that of the interpolation there. It is
//! evaluated `gradients` times, with an L-BFGS step across the tangent (each Cartesian
//! component at most 0.3 Angstrom) between two evaluations, and frozen at its last point.
//! The string ends when the interpolation between the frontiers is shorter than `s`.
//!
//! A. Behn, P. M. Zimmerman, A. T. Bell, M. Head-Gordon, J. Chem. Phys. 2011, 135, 224108
//! (the method; the linear synchronous transit as the interpolation of choice); the number
//! of gradients per node (3) and of nodes (12) as the Q-Chem manual gives them for its
//! implementation (`FSM_NGRAD`, `FSM_NNODE`), and the bound of 0.3 Angstrom per component of
//! a step as arXiv:2407.09763 (the freezing string with interpolation in internal
//! coordinates) describes it. The L-BFGS pairs are kept from node to node of a side (the
//! nodes of a side are neighbours on the path).
//!
//! Hard constraints `c(x) = t` ([`super::string_nodes::PointGeometry::manifold`]; new in
//! mopt-rs, not in the paper): the string lives on their manifold `M`. A new node of the
//! interpolation between the frontiers (which runs through the space of the points) is put
//! onto `M` first, and its tangent is that of the interpolation taken into `T_x M`, so that the
//! force across it is `-(P_M g)` without its part along the tangent; every quasi-Newton step
//! of the node is in `T_x M` and is followed by the restore of the constraints, so that every
//! evaluated (and frozen) node holds them.
//!
//! [`FreezingString`] is an ask/tell state machine over points of `n` variables.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use faer::Col;

use super::interp::{images_between, FirstPath, PathFrame, PathSettings};
use super::points::{check_values, gradient_of, PointRequest, PointValue, SearchAction};
use super::string_nodes::{
    across, point_at_length, segment_lengths, unit, NodeStepSettings, NodeSteps, PointGeometry,
};
use crate::domain::Units;
use crate::error::{MoptError, MoptResult};

/// Settings of a freezing string (molecules: Bohr and Hartree).
#[derive(Debug, Clone)]
pub struct FsmSettings {
    /// Nodes the path between the ends is divided into (the step is its length over this).
    pub nodes: usize,
    /// Evaluations of the gradient per node.
    pub gradients: usize,
    /// The steps across the path.
    pub steps: NodeStepSettings,
    /// The interpolation between the frontiers (structures: any; plain variables: the
    /// straight line).
    pub interpolation: FirstPath,
    /// Images of an interpolation between the frontiers.
    pub images: usize,
}

impl Default for FsmSettings {
    fn default() -> Self {
        Self {
            nodes: 12,
            gradients: 3,
            steps: NodeStepSettings {
                max_step: f64::INFINITY,
                max_component: 0.3 * Units::ANGSTROM_TO_BOHR,
                ..NodeStepSettings::default()
            },
            interpolation: FirstPath::Lst,
            images: 10,
        }
    }
}

/// Why a freezing string ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FsmEnd {
    /// The two sides met.
    Joined,
    /// The interpolation between the frontiers stopped getting shorter (the sides do not
    /// approach each other).
    Stalled,
}

/// What one `tell` did.
#[derive(Debug, Clone)]
pub struct FsmReport {
    /// Side of the node (0: the reactant side, 1: the product side; `None` for the ends).
    pub side: Option<usize>,
    /// The point evaluated (the first of the ends at the start).
    pub x: Col<f64>,
    /// Its energy.
    pub energy: f64,
    /// Whether the node was frozen.
    pub frozen: bool,
    /// Length of the interpolation between the frontiers.
    pub gap: f64,
    /// Why the search ended, if it did.
    pub end: Option<FsmEnd>,
}

#[derive(Debug, Clone)]
struct Growing {
    side: usize,
    x: Col<f64>,
    tangent: Col<f64>,
    evaluations: usize,
}

/// Ask/tell state machine of the freezing string.
pub struct FreezingString {
    settings: FsmSettings,
    geometry: PointGeometry,
    sides: [Vec<(Col<f64>, f64)>; 2],
    ends: Option<[Col<f64>; 2]>,
    step: f64,
    growing: Option<Growing>,
    steps: [NodeSteps; 2],
    gap: f64,
    end: Option<FsmEnd>,
}

impl FreezingString {
    /// A string between `reactant` and `product` (the product laid onto the reactant for
    /// structures without a preferred orientation).
    pub fn new(
        settings: FsmSettings,
        reactant: &Col<f64>,
        product: &Col<f64>,
        geometry: PointGeometry,
    ) -> MoptResult<Self> {
        let n = reactant.nrows();
        if n == 0 || product.nrows() != n || settings.nodes < 2 || settings.gradients == 0 {
            return Err(MoptError::config(
                "freezing string: two ends of one size, at least 2 nodes and 1 gradient per node",
            ));
        }
        let product = geometry.laid_onto(product, reactant);
        let steps = [
            NodeSteps::new(settings.steps),
            NodeSteps::new(settings.steps),
        ];
        Ok(Self {
            settings,
            geometry,
            sides: [Vec::new(), Vec::new()],
            ends: Some([reactant.clone(), product]),
            step: 0.0,
            growing: None,
            steps,
            gap: f64::INFINITY,
            end: None,
        })
    }

    /// The path: the nodes of the reactant side, then those of the product side backwards
    /// (the ends included), with their energies.
    pub fn path(&self) -> Vec<(Col<f64>, f64)> {
        let mut out = self.sides[0].clone();
        out.extend(self.sides[1].iter().rev().cloned());
        out
    }

    /// The highest node between the ends.
    pub fn transition_state_estimate(&self) -> Option<(Col<f64>, f64)> {
        let path = self.path();
        let last = path.len().saturating_sub(1);
        path.into_iter()
            .enumerate()
            .filter(|(k, _)| *k > 0 && *k < last)
            .map(|(_, p)| p)
            .max_by(|a, b| a.1.total_cmp(&b.1))
    }

    /// Why the search ended, once it has.
    pub fn end(&self) -> Option<FsmEnd> {
        self.end
    }

    /// The length of a step.
    pub fn step_length(&self) -> f64 {
        self.step
    }

    fn requests(&self) -> Vec<PointRequest> {
        if let Some(ends) = &self.ends {
            return ends
                .iter()
                .map(|x| PointRequest::energy(x.clone()))
                .collect();
        }
        self.growing
            .iter()
            .map(|g| PointRequest::gradient(g.x.clone()))
            .collect()
    }

    /// Next action.
    pub fn ask(&self) -> SearchAction {
        if self.end.is_some() {
            return SearchAction::Finished;
        }
        SearchAction::Evaluate(self.requests())
    }

    /// The values at the points of [`Self::ask`].
    pub fn tell(&mut self, values: Vec<PointValue>) -> MoptResult<FsmReport> {
        if self.end.is_some() {
            return Err(MoptError::config(
                "tell() called after the freezing string ended",
            ));
        }
        let asked = self.requests();
        let n = asked.first().map_or(0, |p| p.x.nrows());
        check_values("freezing string", &asked, &values, n)?;
        if let Some([a, b]) = self.ends.take() {
            self.sides = [
                vec![(a.clone(), values[0].energy)],
                vec![(b.clone(), values[1].energy)],
            ];
            let path = self.interpolation(&a, &b)?;
            let length: f64 = segment_lengths(&path).iter().sum();
            self.step = length / self.settings.nodes as f64;
            self.gap = length;
            self.next_node(0)?;
            return Ok(FsmReport {
                side: None,
                x: a,
                energy: values[0].energy,
                frozen: false,
                gap: length,
                end: self.end,
            });
        }
        let mut growing = self
            .growing
            .take()
            .ok_or_else(|| MoptError::config("freezing string: nothing was asked"))?;
        let energy = values[0].energy;
        let g = self
            .geometry
            .project(&growing.x, &gradient_of("freezing string", &values[0])?);
        growing.evaluations += 1;
        let x = growing.x.clone();
        if growing.evaluations < self.settings.gradients {
            let t = growing.tangent.clone();
            let geometry = self.geometry.clone();
            let at = x.clone();
            let project = move |v: &Col<f64>| geometry.project(&at, &across(v, &t));
            let p = self.steps[growing.side].step(&x, &across(&g, &growing.tangent), &project);
            // Back onto the hard constraints, which the step leaves at second order.
            growing.x = self.geometry.settle(&(&growing.x + &p))?;
            let side = growing.side;
            self.growing = Some(growing);
            return Ok(FsmReport {
                side: Some(side),
                x,
                energy,
                frozen: false,
                gap: self.gap,
                end: None,
            });
        }
        let side = growing.side;
        self.sides[side].push((x.clone(), energy));
        self.next_node(1 - side)?;
        Ok(FsmReport {
            side: Some(side),
            x,
            energy,
            frozen: true,
            gap: self.gap,
            end: self.end,
        })
    }

    fn interpolation(&self, from: &Col<f64>, to: &Col<f64>) -> MoptResult<Vec<Col<f64>>> {
        let structures = self.geometry.rigid_body.is_some() && from.nrows().is_multiple_of(3);
        let mut line = vec![from.clone()];
        if structures && self.settings.interpolation != FirstPath::Linear {
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
                self.settings.images.max(1),
                &frame,
                &PathSettings::default(),
            )?;
            line.extend(images.images);
        }
        line.push(to.clone());
        Ok(line)
    }

    /// A new node on `side` (0: from the reactant side), or the end.
    fn next_node(&mut self, side: usize) -> MoptResult<()> {
        let from = self.sides[side]
            .last()
            .map(|p| p.0.clone())
            .ok_or_else(|| MoptError::config("freezing string: a side without a node"))?;
        let to = self.sides[1 - side]
            .last()
            .map(|p| p.0.clone())
            .ok_or_else(|| MoptError::config("freezing string: a side without a node"))?;
        let path = self.interpolation(&from, &to)?;
        let gap: f64 = segment_lengths(&path).iter().sum();
        if gap <= self.step {
            self.gap = gap;
            self.end = Some(FsmEnd::Joined);
            return Ok(());
        }
        // Each node takes a step out of the gap: a gap that does not shrink in two nodes
        // (both sides) means the frozen nodes moved back.
        if gap > self.gap + 2.0 * self.step {
            self.gap = gap;
            self.end = Some(FsmEnd::Stalled);
            return Ok(());
        }
        self.gap = gap;
        let (x, d) = point_at_length(&path, self.step)
            .ok_or_else(|| MoptError::calc("freezing string: the frontiers coincide"))?;
        // A new node of the interpolation onto the hard constraints, its tangent in their
        // tangent space there.
        let x = self.geometry.settle(&x)?;
        let tangent = unit(&self.geometry.project(&x, &d))
            .ok_or_else(|| MoptError::calc("freezing string: no tangent at a new node"))?;
        self.growing = Some(Growing {
            side,
            x,
            tangent,
            evaluations: 0,
        });
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::math::linalg::col_norm;

    /// `E = (x^2 - 1)^2 + 2 (y - 0.2 x)^2`.
    fn surface(p: &Col<f64>) -> PointValue {
        let (x, y) = (p[0], p[1]);
        let w = y - 0.2 * x;
        PointValue {
            energy: (x * x - 1.0).powi(2) + 2.0 * w * w,
            gradient: Some(Col::from_fn(2, |i| {
                [4.0 * x * (x * x - 1.0) - 0.8 * w, 4.0 * w][i]
            })),
            hessian: None,
        }
    }

    #[test]
    fn the_frozen_nodes_pass_near_the_saddle() {
        let settings = FsmSettings {
            nodes: 12,
            interpolation: FirstPath::Linear,
            steps: NodeStepSettings {
                first_scale: 0.1,
                max_step: 0.1,
                ..NodeStepSettings::default()
            },
            ..FsmSettings::default()
        };
        let a = Col::from_fn(2, |k| [-1.0, -0.2][k]);
        let b = Col::from_fn(2, |k| [1.0, 0.2][k]);
        let mut s = FreezingString::new(settings, &a, &b, PointGeometry::default()).unwrap();
        let mut evaluations = 0;
        while let SearchAction::Evaluate(points) = s.ask() {
            evaluations += points.len();
            s.tell(points.iter().map(|p| surface(&p.x)).collect())
                .unwrap();
            assert!(evaluations < 1000);
        }
        assert_eq!(s.end(), Some(FsmEnd::Joined));
        let (x, _) = s.transition_state_estimate().unwrap();
        // Within a step of the saddle.
        assert!(col_norm(&x) < 1.2 * s.step_length(), "{x:?}");
        // About nodes x gradients evaluations, and the two ends.
        assert!(evaluations <= 2 + 12 * 3 + 3, "{evaluations}");
    }
}
