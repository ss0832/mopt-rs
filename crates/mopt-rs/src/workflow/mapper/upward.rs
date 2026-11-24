//! The upward search of saddle points of higher order (`method=upward`, experimental): from
//! every transition state joined to a minimum, displaced along each of its softest stable
//! modes (both ways), high-index saddle dynamics of index 2 (the reflected gradient with the
//! unstable mode of the transition state and the displaced direction, `path::hisd`, the
//! shrinking-dimer form) runs to a loose threshold, and the point is handed to the optimization
//! at order 2 with its directions followed; a saddle point of order 2 found so is a `2SP` node
//! and is descended from as the mapper descends from any (`higher_saddles`). Step 1 and 2 of
//! the table of `docs/source/mapper_design.md` ("Higher-order saddles").
//!
//! After the upward searches of the solution landscape (J. Yin, Y. Wang, J. Z. Y. Chen,
//! P. Zhang, L. Zhang, Phys. Rev. Lett. 2020, 124, 090601) and the high-index optimization-based
//! shrinking dimer method (J. Yin, L. Zhang, P. Zhang, SIAM J. Sci. Comput. 2019, 41, A3576).
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::collections::BTreeMap;

use faer::Col;

use super::follow::follow_saddle_point;
use super::method::{ExplorationMethod, Job, JobContext, JobOutcome, Task};
use super::network::{Level, Network, Node, NodeKind};
use crate::core_infra::json::JsonValue;
use crate::error::{MoptError, MoptResult};
use crate::math::linalg::col_norm;
use crate::path::hisd::landscape::upward_starts;

/// The softest stable modes of a transition state that are pushed along (`method_modes`).
pub const DEFAULT_MODES: usize = 2;

/// The displacement of the start along a mode (Bohr, the largest displacement of an atom;
/// the units of the variables for plain variables; `method_displacement`).
pub const DEFAULT_DISPLACEMENT: f64 = 0.1;

/// The settings of the method (`method_KEY=VALUE`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UpwardSettings {
    /// The softest stable modes pushed along.
    pub modes: usize,
    /// The displacement of the start.
    pub displacement: f64,
}

impl UpwardSettings {
    /// From the `method_*` settings (without the prefix).
    pub fn parse(extra: &BTreeMap<String, String>) -> MoptResult<Self> {
        let mut s = Self {
            modes: DEFAULT_MODES,
            displacement: DEFAULT_DISPLACEMENT,
        };
        for (k, v) in extra {
            match k.as_str() {
                "modes" => {
                    s.modes = v
                        .trim()
                        .parse()
                        .ok()
                        .filter(|&n: &usize| n >= 1)
                        .ok_or_else(|| {
                            MoptError::parse(format!(
                                "method_modes is a whole number of at least 1, not '{v}'"
                            ))
                        })?
                }
                "displacement" => {
                    s.displacement = v
                        .trim()
                        .parse()
                        .ok()
                        .filter(|&x: &f64| x.is_finite() && x > 0.0)
                        .ok_or_else(|| {
                            MoptError::parse(format!(
                                "method_displacement is a length above 0, not '{v}'"
                            ))
                        })?
                }
                other => {
                    return Err(MoptError::parse(format!(
                        "the mapper method upward: unknown setting method_{other} (keys: \
                         method_modes, method_displacement)"
                    )))
                }
            }
        }
        Ok(s)
    }
}

/// The method.
pub struct Upward {
    settings: UpwardSettings,
}

impl Upward {
    /// The name in the registry.
    pub const NAME: &'static str = "upward";

    /// The method with `settings`.
    pub fn new(settings: UpwardSettings) -> Self {
        Self { settings }
    }
}

/// `v` of unit length.
fn unit(v: &Col<f64>) -> Option<Col<f64>> {
    let n = col_norm(v);
    (n.is_finite() && n > 0.0).then(|| v * (1.0 / n))
}

impl ExplorationMethod for Upward {
    fn name(&self) -> &str {
        Self::NAME
    }

    fn tasks_follow_network(&self) -> bool {
        true
    }

    /// For every transition state joined to the minimum by its IRC (a node of its own, not a
    /// TS–TS link), each of the softest stable modes and both signs.
    fn tasks(&self, node: &Node, network: &Network) -> MoptResult<Vec<Task>> {
        if node.kind != NodeKind::Minimum {
            return Ok(Vec::new());
        }
        let mut seen = Vec::new();
        let mut out = Vec::new();
        for e in network
            .ts_edges
            .iter()
            .filter(|e| e.level == Level::Exploration && e.ends.contains(&node.id))
        {
            if seen.contains(&e.ts) {
                continue;
            }
            seen.push(e.ts);
            let label = network.node(e.ts)?.label();
            for m in 0..self.settings.modes {
                for sign in ["+", "-"] {
                    let mut p = BTreeMap::new();
                    p.insert("ts".to_string(), JsonValue::Number(e.ts as f64));
                    p.insert("mode".to_string(), JsonValue::Number(m as f64));
                    p.insert(
                        "sign".to_string(),
                        JsonValue::Number(if sign == "+" { 1.0 } else { -1.0 }),
                    );
                    out.push(Task {
                        key: format!("upward:{}:{m}:{sign}", e.ts),
                        description: format!(
                            "upward search from {label} along its stable mode {} ({sign})",
                            m + 1
                        ),
                        parameters: JsonValue::Object(p),
                    });
                }
            }
        }
        Ok(out)
    }

    fn run(&self, job: &Job, ctx: &mut JobContext<'_>) -> MoptResult<JobOutcome> {
        let number = |key: &str| {
            job.task
                .parameters
                .get(key)
                .and_then(JsonValue::as_f64)
                .ok_or_else(|| {
                    MoptError::parse(format!("upward: the task '{}' has no {key}", job.task.key))
                })
        };
        let (ts, mode, sign) = (
            number("ts")? as usize,
            number("mode")? as usize,
            number("sign")?,
        );
        let mut outcome = JobOutcome::default();
        let node = ctx.network.node(ts)?;
        let (point, label) = (node.system.clone(), node.label());
        let Some(modes) = ctx.jobs.modes(&point, false)? else {
            outcome.warnings.push(format!(
                "upward: no Hessian at {label}; nothing is searched"
            ));
            return Ok(outcome);
        };
        let unstable: Vec<Col<f64>> = modes
            .unstable
            .iter()
            .filter_map(|m| unit(&m.displacement))
            .collect();
        let stable: Vec<Col<f64>> = modes
            .modes
            .iter()
            .filter(|m| m.curvature > 0.0)
            .filter_map(|m| unit(&m.displacement))
            .collect();
        if unstable.len() != 1 {
            outcome.warnings.push(format!(
                "upward: {label} has {} negative curvatures, not one; nothing is searched",
                unstable.len()
            ));
            return Ok(outcome);
        }
        let Some(w) = stable.get(mode) else {
            return Ok(outcome);
        };
        let starts = upward_starts(
            &Col::zeros(w.nrows()),
            &unstable,
            std::slice::from_ref(w),
            self.settings.displacement,
        )?;
        let Some(start) = starts.into_iter().find(|s| s.sign * sign > 0.0) else {
            return Ok(outcome);
        };
        let what = format!(
            "high-index saddle dynamics of order 2 from {label} along its stable mode {} ({})",
            mode + 1,
            if sign > 0.0 { "+" } else { "-" }
        );
        let from =
            ctx.jobs
                .displace_mode(&point, &start.along, start.sign, self.settings.displacement)?;
        let search = ctx
            .jobs
            .hisd(&from, start.order, &start.directions, &what, ctx.frames)?;
        if search.interrupted {
            return Ok(JobOutcome::stopped());
        }
        if !search.settled {
            outcome.warnings.push(format!(
                "{what} ended without a saddle point of order 2 ({})",
                search.end
            ));
            return Ok(outcome);
        }
        let follow: Vec<Col<f64>> = search
            .directions
            .iter()
            .take(start.order)
            .cloned()
            .collect();
        if !follow_saddle_point(
            job,
            ctx,
            &search.point,
            start.order,
            &follow,
            &what,
            &mut outcome,
        )? {
            return Ok(JobOutcome::stopped());
        }
        Ok(outcome)
    }
}

/// A method with the upward searches added (`upward=on`): every minimum has the upward searches
/// from its transition states (`upward:`) first, as soon as it has transition states, and the
/// tasks of the method after them, each run by its own.
pub struct WithUpward {
    base: Box<dyn ExplorationMethod>,
    upward: Upward,
    name: String,
}

impl WithUpward {
    /// `base` with the upward searches of `settings`.
    pub fn new(base: Box<dyn ExplorationMethod>, settings: UpwardSettings) -> Self {
        let name = format!("{}+{}", base.name(), Upward::NAME);
        Self {
            base,
            upward: Upward::new(settings),
            name,
        }
    }
}

impl ExplorationMethod for WithUpward {
    fn name(&self) -> &str {
        &self.name
    }

    fn tasks_follow_network(&self) -> bool {
        true
    }

    fn tasks(&self, node: &Node, network: &Network) -> MoptResult<Vec<Task>> {
        let mut tasks = self.upward.tasks(node, network)?;
        tasks.extend(self.base.tasks(node, network)?);
        Ok(tasks)
    }

    fn run(&self, job: &Job, ctx: &mut JobContext<'_>) -> MoptResult<JobOutcome> {
        if job.task.key.starts_with("upward:") {
            self.upward.run(job, ctx)
        } else {
            self.base.run(job, ctx)
        }
    }
}
