//! Steering a run while it goes on: lines a user puts into the file `steer.txt` of the run
//! directory, read before every choice of jobs, applied, kept in the network and in
//! `steer.log`, and removed.
//!
//! | Line | What it does |
//! |---|---|
//! | `explore EQ5` | the next jobs are the open tasks of that minimum, before the priority chooses again (several lines: in their order) |
//! | `skip EQ5` | no more jobs start from it |
//! | `unskip EQ5` | it is explored again (a minimum that a `skip` line left out) |
//! | `max_jobs N` | the limit of jobs from now on (0: none) |
//!
//! A node is named by its label (`EQ5`) or by its number in the network. Empty lines and what
//! follows a `#` are left out. What was applied is kept in the network ([`Steering`]), so a
//! restart goes on with it. The rules are in `docs/source/mapper_design.md` ("Steering a
//! run").
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use super::network::{Network, NodeKind};
use crate::error::{MoptError, MoptResult};

/// The file of the run directory with the lines of the user.
pub const STEER_FILE: &str = "steer.txt";

/// The file of the run directory with every line that was applied, and when.
pub const STEER_LOG: &str = "steer.log";

/// The start of the reason of a minimum that a `skip` line left out.
pub const STEERED_REASON: &str = "skipped by the user while the run went on (steer.txt)";

/// A line of `steer.txt`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SteerCommand {
    /// The next jobs are the open tasks of this minimum (its node).
    Explore(usize),
    /// No more jobs start from this minimum.
    Skip(usize),
    /// A minimum a `skip` line left out is explored again.
    Unskip(usize),
    /// The limit of jobs from now on (0: none).
    MaxJobs(usize),
}

/// A line that was applied.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AppliedLine {
    /// The jobs told when it was applied.
    pub after_jobs: usize,
    /// The line as it was given.
    pub line: String,
}

/// What the lines of `steer.txt` changed, kept in the network.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Steering {
    /// The minima whose open tasks run next, in this order (`explore`).
    pub first: Vec<usize>,
    /// The limit of jobs a line set (`max_jobs`; `None`: that of the settings).
    pub max_jobs: Option<usize>,
    /// Every line that was applied.
    pub applied: Vec<AppliedLine>,
}

impl Steering {
    /// Whether no line was ever applied.
    pub fn is_empty(&self) -> bool {
        self.applied.is_empty()
    }
}

/// The minimum of `network` that `name` names: its label (`EQ5`, any case) or its number in
/// the network.
fn minimum(network: &Network, name: &str) -> MoptResult<usize> {
    let by_label = network
        .nodes
        .iter()
        .find(|n| n.label().eq_ignore_ascii_case(name))
        .map(|n| n.id);
    let id = by_label
        .or_else(|| name.parse::<usize>().ok())
        .filter(|&id| id < network.nodes.len())
        .ok_or_else(|| {
            MoptError::parse(format!(
                "{STEER_FILE}: the network has no node '{name}' (a label such as EQ5, or the \
                 number of a node)"
            ))
        })?;
    let node = network.node(id)?;
    if node.kind != NodeKind::Minimum {
        return Err(MoptError::parse(format!(
            "{STEER_FILE}: {} is no minimum; jobs start from minima",
            node.label()
        )));
    }
    Ok(id)
}

/// The command of `line` for `network`; `None` for an empty line or a comment.
pub fn parse_line(line: &str, network: &Network) -> MoptResult<Option<SteerCommand>> {
    let text = line.split('#').next().unwrap_or_default().trim();
    let mut words = text.split_whitespace();
    let Some(verb) = words.next() else {
        return Ok(None);
    };
    let argument = words.next();
    if words.next().is_some() {
        return Err(MoptError::parse(format!(
            "{STEER_FILE}: '{text}' has more than a command and its argument"
        )));
    }
    let need = || {
        argument
            .ok_or_else(|| MoptError::parse(format!("{STEER_FILE}: '{text}' lacks its argument")))
    };
    let command = match verb.to_ascii_lowercase().as_str() {
        "explore" => SteerCommand::Explore(minimum(network, need()?)?),
        "skip" => SteerCommand::Skip(minimum(network, need()?)?),
        "unskip" => SteerCommand::Unskip(minimum(network, need()?)?),
        "max_jobs" => SteerCommand::MaxJobs(need()?.parse().map_err(|_| {
            MoptError::parse(format!(
                "{STEER_FILE}: '{text}': max_jobs takes a whole number (0: no limit)"
            ))
        })?),
        other => {
            return Err(MoptError::parse(format!(
                "{STEER_FILE}: unknown command '{other}' (explore NODE, skip NODE, unskip NODE, \
                 max_jobs N)"
            )))
        }
    };
    Ok(Some(command))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::workflow::mapper::configuration::VariablePoint;
    use crate::workflow::mapper::network::{ElectronicState, Level, Node, Origin};
    use crate::workflow::mapper::Configuration;

    fn network() -> Network {
        let mut net = Network::of_variables(vec!["x".to_string()]);
        for (kind, x) in [
            (NodeKind::Minimum, 0.0),
            (NodeKind::TransitionState, 0.5),
            (NodeKind::Minimum, 1.0),
        ] {
            let point = Configuration::Variables(VariablePoint {
                names: std::sync::Arc::new(vec!["x".to_string()]),
                values: faer::Col::from_fn(1, |_| x),
            });
            let state = ElectronicState::of_point(&point, 0);
            net.add_node(Node::new(
                kind,
                Level::Exploration,
                point,
                x,
                state,
                Origin::default(),
            ))
            .unwrap();
        }
        net
    }

    #[test]
    fn lines_name_minima_by_label_or_number() {
        let net = network();
        assert_eq!(
            parse_line("explore EQ1  # the second minimum", &net).unwrap(),
            Some(SteerCommand::Explore(2))
        );
        assert_eq!(
            parse_line("skip 0", &net).unwrap(),
            Some(SteerCommand::Skip(0))
        );
        assert_eq!(
            parse_line("UNSKIP eq0", &net).unwrap(),
            Some(SteerCommand::Unskip(0))
        );
        assert_eq!(
            parse_line("max_jobs 40", &net).unwrap(),
            Some(SteerCommand::MaxJobs(40))
        );
        assert_eq!(parse_line("   # nothing", &net).unwrap(), None);
        assert_eq!(parse_line("", &net).unwrap(), None);
        for (line, why) in [
            ("explore TS0", "no minimum"),
            ("explore EQ7", "no node"),
            ("skip", "lacks"),
            ("max_jobs many", "whole number"),
            ("faster EQ0", "unknown command"),
            ("skip EQ0 EQ1", "more than"),
        ] {
            let e = parse_line(line, &net).unwrap_err().to_string();
            assert!(e.contains(why), "{line}: {e}");
        }
    }
}
