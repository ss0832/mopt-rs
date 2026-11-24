//! The calls of the tools: a command tool starts a job, the job tools follow it, the
//! component tools answer at once.
//!
//! Failures that the caller can correct (an unknown option, a path outside the root, a job
//! that is not running) are results with `isError: true`; an unknown tool is a protocol error.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use super::components;
use super::compute;
use super::jobs::JobManager;
use super::protocol::{object, text, RpcError};
use super::schema::CommandTable;
use crate::core_infra::json::JsonValue;
use crate::pes::registry::CalculatorRegistry;

/// What a tool call needs.
#[derive(Debug)]
pub struct ToolContext<'a> {
    /// The command table.
    pub commands: CommandTable<'a>,
    /// The jobs.
    pub jobs: &'a mut JobManager,
    /// The calculators of the synchronous tools.
    pub registry: &'a CalculatorRegistry,
}

/// The result of a tool call (`isError` when `outcome` is an error).
pub fn call_result(outcome: Result<JsonValue, String>) -> JsonValue {
    match outcome {
        Ok(value) => object([
            (
                "content",
                JsonValue::Array(vec![object([
                    ("type", text("text")),
                    ("text", text(value.to_string_pretty())),
                ])]),
            ),
            ("structuredContent", value),
            ("isError", JsonValue::Bool(false)),
        ]),
        Err(message) => object([
            (
                "content",
                JsonValue::Array(vec![object([
                    ("type", text("text")),
                    ("text", text(message)),
                ])]),
            ),
            ("isError", JsonValue::Bool(true)),
        ]),
    }
}

fn job_id(args: &JsonValue) -> Result<&str, String> {
    args.get("job_id")
        .and_then(JsonValue::as_str)
        .ok_or_else(|| "give the job_id".to_string())
}

/// A tool of [`compute`]: the structures read through the sandbox; for `vibrations` with a
/// calculator and no Hessian, the Hessian of the calculator first (as `single_point`).
fn computed_tool(ctx: &ToolContext<'_>, name: &str, args: &JsonValue) -> Result<JsonValue, String> {
    let structures = components::structure_texts(ctx.jobs.sandbox(), name, args)?;
    if name == "vibrations" && args.get("hessian").is_none() {
        if args.get("calculator").is_none() {
            return Err("give the hessian, or a calculator to compute it".to_string());
        }
        let mut point = args.clone();
        if let Some(map) = point.as_object_mut() {
            map.insert("hessian".to_string(), JsonValue::Bool(true));
            for key in ["energy", "temperature", "pressure"] {
                map.remove(key);
            }
        }
        if let Some(n) = structures
            .get("structure")
            .and_then(|t| t.split_whitespace().next())
        {
            if n.parse::<usize>()
                .is_ok_and(|n| n > compute::MAX_HESSIAN_ATOMS)
            {
                return Err(format!(
                    "this tool takes at most {} atoms",
                    compute::MAX_HESSIAN_ATOMS
                ));
            }
        }
        let evaluated =
            components::single_point(ctx.jobs.sandbox(), ctx.jobs.policy(), ctx.registry, &point)?;
        let mut with_hessian = args.clone();
        if let Some(map) = with_hessian.as_object_mut() {
            for (from, to) in [
                ("hessian_hartree_per_bohr2", "hessian"),
                ("energy_hartree", "energy"),
            ] {
                if let Some(v) = evaluated.get(from) {
                    map.insert(to.to_string(), v.clone());
                }
            }
        }
        return compute::compute(name, &with_hessian, &structures);
    }
    compute::compute(name, args, &structures)
}

/// Call the tool `name` with `args`.
pub fn call(
    ctx: &mut ToolContext<'_>,
    name: &str,
    args: &JsonValue,
) -> Result<JsonValue, RpcError> {
    if let Some((command, _, _, options)) = ctx
        .commands
        .iter()
        .find(|(c, implemented, _, _)| *c == name && *implemented)
    {
        return Ok(call_result(ctx.jobs.start(command, options, args)));
    }
    let outcome = match name {
        "job_status" => job_id(args).and_then(|id| {
            let tail = args
                .get("tail_lines")
                .and_then(JsonValue::as_usize)
                .unwrap_or(40)
                .min(500);
            ctx.jobs.status(id, tail)
        }),
        "job_stop" => job_id(args).and_then(|id| ctx.jobs.stop(id)),
        "job_cancel" => job_id(args).and_then(|id| ctx.jobs.cancel(id)),
        "mapper_steer" => job_id(args).and_then(|id| {
            let lines = args
                .get("lines")
                .and_then(JsonValue::as_string_vec)
                .ok_or_else(|| "lines is a list of steering lines".to_string())?;
            ctx.jobs.steer(id, &lines)
        }),
        "job_restart" => job_id(args).and_then(|id| {
            let run = args.get("run_directory").and_then(JsonValue::as_str);
            let force = args
                .get("force")
                .and_then(JsonValue::as_bool)
                .unwrap_or(false);
            ctx.jobs.restart(id, run, force)
        }),
        "job_result" => job_id(args).and_then(|id| ctx.jobs.result(id)),
        "job_list" => ctx.jobs.list(),
        "list_commands" => Ok(components::list_commands(ctx.commands)),
        "command_help" => components::command_help(ctx.commands, args),
        "list_calculators" => Ok(components::list_calculators(ctx.registry)),
        "single_point" => {
            components::single_point(ctx.jobs.sandbox(), ctx.jobs.policy(), ctx.registry, args)
        }
        computed if compute::TOOLS.contains(&computed) => computed_tool(ctx, computed, args),
        other => {
            let hint = if ctx.commands.iter().any(|(c, _, _, _)| *c == other) {
                format!("Unknown tool: {other} (the command is not implemented in this version)")
            } else {
                format!("Unknown tool: {other}")
            };
            return Err(RpcError::invalid_params(hint));
        }
    };
    Ok(call_result(outcome))
}
