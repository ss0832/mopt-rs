//! Prompt templates: how to drive a command through the job tools.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use super::protocol::{object, text, RpcError};
use crate::core_infra::json::JsonValue;

fn argument(name: &str, description: &str, required: bool) -> JsonValue {
    object([
        ("name", text(name)),
        ("description", text(description)),
        ("required", JsonValue::Bool(required)),
    ])
}

/// The prompts.
pub fn list() -> Vec<JsonValue> {
    vec![
        object([
            ("name", text("optimize_structure")),
            ("title", text("Optimize a structure")),
            (
                "description",
                text("Optimize the structure of an xyz file with optmain and report the result."),
            ),
            (
                "arguments",
                JsonValue::Array(vec![
                    argument("structure", "An xyz file under the root of the server.", true),
                    argument("calculator", "The calculator (-os), e.g. lj or gfn1xtb.", false),
                ]),
            ),
        ]),
        object([
            ("name", text("follow_job")),
            ("title", text("Follow a job")),
            (
                "description",
                text("Follow a job to its end, continue it when it was stopped, and read its results."),
            ),
            (
                "arguments",
                JsonValue::Array(vec![argument("job_id", "The job.", true)]),
            ),
        ]),
    ]
}

/// The messages of the prompt `name` with `arguments`.
pub fn get(name: &str, arguments: &JsonValue) -> Result<JsonValue, RpcError> {
    let arg = |key: &str| arguments.get(key).and_then(JsonValue::as_str);
    let (description, body) = match name {
        "optimize_structure" => {
            let structure = arg("structure").ok_or_else(|| {
                RpcError::invalid_params("optimize_structure needs the argument structure")
            })?;
            let calculator = arg("calculator").unwrap_or("lj");
            (
                "Optimize a structure",
                format!(
                    "Optimize the structure in {structure} with mopt-rs. Call the tool optmain \
                     with INPUT [\"{structure}\"] and othersoft \"{calculator}\" (see \
                     list_calculators for the names). Call job_status with the job_id it \
                     returns until the state is no longer running; on failure, read \
                     console_tail and log_tail. Then call job_result, read the file ending in \
                     _optimized.xyz with resources/read, and report the final energy, whether \
                     the optimization converged (mopt.log), and the evaluation totals."
                ),
            )
        }
        "follow_job" => {
            let job = arg("job_id")
                .ok_or_else(|| RpcError::invalid_params("follow_job needs the argument job_id"))?;
            (
                "Follow a job",
                format!(
                    "Follow the mopt-rs job {job}: call job_status until the state is no longer \
                     running (state interrupted: continue with job_restart; failed: report \
                     console_tail and log_tail). Then call job_result and summarize the files of \
                     its run directories and the evaluation totals."
                ),
            )
        }
        other => return Err(RpcError::invalid_params(format!("no prompt '{other}'"))),
    };
    Ok(object([
        ("description", text(description)),
        (
            "messages",
            JsonValue::Array(vec![object([
                ("role", text("user")),
                (
                    "content",
                    object([("type", text("text")), ("text", text(body))]),
                ),
            ])]),
        ),
    ]))
}
