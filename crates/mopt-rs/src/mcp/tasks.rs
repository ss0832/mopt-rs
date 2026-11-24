//! Jobs as tasks: the tasks extension `io.modelcontextprotocol/tasks` of `2026-07-28` and the
//! experimental tasks of `2025-11-25`.
//!
//! A task is the job that a command tool started: its `taskId` is the job id (a token of the
//! server, as every job id), and its status follows the latest attempt of the job:
//!
//! | job | task status |
//! |---|---|
//! | running | `working` (`cancelled` once the task was cancelled: the job then ends at its next check of `STOP`) |
//! | finished (exit code 0) | `completed`, with the result of `job_result` |
//! | interrupted (`STOP`, exit code 3), cancelled (`job_cancel`) | `cancelled` |
//! | failed (another exit code), unknown (its end was not recorded) | `failed` |
//!
//! `tasks/cancel` is cooperative, as the specification has it: the job gets `STOP` and ends
//! cleanly, so that `job_restart` can continue it (a job without a run directory yet is ended
//! at once). Tasks are kept as long as their jobs (`ttlMs`/`ttl` null).
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use super::jobs::{JobState, JobView};
use super::protocol::{object, text, INTERNAL_ERROR};
use crate::core_infra::json::JsonValue;

/// The extension id of tasks in `2026-07-28`.
pub const TASKS_EXTENSION: &str = "io.modelcontextprotocol/tasks";
/// `_meta` key of the task a result of `tasks/result` belongs to (`2025-11-25`).
pub const META_RELATED_TASK: &str = "io.modelcontextprotocol/related-task";
/// The revision with the experimental tasks among those with the handshake.
pub const LEGACY_TASKS_VERSION: &str = "2025-11-25";
/// How often clients are asked to look at a task (milliseconds).
pub const POLL_INTERVAL_MS: f64 = 2000.0;

/// The status of a task.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskStatus {
    /// The job runs.
    Working,
    /// Finished with exit code 0.
    Completed,
    /// Ended with another exit code, or its end is not known.
    Failed,
    /// Stopped, cancelled, or asked to stop by `tasks/cancel`.
    Cancelled,
}

impl TaskStatus {
    /// The name in the protocol.
    pub fn name(self) -> &'static str {
        match self {
            Self::Working => "working",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
        }
    }

    /// Whether the status does not change any more.
    pub fn terminal(self) -> bool {
        self != Self::Working
    }

    /// The status of a task over the job `view`.
    pub fn of(view: &JobView) -> Self {
        if view.cancel_requested {
            return Self::Cancelled;
        }
        match view.state {
            JobState::Running => Self::Working,
            JobState::Finished => Self::Completed,
            JobState::Interrupted | JobState::Cancelled => Self::Cancelled,
            JobState::Failed | JobState::Unknown => Self::Failed,
        }
    }
}

/// The status message of a task over `view`.
pub fn status_message(view: &JobView) -> String {
    let status = TaskStatus::of(view);
    match (status, view.state) {
        (TaskStatus::Cancelled, JobState::Running) => {
            "cancelled: the job ends at its next check of STOP".to_string()
        }
        (TaskStatus::Working, _) => match &view.last_iteration {
            Some(it) => format!("{} running: {it}", view.command),
            None => format!("{} running", view.command),
        },
        (_, JobState::Unknown) => {
            "the job was started by another server process and its end was not recorded".to_string()
        }
        (_, state) => match view.exit_code {
            Some(code) => format!("{} {} (exit code {code})", view.command, state.name()),
            None => format!("{} {}", view.command, state.name()),
        },
    }
}

/// The task `id` over `view` in the form of `2026-07-28` (without the result).
pub fn modern_task(id: &str, view: &JobView) -> JsonValue {
    object([
        ("taskId", text(id)),
        ("status", text(TaskStatus::of(view).name())),
        ("statusMessage", text(status_message(view))),
        ("createdAt", text(&view.created)),
        ("lastUpdatedAt", text(&view.updated)),
        ("ttlMs", JsonValue::Null),
        ("pollIntervalMs", JsonValue::Number(POLL_INTERVAL_MS)),
    ])
}

/// The task `id` over `view` in the form of `2025-11-25`.
pub fn legacy_task(id: &str, view: &JobView) -> JsonValue {
    object([
        ("taskId", text(id)),
        ("status", text(TaskStatus::of(view).name())),
        ("statusMessage", text(status_message(view))),
        ("createdAt", text(&view.created)),
        ("lastUpdatedAt", text(&view.updated)),
        ("ttl", JsonValue::Null),
        ("pollInterval", JsonValue::Number(POLL_INTERVAL_MS)),
    ])
}

/// The JSON-RPC error of a failed task over `view` (the `error` of a `DetailedTask`).
pub fn failure(id: &str, view: &JobView) -> JsonValue {
    object([
        ("code", JsonValue::Number(INTERNAL_ERROR as f64)),
        (
            "message",
            text(format!(
                "{} (job_result shows what it wrote)",
                status_message(view)
            )),
        ),
        (
            "data",
            object([
                ("job_id", text(id)),
                (
                    "exit_code",
                    view.exit_code
                        .map_or(JsonValue::Null, |c| JsonValue::Number(f64::from(c))),
                ),
            ]),
        ),
    ])
}

/// Whether the client capabilities `capabilities` of a request of `2026-07-28` declare the
/// tasks extension.
pub fn client_has_tasks(capabilities: Option<&JsonValue>) -> bool {
    capabilities
        .and_then(|c| c.get("extensions"))
        .and_then(|e| e.get(TASKS_EXTENSION))
        .is_some_and(|t| t.as_object().is_some())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn view(state: JobState, cancel_requested: bool) -> JobView {
        JobView {
            state,
            exit_code: None,
            cancel_requested,
            created: "2026-11-20T00:00:00Z".to_string(),
            updated: "2026-11-20T00:00:01Z".to_string(),
            command: "optmain".to_string(),
            last_iteration: None,
            log_bytes: 0,
        }
    }

    #[test]
    fn job_states_map_to_task_statuses() {
        let cases = [
            (JobState::Running, false, "working"),
            (JobState::Running, true, "cancelled"),
            (JobState::Finished, false, "completed"),
            (JobState::Interrupted, false, "cancelled"),
            (JobState::Cancelled, false, "cancelled"),
            (JobState::Failed, false, "failed"),
            (JobState::Unknown, false, "failed"),
        ];
        for (state, requested, want) in cases {
            assert_eq!(TaskStatus::of(&view(state, requested)).name(), want);
        }
        let t = modern_task("job-1", &view(JobState::Running, false));
        assert_eq!(t.get("ttlMs"), Some(&JsonValue::Null));
        let t = legacy_task("job-1", &view(JobState::Running, false));
        assert!(t.get("pollInterval").is_some());
        let caps = JsonValue::parse(r#"{"extensions":{"io.modelcontextprotocol/tasks":{}}}"#);
        assert!(caps.is_ok_and(|c| client_has_tasks(Some(&c))));
        assert!(!client_has_tasks(None));
    }
}
