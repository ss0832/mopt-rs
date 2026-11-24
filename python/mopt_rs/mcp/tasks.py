"""Jobs as tasks (as ``crates/mopt-rs/src/mcp/tasks.rs``): the tasks extension
``io.modelcontextprotocol/tasks`` of ``2026-07-28`` and the experimental tasks of ``2025-11-25``.

The ``taskId`` is the job id; the status follows the latest attempt of the job: running is
``working`` (``cancelled`` once the task was cancelled), finished is ``completed``, interrupted
and cancelled are ``cancelled``, failed and unknown are ``failed``. ``tasks/cancel`` writes
``STOP`` (cooperative; ``job_restart`` can continue the job).

MultiOptPy: no counterpart (new in mopt-rs).

This file is part of mopt-py (the Python interface of mopt-rs).
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

from typing import Any

from mopt_rs.mcp.jobs import JobView
from mopt_rs.mcp.protocol import INTERNAL_ERROR

TASKS_EXTENSION = "io.modelcontextprotocol/tasks"
META_RELATED_TASK = "io.modelcontextprotocol/related-task"
LEGACY_TASKS_VERSION = "2025-11-25"
POLL_INTERVAL_MS = 2000
TERMINAL = ("completed", "failed", "cancelled")


def status_of(view: JobView) -> str:
    """The status of a task over the job ``view``."""
    if view.cancel_requested:
        return "cancelled"
    return {
        "running": "working",
        "finished": "completed",
        "interrupted": "cancelled",
        "cancelled": "cancelled",
    }.get(view.state, "failed")


def status_message(view: JobView) -> str:
    """The status message of a task over ``view``."""
    status = status_of(view)
    if status == "cancelled" and view.state == "running":
        return "cancelled: the job ends at its next check of STOP"
    if status == "working":
        if view.last_iteration is not None:
            return f"{view.command} running: {view.last_iteration}"
        return f"{view.command} running"
    if view.state == "unknown":
        return "the job was started by another server process and its end was not recorded"
    if view.exit_code is not None:
        return f"{view.command} {view.state} (exit code {view.exit_code})"
    return f"{view.command} {view.state}"


def modern_task(task_id: str, view: JobView) -> dict[str, Any]:
    """The task in the form of ``2026-07-28`` (without the result)."""
    return {
        "taskId": task_id,
        "status": status_of(view),
        "statusMessage": status_message(view),
        "createdAt": view.created,
        "lastUpdatedAt": view.updated,
        "ttlMs": None,
        "pollIntervalMs": POLL_INTERVAL_MS,
    }


def legacy_task(task_id: str, view: JobView) -> dict[str, Any]:
    """The task in the form of ``2025-11-25``."""
    return {
        "taskId": task_id,
        "status": status_of(view),
        "statusMessage": status_message(view),
        "createdAt": view.created,
        "lastUpdatedAt": view.updated,
        "ttl": None,
        "pollInterval": POLL_INTERVAL_MS,
    }


def failure(task_id: str, view: JobView) -> dict[str, Any]:
    """The JSON-RPC error of a failed task."""
    return {
        "code": INTERNAL_ERROR,
        "message": f"{status_message(view)} (job_result shows what it wrote)",
        "data": {"job_id": task_id, "exit_code": view.exit_code},
    }


def client_has_tasks(capabilities: Any) -> bool:
    """Whether the client capabilities of a request of ``2026-07-28`` declare the extension."""
    if not isinstance(capabilities, dict):
        return False
    extensions = capabilities.get("extensions")
    return isinstance(extensions, dict) and isinstance(extensions.get(TASKS_EXTENSION), dict)
