"""The calls of the tools (as ``crates/mopt-rs/src/mcp/tools.rs``): a command tool starts a job,
the job tools follow it, the component tools answer at once.

Failures the caller can correct are results with ``isError``; an unknown tool is the protocol
error ``-32602``.

MultiOptPy: no counterpart (new in mopt-rs).

This file is part of mopt-py (the Python interface of mopt-rs).
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import functools
import json
from collections.abc import Callable
from typing import Any

from mopt_rs.mcp import components, tables
from mopt_rs.mcp.jobs import JobManager
from mopt_rs.mcp.paths import ToolError
from mopt_rs.mcp.protocol import RpcError, invalid_params
from mopt_rs.mcp.tables import Command


def call_result(value: Any, error: str | None = None) -> dict[str, Any]:
    """The result of a tool call."""
    if error is not None:
        return {"content": [{"type": "text", "text": error}], "isError": True}
    text = json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True)
    return {
        "content": [{"type": "text", "text": text}],
        "structuredContent": value,
        "isError": False,
    }


def _job_id(args: dict[str, Any]) -> str:
    job_id = args.get("job_id")
    if not isinstance(job_id, str):
        raise ToolError("give the job_id")
    return job_id


def call(
    name: str,
    args: dict[str, Any],
    commands: list[Command],
    jobs: JobManager,
    version: str,
) -> dict[str, Any]:
    """Call the tool ``name`` with ``args``."""
    command = next((c for c in commands if c["name"] == name and c["implemented"]), None)
    handlers: dict[str, Callable[[], dict[str, Any]]] = {
        "job_status": lambda: jobs.status(
            _job_id(args), min(int(args.get("tail_lines") or 40), 500)
        ),
        "job_stop": lambda: jobs.stop(_job_id(args)),
        "job_cancel": lambda: jobs.cancel(_job_id(args)),
        "mapper_steer": lambda: jobs.steer(_job_id(args), args.get("lines")),
        "job_restart": lambda: jobs.restart(
            _job_id(args), _optional_str(args, "run_directory"), bool(args.get("force", False))
        ),
        "job_result": lambda: jobs.result(_job_id(args)),
        "job_list": jobs.list,
        "list_commands": lambda: components.list_commands(commands),
        "command_help": lambda: components.command_help(
            commands, args, jobs.program, jobs.sandbox.root
        ),
        "list_calculators": components.list_calculators,
        "single_point": lambda: components.single_point(jobs.sandbox, jobs.policy, args),
    }
    _, structures = tables.compute_tools()
    handler: Callable[[], dict[str, Any]]
    if command is not None:
        handler = functools.partial(jobs.start, command, args, version)
    elif name in handlers:
        handler = handlers[name]
    elif name in structures:
        handler = functools.partial(components.computed, jobs.sandbox, jobs.policy, name, args)
    else:
        known = any(c["name"] == name for c in commands)
        hint = " (the command is not implemented in this version)" if known else ""
        raise invalid_params(f"Unknown tool: {name}{hint}")
    try:
        return call_result(handler())
    except (ToolError, ValueError, TypeError) as e:
        return call_result(None, str(e))
    except RpcError:
        raise


def _optional_str(args: dict[str, Any], key: str) -> str | None:
    value = args.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ToolError(f"{key} is a string")
    return value
