"""Prompt templates: how to drive a command through the job tools (as
``crates/mopt-rs/src/mcp/prompts.rs``).

MultiOptPy: no counterpart (new in mopt-rs).

This file is part of mopt-py (the Python interface of mopt-rs).
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

from typing import Any

from mopt_rs.mcp.protocol import invalid_params


def _argument(name: str, description: str, required: bool) -> dict[str, Any]:
    return {"name": name, "description": description, "required": required}


def list_prompts() -> list[dict[str, Any]]:
    """The prompts."""
    return [
        {
            "name": "optimize_structure",
            "title": "Optimize a structure",
            "description": "Optimize the structure of an xyz file with optmain and report the "
            "result.",
            "arguments": [
                _argument("structure", "An xyz file under the root of the server.", True),
                _argument("calculator", "The calculator (-os), e.g. lj or gfn1xtb.", False),
            ],
        },
        {
            "name": "follow_job",
            "title": "Follow a job",
            "description": "Follow a job to its end, continue it when it was stopped, and read "
            "its results.",
            "arguments": [_argument("job_id", "The job.", True)],
        },
    ]


def get_prompt(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """The messages of the prompt ``name``."""
    if name == "optimize_structure":
        structure = arguments.get("structure")
        if not isinstance(structure, str):
            raise invalid_params("optimize_structure needs the argument structure")
        calculator = arguments.get("calculator") or "lj"
        description = "Optimize a structure"
        body = (
            f"Optimize the structure in {structure} with mopt-rs. Call the tool optmain with "
            f'INPUT ["{structure}"] and othersoft "{calculator}" (see list_calculators for the '
            "names). Call job_status with the job_id it returns until the state is no longer "
            "running; on failure, read console_tail and log_tail. Then call job_result, read "
            "the file ending in _optimized.xyz with resources/read, and report the final "
            "energy, whether the optimization converged (mopt.log), and the evaluation totals."
        )
    elif name == "follow_job":
        job = arguments.get("job_id")
        if not isinstance(job, str):
            raise invalid_params("follow_job needs the argument job_id")
        description = "Follow a job"
        body = (
            f"Follow the mopt-rs job {job}: call job_status until the state is no longer "
            "running (state interrupted: continue with job_restart; failed: report console_tail "
            "and log_tail). Then call job_result and summarize the files of its run directories "
            "and the evaluation totals."
        )
    else:
        raise invalid_params(f"no prompt '{name}'")
    return {
        "description": description,
        "messages": [{"role": "user", "content": {"type": "text", "text": body}}],
    }
