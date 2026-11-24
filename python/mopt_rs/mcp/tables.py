"""The command tables and the tool definitions built from them.

``commands.json`` next to this file holds the option tables of ``crates/mopt-rs/src/cli/spec.rs``
with the positional inputs; ``tools/cli/gen_cli.py`` writes both from the same data, and
``crates/mopt-rs/tests/mcp_tables.rs`` fails when they differ. The tools of the commands are
built from it at run time with the rules of ``crates/mopt-rs/src/mcp/schema.rs`` (the full
schema: one property per implemented option, named by its long name, the type from the arity;
the compact schema of ``tools/list``: the inputs and a few common options, the others by name),
so that a command that becomes implemented in the tables is a tool without a change here. The
component tools that compute from structures come from the compiled core
(``mopt_rs._core.mcp_component_tools``), so that both servers define them alike.

MultiOptPy: no counterpart (new in mopt-rs).

This file is part of mopt-py (the Python interface of mopt-rs).
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import copy
import functools
import json
from pathlib import Path
from typing import Any

#: Property of the positional inputs.
INPUT_PROPERTY = "INPUT"
#: Property of the further files copied into the job directory.
FILES_PROPERTY = "files"
#: The positional input of a command the tables give none for.
GENERIC_INPUT = {
    "name": "INPUT",
    "multiple": True,
    "help": "the input files or directories of the command",
}

Command = dict[str, Any]

#: The options that the compact schema of a command lists when the command has them.
COMMON_OPTIONS = (
    "othersoft",
    "opt_method",
    "NSTEP",
    "electronic_charge",
    "spin_multiplicity",
    "calc_option",
)
#: The longest ``job_wait`` (seconds).
MAX_WAIT_SECONDS = 600.0


def tables_path() -> Path:
    """The file of the tables."""
    return Path(__file__).resolve().parent / "commands.json"


def load_commands(path: Path | None = None) -> list[Command]:
    """The commands of the tables."""
    data = json.loads((path or tables_path()).read_text(encoding="utf-8"))
    commands: list[Command] = data["commands"]
    return commands


def positional(command: Command) -> dict[str, Any]:
    """The positional input of ``command``."""
    found = command.get("positional")
    return dict(found) if found else dict(GENERIC_INPUT)


def _no_default(default: str) -> bool:
    return default in ("", "None")


def option_schema(o: dict[str, Any]) -> dict[str, Any]:
    """The JSON Schema of one option."""
    flags = ", ".join(o["flags"])
    description = f"{o['help'].strip()} [command line: {flags}"
    if o["arity"] != "flag" and not _no_default(o["default"]):
        description += f"; default: {o['default']}"
    description += "]"
    scalar = ["string", "number"]
    schema: dict[str, Any] = {}
    arity = o["arity"]
    if arity == "flag":
        schema["type"] = "boolean"
        schema["default"] = False
    elif arity == "single":
        schema["type"] = scalar
        if not _no_default(o["default"]):
            schema["default"] = o["default"]
    elif arity == "optional":
        schema["type"] = ["string", "number", "boolean"]
        implicit = o.get("implicit") or ""
        description += (
            f" true: the flag without a value (which means {implicit})."
            if implicit
            else " true: the flag without a value."
        )
    else:
        schema["type"] = "array"
        schema["items"] = {"type": scalar}
        if arity == "one_or_more":
            schema["minItems"] = 1
    schema["description"] = description
    return schema


def command_tool(command: Command, full: bool = False) -> dict[str, Any]:
    """The tool that starts ``command`` as a job (``full``: every implemented option)."""
    name = command["name"]
    feature = command["feature"]
    given = positional(command)
    input_text = (
        f"{given['help'].rstrip('.')} (the positional argument {given['name']} of the command). "
        "Paths relative to the server root (or absolute inside it); each is copied into the job "
        "directory at the same relative path and given to the command from there."
    )
    input_schema: dict[str, Any]
    if given["multiple"]:
        input_schema = {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "description": input_text,
        }
    else:
        input_schema = {"type": "string", "description": input_text}
    properties: dict[str, Any] = {
        INPUT_PROPERTY: input_schema,
        FILES_PROPERTY: {
            "type": "array",
            "items": {"type": "string"},
            "description": "Further files or directories under the root that options name (an "
            "initial Hessian, a product structure, ...): copied into the job directory at the "
            "same relative paths, so that the paths of the options mean the same files there.",
        },
    }
    for o in command["options"]:
        if o["implemented"] and (full or o["key"] in COMMON_OPTIONS):
            properties[o["key"]] = option_schema(o)
    implemented = sum(1 for o in command["options"] if o["implemented"])
    additional: Any
    if full:
        options_text = (
            "The properties are the options of the command line by their long names (the keys "
            "of settings files)."
        )
        additional = False
    else:
        options_text = (
            f"Besides the properties listed, any of the {implemented} options of the command "
            "line is given by its long name (the key of settings files, e.g. opt_method, "
            f"NSTEP): call command_help with command={name} for each option's type, default "
            "and help (options=[...] to look a few up)."
        )
        additional = {
            "type": ["string", "number", "boolean", "array"],
            "description": "Another option of the command by its long name (see command_help).",
        }
    return {
        "name": name,
        "title": f"{name}: {feature}",
        "description": f"Start `{name}` ({feature}) as a job in a new directory under "
        f"mcp_jobs/; returns its job_id and run directory at once. {options_text} INPUT names "
        "the inputs. Follow the job with job_status or job_wait, read it with job_result; "
        "job_stop writes STOP into the run directory, job_restart continues it with --restart, "
        "job_cancel ends it at once.",
        "inputSchema": {
            "type": "object",
            "properties": properties,
            "required": [INPUT_PROPERTY],
            "additionalProperties": additional,
        },
        "annotations": _annotations(False, False),
    }


def command_tools(commands: list[Command], full: bool = False) -> list[dict[str, Any]]:
    """The tools of the implemented commands, in the order of the tables."""
    return [command_tool(c, full) for c in commands if c["implemented"]]


def _annotations(read_only: bool, idempotent: bool) -> dict[str, Any]:
    return {
        "readOnlyHint": read_only,
        "destructiveHint": False,
        "idempotentHint": idempotent,
        "openWorldHint": False,
    }


def _tool(
    name: str,
    description: str,
    properties: dict[str, Any],
    required: list[str],
    read_only: bool,
) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    schema["additionalProperties"] = False
    return {
        "name": name,
        "description": description,
        "inputSchema": schema,
        "annotations": _annotations(read_only, read_only),
    }


_JOB_ID = {"type": "string", "description": "The job_id that the tool of the command returned."}
_TAIL = {
    "type": "integer",
    "minimum": 0,
    "maximum": 500,
    "default": 40,
    "description": "Lines of the end of mopt.log to return.",
}


def job_tools() -> list[dict[str, Any]]:
    """The tools of the jobs."""
    return [
        _tool(
            "job_status",
            "State of a job (running, finished, interrupted, failed, unknown), its exit code, "
            "its run directories, the tail of mopt.log, the last iteration printed there, and "
            "the tail of the console output.",
            {"job_id": _JOB_ID, "tail_lines": _TAIL},
            ["job_id"],
            True,
        ),
        _tool(
            "job_wait",
            "Wait until a job does not run any more or the time is up, then answer as "
            "job_status (with waited_seconds and timed_out). While it waits, a request that "
            "carries _meta.progressToken gets notifications/progress at every new iteration in "
            "mopt.log; notifications/cancelled ends the wait without an answer.",
            {
                "job_id": _JOB_ID,
                "timeout_seconds": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": MAX_WAIT_SECONDS,
                    "default": 60,
                    "description": "The longest wait (seconds).",
                },
                "tail_lines": _TAIL,
            },
            ["job_id"],
            True,
        ),
        _tool(
            "job_stop",
            "Stop a running job cleanly: writes the file STOP into its run directories; the "
            "command ends at its next check and can be continued with job_restart.",
            {"job_id": _JOB_ID},
            ["job_id"],
            False,
        ),
        _tool(
            "job_cancel",
            "End a running job of this server process at once (the process is killed; state "
            "cancelled). What it wrote stays; job_restart continues it from its evaluation "
            "journal. Prefer job_stop, which ends it cleanly at its next check.",
            {"job_id": _JOB_ID},
            ["job_id"],
            False,
        ),
        _tool(
            "job_restart",
            "Continue a stopped or killed job in its run directory (the command with --restart "
            "RUN_DIR, which reads the settings and the evaluation journal kept there).",
            {
                "job_id": _JOB_ID,
                "run_directory": {
                    "type": "string",
                    "description": "The run directory to continue (relative to the root), when "
                    "the job has several.",
                },
                "force": {
                    "type": "boolean",
                    "default": False,
                    "description": "Restart a job of another server process whose state is "
                    "unknown (make sure it does not run any more).",
                },
            },
            ["job_id"],
            False,
        ),
        _tool(
            "job_result",
            "The files of a job's run directories with their sizes and resource URIs (read them "
            "with resources/read), and the totals of the evaluations of the run.",
            {"job_id": _JOB_ID},
            ["job_id"],
            True,
        ),
        _tool(
            "job_list",
            "The jobs under mcp_jobs/ of the root, with their commands and states.",
            {},
            [],
            True,
        ),
        _tool(
            "mapper_steer",
            "Steer a running run_mapper job: the lines are appended to steer.txt of its run "
            "directory, which the run reads before its next choice of jobs (explore NAME: its "
            "open tasks next; skip NAME / unskip NAME: no more jobs from that minimum, or again; "
            "max_jobs N: the limit of jobs from now on). NAME is a label such as EQ5 or a "
            "number; steer.log tells what became of each line.",
            {
                "job_id": _JOB_ID,
                "lines": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": 100,
                    "description": "Steering lines, e.g. explore EQ3, max_jobs 20.",
                },
            },
            ["job_id", "lines"],
            False,
        ),
    ]


def component_tools(commands: list[Command]) -> list[dict[str, Any]]:
    """The tools of the components."""
    return [
        _tool(
            "list_commands",
            "The commands of mopt-rs with whether they are implemented, their features, their "
            "numbers of options, and the name of the tool that starts each.",
            {},
            [],
            True,
        ),
        _tool(
            "command_help",
            "The --help text of a command (its options with flags, defaults, and what is not "
            "implemented yet), or, with options, the JSON Schema and help of the options named "
            "(long names, flags or parts of them); schema=true adds the full input schema of "
            "the command's tool, with every implemented option.",
            {
                "command": {
                    "type": "string",
                    "enum": [c["name"] for c in commands],
                    "description": "The command.",
                },
                "options": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Options to look up: long names, flags, or parts of them.",
                },
                "schema": {
                    "type": "boolean",
                    "default": False,
                    "description": "Also the full input schema of the command's tool.",
                },
            },
            ["command"],
            True,
        ),
        _tool(
            "list_calculators",
            "The calculators this server can evaluate (the names of -os), the built-in "
            "potentials among them, and the objectives of plain variables.",
            {},
            [],
            True,
        ),
        _tool(
            "single_point",
            "Energy (Hartree) and gradient (Hartree/Bohr) of one structure with a calculator, "
            "computed at once (for jobs, use the tools of the commands).",
            {
                "structure": {
                    "type": "string",
                    "description": "An xyz file under the root (or give the text as xyz).",
                },
                "xyz": {
                    "type": "string",
                    "description": "The text of an xyz file (instead of structure).",
                },
                "calculator": {
                    "type": "string",
                    "description": "Calculator name, as for -os (see list_calculators).",
                },
                "charge": {"type": "integer", "default": 0, "description": "Total charge."},
                "multiplicity": {
                    "type": "integer",
                    "minimum": 1,
                    "default": 1,
                    "description": "Spin multiplicity 2S + 1.",
                },
                "settings": {
                    "type": "object",
                    "additionalProperties": {"type": ["string", "number", "boolean"]},
                    "description": "Settings of the calculator, as KEY=VALUE of -calc_opt.",
                },
                "hessian": {
                    "type": "boolean",
                    "default": False,
                    "description": "Also the Hessian (Hartree/Bohr^2, 3N x 3N, row by row).",
                },
            },
            ["calculator"],
            True,
        ),
    ]


@functools.cache
def compute_tools() -> tuple[list[dict[str, Any]], dict[str, list[tuple[str, str]]]]:
    """The component tools of the compiled core and their structure arguments
    (``(file property, text property)`` per tool)."""
    from mopt_rs import _core

    doc = json.loads(_core.mcp_component_tools())
    structures = {k: [(f, x) for f, x in v] for k, v in doc["structures"].items()}
    return list(doc["tools"]), structures


def all_tools(commands: list[Command], full: bool = False) -> list[dict[str, Any]]:
    """Every tool of the server (``full``: the full schemas of the commands)."""
    computed, _ = compute_tools()
    return (
        command_tools(commands, full)
        + job_tools()
        + component_tools(commands)
        + copy.deepcopy(computed)
    )


#: The most options one query of ``command_help`` returns.
HELP_MATCHES = 40


def matching_options(command: Command, query: str) -> list[dict[str, Any]]:
    """The options that ``query`` names: its long name or a flag exactly, or else those whose
    names hold it (without dashes, ignoring case)."""
    bare = query.lstrip("-").lower()

    def names(o: dict[str, Any]) -> list[str]:
        return [o["key"].lower(), *(f.lstrip("-").lower() for f in o["flags"])]

    exact = [o for o in command["options"] if bare in names(o)]
    if exact or not bare:
        return exact
    return [o for o in command["options"] if any(bare in n for n in names(o))][:HELP_MATCHES]


def find_option(command: Command, key: str) -> dict[str, Any] | None:
    """The option named ``key`` (its key or any of its flags, without dashes)."""
    bare = key.lstrip("-")
    for o in command["options"]:
        if o["key"] == bare:
            return dict(o)
    for o in command["options"]:
        if any(f.lstrip("-") == bare for f in o["flags"]):
            return dict(o)
    return None
