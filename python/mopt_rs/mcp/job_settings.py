"""The arguments of a command tool as a settings file of the command, and the copies of the
inputs into the job directory (the rules of ``crates/mopt-rs/src/mcp/job_settings.rs``).

A flag is ``true``, one value a string or a number, several values an array, an optional value
may be ``true`` for the flag alone. The job runs the command with ``--config
mcp_settings.json``, the same document as a settings file of the command line.

MultiOptPy: no counterpart (new in mopt-rs).

This file is part of mopt-py (the Python interface of mopt-rs).
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import contextlib
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mopt_rs.mcp.paths import Sandbox, ToolError, path_like_parts
from mopt_rs.mcp.policy import Policy
from mopt_rs.mcp.tables import FILES_PROPERTY, INPUT_PROPERTY, Command, find_option

#: The settings file of a job.
SETTINGS_FILE = "mcp_settings.json"
COPY_LIMIT_BYTES = 1 << 30
COPY_LIMIT_FILES = 20_000


@dataclass
class JobRequest:
    """What a command tool was asked for."""

    inputs: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    settings: dict[str, Any] = field(default_factory=dict)


def _strings(value: Any, what: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        return list(value)
    raise ToolError(f"{what}: a path or a list of paths")


def _token(value: Any, key: str) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value.is_integer() and abs(value) < 1e15:
            return str(int(value))
        return repr(value)
    raise ToolError(f"{key}: a string or a number")


def _setting(o: dict[str, Any], value: Any) -> Any:
    key = o["key"]
    arity = o["arity"]
    if value is None:
        return None
    if arity == "flag":
        if value is True or (isinstance(value, str) and value.lower() == "true"):
            return True
        if value is False or (isinstance(value, str) and value.lower() == "false"):
            return None
        raise ToolError(f"{key} is a flag: true or false")
    if arity in ("single", "optional"):
        if arity == "optional" and value is True:
            return o.get("implicit") or ""
        if arity == "optional" and value is False:
            return None
        if isinstance(value, list):
            if len(value) != 1:
                raise ToolError(f"{key} takes one value")
            value = value[0]
        if isinstance(value, bool):
            raise ToolError(f"{key} takes one value (a string or a number)")
        return _token(value, key)
    items = value if isinstance(value, list) else [value]
    tokens = [_token(v, key) for v in items]
    if arity == "one_or_more" and not tokens:
        raise ToolError(f"{key} takes at least one value")
    return tokens


def request_of(command: Command, args: dict[str, Any]) -> JobRequest:
    """The request in the arguments of the tool of ``command``."""
    out = JobRequest()
    unknown: list[str] = []
    unavailable: list[str] = []
    for key, value in args.items():
        if key in (INPUT_PROPERTY, "input"):
            out.inputs = _strings(value, INPUT_PROPERTY)
            continue
        if key == FILES_PROPERTY:
            out.files = _strings(value, FILES_PROPERTY)
            continue
        o = find_option(command, key)
        if o is None:
            unknown.append(key)
        elif not o["implemented"]:
            unavailable.append(f"{'/'.join(o['flags'])} ({o['feature']})")
        else:
            v = _setting(o, value)
            if v is not None:
                out.settings[o["key"]] = v
    name = command["name"]
    if unknown:
        raise ToolError(
            f"{name} has no option {', '.join(unknown)} (the properties are the long option "
            "names of the command line, as in its input schema and in command_help)"
        )
    if unavailable:
        raise ToolError(f"not available in this version of mopt-rs: {', '.join(unavailable)}")
    if not out.inputs:
        raise ToolError(f"{name}: give the inputs as {INPUT_PROPERTY}")
    return out


def check_paths(sandbox: Sandbox, base: Path, settings: dict[str, Any]) -> None:
    """Every part of a value that may be a path must be relative, without ``..``, and stay
    inside the root as seen from ``base``."""
    for key, value in settings.items():
        values = [value] if isinstance(value, str) else value if isinstance(value, list) else []
        for v in values:
            for part in path_like_parts(v):
                try:
                    sandbox.check_value_part(base, part)
                except ToolError as e:
                    raise ToolError(f"{key}: {e}") from None


def check_json_paths(sandbox: Sandbox, base: Path, doc: Any) -> None:
    """The same checks for every string of a JSON document copied into a job."""
    if isinstance(doc, dict):
        for v in doc.values():
            check_json_paths(sandbox, base, v)
    elif isinstance(doc, list):
        for v in doc:
            check_json_paths(sandbox, base, v)
    elif isinstance(doc, str):
        for part in path_like_parts(doc):
            sandbox.check_value_part(base, part)


@dataclass
class CopyBudget:
    """What was copied into a job directory so far."""

    bytes: int = 0
    files: int = 0
    skipped: list[str] = field(default_factory=list)
    json: list[Any] = field(default_factory=list)


def _copy_entry(
    sandbox: Sandbox, policy: Policy, source: Path, target: Path, budget: CopyBudget, depth: int
) -> None:
    if depth > 32:
        raise ToolError("directories nested too deeply")
    if source.is_dir() and not source.is_symlink():
        target.mkdir(parents=True, exist_ok=True)
        for entry in sorted(source.iterdir()):
            resolved = entry
            if entry.is_symlink():
                try:
                    resolved = entry.resolve(strict=True)
                except (OSError, RuntimeError):
                    budget.skipped.append(sandbox.relative(entry))
                    continue
                if not sandbox.contains(resolved) or not resolved.is_file():
                    budget.skipped.append(sandbox.relative(entry))
                    continue
            elif not (entry.is_dir() or entry.is_file()):
                budget.skipped.append(sandbox.relative(entry))
                continue
            _copy_entry(sandbox, policy, resolved, target / entry.name, budget, depth + 1)
        return
    policy.check_file_name(target.name)
    source_file, opened = sandbox.open_regular(source)
    with source_file:
        budget.bytes += opened.st_size
        budget.files += 1
        if budget.bytes > COPY_LIMIT_BYTES or budget.files > COPY_LIMIT_FILES:
            raise ToolError(
                f"more than {COPY_LIMIT_FILES} files or {COPY_LIMIT_BYTES} bytes to copy"
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        # A new file (never through a link).
        with target.open("xb") as out:
            shutil.copyfileobj(source_file, out)
    if target.name.lower().endswith(".json"):
        with contextlib.suppress(ValueError, OSError):
            budget.json.append(json.loads(target.read_text(encoding="utf-8")))


def copy_into(
    sandbox: Sandbox, policy: Policy, jobs: Path, source: Path, job_dir: Path, budget: CopyBudget
) -> str:
    """Copy ``source`` (resolved, inside the root) into ``job_dir`` at its path relative to the
    root; returns that path. Every file is checked again when it is opened."""
    rel = sandbox.relative(source)
    if not rel:
        raise ToolError("the root itself cannot be an input; name its files")
    if source.is_dir() and (jobs == source or source in jobs.parents):
        raise ToolError(f"{rel}: a directory that holds the jobs cannot be copied into a job")
    try:
        _copy_entry(sandbox, policy, source, job_dir / rel, budget, 0)
    except ToolError as e:
        raise ToolError(f"{rel}: {e}") from None
    except OSError as e:
        raise ToolError(f"{rel}: {e}") from None
    return rel


def settings_document(inputs: list[str], settings: dict[str, Any], version: str) -> dict[str, Any]:
    """The settings file of a job."""
    doc = dict(settings)
    doc["INPUT"] = list(inputs)
    doc["mopt_rs_version"] = version
    return dict(sorted(doc.items()))
