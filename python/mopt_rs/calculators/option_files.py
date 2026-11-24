"""Settings of calculators kept in files.

Two things, the same for every calculator:

- **Defaults per calculator.** A JSON file ``{"calculators": {NAME: {KEY: VALUE, ...}}}``
  holds the settings a calculator gets whenever it is used, in the form of ``-calc_opt``
  (``"simple": "RIJCOSX D3BJ"`` for ``orca``, ``"contrl": {"maxit": 200}`` or
  ``"contrl.maxit": 200`` for ``gamessus``). ``NAME`` is what follows ``-os`` (``orca``,
  ``pyscf``, ``ase:module:Class``; for a name with an argument, the part before the colon
  applies too). The files, later ones first in rank:

  1. ``~/.config/mopt-rs/calculator_options.json`` (``$XDG_CONFIG_HOME`` is honoured),
  2. ``mopt_calculator_options.json`` in the working directory,
  3. the files named in ``MOPT_CALCULATOR_OPTIONS`` (separated like ``PATH``).

  The command line (``-calc_opt``, or ``extra`` in Python) overrides them key by key, and
  the run log names the file and the keys that came from it.

- **Values read from files.** A value ``@FILE`` (on the command line or in such a JSON
  file) is the text of ``FILE``: input blocks of several lines need no quoting
  (``-calc_opt blocks=@orca_blocks.inp``). ``@@...`` is a value that starts with ``@``.
  A relative ``FILE`` is looked up in the working directory, or beside the JSON file that
  names it.

MultiOptPy: no counterpart (new in mopt-rs).

This file is part of mopt-py (the Python interface of mopt-rs).
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import json
import os
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import Any

__all__ = ["default_options", "file_value", "option_file_paths", "read_option_file"]

FILE_MARK = "@"
ENVIRONMENT = "MOPT_CALCULATOR_OPTIONS"


def file_value(value: Any, base: Path | None = None) -> Any:
    """``value`` with ``@FILE`` replaced by the text of the file (without the line ends at
    its end) and ``@@x`` by ``@x``; mappings and lists are searched through. ``base`` is
    the directory relative paths start from (the working directory if not given)."""
    if isinstance(value, str):
        if value.startswith(FILE_MARK * 2):
            return value[1:]
        if value.startswith(FILE_MARK) and len(value) > 1:
            path = Path(os.path.expanduser(value[1:].strip()))
            if base is not None and not path.is_absolute():
                path = base / path
            try:
                return path.read_text(encoding="utf-8").rstrip("\r\n")
            except OSError as error:
                raise ValueError(f"option value '{value}': {error}") from None
        return value
    if isinstance(value, Mapping):
        return {k: file_value(v, base) for k, v in value.items()}
    if isinstance(value, list):
        return [file_value(v, base) for v in value]
    return value


def option_file_paths() -> list[Path]:
    """The files of calculator settings that exist, lowest rank first."""
    config = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    candidates = [
        Path(config) / "mopt-rs" / "calculator_options.json",
        Path("mopt_calculator_options.json"),
    ]
    candidates += [Path(p) for p in os.environ.get(ENVIRONMENT, "").split(os.pathsep) if p]
    return [p for p in candidates if p.is_file()]


def read_option_file(path: str | os.PathLike[str]) -> dict[str, dict[str, Any]]:
    """The settings of one file: calculator name (lower case) to its options."""
    p = Path(path)
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{p}: {error}") from None
    table = doc.get("calculators") if isinstance(doc, Mapping) else None
    if not isinstance(table, Mapping):
        raise ValueError(f'{p}: expected {{"calculators": {{NAME: {{KEY: VALUE}}}}}}')
    out: dict[str, dict[str, Any]] = {}
    for name, options in table.items():
        if not isinstance(options, Mapping):
            raise ValueError(f"{p}: the settings of '{name}' are not a mapping of options")
        out[str(name).strip().lower()] = {
            str(k): file_value(v, p.resolve().parent) for k, v in options.items()
        }
    return out


def default_options(
    names: Sequence[str], given: Collection[str] = ()
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """The options the files give the calculator known under ``names`` (the general name
    first, the most specific last), and one remark per file that contributed, for the run
    log. Options in ``given`` (those of the command line) are left out: they win."""
    wanted = [n.strip().lower() for n in names if n]
    merged: dict[str, Any] = {}
    notes: list[str] = []
    for path in option_file_paths():
        table = read_option_file(path)
        found: dict[str, Any] = {}
        for name in wanted:
            found.update({k: v for k, v in table.get(name, {}).items() if k not in given})
        if found:
            merged.update(found)
            notes.append(f"options from {path}: {', '.join(found)}")
    return merged, tuple(notes)
