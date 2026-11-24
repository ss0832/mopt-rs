"""What the server refuses unless it was started with ``--allow-external``: settings that start
programs outside mopt-rs or load code (the rules of ``crates/mopt-rs/src/mcp/policy.rs``).

Refused without the flag: a calculator other than the built-in potentials, the objectives of
plain variables and the in-process libraries ``pyscf``, ``psi4``, ``tblite``, ``dxtb``
(``ase:module:Class``, ASE profiles, external programs, ``multioptpy:NAME``, plugins and OpenMM
need the flag); a reference to code ``module:name``; the settings ``hook``, ``hook.*``,
``excited_setup``, ``state_reader``, ``factory``, ``system`` (also as ``low.system=``); the option
``-osp``; copied files named ``software_path.conf`` or ``mopt_profiles.json``.

MultiOptPy: no counterpart (new in mopt-rs).

This file is part of mopt-py (the Python interface of mopt-rs).
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import re
from typing import Any

from mopt_rs.mcp.paths import ToolError

IN_PROCESS_ENGINES = ("pyscf", "psi4", "tblite", "dxtb")
SAFE_HEADS = ("tblite", "dxtb", "expr")
CODE_KEYS = ("hook", "excited_setup", "state_reader", "factory", "system")
PROGRAM_OPTIONS = ("software_path_file", "osp")
PROGRAM_FILES = ("software_path.conf", "mopt_profiles.json", "mopt_calculator_options.json")
SETTINGS_OPTIONS = ("calc_option", "calc_opt", "oniom_options", "oniom_opt")
_CALCULATOR_OPTIONS = ("othersoft", "os")
_LAYER_OPTIONS = ("oniom_flag", "oniom", "oniom_middle", "oniom_middle_flag")


def _identifier_path(s: str) -> bool:
    return bool(s) and all(seg.isidentifier() and seg.isascii() for seg in s.split("."))


def is_code_key(name: str) -> bool:
    """Whether the setting name ``name`` (``hook``, ``low.system``, ``hook.cmd``) loads code."""
    bare = name.strip().lstrip("-")
    for prefix in ("low.", "middle."):
        if bare.startswith(prefix):
            bare = bare[len(prefix) :]
            break
    return bare in CODE_KEYS or bare.split(".")[0] == "hook"


def is_code_reference(value: str) -> bool:
    """Whether ``value`` reads as ``module.path:name``."""
    v = value.strip()
    if ":" not in v:
        return False
    head, tail = v.split(":", 1)
    if head.lower() in SAFE_HEADS:
        return False
    tail = tail.split(":", 1)[0]
    return _identifier_path(head) and _identifier_path(tail.strip())


class Policy:
    """The policy of a server."""

    def __init__(self, allow_external: bool, calculators: list[str]) -> None:
        self.allow_external = allow_external
        self.calculators = {c.lower() for c in calculators} | set(IN_PROCESS_ENGINES)

    def safe_calculator(self, name: str) -> bool:
        """Whether the calculator ``name`` runs without the flag."""
        head = name.split(":", 1)[0].strip().lower()
        return head in ("", "none") or head in self.calculators

    def _refused(self, what: str) -> None:
        if not self.allow_external:
            raise ToolError(
                f"{what}: settings that start programs outside mopt-rs or load code are refused "
                "by this server (start it with --allow-external to allow them)"
            )

    def check_calculator(self, name: str) -> None:
        """Check the calculator ``name``."""
        if not self.safe_calculator(name):
            self._refused(f"the calculator '{name}'")

    def check_token(self, key: str, token: str, code_values: bool) -> None:
        """Check one token: setting names that load code (at any depth of ``KEY=``) and, in
        the settings of calculators, pieces that name code."""
        if code_values and (
            is_code_reference(token) or any(is_code_reference(p) for p in re.split(r"[=,]", token))
        ):
            self._refused(f"{key}: '{token}' names code")
        for setting in token.split("=")[:-1]:
            for name in setting.split(","):
                if is_code_key(name):
                    self._refused(f"{key}: the setting '{name.strip()}'")

    def check_option(self, key: str, values: list[str]) -> None:
        """Check the option ``key`` with the tokens ``values``."""
        bare = key.lstrip("-")
        if bare in PROGRAM_OPTIONS:
            self._refused(f"the option {bare} (paths of external programs)")
        if is_code_key(bare):
            self._refused(f"the setting '{bare}'")
        if bare in _CALCULATOR_OPTIONS:
            for v in values:
                self.check_calculator(v)
        if bare in _LAYER_OPTIONS and len(values) > 2:
            self.check_calculator(values[2])
        for v in values:
            self.check_token(bare, v, bare in SETTINGS_OPTIONS)

    def check_settings(self, settings: dict[str, Any]) -> None:
        """Check the settings of a job."""
        for key, value in settings.items():
            self.check_option(key, _strings(value))

    def check_file_name(self, name: str) -> None:
        """Check a copied file named ``name``."""
        if name.lower() in PROGRAM_FILES:
            self._refused(f"the file {name}")

    def check_json(self, doc: Any) -> None:
        """Check a JSON document copied into a job."""
        if isinstance(doc, dict):
            for key, value in doc.items():
                self.check_option(str(key), _strings(value))
                self.check_json(value)
        elif isinstance(doc, list):
            for item in doc:
                self.check_json(item)
        elif isinstance(doc, str):
            self.check_token("a settings file", doc, False)


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [v for v in value if isinstance(v, str)]
    return []
