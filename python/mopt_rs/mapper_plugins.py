"""Exploration methods of the mapper written in Python (``run_mapper -map_opt
method=python:MODULE:CLASS``): a class that gives the tasks of a minimum and, for a job, the
transition-state guesses and first paths it proposes. The mapper does the rest on the surface
of the run, as for its own methods: every path is relaxed by a chain method (``neb`` by
default; ``method_path=dmf`` and ``method_path_KEY=VALUE`` choose another and its settings),
its highest maxima join the guesses, and every guess is optimized to a transition state whose
IRC joins two minima.

The class::

    class MyMethod:
        def __init__(self, settings: dict[str, str]) -> None:
            # The method_KEY=VALUE settings of -map_opt (without "method_"), all but path*.
            ...

        def tasks(self, node: dict) -> list[dict]:
            # The tasks of a minimum, the same list in every run with the same settings (a
            # restart makes them again and skips those that were done): each with a key
            # unique among those of the node, a description, and parameters (anything JSON
            # takes), which come back in the job.
            return [{"key": f"mine:{node['id']}", "description": "...", "parameters": {}}]

        def propose(self, job: dict) -> dict:
            # What the job proposes: guesses (points) and paths (lists of points), and notes
            # for the log of the job.
            return {"guesses": [...], "paths": [[...], ...], "notes": ["..."]}

A node (and the ``start`` of a job) is a dictionary: ``id``, ``label`` (``EQ3``), ``energy``
(Hartree; an enthalpy on a variable cell), and either ``symbols``, ``positions`` (Angstrom, one
row per atom), ``charge``, ``multiplicity``, and ``cell`` (the lattice vectors in Angstrom as
rows, or ``None``), or ``variables`` (the values of an objective of plain variables) with
``names``. A job is ``id``, ``node`` (its number), ``task`` (``key``, ``description``,
``parameters``), and ``start``. A point of a proposal is the positions of the atoms (Angstrom,
rows or flat; the atoms, charge, multiplicity, and cell of the start), a dictionary with
``positions`` and ``cell`` (a structure in another cell), or the values of the variables.

A proposer computes what it needs itself (with the calculators of :mod:`mopt_rs.calculators`,
for example); the calculations of the run, their journal, and the restart are the mapper's.
The pure Rust command line refuses ``python:`` methods: they run from the commands of the
Python package (``run_mapper`` installed with ``pip install .``, the distribution
``mopt-py``, or ``mopt_rs.cli``).

MultiOptPy: no counterpart (new in mopt-rs).

This file is part of mopt-py (the Python interface of mopt-rs).
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from mopt_rs._core import ANGSTROM_TO_BOHR as BOHR_PER_ANGSTROM

__all__ = ["make", "propose", "tasks"]


def make(spec: str, settings: Mapping[str, str]) -> Any:
    """The method of the class ``MODULE:CLASS`` (``spec``) made with ``settings``."""
    module_name, sep, class_name = spec.rpartition(":")
    if not sep or not module_name or not class_name:
        raise ValueError(
            f"method=python:{spec}: expected python:MODULE:CLASS (a module that Python imports "
            "and a class in it)"
        )
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name, None)
    if cls is None:
        raise ValueError(f"method=python:{spec}: module {module_name} has no {class_name}")
    method = cls(dict(settings))
    for needed in ("tasks", "propose"):
        if not callable(getattr(method, needed, None)):
            raise ValueError(f"method=python:{spec}: {class_name} has no method {needed}()")
    return method


def tasks(method: Any, node: Mapping[str, Any]) -> list[tuple[str, str, str]]:
    """The tasks of ``node`` as ``(key, description, parameters as JSON)``."""
    out = []
    keys = set()
    for k, task in enumerate(method.tasks(dict(node))):
        if not isinstance(task, Mapping) or "key" not in task:
            raise ValueError(f"tasks(): task {k + 1} is not a dictionary with a key")
        key = str(task["key"])
        if key in keys:
            raise ValueError(f"tasks(): the key {key!r} comes twice")
        keys.add(key)
        out.append((key, str(task.get("description", key)), json.dumps(task.get("parameters", {}))))
    return out


Point = tuple[list[float], list[float] | None]
"""A point for the Rust side: the flat positions in Bohr (or the values of the variables), and
the lattice vectors in Bohr (row by row) where the point has a cell of its own."""


def _point(p: Any, variables: bool, atoms: int) -> Point:
    """A point of a proposal in the units of the Rust side."""
    if variables:
        values = np.asarray(p, dtype=float).ravel()
        return values.tolist(), None
    cell = None
    if isinstance(p, Mapping):
        if p.get("cell") is not None:
            lattice = np.asarray(p["cell"], dtype=float).reshape(3, 3)
            cell = (lattice * BOHR_PER_ANGSTROM).ravel().tolist()
        p = p["positions"]
    positions = np.asarray(p, dtype=float).ravel()
    if positions.size != 3 * atoms:
        raise ValueError(
            f"a point of {positions.size} coordinates for {atoms} atoms ({3 * atoms} coordinates)"
        )
    return (positions * BOHR_PER_ANGSTROM).tolist(), cell


def propose(
    method: Any, job: Mapping[str, Any]
) -> tuple[list[Point], list[list[Point]], list[str]]:
    """What ``job`` proposes: its guesses, its paths, and its notes."""
    start = job["start"]
    variables = start.get("variables") is not None
    atoms = 0 if variables else len(start["symbols"])
    found = method.propose(dict(job)) or {}
    if not isinstance(found, Mapping):
        raise ValueError("propose(): expected a dictionary (guesses, paths, notes)")
    guesses = [_point(p, variables, atoms) for p in found.get("guesses", [])]
    paths: list[list[Point]] = []
    for path in found.get("paths", []):
        if not isinstance(path, Sequence) or isinstance(path, str | bytes):
            raise ValueError("propose(): a path is a list of points")
        paths.append([_point(p, variables, atoms) for p in path])
    notes = [str(n) for n in found.get("notes", [])]
    return guesses, paths, notes
