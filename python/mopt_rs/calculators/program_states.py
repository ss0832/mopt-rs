"""Excited states of a program outside the package, in the form mopt-rs follows them.

Following a state from structure to structure (``-es_follow``) needs, at every structure,
the lowest excited states as orbitals and amplitudes of single excitations
(:meth:`mopt_rs.calculators.base.Engine.state_characters`). The engines of the package
that have them are the built-in GFN0-xTB and PySCF. For a program run through ASE (a
program of :mod:`mopt_rs.calculators.programs`, or any ``ase:module:Class``) the package
has the frame and leaves the two parts that differ from program to program to two
functions, which are named on the command line:

``-calc_opt excited_setup=module:function``
    ``function(calculator, state, count)`` sets the keywords of the ASE calculator so that
    its next calculation is for the excited state ``state`` (``0``: the ground state) and
    computes at least ``count`` states. It is called before every calculation. With it the
    engine takes ``-es N``.

``-calc_opt state_reader=module:function``
    ``function(calculator, structure, count)`` returns the states of the calculation that
    was just made as :class:`RawStates` (what the program wrote: the coefficients of the
    orbitals, the overlap of its basis functions, the excitation energies, and the
    amplitudes), or ``None``. :func:`state_characters` turns them into what is compared.
    ``state_reader=file`` reads :data:`STATES_FILE` in the directory of the calculator
    (:func:`read_states_file`), for a script of your own that converts the output of the
    program after every run.

Nothing here knows a program: no reader of the output of ORCA, Gaussian, Turbomole or any
other program is part of the package, and the frame is tested with a model program only
(``tests/test_program_states.py``). What a reader must respect:

- every orbital of the calculation, the doubly occupied ones first (restricted closed
  shells; frozen or removed orbitals make the set incomplete, which is refused);
- singlet single excitations (the X of a Tamm-Dancoff calculation, or X of a full
  TD-DFT/RPA calculation, which is normalized here; the de-excitations Y are not used);
- between two structures the basis functions are taken as the same functions on the moved
  atoms (the overlap of the two basis sets is not used), the approximation of the engines
  of the package.

MultiOptPy: no counterpart (new in mopt-rs).

This file is part of mopt-py (the Python interface of mopt-rs).
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

STATES_FILE = "mopt_states.json"
"""The file ``state_reader=file`` reads in the directory of the calculator."""

ExcitedSetup = Callable[[Any, int, int], None]
"""``function(calculator, state, count)``: see the module."""

StateReader = Callable[[Any, Any, int], "RawStates | None"]
"""``function(calculator, structure, count)``: see the module."""


@dataclass
class RawStates:
    """The excited states of one calculation as a program writes them."""

    coefficients: FloatArray
    """The orbitals in the basis functions of the program: a row per basis function, a
    column per orbital, the doubly occupied orbitals first."""
    occupied: int
    """The number of doubly occupied orbitals."""
    energies: FloatArray
    """The excitation energies in Hartree, ascending."""
    amplitudes: FloatArray
    """The amplitudes of the single excitations: ``[state, occupied orbital, empty
    orbital]``."""
    overlap: FloatArray | None = None
    """The overlap of the basis functions; ``None``: they are orthonormal."""


def state_characters(raw: RawStates, count: int) -> dict[str, Any]:
    """The lowest ``count`` states of ``raw`` as
    :meth:`~mopt_rs.calculators.base.Engine.state_characters` returns them: the orbitals in
    Loewdin orthogonalized basis functions, and one column of amplitudes of length one per
    state."""
    c = np.asarray(raw.coefficients, dtype=np.float64)
    if c.ndim != 2 or c.shape[0] != c.shape[1]:
        raise ValueError(
            f"{c.shape[1] if c.ndim == 2 else '?'} orbitals in {c.shape[0]} basis functions: "
            "every orbital of the calculation is needed (none frozen or removed)"
        )
    n = c.shape[0]
    occupied = int(raw.occupied)
    if not 0 < occupied < n:
        raise ValueError(f"{occupied} occupied orbitals of {n}")
    empty = n - occupied
    energies = np.asarray(raw.energies, dtype=np.float64).ravel()
    x = np.asarray(raw.amplitudes, dtype=np.float64)
    if x.shape != (energies.size, occupied, empty):
        raise ValueError(
            f"amplitudes of shape {x.shape} for {energies.size} states, {occupied} occupied "
            f"and {empty} empty orbitals (expected [state, occupied, empty])"
        )
    if np.any(np.diff(energies) < 0.0):
        raise ValueError("the excitation energies are not in ascending order")
    if raw.overlap is None:
        orbitals = c
    else:
        s = np.asarray(raw.overlap, dtype=np.float64)
        if s.shape != (n, n):
            raise ValueError(f"an overlap of shape {s.shape} for {n} basis functions")
        values, vectors = np.linalg.eigh(0.5 * (s + s.T))
        if values[0] <= 0.0:
            raise ValueError("the overlap of the basis functions is not positive definite")
        orbitals = (vectors * np.sqrt(values)) @ vectors.T @ c
    kept = min(count, energies.size)
    columns = x[:kept].reshape(kept, occupied * empty).T.copy()
    norms = np.linalg.norm(columns, axis=0)
    if np.any(norms < 1e-12):
        raise ValueError("a state without amplitudes")
    return {
        "orbitals": orbitals,
        "occupied": occupied,
        "energies": energies[:kept],
        "amplitudes": columns / norms,
    }


def write_states_file(path: str | Path, raw: RawStates) -> None:
    """Write ``raw`` as the JSON file that :func:`read_states_file` reads: the keys
    ``coefficients``, ``occupied``, ``energies`` (Hartree), ``amplitudes`` (``[state,
    occupied, empty]``), and ``overlap`` (or ``null``)."""
    data = {
        "coefficients": np.asarray(raw.coefficients, dtype=np.float64).tolist(),
        "occupied": int(raw.occupied),
        "energies": np.asarray(raw.energies, dtype=np.float64).ravel().tolist(),
        "amplitudes": np.asarray(raw.amplitudes, dtype=np.float64).tolist(),
        "overlap": None if raw.overlap is None else np.asarray(raw.overlap).tolist(),
    }
    Path(path).write_text(json.dumps(data), encoding="utf-8")


def read_states_file(path: str | Path) -> RawStates:
    """The states of the file written by :func:`write_states_file` (or by a script in the
    same form)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    try:
        overlap = data.get("overlap")
        return RawStates(
            coefficients=np.asarray(data["coefficients"], dtype=np.float64),
            occupied=int(data["occupied"]),
            energies=np.asarray(data["energies"], dtype=np.float64),
            amplitudes=np.asarray(data["amplitudes"], dtype=np.float64),
            overlap=None if overlap is None else np.asarray(overlap, dtype=np.float64),
        )
    except KeyError as missing:
        raise ValueError(f"{path}: no key {missing}") from None


def file_reader(calculator: Any, structure: Any, count: int) -> RawStates | None:
    """``state_reader=file``: :data:`STATES_FILE` in the directory of the calculator
    (``calculator.directory``, else the working directory); ``None`` if it is not there."""
    del structure, count
    path = Path(str(getattr(calculator, "directory", ".") or ".")) / STATES_FILE
    return read_states_file(path) if path.is_file() else None


def resolve(function: Any, what: str) -> Callable[..., Any]:
    """``function`` as a callable: itself, ``module:function``, or ``file`` for the reader
    of :data:`STATES_FILE`."""
    from mopt_rs.calculators.ase import resolve_object

    if callable(function):
        found: Callable[..., Any] = function
        return found
    text = str(function).strip()
    if what == "state_reader" and text.lower() == "file":
        return file_reader
    found = resolve_object(text)
    if not callable(found):
        raise TypeError(f"{what}={text} is not a function")
    return found


NOT_READ = (
    "{name} has no reader of its excited states: -es_follow needs "
    "-calc_opt state_reader=module:function (and excited_setup=module:function), see "
    "`python -m mopt_rs.docs running` under 'Excited states of a program outside the package'"
)
"""What an engine without a reader says when its states are asked for."""
