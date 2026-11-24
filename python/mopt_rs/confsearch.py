"""``confsearch`` from Python, with search moves of your own.

:func:`conformer_search` runs the command ``confsearch`` (every option of the command line)
and returns where the run is, how many conformers it found, and their energies. Its ``moves``
are the way to plug in a search method: a dict of names and callables, one of which
``-cs_opt method=NAME`` chooses. A callable makes the start of every trial::

    def move(symbols, positions_bohr, trial, conformer, seed):
        ...
        return new_positions_bohr            # or (new_positions_bohr, "what was done")
                                             # or None: nothing more to try, the search ends

- ``symbols``: the element symbols of the atoms;
- ``positions_bohr``: the positions of the conformer the trial starts from, flat (3N, Bohr);
- ``trial``: the number of the trial, from 0;
- ``conformer``: the number of that conformer in the order of discovery (0: the optimized
  input);
- ``seed``: an integer drawn from the random numbers of the search, so that a move that seeds
  its own generator with it repeats when the search is repeated (``-cs_opt seed=``).

The search does the rest: it minimizes what the callable returns on the surface of the command
(any calculator, ONIOM with microiterations, a periodic cell), rejects a structure whose bonds
differ from those of the first conformer (in the mode of conformers), leaves out duplicates
and rotamers, chooses the conformer of the next trial (``-cs_opt priority=``), and ends by the
rules of the command (``-ms``, ``-nr``, ``-nl``). With ONIOM microiterations the structures a
move sees and returns are those of the core. The built-in moves (``torsion``, ``kick``,
``lowmode``; :func:`conformer_move_names`) go the same way in Rust
(``workflow::confsearch::moves::ConformerMove``); the documentation ("Your own search
methods" in ``running.md``, ``python -m mopt_rs.docs running``) describes both sides.

MultiOptPy: `Entrypoints/conformation_search.py` (no counterpart of the moves).

This file is part of mopt-py (the Python interface of mopt-rs).
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from mopt_rs import _core

#: ``move(symbols, positions_bohr, trial, conformer, seed)`` -> positions, (positions, text),
#: or ``None``.
ConformerMove = Callable[[list[str], list[float], int, int, int], Any]


@dataclass(frozen=True)
class ConformerSearchResult:
    """The end of a conformer search."""

    #: The directory of the run (``conformers.xyz``, the logs, the settings).
    directory: str
    #: Number of conformers (of transition states with ``-tsconf``).
    count: int
    #: Their energies (Hartree), ascending.
    energies: list[float]


def conformer_move_names() -> list[str]:
    """The names of the built-in moves (``-cs_opt method=``)."""
    return list(_core.conformer_move_names())


def conformer_search(
    args: Sequence[str], moves: Mapping[str, ConformerMove] | None = None
) -> ConformerSearchResult:
    """Run ``confsearch`` with the arguments ``args`` (without the program name).

    ``moves`` adds search moves by name; ``-cs_opt method=NAME`` among ``args`` chooses one
    (see the module documentation for the callable). The names of the strategies of the
    command (``afir``, ``imtd``) and of the built-in moves cannot be taken.
    """
    directory, count, energies = _core.run_confsearch(
        [str(a) for a in args], dict(moves) if moves else None
    )
    return ConformerSearchResult(directory=directory, count=count, energies=list(energies))
