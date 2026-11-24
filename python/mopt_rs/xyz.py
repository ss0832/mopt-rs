"""Structure files: ``.xyz`` and extended ``.xyz`` read and written by the library.

:func:`read_xyz` reads the first frame of a file, :func:`read_xyz_frames` all of them, with
the reader of the commands (files with a line of the number of atoms, MultiOptPy's files
without one, the lattice of an extended ``.xyz``). :func:`write_xyz` writes frames and
:func:`append_xyz` adds frames at the end of a file, with the writer of the commands: 12
decimals for the coordinates in Angstrom, 15 for the lattice, so that the values a file
gives back are those the run had to within 1e-12 Angstrom.

MultiOptPy: `fileio.py` (`xyz2list`, `traj2list`, `write_xyz_file`).

This file is part of mopt-py (the Python interface of mopt-rs).
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from mopt_rs import _core

FloatArray = npt.NDArray[np.float64]

__all__ = ["XyzFrame", "append_xyz", "read_xyz", "read_xyz_frames", "write_xyz"]


@dataclass
class XyzFrame:
    """One structure of an ``.xyz`` file."""

    #: Element symbols.
    symbols: list[str]
    #: Positions in Bohr, shape (N, 3), as the library holds them after reading.
    positions_bohr: FloatArray
    #: The comment line (the second line of the frame).
    comment: str = ""
    #: Lattice vectors as rows (Bohr, shape (3, 3)) of an extended ``.xyz``, else ``None``.
    lattice_bohr: FloatArray | None = None
    #: Periodic axes (a, b, c).
    pbc: tuple[bool, bool, bool] = (False, False, False)
    #: Energy (Hartree) the comment line gives (``energy=...`` or a number at its end).
    energy: float | None = None
    #: ``(charge, multiplicity)`` when the comment line is two whole numbers (MultiOptPy).
    state: tuple[int, int] | None = None

    @property
    def positions_angstrom(self) -> FloatArray:
        """Positions in Angstrom, shape (N, 3)."""
        return self.positions_bohr * _core.BOHR_TO_ANGSTROM

    @property
    def lattice_angstrom(self) -> FloatArray | None:
        """Lattice vectors as rows in Angstrom, or ``None``."""
        return None if self.lattice_bohr is None else self.lattice_bohr * _core.BOHR_TO_ANGSTROM


def _frame(d: dict[str, Any]) -> XyzFrame:
    lattice = d["lattice_bohr"]
    state = d["state"]
    a, b, c = (bool(v) for v in d["pbc"])
    cell = None if lattice is None else np.asarray(lattice, dtype=np.float64).reshape(3, 3)
    return XyzFrame(
        symbols=list(d["symbols"]),
        positions_bohr=np.asarray(d["positions_bohr"], dtype=np.float64).reshape(-1, 3),
        comment=str(d["comment"]),
        lattice_bohr=cell,
        pbc=(a, b, c),
        energy=None if d["energy"] is None else float(d["energy"]),
        state=None if state is None else (int(state[0]), int(state[1])),
    )


def read_xyz(path: str | os.PathLike[str]) -> XyzFrame:
    """The first frame of ``path``."""
    return _frame(_core.xyz_read(os.fspath(path), True)[0])


def read_xyz_frames(path: str | os.PathLike[str]) -> list[XyzFrame]:
    """All frames of ``path``, in order."""
    return [_frame(d) for d in _core.xyz_read(os.fspath(path), False)]


FrameLike = (
    XyzFrame | tuple[Sequence[str], npt.ArrayLike] | tuple[Sequence[str], npt.ArrayLike, str]
)


def _rows(
    frames: Iterable[FrameLike],
    lattice_angstrom: npt.ArrayLike | None,
    pbc: Sequence[bool] | None,
) -> list[tuple[list[str], list[float], str, list[float] | None, list[bool] | None]]:
    rows = []
    for frame in frames:
        if isinstance(frame, XyzFrame):
            lattice = frame.lattice_angstrom
            rows.append(
                (
                    list(frame.symbols),
                    np.ravel(frame.positions_angstrom).tolist(),
                    frame.comment,
                    None if lattice is None else np.ravel(lattice).tolist(),
                    list(frame.pbc) if lattice is not None else None,
                )
            )
            continue
        symbols, positions = frame[0], frame[1]
        comment = str(frame[2]) if len(frame) > 2 else ""  # type: ignore[misc]
        rows.append(
            (
                list(symbols),
                np.ravel(np.asarray(positions, dtype=np.float64)).tolist(),
                comment,
                None
                if lattice_angstrom is None
                else np.ravel(np.asarray(lattice_angstrom, dtype=np.float64)).tolist(),
                None if pbc is None else [bool(v) for v in pbc],
            )
        )
    return rows


def write_xyz(
    path: str | os.PathLike[str],
    symbols: Sequence[str] | Iterable[FrameLike],
    positions_angstrom: npt.ArrayLike | None = None,
    *,
    comment: str = "",
    lattice_angstrom: npt.ArrayLike | None = None,
    pbc: Sequence[bool] | None = None,
) -> None:
    """Write one frame (``symbols`` and ``positions_angstrom``, shape (N, 3) or (3N,)) to
    ``path``, replacing the file; or, with ``positions_angstrom`` left out, the frames in
    ``symbols`` (:class:`XyzFrame` objects, or tuples ``(symbols, positions_angstrom[,
    comment])``). ``lattice_angstrom`` (rows a, b, c) and ``pbc`` (default: periodic along
    all three) make an extended ``.xyz`` of the frames given as tuples."""
    rows = _frames(symbols, positions_angstrom, comment, lattice_angstrom, pbc)
    _core.xyz_write(os.fspath(path), rows, False)


def append_xyz(
    path: str | os.PathLike[str],
    symbols: Sequence[str] | Iterable[FrameLike],
    positions_angstrom: npt.ArrayLike | None = None,
    *,
    comment: str = "",
    lattice_angstrom: npt.ArrayLike | None = None,
    pbc: Sequence[bool] | None = None,
) -> None:
    """:func:`write_xyz`, adding the frames at the end of ``path`` (made if missing)."""
    rows = _frames(symbols, positions_angstrom, comment, lattice_angstrom, pbc)
    _core.xyz_write(os.fspath(path), rows, True)


def _frames(
    symbols: Sequence[str] | Iterable[FrameLike],
    positions_angstrom: npt.ArrayLike | None,
    comment: str,
    lattice_angstrom: npt.ArrayLike | None,
    pbc: Sequence[bool] | None,
) -> list[tuple[list[str], list[float], str, list[float] | None, list[bool] | None]]:
    if positions_angstrom is None:
        return _rows(symbols, lattice_angstrom, pbc)  # type: ignore[arg-type]
    names = [str(s) for s in symbols]  # type: ignore[union-attr]
    return _rows([(names, positions_angstrom, comment)], lattice_angstrom, pbc)
