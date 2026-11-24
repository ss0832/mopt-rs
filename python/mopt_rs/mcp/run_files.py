"""What every command leaves in its run directory, read the same way for every command (as
``crates/mopt-rs/src/mcp/run_files.rs``): the run directories, the tail of ``mopt.log``, its
last iteration, the totals of the evaluations, the files with their sizes.

MultiOptPy: no counterpart (new in mopt-rs).

This file is part of mopt-py (the Python interface of mopt-rs).
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

RUN_LOG = "mopt.log"
RUN_SETTINGS = "settings.json"
RUN_DEPTH = 6
_TAIL_BYTES = 256 * 1024

_MIME = {
    "md": "text/markdown",
    "csv": "text/csv",
    "json": "application/json",
    "html": "text/html",
    "htm": "text/html",
    "svg": "image/svg+xml",
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "gif": "image/gif",
    "pdf": "application/pdf",
    "sqlite": "application/vnd.sqlite3",
    "db": "application/vnd.sqlite3",
}
_PLAIN = {"log", "txt", "out", "dat", "inp", "xyz", "gjf", "mol", "mol2", "pdb"}


def is_run_directory(d: Path) -> bool:
    """Whether ``d`` is a run directory."""
    return (d / RUN_SETTINGS).is_file() or (d / RUN_LOG).is_file()


def sorted_entries(d: Path) -> list[Path]:
    """The entries of ``d``, sorted (none when it cannot be read)."""
    try:
        return sorted(d.iterdir())
    except OSError:
        return []


def run_directories(d: Path, excluded: list[Path], depth: int = RUN_DEPTH) -> list[Path]:
    """The run directories at most ``depth`` levels below ``d``, outside ``excluded``."""
    out: list[Path] = []
    level = [d]
    for _ in range(depth):
        following: list[Path] = []
        for parent in level:
            for child in sorted_entries(parent):
                if child.is_symlink() or not child.is_dir():
                    continue
                if any(child == x or x in child.parents for x in excluded):
                    continue
                (out if is_run_directory(child) else following).append(child)
        level = following
    return sorted(out)


def tail(path: Path, lines: int) -> str | None:
    """The last ``lines`` lines of a text file (a regular file, not a link; read from its last
    256 KiB)."""
    if path.is_symlink() or not path.is_file():
        return None
    try:
        with path.open("rb") as f:
            f.seek(0, 2)
            size = f.tell()
            start = max(0, size - _TAIL_BYTES)
            f.seek(start)
            data = f.read()
    except OSError:
        return None
    every = data.decode("utf-8", errors="replace").splitlines()
    if start > 0:
        every = every[1:]
    return "\n".join(every[max(0, len(every) - lines) :])


def last_iteration(log: str) -> str | None:
    """From the last line with ``ITR.`` to the end (at most 60 lines)."""
    every = log.splitlines()
    for at in range(len(every) - 1, -1, -1):
        if "ITR." in every[at]:
            return "\n".join(every[at : at + 60])
    return None


def evaluation_totals(content: str) -> dict[str, Any] | None:
    """The totals of the last line ``Evaluations (...): ...`` of ``content``."""
    line = next(
        (
            x.strip()
            for x in reversed(content.splitlines())
            if x.lstrip().startswith("Evaluations (")
        ),
        None,
    )
    if line is None:
        return None
    numbers = line.split("): ", 1)[1] if "): " in line else ""
    gradients: int | None = None
    hessians: int | None = None
    energies = 0
    for part in numbers.rstrip(".").split(", "):
        words = part.split()
        if not words or not words[0].isdigit():
            continue
        n = int(words[0])
        rest = " ".join(words[1:])
        if rest.startswith("gradient"):
            gradients = n
        elif rest.startswith("exact Hessian"):
            hessians = n
        elif "energy alone" in rest:
            energies = n
    return {"line": line, "gradients": gradients, "hessians": hessians, "energies": energies}


def list_files(d: Path, limit: int) -> tuple[list[tuple[Path, int]], bool]:
    """The regular files below ``d`` (symbolic links not followed) with their sizes, sorted, at
    most ``limit``; and whether there were more."""
    out: list[tuple[Path, int]] = []
    more = False
    stack = [d]
    while stack:
        current = stack.pop()
        subdirs = []
        for entry in sorted_entries(current):
            if entry.is_symlink():
                continue
            if entry.is_dir():
                subdirs.append(entry)
            elif entry.is_file():
                if len(out) >= limit:
                    more = True
                    continue
                out.append((entry, entry.stat().st_size))
        stack.extend(reversed(subdirs))
    return sorted(out), more


def mime_type(path: Path) -> str:
    """The media type of a file name."""
    ext = path.suffix.lower().lstrip(".")
    if ext in _PLAIN:
        return "text/plain"
    return _MIME.get(ext, "application/octet-stream")
