"""The root of the server: every path of a request is checked by its text, resolved, and checked
again when it is used (the rules of ``crates/mopt-rs/src/mcp/paths.rs``).

A path from a client is refused by its text when it is empty, holds a NUL character, a
backslash or a colon, a ``..`` component, a component that ends in a dot or a space, a device
name reserved for devices, or starts like a drive or a UNC path. The rest is resolved by the operating
system (symbolic links and junctions as the system follows them) and must lie inside the
resolved root. A file is opened by :meth:`Sandbox.open_regular`, which resolves it again and
compares the opened file with the checked one.

MultiOptPy: no counterpart (new in mopt-rs).

This file is part of mopt-py (the Python interface of mopt-rs).
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import os
import re
import stat
from pathlib import Path
from typing import BinaryIO

#: Directory of the jobs under the root.
JOBS_DIR = "mcp_jobs"

_DEVICES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
    "CONIN$",
    "CONOUT$",
}


class ToolError(ValueError):
    """A failure the caller can correct (a tool result with ``isError``)."""


def refused_path_text(text: str) -> str | None:
    """Why ``text`` cannot be a path of a client (None: it can)."""
    parts = text.split("/")
    if not text:
        why = "an empty path"
    elif "\0" in text:
        why = "a NUL character"
    elif "\\" in text:
        why = "a backslash (paths use /)"
    elif text.startswith("//"):
        why = "a UNC path"
    elif ":" in text:
        why = "a colon (a drive, a device, or a stream)"
    elif ".." in parts:
        why = "a '..' component"
    elif any(len(c) > 1 and (c.endswith(".") or c.endswith(" ")) for c in parts):
        why = "a component that ends in a dot or a space"
    elif any(c.split(".")[0].rstrip().upper() in _DEVICES for c in parts):
        why = "a device name"
    else:
        return None
    shown = text.replace("\0", "\\0")
    return f"'{shown}' is refused: {why}"


def refused_value_part(text: str) -> str | None:
    """Why the part ``text`` of an option value cannot be given to a command (None: it can)."""
    t = text.lstrip("@")
    if "\0" in t:
        why = "a NUL character"
    elif "\\" in t:
        why = "a backslash (paths use /)"
    elif t.startswith("/"):
        why = "an absolute path (paths are relative to the job directory)"
    elif len(t) >= 2 and t[0].isascii() and t[0].isalpha() and t[1] == ":":
        why = "a drive"
    elif t == "~" or t.startswith("~/"):
        why = "a home directory"
    elif ".." in t.split("/"):
        why = "a '..' component"
    else:
        return None
    shown = text.replace("\0", "\\0")
    return f"'{shown}' is refused: {why}"


def path_like_parts(value: str) -> list[str]:
    """The value and its pieces between ``=``, ``,``, ``:``, ``;`` and white space."""
    out = [value]
    for piece in re.split(r"[=,:;\s]", value):
        if piece and piece != value:
            out.append(piece)
    return out


def _same_file(a: os.stat_result, b: os.stat_result) -> bool:
    return (a.st_dev, a.st_ino) == (b.st_dev, b.st_ino)


class Sandbox:
    """The directory the server may read and write."""

    def __init__(self, root: Path) -> None:
        try:
            resolved = root.resolve(strict=True)
        except OSError as e:
            raise ToolError(f"the root {root} cannot be resolved: {e}") from e
        if not resolved.is_dir():
            raise ToolError(f"the root {resolved} is not a directory")
        self.root = resolved

    def contains(self, path: Path) -> bool:
        """Whether the resolved ``path`` is the root or inside it."""
        return path == self.root or self.root in path.parents

    def relative(self, path: Path) -> str:
        """``path`` (resolved, inside the root) relative to the root (empty for the root)."""
        rel = path.relative_to(self.root).as_posix()
        return "" if rel == "." else rel

    def existing(self, user: str) -> Path:
        """The existing file or directory ``user``, resolved and inside the root."""
        why = refused_path_text(user)
        if why is not None:
            raise ToolError(why)
        try:
            resolved = (self.root / user).resolve(strict=True)
        except (OSError, RuntimeError) as e:
            raise ToolError(f"'{user}' does not name a file under the root ({e})") from e
        if not self.contains(resolved):
            raise ToolError(f"'{user}' is outside the root of the server")
        return resolved

    def check_value_part(self, base: Path, text: str) -> None:
        """Check the part ``text`` of an option value as a program started in ``base`` sees it."""
        why = refused_value_part(text)
        if why is not None:
            raise ToolError(why)
        try:
            resolved = (base / text.lstrip("@")).resolve(strict=True)
        except (OSError, RuntimeError, ValueError):
            return
        if not self.contains(resolved):
            raise ToolError(f"'{text}' leads outside the root of the server")

    def open_regular(self, path: Path) -> tuple[BinaryIO, os.stat_result]:
        """Open the regular file ``path`` for reading, resolved and checked again now."""
        shown = self.relative(path) if self.contains(path) else str(path)
        try:
            resolved = path.resolve(strict=True)
        except (OSError, RuntimeError) as e:
            raise ToolError(f"{shown}: {e}") from None
        if not self.contains(resolved):
            raise ToolError(f"{shown} leads outside the root of the server")
        # Checked before it is opened: opening a FIFO or a device could block or act.
        if not resolved.is_file():
            raise ToolError(f"{shown} is not a regular file")
        try:
            f = resolved.open("rb")
        except OSError as e:
            raise ToolError(f"{shown}: {e}") from None
        opened = os.fstat(f.fileno())
        try:
            now = path.resolve(strict=True)
            at_path = os.stat(now)
        except (OSError, RuntimeError) as e:
            f.close()
            raise ToolError(f"{shown}: {e}") from None
        if not stat.S_ISREG(opened.st_mode):
            f.close()
            raise ToolError(f"{shown} is not a regular file")
        if now != resolved or not _same_file(opened, at_path):
            f.close()
            raise ToolError(f"{shown} changed while it was opened")
        return f, opened

    def jobs_dir(self) -> Path:
        """The directory of the jobs (created when missing), resolved and inside the root."""
        d = self.root / JOBS_DIR
        if d.is_symlink() or (d.exists() and not d.is_dir()):
            raise ToolError(f"{JOBS_DIR} under the root is not a directory")
        d.mkdir(parents=True, exist_ok=True)
        resolved = d.resolve(strict=True)
        if not self.contains(resolved) or resolved == self.root:
            raise ToolError(f"{JOBS_DIR} is outside the root of the server")
        return resolved
