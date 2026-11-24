"""Pages of the lists and their opaque cursors (as ``crates/mopt-rs/src/mcp/pagination.rs``).

A cursor is the offset of the next entry with a check value made from a key of this server
process, the list and the offset: any cursor the server did not give out is refused with
``-32602``.

MultiOptPy: no counterpart (new in mopt-rs).

This file is part of mopt-py (the Python interface of mopt-rs).
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from typing import Any

from mopt_rs.mcp.protocol import invalid_params

DEFAULT_PAGE_SIZE = 100
MAX_PAGE_SIZE = 1000


class Pager:
    """The pages of a server process."""

    def __init__(self, size: int = DEFAULT_PAGE_SIZE) -> None:
        self._key = secrets.token_bytes(32)
        self.size = min(max(size, 1), MAX_PAGE_SIZE)

    def _check(self, name: str, offset: int) -> str:
        mac = hmac.new(self._key, f"{name}\0{offset}".encode(), hashlib.sha256)
        return mac.hexdigest()[:16]

    def cursor(self, name: str, offset: int) -> str:
        """The cursor of the entries of ``name`` from ``offset`` on."""
        return f"{offset:x}.{self._check(name, offset)}"

    def offset(self, name: str, cursor: Any) -> int:
        """The offset of ``cursor`` (``None``: the start); raises ``-32602`` for others."""
        if cursor is None:
            return 0
        refused = invalid_params(f"invalid cursor for {name}")
        if not isinstance(cursor, str) or len(cursor) > 64 or "." not in cursor:
            raise refused
        try:
            offset = int(cursor.partition(".")[0], 16)
        except ValueError:
            raise refused from None
        if offset <= 0 or not hmac.compare_digest(cursor, self.cursor(name, offset)):
            raise refused
        return offset

    def page(self, name: str, items: list[Any], cursor: Any) -> tuple[list[Any], str | None]:
        """The page of ``items`` that ``cursor`` asks for, and the cursor of the rest."""
        offset = self.offset(name, cursor)
        if offset > len(items):
            raise invalid_params(f"invalid cursor for {name} (the list has become shorter)")
        end = min(offset + self.size, len(items))
        following = self.cursor(name, end) if end < len(items) else None
        return items[offset:end], following
