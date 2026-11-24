"""Write a database with SQLite itself for the reader of ``core_infra::sqlite``.

The file ``tests/golden/sqlite_written_by_sqlite3.db`` has a page size of 1024 bytes (not the
4096 that mopt-rs writes), a table of 2000 rows of which some were deleted and some updated
(free blocks and moved cells), rows longer than a page (overflow chains), an index (which the
reader must skip), and a second table. The rows that remain are fully determined by this
script, so the test knows them.

Usage:
    python -I tools/golden/sqlite_fixture.py <repository root>

This file is part of mopt-rs.
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    path = root / "tests/golden/sqlite_written_by_sqlite3.db"
    path.unlink(missing_ok=True)
    con = sqlite3.connect(path)
    con.execute("PRAGMA page_size = 1024")
    con.execute("PRAGMA journal_mode = DELETE")
    con.execute("CREATE TABLE nodes(id INTEGER PRIMARY KEY, label TEXT, energy REAL, data BLOB)")
    con.execute("CREATE INDEX nodes_label ON nodes(label)")
    con.execute("CREATE TABLE notes(id INTEGER PRIMARY KEY, text TEXT)")
    rows = []
    for i in range(1, 2001):
        label = f"EQ{i}" if i % 3 else f"TS{i}"
        data = bytes((i * k) % 256 for k in range(3000 if i % 250 == 0 else 8))
        rows.append((i, label, -0.001 * i, data))
    con.executemany("INSERT INTO nodes VALUES (?, ?, ?, ?)", rows)
    con.execute("DELETE FROM nodes WHERE id % 7 = 0")
    con.execute("UPDATE nodes SET label = label || '_updated' WHERE id % 11 = 0")
    con.executemany(
        "INSERT INTO notes VALUES (?, ?)", [(1, "first"), (2, "é" * 600), (3, None)]
    )
    con.commit()
    con.close()
    print(f"wrote {path} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
