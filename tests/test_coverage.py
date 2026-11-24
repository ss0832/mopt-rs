"""Every MultiOptPy 1.21.2 feature must be accounted for in mopt-rs.

``tests/coverage/multioptpy_inventory.json`` (from ``tools/coverage/inventory.py``) lists all
modules, classes, functions, methods, and command-line options of MultiOptPy;
``tools/coverage/mapping.json`` says where each lives in mopt-rs, that it is not implemented
yet, or that it is not provided by decision. This test fails for items without an entry. Items
still pending are listed in ``docs/source/coverage.md``; they are no condition of a release.

This file is part of mopt-py (the Python interface of mopt-rs).
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load() -> tuple[list[dict[str, str]], dict[str, dict[str, str]]]:
    inv = json.loads((ROOT / "tests/coverage/multioptpy_inventory.json").read_text())
    mapping = json.loads((ROOT / "tools/coverage/mapping.json").read_text())
    return inv["items"], mapping["entries"]


def _entry(item_id: str, entries: dict[str, dict[str, str]]) -> dict[str, str] | None:
    best = None
    for key in entries:
        if item_id.startswith(key) and (best is None or len(key) > len(best)):
            best = key
    return None if best is None else entries[best]


def test_every_multioptpy_feature_is_mapped() -> None:
    items, entries = _load()
    unmapped = [it["id"] for it in items if _entry(it["id"], entries) is None]
    assert not unmapped, f"{len(unmapped)} MultiOptPy items without a mapping: {unmapped[:20]}"
    for key, e in entries.items():
        # dropped: not provided by decision (the reason is in "where").
        assert e["status"] in {"done", "pending", "dropped"}, key
        assert e.get("where"), key

