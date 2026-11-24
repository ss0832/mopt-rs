"""Every source file says which package it belongs to and names its MultiOptPy counterpart.

``tools/headers/add_headers.py`` adds the line "This file is part of mopt-rs." (or mopt-py);
the counterpart line (``MultiOptPy: ...`` or "no counterpart (new in mopt-rs)") is written by
hand in the header of every file of the library, the binding crate, and the Python package.

This file is part of mopt-py (the Python interface of mopt-rs).
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]


def _tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "add_headers", ROOT / "tools/headers/add_headers.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _header(path: Path) -> str:
    return "\n".join(path.read_text(encoding="utf-8").split("\n")[:80])


def test_every_source_file_names_its_package() -> None:
    tool = _tool()
    files = tool.source_files(ROOT)
    assert len(files) > 200
    missing = []
    for path in files:
        rel = path.relative_to(ROOT).as_posix()
        if not tool.has_line(path.read_text(encoding="utf-8"), tool.sentence(rel)):
            missing.append(rel)
    assert not missing, f"run tools/headers/add_headers.py: {missing}"


def test_library_files_name_their_multioptpy_counterpart() -> None:
    missing = []
    for top in ["crates/mopt-rs/src", "crates/mopt-py/src", "python/mopt_rs"]:
        for path in sorted((ROOT / top).rglob("*")):
            if path.suffix in {".rs", ".py"} and "__pycache__" not in path.parts:
                header = _header(path)
                if "MultiOptPy:" not in header and "no MultiOptPy" not in header:
                    missing.append(path.relative_to(ROOT).as_posix())
    assert not missing, f"no 'MultiOptPy: ...' line in the header of: {missing}"
