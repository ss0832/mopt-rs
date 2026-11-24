"""Where an engine's program comes from, for run logs: the distribution, its version, and the
file Python imported it from. By default every engine uses what is built into mopt-rs or
installed with pip; ``software_path.conf`` entries only override paths, and engines say when
they used one (``notes``).

MultiOptPy: no counterpart (new in mopt-rs).

This file is part of mopt-py (the Python interface of mopt-rs).
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import importlib
import importlib.metadata as metadata
import sys
from typing import Any

__all__ = ["module_origin", "object_origin"]


def _distribution(top: str) -> tuple[str, str | None]:
    """The installed distribution that provides the top-level module ``top``."""
    names = metadata.packages_distributions().get(top) or [top]
    for name in names:
        try:
            return name, metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return names[0], None


def module_origin(module_name: str) -> str:
    """``package version (path)`` of the top-level package of ``module_name``."""
    top = module_name.split(".")[0]
    module = sys.modules.get(top)
    if module is None:
        try:
            module = importlib.import_module(top)
        except ImportError:
            return f"{top} (not installed)"
    dist, version = _distribution(top)
    if version is None:
        version = getattr(module, "__version__", None)
    path = getattr(module, "__file__", None)
    if path is None:
        locations = list(getattr(module, "__path__", []))
        path = locations[0] if locations else None
    text = dist if version is None else f"{dist} {version}"
    return text if path is None else f"{text} ({path})"


def object_origin(obj: Any) -> str:
    """``module.Name from package version (path)`` for a class, function, or instance."""
    target = obj if isinstance(obj, type) or callable(obj) else type(obj)
    module = getattr(target, "__module__", None) or "unknown"
    name = getattr(target, "__qualname__", None) or getattr(target, "__name__", "?")
    return f"{module}.{name} from {module_origin(module)}"
