"""The packages of the optional engines: which are installed, and how to install the others.

None of the engines of other packages is installed with mopt-rs (the built-in calculators need
none). An engine whose package is missing says so when it is created, with the command that
installs it, and ``python -m mopt_rs.calculators`` lists every engine with its package, the
version installed, or the command to install it.

MultiOptPy: no counterpart (new in mopt-rs).

This file is part of mopt-py (the Python interface of mopt-rs).
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import importlib.metadata as metadata
import importlib.util
from dataclasses import dataclass

from mopt_rs.calculators.base import CalculatorError

__all__ = ["EnginePackage", "PACKAGES", "installed_version", "report", "require"]


@dataclass(frozen=True)
class EnginePackage:
    """The package an engine needs."""

    #: Key of the engine (`require`).
    key: str
    #: How the engine is chosen on the command line.
    options: str
    #: The modules that must be importable (the first is the package).
    modules: tuple[str, ...]
    #: The command that installs it.
    install: str


def _extra(name: str) -> str:
    return f'pip install "mopt-py[{name}]"'


#: The engines of other packages and what they need.
PACKAGES: tuple[EnginePackage, ...] = (
    EnginePackage("pyscf", "-pyscf, -os pyscf", ("pyscf",), _extra("pyscf")),
    EnginePackage("tblite", "-xtb, -os tblite", ("tblite",), _extra("tblite")),
    EnginePackage("dxtb", "-dxtb, -os dxtb", ("dxtb", "torch"), _extra("dxtb")),
    EnginePackage("ase", "-os ase:module:Class (ASE's own calculators)", ("ase",), _extra("ase")),
    EnginePackage("psi4", "-os psi4", ("psi4",), "conda install -c conda-forge psi4"),
    EnginePackage(
        "openmm",
        "-os openmm, the lower level of -oniom",
        ("openmm",),
        "conda install -c conda-forge openmm (or pip install openmm)",
    ),
)


def _missing(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is None
    except (ImportError, ValueError):
        return True


def installed_version(module: str) -> str | None:
    """The version of the distribution that provides ``module``, ``"installed"`` when it is
    importable without one, or ``None`` when it is not installed."""
    if _missing(module):
        return None
    for name in metadata.packages_distributions().get(module, [module]):
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return "installed"


def require(key: str) -> None:
    """Raise :class:`CalculatorError` with the command that installs it when a module the
    engine ``key`` needs cannot be imported."""
    for package in PACKAGES:
        if package.key != key:
            continue
        for module in package.modules:
            if _missing(module):
                raise CalculatorError(
                    f"the engine {package.options} needs the package '{module}', which is not "
                    f"installed (the engines of other packages are optional): {package.install}"
                )
        return
    raise KeyError(f"no engine package '{key}'")


def report() -> str:
    """Every engine of another package: installed (with its version) or how to install it."""
    lines = [
        "Engines built into mopt-rs need nothing else: gfn1xtb, gfn0xtb, gfnff, lj, emt, eam, "
        "meam, tersoff.",
        "Engines of other packages (optional; install the ones you use):",
    ]
    width = max(len(p.options) for p in PACKAGES)
    for package in PACKAGES:
        found = [(m, installed_version(m)) for m in package.modules]
        missing = [m for m, v in found if v is None]
        if missing:
            state = f"not installed ({', '.join(missing)}): {package.install}"
        else:
            state = ", ".join(f"{m} {v}" for m, v in found)
        lines.append(f"  {package.options:<{width}}  {state}")
    return "\n".join(lines)
