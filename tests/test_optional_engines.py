"""The engines of other packages are optional: the package imports without them, and an engine
whose package is missing says which optional dependency to install when it is used.

MultiOptPy: no counterpart (new in mopt-rs).

This file is part of mopt-py (the Python interface of mopt-rs).
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

import pytest

from mopt_rs.calculators import CalculatorError, CalculatorOptions, Structure, create
from mopt_rs.calculators.packages import PACKAGES, installed_version, report

ROOT = Path(__file__).resolve().parents[1]


def water() -> Structure:
    b = 1.0 / 0.529177210903
    return Structure.build(["O", "H", "H"], [0.0, 0.0, 0.0, b, 0.0, 0.0, -0.3 * b, 0.95 * b, 0.0])


def test_each_engine_is_an_optional_dependency_of_its_own() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    required = " ".join(project["dependencies"]).lower()
    optional = project["optional-dependencies"]
    for package in ("pyscf", "tblite", "dxtb", "ase"):
        assert package not in required
        assert any(entry.startswith(package) for entry in optional[package])


@pytest.mark.parametrize(
    ("name", "blocked", "extra"),
    [
        ("pyscf", "pyscf", "pyscf"),
        ("tblite:GFN2-xTB", "tblite.interface", "tblite"),
        ("dxtb:GFN1-xTB", "dxtb", "dxtb"),
    ],
)
def test_a_missing_package_names_its_optional_dependency(
    monkeypatch: pytest.MonkeyPatch, name: str, blocked: str, extra: str
) -> None:
    # As if the package were not installed (an entry of None makes its import fail).
    monkeypatch.setitem(sys.modules, blocked, None)
    if blocked == "tblite.interface":
        monkeypatch.setitem(sys.modules, "tblite", None)
    # Said when the calculator is made, before a run starts, with the command that installs it.
    with pytest.raises(CalculatorError, match=rf"mopt-py\[{extra}\]"):
        create(name, CalculatorOptions(method="hf", basis="sto-3g") if extra == "pyscf" else None)
    # The engine itself says it too, where it is made without the registry.
    if extra == "pyscf":
        from mopt_rs.calculators.pyscf import PySCFEngine

        engine = PySCFEngine("hf", "sto-3g")
        with pytest.raises(CalculatorError, match=r"mopt-py\[pyscf\]"):
            engine.compute(water(), hessian=False)


def test_the_list_of_the_engines_says_what_is_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "pyscf", None)
    text = report()
    assert "gfn1xtb" in text
    line = next(line for line in text.splitlines() if "-pyscf" in line)
    assert 'not installed (pyscf): pip install "mopt-py[pyscf]"' in line
    assert all(p.options in text for p in PACKAGES)
    # numpy is always there, with its version.
    assert installed_version("numpy") not in (None, "")
