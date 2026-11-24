"""ASE-calculator profiles from files: a machine-learned potential by name without code.

This file is part of mopt-py (the Python interface of mopt-rs).
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import fake_ase
import fake_nnp
from mopt_rs import _core
from mopt_rs.calculators import CalculatorOptions, Structure, create, create_engine
from mopt_rs.calculators.profile_files import (
    load_profile_file,
    load_profile_files,
    profile_from_data,
    profile_paths,
)
from mopt_rs.calculators.profiles import PROFILES

BOHR_TO_ANGSTROM = _core.BOHR_TO_ANGSTROM
HARTREE_TO_EV = _core.HARTREE_TO_EV
WATER = np.array([[0.0, 0.0, 0.2], [0.0, 1.4, -0.9], [0.0, -1.4, -0.9]])

TWO_STEP = {
    "factory": "fake_nnp:Potential",
    "args": [
        {
            "factory": "fake_nnp:load_model",
            "kwargs": {
                "name": {"option": "model", "default": "small"},
                "device": {"option": "device", "default": "cpu"},
            },
        }
    ],
    "kwargs": {"device": {"option": "device", "default": "cpu"}},
    "description": "stand-in potential",
}
FROM_PATH = {
    "factory": "fake_nnp:Potential",
    "args": [{"factory": "fake_nnp:load_model", "kwargs": {"path": {"software_path": True}}}],
}
OWN_CALCULATOR = {"factory": "fake_nnp:Zoo", "args": ["large"], "call": "ase"}


@pytest.fixture(autouse=True)
def _clean(monkeypatch: pytest.MonkeyPatch) -> Any:
    try:
        import ase  # noqa: F401
    except ImportError:
        monkeypatch.setitem(sys.modules, "ase", fake_ase.module())
    before = dict(PROFILES)
    fake_nnp.LOADS.clear()
    yield
    PROFILES.clear()
    PROFILES.update(before)


def energy(calc: Any, x: np.ndarray) -> float:
    return float(calc.evaluate_structure(Structure.build(["O", "H", "H"], x)).energy)


def harmonic(k: float, x: np.ndarray, shift: float = 0.0) -> float:
    return 0.5 * k * float(np.sum((x * BOHR_TO_ANGSTROM - shift) ** 2)) / HARTREE_TO_EV


def test_profiles_from_data_build_calculators_in_two_steps() -> None:
    PROFILES["nnp"] = profile_from_data("nnp", TWO_STEP)
    calc = create("nnp")
    assert energy(calc, WATER) == pytest.approx(harmonic(1.0, WATER))
    assert energy(calc, WATER + 0.1) == pytest.approx(harmonic(1.0, WATER + 0.1))
    # One model for both structures.
    assert fake_nnp.LOADS == [{"name": "small", "device": "cpu", "path": None}]
    # Options named by the profile go where it says; the others are calculator keywords.
    fake_nnp.LOADS.clear()
    options = CalculatorOptions(extra={"model": "large", "device": "cuda", "shift": 0.5})
    calc = create("nnp", options)
    assert energy(calc, WATER) == pytest.approx(harmonic(2.0, WATER, 0.5))
    assert fake_nnp.LOADS == [{"name": "large", "device": "cuda", "path": None}]
    # Forces, and a Hessian from finite differences of the forces.
    result = calc.evaluate_structure(Structure.build(["O", "H", "H"], WATER), need_hessian=True)
    k_au = 2.0 / HARTREE_TO_EV * BOHR_TO_ANGSTROM**2
    assert result.hessian is not None and np.allclose(result.hessian, k_au * np.eye(9), atol=1e-8)
    # A keyword the calculator does not have is an error naming it.
    with pytest.raises(Exception, match="no_such_keyword"):
        energy(create("nnp", CalculatorOptions(extra={"no_such_keyword": 1})), WATER)


def test_model_paths_and_calculators_returned_by_the_model() -> None:
    PROFILES["nnp_file"] = profile_from_data("nnp_file", FROM_PATH)
    with pytest.raises(Exception, match="software_path.conf"):
        energy(create("nnp_file"), WATER)
    options = CalculatorOptions(software_paths={"nnp_file": "/models/a.model"})
    assert energy(create("nnp_file", options), WATER) == pytest.approx(harmonic(1.0, WATER))
    assert fake_nnp.LOADS[-1]["path"] == "/models/a.model"
    PROFILES["zoo"] = profile_from_data("zoo", OWN_CALCULATOR)
    assert energy(create("zoo"), WATER) == pytest.approx(harmonic(2.0, WATER))
    assert not create_engine("zoo").capabilities.analytic_hessian


def test_profile_files_are_found_and_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.delenv("MOPT_PROFILES", raising=False)
    assert profile_paths() == []
    (tmp_path / "config" / "mopt-rs").mkdir(parents=True)
    user = tmp_path / "config" / "mopt-rs" / "profiles.json"
    user.write_text(json.dumps({"profiles": {"My-NNP": TWO_STEP, "zoo": OWN_CALCULATOR}}))
    local = tmp_path / "mopt_profiles.json"
    local.write_text(json.dumps({"profiles": {"zoo": TWO_STEP}}))
    extra = tmp_path / "extra.json"
    extra.write_text(json.dumps({"profiles": {"other": FROM_PATH}}))
    monkeypatch.setenv("MOPT_PROFILES", str(extra))
    assert profile_paths() == [user, Path("mopt_profiles.json"), extra]
    assert sorted(load_profile_file(user)) == ["my-nnp", "zoo"]
    assert sorted(load_profile_files()) == ["my-nnp", "other", "zoo", "zoo"]
    # The working directory wins over the user's file; names are case-insensitive.
    assert energy(create("zoo"), WATER) == pytest.approx(harmonic(1.0, WATER))
    assert energy(create("MY-NNP"), WATER) == pytest.approx(harmonic(1.0, WATER))
    assert load_profile_files() == []  # unchanged files are not read again


def test_a_profile_file_is_all_a_command_needs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.delenv("MOPT_PROFILES", raising=False)
    (tmp_path / "mopt_profiles.json").write_text(json.dumps({"profiles": {"cmd-nnp": TWO_STEP}}))
    (tmp_path / "w.xyz").write_text("3\n\nO 0.0 0.0 0.1\nH 0.0 0.7 -0.4\nH 0.0 -0.7 -0.4\n")
    args = ["w.xyz", "-os", "cmd-nnp", "-opt", "rsirfo_bfgs", "-calc_opt", "model=large"]
    assert _core.run_cli("optmain", args) == 0
    assert fake_nnp.LOADS == [{"name": "large", "device": "cpu", "path": None}]
    out = list(tmp_path.glob("*/w_OPT_ASE_*/w_optimized.xyz"))
    assert len(out) == 1
    rows = [line.split() for line in out[0].read_text().splitlines()[2:]]
    assert np.allclose([[float(v) for v in r[1:]] for r in rows], 0.0, atol=1e-2)


def test_mistakes_in_profile_files_are_reported(tmp_path: Path) -> None:
    mistakes: list[tuple[dict[str, Any], str]] = [
        ({"kwargs": {}}, "no 'factory'"),
        ({"factory": "fake_nnp:Potential", "colour": 1}, "unknown keys"),
        ({"factory": "fake_nnp:Potential", "state": "triplet"}, "state"),
        ({"factory": "fake_nnp:Potential", "state": {"via": "letter"}}, "via"),
    ]
    for data, message in mistakes:
        with pytest.raises(ValueError, match=message):
            profile_from_data("bad", data)
    PROFILES["needs"] = profile_from_data(
        "needs", {"factory": "fake_nnp:load_model", "kwargs": {"name": {"option": "model"}}}
    )
    with pytest.raises(Exception, match="-calc_opt model"):
        energy(create("needs"), WATER)
    broken = tmp_path / "broken.json"
    broken.write_text("{not json")
    with pytest.raises(ValueError, match="broken.json"):
        load_profile_file(broken)
    broken.write_text(json.dumps(["profiles"]))
    with pytest.raises(ValueError, match="profiles"):
        load_profile_file(broken)
