"""AIMNet2 through mopt-rs: the profiles ``aimnet2`` and ``aimnet2-nse`` against the calculator
itself, the gradient against finite differences, the state of an open shell, and an
optimization by the command. Run where the package aimnet is installed and its models can be
had (downloaded on first use); skipped otherwise.

MultiOptPy: no counterpart (new in mopt-rs).

This file is part of mopt-py (the Python interface of mopt-rs).
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mopt_rs.calculators import Structure, create

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("aimnet") is None, reason="aimnet is not installed"
)

BOHR = 0.529177210903
WATER = (["O", "H", "H"], [[0.0, 0.0, 0.119], [0.0, 0.763, -0.477], [0.0, -0.763, -0.477]])
METHYL = (
    ["C", "H", "H", "H"],
    [[0.0, 0.0, 0.0], [1.08, 0.0, 0.0], [-0.54, 0.935, 0.0], [-0.54, -0.935, 0.0]],
)


def structure(molecule: tuple[list[str], list[list[float]]], **state: Any) -> Structure:
    symbols, xyz = molecule
    return Structure.build(symbols, (np.asarray(xyz) / BOHR).ravel().tolist(), **state)


def calculator(name: str) -> Any:
    try:
        calc = create(name)
        calc.evaluate_structure(structure(WATER))
    except Exception as e:  # the model could not be had (no network, no cache)
        pytest.skip(f"{name}: {e}")
    return calc


def test_the_profile_is_the_calculator_itself() -> None:
    from aimnet.calculators import AIMNet2ASE
    from ase import Atoms, units

    calc = calculator("aimnet2")
    for molecule, charge in [(WATER, 0), ((["O", "H"], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.97]]), -1)]:
        got = calc.evaluate_structure(structure(molecule, charge=charge))
        atoms = Atoms(molecule[0], positions=molecule[1])
        atoms.calc = AIMNet2ASE("aimnet2", charge=charge)
        energy = atoms.get_potential_energy() / units.Hartree
        gradient = -atoms.get_forces().ravel() * units.Bohr / units.Hartree
        assert abs(got.energy - energy) < 1e-6 * abs(energy)
        assert np.max(np.abs(np.asarray(got.gradient).ravel() - gradient)) < 1e-6
    # The gradient is that of the energy.
    x0 = structure(WATER).positions_bohr.ravel()
    g = np.asarray(calc.evaluate_structure(structure(WATER)).gradient).ravel()
    h = 1e-3
    for i in (0, 4, 8):
        e = []
        for sign in (1.0, -1.0):
            x = x0.copy()
            x[i] += sign * h
            e.append(calc.evaluate_structure(Structure.build(WATER[0], x.tolist())).energy)
        assert abs((e[0] - e[1]) / (2 * h) - g[i]) < 1e-4


def test_an_open_shell_needs_the_nse_model() -> None:
    nse = calculator("aimnet2-nse")
    with warnings.catch_warnings(record=True) as said:
        warnings.simplefilter("always")
        doublet = nse.evaluate_structure(structure(METHYL, multiplicity=2)).energy
        quartet = nse.evaluate_structure(structure(METHYL, multiplicity=4)).energy
    assert not [w for w in said if "mult" in str(w.message)], [str(w.message) for w in said]
    # The multiplicity reaches the model: the quartet of the methyl radical is far above.
    assert quartet - doublet > 0.05


def test_an_optimization_with_the_profile(tmp_path: Path) -> None:
    calculator("aimnet2")
    xyz = tmp_path / "water.xyz"
    xyz.write_text("3\n\nO 0.0 0.0 0.119\nH 0.0 0.80 -0.50\nH 0.0 -0.70 -0.45\n")
    run = subprocess.run(
        [str(Path(sys.executable).parent / "optmain"), str(xyz), "-os", "aimnet2", "-freq"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert run.returncode == 0, run.stdout[-2000:] + run.stderr[-2000:]
    assert "converged!!!" in run.stdout
    # The three vibrations of water, all real, in the range of the bend and the stretches.
    frequencies = [
        float(v)
        for line in run.stdout.splitlines()
        if line.startswith("Freq [cm^-1]")
        for v in line.split()[2:]
    ]
    assert len(frequencies) == 3 and all(f > 0 for f in frequencies), frequencies
    assert 1400 < frequencies[0] < 1800 and all(3500 < f < 4200 for f in frequencies[1:])
