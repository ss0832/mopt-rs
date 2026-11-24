"""PySCF with restricted open-shell (ROHF / ROKS) and generalized (GHF / GKS) wavefunctions.

MultiOptPy: `Calculator/pyscf_calculation_tools.py` (restricted and unrestricted only).

This file is part of mopt-py (the Python interface of mopt-rs).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mopt_rs import _core
from mopt_rs.calculators import CalculatorOptions, create
from mopt_rs.calculators.base import Structure

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("pyscf") is None, reason="PySCF not installed"
)

ANG = _core.ANGSTROM_TO_BOHR
WATER = np.array([[0.0, 0.0, 0.119], [0.0, 0.763, -0.477], [0.0, -0.763, -0.477]]) * ANG


def pyscf(method: str = "hf", **extra: Any) -> Any:
    return create("pyscf", CalculatorOptions(method=method, basis="sto-3g", extra=extra))


def radical(distance: float = 1.83) -> Structure:
    return Structure.build(["O", "H"], [[0.0, 0.0, 0.0], [0.0, 0.0, distance]], multiplicity=2)


def numerical_gradient(calc: Any, structure: Structure, dof: int, step: float = 1e-3) -> float:
    up = calc.evaluate_structure(structure.displaced(dof, step)).energy
    down = calc.evaluate_structure(structure.displaced(dof, -step)).energy
    return float((up - down) / (2.0 * step))


def test_restricted_open_shell() -> None:
    ro = pyscf(reference="ro")
    u = pyscf()
    assert "reference: restricted" in ro.describe()
    ev = ro.evaluate_structure(radical())
    # The restricted open-shell energy lies above the unrestricted one, and no <S^2> is
    # reported for it (it is that of the spin state by construction).
    assert ev.energy > u.evaluate_structure(radical()).energy - 1e-10
    assert ev.spin_square is None
    assert u.evaluate_structure(radical()).spin_square == pytest.approx(0.75, abs=0.02)
    # The analytic gradient is the slope of the energy.
    assert ev.gradient[1, 2] == pytest.approx(numerical_gradient(ro, radical(), 5), abs=2e-6)
    # A closed shell is plain RHF with it.
    closed = Structure.build(["O", "H", "H"], WATER)
    assert ro.evaluate_structure(closed).energy == pytest.approx(
        u.evaluate_structure(closed).energy, abs=1e-8
    )
    # One set of orbitals, filled to different levels by the two spins.
    orbitals = ro.orbitals(radical())
    assert orbitals is not None and len(orbitals["sets"]) == 2
    alpha, beta = orbitals["sets"]
    assert np.array_equal(alpha["coefficients"], beta["coefficients"])
    assert (alpha["occupations"].sum(), beta["occupations"].sum()) == (5.0, 4.0)
    # The Hessian comes from differences of the gradients.
    hessian = ro.evaluate_structure(radical(), need_hessian=True).hessian
    assert hessian is not None and hessian.shape == (6, 6)
    assert np.allclose(hessian, hessian.T, atol=1e-6)
    assert hessian[5, 5] > 0.1


def test_restricted_open_shell_kohn_sham() -> None:
    # The default grid: on a coarse one the analytic gradient, which leaves out the
    # response of the grid, and the slope of the energy differ by more.
    ro = pyscf("lda,vwn", reference="roks")
    ev = ro.evaluate_structure(radical())
    assert ev.gradient[1, 2] == pytest.approx(numerical_gradient(ro, radical(), 5), abs=5e-5)
    # The two forces cancel up to what the integration grid leaves.
    assert ev.gradient[0, 2] == pytest.approx(-ev.gradient[1, 2], abs=1e-5)


def test_generalized_hartree_fock() -> None:
    g = pyscf(reference="g")
    r = pyscf()
    assert "reference: generalized" in g.describe()
    closed = Structure.build(["O", "H", "H"], WATER)
    ev = g.evaluate_structure(closed)
    exact = r.evaluate_structure(closed)
    # For a closed shell the generalized solution is the restricted one; the gradient
    # comes from differences of the energy.
    assert ev.energy == pytest.approx(exact.energy, abs=1e-8)
    assert np.allclose(ev.gradient, exact.gradient, atol=2e-5)
    assert ev.spin_square == pytest.approx(0.0, abs=1e-6)
    # An open shell is at least as low as the unrestricted solution.
    open_shell = g.evaluate_structure(radical())
    assert open_shell.energy < r.evaluate_structure(radical()).energy + 1e-7
    assert open_shell.spin_square is not None
    # Spin orbitals that mix the spins are not handed out as sets of one spin.
    assert g.orbitals(closed) is None


def test_generalized_kohn_sham() -> None:
    hydrogen = Structure.build(["H", "H"], [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]])
    g = pyscf("lda,vwn", reference="gks", collinear="ncol", grid_level=1)
    r = pyscf("lda,vwn", grid_level=1)
    ev = g.evaluate_structure(hydrogen)
    exact = r.evaluate_structure(hydrogen)
    assert ev.energy == pytest.approx(exact.energy, abs=1e-6)
    assert np.allclose(ev.gradient, exact.gradient, atol=5e-5)


def test_what_does_not_go_together() -> None:
    with pytest.raises(ValueError, match="contradict"):
        create(
            "pyscf",
            CalculatorOptions(
                method="hf", basis="sto-3g", unrestricted=True, extra={"reference": "ro"}
            ),
        )
    with pytest.raises(ValueError, match="reference"):
        pyscf(reference="multi")
    with pytest.raises(ValueError, match="collinear"):
        pyscf(reference="g", collinear="sideways")
    excited = Structure.build(
        ["O", "H"], [[0.0, 0.0, 0.0], [0.0, 0.0, 1.83]], multiplicity=2, excited_state=1
    )
    with pytest.raises(Exception, match="excited states"):
        pyscf(reference="ro").evaluate_structure(excited)


def test_optimization_from_the_command_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "oh.xyz").write_text("2\n\nO 0.0 0.0 0.0\nH 0.0 0.0 1.10\n")
    arguments = ["oh.xyz", "-func", "hf", "-bs", "sto-3g", "-spin", "2", "-opt", "rsirfo_bfgs"]
    lengths = {}
    for reference in ("ro", "g"):
        code = _core.run_cli("optmain", [*arguments, "-calc_opt", f"reference={reference}"])
        assert code == 0, reference
        newest = max(tmp_path.rglob("oh_optimized.xyz"), key=lambda p: p.stat().st_mtime)
        z = [float(line.split()[3]) for line in newest.read_text().splitlines()[2:]]
        lengths[reference] = abs(z[1] - z[0])
        log = (newest.parent / "mopt.log").read_text()
        assert f"reference: {'restricted' if reference == 'ro' else 'generalized'}" in log
        # <S^2> is shown at every iteration for the generalized wavefunction only.
        assert ("<S**2>" in log) == (reference == "g")
    # Both find the minimum of the minimal-basis radical near 1.0 Angstrom.
    for value in lengths.values():
        assert 0.9 < value < 1.15, lengths
