"""Hessians of ASE calculators: the routes that are found by themselves (``get_hessian``,
``calculate_hessian``, a method ``hessian``, the ``hessian`` property, ``results["hessian"]``),
methods named by the user (numerical ones among them), the layouts and units of the arrays,
the treatment of asymmetric matrices, the check against differences of the forces, the
fallback to differences, the keys of ``-calc_opt``, and ASE's own ``ForceField``.

The calculators are those of :mod:`fake_ase` (a quadratic surface with a general symmetric
matrix, so that a wrong layout shows); ASE's ``Atoms`` or its stand-in carries them.

This file is part of mopt-py (the Python interface of mopt-rs).
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import importlib.util
import sys
import warnings
from typing import Any

import numpy as np
import pytest

import fake_ase
from mopt_rs import _core
from mopt_rs.calculators import CalculatorOptions, EngineCalculator, Structure, create
from mopt_rs.calculators._bridge import analytic_hessian, calculator_info
from mopt_rs.calculators.ase import (
    FINITE_DIFFERENCES,
    HESSIAN_LAYOUTS,
    HESSIAN_UNITS,
    AseEngine,
    hessian_unit_factor,
    square_hessian,
)
from mopt_rs.calculators.base import CalculatorError

HAS_ASE = importlib.util.find_spec("ase") is not None
BOHR = _core.BOHR_TO_ANGSTROM
TO_AU = BOHR**2 / _core.HARTREE_TO_EV
WATER = np.array([[0.0, 0.0, 0.2214], [0.0, 1.4309, -0.8865], [0.0, -1.4309, -0.8865]])


@pytest.fixture(autouse=True)
def an_ase(monkeypatch: pytest.MonkeyPatch) -> None:
    """ASE, or its stand-in where it is not installed."""
    if not HAS_ASE:
        monkeypatch.setitem(sys.modules, "ase", fake_ase.module())


def _matrix(n3: int, seed: int = 3) -> np.ndarray:
    """A symmetric matrix with distinct entries (eV/Angstrom^2)."""
    a = np.random.default_rng(seed).standard_normal((n3, n3))
    return a @ a.T + n3 * np.eye(n3)


K = _matrix(9)
X0 = (WATER * BOHR).ravel()
STRUCTURE = Structure.build(["O", "H", "H"], WATER + 0.03)


def _engine(cls: type, **kwargs: Any) -> AseEngine:
    options = {k: kwargs.pop(k) for k in ("layout", "scale", "skew") if k in kwargs}
    return AseEngine(cls, kwargs={"x0": X0, "k": K, **options}, electrons=False, **kwargs)


def _hessian(engine: AseEngine) -> tuple[np.ndarray, tuple[str, ...]]:
    ev = EngineCalculator(engine).evaluate_structure(STRUCTURE, need_hessian=True)
    assert ev.hessian is not None
    return np.asarray(ev.hessian), ev.notes


@pytest.mark.parametrize(
    ("cls", "route"),
    [
        (fake_ase.QuadraticGetHessian, "get_hessian()"),
        (fake_ase.QuadraticCalculateHessian, "calculate_hessian()"),
        (fake_ase.QuadraticHessianMethod, "hessian()"),
        (fake_ase.QuadraticHessianProperty, "hessian property"),
        (fake_ase.QuadraticResults, "results['hessian']"),
    ],
)
def test_every_route_is_found_by_itself(cls: type, route: str) -> None:
    engine = _engine(cls)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        h, notes = _hessian(engine)
    assert np.allclose(h, K * TO_AU, rtol=0.0, atol=1e-12)
    assert engine.hessian_route == route and engine.capabilities.analytic_hessian
    # Said in the run log once, and in the description.
    assert any(f"Hessian from {route}, layout (3N, 3N)" in n for n in notes), notes
    assert f"Hessian: {route}" in engine.describe()
    # No differences of the forces were taken: one calculation of the forces.
    calc = engine.calculator(STRUCTURE.state)
    assert calc.force_calls == 1 and calc.hessian_calls >= 1


def test_the_order_of_the_routes() -> None:
    # A method that fails comes first; the results that hold the Hessian are used after it,
    # and the run log says what was tried before.
    engine = _engine(fake_ase.QuadraticBrokenGetHessian)
    assert engine.hessian_route == "get_hessian()"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        h, notes = _hessian(engine)
    assert np.allclose(h, K * TO_AU, atol=1e-12)
    assert engine.hessian_route == "results['hessian']"
    assert any("before it: get_hessian() failed" in n for n in notes), notes


@pytest.mark.parametrize("layout", HESSIAN_LAYOUTS)
@pytest.mark.parametrize("atoms", [2, 3, 4])
def test_every_layout_gives_the_same_matrix(layout: str, atoms: int) -> None:
    # For three atoms (N, 3, N, 3) and (N, N, 3, 3) have the same shape: told apart by symmetry.
    n3 = 3 * atoms
    k = _matrix(n3, seed=atoms)
    h, found = square_hessian(fake_ase.arranged(k, layout), atoms)
    assert found == layout and np.array_equal(h, k)
    structure = Structure.build(["H"] * atoms, np.arange(n3, dtype=np.float64).reshape(-1, 3))
    engine = AseEngine(
        fake_ase.QuadraticGetHessian,
        kwargs={"x0": np.zeros(n3), "k": k, "layout": layout},
        electrons=False,
    )
    ev = EngineCalculator(engine).evaluate_structure(structure, need_hessian=True)
    assert np.allclose(ev.hessian, k * TO_AU, atol=1e-12)
    # Any other shape is refused.
    with pytest.raises(ValueError, match="expected one of"):
        square_hessian(np.zeros((n3, n3 + 1)), atoms)


def test_a_matrix_of_the_wrong_size_falls_back_to_differences() -> None:
    engine = AseEngine(fake_ase.MisshapenHessian, kwargs={"x0": X0.reshape(3, 3)}, electrons=False)
    with pytest.warns(RuntimeWarning, match="finite differences of the forces"):
        h, notes = _hessian(engine)
    assert np.allclose(h, 2.0 * TO_AU * np.eye(9), atol=1e-8)
    assert engine.hessian_route == FINITE_DIFFERENCES
    assert any("shape (2, 2) for 3 atoms" in n for n in notes), notes


@pytest.mark.parametrize("units", list(HESSIAN_UNITS))
def test_the_units_of_plain_arrays(units: str) -> None:
    # The calculator hands out its matrix in `units`; declared, it is converted.
    engine = _engine(
        fake_ase.QuadraticGetHessian, scale=1.0 / HESSIAN_UNITS[units], hessian_units=units
    )
    h, _ = _hessian(engine)
    assert np.allclose(h, K * TO_AU, rtol=1e-12, atol=0.0)
    assert engine.hessian_units == units
    if units != "eV/Angstrom^2":
        assert f"in {units}" in engine.describe()


def test_the_units_are_checked_and_follow_the_constants() -> None:
    assert hessian_unit_factor("hartree/bohr^2") == HESSIAN_UNITS["Hartree/Bohr^2"]
    assert HESSIAN_UNITS["Hartree/Bohr^2"] == pytest.approx(
        _core.HARTREE_TO_EV / _core.BOHR_TO_ANGSTROM**2, rel=1e-15
    )
    assert HESSIAN_UNITS["kcal/mol/Angstrom^2"] == pytest.approx(
        _core.HARTREE_TO_EV / _core.HARTREE_TO_KCAL_MOL, rel=1e-15
    )
    for wrong in ("eV/A^2", "hartree", "au"):
        with pytest.raises(ValueError, match="hessian_units"):
            hessian_unit_factor(wrong)
    with pytest.raises(ValueError, match="hessian_units"):
        _engine(fake_ase.QuadraticGetHessian, hessian_units="Ry/Bohr^2")


def test_asymmetric_matrices() -> None:
    # Rounding: symmetrized without a word.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        h, notes = _hessian(_engine(fake_ase.QuadraticGetHessian, skew=1e-7))
    assert np.allclose(h, K * TO_AU, atol=1e-10)
    assert not any("not symmetric" in n for n in notes)
    # Larger: the symmetric part, with the measured asymmetry in the warning and the log.
    with pytest.warns(RuntimeWarning, match=r"not symmetric \(max \|H - H\^T\| / max \|H\| = "):
        h, notes = _hessian(_engine(fake_ase.QuadraticGetHessian, skew=1e-3))
    assert np.allclose(h, K * TO_AU, atol=1e-12)
    assert any("its symmetric part is used" in n for n in notes)
    # No Hessian at all: not used, differences instead.
    engine = _engine(fake_ase.QuadraticGetHessian, skew=1.0)
    with pytest.warns(RuntimeWarning, match="finite differences"):
        h, _ = _hessian(engine)
    assert np.allclose(h, K * TO_AU, atol=1e-7)
    assert "not symmetric" in engine.describe() and engine.hessian_route == FINITE_DIFFERENCES


def test_the_check_against_differences_of_the_forces() -> None:
    engine = _engine(fake_ase.QuadraticGetHessian, hessian_check=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        h, notes = _hessian(engine)
    line = next(n for n in notes if "hessian_check" in n)
    deviation = float(line.rsplit("deviation ", 1)[1].split()[0].rstrip(";"))
    assert deviation < 1e-6, line
    # Six more calculations of the forces: two along each of three directions.
    assert engine.calculator(STRUCTURE.state).force_calls == 1 + 6
    # A matrix in Hartree/Bohr^2 taken for eV/Angstrom^2: the unit that explains it is named.
    wrong = _engine(
        fake_ase.QuadraticGetHessian,
        scale=1.0 / HESSIAN_UNITS["Hartree/Bohr^2"],
        hessian_check=True,
    )
    with pytest.warns(RuntimeWarning, match=r"hessian_units=Hartree/Bohr\^2"):
        _hessian(wrong)
    # Another matrix (here the entries of K in another order) is no unit: said so.
    swapped = fake_ase.QuadraticGetHessian(X0, K)
    swapped.handed_out = lambda: K[::-1, ::-1].copy()  # type: ignore[method-assign]
    other = AseEngine(swapped, electrons=False, hessian_check=True)
    with pytest.warns(RuntimeWarning, match="no unit explains it"):
        _hessian(other)


def test_numerical_methods_only_by_name() -> None:
    # The automatic route never calls a method that says it is numerical.
    auto = _engine(fake_ase.QuadraticNumerical)
    assert auto.hessian_route == FINITE_DIFFERENCES and not auto.capabilities.analytic_hessian
    h, _ = _hessian(auto)
    assert np.allclose(h, K * TO_AU, atol=1e-7)
    assert auto.calculator(STRUCTURE.state).hessian_calls == 0
    # Named, it is used, as a Hessian of the program that is not analytic.
    named = _engine(fake_ase.QuadraticNumerical, hessian="get_numerical_hessian")
    caps = named.capabilities
    assert caps.numerical_hessian and not caps.analytic_hessian
    h, _ = _hessian(named)
    assert np.allclose(h, K * TO_AU, atol=1e-12)
    assert named.calculator(STRUCTURE.state).hessian_calls == 1
    # The routes by name.
    for choice, route in (("property", "hessian property"), ("results", "results['hessian']")):
        assert _engine(fake_ase.QuadraticHessianProperty, hessian=choice).hessian_route == route
    with pytest.raises(ValueError, match="name of a method"):
        _engine(fake_ase.QuadraticGetHessian, hessian="get hessian")


def test_analytic_is_required_and_numerical_is_differences() -> None:
    engine = AseEngine(fake_ase.Harmonic, kwargs={"x0": X0.reshape(3, 3)}, electrons=False, hessian="analytic")
    assert engine.capabilities.analytic_hessian
    with pytest.raises(CalculatorError, match="hessian=analytic"):
        _hessian(engine)
    numerical = _engine(fake_ase.QuadraticGetHessian, hessian="numerical")
    assert numerical.hessian_route == FINITE_DIFFERENCES
    h, _ = _hessian(numerical)
    assert np.allclose(h, K * TO_AU, atol=1e-7)
    assert numerical.calculator(STRUCTURE.state).hessian_calls == 0


def test_the_keys_of_calc_opt() -> None:
    def made(**extra: Any) -> EngineCalculator:
        return create(
            "ase:fake_ase:QuadraticGetHessian",
            CalculatorOptions(extra={"x0": X0.tolist(), "k": K.tolist(), **extra}),
        )

    assert made().engine.hessian_route == "get_hessian()"
    assert made(hessian="numerical").engine.hessian_route == FINITE_DIFFERENCES
    # The older spelling of the ASE route.
    assert made(hessian_method="numerical").engine.hessian_route == FINITE_DIFFERENCES
    assert made(hessian="get_hessian", hessian_method="get_hessian").engine.hessian_route == (
        "get_hessian()"
    )
    with pytest.raises(ValueError, match="disagree"):
        made(hessian="auto", hessian_method="numerical")
    calc = made(hessian_units="Hartree/Bohr^2", hessian_check="true")
    assert calc.engine.hessian_units == "Hartree/Bohr^2" and calc.engine.hessian_check
    with pytest.raises(ValueError, match="hessian_check"):
        made(hessian_check="maybe")
    # The differences of every engine without a Hessian of its own.
    calc = made(hessian="numerical", hessian_differences="o1numhess", hessian_step=0.002)
    assert calc.hessian_method == "o1numhess" and calc.hessian_step_bohr == 0.002
    assert "hessian_step" not in calc.engine.kwargs and "hessian" not in calc.engine.kwargs
    with pytest.raises(ValueError, match="hessian_differences"):
        made(hessian_differences="forward")
    # The Python field keeps its older name as well.
    old = create("lj", CalculatorOptions(hessian_method="o1numhess"))
    assert old.hessian_method == "o1numhess"
    # Engines that are not ASE calculators take auto, analytic, and numerical.
    lj = create("lj", CalculatorOptions(extra={"hessian": "numerical"}))
    assert not lj.engine.capabilities.analytic_hessian
    assert create("lj", CalculatorOptions(extra={"hessian": "analytic"})).engine
    with pytest.raises(ValueError, match="ASE calculators only"):
        create("lj", CalculatorOptions(extra={"hessian": "get_hessian"}))
    with pytest.raises(ValueError, match="settings of ASE calculators"):
        create("lj", CalculatorOptions(extra={"hessian_units": "Hartree/Bohr^2"}))


def test_the_rust_side_asks_whether_the_hessian_is_analytic() -> None:
    calc = EngineCalculator(_engine(fake_ase.QuadraticGetHessian))
    assert calculator_info(calc)["analytic_hessian"] and analytic_hessian(calc)
    plain = EngineCalculator(AseEngine(fake_ase.Harmonic, kwargs={"x0": X0.reshape(3, 3)}, electrons=False))
    assert not calculator_info(plain)["analytic_hessian"]
    # After a fallback the answer changes.
    failing = EngineCalculator(
        AseEngine(fake_ase.FailingHessian, kwargs={"x0": X0.reshape(3, 3)}, electrons=False)
    )
    assert analytic_hessian(failing)
    with pytest.warns(RuntimeWarning):
        failing.evaluate_structure(STRUCTURE, need_hessian=True)
    assert not analytic_hessian(failing)


def test_a_periodic_cell_gets_the_hessian_of_the_positions() -> None:
    cell = np.diag([12.0, 12.0, 12.0])
    periodic = Structure.build(["O", "H", "H"], WATER + 0.03, cell_bohr=cell)
    engine = _engine(fake_ase.QuadraticGetHessian)
    ev = EngineCalculator(engine).evaluate_structure(periodic, need_hessian=True)
    # Positions only: the strain part comes from differences on the Rust side.
    assert np.asarray(ev.hessian).shape == (9, 9)
    assert np.allclose(ev.hessian, K * TO_AU, atol=1e-12)


@pytest.mark.skipif(not HAS_ASE, reason="ASE is not installed")
def test_ase_force_field_through_the_automatic_route() -> None:
    # ASE's own force field (3.26): its get_hessian asks for a property it does not list, and
    # the Hessian every calculation leaves in its results has only the block of the first atom
    # of each bond with the second filled (the outer and the inner loop over the atoms of a
    # term share one iterator, which the inner loop uses up), so it is not symmetric. Both routes are tried, the matrix is refused, and the
    # Hessian comes from differences of the forces, as it would without any route.
    from ase.calculators.ff import ForceField
    from ase.utils.ff import Bond

    def bonds() -> ForceField:
        return ForceField(bonds=[Bond(0, 1, 30.0, 0.96), Bond(0, 2, 30.0, 0.96)])

    # The force field takes minimum images and so needs a cell: a box far larger than water.
    boxed = Structure.build(["O", "H", "H"], WATER + 0.03, cell_bohr=np.diag([40.0] * 3))
    engine = AseEngine(bonds(), electrons=False)
    assert engine.hessian_route == "get_hessian()"
    with pytest.warns(RuntimeWarning, match=r"results\['hessian'\] failed: not symmetric"):
        ev = EngineCalculator(engine).evaluate_structure(boxed, need_hessian=True)
    assert engine.hessian_route == FINITE_DIFFERENCES
    assert "get_hessian() failed" in engine.describe()
    numerical = EngineCalculator(
        AseEngine(bonds(), electrons=False, hessian="numerical")
    ).evaluate_structure(boxed, need_hessian=True)
    assert np.allclose(ev.hessian, numerical.hessian, atol=1e-10)
    # What it keeps in its results is not the Hessian: rows of the hydrogens are zero.
    calc = bonds()
    atoms = engine.atoms(boxed)
    atoms.calc = calc
    atoms.get_forces()
    kept = np.asarray(calc.results["hessian"])
    assert np.all(kept[3:] == 0.0) and np.any(kept[:3] != 0.0)
