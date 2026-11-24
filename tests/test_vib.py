"""Vibrational analysis from Python: normal modes, point group, thermochemistry.

MultiOptPy: `Thermo/normal_mode_analyzer.py:MolecularVibrations`.

This file is part of mopt-py (the Python interface of mopt-rs).
"""

from __future__ import annotations

import numpy as np
import pytest

from mopt_rs import _core

ANG = _core.ANGSTROM_TO_BOHR
SYMBOLS = ["O", "H", "H"]
START = (np.array([[0.0, 0.0, 0.119], [0.0, 0.763, -0.477], [0.0, -0.763, -0.477]]) * ANG).ravel()


def relaxed_water() -> tuple[np.ndarray, float, np.ndarray]:
    """Water at its GFN-FF minimum with its energy and Hessian."""
    x = START.copy()
    for _ in range(200):
        energy, gradient, hessian = _core.evaluate_native("gfnff", SYMBOLS, x.tolist(), 0, 1, True)
        g = np.asarray(gradient)
        if np.abs(g).max() < 1e-9:
            break
        h = np.asarray(hessian).reshape(9, 9)
        x = x - np.linalg.pinv(h, rcond=1e-8) @ g
    assert np.abs(g).max() < 1e-7
    return x, float(energy), np.asarray(hessian)


def test_normal_modes_point_group_and_thermochemistry() -> None:
    x, energy, hessian = relaxed_water()
    modes = _core.normal_modes(SYMBOLS, x.tolist(), hessian.tolist())
    wavenumbers = np.asarray(modes["wavenumbers"])
    assert wavenumbers.shape == (9,)
    # Six rigid motions at zero and three vibrations: a bend and two stretches.
    vibrations = np.sort(wavenumbers[np.abs(wavenumbers) > 10.0])
    assert vibrations.shape == (3,) and vibrations[0] > 1000.0 and vibrations[2] < 4500.0
    assert np.count_nonzero(np.abs(wavenumbers) < 1e-3) == 6
    assert np.asarray(modes["displacements"]).shape == (9, 9)
    assert all(m > 0.0 for m in np.asarray(modes["reduced_masses"])[np.abs(wavenumbers) > 10.0])
    # Without projection the rigid motions are still there, close to zero at a minimum.
    raw = np.asarray(
        _core.normal_modes(SYMBOLS, x.tolist(), hessian.tolist(), project=False)["wavenumbers"]
    )
    assert np.allclose(np.sort(raw)[-3:], vibrations, atol=0.5)

    label, number, operations, tolerance = _core.point_group(SYMBOLS, x.tolist())
    assert (label, number, operations) == ("C2v", 2, 4) and tolerance <= 0.01
    a, b, c = _core.rotational_constants(SYMBOLS, x.tolist())
    assert a > b > c > 100.0

    found = _core.vibrational_analysis(SYMBOLS, x.tolist(), hessian.tolist(), energy)
    assert found["point_group"] == "C2v" and found["symmetry_number"] == 2
    thermo = found["thermo"]
    assert thermo["vibrations"] == 3 and thermo["imaginary"] == 0
    # The zero-point energy is half the sum of the vibrational quanta.
    hartree_per_wavenumber = 4.556335252912e-6
    assert thermo["zero_point_energy"] == pytest.approx(
        0.5 * vibrations.sum() * hartree_per_wavenumber, rel=1e-6
    )
    assert thermo["energy_0k"] == pytest.approx(energy + thermo["zero_point_energy"])
    gibbs, enthalpy, entropy = (thermo[k]["total"] for k in ("gibbs", "enthalpy", "entropy"))
    assert gibbs == pytest.approx(enthalpy - 298.15 * entropy, abs=1e-12)
    assert entropy > 0.0 and thermo["cp"]["total"] > thermo["cv"]["total"]
    assert "Point group: C2v" in found["text"]

    # The choices of -freq: a higher temperature raises the entropy; a symmetry number
    # given by hand is used as it is.
    hot = _core.vibrational_analysis(
        SYMBOLS, x.tolist(), hessian.tolist(), energy, options={"temperature": 500.0}
    )
    assert hot["thermo"]["entropy"]["total"] > entropy
    by_hand = _core.vibrational_analysis(
        SYMBOLS, x.tolist(), hessian.tolist(), energy, options={"symmetry_number": 1}
    )
    assert by_hand["symmetry_number"] == 1 and by_hand["point_group"] == "C1"
    boltzmann = _core.BOLTZMANN_HARTREE_PER_K
    assert by_hand["thermo"]["entropy"]["rotational"] - thermo["entropy"]["rotational"] == (
        pytest.approx(boltzmann * np.log(2.0), rel=1e-6)
    )
    # A doublet has the electronic entropy k ln 2.
    doublet = _core.vibrational_analysis(
        SYMBOLS, x.tolist(), hessian.tolist(), energy, charge=1, multiplicity=2
    )
    assert doublet["thermo"]["entropy"]["electronic"] == pytest.approx(
        boltzmann * np.log(2.0), rel=1e-6
    )


def test_a_direction_left_out_and_wrong_input() -> None:
    x, _, hessian = relaxed_water()
    # A structure off the minimum along the symmetric stretch: with the gradient left out,
    # one vibration less.
    energy, gradient, moved_hessian = _core.evaluate_native(
        "gfnff", SYMBOLS, (1.03 * x).tolist(), 0, 1, True
    )
    assert moved_hessian is not None
    all_modes = np.asarray(
        _core.normal_modes(SYMBOLS, (1.03 * x).tolist(), moved_hessian)["wavenumbers"]
    )
    fewer = np.asarray(
        _core.normal_modes(SYMBOLS, (1.03 * x).tolist(), moved_hessian, gradient=gradient)[
            "wavenumbers"
        ]
    )
    assert np.count_nonzero(np.abs(all_modes) > 10.0) == 3
    assert np.count_nonzero(np.abs(fewer) > 10.0) == 2
    with pytest.raises(ValueError, match="Hessian"):
        _core.normal_modes(SYMBOLS, x.tolist(), hessian.tolist()[:10])
    with pytest.raises(ValueError, match="gradient"):
        _core.normal_modes(SYMBOLS, x.tolist(), hessian.tolist(), gradient=[0.0])
    with pytest.raises(ValueError, match="unknown setting"):
        _core.vibrational_analysis(SYMBOLS, x.tolist(), hessian.tolist(), 0.0, options={"t": 1.0})
