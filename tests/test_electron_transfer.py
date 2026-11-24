"""The parts of the electron-transfer rate constants from Python (``mopt_rs.electron_transfer``):
the couplings, the charge-transfer index, the inner sphere, Scher-Holstein, the Marcus,
Zhu-Nakamura, and Marcus-Levich-Jortner rates, SAES, and the apparent barrier; the transfer
integral of the projection method, the four-point reorganization energy, and the hopping rate;
and ``etmain`` through ``run_etmain`` on a bridged radical anion with the built-in GFN1-xTB.

This file is part of mopt-py (the Python interface of mopt-rs).
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from mopt_rs import _core
from mopt_rs import electron_transfer as et

EV = 1.0 / _core.HARTREE_TO_EV
# The constant of the compiled module: a value rounded to ten digits (3.166811563e-6) shifts
# the inverted Eyring equation by 1.8e-9 (the exponent of ln(kT/hk) = 12 times 1.1e-10).
KB = _core.BOLTZMANN_HARTREE_PER_K  # Hartree / K


def test_couplings_and_the_charge_transfer_index() -> None:
    assert et.mulliken_hush(0.05, [0.0, 0.0, 0.02], 15.0) == pytest.approx(0.05 * 0.02 / 15.0)
    # Delocalized states: half the gap.
    assert et.generalized_mulliken_hush(0.05, [0.0, 0.0, 5.0], [0.0, 0.0, 1e-6]) == pytest.approx(
        0.025, rel=1e-9
    )
    q, d, gained, lost = et.charge_transfer(
        [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], [-1.0, 1.0]
    )
    assert q == pytest.approx(1.0) and d == pytest.approx(10.0)
    assert gained[0] == pytest.approx(10.0) and lost[0] == pytest.approx(0.0)


def test_the_rates_and_their_limits() -> None:
    t = 298.15
    v, ls = 1e-4, 0.3 * EV
    # Marcus-Levich-Jortner without an inner sphere is Marcus of the solvent.
    mlj = et.marcus_levich_jortner(v, 0.2 * EV, 0.0, ls, -0.1 * EV, t)
    marcus = et.marcus_rate(0.0, v, ls, (ls - 0.1 * EV) ** 2 / (4 * ls), t)
    assert mlj == pytest.approx(marcus, rel=1e-10)
    # Scher-Holstein: no tunnelling for an activationless transfer.
    ln_gamma, _, _ = et.scher_holstein(0.18 * EV, -0.4 * EV, 0.4 * EV, t)
    assert abs(ln_gamma) < 1e-12
    # The apparent barrier inverts the Eyring equation.
    k = 3.0e7
    g = et.apparent_barrier(k, t)
    eyring = 1.380649e-23 * t / 6.62607015e-34 * math.exp(-g / (KB * t))
    assert eyring == pytest.approx(k, rel=1e-10)
    # SAES at alpha = 1: the shifted states cross where the solvent coordinate says.
    gi, gf, xs = et.solvent_shifts(0.5 * EV, 0.5 * EV, 0.2 * EV, 1.0)
    assert gi == pytest.approx(0.5 * EV * xs**2)
    assert gi - gf == pytest.approx(0.2 * EV, rel=1e-9)


def test_the_inner_sphere_and_zhu_nakamura_at_a_crossing() -> None:
    # Two displaced harmonic states along one mass-weighted coordinate (1 amu).
    omega = 0.0072
    k = omega * omega * 1822.888486209
    d, de0 = 0.9, -0.012
    q = (0.5 * k * d * d + de0) / (k * d)
    gi, gf = [k * q], [k * (q - d)]
    barrier = 0.5 * k * q * q
    inner = et.inner_sphere(gi, gf, barrier)
    assert inner["quantum"] == pytest.approx(omega, rel=1e-10)
    assert inner["reaction_energy"] == pytest.approx(de0, abs=1e-12)
    kind, ln_kappa, below, rate = et.zhu_nakamura_rate(gi, gf, barrier, 1e-5, barrier, 298.15)
    assert kind == "peaked" and rate > 0.0 and 0.0 <= below <= 1.0 and math.isfinite(ln_kappa)


def test_etmain_from_python(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    xyz = tmp_path / "pentadiene.xyz"
    xyz.write_text(
        "13\n1,4-pentadiene radical anion\n"
        "C -2.475780 0.086490 0.007191\nC -1.260558 -0.437376 -0.005142\n"
        "C 0.000000 0.388339 0.000000\nC 1.260558 -0.437376 0.005142\n"
        "C 2.475780 0.086490 -0.007191\nH -2.646139 1.155744 0.020778\n"
        "H -3.363258 -0.531520 0.002753\nH -1.130623 -1.516274 -0.018504\n"
        "H -0.005094 1.048767 0.876439\nH 0.005094 1.048767 -0.876439\n"
        "H 1.130623 -1.516274 0.018504\nH 2.646139 1.155744 -0.020778\n"
        "H 3.363258 -0.531520 -0.002753\n"
    )
    monkeypatch.chdir(tmp_path)
    runs = et.run_etmain(
        [
            str(xyz),
            "-os",
            "gfn1xtb",
            "-elec",
            "-1",
            "-spin",
            "2",
            "-et_donor",
            "1,2,6-8",
            "-et_acceptor",
            "4,5,11-13",
            "-et_coupling",
            "edeel",
            "-et_free_energy",
            "off",
        ],
        console=False,
    )
    assert len(runs) == 1
    crossing = runs[0]["crossings"][0]
    assert crossing["barrier"] > 0.0 and crossing["marcus"] > 0.0
    assert crossing["free_energy_barrier"] is None
    assert Path(runs[0]["directory"], "et_summary.json").is_file()


def test_the_transfer_integral_of_a_symmetric_pair() -> None:
    # H = [[e, t], [t, e]] with S = [[1, s], [s, 1]]: J_eff = (t - e s) / (1 - s^2), and the
    # Fock matrix comes back from the generalized eigenvectors.
    e, t, s = -0.4, -0.05, 0.2
    h = np.array([[e, t], [t, e]])
    overlap = np.array([[1.0, s], [s, 1.0]])
    c = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt([2 * (1 + s), 2 * (1 - s)])
    energies = [(e + t) / (1 + s), (e - t) / (1 - s)]
    assert np.allclose(et.fock_from_orbitals(overlap, c, energies), h, atol=1e-14)
    found = et.transfer_integral(h, overlap, [1.0, 0.0], [0.0, 1.0])
    assert found["effective"] == pytest.approx((t - e * s) / (1 - s * s), abs=1e-15)
    assert found["degeneracy"] == [1, 1]


def test_the_hopping_rate_and_the_four_point_reorganization() -> None:
    lam = et.four_point_reorganization((0.006, 0.001), (0.005, 0.0))
    assert lam == pytest.approx(0.010, abs=1e-15)
    k = et.hopping_rate(0.002, lam, 0.0, 298.15)
    marcus = et.marcus_rate(0.0, 0.002, lam, lam / 4.0, 298.15)
    assert k == pytest.approx(marcus, rel=1e-12)


def test_the_projection_coupling_of_an_ethylene_dimer_with_gfn1() -> None:
    # The dimer of HAB11 at 3.5 Angstrom: tests/hab11_couplings.rs gives 118.8 meV with the
    # built-in GFN1-xTB (the reference is 519.2 meV).
    from mopt_rs.calculators import create
    from mopt_rs.calculators.base import Structure

    data = json.loads(
        (Path(__file__).parent / "golden" / "hab11.json").read_text(encoding="utf-8")
    )
    dimer = next(
        d
        for d in data["dimers"]
        if d["name"] == "ethylene" and d["distance_angstrom"] == 3.5
    )
    structure = Structure.build(
        dimer["symbols"], np.asarray(dimer["positions_angstrom"]) / 0.529177210903
    )
    a, b = dimer["molecules"]
    found = et.projection_coupling(create("gfn1xtb"), structure, a, b, "hole")
    assert abs(found["effective"]) * 27211.386245988 == pytest.approx(118.8, abs=0.1)
    assert found["degeneracy"] == [1, 1]


def test_a_hop_from_python(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # etmain -et_method hopping on the ethylene dimer of HAB11 at 4.0 Angstrom (GFN1-xTB):
    # the coupling of tests/hab11_couplings.rs and the same rate both ways.
    data = json.loads(
        (Path(__file__).parent / "golden" / "hab11.json").read_text(encoding="utf-8")
    )
    dimer = next(
        d
        for d in data["dimers"]
        if d["name"] == "ethylene" and d["distance_angstrom"] == 4.0
    )
    lines = [f"{len(dimer['symbols'])}", "ethylene dimer"]
    for s, p in zip(dimer["symbols"], dimer["positions_angstrom"], strict=True):
        lines.append(f"{s} {p[0]:.8f} {p[1]:.8f} {p[2]:.8f}")
    xyz = tmp_path / "dimer.xyz"
    xyz.write_text("\n".join(lines) + "\n")
    a, b = (",".join(str(k + 1) for k in m) for m in dimer["molecules"])
    monkeypatch.chdir(tmp_path)
    runs = et.run_etmain(
        [str(xyz), "-os", "gfn1xtb", "-et_method", "hopping", "-et_donor", a, "-et_acceptor", b],
        console=False,
    )
    run = runs[0]
    assert run["method"] == "hopping"
    assert abs(run["coupling"]) * 27211.386245988 == pytest.approx(35.2, abs=0.1)
    assert run["rates"][0] == pytest.approx(run["rates"][1], rel=1e-2)
    assert run["mean_reorganization"] > 0.0
    assert Path(run["directory"], "hop_summary.json").is_file()


def test_the_fragment_charge_difference_of_two_mixed_states() -> None:
    # Two diabatic states mixed by theta (tan 2 theta = 2 V / eps): their populations and
    # transition density give V back.
    eps, v = 0.01, 0.004
    gap = math.hypot(eps, 2.0 * v)
    theta = 0.5 * math.atan2(2.0 * v, eps)
    c, s = math.cos(theta), math.sin(theta)
    change = [s * s - c * c, c * c - s * s]
    transition = [-c * s, c * s]
    found = et.fragment_charge_difference(gap, transition, change, [0], [1])
    assert found == pytest.approx(v, abs=1e-15)


def test_the_pcet_parts() -> None:
    # A proton at 3000 cm^-1: the harmonic ladder on the grid; the same potential on both
    # sides is Marcus with the electronic coupling; a displaced product slows the deuteron more.
    w = 3000.0 / 219474.63136
    mass = et.PROTON_MASS_AMU * 1822.888486209
    n, first, spacing = 201, -2.5, 0.025

    def potential(m_amu: float, centre: float) -> list[float]:
        m = m_amu * 1822.888486209
        wm = w * math.sqrt(et.PROTON_MASS_AMU / m_amu)
        return [0.5 * m * wm * wm * (first + spacing * i - centre) ** 2 for i in range(n)]

    states = et.proton_states(first, spacing, potential(et.PROTON_MASS_AMU, 0.0), count=4)
    assert states["energies"] == pytest.approx([w * (k + 0.5) for k in range(4)], abs=1e-9)
    assert mass > 1800.0
    v, lam, t = 0.002, 0.03, 298.15
    same = et.pcet_rate(first, spacing, potential(et.PROTON_MASS_AMU, 0.0),
                        potential(et.PROTON_MASS_AMU, 0.0), v, lam, count=4)
    assert same["rate"] == pytest.approx(et.marcus_rate(0.0, v, lam, lam / 4.0, t), rel=1e-6)
    h = et.pcet_rate(first, spacing, potential(et.PROTON_MASS_AMU, 0.0),
                     potential(et.PROTON_MASS_AMU, 0.6), v, lam, count=4)
    d = et.pcet_rate(first, spacing, potential(et.DEUTERON_MASS_AMU, 0.0),
                     potential(et.DEUTERON_MASS_AMU, 0.6), v, lam,
                     mass_amu=et.DEUTERON_MASS_AMU, count=4)
    assert h["rate"] > 2.0 * d["rate"]
    assert tuple(h["channels"][0][:2]) == (0, 0)
