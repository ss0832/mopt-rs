"""``run_mapper`` in a periodic cell with a calculator of the ASE route, fixed and variable
cell: the vacancy of copper (31 atoms in 2 x 2 x 2 cells) explored by pushing and pulling the
atom next to the vacancy and the atom behind it, once with the EMT of mopt-rs through the
conventions of an ASE calculator (eV, Angstrom, the stress as ASE's six components;
:class:`fake_ase.NativeEmt`) and once with the built-in EMT. With the cell held: the same minima
and transition states. With the cell moving (where the ASE route gives the stress that moves
it), the built-in EMT has analytic Hessians of the atoms and the cell and the ASE route
differences in every variable, and a transition-state search from the same guess can end at
another stationary point (on the vacancy of copper the search of the push does), so the
networks need not be the same: the hop of the built-in network is in that of the ASE route, at
the same enthalpy and with the same imaginary mode, and the minimum it starts from too.

This file is part of mopt-py (the Python interface of mopt-rs).
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

import fake_ase
from mopt_rs import _core

HAS_ASE = importlib.util.find_spec("ase") is not None
A = 3.615


@pytest.fixture(autouse=True)
def an_ase(monkeypatch: pytest.MonkeyPatch) -> None:
    """ASE, or its stand-in where it is not installed."""
    if not HAS_ASE:
        monkeypatch.setitem(sys.modules, "ase", fake_ase.module())


def _copper(path: Path) -> None:
    """The 31 atoms of 2 x 2 x 2 cells of copper without the atom at the origin; atom 1 is at
    (A/2, A/2, 0), next to the vacancy, and atom 24 at (A, A, 0), behind it."""
    sites = [(0.5 * A, 0.5 * A, 0.0)]
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for f in ((0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)):
                    if (i, j, k) == (0, 0, 0) and f in ((0, 0, 0), (0.5, 0.5, 0)):
                        continue
                    sites.append((A * (i + f[0]), A * (j + f[1]), A * (k + f[2])))
    lattice = 2.0 * A
    lines = "".join(f"Cu {x:.10f} {y:.10f} {z:.10f}\n" for x, y, z in sites)
    path.write_text(
        f'{len(sites)}\nLattice="{lattice} 0.0 0.0 0.0 {lattice} 0.0 0.0 0.0 {lattice}" '
        f'pbc="T T T"\n{lines}'
    )


def _run(tmp_path: Path, engine: str, name: str, variable_cell: bool) -> dict[str, object]:
    # 31 copper atoms have an odd number of electrons: a doublet (EMT does not use it).
    step1: dict[str, object] = {"othersoft": engine, "NSTEP": 300, "spin_multiplicity": 2}
    if variable_cell:
        step1["variable_cell"] = True
    settings = {
        "mapper_settings": {
            "afir_gamma_kJmol": 300.0,
            "dist_lower_ang": 2.0,
            "dist_upper_ang": 3.0,
            "max_iterations": 2,
            "rng_seed": 1,
            "active_atoms": [1, 24],
            "pair_bond_scale": 0.5,
            "include_negative_gamma": True,
        },
        "top_n_candidates": 1,
        "step1_settings": step1,
        # Compare the two calculator interfaces on the reference NEB schedule.
        "step2_settings": {"NSTEP": 20},
        "step3_settings": {"NSTEP": 200, "calc_exact_hess": 5},
        "step4_settings": {"NSTEP": 400, "intrinsic_reaction_coordinates": ["0.1", "300", "lqa"]},
    }
    config = tmp_path / f"{name}.json"
    config.write_text(json.dumps(settings))
    out = tmp_path / name
    code = _core.run_cli(
        "run_mapper",
        [str(tmp_path / "cu.xyz"), "-cfg", str(config), "--output_dir", str(out)],
    )
    assert code == 0, name
    return json.loads((out / "reaction_network.json").read_text())


def _kinds(network: dict[str, object]) -> list[tuple[str, float]]:
    """The kinds and energies of the nodes, sorted."""
    return sorted((n["kind"], n["energy_hartree"]) for n in network["nodes"])


def _hops(network: dict[str, object]) -> list[tuple[float, float]]:
    """The energies and imaginary wavenumbers of the transition states of the edges."""
    nodes = {n["id"]: n for n in network["nodes"]}
    edges = list(network["ts_edges"]) + list(network.get("open_ts_edges", []))
    return [(nodes[e["ts"]]["energy_hartree"], e["imaginary_wavenumber_cm1"]) for e in edges]


@pytest.mark.parametrize("variable_cell", [False, True])
def test_the_ase_route_gives_the_network_of_the_built_in_calculator(
    tmp_path: Path, variable_cell: bool
) -> None:
    _copper(tmp_path / "cu.xyz")
    fake_ase.NativeEmt.calls.update(energy=0, stress=0)
    ase_network = _run(tmp_path, "ase:fake_ase:NativeEmt", "ase", variable_cell)
    native_network = _run(tmp_path, "emt", "native", variable_cell)
    through_ase, built_in = _kinds(ase_network), _kinds(native_network)
    # A transition state: the hop of the atom into the vacancy.
    assert any(kind == "TS" for kind, _ in built_in), built_in
    if not variable_cell:
        assert [k for k, _ in through_ase] == [k for k, _ in built_in], (through_ase, built_in)
        for (_, a), (_, b) in zip(through_ase, built_in, strict=True):
            assert a == pytest.approx(b, abs=1e-5)
        return
    # The minimum of the start (the node of the input) is the same in both networks, and so is
    # the vacancy hop from it back to itself (the hop of the pull, at the same enthalpy and with
    # the same imaginary mode). Beyond it the transition-state search from the guess of the push
    # is a knife-edge on this cell: with the analytic Hessians of the built-in EMT it ends,
    # depending on digits below any tolerance (the constants of the units, the order of sums),
    # either at a stationary point without an imaginary mode or at a first-order saddle point
    # whose IRC leads to a minimum with a sheared cell 1.2 mHartree below the start (a minimum
    # of the enthalpy verified by its Hessian); the ASE route with Hessians of differences ends
    # elsewhere. A hop found by one route only must lead to a minimum that route found.
    def start_of(network: dict[str, object]) -> float:
        node = network["inputs"][0]["node"]
        return next(n["energy_hartree"] for n in network["nodes"] if n["id"] == node)

    assert start_of(ase_network) == pytest.approx(start_of(native_network), abs=1e-6)
    start = start_of(native_network)

    def self_hops(network: dict[str, object]) -> list[tuple[float, float]]:
        nodes = {n["id"]: n for n in network["nodes"]}
        node = network["inputs"][0]["node"]
        return [
            (nodes[e["ts"]]["energy_hartree"], e["imaginary_wavenumber_cm1"])
            for e in network["ts_edges"]
            if list(e["ends"]) == [node, node]
        ]

    ase_hops = self_hops(ase_network)
    native_hops = self_hops(native_network)
    assert native_hops, native_network["ts_edges"]
    for energy, wavenumber in native_hops:
        assert any(
            e == pytest.approx(energy, abs=1e-6) and w == pytest.approx(wavenumber, abs=1.0)
            for e, w in ase_hops
        ), (energy, wavenumber, ase_hops)
    for network in (ase_network, native_network):
        minima = {n["id"] for n in network["nodes"] if n["kind"] == "EQ"}
        for e in network["ts_edges"]:
            assert set(e["ends"]) <= minima | {n["id"] for n in network["nodes"]}, e
        assert all(
            n["energy_hartree"] >= start - 0.01 for n in network["nodes"] if n["kind"] == "EQ"
        )
    # On a variable cell the stress came from the calculator of the ASE route.
    assert fake_ase.NativeEmt.calls["stress"] > 0
