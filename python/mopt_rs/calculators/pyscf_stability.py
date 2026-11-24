"""The test of an SCF solution of PySCF for instabilities and the way to a stabler one
(``-stability``, :mod:`mopt_rs.calculators.stability`).

PySCF's stability analysis (``mf.stability``) finds the lowest eigenvalue of the
second-order change of the energy with a rotation of the orbitals. If it is negative within
the kind of wavefunction ("internal": RHF, ROHF, UHF, GHF and their Kohn-Sham forms), the
SCF goes on from the orbitals it returns, rotated along that direction, and the test is
repeated (at most five times). An instability towards an unrestricted wavefunction
("external", restricted references only) is said with ``external=report``; with
``external=switch`` the engine changes to UHF/UKS for this and every later structure and
starts from the orbitals PySCF returns. An unrestricted solution that is unstable towards a
generalized one is said only (generalized wavefunctions have no analytic gradients).

MultiOptPy: no counterpart (new in mopt-rs).

This file is part of mopt-py (the Python interface of mopt-rs).
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from mopt_rs.calculators.stability import StabilityReport

if TYPE_CHECKING:
    from mopt_rs.calculators.base import Structure
    from mopt_rs.calculators.pyscf import PySCFEngine

__all__ = ["ROUNDS", "stabilize"]

ROUNDS = 5
"""Largest number of times the SCF goes on from stabler orbitals for one structure."""

_NAMES = {"r": "RHF/RKS", "ro": "ROHF/ROKS", "u": "UHF/UKS", "g": "GHF/GKS"}


def _rerun(mf: Any, dm: Any) -> Any:
    mf.kernel(dm0=dm)
    if not mf.converged:
        mf = mf.newton()
        mf.kernel(dm0=mf.make_rdm1())
    if not mf.converged:
        raise RuntimeError("the SCF from the stabler orbitals did not converge")
    return mf


def _unrestricted_occupations(mo_occ: Any) -> tuple[Any, Any]:
    occ = np.asarray(mo_occ)
    return (occ > 0.5).astype(float), (occ > 1.5).astype(float)


def stabilize(engine: PySCFEngine, structure: Structure, external: str) -> StabilityReport:
    """Test the solution of ``structure`` and move to a stabler one where it is unstable."""
    open_shell = structure.state.unpaired_electrons > 0
    kind = engine.kind(open_shell)
    mf = engine._converged(structure)
    start = float(mf.e_tot)
    lines: list[str] = []
    changed = False
    for _ in range(ROUNDS):
        if kind == "g":
            mo, stable = mf.stability(return_status=True)
            outward = False
            mo_out = None
        else:
            mo, mo_out, stable, stable_out = mf.stability(
                internal=True, external=external != "off", return_status=True
            )
            outward = external != "off" and stable_out is False
        if not stable:
            before = float(mf.e_tot)
            mf = _rerun(mf, mf.make_rdm1(mo, mf.mo_occ))
            lines.append(
                f"SCF stability: the {_NAMES[kind]} solution was unstable within its kind; the "
                f"SCF went on along the unstable rotation ({before:.10f} -> "
                f"{float(mf.e_tot):.10f} Hartree)"
            )
            changed = True
            continue
        if outward:
            if kind == "u":
                lines.append(
                    "SCF stability: the UHF/UKS solution is unstable towards a generalized "
                    "wavefunction (said only: GHF/GKS have no analytic gradients)"
                )
            elif external == "report":
                lines.append(
                    f"SCF stability: the {_NAMES[kind]} solution is unstable towards an "
                    "unrestricted wavefunction; the kind is kept (-stability_external switch "
                    "changes to UHF/UKS)"
                )
            else:
                occupations = _unrestricted_occupations(mf.mo_occ)
                before = float(mf.e_tot)
                engine.switch_to_unrestricted()
                kind = engine.kind(open_shell)
                unrestricted = engine._scf(engine._molecule(structure), open_shell)
                mf = _rerun(unrestricted, unrestricted.make_rdm1(mo_out, occupations))
                lines.append(
                    "SCF stability: the restricted solution was unstable towards an unrestricted "
                    f"wavefunction; the run goes on with UHF/UKS ({before:.10f} -> "
                    f"{float(mf.e_tot):.10f} Hartree)"
                )
                changed = True
                continue
        break
    else:
        lines.append(f"SCF stability: the solution was still unstable after {ROUNDS} rounds")
    if changed:
        key = (structure.symbols, structure.state)
        engine._dm[key] = mf.make_rdm1()
        lines.append(
            f"SCF stability: {start:.10f} -> {float(mf.e_tot):.10f} Hartree; the evaluation is "
            "repeated from the stabler solution"
        )
    return StabilityReport(changed, lines)
