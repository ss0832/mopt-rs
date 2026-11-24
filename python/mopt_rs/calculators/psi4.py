"""Psi4 (MultiOptPy's former default engine; mopt-rs defaults to PySCF).

MultiOptPy: `Calculator/psi4_calculation_tools.py`.

This file is part of mopt-py (the Python interface of mopt-rs).
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mopt_rs.calculators.base import (
    ATOMIC_UNITS,
    Capabilities,
    Engine,
    EngineResult,
    SpinConvention,
    Structure,
)
from mopt_rs.calculators.program_options import ProgramSetup
from mopt_rs.calculators.stability import StabilityReport

__all__ = ["Psi4Engine"]

# MultiOptPy's -grid levels as (radial, spherical) points.
_GRIDS = {1: (50, 194), 2: (75, 302), 3: (99, 590), 4: (150, 770)}


class Psi4Engine(Engine):
    """Psi4 energies, gradients, and Hessians; ``options`` go to ``psi4.set_options`` (any
    Psi4 option: ``pcm``, ``perturb_h`` / ``perturb_dipole`` for an external field, ...);
    ``setup`` hooks get the Psi4 molecule."""

    units = ATOMIC_UNITS
    program = "psi4"
    citations = ("D. G. A. Smith et al., J. Chem. Phys. 2020, 152, 184108 (Psi4 1.4)",)
    capabilities = Capabilities(analytic_hessian=True)

    def __init__(
        self,
        method: str = "b3lyp",
        basis: str = "def2-svp",
        *,
        unrestricted: bool = False,
        grid_level: int | None = None,
        threads: int | None = None,
        memory: str | None = None,
        options: dict[str, Any] | None = None,
        setup: ProgramSetup | None = None,
    ) -> None:
        self.method = method
        self.basis = basis
        self.unrestricted = unrestricted
        self.grid_level = grid_level
        self.threads = threads
        self.memory = memory
        self.options = dict(options or {})
        self.setup = setup or ProgramSetup()
        self.name = f"psi4:{method}/{basis}"
        self._stability_next = False

    def check_stability(self, structure: Structure, external: str) -> StabilityReport | None:
        """Psi4 tests its own solution: the evaluation is repeated with
        ``stability_analysis follow`` for unrestricted references (it goes on from a stabler
        solution where it finds one) and ``check`` for restricted ones (Psi4 reports an
        instability, also towards UHF, in its output, and does not follow it)."""
        self._stability_next = True
        return StabilityReport(
            True,
            [
                "SCF stability: Psi4 tests its solution in the repeated evaluation "
                "(stability_analysis follow for UHF/UKS, check for RHF/RKS)"
            ],
        )

    def compute(self, structure: Structure, hessian: bool) -> EngineResult:
        import psi4

        state = structure.state
        lines = [f"{state.charge} {state.spin_as(SpinConvention.MULTIPLICITY)}"]
        lines += [
            f"{s} {x:.12f} {y:.12f} {z:.12f}"
            for s, (x, y, z) in zip(structure.symbols, structure.positions_bohr, strict=True)
        ]
        lines += ["units bohr", "no_com", "no_reorient", "symmetry c1"]
        psi4.core.be_quiet()
        if self.threads is not None:
            psi4.set_num_threads(self.threads)
        if self.memory is not None:
            psi4.set_memory(self.memory)
        mol = self.setup.run(psi4.geometry("\n".join(lines)))
        open_shell = state.unpaired_electrons > 0 or self.unrestricted
        dft = self.method.lower() not in ("hf", "scf")
        reference = ("uks" if dft else "uhf") if open_shell else ("rks" if dft else "rhf")
        opts: dict[str, Any] = {"basis": self.basis, "reference": reference}
        if self.grid_level in _GRIDS:
            r, s = _GRIDS[self.grid_level]
            opts.update({"dft_radial_points": r, "dft_spherical_points": s})
        if self._stability_next:
            self._stability_next = False
            opts["stability_analysis"] = "follow" if reference in ("uhf", "uks") else "check"
        opts.update(self.options)
        psi4.set_options(opts)
        grad, wfn = psi4.gradient(self.method, molecule=mol, return_wfn=True)
        energy = float(wfn.energy())
        h = None
        if hessian:
            h = np.asarray(psi4.hessian(self.method, molecule=mol, ref_gradient=grad))
        psi4.core.clean()
        return EngineResult(energy, gradient=np.asarray(grad), hessian=h)
