"""Extended tight binding through tblite and dxtb (atomic units, ``uhf``/``spin`` = 2S).

Corrections to MultiOptPy:

- tblite: the previous result restarts the SCF at the next structure (MultiOptPy started
  every calculation, including the 6N displacements of its numerical Hessian, from scratch).
- dxtb: positions in float64 (MultiOptPy used float32, which limits gradients to about 1e-7
  relative precision); ``spin`` is the number of unpaired electrons 2S (MultiOptPy passed the
  multiplicity); energy, gradient, and Hessian come from one SCF and autograd (MultiOptPy ran
  separate SCFs for energy and forces).

This file is part of mopt-py (the Python interface of mopt-rs).
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np

from mopt_rs.calculators.base import (
    ATOMIC_UNITS,
    Capabilities,
    ElectronicState,
    Engine,
    EngineResult,
    SpinConvention,
    Structure,
    missing_package,
)
from mopt_rs.calculators.program_options import ProgramSetup

__all__ = ["DxtbEngine", "TbliteEngine"]

_TBLITE_METHODS = {"gfn2": "GFN2-xTB", "gfn1": "GFN1-xTB", "ipea1": "IPEA1-xTB"}


def _c_array(value: Any) -> Any:
    """Number sequences as cffi ``double[]`` (what tblite's library expects), others as given."""
    if isinstance(value, (list, np.ndarray)):
        ffi = importlib.import_module("tblite.library").ffi
        return ffi.new("double[]", [float(v) for v in np.ravel(value)])
    return value


class TbliteEngine(Engine):
    """tblite's Python interface (``tblite.interface.Calculator``).

    ``solvation`` is ``("alpb", "water")``, ``("cpcm", 78.4)``, ... or ``None``; ``settings``
    go to ``Calculator.set`` (``max-iter``, ``accuracy``, ``temperature``, ...); ``interactions``
    go to ``Calculator.add`` in order (``{"electric-field": [0, 0, 0.001]}``, ``{"gbsa-solvation":
    "water"}``); ``setup`` hooks get the ``Calculator`` last.
    """

    units = ATOMIC_UNITS
    program = "tblite"
    citations = (
        "C. Bannwarth, E. Caldeweyher, S. Ehlert, A. Hansen, P. Pracht, J. Seibert, "
        "S. Spicher, S. Grimme, WIREs Comput. Mol. Sci. 2021, 11, e1493 (xTB methods; tblite)",
        "C. Bannwarth, S. Ehlert, S. Grimme, J. Chem. Theory Comput. 2019, 15, 1652 (GFN2-xTB)",
        "S. Grimme, C. Bannwarth, P. Shushkov, J. Chem. Theory Comput. 2017, 13, 1989 (GFN1-xTB)",
    )
    capabilities = Capabilities(analytic_hessian=False, pbc=True)

    def __init__(
        self,
        method: str = "GFN2-xTB",
        *,
        solvation: tuple[str, Any] | None = None,
        settings: dict[str, Any] | None = None,
        interactions: dict[str, Any] | None = None,
        setup: ProgramSetup | None = None,
        restart: bool = True,
    ) -> None:
        self.method = _TBLITE_METHODS.get(method.lower().replace("-xtb", ""), method)
        self.name = f"tblite:{self.method}"
        self.solvation = solvation
        self.settings = {"max-iter": 500, "verbosity": 0, **(settings or {})}
        self.interactions = dict(interactions or {})
        self.setup = setup or ProgramSetup()
        self.restart = restart
        self._last: dict[tuple[Any, ...], Any] = {}

    def _single_point(self, structure: Structure, keep_integrals: bool = False) -> Any:
        """tblite's result for ``structure`` (restarted from the last one with these atoms and
        this state); with ``keep_integrals`` the overlap matrix is kept in it."""
        try:
            from tblite.interface import Calculator
        except ImportError as e:
            raise missing_package("tblite", "tblite") from e

        state = structure.state
        periodic = np.asarray(structure.pbc) if structure.is_periodic else None
        calc = Calculator(
            self.method,
            structure.numbers,
            structure.positions_bohr,
            charge=float(state.charge),
            uhf=int(state.spin_as(SpinConvention.UNPAIRED)),
            lattice=structure.cell_bohr if structure.is_periodic else None,
            periodic=periodic,
        )
        for k, v in self.settings.items():
            calc.set(k, v)
        if keep_integrals:
            calc.set("save-integrals", 1)
        if self.solvation is not None:
            kind, value = self.solvation
            calc.add(f"{kind}-solvation", value)
        # Vectors (an electric field) go to tblite's C library as double arrays; they are kept
        # alive until the calculation is done.
        keep: list[Any] = []
        for interaction, value in self.interactions.items():
            args = value if isinstance(value, tuple) else (value,)
            converted = [_c_array(a) for a in args]
            keep += converted
            calc.add(interaction, *converted)
        calc = self.setup.run(calc)
        key = (structure.symbols, state, structure.pbc)
        prev = self._last.get(key) if self.restart else None
        res = calc.singlepoint(prev) if prev is not None else calc.singlepoint()
        if self.restart:
            self._last[key] = res
        return res

    def orbitals(self, structure: Structure) -> dict[str, Any] | None:
        res = self._single_point(structure, keep_integrals=True)
        overlap = np.asarray(res.get("overlap-matrix"), dtype=np.float64)
        coefficients = np.asarray(res.get("orbital-coefficients"), dtype=np.float64)
        energies = np.asarray(res.get("orbital-energies"), dtype=np.float64)
        occupations = np.asarray(res.get("orbital-occupations"), dtype=np.float64)
        if coefficients.ndim == 2:
            coefficients, energies, occupations = (
                coefficients[None],
                energies[None],
                occupations[None],
            )
        sets = []
        for c, e, occ in zip(coefficients, energies, occupations, strict=True):
            # The library hands the matrix over in its own storage order: take the
            # orientation in which the orbitals (columns) are orthonormal.
            n = c.shape[0]
            columns = np.abs(c.T @ overlap @ c - np.eye(n)).max()
            rows = np.abs(c @ overlap @ c.T - np.eye(n)).max()
            sets.append(
                {
                    "coefficients": c if columns <= rows else c.T,
                    "energies": e,
                    "occupations": occ,
                }
            )
        return {"overlap": overlap, "sets": sets}

    def compute(self, structure: Structure, hessian: bool) -> EngineResult:
        res = self._single_point(structure)
        return EngineResult(
            float(res.get("energy")),
            gradient=np.asarray(res.get("gradient"), dtype=np.float64),
            extra={"charges": np.asarray(res.get("charges"))},
        )


class DxtbEngine(Engine):
    """dxtb (PyTorch): GFN1-xTB or GFN2-xTB with autograd Hessians."""

    units = ATOMIC_UNITS
    program = "dxtb"
    citations = (
        "M. Friede, C. Holzer, S. Ehlert, S. Grimme, J. Chem. Phys. 2024, 161, 062501 (dxtb)",
        "S. Grimme, C. Bannwarth, P. Shushkov, J. Chem. Theory Comput. 2017, 13, 1989 (GFN1-xTB)",
    )
    capabilities = Capabilities(analytic_hessian=True)

    def __init__(
        self,
        method: str = "GFN1-xTB",
        *,
        device: str = "cpu",
        setup: ProgramSetup | None = None,
        **opts: Any,
    ) -> None:
        self.method = method.upper().replace("-XTB", "")
        if self.method not in ("GFN1", "GFN2"):
            raise ValueError(f"dxtb supports GFN1-xTB and GFN2-xTB, not '{method}'")
        self.name = f"dxtb:{self.method}-xTB"
        self.device = device
        self.opts = opts
        self.setup = setup or ProgramSetup()
        self._calc: dict[tuple[str, ...], Any] = {}

    def _calculator(self, structure: Structure) -> Any:
        try:
            import dxtb
            import torch
        except ImportError as e:
            raise missing_package(e.name or "dxtb", "dxtb") from e

        if structure.symbols not in self._calc:
            numbers = torch.tensor(structure.numbers, dtype=torch.long, device=self.device)
            opts = {"maxiter": 50 * structure.n_atoms + 1000, **self.opts}
            cls = getattr(dxtb.calculators, f"{self.method}Calculator")
            calc = cls(numbers, opts=opts, dtype=torch.float64)
            self._calc[structure.symbols] = self.setup.run(calc)
        return self._calc[structure.symbols]

    def compute(self, structure: Structure, hessian: bool) -> EngineResult:
        calc = self._calculator(structure)
        import torch

        state: ElectronicState = structure.state
        kw: dict[str, Any] = {"chrg": state.charge}
        if state.unpaired_electrons:
            kw["spin"] = state.spin_as(SpinConvention.UNPAIRED)
        pos = torch.tensor(
            structure.positions_bohr, dtype=torch.float64, device=self.device, requires_grad=True
        )
        energy = calc.get_energy(pos, **kw)
        (grad,) = torch.autograd.grad(energy, pos, create_graph=hessian)
        h = None
        if hessian:
            n3 = 3 * structure.n_atoms
            flat = grad.reshape(-1)
            rows = [
                torch.autograd.grad(flat[k], pos, retain_graph=k < n3 - 1)[0].reshape(-1)
                for k in range(n3)
            ]
            h = torch.stack(rows).detach().cpu().numpy()
        calc.reset()
        return EngineResult(
            float(energy.detach().cpu()),
            gradient=grad.detach().cpu().numpy(),
            hessian=h,
        )
