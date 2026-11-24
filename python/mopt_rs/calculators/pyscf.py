"""PySCF: Hartree-Fock and Kohn-Sham DFT, ground and TDA excited states (the default
quantum-chemistry engine of mopt-rs).

The multiplicity ``2S + 1`` is converted to PySCF's ``mol.spin = 2S`` here. Corrections and
generalizations relative to MultiOptPy:

- the scalar-relativistic X2C option applies to every reference (MultiOptPy used X2C for
  unrestricted Kohn-Sham only), and density fitting is optional (MultiOptPy always used it);
- the SCF at a new structure starts from the converged density of the previous one;
- restricted or unrestricted is chosen from the multiplicity, ``unrestricted=True`` forces
  UHF/UKS; ``reference="ro"`` takes the restricted open-shell methods (ROHF/ROKS) for open
  shells and ``reference="g"`` the generalized ones (GHF/GKS, spin orbitals that mix alpha
  and beta: non-collinear spins), which MultiOptPy does not offer;
- analytic Hessians when PySCF has them (HF/DFT ground states), otherwise mopt-rs finite
  differences, instead of failing for excited states;
- any further PySCF settings are passed through instead of being fixed in code: SCF
  attributes (``scf_options``, also dotted paths such as ``with_solvent.eps``), ``gto.M``
  keywords (``mol_options``), SCF methods applied in order (``apply``, e.g. ``ddCOSMO``), and
  user hooks on the SCF object (see :mod:`mopt_rs.calculators.program_options`).

This file is part of mopt-py (the Python interface of mopt-rs).
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import io
from typing import Any

import numpy as np

from mopt_rs.calculators.base import (
    ATOMIC_UNITS,
    Capabilities,
    Engine,
    EngineResult,
    SpinConvention,
    Structure,
    missing_package,
)
from mopt_rs.calculators.program_options import ProgramSetup, set_path
from mopt_rs.calculators.stability import StabilityReport

__all__ = ["PySCFEngine", "REFERENCES"]

REFERENCES = {
    "auto": "auto",
    "r": "restricted",
    "restricted": "restricted",
    "rhf": "restricted",
    "rks": "restricted",
    "ro": "restricted",
    "rohf": "restricted",
    "roks": "restricted",
    "u": "unrestricted",
    "unrestricted": "unrestricted",
    "uhf": "unrestricted",
    "uks": "unrestricted",
    "g": "generalized",
    "generalized": "generalized",
    "ghf": "generalized",
    "gks": "generalized",
    "noncollinear": "generalized",
}
"""Names of the kinds of wavefunction (``reference=``) and what they select."""


class PySCFEngine(Engine):
    """PySCF molecular calculations.

    ``method`` is ``hf`` or an exchange-correlation functional (``b3lyp``, ``wb97x-d``, ...).
    ``basis`` and ``ecp`` take anything ``pyscf.gto.M`` accepts (a name or a per-element dict).
    ``mol_options`` are further ``gto.M`` keywords; ``apply`` names SCF methods called in order
    (``("ddCOSMO",)`` gives ``mf = mf.ddCOSMO()``); ``scf_options`` then set attributes, also
    along dotted paths (``{"with_solvent.eps": 78.4}``); ``setup`` hooks get the SCF object last.

    ``reference`` chooses the kind of wavefunction: ``auto`` (restricted for closed shells,
    unrestricted for open ones or with ``unrestricted=True``), ``ro`` (restricted, that is
    RHF/RKS for closed and ROHF/ROKS for open shells), ``u`` (unrestricted), ``g``
    (generalized: GHF/GKS). PySCF has no analytic gradients for the generalized methods:
    they are taken from central differences of the energy with ``gradient_step_bohr``
    (``6N`` energies per gradient). For GKS, ``collinear`` is PySCF's treatment of the
    functional (``mcol`` where the ``mcfun`` package is installed, else ``ncol``, unless
    given; ``col`` evaluates it as for collinear spins). Hessians of ROHF/ROKS and of the
    generalized methods come from finite differences of the gradients.
    """

    units = ATOMIC_UNITS
    program = "pyscf"
    citations = (
        "Q. Sun et al., WIREs Comput. Mol. Sci. 2018, 8, e1340 (PySCF)",
        "Q. Sun et al., J. Chem. Phys. 2020, 153, 024109 (PySCF)",
    )
    capabilities = Capabilities(analytic_hessian=True, excited_states=True)

    def __init__(
        self,
        method: str = "b3lyp",
        basis: str | dict[str, str] = "def2-svp",
        *,
        ecp: str | dict[str, str] | None = None,
        unrestricted: bool = False,
        density_fit: bool = False,
        x2c: bool = False,
        grid_level: int = 3,
        threads: int | None = None,
        memory_mb: float | None = None,
        scf_options: dict[str, Any] | None = None,
        mol_options: dict[str, Any] | None = None,
        apply: tuple[str, ...] = (),
        setup: ProgramSetup | None = None,
        n_states: int | None = None,
        verbose: int = 4,
        reference: str = "auto",
        collinear: str | None = None,
        gradient_step_bohr: float = 5e-3,
    ) -> None:
        try:
            self.reference = REFERENCES[str(reference).lower()]
        except KeyError:
            raise ValueError(
                f"reference '{reference}' is not one of {', '.join(sorted(REFERENCES))}"
            ) from None
        if unrestricted and self.reference in ("restricted", "generalized"):
            raise ValueError(
                f"unrestricted=True (-u) and reference={reference} contradict each other"
            )
        if collinear is not None and collinear not in ("col", "ncol", "mcol"):
            raise ValueError(f"collinear is col, ncol, or mcol, not '{collinear}'")
        if not gradient_step_bohr > 0.0:
            raise ValueError("gradient_step_bohr must be positive")
        self.collinear = collinear
        self.gradient_step_bohr = float(gradient_step_bohr)
        if self.reference != "auto":
            self.notes = (f"reference: {self.reference}",)
        self.method = method
        self.basis = basis
        self.ecp = ecp
        self.unrestricted = unrestricted
        self.density_fit = density_fit
        self.x2c = x2c
        self.grid_level = grid_level
        self.threads = threads
        self.memory_mb = memory_mb
        self.scf_options = dict(scf_options or {})
        self.mol_options = dict(mol_options or {})
        self.apply = tuple(apply)
        self.setup = setup or ProgramSetup()
        self.n_states = n_states
        self.verbose = verbose
        self.name = f"pyscf:{method}/{basis if isinstance(basis, str) else 'custom'}"
        self._dm: dict[tuple[Any, ...], Any] = {}
        self._switched = False

    @property
    def is_dft(self) -> bool:
        return self.method.lower() not in ("hf", "rhf", "uhf", "scf")

    def _molecule(self, structure: Structure) -> Any:
        try:
            from pyscf import gto
        except ImportError as e:
            raise missing_package("pyscf", "pyscf") from e

        state = structure.state
        kw: dict[str, Any] = {}
        if self.memory_mb is not None:
            kw["max_memory"] = self.memory_mb
        kw.update(self.mol_options)
        mol = gto.M(
            atom=[
                (s, tuple(p))
                for s, p in zip(structure.symbols, structure.positions_bohr, strict=True)
            ],
            unit="Bohr",
            charge=state.charge,
            spin=int(state.spin_as(SpinConvention.UNPAIRED)),
            basis=self.basis,
            ecp=self.ecp,
            verbose=self.verbose,
            **kw,
        )
        # PySCF writes its output (SCF iterations, energies, <S^2>, gradients) to the
        # molecule's stream: keep it, to be shown with the evaluation it belongs to.
        self._printed = io.StringIO()
        mol.stdout = self._printed
        return mol

    def kind(self, open_shell: bool) -> str:
        """The kind of wavefunction for a closed or an open shell: ``r`` (RHF/RKS), ``ro``
        (ROHF/ROKS), ``u`` (UHF/UKS), or ``g`` (GHF/GKS)."""
        if self.reference == "generalized":
            return "g"
        if self._switched:
            return "u"
        if self.reference == "unrestricted":
            return "u"
        if self.reference == "restricted":
            return "ro" if open_shell else "r"
        return "u" if open_shell or self.unrestricted else "r"

    def _collinear(self) -> str:
        if self.collinear is not None:
            return self.collinear
        import importlib.util

        return "mcol" if importlib.util.find_spec("mcfun") is not None else "ncol"

    def _scf(self, mol: Any, open_shell: bool) -> Any:
        kind = self.kind(open_shell)
        if self.is_dft:
            mf = {"r": mol.RKS, "ro": mol.ROKS, "u": mol.UKS, "g": mol.GKS}[kind]()
            mf.xc = self.method
            mf.grids.level = self.grid_level
            if kind == "g":
                mf.collinear = self._collinear()
        else:
            mf = {"r": mol.RHF, "ro": mol.ROHF, "u": mol.UHF, "g": mol.GHF}[kind]()
        if self.x2c:
            mf = mf.x2c()
        if self.density_fit:
            mf = mf.density_fit()
        for name in self.apply:
            method = getattr(mf, name, None)
            if not callable(method):
                raise AttributeError(f"the PySCF SCF object has no method '{name}' (apply=)")
            mf = method()
        mf.max_cycle = 500 + 5 * mol.natm
        for k, v in self.scf_options.items():
            set_path(mf, k, v)
        return self.setup.run(mf)

    def _converged(self, structure: Structure) -> Any:
        """The converged SCF object of ``structure`` (started from the density of the last
        structure with these atoms and this state)."""
        if self.threads is not None:
            from pyscf import lib

            lib.num_threads(self.threads)
        mol = self._molecule(structure)
        mf = self._scf(mol, structure.state.unpaired_electrons > 0)
        key = (structure.symbols, structure.state)
        dm0 = self._dm.get(key)
        mf.kernel(dm0=dm0)
        if not mf.converged:
            mf = mf.newton()
            mf.kernel(dm0=mf.make_rdm1())
        if not mf.converged:
            raise RuntimeError("SCF did not converge")
        self._dm[key] = mf.make_rdm1()
        return mf

    def switch_to_unrestricted(self) -> None:
        """Use UHF/UKS from now on (a restricted solution was unstable towards an
        unrestricted one; ``-stability_external switch``)."""
        self._switched = True
        self.notes = (*self.notes, "switched to UHF/UKS by the stability test")

    def check_stability(self, structure: Structure, external: str) -> StabilityReport | None:
        from mopt_rs.calculators.pyscf_stability import stabilize

        return stabilize(self, structure, external)

    def orbitals(self, structure: Structure) -> dict[str, Any] | None:
        kind = self.kind(structure.state.unpaired_electrons > 0)
        if kind == "g":
            # Spin orbitals that mix alpha and beta are not sets of one spin.
            return None
        mf = self._converged(structure)
        coefficients = np.asarray(mf.mo_coeff)
        energies = np.asarray(mf.mo_energy)
        occupations = np.asarray(mf.mo_occ)
        if kind == "ro":
            # One set of orbitals, filled by the two spins up to different levels.
            sets = [
                {
                    "coefficients": coefficients,
                    "energies": energies,
                    "occupations": (occupations > threshold).astype(float),
                }
                for threshold in (0.5, 1.5)
            ]
            return {
                "overlap": np.asarray(mf.get_ovlp()),
                "sets": sets,
                "atom_of_ao": _atom_of_ao(mf.mol),
            }
        if coefficients.ndim == 3:
            sets = [
                {
                    "coefficients": coefficients[k],
                    "energies": energies[k],
                    "occupations": occupations[k],
                }
                for k in (0, 1)
            ]
        else:
            sets = [
                {"coefficients": coefficients, "energies": energies, "occupations": occupations}
            ]
        return {
            "overlap": np.asarray(mf.get_ovlp()),
            "sets": sets,
            "atom_of_ao": _atom_of_ao(mf.mol),
        }

    def transitions(self, structure: Structure, count: int) -> list[dict[str, Any]] | None:
        """The lowest ``count`` TDA states of a restricted or an unrestricted reference with
        their transition dipoles (``td.transition_dipole()``), the changes of the Mulliken
        populations of the atoms and of the dipole from the unrelaxed difference density, and the
        Mulliken populations of the transition density, for the Mulliken-Hush, generalized
        Mulliken-Hush, and fragment-charge-difference couplings of ``etmain``."""
        kind = self.kind(structure.state.unpaired_electrons > 0)
        if kind not in ("r", "u"):
            return None
        from pyscf import tdscf

        mf = self._converged(structure)
        td = tdscf.TDA(mf)
        td.nstates = count
        td.kernel()
        converged = list(td.converged)
        states = converged.index(False) if False in converged else len(converged)
        if states == 0:
            raise RuntimeError("TDA did not converge for the lowest state")
        dipoles = np.asarray(td.transition_dipole())[:states]
        mol = mf.mol
        overlap = np.asarray(mf.get_ovlp())
        atom_of_ao = np.zeros(overlap.shape[0], dtype=int)
        for atom, (_, _, start, stop) in enumerate(mol.aoslice_by_atom()):
            atom_of_ao[start:stop] = atom
        if kind == "r":
            sets = [(np.asarray(mf.mo_coeff), np.asarray(mf.mo_occ) > 0)]
        else:
            sets = [
                (np.asarray(mf.mo_coeff[s]), np.asarray(mf.mo_occ[s]) > 0) for s in range(2)
            ]
        # The dipole integrals of the electrons' positions (the origin does not matter for a
        # difference density without electrons in all).
        r_ints = np.asarray(mol.intor("int1e_r"))
        out = []
        for k in range(states):
            amplitudes = td.xy[k][0]
            blocks = [amplitudes] if kind == "r" else list(amplitudes)
            # One electron moves: unit amplitudes over both spins (a restricted singlet's
            # spatial amplitudes are scaled to unit length, the spin sum of its two equal parts).
            norm = np.sqrt(sum(float(np.sum(np.asarray(b) ** 2)) for b in blocks))
            # The transition density of the transition dipole: a restricted singlet's two spins
            # are equal, sqrt(2) times its unit spatial amplitudes.
            spins = np.sqrt(2.0) if kind == "r" else 1.0
            change_ao = np.zeros_like(overlap)
            transition_ao = np.zeros_like(overlap)
            for (c, occ), x in zip(sets, blocks, strict=True):
                x = np.asarray(x) / norm
                c_o, c_v = c[:, occ], c[:, ~occ]
                change_ao += c_v @ (x.T @ x) @ c_v.T - c_o @ (x @ x.T) @ c_o.T
                transition_ao += spins * (c_o @ x @ c_v.T)
            populations = np.diag(change_ao @ overlap)
            per_atom = np.bincount(atom_of_ao, weights=populations, minlength=mol.natm)
            symmetric = 0.5 * (transition_ao + transition_ao.T)
            transition = np.bincount(
                atom_of_ao, weights=np.diag(symmetric @ overlap), minlength=mol.natm
            )
            # The dipole of the electrons (charge -1) of the difference density.
            dipole_change = -np.einsum("ij,kji->k", change_ao, r_ints)
            out.append(
                {
                    "excitation": float(td.e[k]),
                    "transition_dipole": dipoles[k],
                    "population_change": per_atom,
                    "dipole_change": dipole_change,
                    "transition_population": transition,
                }
            )
        return out

    def state_characters(self, structure: Structure, count: int) -> dict[str, Any] | None:
        if self.kind(structure.state.unpaired_electrons > 0) != "r":
            # Singlets of single excitations on one closed-shell reference only.
            return None
        from pyscf import tdscf

        mf = self._converged(structure)
        coefficients = np.asarray(mf.mo_coeff)
        if coefficients.shape[0] != coefficients.shape[1]:
            raise RuntimeError(
                "the states can only be followed with as many orbitals as basis functions "
                f"({coefficients.shape[1]} orbitals in {coefficients.shape[0]} functions: "
                "linear dependencies were removed)"
            )
        td = tdscf.TDA(mf)
        td.nstates = count
        td.kernel()
        # The states up to the first one that did not converge.
        converged = list(td.converged)
        states = converged.index(False) if False in converged else len(converged)
        if states == 0:
            raise RuntimeError("TDA did not converge for the lowest state")
        occupied = int(np.count_nonzero(np.asarray(mf.mo_occ) > 0))
        # Loewdin orthogonalized atomic orbitals: S**0.5 C.
        values, vectors = np.linalg.eigh(np.asarray(mf.get_ovlp()))
        half = (vectors * np.sqrt(values)) @ vectors.T
        columns = []
        for k in range(states):
            x = np.asarray(td.xy[k][0]).reshape(-1)
            columns.append(x / np.linalg.norm(x))
        return {
            "orbitals": half @ coefficients,
            "occupied": occupied,
            "energies": np.asarray(td.e[:states]),
            "amplitudes": np.array(columns).T,
        }

    def _numerical_gradient(self, structure: Structure) -> Any:
        """The gradient from central differences of the SCF energy (for the methods PySCF
        has no analytic gradient of): ``6N`` energies, each SCF started from the density
        of the one before."""
        step = self.gradient_step_bohr
        key = (structure.symbols, structure.state)
        reference = self._dm.get(key)
        gradient = np.zeros(3 * structure.n_atoms)
        for dof in range(gradient.size):
            up = float(self._converged(structure.displaced(dof, step)).e_tot)
            down = float(self._converged(structure.displaced(dof, -step)).e_tot)
            gradient[dof] = (up - down) / (2.0 * step)
        if reference is not None:
            # The next structure starts from the density of this one, not of a displaced one.
            self._dm[key] = reference
        return gradient.reshape(-1, 3)

    def compute(self, structure: Structure, hessian: bool, gradient: bool = True) -> EngineResult:
        """Energy and gradient; without ``gradient`` the energy alone (an excited state then
        takes no TDA gradient, the cost of the other states of surface hopping)."""
        kind = self.kind(structure.state.unpaired_electrons > 0)
        mf = self._converged(structure)
        extra: dict[str, Any] = {"scf": mf}
        if kind in ("u", "g"):
            # How far the wavefunction is from the spin state asked for.
            extra["spin_square"] = float(mf.spin_square()[0])
        if structure.excited_state == 0 and not gradient and not hessian:
            extra["output"] = self._printed.getvalue()
            return EngineResult(float(mf.e_tot), extra=extra)
        if structure.excited_state == 0:
            energy = float(mf.e_tot)
            printed = self._printed.getvalue()
            h = None
            if kind == "g":
                grad = self._numerical_gradient(structure)
            else:
                grad = mf.nuc_grad_method().kernel()
                printed = self._printed.getvalue()
                if hessian and kind in ("r", "u"):
                    # PySCF has no Hessians of ROHF/ROKS: those come from differences.
                    h = mf.Hessian().kernel()
                    printed = self._printed.getvalue()
            extra["output"] = printed
            return EngineResult(energy, gradient=grad, hessian=_flat_hessian(h), extra=extra)
        if kind in ("ro", "g"):
            raise NotImplementedError(
                "excited states (TDA) need a restricted closed-shell or an unrestricted reference"
            )
        from pyscf import tdscf

        k = structure.excited_state
        td = tdscf.TDA(mf)
        td.nstates = max(self.n_states or k + 2, k)
        td.kernel()
        if not all(td.converged[:k]):
            raise RuntimeError(f"TDA did not converge for state {k}")
        energy = float(mf.e_tot + td.e[k - 1])
        if not gradient and not hessian:
            extra["tda"] = td
            extra["output"] = self._printed.getvalue()
            return EngineResult(energy, extra=extra)
        grad = td.nuc_grad_method().kernel(state=k)
        extra["tda"] = td
        # Excited-state Hessians come from finite differences of these gradients.
        extra["output"] = self._printed.getvalue()
        return EngineResult(energy, gradient=grad, extra=extra)


def _atom_of_ao(mol: Any) -> list[int]:
    """The atom (0-based) of every atomic orbital of ``mol`` (``Mole.aoslice_by_atom``: for each
    atom its first and last shells and atomic orbitals)."""
    out: list[int] = []
    for atom, (_, _, first, end) in enumerate(mol.aoslice_by_atom()):
        out.extend([atom] * (int(end) - int(first)))
    return out


def _flat_hessian(h: Any) -> Any:
    if h is None:
        return None
    h = np.asarray(h)
    n = h.shape[0]
    return h.transpose(0, 2, 1, 3).reshape(3 * n, 3 * n)
