"""The ASE calculators of the programs run through files, as mopt-rs sets them up, and
the readers of the Hessians of Gaussian and ORCA.

One function per program (:func:`gaussian_calculator`, :func:`orca_calculator`,
:func:`gamess_us_calculator`, :func:`nwchem_calculator`, :func:`mopac_calculator`,
:func:`gpaw_calculator`, :func:`espresso_calculator`) takes the charge, the multiplicity,
and the settings of the program and returns its ASE calculator; they can be used on their
own with ASE. :mod:`mopt_rs.calculators.programs` makes engines of them.

MultiOptPy: `Calculator/ase_tools/gaussian.py`, `orca.py`, `gamess.py`, `nwchem.py`,
`mopac.py`, `Calculator/gpaw_calculation_tools.py`, `Calculator/pwscf_calculation_tools.py`.

This file is part of mopt-py (the Python interface of mopt-rs).
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import os
import re
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

__all__ = [
    "GAUSSIAN_SCF",
    "espresso_calculator",
    "espresso_stress",
    "gamess_basis",
    "gamess_us_calculator",
    "gaussian_basis",
    "gaussian_calculator",
    "gpaw_calculator",
    "mopac_calculator",
    "nwchem_calculator",
    "nwchem_memory",
    "orca_basis_block",
    "orca_calculator",
    "read_gaussian_hessian",
    "read_orca_hessian",
]

FloatArray = npt.NDArray[np.float64]

GAUSSIAN_SCF = "xqc,maxcon=128,maxcyc=32,conver=8"
"""MultiOptPy's SCF settings for Gaussian."""


def _triangle(text: str, start: int, first_index: int, what: str) -> FloatArray:
    """A symmetric matrix printed in blocks of columns, each block headed by a line of
    column numbers and followed by lines of a row number and values (the lower triangle
    only, or full rows), read from line ``start`` of ``text``; indices start at
    ``first_index``."""
    entries: dict[tuple[int, int], float] = {}
    columns: list[int] = []
    for line in text.splitlines()[start:]:
        parts = line.split()
        if not parts:
            if entries:
                break
            continue
        if all(p.isdigit() for p in parts):
            columns = [int(p) - first_index for p in parts]
            continue
        if not parts[0].isdigit() or not columns:
            break
        try:
            values = [float(v.replace("D", "E").replace("d", "e")) for v in parts[1:]]
        except ValueError:
            break
        if len(values) > len(columns):
            raise ValueError(f"{what}: a row has more values than its block has columns")
        row = int(parts[0]) - first_index
        for column, value in zip(columns, values, strict=False):
            entries[(row, column)] = value
    if not entries:
        raise ValueError(f"{what}: no values found")
    n = 1 + max(max(i, j) for i, j in entries)
    h = np.full((n, n), np.nan)
    for (i, j), value in entries.items():
        h[i, j] = value
    # A lower triangle is completed by symmetry; full rows are averaged with their mirror.
    lower = np.where(np.isnan(h), h.T, h)
    if np.isnan(lower).any():
        raise ValueError(f"{what}: the matrix of {n} rows is incomplete")
    return np.asarray(0.5 * (lower + lower.T), dtype=np.float64)


def read_gaussian_hessian(text: str) -> FloatArray:
    """The Cartesian force constants (Hartree/Bohr^2) of a Gaussian output: the last
    section ``Force constants in Cartesian coordinates:`` (printed by a frequency job with
    ``#P``), a lower triangle in blocks of five columns."""
    lines = text.splitlines()
    starts = [
        k for k, line in enumerate(lines) if "Force constants in Cartesian coordinates" in line
    ]
    if not starts:
        raise ValueError(
            "no 'Force constants in Cartesian coordinates' in the Gaussian output (they are "
            "printed by a frequency job with #P)"
        )
    return _triangle(text, starts[-1] + 1, 1, "Gaussian force constants")


def read_orca_hessian(text: str) -> FloatArray:
    """The Cartesian Hessian (Hartree/Bohr^2) of an ORCA ``.hess`` file: the block
    ``$hessian``, full rows in blocks of columns, indices from zero."""
    lines = text.splitlines()
    starts = [k for k, line in enumerate(lines) if line.strip().lower() == "$hessian"]
    if not starts:
        raise ValueError("no $hessian block in the ORCA .hess file")
    start = starts[-1]
    try:
        n = int(lines[start + 1].split()[0])
    except (IndexError, ValueError):
        raise ValueError("the $hessian block of the ORCA .hess file has no dimension") from None
    h = _triangle(text, start + 2, 0, "ORCA Hessian")
    if h.shape != (n, n):
        raise ValueError(f"the ORCA Hessian has {h.shape[0]} rows, {n} were announced")
    return h


def gaussian_basis(basis: str) -> str:
    """``basis`` as Gaussian names it: the def2 sets are written without a hyphen
    (``def2-SVP`` is ``def2SVP``); other names are passed on."""
    return re.sub(r"(?i)^def2-", "def2", basis.strip())


_GAMESS_BASIS = re.compile(
    r"(?i)^(?P<split>3-21|6-31|6-311)(?P<diffuse>\+{0,2})g(?:\((?P<polar>[^)]*)\)|(?P<stars>\*{0,2}))$"
)


def gamess_basis(basis: str) -> dict[str, Any]:
    """The ``$BASIS`` keywords of GAMESS-US for a basis set named as usual: STO-3G, the
    Pople sets (``6-31G(d)``, ``6-311+G(d,p)``, ``6-31G**``, ...), and the correlation
    consistent sets (``cc-pVDZ``, ``aug-cc-pVTZ``, ...). Other sets are given as keywords
    (``-calc_opt basis.gbasis=...``)."""
    name = basis.strip()
    if name.lower() == "sto-3g":
        return {"gbasis": "STO", "ngauss": 3}
    cc = re.fullmatch(r"(?i)(aug-)?cc-pv([dtq56])z", name)
    if cc:
        return {"gbasis": ("ACC" if cc.group(1) else "CC") + cc.group(2).upper()}
    m = _GAMESS_BASIS.match(name)
    if not m:
        raise ValueError(
            f"GAMESS-US: no $BASIS keywords known for '{basis}'; give them as "
            "-calc_opt basis.gbasis=... basis.ngauss=..."
        )
    split = m.group("split")
    out: dict[str, Any] = {
        "gbasis": {"3-21": "N21", "6-31": "N31", "6-311": "N311"}[split],
        "ngauss": 3 if split == "3-21" else 6,
    }
    polar = m.group("polar")
    if polar is None:
        stars = len(m.group("stars") or "")
        polar = ",".join(["d", "p"][:stars])
    for part in filter(None, (p.strip().lower() for p in polar.split(","))):
        count = re.fullmatch(r"(\d*)([dpf])", part)
        if not count:
            raise ValueError(f"GAMESS-US: polarization '{part}' of '{basis}' is not understood")
        key = {"d": "ndfunc", "p": "npfunc", "f": "nffunc"}[count.group(2)]
        out[key] = int(count.group(1) or 1)
    diffuse = len(m.group("diffuse"))
    if diffuse >= 1:
        out["diffsp"] = True
    if diffuse == 2:
        out["diffs"] = True
    return out


def nwchem_memory(memory: str) -> str:
    """``2GB`` as NWChem writes it (``2 gb``)."""
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([A-Za-z]+)\s*", memory)
    if not m:
        raise ValueError(f"memory '{memory}' is not a number with a unit (2GB, 500MB)")
    return f"{m.group(1)} {m.group(2).lower()}"


def _is_pople(basis: str) -> bool:
    return basis.strip().lower().startswith(("6-31", "6-311", "3-21"))


def orca_basis_block(symbols: Sequence[str], basis: str, heavy_atom_basis: str) -> str:
    """ORCA has the Pople basis sets up to zinc only: for heavier elements in ``symbols``
    a ``%basis`` block assigns ``heavy_atom_basis`` (no effective core potential is
    added, as in MultiOptPy); empty when nothing is needed."""
    from ase.data import atomic_numbers

    if not _is_pople(basis):
        return ""
    heavy = sorted({s for s in symbols if atomic_numbers.get(s, 0) > 30})
    if not heavy:
        return ""
    lines = ["%basis", *(f'  NewGTO {s} "{heavy_atom_basis}"' for s in heavy), "end"]
    return "\n".join(lines)


def _executable(path: str | None, *names: str) -> str:
    """The program to run: ``path`` if it is a file, the first of ``names`` found in
    ``path`` if it is a directory, else the first of ``names`` on the search path."""
    if path:
        clean = os.path.normpath(os.path.expanduser(path.strip()))
        if os.path.isdir(clean):
            for name in names:
                for candidate in (name, f"{name}.exe"):
                    full = os.path.join(clean, candidate)
                    if os.path.isfile(full):
                        return full
            raise FileNotFoundError(f"none of {', '.join(names)} in {clean}")
        if os.path.isfile(clean):
            return os.path.abspath(clean)
        found = shutil.which(clean)
        if found:
            return found
        raise FileNotFoundError(f"no program at {clean!r}")
    for name in names:
        found = shutil.which(name)
        if found:
            return found
    return names[0]


def _ase_configured(calculator: Any) -> bool:
    """Whether ASE's own configuration (``ASE_NAME_COMMAND`` or a section of its
    configuration file) says how to run ``calculator``."""
    cfg = getattr(calculator, "cfg", None)
    name = str(getattr(calculator, "name", calculator.__name__.lower()))
    if cfg is None:
        return f"ASE_{name.upper()}_COMMAND" in os.environ
    try:
        return cfg.get(f"ASE_{name.upper()}_COMMAND") is not None or name in cfg.parser
    except (AttributeError, TypeError):
        return False


def _conversion() -> float:
    """Hartree/Bohr^2 in eV/Angstrom^2."""
    from ase import units

    return float(units.Hartree / units.Bohr**2)


def gaussian_calculator(
    *,
    charge: int = 0,
    mult: int = 1,
    xc: str = "b3lyp",
    basis: str = "def2SVP",
    mem: str | None = None,
    nprocshared: int | None = None,
    program: str = "g16",
    path: str | None = None,
    directory: str = ".",
    label: str = "Gaussian",
    scf: str | None = GAUSSIAN_SCF,
    symmetry: bool = False,
    hessian_route: str = "freq=noraman",
    **route: Any,
) -> Any:
    """ASE's Gaussian calculator with MultiOptPy's settings and a ``get_hessian`` that runs
    a frequency job (``hessian_route``, ``freq=noraman``) and reads the force constants
    from its output.

    ``path`` is the directory of the program (``GAUSS_EXEDIR``) or the program itself;
    ``route`` adds keywords to the route section (``empiricaldispersion="gd3bj"``) and
    takes the other keywords of ASE's calculator (``extra``, ``addsec``, ``chk``, ...).
    ``symmetry=True`` leaves Gaussian's reorientation on.
    """
    from ase.calculators.gaussian import Gaussian

    nested = [k for k, v in route.items() if isinstance(v, Mapping)]
    if nested:
        raise ValueError(
            f"Gaussian: {', '.join(nested)}: the route has no groups; write "
            "KEY=VALUE or KEY=OPTION,OPTION=VALUE (scrf=pcm,solvent=water)"
        )

    class GaussianProgram(Gaussian):  # type: ignore[misc]
        def get_hessian(self, atoms: Any) -> FloatArray:
            saved = self.parameters.get("extra")
            self.parameters["extra"] = " ".join(x for x in (saved, hessian_route) if x)
            try:
                self.calculate(atoms, properties=["energy"])
                text = Path(f"{self.label}.log").read_text(encoding="utf-8", errors="replace")
            finally:
                if saved is None:
                    self.parameters.pop("extra", None)
                else:
                    self.parameters["extra"] = saved
                self.results = dict[str, Any]()
            return read_gaussian_hessian(text) * _conversion()

    if path and os.path.isdir(os.path.expanduser(path.strip())):
        os.environ["GAUSS_EXEDIR"] = os.path.expanduser(path.strip())
    executable = _executable(path, program)
    command = (
        f'"{executable}" PREFIX.com PREFIX.log'
        if os.name == "nt"
        else f"{executable} < PREFIX.com > PREFIX.log"
    )
    keywords: dict[str, Any] = {
        "xc": xc,
        "basis": gaussian_basis(basis),
        "mem": mem,
        "nprocshared": nprocshared,
        "scf": scf,
    }
    keywords = {k: v for k, v in keywords.items() if v is not None}
    if not symmetry:
        keywords["nosymm"] = ""
    keywords.update(route)
    return GaussianProgram(
        label=label, directory=directory, command=command, charge=charge, mult=mult, **keywords
    )


def orca_calculator(
    *,
    charge: int = 0,
    mult: int = 1,
    method: str = "B3LYP",
    basis: str = "def2-SVP",
    simple: str = "",
    blocks: str = "",
    nprocs: int | None = None,
    maxcore: int | None = None,
    path: str | None = None,
    directory: str = ".",
    auto_fix_basis: bool = True,
    heavy_atom_basis: str = "def2-SVP",
    hessian_task: str = "Freq",
) -> Any:
    """ASE's ORCA calculator with MultiOptPy's input (``! method basis EnGrad``, the
    ``%basis`` block for elements the Pople sets lack) and a ``get_hessian`` that runs a
    frequency job (``hessian_task``; ``NumFreq`` for methods without analytic second
    derivatives) and reads ``orca.hess``.

    ``simple`` adds keywords to the ``!`` line, ``blocks`` input blocks; ``nprocs`` and
    ``maxcore`` (MB per process) write ``%pal`` and ``%maxcore``. ``path`` is the ORCA
    program or its directory (which is put on the search path, as ORCA needs for its
    parallel runs).
    """
    from ase.calculators.orca import ORCA, OrcaProfile

    def input_blocks(atoms: Any) -> str:
        parts = []
        if nprocs and nprocs > 1:
            parts.append(f"%pal nprocs {nprocs} end")
        if maxcore:
            parts.append(f"%maxcore {maxcore}")
        if blocks.strip():
            parts.append(blocks.strip())
        fix = orca_basis_block(atoms.get_chemical_symbols(), basis, heavy_atom_basis)
        if fix and not auto_fix_basis:
            raise ValueError(
                f"ORCA has the basis set {basis} only up to zinc; choose another basis set "
                "or leave auto_fix_basis on"
            )
        if fix:
            parts.append(fix)
        return "\n".join(parts)

    class OrcaProgram(ORCA):  # type: ignore[misc]
        task = "EnGrad"
        stability_next: str | None = None
        """Set by the stability test (``-stability``): the next input has ORCA test its SCF
        solution (``%scf STABPerform true``; with ``switch`` also
        ``STABRestartUHFifUnstable true``, a restart as UHF/UKS where the solution is
        unstable)."""

        def write_inputfiles(self, atoms: Any, properties: Any) -> None:
            words = (method, basis, simple.strip(), self.task)
            self.parameters["orcasimpleinput"] = " ".join(w for w in words if w)
            blocks = input_blocks(atoms)
            if self.stability_next is not None:
                restart = (
                    " STABRestartUHFifUnstable true" if self.stability_next == "switch" else ""
                )
                blocks = "\n".join(b for b in (blocks, f"%scf STABPerform true{restart} end") if b)
                self.stability_next = None
            self.parameters["orcablocks"] = blocks
            super().write_inputfiles(atoms, properties)

        def get_hessian(self, atoms: Any) -> FloatArray:
            self.task = hessian_task
            try:
                self.calculate(atoms, ["energy"], [])
                text = (Path(self.directory) / "orca.hess").read_text(
                    encoding="utf-8", errors="replace"
                )
            finally:
                self.task = "EnGrad"
                self.results = dict[str, Any]()
            return read_orca_hessian(text) * _conversion()

    profile = None
    if path:
        executable = _executable(path, "orca")
        folder = os.path.dirname(executable)
        if folder and folder not in os.environ.get("PATH", "").split(os.pathsep):
            os.environ["PATH"] = folder + os.pathsep + os.environ.get("PATH", "")
        profile = OrcaProfile(command=executable.replace("\\", "/"))
    return OrcaProgram(profile=profile, directory=directory, charge=charge, mult=mult)


def gamess_us_calculator(
    *,
    charge: int = 0,
    mult: int = 1,
    xc: str | None = "B3LYP",
    basis: Mapping[str, Any] | str | None = None,
    userscr: str | None = None,
    directory: str = ".",
    label: str = "gamess_us",
    command: str | None = None,
    contrl: Mapping[str, Any] | None = None,
    **groups: Any,
) -> Any:
    """ASE's GAMESS-US calculator: the charge and the multiplicity go to ``$CONTRL``
    (``ICHARG``, ``MULT``; ``SCFTYP`` is RHF for singlets and UHF otherwise unless given),
    the functional to ``DFTTYP`` (``xc=None``: no DFT), ``basis`` is a set named as usual
    (see :func:`gamess_basis`) or ``$BASIS`` keywords, and ``groups`` are further input
    groups (``system={"mwords": 100}``). ``userscr`` is the USERSCR directory of ``rungms``.
    """
    from ase.calculators.gamess_us import GAMESSUS

    control: dict[str, Any] = {"icharg": int(charge), "mult": int(mult)}
    control["scftyp"] = "rhf" if int(mult) == 1 else "uhf"
    if xc:
        control["dfttyp"] = xc.upper()
    control.update(contrl or {})
    keywords: dict[str, Any] = dict(groups)
    if isinstance(basis, str):
        keywords["basis"] = gamess_basis(basis)
    elif basis:
        keywords["basis"] = dict(basis)
    return GAMESSUS(
        label=label,
        directory=directory,
        command=command,
        userscr=userscr,
        contrl=control,
        **keywords,
    )


def nwchem_calculator(
    *,
    charge: int = 0,
    mult: int = 1,
    xc: str | None = "b3lyp",
    basis: str | Mapping[str, str] = "def2-svp",
    memory: str | None = None,
    directory: str = ".",
    label: str = "nwchem",
    command: str | None = None,
    dft: Mapping[str, Any] | None = None,
    **keywords: Any,
) -> Any:
    """ASE's NWChem calculator: the multiplicity goes to the ``dft`` block (``mult``, and
    ``odft`` for open shells), ``memory`` is written as NWChem expects (``2 gb``);
    ``keywords`` are further blocks and special keywords of the ASE calculator
    (``theory="mp2"``, ``set={...}``)."""
    from ase.calculators.nwchem import NWChem

    block: dict[str, Any] = {}
    if int(mult) != 1:
        block.update({"mult": int(mult), "odft": None})
    block.update(dft or {})
    kw: dict[str, Any] = {"charge": int(charge), "basis": basis, **keywords}
    if xc and "theory" not in keywords:
        kw["xc"] = xc
    if block:
        kw["dft"] = block
    if memory:
        kw["memory"] = nwchem_memory(memory)
    return NWChem(label=label, directory=directory, command=command, **kw)


def mopac_calculator(
    *,
    charge: int = 0,
    mult: int = 1,
    method: str = "PM7",
    task: str = "1SCF GRADIENTS DISP",
    keywords: str = "",
    threads: int | None = None,
    relscf: float | None = None,
    directory: str = ".",
    label: str = "mopac",
    path: str | None = None,
    command: str | None = None,
) -> Any:
    """ASE's MOPAC calculator. ASE takes the multiplicity from initial magnetic moments,
    not from a keyword: they are set here from ``mult`` on the structure that is written
    (``DOUBLET UHF``, ...).

    ASE writes ``method``, ``task``, ``RELSCF``, the charge, and the spin to the keyword
    line and nothing else: ``keywords`` is appended to that line as it is (``EPS=78.4``,
    ``PRECISE``, ``MOZYME``, ...), and ``threads`` writes ``THREADS=N``.

    ``path`` is the program or its directory. Unless ``command`` is given or ASE is
    configured for MOPAC, the program is run with what it prints to the screen thrown
    away (it reports the end of every job there); its results are in ``label.out``."""
    from ase.calculators.mopac import MOPAC

    unpaired = int(mult) - 1
    if command is None and (path or not _ase_configured(MOPAC)):
        command = f'"{_executable(path, "mopac")}" PREFIX.mop > {os.devnull} 2>&1'
    words = [task.strip(), keywords.strip()]
    if threads and threads > 0:
        words.append(f"THREADS={int(threads)}")
    settings: dict[str, Any] = {"method": method, "task": " ".join(w for w in words if w)}
    if relscf is not None:
        settings["relscf"] = relscf

    class MopacProgram(MOPAC):  # type: ignore[misc]
        def write_input(
            self, atoms: Any, properties: Any = None, system_changes: Any = None
        ) -> None:
            written = atoms.copy()
            moments = np.zeros(len(written))
            if len(written):
                moments[0] = unpaired
            written.set_initial_magnetic_moments(moments)
            super().write_input(written, properties, system_changes)

    return MopacProgram(
        label=label,
        directory=directory,
        command=command,
        charge=int(charge),
        **settings,
    )


def gpaw_calculator(*, charge: int = 0, vacuum: float = 6.0, **keywords: Any) -> Any:
    """GPAW (``mode="lcao"`` and no text output unless given, as MultiOptPy sets it).
    For a multiplicity other than one the engine gives the atoms magnetic moments that add
    up to the number of unpaired electrons, which makes the calculation spin-polarized;
    a structure without a cell is put in a box with ``vacuum`` Angstrom around it by the
    engine. ``keywords`` are GPAW's own (``mode={"name": "pw", "ecut": 400}``, ``h``,
    ``kpts``, ``basis``, ``occupations``, ``txt``, ...).

    The symmetry of the point group is not used (``symmetry={"point_group": False}``)
    unless ``symmetry`` is given: GPAW keeps the symmetry of the first structure and stops
    when a later one breaks it, which every optimization away from a symmetric start
    does."""
    from gpaw import GPAW

    del vacuum  # used by the engine, see FileProgramEngine.atoms
    settings: dict[str, Any] = {
        "mode": "lcao",
        "txt": None,
        "symmetry": {"point_group": False},
        **keywords,
    }
    if charge:
        settings["charge"] = charge
    return GPAW(**settings)


def espresso_calculator(
    *,
    charge: int = 0,
    mult: int = 1,
    xc: str | None = None,
    kpts: Any = (1, 1, 1),
    pseudopotentials: Mapping[str, str] | None = None,
    pseudo_dir: str | None = None,
    command: str | None = None,
    directory: str = ".",
    input_data: Mapping[str, Any] | None = None,
    vacuum: float = 6.0,
    **keywords: Any,
) -> Any:
    """ASE's Quantum ESPRESSO calculator (``pw.x``): ``pseudopotentials`` maps elements to
    pseudopotential files in ``pseudo_dir``; ``command`` is the ``pw.x`` command line
    (``mpirun -n 4 pw.x``). Forces are asked for (``tprnfor``). A charge is written as
    ``tot_charge``, a multiplicity other than one as ``nspin = 2`` with
    ``tot_magnetization`` (the engine also gives the atoms magnetic moments, from which ASE
    writes ``starting_magnetization``); what ``input_data`` says about these wins."""
    from ase.calculators.espresso import Espresso, EspressoProfile

    del vacuum
    if not pseudopotentials:
        raise ValueError(
            "Quantum ESPRESSO needs pseudopotentials: -calc_opt "
            "pseudopotentials={'Si':'Si.pbe-n-rrkjus_psl.1.0.0.UPF'} pseudo_dir=/path"
        )
    data: dict[str, Any] = {"control": {"tprnfor": True}}
    for name, value in (input_data or {}).items():
        if isinstance(value, Mapping):
            data.setdefault(name, {}).update(value)
        elif "." in name:
            # input_data.system.ecutwfc=40 from the command line.
            section, key = name.split(".", 1)
            data.setdefault(section, {})[key] = value
        else:
            # A keyword without its section: ASE sorts it into the right one.
            data[name] = value
    settings: dict[str, Any] = {
        "kpts": kpts,
        "pseudopotentials": dict(pseudopotentials),
        "input_data": data,
        "directory": directory,
        **keywords,
    }
    system = data.setdefault("system", {})
    if xc:
        system["input_dft"] = xc

    def unset(key: str) -> bool:
        return key not in data and key not in system

    if int(charge) and unset("tot_charge"):
        system["tot_charge"] = int(charge)
    if int(mult) != 1:
        if unset("nspin"):
            system["nspin"] = 2
        if unset("tot_magnetization"):
            system["tot_magnetization"] = int(mult) - 1
    if command or pseudo_dir:
        settings["profile"] = EspressoProfile(
            command=command or "pw.x", pseudo_dir=pseudo_dir or "."
        )
    return Espresso(**settings)


def espresso_stress(calculator: Any, wanted: bool) -> None:
    """Make the next run of an :func:`espresso_calculator` print the stress (``tstress``)
    or not: ``pw.x`` computes it only when asked, and ASE does not ask on its own."""
    control = calculator.parameters["input_data"].setdefault("control", {})
    if wanted:
        control["tstress"] = True
    else:
        control.pop("tstress", None)
