"""Programs that run through input and output files, by way of their ASE calculators.

``-os gaussian``, ``orca``, ``gamessus``, ``nwchem``, ``mopac`` (as in MultiOptPy) and
``gpaw``, ``espresso`` (``pwscf``, ``qe``). Each is an ASE calculator set up from the
program-independent options (``-func``, ``-bs``, ``-mem``, ``-core``, the program path of
``software_path.conf``) and from ``-calc_opt`` for everything else, so nothing about a
program is fixed in the code beyond how its input is laid out.

Settings: :func:`describe_program` (``python -m mopt_rs.calculators.programs orca``) lists
what each program takes. ``group.KEY=value`` addresses an input group or block of a program
(``contrl.maxit=200``, ``dft.grid=fine``, ``input_data.system.ecutwfc=40``), ``KEY=@FILE``
gives the text of a file as the value, and defaults for a program can be kept in a file
(:mod:`mopt_rs.calculators.option_files`).

Hessians: Gaussian (``freq``) and ORCA (``Freq``) compute them analytically where the
method allows, and their output is read (:func:`read_gaussian_hessian`,
:func:`read_orca_hessian`); the other programs get finite differences of the forces.

The functions that build the ASE calculators and the readers are in
:mod:`mopt_rs.calculators.program_inputs`.

What differs from MultiOptPy (`Calculator/ase_tools/`):

- The multiplicity reaches every program. MultiOptPy passes ``mult`` to the ASE calculators
  of MOPAC, NWChem, and GAMESS-US, which do not read it there (they take the spin from
  initial magnetic moments or from their own input groups), so open shells ran as singlets.
- The charge reaches GAMESS-US (``$CONTRL ICHARG``).
- Gaussian runs without reorientation (``nosymm``), so that forces and force constants are
  in the frame of the input coordinates whatever the symmetry.
- One calculation at a time per engine: the programs work in one directory with fixed file
  names.
- The basis set of ``-bs`` is translated where a program names it differently (``def2-SVP``
  is ``def2SVP`` in Gaussian; GAMESS-US takes ``$BASIS`` keywords).

MultiOptPy: `Calculator/ase_tools/gaussian.py`, `orca.py`, `gamess.py`, `nwchem.py`,
`mopac.py`, `Calculator/gpaw_calculation_tools.py`, `Calculator/pwscf_calculation_tools.py`.

This file is part of mopt-py (the Python interface of mopt-rs).
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import inspect
import os
import sys
import threading
import warnings
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

from mopt_rs.calculators.ase import AseEngine, StateMapping
from mopt_rs.calculators.base import EngineResult, SpinConvention, Structure, worker_index
from mopt_rs.calculators.program_inputs import (
    GAUSSIAN_SCF,
    espresso_calculator,
    espresso_stress,
    gamess_basis,
    gamess_us_calculator,
    gaussian_basis,
    gaussian_calculator,
    gpaw_calculator,
    mopac_calculator,
    nwchem_calculator,
    nwchem_memory,
    orca_basis_block,
    orca_calculator,
    read_gaussian_hessian,
    read_orca_hessian,
)
from mopt_rs.calculators.program_options import split_program_options
from mopt_rs.calculators.stability import StabilityReport

if TYPE_CHECKING:
    from mopt_rs.calculators.registry import CalculatorOptions

__all__ = [
    "PROGRAMS",
    "FileProgramEngine",
    "Program",
    "describe_program",
    "gamess_basis",
    "gaussian_basis",
    "nwchem_memory",
    "orca_basis_block",
    "program_engine",
    "read_gaussian_hessian",
    "read_orca_hessian",
]

MULT = StateMapping("charge", "mult", SpinConvention.MULTIPLICITY)

ASE_CITATION = (
    "A. H. Larsen et al., J. Phys.: Condens. Matter 2017, 29, 273002 (ASE, through which the "
    "program is run)"
)
"""The interface every program of this module runs through."""


class FileProgramEngine(AseEngine):
    """An :class:`AseEngine` for a program that works in a directory with fixed file names:
    one calculation at a time. ``vacuum`` (Angstrom) puts a structure without a cell in a
    box, for programs that need one (plane waves, real-space grids): the box of the first
    structure, ``vacuum`` away from the atoms on every side, is kept for the structures
    that follow, so that the grid or the plane waves stay the same along a run; when atoms
    come closer to a wall than :data:`BOX_MARGIN` of ``vacuum``, a new box is made (with a
    warning: the energy surface has a step there). ``moments`` gives the
    atoms initial magnetic moments that add up to the number of unpaired electrons, for
    programs that take the spin from them. ``stress_setup`` switches the stress of a
    program on for the runs that need it (see :class:`Program`)."""

    def __init__(
        self,
        *args: Any,
        citations: tuple[str, ...] = (),
        vacuum: float | None = None,
        moments: bool = False,
        stress_setup: Callable[[Any, bool], None] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.citations = (*citations, ASE_CITATION)
        self.vacuum = vacuum
        self.moments = moments
        self.stress_setup = stress_setup
        self._lock = threading.Lock()
        self._box: tuple[tuple[str, ...], Any, Any] | None = None

    def _put_in_box(self, atoms: Any, vacuum: float) -> None:
        """``atoms`` in the box of the run (see the class)."""
        symbols = tuple(atoms.get_chemical_symbols())
        positions = atoms.get_positions()
        box = self._box
        if box is not None and box[0] == symbols:
            _, cell, shift = box
            moved = positions + shift
            lengths = np.diag(cell)
            margin = min(float(moved.min()), float((lengths - moved).min()))
            if margin >= BOX_MARGIN * vacuum:
                atoms.set_cell(cell)
                atoms.set_positions(moved)
                return
            warnings.warn(
                f"{self.name}: atoms came within {margin:.2f} Angstrom of the box "
                f"(vacuum {vacuum:g}); a new box is made, and the energies before and "
                "after it differ by the change of the grid or of the plane waves",
                stacklevel=2,
            )
        atoms.center(vacuum=vacuum)
        self._box = (symbols, atoms.cell.array.copy(), atoms.get_positions() - positions)

    def atoms(self, structure: Structure) -> Any:
        atoms = super().atoms(structure)
        if self.vacuum is not None and not structure.is_periodic:
            self._put_in_box(atoms, self.vacuum)
        unpaired = structure.state.multiplicity - 1
        if self.moments and unpaired and len(atoms):
            atoms.set_initial_magnetic_moments(np.full(len(atoms), unpaired / len(atoms)))
        return atoms

    def check_stability(self, structure: Structure, external: str) -> StabilityReport | None:
        """Programs that test their own SCF solution (ORCA: ``%scf STABPerform``) do it in
        the repeated evaluation; the others have no test (``None``)."""
        calc = self.calculator(structure.state)
        if not hasattr(calc, "stability_next"):
            return None
        calc.stability_next = external
        return StabilityReport(
            True,
            [
                f"SCF stability: {self.name} tests its solution in the repeated evaluation "
                f"(%scf STABPerform{' with a restart as UHF/UKS' if external == 'switch' else ''})"
            ],
        )

    def compute(self, structure: Structure, hessian: bool, stress: bool = False) -> EngineResult:
        with self._lock:
            if self.stress_setup is not None:
                wanted = stress and structure.is_periodic
                self.stress_setup(self.calculator(structure.state), wanted)
            return super().compute(structure, hessian, stress)


def _paths(o: CalculatorOptions, *names: str) -> str | None:
    for name in names:
        path = o.software_paths.get(name)
        if path:
            return path
    return None


def _gaussian(o: CalculatorOptions, plain: dict[str, Any]) -> dict[str, Any]:
    kw: dict[str, Any] = {
        "xc": o.method or "b3lyp",
        "basis": o.basis or "def2-SVP",
        "mem": o.memory,
        "nprocshared": o.threads if o.threads and o.threads > 1 else None,
        "path": _paths(o, "gaussian", "g16"),
    }
    return {**{k: v for k, v in kw.items() if v is not None}, **plain}


def _orca(o: CalculatorOptions, plain: dict[str, Any]) -> dict[str, Any]:
    kw: dict[str, Any] = {
        "method": o.method or "B3LYP",
        "basis": o.basis or "def2-SVP",
        "nprocs": o.threads if o.threads and o.threads > 1 else None,
        "path": _paths(o, "orca"),
    }
    return {**{k: v for k, v in kw.items() if v is not None}, **plain}


def _gamess(o: CalculatorOptions, plain: dict[str, Any]) -> dict[str, Any]:
    kw: dict[str, Any] = {
        "xc": o.method or "B3LYP",
        "basis": o.basis,
        "userscr": _paths(o, "gamessus", "gamess"),
    }
    return {**{k: v for k, v in kw.items() if v is not None}, **plain}


def _nwchem(o: CalculatorOptions, plain: dict[str, Any]) -> dict[str, Any]:
    kw: dict[str, Any] = {
        "xc": o.method or "b3lyp",
        "basis": o.basis_spec() or "def2-svp",
        "memory": o.memory,
    }
    if isinstance(kw["basis"], dict):
        # NWChem's name for "every other element".
        kw["basis"] = {("*" if k == "default" else k): v for k, v in kw["basis"].items()}
    return {**{k: v for k, v in kw.items() if v is not None}, **plain}


def _mopac(o: CalculatorOptions, plain: dict[str, Any]) -> dict[str, Any]:
    kw: dict[str, Any] = {"method": o.method} if o.method else {}
    path = _paths(o, "mopac")
    if path:
        kw["path"] = path
    if o.threads and o.threads > 0:
        kw["threads"] = o.threads
    return {**kw, **plain}


def _gpaw(o: CalculatorOptions, plain: dict[str, Any]) -> dict[str, Any]:
    kw: dict[str, Any] = {"xc": o.method} if o.method else {}
    return {**kw, **plain}


def _espresso(o: CalculatorOptions, plain: dict[str, Any]) -> dict[str, Any]:
    kw: dict[str, Any] = {"xc": o.method} if o.method else {}
    pseudo = _paths(o, "espresso_pseudo", "pseudo_dir")
    if pseudo:
        kw["pseudo_dir"] = pseudo
    command = _paths(o, "espresso", "pwscf", "pw.x")
    if command:
        kw["command"] = command
    return {**kw, **plain}


class Program(NamedTuple):
    """One program run through files."""

    build: Callable[..., Any]
    """The function that builds its ASE calculator."""
    keywords: Callable[[CalculatorOptions, dict[str, Any]], dict[str, Any]]
    """The program-independent options and the plain ``-calc_opt`` keys as its keywords."""
    state: StateMapping
    """How it takes the charge and the spin."""
    hessian: str | None
    """Its Hessian method (``None``: finite differences of the forces)."""
    title: str
    """What ``-func``, ``-bs``, ``-mem``, ``-core``, and ``software_path.conf`` set."""
    options: Mapping[str, str]
    """Its ``-calc_opt`` settings and what they do; ``OTHER`` says where the rest goes."""
    citations: tuple[str, ...]
    moments: bool = False
    """The spin is given as initial magnetic moments of the atoms."""
    cell: bool = False
    """It needs a cell: a structure without one is put in a box (``vacuum``)."""
    stress_setup: Callable[[Any, bool], None] | None = None
    """Called with the ASE calculator before a run, to switch the stress on or off, for a
    program that computes it only when its input says so."""


BOX_MARGIN = 0.5
"""A new box is made when atoms are closer to a wall than this share of the vacuum."""

OTHER = "(other keys)"
"""Key of :attr:`Program.options` for the settings that are passed on as they are."""

_DIRECTORY = "directory the program works in (the current one)"

PROGRAMS: dict[str, Program] = {
    "gaussian": Program(
        gaussian_calculator,
        _gaussian,
        MULT,
        "get_hessian",
        "-func and -bs: method and basis of the route (def2-SVP is written def2SVP); "
        "-mem: %mem; -core: %nprocshared; software_path.conf gaussian:: the directory of "
        "the program (GAUSS_EXEDIR) or the program",
        {
            "program": "name of the program (g16; g09, ...)",
            "scf": f"options of scf(...) ({GAUSSIAN_SCF}); None: Gaussian's own",
            "symmetry": "True: leave Gaussian's reorientation on (nosymm is written otherwise)",
            "hessian_route": "route keyword of the Hessian job (freq=noraman)",
            "extra": "text added to the route as it is",
            "addsec": "text after the structure (basis sets of gen, ModRedundant, ...); "
            "@FILE reads it from a file",
            "chk": "checkpoint file (%chk); the other Link 0 keywords work the same way",
            "label": "name of the input and output files (Gaussian)",
            "directory": _DIRECTORY,
            OTHER: "route keywords: KEY=VALUE is written KEY(VALUE) "
            "(empiricaldispersion=gd3bj, scrf=pcm,solvent=water, int=ultrafine), and the "
            "other keywords of ASE's Gaussian calculator (basisfile, ioplist, ...)",
        },
        (
            "M. J. Frisch et al., Gaussian 16, Gaussian, Inc., Wallingford CT, 2016 (cite the "
            "revision that was run)",
        ),
    ),
    "orca": Program(
        orca_calculator,
        _orca,
        MULT,
        "get_hessian",
        "-func and -bs: the method and the basis of the ! line; -core: %pal nprocs; "
        "software_path.conf orca:: the program or its directory",
        {
            "simple": 'more keywords of the ! line ("RIJCOSX D3BJ TightSCF")',
            "blocks": 'input blocks ("%scf maxiter 300 end"); @FILE reads them from a file',
            "nprocs": "processes (%pal nprocs N end)",
            "maxcore": "memory per process in MB (%maxcore)",
            "hessian_task": "keyword of the Hessian job (Freq; NumFreq for methods without "
            "analytic second derivatives)",
            "auto_fix_basis": "False: a Pople basis set with elements beyond zinc is an "
            "error instead of getting heavy_atom_basis for them",
            "heavy_atom_basis": "basis set of those elements (def2-SVP)",
            "path": "the program or its directory (as software_path.conf)",
            "directory": _DIRECTORY,
        },
        ("F. Neese, WIREs Comput. Mol. Sci. 2022, 12, e1606 (ORCA 5); 2025, 15, e70019 (ORCA 6)",),
    ),
    "gamessus": Program(
        gamess_us_calculator,
        _gamess,
        MULT,
        None,
        "-func: DFTTYP; -bs: $BASIS keywords for STO-3G, Pople, and correlation consistent "
        "sets; software_path.conf gamessus:: the USERSCR directory of rungms",
        {
            "xc": "functional (DFTTYP); None: no DFT",
            "contrl.KEY": "keywords of $CONTRL (contrl.scftyp=rohf, contrl.maxit=200); "
            "ICHARG, MULT, and SCFTYP are set from the charge and the multiplicity",
            "basis.KEY": "keywords of $BASIS (basis.gbasis=N31 basis.ngauss=6)",
            "GROUP.KEY": "keywords of any other input group (system.mwords=200, "
            "scf.dirscf=True, pcm.solvnt=WATER, statpt.nstep=50)",
            "ecp": "effective core potentials per element or atom, as ASE takes them",
            "userscr": "USERSCR directory of rungms",
            "command": "command line (rungms PREFIX.inp > PREFIX.log 2> PREFIX.err)",
            "label": "name of the input and output files (gamess_us)",
            "directory": _DIRECTORY,
        },
        ("G. M. J. Barca et al., J. Chem. Phys. 2020, 152, 154102 (GAMESS)",),
    ),
    "nwchem": Program(
        nwchem_calculator,
        _nwchem,
        MULT,
        None,
        "-func: xc; -bs and -sub_bs: basis; -mem: memory",
        {
            "theory": "theory of the task (dft from -func; scf, mp2, ccsd, tce, ...)",
            "dft.KEY": "keywords of the dft block (dft.grid=fine, dft.convergence=...); mult "
            "and odft are set from the multiplicity",
            "BLOCK.KEY": "keywords of any other block (scf.maxiter=200, set.KEY=VALUE, "
            "mp2.freeze=atomic)",
            "command": "command line (nwchem PREFIX.nwi > PREFIX.nwo)",
            "label": "name of the input and output files (nwchem)",
            "directory": _DIRECTORY,
            OTHER: "the other keywords of ASE's NWChem calculator (symmetry, center, "
            "autosym, autoz, task, pretasks, ...)",
        },
        ("E. Apra et al., J. Chem. Phys. 2020, 152, 184102 (NWChem)",),
    ),
    "mopac": Program(
        mopac_calculator,
        _mopac,
        MULT,
        None,
        "-func: the method (PM7); -core: THREADS=N; software_path.conf mopac:: the "
        "program or its directory",
        {
            "method": "the method (PM7, PM6-D3H4, AM1, ...)",
            "task": "the task keywords (1SCF GRADIENTS DISP)",
            "keywords": 'more keywords of the first line, written as they are ("EPS=78.4 '
            'PRECISE", "MOZYME")',
            "threads": "THREADS=N",
            "relscf": "RELSCF (ASE writes 0.0001)",
            "path": "the program or its directory (as software_path.conf)",
            "command": "command line (PROGRAM PREFIX.mop, with what MOPAC prints to the "
            "screen thrown away)",
            "label": "name of the input and output files (mopac)",
            "directory": _DIRECTORY,
        },
        (
            "J. J. P. Stewart, MOPAC, Stewart Computational Chemistry (cite the version that "
            "was run and the paper of the method, e.g. PM7: J. Mol. Model. 2013, 19, 1)",
        ),
    ),
    "gpaw": Program(
        gpaw_calculator,
        _gpaw,
        StateMapping("charge", None),
        None,
        "-func: xc",
        {
            "mode": "lcao (default), fd, pw, or with settings: mode.name=pw mode.ecut=400",
            "vacuum": "Angstrom around a structure without a cell (6)",
            "txt": "GPAW's text output (None: none; a file name)",
            "symmetry": "GPAW's use of symmetry (point_group off, so that structures "
            "may leave the symmetry of the first one; symmetry.point_group=True)",
            OTHER: "GPAW's keywords (h, kpts, basis, occupations.name=fermi-dirac "
            "occupations.width=0.05, spinpol, hund, convergence.energy=1e-6, ...)",
        },
        ("J. J. Mortensen et al., J. Chem. Phys. 2024, 160, 092503 (GPAW)",),
        moments=True,
        cell=True,
    ),
    "espresso": Program(
        espresso_calculator,
        _espresso,
        MULT,
        None,
        "-func: input_dft; software_path.conf espresso:: the pw.x command, "
        "espresso_pseudo:: the directory of the pseudopotentials",
        {
            "pseudopotentials.ELEMENT": "pseudopotential file of an element (required: "
            "pseudopotentials.Si=Si.pbe-n-rrkjus_psl.1.0.0.UPF)",
            "pseudo_dir": "directory of the pseudopotentials",
            "kpts": "k-points ((4,4,4); (1,1,1) by default)",
            "input_data.SECTION.KEY": "any pw.x input (input_data.system.ecutwfc=40, "
            "input_data.electrons.conv_thr=1e-8); tot_charge, nspin, and tot_magnetization "
            "are set from the charge and the multiplicity unless given",
            "command": 'the pw.x command line ("mpirun -n 4 pw.x")',
            "vacuum": "Angstrom around a structure without a cell (6)",
            "directory": _DIRECTORY,
            OTHER: "the other keywords of ASE's Espresso calculator (kspacing, koffset, "
            "tstress, ...)",
        },
        (
            "P. Giannozzi et al., J. Phys.: Condens. Matter 2009, 21, 395502; 2017, 29, "
            "465901 (Quantum ESPRESSO)",
        ),
        moments=True,
        cell=True,
        stress_setup=espresso_stress,
    ),
}
"""The programs by name."""

ALIASES = {"gamess": "gamessus", "gamess_us": "gamessus", "pwscf": "espresso", "qe": "espresso"}
"""Other names of the programs."""

_EVERYWHERE = {
    "KEY=@FILE": "the text of a file as the value",
    "hook": "module:function called on the ASE calculator when it is created",
    "excited_setup": "module:function(calculator, state, count) that sets the keywords of "
    "the program for the excited state `state` (0: the ground state); with it -es N is taken",
    "state_reader": "module:function(calculator, structure, count) that returns the excited "
    "states of the last calculation (mopt_rs.calculators.program_states.RawStates), or "
    "'file' for mopt_states.json in the directory of the program; with it -es_follow works",
}


def _program(name: str) -> tuple[str, Program]:
    key = ALIASES.get(name.strip().lower(), name.strip().lower())
    try:
        return key, PROGRAMS[key]
    except KeyError:
        raise KeyError(f"no program '{name}' (known: {', '.join(sorted(PROGRAMS))})") from None


def describe_program(name: str) -> str:
    """The settings of the program ``name``, as text: what the shared options set and
    every ``-calc_opt`` key with what it does."""
    key, program = _program(name)
    rows = {**program.options, **_EVERYWHERE}
    width = max(len(k) for k in rows)
    lines = [f"-os {key}", f"  {program.title}", "  -calc_opt:"]
    lines += [f"    {k.ljust(width)}  {text}" for k, text in rows.items()]
    lines.append(
        "  defaults can be kept in mopt_calculator_options.json "
        f'({{"calculators": {{"{key}": {{KEY: VALUE}}}}}})'
    )
    return "\n".join(lines)


def _nested(extra: Mapping[str, Any]) -> dict[str, Any]:
    """``extra`` with ``group.key=value`` entries sorted into ``group`` (a mapping);
    whatever follows the first dot is the key, so deeper paths stay as they are written."""
    out: dict[str, Any] = {}
    for key, value in extra.items():
        head, dot, rest = key.partition(".")
        if not dot:
            out[key] = dict(value) if isinstance(value, Mapping) else value
            continue
        if not head or not rest:
            raise ValueError(f"option '{key}': expected GROUP.KEY")
        group = out.setdefault(head, {})
        if not isinstance(group, dict):
            raise ValueError(f"option '{key}': '{head}' is set to a value, not to a group")
        group[rest] = value
    return out


def _check(key: str, program: Program, kwargs: Mapping[str, Any]) -> None:
    """Settings the program's function does not take are an error that lists what it takes
    (functions that pass keywords on to ASE take everything)."""
    parameters = inspect.signature(program.build).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return
    unknown = [k for k in kwargs if k not in parameters]
    if unknown:
        raise ValueError(
            f"{key}: unknown option{'s' if len(unknown) > 1 else ''} "
            f"{', '.join(unknown)}\n{describe_program(key)}"
        )


def program_engine(name: str, options: CalculatorOptions) -> FileProgramEngine:
    """The engine of the program ``name`` (see :data:`PROGRAMS`)."""
    key, program = _program(name)
    plain, _, setup = split_program_options(options.extra)
    states = {k: plain.pop(k, None) for k in ("excited_setup", "state_reader")}
    kwargs = program.keywords(options, _nested(plain))
    _check(key, program, kwargs)
    worker = worker_index()
    if worker is not None and "directory" in inspect.signature(program.build).parameters:
        # The copies of a calculator in worker processes each work in a directory of their
        # own: the files of a program have fixed names.
        kwargs["directory"] = os.path.join(
            str(kwargs.get("directory") or "."), f"mopt_worker_{worker}"
        )
    return FileProgramEngine(
        program.build,
        kwargs=kwargs,
        state=program.state,
        hessian=program.hessian,
        name=key,
        setup=setup,
        citations=program.citations,
        vacuum=float(kwargs.get("vacuum", 6.0)) if program.cell else None,
        moments=program.moments,
        stress_setup=program.stress_setup,
        electrons=True,
        **states,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m mopt_rs.calculators.programs [NAME ...]``: the settings of the programs
    (all of them without a name)."""
    names = list(sys.argv[1:] if argv is None else argv) or sorted(PROGRAMS)
    try:
        text = "\n\n".join(describe_program(name) for name in names)
    except KeyError as error:
        sys.stderr.write(f"{error.args[0]}\n")
        return 2
    sys.stdout.write(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
