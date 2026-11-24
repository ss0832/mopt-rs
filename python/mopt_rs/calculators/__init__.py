"""Calculators: engines for quantum chemistry, tight binding, machine-learned and classical
potentials, behind one contract.

>>> from mopt_rs.calculators import create, CalculatorOptions
>>> calc = create("pyscf", CalculatorOptions(method="b3lyp", basis="def2-svp"))  # doctest: +SKIP
>>> calc = create("pm6_rs")                # an ASE profile                          # doctest: +SKIP
>>> calc = create("ase:pm6_rs.ase:PM6", method="pm6-d3h4x")                           # doctest: +SKIP

New programs: subclass :class:`Engine` (or describe an ASE calculator with
:class:`AseProfile`) and :func:`register` it, or publish an entry point in the
``mopt_rs.calculators`` group.

MultiOptPy: `Calculator/__init__.py`.

This file is part of mopt-py (the Python interface of mopt-rs).
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

from mopt_rs.calculators.ase import AseEngine, StateMapping, resolve_object
from mopt_rs.calculators.base import (
    ASE_UNITS,
    ATOMIC_UNITS,
    CalculatorError,
    Capabilities,
    ElectronicState,
    Engine,
    EngineCalculator,
    EngineResult,
    EngineUnits,
    SpinConvention,
    Structure,
)
from mopt_rs.calculators.profiles import PROFILES, AseProfile, profile_engine, register_profile
from mopt_rs.calculators.program_options import ProgramSetup, split_program_options
from mopt_rs.calculators.programs import (
    PROGRAMS,
    FileProgramEngine,
    describe_program,
    program_engine,
)
from mopt_rs.calculators.registry import (
    CalculatorOptions,
    available,
    create,
    create_engine,
    read_software_paths,
    register,
)
from mopt_rs.calculators.workers import WorkerCalculator

__all__ = [
    "ASE_UNITS",
    "ATOMIC_UNITS",
    "PROFILES",
    "PROGRAMS",
    "ProgramSetup",
    "AseEngine",
    "AseProfile",
    "CalculatorError",
    "CalculatorOptions",
    "Capabilities",
    "ElectronicState",
    "Engine",
    "EngineCalculator",
    "EngineResult",
    "EngineUnits",
    "FileProgramEngine",
    "SpinConvention",
    "StateMapping",
    "Structure",
    "WorkerCalculator",
    "available",
    "create",
    "create_engine",
    "describe_program",
    "profile_engine",
    "program_engine",
    "read_software_paths",
    "register",
    "register_profile",
    "resolve_object",
    "split_program_options",
]
