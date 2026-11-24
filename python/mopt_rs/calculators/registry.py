"""Calculator names to engines: one place for ``-os``, ``-xtb``, ``-dxtb``, and Python code.

Specifications (case-insensitive):

==========================  =========================================================
``pyscf`` (default)         :class:`~mopt_rs.calculators.pyscf.PySCFEngine`
``psi4``                    :class:`~mopt_rs.calculators.psi4.Psi4Engine`
``tblite[:GFN1-xTB]``       tblite (``-xtb GFN2-xTB`` maps here)
``dxtb[:GFN2-xTB]``         dxtb (``-dxtb``)
``openmm``                  OpenMM force fields (:class:`~mopt_rs.calculators.openmm.OpenMMEngine`)
``lj``, ``emt``, ...        native Rust potentials
``gaussian``, ``orca``, ... programs run through files (:mod:`.programs`)
``<profile>``               an ASE profile (``pm6_rs``, ``mace_mp``, ...)
``ase:module:Class``        any ASE calculator; options become constructor keywords
``engine:module:Class``     an :class:`~mopt_rs.calculators.base.Engine` of your own (a
                            class or a function that returns one); options become its
                            keywords
``multioptpy:<type>``       the installed MultiOptPy's setup for ``<type>``
``<plugin>``                an entry point in the ``mopt_rs.calculators`` group
==========================  =========================================================

:class:`CalculatorOptions` carries the program-independent settings (method, basis, threads,
...) that each factory maps to its program; ``extra`` passes program-specific keywords
(``-calc_opt key=value``), including ``prefix.path=value`` keys and ``hook=module:function``
(see :mod:`mopt_rs.calculators.program_options`).

MultiOptPy: the calculator choice of `optimization.py:OptimizationConfig` (by option flags); the
 registry itself is new.

This file is part of mopt-py (the Python interface of mopt-rs).
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import ast
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from importlib import metadata
from pathlib import Path
from typing import Any

from mopt_rs import _core
from mopt_rs.calculators.ase import AseEngine, StateMapping, resolve_object
from mopt_rs.calculators.base import Engine, EngineCalculator, SpinConvention
from mopt_rs.calculators.packages import require
from mopt_rs.calculators.program_options import split_program_options

__all__ = [
    "CalculatorOptions",
    "available",
    "create",
    "create_engine",
    "read_software_paths",
    "register",
]


@dataclass(frozen=True)
class CalculatorOptions:
    """Program-independent calculator settings (the QM options of the command line)."""

    method: str | None = None
    """Functional or method (``-func``); each program has its own default."""
    basis: str | None = None
    """Basis set (``-bs``)."""
    sub_basis: Mapping[str, str] = field(default_factory=dict)
    """Per-element basis sets (``-sub_bs I LanL2DZ``)."""
    ecp: Mapping[str, str] = field(default_factory=dict)
    unrestricted: bool = False
    grid_level: int | None = None
    threads: int | None = None
    memory: str | None = None
    """``2GB``, ``500MB``."""
    solvation: tuple[str, Any] | None = None
    """``("alpb", "water")`` or ``("cpcm", "water")`` for xTB."""
    software_paths: Mapping[str, str] = field(default_factory=dict)
    hessian_step_bohr: float = 1e-3
    """Step of the differences that make Hessians where the engine has none
    (``-calc_opt hessian_step=BOHR``)."""
    hessian_method: str = "central"
    """The older name of ``hessian_differences``."""
    hessian_differences: str | None = None
    """How Hessians are made where the engine has none: ``central`` differences or
    ``o1numhess`` (``-calc_opt hessian_differences=...``; ``None``: ``hessian_method``)."""
    stability: int = 0
    """Test the SCF solution every this many evaluations (``-stability``; 0: never)."""
    stability_external: str = "report"
    """Instabilities towards another kind of wavefunction: ``off``, ``report``, ``switch``
    (``-stability_external``)."""
    extra: Mapping[str, Any] = field(default_factory=dict)
    """Program-specific keywords."""
    model_atoms: tuple[int, ...] | None = None
    """For a calculator of a model system of ONIOM: the atoms of the whole structure (from
    0) that the model is made of, in its order; the link atoms follow them. ``None``: the
    calculator computes whole structures. The commands set it; a force field cuts the
    system of the model out of that of the whole structure with it."""
    model_links: tuple[tuple[int, int, float], ...] = ()
    """The cut bonds of the model system: (inner atom, outer atom, factor), with the
    factor of ``model_link_placement``."""
    model_link_placement: str = "scaled"
    """``scaled``: a link atom is at ``factor`` times the cut bond from the inner atom;
    ``distance``: at ``factor`` Bohr."""

    @property
    def memory_mb(self) -> float | None:
        if self.memory is None:
            return None
        text = self.memory.strip().upper()
        for unit, f in (("TB", 1024.0**2), ("GB", 1024.0), ("MB", 1.0)):
            if text.endswith(unit):
                return float(text[: -len(unit)]) * f
        return float(text)

    def basis_spec(self) -> str | dict[str, str] | None:
        """``basis``, or a per-element mapping when ``sub_basis`` is given."""
        if not self.sub_basis:
            return self.basis
        default = self.basis or "def2-svp"
        out = {"default": default}
        out.update(self.sub_basis)
        return out

    @staticmethod
    def from_mapping(d: Mapping[str, Any]) -> CalculatorOptions:
        """From a plain mapping (the Rust command line passes one)."""
        known = {f for f in CalculatorOptions.__dataclass_fields__}
        kw = {k: v for k, v in d.items() if k in known and v is not None}
        if "solvation" in kw and kw["solvation"] is not None:
            kw["solvation"] = tuple(kw["solvation"])
        if "model_atoms" in kw:
            kw["model_atoms"] = tuple(int(a) for a in kw["model_atoms"])
        if "model_links" in kw:
            kw["model_links"] = tuple((int(i), int(o), float(f)) for i, o, f in kw["model_links"])
        return CalculatorOptions(**kw)


Factory = Callable[[str, CalculatorOptions], Engine]
_FACTORIES: dict[str, Factory] = {}


def register(name: str, factory: Factory, *, replace_existing: bool = False) -> None:
    """Register ``factory(argument, options) -> Engine`` under ``name`` (``name:argument``)."""
    key = name.lower()
    if key in _FACTORIES and not replace_existing:
        raise KeyError(f"calculator '{name}' is registered")
    _FACTORIES[key] = factory


def _pyscf(_arg: str, o: CalculatorOptions) -> Engine:
    require("pyscf")
    from mopt_rs.calculators.pyscf import PySCFEngine

    kw: dict[str, Any] = {
        "method": o.method or "b3lyp",
        "basis": o.basis_spec() or "def2-svp",
        "ecp": dict(o.ecp) or None,
        "unrestricted": o.unrestricted,
        "threads": o.threads,
        "memory_mb": o.memory_mb,
    }
    if o.grid_level is not None:
        kw["grid_level"] = o.grid_level
    plain, prefixed, setup = split_program_options(o.extra, ("scf", "mol"))
    apply = plain.pop("apply", ())
    kw["apply"] = tuple(apply.split(",")) if isinstance(apply, str) else tuple(apply)
    kw["scf_options"] = {**plain.pop("scf_options", {}), **prefixed["scf"]}
    kw["mol_options"] = {**plain.pop("mol_options", {}), **prefixed["mol"]}
    kw.update(plain)
    return PySCFEngine(**kw, setup=setup)


def _psi4(_arg: str, o: CalculatorOptions) -> Engine:
    require("psi4")
    from mopt_rs.calculators.psi4 import Psi4Engine

    plain, _, setup = split_program_options(o.extra)
    return Psi4Engine(
        o.method or "b3lyp",
        o.basis or "def2-svp",
        unrestricted=o.unrestricted,
        grid_level=o.grid_level,
        threads=o.threads,
        memory=o.memory,
        options=plain,
        setup=setup,
    )


def _tblite(arg: str, o: CalculatorOptions) -> Engine:
    require("tblite")
    from mopt_rs.calculators.xtb import TbliteEngine

    plain, prefixed, setup = split_program_options(o.extra, ("set", "add"))
    return TbliteEngine(
        arg or o.method or "GFN2-xTB",
        solvation=o.solvation,
        settings={**plain, **prefixed["set"]},
        interactions=prefixed["add"],
        setup=setup,
    )


def _dxtb(arg: str, o: CalculatorOptions) -> Engine:
    require("dxtb")
    from mopt_rs.calculators.xtb import DxtbEngine

    plain, _, setup = split_program_options(o.extra)
    return DxtbEngine(arg or o.method or "GFN1-xTB", setup=setup, **plain)


def _openmm(_arg: str, o: CalculatorOptions) -> Engine:
    """``openmm``: ``system`` (serialized systems) or ``forcefield`` with ``topology`` (PDB
    files), one for every set of atoms the calculator computes; ``nonbonded``, ``cutoff``
    (nm), ``platform``; ``properties.NAME`` for the platform and ``create.NAME`` for
    ``ForceField.createSystem``."""
    require("openmm")
    from mopt_rs.calculators.openmm import OpenMMEngine

    plain, prefixed, setup = split_program_options(o.extra, ("properties", "create"))
    known = ("system", "forcefield", "topology", "nonbonded", "cutoff", "platform")
    unknown = sorted(set(plain) - set(known))
    if unknown:
        raise ValueError(
            f"openmm: unknown setting(s) {', '.join(unknown)} (known: {', '.join(known)}, "
            "properties.NAME, create.NAME, hook)"
        )
    return OpenMMEngine(
        system=plain.get("system"),
        forcefield=plain.get("forcefield"),
        topology=plain.get("topology"),
        nonbonded=plain.get("nonbonded"),
        cutoff=float(plain.get("cutoff", 1.0)),
        platform=plain.get("platform"),
        properties={k: str(v) for k, v in prefixed["properties"].items()},
        create=prefixed["create"],
        setup=setup,
        model_atoms=o.model_atoms,
        model_links=o.model_links,
        model_link_placement=o.model_link_placement,
    )


_STATE_KEYS = ("charge_keyword", "spin_keyword", "spin_convention", "state_via")


def _ase(arg: str, o: CalculatorOptions) -> Engine:
    """``ase:module:Class``: everything is detected; ``charge_keyword``, ``spin_keyword``,
    ``spin_convention``, and ``state_via`` override the detection (``hessian=...`` and the
    other keys of :data:`HESSIAN_KEYS` are taken by :func:`create_engine`), other options
    are constructor keywords."""
    extra = dict(o.extra)
    state = None
    if any(k in extra for k in _STATE_KEYS):
        state = StateMapping(
            charge=extra.pop("charge_keyword", "charge") or None,
            spin=extra.pop("spin_keyword", "multiplicity") or None,
            convention=SpinConvention(extra.pop("spin_convention", "multiplicity")),
            via=extra.pop("state_via", "kwargs"),
        )
    states = {k: extra.pop(k, None) for k in ("excited_setup", "state_reader")}
    plain, _, setup = split_program_options(extra)
    return AseEngine(arg, kwargs=plain, state=state, name=f"ase:{arg}", setup=setup, **states)


HESSIAN_KEYS = ("hessian", "hessian_method", "hessian_units", "hessian_check")
"""``-calc_opt`` keys of where an engine's Hessians come from, for every engine of the Python
side (taken out before the program's own keywords): ``hessian=auto|analytic|numerical|NAME``
(``hessian_method=NAME`` is its older spelling), and for ASE calculators ``hessian_units`` and
``hessian_check`` (:meth:`mopt_rs.calculators.ase.AseEngine.use_hessian`)."""
DIFFERENCE_KEYS = ("hessian_differences", "hessian_step")
"""``-calc_opt`` keys of the differences that make Hessians where an engine has none of its
own: ``hessian_differences=central|o1numhess`` and ``hessian_step=BOHR``
(:class:`~mopt_rs.calculators.base.EngineCalculator`)."""


def _flag(key: str, value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("1", "true", "on", "yes"):
        return True
    if text in ("0", "false", "off", "no"):
        return False
    raise ValueError(f"-calc_opt {key}={value}: expected true or false")


def _hessian_choice(extra: dict[str, Any]) -> dict[str, Any] | None:
    """The keys of :data:`HESSIAN_KEYS` taken out of ``extra``, as keywords of
    :meth:`Engine.use_hessian` (``None`` when none is given)."""
    found = {k: extra.pop(k) for k in HESSIAN_KEYS if k in extra}
    if not found:
        return None
    if "hessian" in found and "hessian_method" in found:
        if str(found["hessian"]) != str(found["hessian_method"]):
            raise ValueError(
                f"-calc_opt hessian={found['hessian']} and hessian_method="
                f"{found['hessian_method']} disagree (hessian_method is the older name of hessian)"
            )
    choice = found.get("hessian", found.get("hessian_method", "auto"))
    out: dict[str, Any] = {"hessian": None if choice in (None, "") else str(choice)}
    if "hessian_units" in found:
        out["units"] = str(found["hessian_units"])
    if "hessian_check" in found:
        out["check"] = _flag("hessian_check", found["hessian_check"])
    return out


def _differences(o: CalculatorOptions) -> tuple[CalculatorOptions, str, float]:
    """``o`` without the keys of :data:`DIFFERENCE_KEYS`, the scheme of the differences,
    and their step (Bohr): the keys win over the fields of ``o``."""
    extra = dict(o.extra)
    scheme = str(extra.pop("hessian_differences", o.hessian_differences or o.hessian_method))
    step = extra.pop("hessian_step", o.hessian_step_bohr)
    try:
        step = float(step)
    except (TypeError, ValueError):
        raise ValueError(f"-calc_opt hessian_step={step}: expected a number of Bohr") from None
    return replace(o, extra=extra), scheme, step


def _engine(arg: str, o: CalculatorOptions) -> Engine:
    """``engine:module:Class``: an engine written outside the package, by where it is
    imported from. ``-calc_opt key=value`` are the keywords of the class (or of the function
    that returns the engine); the settings shared by all calculators are passed as
    ``options`` when it has a parameter of that name."""
    import inspect

    made = resolve_object(arg)
    if isinstance(made, Engine):
        if o.extra:
            raise ValueError(f"engine:{arg} is an engine already made; it takes no options")
        return made
    if not callable(made):
        raise TypeError(
            f"engine:{arg} is {type(made).__name__}, not an Engine or a way to make one"
        )
    keywords = dict(o.extra)
    if "options" in inspect.signature(made).parameters:
        keywords["options"] = o
    engine = made(**keywords)
    if not isinstance(engine, Engine):
        raise TypeError(f"engine:{arg} made {type(engine).__name__}, not an Engine")
    return engine


def _multioptpy(arg: str, o: CalculatorOptions) -> Engine:
    from mopt_rs.calculators.bridges import MultiOptPyEngine

    return MultiOptPyEngine(
        arg,
        software_paths=o.software_paths,
        functional=o.method,
        basis_set=o.basis,
        memory=o.memory,
    )


for _name, _factory in {
    "pyscf": _pyscf,
    "psi4": _psi4,
    "tblite": _tblite,
    "xtb": _tblite,
    "dxtb": _dxtb,
    "openmm": _openmm,
    "ase": _ase,
    "engine": _engine,
    "multioptpy": _multioptpy,
}.items():
    register(_name, _factory)


def read_software_paths(path: str | Path) -> dict[str, str]:
    """MultiOptPy's ``software_path.conf`` (``name::path`` per line)."""
    out: dict[str, str] = {}
    p = Path(path)
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "::" not in line:
            continue
        name, _, value = line.partition("::")
        out[name.strip().lower()] = value.strip()
    return out


def _plugins() -> dict[str, Any]:
    try:
        eps = metadata.entry_points(group="mopt_rs.calculators")
    except TypeError:  # pragma: no cover - Python < 3.10
        eps = metadata.entry_points().get("mopt_rs.calculators", [])  # type: ignore[attr-defined]
    return {ep.name.lower(): ep for ep in eps}


def available() -> list[str]:
    """Every name ``create`` accepts (without arguments)."""
    from mopt_rs.calculators.profile_files import load_profile_files
    from mopt_rs.calculators.profiles import PROFILES
    from mopt_rs.calculators.programs import PROGRAMS

    load_profile_files()
    names = set(_FACTORIES) | set(PROFILES) | set(_core.native_calculators()) | set(_plugins())
    names |= set(PROGRAMS)
    return sorted(names)


def _xtb_alias(spec: str) -> str | None:
    lowered = spec.lower()
    if lowered in ("gfn1-xtb", "gfn2-xtb", "ipea1-xtb"):
        return f"tblite:{spec}"
    return None


def create_engine(spec: Any = "pyscf", options: CalculatorOptions | None = None) -> Engine:
    """The engine for ``spec``: a name (see the module documentation), an :class:`Engine`, or
    any ASE calculator class, factory function, or instance (used as it is, see
    :class:`~mopt_rs.calculators.ase.AseEngine`)."""
    from mopt_rs.calculators.option_files import default_options
    from mopt_rs.calculators.profile_files import load_profile_files
    from mopt_rs.calculators.programs import ALIASES

    load_profile_files()
    o = options or CalculatorOptions()
    if isinstance(spec, Engine):
        return spec
    notes: list[str] = []
    if isinstance(spec, str):
        spec = spec.strip()
        if spec.lower() in ("", "none"):
            spec = "pyscf"
        spec = _xtb_alias(spec) or spec
        key = spec.partition(":")[0].lower()
        # Settings kept in files for this calculator; what was given here wins.
        names = [key, ALIASES.get(key, key), spec.lower()]
        defaults, notes = default_options(names, o.extra)
        if defaults:
            o = replace(o, extra={**defaults, **o.extra})
    # The keys of the Hessian are no program keywords: those of the differences belong to
    # the calculator (create), the others are applied to the engine once it is made.
    extra = {k: v for k, v in o.extra.items() if k not in DIFFERENCE_KEYS}
    choice = _hessian_choice(extra)
    o = replace(o, extra=extra)
    if isinstance(spec, str):
        head, _, arg = spec.partition(":")
        engine = _named_engine(spec, head.lower(), arg, o)
    else:
        plain, _, setup = split_program_options(o.extra)
        engine = AseEngine(spec, kwargs=plain, setup=setup)
    if notes:
        engine.notes = (*engine.notes, *notes)
    if choice is not None:
        engine.use_hessian(**choice)
    return engine


def _named_engine(spec: str, key: str, arg: str, o: CalculatorOptions) -> Engine:
    """The engine of the calculator ``key`` (``spec`` is ``key`` or ``key:arg``)."""
    from mopt_rs.calculators.bridges import NativeEngine
    from mopt_rs.calculators.profiles import PROFILES, profile_engine
    from mopt_rs.calculators.programs import ALIASES, PROGRAMS, program_engine

    if key in _FACTORIES:
        return _FACTORIES[key](arg, o)
    if key in _core.native_calculators() or key == "mb":
        return NativeEngine("muller_brown" if key == "mb" else key)
    if key in PROFILES:
        return profile_engine(key, options=o.extra, software_paths=o.software_paths)
    if key in PROGRAMS or key in ALIASES:
        return program_engine(key, o)
    plugins = _plugins()
    if key in plugins:
        factory = plugins[key].load()
        engine = factory(arg, o)
        if not isinstance(engine, Engine):
            raise TypeError(f"plugin '{key}' returned {type(engine).__name__}, not an Engine")
        return engine
    raise KeyError(f"unknown calculator '{spec}' (known: {', '.join(available())})")


def create(
    spec: Any = "pyscf", options: CalculatorOptions | None = None, **extra: Any
) -> EngineCalculator:
    """A calculator for :func:`mopt_rs.optimize.run` (``extra`` adds program keywords)."""
    o = options or CalculatorOptions()
    if extra:
        o = replace(o, extra={**o.extra, **extra})
    o, scheme, step = _differences(o)
    return EngineCalculator(
        create_engine(spec, o),
        hessian_step_bohr=step,
        hessian_differences=scheme,
        stability=o.stability,
        stability_external=o.stability_external,
    )


def parse_option_value(text: str) -> Any:
    """``-calc_opt key=value`` values: the text of a file for ``@FILE`` (``@@x`` is the
    string ``@x``), Python literals when they parse, else strings."""
    from mopt_rs.calculators.option_files import FILE_MARK, file_value

    if text.startswith(FILE_MARK):
        return file_value(text)
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text
