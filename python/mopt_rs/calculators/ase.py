"""Any ASE calculator as an engine, configured by data instead of one wrapper class per program.

MultiOptPy writes a class for every ASE-based program that differ only in the import path, a
few constructor keywords, and how charge and spin are passed. Here these differences are a
:class:`StateMapping` and keyword arguments, so a program is one entry in
:mod:`mopt_rs.calculators.profiles` (or ``-os ase:module:Class`` on the command line).

The mapping is checked against the calculator's signature: a charge or spin keyword the
calculator does not declare is an error instead of being swallowed by ``**kwargs``, where a
misspelled or wrongly converted spin keyword would silently run an open shell as a singlet.

MultiOptPy: `Calculator/ase_calculation_tools.py` and `Calculator/ase_tools/`.

This file is part of mopt-py (the Python interface of mopt-rs).
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import contextlib
import dataclasses
import importlib
import inspect
import re
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from mopt_rs.calculators.base import (
    ASE_UNITS,
    BOHR_TO_ANGSTROM,
    HARTREE_TO_EV,
    HARTREE_TO_KCALMOL,
    HARTREE_TO_KJMOL,
    NOTES,
    CalculatorError,
    Capabilities,
    ElectronicState,
    Engine,
    EngineResult,
    SpinConvention,
    Structure,
)
from mopt_rs.calculators.program_options import ProgramSetup

__all__ = [
    "AUTOMATIC_HESSIAN_METHODS",
    "HESSIAN_LAYOUTS",
    "HESSIAN_UNITS",
    "AseEngine",
    "HessianRoute",
    "StateMapping",
    "detect_state",
    "hessian_routes",
    "hessian_unit_factor",
    "resolve_object",
    "square_hessian",
]


def resolve_object(path: str) -> Any:
    """``"package.module:Name"`` (or ``"package.module.Name"``) to the object."""
    module, sep, name = path.partition(":")
    if not sep:
        module, _, name = path.rpartition(".")
    if not module or not name:
        raise ValueError(f"'{path}' is not 'module:Name'")
    try:
        obj: Any = importlib.import_module(module)
    except ImportError as e:
        raise ImportError(
            f"cannot import '{module}' for '{path}': {e}. Install the package that provides it "
            f'(the calculators of ASE itself: pip install "mopt-py[ase]"; the list of '
            "the engines and what they need: python -m mopt_rs.calculators)"
        ) from e
    for part in name.split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError:
            raise ImportError(f"'{module}' has no attribute '{name}'") from None
    return obj


@dataclass(frozen=True)
class StateMapping:
    """How an ASE calculator receives the charge and the spin.

    ``via="kwargs"`` passes them as constructor keywords ``charge`` and ``spin`` (``None``:
    not passed); ``via="info"`` sets ``atoms.info[charge]`` and ``atoms.info[spin]``
    (FAIRChem, and engines that read the state from the structure).
    """

    charge: str | None = "charge"
    spin: str | None = "multiplicity"
    convention: SpinConvention = SpinConvention.MULTIPLICITY
    via: Literal["kwargs", "info"] = "kwargs"

    def values(self, state: ElectronicState) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.charge is not None:
            out[self.charge] = state.charge
        if self.spin is not None:
            out[self.spin] = state.spin_as(self.convention)
        return out


DEFAULT_STATE = StateMapping()
NO_STATE = StateMapping(charge=None, spin=None)
CHARGE_ONLY = StateMapping(spin=None)

# Keywords by which ASE calculators commonly take the state, in order of preference.
_CHARGE_KEYWORDS = ("charge", "chrg", "total_charge", "net_charge")
_SPIN_KEYWORDS = (
    ("multiplicity", SpinConvention.MULTIPLICITY),
    ("spin_multiplicity", SpinConvention.MULTIPLICITY),
    ("mult", SpinConvention.MULTIPLICITY),
    ("uhf", SpinConvention.UNPAIRED),
    ("unpaired_electrons", SpinConvention.UNPAIRED),
    ("nopen", SpinConvention.UNPAIRED),
    ("spin", SpinConvention.UNPAIRED),
)


def _signature_keywords(factory: Callable[..., Any]) -> tuple[set[str], bool]:
    target = factory.__init__ if inspect.isclass(factory) else factory
    try:
        sig = inspect.signature(target)
    except (TypeError, ValueError):
        return set(), True
    names = {p.name for p in sig.parameters.values() if p.kind is not p.VAR_KEYWORD}
    open_ended = any(p.kind is p.VAR_KEYWORD for p in sig.parameters.values())
    return names - {"self"}, open_ended


def _accepted_keywords(factory: Callable[..., Any]) -> tuple[set[str], bool]:
    """Explicit parameters plus ASE's ``default_parameters``; whether ``**kwargs`` is open."""
    names, open_ended = _signature_keywords(factory)
    defaults = getattr(factory, "default_parameters", None)
    if isinstance(defaults, Mapping):
        names |= set(defaults)
    return names, open_ended


def _documented_keywords(obj: Any) -> set[str]:
    doc = inspect.getdoc(obj) or ""
    init = getattr(obj, "__init__", None)
    if init is not None:
        doc += "\n" + (inspect.getdoc(init) or "")
    # Parameter-like mentions only: "name:", "name=", "name (", "name,", or ``name``.
    words = set(re.findall(r"\b([A-Za-z_]\w*)(?=\s*[:=(,])", doc))
    words |= set(re.findall(r"``([A-Za-z_]\w*)``", doc))
    return words & (set(_CHARGE_KEYWORDS) | {k for k, _ in _SPIN_KEYWORDS})


def _keyword_default(factory: Callable[..., Any], name: str) -> Any:
    defaults = getattr(factory, "default_parameters", None)
    if isinstance(defaults, Mapping) and name in defaults:
        return defaults[name]
    target = factory.__init__ if inspect.isclass(factory) else factory
    try:
        p = inspect.signature(target).parameters.get(name)
    except (TypeError, ValueError):
        return None
    return None if p is None or p.default is p.empty else p.default


def _restricted_reference(factory: Callable[..., Any]) -> bool:
    """Whether the calculator has a ``reference`` keyword that defaults to a restricted one."""
    names, _ = _accepted_keywords(factory)
    if "reference" not in names:
        return False
    default = _keyword_default(factory, "reference")
    return isinstance(default, str) and default.lower() in ("rhf", "rks", "restricted")


def detect_state(factory: Callable[..., Any]) -> StateMapping:
    """How ``factory`` takes charge and spin, from its signature, ASE ``default_parameters``,
    and, for calculators that take ``**kwargs``, the keywords named in its documentation.

    Without any such keyword the state goes to ``atoms.info`` (``charge``, ``multiplicity``),
    where many ASE calculators look for it.
    """
    names, open_ended = _accepted_keywords(factory)
    if open_ended:
        names |= _documented_keywords(factory)
    charge = next((k for k in _CHARGE_KEYWORDS if k in names), None)
    spin, convention = next(
        ((k, c) for k, c in _SPIN_KEYWORDS if k in names), (None, SpinConvention.MULTIPLICITY)
    )
    if charge is None and spin is None:
        return StateMapping("charge", "multiplicity", SpinConvention.MULTIPLICITY, via="info")
    return StateMapping(charge, spin, convention)


HESSIAN_UNITS: dict[str, float] = {
    "eV/Angstrom^2": 1.0,
    "eV/Bohr^2": 1.0 / BOHR_TO_ANGSTROM**2,
    "Hartree/Bohr^2": HARTREE_TO_EV / BOHR_TO_ANGSTROM**2,
    "Hartree/Angstrom^2": HARTREE_TO_EV,
    "kcal/mol/Angstrom^2": HARTREE_TO_EV / HARTREE_TO_KCALMOL,
    "kJ/mol/Angstrom^2": HARTREE_TO_EV / HARTREE_TO_KJMOL,
}
"""The units a calculator's Hessian array may be given in (``hessian_units``), with the factor
that turns each into eV/Angstrom^2 (the CODATA 2018 constants of the compiled module)."""


def hessian_unit_factor(units: str) -> float:
    """The factor from ``units`` (one of :data:`HESSIAN_UNITS`, case and spaces ignored) to
    eV/Angstrom^2; anything else is an error."""
    wanted = units.replace(" ", "").lower()
    for name, factor in HESSIAN_UNITS.items():
        if name.lower() == wanted:
            return factor
    raise ValueError(
        f"hessian_units='{units}' is not one of {', '.join(HESSIAN_UNITS)}"
    )


def hessian_ev_per_angstrom2(value: Any, array_units: str = "eV/Angstrom^2") -> Any:
    """A Hessian as returned by an ASE calculator, in eV/Angstrom^2 (in the layout it came in).

    Arrays are in ``array_units`` (ASE convention: eV/Angstrom^2); result objects are read
    from an attribute whose name states the unit (``hessian_ev_per_angstrom2``,
    ``hessian_hartree_per_bohr2``, ...) or from a plain ``hessian`` attribute (taken as
    ``array_units``).
    """
    if value is None:
        return None
    factor = hessian_unit_factor(array_units)
    if isinstance(value, np.ndarray) or hasattr(value, "__array__"):
        return np.asarray(value, dtype=np.float64) * factor
    if isinstance(value, list | tuple):
        return np.asarray(value, dtype=np.float64) * factor
    names = [n for n in dir(value) if "hessian" in n.lower() and not n.startswith("_")]
    for n in names:
        low = n.lower()
        if "ev" in low and ("ang" in low or "_a2" in low):
            return np.asarray(getattr(value, n), dtype=np.float64)
    for n in names:
        low = n.lower()
        if "hartree" in low and "bohr" in low:
            return np.asarray(getattr(value, n), dtype=np.float64) * HESSIAN_UNITS["Hartree/Bohr^2"]
    if "hessian" in names:
        return np.asarray(value.hessian, dtype=np.float64) * factor
    raise TypeError(f"cannot read a Hessian from {type(value).__name__}")


HESSIAN_LAYOUTS = ("(3N, 3N)", "(N, 3, N, 3)", "(3N, N, 3)", "(N, N, 3, 3)")
"""The layouts of a Hessian array that are understood (rows and columns in the order atom 1
x, y, z, atom 2 x, ...)."""


def asymmetry(h: Any) -> float:
    """``max |H - H^T| / max |H|`` (0 for a zero matrix)."""
    m = np.asarray(h, dtype=np.float64)
    scale = float(np.max(np.abs(m))) if m.size else 0.0
    if scale == 0.0:
        return 0.0
    return float(np.max(np.abs(m - m.T))) / scale


def square_hessian(value: Any, n_atoms: int) -> tuple[Any, str]:
    """The Hessian ``value`` as a (3N, 3N) matrix and the layout it was in (one of
    :data:`HESSIAN_LAYOUTS`); any other shape is an error.

    For three atoms an array of shape (3, 3, 3, 3) is both (N, 3, N, 3) and (N, N, 3, 3):
    it is read as (N, 3, N, 3) (the layout of ASE's ``VibrationsData.get_hessian``) unless
    that reading is not symmetric (:data:`SMALL_ASYMMETRY`) and the other one is more so.
    """
    h = np.asarray(value, dtype=np.float64)
    n, n3 = n_atoms, 3 * n_atoms
    if h.shape == (n3, n3):
        return h, HESSIAN_LAYOUTS[0]
    if h.shape == (n, 3, n, 3):
        atom_first = h.reshape(n3, n3)
        if h.shape == (n, n, 3, 3):
            pairs = h.transpose(0, 2, 1, 3).reshape(n3, n3)
            skew = asymmetry(atom_first)
            if skew > SMALL_ASYMMETRY and asymmetry(pairs) < skew:
                return pairs, HESSIAN_LAYOUTS[3]
        return atom_first, HESSIAN_LAYOUTS[1]
    if h.shape == (n3, n, 3):
        return h.reshape(n3, n3), HESSIAN_LAYOUTS[2]
    if h.shape == (n, n, 3, 3):
        return h.transpose(0, 2, 1, 3).reshape(n3, n3), HESSIAN_LAYOUTS[3]
    raise ValueError(
        f"a Hessian of shape {h.shape} for {n} atoms (expected one of "
        f"{', '.join(HESSIAN_LAYOUTS)})"
    )


FINITE_DIFFERENCES = "finite differences of the forces"

AUTOMATIC_HESSIAN_METHODS = ("get_hessian", "calculate_hessian", "hessian")
"""Methods tried, in this order, by ``hessian="auto"``. ``get_hessian`` is the name in ASE
itself (``ase.calculators.ff.ForceField.get_hessian``, and ``VibrationsData.get_hessian`` for
the result of a vibrational analysis) and follows ASE's ``get_forces``/``get_stress``;
``calculate_hessian`` and a callable ``hessian`` are the other names under which calculators
hand out a Hessian. Names that say the Hessian is numerical (``get_numerical_hessian``) are
never tried automatically."""

_NUMERICAL_MARKS = ("numerical", "numeric", "finite", "fd_", "_fd", "num_")

SMALL_ASYMMETRY = 1e-4
"""Asymmetries (:func:`asymmetry`) up to this are rounding: the matrix is symmetrized silently.
Measured on analytic Hessians of water with PySCF (def2-SVP): 8e-9 (RHF), 7e-10 to 4e-6 (PBE
and B3LYP on grids 0 to 3); see the documentation of the calculators."""
LARGE_ASYMMETRY = 0.1
"""Asymmetries above this mean a matrix that is not a Hessian (another layout, another
quantity; the results of ASE 3.26's ``ForceField`` measure 1.0): the route is not used. In
between it is symmetrized with a warning."""
CHECK_TOLERANCE = 0.05
"""Relative deviation of ``H v`` from differences of the forces above which ``hessian_check``
warns."""


def looks_numerical(name: str) -> bool:
    """Whether a method name says that the Hessian it returns is one of differences."""
    low = name.lower()
    return any(mark in low for mark in _NUMERICAL_MARKS)


@dataclass(frozen=True)
class HessianRoute:
    """One way to get a Hessian from an ASE calculator: a method (``kind="method"``), the
    ``hessian`` property through ``get_property`` (``"property"``), or the entry
    ``results["hessian"]`` after a calculation (``"results"``)."""

    kind: Literal["method", "property", "results"]
    name: str = "hessian"

    def __str__(self) -> str:
        if self.kind == "method":
            return f"{self.name}()"
        if self.kind == "property":
            return "hessian property"
        return "results['hessian']"


def hessian_routes(calculator: Any, hessian: str | None) -> tuple[list[HessianRoute], str]:
    """The routes ``hessian`` asks for on ``calculator`` (a class or an instance), in the
    order they are tried, and the mode: ``"numerical"`` (``None``, ``"numerical"``: no route,
    finite differences of the forces), ``"auto"`` (the methods of
    :data:`AUTOMATIC_HESSIAN_METHODS` the calculator has, the ``hessian`` property when it
    implements one, then ``results["hessian"]``), ``"analytic"`` (as ``auto``, but a
    calculator without any is an error instead of finite differences), or ``"named"`` (one
    method by name; ``"property"`` and ``"results"`` name the other two routes)."""
    if hessian is None:
        return [], "numerical"
    text = str(hessian).strip()
    low = text.lower()
    if low in ("numerical", "numeric", "none", "off", "false"):
        return [], "numerical"
    if low in ("auto", "analytic"):
        routes = [
            HessianRoute("method", m)
            for m in AUTOMATIC_HESSIAN_METHODS
            if callable(getattr(calculator, m, None))
        ]
        if "hessian" in getattr(calculator, "implemented_properties", ()):
            routes.append(HessianRoute("property"))
        routes.append(HessianRoute("results"))
        return routes, low
    if low == "property":
        return [HessianRoute("property")], "named"
    if low == "results":
        return [HessianRoute("results")], "named"
    if not text.isidentifier():
        raise ValueError(
            f"hessian='{hessian}': expected auto, analytic, numerical, property, results, or "
            "the name of a method of the calculator"
        )
    return [HessianRoute("method", text)], "named"


def _call_hessian_method(method: Callable[..., Any], atoms: Any) -> Any:
    """``method(atoms)``, or ``method()`` when it takes no positional argument (the
    calculator then works on the structure of its last calculation)."""
    try:
        params = inspect.signature(method).parameters.values()
    except (TypeError, ValueError):
        try:
            return method(atoms)
        except TypeError:
            return method()
    positional = [
        p for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.VAR_POSITIONAL)
    ]
    return method(atoms) if positional else method()


def _hessian_route(calculator: Any, hessian: str | None) -> str:
    """Where Hessians come from before the first one is asked for: the first route
    :func:`hessian_routes` tries other than ``results["hessian"]`` (whether that one exists
    shows only after a calculation), or finite differences."""
    routes, mode = hessian_routes(calculator, hessian)
    if mode == "named":
        return str(routes[0])
    static = [r for r in routes if r.kind != "results"]
    return str(static[0]) if static else FINITE_DIFFERENCES


def _is_calculator_instance(obj: Any) -> bool:
    return not inspect.isclass(obj) and hasattr(obj, "get_potential_energy")


class AseEngine(Engine):
    citations: tuple[str, ...] = (
        "A. H. Larsen et al., J. Phys.: Condens. Matter 2017, 29, 273002 (ASE); cite the "
        "program behind the ASE calculator as its authors ask",
    )
    """Any ASE calculator, with no code of its own.

    - ``calculator``: a class or function returning an ASE calculator, its import path
      ``"module:Name"``, or a calculator instance.
    - ``state``: how charge and spin are passed; ``None`` detects it (:func:`detect_state`).
      A given mapping is checked against the calculator's keywords.
    - ``hessian``: where Hessians come from (:func:`hessian_routes`). ``"auto"`` (the
      default) tries, at the first Hessian, the methods ``get_hessian``,
      ``calculate_hessian``, and ``hessian`` the calculator has (each called as
      ``method(atoms)``, or ``method()`` when it takes no argument), the ``hessian`` property
      when it implements one, and the entry ``results["hessian"]`` after the calculation,
      and keeps the first that gives a usable matrix; ``"analytic"`` the same, but without
      any it is an error; ``"numerical"`` or ``None``: finite differences of the forces; the
      name of a method (also one that is numerical, such as ``get_numerical_hessian``, which
      ``auto`` never calls), ``"property"``, or ``"results"``: that route alone. ASE itself
      defines no Hessian, so most calculators get finite differences; a route that fails
      later, or a named one that fails, falls back to them (with a warning), and
      :attr:`hessian_route`, :meth:`describe`, and the run log say which was used.
    - ``hessian_units``: the unit of a plain Hessian array, one of :data:`HESSIAN_UNITS`
      (default eV/Angstrom^2, ASE's); arrays may be in any of :data:`HESSIAN_LAYOUTS`.
    - ``hessian_check``: at the first Hessian, compare ``H v`` along three random directions
      with central differences of the forces (six force calculations) and report the
      relative deviation in the run log; a deviation that a unit of :data:`HESSIAN_UNITS`
      explains is named. Off by default.
    - ``reuse``: keep one calculator per electronic state, so engines that restart the SCF from
      the previous structure can do so (``False``: a new calculator for every structure).
    - ``declared_keywords``: keywords the calculator accepts through ``**kwargs`` (checked
      instead of the signature).
    - ``setup``: hooks run on every calculator the engine creates (once on a given instance),
      see :mod:`mopt_rs.calculators.program_options`.
    - ``excited_setup`` and ``state_reader``: the two functions through which a program
      gives excited states (``-es N``) and what is needed to follow them (``-es_follow``),
      see :mod:`mopt_rs.calculators.program_states`.
    """

    units = ASE_UNITS

    def __init__(
        self,
        calculator: Callable[..., Any] | str | Any,
        *,
        kwargs: Mapping[str, Any] | None = None,
        state: StateMapping | None = None,
        hessian: str | None = "auto",
        hessian_units: str = "eV/Angstrom^2",
        hessian_check: bool = False,
        name: str | None = None,
        reuse: bool = True,
        pbc: bool = True,
        electrons: bool | None = None,
        declared_keywords: tuple[str, ...] = (),
        setup: ProgramSetup | None = None,
        excited_setup: Callable[..., Any] | str | None = None,
        state_reader: Callable[..., Any] | str | None = None,
    ) -> None:
        from mopt_rs.calculators import program_states

        self.excited_setup = (
            None
            if excited_setup is None
            else program_states.resolve(excited_setup, "excited_setup")
        )
        self.state_reader = (
            None if state_reader is None else program_states.resolve(state_reader, "state_reader")
        )
        self._reader_missing_said = False
        obj = resolve_object(calculator) if isinstance(calculator, str) else calculator
        self._instance: Any = obj if _is_calculator_instance(obj) else None
        self._factory: Callable[..., Any] = type(obj) if self._instance is not None else obj
        self.kwargs = dict(kwargs or {})
        if self._instance is not None and self.kwargs:
            raise ValueError("keyword arguments need a calculator class, not an instance")
        self.state_mapping = detect_state(self._factory) if state is None else state
        self.setup = setup or ProgramSetup()
        if self._instance is not None and self.setup:
            self._instance = self.setup.run(self._instance)
        self.name = name or str(getattr(self._factory, "__name__", "ase"))
        self.reuse = reuse
        has_state = self.state_mapping.charge is not None or self.state_mapping.spin is not None
        self.capabilities = Capabilities(
            pbc=pbc,
            electrons=has_state if electrons is None else electrons,
            excited_states=self.excited_setup is not None,
        )
        self.use_hessian(hessian, units=hessian_units, check=hessian_check)
        self._calculators: dict[ElectronicState, Any] = {}
        # Open shells need an unrestricted reference when the default is restricted.
        self._open_shell_reference = "reference" not in self.kwargs and _restricted_reference(
            self._factory
        )
        if state is not None and state.via == "kwargs":
            self._check_keywords(declared_keywords)

    def describe(self) -> str:
        from mopt_rs.calculators.provenance import object_origin

        given = " (a given instance)" if self._instance is not None else ""
        hessian = f"Hessian: {self.hessian_route}"
        if self.hessian_route != FINITE_DIFFERENCES and self.hessian_units != "eV/Angstrom^2":
            hessian += f" in {self.hessian_units}"
        if self._hessian_failure:
            hessian += f" ({self._hessian_failure})"
        return "; ".join(
            (
                f"{self.name}: {object_origin(self._factory)}{given}",
                *self.notes,
                hessian,
                *self._hessian_notes,
            )
        )

    def use_hessian(
        self, hessian: str | None = "auto", *, units: str = "eV/Angstrom^2", check: bool = False
    ) -> None:
        """Choose where Hessians come from: the ``hessian``, ``hessian_units``, and
        ``hessian_check`` of the class (``-calc_opt hessian=...`` reaches an ASE engine
        through this). The capabilities follow: a Hessian of the calculator's own is asked of
        the engine (``analytic_hessian``, or ``numerical_hessian`` for a method whose name
        says it is numerical), otherwise :class:`EngineCalculator` takes differences.

        A Hessian of an ASE calculator is that of the positions only: for a periodic cell
        the second derivatives along the strain are not part of it, and a variable-cell run
        takes the columns of the cell from differences of the stress and the forces."""
        factor = hessian_unit_factor(units)
        self.hessian_units = next(u for u, f in HESSIAN_UNITS.items() if f == factor)
        """The unit of the calculator's Hessian arrays."""
        self._unit_factor = factor
        self.hessian_check = bool(check)
        obj = self._instance if self._instance is not None else self._factory
        self._routes, self._hessian_mode = hessian_routes(obj, hessian)
        self.hessian_method = hessian
        self.hessian_route = _hessian_route(obj, hessian)
        """How Hessians are obtained: the calculator's method, property, or results, or
        finite differences of the forces."""
        self._fixed_route: HessianRoute | None = None
        self._results_probed = False
        self._hessian_failure: str | None = None
        self._hessian_notes: list[str] = []
        self._pending_notes: list[str] = []
        self._asymmetry_said = False
        own = self.hessian_route != FINITE_DIFFERENCES or self._hessian_mode == "analytic"
        numerical = (
            self._hessian_mode == "named"
            and self._routes[0].kind == "method"
            and looks_numerical(self._routes[0].name)
        )
        self.capabilities = dataclasses.replace(
            self.capabilities,
            analytic_hessian=own and not numerical,
            numerical_hessian=own and numerical,
        )

    def _note(self, text: str) -> None:
        """A line for the run log (with the next evaluation) and for :meth:`describe`."""
        self._hessian_notes.append(text)
        self._pending_notes.append(f"{self.name}: {text}")

    def _read_hessian(self, route: HessianRoute, calc: Any, atoms: Any) -> tuple[Any, str]:
        """The Hessian of ``route`` (eV/Angstrom^2, (3N, 3N), symmetrized) and the layout
        it came in; an error when there is none or it is unusable."""
        if route.kind == "method":
            method = getattr(calc, route.name, None)
            if not callable(method):
                raise NotImplementedError(f"the calculator has no method {route.name}")
            value = _call_hessian_method(method, atoms)
        elif route.kind == "property":
            value = calc.get_property("hessian", atoms)
        else:
            results = getattr(calc, "results", None)
            value = results.get("hessian") if isinstance(results, Mapping) else None
            if value is None:
                raise LookupError("no entry after the calculation")
        if value is None:
            raise NotImplementedError("no Hessian returned")
        h, layout = square_hessian(hessian_ev_per_angstrom2(value, self.hessian_units), len(atoms))
        if not np.all(np.isfinite(h)):
            raise ValueError("non-finite entries")
        skew = asymmetry(h)
        if skew > LARGE_ASYMMETRY:
            raise ValueError(
                f"not symmetric (max |H - H^T| / max |H| = {skew:.2e}): another layout or not "
                "a Hessian"
            )
        if skew > SMALL_ASYMMETRY and not self._asymmetry_said:
            self._asymmetry_said = True
            text = (
                f"the Hessian of {route} is not symmetric (max |H - H^T| / max |H| = "
                f"{skew:.2e}); its symmetric part is used"
            )
            self._note(text)
            warnings.warn(f"{self.name}: {text}", RuntimeWarning, stacklevel=3)
        return 0.5 * (h + h.T), layout

    def _hessian(self, calc: Any, atoms: Any, structure: Structure) -> Any:
        """The calculator's Hessian (eV/Angstrom^2, (3N, 3N)), or ``None`` for finite
        differences of the forces: when the calculator has no Hessian, or the route found
        fails later or returns something unusable (then finite differences are used from
        here on, with a warning). The first Hessian tries the routes in order and keeps the
        first that works; with ``hessian="analytic"`` finding none is an error."""
        caps = self.capabilities
        if not (caps.analytic_hessian or caps.numerical_hessian):
            return None
        if self._fixed_route is not None:
            try:
                return self._read_hessian(self._fixed_route, calc, atoms)[0]
            except Exception as e:  # noqa: BLE001 - any failure falls back to differences
                return self._no_hessian(f"{self._fixed_route} failed: {e}")
        tried: list[str] = []
        for route in self._routes:
            try:
                h, layout = self._read_hessian(route, calc, atoms)
            except Exception as e:  # noqa: BLE001 - the next route is tried
                tried.append(f"{route} failed: {e}")
                continue
            self._keep_route(route, layout, tried)
            if self.hessian_check:
                self._check_hessian(calc, structure, h)
            return h
        return self._no_hessian("; ".join(tried) or "no Hessian")

    def _keep_route(self, route: HessianRoute, layout: str, tried: list[str]) -> None:
        self._fixed_route = route
        self.hessian_route = str(route)
        self.capabilities = dataclasses.replace(
            self.capabilities,
            analytic_hessian=not self.capabilities.numerical_hessian,
        )
        text = f"Hessian from {route}, layout {layout}, {self.hessian_units}"
        if tried:
            text += f" (before it: {'; '.join(tried)})"
        self._note(text)

    def _no_hessian(self, reason: str) -> None:
        if self._hessian_mode == "analytic":
            raise CalculatorError(
                f"{self.name}: hessian=analytic, but the calculator gives no usable Hessian "
                f"({reason})"
            )
        self._fall_back_to_differences(reason)
        return None

    def _check_hessian(self, calc: Any, structure: Structure, h: Any) -> None:
        """Compare ``H v`` with central differences of the forces along three random unit
        vectors ``v`` and report the largest relative deviation (``hessian_check``)."""
        n3 = h.shape[0]
        step_bohr = 1e-3  # the default step of EngineCalculator's differences
        step = step_bohr * BOHR_TO_ANGSTROM
        rng = np.random.default_rng(0)
        hv_all, fd_all, deviations = [], [], []
        for _ in range(3):
            v = rng.standard_normal(n3)
            v /= np.linalg.norm(v)
            forces = []
            for sign in (1.0, -1.0):
                moved = dataclasses.replace(
                    structure,
                    positions_bohr=structure.positions_bohr + sign * step_bohr * v.reshape(-1, 3),
                )
                atoms = self.atoms(moved)
                atoms.calc = calc
                forces.append(np.asarray(atoms.get_forces(apply_constraint=False)).ravel())
            fd = -(forces[0] - forces[1]) / (2.0 * step)
            hv = h @ v
            scale = max(float(np.linalg.norm(fd)), float(np.linalg.norm(hv)), 1e-300)
            deviations.append(float(np.linalg.norm(hv - fd)) / scale)
            hv_all.append(hv)
            fd_all.append(fd)
        deviation = max(deviations)
        hv_flat, fd_flat = np.concatenate(hv_all), np.concatenate(fd_all)
        hh = float(hv_flat @ hv_flat)
        ratio = float(hv_flat @ fd_flat) / hh if hh > 0.0 else float("nan")
        text = (
            f"hessian_check: H v against central differences of the forces along 3 random "
            f"directions (step {step_bohr} Bohr): largest relative deviation {deviation:.2e}"
        )
        if deviation > CHECK_TOLERANCE:
            matches = [
                u
                for u, f in HESSIAN_UNITS.items()
                if u != self.hessian_units and abs(ratio * self._unit_factor / f - 1.0) < 0.02
            ]
            text += f"; the differences are {ratio:.4g} times H v"
            if matches:
                text += f", as if the matrix were in {matches[0]} (hessian_units={matches[0]})"
            else:
                text += "; no unit explains it (another layout, or a wrong Hessian)"
            warnings.warn(f"{self.name}: {text}", RuntimeWarning, stacklevel=3)
        self._note(text)

    def take_notes(self) -> list[str]:
        """The lines for the run log gathered since the last call."""
        out, self._pending_notes = self._pending_notes, []
        return out

    def _check_keywords(self, declared: tuple[str, ...]) -> None:
        names, open_ended = _accepted_keywords(self._factory)
        names |= set(declared)
        wanted = [k for k in (self.state_mapping.charge, self.state_mapping.spin) if k]
        wanted += list(self.kwargs)
        unknown = [k for k in wanted if k not in names]
        if unknown and not (open_ended and not declared):
            raise TypeError(
                f"{self.name}: the calculator does not declare the keyword(s) "
                f"{', '.join(unknown)} (accepted: {', '.join(sorted(names)) or 'none'})"
            )

    def calculator(self, state: ElectronicState) -> Any:
        """The ASE calculator for ``state``."""
        values = self.state_mapping.values(state) if self.state_mapping.via == "kwargs" else {}
        if self._open_shell_reference and state.unpaired_electrons > 0:
            values["reference"] = "uhf"
        if self._instance is not None:
            return self._configured_instance(values)
        if self.reuse and state in self._calculators:
            return self._calculators[state]
        calc = self.setup.run(self._factory(**{**self.kwargs, **values}))
        if self.reuse:
            self._calculators[state] = calc
        return calc

    def _configured_instance(self, values: Mapping[str, Any]) -> Any:
        """The instance with the state of ``values``: through ASE ``set`` for parameters it
        keeps there, and directly for calculators that keep them as attributes; cached results
        are dropped when anything changed."""
        calc = self._instance
        params = getattr(calc, "parameters", None)
        changed = False
        for k, v in values.items():
            if isinstance(params, Mapping) and k in params:
                if params[k] != v:
                    calc.set(**{k: v})
                    changed = True
            elif hasattr(calc, k) and not callable(getattr(calc, k)):
                if getattr(calc, k) != v:
                    setattr(calc, k, v)
                    changed = True
            elif hasattr(calc, "set"):
                calc.set(**{k: v})
                changed = True
        if changed and callable(getattr(calc, "reset", None)):
            calc.reset()
        return calc

    def atoms(self, structure: Structure) -> Any:
        """The ASE ``Atoms`` of ``structure``."""
        from ase import Atoms

        cell = structure.cell_angstrom
        atoms = Atoms(
            symbols=list(structure.symbols),
            positions=structure.positions_angstrom,
            cell=cell if cell is not None else np.zeros((3, 3)),
            pbc=structure.pbc,
        )
        if self.state_mapping.via == "info":
            atoms.info.update(self.state_mapping.values(structure.state))
        elif self._instance is not None:
            # Instances may read the state from the structure (the ASE convention).
            atoms.info["charge"] = structure.state.charge
            atoms.info["multiplicity"] = structure.state.multiplicity
        return atoms

    def _fall_back_to_differences(self, reason: str) -> None:
        self._hessian_failure = reason
        self.hessian_route = FINITE_DIFFERENCES
        self._fixed_route = None
        self._routes = []
        self.capabilities = dataclasses.replace(
            self.capabilities, analytic_hessian=False, numerical_hessian=False
        )
        self._pending_notes.append(f"{self.name}: {reason}; Hessians from {FINITE_DIFFERENCES}")
        warnings.warn(
            f"{self.name}: {reason}; using finite differences of the forces",
            RuntimeWarning,
            stacklevel=2,
        )

    def _found_in_results(self, calc: Any, atoms: Any, structure: Structure) -> Any:
        """With ``hessian="auto"`` and no method or property: a ``results["hessian"]`` that
        the first calculation left, which is used from then on (looked at once)."""
        if self._results_probed or self._hessian_mode != "auto":
            return None
        self._results_probed = True
        caps = self.capabilities
        if caps.analytic_hessian or caps.numerical_hessian:
            return None
        results = getattr(calc, "results", None)
        if not isinstance(results, Mapping) or results.get("hessian") is None:
            return None
        route = HessianRoute("results")
        try:
            h, layout = self._read_hessian(route, calc, atoms)
        except Exception as e:  # noqa: BLE001 - an unusable entry leaves differences
            self._note(f"{route} is not used: {e}")
            return None
        self._keep_route(route, layout, [])
        if self.hessian_check:
            self._check_hessian(calc, structure, h)
        return h

    def compute(self, structure: Structure, hessian: bool, stress: bool = False) -> EngineResult:
        atoms = self.atoms(structure)
        atoms.calc = self.calculator(structure.state)
        if self.excited_setup is not None:
            # Before every calculation, so that a ground-state calculation after one of an
            # excited state is a ground-state calculation again.
            self.excited_setup(atoms.calc, structure.excited_state, structure.excited_state)
        extra: dict[str, Any] = {}
        if stress and structure.is_periodic:
            # Before the energy and the forces: a program run through files then computes
            # all three in one run instead of one run for the stress alone. A calculator
            # without a stress says so here, and the stress is taken from strains instead.
            with contextlib.suppress(Exception):
                extra["stress_voigt_ev_per_a3"] = np.asarray(
                    atoms.get_stress(voigt=True, apply_constraint=False)
                )
        energy = float(atoms.get_potential_energy(apply_constraint=False))
        forces = np.asarray(atoms.get_forces(apply_constraint=False), dtype=np.float64)
        if hessian:
            h = self._hessian(atoms.calc, atoms, structure)
        else:
            h = self._found_in_results(atoms.calc, atoms, structure)
        notes = self.take_notes()
        if notes:
            extra[NOTES] = notes
        return EngineResult(energy, forces=forces, hessian=h, extra=extra)

    def state_characters(self, structure: Structure, count: int) -> dict[str, Any] | None:
        """The excited states of ``structure`` through the two functions of
        :mod:`mopt_rs.calculators.program_states`: a calculation of ``count`` states (of the
        first one, when the program wants a state named), then the reader. ``None`` (said
        once) without a reader."""
        from mopt_rs.calculators import program_states

        if self.state_reader is None:
            if not self._reader_missing_said:
                self._reader_missing_said = True
                warnings.warn(
                    program_states.NOT_READ.format(name=self.name), RuntimeWarning, stacklevel=2
                )
            return None
        atoms = self.atoms(structure)
        atoms.calc = self.calculator(structure.state)
        if self.excited_setup is not None:
            self.excited_setup(atoms.calc, 1, count)
        atoms.get_potential_energy(apply_constraint=False)
        raw = self.state_reader(atoms.calc, structure, count)
        if raw is None:
            return None
        try:
            return program_states.state_characters(raw, count)
        except ValueError as error:
            raise ValueError(f"the states read for {self.name}: {error}") from None

    def close(self) -> None:
        self._calculators.clear()
