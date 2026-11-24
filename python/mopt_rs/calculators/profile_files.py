"""ASE-calculator profiles from files: a new program is a few lines of JSON, not code.

Machine-learned potentials and other programs with an ASE calculator are used by name
(``-os NAME``) once a profile says how the calculator is built. Profiles are read from

1. ``~/.config/mopt-rs/profiles.json`` (``$XDG_CONFIG_HOME`` is honoured),
2. ``mopt_profiles.json`` in the working directory,
3. the files named in ``MOPT_PROFILES`` (separated like ``PATH``),

later files replacing earlier names, all of them replacing built-in profiles of the same
name. A file is ``{"profiles": {NAME: PROFILE, ...}}`` with

``factory``
    ``"package.module:Name"``: the class or function that returns the ASE calculator.
``args``, ``kwargs``
    Its positional and keyword arguments. A value is plain JSON, or

    - ``{"option": "device", "default": "cpu"}``: the value of ``-calc_opt device=...``
      (``default`` when it is not given; without ``default`` the option is required);
    - ``{"software_path": true}``: the path given for the profile in ``software_path.conf``
      (a model file, for example);
    - ``{"factory": ..., "args": [...], "kwargs": {...}, "call": "method"}``: an object built
      first, for programs that load a model and hand it to the calculator. ``call`` names a
      method called on the result without arguments (``model.ase()``).
``call``
    As above, on the calculator factory's result.
``state``
    How charge and spin are passed: ``"none"`` (default), ``"charge"``, ``"multiplicity"``
    (keywords ``charge`` and ``multiplicity``), ``"unpaired"`` (``charge`` and ``uhf``), or
    ``{"charge": KEYWORD, "spin": KEYWORD, "convention": "multiplicity" | "unpaired",
    "via": "kwargs" | "info"}``.
``hessian``
    The calculator's Hessian method, or ``null`` (default): finite differences of the forces.
``electrons``
    Whether charge and multiplicity are checked against the electron count (default: true
    unless ``state`` is ``"none"``).
``description``
    A line for the run log.

Options that a profile does not consume go to the calculator factory as keywords, so
``-calc_opt`` reaches every keyword of the calculator without being listed here. One
calculator (and so one loaded model) serves a whole run.

MultiOptPy: `Calculator/ase_tools/*.py` (one wrapper class per program there).

This file is part of mopt-py (the Python interface of mopt-rs).
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from mopt_rs.calculators.ase import CHARGE_ONLY, NO_STATE, StateMapping, resolve_object
from mopt_rs.calculators.base import SpinConvention
from mopt_rs.calculators.profiles import MULT, PROFILES, UHF, AseProfile

__all__ = ["load_profile_file", "load_profile_files", "profile_from_data", "profile_paths"]

PATH_OPTION = "__software_path__"
_STATES = {"none": NO_STATE, "charge": CHARGE_ONLY, "multiplicity": MULT, "unpaired": UHF}
_KEYS = {"factory", "args", "kwargs", "call", "state", "hessian", "electrons", "description"}


def _state(value: Any, name: str) -> StateMapping:
    if isinstance(value, str):
        try:
            return _STATES[value.lower()]
        except KeyError:
            raise ValueError(
                f"profile '{name}': state '{value}' is not one of {', '.join(_STATES)}"
            ) from None
    if isinstance(value, Mapping):
        unknown = set(value) - {"charge", "spin", "convention", "via"}
        if unknown:
            raise ValueError(f"profile '{name}': unknown state keys {sorted(unknown)}")
        convention = str(value.get("convention", "multiplicity")).lower()
        conventions = {
            "multiplicity": SpinConvention.MULTIPLICITY,
            "unpaired": SpinConvention.UNPAIRED,
        }
        if convention not in conventions:
            raise ValueError(f"profile '{name}': convention must be multiplicity or unpaired")
        via = str(value.get("via", "kwargs"))
        if via not in ("kwargs", "info"):
            raise ValueError(f"profile '{name}': via must be kwargs or info")
        return StateMapping(
            value.get("charge"),
            value.get("spin"),
            conventions[convention],
            via="info" if via == "info" else "kwargs",
        )
    raise ValueError(f"profile '{name}': state must be a name or an object")


def _options_used(value: Any) -> set[str]:
    if isinstance(value, Mapping):
        if "option" in value:
            return {str(value["option"])}
        return set().union(*(_options_used(v) for v in value.values())) if value else set()
    if isinstance(value, list):
        return set().union(*(_options_used(v) for v in value)) if value else set()
    return set()


def _evaluate(value: Any, options: Mapping[str, Any], name: str) -> Any:
    """A value of a profile with options, the software path, and nested objects filled in."""
    if isinstance(value, list):
        return [_evaluate(v, options, name) for v in value]
    if not isinstance(value, Mapping):
        return value
    if "option" in value:
        key = str(value["option"])
        if key in options:
            return options[key]
        if "default" in value:
            return value["default"]
        raise ValueError(f"profile '{name}' needs -calc_opt {key}=...")
    if value.get("software_path") is True:
        path = options.get(PATH_OPTION)
        if not path:
            raise ValueError(
                f"profile '{name}' needs a path: add '{name}::/path' to software_path.conf"
            )
        return path
    if "factory" in value:
        return _construct(value, options, name)
    return {k: _evaluate(v, options, name) for k, v in value.items()}


def _construct(spec: Mapping[str, Any], options: Mapping[str, Any], name: str) -> Any:
    factory = resolve_object(str(spec["factory"]))
    args = [_evaluate(a, options, name) for a in spec.get("args", [])]
    kwargs = {k: _evaluate(v, options, name) for k, v in dict(spec.get("kwargs", {})).items()}
    obj = factory(*args, **kwargs)
    call = spec.get("call")
    return getattr(obj, str(call))() if call else obj


def profile_from_data(name: str, data: Mapping[str, Any]) -> AseProfile:
    """The profile ``name`` described by ``data`` (one entry of a profile file)."""
    unknown = set(data) - _KEYS
    if unknown:
        raise ValueError(
            f"profile '{name}': unknown keys {sorted(unknown)} (known: {sorted(_KEYS)})"
        )
    if "factory" not in data:
        raise ValueError(f"profile '{name}' has no 'factory'")
    consumed = _options_used(data.get("args", [])) | _options_used(data.get("kwargs", {}))
    state = _state(data.get("state", "none"), name)

    def build(**given: Any) -> Any:
        # Options named by the profile are placed where it says; the rest are keywords of the
        # calculator factory.
        rest = {k: v for k, v in given.items() if k not in consumed and k != PATH_OPTION}
        spec = dict(data)
        spec["kwargs"] = {**dict(data.get("kwargs", {})), **rest}
        return _construct(spec, given, name)

    build.__name__ = name
    return AseProfile(
        build,
        state,
        hessian=data.get("hessian"),
        description=str(data.get("description", "")),
        path_keyword=PATH_OPTION,
        electrons=bool(data.get("electrons", state is not NO_STATE)),
    )


def load_profile_file(path: str | os.PathLike[str]) -> dict[str, AseProfile]:
    """The profiles of one file."""
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise ValueError(f"profile file {path}: {e}") from e
    entries = doc.get("profiles") if isinstance(doc, Mapping) else None
    if not isinstance(entries, Mapping):
        raise ValueError(f'profile file {path}: expected {{"profiles": {{...}}}}')
    return {str(k).lower(): profile_from_data(str(k), v) for k, v in entries.items()}


def profile_paths() -> list[Path]:
    """The profile files that exist, in the order they are read."""
    config = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    candidates = [Path(config) / "mopt-rs" / "profiles.json", Path("mopt_profiles.json")]
    candidates += [Path(p) for p in os.environ.get("MOPT_PROFILES", "").split(os.pathsep) if p]
    return [p for p in candidates if p.is_file()]


_loaded: dict[Path, float] = {}


def load_profile_files(
    register: Callable[[str, AseProfile], None] | None = None,
) -> list[str]:
    """Read the profile files into the table of profiles; returns the names that were set.

    Called whenever a calculator is looked up by name; a file is read again only when it has
    changed.
    """
    names = []
    for path in profile_paths():
        stamp = path.stat().st_mtime
        if _loaded.get(path.resolve()) == stamp:
            continue
        for name, profile in load_profile_file(path).items():
            if register is None:
                PROFILES[name] = profile
            else:
                register(name, profile)
            names.append(name)
        _loaded[path.resolve()] = stamp
    return names
