"""Any setting of a program, without a dedicated mopt-rs option.

Program-specific settings (``-calc_opt key=value`` on the command line, ``extra`` in
:class:`~mopt_rs.calculators.CalculatorOptions`, the ``calc_option`` list of settings files)
reach the program in three ways, the same for every engine:

- plain keys go where the engine documents (constructor keywords of an ASE calculator,
  ``psi4.set_options``, tblite ``Calculator.set``, PySCF engine keywords);
- ``prefix.path=value`` keys address the program's own objects, for the prefixes an engine
  declares (PySCF: ``scf.with_solvent.eps=78.4``, ``mol.cart=True``; tblite:
  ``add.electric-field=[0,0,0.001]``);
- ``hook=module:function`` (several separated by commas) calls user code as
  ``function(obj, **kwargs)`` whenever the engine creates its program object, with keyword
  arguments from ``hook.<name>=value`` keys; a returned object replaces ``obj``. The object is
  the PySCF SCF object (every calculation; the molecule is ``obj.mol``), the tblite
  ``Calculator`` (every calculation), the dxtb calculator (once per composition), the ASE
  calculator (once per electronic state, or once for a given instance), or the Psi4 molecule
  (every calculation).

So solvent models, external fields, or any other feature a program has can be used as soon as
the program has it.

MultiOptPy: no counterpart (new in mopt-rs).

This file is part of mopt-py (the Python interface of mopt-rs).
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = ["ProgramSetup", "set_path", "split_program_options"]

HOOK = "hook"


@dataclass
class ProgramSetup:
    """User hooks run on a program object before every calculation."""

    hooks: tuple[Callable[..., Any], ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return bool(self.hooks)

    def run(self, obj: Any) -> Any:
        """``obj`` after every hook (a hook returning ``None`` changed it in place)."""
        for hook in self.hooks:
            out = hook(obj, **self.kwargs)
            if out is not None:
                obj = out
        return obj


def _hooks(value: Any) -> tuple[Callable[..., Any], ...]:
    from mopt_rs.calculators.ase import resolve_object

    items: Sequence[Any] = value if isinstance(value, (list, tuple)) else str(value).split(",")
    out = []
    for item in items:
        f = resolve_object(item.strip()) if isinstance(item, str) else item
        if not callable(f):
            raise TypeError(f"hook '{item}' is not callable")
        out.append(f)
    return tuple(out)


def split_program_options(
    extra: Mapping[str, Any], prefixes: Sequence[str] = ()
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], ProgramSetup]:
    """``extra`` as (plain keys, ``{prefix: {path: value}}`` for ``prefixes``, hooks)."""
    plain: dict[str, Any] = {}
    prefixed: dict[str, dict[str, Any]] = {p: {} for p in prefixes}
    hooks: tuple[Callable[..., Any], ...] = ()
    kwargs: dict[str, Any] = {}
    for key, value in extra.items():
        head, dot, rest = key.partition(".")
        if key == HOOK:
            hooks += _hooks(value)
        elif dot and head == HOOK:
            kwargs[rest] = value
        elif dot and head in prefixed:
            if not rest:
                raise ValueError(f"'{key}' needs a name after '{head}.'")
            prefixed[head][rest] = value
        else:
            plain[key] = value
    if kwargs and not hooks:
        raise ValueError(f"hook arguments {', '.join(sorted(kwargs))} given without hook=")
    return plain, prefixed, ProgramSetup(hooks, kwargs)


def set_path(obj: Any, path: str, value: Any) -> None:
    """``obj.a.b = value`` for ``path = "a.b"``; keys of mappings on the way work too."""
    *parents, last = path.split(".")
    target = obj
    for name in parents:
        target = target[name] if isinstance(target, Mapping) else getattr(target, name)
    if isinstance(target, dict):
        target[last] = value
    else:
        setattr(target, last, value)
