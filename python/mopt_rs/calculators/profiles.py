"""Named ASE-calculator profiles: one data entry per program instead of one class per program.

Each entry states the import path, fixed keywords, how charge and spin are passed (with the
program's own spin convention), and the Hessian method; keywords are checked against the
calculator's signature (see :mod:`mopt_rs.calculators.ase`). Profiles of the Rust ASE engines
are named after their packages (``pm6_rs``, ``gfn2_rs``, ...). ``dftb_rs`` receives
``spin = 2S`` (the keyword is not documented by the engine).

Register more with :func:`register_profile`, or use ``ase:module:Class`` directly.

MultiOptPy: `Calculator/ase_tools/*.py` (one wrapper class per program there; data here).

This file is part of mopt-py (the Python interface of mopt-rs).
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from mopt_rs.calculators.ase import CHARGE_ONLY, NO_STATE, AseEngine, StateMapping
from mopt_rs.calculators.base import SpinConvention

__all__ = ["AseProfile", "PROFILES", "profile_engine", "register_profile"]

MULT = StateMapping("charge", "multiplicity", SpinConvention.MULTIPLICITY)
UHF = StateMapping("charge", "uhf", SpinConvention.UNPAIRED)
DFTB_SPIN = StateMapping("charge", "spin", SpinConvention.UNPAIRED)
INFO_SPIN_MULT = StateMapping("charge", "spin", SpinConvention.MULTIPLICITY, via="info")
AIMNET_STATE = StateMapping("charge", "mult", SpinConvention.MULTIPLICITY)


@dataclass(frozen=True)
class AseProfile:
    """A named ASE calculator setup."""

    factory: str | Callable[..., Any]
    """Import path ``module:Name`` or a callable returning the calculator."""
    state: StateMapping = MULT
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    hessian: str | None = "get_hessian"
    description: str = ""
    path_keyword: str | None = None
    """Keyword receiving the program path from ``software_path.conf`` (``name::path``)."""
    electrons: bool = True
    declared_keywords: tuple[str, ...] = ()
    """Keywords the factory accepts through ``**kwargs`` (from its documentation)."""


def _mace(model: str) -> Callable[..., Any]:
    def build(**kwargs: Any) -> Any:
        from mace import calculators

        return getattr(calculators, model)(**kwargs)

    build.__name__ = model
    return build


def _fairchem(task_name: str = "omol", device: str = "cpu") -> Callable[..., Any]:
    def build(path: str | None = None, **kwargs: Any) -> Any:
        from fairchem.core import FAIRChemCalculator
        from fairchem.core.units.mlip_unit import load_predict_unit

        if not path:
            raise ValueError("give the model checkpoint path in software_path.conf")
        unit = load_predict_unit(path=path, device=kwargs.pop("device", device))
        return FAIRChemCalculator(
            predict_unit=unit, task_name=kwargs.pop("task_name", task_name), **kwargs
        )

    build.__name__ = "fairchem"
    return build


_AM1_LIKE = ("charge", "multiplicity", "reference", "method")

PROFILES: dict[str, AseProfile] = {
    "gfn0_rs": AseProfile(
        "gfn0_rs:GFN0RSCalculator",
        MULT,
        description="GFN0-xTB (gfn0-rs)",
        path_keyword="param_path",
    ),
    "gfn0_rs_nn": AseProfile(
        "gfn0_rs:GFN0RSCalculator", MULT, {"nn_model": "builtin"}, description="NN-GFN0-xTB"
    ),
    "gfn0_rs_nn_param": AseProfile(
        "gfn0_rs:GFN0RSCalculator",
        MULT,
        {"nn_model": "builtin"},
        description="NN-GFN0-xTB with a parameter file",
        path_keyword="param_path",
    ),
    "gfn1_rs": AseProfile(
        "gfn1_rs:GFN1RSCalculator",
        MULT,
        description="GFN1-xTB (gfn1-rs)",
        path_keyword="param_path",
        declared_keywords=("charge", "multiplicity", "param_path", "camm_preset"),
    ),
    "gfn1_rs_sigma_hole": AseProfile(
        "gfn1_rs:GFN1RSCalculator",
        MULT,
        {"camm_preset": "sigma-hole"},
        description="GFN1-xTB with sigma-hole multipoles",
        path_keyword="param_path",
        declared_keywords=("charge", "multiplicity", "param_path", "camm_preset"),
    ),
    "gfn2_rs": AseProfile(
        "gfn2_rs:GFN2RSCalculator",
        MULT,
        description="GFN2-xTB (gfn2-rs)",
        path_keyword="param_path",
        declared_keywords=(
            "charge",
            "multiplicity",
            "uhf",
            "param_path",
            "nn_weights",
            "nn_fallback",
        ),
    ),
    "am1_rs": AseProfile(
        "am1_rs.ase:AM1", MULT, description="AM1 (am1-rs)", declared_keywords=_AM1_LIKE
    ),
    "pm3_rs": AseProfile("pm3_rs.ase:PM3", MULT, {"method": "pm3"}, description="PM3"),
    "pm3_rs_d3h4x": AseProfile(
        "pm3_rs.ase:PM3", MULT, {"method": "pm3-d3h4x"}, description="PM3-D3H4X"
    ),
    "pm6_rs": AseProfile("pm6_rs.ase:PM6", MULT, {"method": "pm6"}, description="PM6"),
    "pm6_rs_d3h4x": AseProfile(
        "pm6_rs.ase:PM6", MULT, {"method": "pm6-d3h4x"}, description="PM6-D3H4X"
    ),
    "pm7_rs": AseProfile("pm7_rs.ase:PM7", MULT, {"method": "pm7"}, description="PM7"),
    "dftb2_rs": AseProfile(
        "dftb_rs.calculator:DFTB",
        DFTB_SPIN,
        {"params": "3ob", "level": 2},
        description="DFTB2/3ob",
        declared_keywords=("charge", "spin", "params", "level"),
    ),
    "dftb3_rs": AseProfile(
        "dftb_rs.calculator:DFTB",
        DFTB_SPIN,
        {"params": "3ob", "level": 3},
        description="DFTB3/3ob",
        declared_keywords=("charge", "spin", "params", "level"),
    ),
    "dft_rs": AseProfile(
        "dft_rs:DftRs",
        MULT,
        {"basis": "def2-svp", "xc": "wb97xd"},
        description="DFT (dft-rs)",
        declared_keywords=("charge", "multiplicity", "basis", "xc"),
    ),
    "uff_rs": AseProfile(
        "mmff_rs.ase:MMFF",
        NO_STATE,
        {"force_field": "uff-1992"},
        description="UFF (mmff-rs)",
        electrons=False,
    ),
    "mace_mp": AseProfile(_mace("mace_mp"), NO_STATE, hessian=None, electrons=False),
    "mace_off": AseProfile(_mace("mace_off"), NO_STATE, hessian=None, electrons=False),
    "uma-s-1": AseProfile(_fairchem(), INFO_SPIN_MULT, hessian=None, path_keyword="path"),
    "uma-s-1p1": AseProfile(_fairchem(), INFO_SPIN_MULT, hessian=None, path_keyword="path"),
    "uma-s-1p1-cuda": AseProfile(
        _fairchem(device="cuda"), INFO_SPIN_MULT, hessian=None, path_keyword="path"
    ),
    "uma-m-1p1": AseProfile(_fairchem(), INFO_SPIN_MULT, hessian=None, path_keyword="path"),
    "gfn0-xtb": AseProfile("pygfn0:GFN0", CHARGE_ONLY, hessian=None, description="PyGFN0"),
    # AIMNet2 (package aimnet; the model is downloaded on first use and checked against its
    # SHA-256). aimnet2 is closed-shell and ignores the multiplicity; aimnet2-nse is for
    # open shells; aimnet2-rxn is for reactions of neutral molecules.
    **{
        name: AseProfile(
            "aimnet.calculators:AIMNet2ASE",
            AIMNET_STATE,
            {"base_calc": name},
            description=f"AIMNet2 ({model})",
            declared_keywords=("base_calc", "charge", "mult", "validate_species"),
        )
        for name, model in [
            ("aimnet2", "wB97M-D3, closed shell"),
            ("aimnet2-nse", "wB97M-D3, open shell"),
            ("aimnet2-rxn", "reactions, neutral molecules"),
            ("aimnet2-b973c", "B97-3c"),
            ("aimnet2-pd", "palladium chemistry"),
        ]
    },
}
"""Built-in profiles (lowercase names)."""


def register_profile(name: str, profile: AseProfile, *, replace: bool = False) -> None:
    """Add a profile (usable as ``-os name``)."""
    key = name.lower()
    if key in PROFILES and not replace:
        raise KeyError(f"profile '{name}' exists (pass replace=True)")
    PROFILES[key] = profile


def profile_engine(
    name: str,
    *,
    options: Mapping[str, Any] | None = None,
    software_paths: Mapping[str, str] | None = None,
) -> AseEngine:
    """The engine of profile ``name``; ``options`` add or override constructor keywords."""
    try:
        p = PROFILES[name.lower()]
    except KeyError:
        raise KeyError(f"no ASE profile '{name}' (known: {', '.join(sorted(PROFILES))})") from None
    from mopt_rs.calculators.program_options import split_program_options

    plain, _, setup = split_program_options(options or {})
    kwargs = dict(p.kwargs)
    path = (software_paths or {}).get(name.lower()) or (software_paths or {}).get(name)
    if p.path_keyword and path:
        kwargs[p.path_keyword] = path
    kwargs.update(plain)
    declared = p.declared_keywords + ((p.path_keyword,) if p.path_keyword else ())
    engine = AseEngine(
        p.factory,
        kwargs=kwargs,
        state=p.state,
        hessian=p.hessian,
        name=name.lower(),
        electrons=p.electrons,
        declared_keywords=declared if p.declared_keywords else (),
        setup=setup,
    )
    if p.path_keyword and path:
        engine.notes = (f"{p.path_keyword}={path} from software_path.conf",)
    return engine
