"""Generate the MultiOptPy parameter tables that its model Hessians (mopt-rs ``legacy_``
names) use, where they differ from the tables mopt-rs uses otherwise.

Usage:

    python -I tools/params/gen_multioptpy_tables.py <MultiOptPy root> <output .rs>

The tables are read with ``ast`` from MultiOptPy's sources (never executed) and written raw, in
MultiOptPy's units; the conversion to atomic units is done in Rust with MultiOptPy's constants.

This file is part of mopt-rs.
"""

import ast
import sys
from pathlib import Path

SYMBOLS = [
    "X",
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
]  # noqa: SIM905

# (Rust name, source file, function, variable, unit comment)
TABLES = [
    ("MOP_D2_VDW_ANG", "Parameters/d2.py", "D2_VDW_radii_lib", "D2_VDW_distance", "Angstrom"),
    (
        "MOP_D2_C6_J_NM6_MOL",
        "Parameters/d2.py",
        "D2_C6_coeff_lib",
        "C6_coefficients",
        "J nm^6 / mol",
    ),
    (
        "MOP_TRIPLE_COV_ANG",
        "Parameters/covalent_radii.py",
        "triple_covalent_radii_lib",
        "CRL",
        "Angstrom",
    ),
    ("MOP_D3_R4R2", "Parameters/d3.py", "__init__", "self.r4r2_values", "D3Parameters r4r2"),
    ("MOP_D4_R4R2", "Parameters/d4.py", "__init__", "self.r4r2_values", "D4Parameters r4r2"),
    (
        "MOP_D4_POLARIZABILITY",
        "Parameters/d4.py",
        "__init__",
        "self.ref_polarizabilities",
        "D4Parameters",
    ),
    (
        "MOP_D4_ELECTRONEGATIVITY",
        "Parameters/d4.py",
        "__init__",
        "self.electronegativity",
        "D4Parameters",
    ),
]

# (Rust name, source file, function, variable): scalar assignments.
SCALARS = [
    ("MOP_D3_DEFAULT_R4R2", "Parameters/d3.py", "__init__", "self.default_r4r2"),
    ("MOP_D4_DEFAULT_R4R2", "Parameters/d4.py", "__init__", "self.default_r4r2"),
    (
        "MOP_D4_DEFAULT_POLARIZABILITY",
        "Parameters/d4.py",
        "__init__",
        "self.default_polarizability",
    ),
    (
        "MOP_D4_DEFAULT_ELECTRONEGATIVITY",
        "Parameters/d4.py",
        "__init__",
        "self.default_electronegativity",
    ),
    ("MOP_D4_GA", "Parameters/d4.py", "__init__", "self.ga"),
    ("MOP_D4_GC", "Parameters/d4.py", "__init__", "self.gc"),
]

CONSTANTS = [
    "hartree2kcalmol",
    "bohr2angstroms",
    "hartree2kjmol",
    "hartree2eV",
    "hartree2j",
    "mol2au",
]


def _targets(t: ast.expr, var: str) -> bool:
    if var.startswith("self."):
        return (
            isinstance(t, ast.Attribute)
            and isinstance(t.value, ast.Name)
            and t.value.id == "self"
            and t.attr == var[5:]
        )
    return isinstance(t, ast.Name) and t.id == var


def value_in_function(tree: ast.Module, func: str, var: str) -> ast.expr:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func:
            for sub in ast.walk(node):
                if isinstance(sub, ast.Assign) and any(_targets(t, var) for t in sub.targets):
                    return sub.value
    raise SystemExit(f"{func}.{var} not found")


def dict_in_function(tree: ast.Module, func: str, var: str) -> dict:
    return ast.literal_eval(value_in_function(tree, func, var))


def arithmetic(node: ast.AST) -> float:
    """Value of a constant arithmetic expression (numbers, + - * / **, unary minus) only."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -arithmetic(node.operand)
    if isinstance(node, ast.BinOp):
        x, y = arithmetic(node.left), arithmetic(node.right)
        ops = {ast.Add: x + y, ast.Sub: x - y, ast.Mult: x * y}
        if type(node.op) in ops:
            return ops[type(node.op)]
        if isinstance(node.op, ast.Div):
            return x / y
        if isinstance(node.op, ast.Pow):
            return x**y
    raise SystemExit(f"not a constant arithmetic expression: {ast.dump(node)}")


def unit_constants(root: Path) -> dict:
    tree = ast.parse((root / "multioptpy/Parameters/unit_values.py").read_text(encoding="utf-8"))
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            t = node.targets[0]
            if isinstance(t, ast.Attribute) and t.attr in CONSTANTS:
                out[t.attr] = arithmetic(node.value)
    return out


def main() -> None:
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    lines = [
        "//! MultiOptPy 1.21.2's parameter tables used by its model Hessians (the `legacy_` models),",
        "//! in MultiOptPy's units; `f64::NAN` marks elements MultiOptPy has no value for.",
        "//!",
        "//! Generated by tools/params/gen_multioptpy_tables.py; do not edit.",
        "//!",
        "//! MultiOptPy: `Parameters/d2.py`, `d3.py`, `d4.py`, `covalent_radii.py`, and the other",
        "//!  tables of `Parameters/` used by `ModelHessian/*.py`.",
        "//!",
        "//! This file is part of mopt-rs.",
        "//! Copyright (C) 2023-2026 ss0832",
        "//! SPDX-License-Identifier: GPL-3.0-or-later",
        "",
    ]
    consts = unit_constants(root)
    for k in CONSTANTS:
        lines.append(f"/// MultiOptPy's `UnitValueLib.{k}`.")
        lines.append(f"pub const MOP_{k.upper()}: f64 = {consts[k]!r};")
    for name, path, func, var, unit in TABLES:
        tree = ast.parse((root / "multioptpy" / path).read_text(encoding="utf-8"))
        table = dict_in_function(tree, func, var)
        values = [table.get(s, float("nan")) for s in SYMBOLS]
        lines.append("")
        lines.append(f"/// `{func}` ({unit}) by atomic number, X..Rn.")
        lines.append(f"pub static {name}: [f64; {len(SYMBOLS)}] = [")
        lines += ["    f64::NAN," if v != v else f"    {float(v)!r}," for v in values]
        lines.append("];")
    for name, path, func, var in SCALARS:
        tree = ast.parse((root / "multioptpy" / path).read_text(encoding="utf-8"))
        v = arithmetic(value_in_function(tree, func, var))
        lines.append("")
        lines.append(f"/// `{var}` of `{path}`.")
        lines.append(f"pub const {name}: f64 = {v!r};")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
