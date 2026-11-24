"""Generate MultiOptPy's GFN-FF-like and GFN0-like model-Hessian parameters as Rust tables.

Usage:

    python -I tools/params/gen_multioptpy_gfn.py <MultiOptPy root> <output .rs>

The parameters are read with ``ast`` from ``multioptpy/Parameters/gfnff_param.py`` and
``gfn0xtb_param.py`` (never executed): element tables become arrays by atomic number, tables
keyed by element tuples become `match` functions on the symbols (`None` when MultiOptPy has no
entry, so that the Rust side applies MultiOptPy's fallbacks), and arithmetic values (such as
``109.5 * np.pi / 180`` or ``0.1 * self.kcalmol2hartree``) are evaluated here.

These are MultiOptPy's own approximate parameters (inspired by GFN-FF and GFN0-xTB, not the
published parameter sets).

This file is part of mopt-rs.
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

import ast
import math
import sys
from pathlib import Path

SYMBOLS = [
    "X", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S",
    "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge",
    "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
    "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn",
]  # fmt: skip

HARTREE2KCALMOL = 627.509
BOHR2ANG = 0.52917721067


def value(node: ast.AST, names: dict) -> object:
    """Literal values and arithmetic on numbers and the known names only."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, (ast.List, ast.Tuple)):
        return [value(e, names) for e in node.elts]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -value(node.operand, names)
    if isinstance(node, (ast.Name, ast.Attribute)):
        key = ast.unparse(node)
        if key in names:
            return names[key]
        raise SystemExit(f"unknown name {key}")
    if isinstance(node, ast.BinOp):
        a, b = value(node.left, names), value(node.right, names)
        ops = {ast.Add: lambda: a + b, ast.Sub: lambda: a - b, ast.Mult: lambda: a * b,
               ast.Div: lambda: a / b, ast.Pow: lambda: a**b}  # fmt: skip
        for t, f in ops.items():
            if isinstance(node.op, t):
                return f()
    if isinstance(node, ast.Dict):
        return {tuple(k) if isinstance(k, list) else k: value(v, names)
                for k, v in zip((value(k, names) for k in node.keys), node.values, strict=True)}  # fmt: skip
    raise SystemExit(f"unsupported expression: {ast.unparse(node)}")


def class_init(path: Path, cls: str) -> dict:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = {
        "np.pi": math.pi,
        "self.kcalmol2hartree": 1.0 / HARTREE2KCALMOL,
        "self.bohr2ang": BOHR2ANG,
    }
    out: dict = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for f in node.body:
                if isinstance(f, ast.FunctionDef) and f.name == "__init__":
                    for s in f.body:
                        if isinstance(s, ast.Assign):
                            t = ast.unparse(s.targets[0])
                            if t in ("self.bohr2ang", "self.kcalmol2hartree"):
                                continue
                            v = value(s.value, names)
                            out[t[5:]] = v
                            names[t] = v
    return out


def rust_f64(v: float) -> str:
    if float(v) == math.pi:
        return "std::f64::consts::PI"
    r = repr(float(v))
    return r if ("." in r or "e" in r or "inf" in r or "nan" in r) else r + ".0"


def element_array(name: str, table: dict, index: int | None, doc: str) -> list[str]:
    vals = []
    for s in SYMBOLS:
        v = table.get(s)
        if v is None:
            vals.append("f64::NAN")
        else:
            vals.append(rust_f64(v[index] if index is not None else v))
    lines = [f"/// {doc}", f"pub static {name}: [f64; {len(SYMBOLS)}] = ["]
    lines += [f"    {v}," for v in vals]
    lines.append("];")
    return lines


def tuple_match(name: str, table: dict, arity: int, width: int, doc: str) -> list[str]:
    """`fn name(e0, .., [kind]) -> Option<..>`; keys longer than `arity` carry a final tag
    matched against `kind: Option<&str>` (keys of length `arity` match `kind == None`)."""
    tagged = any(isinstance(k, tuple) and len(k) > arity for k in table)
    args = ", ".join(f"e{k}: &str" for k in range(arity)) + (
        ", kind: Option<&str>" if tagged else ""
    )
    pat_vars = ", ".join(f"e{k}" for k in range(arity)) + (", kind" if tagged else "")
    pat_vars = "e0" if arity == 1 and not tagged else f"({pat_vars})"
    ret = "f64" if width == 1 else f"[f64; {width}]"
    lines = [f"/// {doc}", f"pub fn {name}({args}) -> Option<{ret}> {{", f"    match {pat_vars} {{"]
    for key, v in table.items():
        key = key if isinstance(key, tuple) else (key,)
        names = [f'"{k}"' for k in key[:arity]]
        if tagged:
            names.append(f'Some("{key[arity]}")' if len(key) > arity else "None")
        pat = ", ".join(names) if (arity > 1 or tagged) else names[0]
        pat = f"({pat})" if (arity > 1 or tagged) else pat
        val = rust_f64(v) if width == 1 else "[" + ", ".join(rust_f64(x) for x in v) + "]"
        lines.append(f"        {pat} => Some({val}),")
    lines += ["        _ => None,", "    }", "}"]
    return lines


def main() -> None:
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    ff = class_init(root / "multioptpy/Parameters/gfnff_param.py", "GFNFFParameters")
    g0 = class_init(root / "multioptpy/Parameters/gfn0xtb_param.py", "GFN0Parameters")
    lines = [
        "//! MultiOptPy 1.21.2's GFN-FF-like and GFN0-like model-Hessian parameters",
        "//! (`Parameters/gfnff_param.py`, `Parameters/gfn0xtb_param.py`): MultiOptPy's own",
        "//! approximate values, not the published GFN-FF or GFN0-xTB parameter sets. Element arrays",
        "//! are indexed by atomic number (`f64::NAN`: no entry); tuple tables return `None` when",
        "//! MultiOptPy has no entry. Units as in MultiOptPy.",
        "//!",
        "//! Generated by tools/params/gen_multioptpy_gfn.py; do not edit.",
        "//!",
        "//! MultiOptPy: `Parameters/gfnff_param.py:GFNFFParameters`,",
        "//!  `Parameters/gfn0xtb_param.py:GFN0Parameters`.",
        "//!",
        "//! This file is part of mopt-rs.",
        "//! Copyright (C) 2023-2026 ss0832",
        "//! SPDX-License-Identifier: GPL-3.0-or-later",
        "",
    ]
    ep = ff["element_params"]
    for idx, nm, doc in [
        (0, "GFNFF_COV_RADIUS_ANG", "`element_params[...][0]`: covalent radius (Angstrom)."),
        (1, "GFNFF_VDW_RADIUS_ANG", "`element_params[...][1]`: van der Waals radius (Angstrom)."),
        (2, "GFNFF_ELECTRONEGATIVITY", "`element_params[...][2]`: electronegativity."),
        (3, "GFNFF_POLARIZABILITY", "`element_params[...][3]`: polarizability (a.u.)."),
    ]:
        lines += element_array(nm, ep, idx, doc) + [""]
    lines += element_array("GFNFF_REF_CN", ff["ref_cn"], None, "`ref_cn` (default 1.0).") + [""]
    lines += tuple_match(
        "gfnff_bond", ff["bond_params"], 2, 2, "`bond_params` (sorted symbols): [r0 Bohr, k]."
    ) + [""]
    lines += tuple_match(
        "gfnff_angle", ff["angle_params"], 3, 2, "`angle_params`: [theta0 deg, k]."
    ) + [""]
    lines += tuple_match(
        "gfnff_torsion", ff["torsion_params"], 4, 3, "`torsion_params`: [V1, V2, V3] (kcal/mol)."
    ) + [""]
    lines += tuple_match(
        "gfnff_hbond", ff["hbond_params"], 3, 2, "`hbond_params`: [r0 Angstrom, k kcal/mol]."
    ) + [""]
    for k in ["bond_scaling", "bond_decay", "d4_s6", "d4_s8", "d4_s9", "d4_a1", "d4_a2",
              "default_bond_k", "default_angle_k"]:  # fmt: skip
        lines += [
            f"/// `GFNFFParameters.{k}`.",
            f"pub const GFNFF_{k.upper()}: f64 = {rust_f64(ff[k])};",
        ]
    lines += ["/// `GFNFFParameters.default_torsion_v` (kcal/mol).",
              "pub const GFNFF_DEFAULT_TORSION_V: [f64; 3] = ["
              + ", ".join(rust_f64(v) for v in ff["default_torsion_v"]) + "];", ""]  # fmt: skip
    lines += element_array("GFN0_RADIUS_BOHR", g0["rad"], None, "`rad` (Bohr; default 1.0).") + [""]
    lines += element_array("GFN0_EN", g0["en"], None, "`en` (default 2.0).") + [""]
    lines += tuple_match("gfn0_reference_bond_length", g0["referenceBondLength"], 2, 1,
                         "`referenceBondLength` (Angstrom); `kind` is the bond type of tagged entries.")  # fmt: skip
    lines.append("")
    for dct in ["bondForceFactors", "naturalAngles", "CNParams"]:
        lines += tuple_match(
            f"gfn0_{dct.lower()}", {(k,): v for k, v in g0[dct].items()}, 1, 1, f"`{dct}`."
        ) + [""]
    for k in ["kCN", "shellPoly", "kStretchBase", "kAngleBase", "V2Base", "V3Base"]:
        lines += [
            f"/// `GFN0Parameters.{k}`.",
            f"pub const GFN0_{k.upper()}: f64 = {rust_f64(g0[k])};",
        ]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
