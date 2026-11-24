"""Generate the GFN-FF element parameters of mopt-rs (H to Rn, Z = 1 to 86).

Usage:

    python -I tools/params/gen_gfnff.py <xtb source tree> <output .rs> <libxtb.so of xtb 6.7.1>

The elements are H to Rn, those of the xtb 6.7.1 release, the reference of mopt-rs's GFN-FF,
whose tables have 86 entries. The source tree at hand is a later state of xtb (it still calls
itself 6.7.1) with an extension of GFN-FF to the lanthanides and actinides: its tables
(``gfnff_param.f90``, ``gfnff_rab.f``, ``zeta`` of ``gfnff_ini.f90``) have 103 entries, to Lr,
with refitted ``chi``, ``gam``, ``cnf``, ``alp`` and ``r0`` of La to Lu, a bond-length factor of
lanthanide and actinide halides (``scaleF`` of ``gfnff_rab.f``) and a removal of the H atoms
bonded to them (``adjust_NB_LnH_AnH``). The release has none of these: its library holds the
86-entry tables, the lanthanide entries of those five tables differ, and the factors of
``scaleF`` are not in it. Every other entry of H to Rn is the same in both. The xtb 6.7.1
executable gives no usable result for Fr to Lr (Fr and Ra stop in the topology charges, Ac to
Lr find no bonds, UF6 has a NaN dispersion energy), and the D4 reference systems of
``include/param_ref.fh``, from which GFN-FF's dispersion takes its C6 coefficients, stop at Rn
(``refn`` is given for Z = 1 to 86 and 112 to 118 only); see ``docs/source/calculators.md``.

The lanthanide entries of ``chi``, ``gam``, ``cnf``, ``alp`` (``gfnff_param.f90``) and ``r0``
(``gfnff_rab.f``) are therefore those of the xtb 6.7.1 library given as the third argument,
read as data (never executed): each table is found by its first entries, which are the same as
in the source, as 86 little-endian doubles, and every other entry is checked to agree with the
source before the 15 lanthanide entries are taken.

The values are read as text (never executed) from grimme-lab/xtb (LGPL-3.0-or-later):

- ``src/gfnff/gfnff_param.f90``: the GFN-FF (Angew. Chem. 2020) element parameters
  (``chi``, ``gam``, ``cnf``, ``alp``, ``bond``, ``repa``, ``repan``, ``angl``, ``angl2``,
  ``tors``, ``tors2``) and the fixed tables ``en``, ``rad``, ``metal``, ``group``, ``normcn``,
  ``repz``;
- ``src/param/covalentradd3.f90``: the D3 covalent radii (Angstrom);
- ``src/param/sqrtzr4r2.f90``: ``<r^4>/<r^2>``;
- ``src/gfnff/gfnff_rab.f``: the CN-dependent bond-length estimate (``r0``, ``cnfak``, ``en``,
  and the EN polynomial ``p(6,2)``);
- ``src/gfnff/gfnff_ini.f90``: the effective nuclear charges ``zeff`` and chemical hardnesses
  ``c`` of the D4 charge scaling ``zeta`` (GFN-FF's own copy);
- ``include/param_ref.fh``: the D4 reference systems (``refn``, ``refcn``, ``hcount``,
  ``ascale``, ``refsys``, ``alphaiw``, ``sscale``, ``secaiw``). The reference polarizabilities of
  GFN-FF's dispersion model (xtb's ``newD3Model``: D4 reference polarizabilities without charge
  scaling) are written per element with the weights of the Casimir-Polder integration
  (``trapzd``); mopt-rs forms the C6 coefficients of the element pairs of a structure from
  them (a constant table of every pair of elements is too large to compile).

xtb is built with ``-fdefault-real-8``, so the constants of the fixed-form files are double
precision as written.

The SHA-256 of each file is written into the output. Values are written as in the files;
mopt-rs converts them as xtb does.

This file is part of mopt-rs.
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

import hashlib
import re
import struct
import sys
from pathlib import Path

LANTHANIDES = range(57, 72)


def release_table(lib: bytes, source: list[float], name: str) -> list[float]:
    """The 86 entries of the table ``name`` in the xtb 6.7.1 library: the lanthanide entries
    from the library, the others checked against ``source``."""
    pattern = struct.pack("<5d", *source[:5])
    starts = [m.start() for m in re.finditer(re.escape(pattern), lib)]
    if not starts:
        raise SystemExit(f"{name}: not found in the library")
    tables = {struct.unpack_from("<86d", lib, s) for s in starts}
    if len(tables) != 1:
        raise SystemExit(f"{name}: {len(tables)} different tables in the library")
    (table,) = tables
    for z in range(1, 87):
        if z not in LANTHANIDES and table[z - 1] != source[z - 1]:
            raise SystemExit(f"{name}: Z = {z} differs from the source ({table[z - 1]})")
    return list(table) + source[86:]

ELEMENTS = list(range(1, 87))
SYMBOL = {
    z + 1: s
    for z, s in enumerate(
        "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge "
        "As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm "
        "Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th "
        "Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr".split()
    )
}
REALS = ["chi", "gam", "cnf", "alp", "bond", "repa", "repan", "angl", "angl2", "tors", "tors2"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rust_f64(v: float) -> str:
    r = repr(float(v))
    return r if ("." in r or "e" in r) else r + ".0"


def numbers(text: str) -> list[float]:
    text = " ".join(line.split("!")[0] for line in text.splitlines())
    text = re.sub(r"_wp\b", "", text)
    out = []
    for t in re.findall(r"[+-]?\d+\.?\d*(?:[eEdD][+-]?\d+)?", text):
        out.append(float(t.replace("d", "e").replace("D", "e")))
    return out


def array(text: str, name: str) -> list[float]:
    """``name(...) = [ ... ]`` or ``name(...) = (/ ... /)`` (first occurrence)."""
    m = re.search(rf"::\s*{name}\s*\([^)]*\)\s*=\s*(\[|\(/)", text)
    if m is None:
        raise SystemExit(f"missing array {name}")
    close = "]" if m.group(1) == "[" else "/)"
    rest = text[m.end() :]
    return numbers(rest[: rest.index(close)])


def data_array(text: str, name: str) -> list[float]:
    """The first fixed-form ``data name / ... /`` statement."""
    m = re.search(rf"data\s+{name}\s*/", text)
    if m is None:
        raise SystemExit(f"missing data {name}")
    rest = text[m.end() :]
    return numbers(rest[: rest.index("/")])


FREQ = [
    0.000001,
    0.05,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1.0,
    1.2,
    1.4,
    1.6,
    1.8,
    2.0,
    2.5,
    3.0,
    4.0,
    5.0,
    7.5,
    10.0,
]
PI = 3.1415926535897932384626433832795029


def trapzd_weights() -> list[float]:
    """xtb's ``trapzd`` weights (``disp/dftd4.F90``)."""
    w = [FREQ[1] - FREQ[0]]
    for k in range(1, 22):
        w.append((FREQ[k] - FREQ[k - 1]) + (FREQ[k + 1] - FREQ[k]))
    w.append(FREQ[22] - FREQ[21])
    return [0.5 * v for v in w]


def fortran_data(text: str) -> dict[str, dict[tuple[str, ...], list[float]]]:
    """All ``data name(idx) / values /`` statements of a fixed-form include file."""
    lines = []
    for line in text.splitlines():
        line = line.split("!")[0]
        lines.append(line.rstrip().rstrip("&"))
    flat = " ".join(t.lstrip().lstrip("&") for t in lines)
    out: dict[str, dict[tuple[str, ...], list[float]]] = {}
    for name, idx, vals in re.findall(r"data\s+(\w+)\s*\(([^)]*)\)\s*/([^/]*)/", flat):
        key = tuple(t.strip() for t in idx.split(","))
        out.setdefault(name, {})[key] = numbers(vals)
    return out


def zeta_tables(path: Path) -> tuple[list[float], list[float]]:
    """``zeff(103)`` and ``c(1:103)`` of the function ``zeta`` of ``gfnff_ini.f90``."""
    text = path.read_text(encoding="utf-8")
    m = re.search(r"function zeta\(at,q\)(.*?)end function zeta", text, re.S)
    if m is None:
        raise SystemExit("missing function zeta")
    body = m.group(1)
    out = []
    for name in ("zeff", "c"):
        mm = re.search(rf"::\s*{name}\s*\([^)]*\)\s*=\s*\(/", body)
        if mm is None:
            raise SystemExit(f"missing zeta table {name}")
        rest = body[mm.end() :]
        vals = numbers(rest[: rest.index("/)")])
        if len(vals) != 103:
            raise SystemExit(f"zeta table {name}: {len(vals)} values")
        out.append(vals)
    return out[0], out[1]


def dispersion_references(path: Path) -> tuple[list[int], list[list[float]], list]:
    """``refn``, ``refcn`` and the scaled reference polarizabilities ``alpha[a][ia][k]`` of
    ``newD3Model``."""
    d = fortran_data(path.read_text(encoding="utf-8"))
    weights = trapzd_weights()
    nref, refcn, alpha = [], [], []
    for z in ELEMENTS:
        nr = int(d["refn"][(str(z),)][0])
        if not 1 <= nr <= 7:
            raise SystemExit(f"Z={z}: {nr} reference systems")
        nref.append(nr)
        cns, als = [], []
        for r in range(1, nr + 1):
            key = (str(r), str(z))
            cns.append(d["refcn"][key][0])
            system = int(d["refsys"][key][0])
            sec = d["secaiw"][(":", str(system))]
            ss = d["sscale"][(str(system),)][0]
            aiw = d["alphaiw"][(":", str(r), str(z))]
            hc = d["hcount"][key][0]
            asc = d["ascale"][key][0]
            if len(sec) != 23 or len(aiw) != 23:
                raise SystemExit(f"bad polarizability table for Z={z} ref {r}")
            als.append([max(asc * (aiw[k] - hc * (ss * sec[k])), 0.0) for k in range(23)])
        refcn.append(cns + [0.0] * (7 - nr))
        alpha.append(als + [[0.0] * 23 for _ in range(7 - nr)])
    return nref, refcn, alpha


def main() -> None:
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    param = root / "src/gfnff/gfnff_param.f90"
    covd3 = root / "src/param/covalentradd3.f90"
    r4r2 = root / "src/param/sqrtzr4r2.f90"
    rabf = root / "src/gfnff/gfnff_rab.f"
    refs = root / "include/param_ref.fh"
    ini = root / "src/gfnff/gfnff_ini.f90"
    disp_nref, disp_refcn, disp_alpha = dispersion_references(refs)
    zeta_zeff, zeta_c = zeta_tables(ini)
    rtext = rabf.read_text(encoding="utf-8")
    rab_cols = {k: data_array(rtext, k) for k in ("en", "r0", "cnfak")}
    for k, v in rab_cols.items():
        if len(v) != 103:
            raise SystemExit(f"gfnff_rab.f {k}: {len(v)} values")
    rab_p = {}
    for i, j, v in re.findall(r"p\((\d),(\d)\)\s*=\s*([-+.\d]+)", rtext):
        rab_p.setdefault((int(i), int(j)), float(v))
    p = param.read_text(encoding="utf-8")
    tables = {k: array(p, f"{k}_angewChem2020") for k in REALS}
    for k in ["en", "rad", "metal", "group", "normcn", "repz"]:
        tables[k] = array(p, k)
    tables["covd3"] = array(covd3.read_text(encoding="utf-8"), "covalentRadD3")
    tables["r4r2"] = array(r4r2.read_text(encoding="utf-8"), "r4Overr2")
    for k, v in tables.items():
        if len(v) < 103:
            raise SystemExit(f"{k}: only {len(v)} values")
    libpath = Path(sys.argv[3])
    lib = libpath.read_bytes()
    for k in ("chi", "gam", "cnf", "alp"):
        tables[k] = release_table(lib, tables[k], k)
    rab_cols["r0"] = release_table(lib, rab_cols["r0"], "r0")
    o = [
        "//! GFN-FF element parameters for H to Rn (Z = 1 to 86), those of xtb 6.7.1; the reasons",
        "//! why Fr to Lr are left out are in `tools/params/gen_gfnff.py`.",
        "//!",
        "//! Generated by tools/params/gen_gfnff.py from grimme-lab/xtb (LGPL-3.0-or-later);",
        "//! do not edit. The lanthanide entries of `chi`, `gam`, `cnf`, `alp` and `r0` are those of",
        f"//! the xtb 6.7.1 library `{libpath.name}` (SHA-256 {hashlib.sha256(lib).hexdigest()}),",
        "//! the rest those of the source files. Sources and SHA-256:",
        f"//! - `src/gfnff/gfnff_param.f90`: {sha256(param)}",
        f"//! - `src/param/covalentradd3.f90`: {sha256(covd3)}",
        f"//! - `src/param/sqrtzr4r2.f90`: {sha256(r4r2)}",
        f"//! - `src/gfnff/gfnff_rab.f`: {sha256(rabf)}",
        f"//! - `src/gfnff/gfnff_ini.f90`: {sha256(ini)}",
        f"//! - `include/param_ref.fh`: {sha256(refs)}",
        "//!",
        "//! Method: S. Spicher, S. Grimme, Angew. Chem. Int. Ed. 2020, 59, 15665",
        "//! (DOI 10.1002/anie.202004239); parameters of that version (`angewChem2020`).",
        "//!",
        "//! MultiOptPy: no counterpart (new in mopt-rs); MultiOptPy's `Parameters/gfnff_param.py` is in",
        "//!  `multioptpy_gfn`.",
        "//!",
        "//! This file is part of mopt-rs.",
        "//! Copyright (C) 2023-2026 ss0832",
        "//! SPDX-License-Identifier: GPL-3.0-or-later",
        "",
        "/// One element (values as in xtb's tables).",
        "#[derive(Debug, Clone, Copy, PartialEq)]",
        "pub struct GfnffElementRaw {",
        "    /// Atomic number.",
        "    pub z: u8,",
    ]
    for k in REALS:
        o += [f"    /// `{k}_angewChem2020`.", f"    pub {k}: f64,"]
    o += [
        "    /// Pauling electronegativity (`en`).",
        "    pub en: f64,",
        "    /// Covalent radius of the HB/XB terms (`rad`, Angstrom).",
        "    pub rad: f64,",
        "    /// `metal` (0 non-metal).",
        "    pub metal: i32,",
        "    /// `group`.",
        "    pub group: i32,",
        "    /// `normcn`.",
        "    pub normcn: i32,",
        "    /// `repz`.",
        "    pub repz: f64,",
        "    /// D3 covalent radius (`covalentRadD3`, Angstrom).",
        "    pub covalent_radius_ang: f64,",
        "    /// `r4Overr2`.",
        "    pub r4_over_r2: f64,",
        "    /// Bond-length estimate `(en, r0, cnfak)` of `gfnff_rab.f`.",
        "    pub rab: [f64; 3],",
        "    /// `(zeff, c)` of the D4 charge scaling `zeta` of `gfnff_ini.f90`.",
        "    pub zeta: [f64; 2],",
        "}",
        "",
        "/// `gfnff_rab.f` EN polynomial `p(row, 1..2)` by period row 1..6.",
        "pub const GFNFF_RAB_P: [[f64; 2]; 6] = [",
        *[f"    [{rust_f64(rab_p[(r, 1)])}, {rust_f64(rab_p[(r, 2)])}]," for r in range(1, 7)],
        "];",
        "",
        f"/// The elements {SYMBOL[ELEMENTS[0]]} to {SYMBOL[ELEMENTS[-1]]} in the order of Z (the entry of Z is",
        "/// at index `Z - 1`).",
        f"pub static GFNFF_ELEMENTS: [GfnffElementRaw; {len(ELEMENTS)}] = [",
    ]
    for z in ELEMENTS:
        i = z - 1
        fields = [f"z: {z}"] + [f"{k}: {rust_f64(tables[k][i])}" for k in REALS]
        fields += [
            f"en: {rust_f64(tables['en'][i])}",
            f"rad: {rust_f64(tables['rad'][i])}",
            f"metal: {int(tables['metal'][i])}",
            f"group: {int(tables['group'][i])}",
            f"normcn: {int(tables['normcn'][i])}",
            f"repz: {rust_f64(tables['repz'][i])}",
            f"covalent_radius_ang: {rust_f64(tables['covd3'][i])}",
            f"r4_over_r2: {rust_f64(tables['r4r2'][i])}",
            "rab: [" + ", ".join(rust_f64(rab_cols[k][i]) for k in ("en", "r0", "cnfak")) + "]",
            f"zeta: [{rust_f64(zeta_zeff[i])}, {rust_f64(zeta_c[i])}]",
        ]
        o.append(f"    // {SYMBOL[z]}")
        o.append("    GfnffElementRaw { " + ", ".join(fields) + " },")
    o.append("];")
    o += [
        "",
        "/// Number of dispersion reference systems per element (order of `GFNFF_ELEMENTS`).",
        f"pub const GFNFF_DISP_NREF: [usize; {len(ELEMENTS)}] = [" + ", ".join(str(v) for v in disp_nref) + "];",
        "",
        "/// Reference CNs of the dispersion model (`refcn`).",
        f"pub const GFNFF_DISP_REFCN: [[f64; 7]; {len(ELEMENTS)}] = [",
        *["    [" + ", ".join(rust_f64(v) for v in row) + "]," for row in disp_refcn],
        "];",
        "",
        "/// The weights of xtb's `trapzd` over the 23 imaginary frequencies of the D4 references.",
        "pub const GFNFF_DISP_TRAPZD: [f64; 23] = ["
        + ", ".join(rust_f64(v) for v in trapzd_weights())
        + "];",
        "",
        "/// The scaled reference polarizabilities `alpha[a][ref][k]` of xtb's `newD3Model` (D4",
        "/// references without charge scaling) at the 23 frequencies; the C6 of a pair of",
        "/// references is `3 / pi sum_k alpha_a alpha_b w_k`.",
        f"pub static GFNFF_DISP_ALPHA: [[[f64; 23]; 7]; {len(ELEMENTS)}] = [",
    ]
    for rows in disp_alpha:
        o.append("    [")
        for line in rows:
            o.append("        [" + ", ".join(rust_f64(v) for v in line) + "],")
        o.append("    ],")
    o.append("];")
    out.write_text("\n".join(o) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
