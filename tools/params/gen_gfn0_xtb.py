"""Generate the GFN0-xTB parameter tables of mopt-rs (H to Rn, every element of GFN0-xTB).

Usage:

    python -I tools/params/gen_gfn0_xtb.py <upstream snapshots> <output .rs>

``<upstream snapshots>`` is a directory holding the pinned upstream files that gfn0-rs
(github.com/ss0832/gfn0-rs, GPL-3.0-or-later) redistributes under its ``third_party/``:

- ``xtb/param_gfn0-xtb.txt`` (grimme-lab/xtb, LGPL-3.0-or-later): the fitted GFN0-xTB
  parameters (global block and one ``$Z=`` block per element);
- ``xtb-srb/src/approxrab.f90`` (grimme-lab/xtb, LGPL-3.0-or-later): the short-range basis
  correction support data (``r0``, ``cnfak``, ``en``, ``p``);
- ``xtb-srb/src/disp/ncoord.f90`` (grimme-lab/xtb): the Pauling electronegativities of the SRB
  exponent;
- ``multicharge/src/multicharge/param/eeq2019.f90`` (grimme-lab/multicharge, Apache-2.0): the
  EEQ-2019 parameters of the charges entering D4.

The files are read as text (never executed); the SHA-256 of each is written into the output.
Values are written as in the files (eV where the file has eV); mopt-rs converts them as xtb
and gfn0-rs do.

This file is part of mopt-rs.
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

import hashlib
import re
import sys
from pathlib import Path

# Every element of `param_gfn0-xtb.txt` (`$Z=1` to `$Z=86`, xtb's `maxElem = 86`: H to Rn).
ELEMENTS = list(range(1, 87))
SYMBOL = {
    z + 1: s
    for z, s in enumerate(
        "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge "
        "As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm "
        "Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn".split()
    )
}
GLOBALS_REQUIRED = [
    "ks", "kp", "kd", "kdiff", "ens", "enp", "end", "enscale4", "ipeashift", "srbshift",
    "srbpre", "srbexp", "srbken", "a1", "a2", "s8", "s9", "kexp", "kexplight", "renscale",
]  # fmt: skip
GLOBALS_OPTIONAL = {
    "kf": 0.0,
    "enf": 0.0,
    "en_scale_prefactor": 0.005,
    "s6": 1.0,
    "eeq_jscale": 1.0,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rust_f64(v: float) -> str:
    r = repr(float(v))
    return r if ("." in r or "e" in r or "inf" in r or "nan" in r) else r + ".0"


def named_block(lines: list[str], name: str) -> dict[str, float]:
    start = next(i for i, line in enumerate(lines) if line.strip().lower() == name)
    out: dict[str, float] = {}
    for line in lines[start + 1 :]:
        t = line.strip()
        if t.lower() == "$end":
            return out
        if not t or t.startswith("#"):
            continue
        key, value = t.split()[:2]
        out[key.lower()] = float(value)
    raise SystemExit(f"unterminated {name}")


def element_blocks(lines: list[str]) -> dict[int, dict[str, list[str]]]:
    out: dict[int, dict[str, list[str]]] = {}
    i = 0
    while i < len(lines):
        t = lines[i].strip()
        if t.startswith("$Z="):
            z = int(t[3:].split()[0])
            values: dict[str, list[str]] = {}
            i += 1
            while lines[i].strip().lower() != "$end":
                fields = lines[i].strip().replace("=", " ").split()
                if fields:
                    values[fields[0].upper()] = fields[1:]
                i += 1
            out[z] = values
        i += 1
    return out


def pair_block(lines: list[str]) -> list[tuple[int, int, float]]:
    start = next(i for i, line in enumerate(lines) if line.strip().lower() == "$pairpar")
    out = []
    for line in lines[start + 1 :]:
        t = line.strip()
        if t.lower() == "$end":
            return out
        if t and not t.startswith("#"):
            a, b, s = t.split()[:3]
            out.append((int(a), int(b), float(s)))
    raise SystemExit("unterminated $pairpar")


def parse_ao(text: str) -> list[tuple[int, int]]:
    out = []
    for n, letter in re.findall(r"(\d+)([spdf])", text.lower()):
        out.append((int(n), "spdf".index(letter)))
    return out


def fortran_numbers(text: str) -> list[float]:
    text = " ".join(line.split("!")[0] for line in text.splitlines())
    tokens = re.split(r"[^0-9eEdD+\-.]+", text)
    return [float(t.replace("d", "e").replace("D", "e")) for t in tokens if re.search(r"\d", t)]


def data_array(text: str, name: str) -> list[float]:
    m = re.search(rf"data\s+{name}\s*/", text, re.IGNORECASE)
    if m is None:
        raise SystemExit(f"missing data {name}")
    rest = text[m.end() :]
    return fortran_numbers(rest[: rest.index("/")])


def parameter_array(text: str, name: str, opener: str, closer: str) -> list[float]:
    lower = text.lower()
    start = lower.index(f":: {name}(")
    eq = lower.index("=", start)
    begin = lower.index(opener, eq) + len(opener)
    return fortran_numbers(text[begin : lower.index(closer, begin)])


def main() -> None:
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    param = root / "xtb/param_gfn0-xtb.txt"
    approx = root / "xtb-srb/src/approxrab.f90"
    ncoord = root / "xtb-srb/src/disp/ncoord.f90"
    eeq = root / "multicharge/src/multicharge/param/eeq2019.f90"
    lines = param.read_text(encoding="utf-8").splitlines()
    glob = named_block(lines, "$globpar")
    blocks = element_blocks(lines)
    pairs = [(a, b, s) for a, b, s in pair_block(lines) if a in ELEMENTS and b in ELEMENTS]

    atext = approx.read_text(encoding="utf-8", errors="replace")
    srb_en, srb_r0, srb_cnfak, srb_p = (data_array(atext, k) for k in ("en", "r0", "cnfak", "p"))
    pauling = parameter_array(ncoord.read_text(encoding="utf-8"), "en", "(/", "/)")
    etext = eeq.read_text(encoding="utf-8")
    chi, eta, kcn, rad = (
        parameter_array(etext, k, "[", "]") for k in ("eeq_chi", "eeq_eta", "eeq_kcnchi", "eeq_rad")
    )

    o = [
        "//! GFN0-xTB parameters for H to Rn.",
        "//!",
        "//! Generated by tools/params/gen_gfn0_xtb.py from the upstream files that gfn0-rs",
        "//! (github.com/ss0832/gfn0-rs) pins; do not edit. Sources and SHA-256:",
        f"//! - grimme-lab/xtb `param_gfn0-xtb.txt` (LGPL-3.0-or-later): {sha256(param)}",
        f"//! - grimme-lab/xtb `src/approxrab.f90` (LGPL-3.0-or-later): {sha256(approx)}",
        f"//! - grimme-lab/xtb `src/disp/ncoord.f90` (LGPL-3.0-or-later): {sha256(ncoord)}",
        f"//! - grimme-lab/multicharge `param/eeq2019.f90` (Apache-2.0): {sha256(eeq)}",
        "//!",
        "//! Method: P. Pracht, E. Caldeweyher, S. Ehlert, S. Grimme, ChemRxiv 2019,",
        "//! DOI 10.26434/chemrxiv.8326202.v1. Energies in eV where the file has eV.",
        "//!",
        "//! MultiOptPy: no counterpart (new in mopt-rs); MultiOptPy's `Parameters/gfn0xtb_param.py` is in",
        "//!  `multioptpy_gfn`.",
        "//!",
        "//! This file is part of mopt-rs.",
        "//! Copyright (C) 2023-2026 ss0832",
        "//! SPDX-License-Identifier: GPL-3.0-or-later",
        "",
        "/// One shell of an element block (`ao`, `lev`, `exp`, `KCN*`, `KQ*`, `POLY*`).",
        "#[derive(Debug, Clone, Copy, PartialEq)]",
        "pub struct Gfn0ShellRaw {",
        "    /// Principal quantum number.",
        "    pub n: u8,",
        "    /// Angular momentum.",
        "    pub l: u8,",
        "    /// Level (eV).",
        "    pub level_ev: f64,",
        "    /// Slater exponent.",
        "    pub exponent: f64,",
        "    /// CN shift as in the file (`KCN*`, eV; xtb scales it by 0.1).",
        "    pub kcn_raw: f64,",
        "    /// Charge shift (`KQ*`, eV).",
        "    pub kq_ev: f64,",
        "    /// Shell polynomial (`POLY*`).",
        "    pub poly: f64,",
        "}",
        "",
        "/// An element block of `param_gfn0-xtb.txt` (defaults as in xtb's reader).",
        "#[derive(Debug, Clone, Copy, PartialEq)]",
        "pub struct Gfn0ElementRaw {",
        "    /// Atomic number.",
        "    pub z: u8,",
        "    /// Shells in file order.",
        "    pub shells: &'static [Gfn0ShellRaw],",
        "    /// `EN`.",
        "    pub en: f64,",
        "    /// `GAM` (EEQ hardness).",
        "    pub gam: f64,",
        "    /// `KQAT2` (eV).",
        "    pub kqat2_ev: f64,",
        "    /// `REPA`, `REPB`.",
        "    pub repa: f64,",
        "    /// `REPB`.",
        "    pub repb: f64,",
        "    /// `XI` (EEQ electronegativity).",
        "    pub xi: f64,",
        "    /// `KAPPA` (EEQ CN coefficient).",
        "    pub kappa: f64,",
        "    /// `ALPG` (EEQ charge width).",
        "    pub alpg: f64,",
        "    /// SRB `r0`, `cnfak`, `en` (approxrab.f90) and the Pauling `en` (ncoord.f90).",
        "    pub srb: [f64; 4],",
        "    /// EEQ-2019 `chi`, `eta`, `kcnchi`, `rad` (multicharge).",
        "    pub eeq2019: [f64; 4],",
        "}",
        "",
    ]
    for k in GLOBALS_REQUIRED:
        if k not in glob:
            raise SystemExit(f"missing global {k}")
    for k in GLOBALS_REQUIRED + list(GLOBALS_OPTIONAL):
        v = glob.get(k, GLOBALS_OPTIONAL.get(k))
        origin = "`$globpar`" if k in glob else "xtb's default (not in `$globpar`)"
        o.append(f"/// `{k}`: {origin}.")
        o.append(f"pub const GFN0_{k.upper()}: f64 = {rust_f64(v)};")
    o.append("")
    o.append("/// SRB `p(4, 2)` by period row (Fortran column order).")
    o.append("pub const GFN0_SRB_P: [[f64; 2]; 4] = [")
    o += [f"    [{rust_f64(srb_p[r])}, {rust_f64(srb_p[r + 4])}]," for r in range(4)]
    o.append("];")
    o.append("")
    o.append("/// `$pairpar` entries among these elements (`(Z_A, Z_B, scale)`, `Z_A <= Z_B`).")
    o.append(f"pub const GFN0_PAIR_SCALES: [(u8, u8, f64); {len(pairs)}] = [")
    o += [f"    ({min(a, b)}, {max(a, b)}, {rust_f64(s)})," for a, b, s in pairs]
    o.append("];")
    o.append("")
    o.append("/// The element blocks.")
    o.append(f"pub static GFN0_ELEMENTS: [Gfn0ElementRaw; {len(ELEMENTS)}] = [")
    for z in ELEMENTS:
        v = blocks[z]
        ao = parse_ao(v["AO"][0])
        lev = [float(x) for x in v["LEV"]]
        exp = [float(x) for x in v["EXP"]]

        def scalar(key: str, default: float, v: dict = v) -> float:
            return float(v[key][0]) if key in v else default

        en = scalar("EN", pauling[z - 1])
        o.append(f"    // {SYMBOL[z]}")
        o.append(f"    Gfn0ElementRaw {{ z: {z}, shells: &[")
        for i, (n, ang) in enumerate(ao):
            s = "SPDF"[ang]
            o.append(
                f"        Gfn0ShellRaw {{ n: {n}, l: {ang}, level_ev: {rust_f64(lev[i])}, "
                f"exponent: {rust_f64(exp[i])}, kcn_raw: {rust_f64(scalar('KCN' + s, 0.0))}, "
                f"kq_ev: {rust_f64(scalar('KQ' + s, 0.0))}, poly: {rust_f64(scalar('POLY' + s, 0.0))} }},"
            )
        srb = [srb_r0[z - 1], srb_cnfak[z - 1], srb_en[z - 1], pauling[z - 1]]
        e19 = [chi[z - 1], eta[z - 1], kcn[z - 1], rad[z - 1]]
        o.append(
            f"    ], en: {rust_f64(en)}, gam: {rust_f64(scalar('GAM', 0.1))}, "
            f"kqat2_ev: {rust_f64(scalar('KQAT2', 0.0))}, repa: {rust_f64(scalar('REPA', 1.0))}, "
            f"repb: {rust_f64(scalar('REPB', 2.0))}, xi: {rust_f64(scalar('XI', en))}, "
            f"kappa: {rust_f64(scalar('KAPPA', 0.0))}, alpg: {rust_f64(scalar('ALPG', 1.0))},"
        )
        o.append(
            f"      srb: [{', '.join(rust_f64(x) for x in srb)}], "
            f"eeq2019: [{', '.join(rust_f64(x) for x in e19)}] }},"
        )
    o.append("];")
    out.write_text("\n".join(o) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
