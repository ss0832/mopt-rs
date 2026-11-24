"""Generate the GFN1-xTB element table of mopt-rs (H to Rn, every element of GFN1-xTB).

Usage:

    python -I tools/params/gen_gfn1_xtb.py <xtb source tree> <output .rs> [<tblite gfn1.toml>]

The values are read as text (never executed) from grimme-lab/xtb (LGPL-3.0-or-later):

- ``src/xtb/gfn1.f90``: shells (``nShell``, ``angShell``, ``principalQuantumNumber``),
  ``referenceOcc``, ``shellPoly``, ``selfEnergy``, ``slaterExponent``, ``shellHardness``,
  ``chemicalHardness``, ``thirdOrderAtom``, ``repAlpha``, ``repZeff``, ``halogenBond``, and the
  element kinds ``gfn1Kinds`` (whether the self energies shift with the coordination number);
- ``src/param/paulingen.f90``, ``src/param/atomicrad.f90``, ``src/param/covalentradd3.f90``:
  Pauling electronegativities, atomic radii and D3 covalent radii (Angstrom).

The valence shells are the first shell of each angular momentum of an element
(``generateValenceShellData``), and the numbers of primitives follow
``setGFN1NumberOfPrimitives``. ``thirdOrderAtom`` and ``halogenBond`` are written times 0.1, as
xtb's parameter reader (``gam3``, ``cxb`` of ``read_gfn_param.f90``) and tblite take them.

With a third argument, the parameter file that tblite exports for GFN1-xTB
(``tblite.library.export_gfn1_param`` and ``dump_param``), every value written is compared
with tblite's and the script fails on a difference: levels, Slater exponents, primitives,
reference occupations, shell polynomials, CN shifts, hardnesses and their shell scaling,
third-order hardness, repulsion, halogen bond, electronegativity; and tblite's pair parameters
(``kpair``) with the rule of ``pair_scaling`` in ``pes/native/gfn1/params.rs``. The file can be
written by the tblite program (``tblite param --method gfn1 --output gfn1.toml``).

The SHA-256 of each file is written into the output.

This file is part of mopt-rs.
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

import hashlib
import re
import sys
import tomllib
from pathlib import Path

MAX_ELEM = 86
# Every element GFN1-xTB has parameters for (``maxElem = 86`` of gfn1.f90: H to Rn).
ELEMENTS = list(range(1, MAX_ELEM + 1))
SYMBOLS = (
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As "
    "Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu "
    "Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn"
).split()
# cnshell(kind, l) of gfn1.f90: kind 1 for the main group, kind 2 for Sc to Cr; kind 0 has no
# shift.
CN_SHELL = {1: [0.6, -0.3, -0.5, -0.5], 2: [0.6, -0.3, 0.5, 0.5]}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def strip_comments(text: str) -> str:
    return "\n".join(line.split("!")[0] for line in text.splitlines())


def array(src: str, name: str) -> list[float]:
    """The numbers of the array constant ``name`` (between its brackets)."""
    m = re.search(r"parameter\s*::\s*" + re.escape(name) + r"\s*\(", src)
    if m is None:
        raise SystemExit(f"{name} not found")
    start = src.index("[", m.end())
    end = src.index("]", start)
    body = strip_comments(src[start + 1 : end]).replace("&", " ")
    out = []
    for tok in re.findall(r"[-+]?\d+(?:\.\d*)?(?:[eEdD][-+]?\d+)?(?:_wp)?", body):
        out.append(float(tok.replace("_wp", "").replace("d", "e").replace("D", "e")))
    return out


def rust_f64(v: float) -> str:
    r = repr(float(v))
    return r if ("." in r or "e" in r) else r + ".0"


def main() -> None:
    if len(sys.argv) not in (3, 4):
        raise SystemExit(__doc__)
    root = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    files = {
        "gfn1": root / "src/xtb/gfn1.f90",
        "en": root / "src/param/paulingen.f90",
        "rad": root / "src/param/atomicrad.f90",
        "cov": root / "src/param/covalentradd3.f90",
    }
    text = {k: p.read_text() for k, p in files.items()}
    g = text["gfn1"]
    n_shell = [int(v) for v in array(g, "nShell")]
    ang = [int(v) for v in array(g, "angShell")]
    pqn = [int(v) for v in array(g, "principalQuantumNumber")]
    ref_occ = array(g, "referenceOcc")
    poly = array(g, "shellPoly")
    level = array(g, "selfEnergy")
    zeta = array(g, "slaterExponent")
    kappa = array(g, "shellHardness")
    eta = array(g, "chemicalHardness")
    gam3 = array(g, "thirdOrderAtom")
    rep_a = array(g, "repAlpha")
    rep_z = array(g, "repZeff")
    xb = array(g, "halogenBond")
    kinds = [int(v) for v in array(g, "gfn1Kinds")]
    en = array(text["en"], "paulingEN")
    rad = array(text["rad"], "atomicRad")
    cov = array(text["cov"], "covalentRadD3")
    for name, values, size in [
        ("nShell", n_shell, MAX_ELEM),
        ("angShell", ang, 3 * MAX_ELEM),
        ("referenceOcc", ref_occ, 3 * MAX_ELEM),
        ("shellPoly", poly, 4 * MAX_ELEM),
        ("selfEnergy", level, 3 * MAX_ELEM),
        ("slaterExponent", zeta, 3 * MAX_ELEM),
        ("shellHardness", kappa, 3 * MAX_ELEM),
        ("chemicalHardness", eta, MAX_ELEM),
        ("gfn1Kinds", kinds, 118),
        ("paulingEN", en, 118),
        ("atomicRad", rad, 118),
        ("covalentRadD3", cov, 118),
    ]:
        if len(values) != size:
            raise SystemExit(f"{name}: {len(values)} values, expected {size}")

    elements = []
    for z in ELEMENTS:
        i = z - 1
        shells = []
        seen = set()
        for k in range(n_shell[i]):
            l = ang[3 * i + k]
            n = pqn[3 * i + k]
            valence = l not in seen
            seen.add(l)
            if z <= 2:
                nprim = (4 if valence else 3) if l == 0 else 3
            elif l == 0:
                nprim = 6 if valence or n > 5 else 3
            else:
                nprim = 6 if l == 1 else 4
            occ = ref_occ[3 * i + l] if valence else 0.0
            shells.append(
                {
                    "l": l,
                    "n": n,
                    "zeta": zeta[3 * i + k],
                    "nprim": nprim,
                    "valence": valence,
                    "level": level[3 * i + k],
                    # xtb takes the polynomial of a valence shell by its angular momentum
                    # (`shellPoly(l + 1, Z)`, so 3d 4s 4p of the d-block read d s p), and
                    # tblite that of the diffuse hydrogen 2s by its place.
                    "poly": poly[4 * i + (l if valence else k)],
                    "kappa": kappa[3 * i + l],
                    "occ": occ,
                }
            )
        elements.append(
            {
                "z": z,
                "symbol": SYMBOLS[i],
                "shells": shells,
                "eta": eta[i],
                "gamma3": 0.1 * gam3[i],
                "rep_alpha": rep_a[i],
                "rep_zeff": rep_z[i],
                "xb": 0.1 * xb[i],
                "en": en[i],
                "rad": rad[i],
                "cov": cov[i],
                "kind": kinds[i],
            }
        )

    if len(sys.argv) == 4:
        check(elements, Path(sys.argv[3]))

    lines = []
    lines.append(
        "//! The GFN1-xTB element table (H to Rn), generated by\n"
        "//! `tools/params/gen_gfn1_xtb.py` from grimme-lab/xtb (LGPL-3.0-or-later), read as text,\n"
        "//! and checked there value by value against the parameters tblite exports. Do not edit\n"
        "//! by hand.\n//!\n"
    )
    for key, p in files.items():
        rel = p.relative_to(root).as_posix()
        lines.append(f"//! - `{rel}`: SHA-256 `{sha256(p)}`\n")
    lines.append(
        "//!\n//! MultiOptPy: no counterpart (new in mopt-rs).\n//!\n"
        "//! This file is part of mopt-rs.\n"
        "//! Copyright (C) 2023-2026 ss0832\n"
        "//! SPDX-License-Identifier: GPL-3.0-or-later\n\n"
        "use super::params::{sh, Gfn1Element, Gfn1Shell};\n\n"
    )
    for e in elements:
        shells = ",\n".join(
            "    sh({l}, {n}, {zeta}, {nprim}, {valence}, {level}, {poly}, {kappa}, {occ})".format(
                l=s["l"],
                n=s["n"],
                zeta=rust_f64(s["zeta"]),
                nprim=s["nprim"],
                valence="true" if s["valence"] else "false",
                level=rust_f64(s["level"]),
                poly=rust_f64(s["poly"]),
                kappa=rust_f64(s["kappa"]),
                occ=rust_f64(s["occ"]),
            )
            for s in e["shells"]
        )
        name = e["symbol"].upper()
        # `static`, not `const`: a large constant table is copied into every use and costs
        # the compiler far more memory.
        lines.append(f"static {name}: [Gfn1Shell; {len(e['shells'])}] = [\n{shells},\n];\n")
    lines.append("\n/// The elements, by atomic number (H to Rn).\n")
    lines.append(f"pub static ELEMENTS: [Gfn1Element; {len(elements)}] = [\n")
    for e in elements:
        lines.append(
            "    Gfn1Element {{\n"
            "        z: {z},\n"
            "        shells: &{name},\n"
            "        eta: {eta},\n"
            "        gamma3: {gamma3},\n"
            "        rep_alpha: {rep_alpha},\n"
            "        rep_zeff: {rep_zeff},\n"
            "        xb_strength: {xb},\n"
            "        en: {en},\n"
            "        atomic_rad: {rad},\n"
            "        cov_rad_d3: {cov},\n"
            "        cn_kind: {cn},\n"
            "    }},\n".format(
                z=e["z"],
                name=e["symbol"].upper(),
                eta=rust_f64(e["eta"]),
                gamma3=rust_f64(round(e["gamma3"], 12)),
                rep_alpha=rust_f64(e["rep_alpha"]),
                rep_zeff=rust_f64(e["rep_zeff"]),
                xb=rust_f64(round(e["xb"], 12)),
                en=rust_f64(e["en"]),
                rad=rust_f64(e["rad"]),
                cov=rust_f64(e["cov"]),
                cn=e["kind"],
            )
        )
    lines.append("];\n")
    out_path.write_text("".join(lines), encoding="utf-8", newline="\n")
    print(f"wrote {out_path} ({len(elements)} elements)")


def check(elements: list[dict], toml_path: Path) -> None:
    data = tomllib.loads(toml_path.read_text())
    bad = []

    def same(what: str, a: float, b: float) -> None:
        if abs(a - b) > 1e-9 * max(1.0, abs(a), abs(b)):
            bad.append(f"{what}: xtb {a!r}, tblite {b!r}")

    for e in elements:
        t = data["element"][e["symbol"]]
        s = e["shells"]
        if len(t["shells"]) != len(s):
            bad.append(f"{e['symbol']}: shells {t['shells']}")
            continue
        letters = "spdf"
        for k, sh in enumerate(s):
            tag = f"{e['symbol']} {t['shells'][k]}"
            if t["shells"][k] != f"{sh['n']}{letters[sh['l']]}":
                bad.append(f"{tag}: shell {sh['n']}{letters[sh['l']]}")
            same(f"{tag} level", sh["level"], t["levels"][k])
            same(f"{tag} slater", sh["zeta"], t["slater"][k])
            same(f"{tag} ngauss", sh["nprim"], t["ngauss"][k])
            same(f"{tag} refocc", sh["occ"], t["refocc"][k])
            same(f"{tag} shpoly", 0.01 * sh["poly"], t["shpoly"][k])
            kind = e["kind"]
            kcn = -sh["level"] * CN_SHELL[kind][sh["l"]] * 0.01 if kind > 0 else 0.0
            same(f"{tag} kcn", kcn, t["kcn"][k])
            same(f"{tag} lgam", 1.0 + sh["kappa"], t["lgam"][k])
        same(f"{e['symbol']} gam", e["eta"], t["gam"])
        same(f"{e['symbol']} gam3", e["gamma3"], t["gam3"])
        same(f"{e['symbol']} zeff", e["rep_zeff"], t["zeff"])
        same(f"{e['symbol']} arep", e["rep_alpha"], t["arep"])
        same(f"{e['symbol']} xbond", e["xb"], t["xbond"])
        same(f"{e['symbol']} en", e["en"], t["en"])
    # The pair parameters: tblite lists every pair that is not 1; `pair_scaling` of
    # `pes/native/gfn1/params.rs` implements this rule.
    symbol_z = {s: k + 1 for k, s in enumerate(SYMBOLS)}
    expected = {}
    for za in range(1, MAX_ELEM + 1):
        for zb in range(1, za + 1):
            s = pair_rule(za, zb)
            if s != 1.0:
                expected[(za, zb)] = s
    listed = {}
    for key, value in data["hamiltonian"]["xtb"]["kpair"].items():
        a, b = (symbol_z[s] for s in key.split("-"))
        listed[(max(a, b), min(a, b))] = value
    for pair in sorted(set(expected) | set(listed)):
        same(f"kpair {pair}", expected.get(pair, 1.0), listed.get(pair, 1.0))
    if bad:
        raise SystemExit("differences from tblite:\n" + "\n".join(bad))
    print(f"checked {len(elements)} elements and {len(listed)} pair parameters against "
          f"{toml_path.name}")


# The element pairs of gfn1.f90 (`initHamiltonian`) with their own parameter.
SPECIAL_PAIRS = {
    (1, 1): 0.96, (5, 1): 0.95, (7, 1): 1.04, (28, 1): 0.90, (75, 1): 0.80, (78, 1): 0.80,
    (15, 5): 0.97, (14, 7): 1.01,
}  # fmt: skip


def pair_rule(za: int, zb: int) -> float:
    """``kpair`` of ``Z_A >= Z_B`` as gfn1.f90 sets it (``setGFN1PairParam``, then the pairs)."""
    if (za, zb) in SPECIAL_PAIRS:
        return SPECIAL_PAIRS[(za, zb)]

    def row(z: int) -> int:
        return 1 if 20 < z < 30 else 2 if 38 < z < 48 else 3 if 56 < z < 80 else 0

    kp = [1.1, 1.2, 1.2]
    if row(za) and row(zb):
        return 0.5 * (kp[row(za) - 1] + kp[row(zb) - 1])
    return 1.0


if __name__ == "__main__":
    main()
