"""Reference data of the improved elastic image pair method: ieipmain of MultiOptPy on LJ.

Usage (in the reference environment):

    python tools/golden/run_ieip_reference.py <multioptpy_root> <output_dir> <work_dir> [<case> ...]

Each case writes ``ieip_<case>.json`` (Angstrom as MultiOptPy writes its structures,
Hartree): the two structures of every iteration as MultiOptPy evaluated them, the numbers
its ``print_info`` reports (energies, biased energies, distance of the images after the
step), the value of ``BETA`` in every iteration, and the files it writes at the end.

The two input structures are laid onto each other (both centred, the first rotated onto
the second) before they are written, so that the rotation of the first iteration is the
identity: MultiOptPy turns the first structure onto the second without turning its
gradient, which mopt-rs does (see the migration notes).

MultiOptPy lists the input files with ``glob.glob``, whose order is that of the directory;
the script sorts them, which is the order mopt-rs uses (by the letter of the name).

This file is part of mopt-rs.
"""

import contextlib
import glob
import io
import json
import math
import os
import shutil
import sys
from pathlib import Path

import numpy as np

CASES = {
    "lj_ar7_default": ["-ns", "60"],
    "lj_ar7_beta": ["-ns", "40", "-beta", "2.5"],
    # A bias potential: MultiOptPy takes the energies without it to decide which image is
    # higher, and the gradient without it in the scale of the attraction of the images.
    "lj_ar7_bias": ["-ns", "40", "-kp", "0.002", "4.0", "1,6"],
    # The relaxation on hyperspheres (-mi), which mopt-rs does differently (see the
    # migration notes): only the outcome is compared.
    "lj_ar7_micro": ["-ns", "100", "-mi", "3"],
    # The structures in the other order: the first is the higher one (the term of the energy
    # difference then acts the other way round).
    "lj_ar7_swapped": ["-ns", "100"],
}
# Cases whose first structure is the tilted one.
SWAPPED = {"lj_ar7_swapped"}


def centered(coords):
    c = np.asarray(coords, dtype=float)
    return c - c.mean(axis=0)


def bipyramid(a=3.8, tilt_deg=0.0):
    """Seven argon atoms: a pentagon and two caps, the caps tilted about the x axis."""
    r = a / (2.0 * math.sin(math.pi / 5.0))
    pos = []
    for k in range(5):
        t = k * 2.0 * math.pi / 5.0
        pos.append([r * math.cos(t) + 0.1 * math.sin(k), r * math.sin(t), 0.05 * k])
    c, s = math.cos(math.radians(tilt_deg)), math.sin(math.radians(tilt_deg))
    for x, y, z in ([0.0, 0.0, 0.55 * a], [0.2, 0.0, -0.53 * a]):
        pos.append([x, c * y - s * z, s * y + c * z])
    return centered(pos)


def kabsch(p, q):
    """``p`` turned onto ``q`` (both centred), as MultiOptPy's ``kabsch_algorithm``."""
    p = centered(p)
    q = centered(q)
    h = p.T @ q
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T @ u.T
    return (r @ p.T).T, q


def read_xyz(path):
    lines = Path(path).read_text().splitlines()[2:]
    rows = [line.split() for line in lines if line.strip()]
    return [[float(v) for v in row[1:4]] for row in rows]


def frames(path):
    out = []
    lines = Path(path).read_text().splitlines()
    k = 0
    while k < len(lines):
        if not lines[k].strip():
            k += 1
            continue
        n = int(lines[k])
        rows = [line.split() for line in lines[k + 2 : k + 2 + n]]
        out.append([[float(v) for v in row[1:4]] for row in rows])
        k += n + 2
    return out


def numbers(path):
    if not os.path.exists(path):
        return None
    return [int(line) for line in Path(path).read_text().split()]


def run_case(out_dir, work_dir, name, options):
    import multioptpy.ieip as ieip
    import multioptpy.interface as interface
    from multioptpy.OtherMethod.elastic_image_pair import ElasticImagePair

    reports = []
    betas = []
    original_print = ElasticImagePair.print_info

    def recording_print(self, dat):
        reports.append(
            {
                "energy": [float(dat["energy_1"]), float(dat["energy_2"])],
                "biased_energy": [float(dat["bias_energy_1"]), float(dat["bias_energy_2"])],
                "separation_after": float(dat["delta_geometry"]),
                "step_norm": [
                    float(np.linalg.norm(dat["total_disp_1"])),
                    float(np.linalg.norm(dat["total_disp_2"])),
                ],
            }
        )
        betas.append(float(self.config.BETA))
        return original_print(self, dat)

    ElasticImagePair.print_info = recording_print
    original_glob = ieip.glob.glob

    case_dir = os.path.join(work_dir, name)
    shutil.rmtree(case_dir, ignore_errors=True)
    os.makedirs(os.path.join(case_dir, "ar7"))
    if name in SWAPPED:
        first, second = kabsch(bipyramid(tilt_deg=25.0), bipyramid())
    else:
        first, second = kabsch(bipyramid(), bipyramid(tilt_deg=25.0))
    for letter, frame in (("A", first), ("B", second)):
        with open(os.path.join(case_dir, "ar7", f"ar7_{letter}.xyz"), "w") as f:
            f.write("7\n0 1\n")
            for x, y, z in frame:
                f.write(f"Ar {x:.12f} {y:.12f} {z:.12f}\n")
    old_argv, old_cwd = sys.argv, os.getcwd()
    try:
        os.chdir(case_dir)
        ieip.glob.glob = lambda pattern: sorted(original_glob(pattern))
        sys.argv = ["ieipmain", "ar7", "-os", "lj", *options]
        parser = interface.init_parser()
        args = interface.ieipparser(parser)
        with contextlib.redirect_stdout(io.StringIO()):
            ieip.iEIP(args).run()
    finally:
        sys.argv = old_argv
        os.chdir(old_cwd)
        ieip.glob.glob = original_glob
        ElasticImagePair.print_info = original_print
    run = sorted(glob.glob(os.path.join(case_dir, "ar7_iEIP_*")))[-1]

    def structures(letter):
        dirs = glob.glob(os.path.join(run, f"samples_ar7_{letter}_*"))
        dirs.sort(key=lambda p: int(p.rsplit("_", 1)[1]))
        return [read_xyz(os.path.join(d, f"ar7_{letter}_0.xyz")) for d in dirs]

    data = {
        "meta": {"multioptpy": "1.21.2", "numpy": np.__version__},
        "calculator": "lj",
        "symbols": ["Ar"] * 7,
        "arguments": options,
        "input_angstrom": [first.tolist(), second.tolist()],
        # images[k] = [first, second] as written for iteration k (k = 0: the inputs); the
        # last entry is the pair after the last step, which was not evaluated.
        "images_angstrom": [list(p) for p in zip(structures("A"), structures("B"), strict=True)],
        "reports": reports,
        "beta": betas,
        "converged": not os.path.exists(os.path.join(run, "not_converged.txt")),
        "trajectory_angstrom": frames(os.path.join(run, "ar7_A_traj.xyz")),
        "approx_ts": numbers(os.path.join(run, "approx_TS.txt")),
        "approx_eq": numbers(os.path.join(run, "approx_EQ.txt")),
        "local_min_grad": numbers(os.path.join(run, "local_min_grad.txt")),
        "files": sorted(p for p in os.listdir(run) if not p.startswith("samples_")),
    }
    out = os.path.join(out_dir, f"ieip_{name}.json")
    with open(out, "w") as f:
        json.dump(data, f, indent=1)
    print(
        f"wrote {out}: {len(reports)} iterations, converged {data['converged']}",
        file=sys.stderr,
    )


def main():
    root, out_dir, work_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    only = sys.argv[4:]
    unknown = set(only) - set(CASES)
    if unknown:
        raise SystemExit(f"unknown cases: {sorted(unknown)}")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(work_dir, exist_ok=True)
    os.environ.setdefault("MPLBACKEND", "Agg")
    sys.path.insert(0, root)
    for name, options in CASES.items():
        if not only or name in only:
            run_case(out_dir, work_dir, name, options)


if __name__ == "__main__":
    main()
