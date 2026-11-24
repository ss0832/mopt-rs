"""Reference data of the path methods: nebmain of MultiOptPy on a Lennard-Jones path.

Usage (in the reference environment):

    python tools/golden/run_neb_reference.py <multioptpy_root> <output_dir> <work_dir> [<case> ...]

Each case writes ``neb_<case>.json`` (Angstrom as MultiOptPy writes its images, Hartree):
the chain of every iteration, its energies, and the per-image numbers MultiOptPy logs.

MultiOptPy puts the first structure of the input into the chain twice (images 0 and 1
coincide) and reads the last frame of a file only when a line that is not an atom follows
it; the input written here ends with an empty line, and the chain is recorded as it is.

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
    "lj_ar7_default": ["-ns", "8", "-p", "3"],
    "lj_ar7_fixed_ends": ["-ns", "8", "-p", "3", "-fe", "3"],
    "lj_ar7_climbing": ["-ns", "8", "-p", "3", "-cineb", "3"],
    "lj_ar7_steepest": ["-ns", "8", "-p", "3", "-sd", "3"],
    "lj_ar7_equal_spacing": ["-ns", "8", "-p", "3", "-ad", "2"],
    # The way nebmain is used in the articles of the author: a trajectory as input, model
    # Hessians of every image (O1NumHess), images 0.5 Angstrom apart, and a redistribution
    # with the predicted energy every second iteration.
    "lj_ar7_article": ["-ns", "12", "-modelhess", "o1numhess", "-nd", "0.5", "-adpred", "2"],
    "lj_ar7_article_no_redistribution": ["-ns", "6", "-modelhess", "o1numhess", "-nd", "0.5"],
}

TRAJECTORY_CASES = ("lj_ar7_article", "lj_ar7_article_no_redistribution")


def size_one_float(value=0.0):
    """float() as NumPy before 2.0 did it: an array with one element is that element."""
    if isinstance(value, np.ndarray) and value.size == 1:
        return float(value.reshape(-1)[0])
    return float(value)


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


def read_images(directory, stem):
    files = [
        p
        for p in glob.glob(os.path.join(directory, f"{stem}_*.xyz"))
        if not p.endswith("_path.xyz")
    ]
    files.sort(key=lambda p: int(p.rsplit("_", 1)[1][:-4]))
    images = []
    for path in files:
        lines = Path(path).read_text().splitlines()[2:]
        rows = [line.split() for line in lines if line.strip()]
        images.append([[float(v) for v in row[1:4]] for row in rows])
    return images


def table(path):
    if not os.path.exists(path):
        return []
    return [[float(v) for v in line.strip().split(",") if v] for line in open(path) if line.strip()]


def run_case(root, out_dir, work_dir, name, options):
    import multioptpy.interface as interface
    import multioptpy.neb as neb

    # MultiOptPy 1.21.2 calls float() on arrays of one element with two dimensions (block
    # Hessian updates), which NumPy 2 refuses. Its modules get the float of older NumPy.
    import multioptpy.Optimizer.block_hessian_update as block_update

    block_update.float = size_one_float

    # The Hessians of the images as the optimizer loads them (before it adds the
    # curvature along the chain), for the first two iterations.
    import multioptpy.Optimizer.rfo_neb as rfo_neb

    loaded = []
    original_load = rfo_neb.OptimizationAlgorithm._load_or_init_hessian

    def recording_load(self, num, natoms, config):
        hessian = original_load(self, num, natoms, config)
        loaded.append(np.array(hessian, dtype=float).copy())
        return hessian

    rfo_neb.OptimizationAlgorithm._load_or_init_hessian = recording_load

    # The steps of the minimizers of the images before their length is limited.
    raw_steps = []
    original_limit = rfo_neb.OptimizationAlgorithm._limit_step_size

    def recording_limit(self, move_vec, is_endpoint):
        raw_steps.append(np.array(move_vec, dtype=float).reshape(-1).copy())
        return original_limit(self, move_vec, is_endpoint)

    rfo_neb.OptimizationAlgorithm._limit_step_size = recording_limit

    case_dir = os.path.join(work_dir, name)
    shutil.rmtree(case_dir, ignore_errors=True)
    os.makedirs(case_dir)
    if name in TRAJECTORY_CASES:
        frames = [bipyramid(tilt_deg=5.0 * k) for k in range(9)]
    else:
        frames = [bipyramid(), bipyramid(tilt_deg=25.0)]
    with open(os.path.join(case_dir, "ar7.xyz"), "w") as f:
        for frame in frames:
            f.write("7\n0 1\n")
            for x, y, z in frame:
                f.write(f"Ar {x:.12f} {y:.12f} {z:.12f}\n")
        f.write("\n")
    old_argv, old_cwd = sys.argv, os.getcwd()
    try:
        os.chdir(case_dir)
        sys.argv = ["nebmain", "ar7.xyz", "-os", "lj", *options]
        parser = interface.init_parser()
        args = interface.nebparser(parser)
        with contextlib.redirect_stdout(io.StringIO()):
            neb.NEB(args).run()
    finally:
        sys.argv = old_argv
        os.chdir(old_cwd)
        rfo_neb.OptimizationAlgorithm._load_or_init_hessian = original_load
        rfo_neb.OptimizationAlgorithm._limit_step_size = original_limit
    run = sorted(glob.glob(os.path.join(case_dir, "ar7_NEB_*")))[-1]
    steps = sorted(
        glob.glob(os.path.join(run, "path_ITR_*_ar7")),
        key=lambda p: int(os.path.basename(p).split("_")[2]),
    )
    data = {
        "meta": {"multioptpy": "1.21.2", "numpy": np.__version__},
        "calculator": "lj",
        "symbols": ["Ar"] * 7,
        "arguments": options,
        "input_frames_angstrom": [f.tolist() for f in frames],
        # chains[k] is the chain evaluated in iteration k; the last entry is the chain after
        # the last step, which was not evaluated.
        "chains_angstrom": [read_images(step, "ar7") for step in steps],
        "energies_hartree": table(os.path.join(run, "energy_plot.csv")),
        "path_length": table(os.path.join(run, "path_length.csv")),
        "orthogonality": table(os.path.join(run, "orthogonality.csv")),
        "perp_rms_gradient": table(os.path.join(run, "perp_rms_gradient.csv")),
        "perp_max_gradient": table(os.path.join(run, "perp_max_gradient.csv")),
        "bias_force_rms": table(os.path.join(run, "bias_force_rms.csv")),
        "procrustes_distance_1": table(os.path.join(run, "procrustes_distance_1.csv")),
        "procrustes_distance_2": table(os.path.join(run, "procrustes_distance_2.csv")),
    }
    images = len(data["chains_angstrom"][0])
    if loaded:
        # hessians[k][i]: image i in iteration k (Hartree/Bohr^2, row by row).
        data["hessians"] = [
            [loaded[k * images + i].tolist() for i in range(images)]
            for k in range(min(2, len(loaded) // images))
        ]
        # newton_steps[k][i]: what the minimizer of image i proposed in iteration k (the
        # vector MultiOptPy subtracts from the coordinates), for every iteration.
        data["newton_steps"] = [
            [raw_steps[k * images + i].tolist() for i in range(images)]
            for k in range(len(raw_steps) // images)
        ]
    out = os.path.join(out_dir, f"neb_{name}.json")
    with open(out, "w") as f:
        json.dump(data, f, indent=1)
    shapes = [len(chain) for chain in data["chains_angstrom"]]
    print(
        f"wrote {out}: {len(data['energies_hartree'])} iterations, images {shapes}", file=sys.stderr
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
            run_case(root, out_dir, work_dir, name, options)


if __name__ == "__main__":
    main()
