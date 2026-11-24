"""Golden data of MultiOptPy's RCMC for the solver ``legacy_rcmc`` of mopt-rs.

Usage (inside the reference environment created by setup_ref_env.sh):

    python -I tools/golden/rcmc_reference.py <multioptpy_root> <output_json>

``multioptpy/Utils/rcmc.py`` of the MultiOptPy tree is loaded from its file and run as it is
(read only; its import of the mapper falls back to the stub classes of the file, which are all
that ``main`` needs). Every case is a network in MultiOptPy's format with the options of
``run_rcmc``; ``main`` is called with them, and ``RCMCQueue._save_K_matrix`` is wrapped to keep
the Schur complement ``D`` in full precision besides the file it writes. One case calls
``RCMCQueue.pop`` directly without a start node, as MultiOptPy's mapper does without
``rcmc_start_node_id``. The populations are the priorities of the tasks after ``pop``.

The networks: the three models of section 1.4 of ``docs/source/kinetics_beyond_tst.md`` (a
start state with a lifetime of 100 s next to a fast pair, a start state decaying into ten
degenerate conformers, two states with a rate constant of 1e-9 1/s), a saddle point below one
of its minima (MultiOptPy's clamp), free energies with nodes and edges without one, a start
node that is eliminated, and random networks (seeded) at several times.

This file is part of mopt-rs.
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

import contextlib
import importlib.util
import io
import json
import math
import os
import platform
import random
import sys
import tempfile

import numpy as np
import scipy

K_B_J_K = 1.380649e-23
H_J_S = 6.62607015e-34
K_B_HARTREE = 3.166811563e-6


def load(root):
    path = os.path.join(root, "multioptpy", "Utils", "rcmc.py")
    spec = importlib.util.spec_from_file_location("multioptpy_rcmc_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def barrier(rate, temperature):
    """The energy difference (Hartree) of an Eyring rate constant `rate` (1/s)."""
    kt_h = K_B_J_K * temperature / H_J_S
    return K_B_HARTREE * temperature * math.log(kt_h / rate)


def network(energies, edges, free=None, ts_free=None):
    nodes = []
    for k, e in enumerate(energies):
        nodes.append(
            {
                "node_id": k,
                "xyz_file": "",
                "energy_hartree": e,
                "free_energy_hartree": None if free is None else free[k],
            }
        )
    out = []
    for k, (a, b, ts) in enumerate(edges):
        out.append(
            {
                "edge_id": k,
                "node_id_1": a,
                "node_id_2": b,
                "ts_xyz_file": None,
                "ts_energy_hartree": ts,
                "ts_free_energy_hartree": None if ts_free is None else ts_free[k],
            }
        )
    return {"nodes": nodes, "edges": out}


def models():
    t = 300.0
    rt = K_B_HARTREE * t
    cases = []
    # 1. A start state with a lifetime of 100 s next to a fast pair: A -> B at 0.01 1/s,
    # B <-> C at 1e6 1/s, B and C far below A.
    e = [0.0, -20 * rt, -20.5 * rt]
    edges = [(0, 1, barrier(0.01, t)), (1, 2, -20 * rt + barrier(1e6, t))]
    cases.append(("slow_start_fast_pair", network(e, edges), t, 1.0, 0, "electronic"))
    # 2. A start state decaying at 1 1/s into ten degenerate conformers in fast exchange.
    e = [0.0] + [-15 * rt] * 10
    edges = [(0, 1, barrier(1.0, t))]
    edges += [(k, k + 1, -15 * rt + barrier(1e5, t)) for k in range(1, 10)]
    cases.append(("ten_conformers", network(e, edges), t, 0.4, 0, "electronic"))
    # 3. Two states, 1e-9 1/s from A to B, B 2 RT below A.
    e = [0.0, -2 * rt]
    cases.append(("two_states", network(e, [(0, 1, barrier(1e-9, t))]), t, 1.0, 0, "electronic"))
    # A saddle point below its upper minimum (free energies of mixed levels).
    e = [0.0, 0.004, 0.010, -0.002]
    edges = [(0, 1, 0.012), (1, 2, 0.008), (2, 3, 0.020), (0, 3, 0.025)]
    cases.append(("ts_below_minimum", network(e, edges), t, 1e-6, 0, "electronic"))
    # Free energies, a node and an edge without one, a self-loop.
    e = [0.0, 0.003, 0.006, 0.002, 0.009]
    g = [0.0, 0.0025, None, 0.0012, 0.007]
    edges = [(0, 1, 0.02), (1, 2, 0.022), (1, 3, 0.019), (3, 4, 0.025), (0, 4, 0.03), (3, 3, 0.03)]
    tg = [0.019, 0.021, 0.018, None, 0.028, 0.029]
    cases.append(("free_energies", network(e, edges, g, tg), 350.0, 1e-3, 0, "free"))
    # A start node that is eliminated: a shallow, fast-decaying start.
    e = [0.008, 0.0, 0.001, 0.004]
    edges = [(0, 1, 0.0095), (0, 2, 0.0098), (1, 2, 0.016), (2, 3, 0.018), (1, 3, 0.024)]
    cases.append(("start_eliminated", network(e, edges), t, 1e-4, 0, "electronic"))
    # Random networks.
    rng = random.Random(20261117)
    for n, times in [(7, [1e-9, 1e-4, 1.0]), (12, [1e-6, 1e2])]:
        e = [rng.uniform(0.0, 0.02) for _ in range(n)]
        edges = []
        for i in range(1, n):
            j = rng.randrange(i)
            edges.append((i, j, max(e[i], e[j]) + rng.uniform(0.008, 0.035)))
            if rng.random() < 0.5 and i > 2:
                k = rng.randrange(i - 1)
                edges.append((i, k, max(e[i], e[k]) + rng.uniform(0.008, 0.035)))
        net = network(e, edges)
        for t_r in times:
            cases.append((f"random{n}_t{t_r:g}", net, t, t_r, 0, "electronic"))
    return cases


def run_main(module, name, net, temperature, time, start, energy):
    captured = {}
    original = module.RCMCQueue

    class Capturing(original):
        def _save_K_matrix(self, D, T_indices, superstate_members, nodes, pop_count):
            captured["D"] = [[float(x) for x in row] for row in D]
            captured["kept"] = [int(nodes[i].node_id) for i in T_indices]
            captured["members"] = [
                [int(nodes[m].node_id) for m in superstate_members.get(i, [i])] for i in T_indices
            ]
            super()._save_K_matrix(D, T_indices, superstate_members, nodes, pop_count)

        def pop(self):
            top = super().pop()
            captured["populations"] = {
                str(t.node_id): float(t.priority) for t in [top, *self._tasks.values()]
            }
            return top

    with tempfile.TemporaryDirectory() as work:
        path = os.path.join(work, "network.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(net, fh)
        out = os.path.join(work, "out")
        argv = [path, "-T", repr(temperature), "-t", repr(time), "--start-node", str(start)]
        argv += ["--output-dir", out, "--energy-type", energy]
        module.RCMCQueue = Capturing
        buffer = io.StringIO()
        try:
            with contextlib.redirect_stdout(buffer):
                module.main(argv)
        finally:
            module.RCMCQueue = original
        with open(os.path.join(out, "rcmc_K_contracted.csv"), encoding="utf-8") as fh:
            csv = fh.read()
    return {
        "name": name,
        "via": "main",
        "network": net,
        "temperature": temperature,
        "time": time,
        "start": start,
        "energy_type": energy,
        "csv": csv,
        "stdout": buffer.getvalue().replace(path, "network.json").replace(out, "out"),
        **captured,
    }


def run_queue_without_start(module, name, net, temperature, time):
    """RCMCQueue.pop as MultiOptPy's mapper calls it without rcmc_start_node_id."""

    class Node:
        def __init__(self, d):
            self.node_id = d["node_id"]
            self.energy = d["energy_hartree"]
            self.free_energy = d["free_energy_hartree"]
            self.has_real_energy = self.energy is not None

    class Edge:
        def __init__(self, d):
            self.node_id_1 = d["node_id_1"]
            self.node_id_2 = d["node_id_2"]
            self.ts_energy = d["ts_energy_hartree"]
            self.ts_free_energy = d["ts_free_energy_hartree"]

    class Graph:
        def __init__(self, nodes, edges):
            self.nodes, self.edges = nodes, edges

        def all_nodes(self):
            return self.nodes

        def all_edges(self):
            return self.edges

    nodes = [Node(d) for d in net["nodes"]]
    edges = [Edge(d) for d in net["edges"]]
    queue = module.RCMCQueue(temperature_K=temperature, reaction_time_s=time, start_node_id=None)
    queue.set_graph(Graph(nodes, edges))
    for nd in nodes:
        queue.push(module.ExplorationTask(node_id=nd.node_id, metadata={}))
    top = queue.pop()
    populations = {str(t.node_id): float(t.priority) for t in [top, *queue._tasks.values()]}
    return {
        "name": name,
        "via": "queue",
        "network": net,
        "temperature": temperature,
        "time": time,
        "start": None,
        "energy_type": "electronic",
        "populations": populations,
    }


def main():
    root, output = sys.argv[1], sys.argv[2]
    module = load(root)
    cases = [run_main(module, *case) for case in models()]
    # Without a start node: the population starts at the lowest node and the contracted
    # equation from a uniform vector.
    e = [0.004, 0.0, 0.006, 0.002]
    edges = [(0, 1, 0.014), (1, 2, 0.016), (2, 3, 0.015), (0, 3, 0.02)]
    cases.append(
        run_queue_without_start(module, "mapper_without_start", network(e, edges), 300.0, 1e-5)
    )
    doc = {
        "about": "MultiOptPy's Utils/rcmc.py run as it is on these networks (generated by "
        "tools/golden/rcmc_reference.py)",
        "multioptpy": "1.21.2",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "cases": cases,
    }
    with open(output, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(doc, fh, indent=1)
        fh.write("\n")
    print(f"wrote {len(cases)} cases to {output}")


if __name__ == "__main__":
    main()
