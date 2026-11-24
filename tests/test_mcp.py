"""The MCP server of the Python package (``python -m mopt_rs.mcp``) end to end, and its attacks.

The tools are built from ``python/mopt_rs/mcp/commands.json``; a command implemented in the
tables is a tool without a change of the server. A client drives the server over stdio: the
handshake, the tools, a single point, an ``optmain`` job with the built-in ``lj`` calculator
followed to its end and read. Then the attacks: traversal with ``..``, absolute paths,
symbolic links that lead out, encoded separators, job ids that are paths, option values that
name files outside the root, options that start other programs or load code, a copied file that
would stand in for a module, malformed messages. With ``MOPT_BIN`` set to the Rust binary, the
tools of the commands are compared with those of ``mopt mcp``.

This file is part of mopt-py (the Python interface of mopt-rs).
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("mopt_rs._core", reason="the compiled package mopt_rs is not importable")

from mopt_rs.mcp import tables  # noqa: E402
from mopt_rs.mcp.paths import refused_path_text, refused_value_part  # noqa: E402

AR7 = """7
argon cluster
Ar   1.9500   0.0000   0.0000
Ar   0.6026   1.8546   0.0300
Ar  -1.5776   1.1462   0.0700
Ar  -1.5776  -1.1462   0.1100
Ar   0.6026  -1.8546   0.1500
Ar   0.0000   0.0000   2.1000
Ar   0.2000   0.0000  -2.0000
"""


class Client:
    """``python -m mopt_rs.mcp`` as a child process."""

    def __init__(self, root: Path, cwd: Path, *extra: str) -> None:
        self.process = subprocess.Popen(
            [sys.executable, "-P", "-m", "mopt_rs.mcp", "--root", str(root), *extra],
            cwd=cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
        )
        self.next_id = 0

    def send(self, line: str | bytes) -> None:
        assert self.process.stdin is not None
        data = line if isinstance(line, bytes) else line.encode("utf-8")
        self.process.stdin.write(data + b"\n")
        self.process.stdin.flush()

    def read(self) -> dict[str, Any]:
        assert self.process.stdout is not None
        line = self.process.stdout.readline()
        assert line.endswith(b"\n"), "no response"
        message: dict[str, Any] = json.loads(line)
        return message

    def request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        self.next_id += 1
        self.send(
            json.dumps({"jsonrpc": "2.0", "id": self.next_id, "method": method, "params": params})
        )
        response = self.read()
        assert response.get("id") == self.next_id, response
        return response

    def result(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        response = self.request(method, params)
        assert "result" in response, response
        result: dict[str, Any] = response["result"]
        return result

    def tool(self, name: str, arguments: dict[str, Any]) -> tuple[Any, bool, str]:
        r = self.result("tools/call", {"name": name, "arguments": arguments})
        return r.get("structuredContent"), bool(r["isError"]), str(r["content"][0]["text"])

    def ok(self, name: str, arguments: dict[str, Any]) -> Any:
        structured, is_error, text = self.tool(name, arguments)
        assert not is_error, f"{name}: {text}"
        return structured

    def refused(self, name: str, arguments: dict[str, Any]) -> str:
        _, is_error, text = self.tool(name, arguments)
        assert is_error, f"{name} {arguments}: accepted: {text}"
        return text

    def initialize(self) -> None:
        r = self.result(
            "initialize",
            {"protocolVersion": "2025-11-25", "capabilities": {}, "clientInfo": {"name": "t"}},
        )
        assert r["protocolVersion"] == "2025-11-25"
        assert r["serverInfo"]["name"] == "mopt-rs"
        self.send('{"jsonrpc":"2.0","method":"notifications/initialized"}')

    def wait(self, job: str, timeout: float = 300.0) -> dict[str, Any]:
        begin = time.monotonic()
        while True:
            s: dict[str, Any] = self.ok("job_status", {"job_id": job})
            if s["state"] != "running":
                return s
            assert time.monotonic() - begin < timeout, s
            time.sleep(0.1)

    def close(self) -> None:
        assert self.process.stdin is not None
        self.process.stdin.close()
        assert self.process.wait(timeout=60) == 0


def _workspace(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "root"
    root.mkdir()
    (root / "ar7.xyz").write_text(AR7, encoding="utf-8")
    (tmp_path / "outside.xyz").write_text(AR7, encoding="utf-8")
    return root, tmp_path


def test_every_implemented_option_is_a_property_and_flipped_commands_appear() -> None:
    commands = tables.load_commands()
    tools = {t["name"]: t for t in tables.all_tools(commands, full=True)}
    for c in commands:
        if not c["implemented"]:
            assert c["name"] not in tools
            continue
        properties = tools[c["name"]]["inputSchema"]["properties"]
        for o in c["options"]:
            assert (o["key"] in properties) == bool(o["implemented"]), (c["name"], o["key"])
        assert tables.INPUT_PROPERTY in properties
    # The flag decides: a command marked as not implemented leaves the tools, and marked as
    # implemented again (with its options) it comes back. On a copy, as every command of the
    # table is implemented now.
    flipped = copy.deepcopy(commands)
    target = flipped[0]
    target["implemented"] = False
    assert target["name"] not in [t["name"] for t in tables.command_tools(flipped)]
    target["implemented"] = True
    for o in target["options"]:
        o["implemented"] = True
    names = [t["name"] for t in tables.command_tools(flipped)]
    assert target["name"] in names
    # The compact schemas list the inputs and the common options, and take the others by name.
    compact = {t["name"]: t for t in tables.all_tools(commands)}
    assert sorted(compact) == sorted(tools)
    for c in commands:
        if c["implemented"]:
            schema = compact[c["name"]]["inputSchema"]
            assert set(schema["properties"]) <= {
                tables.INPUT_PROPERTY,
                tables.FILES_PROPERTY,
                *tables.COMMON_OPTIONS,
            }
            assert isinstance(schema["additionalProperties"], dict)


def test_the_tools_are_those_of_the_rust_server(tmp_path: Path) -> None:
    binary = os.environ.get("MOPT_BIN")
    if not binary:
        pytest.skip("MOPT_BIN (the Rust binary mopt) is not set")
    for full in (False, True):
        run = subprocess.run(
            [binary, "mcp", "--root", str(tmp_path), "--page-size", "1000"]
            + (["--full-schemas"] if full else []),
            input=b'{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":'
            b'"2025-06-18"}}\n{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}\n',
            capture_output=True,
            check=True,
            timeout=120,
        )
        lines = run.stdout.decode("utf-8").splitlines()
        rust = {t["name"]: t for t in json.loads(lines[1])["result"]["tools"]}
        python = {t["name"]: t for t in tables.all_tools(tables.load_commands(), full)}
        assert sorted(rust) == sorted(python)
        for name, tool in python.items():
            assert rust[name] == tool, (full, name)


def test_path_texts_are_refused_as_in_the_rust_server() -> None:
    for bad in [
        "",
        "a\0b",
        "..",
        "../x",
        "sub/../a",
        "a\\b",
        "C:/x",
        "C:x",
        "//s/x",
        "NUL",
        "con.xyz",
        "a.",
        "x:y",
    ]:
        assert refused_path_text(bad) is not None, bad
    for good in ["a.xyz", "sub/a.xyz", "./a.xyz", "console.log"]:
        assert refused_path_text(good) is None, good
    for bad in ["/etc/passwd", "@/etc/passwd", "../x", "C:\\x", "~/x"]:
        assert refused_value_part(bad) is not None, bad
    for good in ["rsirfo_bfgs", "1,2", "-1.5", "tblite:GFN2-xTB"]:
        assert refused_value_part(good) is None, good


def test_a_client_runs_and_reads_a_job(tmp_path: Path) -> None:
    root, cwd = _workspace(tmp_path)
    c = Client(root, cwd)
    c.initialize()
    names = [t["name"] for t in c.result("tools/list", {})["tools"]]
    assert {"optmain", "job_status", "job_result", "single_point"} <= set(names)
    calculators = c.ok("list_calculators", {})
    assert "lj" in calculators["calculators"]
    help_text = c.ok("command_help", {"command": "optmain"})["help"]
    assert "--NSTEP" in help_text
    sp = c.ok("single_point", {"structure": "ar7.xyz", "calculator": "lj"})
    assert sp["energy_hartree"] > 0.0 and len(sp["gradient_hartree_per_bohr"]) == 7
    started = c.ok(
        "optmain",
        {"INPUT": ["ar7.xyz"], "othersoft": "lj", "opt_method": ["rsirfo_bfgs"], "NSTEP": 300},
    )
    job = started["job_id"]
    status = c.wait(job)
    assert status["state"] == "finished" and status["exit_code"] == 0, status
    result = c.ok("job_result", {"job_id": job})
    files = result["run_directories"][0]["files"]
    optimized = next(f for f in files if f["path"].endswith("ar7_optimized.xyz"))
    read = c.result("resources/read", {"uri": optimized["uri"]})
    assert "OptimizedStructure" in read["contents"][0]["text"]
    assert result["evaluations"]["gradients"] > 5
    docs = c.result("resources/read", {"uri": "mopt://docs/mcp"})
    assert "mopt mcp" in docs["contents"][0]["text"]
    c.close()


def _distance(xyz: str, i: int, j: int) -> float:
    rows = [line.split() for line in xyz.strip().splitlines()[2:]]
    a, b = (np.array([float(v) for v in rows[k][1:4]]) for k in (i, j))
    return float(np.linalg.norm(a - b))


def test_the_options_of_this_release_are_tools_and_work(tmp_path: Path) -> None:
    # -gfix of optmain and ircmain, -pc and -uc of ieipmain: properties of the tools.
    tools = {t["name"]: t for t in tables.all_tools(tables.load_commands(), full=True)}
    for command, keys in (
        ("optmain", ["gradient_fix_atoms"]),
        ("ircmain", ["gradient_fix_atoms"]),
        ("ieipmain", ["projection_constrain", "user_constraint"]),
    ):
        properties = tools[command]["inputSchema"]["properties"]
        for key in keys:
            assert properties[key]["type"] == "array", (command, key)
    # An optmain job that holds the distance of atoms 1 and 2 with -gfix.
    root, cwd = _workspace(tmp_path)
    c = Client(root, cwd)
    c.initialize()

    def optimized(**extra: Any) -> str:
        started = c.ok(
            "optmain",
            {"INPUT": ["ar7.xyz"], "othersoft": "lj", "opt_method": ["rsirfo_bfgs"], **extra},
        )
        status = c.wait(started["job_id"])
        assert status["state"] == "finished" and status["exit_code"] == 0, status
        files = c.ok("job_result", {"job_id": started["job_id"]})["run_directories"][0]["files"]
        found = next(f for f in files if f["path"].endswith("ar7_optimized.xyz"))
        text: str = c.result("resources/read", {"uri": found["uri"]})["contents"][0]["text"]
        return text

    held = optimized(NSTEP=300, gradient_fix_atoms=["1,2"])
    free = optimized(NSTEP=300)
    c.close()
    start = _distance(AR7, 0, 1)
    assert _distance(held, 0, 1) == pytest.approx(start, abs=1e-4)
    # Without it the distance changes.
    assert abs(_distance(free, 0, 1) - start) > 1e-2


def test_attacks_are_refused(tmp_path: Path) -> None:
    root, cwd = _workspace(tmp_path)
    outside = str(tmp_path / "outside.xyz")
    (root / "sub").mkdir()
    (root / "link.xyz").symlink_to(tmp_path / "outside.xyz")
    (root / "updir").symlink_to(tmp_path, target_is_directory=True)
    (root / "software_path.conf").write_text("orca::/usr/bin/orca\n", encoding="utf-8")
    c = Client(root, cwd)
    c.initialize()
    for path in [
        "../outside.xyz",
        "sub/../ar7.xyz",
        outside,
        "C:/x.xyz",
        "//server/share/x.xyz",
        "sub\\..\\..\\outside.xyz",
        "NUL",
        "con.xyz",
        "ar7.xyz\0",
        "link.xyz",
        "updir/outside.xyz",
        ".",
        "mcp_jobs",
    ]:
        c.refused("optmain", {"INPUT": [path], "othersoft": "lj"})
        c.refused("optmain", {"INPUT": ["ar7.xyz"], "othersoft": "lj", "files": [path]})
    for value in [outside, "/etc/passwd", "../../outside.xyz", "~/x", "C:\\x"]:
        c.refused("optmain", {"INPUT": ["ar7.xyz"], "othersoft": "lj", "initial_hessian": value})
    extras: list[dict[str, Any]] = [
        {"calc_option": ["p=@/etc/passwd"]},
        {"calc_option": ["p=a /etc/passwd"]},
        {"coordinate_system": "hyperspherical:/etc/passwd"},
    ]
    for extra in extras:
        c.refused("optmain", {"INPUT": ["ar7.xyz"], "othersoft": "lj", **extra})
    (root / "mopt_calculator_options.json").write_text('{"pyscf": {}}', encoding="utf-8")
    external: list[dict[str, Any]] = [
        {"othersoft": "ase:ase.calculators.emt:EMT"},
        {"othersoft": "orca"},
        {"othersoft": "lj", "calc_option": ["hook=os:system"]},
        {"othersoft": "lj", "calc_option": ["excited_setup=m:f"]},
        {"othersoft": "lj", "calc_option": ["x=a.b:c,d.e:f"]},
        {"othersoft": "lj", "oniom_options": ["low.system=x.xml"]},
        {"othersoft": "lj", "oniom_options": ["low.calc_option=hook=os:system"]},
        {"othersoft": "lj", "software_path_file": "software_path.conf"},
        {"othersoft": "lj", "files": ["software_path.conf"]},
        {"othersoft": "lj", "files": ["mopt_calculator_options.json"]},
    ]
    for args in external:
        text = c.refused("optmain", {"INPUT": ["ar7.xyz"], **args})
        assert "--allow-external" in text, (args, text)
    for job in ["../x", "/etc", "mcp_jobs", "job-../../x", ""]:
        c.refused("job_status", {"job_id": job})
    for uri in [
        "mopt://root/../outside.xyz",
        "mopt://root/%2e%2e/outside.xyz",
        "mopt://root/..%2Foutside.xyz",
        "mopt://root/%252e%252e/outside.xyz",
        "mopt://root/link.xyz",
        "mopt://root//etc/passwd",
        "file:///etc/passwd",
    ]:
        assert "error" in c.request("resources/read", {"uri": uri}), uri
    assert not (root / "mcp_jobs").exists() or not any((root / "mcp_jobs").iterdir())
    # A FIFO is never opened (it would block the reader).
    os.mkfifo(root / "pipe")
    c.refused("optmain", {"INPUT": ["pipe"], "othersoft": "lj"})
    c.refused("single_point", {"structure": "pipe", "calculator": "lj"})
    assert "error" in c.request("resources/read", {"uri": "mopt://root/pipe"})
    # A copied file named like a module does not stand in for it in the job (-P).
    (root / "json.py").write_text(
        "from pathlib import Path\nPath('pwned').write_text('x')\n", encoding="utf-8"
    )
    started = c.ok(
        "optmain", {"INPUT": ["ar7.xyz"], "othersoft": "lj", "NSTEP": 2, "files": ["json.py"]}
    )
    status = c.wait(started["job_id"])
    job_dir = root / started["job_directory"]
    assert not (job_dir / "pwned").exists(), status
    assert status["state"] == "finished", status
    # Malformed messages are answered and the server goes on.
    for line, code in [
        ("{not json", -32700),
        ("[" * 100_000, -32600),
        ('[{"jsonrpc":"2.0","id":1,"method":"ping"}]', -32600),
        ('{"jsonrpc":"2.0","id":null,"method":"ping"}', -32600),
    ]:
        c.send(line)
        assert c.read()["error"]["code"] == code, line[:40]
    c.send(b'{"jsonrpc":"2.0","id":9,"method":"\xff"}')
    assert "error" in c.read()
    assert c.result("ping", {}) == {}
    c.close()
