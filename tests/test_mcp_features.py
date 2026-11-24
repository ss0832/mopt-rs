"""The features of the MCP server of the Python package beyond plain tool calls, end to end over
stdio (as ``crates/mopt-rs/tests/mcp_features.rs`` for the Rust server): the compact tool list,
``command_help`` with options and schemas, pages, ``job_wait`` with progress and cancellation,
``job_cancel``, tasks of ``2026-07-28`` and ``2025-11-25``, subscriptions, the component tools,
and the attacks on these parts.

This file is part of mopt-py (the Python interface of mopt-rs).
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import json
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("mopt_rs._core", reason="the compiled package mopt_rs is not importable")

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
WATER = "3\nwater\nO 0.0 0.0 0.117\nH 0.0 0.757 -0.469\nH 0.0 -0.757 -0.469\n"
MODERN = "2026-07-28"


def meta(tasks: bool = False) -> dict[str, Any]:
    caps: dict[str, Any] = {"extensions": {"io.modelcontextprotocol/tasks": {}}} if tasks else {}
    return {
        "io.modelcontextprotocol/protocolVersion": MODERN,
        "io.modelcontextprotocol/clientCapabilities": caps,
    }


def argon_grid(n: int) -> str:
    lines = [str(n**3), "argon grid"]
    k = 0
    for i in range(n):
        for j in range(n):
            for m in range(n):
                k += 1
                shake = ((k * 7919) % 13) * 0.03
                lines.append(
                    f"Ar {3.6 * i + shake:.4f} {3.6 * j - shake:.4f} {3.6 * m + 0.5 * shake:.4f}"
                )
    return "\n".join(lines) + "\n"


class Client:
    """``python -m mopt_rs.mcp`` as a child process; a thread reads its output."""

    def __init__(self, root: Path, cwd: Path, *extra: str) -> None:
        self.process = subprocess.Popen(
            [sys.executable, "-P", "-m", "mopt_rs.mcp", "--root", str(root), *extra],
            cwd=cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
        self.lines: queue.Queue[bytes] = queue.Queue()
        self.next_id = 0
        self.notifications: list[dict[str, Any]] = []
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        assert self.process.stdout is not None
        for line in self.process.stdout:
            self.lines.put(line)

    def send(self, message: dict[str, Any]) -> None:
        assert self.process.stdin is not None
        self.process.stdin.write(json.dumps(message).encode("utf-8") + b"\n")
        self.process.stdin.flush()

    def read(self, timeout: float = 600.0) -> dict[str, Any] | None:
        try:
            line = self.lines.get(timeout=timeout)
        except queue.Empty:
            return None
        message: dict[str, Any] = json.loads(line)
        return message

    def new_id(self) -> int:
        self.next_id += 1
        return self.next_id

    def response_to(self, rid: int) -> dict[str, Any]:
        while True:
            m = self.read()
            assert m is not None, "no response"
            if "id" not in m and "method" in m:
                self.notifications.append(m)
                continue
            assert m.get("id") == rid, m
            return m

    def request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        rid = self.new_id()
        self.send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})
        return self.response_to(rid)

    def result(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        r = self.request(method, params)
        assert "result" in r, r
        result: dict[str, Any] = r["result"]
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

    def take(self, method: str) -> list[dict[str, Any]]:
        taken = [n for n in self.notifications if n.get("method") == method]
        self.notifications = [n for n in self.notifications if n.get("method") != method]
        return taken

    def wait_notification(self, method: str, timeout: float = 60.0) -> dict[str, Any]:
        begin = time.monotonic()
        while True:
            for k, n in enumerate(self.notifications):
                if n.get("method") == method:
                    return self.notifications.pop(k)
            left = timeout - (time.monotonic() - begin)
            assert left > 0, f"no {method}"
            m = self.read(left)
            if m is not None:
                assert "id" not in m, m
                self.notifications.append(m)

    def initialize(self, version: str = "2025-11-25") -> dict[str, Any]:
        r = self.result(
            "initialize",
            {"protocolVersion": version, "capabilities": {}, "clientInfo": {"name": "t"}},
        )
        self.send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        return r

    def close(self) -> None:
        assert self.process.stdin is not None
        self.process.stdin.close()
        assert self.process.wait(timeout=120) == 0


def _workspace(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "root"
    root.mkdir()
    (root / "ar7.xyz").write_text(AR7, encoding="utf-8")
    (root / "water.xyz").write_text(WATER, encoding="utf-8")
    (tmp_path / "outside.xyz").write_text(AR7, encoding="utf-8")
    return root, tmp_path


def test_compact_tools_command_help_and_pages(tmp_path: Path) -> None:
    root, cwd = _workspace(tmp_path)
    c = Client(root, cwd, "--page-size", "7")
    c.initialize()
    names: list[str] = []
    cursor = None
    pages = 0
    while True:
        page = c.result("tools/list", {} if cursor is None else {"cursor": cursor})
        assert len(page["tools"]) <= 7
        names += [t["name"] for t in page["tools"]]
        pages += 1
        cursor = page.get("nextCursor")
        if cursor is None:
            break
    assert pages >= 4
    assert {"optmain", "job_wait", "job_cancel", "model_hessian", "elements"} <= set(names)
    for bad in ["", "0", "1.0000000000000000", "x" * 1000, 3]:
        r = c.request("tools/list", {"cursor": bad})
        assert r["error"]["code"] == -32602, bad
    optmain = next(t for t in c.result("tools/list", {})["tools"] if t["name"] == "optmain")
    assert len(optmain["inputSchema"]["properties"]) < 10
    found = c.ok(
        "command_help", {"command": "optmain", "options": ["-ns", "modelhess"], "schema": True}
    )
    assert [o["key"] for o in found["matches"][0]["options"]] == ["NSTEP"]
    hessian = next(o for o in found["matches"][1]["options"] if o["key"] == "use_model_hessian")
    assert "gfn1xtb" in hessian["help"]
    assert len(found["inputSchema"]["properties"]) > 100
    # An option that the compact schema does not list is taken by its long name.
    started = c.ok(
        "optmain",
        {"INPUT": ["ar7.xyz"], "othersoft": "lj", "use_model_hessian": "lindh", "NSTEP": 300},
    )
    waited = c.ok("job_wait", {"job_id": started["job_id"], "timeout_seconds": 300})
    assert waited["state"] == "finished", waited
    text = c.refused("optmain", {"INPUT": ["ar7.xyz"], "no_such_option": 1})
    assert "no_such_option" in text
    c.close()


def test_job_wait_progress_cancellation_and_job_cancel(tmp_path: Path) -> None:
    root, cwd = _workspace(tmp_path)
    (root / "ar.xyz").write_text(argon_grid(6), encoding="utf-8")
    c = Client(root, cwd)
    c.initialize()
    job = c.ok("optmain", {"INPUT": ["ar.xyz"], "othersoft": "lj", "NSTEP": 100000})["job_id"]
    heard: list[dict[str, Any]] = []
    for k in range(12):
        rid = c.new_id()
        c.send(
            {
                "jsonrpc": "2.0",
                "id": rid,
                "method": "tools/call",
                "params": {
                    "name": "job_wait",
                    "arguments": {"job_id": job, "timeout_seconds": 5},
                    "_meta": {"progressToken": f"p{k}"},
                },
            }
        )
        waited = c.response_to(rid)["result"]["structuredContent"]
        assert waited["state"] == "running" and waited["timed_out"] is True, waited
        heard = c.take("notifications/progress")
        assert all(n["params"]["progressToken"] == f"p{k}" for n in heard)
        if heard:
            break
    assert heard, "no progress"
    values = [n["params"]["progress"] for n in heard]
    assert values == sorted(set(values))
    cancelled = c.new_id()
    c.send(
        {
            "jsonrpc": "2.0",
            "id": cancelled,
            "method": "tools/call",
            "params": {"name": "job_wait", "arguments": {"job_id": job, "timeout_seconds": 600}},
        }
    )
    c.send(
        {"jsonrpc": "2.0", "method": "notifications/cancelled", "params": {"requestId": cancelled}}
    )
    assert c.ok("job_status", {"job_id": job})["state"] == "running"
    ended = c.ok("job_cancel", {"job_id": job})
    assert ended["state"] == "cancelled", ended
    assert "not running" in c.refused("job_cancel", {"job_id": job})
    while (m := c.read(0.6)) is not None:
        assert "id" not in m, m
    c.close()
    again = Client(root, cwd)
    again.initialize()
    assert again.ok("job_status", {"job_id": job})["state"] == "cancelled"
    assert again.ok("job_restart", {"job_id": job})["attempt"] == 2
    again.ok("job_cancel", {"job_id": job})
    again.close()


def test_tasks_of_both_revisions(tmp_path: Path) -> None:
    root, cwd = _workspace(tmp_path)
    (root / "ar.xyz").write_text(argon_grid(5), encoding="utf-8")
    c = Client(root, cwd)
    discover = c.result("server/discover", {"_meta": meta()})
    assert "io.modelcontextprotocol/tasks" in discover["capabilities"]["extensions"]
    r = c.result(
        "tools/call",
        {
            "name": "optmain",
            "arguments": {"INPUT": ["ar7.xyz"], "othersoft": "lj"},
            "_meta": meta(True),
        },
    )
    assert r["resultType"] == "task" and r["status"] == "working", r
    task = r["taskId"]
    for _ in range(600):
        done = c.result("tasks/get", {"taskId": task, "_meta": meta(True)})
        if done["status"] != "working":
            break
        time.sleep(0.5)
    assert done["status"] == "completed", done
    assert done["result"]["isError"] is False
    long = c.result(
        "tools/call",
        {
            "name": "optmain",
            "arguments": {"INPUT": ["ar.xyz"], "othersoft": "lj", "NSTEP": 100000},
            "_meta": meta(True),
        },
    )["taskId"]
    listen = c.new_id()
    c.send(
        {
            "jsonrpc": "2.0",
            "id": listen,
            "method": "subscriptions/listen",
            "params": {"notifications": {"taskIds": [long]}, "_meta": meta(True)},
        }
    )
    ack = c.wait_notification("notifications/subscriptions/acknowledged")
    assert ack["params"]["_meta"]["io.modelcontextprotocol/subscriptionId"] == listen
    time.sleep(3)
    assert (
        c.result("tasks/cancel", {"taskId": long, "_meta": meta(True)})["resultType"] == "complete"
    )
    heard = c.wait_notification("notifications/tasks")
    assert heard["params"]["taskId"] == long and heard["params"]["status"] == "cancelled"
    for method, code in [
        ("tasks/cancel", -32602),
        ("tasks/update", -32602),
        ("tasks/result", -32601),
        ("tasks/list", -32601),
    ]:
        r = c.request(method, {"taskId": long, "_meta": meta(True)})
        assert r["error"]["code"] == code, method
    c.send({"jsonrpc": "2.0", "method": "notifications/cancelled", "params": {"requestId": listen}})
    c.close()
    legacy = Client(root, cwd)
    init = legacy.initialize()
    assert "call" in init["capabilities"]["tasks"]["requests"]["tools"]
    begin = time.monotonic()
    while legacy.ok("job_status", {"job_id": long})["state"] == "running":
        assert time.monotonic() - begin < 300
        time.sleep(0.2)
    tools = {t["name"]: t for t in legacy.result("tools/list", {})["tools"]}
    assert tools["optmain"]["execution"] == {"taskSupport": "optional"}
    r = legacy.result(
        "tools/call",
        {
            "name": "optmain",
            "arguments": {"INPUT": ["ar7.xyz"], "othersoft": "lj"},
            "task": {"ttl": 60000},
        },
    )
    t = r["task"]["taskId"]
    assert legacy.request("tools/call", {"name": "job_list", "task": {}})["error"]["code"] == -32602
    result = legacy.result("tasks/result", {"taskId": t})
    assert result["isError"] is False
    assert result["_meta"]["io.modelcontextprotocol/related-task"]["taskId"] == t
    status = legacy.wait_notification("notifications/tasks/status", 30)
    assert status["params"]["taskId"] == t
    ids = [x["taskId"] for x in legacy.result("tasks/list", {})["tasks"]]
    assert {t, task, long} <= set(ids)
    assert legacy.result("tasks/get", {"taskId": long})["status"] == "cancelled"
    assert legacy.request("tasks/cancel", {"taskId": t})["error"]["code"] == -32602
    legacy.close()


def test_subscriptions_of_both_revisions(tmp_path: Path) -> None:
    root, cwd = _workspace(tmp_path)
    (root / "watched.txt").write_text("one\n", encoding="utf-8")
    c = Client(root, cwd)
    c.initialize()
    c.result("resources/subscribe", {"uri": "mopt://root/watched.txt"})
    c.result("resources/subscribe", {"uri": "mopt://root/later.txt"})
    time.sleep(0.6)
    (root / "watched.txt").write_text("one\ntwo\n", encoding="utf-8")
    n = c.wait_notification("notifications/resources/updated")
    assert n["params"]["uri"] == "mopt://root/watched.txt"
    (root / "later.txt").write_text("new\n", encoding="utf-8")
    n = c.wait_notification("notifications/resources/updated")
    assert n["params"]["uri"] == "mopt://root/later.txt"
    c.result("resources/unsubscribe", {"uri": "mopt://root/watched.txt"})
    (root / "watched.txt").write_text("three\n", encoding="utf-8")
    time.sleep(1.2)
    assert c.result("ping", {}) == {}
    assert not c.take("notifications/resources/updated")
    # What resources/read refuses is refused here too.
    (root / "link.txt").symlink_to(tmp_path / "outside.xyz")
    for uri in [
        "mopt://root/../outside.xyz",
        "mopt://root/%2e%2e/outside.xyz",
        "mopt://root/link.txt",
        "mopt://root//etc/passwd",
        "mopt://jobs/../x",
        "mopt://docs/../../README",
        "file:///etc/passwd",
    ]:
        assert "error" in c.request("resources/subscribe", {"uri": uri}), uri
    c.close()
    m = Client(root, cwd)
    listen = m.new_id()
    m.send(
        {
            "jsonrpc": "2.0",
            "id": listen,
            "method": "subscriptions/listen",
            "params": {
                "notifications": {
                    "resourceSubscriptions": [
                        "mopt://root/watched.txt",
                        "mopt://root/../outside.xyz",
                        "mopt://docs/mcp",
                    ],
                    "toolsListChanged": True,
                },
                "_meta": meta(),
            },
        }
    )
    ack = m.wait_notification("notifications/subscriptions/acknowledged")
    assert ack["params"]["notifications"] == {
        "resourceSubscriptions": ["mopt://root/watched.txt", "mopt://docs/mcp"]
    }
    time.sleep(0.6)
    (root / "watched.txt").write_text("four\n", encoding="utf-8")
    n = m.wait_notification("notifications/resources/updated")
    assert n["params"]["_meta"]["io.modelcontextprotocol/subscriptionId"] == listen
    m.send({"jsonrpc": "2.0", "method": "notifications/cancelled", "params": {"requestId": listen}})
    time.sleep(0.6)
    (root / "watched.txt").write_text("five\n", encoding="utf-8")
    time.sleep(1.2)
    m.result("server/discover", {"_meta": meta()})
    assert not m.take("notifications/resources/updated")
    m.close()


def test_component_tools(tmp_path: Path) -> None:
    root, cwd = _workspace(tmp_path)
    c = Client(root, cwd)
    c.initialize()
    model = c.ok("model_hessian", {"structure": "water.xyz", "name": "gfn1xtb"})
    point = c.ok(
        "single_point", {"structure": "water.xyz", "calculator": "gfn1xtb", "hessian": True}
    )
    a, b = model["hessian_hartree_per_bohr2"], point["hessian_hartree_per_bohr2"]
    assert len(a) == 81 and max(abs(x - y) for x, y in zip(a, b, strict=True)) < 1e-10
    v = c.ok("vibrations", {"structure": "water.xyz", "calculator": "gfn1xtb"})
    assert sum(1 for w in v["wavenumbers_cm1"] if w > 500) == 3
    assert c.ok("coordinates", {"structure": "water.xyz", "system": "redundant"})["dimension"] >= 3
    bent = WATER.replace("0.757 -0.469\nH 0.0 -0.757", "0.70 -0.52\nH 0.0 -0.80")
    path = c.ok("interpolate", {"start": "water.xyz", "end_xyz": bent, "images": 4})
    assert path["frames"] == 6
    assert (
        c.ok("superpose", {"structure": "water.xyz", "reference_xyz": WATER})["rmsd_angstrom"]
        < 1e-8
    )
    assert len(c.ok("bond_graph", {"structure": "water.xyz"})["bonds"]) == 2
    assert c.ok("dispersion", {"structure": "water.xyz", "model": "d3"})["energy_hartree"] < 0
    u = c.ok(
        "hessian_update",
        {"hessian": [1, 0, 0, 1], "step": [0.1, 0], "gradient_change": [0.3, 0], "method": "sr1"},
    )
    assert abs(u["hessian"][0] - 3.0) < 1e-12
    assert abs(c.ok("units", {"value": 1, "from": "hartree", "to": "ev"})["value"] - 27.211) < 1e-3
    assert c.ok("elements", {"elements": ["Cu", 79]})["elements"][1]["symbol"] == "Au"
    # The attacks on their structure arguments and settings.
    (root / "link.xyz").symlink_to(tmp_path / "outside.xyz")
    for p in ["../outside.xyz", str(tmp_path / "outside.xyz"), "/etc/passwd", "NUL", "link.xyz"]:
        c.refused("model_hessian", {"structure": p, "name": "lindh"})
        c.refused("superpose", {"structure": "ar7.xyz", "reference": p})
        c.refused("interpolate", {"start": "ar7.xyz", "end": p})
    for system in ["hyperspherical:../outside.xyz", "hyperspherical:/etc/passwd"]:
        c.refused("coordinates", {"structure": "ar7.xyz", "system": system})
    for calculator in ["ase:ase.calculators.emt:EMT", "orca"]:
        text = c.refused("vibrations", {"structure": "ar7.xyz", "calculator": calculator})
        assert "--allow-external" in text, text
    c.refused("bond_graph", {"structure": "ar7.xyz", "xyz": AR7})
    c.refused("interpolate", {"start": "ar7.xyz", "end": "ar7.xyz", "images": 10**6})
    for line in ["explore ../../etc/passwd", "explore EQ1\nmax_jobs 0", "max_jobs -1", "rm -rf /"]:
        text = c.refused("mapper_steer", {"job_id": "job-" + "0" * 32, "lines": [line]})
        assert "steering line" in text or "control characters" in text, (line, text)
    for job in ["../x", "fake", "job-" + "0" * 32]:
        c.refused("job_wait", {"job_id": job})
        c.refused("job_cancel", {"job_id": job})
        c.refused("mapper_steer", {"job_id": job, "lines": ["max_jobs 1"]})
        assert c.request("tasks/get", {"taskId": job})["error"]["code"] == -32602
    c.close()
