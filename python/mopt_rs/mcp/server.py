"""The server loop and the routing of the methods (as ``crates/mopt-rs/src/mcp/server.rs``).

A request with ``_meta["io.modelcontextprotocol/protocolVersion"]`` is answered in that revision
(``2026-07-28``: ``resultType``, the cache fields, the server identity in ``_meta``);
``initialize`` selects a revision with the handshake for the rest of the process.

A thread reads the input, so that the loop can answer some requests later and send
notifications meanwhile: ``job_wait`` and ``tasks/result`` (``2025-11-25``) answer when their
job ends (progress notifications with ``_meta.progressToken``; ``notifications/cancelled`` ends
them without an answer); ``subscriptions/listen`` (``2026-07-28``) and ``resources/subscribe``
send ``notifications/resources/updated`` (and ``notifications/tasks``); the tasks made by
task-augmented calls of ``2025-11-25`` send ``notifications/tasks/status``.

MultiOptPy: no counterpart (new in mopt-rs).

This file is part of mopt-py (the Python interface of mopt-rs).
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import json
import queue
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, TextIO

from mopt_rs.mcp import prompts, resources, tables, tasks, tools
from mopt_rs.mcp.jobs import JobManager, JobView
from mopt_rs.mcp.pagination import DEFAULT_PAGE_SIZE, Pager
from mopt_rs.mcp.paths import Sandbox, ToolError
from mopt_rs.mcp.policy import Policy
from mopt_rs.mcp.protocol import (
    INTERNAL_ERROR,
    INVALID_PARAMS,
    INVALID_REQUEST,
    LEGACY_RESOURCE_NOT_FOUND,
    LEGACY_VERSIONS,
    MAX_MESSAGE_BYTES,
    META_CLIENT_CAPABILITIES,
    META_PROTOCOL_VERSION,
    META_SERVER_INFO,
    METHOD_NOT_FOUND,
    MODERN_VERSION,
    Notification,
    Request,
    RpcError,
    error_response,
    invalid_params,
    negotiate_legacy,
    parse_message,
    result_response,
    supported_versions,
    unsupported_version,
)

SERVER_NAME = "mopt-rs"
_TTL_STATIC_MS = 3_600_000
#: How often waits, subscriptions and tasks are looked at (seconds).
TICK = 0.25
META_SUBSCRIPTION_ID = "io.modelcontextprotocol/subscriptionId"
MAX_LISTENS = 16
MAX_WATCHED = 256
MAX_PENDING = 64

INSTRUCTIONS = (
    "mopt-rs: geometry optimization, reaction paths, molecular dynamics, conformer and "
    "orientation searches, scans, IRC, vibrational analysis, AutoTS, reaction networks. Each "
    "implemented command (optmain, nebmain, run_mapper, ...) is a tool that starts a job under "
    "mcp_jobs/ of the server root and returns at once with a job_id (or a task, for clients "
    "with tasks): follow it with job_status or job_wait, read its run directory with job_result "
    "and resources/read, stop it with job_stop (a STOP file), end it with job_cancel, and "
    "continue it with job_restart (--restart). The arguments of a command tool are its "
    "command-line options by their long names, as in settings files (command_help lists them "
    "with their schemas); INPUT names the input files, with paths relative to the root. "
    "list_calculators names the calculators; single_point, model_hessian, vibrations, "
    "coordinates, interpolate, superpose, bond_graph, dispersion, hessian_update, units and "
    "elements answer at once."
)

_LATER = object()


def _version() -> str:
    try:
        from mopt_rs import _core

        return str(_core.__version__)
    except ImportError:
        return "unknown"


def _calculator_names() -> list[str]:
    try:
        from mopt_rs import _core

        return [*_core.native_calculators(), "mb", *_core.objective_names()]
    except ImportError:
        return []


def _notification(method: str, params: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "method": method, "params": params}


def _short(s: str) -> str:
    return (s[:120] + "...") if len(s) > 120 else s


@dataclass
class _Pending:
    id: Any
    version: str
    job: str
    kind: str
    tail: int
    started: float
    deadline: float | None
    token: Any = None
    count: int = 0
    seen: tuple[int, str | None] = (0, None)


@dataclass
class _Watched:
    uri: str
    fingerprint: str


@dataclass
class _Listen:
    id: Any
    resources: list[_Watched] = field(default_factory=list)
    tasks: list[list[str]] = field(default_factory=list)


class Server:
    """The server of a root directory."""

    def __init__(
        self,
        root: Path,
        *,
        allow_external: bool = False,
        max_jobs: int = 4,
        commands: list[tables.Command] | None = None,
        program: list[str] | None = None,
        full_schemas: bool = False,
        page_size: int = DEFAULT_PAGE_SIZE,
    ) -> None:
        self.commands = commands if commands is not None else tables.load_commands()
        self.sandbox = Sandbox(root)
        self.policy = Policy(allow_external, _calculator_names())
        self.jobs = JobManager(self.sandbox, self.policy, max_jobs, program)
        docs = resources.docs_dir()
        self.docs: Path | None = docs if docs.is_dir() else None
        self.legacy: str | None = None
        self.full_schemas = full_schemas
        self.pager = Pager(page_size)
        self.version = _version()
        self.pending: list[_Pending] = []
        self.listens: list[_Listen] = []
        self.subscribed: list[_Watched] = []
        self.legacy_tasks: list[list[str]] = []
        self._last_tick = time.monotonic()
        print(
            f"mopt mcp: root {self.sandbox.root}; external programs and code "
            f"{'allowed' if allow_external else 'refused'}; at most {self.jobs.max_jobs} jobs; "
            f"{'full' if full_schemas else 'compact'} tool schemas",
            file=sys.stderr,
        )

    # The loop.

    def _watching(self) -> bool:
        return bool(self.pending or self.listens or self.subscribed or self.legacy_tasks)

    def serve(self, source: BinaryIO, out: TextIO) -> None:
        """Answer the lines of ``source`` on ``out`` until the input ends."""
        lines: queue.Queue[bytes | None] = queue.Queue()

        def read() -> None:
            while True:
                raw = source.readline(MAX_MESSAGE_BYTES + 1)
                if not raw:
                    lines.put(None)
                    return
                if len(raw) > MAX_MESSAGE_BYTES and not raw.endswith(b"\n"):
                    while raw and not raw.endswith(b"\n"):
                        raw = source.readline(MAX_MESSAGE_BYTES)
                    lines.put(b"")
                    continue
                lines.put(raw)

        threading.Thread(target=read, daemon=True).start()

        def write(messages: list[dict[str, Any]]) -> None:
            for m in messages:
                out.write(json.dumps(m, ensure_ascii=False, separators=(",", ":")) + "\n")
            out.flush()

        while True:
            try:
                raw = lines.get(timeout=TICK) if self._watching() else lines.get()
            except queue.Empty:
                raw = b"\0"
            if raw is None:
                return
            if raw == b"":
                write(
                    [
                        error_response(
                            None,
                            RpcError(
                                INVALID_REQUEST,
                                f"Invalid request: a message is at most {MAX_MESSAGE_BYTES} bytes",
                            ),
                        )
                    ]
                )
            elif raw != b"\0":
                line = raw.decode("utf-8", errors="replace").strip()
                if line:
                    write(self.handle_line(line))
            if self._watching() and time.monotonic() - self._last_tick >= TICK:
                self._last_tick = time.monotonic()
                write(self.tick())

    def handle_line(self, line: str) -> list[dict[str, Any]]:
        """The messages to send for one line (no response for notifications and for requests
        answered later)."""
        message = parse_message(line)
        if message is None:
            return []
        if isinstance(message, dict):
            return [message]
        if isinstance(message, Notification):
            self._notified(message.method, message.params)
            return []
        assert isinstance(message, Request)
        out: list[dict[str, Any]] = []
        try:
            result = self.request(message.id, message.method, message.params, out)
        except RpcError as e:
            return [error_response(message.id, e), *out]
        except Exception:
            traceback.print_exc(file=sys.stderr)
            error = RpcError(INTERNAL_ERROR, f"Internal error while answering {message.method}")
            return [error_response(message.id, error), *out]
        if result is None:
            return out
        return [result_response(message.id, result), *out]

    def _notified(self, method: str, params: dict[str, Any]) -> None:
        if method == "notifications/cancelled" and "requestId" in params:
            rid = params["requestId"]
            self.pending = [p for p in self.pending if p.id != rid]
            self.listens = [s for s in self.listens if s.id != rid]

    # Eras.

    def _era(self, method: str, params: dict[str, Any]) -> str:
        meta = params.get("_meta")
        if isinstance(meta, dict) and META_PROTOCOL_VERSION in meta:
            version = meta[META_PROTOCOL_VERSION]
            if not isinstance(version, str):
                raise invalid_params(f"{META_PROTOCOL_VERSION} is a string")
            if version not in supported_versions():
                raise unsupported_version(version)
            if version == MODERN_VERSION and META_CLIENT_CAPABILITIES not in meta:
                raise invalid_params(f"the request has no _meta {META_CLIENT_CAPABILITIES}")
            return version
        if self.legacy is not None:
            return self.legacy
        if method == "ping":
            return LEGACY_VERSIONS[0]
        raise invalid_params(
            f"the request has no _meta {META_PROTOCOL_VERSION} (protocol {MODERN_VERSION}), and "
            f"no initialize came before (protocols {', '.join(LEGACY_VERSIONS)})",
            {"supported": supported_versions()},
        )

    def _capabilities(self, version: str) -> dict[str, Any]:
        caps: dict[str, Any] = {
            "tools": {"listChanged": False},
            "resources": {"listChanged": False, "subscribe": True},
            "prompts": {"listChanged": False},
        }
        if version == MODERN_VERSION:
            caps["extensions"] = {tasks.TASKS_EXTENSION: {}}
        elif version == tasks.LEGACY_TASKS_VERSION:
            caps["tasks"] = {"list": {}, "cancel": {}, "requests": {"tools": {"call": {}}}}
        return caps

    def _server_info(self) -> dict[str, Any]:
        return {"name": SERVER_NAME, "title": "mopt-rs", "version": self.version}

    def _wrap(self, version: str, result: dict[str, Any], ttl: int | None) -> dict[str, Any]:
        if version == MODERN_VERSION:
            result.setdefault("resultType", "complete")
            result["_meta"] = {META_SERVER_INFO: self._server_info()}
            if ttl is not None:
                result["ttlMs"] = ttl
                result["cacheScope"] = "private"
        return result

    def request(
        self, rid: Any, method: str, params: dict[str, Any], out: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        """The result of one request (``None``: answered later, or never)."""
        meta = params.get("_meta")
        if method == "initialize" and not (
            isinstance(meta, dict) and META_PROTOCOL_VERSION in meta
        ):
            requested = params.get("protocolVersion")
            version = negotiate_legacy(requested if isinstance(requested, str) else None)
            self.legacy = version
            return {
                "protocolVersion": version,
                "capabilities": self._capabilities(version),
                "serverInfo": self._server_info(),
                "instructions": INSTRUCTIONS,
            }
        version = self._era(method, params)
        answer = self._answer(rid, method, params, version, out)
        if answer is _LATER:
            return None
        assert isinstance(answer, tuple)
        result, ttl = answer
        return self._wrap(version, result, ttl)

    def _page(
        self, name: str, key: str, items: list[Any], params: dict[str, Any]
    ) -> dict[str, Any]:
        page, following = self.pager.page(name, items, params.get("cursor"))
        out: dict[str, Any] = {key: page}
        if following is not None:
            out["nextCursor"] = following
        return out

    def _is_command(self, name: str) -> bool:
        return any(c["name"] == name and c["implemented"] for c in self.commands)

    def _tools(self, version: str) -> list[dict[str, Any]]:
        listed = tables.all_tools(self.commands, self.full_schemas)
        if version == tasks.LEGACY_TASKS_VERSION:
            for t in listed:
                if self._is_command(t["name"]):
                    t["execution"] = {"taskSupport": "optional"}
        return listed

    # Tasks.

    def _view(self, task_id: Any) -> JobView:
        if not isinstance(task_id, str):
            raise invalid_params("give the taskId")
        try:
            return self.jobs.view(task_id)
        except ToolError as e:
            raise invalid_params(f"no task '{_short(task_id)}': {e}") from None

    def _detailed(self, task_id: str) -> dict[str, Any]:
        view = self._view(task_id)
        task = tasks.modern_task(task_id, view)
        status = tasks.status_of(view)
        if status == "completed":
            task["result"] = self._job_result(task_id)
        elif status == "failed":
            task["error"] = tasks.failure(task_id, view)
        return task

    def _job_result(self, job: str) -> dict[str, Any]:
        try:
            return tools.call_result(self.jobs.result(job))
        except ToolError as e:
            return tools.call_result(None, str(e))

    def _task_result(self, task_id: str) -> dict[str, Any]:
        view = self._view(task_id)
        result = self._job_result(task_id)
        if tasks.status_of(view) != "completed":
            result["isError"] = True
            result["content"].insert(0, {"type": "text", "text": tasks.status_message(view)})
        result["_meta"] = {tasks.META_RELATED_TASK: {"taskId": task_id}}
        return result

    def _start_task(self, name: str, args: dict[str, Any], version: str) -> dict[str, Any]:
        command = next(c for c in self.commands if c["name"] == name and c["implemented"])
        try:
            started = self.jobs.start(command, args, self.version)
        except (ToolError, ValueError, TypeError) as e:
            return tools.call_result(None, str(e))
        job = started["job_id"]
        view = self.jobs.view(job)
        if version == MODERN_VERSION:
            task = tasks.modern_task(job, view)
            task["resultType"] = "task"
            return task
        self.legacy_tasks.append([job, tasks.status_of(view)])
        return {"task": tasks.legacy_task(job, view)}

    # Waits.

    def _wait_result(self, job: str, tail: int, waited: float) -> dict[str, Any]:
        try:
            status = self.jobs.status(job, tail)
        except ToolError as e:
            return tools.call_result(None, str(e))
        status["waited_seconds"] = round(waited, 1)
        status["timed_out"] = status["state"] == "running"
        return tools.call_result(dict(sorted(status.items())))

    def _job_wait(
        self, rid: Any, args: dict[str, Any], params: dict[str, Any], version: str
    ) -> Any:
        job = args.get("job_id")
        if not isinstance(job, str):
            return tools.call_result(None, "give the job_id"), None
        timeout = args.get("timeout_seconds", 60.0)
        if timeout is None:
            timeout = 60.0
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, int | float)
            or timeout != timeout
            or timeout < 0
        ):
            message = f"timeout_seconds is a number from 0 to {tables.MAX_WAIT_SECONDS:g}"
            return tools.call_result(None, message), None
        timeout = min(float(timeout), tables.MAX_WAIT_SECONDS)
        tail = args.get("tail_lines", 40)
        tail = min(tail, 500) if isinstance(tail, int) and tail >= 0 else 40
        try:
            view = self.jobs.view(job)
        except ToolError as e:
            return tools.call_result(None, str(e)), None
        if view.state != "running" or timeout == 0:
            return self._wait_result(job, tail, 0.0), None
        if len(self.pending) >= MAX_PENDING:
            return tools.call_result(None, f"{MAX_PENDING} requests wait already"), None
        meta = params.get("_meta")
        token = meta.get("progressToken") if isinstance(meta, dict) else None
        if not (isinstance(token, str) or (isinstance(token, int) and not isinstance(token, bool))):
            token = None
        now = time.monotonic()
        self.pending.append(
            _Pending(
                rid,
                version,
                job,
                "job",
                tail,
                now,
                now + timeout,
                token,
                0,
                (view.log_bytes, view.last_iteration),
            )
        )
        return _LATER

    # Subscriptions.

    def _fingerprint(self, uri: str) -> str:
        if uri.startswith(resources.JOBS_PREFIX):
            try:
                v = self.jobs.view(uri[len(resources.JOBS_PREFIX) :])
            except ToolError:
                return "gone"
            return f"{v.state}:{v.log_bytes}:{v.cancel_requested}"
        return resources.fingerprint(self.sandbox, uri)

    def _watched(self, uri: str) -> _Watched:
        if uri.startswith(resources.JOBS_PREFIX):
            self.jobs.job_dir(uri[len(resources.JOBS_PREFIX) :])
        else:
            resources.check_watch(self.sandbox, self.docs, uri)
        return _Watched(uri, self._fingerprint(uri))

    def _watched_count(self) -> int:
        return len(self.subscribed) + sum(len(s.resources) + len(s.tasks) for s in self.listens)

    def _listen(self, rid: Any, params: dict[str, Any], out: list[dict[str, Any]]) -> Any:
        if len(self.listens) >= MAX_LISTENS:
            raise invalid_params(
                f"at most {MAX_LISTENS} subscriptions/listen streams are open at a time"
            )
        wanted = params.get("notifications")
        wanted = wanted if isinstance(wanted, dict) else {}
        listen = _Listen(rid)
        uris = wanted.get("resourceSubscriptions")
        for uri in uris if isinstance(uris, list) else []:
            if not isinstance(uri, str):
                continue
            if self._watched_count() + len(listen.resources) + len(listen.tasks) >= MAX_WATCHED:
                continue
            try:
                listen.resources.append(self._watched(uri))
            except ToolError as e:
                print(f"mopt mcp: subscription refused: {_short(uri)}: {e}", file=sys.stderr)
        ids = wanted.get("taskIds")
        for task in ids if isinstance(ids, list) else []:
            if not isinstance(task, str):
                continue
            try:
                listen.tasks.append([task, tasks.status_of(self.jobs.view(task))])
            except ToolError as e:
                print(f"mopt mcp: subscription refused: {_short(task)}: {e}", file=sys.stderr)
        agreed: dict[str, Any] = {}
        if listen.resources:
            agreed["resourceSubscriptions"] = [w.uri for w in listen.resources]
        if listen.tasks:
            agreed["taskIds"] = [t for t, _ in listen.tasks]
        out.append(
            _notification(
                "notifications/subscriptions/acknowledged",
                {"_meta": {META_SUBSCRIPTION_ID: rid}, "notifications": agreed},
            )
        )
        self.listens.append(listen)
        return _LATER

    # Methods.

    def _answer(
        self,
        rid: Any,
        method: str,
        params: dict[str, Any],
        version: str,
        out: list[dict[str, Any]],
    ) -> Any:
        modern = version == MODERN_VERSION
        legacy_tasks = version == tasks.LEGACY_TASKS_VERSION
        if method == "server/discover":
            return {
                "supportedVersions": supported_versions(),
                "capabilities": self._capabilities(MODERN_VERSION),
                "instructions": INSTRUCTIONS,
            }, _TTL_STATIC_MS
        if method == "ping" and not modern:
            return {}, None
        if method == "tools/list":
            return self._page("tools/list", "tools", self._tools(version), params), _TTL_STATIC_MS
        if method == "tools/call":
            name = params.get("name")
            if not isinstance(name, str):
                raise invalid_params("tools/call needs the name of the tool")
            args = params.get("arguments")
            if args is None:
                args = {}
            if not isinstance(args, dict):
                raise invalid_params("the arguments of a tool are an object")
            if modern:
                meta = params.get("_meta")
                caps = meta.get(META_CLIENT_CAPABILITIES) if isinstance(meta, dict) else None
                as_task = tasks.client_has_tasks(caps) and self._is_command(name)
            else:
                task = params.get("task")
                as_task = False
                if task is not None:
                    if not (legacy_tasks and isinstance(task, dict)):
                        raise invalid_params("task is an object, in protocol 2025-11-25")
                    if not self._is_command(name):
                        raise invalid_params(
                            f"the tool {name} does not run as a task (the tools of the commands do)"
                        )
                    as_task = True
            if as_task:
                return self._start_task(name, args, version), None
            if name == "job_wait":
                return self._job_wait(rid, args, params, version)
            return tools.call(name, args, self.commands, self.jobs, self.version), None
        if method == "tasks/get" and (modern or legacy_tasks):
            task_id = params.get("taskId")
            view = self._view(task_id)
            assert isinstance(task_id, str)
            if modern:
                return self._detailed(task_id), None
            return tasks.legacy_task(task_id, view), None
        if method == "tasks/cancel" and (modern or legacy_tasks):
            task_id = params.get("taskId")
            view = self._view(task_id)
            assert isinstance(task_id, str)
            status = tasks.status_of(view)
            if status in tasks.TERMINAL:
                raise invalid_params(f"task {task_id} is {status} already")
            try:
                self.jobs.request_cancel(task_id)
            except ToolError as e:
                raise invalid_params(str(e)) from None
            if modern:
                return {}, None
            return tasks.legacy_task(task_id, self._view(task_id)), None
        if method == "tasks/update" and modern:
            task_id = params.get("taskId")
            self._view(task_id)
            raise invalid_params(
                f"task {task_id} asks for no input (the jobs of mopt-rs run without questions)"
            )
        if method == "tasks/list" and legacy_tasks:
            listed = []
            for job in self.jobs.ids():
                try:
                    listed.append(tasks.legacy_task(job, self.jobs.view(job)))
                except ToolError:
                    continue
            return self._page("tasks/list", "tasks", listed, params), None
        if method == "tasks/result" and legacy_tasks:
            task_id = params.get("taskId")
            view = self._view(task_id)
            assert isinstance(task_id, str)
            if tasks.status_of(view) in tasks.TERMINAL:
                return self._task_result(task_id), None
            if len(self.pending) >= MAX_PENDING:
                raise invalid_params(f"{MAX_PENDING} requests wait already; use tasks/get")
            self.pending.append(_Pending(rid, version, task_id, "task", 0, time.monotonic(), None))
            return _LATER
        if method == "subscriptions/listen" and modern:
            return self._listen(rid, params, out)
        if method in ("resources/subscribe", "resources/unsubscribe") and not modern:
            uri = params.get("uri")
            if not isinstance(uri, str):
                raise invalid_params(f"{method} needs the uri")
            if method == "resources/unsubscribe":
                self.subscribed = [w for w in self.subscribed if w.uri != uri]
                return {}, None
            if any(w.uri == uri for w in self.subscribed):
                return {}, None
            if self._watched_count() >= MAX_WATCHED:
                raise invalid_params(f"at most {MAX_WATCHED} resources are watched at a time")
            try:
                self.subscribed.append(self._watched(uri))
            except ToolError as e:
                raise RpcError(
                    LEGACY_RESOURCE_NOT_FOUND, f"Resource not found: {e}", {"uri": _short(uri)}
                ) from None
            return {}, None
        if method == "resources/list":
            listed = resources.list_resources(self.sandbox, self.docs)
            for job in self.jobs.ids():
                listed.append(
                    {
                        "uri": resources.JOBS_PREFIX + job,
                        "name": f"job {job}",
                        "description": "status of a job",
                        "mimeType": "application/json",
                    }
                )
            return self._page("resources/list", "resources", listed, params), 0
        if method == "resources/templates/list":
            page = self._page(
                "resources/templates/list", "resourceTemplates", resources.templates(), params
            )
            return page, _TTL_STATIC_MS
        if method == "resources/read":
            uri = params.get("uri")
            if not isinstance(uri, str):
                raise invalid_params("resources/read needs the uri")
            code = INVALID_PARAMS if modern else LEGACY_RESOURCE_NOT_FOUND
            if uri.startswith(resources.JOBS_PREFIX):
                try:
                    job_status = self.jobs.status(uri[len(resources.JOBS_PREFIX) :], 40)
                except ToolError as e:
                    raise RpcError(code, f"Resource not found: {e}", {"uri": _short(uri)}) from None
                text = json.dumps(job_status, indent=2, sort_keys=True)
                return {"contents": [{"uri": uri, "mimeType": "application/json", "text": text}]}, 0
            try:
                return {"contents": resources.read(self.sandbox, self.docs, uri)}, 0
            except resources.ResourceNotFound as e:
                raise RpcError(code, f"Resource not found: {e}", {"uri": _short(uri)}) from None
        if method == "prompts/list":
            return self._page(
                "prompts/list", "prompts", prompts.list_prompts(), params
            ), _TTL_STATIC_MS
        if method == "prompts/get":
            name = params.get("name")
            if not isinstance(name, str):
                raise invalid_params("prompts/get needs the name")
            arguments = params.get("arguments") or {}
            if not isinstance(arguments, dict):
                raise invalid_params("the arguments of a prompt are an object")
            return prompts.get_prompt(name, arguments), None
        raise RpcError(METHOD_NOT_FOUND, f"Method not found: {method}")

    # What is watched.

    def tick(self) -> list[dict[str, Any]]:
        """Look at what is waited for and watched; the messages to send."""
        out: list[dict[str, Any]] = []
        kept: list[_Pending] = []
        now = time.monotonic()
        for p in self.pending:
            try:
                view: JobView | None = self.jobs.view(p.job)
            except ToolError:
                view = None
            if view is None:
                done = True
            elif p.kind == "job":
                done = view.state != "running"
            else:
                done = tasks.status_of(view) in tasks.TERMINAL
            late = p.deadline is not None and now >= p.deadline
            if done or late:
                try:
                    if p.kind == "job":
                        result = self._wait_result(p.job, p.tail, now - p.started)
                    else:
                        result = self._task_result(p.job)
                    out.append(result_response(p.id, self._wrap(p.version, result, None)))
                except RpcError as e:
                    out.append(error_response(p.id, e))
                continue
            if p.token is not None and view is not None:
                seen = (view.log_bytes, view.last_iteration)
                if seen != p.seen:
                    p.seen = seen
                    p.count += 1
                    out.append(
                        _notification(
                            "notifications/progress",
                            {
                                "progressToken": p.token,
                                "progress": p.count,
                                "message": view.last_iteration or f"{view.command} running",
                            },
                        )
                    )
            kept.append(p)
        self.pending = kept
        for w in self.subscribed:
            fp = self._fingerprint(w.uri)
            if fp != w.fingerprint:
                w.fingerprint = fp
                out.append(_notification("notifications/resources/updated", {"uri": w.uri}))
        for listen in self.listens:
            meta = {META_SUBSCRIPTION_ID: listen.id}
            for w in listen.resources:
                fp = self._fingerprint(w.uri)
                if fp != w.fingerprint:
                    w.fingerprint = fp
                    out.append(
                        _notification(
                            "notifications/resources/updated", {"uri": w.uri, "_meta": meta}
                        )
                    )
            for entry in listen.tasks:
                try:
                    status = tasks.status_of(self.jobs.view(entry[0]))
                except ToolError:
                    continue
                if status != entry[1]:
                    entry[1] = status
                    try:
                        detailed = self._detailed(entry[0])
                    except RpcError:
                        continue
                    detailed["_meta"] = meta
                    out.append(_notification("notifications/tasks", detailed))
        still: list[list[str]] = []
        for entry in self.legacy_tasks:
            try:
                view = self.jobs.view(entry[0])
            except ToolError:
                continue
            status = tasks.status_of(view)
            if status != entry[1]:
                entry[1] = status
                out.append(
                    _notification("notifications/tasks/status", tasks.legacy_task(entry[0], view))
                )
            if status not in tasks.TERMINAL:
                still.append(entry)
        self.legacy_tasks = still
        return out
