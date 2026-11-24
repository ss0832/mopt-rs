"""Jobs: a command run as a child process (the pip-installed command, through
``python -m mopt_rs.mcp.job``) in its own directory ``ROOT/mcp_jobs/JOB_ID/``.

The job model of ``crates/mopt-rs/src/mcp/jobs.rs``: the inputs are copied into the job
directory, the command reads ``mcp_settings.json`` (``--config``), stops at ``STOP`` in its run
directory, continues with ``--restart RUN_DIR``; the console output goes to ``console.log``, the
child records its exit code in ``mcp_exit_code``, and ``mcp_job.json`` keeps what was started
(and marks an attempt killed by ``job_cancel`` or asked to stop by a cancelled task). A job runs
in a process group of its own, so that ``job_cancel`` also ends the workers an engine starts.

MultiOptPy: no counterpart (new in mopt-rs).

This file is part of mopt-py (the Python interface of mopt-rs).
Copyright (C) 2023-2026 ss0832
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import contextlib
import json
import os
import secrets
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mopt_rs.mcp import job_settings, run_files
from mopt_rs.mcp.paths import Sandbox, ToolError
from mopt_rs.mcp.policy import Policy
from mopt_rs.mcp.resources import root_uri
from mopt_rs.mcp.tables import Command

JOB_FILE = "mcp_job.json"
EXIT_FILE = "mcp_exit_code"
CONSOLE_FILE = "console.log"
STOP_FILE = "STOP"
EXIT_OK = 0
EXIT_INTERRUPTED = 3
_START_WAIT = 3.0
_RECORD_LIMIT = 1 << 20
_LOG_LIMIT = 64 << 20


def _plain(path: Path) -> bool:
    """Whether ``path`` is a regular file itself (not a link)."""
    return path.is_file() and not path.is_symlink()


def _read_text(path: Path, limit: int) -> str | None:
    if not _plain(path) or path.stat().st_size > limit:
        return None
    try:
        return path.read_bytes().decode("utf-8", errors="replace")
    except OSError:
        return None


def _read_json(path: Path) -> dict[str, Any] | None:
    text = _read_text(path, _RECORD_LIMIT)
    if text is None:
        return None
    try:
        data = json.loads(text)
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def _open_plain(path: Path, append: bool) -> Any:
    """``path`` opened for writing, refusing anything there that is not a regular file."""
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise ToolError(f"{path.name} is not a regular file")
    # O_NOFOLLOW (where the system has it): a link swapped in after the check is not followed.
    flags = os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
    if not path.exists():
        flags |= os.O_CREAT | os.O_EXCL
    flags |= os.O_APPEND if append else os.O_TRUNC
    try:
        fd = os.open(path, flags, 0o644)
    except OSError as e:
        raise ToolError(f"{path.name}: {e.strerror}") from None
    return os.fdopen(fd, "ab" if append else "wb")


def _write_json(path: Path, value: Any) -> None:
    with _open_plain(path, False) as f:
        f.write((json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))


def is_job_id(job_id: str) -> bool:
    """Whether ``job_id`` has the form of the ids this server makes."""
    h = job_id[4:] if job_id.startswith("job-") else ""
    return len(h) == 32 and all(c in "0123456789abcdef" for c in h)


def _stamp() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _state_of(code: int | None) -> str:
    if code == EXIT_OK:
        return "finished"
    if code == EXIT_INTERRUPTED:
        return "interrupted"
    return "failed"


def _alive_in(pid: int, d: Path) -> bool:
    try:
        return Path(os.readlink(f"/proc/{pid}/cwd")).resolve() == d
    except OSError:
        return False


@dataclass
class JobView:
    """What a task needs to know about a job (as ``JobView`` of the Rust server)."""

    state: str
    exit_code: int | None
    cancel_requested: bool
    created: str
    updated: str
    command: str
    last_iteration: str | None
    log_bytes: int


def _stamp_of(seconds: float) -> str:
    return datetime.fromtimestamp(seconds, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


#: The most lines of one ``mapper_steer`` call.
MAX_STEER_LINES = 100


def steer_line_refused(line: str) -> str | None:
    """Why ``line`` is not a steering line of ``run_mapper`` (as the Rust server)."""
    shown = line[:80].replace("\0", "\\0")
    if (
        any(not ch.isprintable() or ch in "\t\n\r\x0b\x0c" for ch in line)
        or "#" in line
        or len(line) > 200
    ):
        return f"'{shown}': one line without control characters or comments"
    words = line.split()

    def name(w: str) -> bool:
        return 0 < len(w) <= 32 and all(c.isascii() and (c.isalnum() or c in "_-") for c in w)

    valid = len(words) == 2 and (
        (words[0] in ("explore", "skip", "unskip") and name(words[1]))
        or (words[0] == "max_jobs" and words[1].isascii() and words[1].isdigit())
    )
    if valid:
        return None
    return f"'{shown}' is not a steering line (explore NAME, skip NAME, unskip NAME, max_jobs N)"


def _lines(text: str, n: int) -> str:
    every = text.splitlines()
    return "\n".join(every[max(0, len(every) - n) :])


class JobManager:
    """The jobs of a server."""

    def __init__(
        self,
        sandbox: Sandbox,
        policy: Policy,
        max_jobs: int = 4,
        program: list[str] | None = None,
    ) -> None:
        self.sandbox = sandbox
        self.policy = policy
        self.max_jobs = max(1, max_jobs)
        #: The command line before the command; ``-P``: the job directory is not searched for
        #: modules, so that a copied file cannot stand in for a module.
        self.program = program or [sys.executable, "-P", "-m", "mopt_rs.mcp.job"]
        self._children: dict[str, subprocess.Popen[bytes]] = {}

    def _rel(self, path: Path) -> str:
        return self.sandbox.relative(path)

    def _new_id(self, jobs: Path) -> str:
        while True:
            job_id = f"job-{secrets.token_hex(16)}"
            if not (jobs / job_id).exists():
                return job_id

    def job_dir(self, job_id: str) -> Path:
        """The directory whose record names ``job_id`` (compared, never used as a path)."""
        if not is_job_id(job_id):
            raise ToolError(f"'{job_id[:80]}' is not a job id of this server")
        jobs = self.sandbox.jobs_dir()
        for entry in run_files.sorted_entries(jobs):
            if entry.is_symlink() or not entry.is_dir():
                continue
            record = _read_json(entry / JOB_FILE)
            if record is not None and record.get("job_id") == job_id:
                d = entry.resolve(strict=True)
                if self.sandbox.contains(d):
                    return d
        raise ToolError(f"no job '{job_id}'")

    def _running(self) -> int:
        return sum(1 for c in self._children.values() if c.poll() is None)

    def run_directories(self, d: Path) -> list[Path]:
        """The run directories of the job in ``d``."""
        record = _read_json(d / JOB_FILE) or {}
        copied = [d / c for c in record.get("copied", [])]
        return run_files.run_directories(d, copied)

    def start(self, command: Command, args: dict[str, Any], version: str) -> dict[str, Any]:
        """Start ``command`` for the arguments of its tool."""
        request = job_settings.request_of(command, args)
        self.policy.check_settings(request.settings)
        if self._running() >= self.max_jobs:
            raise ToolError(
                f"{self.max_jobs} jobs of this server are running, the most it starts at a time "
                "(--max-jobs); wait for one to end or stop one"
            )
        jobs = self.sandbox.jobs_dir()
        sources = [self.sandbox.existing(p) for p in request.inputs + request.files]
        job_id = self._new_id(jobs)
        d = jobs / job_id
        d.mkdir()
        d = d.resolve(strict=True)
        budget = job_settings.CopyBudget()
        copied: list[str] = []
        try:
            job_settings.check_paths(self.sandbox, d, request.settings)
            done: list[Path] = []
            # Each file once; a file inside a directory that is copied comes with it.
            for source in sources:
                covered = any(o != source and o.is_dir() and o in source.parents for o in sources)
                if covered or source in done:
                    continue
                copied.append(
                    job_settings.copy_into(self.sandbox, self.policy, jobs, source, d, budget)
                )
                done.append(source)
            for doc in budget.json:
                self.policy.check_json(doc)
                job_settings.check_json_paths(self.sandbox, d, doc)
        except ToolError:
            shutil.rmtree(d, ignore_errors=True)
            raise
        inputs = [self._rel(s) for s in sources[: len(request.inputs)]]
        document = job_settings.settings_document(inputs, request.settings, version)
        _write_json(d / job_settings.SETTINGS_FILE, document)
        _write_json(
            d / JOB_FILE,
            {
                "job_id": job_id,
                "command": command["name"],
                "created": _stamp(),
                "inputs": inputs,
                "copied": copied,
                "attempts": [],
            },
        )
        self._spawn(job_id, d, command["name"], ["--config", job_settings.SETTINGS_FILE])
        runs = self._wait_for_run_directory(job_id, d)
        out = self._status_of(job_id, d, 10)
        out["run_directory"] = self._rel(runs[0]) if runs else None
        if budget.skipped:
            out["skipped_links"] = budget.skipped
        out["next"] = (
            "call job_status with this job_id until the state is not running; then job_result"
        )
        return out

    def _spawn(self, job_id: str, d: Path, command: str, arguments: list[str]) -> None:
        (d / EXIT_FILE).unlink(missing_ok=True)
        with _open_plain(d / CONSOLE_FILE, True) as console:
            child = subprocess.Popen(
                [*self.program, command, *arguments],
                cwd=d,
                stdin=subprocess.DEVNULL,
                stdout=console,
                stderr=subprocess.STDOUT,
                start_new_session=os.name == "posix",
            )
        record = _read_json(d / JOB_FILE) or {}
        record.setdefault("attempts", []).append(
            {"started": _stamp(), "arguments": [command, *arguments], "pid": child.pid}
        )
        _write_json(d / JOB_FILE, record)
        print(f"mopt mcp: job {job_id}: {command} {' '.join(arguments)}", file=sys.stderr)
        self._children[job_id] = child

    def _wait_for_run_directory(self, job_id: str, d: Path) -> list[Path]:
        begin = time.monotonic()
        while True:
            runs = self.run_directories(d)
            if runs or time.monotonic() - begin > _START_WAIT:
                return runs
            child = self._children.get(job_id)
            if child is not None and child.poll() is not None:
                return runs
            time.sleep(0.05)

    def state(self, job_id: str, d: Path) -> tuple[str, int | None]:
        """The state and exit code of the job ``job_id`` in ``d``."""
        recorded: int | None = None
        text = _read_text(d / EXIT_FILE, _RECORD_LIMIT)
        try:
            recorded = int(text.strip()) if text is not None else None
        except ValueError:
            recorded = None
        child = self._children.get(job_id)
        if child is not None:
            code = child.poll()
            if code is None:
                return "running", None
            del self._children[job_id]
            if recorded is None and self._marked(d, "cancelled"):
                return "cancelled", None
            final = recorded if recorded is not None else (code if code >= 0 else None)
            return _state_of(final), final
        if recorded is not None:
            return _state_of(recorded), recorded
        last = self._last_attempt(d)
        if last.get("cancelled") is True:
            return "cancelled", None
        pid = last.get("pid")
        if isinstance(pid, int) and _alive_in(pid, d):
            return "running", None
        return "unknown", None

    @staticmethod
    def _last_attempt(d: Path) -> dict[str, Any]:
        record = _read_json(d / JOB_FILE) or {}
        attempts = record.get("attempts") or []
        last = attempts[-1] if attempts else {}
        return last if isinstance(last, dict) else {}

    def _marked(self, d: Path, key: str) -> bool:
        return self._last_attempt(d).get(key) is True

    @staticmethod
    def _mark(d: Path, key: str) -> None:
        record = _read_json(d / JOB_FILE)
        if record is None:
            raise ToolError(f"{JOB_FILE} cannot be read")
        attempts = record.get("attempts") or []
        if attempts and isinstance(attempts[-1], dict):
            attempts[-1][key] = True
        _write_json(d / JOB_FILE, record)

    def cancel(self, job_id: str) -> dict[str, Any]:
        """End the running job of this server process at once (its process group is killed)."""
        d = self.job_dir(job_id)
        state, _ = self.state(job_id, d)
        if state != "running":
            raise ToolError(f"job {job_id} is not running (state {state})")
        child = self._children.get(job_id)
        if child is None:
            raise ToolError(
                f"job {job_id} was started by another server process, which this one cannot "
                "end; job_stop asks it to end at its next check"
            )
        try:
            if os.name == "posix":
                os.killpg(child.pid, signal.SIGKILL)
            else:
                child.kill()
        except (ProcessLookupError, PermissionError, OSError):
            child.kill()
        child.wait()
        del self._children[job_id]
        if not (d / EXIT_FILE).exists():
            self._mark(d, "cancelled")
        out = self._status_of(job_id, d, 10)
        out["next"] = "job_restart continues the run from its evaluation journal"
        return out

    def request_cancel(self, job_id: str) -> None:
        """Ask the job to end for a cancelled task: ``STOP``, or the end of the process when it
        has no run directory yet; the attempt is marked, so that the task stays cancelled."""
        d = self.job_dir(job_id)
        state, _ = self.state(job_id, d)
        if state != "running":
            raise ToolError(f"job {job_id} is not running (state {state})")
        self._mark(d, "cancel_requested")
        if self.run_directories(d):
            self.stop(job_id)
        else:
            self.cancel(job_id)

    def view(self, job_id: str) -> JobView:
        """What a task needs to know about the job ``job_id``."""
        d = self.job_dir(job_id)
        state, code = self.state(job_id, d)
        record = _read_json(d / JOB_FILE) or {}
        runs = self.run_directories(d)
        log = runs[-1] / run_files.RUN_LOG if runs else None
        log_bytes = log.stat().st_size if log is not None and _plain(log) else 0
        text = run_files.tail(log, 400) if log is not None else None
        times = []
        for f in [d / JOB_FILE, d / EXIT_FILE, *([log] if log is not None else [])]:
            with contextlib.suppress(OSError):
                times.append(f.lstat().st_mtime)
        updated = _stamp_of(max(times)) if times else _stamp()
        created = record.get("created")
        return JobView(
            state=state,
            exit_code=code,
            cancel_requested=self._marked(d, "cancel_requested"),
            created=created if isinstance(created, str) else updated,
            updated=updated,
            command=str(record.get("command") or ""),
            last_iteration=run_files.last_iteration(text) if text is not None else None,
            log_bytes=log_bytes,
        )

    def steer(self, job_id: str, lines: Any) -> dict[str, Any]:
        """Append steering lines to ``steer.txt`` of the running ``run_mapper`` job."""
        if (
            not isinstance(lines, list)
            or not 1 <= len(lines) <= MAX_STEER_LINES
            or not all(isinstance(x, str) for x in lines)
        ):
            raise ToolError(f"give 1 to {MAX_STEER_LINES} lines")
        for line in lines:
            why = steer_line_refused(line)
            if why is not None:
                raise ToolError(why)
        d = self.job_dir(job_id)
        record = _read_json(d / JOB_FILE) or {}
        if record.get("command") != "run_mapper":
            raise ToolError(f"job {job_id} is no run_mapper job")
        state, _ = self.state(job_id, d)
        if state != "running":
            raise ToolError(f"job {job_id} is not running (state {state})")
        runs = self.run_directories(d)
        if not runs:
            raise ToolError(f"job {job_id} has no run directory yet; try again in a moment")
        target = runs[-1] / "steer.txt"
        with _open_plain(target, True) as f:
            f.write("".join(f"{x.strip()}\n" for x in lines).encode("utf-8"))
        return {
            "job_id": job_id,
            "steer_file": self._rel(target),
            "lines": [x.strip() for x in lines],
            "log": root_uri(self._rel(runs[-1] / "steer.log")),
            "next": "the run reads the lines before its next choice of jobs; steer.log (the log "
            "resource) tells what became of each",
        }

    def ids(self) -> list[str]:
        """The ids of the jobs under the root, in the order of their directories."""
        try:
            jobs = self.sandbox.jobs_dir()
        except ToolError:
            return []
        out = []
        for entry in run_files.sorted_entries(jobs):
            if entry.is_symlink() or not entry.is_dir():
                continue
            record = _read_json(entry / JOB_FILE)
            job_id = record.get("job_id") if record else None
            if isinstance(job_id, str) and is_job_id(job_id):
                out.append(job_id)
        return out

    def status(self, job_id: str, tail_lines: int) -> dict[str, Any]:
        """The status of the job ``job_id``."""
        return self._status_of(job_id, self.job_dir(job_id), tail_lines)

    def _status_of(self, job_id: str, d: Path, tail_lines: int) -> dict[str, Any]:
        state, code = self.state(job_id, d)
        record = _read_json(d / JOB_FILE) or {}
        runs = self.run_directories(d)
        log = run_files.tail(runs[-1] / run_files.RUN_LOG, 400) if runs else None
        out: dict[str, Any] = {
            "job_id": job_id,
            "command": record.get("command"),
            "state": state,
            "exit_code": code,
            "attempt": len(record.get("attempts") or []),
            "job_directory": self._rel(d),
            "run_directories": [self._rel(r) for r in runs],
            "log_tail": _lines(log, tail_lines) if log is not None else None,
            "last_iteration": run_files.last_iteration(log) if log is not None else None,
            "console_tail": run_files.tail(d / CONSOLE_FILE, 20),
        }
        if state == "unknown":
            out["note"] = (
                "this job was started by another server process and no exit code was recorded: "
                "it may still run there, or it ended without recording one"
            )
        return dict(sorted(out.items()))

    def stop(self, job_id: str) -> dict[str, Any]:
        """Write ``STOP`` into the run directories of the running job."""
        d = self.job_dir(job_id)
        state, _ = self.state(job_id, d)
        if state not in ("running", "unknown"):
            raise ToolError(f"job {job_id} is not running (state {state})")
        runs = self.run_directories(d)
        if not runs:
            raise ToolError(f"job {job_id} has no run directory yet; try again in a moment")
        for r in runs:
            with _open_plain(r / STOP_FILE, False):
                pass
        return {
            "job_id": job_id,
            "stop_files": [self._rel(r / STOP_FILE) for r in runs],
            "next": "the command ends at its next check of the STOP file; job_status shows "
            "state interrupted then, and job_restart continues the run",
        }

    def restart(self, job_id: str, run: str | None, force: bool) -> dict[str, Any]:
        """Continue the job with ``--restart`` in its run directory."""
        d = self.job_dir(job_id)
        state, _ = self.state(job_id, d)
        if state == "running":
            raise ToolError(f"job {job_id} is running; stop it first")
        if state == "unknown" and not force:
            raise ToolError(
                f"the state of job {job_id} is unknown (another server process started it); "
                "give force=true once it surely does not run any more"
            )
        runs = self.run_directories(d)
        if run is not None:
            wanted = self.sandbox.existing(run)
            if wanted not in runs:
                raise ToolError(f"{run} is not a run directory of job {job_id}")
            chosen = wanted
        elif len(runs) == 1:
            chosen = runs[0]
        elif not runs:
            raise ToolError(f"job {job_id} has no run directory to continue")
        else:
            names = ", ".join(self._rel(r) for r in runs)
            raise ToolError(
                f"job {job_id} has several run directories; name one as run_directory: {names}"
            )
        record = _read_json(d / JOB_FILE) or {}
        command = record.get("command")
        if not isinstance(command, str):
            raise ToolError(f"{JOB_FILE} of job {job_id} names no command")
        self._spawn(job_id, d, command, ["--restart", chosen.relative_to(d).as_posix()])
        out = self._status_of(job_id, d, 10)
        out["run_directory"] = self._rel(chosen)
        return out

    def result(self, job_id: str) -> dict[str, Any]:
        """The files of the run directories and the totals of the evaluations."""
        d = self.job_dir(job_id)
        state, code = self.state(job_id, d)
        entries = []
        for r in self.run_directories(d):
            files, more = run_files.list_files(r, 2000)
            log = _read_text(r / run_files.RUN_LOG, _LOG_LIMIT) or ""
            entries.append(
                {
                    "path": self._rel(r),
                    "files": [
                        {
                            "path": self._rel(f),
                            "size": size,
                            "mimeType": run_files.mime_type(f),
                            "uri": root_uri(self._rel(f)),
                        }
                        for f, size in files
                    ],
                    "more_files": more,
                    "evaluations": run_files.evaluation_totals(log),
                }
            )
        console = _read_text(d / CONSOLE_FILE, _LOG_LIMIT) or ""
        own = [JOB_FILE, job_settings.SETTINGS_FILE, CONSOLE_FILE, EXIT_FILE]
        return {
            "job_id": job_id,
            "state": state,
            "exit_code": code,
            "run_directories": entries,
            "evaluations": run_files.evaluation_totals(console),
            "job_files": [root_uri(self._rel(d / f)) for f in own if _plain(d / f)],
        }

    def list(self) -> dict[str, Any]:
        """The jobs under the root."""
        out = []
        for entry in run_files.sorted_entries(self.sandbox.jobs_dir()):
            if entry.is_symlink() or not entry.is_dir():
                continue
            record = _read_json(entry / JOB_FILE)
            if record is None or not isinstance(record.get("job_id"), str):
                continue
            job_id = record["job_id"]
            d = entry.resolve(strict=True)
            if not is_job_id(job_id) or not self.sandbox.contains(d):
                continue
            state, code = self.state(job_id, d)
            out.append(
                {
                    "job_id": job_id,
                    "command": record.get("command"),
                    "created": record.get("created"),
                    "state": state,
                    "exit_code": code,
                    "run_directories": [self._rel(r) for r in self.run_directories(d)],
                }
            )
        return {"jobs": out}
