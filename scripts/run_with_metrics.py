#!/usr/bin/env python3
"""Run a local process group and atomically record bounded process metrics."""

from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any

SCHEMA_VERSION = "qcmlforge-process-metrics-v1"


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _group_rss_bytes(pgid: int) -> tuple[bool, int | None]:
    proc = Path("/proc")
    if not proc.is_dir():
        return False, None
    total = 0
    found = False
    for status in proc.glob("[0-9]*/status"):
        try:
            fields = {}
            for line in status.read_text().splitlines():
                if ":" in line:
                    key, value = line.split(":", 1)
                    fields[key] = value.strip()
            if int(fields.get("NSpgid", fields.get("Pgid", "-1")).split()[0]) != pgid:
                continue
            total += int(fields["VmRSS"].split()[0]) * 1024
            found = True
        except (OSError, ValueError, KeyError, IndexError):
            continue
    return True, total if found else None


def _maxrss_bytes() -> tuple[int, str]:
    value = int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
    if sys.platform == "darwin":
        return value, "resource.ru_maxrss-bytes"
    return value * 1024, "resource.ru_maxrss-kibibytes"


def _enable_subreaper() -> None:
    """Adopt orphaned descendants on Linux so they can be reaped."""

    if sys.platform.startswith("linux"):
        try:
            ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0)
        except (AttributeError, OSError):
            pass


def _reap_group_children(pgid: int, timeout: float) -> None:
    deadline = time.monotonic() + max(timeout, 0.2)
    while time.monotonic() < deadline:
        reaped = False
        while True:
            try:
                pid, _status = os.waitpid(-pgid, os.WNOHANG)
            except ChildProcessError:
                return
            if pid == 0:
                break
            reaped = True
        if not _group_has_live_members(pgid):
            return
        if not reaped:
            time.sleep(0.01)


def _group_has_live_members(pgid: int) -> bool:
    """Return whether a process group has any non-zombie members."""

    proc = Path("/proc")
    if proc.is_dir():
        for status in proc.glob("[0-9]*/status"):
            try:
                fields = dict(
                    line.split(":", 1) for line in status.read_text().splitlines()
                    if ":" in line
                )
                member = int(
                    fields.get("NSpgid", fields.get("Pgid", "-1")).strip().split()[0]
                )
                state = fields.get("State", "").strip()
                if member == pgid and not state.startswith("Z"):
                    return True
            except (OSError, ValueError, IndexError):
                continue
        return False
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    return True


def _terminate_group(pgid: int, grace: float) -> int | None:
    """TERM a group, inspect it for the full grace, then KILL survivors."""

    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return None
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        if not _group_has_live_members(pgid):
            return None
        time.sleep(min(0.02, max(0.001, deadline - time.monotonic())))
    if _group_has_live_members(pgid):
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            return None
        kill_deadline = time.monotonic() + max(grace, 0.2)
        while time.monotonic() < kill_deadline and _group_has_live_members(pgid):
            time.sleep(0.01)
        return signal.SIGKILL
    return None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--timeout-seconds", type=float)
    parser.add_argument("--kill-grace-seconds", type=float, default=2.0)
    parser.add_argument("--sample-interval-seconds", type=float, default=0.1)
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a command is required after --")
    if args.timeout_seconds is not None and args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    if args.kill_grace_seconds < 0 or args.sample_interval_seconds <= 0:
        parser.error("grace must be nonnegative and sample interval positive")
    if args.max_samples < 1:
        parser.error("--max-samples must be positive")

    output = Path(args.output)
    _enable_subreaper()
    started_at = _utc()
    started_mono = time.monotonic()
    child = None
    pgid = None
    received_signal = None
    signal_forwarded_at = None
    timed_out = False
    launch_error = None
    samples = []
    proc_max_rss = 0
    escalation_signal = None
    sampling_available = Path("/proc").is_dir()

    def forward(signum, _frame):
        nonlocal received_signal, signal_forwarded_at
        received_signal = signum
        signal_forwarded_at = time.monotonic()
        if pgid is not None:
            try:
                os.killpg(pgid, signum)
            except ProcessLookupError:
                pass

    old_handlers = {
        signum: signal.signal(signum, forward)
        for signum in (signal.SIGTERM, signal.SIGINT)
    }
    try:
        try:
            child = subprocess.Popen(command, start_new_session=True)
            pgid = os.getpgid(child.pid)
        except (OSError, ValueError) as exc:
            launch_error = {"type": type(exc).__name__, "message": str(exc)}

        if child is not None:
            deadline = (
                started_mono + args.timeout_seconds
                if args.timeout_seconds is not None else None
            )
            while child.poll() is None:
                now = time.monotonic()
                available, rss = _group_rss_bytes(pgid)
                sampling_available = available
                if rss is not None:
                    proc_max_rss = max(proc_max_rss, rss)
                    sample = {"monotonic_seconds": now, "rss_bytes": rss}
                    if len(samples) < args.max_samples:
                        samples.append(sample)
                    else:
                        samples[-1] = sample
                if received_signal is not None:
                    escalation_signal = _terminate_group(
                        pgid, args.kill_grace_seconds
                    )
                    break
                if deadline is not None and now >= deadline:
                    timed_out = True
                    escalation_signal = _terminate_group(
                        pgid, args.kill_grace_seconds
                    )
                    break
                time.sleep(args.sample_interval_seconds)
            child.wait()
            _reap_group_children(pgid, args.kill_grace_seconds)
            # The leader may have obeyed TERM before its descendants.  A
            # wrapper return after timeout/signal still guarantees the group
            # contains no live descendants.
            if (timed_out or received_signal is not None) and _group_has_live_members(pgid):
                escalation_signal = _terminate_group(pgid, args.kill_grace_seconds) or escalation_signal
                _reap_group_children(pgid, args.kill_grace_seconds)
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)

    ended_mono = time.monotonic()
    max_rss, rss_source = _maxrss_bytes()
    child_returncode = child.returncode if child is not None else None
    child_final_signal = (
        -child_returncode if child_returncode is not None and child_returncode < 0
        else None
    )
    terminating_signal = received_signal or child_final_signal
    wrapper_exit = (
        127 if launch_error else 124 if timed_out else
        128 + received_signal if received_signal else
        128 + child_final_signal if child_final_signal else int(child_returncode or 0)
    )
    outcome = (
        "launch_failure" if launch_error else "timeout" if timed_out else
        "signal" if received_signal or child_final_signal else
        "success" if child_returncode == 0 else "nonzero_exit"
    )
    record = {
        "schema_version": SCHEMA_VERSION,
        "outcome": outcome,
        "argv": command,
        "child_pid": child.pid if child is not None else None,
        "process_group_id": pgid,
        "started_at": started_at,
        "ended_at": _utc(),
        "started_monotonic_seconds": started_mono,
        "ended_monotonic_seconds": ended_mono,
        "wall_seconds": ended_mono - started_mono,
        "exit_code": child_returncode,
        "wrapper_exit_code": wrapper_exit,
        "terminating_signal": terminating_signal,
        "received_signal": received_signal,
        "forwarded_signal": received_signal,
        "child_final_signal": child_final_signal,
        "escalation_signal": escalation_signal,
        "timed_out": timed_out,
        "launch_error": launch_error,
        "host_rss": {
            "available": bool(max_rss or samples),
            "max_bytes": max(max_rss, proc_max_rss),
            "source": [
                rss_source,
                *(["proc-process-group-vmrss-sum"] if sampling_available else []),
            ],
            "unit": "bytes",
            "sampling_available": sampling_available,
            "sample_interval_seconds": args.sample_interval_seconds,
            "sample_limit": args.max_samples,
            "samples": samples,
        },
        "capabilities": {
            "gpu_memory": False,
            "gpu_utilization": False,
            "throughput": False,
            "slurm_accounting": False,
        },
    }
    _atomic_json(output, record)
    return wrapper_exit


if __name__ == "__main__":
    raise SystemExit(main())
