#!/usr/bin/env python3
"""Declarative authority for local CPU wiring validation (CI-0 through CI-2)."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import signal
import stat
import subprocess
import sys
import tempfile
import time
import uuid
import xml.etree.ElementTree as ET

from packaging.version import Version
import torch

try:
    from scripts.run_with_metrics import (
        _enable_subreaper, _group_has_live_members, _reap_group_children,
        _terminate_group,
    )
except ModuleNotFoundError:  # Direct ``python scripts/validate_mace_local.py``.
    from run_with_metrics import (
        _enable_subreaper, _group_has_live_members, _reap_group_children,
        _terminate_group,
    )

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "qcmlforge-mace-local-validation-v1"
POLARMACE_SHA256 = "e4495612037b3b3312633182882a38a694ecac9ea0be2b9889ac0b2a84a99510"
POLARMACE_SIZE = 33_375_439
GRAPH_LONGRANGE_COMMIT = "0e21d5546c482d08388a08eb4d948e833227ce47"
CI2_ATTESTATION_ENV = "QCMLFORGE_CI2_POLICY_ATTESTED"
SCOPE = "local-cpu-wiring-only"
CLAIM_BOUNDARY = {
    "scientific_accuracy": False,
    "cuda_parity": False,
    "scheduler_execution": False,
    "cluster_readiness": False,
    "production_readiness": False,
}
_RECEIVED_SIGNAL = None

CHECK_REGISTRY = (
    {
        "id": "ci0-local-contracts",
        "ci_tier": "CI-0",
        "required": True,
        "spec_refs": ["V1.1", "V1.5", "V1.7"],
        "summary": "Offline local schema, cache, wheel-policy, and script wiring checks",
        "tests": [
            "tests/test_mace_local_validation.py",
            "tests/test_mace_cache_validation.py",
            "tests/test_mace_slurm_scripts.py",
            "tests/test_mace_pass_b_contracts.py",
            "tests/test_mace_review_contracts.py",
        ],
    },
    {
        "id": "ci1-cpu-regressions",
        "ci_tier": "CI-1",
        "required": True,
        "spec_refs": ["V1.3", "V1.4", "V1.5"],
        "summary": "CPU stub and ordinary MACE regressions",
        "tests": [
            "tests/test_mace_ap3d3_architectures.py",
            "tests/test_mace_atomic_properties.py",
            "tests/test_mace_h1_pair.py",
            "tests/test_mace_h2_pair.py",
            "tests/test_mace_model_harness.py",
            "tests/test_mace_one_epoch.py",
            "tests/test_mace_polar_adapter.py",
        ],
    },
    {
        "id": "ci2-pinned-polarmace-cpu",
        "ci_tier": "CI-2",
        "required": True,
        "spec_refs": ["V1.2"],
        "summary": "Pinned real PolarMACE CPU integration",
        "tests": ["tests/"],
    },
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def aggregate_status(checks):
    required = [check for check in checks if check["required"]]
    if any(check["status"] == "FAIL" for check in required):
        return "FAIL", 1
    if any(check["status"] == "BLOCKED" for check in required):
        return "BLOCKED", 2
    if any(check["status"] == "SKIP" for check in required):
        return "FAIL", 1
    return "PASS", 0


def _atomic_json(path: Path, value) -> None:
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


def _source_state():
    def git(*args):
        try:
            return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return "unknown"
    commit = git("rev-parse", "HEAD")
    status = git("status", "--porcelain")
    return commit, None if status == "unknown" else bool(status)


def _junit_counts(path: Path):
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    counts = {
        name: sum(int(suite.attrib.get(name, "0")) for suite in suites)
        for name in ("tests", "failures", "errors", "skipped")
    }
    cases = list(root.iter("testcase"))
    counts["files"] = sorted({
        case.attrib["file"] for case in cases if case.get("file")
    })
    counts["class_names"] = sorted({
        case.attrib["classname"] for case in cases if case.get("classname")
    })
    return counts


def _claimed_test_files_executed(test_paths, counts) -> bool:
    """Recognize claimed files in either legacy or xunit2 pytest JUnit."""

    observed_files = set(counts.get("files", ()))
    observed_classes = set(counts.get("class_names", ()))
    for value in test_paths:
        if not value.endswith(".py"):
            continue
        expected = str(Path(value))
        module = ".".join(Path(value).with_suffix("").parts)
        if any(path.endswith(expected) for path in observed_files):
            continue
        if any(name == module or name.startswith(f"{module}.") for name in observed_classes):
            continue
        return False
    return True


def _evidence(output_dir: Path, paths):
    result = []
    for path in paths:
        if path.is_file():
            result.append({
                "path": str(path.relative_to(output_dir)),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            })
    return result


def _base_check(
    entry, status, reason, command, started, ended, exit_code, evidence,
    observations=None,
):
    return {
        "id": entry["id"], "spec_refs": entry["spec_refs"],
        "ci_tier": entry["ci_tier"], "required": entry["required"],
        "status": status, "summary": entry["summary"], "reason": reason,
        "command": command, "started_at": started[0], "ended_at": ended[0],
        "duration_seconds": ended[1] - started[1], "exit_code": exit_code,
        "evidence": evidence, "claim_boundary": dict(CLAIM_BOUNDARY),
        "observations": observations or {},
    }


def _offline_environment():
    env = os.environ.copy()
    for name in (
        "QCMLFORGE_AUTO_DOWNLOAD_PRETRAINED", "HF_ENDPOINT", "HUGGINGFACE_HUB_CACHE",
        "TRANSFORMERS_CACHE", "TORCH_HOME",
    ):
        env.pop(name, None)
    env.update({
        "QCMLFORGE_AUTO_DOWNLOAD_PRETRAINED": "0",
        "QCMLFORGE_MACE_OFFLINE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    })
    return env


def _run_bounded(command, *, cwd, env, stdout, stderr, timeout, grace):
    _enable_subreaper()
    child = subprocess.Popen(
        command, cwd=cwd, env=env, stdout=stdout, stderr=stderr,
        shell=False, start_new_session=True,
    )
    pgid = os.getpgid(child.pid)
    interrupted = None
    timed_out = False
    try:
        child.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        _terminate_group(pgid, grace)
    except KeyboardInterrupt:
        interrupted = _RECEIVED_SIGNAL or signal.SIGINT
        _terminate_group(pgid, grace)
    finally:
        if (timed_out or interrupted) and _group_has_live_members(pgid):
            _terminate_group(pgid, grace)
        child.wait()
        _reap_group_children(pgid, grace)
    return child.returncode, timed_out, interrupted


def run_pytest_check(entry, output_dir, artifact=None, timeout=900.0, grace=2.0):
    check_dir = output_dir / entry["id"]
    check_dir.mkdir(parents=True, exist_ok=True)
    stdout, stderr, junit = check_dir / "stdout.log", check_dir / "stderr.log", check_dir / "junit.xml"
    command = [sys.executable, "-m", "pytest", *entry["tests"], "--strict-markers", f"--junitxml={junit}"]
    env = _offline_environment()
    if entry["ci_tier"] == "CI-1":
        command += ["-m", "not mace_integration"]
    elif entry["ci_tier"] == "CI-2":
        command += ["-m", "mace_integration"]
        env["QCMLFORGE_REQUIRE_POLARMACE"] = "1"
        env["QCMLFORGE_POLARMACE_ARTIFACT"] = str(artifact)
        env["QCMLFORGE_MACE_OFFLINE"] = "1"
    started = (utc_now(), time.monotonic())
    with stdout.open("wb") as out, stderr.open("wb") as err:
        returncode, timed_out, interrupted = _run_bounded(
            command, cwd=ROOT, env=env, stdout=out, stderr=err,
            timeout=timeout, grace=grace,
        )
    ended = (utc_now(), time.monotonic())
    status, reason = "PASS", "command and JUnit criteria passed"
    if timed_out:
        status, reason = "FAIL", f"pytest timed out after {timeout} seconds"
    elif interrupted:
        status, reason = "FAIL", f"pytest interrupted by signal {interrupted}"
    else:
        try:
            counts = _junit_counts(junit)
            claimed_files_executed = _claimed_test_files_executed(
                entry["tests"], counts
            )
            counts.pop("files")
            counts.pop("class_names")
            if returncode or counts["failures"] or counts["errors"]:
                status, reason = "FAIL", f"pytest failed: {counts}"
            elif not counts["tests"] or counts["skipped"]:
                status, reason = "FAIL", f"pytest empty or skipped: {counts}"
            elif not claimed_files_executed:
                status, reason = "FAIL", "not every claimed test file executed"
            elif entry["ci_tier"] == "CI-2" and counts["tests"] != 4:
                status, reason = "FAIL", f"expected 4 integration tests, observed {counts['tests']}"
        except (OSError, ET.ParseError, ValueError) as exc:
            status, reason = "FAIL", f"JUnit evidence is malformed: {exc}"
    return _base_check(
        entry, status, reason, command, started, ended, returncode,
        _evidence(output_dir, (stdout, stderr, junit)),
        {"timed_out": timed_out, "received_signal": interrupted},
    )


def _graph_longrange_commit() -> str | None:
    distribution = importlib.metadata.distribution("graph-longrange")
    raw = distribution.read_text("direct_url.json")
    if not raw:
        return None
    value = json.loads(raw)
    return value.get("vcs_info", {}).get("commit_id")


def preflight_ci2(entry, artifact_value, output_dir):
    started = (utc_now(), time.monotonic())
    artifact = (
        Path(artifact_value).expanduser().resolve() if artifact_value else None
    )
    policy_attested = os.environ.get(CI2_ATTESTATION_ENV, "").lower() == "true"
    observations = {
        "artifact_locator": str(artifact) if artifact else None,
        "artifact_expected_size": POLARMACE_SIZE,
        "artifact_expected_sha256": POLARMACE_SHA256,
        "verified_property": "regular readable file with no Unix write permission bits",
        "administrator_policy_attested": policy_attested,
        "administrator_policy_attestation_source": CI2_ATTESTATION_ENV,
        "mount_immutability_attested": policy_attested,
    }
    status, reason = "PASS", "artifact, dependency identity, and external policy attestation verified"
    if not policy_attested:
        status, reason = "BLOCKED", "administrator-controlled CI-2 policy attestation is missing or false"
    elif artifact is None or not artifact.is_file() or not os.access(artifact, os.R_OK):
        status, reason = "BLOCKED", "required administrator-provided PolarMACE artifact is absent or unreadable"
    else:
        observations["artifact_size"] = artifact.stat().st_size
        if artifact.stat().st_size != POLARMACE_SIZE:
            status, reason = "FAIL", f"artifact size mismatch: expected {POLARMACE_SIZE}, got {artifact.stat().st_size}"
        elif artifact.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
            status, reason = "FAIL", "protected artifact has Unix write permission bits"
        else:
            digest = sha256_file(artifact)
            observations["artifact_sha256"] = digest
            if digest != POLARMACE_SHA256:
                status, reason = "FAIL", f"artifact SHA-256 mismatch: expected {POLARMACE_SHA256}, got {digest}"
    if status == "PASS":
        required = {"mace-torch": "0.3.16", "e3nn": "0.4.4"}
        observations.update({
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
        })
        try:
            installed = {name: importlib.metadata.version(name) for name in required}
            graph_commit = _graph_longrange_commit()
        except importlib.metadata.PackageNotFoundError as exc:
            status, reason = "BLOCKED", f"required optional CPU dependency is missing: {exc}"
        except (json.JSONDecodeError, OSError) as exc:
            status, reason = "FAIL", f"installed dependency identity is malformed: {exc}"
        else:
            observations.update(installed)
            observations["graph_longrange_commit"] = graph_commit
            wrong = {name: value for name, value in installed.items() if value != required[name]}
            if Version(platform.python_version()).release[:2] != (3, 12):
                wrong["python"] = platform.python_version()
            if not (Version("2.10") <= Version(torch.__version__.split("+")[0]) < Version("2.11")):
                wrong["torch"] = torch.__version__
            if graph_commit != GRAPH_LONGRANGE_COMMIT:
                wrong["graph-longrange"] = graph_commit
            if wrong or importlib.util.find_spec("mace") is None:
                status, reason = "FAIL", f"protected CPU dependency preflight failed: {wrong}"
    ended = (utc_now(), time.monotonic())
    return status, reason, artifact, started, ended, observations


def validate_report_semantics(report):
    checks = report.get("checks")
    if not isinstance(checks, list) or not checks:
        raise ValueError("report must contain checks")
    if any(check.get("ci_tier") != report.get("requested_tier") for check in checks):
        raise ValueError("check tier disagrees with requested tier")
    expected, _ = aggregate_status(checks)
    if report.get("overall_status") != expected:
        raise ValueError("aggregate status disagrees with checks")
    if report.get("claim_boundary") != CLAIM_BOUNDARY:
        raise ValueError("report claim boundary is invalid")
    if report.get("overall_status") == "PASS":
        if report.get("dirty_tree") is not False:
            raise ValueError("PASS requires a clean source tree")
        if report.get("source_commit") in {None, "", "unknown"}:
            raise ValueError("PASS requires a known source commit")
        if report.get("requested_tier") == "CI-2":
            preflight = next(
                (check for check in checks
                 if check.get("id") == "ci2-artifact-dependency-preflight"),
                None,
            )
            if not preflight or preflight.get("status") != "PASS" or not (
                preflight.get("observations", {}).get(
                    "administrator_policy_attested"
                ) is True
            ):
                raise ValueError("CI-2 PASS requires an attested preflight")
    for check in checks:
        if check.get("claim_boundary") != CLAIM_BOUNDARY:
            raise ValueError("check claim boundary is invalid")
        if check.get("status") == "PASS" and check.get("exit_code") != 0:
            raise ValueError("PASS check must have exit code zero")
        if check.get("status") in {"PASS", "FAIL"} and not check.get("command"):
            raise ValueError("executed check requires a command")
        if check.get("status") == "PASS" and not (
            check.get("evidence") or check.get("observations")
        ):
            raise ValueError("PASS check requires evidence or observations")
    return True


def run(
    tier, output_dir, artifact_value=None, overwrite=False,
    timeout_seconds=900.0, kill_grace_seconds=2.0,
):
    # Capture authority before creating any evidence output.
    commit, dirty_state = _source_state()
    dirty = dirty_state is not False
    output_dir = Path(output_dir).expanduser().resolve()
    artifact_value = str(Path(artifact_value).expanduser().resolve()) if artifact_value else None
    report_path = output_dir / "validation-report.json"
    if report_path.exists() and not overwrite:
        raise FileExistsError("completed validation report exists; pass --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = utc_now()
    selected = next(entry for entry in CHECK_REGISTRY if entry["ci_tier"] == tier)
    checks = []
    if tier == "CI-2":
        status, reason, artifact, started, ended, observations = preflight_ci2(
            selected, artifact_value, output_dir
        )
        preflight_entry = {
            **selected,
            "id": "ci2-artifact-dependency-preflight",
            "summary": "Pinned artifact and CPU dependency preflight",
        }
        checks.append(_base_check(
            preflight_entry, status, reason, ["internal:ci2-preflight"],
            started, ended, 0 if status == "PASS" else None, [], observations,
        ))
        if status == "PASS":
            checks.append(run_pytest_check(
                selected, output_dir, artifact, timeout_seconds, kill_grace_seconds
            ))
    else:
        checks = [run_pytest_check(
            selected, output_dir, timeout=timeout_seconds, grace=kill_grace_seconds
        )]
    if commit == "unknown" or dirty:
        now = (utc_now(), time.monotonic())
        source_entry = {
            **selected, "id": "controlled-source-state",
            "summary": "Controlled source identity",
        }
        checks.append(_base_check(
            source_entry, "FAIL",
            "controlled evidence requires a known clean source commit",
            ["git", "status", "--porcelain"], now, now, 1, [],
            {"source_commit": commit, "dirty_tree": dirty},
        ))
    overall, exit_code = aggregate_status(checks)
    report = {
        "schema_version": SCHEMA_VERSION, "run_id": str(uuid.uuid4()),
        "requested_tier": tier, "scope": SCOPE, "started_at": started_at,
        "ended_at": utc_now(), "source_commit": commit, "dirty_tree": dirty,
        "platform": platform.platform(), "python": sys.version,
        "overall_status": overall, "checks": checks,
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    validate_report_semantics(report)
    _atomic_json(report_path, report)
    return report, exit_code


def _write_interrupted_marker(output_dir, tier, signum):
    """Atomically preserve terminal evidence when the authority is interrupted."""
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    marker = {
        "schema_version": SCHEMA_VERSION,
        "record_type": "validation-interrupted",
        "utc_time": utc_now(),
        "requested_tier": tier,
        "status": "FAIL",
        "interruption_status": "interrupted",
        "signal": signum,
        "signal_name": signal.Signals(signum).name,
        "scope": SCOPE,
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "readiness_claims": [],
    }
    _atomic_json(output_dir / "validation-interrupted.json", marker)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier")
    parser.add_argument("--output-dir")
    parser.add_argument("--mace-artifact", default=os.environ.get("QCMLFORGE_POLARMACE_ARTIFACT"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--kill-grace-seconds", type=float, default=2.0)
    parser.add_argument("--list-checks", action="store_true")
    args = parser.parse_args(argv)
    if args.list_checks:
        print(json.dumps(CHECK_REGISTRY, indent=2))
        return 0
    if args.tier not in {"CI-0", "CI-1", "CI-2"}:
        parser.error("CI-3, CI-4, and CI-5 are nonlocal and cannot be run by this authority")
    if not args.output_dir:
        parser.error("--output-dir is required")
    if args.timeout_seconds <= 0 or args.kill_grace_seconds < 0:
        parser.error("timeout must be positive and grace nonnegative")

    received_signal = None
    global _RECEIVED_SIGNAL
    _RECEIVED_SIGNAL = None

    def interrupt(signum, _frame):
        nonlocal received_signal
        received_signal = signum
        _RECEIVED_SIGNAL = signum
        raise KeyboardInterrupt

    old_handlers = {
        signum: signal.signal(signum, interrupt)
        for signum in (signal.SIGTERM, signal.SIGINT)
    }
    try:
        try:
            report, exit_code = run(
                args.tier, args.output_dir, args.mace_artifact, args.overwrite,
                args.timeout_seconds, args.kill_grace_seconds,
            )
        except FileExistsError as exc:
            parser.error(str(exc))
        except KeyboardInterrupt:
            signum = received_signal or signal.SIGINT
            _write_interrupted_marker(args.output_dir, args.tier, signum)
            return 128 + signum
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
    print(json.dumps({"overall_status": report["overall_status"], "report": str(Path(args.output_dir) / "validation-report.json")}, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
