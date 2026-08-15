"""Qualification tests for local evidence, protected preflight, and metrics."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import zipfile

import pytest

import scripts.validate_mace_local as validation
import scripts.ci.probe_built_wheel as wheel_probe

ROOT = Path(__file__).resolve().parents[1]


def test_registry_and_claim_boundaries_are_declarative():
    ids = [item["id"] for item in validation.CHECK_REGISTRY]
    assert len(ids) == len(set(ids))
    assert {item["ci_tier"] for item in validation.CHECK_REGISTRY} == {"CI-0", "CI-1", "CI-2"}
    assert all(value is False for value in validation.CLAIM_BOUNDARY.values())


def test_claimed_test_files_support_xunit2_classnames():
    paths = ["tests/test_alpha.py", "tests/test_beta.py"]
    counts = {
        "files": [],
        "class_names": ["tests.test_alpha", "tests.test_beta.TestCase"],
    }
    assert validation._claimed_test_files_executed(paths, counts)
    assert not validation._claimed_test_files_executed(
        [*paths, "tests/test_missing.py"], counts
    )


def test_aggregate_precedence_and_required_skip_not_green():
    def check(status):
        return {"required": True, "status": status}
    assert validation.aggregate_status([check("PASS")]) == ("PASS", 0)
    assert validation.aggregate_status([check("BLOCKED")]) == ("BLOCKED", 2)
    assert validation.aggregate_status([check("FAIL"), check("BLOCKED")]) == ("FAIL", 1)
    assert validation.aggregate_status([check("SKIP")]) == ("FAIL", 1)


def test_report_semantics_reject_false_green_and_tier_mismatch():
    check = {
        "required": True, "status": "FAIL", "ci_tier": "CI-0",
        "command": ["false"], "exit_code": 1, "evidence": [],
        "observations": {}, "claim_boundary": dict(validation.CLAIM_BOUNDARY),
    }
    report = {
        "requested_tier": "CI-0", "overall_status": "PASS", "checks": [check],
        "claim_boundary": dict(validation.CLAIM_BOUNDARY),
    }
    with pytest.raises(ValueError, match="aggregate"):
        validation.validate_report_semantics(report)
    report["overall_status"] = "FAIL"
    check["ci_tier"] = "CI-1"
    with pytest.raises(ValueError, match="tier"):
        validation.validate_report_semantics(report)


def test_report_semantics_reject_pass_source_exit_evidence_and_ci2_attestation():
    check = {
        "id": "ci2-artifact-dependency-preflight", "required": True,
        "status": "PASS", "ci_tier": "CI-2", "command": ["internal"],
        "exit_code": 0, "evidence": [],
        "observations": {"administrator_policy_attested": True},
        "claim_boundary": dict(validation.CLAIM_BOUNDARY),
    }
    report = {
        "requested_tier": "CI-2", "overall_status": "PASS", "checks": [check],
        "source_commit": "a" * 40, "dirty_tree": False,
        "claim_boundary": dict(validation.CLAIM_BOUNDARY),
    }
    assert validation.validate_report_semantics(report)
    for field, value, message in (
        ("dirty_tree", True, "clean"),
        ("source_commit", "unknown", "known"),
    ):
        broken = {**report, field: value}
        with pytest.raises(ValueError, match=message):
            validation.validate_report_semantics(broken)
    check["exit_code"] = 1
    with pytest.raises(ValueError, match="exit code"):
        validation.validate_report_semantics(report)
    check["exit_code"] = 0
    check["observations"] = {}
    check["ci_tier"] = report["requested_tier"] = "CI-0"
    with pytest.raises(ValueError, match="evidence or observations"):
        validation.validate_report_semantics(report)
    check["ci_tier"] = report["requested_tier"] = "CI-2"
    check["observations"] = {"administrator_policy_attested": False}
    with pytest.raises(ValueError, match="attested"):
        validation.validate_report_semantics(report)


def test_published_schema_rejects_contradictory_pass_records():
    schema = json.loads(
        (ROOT / "docs/schemas/mace-local-validation-v1.schema.json").read_text()
    )
    try:
        import jsonschema
    except ImportError:
        # Runtime generation intentionally has no broad JSON Schema dependency;
        # Python semantic contradiction tests above remain authoritative.
        encoded = json.dumps(schema)
        for token in (
            "administrator_policy_attested", "dirty_tree", "source_commit",
            "minProperties", "minContains",
        ):
            assert token in encoded
        return
    now = "2026-01-01T00:00:00Z"
    check = {
        "id": "ci2-artifact-dependency-preflight", "spec_refs": ["V1.2"],
        "ci_tier": "CI-2", "required": True, "status": "PASS",
        "summary": "preflight", "reason": "verified", "command": ["internal"],
        "started_at": now, "ended_at": now, "duration_seconds": 0,
        "exit_code": 0, "evidence": [],
        "observations": {"administrator_policy_attested": True},
        "claim_boundary": dict(validation.CLAIM_BOUNDARY),
    }
    report = {
        "schema_version": validation.SCHEMA_VERSION, "run_id": "test",
        "requested_tier": "CI-2", "scope": validation.SCOPE,
        "started_at": now, "ended_at": now, "source_commit": "a" * 40,
        "dirty_tree": False, "platform": "test", "python": "3.12",
        "overall_status": "PASS", "checks": [check],
        "claim_boundary": dict(validation.CLAIM_BOUNDARY),
    }
    jsonschema.Draft202012Validator(schema).validate(report)
    mutations = [
        lambda value: value.update(dirty_tree=True),
        lambda value: value.update(source_commit="unknown"),
        lambda value: value["checks"][0].update(status="FAIL"),
        lambda value: value["checks"][0].update(ci_tier="CI-1"),
        lambda value: value["checks"][0].update(exit_code=1),
        lambda value: value["checks"][0].update(evidence=[], observations={}),
        lambda value: value["checks"][0]["observations"].update(
            administrator_policy_attested=False
        ),
    ]
    for mutate in mutations:
        broken = json.loads(json.dumps(report))
        mutate(broken)
        assert list(jsonschema.Draft202012Validator(schema).iter_errors(broken))


def test_ci2_missing_attestation_is_blocked_and_recorded(tmp_path, monkeypatch):
    monkeypatch.delenv("QCMLFORGE_CI2_POLICY_ATTESTED", raising=False)
    entry = next(item for item in validation.CHECK_REGISTRY if item["ci_tier"] == "CI-2")
    status, reason, *_, observations = validation.preflight_ci2(entry, None, tmp_path)
    assert status == "BLOCKED" and "attestation" in reason
    assert observations["administrator_policy_attested"] is False


def test_ci2_missing_is_blocked_and_wrong_digest_fails_without_pytest(tmp_path, monkeypatch):
    monkeypatch.setenv("QCMLFORGE_CI2_POLICY_ATTESTED", "true")
    monkeypatch.setattr(validation, "POLARMACE_SIZE", 3)
    entry = next(item for item in validation.CHECK_REGISTRY if item["ci_tier"] == "CI-2")
    status, *_ = validation.preflight_ci2(entry, None, tmp_path)
    assert status == "BLOCKED"
    artifact = tmp_path / "bad.model"
    artifact.write_bytes(b"bad")
    artifact.chmod(0o444)
    status, reason, *_ = validation.preflight_ci2(entry, str(artifact), tmp_path)
    assert status == "FAIL" and "SHA-256" in reason


def _pinned_preflight(tmp_path, monkeypatch):
    artifact = tmp_path / "tiny.model"
    artifact.write_bytes(b"ok")
    artifact.chmod(0o444)
    monkeypatch.setenv("QCMLFORGE_CI2_POLICY_ATTESTED", "true")
    monkeypatch.setattr(validation, "POLARMACE_SIZE", 2)
    monkeypatch.setattr(validation, "POLARMACE_SHA256", validation.sha256_file(artifact))
    monkeypatch.setattr(validation.platform, "python_version", lambda: "3.12.9")
    monkeypatch.setattr(validation.torch, "__version__", "2.10.0")
    monkeypatch.setattr(validation.importlib.metadata, "version", lambda name: {
        "mace-torch": "0.3.16", "e3nn": "0.4.4"
    }[name])
    monkeypatch.setattr(validation, "_graph_longrange_commit", lambda: validation.GRAPH_LONGRANGE_COMMIT)
    monkeypatch.setattr(validation.importlib.util, "find_spec", lambda name: object())
    entry = next(item for item in validation.CHECK_REGISTRY if item["ci_tier"] == "CI-2")
    return entry, artifact


def test_ci2_exact_pins_pass_with_policy_attested(tmp_path, monkeypatch):
    entry, artifact = _pinned_preflight(tmp_path, monkeypatch)
    status, _, *_, observations = validation.preflight_ci2(entry, artifact, tmp_path)
    assert status == "PASS"
    assert observations["administrator_policy_attested"] is True


@pytest.mark.parametrize("wrong", ["python", "torch", "mace-torch", "e3nn", "graph"])
def test_ci2_wrong_dependency_identity_fails(tmp_path, monkeypatch, wrong):
    entry, artifact = _pinned_preflight(tmp_path, monkeypatch)
    if wrong == "python":
        monkeypatch.setattr(validation.platform, "python_version", lambda: "3.11.9")
    elif wrong == "torch":
        monkeypatch.setattr(validation.torch, "__version__", "2.9.0")
    elif wrong == "graph":
        monkeypatch.setattr(validation, "_graph_longrange_commit", lambda: "wrong")
    else:
        monkeypatch.setattr(validation.importlib.metadata, "version", lambda name: "9.9" if name == wrong else {"mace-torch": "0.3.16", "e3nn": "0.4.4"}[name])
    status, reason, *_ = validation.preflight_ci2(entry, artifact, tmp_path)
    assert status == "FAIL" and "dependency preflight" in reason


def test_ci2_missing_dependency_is_blocked(tmp_path, monkeypatch):
    entry, artifact = _pinned_preflight(tmp_path, monkeypatch)
    def missing(_name):
        raise validation.importlib.metadata.PackageNotFoundError("missing")
    monkeypatch.setattr(validation.importlib.metadata, "version", missing)
    status, reason, *_ = validation.preflight_ci2(entry, artifact, tmp_path)
    assert status == "BLOCKED" and "missing" in reason


def test_offline_environment_scrubs_inherited_download_controls(monkeypatch):
    for name in ("QCMLFORGE_AUTO_DOWNLOAD_PRETRAINED", "HF_ENDPOINT", "TORCH_HOME"):
        monkeypatch.setenv(name, "download-enabled")
    env = validation._offline_environment()
    assert env["QCMLFORGE_AUTO_DOWNLOAD_PRETRAINED"] == "0"
    assert env["QCMLFORGE_MACE_OFFLINE"] == "1"
    assert "HF_ENDPOINT" not in env and "TORCH_HOME" not in env


def test_cli_rejects_nonlocal_tiers_and_lists_registry(tmp_path):
    command = [sys.executable, str(ROOT / "scripts/validate_mace_local.py")]
    listed = subprocess.run([*command, "--list-checks"], capture_output=True, text=True)
    assert listed.returncode == 0 and "ci2-pinned-polarmace-cpu" in listed.stdout
    rejected = subprocess.run([*command, "--tier", "CI-4", "--output-dir", str(tmp_path)], capture_output=True, text=True)
    assert rejected.returncode != 0 and "nonlocal" in rejected.stderr


def _run_metrics(tmp_path, child, *options):
    output = tmp_path / "metrics.json"
    result = subprocess.run([
        sys.executable, str(ROOT / "scripts/run_with_metrics.py"),
        "--output", str(output), *options, "--", *child,
    ])
    return result, json.loads(output.read_text())


def test_metrics_success_nonzero_launch_failure_and_capabilities(tmp_path):
    result, record = _run_metrics(tmp_path, [sys.executable, "-c", "print('ok')"])
    assert result.returncode == 0 and record["schema_version"].endswith("v1")
    assert record["child_pid"] and record["process_group_id"]
    assert all(value is False for value in record["capabilities"].values())
    result, record = _run_metrics(tmp_path, [sys.executable, "-c", "raise SystemExit(7)"])
    assert result.returncode == 7 and record["exit_code"] == 7
    result, record = _run_metrics(tmp_path, [str(tmp_path / "does-not-exist")])
    assert result.returncode == 127 and record["launch_error"]


def test_metrics_forwarded_signal_is_recorded_and_bounded(tmp_path):
    output = tmp_path / "signal-metrics.json"
    wrapper = subprocess.Popen([
        sys.executable, str(ROOT / "scripts/run_with_metrics.py"),
        "--output", str(output), "--kill-grace-seconds", "0.1", "--",
        sys.executable, "-c", "import time; time.sleep(30)",
    ])
    time.sleep(0.2)
    wrapper.send_signal(signal.SIGTERM)
    assert wrapper.wait(timeout=5) == 128 + signal.SIGTERM
    record = json.loads(output.read_text())
    assert record["forwarded_signal"] == signal.SIGTERM
    assert record["terminating_signal"] == signal.SIGTERM
    assert record["outcome"] == "signal"


@pytest.mark.parametrize("forward_term", [False, True])
def test_metrics_kills_term_ignoring_descendants(tmp_path, forward_term):
    output = tmp_path / "descendant-metrics.json"
    child_code = (
        "import signal,subprocess,time; "
        "subprocess.Popen([%r,'-c',"
        "'import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)']); "
        "time.sleep(30)" % sys.executable
    )
    command = [
        sys.executable, str(ROOT / "scripts/run_with_metrics.py"),
        "--output", str(output), "--kill-grace-seconds", "0.1",
        "--timeout-seconds", "5" if forward_term else "0.2", "--",
        sys.executable, "-c", child_code,
    ]
    wrapper = subprocess.Popen(command)
    if forward_term:
        time.sleep(0.3)
        wrapper.send_signal(signal.SIGTERM)
    expected = 128 + signal.SIGTERM if forward_term else 124
    assert wrapper.wait(timeout=5) == expected
    record = json.loads(output.read_text())
    with pytest.raises(ProcessLookupError):
        os.killpg(record["process_group_id"], 0)
    assert record["received_signal"] == (signal.SIGTERM if forward_term else None)
    assert record["escalation_signal"] == signal.SIGKILL


def test_validator_bounded_runner_kills_timeout_descendants(tmp_path):
    child_code = (
        "import signal,subprocess,time; "
        "subprocess.Popen([%r,'-c',"
        "'import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)']); "
        "time.sleep(30)" % sys.executable
    )
    stdout = tmp_path / "validator-stdout"
    stderr = tmp_path / "validator-stderr"
    with stdout.open("wb") as out, stderr.open("wb") as err:
        returncode, timed_out, interrupted = validation._run_bounded(
            [sys.executable, "-c", child_code], cwd=ROOT,
            env=os.environ.copy(), stdout=out, stderr=err,
            timeout=0.2, grace=0.1,
        )
    assert returncode < 0 and timed_out and interrupted is None
    # The subreaper-backed runner would leave waitable descendants here if it
    # had only reaped the group leader.
    with pytest.raises(ChildProcessError):
        os.waitpid(-1, os.WNOHANG)


def test_wheel_probe_rejects_foundation_digest_under_renamed_extension(
    tmp_path, monkeypatch
):
    wheel = tmp_path / "probe.whl"
    payload = b"renamed forbidden content"
    monkeypatch.setattr(
        wheel_probe, "POLARMACE_SHA256", __import__("hashlib").sha256(payload).hexdigest()
    )
    with zipfile.ZipFile(wheel, "w") as archive:
        for name in wheel_probe.REQUIRED:
            archive.writestr(name, b"approved d3" if name.endswith("reference-c6.pt") else b"x")
        archive.writestr("apnet_pt/data/unrelated.bin", payload)
    with pytest.raises(RuntimeError, match="foundation artifact digest"):
        wheel_probe.inspect_wheel(wheel)


def test_protected_workflow_is_unconditionally_disabled_with_trusted_ref_defense():
    workflow = (ROOT / ".github/workflows/mace-integration.yml").read_text()
    assert "if: ${{ false && github.ref == 'refs/heads/main' }}" in workflow
    assert "inputs:" not in workflow
    assert "QCMLFORGE_CI2_POLICY_ATTESTED" in workflow
    assert "ref: ${{ github.sha }}" in workflow
    docs = (ROOT / "docs/mace-apnet-environment.md").read_text()
    for phrase in ("required reviewers", "read-only approved mount", "runner cleanup"):
        assert phrase in docs


def test_authority_sigterm_during_preflight_writes_terminal_marker(tmp_path):
    output = tmp_path / "evidence"
    code = (
        "import time\nimport scripts.validate_mace_local as v\n"
        "def delayed(*a, **k):\n    print('PREFLIGHT', flush=True)\n    time.sleep(30)\n"
        "v.preflight_ci2 = delayed\n"
        f"raise SystemExit(v.main(['--tier','CI-2','--output-dir',{str(output)!r}]))\n"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", code], cwd=ROOT, stdout=subprocess.PIPE, text=True
    )
    assert process.stdout.readline().strip() == "PREFLIGHT"
    process.send_signal(signal.SIGTERM)
    assert process.wait(timeout=5) != 0
    marker = json.loads((output / "validation-interrupted.json").read_text())
    assert marker["status"] == "FAIL"
    assert marker["interruption_status"] == "interrupted"
    assert marker["requested_tier"] == "CI-2"
    assert marker["signal"] == signal.SIGTERM
    assert all(value is False for value in marker["claim_boundary"].values())


def test_generated_evidence_paths_are_ignored():
    for path in ("artifacts/mace-local/ci0/validation-report.json", "junit-ordinary.xml"):
        result = subprocess.run(["git", "check-ignore", "-q", path], cwd=ROOT)
        assert result.returncode == 0


def test_metrics_timeout_cleans_process_group(tmp_path):
    result, record = _run_metrics(
        tmp_path, [sys.executable, "-c", "import time; time.sleep(30)"],
        "--timeout-seconds", "0.15", "--kill-grace-seconds", "0.1",
        "--sample-interval-seconds", "0.02",
    )
    assert result.returncode == 124 and record["timed_out"] is True
    with pytest.raises(ProcessLookupError):
        os.killpg(record["process_group_id"], 0)
