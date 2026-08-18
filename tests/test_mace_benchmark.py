import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "slurm" / "benchmark.py"


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return path


def _benchmark_inputs(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir(parents=True)
    files = {}
    for name in (
        "1600K_train_dimers-fixed.pkl",
        "1600K_test_dimers-fixed.pkl",
        "monomers_ap3_spec_1_pbe0.pkl",
        "monomers_ap3_spec_5_pbe0.pkl",
        "MACE-POLAR-1-S.model",
        "am.pt",
        "param1.pt",
        "param2.pt",
        "physics.json",
    ):
        path = inputs / name
        path.write_bytes(f"fixture:{name}".encode())
        files[name] = path

    from apnet_pt.mace.schema import PhysicsConfig

    physics = PhysicsConfig()
    physics_record = asdict(physics)
    physics_record["physics_hash"] = physics.physics_hash
    _write_json(files["physics.json"], physics_record)

    provenance = {
        "schema_version": 1,
        "status": "approved",
        "approved_by": "test",
        "pair": {
            "spec_type": 2,
            "target_method": "SAPT0/jun-cc-pVDZ",
            "component_order": ["elst", "exch", "indu", "disp"],
            "columns": [
                "Total_aug",
                "Elst_aug",
                "Exch_aug",
                "Ind_aug",
                "Disp_aug",
            ],
            "units": "kcal/mol",
            "license": "approved-internal-test-fixture",
            "files": {
                "train": _sha(files["1600K_train_dimers-fixed.pkl"]),
                "test": _sha(files["1600K_test_dimers-fixed.pkl"]),
            },
        },
        "atomic": {
            "target_method": "PBE0/MBIS",
            "properties": ["q", "mu", "quadrupole"],
            "units": ["e", "e·bohr", "e·bohr²"],
            "license": "approved-internal-test-fixture",
            "files": {
                "spec_1": _sha(files["monomers_ap3_spec_1_pbe0.pkl"]),
                "spec_5": _sha(files["monomers_ap3_spec_5_pbe0.pkl"]),
            },
        },
    }
    provenance_path = _write_json(inputs / "provenance.json", provenance)

    split = {
        "schema_version": 1,
        "pair_spec_type": 2,
        "index_basis": {
            "pair": "processed-ap3-fused-spec2-order-v1",
            "atomic": "processed-pbe0-mbis-order-v1",
        },
        "leakage_audit": {
            "status": "passed",
            "group_key": "unordered-monomer-pair-identity",
            "audited_by": "test",
            "cross_dataset_policy": "no-pair-test-monomer-in-atomic-train",
            "cross_dataset_overlap_count": 0,
        },
        "pair": {
            "train": {
                "source": "train",
                "indices": [0, 1, 2],
                "group_ids": ["pair-a", "pair-b", "pair-c"],
            },
            "validation": {
                "source": "train",
                "indices": [3],
                "group_ids": ["pair-d"],
            },
            "test": {
                "source": "test",
                "indices": [0, 1],
                "group_ids": ["pair-e", "pair-f"],
            },
        },
        "atomic": {
            "train": {
                "spec_1": {
                    "indices": [0, 1],
                    "group_ids": ["atom-a", "atom-b"],
                },
                "spec_5": {"indices": [0], "group_ids": ["atom-c"]},
            },
            "validation": {
                "spec_1": {"indices": [2], "group_ids": ["atom-d"]},
                "spec_5": {"indices": [1], "group_ids": ["atom-e"]},
            },
            "test": {
                "spec_1": {"indices": [3], "group_ids": ["atom-f"]},
                "spec_5": {"indices": [2], "group_ids": ["atom-g"]},
            },
        },
    }
    split_path = _write_json(inputs / "split.json", split)

    def record(path):
        return {"path": str(path), "sha256": _sha(path)}

    config = {
        "schema_version": 1,
        "benchmark_id": "ap3-mace-production-test",
        "output_root": str(tmp_path / "benchmark-runs"),
        "dataset": {
            "pair_spec_type": 2,
            "pair_files": {
                "train": record(files["1600K_train_dimers-fixed.pkl"]),
                "test": record(files["1600K_test_dimers-fixed.pkl"]),
            },
            "atomic_files": {
                "spec_1": record(files["monomers_ap3_spec_1_pbe0.pkl"]),
                "spec_5": record(files["monomers_ap3_spec_5_pbe0.pkl"]),
            },
            "split_manifest": record(split_path),
            "provenance_manifest": record(provenance_path),
            "target_method": "SAPT0/jun-cc-pVDZ",
            "atomic_target_method": "PBE0/MBIS",
        },
        "artifacts": {
            "mace": record(files["MACE-POLAR-1-S.model"]),
            "legacy_atom_model": record(files["am.pt"]),
            "legacy_parameter_model": record(files["param1.pt"]),
            "legacy_parameter_model_2": record(files["param2.pt"]),
            "physics_config": record(files["physics.json"]),
        },
        "training": {
            "seeds": [0, 1, 2],
            "epochs": 50,
            "atomic_epochs": 100,
            "batch_size": 16,
            "atomic_batch_size": 16,
            "learning_rate": 0.0005,
            "end_learning_rate": 0.00001,
            "include_total_mse": True,
            "early_stopping_patience": 10,
            "num_workers": 4,
        },
        "routes": ["BASE", "H1", "H2", "DirectPolar", "AtomHead"],
        "slurm": {
            "account": "acct",
            "partition": "gpu",
            "time": "12:00:00",
            "cpus_per_task": 8,
            "memory": "64G",
            "gpus_per_task": 1,
            "python": sys.executable,
        },
    }
    config_path = _write_json(tmp_path / "benchmark.json", config)
    return config_path, config, files


def _run(*args, env=None):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *map(str, args)],
        cwd=ROOT,
        env={**os.environ, **(env or {})},
        capture_output=True,
        text=True,
        check=False,
    )


def _load_module():
    spec = importlib.util.spec_from_file_location("mace_benchmark_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_generate_locks_inputs_and_writes_dependency_array_groups(tmp_path):
    config_path, config, _ = _benchmark_inputs(tmp_path)
    result = _run("--config", config_path)
    assert result.returncode == 0, result.stderr
    assert "generated" in result.stdout.lower()
    assert "not submitted" in result.stdout.lower()

    root = Path(config["output_root"]) / config["benchmark_id"]
    lock = json.loads((root / "benchmark.lock.json").read_text())
    assert lock["status"] == "LOCKED"
    assert len(lock["benchmark_hash"]) == 64
    assert lock["config"]["dataset"]["pair_spec_type"] == 2
    assert lock["config"]["training"]["seeds"] == [0, 1, 2]
    assert lock["config"]["training"]["optimizer"] == "Adam"
    assert lock["config"]["training"]["checkpoint_selection"] == (
        "minimum-validation-loss"
    )
    assert lock["config"]["training"]["model_hyperparameters"]["n_neuron"] == 128

    jobs = json.loads((root / "submission" / "jobs.json").read_text())
    assert [job["group"] for job in jobs] == [
        "dataset",
        "prepare",
        "baseline",
        "hybrid",
        "atomic",
        "polar",
        "report",
    ]
    counts = {job["group"]: job["task_count"] for job in jobs}
    assert counts == {
        "dataset": 1,
        "prepare": 1,
        "baseline": 3,
        "hybrid": 6,
        "atomic": 6,
        "polar": 6,
        "report": 1,
    }
    dependencies = {job["group"]: job["depends_on"] for job in jobs}
    assert dependencies == {
        "dataset": [],
        "prepare": ["dataset"],
        "baseline": ["dataset"],
        "hybrid": ["prepare"],
        "atomic": ["prepare"],
        "polar": ["atomic"],
        "report": ["baseline", "hybrid", "polar"],
    }
    assert jobs[-1]["dependency_mode"] == "afterany"
    for job in jobs:
        script = Path(job["script"])
        assert script.is_file()
        syntax = subprocess.run(
            ["bash", "-n", str(script)], capture_output=True, text=True, check=False
        )
        assert syntax.returncode == 0, syntax.stderr
        text = script.read_text()
        assert "#SBATCH --ntasks=1" in text
        if job["group"] == "report":
            assert "#SBATCH --gpus-per-task" not in text
        else:
            assert "#SBATCH --gpus-per-task=1" in text
        assert "--run-task" in text
        if job["task_count"] > 1:
            assert f"#SBATCH --array=0-{job['task_count'] - 1}" in text


def test_checked_json_schemas_accept_the_frozen_config_and_split(tmp_path):
    jsonschema = pytest.importorskip("jsonschema")
    config_path, config, _ = _benchmark_inputs(tmp_path)
    split = json.loads(Path(config["dataset"]["split_manifest"]["path"]).read_text())
    config_schema = json.loads(
        (ROOT / "docs" / "schemas" / "mace-benchmark-v1.schema.json").read_text()
    )
    split_schema = json.loads(
        (ROOT / "docs" / "schemas" / "mace-benchmark-split-v1.schema.json").read_text()
    )
    provenance_schema = json.loads(
        (
            ROOT / "docs" / "schemas" / "mace-benchmark-provenance-v1.schema.json"
        ).read_text()
    )
    jsonschema.Draft202012Validator(config_schema).validate(
        json.loads(config_path.read_text())
    )
    jsonschema.Draft202012Validator(split_schema).validate(split)
    provenance = json.loads(
        Path(config["dataset"]["provenance_manifest"]["path"]).read_text()
    )
    jsonschema.Draft202012Validator(provenance_schema).validate(provenance)


def test_generation_fails_closed_on_digest_or_split_audit(tmp_path):
    config_path, config, files = _benchmark_inputs(tmp_path)
    files["am.pt"].write_bytes(b"changed")
    result = _run("--config", config_path)
    assert result.returncode != 0
    assert "sha-256 mismatch" in result.stderr.lower()

    config_path, config, _ = _benchmark_inputs(tmp_path / "second")
    split_path = Path(config["dataset"]["split_manifest"]["path"])
    split = json.loads(split_path.read_text())
    split["leakage_audit"]["status"] = "pending"
    _write_json(split_path, split)
    config["dataset"]["split_manifest"]["sha256"] = _sha(split_path)
    _write_json(config_path, config)
    result = _run("--config", config_path)
    assert result.returncode != 0
    assert "leakage audit" in result.stderr.lower()


def test_worker_records_missing_cuda_as_blocked(tmp_path):
    config_path, config, _ = _benchmark_inputs(tmp_path)
    assert _run("--config", config_path).returncode == 0
    root = Path(config["output_root"]) / config["benchmark_id"]
    result = _run(
        "--run-task",
        "--lock",
        root / "benchmark.lock.json",
        "--task-file",
        root / "submission" / "baseline.tasks.json",
        "--task-index",
        "0",
        env={"CUDA_VISIBLE_DEVICES": ""},
    )
    assert result.returncode == 2
    record = json.loads((root / "results" / "BASE-seed0.json").read_text())
    assert record["status"] == "BLOCKED"
    assert record["termination_reason"] == "exception"
    assert "require CUDA" in record["error"]["message"]


def test_submit_uses_arrays_and_preserves_baseline_independence(tmp_path):
    config_path, config, _ = _benchmark_inputs(tmp_path)
    fake = tmp_path / "sbatch"
    calls = tmp_path / "sbatch.calls"
    counter = tmp_path / "counter"
    fake.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f'echo "$*" >> {calls}\n'
        f"n=$(cat {counter} 2>/dev/null || echo 100)\n"
        "n=$((n+1))\n"
        f"echo $n > {counter}\n"
        "echo Submitted batch job $n\n"
    )
    fake.chmod(0o755)
    result = _run("--config", config_path, "--submit", "--sbatch-bin", fake)
    assert result.returncode == 0, result.stderr
    submitted = calls.read_text().splitlines()
    assert len(submitted) == 7
    assert "--dependency" not in submitted[0]
    assert "--dependency=afterok:101" in submitted[1]
    assert "--dependency=afterok:101" in submitted[2]
    assert "--dependency=afterok:102" in submitted[3]
    assert "--dependency=afterok:102" in submitted[4]
    assert "--dependency=afterok:105" in submitted[5]
    assert "--dependency=afterany:103:104:106" in submitted[6]

    root = Path(config["output_root"]) / config["benchmark_id"]
    receipt = json.loads((root / "submission" / "submitted.json").read_text())
    assert receipt["status"] == "SUBMITTED"
    assert receipt["job_ids"] == {
        "dataset": "101",
        "prepare": "102",
        "baseline": "103",
        "hybrid": "104",
        "atomic": "105",
        "polar": "106",
        "report": "107",
    }


def _result(lock, route, seed, status="PASS", epochs=50):
    scale = {"BASE": 1.0, "H1": 0.9, "H2": 0.8, "DirectPolar": 0.85, "AtomHead": 0.82}[
        route
    ]
    return {
        "schema_version": 1,
        "benchmark_hash": lock["benchmark_hash"],
        "route": route,
        "seed": seed,
        "status": status,
        "epochs_requested": 50,
        "epochs_completed": epochs,
        "metrics": {
            "test": {
                "total": {"mae": scale, "rmse": scale * 1.2},
                "elst": {"mae": scale, "rmse": scale * 1.1},
                "exch": {"mae": scale, "rmse": scale * 1.1},
                "indu": {"mae": scale, "rmse": scale * 1.1},
                "disp": {"mae": scale, "rmse": scale * 1.1},
            }
        },
        "resources": {"elapsed_seconds": 12.0, "peak_rss_mb": 100.0},
        "history": [
            {
                "epoch": epoch,
                "train_loss": 2.0 / epoch,
                "validation_loss": 2.5 / epoch,
            }
            for epoch in range(1, epochs + 1)
        ],
    }


def _atomic_result(lock, property_mode, seed, epochs=100):
    scale = 0.5 if property_mode == "learned" else 0.7
    return {
        "schema_version": 1,
        "benchmark_hash": lock["benchmark_hash"],
        "property_mode": property_mode,
        "seed": seed,
        "status": "PASS",
        "epochs_requested": 100,
        "epochs_completed": epochs,
        "termination_reason": "epoch_budget",
        "metrics": {
            split: {
                name: {"mae": scale, "rmse": scale * 1.2}
                for name in ("q", "mu", "quadrupole")
            }
            for split in ("train", "validation", "test")
        },
        "history": [
            {
                "epoch": epoch,
                "train_loss": 2.0 / epoch,
                "validation_loss": 2.5 / epoch,
            }
            for epoch in range(1, epochs + 1)
        ],
    }


def test_plot_marks_partial_then_complete_and_writes_machine_readable_tables(tmp_path):
    pytest.importorskip("matplotlib")
    config_path, config, _ = _benchmark_inputs(tmp_path)
    assert _run("--config", config_path).returncode == 0
    root = Path(config["output_root"]) / config["benchmark_id"]
    lock = json.loads((root / "benchmark.lock.json").read_text())
    result_dir = root / "results"
    result_dir.mkdir(exist_ok=True)
    _write_json(result_dir / "BASE-seed0.json", _result(lock, "BASE", 0))
    _write_json(result_dir / "H1-seed0.json", _result(lock, "H1", 0, epochs=21))

    result = _run("--config", config_path, "--plot")
    assert result.returncode == 0, result.stderr
    summary = json.loads((root / "plots" / "summary.json").read_text())
    assert summary["completion_status"] == "PARTIAL"
    assert summary["completed_runs"] == 1
    assert summary["reported_runs"] == 2
    assert summary["expected_runs"] == 15
    status_text = (root / "plots" / "STATUS.txt").read_text()
    assert "PARTIAL: pair 1/15" in status_text
    assert "atomic 0/6" in status_text
    assert (root / "plots" / "accuracy.png").is_file()
    assert (root / "plots" / "components.png").is_file()
    assert (root / "plots" / "learning_curves.png").is_file()
    assert (root / "plots" / "baseline_delta.png").is_file()
    assert (root / "plots" / "resources.png").is_file()
    assert (root / "plots" / "atomic_properties.png").is_file()
    assert (root / "plots" / "metrics.csv").is_file()
    assert (root / "plots" / "atomic_metrics.csv").is_file()
    assert (root / "plots" / "aggregate.csv").is_file()
    assert (root / "plots" / "baseline_deltas.csv").is_file()

    for route in config["routes"]:
        for seed in config["training"]["seeds"]:
            _write_json(
                result_dir / f"{route}-seed{seed}.json",
                _result(lock, route, seed),
            )
    for mode in ("direct-completion", "learned"):
        for seed in config["training"]["seeds"]:
            atomic_dir = root / "atomic" / mode / f"seed-{seed}"
            atomic_dir.mkdir(parents=True, exist_ok=True)
            _write_json(
                atomic_dir / "result.json",
                _atomic_result(lock, mode, seed),
            )
    result = _run("--config", config_path, "--plot")
    assert result.returncode == 0, result.stderr
    summary = json.loads((root / "plots" / "summary.json").read_text())
    assert summary["completion_status"] == "COMPLETE"
    assert summary["completed_runs"] == 15
    assert summary["atomic_completed_runs"] == 6
    status_text = (root / "plots" / "STATUS.txt").read_text()
    assert "COMPLETE: pair 15/15" in status_text
    assert "atomic 6/6" in status_text


def test_result_validation_rejects_wrong_identity_and_atomic_metrics_are_explicit(
    tmp_path,
):
    module = _load_module()
    config_path, config, _ = _benchmark_inputs(tmp_path)
    assert _run("--config", config_path).returncode == 0
    root = Path(config["output_root"]) / config["benchmark_id"]
    lock = json.loads((root / "benchmark.lock.json").read_text())
    record = _result(lock, "BASE", 0)
    record["benchmark_hash"] = "0" * 64
    with pytest.raises(ValueError, match="benchmark hash"):
        module.validate_result(record, lock)
    assert module.ATOMIC_METRICS == ("q", "mu", "quadrupole")


def test_benchmark_metric_helpers_cover_energy_and_mbis_properties():
    from apnet_pt.mace.schema import AtomicPropertyBundle
    from apnet_pt.training.mace_benchmark import atomic_metrics, energy_metrics

    target = torch.tensor([[1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0]])
    prediction = target + torch.tensor([[1.0, -1.0, 2.0, -2.0], [-1.0, 1.0, -2.0, 2.0]])
    metrics = energy_metrics(prediction, target)
    assert metrics["elst"]["mae"] == pytest.approx(1.0)
    assert metrics["indu"]["rmse"] == pytest.approx(2.0)
    assert metrics["total"]["mae"] == pytest.approx(0.0)

    zeros = torch.zeros(2, 1)
    target_bundle = AtomicPropertyBundle(
        q=zeros,
        mu=torch.zeros(2, 3),
        quadrupole=torch.zeros(2, 3, 3),
        hfvr=torch.ones(2, 1),
        valence_width=torch.ones(2, 1),
        alpha=torch.ones(2, 1),
        damping=torch.ones(2, 1),
    )
    prediction_bundle = AtomicPropertyBundle(
        q=torch.ones(2, 1),
        mu=torch.ones(2, 3) * 2,
        quadrupole=torch.ones(2, 3, 3) * 3,
        hfvr=torch.ones(2, 1),
        valence_width=torch.ones(2, 1),
        alpha=torch.ones(2, 1),
        damping=torch.ones(2, 1),
    )
    properties = atomic_metrics(prediction_bundle, target_bundle)
    assert set(properties) == {"q", "mu", "quadrupole"}
    assert properties["q"]["mae"] == pytest.approx(1.0)
    assert properties["mu"]["rmse"] == pytest.approx(2.0)
    assert properties["quadrupole"]["mae"] == pytest.approx(3.0)


def test_atomic_dataset_preserves_multiplicity_for_mace():
    qcel = pytest.importorskip("qcelemental")
    from apnet_pt.atomic_datasets import atomic_collate_update, qcel_mon_to_pyg_data

    doublet = qcel.models.Molecule.from_data(
        "0 2\nH 0 0 0\nunits angstrom\nno_com\nno_reorient"
    )
    singlet = qcel.models.Molecule.from_data(
        "0 1\nHe 0 0 0\nunits angstrom\nno_com\nno_reorient"
    )
    records = []
    for molecule in (doublet, singlet):
        data = qcel_mon_to_pyg_data(molecule, full_indices=True)
        data.charges = torch.zeros(data.x.numel())
        data.dipoles = torch.zeros(data.x.numel(), 3)
        data.quadrupoles = torch.zeros(data.x.numel(), 3, 3)
        records.append(data)
    batch = atomic_collate_update(records)
    assert batch.total_spin.dtype.is_floating_point
    assert batch.total_spin.tolist() == [2.0, 1.0]


def test_cli_help_describes_generation_submission_collection_and_worker():
    result = _run("--help")
    assert result.returncode == 0
    for flag in ("--config", "--submit", "--plot", "--run-task"):
        assert flag in result.stdout
