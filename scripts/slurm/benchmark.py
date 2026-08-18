#!/usr/bin/env python3
"""Generate, submit, execute, collect, and plot MACE/AP3D3 benchmarks.

Generation is the default and never submits work.  ``--submit`` is the only
submission switch.  Every generated array task reads an immutable lock file
and a task table, so all routes use exactly the same scientific and training
configuration.  Result plots carry an explicit COMPLETE/PARTIAL watermark.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
ROUTES = ("BASE", "H1", "H2", "DirectPolar", "AtomHead")
COMPONENTS = ("total", "elst", "exch", "indu", "disp")
ATOMIC_METRICS = ("q", "mu", "quadrupole")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


class BenchmarkError(RuntimeError):
    """Fail-closed benchmark contract error."""


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def _write_json(path: Path, value: Any) -> None:
    _atomic_write(path, json.dumps(value, indent=2, sort_keys=True).encode() + b"\n")


def _read_json(path: str | Path, description: str) -> Any:
    source = Path(path)
    if not source.is_file():
        raise BenchmarkError(f"{description} is missing: {source}")
    try:
        return json.loads(source.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise BenchmarkError(f"unable to read {description}: {source}: {exc}") from exc


def _require_mapping(value: Any, description: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BenchmarkError(f"{description} must be a JSON object")
    return value


def _require_keys(
    value: Mapping[str, Any], keys: Sequence[str], description: str
) -> None:
    missing = sorted(set(keys) - set(value))
    if missing:
        raise BenchmarkError(f"{description} is missing: {', '.join(missing)}")


def _verify_record(record: Any, description: str) -> dict[str, str]:
    item = _require_mapping(record, description)
    _require_keys(item, ("path", "sha256"), description)
    path = Path(str(item["path"])).expanduser().resolve()
    expected = str(item["sha256"]).lower()
    if not SHA256_RE.fullmatch(expected):
        raise BenchmarkError(f"{description} SHA-256 must be 64 lowercase hex digits")
    if not path.is_file():
        raise BenchmarkError(f"{description} is missing: {path}")
    actual = _sha256_file(path)
    if actual != expected:
        raise BenchmarkError(
            f"{description} SHA-256 mismatch: expected {expected}, got {actual}"
        )
    return {"path": str(path), "sha256": expected}


def _positive_int(value: Any, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise BenchmarkError(f"{description} must be a positive integer")
    return value


def _positive_float(value: Any, description: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BenchmarkError(f"{description} must be a positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise BenchmarkError(f"{description} must be a positive finite number")
    return result


def _validate_split_manifest(value: Any) -> Mapping[str, Any]:
    split = _require_mapping(value, "split manifest")
    _require_keys(
        split,
        (
            "schema_version",
            "pair_spec_type",
            "index_basis",
            "leakage_audit",
            "pair",
            "atomic",
        ),
        "split manifest",
    )
    if split["schema_version"] != 1 or split["pair_spec_type"] != 2:
        raise BenchmarkError(
            "split manifest must use schema_version=1 and pair_spec_type=2"
        )
    if split["index_basis"] != {
        "pair": "processed-ap3-fused-spec2-order-v1",
        "atomic": "processed-pbe0-mbis-order-v1",
    }:
        raise BenchmarkError("split manifest index basis is unsupported")
    audit = _require_mapping(split["leakage_audit"], "split leakage audit")
    if audit.get("status") != "passed":
        raise BenchmarkError("split leakage audit must have status='passed'")
    if not audit.get("group_key") or not audit.get("audited_by"):
        raise BenchmarkError("split leakage audit requires group_key and audited_by")
    if (
        audit.get("cross_dataset_policy") != "no-pair-test-monomer-in-atomic-train"
        or audit.get("cross_dataset_overlap_count") != 0
    ):
        raise BenchmarkError(
            "split leakage audit must attest zero atomic-train/pair-test "
            "monomer overlap under the canonical cross-dataset policy"
        )
    pair = _require_mapping(split["pair"], "pair split membership")
    if set(pair) != {"train", "validation", "test"}:
        raise BenchmarkError("pair split membership must define train/validation/test")
    seen: dict[str, set[int]] = {}
    split_groups: dict[str, set[str]] = {}
    for name, member in pair.items():
        member = _require_mapping(member, f"pair {name} membership")
        _require_keys(
            member,
            ("source", "indices", "group_ids"),
            f"pair {name} membership",
        )
        source = str(member["source"])
        expected_source = "test" if name == "test" else "train"
        if source != expected_source:
            raise BenchmarkError(
                f"pair {name} source must be canonical spec-2 {expected_source}"
            )
        indices = member["indices"]
        if not isinstance(indices, list) or not indices:
            raise BenchmarkError(f"pair {name} indices must be a non-empty list")
        if any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in indices):
            raise BenchmarkError(f"pair {name} indices must be non-negative integers")
        values = set(indices)
        if len(values) != len(indices):
            raise BenchmarkError(f"pair {name} indices contain duplicates")
        overlap = seen.setdefault(source, set()).intersection(values)
        if overlap:
            raise BenchmarkError(f"pair split memberships overlap for source {source}")
        seen[source].update(values)
        group_ids = member["group_ids"]
        if (
            not isinstance(group_ids, list)
            or len(group_ids) != len(indices)
            or any(not isinstance(value, str) or not value for value in group_ids)
        ):
            raise BenchmarkError(
                f"pair {name} group_ids must contain one non-empty ID per row"
            )
        split_groups[name] = set(group_ids)
    for left, right in (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ):
        if split_groups[left].intersection(split_groups[right]):
            raise BenchmarkError(
                f"pair leakage audit contains group overlap between {left} and {right}"
            )
    atomic = _require_mapping(split["atomic"], "atomic split membership")
    if set(atomic) != {"train", "validation", "test"}:
        raise BenchmarkError(
            "atomic split membership must define train/validation/test"
        )
    atomic_seen = {"spec_1": set(), "spec_5": set()}
    atomic_groups: dict[str, set[str]] = {
        "train": set(),
        "validation": set(),
        "test": set(),
    }
    for name, members in atomic.items():
        members = _require_mapping(members, f"atomic {name} membership")
        if set(members) != {"spec_1", "spec_5"}:
            raise BenchmarkError(f"atomic {name} must define spec_1 and spec_5")
        for source, membership in members.items():
            membership = _require_mapping(
                membership, f"atomic {name}/{source} membership"
            )
            _require_keys(
                membership,
                ("indices", "group_ids"),
                f"atomic {name}/{source} membership",
            )
            indices = membership["indices"]
            if not isinstance(indices, list) or any(
                isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in indices
            ):
                raise BenchmarkError(f"atomic {name}/{source} indices are invalid")
            values = set(indices)
            if len(values) != len(indices) or atomic_seen[source].intersection(values):
                raise BenchmarkError(f"atomic split memberships overlap for {source}")
            atomic_seen[source].update(values)
            group_ids = membership["group_ids"]
            if (
                not isinstance(group_ids, list)
                or len(group_ids) != len(indices)
                or any(not isinstance(value, str) or not value for value in group_ids)
            ):
                raise BenchmarkError(
                    f"atomic {name}/{source} group_ids must contain one ID per row"
                )
            atomic_groups[name].update(group_ids)
    for left, right in (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ):
        if atomic_groups[left].intersection(atomic_groups[right]):
            raise BenchmarkError(
                f"atomic leakage audit contains group overlap between {left} and {right}"
            )
    return split


def _validate_provenance(
    value: Any,
    *,
    pair_files: Mapping[str, Mapping[str, str]],
    atomic_files: Mapping[str, Mapping[str, str]],
    target_method: str,
    atomic_target_method: str,
) -> Mapping[str, Any]:
    provenance = _require_mapping(value, "dataset provenance manifest")
    _require_keys(
        provenance,
        ("schema_version", "status", "approved_by", "pair", "atomic"),
        "dataset provenance manifest",
    )
    if provenance["schema_version"] != 1 or provenance["status"] != "approved":
        raise BenchmarkError("dataset provenance must be schema v1 and approved")
    if not str(provenance["approved_by"]).strip():
        raise BenchmarkError("dataset provenance requires an approver identity")
    pair = _require_mapping(provenance["pair"], "pair provenance")
    _require_keys(
        pair,
        (
            "spec_type",
            "target_method",
            "component_order",
            "columns",
            "units",
            "license",
            "files",
        ),
        "pair provenance",
    )
    if pair["spec_type"] != 2 or pair["target_method"] != target_method:
        raise BenchmarkError("pair provenance does not match locked spec/target")
    if pair["component_order"] != ["elst", "exch", "indu", "disp"]:
        raise BenchmarkError("pair provenance component order is not canonical")
    if pair["columns"] != [
        "Total_aug",
        "Elst_aug",
        "Exch_aug",
        "Ind_aug",
        "Disp_aug",
    ]:
        raise BenchmarkError("pair provenance columns are not canonical spec-2 labels")
    if pair["units"] != "kcal/mol" or not str(pair["license"]).strip():
        raise BenchmarkError("pair provenance units/license are incomplete")
    if pair["files"] != {name: record["sha256"] for name, record in pair_files.items()}:
        raise BenchmarkError("pair provenance file identities do not match inputs")
    atomic = _require_mapping(provenance["atomic"], "atomic provenance")
    _require_keys(
        atomic,
        ("target_method", "properties", "units", "license", "files"),
        "atomic provenance",
    )
    if atomic["target_method"] != atomic_target_method:
        raise BenchmarkError("atomic provenance target method does not match config")
    if atomic["properties"] != ["q", "mu", "quadrupole"]:
        raise BenchmarkError("atomic provenance properties must be q/mu/quadrupole")
    if atomic["units"] != ["e", "e·bohr", "e·bohr²"]:
        raise BenchmarkError("atomic provenance units are not canonical")
    if not str(atomic["license"]).strip() or atomic["files"] != {
        name: record["sha256"] for name, record in atomic_files.items()
    }:
        raise BenchmarkError("atomic provenance license/file identities are invalid")
    return provenance


def normalize_config(
    config_path: str | Path,
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    raw = _require_mapping(
        _read_json(config_path, "benchmark config"), "benchmark config"
    )
    _require_keys(
        raw,
        (
            "schema_version",
            "benchmark_id",
            "output_root",
            "dataset",
            "artifacts",
            "training",
            "routes",
            "slurm",
        ),
        "benchmark config",
    )
    if raw["schema_version"] != SCHEMA_VERSION:
        raise BenchmarkError(
            f"benchmark config schema_version must be {SCHEMA_VERSION}"
        )
    benchmark_id = str(raw["benchmark_id"])
    if not ID_RE.fullmatch(benchmark_id):
        raise BenchmarkError("benchmark_id contains unsupported characters")

    dataset = dict(_require_mapping(raw["dataset"], "dataset config"))
    _require_keys(
        dataset,
        (
            "pair_spec_type",
            "pair_files",
            "atomic_files",
            "split_manifest",
            "provenance_manifest",
            "target_method",
            "atomic_target_method",
        ),
        "dataset config",
    )
    if dataset["pair_spec_type"] != 2:
        raise BenchmarkError(
            "the initial production benchmark is locked to AP3 pair spec_type=2"
        )
    pair_files = _require_mapping(dataset["pair_files"], "pair_files")
    atomic_files = _require_mapping(dataset["atomic_files"], "atomic_files")
    if set(pair_files) != {"train", "test"}:
        raise BenchmarkError("pair_files must contain train and test")
    if set(atomic_files) != {"spec_1", "spec_5"}:
        raise BenchmarkError("atomic_files must contain spec_1 and spec_5")
    dataset["pair_files"] = {
        name: _verify_record(value, f"pair {name} file")
        for name, value in pair_files.items()
    }
    dataset["atomic_files"] = {
        name: _verify_record(value, f"atomic {name} file")
        for name, value in atomic_files.items()
    }
    dataset["split_manifest"] = _verify_record(
        dataset["split_manifest"], "split manifest"
    )
    dataset["provenance_manifest"] = _verify_record(
        dataset["provenance_manifest"], "dataset provenance manifest"
    )
    split = _validate_split_manifest(
        _read_json(dataset["split_manifest"]["path"], "split manifest")
    )
    if split["pair_spec_type"] != dataset["pair_spec_type"]:
        raise BenchmarkError("config and split manifest pair spec types differ")
    if not str(dataset["target_method"]).strip():
        raise BenchmarkError("pair target_method must be explicit")
    if dataset["atomic_target_method"] != "PBE0/MBIS":
        raise BenchmarkError("initial atomic target_method must be PBE0/MBIS")
    provenance = _validate_provenance(
        _read_json(
            dataset["provenance_manifest"]["path"],
            "dataset provenance manifest",
        ),
        pair_files=dataset["pair_files"],
        atomic_files=dataset["atomic_files"],
        target_method=dataset["target_method"],
        atomic_target_method=dataset["atomic_target_method"],
    )

    artifacts_raw = _require_mapping(raw["artifacts"], "artifacts config")
    required_artifacts = (
        "mace",
        "legacy_atom_model",
        "legacy_parameter_model",
        "legacy_parameter_model_2",
        "physics_config",
    )
    _require_keys(artifacts_raw, required_artifacts, "artifacts config")
    artifacts = {
        name: _verify_record(artifacts_raw[name], name.replace("_", " "))
        for name in required_artifacts
    }
    physics_record = _require_mapping(
        _read_json(artifacts["physics_config"]["path"], "physics config"),
        "physics config",
    )
    declared_physics_hash = physics_record.get("physics_hash")
    physics_values = {
        key: value for key, value in physics_record.items() if key != "physics_hash"
    }
    try:
        project_root = Path(__file__).resolve().parents[2]
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        from apnet_pt.mace.schema import PhysicsConfig

        physics = PhysicsConfig(**physics_values)
    except (ImportError, TypeError, ValueError) as exc:
        raise BenchmarkError(f"invalid immutable PhysicsConfig: {exc}") from exc
    if declared_physics_hash != physics.physics_hash:
        raise BenchmarkError("physics config hash does not match its scientific fields")
    if physics.electrostatics_mode != "damped-cliff":
        raise BenchmarkError(
            "matched BASE-versus-MACE production benchmarks currently require "
            "damped-cliff electrostatics; AMOEBA and undamped are not matched"
        )
    supported_factory_physics = PhysicsConfig(
        electrostatics_mode=physics.electrostatics_mode,
        d3_parameters=physics.d3_parameters,
        neural_cutoff=physics.neural_cutoff,
    )
    if supported_factory_physics.physics_hash != physics.physics_hash:
        raise BenchmarkError(
            "production factory cannot yet override non-default Thole/SCF physics; "
            "the locked PhysicsConfig must match the supported factory contract"
        )
    physics_values = {
        **physics_values,
        "physics_hash": physics.physics_hash,
    }

    training = dict(_require_mapping(raw["training"], "training config"))
    training_keys = (
        "seeds",
        "epochs",
        "atomic_epochs",
        "batch_size",
        "atomic_batch_size",
        "learning_rate",
        "end_learning_rate",
        "include_total_mse",
        "early_stopping_patience",
        "num_workers",
    )
    _require_keys(training, training_keys, "training config")
    seeds = training["seeds"]
    if (
        not isinstance(seeds, list)
        or len(seeds) < 2
        or any(
            isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
            for seed in seeds
        )
    ):
        raise BenchmarkError(
            "training seeds must contain at least two non-negative integers"
        )
    if len(set(seeds)) != len(seeds):
        raise BenchmarkError("training seeds must be unique")
    training["seeds"] = sorted(seeds)
    for key in (
        "epochs",
        "atomic_epochs",
        "batch_size",
        "atomic_batch_size",
        "early_stopping_patience",
        "num_workers",
    ):
        training[key] = _positive_int(training[key], f"training {key}")
    for key in ("learning_rate", "end_learning_rate"):
        training[key] = _positive_float(training[key], f"training {key}")
    if training["end_learning_rate"] > training["learning_rate"]:
        raise BenchmarkError("end_learning_rate cannot exceed learning_rate")
    if not isinstance(training["include_total_mse"], bool):
        raise BenchmarkError("include_total_mse must be boolean")
    training["optimizer"] = "Adam"
    training["scheduler"] = "exponential-start-to-end"
    training["checkpoint_selection"] = "minimum-validation-loss"
    training["loss"] = (
        "mean(component-mse-plus-total-mse)"
        if training["include_total_mse"]
        else "mean(component-mse)"
    )
    training["model_hyperparameters"] = {
        "n_message": 3,
        "n_rbf": 8,
        "n_neuron": 128,
        "n_embed": 8,
        "r_cut_angstrom": 5.0,
        "r_cut_im_angstrom": 8.0,
        "mace_dtype": "float32",
        "mace_backbone": "frozen",
        "legacy_property_models": "frozen",
        "no_disp_nn": False,
    }

    routes = raw["routes"]
    if not isinstance(routes, list) or routes != list(ROUTES):
        raise BenchmarkError(
            f"routes must be the canonical ordered list: {list(ROUTES)}"
        )

    slurm = dict(_require_mapping(raw["slurm"], "slurm config"))
    _require_keys(
        slurm,
        (
            "account",
            "partition",
            "time",
            "cpus_per_task",
            "memory",
            "gpus_per_task",
            "python",
        ),
        "slurm config",
    )
    for key in ("account", "partition", "time", "memory", "python"):
        if not str(slurm[key]).strip() or "\n" in str(slurm[key]):
            raise BenchmarkError(f"slurm {key} must be a non-empty single-line value")
    for key in ("account", "partition"):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", str(slurm[key])):
            raise BenchmarkError(f"slurm {key} contains unsupported characters")
    if not re.fullmatch(r"[0-9:-]+", str(slurm["time"])):
        raise BenchmarkError("slurm time contains unsupported characters")
    if not re.fullmatch(r"[0-9]+(?:[KMGTP]B?)?", str(slurm["memory"]), re.IGNORECASE):
        raise BenchmarkError("slurm memory contains unsupported characters")
    slurm["cpus_per_task"] = _positive_int(
        slurm["cpus_per_task"], "slurm cpus_per_task"
    )
    slurm["gpus_per_task"] = _positive_int(
        slurm["gpus_per_task"], "slurm gpus_per_task"
    )

    output_root = str(Path(raw["output_root"]).expanduser().resolve())
    if any(character.isspace() for character in output_root):
        raise BenchmarkError(
            "output_root cannot contain whitespace in SLURM directives"
        )
    normalized = {
        "schema_version": SCHEMA_VERSION,
        "benchmark_id": benchmark_id,
        "output_root": output_root,
        "dataset": dataset,
        "dataset_provenance": provenance,
        "artifacts": artifacts,
        "physics": physics_values,
        "training": training,
        "routes": list(ROUTES),
        "slurm": slurm,
    }
    return normalized, split


def _source_commit() -> str:
    project_root = Path(__file__).resolve().parents[2]
    try:
        value = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise BenchmarkError(
            "unable to resolve source commit for benchmark lock"
        ) from exc
    if not re.fullmatch(r"[0-9a-f]{40}", value):
        raise BenchmarkError("source commit is not a canonical Git SHA-1")
    return value


def benchmark_root(config: Mapping[str, Any]) -> Path:
    return Path(config["output_root"]) / str(config["benchmark_id"])


def _task_groups(
    config: Mapping[str, Any], config_path: str | Path
) -> list[dict[str, Any]]:
    seeds = config["training"]["seeds"]
    return [
        {"group": "dataset", "depends_on": [], "tasks": [{"kind": "dataset"}]},
        {"group": "prepare", "depends_on": ["dataset"], "tasks": [{"kind": "prepare"}]},
        {
            "group": "baseline",
            "depends_on": ["dataset"],
            "tasks": [
                {"kind": "pair", "route": "BASE", "seed": seed} for seed in seeds
            ],
        },
        {
            "group": "hybrid",
            "depends_on": ["prepare"],
            "tasks": [
                {"kind": "pair", "route": route, "seed": seed}
                for route in ("H1", "H2")
                for seed in seeds
            ],
        },
        {
            "group": "atomic",
            "depends_on": ["prepare"],
            "tasks": [
                {"kind": "atomic", "property_mode": mode, "seed": seed}
                for mode in ("direct-completion", "learned")
                for seed in seeds
            ],
        },
        {
            "group": "polar",
            "depends_on": ["atomic"],
            "tasks": [
                {"kind": "pair", "route": route, "seed": seed}
                for route in ("DirectPolar", "AtomHead")
                for seed in seeds
            ],
        },
        {
            "group": "report",
            "depends_on": ["baseline", "hybrid", "polar"],
            "dependency_mode": "afterany",
            "tasks": [
                {
                    "kind": "report",
                    "config_path": str(Path(config_path).expanduser().resolve()),
                }
            ],
        },
    ]


def _shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def _render_sbatch(
    lock_path: Path,
    task_path: Path,
    group: Mapping[str, Any],
    config: Mapping[str, Any],
) -> str:
    slurm = config["slurm"]
    root = lock_path.parent
    count = len(group["tasks"])
    array = f"#SBATCH --array=0-{count - 1}\n" if count > 1 else ""
    gpu = (
        ""
        if group["group"] == "report"
        else f"#SBATCH --gpus-per-task={slurm['gpus_per_task']}\n"
    )
    return f"""#!/usr/bin/env bash
#SBATCH --job-name=mace-{group["group"]}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={slurm["cpus_per_task"]}
{gpu}#SBATCH --mem={slurm["memory"]}
#SBATCH --time={slurm["time"]}
#SBATCH --account={slurm["account"]}
#SBATCH --partition={slurm["partition"]}
{array}#SBATCH --output={root}/slurm/%x-%A_%a.out
#SBATCH --error={root}/slurm/%x-%A_%a.err

set -euo pipefail
export OMP_NUM_THREADS=${{SLURM_CPUS_PER_TASK:-{slurm["cpus_per_task"]}}}
TASK_INDEX=${{SLURM_ARRAY_TASK_ID:-0}}
exec srun {_shell_quote(str(slurm["python"]))} {_shell_quote(str(Path(__file__).resolve()))} \\
  --run-task --lock {_shell_quote(str(lock_path))} \\
  --task-file {_shell_quote(str(task_path))} --task-index "$TASK_INDEX"
"""


def generate(config_path: str | Path) -> tuple[Path, dict[str, Any]]:
    config, split = normalize_config(config_path)
    root = benchmark_root(config)
    lock_path = root / "benchmark.lock.json"
    groups = _task_groups(config, config_path)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "source_commit": _source_commit(),
        "config": config,
        "split_manifest": split,
        "task_plan": {
            group["group"]: {
                "depends_on": group["depends_on"],
                "dependency_mode": group.get("dependency_mode", "afterok"),
                "tasks": group["tasks"],
            }
            for group in groups
        },
    }
    benchmark_hash = _sha256_bytes(_json_bytes(payload))
    lock = {**payload, "benchmark_hash": benchmark_hash, "status": "LOCKED"}
    if lock_path.exists():
        existing = _read_json(lock_path, "existing benchmark lock")
        if existing != lock:
            raise BenchmarkError(
                f"benchmark lock already exists with different content: {lock_path}; use a new benchmark_id"
            )
    else:
        _write_json(lock_path, lock)

    (root / "slurm").mkdir(parents=True, exist_ok=True)
    (root / "results").mkdir(parents=True, exist_ok=True)
    submission = root / "submission"
    jobs = []
    for group in groups:
        task_path = submission / f"{group['group']}.tasks.json"
        task_record = {
            "schema_version": SCHEMA_VERSION,
            "benchmark_hash": benchmark_hash,
            "group": group["group"],
            "tasks": group["tasks"],
        }
        _write_json(task_path, task_record)
        script_path = submission / f"{group['group']}.sbatch"
        _atomic_write(
            script_path, _render_sbatch(lock_path, task_path, group, config).encode()
        )
        script_path.chmod(0o755)
        jobs.append(
            {
                "group": group["group"],
                "depends_on": group["depends_on"],
                "dependency_mode": group.get("dependency_mode", "afterok"),
                "task_count": len(group["tasks"]),
                "task_file": str(task_path),
                "script": str(script_path),
            }
        )
    _write_json(submission / "jobs.json", jobs)
    return root, lock


def _parse_job_id(output: str) -> str:
    matches = re.findall(r"(?m)(?:Submitted batch job\s+)?([0-9]+)\s*$", output)
    if not matches:
        raise BenchmarkError(f"unable to parse sbatch job id from: {output!r}")
    return matches[-1]


def submit(root: Path, *, sbatch_bin: str = "sbatch") -> Mapping[str, str]:
    jobs = _read_json(root / "submission" / "jobs.json", "generated jobs")
    if not isinstance(jobs, list):
        raise BenchmarkError("generated jobs manifest must be a list")
    receipt_path = root / "submission" / "submitted.json"
    if receipt_path.exists():
        raise BenchmarkError(f"submission receipt already exists: {receipt_path}")
    job_ids: dict[str, str] = {}
    commands = []
    for job in jobs:
        command = [sbatch_bin]
        dependencies = job["depends_on"]
        if dependencies:
            missing = [name for name in dependencies if name not in job_ids]
            if missing:
                raise BenchmarkError(f"submission dependency is unresolved: {missing}")
            dependency_mode = job.get("dependency_mode", "afterok")
            if dependency_mode not in {"afterok", "afterany"}:
                raise BenchmarkError(
                    f"unsupported dependency mode for {job['group']}: {dependency_mode}"
                )
            command.append(
                f"--dependency={dependency_mode}:"
                + ":".join(job_ids[name] for name in dependencies)
            )
        command.append(job["script"])
        try:
            completed = subprocess.run(
                command, capture_output=True, text=True, check=True
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            stderr = getattr(exc, "stderr", "")
            raise BenchmarkError(f"sbatch failed for {job['group']}: {stderr}") from exc
        job_id = _parse_job_id(completed.stdout)
        job_ids[job["group"]] = job_id
        commands.append({"group": job["group"], "command": command, "job_id": job_id})
        _write_json(
            receipt_path,
            {
                "schema_version": SCHEMA_VERSION,
                "status": "PARTIAL",
                "job_ids": job_ids,
                "commands": commands,
            },
        )
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": "SUBMITTED",
        "job_ids": job_ids,
        "commands": commands,
    }
    _write_json(receipt_path, receipt)
    return job_ids


def validate_result(record: Any, lock: Mapping[str, Any]) -> Mapping[str, Any]:
    value = _require_mapping(record, "benchmark result")
    _require_keys(
        value,
        (
            "schema_version",
            "benchmark_hash",
            "route",
            "seed",
            "status",
            "epochs_requested",
            "epochs_completed",
            "metrics",
            "resources",
            "history",
        ),
        "benchmark result",
    )
    if value["schema_version"] != SCHEMA_VERSION:
        raise ValueError("benchmark result schema version mismatch")
    if value["benchmark_hash"] != lock["benchmark_hash"]:
        raise ValueError("benchmark hash mismatch in result")
    if value["route"] not in ROUTES:
        raise ValueError("benchmark result route is invalid")
    if value["seed"] not in lock["config"]["training"]["seeds"]:
        raise ValueError("benchmark result seed is invalid")
    if value["status"] not in {"PASS", "FAIL", "BLOCKED"}:
        raise ValueError("benchmark result status must be PASS, FAIL, or BLOCKED")
    if not isinstance(value["epochs_completed"], int) or value["epochs_completed"] < 0:
        raise ValueError("epochs_completed is invalid")
    if not isinstance(value["epochs_requested"], int) or value["epochs_requested"] < 1:
        raise ValueError("epochs_requested is invalid")
    history = value["history"]
    if not isinstance(history, list) or len(history) != value["epochs_completed"]:
        raise ValueError("result history must contain one record per completed epoch")
    for expected_epoch, item in enumerate(history, start=1):
        if (
            not isinstance(item, Mapping)
            or item.get("epoch") != expected_epoch
            or any(
                not isinstance(item.get(name), (int, float))
                or not math.isfinite(float(item[name]))
                for name in ("train_loss", "validation_loss")
            )
        ):
            raise ValueError("result history is malformed or non-finite")
    resources = _require_mapping(value["resources"], "result resources")
    for name in ("elapsed_seconds", "peak_rss_mb"):
        number = resources.get(name)
        if (
            not isinstance(number, (int, float))
            or not math.isfinite(float(number))
            or number < 0
        ):
            raise ValueError(f"result resource {name} is invalid")
    if value["status"] == "PASS":
        test = _require_mapping(
            _require_mapping(value["metrics"], "result metrics").get("test"),
            "test metrics",
        )
        for component in COMPONENTS:
            metric = _require_mapping(test.get(component), f"test {component} metrics")
            for name in ("mae", "rmse"):
                number = metric.get(name)
                if (
                    not isinstance(number, (int, float))
                    or not math.isfinite(float(number))
                    or number < 0
                ):
                    raise ValueError(f"test {component} {name} is invalid")
    return value


def _expected_keys(lock: Mapping[str, Any]) -> set[tuple[str, int]]:
    return {
        (route, seed)
        for route in lock["config"]["routes"]
        for seed in lock["config"]["training"]["seeds"]
    }


def collect_results(
    root: Path, lock: Mapping[str, Any]
) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    found = {}
    errors = []
    for path in sorted((root / "results").glob("*.json")):
        try:
            record = validate_result(_read_json(path, f"result {path.name}"), lock)
        except (BenchmarkError, ValueError) as exc:
            errors.append(f"{path.name}: {exc}")
            continue
        key = (record["route"], record["seed"])
        if key in found:
            errors.append(f"duplicate result for {key}: {path.name}")
            continue
        found[key] = record
    expected = _expected_keys(lock)
    requested_epochs = lock["config"]["training"]["epochs"]
    complete = {
        key
        for key, value in found.items()
        if value["status"] == "PASS"
        and value["epochs_requested"] == requested_epochs
        and (
            value["epochs_completed"] == requested_epochs
            or (
                value.get("termination_reason") == "early_stopping"
                and 0 < value["epochs_completed"] < requested_epochs
            )
        )
    }
    missing = sorted(expected - set(found))
    incomplete = sorted(set(found) - complete)
    status = "COMPLETE" if complete == expected and not errors else "PARTIAL"
    summary = {
        "schema_version": SCHEMA_VERSION,
        "benchmark_hash": lock["benchmark_hash"],
        "completion_status": status,
        "completed_runs": len(complete),
        "reported_runs": len(found),
        "expected_runs": len(expected),
        "missing_runs": [{"route": route, "seed": seed} for route, seed in missing],
        "incomplete_runs": [
            {"route": route, "seed": seed} for route, seed in incomplete
        ],
        "invalid_results": errors,
    }
    return [found[key] for key in sorted(found)], summary


def _collect_atomic_results(
    root: Path, lock: Mapping[str, Any]
) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    expected = {
        (mode, seed)
        for mode in ("direct-completion", "learned")
        for seed in lock["config"]["training"]["seeds"]
    }
    found = {}
    errors = []
    requested = lock["config"]["training"]["atomic_epochs"]
    for path in sorted((root / "atomic").glob("*/seed-*/result.json")):
        try:
            record = _require_mapping(
                _read_json(path, f"atomic result {path}"), "atomic result"
            )
            if record.get("benchmark_hash") != lock["benchmark_hash"]:
                raise ValueError("benchmark hash mismatch")
            key = (record.get("property_mode"), record.get("seed"))
            if key not in expected or key in found:
                raise ValueError("unexpected or duplicate atomic route/seed")
            if record.get("status") not in {"PASS", "FAIL", "BLOCKED"}:
                raise ValueError("invalid atomic status")
            if record.get("epochs_requested") != requested:
                raise ValueError("atomic epoch budget mismatch")
            completed = record.get("epochs_completed")
            history = record.get("history")
            if (
                not isinstance(completed, int)
                or completed < 0
                or not isinstance(history, list)
                or len(history) != completed
            ):
                raise ValueError("atomic history/epoch count mismatch")
            for expected_epoch, item in enumerate(history, start=1):
                if (
                    not isinstance(item, Mapping)
                    or item.get("epoch") != expected_epoch
                    or any(
                        not isinstance(item.get(name), (int, float))
                        or not math.isfinite(float(item[name]))
                        for name in ("train_loss", "validation_loss")
                    )
                ):
                    raise ValueError("atomic history is malformed or non-finite")
            if record["status"] == "PASS":
                test = _require_mapping(
                    _require_mapping(record.get("metrics"), "atomic metrics").get(
                        "test"
                    ),
                    "atomic test metrics",
                )
                for property_name in ATOMIC_METRICS:
                    metric = _require_mapping(
                        test.get(property_name),
                        f"atomic {property_name} metrics",
                    )
                    for name in ("mae", "rmse"):
                        number = metric.get(name)
                        if (
                            not isinstance(number, (int, float))
                            or not math.isfinite(float(number))
                            or number < 0
                        ):
                            raise ValueError(f"invalid atomic {property_name} {name}")
            found[key] = record
        except (BenchmarkError, ValueError) as exc:
            errors.append(f"{path}: {exc}")
    complete = {
        key
        for key, record in found.items()
        if record["status"] == "PASS"
        and (
            record["epochs_completed"] == requested
            or (
                record.get("termination_reason") == "early_stopping"
                and 0 < record["epochs_completed"] < requested
            )
        )
    }
    summary = {
        "atomic_completed_runs": len(complete),
        "atomic_reported_runs": len(found),
        "atomic_expected_runs": len(expected),
        "atomic_missing_runs": [
            {"property_mode": mode, "seed": seed}
            for mode, seed in sorted(expected - set(found))
        ],
        "atomic_incomplete_runs": [
            {"property_mode": mode, "seed": seed}
            for mode, seed in sorted(set(found) - complete)
        ],
        "atomic_invalid_results": errors,
    }
    return [found[key] for key in sorted(found)], summary


def _write_atomic_csv(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    fields = ["property_mode", "seed", "status", "epochs_completed"] + [
        f"test_{property_name}_{metric}"
        for property_name in ATOMIC_METRICS
        for metric in ("mae", "rmse")
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = {name: record.get(name) for name in fields[:4]}
            test = record.get("metrics", {}).get("test", {})
            for property_name in ATOMIC_METRICS:
                for metric in ("mae", "rmse"):
                    row[f"test_{property_name}_{metric}"] = test.get(
                        property_name, {}
                    ).get(metric, "")
            writer.writerow(row)


def _write_metrics_csv(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    fields = ["route", "seed", "status", "epochs_requested", "epochs_completed"]
    for split in ("test",):
        for component in COMPONENTS:
            for metric in ("mae", "rmse"):
                fields.append(f"{split}_{component}_{metric}")
    fields.extend(("elapsed_seconds", "peak_rss_mb"))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = {key: record.get(key) for key in fields[:5]}
            test = record.get("metrics", {}).get("test", {})
            for component in COMPONENTS:
                for metric in ("mae", "rmse"):
                    row[f"test_{component}_{metric}"] = test.get(component, {}).get(
                        metric, ""
                    )
            resources = record.get("resources", {})
            row["elapsed_seconds"] = resources.get("elapsed_seconds", "")
            row["peak_rss_mb"] = resources.get("peak_rss_mb", "")
            writer.writerow(row)


def _mean_ci95(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    t_critical = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
    }.get(len(values) - 1, 1.96)
    return mean, t_critical * math.sqrt(variance / len(values))


def _write_aggregate_csv(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    passed = [record for record in records if record["status"] == "PASS"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("route", "component", "metric", "n", "mean", "ci95"),
        )
        writer.writeheader()
        for route in ROUTES:
            route_records = [record for record in passed if record["route"] == route]
            for component in COMPONENTS:
                for metric in ("mae", "rmse"):
                    values = [
                        float(record["metrics"]["test"][component][metric])
                        for record in route_records
                    ]
                    if values:
                        mean, ci95 = _mean_ci95(values)
                        writer.writerow(
                            {
                                "route": route,
                                "component": component,
                                "metric": metric,
                                "n": len(values),
                                "mean": mean,
                                "ci95": ci95,
                            }
                        )


def _write_baseline_deltas(
    path: Path, records: Sequence[Mapping[str, Any]]
) -> dict[str, tuple[float, float, int]]:
    passed = [record for record in records if record["status"] == "PASS"]
    baseline = {
        record["seed"]: float(record["metrics"]["test"]["total"]["mae"])
        for record in passed
        if record["route"] == "BASE"
    }
    aggregates = {}
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "route",
                "seed",
                "base_total_mae",
                "route_total_mae",
                "mae_delta_route_minus_base",
            ),
        )
        writer.writeheader()
        for route in ROUTES[1:]:
            deltas = []
            for record in passed:
                if record["route"] != route or record["seed"] not in baseline:
                    continue
                route_mae = float(record["metrics"]["test"]["total"]["mae"])
                delta = route_mae - baseline[record["seed"]]
                deltas.append(delta)
                writer.writerow(
                    {
                        "route": route,
                        "seed": record["seed"],
                        "base_total_mae": baseline[record["seed"]],
                        "route_total_mae": route_mae,
                        "mae_delta_route_minus_base": delta,
                    }
                )
            if deltas:
                mean, ci95 = _mean_ci95(deltas)
                aggregates[route] = (mean, ci95, len(deltas))
    return aggregates


def _watermark(figure, summary: Mapping[str, Any]) -> str:
    label = (
        f"{summary['completion_status']}: "
        f"PAIR {summary['completed_runs']}/{summary['expected_runs']}; "
        f"ATOMIC {summary['atomic_completed_runs']}/"
        f"{summary['atomic_expected_runs']} COMPLETE RUNS"
    )
    color = "#8b1e1e" if summary["completion_status"] == "PARTIAL" else "#1d6b3d"
    figure.suptitle(label, color=color, fontweight="bold")
    if summary["completion_status"] == "PARTIAL":
        figure.text(
            0.5,
            0.5,
            "PARTIAL",
            fontsize=54,
            color=color,
            alpha=0.10,
            ha="center",
            va="center",
            rotation=25,
        )
    return label


def plot_results(root: Path, lock: Mapping[str, Any]) -> Mapping[str, Any]:
    records, summary = collect_results(root, lock)
    atomic_records, atomic_summary = _collect_atomic_results(root, lock)
    summary.update(atomic_summary)
    if (
        summary["atomic_completed_runs"] != summary["atomic_expected_runs"]
        or summary["atomic_invalid_results"]
    ):
        summary["completion_status"] = "PARTIAL"
    plot_dir = root / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    _write_json(plot_dir / "summary.json", summary)
    _atomic_write(
        plot_dir / "STATUS.txt",
        (
            f"{summary['completion_status']}: "
            f"pair {summary['completed_runs']}/{summary['expected_runs']} complete "
            f"({summary['reported_runs']} reported); "
            f"atomic {summary['atomic_completed_runs']}/"
            f"{summary['atomic_expected_runs']} complete "
            f"({summary['atomic_reported_runs']} reported)\n"
        ).encode(),
    )
    _write_metrics_csv(plot_dir / "metrics.csv", records)
    _write_atomic_csv(plot_dir / "atomic_metrics.csv", atomic_records)
    _write_aggregate_csv(plot_dir / "aggregate.csv", records)
    baseline_deltas = _write_baseline_deltas(plot_dir / "baseline_deltas.csv", records)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise BenchmarkError("plotting requires matplotlib") from exc

    passed = [record for record in records if record["status"] == "PASS"]
    colors = {
        route: color
        for route, color in zip(
            ROUTES, ("#555555", "#0072B2", "#56B4E9", "#D55E00", "#009E73")
        )
    }

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for axis, metric in zip(axes, ("mae", "rmse")):
        labels, means, errors, bar_colors = [], [], [], []
        for route in ROUTES:
            values = [
                float(r["metrics"]["test"]["total"][metric])
                for r in passed
                if r["route"] == route
            ]
            if not values:
                continue
            mean, ci95 = _mean_ci95(values)
            labels.append(route)
            means.append(mean)
            errors.append(ci95)
            bar_colors.append(colors[route])
        axis.bar(labels, means, yerr=errors, color=bar_colors, capsize=4)
        axis.set_ylabel(f"Total test {metric.upper()} (kcal/mol; 95% CI)")
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.25)
    _watermark(fig, summary)
    fig.tight_layout()
    fig.savefig(plot_dir / "accuracy.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(12, 6))
    component_names = ("elst", "exch", "indu", "disp")
    x_positions = list(range(len(component_names)))
    width = 0.15
    for route_index, route in enumerate(ROUTES):
        route_records = [record for record in passed if record["route"] == route]
        if not route_records:
            continue
        means = []
        errors = []
        for component in component_names:
            values = [
                float(record["metrics"]["test"][component]["mae"])
                for record in route_records
            ]
            mean, ci95 = _mean_ci95(values)
            means.append(mean)
            errors.append(ci95)
        offsets = [
            value + (route_index - (len(ROUTES) - 1) / 2) * width
            for value in x_positions
        ]
        axis.bar(
            offsets,
            means,
            width=width,
            yerr=errors,
            capsize=2,
            label=route,
            color=colors[route],
        )
    axis.set_xticks(x_positions, [name.upper() for name in component_names])
    axis.set_ylabel("Test component MAE (kcal/mol; 95% CI)")
    axis.grid(axis="y", alpha=0.25)
    if passed:
        axis.legend(fontsize=8)
    _watermark(fig, summary)
    fig.tight_layout()
    fig.savefig(plot_dir / "components.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(10, 6))
    for record in passed:
        history = record.get("history", [])
        epochs = [item["epoch"] for item in history]
        values = [item["validation_loss"] for item in history]
        axis.plot(
            epochs,
            values,
            color=colors[record["route"]],
            alpha=0.5,
            label=f"{record['route']} seed {record['seed']}",
        )
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Validation loss")
    axis.set_yscale("log")
    axis.grid(alpha=0.25)
    if passed:
        axis.legend(fontsize=7, ncol=2)
    _watermark(fig, summary)
    fig.tight_layout()
    fig.savefig(plot_dir / "learning_curves.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 5.5))
    delta_routes = [route for route in ROUTES[1:] if route in baseline_deltas]
    delta_means = [baseline_deltas[route][0] for route in delta_routes]
    delta_errors = [baseline_deltas[route][1] for route in delta_routes]
    axis.bar(
        delta_routes,
        delta_means,
        yerr=delta_errors,
        color=[colors[route] for route in delta_routes],
        capsize=4,
    )
    axis.axhline(0.0, color="black", linewidth=1)
    axis.set_ylabel("Paired total-MAE delta vs BASE (kcal/mol; 95% CI)")
    axis.tick_params(axis="x", rotation=25)
    axis.grid(axis="y", alpha=0.25)
    _watermark(fig, summary)
    fig.tight_layout()
    fig.savefig(plot_dir / "baseline_delta.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    atomic_passed = [record for record in atomic_records if record["status"] == "PASS"]
    atomic_colors = {
        "direct-completion": "#D55E00",
        "learned": "#009E73",
    }
    for axis, property_name in zip(axes, ATOMIC_METRICS):
        labels = []
        means = []
        errors = []
        bar_colors = []
        for mode in ("direct-completion", "learned"):
            values = [
                float(record["metrics"]["test"][property_name]["mae"])
                for record in atomic_passed
                if record["property_mode"] == mode
            ]
            if not values:
                continue
            mean, ci95 = _mean_ci95(values)
            labels.append(mode)
            means.append(mean)
            errors.append(ci95)
            bar_colors.append(atomic_colors[mode])
        axis.bar(labels, means, yerr=errors, color=bar_colors, capsize=4)
        axis.set_title(property_name)
        axis.set_ylabel("Test MAE (canonical property units; 95% CI)")
        axis.tick_params(axis="x", rotation=20)
        axis.grid(axis="y", alpha=0.25)
    _watermark(fig, summary)
    fig.tight_layout()
    fig.savefig(plot_dir / "atomic_properties.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for axis, key, label in zip(
        axes, ("elapsed_seconds", "peak_rss_mb"), ("Elapsed time (s)", "Peak RSS (MB)")
    ):
        for route in ROUTES:
            xs = [r["seed"] for r in passed if r["route"] == route]
            ys = [
                r.get("resources", {}).get(key, float("nan"))
                for r in passed
                if r["route"] == route
            ]
            if xs:
                axis.scatter(xs, ys, label=route, color=colors[route])
        axis.set_xlabel("Seed")
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
    if passed:
        axes[1].legend(fontsize=8)
    _watermark(fig, summary)
    fig.tight_layout()
    fig.savefig(plot_dir / "resources.png", dpi=180)
    plt.close(fig)
    return summary


def _load_task(
    lock_path: Path, task_file: Path, index: int
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    lock = _require_mapping(_read_json(lock_path, "benchmark lock"), "benchmark lock")
    if lock.get("status") != "LOCKED":
        raise BenchmarkError("benchmark lock status is not LOCKED")
    tasks = _require_mapping(_read_json(task_file, "task table"), "task table")
    if tasks.get("benchmark_hash") != lock.get("benchmark_hash"):
        raise BenchmarkError("task table benchmark hash does not match lock")
    group = tasks.get("group")
    task_plan = lock.get("task_plan", {}).get(group)
    if not isinstance(task_plan, Mapping) or tasks.get("tasks") != task_plan.get(
        "tasks"
    ):
        raise BenchmarkError("task table content does not match immutable lock plan")
    values = tasks.get("tasks")
    if not isinstance(values, list) or index < 0 or index >= len(values):
        raise BenchmarkError(f"task index {index} is outside task table")
    return lock, _require_mapping(values[index], "task")


def run_task(lock_path: Path, task_file: Path, index: int) -> int:
    """Execute one production task through the benchmark training module."""

    lock, task = _load_task(lock_path, task_file, index)
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    try:
        from apnet_pt.training.mace_benchmark import run_benchmark_task
    except ImportError as exc:
        raise BenchmarkError(
            f"unable to import production benchmark trainer: {exc}"
        ) from exc
    try:
        run_benchmark_task(lock=lock, task=task, benchmark_root=lock_path.parent)
    except Exception as exc:
        raise BenchmarkError(f"benchmark worker failed: {exc}") from exc
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", help="versioned benchmark JSON; generates jobs by default"
    )
    parser.add_argument(
        "--submit", action="store_true", help="submit generated arrays with sbatch"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="collect available results and render clearly marked plots",
    )
    parser.add_argument(
        "--sbatch-bin",
        default="sbatch",
        help="sbatch executable (testing/cluster override)",
    )
    parser.add_argument(
        "--run-task",
        action="store_true",
        help="internal worker mode used by generated SLURM arrays",
    )
    parser.add_argument("--lock", help="internal worker benchmark.lock.json")
    parser.add_argument("--task-file", help="internal worker task table")
    parser.add_argument("--task-index", type=int, help="internal worker array index")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.run_task:
            if (
                args.submit
                or args.plot
                or not args.lock
                or not args.task_file
                or args.task_index is None
            ):
                raise BenchmarkError(
                    "--run-task requires --lock, --task-file, and --task-index only"
                )
            return run_task(Path(args.lock), Path(args.task_file), args.task_index)
        if not args.config:
            raise BenchmarkError("--config is required outside --run-task mode")
        root, lock = generate(args.config)
        print(f"Generated benchmark jobs under {root}; not submitted.")
        if args.submit:
            job_ids = submit(root, sbatch_bin=args.sbatch_bin)
            print(
                "Submitted arrays: "
                + ", ".join(f"{name}={value}" for name, value in job_ids.items())
            )
        if args.plot:
            summary = plot_results(root, lock)
            print(
                f"Plots: {summary['completion_status']} {summary['completed_runs']}/{summary['expected_runs']} complete runs"
            )
        return 0
    except (BenchmarkError, ValueError) as exc:
        print(f"benchmark.py: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
