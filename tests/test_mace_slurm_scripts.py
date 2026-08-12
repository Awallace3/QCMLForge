import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

import pytest
import torch

from apnet_pt.mace.schema import PhysicsConfig
from apnet_pt.training.smoke import load_prepared_feature_cache


ROOT = Path(__file__).parents[1]
SLURM = ROOT / "scripts" / "slurm"
JOBS = [
    SLURM / "prepare_mace_ap3d3_features.sbatch",
    SLURM / "train_mace_atomic_properties.sbatch",
    SLURM / "train_mace_ap3d3.sbatch",
]
SUBMIT = SLURM / "submit_mace_ap3d3_matrix.sh"
PREPARE_PY = ROOT / "scripts" / "prepare_mace_ap3d3_features.py"


def _run(path, env):
    return subprocess.run(
        ["bash", str(path)],
        cwd=ROOT,
        env={**os.environ, **env},
        capture_output=True,
        text=True,
        check=False,
    )


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _base_pair_env(tmp_path, option, run_id):
    physics = tmp_path / "physics.json"
    physics.write_text(
        '{"electrostatics_mode":"damped-cliff","d3_parameters":[], '
        '"physics_hash":"' + "1" * 64 + '"}'
    )
    data = ROOT / "tests" / "dataset_data" / "mace_ap3d3_smoke.pkl"
    run_root = tmp_path / "runs"
    run_dir = run_root / run_id
    return {
        "MODEL_OPTION": option,
        "SEED": "7",
        "DATA_DIR": str(data),
        "MODEL_OUT": str(run_dir / "checkpoints" / "model.pt"),
        "RUN_ROOT": str(run_root),
        "RUN_ID": run_id,
        "PHYSICS_CONFIG_PATH": str(physics),
        "PHYSICS_CONFIG_SHA256": _sha(physics),
        "ELECTROSTATICS_MODE": "damped-cliff",
        "N_EPOCHS": "1",
        "BATCH_SIZE": "8",
        "VALIDATE_ONLY": "1",
    }


def _add_artifact(env, tmp_path):
    artifact = tmp_path / "mace.model"
    artifact.write_bytes(b"verified-mace")
    env.update(
        MACE_MODEL_PATH=str(artifact),
        MACE_MODEL_SHA256=_sha(artifact),
    )
    return artifact


def _add_legacy(env, tmp_path):
    names = ["am", "param1", "param2"]
    variables = [
        ("AM_MODEL_PATH", "AM_MODEL_SHA256"),
        ("ATOM_TYPE_PARAM_MODEL_PATH", "ATOM_TYPE_PARAM_MODEL_SHA256"),
        ("ATOM_TYPE_PARAM_MODEL_PATH2", "ATOM_TYPE_PARAM_MODEL_SHA256_2"),
    ]
    for name, (path_var, digest_var) in zip(names, variables):
        path = tmp_path / f"{name}.pt"
        path.write_bytes(name.encode())
        env[path_var] = str(path)
        env[digest_var] = _sha(path)


def _add_complete_cache(env, tmp_path):
    cache = tmp_path / "feature-cache"
    cache.mkdir(exist_ok=True)
    (cache / "COMPLETE.json").write_text(
        '{"status":"complete","mace_sha256":"'
        + env["MACE_MODEL_SHA256"]
        + '","feature_schemas":{},"entry_count":0}'
    )
    env["FEATURE_CACHE_DIR"] = str(cache)


def _add_atom_checkpoint(env, tmp_path):
    path = tmp_path / "mace-atom.pt"
    path.write_bytes(b"atom-head")
    env["MACE_ATOM_MODEL_PATH"] = str(path)
    env["MACE_ATOM_MODEL_SHA256"] = _sha(path)


def _submission_env(tmp_path):
    return {
        "RUN_ROOT": str(tmp_path / "matrix"),
        "MATRIX_ID": "contract-test",
        "MACE_MODEL_PATH": "/verified/MACE-POLAR-1-S.model",
        "MACE_MODEL_SHA256": "a" * 64,
        "PAIR_DATA_PATH": "tests/dataset_data/mace_ap3d3_smoke.pkl",
        "ATOM_DATA_PATH": "tests/dataset_data/mace_atomic_properties_smoke.pkl",
        "AM_MODEL_PATH": "/models/am.pt",
        "AM_MODEL_SHA256": "b" * 64,
        "ATOM_TYPE_PARAM_MODEL_PATH": "/models/param1.pt",
        "ATOM_TYPE_PARAM_MODEL_SHA256": "c" * 64,
        "ATOM_TYPE_PARAM_MODEL_PATH2": "/models/param2.pt",
        "ATOM_TYPE_PARAM_MODEL_SHA256_2": "d" * 64,
        "PHYSICS_CONFIG_PATH": "/configs/physics.json",
        "PHYSICS_CONFIG_SHA256": "e" * 64,
        "ELECTROSTATICS_MODE": "damped-cliff",
        "SMALL_VERIFICATION_APPROVED": "1",
    }


def test_shell_syntax_single_task_single_srun_and_offline_contract():
    for path in [*JOBS, SUBMIT]:
        assert path.is_file()
        result = subprocess.run(
            ["bash", "-n", str(path)], capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, result.stderr
        text = path.read_text()
        assert "set -euo pipefail" in text
        for block in re.findall(r"<<'PY'\n(.*?)\nPY", text, flags=re.DOTALL):
            compile(block, f"{path}:heredoc", "exec")
        assert not re.search(
            r"\b(curl|wget)\b|git\s+clone|pip\s+install|huggingface.*download",
            text,
            flags=re.IGNORECASE,
        )
    for path in JOBS:
        text = path.read_text()
        assert "#SBATCH --ntasks=1" in text
        assert "#SBATCH --gpus-per-task=1" in text
        assert len(re.findall(r"(?m)^\s*srun\b", text)) == 1
    assert "--mace_offline" in JOBS[0].read_text()
    assert "--mace_offline" in JOBS[1].read_text()
    assert "--mace_offline" in JOBS[2].read_text()
    combined = "\n".join(path.read_text() for path in JOBS)
    for directory in ("logs", "cache", "data", "diagnostics", "tmp", "checkpoints"):
        assert directory in combined
    assert PREPARE_PY.is_file()
    assert not re.search(
        r"https?://|\b(curl|wget|requests)\b|download",
        PREPARE_PY.read_text(),
        flags=re.IGNORECASE,
    )


def test_prepare_digest_mismatch_and_missing_complete_manifest_fail(tmp_path):
    artifact = tmp_path / "mace.model"
    artifact.write_bytes(b"bad")
    common = {
        "RUN_ROOT": str(tmp_path / "runs"),
        "RUN_ID": "prepare-bad-digest",
        "MACE_MODEL_PATH": str(artifact),
        "MACE_MODEL_SHA256": "0" * 64,
        "PAIR_DATA_PATH": str(
            ROOT / "tests" / "dataset_data" / "mace_ap3d3_smoke.pkl"
        ),
        "ATOM_DATA_PATH": str(
            ROOT / "tests" / "dataset_data" / "mace_atomic_properties_smoke.pkl"
        ),
    }
    result = _run(JOBS[0], common)
    assert result.returncode != 0
    assert "SHA-256 mismatch" in result.stderr

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_srun = fake_bin / "srun"
    fake_srun.write_text("#!/usr/bin/env bash\nexit 0\n")
    fake_srun.chmod(0o755)
    common.update(
        RUN_ID="prepare-no-manifest",
        MACE_MODEL_SHA256=_sha(artifact),
        PATH=f"{fake_bin}:{os.environ['PATH']}",
    )
    result = _run(JOBS[0], common)
    assert result.returncode != 0
    assert "complete cache manifest" in result.stderr.lower()


def test_pair_script_accepts_complete_declared_variable_contract():
    text = JOBS[2].read_text()
    for variable in (
        "MODEL_OPTION", "SEED", "DATA_DIR", "FEATURE_CACHE_DIR", "MODEL_OUT",
        "MACE_MODEL_PATH", "MACE_MODEL_SHA256", "AM_MODEL_PATH",
        "AM_MODEL_SHA256", "ATOM_TYPE_PARAM_MODEL_PATH",
        "ATOM_TYPE_PARAM_MODEL_SHA256", "ATOM_TYPE_PARAM_MODEL_PATH2",
        "ATOM_TYPE_PARAM_MODEL_SHA256_2", "MACE_ATOM_MODEL_PATH",
        "MACE_ATOM_MODEL_SHA256", "PHYSICS_CONFIG_PATH",
        "PHYSICS_CONFIG_SHA256", "ELECTROSTATICS_MODE", "N_EPOCHS",
        "BATCH_SIZE",
    ):
        assert variable in text


def test_pair_route_inputs_are_validated_without_irrelevant_requirements(tmp_path):
    base = _base_pair_env(tmp_path, "BASE", "base-valid")
    _add_legacy(base, tmp_path)
    result = _run(JOBS[2], base)
    assert result.returncode == 0, result.stderr
    assert "MACE_MODEL_PATH" not in result.stderr

    h1 = _base_pair_env(tmp_path, "MACE-AP3D3-H1", "h1-missing-legacy")
    _add_artifact(h1, tmp_path)
    _add_complete_cache(h1, tmp_path)
    result = _run(JOBS[2], h1)
    assert result.returncode != 0
    assert "AM_MODEL_PATH" in result.stderr

    direct = _base_pair_env(
        tmp_path, "MACE-AP3D3-DirectPolar", "direct-missing-head"
    )
    _add_artifact(direct, tmp_path)
    _add_complete_cache(direct, tmp_path)
    result = _run(JOBS[2], direct)
    assert result.returncode != 0
    assert "MACE_ATOM_MODEL_PATH" in result.stderr


def test_pair_rejects_existing_output_without_implicit_resume(tmp_path):
    env = _base_pair_env(tmp_path, "BASE", "base-existing-output")
    _add_legacy(env, tmp_path)
    output = Path(env["MODEL_OUT"])
    output.parent.mkdir(parents=True)
    output.write_bytes(b"stale")
    result = _run(JOBS[2], env)
    assert result.returncode != 0
    assert "existing MODEL_OUT" in result.stderr
    assert "no implicit resume" in result.stderr


def test_pair_rejects_mace_digest_mismatch_and_partial_cache(tmp_path):
    env = _base_pair_env(tmp_path, "MACE-AP3D3-H2", "h2-bad-digest")
    artifact = _add_artifact(env, tmp_path)
    _add_legacy(env, tmp_path)
    env["MACE_MODEL_SHA256"] = "f" * 64
    result = _run(JOBS[2], env)
    assert result.returncode != 0
    assert "SHA-256 mismatch" in result.stderr

    env = _base_pair_env(tmp_path, "MACE-AP3D3-H2", "h2-partial-cache")
    _add_artifact(env, tmp_path)
    _add_legacy(env, tmp_path)
    cache = tmp_path / "partial-cache"
    cache.mkdir()
    (cache / "orphan.pt").write_bytes(b"partial")
    env["FEATURE_CACHE_DIR"] = str(cache)
    result = _run(JOBS[2], env)
    assert result.returncode != 0
    assert "partial feature cache" in result.stderr.lower()


def test_direct_route_accepts_only_relevant_verified_inputs(tmp_path):
    env = _base_pair_env(
        tmp_path, "MACE-AP3D3-DirectPolar", "direct-valid"
    )
    _add_artifact(env, tmp_path)
    _add_complete_cache(env, tmp_path)
    _add_atom_checkpoint(env, tmp_path)
    result = _run(JOBS[2], env)
    assert result.returncode == 0, result.stderr
    assert "AM_MODEL_PATH" not in result.stderr


def test_atomic_job_rejects_bad_artifact_and_partial_feature_cache(tmp_path):
    artifact = tmp_path / "mace.model"
    artifact.write_bytes(b"mace")
    atom_data = ROOT / "tests" / "dataset_data" / "mace_atomic_properties_smoke.pkl"
    prep_manifest = tmp_path / "prep.json"
    prep_manifest.write_text("{}")
    cache = tmp_path / "cache"
    cache.mkdir()
    env = {
        "RUN_ROOT": str(tmp_path / "runs"),
        "RUN_ID": "atomic-bad-digest",
        "SEED": "0",
        "MACE_MODEL_PATH": str(artifact),
        "MACE_MODEL_SHA256": "0" * 64,
        "FEATURE_CACHE_DIR": str(cache),
        "PREP_MANIFEST": str(prep_manifest),
        "ATOM_DATA_PATH": str(atom_data),
    }
    result = _run(JOBS[1], env)
    assert result.returncode != 0
    assert "SHA-256 mismatch" in result.stderr

    env.update(
        RUN_ID="atomic-partial-cache",
        MACE_MODEL_SHA256=_sha(artifact),
    )
    result = _run(JOBS[1], env)
    assert result.returncode != 0
    assert "partial feature cache" in result.stderr.lower()


def test_prepared_cache_consumer_requires_complete_identity_checked_entries(tmp_path):
    cache = tmp_path / "cache"
    mode = "final-layer-scalars"
    entry_dir = cache / mode
    entry_dir.mkdir(parents=True)
    key = "a" * 64
    mace_sha = "b" * 64
    physics_hash = PhysicsConfig().physics_hash
    identity = {
        "format": "qcmlforge-mace-monomer-cache-v1",
        "monomer_hash": "c" * 64,
        "cache_key": key,
        "feature_mode": mode,
        "mace_sha256": mace_sha,
        "mace_model_id": "polar-1-s",
        "physics_hash": physics_hash,
        "dtype": "float32",
    }
    tensors = {
        "invariant": torch.ones(1, 2),
        "equivariant": torch.zeros(1, 0),
        "batch": torch.zeros(1, dtype=torch.long),
        "atomic_numbers": torch.ones(1, dtype=torch.long),
        "total_charge": torch.zeros(1),
        "total_spin": torch.ones(1),
        "density_coefficients": torch.zeros(1, 4),
        "charges": torch.zeros(1),
        "molecular_dipole_eangstrom": torch.zeros(1, 3),
        "positions_angstrom": torch.zeros(1, 3),
    }
    entry = entry_dir / f"{key}.pt"
    torch.save(
        {
            "identity": identity,
            "feature_schema": f"stub:mode={mode}:cache",
            "symbols": ["H"],
            "tensors": tensors,
        },
        entry,
    )
    manifest = {
        "status": "complete",
        "mace_sha256": mace_sha,
        "physics_hash": physics_hash,
        "dtype": "float32",
        "entries": [
            {
                "path": str(entry.relative_to(cache)),
                "sha256": _sha(entry),
                "cache_key": key,
                "feature_mode": mode,
            }
        ],
    }
    (cache / "COMPLETE.json").write_text(json.dumps(manifest))
    loaded = load_prepared_feature_cache(
        cache,
        feature_mode=mode,
        mace_sha256=mace_sha,
        physics_hash=physics_hash,
        dtype=torch.float32,
    )
    assert set(loaded) == {key}
    (cache / "COMPLETE.json").unlink()
    with pytest.raises(RuntimeError, match="partial feature cache"):
        load_prepared_feature_cache(
            cache,
            feature_mode=mode,
            mace_sha256=mace_sha,
            physics_hash=physics_hash,
            dtype=torch.float32,
        )


def test_prepare_and_training_manifests_cover_reproducibility_fields():
    combined = "\n".join(path.read_text() for path in [PREPARE_PY, *JOBS])
    for field in (
        "source_commit",
        "environment",
        "mace_version",
        "mace_sha256",
        "feature_schemas",
        "dataset_identity",
        "dataset_counts",
        "dataset_hash",
        "preprocessing_hash",
        "split_hash",
        "physics_hash",
        "submodels",
        "seed",
        "parameter_counts",
        "elapsed",
        "maximum_resident",
    ):
        assert field in combined
    prepare_text = PREPARE_PY.read_text()
    assert "os.replace" in prepare_text
    assert "existing" in prepare_text and "skip" in prepare_text
    train_text = JOBS[2].read_text()
    assert "existing MODEL_OUT" in train_text
    assert "--resume" not in train_text
    assert "requeue_safe_checkpoint" in train_text


def test_submission_dry_run_has_full_matrix_and_afterok_graph(tmp_path):
    result = _run(SUBMIT, {**_submission_env(tmp_path), "DRY_RUN": "1"})
    assert result.returncode == 0, result.stderr
    output = result.stdout
    assert output.count("JOB prepare") == 1
    assert output.count("JOB atomic") == 3
    assert output.count("JOB pair") == 15
    for option in ("BASE", "H1", "H2", "DirectPolar", "AtomHead"):
        assert output.count(f"MODEL_OPTION={option}") == 3
    assert output.count("dependency=afterok:prepare") >= 4
    for seed in ("0", "1", "2"):
        assert f"dependency=afterok:atomic-{seed}" in output
    run_ids = re.findall(r"RUN_ID=([^\s]+)", output)
    assert len(run_ids) == len(set(run_ids))
    assert "FULL SCALE PROHIBITED" in output
    manifest = (
        Path(_submission_env(tmp_path)["RUN_ROOT"])
        / "contract-test"
        / "submission"
        / "jobs.tsv"
    )
    assert len(manifest.read_text().splitlines()) == 20


def test_readme_documents_policy_gate_and_cuda_limitations():
    text = (ROOT / "README.md").read_text()
    guide = (ROOT / "docs" / "mace-apnet-slurm.md").read_text()
    combined = text + guide
    assert ".[mace]" in combined
    assert "--check" in combined
    assert "DRY_RUN=1" in combined
    assert "ASL" in combined
    assert "Full-scale" in combined
    assert "CUDA" in combined
    assert "requeue_safe_checkpoint: false" in combined


def test_submission_uses_fake_sbatch_without_cluster_access(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    log = tmp_path / "sbatch.log"
    counter = tmp_path / "counter"
    fake_sbatch = fake_bin / "sbatch"
    fake_sbatch.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"echo \"$*\" >> {log}\n"
        f"n=$(cat {counter} 2>/dev/null || echo 1000)\n"
        "n=$((n + 1))\n"
        f"echo $n > {counter}\n"
        "echo $n\n"
    )
    fake_sbatch.chmod(0o755)
    env = _submission_env(tmp_path)
    env.update(
        SBATCH_BIN=str(fake_sbatch),
        DRY_RUN="0",
        PATH=f"{fake_bin}:{os.environ['PATH']}",
    )
    result = _run(SUBMIT, env)
    assert result.returncode == 0, result.stderr
    calls = log.read_text().splitlines()
    assert len(calls) == 19
    assert sum("--dependency=afterok:" in call for call in calls) == 18
    assert any("train_mace_atomic_properties.sbatch" in call for call in calls)
    assert sum("train_mace_ap3d3.sbatch" in call for call in calls) == 15
