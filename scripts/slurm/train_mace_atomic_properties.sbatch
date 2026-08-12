#!/usr/bin/env bash
#SBATCH --job-name=mace-atomic-heads
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --requeue

set -euo pipefail

fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
require_var() { [[ -n "${!1:-}" ]] || fail "required variable $1 is unset"; }
sha256_file() { sha256sum "$1" | awk '{print $1}'; }
verify_file() {
    local path="$1" expected="$2" label="$3" actual
    [[ -f "$path" ]] || fail "$label is missing: $path"
    actual="$(sha256_file "$path")"
    [[ "$actual" == "$expected" ]] || fail "$label SHA-256 mismatch: expected $expected, got $actual"
}

for variable in RUN_ROOT RUN_ID SEED MACE_MODEL_PATH MACE_MODEL_SHA256 \
    FEATURE_CACHE_DIR PREP_MANIFEST ATOM_DATA_PATH; do
    require_var "$variable"
done

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_DIR="${RUN_ROOT}/${RUN_ID}"
DIRECT_MODEL_OUT="${DIRECT_MODEL_OUT:-${RUN_DIR}/checkpoints/direct-completion.pt}"
LEARNED_MODEL_OUT="${LEARNED_MODEL_OUT:-${RUN_DIR}/checkpoints/learned.pt}"
[[ "$DIRECT_MODEL_OUT" == "${RUN_DIR}/checkpoints/direct-completion.pt" ]] || \
    fail "DIRECT_MODEL_OUT must be inside the unique run checkpoint directory"
[[ "$LEARNED_MODEL_OUT" == "${RUN_DIR}/checkpoints/learned.pt" ]] || \
    fail "LEARNED_MODEL_OUT must be inside the unique run checkpoint directory"

mkdir -p \
    "$RUN_DIR/logs" "$RUN_DIR/cache" "$RUN_DIR/data" \
    "$RUN_DIR/diagnostics" "$RUN_DIR/tmp" "$RUN_DIR/checkpoints"
export TMPDIR="$RUN_DIR/tmp"
verify_file "$MACE_MODEL_PATH" "$MACE_MODEL_SHA256" "MACE artifact"
[[ -f "$FEATURE_CACHE_DIR/COMPLETE.json" ]] || \
    fail "partial feature cache: COMPLETE.json is missing"
[[ -f "$PREP_MANIFEST" ]] || fail "feature-preparation manifest is missing: $PREP_MANIFEST"
[[ -f "$ATOM_DATA_PATH" ]] || fail "atomic smoke dataset is missing: $ATOM_DATA_PATH"
[[ ! -e "$DIRECT_MODEL_OUT" && ! -e "$LEARNED_MODEL_OUT" ]] || \
    fail "existing atomic MODEL_OUT is forbidden; use a new RUN_ID (no implicit resume)"

"$PYTHON_BIN" - "$FEATURE_CACHE_DIR/COMPLETE.json" "$MACE_MODEL_SHA256" <<'PY'
import json
import sys
from pathlib import Path
record = json.loads(Path(sys.argv[1]).read_text())
if record.get("status") != "complete":
    raise SystemExit("partial feature cache manifest")
if record.get("mace_sha256") != sys.argv[2]:
    raise SystemExit("feature cache MACE digest mismatch")
PY

SOURCE_COMMIT="$(git -C "$PROJECT_ROOT" rev-parse HEAD 2>/dev/null || printf unknown)"
{
    printf 'source_commit=%s\n' "$SOURCE_COMMIT"
    printf 'hostname=%s\n' "$(hostname)"
    printf 'python='; "$PYTHON_BIN" --version 2>&1
    printf 'slurm_job_id=%s\n' "${SLURM_JOB_ID:-local}"
    "$PYTHON_BIN" -m pip freeze 2>/dev/null || printf 'pip-freeze-unavailable\n'
} > "$RUN_DIR/diagnostics/environment.txt"
START_EPOCH="$(date +%s)"
"$PYTHON_BIN" - "$RUN_DIR/manifest.before.json" "$ATOM_DATA_PATH" \
    "$PREP_MANIFEST" "$SOURCE_COMMIT" "$MACE_MODEL_PATH" \
    "$MACE_MODEL_SHA256" "$SEED" "$START_EPOCH" <<'PY'
import json
import pickle
import sys
from pathlib import Path
output, atom_data, prep, commit, mace_path, mace_sha, seed, start_epoch = sys.argv[1:]
with Path(atom_data).open("rb") as handle:
    fixture = pickle.load(handle)
record = {
    "status": "running",
    "source_commit": commit,
    "environment": "diagnostics/environment.txt",
    "mace_version": "recorded in feature preparation manifest",
    "mace_sha256": mace_sha,
    "feature_schemas": "recorded in feature preparation manifest",
    "dataset_identity": {
        "path": atom_data,
        "content_hash": fixture["content_hash"],
    },
    "dataset_counts": {"atomic_monomers": len(fixture["records"])},
    "dataset_hash": fixture["content_hash"],
    "preprocessing_hash": fixture["preprocessing_hash"],
    "split_hash": fixture["split_hash"],
    "physics_hash": None,
    "seed": int(seed),
    "start_epoch": int(start_epoch),
    "submodels": {
        "mace": {"path": str(Path(mace_path).resolve()), "sha256": mace_sha}
    },
    "parameter_counts": {"status": "pending model construction"},
    "feature_preparation_manifest": str(Path(prep).resolve()),
    "requeue_safe_checkpoint": False,
}
Path(output).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
PY

export PYTHON_BIN MACE_MODEL_PATH MACE_MODEL_SHA256 FEATURE_CACHE_DIR
export ATOM_DATA_PATH SEED DIRECT_MODEL_OUT LEARNED_MODEL_OUT
cat > "$RUN_DIR/diagnostics/run-atomic.sh" <<'BASH'
#!/usr/bin/env bash
set -euo pipefail
"$PYTHON_BIN" -u train_models.py \
    --train_am MACE-AtomicProperties \
    --mace_property_mode direct-completion \
    --mace_model polar-1-s \
    --mace_model_path "$MACE_MODEL_PATH" \
    --mace_model_sha256 "$MACE_MODEL_SHA256" \
    --mace_offline \
    --mace_cache_dir "$FEATURE_CACHE_DIR" \
    --smoke_atom_data_path "$ATOM_DATA_PATH" \
    --n_epochs_atom 1 --random_seed "$SEED" --world_size_ddp 1 \
    --dataloader_num_workers 0 --skip_compile \
    --am_model_path "$DIRECT_MODEL_OUT"
"$PYTHON_BIN" -u train_models.py \
    --train_am MACE-AtomicProperties \
    --mace_property_mode learned \
    --mace_model polar-1-s \
    --mace_model_path "$MACE_MODEL_PATH" \
    --mace_model_sha256 "$MACE_MODEL_SHA256" \
    --mace_offline \
    --mace_cache_dir "$FEATURE_CACHE_DIR" \
    --smoke_atom_data_path "$ATOM_DATA_PATH" \
    --n_epochs_atom 1 --random_seed "$SEED" --world_size_ddp 1 \
    --dataloader_num_workers 0 --skip_compile \
    --am_model_path "$LEARNED_MODEL_OUT"
BASH
chmod 700 "$RUN_DIR/diagnostics/run-atomic.sh"
cd "$PROJECT_ROOT"
srun "$PYTHON_BIN" scripts/run_with_metrics.py \
    --output "$RUN_DIR/diagnostics/time.txt" -- \
    bash "$RUN_DIR/diagnostics/run-atomic.sh" \
    2>&1 | tee "$RUN_DIR/logs/train-atomic.log"

[[ -f "$DIRECT_MODEL_OUT" && -f "$LEARNED_MODEL_OUT" ]] || \
    fail "atomic training did not produce both required checkpoints"
sha256_file "$DIRECT_MODEL_OUT" > "$DIRECT_MODEL_OUT.sha256"
sha256_file "$LEARNED_MODEL_OUT" > "$LEARNED_MODEL_OUT.sha256"
END_EPOCH="$(date +%s)"
ELAPSED="$((END_EPOCH - START_EPOCH))"
MAXIMUM_RESIDENT="$(awk -F: '/Maximum resident set size/ {gsub(/^ +/, "", $2); print $2}' "$RUN_DIR/diagnostics/time.txt")"
"$PYTHON_BIN" - "$RUN_DIR/manifest.json" "$PREP_MANIFEST" "$ATOM_DATA_PATH" \
    "$SOURCE_COMMIT" "$MACE_MODEL_PATH" "$MACE_MODEL_SHA256" "$SEED" \
    "$DIRECT_MODEL_OUT" "$LEARNED_MODEL_OUT" "$START_EPOCH" "$END_EPOCH" \
    "$ELAPSED" "$MAXIMUM_RESIDENT" <<'PY'
import hashlib
import json
import pickle
import sys
from pathlib import Path
import torch

(output, prep_manifest, atom_data, commit, mace_path, mace_sha, seed,
 direct_path, learned_path, started, ended, elapsed,
 maximum_resident) = sys.argv[1:]
with Path(atom_data).open("rb") as handle:
    fixture = pickle.load(handle)
def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def count(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    return sum(value.numel() for value in checkpoint["model_state_dict"].values() if torch.is_tensor(value))
record = {
    "status": "complete",
    "source_commit": commit,
    "environment": "diagnostics/environment.txt",
    "mace_version": "recorded in feature preparation manifest",
    "mace_path": str(Path(mace_path).resolve()),
    "mace_sha256": mace_sha,
    "feature_schemas": "recorded in feature preparation manifest",
    "dataset_identity": fixture["content_hash"],
    "dataset_counts": {"atomic_monomers": len(fixture["records"])},
    "dataset_hash": fixture["content_hash"],
    "preprocessing_hash": fixture["preprocessing_hash"],
    "split_hash": fixture["split_hash"],
    "physics_hash": None,
    "seed": int(seed),
    "submodels": {
        "direct-completion": {"path": direct_path, "sha256": digest(direct_path)},
        "learned": {"path": learned_path, "sha256": digest(learned_path)},
    },
    "parameter_counts": {
        "direct-completion": count(direct_path),
        "learned": count(learned_path),
    },
    "feature_preparation_manifest": str(Path(prep_manifest).resolve()),
    "start_epoch": int(started),
    "end_epoch": int(ended),
    "elapsed": int(elapsed),
    "maximum_resident": maximum_resident,
    "requeue_safe_checkpoint": False,
    "requeue_limitation": "current harness writes only final one-epoch checkpoints",
}
Path(output).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
PY
printf 'Atomic head checkpoints complete: %s %s\n' "$DIRECT_MODEL_OUT" "$LEARNED_MODEL_OUT"
