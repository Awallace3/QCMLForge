#!/usr/bin/env bash
#SBATCH --job-name=mace-ap3d3-prepare
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
    [[ "$expected" =~ ^[0-9a-f]{64}$ ]] || fail "$label digest is not lowercase SHA-256"
    actual="$(sha256_file "$path")"
    [[ "$actual" == "$expected" ]] || fail "$label SHA-256 mismatch: expected $expected, got $actual"
}

for variable in RUN_ROOT RUN_ID MACE_MODEL_PATH MACE_MODEL_SHA256 PAIR_DATA_PATH ATOM_DATA_PATH; do
    require_var "$variable"
done

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_DIR="${RUN_ROOT}/${RUN_ID}"
FEATURE_CACHE_DIR="${FEATURE_CACHE_DIR:-${RUN_DIR}/cache/features}"
[[ "$FEATURE_CACHE_DIR" == "${RUN_DIR}/cache/features" ]] || \
    fail "FEATURE_CACHE_DIR must be the unique run cache: ${RUN_DIR}/cache/features"

mkdir -p \
    "$RUN_DIR/logs" "$RUN_DIR/cache" "$RUN_DIR/data" \
    "$RUN_DIR/diagnostics" "$RUN_DIR/tmp" "$RUN_DIR/checkpoints"
export TMPDIR="$RUN_DIR/tmp"
verify_file "$MACE_MODEL_PATH" "$MACE_MODEL_SHA256" "MACE artifact"
verify_file "$PAIR_DATA_PATH" "$(sha256_file "$PAIR_DATA_PATH")" "pair dataset"
verify_file "$ATOM_DATA_PATH" "$(sha256_file "$ATOM_DATA_PATH")" "atomic dataset"

SOURCE_COMMIT="$(git -C "$PROJECT_ROOT" rev-parse HEAD 2>/dev/null || printf unknown)"
{
    printf 'source_commit=%s\n' "$SOURCE_COMMIT"
    printf 'hostname=%s\n' "$(hostname)"
    printf 'python='; "$PYTHON_BIN" --version 2>&1
    printf 'slurm_job_id=%s\n' "${SLURM_JOB_ID:-local}"
    "$PYTHON_BIN" -m pip freeze 2>/dev/null || printf 'pip-freeze-unavailable\n'
} > "$RUN_DIR/diagnostics/environment.txt"

# The helper uses digest-verified local loading with offline=True. This is the
# runtime equivalent of train_models.py --mace_offline; no download is allowed.
export QCMLFORGE_MACE_OFFLINE=1
START_EPOCH="$(date +%s)"
printf '{"status":"running","source_commit":"%s","mace_sha256":"%s","environment":"diagnostics/environment.txt","start_epoch":%s}\n' \
    "$SOURCE_COMMIT" "$MACE_MODEL_SHA256" "$START_EPOCH" \
    > "$RUN_DIR/manifest.running.json"

cd "$PROJECT_ROOT"
srun "$PYTHON_BIN" scripts/run_with_metrics.py \
    --output "$RUN_DIR/diagnostics/time.txt" -- \
    "$PYTHON_BIN" -u scripts/prepare_mace_ap3d3_features.py \
    --mace-path "$MACE_MODEL_PATH" --mace-sha256 "$MACE_MODEL_SHA256" \
    --pair-data "$PAIR_DATA_PATH" --atom-data "$ATOM_DATA_PATH" \
    --cache-dir "$FEATURE_CACHE_DIR" --device "${PREPARE_DEVICE:-cpu}" \
    --dtype "${MACE_DEFAULT_DTYPE:-float32}" \
    2>&1 | tee "$RUN_DIR/logs/prepare.log"

[[ -f "$FEATURE_CACHE_DIR/COMPLETE.json" ]] || \
    fail "complete cache manifest was not produced; partial cache is rejected"
END_EPOCH="$(date +%s)"
MAXIMUM_RESIDENT="$(awk -F: '/Maximum resident set size/ {gsub(/^ +/, "", $2); print $2}' "$RUN_DIR/diagnostics/time.txt")"
ELAPSED="$((END_EPOCH - START_EPOCH))"
"$PYTHON_BIN" - "$FEATURE_CACHE_DIR/COMPLETE.json" "$RUN_DIR/manifest.json" \
    "$SOURCE_COMMIT" "$START_EPOCH" "$END_EPOCH" "$ELAPSED" "$MAXIMUM_RESIDENT" <<'PY'
import json
import sys
from pathlib import Path

cache_manifest, output, commit, started, ended, elapsed, maximum_resident = sys.argv[1:]
record = json.loads(Path(cache_manifest).read_text())
record.update({
    "source_commit": commit,
    "start_epoch": int(started),
    "end_epoch": int(ended),
    "elapsed": int(elapsed),
    "maximum_resident": maximum_resident,
    "environment": "diagnostics/environment.txt",
    "requeue_safe_checkpoint": True,
})
Path(output).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
PY
rm -f "$RUN_DIR/manifest.running.json"
printf 'Prepared complete feature cache: %s\n' "$FEATURE_CACHE_DIR"
