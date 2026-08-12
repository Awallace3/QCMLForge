#!/usr/bin/env bash
#SBATCH --job-name=mace-ap3d3-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
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
verify_named_pair() {
    require_var "$1"; require_var "$2"
    verify_file "${!1}" "${!2}" "$3"
}

for variable in MODEL_OPTION SEED DATA_DIR MODEL_OUT RUN_ROOT RUN_ID \
    PHYSICS_CONFIG_PATH PHYSICS_CONFIG_SHA256 ELECTROSTATICS_MODE N_EPOCHS BATCH_SIZE; do
    require_var "$variable"
done
[[ "$SEED" =~ ^[0-9]+$ ]] || fail "SEED must be a non-negative integer"
[[ "$N_EPOCHS" =~ ^[1-9][0-9]*$ ]] || fail "N_EPOCHS must be positive"
[[ "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || fail "BATCH_SIZE must be positive"
[[ "$ELECTROSTATICS_MODE" =~ ^(damped-cliff|damped-amoeba|undamped)$ ]] || \
    fail "unsupported ELECTROSTATICS_MODE"
[[ -f "$DATA_DIR" ]] || fail "DATA_DIR smoke fixture is missing: $DATA_DIR"
verify_file "$PHYSICS_CONFIG_PATH" "$PHYSICS_CONFIG_SHA256" "physics config"

case "$MODEL_OPTION" in
    BASE) CLI_OPTION="APNet3-fused-d3"; ARCHITECTURE="baseline"; ROUTE="legacy" ;;
    H1|MACE-AP3D3-H1) CLI_OPTION="MACE-AP3D3-H1"; ARCHITECTURE="hybrid-h1"; ROUTE="legacy-mace" ;;
    H2|MACE-AP3D3-H2) CLI_OPTION="MACE-AP3D3-H2"; ARCHITECTURE="hybrid-h2"; ROUTE="legacy-mace" ;;
    DirectPolar|MACE-AP3D3-DirectPolar) CLI_OPTION="MACE-AP3D3-DirectPolar"; ARCHITECTURE="direct-polar"; ROUTE="atom-mace" ;;
    AtomHead|MACE-AP3D3-AtomHead) CLI_OPTION="MACE-AP3D3-AtomHead"; ARCHITECTURE="atomhead"; ROUTE="atom-mace" ;;
    *) fail "MODEL_OPTION must be BASE, H1, H2, DirectPolar, or AtomHead" ;;
esac

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_DIR="${RUN_ROOT}/${RUN_ID}"
[[ "$MODEL_OUT" == "${RUN_DIR}/checkpoints/model.pt" ]] || \
    fail "MODEL_OUT must be ${RUN_DIR}/checkpoints/model.pt"
[[ ! -e "$MODEL_OUT" ]] || \
    fail "existing MODEL_OUT is forbidden even when valid; use a new RUN_ID (no implicit resume)"
mkdir -p \
    "$RUN_DIR/logs" "$RUN_DIR/cache" "$RUN_DIR/data" \
    "$RUN_DIR/diagnostics" "$RUN_DIR/tmp" "$RUN_DIR/checkpoints"
export TMPDIR="$RUN_DIR/tmp"

SUBMODEL_RECORDS=()
if [[ "$ROUTE" == legacy* ]]; then
    verify_named_pair AM_MODEL_PATH AM_MODEL_SHA256 "legacy atom model"
    verify_named_pair ATOM_TYPE_PARAM_MODEL_PATH ATOM_TYPE_PARAM_MODEL_SHA256 \
        "legacy atom-type parameter model"
    verify_named_pair ATOM_TYPE_PARAM_MODEL_PATH2 ATOM_TYPE_PARAM_MODEL_SHA256_2 \
        "legacy dimer parameter model"
    SUBMODEL_RECORDS+=(
        "atom_model=${AM_MODEL_PATH}|${AM_MODEL_SHA256}"
        "atom_type_param_model=${ATOM_TYPE_PARAM_MODEL_PATH}|${ATOM_TYPE_PARAM_MODEL_SHA256}"
        "dimer_param_model=${ATOM_TYPE_PARAM_MODEL_PATH2}|${ATOM_TYPE_PARAM_MODEL_SHA256_2}"
    )
fi

if [[ "$ROUTE" == *-mace ]]; then
    verify_named_pair MACE_MODEL_PATH MACE_MODEL_SHA256 "MACE artifact"
    require_var FEATURE_CACHE_DIR
    [[ -f "$FEATURE_CACHE_DIR/COMPLETE.json" ]] || \
        fail "partial feature cache: COMPLETE.json is missing"
    "${PYTHON_BIN}" - "$FEATURE_CACHE_DIR/COMPLETE.json" "$MACE_MODEL_SHA256" <<'PY'
import json
import sys
from pathlib import Path
record = json.loads(Path(sys.argv[1]).read_text())
if record.get("status") != "complete":
    raise SystemExit("partial feature cache manifest")
if record.get("mace_sha256") != sys.argv[2]:
    raise SystemExit("feature cache MACE digest mismatch")
PY
    SUBMODEL_RECORDS+=("mace=${MACE_MODEL_PATH}|${MACE_MODEL_SHA256}")
fi

if [[ "$ROUTE" == atom-mace ]]; then
    require_var MACE_ATOM_MODEL_PATH
    if [[ -z "${MACE_ATOM_MODEL_SHA256:-}" ]]; then
        require_var MACE_ATOM_MODEL_SHA256_FILE
        [[ -f "$MACE_ATOM_MODEL_SHA256_FILE" ]] || \
            fail "MACE atom digest file is missing: $MACE_ATOM_MODEL_SHA256_FILE"
        read -r MACE_ATOM_MODEL_SHA256 < "$MACE_ATOM_MODEL_SHA256_FILE"
    fi
    verify_named_pair MACE_ATOM_MODEL_PATH MACE_ATOM_MODEL_SHA256 \
        "MACE atomic-property checkpoint"
    SUBMODEL_RECORDS+=("mace_atom_model=${MACE_ATOM_MODEL_PATH}|${MACE_ATOM_MODEL_SHA256}")
fi

PHYSICS_VALUES="$("$PYTHON_BIN" - "$PHYSICS_CONFIG_PATH" "$ELECTROSTATICS_MODE" <<'PY'
import json
import sys
from pathlib import Path
record = json.loads(Path(sys.argv[1]).read_text())
if record.get("electrostatics_mode") != sys.argv[2]:
    raise SystemExit("physics config electrostatics mode mismatch")
physics_hash = record.get("physics_hash")
if not isinstance(physics_hash, str) or len(physics_hash) != 64:
    raise SystemExit("physics config physics_hash is missing")
d3 = record.get("d3_parameters", [])
if len(d3) not in (0, 4):
    raise SystemExit("physics config d3_parameters must be empty or length four")
print(physics_hash + "|" + (",".join(str(value) for value in d3) if d3 else "default"))
PY
)" || fail "physics config validation failed"
PHYSICS_HASH="${PHYSICS_VALUES%%|*}"
D3_PARAMS="${PHYSICS_VALUES#*|}"

if [[ "${VALIDATE_ONLY:-0}" == 1 ]]; then
    printf 'validation complete for MODEL_OPTION=%s\n' "$MODEL_OPTION"
    exit 0
fi
[[ "$N_EPOCHS" -le 2 ]] || \
    fail "FULL SCALE PROHIBITED: small verification jobs are limited to two epochs"

SOURCE_COMMIT="$(git -C "$PROJECT_ROOT" rev-parse HEAD 2>/dev/null || printf unknown)"
{
    printf 'source_commit=%s\n' "$SOURCE_COMMIT"
    printf 'hostname=%s\n' "$(hostname)"
    printf 'python='; "$PYTHON_BIN" --version 2>&1
    printf 'slurm_job_id=%s\n' "${SLURM_JOB_ID:-local}"
    "$PYTHON_BIN" -m pip freeze 2>/dev/null || printf 'pip-freeze-unavailable\n'
} > "$RUN_DIR/diagnostics/environment.txt"
DATASET_SHA256="$(sha256_file "$DATA_DIR")"
START_EPOCH="$(date +%s)"

TRAIN_ARGS=(
    --train_apnet "$CLI_OPTION"
    --smoke_data_path "$DATA_DIR"
    --data_dir "$RUN_DIR/data"
    --ap_model_path "$MODEL_OUT"
    --n_epochs "$N_EPOCHS"
    --lr "${LEARNING_RATE:-5e-4}"
    --random_seed "$SEED"
    --world_size_ddp 1
    --omp_num_threads "${OMP_NUM_THREADS:-2}"
    --dataloader_num_workers 0
    --batch_size "$BATCH_SIZE"
    --skip_compile
    --include_total_mse
    --long_range_elst "$ELECTROSTATICS_MODE"
    --d3_params "$D3_PARAMS"
)
if [[ "$ROUTE" == legacy* ]]; then
    TRAIN_ARGS+=(
        --am_model_path "$AM_MODEL_PATH"
        --atom_type_param_model_path "$ATOM_TYPE_PARAM_MODEL_PATH"
        --atom_type_param_model_path2 "$ATOM_TYPE_PARAM_MODEL_PATH2"
    )
fi
if [[ "$ROUTE" == *-mace ]]; then
    TRAIN_ARGS+=(
        --mace_model polar-1-s
        --mace_model_path "$MACE_MODEL_PATH"
        --mace_model_sha256 "$MACE_MODEL_SHA256"
        --mace_offline
        --mace_cache_dir "$FEATURE_CACHE_DIR"
    )
fi
if [[ "$ROUTE" == atom-mace ]]; then
    TRAIN_ARGS+=(--mace_atom_model_path "$MACE_ATOM_MODEL_PATH")
fi

"$PYTHON_BIN" - "$RUN_DIR/manifest.before.json" "$MODEL_OPTION" "$ARCHITECTURE" \
    "$SEED" "$SOURCE_COMMIT" "$DATA_DIR" "$DATASET_SHA256" \
    "$PHYSICS_CONFIG_PATH" "$PHYSICS_CONFIG_SHA256" "$PHYSICS_HASH" \
    "$BATCH_SIZE" "$START_EPOCH" "${SUBMODEL_RECORDS[@]}" <<'PY'
import json
import pickle
import sys
from pathlib import Path
(output, option, architecture, seed, commit, dataset_path, dataset_sha,
 physics_path, physics_sha, physics_hash, batch_size, start_epoch,
 *submodels) = sys.argv[1:]
with Path(dataset_path).open("rb") as handle:
    fixture = pickle.load(handle)
submodel_records = {}
for value in submodels:
    name, record = value.split("=", 1)
    path, digest = record.rsplit("|", 1)
    submodel_records[name] = {"path": path, "sha256": digest}
manifest = {
    "status": "running",
    "model_option": option,
    "architecture": architecture,
    "seed": int(seed),
    "source_commit": commit,
    "environment": "diagnostics/environment.txt",
    "dataset_identity": {
        "path": dataset_path,
        "file_sha256": dataset_sha,
        "content_hash": fixture["content_hash"],
    },
    "dataset_counts": {"dimers": len(fixture["records"])},
    "dataset_hash": fixture["content_hash"],
    "preprocessing_hash": fixture["preprocessing_hash"],
    "split_hash": fixture["split_hash"],
    "physics_config": {"path": physics_path, "sha256": physics_sha},
    "physics_hash": physics_hash,
    "submodels": submodel_records,
    "batch_size": int(batch_size),
    "start_epoch": int(start_epoch),
    "parameter_counts": {"status": "pending model construction"},
    "requeue_safe_checkpoint": False,
    "requeue_limitation": "current harness has no periodic SIGTERM checkpoint hook",
}
Path(output).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
PY

cd "$PROJECT_ROOT"
srun "$PYTHON_BIN" scripts/run_with_metrics.py \
    --output "$RUN_DIR/diagnostics/time.txt" -- \
    "$PYTHON_BIN" -u train_models.py "${TRAIN_ARGS[@]}" \
    2>&1 | tee "$RUN_DIR/logs/train-pair.log"

[[ -f "$MODEL_OUT" ]] || fail "training completed without MODEL_OUT"
END_EPOCH="$(date +%s)"
ELAPSED="$((END_EPOCH - START_EPOCH))"
MAXIMUM_RESIDENT="$(awk -F: '/Maximum resident set size/ {gsub(/^ +/, "", $2); print $2}' "$RUN_DIR/diagnostics/time.txt")"
"$PYTHON_BIN" - "$RUN_DIR/manifest.before.json" "$RUN_DIR/manifest.json" \
    "$MODEL_OUT" "$MODEL_OUT.diagnostics.json" "$END_EPOCH" "$ELAPSED" \
    "$MAXIMUM_RESIDENT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path
import torch
(
    before, output, model_path, diagnostics_path, end_epoch, elapsed,
    maximum_resident,
) = sys.argv[1:]
manifest = json.loads(Path(before).read_text())
checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
config = checkpoint.get("config", {})
data = config.get("data", {})
expected_architecture = manifest["architecture"]
if expected_architecture == "baseline":
    if checkpoint.get("model_type") != "APNet3D3_AtomType_MPNN":
        raise SystemExit("fresh output model_type mismatch")
    actual_dataset = config.get("dataset_hash")
    actual_split = config.get("split_hash")
    actual_physics = config.get("physics_hash")
else:
    if checkpoint.get("model_type") != "MACEAP3D3":
        raise SystemExit("fresh output model_type mismatch")
    if config.get("architecture") != expected_architecture:
        raise SystemExit("fresh output architecture mismatch")
    mace = config.get("mace", {})
    expected_mace = manifest["submodels"]["mace"]["sha256"]
    if mace.get("sha256") != expected_mace:
        raise SystemExit("fresh output MACE digest mismatch")
    if not mace.get("feature_schema"):
        raise SystemExit("fresh output feature schema is missing")
    manifest["mace_version"] = mace.get("version")
    manifest["feature_schemas"] = {mace.get("feature_mode"): mace["feature_schema"]}
    actual_dataset = data.get("dataset_hash")
    actual_split = data.get("split_hash")
    actual_physics = config.get("physics", {}).get("physics_hash")
    expected_route = {
        name: record["sha256"]
        for name, record in manifest["submodels"].items()
        if name != "mace"
    }
    if config.get("route_submodel_digests") != expected_route:
        raise SystemExit("fresh output route submodel digest mismatch")
    if not Path(diagnostics_path).is_file():
        raise SystemExit("fresh output induction diagnostics are missing")
    induction = json.loads(Path(diagnostics_path).read_text())
    required = {"converged", "iterations", "residual", "policy"}
    if set(induction) != required:
        raise SystemExit("fresh output induction diagnostics are incomplete")
    manifest["induction_diagnostics"] = induction
if expected_architecture == "baseline":
    manifest["induction_diagnostics"] = {
        "converged": None,
        "iterations": None,
        "residual": None,
        "policy": "legacy APNet3 baseline does not expose SCF diagnostics",
    }
if actual_dataset != manifest["dataset_hash"]:
    raise SystemExit("fresh output dataset hash mismatch")
if actual_split != manifest["split_hash"]:
    raise SystemExit("fresh output split hash mismatch")
if actual_physics != manifest["physics_hash"]:
    raise SystemExit("fresh output physics hash mismatch")
counts = config.get("parameter_counts")
if counts is None:
    state = checkpoint.get("model_state_dict", {})
    counts = {"serialized": sum(value.numel() for value in state.values() if torch.is_tensor(value))}
manifest.update({
    "status": "complete",
    "checkpoint": {
        "path": model_path,
        "sha256": hashlib.sha256(Path(model_path).read_bytes()).hexdigest(),
    },
    "dataset_hash": actual_dataset,
    "preprocessing_hash": data.get(
        "preprocessing_hash", manifest["preprocessing_hash"]
    ),
    "split_hash": actual_split,
    "parameter_counts": counts,
    "end_epoch": int(end_epoch),
    "elapsed": int(elapsed),
    "maximum_resident": maximum_resident,
})
Path(output).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
PY
printf 'Pair checkpoint complete: %s\n' "$MODEL_OUT"
