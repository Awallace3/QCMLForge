#!/usr/bin/env bash
set -euo pipefail

fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
require_var() { [[ -n "${!1:-}" ]] || fail "required variable $1 is unset"; }

for variable in RUN_ROOT MATRIX_ID MACE_MODEL_PATH MACE_MODEL_SHA256 \
    PAIR_DATA_PATH ATOM_DATA_PATH AM_MODEL_PATH AM_MODEL_SHA256 \
    ATOM_TYPE_PARAM_MODEL_PATH ATOM_TYPE_PARAM_MODEL_SHA256 \
    ATOM_TYPE_PARAM_MODEL_PATH2 ATOM_TYPE_PARAM_MODEL_SHA256_2 \
    PHYSICS_CONFIG_PATH PHYSICS_CONFIG_SHA256 ELECTROSTATICS_MODE \
    SMALL_VERIFICATION_APPROVED; do
    require_var "$variable"
done
[[ "$SMALL_VERIFICATION_APPROVED" == 1 ]] || \
    fail "FULL SCALE PROHIBITED until SMALL_VERIFICATION_APPROVED=1"

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
SBATCH_BIN="${SBATCH_BIN:-sbatch}"
DRY_RUN="${DRY_RUN:-0}"
N_EPOCHS="${N_EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-8}"
SEEDS=(0 1 2)
OPTIONS=(BASE H1 H2 DirectPolar AtomHead)
MATRIX_ROOT="${RUN_ROOT}/${MATRIX_ID}"
PREP_RUN_ID="prepare"
FEATURE_CACHE_DIR="${MATRIX_ROOT}/${PREP_RUN_ID}/cache/features"
PREP_MANIFEST="${MATRIX_ROOT}/${PREP_RUN_ID}/manifest.json"
mkdir -p "$MATRIX_ROOT/submission"
SUBMISSION_MANIFEST="$MATRIX_ROOT/submission/jobs.tsv"
[[ ! -e "$SUBMISSION_MANIFEST" ]] || \
    fail "MATRIX_ID already has a submission manifest; choose a unique MATRIX_ID"
printf 'kind\trun_id\tmodel_option\tjob_id\tdependency\tscript\n' \
    > "$SUBMISSION_MANIFEST"

[[ "$N_EPOCHS" == "1" ]] || \
    fail "VERIFICATION ONLY: matrix N_EPOCHS must be exactly one"
printf 'FULL SCALE PROHIBITED: VERIFICATION/SMOKE ONLY; no production training is implemented.\n'

LAST_JOB_ID=""
submit_job() {
    local kind="$1" run_id="$2" option="$3" dependency="$4" script="$5" exports="$6"
    local dependency_arg=()
    local slurm_log_dir="$MATRIX_ROOT/$run_id/logs"
    mkdir -p "$slurm_log_dir"
    if [[ -n "$dependency" ]]; then
        dependency_arg=("--dependency=afterok:${dependency}")
    fi
    if [[ "$DRY_RUN" == 1 ]]; then
        LAST_JOB_ID="$run_id"
        local display_exports="${exports//MODEL_OPTION=/OPTION=}"
        printf 'JOB %s RUN_ID=%s MODEL_OPTION=%s dependency=%s script=%s exports=%s\n' \
            "$kind" "$run_id" "$option" \
            "${dependency:+afterok:}${dependency:-none}" "$script" "$display_exports"
    else
        local submitted
        submitted="$(
            "$SBATCH_BIN" --parsable "${dependency_arg[@]}" \
                --output="$slurm_log_dir/slurm-%j.out" \
                --error="$slurm_log_dir/slurm-%j.err" \
                --export="ALL,${exports}" "$script"
        )"
        LAST_JOB_ID="${submitted%%;*}"
        printf 'SUBMITTED %s RUN_ID=%s MODEL_OPTION=%s job_id=%s dependency=%s\n' \
            "$kind" "$run_id" "$option" "$LAST_JOB_ID" "${dependency:-none}"
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$kind" "$run_id" "$option" "$LAST_JOB_ID" \
        "${dependency:-none}" "$script" >> "$SUBMISSION_MANIFEST"
}

PREP_EXPORTS="RUN_ROOT=${MATRIX_ROOT},RUN_ID=${PREP_RUN_ID},PROJECT_ROOT=${PROJECT_ROOT},MACE_MODEL_PATH=${MACE_MODEL_PATH},MACE_MODEL_SHA256=${MACE_MODEL_SHA256},PAIR_DATA_PATH=${PAIR_DATA_PATH},ATOM_DATA_PATH=${ATOM_DATA_PATH},FEATURE_CACHE_DIR=${FEATURE_CACHE_DIR}"
submit_job prepare "$PREP_RUN_ID" none "" \
    "$PROJECT_ROOT/scripts/slurm/prepare_mace_ap3d3_features.sbatch" "$PREP_EXPORTS"
PREP_JOB_ID="$LAST_JOB_ID"

for seed in "${SEEDS[@]}"; do
    run_id="atomic-${seed}"
    direct_model="${MATRIX_ROOT}/${run_id}/checkpoints/direct-completion.pt"
    learned_model="${MATRIX_ROOT}/${run_id}/checkpoints/learned.pt"
    exports="RUN_ROOT=${MATRIX_ROOT},RUN_ID=${run_id},PROJECT_ROOT=${PROJECT_ROOT},SEED=${seed},MACE_MODEL_PATH=${MACE_MODEL_PATH},MACE_MODEL_SHA256=${MACE_MODEL_SHA256},FEATURE_CACHE_DIR=${FEATURE_CACHE_DIR},PREP_MANIFEST=${PREP_MANIFEST},ATOM_DATA_PATH=${ATOM_DATA_PATH},DIRECT_MODEL_OUT=${direct_model},LEARNED_MODEL_OUT=${learned_model}"
    submit_job atomic "$run_id" none "$PREP_JOB_ID" \
        "$PROJECT_ROOT/scripts/slurm/train_mace_atomic_properties.sbatch" "$exports"
    eval "ATOMIC_JOB_${seed}=\$LAST_JOB_ID"
done

for seed in "${SEEDS[@]}"; do
    eval "atomic_job=\$ATOMIC_JOB_${seed}"
    for option in "${OPTIONS[@]}"; do
        run_id="pair-${option}-${seed}"
        model_out="${MATRIX_ROOT}/${run_id}/checkpoints/model.pt"
        dependency="$PREP_JOB_ID"
        route_exports=""
        case "$option" in
            BASE)
                route_exports="AM_MODEL_PATH=${AM_MODEL_PATH},AM_MODEL_SHA256=${AM_MODEL_SHA256},ATOM_TYPE_PARAM_MODEL_PATH=${ATOM_TYPE_PARAM_MODEL_PATH},ATOM_TYPE_PARAM_MODEL_SHA256=${ATOM_TYPE_PARAM_MODEL_SHA256},ATOM_TYPE_PARAM_MODEL_PATH2=${ATOM_TYPE_PARAM_MODEL_PATH2},ATOM_TYPE_PARAM_MODEL_SHA256_2=${ATOM_TYPE_PARAM_MODEL_SHA256_2}"
                ;;
            H1|H2)
                route_exports="MACE_MODEL_PATH=${MACE_MODEL_PATH},MACE_MODEL_SHA256=${MACE_MODEL_SHA256},FEATURE_CACHE_DIR=${FEATURE_CACHE_DIR},AM_MODEL_PATH=${AM_MODEL_PATH},AM_MODEL_SHA256=${AM_MODEL_SHA256},ATOM_TYPE_PARAM_MODEL_PATH=${ATOM_TYPE_PARAM_MODEL_PATH},ATOM_TYPE_PARAM_MODEL_SHA256=${ATOM_TYPE_PARAM_MODEL_SHA256},ATOM_TYPE_PARAM_MODEL_PATH2=${ATOM_TYPE_PARAM_MODEL_PATH2},ATOM_TYPE_PARAM_MODEL_SHA256_2=${ATOM_TYPE_PARAM_MODEL_SHA256_2}"
                ;;
            DirectPolar)
                dependency="$atomic_job"
                route_exports="MACE_MODEL_PATH=${MACE_MODEL_PATH},MACE_MODEL_SHA256=${MACE_MODEL_SHA256},FEATURE_CACHE_DIR=${FEATURE_CACHE_DIR},MACE_ATOM_MODEL_PATH=${MATRIX_ROOT}/atomic-${seed}/checkpoints/direct-completion.pt,MACE_ATOM_MODEL_SHA256_FILE=${MATRIX_ROOT}/atomic-${seed}/checkpoints/direct-completion.pt.sha256"
                ;;
            AtomHead)
                dependency="$atomic_job"
                route_exports="MACE_MODEL_PATH=${MACE_MODEL_PATH},MACE_MODEL_SHA256=${MACE_MODEL_SHA256},FEATURE_CACHE_DIR=${FEATURE_CACHE_DIR},MACE_ATOM_MODEL_PATH=${MATRIX_ROOT}/atomic-${seed}/checkpoints/learned.pt,MACE_ATOM_MODEL_SHA256_FILE=${MATRIX_ROOT}/atomic-${seed}/checkpoints/learned.pt.sha256"
                ;;
        esac
        exports="RUN_ROOT=${MATRIX_ROOT},RUN_ID=${run_id},PROJECT_ROOT=${PROJECT_ROOT},MODEL_OPTION=${option},SEED=${seed},DATA_DIR=${PAIR_DATA_PATH},MODEL_OUT=${model_out},PHYSICS_CONFIG_PATH=${PHYSICS_CONFIG_PATH},PHYSICS_CONFIG_SHA256=${PHYSICS_CONFIG_SHA256},ELECTROSTATICS_MODE=${ELECTROSTATICS_MODE},N_EPOCHS=${N_EPOCHS},BATCH_SIZE=${BATCH_SIZE},${route_exports}"
        submit_job pair "$run_id" "$option" "$dependency" \
            "$PROJECT_ROOT/scripts/slurm/train_mace_ap3d3.sbatch" "$exports"
    done
done

printf 'Submission matrix recorded under %s\n' "$MATRIX_ROOT/submission"
