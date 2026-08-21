#!/usr/bin/env bash
# Sequential launcher for the three CLIFF classical routes, with W&B logging.
#
# Trains, in order:
#   1. CliffExchangeModel         -- exchange alone, vs SAPT Exch
#   2. CliffClassicalModel        -- elst + exch + induction (no overlap term)
#   3. CliffClassicalOverlapModel -- same three, with the short-range
#                                    induction overlap correction
#
# All three land in one W&B group so they read as a single experiment.
set -euo pipefail

PYTHON="${PYTHON:-python3}"
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# The environment carries an editable install of apnet_pt that resolves to a
# *different* worktree, so without this the launcher trains code that has no
# CLIFF routes at all (an AttributeError at best, silently stale code at worst).
# Prepend this checkout's src and then assert we actually got it.
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
RESOLVED_APNET_PT="$("${PYTHON}" -c 'import apnet_pt; print(apnet_pt.__file__)')"
if [[ "${RESOLVED_APNET_PT}" != "${REPO_ROOT}/src/apnet_pt/"* ]]; then
    printf 'Error: apnet_pt resolves to %q, not this checkout (%q)\n' \
        "${RESOLVED_APNET_PT}" "${REPO_ROOT}/src/apnet_pt" >&2
    exit 2
fi
ITER="${ITER:-1}"
MODEL_DIR="${MODEL_DIR:-./models/cliff/${ITER}}"
# Prerequisites: an AtomMPNN multipole model and an AtomTypeParamNN HFVR /
# valence-width model. Both CLIFF routes read valence widths from the latter,
# so it must be an n_params=2 checkpoint whose config carries param_start_mean
# (older *_hfvr_vw.pt checkpoints predate that key and will KeyError).
AM_MODEL_PATH="${AM_MODEL_PATH:-./models/ap3_saptpbe0/1/am_ap2_1.pt}"
ATOM_TYPE_PARAM_MODEL_PATH="${ATOM_TYPE_PARAM_MODEL_PATH:-./models/ap3_saptpbe0/1/atp_hfvr_1.pt}"
DATA_DIR="${DATA_DIR:-../qcmlforge/data_dir}"
RANDOM_SEED="${RANDOM_SEED:-${ITER}}"

# Small-subset defaults: this is a first experimental run, not a production fit.
# DS_MAX_SIZE truncates both the train and test splits; unset it for the full
# ~1.5M-dimer set.
DS_MAX_SIZE="${DS_MAX_SIZE:-5000}"
N_EPOCHS="${N_EPOCHS:-15}"
# 1e-4 rather than the 5e-4 default. At 5e-4 over 100 epochs Adam's cumulative
# displacement budget (~lr per step) is large enough to carry a raw parameter
# the full distance from its seed into softplus saturation, which is exactly
# what the first 100-epoch run did to four of the five columns.
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
# Global gradient-norm clip. Empty disables it.
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-1.0}"
N_RBF="${N_RBF:-8}"
N_NEURON="${N_NEURON:-64}"
N_EMBED="${N_EMBED:-8}"
SPEC_TYPE_AP="${SPEC_TYPE_AP:-2}"
# Default false on purpose: in-memory loading yields an EMPTY dataset on the
# chunked spec_2 store (num_samples=0 at the DataLoader), and the full set
# would not fit in RAM regardless.
DS_IN_MEMORY="${DS_IN_MEMORY:-false}"
WORLD_SIZE_DDP="${WORLD_SIZE_DDP:-1}"
# Deliberately not named OMP_NUM_THREADS: batch schedulers routinely export
# that standard OpenMP variable (e.g. SLURM sets it from --cpus-per-task),
# which would silently override the default below.
TRAIN_OMP_NUM_THREADS="${TRAIN_OMP_NUM_THREADS:-16}"

# CLIFF Eq. (23) component/total weighting, applied to the two combined routes
# only. Unset means the legacy plain multi-column MSE; 0.4 is the paper's
# fitted value. CliffExchangeModel is single-component and never takes it.
COMPONENT_GAMMA="${COMPONENT_GAMMA:-0.4}"
TOTAL_INCLUDES_D3="${TOTAL_INCLUDES_D3:-false}"

WANDB_MODE_ARG="${WANDB_MODE_ARG:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-qcmlforge-cliff}"
WANDB_GROUP="${WANDB_GROUP:-cliff-classical-${ITER}}"
WANDB_ENTITY="${WANDB_ENTITY:-}"

EXCH_MODEL_OUT="${EXCH_MODEL_OUT:-${MODEL_DIR}/cliff_exch_${ITER}.pt}"
CLASSICAL_MODEL_OUT="${CLASSICAL_MODEL_OUT:-${MODEL_DIR}/cliff_classical_${ITER}.pt}"
CLASSICAL_OVERLAP_MODEL_OUT="${CLASSICAL_OVERLAP_MODEL_OUT:-${MODEL_DIR}/cliff_classical_overlap_${ITER}.pt}"

case "${WORLD_SIZE_DDP}" in
    1) ;;
    *)
        printf 'Error: WORLD_SIZE_DDP must be exactly 1 (got %q); the CLIFF heads have no DDP path\n' \
            "${WORLD_SIZE_DDP}" >&2
        exit 2
        ;;
esac

normalize_bool() {
    case "$2" in
        [Tt][Rr][Uu][Ee]) printf 'true' ;;
        [Ff][Aa][Ll][Ss][Ee]) printf 'false' ;;
        *)
            printf 'Error: %s must be true or false (got %q)\n' "$1" "$2" >&2
            exit 2
            ;;
    esac
}
# RANDOM_SEED defaults to ITER, so a non-numeric ITER (a run label like
# "smoke") would otherwise fail deep inside argparse. Catch it here instead.
if ! [[ "${RANDOM_SEED}" =~ ^-?[0-9]+$ ]]; then
    printf 'Error: RANDOM_SEED must be an integer (got %q). It defaults to ITER=%q, so set RANDOM_SEED explicitly when ITER is not numeric.\n' \
        "${RANDOM_SEED}" "${ITER}" >&2
    exit 2
fi

DS_IN_MEMORY="$(normalize_bool DS_IN_MEMORY "${DS_IN_MEMORY}")"
TOTAL_INCLUDES_D3="$(normalize_bool TOTAL_INCLUDES_D3 "${TOTAL_INCLUDES_D3}")"

if [[ -n "${COMPONENT_GAMMA}" ]]; then
    if ! awk -v g="${COMPONENT_GAMMA}" \
        'BEGIN { exit !(g ~ /^[0-9]*\.?[0-9]+$/ && g >= 0 && g <= 1) }'; then
        printf 'Error: COMPONENT_GAMMA must be a number in [0, 1] (got %q)\n' \
            "${COMPONENT_GAMMA}" >&2
        exit 2
    fi
fi
if [[ "${TOTAL_INCLUDES_D3}" == true && -z "${COMPONENT_GAMMA}" ]]; then
    printf 'Error: TOTAL_INCLUDES_D3=true requires an explicit COMPONENT_GAMMA\n' >&2
    exit 2
fi

for prerequisite in "${AM_MODEL_PATH}" "${ATOM_TYPE_PARAM_MODEL_PATH}"; do
    if [[ ! -f "${prerequisite}" ]]; then
        printf 'Error: prerequisite model not found: %q\n' "${prerequisite}" >&2
        exit 2
    fi
done

mkdir -p "${MODEL_DIR}"

COMMON_ARGS=(
    --am_model_path "${AM_MODEL_PATH}"
    --atom_type_param_model_path "${ATOM_TYPE_PARAM_MODEL_PATH}"
    --random_seed "${RANDOM_SEED}"
    --n_epochs "${N_EPOCHS}"
    --n_rbf "${N_RBF}"
    --n_neuron "${N_NEURON}"
    --n_embed "${N_EMBED}"
    --data_dir "${DATA_DIR}"
    --spec_type_ap "${SPEC_TYPE_AP}"
    --lr "${LEARNING_RATE}"
    --ds_in_memory "${DS_IN_MEMORY}"
    --world_size_ddp "${WORLD_SIZE_DDP}"
    --omp_num_threads "${TRAIN_OMP_NUM_THREADS}"
    --wandb-mode "${WANDB_MODE_ARG}"
    --wandb-project "${WANDB_PROJECT}"
    --wandb-group "${WANDB_GROUP}"
)

if [[ -n "${DS_MAX_SIZE}" ]]; then
    COMMON_ARGS+=(--ds_max_size "${DS_MAX_SIZE}")
fi
if [[ -n "${GRAD_CLIP_NORM}" ]]; then
    if ! awk -v c="${GRAD_CLIP_NORM}" \
        'BEGIN { exit !(c ~ /^[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$/ && c > 0) }'; then
        printf 'Error: GRAD_CLIP_NORM must be a positive number (got %q)\n' \
            "${GRAD_CLIP_NORM}" >&2
        exit 2
    fi
    COMMON_ARGS+=(--grad_clip_norm "${GRAD_CLIP_NORM}")
fi
if [[ -n "${WANDB_ENTITY}" ]]; then
    COMMON_ARGS+=(--wandb-entity "${WANDB_ENTITY}")
fi

# Applied to the combined routes only.
COMBINED_ARGS=()
if [[ -n "${COMPONENT_GAMMA}" ]]; then
    COMBINED_ARGS+=(--component_gamma "${COMPONENT_GAMMA}")
fi
if [[ "${TOTAL_INCLUDES_D3}" == true ]]; then
    COMBINED_ARGS+=(--total_includes_d3)
fi

printf '\n=== [1/3] CliffExchangeModel -> %s ===\n\n' "${EXCH_MODEL_OUT}"
"${PYTHON}" -u ./train_models.py \
    --train_apnet CliffExchangeModel \
    --ap_model_path "${EXCH_MODEL_OUT}" \
    --wandb-name "cliff-exch-${ITER}" \
    --wandb-tags cliff exchange \
    "${COMMON_ARGS[@]}"

printf '\n=== [2/3] CliffClassicalModel -> %s ===\n\n' "${CLASSICAL_MODEL_OUT}"
"${PYTHON}" -u ./train_models.py \
    --train_apnet CliffClassicalModel \
    --ap_model_path "${CLASSICAL_MODEL_OUT}" \
    --wandb-name "cliff-classical-${ITER}" \
    --wandb-tags cliff classical \
    "${COMMON_ARGS[@]}" "${COMBINED_ARGS[@]}"

printf '\n=== [3/3] CliffClassicalOverlapModel -> %s ===\n\n' "${CLASSICAL_OVERLAP_MODEL_OUT}"
"${PYTHON}" -u ./train_models.py \
    --train_apnet CliffClassicalOverlapModel \
    --ap_model_path "${CLASSICAL_OVERLAP_MODEL_OUT}" \
    --wandb-name "cliff-classical-overlap-${ITER}" \
    --wandb-tags cliff classical overlap \
    "${COMMON_ARGS[@]}" "${COMBINED_ARGS[@]}"

printf '\nAll three CLIFF routes finished. W&B group: %s\n' "${WANDB_GROUP}"
