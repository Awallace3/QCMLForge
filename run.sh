#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"
ITER="${ITER:-1}"
MODEL_DIR="${MODEL_DIR:-./models/ap3_saptpbe0/${ITER}}"
AM_MODEL_PATH="${AM_MODEL_PATH:-${MODEL_DIR}/am_ap2_${ITER}.pt}"
ATOM_TYPE_PARAM_MODEL_PATH="${ATOM_TYPE_PARAM_MODEL_PATH:-${MODEL_DIR}/atp_hfvr_${ITER}.pt}"
DATA_DIR="${DATA_DIR:-../qcmlforge/data_dir}"
RANDOM_SEED="${RANDOM_SEED:-${ITER}}"
N_EPOCHS="${N_EPOCHS:-25}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
N_RBF="${N_RBF:-8}"
N_NEURON="${N_NEURON:-64}"
N_EMBED="${N_EMBED:-8}"
SPEC_TYPE_AP="${SPEC_TYPE_AP:-2}"
DS_IN_MEMORY="${DS_IN_MEMORY:-True}"
WORLD_SIZE_DDP="${WORLD_SIZE_DDP:-1}"
# Deliberately not named OMP_NUM_THREADS: batch schedulers routinely export
# that standard OpenMP variable (e.g. SLURM sets it from --cpus-per-task),
# which would silently override the default below.
TRAIN_OMP_NUM_THREADS="${TRAIN_OMP_NUM_THREADS:-16}"
RACKERS_MODEL_OUT="${RACKERS_MODEL_OUT:-${MODEL_DIR}/rackers_thole_${ITER}.pt}"
RACKERS_OVERLAP_MODEL_OUT="${RACKERS_OVERLAP_MODEL_OUT:-${MODEL_DIR}/rackers_thole_overlap_${ITER}.pt}"

case "${WORLD_SIZE_DDP}" in
    1)
        ;;
    *)
        printf 'Error: WORLD_SIZE_DDP must be exactly 1 (got %q)\n' \
            "${WORLD_SIZE_DDP}" >&2
        exit 2
        ;;
esac

case "${DS_IN_MEMORY}" in
    [Tt][Rr][Uu][Ee])
        DS_IN_MEMORY=true
        ;;
    [Ff][Aa][Ll][Ss][Ee])
        DS_IN_MEMORY=false
        ;;
    *)
        printf 'Error: DS_IN_MEMORY must be true or false (got %q)\n' \
            "${DS_IN_MEMORY}" >&2
        exit 2
        ;;
esac

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
)

if [[ "${DS_IN_MEMORY}" == true ]]; then
    COMMON_ARGS+=(--ds_in_memory True)
fi

COMMON_ARGS+=(
    --world_size_ddp "${WORLD_SIZE_DDP}"
    --omp_num_threads "${TRAIN_OMP_NUM_THREADS}"
)

"${PYTHON}" -u ./train_models.py \
    --train_apnet RackersTholeDampingModel \
    --ap_model_path "${RACKERS_MODEL_OUT}" \
    "${COMMON_ARGS[@]}"

"${PYTHON}" -u ./train_models.py \
    --train_apnet RackersTholeDampingOverlapModel \
    --ap_model_path "${RACKERS_OVERLAP_MODEL_OUT}" \
    "${COMMON_ARGS[@]}"
