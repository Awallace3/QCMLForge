#!/bin/bash

set -euo pipefail

# Retrain of the whole AP3D3 SAPT(PBE0) stack on top of the edgeless-atom fix
# in AtomMPNN.forward.
#
# The previous stack was trained through a defect: the atom model dropped atoms
# with no intramonomer edge, then scattered the dipole and quadrupole messages
# with the unfiltered atom count, so one monatomic monomer shifted every later
# atom's multipoles onto the wrong atom.  3.98% of Splinter dimers have a
# monatomic monomer, and at the training batch sizes each one corrupted the rest
# of its batch, so the damage is not confined to those dimers.
#
# Every stage is therefore re-enabled, including the two atom-level stages that
# are commented out in train_ap3d3_saptdft_local_1.sh.  The fix lives in the
# AtomModel and everything downstream consumes its h_list and multipoles, so
# reusing the old am_ap2_1.pt / atp_hfvr_1.pt would leave the corruption baked
# into the frozen inputs.
#
# MODEL_DIR is new so nothing in models/ap3_saptpbe0/${ITER} is touched.  The
# original script hardcoded every literal path; here they all derive from
# MODEL_DIR, so the directory is a single knob.

ITER=1
MODEL_DIR=${MODEL_DIR:-./models/ap3_saptpbe0_edgeless_fix/${ITER}}
DATA_DIR=${DATA_DIR:-../qcmlforge/data_dir}

# Every stage's training data sits in a different directory on Phoenix -- spec 4
# in data_0, spec 1 in data_dir, the spec 2 / spec 10 dimers in data_dimer_1 --
# so the data dir is per stage.  Each one defaults to DATA_DIR, which keeps a
# single-directory local run working unchanged.
DATA_DIR_S1=${DATA_DIR_S1:-${DATA_DIR}}
DATA_DIR_S2=${DATA_DIR_S2:-${DATA_DIR}}
DATA_DIR_S3=${DATA_DIR_S3:-${DATA_DIR}}
DATA_DIR_S4=${DATA_DIR_S4:-${DATA_DIR}}
DATA_DIR_S5=${DATA_DIR_S5:-${DATA_DIR}}

# All five stages report to one dedicated W&B project, grouped so the sequence
# reads as a single retrain.  --wandb-mode defaults to "disabled" in
# train_models.py, so every stage passes "online" explicitly; without it the
# runs would train fine and log nothing.
WANDB_PROJECT=${WANDB_PROJECT:-ap3d3-edgeless-fix-retrain}
WANDB_GROUP=${WANDB_GROUP:-ap3d3-saptpbe0-edgeless-fix-iter${ITER}}
WANDB_DIR=${WANDB_DIR:-${MODEL_DIR}/wandb}

# No DDP anywhere.  batch_size is hardcoded to 16 in train_models.py and
# DistributedSampler shards the dataset per rank, so --world_size_ddp 4 was
# training the atom model at an effective batch of 64.  Single-process keeps the
# AP2 hyperparameters intact, and gloo all-reduce buys little at batch 16.

# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"

# qcml_main resolves apnet_pt out of a different worktree unless PYTHONPATH
# wins, and a retrain against the unfixed source would be indistinguishable
# from success until the numbers came back unchanged.
python3 -c "
import sys, pathlib, apnet_pt
from apnet_pt.AtomModels import ap2_atom_model
resolved = pathlib.Path(apnet_pt.__file__).resolve()
expected = pathlib.Path('${REPO_ROOT}/src/apnet_pt/__init__.py').resolve()
if resolved != expected:
    sys.exit(f'apnet_pt resolved to {resolved}, expected {expected}')
source = pathlib.Path(ap2_atom_model.__file__).read_text()
if 'keep_mask' in source:
    sys.exit('AtomMPNN.forward still filters edgeless atoms; the fix is absent')
print(f'preflight ok: apnet_pt at {resolved}, edgeless fix present')
"

# The original script's relative ../qcmlforge/data_dir resolves only from the
# author's checkout layout, not from a worktree.  Fail here rather than after
# stage 1 has already spent GPU hours.
for d in "${DATA_DIR_S1}" "${DATA_DIR_S2}" "${DATA_DIR_S3}" "${DATA_DIR_S4}" "${DATA_DIR_S5}"; do
    if [ ! -d "${d}" ]; then
        echo "Data dir ${d} does not exist (cwd $(pwd))." >&2
        echo "Set DATA_DIR, or DATA_DIR_S1..DATA_DIR_S5, and re-run." >&2
        exit 1
    fi
done


# STAGES selects which stages this invocation runs, so the stack can be split
# across jobs: stage 1's training data (spec 4) is on Phoenix already, while
# stages 2 and 5 need spec 1 / spec 10 staged there first.  Stage N still
# requires stage N-1's checkpoint to exist, which is asserted below.
STAGES=${STAGES:-1 2 3 4 5}

# skip_stage <n> <output.pt>
#
# Nothing is ever overwritten by accident.  A stage whose completion marker
# exists is redone only under RESUME=1, so a job that dies in stage 4 picks up
# there instead of repeating 500 atom epochs.  A stage whose output exists with
# *no* marker is a partial or foreign checkpoint, and that always refuses --
# which is also what protects models/ap3_saptpbe0/${ITER} if MODEL_DIR is ever
# pointed back at it.  The marker is written only after the stage exits 0, which
# matters for stage 5: its checkpoint exists from the `cp` before a single epoch
# has run.
skip_stage() {
    local n="$1" out="$2"
    case " ${STAGES} " in
        *" ${n} "*) ;;
        *) echo "STAGES=${STAGES}: stage ${n} not selected, skipping" >&2; return 0 ;;
    esac
    if [ -f "${MODEL_DIR}/.stage${n}.done" ]; then
        if [ "${RESUME:-0}" = "1" ]; then
            echo "RESUME: stage ${n} already complete, skipping" >&2
            return 0
        fi
        echo "Stage ${n} already completed in ${MODEL_DIR}." >&2
        echo "Pass RESUME=1 to skip completed stages, or bump ITER/MODEL_DIR." >&2
        exit 1
    fi
    if [ -e "${out}" ]; then
        echo "Stage ${n} output ${out} exists with no completion marker." >&2
        echo "Refusing to overwrite a partial or foreign checkpoint." >&2
        exit 1
    fi
    return 1
}

# A stage silently training against a missing upstream checkpoint is the one
# failure mode that would waste the whole run, so require the inputs up front.
require_input() {
    if [ ! -f "$1" ]; then
        echo "Missing upstream checkpoint $1" >&2
        echo "Run the earlier stages first (STAGES=... ) or point MODEL_DIR at them." >&2
        exit 1
    fi
}

mkdir -p "${MODEL_DIR}" "${WANDB_DIR}"

# wandb needs credentials before stage 1 starts, not 500 epochs later.
python3 -c "
import sys, netrc, os, pathlib
if os.environ.get('WANDB_API_KEY'):
    sys.exit(0)
try:
    hosts = netrc.netrc(pathlib.Path.home() / '.netrc').hosts
except Exception as exc:
    sys.exit(f'no WANDB_API_KEY and ~/.netrc unreadable ({exc}); run wandb login')
if not any('wandb' in host for host in hosts):
    sys.exit('no WANDB_API_KEY and no wandb entry in ~/.netrc; run wandb login')
"

# ---------------------------------------------------------------------------
# Stage 1: AP2 AtomMPNN on PBE0 monomers (spec 4 -> monomers_ap3_spec_1_pbe0.pkl)
#
# This is the stage the fix is in.  Re-enabled.
# ---------------------------------------------------------------------------
if ! skip_stage 1 "${MODEL_DIR}/am_ap2_1.pt"; then
python3 \
    -u \
    ./train_models.py \
    --train_am \
    AtomModel \
    --am_model_path \
    "${MODEL_DIR}/am_ap2_1.pt" \
    --random_seed \
    1 \
    --n_epochs_atom \
    500 \
    --lr \
    5e-4 \
    --n_message_atom \
    3 \
    --n_rbf_atom \
    8 \
    --n_neuron_atom \
    128 \
    --n_embed_atom \
    8 \
    --data_dir \
    "${DATA_DIR_S1}" \
    --spec_type_am \
    4 \
    --world_size_ddp \
    1 \
    --omp_num_threads \
    8 \
    --wandb-mode \
    online \
    --wandb-project \
    "${WANDB_PROJECT}" \
    --wandb-group \
    "${WANDB_GROUP}" \
    --wandb-name \
    s1-am-ap2 \
    --wandb-job-type \
    atom-model \
    --wandb-tags \
    edgeless-fix ap3d3 saptpbe0 retrain \
    --wandb-dir \
    "${WANDB_DIR}"
touch "${MODEL_DIR}/.stage1.done"
fi

# ---------------------------------------------------------------------------
# Stage 2: Hirshfeld volume-ratio/valence-width AtomTypeParamNN on PBE0
# monomers (spec 1) using AP2 h_list.
#
# Consumes the stage-1 atom model, so it must be retrained too.  Re-enabled.
# ---------------------------------------------------------------------------
if ! skip_stage 2 "${MODEL_DIR}/atp_hfvr_1.pt"; then
require_input "${MODEL_DIR}/am_ap2_1.pt"
python3 \
    -u \
    ./train_models.py \
    --train_apnet \
    AtomTypeParamModel \
    --am_model_path \
    "${MODEL_DIR}/am_ap2_1.pt" \
    --random_seed \
    1 \
    --lr \
    5e-5 \
    --ap_model_path \
    "${MODEL_DIR}/atp_hfvr_1.pt" \
    --n_epochs \
    100 \
    --n_rbf \
    8 \
    --n_neuron \
    32 \
    --n_embed \
    8 \
    --data_dir \
    "${DATA_DIR_S2}" \
    --spec_type_ap \
    1 \
    --world_size_ddp \
    1 \
    --omp_num_threads \
    16 \
    --wandb-mode \
    online \
    --wandb-project \
    "${WANDB_PROJECT}" \
    --wandb-group \
    "${WANDB_GROUP}" \
    --wandb-name \
    s2-atp-hfvr \
    --wandb-job-type \
    atom-type-param \
    --wandb-tags \
    edgeless-fix ap3d3 saptpbe0 retrain \
    --wandb-dir \
    "${WANDB_DIR}"
touch "${MODEL_DIR}/.stage2.done"
fi

# ---------------------------------------------------------------------------
# Stage 3: Electrostatic K AtomTypeParamNN on Splinter SAPT0/aug-cc-pVDZ
# dimers (spec 2)
# ---------------------------------------------------------------------------
if ! skip_stage 3 "${MODEL_DIR}/atp_elst_1.pt"; then
require_input "${MODEL_DIR}/am_ap2_1.pt"
require_input "${MODEL_DIR}/atp_hfvr_1.pt"
python3 \
    -u \
    ./train_models.py \
    --train_apnet \
    AM-DimerParam \
    --am_model_path \
    "${MODEL_DIR}/am_ap2_1.pt" \
    --atom_type_param_model_path \
    "${MODEL_DIR}/atp_hfvr_1.pt" \
    --random_seed \
    1 \
    --ap_model_path \
    "${MODEL_DIR}/atp_elst_1.pt" \
    --n_epochs \
    25 \
    --n_rbf \
    8 \
    --n_neuron \
    64 \
    --n_embed \
    8 \
    --n_params \
    1 \
    --data_dir \
    "${DATA_DIR_S3}" \
    --spec_type_ap \
    2 \
    --lr \
    5e-5 \
    --dimer_eval_type \
    elst_damping \
    --param_start_mean \
    1.6 \
    --param_start_std \
    0.25 \
    --ds_in_memory \
    True \
    --world_size_ddp \
    1 \
    --omp_num_threads \
    16 \
    --wandb-mode \
    online \
    --wandb-project \
    "${WANDB_PROJECT}" \
    --wandb-group \
    "${WANDB_GROUP}" \
    --wandb-name \
    s3-atp-elst \
    --wandb-job-type \
    dimer-param \
    --wandb-tags \
    edgeless-fix ap3d3 saptpbe0 retrain \
    --wandb-dir \
    "${WANDB_DIR}"
touch "${MODEL_DIR}/.stage3.done"
fi

# ---------------------------------------------------------------------------
# Stage 4: APNet3D3 on Splinter SAPT0/aug-cc-pVDZ (spec 2), with -D3 + NN
# dispersion
# ---------------------------------------------------------------------------
if ! skip_stage 4 "${MODEL_DIR}/ap3d3_1.pt"; then
require_input "${MODEL_DIR}/am_ap2_1.pt"
require_input "${MODEL_DIR}/atp_hfvr_1.pt"
require_input "${MODEL_DIR}/atp_elst_1.pt"
python3 \
    -u \
    ./train_models.py \
    --train_apnet \
    APNet3-fused-d3 \
    --am_model_path \
    "${MODEL_DIR}/am_ap2_1.pt" \
    --atom_type_param_model_path \
    "${MODEL_DIR}/atp_hfvr_1.pt" \
    --atom_type_param_model_path2 \
    "${MODEL_DIR}/atp_elst_1.pt" \
    --random_seed \
    1 \
    --ap_model_path \
    "${MODEL_DIR}/ap3d3_1.pt" \
    --n_epochs \
    50 \
    --n_rbf \
    8 \
    --n_neuron \
    128 \
    --n_embed \
    8 \
    --data_dir \
    "${DATA_DIR_S4}" \
    --spec_type_ap \
    2 \
    --ds_class_type \
    lmdb \
    --lr \
    5e-4 \
    --wandb-mode \
    online \
    --wandb-project \
    "${WANDB_PROJECT}" \
    --wandb-group \
    "${WANDB_GROUP}" \
    --wandb-name \
    s4-ap3d3-sapt0 \
    --wandb-job-type \
    ap3d3-sapt0 \
    --wandb-tags \
    edgeless-fix ap3d3 saptpbe0 retrain \
    --wandb-dir \
    "${WANDB_DIR}"
touch "${MODEL_DIR}/.stage4.done"
fi

# ---------------------------------------------------------------------------
# Stage 5: copy the SAPT0/aug-cc-pVDZ model, then fine-tune on 124k
# SAPT(PBE0)-D4(I)/aug-cc-pVDZ (spec 10)
# ---------------------------------------------------------------------------

if ! skip_stage 5 "${MODEL_DIR}/ap3d3_1_saptpbe0.pt"; then
require_input "${MODEL_DIR}/am_ap2_1.pt"
require_input "${MODEL_DIR}/atp_hfvr_1.pt"
require_input "${MODEL_DIR}/atp_elst_1.pt"
require_input "${MODEL_DIR}/ap3d3_1.pt"
cp "${MODEL_DIR}/ap3d3_1.pt" "${MODEL_DIR}/ap3d3_1_saptpbe0.pt"
python3 \
    -u \
    ./train_models.py \
    --train_apnet \
    APNet3-fused-d3 \
    --am_model_path \
    "${MODEL_DIR}/am_ap2_1.pt" \
    --atom_type_param_model_path \
    "${MODEL_DIR}/atp_hfvr_1.pt" \
    --atom_type_param_model_path2 \
    "${MODEL_DIR}/atp_elst_1.pt" \
    --random_seed \
    1 \
    --ap_model_path \
    "${MODEL_DIR}/ap3d3_1_saptpbe0.pt" \
    --n_epochs \
    50 \
    --n_rbf \
    8 \
    --n_neuron \
    128 \
    --n_embed \
    8 \
    --data_dir \
    "${DATA_DIR_S5}" \
    --spec_type_ap \
    10 \
    --lr \
    5e-4 \
    --ds_class_type \
    lmdb \
    --unfreeze_dimer_prop_model \
    --unfreeze_atom_model \
    --wandb-mode \
    online \
    --wandb-project \
    "${WANDB_PROJECT}" \
    --wandb-group \
    "${WANDB_GROUP}" \
    --wandb-name \
    s5-ap3d3-saptpbe0 \
    --wandb-job-type \
    ap3d3-saptpbe0-finetune \
    --wandb-tags \
    edgeless-fix ap3d3 saptpbe0 retrain \
    --wandb-dir \
    "${WANDB_DIR}"
touch "${MODEL_DIR}/.stage5.done"
fi
