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
MODEL_DIR=./models/ap3_saptpbe0_edgeless_fix/${ITER}
DATA_DIR=${DATA_DIR:-../qcmlforge/data_dir}

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
if [ ! -d "${DATA_DIR}" ]; then
    echo "DATA_DIR ${DATA_DIR} does not exist (cwd $(pwd))." >&2
    echo "Set DATA_DIR=/path/to/data_dir and re-run." >&2
    exit 1
fi

if compgen -G "${MODEL_DIR}/*.pt" > /dev/null; then
    echo "Refusing to overwrite checkpoints already in ${MODEL_DIR}" >&2
    echo "Bump ITER or MODEL_DIR to start a new retrain." >&2
    exit 1
fi

mkdir -p "${MODEL_DIR}"

# ---------------------------------------------------------------------------
# Stage 1: AP2 AtomMPNN on PBE0 monomers (spec 4 -> monomers_ap3_spec_1_pbe0.pkl)
#
# This is the stage the fix is in.  Re-enabled.
# ---------------------------------------------------------------------------
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
    "${DATA_DIR}" \
    --spec_type_am \
    4 \
    --world_size_ddp \
    4 \
    --omp_num_threads \
    4

# ---------------------------------------------------------------------------
# Stage 2: Hirshfeld volume-ratio/valence-width AtomTypeParamNN on PBE0
# monomers (spec 1) using AP2 h_list.
#
# Consumes the stage-1 atom model, so it must be retrained too.  Re-enabled.
# ---------------------------------------------------------------------------
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
    "${DATA_DIR}" \
    --spec_type_ap \
    1 \
    --world_size_ddp \
    1 \
    --omp_num_threads \
    16

# ---------------------------------------------------------------------------
# Stage 3: Electrostatic K AtomTypeParamNN on Splinter SAPT0/aug-cc-pVDZ
# dimers (spec 2)
# ---------------------------------------------------------------------------
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
    "${DATA_DIR}" \
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
    16

# ---------------------------------------------------------------------------
# Stage 4: APNet3D3 on Splinter SAPT0/aug-cc-pVDZ (spec 2), with -D3 + NN
# dispersion
# ---------------------------------------------------------------------------
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
    "${DATA_DIR}" \
    --spec_type_ap \
    2 \
    --lr \
    5e-4

# ---------------------------------------------------------------------------
# Stage 5: copy the SAPT0/aug-cc-pVDZ model, then fine-tune on 124k
# SAPT(PBE0)-D4(I)/aug-cc-pVDZ (spec 10)
# ---------------------------------------------------------------------------
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
    "${DATA_DIR}" \
    --spec_type_ap \
    10 \
    --lr \
    5e-4 \
    --ds_class_type \
    lmdb \
    --unfreeze_dimer_prop_model \
    --unfreeze_atom_model
