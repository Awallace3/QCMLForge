#!/bin/bash

set -euo pipefail

# cd /home/amwalla3/gits/qcmlforge.ap3d3_am
# export PYTHONPATH="/home/amwalla3/gits/qcmlforge.ap3d3_am/src${PYTHONPATH:+:$PYTHONPATH}"

ITER=1
MODEL_DIR=./models/ap3_saptpbe0/1
mkdir -p "${MODEL_DIR}"

# AP2 AtomMPNN on PBE0 monomers (spec 4 -> monomers_ap3_spec_1_pbe0.pkl)
# python3 \
#     -u \
#     ./train_models.py \
#     --train_am \
#     AtomModel \
#     --am_model_path \
#     ./models/ap3_saptpbe0/1/am_ap2_1.pt \
#     --random_seed \
#     1 \
#     --n_epochs_atom \
#     500 \
#     --lr \
#     5e-4 \
#     --n_message_atom \
#     3 \
#     --n_rbf_atom \
#     8 \
#     --n_neuron_atom \
#     128 \
#     --n_embed_atom \
#     8 \
#     --data_dir \
#     ../qcmlforge/data_dir \
#     --spec_type_am \
#     4 \
#     --world_size_ddp \
#     4 \
#     --omp_num_threads \
#     4

# Hirshfeld volume-ratio/valence-width AtomTypeParamNN on PBE0 monomers (spec 1) using AP2 h_list
# python3 \
#     -u \
#     ./train_models.py \
#     --train_apnet \
#     AtomTypeParamModel \
#     --am_model_path \
#     ./models/ap3_saptpbe0/1/am_ap2_1.pt \
#     --random_seed \
#     1 \
#     --lr \
#     5e-5 \
#     --ap_model_path \
#     ./models/ap3_saptpbe0/1/atp_hfvr_1.pt \
#     --n_epochs \
#     100 \
#     --n_rbf \
#     8 \
#     --n_neuron \
#     32 \
#     --n_embed \
#     8 \
#     --data_dir \
#     ../qcmlforge/data_dir \
#     --spec_type_ap \
#     1 \
#     --world_size_ddp \
#     1 \
#     --omp_num_threads \
#     16

# Electrostatic K AtomTypeParamNN on Splinter SAPT0/aug-cc-pVDZ dimers (spec 2)
python3 \
    -u \
    ./train_models.py \
    --train_apnet \
    AM-DimerParam \
    --am_model_path \
    ./models/ap3_saptpbe0/1/am_ap2_1.pt \
    --atom_type_param_model_path \
    ./models/ap3_saptpbe0/1/atp_hfvr_1.pt \
    --random_seed \
    1 \
    --ap_model_path \
    ./models/ap3_saptpbe0/1/atp_elst_1.pt \
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
    ../qcmlforge/data_dir \
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


# APNet3D3 on Splinter SAPT0/aug-cc-pVDZ (spec 2), with -D3 + NN dispersion
python3 \
    -u \
    ./train_models.py \
    --train_apnet \
    APNet3-fused-d3 \
    --am_model_path \
    ./models/ap3_saptpbe0/1/am_ap2_1.pt \
    --atom_type_param_model_path \
    ./models/ap3_saptpbe0/1/atp_hfvr_1.pt \
    --atom_type_param_model_path2 \
    ./models/ap3_saptpbe0/1/atp_elst_1.pt \
    --random_seed \
    1 \
    --ap_model_path \
    ./models/ap3_saptpbe0/1/ap3d3_1.pt \
    --n_epochs \
    50 \
    --n_rbf \
    8 \
    --n_neuron \
    128 \
    --n_embed \
    8 \
    --data_dir \
    ../qcmlforge/data_dir \
    --spec_type_ap \
    2 \
    --lr \
    5e-4

# Copy SAPT0/aug-cc-pVDZ model, then fine-tune on 124k SAPT(PBE0)-D4(I)/aug-cc-pVDZ (spec 10)
cp ./models/ap3_saptpbe0/1/ap3d3_1.pt ./models/ap3_saptpbe0/1/ap3d3_1_saptpbe0.pt

python3 \
    -u \
    ./train_models.py \
    --train_apnet \
    APNet3-fused-d3 \
    --am_model_path \
    ./models/ap3_saptpbe0/1/am_ap2_1.pt \
    --atom_type_param_model_path \
    ./models/ap3_saptpbe0/1/atp_hfvr_1.pt \
    --atom_type_param_model_path2 \
    ./models/ap3_saptpbe0/1/atp_elst_1.pt \
    --random_seed \
    1 \
    --ap_model_path \
    ./models/ap3_saptpbe0/1/ap3d3_1_saptpbe0.pt \
    --n_epochs \
    50 \
    --n_rbf \
    8 \
    --n_neuron \
    128 \
    --n_embed \
    8 \
    --data_dir \
    ../qcmlforge/data_dir \
    --spec_type_ap \
    10 \
    --lr \
    5e-4 \
    --ds_class_type \
    lmdb \
    --unfreeze_dimer_prop_model \
    --unfreeze_atom_model

