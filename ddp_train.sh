#!/bin/bash

# DDP Training Script for APNet3-fused Model
# This script demonstrates how to train the APNet3-fused model using
# Distributed Data Parallel (DDP) for multi-GPU or multi-CPU training.

# Exit on error
set -e

# Configuration
DATA_DIR="./data_dimer_0"
AM_MODEL_PATH="./models/ap3_ensemble/0/am_3.pt"
ATOM_TYPE_PARAM_MODEL_PATH="./models/ap3_ensemble/0/am_h+1_3.pt"
ATOM_TYPE_PARAM_MODEL_PATH2="./models/ap3_ensemble/0/am_elst_h+1_3.pt"
AP_MODEL_PATH="./models/ap3_ensemble/0/ap3_testing.pt"
N_EPOCHS=3
LR=5e-4
RANDOM_SEED=42

# Hyperparameters
R_CUT=5.0
R_CUT_IM=8.0
N_RBF=8
N_NEURON=128
N_EMBED=8

# Dataset parameters
SPEC_TYPE_AP=8
DS_CLASS_TYPE="lmdb"  # Options: "pt" or "lmdb"

echo "========================================"
echo "APNet3-fused DDP Training Script"
echo "========================================"
echo ""
echo "Configuration:"
echo "  Data directory: $DATA_DIR"
echo "  Atom model: $AM_MODEL_PATH"
echo "  Atom type param model 1: $ATOM_TYPE_PARAM_MODEL_PATH"
echo "  Atom type param model 2: $ATOM_TYPE_PARAM_MODEL_PATH2"
echo "  Output model: $AP_MODEL_PATH"
echo "  Epochs: $N_EPOCHS"
echo "  Learning rate: $LR"
echo "  Random seed: $RANDOM_SEED"
echo ""

# Detect number of GPUs
if command -v nvidia-smi &> /dev/null; then
    N_GPUS=$(nvidia-smi --list-gpus | wc -l)
    echo "Detected $N_GPUS GPU(s)"
    if [ $N_GPUS -gt 0 ]; then
        echo "Using multi-GPU DDP training with $N_GPUS GPUs"
        WORLD_SIZE=$N_GPUS
    else
        echo "No GPUs detected, using single-process CPU training"
        WORLD_SIZE=1
    fi
else
    echo "nvidia-smi not found, using single-process CPU training"
    WORLD_SIZE=1
fi

echo ""
echo "========================================"
echo "Starting training..."
echo "========================================"
echo ""

# Run training
python train_models.py \
    --train_apnet "APNet3-fused" \
    --am_model_path "$AM_MODEL_PATH" \
    --atom_type_param_model_path "$ATOM_TYPE_PARAM_MODEL_PATH" \
    --atom_type_param_model_path2 "$ATOM_TYPE_PARAM_MODEL_PATH2" \
    --ap_model_path "$AP_MODEL_PATH" \
    --data_dir "$DATA_DIR" \
    --n_epochs $N_EPOCHS \
    --lr $LR \
    --random_seed $RANDOM_SEED \
    --spec_type_ap $SPEC_TYPE_AP \
    --r_cut $R_CUT \
    --r_cut_im $R_CUT_IM \
    --n_rbf $N_RBF \
    --n_neuron $N_NEURON \
    --n_embed $N_EMBED \
    --ds_class_type "$DS_CLASS_TYPE" \
    --world_size_ddp  $WORLD_SIZE \

echo ""
echo "========================================"
echo "Training completed!"
echo "========================================"
echo "Model saved to: $AP_MODEL_PATH"
