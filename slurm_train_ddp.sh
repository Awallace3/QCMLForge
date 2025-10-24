#!/bin/bash
#SBATCH --job-name=apnet3_ddp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/apnet3_ddp_%j.out
#SBATCH --error=logs/apnet3_ddp_%j.err

# APNet3-fused DDP Training on SLURM
# 
# Key SLURM Settings for Current Implementation (mp.spawn):
# - --nodes=1              : Single node only (mp.spawn doesn't support multi-node)
# - --ntasks-per-node=1    : Only 1 task (mp.spawn creates processes internally)
# - --gres=gpu:4           : Request 4 GPUs (or 2, 8, etc.)
# - --cpus-per-task=32     : CPUs for dataloader workers
#
# DO NOT use srun to launch Python - just run python directly!

set -e

echo "=========================================="
echo "SLURM Job Information"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo ""

# Configuration
DATA_DIR="./data_dimer_0"
AM_MODEL_PATH="./models/ap3_ensemble/0/am_3.pt"
ATOM_TYPE_PARAM_MODEL_PATH="./models/ap3_ensemble/0/am_h+1_3.pt"
ATOM_TYPE_PARAM_MODEL_PATH2="./models/ap3_ensemble/0/am_elst_h+1_3.pt"
AP_MODEL_PATH="./models/ap3_ensemble/0/ap3_slurm_${SLURM_JOB_ID}.pt"
N_EPOCHS=50
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
DS_CLASS_TYPE="lmdb"

echo "Configuration:"
echo "  Data directory: $DATA_DIR"
echo "  Output model: $AP_MODEL_PATH"
echo "  Epochs: $N_EPOCHS"
echo ""

# Detect GPUs (auto-detected by train_models.py from torch.cuda.device_count())
# The code at train_models.py:134 will automatically detect available GPUs
# No need to set WORLD_SIZE manually

echo "=========================================="
echo "Starting training..."
echo "=========================================="
echo ""

# Run training directly (NO srun!)
# mp.spawn() will handle process creation internally
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
    --ds_class_type "$DS_CLASS_TYPE"

echo ""
echo "=========================================="
echo "Training completed!"
echo "=========================================="
echo "Model saved to: $AP_MODEL_PATH"
