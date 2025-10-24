#!/bin/bash

#SBATCH --job-name=ap3_torchrun_ddp
#SBATCH --output=logs/ap3_torchrun_%j.out
#SBATCH --error=logs/ap3_torchrun_%j.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=24:00:00
#SBATCH --partition=cpu

# slurm_torchrun.sh - Multi-node CPU DDP training with SLURM + torchrun
#
# This script uses torchrun for multi-node distributed training on CPU.
# Unlike mp.spawn which only works on a single node, torchrun can coordinate
# processes across multiple nodes using SLURM's environment variables.
#
# Key differences from slurm_train_ddp.sh (mp.spawn version):
# - Can use multiple nodes (--nodes=2 or more)
# - Still uses --ntasks-per-node=1 (torchrun spawns processes internally)
# - Uses srun to launch torchrun (SLURM sets up network for inter-node comms)
# - torchrun reads SLURM env vars to set RANK, WORLD_SIZE, etc.

# Fail on any error
set -e

# Create log directory
mkdir -p logs

# Training parameters
MODEL_PATH="./models/ap3_ensemble/slurm_torchrun/ap3_0.pt"
N_EPOCHS=50
LR=5e-4
SPLIT_PERCENT=0.9

# Number of processes per node (CPU cores / OMP_NUM_THREADS)
NPROC_PER_NODE=4

# OpenMP threads per process
export OMP_NUM_THREADS=6

# Get SLURM network info for torchrun
# SLURM sets these automatically in multi-node jobs
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

# Create model directory
mkdir -p "$(dirname "$MODEL_PATH")"

echo "========================================"
echo "APNet3-fused Multi-Node DDP Training"
echo "========================================"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "Tasks per node: $SLURM_NTASKS_PER_NODE"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Processes per node: $NPROC_PER_NODE"
echo "Total world size: $((SLURM_JOB_NUM_NODES * NPROC_PER_NODE))"
echo "OMP threads/process: $OMP_NUM_THREADS"
echo "Master addr: $MASTER_ADDR"
echo "Master port: $MASTER_PORT"
echo "Model path: $MODEL_PATH"
echo "========================================"

# Launch with srun + torchrun
# srun creates one task per node (--ntasks-per-node=1)
# torchrun on each node spawns NPROC_PER_NODE processes
# torchrun --rdzv-backend=c10d uses PyTorch's rendezvous for coordination
srun torchrun \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=$NPROC_PER_NODE \
    --rdzv-backend=c10d \
    --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT \
    train_models.py \
    --train_apnet APNet3-fused \
    --ap_model_path "$MODEL_PATH" \
    --n_epochs $N_EPOCHS \
    --lr $LR

echo "========================================"
echo "Training complete!"
echo "========================================"
