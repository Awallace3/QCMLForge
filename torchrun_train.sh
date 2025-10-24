#!/bin/bash

# torchrun_train.sh - Launch APNet3-fused training with torchrun
# This script demonstrates using torchrun for multi-process DDP training
# on a single node (multi-CPU or multi-GPU)

# torchrun automatically sets these environment variables:
#   RANK          - Global rank (0 to world_size-1)
#   LOCAL_RANK    - Local rank on this node (0 to nproc_per_node-1)
#   WORLD_SIZE    - Total number of processes across all nodes
#   MASTER_ADDR   - Address of rank 0 process
#   MASTER_PORT   - Port for communication

# Number of processes to spawn on this node
NPROC_PER_NODE=4

# Master address/port (only needed for multi-node, but set for completeness)
export MASTER_ADDR=localhost
export MASTER_PORT=29500

# OpenMP threads per process (adjust based on your CPU cores)
# Formula: OMP_NUM_THREADS = total_cores / nproc_per_node
export OMP_NUM_THREADS=6

# Training parameters
MODEL_PATH="./models/ap3_ensemble/torchrun_test/ap3_0.pt"
N_EPOCHS=3
LR=5e-4
SPLIT_PERCENT=0.9

# Create model directory
mkdir -p "$(dirname "$MODEL_PATH")"

echo "========================================"
echo "APNet3-fused Training with torchrun"
echo "========================================"
echo "Processes per node: $NPROC_PER_NODE"
echo "OMP threads/process: $OMP_NUM_THREADS"
echo "Model path: $MODEL_PATH"
echo "Epochs: $N_EPOCHS"
echo "========================================"

# Launch training with torchrun
# NOTE: Do NOT set world_size_ddp argument to train_models.py - torchrun handles this
torchrun \
    --nnodes=1 \
    --nproc_per_node=$NPROC_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train_models.py \
    --train_apnet APNet3-fused \
    --ap_model_path "$MODEL_PATH" \
    --n_epochs $N_EPOCHS \
    --lr $LR

echo "========================================"
echo "Training complete!"
echo "========================================"
