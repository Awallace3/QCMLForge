#!/usr/bin/env python
"""
QCMLForge Distributed Data Parallel (DDP) Training Script for SLURM

This script is designed to be launched via srun on SLURM clusters for
multi-node, multi-process distributed training.

Environment Variables Required:
    RANK: Global rank of the process (set by SLURM via srun)
    LOCAL_RANK: Local rank on the node (set by SLURM via srun)
    WORLD_SIZE: Total number of processes (set by SLURM via srun)
    MASTER_ADDR: Address of rank 0 process
    MASTER_PORT: Port for communication

Usage:
    srun python train_ddp_slurm.py [args]
"""

import os
import sys
import argparse
import torch
import torch.distributed as dist

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from apnet_pt import AtomModels
from apnet_pt import atomic_datasets


def parse_args():
    """
    Build and parse command-line arguments for dataset, training, dataloader, and DDP configuration.
    
    The parser defines required dataset paths (`--data_root`, `--atp_model_path`), training hyperparameters (epochs, batch size, learning rate, split percent, optional `--model_save_path`), dataloader options (`--num_workers`, `--omp_num_threads`), and DDP/SLURM-related settings (`--rank`, `--local_rank`, `--world_size`, `--master_addr`, `--master_port`). `--max_size` accepts the string "None" to indicate the full dataset; `--use_lmdb` and `--precompute_hfvr` accept common true/false string values.
    
    Returns:
        argparse.Namespace: Parsed arguments with attributes matching the defined CLI options.
    """
    parser = argparse.ArgumentParser(
        description="QCMLForge DDP Training on SLURM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dataset arguments
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Root directory for dataset",
    )
    parser.add_argument(
        "--atp_model_path",
        type=str,
        required=True,
        help="Path to pre-trained AtomTypeParamModel",
    )
    parser.add_argument(
        "--spec_type",
        type=int,
        default=5,
        help="Dataset spec_type",
    )
    parser.add_argument(
        "--max_size",
        type=str,
        default="None",
        help="Maximum dataset size (use 'None' for full dataset)",
    )
    parser.add_argument(
        "--use_lmdb",
        type=str,
        default="true",
        help="Use LMDB dataset (true/false)",
    )
    parser.add_argument(
        "--precompute_hfvr",
        type=str,
        default="true",
        help="Pre-compute volume_ratios and valence_widths (true/false)",
    )

    # Training arguments
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size per process",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--split_percent",
        type=float,
        default=0.9,
        help="Train/test split percentage",
    )
    parser.add_argument(
        "--model_save_path",
        type=str,
        default=None,
        help="Path to save the trained model",
    )

    # Dataloader arguments
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="Number of dataloader workers per process",
    )
    parser.add_argument(
        "--omp_num_threads",
        type=int,
        default=None,
        help="OMP_NUM_THREADS value",
    )

    # DDP arguments (typically set by SLURM)
    parser.add_argument(
        "--rank",
        type=int,
        default=None,
        help="Global rank (usually from SLURM_PROCID)",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=None,
        help="Local rank (usually from SLURM_LOCALID)",
    )
    parser.add_argument(
        "--world_size",
        type=int,
        default=None,
        help="World size (usually from SLURM_NTASKS)",
    )
    parser.add_argument(
        "--master_addr",
        type=str,
        default=None,
        help="Master address (usually computed from SLURM_JOB_NODELIST)",
    )
    parser.add_argument(
        "--master_port",
        type=str,
        default="29500",
        help="Master port",
    )

    return parser.parse_args()


def setup_distributed(args):
    """
    Configure PyTorch distributed environment variables from provided args or SLURM environment.
    
    If `args` is missing rank, local_rank, world_size, or master_addr, those values are read from the SLURM environment variables SLURM_PROCID, SLURM_LOCALID, SLURM_NTASKS, and MASTER_ADDR (respectively). The function sets the environment variables RANK, LOCAL_RANK, WORLD_SIZE, MASTER_ADDR, and MASTER_PORT to values derived from `args`. If `args.omp_num_threads` is provided, OMP_NUM_THREADS is set. A brief summary of the resolved distributed configuration is printed when the global rank is 0.
    
    Parameters:
        args (argparse.Namespace): Namespace with optional attributes `rank`, `local_rank`, `world_size`, `master_addr`, `master_port`, and `omp_num_threads`. The namespace is mutated in-place.
    
    Returns:
        argparse.Namespace: The same `args` namespace with any missing distributed attributes filled in.
    """
    # Get DDP parameters from environment if not provided
    if args.rank is None:
        args.rank = int(os.environ.get("SLURM_PROCID", 0))
    if args.local_rank is None:
        args.local_rank = int(os.environ.get("SLURM_LOCALID", 0))
    if args.world_size is None:
        args.world_size = int(os.environ.get("SLURM_NTASKS", 1))
    if args.master_addr is None:
        args.master_addr = os.environ.get("MASTER_ADDR", "localhost")

    # Set environment variables for PyTorch DDP
    os.environ["RANK"] = str(args.rank)
    # os.environ["LOCAL_RANK"] = str(args.local_rank)
    os.environ["LOCAL_RANK"] = str(args.rank)
    os.environ["WORLD_SIZE"] = str(args.world_size)
    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = args.master_port

    # Set OMP_NUM_THREADS if provided
    if args.omp_num_threads is not None:
        os.environ["OMP_NUM_THREADS"] = str(args.omp_num_threads)

    # Print info only from rank 0
    if args.rank == 0:
        print("=" * 60)
        print("Distributed Training Setup")
        print("=" * 60)
        print(f"Rank: {args.rank}")
        print(f"Local Rank: {args.local_rank}")
        print(f"World Size: {args.world_size}")
        print(f"Master Addr: {args.master_addr}")
        print(f"Master Port: {args.master_port}")
        print(f"OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS', 'not set')}")
        print("=" * 60)
        print()

    return args


def main():
    """
    Orchestrates end-to-end distributed training: parse CLI args, configure the DDP environment, load the pre-trained AtomTypeParamModel, initialize the AtomInducedDipoleModel with dataset settings, and run training across processes.
    
    This function performs the following observable actions:
    - Reads and normalizes command-line arguments and SLURM/environment DDP settings.
    - Loads a pre-trained AtomTypeParamModel and uses it to construct an AtomInducedDipoleModel configured for the specified dataset.
    - Starts distributed training with the configured hyperparameters and dataloader options.
    - Emits configuration and progress messages from the global rank 0 process.
    
    Side effects:
    - Modifies process environment variables used for PyTorch DDP.
    - Loads model artifacts and may write the trained model to disk if a save path is provided.
    """
    args = parse_args()
    args = setup_distributed(args)

    # Parse boolean arguments
    use_lmdb = args.use_lmdb.lower() in ("true", "yes", "1", "t", "y")
    precompute_hfvr = args.precompute_hfvr.lower() in ("true", "yes", "1", "t", "y")

    # Parse max_size
    if args.max_size.lower() == "none":
        max_size = None
    else:
        max_size = int(args.max_size)

    # Print configuration from rank 0
    if args.rank == 0:
        print("Training Configuration")
        print("=" * 60)
        print(f"Data root: {args.data_root}")
        print(f"AtomTypeParam model: {args.atp_model_path}")
        print(f"Spec type: {args.spec_type}")
        print(f"Max dataset size: {max_size}")
        print(f"Use LMDB: {use_lmdb}")
        print(f"Precompute HFVR: {precompute_hfvr}")
        print(f"Epochs: {args.n_epochs}")
        print(f"Batch size (per process): {args.batch_size}")
        print(f"Learning rate: {args.lr}")
        print(f"Split percent: {args.split_percent}")
        print(f"Model save path: {args.model_save_path}")
        print(f"Dataloader workers: {args.num_workers}")
        print("=" * 60)
        print()

    # Load pre-trained AtomTypeParam model
    if args.rank == 0:
        print("Loading AtomTypeParam model...")

    atpm = AtomModels.ap3_atomtype_mpnn.AtomTypeParamModel(
        use_GPU=False,  # Adjust if using GPUs
        ignore_database_null=True,
        pre_trained_model_path=args.atp_model_path,
    )

    if args.rank == 0:
        print("AtomTypeParam model loaded successfully")
        print()

    # Create AtomInducedDipoleModel with appropriate dataset
    if args.rank == 0:
        print("Initializing AtomInducedDipoleModel...")

    am = AtomModels.ap3_atom_model.AtomInducedDipoleModel(
        atomtype_hfvr_model=atpm.model,
        use_GPU=False,  # Adjust if using GPUs
        ignore_database_null=False,
        ds_root=args.data_root,
        ds_spec_type=args.spec_type,
        ds_max_size=max_size,
        ds_use_lmdb=use_lmdb,
        ds_in_memory=False if use_lmdb else True,
        precompute_hfvr=precompute_hfvr,
    )

    if args.rank == 0:
        print("AtomInducedDipoleModel initialized")
        print()

    # Train with DDP
    if args.rank == 0:
        print("Starting distributed training...")
        print()

    am.train(
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        split_percent=args.split_percent,
        model_path=args.model_save_path,
        shuffle=True,
        skip_compile=True,  # Set to False if you want torch.compile
        dataloader_num_workers=args.num_workers,
        world_size=args.world_size,
        omp_num_threads_per_process=args.omp_num_threads,
        random_seed=42,
    )

    if args.rank == 0:
        print()
        print("=" * 60)
        print("Training Complete!")
        print("=" * 60)
        if args.model_save_path:
            print(f"Model saved to: {args.model_save_path}")


if __name__ == "__main__":
    main()