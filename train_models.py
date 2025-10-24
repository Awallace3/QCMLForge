from apnet_pt import AtomModels
from apnet_pt import AtomPairwiseModels
import torch
import argparse
import numpy as np
import random
import os
from pprint import pprint


def train_atom_model(
    atom_model_type="AtomModel",
    model_path="./models/am_amw_1.pt",
    data_dir="data_atomic",
    spec_type=3,
    testing=False,
    n_epochs=500,
    random_seed=42,
    ds_max_size=None,
    world_size=1,
    omp_num_threads=1,
):
    if atom_model_type == "AtomModel":
        AM = AtomModels.ap2_atom_model.AtomModel
        batch_size = 16
    elif atom_model_type == "AtomHirshfeldModel":
        AM = AtomModels.ap3_atom_model.AtomHirshfeldModel
        batch_size = 1
    else:
        raise ValueError("Invalid Atom Model Type")
    pretrained_model = None
    if os.path.exists(model_path):
        pretrained_model = model_path
    print("Training {}...".format(atom_model_type))
    atom_model = AM(
        n_message=3,
        n_rbf=8,
        n_neuron=128,
        n_embed=8,
        r_cut=5.0,
        ds_root=data_dir,
        ds_spec_type=spec_type,
        ds_max_size=ds_max_size,
        ignore_database_null=False,
        ds_in_memory=True,
        use_GPU=True,
        pre_trained_model_path=pretrained_model,
    )
    dataloader_num_workers = 0
    if torch.cuda.is_available() and omp_num_threads > 2:
        dataloader_num_workers = omp_num_threads - 2
    print(atom_model.dataset)
    atom_model.train(
        n_epochs=n_epochs,
        batch_size=batch_size,
        lr=5e-4,
        split_percent=0.9,
        model_path=model_path,
        shuffle=True,
        dataloader_num_workers=dataloader_num_workers,
        world_size=world_size,
        omp_num_threads_per_process=omp_num_threads,
        random_seed=random_seed,
    )
    return


def train_pairwise_model(
    apnet_model_type="APNet2",
    model_out="./models/ap2_ensemble/ap2_1.pt",
    am_model_path="./models/ap2_ensemble/am_1.pt",
    atom_type_param_model_path="./models/ap_atomTypeParamModel/am_0.pt",
    atom_type_param_model_path2="./models/ap_atomTypeParamModel/am_0.pt",
    data_dir="./data_pairwise",
    n_epochs=50,
    lr=5e-4,
    lr_decay=None,
    random_seed=42,
    spec_type=2,
    r_cut_im=8.0,
    r_cut=5.0,
    n_rbf=8,
    n_neuron=128,
    n_embed=8,
    n_params=2,
    m1="",
    m2="",
    pre_trained_model_path="./models/dapnet2/ap2_0.pt",
    param_start_mean=1.5,
    param_start_std=0.1,
    dimer_eval_type="elst_damping",
    ds_in_memory=False,
    ds_class_type="pt",
    DimerProp_model_type="AtomTypeParamNN",
    ap2_pretrained_model_only=None,
    world_size=1,
):
    # Ensure param_start_mean and param_start_std are lists
    if not isinstance(param_start_mean, (list, tuple)):
        param_start_mean = [param_start_mean] * n_params
    if not isinstance(param_start_std, (list, tuple)):
        param_start_std = [param_start_std] * n_params
    ds_atomic_batch_size = 4  * 256
    ds_datapoint_storage_n_objects = 16
    if apnet_model_type == "APNet2":
        APNet = AtomPairwiseModels.apnet2.APNet2Model
    elif apnet_model_type == "APNet2-fused":
        APNet = AtomPairwiseModels.apnet2_fused.APNet2_AM_Model
    elif apnet_model_type == "APNet3-fused":
        APNet = AtomPairwiseModels.apnet3_fused.APNet3_AtomType_Model
        # Note: presently ap3_fused_ds requires atomic batch size to be <=
        # n_objects. NEDS FIXED
        ds_atomic_batch_size = 16
        ds_datapoint_storage_n_objects = 16
    elif apnet_model_type == "AM-DimerParam":
        APNet = AtomPairwiseModels.mtp_mtp.AM_DimerParam_Model
    elif apnet_model_type == "dAPNet2":
        APNet = AtomPairwiseModels.dapnet2.dAPNet2Model
        apnet2_model = AtomPairwiseModels.apnet2.APNet2Model(
            n_rbf=n_rbf,
            n_neuron=n_neuron,
            n_embed=n_embed,
            r_cut=r_cut,
            r_cut_im=r_cut_im,
            atom_model_pre_trained_path=am_model_path,
            pre_trained_model_path=pre_trained_model_path,
        )
        apnet2_model.model.return_hidden_states = True
    elif apnet_model_type == "AtomTypeParamModel":
        APNet = AtomPairwiseModels.mtp_mtp.AtomTypeParamModel
    else:
        raise ValueError("Invalid Atom Model Type")
    print("Training {}...".format(apnet_model_type))
    # if torch.cuda.is_available():
    #     world_size = torch.cuda.device_count()
    # else:
    #     world_size = 1
    if os.environ.get("RANK") is not None:
        print("Detected torchrun environment - DDP handled by torchrun")
        world_size = 1
    print("World Size", world_size)

    omp_num_threads_per_process = 8
    if os.path.exists(model_out):
        pretrained_model = model_out
        print(f"\nTraining from {model_out}\n")
    else:
        pretrained_model = None
        print("\nTraining from scratch...\n")
    if apnet_model_type.startswith("dAPNet"):
        apnet = APNet(
            apnet2_model=apnet2_model,
            atom_model_pre_trained_path=am_model_path,
            pre_trained_model_path=pretrained_model,
            n_rbf=n_rbf,
            n_neuron=n_neuron,
            n_embed=n_embed,
            r_cut=r_cut,
            r_cut_im=r_cut_im,
            ds_spec_type=spec_type,
            ds_root=data_dir,
            ignore_database_null=False,
            ds_atomic_batch_size=ds_atomic_batch_size,
            ds_num_devices=1,
            ds_skip_process=False,
            ds_datapoint_storage_n_objects=ds_datapoint_storage_n_objects,
            ds_prebatched=True,
            ds_m1=m1,
            ds_m2=m2,
        )
    elif apnet_model_type in ["AM-DimerParam"]:
        if dimer_eval_type in ["elst_damping__induced_dipole", "elst_damping"] and atom_type_param_model_path is not None:
            print("Using AtomTypeParamModel for Dimer Prop Model")
            atom_model = AtomPairwiseModels.mtp_mtp.AtomTypeParamModel(
                ds_root=None,
                use_GPU=False,
                ignore_database_null=True,
                atom_model_pre_trained_path=am_model_path,
                pre_trained_model_path=atom_type_param_model_path,
            ).model
            am_model_path = None
            atom_model_type = "AtomTypeParamNN"
        else:
            atom_model = None
            atom_model_type = "AtomModel"

        apnet = APNet(
            atom_model=atom_model,
            atom_model_pre_trained_path=am_model_path,
            atom_model_type=atom_model_type,
            pre_trained_model_path=pretrained_model,
            n_rbf=n_rbf,
            n_neuron=n_neuron,
            n_embed=n_embed,
            r_cut=r_cut,
            ds_spec_type=spec_type,
            ds_root=data_dir,
            ignore_database_null=False,
            ds_atomic_batch_size=ds_atomic_batch_size,
            ds_num_devices=1,
            ds_skip_process=False,
            ds_datapoint_storage_n_objects=ds_datapoint_storage_n_objects,
            ds_prebatched=False,
            ds_random_seed=random_seed,
            param_start_mean=param_start_mean,
            param_start_std=param_start_std,
            dimer_eval_type=dimer_eval_type,
            n_params=n_params,
            model_type=DimerProp_model_type
        )
    elif apnet_model_type in ["APNet3-fused"]:
        print("Setting AtomTypeParams...")
        atom_type_hf_vw_model = AtomPairwiseModels.mtp_mtp.AtomTypeParamModel(
            ds_root=None,
            use_GPU=False,
            ignore_database_null=True,
            atom_model_pre_trained_path=am_model_path,
            pre_trained_model_path=atom_type_param_model_path,
        )
        atom_type_elst_model = AtomPairwiseModels.mtp_mtp.AM_DimerParam_Model(
            ds_root=None,
            use_GPU=False,
            ignore_database_null=True,
            atom_model=atom_type_hf_vw_model.model,
            atom_model_type="AtomTypeParamNN",
            pre_trained_model_path=atom_type_param_model_path2,
        )
        am_model_path = None
        print(f"{ds_atomic_batch_size=}, {ds_datapoint_storage_n_objects=}")
        apnet = APNet(
            atom_type_model=atom_type_hf_vw_model.model,
            dimer_prop_model=atom_type_elst_model.dimer_model,
            pre_trained_model_path=pretrained_model,
            n_rbf=n_rbf,
            n_neuron=n_neuron,
            n_embed=n_embed,
            r_cut=r_cut,
            ds_spec_type=spec_type,
            ds_root=data_dir,
            ignore_database_null=False,
            ds_atomic_batch_size=ds_atomic_batch_size,
            ds_num_devices=1,
            ds_skip_process=False,
            ds_datapoint_storage_n_objects=ds_datapoint_storage_n_objects,
            ds_prebatched=False,
            ds_random_seed=random_seed,
            ds_class_type=ds_class_type,
            use_precomputed_classical=True,
        )
        if ap2_pretrained_model_only is not None:
            print(f"Loading AP2 pretrained weights from {ap2_pretrained_model_only}")
            apnet.load_ap2_pretrained_weights(ap2_pretrained_model_only)
    elif apnet_model_type in ["AtomTypeParamModel"]:
        apnet = APNet(
            atom_model_pre_trained_path=am_model_path,
            pre_trained_model_path=pretrained_model,
            n_rbf=n_rbf,
            n_neuron=n_neuron,
            n_embed=n_embed,
            r_cut=r_cut,
            ds_spec_type=spec_type,
            ds_root=data_dir,
            ignore_database_null=False,
            ds_in_memory=ds_in_memory,
            use_GPU=True,
            param_start_mean=param_start_mean,
            param_start_std=param_start_std,
        )
    else:
        apnet = APNet(
            atom_model_pre_trained_path=am_model_path,
            pre_trained_model_path=pretrained_model,
            n_rbf=n_rbf,
            n_neuron=n_neuron,
            n_embed=n_embed,
            r_cut=r_cut,
            r_cut_im=r_cut_im,
            ds_spec_type=spec_type,
            ds_root=data_dir,
            ignore_database_null=False,
            ds_atomic_batch_size=ds_atomic_batch_size,
            ds_num_devices=1,
            ds_skip_process=False,
            ds_datapoint_storage_n_objects=ds_datapoint_storage_n_objects,
            ds_prebatched=True,
            ds_random_seed=random_seed,
        )
    apnet.train(
        model_path=model_out,
        n_epochs=n_epochs,
        world_size=world_size,
        omp_num_threads_per_process=omp_num_threads_per_process,
        lr=lr,
        # lr_decay=lr_decay,
        dataloader_num_workers=4,
        random_seed=random_seed,
    )
    return


def set_all_seeds(seed=42, cudnn_reproducibility=False):
    """
    Set all relevant random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU
        # For CuDNN, setting these flags ensures reproducible but potentially
        # slower performance.
        if cudnn_reproducibility:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    return


def parse_param_list(param_str):
    """Parse comma-separated string to list of floats, or single float if no comma."""
    if "," in param_str:
        return [float(x.strip()) for x in param_str.split(",")]
    else:
        return float(param_str)


def main():
    args = argparse.ArgumentParser()
    args.add_argument(
        "--am_model_path",
        type=str,
        default="./models/am_ensemble/am_0.pt",
        help="specify where to save output model (default: ./models/am_ensemble/am_1.pt)",
    )
    args.add_argument(
        "--atom_type_param_model_path",
        type=str,
        default=None,
        help="specify AtomTypeParamModel to use for AtomTypeParam Dimer props (default: None)",
    )
    args.add_argument(
        "--atom_type_param_model_path2",
        type=str,
        default=None,
        help="specify AtomTypeParamModel to use for AtomTypeParam Dimer props in AP3 (default: None)",
    )
    args.add_argument(
        "--ap_model_path",
        type=str,
        default="./models/ap2_ensemble/ap2_0.pt",
        help="specify where to save output model (default: ./models/ap2_ensemble/ap2_1.pt)",
    )
    args.add_argument(
        "--ap_pretrained_model_path",
        type=str,
        default="./models/dapnet2/ap2_0.pt",
        help="specify a special loaded model. Currently only used for dAP-Net2 training (default: ./models/dapnet2/ap2_0.pt)",
    )
    args.add_argument(
        "--ap2_pretrained_model_only",
        type=str,
        default=None,
        help="Load AP2 pretrained weights for AP3 model initialization (path to AP2 model)",
    )
    args.add_argument(
        "--train_am",
        type=str,
        default="",
        help="Train AtomModel: (AtomModel, AtomHirshfeldModel)",
    )
    args.add_argument(
        "--train_apnet",
        type=str,
        default="",
        help="Train APNet Model: (APNet2, APNet3-fused, dAPNet2, APNet2-fused, AM-DimerParam)",
    )
    args.add_argument(
        "--dimer_eval_type",
        type=str,
        default="elst_damping",
        help="Specify dimer eval type for AM-DimerParam (default: 'elst_damping', other options: 'induced_dipole)",
    )
    args.add_argument(
        "--random_seed", type=int, default=0, help="Random seed for initialization"
    )
    args.add_argument(
        "--spec_type_am",
        type=int,
        default=3,
        help="dataset spec_type recommended: (3 for AM)",
    )
    args.add_argument(
        "--spec_type_ap",
        type=int,
        default=2,
        help="dataset spec_type recommended: (2 for AP2)",
    )
    args.add_argument(
        "--data_dir_atom",
        type=str,
        default="./data_dir",
        help="specify data_dir for datasets (default: ./data_dir)",
    )
    args.add_argument(
        "--data_dir",
        type=str,
        default="./data_dir",
        help="specify data_dir for datasets (default: ./data_dir)",
    )
    args.add_argument(
        "--n_epochs_atom", type=int, default=500, help="Number of epochs for training"
    )
    args.add_argument(
        "--n_epochs", type=int, default=50, help="Number of epochs for training"
    )
    args.add_argument(
        "--ds_max_size",
        type=int,
        default=None,
        help="Limit dataset to N dataset objects",
    )
    args.add_argument(
        "--lr", type=float, default=5e-4, help="Learning Rate: (5e-4 is default)"
    )
    args.add_argument(
        "--lr_decay",
        type=float,
        default=None,
        help="Learning Rate Decay: (None is default, takes in float)",
    )
    args.add_argument(
        "--m1",
        type=str,
        default="",
        help="specify dAP-Net level of theory 1 (default: '')",
    )
    args.add_argument(
        "--m2",
        type=str,
        default="",
        help="specify dAP-Net level of theory 2 (default: '')",
    )
    args.add_argument(
        "--r_cut_im", type=float, default=8.0, help="specify AP r_cut_im (default: 8.0)"
    )
    args.add_argument(
        "--r_cut", type=float, default=5.0, help="specify AP r_cut (default: 5.0)"
    )
    # create args for n_rbf, n_neuron, n_embed
    args.add_argument(
        "--n_rbf", type=int, default=8, help="specify AP n_rbf (default: 8)"
    )
    args.add_argument(
        "--n_neuron", type=int, default=128, help="specify AP n_neuron (default: 128)"
    )
    args.add_argument(
        "--n_embed", type=int, default=8, help="specify AP n_embed (default: 8)"
    )
    args.add_argument(
        "--n_params", type=int, default=2, help="specify AP n_params (default: 2)"
    )
    args.add_argument(
        "--param_start_mean",
        type=str,
        default="2.0",
        help="specify AM-DimerParam Embedding Start Mean (default: 2.0, or comma-separated list)",
    )
    args.add_argument(
        "--param_start_std",
        type=str,
        default="0.1",
        help="specify AM-DimerParam Embedding Start std (default: 0.1, or comma-separated list)",
    )
    args.add_argument(
        "--world_size_ddp",
        type=int,
        default=1,
        help="specify world_size for DDP only for AtomModels currently (default: 1)",
    )
    args.add_argument(
        "--omp_num_threads",
        type=int,
        default=1,
        help="specify omp_num_threads for DDP only for AtomModels currently (default: 1)",
    )
    args.add_argument(
        "--ds_in_memory",
        type=bool,
        default=False,
        help="Load dataset in memory (default: False).",
    )
    args.add_argument(
        "--ds_class_type",
        type=str,
        default="pt",
        help="Dataset class type: (pt or lmdb) (default: pt)",
    )
    args.add_argument(
        "--DimerProp_model_type", type=str, default="AtomTypeParamNN", help="Dimer Prop Model Type (default: AtomTypeParamNN, other options: AtomTypeParamMPNN)"
    )
    args = args.parse_args()
    # Parse param_start_mean and param_start_std
    args.param_start_mean = parse_param_list(args.param_start_mean)
    args.param_start_std = parse_param_list(args.param_start_std)
    pprint(args)
    set_all_seeds(args.random_seed)
    if args.train_am != "":
        train_atom_model(
            atom_model_type=args.train_am,
            model_path=args.am_model_path,
            data_dir=args.data_dir_atom,
            spec_type=args.spec_type_am,
            n_epochs=args.n_epochs_atom,
            random_seed=args.random_seed,
            ds_max_size=args.ds_max_size,
            world_size=args.world_size_ddp,
            omp_num_threads=args.omp_num_threads,
        )
    if args.train_apnet != "":
        train_pairwise_model(
            apnet_model_type=args.train_apnet,
            model_out=args.ap_model_path,
            am_model_path=args.am_model_path,
            atom_type_param_model_path=args.atom_type_param_model_path,
            atom_type_param_model_path2=args.atom_type_param_model_path2,
            data_dir=args.data_dir,
            n_epochs=args.n_epochs,
            lr=args.lr,
            lr_decay=args.lr_decay,
            random_seed=args.random_seed,
            spec_type=args.spec_type_ap,
            r_cut=args.r_cut,
            r_cut_im=args.r_cut_im,
            n_rbf=args.n_rbf,
            n_neuron=args.n_neuron,
            n_embed=args.n_embed,
            n_params=args.n_params,
            m1=args.m1,
            m2=args.m2,
            pre_trained_model_path=args.ap_pretrained_model_path,
            param_start_mean=args.param_start_mean,
            param_start_std=args.param_start_std,
            dimer_eval_type=args.dimer_eval_type,
            ds_in_memory=args.ds_in_memory,
            ds_class_type=args.ds_class_type,
            DimerProp_model_type=args.DimerProp_model_type,
            ap2_pretrained_model_only=args.ap2_pretrained_model_only,
            world_size=args.world_size_ddp,
        )
    return


if __name__ == "__main__":
    main()
