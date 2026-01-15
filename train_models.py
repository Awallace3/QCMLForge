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
    atom_type_param_model_path=None,
    atom_mpnn_pretrained_path=None,
    data_dir="data_atomic",
    spec_type=3,
    testing=False,
    n_epochs=500,
    random_seed=42,
    ds_max_size=None,
    world_size=1,
    omp_num_threads=1,
    lr=5e-4,
    n_message=3,
    n_rbf=8,
    n_neuron=128,
    n_embed=8,
    r_cut=5.0,
    use_nn_screening=False,
    precompute_hfvr=False,
    ds_use_lmdb=False,
):
    """
    Train an atom-level model specified by `atom_model_type` using the dataset and hyperparameters provided.
    
    Selects and instantiates the appropriate atom model class, optionally initializes from an existing checkpoint (if `model_path` exists), and runs training with the supplied dataset and training hyperparameters. Raises ValueError if `atom_model_type` is not recognized.
    
    Parameters:
        atom_model_type (str): Identifier of the atom model to train (e.g., "AtomModel", "AtomHirshfeldModel", "AtomTypeParamModel", "AtomInducedDipoleModel", "InducedDipoleModel").
        model_path (str): Output path for the trained model and path used to detect/load a pretrained checkpoint if the file exists.
        atom_type_param_model_path (str or None): Path to a pretrained atom-type parameter model (used by Hirshfeld/induced-dipole variants).
        atom_mpnn_pretrained_path (str or None): Path to a pretrained AtomMPNN model (used by InducedDipoleModel).
        data_dir (str): Root directory containing the atomic dataset.
        spec_type (int): Dataset specification/type index used by the dataset loader.
        testing (bool): Reserved flag for test mode (not used for return value).
        n_epochs (int): Number of training epochs.
        random_seed (int): Seed for reproducible training runs.
        ds_max_size (int or None): Maximum number of dataset entries to load; None means no explicit cap.
        world_size (int): Number of devices/processes participating in distributed training.
        omp_num_threads (int): Number of OpenMP threads available per process; used to derive dataloader worker count.
        lr (float): Initial learning rate.
        n_message (int): Number of message-passing steps (when applicable to the chosen model).
        n_rbf (int): Number of radial basis functions used by the model.
        n_neuron (int): Number of neurons per hidden layer.
        n_embed (int): Size of embedding vectors.
        r_cut (float): Cutoff radius for neighbor interactions.
        use_nn_screening (bool): Enable neural-network–based screening of interactions (applies to induced-dipole variants).
        precompute_hfvr (bool): Precompute Hirshfeld volumes/widths for models that use them.
        ds_use_lmdb (bool): Use LMDB-backed dataset storage when supported.
    
    """
    if atom_model_type == "AtomModel":
        AM = AtomModels.ap2_atom_model.AtomModel
        batch_size = 16
    elif atom_model_type == "AtomHirshfeldModel":
        AM = AtomModels.ap2_hirshfeld_atom_model.AtomHirshfeldModel
        batch_size = 1
    elif atom_model_type == "AtomTypeParamModel":
        AM = AtomModels.ap3_atomtype_mpnn.AtomTypeParamModel
        batch_size = 16
    elif atom_model_type == "AtomInducedDipoleModel":
        AM = AtomModels.ap3_atom_model.AtomInducedDipoleModel
        batch_size = 16
    elif atom_model_type == "InducedDipoleModel":
        AM = AtomModels.ap3_atom_model_frozen.InducedDipoleModel
        batch_size = 16
    else:
        raise ValueError("Invalid Atom Model Type")
    pretrained_model = None
    if os.path.exists(model_path):
        pretrained_model = model_path
    print("Training {}...".format(atom_model_type))
    # TODO complete
    if atom_model_type in ["AtomModel", "AtomHirshfeldModel", "AtomTypeParamModel"]:
        atom_model = AM(
            n_message=n_message,
            n_rbf=n_rbf,
            n_neuron=n_neuron,
            n_embed=n_embed,
            r_cut=r_cut,
            ds_root=data_dir,
            ds_spec_type=spec_type,
            ds_max_size=ds_max_size,
            ignore_database_null=False,
            ds_in_memory=True,
            use_GPU=True,
            pre_trained_model_path=pretrained_model,
        )
        skip_compile = False
    elif atom_model_type in ["AtomInducedDipoleModel"]:
        atom_model = AM(
            atomtype_hfvr_pre_trained_path=atom_type_param_model_path,
            n_rbf=n_rbf,
            n_neuron=n_neuron,
            n_embed=n_embed,
            r_cut=r_cut,
            use_nn_screening=use_nn_screening,
            precompute_hfvr=precompute_hfvr,
            ds_root=data_dir,
            ds_spec_type=spec_type,
            ds_max_size=ds_max_size,
            ignore_database_null=False,
            ds_in_memory=True,
            use_GPU=True,
            pre_trained_model_path=pretrained_model,
        )
        skip_compile = False
    elif atom_model_type in ["InducedDipoleModel"]:
        atom_model = AM(
            atomtype_hfvr_pre_trained_path=atom_type_param_model_path,
            atom_mpnn_pre_trained_path=atom_mpnn_pretrained_path,
            n_rbf=n_rbf,
            n_neuron=n_neuron,
            n_embed=n_embed,
            r_cut=r_cut,
            use_nn_screening=use_nn_screening,
            precompute_hfvr=precompute_hfvr,
            ds_use_lmdb=ds_use_lmdb,
            ds_root=data_dir,
            ds_spec_type=spec_type,
            ds_max_size=ds_max_size,
            ignore_database_null=False,
            ds_in_memory=True,
            use_GPU=True,
            pre_trained_model_path=pretrained_model,
        )
        skip_compile = False
    dataloader_num_workers = 0
    if torch.cuda.is_available() and omp_num_threads > 2:
        dataloader_num_workers = omp_num_threads - 2
    print(atom_model.dataset)
    atom_model.train(
        n_epochs=n_epochs,
        batch_size=batch_size,
        lr=lr,
        split_percent=0.9,
        model_path=model_path,
        shuffle=True,
        dataloader_num_workers=dataloader_num_workers,
        world_size=world_size,
        omp_num_threads_per_process=omp_num_threads,
        random_seed=random_seed,
        skip_compile=skip_compile,
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
    ds_type="total_component_energies",
):
    # Ensure param_start_mean and param_start_std are lists
    """
    Train a pairwise APNet-style model using specified architecture, pretrained components, dataset, and training hyperparameters.
    
    Parameters:
        apnet_model_type (str): Identifier of the APNet variant to train (e.g., "APNet2", "APNet2-fused", "APNet3-fused", "dAPNet2", "AM-DimerParam", "AtomTypeParamModel").
        model_out (str): Path where the trained model will be saved.
        am_model_path (str): Path to a pretrained atom-model checkpoint used to initialize the atom encoder.
        atom_type_param_model_path (str): Path to a pretrained AtomTypeParam model (used by fused or dimer-prop workflows).
        atom_type_param_model_path2 (str): Secondary AtomTypeParam model path (used for fused variants that separate HF/VW and ELST parts).
        data_dir (str): Root directory of the pairwise/dimer dataset.
        n_epochs (int): Number of training epochs.
        lr (float): Initial learning rate.
        random_seed (int): Seed used for dataset splitting and training reproducibility.
        spec_type (int): Dataset specification/version selector passed through to dataset construction.
        r_cut_im (float): Interaction cutoff distance used by long-range components (if applicable).
        r_cut (float): Local neighbor cutoff distance used by the model.
        n_rbf (int): Number of radial basis functions used for distance expansion.
        n_neuron (int): Width of dense layers in network blocks.
        n_embed (int): Dimensionality of learned embeddings.
        n_params (int): Number of learnable dimer parameters when training parametric dimer models.
        m1, m2 (str): Optional mass/identifier overrides forwarded into dataset construction (passed through as-is).
        pre_trained_model_path (str): Path to an APNet checkpoint to initialize the pairwise model (overrides model_out if provided).
        param_start_mean (float or list[float]): Initial mean(s) for dimer parameter initialization; broadcast to length `n_params` if scalar.
        param_start_std (float or list[float]): Initial standard deviation(s) for dimer parameter initialization; broadcast to length `n_params` if scalar.
        dimer_eval_type (str): Evaluation type for dimer properties (controls which param/model variant is used, e.g., "elst_damping").
        ds_in_memory (bool): If True, load dataset into memory when supported.
        ds_class_type (str): Dataset class selection token forwarded to dataset constructor.
        DimerProp_model_type (str): Dimer property/model type name used by AM-DimerParam workflows.
        ap2_pretrained_model_only (str or None): If provided for APNet3-fused variants, path to AP2-only weights to load into the APNet3 model.
        ds_type (str): Dataset energy/type selector (e.g., "total_component_energies" or "fsapt_energies").
    
    Notes:
        - Scalar `param_start_mean` and `param_start_std` values are automatically expanded to lists of length `n_params`.
        - The function prints training status, infers available GPUs to set world size, instantiates the requested APNet variant with appropriate submodels and dataset settings, and calls its `train(...)` method. It does not return a value.
    """
    if not isinstance(param_start_mean, (list, tuple)):
        param_start_mean = [param_start_mean] * n_params
    if not isinstance(param_start_std, (list, tuple)):
        param_start_std = [param_start_std] * n_params
    ds_atomic_batch_size = 4 * 256
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
        ds_batch_size = 16
    elif apnet_model_type == "APNet3-fused-variant":
        APNet = AtomPairwiseModels.apnet3_fused_variants.APNet3_AtomType_Model
        # Note: presently ap3_fused_ds requires atomic batch size to be <=
        # n_objects. NEDS FIXED
        ds_atomic_batch_size = 16
        ds_datapoint_storage_n_objects = 16
        ds_batch_size = 16
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
    if torch.cuda.is_available():
        world_size = torch.cuda.device_count()
    else:
        world_size = 1
    print("World Size", world_size)

    omp_num_threads_per_process = 8
    if os.path.exists(model_out) and pre_trained_model_path is None:
        pretrained_model = model_out
        print(f"\nTraining from {model_out}\n")
    elif pre_trained_model_path is not None:
        pretrained_model = pre_trained_model_path
        print(f"\nTraining from {pre_trained_model_path}\n")
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
        if (
            dimer_eval_type in ["elst_damping__induced_dipole", "elst_damping"]
            and atom_type_param_model_path is not None
        ):
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
            model_type=DimerProp_model_type,
        )
    elif apnet_model_type in ["APNet3-fused", "APNet3-fused-variant"]:
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
        if ds_type == "fsapt_energies":
            use_precomputed_classical = False
        else:
            use_precomputed_classical = True
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
            use_precomputed_classical=use_precomputed_classical,
            ds_type=ds_type,
            ds_batch_size=ds_batch_size,
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
    """
    Parse command-line arguments, configure randomness, and invoke atom-level and pairwise model training based on the provided options.
    
    This function builds an argparse.ArgumentParser with many training and dataset options, parses the command-line, normalizes embedding start parameters, pretty-prints the parsed arguments, sets global random seeds for reproducibility, and then conditionally calls train_atom_model and/or train_pairwise_model when their respective flags are provided. The function does not return a value.
    """
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
        help="specify AtomTypeParamModel to use for AtomTypeParam Dimer props or AtomInducedDipoleModel (default: None)",
    )
    args.add_argument(
        "--atom_mpnn_pretrained_path",
        type=str,
        default=None,
        help="specify pretrained AtomMPNN model path for InducedDipoleModel with frozen charge/dipole/quadrupole layers (default: None)",
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
        default=None,
        help="specify a special loaded model. Currently only used for dAP-Net2 and AP-Net3-fused training. If set to None for AP3, ap_model_path will be treated as both model_out and pretrained_model (default: None)",
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
        help="Train APNet Model: (APNet2, APNet3-fused, APNet3-fused-variant, dAPNet2, APNet2-fused, AM-DimerParam)",
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
        "--n_message_atom",
        type=int,
        default=3,
        help="specify AtomModel n_message (default: 3)",
    )
    args.add_argument(
        "--n_rbf_atom", type=int, default=8, help="specify AtomModel n_rbf (default: 8)"
    )
    args.add_argument(
        "--n_neuron_atom",
        type=int,
        default=128,
        help="specify AtomModel n_neuron (default: 128)",
    )
    args.add_argument(
        "--n_embed_atom",
        type=int,
        default=8,
        help="specify AtomModel n_embed (default: 8)",
    )
    args.add_argument(
        "--r_cut_atom",
        type=float,
        default=5.0,
        help="specify AtomModel r_cut (default: 5.0)",
    )
    args.add_argument(
        "--use_nn_screening",
        action="store_true",
        default=False,
        help="use NN-based screening for induced dipole calculation in AtomInducedDipoleModel (default: False)",
    )
    args.add_argument(
        "--precompute_hfvr",
        action="store_true",
        default=False,
        help="pre-compute Hirshfeld volume ratios and valence widths during dataset processing for faster training (default: False)",
    )
    args.add_argument(
        "--ds_use_lmdb",
        action="store_true",
        default=False,
        help="use LMDB-based dataset storage for InducedDipoleModel training (default: False). Requires spec_type_am to be 5, 9, 10, or 11",
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
        "--DimerProp_model_type",
        type=str,
        default="AtomTypeParamNN",
        help="Dimer Prop Model Type (default: AtomTypeParamNN, other options: AtomTypeParamMPNN)",
    )
    args.add_argument(
        "--ds_type",
        type=str,
        default="total_component_energies",
        help="Dataset type for APNet3-fused only (default: total_component_energies, other options: fsapt_energies)",
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
            atom_type_param_model_path=args.atom_type_param_model_path,
            atom_mpnn_pretrained_path=args.atom_mpnn_pretrained_path,
            model_path=args.am_model_path,
            data_dir=args.data_dir,
            spec_type=args.spec_type_am,
            n_epochs=args.n_epochs_atom,
            random_seed=args.random_seed,
            ds_max_size=args.ds_max_size,
            world_size=args.world_size_ddp,
            omp_num_threads=args.omp_num_threads,
            lr=args.lr,
            n_message=args.n_message_atom,
            n_rbf=args.n_rbf_atom,
            n_neuron=args.n_neuron_atom,
            n_embed=args.n_embed_atom,
            r_cut=args.r_cut_atom,
            use_nn_screening=args.use_nn_screening,
            precompute_hfvr=args.precompute_hfvr,
            ds_use_lmdb=args.ds_use_lmdb,
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
            ds_type=args.ds_type,
        )
    return


if __name__ == "__main__":
    main()