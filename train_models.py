from apnet_pt import AtomModels
from apnet_pt import AtomPairwiseModels
import argparse
import inspect
import os
import random
from pprint import pprint

import numpy as np
import torch


def maybe_skip_training_after_dataset_setup(model_name, dataset, build_dataset_only):
    """Print dataset info and optionally stop after dataset construction."""
    print(dataset)
    if dataset is not None:
        try:
            print(f"Dataset size: {len(dataset)}")
        except Exception as exc:
            print(f"Unable to determine dataset size: {exc}")
    if build_dataset_only:
        print(
            f"Dataset build complete for {model_name}; "
            "skipping training (--build_dataset_only)."
        )
        return True
    return False


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
    build_dataset_only=False,
):
    """
    Train a single-atom model of the specified type using data in data_dir.

    Parameters:
        atom_model_type (str): One of "AtomModel", "AtomHirshfeldModel", "AtomTypeParamModel",
            "AtomInducedDipoleModel", or "InducedDipoleModel"; selects the model class and default batch size.
        model_path (str): Path where the trained model will be saved or an existing model loaded as a pretrained checkpoint.
        atom_type_param_model_path (str or None): Path to a pretrained atom-type/HF/VR parameter model used by induced-dipole variants.
        atom_mpnn_pretrained_path (str or None): Path to a pretrained atom MPNN model (used by InducedDipoleModel).
        data_dir (str): Root directory containing the atomic dataset.
        spec_type (int): Dataset specification/type identifier used by the dataset loader.
        testing (bool): Reserved flag (no effect on training flow).
        n_epochs (int): Number of training epochs.
        random_seed (int): Seed for RNGs to support reproducibility.
        ds_max_size (int or None): Maximum number of datapoints to load from the dataset; None for no limit.
        world_size (int): Number of distributed processes (GPUs) participating in training.
        omp_num_threads (int): Number of OpenMP threads available to each process; used to configure dataloader workers.
        lr (float): Learning rate for training.
        n_message (int): Number of message-passing steps (used by relevant atom model types).
        n_rbf (int): Number of radial basis functions used by the model.
        n_neuron (int): Width of hidden layers (neurons) in network components.
        n_embed (int): Size of embedding vectors for atomic features.
        r_cut (float): Cutoff radius for neighbor interactions.
        use_nn_screening (bool): If true, enable learned neural-network screening used by induced-dipole models.
        precompute_hfvr (bool): If true, enable precomputation of HF/VR features where supported.
        ds_use_lmdb (bool): If true, configure dataset to use LMDB storage (applied to InducedDipoleModel).
        build_dataset_only (bool): If true, build/process the dataset and exit without training.

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
    if maybe_skip_training_after_dataset_setup(
        atom_model_type,
        atom_model.dataset,
        build_dataset_only,
    ):
        return
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
    end_lr=None,
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
    elst_damping_type="CLIFF",
    ds_in_memory=False,
    ds_class_type="pt",
    DimerProp_model_type="AtomTypeParamNN",
    ap2_pretrained_model_only=None,
    ds_type="total_component_energies",
    no_disp_nn=False,
    use_precomputed_classical=None,
    freeze_dimer_prop_model=True,
    freeze_atom_model=True,
    build_dataset_only=False,
    include_total_mse=False,
):
    # Ensure param_start_mean and param_start_std are lists
    """
    Create and train an APNet-style pairwise model variant on the specified dataset.

    This function selects and configures an APNet variant (e.g., APNet2, dAPNet2, APNet3-fused, AM-DimerParam, AtomTypeParamModel), prepares any required submodels or pretrained weights, configures dataset and training hyperparameters, and runs training to save the resulting model to model_out.

    Parameters:
        apnet_model_type (str): Which APNet variant to train (e.g., "APNet2", "dAPNet2", "APNet3-fused", "APNetD3", "AM-DimerParam", "AtomTypeParamModel").
        model_out (str): Path where the trained APNet model will be written.
        am_model_path (str): Path to a pretrained single-atom model used by APNet as needed.
        atom_type_param_model_path (str): Path to a pretrained AtomTypeParamModel (used by some fused/dimer variants).
        atom_type_param_model_path2 (str): Optional second AtomTypeParamModel path used by fused variants for the dimer prop model.
        data_dir (str): Root directory of the pairwise dataset.
        n_epochs (int): Number of training epochs.
        lr (float): Initial learning rate.
        end_lr (float or None): Final learning rate for exponential decay over n_epochs; currently supported for APNetD3.
        lr_decay (float or None): Learning-rate decay factor (unused by default in this function).
        random_seed (int): Seed for dataset/model randomness.
        spec_type (int): Dataset specification/type identifier passed to dataset constructors.
        r_cut_im (float): Imaginary/long-range cutoff radius used by some models.
        r_cut (float): Short-range cutoff radius used by the models.
        n_rbf (int): Number of radial basis functions in the model.
        n_neuron (int): Width of dense layers in the network.
        n_embed (int): Size of embedding vectors for atomic features.
        n_params (int): Number of per-dimer parameters when training parametric dimer models.
        m1 (str): Optional molecular identifier or filter passed into dataset creation (used by some variants).
        m2 (str): Optional second molecular identifier or filter passed into dataset creation.
        pre_trained_model_path (str or None): External APNet pretrained checkpoint to initialize from.
        param_start_mean (float or list[float]): Initial mean(s) for parametric dimer parameters; broadcast to length n_params if scalar.
        param_start_std (float or list[float]): Initial stddev(s) for parametric dimer parameters; broadcast to length n_params if scalar.
        dimer_eval_type (str): Evaluation mode for dimer models (e.g., "elst_damping", "elst_damping__induced_dipole").
        elst_damping_type (str): Electrostatic damping variant for dimer prop models (e.g., "CLIFF", "AMOEBA").
        ds_in_memory (bool): Whether datasets should be loaded entirely into memory for applicable model types.
        ds_class_type (str): Dataset class/storage type identifier (e.g., "pt").
        DimerProp_model_type (str): Dimer property model type name used when constructing AM-DimerParam models.
        ap2_pretrained_model_only (str or None): If provided for APNet3-fused variants, load AP2 weights from this path into the APNet.
        ds_type (str): Dataset energy-type selector (e.g., "total_component_energies", "fsapt_energies").
        no_disp_nn (bool): Skip the dispersion readout when training APNet3-fused-d3 and compute D3 at predict time instead.
        build_dataset_only (bool): If true, build/process the dataset and exit without training.
        include_total_mse (bool): If true, add an extra MSE term on the total energy in addition to the four component-wise terms.

    """
    if not isinstance(param_start_mean, (list, tuple)):
        param_start_mean = [param_start_mean] * n_params
    if not isinstance(param_start_std, (list, tuple)):
        param_start_std = [param_start_std] * n_params
    ds_atomic_batch_size = 4 * 256
    ds_datapoint_storage_n_objects = 16
    ds_batch_size = 16
    if no_disp_nn and apnet_model_type != "APNet3-fused-d3":
        print(
            f"WARNING: --no_disp_nn applies only to APNet3-fused-d3 (requested {apnet_model_type}); ignoring flag."
        )
        no_disp_nn = False
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
    elif apnet_model_type in ["APNetD3", "APNet3D3", "APNet3-d3-fused"]:
        APNet = AtomPairwiseModels.apnet3_d3_fused.APNet3D3_AtomType_Model
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
    elif apnet_model_type == "APNet3-fused-d3":
        APNet = AtomPairwiseModels.apnet3_d3_fused.APNet3D3_AtomType_Model
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
    normalized_type = apnet_model_type.lower()
    supports_end_lr = normalized_type in {
        "apnetd3",
        "apnet3d3",
        "apnet3-d3-fused",
        "apnet3-fused-d3",
    }
    if end_lr is not None and not supports_end_lr:
        raise ValueError("end_lr is currently only supported for APNetD3 training")
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
            elst_damping_type=elst_damping_type,
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
            freeze_atom_model=True,
        )
        atom_type_elst_model = AtomPairwiseModels.mtp_mtp.AM_DimerParam_Model(
            ds_root=None,
            use_GPU=False,
            ignore_database_null=True,
            atom_model=atom_type_hf_vw_model.model,
            atom_model_type="AtomTypeParamNN",
            pre_trained_model_path=atom_type_param_model_path2,
            elst_damping_type=elst_damping_type,
            freeze_atom_model=freeze_atom_model,
        )
        am_model_path = None
        print(f"{ds_atomic_batch_size=}, {ds_datapoint_storage_n_objects=}")
        if use_precomputed_classical is None:
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
            freeze_dimer_prop_model=freeze_dimer_prop_model,
        )
        if ap2_pretrained_model_only is not None:
            print(f"Loading AP2 pretrained weights from {ap2_pretrained_model_only}")
            apnet.load_ap2_pretrained_weights(ap2_pretrained_model_only)
    elif apnet_model_type in ["APNet3-fused-d3"]:
        print("Setting AtomTypeParams...")
        atom_type_hf_vw_model = AtomPairwiseModels.mtp_mtp.AtomTypeParamModel(
            ds_root=None,
            use_GPU=False,
            ignore_database_null=True,
            atom_model_pre_trained_path=am_model_path,
            pre_trained_model_path=atom_type_param_model_path,
            freeze_atom_model=True,
        )
        atom_type_elst_model = AtomPairwiseModels.mtp_mtp.AM_DimerParam_Model(
            ds_root=None,
            use_GPU=False,
            ignore_database_null=True,
            atom_model=atom_type_hf_vw_model.model,
            atom_model_type="AtomTypeParamNN",
            pre_trained_model_path=atom_type_param_model_path2,
            elst_damping_type=elst_damping_type,
            freeze_atom_model=freeze_atom_model,
        )
        am_model_path = None
        print(f"{ds_atomic_batch_size=}, {ds_datapoint_storage_n_objects=}")
        if use_precomputed_classical is None:
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
            no_disp_nn=no_disp_nn,
            ds_batch_size=ds_batch_size,
            freeze_dimer_prop_model=freeze_dimer_prop_model,
        )
        if ap2_pretrained_model_only is not None:
            print(f"Loading AP2 pretrained weights from {ap2_pretrained_model_only}")
            apnet.load_ap2_pretrained_weights(ap2_pretrained_model_only)
    elif apnet_model_type in ["APNetD3", "APNet3D3", "APNet3-d3-fused"]:
        print("Setting AtomTypeParams...")
        atom_type_hf_vw_model = AtomPairwiseModels.mtp_mtp.AtomTypeParamModel(
            ds_root=None,
            use_GPU=False,
            ignore_database_null=True,
            atom_model_pre_trained_path=am_model_path,
            pre_trained_model_path=atom_type_param_model_path,
            freeze_atom_model=True,
        )
        atom_type_elst_model = AtomPairwiseModels.mtp_mtp.AM_DimerParam_Model(
            ds_root=None,
            use_GPU=False,
            ignore_database_null=True,
            atom_model=atom_type_hf_vw_model.model,
            atom_model_type="AtomTypeParamNN",
            pre_trained_model_path=atom_type_param_model_path2,
            elst_damping_type=elst_damping_type,
            freeze_atom_model=freeze_atom_model,
        )
        am_model_path = None
        print(f"{ds_atomic_batch_size=}, {ds_datapoint_storage_n_objects=}")
        if use_precomputed_classical is None:
            if ds_type == "fsapt_energies":
                use_precomputed_classical = False
            else:
                use_precomputed_classical = True
        apnet = APNet(
            atom_type_model=atom_type_hf_vw_model.model,
            dimer_prop_model=atom_type_elst_model.dimer_model,
            am_dimer_param_model=atom_type_elst_model,
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
    dataset = getattr(apnet, "dataset", None)
    if maybe_skip_training_after_dataset_setup(
        apnet_model_type,
        dataset,
        build_dataset_only,
    ):
        return
    train_kwargs = dict(
        model_path=model_out,
        n_epochs=n_epochs,
        world_size=world_size,
        omp_num_threads_per_process=omp_num_threads_per_process,
        lr=lr,
        dataloader_num_workers=4,
        random_seed=random_seed,
        include_total_mse=include_total_mse,
    )
    if apnet_model_type in ["APNetD3", "APNet3D3", "APNet3-d3-fused"]:
        train_kwargs["end_lr"] = end_lr
    else:
        train_kwargs["lr_decay"] = lr_decay
    supported_train_kwargs = inspect.signature(apnet.train).parameters
    unsupported_train_kwargs = sorted(
        key for key in train_kwargs if key not in supported_train_kwargs
    )
    if unsupported_train_kwargs:
        print(
            "Skipping unsupported train() kwargs for "
            f"{apnet_model_type}: {', '.join(unsupported_train_kwargs)}"
        )
        train_kwargs = {
            key: value
            for key, value in train_kwargs.items()
            if key in supported_train_kwargs
        }
    apnet.train(**train_kwargs)
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
    Parse command-line arguments and run configured model training routines.

    Parses command-line options that configure atom and pairwise (APNet) training, converts the parameter-start mean/std strings to numeric lists, sets global random seeds, prints the parsed arguments, and invokes train_atom_model and/or train_pairwise_model when the corresponding flags are provided.
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
        default="./models/ap_default.pt",
        help="specify where to save output model (default: ./models/ap_default.pt)",
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
        help="Train APNet Model: (APNet2, APNet3-fused, APNet3-fused-variant, APNet3-fused-d3, dAPNet2, APNet2-fused, AM-DimerParam)",
    )
    args.add_argument(
        "--dimer_eval_type",
        type=str,
        default="elst_damping",
        help="Specify dimer eval type for AM-DimerParam (default: 'elst_damping', other options: 'induced_dipole)",
    )
    args.add_argument(
        "--elst_damping_type",
        type=str,
        default="CLIFF",
        choices=["CLIFF", "AMOEBA"],
        help="Electrostatic damping type: 'CLIFF' (CLIFF/GORDON2) or 'AMOEBA' (GORDON1) (default: 'CLIFF')",
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
        "--end_lr",
        type=float,
        default=None,
        help="Final learning rate for exponential decay over n_epochs (APNetD3 only)",
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
        "--use_precomputed_classical",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Override whether APNet3-fused/APNet3-fused-d3 uses precomputed "
            "classical terms. When unset, the existing model-specific default "
            "behavior is used."
        ),
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
    args.add_argument(
        "--include_total_mse",
        action="store_true",
        default=False,
        help=(
            "AP2/AP3-D3 training: add a fifth MSE term on the total energy "
            "in addition to the four component losses."
        ),
    )
    args.add_argument(
        "--no_disp_nn",
        action="store_true",
        default=False,
        help="APNet3-fused-d3 only: train elst/exch/indu (three components) and compute D3 at predict time instead of a dispersion NN.",
    )
    args.add_argument(
        "--unfreeze_dimer_prop_model",
        action="store_true",
        default=False,
        help="APNet3-fused/APNet3-fused-d3: unfreeze the dimer_prop_model submodel during training (default: frozen).",
    )
    args.add_argument(
        "--unfreeze_atom_model",
        action="store_true",
        default=False,
        help="APNet3-fused/APNet3-fused-d3: unfreeze the atom-type submodel feeding DimerProp during training (default: frozen).",
    )
    args.add_argument(
        "--build_dataset_only",
        action="store_true",
        default=False,
        help="Build/process the requested dataset and exit without training.",
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
            build_dataset_only=args.build_dataset_only,
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
            end_lr=args.end_lr,
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
            elst_damping_type=args.elst_damping_type,
            ds_in_memory=args.ds_in_memory,
            ds_class_type=args.ds_class_type,
            DimerProp_model_type=args.DimerProp_model_type,
            ap2_pretrained_model_only=args.ap2_pretrained_model_only,
            ds_type=args.ds_type,
            no_disp_nn=args.no_disp_nn,
            use_precomputed_classical=args.use_precomputed_classical,
            freeze_dimer_prop_model=not args.unfreeze_dimer_prop_model,
            freeze_atom_model=not args.unfreeze_atom_model,
            build_dataset_only=args.build_dataset_only,
            include_total_mse=args.include_total_mse,
        )
    return


if __name__ == "__main__":
    main()
