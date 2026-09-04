from apnet_pt import AtomModels
from apnet_pt import AtomPairwiseModels
from apnet_pt import ddp_launch
from apnet_pt.training_tracking import WandbConfig
from apnet_pt.util import load_split_manifest
import argparse
import inspect
import json
import os
import random
from dataclasses import replace
from pprint import pprint
from uuid import uuid4

import numpy as np
import torch


RACKERS_MODEL_TYPES = {
    "RackersTholeDampingModel",
    "RackersTholeDampingOverlapModel",
}
RACKERS_PARAM_START_MEAN = [1.8, 0.34, 0.39, 1.8]
RACKERS_PARAM_START_STD = [0.01, 0.01, 0.01, 0.01]

# CLIFF classical routes.  Each identifier is deliberately spelled exactly like
# its `mtp_mtp` harness class, so the dispatch below resolves it by name.
CLIFF_MODEL_TYPES = {
    "CliffExchangeModel",
    "CliffClassicalModel",
    "CliffClassicalOverlapModel",
    "CliffClassicalOverlapMPNNModel",
}
# The CLIFF routes whose parameter head does its own message passing, and which
# therefore accept the `--param_*` architecture flags. Rejected elsewhere: the
# other heads have no message passing to size, so accepting the flags would
# report an architecture the run never had.
CLIFF_MPNN_MODEL_TYPES = {
    "CliffClassicalOverlapMPNNModel",
}
# The CLIFF routes predicting more than one SAPT component.  Only these have a
# meaningful total-versus-component loss split, so `--component_gamma` and
# `--total_includes_d3` are accepted here and rejected everywhere else --
# including on `CliffExchangeModel`, which fits `Exch` alone.
COMBINED_CLIFF_MODEL_TYPES = {
    "CliffClassicalModel",
    "CliffClassicalOverlapModel",
    "CliffClassicalOverlapMPNNModel",
}
# Per-component clipping needs disjoint trainable parameter groups. The dense
# heads have one embedding/readout stack per physical column and a frozen nested
# atom model; the MPNN head has a shared trainable featurizer and is rejected.
COMPONENT_CLIP_CLIFF_MODEL_TYPES = {
    "CliffClassicalModel",
    "CliffClassicalOverlapModel",
}
# `--include_total_mse` predates `--component_gamma` and is filtered out of
# `AM_DimerParam_Model.train`, which never accepted it.  On a CLIFF route it is
# reinterpreted as this gamma rather than silently dropped.
CLIFF_INCLUDE_TOTAL_MSE_GAMMA = 0.5
# Every route whose parameter head follows an `AtomTypeParamNN` positive-
# parameter contract.  The two-stage HFVR/valence-width construction,
# `world_size = 1`, the absent legacy checkpoint default, and list-only
# parameter initialization are shared by all of them.
POSITIVE_PARAM_MODEL_TYPES = RACKERS_MODEL_TYPES | CLIFF_MODEL_TYPES

LEGACY_PAIRWISE_PRETRAINED_MODEL_PATH = "./models/dapnet2/ap2_0.pt"
_PAIRWISE_PRETRAINED_MODEL_PATH_UNSET = object()


def _cliff_parameter_contract(apnet_model_type):
    """Return ``(parameter_names, default_means, default_stds)`` for a CLIFF route.

    Read out of `mtp_mtp` rather than restated here, so the CLI's expected
    override length and its default initialization cannot drift from the harness
    contract they are validated against.  Resolved per call so a test that
    monkeypatches the module still sees its own constants.
    """
    mtp = AtomPairwiseModels.mtp_mtp
    if apnet_model_type == "CliffExchangeModel":
        return (
            mtp.CLIFF_EXCH_PARAMETER_NAMES,
            mtp.CLIFF_EXCH_INITIAL_VALUES,
            mtp.CLIFF_EXCH_INITIAL_STDS,
        )
    return (
        mtp.CLIFF_CLASSICAL_PARAMETER_NAMES,
        mtp.CLIFF_CLASSICAL_INITIAL_VALUES,
        mtp.CLIFF_CLASSICAL_INITIAL_STDS,
    )


def str2bool(value):
    """Parse a boolean CLI value.

    ``type=bool`` is a trap for argparse: it applies ``bool()`` to the raw
    string, so ``--flag False`` evaluates to ``True`` and the only way to get
    ``False`` is to omit the flag entirely.  This accepts the spellings a shell
    launcher is likely to pass and rejects anything ambiguous.
    """
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in ("true", "t", "yes", "y", "1"):
        return True
    if normalized in ("false", "f", "no", "n", "0"):
        return False
    raise argparse.ArgumentTypeError(
        f"expected a boolean value, got {value!r}"
    )


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


def build_wandb_run_configs(args, environment=None):
    """Build atom/pairwise W&B configs, grouping sequential runs together.

    ``environment`` defaults to ``os.environ`` so callers and tests can supply
    an explicit mapping instead of mutating the process environment.
    """

    env = os.environ if environment is None else environment
    extra_config = {}
    wandb_config_file = getattr(args, "wandb_config_file", None)
    if wandb_config_file is not None:
        with open(wandb_config_file) as config_file:
            extra_config = json.load(config_file)
        if not isinstance(extra_config, dict):
            raise ValueError("--wandb-config-file must contain a JSON object")

    base_config = WandbConfig(
        mode=args.wandb_mode,
        run_id=getattr(args, "wandb_run_id", None),
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name,
        group=args.wandb_group,
        tags=tuple(args.wandb_tags),
        job_type=args.wandb_job_type,
        notes=args.wandb_notes,
        directory=args.wandb_dir,
        extra_config=extra_config,
    )
    dual_run = args.train_am != "" and args.train_apnet != ""
    resolved_group = base_config.group or env.get("WANDB_RUN_GROUP")
    if dual_run and resolved_group is None:
        resolved_group = f"train-models-{uuid4().hex[:12]}"
    resolved_name = base_config.name or env.get("WANDB_NAME")
    atom_config = replace(
        base_config,
        group=resolved_group,
        name=f"{resolved_name}-atom" if dual_run and resolved_name else resolved_name,
    )
    pairwise_config = replace(
        base_config,
        group=resolved_group,
        name=(
            f"{resolved_name}-pairwise" if dual_run and resolved_name else resolved_name
        ),
    )
    return atom_config, pairwise_config


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
    split_manifest=None,
    split_verify="all",
    skip_compile=None,
    am_model_path_for_inner=None,
    freeze_inner_atom_model=True,
    wandb_config=None,
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
        split_manifest (str or None): Path to a train/test split manifest CSV (columns index, split, fingerprint). When set, replaces the uniform random split_percent draw with the split the manifest describes.
        split_verify (str or int): How much of the manifest to verify against the dataset -- "all", "none", or a sample count. Verification is what distinguishes a valid manifest from a stale one.
        skip_compile (bool or None): Force torch.compile off (True) or on (False). None keeps the per-model-type default. AtomTypeParamModel's forward writes into a slice of a mask-filtered tensor, which Inductor cannot guard on and which raises GuardOnDataDependentSymNode on some torch builds; running eager is the workaround.
        am_model_path_for_inner (str or None): AtomTypeParamNN only -- the pretrained AtomMPNN checkpoint that supplies the inner multipole model.
        freeze_inner_atom_model (bool): AtomTypeParamNN only -- keep the inner AtomMPNN frozen (default) so only the parameter head is fitted.

    """
    # The per-model-type branches below assign `skip_compile` themselves, so
    # the caller's request has to be captured before they clobber it.
    skip_compile_requested = skip_compile
    if atom_model_type == "AtomModel":
        AM = AtomModels.ap2_atom_model.AtomModel
        batch_size = 16
    elif atom_model_type == "AtomHirshfeldModel":
        AM = AtomModels.ap2_hirshfeld_atom_model.AtomHirshfeldModel
        batch_size = 1
    elif atom_model_type == "AtomTypeParamModel":
        # NOTE: this is AtomModels.ap3_atomtype_mpnn.AtomTypeParamModel, whose
        # module is AtomTypeParamMPNN -- a standalone MPNN with its own
        # embedding and per-parameter layers (133 tensors). It predicts the same
        # two targets but is NOT the class the CLIFF routes load. Use
        # "AtomTypeParamNN" below for that. The name collision between the two
        # AtomTypeParamModel classes is easy to trip over: a checkpoint from
        # this route fails to load into CLIFF with a wall of missing
        # "atom_model.*" keys.
        AM = AtomModels.ap3_atomtype_mpnn.AtomTypeParamModel
        batch_size = 16
    elif atom_model_type == "AtomTypeParamNN":
        # AtomPairwiseModels.mtp_mtp.AtomTypeParamModel, whose module is
        # AtomTypeParamNN: a full AtomMPNN under `atom_model.*` plus a
        # guess_layer and per-parameter readouts (201 tensors). This is what
        # --atom_type_param_model_path feeds to the CLIFF and Rackers routes,
        # and what models/ap3_saptpbe0/1/atp_hfvr_1.pt is.
        AM = AtomPairwiseModels.mtp_mtp.AtomTypeParamModel
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
    if atom_model_type == "AtomTypeParamNN":
        # This class takes its inner AtomMPNN from a checkpoint rather than
        # building one from n_* hyperparameters, so its constructor differs
        # from the three below.
        atom_model = AM(
            atom_model_pre_trained_path=am_model_path_for_inner,
            pre_trained_model_path=pretrained_model,
            n_message=n_message,
            n_neuron=n_neuron,
            n_embed=n_embed,
            r_cut=r_cut,
            ds_root=data_dir,
            ds_spec_type=spec_type,
            ds_max_size=ds_max_size,
            ignore_database_null=False,
            ds_in_memory=True,
            use_GPU=True,
            freeze_atom_model=freeze_inner_atom_model,
        )
        skip_compile = False
    elif atom_model_type in ["AtomModel", "AtomHirshfeldModel", "AtomTypeParamModel"]:
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
    if skip_compile_requested is not None:
        skip_compile = bool(skip_compile_requested)
    train_indices = test_indices = None
    if split_manifest:
        train_indices, test_indices = load_split_manifest(
            split_manifest,
            dataset=atom_model.dataset,
            verify=split_verify,
        )
    atom_model.train(
        n_epochs=n_epochs,
        batch_size=batch_size,
        lr=lr,
        split_percent=0.9,
        train_indices=train_indices,
        test_indices=test_indices,
        model_path=model_path,
        shuffle=True,
        dataloader_num_workers=dataloader_num_workers,
        world_size=world_size,
        omp_num_threads_per_process=omp_num_threads,
        random_seed=random_seed,
        skip_compile=skip_compile,
        wandb_config=wandb_config,
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
    pre_trained_model_path=_PAIRWISE_PRETRAINED_MODEL_PATH_UNSET,
    param_start_mean=None,
    param_start_std=None,
    dimer_eval_type="elst_damping",
    elst_damping_type="CLIFF",
    ds_in_memory=False,
    ds_max_size=None,
    ds_max_size_val=None,
    batch_size=None,
    param_n_message=None,
    param_n_rbf=None,
    param_hidden=None,
    param_r_cut=None,
    frozen_parameters=None,
    shared_damping_parameters=None,
    ds_exclude_elements=None,
    ds_exclude_train_indices_path=None,
    split_manifest=None,
    ds_class_type="pt",
    DimerProp_model_type="AtomTypeParamNN",
    ap2_pretrained_model_only=None,
    ds_type="total_component_energies",
    no_disp_nn=False,
    use_precomputed_classical=None,
    readout_decay_mode="legacy-r3",
    readout_exchange_scale=1.0,
    readout_induction_scale=1.0,
    readout_hybrid_start=2.5,
    readout_hybrid_end=3.5,
    freeze_dimer_prop_model=True,
    freeze_atom_model=True,
    build_dataset_only=False,
    include_total_mse=False,
    loss_mode="mse",
    huber_delta=1.0,
    closest_contact_bin_edges=None,
    closest_contact_bin_counts=None,
    component_gamma=None,
    total_includes_d3=False,
    grad_clip_norm=None,
    grad_clip_mode="global",
    thole_lr=None,
    trainable_polarizability_scale=False,
    polarizability_lr=None,
    atom_model_lr=None,
    induction_diagnostics=False,
    induction_convergence_threshold=None,
    induction_max_iterations=None,
    induction_convergence_norm=None,
    shard_locality_block_shards=0,
    omp_num_threads=8,
    ddp_world_size=1,
    external_rank=None,
    external_local_rank=None,
    dataloader_num_workers=None,
    wandb_config=None,
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
        pre_trained_model_path (str or None): External APNet pretrained checkpoint to initialize from. When omitted, Rackers and CLIFF routes start without an outer checkpoint while legacy routes retain the historical dAPNet checkpoint default.
        param_start_mean (float or list[float] or None): Initial parameter means. Rackers routes require exactly four values, CliffExchangeModel exactly one, and the CliffClassical routes exactly five; each uses its physical defaults when unset. Other routes use 1.5 when unset and broadcast scalars to n_params.
        param_start_std (float or list[float] or None): Initial parameter standard deviations. Rackers routes require exactly four values, CliffExchangeModel exactly one, and the CliffClassical routes exactly five; each uses its physical defaults when unset. Other routes use 0.1 when unset and broadcast scalars to n_params.
        dimer_eval_type (str): Evaluation mode for dimer models (e.g., "elst_damping", "elst_damping__induced_dipole").
        elst_damping_type (str): Electrostatic damping variant for dimer prop models (e.g., "CLIFF", "AMOEBA").
        ds_in_memory (bool): Whether datasets should be loaded entirely into memory for applicable model types.
        ds_max_size (int or None): Truncate the pairwise dataset to N datapoints. Useful for small smoke-test runs; None uses the full dataset. On a split store it caps both splits unless ds_max_size_val overrides the validation one.
        ds_max_size_val (int or None): Separate cap for the validation split. None (the default) reuses ds_max_size, which is the historical behaviour. Positive-parameter routes only (Rackers and CLIFF), and requires ds_max_size.
        param_n_message (int or None): Message-passing depth of the parameter head. Message-passing CLIFF routes only; None keeps the head's default.
        param_n_rbf (int or None): Radial basis size of the parameter head's own distance expansion. Message-passing CLIFF routes only.
        param_hidden (int or None): Hidden-state width of the parameter head's message passing. Message-passing CLIFF routes only.
        param_r_cut (float or None): Cutoff radius for the parameter head's message passing. Message-passing CLIFF routes only.
        frozen_parameters (list[str] or None): Parameter columns to hold at their per-element seed instead of fitting, e.g. thole_direct thole_mutual to stop the induced-dipole response operator being a learned object. CLIFF routes only.
        shared_damping_parameters (list[str] or None): Damping columns to fit as one global scalar shared by every atom and every listed column, instead of one value per element. CLIFF routes only, and disjoint from frozen_parameters.
        batch_size (int or None): Dimers per optimizer step. None (the default) keeps each route's historical dataset batch size. The trainers read this off the dataset as `training_batch_size`, so it is set there rather than on train().
        ds_exclude_elements (list[int] or None): Atomic numbers to exclude. Any dimer containing one is dropped before ds_max_size is applied, so ds_max_size counts surviving dimers. Positive-parameter routes only (Rackers and CLIFF).
        ds_exclude_train_indices_path (str or None): Sorted unique `.npy` indices to remove from the capped training split while leaving validation unchanged. Positive-parameter split-store routes only; mutually exclusive with ds_exclude_elements.
        ds_class_type (str): Dataset class/storage type identifier (e.g., "pt").
        DimerProp_model_type (str): Dimer property model type name used when constructing AM-DimerParam models.
        ap2_pretrained_model_only (str or None): If provided for APNet3-fused variants, load AP2 weights from this path into the APNet.
        ds_type (str): Dataset energy-type selector (e.g., "total_component_energies", "fsapt_energies").
        no_disp_nn (bool): Skip the dispersion readout when training APNet3-fused-d3 and compute D3 at predict time instead.
        readout_decay_mode (str): APNet3-fused-d3 readout envelope. legacy-r3 preserves the shared inverse cube; exchange-overlap uses the physical Slater overlap for exchange; exchange-overlap-induction-r6 additionally uses inverse-sixth induction.
        readout_exchange_scale (float): Fixed positive scale applied to the Slater exchange envelope.
        readout_induction_scale (float): Fixed positive scale applied to the inverse-sixth induction envelope.
        readout_hybrid_start (float): Distance below which hybrid exchange uses exactly legacy inverse-cube decay.
        readout_hybrid_end (float): Distance above which hybrid exchange uses exactly Slater-overlap decay.
        build_dataset_only (bool): If true, build/process the dataset and exit without training.
        include_total_mse (bool): If true, add a matching total-energy term to the component loss. Historical MSE behavior is unchanged; Huber modes add total Huber. On a CLIFF route it is instead shorthand for component_gamma=0.5 and cannot be combined with an explicit component_gamma.
    loss_mode (str): APNet3-fused-d3 component objective: mse, huber, or closest-contact-macro-huber.
    huber_delta (float): Positive Huber transition in kcal/mol for Huber loss modes.
    closest_contact_bin_edges (sequence[float] or None): Nine frozen Angstrom boundaries defining ten bins for closest-contact macro-Huber.
    closest_contact_bin_counts (sequence[int] or None): Ten positive global training counts used for inverse-frequency macro weights.
        component_gamma (float or None): CLIFF Eq. (23) component/total loss weight for the combined CLIFF routes. None (the default) keeps the legacy plain multi-column MSE; any float in [0.0, 1.0] selects the Eq. (23) functional. Rejected on CliffExchangeModel and on every pre-existing route.
        total_includes_d3 (bool): If true, the CLIFF Eq. (23) total term includes D3 dispersion and is compared against all four SAPT columns. Requires an explicit component_gamma and one of the combined CLIFF routes.
        grad_clip_norm (float or None): Gradient-norm clip applied before each optimizer step. None leaves the update unclipped.
        grad_clip_mode (str): `global` clips all parameters together; `component` clips dense CLIFF ELST, EXCH, and IND groups independently.
        thole_lr (float or None): Optional Adam learning rate for only the trainable direct/mutual Thole columns; all other columns retain `lr`.
        trainable_polarizability_scale (bool): Promote the static free-atom polarizability table to a trainable per-element scale, `alpha_0(Z) -> alpha_0(Z) * exp(s_Z)`, seeded at `s = 0`. This is the only parameter that scales long-range induction; without it `alpha` is fixed by a static table and the frozen atom model's Hirshfeld ratios.
        polarizability_lr (float or None): Required whenever the scale is enabled, and the Adam learning rate for it alone. `0.0` is legal and is the control arm: the scale rides through the checkpoint at its seed, so the run stays bit-identical to one without it.
        atom_model_lr (float or None): Required whenever `--unfreeze_atom_model` trains the nested atom model on a dense combined CLIFF route, and the Adam learning rate for that trunk alone. The trunk carries 1.89M pretrained parameters against a 231k head; at the head's rate it destroys the model within one epoch, so it has to be stated rather than inherited.
        induction_diagnostics (bool): Collect and log per-epoch SCF convergence, residual, induced-dipole, and induction-energy health metrics.
        shard_locality_block_shards (int): 0 disables. Above 0, shuffle shards rather than dimers and give each loader worker a disjoint block of this many shards, backed by an LRU of the same size. Trades a smaller shuffle window for roughly one shard read per shard per epoch instead of one per dimer.
        induction_convergence_threshold (float or None): Optional Rackers/Thole SCF residual threshold. None preserves the checkpoint/default value (historically 1e-8).
        induction_max_iterations (int or None): Optional Rackers/Thole SCF iteration cap. None preserves the checkpoint/default value (historically 200).
        induction_convergence_norm (str or None): Optional reduction used by the Rackers/Thole stopping rule -- "l2" (historical, extensive in batch size), "rms", or "max". None preserves the checkpoint/default value (historically "l2").
        omp_num_threads (int): Number of OpenMP threads assigned to each training process.

    """
    is_rackers_model = apnet_model_type in RACKERS_MODEL_TYPES
    is_cliff_model = apnet_model_type in CLIFF_MODEL_TYPES
    is_positive_param_model = apnet_model_type in POSITIVE_PARAM_MODEL_TYPES
    grad_clip_mode = str(grad_clip_mode).strip().lower()
    if grad_clip_mode not in AtomPairwiseModels.mtp_mtp.CLIFF_GRAD_CLIP_MODES:
        raise ValueError(
            "grad_clip_mode must be one of "
            f"{list(AtomPairwiseModels.mtp_mtp.CLIFF_GRAD_CLIP_MODES)}, "
            f"got {grad_clip_mode!r}"
        )
    if grad_clip_mode == "component":
        if grad_clip_norm is None:
            raise ValueError(
                "grad_clip_mode='component' requires grad_clip_norm"
            )
        if apnet_model_type not in COMPONENT_CLIP_CLIFF_MODEL_TYPES:
            raise ValueError(
                "component gradient clipping is only supported on the dense "
                "combined CLIFF routes "
                f"{sorted(COMPONENT_CLIP_CLIFF_MODEL_TYPES)}, not "
                f"{apnet_model_type!r}"
            )
    if thole_lr is not None:
        thole_lr = AtomPairwiseModels.mtp_mtp._validate_bound_scale(
            thole_lr, "thole_lr"
        )
        if apnet_model_type not in COMPONENT_CLIP_CLIFF_MODEL_TYPES:
            raise ValueError(
                "thole_lr is only supported on the dense combined CLIFF "
                f"routes {sorted(COMPONENT_CLIP_CLIFF_MODEL_TYPES)}, not "
                f"{apnet_model_type!r}"
            )
    if atom_model_lr is not None:
        atom_model_lr = AtomPairwiseModels.mtp_mtp._validate_polarizability_lr(
            atom_model_lr, name="atom_model_lr"
        )
        if apnet_model_type not in COMPONENT_CLIP_CLIFF_MODEL_TYPES:
            raise ValueError(
                "atom_model_lr is only supported on the dense combined CLIFF "
                f"routes {sorted(COMPONENT_CLIP_CLIFF_MODEL_TYPES)}, not "
                f"{apnet_model_type!r}"
            )
    if trainable_polarizability_scale or polarizability_lr is not None:
        polarizability_lr = (
            AtomPairwiseModels.mtp_mtp._validate_polarizability_lr(
                polarizability_lr
            )
        )
        if apnet_model_type not in COMPONENT_CLIP_CLIFF_MODEL_TYPES:
            raise ValueError(
                "trainable_polarizability_scale is only supported on the "
                "dense combined CLIFF routes "
                f"{sorted(COMPONENT_CLIP_CLIFF_MODEL_TYPES)}, not "
                f"{apnet_model_type!r}"
            )
        # Neither half is useful alone, and each silent default would be the
        # wrong one: a rate with no parameter is a no-op, and a parameter with
        # no rate lands in the base group at the trunk's rate.
        if not trainable_polarizability_scale:
            raise ValueError(
                "polarizability_lr requires trainable_polarizability_scale"
            )
        if polarizability_lr is None:
            raise ValueError(
                "trainable_polarizability_scale requires polarizability_lr "
                "(0.0 freezes the scale at its seed)"
            )
    if induction_diagnostics and apnet_model_type not in COMBINED_CLIFF_MODEL_TYPES:
        raise ValueError(
            "induction_diagnostics is only supported on the combined CLIFF "
            f"routes {sorted(COMBINED_CLIFF_MODEL_TYPES)}, not "
            f"{apnet_model_type!r}"
        )
    if (
        induction_convergence_threshold is not None
        or induction_max_iterations is not None
        or induction_convergence_norm is not None
    ):
        if apnet_model_type not in COMBINED_CLIFF_MODEL_TYPES:
            raise ValueError(
                "induction solver controls are only supported on the combined "
                f"CLIFF routes {sorted(COMBINED_CLIFF_MODEL_TYPES)}, not "
                f"{apnet_model_type!r}"
            )
        (
            induction_convergence_threshold,
            induction_max_iterations,
        ) = AtomPairwiseModels.mtp_mtp._validate_induction_solver_controls(
            (
                AtomPairwiseModels.mtp_mtp.DEFAULT_INDUCTION_CONVERGENCE_THRESHOLD
                if induction_convergence_threshold is None
                else induction_convergence_threshold
            ),
            (
                AtomPairwiseModels.mtp_mtp.DEFAULT_INDUCTION_MAX_ITERATIONS
                if induction_max_iterations is None
                else induction_max_iterations
            ),
        )
        if induction_convergence_norm is not None:
            induction_convergence_norm = (
                AtomPairwiseModels.mtp_mtp
                ._validate_induction_convergence_norm(
                    induction_convergence_norm
                )
            )
    if split_manifest:
        # Only the atom-model route resolves a manifest into indices. Accepting
        # it here would train on the trainer's own uniform draw while the run
        # record claimed a designed split.
        raise ValueError(
            "split_manifest is only supported on the atom-model routes "
            "(--train_am), not on --train_apnet "
            f"{apnet_model_type}"
        )
    if (
        ds_exclude_train_indices_path is not None
        and not is_positive_param_model
    ):
        raise ValueError(
            "ds_exclude_train_indices_path is only supported on the Rackers "
            f"and CLIFF positive-parameter routes, not {apnet_model_type}"
        )
    if ds_exclude_elements and ds_exclude_train_indices_path is not None:
        raise ValueError(
            "ds_exclude_elements and ds_exclude_train_indices_path are "
            "mutually exclusive"
        )
    if ds_exclude_elements and not is_positive_param_model:
        # Only the positive-parameter branch forwards ds_exclude_elements into
        # the dataset constructor. Accepting it elsewhere would train on the
        # full set while the run config claimed otherwise.
        raise ValueError(
            "ds_exclude_elements is only supported on the Rackers and CLIFF "
            f"positive-parameter routes, not {apnet_model_type}"
        )
    if ds_max_size_val is not None and not is_positive_param_model:
        # Same reason as ds_exclude_elements: only that branch forwards it, so
        # elsewhere the run would evaluate the full validation split while its
        # record claimed a capped one.
        raise ValueError(
            "ds_max_size_val is only supported on the Rackers and CLIFF "
            f"positive-parameter routes, not {apnet_model_type}"
        )
    if batch_size is not None:
        # Validated here so a bad value costs nothing: the alternative is
        # discovering it after the atom model and the dataset are built.
        batch_size = AtomPairwiseModels.mtp_mtp._validate_positive_count(
            batch_size, "batch_size"
        )
    parameter_head_kwargs = {
        name: value
        for name, value in (
            ("param_n_message", param_n_message),
            ("param_n_rbf", param_n_rbf),
            ("param_hidden", param_hidden),
            ("param_r_cut", param_r_cut),
        )
        if value is not None
    }
    if frozen_parameters:
        if not is_cliff_model:
            raise ValueError(
                "frozen_parameters is only supported on the CLIFF routes "
                f"{sorted(CLIFF_MODEL_TYPES)}, not {apnet_model_type!r}"
            )
        parameter_head_kwargs["frozen_parameters"] = tuple(frozen_parameters)
    if shared_damping_parameters:
        if not is_cliff_model:
            raise ValueError(
                "shared_damping_parameters is only supported on the CLIFF "
                f"routes {sorted(CLIFF_MODEL_TYPES)}, not {apnet_model_type!r}"
            )
        parameter_head_kwargs["shared_damping_parameters"] = tuple(
            shared_damping_parameters
        )
    if parameter_head_kwargs and apnet_model_type not in CLIFF_MPNN_MODEL_TYPES:
        # Rejected, not dropped: the other heads have no message passing to
        # size, so accepting these would leave a run record describing an
        # architecture that never existed.
        architecture_only = sorted(
            set(parameter_head_kwargs)
            - {"frozen_parameters", "shared_damping_parameters"}
        )
        if architecture_only:
            raise ValueError(
                f"{', '.join(architecture_only)} "
                "is only supported on the message-passing CLIFF routes "
                f"{sorted(CLIFF_MPNN_MODEL_TYPES)}, not {apnet_model_type!r}"
            )
    if is_cliff_model and include_total_mse:
        # `--include_total_mse` is the pre-CLIFF spelling of "also fit the
        # total".  Reinterpreting it keeps the flag meaningful on the new
        # routes; accepting both spellings at once would leave which one wins
        # ambiguous, so that is an error rather than a precedence rule.
        if component_gamma is not None:
            raise ValueError(
                "include_total_mse and component_gamma cannot both be "
                "supplied on a CLIFF route; include_total_mse is shorthand "
                f"for component_gamma={CLIFF_INCLUDE_TOTAL_MSE_GAMMA}"
            )
        component_gamma = CLIFF_INCLUDE_TOTAL_MSE_GAMMA
    if apnet_model_type not in COMBINED_CLIFF_MODEL_TYPES:
        # Rejected rather than silently dropped: the shared pairwise tail
        # filters train_kwargs by signature, so a route whose train() lacks
        # these would otherwise discard them without a word.
        if component_gamma is not None:
            raise ValueError(
                "component_gamma is only supported for the combined CLIFF "
                f"routes {sorted(COMBINED_CLIFF_MODEL_TYPES)}, not "
                f"{apnet_model_type!r}"
            )
        if total_includes_d3:
            raise ValueError(
                "total_includes_d3 is only supported for the combined CLIFF "
                f"routes {sorted(COMBINED_CLIFF_MODEL_TYPES)}, not "
                f"{apnet_model_type!r}"
            )
    if pre_trained_model_path is _PAIRWISE_PRETRAINED_MODEL_PATH_UNSET:
        if is_positive_param_model:
            pre_trained_model_path = None
        else:
            pre_trained_model_path = LEGACY_PAIRWISE_PRETRAINED_MODEL_PATH
    if is_cliff_model:
        parameter_names, default_mean, default_std = _cliff_parameter_contract(
            apnet_model_type
        )
        if param_start_mean is None:
            param_start_mean = list(default_mean)
        if param_start_std is None:
            param_start_std = list(default_std)
        # Scalars are deliberately *not* broadcast here.  A CLIFF classical
        # contract mixes electrostatic, Thole, overlap, and exchange scales, so
        # one number cannot express the intent unambiguously; the shared
        # validator rejects any non-sequence and any wrong length, reporting the
        # expected count derived from `parameter_names`.
        param_start_mean, param_start_std, _, _ = (
            AtomPairwiseModels.mtp_mtp._validate_positive_initialization(
                parameter_names,
                param_start_mean,
                param_start_std,
                AtomPairwiseModels.mtp_mtp.RACKERS_POSITIVITY_EPSILON,
            )
        )
    elif is_rackers_model:
        if param_start_mean is None:
            param_start_mean = list(RACKERS_PARAM_START_MEAN)
        elif not isinstance(param_start_mean, (list, tuple)) or len(
            param_start_mean
        ) != 4:
            raise ValueError("param_start_mean must contain exactly four values")
        else:
            param_start_mean = list(param_start_mean)
        if param_start_std is None:
            param_start_std = list(RACKERS_PARAM_START_STD)
        elif not isinstance(param_start_std, (list, tuple)) or len(
            param_start_std
        ) != 4:
            raise ValueError("param_start_std must contain exactly four values")
        else:
            param_start_std = list(param_start_std)
        param_start_mean, param_start_std, _, _ = (
            AtomPairwiseModels.mtp_mtp._validate_rackers_initialization(
                param_start_mean,
                param_start_std,
                AtomPairwiseModels.mtp_mtp.RACKERS_POSITIVITY_EPSILON,
            )
        )
    else:
        if param_start_mean is None:
            param_start_mean = 1.5
        if param_start_std is None:
            param_start_std = 0.1
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
    if apnet_model_type != "APNet3-fused-d3" and readout_decay_mode != "legacy-r3":
        raise ValueError("readout_decay_mode is only supported by APNet3-fused-d3")
    if apnet_model_type != "APNet3-fused-d3" and loss_mode != "mse":
        raise ValueError("loss_mode is only supported by APNet3-fused-d3")
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
    elif apnet_model_type == "RackersTholeDampingModel":
        APNet = AtomPairwiseModels.mtp_mtp.RackersTholeDampingModel
    elif apnet_model_type == "RackersTholeDampingOverlapModel":
        APNet = AtomPairwiseModels.mtp_mtp.RackersTholeDampingOverlapModel
    elif is_cliff_model:
        # Each CLIFF identifier is its harness class name, and the class fixes
        # its own dimer mode, model type, and parameter count -- so there is
        # nothing per-route to branch on here.
        APNet = getattr(AtomPairwiseModels.mtp_mtp, apnet_model_type)
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
    if is_positive_param_model:
        # These heads (Rackers, CLIFF) do have a DDP path now, but it is opt-in:
        # `--world_size_ddp`, or an external launcher that reports its own world
        # size. Deliberately *not* `torch.cuda.device_count()` -- a single-GPU
        # chunk that landed on a 2-GPU node must stay a single-GPU chunk, since
        # the chunk chain's epoch budget and its learning rate were measured
        # that way and the effective batch size would silently double.
        world_size = max(int(ddp_world_size or 1), 1)
        if external_rank is not None:
            print(
                f"External DDP rank {external_rank} "
                f"(local {external_local_rank}) of {world_size}",
                flush=True,
            )
    elif torch.cuda.is_available():
        world_size = torch.cuda.device_count()
    else:
        world_size = 1
    print("World Size", world_size)

    omp_num_threads_per_process = omp_num_threads
    if os.path.exists(model_out) and pre_trained_model_path is None:
        pretrained_model = model_out
        print(f"\nTraining from {model_out}\n")
    elif pre_trained_model_path is not None:
        pretrained_model = pre_trained_model_path
        print(f"\nTraining from {pre_trained_model_path}\n")
    else:
        pretrained_model = None
        print("\nTraining from scratch...\n")
    if batch_size is not None:
        # Applied after the per-route defaults above so an explicit request
        # wins over them rather than being overwritten by the branch.
        ds_batch_size = batch_size
    if is_positive_param_model:
        # Two-stage construction shared by the Rackers and CLIFF routes: build
        # the HFVR/valence-width AtomTypeParamModel from --am_model_path plus
        # --atom_type_param_model_path, then wrap its `.model` in the selected
        # harness.  `n_params` is intentionally not forwarded -- every one of
        # these harnesses fixes its own parameter count.
        atom_type_hf_vw_model = AtomPairwiseModels.mtp_mtp.AtomTypeParamModel(
            ds_root=None,
            use_GPU=False,
            ignore_database_null=True,
            atom_model_pre_trained_path=am_model_path,
            pre_trained_model_path=atom_type_param_model_path,
            freeze_atom_model=freeze_atom_model,
        )
        apnet = APNet(
            atom_model=atom_type_hf_vw_model.model,
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
            ds_in_memory=ds_in_memory,
            ds_max_size=ds_max_size,
            ds_max_size_val=ds_max_size_val,
            ds_batch_size=ds_batch_size,
            ds_exclude_elements=ds_exclude_elements,
            ds_exclude_train_indices_path=ds_exclude_train_indices_path,
            param_start_mean=param_start_mean,
            param_start_std=param_start_std,
            elst_damping_type=elst_damping_type,
            freeze_atom_model=freeze_atom_model,
            # Empty for every head without its own architecture, so the call is
            # byte-for-byte what it was for those routes.
            **parameter_head_kwargs,
        )
    elif apnet_model_type.startswith("dAPNet"):
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
            ds_batch_size=ds_batch_size,
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
            readout_decay_mode=readout_decay_mode,
            readout_exchange_scale=readout_exchange_scale,
            readout_induction_scale=readout_induction_scale,
            readout_hybrid_start=readout_hybrid_start,
            readout_hybrid_end=readout_hybrid_end,
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
        # 4 remains the default so every existing route's behaviour is
        # unchanged; a DDP launcher passes its own, because `--cpus-per-task`
        # divided by the tasks per node is the real budget and 4 workers per
        # rank on an 8-core allocation oversubscribes it.
        dataloader_num_workers=(
            4 if dataloader_num_workers is None else int(dataloader_num_workers)
        ),
        random_seed=random_seed,
        include_total_mse=include_total_mse,
        wandb_config=wandb_config,
    )
    if apnet_model_type == "APNet3-fused-d3":
        train_kwargs.update(
            loss_mode=loss_mode,
            huber_delta=huber_delta,
            closest_contact_bin_edges=closest_contact_bin_edges or (),
            closest_contact_bin_counts=closest_contact_bin_counts or (),
        )
    if grad_clip_norm is not None:
        # Only inserted when requested so routes whose `train` has no
        # `grad_clip_norm` parameter do not print a spurious "skipping
        # unsupported kwarg" line on every unclipped run.
        train_kwargs["grad_clip_norm"] = grad_clip_norm
        train_kwargs["grad_clip_mode"] = grad_clip_mode
    if external_rank is not None:
        # Same reasoning: only the externally launched DDP routes see these, so
        # no other route prints an "unsupported kwarg" line.
        train_kwargs["_external_rank"] = external_rank
        train_kwargs["_external_local_rank"] = external_local_rank
    if shard_locality_block_shards:
        # Same reasoning again -- and left out entirely at the default of 0, so
        # a run that does not ask for it produces the identical epoch ordering
        # it always did.  Routes whose `train` has no such parameter drop it
        # through the unsupported-kwarg filter.
        train_kwargs["shard_locality_block_shards"] = int(
            shard_locality_block_shards
        )
    if is_cliff_model:
        # Added only for the CLIFF routes so every other route's train_kwargs
        # stay exactly as they were.  `component_gamma` is forwarded as-is,
        # including its `None` default: `None` is the harness's sentinel for the
        # legacy plain multi-column MSE, and coercing it to a float here would
        # silently switch the run onto the Eq. (23) functional.
        train_kwargs["component_gamma"] = component_gamma
        train_kwargs["total_includes_d3"] = total_includes_d3
        train_kwargs["thole_lr"] = thole_lr
        train_kwargs["trainable_polarizability_scale"] = (
            trainable_polarizability_scale
        )
        train_kwargs["polarizability_lr"] = polarizability_lr
        train_kwargs["atom_model_lr"] = atom_model_lr
        train_kwargs["induction_diagnostics"] = induction_diagnostics
        if induction_convergence_threshold is not None:
            train_kwargs["induction_convergence_threshold"] = (
                induction_convergence_threshold
            )
        if induction_max_iterations is not None:
            train_kwargs["induction_max_iterations"] = induction_max_iterations
        if induction_convergence_norm is not None:
            train_kwargs["induction_convergence_norm"] = (
                induction_convergence_norm
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
        help=(
            "Train an atom model: AtomModel, AtomHirshfeldModel, "
            "AtomTypeParamNN, AtomTypeParamModel, AtomInducedDipoleModel, or "
            "InducedDipoleModel. AtomTypeParamNN is the class the CLIFF and "
            "Rackers routes load via --atom_type_param_model_path; "
            "AtomTypeParamModel is a different, standalone architecture that "
            "predicts the same targets but will not load there."
        ),
    )
    args.add_argument(
        "--train_apnet",
        type=str,
        default="",
        help=(
            "Train APNet model, including RackersTholeDampingModel, "
            "RackersTholeDampingOverlapModel, CliffExchangeModel, "
            "CliffClassicalModel, CliffClassicalOverlapModel, or "
            "CliffClassicalOverlapMPNNModel (plus legacy "
            "APNet2, APNet3-fused variants, dAPNet2, APNet2-fused, and "
            "AM-DimerParam routes)."
        ),
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
        "--ds_max_size_val",
        type=int,
        default=None,
        help=(
            "Separate cap on the validation split of a split dataset. "
            "Unset, --ds_max_size caps both splits, so a 100k-dimer run "
            "evaluates 100k validation dimers every epoch -- as much work as "
            "the training pass. Requires --ds_max_size. Supported on the "
            "Rackers and CLIFF positive-parameter routes."
        ),
    )
    args.add_argument(
        "--param_n_message",
        type=int,
        default=None,
        help=(
            "Message-passing depth of the parameter head "
            "(CliffClassicalOverlapMPNNModel only). Unset, the head's own "
            "default applies."
        ),
    )
    args.add_argument(
        "--param_n_rbf",
        type=int,
        default=None,
        help=(
            "Radial basis size for the parameter head's own distance "
            "expansion (CliffClassicalOverlapMPNNModel only)."
        ),
    )
    args.add_argument(
        "--param_hidden",
        type=int,
        default=None,
        help=(
            "Hidden-state width of the parameter head's message passing "
            "(CliffClassicalOverlapMPNNModel only)."
        ),
    )
    args.add_argument(
        "--param_r_cut",
        type=float,
        default=None,
        help=(
            "Cutoff radius for the parameter head's message passing "
            "(CliffClassicalOverlapMPNNModel only)."
        ),
    )
    args.add_argument(
        "--frozen_parameters",
        type=str,
        nargs="+",
        default=None,
        metavar="NAME",
        help=(
            "Parameter columns to hold at their per-element seed instead of "
            "fitting, e.g. --frozen_parameters thole_direct thole_mutual. "
            "Fitting the Thole parameters per atom makes the induced-dipole "
            "response operator a learned object, whose positive definiteness "
            "is what the interaction induction needs to be attractive. CLIFF "
            "routes only."
        ),
    )
    args.add_argument(
        "--shared_damping_parameters",
        type=str,
        nargs="+",
        default=None,
        metavar="NAME",
        help=(
            "Damping columns to fit as a single global scalar shared by every "
            "atom and every listed column, e.g. "
            "--shared_damping_parameters thole_direct thole_mutual. CLIFF "
            "publishes one smearing coefficient, not one per element, so this "
            "is the arm that fits what CLIFF actually parameterises; the "
            "per-element default is the more ambitious fit. Disjoint from "
            "--frozen_parameters. CLIFF routes only."
        ),
    )
    args.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help=(
            "Dimers per optimizer step for the pairwise routes. Unset, each "
            "route keeps its historical dataset batch size of 16, which "
            "starves a GPU on a model of any size."
        ),
    )
    args.add_argument(
        "--skip_compile",
        action="store_true",
        help=(
            "Run the atom model eager instead of under torch.compile. "
            "AtomTypeParamModel's forward writes into a slice of a "
            "mask-filtered tensor, which Inductor cannot guard on; on some "
            "torch builds that raises GuardOnDataDependentSymNode after the "
            "pre-training evaluation."
        ),
    )
    args.add_argument(
        "--split_manifest",
        type=str,
        default=None,
        help=(
            "Path to a train/test split manifest CSV with columns index, "
            "split, fingerprint. Replaces the uniform random 90/10 draw with "
            "the split the manifest describes. Atom-model routes only."
        ),
    )
    args.add_argument(
        "--split_verify",
        default="all",
        help=(
            "How much of --split_manifest to verify against the dataset: "
            "'all' (default), 'none', or an integer sample count. A stale "
            "manifest silently scrambles the split, so 'none' is a deliberate "
            "choice."
        ),
    )
    args.add_argument(
        "--ds_exclude_elements",
        type=int,
        nargs="+",
        default=None,
        metavar="Z",
        help=(
            "Atomic numbers to exclude, e.g. --ds_exclude_elements 11 17 to "
            "drop every dimer containing Na or Cl. Filtering runs before "
            "--ds_max_size, so --ds_max_size counts surviving dimers. "
            "Supported on the Rackers and CLIFF positive-parameter routes."
        ),
    )
    args.add_argument(
        "--ds_exclude_train_indices_path",
        type=str,
        default=None,
        help=(
            "Sorted unique .npy dataset indices to remove from the capped "
            "training split. Validation is unchanged. Supported on Rackers "
            "and CLIFF positive-parameter routes and mutually exclusive with "
            "--ds_exclude_elements."
        ),
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
        default=None,
        help=(
            "Parameter initialization mean. Unset uses 2.0 for legacy CLI "
            "routes, [1.8, 0.34, 0.39, 1.8] for Rackers routes, [2.5] for "
            "CliffExchangeModel, and [1.8, 0.34, 0.39, 1.8, 2.5] for the "
            "CliffClassical routes. Custom comma-separated values must "
            "contain exactly four values for Rackers routes, exactly one "
            "value for CliffExchangeModel, and exactly five values for the "
            "CliffClassical routes; a bare scalar is rejected on all of them."
        ),
    )
    args.add_argument(
        "--param_start_std",
        type=str,
        default=None,
        help=(
            "Parameter initialization std. Unset uses 0.1 for legacy CLI "
            "routes, [0.01, 0.01, 0.01, 0.01] for Rackers routes, [0.01] for "
            "CliffExchangeModel, and five 0.01 values for the CliffClassical "
            "routes. Custom comma-separated values must contain exactly four "
            "values for Rackers routes, exactly one value for "
            "CliffExchangeModel, and exactly five values for the "
            "CliffClassical routes; a bare scalar is rejected on all of them."
        ),
    )
    args.add_argument(
        "--world_size_ddp",
        type=int,
        default=1,
        help=(
            "world_size for DDP: AtomModels, and the positive-parameter "
            "pairwise routes (Rackers/CLIFF). Ignored when --ddp_srun is "
            "given, which reads the real world size from the launcher "
            "(default: 1)"
        ),
    )
    args.add_argument(
        "--ddp_srun",
        action="store_true",
        help=(
            "This process is one rank of an externally launched job (srun or "
            "torchrun). Rank, local rank, world size and the rendezvous "
            "endpoint are read from the environment; this is the only launch "
            "mode that works across nodes. Opt-in rather than auto-detected so "
            "an ordinary one-task srun job keeps running single-process."
        ),
    )
    args.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=None,
        help=(
            "DataLoader workers per process (default: 4). Under DDP this is "
            "per rank, so the sum across ranks on a node must fit "
            "--cpus-per-task."
        ),
    )
    args.add_argument(
        "--omp_num_threads",
        type=int,
        default=None,
        help=(
            "OpenMP threads per training process "
            "(default: 1 for atom models, 8 for pairwise models)"
        ),
    )
    args.add_argument(
        "--ds_in_memory",
        type=str2bool,
        default=False,
        help=(
            "Load dataset in memory (default: False). Accepts "
            "true/false/yes/no/1/0."
        ),
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
            "in addition to the four component losses. On a CLIFF route this "
            "is shorthand for --component_gamma 0.5 and cannot be combined "
            "with an explicit --component_gamma."
        ),
    )
    args.add_argument(
        "--component_gamma",
        type=float,
        default=None,
        help=(
            "CliffClassicalModel/CliffClassicalOverlapModel only: CLIFF "
            "Eq. (23) weight, L = (1-gamma) MSE(total) + gamma sum_C MSE(E_C). "
            "Unset (the default) keeps the legacy plain multi-column MSE "
            "bitwise unchanged; any value in [0.0, 1.0] selects the Eq. (23) "
            "functional (CLIFF's fitted value is 0.4). Rejected on "
            "CliffExchangeModel and on every pre-existing route."
        ),
    )
    args.add_argument(
        "--grad_clip_norm",
        type=float,
        default=None,
        help=(
            "Clip gradient norms to this value before each optimizer step. "
            "Unset (the default) leaves the update unclipped. SAPT components "
            "reach ~240 kcal/mol, so a single close-contact dimer can otherwise "
            "dominate a step under MSE."
        ),
    )
    args.add_argument(
        "--grad_clip_mode",
        choices=AtomPairwiseModels.mtp_mtp.CLIFF_GRAD_CLIP_MODES,
        default="global",
        help=(
            "Gradient clipping scope. 'global' clips all trainable parameters "
            "together. 'component' clips dense CLIFF ELST, EXCH, and IND "
            "parameter groups independently and requires --grad_clip_norm."
        ),
    )
    args.add_argument(
        "--thole_lr",
        type=float,
        default=None,
        help=(
            "Dense combined CLIFF routes only: use this Adam learning rate for "
            "trainable thole_direct/thole_mutual parameters while every other "
            "parameter uses --lr."
        ),
    )
    args.add_argument(
        "--trainable_polarizability_scale",
        action="store_true",
        help=(
            "Dense combined CLIFF routes only: make the free-atom "
            "polarizability table a trainable per-element scale seeded at "
            "1.0. Requires --polarizability_lr."
        ),
    )
    args.add_argument(
        "--polarizability_lr",
        type=float,
        default=None,
        help=(
            "Adam learning rate for the polarizability scale alone. 0.0 is "
            "legal and freezes it at its seed, which is the control arm."
        ),
    )
    args.add_argument(
        "--atom_model_lr",
        type=float,
        default=None,
        help=(
            "Dense combined CLIFF routes only: Adam learning rate for the "
            "nested atom_model alone when --unfreeze_atom_model trains it. "
            "Required in that case -- the trunk is 1.89M pretrained "
            "parameters against a 231k head, and at the head's rate it "
            "diverges inside one epoch. 0.0 is legal and freezes it in place."
        ),
    )
    args.add_argument(
        "--induction_diagnostics",
        action="store_true",
        default=False,
        help=(
            "Combined CLIFF routes only: log SCF convergence/residual, maximum "
            "induced dipole, and maximum absolute induction edge energy."
        ),
    )
    args.add_argument(
        "--shard_locality_block_shards",
        type=int,
        default=0,
        help=(
            "0 (default) keeps uniform shuffling. Above 0, sample with "
            "shard locality: shuffle shards instead of dimers, hand each "
            "dataloader worker a disjoint block of this many shards, and "
            "size the dataset's shard LRU to match. Each shard is then read "
            "about once per epoch instead of once per dimer drawn from it, "
            "at the cost of a shuffle window of block x shard_size dimers "
            "rather than the whole store. 256 is a reasonable starting "
            "point on the 16-dimer CLIFF2 store (4,096-dimer window, "
            "~54 MB of shard cache per worker)."
        ),
    )
    args.add_argument(
        "--induction_convergence_threshold",
        type=float,
        default=None,
        help=(
            "Combined CLIFF routes only: Rackers/Thole SCF residual stopping "
            "threshold. Unset preserves the checkpoint or historical 1e-8 default."
        ),
    )
    args.add_argument(
        "--induction_max_iterations",
        type=int,
        default=None,
        help=(
            "Combined CLIFF routes only: Rackers/Thole SCF iteration cap. "
            "Unset preserves the checkpoint or historical 200 default."
        ),
    )
    args.add_argument(
        "--induction_convergence_norm",
        type=str,
        default=None,
        choices=["l2", "rms", "max"],
        help=(
            "Combined CLIFF routes only: how the induced-dipole change is "
            "reduced before the threshold test. 'l2' is the historical "
            "unnormalised batch-wide norm, which grows as sqrt(n_atoms) and so "
            "tightens the effective tolerance as the batch grows; 'rms' and "
            "'max' are batch-size independent. Unset preserves the checkpoint "
            "or historical 'l2' default."
        ),
    )
    args.add_argument(
        "--total_includes_d3",
        action="store_true",
        default=False,
        help=(
            "CliffClassicalModel/CliffClassicalOverlapModel only: include D3 "
            "dispersion in the Eq. (23) total term and compare it against all "
            "four SAPT columns. Requires an explicit --component_gamma."
        ),
    )
    args.add_argument(
        "--merge_rackers_checkpoint",
        type=str,
        default=None,
        help=(
            "Stage-two warm start: RackersTholeDampingNN checkpoint whose "
            "elst/thole_direct/thole_mutual/ind_overlap columns are copied "
            "into a CliffClassicalNN checkpoint by parameter name. Requires "
            "--merge_output_path."
        ),
    )
    args.add_argument(
        "--merge_exchange_checkpoint",
        type=str,
        default=None,
        help=(
            "Stage-two warm start: CliffExchangeNN checkpoint whose exch "
            "column is copied into a CliffClassicalNN checkpoint by parameter "
            "name. Requires --merge_output_path."
        ),
    )
    args.add_argument(
        "--merge_output_path",
        type=str,
        default=None,
        help=(
            "Destination CliffClassicalNN checkpoint for "
            "--merge_rackers_checkpoint / --merge_exchange_checkpoint. "
            "Merging runs on its own and exits without training."
        ),
    )
    args.add_argument(
        "--no_disp_nn",
        action="store_true",
        default=False,
        help="APNet3-fused-d3 only: train elst/exch/indu (three components) and compute D3 at predict time instead of a dispersion NN.",
    )
    args.add_argument(
        "--readout_decay_mode",
        choices=AtomPairwiseModels.apnet3_d3_fused.READOUT_DECAY_MODES,
        default="legacy-r3",
        help=(
            "APNet3-fused-d3 readout envelope. 'legacy-r3' preserves current "
            "behavior; 'exchange-overlap' changes exchange only; "
            "'exchange-overlap-induction-r6' additionally changes induction; "
            "'exchange-hybrid-r3-overlap' smoothly joins legacy short range "
            "to the Slater-overlap tail."
        ),
    )
    args.add_argument(
        "--readout_exchange_scale",
        type=float,
        default=1.0,
        help="Fixed positive scale on the Slater exchange envelope (default: 1).",
    )
    args.add_argument(
        "--readout_induction_scale",
        type=float,
        default=1.0,
        help="Fixed positive scale on the inverse-sixth induction envelope (default: 1).",
    )
    args.add_argument(
        "--readout_hybrid_start",
        type=float,
        default=2.5,
        help="Hybrid mode: legacy exchange below this distance in Angstrom (default: 2.5).",
    )
    args.add_argument(
        "--readout_hybrid_end",
        type=float,
        default=3.5,
        help="Hybrid mode: overlap exchange above this distance in Angstrom (default: 3.5).",
    )
    args.add_argument(
        "--loss-mode",
        choices=AtomPairwiseModels.apnet3_d3_fused.LOSS_MODES,
        default="mse",
        help="APNet3-fused-d3 loss: historical MSE, plain Huber, or 10-bin closest-contact macro-Huber.",
    )
    args.add_argument(
        "--huber-delta",
        type=float,
        default=1.0,
        help="Positive Huber transition in kcal/mol (default: 1.0).",
    )
    args.add_argument(
        "--closest-contact-bin-edges",
        type=float,
        nargs=9,
        default=None,
        metavar="ANGSTROM",
        help="Nine frozen closest-contact boundaries for ten-bin macro-Huber.",
    )
    args.add_argument(
        "--closest-contact-bin-counts",
        type=int,
        nargs=10,
        default=None,
        metavar="COUNT",
        help="Ten positive global training counts for macro-Huber weighting.",
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
        help=(
            "Unfreeze the nested atom-type model for APNet3-fused variants, "
            "RackersTholeDampingModel, RackersTholeDampingOverlapModel, "
            "CliffExchangeModel, CliffClassicalModel, or "
            "CliffClassicalOverlapModel (default: frozen)."
        ),
    )
    args.add_argument(
        "--build_dataset_only",
        action="store_true",
        default=False,
        help="Build/process the requested dataset and exit without training.",
    )
    wandb_mode_default = os.getenv("WANDB_MODE", "disabled")
    if wandb_mode_default not in {"disabled", "online", "offline"}:
        wandb_mode_default = "disabled"
    args.add_argument(
        "--wandb-mode",
        choices=("disabled", "online", "offline"),
        default=wandb_mode_default,
    )
    args.add_argument("--wandb-run-id", default=None)
    args.add_argument("--wandb-project", default=None)
    args.add_argument("--wandb-entity", default=None)
    args.add_argument("--wandb-name", default=None)
    args.add_argument("--wandb-group", default=None)
    args.add_argument("--wandb-tags", nargs="*", default=())
    args.add_argument("--wandb-job-type", default=None)
    args.add_argument("--wandb-notes", default=None)
    args.add_argument("--wandb-dir", default=None)
    args.add_argument(
        "--wandb-config-file",
        default=None,
        help="JSON object merged into the structured W&B run config.",
    )
    args = args.parse_args()
    # Parse only explicitly supplied parameter initialization values.
    if args.param_start_mean is not None:
        args.param_start_mean = parse_param_list(args.param_start_mean)
    if args.param_start_std is not None:
        args.param_start_std = parse_param_list(args.param_start_std)
    pprint(args)
    set_all_seeds(args.random_seed)
    merge_requested = (
        args.merge_rackers_checkpoint is not None
        or args.merge_exchange_checkpoint is not None
        or args.merge_output_path is not None
    )
    if merge_requested:
        # Stage-two warm start is a standalone operation: it rewrites
        # checkpoints and exits rather than falling through into training.
        if args.merge_output_path is None:
            raise ValueError(
                "--merge_output_path is required when "
                "--merge_rackers_checkpoint or "
                "--merge_exchange_checkpoint is supplied"
            )
        if (
            args.merge_rackers_checkpoint is None
            and args.merge_exchange_checkpoint is None
        ):
            raise ValueError(
                "--merge_output_path requires at least one of "
                "--merge_rackers_checkpoint or "
                "--merge_exchange_checkpoint"
            )
        from apnet_pt.AtomPairwiseModels.cliff_2 import (
            merge_classical_parameter_checkpoints,
        )

        merge_classical_parameter_checkpoints(
            args.merge_rackers_checkpoint,
            args.merge_exchange_checkpoint,
            args.merge_output_path,
        )
        print(
            f"Merged classical parameter checkpoint written to "
            f"{args.merge_output_path}"
        )
        return
    atom_wandb_config, pairwise_wandb_config = build_wandb_run_configs(args)
    if args.train_am != "":
        # Rejected rather than ignored. The atom routes have their own hard-coded
        # batch size and a single monomer store, so accepting either flag here
        # would leave a run record claiming a shape the run never had.
        for flag, value in (
            ("--batch_size", args.batch_size),
            ("--ds_max_size_val", args.ds_max_size_val),
        ):
            if value is not None:
                raise ValueError(
                    f"{flag} applies to the pairwise routes (--train_apnet), "
                    "not --train_am"
                )
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
            omp_num_threads=(
                args.omp_num_threads
                if args.omp_num_threads is not None
                else 1
            ),
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
            split_manifest=args.split_manifest,
            split_verify=args.split_verify,
            skip_compile=True if args.skip_compile else None,
            am_model_path_for_inner=args.atom_mpnn_pretrained_path,
            freeze_inner_atom_model=not args.unfreeze_atom_model,
            wandb_config=atom_wandb_config,
        )
    if args.train_apnet != "":
        param_start_mean = args.param_start_mean
        param_start_std = args.param_start_std
        if args.train_apnet not in POSITIVE_PARAM_MODEL_TYPES:
            # The unset sentinel only survives for the positive-parameter
            # routes, which resolve it to their own physical defaults; every
            # other route keeps its historical scalar default.
            if param_start_mean is None:
                param_start_mean = 2.0
            if param_start_std is None:
                param_start_std = 0.1
        # Resolve the launch topology once, here, so the rest of the call is
        # identical whether this is a plain single-process run, an in-process
        # `mp.spawn` run (--world_size_ddp on one node) or one rank of an
        # `srun`/`torchrun` job (--ddp_srun, the only mode that spans nodes).
        pairwise_external_rank = None
        pairwise_external_local_rank = None
        pairwise_world_size = args.world_size_ddp
        if args.ddp_srun:
            rendezvous = ddp_launch.export_rendezvous(
                ddp_launch.resolve_rendezvous(),
                omp_num_threads=args.omp_num_threads,
            )
            pairwise_external_rank = rendezvous.rank
            pairwise_external_local_rank = rendezvous.local_rank
            # The launcher is authoritative: `--world_size_ddp` is whatever the
            # sbatch template happened to say, while WORLD_SIZE/SLURM_NTASKS is
            # the number of processes that will actually reach the collectives.
            # Disagreeing with it is the difference between a run and a hang.
            pairwise_world_size = rendezvous.world_size
            if rendezvous.rank == 0:
                print(ddp_launch.describe_rendezvous(rendezvous), flush=True)
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
            param_start_mean=param_start_mean,
            param_start_std=param_start_std,
            dimer_eval_type=args.dimer_eval_type,
            elst_damping_type=args.elst_damping_type,
            ds_in_memory=args.ds_in_memory,
            ds_class_type=args.ds_class_type,
            DimerProp_model_type=args.DimerProp_model_type,
            ap2_pretrained_model_only=args.ap2_pretrained_model_only,
            ds_type=args.ds_type,
            no_disp_nn=args.no_disp_nn,
            use_precomputed_classical=args.use_precomputed_classical,
            readout_decay_mode=args.readout_decay_mode,
            readout_exchange_scale=args.readout_exchange_scale,
            readout_induction_scale=args.readout_induction_scale,
            readout_hybrid_start=args.readout_hybrid_start,
            readout_hybrid_end=args.readout_hybrid_end,
            freeze_dimer_prop_model=not args.unfreeze_dimer_prop_model,
            freeze_atom_model=not args.unfreeze_atom_model,
            build_dataset_only=args.build_dataset_only,
            include_total_mse=args.include_total_mse,
            loss_mode=args.loss_mode,
            huber_delta=args.huber_delta,
            closest_contact_bin_edges=args.closest_contact_bin_edges,
            closest_contact_bin_counts=args.closest_contact_bin_counts,
            ds_max_size=args.ds_max_size,
            ds_max_size_val=args.ds_max_size_val,
            batch_size=args.batch_size,
            param_n_message=args.param_n_message,
            param_n_rbf=args.param_n_rbf,
            param_hidden=args.param_hidden,
            param_r_cut=args.param_r_cut,
            frozen_parameters=args.frozen_parameters,
            shared_damping_parameters=args.shared_damping_parameters,
            ds_exclude_elements=args.ds_exclude_elements,
            ds_exclude_train_indices_path=(
                args.ds_exclude_train_indices_path
            ),
            split_manifest=args.split_manifest,
            component_gamma=args.component_gamma,
            total_includes_d3=args.total_includes_d3,
            grad_clip_norm=args.grad_clip_norm,
            grad_clip_mode=args.grad_clip_mode,
            thole_lr=args.thole_lr,
            trainable_polarizability_scale=(
                args.trainable_polarizability_scale
            ),
            polarizability_lr=args.polarizability_lr,
            atom_model_lr=args.atom_model_lr,
            induction_diagnostics=args.induction_diagnostics,
            induction_convergence_threshold=(
                args.induction_convergence_threshold
            ),
            induction_max_iterations=args.induction_max_iterations,
            induction_convergence_norm=args.induction_convergence_norm,
            shard_locality_block_shards=args.shard_locality_block_shards,
            omp_num_threads=(
                args.omp_num_threads
                if args.omp_num_threads is not None
                else 8
            ),
            ddp_world_size=pairwise_world_size,
            external_rank=pairwise_external_rank,
            external_local_rank=pairwise_external_local_rank,
            dataloader_num_workers=args.dataloader_num_workers,
            wandb_config=pairwise_wandb_config,
        )
    return


if __name__ == "__main__":
    main()
