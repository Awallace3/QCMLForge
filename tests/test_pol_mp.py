import apnet_pt
from apnet_pt import atomic_datasets
from apnet_pt import AtomModels
import os
import numpy as np
import pytest
from glob import glob
import qcelemental as qcel
import torch
import pandas as pd
from pprint import pprint as pp

# Handle both pytest and direct execution
try:
    from . import mols
except ImportError:
    import mols


torch.manual_seed(42)
spec_type = 5
current_file_path = os.path.dirname(os.path.realpath(__file__))
data_path = f"{current_file_path}/test_data_path"
am_path = f"{current_file_path}/../src/apnet_pt/models/am_ensemble/am_0.pt"

am_path = f"{current_file_path}/test_models/ap3_ensemble_0/am_3.pt"
atp_path = f"{current_file_path}/test_models/ap3_ensemble_0/atp_mpnn_1.pt"
# aidm_path = f"{current_file_path}/test_models/ap3_ensemble_0/atomInducedDipole_atp_screeningNN_1.pt"
aidm_path = f"{current_file_path}/test_models/ap3_ensemble_0/atomInducedDipole_atp_screeningNN_lr_1.pt"


def test_train_ap3_atomTypeparamMPNN():
    """
    Builds a small Hirshfeld atomic dataset and trains an AP3 AtomTypeParamModel for a short smoke test.
    
    Creates an in-memory atomic Hirshfeld dataset from the test data path, configures threading, instantiates an AP3 AtomTypeParamModel, and runs a 3-epoch training session to validate the training loop and integration with the dataset and model configuration.
    """
    ds = atomic_datasets.atomic_hirshfeld_module_dataset(
        root=data_path,
        transform=None,
        pre_transform=None,
        r_cut=5.0,
        testing=False,
        spec_type=5,
        max_size=None,
        force_reprocess=False,
        in_memory=True,
        batch_size=16,
    )
    print(ds)
    # DDP
    os.environ["OMP_NUM_THREADS"] = "4"
    atpm = AtomModels.ap3_atomtype_mpnn.AtomTypeParamModel(
        use_GPU=False,
        ignore_database_null=False,
        dataset=ds,
    )
    atpm.train(
        n_epochs=3,
        batch_size=16,
        lr=5e-4,
        split_percent=0.5,
        model_path=None,
        shuffle=True,
        skip_compile=True,
        dataloader_num_workers=0,
        world_size=1,
        omp_num_threads_per_process=4,
        random_seed=42,
    )
    return


def test_train_ap3_atom_model():
    """
    Train an AtomInducedDipoleModel on a small in-memory test dataset using a pretrained AtomTypeParamModel.
    
    Builds an in-memory atomic dataset from the repository test data, instantiates an AtomTypeParamModel from a pretrained checkpoint, constructs an AtomInducedDipoleModel that wraps the atom-type model, and runs a short training session intended for integration/testing. The training runs for three epochs with a learning rate of 5e-4 and a batch size of 1; it configures OpenMP threads and single-process (non-DDP) execution.
    """
    ds = atomic_datasets.atomic_module_dataset(
        root=data_path,
        transform=None,
        pre_transform=None,
        r_cut=5.0,
        testing=False,
        spec_type=6,
        max_size=None,
        force_reprocess=False,
        in_memory=True,
        batch_size=1,
    )
    print(ds)
    atpm = AtomModels.ap3_atomtype_mpnn.AtomTypeParamModel(
        use_GPU=False,
        ignore_database_null=False,
        dataset=ds,
        pre_trained_model_path=atp_path,
    )
    print(atpm)
    # DDP
    os.environ["OMP_NUM_THREADS"] = "4"
    am = AtomModels.ap3_atom_model.AtomInducedDipoleModel(
        atomtype_hfvr_model=atpm.model,
        use_GPU=False,
        ignore_database_null=False,
        dataset=ds,
    )
    am.train(
        n_epochs=3,
        batch_size=1,
        lr=5e-4,
        split_percent=0.5,
        model_path=None,
        shuffle=True,
        skip_compile=True,
        dataloader_num_workers=0,
        world_size=1,
        omp_num_threads_per_process=4,
        random_seed=42,
    )
    return


def test_inference_ap3_atom_model():
    """
    Run inference with the AP3 Atom Induced Dipole model on sample dimers and validate outputs against a saved reference.
    
    This test loads a pretrained AtomTypeParamModel and AtomInducedDipoleModel (with neural-network screening enabled), compiles the model, predicts properties for two water-dimer molecules, and asserts element-wise closeness to a previously saved reference tensor within an absolute tolerance of 1e-6.
    """
    atpm = AtomModels.ap3_atomtype_mpnn.AtomTypeParamModel(
        use_GPU=False,
        ignore_database_null=True,
        pre_trained_model_path=atp_path,
    )
    am = AtomModels.ap3_atom_model.AtomInducedDipoleModel(
        atomtype_hfvr_model=atpm.model,
        use_GPU=False,
        ignore_database_null=True,
        pre_trained_model_path=aidm_path,
        use_nn_screening=True,
    )
    am.compile_model()
    v = am.predict_qcel_mols(
        mols=[
            mols.lr_water_dimer,
            mols.lr_water_dimer,
            # mols.mol_cliff_water_close,
        ],
        batch_size=2,
    )
    print(f"{v = }")
    # torch save v for reference
    # torch.save(v, f"{current_file_path}/test_data_path/ap3_atom_model_inference.pt")
    ref = torch.load(f"{current_file_path}/test_data_path/ap3_atom_model_inference.pt")
    for i in range(len(v[0])):
        assert torch.allclose(v[0][i], ref[0][i], atol=1e-6), (
            f"{i}, {v[0][i]}, {ref[0][i]}"
        )
    return


def train_ap3_atom_model():
    """
    Train an AP3 Atom Induced Dipole model using a local dimer dataset and a pretrained atom-type MPNN.
    
    Builds an atomic dataset from tests data (spec_type 10, r_cut 5.0, in-memory) and initializes an AtomInducedDipoleModel with a pretrained AtomTypeParam MPNN. Trains the induced-dipole model (100 epochs, lr=5e-4, batch_size=16) and saves the trained weights to the repository models path.
    
    Note: This function has side effects (training, file I/O, environment variable changes) and does not return a value.
    """
    ds = atomic_datasets.atomic_module_dataset(
        root=f"{current_file_path}/../data_dimer_1",
        transform=None,
        pre_transform=None,
        r_cut=5.0,
        testing=False,
        spec_type=10,
        max_size=None,
        force_reprocess=False,
        in_memory=True,
        batch_size=16,
    )
    torch.set_printoptions(profile="full")
    print(ds)
    atpm = AtomModels.ap3_atomtype_mpnn.AtomTypeParamModel(
        use_GPU=False,
        ignore_database_null=False,
        dataset=ds,
        pre_trained_model_path=atp_path,
    )
    print(atpm)
    # DDP
    os.environ["OMP_NUM_THREADS"] = "4"
    am = AtomModels.ap3_atom_model.AtomInducedDipoleModel(
        atomtype_hfvr_model=atpm.model,
        use_GPU=False,
        ignore_database_null=False,
        dataset=ds,
    )
    am.train(
        n_epochs=100,
        batch_size=16,
        lr=5e-4,
        split_percent=0.9,
        shuffle=True,
        skip_compile=True,
        dataloader_num_workers=0,
        world_size=1,
        omp_num_threads_per_process=4,
        random_seed=42,
        model_path=f"{current_file_path}/..models/ap3_ensemble/ap3_am_spec10.pt",
    )
    return


def test_lmdb_dataset_creation():
    """Test LMDB dataset creation and basic functionality"""
    # Load pre-trained atomtype model
    atpm = AtomModels.ap3_atomtype_mpnn.AtomTypeParamModel(
        use_GPU=False,
        ignore_database_null=True,
        pre_trained_model_path=atp_path,
    )

    # Create LMDB dataset
    from apnet_pt.atomic_datasets import atomic_module_dataset_lmdb

    ds_lmdb = atomic_module_dataset_lmdb(
        root=data_path,
        atomtype_hfvr_model=atpm.model,
        testing=False,
        spec_type=5,
        max_size=50,  # Small for testing
        force_reprocess=False,
        in_memory=False,
        cache_size=10,
    )

    print(f"LMDB dataset length: {len(ds_lmdb)}")
    assert len(ds_lmdb) > 0, "Dataset should have items"

    # Test item retrieval
    item = ds_lmdb[0]
    assert hasattr(item, "x"), "Item should have node features"
    assert hasattr(item, "edge_index"), "Item should have edges"
    assert hasattr(item, "volume_ratios"), "Item should have volume ratios"
    assert hasattr(item, "valence_widths"), "Item should have valence widths"

    print("LMDB dataset tests passed!")
    return


def test_train_with_lmdb():
    """
    Train an AtomInducedDipoleModel using an LMDB-backed dataset.
    
    Loads a pretrained AtomTypeParamModel, constructs an AtomInducedDipoleModel configured to use an LMDB dataset with HFVR precomputation, and runs a short single-process training session for integration testing.
    """
    # Load pre-trained atomtype model
    atpm = AtomModels.ap3_atomtype_mpnn.AtomTypeParamModel(
        use_GPU=False,
        ignore_database_null=True,
        pre_trained_model_path=atp_path,
    )

    # Set up environment
    os.environ["OMP_NUM_THREADS"] = "4"

    # Create AtomInducedDipoleModel with LMDB dataset
    am = AtomModels.ap3_atom_model.AtomInducedDipoleModel(
        atomtype_hfvr_model=atpm.model,
        use_GPU=False,
        ignore_database_null=False,
        ds_root=data_path,
        ds_spec_type=5,
        ds_max_size=50,  # Small for testing
        ds_use_lmdb=True,  # KEY: Use LMDB dataset
        ds_in_memory=False,  # Don't load all into memory
        precompute_hfvr=True,  # Pre-compute hfvr/vw during dataset processing
    )

    # Train (single process)
    am.train(
        n_epochs=2,
        batch_size=2,
        lr=5e-4,
        split_percent=0.5,
        model_path=None,
        shuffle=True,
        skip_compile=True,
        dataloader_num_workers=0,
        world_size=1,  # Single process
        omp_num_threads_per_process=4,
        random_seed=42,
    )
    print("Training with LMDB dataset completed successfully!")
    return


def test_ddp_train_ap3_atom_model():
    """Test distributed data parallel training with world_size=2"""
    # Load pre-trained atomtype model
    atpm = AtomModels.ap3_atomtype_mpnn.AtomTypeParamModel(
        use_GPU=False,
        ignore_database_null=True,
        pre_trained_model_path=atp_path,
    )

    # Set up DDP environment
    os.environ["OMP_NUM_THREADS"] = "2"

    # Create AtomInducedDipoleModel with LMDB dataset
    am = AtomModels.ap3_atom_model.AtomInducedDipoleModel(
        atomtype_hfvr_model=atpm.model,
        use_GPU=False,
        ignore_database_null=False,
        ds_root=data_path,
        ds_spec_type=5,
        ds_max_size=50,  # Small for testing
        ds_use_lmdb=True,  # KEY: Use LMDB dataset
        ds_in_memory=False,  # Don't load all into memory
        precompute_hfvr=True,  # Pre-compute hfvr/vw during dataset processing
    )

    # Train with DDP (world_size=2)
    am.train(
        n_epochs=2,
        batch_size=2,
        lr=5e-4,
        split_percent=0.5,
        model_path=None,
        shuffle=True,
        skip_compile=True,
        dataloader_num_workers=0,
        world_size=2,  # KEY: DDP with 2 processes
        omp_num_threads_per_process=2,
        random_seed=42,
    )
    return


def debug_ap3_atom_model():
    """
    Builds debug AtomTypeParam and frozen InducedDipole models and runs a forward pass on a saved debug batch.
    
    Constructs an AtomTypeParamModel and an InducedDipoleModel using the configured pretrained weights and environment, loads the debug batch tensor from ../debug_batch.pt, performs a forward pass, and prints the instantiated AtomTypeParamModel and the forward output.
    """
    atpm = AtomModels.ap3_atomtype_mpnn.AtomTypeParamModel(
        use_GPU=False,
        ignore_database_null=True,
        dataset=None,
        pre_trained_model_path=atp_path,
    )
    print(atpm)
    # DDP
    os.environ["OMP_NUM_THREADS"] = "4"
    idm = AtomModels.ap3_atom_model_frozen.InducedDipoleModel(
        atomtype_hfvr_model=atpm.model,
        atom_mpnn_pre_trained_path='./models/spice/am_3.pt',
        use_GPU=False,
        ignore_database_null=True,
        dataset=None,
    )
    v = idm.model(torch.load(f"{current_file_path}/../debug_batch.pt", weights_only=False))
    print(f"{v = }")
    return


def test_mtp_elst_dimers():
    atpm = AtomModels.ap3_atomtype_mpnn.AtomTypeParamModel(
        use_GPU=False,
        ignore_database_null=True,
        pre_trained_model_path=atp_path,
    )
    am = AtomModels.ap3_atom_model.AtomInducedDipoleModel(
        atomtype_hfvr_model=atpm.model,
        use_GPU=False,
        ignore_database_null=True,
        pre_trained_model_path=aidm_path,
        use_nn_screening=True,
    )
    ms = [
        mols.lr_water_dimer,
        mols.mol_cliff_water_close,
    ]
    E_elst, E_elst_dimer, E_indu = am.predict_elst_ind_dimer(
        ms,
        batch_size=2,
    )
    for i, m in enumerate(ms):
        elst = E_elst[i]
        elst_dimer = E_elst_dimer[i]
        indu = E_indu[i]
        print(f"mon   {i} elst: {elst:.4f} kcal/mol")
        print(f"Dimer {i} elst: {elst_dimer:.4f} kcal/mol")
        print(f"Dimer {i} indu: {indu:.4f} kcal/mol")
    return


def test_train_frozen_dipole_with_pretrained_atom_mpnn():
    """
    Test training InducedDipoleModel with a pretrained AtomMPNN model.
    Only dipole_update_layers and dipole_readout_layers should be trainable.
    """
    # Load dataset
    ds = atomic_datasets.atomic_module_dataset(
        root=data_path,
        transform=None,
        pre_transform=None,
        r_cut=5.0,
        testing=False,
        spec_type=6,  # Use spec_type 6 which is valid for atomic_module_dataset
        max_size=None,
        force_reprocess=False,
        in_memory=True,
        batch_size=16,
    )
    print(f"Loaded dataset with {len(ds)} samples")

    # Load pretrained atomtype model for HFVR
    atpm = AtomModels.ap3_atomtype_mpnn.AtomTypeParamModel(
        use_GPU=False,
        ignore_database_null=True,
        pre_trained_model_path=atp_path,
    )
    print("Loaded pretrained AtomTypeParamMPNN model for HFVR")

    # Set up environment
    os.environ["OMP_NUM_THREADS"] = "4"

    # Create InducedDipoleModel with a pretrained AtomMPNN model
    # For testing, we'll use the am_path which contains a pretrained AtomMPNN
    am = AtomModels.ap3_atom_model_frozen.InducedDipoleModel(
        atomtype_hfvr_model=atpm.model,
        atom_mpnn_pre_trained_path=am_path,  # Load pretrained AtomMPNN
        use_GPU=False,
        ignore_database_null=False,
        dataset=ds,
    )
    print("Created InducedDipoleModel with pretrained AtomMPNN")

    # Verify that only dipole layers are trainable
    trainable_params = []
    frozen_params = []
    for name, param in am.model.named_parameters():
        if param.requires_grad:
            trainable_params.append(name)
        else:
            frozen_params.append(name)

    print(f"\nTrainable parameters ({len(trainable_params)}):")
    for name in trainable_params:
        print(f"  {name}")

    print(f"\nFrozen parameters ({len(frozen_params)}):")
    for name in frozen_params[:10]:  # Print first 10
        print(f"  {name}")
    if len(frozen_params) > 10:
        print(f"  ... and {len(frozen_params) - 10} more")

    # Verify that dipole layers are trainable and others are frozen
    assert any("dipole_update_layers" in name for name in trainable_params), (
        "dipole_update_layers should be trainable"
    )
    assert any("dipole_readout_layers" in name for name in trainable_params), (
        "dipole_readout_layers should be trainable"
    )
    assert any("charge_update_layers" in name for name in frozen_params), (
        "charge_update_layers should be frozen"
    )
    assert any("charge_readout_layers" in name for name in frozen_params), (
        "charge_readout_layers should be frozen"
    )
    assert any("qpole1_update_layers" in name for name in frozen_params), (
        "qpole1_update_layers should be frozen"
    )
    assert any("qpole2_update_layers" in name for name in frozen_params), (
        "qpole2_update_layers should be frozen"
    )
    assert any("qpole_readout_layers" in name for name in frozen_params), (
        "qpole_readout_layers should be frozen"
    )

    print("\n✓ Layer freezing verification passed!")

    # Train for a few epochs
    print("\nStarting training...")
    am.train(
        n_epochs=2,
        batch_size=8,
        lr=5e-4,
        split_percent=0.5,
        model_path=None,
        shuffle=True,
        skip_compile=True,
        dataloader_num_workers=0,
        world_size=1,
        omp_num_threads_per_process=4,
        random_seed=42,
    )

    print("\n✓ Training completed successfully!")
    print(
        "\nTest passed: Frozen AtomMPNN layers with trainable dipole layers working correctly"
    )
    return


def test_save_load_induced_dipole_model_with_atom_mpnn():
    """
    Test that InducedDipoleModel correctly saves and loads AtomMPNN weights.

    This test verifies:
    1. Create InducedDipoleModel with pretrained AtomMPNN
    2. Train for 1 epoch and save
    3. Load saved model WITHOUT specifying atom_mpnn_pre_trained_path
    4. Verify loaded model has AtomMPNN with correct weights
    """
    import tempfile
    import os

    # Get test data
    ds = atomic_datasets.atomic_module_dataset(
        "./tests/test_data_path", spec_type=6, testing=False, in_memory=True
    )

    # Paths to pretrained models
    atp_path = "./tests/test_models/ap3_ensemble_0/atp_mpnn_1.pt"
    am_path = "./tests/test_models/ap3_ensemble_0/am_3.pt"

    # Load pretrained atomtype model for HFVR
    atpm = AtomModels.ap3_atomtype_mpnn.AtomTypeParamModel(
        use_GPU=False,
        ignore_database_null=True,
        pre_trained_model_path=atp_path,
    )

    # Create temporary file for saving
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp_file:
        temp_model_path = tmp_file.name

    try:
        print(
            "\n=== Step 1: Create and train InducedDipoleModel with pretrained AtomMPNN ==="
        )
        am1 = AtomModels.ap3_atom_model_frozen.InducedDipoleModel(
            atomtype_hfvr_model=atpm.model,
            atom_mpnn_pre_trained_path=am_path,  # Load pretrained AtomMPNN
            use_GPU=False,
            ignore_database_null=False,
            dataset=ds,
        )

        # Get a sample of weights from the AtomMPNN before training
        sample_weight_before = (
            am1.model.atom_mpnn_model.embed_layer.weight.clone().detach()
        )
        print(f"Sample weight before training: {sample_weight_before[0, :5]}")

        # Train for 1 epoch (this will save the model)
        am1.train(
            n_epochs=1,
            batch_size=8,
            lr=5e-4,
            split_percent=0.5,
            model_path=temp_model_path,
            shuffle=False,
            skip_compile=True,
            dataloader_num_workers=0,
            world_size=1,
            omp_num_threads_per_process=4,
            random_seed=42,
        )
        print(f"✓ Model saved to {temp_model_path}")

        # Get weight after training (should be same since frozen)
        sample_weight_after = (
            am1.model.atom_mpnn_model.embed_layer.weight.clone().detach()
        )
        print(f"Sample weight after training: {sample_weight_after[0, :5]}")

        # Verify frozen layers didn't change
        assert torch.allclose(sample_weight_before, sample_weight_after), (
            "Frozen layers should not change during training"
        )

        print(
            "\n=== Step 2: Load model WITHOUT specifying atom_mpnn_pre_trained_path ==="
        )
        # Load the model back WITHOUT specifying atom_mpnn_pre_trained_path
        am2 = AtomModels.ap3_atom_model_frozen.InducedDipoleModel(
            # atomtype_hfvr_model=atpm.model,
            # NOTE: No atom_mpnn_pre_trained_path here!
            use_GPU=False,
            ignore_database_null=False,
            dataset=ds,
            pre_trained_model_path=temp_model_path,  # Load saved model
        )

        print("\n=== Step 3: Verify loaded model has AtomMPNN ===")
        # Verify that atom_mpnn_model exists
        assert am2.model.atom_mpnn_model is not None, (
            "Loaded model should have atom_mpnn_model"
        )
        print("✓ atom_mpnn_model exists in loaded model")

        # Verify weights match
        sample_weight_loaded = (
            am2.model.atom_mpnn_model.embed_layer.weight.clone().detach()
        )
        print(f"Sample weight from loaded model: {sample_weight_loaded[0, :5]}")

        assert torch.allclose(sample_weight_before, sample_weight_loaded), (
            "Loaded model weights should match original"
        )
        print("✓ Weights match!")

        # Verify freezing is still correct
        trainable_count = sum(1 for p in am2.model.parameters() if p.requires_grad)
        frozen_count = sum(1 for p in am2.model.parameters() if not p.requires_grad)
        print(
            f"✓ Loaded model has {trainable_count} trainable and {frozen_count} frozen parameters"
        )

        assert trainable_count == 30, (
            f"Expected 30 trainable params, got {trainable_count}"
        )
        assert frozen_count == 238, f"Expected 238 frozen params, got {frozen_count}"

        print(
            "\n✓✓✓ Test passed: InducedDipoleModel correctly saves and loads AtomMPNN!"
        )

    finally:
        # Clean up temporary file
        if os.path.exists(temp_model_path):
            os.remove(temp_model_path)
            print(f"Cleaned up {temp_model_path}")


if __name__ == "__main__":
    # test_train_ap3_atomTypeparamMPNN()
    # test_train_ap3_atom_model()
    # train_ap3_atom_model()
    debug_ap3_atom_model()
    # test_inference_ap3_atom_model()
    # test_ddp_train_ap3_atom_model()
    # test_mtp_elst_dimers()
    # test_train_frozen_dipole_with_pretrained_atom_mpnn()
    # test_save_load_induced_dipole_model_with_atom_mpnn()
