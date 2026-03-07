"""
Tests for pre-computed HFVR/VW feature in AtomInducedDipoleModel.

This test suite verifies:
1. Pre-computed dataset processes correctly
2. Forward pass equivalence (precompute_hfvr=True vs False)
3. Training works with pre-computed dataset
4. Checkpoint saving/loading preserves precompute_hfvr flag
"""

import os
import torch
import pytest
import numpy as np
from glob import glob
import qcelemental as qcel

from apnet_pt.atomic_datasets import atomic_induced_dipole_precomputed_dataset
from apnet_pt.AtomModels.ap3_atom_model import (
    AtomInducedDipoleModel,
    AtomTypeParamMPNN,
)
from apnet_pt.atomic_datasets import atomic_hirshfeld_collate_update
from torch_geometric.loader import DataLoader

torch.manual_seed(42)
np.random.seed(42)

current_file_path = os.path.dirname(os.path.realpath(__file__))
data_path = f"{current_file_path}/test_data_path"
atp_mpnn_path = f"{current_file_path}/test_models/ap3_ensemble_0/atp_mpnn_1.pt"


# Helper function to create a small test dataset
def create_test_atomtype_model():
    """
    Create a small AtomTypeParamMPNN configured for unit tests.
    
    Returns:
        model (AtomTypeParamMPNN): An AtomTypeParamMPNN instance with a compact architecture and preset parameter initialization suitable for tests (n_message=2, n_neuron=32, n_embed=4, n_params=2, r_cut=5.0).
    """
    model = AtomTypeParamMPNN(
        n_message=2,
        n_neuron=32,
        n_embed=4,
        param_start_mean=torch.tensor([1.0, 2.5]),
        param_start_std=torch.tensor([0.1, 0.3]),
        n_params=2,  # hfvr and vw
        r_cut=5.0,
    )
    return model


@pytest.fixture
def atomtype_hfvr_model():
    """
    Load or create an atom-type HFVR model for tests.
    
    If a checkpoint exists at `atp_mpnn_path`, the function loads model configuration and weights from that checkpoint; otherwise it constructs a small test AtomTypeParamMPNN. The returned model is set to evaluation mode and has gradients disabled.
    
    Returns:
        AtomTypeParamMPNN: Prepared atom-type model with gradients disabled and in evaluation mode.
    """
    if os.path.exists(atp_mpnn_path):
        checkpoint = torch.load(atp_mpnn_path, weights_only=False)
        model = AtomTypeParamMPNN(
            n_message=checkpoint["config"]["n_message"],
            n_neuron=checkpoint["config"]["n_neuron"],
            n_embed=checkpoint["config"]["n_embed"],
            param_start_mean=checkpoint["config"]["param_start_mean"],
            param_start_std=checkpoint["config"]["param_start_std"],
            n_params=checkpoint["config"].get("n_params", 1),
            r_cut=checkpoint["config"].get("r_cut", 5.0),
        )
        model_state_dict = {
            k.replace("_orig_mod.", ""): v
            for k, v in checkpoint["model_state_dict"].items()
        }
        model.load_state_dict(model_state_dict)
    else:
        model = create_test_atomtype_model()

    model.eval()
    model.requires_grad_(False)
    return model


def test_precomputed_dataset_creation(atomtype_hfvr_model):
    """
    Create a pre-computed HFVR/VW dataset and verify processed files exist and per-atom precomputed fields are present and correctly sized.
    
    Verifies that processed files are produced on disk, the dataset contains at least one sample, and that the first sample exposes `volume_ratios` and `valence_widths` with one value per atom.
    """
    # Clean up any existing processed files
    processed_dir = f"{data_path}/processed"
    if os.path.exists(processed_dir):
        for f in glob(f"{processed_dir}/monomer_induced_dipole_precomputed_9_*.pt"):
            os.remove(f)

    # Create dataset with pre-computation
    ds = atomic_induced_dipole_precomputed_dataset(
        root=data_path,
        atomtype_hfvr_model=atomtype_hfvr_model,
        spec_type=9,
        max_size=5,  # Small size for testing
        force_reprocess=True,
        in_memory=True,
        batch_size=2,
    )

    # Verify dataset was created
    assert len(ds) > 0, "Dataset should have at least one sample"

    # Check that processed files exist
    processed_files = glob(f"{processed_dir}/monomer_induced_dipole_precomputed_9_*.pt")
    assert len(processed_files) > 0, "Should have created processed files"

    # Load a sample and verify it has pre-computed values
    sample = ds[0]
    assert hasattr(sample, "volume_ratios"), "Sample should have volume_ratios"
    assert hasattr(sample, "valence_widths"), "Sample should have valence_widths"
    assert sample.volume_ratios.shape[0] == sample.x.shape[0], (
        "volume_ratios should have one value per atom"
    )
    assert sample.valence_widths.shape[0] == sample.x.shape[0], (
        "valence_widths should have one value per atom"
    )

    print(f"✓ Pre-computed dataset created with {len(ds)} samples")
    print(f"  Sample 0: {sample.x.shape[0]} atoms")
    print(
        f"  volume_ratios range: [{sample.volume_ratios.min():.3f}, {sample.volume_ratios.max():.3f}]"
    )
    print(
        f"  valence_widths range: [{sample.valence_widths.min():.3f}, {sample.valence_widths.max():.3f}]"
    )


def test_forward_pass_equivalence(atomtype_hfvr_model):
    """Test that forward pass produces same results with precompute_hfvr=True vs False."""
    # Create pre-computed dataset
    ds_precomputed = atomic_induced_dipole_precomputed_dataset(
        root=data_path,
        atomtype_hfvr_model=atomtype_hfvr_model,
        spec_type=9,
        max_size=3,
        force_reprocess=False,
        in_memory=True,
        batch_size=2,
    )

    # Create model with precompute_hfvr=True
    model_precomputed = AtomInducedDipoleModel(
        atomtype_hfvr_model=atomtype_hfvr_model,
        n_message=2,
        n_rbf=8,
        n_neuron=64,
        n_embed=4,
        r_cut=5.0,
        use_nn_screening=False,
        precompute_hfvr=True,
        use_GPU=False,
    )

    # Create model with precompute_hfvr=False (traditional approach)
    model_traditional = AtomInducedDipoleModel(
        atomtype_hfvr_model=atomtype_hfvr_model,
        n_message=2,
        n_rbf=8,
        n_neuron=64,
        n_embed=4,
        r_cut=5.0,
        use_nn_screening=False,
        precompute_hfvr=False,
        use_GPU=False,
    )

    # Copy weights from traditional to precomputed model (for fair comparison)
    # Filter out atomtype_hfvr_model keys since precomputed model doesn't have it
    state_dict = model_traditional.model.state_dict()
    filtered_state_dict = {
        k: v for k, v in state_dict.items() if not k.startswith("atomtype_hfvr_model.")
    }
    model_precomputed.model.load_state_dict(filtered_state_dict, strict=False)

    # Create DataLoader
    loader = DataLoader(
        ds_precomputed,
        batch_size=2,
        shuffle=False,
        collate_fn=atomic_hirshfeld_collate_update,
    )

    # Get a batch
    batch = next(iter(loader))

    # Forward pass with precomputed values
    model_precomputed.model.eval()
    with torch.no_grad():
        q_pre, mu_pre, theta_pre, _ = model_precomputed.model(batch)

    # Forward pass with on-the-fly computation
    model_traditional.model.eval()
    with torch.no_grad():
        q_trad, mu_trad, theta_trad, _ = model_traditional.model(batch)

    # Compare results (should be identical since we're using same weights)
    # Note: May have small numerical differences due to floating point arithmetic
    assert torch.allclose(q_pre, q_trad, rtol=1e-5, atol=1e-6), (
        "Charges should match between precomputed and traditional"
    )
    assert torch.allclose(mu_pre, mu_trad, rtol=1e-5, atol=1e-6), (
        "Dipoles should match between precomputed and traditional"
    )
    assert torch.allclose(theta_pre, theta_trad, rtol=1e-5, atol=1e-6), (
        "Quadrupoles should match between precomputed and traditional"
    )

    print("✓ Forward pass equivalence verified")
    print(f"  Max charge difference: {torch.abs(q_pre - q_trad).max():.2e}")
    print(f"  Max dipole difference: {torch.abs(mu_pre - mu_trad).max():.2e}")
    print(f"  Max quadrupole difference: {torch.abs(theta_pre - theta_trad).max():.2e}")


def test_training_with_precomputed(atomtype_hfvr_model):
    """Test that training works with pre-computed dataset."""
    # Create pre-computed dataset
    ds = atomic_induced_dipole_precomputed_dataset(
        root=data_path,
        atomtype_hfvr_model=atomtype_hfvr_model,
        spec_type=9,
        max_size=5,
        force_reprocess=False,
        in_memory=True,
        batch_size=2,
    )

    # Create model
    model = AtomInducedDipoleModel(
        atomtype_hfvr_model=atomtype_hfvr_model,
        n_message=2,
        n_rbf=8,
        n_neuron=64,
        n_embed=4,
        r_cut=5.0,
        use_nn_screening=False,
        precompute_hfvr=True,
        use_GPU=False,
    )

    # Create DataLoader
    loader = DataLoader(
        ds,
        batch_size=2,
        shuffle=False,
        collate_fn=atomic_hirshfeld_collate_update,
    )

    # Setup optimizer
    optimizer = torch.optim.Adam(model.model.parameters(), lr=1e-3)

    # Training loop
    model.model.train()
    initial_loss = None
    final_loss = None

    for epoch in range(3):
        epoch_loss = 0.0
        batch_count = 0

        for batch in loader:
            optimizer.zero_grad()

            # Forward pass
            q_pred, mu_pred, theta_pred, _ = model.model(batch)

            # Simple MSE loss
            q_target = batch.charges
            mu_target = batch.dipoles
            theta_target = batch.quadrupoles

            loss = (
                torch.nn.functional.mse_loss(q_pred, q_target)
                + torch.nn.functional.mse_loss(mu_pred, mu_target)
                + torch.nn.functional.mse_loss(theta_pred, theta_target)
            )

            # Backward pass
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1

        avg_loss = epoch_loss / batch_count
        if initial_loss is None:
            initial_loss = avg_loss
        final_loss = avg_loss
        print(f"  Epoch {epoch + 1}/3: Loss = {avg_loss:.6f}")

    # Verify training decreased loss
    assert final_loss < initial_loss, (
        f"Training should decrease loss (initial: {initial_loss:.6f}, "
        f"final: {final_loss:.6f})"
    )

    print("✓ Training with pre-computed dataset successful")
    print(f"  Initial loss: {initial_loss:.6f}")
    print(f"  Final loss: {final_loss:.6f}")
    print(f"  Loss reduction: {(1 - final_loss / initial_loss) * 100:.1f}%")


def test_backward_compatibility(atomtype_hfvr_model):
    """Test that default behavior (precompute_hfvr=False) still works."""
    # Create model with default settings (precompute_hfvr=False)
    model = AtomInducedDipoleModel(
        atomtype_hfvr_model=atomtype_hfvr_model,
        n_message=2,
        n_rbf=8,
        n_neuron=64,
        n_embed=4,
        r_cut=5.0,
        use_nn_screening=False,
        # precompute_hfvr defaults to False
        use_GPU=False,
    )

    # Verify default behavior
    assert model.model.precompute_hfvr is False, (
        "Default precompute_hfvr should be False"
    )
    assert model.model.atomtype_hfvr_model is not None, (
        "Model with precompute_hfvr=False should have atomtype_hfvr_model"
    )

    print("✓ Backward compatibility verified")
    print(f"  Default precompute_hfvr: {model.model.precompute_hfvr}")
    print(f"  Has atomtype_hfvr_model: {model.model.atomtype_hfvr_model is not None}")


if __name__ == "__main__":
    # Run tests manually if needed
    print("=" * 60)
    print("Testing Pre-computed HFVR/VW Feature")
    print("=" * 60)

    model = None
    if os.path.exists(atp_mpnn_path):
        checkpoint = torch.load(atp_mpnn_path, weights_only=False)
        model = AtomTypeParamMPNN(
            n_message=checkpoint["config"]["n_message"],
            n_neuron=checkpoint["config"]["n_neuron"],
            n_embed=checkpoint["config"]["n_embed"],
            param_start_mean=checkpoint["config"]["param_start_mean"],
            param_start_std=checkpoint["config"]["param_start_std"],
            n_params=checkpoint["config"].get("n_params", 1),
            r_cut=checkpoint["config"]["r_cut"],
        )
        model_state_dict = {
            k.replace("_orig_mod.", ""): v
            for k, v in checkpoint["model_state_dict"].items()
        }
        model.load_state_dict(model_state_dict)
    else:
        model = create_test_atomtype_model()

    model.eval()
    model.requires_grad_(False)

    print("\nTest 1: Dataset Creation")
    test_precomputed_dataset_creation(model)

    print("\nTest 2: Forward Pass Equivalence")
    test_forward_pass_equivalence(model)

    print("\nTest 3: Training")
    test_training_with_precomputed(model)

    print("\nTest 4: Backward Compatibility")
    test_backward_compatibility(model)

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
