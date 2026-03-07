"""
Tests for the model_io module.

This module tests the checkpoint format utilities including:
- Helper functions (unwrap_model, strip_prefix_from_state_dict, get_checkpoint_version)
- Checkpoint creation (create_checkpoint, create_submodel_checkpoint)
- Save/load operations (save_checkpoint, load_checkpoint)
- State dict extraction (load_state_dict_from_checkpoint, load_config_from_checkpoint)
- Submodel handling (get_submodel_checkpoint, has_embedded_submodel, warn_submodel_override)
- Validation (validate_checkpoint)
- V1 to V2 upgrade (upgrade_v1_checkpoint)
- Model integration (get_config methods, save_model methods)
"""

import os
import tempfile
import warnings

import numpy as np
import pytest
import qcelemental as qcel
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from apnet_pt import model_io


# ===========================================================================
# Test fixtures and helpers
# ===========================================================================


class SimpleModel(nn.Module):
    """A simple model for testing checkpoint operations."""

    def __init__(self, n_hidden=32, n_layers=2):
        super().__init__()
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.layers = nn.Sequential(
            nn.Linear(10, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, 5),
        )

    def forward(self, x):
        return self.layers(x)

    def get_config(self) -> dict:
        return {
            "n_hidden": self.n_hidden,
            "n_layers": self.n_layers,
        }


class MockDDPWrapper:
    """Mock DDP wrapper for testing unwrap_model."""

    def __init__(self, model):
        self.module = model


@pytest.fixture
def simple_model():
    """Create a simple model for testing."""
    return SimpleModel(n_hidden=64, n_layers=3)


@pytest.fixture
def temp_checkpoint_path():
    """Create a temporary file path for checkpoint testing."""
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        yield f.name
    # Cleanup after test
    if os.path.exists(f.name):
        os.unlink(f.name)


# ===========================================================================
# Tests for helper functions
# ===========================================================================


class TestUnwrapModel:
    """Tests for the unwrap_model function."""

    def test_unwrap_regular_model(self, simple_model):
        """Test that regular models are returned unchanged."""
        result = model_io.unwrap_model(simple_model)
        assert result is simple_model

    def test_unwrap_ddp_wrapped_model(self, simple_model):
        """Test that DDP-wrapped models are unwrapped correctly."""
        wrapped = MockDDPWrapper(simple_model)
        result = model_io.unwrap_model(wrapped)
        assert result is simple_model

    def test_unwrap_nested_wrapper(self, simple_model):
        """Test unwrapping only one level of wrapper."""
        inner_wrap = MockDDPWrapper(simple_model)
        outer_wrap = MockDDPWrapper(inner_wrap)
        result = model_io.unwrap_model(outer_wrap)
        # Should only unwrap one level
        assert result is inner_wrap


class TestStripPrefixFromStateDict:
    """Tests for the strip_prefix_from_state_dict function."""

    def test_strip_default_prefix(self):
        """Test stripping the default _orig_mod. prefix."""
        state_dict = {
            "_orig_mod.layers.0.weight": torch.randn(32, 10),
            "_orig_mod.layers.0.bias": torch.randn(32),
            "_orig_mod.layers.2.weight": torch.randn(5, 32),
            "_orig_mod.layers.2.bias": torch.randn(5),
        }
        result = model_io.strip_prefix_from_state_dict(state_dict)

        assert "layers.0.weight" in result
        assert "layers.0.bias" in result
        assert "layers.2.weight" in result
        assert "layers.2.bias" in result
        assert "_orig_mod.layers.0.weight" not in result

    def test_strip_custom_prefix(self):
        """Test stripping a custom prefix."""
        state_dict = {
            "custom_prefix.weight": torch.randn(10, 10),
            "custom_prefix.bias": torch.randn(10),
        }
        result = model_io.strip_prefix_from_state_dict(
            state_dict, prefix="custom_prefix."
        )

        assert "weight" in result
        assert "bias" in result
        assert "custom_prefix.weight" not in result

    def test_mixed_prefixed_and_unprefixed(self):
        """Test handling state dict with mixed prefixed and unprefixed keys."""
        state_dict = {
            "_orig_mod.layer1.weight": torch.randn(10, 10),
            "layer2.weight": torch.randn(10, 10),  # No prefix
        }
        result = model_io.strip_prefix_from_state_dict(state_dict)

        assert "layer1.weight" in result
        assert "layer2.weight" in result
        assert len(result) == 2

    def test_empty_state_dict(self):
        """Test with an empty state dict."""
        result = model_io.strip_prefix_from_state_dict({})
        assert result == {}


class TestGetCheckpointVersion:
    """Tests for the get_checkpoint_version function."""

    def test_v1_checkpoint_no_version_key(self):
        """Test that checkpoints without version key default to v1."""
        checkpoint = {
            "model_state_dict": {},
            "config": {},
        }
        assert model_io.get_checkpoint_version(checkpoint) == 1

    def test_v2_checkpoint(self):
        """Test v2 checkpoint version detection."""
        checkpoint = {
            "checkpoint_version": 2,
            "model_state_dict": {},
            "config": {},
        }
        assert model_io.get_checkpoint_version(checkpoint) == 2

    def test_future_version(self):
        """Test handling of future checkpoint versions."""
        checkpoint = {
            "checkpoint_version": 3,
            "model_state_dict": {},
        }
        assert model_io.get_checkpoint_version(checkpoint) == 3


# ===========================================================================
# Tests for checkpoint creation
# ===========================================================================


class TestCreateCheckpoint:
    """Tests for the create_checkpoint function."""

    def test_create_basic_checkpoint(self, simple_model):
        """Test creating a basic v2 checkpoint."""
        config = {"n_hidden": 64, "n_layers": 3}
        checkpoint = model_io.create_checkpoint(
            model=simple_model,
            config=config,
            model_type="SimpleModel",
        )

        assert checkpoint["checkpoint_version"] == 2
        assert checkpoint["model_type"] == "SimpleModel"
        assert checkpoint["config"] == config
        assert "model_state_dict" in checkpoint
        assert "metadata" in checkpoint
        assert "apnet_version" in checkpoint["metadata"]
        assert "save_date" in checkpoint["metadata"]

    def test_create_checkpoint_with_metadata(self, simple_model):
        """Test creating checkpoint with custom metadata."""
        config = {"n_hidden": 64}
        metadata = {"training_epochs": 100, "final_loss": 0.001}
        checkpoint = model_io.create_checkpoint(
            model=simple_model,
            config=config,
            model_type="SimpleModel",
            metadata=metadata,
        )

        assert checkpoint["metadata"]["training_epochs"] == 100
        assert checkpoint["metadata"]["final_loss"] == 0.001
        assert "apnet_version" in checkpoint["metadata"]

    def test_create_checkpoint_with_submodels(self, simple_model):
        """Test creating checkpoint with embedded submodels."""
        submodel = SimpleModel(n_hidden=16, n_layers=1)
        submodel_checkpoint = model_io.create_submodel_checkpoint(
            model=submodel,
            config={"n_hidden": 16, "n_layers": 1},
            model_type="SubSimpleModel",
        )

        checkpoint = model_io.create_checkpoint(
            model=simple_model,
            config={"n_hidden": 64},
            model_type="SimpleModel",
            submodels={"inner_model": submodel_checkpoint},
        )

        assert "submodels" in checkpoint
        assert "inner_model" in checkpoint["submodels"]
        assert checkpoint["submodels"]["inner_model"]["model_type"] == "SubSimpleModel"

    def test_create_checkpoint_strips_compile_prefix(self):
        """Test that checkpoint creation strips _orig_mod. prefix from state dict."""
        model = SimpleModel()
        # Simulate torch.compile prefix
        original_state_dict = model.state_dict()
        prefixed_state_dict = {
            f"_orig_mod.{k}": v for k, v in original_state_dict.items()
        }

        # Mock the model's state_dict to return prefixed version
        class PrefixedModel(SimpleModel):
            def state_dict(self):
                return prefixed_state_dict

        prefixed_model = PrefixedModel()
        checkpoint = model_io.create_checkpoint(
            model=prefixed_model,
            config={},
            model_type="SimpleModel",
        )

        # Check that prefix was stripped
        for key in checkpoint["model_state_dict"].keys():
            assert not key.startswith("_orig_mod.")


class TestCreateSubmodelCheckpoint:
    """Tests for the create_submodel_checkpoint function."""

    def test_create_submodel_checkpoint(self, simple_model):
        """Test creating a submodel checkpoint."""
        config = {"n_hidden": 64}
        submodel_ckpt = model_io.create_submodel_checkpoint(
            model=simple_model,
            config=config,
            model_type="SimpleModel",
        )

        assert "model_state_dict" in submodel_ckpt
        assert "config" in submodel_ckpt
        assert "model_type" in submodel_ckpt
        assert submodel_ckpt["model_type"] == "SimpleModel"
        # Submodel checkpoints should not have top-level metadata
        assert "metadata" not in submodel_ckpt
        assert "checkpoint_version" not in submodel_ckpt

    def test_create_submodel_checkpoint_with_nested_submodels(self, simple_model):
        """Test creating submodel checkpoint with nested submodels."""
        inner_submodel = SimpleModel(n_hidden=8)
        inner_ckpt = model_io.create_submodel_checkpoint(
            model=inner_submodel,
            config={"n_hidden": 8},
            model_type="InnerModel",
        )

        outer_ckpt = model_io.create_submodel_checkpoint(
            model=simple_model,
            config={"n_hidden": 64},
            model_type="OuterModel",
            submodels={"inner": inner_ckpt},
        )

        assert "submodels" in outer_ckpt
        assert "inner" in outer_ckpt["submodels"]


# ===========================================================================
# Tests for save/load operations
# ===========================================================================


class TestSaveLoadCheckpoint:
    """Tests for save_checkpoint and load_checkpoint functions."""

    def test_save_and_load_checkpoint(self, simple_model, temp_checkpoint_path):
        """Test saving and loading a checkpoint."""
        checkpoint = model_io.create_checkpoint(
            model=simple_model,
            config={"n_hidden": 64},
            model_type="SimpleModel",
        )

        model_io.save_checkpoint(checkpoint, temp_checkpoint_path)
        loaded = model_io.load_checkpoint(temp_checkpoint_path)

        assert loaded["checkpoint_version"] == checkpoint["checkpoint_version"]
        assert loaded["model_type"] == checkpoint["model_type"]
        assert loaded["config"] == checkpoint["config"]

        # Check state dict values match
        for key in checkpoint["model_state_dict"]:
            assert key in loaded["model_state_dict"]
            assert torch.allclose(
                checkpoint["model_state_dict"][key],
                loaded["model_state_dict"][key],
            )

    def test_load_checkpoint_to_cpu(self, simple_model, temp_checkpoint_path):
        """Test loading checkpoint with map_location='cpu'."""
        checkpoint = model_io.create_checkpoint(
            model=simple_model,
            config={},
            model_type="SimpleModel",
        )
        model_io.save_checkpoint(checkpoint, temp_checkpoint_path)

        loaded = model_io.load_checkpoint(temp_checkpoint_path, map_location="cpu")
        for tensor in loaded["model_state_dict"].values():
            assert tensor.device.type == "cpu"

    def test_load_checkpoint_default_cpu(self, simple_model, temp_checkpoint_path):
        """Test that load_checkpoint defaults to CPU when map_location is None."""
        checkpoint = model_io.create_checkpoint(
            model=simple_model,
            config={},
            model_type="SimpleModel",
        )
        model_io.save_checkpoint(checkpoint, temp_checkpoint_path)

        # map_location=None should default to CPU
        loaded = model_io.load_checkpoint(temp_checkpoint_path, map_location=None)
        for tensor in loaded["model_state_dict"].values():
            assert tensor.device.type == "cpu"


# ===========================================================================
# Tests for state dict extraction
# ===========================================================================


class TestLoadStateDictFromCheckpoint:
    """Tests for the load_state_dict_from_checkpoint function."""

    def test_load_state_dict_v2(self, simple_model):
        """Test extracting state dict from v2 checkpoint."""
        checkpoint = model_io.create_checkpoint(
            model=simple_model,
            config={},
            model_type="SimpleModel",
        )

        state_dict = model_io.load_state_dict_from_checkpoint(checkpoint)

        assert "layers.0.weight" in state_dict
        assert "layers.0.bias" in state_dict

    def test_load_state_dict_v1(self, simple_model):
        """Test extracting state dict from v1 checkpoint."""
        v1_checkpoint = {
            "model_state_dict": simple_model.state_dict(),
            "config": {},
        }

        state_dict = model_io.load_state_dict_from_checkpoint(v1_checkpoint)

        assert "layers.0.weight" in state_dict

    def test_load_state_dict_strips_prefix(self, simple_model):
        """Test that state dict loading strips compile prefix."""
        original_state = simple_model.state_dict()
        prefixed_state = {f"_orig_mod.{k}": v for k, v in original_state.items()}
        checkpoint = {
            "checkpoint_version": 2,
            "model_state_dict": prefixed_state,
            "config": {},
            "model_type": "SimpleModel",
        }

        state_dict = model_io.load_state_dict_from_checkpoint(
            checkpoint, strip_compile_prefix=True
        )

        for key in state_dict:
            assert not key.startswith("_orig_mod.")

    def test_load_state_dict_preserve_prefix(self, simple_model):
        """Test that state dict loading can preserve prefix."""
        original_state = simple_model.state_dict()
        prefixed_state = {f"_orig_mod.{k}": v for k, v in original_state.items()}
        checkpoint = {
            "checkpoint_version": 2,
            "model_state_dict": prefixed_state,
            "config": {},
            "model_type": "SimpleModel",
        }

        state_dict = model_io.load_state_dict_from_checkpoint(
            checkpoint, strip_compile_prefix=False
        )

        assert any(key.startswith("_orig_mod.") for key in state_dict)


class TestLoadConfigFromCheckpoint:
    """Tests for the load_config_from_checkpoint function."""

    def test_load_config_v2(self, simple_model):
        """Test extracting config from v2 checkpoint."""
        config = {"n_hidden": 64, "n_layers": 3}
        checkpoint = model_io.create_checkpoint(
            model=simple_model,
            config=config,
            model_type="SimpleModel",
        )

        loaded_config = model_io.load_config_from_checkpoint(checkpoint)
        assert loaded_config == config

    def test_load_config_v1(self, simple_model):
        """Test extracting config from v1 checkpoint."""
        config = {"n_hidden": 64}
        v1_checkpoint = {
            "model_state_dict": simple_model.state_dict(),
            "config": config,
        }

        loaded_config = model_io.load_config_from_checkpoint(v1_checkpoint)
        assert loaded_config == config

    def test_load_config_missing(self, simple_model):
        """Test behavior when config is missing."""
        checkpoint = {
            "model_state_dict": simple_model.state_dict(),
        }

        loaded_config = model_io.load_config_from_checkpoint(checkpoint)
        assert loaded_config is None


# ===========================================================================
# Tests for submodel handling
# ===========================================================================


class TestGetSubmodelCheckpoint:
    """Tests for the get_submodel_checkpoint function."""

    def test_get_existing_submodel(self, simple_model):
        """Test extracting an existing submodel checkpoint."""
        submodel = SimpleModel(n_hidden=16)
        submodel_ckpt = model_io.create_submodel_checkpoint(
            model=submodel,
            config={"n_hidden": 16},
            model_type="SubModel",
        )

        checkpoint = model_io.create_checkpoint(
            model=simple_model,
            config={},
            model_type="MainModel",
            submodels={"atom_model": submodel_ckpt},
        )

        extracted = model_io.get_submodel_checkpoint(checkpoint, "atom_model")

        assert extracted is not None
        assert extracted["model_type"] == "SubModel"
        assert extracted["config"]["n_hidden"] == 16

    def test_get_nonexistent_submodel(self, simple_model):
        """Test extracting a non-existent submodel returns None."""
        checkpoint = model_io.create_checkpoint(
            model=simple_model,
            config={},
            model_type="MainModel",
        )

        extracted = model_io.get_submodel_checkpoint(checkpoint, "nonexistent")
        assert extracted is None

    def test_get_submodel_no_submodels_key(self, simple_model):
        """Test handling checkpoint without submodels key."""
        checkpoint = {
            "checkpoint_version": 2,
            "model_state_dict": simple_model.state_dict(),
            "config": {},
            "model_type": "MainModel",
        }

        extracted = model_io.get_submodel_checkpoint(checkpoint, "any_name")
        assert extracted is None


class TestHasEmbeddedSubmodel:
    """Tests for the has_embedded_submodel function."""

    def test_has_embedded_submodel_true(self, simple_model):
        """Test detecting an embedded submodel."""
        submodel = SimpleModel(n_hidden=16)
        submodel_ckpt = model_io.create_submodel_checkpoint(
            model=submodel,
            config={},
            model_type="SubModel",
        )

        checkpoint = model_io.create_checkpoint(
            model=simple_model,
            config={},
            model_type="MainModel",
            submodels={"atom_model": submodel_ckpt},
        )

        assert model_io.has_embedded_submodel(checkpoint, "atom_model") is True

    def test_has_embedded_submodel_false(self, simple_model):
        """Test detecting absence of a submodel."""
        checkpoint = model_io.create_checkpoint(
            model=simple_model,
            config={},
            model_type="MainModel",
        )

        assert model_io.has_embedded_submodel(checkpoint, "atom_model") is False


class TestWarnSubmodelOverride:
    """Tests for the warn_submodel_override function."""

    def test_warn_submodel_override_with_path(self):
        """Test warning message includes external path."""
        with pytest.warns(UserWarning, match="embedded"):
            model_io.warn_submodel_override(
                submodel_name="atom_model",
                embedded_type="AtomMPNN",
                external_path="/path/to/model.pt",
            )

    def test_warn_submodel_override_without_path(self):
        """Test warning message without external path."""
        with pytest.warns(UserWarning, match="embedded"):
            model_io.warn_submodel_override(
                submodel_name="atom_model",
                embedded_type="AtomMPNN",
            )


# ===========================================================================
# Tests for validation
# ===========================================================================


class TestValidateCheckpoint:
    """Tests for the validate_checkpoint function."""

    def test_validate_v2_checkpoint_valid(self, simple_model):
        """Test validating a valid v2 checkpoint."""
        checkpoint = model_io.create_checkpoint(
            model=simple_model,
            config={},
            model_type="SimpleModel",
        )

        result = model_io.validate_checkpoint(checkpoint)
        assert result is True

    def test_validate_v2_checkpoint_missing_key(self, simple_model):
        """Test validating v2 checkpoint missing required key."""
        checkpoint = {
            "checkpoint_version": 2,
            "model_state_dict": simple_model.state_dict(),
            # Missing 'config' and 'model_type'
        }

        with pytest.raises(ValueError, match="missing required key"):
            model_io.validate_checkpoint(checkpoint)

    def test_validate_v2_checkpoint_type_mismatch(self, simple_model):
        """Test validating v2 checkpoint with wrong model type."""
        checkpoint = model_io.create_checkpoint(
            model=simple_model,
            config={},
            model_type="SimpleModel",
        )

        with pytest.raises(ValueError, match="model_type mismatch"):
            model_io.validate_checkpoint(checkpoint, expected_type="AtomMPNN")

    def test_validate_v2_checkpoint_type_match(self, simple_model):
        """Test validating v2 checkpoint with matching model type."""
        checkpoint = model_io.create_checkpoint(
            model=simple_model,
            config={},
            model_type="SimpleModel",
        )

        result = model_io.validate_checkpoint(checkpoint, expected_type="SimpleModel")
        assert result is True

    def test_validate_v1_checkpoint_valid(self, simple_model):
        """Test validating a valid v1 checkpoint."""
        v1_checkpoint = {
            "model_state_dict": simple_model.state_dict(),
            "config": {},
        }

        result = model_io.validate_checkpoint(v1_checkpoint)
        assert result is True


# ===========================================================================
# Tests for v1 to v2 upgrade
# ===========================================================================


class TestUpgradeV1Checkpoint:
    """Tests for the upgrade_v1_checkpoint function."""

    def test_upgrade_v1_to_v2(self, simple_model):
        """Test upgrading a v1 checkpoint to v2 format."""
        v1_checkpoint = {
            "model_state_dict": simple_model.state_dict(),
            "config": {"old_key": "old_value"},
        }

        upgraded = model_io.upgrade_v1_checkpoint(
            checkpoint=v1_checkpoint,
            config={"n_hidden": 64},
            model_type="SimpleModel",
        )

        assert upgraded["checkpoint_version"] == 2
        assert upgraded["model_type"] == "SimpleModel"
        # New config should override old config for matching keys
        assert upgraded["config"]["n_hidden"] == 64
        # Old config keys should be preserved
        assert upgraded["config"]["old_key"] == "old_value"
        assert upgraded["metadata"]["upgraded_from_v1"] is True

    def test_upgrade_preserves_state_dict(self, simple_model):
        """Test that upgrade preserves the state dict."""
        original_state = simple_model.state_dict()
        v1_checkpoint = {
            "model_state_dict": original_state,
        }

        upgraded = model_io.upgrade_v1_checkpoint(
            checkpoint=v1_checkpoint,
            config={},
            model_type="SimpleModel",
        )

        for key in original_state:
            assert torch.allclose(
                upgraded["model_state_dict"][key],
                original_state[key],
            )


# ===========================================================================
# Tests for model integration (get_config methods)
# ===========================================================================


class TestAtomMPNNGetConfig:
    """Tests for AtomMPNN.get_config method."""

    def test_atom_mpnn_get_config(self):
        """Test that AtomMPNN.get_config returns expected keys."""
        from apnet_pt.AtomModels.ap2_atom_model import AtomMPNN

        model = AtomMPNN(
            n_message=3,
            n_rbf=8,
            n_neuron=128,
            n_embed=16,
            r_cut=5.0,
        )

        config = model.get_config()

        assert config["n_message"] == 3
        assert config["n_rbf"] == 8
        assert config["n_neuron"] == 128
        assert config["n_embed"] == 16
        assert config["r_cut"] == 5.0


class TestAPNet2MPNNGetConfig:
    """Tests for APNet2_MPNN.get_config method."""

    def test_apnet2_mpnn_get_config(self):
        """Test that APNet2_MPNN.get_config returns expected keys."""
        from apnet_pt.AtomPairwiseModels.apnet2 import APNet2_MPNN

        model = APNet2_MPNN(
            n_message=3,
            n_rbf=8,
            n_neuron=128,
            n_embed=16,
            r_cut_im=8.0,
            r_cut=5.0,
        )

        config = model.get_config()

        assert config["n_message"] == 3
        assert config["n_rbf"] == 8
        assert config["n_neuron"] == 128
        assert config["n_embed"] == 16
        assert config["r_cut_im"] == 8.0
        assert config["r_cut"] == 5.0


# ===========================================================================
# Integration tests for model save/load roundtrip
# ===========================================================================


class TestAtomModelSaveLoad:
    """Integration tests for AtomModel save and load."""

    def test_atom_model_save_and_load(self, temp_checkpoint_path):
        """Test saving and loading an AtomModel checkpoint."""
        from apnet_pt.AtomModels.ap2_atom_model import AtomModel, AtomMPNN

        # Create a model
        model = AtomModel(
            ds_root=None,
            ignore_database_null=True,
            use_GPU=False,
            n_message=2,
            n_rbf=4,
            n_neuron=32,
            n_embed=8,
            r_cut=4.0,
        )

        # Save the model
        model.save_model(temp_checkpoint_path, metadata={"test": "value"})

        # Load the checkpoint and verify structure
        checkpoint = model_io.load_checkpoint(temp_checkpoint_path)

        assert checkpoint["checkpoint_version"] == 2
        assert checkpoint["model_type"] == "AtomMPNN"
        assert checkpoint["config"]["n_message"] == 2
        assert checkpoint["config"]["n_rbf"] == 4
        assert checkpoint["config"]["n_neuron"] == 32
        assert checkpoint["config"]["n_embed"] == 8
        assert checkpoint["config"]["r_cut"] == 4.0
        assert checkpoint["metadata"]["test"] == "value"

    def test_atom_model_create_checkpoint(self):
        """Test creating a checkpoint from AtomModel."""
        from apnet_pt.AtomModels.ap2_atom_model import AtomModel

        model = AtomModel(
            ds_root=None,
            ignore_database_null=True,
            use_GPU=False,
        )

        checkpoint = model._create_checkpoint(metadata={"custom": "data"})

        assert checkpoint["checkpoint_version"] == 2
        assert checkpoint["model_type"] == "AtomMPNN"
        assert "model_state_dict" in checkpoint
        assert "config" in checkpoint
        assert checkpoint["metadata"]["custom"] == "data"


class TestAPNet2ModelSaveLoad:
    """Integration tests for APNet2Model save and load."""

    def test_apnet2_model_save_with_embedded_atom_model(self, temp_checkpoint_path):
        """Test saving APNet2Model with embedded atom_model."""
        from apnet_pt.AtomPairwiseModels.apnet2 import APNet2Model

        model = APNet2Model(
            ds_root=None,
            ignore_database_null=True,
            use_GPU=False,
            n_message=2,
            n_rbf=4,
            n_neuron=32,
            n_embed=8,
            r_cut_im=6.0,
            r_cut=4.0,
        )

        # Save with embedded atom_model
        model.save_model(temp_checkpoint_path, embed_atom_model=True)

        # Load and verify
        checkpoint = model_io.load_checkpoint(temp_checkpoint_path)

        assert checkpoint["checkpoint_version"] == 2
        assert checkpoint["model_type"] == "APNet2_MPNN"
        assert model_io.has_embedded_submodel(checkpoint, "atom_model")

        # Verify embedded atom_model
        atom_model_ckpt = model_io.get_submodel_checkpoint(checkpoint, "atom_model")
        assert atom_model_ckpt["model_type"] == "AtomMPNN"
        assert "model_state_dict" in atom_model_ckpt
        assert "config" in atom_model_ckpt

    def test_apnet2_model_save_without_embedded_atom_model(self, temp_checkpoint_path):
        """Test saving APNet2Model without embedded atom_model."""
        from apnet_pt.AtomPairwiseModels.apnet2 import APNet2Model

        model = APNet2Model(
            ds_root=None,
            ignore_database_null=True,
            use_GPU=False,
        )

        # Save without embedded atom_model
        model.save_model(temp_checkpoint_path, embed_atom_model=False)

        checkpoint = model_io.load_checkpoint(temp_checkpoint_path)

        assert checkpoint["checkpoint_version"] == 2
        assert not model_io.has_embedded_submodel(checkpoint, "atom_model")

    def test_apnet2_model_create_checkpoint(self):
        """Test creating a checkpoint from APNet2Model."""
        from apnet_pt.AtomPairwiseModels.apnet2 import APNet2Model

        model = APNet2Model(
            ds_root=None,
            ignore_database_null=True,
            use_GPU=False,
        )

        checkpoint = model._create_checkpoint(
            embed_atom_model=True,
            metadata={"test": "metadata"},
        )

        assert checkpoint["checkpoint_version"] == 2
        assert checkpoint["model_type"] == "APNet2_MPNN"
        assert checkpoint["metadata"]["test"] == "metadata"
        assert "submodels" in checkpoint
        assert "atom_model" in checkpoint["submodels"]


# ===========================================================================
# Tests for backward compatibility with v1 checkpoints
# ===========================================================================


class TestBackwardCompatibility:
    """Tests for backward compatibility with v1 checkpoints."""

    def test_load_v1_checkpoint_state_dict(self, simple_model, temp_checkpoint_path):
        """Test loading a v1 checkpoint (legacy format)."""
        # Create v1 checkpoint
        v1_checkpoint = {
            "model_state_dict": simple_model.state_dict(),
            "config": {"n_hidden": 64, "n_layers": 3},
        }
        torch.save(v1_checkpoint, temp_checkpoint_path)

        # Load and verify
        loaded = model_io.load_checkpoint(temp_checkpoint_path)

        assert model_io.get_checkpoint_version(loaded) == 1
        state_dict = model_io.load_state_dict_from_checkpoint(loaded)

        # Should be able to load into model
        new_model = SimpleModel(n_hidden=64, n_layers=3)
        new_model.load_state_dict(state_dict)

    def test_v1_checkpoint_validation(self, simple_model):
        """Test that v1 checkpoints pass validation."""
        v1_checkpoint = {
            "model_state_dict": simple_model.state_dict(),
            "config": {},
        }

        assert model_io.validate_checkpoint(v1_checkpoint) is True

    def test_v1_checkpoint_config_extraction(self, simple_model):
        """Test extracting config from v1 checkpoint."""
        config = {"n_hidden": 64, "n_layers": 3}
        v1_checkpoint = {
            "model_state_dict": simple_model.state_dict(),
            "config": config,
        }

        loaded_config = model_io.load_config_from_checkpoint(v1_checkpoint)
        assert loaded_config == config


# ===========================================================================
# Edge cases and error handling
# ===========================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_create_checkpoint_with_empty_config(self, simple_model):
        """Test creating checkpoint with empty config."""
        checkpoint = model_io.create_checkpoint(
            model=simple_model,
            config={},
            model_type="SimpleModel",
        )

        assert checkpoint["config"] == {}

    def test_create_checkpoint_with_none_metadata(self, simple_model):
        """Test creating checkpoint with None metadata."""
        checkpoint = model_io.create_checkpoint(
            model=simple_model,
            config={},
            model_type="SimpleModel",
            metadata=None,
        )

        # Should still have default metadata fields
        assert "apnet_version" in checkpoint["metadata"]
        assert "save_date" in checkpoint["metadata"]

    def test_checkpoint_version_constant(self):
        """Test that CHECKPOINT_VERSION constant is set."""
        assert model_io.CHECKPOINT_VERSION == 2

    def test_validate_checkpoint_invalid_format(self):
        """Test validating an invalid checkpoint format."""
        invalid_checkpoint = {"random_key": "random_value"}

        with pytest.raises(ValueError):
            model_io.validate_checkpoint(invalid_checkpoint)


# ===========================================================================
# APNet3_fused integration test - train, save, load roundtrip
# ===========================================================================


class TestAPNet3FusedSaveLoad:
    """Integration tests for APNet3_AtomType_Model save and load with embedded submodels."""

    # Test molecule fixtures
    mol_cliff_water_close = qcel.models.Molecule.from_data("""
0 1
O                    -1.326958220000    -0.105938540000     0.018788150000
H                    -1.931665230000     1.600174310000    -0.021710520000
H                     0.486644270000     0.079598100000     0.009862480000
--
0 1
O                     3.907523240000     0.052757410000     0.001850160000
H                     4.619234940000    -0.775660840000     1.449615410000
H                     4.611000850000    -0.847154680000    -1.406756420000
units bohr
no_com
no_reorient
""")

    @pytest.fixture
    def current_file_path(self):
        """Get current test file path."""
        return os.path.dirname(os.path.realpath(__file__))

    @pytest.fixture
    def test_models_path(self, current_file_path):
        """Get path to test models."""
        return f"{current_file_path}/test_models/ap3_ensemble_0"

    @pytest.fixture
    def data_path(self, current_file_path):
        """Get path to test data."""
        return f"{current_file_path}/test_data_path"

    @pytest.fixture
    def ap3_temp_checkpoint_path(self):
        """Create a temporary file path for APNet3 checkpoint testing."""
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            yield f.name
        # Cleanup after test
        if os.path.exists(f.name):
            os.unlink(f.name)

    def test_apnet3_fused_train_save_load_roundtrip(
        self, test_models_path, data_path, ap3_temp_checkpoint_path
    ):
        """
        Test full APNet3_AtomType_Model train, save, load roundtrip.

        This test:
        1. Creates required sub-models (AtomTypeParamModel, AM_DimerParam_Model)
        2. Creates an in-memory dataset
        3. Creates and trains an APNet3_AtomType_Model for a few epochs
        4. Makes predictions before saving
        5. Saves the model using v2 checkpoint format with embedded dimer_prop_model
        6. Loads the model back via pre_trained_model_path
        7. Verifies predictions match the original model
        8. Verifies the checkpoint structure has expected v2 format keys
        """
        import apnet_pt
        from apnet_pt.AtomPairwiseModels.apnet3_fused import APNet3_AtomType_Model
        from apnet_pt.pt_datasets.ap3_fused_ds import ap3_fused_module_dataset

        # Model paths
        am_path = f"{test_models_path}/am_3.pt"
        at_hf_vw_path = f"{test_models_path}/am_h+1_3.pt"
        at_elst_path = f"{test_models_path}/am_elst_h+1_3.pt"

        # Dataset parameters
        batch_size = 2
        atomic_batch_size = 4
        datapoint_storage_n_objects = 6

        # Create test data
        qcel_molecules = [self.mol_cliff_water_close] * 4
        energy_labels = [
            np.array(
                [
                    -10.779292828139122,
                    11.390991215401051,
                    -3.414543432719425,
                    -2.436025699701581,
                ]
            )
            for _ in range(len(qcel_molecules))
        ]

        # Create sub-models
        atom_type_hf_vw_model = apnet_pt.AtomPairwiseModels.mtp_mtp.AtomTypeParamModel(
            ds_root=None,
            use_GPU=False,
            ignore_database_null=True,
            atom_model_pre_trained_path=am_path,
            pre_trained_model_path=at_hf_vw_path,
        )
        atom_type_elst_model = apnet_pt.AtomPairwiseModels.mtp_mtp.AM_DimerParam_Model(
            ds_root=None,
            use_GPU=False,
            ignore_database_null=True,
            atom_model=atom_type_hf_vw_model.model,
            atom_model_type="AtomTypeParamNN",
            pre_trained_model_path=at_elst_path,
        )

        # Create dataset
        ds = ap3_fused_module_dataset(
            root=data_path,
            r_cut=5.0,
            r_cut_im=8.0,
            spec_type=None,
            max_size=None,
            force_reprocess=True,
            atomic_batch_size=atomic_batch_size,
            dimer_prop_model=atom_type_elst_model.dimer_model,
            datapoint_storage_n_objects=datapoint_storage_n_objects,
            batch_size=batch_size,
            num_devices=1,
            skip_processed=True,
            skip_compile=True,
            print_level=2,
            qcel_molecules=qcel_molecules,
            energy_labels=energy_labels,
            in_memory=True,
            random_seed=42,  # For reproducibility
        )

        # Create APNet3_AtomType_Model
        ap3_model = APNet3_AtomType_Model(
            ds_root=None,
            atom_type_model=atom_type_hf_vw_model.model,
            dimer_prop_model=atom_type_elst_model.dimer_model,
            am_dimer_param_model=atom_type_elst_model,
            use_precomputed_classical=False,
            use_GPU=False,
        )

        # Train for a few epochs
        ap3_model.train(
            ds,
            n_epochs=3,
            skip_compile=True,
            transfer_learning=False,
            lr=0.0005,
        )

        # Make predictions before saving
        predictions_before = ap3_model.predict_qcel_mols(
            qcel_molecules[0:2], batch_size=2
        )

        # Save the model with embedded submodels
        ap3_model.save_model(
            ap3_temp_checkpoint_path,
            metadata={"test_key": "test_value", "training_epochs": 3},
        )

        # Load and verify checkpoint structure
        checkpoint = model_io.load_checkpoint(ap3_temp_checkpoint_path)

        # Verify v2 checkpoint structure
        assert checkpoint["checkpoint_version"] == 2
        assert checkpoint["model_type"] == "APNet3_AtomType_MPNN"
        assert "model_state_dict" in checkpoint
        assert "config" in checkpoint
        assert "metadata" in checkpoint

        # Verify config contains expected keys
        config = checkpoint["config"]
        assert "n_message" in config
        assert "n_rbf" in config
        assert "n_neuron" in config
        assert "n_embed" in config
        assert "r_cut_im" in config
        assert "r_cut" in config

        # Verify metadata
        assert checkpoint["metadata"]["test_key"] == "test_value"
        assert checkpoint["metadata"]["training_epochs"] == 3
        assert "apnet_version" in checkpoint["metadata"]
        assert "save_date" in checkpoint["metadata"]

        # Verify embedded dimer_prop_model submodel
        assert model_io.has_embedded_submodel(checkpoint, "dimer_prop_model")
        dimer_prop_ckpt = model_io.get_submodel_checkpoint(
            checkpoint, "dimer_prop_model"
        )
        assert dimer_prop_ckpt is not None
        assert dimer_prop_ckpt["model_type"] == "DimerProp"
        assert "model_state_dict" in dimer_prop_ckpt
        assert "config" in dimer_prop_ckpt

        # Load the model back using the checkpoint
        # Create fresh sub-models for loaded model
        atom_type_hf_vw_model_loaded = (
            apnet_pt.AtomPairwiseModels.mtp_mtp.AtomTypeParamModel(
                ds_root=None,
                use_GPU=False,
                ignore_database_null=True,
                atom_model_pre_trained_path=am_path,
                pre_trained_model_path=at_hf_vw_path,
            )
        )
        atom_type_elst_model_loaded = (
            apnet_pt.AtomPairwiseModels.mtp_mtp.AM_DimerParam_Model(
                ds_root=None,
                use_GPU=False,
                ignore_database_null=True,
                atom_model=atom_type_hf_vw_model_loaded.model,
                atom_model_type="AtomTypeParamNN",
                pre_trained_model_path=at_elst_path,
            )
        )

        ap3_model_loaded = APNet3_AtomType_Model(
            ds_root=None,
            atom_type_model=atom_type_hf_vw_model_loaded.model,
            dimer_prop_model=atom_type_elst_model_loaded.dimer_model,
            am_dimer_param_model=atom_type_elst_model_loaded,
            pre_trained_model_path=ap3_temp_checkpoint_path,
            use_precomputed_classical=False,
            use_GPU=False,
        )

        # Make predictions after loading
        predictions_after = ap3_model_loaded.predict_qcel_mols(
            qcel_molecules[0:2], batch_size=2
        )

        # Verify predictions match
        assert np.allclose(predictions_before, predictions_after, atol=1e-6), (
            f"Predictions mismatch:\n"
            f"Before: {predictions_before}\n"
            f"After: {predictions_after}"
        )

    def test_apnet3_fused_from_scratch_train_save_load_nested_models(
        self, tmp_path, ap3_temp_checkpoint_path
    ):
        """
        Build APNet3 fused stack from scratch, train briefly, and verify save/load.

        This test intentionally avoids pretrained checkpoints and validates that:
        1) the full nested stack can be constructed from fresh model instances,
        2) APNet3 can be trained on a tiny in-memory dataset,
        3) v2 checkpoint save includes nested submodel metadata, and
        4) loading restores predictions and nested dimer-prop parameters.
        """
        from apnet_pt.AtomModels.ap2_atom_model import AtomModel
        from apnet_pt.AtomPairwiseModels.apnet3_fused import APNet3_AtomType_Model
        from apnet_pt.AtomPairwiseModels.mtp_mtp import (
            AM_DimerParam_Model,
            AtomTypeParamModel,
        )
        from apnet_pt.pt_datasets.ap3_fused_ds import ap3_fused_module_dataset

        np.random.seed(7)
        torch.manual_seed(7)

        qcel_molecules = [self.mol_cliff_water_close] * 4
        energy_labels = [
            np.array(
                [
                    -10.779292828139122,
                    11.390991215401051,
                    -3.414543432719425,
                    -2.436025699701581,
                ],
                dtype=np.float32,
            )
            for _ in range(len(qcel_molecules))
        ]

        atom_model = AtomModel(
            ds_root=None,
            ignore_database_null=True,
            use_GPU=False,
            n_message=1,
            n_rbf=4,
            n_neuron=16,
            n_embed=4,
            r_cut=4.0,
        )

        atom_type_hf_vw_model = AtomTypeParamModel(
            ds_root=None,
            use_GPU=False,
            ignore_database_null=True,
            atom_model=atom_model.model,
            atom_model_type="AtomMPNN",
            n_message=1,
            n_neuron=16,
            n_embed=4,
            param_start_mean=1.6,
            param_start_std=0.05,
            model_save_path=None,
            monomer_eval_type="hirshfeld_volume_ratio__valence_width",
        )

        atom_type_elst_model = AM_DimerParam_Model(
            ds_root=None,
            use_GPU=False,
            ignore_database_null=True,
            atom_model=atom_type_hf_vw_model.model,
            atom_model_type="AtomTypeParamNN",
            model_type="AtomTypeParamNN",
            n_message=1,
            n_neuron=16,
            n_embed=4,
            n_params=1,
            dimer_eval_type="ap3_elst_damping__induced_dipole",
        )

        ds_root = tmp_path / "ap3_fused_scratch_ds"
        (ds_root / "raw").mkdir(parents=True, exist_ok=True)
        (ds_root / "processed").mkdir(parents=True, exist_ok=True)

        ds = ap3_fused_module_dataset(
            root=str(ds_root),
            r_cut=4.0,
            r_cut_im=6.0,
            spec_type=None,
            max_size=None,
            force_reprocess=True,
            atomic_batch_size=2,
            dimer_prop_model=atom_type_elst_model.dimer_model,
            datapoint_storage_n_objects=4,
            batch_size=2,
            num_devices=1,
            skip_processed=True,
            skip_compile=True,
            print_level=0,
            qcel_molecules=qcel_molecules,
            energy_labels=energy_labels,
            in_memory=True,
            random_seed=7,
        )

        ap3_model = APNet3_AtomType_Model(
            ds_root=None,
            atom_type_model=atom_type_hf_vw_model.model,
            dimer_prop_model=atom_type_elst_model.dimer_model,
            am_dimer_param_model=atom_type_elst_model,
            use_precomputed_classical=False,
            use_GPU=False,
            n_message=1,
            n_rbf=4,
            n_neuron=16,
            n_embed=4,
            r_cut_im=6.0,
            r_cut=4.0,
        )

        ap3_model.train(
            ds,
            n_epochs=1,
            skip_compile=True,
            transfer_learning=False,
            lr=5e-4,
            dataloader_num_workers=0,
        )

        predictions_before = ap3_model.predict_qcel_mols(
            qcel_molecules[:2], batch_size=2
        )

        ap3_model.save_model(
            ap3_temp_checkpoint_path,
            metadata={"scratch_build": True, "training_epochs": 1},
        )

        checkpoint = model_io.load_checkpoint(ap3_temp_checkpoint_path)
        assert checkpoint["checkpoint_version"] == 2
        assert checkpoint["model_type"] == "APNet3_AtomType_MPNN"
        assert checkpoint["metadata"]["scratch_build"] is True
        assert model_io.has_embedded_submodel(checkpoint, "dimer_prop_model")

        dimer_prop_ckpt = model_io.get_submodel_checkpoint(
            checkpoint, "dimer_prop_model"
        )
        assert dimer_prop_ckpt is not None
        assert dimer_prop_ckpt["model_type"] == "DimerProp"
        assert "model_state_dict" in dimer_prop_ckpt
        assert "config" in dimer_prop_ckpt

        fresh_atom_model = AtomModel(
            ds_root=None,
            ignore_database_null=True,
            use_GPU=False,
            n_message=1,
            n_rbf=4,
            n_neuron=16,
            n_embed=4,
            r_cut=4.0,
        )
        fresh_atom_type_hf_vw_model = AtomTypeParamModel(
            ds_root=None,
            use_GPU=False,
            ignore_database_null=True,
            atom_model=fresh_atom_model.model,
            atom_model_type="AtomMPNN",
            n_message=1,
            n_neuron=16,
            n_embed=4,
            param_start_mean=1.6,
            param_start_std=0.05,
            model_save_path=None,
            monomer_eval_type="hirshfeld_volume_ratio__valence_width",
        )
        fresh_atom_type_elst_model = AM_DimerParam_Model(
            ds_root=None,
            use_GPU=False,
            ignore_database_null=True,
            atom_model=fresh_atom_type_hf_vw_model.model,
            atom_model_type="AtomTypeParamNN",
            model_type="AtomTypeParamNN",
            n_message=1,
            n_neuron=16,
            n_embed=4,
            n_params=1,
            dimer_eval_type="ap3_elst_damping__induced_dipole",
        )

        nested_state_before = fresh_atom_type_elst_model.dimer_model.state_dict()
        tracked_key = next(
            k
            for k in nested_state_before
            if "param_readout_layers" in k and "weight" in k
        )
        nested_param_before = nested_state_before[tracked_key].detach().clone()

        ap3_model_loaded = APNet3_AtomType_Model(
            ds_root=None,
            atom_type_model=fresh_atom_type_hf_vw_model.model,
            dimer_prop_model=fresh_atom_type_elst_model.dimer_model,
            am_dimer_param_model=fresh_atom_type_elst_model,
            pre_trained_model_path=ap3_temp_checkpoint_path,
            use_precomputed_classical=False,
            use_GPU=False,
            n_message=1,
            n_rbf=4,
            n_neuron=16,
            n_embed=4,
            r_cut_im=6.0,
            r_cut=4.0,
        )

        nested_param_after = (
            ap3_model_loaded.dimer_prop_model.state_dict()[tracked_key].detach().clone()
        )
        assert not torch.allclose(nested_param_before, nested_param_after)

        predictions_after = ap3_model_loaded.predict_qcel_mols(
            qcel_molecules[:2], batch_size=2
        )

        assert np.allclose(predictions_before, predictions_after, atol=1e-5), (
            f"Predictions mismatch after scratch save/load roundtrip:\n"
            f"Before: {predictions_before}\n"
            f"After: {predictions_after}"
        )

    def test_apnet3_fused_checkpoint_config_roundtrip(
        self, test_models_path, ap3_temp_checkpoint_path
    ):
        """
        Test that APNet3 model config is correctly saved and can be extracted.
        """
        import apnet_pt
        from apnet_pt.AtomPairwiseModels.apnet3_fused import APNet3_AtomType_Model

        # Model paths
        am_path = f"{test_models_path}/am_3.pt"
        at_hf_vw_path = f"{test_models_path}/am_h+1_3.pt"
        at_elst_path = f"{test_models_path}/am_elst_h+1_3.pt"

        # Create sub-models
        atom_type_hf_vw_model = apnet_pt.AtomPairwiseModels.mtp_mtp.AtomTypeParamModel(
            ds_root=None,
            use_GPU=False,
            ignore_database_null=True,
            atom_model_pre_trained_path=am_path,
            pre_trained_model_path=at_hf_vw_path,
        )
        atom_type_elst_model = apnet_pt.AtomPairwiseModels.mtp_mtp.AM_DimerParam_Model(
            ds_root=None,
            use_GPU=False,
            ignore_database_null=True,
            atom_model=atom_type_hf_vw_model.model,
            atom_model_type="AtomTypeParamNN",
            pre_trained_model_path=at_elst_path,
        )

        # Create APNet3_AtomType_Model with specific config
        original_config = {
            "n_message": 2,
            "n_rbf": 6,
            "n_neuron": 64,
            "n_embed": 4,
            "r_cut_im": 7.0,
            "r_cut": 4.5,
        }

        ap3_model = APNet3_AtomType_Model(
            ds_root=None,
            atom_type_model=atom_type_hf_vw_model.model,
            dimer_prop_model=atom_type_elst_model.dimer_model,
            am_dimer_param_model=atom_type_elst_model,
            use_precomputed_classical=False,
            use_GPU=False,
            **original_config,
        )

        # Save the model
        ap3_model.save_model(ap3_temp_checkpoint_path)

        # Load checkpoint and extract config
        checkpoint = model_io.load_checkpoint(ap3_temp_checkpoint_path)
        loaded_config = model_io.load_config_from_checkpoint(checkpoint)

        # Verify config values match
        for key in original_config:
            assert loaded_config[key] == original_config[key], (
                f"Config mismatch for {key}: "
                f"expected {original_config[key]}, got {loaded_config[key]}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
