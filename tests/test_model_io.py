"""Focused tests for the model_io checkpoint utilities."""

import numpy as np
import pytest
import qcelemental as qcel
import torch
import torch.nn as nn

from apnet_pt import model_io


class SimpleModel(nn.Module):
    """A small model for checkpoint utility tests."""

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


class PrefixedModel(SimpleModel):
    """Model whose state dict mimics torch.compile prefixing."""

    def __init__(self):
        super().__init__()
        original_state_dict = super().state_dict()
        self.prefixed_state_dict = {
            f"_orig_mod.{key}": value
            for key, value in original_state_dict.items()
        }

    def state_dict(self):
        return self.prefixed_state_dict


@pytest.fixture
def simple_model():
    return SimpleModel(n_hidden=64, n_layers=3)


@pytest.fixture
def submodel_checkpoint():
    submodel = SimpleModel(n_hidden=16, n_layers=1)
    return model_io.create_submodel_checkpoint(
        model=submodel,
        config={"n_hidden": 16, "n_layers": 1},
        model_type="SubSimpleModel",
    )


class TestAPNet3FusedSaveLoad:
    mol_cliff_water_close = qcel.models.Molecule.from_data(
        """
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
"""
    )

    def test_apnet3_fused_scratch_train_save_load_nested_models(
        self, tmp_path
    ):
        from apnet_pt.AtomModels.ap2_atom_model import AtomModel
        from apnet_pt.AtomPairwiseModels.apnet3_fused import (
            APNet3_AtomType_Model,
        )
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
                [-10.779292828139122, 11.390991215401051,
                 -3.414543432719425, -2.436025699701581],
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
            qcel_molecules[:2],
            batch_size=2,
        )

        checkpoint_path = tmp_path / "ap3_roundtrip.pt"
        ap3_model.save_model(
            checkpoint_path,
            metadata={"scratch_build": True, "training_epochs": 1},
        )

        checkpoint = model_io.load_checkpoint(checkpoint_path)
        assert checkpoint["checkpoint_version"] == 2
        assert checkpoint["model_type"] == "APNet3_AtomType_MPNN"
        assert checkpoint["metadata"]["scratch_build"] is True
        assert model_io.has_embedded_submodel(checkpoint, "dimer_prop_model")

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

        ap3_model_loaded = APNet3_AtomType_Model(
            ds_root=None,
            atom_type_model=fresh_atom_type_hf_vw_model.model,
            dimer_prop_model=fresh_atom_type_elst_model.dimer_model,
            am_dimer_param_model=fresh_atom_type_elst_model,
            pre_trained_model_path=checkpoint_path,
            use_precomputed_classical=False,
            use_GPU=False,
            n_message=1,
            n_rbf=4,
            n_neuron=16,
            n_embed=4,
            r_cut_im=6.0,
            r_cut=4.0,
        )

        predictions_after = ap3_model_loaded.predict_qcel_mols(
            qcel_molecules[:2],
            batch_size=2,
        )
        assert np.allclose(predictions_before, predictions_after, atol=1e-5), (
            f"Predictions mismatch after roundtrip.\n"
            f"Before: {predictions_before}\n"
            f"After: {predictions_after}"
        )


def test_unwrap_model_handles_ddp_wrapper(simple_model):
    wrapped = MockDDPWrapper(simple_model)
    assert model_io.unwrap_model(wrapped) is simple_model


def test_strip_prefix_from_state_dict_handles_mixed_keys():
    state_dict = {
        "_orig_mod.layer1.weight": torch.randn(10, 10),
        "layer2.weight": torch.randn(10, 10),
    }
    result = model_io.strip_prefix_from_state_dict(state_dict)
    assert set(result) == {"layer1.weight", "layer2.weight"}


def test_create_checkpoint_sets_v2_structure_and_strips_compile_prefix():
    checkpoint = model_io.create_checkpoint(
        model=PrefixedModel(),
        config={"n_hidden": 32},
        model_type="SimpleModel",
        metadata={"training_epochs": 3},
    )
    assert checkpoint["checkpoint_version"] == 2
    assert checkpoint["model_type"] == "SimpleModel"
    assert checkpoint["config"] == {"n_hidden": 32}
    assert checkpoint["metadata"]["training_epochs"] == 3
    assert "apnet_version" in checkpoint["metadata"]
    assert "save_date" in checkpoint["metadata"]
    assert all(
        not key.startswith("_orig_mod.")
        for key in checkpoint["model_state_dict"]
    )


def test_save_and_load_checkpoint_roundtrip(simple_model, tmp_path):
    checkpoint = model_io.create_checkpoint(
        model=simple_model,
        config=simple_model.get_config(),
        model_type="SimpleModel",
    )
    checkpoint_path = tmp_path / "simple_model.pt"
    model_io.save_checkpoint(checkpoint, checkpoint_path)
    loaded = model_io.load_checkpoint(checkpoint_path)

    assert loaded["checkpoint_version"] == checkpoint["checkpoint_version"]
    assert loaded["model_type"] == checkpoint["model_type"]
    assert loaded["config"] == checkpoint["config"]
    for key, value in checkpoint["model_state_dict"].items():
        assert key in loaded["model_state_dict"]
        assert torch.allclose(value, loaded["model_state_dict"][key])


def test_load_state_dict_from_checkpoint_strips_prefix(simple_model):
    original_state = simple_model.state_dict()
    checkpoint = {
        "checkpoint_version": 2,
        "model_state_dict": {
            f"_orig_mod.{key}": value for key, value in original_state.items()
        },
        "config": {},
        "model_type": "SimpleModel",
    }
    state_dict = model_io.load_state_dict_from_checkpoint(
        checkpoint,
        strip_compile_prefix=True,
    )
    assert all(not key.startswith("_orig_mod.") for key in state_dict)
    assert "layers.0.weight" in state_dict


def test_submodel_helpers_detect_and_extract_embedded_checkpoint(
    simple_model, submodel_checkpoint
):
    checkpoint = model_io.create_checkpoint(
        model=simple_model,
        config={},
        model_type="MainModel",
        submodels={"atom_model": submodel_checkpoint},
    )
    assert model_io.has_embedded_submodel(checkpoint, "atom_model") is True
    extracted = model_io.get_submodel_checkpoint(checkpoint, "atom_model")
    assert extracted is not None
    assert extracted["model_type"] == "SubSimpleModel"
    assert extracted["config"]["n_hidden"] == 16


def test_validate_checkpoint_rejects_missing_required_keys(simple_model):
    checkpoint = {
        "checkpoint_version": 2,
        "model_state_dict": simple_model.state_dict(),
    }
    with pytest.raises(ValueError, match="missing required key"):
        model_io.validate_checkpoint(checkpoint)


def test_upgrade_v1_checkpoint_preserves_state_and_merges_config(simple_model):
    original_state = simple_model.state_dict()
    upgraded = model_io.upgrade_v1_checkpoint(
        checkpoint={
            "model_state_dict": original_state,
            "config": {"old_key": "old_value"},
        },
        config={"n_hidden": 64},
        model_type="SimpleModel",
    )
    assert upgraded["checkpoint_version"] == 2
    assert upgraded["model_type"] == "SimpleModel"
    assert upgraded["config"]["n_hidden"] == 64
    assert upgraded["config"]["old_key"] == "old_value"
    assert upgraded["metadata"]["upgraded_from_v1"] is True
    for key, value in original_state.items():
        assert torch.allclose(upgraded["model_state_dict"][key], value)


def test_atom_model_save_model_writes_v2_checkpoint(tmp_path):
    from apnet_pt.AtomModels.ap2_atom_model import AtomModel

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
    checkpoint_path = tmp_path / "atom_model.pt"
    model.save_model(checkpoint_path, metadata={"test": "value"})

    checkpoint = model_io.load_checkpoint(checkpoint_path)
    assert checkpoint["checkpoint_version"] == 2
    assert checkpoint["model_type"] == "AtomMPNN"
    assert checkpoint["config"]["n_message"] == 2
    assert checkpoint["config"]["r_cut"] == 4.0
    assert checkpoint["metadata"]["test"] == "value"
