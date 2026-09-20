"""Loading legacy AtomTypeParamMPNN checkpoints that predate the ``r_cut`` config key.

The shipped ``atp_hfvr_*.pt`` / ``atp_elst_*.pt`` weights were written before
``r_cut`` was added to the saved ``config`` dict, so every loader that reads it
must fall back to the caller's ``r_cut`` rather than raising ``KeyError``.
"""

import pytest
import torch

from apnet_pt.AtomModels.ap3_atomtype_mpnn import (
    AtomTypeParamMPNN,
    AtomTypeParamModel,
)
from apnet_pt.AtomModels.ap3_atom_model_frozen import InducedDipoleModel

LEGACY_CONFIG_KEYS = (
    "n_message",
    "n_neuron",
    "n_embed",
    "param_start_mean",
    "param_start_std",
    "n_params",
)


@pytest.fixture
def legacy_checkpoint(tmp_path):
    """A checkpoint whose ``config`` omits ``r_cut``, as the shipped weights do."""
    torch.manual_seed(42)
    model = AtomTypeParamMPNN(
        n_message=2, n_neuron=16, n_embed=4, r_cut=5.0, n_params=2
    )
    config = {
        "n_message": model.n_message,
        "n_neuron": model.n_neuron,
        "n_embed": model.n_embed,
        "param_start_mean": model.param_start_mean,
        "param_start_std": model.param_start_std,
        "n_params": model.n_params,
    }
    assert set(config) == set(LEGACY_CONFIG_KEYS)
    path = tmp_path / "atp_legacy.pt"
    torch.save({"model_state_dict": model.state_dict(), "config": config}, path)
    return path


def test_atomtype_param_model_loads_legacy_checkpoint(legacy_checkpoint):
    harness = AtomTypeParamModel(
        pre_trained_model_path=str(legacy_checkpoint),
        r_cut=5.0,
        ignore_database_null=True,
        use_GPU=False,
    )
    assert harness.model.r_cut == 5.0


def test_atomtype_param_model_legacy_checkpoint_honors_caller_r_cut(legacy_checkpoint):
    harness = AtomTypeParamModel(
        pre_trained_model_path=str(legacy_checkpoint),
        r_cut=7.0,
        ignore_database_null=True,
        use_GPU=False,
    )
    assert harness.model.r_cut == 7.0


def test_induced_dipole_model_loads_legacy_hfvr_checkpoint(legacy_checkpoint):
    harness = InducedDipoleModel(
        atomtype_hfvr_pre_trained_path=str(legacy_checkpoint),
        r_cut=5.0,
        ignore_database_null=True,
        use_GPU=False,
    )
    assert harness.atomtype_hfvr_model.r_cut == 5.0
