import math
import os

import pytest
import torch

from apnet_pt.AtomModels.ap2_atom_model import AtomMPNN
from apnet_pt.AtomPairwiseModels.apnet2 import APNet2_MPNN
from apnet_pt.AtomPairwiseModels.apnet2_fused import APNet2_AM_MPNN
from apnet_pt.AtomPairwiseModels.apnet2_parity import checkpoint_score
from apnet_pt.training_tracking import tracked_ddp_worker

MODEL_FACTORIES = [
    lambda **kwargs: APNet2_MPNN(**kwargs),
    lambda **kwargs: APNet2_AM_MPNN(atom_model=AtomMPNN(), **kwargs),
]


@pytest.mark.parametrize("model_factory", MODEL_FACTORIES)
def test_quadrupole_scale_only_changes_charge_quadrupole_term(model_factory):
    common = {
        "qA": torch.tensor([[1.0]]),
        "muA": torch.zeros((1, 3)),
        "quadA": torch.tensor([[[1.0, 0.0, 0.0], [0.0, -0.5, 0.0], [0.0, 0.0, -0.5]]]),
        "qB": torch.tensor([[2.0]]),
        "muB": torch.zeros((1, 3)),
        "quadB": torch.tensor([[[0.5, 0.0, 0.0], [0.0, -0.25, 0.0], [0.0, 0.0, -0.25]]]),
        "e_ABsr_source": torch.tensor([0]),
        "e_ABsr_target": torch.tensor([0]),
        "dR_ang": torch.tensor([2.0]),
        "dR_xyz_ang": torch.tensor([[2.0, 0.0, 0.0]]),
    }
    no_quadrupoles = dict(common)
    no_quadrupoles["quadA"] = torch.zeros_like(common["quadA"])
    no_quadrupoles["quadB"] = torch.zeros_like(common["quadB"])

    model_1 = model_factory(quadrupole_scale=1.0)
    model_15 = model_factory(quadrupole_scale=1.5)
    base = model_1.mtp_elst(**no_quadrupoles)
    q_term_1 = model_1.mtp_elst(**common) - base
    q_term_15 = model_15.mtp_elst(**common) - base

    assert torch.allclose(q_term_15, 1.5 * q_term_1)


@pytest.mark.parametrize("model_factory", MODEL_FACTORIES)
def test_tensorflow_parameter_initialization_matches_keras_defaults(model_factory):
    torch.manual_seed(7)
    model = model_factory(parameter_initialization="tensorflow")
    first_layer = model.readout_layer_elst[0]
    first_layer(torch.zeros((2, 13)))

    fan_in, fan_out = 13, model.n_neuron * 2
    bound = math.sqrt(6.0 / (fan_in + fan_out))
    assert torch.max(torch.abs(first_layer.weight)) <= bound
    assert torch.count_nonzero(first_layer.bias) == 0
    assert torch.min(model.embed_layer.weight) >= -0.05
    assert torch.max(model.embed_layer.weight) <= 0.05
    assert model.get_config()["parameter_initialization"] == "tensorflow"


def test_tensorflow_pair_initialization_does_not_modify_frozen_atom_model():
    atom_model = AtomMPNN()
    before = {
        name: parameter.detach().clone()
        for name, parameter in atom_model.named_parameters()
    }

    APNet2_AM_MPNN(
        atom_model=atom_model,
        parameter_initialization="tensorflow",
    )

    for name, parameter in atom_model.named_parameters():
        assert torch.equal(parameter, before[name])


def test_checkpoint_score_supports_tensorflow_total_mae_policy():
    component_mse = torch.tensor(0.25)
    total_mae = torch.tensor(0.5)

    assert checkpoint_score("component_mse", component_mse, total_mae) is component_mse
    assert checkpoint_score("total_mae", component_mse, total_mae) is total_mae
    with pytest.raises(ValueError, match="checkpoint metric"):
        checkpoint_score("unknown", component_mse, total_mae)


def test_set_all_seeds_can_request_deterministic_algorithms():
    import train_models

    env_names = ("CUBLAS_WORKSPACE_CONFIG", "QCMLFORGE_DETERMINISTIC")
    previous_env = {name: os.environ.get(name) for name in env_names}
    previously_enabled = torch.are_deterministic_algorithms_enabled()
    try:
        train_models.set_all_seeds(4201, deterministic=False)
        assert torch.are_deterministic_algorithms_enabled() is False

        train_models.set_all_seeds(4201, deterministic=True)
        assert torch.are_deterministic_algorithms_enabled() is True
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
        assert os.environ["QCMLFORGE_DETERMINISTIC"] == "1"
    finally:
        torch.use_deterministic_algorithms(previously_enabled)
        for name, value in previous_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def test_ddp_worker_enables_requested_deterministic_algorithms():
    class DeterminismProbe:
        def run(self, rank):
            return rank, torch.are_deterministic_algorithms_enabled()

    previous_env = os.environ.get("QCMLFORGE_DETERMINISTIC")
    previously_enabled = torch.are_deterministic_algorithms_enabled()
    try:
        os.environ["QCMLFORGE_DETERMINISTIC"] = "1"
        torch.use_deterministic_algorithms(False)
        assert tracked_ddp_worker(3, DeterminismProbe().run) == (3, True)
    finally:
        torch.use_deterministic_algorithms(previously_enabled)
        if previous_env is None:
            os.environ.pop("QCMLFORGE_DETERMINISTIC", None)
        else:
            os.environ["QCMLFORGE_DETERMINISTIC"] = previous_env


def test_fused_set_pretrained_model_adopts_quadrupole_scale(tmp_path):
    """Restore the forward-pass scale, which is absent from the state dict."""
    from apnet_pt.AtomPairwiseModels.apnet2_fused import APNet2_AM_Model

    saved = APNet2_AM_MPNN(atom_model=AtomMPNN(), quadrupole_scale=1.5)
    checkpoint_path = tmp_path / "ap2_fused.pt"
    torch.save(
        {
            "model_state_dict": saved.state_dict(),
            "config": saved.get_config(),
        },
        checkpoint_path,
    )

    model = APNet2_AM_Model(
        atom_model=AtomMPNN(), ds_root=None, ignore_database_null=True, use_GPU=False
    )
    assert model.model.quadrupole_scale == 1.0
    model.set_pretrained_model(ap2_model_path=str(checkpoint_path))
    assert model.model.quadrupole_scale == 1.5
