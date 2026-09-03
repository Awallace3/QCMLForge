import math
import os

import pytest
import torch

from apnet_pt.AtomModels.ap2_atom_model import AtomMPNN
from apnet_pt.AtomPairwiseModels.apnet2 import APNet2_MPNN
from apnet_pt.AtomPairwiseModels.apnet2_fused import APNet2_AM_MPNN
from apnet_pt.AtomPairwiseModels.apnet2_parity import checkpoint_score


@pytest.mark.parametrize(
    "model_factory",
    [
        lambda **kwargs: APNet2_MPNN(**kwargs),
        lambda **kwargs: APNet2_AM_MPNN(atom_model=AtomMPNN(), **kwargs),
    ],
)
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


@pytest.mark.parametrize(
    "model_factory",
    [
        lambda **kwargs: APNet2_MPNN(**kwargs),
        lambda **kwargs: APNet2_AM_MPNN(atom_model=AtomMPNN(), **kwargs),
    ],
)
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


def _global_rng_probe():
    return torch.randint(0, 2**31 - 1, (4,)).tolist()


def test_initialization_policies_consume_different_global_rng_draws():
    """The confound behind the shuffling fix.

    The TensorFlow policy re-initializes the pair dense kernels on top of the
    PyTorch defaults, so it draws a different number of values from the global
    RNG. Anything downstream that reseeds off the global RNG therefore changes
    with the initialization policy.
    """
    probes = {}
    for policy in ("pytorch", "tensorflow"):
        torch.manual_seed(4201)
        APNet2_AM_MPNN(atom_model=AtomMPNN(), parameter_initialization=policy)
        probes[policy] = _global_rng_probe()

    assert probes["pytorch"] != probes["tensorflow"]


def test_seeded_loader_generator_makes_batch_order_independent_of_init_policy():
    """Regression test for the shuffling confound.

    ``single_proc_train`` hands the training loader its own generator. Without
    one, ``RandomSampler`` reseeds from the global torch RNG each epoch, so the
    initialization policy silently changes batch order too.
    """
    from apnet_pt.pt_datasets.ap2_fused_ds import APNet2_fused_DataLoader

    dataset = list(range(64))

    def order(policy, *, seeded):
        torch.manual_seed(4201)
        APNet2_AM_MPNN(atom_model=AtomMPNN(), parameter_initialization=policy)
        kwargs = {}
        if seeded:
            generator = torch.Generator()
            generator.manual_seed(4201)
            kwargs["generator"] = generator
        loader = APNet2_fused_DataLoader(
            dataset=dataset,
            batch_size=8,
            shuffle=True,
            collate_fn=list,
            **kwargs,
        )
        # Two passes, because the leak reappears at every epoch boundary.
        return [list(batch) for _ in range(2) for batch in loader]

    assert order("pytorch", seeded=True) == order("tensorflow", seeded=True)
    # Guard the premise: without the generator the policies really do diverge,
    # so this test would fail to detect a regression if it were vacuous.
    assert order("pytorch", seeded=False) != order("tensorflow", seeded=False)


def test_set_all_seeds_can_request_deterministic_algorithms():
    import train_models

    previously_enabled = torch.are_deterministic_algorithms_enabled()
    try:
        train_models.set_all_seeds(4201, deterministic=False)
        assert torch.are_deterministic_algorithms_enabled() is False

        train_models.set_all_seeds(4201, deterministic=True)
        assert torch.are_deterministic_algorithms_enabled() is True
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    finally:
        torch.use_deterministic_algorithms(previously_enabled)
