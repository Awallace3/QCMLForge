import copy

import pytest
import torch
from torch_geometric.data import Data

from apnet_pt.AtomModels.ap2_atom_model import AtomMPNN
from apnet_pt.AtomPairwiseModels.mtp_mtp import (
    RACKERS_ELST_INDEX,
    RACKERS_IND_OVERLAP_INDEX,
    RACKERS_INITIAL_STDS,
    RACKERS_INITIAL_VALUES,
    RACKERS_PARAMETER_NAMES,
    RACKERS_POSITIVITY_EPSILON,
    RACKERS_THOLE_DIRECT_INDEX,
    RACKERS_THOLE_MUTUAL_INDEX,
    AtomTypeParamNN,
    RackersTholeDampingNN,
    geometric_mean_edge_values,
)
from apnet_pt.pt_datasets.ap2_fused_ds import ap2_fused_collate_update
from apnet_pt.torch_util import set_weights_to_value


def _make_collate_item(y_scale: float) -> Data:
    return Data(
        y=torch.tensor(
            [-1.0, 2.0, -3.0, 4.0], dtype=torch.float32
        ) * y_scale,
        ZA=torch.tensor([8, 1], dtype=torch.long),
        RA=torch.tensor(
            [[0.0, 0.0, 0.0], [0.9, 0.0, 0.0]],
            dtype=torch.float32,
        ),
        ZB=torch.tensor([8, 1], dtype=torch.long),
        RB=torch.tensor(
            [[3.0, 0.0, 0.0], [7.0, 0.0, 0.0]],
            dtype=torch.float32,
        ),
        e_ABsr_source=torch.tensor([0, 1], dtype=torch.long),
        e_ABsr_target=torch.tensor([0, 0], dtype=torch.long),
        e_ABlr_source=torch.tensor([0, 1], dtype=torch.long),
        e_ABlr_target=torch.tensor([1, 1], dtype=torch.long),
        e_AA_source=torch.tensor([0, 1], dtype=torch.long),
        e_AA_target=torch.tensor([1, 0], dtype=torch.long),
        e_BB_source=torch.tensor([0, 1], dtype=torch.long),
        e_BB_target=torch.tensor([1, 0], dtype=torch.long),
        dimer_ind=torch.zeros(2, dtype=torch.long),
        dimer_ind_lr=torch.zeros(2, dtype=torch.long),
        molecule_ind_A=torch.zeros(2, dtype=torch.long),
        molecule_ind_B=torch.zeros(2, dtype=torch.long),
        total_charge_A=torch.tensor(0.0),
        total_charge_B=torch.tensor(0.0),
    )


def test_target_collate_emits_full_edge_domain():
    batch = ap2_fused_collate_update(
        [_make_collate_item(1.0), _make_collate_item(2.0)]
    )

    assert torch.equal(
        batch.e_ABfull_source,
        torch.cat((batch.e_ABsr_source, batch.e_ABlr_source)),
    )
    assert torch.equal(
        batch.e_ABfull_target,
        torch.cat((batch.e_ABsr_target, batch.e_ABlr_target)),
    )
    assert torch.equal(
        batch.dimer_ind_full,
        torch.cat((batch.dimer_ind, batch.dimer_ind_lr)),
    )
    assert batch.e_ABfull_source.numel() == batch.dimer_ind_full.numel()
    assert batch.dimer_ind_full.tolist() == [0, 0, 1, 1, 0, 0, 1, 1]


@pytest.fixture
def atomic_batch() -> Data:
    return Data(
        x=torch.tensor([8, 1, 1], dtype=torch.long),
        R=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.9, 0.0, 0.0],
                [-0.3, 0.8, 0.0],
            ],
            dtype=torch.float32,
        ),
        edge_index=torch.tensor(
            [[0, 1, 0, 2, 1, 2], [1, 0, 2, 0, 2, 1]],
            dtype=torch.long,
        ),
        molecule_ind=torch.zeros(3, dtype=torch.long),
        total_charge=torch.tensor([0.0], dtype=torch.float32),
        natom_per_mol=torch.tensor([3], dtype=torch.long),
    )


@pytest.fixture
def nested_hfvr_vw_model() -> AtomTypeParamNN:
    atom_model = AtomMPNN(
        n_message=1,
        n_rbf=2,
        n_neuron=8,
        n_embed=4,
        r_cut=5.0,
    )
    nested = AtomTypeParamNN(
        atom_model=atom_model,
        n_message=1,
        n_neuron=8,
        n_embed=4,
        param_start_mean=[1.0, 0.4],
        param_start_std=[0.0, 0.0],
        n_params=2,
        freeze_atom_model=False,
    )
    set_weights_to_value(nested, 0.01)
    return nested


def test_rackers_parameter_head_contract(
    atomic_batch, nested_hfvr_vw_model
):
    model = RackersTholeDampingNN(
        atom_model=nested_hfvr_vw_model,
        n_message=1,
        n_neuron=8,
        n_embed=4,
        freeze_atom_model=True,
    )
    with torch.no_grad():
        for head in model.param_readout_layers:
            for readout in head:
                for parameter in readout.parameters():
                    parameter.zero_()

    nested_output = nested_hfvr_vw_model(atomic_batch)
    output = model(atomic_batch)
    parameters = output[-1]

    assert parameters.shape == (3, 4)
    assert torch.isfinite(parameters).all()
    assert torch.all(parameters > 0)
    assert torch.allclose(
        parameters.mean(dim=0),
        torch.tensor([1.8, 0.34, 0.39, 1.8]),
        atol=0.05,
    )
    for wrapped, expected in zip(output[:-1], nested_output):
        assert torch.allclose(wrapped, expected)

    parameters.sum().backward()
    for head in model.param_readout_layers:
        gradients = [
            parameter.grad
            for readout in head
            for parameter in readout.parameters()
        ]
        assert all(gradient is not None for gradient in gradients)
        assert all(torch.isfinite(gradient).all() for gradient in gradients)

    assert all(
        not parameter.requires_grad
        for parameter in model.atom_model.parameters()
    )
    unfrozen = RackersTholeDampingNN(
        atom_model=copy.deepcopy(nested_hfvr_vw_model),
        freeze_atom_model=False,
    )
    assert all(
        parameter.requires_grad
        for parameter in unfrozen.atom_model.parameters()
    )


def test_rackers_parameter_head_freeze_and_validation(
    nested_hfvr_vw_model
):
    frozen = RackersTholeDampingNN(
        atom_model=nested_hfvr_vw_model,
        freeze_atom_model=True,
    )
    assert all(
        not parameter.requires_grad
        for parameter in frozen.atom_model.parameters()
    )

    with pytest.raises(ValueError, match="exactly four"):
        RackersTholeDampingNN(
            atom_model=nested_hfvr_vw_model,
            param_start_mean=[1.8, 0.34, 0.39],
        )
    with pytest.raises(ValueError, match="exactly four"):
        RackersTholeDampingNN(
            atom_model=nested_hfvr_vw_model,
            param_start_std=[0.01],
        )
    with pytest.raises(ValueError, match="AtomTypeParamNN"):
        RackersTholeDampingNN(
            atom_model=AtomMPNN(
                n_message=1,
                n_rbf=2,
                n_neuron=8,
                n_embed=4,
            )
        )


def test_rackers_parameter_head_constants_and_config(
    nested_hfvr_vw_model,
):
    assert RACKERS_PARAMETER_NAMES == (
        "elst",
        "thole_direct",
        "thole_mutual",
        "ind_overlap",
    )
    assert RACKERS_INITIAL_VALUES == (1.8, 0.34, 0.39, 1.8)
    assert RACKERS_INITIAL_STDS == (0.01, 0.01, 0.01, 0.01)
    assert RACKERS_POSITIVITY_EPSILON == 1e-8
    assert (
        RACKERS_ELST_INDEX,
        RACKERS_THOLE_DIRECT_INDEX,
        RACKERS_THOLE_MUTUAL_INDEX,
        RACKERS_IND_OVERLAP_INDEX,
    ) == (0, 1, 2, 3)

    config = RackersTholeDampingNN(
        atom_model=nested_hfvr_vw_model,
    ).get_config()

    assert config["model_type"] == "RackersTholeDampingNN"
    assert config["parameter_names"] == list(RACKERS_PARAMETER_NAMES)
    assert config["param_start_mean"] == list(RACKERS_INITIAL_VALUES)
    assert config["param_start_std"] == list(RACKERS_INITIAL_STDS)
    assert config["positivity_epsilon"] == RACKERS_POSITIVITY_EPSILON
    assert config["nested_atom_model"]["model_type"] == "AtomTypeParamNN"
    assert (
        config["nested_atom_model"]["atom_model"]["model_type"]
        == "AtomMPNN"
    )


def test_geometric_mean_edge_values_contract():
    source = torch.tensor([1.0, 4.0, 9.0], dtype=torch.float64)
    target = torch.tensor([16.0, 25.0], dtype=torch.float64)
    e_source = torch.tensor([0, 1, 2], dtype=torch.long)
    e_target = torch.tensor([1, 0, 1], dtype=torch.long)

    actual = geometric_mean_edge_values(
        source, target, e_source, e_target
    )
    expected = torch.tensor([5.0, 8.0, 15.0], dtype=torch.float64)

    assert torch.equal(actual, expected)
    assert actual.dtype == source.dtype
    assert actual.device == source.device

    exchanged = geometric_mean_edge_values(
        target, source, e_target, e_source
    )
    assert torch.equal(exchanged, expected)


@pytest.mark.parametrize(
    "source,target",
    [
        (
            torch.tensor([1.0, float("nan")]),
            torch.tensor([4.0, 9.0]),
        ),
        (
            torch.tensor([1.0, 4.0]),
            torch.tensor([float("inf"), 9.0]),
        ),
    ],
)
def test_geometric_mean_edge_values_rejects_non_finite(
    source, target
):
    edge = torch.tensor([0, 1], dtype=torch.long)
    with pytest.raises(ValueError, match="finite"):
        geometric_mean_edge_values(source, target, edge, edge)
