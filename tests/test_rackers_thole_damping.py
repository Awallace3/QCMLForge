import copy

import pytest
import torch
from torch_geometric.data import Data

from apnet_pt import constants
from apnet_pt.AtomModels.ap2_atom_model import AtomMPNN
from apnet_pt.AtomPairwiseModels import mtp_mtp
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


def _rackers_kernel_inputs() -> dict[str, torch.Tensor]:
    dtype = torch.float64
    return {
        "ZA": torch.tensor([1, 1], dtype=torch.long),
        "RA": torch.tensor(
            [[0.0, 0.0, 0.0], [1.1, 0.2, 0.0]], dtype=dtype
        ),
        "qA": torch.tensor([0.25, -0.15], dtype=dtype),
        "muA": torch.tensor(
            [[0.02, -0.01, 0.03], [-0.01, 0.03, 0.01]], dtype=dtype
        ),
        "quadA": torch.zeros((2, 3, 3), dtype=dtype),
        "ZB": torch.tensor([1, 1], dtype=torch.long),
        "RB": torch.tensor(
            [[4.0, 0.4, 0.1], [5.2, -0.3, 0.2]], dtype=dtype
        ),
        "qB": torch.tensor([-0.2, 0.1], dtype=dtype),
        "muB": torch.tensor(
            [[-0.03, 0.02, 0.01], [0.01, -0.02, 0.02]], dtype=dtype
        ),
        "quadB": torch.zeros((2, 3, 3), dtype=dtype),
        "e_AB_source": torch.tensor([0, 0, 1, 1], dtype=torch.long),
        "e_AB_target": torch.tensor([0, 1, 0, 1], dtype=torch.long),
        "e_AA_source": torch.tensor([0, 1], dtype=torch.long),
        "e_BB_source": torch.tensor([0, 1], dtype=torch.long),
        "e_AA_target": torch.tensor([1, 0], dtype=torch.long),
        "e_BB_target": torch.tensor([1, 0], dtype=torch.long),
        "hirshfeld_volume_ratio_A": torch.tensor(
            [0.8, 1.1], dtype=dtype
        ),
        "hirshfeld_volume_ratio_B": torch.tensor(
            [0.9, 1.2], dtype=dtype
        ),
        "valence_widths_A": torch.tensor([0.4, 0.6], dtype=dtype),
        "valence_widths_B": torch.tensor([0.5, 0.7], dtype=dtype),
        "thole_direct_A": torch.tensor([0.16, 0.25], dtype=dtype),
        "thole_direct_B": torch.tensor([0.36, 0.49], dtype=dtype),
        "thole_mutual_A": torch.tensor([0.64, 0.81], dtype=dtype),
        "thole_mutual_B": torch.tensor([1.0, 1.21], dtype=dtype),
        "ind_overlap_A": torch.tensor([0.7, 0.9], dtype=dtype),
        "ind_overlap_B": torch.tensor([1.1, 1.3], dtype=dtype),
    }


def test_rackers_kernel_routes_distinct_parameters_and_overlap(monkeypatch):
    inputs = _rackers_kernel_inputs()
    direct_calls = []
    mutual_calls = []
    original_direct = mtp_mtp.thole_damping_direct_torch
    original_mutual = mtp_mtp.thole_damping_mutual_torch

    def record_direct(r_ij, alpha_i, alpha_j, a):
        direct_calls.append(a.detach().clone())
        return original_direct(r_ij, alpha_i, alpha_j, a)

    def record_mutual(r_ij, alpha_i, alpha_j, a):
        mutual_calls.append(a.detach().clone())
        return original_mutual(r_ij, alpha_i, alpha_j, a)

    monkeypatch.setattr(mtp_mtp, "thole_damping_direct_torch", record_direct)
    monkeypatch.setattr(mtp_mtp, "thole_damping_mutual_torch", record_mutual)

    pure_energy = mtp_mtp.rackers_thole_induction(
        **inputs, include_overlap=False, max_iterations=4
    )

    expected_direct = [
        geometric_mean_edge_values(
            inputs["thole_direct_A"],
            inputs["thole_direct_B"],
            inputs["e_AB_source"],
            inputs["e_AB_target"],
        ),
        geometric_mean_edge_values(
            inputs["thole_direct_A"],
            inputs["thole_direct_A"],
            inputs["e_AA_source"],
            inputs["e_AA_target"],
        ),
        geometric_mean_edge_values(
            inputs["thole_direct_B"],
            inputs["thole_direct_B"],
            inputs["e_BB_source"],
            inputs["e_BB_target"],
        ),
    ]
    expected_mutual = [
        geometric_mean_edge_values(
            inputs["thole_mutual_A"],
            inputs["thole_mutual_B"],
            inputs["e_AB_source"],
            inputs["e_AB_target"],
        ),
        geometric_mean_edge_values(
            inputs["thole_mutual_A"],
            inputs["thole_mutual_A"],
            inputs["e_AA_source"],
            inputs["e_AA_target"],
        ),
        geometric_mean_edge_values(
            inputs["thole_mutual_B"],
            inputs["thole_mutual_B"],
            inputs["e_BB_source"],
            inputs["e_BB_target"],
        ),
    ]
    assert len(direct_calls) == 3
    assert len(mutual_calls) == 3
    for actual, expected in zip(direct_calls, expected_direct):
        assert torch.equal(actual, expected)
    for actual, expected in zip(mutual_calls, expected_mutual):
        assert torch.equal(actual, expected)

    monkeypatch.setattr(
        mtp_mtp, "thole_damping_direct_torch", original_direct
    )
    monkeypatch.setattr(
        mtp_mtp, "thole_damping_mutual_torch", original_mutual
    )
    changed_overlap_inputs = {
        **inputs,
        "ind_overlap_A": torch.tensor([8.0, 9.0], dtype=torch.float64),
        "ind_overlap_B": torch.tensor([10.0, 11.0], dtype=torch.float64),
    }
    pure_energy_changed_overlap = mtp_mtp.rackers_thole_induction(
        **changed_overlap_inputs, include_overlap=False, max_iterations=4
    )
    overlap_energy = mtp_mtp.rackers_thole_induction(
        **inputs, include_overlap=True, max_iterations=4
    )
    assert torch.equal(pure_energy, pure_energy_changed_overlap)

    dR_AB, _ = mtp_mtp.get_distances(
        inputs["RA"],
        inputs["RB"],
        inputs["e_AB_source"],
        inputs["e_AB_target"],
    )
    dR_AB = dR_AB / constants.au2ang
    sigma_A = inputs["valence_widths_A"].index_select(
        0, inputs["e_AB_source"]
    )
    sigma_B = inputs["valence_widths_B"].index_select(
        0, inputs["e_AB_target"]
    )
    B_ij = torch.sqrt(1.0 / (sigma_A * sigma_B))
    S_ij = (
        (B_ij * dR_AB) ** 2 / 3.0 + B_ij * dR_AB + 1.0
    ) * torch.exp(-B_ij * dR_AB)
    expected_overlap = (
        inputs["ind_overlap_A"].index_select(
            0, inputs["e_AB_source"]
        )
        * S_ij
        * inputs["ind_overlap_B"].index_select(
            0, inputs["e_AB_target"]
        )
        * constants.h2kcalmol
    )
    assert torch.allclose(
        pure_energy - overlap_energy, expected_overlap, atol=1e-6
    )


def test_rackers_kernel_routes_direct_fields_and_mutual_scf_effect():
    dtype = torch.float64
    alpha_A = torch.tensor([1.0, 1.5], dtype=dtype)
    alpha_B = torch.tensor([0.8, 1.2], dtype=dtype)
    qA = torch.tensor([0.2, -0.1], dtype=dtype)
    qB = torch.tensor([-0.3, 0.15], dtype=dtype)
    muA = torch.tensor(
        [[0.1, 0.0, 0.0], [0.0, 0.1, 0.0]], dtype=dtype
    )
    muB = torch.tensor(
        [[0.0, -0.1, 0.0], [0.0, 0.0, 0.1]], dtype=dtype
    )
    e_AB_source = torch.tensor([0, 1], dtype=torch.long)
    e_AB_target = torch.tensor([0, 1], dtype=torch.long)
    e_AA_source = torch.tensor([0, 1], dtype=torch.long)
    e_AA_target = torch.tensor([1, 0], dtype=torch.long)
    e_BB_source = torch.tensor([0, 1], dtype=torch.long)
    e_BB_target = torch.tensor([1, 0], dtype=torch.long)
    T1_AB = torch.ones((2, 3), dtype=dtype) * 0.2
    T2_AB = torch.eye(3, dtype=dtype).repeat(2, 1, 1) * 0.1
    zero_T1 = torch.zeros((2, 3), dtype=dtype)
    zero_T2 = torch.zeros((2, 3, 3), dtype=dtype)

    helper_args = (
        alpha_A,
        alpha_B,
        qA,
        muA,
        qB,
        muB,
        e_AB_source,
        e_AB_target,
        e_AA_source,
        e_AA_target,
        e_BB_source,
        e_BB_target,
        T1_AB,
        T2_AB,
    )
    base_A, base_B = mtp_mtp._rackers_initial_permanent_fields(
        *helper_args, zero_T1, zero_T2, zero_T1, zero_T2
    )
    direct_AA_A, direct_AA_B = mtp_mtp._rackers_initial_permanent_fields(
        *helper_args, T1_AB, T2_AB, zero_T1, zero_T2
    )
    direct_BB_A, direct_BB_B = mtp_mtp._rackers_initial_permanent_fields(
        *helper_args, zero_T1, zero_T2, T1_AB, T2_AB
    )
    assert not torch.equal(base_A, direct_AA_A)
    assert torch.equal(base_B, direct_AA_B)
    assert torch.equal(base_A, direct_BB_A)
    assert not torch.equal(base_B, direct_BB_B)

    initial_A_before_mutual_update = base_A.clone()
    initial_B_before_mutual_update = base_B.clone()
    zero_mutual_A, zero_mutual_B = mtp_mtp._rackers_scf_update(
        alpha_A,
        alpha_B,
        e_AB_source,
        e_AB_target,
        e_AA_source,
        e_AA_target,
        e_BB_source,
        e_BB_target,
        zero_T2,
        zero_T2,
        zero_T2,
        base_A,
        base_B,
        base_A,
        base_B,
    )
    changed_mutual_A, changed_mutual_B = mtp_mtp._rackers_scf_update(
        alpha_A,
        alpha_B,
        e_AB_source,
        e_AB_target,
        e_AA_source,
        e_AA_target,
        e_BB_source,
        e_BB_target,
        T2_AB,
        T2_AB,
        T2_AB,
        base_A,
        base_B,
        base_A,
        base_B,
    )
    assert torch.equal(base_A, initial_A_before_mutual_update)
    assert torch.equal(base_B, initial_B_before_mutual_update)
    assert not torch.equal(zero_mutual_A, changed_mutual_A)
    assert not torch.equal(zero_mutual_B, changed_mutual_B)


def test_rackers_kernel_routes_tensor_builder_rejects_invalid_type():
    dtype = torch.float64
    edge = torch.tensor([0], dtype=torch.long)
    coordinates = torch.tensor([[0.0, 0.0, 0.0]], dtype=dtype)
    alpha = torch.tensor([1.0], dtype=dtype)
    damping = torch.tensor([0.4], dtype=dtype)

    with pytest.raises(ValueError, match="Invalid Rackers damping type"):
        mtp_mtp._rackers_distance_tensors(
            coordinates,
            coordinates,
            edge,
            edge,
            alpha,
            alpha,
            damping,
            "unsupported",
        )


def test_rackers_kernel_routes_parameter_gradients():
    inputs = _rackers_kernel_inputs()
    parameter_names = (
        "thole_direct_A",
        "thole_direct_B",
        "thole_mutual_A",
        "thole_mutual_B",
        "ind_overlap_A",
        "ind_overlap_B",
    )
    for name in parameter_names:
        inputs[name] = inputs[name].requires_grad_()

    energy = mtp_mtp.rackers_thole_induction(
        **inputs, include_overlap=True, max_iterations=4
    )
    energy.sum().backward()

    for name in parameter_names:
        gradient = inputs[name].grad
        assert gradient is not None, name
        assert torch.isfinite(gradient).all(), name


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
