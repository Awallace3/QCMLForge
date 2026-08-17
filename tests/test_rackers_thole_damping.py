import copy
import sys

import numpy as np
import pytest
import qcelemental as qcel
import torch
from torch_geometric.data import Data

import train_models
from apnet_pt import constants, model_io
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
    AM_DimerParam_Model,
    AtomTypeParamNN,
    DimerProp,
    RackersTholeDampingModel,
    RackersTholeDampingNN,
    RackersTholeDampingOverlapModel,
    geometric_mean_edge_values,
)
from apnet_pt.pt_datasets.ap2_fused_ds import (
    ap2_fused_collate_update,
    ap2_fused_collate_update_no_target,
    ap3_fused_collate_update,
)
from apnet_pt.pt_datasets.ap3_fused_ds import (
    ap3_fused_collate_update as ap3_ds_fused_collate_update,
    ap3_fused_collate_update_no_target as ap3_ds_fused_collate_update_no_target,
)
from apnet_pt.torch_util import set_weights_to_value
from apnet_pt.util import scatter_sum_compile


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


@pytest.mark.parametrize(
    "collate_fn",
    [
        ap2_fused_collate_update,
        ap2_fused_collate_update_no_target,
        ap3_fused_collate_update,
        ap3_ds_fused_collate_update,
        ap3_ds_fused_collate_update_no_target,
    ],
    ids=lambda fn: f"{fn.__module__.rsplit('.', 1)[-1]}.{fn.__name__}",
)
def test_full_edge_dimer_index_aligns_with_full_edge_lists(collate_fn):
    """`dimer_ind_full[k]` must identify the dimer owning full edge `k`.

    Both interleaved (per-item ``[sr_i, lr_i]``) and grouped (all short-range
    then all long-range) layouts satisfy this; what must never happen is
    ``e_ABfull_*`` using one layout while ``dimer_ind_full`` uses the other,
    which silently attributes edges to the wrong dimer for batches > 1.
    Monomer A and B are given different atom counts so a transposed or
    mis-grouped index cannot coincidentally line up.
    """
    items = []
    for scale in (1.0, 2.0, 3.0):
        item = _make_collate_item(scale)
        item.ZB = torch.tensor([8, 1, 1], dtype=torch.long)
        item.RB = torch.tensor(
            [[3.0, 0.0, 0.0], [7.0, 0.0, 0.0], [9.0, 0.0, 0.0]],
            dtype=torch.float32,
        )
        item.molecule_ind_B = torch.zeros(3, dtype=torch.long)
        item.e_BB_source = torch.tensor([0, 1, 2], dtype=torch.long)
        item.e_BB_target = torch.tensor([1, 2, 0], dtype=torch.long)
        # Three short-range and one long-range AB edge, so the two layouts
        # produce genuinely different orderings.
        item.e_ABsr_source = torch.tensor([0, 1, 1], dtype=torch.long)
        item.e_ABsr_target = torch.tensor([0, 0, 2], dtype=torch.long)
        item.e_ABlr_source = torch.tensor([0], dtype=torch.long)
        item.e_ABlr_target = torch.tensor([1], dtype=torch.long)
        item.dimer_ind = torch.zeros(3, dtype=torch.long)
        item.dimer_ind_lr = torch.zeros(1, dtype=torch.long)
        items.append(item)

    batch = collate_fn(items)

    n_full = batch.e_ABfull_source.numel()
    assert n_full == batch.e_ABfull_target.numel()
    assert n_full == batch.dimer_ind_full.numel()
    assert n_full == batch.dimer_ind.numel() + batch.dimer_ind_lr.numel()

    # molecule_ind_A/B map each globally offset atom back to its batch item,
    # giving a layout-independent ground truth for every full edge.
    expected_from_source = batch.molecule_ind_A[batch.e_ABfull_source]
    expected_from_target = batch.molecule_ind_B[batch.e_ABfull_target]
    assert torch.equal(batch.dimer_ind_full, expected_from_source)
    assert torch.equal(batch.dimer_ind_full, expected_from_target)
    assert batch.dimer_ind_full.unique().tolist() == [0, 1, 2]


@pytest.fixture
def synthetic_dimer_batch() -> Data:
    items = [_make_collate_item(1.0), _make_collate_item(2.0)]
    for item in items:
        item.RB = torch.tensor(
            [[1.8, 0.3, 0.0], [2.7, -0.2, 0.0]],
            dtype=torch.float32,
        )
    return ap2_fused_collate_update(items)


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
def synthetic_qcel_dimers():
    first = qcel.models.Molecule.from_data("""
0 1
O  0.000000  0.000000  0.000000
H  0.758602  0.000000  0.504284
H -0.260455  0.000000 -0.872893
--
0 1
O  3.000000  0.500000  0.000000
H  3.758602  0.500000  0.504284
H  2.739545  0.500000 -0.872893
units angstrom
""")
    second = qcel.models.Molecule.from_data("""
0 1
O  0.000000  0.000000  0.000000
H  0.758602  0.000000  0.504284
H -0.260455  0.000000 -0.872893
--
0 1
O  3.500000 -0.250000  0.100000
H  4.258602 -0.250000  0.604284
H  3.239545 -0.250000 -0.772893
units angstrom
""")
    return [first, second]


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


@pytest.mark.parametrize(
    "harness_type,expected_mode",
    [
        (RackersTholeDampingModel, "rackers_thole"),
        (
            RackersTholeDampingOverlapModel,
            "rackers_thole_overlap",
        ),
    ],
)
def test_rackers_harness_contract(
    harness_type, expected_mode, nested_hfvr_vw_model
):
    harness = harness_type(
        atom_model=copy.deepcopy(nested_hfvr_vw_model),
        dataset=None,
        ignore_database_null=True,
        use_GPU=False,
        n_message=1,
        n_neuron=8,
        n_embed=4,
    )

    assert type(harness.model) is RackersTholeDampingNN
    assert harness.dimer_eval_type == expected_mode
    assert harness.n_params == 4
    assert harness.model.n_params == 4
    assert all(
        not parameter.requires_grad
        for parameter in harness.model.atom_model.parameters()
    )

    for keyword in ("param_start_mean", "param_start_std"):
        with pytest.raises(ValueError, match="exactly four"):
            harness_type(
                atom_model=copy.deepcopy(nested_hfvr_vw_model),
                dataset=None,
                ignore_database_null=True,
                use_GPU=False,
                **{keyword: [0.1, 0.2, 0.3]},
            )


@pytest.mark.parametrize(
    "harness_type",
    [RackersTholeDampingModel, RackersTholeDampingOverlapModel],
)
def test_rackers_harness_large_initialization_boundaries(
    harness_type, nested_hfvr_vw_model
):
    harness = harness_type(
        atom_model=copy.deepcopy(nested_hfvr_vw_model),
        dataset=None,
        ignore_database_null=True,
        use_GPU=False,
        n_message=1,
        n_neuron=8,
        n_embed=4,
        param_start_mean=[1000.0, 1.0, 1.0, 1.0],
        param_start_std=[0.0, 0.0, 0.0, 0.0],
    )
    assert all(
        torch.isfinite(layer.weight).all()
        for layer in harness.model.guess_layer
    )

    invalid_values = (
        (
            "param_start_mean",
            [1e39, 1.0, 1.0, 1.0],
            "transformed param_start_mean values must be finite and representable",
        ),
        (
            "param_start_std",
            [1e39, 0.0, 0.0, 0.0],
            "param_start_std values must be representable",
        ),
    )
    for field, values, match in invalid_values:
        with pytest.raises(ValueError, match=match):
            harness_type(
                atom_model=copy.deepcopy(nested_hfvr_vw_model),
                dataset=None,
                ignore_database_null=True,
                use_GPU=False,
                **{field: values},
            )


@pytest.mark.parametrize(
    "harness_type",
    [RackersTholeDampingModel, RackersTholeDampingOverlapModel],
)
@pytest.mark.parametrize("freeze_atom_model", [True, False])
def test_rackers_harness_freeze_round_trip(
    tmp_path,
    harness_type,
    freeze_atom_model,
    nested_hfvr_vw_model,
):
    harness = harness_type(
        atom_model=copy.deepcopy(nested_hfvr_vw_model),
        dataset=None,
        ignore_database_null=True,
        use_GPU=False,
        n_message=1,
        n_neuron=8,
        n_embed=4,
        freeze_atom_model=freeze_atom_model,
    )
    assert all(
        parameter.requires_grad is not freeze_atom_model
        for parameter in harness.model.atom_model.parameters()
    )

    path = tmp_path / "freeze-round-trip.pt"
    harness.save_model(path)
    loaded = harness_type(
        pre_trained_model_path=path,
        atom_model=None,
        dataset=None,
        ignore_database_null=True,
        use_GPU=False,
        freeze_atom_model=freeze_atom_model,
    )
    assert all(
        parameter.requires_grad is not freeze_atom_model
        for parameter in loaded.model.atom_model.parameters()
    )
    assert loaded.atom_model is loaded.model.atom_model
    assert loaded.dimer_model.AtomTypeParam is loaded.model


@pytest.mark.parametrize(
    "harness_type,expected_mode",
    [
        (RackersTholeDampingModel, "rackers_thole"),
        (
            RackersTholeDampingOverlapModel,
            "rackers_thole_overlap",
        ),
    ],
)
def test_rackers_checkpoint_round_trip(
    tmp_path,
    harness_type,
    expected_mode,
    nested_hfvr_vw_model,
    synthetic_qcel_dimers,
):
    harness = harness_type(
        atom_model=copy.deepcopy(nested_hfvr_vw_model),
        dataset=None,
        ignore_database_null=True,
        use_GPU=False,
        n_message=1,
        n_neuron=8,
        n_embed=4,
    )
    before = harness.predict_qcel_mols_dimer(
        synthetic_qcel_dimers, batch_size=2
    )

    checkpoint_path = tmp_path / f"{expected_mode}.pt"
    harness.save_model(checkpoint_path)

    checkpoint = model_io.load_checkpoint(checkpoint_path)
    assert checkpoint["model_type"] == "RackersTholeDampingNN"
    assert checkpoint["config"]["parameter_names"] == list(
        RACKERS_PARAMETER_NAMES
    )
    assert checkpoint["config"]["dimer_eval"] == expected_mode

    loaded = harness_type(
        pre_trained_model_path=checkpoint_path,
        atom_model=None,
        dataset=None,
        ignore_database_null=True,
        use_GPU=False,
    )
    assert (loaded.model.n_message, loaded.model.n_neuron, loaded.model.n_embed) == (
        1,
        8,
        4,
    )
    after = loaded.predict_qcel_mols_dimer(
        synthetic_qcel_dimers, batch_size=2
    )
    assert np.allclose(before, after, atol=1e-6)

    second_path = tmp_path / f"{expected_mode}-second.pt"
    loaded.save_model(second_path)
    reloaded = harness_type(
        pre_trained_model_path=second_path,
        atom_model=None,
        dataset=None,
        ignore_database_null=True,
        use_GPU=False,
    )
    assert reloaded.model.get_config() == loaded.model.get_config()
    second_predictions = reloaded.predict_qcel_mols_dimer(
        synthetic_qcel_dimers, batch_size=2
    )
    assert np.allclose(after, second_predictions, atol=1e-6)


@pytest.mark.parametrize(
    "harness_type,expected_mode",
    [
        (RackersTholeDampingModel, "rackers_thole"),
        (
            RackersTholeDampingOverlapModel,
            "rackers_thole_overlap",
        ),
    ],
)
def test_rackers_compiled_checkpoint_round_trip(
    tmp_path,
    harness_type,
    expected_mode,
    nested_hfvr_vw_model,
    synthetic_qcel_dimers,
):
    harness = harness_type(
        atom_model=copy.deepcopy(nested_hfvr_vw_model),
        dataset=None,
        ignore_database_null=True,
        use_GPU=False,
        n_message=1,
        n_neuron=8,
        n_embed=4,
    )
    before = harness.predict_qcel_mols_dimer(
        synthetic_qcel_dimers, batch_size=2
    )
    harness.model = torch.compile(harness.model, backend="eager")

    path = tmp_path / f"compiled-{expected_mode}.pt"
    harness.save_model(path)
    checkpoint = model_io.load_checkpoint(path)

    assert checkpoint["model_type"] == "RackersTholeDampingNN"
    assert checkpoint["config"]["dimer_eval"] == expected_mode
    assert all(
        not key.startswith("_orig_mod.")
        for key in checkpoint["model_state_dict"]
    )
    loaded = harness_type(
        pre_trained_model_path=path,
        atom_model=None,
        dataset=None,
        ignore_database_null=True,
        use_GPU=False,
    )
    after = loaded.predict_qcel_mols_dimer(
        synthetic_qcel_dimers, batch_size=2
    )
    assert np.allclose(before, after, atol=1e-6)


@pytest.mark.parametrize(
    "tamper,match",
    [
        (
            lambda checkpoint: checkpoint["config"].__setitem__(
                "parameter_names", list(reversed(RACKERS_PARAMETER_NAMES))
            ),
            "parameter_names",
        ),
        (
            lambda checkpoint: checkpoint["config"].pop("parameter_names"),
            "parameter_names",
        ),
        (
            lambda checkpoint: checkpoint.__setitem__(
                "model_type", "AtomTypeParamNN"
            ),
            "model_type",
        ),
        (
            lambda checkpoint: checkpoint.pop("checkpoint_version"),
            "checkpoint_version",
        ),
        (
            lambda checkpoint: checkpoint.__setitem__(
                "checkpoint_version", 1
            ),
            "checkpoint_version",
        ),
        (
            lambda checkpoint: checkpoint["config"].pop(
                "nested_atom_model"
            ),
            "nested_atom_model",
        ),
        (
            lambda checkpoint: checkpoint["config"][
                "nested_atom_model"
            ].__setitem__("model_type", "UnsupportedNestedModel"),
            "Unsupported nested atom model type",
        ),
        (
            lambda checkpoint: checkpoint["config"][
                "param_start_mean"
            ].__setitem__(0, 0.0),
            "param_start_mean values must be finite and strictly greater",
        ),
        (
            lambda checkpoint: checkpoint["config"][
                "param_start_std"
            ].__setitem__(1, float("inf")),
            "param_start_std values must be finite and greater than or equal",
        ),
        (
            lambda checkpoint: checkpoint["config"].__setitem__(
                "positivity_epsilon", float("nan")
            ),
            "positivity_epsilon must be finite and strictly greater than zero",
        ),
        (
            lambda checkpoint: checkpoint["config"][
                "param_start_mean"
            ].__setitem__(0, 1e39),
            "transformed param_start_mean values must be finite and representable",
        ),
        (
            lambda checkpoint: checkpoint["config"][
                "param_start_std"
            ].__setitem__(0, 1e39),
            "param_start_std values must be representable",
        ),
    ],
)
def test_rackers_checkpoint_rejects_invalid_metadata(
    tmp_path, nested_hfvr_vw_model, tamper, match
):
    harness = RackersTholeDampingModel(
        atom_model=copy.deepcopy(nested_hfvr_vw_model),
        dataset=None,
        ignore_database_null=True,
        use_GPU=False,
        n_message=1,
        n_neuron=8,
        n_embed=4,
    )
    checkpoint = harness._create_checkpoint()
    tamper(checkpoint)
    path = tmp_path / "tampered.pt"
    model_io.save_checkpoint(checkpoint, path)

    with pytest.raises(ValueError, match=match):
        RackersTholeDampingModel(
            pre_trained_model_path=path,
            atom_model=None,
            dataset=None,
            ignore_database_null=True,
            use_GPU=False,
        )


@pytest.mark.parametrize(
    "source_type,destination_type",
    [
        (RackersTholeDampingModel, RackersTholeDampingOverlapModel),
        (RackersTholeDampingOverlapModel, RackersTholeDampingModel),
    ],
)
def test_rackers_checkpoint_rejects_wrong_harness_mode(
    tmp_path, nested_hfvr_vw_model, source_type, destination_type
):
    harness = source_type(
        atom_model=copy.deepcopy(nested_hfvr_vw_model),
        dataset=None,
        ignore_database_null=True,
        use_GPU=False,
    )
    path = tmp_path / "wrong-mode.pt"
    harness.save_model(path)

    with pytest.raises(ValueError, match="dimer_eval"):
        destination_type(
            pre_trained_model_path=path,
            atom_model=None,
            dataset=None,
            ignore_database_null=True,
            use_GPU=False,
        )


class _FullEdgeTrainModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, batch):
        output = self.scale * torch.ones(
            (batch.dimer_ind_full.numel(), 2),
            dtype=self.scale.dtype,
            device=self.scale.device,
        )
        return (output,)


@pytest.mark.parametrize(
    "mode", ["rackers_thole", "rackers_thole_overlap"]
)
def test_rackers_training_uses_full_edge_aggregation(
    mode, synthetic_dimer_batch
):
    synthetic_dimer_batch.dimer_ind = torch.zeros(1, dtype=torch.long)
    synthetic_dimer_batch.y = torch.zeros((2, 4), dtype=torch.float32)
    harness = AM_DimerParam_Model.__new__(AM_DimerParam_Model)
    harness.model = _FullEdgeTrainModel()
    harness.dimer_model = harness.model
    harness.dimer_eval_type = mode
    optimizer = torch.optim.SGD(harness.model.parameters(), lr=0.01)

    train_result = harness._AM_DimerParam_Model__train_batches_single_proc(
        [synthetic_dimer_batch],
        loss_fn=torch.nn.MSELoss(),
        optimizer=optimizer,
        rank_device=torch.device("cpu"),
        scheduler=None,
        y_ind=torch.tensor([0, 2]),
    )
    eval_result = harness._AM_DimerParam_Model__evaluate_batches_single_proc(
        [synthetic_dimer_batch],
        loss_fn=torch.nn.MSELoss(),
        rank_device=torch.device("cpu"),
        y_ind=torch.tensor([0, 2]),
    )

    assert np.isfinite(train_result[0])
    assert torch.isfinite(train_result[1]).all()
    assert np.isfinite(eval_result[0])
    assert torch.isfinite(eval_result[1]).all()


@pytest.mark.parametrize(
    "harness_type",
    [RackersTholeDampingModel, RackersTholeDampingOverlapModel],
)
def test_rackers_default_training_preserves_hierarchy_and_checkpoint(
    tmp_path,
    harness_type,
    nested_hfvr_vw_model,
    synthetic_dimer_batch,
    synthetic_qcel_dimers,
    monkeypatch,
):
    harness = harness_type(
        atom_model=copy.deepcopy(nested_hfvr_vw_model),
        dataset=None,
        ignore_database_null=True,
        use_GPU=False,
        n_message=1,
        n_neuron=8,
        n_embed=4,
    )
    harness.example_input = lambda: synthetic_dimer_batch.batch_atomic_A
    harness.compile_model = lambda: setattr(
        harness,
        "model",
        torch.compile(harness.model, backend="eager"),
    )
    selected_targets = []
    original_train_batches = (
        harness._AM_DimerParam_Model__train_batches_single_proc
    )

    def record_train_targets(*args, **kwargs):
        selected_targets.append(kwargs["y_ind"].detach().cpu().clone())
        return original_train_batches(*args, **kwargs)

    monkeypatch.setattr(
        harness,
        "_AM_DimerParam_Model__train_batches_single_proc",
        record_train_targets,
    )
    optimizer_parameter_ids = set()
    original_adam = torch.optim.Adam

    def record_optimizer(parameters, *args, **kwargs):
        parameter_list = list(parameters)
        optimizer_parameter_ids.update(map(id, parameter_list))
        return original_adam(parameter_list, *args, **kwargs)

    monkeypatch.setattr(torch.optim, "Adam", record_optimizer)
    harness.model_save_path = tmp_path / "default-training.pt"
    train_item = _make_collate_item(1.0)
    test_item = _make_collate_item(1.1)

    harness.single_proc_train(
        train_dataset=[train_item],
        test_dataset=[test_item],
        n_epochs=1,
        batch_size=1,
        lr=1e-5,
        pin_memory=False,
        num_workers=0,
    )

    underlying_model = model_io.unwrap_model(harness.model)
    assert len(selected_targets) == 1
    assert torch.equal(selected_targets[0], torch.tensor([0, 2]))
    assert optimizer_parameter_ids == {
        id(parameter) for parameter in underlying_model.parameters()
    }
    assert underlying_model is harness.dimer_model.AtomTypeParam
    assert harness.atom_model is underlying_model.atom_model

    before = harness.predict_qcel_mols_dimer(
        synthetic_qcel_dimers, batch_size=2
    )
    checkpoint = model_io.load_checkpoint(harness.model_save_path)
    assert checkpoint["model_type"] == "RackersTholeDampingNN"
    assert "dimer_eval" in checkpoint["config"]
    assert all(
        not key.startswith("_orig_mod.")
        for key in checkpoint["model_state_dict"]
    )
    loaded = harness_type(
        pre_trained_model_path=harness.model_save_path,
        atom_model=None,
        dataset=None,
        ignore_database_null=True,
        use_GPU=False,
    )
    after = loaded.predict_qcel_mols_dimer(
        synthetic_qcel_dimers, batch_size=2
    )
    assert np.allclose(before, after, atol=1e-6)


@pytest.mark.parametrize(
    "mode,expected_index",
    [
        ("rackers_thole", "dimer_ind_full"),
        ("rackers_thole_overlap", "dimer_ind_full"),
        ("elst_damping", "dimer_ind"),
    ],
)
def test_dimer_aggregation_selector_preserves_legacy_short_edges(
    mode, expected_index, synthetic_dimer_batch
):
    harness = AM_DimerParam_Model.__new__(AM_DimerParam_Model)
    harness.dimer_eval_type = mode
    selected = harness._dimer_index_for_output(synthetic_dimer_batch)

    assert selected is getattr(synthetic_dimer_batch, expected_index)
    if mode.startswith("rackers"):
        dimer = DimerProp(
            ATParam=_ControlledRackersAtomParam(), dimer_eval=mode
        )
        edge_output = dimer(synthetic_dimer_batch)[0]
        assert synthetic_dimer_batch.e_ABlr_source.numel() > 0
    else:
        edge_output = torch.empty(
            synthetic_dimer_batch.e_ABsr_source.numel(), 1
        )
    assert edge_output.size(0) == selected.numel()


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


@pytest.mark.parametrize(
    "positivity_epsilon",
    [0.0, -1e-8, float("nan"), float("inf"), -float("inf")],
)
def test_rackers_initialization_rejects_invalid_positivity_epsilon(
    nested_hfvr_vw_model, positivity_epsilon
):
    with pytest.raises(
        ValueError,
        match="positivity_epsilon must be finite and strictly greater than zero",
    ):
        RackersTholeDampingNN(
            atom_model=nested_hfvr_vw_model,
            positivity_epsilon=positivity_epsilon,
        )


@pytest.mark.parametrize(
    "invalid_mean",
    [0.0, -0.1, 1e-8, float("nan"), float("inf"), -float("inf")],
)
def test_rackers_initialization_rejects_invalid_means(
    nested_hfvr_vw_model, invalid_mean
):
    means = list(RACKERS_INITIAL_VALUES)
    means[1] = invalid_mean
    with pytest.raises(
        ValueError,
        match="param_start_mean values must be finite and strictly greater",
    ):
        RackersTholeDampingNN(
            atom_model=nested_hfvr_vw_model,
            param_start_mean=means,
        )


@pytest.mark.parametrize(
    "invalid_std",
    [-0.1, float("nan"), float("inf"), -float("inf")],
)
def test_rackers_initialization_rejects_invalid_raw_stds(
    nested_hfvr_vw_model, invalid_std
):
    stds = list(RACKERS_INITIAL_STDS)
    stds[2] = invalid_std
    with pytest.raises(
        ValueError,
        match="param_start_std values must be finite and greater than or equal",
    ):
        RackersTholeDampingNN(
            atom_model=nested_hfvr_vw_model,
            param_start_std=stds,
        )


def test_rackers_initialization_accepts_valid_custom_exact_four_values(
    nested_hfvr_vw_model,
):
    means = [0.25, 0.5, 0.75, 1.0]
    stds = [0.0, 0.02, 0.0, 0.04]
    epsilon = 1e-6
    model = RackersTholeDampingNN(
        atom_model=nested_hfvr_vw_model,
        param_start_mean=means,
        param_start_std=stds,
        positivity_epsilon=epsilon,
    )

    config = model.get_config()
    assert config["param_start_mean"] == means
    assert config["param_start_std"] == stds
    assert config["positivity_epsilon"] == epsilon
    assert torch.isfinite(torch.tensor(model.raw_param_start_mean)).all()


def test_rackers_initialization_accepts_large_representable_mean(
    atomic_batch, nested_hfvr_vw_model
):
    model = RackersTholeDampingNN(
        atom_model=nested_hfvr_vw_model,
        n_message=1,
        n_neuron=8,
        n_embed=4,
        param_start_mean=[1000.0, 1.0, 1.0, 1.0],
        param_start_std=[0.0, 0.0, 0.0, 0.0],
    )

    parameters = model(atomic_batch)[-1]
    assert torch.isfinite(torch.tensor(model.raw_param_start_mean)).all()
    assert torch.isfinite(parameters).all()
    assert torch.all(parameters > 0)


@pytest.mark.parametrize(
    "field,values,match",
    [
        (
            "param_start_mean",
            [1e39, 1.0, 1.0, 1.0],
            "transformed param_start_mean values must be finite and representable",
        ),
        (
            "param_start_std",
            [1e39, 0.0, 0.0, 0.0],
            "param_start_std values must be representable",
        ),
    ],
)
def test_rackers_initialization_rejects_embedding_dtype_overflow(
    nested_hfvr_vw_model, field, values, match
):
    kwargs = {
        "param_start_mean": list(RACKERS_INITIAL_VALUES),
        "param_start_std": list(RACKERS_INITIAL_STDS),
    }
    kwargs[field] = values
    with pytest.raises(ValueError, match=match):
        RackersTholeDampingNN(
            atom_model=nested_hfvr_vw_model,
            **kwargs,
        )


def test_rackers_initialization_rejects_generated_non_finite_embedding(
    monkeypatch, nested_hfvr_vw_model
):
    def non_finite_noise(tensor):
        return torch.full_like(tensor, float("inf"))

    monkeypatch.setattr(torch, "randn_like", non_finite_noise)
    with pytest.raises(
        ValueError,
        match="Rackers embedding initialization produced non-finite parameters",
    ):
        RackersTholeDampingNN(
            atom_model=nested_hfvr_vw_model,
            param_start_std=[0.0, 0.0, 0.0, 0.0],
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


def _analytic_rackers_tensors(
    Ri,
    Rj,
    e_source,
    e_target,
    alpha_i,
    alpha_j,
    damping,
    damping_type,
):
    displacement = (
        Rj.index_select(0, e_target) - Ri.index_select(0, e_source)
    ) / constants.au2ang
    distance = torch.linalg.vector_norm(displacement, dim=1)
    alpha_source = alpha_i.index_select(0, e_source)
    alpha_target = alpha_j.index_select(0, e_target)
    u = distance / ((alpha_source * alpha_target) ** (1.0 / 6.0))
    if damping_type == "direct":
        au3 = damping * u ** (3.0 / 2.0)
        lam_3 = 1.0 - torch.exp(-au3)
        lam_5 = 1.0 - (1.0 + 0.5 * au3) * torch.exp(-au3)
    else:
        assert damping_type == "mutual"
        au3 = damping * u**3
        lam_3 = 1.0 - torch.exp(-au3)
        lam_5 = 1.0 - (1.0 + au3) * torch.exp(-au3)

    inverse_distance = 1.0 / distance
    T1 = (
        -(inverse_distance**3 * lam_3).unsqueeze(1) * displacement
    )
    displacement_outer = torch.einsum(
        "ai,aj->aij", displacement, displacement
    )
    identity = torch.eye(3, dtype=distance.dtype, device=distance.device)
    T2 = (
        3.0 * displacement_outer * lam_5[:, None, None]
        - distance.square()[:, None, None]
        * identity[None, :, :]
        * lam_3[:, None, None]
    ) * inverse_distance.pow(5)[:, None, None]
    return displacement, T1, T2


class _ControlledRackersAtomParam(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.atom_model = torch.nn.Identity()
        self.batch_calls = []

    def forward(self, batch):
        self.batch_calls.append(batch)
        atom_index = torch.arange(
            batch.x.numel(), dtype=batch.R.dtype, device=batch.x.device
        )
        charge = 0.1 + 0.05 * atom_index
        dipole = torch.stack(
            (charge, charge + 0.1, charge + 0.2), dim=1
        )
        quadrupole = torch.zeros(
            (batch.x.numel(), 3, 3),
            dtype=batch.R.dtype,
            device=batch.x.device,
        )
        hfvr_vw = torch.stack(
            (-(0.8 + 0.1 * atom_index), 0.4 + 0.05 * atom_index),
            dim=1,
        )
        # Stack along dim=1 so the sentinel matches the production
        # [n_atoms, 4] parameter-head contract for any atom count. Column 0 is
        # negative and column 1 is below the positivity epsilon so that the
        # dimer forward pass is shown to pass raw head values through without
        # re-clamping them.
        atom_scale = 1.0 + 0.1 * atom_index
        parameters = torch.stack(
            (
                -atom_scale,
                RACKERS_POSITIVITY_EPSILON * atom_scale / 10.0,
                atom_scale + 0.25,
                atom_scale + 1.25,
            ),
            dim=1,
        )
        return charge, dipole, quadrupole, hfvr_vw, parameters


@pytest.mark.parametrize(
    "mode,include_overlap",
    [("rackers_thole", False), ("rackers_thole_overlap", True)],
)
@pytest.mark.parametrize("elst_damping_type", ["CLIFF", "AMOEBA"])
def test_rackers_dimer_forward_routes_columns_edges_and_preserves_charge(
    mode,
    include_overlap,
    elst_damping_type,
    synthetic_dimer_batch,
    monkeypatch,
):
    electrostatic_calls = {"CLIFF": [], "AMOEBA": []}
    induction_calls = []

    def electrostatic_stub(damping_type):
        def evaluate(**kwargs):
            electrostatic_calls[damping_type].append(
                {
                    key: value.detach().clone()
                    for key, value in kwargs.items()
                    if isinstance(value, torch.Tensor)
                }
            )
            kwargs["qA_0"].add_(100.0)
            kwargs["qB_0"].sub_(100.0)
            return torch.ones_like(kwargs["e_AB_source"], dtype=kwargs["RA"].dtype)

        return evaluate

    def induction_stub(**kwargs):
        induction_calls.append(
            {
                key: value.detach().clone()
                if isinstance(value, torch.Tensor)
                else value
                for key, value in kwargs.items()
            }
        )
        return torch.full_like(
            kwargs["e_AB_source"], 2.0, dtype=kwargs["RA"].dtype
        )

    monkeypatch.setattr(
        mtp_mtp, "mtp_elst_damping", electrostatic_stub("CLIFF")
    )
    monkeypatch.setattr(
        mtp_mtp,
        "mtp_elst_damping_AMOEBA",
        electrostatic_stub("AMOEBA"),
    )
    monkeypatch.setattr(
        mtp_mtp, "rackers_thole_induction", induction_stub
    )

    atom_parameters = _ControlledRackersAtomParam()
    dimer = DimerProp(
        ATParam=atom_parameters,
        dimer_eval=mode,
        elst_damping_type=elst_damping_type,
    )
    edge_energy, output_A, output_B = dimer(synthetic_dimer_batch)

    assert len(atom_parameters.batch_calls) == 2
    assert sum(
        call is synthetic_dimer_batch.batch_atomic_A
        for call in atom_parameters.batch_calls
    ) == 1
    assert sum(
        call is synthetic_dimer_batch.batch_atomic_B
        for call in atom_parameters.batch_calls
    ) == 1
    for output in (output_A, output_B):
        assert output[-1].shape == (output[0].numel(), 4)
        assert torch.all(output[-1][:, 0] < 0)
        assert torch.all(output[-1][:, 1] > 0)
        assert torch.all(
            output[-1][:, 1] < RACKERS_POSITIVITY_EPSILON
        )

    other_damping_type = (
        "AMOEBA" if elst_damping_type == "CLIFF" else "CLIFF"
    )
    assert len(electrostatic_calls[elst_damping_type]) == 1
    assert electrostatic_calls[other_damping_type] == []
    assert len(induction_calls) == 1
    electrostatic = electrostatic_calls[elst_damping_type][0]
    induction = induction_calls[0]

    assert torch.equal(
        electrostatic["Ka"], output_A[-1][:, RACKERS_ELST_INDEX]
    )
    assert torch.equal(
        electrostatic["Kb"], output_B[-1][:, RACKERS_ELST_INDEX]
    )
    assert torch.equal(
        induction["thole_direct_A"],
        output_A[-1][:, RACKERS_THOLE_DIRECT_INDEX],
    )
    assert torch.equal(
        induction["thole_direct_B"],
        output_B[-1][:, RACKERS_THOLE_DIRECT_INDEX],
    )
    assert torch.equal(
        induction["thole_mutual_A"],
        output_A[-1][:, RACKERS_THOLE_MUTUAL_INDEX],
    )
    assert torch.equal(
        induction["thole_mutual_B"],
        output_B[-1][:, RACKERS_THOLE_MUTUAL_INDEX],
    )
    assert torch.equal(
        induction["ind_overlap_A"],
        output_A[-1][:, RACKERS_IND_OVERLAP_INDEX],
    )
    assert torch.equal(
        induction["ind_overlap_B"],
        output_B[-1][:, RACKERS_IND_OVERLAP_INDEX],
    )
    assert torch.equal(
        induction["hirshfeld_volume_ratio_A"],
        output_A[-2][:, 0].abs(),
    )
    assert torch.equal(
        induction["hirshfeld_volume_ratio_B"],
        output_B[-2][:, 0].abs(),
    )
    assert torch.equal(
        induction["valence_widths_A"], output_A[-2][:, 1]
    )
    assert torch.equal(
        induction["valence_widths_B"], output_B[-2][:, 1]
    )
    assert induction["include_overlap"] is include_overlap
    assert torch.equal(induction["qA"], electrostatic["qA_0"])
    assert torch.equal(induction["qB"], electrostatic["qB_0"])
    assert torch.equal(induction["qA"], output_A[0])
    assert torch.equal(induction["qB"], output_B[0])
    for call in (electrostatic, induction):
        assert torch.equal(
            call["e_AB_source"], synthetic_dimer_batch.e_ABfull_source
        )
        assert torch.equal(
            call["e_AB_target"], synthetic_dimer_batch.e_ABfull_target
        )
    assert edge_energy.shape == (
        synthetic_dimer_batch.dimer_ind_full.numel(),
        2,
    )
    assert torch.equal(edge_energy[:, 0], torch.ones_like(edge_energy[:, 0]))
    assert torch.equal(
        edge_energy[:, 1], torch.full_like(edge_energy[:, 1], 2.0)
    )


def test_rackers_dimer_forward_rejects_unknown_elst_damping(
    synthetic_dimer_batch,
):
    dimer = DimerProp(
        ATParam=_ControlledRackersAtomParam(),
        dimer_eval="rackers_thole",
        elst_damping_type="unsupported",
    )
    with pytest.raises(ValueError, match="Unsupported elst_damping_type"):
        dimer(synthetic_dimer_batch)


def test_rackers_dimer_forward_valence_width_energy_activity():
    inputs = _rackers_kernel_inputs()
    changed_width_inputs = {
        **inputs,
        "valence_widths_A": inputs["valence_widths_A"] * 1.7,
        "valence_widths_B": inputs["valence_widths_B"] * 0.6,
    }

    pure_energy = mtp_mtp.rackers_thole_induction(
        **inputs, include_overlap=False, max_iterations=4
    )
    changed_pure_energy = mtp_mtp.rackers_thole_induction(
        **changed_width_inputs, include_overlap=False, max_iterations=4
    )
    overlap_energy = mtp_mtp.rackers_thole_induction(
        **inputs, include_overlap=True, max_iterations=4
    )
    changed_overlap_energy = mtp_mtp.rackers_thole_induction(
        **changed_width_inputs, include_overlap=True, max_iterations=4
    )

    assert torch.equal(pure_energy, changed_pure_energy)
    assert not torch.allclose(overlap_energy, changed_overlap_energy)


@pytest.mark.parametrize(
    "mode,expected_active_heads",
    [
        ("rackers_thole", {0, 1, 2}),
        ("rackers_thole_overlap", {0, 1, 2, 3}),
    ],
)
def test_rackers_joint_forward_scatter_and_gradients(
    mode,
    expected_active_heads,
    nested_hfvr_vw_model,
    synthetic_dimer_batch,
):
    # Keep initialization deterministic while checking the gradient seam below.
    # A random ReLU readout may be entirely dead and legitimately have zero
    # parameter gradients, so activity is asserted at each raw guess embedding
    # output instead of requiring every random MLP parameter to be nonzero.
    with torch.random.fork_rng():
        torch.manual_seed(0)
        model = RackersTholeDampingNN(
            atom_model=copy.deepcopy(nested_hfvr_vw_model),
            n_message=1,
            n_neuron=8,
            n_embed=4,
            freeze_atom_model=True,
        )
    dimer = DimerProp(
        ATParam=model,
        dimer_eval=mode,
        freeze_atom_model=True,
    )
    guess_outputs = [[] for _ in model.guess_layer]
    hook_handles = []
    for index, guess_layer in enumerate(model.guess_layer):
        def capture_guess_output(module, inputs, output, head_index=index):
            output.retain_grad()
            guess_outputs[head_index].append(output)

        hook_handles.append(
            guess_layer.register_forward_hook(capture_guess_output)
        )

    edge_energy, output_A, output_B = dimer(synthetic_dimer_batch)
    assert edge_energy.shape == (
        synthetic_dimer_batch.e_ABfull_source.numel(),
        2,
    )
    assert torch.isfinite(edge_energy).all()

    dimer_energy = scatter_sum_compile(
        edge_energy,
        synthetic_dimer_batch.dimer_ind_full,
        dim_size=synthetic_dimer_batch.total_charge_A.size(0),
    )
    assert dimer_energy.shape == (2, 2)
    assert torch.isfinite(dimer_energy).all()

    dimer_energy.square().mean().backward()
    for handle in hook_handles:
        handle.remove()
    for index, outputs in enumerate(guess_outputs):
        assert len(outputs) == 2
        assert all(output.grad is not None for output in outputs)
        assert all(torch.isfinite(output.grad).all() for output in outputs)
        has_nonzero_gradient = any(
            torch.count_nonzero(output.grad) > 0 for output in outputs
        )
        assert has_nonzero_gradient == (index in expected_active_heads)

    for head in model.param_readout_layers:
        gradients = [
            parameter.grad
            for readout in head
            for parameter in readout.parameters()
        ]
        assert all(
            torch.isfinite(gradient).all()
            for gradient in gradients
            if gradient is not None
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    optimizer.step()
    updated_parameters = model(synthetic_dimer_batch.batch_atomic_A)[-1]
    assert torch.isfinite(updated_parameters).all()
    assert torch.all(updated_parameters > 0)


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


def test_rackers_kernel_routes_top_level_direct_monomer_effects(
    monkeypatch,
):
    inputs = _rackers_kernel_inputs()
    original_builder = mtp_mtp._rackers_distance_tensors
    original_initial_fields = mtp_mtp._rackers_initial_permanent_fields

    def run_with_perturbation(direct_call_to_perturb):
        direct_call_index = 0
        captured_initial_fields = []

        def perturb_builder(*args, **kwargs):
            nonlocal direct_call_index
            tensors = original_builder(*args, **kwargs)
            damping_type = args[-1]
            if damping_type != "direct":
                return tensors

            current_index = direct_call_index
            direct_call_index += 1
            if current_index != direct_call_to_perturb:
                return tensors

            perturbed = list(tensors)
            perturbed[3] = tensors[3] * 1.7
            perturbed[4] = tensors[4] * 1.7
            return tuple(perturbed)

        def capture_initial_fields(*args, **kwargs):
            fields = original_initial_fields(*args, **kwargs)
            captured_initial_fields.append(
                tuple(field.detach().clone() for field in fields)
            )
            return fields

        monkeypatch.setattr(
            mtp_mtp, "_rackers_distance_tensors", perturb_builder
        )
        monkeypatch.setattr(
            mtp_mtp,
            "_rackers_initial_permanent_fields",
            capture_initial_fields,
        )
        energy = mtp_mtp.rackers_thole_induction(
            **inputs, include_overlap=False, max_iterations=4
        )
        assert len(captured_initial_fields) == 1
        assert direct_call_index == 3
        return energy.detach().clone(), captured_initial_fields[0]

    baseline_energy, (baseline_A, baseline_B) = run_with_perturbation(None)
    aa_energy, (aa_A, aa_B) = run_with_perturbation(1)
    bb_energy, (bb_A, bb_B) = run_with_perturbation(2)

    assert not torch.equal(aa_A, baseline_A)
    assert torch.equal(aa_B, baseline_B)
    assert not torch.allclose(aa_energy, baseline_energy)
    assert torch.equal(bb_A, baseline_A)
    assert not torch.equal(bb_B, baseline_B)
    assert not torch.allclose(bb_energy, baseline_energy)


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


def test_rackers_kernel_routes_charge_oracle_orientation_and_symmetry():
    dtype = torch.float64
    RA = torch.tensor(
        [[0.2, -0.4, 0.3], [1.4, 0.6, -0.2]], dtype=dtype
    )
    RB = torch.tensor(
        [[3.7, -0.8, 1.1], [4.4, 1.3, -0.7]], dtype=dtype
    )
    alpha_A = torch.tensor([0.7, 1.3], dtype=dtype)
    alpha_B = torch.tensor([1.1, 0.6], dtype=dtype)
    qA = torch.tensor([0.35, 0.8], dtype=dtype)
    qB = torch.tensor([0.55, 0.25], dtype=dtype)
    e_AB_source = torch.tensor([0, 1], dtype=torch.long)
    e_AB_target = torch.tensor([1, 0], dtype=torch.long)
    e_AA_source = torch.tensor([0, 1], dtype=torch.long)
    e_AA_target = torch.tensor([1, 0], dtype=torch.long)
    e_BB_source = torch.tensor([1, 0], dtype=torch.long)
    e_BB_target = torch.tensor([0, 1], dtype=torch.long)
    direct_AB = torch.tensor([0.31, 0.47], dtype=dtype)
    direct_AA = torch.tensor([0.4, 0.4], dtype=dtype)
    direct_BB = torch.tensor([0.38, 0.38], dtype=dtype)
    mutual_AB = torch.tensor([0.61, 0.79], dtype=dtype)
    mutual_AA = torch.tensor([0.7, 0.7], dtype=dtype)
    mutual_BB = torch.tensor([0.72, 0.72], dtype=dtype)

    def evaluate_production(
        R_first,
        R_second,
        alpha_first,
        alpha_second,
        q_first,
        q_second,
        cross_source,
        cross_target,
        first_source,
        first_target,
        second_source,
        second_target,
        direct_cross,
        direct_first,
        direct_second,
        mutual_cross,
        mutual_first,
        mutual_second,
    ):
        direct_tensors = (
            mtp_mtp._rackers_distance_tensors(
                R_first,
                R_second,
                cross_source,
                cross_target,
                alpha_first,
                alpha_second,
                direct_cross,
                "direct",
            ),
            mtp_mtp._rackers_distance_tensors(
                R_first,
                R_first,
                first_source,
                first_target,
                alpha_first,
                alpha_first,
                direct_first,
                "direct",
            ),
            mtp_mtp._rackers_distance_tensors(
                R_second,
                R_second,
                second_source,
                second_target,
                alpha_second,
                alpha_second,
                direct_second,
                "direct",
            ),
        )
        initial_fields = mtp_mtp._rackers_initial_permanent_fields(
            alpha_first,
            alpha_second,
            q_first,
            torch.zeros_like(R_first),
            q_second,
            torch.zeros_like(R_second),
            cross_source,
            cross_target,
            first_source,
            first_target,
            second_source,
            second_target,
            direct_tensors[0][3],
            direct_tensors[0][4],
            direct_tensors[1][3],
            direct_tensors[1][4],
            direct_tensors[2][3],
            direct_tensors[2][4],
        )
        mutual_tensors = (
            mtp_mtp._rackers_distance_tensors(
                R_first,
                R_second,
                cross_source,
                cross_target,
                alpha_first,
                alpha_second,
                mutual_cross,
                "mutual",
            ),
            mtp_mtp._rackers_distance_tensors(
                R_first,
                R_first,
                first_source,
                first_target,
                alpha_first,
                alpha_first,
                mutual_first,
                "mutual",
            ),
            mtp_mtp._rackers_distance_tensors(
                R_second,
                R_second,
                second_source,
                second_target,
                alpha_second,
                alpha_second,
                mutual_second,
                "mutual",
            ),
        )
        update = mtp_mtp._rackers_scf_update(
            alpha_first,
            alpha_second,
            cross_source,
            cross_target,
            first_source,
            first_target,
            second_source,
            second_target,
            mutual_tensors[0][4],
            mutual_tensors[1][4],
            mutual_tensors[2][4],
            initial_fields[0],
            initial_fields[1],
            initial_fields[0],
            initial_fields[1],
        )
        return direct_tensors, mutual_tensors, initial_fields, update

    actual_direct, actual_mutual, actual_initial, actual_update = (
        evaluate_production(
            RA,
            RB,
            alpha_A,
            alpha_B,
            qA,
            qB,
            e_AB_source,
            e_AB_target,
            e_AA_source,
            e_AA_target,
            e_BB_source,
            e_BB_target,
            direct_AB,
            direct_AA,
            direct_BB,
            mutual_AB,
            mutual_AA,
            mutual_BB,
        )
    )

    displacement_AB, expected_T1_AB, expected_direct_T2_AB = (
        _analytic_rackers_tensors(
            RA,
            RB,
            e_AB_source,
            e_AB_target,
            alpha_A,
            alpha_B,
            direct_AB,
            "direct",
        )
    )
    _, expected_T1_AA, expected_direct_T2_AA = (
        _analytic_rackers_tensors(
            RA,
            RA,
            e_AA_source,
            e_AA_target,
            alpha_A,
            alpha_A,
            direct_AA,
            "direct",
        )
    )
    _, expected_T1_BB, expected_direct_T2_BB = (
        _analytic_rackers_tensors(
            RB,
            RB,
            e_BB_source,
            e_BB_target,
            alpha_B,
            alpha_B,
            direct_BB,
            "direct",
        )
    )
    _, _, expected_mutual_T2_AB = _analytic_rackers_tensors(
        RA,
        RB,
        e_AB_source,
        e_AB_target,
        alpha_A,
        alpha_B,
        mutual_AB,
        "mutual",
    )
    _, _, expected_mutual_T2_AA = _analytic_rackers_tensors(
        RA,
        RA,
        e_AA_source,
        e_AA_target,
        alpha_A,
        alpha_A,
        mutual_AA,
        "mutual",
    )
    _, _, expected_mutual_T2_BB = _analytic_rackers_tensors(
        RB,
        RB,
        e_BB_source,
        e_BB_target,
        alpha_B,
        alpha_B,
        mutual_BB,
        "mutual",
    )
    expected_direct = (
        (expected_T1_AB, expected_direct_T2_AB),
        (expected_T1_AA, expected_direct_T2_AA),
        (expected_T1_BB, expected_direct_T2_BB),
    )
    expected_mutual_T2 = (
        expected_mutual_T2_AB,
        expected_mutual_T2_AA,
        expected_mutual_T2_BB,
    )
    for actual, expected in zip(actual_direct, expected_direct):
        assert torch.allclose(actual[3], expected[0], atol=1e-12)
        assert torch.allclose(actual[4], expected[1], atol=1e-12)
    for actual, expected in zip(actual_mutual, expected_mutual_T2):
        assert torch.allclose(actual[4], expected, atol=1e-12)

    alpha_A_cross = alpha_A.index_select(0, e_AB_source)
    alpha_B_cross = alpha_B.index_select(0, e_AB_target)
    qA_cross = qA.index_select(0, e_AB_source)
    qB_cross = qB.index_select(0, e_AB_target)
    cross_field_A = (
        alpha_A_cross[:, None] * expected_T1_AB * qB_cross[:, None]
    )
    cross_field_B = (
        alpha_B_cross[:, None] * -expected_T1_AB * qA_cross[:, None]
    )
    assert torch.all(
        torch.einsum("ai,ai->a", cross_field_A, displacement_AB) < 0
    )
    assert torch.all(
        torch.einsum("ai,ai->a", cross_field_B, displacement_AB) > 0
    )

    expected_initial_A = torch.zeros_like(RA)
    expected_initial_A.index_add_(0, e_AB_source, cross_field_A)
    expected_initial_A.index_add_(
        0,
        e_AA_target,
        alpha_A.index_select(0, e_AA_target)[:, None]
        * -expected_T1_AA
        * qA.index_select(0, e_AA_source)[:, None],
    )
    expected_initial_B = torch.zeros_like(RB)
    expected_initial_B.index_add_(0, e_AB_target, cross_field_B)
    expected_initial_B.index_add_(
        0,
        e_BB_target,
        alpha_B.index_select(0, e_BB_target)[:, None]
        * -expected_T1_BB
        * qB.index_select(0, e_BB_source)[:, None],
    )
    assert torch.allclose(
        actual_initial[0], expected_initial_A, atol=1e-12
    )
    assert torch.allclose(
        actual_initial[1], expected_initial_B, atol=1e-12
    )

    expected_update_A = expected_initial_A.clone()
    expected_update_A.index_add_(
        0,
        e_AB_source,
        alpha_A_cross[:, None]
        * torch.einsum(
            "aij,aj->ai",
            expected_mutual_T2_AB,
            expected_initial_B.index_select(0, e_AB_target),
        ),
    )
    expected_update_A.index_add_(
        0,
        e_AA_target,
        alpha_A.index_select(0, e_AA_target)[:, None]
        * torch.einsum(
            "aij,aj->ai",
            expected_mutual_T2_AA,
            expected_initial_A.index_select(0, e_AA_source),
        ),
    )
    expected_update_B = expected_initial_B.clone()
    expected_update_B.index_add_(
        0,
        e_AB_target,
        alpha_B_cross[:, None]
        * torch.einsum(
            "aij,aj->ai",
            expected_mutual_T2_AB,
            expected_initial_A.index_select(0, e_AB_source),
        ),
    )
    expected_update_B.index_add_(
        0,
        e_BB_target,
        alpha_B.index_select(0, e_BB_target)[:, None]
        * torch.einsum(
            "aij,aj->ai",
            expected_mutual_T2_BB,
            expected_initial_B.index_select(0, e_BB_source),
        ),
    )
    assert torch.allclose(actual_update[0], expected_update_A, atol=1e-12)
    assert torch.allclose(actual_update[1], expected_update_B, atol=1e-12)

    _, _, exchanged_initial, exchanged_update = evaluate_production(
        RB,
        RA,
        alpha_B,
        alpha_A,
        qB,
        qA,
        e_AB_target,
        e_AB_source,
        e_BB_source,
        e_BB_target,
        e_AA_source,
        e_AA_target,
        direct_AB,
        direct_BB,
        direct_AA,
        mutual_AB,
        mutual_BB,
        mutual_AA,
    )
    assert torch.allclose(exchanged_initial[0], actual_initial[1])
    assert torch.allclose(exchanged_initial[1], actual_initial[0])
    assert torch.allclose(exchanged_update[0], actual_update[1])
    assert torch.allclose(exchanged_update[1], actual_update[0])

    rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=dtype,
    )
    rotated_RA = RA @ rotation.T
    rotated_RB = RB @ rotation.T
    _, _, rotated_initial, rotated_update = evaluate_production(
        rotated_RA,
        rotated_RB,
        alpha_A,
        alpha_B,
        qA,
        qB,
        e_AB_source,
        e_AB_target,
        e_AA_source,
        e_AA_target,
        e_BB_source,
        e_BB_target,
        direct_AB,
        direct_AA,
        direct_BB,
        mutual_AB,
        mutual_AA,
        mutual_BB,
    )
    assert torch.allclose(rotated_initial[0], actual_initial[0] @ rotation.T)
    assert torch.allclose(rotated_initial[1], actual_initial[1] @ rotation.T)
    assert torch.allclose(rotated_update[0], actual_update[0] @ rotation.T)
    assert torch.allclose(rotated_update[1], actual_update[1] @ rotation.T)


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


def test_geometric_mean_edge_values_is_compile_safe():
    """The eager-only finite check must not break Dynamo tracing."""
    source = torch.tensor([1.0, 4.0, 9.0], dtype=torch.float64)
    target = torch.tensor([16.0, 25.0], dtype=torch.float64)
    e_source = torch.tensor([0, 1, 2], dtype=torch.long)
    e_target = torch.tensor([1, 0, 1], dtype=torch.long)

    compiled = torch.compile(
        geometric_mean_edge_values, backend="eager", fullgraph=True
    )
    actual = compiled(source, target, e_source, e_target)

    assert torch.equal(
        actual, torch.tensor([5.0, 8.0, 15.0], dtype=torch.float64)
    )


class _FakeHFVRModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.atom_model = torch.nn.Sequential(
            torch.nn.Linear(2, 3),
            torch.nn.Linear(3, 1),
        )
        self.hfvr_head = torch.nn.Linear(1, 2)


class _FakeAtomTypeParamModel:
    calls = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model = _FakeHFVRModel()
        self.model.requires_grad_(not kwargs["freeze_atom_model"])
        type(self).calls.append(self)


class _FakeRackersHarnessBase:
    calls = []

    def __init__(self, atom_model, **kwargs):
        self.kwargs = {"atom_model": atom_model, **kwargs}
        self.model = atom_model
        self.model.requires_grad_(not kwargs["freeze_atom_model"])
        self.dataset = object()
        self.train_calls = []
        type(self).calls.append(self)

    def train(
        self,
        model_path=None,
        n_epochs=50,
        world_size=1,
        omp_num_threads_per_process=6,
        lr=5e-4,
        dataloader_num_workers=4,
        random_seed=42,
        lr_decay=None,
    ):
        self.train_calls.append(
            {
                "model_path": model_path,
                "n_epochs": n_epochs,
                "world_size": world_size,
                "omp_num_threads_per_process": omp_num_threads_per_process,
                "lr": lr,
                "dataloader_num_workers": dataloader_num_workers,
                "random_seed": random_seed,
                "lr_decay": lr_decay,
            }
        )


class _FakeRackersTholeDampingModel(_FakeRackersHarnessBase):
    calls = []


class _FakeRackersTholeDampingOverlapModel(_FakeRackersHarnessBase):
    calls = []


def _patch_rackers_dispatch_fakes(monkeypatch):
    _FakeAtomTypeParamModel.calls.clear()
    _FakeRackersTholeDampingModel.calls.clear()
    _FakeRackersTholeDampingOverlapModel.calls.clear()
    monkeypatch.setattr(
        train_models.AtomPairwiseModels.mtp_mtp,
        "AtomTypeParamModel",
        _FakeAtomTypeParamModel,
    )
    monkeypatch.setattr(
        train_models.AtomPairwiseModels.mtp_mtp,
        "RackersTholeDampingModel",
        _FakeRackersTholeDampingModel,
    )
    monkeypatch.setattr(
        train_models.AtomPairwiseModels.mtp_mtp,
        "RackersTholeDampingOverlapModel",
        _FakeRackersTholeDampingOverlapModel,
    )


@pytest.mark.parametrize(
    "model_identifier,harness_type,other_harness_type",
    [
        (
            "RackersTholeDampingModel",
            _FakeRackersTholeDampingModel,
            _FakeRackersTholeDampingOverlapModel,
        ),
        (
            "RackersTholeDampingOverlapModel",
            _FakeRackersTholeDampingOverlapModel,
            _FakeRackersTholeDampingModel,
        ),
    ],
)
@pytest.mark.parametrize("freeze_atom_model", [True, False])
def test_rackers_dispatch_selects_harness_and_forwards_contract(
    tmp_path,
    monkeypatch,
    model_identifier,
    harness_type,
    other_harness_type,
    freeze_atom_model,
):
    _patch_rackers_dispatch_fakes(monkeypatch)

    model_out = tmp_path / "rackers-output.pt"
    train_models.train_pairwise_model(
        apnet_model_type=model_identifier,
        model_out=str(model_out),
        am_model_path="atom-checkpoint.pt",
        atom_type_param_model_path="hfvr-vw-checkpoint.pt",
        data_dir="rackers-data",
        n_epochs=7,
        lr=3e-4,
        lr_decay=0.75,
        random_seed=19,
        spec_type=11,
        r_cut=6.5,
        n_rbf=6,
        n_neuron=48,
        n_embed=12,
        n_params=99,
        pre_trained_model_path="rackers-checkpoint.pt",
        elst_damping_type="AMOEBA",
        ds_in_memory=True,
        freeze_atom_model=freeze_atom_model,
        omp_num_threads=23,
    )

    assert len(_FakeAtomTypeParamModel.calls) == 1
    hfvr_wrapper = _FakeAtomTypeParamModel.calls[0]
    assert hfvr_wrapper.kwargs == {
        "ds_root": None,
        "use_GPU": False,
        "ignore_database_null": True,
        "atom_model_pre_trained_path": "atom-checkpoint.pt",
        "pre_trained_model_path": "hfvr-vw-checkpoint.pt",
        "freeze_atom_model": freeze_atom_model,
    }
    assert len(harness_type.calls) == 1
    assert other_harness_type.calls == []
    rackers = harness_type.calls[0]
    assert rackers.kwargs["atom_model"] is hfvr_wrapper.model
    assert rackers.kwargs["pre_trained_model_path"] == "rackers-checkpoint.pt"
    assert rackers.kwargs["param_start_mean"] == [1.8, 0.34, 0.39, 1.8]
    assert rackers.kwargs["param_start_std"] == [0.01, 0.01, 0.01, 0.01]
    assert rackers.kwargs["freeze_atom_model"] is freeze_atom_model
    assert rackers.kwargs["elst_damping_type"] == "AMOEBA"
    assert rackers.kwargs["n_rbf"] == 6
    assert rackers.kwargs["n_neuron"] == 48
    assert rackers.kwargs["n_embed"] == 12
    assert rackers.kwargs["r_cut"] == 6.5
    assert rackers.kwargs["ds_spec_type"] == 11
    assert rackers.kwargs["ds_root"] == "rackers-data"
    assert rackers.kwargs["ds_random_seed"] == 19
    assert rackers.kwargs["ds_in_memory"] is True
    assert "n_params" not in rackers.kwargs
    assert all(
        parameter.requires_grad is not freeze_atom_model
        for parameter in rackers.model.parameters()
    )
    assert rackers.train_calls == [
        {
            "model_path": str(model_out),
            "n_epochs": 7,
            "world_size": 1,
            "omp_num_threads_per_process": 23,
            "lr": 3e-4,
            "dataloader_num_workers": 4,
            "random_seed": 19,
            "lr_decay": 0.75,
        }
    ]


@pytest.mark.parametrize(
    "field,value",
    [
        ("param_start_mean", 1.8),
        ("param_start_std", 0.01),
        ("param_start_mean", [1.8, 0.34, 0.39]),
        ("param_start_std", [0.01, 0.01, 0.01]),
    ],
)
def test_rackers_dispatch_rejects_ambiguous_parameter_lists(field, value):
    kwargs = {
        "apnet_model_type": "RackersTholeDampingModel",
        "pre_trained_model_path": None,
        "param_start_mean": [1.8, 0.34, 0.39, 1.8],
        "param_start_std": [0.01, 0.01, 0.01, 0.01],
    }
    kwargs[field] = value
    with pytest.raises(ValueError, match="exactly four"):
        train_models.train_pairwise_model(**kwargs)


@pytest.mark.parametrize(
    "field,index,value,match",
    [
        (
            "param_start_mean",
            0,
            0.0,
            "param_start_mean values must be finite and strictly greater",
        ),
        (
            "param_start_mean",
            1,
            -0.1,
            "param_start_mean values must be finite and strictly greater",
        ),
        (
            "param_start_mean",
            2,
            float("nan"),
            "param_start_mean values must be finite and strictly greater",
        ),
        (
            "param_start_mean",
            3,
            float("inf"),
            "param_start_mean values must be finite and strictly greater",
        ),
        (
            "param_start_std",
            0,
            -0.1,
            "param_start_std values must be finite and greater than or equal",
        ),
        (
            "param_start_std",
            1,
            float("nan"),
            "param_start_std values must be finite and greater than or equal",
        ),
        (
            "param_start_std",
            2,
            float("inf"),
            "param_start_std values must be finite and greater than or equal",
        ),
    ],
)
def test_rackers_dispatch_rejects_invalid_initialization_domains(
    monkeypatch, field, index, value, match
):
    _patch_rackers_dispatch_fakes(monkeypatch)
    kwargs = {
        "apnet_model_type": "RackersTholeDampingModel",
        "pre_trained_model_path": None,
        "param_start_mean": list(RACKERS_INITIAL_VALUES),
        "param_start_std": list(RACKERS_INITIAL_STDS),
    }
    kwargs[field][index] = value

    with pytest.raises(ValueError, match=match):
        train_models.train_pairwise_model(**kwargs)

    assert _FakeAtomTypeParamModel.calls == []
    assert _FakeRackersTholeDampingModel.calls == []


@pytest.mark.parametrize(
    "field,values,match",
    [
        (
            "param_start_mean",
            [1e39, 1.0, 1.0, 1.0],
            "transformed param_start_mean values must be finite and representable",
        ),
        (
            "param_start_std",
            [1e39, 0.0, 0.0, 0.0],
            "param_start_std values must be representable",
        ),
    ],
)
def test_rackers_dispatch_rejects_embedding_dtype_overflow(
    monkeypatch, field, values, match
):
    _patch_rackers_dispatch_fakes(monkeypatch)
    kwargs = {
        "apnet_model_type": "RackersTholeDampingModel",
        "pre_trained_model_path": None,
        "param_start_mean": list(RACKERS_INITIAL_VALUES),
        "param_start_std": list(RACKERS_INITIAL_STDS),
    }
    kwargs[field] = values

    with pytest.raises(ValueError, match=match):
        train_models.train_pairwise_model(**kwargs)

    assert _FakeAtomTypeParamModel.calls == []
    assert _FakeRackersTholeDampingModel.calls == []


def test_rackers_dispatch_accepts_large_representable_mean(
    monkeypatch, tmp_path
):
    _patch_rackers_dispatch_fakes(monkeypatch)
    means = [1000.0, 1.0, 1.0, 1.0]
    train_models.train_pairwise_model(
        apnet_model_type="RackersTholeDampingModel",
        model_out=str(tmp_path / "valid-large-mean.pt"),
        pre_trained_model_path=None,
        param_start_mean=means,
        param_start_std=[0.0, 0.0, 0.0, 0.0],
    )

    assert _FakeRackersTholeDampingModel.calls[0].kwargs[
        "param_start_mean"
    ] == means


def test_rackers_dispatch_accepts_zero_raw_std(monkeypatch, tmp_path):
    _patch_rackers_dispatch_fakes(monkeypatch)
    stds = [0.0, 0.01, 0.0, 0.02]
    train_models.train_pairwise_model(
        apnet_model_type="RackersTholeDampingModel",
        model_out=str(tmp_path / "valid-zero-std.pt"),
        pre_trained_model_path=None,
        param_start_std=stds,
    )

    assert _FakeRackersTholeDampingModel.calls[0].kwargs[
        "param_start_std"
    ] == stds


class _FakeLegacyPairwiseHarness:
    calls = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.dataset = object()
        self.train_calls = 0
        self.train_kwargs = None
        type(self).calls.append(self)

    def train(
        self,
        model_path=None,
        n_epochs=50,
        world_size=1,
        omp_num_threads_per_process=6,
        lr=5e-4,
        dataloader_num_workers=4,
        random_seed=42,
        lr_decay=None,
    ):
        self.train_calls += 1
        self.train_kwargs = {
            "model_path": model_path,
            "n_epochs": n_epochs,
            "world_size": world_size,
            "omp_num_threads_per_process": omp_num_threads_per_process,
            "lr": lr,
            "dataloader_num_workers": dataloader_num_workers,
            "random_seed": random_seed,
            "lr_decay": lr_decay,
        }


@pytest.mark.parametrize(
    "mean,std,expected_mean,expected_std",
    [
        (2.25, 0.2, [2.25, 2.25, 2.25], [0.2, 0.2, 0.2]),
        (None, None, [1.5, 1.5, 1.5], [0.1, 0.1, 0.1]),
    ],
)
def test_legacy_dispatch_still_broadcasts_scalar_defaults(
    tmp_path, monkeypatch, mean, std, expected_mean, expected_std
):
    _FakeLegacyPairwiseHarness.calls.clear()
    monkeypatch.setattr(
        train_models.AtomPairwiseModels.mtp_mtp,
        "AtomTypeParamModel",
        _FakeLegacyPairwiseHarness,
    )

    train_models.train_pairwise_model(
        apnet_model_type="AtomTypeParamModel",
        model_out=str(tmp_path / "legacy.pt"),
        pre_trained_model_path=None,
        n_params=3,
        param_start_mean=mean,
        param_start_std=std,
    )

    harness = _FakeLegacyPairwiseHarness.calls[0]
    assert harness.kwargs["param_start_mean"] == expected_mean
    assert harness.kwargs["param_start_std"] == expected_std
    assert harness.train_calls == 1


@pytest.mark.parametrize(
    "model_identifier,harness_type",
    [
        ("RackersTholeDampingModel", _FakeRackersTholeDampingModel),
        (
            "RackersTholeDampingOverlapModel",
            _FakeRackersTholeDampingOverlapModel,
        ),
    ],
)
@pytest.mark.parametrize(
    "checkpoint_kwargs,expected_checkpoint",
    [
        pytest.param({}, None, id="omitted"),
        pytest.param(
            {"pre_trained_model_path": "explicit-rackers.pt"},
            "explicit-rackers.pt",
            id="explicit",
        ),
    ],
)
def test_rackers_dispatch_checkpoint_resolution(
    tmp_path,
    monkeypatch,
    model_identifier,
    harness_type,
    checkpoint_kwargs,
    expected_checkpoint,
):
    _patch_rackers_dispatch_fakes(monkeypatch)

    train_models.train_pairwise_model(
        apnet_model_type=model_identifier,
        model_out=str(tmp_path / "rackers.pt"),
        **checkpoint_kwargs,
    )

    assert harness_type.calls[0].kwargs["pre_trained_model_path"] == (
        expected_checkpoint
    )


@pytest.mark.parametrize(
    "model_identifier,harness_type",
    [
        ("RackersTholeDampingModel", _FakeRackersTholeDampingModel),
        (
            "RackersTholeDampingOverlapModel",
            _FakeRackersTholeDampingOverlapModel,
        ),
    ],
)
def test_rackers_dispatch_build_dataset_only_skips_train(
    tmp_path, monkeypatch, model_identifier, harness_type
):
    _patch_rackers_dispatch_fakes(monkeypatch)

    train_models.train_pairwise_model(
        apnet_model_type=model_identifier,
        model_out=str(tmp_path / "rackers.pt"),
        build_dataset_only=True,
    )

    rackers = harness_type.calls[0]
    assert rackers.kwargs["pre_trained_model_path"] is None
    assert rackers.train_calls == []


def test_legacy_dispatch_preserves_omitted_checkpoint_default(
    tmp_path, monkeypatch
):
    _FakeLegacyPairwiseHarness.calls.clear()
    monkeypatch.setattr(
        train_models.AtomPairwiseModels.mtp_mtp,
        "AtomTypeParamModel",
        _FakeLegacyPairwiseHarness,
    )

    train_models.train_pairwise_model(
        apnet_model_type="AtomTypeParamModel",
        model_out=str(tmp_path / "legacy.pt"),
    )

    legacy = _FakeLegacyPairwiseHarness.calls[0]
    assert legacy.kwargs["pre_trained_model_path"] == (
        "./models/dapnet2/ap2_0.pt"
    )


@pytest.mark.parametrize(
    "model_identifier,harness_type",
    [
        ("RackersTholeDampingModel", _FakeRackersTholeDampingModel),
        (
            "RackersTholeDampingOverlapModel",
            _FakeRackersTholeDampingOverlapModel,
        ),
    ],
)
def test_rackers_dispatch_forces_single_process_on_multi_gpu(
    tmp_path, monkeypatch, model_identifier, harness_type
):
    _patch_rackers_dispatch_fakes(monkeypatch)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

    train_models.train_pairwise_model(
        apnet_model_type=model_identifier,
        model_out=str(tmp_path / "rackers.pt"),
    )

    assert harness_type.calls[0].train_calls[0]["world_size"] == 1


def test_legacy_dispatch_retains_multi_gpu_world_size(tmp_path, monkeypatch):
    _FakeLegacyPairwiseHarness.calls.clear()
    monkeypatch.setattr(
        train_models.AtomPairwiseModels.mtp_mtp,
        "AtomTypeParamModel",
        _FakeLegacyPairwiseHarness,
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

    train_models.train_pairwise_model(
        apnet_model_type="AtomTypeParamModel",
        model_out=str(tmp_path / "legacy.pt"),
        pre_trained_model_path=None,
    )

    assert _FakeLegacyPairwiseHarness.calls[0].train_kwargs["world_size"] == 2


@pytest.mark.parametrize(
    "model_identifier,expected_mean,expected_std",
    [
        ("AtomTypeParamModel", 2.0, 0.1),
        ("RackersTholeDampingModel", None, None),
        ("RackersTholeDampingOverlapModel", None, None),
    ],
)
def test_cli_resolves_unset_parameter_defaults_by_route(
    monkeypatch, model_identifier, expected_mean, expected_std
):
    calls = []
    monkeypatch.setattr(
        sys,
        "argv",
        ["train_models.py", "--train_apnet", model_identifier],
    )
    monkeypatch.setattr(train_models, "set_all_seeds", lambda seed: None)
    monkeypatch.setattr(
        train_models,
        "train_pairwise_model",
        lambda **kwargs: calls.append(kwargs),
    )

    train_models.main()

    assert calls[0]["param_start_mean"] == expected_mean
    assert calls[0]["param_start_std"] == expected_std


def test_pairwise_cli_omitted_omp_threads_uses_legacy_default(
    tmp_path, monkeypatch
):
    _patch_rackers_dispatch_fakes(monkeypatch)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_models.py",
            "--train_apnet",
            "RackersTholeDampingModel",
            "--ap_model_path",
            str(tmp_path / "rackers.pt"),
        ],
    )
    monkeypatch.setattr(train_models, "set_all_seeds", lambda seed: None)

    train_models.main()

    train_call = _FakeRackersTholeDampingModel.calls[0].train_calls[0]
    assert train_call["world_size"] == 1
    assert train_call["omp_num_threads_per_process"] == 8


def test_atom_cli_omitted_omp_threads_uses_atom_default(monkeypatch):
    calls = []
    monkeypatch.setattr(
        sys,
        "argv",
        ["train_models.py", "--train_am", "AtomModel"],
    )
    monkeypatch.setattr(train_models, "set_all_seeds", lambda seed: None)
    monkeypatch.setattr(
        train_models,
        "train_atom_model",
        lambda **kwargs: calls.append(kwargs),
    )

    train_models.main()

    assert calls[0]["omp_num_threads"] == 1


def test_cli_forwards_omp_threads_to_pairwise_training(monkeypatch):
    calls = []
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_models.py",
            "--train_apnet",
            "RackersTholeDampingModel",
            "--omp_num_threads",
            "23",
        ],
    )
    monkeypatch.setattr(train_models, "set_all_seeds", lambda seed: None)
    monkeypatch.setattr(
        train_models,
        "train_pairwise_model",
        lambda **kwargs: calls.append(kwargs),
    )

    train_models.main()

    assert calls[0]["omp_num_threads"] == 23


def test_cli_help_names_both_rackers_routes(capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train_models.py", "--help"])
    with pytest.raises(SystemExit) as exc_info:
        train_models.main()
    assert exc_info.value.code == 0
    help_output = capsys.readouterr().out
    assert "RackersTholeDampingModel" in help_output
    assert "RackersTholeDampingOverlapModel" in help_output
    assert "exactly four" in help_output
