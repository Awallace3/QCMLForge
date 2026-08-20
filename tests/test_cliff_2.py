"""Tests for the CLIFF-2 inference assembly and the two-stage merge helper.

Covers the spec's "Merge helper test" and "``CLIFF2Model`` tests" sections of
``docs/superpowers/specs/2026-08-20-cliff-classical-exchange-and-cliff2-design.md``.

Everything here is CPU-only and needs no network or database access.  The
shared ``nested_hfvr_vw_model`` / ``synthetic_dimer_batch`` /
``synthetic_qcel_dimers`` fixtures come from ``tests/conftest.py``.
"""

import os
import subprocess
import sys
import copy
import inspect
import pathlib

import numpy as np
import pytest
import qcelemental as qcel
import torch

from apnet_pt import model_io
from apnet_pt.AtomPairwiseModels import cliff_2
from apnet_pt.AtomPairwiseModels.cliff_2 import (
    CLIFF2_COMPONENT_LABELS,
    CLIFF2_DIMER_EVAL,
    CLIFF2_MODEL_TYPE,
    CLIFF2Model,
    merge_classical_parameter_checkpoints,
)
from apnet_pt.AtomPairwiseModels.mtp_mtp import (
    CLIFF_CLASSICAL_EXCH_INDEX,
    CLIFF_CLASSICAL_INITIAL_VALUES,
    CLIFF_CLASSICAL_PARAMETER_NAMES,
    FULL_EDGE_DIMER_EVAL_MODES,
    POSITIVE_PARAMETER_CONTRACTS,
    RACKERS_PARAMETER_NAMES,
    CliffClassicalModel,
    CliffClassicalOverlapModel,
    CliffExchangeModel,
    RackersTholeDampingModel,
    RackersTholeDampingOverlapModel,
)

_GEOM_ROOT = (
    pathlib.Path(__file__).parent / "test_data_path" / "test_geoms"
)
_TWO_GEOM = _GEOM_ROOT / "two_geom"
_WATER_CLOSE = _GEOM_ROOT / "many_geom" / "mol_cliff_water_close.dat"

# Small architecture shared by every harness built here.  It must be identical
# across component checkpoints or the merge helper rejects them.
_ARCH = dict(n_message=1, n_neuron=8, n_embed=4)


# ---------------------------------------------------------------------------
# Local helpers and fixtures
# ---------------------------------------------------------------------------


def _build_harness(harness_type, nested_model, **kwargs):
    """Construct a parameter harness with no dataset and no GPU.

    ``ignore_database_null=True`` is the same escape hatch the Rackers and
    CLIFF harness tests use: it keeps ``AM_DimerParam_Model.__init__`` from
    building an on-disk pairwise dataset.
    """
    options = dict(
        atom_model=copy.deepcopy(nested_model),
        dataset=None,
        ignore_database_null=True,
        use_GPU=False,
        **_ARCH,
    )
    options.update(kwargs)
    return harness_type(**options)


def _zero_readout_heads(model):
    """Zero every correction MLP so only the guess embedding survives."""
    with torch.no_grad():
        for head in model.param_readout_layers:
            for readout in head:
                for parameter in readout.parameters():
                    parameter.zero_()


def _load_dimer(path):
    with open(path) as handle:
        return qcel.models.Molecule.from_data(handle.read())


@pytest.fixture
def two_geom_dimers():
    """Two real dimers: a separated water dimer and a close-contact one."""
    return [
        _load_dimer(_TWO_GEOM / "water_dimer.dat"),
        _load_dimer(_WATER_CLOSE),
    ]


@pytest.fixture
def component_checkpoints(tmp_path, nested_hfvr_vw_model):
    """A Rackers-overlap and a CliffExchange checkpoint sharing one nested model.

    This is the two-stage fitting route's stage-one output: elst + induction
    fitted by one harness and exchange by another, both wrapping the *same*
    frozen HFVR / valence-width model.
    """
    rackers = _build_harness(
        RackersTholeDampingOverlapModel, nested_hfvr_vw_model
    )
    exchange = _build_harness(CliffExchangeModel, nested_hfvr_vw_model)
    rackers_path = tmp_path / "rackers.pt"
    exchange_path = tmp_path / "exch.pt"
    rackers.save_model(rackers_path)
    exchange.save_model(exchange_path)
    return {
        "rackers": rackers,
        "exchange": exchange,
        "rackers_path": rackers_path,
        "exchange_path": exchange_path,
    }


@pytest.fixture
def merged_classical_path(tmp_path, component_checkpoints):
    path = tmp_path / "merged.pt"
    merge_classical_parameter_checkpoints(
        component_checkpoints["rackers_path"],
        component_checkpoints["exchange_path"],
        path,
    )
    return path


#: Physical CLIFF-scale valence width (bohr) for the physics sanity fixture.
#: ``sigma_O = 0.39`` is the CLIFF Table I / Fig. 5 value the golden overlap
#: test also uses.
_PHYSICAL_VALENCE_WIDTH = 0.39


def _make_parameters_physical(head):
    """Pin a head to its documented initialization with physical widths.

    ``nested_hfvr_vw_model`` runs ``set_weights_to_value(..., 0.01)``, which
    also flattens the *guess* embeddings, so its predicted valence widths are
    ``0.01`` bohr -- three orders of magnitude below anything physical, and
    below the exchange width floor.  Zeroing every correction MLP makes the
    CLIFF head reproduce its documented initialization
    (``K_exch = 2.5``, ``K_elst = 1.8``, ...) and re-seeding the nested guess
    embeddings restores a physical Hirshfeld ratio (1.0) and valence width, so
    the predicted energies land on a chemically meaningful scale.
    """
    _zero_readout_heads(head)
    nested = head.atom_model
    _zero_readout_heads(nested)
    with torch.no_grad():
        nested.guess_layer[0].weight.fill_(1.0)
        nested.guess_layer[1].weight.fill_(_PHYSICAL_VALENCE_WIDTH)


def _classical_checkpoint(tmp_path, nested_model, harness_type, name,
                          physical=False):
    """Save a single-stage classical checkpoint from ``harness_type``."""
    harness = _build_harness(harness_type, nested_model)
    if physical:
        _make_parameters_physical(harness.model)
    path = tmp_path / name
    harness.save_model(path)
    return harness, path


@pytest.fixture
def classical_overlap_checkpoint(tmp_path, nested_hfvr_vw_model):
    _, path = _classical_checkpoint(
        tmp_path,
        nested_hfvr_vw_model,
        CliffClassicalOverlapModel,
        "classical_overlap.pt",
    )
    return path


# ---------------------------------------------------------------------------
# Merge helper
# ---------------------------------------------------------------------------


def _guess_weight(state, column):
    return state[f"guess_layer.{column}.weight"]


def _readout_keys(state, column):
    prefix = f"param_readout_layers.{column}."
    return sorted(key for key in state if key.startswith(prefix))


def _readout_tensors(state, column):
    return [state[key] for key in _readout_keys(state, column)]


def _columns_equal(left_state, left_col, right_state, right_col):
    if not torch.equal(
        _guess_weight(left_state, left_col),
        _guess_weight(right_state, right_col),
    ):
        return False
    left = _readout_tensors(left_state, left_col)
    right = _readout_tensors(right_state, right_col)
    if len(left) != len(right):
        return False
    return all(torch.equal(a, b) for a, b in zip(left, right))


def test_merge_requires_at_least_one_source():
    with pytest.raises(ValueError, match="at least one of"):
        merge_classical_parameter_checkpoints(None, None, None)


def test_merge_maps_every_column_by_name(component_checkpoints, tmp_path):
    """Rackers columns 0-3 and exchange column 0 land on classical 0-4."""
    rackers_state = component_checkpoints["rackers"]._create_checkpoint()[
        "model_state_dict"
    ]
    exchange_state = component_checkpoints["exchange"]._create_checkpoint()[
        "model_state_dict"
    ]
    output_path = tmp_path / "merged.pt"
    merged = merge_classical_parameter_checkpoints(
        component_checkpoints["rackers_path"],
        component_checkpoints["exchange_path"],
        output_path,
    )
    merged_state = merged["model_state_dict"]

    for source_column, name in enumerate(RACKERS_PARAMETER_NAMES):
        destination = CLIFF_CLASSICAL_PARAMETER_NAMES.index(name)
        assert _columns_equal(
            merged_state, destination, rackers_state, source_column
        ), name
    assert _columns_equal(
        merged_state,
        CLIFF_CLASSICAL_EXCH_INDEX,
        exchange_state,
        0,
    )
    # The exchange source's single column is column 0 in the *source*; a
    # positional copy would have overwritten `elst`.
    assert not _columns_equal(merged_state, 0, exchange_state, 0)

    assert merged["model_type"] == "CliffClassicalNN"
    config = merged["config"]
    assert config["model_type"] == "CliffClassicalNN"
    assert config["parameter_names"] == list(CLIFF_CLASSICAL_PARAMETER_NAMES)
    # The Rackers *overlap* route fitted `ind_overlap`, so the merged model
    # inherits the overlap-enabled classical mode.
    assert config["dimer_eval"] == "cliff_classical_overlap"
    assert config["component_gamma"] is None
    assert config["total_includes_d3"] is False
    assert set(merged["metadata"]["merged_columns"]) == set(
        CLIFF_CLASSICAL_PARAMETER_NAMES
    )
    assert merged["metadata"]["unclaimed_columns"] == []
    # The write is a side effect; the dict is the return value.
    assert output_path.exists()
    on_disk = model_io.load_checkpoint(output_path)
    assert on_disk["config"]["parameter_names"] == config["parameter_names"]


def test_merge_without_output_path_writes_nothing(
    component_checkpoints, tmp_path
):
    before = set(p.name for p in tmp_path.iterdir())
    merged = merge_classical_parameter_checkpoints(
        component_checkpoints["rackers_path"],
        component_checkpoints["exchange_path"],
        None,
    )
    assert merged["model_type"] == "CliffClassicalNN"
    assert set(p.name for p in tmp_path.iterdir()) == before


def test_merge_rackers_only_leaves_exchange_at_initialization(
    component_checkpoints,
):
    rackers_state = component_checkpoints["rackers"]._create_checkpoint()[
        "model_state_dict"
    ]
    merged = merge_classical_parameter_checkpoints(
        component_checkpoints["rackers_path"], None, None
    )
    merged_state = merged["model_state_dict"]
    for column, name in enumerate(RACKERS_PARAMETER_NAMES):
        assert _columns_equal(merged_state, column, rackers_state, column), name
    # `exch` is unclaimed: it must not have been filled from any Rackers
    # column, and it must still sit at its documented initialization value.
    for column in range(len(RACKERS_PARAMETER_NAMES)):
        assert not _columns_equal(
            merged_state, CLIFF_CLASSICAL_EXCH_INDEX, rackers_state, column
        )
    epsilon = merged["config"]["positivity_epsilon"]
    initialized = (
        torch.nn.functional.softplus(
            _guess_weight(merged_state, CLIFF_CLASSICAL_EXCH_INDEX)
        )
        + epsilon
    )
    assert initialized.mean().item() == pytest.approx(
        CLIFF_CLASSICAL_INITIAL_VALUES[CLIFF_CLASSICAL_EXCH_INDEX],
        abs=0.1,
    )
    assert merged["metadata"]["unclaimed_columns"] == ["exch"]
    assert merged["config"]["dimer_eval"] == "cliff_classical_overlap"


def test_merge_exchange_only_does_not_touch_column_zero(
    component_checkpoints,
):
    exchange_state = component_checkpoints["exchange"]._create_checkpoint()[
        "model_state_dict"
    ]
    merged = merge_classical_parameter_checkpoints(
        None, component_checkpoints["exchange_path"], None
    )
    merged_state = merged["model_state_dict"]
    assert _columns_equal(
        merged_state, CLIFF_CLASSICAL_EXCH_INDEX, exchange_state, 0
    )
    for column in range(CLIFF_CLASSICAL_EXCH_INDEX):
        assert not _columns_equal(merged_state, column, exchange_state, 0)
    assert merged["metadata"]["unclaimed_columns"] == list(
        RACKERS_PARAMETER_NAMES
    )
    # No Rackers source => no induction fit => the plain classical mode.
    assert merged["config"]["dimer_eval"] == "cliff_classical"


def test_merge_preserves_the_nested_atom_model_state(component_checkpoints):
    rackers_state = component_checkpoints["rackers"]._create_checkpoint()[
        "model_state_dict"
    ]
    merged = merge_classical_parameter_checkpoints(
        component_checkpoints["rackers_path"],
        component_checkpoints["exchange_path"],
        None,
    )
    merged_state = merged["model_state_dict"]
    nested_keys = [
        key for key in rackers_state if key.startswith("atom_model.")
    ]
    assert nested_keys
    for key in nested_keys:
        assert torch.equal(merged_state[key], rackers_state[key]), key
    assert merged["metadata"]["nested_state_source"] == (
        "rackers_checkpoint_path"
    )


def test_merged_checkpoint_warm_starts_the_joint_harness(
    merged_classical_path, nested_hfvr_vw_model
):
    """The whole point of the merge: it is a loadable ``pre_trained_model_path``."""
    harness = CliffClassicalOverlapModel(
        pre_trained_model_path=merged_classical_path,
        atom_model=None,
        dataset=None,
        ignore_database_null=True,
        use_GPU=False,
    )
    merged = model_io.load_checkpoint(merged_classical_path)
    assert harness.model.get_config()["parameter_names"] == list(
        CLIFF_CLASSICAL_PARAMETER_NAMES
    )
    loaded_state = harness.model.state_dict()
    for key, value in merged["model_state_dict"].items():
        assert torch.equal(loaded_state[key], value), key


def test_merge_rejects_nested_config_mismatch(tmp_path, nested_hfvr_vw_model):
    from apnet_pt.AtomModels.ap2_atom_model import AtomMPNN
    from apnet_pt.AtomPairwiseModels.mtp_mtp import AtomTypeParamNN

    other_nested = AtomTypeParamNN(
        atom_model=AtomMPNN(
            n_message=1, n_rbf=2, n_neuron=8, n_embed=4, r_cut=5.0
        ),
        n_message=1,
        n_neuron=16,  # differs from the shared fixture's 8
        n_embed=4,
        param_start_mean=[1.0, 0.4],
        param_start_std=[0.0, 0.0],
        n_params=2,
        freeze_atom_model=False,
    )
    rackers = _build_harness(
        RackersTholeDampingOverlapModel, nested_hfvr_vw_model
    )
    exchange = _build_harness(CliffExchangeModel, other_nested)
    rackers_path = tmp_path / "rackers.pt"
    exchange_path = tmp_path / "exch.pt"
    rackers.save_model(rackers_path)
    exchange.save_model(exchange_path)

    with pytest.raises(ValueError, match="disagree on nested_atom_model"):
        merge_classical_parameter_checkpoints(
            rackers_path, exchange_path, None
        )


def test_merge_rejects_architecture_mismatch(tmp_path, nested_hfvr_vw_model):
    rackers = _build_harness(
        RackersTholeDampingOverlapModel, nested_hfvr_vw_model
    )
    exchange = _build_harness(
        CliffExchangeModel, nested_hfvr_vw_model, n_neuron=16
    )
    rackers_path = tmp_path / "rackers.pt"
    exchange_path = tmp_path / "exch.pt"
    rackers.save_model(rackers_path)
    exchange.save_model(exchange_path)

    with pytest.raises(ValueError, match="disagree on n_neuron"):
        merge_classical_parameter_checkpoints(
            rackers_path, exchange_path, None
        )


def test_merge_rejects_reordered_parameter_names(component_checkpoints):
    path = component_checkpoints["rackers_path"]
    checkpoint = model_io.load_checkpoint(path)
    checkpoint["config"]["parameter_names"] = list(
        reversed(RACKERS_PARAMETER_NAMES)
    )
    model_io.save_checkpoint(checkpoint, path)
    with pytest.raises(
        ValueError,
        match=r"rackers_checkpoint_path checkpoint parameter_names must "
        r"exactly match",
    ):
        merge_classical_parameter_checkpoints(
            path, component_checkpoints["exchange_path"], None
        )


def test_merge_rejects_a_missing_parameter_name_list(component_checkpoints):
    path = component_checkpoints["exchange_path"]
    checkpoint = model_io.load_checkpoint(path)
    checkpoint["config"].pop("parameter_names")
    model_io.save_checkpoint(checkpoint, path)
    with pytest.raises(
        ValueError, match="exchange_checkpoint_path checkpoint parameter_names"
    ):
        merge_classical_parameter_checkpoints(
            component_checkpoints["rackers_path"], path, None
        )


def test_merge_rejects_a_non_parameter_head_checkpoint(
    tmp_path, component_checkpoints
):
    path = component_checkpoints["rackers_path"]
    checkpoint = model_io.load_checkpoint(path)
    checkpoint["model_type"] = "AtomTypeParamNN"
    model_io.save_checkpoint(checkpoint, path)
    with pytest.raises(
        ValueError, match="is not a positive .*per-atom parameter head"
    ):
        merge_classical_parameter_checkpoints(path, None, None)


def test_merge_rejects_names_absent_from_the_classical_contract(
    monkeypatch, component_checkpoints
):
    """A head whose contract has no classical column cannot be merged."""
    monkeypatch.setitem(
        POSITIVE_PARAMETER_CONTRACTS, "CliffExchangeNN", ("bogus",)
    )
    path = component_checkpoints["exchange_path"]
    checkpoint = model_io.load_checkpoint(path)
    checkpoint["config"]["parameter_names"] = ["bogus"]
    model_io.save_checkpoint(checkpoint, path)
    with pytest.raises(
        ValueError, match=r"no column in the classical contract"
    ):
        merge_classical_parameter_checkpoints(None, path, None)


def test_merge_rejects_a_duplicate_parameter_claim(
    tmp_path, nested_hfvr_vw_model
):
    classical = _build_harness(
        CliffClassicalOverlapModel, nested_hfvr_vw_model
    )
    exchange = _build_harness(CliffExchangeModel, nested_hfvr_vw_model)
    classical_path = tmp_path / "classical.pt"
    exchange_path = tmp_path / "exch.pt"
    classical.save_model(classical_path)
    exchange.save_model(exchange_path)
    with pytest.raises(ValueError, match="claimed by both"):
        merge_classical_parameter_checkpoints(
            classical_path, exchange_path, None
        )


def test_merge_rejects_a_tensor_shape_mismatch(component_checkpoints):
    path = component_checkpoints["exchange_path"]
    checkpoint = model_io.load_checkpoint(path)
    key = "guess_layer.0.weight"
    original = checkpoint["model_state_dict"][key]
    checkpoint["model_state_dict"][key] = torch.zeros(
        original.shape[0], original.shape[1] + 1
    )
    model_io.save_checkpoint(checkpoint, path)
    with pytest.raises(ValueError, match="expects"):
        merge_classical_parameter_checkpoints(
            component_checkpoints["rackers_path"], path, None
        )


def test_merge_warns_when_the_nested_states_diverge(
    tmp_path, nested_hfvr_vw_model
):
    rackers = _build_harness(
        RackersTholeDampingOverlapModel, nested_hfvr_vw_model
    )
    exchange = _build_harness(CliffExchangeModel, nested_hfvr_vw_model)
    with torch.no_grad():
        for parameter in exchange.model.atom_model.parameters():
            parameter.add_(0.5)
    rackers_path = tmp_path / "rackers.pt"
    exchange_path = tmp_path / "exch.pt"
    rackers.save_model(rackers_path)
    exchange.save_model(exchange_path)
    with pytest.warns(RuntimeWarning, match="nested atom_model weights"):
        merged = merge_classical_parameter_checkpoints(
            rackers_path, exchange_path, None
        )
    assert merged["metadata"]["nested_state_source"] == (
        "rackers_checkpoint_path"
    )


# ---------------------------------------------------------------------------
# CLIFF2Model construction contract
# ---------------------------------------------------------------------------


def test_both_construction_forms_raise(
    merged_classical_path, component_checkpoints
):
    with pytest.raises(ValueError, match="not both"):
        CLIFF2Model(
            classical_model_path=merged_classical_path,
            rackers_model_path=component_checkpoints["rackers_path"],
            use_GPU=False,
        )
    with pytest.raises(ValueError, match="not both"):
        CLIFF2Model(
            classical_model_path=merged_classical_path,
            exchange_model_path=component_checkpoints["exchange_path"],
            use_GPU=False,
        )


def test_neither_construction_form_raises():
    with pytest.raises(ValueError, match="requires either"):
        CLIFF2Model(use_GPU=False)


def test_single_checkpoint_and_component_paths_agree(
    merged_classical_path, component_checkpoints, two_geom_dimers
):
    """The two documented construction forms are the same model."""
    single = CLIFF2Model(
        classical_model_path=merged_classical_path, use_GPU=False
    )
    components = CLIFF2Model(
        rackers_model_path=component_checkpoints["rackers_path"],
        exchange_model_path=component_checkpoints["exchange_path"],
        use_GPU=False,
    )
    single_state = single.model.state_dict()
    component_state = components.model.state_dict()
    assert set(single_state) == set(component_state)
    for key, value in single_state.items():
        assert torch.equal(value, component_state[key]), key

    from_single = single.predict_qcel_mols_dimer(two_geom_dimers, batch_size=2)
    from_components = components.predict_qcel_mols_dimer(
        two_geom_dimers, batch_size=2
    )
    assert np.array_equal(from_single, from_components)
    assert single.include_overlap == components.include_overlap


def test_constructor_does_not_expose_include_overlap():
    parameters = inspect.signature(CLIFF2Model.__init__).parameters
    assert "include_overlap" not in parameters


@pytest.mark.parametrize(
    "harness_type,expected_overlap,expected_mode",
    [
        (CliffClassicalModel, False, "cliff_classical"),
        (CliffClassicalOverlapModel, True, "cliff_classical_overlap"),
    ],
    ids=["cliff_classical", "cliff_classical_overlap"],
)
def test_include_overlap_comes_from_the_checkpoint(
    tmp_path,
    nested_hfvr_vw_model,
    harness_type,
    expected_overlap,
    expected_mode,
):
    _, path = _classical_checkpoint(
        tmp_path, nested_hfvr_vw_model, harness_type, "classical.pt"
    )
    model = CLIFF2Model(classical_model_path=path, use_GPU=False)
    assert model.source_dimer_eval == expected_mode
    assert model.include_overlap is expected_overlap
    assert model.dimer_model.include_overlap is expected_overlap
    # The DimerProp mode itself is always `cliff_classical_d3`.
    assert model.DIMER_EVAL == CLIFF2_DIMER_EVAL
    assert model.dimer_model.forward.__name__ == (
        "_cliff_classical_d3_forward"
    )


def test_include_overlap_reaches_the_induction_kernel(
    tmp_path, nested_hfvr_vw_model, synthetic_dimer_batch, monkeypatch
):
    """The checkpoint's overlap flag is what ``rackers_thole_induction`` sees.

    Asserted on the kernel argument rather than on an energy difference: the
    overlap correction can be numerically negligible for a given parameter set,
    which would make an energy-difference assertion pass or fail for reasons
    unrelated to the plumbing under test.
    """
    from apnet_pt.AtomPairwiseModels import mtp_mtp

    for expected, harness_type, name in (
        (False, CliffClassicalModel, "plain.pt"),
        (True, CliffClassicalOverlapModel, "overlap.pt"),
    ):
        _, path = _classical_checkpoint(
            tmp_path, nested_hfvr_vw_model, harness_type, name
        )
        model = CLIFF2Model(classical_model_path=path, use_GPU=False)
        seen = []
        original = mtp_mtp.rackers_thole_induction

        def _spy(*args, **kwargs):
            seen.append(kwargs["include_overlap"])
            return original(*args, **kwargs)

        monkeypatch.setattr(mtp_mtp, "rackers_thole_induction", _spy)
        model(synthetic_dimer_batch)
        monkeypatch.undo()
        assert seen == [expected], name


def test_rejects_a_checkpoint_that_is_not_a_classical_head(
    component_checkpoints,
):
    with pytest.raises(ValueError, match="model_type mismatch"):
        CLIFF2Model(
            classical_model_path=component_checkpoints["exchange_path"],
            use_GPU=False,
        )


def test_rejects_a_reordered_classical_parameter_list(merged_classical_path):
    checkpoint = model_io.load_checkpoint(merged_classical_path)
    checkpoint["config"]["parameter_names"] = list(
        reversed(CLIFF_CLASSICAL_PARAMETER_NAMES)
    )
    model_io.save_checkpoint(checkpoint, merged_classical_path)
    with pytest.raises(ValueError, match="parameter_names must exactly match"):
        CLIFF2Model(
            classical_model_path=merged_classical_path, use_GPU=False
        )


def test_rejects_an_unknown_dimer_eval(merged_classical_path):
    checkpoint = model_io.load_checkpoint(merged_classical_path)
    checkpoint["config"]["dimer_eval"] = "rackers_thole"
    model_io.save_checkpoint(checkpoint, merged_classical_path)
    with pytest.raises(ValueError, match="is not a CLIFF classical mode"):
        CLIFF2Model(
            classical_model_path=merged_classical_path, use_GPU=False
        )


# ---------------------------------------------------------------------------
# CLIFF2Model inference surface
# ---------------------------------------------------------------------------


def test_forward_returns_four_columns_per_edge(
    classical_overlap_checkpoint, synthetic_dimer_batch
):
    model = CLIFF2Model(
        classical_model_path=classical_overlap_checkpoint, use_GPU=False
    )
    edge_energy = model(synthetic_dimer_batch)
    n_edges = synthetic_dimer_batch.e_ABfull_source.numel()
    assert edge_energy.shape == (n_edges, 4)
    assert torch.isfinite(edge_energy).all()


def test_predict_batch_shape_and_total(
    classical_overlap_checkpoint, synthetic_dimer_batch
):
    model = CLIFF2Model(
        classical_model_path=classical_overlap_checkpoint, use_GPU=False
    )
    preds = model.predict_batch(synthetic_dimer_batch)
    n_dimers = synthetic_dimer_batch.total_charge_A.size(0)
    assert preds.shape == (n_dimers, 5)
    assert torch.isfinite(preds).all()
    assert torch.allclose(preds[:, :4].sum(dim=-1), preds[:, 4], atol=1e-6)
    assert len(CLIFF2_COMPONENT_LABELS) == preds.shape[1]


def test_predict_batch_matches_scattered_forward(
    classical_overlap_checkpoint, synthetic_dimer_batch
):
    """``predict_batch`` is the forward aggregated with ``dimer_ind_full``."""
    from apnet_pt.util import scatter_sum_compile

    model = CLIFF2Model(
        classical_model_path=classical_overlap_checkpoint, use_GPU=False
    )
    edge_energy = model(synthetic_dimer_batch)
    index = model._dimer_index_for_output(synthetic_dimer_batch)
    assert index is synthetic_dimer_batch.dimer_ind_full
    assert CLIFF2_DIMER_EVAL in FULL_EDGE_DIMER_EVAL_MODES
    expected = scatter_sum_compile(
        edge_energy,
        index,
        dim_size=synthetic_dimer_batch.total_charge_A.size(0),
    )
    preds = model.predict_batch(synthetic_dimer_batch)
    assert torch.allclose(preds[:, :4], expected, atol=1e-8)


def test_component_labels():
    assert CLIFF2Model.component_labels == (
        "Elst",
        "Exch",
        "Indu",
        "Disp",
        "Total",
    )
    assert CLIFF2_COMPONENT_LABELS == CLIFF2Model.component_labels


def test_model_is_eval_and_gradient_free(classical_overlap_checkpoint):
    model = CLIFF2Model(
        classical_model_path=classical_overlap_checkpoint, use_GPU=False
    )
    assert not model.training
    for module in model.modules():
        assert not module.training
    parameters = list(model.parameters())
    assert parameters
    assert all(not parameter.requires_grad for parameter in parameters)


def test_prediction_entry_points_run_under_inference_mode(
    classical_overlap_checkpoint, synthetic_dimer_batch
):
    model = CLIFF2Model(
        classical_model_path=classical_overlap_checkpoint, use_GPU=False
    )
    preds = model.predict_batch(synthetic_dimer_batch)
    assert preds.is_inference()


def test_d3_override_changes_only_the_dispersion_column(
    classical_overlap_checkpoint, synthetic_dimer_batch
):
    default_model = CLIFF2Model(
        classical_model_path=classical_overlap_checkpoint, use_GPU=False
    )
    overridden = CLIFF2Model(
        classical_model_path=classical_overlap_checkpoint,
        d3_damping_parameters={"s6": 0.5, "s8": 0.0},
        use_GPU=False,
    )
    assert overridden.d3_damping_parameters["s6"] == 0.5
    assert overridden.d3_damping_parameters["s8"] == 0.0
    # Untouched keys keep the resolved defaults.
    assert (
        overridden.d3_damping_parameters["a1"]
        == default_model.d3_damping_parameters["a1"]
    )

    baseline = default_model.predict_batch(synthetic_dimer_batch)
    changed = overridden.predict_batch(synthetic_dimer_batch)
    for column in range(3):
        assert torch.allclose(
            baseline[:, column], changed[:, column], atol=0.0, rtol=0.0
        ), column
    assert not torch.allclose(baseline[:, 3], changed[:, 3])
    # The total tracks the changed dispersion.
    assert torch.allclose(changed[:, :4].sum(dim=-1), changed[:, 4], atol=1e-6)


def test_d3_override_rejects_unknown_parameters(
    classical_overlap_checkpoint,
):
    with pytest.raises(ValueError, match="Unknown D3 damping parameter"):
        CLIFF2Model(
            classical_model_path=classical_overlap_checkpoint,
            d3_damping_parameters={"nope": 1.0},
            use_GPU=False,
        )


def test_predict_qcel_mols_dimer_on_a_two_dimer_fixture(
    classical_overlap_checkpoint, two_geom_dimers
):
    model = CLIFF2Model(
        classical_model_path=classical_overlap_checkpoint, use_GPU=False
    )
    preds = model.predict_qcel_mols_dimer(two_geom_dimers, batch_size=2)
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (2, 5)
    assert np.isfinite(preds).all()
    assert np.allclose(preds[:, :4].sum(axis=1), preds[:, 4], atol=1e-6)
    # Batching must not change the answer.
    one_at_a_time = model.predict_qcel_mols_dimer(
        two_geom_dimers, batch_size=1
    )
    assert np.allclose(preds, one_at_a_time, atol=1e-6)


def test_predict_qcel_mols_dimer_mirrors_the_harness_signature():
    harness_parameters = inspect.signature(
        RackersTholeDampingModel.predict_qcel_mols_dimer
    ).parameters
    cliff2_parameters = inspect.signature(
        CLIFF2Model.predict_qcel_mols_dimer
    ).parameters
    for name in ("mols", "batch_size", "r_cut", "verbose"):
        assert name in cliff2_parameters, name
        assert (
            cliff2_parameters[name].default
            == harness_parameters[name].default
        ), name


def test_save_load_round_trip_reproduces_predictions(
    tmp_path, merged_classical_path, two_geom_dimers
):
    model = CLIFF2Model(
        classical_model_path=merged_classical_path,
        d3_damping_parameters={"s8": 0.5},
        use_GPU=False,
    )
    before = model.predict_qcel_mols_dimer(two_geom_dimers, batch_size=2)

    path = tmp_path / "cliff2.pt"
    model.save_model(path)
    checkpoint = model_io.load_checkpoint(path)
    assert checkpoint["model_type"] == CLIFF2_MODEL_TYPE
    config = checkpoint["config"]
    assert config["model_type"] == CLIFF2_MODEL_TYPE
    # The source classical config plus the resolved D3 parameters, so the
    # reload needs none of the constituent files.
    assert config["classical_config"]["parameter_names"] == list(
        CLIFF_CLASSICAL_PARAMETER_NAMES
    )
    assert config["d3_damping_parameters"]["s8"] == 0.5
    assert config["include_overlap"] is model.include_overlap
    assert config["component_labels"] == list(CLIFF2_COMPONENT_LABELS)

    # Delete the source so a lingering dependency would fail loudly.
    merged_classical_path.unlink()
    reloaded = CLIFF2Model.from_checkpoint(path, use_GPU=False)
    assert reloaded.d3_damping_parameters == model.d3_damping_parameters
    assert reloaded.include_overlap == model.include_overlap
    after = reloaded.predict_qcel_mols_dimer(two_geom_dimers, batch_size=2)
    assert np.allclose(before, after, atol=1e-10)

    # And it round-trips again.
    second = tmp_path / "cliff2-second.pt"
    reloaded.save_model(second)
    twice = CLIFF2Model.from_checkpoint(second, use_GPU=False)
    assert np.allclose(
        after,
        twice.predict_qcel_mols_dimer(two_geom_dimers, batch_size=2),
        atol=1e-10,
    )


def test_cliff2_checkpoint_missing_classical_config_raises(
    tmp_path, merged_classical_path
):
    model = CLIFF2Model(
        classical_model_path=merged_classical_path, use_GPU=False
    )
    path = tmp_path / "broken.pt"
    model.save_model(path)
    checkpoint = model_io.load_checkpoint(path)
    checkpoint["config"].pop("classical_config")
    model_io.save_checkpoint(checkpoint, path)
    with pytest.raises(ValueError, match="missing classical_config"):
        CLIFF2Model.from_checkpoint(path, use_GPU=False)


def test_info_renders_a_model_tree(classical_overlap_checkpoint, capsys):
    model = CLIFF2Model(
        classical_model_path=classical_overlap_checkpoint, use_GPU=False
    )
    model.info()
    printed = capsys.readouterr().out
    assert "CLIFF2Model" in printed
    assert "AtomMPNN" in printed


# ---------------------------------------------------------------------------
# Physics sanity
# ---------------------------------------------------------------------------


def test_close_contact_exchange_is_repulsive_and_dispersion_attractive(
    tmp_path, nested_hfvr_vw_model
):
    """On a close-contact water dimer, Exch > 0 and Disp < 0.

    The correction MLPs are zeroed so the head reproduces its documented
    initialization (``K_exch = 2.5``, valence widths ``0.4`` bohr) and the
    predicted magnitudes land on a physical scale rather than a random one.
    """
    _, path = _classical_checkpoint(
        tmp_path,
        nested_hfvr_vw_model,
        CliffClassicalOverlapModel,
        "physical.pt",
        physical=True,
    )
    model = CLIFF2Model(classical_model_path=path, use_GPU=False)

    close = _load_dimer(_WATER_CLOSE)
    far = _load_dimer(_TWO_GEOM / "water_dimer.dat")
    preds = model.predict_qcel_mols_dimer([close, far], batch_size=2)

    exch_close, disp_close = preds[0, 1], preds[0, 3]
    exch_far, disp_far = preds[1, 1], preds[1, 3]

    assert exch_close > 0.0, "exchange must be repulsive"
    assert exch_far > 0.0
    assert disp_close < 0.0, "dispersion must be attractive"
    assert disp_far < 0.0
    # Both terms are short ranged, so the close contact dominates.
    assert exch_close > exch_far
    assert disp_close < disp_far

    # Per-edge exchange is strictly positive everywhere, not merely on the sum.
    from apnet_pt.pt_datasets.ap2_fused_ds import (
        ap2_fused_collate_update_no_target,
        qcel_dimer_to_fused_data,
    )

    batch = ap2_fused_collate_update_no_target(
        [qcel_dimer_to_fused_data(close, r_cut=5.0, dimer_ind=0,
                                  r_cut_im=torch.inf)]
    )
    edge_energy = model(batch)
    assert torch.all(edge_energy[:, 1] > 0.0)
    assert torch.all(edge_energy[:, 3] < 0.0)


# ---------------------------------------------------------------------------
# Packaging / inference-only posture
# ---------------------------------------------------------------------------


def test_cliff2_is_registered_in_the_package_namespace():
    from apnet_pt import AtomPairwiseModels

    assert AtomPairwiseModels.CLIFF2Model is CLIFF2Model
    assert (
        AtomPairwiseModels.merge_classical_parameter_checkpoints
        is merge_classical_parameter_checkpoints
    )
    assert AtomPairwiseModels.cliff_2 is cliff_2
    assert "CLIFF2Model" in AtomPairwiseModels.__all__


@pytest.mark.parametrize(
    "forbidden",
    [
        "torch.optim",
        "DistributedDataParallel",
        "torch.distributed",
        "module_dataset",
        "DataLoader",
        "backward()",
    ],
)
def test_module_contains_no_training_machinery(forbidden):
    """``cliff_2.py`` is inference-only by construction, not by convention."""
    source = pathlib.Path(cliff_2.__file__).read_text()
    assert forbidden not in source
    assert not hasattr(CLIFF2Model, "train_model")
    # `nn.Module.train` is inherited and unavoidable; nothing else may be.
    assert "def train" not in source


# ---------------------------------------------------------------------------
# train_models.py --merge_* CLI entry point
# ---------------------------------------------------------------------------


def _run_train_models(*cli_args):
    """Invoke train_models.py as a subprocess from the repository root."""
    root = pathlib.Path(__file__).resolve().parents[1]
    # The ambient environment may have apnet_pt installed from a different
    # checkout, so pin the subprocess to this worktree's sources.
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(root / "src")] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    return subprocess.run(
        [sys.executable, "train_models.py", *cli_args],
        capture_output=True,
        text=True,
        cwd=root,
        env=env,
    )


def test_merge_cli_writes_a_name_mapped_classical_checkpoint(
    tmp_path, component_checkpoints
):
    """--merge_* merges by parameter name and exits without training."""
    output_path = tmp_path / "merged_via_cli.pt"
    result = _run_train_models(
        "--merge_rackers_checkpoint",
        str(component_checkpoints["rackers_path"]),
        "--merge_exchange_checkpoint",
        str(component_checkpoints["exchange_path"]),
        "--merge_output_path",
        str(output_path),
    )
    assert result.returncode == 0, result.stderr
    assert output_path.exists()

    merged = model_io.load_checkpoint(str(output_path))
    assert merged["model_type"] == "CliffClassicalNN"
    assert merged["config"]["parameter_names"] == list(
        CLIFF_CLASSICAL_PARAMETER_NAMES
    )

    merged_state = merged["model_state_dict"]
    rackers_state = model_io.load_checkpoint(
        str(component_checkpoints["rackers_path"])
    )["model_state_dict"]
    exchange_state = model_io.load_checkpoint(
        str(component_checkpoints["exchange_path"])
    )["model_state_dict"]

    for source_column, name in enumerate(RACKERS_PARAMETER_NAMES):
        destination = CLIFF_CLASSICAL_PARAMETER_NAMES.index(name)
        assert _columns_equal(
            merged_state, destination, rackers_state, source_column
        ), name
    assert _columns_equal(
        merged_state,
        CLIFF_CLASSICAL_EXCH_INDEX,
        exchange_state,
        0,
    )


@pytest.mark.parametrize(
    "cli_args, reason",
    [
        (("--merge_rackers_checkpoint",), "missing --merge_output_path"),
        (("--merge_exchange_checkpoint",), "missing --merge_output_path"),
        (("--merge_output_path",), "no source checkpoint"),
    ],
)
def test_merge_cli_rejects_incomplete_argument_sets(
    tmp_path, component_checkpoints, cli_args, reason
):
    """Incomplete --merge_* combinations fail loudly instead of training."""
    if cli_args[0] == "--merge_output_path":
        value = str(tmp_path / "unused.pt")
    elif cli_args[0] == "--merge_rackers_checkpoint":
        value = str(component_checkpoints["rackers_path"])
    else:
        value = str(component_checkpoints["exchange_path"])

    result = _run_train_models(cli_args[0], value)
    assert result.returncode != 0, reason
    assert "ValueError" in result.stderr
    assert not (tmp_path / "unused.pt").exists()
