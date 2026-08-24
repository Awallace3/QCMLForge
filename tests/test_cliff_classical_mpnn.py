"""Tests for the message-passing CLIFF classical parameter head.

``CliffClassicalNN`` reads the frozen ``AtomMPNN`` hidden states through one MLP
per parameter per message step. Those states were fitted to reproduce
multipoles, and nothing in that objective asks them to encode what a damping
exponent or an exchange amplitude depends on; on top of that, every parameter is
a purely per-atom function of them, with no learnable exchange of information
between neighbours.

``CliffClassicalMPNN`` keeps the output contract byte-for-byte -- five columns in
``CLIFF_CLASSICAL_PARAMETER_NAMES`` order, strictly positive, with the nested
model's Hirshfeld volume ratio and valence width still at ``output[-2]`` -- and
replaces the featurizer with its own trainable message passing. The tests here
are mostly about that contract holding, because it is what lets the classical
physics path stay untouched.
"""
import inspect
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apnet_pt import model_io  # noqa: E402
from apnet_pt.AtomModels.ap2_atom_model import AtomMPNN  # noqa: E402
from apnet_pt.AtomPairwiseModels import mtp_mtp  # noqa: E402
from apnet_pt.AtomPairwiseModels.mtp_mtp import (  # noqa: E402
    CLIFF_CLASSICAL_ELST_INDEX,
    CLIFF_CLASSICAL_EXCH_INDEX,
    CLIFF_CLASSICAL_INITIAL_VALUES,
    CLIFF_CLASSICAL_IND_OVERLAP_INDEX,
    CLIFF_CLASSICAL_PARAMETER_NAMES,
    CLIFF_CLASSICAL_THOLE_DIRECT_INDEX,
    CLIFF_CLASSICAL_THOLE_MUTUAL_INDEX,
    CLIFF_EXCH_INITIAL_VALUES_BY_Z,
    CLIFF_MPNN_SCALAR_FEATURES,
    AtomTypeParamNN,
    CliffClassicalMPNN,
    CliffClassicalNN,
    CliffClassicalOverlapMPNNModel,
)

import train_models  # noqa: E402

HEAD_KWARGS = dict(n_message=1, n_neuron=8, n_embed=4)
MPNN_KWARGS = dict(
    param_n_message=2, param_n_rbf=4, param_hidden=8, param_r_cut=5.0
)


def _head(nested, **overrides):
    kwargs = {**HEAD_KWARGS, **MPNN_KWARGS, **overrides}
    return CliffClassicalMPNN(atom_model=nested, **kwargs)


# ---------------------------------------------------------------------------
# Output contract: what the untouched physics path reads


def test_output_contract_matches_the_dense_head(
    atomic_batch, nested_hfvr_vw_model
):
    """Same tuple layout, so `DimerProp` needs no branch on head type.

    The classical physics reads `output[-1]` for the five parameters and
    `output[-2]` for the nested volume ratio and valence width. A head that
    shifted either index would produce a wrong-but-plausible energy rather than
    an error.
    """
    torch.manual_seed(0)
    mpnn = _head(nested_hfvr_vw_model)
    dense = CliffClassicalNN(atom_model=nested_hfvr_vw_model, **HEAD_KWARGS)

    mpnn_out = mpnn(atomic_batch)
    dense_out = dense(atomic_batch)
    assert len(mpnn_out) == len(dense_out)
    assert mpnn_out[-1].shape == dense_out[-1].shape
    assert mpnn_out[-2].shape == dense_out[-2].shape
    n_atoms = atomic_batch.x.numel()
    assert mpnn_out[-1].shape == (n_atoms, len(CLIFF_CLASSICAL_PARAMETER_NAMES))
    # The nested physical outputs must pass through untouched, not be recomputed.
    assert torch.equal(mpnn_out[-2], dense_out[-2])


def test_parameters_are_finite_and_strictly_positive(
    atomic_batch, nested_hfvr_vw_model
):
    torch.manual_seed(0)
    parameters = _head(nested_hfvr_vw_model)(atomic_batch)[-1]
    assert torch.isfinite(parameters).all()
    assert (parameters > 0).all()


def test_contract_and_model_type_are_registered(nested_hfvr_vw_model):
    assert (
        mtp_mtp.POSITIVE_PARAMETER_CONTRACTS["CliffClassicalMPNN"]
        == CLIFF_CLASSICAL_PARAMETER_NAMES
    )
    assert mtp_mtp._CLIFF_PARAMETER_HEADS["CliffClassicalMPNN"] is CliffClassicalMPNN
    torch.manual_seed(0)
    head = _head(nested_hfvr_vw_model)
    assert head.get_config()["model_type"] == "CliffClassicalMPNN"
    assert head.n_params == 5


def test_n_params_is_not_a_constructor_argument():
    """The contract fixes the count, as for every other positive head."""
    assert "n_params" not in inspect.signature(
        CliffClassicalMPNN.__init__
    ).parameters


# ---------------------------------------------------------------------------
# Initialization: the per-element CLIFF seeds must survive the new featurizer


def test_zeroed_corrections_recover_the_per_element_seeds(
    atomic_batch, nested_hfvr_vw_model
):
    """With every correction silenced, the seed is the output.

    This is the property the exchange collapse investigation depends on: the
    head must *start* at CLIFF Table I values, not somewhere a random readout
    put it. The element embedding is initialized to zero for the same reason,
    so silencing the readouts is enough.
    """
    torch.manual_seed(0)
    # Zero the initialization spread so the seed is exact rather than
    # seed +- noise, exactly as the dense head's equivalent test does.
    head = _head(
        nested_hfvr_vw_model,
        param_start_std=[0.0] * len(CLIFF_CLASSICAL_PARAMETER_NAMES),
    )
    with torch.no_grad():
        for parameter_head in head.param_readout_layers:
            for readout in parameter_head:
                for parameter in readout.parameters():
                    parameter.zero_()
    parameters = head(atomic_batch)[-1]

    # atomic_batch is water: O, H, H.
    oxygen, hydrogen = parameters[0], parameters[1]
    assert oxygen[CLIFF_CLASSICAL_EXCH_INDEX].item() == pytest.approx(
        CLIFF_EXCH_INITIAL_VALUES_BY_Z[8], rel=1e-3
    )
    assert hydrogen[CLIFF_CLASSICAL_EXCH_INDEX].item() == pytest.approx(
        CLIFF_EXCH_INITIAL_VALUES_BY_Z[1], rel=1e-3
    )
    for index in (
        CLIFF_CLASSICAL_ELST_INDEX,
        CLIFF_CLASSICAL_THOLE_DIRECT_INDEX,
        CLIFF_CLASSICAL_THOLE_MUTUAL_INDEX,
        CLIFF_CLASSICAL_IND_OVERLAP_INDEX,
    ):
        # Columns with no per-element table keep their scalar seed.
        assert oxygen[index].item() == pytest.approx(
            CLIFF_CLASSICAL_INITIAL_VALUES[index], rel=1e-4
        )
    # Water is O/H/H, so the exchange column must not come back constant --
    # that was the failure mode a uniform seed produced.
    assert parameters[:, CLIFF_CLASSICAL_EXCH_INDEX].std().item() > 1.0


def test_element_embedding_starts_at_zero(nested_hfvr_vw_model):
    """A randomly seeded element embedding would move every parameter off its
    Table I value before training starts."""
    torch.manual_seed(0)
    head = _head(nested_hfvr_vw_model)
    assert torch.count_nonzero(head.param_type_embed.weight) == 0


# ---------------------------------------------------------------------------
# The per-parameter independence the dense head has, kept


def test_parameter_heads_stay_independent(atomic_batch, nested_hfvr_vw_model):
    """Each column's gradient must touch only its own readout.

    These are per-component atom-type parameters: electrostatic damping and
    Thole damping are separate physics and must not share a learned parameter.
    The message passing is deliberately *shared* -- it is one featurizer -- so
    the independence has to live in the readouts, and this pins it there.
    """
    torch.manual_seed(0)
    head = _head(nested_hfvr_vw_model)
    names = list(CLIFF_CLASSICAL_PARAMETER_NAMES)

    for column in range(len(names)):
        head.zero_grad(set_to_none=True)
        head(atomic_batch)[-1][:, column].sum().backward()
        for p, name in enumerate(names):
            touched = any(
                q.grad is not None and bool(q.grad.abs().sum() > 0)
                for readout in head.param_readout_layers[p]
                for q in readout.parameters()
            )
            embedding_grad = head.guess_layer[p].weight.grad
            touched = touched or (
                embedding_grad is not None
                and bool(embedding_grad.abs().sum() > 0)
            )
            assert touched is (p == column), (
                f"d(K[:, {column}])/d({name} head) should be "
                f"{'nonzero' if p == column else 'zero'}"
            )


# ---------------------------------------------------------------------------
# What the new featurizer buys, asserted rather than assumed


def test_message_passing_parameters_are_trainable_and_reached(
    atomic_batch, nested_hfvr_vw_model
):
    torch.manual_seed(0)
    head = _head(nested_hfvr_vw_model)
    head(atomic_batch)[-1].sum().backward()
    for name in (
        "param_input_layer",
        "param_type_embed",
        "param_update_layers",
        "param_distance_layer",
    ):
        module = getattr(head, name)
        gradients = [q.grad for q in module.parameters()]
        assert gradients, name
        assert all(g is not None for g in gradients), name
        assert any(bool(g.abs().sum() > 0) for g in gradients), name


def test_nested_atom_model_is_still_frozen_by_default(nested_hfvr_vw_model):
    """The point is a freer parameter featurizer, not fine-tuning multipoles."""
    torch.manual_seed(0)
    head = _head(nested_hfvr_vw_model)
    trunk = [
        q for n, q in head.named_parameters() if n.startswith("atom_model.")
    ]
    assert trunk
    assert not any(q.requires_grad for q in trunk)


def test_a_neighbours_geometry_moves_a_parameter(
    atomic_batch, nested_hfvr_vw_model
):
    """The head's own message passing must actually consume the graph.

    If the distance basis or the aggregation were disconnected, this would still
    pass through the frozen hidden states -- so the trainable path is isolated
    by zeroing the readouts' dependence on nothing and instead checking that a
    gradient flows to the *parameter head's own* radial basis.
    """
    torch.manual_seed(0)
    head = _head(nested_hfvr_vw_model)
    head.zero_grad(set_to_none=True)
    head(atomic_batch)[-1].sum().backward()
    frequencies = head.param_distance_layer.frequencies
    assert frequencies.grad is not None
    assert bool(frequencies.grad.abs().sum() > 0)


def test_scalar_features_are_the_documented_five(nested_hfvr_vw_model):
    """The feature width is static, and its parts are named.

    A silently wider input layer would still train, and would then refuse to
    load its own checkpoint.
    """
    assert CLIFF_MPNN_SCALAR_FEATURES == (
        "charge",
        "dipole_norm",
        "quadrupole_norm",
        "hirshfeld_volume_ratio",
        "valence_width",
    )
    torch.manual_seed(0)
    head = _head(nested_hfvr_vw_model)
    inner = nested_hfvr_vw_model.atom_model
    expected = (inner.n_message + 1) * inner.n_embed + 5
    assert head.param_feature_width == expected
    assert head.param_input_layer.in_features == expected


def test_single_atom_monomer_returns_the_seed(nested_hfvr_vw_model):
    """No edges means no message passing, so the seed is the answer."""
    from torch_geometric.data import Data

    lone = Data(
        x=torch.tensor([8], dtype=torch.long),
        R=torch.zeros(1, 3),
        edge_index=torch.zeros(2, 0, dtype=torch.long),
        molecule_ind=torch.zeros(1, dtype=torch.long),
        total_charge=torch.tensor([0.0]),
        natom_per_mol=torch.tensor([1], dtype=torch.long),
    )
    torch.manual_seed(0)
    head = _head(nested_hfvr_vw_model)
    parameters = head(lone)[-1]
    assert parameters.shape == (1, len(CLIFF_CLASSICAL_PARAMETER_NAMES))
    assert torch.isfinite(parameters).all()
    assert (parameters > 0).all()


# ---------------------------------------------------------------------------
# Architecture validation and round-tripping


@pytest.mark.parametrize(
    "knob", ["param_n_message", "param_n_rbf", "param_hidden"]
)
@pytest.mark.parametrize("bad", [0, -1, 1.5, "2"])
def test_integer_architecture_knobs_are_validated(
    nested_hfvr_vw_model, knob, bad
):
    with pytest.raises((ValueError, TypeError), match=knob):
        _head(nested_hfvr_vw_model, **{knob: bad})


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf"), "5"])
def test_cutoff_is_validated(nested_hfvr_vw_model, bad):
    with pytest.raises((ValueError, TypeError), match="param_r_cut"):
        _head(nested_hfvr_vw_model, param_r_cut=bad)


def test_config_records_the_architecture(nested_hfvr_vw_model):
    torch.manual_seed(0)
    head = _head(nested_hfvr_vw_model)
    config = head.get_config()
    assert CliffClassicalMPNN.ARCHITECTURE_CONFIG_KEYS == (
        "param_n_message",
        "param_n_rbf",
        "param_hidden",
        "param_r_cut",
    )
    for key, value in MPNN_KWARGS.items():
        assert config[key] == value
    # The shared keys must survive too, or a checkpoint loses its seeds.
    assert config["parameter_names"] == list(CLIFF_CLASSICAL_PARAMETER_NAMES)


def test_dense_head_declares_no_architecture_keys():
    """So the shared construction path forwards nothing to it."""
    assert CliffClassicalNN.ARCHITECTURE_CONFIG_KEYS == ()
    assert mtp_mtp._CliffPositiveParamNN.ARCHITECTURE_CONFIG_KEYS == ()


def test_nested_model_must_be_an_atomtypeparamnn():
    with pytest.raises(ValueError, match="AtomTypeParamNN"):
        CliffClassicalMPNN(atom_model=AtomMPNN(n_message=1, n_neuron=8, n_embed=4))


def test_innermost_atom_mpnn_walks_the_stack(nested_hfvr_vw_model):
    inner = mtp_mtp._innermost_atom_mpnn(nested_hfvr_vw_model)
    assert inner is nested_hfvr_vw_model.atom_model
    with pytest.raises(ValueError, match="AtomMPNN"):
        mtp_mtp._innermost_atom_mpnn(torch.nn.Linear(1, 1))


# ---------------------------------------------------------------------------
# Harness and CLI


def test_harness_selects_the_new_head_and_the_overlap_mode():
    assert CliffClassicalOverlapMPNNModel.MODEL_TYPE == "CliffClassicalMPNN"
    assert CliffClassicalOverlapMPNNModel.DIMER_EVAL == "cliff_classical_overlap"
    assert (
        CliffClassicalOverlapMPNNModel.PARAMETER_NAMES
        == CLIFF_CLASSICAL_PARAMETER_NAMES
    )
    # Same physics as the dense overlap route, so the same dimer mode.
    assert (
        CliffClassicalOverlapMPNNModel.DIMER_EVAL
        == mtp_mtp.CliffClassicalOverlapModel.DIMER_EVAL
    )


def test_cli_route_resolves_to_the_harness():
    route = "CliffClassicalOverlapMPNNModel"
    assert route in train_models.CLIFF_MODEL_TYPES
    assert route in train_models.COMBINED_CLIFF_MODEL_TYPES
    assert route in train_models.CLIFF_MPNN_MODEL_TYPES
    assert (
        getattr(train_models.AtomPairwiseModels.mtp_mtp, route)
        is CliffClassicalOverlapMPNNModel
    )
    # It inherits the five-parameter contract from the shared lookup.
    names, means, stds = train_models._cliff_parameter_contract(route)
    assert names == CLIFF_CLASSICAL_PARAMETER_NAMES
    assert len(means) == 5 and len(stds) == 5


@pytest.mark.parametrize(
    "flag", ["param_n_message", "param_n_rbf", "param_hidden", "param_r_cut"]
)
def test_architecture_flags_rejected_on_other_routes(tmp_path, flag):
    """The other heads have no message passing to size."""
    with pytest.raises(ValueError, match=flag):
        train_models.train_pairwise_model(
            apnet_model_type="CliffClassicalOverlapModel",
            model_out=str(tmp_path / "out.pt"),
            **{flag: 2 if flag != "param_r_cut" else 4.0},
        )


def test_harness_rejects_architecture_knobs_for_a_dense_head(
    nested_hfvr_vw_model,
):
    """Reached through `AM_DimerParam_Model`, which validates against the head."""
    with pytest.raises(ValueError, match="param_n_message"):
        mtp_mtp.AM_DimerParam_Model(
            atom_model=nested_hfvr_vw_model,
            atom_model_type="AtomTypeParamNN",
            model_type="CliffClassicalNN",
            ds_root=None,
            use_GPU=False,
            ignore_database_null=True,
            param_start_mean=CLIFF_CLASSICAL_INITIAL_VALUES,
            param_start_std=mtp_mtp.CLIFF_CLASSICAL_INITIAL_STDS,
            n_params=5,
            dimer_eval_type="cliff_classical_overlap",
            param_n_message=2,
        )


def test_help_advertises_the_route_and_its_flags():
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT / "src"), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    result = subprocess.run(
        [sys.executable, "train_models.py", "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    assert "CliffClassicalOverlapMPNNModel" in result.stdout
    for flag in (
        "--param_n_message",
        "--param_n_rbf",
        "--param_hidden",
        "--param_r_cut",
    ):
        assert flag in result.stdout


# ---------------------------------------------------------------------------
# End to end: the physics path and the checkpoint round trip


def test_drives_the_unchanged_classical_overlap_physics(
    nested_hfvr_vw_model, synthetic_dimer_batch
):
    """A real `DimerProp` forward in the mode the new harness selects.

    Three energy columns (electrostatics, exchange, induction) per
    intermolecular edge, finite, and the same shape the dense head produces --
    which is the whole claim: the physics is untouched.
    """
    import copy

    with torch.random.fork_rng():
        torch.manual_seed(0)
        mpnn_head = _head(copy.deepcopy(nested_hfvr_vw_model))
        dense_head = CliffClassicalNN(
            atom_model=copy.deepcopy(nested_hfvr_vw_model),
            freeze_atom_model=True,
            **HEAD_KWARGS,
        )
    mode = CliffClassicalOverlapMPNNModel.DIMER_EVAL
    mpnn_energy, _, _ = mtp_mtp.DimerProp(
        ATParam=mpnn_head, dimer_eval=mode, freeze_atom_model=True
    )(synthetic_dimer_batch)
    dense_energy, _, _ = mtp_mtp.DimerProp(
        ATParam=dense_head, dimer_eval=mode, freeze_atom_model=True
    )(synthetic_dimer_batch)

    n_edges = synthetic_dimer_batch.e_ABfull_source.numel()
    assert mpnn_energy.shape == (n_edges, 3)
    assert mpnn_energy.shape == dense_energy.shape
    assert torch.isfinite(mpnn_energy).all()


def test_energy_gradients_reach_the_message_passing(
    nested_hfvr_vw_model, synthetic_dimer_batch
):
    """The new featurizer must be trainable *through the physics*, not just
    through a bare head forward."""
    import copy

    with torch.random.fork_rng():
        torch.manual_seed(0)
        head = _head(copy.deepcopy(nested_hfvr_vw_model))
    dimer = mtp_mtp.DimerProp(
        ATParam=head,
        dimer_eval=CliffClassicalOverlapMPNNModel.DIMER_EVAL,
        freeze_atom_model=True,
    )
    energy, _, _ = dimer(synthetic_dimer_batch)
    energy.sum().backward()
    for name in ("param_input_layer", "param_update_layers"):
        gradients = [q.grad for q in getattr(head, name).parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in gradients), name
        assert any(bool(g.abs().sum() > 0) for g in gradients), name


def test_checkpoint_round_trips_the_architecture(
    tmp_path, nested_hfvr_vw_model, atomic_batch
):
    """Reloading must rebuild the same shape and reproduce the predictions.

    The architecture knobs are not in `state_dict`, so if `get_config` or the
    replay on load dropped one, the rebuilt head would either mismatch on
    `load_state_dict` or -- worse, if only the cutoff were lost -- load cleanly
    and predict something else.
    """
    architecture = dict(
        param_n_message=3, param_n_rbf=5, param_hidden=6, param_r_cut=4.25
    )
    harness = mtp_mtp.CliffClassicalOverlapMPNNModel(
        atom_model=nested_hfvr_vw_model,
        ds_root=None,
        use_GPU=False,
        ignore_database_null=True,
        n_message=1,
        n_neuron=8,
        n_embed=4,
        **architecture,
    )
    assert type(harness.model) is CliffClassicalMPNN
    for key, value in architecture.items():
        assert getattr(harness.model, key) == value

    before = harness.model(atomic_batch)[-1].detach().clone()
    path = tmp_path / "cliff2_mpnn.pt"
    model_io.save_checkpoint(harness._create_checkpoint(), str(path))

    reloaded = mtp_mtp.CliffClassicalOverlapMPNNModel(
        atom_model=None,
        pre_trained_model_path=str(path),
        ds_root=None,
        use_GPU=False,
        ignore_database_null=True,
    )
    assert type(reloaded.model) is CliffClassicalMPNN
    for key, value in architecture.items():
        assert getattr(reloaded.model, key) == value
    after = reloaded.model(atomic_batch)[-1].detach()
    assert torch.allclose(before, after, atol=1e-6)


def test_checkpoint_records_the_head_type_not_the_dense_one(
    tmp_path, nested_hfvr_vw_model
):
    """`model_type` has to name the architecture that produced the weights."""
    harness = mtp_mtp.CliffClassicalOverlapMPNNModel(
        atom_model=nested_hfvr_vw_model,
        ds_root=None,
        use_GPU=False,
        ignore_database_null=True,
        n_message=1,
        n_neuron=8,
        n_embed=4,
        **MPNN_KWARGS,
    )
    checkpoint = harness._create_checkpoint()
    assert checkpoint["model_type"] == "CliffClassicalMPNN"
    assert checkpoint["config"]["model_type"] == "CliffClassicalMPNN"


# ---------------------------------------------------------------------------
# Per-column bounds and the non-finite-gradient guard
#
# Measured, not assumed. With one global floor fraction of 0.05 the Thole floor
# is 0.017 against a 0.34 seed -- twenty times below any physical value. Both
# dense 50-epoch runs drove an induction column onto that floor, and this head
# reached all three inside a single epoch and then produced non-finite Thole
# values, killing job 12229494 in epoch 1 via `geometric_mean_edge_values`.


def test_every_column_floor_is_the_measured_best():
    """0.05 everywhere, restored after measuring the alternatives.

    Two tighter settings were tried over full 50-epoch runs on 100k dimers and
    both were worse on every component. Decisively, they made *induction* worse
    (2.431 against 1.897) -- the component the bound existed to protect -- which
    removes the rationale rather than weakening it. `exch` also cannot take a
    tight floor at all: hydrogen's Table I value is 0.31x the scalar seed.
    """
    floors = dict(
        zip(
            CLIFF_CLASSICAL_PARAMETER_NAMES,
            mtp_mtp.CLIFF_CLASSICAL_PARAM_FLOOR_FRACTION,
        )
    )
    assert set(floors.values()) == {0.05}
    hydrogen_fraction = (
        CLIFF_EXCH_INITIAL_VALUES_BY_Z[1] / CLIFF_CLASSICAL_INITIAL_VALUES[4]
    )
    assert floors["exch"] < hydrogen_fraction
    # The reasoning has to survive in the source, or the next person retries
    # the tighter floors without knowing they were measured.
    src = inspect.getsource(mtp_mtp)
    assert "A bound that fights the fit is worse than" in src


@pytest.mark.parametrize(
    "head_type", ["CliffClassicalNN", "CliffClassicalMPNN"]
)
def test_both_five_column_heads_share_the_same_floor(
    nested_hfvr_vw_model, head_type
):
    """The dense head has the same drift, so it gets the same treatment."""
    torch.manual_seed(0)
    cls = getattr(mtp_mtp, head_type)
    kwargs = dict(HEAD_KWARGS)
    if head_type == "CliffClassicalMPNN":
        kwargs.update(MPNN_KWARGS)
    head = cls(atom_model=nested_hfvr_vw_model, **kwargs)
    assert list(head.param_floor_fraction) == list(
        mtp_mtp.CLIFF_CLASSICAL_PARAM_FLOOR_FRACTION
    )
    floor = (
        torch.nn.functional.softplus(head.raw_parameter_floor)
        + head.positivity_epsilon
    ).reshape(-1)
    seeds = torch.tensor(CLIFF_CLASSICAL_INITIAL_VALUES)
    assert torch.allclose(floor, 0.05 * seeds, atol=1e-5)


def test_a_scalar_floor_is_still_accepted(nested_hfvr_vw_model):
    """Checkpoints written before this change record a single float."""
    torch.manual_seed(0)
    head = _head(nested_hfvr_vw_model, param_floor_fraction=0.05)
    floor = (
        torch.nn.functional.softplus(head.raw_parameter_floor)
        + head.positivity_epsilon
    ).reshape(-1)
    seeds = torch.tensor(CLIFF_CLASSICAL_INITIAL_VALUES)
    assert torch.allclose(floor, 0.05 * seeds, atol=1e-5)


@pytest.mark.parametrize(
    "bad,match",
    [
        ([0.5, 0.5], "exactly 5"),
        ([0.5, 0.5, 0.5, 0.5, 0.0], r"param_floor_fraction\[4\]"),
        ([0.5, 0.5, 0.5, 0.5, -1.0], r"param_floor_fraction\[4\]"),
        ("0.5", "sequence"),
    ],
)
def test_per_column_floor_is_validated(nested_hfvr_vw_model, bad, match):
    with pytest.raises(ValueError, match=match):
        _head(nested_hfvr_vw_model, param_floor_fraction=bad)


def test_per_column_floor_must_stay_below_the_ceiling(nested_hfvr_vw_model):
    """Compared per column, since either side may now vary across them."""
    with pytest.raises(ValueError, match="thole_direct"):
        _head(
            nested_hfvr_vw_model,
            param_floor_fraction=[0.05, 20.0, 0.5, 0.5, 0.05],
            param_ceiling_multiple=10.0,
        )


def test_per_column_floor_round_trips_through_config(
    tmp_path, nested_hfvr_vw_model, atomic_batch
):
    harness = mtp_mtp.CliffClassicalOverlapMPNNModel(
        atom_model=nested_hfvr_vw_model,
        ds_root=None,
        use_GPU=False,
        ignore_database_null=True,
        n_message=1,
        n_neuron=8,
        n_embed=4,
        **MPNN_KWARGS,
    )
    assert list(harness.model.param_floor_fraction) == list(
        mtp_mtp.CLIFF_CLASSICAL_PARAM_FLOOR_FRACTION
    )
    before = harness.model(atomic_batch)[-1].detach().clone()
    path = tmp_path / "floors.pt"
    model_io.save_checkpoint(harness._create_checkpoint(), str(path))
    reloaded = mtp_mtp.CliffClassicalOverlapMPNNModel(
        atom_model=None,
        pre_trained_model_path=str(path),
        ds_root=None,
        use_GPU=False,
        ignore_database_null=True,
    )
    assert list(reloaded.model.param_floor_fraction) == list(
        mtp_mtp.CLIFF_CLASSICAL_PARAM_FLOOR_FRACTION
    )
    assert torch.allclose(
        before, reloaded.model(atomic_batch)[-1].detach(), atol=1e-6
    )


def test_non_finite_gradient_skips_the_step_instead_of_the_run():
    """`clip_grad_norm_` scales by `max_norm / total_norm`.

    With a non-finite total norm that leaves every gradient nan, so the step
    writes nan into every weight and the run is over -- which is how job
    12229494 died. The guard drops the batch instead.
    """
    src = inspect.getsource(
        mtp_mtp.AM_DimerParam_Model._AM_DimerParam_Model__train_batches_single_proc
    )
    assert "total_norm = torch.nn.utils.clip_grad_norm_" in src
    assert "if not torch.isfinite(total_norm):" in src
    # The batch is dropped, not stepped, and the count is surfaced.
    assert "n_skipped += 1" in src
    assert "continue" in src
    assert "self.last_epoch_skipped_batches = n_skipped" in src
    # And a silent drop would misreport the run, so it prints.
    assert "WARNING: skipped" in src


# ---------------------------------------------------------------------------
# Hidden-state normalization
#
# Without it this head's pre-clip gradient norm measured 6.7e4 against the dense
# head's 1.2 on the same objective: `h (x) rbf` feeds a wide MLP whose input is
# summed over neighbours, and the output is fed back in for the next message
# step with nothing bounding the recursion. On real dimers that overflows
# float32 -- job 12235379 skipped 753 of 782 batches on non-finite gradients and
# its validation metrics never moved across an entire epoch.


def test_every_hidden_state_is_normalized(nested_hfvr_vw_model):
    torch.manual_seed(0)
    head = _head(nested_hfvr_vw_model)
    assert len(head.param_hidden_norms) == head.param_n_message + 1
    for norm in head.param_hidden_norms:
        assert isinstance(norm, torch.nn.LayerNorm)
        assert norm.normalized_shape == (head.param_hidden,)
    src = inspect.getsource(mtp_mtp.CliffClassicalMPNN._raw_head_output)
    # Normalized before being stored, so the next message step and every
    # readout see the bounded state, not just the readouts.
    assert "self.param_hidden_norms[0](" in src
    assert "self.param_hidden_norms[i + 1](" in src


def test_hidden_states_stay_order_one(atomic_batch, nested_hfvr_vw_model):
    """What the readouts consume must not scale with depth or coordination."""
    torch.manual_seed(0)
    head = _head(nested_hfvr_vw_model, param_n_message=3)
    captured = []
    for norm in head.param_hidden_norms:
        norm.register_forward_hook(
            lambda _m, _i, out: captured.append(out.detach())
        )
    head(atomic_batch)
    assert len(captured) == head.param_n_message + 1
    for step, state in enumerate(captured):
        rms = float(state.pow(2).mean().sqrt())
        assert rms < 10.0, f"hidden state {step} rms {rms}"


def test_gradient_scale_stays_near_the_dense_head(
    atomic_batch, nested_hfvr_vw_model
):
    """A regression bound on the failure that killed job 12235379.

    Deliberately loose -- two orders of magnitude -- because the point is to
    catch a return to five, not to pin a number.
    """
    import copy

    target = torch.zeros(atomic_batch.x.numel(), 5)
    target[:, 0] = CLIFF_CLASSICAL_INITIAL_VALUES[0]
    target[:, 4] = CLIFF_CLASSICAL_INITIAL_VALUES[4]

    def worst_grad_norm(build):
        torch.manual_seed(0)
        head = build()
        opt = torch.optim.Adam(
            [q for q in head.parameters() if q.requires_grad], lr=5e-4
        )
        worst = 0.0
        for _ in range(150):
            opt.zero_grad(set_to_none=True)
            ((head(atomic_batch)[-1] - target) ** 2).mean().backward()
            norm = torch.nn.utils.clip_grad_norm_(
                head.parameters(), max_norm=1.0
            )
            assert torch.isfinite(norm)
            worst = max(worst, float(norm))
            opt.step()
        return worst

    dense = worst_grad_norm(
        lambda: CliffClassicalNN(
            atom_model=copy.deepcopy(nested_hfvr_vw_model), **HEAD_KWARGS
        )
    )
    mpnn = worst_grad_norm(
        lambda: CliffClassicalMPNN(
            atom_model=copy.deepcopy(nested_hfvr_vw_model),
            **{**HEAD_KWARGS, **MPNN_KWARGS},
        )
    )
    assert mpnn < 100.0 * max(dense, 1.0), (
        f"message-passing head gradient norm {mpnn:.3e} vs dense {dense:.3e}"
    )


# ---------------------------------------------------------------------------
# Bound occupancy: measured, not constrained
#
# The tighter floors were reverted after full 50-epoch runs measured them worse
# on every component -- including induction itself, which was the thing the
# bound existed to protect. The drift onto a bound is real, but a bound that
# fights the fit costs more than the drift. So it is logged instead.


def test_classical_floor_is_back_to_the_measured_best():
    assert mtp_mtp.CLIFF_CLASSICAL_PARAM_FLOOR_FRACTION == (
        0.05, 0.05, 0.05, 0.05, 0.05
    )
    # The per-column machinery stays, so retrying is a one-line change.
    sig = inspect.signature(CliffClassicalMPNN.__init__)
    assert sig.parameters["param_floor_fraction"].default is (
        mtp_mtp.CLIFF_CLASSICAL_PARAM_FLOOR_FRACTION
    )


def test_bound_occupancy_reports_a_fraction_per_column_and_bound(
    atomic_batch, nested_hfvr_vw_model
):
    torch.manual_seed(0)
    head = _head(nested_hfvr_vw_model)
    occupancy = head.bound_occupancy(atomic_batch)
    for name in CLIFF_CLASSICAL_PARAMETER_NAMES:
        for bound in ("floor", "ceiling"):
            key = f"bounds/{name}_at_{bound}"
            assert key in occupancy, key
            assert 0.0 <= occupancy[key] <= 1.0
    # A freshly seeded head sits at neither bound.
    assert all(v == 0.0 for v in occupancy.values())


def test_bound_occupancy_detects_a_pinned_column(
    atomic_batch, nested_hfvr_vw_model
):
    """Drive one column onto its floor and confirm it is reported.

    This is the situation both dense 50-epoch runs reached without anything in
    the logs saying so.
    """
    torch.manual_seed(0)
    head = _head(nested_hfvr_vw_model)
    column = CLIFF_CLASSICAL_THOLE_DIRECT_INDEX
    with torch.no_grad():
        # Push the seed embedding far below the floor for every element.
        head.guess_layer[column].weight.fill_(-50.0)
        for readout in head.param_readout_layers[column]:
            for parameter in readout.parameters():
                parameter.zero_()
    occupancy = head.bound_occupancy(atomic_batch)
    pinned = f"bounds/{CLIFF_CLASSICAL_PARAMETER_NAMES[column]}_at_floor"
    assert occupancy[pinned] == 1.0
    for other, name in enumerate(CLIFF_CLASSICAL_PARAMETER_NAMES):
        if other != column:
            assert occupancy[f"bounds/{name}_at_floor"] == 0.0, name
    # And the clamp still returns a usable positive parameter, which is exactly
    # why this is invisible without the metric.
    parameters = head(atomic_batch)[-1]
    assert torch.isfinite(parameters).all() and (parameters > 0).all()


def test_bound_occupancy_reaches_the_tracked_epoch_payload():
    """It has to appear per epoch, not once in the run config."""
    eval_src = inspect.getsource(
        mtp_mtp.AM_DimerParam_Model._AM_DimerParam_Model__evaluate_batches_single_proc
    )
    assert "self.last_bound_occupancy" in eval_src
    # First validation batch only: one extra forward, not one per batch.
    assert "if n == 0" in eval_src

    from apnet_pt import training_tracking

    boundary_src = inspect.getsource(
        training_tracking._track_evaluation_boundary
    )
    assert 'extra_metrics=getattr(harness, "last_bound_occupancy", None)' in (
        boundary_src
    )
    log_src = inspect.getsource(training_tracking.log_epoch_metrics)
    assert "for key, value in (extra_metrics or {}).items():" in log_src
