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
        "frozen_parameters",
        "shared_damping_parameters",
        "param_n_message",
        "param_n_rbf",
        "param_hidden",
        "param_r_cut",
    )
    for key, value in MPNN_KWARGS.items():
        assert config[key] == value
    # The shared keys must survive too, or a checkpoint loses its seeds.
    assert config["parameter_names"] == list(CLIFF_CLASSICAL_PARAMETER_NAMES)


def test_dense_head_declares_only_the_shared_architecture_key():
    """Every positive head can freeze columns; only the MPNN head adds more."""
    shared = ("frozen_parameters", "shared_damping_parameters")
    assert CliffClassicalNN.ARCHITECTURE_CONFIG_KEYS == shared
    assert mtp_mtp._CliffPositiveParamNN.ARCHITECTURE_CONFIG_KEYS == shared


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
    assert "gradient_norms = self._clip_gradient_norms" in src
    # The decision is named rather than tested inline, because under DDP it has
    # to be all-reduced before it is acted on -- a rank that skipped alone would
    # `continue` past its peers' collectives and hang the job. See
    # `tests/test_cliff_induction_ddp.py::test_grad_norm_skip_is_collective`.
    assert "for norm in gradient_norms.values()" in src
    assert "not bool(torch.isfinite(norm))" in src
    assert "if skip_batch:" in src
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


# ---------------------------------------------------------------------------
# Induction audit instrumentation
#
# From analysis/s66x8_classical_profile/.../ARCHITECTURE_HANDOFF.md: CLIFF2
# induction is *positive* on 16 of 32 S66x8 geometries, which is physically
# impossible. The doc asks for these quantities to be exported before any
# induction redesign, so that nonconvergence, loss of positive definiteness,
# and an energy-contraction/sign bug can be told apart.


def _induction_kwargs(head, batch):
    outA = head(batch.batch_atomic_A)
    outB = head(batch.batch_atomic_B)
    pA, pB = outA[-1], outB[-1]
    return dict(
        ZA=batch.ZA, RA=batch.RA, qA=outA[0], muA=outA[1], quadA=outA[2],
        ZB=batch.ZB, RB=batch.RB, qB=outB[0], muB=outB[1], quadB=outB[2],
        e_AB_source=batch.e_ABfull_source,
        e_AB_target=batch.e_ABfull_target,
        e_AA_source=batch.e_AA_source, e_BB_source=batch.e_BB_source,
        e_AA_target=batch.e_AA_target, e_BB_target=batch.e_BB_target,
        hirshfeld_volume_ratio_A=torch.abs(outA[-2][:, 0]),
        hirshfeld_volume_ratio_B=torch.abs(outB[-2][:, 0]),
        valence_widths_A=outA[-2][:, 1],
        valence_widths_B=outB[-2][:, 1],
        thole_direct_A=pA[:, CLIFF_CLASSICAL_THOLE_DIRECT_INDEX],
        thole_direct_B=pB[:, CLIFF_CLASSICAL_THOLE_DIRECT_INDEX],
        thole_mutual_A=pA[:, CLIFF_CLASSICAL_THOLE_MUTUAL_INDEX],
        thole_mutual_B=pB[:, CLIFF_CLASSICAL_THOLE_MUTUAL_INDEX],
        ind_overlap_A=pA[:, CLIFF_CLASSICAL_IND_OVERLAP_INDEX],
        ind_overlap_B=pB[:, CLIFF_CLASSICAL_IND_OVERLAP_INDEX],
    )


def test_induction_diagnostics_are_off_by_default(
    nested_hfvr_vw_model, synthetic_dimer_batch
):
    """The training path must be unchanged: a bare tensor, same values."""
    import copy

    torch.manual_seed(0)
    head = CliffClassicalNN(
        atom_model=copy.deepcopy(nested_hfvr_vw_model), **HEAD_KWARGS
    )
    kwargs = _induction_kwargs(head, synthetic_dimer_batch)
    plain = mtp_mtp.rackers_thole_induction(**kwargs, include_overlap=True)
    assert isinstance(plain, torch.Tensor)
    energy, diagnostics = mtp_mtp.rackers_thole_induction(
        **kwargs, include_overlap=True, return_diagnostics=True
    )
    assert isinstance(diagnostics, dict)
    # Instrumenting must not perturb the number the model trains on.
    assert torch.equal(plain, energy)


def test_induction_diagnostics_report_the_audit_quantities(
    nested_hfvr_vw_model, synthetic_dimer_batch
):
    import copy

    torch.manual_seed(0)
    head = CliffClassicalNN(
        atom_model=copy.deepcopy(nested_hfvr_vw_model), **HEAD_KWARGS
    )
    _, diagnostics = mtp_mtp.rackers_thole_induction(
        **_induction_kwargs(head, synthetic_dimer_batch),
        include_overlap=True,
        return_diagnostics=True,
    )
    for key in (
        "scf_iterations",
        "scf_residual",
        "scf_converged",
        "all_finite",
        "max_induced_dipole",
        "max_abs_energy_edge",
        "energy_edge_contraction",
        "energy_qu",
        "energy_uu",
        "energy_variational_total",
        "overlap_contribution",
        "n_edges_positive",
        "n_edges",
    ):
        assert key in diagnostics, key
    # A converged solve must report a residual under the threshold, and an
    # unconverged one must be distinguishable -- returning the last iterate
    # silently is what the handoff doc calls out as breaking the contract.
    if diagnostics["scf_converged"]:
        assert diagnostics["scf_residual"] < 1e-8
    assert 1 <= diagnostics["scf_iterations"] <= 200
    assert diagnostics["all_finite"]
    assert diagnostics["max_induced_dipole"] >= 0.0
    assert diagnostics["max_abs_energy_edge"] >= 0.0
    assert diagnostics["n_edges"] == synthetic_dimer_batch.e_ABfull_source.numel()


def test_dimer_prop_accumulates_opt_in_induction_health(
    nested_hfvr_vw_model, synthetic_dimer_batch
):
    import copy

    torch.manual_seed(0)
    head = CliffClassicalNN(
        atom_model=copy.deepcopy(nested_hfvr_vw_model), **HEAD_KWARGS
    )
    dimer = mtp_mtp.DimerProp(
        ATParam=head,
        dimer_eval="cliff_classical_overlap",
        freeze_atom_model=True,
    )
    assert dimer.induction_diagnostic_totals()["calls"] == 0.0
    dimer(synthetic_dimer_batch)
    assert dimer.induction_diagnostic_totals()["calls"] == 0.0

    dimer.collect_induction_diagnostics = True
    energy, _, _ = dimer(synthetic_dimer_batch)
    totals = dimer.induction_diagnostic_totals()
    assert torch.isfinite(energy).all()
    assert totals["calls"] == 1.0
    assert totals["finite"] == 1.0
    assert 1.0 <= totals["iterations_sum"] <= 200.0
    assert totals["iterations_max"] == totals["iterations_sum"]
    assert totals["max_induced_dipole"] >= 0.0
    assert totals["max_abs_energy_edge"] >= 0.0
    assert totals["edges"] == synthetic_dimer_batch.e_ABfull_source.numel()

    dimer.reset_induction_diagnostics()
    assert dimer.induction_diagnostic_totals()["calls"] == 0.0


def test_induction_diagnostic_maxima_preserve_nonfinite_failures(
    nested_hfvr_vw_model,
):
    dimer = mtp_mtp.DimerProp(
        ATParam=_head(nested_hfvr_vw_model),
        dimer_eval="cliff_classical_overlap",
        freeze_atom_model=True,
    )
    dimer._record_induction_diagnostics(
        {
            "scf_converged": False,
            "all_finite": False,
            "scf_iterations": 200,
            "scf_residual": float("nan"),
            "max_induced_dipole": float("nan"),
            "max_abs_energy_edge": float("nan"),
            "n_edges_positive": 0,
            "n_edges": 1,
        }
    )
    totals = dimer.induction_diagnostic_totals()
    assert totals["finite"] == 0.0
    assert totals["converged"] == 0.0
    assert totals["residual_max"] == float("inf")
    assert totals["max_induced_dipole"] == float("inf")
    assert totals["max_abs_energy_edge"] == float("inf")


def test_induction_energy_is_not_the_variational_functional_of_its_solve(
    nested_hfvr_vw_model, synthetic_dimer_batch
):
    """The two energies now agree, which is the whole point of the fix.

    This test used to assert they *disagreed*, and told its own successor what
    to do: "if the response solve and energy expression were made consistent,
    delete this test and assert the sign invariant instead". That is what
    happened. The permanent field is now intermolecular, so the dipoles are the
    response to the partner and the intermolecular edge contraction *is* the
    variational functional `-1/2 mu . E_perm` of the solve that produced them.

    That matters more than the numbers: a variational energy of a converged
    linear response is non-positive by construction, so attractive induction is
    now a structural property rather than something to be checked geometry by
    geometry.
    """
    import copy

    torch.manual_seed(0)
    head = CliffClassicalNN(
        atom_model=copy.deepcopy(nested_hfvr_vw_model), **HEAD_KWARGS
    )
    _, diagnostics = mtp_mtp.rackers_thole_induction(
        **_induction_kwargs(head, synthetic_dimer_batch),
        include_overlap=False,
        return_diagnostics=True,
    )
    contraction = diagnostics["energy_edge_contraction"]
    variational = diagnostics["energy_variational_total"]
    assert contraction == pytest.approx(variational, rel=1e-6), (
        "the edge contraction is no longer the variational functional of its "
        "own solve -- the permanent field and the energy have gone back to "
        "covering different edge sets"
    )
    # Both, not just the variational one: they are the same quantity now.
    assert variational < 0.0
    assert contraction < 0.0

    src = inspect.getsource(mtp_mtp.rackers_thole_induction)
    assert "direct_tensors_AB[3]" in src and "direct_tensors_AB[4]" in src
    assert "e_AA_source" in src and "e_BB_source" in src


def test_the_legacy_path_still_shows_the_discrepancy_it_was_written_for(
    nested_hfvr_vw_model, synthetic_dimer_batch
):
    """The same diagnostic on the pre-fix construction, as the control.

    Without this, the agreement above could equally mean the diagnostic stopped
    measuring anything.
    """
    import copy

    torch.manual_seed(0)
    head = CliffClassicalNN(
        atom_model=copy.deepcopy(nested_hfvr_vw_model), **HEAD_KWARGS
    )
    _, diagnostics = mtp_mtp.rackers_thole_induction(
        **_induction_kwargs(head, synthetic_dimer_batch),
        include_overlap=False,
        return_diagnostics=True,
        intramolecular_permanent_field=True,
    )
    assert diagnostics["energy_edge_contraction"] != pytest.approx(
        diagnostics["energy_variational_total"], rel=1e-3
    )


# ---------------------------------------------------------------------------
# Variational interaction induction
#
# The legacy energy contracts E_qu/E_uu over AB edges only while the dipoles
# responded to the full AA+BB+AB field, so it is not the variational functional
# of its own solve and carries no sign guarantee. On a trained overlap
# checkpoint it returns +0.172 kcal/mol for a dimer -- repulsive induction,
# which cannot happen. Computing E_pol(dimer) - E_pol(monomers) with the same
# functional the solve minimizes gives -0.871 on the same inputs.


def _variational_kwargs(head, batch):
    kwargs = _induction_kwargs(head, batch)
    kwargs.update(
        variational_energy=True,
        molecule_ind_A=batch.molecule_ind_A,
        molecule_ind_B=batch.molecule_ind_B,
    )
    return kwargs


def test_variational_induction_is_attractive_where_legacy_is_not(
    nested_hfvr_vw_model, synthetic_dimer_batch
):
    import copy

    torch.manual_seed(0)
    head = CliffClassicalNN(
        atom_model=copy.deepcopy(nested_hfvr_vw_model), **HEAD_KWARGS
    )
    from apnet_pt.util import scatter_sum_compile

    dimer_ind = synthetic_dimer_batch.dimer_ind_full
    n_dimers = int(dimer_ind.max()) + 1
    legacy = mtp_mtp.rackers_thole_induction(
        **_induction_kwargs(head, synthetic_dimer_batch), include_overlap=False
    )
    variational = mtp_mtp.rackers_thole_induction(
        **_variational_kwargs(head, synthetic_dimer_batch),
        include_overlap=False,
    )
    per_dimer = scatter_sum_compile(variational, dimer_ind, dim_size=n_dimers)
    # The invariant from ARCHITECTURE_HANDOFF.md acceptance test 1/2.
    assert bool((per_dimer <= 1e-8).all()), per_dimer
    assert variational.shape == legacy.shape


def test_variational_induction_preserves_the_per_edge_contract(
    nested_hfvr_vw_model, synthetic_dimer_batch
):
    """Polarization is many-body, so per-edge values are a representation.

    What must be exact is the per-dimer sum, because that is what the harness
    scatter-sums and what the loss sees.
    """
    import copy

    from apnet_pt.util import scatter_sum_compile

    torch.manual_seed(0)
    head = CliffClassicalNN(
        atom_model=copy.deepcopy(nested_hfvr_vw_model), **HEAD_KWARGS
    )
    energy, diagnostics = mtp_mtp.rackers_thole_induction(
        **_variational_kwargs(head, synthetic_dimer_batch),
        include_overlap=False,
        return_diagnostics=True,
    )
    dimer_ind = synthetic_dimer_batch.dimer_ind_full
    n_dimers = int(dimer_ind.max()) + 1
    per_dimer = scatter_sum_compile(energy, dimer_ind, dim_size=n_dimers)
    # Every edge of a dimer carries the same share, so the scatter-sum is exact.
    for dimer in range(n_dimers):
        mask = dimer_ind == dimer
        assert torch.allclose(
            energy[mask], energy[mask][0].expand(int(mask.sum())), atol=1e-6
        )
    assert torch.isfinite(per_dimer).all()
    assert diagnostics["scf_converged"] in (True, False)


def test_variational_induction_requires_the_dimer_index(
    nested_hfvr_vw_model, synthetic_dimer_batch
):
    """Silently falling back to the legacy contraction would be worse."""
    import copy

    torch.manual_seed(0)
    head = CliffClassicalNN(
        atom_model=copy.deepcopy(nested_hfvr_vw_model), **HEAD_KWARGS
    )
    with pytest.raises(ValueError, match="molecule_ind"):
        mtp_mtp.rackers_thole_induction(
            **_induction_kwargs(head, synthetic_dimer_batch),
            include_overlap=False,
            variational_energy=True,
        )


def test_variational_induction_is_off_by_default_on_dimerprop(
    nested_hfvr_vw_model
):
    """Existing checkpoints must keep predicting what they predicted."""
    import copy

    torch.manual_seed(0)
    head = CliffClassicalNN(
        atom_model=copy.deepcopy(nested_hfvr_vw_model), **HEAD_KWARGS
    )
    dimer = mtp_mtp.DimerProp(
        ATParam=head, dimer_eval="cliff_classical_overlap",
        freeze_atom_model=True,
    )
    assert dimer.variational_induction is False
    opted_in = mtp_mtp.DimerProp(
        ATParam=head, dimer_eval="cliff_classical_overlap",
        freeze_atom_model=True, variational_induction=True,
    )
    assert opted_in.variational_induction is True


def test_variational_induction_changes_the_dimerprop_energy(
    nested_hfvr_vw_model, synthetic_dimer_batch
):
    import copy

    torch.manual_seed(0)
    head = CliffClassicalNN(
        atom_model=copy.deepcopy(nested_hfvr_vw_model), **HEAD_KWARGS
    )
    legacy = mtp_mtp.DimerProp(
        ATParam=head, dimer_eval="cliff_classical_overlap",
        freeze_atom_model=True,
    )(synthetic_dimer_batch)[0]
    variational = mtp_mtp.DimerProp(
        ATParam=head, dimer_eval="cliff_classical_overlap",
        freeze_atom_model=True, variational_induction=True,
    )(synthetic_dimer_batch)[0]
    # Electrostatics and exchange are untouched: the handoff doc says to keep
    # exchange as a control while induction is redesigned.
    assert torch.allclose(legacy[:, 0], variational[:, 0], atol=1e-6)
    assert torch.allclose(legacy[:, 1], variational[:, 1], atol=1e-6)
    assert not torch.allclose(legacy[:, 2], variational[:, 2], atol=1e-6)


def test_variational_induction_gradients_are_finite(
    nested_hfvr_vw_model, synthetic_dimer_batch
):
    import copy

    torch.manual_seed(0)
    head = CliffClassicalNN(
        atom_model=copy.deepcopy(nested_hfvr_vw_model), **HEAD_KWARGS
    )
    dimer = mtp_mtp.DimerProp(
        ATParam=head, dimer_eval="cliff_classical_overlap",
        freeze_atom_model=True, variational_induction=True,
    )
    dimer(synthetic_dimer_batch)[0].sum().backward()
    grads = [
        q.grad for q in head.parameters()
        if q.requires_grad and q.grad is not None
    ]
    assert grads
    assert all(torch.isfinite(g).all() for g in grads)
    assert any(bool(g.abs().sum() > 0) for g in grads)


# ---------------------------------------------------------------------------
# Frozen induction damping
#
# Fitting the Thole parameters per atom makes the response operator a learned
# object, and the interaction induction E_pol(dimer) - E_pol(monomers) is only
# guaranteed attractive when that operator is positive definite. Making the
# energy variational was not enough on its own: on S66x8 it left 24/32
# geometries positive, because a difference of two negative energies has no
# sign guarantee without that condition. Holding the damping at CLIFF's fitted
# values leaves induction one learnable term, `-S_ij K_i K_j`, which is
# attractive by construction.


def test_induction_damping_columns_are_named_once():
    assert mtp_mtp.CLIFF_INDUCTION_DAMPING_PARAMETERS == (
        "thole_direct", "thole_mutual"
    )
    for name in mtp_mtp.CLIFF_INDUCTION_DAMPING_PARAMETERS:
        assert name in CLIFF_CLASSICAL_PARAMETER_NAMES


def test_frozen_columns_hold_their_seed_exactly(
    atomic_batch, nested_hfvr_vw_model
):
    torch.manual_seed(0)
    head = _head(
        nested_hfvr_vw_model,
        param_start_std=[0.0] * 5,
        frozen_parameters=mtp_mtp.CLIFF_INDUCTION_DAMPING_PARAMETERS,
    )
    parameters = head(atomic_batch)[-1].detach()
    for name in mtp_mtp.CLIFF_INDUCTION_DAMPING_PARAMETERS:
        column = CLIFF_CLASSICAL_PARAMETER_NAMES.index(name)
        values = parameters[:, column]
        # Identical across atoms and equal to the seed.
        assert float(values.max() - values.min()) < 1e-6, name
        assert float(values[0]) == pytest.approx(
            CLIFF_CLASSICAL_INITIAL_VALUES[column], rel=1e-4
        ), name
    # The unfrozen columns still vary per atom.
    exch = parameters[:, CLIFF_CLASSICAL_EXCH_INDEX]
    assert float(exch.max() - exch.min()) > 1.0


def test_frozen_columns_receive_no_gradient(
    atomic_batch, nested_hfvr_vw_model
):
    """Detached, not merely unused: an optimizer must not touch them."""
    torch.manual_seed(0)
    head = _head(
        nested_hfvr_vw_model,
        frozen_parameters=mtp_mtp.CLIFF_INDUCTION_DAMPING_PARAMETERS,
    )
    head(atomic_batch)[-1].sum().backward()
    for name in CLIFF_CLASSICAL_PARAMETER_NAMES:
        column = CLIFF_CLASSICAL_PARAMETER_NAMES.index(name)
        frozen = name in mtp_mtp.CLIFF_INDUCTION_DAMPING_PARAMETERS
        embedding = head.guess_layer[column]
        assert embedding.weight.requires_grad is (not frozen), name
        touched = (
            embedding.weight.grad is not None
            and bool(embedding.weight.grad.abs().sum() > 0)
        )
        assert touched is (not frozen), name
        for readout in head.param_readout_layers[column]:
            for parameter in readout.parameters():
                assert parameter.requires_grad is (not frozen), name


def test_freezing_reduces_the_trainable_parameter_count(
    nested_hfvr_vw_model
):
    import copy

    torch.manual_seed(0)
    learned = _head(copy.deepcopy(nested_hfvr_vw_model))
    torch.manual_seed(0)
    frozen = _head(
        copy.deepcopy(nested_hfvr_vw_model),
        frozen_parameters=mtp_mtp.CLIFF_INDUCTION_DAMPING_PARAMETERS,
    )
    count = lambda m: sum(  # noqa: E731
        q.numel() for q in m.parameters() if q.requires_grad
    )
    assert count(frozen) < count(learned)


def test_the_dense_head_can_freeze_too(atomic_batch, nested_hfvr_vw_model):
    """Needed as the control for the message-passing experiment."""
    torch.manual_seed(0)
    head = CliffClassicalNN(
        atom_model=nested_hfvr_vw_model,
        param_start_std=[0.0] * 5,
        frozen_parameters=mtp_mtp.CLIFF_INDUCTION_DAMPING_PARAMETERS,
        **HEAD_KWARGS,
    )
    parameters = head(atomic_batch)[-1].detach()
    column = CLIFF_CLASSICAL_THOLE_DIRECT_INDEX
    assert float(parameters[:, column].max() - parameters[:, column].min()) < 1e-6
    assert float(parameters[0, column]) == pytest.approx(
        CLIFF_CLASSICAL_INITIAL_VALUES[column], rel=1e-4
    )


@pytest.mark.parametrize(
    "bad,error", [("thole_direct", TypeError), (["nope"], ValueError)]
)
def test_frozen_parameters_are_validated(nested_hfvr_vw_model, bad, error):
    with pytest.raises(error):
        _head(nested_hfvr_vw_model, frozen_parameters=bad)


def test_frozen_parameters_round_trip_through_a_checkpoint(
    tmp_path, nested_hfvr_vw_model, atomic_batch
):
    """A checkpoint that forgot what it froze would silently unfreeze on load."""
    harness = mtp_mtp.CliffClassicalOverlapMPNNModel(
        atom_model=nested_hfvr_vw_model,
        ds_root=None,
        use_GPU=False,
        ignore_database_null=True,
        n_message=1,
        n_neuron=8,
        n_embed=4,
        frozen_parameters=mtp_mtp.CLIFF_INDUCTION_DAMPING_PARAMETERS,
        **MPNN_KWARGS,
    )
    assert harness.model.frozen_parameters == (
        mtp_mtp.CLIFF_INDUCTION_DAMPING_PARAMETERS
    )
    before = harness.model(atomic_batch)[-1].detach().clone()
    path = tmp_path / "frozen.pt"
    model_io.save_checkpoint(harness._create_checkpoint(), str(path))
    reloaded = mtp_mtp.CliffClassicalOverlapMPNNModel(
        atom_model=None,
        pre_trained_model_path=str(path),
        ds_root=None,
        use_GPU=False,
        ignore_database_null=True,
    )
    assert tuple(reloaded.model.frozen_parameters) == (
        mtp_mtp.CLIFF_INDUCTION_DAMPING_PARAMETERS
    )
    assert reloaded.model._frozen_parameter_indices == (1, 2)
    assert torch.allclose(
        before, reloaded.model(atomic_batch)[-1].detach(), atol=1e-6
    )


def test_shared_damping_reaches_the_dense_route_and_survives_a_reload(
    tmp_path, nested_hfvr_vw_model, atomic_batch
):
    """The dense harness is the route the shared-damping arm actually runs on.

    `CliffClassicalOverlapModel.__init__` collects unrecognised keywords into
    `**dataset_kwargs` and hands them to `AM_DimerParam_Model`, so whether a
    head keyword reaches the head is a property of that forwarding chain, not
    of the head's own signature. A keyword that fell out of the chain would be
    accepted in silence and the run would fit per-element damping while its
    W&B config claimed otherwise.
    """
    harness = mtp_mtp.CliffClassicalOverlapModel(
        atom_model=nested_hfvr_vw_model,
        ds_root=None,
        use_GPU=False,
        ignore_database_null=True,
        n_message=1,
        n_neuron=8,
        n_embed=4,
        shared_damping_parameters=mtp_mtp.CLIFF_INDUCTION_DAMPING_PARAMETERS,
    )
    assert tuple(harness.model.shared_damping_parameters) == (
        mtp_mtp.CLIFF_INDUCTION_DAMPING_PARAMETERS
    )
    params = harness.model(atomic_batch)[-1]
    direct = params[:, mtp_mtp.CLIFF_CLASSICAL_THOLE_DIRECT_INDEX]
    mutual = params[:, mtp_mtp.CLIFF_CLASSICAL_THOLE_MUTUAL_INDEX]
    # One scalar: constant across atoms, and the same one in both columns.
    assert torch.allclose(direct, direct[0].expand_as(direct), atol=1e-6)
    assert torch.allclose(mutual, direct, atol=1e-6)
    # Still learnable -- a shared parameter is not a frozen one.
    direct.sum().backward()
    assert harness.model.shared_damping_raw.grad is not None
    assert torch.any(harness.model.shared_damping_raw.grad != 0)

    path = tmp_path / "shared.pt"
    model_io.save_checkpoint(harness._create_checkpoint(), str(path))
    reloaded = mtp_mtp.CliffClassicalOverlapModel(
        atom_model=None,
        pre_trained_model_path=str(path),
        ds_root=None,
        use_GPU=False,
        ignore_database_null=True,
    )
    assert tuple(reloaded.model.shared_damping_parameters) == (
        mtp_mtp.CLIFF_INDUCTION_DAMPING_PARAMETERS
    )
    assert torch.allclose(
        params.detach(), reloaded.model(atomic_batch)[-1].detach(), atol=1e-6
    )


def test_nothing_is_frozen_by_default(nested_hfvr_vw_model):
    torch.manual_seed(0)
    head = _head(nested_hfvr_vw_model)
    assert head.frozen_parameters == ()
    assert head._frozen_parameter_indices == ()


def test_dense_component_gradient_groups_clip_independently(
    nested_hfvr_vw_model,
):
    """Each physical term gets the full clip budget, not a shared scale."""
    torch.manual_seed(0)
    harness = mtp_mtp.CliffClassicalOverlapModel(
        atom_model=nested_hfvr_vw_model,
        ds_root=None,
        use_GPU=False,
        ignore_database_null=True,
        n_message=1,
        n_neuron=8,
        n_embed=4,
    )
    groups = harness._component_gradient_parameter_groups()
    assert tuple(groups) == ("electrostatics", "exchange", "induction")

    before = {}
    for scale, (component, parameters) in enumerate(groups.items(), start=2):
        assert parameters
        for parameter in parameters:
            parameter.grad = torch.full_like(parameter, float(scale))
        before[component] = torch.sqrt(
            sum(torch.sum(parameter.grad.square()) for parameter in parameters)
        )
        assert before[component] > 1.0

    reported = harness._clip_gradient_norms(1.0, "component")
    assert set(reported) == set(groups)
    for component, parameters in groups.items():
        assert reported[component] == pytest.approx(before[component])
        after = torch.sqrt(
            sum(torch.sum(parameter.grad.square()) for parameter in parameters)
        )
        assert after == pytest.approx(1.0, rel=2e-5)


def test_thole_optimizer_lr_partitions_independent_dense_head(
    nested_hfvr_vw_model,
):
    harness = mtp_mtp.CliffClassicalOverlapModel(
        atom_model=nested_hfvr_vw_model,
        ds_root=None,
        use_GPU=False,
        ignore_database_null=True,
        n_message=1,
        n_neuron=8,
        n_embed=4,
    )
    groups = harness._optimizer_parameter_groups(5e-4, 2.5e-5)
    assert [group["group_name"] for group in groups] == ["base", "thole"]
    assert [group["lr"] for group in groups] == [5e-4, 2.5e-5]
    thole_ids = {id(parameter) for parameter in groups[1]["params"]}
    expected = {
        id(parameter)
        for column in (
            mtp_mtp.CLIFF_CLASSICAL_THOLE_DIRECT_INDEX,
            mtp_mtp.CLIFF_CLASSICAL_THOLE_MUTUAL_INDEX,
        )
        for module in (
            harness.model.guess_layer[column],
            harness.model.param_readout_layers[column],
        )
        for parameter in module.parameters()
        if parameter.requires_grad
    }
    assert thole_ids == expected
    all_grouped = [
        parameter for group in groups for parameter in group["params"]
    ]
    assert len(all_grouped) == len({id(parameter) for parameter in all_grouped})
    assert {id(parameter) for parameter in all_grouped} == {
        id(parameter)
        for parameter in harness.model.parameters()
        if parameter.requires_grad
    }


def test_low_thole_trainstate_identity_guards_both_learning_rates(
    tmp_path, nested_hfvr_vw_model
):
    harness = mtp_mtp.CliffClassicalOverlapModel(
        atom_model=nested_hfvr_vw_model,
        ds_root=None,
        use_GPU=False,
        ignore_database_null=True,
        n_message=1,
        n_neuron=8,
        n_embed=4,
    )
    optimizer = torch.optim.Adam(
        harness._optimizer_parameter_groups(5e-4, 2.5e-5), lr=5e-4
    )
    path = tmp_path / "low-thole.trainstate.pt"
    identity = {"base_lr": 5e-4, "thole_lr": 2.5e-5}
    model_io.save_train_state(
        str(path),
        model=harness.model,
        optimizer=optimizer,
        epochs_completed=3,
        lowest_test_loss=1.25,
        identity=identity,
    )

    restored = torch.optim.Adam(
        harness._optimizer_parameter_groups(5e-4, 2.5e-5), lr=5e-4
    )
    assert model_io.load_train_state(
        str(path),
        model=harness.model,
        optimizer=restored,
        identity=identity,
    ) == (3, 1.25)
    assert [group["lr"] for group in restored.param_groups] == [5e-4, 2.5e-5]

    mismatched = torch.optim.Adam(
        harness._optimizer_parameter_groups(1e-4, 2.5e-5), lr=1e-4
    )
    with pytest.warns(UserWarning, match="identity mismatch"):
        assert (
            model_io.load_train_state(
                str(path),
                model=harness.model,
                optimizer=mismatched,
                identity={"base_lr": 1e-4, "thole_lr": 2.5e-5},
            )
            is None
        )


def test_thole_optimizer_lr_supports_one_shared_damping_scalar(
    nested_hfvr_vw_model,
):
    harness = mtp_mtp.CliffClassicalOverlapModel(
        atom_model=nested_hfvr_vw_model,
        ds_root=None,
        use_GPU=False,
        ignore_database_null=True,
        n_message=1,
        n_neuron=8,
        n_embed=4,
        shared_damping_parameters=mtp_mtp.CLIFF_INDUCTION_DAMPING_PARAMETERS,
    )
    groups = harness._optimizer_parameter_groups(5e-4, 2.5e-5)
    assert groups[1]["params"] == [harness.model.shared_damping_raw]


def test_thole_optimizer_lr_rejects_frozen_damping(
    nested_hfvr_vw_model,
):
    harness = mtp_mtp.CliffClassicalOverlapModel(
        atom_model=nested_hfvr_vw_model,
        ds_root=None,
        use_GPU=False,
        ignore_database_null=True,
        n_message=1,
        n_neuron=8,
        n_embed=4,
        frozen_parameters=mtp_mtp.CLIFF_INDUCTION_DAMPING_PARAMETERS,
    )
    with pytest.raises(ValueError, match="no trainable direct or mutual"):
        harness._optimizer_parameter_groups(5e-4, 2.5e-5)


def test_zero_call_ddp_diagnostics_still_enter_collectives(
    nested_hfvr_vw_model, monkeypatch
):
    harness = mtp_mtp.CliffClassicalOverlapModel(
        atom_model=nested_hfvr_vw_model,
        ds_root=None,
        use_GPU=False,
        ignore_database_null=True,
        n_message=1,
        n_neuron=8,
        n_embed=4,
    )
    operations = []

    def fake_all_reduce(tensor, op="sum"):
        operations.append(op)
        return tensor

    monkeypatch.setattr(harness, "_ddp_all_reduce", fake_all_reduce)
    assert harness._reduced_induction_diagnostics("train", world_size=2) == {}
    assert operations == ["sum", "max"]


def test_component_gradient_groups_reject_the_shared_mpnn_head(
    nested_hfvr_vw_model,
):
    harness = CliffClassicalOverlapMPNNModel(
        atom_model=nested_hfvr_vw_model,
        ds_root=None,
        use_GPU=False,
        ignore_database_null=True,
        n_message=1,
        n_neuron=8,
        n_embed=4,
        **MPNN_KWARGS,
    )
    with pytest.raises(ValueError, match="dense CliffClassicalNN"):
        harness._component_gradient_parameter_groups()


class _FakeCliffHarness:
    """Records constructor kwargs without building anything heavy."""

    calls: list = []

    def __init__(self, **kwargs):
        type(self).calls.append(self)
        self.kwargs = kwargs
        self.dataset = object()
        self.model = None

    def train(
        self,
        grad_clip_norm=None,
        grad_clip_mode="global",
        thole_lr=None,
        induction_diagnostics=False,
        **kwargs,
    ):
        self.train_kwargs = {
            "grad_clip_norm": grad_clip_norm,
            "grad_clip_mode": grad_clip_mode,
            "thole_lr": thole_lr,
            "induction_diagnostics": induction_diagnostics,
            **kwargs,
        }


class _FakeAtomTypeWrapper:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model = None


@pytest.fixture
def cliff_dispatch(monkeypatch):
    _FakeCliffHarness.calls = []
    monkeypatch.setattr(
        mtp_mtp, "AtomTypeParamModel", _FakeAtomTypeWrapper
    )
    monkeypatch.setattr(mtp_mtp, "CliffClassicalModel", _FakeCliffHarness)
    return _FakeCliffHarness


def test_frozen_parameters_dispatch_on_both_cliff_heads(tmp_path, cliff_dispatch):
    """Allowed on every CLIFF route: the dense one is the control."""
    train_models.train_pairwise_model(
        apnet_model_type="CliffClassicalModel",
        model_out=str(tmp_path / "out.pt"),
        frozen_parameters=["thole_direct", "thole_mutual"],
        ds_max_size=100,
    )
    harness = cliff_dispatch.calls[0]
    assert harness.kwargs["frozen_parameters"] == (
        "thole_direct",
        "thole_mutual",
    )


def test_frozen_parameters_rejected_off_the_cliff_routes(tmp_path):
    with pytest.raises(ValueError, match="frozen_parameters"):
        train_models.train_pairwise_model(
            apnet_model_type="APNet2",
            model_out=str(tmp_path / "out.pt"),
            frozen_parameters=["thole_direct"],
        )


def test_component_clip_mode_reaches_dense_training(tmp_path, cliff_dispatch):
    train_models.train_pairwise_model(
        apnet_model_type="CliffClassicalModel",
        model_out=str(tmp_path / "out.pt"),
        grad_clip_norm=1.0,
        grad_clip_mode="component",
        ds_max_size=100,
    )
    assert cliff_dispatch.calls[0].train_kwargs["grad_clip_norm"] == 1.0
    assert cliff_dispatch.calls[0].train_kwargs["grad_clip_mode"] == "component"


def test_stability_controls_reach_dense_training(tmp_path, cliff_dispatch):
    train_models.train_pairwise_model(
        apnet_model_type="CliffClassicalModel",
        model_out=str(tmp_path / "out.pt"),
        thole_lr=2.5e-5,
        induction_diagnostics=True,
        ds_max_size=100,
    )
    training = cliff_dispatch.calls[0].train_kwargs
    assert training["thole_lr"] == 2.5e-5
    assert training["induction_diagnostics"] is True


def test_stability_controls_reject_non_cliff_route(tmp_path):
    with pytest.raises(ValueError, match="thole_lr"):
        train_models.train_pairwise_model(
            apnet_model_type="APNet2",
            model_out=str(tmp_path / "out.pt"),
            thole_lr=2.5e-5,
        )
    with pytest.raises(ValueError, match="induction_diagnostics"):
        train_models.train_pairwise_model(
            apnet_model_type="APNet2",
            model_out=str(tmp_path / "out.pt"),
            induction_diagnostics=True,
        )


def test_component_clip_mode_requires_a_norm(tmp_path):
    with pytest.raises(ValueError, match="requires grad_clip_norm"):
        train_models.train_pairwise_model(
            apnet_model_type="CliffClassicalOverlapModel",
            model_out=str(tmp_path / "out.pt"),
            grad_clip_mode="component",
        )


def test_component_clip_mode_rejects_shared_mpnn_head(tmp_path):
    with pytest.raises(ValueError, match="dense combined CLIFF"):
        train_models.train_pairwise_model(
            apnet_model_type="CliffClassicalOverlapMPNNModel",
            model_out=str(tmp_path / "out.pt"),
            grad_clip_norm=1.0,
            grad_clip_mode="component",
        )


def test_mpnn_architecture_flags_still_rejected_on_dense_routes(tmp_path):
    """Freezing is general; message-passing geometry is not."""
    with pytest.raises(ValueError, match="param_hidden"):
        train_models.train_pairwise_model(
            apnet_model_type="CliffClassicalOverlapModel",
            model_out=str(tmp_path / "out.pt"),
            param_hidden=32,
        )


def test_shared_damping_dispatches_onto_the_dense_cliff_head(
    tmp_path, cliff_dispatch
):
    """The shared-damping arm is a dense-route experiment, not an MPNN one.

    `_CliffPositiveParamNN` has accepted `shared_damping_parameters` since the
    seeds were aligned with CLIFF, but no training flag ever reached it, so the
    arm could only be run from Python. A Phoenix chunk configures its run
    entirely through `train_models.py`, so without this the arm is unrunnable
    there.
    """
    train_models.train_pairwise_model(
        apnet_model_type="CliffClassicalModel",
        model_out=str(tmp_path / "out.pt"),
        shared_damping_parameters=["thole_direct", "thole_mutual"],
        ds_max_size=100,
    )
    harness = cliff_dispatch.calls[0]
    assert harness.kwargs["shared_damping_parameters"] == (
        "thole_direct",
        "thole_mutual",
    )


def test_shared_damping_rejected_off_the_cliff_routes(tmp_path):
    with pytest.raises(ValueError, match="shared_damping_parameters"):
        train_models.train_pairwise_model(
            apnet_model_type="APNet2",
            model_out=str(tmp_path / "out.pt"),
            shared_damping_parameters=["thole_direct"],
        )


def test_shared_damping_is_not_an_mpnn_only_architecture_flag(
    tmp_path, cliff_dispatch
):
    """It must not be swept up by the `--param_*` rejection.

    The architecture guard rejects anything left in `parameter_head_kwargs` on
    a dense route apart from an explicit allow-list. Adding a new head kwarg
    without extending that list makes the dense route -- the only route this
    arm runs on -- reject its own flag.
    """
    train_models.train_pairwise_model(
        apnet_model_type="CliffClassicalModel",
        model_out=str(tmp_path / "out.pt"),
        shared_damping_parameters=["thole_direct", "thole_mutual"],
        ds_max_size=100,
    )
    assert cliff_dispatch.calls, "dense route rejected the shared-damping flag"


def test_help_advertises_shared_damping_parameters():
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT / "src"), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    result = subprocess.run(
        [sys.executable, "train_models.py", "--help"],
        cwd=REPO_ROOT, capture_output=True, text=True, env=env,
    )
    assert result.returncode == 0, result.stderr
    assert "--shared_damping_parameters" in result.stdout


def test_help_advertises_frozen_parameters():
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT / "src"), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    result = subprocess.run(
        [sys.executable, "train_models.py", "--help"],
        cwd=REPO_ROOT, capture_output=True, text=True, env=env,
    )
    assert result.returncode == 0, result.stderr
    assert "--frozen_parameters" in result.stdout


# ---------------------------------------------------------------------------
# CLIFF parity: seeds, damping form, energy prefactor
#
# Exchange reached parity with CLIFF (MAE 0.698 vs 0.695 on S66x8, each model
# against its own fitting reference) while electrostatics sat at 1.41x and
# induction at 3.05x. Exchange is also the only component with hand-computed
# golden values, unit-conversion tests, limit tests and per-element Table I
# seeding. These tests close that gap for the induction side.


def test_elst_seed_matches_cliff_table_i_in_its_own_units():
    """1.8 bohr^-1 is CLIFF's ~3.45 Ang^-1; the seeds agree, the units differ.

    Pinned because reading 1.8 against Table I's 3.04-4.32 invites "fixing" it
    upward by a factor of the Bohr radius, which would double the damping.
    """
    bohr_to_angstrom = 0.529177210903
    in_angstrom_inverse = (
        mtp_mtp.CLIFF_ELST_SEED_BOHR_INVERSE / bohr_to_angstrom
    )
    assert 3.0371 <= in_angstrom_inverse <= 4.3157
    assert in_angstrom_inverse == pytest.approx(3.40, abs=0.02)
    assert (
        CLIFF_CLASSICAL_INITIAL_VALUES[CLIFF_CLASSICAL_ELST_INDEX]
        == mtp_mtp.CLIFF_ELST_SEED_BOHR_INVERSE
    )


def test_induction_seeds_start_inside_cliffs_published_range():
    """K^indu spans 2.1e-05 to 1.7546 in CLIFF Table I.

    The old 1.8 seed sat above the entire range and entered the energy as the
    product K_i K_j = 3.24 against CLIFF's typical ~0.64, over-polarizing from
    step one; every run before this pinned the column at its floor.
    """
    seed = CLIFF_CLASSICAL_INITIAL_VALUES[CLIFF_CLASSICAL_IND_OVERLAP_INDEX]
    assert seed == mtp_mtp.CLIFF_IND_OVERLAP_SEED == 0.2
    assert 2.1e-05 <= seed <= 1.7546
    assert seed * seed < 0.64  # below CLIFF's typical pair product


def test_both_thole_columns_seed_at_cliffs_single_smearing_coefficient():
    """CLIFF refits one global coefficient, 0.38539, for direct and mutual."""
    assert mtp_mtp.CLIFF_THOLE_SMEARING == 0.38539
    direct = CLIFF_CLASSICAL_INITIAL_VALUES[CLIFF_CLASSICAL_THOLE_DIRECT_INDEX]
    mutual = CLIFF_CLASSICAL_INITIAL_VALUES[CLIFF_CLASSICAL_THOLE_MUTUAL_INDEX]
    assert direct == mutual == mtp_mtp.CLIFF_THOLE_SMEARING
    # The Rackers contract keeps its historical seeds: its checkpoints were
    # trained with them.
    assert mtp_mtp.RACKERS_INITIAL_VALUES == (1.8, 0.34, 0.39, 1.8)


def test_direct_thole_exponent_is_amoeba_plus_and_cliff_is_selectable():
    """AMOEBA+ damps the permanent field with u**1.5, CLIFF with u**3.

    AMOEBA+ (JCTC 2019) reports better three-body distance dependence for
    `1 - exp(-a u**1.5)` on the *permanent* field and leaves mutual at u**3.
    That is the default here. It is a real divergence from CLIFF Eq. (22),
    which uses u**3 for both, so it must be selectable rather than implicit.
    """
    from apnet_pt.multipole import (
        thole_damping_direct_torch,
        thole_damping_mutual_torch,
    )

    assert mtp_mtp.THOLE_DIRECT_EXPONENT_AMOEBA_PLUS == 1.5
    assert mtp_mtp.THOLE_DIRECT_EXPONENT_CLIFF == 3.0
    r = torch.tensor([3.0], dtype=torch.float64)
    alpha = torch.tensor([1.5], dtype=torch.float64)
    a = torch.tensor([mtp_mtp.CLIFF_THOLE_SMEARING], dtype=torch.float64)

    u = r / (alpha * alpha) ** (1.0 / 6.0)
    for exponent in (1.5, 3.0):
        au, lam3, _ = thole_damping_direct_torch(
            r, alpha, alpha, a, exponent=exponent
        )
        expected = a * u ** exponent
        assert torch.allclose(au, expected), exponent
        assert torch.allclose(lam3, 1 - torch.exp(-expected)), exponent
    # Default is AMOEBA+, and mutual is unconditionally u**3.
    default_au, _, _ = thole_damping_direct_torch(r, alpha, alpha, a)
    assert torch.allclose(default_au, a * u ** 1.5)
    mutual_au, _, _ = thole_damping_mutual_torch(r, alpha, alpha, a)
    assert torch.allclose(mutual_au, a * u ** 3)


def test_energy_half_factor_defaults_to_the_historical_prefactor(
    nested_hfvr_vw_model, synthetic_dimer_batch
):
    """CLIFF Eq. (19) carries no 1/2; this module always has.

    Keeping the half by default means no trained checkpoint changes meaning,
    and `energy_half_factor=False` is the Eq. (19) form.
    """
    import copy
    import inspect as _inspect

    signature = _inspect.signature(mtp_mtp.rackers_thole_induction)
    assert signature.parameters["energy_half_factor"].default is True

    torch.manual_seed(0)
    head = CliffClassicalNN(
        atom_model=copy.deepcopy(nested_hfvr_vw_model), **HEAD_KWARGS
    )
    kwargs = _induction_kwargs(head, synthetic_dimer_batch)
    halved = mtp_mtp.rackers_thole_induction(**kwargs, include_overlap=False)
    whole = mtp_mtp.rackers_thole_induction(
        **kwargs, include_overlap=False, energy_half_factor=False
    )
    # Without the overlap term the two differ by exactly a factor of two.
    assert torch.allclose(whole, 2.0 * halved, atol=1e-6)


def test_shared_damping_is_one_scalar_for_both_columns_and_all_atoms(
    atomic_batch, nested_hfvr_vw_model
):
    """CLIFF has one global smearing coefficient, not a per-atom field.

    Per-atom damping is badly conditioned: the physics tolerates a narrow band
    and degrades sharply outside it, which is how every previous run ended with
    a Thole column on a bound.
    """
    torch.manual_seed(0)
    head = _head(
        nested_hfvr_vw_model,
        shared_damping_parameters=mtp_mtp.CLIFF_INDUCTION_DAMPING_PARAMETERS,
    )
    parameters = head(atomic_batch)[-1].detach()
    direct = parameters[:, CLIFF_CLASSICAL_THOLE_DIRECT_INDEX]
    mutual = parameters[:, CLIFF_CLASSICAL_THOLE_MUTUAL_INDEX]
    assert float(direct.max() - direct.min()) < 1e-7
    assert torch.allclose(direct, mutual, atol=1e-9)
    assert float(direct[0]) == pytest.approx(
        mtp_mtp.CLIFF_THOLE_SMEARING, rel=1e-4
    )
    # Still learnable -- one degree of freedom, not zero.
    head(atomic_batch)[-1].sum().backward()
    assert head.shared_damping_raw.requires_grad
    assert head.shared_damping_raw.grad is not None
    assert float(head.shared_damping_raw.grad.abs()) > 0
    # And the per-column machinery it replaces is detached.
    for index in (
        CLIFF_CLASSICAL_THOLE_DIRECT_INDEX,
        CLIFF_CLASSICAL_THOLE_MUTUAL_INDEX,
    ):
        assert not head.guess_layer[index].weight.requires_grad


def test_shared_and_frozen_are_mutually_exclusive(nested_hfvr_vw_model):
    with pytest.raises(ValueError, match="frozen and shared"):
        _head(
            nested_hfvr_vw_model,
            frozen_parameters=("thole_direct",),
            shared_damping_parameters=("thole_direct",),
        )


def test_shared_damping_requires_one_seed(nested_hfvr_vw_model):
    """Two columns cannot share a scalar if they disagree about its value."""
    with pytest.raises(ValueError, match="share one seed"):
        _head(
            nested_hfvr_vw_model,
            param_start_mean=[1.8, 0.30, 0.45, 0.2, 2.5],
            shared_damping_parameters=("thole_direct", "thole_mutual"),
        )


def test_induction_overlap_term_carries_a_scale_not_a_unit_conversion():
    """`K_i S_ij K_j` is dimensionless, so `h2kcalmol` there is a scale factor.

    Exchange has three unit-conversion tests and an explicit assertion that the
    overlap helper applies none; induction had neither, and the factor of 627.5
    on a dimensionless product is entirely absorbed into a learned parameter.
    Pinned so the arbitrariness stays visible rather than looking like physics.
    """
    source = inspect.getsource(mtp_mtp.rackers_thole_induction)
    assert "E_ind -= K_A * S_ij * K_B * constants.h2kcalmol" in source

    # The property that makes the factor a scale and not a conversion: S_ij is
    # a dimensionless overlap in (0, 1], so `K_i S_ij K_j` carries no energy
    # unit for `h2kcalmol` to convert. Asserted numerically rather than by
    # grepping the helper, whose comments mention the constant precisely to say
    # it is not applied.
    widths = torch.tensor([0.4, 0.6, 0.5], dtype=torch.float64)
    source_index = torch.tensor([0, 1, 2])
    target_index = torch.tensor([1, 2, 0])
    for distance in (0.5, 2.0, 6.0):
        overlap = mtp_mtp.atomic_overlap_S_ij(
            widths,
            widths,
            source_index,
            target_index,
            torch.full((3,), distance, dtype=torch.float64),
            width_floor=0.0,
        )
        assert torch.all(overlap > 0) and torch.all(overlap <= 1.0), distance
    # Monotonically decreasing in r, i.e. an overlap and not an energy.
    close = mtp_mtp.atomic_overlap_S_ij(
        widths, widths, source_index, target_index,
        torch.full((3,), 1.0, dtype=torch.float64), width_floor=0.0,
    )
    far = mtp_mtp.atomic_overlap_S_ij(
        widths, widths, source_index, target_index,
        torch.full((3,), 5.0, dtype=torch.float64), width_floor=0.0,
    )
    assert torch.all(far < close)


def test_quadrupole_constant_defaults_to_physics_not_cliffs_value():
    """CLIFF's released code carries a Q_const that over-weights quadrupoles.

    This module defaults to 3.0, the physically correct value, and documents
    1.0 as the CLIFF-matching setting. A run that wants bit-parity with CLIFF
    opts in; nobody gets CLIFF's error by default.
    """
    source = inspect.getsource(mtp_mtp)
    assert "Q_const=3.0" in source
    assert "set to 1.0 to agree with CLIFF" in source
