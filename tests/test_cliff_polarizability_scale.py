"""Tests for the optional trainable per-element polarizability scale.

Long-range induction scales as ``alpha = alpha_0(Z) * HFVR**(4/3)``. ``HFVR``
comes from the frozen ``atom_model`` and ``alpha_0`` was a static table cloned
onto ``DimerProp`` as a plain tensor, so before this existed *no trainable
parameter scaled long-range induction at all* -- the Thole smearing and the
overlap correction both act only at short range. The scale closes that gap by
replacing ``alpha_0(Z)`` with ``alpha_0(Z) * exp(s_Z)``.

The whole point of the design is that it changes nothing until it is asked to,
so most of what is tested here is absence: the parameter does not exist, does
not appear in a ``state_dict``, does not split the optimizer, and does not move
a single energy, unless the run turns it on. ``s`` is seeded at exactly zero,
which is what makes warm starting an existing checkpoint into it safe and what
makes an ``lr = 0`` arm a true control rather than an approximation of one.
"""
import math
import subprocess
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apnet_pt import constants, model_io  # noqa: E402
from apnet_pt.AtomPairwiseModels import mtp_mtp  # noqa: E402
from apnet_pt.AtomPairwiseModels.mtp_mtp import (  # noqa: E402
    CLIFF_CLASSICAL_THOLE_DIRECT_INDEX,
    CLIFF_CLASSICAL_THOLE_MUTUAL_INDEX,
    CliffClassicalNN,
)

HEAD_KWARGS = dict(n_message=1, n_neuron=8, n_embed=4)


def _harness(nested, trainable_polarizability_scale=False, **overrides):
    """Build the dense-head harness, optionally with the scale turned on.

    Enabled after construction rather than through a wrapper argument because
    that is how production reaches it: ``train()`` turns it on so a checkpoint
    written before the scale existed can be warm started into an arm that
    trains it. The wrapper constructor deliberately gains no new knob.
    """
    harness = mtp_mtp.CliffClassicalOverlapModel(
        atom_model=nested,
        ds_root=None,
        use_GPU=False,
        ignore_database_null=True,
        **{**HEAD_KWARGS, **overrides},
    )
    if trainable_polarizability_scale:
        harness.model.enable_trainable_polarizability_scale()
    return harness


# --------------------------------------------------------------- default off


def test_the_scale_is_absent_unless_asked_for(nested_hfvr_vw_model):
    """A head built the way every existing checkpoint was carries nothing."""
    torch.manual_seed(0)
    head = CliffClassicalNN(atom_model=nested_hfvr_vw_model, **HEAD_KWARGS)
    assert head.trainable_polarizability_scale is False
    assert head.polarizability_log_scale is None
    assert not [
        key for key in head.state_dict() if "polarizability" in key
    ]
    assert head.get_config()["trainable_polarizability_scale"] is False


def test_enabled_head_carries_one_zero_seeded_vector(nested_hfvr_vw_model):
    torch.manual_seed(0)
    head = CliffClassicalNN(
        atom_model=nested_hfvr_vw_model,
        trainable_polarizability_scale=True,
        **HEAD_KWARGS,
    )
    scale = head.polarizability_log_scale
    assert isinstance(scale, torch.nn.Parameter)
    assert scale.requires_grad
    assert scale.shape == constants.polarizability_table.shape
    # Seeded at exactly zero, not merely close to it: `exp(0) == 1` bitwise is
    # what the bit-identity test below relies on.
    assert torch.equal(scale.detach(), torch.zeros_like(scale))
    assert "polarizability_log_scale" in head.state_dict()
    assert head.get_config()["trainable_polarizability_scale"] is True


def test_the_default_table_is_the_same_object_it_always_was(
    nested_hfvr_vw_model,
):
    """No copy, no `.to()`, no rebuild on the untouched path."""
    harness = _harness(nested_hfvr_vw_model)
    dimer = harness.dimer_model
    assert dimer._polarizability_table() is dimer.polarizability_table


# ------------------------------------------------------------- the physics


def test_a_zero_scale_reproduces_the_constant_table(nested_hfvr_vw_model):
    harness = _harness(
        nested_hfvr_vw_model, trainable_polarizability_scale=True
    )
    dimer = harness.dimer_model
    # `equal_nan` because 16 of the 103 entries are NaN -- elements the table
    # carries no free-atom value for -- and they have to come through as NaN.
    torch.testing.assert_close(
        dimer._polarizability_table().detach(),
        dimer.polarizability_table,
        rtol=0,
        atol=0,
        equal_nan=True,
    )


def test_the_scale_multiplies_alpha_exponentially(nested_hfvr_vw_model):
    """Exponential so the scale cannot cross zero and flip a polarizability."""
    harness = _harness(
        nested_hfvr_vw_model, trainable_polarizability_scale=True
    )
    head = harness.model
    with torch.no_grad():
        head.polarizability_log_scale[8] = math.log(1.5)
        head.polarizability_log_scale[1] = -math.log(2.0)
    table = harness.dimer_model._polarizability_table()
    base = harness.dimer_model.polarizability_table
    assert table[8].item() == pytest.approx(1.5 * base[8].item())
    assert table[1].item() == pytest.approx(0.5 * base[1].item())
    # Every element the run never touched is left exactly alone, NaNs
    # included.
    untouched = [z for z in range(base.numel()) if z not in (1, 8)]
    torch.testing.assert_close(
        table[untouched].detach(),
        base[untouched],
        rtol=0,
        atol=0,
        equal_nan=True,
    )
    assert torch.isnan(table).sum() == torch.isnan(base).sum()


def test_enabling_the_scale_changes_no_energy(
    nested_hfvr_vw_model, synthetic_dimer_batch
):
    """The warm-start invariant: bit-identical, not merely close."""
    torch.manual_seed(0)
    harness = _harness(nested_hfvr_vw_model)
    with torch.no_grad():
        before = harness.dimer_model(synthetic_dimer_batch)[0].clone()
    harness.model.enable_trainable_polarizability_scale()
    with torch.no_grad():
        after = harness.dimer_model(synthetic_dimer_batch)[0]
    assert torch.equal(before, after)


def test_a_nonzero_scale_does_change_the_induction_energy(
    nested_hfvr_vw_model, synthetic_dimer_batch
):
    """The complement of the test above: the lever is actually connected."""
    torch.manual_seed(0)
    harness = _harness(
        nested_hfvr_vw_model, trainable_polarizability_scale=True
    )
    with torch.no_grad():
        before = harness.dimer_model(synthetic_dimer_batch)[0].clone()
        harness.model.polarizability_log_scale.fill_(math.log(1.3))
        after = harness.dimer_model(synthetic_dimer_batch)[0]
    # Column order is (elst, exch, ind); only induction may move.
    assert torch.equal(before[:, 0], after[:, 0])
    assert torch.equal(before[:, 1], after[:, 1])
    assert not torch.equal(before[:, 2], after[:, 2])


def test_the_energy_gradient_reaches_the_scale(
    nested_hfvr_vw_model, synthetic_dimer_batch
):
    torch.manual_seed(0)
    harness = _harness(
        nested_hfvr_vw_model, trainable_polarizability_scale=True
    )
    scale = harness.model.polarizability_log_scale
    harness.dimer_model(synthetic_dimer_batch)[0][:, 2].sum().backward()
    assert scale.grad is not None
    # Only the elements actually present in the batch (H and O) get a
    # gradient; `index_select` leaves the rest of the table untouched.
    present = torch.zeros_like(scale, dtype=torch.bool)
    present[1] = True
    present[8] = True
    assert torch.any(scale.grad[present] != 0.0)
    # Exactly zero, and in particular NOT NaN. The table's 16 NaN entries would
    # otherwise reach here as `0 * NaN`, and component clipping takes one norm
    # over the whole induction group -- one NaN there turns every induction
    # gradient into NaN on the first step.
    assert torch.all(torch.isfinite(scale.grad))
    assert torch.equal(
        scale.grad[~present], torch.zeros_like(scale.grad[~present])
    )


# ------------------------------------------------------------- checkpoints


def test_checkpoint_round_trips_the_scale(tmp_path, nested_hfvr_vw_model):
    torch.manual_seed(0)
    harness = _harness(
        nested_hfvr_vw_model, trainable_polarizability_scale=True
    )
    with torch.no_grad():
        harness.model.polarizability_log_scale[8] = math.log(1.25)
    path = tmp_path / "scaled.pt"
    model_io.save_checkpoint(harness._create_checkpoint(), str(path))
    reloaded = mtp_mtp.CliffClassicalOverlapModel(
        atom_model=None,
        pre_trained_model_path=str(path),
        ds_root=None,
        use_GPU=False,
        ignore_database_null=True,
    )
    assert reloaded.model.trainable_polarizability_scale is True
    assert torch.allclose(
        reloaded.model.polarizability_log_scale.detach(),
        harness.model.polarizability_log_scale.detach(),
    )


def test_a_checkpoint_written_before_this_existed_still_loads(
    tmp_path, nested_hfvr_vw_model
):
    """Strictly, with no key for the scale anywhere in it.

    This is the reason the flag is off at construction and the parameter is
    registered as ``None``: an old checkpoint replays its own recorded
    architecture, so it rebuilds a head that has no such key and
    ``load_state_dict`` stays strict.
    """
    torch.manual_seed(0)
    harness = _harness(nested_hfvr_vw_model)
    path = tmp_path / "legacy.pt"
    checkpoint = harness._create_checkpoint()
    # Emulate a checkpoint predating the flag: drop it from the config the way
    # a run that never knew about it would have written one.
    for value in checkpoint.values():
        if isinstance(value, dict):
            value.pop("trainable_polarizability_scale", None)
    model_io.save_checkpoint(checkpoint, str(path))
    reloaded = mtp_mtp.CliffClassicalOverlapModel(
        atom_model=None,
        pre_trained_model_path=str(path),
        ds_root=None,
        use_GPU=False,
        ignore_database_null=True,
    )
    assert reloaded.model.trainable_polarizability_scale is False
    assert reloaded.model.polarizability_log_scale is None


# --------------------------------------------------------------- optimizer


def test_optimizer_groups_stay_legacy_without_the_scale(nested_hfvr_vw_model):
    harness = _harness(nested_hfvr_vw_model)
    groups = harness._optimizer_parameter_groups(5e-4, None)
    assert not isinstance(groups, list)


def test_optimizer_splits_three_ways_with_thole_and_alpha(
    nested_hfvr_vw_model,
):
    harness = _harness(
        nested_hfvr_vw_model, trainable_polarizability_scale=True
    )
    groups = harness._optimizer_parameter_groups(5e-4, 2.5e-5, 1e-3)
    assert [group["group_name"] for group in groups] == [
        "base",
        "thole",
        "polarizability",
    ]
    assert [group["lr"] for group in groups] == [5e-4, 2.5e-5, 1e-3]
    assert groups[2]["params"] == [harness.model.polarizability_log_scale]
    thole_ids = {id(parameter) for parameter in groups[1]["params"]}
    assert thole_ids == {
        id(parameter)
        for column in (
            CLIFF_CLASSICAL_THOLE_DIRECT_INDEX,
            CLIFF_CLASSICAL_THOLE_MUTUAL_INDEX,
        )
        for module in (
            harness.model.guess_layer[column],
            harness.model.param_readout_layers[column],
        )
        for parameter in module.parameters()
        if parameter.requires_grad
    }
    grouped = [parameter for group in groups for parameter in group["params"]]
    assert len(grouped) == len({id(parameter) for parameter in grouped})
    assert {id(parameter) for parameter in grouped} == {
        id(parameter)
        for parameter in harness.model.parameters()
        if parameter.requires_grad
    }


def test_alpha_alone_splits_two_ways(nested_hfvr_vw_model):
    """No Thole split requested: base and polarizability, nothing else."""
    harness = _harness(
        nested_hfvr_vw_model, trainable_polarizability_scale=True
    )
    groups = harness._optimizer_parameter_groups(5e-4, None, 1e-3)
    assert [group["group_name"] for group in groups] == [
        "base",
        "polarizability",
    ]


def test_a_zero_rate_is_a_legal_control_arm(nested_hfvr_vw_model):
    """`thole_lr` rejects zero; this must not, because zero is the control."""
    harness = _harness(
        nested_hfvr_vw_model, trainable_polarizability_scale=True
    )
    groups = harness._optimizer_parameter_groups(5e-4, None, 0.0)
    assert groups[-1]["lr"] == 0.0
    assert groups[-1]["params"] == [harness.model.polarizability_log_scale]


def test_a_scale_with_no_rate_fails_closed(nested_hfvr_vw_model):
    """Sweeping it into `base` at the trunk's rate is not a small mistake."""
    harness = _harness(
        nested_hfvr_vw_model, trainable_polarizability_scale=True
    )
    with pytest.raises(ValueError, match="polarizability_lr must be given"):
        harness._optimizer_parameter_groups(5e-4, None)


def test_a_rate_with_no_scale_fails_closed(nested_hfvr_vw_model):
    harness = _harness(nested_hfvr_vw_model)
    with pytest.raises(ValueError, match="has no trainable polarizability"):
        harness._optimizer_parameter_groups(5e-4, None, 1e-3)


@pytest.mark.parametrize("rate", [-1e-3, float("nan"), float("inf")])
def test_a_nonfinite_or_negative_rate_is_rejected(rate):
    with pytest.raises(ValueError, match="finite and non-negative"):
        mtp_mtp._validate_polarizability_lr(rate)


# ------------------------------------------------- component gradient clip


def test_the_scale_clips_with_induction(nested_hfvr_vw_model):
    """It multiplies the induced dipoles and nothing else.

    Mis-grouping it would not be silent: the exact-coverage check inside
    ``_component_gradient_parameter_groups`` raises on every step.
    """
    harness = _harness(
        nested_hfvr_vw_model, trainable_polarizability_scale=True
    )
    groups = harness._component_gradient_parameter_groups()
    assert tuple(groups) == ("electrostatics", "exchange", "induction")
    scale = harness.model.polarizability_log_scale
    assert any(
        parameter is scale for parameter in groups["induction"]
    )
    assert not any(
        parameter is scale
        for component in ("electrostatics", "exchange")
        for parameter in groups[component]
    )


def test_component_clipping_runs_with_the_scale_enabled(nested_hfvr_vw_model):
    harness = _harness(
        nested_hfvr_vw_model, trainable_polarizability_scale=True
    )
    for parameter in harness.model.parameters():
        if parameter.requires_grad:
            parameter.grad = torch.full_like(parameter, 2.0)
    reported = harness._clip_gradient_norms(1.0, "component")
    assert set(reported) == {"electrostatics", "exchange", "induction"}


def test_component_clipping_stays_finite_after_a_real_backward(
    nested_hfvr_vw_model, synthetic_dimer_batch
):
    """The regression that motivates the NaN mask in `_polarizability_table`.

    Filling `.grad` by hand cannot catch it: the NaNs only appear when the
    gradient actually flows back through `index_select` into a table that has
    NaN entries for the elements the batch does not contain.
    """
    torch.manual_seed(0)
    harness = _harness(
        nested_hfvr_vw_model, trainable_polarizability_scale=True
    )
    harness.dimer_model(synthetic_dimer_batch)[0].sum().backward()
    reported = harness._clip_gradient_norms(1.0, "component")
    assert all(math.isfinite(float(value)) for value in reported.values())
    for parameter in harness.model.parameters():
        if parameter.grad is not None:
            assert torch.all(torch.isfinite(parameter.grad))


# --------------------------------------------------------------------- CLI


def test_the_cli_advertises_both_flags():
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "train_models.py"), "--help"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, result.stderr
    assert "--trainable_polarizability_scale" in result.stdout
    assert "--polarizability_lr" in result.stdout
