"""Tests for ``atom_model_lr``: a separate rate for the pretrained trunk.

``--unfreeze_atom_model`` makes the nested ``atom_model`` trainable. Before
this existed the optimizer had no way to express "trunk slower than head":
``_optimizer_parameter_groups`` splits out Thole and the polarizability scale
and sweeps *everything else* into ``base`` at a single rate. So unfreezing put
1,892,550 pretrained parameters at the rate tuned for a 231,015-parameter
readout head.

That was measured, not assumed. Jobs 12632350 (lr 1.5e-4) and 12632352
(lr 3e-5) each warm started from the campaign's best checkpoint, and each
destroyed it inside a single epoch: validation induction went non-finite in
both, 7735 and 6088 batches respectively were skipped on non-finite gradient
norms, and neither ever wrote a checkpoint. A 5x reduction in rate changed
which numbers were worst but not the outcome, which is the argument for
separating the two rates rather than hunting for a safe single one.

Hence the shape tested here: the trunk gets its own group, and an unfrozen
trunk with no rate of its own is an error rather than a silent inheritance.
"""
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apnet_pt.AtomPairwiseModels import mtp_mtp  # noqa: E402

HEAD_KWARGS = dict(n_message=1, n_neuron=8, n_embed=4)


def _harness(nested, **overrides):
    return mtp_mtp.CliffClassicalOverlapModel(
        atom_model=nested,
        ds_root=None,
        use_GPU=False,
        ignore_database_null=True,
        **{**HEAD_KWARGS, **overrides},
    )


def _unfreeze(harness):
    trunk = list(harness.model.atom_model.parameters())
    for parameter in trunk:
        parameter.requires_grad_(True)
    assert trunk, "fixture has no trunk parameters to unfreeze"
    return trunk


# ------------------------------------------------------------- default is off


def test_frozen_trunk_needs_no_rate_and_adds_no_group(nested_hfvr_vw_model):
    """The default path is untouched, so existing runs are bit-identical."""
    harness = _harness(nested_hfvr_vw_model)
    groups = harness._optimizer_parameter_groups(5e-4, 2.5e-5)
    assert [group["group_name"] for group in groups] == ["base", "thole"]


def test_a_rate_alone_still_takes_the_legacy_path(nested_hfvr_vw_model):
    """No split of any kind requested keeps the historical single iterator."""
    harness = _harness(nested_hfvr_vw_model)
    assert not isinstance(
        harness._optimizer_parameter_groups(5e-4, None), list
    )


# ------------------------------------------------------------------ the group


def test_unfrozen_trunk_gets_its_own_group_at_its_own_rate(
    nested_hfvr_vw_model,
):
    harness = _harness(nested_hfvr_vw_model)
    trunk = _unfreeze(harness)
    groups = harness._optimizer_parameter_groups(5e-4, 2.5e-5, None, 1e-6)
    assert [group["group_name"] for group in groups] == [
        "base",
        "thole",
        "atom_model",
    ]
    assert [group["lr"] for group in groups] == [5e-4, 2.5e-5, 1e-6]
    assert {id(p) for p in groups[-1]["params"]} == {id(p) for p in trunk}
    # The trunk left `base`: that is the entire point.  1.89M parameters at
    # the head's rate is what the two runaway jobs did.
    assert not {id(p) for p in groups[0]["params"]} & {id(p) for p in trunk}
    # Still a partition of the trainable set, disjoint and complete.
    grouped = [p for group in groups for p in group["params"]]
    assert len(grouped) == len({id(p) for p in grouped})
    assert {id(p) for p in grouped} == {
        id(p) for p in harness.model.parameters() if p.requires_grad
    }


def test_the_trunk_rate_survives_without_a_thole_split(nested_hfvr_vw_model):
    """`atom_model_lr` alone must not fall through to the legacy iterator."""
    harness = _harness(nested_hfvr_vw_model)
    _unfreeze(harness)
    groups = harness._optimizer_parameter_groups(5e-4, None, None, 1e-6)
    assert [group["group_name"] for group in groups] == ["base", "atom_model"]


def test_zero_is_a_legal_rate(nested_hfvr_vw_model):
    """Zero carries the trunk through the checkpoint without moving it.

    That is the control arm for any trunk experiment, and it is why this
    borrows the polarizability validator rather than `_validate_bound_scale`,
    which rejects zero.
    """
    harness = _harness(nested_hfvr_vw_model)
    _unfreeze(harness)
    groups = harness._optimizer_parameter_groups(5e-4, None, None, 0.0)
    assert groups[-1]["group_name"] == "atom_model"
    assert groups[-1]["lr"] == 0.0


def test_an_optimizer_actually_steps_the_two_rates_apart(
    nested_hfvr_vw_model,
):
    """End to end through Adam: the group is not just a label."""
    harness = _harness(nested_hfvr_vw_model)
    trunk = _unfreeze(harness)
    groups = harness._optimizer_parameter_groups(5e-4, None, None, 0.0)
    optimizer = torch.optim.Adam(groups, lr=5e-4)
    head_parameter = groups[0]["params"][0]
    before_trunk = [p.detach().clone() for p in trunk]
    before_head = head_parameter.detach().clone()
    for parameter in (*trunk, head_parameter):
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    # lr 0.0 on the trunk means it does not move at all...
    for parameter, before in zip(trunk, before_trunk):
        assert torch.equal(parameter, before)
    # ...while the head takes its ordinary step.
    assert not torch.equal(head_parameter, before_head)


# ------------------------------------------------------------- fails closed


def test_an_unfrozen_trunk_with_no_rate_is_an_error(nested_hfvr_vw_model):
    """The bug that cost jobs 12632350/12632352, caught before the dataset.

    Silently inheriting `lr` is exactly what happened, and it is not a small
    mistake: it moves more parameters than the rest of the model combined.
    """
    harness = _harness(nested_hfvr_vw_model)
    _unfreeze(harness)
    with pytest.raises(ValueError, match="no atom_model_lr"):
        harness._optimizer_parameter_groups(5e-4, 2.5e-5)


def test_a_rate_on_a_frozen_trunk_is_an_error(nested_hfvr_vw_model):
    """The other half of the same typo: a rate with nothing to apply it to."""
    harness = _harness(nested_hfvr_vw_model)
    with pytest.raises(ValueError, match="atom_model is frozen"):
        harness._optimizer_parameter_groups(5e-4, 2.5e-5, None, 1e-6)


@pytest.mark.parametrize("bad", [-1e-6, float("nan"), float("inf"), "fast"])
def test_a_malformed_rate_is_rejected(nested_hfvr_vw_model, bad):
    harness = _harness(nested_hfvr_vw_model)
    _unfreeze(harness)
    with pytest.raises(ValueError, match="atom_model_lr must be finite"):
        harness._optimizer_parameter_groups(5e-4, None, None, bad)
