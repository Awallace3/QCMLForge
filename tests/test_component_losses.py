"""Tests for the configurable component-wise pairwise losses."""

import pickle

import pytest
import torch

from apnet_pt.AtomPairwiseModels.component_losses import (
    COMPONENT_LOSS_NAMES,
    build_component_loss,
    component_huber,
    component_mse,
    component_relative_mse,
    component_weighted_mse,
    per_component_mse,
    validate_loss_route,
)

PREDS = torch.tensor(
    [
        [1.0, 2.0, 3.0, 4.0],
        [-5.0, 0.5, 2.0, -1.0],
        [20.0, 10.0, 7.0, 3.0],
    ]
)
LABELS = torch.tensor(
    [
        [1.5, 1.0, 3.5, 4.5],
        [-4.0, 0.0, 1.0, -1.5],
        [25.0, 12.0, 9.0, 3.5],
    ]
)


def test_component_mse_matches_the_inlined_default():
    """The registry baseline must reproduce the loss the harnesses inline today."""
    expected = torch.mean(torch.square(PREDS - LABELS))
    assert torch.allclose(component_mse(PREDS, LABELS), expected)


def test_component_mse_matches_torch_mseloss():
    assert torch.allclose(
        component_mse(PREDS, LABELS), torch.nn.MSELoss()(PREDS, LABELS)
    )


def test_huber_is_quadratic_well_inside_delta():
    """With delta far above every residual, Huber degenerates to half the MSE."""
    loss = component_huber(PREDS, LABELS, delta=1000.0)
    assert torch.allclose(loss, 0.5 * component_mse(PREDS, LABELS), atol=1e-6)


def test_huber_saturates_the_large_error_tail():
    """Doubling an already-large residual must not quadruple its contribution."""
    labels = torch.zeros(1, 4)
    small = component_huber(torch.tensor([[10.0, 0.0, 0.0, 0.0]]), labels, delta=1.0)
    large = component_huber(torch.tensor([[20.0, 0.0, 0.0, 0.0]]), labels, delta=1.0)
    assert large < 4.0 * small
    # linear regime: delta * (|e| - delta / 2), so the ratio is (20-0.5)/(10-0.5)
    assert torch.allclose(large / small, torch.tensor(19.5 / 9.5), atol=1e-6)


def test_relative_mse_downweights_large_magnitude_targets():
    """Equal absolute errors on a big and a small target are not equally costly."""
    preds = torch.tensor([[1.0, 0.0, 0.0, 0.0], [21.0, 0.0, 0.0, 0.0]])
    labels = torch.tensor([[0.0, 0.0, 0.0, 0.0], [20.0, 0.0, 0.0, 0.0]])
    per_row = [
        component_relative_mse(preds[i : i + 1], labels[i : i + 1], eps=1.0)
        for i in range(2)
    ]
    assert per_row[1] < per_row[0]


def test_relative_mse_reduces_to_mse_when_eps_dominates():
    big_eps = component_relative_mse(PREDS, LABELS, eps=1e4)
    assert torch.allclose(big_eps * 1e8, component_mse(PREDS, LABELS), rtol=1e-3)


def test_weighted_mse_respects_per_component_weights():
    """A weight vector of ones is the baseline; lifting one component raises it."""
    ones = component_weighted_mse(PREDS, LABELS, weights=(1.0, 1.0, 1.0, 1.0))
    assert torch.allclose(ones, component_mse(PREDS, LABELS))
    lifted = component_weighted_mse(PREDS, LABELS, weights=(4.0, 1.0, 1.0, 1.0))
    assert lifted > ones


def test_weighted_mse_rejects_a_mismatched_weight_vector():
    with pytest.raises(ValueError):
        component_weighted_mse(PREDS, LABELS, weights=(1.0, 1.0))


@pytest.mark.parametrize("name", sorted(COMPONENT_LOSS_NAMES))
def test_every_registered_loss_builds_and_returns_a_finite_scalar(name):
    loss_fn = build_component_loss(name)
    value = loss_fn(PREDS, LABELS)
    assert value.ndim == 0
    assert torch.isfinite(value)


def test_build_component_loss_forwards_keyword_arguments():
    built = build_component_loss("component_huber", delta=1000.0)
    assert torch.allclose(
        built(PREDS, LABELS), component_huber(PREDS, LABELS, delta=1000.0)
    )


def test_build_component_loss_rejects_an_unknown_name():
    with pytest.raises(ValueError, match="unknown component loss"):
        build_component_loss("not_a_loss")


def test_registered_losses_are_differentiable():
    for name in sorted(COMPONENT_LOSS_NAMES):
        preds = PREDS.clone().requires_grad_(True)
        build_component_loss(name)(preds, LABELS).backward()
        assert preds.grad is not None and torch.isfinite(preds.grad).all()


def test_configured_losses_survive_pickling_for_the_ddp_path():
    """mp.spawn pickles the criterion, so a configured loss must not be a closure."""
    built = build_component_loss(
        "component_weighted_mse", weights=(2.0, 1.0, 1.0, 1.0)
    )
    revived = pickle.loads(pickle.dumps(built))
    assert torch.allclose(revived(PREDS, LABELS), built(PREDS, LABELS))


def test_weighted_mse_takes_three_weights_without_dispersion():
    """A ``no_disp_nn`` model predicts three components, so four weights fail."""
    preds, labels = PREDS[:, :3], LABELS[:, :3]
    assert torch.allclose(
        component_weighted_mse(preds, labels, weights=(1.0, 1.0, 1.0)),
        component_mse(preds, labels),
    )
    with pytest.raises(ValueError, match="3 predicted components"):
        component_weighted_mse(preds, labels, weights=(1.0, 1.0, 1.0, 1.0))


def test_per_component_mse_matches_each_column():
    terms = per_component_mse(PREDS - LABELS)
    expected = torch.mean(torch.square(PREDS - LABELS), dim=0)
    assert torch.equal(torch.stack(terms), expected)


def test_per_component_mse_zero_dispersion_follows_the_errors():
    errors = (PREDS - LABELS)[:, :3].to(torch.float64)
    elst, exch, ind, disp = per_component_mse(errors)
    assert disp.item() == 0.0
    assert disp.dtype == errors.dtype and disp.device == errors.device


def test_transfer_learning_refuses_a_selected_loss():
    validate_loss_route(None, transfer_learning=True)
    validate_loss_route(component_huber, transfer_learning=False)
    with pytest.raises(ValueError, match="transfer_learning"):
        validate_loss_route(component_huber, transfer_learning=True)
