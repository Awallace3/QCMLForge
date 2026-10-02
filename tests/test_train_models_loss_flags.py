"""Focused tests for the component-loss flag routing in ``train_models.py``."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(REPO_ROOT))
import train_models  # noqa: E402

from apnet_pt.AtomPairwiseModels.apnet2 import APNet2Model  # noqa: E402
from apnet_pt.AtomPairwiseModels.apnet2_fused import (  # noqa: E402
    APNet2_AM_Model,
)
from apnet_pt.AtomPairwiseModels.apnet3_d3_fused import (  # noqa: E402
    APNet3D3_AtomType_Model,
)
from apnet_pt.AtomPairwiseModels.component_losses import (  # noqa: E402
    component_huber,
    component_mse,
)

PREDS = torch.tensor([[1.0, 2.0, 3.0, 4.0], [-5.0, 0.5, 2.0, -1.0]])
LABELS = torch.tensor([[1.5, 1.0, 3.5, 4.5], [-4.0, 0.0, 1.0, -1.5]])


@pytest.mark.parametrize(
    "model_cls", [APNet2Model, APNet2_AM_Model, APNet3D3_AtomType_Model]
)
def test_both_pairwise_harnesses_expose_loss_fn(model_cls):
    """``train_models`` routes the criterion by keyword, so it must be a parameter."""
    assert "loss_fn" in inspect.signature(model_cls.train).parameters


def _loss_kwargs(component_loss, **kwargs):
    return train_models.component_loss_train_kwargs(
        "APNet2", APNet2Model.train, component_loss, **kwargs
    )


def test_default_route_passes_no_loss_fn():
    """The default adds nothing, so ``train()`` is called exactly as before."""
    assert _loss_kwargs("component_mse") == {}


def test_huber_flag_builds_a_configured_loss():
    kwargs = _loss_kwargs("component_huber", huber_delta=2.5)
    loss_fn = kwargs["loss_fn"]
    assert torch.allclose(
        loss_fn(PREDS, LABELS), component_huber(PREDS, LABELS, delta=2.5)
    )


def test_relative_flag_forwards_eps():
    a = _loss_kwargs("component_relative_mse", relative_loss_eps=3.0)["loss_fn"](
        PREDS, LABELS
    )
    b = _loss_kwargs("component_relative_mse", relative_loss_eps=100.0)["loss_fn"](
        PREDS, LABELS
    )
    assert a > b


def test_weighted_flag_forwards_the_weight_vector():
    kwargs = _loss_kwargs(
        "component_weighted_mse", component_loss_weights=(1.0, 1.0, 1.0, 1.0)
    )
    assert torch.allclose(
        kwargs["loss_fn"](PREDS, LABELS), component_mse(PREDS, LABELS)
    )


def test_weighted_flag_requires_weights():
    """Silently defaulting to ones would make the flag inert without warning."""
    with pytest.raises(ValueError, match="component_loss_weights"):
        _loss_kwargs("component_weighted_mse")


def test_unknown_loss_name_is_rejected():
    with pytest.raises(ValueError, match="unknown component loss"):
        _loss_kwargs("not_a_loss")


def test_route_without_loss_fn_refuses_a_selected_loss():
    """Dropping the flag would train the default objective under a loss label."""

    def train(model_path=None, n_epochs=1):
        pass

    with pytest.raises(ValueError, match="does not support --component_loss"):
        train_models.component_loss_train_kwargs(
            "APNet3", train, "component_huber"
        )
    assert (
        train_models.component_loss_train_kwargs("APNet3", train, "component_mse")
        == {}
    )


def test_cli_accepts_the_loss_flags():
    parser = train_models.build_arg_parser()
    args = parser.parse_args(
        [
            "--component_loss", "component_huber",
            "--huber_delta", "2.0",
            "--component_loss_weights", "2,1,1,1",
        ]
    )
    assert args.component_loss == "component_huber"
    assert args.huber_delta == 2.0
    assert args.component_loss_weights == [2.0, 1.0, 1.0, 1.0]


def test_cli_defaults_to_the_baseline_loss():
    args = train_models.build_arg_parser().parse_args([])
    assert args.component_loss == "component_mse"
    assert args.component_loss_weights is None


@pytest.mark.parametrize("model_cls", [APNet2Model, APNet3D3_AtomType_Model])
def test_pairwise_harnesses_expose_checkpoint_metric(model_cls):
    """Non-default-loss arms select by total MAE, so the route must accept it."""
    assert "checkpoint_metric" in inspect.signature(model_cls.train).parameters
