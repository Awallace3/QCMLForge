"""Focused tests for the learning-rate flag routing in ``train_models.py``."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(REPO_ROOT))
import train_models  # noqa: E402

# Every spelling of the AP3-D3 route that ``--train_apnet`` accepts.  The
# dispatch in ``train_pairwise_model`` matches these case-insensitively.
APNETD3_ALIASES = (
    "APNetD3",
    "APNet3D3",
    "APNet3-d3-fused",
    "APNet3-fused-d3",
)


@pytest.mark.parametrize("alias", APNETD3_ALIASES)
def test_apnetd3_aliases_are_recognized(alias):
    assert train_models.is_apnetd3_model_type(alias)


@pytest.mark.parametrize("alias", ["APNet2", "APNet2-fused", "APNet3-fused"])
def test_non_apnetd3_routes_are_not_recognized(alias):
    assert not train_models.is_apnetd3_model_type(alias)


@pytest.mark.parametrize("alias", APNETD3_ALIASES)
def test_end_lr_reaches_train_for_every_apnetd3_alias(alias):
    """``--end_lr`` must be forwarded by every alias that accepts it.

    ``APNet3-fused-d3`` used to pass the ``end_lr`` validation gate and then be
    dropped by a dispatch branch that listed only the other three spellings, so
    the run silently trained at a flat learning rate.
    """
    kwargs = train_models.lr_schedule_train_kwargs(
        alias, end_lr=5e-6, lr_decay=None
    )
    assert kwargs == {"end_lr": 5e-6}


def test_non_apnetd3_routes_receive_lr_decay():
    kwargs = train_models.lr_schedule_train_kwargs(
        "APNet2", end_lr=None, lr_decay=1.0
    )
    assert kwargs == {"lr_decay": 1.0}
