"""Device/dtype placement regressions for tensors created inside ``forward``.

These tests run on any host. Inputs live on the ``meta`` device in float64, so a
tensor created with the default device (CPU) or default dtype (float32) is
caught without needing a GPU.
"""

import inspect

import pytest
import torch

from apnet_pt.AtomPairwiseModels import (
    apnet2,
    apnet2_fused,
    apnet3,
    apnet3_d3_fused,
    apnet3_fused,
    apnet3_fused_variants,
)


def _build(module_cls):
    """Construct ``module_cls`` with placeholder submodules for required args."""
    required = [
        name
        for name, param in inspect.signature(module_cls.__init__).parameters.items()
        if name != "self" and param.default is inspect.Parameter.empty
    ]
    return module_cls(**{name: torch.nn.Identity() for name in required})


@pytest.mark.parametrize(
    "module_cls",
    [
        apnet2.APNet2_MPNN,
        apnet2_fused.APNet2_AM_MPNN,
        apnet3.APNet3_MPNN,
        apnet3_fused.APNet3_AtomType_MPNN,
        apnet3_fused_variants.APNet3_AtomType_MPNN,
        apnet3_d3_fused.APNet3D3_AtomType_MPNN,
    ],
    ids=lambda cls: f"{cls.__module__.rsplit('.', 1)[-1]}.{cls.__name__}",
)
def test_get_messages_without_edges_follows_input_device_and_dtype(module_cls):
    """Single-atom monomers have no edges; the empty message tensor must still
    match the device and dtype of the atom features it is combined with."""
    model = _build(module_cls)
    h = torch.zeros(3, model.n_embed, dtype=torch.float64, device="meta")
    rbf = torch.zeros(0, model.n_rbf, dtype=torch.float64, device="meta")
    no_edges = torch.zeros(0, dtype=torch.long, device="meta")

    messages = model.get_messages(h, h, rbf, no_edges, no_edges)

    assert messages.device == h.device
    assert messages.dtype == h.dtype
    assert messages.shape[0] == 0
