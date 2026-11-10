"""Device/dtype placement regressions found by running the suite on CUDA.

``get_messages`` checks run on any host: inputs live on the ``meta`` device in
float64, so a tensor created with the default device (CPU) or default dtype
(float32) is caught without needing a GPU. Host-side NumPy assembly helpers
need a real CUDA tensor to reproduce, so those checks are marked ``gpu``.
"""

import inspect

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from apnet_pt.AtomPairwiseModels import (
    apnet2,
    apnet2_fused,
    apnet3,
    apnet3_d3_fused,
    apnet3_fused,
    apnet3_fused_variants,
    mtp_mtp,
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


def _two_dimer_pair_batch():
    """Two dimers: dimer 0 has A atoms {0, 1} and B atom {0}; dimer 1 has A
    atom {2} and B atoms {1, 2}. AB edges are split into short/long range."""
    source = torch.tensor([0, 1, 2, 2])
    target = torch.tensor([0, 0, 1, 2])
    return Data(
        e_ABsr_source=source[:3],
        e_ABsr_target=target[:3],
        e_ABlr_source=source[3:],
        e_ABlr_target=target[3:],
        dimer_ind=torch.tensor([0, 0, 1]),
        e_ABfull_source=source,
        e_ABfull_target=target,
        dimer_ind_full=torch.tensor([0, 0, 1, 1]),
        indA=torch.tensor([0, 0, 1]),
        indB=torch.tensor([0, 1, 1]),
    )


def _assemble_args(harness_cls, device):
    batch = _two_dimer_pair_batch().to(device)
    energies = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64, device=device)
    if harness_cls in (apnet2.APNet2Model, apnet2_fused.APNet2_AM_Model,
                       mtp_mtp.AM_DimerParam_Model):
        return batch, energies[:3], energies[3:]
    return batch, energies, 10.0 * energies


@pytest.mark.gpu
@pytest.mark.parametrize(
    "harness_cls",
    [
        apnet2.APNet2Model,
        apnet2_fused.APNet2_AM_Model,
        apnet3_fused.APNet3_AtomType_Model,
        apnet3_fused_variants.APNet3_AtomType_Model,
        mtp_mtp.AM_DimerParam_Model,
    ],
    ids=lambda cls: f"{cls.__module__.rsplit('.', 1)[-1]}.{cls.__name__}",
)
def test_assemble_mtp_pairs_accepts_cuda_batches(harness_cls):
    """``predict_qcel_mols(return_elst=/return_classical_pairs=True)`` hands
    the helper on-device tensors; the host-side pair tables must not depend on
    where the batch lives."""
    harness = harness_cls(use_GPU=False)

    expected = harness._assemble_mtp_pairs(*_assemble_args(harness_cls, "cpu"))
    actual = harness._assemble_mtp_pairs(*_assemble_args(harness_cls, "cuda"))

    if isinstance(expected, tuple):
        expected = [t for group in expected for t in group]
        actual = [t for group in actual for t in group]
    assert len(actual) == len(expected)
    for got, want in zip(actual, expected):
        np.testing.assert_array_equal(np.asarray(got, dtype=float), want)
