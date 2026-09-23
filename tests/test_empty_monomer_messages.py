"""Regression tests for the zero-intramolecular-edge branch of ``get_messages``.

A monatomic monomer (a bare ion such as ``Na+`` or ``Cl-``) contributes no
intramolecular edges.  When *every* monomer A in a batch is monatomic,
``e_AA_source`` is empty and ``get_messages`` takes its early-return branch.
That branch used to build its empty message block with a bare
``torch.zeros(0, width)``, which ignores the device and dtype of the inputs and
so always lands on CPU in float32.  On GPU the very next line,

    mA_i = scatter_sum_compile(mA_ij, e_AA_source, int(natomA))

allocates ``mA_ij.new_zeros(...)`` on CPU and then scatter-adds a CUDA index
into it, raising::

    RuntimeError: Expected all tensors to be on the same device, but got index
    is on cuda:0, different from other tensors on cpu

The batch composition required is rare, so the failure surfaces deep into a
training run rather than at step zero.  These tests pin device and dtype
propagation for every model family that carries a copy of this method.
"""

from __future__ import annotations

import pytest
import torch

from apnet_pt.AtomModels.ap2_atom_model import AtomMPNN
from apnet_pt.AtomModels.ap2_hirshfeld_atom_model import AtomHirshfeldMPNN
from apnet_pt.AtomPairwiseModels.apnet2 import APNet2_MPNN
from apnet_pt.AtomPairwiseModels.apnet2_fused import APNet2_AM_MPNN
from apnet_pt.AtomPairwiseModels.apnet3 import APNet3_MPNN
from apnet_pt.AtomPairwiseModels.apnet3_d3_fused import APNet3D3_AtomType_MPNN
from apnet_pt.AtomPairwiseModels.apnet3_fused import APNet3_AtomType_MPNN
from apnet_pt.AtomPairwiseModels.apnet3_fused_variants import (
    APNet3_AtomType_MPNN as APNet3_AtomType_MPNN_Variants,
)
from apnet_pt.AtomPairwiseModels.mtp_mtp import AtomTypeParamNN, DimerProp
from apnet_pt.util import scatter_sum_compile

SMALL = dict(n_message=1, n_rbf=4, n_neuron=16, n_embed=4)


@pytest.fixture(scope="module")
def dimer_prop():
    hirshfeld = AtomHirshfeldMPNN(n_message=1, n_rbf=4, n_neuron=16, n_embed=4)
    atpnn = AtomTypeParamNN(
        atom_model=hirshfeld,
        n_message=1,
        n_neuron=16,
        n_embed=4,
        n_params=1,
        freeze_atom_model=True,
    )
    return DimerProp(ATParam=atpnn, freeze_atom_model=True)


@pytest.fixture(scope="module")
def mpnns(dimer_prop):
    """One small instance of every class that defines ``get_messages``."""
    atom_model = AtomMPNN(n_message=1, n_rbf=4, n_neuron=16, n_embed=4)
    return {
        "apnet2": APNet2_MPNN(**SMALL),
        "apnet2_fused": APNet2_AM_MPNN(atom_model=atom_model, **SMALL),
        "apnet3": APNet3_MPNN(**SMALL),
        "apnet3_fused": APNet3_AtomType_MPNN(dimer_prop_model=dimer_prop, **SMALL),
        "apnet3_fused_variants": APNet3_AtomType_MPNN_Variants(
            dimer_prop_model=dimer_prop, **SMALL
        ),
        "apnet3_d3_fused": APNet3D3_AtomType_MPNN(
            dimer_prop_model=dimer_prop, **SMALL
        ),
    }


MODEL_KEYS = (
    "apnet2",
    "apnet2_fused",
    "apnet3",
    "apnet3_fused",
    "apnet3_fused_variants",
    "apnet3_d3_fused",
)


def _empty_edge_inputs(mpnn, dtype, device):
    """Inputs for a batch whose monomers are all single atoms: zero edges."""
    natom = 2
    h0 = torch.zeros(natom, mpnn.n_embed, dtype=dtype, device=device)
    h = torch.zeros(natom, mpnn.n_embed, dtype=dtype, device=device)
    rbf = torch.zeros(0, mpnn.n_rbf, dtype=dtype, device=device)
    e_source = torch.zeros(0, dtype=torch.long, device=device)
    e_target = torch.zeros(0, dtype=torch.long, device=device)
    return h0, h, rbf, e_source, e_target


def _expected_width(mpnn):
    return mpnn.n_embed * 4 * mpnn.n_rbf + mpnn.n_embed * 4 + mpnn.n_rbf


@pytest.mark.parametrize("key", MODEL_KEYS)
def test_empty_messages_keep_the_expected_width(mpnns, key):
    mpnn = mpnns[key]
    args = _empty_edge_inputs(mpnn, torch.float32, torch.device("cpu"))
    m_ij = mpnn.get_messages(*args)
    assert m_ij.shape == (0, _expected_width(mpnn))


@pytest.mark.parametrize("key", MODEL_KEYS)
def test_empty_messages_follow_the_input_dtype(mpnns, key):
    """The populated branch is dtype-transparent, so the empty one must be too.

    ``torch.zeros(0, width)`` hardcodes the global default dtype instead, which
    is the same class of bug as hardcoding the CPU device.
    """
    mpnn = mpnns[key]
    args = _empty_edge_inputs(mpnn, torch.float64, torch.device("cpu"))
    m_ij = mpnn.get_messages(*args)
    assert m_ij.dtype == torch.float64


@pytest.mark.parametrize("key", MODEL_KEYS)
def test_empty_messages_scatter_without_a_device_mismatch(mpnns, key):
    """The line that actually crashed: scatter the empty block back to atoms."""
    mpnn = mpnns[key]
    h0, h, rbf, e_source, e_target = _empty_edge_inputs(
        mpnn, torch.float32, torch.device("cpu")
    )
    m_ij = mpnn.get_messages(h0, h, rbf, e_source, e_target)
    m_i = scatter_sum_compile(m_ij, e_source, h.shape[0])
    assert m_i.shape == (h.shape[0], _expected_width(mpnn))
    assert torch.all(m_i == 0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
@pytest.mark.parametrize("key", MODEL_KEYS)
def test_empty_messages_stay_on_the_input_device(mpnns, key):
    mpnn = mpnns[key]
    device = torch.device("cuda:0")
    h0, h, rbf, e_source, e_target = _empty_edge_inputs(mpnn, torch.float32, device)
    m_ij = mpnn.get_messages(h0, h, rbf, e_source, e_target)
    assert m_ij.device.type == "cuda"
    m_i = scatter_sum_compile(m_ij, e_source, h.shape[0])
    assert m_i.device.type == "cuda"
