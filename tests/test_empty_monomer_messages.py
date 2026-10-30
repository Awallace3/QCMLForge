"""Regression tests for the zero-intramolecular-edge branch of ``get_messages``.

When every monomer A in a batch is monatomic (a bare ion), ``get_messages``
returned ``torch.zeros(0, width)`` on CPU, and the following scatter failed
with a device mismatch on GPU.  Pin device and dtype for every copy.
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


def _empty_messages(mpnn, dtype, device):
    """Run ``get_messages`` for a batch of single-atom monomers (zero edges)."""
    h = torch.zeros(2, mpnn.n_embed, dtype=dtype, device=device)
    rbf = torch.zeros(0, mpnn.n_rbf, dtype=dtype, device=device)
    edges = torch.zeros(0, dtype=torch.long, device=device)
    return mpnn.get_messages(h, h, rbf, edges, edges), edges


@pytest.mark.parametrize("key", MODEL_KEYS)
def test_empty_messages_keep_width_and_dtype_and_scatter(mpnns, key):
    mpnn = mpnns[key]
    m_ij, edges = _empty_messages(mpnn, torch.float64, torch.device("cpu"))
    width = mpnn.n_embed * 4 * mpnn.n_rbf + mpnn.n_embed * 4 + mpnn.n_rbf
    assert m_ij.shape == (0, width) and m_ij.dtype == torch.float64
    m_i = scatter_sum_compile(m_ij, edges, 2)
    assert m_i.shape == (2, width) and torch.all(m_i == 0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
@pytest.mark.parametrize("key", MODEL_KEYS)
def test_empty_messages_stay_on_the_input_device(mpnns, key):
    m_ij, edges = _empty_messages(mpnns[key], torch.float32, torch.device("cuda:0"))
    assert m_ij.device.type == "cuda"
    assert scatter_sum_compile(m_ij, edges, 2).device.type == "cuda"
