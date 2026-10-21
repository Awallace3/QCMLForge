"""AtomMPNN runs in the dtype of its parameters.

The dipole and quadrupole accumulators used to be hard-coded float32. A
float64 model then added float64 updates into float32 buffers in place, so
mu and Theta came back silently rounded to single precision while q stayed
float64. A model trained end to end in float64 needs all three in float64.
"""

import copy
from types import SimpleNamespace

import pytest
import torch

from apnet_pt.AtomModels.ap2_atom_model import AtomMPNN


def _batch(dtype):
    # Water plus a lone Na+: one monomer with edges, one without.
    R = torch.tensor(
        [
            [0.000, 0.000, 0.000],
            [0.759, 0.000, 0.504],
            [0.260, 0.000, -0.873],
            [4.000, 1.000, 0.500],
        ],
        dtype=dtype,
    )
    src = [0, 0, 1, 1, 2, 2]
    dst = [1, 2, 0, 2, 0, 1]
    return SimpleNamespace(
        x=torch.tensor([8, 1, 1, 11]),
        edge_index=torch.tensor([src, dst]),
        R=R,
        molecule_ind=torch.tensor([0, 0, 0, 1]),
        total_charge=torch.tensor([0.0, 1.0], dtype=dtype),
        natom_per_mol=torch.tensor([3, 1]),
    )


@pytest.fixture(scope="module")
def model32():
    torch.manual_seed(7)
    model = AtomMPNN(n_message=2, n_neuron=16, n_embed=4)
    # Default init leaves the dipole/qpole readouts near zero on so small a
    # molecule; widen them so a precision loss would be visible.
    with torch.no_grad():
        for p in model.parameters():
            p.mul_(3.0)
    return model.eval()


def test_float32_outputs_stay_float32(model32):
    q, mu, th, h = model32(_batch(torch.float32))
    for t in (q, mu, th, h):
        assert t.dtype == torch.float32


def test_float64_outputs_are_float64_and_match_float32(model32):
    model64 = copy.deepcopy(model32).double()
    out32 = model32(_batch(torch.float32))
    out64 = model64(_batch(torch.float64))
    for name, a, b in zip(("q", "mu", "theta", "h"), out32, out64):
        assert b.dtype == torch.float64, name
        scale = max(b.abs().max().item(), 1.0)
        assert torch.allclose(a.double(), b, atol=1e-5 * scale), name
    # The float64 multipoles carry more than single precision: rounding them
    # to float32 and back must change something. Before the fix, mu and
    # Theta were float32 already and this held trivially false.
    mu64, th64 = out64[1], out64[2]
    assert not torch.equal(mu64, mu64.float().double())
    assert not torch.equal(th64, th64.float().double())


def test_float64_multipole_gradients_pass_gradcheck(model32):
    """Finite differences agree only if the whole graph is float64."""
    model64 = copy.deepcopy(model32).double()
    batch = _batch(torch.float64)
    R0 = batch.R.clone()

    def f(R):
        batch.R = R
        q, mu, th, _ = model64(batch)
        return q, mu, th

    R = R0.clone().requires_grad_(True)
    assert torch.autograd.gradcheck(f, (R,), eps=1e-6, atol=1e-6)
