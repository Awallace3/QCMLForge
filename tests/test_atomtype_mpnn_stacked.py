"""The stacked AtomTypeParamMPNN forward is the loop forward, batched over p."""

from types import SimpleNamespace

import pytest
import torch

from apnet_pt.AtomModels.ap3_atomtype_mpnn import AtomTypeParamMPNN


def graph(seed, n_mol=4, dtype=torch.float64, isolated=True):
    g = torch.Generator().manual_seed(seed)
    zs, rs, mol, edges, offset = [], [], [], [], 0
    for m in range(n_mol):
        n = int(torch.randint(2, 7, (1,), generator=g))
        zs.append(torch.tensor([1, 6, 7, 8, 16])[torch.randint(0, 5, (n,), generator=g)])
        R = 1.4 * torch.randn(n, 3, generator=g, dtype=dtype)
        rs.append(R)
        mol.append(torch.full((n,), m))
        d = torch.cdist(R, R)
        i, j = torch.nonzero((d > 0) & (d < 5.0), as_tuple=True)
        edges.append(torch.stack([i, j]) + offset)
        offset += n
    if isolated:  # an edgeless atom takes the keep-mask branch
        zs.append(torch.tensor([11]))
        rs.append(torch.full((1, 3), 40.0, dtype=dtype))
        mol.append(torch.tensor([n_mol]))
    return SimpleNamespace(x=torch.cat(zs), R=torch.cat(rs),
                           edge_index=(torch.cat(edges, 1) if edges else
                                       torch.zeros((2, 0), dtype=torch.long)),
                           molecule_ind=torch.cat(mol))


def model(n_params, dtype=torch.float64, seed=0):
    torch.manual_seed(seed)
    m = AtomTypeParamMPNN(n_message=3, n_rbf=8, n_neuron=16, n_embed=4,
                          n_params=n_params, param_start_mean=[0.3] * n_params,
                          param_start_std=[0.05] * n_params).to(dtype)
    with torch.no_grad():  # nonzero readouts so every layer matters
        for p in m.parameters():
            p.add_(0.05 * torch.randn_like(p))
    return m


@pytest.mark.parametrize("n_params", [1, 2, 16])
@pytest.mark.parametrize("dtype,rtol", [(torch.float64, 1e-12), (torch.float32, 1e-5)])
def test_stacked_forward_and_gradients_match_the_loop(n_params, dtype, rtol):
    m = model(n_params, dtype)
    batch = graph(1, dtype=dtype)
    loop = m(batch)
    m.stacked_forward = True
    stacked = m(batch)
    assert stacked.shape == loop.shape
    torch.testing.assert_close(stacked, loop, rtol=rtol, atol=rtol * float(loop.detach().abs().max()))
    weights = torch.randn_like(loop)
    grads = {}
    for flag in (False, True):
        m.stacked_forward = flag
        m.zero_grad()
        (m(batch) * weights).sum().backward()
        grads[flag] = {k: p.grad.clone() for k, p in m.named_parameters()
                       if p.grad is not None}
    assert grads[False].keys() == grads[True].keys()
    for k in grads[False]:
        scale = float(grads[False][k].abs().max().clamp(min=1e-30))
        torch.testing.assert_close(grads[True][k], grads[False][k], rtol=rtol,
                                   atol=rtol * scale)


def test_stacked_forward_leaves_state_dict_and_edgeless_batch_alone():
    m = model(3)
    keys = list(m.state_dict())
    m.stacked_forward = True
    assert list(m.state_dict()) == keys
    edgeless = graph(2, n_mol=0)
    edgeless.edge_index = torch.zeros((2, 0), dtype=torch.long)
    out = m(edgeless)
    m.stacked_forward = False
    torch.testing.assert_close(out, m(edgeless))
