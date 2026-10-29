"""``featurize_monomer``: the AP3-D3 intramonomer MPNN as its own step.

The pair model's per-monomer message passing is callable on its own so that
its atom features can be shared with another model (e.g. a classical-term
trunk) ahead of the pairwise corrections.  These tests pin that the split
did not change the forward, and that the features of several monomers can
be computed in one call.
"""

import pytest
import torch

from apnet_pt.util import scatter_sum_compile
from test_ap3_d3_fused_cutoff import _batch, _model


def _reference_states(model, Z, R, e_source, e_target):
    """The intramonomer loop as written inline in forward before the split."""
    dR, dR_xyz = model.get_distances(R, R, e_source, e_target)
    dR_unit = dR_xyz / dR.unsqueeze(1)
    rbf = model.distance_layer(dR)
    h_list = [model.embed_layer(Z).view(Z.size(0), -1)]
    h_dir_list = []
    for i in range(model.n_message):
        m_ij = model.get_messages(h_list[0], h_list[-1], rbf, e_source, e_target)
        m_i = scatter_sum_compile(m_ij, e_source, int(Z.size(0)))
        h_list.append(model.update_layers[i](m_i))
        m_ij_dir = model.directional_layers[i](m_ij)
        m_ij_dir = torch.einsum("ex,em->exm", dR_unit, m_ij_dir)
        h_dir_list.append(scatter_sum_compile(m_ij_dir, e_source, int(Z.size(0))))
    return torch.cat(h_list, dim=-1), torch.cat(h_dir_list, dim=-1)


@pytest.fixture(scope="module")
def ap3d3_and_batch():
    ap3d3 = _model(4.0, sr_switch=True)
    return ap3d3, _batch(ap3d3, 8.0)


def test_featurize_monomer_matches_the_inline_loop(ap3d3_and_batch):
    ap3d3, batch = ap3d3_and_batch
    model = ap3d3.model
    with torch.no_grad():
        for Z, R, s, t in ((batch.ZA, batch.RA, batch.e_AA_source, batch.e_AA_target),
                           (batch.ZB, batch.RB, batch.e_BB_source, batch.e_BB_target)):
            h, h_dir = model.featurize_monomer(Z, R, s, t)
            h_ref, h_dir_ref = _reference_states(model, Z, R, s, t)
            assert h.shape == (Z.size(0), model.n_embed * (model.n_message + 1))
            assert h_dir.shape == (Z.size(0), 3, model.n_embed * model.n_message)
            assert torch.equal(h, h_ref)
            assert torch.equal(h_dir, h_dir_ref)


def test_forward_reads_featurize_monomer(ap3d3_and_batch, monkeypatch):
    """forward is unchanged with the reference loop swapped in, and the
    features really come from featurize_monomer (perturbing it moves E)."""
    ap3d3, batch = ap3d3_and_batch
    model = ap3d3.model
    with torch.no_grad():
        E = model(batch)[0]
        monkeypatch.setattr(model, "featurize_monomer",
                            lambda *a: _reference_states(model, *a))
        assert torch.equal(model(batch)[0], E)
        monkeypatch.setattr(model, "featurize_monomer",
                            lambda *a: tuple(1.01 * x for x in
                                             _reference_states(model, *a)))
        assert not torch.allclose(model(batch)[0], E)


def test_monomers_featurize_together(ap3d3_and_batch):
    """Messages stay within a monomer: A and B in one call equal two calls."""
    ap3d3, batch = ap3d3_and_batch
    model = ap3d3.model
    nA = batch.ZA.size(0)
    Z = torch.cat([batch.ZA, batch.ZB])
    R = torch.cat([batch.RA, batch.RB])
    s = torch.cat([batch.e_AA_source, batch.e_BB_source + nA])
    t = torch.cat([batch.e_AA_target, batch.e_BB_target + nA])
    with torch.no_grad():
        h, h_dir = model.featurize_monomer(Z, R, s, t)
        hA, hA_dir = model.featurize_monomer(batch.ZA, batch.RA, batch.e_AA_source,
                                             batch.e_AA_target)
        hB, hB_dir = model.featurize_monomer(batch.ZB, batch.RB, batch.e_BB_source,
                                             batch.e_BB_target)
    assert torch.allclose(h, torch.cat([hA, hB]), atol=1e-6)
    assert torch.allclose(h_dir, torch.cat([hA_dir, hB_dir]), atol=1e-6)
