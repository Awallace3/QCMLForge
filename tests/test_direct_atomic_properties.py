"""Synthetic invariants for energy-trained, checkpoint-independent properties."""

from types import SimpleNamespace

import pytest
import torch
from apnet_pt.AtomModels.direct_atomic_properties import DirectAtomicProperties


def graph(charge=0.0):
    return SimpleNamespace(
        x=torch.tensor([8, 1, 1, 11]),
        R=torch.tensor(
            [[0.0, 0.0, 0.0], [0.8, 0.5, 0.0], [-0.8, 0.5, 0.0], [5.0, 0.0, 0.0]],
            dtype=torch.float64,
        ),
        edge_index=torch.tensor([[0, 1, 0, 2, 1, 2], [1, 0, 2, 0, 2, 1]]),
        molecule_ind=torch.tensor([0, 0, 0, 1]),
        total_charge=torch.tensor([charge, 1.0], dtype=torch.float64),
    )


def model():
    torch.manual_seed(42)
    return DirectAtomicProperties(n_message=2, n_neuron=16, n_embed=4, n_rbf=4).double()


def test_conservation_positive_hfvr_and_nonzero_neutral_start():
    m, g = model(), graph()
    p = m(g)
    sums = p["q"].new_zeros(2).index_add(0, g.molecule_ind, p["q"])
    torch.testing.assert_close(sums, g.total_charge, atol=1e-14, rtol=0)
    assert torch.count_nonzero(p["q"][:3]) > 0
    assert torch.all((p["hfvr"] > m.hfvr_min) & (p["hfvr"] < m.hfvr_max))
    torch.testing.assert_close(p["quad"], p["quad"].transpose(-1, -2))
    torch.testing.assert_close(
        p["quad"].diagonal(dim1=-2, dim2=-1).sum(-1),
        torch.zeros(4, dtype=torch.float64),
        atol=1e-14,
        rtol=0,
    )
    assert p["mu"][3].count_nonzero() == p["quad"][3].count_nonzero() == 0


def test_rotation_translation_and_permutation():
    m, g = model(), graph()
    p = m(g)
    rot = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64
    )
    moved = SimpleNamespace(**{**vars(g), "R": g.R @ rot.T + 3.0})
    pp = m(moved)
    for key in ("q", "hfvr"):
        torch.testing.assert_close(pp[key], p[key])
    torch.testing.assert_close(pp["mu"], p["mu"] @ rot.T)
    torch.testing.assert_close(pp["quad"], rot @ p["quad"] @ rot.T)
    perm = torch.tensor([3, 2, 0, 1])
    inv = torch.argsort(perm)
    shuffled = SimpleNamespace(
        **{
            **vars(g),
            "x": g.x[perm],
            "R": g.R[perm],
            "molecule_ind": g.molecule_ind[perm],
            "edge_index": inv[g.edge_index],
        }
    )
    for key, value in m(shuffled).items():
        torch.testing.assert_close(value, p[key][perm])


def test_charge_conditions_distribution_and_hfvr_not_only_sum():
    m = model()
    neutral, charged = m(graph()), m(graph(1.0))
    assert not torch.allclose(charged["hfvr"][:3], neutral["hfvr"][:3])
    assert not torch.allclose(charged["q"][:3] - 1 / 3, neutral["q"][:3])


def test_every_property_head_has_finite_gradients():
    m = model()
    p = m(graph())
    loss = sum(v.square().sum() for v in p.values())
    loss.backward()
    for head in (m.scalar_head, m.edge_head):
        assert any(
            x.grad is not None and x.grad.abs().max() > 0 for x in head.parameters()
        )
    assert all(
        torch.isfinite(x.grad).all() for x in m.parameters() if x.grad is not None
    )


def test_strong_cation_polarizability_suppression_is_representable():
    m = model()
    with torch.no_grad():
        m.scalar_head[-1].bias[1] = -7.
    p = m(graph())
    # The charged monatomic Na can reach <0.1% of free-atom alpha.
    # An arbitrary HFVR floor of 0.1 would forbid this regime.
    assert p["hfvr"][-1] ** (4 / 3) < 1e-3


def test_property_forward_compiles_without_graph_breaks():
    m, g = model(), graph()
    compiled = torch.compile(m, backend="eager", fullgraph=True)
    for key, value in compiled(g).items():
        torch.testing.assert_close(value, m(g)[key])


def test_invalid_hfvr_bounds():
    with pytest.raises(ValueError, match="HFVR"):
        DirectAtomicProperties(hfvr_min=1.0, hfvr_max=0.5)
