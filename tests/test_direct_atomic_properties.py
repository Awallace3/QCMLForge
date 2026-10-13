"""Synthetic invariants for energy-trained, checkpoint-independent properties."""

from types import SimpleNamespace

import pytest
import torch

from apnet_pt import constants
from apnet_pt.AtomModels.direct_atomic_properties import DirectAtomicProperties

MODES = ("log", "linear")


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


def model(mode="log"):
    torch.manual_seed(42)
    return DirectAtomicProperties(
        n_message=2, n_neuron=16, n_embed=4, n_rbf=4, alpha_mode=mode
    ).double()


@pytest.mark.parametrize("mode", MODES)
def test_conservation_positive_alpha_and_nonzero_neutral_start(mode):
    m, g = model(mode), graph()
    p = m(g)
    sums = p["q"].new_zeros(2).index_add(0, g.molecule_ind, p["q"])
    torch.testing.assert_close(sums, g.total_charge, atol=1e-14, rtol=0)
    assert torch.count_nonzero(p["q"][:3]) > 0
    assert torch.all(p["alpha"] > 0)
    torch.testing.assert_close(p["quad"], p["quad"].transpose(-1, -2))
    torch.testing.assert_close(
        p["quad"].diagonal(dim1=-2, dim2=-1).sum(-1),
        torch.zeros(4, dtype=torch.float64),
        atol=1e-14,
        rtol=0,
    )
    assert p["mu"][3].count_nonzero() == p["quad"][3].count_nonzero() == 0


@pytest.mark.parametrize("mode", MODES)
def test_alpha_starts_at_neutral_free_atom_value(mode):
    m, g = model(mode), graph()
    free = constants.polarizability_table.to(torch.float64)[g.x]
    # Only the small readout perturbation separates the start from alpha_free;
    # the Na cation also starts at the neutral-atom value (163.08 bohr^3).
    torch.testing.assert_close(m(g)["alpha"], free, rtol=0.02, atol=0)
    assert abs(float(free[-1]) - 163.08) < 0.01


@pytest.mark.parametrize("mode", MODES)
def test_rotation_translation_and_permutation(mode):
    m, g = model(mode), graph()
    p = m(g)
    rot = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64
    )
    moved = SimpleNamespace(**{**vars(g), "R": g.R @ rot.T + 3.0})
    pp = m(moved)
    for key in ("q", "alpha"):
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


@pytest.mark.parametrize("mode", MODES)
def test_charge_conditions_distribution_and_alpha_not_only_sum(mode):
    m = model(mode)
    neutral, charged = m(graph()), m(graph(1.0))
    assert not torch.equal(charged["alpha"][:3], neutral["alpha"][:3])
    assert not torch.allclose(charged["q"][:3] - 1 / 3, neutral["q"][:3])


@pytest.mark.parametrize("mode", MODES)
def test_every_property_head_has_finite_gradients(mode):
    m = model(mode)
    p = m(graph())
    loss = sum(v.square().sum() for v in p.values())
    loss.backward()
    for head in (m.scalar_head, m.edge_head, m.alpha_base):
        assert any(
            x.grad is not None and x.grad.abs().max() > 0 for x in head.parameters()
        )
    assert all(
        torch.isfinite(x.grad).all() for x in m.parameters() if x.grad is not None
    )


def test_strong_cation_polarizability_suppression_is_representable():
    m = model("log")
    with torch.no_grad():
        m.scalar_head[-1].bias[1] = -7.
    # The charged monatomic Na can reach <0.1% of free-atom alpha.
    assert m(graph())["alpha"][-1] < 1e-3 * 163.08


def test_linear_alpha_is_floored_and_additive():
    m = model("linear")
    with torch.no_grad():
        m.scalar_head[-1].bias[1] = -1000.
    assert torch.all(m(graph())["alpha"] == m.alpha_floor)
    with torch.no_grad():
        m.scalar_head[-1].bias[1] = 0.
        m.scalar_head[-1].weight.zero_()
    free = constants.polarizability_table.to(torch.float64)[graph().x]
    torch.testing.assert_close(m(graph())["alpha"], free, rtol=1e-6, atol=0)


@pytest.mark.parametrize("mode", MODES)
def test_property_forward_compiles_without_graph_breaks(mode):
    m, g = model(mode), graph()
    compiled = torch.compile(m, backend="eager", fullgraph=True)
    for key, value in compiled(g).items():
        torch.testing.assert_close(value, m(g)[key])


def test_invalid_alpha_options():
    with pytest.raises(ValueError, match="alpha_mode"):
        DirectAtomicProperties(alpha_mode="sqrt")
    with pytest.raises(ValueError, match="alpha_floor"):
        DirectAtomicProperties(alpha_mode="linear", alpha_floor=0.0)
