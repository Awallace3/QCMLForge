"""FF-3 frozen-backbone arms C/D (apnet_pt.ff2.so3lr_backbone).

The cached SO3LR-T-L features are stood in for by a synthetic network with
T-L's layout and transformation law: ``x`` (256) invariant, ``ev`` (48) with
degrees ``[1, 1, 2, 2, 3, 3, 4, 4]`` built from bond harmonics in so3lr's
real-SH convention (``real_sph`` = so3lr ``spherical_basis.py`` on unit
vectors).  The equivariance tests recompute these features on the rotated
geometry, so they cover the whole C/D path including the SH -> Cartesian map.
The real T-L features are checked the same way against so3lr itself in
qcmlforge-exp ``scripts/ff3_so3lr/tl_cache_parity.py``."""

import math

import pytest
import torch
from torch._dynamo.utils import counters

from apnet_pt.ff1.batching import batch_from_store as ff1_batch_from_store
from apnet_pt.ff1.trunk import f_cut
from apnet_pt.ff2 import FF2Model, compile_ff2
from apnet_pt.ff2.batching import batch_from_dimers, with_coordinates
from apnet_pt.ff2.so3lr_backbone import (
    BACKBONES,
    TL_DEGREES,
    TL_X_DIM,
    SO3LRFeatureTrunk,
    degree_blocks,
    ev_cartesian,
    l1_to_cartesian,
    l2_to_cartesian,
)
from apnet_pt.ff2.trunk import real_sph

from .test_ff2 import dimers, grads, random_orthogonal
from .test_ff3_backbone import edge_case_dimers

F64 = torch.float64
SPH_SLICES = {1: (0, 3), 2: (3, 8), 3: (8, 15), 4: (15, 24)}


def fake_tl_features(b, seed=0):
    """T-L-shaped features of a batch: ``ev`` block (l, copy k) is
    ``sum_j w_lk(r_ij) Y_l(r_ij)`` over the intramonomer edges, ``x`` a fixed
    random map of invariant radial sums and of every block's norm."""
    g = torch.Generator().manual_seed(seed)
    src, dst = b["edges"]
    d = b["R"][dst] - b["R"][src]
    r = d.norm(dim=1)
    unit = d / r.clamp_min(1e-12)[:, None]  # coincident atoms: Y_l(0) = 0
    ylm = real_sph(unit)
    n = len(b["z"])
    centers = torch.linspace(0.8, 4.5, 8, dtype=F64)
    rbf = torch.exp(-((r[:, None] - centers) ** 2)) * f_cut(r, 5.0)[:, None]
    blocks = []
    for k, deg in enumerate(TL_DEGREES):
        w = rbf @ torch.randn(8, generator=g, dtype=F64)
        a, c = SPH_SLICES[deg]
        blocks.append(torch.zeros(n, c - a, dtype=F64).index_add(
            0, src, w[:, None] * ylm[:, a:c]))
    ev = torch.cat(blocks, 1)
    radial = torch.zeros(n, 8, dtype=F64).index_add(0, src, rbf)
    norms = torch.stack([ev[:, a:c].norm(dim=1) for _, a, c in degree_blocks()], 1)
    elem = torch.nn.functional.one_hot(b["z"].long(), 20).to(F64)
    inv = torch.cat([radial, norms, elem], 1)
    x = torch.tanh(inv @ torch.randn(inv.shape[1], TL_X_DIM, generator=g, dtype=F64))
    return x, ev


def with_features(b, seed=0):
    x, ev = fake_tl_features(b, seed)
    return {**b, "so3lr_x": x, "so3lr_ev": ev}


def make_model(backbone, seed=0, scale=0.0):
    torch.manual_seed(seed)
    model = FF2Model(n_feat=32, n_chan=4, n_layers=2, n_rbf=8, trunk_dtype=F64,
                     physics_dtype=F64, sr_shell=False, transverse_invariants=False,
                     alpha_anisotropy=False, so3lr_backbone=backbone)
    if scale:
        g = torch.Generator().manual_seed(seed + 1)
        with torch.no_grad():
            for p in model.parameters():
                p.add_(scale * torch.randn(p.shape, generator=g, dtype=p.dtype))
    return model


def trunk_out(model, b):
    return model.trunk(b["z"].long(), b["R"], b["edges"], b["so3lr_x"],
                       b["so3lr_ev"])[:3]


# --- the fixed real-SH -> Cartesian maps ---------------------------------------

def test_cartesian_maps_invert_so3lr_harmonics():
    """P1 Y_1(r) = r and P2 Y_2(r) = r r^T - I/3 for unit r: the constants,
    the m order (y, z, x for l = 1) and the signs are so3lr's."""
    g = torch.Generator().manual_seed(4)
    r = torch.randn(64, 3, generator=g, dtype=F64)
    r = r / r.norm(dim=1, keepdim=True)
    y = real_sph(r)
    assert torch.allclose(l1_to_cartesian(y[:, 0:3]), r, atol=1e-14)
    want = r[:, :, None] * r[:, None, :] - torch.eye(3, dtype=F64) / 3
    T = l2_to_cartesian(y[:, 3:8])
    assert torch.allclose(T, want, atol=1e-14)
    assert torch.allclose(T, T.transpose(1, 2), atol=0)
    assert float(torch.diagonal(T, dim1=1, dim2=2).sum(-1).abs().max()) < 1e-15


def test_ev_layout_is_tl():
    assert [b[0] for b in degree_blocks()] == list(TL_DEGREES)
    assert degree_blocks()[-1][2] == 48
    U, T = ev_cartesian(torch.zeros(5, 48, dtype=F64))
    assert U.shape == (5, 3, 2) and T.shape == (5, 3, 3, 2)


# --- equivariance of the full C/D path -----------------------------------------

@pytest.mark.parametrize("backbone", BACKBONES)
def test_trunk_and_energies_are_o3_equivariant(backbone):
    """Random O(3) (det -1) + translation, features recomputed on the moved
    geometry: h invariant, u -> R u, Q -> R Q R^T, energies invariant."""
    model = make_model(backbone, seed=1, scale=0.3)
    b = with_features(batch_from_dimers(dimers() + edge_case_dimers()))
    O = random_orthogonal(-1.0, seed=7)
    t = torch.tensor([0.7, -1.2, 2.1], dtype=F64)
    bt = with_features(with_coordinates(b, b["R"] @ O.T + t))
    h0, u0, Q0 = trunk_out(model, b)
    h1, u1, Q1 = trunk_out(model, bt)
    assert float(u0.abs().max()) > 1e-3 and float(Q0.abs().max()) > 1e-3
    assert torch.allclose(h1, h0, atol=1e-10)
    assert torch.allclose(u1, torch.einsum("ab,nbc->nac", O, u0), atol=1e-10)
    assert torch.allclose(Q1, torch.einsum("ab,nbdc,ed->naec", O, Q0, O), atol=1e-10)
    tr = torch.diagonal(Q0, dim1=1, dim2=2).sum(-1)
    assert float(tr.abs().max()) < 1e-12
    p0, _ = model(b)
    p1, _ = model(bt)
    assert torch.allclose(p0, p1, rtol=0, atol=1e-8), (p0 - p1).abs().max()


def test_linear_arm_channels_span_two_dimensions():
    """C's limitation, stated in the proposal: every u channel of an atom lies
    in the plane of its two l = 1 blocks; D's adapter lifts it."""
    b = with_features(batch_from_dimers(dimers()))
    for backbone, planar in (("tl-linear", True), ("tl-adapter", False)):
        _, u, _ = trunk_out(make_model(backbone, seed=2, scale=0.3), b)
        sv = torch.linalg.svdvals(u)  # (N, 3) per atom over 4 channels
        rel = sv[:, 2] / sv[:, 0].clamp_min(1e-300)
        busy = sv[:, 0] > 1e-6
        if planar:
            assert float(rel[busy].max()) < 1e-10
        else:
            assert float(rel[busy].max()) > 1e-3


# --- wiring --------------------------------------------------------------------

def test_default_model_has_no_so3lr_backbone():
    torch.manual_seed(0)
    model = FF2Model(n_feat=32, n_chan=4, n_layers=2, n_rbf=8)
    assert model.backbone()["so3lr_backbone"] == ""
    assert not any(k.startswith(("trunk.h_map", "trunk.u_gate")) for k in model.state_dict())


@pytest.mark.parametrize("backbone,n_layers", [("tl-linear", 0), ("tl-adapter", 1)])
def test_backbone_replaces_the_trunk(backbone, n_layers):
    model = make_model(backbone)
    assert isinstance(model.trunk, SO3LRFeatureTrunk)
    assert len(model.trunk.layers) == n_layers
    assert model.config()["so3lr_backbone"] == backbone
    assert model.identity_init_keys() == []
    assert not any(k.startswith("trunk.embed") for k in model.state_dict())


def test_backbone_rejects_other_trunk_options():
    with pytest.raises(ValueError):
        make_model("tl-adapter").__class__(n_feat=32, n_chan=4, n_rbf=8,
                                           so3lr_backbone="tl-adapter",
                                           charge_embedding=True)
    with pytest.raises(ValueError):
        FF2Model(n_feat=32, n_chan=4, n_rbf=8, so3lr_backbone="tl-huge")


@pytest.mark.parametrize("backbone", BACKBONES)
def test_every_trunk_parameter_gets_a_gradient(backbone):
    """Heads perturbed off FF-1's zero-readout init (at that init no head
    reads h yet, for C exactly as for FF-1's own embedding)."""
    model = make_model(backbone, seed=3, scale=0.1)
    pred, _ = model(with_features(batch_from_dimers(dimers() + edge_case_dimers())))
    g = grads(model, (pred * torch.arange(1, 5, dtype=F64)).sum())
    names = [n for n in g if n.startswith("trunk.")]
    dead = [n for n in names if g[n] is None or float(g[n].abs().max()) == 0.0]
    assert names and not dead, dead


def test_monomer_forward_reads_the_features():
    """The MBIS-stage forward (monomers only) uses the same features."""
    model = make_model("tl-adapter", seed=5, scale=0.2)
    b = with_features(batch_from_dimers(dimers()))
    _, a0 = model.forward_monomers(b)
    b2 = {**b, "so3lr_x": b["so3lr_x"] * 1.1}
    _, a1 = model.forward_monomers(b2)
    assert float((a1["q"] - a0["q"]).abs().max()) > 1e-8


def test_store_batch_passes_cached_features_through():
    from types import SimpleNamespace

    n = 5
    graph = SimpleNamespace(edge_index=torch.zeros(2, 0, dtype=torch.long),
                            molecule_ind=torch.tensor([0, 0, 1, 1, 1]),
                            total_charge=torch.zeros(2, dtype=F64))
    atoms = {"z": torch.ones(n, dtype=torch.long), "R": torch.zeros(n, 3, dtype=F64),
             "so3lr_x": torch.randn(n, 256), "so3lr_ev": torch.randn(n, 48)}
    b = ff1_batch_from_store({"graph": graph, "atoms": atoms, "ai": None, "aj": None,
                              "pair_dimer": None, "n": 1})
    assert b["so3lr_x"] is atoms["so3lr_x"] and b["so3lr_ev"] is atoms["so3lr_ev"]
    del atoms["so3lr_x"], atoms["so3lr_ev"]
    b = ff1_batch_from_store({"graph": graph, "atoms": atoms, "ai": None, "aj": None,
                              "pair_dimer": None, "n": 1})
    assert "so3lr_x" not in b


# --- torch.compile -------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize("backbone", BACKBONES)
def test_backbone_compiles_to_one_graph(backbone):
    torch._dynamo.reset()
    counters.clear()
    model = make_model(backbone, seed=6, scale=0.05)
    compiled = compile_ff2(model)
    for batch in (batch_from_dimers(dimers() + edge_case_dimers()[:1]),
                  batch_from_dimers(dimers()[:3]),
                  batch_from_dimers(dimers()[1:2])):
        batch = with_features(batch)
        pe, _ = model(batch)
        pc, _ = compiled(batch)
        assert torch.allclose(pe, pc, rtol=0, atol=1e-6), (pe - pc).abs().max()
        gc = grads(model, pc.sum())
        assert all(v is None or torch.isfinite(v).all() for v in gc.values())
    assert counters["stats"]["unique_graphs"] == 1
    assert not counters["graph_break"], dict(counters["graph_break"])
    assert sum(counters["recompiles"].values()) == 0


def test_constants_are_the_proposal_values():
    from apnet_pt.ff2 import so3lr_backbone as S

    assert S.N1 == math.sqrt(3 / (4 * math.pi))
    assert S.C2 == 0.5 * math.sqrt(15 / math.pi)
    assert S.D2 == 0.25 * math.sqrt(5 / math.pi)
    assert S.E2 == 0.25 * math.sqrt(15 / math.pi)
