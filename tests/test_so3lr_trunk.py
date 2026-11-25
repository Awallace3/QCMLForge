"""Tests for pure-PyTorch SO3LRTrunk and fused T-S / T-M / T-L FF2Model backbones."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch._dynamo.utils import counters

from apnet_pt.ff2 import FF2Model, SO3LRTrunk, compile_ff2
from apnet_pt.ff2.batching import batch_from_dimers
from apnet_pt.ff2.frozen import FrozenAtoms
from apnet_pt.ff2.pme import FF2LongRangeEnergy
from apnet_pt.ff2.so3lr_backbone import BACKBONE_SPECS, BACKBONES
from apnet_pt.ff2.so3lr_trunk import SO3LR_PRESETS

F64 = torch.float64
SO3LR_MIRROR = Path.home() / "artifacts" / "qcmlforge" / "so3lr-tauri-models-2606"
PARITY_NPZ = SO3LR_MIRROR / "ref_parity_2606.npz"

WATER = [[0.0, 0.0, 0.0], [0.7586, 0.0, 0.5043], [-0.7586, 0.0, 0.5043]]
NH3 = [
    [0.0, 0.0, 0.0],
    [0.94, 0.0, -0.33],
    [-0.47, 0.81, -0.33],
    [-0.47, -0.81, -0.33],
]
ACETATE = (
    [6, 6, 8, 8, 1, 1, 1],
    [
        [0.0, 0.0, 0.0],
        [1.52, 0.0, 0.0],
        [2.15, 1.07, 0.0],
        [2.15, -1.07, 0.0],
        [-0.39, 1.02, 0.0],
        [-0.39, -0.51, 0.88],
        [-0.39, -0.51, -0.88],
    ],
)


def _shift(coords, dx=0.0, dy=0.0, dz=0.0):
    return [[x + dx, y + dy, z + dz] for x, y, z in coords]


def _sample_dimers():
    return [
        {"zA": [8, 1, 1], "RA": WATER, "zB": [7, 1, 1, 1], "RB": _shift(NH3, 0.3, 0.4, 3.0)},
        {
            "zA": ACETATE[0],
            "RA": ACETATE[1],
            "qA": -1,
            "zB": [8, 1, 1],
            "RB": _shift(WATER, 0.0, 0.0, 3.2),
        },
        {
            "zA": [8, 1, 1, 8, 1, 1],
            "RA": WATER + _shift(WATER, 2.9, 0.0, 0.0),
            "qA": 1,
            "zB": [7, 1, 1, 1],
            "RB": _shift(NH3, 0.0, 1.0, -3.1),
        },
    ]


def _edges_from_R(R: np.ndarray, cutoff: float = 5.0) -> torch.Tensor:
    d = np.linalg.norm(R[:, None, :] - R[None, :, :], axis=-1)
    src, dst = np.where((d > 0.0) & (d < cutoff))
    return torch.tensor(np.stack([src, dst], 0), dtype=torch.long)


def _random_rotation(seed: int = 7) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(3, 3, generator=g, dtype=F64))
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def _small_trunk(preset: str, seed: int = 0, dtype=F64) -> SO3LRTrunk:
    torch.manual_seed(seed)
    trunk = SO3LRTrunk.from_preset(preset).to(dtype)
    with torch.no_grad():
        for p in trunk.parameters():
            p.mul_(0.2)
    return trunk


# --- Exact JAX parity across all three 2606 flavors -------------------------


@pytest.mark.skipif(
    not PARITY_NPZ.is_file(),
    reason="Local SO3LR-Tauri 2606 parity fixture not present",
)
@pytest.mark.parametrize("model_name", tuple(SO3LR_PRESETS))
def test_so3lr_trunk_jax_parity_all_flavors(model_name):
    """Pure-PyTorch SO3LRTrunk matches JAX fp64 SO3LR-Tauri across S/M/L on
    neutral, charged, odd-electron, close-contact (<0.6 A), 2-fragment, and
    heavy-element monomers using a shared 5.0 A edge list."""
    ref = np.load(PARITY_NPZ)
    names = [str(n) for n in ref["names"]]

    for dtype, x_tol, ev_tol, q_tol in (
        (torch.float64, 5e-7, 1e-8, 1e-7),
        (torch.float32, 2e-4, 2e-5, 5e-5),
    ):
        trunk = SO3LRTrunk.from_so3lr_dir(SO3LR_MIRROR / model_name, dtype=dtype)
        zs, Rs, Qs, mons, edges_list = [], [], [], [], []
        off = 0
        for i in range(len(names)):
            z = ref[f"z_{i}"]
            R = ref[f"R_{i}"]
            Q = float(ref[f"Q_{i}"])
            edges_list.append(_edges_from_R(R, cutoff=5.0) + off)
            zs.append(torch.tensor(z, dtype=torch.long))
            Rs.append(torch.tensor(R, dtype=dtype))
            Qs.append(Q)
            mons.append(torch.full((len(z),), i, dtype=torch.long))
            off += len(z)
        with torch.no_grad():
            x, ev, q = trunk(
                torch.cat(zs),
                torch.cat(Rs),
                torch.cat(edges_list, 1),
                torch.cat(mons),
                torch.tensor(Qs, dtype=dtype),
            )
        rx = np.concatenate([ref[f"{model_name}_x_{i}"] for i in range(len(names))], 0)
        rev = np.concatenate([ref[f"{model_name}_ev_{i}"] for i in range(len(names))], 0)
        rq = np.concatenate([ref[f"{model_name}_q_{i}"] for i in range(len(names))], 0)
        assert float(np.abs(x.double().numpy() - rx).max()) < x_tol
        assert float(np.abs(ev.double().numpy() - rev).max()) < ev_tol
        assert float(np.abs(q.double().numpy() - rq).max()) < q_tol


# --- Symmetry, conservation, clash gradient safety --------------------------


@pytest.mark.parametrize("preset", ("ts", "tm", "tl"))
def test_so3lr_trunk_symmetry_charge_and_clash_safety(preset):
    trunk = _small_trunk(preset, seed=11, dtype=F64)
    b = batch_from_dimers(_sample_dimers()[:2])
    z = b["z"].long()
    R = b["R"].to(F64)
    edges = b["edges"]
    mon = b["monomer"]
    q_mon = b["monomer_charge"].to(F64)

    x1, ev1, q1 = trunk(z, R, edges, mon, q_mon)
    # Charge conservation per monomer
    q_tot = torch.zeros_like(q_mon).index_add(0, mon, q1)
    assert torch.allclose(q_tot, q_mon, atol=1e-12)

    # Rotation invariance of x, q and l=1 equivariance of ev
    rot = _random_rotation(5)
    R_rot = R @ rot.T
    x2, ev2, q2 = trunk(z, R_rot, edges, mon, q_mon)
    assert torch.allclose(x1, x2, atol=1e-11)
    assert torch.allclose(q1, q2, atol=1e-11)
    # l=1 block is (y, z, x) -> Cartesian (x, y, z)
    from apnet_pt.ff2.so3lr_backbone import ev_cartesian

    u1, t1 = ev_cartesian(ev1, trunk.degrees)
    u2, t2 = ev_cartesian(ev2, trunk.degrees)
    assert torch.allclose(torch.einsum("ab,nbk->nak", rot, u1), u2, atol=1e-11)
    assert torch.allclose(
        torch.einsum("ac,bd,ncdk->nabk", rot, rot, t1), t2, atol=1e-11
    )

    # Clash distance (below rmin floor and even exact overlap d = 0) gives finite grad
    R_clash = torch.tensor(
        [[0.0, 0.0, 0.0], [0.25, 0.0, 0.0], [0.25, 0.0, 0.0]],
        dtype=F64,
        requires_grad=True,
    )
    z_c = torch.tensor([6, 1, 1], dtype=torch.long)
    e_c = torch.tensor([[0, 1, 0, 2, 1, 2], [1, 0, 2, 0, 2, 1]], dtype=torch.long)
    m_c = torch.zeros(3, dtype=torch.long)
    q_c = torch.zeros(1, dtype=F64)
    xc, evc, qc = trunk(z_c, R_clash, e_c, m_c, q_c)
    loss = (xc * xc).sum() + (evc * evc).sum() + (qc * qc).sum()
    loss.backward()
    assert torch.isfinite(xc).all() and torch.isfinite(evc).all() and torch.isfinite(qc).all()
    assert torch.isfinite(R_clash.grad).all()


# --- Fused FF2Model: cached parity, compile, forces, FrozenAtoms ------------


@pytest.mark.parametrize("backbone", BACKBONES)
@pytest.mark.parametrize("charge_prior", ("bond_flow", "so3lr_qeq"))
def test_ff2_fused_vs_cached_parity_all_backbones(backbone, charge_prior):
    """Fused FF2Model (embed_so3lr_trunk=True and attach_so3lr_trunk) matches
    pre-cached so3lr_x/ev/q bit-for-bit across ts/tm/tl and bond_flow/so3lr_qeq."""
    torch.manual_seed(3)
    preset = BACKBONE_SPECS[backbone]["preset"]
    trunk = _small_trunk(preset, seed=19, dtype=F64)

    common = dict(
        n_feat=32,
        n_chan=4,
        n_rbf=8,
        trunk_dtype=F64,
        physics_dtype=F64,
        sr_shell=False,
        transverse_invariants=False,
        alpha_anisotropy=False,
        so3lr_backbone=backbone,
        pol_field="net_multipole",
        learnable_a_pol_field=True,
        covalent_bond_flow=True,
        charge_prior=charge_prior,
        rho_nu=0.0 if charge_prior == "so3lr_qeq" else 0.5,
    )
    m_cached = FF2Model(**common)
    m_fused = FF2Model(**common, embed_so3lr_trunk=True)
    # Copy heads/adapter from m_cached and trunk weights into m_fused
    missing, unexpected = m_fused.load_state_dict(m_cached.state_dict(), strict=False)
    assert not unexpected
    assert all(k.startswith("so3lr_trunk.") for k in missing)
    m_fused.attach_so3lr_trunk(trunk, freeze=True)

    # Also test attach_so3lr_trunk on a fresh cached model + config/state_dict roundtrip
    cfg = m_fused.config()
    assert cfg["embed_so3lr_trunk"] is True
    assert "embed_so3lr_trunk" not in m_cached.config()
    reload_cfg = {
        k: v for k, v in cfg.items()
        if k not in ("elst_beta_def", "elst_sr_shell", "trunk_dtype", "physics_dtype")
    }
    m_reload = FF2Model(
        n_feat=32, n_chan=4, n_rbf=8, trunk_dtype=F64, physics_dtype=F64, **reload_cfg
    )
    m_reload.load_state_dict(m_fused.state_dict(), strict=True)

    b = batch_from_dimers(_sample_dimers())
    with torch.no_grad():
        x, ev, q = trunk(
            b["z"].long(), b["R"].to(F64), b["edges"], b["monomer"], b["monomer_charge"]
        )
        p_cached, ex_cached = m_cached({**b, "so3lr_x": x, "so3lr_ev": ev, "so3lr_q": q})
        p_fused, ex_fused = m_fused(b)
        p_reload, _ = m_reload(b)

    assert torch.allclose(p_cached, p_fused, rtol=0, atol=1e-12)
    assert torch.allclose(ex_cached["q"], ex_fused["q"], rtol=0, atol=1e-12)
    assert torch.allclose(p_fused, p_reload, rtol=0, atol=1e-12)


@pytest.mark.parametrize("backbone", ("ts-adapter", "tm-adapter", "tl-adapter"))
def test_ff2_fused_compile_single_graph_and_forces(backbone):
    """Fused FF2Model (with so3lr_qeq + rho_nu=0) compiles to 1 graph across
    chunk shapes, yields exact autograd forces, and works with FrozenAtoms."""
    torch.manual_seed(29)
    model = FF2Model(
        n_feat=32,
        n_chan=4,
        n_rbf=8,
        trunk_dtype=F64,
        physics_dtype=F64,
        sr_shell=False,
        transverse_invariants=False,
        alpha_anisotropy=False,
        so3lr_backbone=backbone,
        embed_so3lr_trunk=True,
        pol_field="net_multipole",
        learnable_a_pol_field=True,
        covalent_bond_flow=True,
        charge_prior="so3lr_qeq",
        rho_nu=0.0,
    )
    with torch.no_grad():
        for p in model.so3lr_trunk.parameters():
            p.mul_(0.15)
        model.edge_head[-1].weight.fill_(0.05)

    ds = _sample_dimers()
    counters.clear()
    compiled = compile_ff2(model, backend="eager")
    for sub in (ds, ds[:2], ds[1:2]):
        b = batch_from_dimers(sub)
        pe, _ = model(b)
        pc, _ = compiled(b)
        assert torch.allclose(pe, pc, rtol=0, atol=1e-10)
    assert counters["stats"]["unique_graphs"] == 1

    # Autograd vs central finite differences through the live fused SO3LRTrunk
    b0 = batch_from_dimers(ds[:1])
    R_req = b0["R"].clone().detach().requires_grad_(True)
    pred, _ = model({**b0, "R": R_req})
    e0 = pred.sum()
    (g_ag,) = torch.autograd.grad(e0, (R_req,))
    eps = 1e-6
    g_fd = torch.zeros_like(g_ag)
    with torch.no_grad():
        for idx in range(R_req.shape[0]):
            for ax in range(3):
                Rp = b0["R"].clone()
                Rp[idx, ax] += eps
                Rm = b0["R"].clone()
                Rm[idx, ax] -= eps
                ep = model({**b0, "R": Rp})[0].sum()
                em = model({**b0, "R": Rm})[0].sum()
                g_fd[idx, ax] = (ep - em) / (2.0 * eps)
    assert torch.allclose(g_ag, g_fd, rtol=1e-4, atol=1e-4), (g_ag - g_fd).abs().max()

    # Standalone FrozenAtoms (no extra_batch!) matches FF2Model.forward
    model.eval()
    d0 = ds[0]
    z = list(d0["zA"]) + list(d0["zB"])
    na, nb = len(d0["zA"]), len(d0["zB"])
    mon = [0] * na + [1] * nb
    q = [0.0, 0.0]
    R = torch.tensor(list(d0["RA"]) + list(d0["RB"]), dtype=F64)
    bonds = [[0, 1], [0, 2], [3, 4], [3, 5], [3, 6]]
    fa = FrozenAtoms(model, z, mon, q, R.numpy(), bonds)
    eng = FF2LongRangeEnergy(
        model, z, mon, q, cell=None, cutoff_switch="none", r_cut=50.0, atom_params=fa
    )
    comp_pme = eng.components(R / 10.0, None)
    with torch.no_grad():
        comp_ref, _ = model(b0)
    assert torch.allclose(comp_pme, comp_ref[0], rtol=1e-7, atol=1e-8)
