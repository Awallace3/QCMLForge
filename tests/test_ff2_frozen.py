"""Frozen monomer parameters (apnet_pt.ff2.frozen) in the FF-2 long-range engine."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from apnet_pt.ff1 import FF1Model
from apnet_pt.ff2 import FF2Model
from apnet_pt.ff2.frozen import ORIENTED, SCALARS, FrozenAtoms
from apnet_pt.ff2.md import cluster_batch
from apnet_pt.ff2.pme import FF2LongRangeEnergy

F64 = torch.float64
WATER = [[0.0, 0.0, 0.0], [0.7586, 0.0, 0.5043], [-0.7586, 0.0, 0.5043]]
NH3 = [[0.0, 0.0, 0.0], [0.94, 0.0, -0.38], [-0.47, 0.81, -0.38], [-0.47, -0.81, -0.38]]


def small_ff2(seed=0, scale=0.10):
    torch.manual_seed(seed)
    ff1 = FF1Model(n_feat=32, n_chan=4, n_layers=2, n_rbf=8, trunk_dtype=F64,
                   physics_dtype=F64, g0_init=-2.5, elst_beta_def=True,
                   elst_sr_shell=True)
    with torch.no_grad():
        for p in ff1.parameters():
            p.add_(scale * torch.randn_like(p))
    return FF2Model.from_ff1(ff1).eval()


def bonds_of(R, mon, r_max=1.2):
    D = np.linalg.norm(R[:, None] - R[None], axis=-1)
    i, j = np.nonzero(np.triu((D < r_max) & (mon[:, None] == mon[None]), 1))
    return np.stack([i, j], 1)


def cluster():
    z = [8, 1, 1, 7, 1, 1, 1, 8, 1, 1]
    mon = [0, 0, 0, 1, 1, 1, 1, 2, 2, 2]
    R = np.array(WATER + [[x + 0.4, y + 0.5, w + 3.2] for x, y, w in NH3]
                 + [[x - 2.5, y + 2.2, w + 1.8] for x, y, w in WATER])
    return z, mon, [0.0, 0.0, 0.0], R


def rotation(seed):
    q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((3, 3)))
    return q * np.sign(np.linalg.det(q))


def rigid_moves(R, mon, seed=3, step=0.3):
    """Rotate each monomer about its centroid and shift it: internal geometry kept."""
    R2, mon = R.copy(), np.asarray(mon)
    rng = np.random.default_rng(seed)
    for k in np.unique(mon):
        s = mon == k
        c = R[s].mean(0)
        R2[s] = (R[s] - c) @ rotation(seed + k).T + c + step * rng.standard_normal(3)
    return R2


def live_atoms(model, z, mon, q, R):
    b = cluster_batch(z, mon, q)
    return model._forward_atoms({"z": b["z"], "R": torch.as_tensor(R, dtype=F64),
                                 "edges": b["edges"], "monomer": b["monomer"],
                                 "monomer_charge": b["monomer_charge"]})[1]


def test_frozen_atoms_reproduce_live_parameters_under_rigid_motion():
    model = small_ff2(seed=1)
    z, mon, q, R = cluster()
    fa = FrozenAtoms(model, z, mon, q, R, bonds_of(R, np.asarray(mon)))
    for X in (R, rigid_moves(R, mon)):
        live = live_atoms(model, z, mon, q, X)
        _, frozen = fa(torch.as_tensor(X, dtype=F64))
        for k in (*SCALARS, *ORIENTED):
            torch.testing.assert_close(frozen[k], live[k].to(F64), rtol=0, atol=1e-10,
                                       msg=k)


def test_frozen_atoms_ignore_internal_distortion():
    model = small_ff2(seed=1)
    z, mon, q, R = cluster()
    fa = FrozenAtoms(model, z, mon, q, R, bonds_of(R, np.asarray(mon)))
    X = R.copy()
    X[1] += [0.25, 0.0, 0.1]                                  # stretch one O-H
    _, frozen = fa(torch.as_tensor(X, dtype=F64))
    _, ref = fa(torch.as_tensor(R, dtype=F64))
    for k in SCALARS:
        assert torch.equal(frozen[k], ref[k]), k
    assert not torch.allclose(live_atoms(model, z, mon, q, X)["q"].to(F64), ref["q"])


def test_frozen_long_range_cluster_matches_live_on_rigid_moves_and_fd_forces():
    model = small_ff2(seed=1)
    z, mon, q, R = cluster()
    kw = dict(r_cut=50.0, skin=1.0, pbc=False, cutoff_switch="none", pol_cg_iters=25)
    live = FF2LongRangeEnergy(model, z, mon, q, **kw)
    fa = FrozenAtoms(model, z, mon, q, R, bonds_of(R, np.asarray(mon)))
    frozen = FF2LongRangeEnergy(model, z, mon, q, atom_params=fa, **kw)
    X = torch.as_tensor(rigid_moves(R, mon), dtype=F64) / 10.0
    e_l, _ = live.energy_forces(X)
    e_f, f_f = frozen.energy_forces(X)
    assert abs(float(e_l - e_f)) < 1e-9
    h = 1e-6
    for a, c in ((1, 0), (4, 2), (8, 1)):
        Xp, Xm = X.clone(), X.clone()
        Xp[a, c] += h
        Xm[a, c] -= h
        fd = -(frozen.energy_forces(Xp)[0] - frozen.energy_forces(Xm)[0]) / (2 * h)
        assert abs(float(fd - f_f[a, c])) < 1e-5 * max(1.0, abs(float(fd)))


def periodic_box():
    box = np.array([[1.6, 0.0, 0.0], [0.0, 1.55, 0.0], [0.0, 0.0, 1.65]])
    z, mon, R = [], [], []
    for k, (ix, iy, iz) in enumerate(np.ndindex(2, 2, 2)):
        base = np.array([ix * 7.8 + 1.5, iy * 7.5 + 1.5, iz * 8.0 + 1.5])
        geo, zz = (WATER, [8, 1, 1]) if k % 2 == 0 else (NH3, [7, 1, 1, 1])
        R += [base + np.asarray(p) for p in geo]
        z += zz
        mon += [k] * len(zz)
    return z, mon, [0.0] * 8, np.array(R), box


def test_frozen_long_range_periodic_matches_live_and_survives_wrapping():
    model = small_ff2(seed=0)
    z, mon, q, R, box = periodic_box()
    kw = dict(r_cut=12.0, skin=1.0, cell=torch.as_tensor(box, dtype=F64),
              elst_alpha=0.19, elst_mesh=(32, 32, 32), elst_order=6,
              disp_mode="pme_lowrank", disp_rank=4, disp_alpha=0.16,
              disp_mesh=(24, 24, 24), disp_order=4, pol_cg_iters=25)
    live = FF2LongRangeEnergy(model, z, mon, q, **kw)
    fa = FrozenAtoms(model, z, mon, q, R, bonds_of(R, np.asarray(mon)))
    frozen = FF2LongRangeEnergy(model, z, mon, q, atom_params=fa, **kw)
    cell = torch.as_tensor(box, dtype=F64)
    X = torch.as_tensor(rigid_moves(R, mon, step=0.2), dtype=F64) / 10.0
    e_l, _ = live.energy_forces(X, cell)
    e_f, f_f = frozen.energy_forces(X, cell)
    assert abs(float(e_l - e_f)) < 1e-8
    # wrap one hydrogen across the box: the energy must not change
    Xw = X.clone()
    Xw[1] -= cell[0]
    e_w, f_w = frozen.energy_forces(Xw, cell)
    assert abs(float(e_w - e_f)) < 1e-8
    assert float((f_w - f_f).abs().max()) < 1e-7
    assert math.isfinite(float(e_f))


def test_frozen_atoms_reject_cross_monomer_bonds():
    model = small_ff2(seed=1)
    z, mon, q, R = cluster()
    with pytest.raises(ValueError, match="joins two monomers"):
        FrozenAtoms(model, z, mon, q, R, np.array([[0, 3]]))
