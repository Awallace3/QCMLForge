"""Frozen SO3LR-Tauri features as FF-2's trunk (FF-3 arms C and D).

SO3LR-T-L-2606 is run once per monomer (outside torch, see qcmlforge-exp
``scripts/ff3_so3lr``) and its last-layer per-atom features are cached:

* ``x`` (N, 256): invariant features;
* ``ev`` (N, 48): Euclidean variables of degrees ``[1, 1, 2, 2, 3, 3, 4, 4]``,
  each block in so3lr's real spherical-harmonic order ``m = -l..l``
  (``sph_ops/spherical_basis.py``; ``l = 1`` is ``(y, z, x)``).

:class:`SO3LRFeatureTrunk` maps them onto FF-1's trunk contract
``(h (N, F), u (N, 3, C), Q (N, 3, 3, C) traceless, geo)``:

``h = W_h [x, kappa(ev)]``, ``u_c = G_rho(sum_k g^u_ck(x) P1 e^(1,k))``,
``Q_c = G_rho(sum_k g^Q_ck(x) P2 e^(2,k))`` with invariant MLPs ``g`` of
``x`` and the fixed real-SH -> Cartesian maps

``P1 e = (e_{+1}, e_{-1}, e_0) / N1``,
``P2 e: T_xy = e_{-2}/c, T_yz = e_{-1}/c, T_xz = e_{+1}/c, T_zz = e_0/(3d),
T_xx - T_yy = e_{+2}/e', tr T = 0``,
``N1 = sqrt(3/4pi), c = sqrt(15/pi)/2, d = sqrt(5/pi)/4, e' = sqrt(15/pi)/4``.

``kappa(ev)`` are the eight per-block invariants ``|e|^2 / sqrt(2l + 1)``.
Because T-L carries two copies of each degree, every ``u`` channel of one atom
lies in a 2-dimensional subspace of the l = 1 space and every ``Q`` channel in
a 2-dimensional subspace of the l = 2 space.  Arm D (``adapter``) therefore
adds one trainable :class:`~apnet_pt.ff1.trunk.FF1Layer` over the 5 A
intramonomer edges, which restores independent channels and FF-1's
tensor-product messages; arm C (``linear``) has no message passing.

``geo`` is FF-1's (cutoff, smoothed distances, unit vectors, 16 Gaussians), so
every head and the charge solve read the same geometry as FF-2.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from ..ff1.trunk import (
    FF1Layer,
    FF1MonomerTrunk,
    GaussianRBF,
    f_cut,
    mlp,
    norm_cap,
)

N1 = math.sqrt(3 / (4 * math.pi))
C2 = 0.5 * math.sqrt(15 / math.pi)
D2 = 0.25 * math.sqrt(5 / math.pi)
E2 = 0.25 * math.sqrt(15 / math.pi)
TS_DEGREES = (1, 2, 3, 4)
TS_X_DIM = 128
TM_DEGREES = (1, 2, 3, 4)
TM_X_DIM = 128
TL_DEGREES = (1, 1, 2, 2, 3, 3, 4, 4)
TL_X_DIM = 256
BACKBONE_SPECS = {
    "ts-linear": {"preset": "so3lr-tauri-s-2606", "adapter": False,
                  "x_dim": TS_X_DIM, "degrees": TS_DEGREES},
    "ts-adapter": {"preset": "so3lr-tauri-s-2606", "adapter": True,
                   "x_dim": TS_X_DIM, "degrees": TS_DEGREES},
    "tm-linear": {"preset": "so3lr-tauri-m-2606", "adapter": False,
                  "x_dim": TM_X_DIM, "degrees": TM_DEGREES},
    "tm-adapter": {"preset": "so3lr-tauri-m-2606", "adapter": True,
                   "x_dim": TM_X_DIM, "degrees": TM_DEGREES},
    "tl-linear": {"preset": "so3lr-tauri-l-2606", "adapter": False,
                  "x_dim": TL_X_DIM, "degrees": TL_DEGREES},
    "tl-adapter": {"preset": "so3lr-tauri-l-2606", "adapter": True,
                   "x_dim": TL_X_DIM, "degrees": TL_DEGREES},
}
BACKBONES = tuple(BACKBONE_SPECS)
FEATURE_KEYS = ("so3lr_x", "so3lr_ev")


def degree_blocks(degrees=TL_DEGREES) -> list[tuple[int, int, int]]:
    """``(l, start, stop)`` of each block of an ``ev`` row."""
    out, lo = [], 0
    for deg in degrees:
        out.append((deg, lo, lo + 2 * deg + 1))
        lo += 2 * deg + 1
    return out


def l1_to_cartesian(e: Tensor) -> Tensor:
    """(..., 3) l = 1 coefficients (m = -1, 0, 1) -> (..., 3) Cartesian."""
    return torch.stack([e[..., 2], e[..., 0], e[..., 1]], -1) / N1


def l2_to_cartesian(e: Tensor) -> Tensor:
    """(..., 5) l = 2 coefficients (m = -2..2) -> (..., 3, 3) traceless
    symmetric tensor ``T`` with ``e = Y_2(T)`` for ``T = sym0(r r^T)``."""
    xy, yz, xz = e[..., 0] / C2, e[..., 1] / C2, e[..., 3] / C2
    zz = e[..., 2] / (3 * D2)
    dxy = e[..., 4] / E2
    xx, yy = 0.5 * (dxy - zz), -0.5 * (dxy + zz)
    return torch.stack([torch.stack([xx, xy, xz], -1),
                        torch.stack([xy, yy, yz], -1),
                        torch.stack([xz, yz, zz], -1)], -2)


def ev_invariants(ev: Tensor, degrees=TL_DEGREES) -> Tensor:
    """(N, n_ev) -> (N, len(degrees)): ``|e|^2 / sqrt(2l + 1)`` per block."""
    return torch.stack([(ev[:, a:b] * ev[:, a:b]).sum(-1) / math.sqrt(b - a)
                        for _, a, b in degree_blocks(degrees)], -1)


def ev_cartesian(ev: Tensor, degrees=TL_DEGREES) -> tuple[Tensor, Tensor]:
    """``(U (N, 3, K1), T (N, 3, 3, K2))``: every l = 1 block as a Cartesian
    vector and every l = 2 block as a traceless tensor (K1, K2 copies)."""
    blocks = degree_blocks(degrees)
    U = [l1_to_cartesian(ev[:, a:b]) for deg, a, b in blocks if deg == 1]
    T = [l2_to_cartesian(ev[:, a:b]) for deg, a, b in blocks if deg == 2]
    return torch.stack(U, -1), torch.stack(T, -1)


class SO3LRFeatureTrunk(nn.Module):
    """Cached SO3LR-T features -> FF-1's ``(h, u, Q, geo)`` (module docstring).

    Parameters
    ----------
    n_feat, n_chan, n_rbf, r_cut, rho_tr
        FF-1's trunk widths, radial basis, cutoff (Angstrom) and norm cap.
    adapter : bool
        Arm D: one trainable FF1Layer after the projection; False is arm C.
    x_dim, degrees
        The cached feature layout (T-L: 256, ``[1, 1, 2, 2, 3, 3, 4, 4]``).
    """

    def __init__(self, n_feat: int = 128, n_chan: int = 16, n_rbf: int = 16,
                 r_cut: float = 5.0, rho_tr: float = 4.0, adapter: bool = False,
                 x_dim: int = TL_X_DIM, degrees=TL_DEGREES):
        super().__init__()
        self.n_feat, self.n_chan, self.r_cut, self.rho = n_feat, n_chan, r_cut, rho_tr
        self.degrees = tuple(int(d) for d in degrees)
        self.n_ev = sum(2 * d + 1 for d in self.degrees)
        self.k1 = sum(d == 1 for d in self.degrees)
        self.k2 = sum(d == 2 for d in self.degrees)
        if not (self.k1 and self.k2):
            raise ValueError(f"degrees {degrees} need at least one l = 1 and l = 2 block")
        self.adapter = bool(adapter)
        self.rbf = GaussianRBF(n_rbf, r_cut)
        self.h_map = nn.Linear(x_dim + len(self.degrees), n_feat)
        self.u_gate = mlp(x_dim, n_feat, n_chan * self.k1, 2)
        self.q_gate = mlp(x_dim, n_feat, n_chan * self.k2, 2)
        self.layers = nn.ModuleList(
            [FF1Layer(n_feat, n_chan, n_rbf, rho_tr)] if self.adapter else [])

    edge_geometry = staticmethod(FF1MonomerTrunk.edge_geometry)

    def project(self, x: Tensor, ev: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Arm C's ``(h, u, Q)`` from the cached features."""
        n, C = x.shape[0], self.n_chan
        h = self.h_map(torch.cat([x, ev_invariants(ev, self.degrees)], 1))
        U, T = ev_cartesian(ev, self.degrees)
        gu = self.u_gate(x).view(n, C, self.k1)
        gq = self.q_gate(x).view(n, C, self.k2)
        u = torch.einsum("nak,nck->nac", U, gu)
        Q = torch.einsum("nabk,nck->nabc", T, gq)
        return h, norm_cap(u, self.rho, (1,)), norm_cap(Q, self.rho, (1, 2))

    def forward(self, z: Tensor, R: Tensor, edges: Tensor, x: Tensor, ev: Tensor):
        """(h, u, Q, edge geometry); ``x``/``ev`` are the cached features of
        the atoms ``z`` at ``R`` (Angstrom), in the same atom order."""
        src, dst = edges
        r, rt_len, rt = self.edge_geometry(R, edges)
        fc = f_cut(r, self.r_cut)
        rbf = self.rbf(rt_len)
        dt = R.dtype
        h, u, Q = self.project(x.to(dt), ev.to(dt))
        for layer in self.layers:
            h, u, Q = layer(h, u, Q, src, dst, rt, rbf, fc)
        return h, u, Q, {"r": r, "rt": rt, "fc": fc, "rbf": rbf}
