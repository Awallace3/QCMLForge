"""FF-2 pair physics: FF-1's Eqs. 7-19 with every term vectorized.

Per-pair "side" tensors carry each site's two multipole shells stacked on
axis 1 (``M`` (P, 2, 3), ``T`` (P, 2, 3, 3), ``bs`` (P, 2)): shell 0 is
``(mu + d_sr, Theta + Q_sr)`` on ``beta_def`` and shell 1 is
``(-d_sr, -Q_sr)`` on ``beta`` (FF-1's ``multipole_shells`` with
``elst_beta_def`` and ``elst_sr_shell``).  A model without those extensions is
the special case ``beta_def = beta``, ``d_sr = Q_sr = 0``.  Charges keep the
three-charge (nucleus, frozen core, active cloud) form.

Exchange, charge transfer and dispersion are FF-1's own functions
(:mod:`apnet_pt.ff1.physics`), which are already elementwise.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ..ff1.physics import (beta_pol, charge_transfer_pairs, dispersion_pairs,
                           exchange_pairs)
from . import kernels as K

__all__ = ["beta_pol", "charge_transfer_pairs", "dispersion_pairs",
           "exchange_pairs", "elst_pairs", "permanent_field",
           "transverse_invariants", "polarization", "expm_sym"]


def _dot(a: Tensor, b: Tensor) -> Tensor:
    return (a * b).sum(-1)


def _mat_vec(T: Tensor, v: Tensor) -> Tensor:
    """T @ v over the last two / last axis with broadcasting."""
    return (T * v[..., None, :]).sum(-1)


def elst_pairs(r: Tensor, rhat: Tensor, i: dict, j: dict) -> Tensor:
    """Exact Coulomb energy (E_h) of two two-shell Eq. 7 densities, per pair.

    ``i``/``j``: ``z``, ``nc``, ``bc``, ``n``, ``b``, ``q`` (P,) and the
    stacked shells ``M``, ``T``, ``bs``; ``rhat`` points from i to j.
    """
    rc = r[:, None]
    # monopole: q_i q_j / R minus the decaying complements
    tc = K.two_center(torch.stack([i["b"], i["bc"], i["b"], i["bc"]], 1),
                      torch.stack([j["b"], j["b"], j["bc"], j["bc"]], 1),
                      rc, K.BAR1)[..., 0]
    oc = K.one_center(torch.stack([j["b"], i["b"], j["bc"], i["bc"]], 1),
                      rc, K.BAR1)[..., 0]
    bar = (-i["z"] * (j["nc"] * oc[:, 2] + j["n"] * oc[:, 0])
           - j["z"] * (i["nc"] * oc[:, 3] + i["n"] * oc[:, 1])
           + i["nc"] * j["nc"] * tc[:, 3] + i["nc"] * j["n"] * tc[:, 1]
           + i["n"] * j["nc"] * tc[:, 2] + i["n"] * j["n"] * tc[:, 0])
    energy = i["q"] * j["q"] / r - bar

    # charges of one side (axis 1: i, j) vs the other side's shells (axis 3)
    def two(key):
        return torch.stack([i[key], j[key]], 1)

    shell_b = torch.stack([j["bs"], i["bs"]], 1)                 # (P, 2, 2)
    shell_M = torch.stack([j["M"], i["M"]], 1)                   # (P, 2, 2, 3)
    shell_T = torch.stack([j["T"], i["T"]], 1)                   # (P, 2, 2, 3, 3)
    clouds = torch.stack([two("bc"), two("b")], 2)               # (P, 2, 2)
    r4 = r[:, None, None, None]
    cloud = K.two_center(clouds[..., None], shell_b[:, :, None, :], r4,
                         K.CHARGE_SHELL)                         # (P, 2, 2c, 2s, 2)
    nuc = K.one_center(shell_b, r[:, None, None], K.CHARGE_SHELL)  # (P, 2, 2s, 2)
    k = (two("z")[:, :, None, None] * nuc
         - two("nc")[:, :, None, None] * cloud[:, :, 0]
         - two("n")[:, :, None, None] * cloud[:, :, 1])
    rh = rhat[:, None, None, :]
    u = _dot(shell_M, rh)                                        # (P, 2, 2)
    t = _dot(_mat_vec(shell_T, rh), rh)
    sign = torch.tensor([-1.0, 1.0], dtype=r.dtype, device=r.device)[:, None]
    energy = energy + (sign * k[..., 0] * u + k[..., 1] * t).sum((1, 2))

    # multipole-multipole over the 2 x 2 shell combinations
    a = K.two_center(i["bs"][:, :, None], j["bs"][:, None, :], r[:, None, None],
                     K.MM)                                       # (P, 2, 2, 7)
    a53, a33, a74, a54, a95, a75, a55 = a.unbind(-1)
    rv = rhat[:, None, :]
    ui, uj = _dot(i["M"], rv), _dot(j["M"], rv)                  # (P, 2)
    thi_r, thj_r = _mat_vec(i["T"], rv), _mat_vec(j["T"], rv)    # (P, 2, 3)
    ti, tj = _dot(thi_r, rv), _dot(thj_r, rv)
    Mi, Mj = i["M"][:, :, None], j["M"][:, None]
    Ri, Rj = thi_r[:, :, None], thj_r[:, None]
    UI, UJ, TI, TJ = ui[:, :, None], uj[:, None], ti[:, :, None], tj[:, None]
    mm = (-(3.0 * a53 * UI * UJ - a33 * _dot(Mi, Mj))
          + (5.0 * a74 * UI * TJ - 2.0 * a54 * _dot(Mi, Rj))
          - (5.0 * a74 * UJ * TI - 2.0 * a54 * _dot(Mj, Ri))
          + (35.0 / 3.0 * a95 * TI * TJ - 20.0 / 3.0 * a75 * _dot(Ri, Rj)
             + 2.0 / 3.0 * a55 * (i["T"][:, :, None] * j["T"][:, None]).sum((-1, -2))))
    return energy + mm.sum((1, 2))


def permanent_field(r: Tensor, rhat: Tensor, beta_probe: Tensor, src: dict) -> Tensor:
    """Field (E_h/(e bohr)) of ``src``'s two-shell density on a
    Slater(beta_probe) dipole; ``rhat`` points from the source to the probe."""
    rc = r[:, None]
    tc = K.two_center(beta_probe[:, None], torch.stack([src["bc"], src["b"]], 1),
                      rc, K.CHARGE_FIELD)[..., 0]
    nuc = K.one_center(beta_probe, r, K.CHARGE_FIELD)[..., 0]
    charge = src["z"] * nuc - src["nc"] * tc[:, 0] - src["n"] * tc[:, 1]
    act = K.two_center(beta_probe[:, None], src["bs"], rc, K.SHELL_FIELD)
    a53, a33, a74, a54 = act.unbind(-1)                          # (P, 2)
    rv = rhat[:, None, :]
    ur = _dot(src["M"], rv)
    th_r = _mat_vec(src["T"], rv)
    tr = _dot(th_r, rv)
    shells = ((3.0 * a53 * ur + 5.0 * a74 * tr)[..., None] * rv
              - a33[..., None] * src["M"] - 2.0 * a54[..., None] * th_r)
    return charge[:, None] * rhat + shells.sum(1)


def transverse_invariants(rhat: Tensor, ui: Tensor, Qi: Tensor, uj: Tensor,
                          Qj: Tensor) -> Tensor:
    """FF-1's swap-symmetric SO(2) invariants omega (P, 4), elementwise."""
    def perp_vec(v):
        return v - _dot(v, rhat)[:, None] * rhat

    pu_i, pu_j = perp_vec(ui), perp_vec(uj)
    qr_i = perp_vec(_mat_vec(Qi, rhat))
    qr_j = -perp_vec(_mat_vec(Qj, rhat))
    w11 = _dot(pu_i, pu_j)
    w12 = _dot(pu_i, qr_j) + _dot(pu_j, qr_i)
    w221 = _dot(qr_i, qr_j)
    proj = (torch.eye(3, dtype=rhat.dtype, device=rhat.device)
            - rhat[:, :, None] * rhat[:, None, :])

    def perp(Q):
        Qr = _mat_vec(Q, rhat)
        rQr = _dot(Qr, rhat)
        pq = (Q - rhat[:, :, None] * Qr[:, None, :] - Qr[:, :, None] * rhat[:, None, :]
              + rQr[:, None, None] * rhat[:, :, None] * rhat[:, None, :])
        tr = pq[:, 0, 0] + pq[:, 1, 1] + pq[:, 2, 2]
        return pq - 0.5 * tr[:, None, None] * proj

    w222 = (perp(Qi) * perp(Qj)).sum((-1, -2))
    return torch.stack([w11, w12, w221, w222], dim=-1)


def expm_sym(A: Tensor, degree: int = 14) -> Tensor:
    """exp(A) of a batch of small 3 x 3 matrices by a fixed-degree Taylor
    series (Horner).  For ``|A|_F <= 0.2`` (FF-1's capped alpha tensor is
    ``<= 0.175``) the truncation error is below 1e-25: no norm-dependent
    branching, unlike ``torch.linalg.matrix_exp``."""
    eye = torch.eye(3, dtype=A.dtype, device=A.device).expand_as(A)
    P = eye
    for k in range(degree, 0, -1):
        P = eye + (A[..., :, :, None] * P[..., None, :, :]).sum(-2) / k
    return P


def polarization(R_b: Tensor, sqrt_alpha: Tensor, beta_p: Tensor, field: Tensor,
                 b: dict, use_cholesky: bool = True):
    """Variational Gordon1-whitened polarization per dimer (E_h, <= 0).

    Solves ``(I + K) w = S F`` per dimer on the dense slot layout
    (``batching.dimer_layout``) with ``K_kl = S_k T_kl S_l`` and evaluates
    ``U = 1/2 |w|^2 + sum_{k<l} p_k T_kl p_l - p . F`` with ``p = S w*`` and
    ``w*`` held fixed (envelope theorem: exact forces and parameter
    gradients, FF-1's ``polarization``), so the dense matrix is never
    differentiated.  Returns ``(energy (n,), induced dipoles p (N, 3))``.
    """
    k, l, d = b["pol_k"], b["pol_l"], b["pol_dimer"]
    lk, ll, local, adim = b["pol_lk"], b["pol_ll"], b["atom_local"], b["atom_dimer"]
    n, m = b["slots"].shape
    r, rhat = K.safe_unit(R_b[k] - R_b[l])
    lam = K.two_center(beta_p[k], beta_p[l], r, K.DIPOLE_T)
    l33, l53 = lam[:, 0], lam[:, 1]
    rhs = _mat_vec(sqrt_alpha, field)                            # (N, 3)
    with torch.no_grad():
        S_k, S_l = sqrt_alpha[k].detach(), sqrt_alpha[l].detach()
        sk, sl = _mat_vec(S_k, rhat.detach()), _mat_vec(S_l, rhat.detach())
        SS = (S_k[..., :, :, None] * S_l[..., None, :, :]).sum(-2)
        Kkl = (l33.detach()[:, None, None] * SS
               - 3.0 * l53.detach()[:, None, None] * sk[:, :, None] * sl[:, None, :])
        blocks = R_b.new_zeros((n, m, m, 3, 3))
        blocks = blocks.index_put((d, lk, ll), Kkl).index_put((d, ll, lk),
                                                               Kkl.transpose(1, 2))
        A = blocks.permute(0, 1, 3, 2, 4).reshape(n, 3 * m, 3 * m)
        A = A + torch.eye(3 * m, dtype=A.dtype, device=A.device)
        rhs_dense = R_b.new_zeros((n, m, 3)).index_put((adim, local), rhs.detach())
        rhs_dense = rhs_dense.reshape(n, 3 * m, 1)
        if use_cholesky:
            L, _ = torch.linalg.cholesky_ex(A)
            w = torch.cholesky_solve(rhs_dense, L)
        else:
            w = torch.linalg.solve(A, rhs_dense)
        w = w.reshape(n, m, 3)[adim, local]                      # (N, 3)
    p = _mat_vec(sqrt_alpha, w)
    per_atom = 0.5 * _dot(w, w) - _dot(p, field)
    pk, pl = p[k], p[l]
    per_pair = l33 * _dot(pk, pl) - 3.0 * l53 * _dot(pk, rhat) * _dot(pl, rhat)
    energy = (R_b.new_zeros(n).index_add(0, adim, per_atom)
              .index_add(0, d, per_pair))
    return energy, p
