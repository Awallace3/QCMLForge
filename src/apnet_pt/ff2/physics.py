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

from ..ff1.physics import (A_POL, beta_pol, charge_transfer_pairs,
                           dispersion_pairs, exchange_pairs)
from . import kernels as K

__all__ = ["A_POL", "beta_pol", "charge_transfer_pairs", "dispersion_pairs",
           "exchange_pairs", "elst_pairs", "elst_pairs_single",
           "permanent_field", "permanent_field_net", "permanent_field_net_pair",
           "transverse_invariants", "polarization", "polarization_system",
           "cg_solve", "by_range", "expm_sym"]


def _dot(a: Tensor, b: Tensor) -> Tensor:
    return (a * b).sum(-1)


def _mat_vec(T: Tensor, v: Tensor) -> Tensor:
    """T @ v over the last two / last axis with broadcasting."""
    return (T * v[..., None, :]).sum(-1)


MODES = ("exact", "table", "table2d", "point")


def _two(bi: Tensor, bj: Tensor, r: Tensor, specs, mode: str) -> Tensor:
    """Two-center kernel: series ("exact"), tabulated reduced functions
    ("table"), one 2D lookup of the whole quadrature ("table2d") or the
    point-multipole limit ("point")."""
    if mode == "point":
        return K.point(r, torch.broadcast_shapes(bi.shape, bj.shape, r.shape), specs)
    if mode == "table":
        return K.two_center_table(bi, bj, r, specs)
    if mode == "table2d":
        return K.two_center_table2d(bi, bj, r, specs)
    return K.two_center(bi, bj, r, specs)


def _one(b: Tensor, r: Tensor, specs, mode: str) -> Tensor:
    if mode == "point":
        return K.point(r, torch.broadcast_shapes(b.shape, r.shape), specs)
    if mode in ("table", "table2d"):  # one-center has no quadrature
        return K.one_center_table(b, r, specs)
    return K.one_center(b, r, specs)


def _take(x, idx: Tensor):
    return {k: v[idx] for k, v in x.items()} if isinstance(x, dict) else x[idx]


def by_range(fn, x_min: Tensor, point_x, mode: str, *args):
    """``fn(*args, mode=mode)`` per pair, with pairs whose smallest Slater
    argument ``x_min = beta_min R`` exceeds ``point_x`` evaluated in the
    point-multipole limit (``fn(..., mode="point")``) instead.

    Every ``args`` entry is a per-pair tensor (or a dict of them) indexed on
    axis 0.  ``point_x=None`` runs every pair damped (exact).  The split is a
    data-dependent gather, so the cost of the damped kernels scales with the
    number of near pairs."""
    if mode not in MODES:
        raise ValueError(f"kernel mode {mode!r} not in {MODES}")
    if point_x is None:
        return fn(*args, mode=mode)
    far = x_min > point_x
    near_i = torch.nonzero(~far).squeeze(-1)
    far_i = torch.nonzero(far).squeeze(-1)
    a = fn(*[_take(v, near_i) for v in args], mode=mode)
    b = fn(*[_take(v, far_i) for v in args], mode="point")
    out = a.new_zeros((x_min.shape[0],) + a.shape[1:])
    return out.index_put((near_i,), a).index_put((far_i,), b)


def elst_pairs(r: Tensor, rhat: Tensor, i: dict, j: dict,
               mode: str = "exact") -> Tensor:
    """Exact Coulomb energy (E_h) of two two-shell Eq. 7 densities, per pair.

    ``i``/``j``: ``z``, ``nc``, ``bc``, ``n``, ``b``, ``q`` (P,) and the
    stacked shells ``M``, ``T``, ``bs``; ``rhat`` points from i to j.
    """
    rc = r[:, None]
    # monopole: q_i q_j / R minus the decaying complements
    tc = _two(torch.stack([i["b"], i["bc"], i["b"], i["bc"]], 1),
              torch.stack([j["b"], j["b"], j["bc"], j["bc"]], 1),
              rc, K.BAR1, mode)[..., 0]
    oc = _one(torch.stack([j["b"], i["b"], j["bc"], i["bc"]], 1),
              rc, K.BAR1, mode)[..., 0]
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
    cloud = _two(clouds[..., None], shell_b[:, :, None, :], r4,
                 K.CHARGE_SHELL, mode)                          # (P, 2, 2c, 2s, 2)
    nuc = _one(shell_b, r[:, None, None], K.CHARGE_SHELL, mode)  # (P, 2, 2s, 2)
    k = (two("z")[:, :, None, None] * nuc
         - two("nc")[:, :, None, None] * cloud[:, :, 0]
         - two("n")[:, :, None, None] * cloud[:, :, 1])
    rh = rhat[:, None, None, :]
    u = _dot(shell_M, rh)                                        # (P, 2, 2)
    t = _dot(_mat_vec(shell_T, rh), rh)
    sign = torch.tensor([-1.0, 1.0], dtype=r.dtype, device=r.device)[:, None]
    energy = energy + (sign * k[..., 0] * u + k[..., 1] * t).sum((1, 2))

    # multipole-multipole over the 2 x 2 shell combinations
    a = _two(i["bs"][:, :, None], j["bs"][:, None, :], r[:, None, None],
             K.MM, mode)                                        # (P, 2, 2, 7)
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


def elst_pairs_single(r: Tensor, rhat: Tensor, i: dict, j: dict,
                      mode: str = "exact") -> Tensor:
    """Exact single-shell Coulomb energy (E_h) when ``sr_shell=False``
    (shell 1 moments ``M[:, 1]`` and ``T[:, 1]`` are identically zero).

    Mathematically identical to :func:`elst_pairs` for ``sr_shell=False``,
    eliminating 3/4 of the two-center ``MM`` kernel evaluations and 1/2 of
    the ``CHARGE_SHELL`` evaluations.
    """
    rc = r[:, None]
    tc = _two(torch.stack([i["b"], i["bc"], i["b"], i["bc"]], 1),
              torch.stack([j["b"], j["b"], j["bc"], j["bc"]], 1),
              rc, K.BAR1, mode)[..., 0]
    oc = _one(torch.stack([j["b"], i["b"], j["bc"], i["bc"]], 1),
              rc, K.BAR1, mode)[..., 0]
    bar = (-i["z"] * (j["nc"] * oc[:, 2] + j["n"] * oc[:, 0])
           - j["z"] * (i["nc"] * oc[:, 3] + i["n"] * oc[:, 1])
           + i["nc"] * j["nc"] * tc[:, 3] + i["nc"] * j["n"] * tc[:, 1]
           + i["n"] * j["nc"] * tc[:, 2] + i["n"] * j["n"] * tc[:, 0])
    energy = i["q"] * j["q"] / r - bar

    def two(key):
        return torch.stack([i[key], j[key]], 1)

    bi0, bj0 = i["bs"][:, 0], j["bs"][:, 0]
    Mi, Mj = i["M"][:, 0], j["M"][:, 0]
    Ti, Tj = i["T"][:, 0], j["T"][:, 0]

    # charges of one side (axis 1: i, j) vs the other side's single shell
    shell_b = torch.stack([bj0, bi0], 1)                         # (P, 2)
    shell_M = torch.stack([Mj, Mi], 1)                           # (P, 2, 3)
    shell_T = torch.stack([Tj, Ti], 1)                           # (P, 2, 3, 3)
    clouds = torch.stack([two("bc"), two("b")], 2)               # (P, 2, 2)
    cloud = _two(clouds, shell_b[:, :, None], r[:, None, None],
                 K.CHARGE_SHELL, mode)                           # (P, 2, 2c, 2)
    nuc = _one(shell_b, r[:, None], K.CHARGE_SHELL, mode)        # (P, 2, 2)
    k = (two("z")[:, :, None] * nuc
         - two("nc")[:, :, None] * cloud[:, :, 0]
         - two("n")[:, :, None] * cloud[:, :, 1])
    rh = rhat[:, None, :]
    u = _dot(shell_M, rh)                                        # (P, 2)
    t = _dot(_mat_vec(shell_T, rh), rh)
    sign = torch.tensor([-1.0, 1.0], dtype=r.dtype, device=r.device)
    energy = energy + (sign * k[..., 0] * u + k[..., 1] * t).sum(1)

    # single-shell multipole-multipole (1 x 1 instead of 2 x 2)
    a = _two(bi0, bj0, r, K.MM, mode)                            # (P, 7)
    a53, a33, a74, a54, a95, a75, a55 = a.unbind(-1)
    ui, uj = _dot(Mi, rhat), _dot(Mj, rhat)
    Ri, Rj = _mat_vec(Ti, rhat), _mat_vec(Tj, rhat)
    ti, tj = _dot(Ri, rhat), _dot(Rj, rhat)
    mm = (-(3.0 * a53 * ui * uj - a33 * _dot(Mi, Mj))
          + (5.0 * a74 * ui * tj - 2.0 * a54 * _dot(Mi, Rj))
          - (5.0 * a74 * uj * ti - 2.0 * a54 * _dot(Mj, Ri))
          + (35.0 / 3.0 * a95 * ti * tj - 20.0 / 3.0 * a75 * _dot(Ri, Rj)
             + 2.0 / 3.0 * a55 * (Ti * Tj).sum((-1, -2))))
    return energy + mm


def permanent_field(r: Tensor, rhat: Tensor, beta_probe: Tensor, src: dict,
                    mode: str = "exact") -> Tensor:
    """Field (E_h/(e bohr)) of ``src``'s two-shell density on a
    Slater(beta_probe) dipole; ``rhat`` points from the source to the probe."""
    rc = r[:, None]
    tc = _two(beta_probe[:, None], torch.stack([src["bc"], src["b"]], 1),
              rc, K.CHARGE_FIELD, mode)[..., 0]
    nuc = _one(beta_probe, r, K.CHARGE_FIELD, mode)[..., 0]
    charge = src["z"] * nuc - src["nc"] * tc[:, 0] - src["n"] * tc[:, 1]
    act = _two(beta_probe[:, None], src["bs"], rc, K.SHELL_FIELD, mode)
    a53, a33, a74, a54 = act.unbind(-1)                          # (P, 2)
    rv = rhat[:, None, :]
    ur = _dot(src["M"], rv)
    th_r = _mat_vec(src["T"], rv)
    tr = _dot(th_r, rv)
    shells = ((3.0 * a53 * ur + 5.0 * a74 * tr)[..., None] * rv
              - a33[..., None] * src["M"] - 2.0 * a54[..., None] * th_r)
    return charge[:, None] * rhat + shells.sum(1)


def permanent_field_net(r: Tensor, rhat: Tensor, beta_probe: Tensor,
                        beta_src: Tensor, q: Tensor, mu: Tensor, th: Tensor,
                        mode: str = "exact") -> Tensor:
    """Field (E_h/(e bohr)) of a source's net permanent multipoles ``(q, mu, th)``
    on a Slater(``beta_probe``) probe, damped by a single two-center Gordon1
    kernel ``G1(beta_probe, beta_src)``; ``rhat`` points from source to probe."""
    a = _two(beta_probe, beta_src, r, K.FIELD_SPECS, mode)
    a32, a53, a33, a74, a54 = a.unbind(-1)
    ur = _dot(mu, rhat)
    th_r = _mat_vec(th, rhat)
    tr = _dot(th_r, rhat)
    return ((a32 * q + 3.0 * a53 * ur + 5.0 * a74 * tr)[:, None] * rhat
            - a33[:, None] * mu - 2.0 * a54[:, None] * th_r)


def permanent_field_net_pair(
    r: Tensor,
    rhat: Tensor,
    beta_i: Tensor,
    beta_j: Tensor,
    qi: Tensor,
    mui: Tensor,
    thi: Tensor,
    qj: Tensor,
    muj: Tensor,
    thj: Tensor,
    mode: str = "exact",
) -> Tensor:
    """Symmetric pair evaluation of :func:`permanent_field_net` for both
    directions ``(i <- j, j <- i)`` sharing one symmetric two-center kernel call.

    ``rhat`` points from ``i`` to ``j``. Returns shape ``(P, 2, 3)`` where
    ``[:, 0]`` is the field on ``i`` due to ``j`` (source-to-probe ``-rhat``)
    and ``[:, 1]`` is the field on ``j`` due to ``i`` (source-to-probe ``+rhat``).
    """
    a = _two(beta_i, beta_j, r, K.FIELD_SPECS, mode)
    a32, a53, a33, a74, a54 = a.unbind(-1)

    ur_j = _dot(muj, rhat)
    th_r_j = _mat_vec(thj, rhat)
    tr_j = _dot(th_r_j, rhat)
    f_on_i = (
        (-a32 * qj + 3.0 * a53 * ur_j - 5.0 * a74 * tr_j)[:, None] * rhat
        - a33[:, None] * muj
        + 2.0 * a54[:, None] * th_r_j
    )

    ur_i = _dot(mui, rhat)
    th_r_i = _mat_vec(thi, rhat)
    tr_i = _dot(th_r_i, rhat)
    f_on_j = (
        (a32 * qi + 3.0 * a53 * ur_i + 5.0 * a74 * tr_i)[:, None] * rhat
        - a33[:, None] * mui
        - 2.0 * a54[:, None] * th_r_i
    )
    return torch.stack([f_on_i, f_on_j], dim=1)


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


def cg_solve(A: Tensor, rhs: Tensor, n_iter: int) -> Tensor:
    """``A^-1 rhs`` for a batch of SPD matrices ``A`` (n, d, d), ``rhs``
    (n, d, 1), by ``n_iter`` steps of conjugate gradients preconditioned with
    the 3 x 3 diagonal blocks (one per atom).  The step count is fixed (no
    early exit), so a traced or compiled graph runs the same solve at every
    geometry; a residual too small to divide by ends the update."""
    n, d, _ = A.shape
    blocks = A.reshape(n, d // 3, 3, d // 3, 3).diagonal(dim1=1, dim2=3)
    P = torch.linalg.inv(blocks.permute(0, 3, 1, 2))          # (n, d/3, 3, 3)

    def precond(v):
        return (P @ v.reshape(n, d // 3, 3, 1)).reshape(n, d, 1)

    x = precond(rhs)
    r = rhs - A @ x
    z = precond(r)
    p, rz = z, (r * z).sum(1, keepdim=True)
    tiny = torch.finfo(A.dtype).tiny
    for _ in range(n_iter):
        Ap = A @ p
        alpha = rz / (p * Ap).sum(1, keepdim=True).clamp_min(tiny)
        x = x + alpha * p
        r = r - alpha * Ap
        z = precond(r)
        rz_new = (r * z).sum(1, keepdim=True)
        p = z + rz_new / rz.clamp_min(tiny) * p
        rz = rz_new
    return x


def polarization_system(sqrt_alpha: Tensor, rhat: Tensor, l33: Tensor, l53: Tensor,
                        rhs: Tensor, b: dict) -> tuple[Tensor, Tensor]:
    """The dense whitened system ``A = I + K`` (n, 3m, 3m) and its right-hand
    side (n, 3m, 1) on the slot layout, from the per-pair dipole kernels
    ``l33``, ``l53`` and per-atom ``rhs = S F`` (all detached here)."""
    k, l, d = b["pol_k"], b["pol_l"], b["pol_dimer"]
    lk, ll, local, adim = b["pol_lk"], b["pol_ll"], b["atom_local"], b["atom_dimer"]
    n, m = b["slots"].shape
    S_k, S_l = sqrt_alpha[k].detach(), sqrt_alpha[l].detach()
    sk, sl = _mat_vec(S_k, rhat.detach()), _mat_vec(S_l, rhat.detach())
    SS = (S_k[..., :, :, None] * S_l[..., None, :, :]).sum(-2)
    Kkl = (l33.detach()[:, None, None] * SS
           - 3.0 * l53.detach()[:, None, None] * sk[:, :, None] * sl[:, None, :])
    blocks = rhs.new_zeros((n, m, m, 3, 3))
    blocks = blocks.index_put((d, lk, ll), Kkl).index_put((d, ll, lk),
                                                           Kkl.transpose(1, 2))
    A = blocks.permute(0, 1, 3, 2, 4).reshape(n, 3 * m, 3 * m)
    A = A + torch.eye(3 * m, dtype=A.dtype, device=A.device)
    rhs_dense = rhs.new_zeros((n, m, 3)).index_put((adim, local), rhs.detach())
    return A, rhs_dense.reshape(n, 3 * m, 1)


def polarization(R_b: Tensor, sqrt_alpha: Tensor, beta_p: Tensor, field: Tensor,
                 b: dict, use_cholesky: bool = True, point_x=None,
                 mode: str = "exact", cg_iters=None):
    """Variational Gordon1-whitened polarization per dimer (E_h, <= 0).

    Solves ``(I + K) w = S F`` per dimer on the dense slot layout
    (``batching.dimer_layout``) with ``K_kl = S_k T_kl S_l`` and evaluates
    ``U = 1/2 |w|^2 + sum_{k<l} p_k T_kl p_l - p . F`` with ``p = S w*`` and
    ``w*`` held fixed (envelope theorem: exact forces and parameter
    gradients, FF-1's ``polarization``), so the dense matrix is never
    differentiated.  ``point_x``, ``mode``: see :func:`by_range`.
    ``cg_iters``: solve by :func:`cg_solve` instead of a factorization.
    Returns ``(energy (n,), induced dipoles p (N, 3))``.
    """
    k, l, d = b["pol_k"], b["pol_l"], b["pol_dimer"]
    local, adim = b["atom_local"], b["atom_dimer"]
    n, m = b["slots"].shape
    r, rhat = K.safe_unit(R_b[k] - R_b[l])
    lam = by_range(lambda bk, bl, rr, mode: _two(bk, bl, rr, K.DIPOLE_T, mode),
                   r * torch.minimum(beta_p[k], beta_p[l]), point_x, mode,
                   beta_p[k], beta_p[l], r)
    l33, l53 = lam[:, 0], lam[:, 1]
    rhs = _mat_vec(sqrt_alpha, field)                            # (N, 3)
    with torch.no_grad():
        A, rhs_dense = polarization_system(sqrt_alpha, rhat, l33, l53, rhs, b)
        if cg_iters is not None:
            w = cg_solve(A, rhs_dense, int(cg_iters))
        elif use_cholesky:
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
