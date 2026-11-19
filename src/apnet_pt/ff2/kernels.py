"""Batched Gordon1 (1s-Slater) Coulomb kernels for FF-2.

The same reduced functions as :mod:`apnet_pt.ff1.gordon1` (see its docstring
for the mathematics), evaluated so one call covers *many* exponent pairs: the
exponent arguments carry an extra trailing "combo" axis that broadcasts
against ``r``, and every requested ``(n, p)`` spec comes back stacked on the
last axis.  FF-1 issues one small kernel chain per (exponent pair, spec); FF-2
issues one chain per spec *set*, which ``torch.compile`` fuses into a handful
of kernels.  Nothing here branches on data, syncs with the host, or loops
over atoms.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from ..ff1.gordon1 import (_1C_BAR1, _1C_POLY, _1C_TAIL, _EQ_BAR1, _EQ_POLY,
                           _EQ_TAIL, K_MAX, N_SERIES, X_SWITCH, gauss_jacobi_11)

Spec = tuple  # (n, p) with n in {1, 3, 5, 7, 9, "bar1"}

BAR1 = (("bar1", 1),)
CHARGE_SHELL = ((3, 2), (5, 3))
MM = ((5, 3), (3, 3), (7, 4), (5, 4), (9, 5), (7, 5), (5, 5))
SHELL_FIELD = ((5, 3), (3, 3), (7, 4), (5, 4))
CHARGE_FIELD = ((3, 2),)
DIPOLE_T = ((3, 3), (5, 3))
# quadrature evaluated once at import (numpy is not traceable)
NODES, WEIGHTS = gauss_jacobi_11()


def _tails(x: Tensor) -> list[Tensor]:
    """[R_0(x), ..., R_K_MAX(x)] for 0 <= x <= X_SWITCH (FF-1's recurrence)."""
    term = torch.full_like(x, 1.0 / math.factorial(K_MAX))
    acc = term
    for m in range(1, N_SERIES):
        term = term * (x / (m + K_MAX))
        acc = acc + term
    out = [acc]
    for k in range(K_MAX - 1, -1, -1):
        out.append(1.0 / math.factorial(k) + x * out[-1])
    return out[::-1]


def _reduced(x: Tensor, specs, two_center: bool) -> Tensor:
    """lambda_n(x)/x^p for every spec, stacked on a new last axis."""
    poly, tail, bar = ((_EQ_POLY, _EQ_TAIL, _EQ_BAR1) if two_center
                       else (_1C_POLY, _1C_TAIL, _1C_BAR1))
    x = x.clamp_min(0.0)
    small_region = x <= X_SWITCH
    # each branch's argument by where, not clamp: at x == X_SWITCH the slope
    # must follow the branch evaluated there, and torch >= 2.15 gives clamp a
    # zero gradient exactly at its bound (table nodes sit on X_SWITCH)
    switch = torch.full_like(x, X_SWITCH)
    xs = torch.where(small_region, x, switch)
    xl = torch.where(small_region, switch, x)
    ex, exl = torch.exp(-x), torch.exp(-xl)
    needs_tail = any(n != "bar1" for n, _ in specs)
    tails = _tails(xs) if needs_tail else None
    out = []
    for n, p in specs:
        if n == "bar1":
            val = sum(c * x**m for m, c in enumerate(bar)) * ex
            out.append(val / x**p if p else val)
            continue
        val = torch.zeros_like(x)
        for m, c in enumerate(poly[n]):
            if c != 0.0:
                val = val + c * x ** (m - p)
        val = val * ex
        k = tail[n]
        small = ex * x ** (k - p) * tails[k] if k > p else ex * tails[k]
        head = sum(xl**m / math.factorial(m) for m in range(k))
        large = (1.0 - exl * head) / xl**p
        out.append(val + torch.where(small_region, small, large))
    return torch.stack(out, dim=-1)


def _powers(c: Tensor, specs) -> Tensor:
    return torch.stack([c ** p for _, p in specs], dim=-1)


def one_center(beta: Tensor, r: Tensor, specs) -> Tensor:
    """Point charge vs Slater(beta): lambda_n / R^p, shape ``(..., n_specs)``.

    ``beta`` and ``r`` broadcast (e.g. ``beta`` (P, K) against ``r`` (P, 1)).
    """
    return _reduced(beta * r, specs, False) * _powers(beta, specs)


def two_center(beta_i: Tensor, beta_j: Tensor, r: Tensor, specs) -> Tensor:
    """Slater(beta_i) vs Slater(beta_j): lambda_n / R^p, ``(..., n_specs)``.

    FF-1's 16-point Gauss-Jacobi(1, 1) Feynman-Schwinger rule, vectorized
    over the broadcast shape of ``beta_i``, ``beta_j`` and ``r``.
    """
    v = torch.tensor(NODES, dtype=r.dtype, device=r.device)
    w = torch.tensor(WEIGHTS, dtype=r.dtype, device=r.device)
    bi, bj, rr = beta_i[..., None], beta_j[..., None], r[..., None]
    u = (1.0 - v) * bi + v * bj
    c = bi * bj / u
    wt = 12.0 * w * u * (bi + u) * (bj + u) / (bi + bj) ** 3
    return (wt[..., None] * _powers(c, specs) * _reduced(c * rr, specs, True)
            ).sum(-2)


def point(r: Tensor, shape, specs) -> Tensor:
    """The large-separation limit of :func:`two_center` / :func:`one_center`:
    ``R^-p`` for every spec and 0 for ``"bar1"``, broadcast to ``shape``.

    Every Slater term decays as ``exp(-beta R)`` times a polynomial, so at
    ``x = beta_min R >= 40`` the damped kernels equal this to ~6e-12 relative.
    """
    out = [torch.zeros_like(r) if n == "bar1" else r ** -p for n, p in specs]
    return torch.stack(out, dim=-1).expand(*shape, len(specs))


# ---- tabulated reduced functions (mode "table") -----------------------------
X_TAB = 40.0      # beyond, the point limit (see point())
H_TAB = 0.005     # cubic Hermite cell width in x
_TABLES: dict = {}


def _weight(x: Tensor, specs) -> Tensor:
    """(1 + x)^p for each spec, x^p for "bar1" (its R^-1 is singular at 0)."""
    return torch.stack([x ** p if n == "bar1" else (1.0 + x) ** p
                        for n, p in specs], -1)


def _table(specs, two_center: bool, dtype, device) -> Tensor:
    """(n_cells, n_specs, 4) cubic Hermite coefficients of the weighted
    reduced functions on [0, X_TAB], built once from :func:`_reduced` values
    and autograd slopes in float64."""
    key = (tuple(specs), two_center, dtype, device)
    if key not in _TABLES:
        n = int(round(X_TAB / H_TAB))
        with torch.enable_grad():  # slopes by autograd, even under no_grad
            x = torch.linspace(0.0, X_TAB, n + 1, dtype=torch.float64).requires_grad_(True)
            xs = x.clamp_min(1e-12)
            g = _reduced(xs, specs, two_center) * _weight(xs, specs)
            dg = torch.stack([torch.autograd.grad(g[:, k].sum(), x, retain_graph=True)[0]
                              for k in range(len(specs))], -1)
        g, dg = g.detach(), dg * H_TAB          # slopes per unit t
        g0, g1, d0, d1 = g[:-1], g[1:], dg[:-1], dg[1:]
        coef = torch.stack([g0, d0, 3 * (g1 - g0) - 2 * d0 - d1,
                            2 * (g0 - g1) + d0 + d1], -1)
        if not bool(torch.isfinite(coef).all()):
            raise ValueError(f"non-finite reduced-function table for {specs}")
        _TABLES[key] = coef.to(dtype=dtype, device=device)
    return _TABLES[key]


def _reduced_table(x: Tensor, specs, two_center: bool) -> Tensor:
    """:func:`_reduced` from the cubic Hermite table (C1 in x), with the
    point limit past X_TAB."""
    tab = _table(specs, two_center, x.dtype, x.device)
    xc = x.clamp(0.0, X_TAB)
    s = xc / H_TAB
    idx = s.detach().floor().clamp_max(tab.shape[0] - 1)
    t = (s - idx)[..., None]
    c = tab[idx.long()]                         # (..., n_specs, 4)
    g = ((c[..., 3] * t + c[..., 2]) * t + c[..., 1]) * t + c[..., 0]
    near = g / _weight(xc.clamp_min(1e-300), specs)
    far = torch.stack([torch.zeros_like(x) if n == "bar1" else
                       x.clamp_min(X_TAB) ** -p for n, p in specs], -1)
    return torch.where((x > X_TAB)[..., None], far, near)


def one_center_table(beta: Tensor, r: Tensor, specs) -> Tensor:
    """:func:`one_center` with tabulated reduced functions."""
    return _reduced_table(beta * r, specs, False) * _powers(beta, specs)


def two_center_table(beta_i: Tensor, beta_j: Tensor, r: Tensor, specs) -> Tensor:
    """:func:`two_center` with tabulated reduced functions (same quadrature)."""
    v = torch.tensor(NODES, dtype=r.dtype, device=r.device)
    w = torch.tensor(WEIGHTS, dtype=r.dtype, device=r.device)
    bi, bj, rr = beta_i[..., None], beta_j[..., None], r[..., None]
    u = (1.0 - v) * bi + v * bj
    c = bi * bj / u
    wt = 12.0 * w * u * (bi + u) * (bj + u) / (bi + bj) ** 3
    return (wt[..., None] * _powers(c, specs) * _reduced_table(c * rr, specs, True)
            ).sum(-2)


# ---- 2D table of the whole two-center kernel (mode "table2d") ---------------
# With b = min(beta_i, beta_j), rho = b / max(beta_i, beta_j) and x = b R, the
# quadrature in two_center is b^p G(x, rho): every node argument c R is
# b h_k(rho) R with h_k = 1 / (rho + v_k (1 - rho)).  G is tabulated, weighted
# like the 1D table, by bicubic Hermite on u = log(1 + x) / log(1 + X_TAB)
# and s = log(rho / RHO_TAB) / log(1 / RHO_TAB): the log in rho resolves the
# rho ~ v_k structure of the small-node terms, the log in x the small-x one.
# Error against two_center, times R^p: < 5e-9 for x >= 0.05 (bar1 reaches
# ~3e-4 only below x = 0.05, i.e. R < 0.05 bohr).
RHO_TAB = 0.02    # smallest beta_min / beta_max tabulated; smaller is clamped
N_U2, N_S2 = 800, 192
_LU, _LS = math.log1p(X_TAB), math.log(1.0 / RHO_TAB)


def _g2(u: Tensor, s: Tensor, specs) -> Tensor:
    """Weighted G at grid coordinates (u, s), float64, stacked on the last axis."""
    v = torch.tensor(NODES, dtype=u.dtype, device=u.device)
    w = torch.tensor(WEIGHTS, dtype=u.dtype, device=u.device)
    x = torch.expm1(u * _LU).clamp_min(1e-12)
    rho = RHO_TAB * torch.exp(s * _LS)
    xn, rn = x[..., None], rho[..., None]
    q = (1.0 - v) * rn + v                       # u / beta_max
    wt = 12.0 * w * q * (rn + q) * (1.0 + q) / (rn + 1.0) ** 3
    h = 1.0 / q
    g = (wt[..., None] * _powers(h, specs) * _reduced(h * xn, specs, True)).sum(-2)
    return g * _weight(x, specs)


def _table2d(specs, dtype, device) -> Tensor:
    """(N_U2 + 1, N_S2 + 1, n_specs, 4) node values and scaled derivatives
    (g, g_u, g_s, g_us) of :func:`_g2`, by forward-mode AD in float64.  Built
    once per spec set (about 30 s for MM on one core)."""
    from torch.func import jvp

    key = ("2d", tuple(specs), dtype, device)
    if key not in _TABLES:
        u = torch.linspace(0.0, 1.0, N_U2 + 1, dtype=torch.float64)
        s = torch.linspace(0.0, 1.0, N_S2 + 1, dtype=torch.float64)
        uu, ss = (t.clone() for t in torch.meshgrid(u, s, indexing="ij"))
        one, zero = torch.ones_like(uu), torch.zeros_like(uu)

        def g(a, b):
            return _g2(a, b, specs)

        def g_s(a, b):
            return jvp(g, (a, b), (zero, one))[1]

        val, gu = jvp(g, (uu, ss), (one, zero))
        gs, gus = jvp(g_s, (uu, ss), (one, zero))
        tab = torch.stack([val, gu / N_U2, gs / N_S2, gus / (N_U2 * N_S2)], -1)
        if not bool(torch.isfinite(tab).all()):
            raise ValueError(f"non-finite two-center table for {specs}")
        _TABLES[key] = tab.to(dtype=dtype, device=device)
    return _TABLES[key]


def _hermite(t: Tensor) -> Tensor:
    """Cubic Hermite basis (h00, h10, h01, h11) on a new last axis."""
    t2 = t * t
    t3 = t2 * t
    return torch.stack([2 * t3 - 3 * t2 + 1, t3 - 2 * t2 + t, 3 * t2 - 2 * t3,
                        t3 - t2], -1)


def two_center_table2d(beta_i: Tensor, beta_j: Tensor, r: Tensor, specs) -> Tensor:
    """:func:`two_center` by one bicubic lookup per pair instead of the 16-node
    quadrature; past ``x = X_TAB`` the point limit, as :func:`point`.

    ``beta_min / beta_max`` below :data:`RHO_TAB` is clamped to it, so the
    kernel there is that of a ratio of ``RHO_TAB`` (FF-2's exponents stay
    above ~0.07).
    """
    b = torch.minimum(beta_i, beta_j)
    rho = (b / torch.maximum(beta_i, beta_j)).clamp_min(RHO_TAB)
    b, rho, r = torch.broadcast_tensors(b, rho, r)
    x = b * r
    tab = _table2d(specs, x.dtype, x.device)
    su = torch.log1p(x.clamp(0.0, X_TAB)) * (N_U2 / _LU)
    ss = torch.log(rho / RHO_TAB) * (N_S2 / _LS)
    iu = su.detach().floor().clamp(0, N_U2 - 1)
    i_s = ss.detach().floor().clamp(0, N_S2 - 1)
    hu, hs = _hermite(su - iu), _hermite(ss - i_s)
    iu, i_s = iu.long(), i_s.long()
    g = torch.zeros(x.shape + (len(specs),), dtype=x.dtype, device=x.device)
    for du in (0, 1):
        for ds in (0, 1):
            c = tab[iu + du, i_s + ds]                     # (..., n_specs, 4)
            fu, du_ = hu[..., 2 * du, None], hu[..., 2 * du + 1, None]
            fs, ds_ = hs[..., 2 * ds, None], hs[..., 2 * ds + 1, None]
            g = g + (fu * fs * c[..., 0] + du_ * fs * c[..., 1]
                     + fu * ds_ * c[..., 2] + du_ * ds_ * c[..., 3])
    near = g / _weight(x.clamp(1e-300, X_TAB), specs) * _powers(b, specs)
    far = point(r, x.shape, specs)
    return torch.where((x > X_TAB)[..., None], far, near)


def safe_unit(vec: Tensor, eps: float = 1e-300) -> tuple[Tensor, Tensor]:
    """(|vec|, vec/|vec|), finite with finite gradients at vec = 0."""
    r2 = (vec * vec).sum(-1)
    r = torch.sqrt(r2.clamp_min(max(eps, torch.finfo(vec.dtype).tiny)))
    return r, vec / r[..., None]
