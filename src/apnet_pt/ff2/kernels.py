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
    xs = x.clamp_max(X_SWITCH)
    xl = x.clamp_min(X_SWITCH)
    ex, exl = torch.exp(-x), torch.exp(-xl)
    small_region = x <= X_SWITCH
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


def safe_unit(vec: Tensor, eps: float = 1e-300) -> tuple[Tensor, Tensor]:
    """(|vec|, vec/|vec|), finite with finite gradients at vec = 0."""
    r2 = (vec * vec).sum(-1)
    r = torch.sqrt(r2.clamp_min(max(eps, torch.finfo(vec.dtype).tiny)))
    return r, vec / r[..., None]
