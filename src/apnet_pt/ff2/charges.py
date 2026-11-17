"""One chemical potential per monomer ("one-mu") signed-Fukui charges.

Every atom of monomer M shares one ``mu_M``; the active populations
``N_i = N_i^0 exp([mu_M]_+ s_i^+ - [mu_M]_- s_i^-)`` are solved so
``sum_{i in M} N_i = Z_M^act - Q_M`` exactly.  This is FF-1's
``FukuiSolve`` with the fragment set to the monomer, rebuilt for
``torch.compile``:

* the bracketed Newton-bisection runs a *fixed* number of iterations (no
  ``float(...)`` convergence test, so no host sync per iteration; once
  converged an iteration is a fixed point);
* the implicit-function gradient comes from one differentiable correction
  ``mu = mu* - (f(mu*) - stop_grad f(mu*)) / stop_grad f'(mu*)``, whose value
  is exactly ``mu*`` and whose first derivative is FF-1's IFT backward --
  no custom autograd Function.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ..ff1.charges import EPS_MU, MU_BRACKET, NEWTON_TOL, smooth_pos_neg

N_ITERS = 12
PAD = 1.0


def _dpos_dneg(mu: Tensor, eps: float = EPS_MU) -> tuple[Tensor, Tensor]:
    t = mu / torch.sqrt(mu * mu + eps * eps)
    return 0.5 * (t + 1.0), 0.5 * (t - 1.0)


def _log_sum(log_n: Tensor, mon: Tensor, n_mon: int) -> Tensor:
    m = torch.full((n_mon,), -float("inf"), dtype=log_n.dtype, device=log_n.device)
    m = m.scatter_reduce(0, mon, log_n, reduce="amax")
    m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
    s = torch.zeros_like(m).index_add(0, mon, torch.exp(log_n - m[mon]))
    return torch.log(s) + m


def _residual(mu, log_n0, s_pos, s_neg, mon, log_t):
    """(f = log sum N - log T, log N_i) at ``mu`` per monomer."""
    p, m = smooth_pos_neg(mu)
    log_n = log_n0 + p[mon] * s_pos - m[mon] * s_neg
    return _log_sum(log_n, mon, log_t.shape[0]) - log_t, log_n


def _slope(mu, log_n, f, log_t, s_pos, s_neg, mon):
    """df/dmu per monomer."""
    dp, dm = _dpos_dneg(mu)
    w = torch.exp(log_n - (f + log_t)[mon])  # N_i / S
    return torch.zeros_like(mu).index_add(
        0, mon, w * (dp[mon] * s_pos - dm[mon] * s_neg))


def _bracket(log_n0, s_pos, s_neg, mon, log_t):
    """Per-monomer root bracket and Newton start.

    With ``L = log T - log sum N0``, ``N_i`` lies between ``N0_i e^{mu s_min}``
    and ``N0_i e^{mu s_max}`` (up to the eps smoothing of ``[mu]_+-``), so the
    root sits in ``[L/s_max, L/s_min]`` (mu > 0) or ``[L/s_min, L/s_max]``
    (mu < 0), widened by ``PAD``.  The start is ``L / <s>`` with ``<s>`` the
    N0-weighted mean of the active sensitivity.  A bracket whose end points
    do not straddle the root falls back to FF-1's ``+-MU_BRACKET``.
    """
    n_mon = log_t.shape[0]
    log_s0 = _log_sum(log_n0, mon, n_mon)
    L = log_t - log_s0
    up = L >= 0
    s = torch.where(up[mon], s_pos, s_neg)
    s_min = torch.full_like(L, float("inf")).scatter_reduce(0, mon, s, reduce="amin")
    s_max = torch.zeros_like(L).scatter_reduce(0, mon, s, reduce="amax")
    w = torch.exp(log_n0 - log_s0[mon])
    s_mean = torch.zeros_like(L).index_add(0, mon, w * s)
    a, b = L / s_max, L / s_min
    lo = torch.minimum(a, b) - PAD
    hi = torch.maximum(a, b) + PAD
    f_lo, _ = _residual(lo, log_n0, s_pos, s_neg, mon, log_t)
    f_hi, _ = _residual(hi, log_n0, s_pos, s_neg, mon, log_t)
    ok = (f_lo < 0) & (f_hi > 0)
    lo = torch.where(ok, lo, torch.full_like(lo, -MU_BRACKET))
    hi = torch.where(ok, hi, torch.full_like(hi, MU_BRACKET))
    mu0 = torch.minimum(torch.maximum(L / s_mean, lo), hi)
    return lo, hi, mu0


@torch.no_grad()
def solve_mu(log_n0: Tensor, s_pos: Tensor, s_neg: Tensor, mon: Tensor,
             log_t: Tensor, n_iters: int = N_ITERS) -> tuple[Tensor, Tensor]:
    """(mu*, |f(mu*)|): FF-1's bracketed Newton-bisection for a fixed number
    of iterations from :func:`_bracket`'s start; a monomer whose residual is
    below FF-1's ``NEWTON_TOL`` is frozen (FF-1 stops there)."""
    lo, hi, mu = _bracket(log_n0, s_pos, s_neg, mon, log_t)
    for _ in range(n_iters):
        f, log_n = _residual(mu, log_n0, s_pos, s_neg, mon, log_t)
        done = f.abs() < NEWTON_TOL
        lo = torch.where(f < 0, mu, lo)
        hi = torch.where(f >= 0, mu, hi)
        fp = _slope(mu, log_n, f, log_t, s_pos, s_neg, mon)
        step = mu - f / fp
        inside = (step > lo) & (step < hi) & torch.isfinite(step)
        mu = torch.where(done, mu, torch.where(inside, step, 0.5 * (lo + hi)))
    f, _ = _residual(mu, log_n0, s_pos, s_neg, mon, log_t)
    return mu, f.abs()


def monomer_populations(n0: Tensor, s_pos: Tensor, s_neg: Tensor, mon: Tensor,
                        target: Tensor, n_iters: int = N_ITERS):
    """(N_act per atom, mu per monomer, |residual| per monomer), float64.

    ``target`` is ``Z_M^act - Q_M`` per monomer; a monomer with
    ``target == 0`` (a bare proton) gets ``N = 0`` and ``mu = 0``.
    """
    n0, s_pos, s_neg, target = n0.double(), s_pos.double(), s_neg.double(), \
        target.double()
    empty = target <= 1e-12
    log_t = torch.log(torch.where(empty, torch.ones_like(target), target))
    log_n0 = torch.log(n0)
    mu_star, resid = solve_mu(log_n0.detach(), s_pos.detach(), s_neg.detach(),
                              mon, log_t.detach(), n_iters)
    f, log_n = _residual(mu_star, log_n0, s_pos, s_neg, mon, log_t)
    with torch.no_grad():
        fp = _slope(mu_star, log_n, f, log_t, s_pos, s_neg, mon)
    mu = mu_star - (f - f.detach()) / fp
    mu = torch.where(empty, torch.zeros_like(mu), mu)
    p, m = smooth_pos_neg(mu)
    n = n0 * torch.exp(p[mon] * s_pos - m[mon] * s_neg)
    n = torch.where(empty[mon], torch.zeros_like(n), n)
    return n, mu, torch.where(empty, torch.zeros_like(resid), resid)
