"""Covalent fragments and signed-Fukui charge conservation (proposal Eqs. 4-6).

Each monomer is partitioned once into covalent fragments (bonded when
``r_ik < 1.25 (r_cov,i + r_cov,k)``).  Within each fragment F the active
populations ``N_i = N_i^0 exp([mu_F]_+ s_i^+ - [mu_F]_- s_i^-)`` are solved so
``sum_{i in F} N_i = Z_F^act - Q_F`` exactly; the chemical potential ``mu_F``
is found in float64 by bracketed Newton-bisection in log space and
differentiated by the implicit function theorem (a custom autograd Function),
so charge never leaks between fragments and forces stay conservative.
"""

from __future__ import annotations

import torch
from torch import Tensor

EPS_MU = 1e-2
MU_BRACKET = 2000.0
NEWTON_ITERS = 200
NEWTON_TOL = 1e-13


def smooth_pos_neg(mu: Tensor, eps: float = EPS_MU) -> tuple[Tensor, Tensor]:
    """([mu]_+, [mu]_-), both >= 0, exactly 0 at mu = 0."""
    s = torch.sqrt(mu * mu + eps * eps) - eps
    return 0.5 * (s + mu), 0.5 * (s - mu)


def _dpos_dneg(mu: Tensor, eps: float = EPS_MU) -> tuple[Tensor, Tensor]:
    t = mu / torch.sqrt(mu * mu + eps * eps)
    return 0.5 * (t + 1.0), 0.5 * (t - 1.0)


@torch.no_grad()
def connected_components(n: int, edges: Tensor) -> Tensor:
    """Compact component labels for n nodes under undirected ``edges``."""
    label = torch.arange(n, device=edges.device)
    if edges.numel():
        src, dst = edges
        while True:
            new = label.clone()
            new.scatter_reduce_(0, src, label[dst], reduce="amin")
            new.scatter_reduce_(0, dst, label[src], reduce="amin")
            new = new[new]
            if torch.equal(new, label):
                break
            label = new
    return torch.unique(label, return_inverse=True)[1]


@torch.no_grad()
def covalent_fragments(z: Tensor, r: Tensor, edges: Tensor, monomer: Tensor,
                       monomer_charge: Tensor, z_act: Tensor, r_cov: Tensor,
                       fragment: Tensor | None = None,
                       fragment_charge: Tensor | None = None):
    """(fragment index per atom, Q_F per fragment, #monomers assigned by rule).

    Without explicit ``fragment``/``fragment_charge``: a single-fragment
    monomer takes ``Q_F = Q_M``; a multi-fragment neutral monomer gives every
    fragment ``Q_F = 0``; a charged multi-fragment monomer puts ``sign(Q_M)``
    on each odd-electron fragment when their count equals ``|Q_M|``, else all
    of ``Q_M`` on its fragment with the most active electrons.
    Raises ``ValueError`` when any ``Q_F > Z_F^act``.
    """
    if fragment is None:
        src, dst = edges
        bonded = r < 1.25 * (r_cov[z[src]] + r_cov[z[dst]])
        fragment = connected_components(len(z), edges[:, bonded])
        n_frag = int(fragment.max()) + 1 if len(z) else 0
        frag_mon = torch.zeros(n_frag, dtype=torch.long, device=z.device)
        frag_mon[fragment] = monomer
        n_mon = len(monomer_charge)
        count = torch.zeros(n_mon, dtype=torch.long, device=z.device)
        count.index_add_(0, frag_mon, torch.ones_like(frag_mon))
        q_m = monomer_charge.to(torch.float64).round()
        zf = torch.zeros(n_frag, dtype=torch.long, device=z.device)
        zf.index_add_(0, fragment, z)
        zact_f = torch.zeros(n_frag, dtype=torch.float64, device=z.device)
        zact_f.index_add_(0, fragment, z_act[z].to(torch.float64))
        odd = (zf % 2 == 1).long()
        n_odd = torch.zeros(n_mon, dtype=torch.long, device=z.device)
        n_odd.index_add_(0, frag_mon, odd)
        single = count[frag_mon] == 1
        parity = (n_odd[frag_mon] == q_m[frag_mon].abs().long()) & (q_m[frag_mon] != 0)
        # largest fragment per monomer (ties: lowest index)
        key = zact_f * (n_frag + 1) - torch.arange(n_frag, device=z.device)
        best = torch.full((n_mon,), -float("inf"), dtype=torch.float64, device=z.device)
        best.scatter_reduce_(0, frag_mon, key, reduce="amax")
        is_best = key == best[frag_mon]
        qf = torch.where(
            single, q_m[frag_mon],
            torch.where(parity, odd * torch.sign(q_m[frag_mon]),
                        torch.where(is_best, q_m[frag_mon], torch.zeros_like(zact_f))))
        multi_charged = int(((count > 1) & (q_m != 0)).sum())
        fragment_charge = qf
    else:
        n_frag = int(fragment.max()) + 1
        zact_f = torch.zeros(n_frag, dtype=torch.float64, device=z.device)
        zact_f.index_add_(0, fragment, z_act[z].to(torch.float64))
        fragment_charge = fragment_charge.to(torch.float64)
        multi_charged = 0
    if bool((fragment_charge > zact_f + 1e-9).any()):
        raise ValueError("a fragment charge Q_F exceeds its active nuclear charge")
    return fragment, fragment_charge, multi_charged


def _log_sum(log_n: Tensor, frag: Tensor, n_frag: int) -> Tensor:
    m = torch.full((n_frag,), -float("inf"), dtype=log_n.dtype, device=log_n.device)
    m = m.scatter_reduce(0, frag, log_n, reduce="amax")
    m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
    s = torch.zeros_like(m).index_add(0, frag, torch.exp(log_n - m[frag]))
    return torch.log(s) + m


class FukuiSolve(torch.autograd.Function):
    """mu_F solving sum_{i in F} N_i(mu_F) = T_F, with IFT gradients."""

    @staticmethod
    def forward(ctx, log_n0, s_pos, s_neg, frag, log_t, n_frag):
        with torch.no_grad():
            lo = torch.full_like(log_t, -MU_BRACKET)
            hi = torch.full_like(log_t, MU_BRACKET)
            mu = torch.zeros_like(log_t)
            for _ in range(NEWTON_ITERS):
                p, m = smooth_pos_neg(mu)
                log_n = log_n0 + p[frag] * s_pos - m[frag] * s_neg
                f = _log_sum(log_n, frag, n_frag) - log_t
                if float(f.abs().max()) < NEWTON_TOL:
                    break
                lo = torch.where(f < 0, mu, lo)
                hi = torch.where(f >= 0, mu, hi)
                dp, dm = _dpos_dneg(mu)
                w = torch.exp(log_n - (f + log_t)[frag])  # N_i / S
                fp = torch.zeros_like(mu).index_add(
                    0, frag, w * (dp[frag] * s_pos - dm[frag] * s_neg))
                step = mu - f / fp
                inside = (step > lo) & (step < hi) & torch.isfinite(step)
                mu = torch.where(inside, step, 0.5 * (lo + hi))
        ctx.save_for_backward(log_n0, s_pos, s_neg, frag, mu)
        ctx.n_frag = n_frag
        return mu

    @staticmethod
    def backward(ctx, grad_mu):
        log_n0, s_pos, s_neg, frag, mu = ctx.saved_tensors
        p, m = smooth_pos_neg(mu)
        dp, dm = _dpos_dneg(mu)
        n = torch.exp(log_n0 + p[frag] * s_pos - m[frag] * s_neg)
        ds = torch.zeros_like(mu).index_add(
            0, frag, n * (dp[frag] * s_pos - dm[frag] * s_neg))
        g = (grad_mu / ds)[frag]
        return (-g * n, -g * n * p[frag], g * n * m[frag], None,
                (grad_mu / ds) * torch.exp(_log_sum(log_n0 + p[frag] * s_pos
                                                     - m[frag] * s_neg, frag,
                                                     ctx.n_frag)),
                None)


def fukui_populations(n0: Tensor, s_pos: Tensor, s_neg: Tensor, frag: Tensor,
                      target: Tensor) -> tuple[Tensor, Tensor]:
    """(N_act per atom, mu_F per fragment) in float64; fragments with
    ``target == 0`` (e.g. a bare proton) get N = 0."""
    dtype = n0.dtype
    n0, s_pos, s_neg = n0.double(), s_pos.double(), s_neg.double()
    target = target.double()
    empty = target <= 1e-12
    log_t = torch.log(torch.where(empty, torch.ones_like(target), target))
    mu = FukuiSolve.apply(torch.log(n0), s_pos, s_neg, frag, log_t, len(target))
    mu = torch.where(empty, torch.zeros_like(mu), mu)
    p, m = smooth_pos_neg(mu)
    n = n0 * torch.exp(p[frag] * s_pos - m[frag] * s_neg)
    n = torch.where(empty[frag], torch.zeros_like(n), n)
    return n.to(dtype), mu
