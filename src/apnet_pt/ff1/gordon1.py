"""True 1s-Slater ("Gordon1") Coulomb damping kernels (proposal Eqs. 7-9).

A density ``f0(r; beta) = beta^3/(8 pi) exp(-beta r)`` has Fourier transform
``beta^4/(beta^2 + k^2)^2``.  The Coulomb energy of two unit monopoles is
``g(R) = lambda_1(R)/R``; higher multipole kernels follow the recursion
``lambda_{2n+3} = lambda_{2n+1} - R lambda'_{2n+1}/(2n+1)``, so the damped
Cartesian derivative tensors of ``g`` keep their point-multipole form.

Every kernel is evaluated as a *reduced* function ``lambda_n(x)/x^p``
(``x = beta R``), written as non-negative combinations of ``x^m exp(-x)`` and
the incomplete-gamma tail ``P_k(x) = exp(-x) sum_{n>=k} x^n/n!``.  The tails
use ``R_k(x) = sum_m x^m/(m+k)!`` (``P_k = exp(-x) x^k R_k``) by a positive
Horner series plus the positive backward recurrence ``R_k = 1/k! + x R_{k+1}``
for ``x <= 8`` and ``1 - exp(-x) sum_{n<k} x^n/n!`` (``> 0.54``) above, so no
cancellation occurs and ``R = 0`` is exact.

Unequal exponents use the Feynman-Schwinger representation (verified against
the Fourier integral to ~1e-17)::

    lambda^2c(bi, bj, R) = 12 int_0^1 v(1-v) u (bi+u)(bj+u)/(bi+bj)^3
                           lambda^eq(bi bj R/u) dv,   u = (1-v) bi + v bj

on a fixed 16-point Gauss-Jacobi(1, 1) rule -- no branch on ``bi - bj``.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import torch
from torch import Tensor

X_SWITCH = 8.0
K_MAX = 8
N_SERIES = 40
N_QUAD = 16

# lambda^eq_n = poly_n(x) e^-x + P_{k_n}(x); poly coefficients by power of x.
_EQ_POLY = {
    1: (0.0, 5.0 / 16.0, 5.0 / 16.0, 7.0 / 48.0),
    3: (0.0, 0.0, 0.0, 1.0 / 48.0, 1.0 / 48.0),
    5: (0.0, 0.0, 0.0, 0.0, 0.0, 1.0 / 720.0),
    7: (),
    9: (),
}
_EQ_TAIL = {1: 4, 3: 5, 5: 6, 7: 7, 9: 8}
# 1-center (point vs Slater): lambda1 = x e^-x / 2 + P2, lambda3 = P3, lambda5 = P4.
_1C_POLY = {1: (0.0, 0.5), 3: (), 5: ()}
_1C_TAIL = {1: 2, 3: 3, 5: 4}
# Upper-tail complements 1 - lambda_1 = poly(x) e^-x.
_EQ_BAR1 = (1.0, 11.0 / 16.0, 3.0 / 16.0, 1.0 / 48.0)
_1C_BAR1 = (1.0, 0.5)


@lru_cache(maxsize=None)
def gauss_jacobi_11(n: int = N_QUAD) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Nodes v and weights w on [0, 1] with sum w g(v) = int v(1-v) g(v) dv.

    Gauss-Jacobi(1, 1) nodes are the roots of P'_{n+1} (Legendre); weights
    integrate each Lagrange basis polynomial against v(1 - v) with a 64-point
    Gauss-Legendre rule (exact for these degrees).
    """
    leg = np.polynomial.legendre
    t = np.sort(np.real(leg.legroots(leg.legder([0] * (n + 1) + [1]))))
    xs, ws = leg.leggauss(64)
    weights = []
    for i in range(n):
        others = np.delete(t, i)
        basis = np.prod((xs[:, None] - others) / (t[i] - others), axis=1)
        weights.append(float(np.sum(ws * (1 - xs**2) * basis)))
    v = (1.0 + t) / 2.0
    w = np.asarray(weights) / 8.0  # v(1-v) dv = (1 - t^2)/4 dt/2
    return tuple(v.tolist()), tuple(w.tolist())


def _tails(x: Tensor) -> list[Tensor]:
    """[R_0(x), ..., R_K_MAX(x)] for 0 <= x <= X_SWITCH (all positive).

    R_K_MAX is summed by the forward term recurrence
    t_{m+1} = t_m x / (m + K_MAX + 1), whose factors are <= X_SWITCH /
    (K_MAX + 1) < 1, so neither the sum nor its backward pass can overflow.
    (A Horner loop multiplies the gradient by x once per term: ~x^40, past
    the float32 range at x ~ 8.)
    """
    term = torch.full_like(x, 1.0 / math.factorial(K_MAX))
    acc = term
    for m in range(1, N_SERIES):
        term = term * (x / (m + K_MAX))
        acc = acc + term
    out = [acc]
    for k in range(K_MAX - 1, -1, -1):
        out.append(1.0 / math.factorial(k) + x * out[-1])
    return out[::-1]


def _reduced_tail(k: int, p: int, x: Tensor, tails_small: list[Tensor],
                  x_large: Tensor) -> Tensor:
    """P_k(x)/x^p (p <= k), stable on both sides of X_SWITCH."""
    small = torch.exp(-x) * x ** (k - p) * tails_small[k] if k > p else (
        torch.exp(-x) * tails_small[k])
    head = sum(x_large**n / math.factorial(n) for n in range(k))
    large = (1.0 - torch.exp(-x_large) * head) / x_large**p
    return torch.where(x <= X_SWITCH, small, large)


def _reduced_poly(coeffs: tuple[float, ...], p: int, x: Tensor) -> Tensor:
    out = torch.zeros_like(x)
    for m, c in enumerate(coeffs):
        if c != 0.0:
            if m < p:
                raise ValueError(f"x^{m} e^-x / x^{p} is singular at 0")
            out = out + c * x ** (m - p)
    return out * torch.exp(-x)


def _kernel(x: Tensor, specs, poly, tail) -> dict:
    x = x.clamp_min(0.0)
    x_small = x.clamp_max(X_SWITCH)
    x_large = x.clamp_min(X_SWITCH)
    tails = _tails(x_small)
    out = {}
    for n, p in specs:
        if n == "bar1":
            coeffs = _EQ_BAR1 if poly is _EQ_POLY else _1C_BAR1
            out[(n, p)] = _bar_reduced(coeffs, p, x)
        else:
            out[(n, p)] = (_reduced_poly(poly[n], p, x)
                           + _reduced_tail(tail[n], p, x, tails, x_large))
    return out


def _bar_reduced(coeffs, p, x):
    """(1 - lambda_1)(x)/x^p for p in {0, 1}; diverges at x = 0 for p = 1
    exactly as the bare 1/R it multiplies."""
    poly = sum(c * x**m for m, c in enumerate(coeffs)) * torch.exp(-x)
    return poly / x**p if p else poly


def one_center(beta: Tensor, r: Tensor, specs) -> dict:
    """Point charge vs Slater(beta): {(n, p): lambda_n / R^p} (bohr^-p)."""
    k = _kernel(beta * r, specs, _1C_POLY, _1C_TAIL)
    return {key: v * beta ** key[1] for key, v in k.items()}


def two_center(beta_i: Tensor, beta_j: Tensor, r: Tensor, specs) -> dict:
    """Slater(beta_i) vs Slater(beta_j): {(n, p): lambda_n / R^p} (bohr^-p).

    ``n`` is 1, 3, 5, 7, 9 or ``"bar1"`` (the complement 1 - lambda_1).  The
    rule is branch-free in ``beta_i - beta_j``; at equal exponents it equals
    the closed form to rounding.
    """
    nodes, weights = gauss_jacobi_11()
    v = torch.as_tensor(nodes, dtype=r.dtype, device=r.device)
    w = torch.as_tensor(weights, dtype=r.dtype, device=r.device)
    bi, bj, rr = beta_i[..., None], beta_j[..., None], r[..., None]
    u = (1.0 - v) * bi + v * bj
    c = bi * bj / u
    wt = 12.0 * w * u * (bi + u) * (bj + u) / (bi + bj) ** 3
    k = _kernel(c * rr, specs, _EQ_POLY, _EQ_TAIL)
    return {key: (wt * c ** key[1] * val).sum(-1) for key, val in k.items()}


def equal_center(beta: Tensor, r: Tensor, specs) -> dict:
    """Closed-form equal-exponent kernels (reference for tests)."""
    k = _kernel(beta * r, specs, _EQ_POLY, _EQ_TAIL)
    return {key: v * beta ** key[1] for key, v in k.items()}


# --- multipole contractions ------------------------------------------------

def _bilinear(a: Tensor, m: Tensor, b: Tensor) -> Tensor:
    return torch.einsum("pa,pab,pb->p", a, m, b)


def safe_unit(vec: Tensor, eps: float = 1e-300) -> tuple[Tensor, Tensor]:
    """(|vec|, vec/|vec|), finite with finite gradients at vec = 0.  The floor
    is at least the dtype's smallest normal (1e-300 is 0 in float32)."""
    r2 = (vec * vec).sum(-1)
    r = torch.sqrt(r2.clamp_min(max(eps, torch.finfo(vec.dtype).tiny)))
    return r, vec / r[..., None]


ELST_AA_SPECS = (("bar1", 1), (3, 2), (5, 3), (3, 3), (7, 4), (5, 4), (9, 5),
                 (7, 5), (5, 5))
CHARGE_SPECS = (("bar1", 1), (3, 2), (5, 3))


def _has_deformation(side: dict) -> bool:
    return side.get("b_def") is not None


def multipole_shells(side: dict) -> list[tuple[Tensor, Tensor, Tensor]]:
    """``[(mu, th, beta), ...]``: the site's l = 1, 2 density as Slater shells.

    Without ``b_def`` the moments sit on the valence exponent ``b``.  With
    ``b_def`` they sit on ``b_def``; a zero-moment short-range shell
    ``(d_sr, Q_sr)`` adds ``(+d, +Q)`` on ``b_def`` and ``(-d, -Q)`` on ``b``,
    i.e. ``[-d . grad + Q : grad grad / 3] (f0(b_def) - f0(b))``, whose every
    multipole moment vanishes (its exterior potential decays as exp(-beta R)).
    """
    if not _has_deformation(side):
        return [(side["mu"], side["th"], side["b"])]
    d, Q = side.get("d_sr"), side.get("Q_sr")
    if d is None:
        return [(side["mu"], side["th"], side["b_def"])]
    return [(side["mu"] + d, side["th"] + Q, side["b_def"]), (-d, -Q, side["b"])]


def _charges_vs_shell(src: dict, beta_shell: Tensor, r: Tensor, specs) -> dict:
    """sum_a c_a lambda_n / R^p of src's three charges (nucleus z, core -nc
    on bc, active -n on b) against a Slater(beta_shell) multipole."""
    nuc = one_center(beta_shell, r, specs)
    core = two_center(src["bc"], beta_shell, r, specs)
    act = two_center(src["b"], beta_shell, r, specs)
    return {key: src["z"] * nuc[key] - src["nc"] * core[key] - src["n"] * act[key]
            for key in specs}


MM_SPECS = ((5, 3), (3, 3), (7, 4), (5, 4), (9, 5), (7, 5), (5, 5))


def _monopole_energy(r: Tensor, i: dict, j: dict) -> Tensor:
    """q_i q_j / R minus the decaying complements of the three-charge sites."""
    bar1 = (("bar1", 1),)
    aa = two_center(i["b"], j["b"], r, bar1)
    ci_cj = two_center(i["bc"], j["b"], r, bar1)
    ci_j = two_center(i["b"], j["bc"], r, bar1)
    cc = two_center(i["bc"], j["bc"], r, bar1)
    pi_cj = one_center(j["b"], r, bar1)
    pj_ci = one_center(i["b"], r, bar1)
    pi_ccj = one_center(j["bc"], r, bar1)
    pj_cci = one_center(i["bc"], r, bar1)
    k = ("bar1", 1)
    bar = (-i["z"] * (j["nc"] * pi_ccj[k] + j["n"] * pi_cj[k])
           - j["z"] * (i["nc"] * pj_cci[k] + i["n"] * pj_ci[k])
           + i["nc"] * j["nc"] * cc[k] + i["nc"] * j["n"] * ci_cj[k]
           + i["n"] * j["nc"] * ci_j[k] + i["n"] * j["n"] * aa[k])
    return i["q"] * j["q"] / r - bar


def _elst_pairs_shells(r: Tensor, rhat: Tensor, i: dict, j: dict) -> Tensor:
    """``elst_pairs`` with each site's multipoles on their own Slater shells
    (``multipole_shells``): charges keep (z, nc/bc, n/b); every charge-multipole
    and multipole-multipole kernel takes the multipole shell's exponent."""
    energy = _monopole_energy(r, i, j)
    cs = ((3, 2), (5, 3))
    for mu, th, b in multipole_shells(j):  # i's charges vs j's multipoles
        k = _charges_vs_shell(i, b, r, cs)
        energy = energy - k[(3, 2)] * (mu * rhat).sum(-1) \
            + k[(5, 3)] * _bilinear(rhat, th, rhat)
    for mu, th, b in multipole_shells(i):  # j's charges vs i's multipoles
        k = _charges_vs_shell(j, b, r, cs)
        energy = energy + k[(3, 2)] * (mu * rhat).sum(-1) \
            + k[(5, 3)] * _bilinear(rhat, th, rhat)
    for mui, thi, bi in multipole_shells(i):
        ui, ti = (mui * rhat).sum(-1), _bilinear(rhat, thi, rhat)
        thi_r = torch.einsum("pab,pb->pa", thi, rhat)
        for muj, thj, bj in multipole_shells(j):
            aa = two_center(bi, bj, r, MM_SPECS)
            uj, tj = (muj * rhat).sum(-1), _bilinear(rhat, thj, rhat)
            thj_r = torch.einsum("pab,pb->pa", thj, rhat)
            energy = energy - (3.0 * aa[(5, 3)] * ui * uj
                               - aa[(3, 3)] * (mui * muj).sum(-1))
            energy = energy + (5.0 * aa[(7, 4)] * ui * tj
                               - 2.0 * aa[(5, 4)] * (mui * thj_r).sum(-1))
            energy = energy - (5.0 * aa[(7, 4)] * uj * ti
                               - 2.0 * aa[(5, 4)] * (muj * thi_r).sum(-1))
            energy = energy + (35.0 / 3.0 * aa[(9, 5)] * ti * tj
                               - 20.0 / 3.0 * aa[(7, 5)] * (thi_r * thj_r).sum(-1)
                               + 2.0 / 3.0 * aa[(5, 5)] * (thi * thj).sum((-1, -2)))
    return energy


def elst_pairs(r: Tensor, rhat: Tensor, i: dict, j: dict) -> Tensor:
    """Exact Coulomb energy (E_h) of two-shell Eq. 7 densities, per pair.

    ``i`` and ``j`` hold per-pair tensors: ``z`` (full nuclear charge),
    ``nc``/``bc`` (frozen core electrons and exponent), ``n``/``b`` (active
    electrons and valence exponent), ``q`` (net charge), ``mu`` (3,), ``th``
    (3, 3, traceless Buckingham).  ``rhat`` points from i to j.  Optional
    ``b_def`` (the l = 1, 2 deformation exponent) and ``d_sr``/``Q_sr`` (a
    zero-moment short-range shell) move the multipoles onto their own shells
    (``multipole_shells``); without them this is the single-exponent Eq. 7.
    """
    if _has_deformation(i) or _has_deformation(j):
        return _elst_pairs_shells(r, rhat, i, j)
    aa = two_center(i["b"], j["b"], r, ELST_AA_SPECS)
    # i's charges seen by j's cloud moments, and vice versa.
    pi_cj = one_center(j["b"], r, CHARGE_SPECS)
    pj_ci = one_center(i["b"], r, CHARGE_SPECS)
    ci_cj = two_center(i["bc"], j["b"], r, CHARGE_SPECS)
    ci_j = two_center(i["b"], j["bc"], r, CHARGE_SPECS)
    cc = two_center(i["bc"], j["bc"], r, (("bar1", 1),))
    pi_ccj = one_center(j["bc"], r, (("bar1", 1),))
    pj_cci = one_center(i["bc"], r, (("bar1", 1),))
    # monopole: q_i q_j / R minus the decaying complements (no cancellation)
    bar = (-i["z"] * (j["nc"] * pi_ccj[("bar1", 1)] + j["n"] * pi_cj[("bar1", 1)])
           - j["z"] * (i["nc"] * pj_cci[("bar1", 1)] + i["n"] * pj_ci[("bar1", 1)])
           + i["nc"] * j["nc"] * cc[("bar1", 1)]
           + i["nc"] * j["n"] * ci_cj[("bar1", 1)]
           + i["n"] * j["nc"] * ci_j[("bar1", 1)]
           + i["n"] * j["n"] * aa[("bar1", 1)])
    energy = i["q"] * j["q"] / r - bar

    def charges_vs(src, own_1c, core_2c, n):
        """sum_a c_a lambda_n / R^p of src's three charges vs the other cloud."""
        key = (n, 2) if n == 3 else (n, 3)
        return (src["z"] * own_1c[key] - src["nc"] * core_2c[key]
                - src["n"] * aa[key])

    qi3, qi5 = charges_vs(i, pi_cj, ci_cj, 3), charges_vs(i, pi_cj, ci_cj, 5)
    qj3, qj5 = charges_vs(j, pj_ci, ci_j, 3), charges_vs(j, pj_ci, ci_j, 5)
    mui, muj, thi, thj = i["mu"], j["mu"], i["th"], j["th"]
    ui, uj = (mui * rhat).sum(-1), (muj * rhat).sum(-1)
    ti, tj = _bilinear(rhat, thi, rhat), _bilinear(rhat, thj, rhat)
    energy = energy - qi3 * uj + qj3 * ui + qi5 * tj + qj5 * ti
    energy = energy - (3.0 * aa[(5, 3)] * ui * uj
                       - aa[(3, 3)] * (mui * muj).sum(-1))
    thj_r = torch.einsum("pab,pb->pa", thj, rhat)
    thi_r = torch.einsum("pab,pb->pa", thi, rhat)
    energy = energy + (5.0 * aa[(7, 4)] * ui * tj
                       - 2.0 * aa[(5, 4)] * (mui * thj_r).sum(-1))
    energy = energy - (5.0 * aa[(7, 4)] * uj * ti
                       - 2.0 * aa[(5, 4)] * (muj * thi_r).sum(-1))
    energy = energy + (35.0 / 3.0 * aa[(9, 5)] * ti * tj
                       - 20.0 / 3.0 * aa[(7, 5)] * (thi_r * thj_r).sum(-1)
                       + 2.0 / 3.0 * aa[(5, 5)] * (thi * thj).sum((-1, -2)))
    return energy


FIELD_SPECS = ((3, 2), (5, 3), (3, 3), (7, 4), (5, 4))


def permanent_field(r: Tensor, rhat: Tensor, beta_probe: Tensor, j: dict) -> Tensor:
    """Field (E_h/(e bohr)) of j's Eq. 7 density on a Slater(beta_probe) dipole.

    ``rhat`` points from source j to the probe k.  With ``b_def`` (and
    ``d_sr``/``Q_sr``) in ``j`` the moments' field uses their own shells.
    """
    if _has_deformation(j):
        return _permanent_field_shells(r, rhat, beta_probe, j)
    act = two_center(beta_probe, j["b"], r, FIELD_SPECS)
    core = two_center(beta_probe, j["bc"], r, ((3, 2),))
    nuc = one_center(beta_probe, r, ((3, 2),))
    charge = (j["z"] * nuc[(3, 2)] - j["nc"] * core[(3, 2)]
              - j["n"] * act[(3, 2)])
    mu, th = j["mu"], j["th"]
    ur = (mu * rhat).sum(-1)
    tr = _bilinear(rhat, th, rhat)
    th_r = torch.einsum("pab,pb->pa", th, rhat)
    field = charge[:, None] * rhat
    field = field + 3.0 * act[(5, 3)][:, None] * ur[:, None] * rhat \
        - act[(3, 3)][:, None] * mu
    field = field + 5.0 * act[(7, 4)][:, None] * tr[:, None] * rhat \
        - 2.0 * act[(5, 4)][:, None] * th_r
    return field


def _permanent_field_shells(r: Tensor, rhat: Tensor, beta_probe: Tensor,
                            j: dict) -> Tensor:
    act_0 = two_center(beta_probe, j["b"], r, ((3, 2),))
    core = two_center(beta_probe, j["bc"], r, ((3, 2),))
    nuc = one_center(beta_probe, r, ((3, 2),))
    charge = (j["z"] * nuc[(3, 2)] - j["nc"] * core[(3, 2)]
              - j["n"] * act_0[(3, 2)])
    field = charge[:, None] * rhat
    for mu, th, b in multipole_shells(j):
        act = two_center(beta_probe, b, r, ((5, 3), (3, 3), (7, 4), (5, 4)))
        ur = (mu * rhat).sum(-1)
        tr = _bilinear(rhat, th, rhat)
        th_r = torch.einsum("pab,pb->pa", th, rhat)
        field = field + 3.0 * act[(5, 3)][:, None] * ur[:, None] * rhat \
            - act[(3, 3)][:, None] * mu
        field = field + 5.0 * act[(7, 4)][:, None] * tr[:, None] * rhat \
            - 2.0 * act[(5, 4)][:, None] * th_r
    return field


def dipole_tensor(r: Tensor, rhat: Tensor, beta_k: Tensor, beta_l: Tensor) -> Tensor:
    """Gordon1 T^(2) (bohr^-3): (lambda3 I - 3 lambda5 Rhat Rhat)/R^3."""
    lam = two_center(beta_k, beta_l, r, ((3, 3), (5, 3)))
    eye = torch.eye(3, dtype=r.dtype, device=r.device)
    return (lam[(3, 3)][:, None, None] * eye
            - 3.0 * lam[(5, 3)][:, None, None] * rhat[:, :, None] * rhat[:, None, :])
