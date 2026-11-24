"""Ewald real-space radial (Boys) functions and multipolar pair kernels.

Provides numerically stable, differentiable, ``torch.compile``-friendly
implementations of:

* Ewald ``erfc`` and ``erf`` Boys functions :math:`B_0 \\dots B_4`, where
  :math:`B_l^{\\mathrm{erf}}(r; \\alpha)` uses an 18-term Horner series for
  :math:`\\alpha r \\le 1` so there is zero cancellation even at ``r = 0``
  (important for bonded intramonomer exclusions).
* Exact rank-0/1/2 point-multipole pair energy and electric field matching
  FF-2's traceless Buckingham quadrupole convention
  :math:`\\hat{L}_i = q_i + \\boldsymbol{\\mu}_i \\cdot \\nabla_i + \\frac{1}{3} \\boldsymbol{\\Theta}_i : \\nabla_i^2`.
* Exact Ewald :math:`r^{-6}` dispersion real-space kernels
  :math:`\\psi^{(6)}_{\\mathrm{dir}}(r; \\alpha)` and
  :math:`\\psi^{(6)}_{\\mathrm{rec}}(r; \\alpha)` (also cancellation-free at
  small ``r``).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

_INV_SQRT_PI = 1.0 / math.sqrt(math.pi)
_BOYS_SERIES_TERMS = 18
_DISP_SERIES_TERMS = 16


def _dot(a: Tensor, b: Tensor) -> Tensor:
    return (a * b).sum(dim=-1)


def _mat_vec(T: Tensor, v: Tensor) -> Tensor:
    return (T * v[..., None, :]).sum(dim=-1)


def ewald_boys_erfc(r: Tensor, alpha: float | Tensor) -> tuple[Tensor, ...]:
    """Ewald direct-space Boys functions :math:`(B_0, B_1, B_2, B_3, B_4)` for
    :math:`B_0(r) = \\mathrm{erfc}(\\alpha r) / r` satisfying
    :math:`d B_l / dr = -r B_{l+1}`.

    Parameters
    ----------
    r : Tensor
        Strictly positive pair distances.
    alpha : float or Tensor
        Ewald splitting parameter (inverse length).
    """
    ar = alpha * r
    inv_r2 = 1.0 / (r * r)
    c = (2.0 * _INV_SQRT_PI) * alpha * torch.exp(-(ar * ar))
    a2 = 2.0 * alpha * alpha
    b0 = torch.erfc(ar) / r
    b1 = (b0 + c) * inv_r2
    b2 = (3.0 * b1 + a2 * c) * inv_r2
    b3 = (5.0 * b2 + (a2 * a2) * c) * inv_r2
    b4 = (7.0 * b3 + (a2 * a2 * a2) * c) * inv_r2
    return (b0, b1, b2, b3, b4)


def ewald_boys_point(r: Tensor) -> tuple[Tensor, ...]:
    """Unscreened point-multipole Boys functions
    :math:`B_l^{\\mathrm{point}}(r) = (2l - 1)!! / r^{2l + 1}` for
    ``l = 0 .. 4``."""
    inv_r = 1.0 / r
    inv_r2 = inv_r * inv_r
    b0 = inv_r
    b1 = b0 * inv_r2
    b2 = 3.0 * b1 * inv_r2
    b3 = 5.0 * b2 * inv_r2
    b4 = 7.0 * b3 * inv_r2
    return (b0, b1, b2, b3, b4)


def ewald_boys_erf(r: Tensor, alpha: float | Tensor) -> tuple[Tensor, ...]:
    """Ewald long-range (Gaussian-screened) Boys functions
    :math:`B_l^{\\mathrm{erf}}(r) = B_l^{\\mathrm{point}}(r) - B_l^{\\mathrm{erfc}}(r)`
    for ``l = 0 .. 4``.

    Cancellation-free and smooth for all :math:`r \\ge 0`: uses an 18-term
    Horner series for :math:`\\alpha r \\le 1` and the exact difference for
    :math:`\\alpha r > 1`.
    """
    ar = alpha * r
    small = ar <= 1.0
    # Safe branch arguments so autograd never sees NaN on the unselected branch
    x2_s = torch.where(small, ar * ar, torch.zeros_like(ar))
    r_l = torch.where(small, torch.ones_like(r) / alpha, r)

    # Large-ar branch: point - erfc
    pt_l = ewald_boys_point(r_l)
    erfc_l = ewald_boys_erfc(r_l, alpha)
    large_vals = [p - e for p, e in zip(pt_l, erfc_l)]

    # Small-ar branch: B_l^erf(r) = (2^{l+1} alpha^{2l+1} / sqrt(pi)) *
    # sum_{m=0}^{M-1} (-x^2)^m / (m! (2m + 2l + 1))
    # Compute shared factorials via Horner in -x^2
    neg_x2 = -x2_s
    small_vals = []
    prefac = (2.0 * _INV_SQRT_PI) * alpha
    a2 = 2.0 * alpha * alpha
    for ell in range(5):
        acc = torch.full_like(
            x2_s,
            1.0 / (math.factorial(_BOYS_SERIES_TERMS - 1) * (2 * (_BOYS_SERIES_TERMS - 1) + 2 * ell + 1)),
        )
        for m in range(_BOYS_SERIES_TERMS - 2, -1, -1):
            c_m = 1.0 / (math.factorial(m) * (2 * m + 2 * ell + 1))
            acc = acc * neg_x2 + c_m
        small_vals.append(prefac * acc)
        prefac = prefac * a2

    return tuple(
        torch.where(small, sv, lv) for sv, lv in zip(small_vals, large_vals)
    )


def multipole_pair_energy(
    r: Tensor,
    rhat: Tensor,
    q_i: Tensor,
    mu_i: Tensor | None,
    th_i: Tensor | None,
    q_j: Tensor,
    mu_j: Tensor | None,
    th_j: Tensor | None,
    boys: tuple[Tensor, ...],
) -> Tensor:
    """Rank-0/1/2 point-multipole interaction energy per pair ``(i, j)``.

    Uses FF-2's exact conventions: ``rhat = (R_j - R_i) / r`` points from
    ``i`` to ``j``, and ``th`` is the symmetric traceless Buckingham quadrupole
    (so :math:`\\tilde{Q} = \\frac{1}{3} \\Theta`).  Works with any Boys tuple
    ``(B_0, B_1, B_2, B_3, B_4)`` (point, ``erfc``, or ``erf``).
    """
    b0, b1, b2, b3, b4 = boys
    r2 = r * r
    rvec = r[:, None] * rhat

    # Monopole-monopole
    e = q_i * q_j * b0

    has_mu = mu_i is not None and mu_j is not None
    has_th = th_i is not None and th_j is not None

    if has_mu:
        ur_i = _dot(mu_i, rvec)
        ur_j = _dot(mu_j, rvec)
        # Charge-dipole: (-q_i (mu_j . r) + q_j (mu_i . r)) B_1
        e = e + (q_j * ur_i - q_i * ur_j) * b1
        # Dipole-dipole: -(mu_i . r)(mu_j . r) B_2 + (mu_i . mu_j) B_1
        e = e + _dot(mu_i, mu_j) * b1 - ur_i * ur_j * b2

    if has_th:
        # Use Q_tilde = th / 3 so every contraction matches FF-2 exactly
        qi_r = _mat_vec(th_i, rvec) / 3.0
        qj_r = _mat_vec(th_j, rvec) / 3.0
        rq_i = _dot(qi_r, rvec)
        rq_j = _dot(qj_r, rvec)
        # Charge-quadrupole: (q_i (r^T Q_j r) + q_j (r^T Q_i r)) B_2
        e = e + (q_i * rq_j + q_j * rq_i) * b2

        if has_mu:
            # Dipole-quadrupole:
            # + (mu_i . r)(r^T Q_j r) B_3 - 2 (mu_i . (Q_j r)) B_2
            # - (mu_j . r)(r^T Q_i r) B_3 + 2 (mu_j . (Q_i r)) B_2
            e = e + (ur_i * rq_j - ur_j * rq_i) * b3 - 2.0 * (
                _dot(mu_i, qj_r) - _dot(mu_j, qi_r)
            ) * b2

        # Quadrupole-quadrupole:
        # (r^T Q_i r)(r^T Q_j r) B_4 - 4 (Q_i r).(Q_j r) B_3 + 2 (Q_i : Q_j) B_2
        qq_trace = (th_i * th_j).sum(dim=(-1, -2)) / 9.0
        e = e + rq_i * rq_j * b4 - 4.0 * _dot(qi_r, qj_r) * b3 + 2.0 * qq_trace * b2

    return e


def multipole_pair_field(
    r: Tensor,
    rhat: Tensor,
    q_src: Tensor,
    mu_src: Tensor | None,
    th_src: Tensor | None,
    boys: tuple[Tensor, ...],
) -> Tensor:
    """Electric field at probe due to source multipoles ``(q, mu, th)``.

    ``rhat = (R_probe - R_src) / r`` points from source to probe (matching
    :func:`apnet_pt.ff2.physics.permanent_field`).
    """
    _, b1, b2, b3, _ = boys
    rvec = r[:, None] * rhat
    # Charge field: q_src * B_1 * rvec
    f = (q_src * b1)[:, None] * rvec
    if mu_src is not None:
        # Dipole field: (mu_src . rvec) B_2 rvec - B_1 mu_src
        ur = _dot(mu_src, rvec)
        f = f + (ur * b2)[:, None] * rvec - b1[:, None] * mu_src
    if th_src is not None:
        # Quadrupole field (Q = th / 3):
        # (r^T Q r) B_3 rvec - 2 B_2 (Q rvec)
        qr = _mat_vec(th_src, rvec) / 3.0
        rqr = _dot(qr, rvec)
        f = f + (rqr * b3)[:, None] * rvec - (2.0 * b2)[:, None] * qr
    return f


def dipole_tensor_coeffs(r: Tensor, boys: tuple[Tensor, ...]) -> tuple[Tensor, Tensor]:
    """Return ``(l33, l53)`` for FF-2's polarization dipole tensor
    :math:`T^{\\mathrm{pol}}_{kl} = l_{33} I - 3 l_{53} \\hat{r}\\hat{r}^T`.

    Matches :func:`apnet_pt.ff2.physics.polarization` where in the point limit
    ``l33 = l53 = r^-3``.
    """
    _, b1, b2, _, _ = boys
    l33 = b1
    l53 = (r * r / 3.0) * b2
    return l33, l53


def dispersion_r6_erfc(r: Tensor, alpha: float | Tensor) -> Tensor:
    """Direct-space Ewald :math:`r^{-6}` kernel:
    :math:`\\psi^{(6)}_{\\mathrm{dir}}(r) = e^{-x}(1 + x + x^2/2) / r^6`,
    :math:`x = \\alpha^2 r^2`."""
    x = (alpha * r) ** 2
    r6 = (r * r) ** 3
    return torch.exp(-x) * (1.0 + x + 0.5 * x * x) / r6


def dispersion_r6_erf(r: Tensor, alpha: float | Tensor) -> Tensor:
    """Long-range (reciprocal-space) Ewald :math:`r^{-6}` kernel:
    :math:`\\psi^{(6)}_{\\mathrm{rec}}(r) = [1 - e^{-x}(1 + x + x^2/2)] / r^6`,
    :math:`x = \\alpha^2 r^2`.

    Cancellation-free for all :math:`r \\ge 0`: for ``x <= 0.5`` uses the
    strictly positive series
    :math:`\\alpha^6 e^{-x} \\sum_{m=0}^{15} x^m / (m + 3)!`.
    """
    x = (alpha * r) ** 2
    small = x <= 0.5
    xs = torch.where(small, x, torch.zeros_like(x))
    xl = torch.where(small, torch.ones_like(x), x)
    rl = torch.where(small, torch.ones_like(r) / alpha, r)

    acc = torch.full_like(xs, 1.0 / math.factorial(_DISP_SERIES_TERMS + 2))
    for m in range(_DISP_SERIES_TERMS - 2, -1, -1):
        acc = acc * xs + 1.0 / math.factorial(m + 3)
    val_small = (alpha ** 6) * torch.exp(-xs) * acc

    r6_l = (rl * rl) ** 3
    val_large = (1.0 - torch.exp(-xl) * (1.0 + xl + 0.5 * xl * xl)) / r6_l
    return torch.where(small, val_small, val_large)


__all__ = [
    "dipole_tensor_coeffs",
    "dispersion_r6_erf",
    "dispersion_r6_erfc",
    "ewald_boys_erf",
    "ewald_boys_erfc",
    "ewald_boys_point",
    "multipole_pair_energy",
    "multipole_pair_field",
]
