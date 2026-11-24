"""Cardinal B-splines and derivatives for Smooth Particle Mesh Ewald (SPME).

Implements vectorized, ``torch.compile``-friendly evaluation of Cardinal
B-splines :math:`M_p(u)` and their first two derivatives :math:`M_p'(u)`,
:math:`M_p''(u)` via exact Cox--de Boor recurrences (Essmann et al.,
*J. Chem. Phys.* **103**, 8577 (1995); Sagui, Pedersen & Darden,
*J. Chem. Phys.* **120**, 73 (2004)), together with the exact Fourier modulus
factors :math:`|b_a(m_a)|^2`.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


def next_smooth_size(n: int, even: bool = True) -> int:
    """Return the smallest integer ``>= max(n, 2)`` whose prime factors lie
    in ``{2, 3, 5}`` (and even when ``even=True``), for fast 3D FFTs."""
    m = max(int(n), 2)
    if even and m % 2 != 0:
        m += 1
    while True:
        x = m
        for p in (2, 3, 5):
            while x % p == 0:
                x //= p
        if x == 1:
            return m
        m += 2 if even else 1


def bspline_weights(
    delta: Tensor,
    order: int,
    deriv_order: int = 2,
) -> tuple[Tensor, ...]:
    """Evaluate Cardinal B-spline :math:`M_p(\\delta + k)` and derivatives.

    Parameters
    ----------
    delta : Tensor, shape ``(...,)``
        Fractional grid offsets in ``[0, 1)``.
    order : int
        B-spline order ``p`` (``p >= 2 + deriv_order``, e.g. ``p = 4`` for
        charges/dipoles and ``p = 6`` for quadrupoles so ``M_p''`` is ``C^2``
        and forces ``M_p'''`` are ``C^1``).
    deriv_order : int
        Highest derivative to return: ``0``, ``1``, or ``2``.

    Returns
    -------
    tuple of Tensor
        ``(w0,)`` for ``deriv_order=0``, ``(w0, w1)`` for ``1``, or
        ``(w0, w1, w2)`` for ``2``, each of shape ``(*delta.shape, order)``,
        where slot ``k`` holds :math:`M_p^{(d)}(\\delta + k)` for
        ``k = 0 .. order - 1``.
    """
    if deriv_order not in (0, 1, 2):
        raise ValueError(f"deriv_order must be 0, 1, or 2, got {deriv_order}")
    if order < 2 + deriv_order:
        raise ValueError(
            f"B-spline order must be >= {2 + deriv_order} for "
            f"deriv_order={deriv_order}, got {order}"
        )

    # Order 2: M_2(delta) = delta, M_2(delta + 1) = 1 - delta
    m = [delta, 1.0 - delta]
    m_prev2: list[Tensor] | None = None
    m_prev1: list[Tensor] | None = None

    for n in range(3, order + 1):
        if n == order - 1:
            m_prev2 = m
        if n == order:
            m_prev1 = m
        inv = 1.0 / (n - 1)
        nxt: list[Tensor] = [delta * inv * m[0]]
        for k in range(1, n - 1):
            nxt.append(((delta + k) * m[k] + (n - k - delta) * m[k - 1]) * inv)
        nxt.append((1.0 - delta) * inv * m[n - 2])
        m = nxt

    if order == 3:
        m_prev1 = [delta, 1.0 - delta]
    if order == 4:
        m_prev2 = [delta, 1.0 - delta]

    w0 = torch.stack(m, dim=-1)
    if deriv_order == 0:
        return (w0,)

    assert m_prev1 is not None
    d1: list[Tensor] = [m_prev1[0]]
    for k in range(1, order - 1):
        d1.append(m_prev1[k] - m_prev1[k - 1])
    d1.append(-m_prev1[order - 2])
    w1 = torch.stack(d1, dim=-1)
    if deriv_order == 1:
        return (w0, w1)

    assert m_prev2 is not None
    # M_p''(u) = M_{p-2}(u) - 2 M_{p-2}(u - 1) + M_{p-2}(u - 2)
    z = torch.zeros_like(delta)
    pad = [z, z] + m_prev2 + [z, z]
    d2 = [pad[k + 2] - 2.0 * pad[k + 1] + pad[k] for k in range(order)]
    w2 = torch.stack(d2, dim=-1)
    return (w0, w1, w2)


def bspline_modulus_sq_1d(
    n_grid: int,
    order: int,
    rfft: bool = False,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> Tensor:
    """Compute :math:`|b(m)|^2` along one 1D FFT axis of length ``n_grid``.

    Parameters
    ----------
    n_grid : int
        Number of grid points along this axis.
    order : int
        B-spline order ``p``.
    rfft : bool
        If ``True``, return only the non-negative frequencies ``0 .. n_grid//2``
        matching :func:`torch.fft.rfftfreq`.
    dtype, device
        Output dtype and device.

    Returns
    -------
    Tensor
        Real tensor of shape ``(n_grid // 2 + 1,)`` if ``rfft`` else ``(n_grid,)``.
    """
    # Evaluate M_p(1), ..., M_p(p - 1) using delta = 0
    zero = torch.zeros((), dtype=torch.float64, device=device)
    m_vals = bspline_weights(zero, order, deriv_order=0)[0][1:]  # (p - 1,)

    if rfft:
        freqs = torch.arange(n_grid // 2 + 1, dtype=torch.float64, device=device)
    else:
        freqs = (
            torch.fft.fftfreq(n_grid, dtype=torch.float64, device=device) * n_grid
        )

    ks = torch.arange(order - 1, dtype=torch.float64, device=device)
    angle = (2.0 * math.pi / n_grid) * freqs[:, None] * ks[None, :]
    re = (m_vals[None, :] * torch.cos(angle)).sum(dim=-1)
    im = (m_vals[None, :] * torch.sin(angle)).sum(dim=-1)
    denom = re * re + im * im
    safe = torch.where(denom > 1e-30, denom, torch.ones_like(denom))
    mod_sq = torch.where(denom > 1e-30, 1.0 / safe, torch.zeros_like(denom))
    return mod_sq.to(dtype=dtype)


def bspline_modulus_sq_3d(
    ns_mesh: tuple[int, int, int],
    order: int,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> Tensor:
    """3D separable B-spline modulus factor :math:`|b_1|^2 |b_2|^2 |b_3|^2`
    on the ``rfftn`` grid of shape ``(K1, K2, K3 // 2 + 1)``."""
    b1 = bspline_modulus_sq_1d(ns_mesh[0], order, rfft=False, dtype=dtype, device=device)
    b2 = bspline_modulus_sq_1d(ns_mesh[1], order, rfft=False, dtype=dtype, device=device)
    b3 = bspline_modulus_sq_1d(ns_mesh[2], order, rfft=True, dtype=dtype, device=device)
    return b1[:, None, None] * b2[None, :, None] * b3[None, None, :]


__all__ = [
    "bspline_modulus_sq_1d",
    "bspline_modulus_sq_3d",
    "bspline_weights",
    "next_smooth_size",
]
