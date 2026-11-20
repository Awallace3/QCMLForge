"""Error-bound parameter tuners for Multipolar SPME and Dispersion PME.

Computes Ewald splitting parameters ``alpha`` and smooth FFT grid dimensions
``ns_mesh`` from a target accuracy bound and real-space cutoff, following
Essmann et al. (1995), Kolafa & Perram (1992), and Wang et al. (2016).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from ..lib.bsplines import next_smooth_size


@dataclass(frozen=True)
class MultipolePMEParams:
    """Tuned parameters for :class:`torch_pes.calculators.MultipolePME`."""

    alpha: float
    ns_mesh: tuple[int, int, int]
    order: int
    cutoff: float
    accuracy: float


@dataclass(frozen=True)
class DispersionPMEParams:
    """Tuned parameters for :class:`torch_pes.calculators.DispersionPME`."""

    alpha: float
    ns_mesh: tuple[int, int, int]
    order: int
    rank: int
    cutoff: float
    accuracy: float


def _solve_alpha_from_cutoff(cutoff: float, accuracy: float, max_rank: int) -> float:
    """Find minimal ``alpha`` so that the real-space multipole truncation
    factor ``exp(-(alpha * r_c)^2) * (1 + (alpha * r_c)^(2 * max_rank)) <= accuracy``."""
    # Monotone decreasing for x = alpha * r_c >= sqrt(max_rank); solve by bisection
    lo = math.sqrt(max(max_rank, 1))
    hi = max(lo + 1.0, math.sqrt(-math.log(max(accuracy, 1e-15))) + 4.0)

    def tail(x: float) -> float:
        return math.exp(-(x * x)) * (1.0 + x ** (2 * max_rank))

    while tail(hi) > accuracy:
        hi *= 1.5
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if tail(mid) > accuracy:
            lo = mid
        else:
            hi = mid
    return hi / cutoff


def tune_multipole_pme(
    cell: Tensor,
    cutoff: float,
    accuracy: float = 1e-6,
    max_rank: int = 2,
    order: int = 6,
) -> MultipolePMEParams:
    """Choose ``(alpha, ns_mesh, order)`` for Multipolar SPME from a target
    relative accuracy bound and real-space ``cutoff``.

    Parameters
    ----------
    cell : Tensor, shape ``(3, 3)``
        Periodic box vectors (same length unit as ``cutoff``, e.g. bohr).
    cutoff : float
        Real-space cutoff distance.
    accuracy : float
        Target relative error tolerance (e.g. ``1e-6`` for MD forces).
    max_rank : int
        Maximum multipole rank: ``0`` (charges), ``1`` (dipoles), or ``2``
        (quadrupoles, FF-2 default).
    order : int
        Cardinal B-spline order (default ``6`` for quadrupoles).
    """
    if cutoff <= 0:
        raise ValueError(f"cutoff must be positive, got {cutoff}")
    if accuracy <= 0 or accuracy >= 1:
        raise ValueError(f"accuracy must be in (0, 1), got {accuracy}")
    if max_rank not in (0, 1, 2):
        raise ValueError(f"max_rank must be 0, 1, or 2, got {max_rank}")

    alpha = _solve_alpha_from_cutoff(cutoff, accuracy, max_rank=max_rank)

    # k-space cutoff: exp(-k_max^2 / (4 alpha^2)) ~ accuracy
    # Also ensure B-spline aliasing error (h * alpha)^order is small:
    # empirical SPME bound h <= 1 / (c * alpha) where c grows mildly with -log10(acc)
    # Using k_max = 2 * alpha * sqrt(-ln(accuracy)) -> Nyquist spacing h_nyq = pi / k_max
    # Add B-spline oversampling factor depending on order and max_rank
    k_factor = math.sqrt(-math.log(max(accuracy, 1e-15)))
    # Oversampling factor ensures B-spline approximation error <= accuracy for rank L
    # For order 6, rank 2, accuracy 1e-6: oversample ~ 1.35 gives h ~ 1/(3.2 * alpha)
    oversample = 1.0 + 0.12 * (max_rank + 1) * max(1.0, 6.0 / order)
    target_spacing = math.pi / (2.0 * alpha * k_factor * oversample)

    lengths = torch.linalg.norm(cell, dim=1)
    ns = tuple(
        next_smooth_size(int(math.ceil(float(L) / target_spacing)), even=True)
        for L in lengths
    )
    return MultipolePMEParams(
        alpha=alpha,
        ns_mesh=(ns[0], ns[1], ns[2]),
        order=int(order),
        cutoff=float(cutoff),
        accuracy=float(accuracy),
    )


def tune_dispersion_pme(
    cell: Tensor,
    cutoff: float,
    accuracy: float = 1e-5,
    order: int = 4,
    rank: int = 4,
) -> DispersionPMEParams:
    """Choose ``(alpha, ns_mesh, order, rank)`` for :math:`r^{-6}` Dispersion PME."""
    if cutoff <= 0:
        raise ValueError(f"cutoff must be positive, got {cutoff}")
    if accuracy <= 0 or accuracy >= 1:
        raise ValueError(f"accuracy must be in (0, 1), got {accuracy}")

    # Real-space r^-6 damping: exp(-x^2) * (1 + x^2 + 0.5 x^4) <= accuracy
    lo, hi = 1.0, math.sqrt(-math.log(max(accuracy, 1e-15))) + 3.0

    def tail6(x: float) -> float:
        x2 = x * x
        return math.exp(-x2) * (1.0 + x2 + 0.5 * x2 * x2)

    while tail6(hi) > accuracy:
        hi *= 1.5
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if tail6(mid) > accuracy:
            lo = mid
        else:
            hi = mid
    alpha = hi / cutoff

    # Dispersion r^-6 k-space needs enough mesh resolution for B-spline aliasing
    k_factor = math.sqrt(-math.log(max(accuracy, 1e-15)))
    oversample = 1.15 * max(1.0, 5.0 / order)
    target_spacing = math.pi / (2.0 * alpha * k_factor * oversample)
    lengths = torch.linalg.norm(cell, dim=1)
    ns = tuple(
        next_smooth_size(int(math.ceil(float(L) / target_spacing)), even=True)
        for L in lengths
    )
    return DispersionPMEParams(
        alpha=alpha,
        ns_mesh=(ns[0], ns[1], ns[2]),
        order=int(order),
        rank=int(rank),
        cutoff=float(cutoff),
        accuracy=float(accuracy),
    )


__all__ = [
    "DispersionPMEParams",
    "MultipolePMEParams",
    "tune_dispersion_pme",
    "tune_multipole_pme",
]
