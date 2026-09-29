"""Feature-producer-independent positive exponent-anisotropic exchange."""

from __future__ import annotations

import torch


def slater_exponent_exchange(
    a_i: torch.Tensor,
    a_j: torch.Tensor,
    b_i: torch.Tensor,
    b_j: torch.Tensor,
    distance: torch.Tensor,
    s_i: torch.Tensor,
    s_j: torch.Tensor,
) -> torch.Tensor:
    """Evaluate positive Slater-overlap exchange with directional decay.

    Inputs broadcast elementwise. ``a`` is in sqrt(kcal/mol), ``b`` in
    inverse angstrom, distance in angstrom, and ``s`` is a dimensionless
    harmonic contraction. The caller supplies positive A/B and nonnegative
    distances. Both the polynomial and exponential use the anisotropic x;
    anisotropy is not an external amplitude multiplier.

    This pure tensor kernel neither constructs molecular frames nor imposes
    an intermonomer cutoff. Validation belongs at the model/input boundary.
    """
    # Taking roots before multiplying avoids underflow of B_i*B_j to zero
    # (and NaN sqrt gradients) for representable small positive B.
    decay = b_i.sqrt() * b_j.sqrt() * torch.exp(0.5 * (s_i + s_j))
    x = distance * decay
    return a_i * a_j * (1 + x + x.square() / 3) * torch.exp(-x)
