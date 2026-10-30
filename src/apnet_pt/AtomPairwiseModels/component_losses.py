"""Configurable component-wise losses for the pairwise SAPT models.

Both pairwise harnesses default to an unweighted MSE over the four SAPT
components in absolute kcal/mol.  On the spec-11 corpus that objective is
dominated by ionic dimers: they hold ~95% of the electrostatics and ~91% of the
induction squared-error mass, so the optimiser trades neutral-dimer accuracy for
ionic accuracy and the S66-like slice regresses.

Every loss here has the ``(preds, labels) -> scalar`` signature that
``APNet2Model.train`` and ``APNet3D3_AtomType_Model.train`` already thread
through as ``criterion``, so they are drop-in alternatives that need no batch
metadata.  Per-sample reweighting is deliberately *not* expressed here -- the
hook has no access to the batch -- and is delivered by dataset construction
instead.

Notes
-----
``preds`` and ``labels`` are ``(n_dimer, n_component)`` tensors ordered
``(elst, exch, ind, disp)``.
"""

import functools
from typing import Callable, Sequence

import torch

__all__ = [
    "COMPONENT_LOSS_NAMES",
    "build_component_loss",
    "component_huber",
    "component_mse",
    "component_relative_mse",
    "component_weighted_mse",
    "per_component_mse",
    "validate_loss_route",
]


def component_mse(preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Unweighted mean squared error over every component.

    This reproduces the loss both harnesses inline today and is the baseline
    every other option is compared against.
    """
    return torch.mean(torch.square(preds - labels))


def component_huber(
    preds: torch.Tensor, labels: torch.Tensor, delta: float = 1.0
) -> torch.Tensor:
    """Mean Huber loss, quadratic within ``delta`` and linear beyond it.

    The linear tail caps the per-row gradient at ``delta``, which is the
    cheapest way to stop high-magnitude ionic dimers from owning the update
    while still fitting them.
    """
    errors = preds - labels
    abs_errors = torch.abs(errors)
    quadratic = 0.5 * torch.square(errors)
    linear = delta * (abs_errors - 0.5 * delta)
    return torch.mean(torch.where(abs_errors <= delta, quadratic, linear))


def component_relative_mse(
    preds: torch.Tensor, labels: torch.Tensor, eps: float = 1.0
) -> torch.Tensor:
    """Magnitude-normalised MSE, ``err**2 / (label**2 + eps**2)``.

    ``eps`` (kcal/mol) floors the denominator so that near-zero targets do not
    dominate.  Larger ``eps`` interpolates back towards :func:`component_mse`.
    """
    errors = preds - labels
    return torch.mean(torch.square(errors) / (torch.square(labels) + eps**2))


def component_weighted_mse(
    preds: torch.Tensor,
    labels: torch.Tensor,
    weights: Sequence[float] = (1.0, 1.0, 1.0, 1.0),
) -> torch.Tensor:
    """MSE with an explicit per-component weight vector ``(elst, exch, ind, disp)``.

    Raises
    ------
    ValueError
        If ``weights`` does not have one entry per predicted component; a model
        built with ``no_disp_nn`` predicts three, ``(elst, exch, ind)``.
    """
    n_components = preds.shape[-1]
    if len(weights) != n_components:
        raise ValueError(
            f"component_weighted_mse got {len(weights)} weights for "
            f"{n_components} predicted components; pass one per component "
            "in (elst, exch, ind, disp) order, omitting disp under no_disp_nn"
        )
    w = torch.as_tensor(weights, dtype=preds.dtype, device=preds.device)
    return torch.mean(w * torch.square(preds - labels))


def per_component_mse(
    comp_errors: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Each component's mean squared error, ``(elst, exch, ind, disp)``.

    ``comp_errors`` is ``(n_dimer, 3)`` under ``no_disp_nn``; the missing
    dispersion term is then a zero on the errors' own device and dtype.
    """
    mse = torch.mean(torch.square(comp_errors), dim=0)
    if mse.numel() == 3:
        mse = torch.cat((mse, mse.new_zeros(1)))
    elst, exch, ind, disp = mse.unbind()
    return elst, exch, ind, disp


def validate_loss_route(loss_fn: Callable | None, *, transfer_learning: bool) -> None:
    """Refuse a selected component loss on the transfer-learning route.

    Transfer learning fits one summed energy per dimer, so a loss here would get
    ``(n_dimer,)`` totals instead of ``(n_dimer, n_component)`` rows, compared
    against labels the route never squeezes: the relative and Huber losses could
    broadcast silently and the weighted loss fails mid-run.  Only the default
    criterion is defined there.

    Raises
    ------
    ValueError
        If ``loss_fn`` is set and ``transfer_learning`` is true.
    """
    if loss_fn is not None and transfer_learning:
        raise ValueError(
            "a component loss cannot be combined with transfer_learning=True; "
            "the transfer route fits a single summed energy per dimer"
        )


_COMPONENT_LOSSES: dict[str, Callable[..., torch.Tensor]] = {
    "component_mse": component_mse,
    "component_huber": component_huber,
    "component_relative_mse": component_relative_mse,
    "component_weighted_mse": component_weighted_mse,
}

COMPONENT_LOSS_NAMES = tuple(sorted(_COMPONENT_LOSSES))


def build_component_loss(name: str, **kwargs) -> Callable[
    [torch.Tensor, torch.Tensor], torch.Tensor
]:
    """Return a ``(preds, labels) -> scalar`` callable for a registered loss.

    Parameters
    ----------
    name
        One of :data:`COMPONENT_LOSS_NAMES`.
    **kwargs
        Hyper-parameters forwarded to the underlying loss, such as ``delta``
        for ``component_huber`` or ``weights`` for ``component_weighted_mse``.

    Raises
    ------
    ValueError
        If ``name`` is not a registered loss.
    """
    if name not in _COMPONENT_LOSSES:
        raise ValueError(
            f"unknown component loss {name!r}; expected one of "
            f"{', '.join(COMPONENT_LOSS_NAMES)}"
        )
    loss = _COMPONENT_LOSSES[name]
    if not kwargs:
        return loss
    # functools.partial, not a closure: the DDP path ships the criterion through
    # mp.spawn, which pickles its arguments.
    return functools.partial(loss, **kwargs)
