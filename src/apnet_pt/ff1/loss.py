"""FF-1 training objective (proposal Eq. 22)."""

from __future__ import annotations

import torch
from torch import Tensor

DEFAULTS = {"delta_e": 1.0, "w_tot": 0.5, "delta_rel": 0.05, "tau_ind": 0.25,
            "tau_disp": 0.25, "w_ind_rel": 1.0, "w_disp_rel": 1.0, "w_nu": 1e-3}


def _huber(x: Tensor, delta: float) -> Tensor:
    return torch.nn.functional.huber_loss(x, torch.zeros_like(x), delta=delta,
                                          reduction="none")


def ff1_loss(pred: Tensor, target: Tensor, nu: Tensor | None = None,
             **options) -> tuple[Tensor, dict]:
    """Eq. 22 averaged over dimers; ``pred``/``target`` are (n, 4) kcal/mol
    in [elst, exch, ind, disp] order.  Returns (loss, per-term values)."""
    o = {**DEFAULTS, **options}
    target = target.to(pred.dtype)
    n = pred.shape[0]
    comp = _huber(pred - target, o["delta_e"]).sum(1).mean()
    total = o["w_tot"] * _huber(pred.sum(1) - target.sum(1), o["delta_e"]).mean()
    rel = pred.new_zeros(())
    for col, key in ((2, "ind"), (3, "disp")):
        r = (pred[:, col] - target[:, col]) / (target[:, col].abs() + o[f"tau_{key}"])
        rel = rel + o[f"w_{key}_rel"] * _huber(r, o["delta_rel"]).mean()
    reg = pred.new_zeros(())
    if nu is not None:
        reg = o["w_nu"] * nu.to(pred.dtype).square().sum() / n
    loss = comp + total + rel + reg
    return loss, {"components": comp.detach(), "total": total.detach(),
                  "relative": rel.detach(), "nu_l2": reg.detach()}
