"""TensorFlow-parity controls shared by APNet2 implementations."""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn

ParameterInitialization = Literal["pytorch", "tensorflow"]
CheckpointMetric = Literal["component_mse", "total_mae"]

_PARAMETER_INITIALIZATIONS = {"pytorch", "tensorflow"}
_CHECKPOINT_METRICS = {"component_mse", "total_mae"}


def validate_parameter_initialization(value: str) -> ParameterInitialization:
    """Validate and return an APNet2 parameter-initialization policy."""
    if value not in _PARAMETER_INITIALIZATIONS:
        raise ValueError(
            "parameter initialization must be one of "
            f"{sorted(_PARAMETER_INITIALIZATIONS)}, got {value!r}"
        )
    return value  # type: ignore[return-value]


class APNetLazyLinear(nn.LazyLinear):
    """Lazy linear layer supporting Keras-compatible Glorot initialization."""

    def __init__(
        self,
        out_features: int,
        parameter_initialization: ParameterInitialization,
    ) -> None:
        self.parameter_initialization = validate_parameter_initialization(
            parameter_initialization
        )
        super().__init__(out_features)

    def reset_parameters(self) -> None:
        super().reset_parameters()
        if (
            getattr(self, "parameter_initialization", "pytorch") == "tensorflow"
            and self.weight.numel() > 0
        ):
            nn.init.xavier_uniform_(self.weight)
            if self.bias is not None:
                nn.init.zeros_(self.bias)


def initialize_tensorflow_defaults(module: nn.Module) -> None:
    """Apply TensorFlow/Keras 2.x defaults to initialized dense/embedding layers."""
    if isinstance(module, nn.LazyLinear):
        return
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.uniform_(module.weight, -0.05, 0.05)


def elst_uQ_QQ(dR, dR_xyz, oodR, delta, muA, muB, quadA, quadB):
    """Dipole-quadrupole and quadrupole-quadrupole electrostatics, in a.u.

    ``T3``/``T4`` reproduce ``multipoles.py::T_cart`` at ``593d655^`` verbatim,
    unsymmetrised ``T3`` included; its extra terms vanish against a traceless Q.
    """
    dR2 = dR * dR

    Rdd = torch.einsum("xy,zw->xyzw", dR_xyz, delta)
    T3 = -1.0 * torch.einsum(
        "x,xyzw->xyzw",
        oodR**7,
        15.0 * torch.einsum("xy,xz,xw->xyzw", dR_xyz, dR_xyz, dR_xyz)
        - 3.0
        * torch.einsum(
            "x,xyzw->xyzw",
            dR2,
            Rdd + Rdd.permute(0, 2, 1, 3) + Rdd.permute(0, 3, 1, 2),
        ),
    )
    uQ = torch.einsum("xy,xzw->xyzw", muA, quadB) - torch.einsum(
        "xy,xzw->xyzw", muB, quadA
    )
    E_uQ = (-1.0 / 3.0) * torch.einsum("xyzw,xyzw->x", T3, uQ)

    RRdd = torch.einsum("xy,xz,wv->xyzwv", dR_xyz, dR_xyz, delta)
    dddd = torch.einsum("yz,wv->yzwv", delta, delta).unsqueeze(0)
    T4 = torch.einsum(
        "x,xyzwv->xyzwv",
        oodR**9,
        105.0 * torch.einsum("xy,xz,xw,xv->xyzwv", dR_xyz, dR_xyz, dR_xyz, dR_xyz)
        - 15.0
        * torch.einsum(
            "x,xyzwv->xyzwv",
            dR2,
            RRdd
            + RRdd.permute(0, 1, 3, 2, 4)
            + RRdd.permute(0, 1, 4, 3, 2)
            + RRdd.permute(0, 3, 2, 1, 4)
            + RRdd.permute(0, 4, 2, 3, 1)
            + RRdd.permute(0, 3, 4, 1, 2),
        )
        + 3.0
        * torch.einsum(
            "x,xyzwv->xyzwv",
            dR2 * dR2,
            (dddd + dddd.permute(0, 1, 3, 2, 4) + dddd.permute(0, 1, 4, 3, 2)).expand(
                dR.shape[0], 3, 3, 3, 3
            ),
        ),
    )
    E_QQ = (1.0 / 9.0) * torch.einsum("xyzwv,xyz,xwv->x", T4, quadA, quadB)

    return E_uQ + E_QQ


def checkpoint_score(
    metric: str,
    component_mse: torch.Tensor,
    total_mae: torch.Tensor,
) -> torch.Tensor:
    """Return the configured validation quantity used to select a checkpoint."""
    if metric not in _CHECKPOINT_METRICS:
        raise ValueError(
            f"checkpoint metric must be one of {sorted(_CHECKPOINT_METRICS)}, "
            f"got {metric!r}"
        )
    return component_mse if metric == "component_mse" else total_mae
