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
