"""Small, model-neutral reducers for distributed semantic MAE metrics."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.distributed as dist


def globally_reduced_mae(
    error_sums: Sequence[float | torch.Tensor],
    sample_count: int | torch.Tensor,
    *,
    component_widths: Sequence[int] | None = None,
    device: str | torch.device = "cpu",
) -> tuple[float, ...]:
    """Reduce absolute-error numerators and return sample-normalized MAEs.

    ``sample_count`` is the number of samples (dimers for pairwise metrics or
    atoms for atomic metrics), never the number of scalar prediction elements.
    ``component_widths`` supports atomic vector/tensor properties whose error
    sums contain three or nine scalar values per atom.
    """

    if not error_sums:
        raise ValueError("At least one error numerator is required")
    widths = tuple(component_widths or (1,) * len(error_sums))
    if len(widths) != len(error_sums) or any(width < 1 for width in widths):
        raise ValueError("component_widths must contain one positive value per metric")
    values = [torch.as_tensor(value, dtype=torch.float64) for value in error_sums]
    values.append(torch.as_tensor(sample_count, dtype=torch.float64))
    packed = torch.stack(values).to(device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    count = float(packed[-1].item())
    if count <= 0:
        raise ValueError("Global sample count must be positive")
    return tuple(
        float(packed[index].item()) / (count * width)
        for index, width in enumerate(widths)
    )
