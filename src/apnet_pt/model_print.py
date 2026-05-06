"""
Human-readable model architecture printer for AP-Net models.

Provides ModelInfo, get_model_info, and print_model_tree — analogous to
model_io.py but focused on rendering sub-models, data flow, and shared-weight
call patterns as an ASCII/Unicode tree.
"""

from __future__ import annotations

import io
import sys
from dataclasses import dataclass, field

import torch.nn as nn
from torch.nn.parameter import UninitializedParameter


def _safe_numel(p) -> int:
    """Return numel for *p*, or 0 if it is an uninitialized LazyModule parameter."""
    if isinstance(p, UninitializedParameter):
        return 0
    return p.numel()


@dataclass
class ModelInfo:
    name: str
    role: str = ""
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    passes: list[str] = field(default_factory=list)  # tensors forwarded unchanged
    frozen: bool = False
    n_params: int = 0  # trainable params
    n_params_total: int = 0  # all params
    n_calls: int = 1  # invocations per dimer forward pass
    call_note: str = ""  # explains n_calls > 1 or internal dual-run
    children: list["ModelInfo"] = field(default_factory=list)
    is_group: bool = False  # synthetic group node (no [state] prefix, no param count)


# ── public entry points ────────────────────────────────────────────────────────


def get_model_info(model) -> ModelInfo:
    """Return a ModelInfo tree for *model*.

    Delegates to ``model.get_model_info()`` when present; falls back to generic
    inspection of ``named_children()`` with ``n_calls = 1``.
    """
    if hasattr(model, "get_model_info"):
        return model.get_model_info()
    if isinstance(model, nn.Module):
        return _generic_model_info(model)
    return ModelInfo(name=type(model).__name__)


def print_model_tree(model_or_info, file=None, unicode: bool = True) -> None:
    """Render a human-readable sub-model / data-flow tree to *file*.

    Parameters
    ----------
    model_or_info:
        An ``nn.Module``, a wrapper with ``get_model_info()``, or a
        ``ModelInfo`` directly.
    file:
        Output stream (defaults to ``sys.stdout``).
    unicode:
        Use Unicode box-drawing characters (``True``) or plain ASCII (``False``).
    """
    if file is None:
        file = sys.stdout

    if isinstance(model_or_info, ModelInfo):
        info = model_or_info
    elif hasattr(model_or_info, "get_model_info"):
        info = model_or_info.get_model_info()
    elif isinstance(model_or_info, nn.Module):
        info = _generic_model_info(model_or_info)
    else:
        info = ModelInfo(name=type(model_or_info).__name__)

    C = _U if unicode else _A

    _populate_aggregate_counts(info)

    # Root node: just the name and total params (no connector, no state prefix)
    print(f"{info.name} ({_fmt_params(info.n_params_total)} params)", file=file)
    for i, child in enumerate(info.children):
        _render(child, prefix="", last=(i == len(info.children) - 1), file=file, C=C)


def model_tree_string(model_or_info, unicode: bool = True) -> str:
    """Return the rendered model tree as a string."""
    buf = io.StringIO()
    print_model_tree(model_or_info, file=buf, unicode=unicode)
    return buf.getvalue().rstrip("\n")


# ── internal helpers ───────────────────────────────────────────────────────────


def _generic_model_info(model: nn.Module) -> ModelInfo:
    n_total = sum(_safe_numel(p) for p in model.parameters())
    n_train = sum(_safe_numel(p) for p in model.parameters() if p.requires_grad)
    return ModelInfo(
        name=type(model).__name__,
        frozen=(n_train == 0),
        n_params=n_train,
        n_params_total=n_total,
        children=[get_model_info(c) for _, c in model.named_children()],
    )


def _fmt_params(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1e6:.2g} M"
    if n >= 1_000:
        return f"{n / 1e3:.0f} k"
    return str(n)


def _populate_aggregate_counts(info: ModelInfo) -> tuple[int, int]:
    """Fill missing aggregate counts from children and return (trainable, total)."""
    child_counts = [_populate_aggregate_counts(child) for child in info.children]
    if info.children and info.n_params == 0 and info.n_params_total == 0:
        info.n_params = sum(n_train for n_train, _ in child_counts)
        info.n_params_total = sum(n_total for _, n_total in child_counts)
    return info.n_params, info.n_params_total


_U = {"branch": "├─ ", "last": "└─ ", "pipe": "│   ", "blank": "    "}
_A = {"branch": "+- ", "last": "\\- ", "pipe": "|   ", "blank": "    "}


def _render(info: ModelInfo, prefix: str, last: bool, file, C: dict) -> None:
    connector = C["last"] if last else C["branch"]
    child_prefix = prefix + (C["blank"] if last else C["pipe"])

    # ── header line ──────────────────────────────────────────────────────────
    if info.is_group:
        # Synthetic classical/non-trainable groups: simpler header
        print(f"{prefix}{connector}{info.name}  (non-trainable)", file=file)
    else:
        state = "frozen" if info.frozen else "train"
        calls = f" ×{info.n_calls}" if info.n_calls != 1 else ""
        param_key = "n_params_total" if info.frozen else "n_params"
        param_val = getattr(info, param_key)
        print(
            f"{prefix}{connector}[{state}{calls}] {info.name}"
            f"  ({_fmt_params(param_val)} params)",
            file=file,
        )

    # ── optional role / note lines (aligned under module name) ───────────────
    indent = child_prefix + "             "
    if info.role:
        print(f"{indent}role: {info.role}", file=file)
    if info.call_note:
        print(f"{indent}note: {info.call_note}", file=file)

    # ── I/O lines ────────────────────────────────────────────────────────────
    if info.inputs:
        print(f"{child_prefix}  Inputs : {', '.join(info.inputs)}", file=file)
    if info.outputs:
        print(f"{child_prefix}  Outputs: {', '.join(info.outputs)}", file=file)
    if info.passes:
        print(f"{child_prefix}  Passes : {', '.join(info.passes)}", file=file)
    if info.inputs or info.outputs or info.passes:
        print(child_prefix, file=file)

    # ── recurse into children ────────────────────────────────────────────────
    for i, child in enumerate(info.children):
        _render(child, child_prefix, last=(i == len(info.children) - 1), file=file, C=C)
