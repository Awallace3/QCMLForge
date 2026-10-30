"""Resumable training state for the single-process harness loops.

The best-model checkpoint alone cannot continue a preempted run: it restarts
Adam, the learning-rate schedule, and the shuffle order.  The resume state holds
everything the next epoch depends on, so a resumed run reproduces the
uninterrupted one.
"""

from __future__ import annotations

import functools
import os
import pickle
import random
from typing import Any, Callable, Mapping

import numpy as np
import torch
from torch import nn

from . import model_io

# v2 holds only tensors and plain Python values (loads with weights_only=True).
RESUME_STATE_FORMAT = "qcmlforge-training-resume-v2"


def bare_module(model: nn.Module) -> nn.Module:
    """Strip DDP and ``torch.compile`` wrappers so state-dict keys are plain."""
    model = model_io.unwrap_model(model)
    return getattr(model, "_orig_mod", model)


def describe_loss_fn(loss_fn: Callable | None) -> str | None:
    """A stable description of a criterion, for comparing two runs' setups."""
    if loss_fn is None:
        return None
    if isinstance(loss_fn, functools.partial):
        keywords = ", ".join(
            f"{key}={value!r}" for key, value in sorted(loss_fn.keywords.items())
        )
        return f"{describe_loss_fn(loss_fn.func)}({keywords})"
    name = getattr(loss_fn, "__qualname__", type(loss_fn).__qualname__)
    return f"{getattr(loss_fn, '__module__', '')}.{name}"


def _plain(value: Any) -> Any:
    """NumPy RNG keys to int64 tensors and NumPy scalars to Python numbers.

    ``torch.load(weights_only=True)`` refuses NumPy objects, which reach the
    state through the NumPy RNG and the inverse-time schedule's learning rates.
    """
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value.astype(np.int64))
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_plain(item) for item in value)
    return value


def _capture_rng_state(generator: torch.Generator) -> dict[str, Any]:
    state = {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
        "loader": generator.get_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Mapping[str, Any], generator: torch.Generator):
    torch.set_rng_state(state["torch"])
    name, keys, *rest = state["numpy"]
    np.random.set_state((name, keys.numpy().astype(np.uint32), *rest))
    random.setstate(state["python"])
    generator.set_state(state["loader"])
    cuda = state.get("cuda")
    if cuda is not None and torch.cuda.is_available():
        if len(cuda) != torch.cuda.device_count():
            raise ValueError(
                f"Resume state holds {len(cuda)} CUDA RNG states but "
                f"{torch.cuda.device_count()} devices are visible"
            )
        torch.cuda.set_rng_state_all(cuda)


def _cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().to("cpu", copy=True)
        for key, value in bare_module(model).state_dict().items()
    }


def save_training_state(
    path: str,
    *,
    model: nn.Module,
    best_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    generator: torch.Generator,
    epochs_completed: int,
    n_epochs: int,
    best_epoch: int,
    best_score: float,
    fingerprint: Mapping[str, Any],
) -> None:
    """Atomically write the state that continues training after an epoch.

    Call this at the very end of an epoch, after every draw that epoch makes
    from any RNG.  The file is written beside ``path`` and renamed over it, so
    a kill mid-write leaves the previous epoch's state intact.
    """
    state = {
        "format": RESUME_STATE_FORMAT,
        "fingerprint": dict(fingerprint),
        "epochs_completed": int(epochs_completed),
        "n_epochs": int(n_epochs),
        "best_epoch": int(best_epoch),
        "best_score": float(best_score),
        "model_state_dict": _cpu_state_dict(model),
        "best_model_state_dict": _cpu_state_dict(best_model),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": (
            None if scheduler is None else scheduler.state_dict()
        ),
        "rng": _capture_rng_state(generator),
    }
    partial_path = f"{path}.partial"
    with open(partial_path, "wb") as handle:
        torch.save(_plain(state), handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial_path, path)


def load_training_state(
    path: str, fingerprint: Mapping[str, Any] | None
) -> dict[str, Any] | None:
    """Read a resume state, or ``None`` if ``path`` does not exist yet.

    Raises ``ValueError`` if the file is not a resume state or was written
    under a setup that differs from ``fingerprint`` (``None`` skips the check).
    """
    if not os.path.exists(path):
        return None
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError as error:
        raise ValueError(
            f"{path} is not a {RESUME_STATE_FORMAT} file (a v1 state must be "
            "resumed with the source commit that wrote it)"
        ) from error
    if not isinstance(state, dict) or state.get("format") != RESUME_STATE_FORMAT:
        raise ValueError(f"{path} is not a {RESUME_STATE_FORMAT} file")
    if fingerprint is not None:
        stored = state["fingerprint"]
        details = ", ".join(
            f"{key}: stored {stored.get(key)!r}, requested {value!r}"
            for key, value in sorted(fingerprint.items())
            if stored.get(key) != value
        )
        if details:
            raise ValueError(f"Cannot resume {path}; setup changed ({details})")
    return state


def apply_training_state(
    state: Mapping[str, Any],
    *,
    model: nn.Module,
    best_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    generator: torch.Generator,
) -> None:
    """Load a resume state into a freshly built loop, RNG streams last."""
    bare_module(model).load_state_dict(state["model_state_dict"])
    bare_module(best_model).load_state_dict(state["best_model_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])
    if scheduler is not None:
        scheduler.load_state_dict(state["scheduler_state_dict"])
    _restore_rng_state(state["rng"], generator)
