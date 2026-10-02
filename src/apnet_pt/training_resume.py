"""Resumable training state for the single-process harness loops.

A preemptible queue kills a job at an arbitrary point.  The best-model
checkpoint alone cannot continue such a run: it restarts Adam with zero
moments, restarts the learning-rate schedule at step 0, and replays the first
epoch's shuffle order.  The resume state holds everything the next epoch
depends on, so a run that is interrupted and resumed reproduces the
uninterrupted run.
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

# v2 stores every RNG stream as tensors and plain Python values, so the file
# loads with ``weights_only=True``.  v1 pickled NumPy state and cannot.
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


def _numpy_rng_state() -> tuple:
    """NumPy's global RNG state with its key array held as a tensor.

    ``torch.load(weights_only=True)`` refuses NumPy arrays, so the uint32
    Mersenne Twister keys travel as int64 and are narrowed back on restore.
    """
    name, keys, position, has_gauss, cached_gaussian = np.random.get_state()
    return (
        name,
        torch.from_numpy(keys.astype(np.int64)),
        int(position),
        int(has_gauss),
        float(cached_gaussian),
    )


def _set_numpy_rng_state(state: tuple) -> None:
    name, keys, position, has_gauss, cached_gaussian = state
    np.random.set_state(
        (name, keys.numpy().astype(np.uint32), position, has_gauss, cached_gaussian)
    )


def _plain_scalars(value: Any) -> Any:
    """Replace NumPy scalars with the equal Python number, recursively.

    The inverse-time schedule returns ``numpy.float64`` learning rates, which
    reach the optimizer and scheduler state dicts and which
    ``torch.load(weights_only=True)`` refuses.  The value is unchanged.
    """
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _plain_scalars(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_plain_scalars(item) for item in value)
    return value


def _capture_rng_state(generator: torch.Generator) -> dict[str, Any]:
    state = {
        "torch": torch.get_rng_state(),
        "numpy": _numpy_rng_state(),
        "python": random.getstate(),
        "loader": generator.get_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Mapping[str, Any], generator: torch.Generator):
    torch.set_rng_state(state["torch"])
    _set_numpy_rng_state(state["numpy"])
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
        "optimizer_state_dict": _plain_scalars(optimizer.state_dict()),
        "scheduler_state_dict": (
            None if scheduler is None else _plain_scalars(scheduler.state_dict())
        ),
        "rng": _capture_rng_state(generator),
    }
    partial_path = f"{path}.partial"
    with open(partial_path, "wb") as handle:
        torch.save(state, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial_path, path)


def load_training_state(
    path: str, fingerprint: Mapping[str, Any] | None
) -> dict[str, Any] | None:
    """Read a resume state, or return ``None`` when no run has started yet.

    Parameters
    ----------
    path : str
        Resume-state file written by :func:`save_training_state`.
    fingerprint : mapping or None
        The current run's setup.  Every key must match the stored setup;
        ``None`` skips the comparison (for inspecting a state file).

    Raises
    ------
    ValueError
        If the file is not a resume state, or the setup it was written under
        differs from ``fingerprint``.
    """
    if not os.path.exists(path):
        return None
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError as error:
        raise ValueError(
            f"{path} is not a {RESUME_STATE_FORMAT} file; a "
            "qcmlforge-training-resume-v1 state must be resumed with the "
            "source commit that wrote it"
        ) from error
    if not isinstance(state, dict) or state.get("format") != RESUME_STATE_FORMAT:
        raise ValueError(f"{path} is not a {RESUME_STATE_FORMAT} file")
    if fingerprint is not None:
        stored = state["fingerprint"]
        mismatched = {
            key: (stored.get(key), value)
            for key, value in fingerprint.items()
            if stored.get(key) != value
        }
        if mismatched:
            details = ", ".join(
                f"{key}: stored {old!r}, requested {new!r}"
                for key, (old, new) in sorted(mismatched.items())
            )
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
