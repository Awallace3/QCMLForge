"""
Model I/O utilities for saving and loading checkpoints.

This module provides helper functions for creating, saving, and loading
model checkpoints in the v2 format, which supports embedded submodels
and maintains backward compatibility with v1 checkpoints.

Checkpoint v2 Format Specification
----------------------------------
checkpoint = {
    "checkpoint_version": 2,
    "model_state_dict": model.state_dict(),
    "config": { ... model hyperparameters ... },
    "model_type": "APNet2_MPNN",
    "metadata": {
        "apnet_version": "0.0.1",
        "save_date": "2024-01-01T12:00:00",
        "device": "cuda",
    },
    "submodels": {
        "atom_model": {
            "model_state_dict": ...,
            "config": { ... },
            "model_type": "AtomMPNN",
            "submodels": { ... }  # nested if needed
        }
    }
}
"""

import json
import os
import warnings
from datetime import datetime
from typing import Any

import torch
import torch.nn as nn

from . import __version__

# Current checkpoint version
CHECKPOINT_VERSION = 2


def unwrap_model(model: nn.Module) -> nn.Module:
    """
    Unwrap a model from DDP or other wrappers.

    Parameters
    ----------
    model : nn.Module
        The model to unwrap, possibly wrapped in DistributedDataParallel

    Returns
    -------
    nn.Module
        The unwrapped model
    """
    seen = set()
    while True:
        model_id = id(model)
        if model_id in seen:
            raise ValueError("Cycle detected while unwrapping model")
        seen.add(model_id)
        if hasattr(model, "module"):
            model = model.module
        elif hasattr(model, "_orig_mod"):
            model = model._orig_mod
        else:
            return model


def strip_prefix_from_state_dict(
    state_dict: dict[str, Any], prefix: str = "_orig_mod."
) -> dict[str, Any]:
    """
    Strip a prefix from state dict keys (e.g., from torch.compile).

    Parameters
    ----------
    state_dict : dict
        The state dict with potentially prefixed keys
    prefix : str
        The prefix to strip, default "_orig_mod."

    Returns
    -------
    dict
        State dict with prefix stripped from keys
    """
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith(prefix):
            new_key = key[len(prefix) :]
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict


def add_prefix_to_state_dict(
    state_dict: dict[str, Any], prefix: str = "_orig_mod."
) -> dict[str, Any]:
    """
    Add a prefix to state dict keys (e.g., for loading into torch.compile model).

    Parameters
    ----------
    state_dict : dict
        The state dict without prefixed keys
    prefix : str
        The prefix to add, default "_orig_mod."

    Returns
    -------
    dict
        State dict with prefix added to keys
    """
    new_state_dict = {}
    for key, value in state_dict.items():
        if not key.startswith(prefix):
            new_key = prefix + key
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict


def get_checkpoint_version(checkpoint: dict[str, Any]) -> int:
    """
    Determine the version of a checkpoint.

    Parameters
    ----------
    checkpoint : dict
        The loaded checkpoint dictionary

    Returns
    -------
    int
        Checkpoint version (1 for legacy, 2 for new format)
    """
    return checkpoint.get("checkpoint_version", 1)


def create_checkpoint(
    model: nn.Module,
    config: dict[str, Any],
    model_type: str,
    submodels: dict[str, dict] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Create a v2 checkpoint dictionary.

    Parameters
    ----------
    model : nn.Module
        The model to save (will be unwrapped if needed)
    config : dict
        Model hyperparameters/configuration
    model_type : str
        String identifier for the model class (e.g., "AtomMPNN", "APNet2_MPNN")
    submodels : dict, optional
        Dictionary of embedded submodel checkpoints, keyed by submodel name
    metadata : dict, optional
        Additional metadata to include (e.g., training info)

    Returns
    -------
    dict
        Complete checkpoint dictionary in v2 format
    """
    unwrapped = unwrap_model(model)
    state_dict = unwrapped.state_dict()
    state_dict = strip_prefix_from_state_dict(state_dict)

    checkpoint = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "model_state_dict": state_dict,
        "config": config,
        "model_type": model_type,
        "metadata": {
            "apnet_version": __version__,
            "save_date": datetime.now().isoformat(),
            **(metadata or {}),
        },
    }

    if submodels:
        checkpoint["submodels"] = submodels

    return checkpoint


def create_submodel_checkpoint(
    model: nn.Module,
    config: dict[str, Any],
    model_type: str,
    submodels: dict[str, dict] | None = None,
) -> dict[str, Any]:
    """
    Create a checkpoint dictionary for embedding as a submodel.

    This is similar to create_checkpoint but without top-level metadata
    to keep the nested structure cleaner.

    Parameters
    ----------
    model : nn.Module
        The submodel to save
    config : dict
        Submodel hyperparameters/configuration
    model_type : str
        String identifier for the submodel class
    submodels : dict, optional
        Nested submodels if any

    Returns
    -------
    dict
        Submodel checkpoint dictionary
    """
    unwrapped = unwrap_model(model)
    state_dict = unwrapped.state_dict()
    state_dict = strip_prefix_from_state_dict(state_dict)

    submodel_checkpoint = {
        "model_state_dict": state_dict,
        "config": config,
        "model_type": model_type,
    }

    if submodels:
        submodel_checkpoint["submodels"] = submodels

    return submodel_checkpoint


def save_checkpoint(checkpoint: dict[str, Any], path: str) -> None:
    """
    Save a checkpoint to disk.

    Parameters
    ----------
    checkpoint : dict
        The checkpoint dictionary to save
    path : str
        Path to save the checkpoint to
    """
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str, map_location: str | torch.device | None = None
) -> dict[str, Any]:
    """
    Load a checkpoint from disk.

    Parameters
    ----------
    path : str
        Path to the checkpoint file
    map_location : str or torch.device, optional
        Device to map tensors to (e.g., "cpu", "cuda")

    Returns
    -------
    dict
        The loaded checkpoint dictionary
    """
    if map_location is None:
        map_location = "cpu"
    return torch.load(path, map_location=map_location, weights_only=False)


def load_state_dict_from_checkpoint(
    checkpoint: dict[str, Any],
    strip_compile_prefix: bool = True,
) -> dict[str, Any]:
    """
    Extract and clean the model state dict from a checkpoint.

    Handles both v1 and v2 checkpoint formats.

    Parameters
    ----------
    checkpoint : dict
        The checkpoint dictionary
    strip_compile_prefix : bool
        Whether to strip torch.compile prefixes

    Returns
    -------
    dict
        The cleaned model state dict
    """
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    if strip_compile_prefix:
        state_dict = strip_prefix_from_state_dict(state_dict)

    return state_dict


def load_config_from_checkpoint(checkpoint: dict[str, Any]) -> dict[str, Any] | None:
    """
    Extract config from a checkpoint.

    Parameters
    ----------
    checkpoint : dict
        The checkpoint dictionary

    Returns
    -------
    dict or None
        The config dictionary, or None if not present
    """
    return checkpoint.get("config")


def get_submodel_checkpoint(
    checkpoint: dict[str, Any],
    submodel_name: str,
) -> dict[str, Any] | None:
    """
    Extract an embedded submodel checkpoint.

    Parameters
    ----------
    checkpoint : dict
        The parent checkpoint dictionary
    submodel_name : str
        Name of the submodel to extract (e.g., "atom_model")

    Returns
    -------
    dict or None
        The submodel checkpoint, or None if not present
    """
    submodels = checkpoint.get("submodels", {})
    return submodels.get(submodel_name)


def has_embedded_submodel(checkpoint: dict[str, Any], submodel_name: str) -> bool:
    """
    Check if a checkpoint has an embedded submodel.

    Parameters
    ----------
    checkpoint : dict
        The checkpoint dictionary
    submodel_name : str
        Name of the submodel to check for

    Returns
    -------
    bool
        True if the submodel is embedded
    """
    return get_submodel_checkpoint(checkpoint, submodel_name) is not None


def warn_submodel_override(
    submodel_name: str,
    embedded_type: str | None = None,
    external_path: str | None = None,
) -> None:
    """
    Emit a warning when using embedded submodel instead of external path.

    Parameters
    ----------
    submodel_name : str
        Name of the submodel
    embedded_type : str, optional
        Type of the embedded submodel
    external_path : str, optional
        The external path that was provided but will be ignored
    """
    msg = f"Checkpoint contains embedded '{submodel_name}' (type: {embedded_type}). "
    if external_path:
        msg += f"Ignoring externally provided path: {external_path}. "
    msg += "Using embedded submodel for consistency."
    warnings.warn(msg, UserWarning)


def validate_checkpoint(
    checkpoint: dict[str, Any], expected_type: str | None = None
) -> bool:
    """
    Validate a checkpoint has required fields.

    Parameters
    ----------
    checkpoint : dict
        The checkpoint dictionary to validate
    expected_type : str, optional
        Expected model_type value

    Returns
    -------
    bool
        True if valid

    Raises
    ------
    ValueError
        If checkpoint is invalid
    """
    version = get_checkpoint_version(checkpoint)

    if version >= 2:
        required_keys = ["model_state_dict", "config", "model_type"]
        for key in required_keys:
            if key not in checkpoint:
                raise ValueError(f"Checkpoint missing required key: {key}")

        if expected_type and checkpoint["model_type"] != expected_type:
            raise ValueError(
                f"Checkpoint model_type mismatch: expected {expected_type}, "
                f"got {checkpoint['model_type']}"
            )
    else:
        # v1 checkpoints just need model_state_dict (or be a raw state dict)
        if "model_state_dict" not in checkpoint and not isinstance(
            next(iter(checkpoint.values()), None), torch.Tensor
        ):
            # Check if it looks like a state dict (keys are parameter names)
            if not any("weight" in k or "bias" in k for k in checkpoint.keys()):
                raise ValueError("Checkpoint appears to be neither v1 nor v2 format")

    return True


def upgrade_v1_checkpoint(
    checkpoint: dict[str, Any],
    config: dict[str, Any],
    model_type: str,
) -> dict[str, Any]:
    """
    Upgrade a v1 checkpoint to v2 format (in memory, not saved).

    Parameters
    ----------
    checkpoint : dict
        The v1 checkpoint
    config : dict
        Config to add (must be provided externally for v1)
    model_type : str
        Model type to add

    Returns
    -------
    dict
        Upgraded checkpoint in v2 format
    """
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    existing_config = checkpoint.get("config", {})

    # Merge existing config with provided config (provided takes precedence)
    merged_config = {**existing_config, **config}

    return {
        "checkpoint_version": CHECKPOINT_VERSION,
        "model_state_dict": state_dict,
        "config": merged_config,
        "model_type": model_type,
        "metadata": {
            "apnet_version": __version__,
            "upgraded_from_v1": True,
        },
    }


# ---------------------------------------------------------------------------
# Resumable training state
#
# The model checkpoint is the *deliverable*: it holds the best-validation
# weights, and every downstream consumer (the S66x8 profiler, the classical
# merge, `AM_DimerParam_Model.__init__`) reads it. Training state is a
# different object with a different lifetime -- last-epoch weights, the Adam
# moments, the epoch counter, and the best loss seen so far -- so it lives in a
# sidecar beside the checkpoint rather than inside it. Keeping them apart means
# the ~21 MB of optimizer moments never reaches a consumer that only wants
# weights, and an absent sidecar is simply "no resume information", which is
# what every checkpoint written before this existed says.
#
# Why this is needed at all: an 8-hour QoS cap forces long training into
# chunks that warm-start from the previous chunk's file. Without the sidecar
# each chunk restarts its best-loss tracking at +inf, so its first epoch
# overwrites the deliverable unconditionally -- and since a chunk's first epoch
# runs on a freshly zeroed Adam state, it is usually slightly *worse* than the
# weights it just loaded. Repeated preemption on a preemptible queue turns that
# into a ratchet that walks the model backwards.
TRAIN_STATE_VERSION = 1
TRAIN_STATE_SUFFIX = ".trainstate.pt"


def train_state_path(model_save_path: str) -> str:
    """Sidecar path for ``model_save_path``'s resumable training state."""
    return f"{model_save_path}{TRAIN_STATE_SUFFIX}"


def save_train_state(
    path: str,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs_completed: int,
    lowest_test_loss: float,
    identity: dict[str, Any] | None = None,
) -> None:
    """Write resumable training state to ``path``, atomically.

    Written once per epoch, so the write has to be crash-safe: a job killed
    mid-``torch.save`` would otherwise leave a truncated sidecar that the next
    chunk cannot read, and the chunk would silently fall back to restarting.
    Saving to a temporary file and ``os.replace``-ing it means the sidecar on
    disk is always either the previous epoch's or this one's, never a fragment.

    ``identity`` is recorded verbatim and checked on load; see
    :func:`load_train_state`.
    """
    state_dict = strip_prefix_from_state_dict(unwrap_model(model).state_dict())
    payload = {
        "train_state_version": TRAIN_STATE_VERSION,
        "apnet_version": __version__,
        "save_date": datetime.now().isoformat(),
        "epochs_completed": int(epochs_completed),
        "lowest_test_loss": float(lowest_test_loss),
        # CPU tensors so the sidecar is portable between a V100 chunk and a
        # CPU-only inspection, and so `torch.save` does not pin GPU memory.
        "model_state_dict": {k: v.detach().cpu() for k, v in state_dict.items()},
        "optimizer_state_dict": optimizer.state_dict(),
        "identity": dict(identity or {}),
    }
    tmp = f"{path}.tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_train_state(
    path: str,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    identity: dict[str, Any] | None = None,
) -> tuple[int, float] | None:
    """Restore training state from ``path`` into ``model`` and ``optimizer``.

    Returns ``(epochs_completed, lowest_test_loss)`` on success and ``None``
    when there is nothing usable to resume from -- no sidecar, a version this
    build does not understand, an ``identity`` that disagrees with the caller's,
    or a state dict that does not fit the model.

    Every one of those is a *warning*, not an error. A resume that cannot
    proceed should cost the run its Adam moments and its epoch counter, not the
    whole job; the weights it would have loaded are still on disk in the
    checkpoint the harness warm-started from. The one thing that must never
    happen is resuming from state that belongs to different physics, which is
    what the ``identity`` check is for: a sidecar written by the pre-fix
    induction functional would otherwise reinstate its weights on top of a
    corrected checkpoint.
    """
    if not path or not os.path.exists(path):
        return None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # truncated, or written by an incompatible torch
        warnings.warn(f"Ignoring unreadable training state {path}: {exc}")
        return None

    version = payload.get("train_state_version")
    if version != TRAIN_STATE_VERSION:
        warnings.warn(
            f"Ignoring training state {path}: version {version!r}, "
            f"this build writes {TRAIN_STATE_VERSION}"
        )
        return None

    want = dict(identity or {})
    got = payload.get("identity") or {}
    mismatched = {k: (v, got.get(k)) for k, v in want.items() if got.get(k) != v}
    if mismatched:
        warnings.warn(
            f"Ignoring training state {path}: identity mismatch {mismatched!r} "
            "(expected, found). Not resuming rather than mixing runs."
        )
        return None

    # Check the state dict fits *before* copying anything into the model.
    # `load_state_dict` reports every key and shape problem it found, but it
    # raises only after having already copied the parameters that did fit, so
    # letting it fail would leave the warm-started weights half-overwritten by
    # the very state we decided not to trust.
    target = unwrap_model(model)
    current = target.state_dict()
    saved = payload.get("model_state_dict") or {}
    problems = []
    if set(saved) != set(current):
        missing = sorted(set(current) - set(saved))
        unexpected = sorted(set(saved) - set(current))
        if missing:
            problems.append(f"missing keys {missing[:5]}")
        if unexpected:
            problems.append(f"unexpected keys {unexpected[:5]}")
    shape_mismatch = [
        key
        for key, value in saved.items()
        if key in current
        and getattr(value, "shape", None) != getattr(current[key], "shape", None)
    ]
    if shape_mismatch:
        problems.append(f"shape mismatch for {sorted(shape_mismatch)[:5]}")
    if problems:
        warnings.warn(
            f"Ignoring training state {path}: does not fit this model "
            f"({'; '.join(problems)}). Model left untouched."
        )
        return None

    try:
        target.load_state_dict(saved, strict=True)
    except Exception as exc:
        warnings.warn(f"Ignoring training state {path}: {exc}")
        return None

    try:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    except Exception as exc:
        # The weights are already restored at this point, and they are the
        # valuable half. Losing the moments costs a short re-warm, so this is
        # reported and continued rather than unwound.
        warnings.warn(
            f"Training state {path}: restored weights but not optimizer "
            f"state ({exc}); Adam moments restart from zero."
        )

    return int(payload["epochs_completed"]), float(payload["lowest_test_loss"])


# ---------------------------------------------------------------------------
# Best-MAE sidecar
#
# The primary checkpoint is starred on validation MSE, but every table these
# models are read in -- the S66x8 gate, the per-component breakdowns -- is in
# MAE, and the two selectors disagree often enough to matter. The l<=2 exchange
# arm last starred epoch 3 of 11 while validation exchange kept improving
# through epoch 10; with no sidecar configured those weights were unrecoverable.
# This preserves the best-MAE epoch *beside* the primary artifact instead of
# displacing it.
BEST_MAE_SELECTOR = "val_total_MAE"


def best_mae_sidecar_paths(model_save_path: str) -> tuple[str, str]:
    """Checkpoint and record paths for the MAE-selected sidecar."""
    base, _ = os.path.splitext(model_save_path)
    return base + ".best-mae.pt", base + ".best-mae.json"


def best_mae_sidecar_floor(model_save_path: str | None) -> float:
    """Best validation MAE a previous chunk already banked at this path.

    Long trainings run as a chain of warm-started chunks, and each chunk seeds
    its selector from its own fresh pre-training eval. Without this floor a
    later chunk would overwrite an earlier chunk's sidecar with a worse epoch,
    because it only ever compares against where it happened to start. A
    missing, unreadable, or foreign record returns ``inf``, which is exactly the
    single-run behaviour.
    """
    if not model_save_path:
        return float("inf")
    checkpoint_path, record_path = best_mae_sidecar_paths(model_save_path)
    if not os.path.exists(checkpoint_path):
        return float("inf")
    try:
        with open(record_path) as f:
            record = json.load(f)
        if record.get("model_save_path") != model_save_path:
            return float("inf")
        return float(record[BEST_MAE_SELECTOR])
    except (OSError, ValueError, TypeError, KeyError):
        return float("inf")


def save_best_mae_record(
    path: str,
    *,
    model_save_path: str,
    checkpoint: str,
    val_total_MAE: float,
    component_MAE: list[float],
    epoch: int,
) -> None:
    """Write the sidecar's record file.

    ``epoch`` is the *global* epoch counter the trainer is iterating, not a
    chunk-local index: a chunk-local one cannot be compared across the chain,
    which is a defect the earlier sidecar shipped with.

    The caller must write the checkpoint first. A torn write then leaves the
    floor pointing at the older, fully-written pair rather than at a fragment.
    """
    payload = {
        "model_save_path": model_save_path,
        "checkpoint": checkpoint,
        "selector": BEST_MAE_SELECTOR,
        BEST_MAE_SELECTOR: float(val_total_MAE),
        "component_MAE": [float(v) for v in component_MAE],
        "epoch": int(epoch),
        "epoch_is_global": True,
        "apnet_version": __version__,
        "save_date": datetime.now().isoformat(),
    }
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)
