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
    if hasattr(model, "module"):
        return model.module
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
