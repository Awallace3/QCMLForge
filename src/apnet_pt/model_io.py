"""
Model I/O utilities for saving and loading checkpoints.

This module provides the legacy-compatible v1/v2 helpers plus strict v3
helpers for MACE models whose foundation backbone remains external. The v2
writer remains the default for all existing QCMLForge models.

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

import copy
import hashlib
import math
from collections.abc import Callable, Mapping
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from . import __version__

# Legacy/default writers remain at v2. MACE external-backbone checkpoints use
# the separate, explicit v3 helpers below.
CHECKPOINT_VERSION = 2
MACE_CHECKPOINT_VERSION = 3
MACE_BACKBONE_STATE_PREFIXES = ("featurizer.backbone.",)


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
        Checkpoint version (1 for legacy, 2 for embedded, 3 for external MACE)
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

    if version == MACE_CHECKPOINT_VERSION:
        validate_mace_checkpoint_v3(checkpoint)

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


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Hash an external artifact without deserializing it."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_keys(record: Mapping[str, Any], keys: set[str], label: str) -> None:
    missing = sorted(keys - set(record))
    if missing:
        raise ValueError(f"{label} missing required key(s): {missing}")


def _validate_safe_metadata(value: object, path: str = "metadata") -> None:
    """Reject modules/calculators and other pickle-only checkpoint records."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} keys must be strings")
            _validate_safe_metadata(item, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_safe_metadata(item, f"{path}[{index}]")
        return
    raise TypeError(f"{path} contains a non-record value of type {type(value).__name__}")


def _validate_mace_v3_config(config: Mapping[str, Any]) -> None:
    _require_keys(
        config,
        {
            "architecture",
            "mace",
            "pair_mode",
            "dtype_policy",
            "atomic_property_schema",
            "physics",
            "data",
            "seed",
            "parameter_counts",
            "source_commit",
            "route_submodel_digests",
        },
        "MACE v3 config",
    )
    if config["architecture"] not in {
        "direct-polar",
        "hybrid-h1",
        "hybrid-h2",
        "atomhead",
    }:
        raise ValueError("MACE v3 config has an unsupported architecture")
    if config["pair_mode"] not in {"h1", "h2"}:
        raise ValueError("MACE v3 pair_mode must be h1 or h2")
    if config["dtype_policy"] not in {"float32", "float64"}:
        raise ValueError("MACE v3 dtype_policy must be float32 or float64")
    if config["atomic_property_schema"] != (
        "ap3-atomic-properties-cartesian-v1"
    ):
        raise ValueError("MACE v3 atomic property schema is unsupported")
    if isinstance(config["seed"], bool) or not isinstance(config["seed"], int):
        raise ValueError("MACE v3 seed must be an integer")
    source_commit = config["source_commit"]
    if (
        not isinstance(source_commit, str)
        or len(source_commit) != 40
        or any(character not in "0123456789abcdef" for character in source_commit)
    ):
        raise ValueError("MACE v3 source_commit must be a 40-character git hash")

    mace = config["mace"]
    if not isinstance(mace, Mapping):
        raise ValueError("MACE v3 mace record must be a mapping")
    _require_keys(
        mace,
        {"model_id", "version", "sha256", "feature_schema", "feature_mode"},
        "MACE v3 mace record",
    )
    if not _is_sha256(mace["sha256"]):
        raise ValueError("MACE v3 mace digest must be a lowercase SHA-256")
    for key in ("model_id", "version", "feature_schema", "feature_mode"):
        if not isinstance(mace[key], str) or not mace[key]:
            raise ValueError(f"MACE v3 mace.{key} must be non-empty")
    if mace["feature_mode"] not in {
        "final-layer-scalars",
        "all-scalars+norms",
    }:
        raise ValueError("MACE v3 feature mode is unsupported")
    if f":mode={mace['feature_mode']}:" not in mace["feature_schema"]:
        raise ValueError("MACE v3 feature schema and mode disagree")
    route_contracts = {
        "direct-polar": ("h1", "all-scalars+norms"),
        "hybrid-h1": ("h1", "final-layer-scalars"),
        "hybrid-h2": ("h2", "all-scalars+norms"),
        "atomhead": ("h1", "all-scalars+norms"),
    }
    expected_pair, expected_features = route_contracts[config["architecture"]]
    if config["pair_mode"] != expected_pair:
        raise ValueError("MACE v3 pair mode disagrees with architecture")
    if mace["feature_mode"] != expected_features:
        raise ValueError("MACE v3 feature mode disagrees with architecture")

    physics = config["physics"]
    if not isinstance(physics, Mapping):
        raise ValueError("MACE v3 physics record must be a mapping")
    _require_keys(
        physics,
        {
            "electrostatics_mode",
            "induction_mode",
            "dispersion_mode",
            "d3_parameters",
            "component_order",
            "length_unit",
            "energy_unit",
            "neural_cutoff",
            "physics_hash",
        },
        "MACE v3 physics record",
    )
    if tuple(physics["component_order"]) != ("elst", "exch", "indu", "disp"):
        raise ValueError("MACE v3 component order must be elst/exch/indu/disp")
    if physics["electrostatics_mode"] not in {
        "damped-cliff",
        "damped-amoeba",
        "undamped",
    }:
        raise ValueError("MACE v3 electrostatics mode is unsupported")
    if physics["induction_mode"] != "thole-scf":
        raise ValueError("MACE v3 induction mode must be thole-scf")
    if physics["dispersion_mode"] != "d3":
        raise ValueError("MACE v3 dispersion mode must be d3")
    d3_parameters = physics["d3_parameters"]
    if not isinstance(d3_parameters, (list, tuple)) or len(d3_parameters) not in {
        0,
        4,
    }:
        raise ValueError("MACE v3 D3 parameters must be empty or have four values")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        for value in d3_parameters
    ):
        raise ValueError("MACE v3 D3 parameters must be finite numeric values")
    neural_cutoff = physics["neural_cutoff"]
    if (
        isinstance(neural_cutoff, bool)
        or not isinstance(neural_cutoff, (int, float))
        or not math.isfinite(neural_cutoff)
        or neural_cutoff <= 0
    ):
        raise ValueError("MACE v3 neural cutoff must be finite and positive")
    if physics["length_unit"] != "angstrom" or physics["energy_unit"] != "kcal/mol":
        raise ValueError("MACE v3 public units must be angstrom and kcal/mol")
    if not _is_sha256(physics["physics_hash"]):
        raise ValueError("MACE v3 physics_hash must be a lowercase SHA-256")

    data = config["data"]
    if not isinstance(data, Mapping):
        raise ValueError("MACE v3 data record must be a mapping")
    _require_keys(
        data,
        {"dataset_hash", "preprocessing_hash", "split_hash"},
        "MACE v3 data record",
    )
    for key in ("dataset_hash", "preprocessing_hash", "split_hash"):
        if not _is_sha256(data[key]):
            raise ValueError(f"MACE v3 {key} must be a lowercase SHA-256")

    digests = config["route_submodel_digests"]
    if not isinstance(digests, Mapping) or not digests:
        raise ValueError("MACE v3 route_submodel_digests must be a non-empty mapping")
    for name, digest in digests.items():
        if not isinstance(name, str) or not _is_sha256(digest):
            raise ValueError("route submodel digests must be named lowercase SHA-256s")

    counts = config["parameter_counts"]
    if not isinstance(counts, Mapping):
        raise ValueError("MACE v3 parameter_counts must be a mapping")
    _require_keys(
        counts,
        {"total", "trainable", "external", "serialized"},
        "MACE v3 parameter counts",
    )
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in counts.values()
    ):
        raise ValueError("MACE v3 parameter counts must be non-negative integers")


def _validate_external_mace_record(record: Mapping[str, Any]) -> None:
    _require_keys(
        record,
        {
            "canonical_locator",
            "sha256",
            "model_id",
            "version",
            "model_class",
            "license",
            "license_acknowledged",
            "state_prefixes",
        },
        "external_submodels.mace",
    )
    for key in (
        "canonical_locator",
        "model_id",
        "version",
        "model_class",
        "license",
    ):
        if not isinstance(record[key], str) or not record[key]:
            raise ValueError(f"external MACE {key} must be non-empty")
    if "://" not in record["canonical_locator"]:
        raise ValueError("external MACE canonical_locator must use a URI scheme")
    if not _is_sha256(record["sha256"]):
        raise ValueError("external MACE digest must be a lowercase SHA-256")
    if record["license_acknowledged"] is not True:
        raise ValueError("external MACE license must be explicitly acknowledged")
    if tuple(record["state_prefixes"]) != MACE_BACKBONE_STATE_PREFIXES:
        raise ValueError("external MACE state prefixes are not registered")


def validate_mace_checkpoint_v3(checkpoint: Mapping[str, Any]) -> bool:
    """Validate all mandatory external-backbone v3 records."""

    if checkpoint.get("checkpoint_version") != MACE_CHECKPOINT_VERSION:
        raise ValueError("MACE checkpoint must use checkpoint_version 3")
    _require_keys(
        checkpoint,
        {
            "model_state_dict",
            "config",
            "model_type",
            "architecture",
            "metadata",
            "external_submodels",
        },
        "MACE checkpoint v3",
    )
    if checkpoint["model_type"] != "MACEAP3D3":
        raise ValueError("MACE v3 model_type must be MACEAP3D3")
    config = checkpoint["config"]
    if not isinstance(config, Mapping):
        raise ValueError("MACE v3 config must be a mapping")
    _validate_mace_v3_config(config)
    if checkpoint["architecture"] != config["architecture"]:
        raise ValueError("MACE v3 architecture records disagree")
    external = checkpoint["external_submodels"]
    if not isinstance(external, Mapping) or set(external) != {"mace"}:
        raise ValueError("MACE v3 requires exactly one external MACE submodel")
    mace_external = external["mace"]
    if not isinstance(mace_external, Mapping):
        raise ValueError("external_submodels.mace must be a mapping")
    _validate_external_mace_record(mace_external)
    for key in ("sha256", "model_id", "version"):
        if mace_external[key] != config["mace"][key]:
            raise ValueError(f"external MACE {key} disagrees with config")
    _validate_safe_metadata(config, "config")
    _validate_safe_metadata(external, "external_submodels")
    _validate_safe_metadata(checkpoint["metadata"], "metadata")
    return True


def _filter_external_prefixes(
    state_dict: Mapping[str, Any],
    prefixes: tuple[str, ...],
) -> dict[str, Any]:
    if prefixes != MACE_BACKBONE_STATE_PREFIXES:
        raise ValueError("only registered MACE backbone prefixes may be filtered")
    return {
        key: value
        for key, value in state_dict.items()
        if not any(key.startswith(prefix) for prefix in prefixes)
    }


def _model_parameter_counts(
    model: nn.Module,
    external_prefixes: tuple[str, ...],
) -> dict[str, int]:
    named = tuple(model.named_parameters())
    total = sum(parameter.numel() for _, parameter in named)
    trainable = sum(
        parameter.numel() for _, parameter in named if parameter.requires_grad
    )
    external = sum(
        parameter.numel()
        for name, parameter in named
        if any(name.startswith(prefix) for prefix in external_prefixes)
    )
    return {
        "total": total,
        "trainable": trainable,
        "external": external,
        "serialized": total - external,
    }


def create_mace_checkpoint_v3(
    model: nn.Module,
    *,
    config: Mapping[str, Any],
    external_mace: Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a v3 state-only checkpoint with the MACE backbone excluded."""

    unwrapped = unwrap_model(model)
    if type(unwrapped).__name__ != "MACEAP3D3":
        raise ValueError("v3 external-backbone writer requires model_type MACEAP3D3")
    prefixes = tuple(
        getattr(unwrapped, "external_backbone_state_prefixes", ())
    )
    if prefixes != MACE_BACKBONE_STATE_PREFIXES:
        raise ValueError("MACEAP3D3 did not register the exact backbone prefixes")
    external_record = copy.deepcopy(dict(external_mace))
    _validate_external_mace_record(external_record)
    config_record = copy.deepcopy(dict(config))
    config_record["parameter_counts"] = _model_parameter_counts(
        unwrapped, prefixes
    )
    _validate_mace_v3_config(config_record)
    if getattr(unwrapped, "architecture", None) != config_record.get("architecture"):
        raise ValueError("MACE checkpoint architecture does not match the model")
    backbone = unwrapped.featurizer.backbone
    if _model_class_name(backbone) != external_record["model_class"]:
        raise ValueError("external MACE model class does not match the model")
    _validate_reconstructed_mace_model(unwrapped, config_record, backbone)
    state_dict = strip_prefix_from_state_dict(unwrapped.state_dict())
    filtered_state = _filter_external_prefixes(state_dict, prefixes)
    for key, value in filtered_state.items():
        if not torch.is_tensor(value):
            _validate_safe_metadata(value, f"model_state_dict.{key}")
    checkpoint = {
        "checkpoint_version": MACE_CHECKPOINT_VERSION,
        "model_state_dict": filtered_state,
        "config": config_record,
        "model_type": "MACEAP3D3",
        "architecture": config_record.get("architecture"),
        "external_submodels": {"mace": external_record},
        "metadata": {
            "apnet_version": __version__,
            "save_date": datetime.now().isoformat(),
            **dict(metadata or {}),
        },
    }
    validate_mace_checkpoint_v3(checkpoint)
    return checkpoint


def _assert_semantic_subset(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    label: str,
    path: str = "config",
) -> None:
    for key, expected_value in expected.items():
        item_path = f"{path}.{key}"
        if key not in actual:
            raise ValueError(f"{label}: {item_path} is absent")
        actual_value = actual[key]
        if isinstance(expected_value, Mapping):
            if not isinstance(actual_value, Mapping):
                raise ValueError(f"{label}: {item_path} is not a mapping")
            if item_path == "config.route_submodel_digests":
                if dict(actual_value) != dict(expected_value):
                    raise ValueError(f"{label}: {item_path} differs")
                continue
            _assert_semantic_subset(
                actual_value,
                expected_value,
                label=label,
                path=item_path,
            )
        elif actual_value != expected_value:
            raise ValueError(
                f"{label}: {item_path} expected {expected_value!r}, "
                f"got {actual_value!r}"
            )


def _model_class_name(model: nn.Module) -> str:
    return f"{type(model).__module__}.{type(model).__qualname__}"


def _validate_reconstructed_mace_model(
    model: nn.Module,
    config: Mapping[str, Any],
    backbone: nn.Module,
) -> None:
    if getattr(model, "architecture", None) != config["architecture"]:
        raise ValueError("reconstructed model architecture semantic mismatch")
    pair_core = getattr(model, "pair_core", None)
    if getattr(pair_core, "pair_mode", None) != config["pair_mode"]:
        raise ValueError("reconstructed model pair mode semantic mismatch")
    featurizer = getattr(model, "featurizer", None)
    if featurizer is None or getattr(featurizer, "backbone", None) is not backbone:
        raise ValueError("model factory did not install the verified MACE backbone")
    mace = config["mace"]
    checks = {
        "feature_mode": getattr(featurizer, "feature_mode", None),
        "model_id": getattr(featurizer, "model_id", None),
        "version": getattr(featurizer, "mace_version", None),
        "sha256": getattr(featurizer, "checkpoint_sha256", None),
    }
    for key, actual in checks.items():
        if actual != mace[key]:
            raise ValueError(f"reconstructed MACE {key} semantic mismatch")
    resolved_schema = getattr(featurizer, "resolved_feature_schema", None)
    if resolved_schema not in {None, mace["feature_schema"]}:
        raise ValueError("reconstructed MACE feature schema semantic mismatch")
    dtype = str(getattr(featurizer, "dtype", "")).removeprefix("torch.")
    if dtype != config["dtype_policy"]:
        raise ValueError("reconstructed MACE dtype policy semantic mismatch")
    if getattr(model, "atomic_property_schema", None) != config[
        "atomic_property_schema"
    ]:
        raise ValueError("reconstructed atomic property schema semantic mismatch")
    physics_config = getattr(getattr(model, "long_range_provider", None), "config", None)
    if physics_config is not None:
        physics = config["physics"]
        if physics_config.physics_hash != physics["physics_hash"]:
            raise ValueError("reconstructed physics hash semantic mismatch")
        if physics_config.electrostatics_mode != physics["electrostatics_mode"]:
            raise ValueError("reconstructed electrostatics mode semantic mismatch")
        if tuple(physics_config.d3_parameters) != tuple(physics["d3_parameters"]):
            raise ValueError("reconstructed D3 parameters semantic mismatch")


def load_mace_checkpoint_v3(
    checkpoint_path: str | Path,
    *,
    mace_artifact_path: str | Path,
    model_factory: Callable[[Mapping[str, Any], nn.Module], nn.Module],
    backbone_loader: Callable[..., nn.Module],
    semantic_expectations: Mapping[str, Any] | None = None,
    constructor_overrides: Mapping[str, Any] | None = None,
    map_location: str | torch.device = "cpu",
) -> nn.Module:
    """Strictly reconstruct a v3 model around a verified external backbone."""

    checkpoint = torch.load(
        checkpoint_path,
        map_location=map_location,
        weights_only=True,
    )
    validate_mace_checkpoint_v3(checkpoint)
    config = checkpoint["config"]
    if semantic_expectations:
        _assert_semantic_subset(
            config,
            semantic_expectations,
            label="checkpoint semantic mismatch",
        )
    if constructor_overrides:
        _assert_semantic_subset(
            config,
            constructor_overrides,
            label="constructor override conflicts with checkpoint semantics",
        )

    artifact = Path(mace_artifact_path)
    if not artifact.is_file():
        raise FileNotFoundError(f"external MACE artifact not found: {artifact}")
    external = checkpoint["external_submodels"]["mace"]
    actual_digest = sha256_file(artifact)
    if actual_digest != external["sha256"]:
        raise ValueError(
            "external MACE artifact SHA-256 mismatch: "
            f"expected {external['sha256']}, got {actual_digest}"
        )
    backbone = backbone_loader(artifact, map_location=map_location)
    if not isinstance(backbone, nn.Module):
        raise TypeError("external MACE loader did not return an nn.Module")
    actual_class = _model_class_name(backbone)
    if actual_class != external["model_class"]:
        raise TypeError(
            "external MACE model class mismatch: "
            f"expected {external['model_class']}, got {actual_class}"
        )
    model = model_factory(config, backbone)
    if not isinstance(model, nn.Module):
        raise TypeError("MACE model factory did not return an nn.Module")
    _validate_reconstructed_mace_model(model, config, backbone)

    incompatible = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    prefixes = tuple(external["state_prefixes"])
    invalid_missing = [
        key
        for key in incompatible.missing_keys
        if not any(key.startswith(prefix) for prefix in prefixes)
    ]
    if invalid_missing:
        raise RuntimeError(f"missing non-external state key(s): {invalid_missing}")
    if incompatible.unexpected_keys:
        raise RuntimeError(
            f"unexpected checkpoint state key(s): {incompatible.unexpected_keys}"
        )
    counts = _model_parameter_counts(model, MACE_BACKBONE_STATE_PREFIXES)
    if counts != dict(config["parameter_counts"]):
        raise ValueError("reconstructed parameter counts semantic mismatch")
    model.featurizer.backbone.requires_grad_(False)
    model.featurizer.backbone.eval()
    return model
