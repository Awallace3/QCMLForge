"""CLIFF-2: inference-only assembly of the trained classical terms plus D3.

Two public entry points live here:

``merge_classical_parameter_checkpoints``
    Stage-two warm start for the two-stage CLIFF fit.  It folds a
    ``RackersTholeDampingNN`` checkpoint (columns ``elst``, ``thole_direct``,
    ``thole_mutual``, ``ind_overlap``) and a ``CliffExchangeNN`` checkpoint
    (column ``exch``) into a single ``CliffClassicalNN`` checkpoint.  The remap
    is driven entirely by each source checkpoint's recorded
    ``parameter_names``, never by column position.

``CLIFF2Model``
    An inference-only ``nn.Module`` combining a trained ``CliffClassicalNN``
    parameter set with ``qcml_dftd3`` dispersion to emit the four SAPT0
    components plus a total.  There is deliberately no training, optimizer,
    dataset-construction, or DDP code in this file: the constructor freezes and
    ``eval()``s the whole hierarchy and every prediction entry point runs under
    ``torch.inference_mode()``.
"""

import warnings
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn

from qcml_dftd3.d3 import resolve_d3_damping_parameters

from .. import model_io
from ..pt_datasets.ap2_fused_ds import (
    ap2_fused_collate_update_no_target,
    qcel_dimer_to_fused_data,
)
from ..util import scatter_sum_compile
from .mtp_mtp import (
    CLIFF_CLASSICAL_INITIAL_STDS,
    CLIFF_CLASSICAL_INITIAL_VALUES,
    CLIFF_CLASSICAL_PARAMETER_NAMES,
    FULL_EDGE_DIMER_EVAL_MODES,
    OVERLAP_WIDTH_FLOOR,
    POSITIVE_PARAMETER_CONTRACTS,
    CliffClassicalNN,
    DimerProp,
    _rebuild_nested_atom_model,
)

__all__ = [
    "CLIFF2Model",
    "CLIFF2_COMPONENT_LABELS",
    "CLIFF2_MODEL_TYPE",
    "CLIFF2_DIMER_EVAL",
    "merge_classical_parameter_checkpoints",
]

#: ``model_type`` recorded in a ``CLIFF2Model`` checkpoint.  A CLIFF-2
#: checkpoint embeds the *source* classical config plus the resolved D3
#: parameters, so it reloads without its constituent files.
CLIFF2_MODEL_TYPE = "CLIFF2Model"

#: The one ``DimerProp`` mode CLIFF-2 runs.  It is a member of
#: ``FULL_EDGE_DIMER_EVAL_MODES``, so per-dimer aggregation uses
#: ``batch.dimer_ind_full``.
CLIFF2_DIMER_EVAL = "cliff_classical_d3"

#: Column labels of ``predict_batch``'s output: the four SAPT0 components in
#: the dataset's ``y = [Elst, Exch, Ind, Disp]`` order plus a trailing total.
CLIFF2_COMPONENT_LABELS = ("Elst", "Exch", "Indu", "Disp", "Total")

#: ``model_type`` a merged / loadable classical parameter set must declare.
_CLASSICAL_MODEL_TYPE = "CliffClassicalNN"

#: Whether a trained ``dimer_eval`` fitted the short-range induction overlap
#: correction.  ``CLIFF2Model`` reads ``include_overlap`` from here rather than
#: from the caller: running a checkpoint that was never fitted with the overlap
#: term through the overlap route (or the reverse) silently changes the
#: induction energy the parameters were trained to reproduce.
_INCLUDE_OVERLAP_BY_DIMER_EVAL = {
    "cliff_classical": False,
    "cliff_classical_overlap": True,
    "cliff_classical_d3": True,
}

#: Classical ``dimer_eval`` a merged checkpoint inherits from its Rackers
#: source.  The Rackers overlap route is the only one that fits the
#: ``ind_overlap`` column, so the merged model must keep using it.
_MERGED_DIMER_EVAL_BY_RACKERS_MODE = {
    "rackers_thole": "cliff_classical",
    "rackers_thole_overlap": "cliff_classical_overlap",
    # A classical checkpoint may itself be a merge source (re-merging a
    # partially fitted set); its mode passes straight through.
    "cliff_classical": "cliff_classical",
    "cliff_classical_overlap": "cliff_classical_overlap",
}

#: Destination column index for every classical parameter name.  This mapping
#: -- not enumeration order -- is what drives the merge.
_CLASSICAL_INDEX_BY_NAME = {
    name: index
    for index, name in enumerate(CLIFF_CLASSICAL_PARAMETER_NAMES)
}


# ---------------------------------------------------------------------------
# Two-stage fitting route: name-driven checkpoint merge
# ---------------------------------------------------------------------------


def _load_positive_parameter_checkpoint(path, role):
    """Load and contract-validate one positive per-atom parameter checkpoint.

    ``role`` names the constructor argument in error messages so a user can
    tell which of the two paths is at fault.
    """
    checkpoint = model_io.load_checkpoint(path, map_location="cpu")
    version = checkpoint.get("checkpoint_version")
    if version != model_io.CHECKPOINT_VERSION:
        raise ValueError(
            f"{role} checkpoint_version mismatch: expected "
            f"{model_io.CHECKPOINT_VERSION}, got {version!r}"
        )
    model_type = checkpoint.get("model_type")
    if model_type not in POSITIVE_PARAMETER_CONTRACTS:
        raise ValueError(
            f"{role} checkpoint model_type {model_type!r} is not a positive "
            "per-atom parameter head; expected one of "
            f"{sorted(POSITIVE_PARAMETER_CONTRACTS)}"
        )
    model_io.validate_checkpoint(checkpoint, expected_type=model_type)
    config = model_io.load_config_from_checkpoint(checkpoint) or {}
    expected_names = list(POSITIVE_PARAMETER_CONTRACTS[model_type])
    if config.get("parameter_names") != expected_names:
        raise ValueError(
            f"{role} checkpoint parameter_names must exactly match "
            f"{expected_names}"
        )
    unmergeable = [
        name
        for name in expected_names
        if name not in _CLASSICAL_INDEX_BY_NAME
    ]
    if unmergeable:
        raise ValueError(
            f"{role} checkpoint declares parameter names {unmergeable} that "
            "have no column in the classical contract "
            f"{list(CLIFF_CLASSICAL_PARAMETER_NAMES)}"
        )
    if "nested_atom_model" not in config:
        raise ValueError(
            f"{role} checkpoint missing nested_atom_model metadata"
        )
    state = model_io.load_state_dict_from_checkpoint(checkpoint)
    return {
        "role": role,
        "model_type": model_type,
        "config": config,
        "state": state,
        "parameter_names": expected_names,
    }


def _require_agreement(sources, key, default=None):
    """Return one shared config value, raising if the sources disagree."""
    values = [(source, source["config"].get(key, default)) for source in sources]
    first_source, first = values[0]
    for source, value in values[1:]:
        if value != first:
            raise ValueError(
                f"{first_source['role']} and {source['role']} checkpoints "
                f"disagree on {key}: {first!r} != {value!r}"
            )
    return first


def _copy_parameter_column(source, merged_state, src_index, dst_index, name):
    """Copy one parameter column's modules from a source into ``merged_state``.

    Per-parameter state lives in two parallel ``nn.ModuleList``s indexed by
    parameter position -- ``guess_layer[p]`` (the ``[max_Z + 1, 1]``
    per-element embedding) and ``param_readout_layers[p]`` (its ``n_message +
    1`` MLPs) -- so remapping a column is purely a module-index remap of those
    two lists.
    """
    state = source["state"]
    for prefix in ("guess_layer.", "param_readout_layers."):
        src_prefix = f"{prefix}{src_index}."
        dst_prefix = f"{prefix}{dst_index}."
        matched = sorted(key for key in state if key.startswith(src_prefix))
        if not matched:
            raise ValueError(
                f"{source['role']} checkpoint has no {src_prefix}* state for "
                f"parameter {name!r}"
            )
        for key in matched:
            dst_key = dst_prefix + key[len(src_prefix):]
            if dst_key not in merged_state:
                raise ValueError(
                    f"{source['role']} checkpoint key {key!r} has no "
                    f"destination {dst_key!r} in the classical model"
                )
            if merged_state[dst_key].shape != state[key].shape:
                raise ValueError(
                    f"{source['role']} checkpoint key {key!r} has shape "
                    f"{tuple(state[key].shape)}, but destination "
                    f"{dst_key!r} expects "
                    f"{tuple(merged_state[dst_key].shape)}"
                )
            merged_state[dst_key] = state[key].detach().clone()


def merge_classical_parameter_checkpoints(
    rackers_checkpoint_path: str | None,
    exchange_checkpoint_path: str | None,
    output_path: str | None = None,
) -> dict:
    """Merge component parameter checkpoints into one classical checkpoint.

    CLIFF fits its components individually and then refits jointly.  This is
    the plumbing for step two: it maps a ``RackersTholeDampingNN``
    checkpoint's four columns and a ``CliffExchangeNN`` checkpoint's single
    column into the five-column
    :data:`~apnet_pt.AtomPairwiseModels.mtp_mtp.CLIFF_CLASSICAL_PARAMETER_NAMES`
    contract, producing a checkpoint that
    ``CliffClassicalModel(pre_trained_model_path=...)`` (or
    :class:`CLIFF2Model`) loads directly.

    The remap is *name driven*.  Each source's recorded ``parameter_names``
    resolves through :data:`_CLASSICAL_INDEX_BY_NAME` to a destination index,
    and only then are ``guess_layer.{p}`` and ``param_readout_layers.{p}.*``
    copied.  Nothing assumes that source column ``p`` belongs at destination
    column ``p``; that assumption happens to hold for the Rackers columns
    (0-3) purely because the classical contract was defined to mirror them,
    and it is false for exchange (source column 0 -> destination column 4).

    Parameters
    ----------
    rackers_checkpoint_path:
        Path to the electrostatics/induction source, or ``None``.
    exchange_checkpoint_path:
        Path to the exchange source, or ``None``.  At least one of the two
        paths is required.
    output_path:
        Where to write the merged checkpoint.  ``None`` skips the write and
        only returns the checkpoint dictionary, which is how
        :class:`CLIFF2Model` builds an in-memory merged model.

    Returns
    -------
    dict
        The merged ``CliffClassicalNN`` checkpoint in ``model_io`` v-current
        format.

    Raises
    ------
    ValueError
        If neither path is supplied; if a source checkpoint is not a
        positive-parameter head, has absent/reordered/foreign
        ``parameter_names``, or lacks ``nested_atom_model`` metadata; if the
        two sources disagree on the nested ``AtomTypeParamNN`` configuration,
        on ``n_message`` / ``n_neuron`` / ``n_embed``, or on
        ``positivity_epsilon``; if two sources claim the same parameter name;
        or if a copied tensor's shape does not match its destination.
    """
    if rackers_checkpoint_path is None and exchange_checkpoint_path is None:
        raise ValueError(
            "merge_classical_parameter_checkpoints requires at least one of "
            "rackers_checkpoint_path or exchange_checkpoint_path"
        )

    sources = []
    rackers = exchange = None
    if rackers_checkpoint_path is not None:
        rackers = _load_positive_parameter_checkpoint(
            rackers_checkpoint_path, "rackers_checkpoint_path"
        )
        sources.append(rackers)
    if exchange_checkpoint_path is not None:
        exchange = _load_positive_parameter_checkpoint(
            exchange_checkpoint_path, "exchange_checkpoint_path"
        )
        sources.append(exchange)

    # Architecture agreement.  The readout MLPs are copied verbatim, so a
    # disagreement here would either fail a shape check later or -- worse --
    # pass one while meaning something different.
    nested_metadata = _require_agreement(sources, "nested_atom_model")
    n_message = _require_agreement(sources, "n_message")
    n_neuron = _require_agreement(sources, "n_neuron")
    n_embed = _require_agreement(sources, "n_embed")
    # `positivity_epsilon` is part of the raw -> positive mapping
    # (`softplus(raw) + eps`), so copied raw weights only mean the same thing
    # under the same epsilon.
    positivity_epsilon = _require_agreement(sources, "positivity_epsilon")
    # Only the CLIFF heads record `width_floor`; a Rackers-only merge inherits
    # the module default.
    width_floor = OVERLAP_WIDTH_FLOOR
    for source in sources:
        if "width_floor" in source["config"]:
            width_floor = source["config"]["width_floor"]
            break

    # Destination initialization.  Claimed columns are overwritten below;
    # unclaimed ones keep exactly these values, which is the documented
    # behavior of a partial merge.
    param_start_mean = list(CLIFF_CLASSICAL_INITIAL_VALUES)
    param_start_std = list(CLIFF_CLASSICAL_INITIAL_STDS)
    for source in sources:
        source_means = source["config"].get("param_start_mean") or []
        source_stds = source["config"].get("param_start_std") or []
        for position, name in enumerate(source["parameter_names"]):
            index = _CLASSICAL_INDEX_BY_NAME[name]
            if position < len(source_means):
                param_start_mean[index] = source_means[position]
            if position < len(source_stds):
                param_start_std[index] = source_stds[position]

    destination = CliffClassicalNN(
        atom_model=_rebuild_nested_atom_model(
            deepcopy(nested_metadata), freeze_atom_model=True
        ),
        n_message=n_message,
        n_neuron=n_neuron,
        n_embed=n_embed,
        param_start_mean=param_start_mean,
        param_start_std=param_start_std,
        positivity_epsilon=positivity_epsilon,
        width_floor=width_floor,
        freeze_atom_model=True,
    )

    merged_state = {
        key: value.detach().clone()
        for key, value in destination.state_dict().items()
    }

    claimed_by: dict[str, str] = {}
    for source in sources:
        for position, name in enumerate(source["parameter_names"]):
            if name in claimed_by:
                raise ValueError(
                    f"parameter {name!r} is claimed by both the "
                    f"{claimed_by[name]} and {source['role']} checkpoints; "
                    "the merge would be ambiguous"
                )
            claimed_by[name] = source["role"]
            _copy_parameter_column(
                source,
                merged_state,
                src_index=position,
                dst_index=_CLASSICAL_INDEX_BY_NAME[name],
                name=name,
            )

    # The nested HFVR / valence-width model feeds every column, so exactly one
    # of the sources can supply it.  Prefer the Rackers checkpoint and say so
    # when the two actually differ (which only happens if one stage was run
    # with the nested model unfrozen).
    nested_source = rackers or exchange
    other = exchange if nested_source is rackers else rackers
    _copy_nested_atom_model_state(nested_source, merged_state)
    if other is not None and not _nested_states_agree(nested_source, other):
        warnings.warn(
            "nested atom_model weights differ between the "
            f"{nested_source['role']} and {other['role']} checkpoints; "
            f"keeping the {nested_source['role']} weights",
            RuntimeWarning,
            stacklevel=2,
        )

    destination.load_state_dict(merged_state)

    rackers_mode = (rackers or {}).get("config", {}).get("dimer_eval")
    dimer_eval = _MERGED_DIMER_EVAL_BY_RACKERS_MODE.get(
        rackers_mode, "cliff_classical"
    )
    d3_damping_parameters = None
    for source in reversed(sources):
        if source["config"].get("d3_damping_parameters") is not None:
            d3_damping_parameters = source["config"]["d3_damping_parameters"]
            break

    config = destination.get_config()
    config["elst_damping_type"] = _first_config_value(
        sources, "elst_damping_type", "CLIFF"
    )
    config["dimer_eval"] = dimer_eval
    config["dimer_eval_type"] = dimer_eval
    config["d3_damping_parameters"] = resolve_d3_damping_parameters(
        d3_damping_parameters
    )
    # A merged checkpoint is a warm start, not a completed joint fit, so the
    # loss weighting is reset to the neutral default.
    config["component_gamma"] = None
    config["total_includes_d3"] = False

    checkpoint = model_io.create_checkpoint(
        model=destination,
        config=config,
        model_type=_CLASSICAL_MODEL_TYPE,
        submodels={
            "atom_model": model_io.create_submodel_checkpoint(
                model=destination.atom_model,
                config=destination.atom_model.get_config(),
                model_type=type(destination.atom_model).__name__,
            )
        },
        metadata={
            "merged_from": {
                source["role"]: {
                    "model_type": source["model_type"],
                    "parameter_names": list(source["parameter_names"]),
                }
                for source in sources
            },
            "merged_columns": {
                name: _CLASSICAL_INDEX_BY_NAME[name]
                for name in claimed_by
            },
            "unclaimed_columns": [
                name
                for name in CLIFF_CLASSICAL_PARAMETER_NAMES
                if name not in claimed_by
            ],
            "nested_state_source": nested_source["role"],
        },
    )
    if output_path is not None:
        model_io.save_checkpoint(checkpoint, output_path)
    return checkpoint


def _first_config_value(sources, key, default):
    for source in sources:
        if source["config"].get(key) is not None:
            return source["config"][key]
    return default


def _nested_atom_model_keys(state):
    return sorted(key for key in state if key.startswith("atom_model."))


def _copy_nested_atom_model_state(source, merged_state):
    for key in _nested_atom_model_keys(source["state"]):
        if key not in merged_state:
            raise ValueError(
                f"{source['role']} checkpoint key {key!r} has no destination "
                "in the classical model"
            )
        value = source["state"][key]
        if merged_state[key].shape != value.shape:
            raise ValueError(
                f"{source['role']} checkpoint key {key!r} has shape "
                f"{tuple(value.shape)}, but destination expects "
                f"{tuple(merged_state[key].shape)}"
            )
        merged_state[key] = value.detach().clone()


def _nested_states_agree(first, second):
    first_keys = _nested_atom_model_keys(first["state"])
    if first_keys != _nested_atom_model_keys(second["state"]):
        return False
    return all(
        torch.equal(first["state"][key], second["state"][key])
        for key in first_keys
    )


# ---------------------------------------------------------------------------
# CLIFF2Model
# ---------------------------------------------------------------------------


class _CliffClassicalD3DimerProp(DimerProp):
    """``cliff_classical_d3`` whose overlap term follows the checkpoint.

    ``DimerProp._cliff_classical_d3_forward`` hard-codes
    ``include_overlap=True`` because that is the right default for the CLIFF-2
    assembly.  A parameter set fitted by ``CliffClassicalModel`` (no overlap)
    must nevertheless be evaluated without it, so this subclass overrides the
    mode's forward -- keeping its name, and therefore
    ``DimerProp.get_config()["dimer_eval"]``, unchanged -- and reads
    ``include_overlap`` off the instance instead.
    """

    def __init__(
        self,
        ATParam,
        include_overlap: bool,
        elst_damping_type: str = "CLIFF",
        d3_damping_parameters=None,
    ):
        super().__init__(
            ATParam=ATParam,
            dimer_eval=CLIFF2_DIMER_EVAL,
            elst_damping_type=elst_damping_type,
            d3_damping_parameters=d3_damping_parameters,
            freeze_atom_model=True,
        )
        self.include_overlap = bool(include_overlap)

    def _cliff_classical_d3_forward(self, batch):
        return self._cliff_classical_common_forward(
            batch,
            include_overlap=self.include_overlap,
            include_d3=True,
        )


class CLIFF2Model(nn.Module):
    """Inference-only CLIFF-2: classical MPNN parameters plus D3 dispersion.

    Combines a trained ``CliffClassicalNN`` parameter set (electrostatics,
    exchange, induction) with ``qcml_dftd3`` dispersion into a single
    four-component SAPT0 prediction plus total.  It doubles as an
    MPNN-parameterized advanced force field and as the classical baseline that
    AP3-D3 has to beat.

    Construction takes exactly one of two forms:

    * ``classical_model_path`` -- a single ``CliffClassicalNN`` checkpoint
      (or a previously saved :class:`CLIFF2Model` checkpoint), or
    * ``rackers_model_path`` and/or ``exchange_model_path`` -- component
      checkpoints, routed through
      :func:`merge_classical_parameter_checkpoints` into an in-memory merged
      model.

    Supplying both forms, or neither, raises ``ValueError``.

    ``include_overlap`` is deliberately *not* a constructor argument: it is
    read from the loaded checkpoint's ``dimer_eval``, because evaluating a
    parameter set with an induction overlap term it was never fitted with (or
    without one it was) silently changes the physics.

    The whole hierarchy is ``eval()``-ed and ``requires_grad_(False)``-ed at
    construction, and every prediction entry point runs under
    ``torch.inference_mode()``.
    """

    DIMER_EVAL = CLIFF2_DIMER_EVAL
    MODEL_TYPE = CLIFF2_MODEL_TYPE
    #: ``predict_batch`` column labels, components first and total last.
    component_labels = CLIFF2_COMPONENT_LABELS

    def __init__(
        self,
        classical_model_path=None,
        rackers_model_path=None,
        exchange_model_path=None,
        d3_damping_parameters=None,
        use_GPU=None,
        merged_output_path=None,
    ):
        """
        Parameters
        ----------
        classical_model_path:
            A single trained ``CliffClassicalNN`` checkpoint, or a checkpoint
            previously written by :meth:`save_model`.
        rackers_model_path, exchange_model_path:
            Component checkpoints merged in memory by
            :func:`merge_classical_parameter_checkpoints`.  Mutually exclusive
            with ``classical_model_path``.
        d3_damping_parameters:
            Optional override forwarded to ``resolve_d3_damping_parameters``.
            Defaults to the value recorded in the loaded checkpoint config.
        use_GPU:
            ``False`` forces CPU; ``None`` uses CUDA when available.
        merged_output_path:
            Optional path at which to also persist the merged checkpoint when
            constructing from component checkpoints.
        """
        super().__init__()

        has_single = classical_model_path is not None
        has_components = (
            rackers_model_path is not None or exchange_model_path is not None
        )
        if has_single and has_components:
            raise ValueError(
                "CLIFF2Model accepts either classical_model_path or "
                "rackers_model_path/exchange_model_path, not both"
            )
        if not has_single and not has_components:
            raise ValueError(
                "CLIFF2Model requires either classical_model_path or at least "
                "one of rackers_model_path/exchange_model_path"
            )

        if torch.cuda.is_available() and use_GPU is not False:
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")

        if has_single:
            config, state_dict = self._load_classical_source(
                classical_model_path
            )
        else:
            checkpoint = merge_classical_parameter_checkpoints(
                rackers_model_path,
                exchange_model_path,
                merged_output_path,
            )
            config = model_io.load_config_from_checkpoint(checkpoint) or {}
            state_dict = model_io.load_state_dict_from_checkpoint(checkpoint)

        self._validate_classical_config(config)
        dimer_eval = config["dimer_eval"]

        self.classical_config = deepcopy(config)
        self.include_overlap = _INCLUDE_OVERLAP_BY_DIMER_EVAL[dimer_eval]
        self.source_dimer_eval = dimer_eval
        self.elst_damping_type = config.get("elst_damping_type", "CLIFF")
        self.d3_damping_parameters = resolve_d3_damping_parameters(
            d3_damping_parameters
            if d3_damping_parameters is not None
            else config.get("d3_damping_parameters")
        )

        self.model = CliffClassicalNN(
            atom_model=_rebuild_nested_atom_model(
                deepcopy(config["nested_atom_model"]),
                freeze_atom_model=True,
            ),
            n_message=config["n_message"],
            n_neuron=config["n_neuron"],
            n_embed=config["n_embed"],
            param_start_mean=config["param_start_mean"],
            param_start_std=config["param_start_std"],
            positivity_epsilon=config["positivity_epsilon"],
            width_floor=config.get("width_floor", OVERLAP_WIDTH_FLOOR),
            freeze_atom_model=True,
        )
        self.model.load_state_dict(state_dict)

        self.dimer_model = _CliffClassicalD3DimerProp(
            ATParam=self.model,
            include_overlap=self.include_overlap,
            elst_damping_type=self.elst_damping_type,
            d3_damping_parameters=self.d3_damping_parameters,
        )

        self.device = device
        self.to(device)
        # Inference posture: no gradients anywhere, no training-mode modules.
        self.eval()
        self.requires_grad_(False)

    # -- construction helpers ------------------------------------------

    @classmethod
    def from_checkpoint(cls, path, d3_damping_parameters=None, use_GPU=None):
        """Reload a :meth:`save_model` checkpoint without its source files."""
        return cls(
            classical_model_path=path,
            d3_damping_parameters=d3_damping_parameters,
            use_GPU=use_GPU,
        )

    @staticmethod
    def _load_classical_source(path):
        """Return ``(classical_config, state_dict)`` for a checkpoint path.

        Accepts both a ``CliffClassicalNN`` checkpoint and a
        :class:`CLIFF2Model` checkpoint; the latter embeds the source
        classical config plus the resolved D3 parameters, so unwrapping it here
        is what lets a CLIFF-2 model round-trip without its constituent files.
        """
        checkpoint = model_io.load_checkpoint(path, map_location="cpu")
        version = checkpoint.get("checkpoint_version")
        if version != model_io.CHECKPOINT_VERSION:
            raise ValueError(
                f"classical checkpoint_version mismatch: expected "
                f"{model_io.CHECKPOINT_VERSION}, got {version!r}"
            )
        model_type = checkpoint.get("model_type")
        state_dict = model_io.load_state_dict_from_checkpoint(checkpoint)
        config = model_io.load_config_from_checkpoint(checkpoint) or {}
        if model_type == CLIFF2_MODEL_TYPE:
            classical_config = config.get("classical_config")
            if not isinstance(classical_config, dict):
                raise ValueError(
                    "CLIFF2Model checkpoint missing classical_config metadata"
                )
            classical_config = deepcopy(classical_config)
            # The resolved (possibly overridden) D3 parameters live at the top
            # level of a CLIFF-2 config and take precedence over whatever the
            # source classical checkpoint happened to record.
            classical_config["d3_damping_parameters"] = config.get(
                "d3_damping_parameters"
            )
            return classical_config, state_dict
        if model_type != _CLASSICAL_MODEL_TYPE:
            raise ValueError(
                f"classical checkpoint model_type mismatch: expected "
                f"{_CLASSICAL_MODEL_TYPE} or {CLIFF2_MODEL_TYPE}, got "
                f"{model_type!r}"
            )
        model_io.validate_checkpoint(
            checkpoint, expected_type=_CLASSICAL_MODEL_TYPE
        )
        return config, state_dict

    @staticmethod
    def _validate_classical_config(config):
        expected_names = list(CLIFF_CLASSICAL_PARAMETER_NAMES)
        if config.get("model_type") != _CLASSICAL_MODEL_TYPE:
            raise ValueError(
                "classical checkpoint config model_type mismatch: expected "
                f"{_CLASSICAL_MODEL_TYPE}, got {config.get('model_type')!r}"
            )
        if config.get("parameter_names") != expected_names:
            raise ValueError(
                "classical checkpoint parameter_names must exactly match "
                f"{expected_names}"
            )
        if "nested_atom_model" not in config:
            raise ValueError(
                "classical checkpoint missing nested_atom_model metadata"
            )
        dimer_eval = config.get("dimer_eval")
        if dimer_eval not in _INCLUDE_OVERLAP_BY_DIMER_EVAL:
            raise ValueError(
                f"classical checkpoint dimer_eval {dimer_eval!r} is not a "
                "CLIFF classical mode; expected one of "
                f"{sorted(_INCLUDE_OVERLAP_BY_DIMER_EVAL)}"
            )

    # -- inference -----------------------------------------------------

    def forward(self, batch):
        """Per-edge ``[n_edges, 4]`` energies, ``(Elst, Exch, Indu, Disp)``.

        Columns are in kcal/mol on the full intermolecular edge domain
        (``e_ABfull_*``); aggregate them with
        :meth:`_dimer_index_for_output`, or just call :meth:`predict_batch`.
        """
        edge_energy, _, _ = self.dimer_model(batch)
        return edge_energy

    def _dimer_index_for_output(self, batch):
        """Per-edge dimer index, matching ``AM_DimerParam_Model``'s convention.

        ``cliff_classical_d3`` is a member of ``FULL_EDGE_DIMER_EVAL_MODES``,
        so this resolves to ``batch.dimer_ind_full``; the membership test
        rather than a literal keeps the training and inference aggregation from
        drifting apart.
        """
        if self.DIMER_EVAL in FULL_EDGE_DIMER_EVAL_MODES:
            return batch.dimer_ind_full
        return batch.dimer_ind

    @torch.inference_mode()
    def predict_batch(self, batch):
        """Per-dimer ``[n_dimers, 5]``: four components then their total."""
        edge_energy, _, _ = self.dimer_model(batch)
        components = scatter_sum_compile(
            edge_energy,
            self._dimer_index_for_output(batch),
            dim_size=batch.total_charge_A.size(0),
        )
        total = components.sum(dim=-1, keepdim=True)
        return torch.cat((components, total), dim=-1)

    def _r_cut(self):
        """Deepest declared ``r_cut`` in the nested hierarchy."""
        current = self.model
        r_cut = None
        while current is not None:
            if hasattr(current, "r_cut"):
                r_cut = current.r_cut
            current = getattr(current, "atom_model", None)
        return 5.0 if r_cut is None else r_cut

    @torch.inference_mode()
    def predict_qcel_mols_dimer(
        self,
        mols,
        batch_size=1,
        r_cut=None,
        verbose=False,
    ):
        """Predict per-dimer components and total for qcel dimer molecules.

        Mirrors ``AM_DimerParam_Model.predict_qcel_mols_dimer``: ``mols`` is an
        iterable of qcel dimers, ``batch_size`` dimers are processed per
        forward pass, ``r_cut`` defaults to the nested atom model's own cutoff,
        and the return value is a ``numpy.ndarray`` of shape ``(N, M)``.  Here
        ``M`` is always 5, labelled by :attr:`component_labels`.
        """
        if r_cut is None:
            r_cut = self._r_cut()
        n_dimers = len(mols)
        predictions = np.zeros((n_dimers, len(self.component_labels)))
        for start in range(0, n_dimers, batch_size):
            stop = min(start + batch_size, n_dimers)
            dimer_batch = ap2_fused_collate_update_no_target(
                [
                    qcel_dimer_to_fused_data(
                        dimer, r_cut=r_cut, dimer_ind=n, r_cut_im=torch.inf
                    )
                    for n, dimer in enumerate(mols[start:stop])
                ]
            )
            dimer_batch.to(device=self.device)
            preds = self.predict_batch(dimer_batch)
            predictions[start:stop] = preds.cpu().numpy().reshape(
                stop - start, -1
            )
            if verbose:
                print(f"Predictions for {start} to {stop} out of {n_dimers}")
        return predictions

    # -- checkpoint / reporting ----------------------------------------

    def get_config(self) -> dict:
        """Config recorded in a CLIFF-2 checkpoint.

        Carries the *source* classical checkpoint config verbatim plus the
        resolved D3 parameters, which is what lets a saved CLIFF-2 model be
        reloaded without any of its constituent files.
        """
        return {
            "model_type": CLIFF2_MODEL_TYPE,
            "classical_config": deepcopy(self.classical_config),
            "d3_damping_parameters": deepcopy(self.d3_damping_parameters),
            "elst_damping_type": self.elst_damping_type,
            "dimer_eval": self.DIMER_EVAL,
            "source_dimer_eval": self.source_dimer_eval,
            "include_overlap": self.include_overlap,
            "component_labels": list(self.component_labels),
        }

    def _create_checkpoint(self, metadata: dict | None = None) -> dict:
        return model_io.create_checkpoint(
            model=self.model,
            config=self.get_config(),
            model_type=CLIFF2_MODEL_TYPE,
            metadata=metadata,
        )

    def save_model(self, path: str, metadata: dict | None = None) -> None:
        """Write a self-contained CLIFF-2 checkpoint."""
        model_io.save_checkpoint(self._create_checkpoint(metadata), path)

    def info(self):
        """Print a Unicode model tree for this model."""
        from apnet_pt.model_print import model_tree_string

        print(model_tree_string(self, unicode=True))
