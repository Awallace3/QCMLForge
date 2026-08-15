"""Normalized registry, validation, and injectable MACE/AP3D3 factories."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping
import warnings

import torch


MACE_AP3D3_OPTIONS = {
    "MACE-AP3D3-DirectPolar": {
        "properties": "direct",
        "pair_mode": "h1",
        "feature_mode": "all-scalars+norms",
    },
    "MACE-AP3D3-H1": {
        "properties": "legacy",
        "pair_mode": "h1",
        "feature_mode": "final-layer-scalars",
    },
    "MACE-AP3D3-H2": {
        "properties": "legacy",
        "pair_mode": "h2",
        "feature_mode": "all-scalars+norms",
    },
    "MACE-AP3D3-AtomHead": {
        "properties": "atomhead",
        "pair_mode": "h1",
        "feature_mode": "all-scalars+norms",
    },
}

_INTERNAL_ARCHITECTURES = {
    "MACE-AP3D3-DirectPolar": "direct-polar",
    "MACE-AP3D3-H1": "hybrid-h1",
    "MACE-AP3D3-H2": "hybrid-h2",
    "MACE-AP3D3-AtomHead": "atomhead",
}

_FEATURE_MODES = {"final-layer-scalars", "all-scalars+norms"}
_ATOMIC_OPTION = "MACE-AtomicProperties"
_D3_PRESETS = {
    "default": (),
    "sapt-pbe0-d3": (1.0, 0.8614, 0.7171, 0.5375),
}


@dataclass(frozen=True)
class ResolvedMACEOption:
    public_name: str
    internal_architecture: str
    properties: str
    pair_mode: str
    feature_mode: str
    run_name: str
    is_ablation: bool


@dataclass(frozen=True)
class MACETrainingPlan:
    kind: str
    public_name: str
    internal_architecture: str
    run_name: str
    is_ablation: bool
    properties: str
    pair_mode: str
    feature_mode: str
    mace_model: str
    mace_model_path: str | None
    mace_sha256: str
    mace_default_dtype: str
    mace_cache_dir: str | None
    mace_offline: bool
    mace_atom_model_path: str | None
    property_mode: str
    train_atomic_heads: bool
    freeze_mace_backbone: bool
    long_range_elst: str
    d3_parameters: tuple[float, ...]
    physics_hash: str
    data_hash: str
    preprocessing_hash: str
    split_hash: str
    route_submodel_digests: Mapping[str, str]
    legacy_model_paths: Mapping[str, str]
    output_path: str
    random_seed: int
    n_epochs: int
    learning_rate: float
    overwrite: bool
    resume: bool
    world_size_ddp: int
    omp_num_threads: int
    no_disp_nn: bool
    include_total_mse: bool
    use_precomputed_classical: bool | None
    unfreeze_dimer_prop_model: bool
    unfreeze_atom_model: bool
    build_dataset_only: bool
    skip_compile: bool
    dataloader_num_workers: int | None
    smoke_data_path: str | None
    smoke_atom_data_path: str | None
    r_cut: float
    r_cut_im: float
    neural_cutoff: float
    batch_size: int
    device: str


@dataclass
class MACEFactoryDependencies:
    """Injectable construction seams used by tests and later CLI lifecycle work."""

    featurizer_builder: Callable[[MACETrainingPlan], Any] | None = None
    property_provider_builder: Callable[[MACETrainingPlan, Any], Any] | None = None
    pair_core_builder: Callable[[MACETrainingPlan], Any] | None = None
    long_range_builder: Callable[[MACETrainingPlan], Any] | None = None
    model_builder: Callable[[MACETrainingPlan, Any, Any, Any, Any], Any] | None = None
    dataset_builder: Callable[[MACETrainingPlan], Any] | None = None
    lifecycle_runner: Callable[[MACETrainingPlan, Any, Any], Any] | None = None


@dataclass(frozen=True)
class MACEFactoryResult:
    plan: MACETrainingPlan
    harness: Any
    dataset: Any
    lifecycle: Any = None


def looks_like_mace_option(value: str) -> bool:
    return value.lower().startswith("mace-")


def resolve_mace_option(
    name: str,
    *,
    feature_mode: str = "auto",
) -> ResolvedMACEOption:
    if name not in MACE_AP3D3_OPTIONS:
        raise ValueError(
            "MACE architecture must use an exact case-sensitive canonical name: "
            + ", ".join(MACE_AP3D3_OPTIONS)
        )
    record = MACE_AP3D3_OPTIONS[name]
    canonical_mode = record["feature_mode"]
    resolved_mode = canonical_mode if feature_mode == "auto" else feature_mode
    if resolved_mode not in _FEATURE_MODES:
        raise ValueError(f"unsupported MACE feature mode: {resolved_mode}")
    is_ablation = resolved_mode != canonical_mode
    run_name = name
    if is_ablation:
        run_name = f"{name}__ablation-feature-{resolved_mode}"
    return ResolvedMACEOption(
        public_name=name,
        internal_architecture=_INTERNAL_ARCHITECTURES[name],
        properties=record["properties"],
        pair_mode=record["pair_mode"],
        feature_mode=resolved_mode,
        run_name=run_name,
        is_ablation=is_ablation,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_record(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _hash_data_source(path_value: str | None, fallback: str) -> str:
    if path_value is None:
        return _hash_record({"locator": fallback})
    path = Path(path_value)
    if path.is_file():
        return _sha256_file(path)
    if path.is_dir():
        entries = []
        for child in sorted(item for item in path.rglob("*") if item.is_file()):
            entries.append((str(child.relative_to(path)), _sha256_file(child)))
        return _hash_record(entries)
    return _hash_record({"missing_locator": str(path)})


def _require_file(value: str | None, label: str) -> Path:
    if not value:
        raise ValueError(f"{label} is required")
    path = Path(value)
    if not path.is_file():
        raise FileNotFoundError(f"{label} was not found: {path}")
    return path


def _resolve_d3_parameters(value: str) -> tuple[float, ...]:
    if value in _D3_PRESETS:
        return _D3_PRESETS[value]
    path = Path(value)
    if path.is_file():
        raw = json.loads(path.read_text())
        if isinstance(raw, Mapping):
            raw = [raw[key] for key in ("s6", "s8", "a1", "a2")]
        values = tuple(float(item) for item in raw)
    else:
        try:
            values = tuple(float(item.strip()) for item in value.split(","))
        except ValueError as exc:
            raise ValueError(
                "d3_params must be a named preset, JSON file, or four numbers"
            ) from exc
    if len(values) != 4:
        raise ValueError("d3_params must contain exactly four values")
    return values


def _resolve_mace_artifact(args: Any) -> tuple[str | None, str]:
    path_value = args.mace_model_path
    supplied_digest = args.mace_model_sha256
    if path_value:
        path = _require_file(path_value, "mace_model_path")
        if not supplied_digest:
            raise ValueError("mace_model_sha256 is required with mace_model_path")
        actual = _sha256_file(path)
        if actual != supplied_digest:
            raise ValueError(
                f"MACE artifact SHA-256 mismatch: expected {supplied_digest}, got {actual}"
            )
        return str(path), supplied_digest
    if args.mace_offline:
        raise FileNotFoundError(
            "mace_offline requires a local --mace_model_path and exact SHA-256"
        )
    if args.mace_model != "polar-1-s":
        raise ValueError(
            "unknown MACE model; provide an explicit local artifact and SHA-256"
        )
    # Canonical registry identity; downloading remains the later lifecycle's job.
    canonical = "e4495612037b3b3312633182882a38a694ecac9ea0be2b9889ac0b2a84a99510"
    if supplied_digest and supplied_digest != canonical:
        raise ValueError("mace_model_sha256 disagrees with canonical polar-1-s")
    return None, canonical


def _route_digests(
    args: Any, properties: str, training_heads: bool
) -> dict[str, str]:
    if properties == "legacy":
        values = {
            "atom_model": args.am_model_path,
            "atom_type_param_model": args.atom_type_param_model_path,
            "dimer_param_model": args.atom_type_param_model_path2,
        }
        if any(not value for value in values.values()):
            raise ValueError("H1/H2 require all three legacy model paths")
        return {
            name: _sha256_file(_require_file(path, name))
            for name, path in values.items()
        }
    if training_heads:
        return {"atomic_heads": _hash_record({"trained_in_run": True})}
    path = _require_file(args.mace_atom_model_path, "mace_atom_model_path")
    return {"mace_atom_model": _sha256_file(path)}


def _resolved_option_from_args(args: Any) -> ResolvedMACEOption:
    if args.train_apnet:
        return resolve_mace_option(
            args.train_apnet,
            feature_mode=args.mace_feature_mode,
        )
    if args.train_am != _ATOMIC_OPTION:
        raise ValueError(
            "MACE atom training must use exact option MACE-AtomicProperties"
        )
    feature_mode = args.mace_feature_mode
    if feature_mode == "auto":
        feature_mode = "all-scalars+norms"
    if feature_mode != "all-scalars+norms":
        raise ValueError("MACE atomic properties require all-scalars+norms")
    return ResolvedMACEOption(
        public_name=_ATOMIC_OPTION,
        internal_architecture="atomic-properties",
        properties="atomhead" if args.mace_property_mode == "learned" else "direct",
        pair_mode="",
        feature_mode=feature_mode,
        run_name=_ATOMIC_OPTION,
        is_ablation=False,
    )


def _make_plan(args: Any, *, emit_warning: bool) -> MACETrainingPlan:
    resolved = _resolved_option_from_args(args)
    if getattr(args, "resume", False):
        raise ValueError(
            "resume is not implemented for MACE routes; start a new explicit run"
        )
    if not getattr(args, "skip_compile", False):
        raise ValueError(
            "MACE eager execution requires --skip_compile; compile is unsupported"
        )
    device_policy = getattr(args, "mace_device", "auto")
    if device_policy not in {"cpu", "cuda", "auto"}:
        raise ValueError("mace_device must be cpu, cuda, or auto")
    if device_policy == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = (
        "cuda"
        if device_policy == "cuda"
        or (device_policy == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    if args.train_apnet and args.train_am:
        raise ValueError("MACE atom and pair training cannot run simultaneously")
    if args.world_size_ddp < 1:
        raise ValueError("world_size_ddp must be positive")
    if args.omp_num_threads < 1:
        raise ValueError("omp_num_threads must be positive")
    selected_epochs = args.n_epochs if args.train_apnet else args.n_epochs_atom
    if selected_epochs < 1:
        raise ValueError("n_epochs must be positive")
    selected_smoke_path = (
        args.smoke_data_path if args.train_apnet else args.smoke_atom_data_path
    )
    if selected_smoke_path and not args.build_dataset_only and selected_epochs != 1:
        raise ValueError("checked smoke lifecycles require exactly one epoch")
    if args.lr <= 0:
        raise ValueError("lr must be positive")
    if args.dataloader_num_workers is not None and args.dataloader_num_workers < 0:
        raise ValueError("dataloader_num_workers must be non-negative")
    if args.batch_size < 1:
        raise ValueError("batch_size must be positive")
    if args.mace_default_dtype not in {"float32", "float64"}:
        raise ValueError("mace_default_dtype must be float32 or float64")
    if args.mace_property_mode not in {"direct-completion", "learned"}:
        raise ValueError("mace_property_mode must be direct-completion or learned")
    if args.long_range_elst not in {"damped-cliff", "damped-amoeba", "undamped"}:
        raise ValueError("long_range_elst is unsupported")
    if args.long_range_elst == "damped-amoeba":
        raise ValueError(
            "damped-amoeba requires explicit per-atom AMOEBA damping inputs; "
            "no current MACE route dataset supplies them"
        )
    if args.train_apnet and (args.r_cut != 5.0 or args.r_cut_im != 8.0):
        raise ValueError(
            "canonical MACE/AP3D3 requires r_cut=5.0 and r_cut_im/neural "
            "cutoff=8.0"
        )
    if args.use_precomputed_classical:
        raise ValueError(
            "requested precomputed classical path requires dataset ledgers "
            "with a matching PhysicsConfig hash; this dataset has none"
        )
    if args.resume and args.overwrite:
        raise ValueError("overwrite and resume are mutually exclusive")
    if args.ap_pretrained_model_path and args.train_apnet:
        raise ValueError("MACE routes use explicit --resume, not ap_pretrained_model_path")

    mace_path, mace_digest = _resolve_mace_artifact(args)
    if resolved.properties == "legacy":
        if args.mace_atom_model_path:
            raise ValueError("H1/H2 rejects mace_atom_model_path")
    elif args.train_apnet and not args.mace_atom_model_path and not args.train_atomic_heads:
        raise ValueError(
            "DirectPolar/AtomHead requires mace_atom_model_path unless training heads"
        )
    if (
        (args.train_atomic_heads or not args.train_apnet)
        and not args.smoke_atom_data_path
        and emit_warning
    ):
        warnings.warn(
            "train_atomic_heads has no atomic labels; run is code-validation only",
            RuntimeWarning,
            stacklevel=3,
        )
    if args.smoke_atom_data_path:
        _require_file(args.smoke_atom_data_path, "smoke_atom_data_path")
    if args.smoke_data_path:
        _require_file(args.smoke_data_path, "smoke_data_path")

    training_heads = args.train_atomic_heads or not args.train_apnet
    route_digests = _route_digests(args, resolved.properties, training_heads)
    d3_parameters = _resolve_d3_parameters(args.d3_params)
    from apnet_pt.mace.schema import PhysicsConfig

    physics = PhysicsConfig(
        electrostatics_mode=args.long_range_elst,
        d3_parameters=d3_parameters,
        neural_cutoff=args.r_cut_im,
    )
    smoke_path = (
        args.smoke_data_path if args.train_apnet else args.smoke_atom_data_path
    )
    if smoke_path:
        from apnet_pt.training.smoke import load_smoke_fixture_metadata

        smoke_metadata = load_smoke_fixture_metadata(smoke_path)
        data_hash = smoke_metadata["content_hash"]
        preprocessing_hash = smoke_metadata["preprocessing_hash"]
        split_hash = smoke_metadata["split_hash"]
    else:
        data_hash = _hash_data_source(None, args.data_dir)
        preprocessing_hash = _hash_record(
            {
                "r_cut": args.r_cut,
                "r_cut_im": args.r_cut_im,
                "spec_type": args.spec_type_ap,
                "feature_mode": resolved.feature_mode,
                "use_precomputed_classical": args.use_precomputed_classical,
            }
        )
        split_hash = _hash_record(
            {"data_hash": data_hash, "seed": args.random_seed, "split": 0.9}
        )
    output_path = args.am_model_path if not args.train_apnet else args.ap_model_path
    legacy_model_paths = {}
    if resolved.properties == "legacy":
        legacy_model_paths = {
            "atom_model": args.am_model_path,
            "atom_type_param_model": args.atom_type_param_model_path,
            "dimer_param_model": args.atom_type_param_model_path2,
        }
    return MACETrainingPlan(
        kind="pair" if args.train_apnet else "atomic",
        public_name=resolved.public_name,
        internal_architecture=resolved.internal_architecture,
        run_name=resolved.run_name,
        is_ablation=resolved.is_ablation,
        properties=resolved.properties,
        pair_mode=resolved.pair_mode,
        feature_mode=resolved.feature_mode,
        mace_model=args.mace_model,
        mace_model_path=mace_path,
        mace_sha256=mace_digest,
        mace_default_dtype=args.mace_default_dtype,
        mace_cache_dir=args.mace_cache_dir,
        mace_offline=args.mace_offline,
        mace_atom_model_path=args.mace_atom_model_path,
        property_mode=args.mace_property_mode,
        train_atomic_heads=training_heads,
        freeze_mace_backbone=True,
        long_range_elst=args.long_range_elst,
        d3_parameters=d3_parameters,
        physics_hash=physics.physics_hash,
        data_hash=data_hash,
        preprocessing_hash=preprocessing_hash,
        split_hash=split_hash,
        route_submodel_digests=route_digests,
        legacy_model_paths=legacy_model_paths,
        output_path=output_path,
        random_seed=args.random_seed,
        n_epochs=args.n_epochs if args.train_apnet else args.n_epochs_atom,
        learning_rate=args.lr,
        overwrite=args.overwrite,
        resume=args.resume,
        world_size_ddp=args.world_size_ddp,
        omp_num_threads=args.omp_num_threads,
        no_disp_nn=args.no_disp_nn,
        include_total_mse=args.include_total_mse,
        use_precomputed_classical=args.use_precomputed_classical,
        unfreeze_dimer_prop_model=args.unfreeze_dimer_prop_model,
        unfreeze_atom_model=args.unfreeze_atom_model,
        build_dataset_only=args.build_dataset_only,
        skip_compile=args.skip_compile,
        dataloader_num_workers=args.dataloader_num_workers,
        smoke_data_path=args.smoke_data_path,
        smoke_atom_data_path=args.smoke_atom_data_path,
        r_cut=args.r_cut,
        r_cut_im=args.r_cut_im,
        neural_cutoff=physics.neural_cutoff,
        batch_size=args.batch_size,
        device=device,
    )


def expected_resume_semantics(args: Any) -> dict[str, Any]:
    plan = _make_plan(args, emit_warning=False)
    return {
        "architecture": plan.internal_architecture,
        "mace": {
            "sha256": plan.mace_sha256,
            "feature_mode": plan.feature_mode,
        },
        "pair_mode": plan.pair_mode,
        "dtype_policy": plan.mace_default_dtype,
        "physics": {"physics_hash": plan.physics_hash},
        "data": {
            "dataset_hash": plan.data_hash,
            "preprocessing_hash": plan.preprocessing_hash,
            "split_hash": plan.split_hash,
        },
        "route_submodel_digests": dict(plan.route_submodel_digests),
    }


def _canonical_feature_schema(plan: MACETrainingPlan) -> str | None:
    canonical_digest = (
        "e4495612037b3b3312633182882a38a694ecac9ea0be2b9889ac0b2a84a99510"
    )
    if plan.mace_model != "polar-1-s" or plan.mace_sha256 != canonical_digest:
        return None
    if plan.feature_mode == "final-layer-scalars":
        return (
            "polar-1-s:mace=0.3.16:mode=final-layer-scalars:adapter=public:"
            "inv=512:equiv=0:layers=1:public"
        )
    return (
        "polar-1-s:mace=0.3.16:mode=all-scalars+norms:"
        "adapter=polar-private-mace-0.3.16-v1:inv=2560:equiv=8192:layers=1:"
        "private=polar-private-mace-0.3.16-v1:"
        "irreps=512x0e+512x1o+512x2e+512x3o"
    )


def _validate_resume_feature_schema(
    config: Mapping[str, Any], plan: MACETrainingPlan
) -> None:
    mace = config.get("mace")
    if not isinstance(mace, Mapping):
        raise ValueError("resume semantic mismatch: missing config.mace")
    schema = mace.get("feature_schema")
    if not isinstance(schema, str) or f":mode={plan.feature_mode}:" not in schema:
        raise ValueError("resume semantic mismatch: config.mace.feature_schema")
    canonical = _canonical_feature_schema(plan)
    if canonical is not None and schema != canonical:
        raise ValueError("resume semantic mismatch: config.mace.feature_schema")


def _default_checkpoint_loader(path: str) -> Mapping[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    from apnet_pt.model_io import validate_mace_checkpoint_v3

    validate_mace_checkpoint_v3(checkpoint)
    return checkpoint


def _assert_resume_subset(
    actual: Mapping[str, Any], expected: Mapping[str, Any], path: str = "config"
) -> None:
    for key, expected_value in expected.items():
        location = f"{path}.{key}"
        if key not in actual:
            raise ValueError(f"resume semantic mismatch: missing {location}")
        actual_value = actual[key]
        if isinstance(expected_value, Mapping):
            if not isinstance(actual_value, Mapping):
                raise ValueError(f"resume semantic mismatch: {location}")
            if key == "route_submodel_digests":
                if dict(actual_value) != dict(expected_value):
                    raise ValueError(f"resume semantic mismatch: {location}")
            else:
                _assert_resume_subset(actual_value, expected_value, location)
        elif actual_value != expected_value:
            raise ValueError(
                f"resume semantic mismatch: {location} expected "
                f"{expected_value!r}, got {actual_value!r}"
            )


def validate_mace_cli_args(
    args: Any,
    *,
    checkpoint_loader: Callable[[str], Mapping[str, Any]] | None = None,
) -> MACETrainingPlan:
    """Validate every MACE semantic before model or dataset construction."""

    plan = _make_plan(args, emit_warning=True)
    output = Path(plan.output_path)
    if output.exists() and not plan.overwrite:
        raise FileExistsError(
            f"output exists: {output}; pass --overwrite or explicit --resume"
        )
    return plan


def validate_atomic_property_checkpoint(
    checkpoint: Mapping[str, Any], expected: Mapping[str, Any]
) -> None:
    """Validate the complete versioned atomic-property checkpoint identity."""

    if checkpoint.get("checkpoint_version") != 3:
        raise ValueError("atomic checkpoint semantic mismatch: checkpoint_version")
    if checkpoint.get("model_type") != "MACEAtomicProperties":
        raise ValueError("atomic checkpoint semantic mismatch: model_type")
    config = checkpoint.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("atomic checkpoint semantic mismatch: config")

    required = {
        "property_mode",
        "provider_kind",
        "mace",
        "dtype_policy",
        "atomic_property_schema",
        "quadrupole_convention",
        "physics_hash",
        "data",
    }
    if not required.issubset(config):
        missing = sorted(required.difference(config))
        raise ValueError(f"atomic checkpoint semantic mismatch: missing {missing}")

    def compare(actual: Mapping[str, Any], wanted: Mapping[str, Any], path: str) -> None:
        for key, wanted_value in wanted.items():
            location = f"{path}.{key}"
            if key not in actual:
                raise ValueError(f"atomic checkpoint semantic mismatch: {location}")
            actual_value = actual[key]
            if isinstance(wanted_value, Mapping):
                if not isinstance(actual_value, Mapping):
                    raise ValueError(f"atomic checkpoint semantic mismatch: {location}")
                compare(actual_value, wanted_value, location)
            elif actual_value != wanted_value:
                raise ValueError(f"atomic checkpoint semantic mismatch: {location}")

    compare(config, expected, "config")


def _load_atomic_provider_state(
    provider: torch.nn.Module,
    path: str,
    *,
    expected_config: Mapping[str, Any] | None = None,
) -> None:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise RuntimeError(
            f"Unable to load MACE atomic-property checkpoint {path}: {exc}"
        ) from exc
    if expected_config is not None:
        validate_atomic_property_checkpoint(checkpoint, expected_config)
    state = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state, Mapping):
        raise ValueError("MACE atomic-property checkpoint has no state dictionary")
    if state and all(key.startswith("property_provider.") for key in state):
        state = {
            key.removeprefix("property_provider."): value
            for key, value in state.items()
        }
    incompatible = provider.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "MACE atomic-property checkpoint is incompatible: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )


def _default_factory_dependencies(plan: MACETrainingPlan) -> MACEFactoryDependencies:
    """Create lazy production builders after all CLI validation succeeds."""

    shared: dict[str, Any] = {}

    def physics_config():
        if "physics" not in shared:
            from apnet_pt.mace.schema import PhysicsConfig

            shared["physics"] = PhysicsConfig(
                electrostatics_mode=plan.long_range_elst,
                d3_parameters=plan.d3_parameters,
                neural_cutoff=plan.neural_cutoff,
            )
        return shared["physics"]

    def featurizer_builder(_plan):
        if "featurizer" in shared:
            return shared["featurizer"]
        if not plan.mace_model_path:
            raise FileNotFoundError(
                "MACE model construction requires a verified local "
                "--mace_model_path; automatic download is disabled"
            )
        try:
            from apnet_pt.mace.encoder import (
                MACEPolarFeaturizer,
                load_verified_polar_mace,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Install the pinned optional mace-torch/e3nn stack to use MACE routes"
            ) from exc
        dtype = torch.float32 if plan.mace_default_dtype == "float32" else torch.float64
        # The validated policy is explicit; CUDA remains an external parity gate.
        device = plan.device
        try:
            backbone = load_verified_polar_mace(
                plan.mace_model_path,
                expected_sha256=plan.mace_sha256,
                device=device,
                offline=plan.mace_offline,
            )
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "The verified MACE artifact is present, but the pinned "
                "optional mace-torch stack is not installed"
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"Unable to reconstruct verified MACE artifact: {exc}"
            ) from exc
        prepared_cache = None
        if plan.mace_cache_dir:
            from apnet_pt.training.smoke import load_prepared_feature_cache

            prepared_cache = load_prepared_feature_cache(
                plan.mace_cache_dir,
                feature_mode=plan.feature_mode,
                mace_sha256=plan.mace_sha256,
                mace_model_id=plan.mace_model,
                physics_hash=physics_config().physics_hash,
                dataset_kind=plan.kind,
                dataset_hash=plan.data_hash,
                preprocessing_hash=plan.preprocessing_hash,
                split_hash=plan.split_hash,
                dtype=dtype,
            )
        featurizer = MACEPolarFeaturizer(
            backbone,
            checkpoint_sha256=plan.mace_sha256,
            model_id=plan.mace_model,
            feature_mode=plan.feature_mode,
            dtype=dtype,
            physics_config=physics_config(),
            cache=prepared_cache,
        )
        featurizer.cache_dir = plan.mace_cache_dir
        shared["featurizer"] = featurizer
        return featurizer

    def property_provider_builder(_plan, _featurizer):
        if "property_provider" in shared:
            return shared["property_provider"]
        from apnet_pt.mace.properties import (
            LegacyAtomMPNNPropertyProvider,
            MACEAtomPropertyModel,
            MACEPropertyCompletionHeads,
            PolarDirectPropertyProvider,
        )

        if plan.properties == "legacy":
            from apnet_pt.AtomPairwiseModels.mtp_mtp import (
                AM_DimerParam_Model,
                AtomTypeParamModel,
            )

            atom_type_model = AtomTypeParamModel(
                ds_root=None,
                use_GPU=False,
                ignore_database_null=True,
                atom_model_pre_trained_path=_plan_legacy_path(plan, "atom_model"),
                pre_trained_model_path=_plan_legacy_path(
                    plan, "atom_type_param_model"
                ),
                freeze_atom_model=not plan.unfreeze_atom_model,
            )
            dimer_model = AM_DimerParam_Model(
                ds_root=None,
                use_GPU=False,
                ignore_database_null=True,
                atom_model=atom_type_model.model,
                atom_model_type="AtomTypeParamNN",
                pre_trained_model_path=_plan_legacy_path(
                    plan, "dimer_param_model"
                ),
                freeze_atom_model=not plan.unfreeze_atom_model,
            )
            provider = LegacyAtomMPNNPropertyProvider(
                dimer_model.dimer_model,
                freeze_atom_model=not plan.unfreeze_atom_model,
                freeze_dimer_parameters=not plan.unfreeze_dimer_prop_model,
            )
        else:
            heads = MACEPropertyCompletionHeads(
                invariant_dim=2560,
                equivariant_irreps=(
                    "512x0e+512x1o+512x2e+512x3o"
                ),
            )
            if plan.properties == "direct":
                provider = PolarDirectPropertyProvider(heads)
            else:
                provider = MACEAtomPropertyModel(heads)
            if not plan.train_atomic_heads:
                checkpoint = torch.load(
                    plan.mace_atom_model_path,
                    map_location="cpu",
                    weights_only=True,
                )
                config = checkpoint.get("config")
                if not isinstance(config, Mapping):
                    raise ValueError("atomic checkpoint semantic mismatch: config")
                mace_config = config.get("mace")
                data_config = config.get("data")
                if not isinstance(mace_config, Mapping) or not isinstance(
                    data_config, Mapping
                ):
                    raise ValueError("atomic checkpoint semantic mismatch: identity")
                expected_mode = (
                    "direct-completion" if plan.properties == "direct" else "learned"
                )
                expected_kind = "direct" if plan.properties == "direct" else "atomhead"
                expected_config = {
                    "property_mode": expected_mode,
                    "provider_kind": expected_kind,
                    "mace": {
                        "sha256": plan.mace_sha256,
                        "version": getattr(_featurizer, "mace_version", "0.3.16"),
                        "model_class": (
                            type(_featurizer.backbone).__module__
                            + "."
                            + type(_featurizer.backbone).__name__
                        ),
                        "feature_schema": mace_config.get("feature_schema"),
                        "feature_mode": plan.feature_mode,
                    },
                    "dtype_policy": plan.mace_default_dtype,
                    "atomic_property_schema": "ap3-atomic-properties-cartesian-v1",
                    "quadrupole_convention": (
                        "cartesian-symmetric-traceless-3x3"
                    ),
                    "physics_hash": plan.physics_hash,
                    "data": dict(data_config),
                }
                schema = mace_config.get("feature_schema")
                if not isinstance(schema, str) or f":mode={plan.feature_mode}:" not in schema:
                    raise ValueError(
                        "atomic checkpoint semantic mismatch: config.mace.feature_schema"
                    )
                _load_atomic_provider_state(
                    provider,
                    plan.mace_atom_model_path,
                    expected_config=expected_config,
                )
                provider.requires_grad_(False)
                provider.eval()
        shared["property_provider"] = provider
        return provider

    def pair_core_builder(_plan):
        from apnet_pt.AtomPairwiseModels.apnet3_d3_fused import (
            APNet3D3_AtomType_MPNN,
        )
        from apnet_pt.mace.pair import MACEPairResidualCore

        ap3 = APNet3D3_AtomType_MPNN(
            dimer_prop_model=None,
            use_precomputed_classical=True,
            no_disp_nn=plan.no_disp_nn,
            r_cut=plan.r_cut,
            r_cut_im=plan.r_cut_im,
        )
        feature_dim = 512 if plan.feature_mode == "final-layer-scalars" else 2560
        kwargs = {}
        if plan.internal_architecture in {"direct-polar", "atomhead"}:
            kwargs["architecture_id"] = plan.internal_architecture
        return MACEPairResidualCore(
            ap3,
            mace_feature_dim=feature_dim,
            pair_mode=plan.pair_mode,
            feature_mode=plan.feature_mode,
            **kwargs,
        )

    def long_range_builder(_plan):
        from apnet_pt.mace.long_range import LongRangeSAPTProvider

        return LongRangeSAPTProvider(physics_config())

    def dataset_builder(_plan):
        from apnet_pt.training.smoke import (
            load_atomic_smoke_fixture,
            load_pair_smoke_fixture,
        )

        if plan.kind == "pair":
            if not plan.smoke_data_path:
                return None
            dataset = load_pair_smoke_fixture(
                plan.smoke_data_path,
                batch_size=plan.batch_size,
                r_cut=plan.r_cut,
                r_cut_im=plan.r_cut_im,
            )
            if dataset.physics_hash != plan.physics_hash:
                raise ValueError("smoke fixture physics hash does not match training plan")
            return dataset
        if not plan.smoke_atom_data_path:
            return None
        return load_atomic_smoke_fixture(plan.smoke_atom_data_path)

    def lifecycle_runner(_plan, harness, dataset):
        if dataset is None or plan.build_dataset_only:
            return None
        from apnet_pt.training.smoke import (
            run_atomic_smoke_lifecycle,
            run_pair_smoke_lifecycle,
        )

        if plan.kind == "pair":
            report = run_pair_smoke_lifecycle(
                harness.model,
                dataset,
                output_path=plan.output_path,
                learning_rate=plan.learning_rate,
                include_total_mse=plan.include_total_mse,
                plan=plan,
            )
            print(f"smoke loss={report.loss:.8g}")
            print(f"smoke component_losses={dict(report.component_losses)}")
            print(f"smoke classical_ledger={dict(report.classical_ledger)}")
            print(f"smoke residual_ledger={dict(report.residual_ledger)}")
            print(
                "smoke induction_diagnostics="
                f"{{'converged': {report.induction_converged}, "
                f"'iterations': {report.induction_iterations}, "
                f"'residual': {report.induction_residual}, "
                f"'policy': '{report.induction_policy}'}}"
            )
            return report
        report = run_atomic_smoke_lifecycle(
            harness,
            dataset,
            output_path=plan.output_path,
            learning_rate=plan.learning_rate,
            physics_hash=plan.physics_hash,
        )
        print(f"atomic smoke loss={report.loss:.8g}")
        print(f"atomic smoke property_losses={dict(report.losses)}")
        return report

    return MACEFactoryDependencies(
        featurizer_builder=featurizer_builder,
        property_provider_builder=property_provider_builder,
        pair_core_builder=pair_core_builder,
        long_range_builder=long_range_builder,
        dataset_builder=dataset_builder,
        lifecycle_runner=lifecycle_runner,
    )


def _plan_legacy_path(plan: MACETrainingPlan, name: str) -> str:
    paths = getattr(plan, "legacy_model_paths", None)
    if not isinstance(paths, Mapping) or name not in paths:
        raise RuntimeError(f"validated legacy path {name} is unavailable")
    return str(paths[name])


def _require_builder(value: Callable[..., Any] | None, name: str) -> Callable[..., Any]:
    if value is None:
        raise RuntimeError(
            f"MACE {name} is unavailable. Install the pinned optional MACE stack, "
            "provide verified local artifacts, or inject factory dependencies."
        )
    return value


def build_mace_ap3d3_harness(
    plan: MACETrainingPlan,
    *,
    dependencies: MACEFactoryDependencies | None = None,
):
    """Build the shared pair harness through normalized injectable seams."""

    if plan.kind != "pair":
        raise ValueError("pair harness requires a pair training plan")
    dependencies = dependencies or MACEFactoryDependencies()
    featurizer = _require_builder(
        dependencies.featurizer_builder, "featurizer builder"
    )(plan)
    backbone = getattr(featurizer, "backbone", None)
    if not isinstance(backbone, torch.nn.Module):
        raise TypeError("MACE featurizer builder must expose an nn.Module backbone")
    backbone.requires_grad_(False)
    backbone.eval()
    property_provider = _require_builder(
        dependencies.property_provider_builder, "property provider builder"
    )(plan, featurizer)
    pair_core = _require_builder(
        dependencies.pair_core_builder, "pair core builder"
    )(plan)
    ap3_core = getattr(pair_core, "ap3_core", None)
    actual_no_disp = bool(
        getattr(pair_core, "no_disp_nn", getattr(ap3_core, "no_disp_nn", False))
    )
    if actual_no_disp != plan.no_disp_nn:
        raise ValueError("pair core no_disp_nn does not match the training plan")
    long_range = _require_builder(
        dependencies.long_range_builder, "long-range builder"
    )(plan)
    if dependencies.model_builder is not None:
        model = dependencies.model_builder(
            plan, featurizer, property_provider, pair_core, long_range
        )
    else:
        if plan.is_ablation:
            raise RuntimeError(
                "named feature ablations require an explicit model_builder"
            )
        from apnet_pt.mace.model import MACEAP3D3

        model = MACEAP3D3(
            architecture=plan.internal_architecture,
            featurizer=featurizer,
            property_provider=property_provider,
            pair_core=pair_core,
            long_range_provider=long_range,
            use_precomputed_classical=bool(plan.use_precomputed_classical),
        )
    if bool(getattr(model, "use_precomputed_classical", False)) != bool(
        plan.use_precomputed_classical
    ):
        raise ValueError(
            "model precomputed-classical mode does not match the training plan"
        )
    from apnet_pt.mace.model import MACEAP3D3Model

    return MACEAP3D3Model(
        model,
        include_total_mse=plan.include_total_mse,
    )


def build_mace_atomic_harness(
    plan: MACETrainingPlan,
    *,
    dependencies: MACEFactoryDependencies | None = None,
):
    """Build the frozen-feature atomic-property optimization harness."""

    if plan.kind != "atomic":
        raise ValueError("atomic harness requires an atomic training plan")
    dependencies = dependencies or MACEFactoryDependencies()
    featurizer = _require_builder(
        dependencies.featurizer_builder, "featurizer builder"
    )(plan)
    backbone = getattr(featurizer, "backbone", None)
    if not isinstance(backbone, torch.nn.Module):
        raise TypeError("MACE featurizer builder must expose an nn.Module backbone")
    backbone.requires_grad_(False)
    backbone.eval()
    provider = _require_builder(
        dependencies.property_provider_builder, "property provider builder"
    )(plan, featurizer)
    from apnet_pt.mace.model import MACEAtomicPropertiesModel

    return MACEAtomicPropertiesModel(
        property_mode=plan.property_mode,
        featurizer=featurizer,
        property_provider=provider,
    )


def dispatch_mace_cli(
    args: Any,
    *,
    dependencies: MACEFactoryDependencies | None = None,
    checkpoint_loader: Callable[[str], Mapping[str, Any]] | None = None,
) -> MACEFactoryResult:
    """Validate first, then build injected model and dataset lifecycle objects."""

    plan = validate_mace_cli_args(args, checkpoint_loader=checkpoint_loader)
    dependencies = dependencies or _default_factory_dependencies(plan)
    if plan.kind == "pair":
        harness = build_mace_ap3d3_harness(plan, dependencies=dependencies)
    else:
        harness = build_mace_atomic_harness(plan, dependencies=dependencies)
    dataset = (
        dependencies.dataset_builder(plan)
        if dependencies.dataset_builder is not None
        else None
    )
    if dataset is None:
        raise RuntimeError(
            "MACE verification dataset is absent; production no-op dispatch is forbidden"
        )
    if dependencies.lifecycle_runner is None:
        raise RuntimeError("MACE lifecycle is absent; no-op dispatch is forbidden")
    lifecycle = dependencies.lifecycle_runner(plan, harness, dataset)
    if lifecycle is None:
        raise RuntimeError("MACE lifecycle is absent; no-op dispatch is forbidden")
    return MACEFactoryResult(
        plan=plan,
        harness=harness,
        dataset=dataset,
        lifecycle=lifecycle,
    )
