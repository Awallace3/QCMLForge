"""Production training primitives for the locked MACE/AP3D3 SLURM benchmark.

This module intentionally contains no import-time dependency on MACE.  Heavy
MACE/e3nn imports occur only inside MACE tasks.  The controller lives in
``scripts/slurm/benchmark.py`` and passes an immutable lock plus one array task.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from apnet_pt.mace.schema import COMPONENT_ORDER, AtomicPropertyBundle

PAIR_ROUTE_OPTIONS = {
    "H1": "MACE-AP3D3-H1",
    "H2": "MACE-AP3D3-H2",
    "DirectPolar": "MACE-AP3D3-DirectPolar",
    "AtomHead": "MACE-AP3D3-AtomHead",
}
ATOMIC_METRICS = ("q", "mu", "quadrupole")
CACHE_FORMAT = "qcmlforge-mace-monomer-cache-v2"
FEATURE_MODES = ("final-layer-scalars", "all-scalars+norms")


class ProductionBenchmarkError(RuntimeError):
    """A production task cannot satisfy its frozen benchmark contract."""


class BenchmarkBlocked(ProductionBenchmarkError):
    """An external production prerequisite is absent or unattested."""


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _hash_record(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(json.dumps(value, indent=2, sort_keys=True).encode() + b"\n")
    os.replace(temporary, path)


def _atomic_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _environment() -> dict[str, Any]:
    import sys

    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_device": (
            torch.cuda.get_device_name(torch.cuda.current_device())
            if torch.cuda.is_available()
            else None
        ),
    }


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device() -> torch.device:
    if not torch.cuda.is_available():
        raise BenchmarkBlocked(
            "production MACE/AP3D3 benchmark tasks require CUDA; CUDA parity must "
            "be validated on the target cluster before scientific use"
        )
    return torch.device("cuda")


def _metric(error: torch.Tensor) -> dict[str, float]:
    values = error.detach().double().reshape(-1)
    return {
        "mae": float(values.abs().mean()),
        "rmse": float(values.square().mean().sqrt()),
    }


def energy_metrics(
    prediction: torch.Tensor, target: torch.Tensor
) -> dict[str, dict[str, float]]:
    """Return component and total MAE/RMSE in canonical component order."""

    if (
        prediction.shape != target.shape
        or prediction.ndim != 2
        or prediction.shape[1] != 4
    ):
        raise ValueError("energy prediction and target must both have shape [N, 4]")
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError("energy metrics require finite tensors")
    error = prediction - target
    result = {
        name: _metric(error[:, index]) for index, name in enumerate(COMPONENT_ORDER)
    }
    result["total"] = _metric(error.sum(dim=1))
    return result


def atomic_metrics(
    prediction: AtomicPropertyBundle,
    target: AtomicPropertyBundle,
) -> dict[str, dict[str, float]]:
    """Evaluate only independently supervised PBE0/MBIS q, mu, and Q."""

    return {
        name: _metric(getattr(prediction, name) - getattr(target, name))
        for name in ATOMIC_METRICS
    }


def _verify_locked_inputs(lock: Mapping[str, Any]) -> None:
    project_root = Path(__file__).resolve().parents[3]
    try:
        source_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ProductionBenchmarkError("unable to verify worker source commit") from exc
    if source_commit != lock.get("source_commit"):
        raise ProductionBenchmarkError(
            "worker source commit does not match benchmark lock"
        )
    config = lock["config"]
    records = []
    dataset = config["dataset"]
    records.extend(dataset["pair_files"].values())
    records.extend(dataset["atomic_files"].values())
    records.extend((dataset["split_manifest"], dataset["provenance_manifest"]))
    records.extend(config["artifacts"].values())
    for record in records:
        path = Path(record["path"])
        if not path.is_file():
            raise ProductionBenchmarkError(f"locked input is missing: {path}")
        actual = _sha256_file(path)
        if actual != record["sha256"]:
            raise ProductionBenchmarkError(
                f"locked input SHA-256 mismatch for {path}: {actual}"
            )


def _safe_link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if destination.resolve() != source.resolve():
            raise ProductionBenchmarkError(
                f"staged data path points at a different source: {destination}"
            )
        return
    destination.symlink_to(source.resolve())


def _staged_roots(root: Path) -> tuple[Path, Path]:
    return root / "data" / "pair", root / "data" / "atomic"


def _stage_raw_data(lock: Mapping[str, Any], root: Path) -> tuple[Path, Path]:
    pair_root, atomic_root = _staged_roots(root)
    dataset = lock["config"]["dataset"]
    for record in dataset["pair_files"].values():
        source = Path(record["path"])
        _safe_link(source, pair_root / "raw" / source.name)
    for record in dataset["atomic_files"].values():
        source = Path(record["path"])
        _safe_link(source, atomic_root / "raw" / source.name)
    return pair_root, atomic_root


def _legacy_models(config: Mapping[str, Any]):
    """Construct the exact frozen legacy property hierarchy used by BASE/H1/H2."""

    from apnet_pt.AtomPairwiseModels.mtp_mtp import (
        AM_DimerParam_Model,
        AtomTypeParamModel,
    )

    artifacts = config["artifacts"]
    atom_type = AtomTypeParamModel(
        ds_root=None,
        use_GPU=False,
        ignore_database_null=True,
        atom_model_pre_trained_path=artifacts["legacy_atom_model"]["path"],
        pre_trained_model_path=artifacts["legacy_parameter_model"]["path"],
        freeze_atom_model=True,
    )
    dimer = AM_DimerParam_Model(
        ds_root=None,
        use_GPU=False,
        ignore_database_null=True,
        atom_model=atom_type.model,
        atom_model_type="AtomTypeParamNN",
        pre_trained_model_path=artifacts["legacy_parameter_model_2"]["path"],
        elst_damping_type="CLIFF",
        freeze_atom_model=True,
    )
    return atom_type, dimer


def _build_pair_wrapper(lock: Mapping[str, Any], root: Path):
    from apnet_pt.AtomPairwiseModels.apnet3_d3_fused import (
        APNet3D3_AtomType_Model,
    )

    config = lock["config"]
    pair_root, _ = _stage_raw_data(lock, root)
    atom_type, dimer = _legacy_models(config)
    return APNet3D3_AtomType_Model(
        atom_type_model=atom_type.model,
        dimer_prop_model=dimer.dimer_model,
        am_dimer_param_model=dimer,
        n_rbf=8,
        n_neuron=128,
        n_embed=8,
        r_cut=5.0,
        r_cut_im=8.0,
        ds_spec_type=2,
        ds_root=str(pair_root),
        ignore_database_null=False,
        ds_atomic_batch_size=16,
        ds_num_devices=1,
        ds_skip_process=False,
        ds_skip_compile=True,
        ds_datapoint_storage_n_objects=16,
        ds_random_seed=0,
        ds_class_type="lmdb",
        use_precomputed_classical=True,
        ds_type="total_component_energies",
        ds_batch_size=config["training"]["batch_size"],
        freeze_dimer_prop_model=True,
        d3_damping_parameters=(config["physics"]["d3_parameters"] or None),
        use_GPU=True,
    )


def _build_atomic_datasets(lock: Mapping[str, Any], root: Path):
    from apnet_pt.atomic_datasets import atomic_module_dataset

    _, atomic_root = _stage_raw_data(lock, root)
    return {
        "spec_1": atomic_module_dataset(
            root=str(atomic_root), spec_type=4, in_memory=False, r_cut=5.0
        ),
        "spec_5": atomic_module_dataset(
            root=str(atomic_root), spec_type=9, in_memory=False, r_cut=5.0
        ),
    }


def _check_indices(dataset: Any, indices: Sequence[int], description: str) -> None:
    if not indices:
        raise ProductionBenchmarkError(f"{description} split is empty")
    maximum = max(indices)
    if maximum >= len(dataset):
        raise ProductionBenchmarkError(
            f"{description} split index {maximum} exceeds dataset length {len(dataset)}"
        )


def _pair_subsets(wrapper: Any, split: Mapping[str, Any]):
    from torch.utils.data import Subset

    if not isinstance(wrapper.dataset, list) or len(wrapper.dataset) != 2:
        raise ProductionBenchmarkError(
            "AP3 spec 2 must expose fixed train/test datasets"
        )
    sources = {"train": wrapper.dataset[0], "test": wrapper.dataset[1]}
    subsets = {}
    for name in ("train", "validation", "test"):
        record = split["pair"][name]
        dataset = sources[record["source"]]
        _check_indices(dataset, record["indices"], f"pair {name}")
        subsets[name] = Subset(dataset, record["indices"])
    return subsets


def _release_wrapper_models(wrapper: Any) -> None:
    wrapper.model.to("cpu")
    for name in ("dimer_prop_model", "atom_type_model", "am_dimer_param_model"):
        value = getattr(wrapper, name, None)
        if isinstance(value, torch.nn.Module):
            value.to("cpu")
    if isinstance(wrapper.dataset, list):
        for source in wrapper.dataset:
            for name in ("atom_model", "dimer_prop_model"):
                if hasattr(source, name):
                    setattr(source, name, None)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _atomic_subsets(datasets: Mapping[str, Any], split: Mapping[str, Any]):
    from torch.utils.data import ConcatDataset, Subset

    result = {}
    for name in ("train", "validation", "test"):
        parts = []
        for source in ("spec_1", "spec_5"):
            indices = split["atomic"][name][source]["indices"]
            if indices:
                _check_indices(datasets[source], indices, f"atomic {name}/{source}")
                parts.append(Subset(datasets[source], indices))
        if not parts:
            raise ProductionBenchmarkError(f"atomic {name} split is empty")
        result[name] = ConcatDataset(parts)
    return result


def _require_spin(dataset: Iterable[Any], description: str) -> None:
    for item in dataset:
        if not hasattr(item, "total_spin"):
            raise ProductionBenchmarkError(
                f"{description} lacks molecular multiplicity; rebuild processed data "
                "with the current qcel_mon_to_pyg_data contract"
            )


def prepare_datasets(lock: Mapping[str, Any], root: Path) -> Mapping[str, Any]:
    wrapper = _build_pair_wrapper(lock, root)
    pair = _pair_subsets(wrapper, lock["split_manifest"])
    atomic_data = _build_atomic_datasets(lock, root)
    atomic = _atomic_subsets(atomic_data, lock["split_manifest"])
    for name, dataset in atomic.items():
        _require_spin(dataset, f"atomic {name}")
    record = {
        "status": "complete",
        "benchmark_hash": lock["benchmark_hash"],
        "pair_counts": {name: len(data) for name, data in pair.items()},
        "atomic_counts": {name: len(data) for name, data in atomic.items()},
        "split_hash": lock["config"]["dataset"]["split_manifest"]["sha256"],
    }
    _atomic_json(root / "data" / "COMPLETE.json", record)
    return record


def _dataset_identities(lock: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    dataset = lock["config"]["dataset"]
    return {
        "pair": {
            "dataset_hash": _hash_record(dataset["pair_files"]),
            "preprocessing_hash": _hash_record(
                {"r_cut": 5.0, "r_cut_im": 8.0, "pair_spec_type": 2}
            ),
            "split_hash": dataset["split_manifest"]["sha256"],
        },
        "atomic": {
            "dataset_hash": _hash_record(dataset["atomic_files"]),
            "preprocessing_hash": _hash_record(
                {"r_cut": 5.0, "targets": list(ATOMIC_METRICS)}
            ),
            "split_hash": dataset["split_manifest"]["sha256"],
        },
    }


def _monomer_hash(numbers, positions, charge, spin) -> str:
    payload = {
        "atomic_numbers": numbers.detach().cpu().long().tolist(),
        "positions_angstrom": positions.detach().cpu().double().tolist(),
        "total_charge": float(charge.detach().cpu().reshape(-1)[0]),
        "total_spin": float(spin.detach().cpu().reshape(-1)[0]),
    }
    return _hash_record(payload)


class _StreamingCache(dict):
    """Write-through cache used only by the dedicated preparation task."""

    def __init__(self, root: Path, mode: str, identity: Mapping[str, Any]):
        super().__init__()
        self.root = root
        self.mode = mode
        self.identity = identity
        self.scope = "pair"
        self.entries: dict[str, dict[str, Any]] = {}
        self.membership = {"pair": set(), "atomic": set()}
        self.feature_schema: str | None = None
        for path in sorted((self.root / self.mode).glob("*.pt")):
            record = torch.load(path, map_location="cpu", weights_only=True)
            record_identity = record.get("identity", {})
            key = record_identity.get("cache_key")
            if (
                record_identity.get("format") != CACHE_FORMAT
                or record_identity.get("feature_mode") != self.mode
                or record_identity.get("mace_sha256") != self.identity["mace_sha256"]
                or record_identity.get("mace_model_id") != "polar-1-s"
                or record_identity.get("physics_hash") != self.identity["physics_hash"]
                or record_identity.get("dtype") != "float32"
                or not isinstance(key, str)
            ):
                raise ProductionBenchmarkError(f"stale partial cache entry: {path}")
            feature_schema = record.get("feature_schema")
            if not isinstance(feature_schema, str) or (
                self.feature_schema is not None
                and self.feature_schema != feature_schema
            ):
                raise ProductionBenchmarkError(
                    f"partial cache feature schema mismatch: {path}"
                )
            self.feature_schema = feature_schema
            self.entries[key] = {
                "path": str(path.relative_to(self.root)),
                "sha256": _sha256_file(path),
                "monomer_hash": record_identity["monomer_hash"],
                "cache_key": key,
                "feature_mode": self.mode,
            }

    def __contains__(self, key: object) -> bool:
        if key not in self.entries:
            return False
        self.membership[self.scope].add(self.entries[str(key)]["monomer_hash"])
        return True

    def __getitem__(self, key: str):
        from apnet_pt.mace.schema import MACEAtomicFeatures, PolarMACEDirectOutputs

        entry = self.entries[key]
        record = torch.load(
            self.root / entry["path"], map_location="cpu", weights_only=True
        )
        tensors = record["tensors"]
        features = MACEAtomicFeatures(
            invariant=tensors["invariant"],
            equivariant=tensors["equivariant"],
            batch=tensors["batch"],
            atomic_numbers=tensors["atomic_numbers"],
            total_charge=tensors["total_charge"],
            total_spin=tensors["total_spin"],
            feature_schema=record["feature_schema"],
        )
        direct = PolarMACEDirectOutputs(
            density_coefficients=tensors["density_coefficients"],
            charges=tensors["charges"],
            molecular_dipole_eangstrom=tensors["molecular_dipole_eangstrom"],
            positions_angstrom=tensors["positions_angstrom"],
            batch=tensors["batch"],
            total_charge=tensors["total_charge"],
        )
        return features, direct

    def __setitem__(self, key: str, bundles: Any) -> None:
        features, direct = bundles
        monomer_hash = _monomer_hash(
            features.atomic_numbers,
            direct.positions_angstrom,
            features.total_charge,
            features.total_spin,
        )
        self.membership[self.scope].add(monomer_hash)
        destination = self.root / self.mode / f"{key}.pt"
        identity = {
            "format": CACHE_FORMAT,
            "monomer_hash": monomer_hash,
            "cache_key": key,
            "feature_mode": self.mode,
            "mace_sha256": self.identity["mace_sha256"],
            "mace_model_id": "polar-1-s",
            "physics_hash": self.identity["physics_hash"],
            "dtype": "float32",
        }
        tensors = {
            "invariant": features.invariant.detach().cpu(),
            "equivariant": features.equivariant.detach().cpu(),
            "batch": features.batch.detach().cpu(),
            "atomic_numbers": features.atomic_numbers.detach().cpu(),
            "total_charge": features.total_charge.detach().cpu(),
            "total_spin": features.total_spin.detach().cpu(),
            "density_coefficients": direct.density_coefficients.detach().cpu(),
            "charges": direct.charges.detach().cpu(),
            "molecular_dipole_eangstrom": direct.molecular_dipole_eangstrom.detach().cpu(),
            "positions_angstrom": direct.positions_angstrom.detach().cpu(),
        }
        record = {
            "identity": identity,
            "feature_schema": features.feature_schema,
            "symbols": [],
            "tensors": tensors,
        }
        if destination.exists():
            loaded = torch.load(destination, map_location="cpu", weights_only=True)
            if loaded.get("identity") != identity:
                raise ProductionBenchmarkError(
                    f"existing cache identity mismatch: {destination}"
                )
        else:
            _atomic_torch(destination, record)
        self.feature_schema = features.feature_schema
        self.entries[key] = {
            "path": str(destination.relative_to(self.root)),
            "sha256": _sha256_file(destination),
            "monomer_hash": monomer_hash,
            "cache_key": key,
            "feature_mode": self.mode,
        }


def _pair_loader(
    dataset: Any,
    batch_size: int,
    workers: int,
    shuffle: bool = False,
    seed: int | None = None,
):
    from apnet_pt.pt_datasets.ap3_fused_ds import (
        APNet2_fused_DataLoader,
        ap3_fused_collate_update,
    )

    return APNet2_fused_DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        collate_fn=ap3_fused_collate_update,
        generator=(torch.Generator().manual_seed(seed) if seed is not None else None),
    )


def _atomic_loader(
    dataset: Any,
    batch_size: int,
    workers: int,
    shuffle: bool = False,
    seed: int | None = None,
):
    from apnet_pt.atomic_datasets import AtomicDataLoader, atomic_collate_update

    return AtomicDataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        collate_fn=atomic_collate_update,
        generator=(torch.Generator().manual_seed(seed) if seed is not None else None),
    )


def prepare_features(lock: Mapping[str, Any], root: Path) -> Mapping[str, Any]:
    complete = root / "data" / "COMPLETE.json"
    if not complete.is_file():
        raise ProductionBenchmarkError("dataset preparation must complete first")
    device = _device()
    config = lock["config"]
    wrapper = _build_pair_wrapper(lock, root)
    pair = _pair_subsets(wrapper, lock["split_manifest"])
    _release_wrapper_models(wrapper)
    atomic = _atomic_subsets(_build_atomic_datasets(lock, root), lock["split_manifest"])
    cache_root = root / "feature-cache"

    from apnet_pt.mace.encoder import MACEPolarFeaturizer, load_verified_polar_mace
    from apnet_pt.mace.schema import PhysicsConfig

    physics_values = {
        key: value for key, value in config["physics"].items() if key != "physics_hash"
    }
    physics = PhysicsConfig(**physics_values)
    if physics.physics_hash != config["physics"]["physics_hash"]:
        raise ProductionBenchmarkError("locked PhysicsConfig hash mismatch")
    identities = _dataset_identities(lock)
    if (cache_root / "COMPLETE.json").exists():
        from apnet_pt.training.smoke import load_prepared_feature_cache

        manifest = json.loads((cache_root / "COMPLETE.json").read_text())
        if manifest.get("benchmark_hash") != lock["benchmark_hash"]:
            raise ProductionBenchmarkError(
                "existing feature cache belongs to another benchmark"
            )
        for kind, identity in identities.items():
            for mode in FEATURE_MODES:
                load_prepared_feature_cache(
                    cache_root,
                    feature_mode=mode,
                    mace_sha256=config["artifacts"]["mace"]["sha256"],
                    mace_model_id="polar-1-s",
                    physics_hash=physics.physics_hash,
                    dataset_kind=kind,
                    dataset_hash=identity["dataset_hash"],
                    preprocessing_hash=identity["preprocessing_hash"],
                    split_hash=identity["split_hash"],
                    dtype=torch.float32,
                )
        return manifest
    all_entries = []
    schemas = {}
    memberships = {kind: set() for kind in ("pair", "atomic")}
    for mode in FEATURE_MODES:
        backbone = load_verified_polar_mace(
            config["artifacts"]["mace"]["path"],
            expected_sha256=config["artifacts"]["mace"]["sha256"],
            device=device,
            offline=True,
        )
        writer = _StreamingCache(
            cache_root,
            mode,
            {
                "mace_sha256": config["artifacts"]["mace"]["sha256"],
                "physics_hash": physics.physics_hash,
            },
        )
        featurizer = MACEPolarFeaturizer(
            backbone,
            checkpoint_sha256=config["artifacts"]["mace"]["sha256"],
            model_id="polar-1-s",
            feature_mode=mode,
            dtype=torch.float32,
            physics_config=physics,
            cache=writer,
        )
        writer.scope = "pair"
        for split_dataset in pair.values():
            for batch in _pair_loader(
                split_dataset,
                config["training"]["batch_size"],
                config["training"]["num_workers"],
            ):
                batch = batch.to(device)
                with torch.no_grad():
                    featurizer.forward_dimer(batch)
        writer.scope = "atomic"
        for split_dataset in atomic.values():
            for batch in _atomic_loader(
                split_dataset,
                config["training"]["atomic_batch_size"],
                config["training"]["num_workers"],
            ):
                batch = batch.to(device)
                with torch.no_grad():
                    featurizer.forward_monomer(
                        batch.R,
                        batch.x.long(),
                        batch.total_charge.float(),
                        batch.total_spin.float(),
                        batch=batch.molecule_ind.long(),
                    )
        if not writer.feature_schema:
            raise ProductionBenchmarkError("feature preparation produced no records")
        schemas[mode] = writer.feature_schema
        all_entries.extend(writer.entries.values())
        for kind, values in memberships.items():
            values.update(writer.membership[kind])
        del featurizer, backbone
        torch.cuda.empty_cache()

    dataset_identity = {}
    for kind, identity in identities.items():
        dataset_identity[kind] = {
            **identity,
            "monomer_count": len(memberships[kind]),
            "monomer_hashes": sorted(memberships[kind]),
        }
    in_scope_monomers = memberships["pair"] | memberships["atomic"]
    if any(entry["monomer_hash"] not in in_scope_monomers for entry in all_entries):
        raise ProductionBenchmarkError(
            "feature cache contains a monomer outside the frozen split membership"
        )
    listed_paths = {entry["path"] for entry in all_entries}
    actual_paths = {
        str(path.relative_to(cache_root)) for path in cache_root.glob("*/*.pt")
    }
    if actual_paths != listed_paths:
        raise ProductionBenchmarkError(
            "feature cache contains stale, missing, or unlisted entries"
        )
    manifest = {
        "status": "complete",
        "cache_format": CACHE_FORMAT,
        "benchmark_hash": lock["benchmark_hash"],
        "source_commit": lock["source_commit"],
        "environment": _environment(),
        "mace_sha256": config["artifacts"]["mace"]["sha256"],
        "mace_model_id": "polar-1-s",
        "physics_hash": physics.physics_hash,
        "dtype": "float32",
        "dataset_identity": dataset_identity,
        "feature_schemas": schemas,
        "entry_count": len(all_entries),
        "entries": sorted(
            all_entries, key=lambda item: (item["feature_mode"], item["cache_key"])
        ),
    }
    _atomic_json(cache_root / "COMPLETE.json", manifest)
    return manifest


def _mace_args(
    lock: Mapping[str, Any],
    root: Path,
    *,
    route: str | None = None,
    property_mode: str | None = None,
    seed: int,
):
    from train_models import build_parser

    config = lock["config"]
    artifacts = config["artifacts"]
    output = root / "tmp" / f"factory-{route or property_mode}-{seed}.pt"
    pair_root, atomic_root = _staged_roots(root)
    data_root = pair_root if route is not None else atomic_root
    values = [
        "--skip_compile",
        "--mace_offline",
        "--mace_device",
        "cuda",
        "--mace_model_path",
        artifacts["mace"]["path"],
        "--mace_model_sha256",
        artifacts["mace"]["sha256"],
        "--random_seed",
        str(seed),
        "--lr",
        str(config["training"]["learning_rate"]),
        "--batch_size",
        str(
            config["training"]["batch_size"]
            if route is not None
            else config["training"]["atomic_batch_size"]
        ),
        "--data_dir",
        str(data_root),
        "--long_range_elst",
        str(config["physics"]["electrostatics_mode"]),
    ]
    d3_parameters = config["physics"].get("d3_parameters", [])
    if d3_parameters:
        values.extend(["--d3_params", ",".join(str(value) for value in d3_parameters)])
    if route is not None:
        values.extend(
            [
                "--train_apnet",
                PAIR_ROUTE_OPTIONS[route],
                "--ap_model_path",
                str(output),
                "--n_epochs",
                str(config["training"]["epochs"]),
            ]
        )
        if config["training"]["include_total_mse"]:
            values.append("--include_total_mse")
        if route in {"H1", "H2"}:
            values.extend(
                [
                    "--am_model_path",
                    artifacts["legacy_atom_model"]["path"],
                    "--atom_type_param_model_path",
                    artifacts["legacy_parameter_model"]["path"],
                    "--atom_type_param_model_path2",
                    artifacts["legacy_parameter_model_2"]["path"],
                ]
            )
        else:
            checkpoint_mode = (
                "direct-completion" if route == "DirectPolar" else "learned"
            )
            values.extend(
                [
                    "--mace_atom_model_path",
                    str(
                        root
                        / "atomic"
                        / checkpoint_mode
                        / f"seed-{seed}"
                        / "checkpoint.pt"
                    ),
                ]
            )
    else:
        values.extend(
            [
                "--train_am",
                "MACE-AtomicProperties",
                "--am_model_path",
                str(output),
                "--n_epochs_atom",
                str(config["training"]["atomic_epochs"]),
                "--mace_property_mode",
                str(property_mode),
                "--train_atomic_heads",
            ]
        )
    return build_parser().parse_args(values)


def _build_mace_harness(args: Any, *, atomic: bool):
    from apnet_pt.training.mace_ap3d3_factory import (
        _default_factory_dependencies,
        build_mace_ap3d3_harness,
        build_mace_atomic_harness,
        validate_mace_cli_args,
    )

    try:
        plan = validate_mace_cli_args(args)
        dependencies = _default_factory_dependencies(plan)
        harness = (
            build_mace_atomic_harness(plan, dependencies=dependencies)
            if atomic
            else build_mace_ap3d3_harness(plan, dependencies=dependencies)
        )
    except (ImportError, ModuleNotFoundError) as exc:
        raise BenchmarkBlocked(
            "the pinned optional MACE/e3nn production environment is unavailable"
        ) from exc
    except RuntimeError as exc:
        if "pinned optional" in str(exc) or "not installed" in str(exc):
            raise BenchmarkBlocked(str(exc)) from exc
        raise
    return plan, harness


def _attach_prepared_cache(
    lock: Mapping[str, Any], root: Path, featurizer: Any, kind: str
) -> None:
    from apnet_pt.training.smoke import load_prepared_feature_cache

    identity = _dataset_identities(lock)[kind]
    featurizer.cache = load_prepared_feature_cache(
        root / "feature-cache",
        feature_mode=featurizer.feature_mode,
        mace_sha256=lock["config"]["artifacts"]["mace"]["sha256"],
        mace_model_id="polar-1-s",
        physics_hash=featurizer.physics_config.physics_hash,
        dataset_kind=kind,
        dataset_hash=identity["dataset_hash"],
        preprocessing_hash=identity["preprocessing_hash"],
        split_hash=identity["split_hash"],
        dtype=torch.float32,
    )


def _target_bundle(
    batch: Any, prediction: AtomicPropertyBundle
) -> AtomicPropertyBundle:
    """Build MBIS q/mu/Q targets; response fields are excluded from science metrics."""

    return AtomicPropertyBundle(
        q=batch.charges.reshape(-1, 1).to(prediction.q),
        mu=batch.dipoles.to(prediction.mu),
        quadrupole=batch.quadrupoles.to(prediction.quadrupole),
        hfvr=prediction.hfvr.detach(),
        valence_width=prediction.valence_width.detach(),
        alpha=prediction.alpha.detach(),
        damping=prediction.damping.detach(),
    )


def _atomic_loss(
    prediction: AtomicPropertyBundle, target: AtomicPropertyBundle
) -> torch.Tensor:
    losses = [
        (prediction.q - target.q).square().mean(),
        (prediction.mu - target.mu).square().mean(),
        (prediction.quadrupole - target.quadrupole).square().mean(),
    ]
    return torch.stack(losses).mean()


def _evaluate_atomic(model: Any, loader: Iterable[Any], device: torch.device):
    predictions = {name: [] for name in ATOMIC_METRICS}
    targets = {name: [] for name in ATOMIC_METRICS}
    losses = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            prediction = model(
                batch.R,
                batch.x.long(),
                batch.total_charge.float(),
                batch.total_spin.float(),
                batch=batch.molecule_ind.long(),
            )
            target = _target_bundle(batch, prediction)
            losses.append(float(_atomic_loss(prediction, target)))
            for name in ATOMIC_METRICS:
                predictions[name].append(getattr(prediction, name).detach().cpu())
                targets[name].append(getattr(target, name).detach().cpu())
    if not losses:
        raise ProductionBenchmarkError("atomic evaluation loader is empty")
    metrics = {
        name: _metric(torch.cat(predictions[name]) - torch.cat(targets[name]))
        for name in ATOMIC_METRICS
    }
    return sum(losses) / len(losses), metrics


def train_atomic(
    lock: Mapping[str, Any], root: Path, property_mode: str, seed: int
) -> Mapping[str, Any]:
    run_dir = root / "atomic" / property_mode / f"seed-{seed}"
    if (run_dir / "checkpoint.pt").exists() or (run_dir / "result.json").exists():
        raise ProductionBenchmarkError(
            f"atomic run output already exists; implicit resume is forbidden: {run_dir}"
        )
    device = _device()
    _set_seeds(seed)
    config = lock["config"]
    subsets = _atomic_subsets(
        _build_atomic_datasets(lock, root), lock["split_manifest"]
    )
    args = _mace_args(lock, root, property_mode=property_mode, seed=seed)
    _plan, model = _build_mace_harness(args, atomic=True)
    _attach_prepared_cache(lock, root, model.featurizer, "atomic")
    model.to(device)
    loaders = {
        name: _atomic_loader(
            data,
            config["training"]["atomic_batch_size"],
            config["training"]["num_workers"],
            shuffle=name == "train",
            seed=seed if name == "train" else None,
        )
        for name, data in subsets.items()
    }
    parameters = model.trainable_parameters()
    optimizer = torch.optim.Adam(parameters, lr=config["training"]["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=(
            config["training"]["end_learning_rate"]
            / config["training"]["learning_rate"]
        )
        ** (1.0 / max(1, config["training"]["atomic_epochs"])),
    )
    best = None
    best_loss = float("inf")
    best_epoch = 0
    history = []
    patience = 0
    for epoch in range(1, config["training"]["atomic_epochs"] + 1):
        model.train()
        train_losses = []
        for batch in loaders["train"]:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(
                batch.R,
                batch.x.long(),
                batch.total_charge.float(),
                batch.total_spin.float(),
                batch=batch.molecule_ind.long(),
            )
            target = _target_bundle(batch, prediction)
            loss = _atomic_loss(prediction, target)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach()))
        scheduler.step()
        validation_loss, _ = _evaluate_atomic(model, loaders["validation"], device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": sum(train_losses) / len(train_losses),
                "validation_loss": validation_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            patience = 0
            best = {
                key: value.detach().cpu().clone()
                for key, value in model.property_provider.state_dict().items()
            }
        else:
            patience += 1
            if patience >= config["training"]["early_stopping_patience"]:
                break
    if best is None:
        raise ProductionBenchmarkError("atomic training produced no checkpoint")
    model.property_provider.load_state_dict(best)
    model.to(device)
    metrics = {
        name: _evaluate_atomic(model, loader, device)[1]
        for name, loader in loaders.items()
    }
    checkpoint = {
        "checkpoint_version": 3,
        "model_type": "MACEAtomicProperties",
        "model_state_dict": best,
        "config": {
            "property_mode": property_mode,
            "provider_kind": "atomhead" if property_mode == "learned" else "direct",
            "mace": {
                "sha256": config["artifacts"]["mace"]["sha256"],
                "version": model.featurizer.mace_version,
                "model_class": type(model.featurizer.backbone).__module__
                + "."
                + type(model.featurizer.backbone).__name__,
                "feature_schema": model.featurizer.resolved_feature_schema,
                "feature_mode": model.featurizer.feature_mode,
            },
            "dtype_policy": "float32",
            "atomic_property_schema": "ap3-atomic-properties-cartesian-v1",
            "quadrupole_convention": "cartesian-symmetric-traceless-3x3",
            "physics_hash": model.featurizer.physics_config.physics_hash,
            "data": {
                **_dataset_identities(lock)["atomic"],
                "target_method": config["dataset"]["atomic_target_method"],
            },
            "seed": seed,
            "best_epoch": best_epoch,
        },
    }
    _atomic_torch(run_dir / "checkpoint.pt", checkpoint)
    result = {
        "schema_version": 1,
        "benchmark_hash": lock["benchmark_hash"],
        "property_mode": property_mode,
        "seed": seed,
        "status": "PASS",
        "epochs_requested": config["training"]["atomic_epochs"],
        "epochs_completed": len(history),
        "termination_reason": "epoch_budget"
        if len(history) == config["training"]["atomic_epochs"]
        else "early_stopping",
        "best_epoch": best_epoch,
        "metrics": metrics,
        "history": history,
        "environment": _environment(),
    }
    _atomic_json(run_dir / "result.json", result)
    return result


def _full_base_prediction(model: Any, batch: Any) -> torch.Tensor:
    residual = model(batch)[0].reshape(-1, 4)
    prediction = residual.clone()
    prediction[:, 0] += batch.E_classical_elst
    prediction[:, 2] += batch.E_classical_ind
    prediction[:, 3] += batch.E_classical_disp
    return prediction


def _pair_loss(
    prediction: torch.Tensor, target: torch.Tensor, include_total: bool
) -> torch.Tensor:
    terms = [
        (prediction[:, index] - target[:, index]).square().mean() for index in range(4)
    ]
    if include_total:
        terms.append((prediction.sum(1) - target.sum(1)).square().mean())
    return torch.stack(terms).mean()


def _evaluate_pair(
    model: Any,
    loader: Iterable[Any],
    device: torch.device,
    route: str,
    include_total: bool,
):
    predictions = []
    targets = []
    losses = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            prediction = (
                _full_base_prediction(model, batch) if route == "BASE" else model(batch)
            )
            target = batch.y[:, :4].to(prediction)
            losses.append(float(_pair_loss(prediction, target, include_total)))
            predictions.append(prediction.detach().cpu())
            targets.append(target.detach().cpu())
    if not losses:
        raise ProductionBenchmarkError("pair evaluation loader is empty")
    prediction = torch.cat(predictions)
    target = torch.cat(targets)
    return sum(losses) / len(losses), energy_metrics(prediction, target)


def _save_pair_checkpoint(
    lock: Mapping[str, Any],
    root: Path,
    route: str,
    seed: int,
    model: Any,
    wrapper: Any,
    plan: Any | None,
    state: Mapping[str, Any],
    best_epoch: int,
) -> Path:
    run_dir = root / "runs" / route / f"seed-{seed}"
    path = run_dir / "checkpoint.pt"
    if route == "BASE":
        wrapper.model.load_state_dict(state)
        wrapper.model_save_path = str(path)
        wrapper.save_model(
            str(path),
            metadata={
                "purpose": "production BASE-versus-MACE benchmark",
                "benchmark_hash": lock["benchmark_hash"],
                "seed": seed,
                "best_epoch": best_epoch,
                "data": {
                    **_dataset_identities(lock)["pair"],
                    "pair_spec_type": 2,
                    "target_method": lock["config"]["dataset"]["target_method"],
                },
                "physics_hash": lock["config"]["physics"]["physics_hash"],
            },
        )
    else:
        incompatible = model.load_state_dict(state, strict=False)
        invalid = [
            key
            for key in incompatible.missing_keys
            if not key.startswith("featurizer.backbone.")
        ]
        if invalid or incompatible.unexpected_keys:
            raise ProductionBenchmarkError(
                f"MACE checkpoint state mismatch: {invalid}, {incompatible.unexpected_keys}"
            )
        from apnet_pt.training.smoke import _production_pair_checkpoint

        data_identity = _dataset_identities(lock)["pair"]
        dataset_record = SimpleNamespace(
            content_hash=data_identity["dataset_hash"],
            split_hash=data_identity["split_hash"],
        )
        diagnostics = model.last_induction_diagnostics
        if diagnostics is None:
            raise ProductionBenchmarkError(
                "MACE checkpoint requires induction diagnostics from evaluation"
            )
        diag = {
            "converged": diagnostics.converged,
            "iterations": diagnostics.iterations,
            "residual": diagnostics.residual,
            "policy": model.long_range_provider.config.scf_nonconvergence,
        }
        checkpoint = _production_pair_checkpoint(model, dataset_record, plan, diag)
        checkpoint["config"]["data"] = {
            **data_identity,
            "pair_spec_type": 2,
            "target_method": lock["config"]["dataset"]["target_method"],
        }
        checkpoint["metadata"] = {
            **checkpoint.get("metadata", {}),
            "purpose": "production BASE-versus-MACE benchmark",
            "benchmark_hash": lock["benchmark_hash"],
            "seed": seed,
            "best_epoch": best_epoch,
        }
        _atomic_torch(path, checkpoint)
    return path


def train_pair(
    lock: Mapping[str, Any], root: Path, route: str, seed: int
) -> Mapping[str, Any]:
    run_dir = root / "runs" / route / f"seed-{seed}"
    result_path = root / "results" / f"{route}-seed{seed}.json"
    if (run_dir / "checkpoint.pt").exists() or result_path.exists():
        raise ProductionBenchmarkError(
            f"pair run output already exists; implicit resume is forbidden: {run_dir}"
        )
    device = _device()
    _set_seeds(seed)
    started = time.monotonic()
    config = lock["config"]
    wrapper = _build_pair_wrapper(lock, root)
    subsets = _pair_subsets(wrapper, lock["split_manifest"])
    plan = None
    if route == "BASE":
        model = wrapper.model.to(device)
    else:
        # The wrapper is used only to open the canonical processed datasets for
        # MACE routes. Release its duplicate legacy model hierarchy before
        # constructing PolarMACE on the same GPU.
        _release_wrapper_models(wrapper)
        args = _mace_args(lock, root, route=route, seed=seed)
        plan, harness = _build_mace_harness(args, atomic=False)
        model = harness.model.to(device)
        _attach_prepared_cache(lock, root, model.featurizer, "pair")
    loaders = {
        name: _pair_loader(
            data,
            config["training"]["batch_size"],
            config["training"]["num_workers"],
            shuffle=name == "train",
            seed=seed if name == "train" else None,
        )
        for name, data in subsets.items()
    }
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.Adam(parameters, lr=config["training"]["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=(
            config["training"]["end_learning_rate"]
            / config["training"]["learning_rate"]
        )
        ** (1.0 / max(1, config["training"]["epochs"])),
    )
    best = None
    best_loss = float("inf")
    best_epoch = 0
    history = []
    patience = 0
    for epoch in range(1, config["training"]["epochs"] + 1):
        model.train()
        losses = []
        for batch in loaders["train"]:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = (
                _full_base_prediction(model, batch) if route == "BASE" else model(batch)
            )
            target = batch.y[:, :4].to(prediction)
            loss = _pair_loss(
                prediction, target, config["training"]["include_total_mse"]
            )
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        validation_loss, validation_metrics = _evaluate_pair(
            model,
            loaders["validation"],
            device,
            route,
            config["training"]["include_total_mse"],
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": sum(losses) / len(losses),
                "validation_loss": validation_loss,
                "validation_total_mae": validation_metrics["total"]["mae"],
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            patience = 0
            best = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
                if not key.startswith("featurizer.backbone.")
            }
        else:
            patience += 1
            if patience >= config["training"]["early_stopping_patience"]:
                break
    if best is None:
        raise ProductionBenchmarkError("pair training produced no checkpoint")
    if route == "BASE":
        model.load_state_dict(best)
    else:
        model.load_state_dict(best, strict=False)
    model.to(device)
    metrics = {
        name: _evaluate_pair(
            model, loader, device, route, config["training"]["include_total_mse"]
        )[1]
        for name, loader in loaders.items()
    }
    checkpoint = _save_pair_checkpoint(
        lock, root, route, seed, model, wrapper, plan, best, best_epoch
    )
    result = {
        "schema_version": 1,
        "benchmark_hash": lock["benchmark_hash"],
        "route": route,
        "seed": seed,
        "status": "PASS",
        "epochs_requested": config["training"]["epochs"],
        "epochs_completed": len(history),
        "termination_reason": (
            "epoch_budget"
            if len(history) == config["training"]["epochs"]
            else "early_stopping"
        ),
        "best_epoch": best_epoch,
        "metrics": metrics,
        "resources": {
            "elapsed_seconds": time.monotonic() - started,
            "peak_rss_mb": _peak_rss_mb(),
        },
        "history": history,
        "checkpoint": str(checkpoint),
        "environment": _environment(),
        "parameter_counts": {
            "total": sum(parameter.numel() for parameter in model.parameters()),
            "trainable": sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
            "external_frozen_backbone_excluded_from_checkpoint": route != "BASE",
        },
    }
    _atomic_json(result_path, result)
    return result


def _peak_rss_mb() -> float:
    import resource

    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / 1024.0 if sys_platform() != "darwin" else value / (1024.0 * 1024.0)


def sys_platform() -> str:
    import sys

    return sys.platform


def sys_executable() -> str:
    import sys

    return sys.executable


def run_benchmark_task(
    *, lock: Mapping[str, Any], task: Mapping[str, Any], benchmark_root: str | Path
) -> Mapping[str, Any]:
    """Execute one generated task and atomically publish its evidence."""

    root = Path(benchmark_root)
    started = time.monotonic()
    kind = task.get("kind")
    try:
        _verify_locked_inputs(lock)
        if kind == "dataset":
            return prepare_datasets(lock, root)
        if kind == "prepare":
            return prepare_features(lock, root)
        if kind == "atomic":
            return train_atomic(
                lock, root, str(task["property_mode"]), int(task["seed"])
            )
        if kind == "report":
            config_path = Path(str(task["config_path"]))
            script = (
                Path(__file__).resolve().parents[3]
                / "scripts"
                / "slurm"
                / "benchmark.py"
            )
            completed = subprocess.run(
                [sys_executable(), str(script), "--config", str(config_path), "--plot"],
                capture_output=True,
                text=True,
                check=False,
            )
            if completed.returncode != 0:
                raise ProductionBenchmarkError(
                    f"automatic report generation failed: {completed.stderr}"
                )
            return json.loads((root / "plots" / "summary.json").read_text())
        if kind == "pair":
            route = str(task["route"])
            if route not in {"BASE", *PAIR_ROUTE_OPTIONS}:
                raise ProductionBenchmarkError(f"unsupported pair route: {route}")
            return train_pair(lock, root, route, int(task["seed"]))
        raise ProductionBenchmarkError(f"unsupported benchmark task kind: {kind}")
    except Exception as exc:
        if kind == "pair" and "route" in task and "seed" in task:
            route = str(task["route"])
            seed = int(task["seed"])
            failure = {
                "schema_version": 1,
                "benchmark_hash": lock["benchmark_hash"],
                "route": route,
                "seed": seed,
                "status": "BLOCKED" if isinstance(exc, BenchmarkBlocked) else "FAIL",
                "epochs_requested": lock["config"]["training"]["epochs"],
                "epochs_completed": 0,
                "termination_reason": "exception",
                "metrics": {},
                "resources": {
                    "elapsed_seconds": time.monotonic() - started,
                    "peak_rss_mb": _peak_rss_mb(),
                },
                "history": [],
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
            failure_path = root / "results" / f"{route}-seed{seed}.json"
            if not failure_path.exists():
                _atomic_json(failure_path, failure)
        elif kind == "atomic" and "property_mode" in task and "seed" in task:
            mode = str(task["property_mode"])
            seed = int(task["seed"])
            failure = {
                "schema_version": 1,
                "benchmark_hash": lock["benchmark_hash"],
                "property_mode": mode,
                "seed": seed,
                "status": "BLOCKED" if isinstance(exc, BenchmarkBlocked) else "FAIL",
                "epochs_requested": lock["config"]["training"]["atomic_epochs"],
                "epochs_completed": 0,
                "termination_reason": "exception",
                "metrics": {},
                "history": [],
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
            failure_path = root / "atomic" / mode / f"seed-{seed}" / "result.json"
            if not failure_path.exists():
                _atomic_json(failure_path, failure)
        raise
