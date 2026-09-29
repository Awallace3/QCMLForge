"""Deterministic, network-free smoke fixtures and one-epoch lifecycles."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import pickle
import subprocess
from collections.abc import Iterator, MutableMapping
from typing import Any, Mapping, Sequence

import numpy as np
import qcelemental as qcel
import torch

from apnet_pt import constants
from apnet_pt.mace.schema import (
    AtomicPropertyBundle,
    MACEAtomicFeatures,
    PolarMACEDirectOutputs,
)
from apnet_pt.pt_datasets.ap3_fused_ds import (
    ap3_fused_collate_update,
    qcel_dimer_to_fused_data,
)


PAIR_SMOKE_SCHEMA = "qcmlforge-mace-ap3d3-smoke-v1"
ATOMIC_SMOKE_SCHEMA = "qcmlforge-mace-atomic-properties-smoke-v1"


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_value(value: Any) -> Any:
    if isinstance(value, qcel.models.Molecule):
        return json.loads(value.json())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if key != "content_hash"
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    return value


def fixture_content_hash(fixture: Mapping[str, Any]) -> str:
    """Hash fixture semantics independently of pickle implementation details."""

    encoded = json.dumps(
        _canonical_value(fixture),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _load_fixture(path: str | Path, schema: str) -> dict[str, Any]:
    fixture_path = Path(path)
    if not fixture_path.is_file():
        raise FileNotFoundError(f"smoke fixture was not found: {fixture_path}")
    with fixture_path.open("rb") as handle:
        fixture = pickle.load(handle)
    if not isinstance(fixture, dict) or fixture.get("schema") != schema:
        raise ValueError(f"smoke fixture schema must be {schema}")
    expected_hash = fixture.get("content_hash")
    actual_hash = fixture_content_hash(fixture)
    if expected_hash != actual_hash:
        raise ValueError(
            f"smoke fixture content hash mismatch: expected {expected_hash}, "
            f"got {actual_hash}"
        )
    return fixture


class PreparedFeatureCache(MutableMapping):
    """Read-only prepared features with strict scoped identity validation."""

    SUPPORTED_FORMATS = {
        "qcmlforge-mace-monomer-cache-v1",
        "qcmlforge-mace-monomer-cache-v2",
    }
    REQUIRED_TENSORS = {
        "invariant", "equivariant", "batch", "atomic_numbers", "total_charge",
        "total_spin", "density_coefficients", "charges",
        "molecular_dipole_eangstrom", "positions_angstrom",
    }

    def __init__(
        self,
        path: str | Path,
        *,
        feature_mode: str,
        mace_sha256: str,
        mace_model_id: str,
        physics_hash: str,
        dataset_kind: str,
        dataset_hash: str,
        preprocessing_hash: str,
        split_hash: str,
        dtype: torch.dtype,
    ) -> None:
        self.cache_dir = Path(path)
        complete = self.cache_dir / "COMPLETE.json"
        if not complete.is_file():
            raise RuntimeError("partial feature cache: COMPLETE.json is missing")
        try:
            self.manifest = json.loads(complete.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("partial feature cache: COMPLETE.json is invalid") from exc
        if dataset_kind not in {"pair", "atomic"}:
            raise ValueError("dataset_kind must be 'pair' or 'atomic'")
        for name, value in {
            "mace_sha256": mace_sha256,
            "mace_model_id": mace_model_id,
            "physics_hash": physics_hash,
            "dataset_hash": dataset_hash,
            "preprocessing_hash": preprocessing_hash,
            "split_hash": split_hash,
        }.items():
            if not isinstance(value, str) or not value:
                raise ValueError(f"prepared feature cache requires explicit {name}")

        cache_format = self.manifest.get("cache_format")
        if cache_format not in self.SUPPORTED_FORMATS:
            raise RuntimeError("prepared feature cache mismatch for cache_format")
        identity = self._dataset_identity(cache_format, dataset_kind)
        scoped_monomers = None
        if cache_format.endswith("v2"):
            values = identity.get("monomer_hashes")
            if (
                not isinstance(values, list)
                or not values
                or any(not isinstance(value, str) or not value for value in values)
                or len(values) != len(set(values))
                or identity.get("monomer_count") != len(values)
            ):
                raise RuntimeError(
                    f"prepared feature cache {dataset_kind} membership is invalid"
                )
            scoped_monomers = set(values)
        expected = {
            "status": "complete",
            "mace_sha256": mace_sha256,
            "mace_model_id": mace_model_id,
            "physics_hash": physics_hash,
            "dtype": str(dtype).removeprefix("torch."),
        }
        for name, value in expected.items():
            if self.manifest.get(name) != value:
                raise RuntimeError(f"prepared feature cache mismatch for {name}")
        for name, value in {
            "dataset_hash": dataset_hash,
            "preprocessing_hash": preprocessing_hash,
            "split_hash": split_hash,
        }.items():
            if identity.get(name) != value:
                raise RuntimeError(f"prepared feature cache mismatch for {name}")

        entries = self.manifest.get("entries")
        if not isinstance(entries, list) or self.manifest.get("entry_count") != len(entries):
            raise RuntimeError("prepared feature cache entry_count is inconsistent")
        schemas = self.manifest.get("feature_schemas")
        if not isinstance(schemas, Mapping) or feature_mode not in schemas:
            raise RuntimeError("prepared feature cache feature schema is missing")
        if cache_format.endswith("v2") and not set(("final-layer-scalars", "all-scalars+norms")).issubset(schemas):
            raise RuntimeError("prepared feature cache is missing a required feature mode")

        self.feature_mode = feature_mode
        self.mace_sha256 = mace_sha256
        self.mace_model_id = mace_model_id
        self.physics_hash = physics_hash
        self.dtype = dtype
        self.cache_format = cache_format
        self.feature_schema = schemas[feature_mode]
        self._entries = {}
        listed = set()
        seen = set()
        mode_monomers: dict[str, set[str]] = {mode: set() for mode in schemas}
        all_v2_monomers: set[str] | None = None
        if cache_format.endswith("v2"):
            identities = self.manifest["dataset_identity"]
            if set(identities) != {"pair", "atomic"}:
                raise RuntimeError("prepared feature cache dataset scopes are invalid")
            all_v2_monomers = set()
            for kind in ("pair", "atomic"):
                values = identities[kind].get("monomer_hashes")
                if not isinstance(values, list):
                    raise RuntimeError("prepared feature cache membership is missing")
                all_v2_monomers.update(values)
        for entry in entries:
            if not isinstance(entry, Mapping):
                raise RuntimeError("prepared feature cache entry is invalid")
            required = {"path", "sha256", "cache_key", "feature_mode", "monomer_hash"}
            if not required.issubset(entry):
                raise RuntimeError("prepared feature cache entry metadata is incomplete")
            pair = (entry["feature_mode"], entry["cache_key"])
            if pair in seen:
                raise RuntimeError("prepared feature cache contains duplicate cache key")
            seen.add(pair)
            relative = Path(entry["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise RuntimeError("prepared feature cache entry path escapes cache root")
            entry_path = (self.cache_dir / relative).resolve()
            try:
                entry_path.relative_to(self.cache_dir.resolve())
            except ValueError as exc:
                raise RuntimeError("prepared feature cache entry path escapes cache root") from exc
            listed.add(entry_path)
            if not entry_path.is_file() or _sha256_file(entry_path) != entry["sha256"]:
                raise RuntimeError(f"prepared feature cache entry is corrupt: {entry_path}")
            mode = entry["feature_mode"]
            if mode not in schemas:
                raise RuntimeError("prepared feature cache entry has no feature schema")
            monomer_hash = entry["monomer_hash"]
            if all_v2_monomers is not None and monomer_hash not in all_v2_monomers:
                raise RuntimeError("prepared feature cache contains an out-of-scope monomer")
            if monomer_hash in mode_monomers[mode]:
                raise RuntimeError("prepared feature cache contains duplicate monomer")
            mode_monomers[mode].add(monomer_hash)
            self._validate_record(dict(entry), mode, schemas[mode])
            if mode == feature_mode and (
                scoped_monomers is None or monomer_hash in scoped_monomers
            ):
                self._entries[entry["cache_key"]] = dict(entry)
        actual = {item.resolve() for item in self.cache_dir.rglob("*.pt")}
        if actual != listed:
            raise RuntimeError("prepared feature cache contains unlisted or missing entries")
        if scoped_monomers is not None:
            for mode in ("final-layer-scalars", "all-scalars+norms"):
                missing = scoped_monomers - mode_monomers[mode]
                if missing:
                    raise RuntimeError(
                        f"prepared feature cache {dataset_kind} membership is incomplete "
                        f"for {mode}"
                    )
            selected = {
                entry["monomer_hash"] for entry in self._entries.values()
            }
            if selected != scoped_monomers:
                raise RuntimeError("prepared feature cache selected membership is inconsistent")
        if not self._entries:
            raise RuntimeError(f"prepared cache has no entries for {feature_mode}")
        self.loaded_entries = 0
        self.strict_read_only = True

    def _dataset_identity(self, cache_format: str, dataset_kind: str) -> Mapping[str, Any]:
        if cache_format.endswith("v2"):
            identities = self.manifest.get("dataset_identity")
            if not isinstance(identities, Mapping) or not isinstance(identities.get(dataset_kind), Mapping):
                raise RuntimeError(f"prepared feature cache has no explicit {dataset_kind} identity")
            return identities[dataset_kind]
        # Compatibility is deliberately limited to explicit v1 identities.
        nested = self.manifest.get("dataset_identity")
        if isinstance(nested, Mapping):
            scoped = nested.get(dataset_kind)
            if isinstance(scoped, Mapping):
                return scoped
            prefix = "pair" if dataset_kind == "pair" else "atomic"
            legacy = {
                "dataset_hash": nested.get(f"{prefix}_content_hash"),
                "preprocessing_hash": nested.get(f"{prefix}_preprocessing_hash"),
                "split_hash": nested.get(f"{prefix}_split_hash"),
            }
            if all(isinstance(value, str) and value for value in legacy.values()):
                return legacy
        top = {name: self.manifest.get(name) for name in ("dataset_hash", "preprocessing_hash", "split_hash")}
        if all(isinstance(value, str) and value for value in top.values()):
            return top
        raise RuntimeError("prepared feature cache v1 identity is not explicit")

    def _validate_record(self, entry, mode, schema):
        entry_path = self.cache_dir / entry["path"]
        record = torch.load(entry_path, map_location="cpu", weights_only=True)
        identity = record.get("identity")
        expected = {
            "format": self.cache_format,
            "monomer_hash": entry["monomer_hash"],
            "cache_key": entry["cache_key"],
            "feature_mode": mode,
            "mace_sha256": self.mace_sha256,
            "mace_model_id": self.mace_model_id,
            "physics_hash": self.physics_hash,
            "dtype": str(self.dtype).removeprefix("torch."),
        }
        if not isinstance(identity, Mapping) or any(
            identity.get(name) != value for name, value in expected.items()
        ):
            raise RuntimeError(f"prepared feature cache identity is stale: {entry_path}")
        if record.get("feature_schema") != schema:
            raise RuntimeError("prepared feature cache feature schema mismatch")
        tensors = record.get("tensors")
        if not isinstance(tensors, Mapping) or not self.REQUIRED_TENSORS.issubset(tensors):
            raise RuntimeError("prepared feature cache tensors are incomplete")
        if any(not torch.is_tensor(value) for value in tensors.values()):
            raise RuntimeError("prepared feature cache tensors are invalid")
        expected_dtype = self.dtype
        floating_names = self.REQUIRED_TENSORS - {"batch", "atomic_numbers"}
        if any(
            not torch.is_floating_point(tensors[name])
            or tensors[name].dtype != expected_dtype
            or not torch.isfinite(tensors[name]).all()
            for name in floating_names
        ):
            raise RuntimeError(
                "prepared feature cache floating tensor dtype/value contract is invalid"
            )
        for name in ("batch", "atomic_numbers"):
            if tensors[name].dtype not in {torch.int32, torch.int64}:
                raise RuntimeError(
                    "prepared feature cache integer tensor contract is invalid"
                )
        try:
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
            natom = features.natom
            nmonomer = features.total_charge.numel()
            if natom == 0 or features.batch.unique(sorted=True).tolist() != list(range(nmonomer)):
                raise ValueError("feature batch must be zero-based and contiguous")
            if not torch.equal(features.batch, direct.batch):
                raise ValueError("feature and direct batches must agree")
        except (TypeError, ValueError, KeyError) as exc:
            raise RuntimeError(
                f"prepared feature cache tensor schema is invalid: {exc}"
            ) from exc
        return record, tensors

    def __getitem__(self, key):
        if key not in self._entries:
            raise KeyError(f"prepared feature cache miss: {key}")
        entry = self._entries[key]
        record, tensors = self._validate_record(
            entry, self.feature_mode, self.feature_schema
        )
        self.loaded_entries += 1
        return (
            MACEAtomicFeatures(
                invariant=tensors["invariant"], equivariant=tensors["equivariant"],
                batch=tensors["batch"], atomic_numbers=tensors["atomic_numbers"],
                total_charge=tensors["total_charge"], total_spin=tensors["total_spin"],
                feature_schema=record["feature_schema"],
            ),
            PolarMACEDirectOutputs(
                density_coefficients=tensors["density_coefficients"], charges=tensors["charges"],
                molecular_dipole_eangstrom=tensors["molecular_dipole_eangstrom"],
                positions_angstrom=tensors["positions_angstrom"], batch=tensors["batch"],
                total_charge=tensors["total_charge"],
            ),
        )

    def __setitem__(self, key, value):
        raise TypeError("prepared feature cache is read-only")

    def __delitem__(self, key):
        raise TypeError("prepared feature cache is read-only")

    def __iter__(self) -> Iterator[str]:
        return iter(self._entries)

    def __contains__(self, key: object) -> bool:
        return key in self._entries

    def __len__(self) -> int:
        return len(self._entries)


def load_prepared_feature_cache(
    path: str | Path,
    *,
    feature_mode: str,
    mace_sha256: str,
    mace_model_id: str,
    physics_hash: str,
    dataset_kind: str,
    dataset_hash: str,
    preprocessing_hash: str,
    split_hash: str,
    dtype: torch.dtype,
) -> PreparedFeatureCache:
    """Open a complete prepared cache with an explicit dataset scope."""

    return PreparedFeatureCache(
        path, feature_mode=feature_mode, mace_sha256=mace_sha256,
        mace_model_id=mace_model_id, physics_hash=physics_hash,
        dataset_kind=dataset_kind, dataset_hash=dataset_hash,
        preprocessing_hash=preprocessing_hash, split_hash=split_hash, dtype=dtype,
    )


def load_smoke_fixture_metadata(path: str | Path) -> dict[str, str]:
    """Read and validate only normalized fixture hashes for CLI planning."""

    with Path(path).open("rb") as handle:
        header = pickle.load(handle)
    schema = header.get("schema") if isinstance(header, Mapping) else None
    if schema not in {PAIR_SMOKE_SCHEMA, ATOMIC_SMOKE_SCHEMA}:
        raise ValueError("unsupported MACE smoke fixture schema")
    fixture = _load_fixture(path, str(schema))
    required = ("content_hash", "split_hash", "preprocessing_hash")
    for name in required:
        value = fixture.get(name)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"smoke fixture {name} must be a SHA-256")
    return {name: str(fixture[name]) for name in required}


@dataclass(frozen=True)
class PairSmokeDataset:
    train_batches: tuple[Any, ...]
    test_batches: tuple[Any, ...]
    train_ids: tuple[str, ...]
    test_ids: tuple[str, ...]
    long_range_test_indices: tuple[int, ...]
    content_hash: str
    split_hash: str
    preprocessing_hash: str
    physics_hash: str
    fixture: Mapping[str, Any]


@dataclass(frozen=True)
class AtomicSmokeBatch:
    positions: torch.Tensor
    atomic_numbers: torch.Tensor
    total_charge: torch.Tensor
    total_spin: torch.Tensor
    batch: torch.Tensor
    target: AtomicPropertyBundle
    ids: tuple[str, ...]


@dataclass(frozen=True)
class AtomicSmokeDataset:
    train_batch: AtomicSmokeBatch
    test_batch: AtomicSmokeBatch
    content_hash: str
    split_hash: str
    preprocessing_hash: str
    fixture: Mapping[str, Any]


def _qcel_molecule(record: object) -> qcel.models.Molecule:
    if not isinstance(record, Mapping):
        raise ValueError("smoke molecules must use a primitive molecule record")
    if record.get("format") != "qcel-psi4-text-v1":
        raise ValueError("unsupported primitive smoke molecule format")
    data = record.get("data")
    if not isinstance(data, str) or not data.strip():
        raise ValueError("primitive smoke molecule data must be non-empty text")
    return qcel.models.Molecule.from_data(data)


def _pair_batch(records: Sequence[Mapping[str, Any]], r_cut: float, r_cut_im: float):
    data = []
    for index, record in enumerate(records):
        dimer = _qcel_molecule(record["dimer"])
        if len(dimer.fragments) != 2:
            raise ValueError("pair smoke records must reconstruct two fragments")
        labels = torch.as_tensor(record["labels"], dtype=torch.float32)
        if labels.shape != (4,) or not torch.isfinite(labels).all():
            raise ValueError("pair smoke labels must be finite [4] vectors")
        item = qcel_dimer_to_fused_data(
            dimer,
            r_cut=r_cut,
            r_cut_im=r_cut_im,
            dimer_ind=index,
            y=labels,
        )
        if item is None:
            raise ValueError(f"pair smoke molecule {record['id']} is invalid")
        data.append(item)
    batch = ap3_fused_collate_update(data)
    batch.total_charge_A = batch.total_charge_A.to(torch.float32)
    batch.total_charge_B = batch.total_charge_B.to(torch.float32)
    batch.batch_atomic_A.total_charge = batch.total_charge_A
    batch.batch_atomic_B.total_charge = batch.total_charge_B
    return batch


def _partition_batches(records, split_ids, batch_size, r_cut, r_cut_im):
    by_id = {record["id"]: record for record in records}
    ids = tuple(split_ids)
    ordered = [by_id[record_id] for record_id in ids]
    return tuple(
        _pair_batch(ordered[start : start + batch_size], r_cut, r_cut_im)
        for start in range(0, len(ordered), batch_size)
    ), ids


def load_pair_smoke_fixture(
    path: str | Path,
    *,
    batch_size: int = 16,
    r_cut: float = 5.0,
    r_cut_im: float = 8.0,
) -> PairSmokeDataset:
    """Load a checked-in dimer fixture without database or network access."""

    if batch_size < 1:
        raise ValueError("smoke batch_size must be positive")
    fixture = _load_fixture(path, PAIR_SMOKE_SCHEMA)
    records = fixture.get("records", [])
    if not 8 <= len(records) <= 16:
        raise ValueError("pair smoke fixture must contain 8-16 records")
    order = [record.get("id") for record in records]
    if order != fixture.get("order") or len(set(order)) != len(order):
        raise ValueError("pair smoke order must be fixed and unique")
    split_ids = fixture.get("split_ids")
    if not isinstance(split_ids, Mapping) or set(split_ids) != {"train", "test"}:
        raise ValueError("pair smoke fixture requires train/test split IDs")
    if list(split_ids["train"]) + list(split_ids["test"]) != order:
        raise ValueError("pair smoke split IDs must preserve fixed order")
    train_batches, train_ids = _partition_batches(
        records, split_ids["train"], batch_size, r_cut, r_cut_im
    )
    test_batches, test_ids = _partition_batches(
        records, split_ids["test"], batch_size, r_cut, r_cut_im
    )
    by_id = {record["id"]: record for record in records}
    long_indices = tuple(
        index for index, record_id in enumerate(test_ids)
        if by_id[record_id]["range"] == "long"
    )
    return PairSmokeDataset(
        train_batches=train_batches,
        test_batches=test_batches,
        train_ids=train_ids,
        test_ids=test_ids,
        long_range_test_indices=long_indices,
        content_hash=fixture["content_hash"],
        split_hash=fixture["split_hash"],
        preprocessing_hash=fixture["preprocessing_hash"],
        physics_hash=fixture["physics_hash"],
        fixture=fixture,
    )


def _atomic_batch(records: Sequence[Mapping[str, Any]]) -> AtomicSmokeBatch:
    positions = []
    numbers = []
    charges = []
    spins = []
    indices = []
    targets: dict[str, list[torch.Tensor]] = {
        name: [] for name in AtomicPropertyBundle.__dataclass_fields__
    }
    ids = []
    for index, record in enumerate(records):
        monomer = _qcel_molecule(record["monomer"])
        if len(monomer.fragments) != 1:
            raise ValueError("atomic smoke records must reconstruct one fragment")
        natom = len(monomer.symbols)
        positions.append(
            torch.as_tensor(monomer.geometry * constants.au2ang, dtype=torch.float32)
        )
        numbers.append(torch.as_tensor(monomer.atomic_numbers, dtype=torch.long))
        charges.append(float(monomer.molecular_charge))
        spins.append(float(monomer.molecular_multiplicity))
        indices.append(torch.full((natom,), index, dtype=torch.long))
        ids.append(record["id"])
        for name in targets:
            targets[name].append(
                torch.as_tensor(record["targets"][name], dtype=torch.float32)
            )
    bundle = AtomicPropertyBundle(
        **{name: torch.cat(values, dim=0) for name, values in targets.items()}
    )
    return AtomicSmokeBatch(
        positions=torch.cat(positions, dim=0),
        atomic_numbers=torch.cat(numbers, dim=0),
        total_charge=torch.tensor(charges, dtype=torch.float32),
        total_spin=torch.tensor(spins, dtype=torch.float32),
        batch=torch.cat(indices, dim=0),
        target=bundle,
        ids=tuple(ids),
    )


def load_atomic_smoke_fixture(path: str | Path) -> AtomicSmokeDataset:
    """Load complete monomer targets with deterministic train/test IDs."""

    fixture = _load_fixture(path, ATOMIC_SMOKE_SCHEMA)
    records = fixture.get("records", [])
    order = [record.get("id") for record in records]
    if order != fixture.get("order") or len(set(order)) != len(order):
        raise ValueError("atomic smoke order must be fixed and unique")
    split_ids = fixture.get("split_ids")
    if not isinstance(split_ids, Mapping) or set(split_ids) != {"train", "test"}:
        raise ValueError("atomic smoke fixture requires train/test split IDs")
    if list(split_ids["train"]) + list(split_ids["test"]) != order:
        raise ValueError("atomic smoke split IDs must preserve fixed order")
    by_id = {record["id"]: record for record in records}
    train = [by_id[record_id] for record_id in split_ids["train"]]
    test = [by_id[record_id] for record_id in split_ids["test"]]
    return AtomicSmokeDataset(
        train_batch=_atomic_batch(train),
        test_batch=_atomic_batch(test),
        content_hash=fixture["content_hash"],
        split_hash=fixture["split_hash"],
        preprocessing_hash=fixture["preprocessing_hash"],
        fixture=fixture,
    )


@dataclass(frozen=True)
class AtomicSmokeReport:
    epochs: int
    loss: float
    losses: Mapping[str, float]
    gradients_finite: bool
    backbone_frozen: bool
    reload_equal: bool
    prediction_atoms: int


@dataclass(frozen=True)
class PairSmokeReport:
    epochs: int
    loss: float
    component_losses: Mapping[str, float]
    gradients_finite: bool
    backbone_frozen: bool
    reload_equal: bool
    prediction_shape: tuple[int, int]
    classical_ledger: Mapping[str, float]
    residual_ledger: Mapping[str, float]
    long_range_classical_nonzero: bool
    long_range_residual_zero: bool
    split_hash: str
    physics_hash: str
    baseline_name: str | None
    induction_converged: bool
    induction_iterations: int
    induction_residual: float
    induction_policy: str


@dataclass(frozen=True)
class BaselineSmokeReport:
    epochs: int
    loss: float
    component_losses: Mapping[str, float]
    gradients_finite: bool
    reload_equal: bool
    prediction_shape: tuple[int, int]
    classical_ledger: Mapping[str, float]
    residual_ledger: Mapping[str, float]
    long_range_classical_nonzero: bool
    long_range_residual_zero: bool
    split_hash: str
    physics_hash: str
    baseline_name: str = "APNet3-fused-d3"


def _backbone_is_frozen(model: torch.nn.Module) -> bool:
    backbone = model.featurizer.backbone
    return (
        not backbone.training
        and all(not parameter.requires_grad for parameter in backbone.parameters())
        and all(parameter.grad is None for parameter in backbone.parameters())
    )


def _gradients_are_finite(model: torch.nn.Module) -> bool:
    """Require every intentionally trainable parameter to receive finite gradients."""

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    return bool(parameters) and all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in parameters
    )


def _atomic_prediction(model, batch: AtomicSmokeBatch) -> AtomicPropertyBundle:
    model.eval()
    with torch.no_grad():
        return model(
            batch.positions,
            batch.atomic_numbers,
            batch.total_charge,
            batch.total_spin,
            batch=batch.batch,
        )


def _bundles_equal(left: AtomicPropertyBundle, right: AtomicPropertyBundle) -> bool:
    return all(
        torch.equal(getattr(left, name), getattr(right, name))
        for name in left.__dataclass_fields__
    )


def run_atomic_smoke_lifecycle(
    model: torch.nn.Module,
    dataset: AtomicSmokeDataset,
    *,
    output_path: str | Path,
    learning_rate: float,
    physics_hash: str = "0" * 64,
) -> AtomicSmokeReport:
    """Run exactly one deterministic atomic-head epoch and state reload."""

    parameters = model.trainable_parameters()
    if not parameters:
        raise ValueError("atomic smoke lifecycle requires trainable head parameters")
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    batch = dataset.train_batch
    loss = model.train_step(
        batch.positions,
        batch.atomic_numbers,
        batch.total_charge,
        batch.total_spin,
        batch=batch.batch,
        target=batch.target,
        optimizer=optimizer,
    )
    gradients_finite = _gradients_are_finite(model)
    expected = _atomic_prediction(model, dataset.test_batch)
    state = {
        key: value.detach().cpu().clone()
        for key, value in model.property_provider.state_dict().items()
    }
    feature_schema = getattr(
        model.featurizer,
        "resolved_feature_schema",
        f"stub:mode=all-scalars+norms:inv=16:equiv=0",
    )
    feature_mode = (
        "all-scalars+norms"
        if ":mode=all-scalars+norms:" in feature_schema
        else "final-layer-scalars"
    )
    checkpoint = {
        "checkpoint_version": 3,
        "model_type": "MACEAtomicProperties",
        "model_state_dict": state,
        "config": {
            "property_mode": model.property_mode,
            "provider_kind": (
                "atomhead" if model.property_mode == "learned" else "direct"
            ),
            "mace": {
                "sha256": getattr(model.featurizer, "checkpoint_sha256", "0" * 64),
                "version": getattr(model.featurizer, "mace_version", "unknown"),
                "model_class": (
                    type(model.featurizer.backbone).__module__
                    + "."
                    + type(model.featurizer.backbone).__name__
                ),
                "feature_schema": feature_schema,
                "feature_mode": feature_mode,
            },
            "dtype_policy": "float32",
            "atomic_property_schema": "ap3-atomic-properties-cartesian-v1",
            "quadrupole_convention": "cartesian-symmetric-traceless-3x3",
            "physics_hash": physics_hash,
            "data": {
                "dataset_hash": dataset.content_hash,
                "preprocessing_hash": dataset.preprocessing_hash,
                "split_hash": dataset.split_hash,
            },
        },
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    with torch.no_grad():
        for parameter in model.property_provider.parameters():
            parameter.zero_()
    restored = torch.load(output, map_location="cpu", weights_only=True)
    model.property_provider.load_state_dict(restored["model_state_dict"], strict=True)
    actual = _atomic_prediction(model, dataset.test_batch)
    losses = {
        name: float(value) for name, value in model.last_property_losses.items()
    }
    return AtomicSmokeReport(
        epochs=1,
        loss=float(loss),
        losses=losses,
        gradients_finite=gradients_finite,
        backbone_frozen=_backbone_is_frozen(model),
        reload_equal=_bundles_equal(expected, actual),
        prediction_atoms=expected.natom,
    )


def _filtered_model_state(model: torch.nn.Module) -> dict[str, Any]:
    state = {}
    for key, value in model.state_dict().items():
        if key.startswith("featurizer.backbone."):
            continue
        state[key] = (
            value.detach().cpu().clone() if torch.is_tensor(value) else deepcopy(value)
        )
    return state


def _restore_filtered_state(model: torch.nn.Module, state: Mapping[str, Any]):
    incompatible = model.load_state_dict(state, strict=False)
    invalid_missing = [
        key for key in incompatible.missing_keys
        if not key.startswith("featurizer.backbone.")
    ]
    if invalid_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            f"smoke checkpoint state mismatch: missing={invalid_missing}, "
            f"unexpected={incompatible.unexpected_keys}"
        )


def _production_pair_checkpoint(
    model, dataset, plan, induction_diagnostics: Mapping[str, Any]
) -> dict[str, Any]:
    """Build the strict external-backbone v3 record for a real smoke run."""

    physics = model.long_range_provider.config
    try:
        source_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[3],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        source_commit = "0" * 40
    if len(source_commit) != 40:
        source_commit = "0" * 40
    feature_schema = getattr(model.featurizer, "resolved_feature_schema", None)
    if not feature_schema:
        raise RuntimeError("real smoke checkpoint requires a resolved feature schema")
    config = {
        "architecture": model.architecture,
        "mace": {
            "model_id": model.featurizer.model_id,
            "version": model.featurizer.mace_version,
            "sha256": plan.mace_sha256,
            "feature_schema": feature_schema,
            "feature_mode": model.featurizer.feature_mode,
        },
        "pair_mode": model.pair_core.pair_mode,
        "dtype_policy": plan.mace_default_dtype,
        "atomic_property_schema": model.atomic_property_schema,
        "physics": {
            "electrostatics_mode": physics.electrostatics_mode,
            "induction_mode": "thole-scf",
            "dispersion_mode": "d3",
            "d3_parameters": physics.d3_parameters,
            "component_order": physics.component_order,
            "length_unit": physics.length_unit,
            "energy_unit": physics.energy_unit,
            "physics_hash": physics.physics_hash,
            "neural_cutoff": physics.neural_cutoff,
            "induction_diagnostics": dict(induction_diagnostics),
        },
        "data": {
            "dataset_hash": dataset.content_hash,
            "preprocessing_hash": plan.preprocessing_hash,
            "split_hash": dataset.split_hash,
        },
        "seed": plan.random_seed,
        "source_commit": source_commit,
        "route_submodel_digests": dict(plan.route_submodel_digests),
    }
    backbone = model.featurizer.backbone
    external = {
        "canonical_locator": f"file://{Path(plan.mace_model_path).resolve()}",
        "sha256": plan.mace_sha256,
        "model_id": model.featurizer.model_id,
        "version": model.featurizer.mace_version,
        "model_class": f"{type(backbone).__module__}.{type(backbone).__qualname__}",
        "license": "ASL academic non-commercial",
        "license_acknowledged": True,
        "state_prefixes": ["featurizer.backbone."],
    }
    return model.create_checkpoint_v3(
        config=config,
        external_mace=external,
        metadata={"purpose": "one-epoch wiring smoke"},
    )


def run_pair_smoke_lifecycle(
    model: torch.nn.Module,
    dataset: PairSmokeDataset,
    *,
    output_path: str | Path,
    learning_rate: float,
    include_total_mse: bool = False,
    baseline_name: str | None = None,
    plan: Any | None = None,
) -> PairSmokeReport:
    """Run one pair epoch, persist state, reload, and verify asymptotic ledgers."""

    from apnet_pt.mace.model import MACEAP3D3Model

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("pair smoke lifecycle requires trainable parameters")
    harness = MACEAP3D3Model(model, include_total_mse=include_total_mse)
    optimizer = torch.optim.Adam(trainable, lr=learning_rate)
    loss = harness.fit_epoch(dataset.train_batches, optimizer)
    gradients_finite = _gradients_are_finite(model)
    expected = harness.predict(dataset.test_batches)
    diagnostics = model.last_induction_diagnostics
    if diagnostics is None:
        raise RuntimeError("pair lifecycle did not produce induction diagnostics")
    policy = model.long_range_provider.config.scf_nonconvergence
    diagnostics_record = {
        "converged": diagnostics.converged,
        "iterations": diagnostics.iterations,
        "residual": diagnostics.residual,
        "policy": policy,
    }
    state = _filtered_model_state(model)
    if plan is None:
        checkpoint = {
            "checkpoint_version": 2,
            "model_type": "MACEAP3D3Smoke",
            "model_state_dict": state,
            "config": {
                "architecture": model.architecture,
                "dataset_hash": dataset.content_hash,
                "split_hash": dataset.split_hash,
                "physics_hash": dataset.physics_hash,
                "induction_diagnostics": diagnostics_record,
            },
        }
    else:
        checkpoint = _production_pair_checkpoint(
            model, dataset, plan, diagnostics_record
        )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    Path(f"{output}.diagnostics.json").write_text(
        json.dumps(diagnostics_record, sort_keys=True, indent=2) + "\n"
    )
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if not name.startswith("featurizer.backbone."):
                parameter.zero_()
    restored = torch.load(output, map_location="cpu", weights_only=True)
    _restore_filtered_state(model, restored["model_state_dict"])
    actual = harness.predict(dataset.test_batches)

    details = model(dataset.test_batches[0], return_details=True)
    classical = {
        name: float(value.detach().abs().mean())
        for name, value in details.classical_ledger.items()
    }
    residual = {
        name: float(value.detach().abs().mean())
        for name, value in details.residual_ledger.items()
    }
    long_indices = torch.tensor(dataset.long_range_test_indices, dtype=torch.long)
    if long_indices.numel():
        long_classical = torch.stack(
            [value[long_indices].abs() for value in details.classical_ledger.values()]
        )
        long_residual = details.residual[long_indices]
        long_classical_nonzero = bool((long_classical > 0).any())
        long_residual_zero = bool(torch.allclose(long_residual, torch.zeros_like(long_residual)))
    else:
        long_classical_nonzero = False
        long_residual_zero = False
    return PairSmokeReport(
        epochs=1,
        loss=loss,
        component_losses={
            name: float(value) for name, value in harness.last_component_losses.items()
        },
        gradients_finite=gradients_finite,
        backbone_frozen=_backbone_is_frozen(model),
        reload_equal=torch.equal(expected, actual),
        prediction_shape=tuple(expected.shape),
        classical_ledger=classical,
        residual_ledger=residual,
        long_range_classical_nonzero=long_classical_nonzero,
        long_range_residual_zero=long_residual_zero,
        split_hash=dataset.split_hash,
        physics_hash=dataset.physics_hash,
        baseline_name=baseline_name,
        induction_converged=diagnostics.converged,
        induction_iterations=diagnostics.iterations,
        induction_residual=diagnostics.residual,
        induction_policy=policy,
    )


def run_matched_ap3d3_baseline_smoke(args: Any) -> BaselineSmokeReport:
    """Run the legacy APNet3-fused-d3 baseline on the identical smoke split."""

    if not args.smoke_data_path:
        raise ValueError("matched baseline smoke requires --smoke_data_path")
    if args.long_range_elst != "damped-cliff" or args.d3_params != "default":
        raise ValueError("matched baseline smoke currently requires default damped-cliff physics")
    if args.resume:
        raise ValueError("matched baseline smoke does not support implicit resume")
    if args.n_epochs != 1:
        raise ValueError("matched baseline smoke requires exactly one epoch")
    output = Path(args.ap_model_path)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"output exists: {output}; pass --overwrite")
    for value, label in (
        (args.am_model_path, "am_model_path"),
        (args.atom_type_param_model_path, "atom_type_param_model_path"),
        (args.atom_type_param_model_path2, "atom_type_param_model_path2"),
    ):
        if not value or not Path(value).is_file():
            raise FileNotFoundError(f"matched baseline requires {label}: {value}")

    from apnet_pt.AtomPairwiseModels.apnet3_d3_fused import (
        APNet3D3_AtomType_Model,
    )
    from apnet_pt.AtomPairwiseModels.mtp_mtp import (
        AM_DimerParam_Model,
        AtomTypeParamModel,
    )

    dataset = load_pair_smoke_fixture(
        args.smoke_data_path,
        batch_size=args.batch_size,
        r_cut=args.r_cut,
        r_cut_im=args.r_cut_im,
    )
    atom_type = AtomTypeParamModel(
        ds_root=None,
        use_GPU=False,
        ignore_database_null=True,
        atom_model_pre_trained_path=args.am_model_path,
        pre_trained_model_path=args.atom_type_param_model_path,
        freeze_atom_model=True,
    )
    dimer = AM_DimerParam_Model(
        ds_root=None,
        use_GPU=False,
        ignore_database_null=True,
        atom_model=atom_type.model,
        atom_model_type="AtomTypeParamNN",
        pre_trained_model_path=args.atom_type_param_model_path2,
        freeze_atom_model=True,
    )
    baseline = APNet3D3_AtomType_Model(
        ds_root=None,
        use_GPU=False,
        ignore_database_null=True,
        atom_type_model=atom_type.model,
        dimer_prop_model=dimer.dimer_model,
        am_dimer_param_model=dimer,
        use_precomputed_classical=False,
        freeze_dimer_prop_model=True,
        no_disp_nn=args.no_disp_nn,
        r_cut=args.r_cut,
        r_cut_im=args.r_cut_im,
    )
    model = baseline.model
    model.return_hidden_states = True
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=args.lr)
    component_losses = {}
    epoch_losses = []
    model.train()
    for batch in dataset.train_batches:
        optimizer.zero_grad(set_to_none=True)
        prediction = model(batch)[0]
        if prediction.shape[1] == 3:
            prediction = torch.cat(
                (prediction, prediction.new_zeros((prediction.shape[0], 1))), dim=1
            )
        component_losses = {
            name: (prediction[:, index] - batch.y[:, index]).square().mean()
            for index, name in enumerate(("elst", "exch", "indu", "disp"))
        }
        terms = list(component_losses.values())
        if args.include_total_mse:
            terms.append(
                (prediction.sum(dim=1) - batch.y.sum(dim=1)).square().mean()
            )
        loss = torch.stack(terms).mean()
        if not torch.isfinite(loss):
            raise RuntimeError("matched baseline smoke loss is non-finite")
        loss.backward()
        optimizer.step()
        epoch_losses.append(float(loss.detach()))
    gradients_finite = _gradients_are_finite(model)

    def predict():
        model.eval()
        outputs = []
        with torch.no_grad():
            for batch in dataset.test_batches:
                value = model(batch)[0]
                if value.shape[1] == 3:
                    value = torch.cat(
                        (value, value.new_zeros((value.shape[0], 1))), dim=1
                    )
                outputs.append(value)
        return torch.cat(outputs, dim=0)

    expected = predict()
    state = {
        key: value.detach().cpu().clone() if torch.is_tensor(value) else deepcopy(value)
        for key, value in model.state_dict().items()
    }
    checkpoint = {
        "checkpoint_version": 2,
        "model_type": "APNet3D3_AtomType_MPNN",
        "model_state_dict": state,
        "config": {
            **model.get_config(),
            "dataset_hash": dataset.content_hash,
            "split_hash": dataset.split_hash,
            "physics_hash": dataset.physics_hash,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    restored = torch.load(output, map_location="cpu", weights_only=True)
    model.load_state_dict(restored["model_state_dict"], strict=True)
    actual = predict()
    details = model(dataset.test_batches[0])
    residual_values = details[1]
    long_indices = torch.tensor(dataset.long_range_test_indices, dtype=torch.long)
    classical_values = [value for value in details[2:5] if torch.is_tensor(value)]
    ndimer = dataset.test_batches[0].total_charge_A.numel()
    aggregated_classical = []
    for value in classical_values:
        flat = value.reshape(-1)
        if flat.numel() == dataset.test_batches[0].dimer_ind_full.numel():
            aggregate = flat.new_zeros(ndimer)
            aggregate.index_add_(
                0, dataset.test_batches[0].dimer_ind_full, flat
            )
        elif flat.numel() == ndimer:
            aggregate = flat
        else:
            raise RuntimeError("baseline classical ledger has incompatible shape")
        aggregated_classical.append(aggregate)
    long_classical_nonzero = bool(
        long_indices.numel()
        and aggregated_classical
        and any(
            value[long_indices].detach().abs().sum() > 0
            for value in aggregated_classical
        )
    )
    long_residual_zero = bool(
        long_indices.numel()
        and torch.allclose(
            residual_values[long_indices],
            torch.zeros_like(residual_values[long_indices]),
        )
    )
    classical_names = ("elst", "indu", "disp")
    classical_ledger = {
        name: float(value.detach().abs().mean())
        for name, value in zip(classical_names, aggregated_classical)
    }
    residual_ledger = {
        name: float(residual_values[:, index].detach().abs().mean())
        for index, name in enumerate(("elst", "exch", "indu", "disp"))
    }
    return BaselineSmokeReport(
        epochs=1,
        loss=sum(epoch_losses) / len(epoch_losses),
        component_losses={
            name: float(value.detach()) for name, value in component_losses.items()
        },
        gradients_finite=gradients_finite,
        reload_equal=torch.equal(expected, actual),
        prediction_shape=tuple(expected.shape),
        classical_ledger=classical_ledger,
        residual_ledger=residual_ledger,
        long_range_classical_nonzero=long_classical_nonzero,
        long_range_residual_zero=long_residual_zero,
        split_hash=dataset.split_hash,
        physics_hash=dataset.physics_hash,
    )
