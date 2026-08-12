#!/usr/bin/env python3
"""Restartable, digest-verified monomer feature-cache preparation."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from typing import Any, Mapping

import qcelemental as qcel
import torch

from apnet_pt import constants
from apnet_pt.mace.encoder import MACEPolarFeaturizer, load_verified_polar_mace
from apnet_pt.mace.schema import PhysicsConfig
from apnet_pt.training.smoke import (
    ATOMIC_SMOKE_SCHEMA,
    PAIR_SMOKE_SCHEMA,
    fixture_content_hash,
)


CACHE_FORMAT = "qcmlforge-mace-monomer-cache-v1"
MODES = ("final-layer-scalars", "all-scalars+norms")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _load_fixture(path: Path, schema: str) -> Mapping[str, Any]:
    import pickle

    with path.open("rb") as handle:
        fixture = pickle.load(handle)
    if fixture.get("schema") != schema:
        raise ValueError(f"fixture {path} does not use schema {schema}")
    if fixture.get("content_hash") != fixture_content_hash(fixture):
        raise ValueError(f"fixture {path} content hash is invalid")
    return fixture


def _molecule_identity(molecule: qcel.models.Molecule) -> str:
    return _canonical_hash(json.loads(molecule.json()))


def _primitive_molecule(record) -> qcel.models.Molecule:
    if (
        not isinstance(record, dict)
        or record.get("format") != "qcel-psi4-text-v1"
        or not isinstance(record.get("data"), str)
    ):
        raise ValueError("feature preparation requires primitive molecule text")
    return qcel.models.Molecule.from_data(record["data"])


def _collect_monomers(pair_fixture, atom_fixture):
    monomers: dict[str, qcel.models.Molecule] = {}
    for record in pair_fixture["records"]:
        dimer = _primitive_molecule(record["dimer"])
        for fragment in (0, 1):
            molecule = dimer.get_fragment(fragment)
            monomers.setdefault(_molecule_identity(molecule), molecule)
    for record in atom_fixture["records"]:
        molecule = _primitive_molecule(record["monomer"])
        monomers.setdefault(_molecule_identity(molecule), molecule)
    return monomers


def _entry_identity(key, cache_key, mode, args, physics_hash):
    return {
        "format": CACHE_FORMAT,
        "monomer_hash": key,
        "cache_key": cache_key,
        "feature_mode": mode,
        "mace_sha256": args.mace_sha256,
        "mace_model_id": args.mace_model_id,
        "physics_hash": physics_hash,
        "dtype": args.dtype,
    }


def _validate_entry(path: Path, expected_identity: Mapping[str, Any]) -> bool:
    try:
        existing = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise RuntimeError(f"existing cache entry is unreadable: {path}: {exc}") from exc
    if existing.get("identity") != dict(expected_identity):
        raise RuntimeError(f"existing cache entry identity is stale: {path}")
    tensors = existing.get("tensors")
    if not isinstance(tensors, Mapping) or not tensors:
        raise RuntimeError(f"existing cache entry has no tensors: {path}")
    if not all(
        torch.is_tensor(value) and torch.isfinite(value).all()
        for value in tensors.values()
    ):
        raise RuntimeError(f"existing cache entry contains invalid tensors: {path}")
    return True


def _atomic_torch_save(record: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary_path = Path(temporary)
    try:
        torch.save(dict(record), temporary_path)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _atomic_json_save(record: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary_path = Path(temporary)
    try:
        temporary_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _source_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _validate_complete_manifest(cache_dir: Path, expected: Mapping[str, Any]) -> bool:
    complete = cache_dir / "COMPLETE.json"
    if not complete.is_file():
        return False
    manifest = json.loads(complete.read_text())
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise RuntimeError(f"complete cache manifest mismatch for {key}")
    entries = manifest.get("entries", [])
    if manifest.get("entry_count") != len(entries):
        raise RuntimeError("complete cache manifest entry count is inconsistent")
    listed = set()
    for entry in entries:
        path = cache_dir / entry["path"]
        listed.add(path.resolve())
        if not path.is_file() or sha256_file(path) != entry["sha256"]:
            raise RuntimeError(f"complete cache entry is missing or corrupt: {path}")
    actual = {path.resolve() for path in cache_dir.glob("*/*.pt")}
    if actual != listed:
        raise RuntimeError("complete cache contains unlisted or missing entries")
    print(f"reusing {len(entries)} existing valid cache entries")
    return True


def _feature_record(featurizer, molecule, key, identity, device):
    positions = torch.as_tensor(
        molecule.geometry * constants.au2ang,
        dtype=featurizer.dtype,
        device=device,
    )
    numbers = torch.as_tensor(molecule.atomic_numbers, dtype=torch.long, device=device)
    charge = torch.tensor(
        [float(molecule.molecular_charge)], dtype=featurizer.dtype, device=device
    )
    spin = torch.tensor(
        [float(molecule.molecular_multiplicity)],
        dtype=featurizer.dtype,
        device=device,
    )
    features, direct = featurizer.forward_monomer(positions, numbers, charge, spin)
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
    return {
        "identity": dict(identity),
        "feature_schema": features.feature_schema,
        "symbols": [str(symbol) for symbol in molecule.symbols],
        "tensors": tensors,
    }


def prepare(args) -> Mapping[str, Any]:
    artifact = Path(args.mace_path)
    if not artifact.is_file():
        raise FileNotFoundError(f"MACE artifact is missing: {artifact}")
    actual_digest = sha256_file(artifact)
    if actual_digest != args.mace_sha256:
        raise ValueError(
            f"MACE SHA-256 mismatch: expected {args.mace_sha256}, got {actual_digest}"
        )
    pair_path = Path(args.pair_data)
    atom_path = Path(args.atom_data)
    pair_fixture = _load_fixture(pair_path, PAIR_SMOKE_SCHEMA)
    atom_fixture = _load_fixture(atom_path, ATOMIC_SMOKE_SCHEMA)
    monomers = _collect_monomers(pair_fixture, atom_fixture)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    physics = PhysicsConfig()
    dataset_identity = {
        "pair_content_hash": pair_fixture["content_hash"],
        "pair_split_hash": pair_fixture["split_hash"],
        "pair_preprocessing_hash": pair_fixture["preprocessing_hash"],
        "atomic_content_hash": atom_fixture["content_hash"],
        "atomic_split_hash": atom_fixture["split_hash"],
        "atomic_preprocessing_hash": atom_fixture["preprocessing_hash"],
    }
    expected_manifest = {
        "status": "complete",
        "cache_format": CACHE_FORMAT,
        "mace_sha256": args.mace_sha256,
        "dataset_identity": dataset_identity,
        "physics_hash": physics.physics_hash,
        "dtype": args.dtype,
    }
    if _validate_complete_manifest(cache_dir, expected_manifest):
        return json.loads((cache_dir / "COMPLETE.json").read_text())

    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    device = torch.device(args.device)
    fail_after = int(os.environ.get("QCMLFORGE_PREPARE_FAIL_AFTER", "0"))
    entries = []
    feature_schemas = {}
    created = 0
    for mode in MODES:
        backbone = load_verified_polar_mace(
            artifact,
            expected_sha256=args.mace_sha256,
            device=device,
            offline=True,
        )
        featurizer = MACEPolarFeaturizer(
            backbone,
            checkpoint_sha256=args.mace_sha256,
            model_id=args.mace_model_id,
            feature_mode=mode,
            dtype=dtype,
            physics_config=physics,
        )
        for key, molecule in monomers.items():
            positions = torch.as_tensor(
                molecule.geometry * constants.au2ang,
                dtype=dtype,
                device=device,
            )
            numbers = torch.as_tensor(
                molecule.atomic_numbers, dtype=torch.long, device=device
            )
            charge = torch.tensor(
                [float(molecule.molecular_charge)], dtype=dtype, device=device
            )
            spin = torch.tensor(
                [float(molecule.molecular_multiplicity)], dtype=dtype, device=device
            )
            cache_key = featurizer._cache_key(positions, numbers, charge, spin)
            identity = _entry_identity(
                key, cache_key, mode, args, physics.physics_hash
            )
            relative = Path(mode) / f"{cache_key}.pt"
            destination = cache_dir / relative
            if destination.is_file():
                _validate_entry(destination, identity)
                action = "skip existing valid entry"
            else:
                record = _feature_record(featurizer, molecule, key, identity, device)
                _atomic_torch_save(record, destination)
                created += 1
                action = "created"
                if fail_after and created >= fail_after:
                    raise RuntimeError("deliberate partial-cache test failure")
            loaded = torch.load(destination, map_location="cpu", weights_only=True)
            feature_schemas[mode] = loaded["feature_schema"]
            entries.append(
                {
                    "path": str(relative),
                    "sha256": sha256_file(destination),
                    "monomer_hash": key,
                    "cache_key": cache_key,
                    "feature_mode": mode,
                    "action": action,
                }
            )
        del featurizer, backbone
        if device.type == "cuda":
            torch.cuda.empty_cache()

    expected_count = len(monomers) * len(MODES)
    if len(entries) != expected_count:
        raise RuntimeError(
            f"partial cache generation: expected {expected_count}, got {len(entries)}"
        )
    environment = {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": str(device),
    }
    manifest = {
        **expected_manifest,
        "source_commit": _source_commit(),
        "environment": environment,
        "mace_version": importlib.metadata.version("mace-torch"),
        "mace_model_id": args.mace_model_id,
        "mace_path": str(artifact.resolve()),
        "feature_schemas": feature_schemas,
        "dataset_counts": {
            "dimers": len(pair_fixture["records"]),
            "atomic_monomers": len(atom_fixture["records"]),
            "unique_monomers": len(monomers),
            "feature_modes": len(MODES),
        },
        "entry_count": len(entries),
        "entries": entries,
        "created_entries": created,
    }
    _atomic_json_save(manifest, cache_dir / "COMPLETE.json")
    return manifest


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mace-path", required=True)
    parser.add_argument("--mace-sha256", required=True)
    parser.add_argument("--mace-model-id", default="polar-1-s")
    parser.add_argument("--pair-data", required=True)
    parser.add_argument("--atom-data", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    return parser


def main(argv=None) -> int:
    manifest = prepare(build_parser().parse_args(argv))
    print(json.dumps({
        "status": manifest["status"],
        "entry_count": manifest["entry_count"],
        "mace_sha256": manifest["mace_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
