"""TDD regressions for the real prepared-cache producer/consumer seam."""

from argparse import Namespace
import hashlib
import json
from pathlib import Path

import pytest
import torch

from apnet_pt import constants
from apnet_pt.mace.encoder import MACEPolarFeaturizer
from apnet_pt.mace.schema import MACEAtomicFeatures, PhysicsConfig, PolarMACEDirectOutputs
from apnet_pt.training.smoke import load_prepared_feature_cache, load_smoke_fixture_metadata
import scripts.prepare_mace_ap3d3_features as producer
import apnet_pt.mace.encoder as encoder_module
import apnet_pt.training.smoke as smoke_module
from apnet_pt.training.mace_ap3d3_factory import _default_factory_dependencies

DATA = Path(__file__).parent / "dataset_data"


class StubFeaturizer:
    calls = 0

    def __init__(self, _backbone, *, checkpoint_sha256, model_id, feature_mode, dtype, physics_config):
        self.dtype = dtype
        self.feature_mode = feature_mode
        self.checkpoint_sha256 = checkpoint_sha256
        self.model_id = model_id
        self.mace_version = "0.3.16"
        self.physics_config = physics_config
        self.private_adapter = None

    schema_identity = MACEPolarFeaturizer.schema_identity

    def _cache_key(self, positions, numbers, charge, spin):
        return MACEPolarFeaturizer._cache_key(
            self, positions, numbers, charge, spin
        )

    def forward_monomer(self, positions, numbers, charge, spin):
        type(self).calls += 1
        count = numbers.numel()
        width = 2 if self.feature_mode == "final-layer-scalars" else 3
        batch = torch.zeros(count, dtype=torch.long)
        features = MACEAtomicFeatures(
            invariant=torch.arange(count * width, dtype=self.dtype).reshape(count, width),
            equivariant=torch.zeros(count, 0, dtype=self.dtype), batch=batch,
            atomic_numbers=numbers.cpu(), total_charge=charge.cpu(), total_spin=spin.cpu(),
            feature_schema=f"stub:mode={self.feature_mode}:width={width}",
        )
        direct = PolarMACEDirectOutputs(
            density_coefficients=torch.zeros(count, 4, dtype=self.dtype),
            charges=torch.zeros(count, dtype=self.dtype),
            molecular_dipole_eangstrom=torch.zeros(1, 3, dtype=self.dtype),
            positions_angstrom=positions.cpu(), batch=batch,
            total_charge=charge.cpu(),
        )
        return features, direct


def _produce(tmp_path, monkeypatch):
    artifact = tmp_path / "external.model"
    artifact.write_bytes(b"stub artifact")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    monkeypatch.setattr(producer, "load_verified_polar_mace", lambda *args, **kwargs: object())
    monkeypatch.setattr(producer, "MACEPolarFeaturizer", StubFeaturizer)
    monkeypatch.setattr(producer.importlib.metadata, "version", lambda _: "0.3.16")
    args = Namespace(
        mace_path=str(artifact), mace_sha256=digest, mace_model_id="polar-1-s",
        pair_data=str(DATA / "mace_ap3d3_smoke.pkl"),
        atom_data=str(DATA / "mace_atomic_properties_smoke.pkl"),
        cache_dir=str(tmp_path / "cache"), device="cpu", dtype="float32",
    )
    return Path(args.cache_dir), producer.prepare(args), digest


def _open(cache, manifest, digest, kind, mode):
    meta = manifest["dataset_identity"][kind]
    return load_prepared_feature_cache(
        cache, feature_mode=mode, mace_sha256=digest, mace_model_id="polar-1-s",
        physics_hash=PhysicsConfig().physics_hash, dataset_kind=kind,
        dataset_hash=meta["dataset_hash"], preprocessing_hash=meta["preprocessing_hash"],
        split_hash=meta["split_hash"], dtype=torch.float32,
    )


class _NoForwardBackbone(torch.nn.Module):
    atomic_multipoles_max_l = 1

    def __init__(self):
        super().__init__()
        self.register_buffer("atomic_numbers", torch.arange(1, 19))
        self.forward_calls = 0

    def forward(self, *args, **kwargs):
        self.forward_calls += 1
        raise AssertionError("backbone forward must not run on a prepared-cache hit")


@pytest.mark.parametrize("kind", ["pair", "atomic"])
@pytest.mark.parametrize("mode", producer.MODES)
def test_real_featurizer_hits_every_producer_fixture_monomer_without_backbone(
    tmp_path, monkeypatch, kind, mode
):
    cache_dir, manifest, digest = _produce(tmp_path, monkeypatch)
    cache = _open(cache_dir, manifest, digest, kind, mode)
    pair_fixture = producer._load_fixture(
        DATA / "mace_ap3d3_smoke.pkl", producer.PAIR_SMOKE_SCHEMA
    )
    atom_fixture = producer._load_fixture(
        DATA / "mace_atomic_properties_smoke.pkl", producer.ATOMIC_SMOKE_SCHEMA
    )
    monomers, membership = producer._collect_monomers(pair_fixture, atom_fixture)
    backbone = _NoForwardBackbone()
    private_adapter = object() if mode == "all-scalars+norms" else None
    featurizer = MACEPolarFeaturizer(
        backbone,
        checkpoint_sha256=digest,
        model_id="polar-1-s",
        feature_mode=mode,
        dtype=torch.float32,
        physics_config=PhysicsConfig(),
        cache=cache,
        graph_builder=lambda *args: (_ for _ in ()).throw(
            AssertionError("graph construction must not run on a cache hit")
        ),
        private_adapter=private_adapter,
    )
    for key in membership[kind]:
        molecule = monomers[key]
        positions = torch.tensor(
            molecule.geometry * constants.au2ang, dtype=torch.float32
        )
        numbers = torch.tensor(molecule.atomic_numbers, dtype=torch.long)
        charge = torch.tensor([float(molecule.molecular_charge)])
        spin = torch.tensor([float(molecule.molecular_multiplicity)])
        features, _direct = featurizer.forward_monomer(
            positions, numbers, charge, spin
        )
        assert features.natom == len(molecule.atomic_numbers)
    assert cache.loaded_entries == len(membership[kind])
    if kind == "atomic":
        wrong_scope = next(iter(set(membership["pair"]) - set(membership["atomic"])))
        molecule = monomers[wrong_scope]
        with pytest.raises(KeyError, match="prepared feature cache miss"):
            featurizer.forward_monomer(
                torch.tensor(molecule.geometry * constants.au2ang, dtype=torch.float32),
                torch.tensor(molecule.atomic_numbers, dtype=torch.long),
                torch.tensor([float(molecule.molecular_charge)]),
                torch.tensor([float(molecule.molecular_multiplicity)]),
            )
    assert backbone.forward_calls == 0


def test_real_stubbed_producer_feeds_pair_and_atomic_consumers(tmp_path, monkeypatch):
    cache, manifest, digest = _produce(tmp_path, monkeypatch)
    assert manifest["cache_format"].endswith("v2")
    assert manifest["dataset_counts"]["unique_monomers"] == 10
    assert manifest["entry_count"] == 20
    producer_calls = StubFeaturizer.calls
    pair = _open(cache, manifest, digest, "pair", "final-layer-scalars")
    atomic = _open(cache, manifest, digest, "atomic", "all-scalars+norms")
    pair_value = pair[next(iter(pair))]
    atomic_value = atomic[next(iter(atomic))]
    assert pair_value[0].invariant.shape[1] == 2
    assert atomic_value[0].invariant.shape[1] == 3
    assert StubFeaturizer.calls == producer_calls  # consumer stayed offline


@pytest.mark.parametrize("field", ["dataset_hash", "preprocessing_hash", "split_hash"])
def test_scoped_identity_invalidation_is_fail_closed(tmp_path, monkeypatch, field):
    cache, manifest, digest = _produce(tmp_path, monkeypatch)
    identity = {
        name: manifest["dataset_identity"]["pair"][name]
        for name in ("dataset_hash", "preprocessing_hash", "split_hash")
    }
    identity[field] = "f" * 64
    with pytest.raises(RuntimeError, match=field):
        load_prepared_feature_cache(
            cache, feature_mode="final-layer-scalars", mace_sha256=digest,
            mace_model_id="polar-1-s", physics_hash=PhysicsConfig().physics_hash,
            dataset_kind="pair", dtype=torch.float32, **identity,
        )


def test_completeness_digest_and_offline_miss_are_strict(tmp_path, monkeypatch):
    cache, manifest, digest = _produce(tmp_path, monkeypatch)
    loaded = _open(cache, manifest, digest, "pair", "final-layer-scalars")
    with pytest.raises(KeyError, match="prepared feature cache miss"):
        loaded["absent"]
    first = cache / manifest["entries"][0]["path"]
    first.write_bytes(first.read_bytes() + b"corrupt")
    with pytest.raises(RuntimeError, match="corrupt"):
        _open(cache, manifest, digest, "pair", "final-layer-scalars")


def _rewrite_manifest(cache, manifest):
    (cache / "COMPLETE.json").write_text(json.dumps(manifest))


def test_v2_selected_scope_membership_rejects_pair_removal_but_allows_overlap(
    tmp_path, monkeypatch
):
    cache, manifest, digest = _produce(tmp_path, monkeypatch)
    pair = set(manifest["dataset_identity"]["pair"]["monomer_hashes"])
    atomic = set(manifest["dataset_identity"]["atomic"]["monomer_hashes"])
    assert pair & atomic  # overlap is represented once in the union cache
    victim = next(iter(pair - atomic))
    removed = [entry for entry in manifest["entries"] if entry["monomer_hash"] == victim]
    assert {entry["feature_mode"] for entry in removed} == set(producer.MODES)
    for entry in removed:
        (cache / entry["path"]).unlink()
    manifest["entries"] = [
        entry for entry in manifest["entries"] if entry["monomer_hash"] != victim
    ]
    manifest["entry_count"] = len(manifest["entries"])
    _rewrite_manifest(cache, manifest)
    with pytest.raises(RuntimeError, match="pair membership is incomplete"):
        _open(cache, manifest, digest, "pair", "final-layer-scalars")
    atomic_cache = _open(cache, manifest, digest, "atomic", "final-layer-scalars")
    assert len(atomic_cache) == len(atomic)


def test_completed_restart_rejects_changed_model_and_mutated_record(
    tmp_path, monkeypatch
):
    cache, manifest, _digest = _produce(tmp_path, monkeypatch)
    artifact = tmp_path / "external.model"
    args = Namespace(
        mace_path=str(artifact), mace_sha256=manifest["mace_sha256"],
        mace_model_id="different-model",
        pair_data=str(DATA / "mace_ap3d3_smoke.pkl"),
        atom_data=str(DATA / "mace_atomic_properties_smoke.pkl"),
        cache_dir=str(cache), device="cpu", dtype="float32",
    )
    with pytest.raises(RuntimeError, match="mace_model_id"):
        producer.prepare(args)

    args.mace_model_id = "polar-1-s"
    entry = manifest["entries"][0]
    path = cache / entry["path"]
    record = torch.load(path, map_location="cpu", weights_only=True)
    record["identity"]["cache_key"] = "self-consistent-but-wrong"
    torch.save(record, path)
    entry["sha256"] = producer.sha256_file(path)
    _rewrite_manifest(cache, manifest)
    with pytest.raises(RuntimeError, match="identity is stale"):
        producer.prepare(args)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda tensors: tensors.__setitem__("invariant", tensors["invariant"].double()),
        lambda tensors: tensors.__setitem__("batch", tensors["batch"].float()),
        lambda tensors: tensors.__setitem__("atomic_numbers", tensors["atomic_numbers"].float() + 0.5),
        lambda tensors: tensors.__setitem__("density_coefficients", tensors["density_coefficients"][:, :3]),
        lambda tensors: tensors.__setitem__("positions_angstrom", tensors["positions_angstrom"][:, :2]),
        lambda tensors: tensors.__setitem__("molecular_dipole_eangstrom", torch.zeros(3)),
        lambda tensors: tensors.__setitem__("total_spin", torch.ones(2)),
        lambda tensors: tensors.__setitem__("batch", torch.ones_like(tensors["batch"])),
    ],
)
def test_adversarial_payload_contracts_are_rejected(tmp_path, monkeypatch, mutation):
    cache, manifest, digest = _produce(tmp_path, monkeypatch)
    pair_members = set(manifest["dataset_identity"]["pair"]["monomer_hashes"])
    entry = next(
        item for item in manifest["entries"]
        if item["feature_mode"] == "final-layer-scalars"
        and item["monomer_hash"] in pair_members
    )
    path = cache / entry["path"]
    record = torch.load(path, map_location="cpu", weights_only=True)
    mutation(record["tensors"])
    torch.save(record, path)
    entry["sha256"] = producer.sha256_file(path)
    _rewrite_manifest(cache, manifest)
    with pytest.raises(RuntimeError, match="tensor"):
        _open(cache, manifest, digest, "pair", "final-layer-scalars")


@pytest.mark.parametrize("kind", ["pair", "atomic"])
def test_default_factory_forwards_prepared_cache_scope(kind, tmp_path, monkeypatch):
    captured = {}

    class FactoryFeaturizer:
        def __init__(self, backbone, **kwargs):
            captured["backbone"] = backbone
            captured["featurizer"] = kwargs

    monkeypatch.setattr(encoder_module, "load_verified_polar_mace", lambda *a, **k: "backbone")
    monkeypatch.setattr(encoder_module, "MACEPolarFeaturizer", FactoryFeaturizer)
    monkeypatch.setattr(
        smoke_module, "load_prepared_feature_cache",
        lambda path, **kwargs: captured.update(path=path, cache=kwargs) or "prepared",
    )
    plan = Namespace(
        kind=kind, mace_model_path=str(tmp_path / "model"), mace_sha256="a" * 64,
        mace_offline=True, mace_default_dtype="float32", device="cpu",
        mace_cache_dir=str(tmp_path / "cache"), feature_mode="final-layer-scalars",
        mace_model="polar-1-s", data_hash="b" * 64,
        preprocessing_hash="c" * 64, split_hash="d" * 64,
        long_range_elst="damped-cliff", d3_parameters=(), neural_cutoff=8.0,
    )
    featurizer = _default_factory_dependencies(plan).featurizer_builder(plan)
    assert captured["path"] == plan.mace_cache_dir
    assert captured["cache"]["dataset_kind"] == kind
    assert captured["cache"]["mace_model_id"] == plan.mace_model
    assert captured["cache"]["dataset_hash"] == plan.data_hash
    assert captured["cache"]["preprocessing_hash"] == plan.preprocessing_hash
    assert captured["cache"]["split_hash"] == plan.split_hash
    assert captured["featurizer"]["cache"] == "prepared"
    assert featurizer.cache_dir == plan.mace_cache_dir


def test_missing_identity_values_never_match(tmp_path, monkeypatch):
    cache, manifest, digest = _produce(tmp_path, monkeypatch)
    identity = {
        name: manifest["dataset_identity"]["pair"][name]
        for name in ("dataset_hash", "preprocessing_hash", "split_hash")
    }
    identity["dataset_hash"] = ""
    with pytest.raises(ValueError, match="explicit dataset_hash"):
        load_prepared_feature_cache(
            cache, feature_mode="final-layer-scalars", mace_sha256=digest,
            mace_model_id="polar-1-s", physics_hash=PhysicsConfig().physics_hash,
            dataset_kind="pair", dtype=torch.float32, **identity,
        )
