"""Production/reproducibility contracts from final review pass B."""

from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

import train_models
from apnet_pt.mace.encoder import MACEPolarFeaturizer
from apnet_pt.mace.properties import MACEAtomPropertyModel
from apnet_pt.training.mace_ap3d3_factory import (
    MACEFactoryDependencies,
    dispatch_mace_cli,
    validate_atomic_property_checkpoint,
    validate_mace_cli_args,
)
from apnet_pt.training.smoke import (
    PreparedFeatureCache,
    load_atomic_smoke_fixture,
    run_atomic_smoke_lifecycle,
)
from tests.test_mace_ap3d3_cli import _base_cli
from tests.test_mace_atomic_properties import _heads
from tests.test_mace_model_harness import StubFeaturizer, StubPropertyProvider, _make_model


DATA = Path(__file__).parent / "dataset_data"


def _deps(model, *, dataset=None, lifecycle=None):
    return MACEFactoryDependencies(
        featurizer_builder=lambda plan: model.featurizer,
        property_provider_builder=lambda plan, featurizer: model.property_provider,
        pair_core_builder=lambda plan: model.pair_core,
        long_range_builder=lambda plan: model.long_range_provider,
        model_builder=lambda plan, *unused: model,
        dataset_builder=lambda plan: dataset,
        lifecycle_runner=(lambda plan, harness, value: lifecycle),
    )


def test_dispatch_rejects_absent_dataset_and_absent_lifecycle(tmp_path):
    args, _ = _base_cli(tmp_path, "MACE-AP3D3-H1")
    args.skip_compile = True
    model, _, _ = _make_model("hybrid-h1")
    with pytest.raises(RuntimeError, match="dataset.*absent|verification dataset"):
        dispatch_mace_cli(args, dependencies=_deps(model, dataset=None, lifecycle=None))
    with pytest.raises(RuntimeError, match="lifecycle.*absent|no-op"):
        dispatch_mace_cli(
            args,
            dependencies=_deps(model, dataset=object(), lifecycle=None),
        )


def test_resume_and_uncompiled_mace_are_rejected_before_build(tmp_path):
    args, _ = _base_cli(tmp_path, "MACE-AP3D3-H1")
    calls = []
    args.resume = True
    with pytest.raises(ValueError, match="resume.*not implemented"):
        dispatch_mace_cli(
            args,
            dependencies=MACEFactoryDependencies(
                featurizer_builder=lambda plan: calls.append("model")
            ),
        )
    assert calls == []

    args.resume = False
    args.skip_compile = False
    with pytest.raises(ValueError, match="skip_compile|compile.*unsupported"):
        dispatch_mace_cli(
            args,
            dependencies=MACEFactoryDependencies(
                featurizer_builder=lambda plan: calls.append("model")
            ),
        )
    assert calls == []


def test_device_policy_is_explicit_and_validated(tmp_path, monkeypatch):
    args, _ = _base_cli(tmp_path, "MACE-AP3D3-H1")
    args.skip_compile = True
    args.mace_device = "cpu"
    assert validate_mace_cli_args(args).device == "cpu"
    args.mace_device = "auto"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert validate_mace_cli_args(args).device == "cpu"
    args.mace_device = "cuda"
    with pytest.raises(RuntimeError, match="CUDA.*unavailable"):
        validate_mace_cli_args(args)
    args.mace_device = "tpu"
    with pytest.raises(ValueError, match="mace_device"):
        validate_mace_cli_args(args)


def _atomic_metadata(model, dataset, mode="learned"):
    return {
        "checkpoint_version": 3,
        "model_type": "MACEAtomicProperties",
        "model_state_dict": model.property_provider.state_dict(),
        "config": {
            "property_mode": mode,
            "provider_kind": "atomhead" if mode == "learned" else "direct",
            "mace": {
                "sha256": "a" * 64,
                "version": "0.3.16",
                "model_class": "stub.Backbone",
                "feature_schema": "stub:mode=all-scalars+norms:inv=16:equiv=0",
                "feature_mode": "all-scalars+norms",
            },
            "dtype_policy": "float32",
            "atomic_property_schema": "ap3-atomic-properties-cartesian-v1",
            "quadrupole_convention": "cartesian-symmetric-traceless-3x3",
            "physics_hash": "b" * 64,
            "data": {
                "dataset_hash": dataset.content_hash,
                "preprocessing_hash": dataset.preprocessing_hash,
                "split_hash": dataset.split_hash,
            },
        },
    }


def test_atomic_checkpoint_wrong_route_digest_and_schema_rejected(tmp_path):
    dataset = load_atomic_smoke_fixture(DATA / "mace_atomic_properties_smoke.pkl")
    model = MACEAtomPropertyModel(_heads())
    record = _atomic_metadata(model, dataset)
    expectations = deepcopy(record["config"])
    validate_atomic_property_checkpoint(record, expectations)
    for path, value in (
        (("property_mode",), "direct-completion"),
        (("mace", "sha256"), "c" * 64),
        (("mace", "feature_schema"), "other-schema"),
    ):
        bad = deepcopy(record)
        target = bad["config"]
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        with pytest.raises(ValueError, match="atomic checkpoint semantic mismatch"):
            validate_atomic_property_checkpoint(bad, expectations)


def test_atomic_lifecycle_persists_complete_versioned_metadata(tmp_path):
    dataset = load_atomic_smoke_fixture(DATA / "mace_atomic_properties_smoke.pkl")
    featurizer = StubFeaturizer("all-scalars+norms")
    featurizer.checkpoint_sha256 = "a" * 64
    featurizer.mace_version = "0.3.16"
    featurizer.resolved_feature_schema = (
        "stub:mode=all-scalars+norms:inv=16:equiv=0"
    )
    model = __import__(
        "apnet_pt.mace.model", fromlist=["MACEAtomicPropertiesModel"]
    ).MACEAtomicPropertiesModel(
        property_mode="learned",
        featurizer=featurizer,
        property_provider=StubPropertyProvider("atomhead"),
    )
    output = tmp_path / "atomic.pt"
    run_atomic_smoke_lifecycle(
        model,
        dataset,
        output_path=output,
        learning_rate=1.0e-3,
        physics_hash="b" * 64,
    )
    checkpoint = torch.load(output, map_location="cpu", weights_only=True)
    assert checkpoint["checkpoint_version"] == 3
    expected = checkpoint["config"]
    validate_atomic_property_checkpoint(checkpoint, expected)
    assert set(expected["data"]) == {
        "dataset_hash", "preprocessing_hash", "split_hash"
    }


def test_all_remaining_trainable_pair_parameters_have_gradient_and_delta():
    from apnet_pt.training.smoke import load_pair_smoke_fixture

    dataset = load_pair_smoke_fixture(DATA / "mace_ap3d3_smoke.pkl")
    for route in ("direct-polar", "hybrid-h1", "hybrid-h2", "atomhead"):
        torch.manual_seed(123)
        model, _, _ = _make_model(route)
        trainable = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad], lr=1.0e-3
        )
        from apnet_pt.mace.model import MACEAP3D3Model

        MACEAP3D3Model(model).train_step(dataset.train_batches[0], optimizer)
        for name, before in trainable.items():
            parameter = dict(model.named_parameters())[name]
            assert parameter.grad is not None, (route, name)
            assert torch.isfinite(parameter.grad).all(), (route, name)
            assert not torch.equal(before, parameter.detach()), (route, name)
        assert not model.pair_core.ap3_core.embed_layer.weight.requires_grad
        if model.pair_core.pair_mode == "h2":
            assert all(
                not parameter.requires_grad
                for module in (
                    model.pair_core.ap3_core.update_layers,
                    model.pair_core.ap3_core.directional_layers,
                )
                for parameter in module.parameters()
            )


def test_prepared_cache_is_lazy_read_only_strict_and_dataset_bound(tmp_path):
    entry = tmp_path / "entry.pt"
    torch.save({"identity": {}, "tensors": {}}, entry)
    manifest = {
        "status": "complete",
        "mace_sha256": "a" * 64,
        "physics_hash": "b" * 64,
        "dtype": "float32",
        "dataset_hash": "c" * 64,
        "preprocessing_hash": "d" * 64,
        "split_hash": "e" * 64,
        "entries": [{
            "feature_mode": "final-layer-scalars",
            "cache_key": "key",
            "path": entry.name,
            "sha256": __import__("hashlib").sha256(entry.read_bytes()).hexdigest(),
        }],
    }
    (tmp_path / "COMPLETE.json").write_text(json.dumps(manifest))
    cache = PreparedFeatureCache(
        tmp_path,
        feature_mode="final-layer-scalars",
        mace_sha256="a" * 64,
        physics_hash="b" * 64,
        dataset_hash="c" * 64,
        preprocessing_hash="d" * 64,
        split_hash="e" * 64,
        dtype=torch.float32,
    )
    assert cache.loaded_entries == 0
    with pytest.raises(KeyError, match="prepared feature cache miss"):
        cache["missing"]
    with pytest.raises(TypeError, match="read-only"):
        cache["key"] = object()
    with pytest.raises(RuntimeError, match="dataset_hash"):
        PreparedFeatureCache(
            tmp_path,
            feature_mode="final-layer-scalars",
            mace_sha256="a" * 64,
            physics_hash="b" * 64,
            dataset_hash="f" * 64,
            preprocessing_hash="d" * 64,
            split_hash="e" * 64,
            dtype=torch.float32,
        )
