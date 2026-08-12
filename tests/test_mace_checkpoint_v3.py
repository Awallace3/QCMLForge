from copy import deepcopy
from pathlib import Path

import pytest
import torch

from apnet_pt import model_io
from apnet_pt.mace.model import MACEAP3D3
from apnet_pt.mace.schema import PhysicsConfig
from tests.test_mace_model_harness import _augment_batch, _batch, _make_model


class DeterministicBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.foundation_table = torch.nn.Parameter(torch.arange(200_000.0))


class WrongBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.foundation_table = torch.nn.Parameter(torch.zeros(1))


def _save_external_artifact(path):
    backbone = DeterministicBackbone()
    torch.save(backbone.state_dict(), path)
    return backbone, model_io.sha256_file(path)


def _external_loader(calls, backbone_type=DeterministicBackbone):
    def load(path, *, map_location="cpu"):
        calls.append(Path(path))
        backbone = backbone_type()
        state = torch.load(path, map_location=map_location, weights_only=True)
        if backbone_type is DeterministicBackbone:
            backbone.load_state_dict(state)
        return backbone

    return load


def _configure_model(backbone, route="hybrid-h1"):
    model, _, _ = _make_model(route)
    model.featurizer.backbone = backbone
    backbone.requires_grad_(False)
    backbone.eval()
    model.featurizer.checkpoint_sha256 = None
    model.featurizer.model_id = "polar-1-s-test"
    model.featurizer.mace_version = "0.3.16"
    model.featurizer.dtype = torch.float32
    model.featurizer.resolved_feature_schema = (
        f"stub:mace=0.3.16:mode={model.featurizer.feature_mode}:"
        "adapter=stub:inv=16:equiv=0:layers=4"
    )
    # This name contains 'backbone' but is intentionally not external.
    model.property_provider.backbone_decoy = torch.nn.Linear(1, 1)
    return model


def _config(artifact_sha, route="hybrid-h1"):
    pair_mode = "h2" if route == "hybrid-h2" else "h1"
    feature_mode = (
        "final-layer-scalars" if route == "hybrid-h1" else "all-scalars+norms"
    )
    physics = PhysicsConfig()
    return {
        "architecture": route,
        "mace": {
            "model_id": "polar-1-s-test",
            "version": "0.3.16",
            "sha256": artifact_sha,
            "feature_schema": (
                f"stub:mace=0.3.16:mode={feature_mode}:"
                "adapter=stub:inv=16:equiv=0:layers=4"
            ),
            "feature_mode": feature_mode,
        },
        "pair_mode": pair_mode,
        "dtype_policy": "float32",
        "atomic_property_schema": "ap3-atomic-properties-cartesian-v1",
        "physics": {
            "electrostatics_mode": physics.electrostatics_mode,
            "induction_mode": "thole-scf",
            "dispersion_mode": "d3",
            "d3_parameters": physics.d3_parameters,
            "component_order": physics.component_order,
            "length_unit": physics.length_unit,
            "energy_unit": physics.energy_unit,
            "neural_cutoff": physics.neural_cutoff,
            "physics_hash": physics.physics_hash,
        },
        "data": {
            "dataset_hash": "1" * 64,
            "preprocessing_hash": "2" * 64,
            "split_hash": "3" * 64,
        },
        "seed": 17,
        "source_commit": "a" * 40,
        "route_submodel_digests": {
            "legacy_atom_model": "4" * 64,
            "legacy_parameter_model": "5" * 64,
        },
    }


def _external_metadata(artifact_sha):
    return {
        "canonical_locator": "mace-foundations://MACE-POLAR-1-S.model",
        "sha256": artifact_sha,
        "model_id": "polar-1-s-test",
        "version": "0.3.16",
        "model_class": (
            "tests.test_mace_checkpoint_v3.DeterministicBackbone"
        ),
        "license": "ASL",
        "license_acknowledged": True,
        "state_prefixes": ["featurizer.backbone."],
    }


def _factory(config, backbone):
    model = _configure_model(backbone, config["architecture"])
    model.featurizer.checkpoint_sha256 = config["mace"]["sha256"]
    return model


def _initialized_model(tmp_path, route="hybrid-h1"):
    artifact_path = tmp_path / "foundation.model"
    backbone, digest = _save_external_artifact(artifact_path)
    model = _configure_model(backbone, route)
    model.featurizer.checkpoint_sha256 = digest
    batch = _augment_batch(_batch())
    prediction = model(batch).detach()
    return model, batch, prediction, artifact_path, digest


def test_v3_exact_prefix_filtering_and_required_records(tmp_path):
    model, _, _, _, digest = _initialized_model(tmp_path)
    checkpoint = model.create_checkpoint_v3(
        config=_config(digest),
        external_mace=_external_metadata(digest),
        metadata={"purpose": "unit-test"},
    )

    assert checkpoint["checkpoint_version"] == 3
    assert checkpoint["model_type"] == "MACEAP3D3"
    assert model_io.validate_checkpoint(checkpoint, expected_type="MACEAP3D3")
    assert checkpoint["architecture"] == "hybrid-h1"
    assert set(checkpoint["external_submodels"]) == {"mace"}
    assert checkpoint["external_submodels"]["mace"]["sha256"] == digest
    assert checkpoint["metadata"]["purpose"] == "unit-test"
    state = checkpoint["model_state_dict"]
    assert not any(key.startswith("featurizer.backbone.") for key in state)
    assert "property_provider.backbone_decoy.weight" in state
    assert not any(
        torch.is_tensor(value) and value.numel() >= 200_000
        for value in state.values()
    )
    required_config = {
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
    }
    assert set(checkpoint["config"]) == required_config
    assert checkpoint["config"]["parameter_counts"]["external"] == 200_000
    assert checkpoint["config"]["physics"]["component_order"] == (
        "elst",
        "exch",
        "indu",
        "disp",
    )


def test_v3_prediction_equality_after_external_reconstruction(tmp_path):
    model, batch, expected, artifact_path, digest = _initialized_model(tmp_path)
    checkpoint_path = tmp_path / "mace-ap3d3.pt"
    model.save_checkpoint_v3(
        checkpoint_path,
        config=_config(digest),
        external_mace=_external_metadata(digest),
    )
    calls = []
    restored = MACEAP3D3.load_checkpoint_v3(
        checkpoint_path,
        mace_artifact_path=artifact_path,
        model_factory=_factory,
        backbone_loader=_external_loader(calls),
        semantic_expectations={
            "architecture": "hybrid-h1",
            "mace": {
                "feature_schema": _config(digest)["mace"]["feature_schema"],
                "feature_mode": "final-layer-scalars",
            },
            "dtype_policy": "float32",
            "physics": {"physics_hash": _config(digest)["physics"]["physics_hash"]},
            "data": {
                "dataset_hash": "1" * 64,
                "split_hash": "3" * 64,
            },
            "route_submodel_digests": _config(digest)[
                "route_submodel_digests"
            ],
        },
    )
    actual = restored(batch).detach()
    assert calls == [artifact_path]
    assert torch.equal(actual, expected)
    assert all(
        not parameter.requires_grad
        for parameter in restored.featurizer.backbone.parameters()
    )


def test_v3_digest_path_and_class_fail_before_reconstruction(tmp_path):
    model, _, _, artifact_path, digest = _initialized_model(tmp_path)
    checkpoint_path = tmp_path / "model.pt"
    model.save_checkpoint_v3(
        checkpoint_path,
        config=_config(digest),
        external_mace=_external_metadata(digest),
    )

    calls = []
    with pytest.raises(FileNotFoundError, match="external MACE artifact"):
        MACEAP3D3.load_checkpoint_v3(
            checkpoint_path,
            mace_artifact_path=tmp_path / "missing.model",
            model_factory=_factory,
            backbone_loader=_external_loader(calls),
        )
    assert calls == []

    artifact_path.write_bytes(artifact_path.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="SHA-256"):
        MACEAP3D3.load_checkpoint_v3(
            checkpoint_path,
            mace_artifact_path=artifact_path,
            model_factory=_factory,
            backbone_loader=_external_loader(calls),
        )
    assert calls == []

    _, clean_digest = _save_external_artifact(artifact_path)
    assert clean_digest == digest
    with pytest.raises(TypeError, match="model class"):
        MACEAP3D3.load_checkpoint_v3(
            checkpoint_path,
            mace_artifact_path=artifact_path,
            model_factory=_factory,
            backbone_loader=_external_loader(calls, WrongBackbone),
        )
    assert calls == [artifact_path]


@pytest.mark.parametrize("mutation", ["missing", "unexpected"])
def test_v3_state_loading_is_strict_outside_external_prefixes(tmp_path, mutation):
    model, _, _, artifact_path, digest = _initialized_model(tmp_path)
    checkpoint = model.create_checkpoint_v3(
        config=_config(digest),
        external_mace=_external_metadata(digest),
    )
    if mutation == "missing":
        checkpoint["model_state_dict"].pop("pair_core.h0_projection.bias")
    else:
        checkpoint["model_state_dict"]["unexpected.weight"] = torch.zeros(1)
    path = tmp_path / f"{mutation}.pt"
    torch.save(checkpoint, path)
    with pytest.raises(RuntimeError, match=mutation):
        MACEAP3D3.load_checkpoint_v3(
            path,
            mace_artifact_path=artifact_path,
            model_factory=_factory,
            backbone_loader=_external_loader([]),
        )


@pytest.mark.parametrize(
    "expectation",
    [
        {"architecture": "hybrid-h2"},
        {"mace": {"feature_schema": "other-schema"}},
        {"mace": {"feature_mode": "all-scalars+norms"}},
        {"dtype_policy": "float64"},
        {"physics": {"physics_hash": "9" * 64}},
        {"data": {"dataset_hash": "8" * 64}},
        {"data": {"preprocessing_hash": "0" * 64}},
        {"data": {"split_hash": "7" * 64}},
        {"route_submodel_digests": {"legacy_atom_model": "6" * 64}},
    ],
)
def test_v3_semantic_mismatches_fail_before_external_deserialization(
    tmp_path, expectation
):
    model, _, _, artifact_path, digest = _initialized_model(tmp_path)
    path = tmp_path / "model.pt"
    model.save_checkpoint_v3(
        path,
        config=_config(digest),
        external_mace=_external_metadata(digest),
    )
    calls = []
    with pytest.raises(ValueError, match="semantic mismatch"):
        MACEAP3D3.load_checkpoint_v3(
            path,
            mace_artifact_path=artifact_path,
            model_factory=_factory,
            backbone_loader=_external_loader(calls),
            semantic_expectations=expectation,
        )
    assert calls == []


def test_v3_constructor_overrides_and_license_are_strict(tmp_path):
    model, _, _, artifact_path, digest = _initialized_model(tmp_path)
    external = _external_metadata(digest)
    external["license_acknowledged"] = False
    with pytest.raises(ValueError, match="license"):
        model.create_checkpoint_v3(config=_config(digest), external_mace=external)
    with pytest.raises(TypeError, match="non-record"):
        model.create_checkpoint_v3(
            config=_config(digest),
            external_mace=_external_metadata(digest),
            metadata={"ase_calculator": torch.nn.Linear(1, 1)},
        )

    path = tmp_path / "model.pt"
    model.save_checkpoint_v3(
        path,
        config=_config(digest),
        external_mace=_external_metadata(digest),
    )
    with pytest.raises(ValueError, match="constructor override"):
        MACEAP3D3.load_checkpoint_v3(
            path,
            mace_artifact_path=artifact_path,
            model_factory=_factory,
            backbone_loader=_external_loader([]),
            constructor_overrides={"pair_mode": "h2"},
        )


def test_v1_v2_checkpoint_versions_remain_unchanged():
    assert model_io.get_checkpoint_version({"layers.weight": torch.zeros(1)}) == 1
    checkpoint = model_io.create_checkpoint(
        torch.nn.Linear(2, 1),
        config={"in_features": 2},
        model_type="Linear",
    )
    assert checkpoint["checkpoint_version"] == 2
    assert model_io.CHECKPOINT_VERSION == 2
