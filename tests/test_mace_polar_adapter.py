import hashlib
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from apnet_pt.mace.encoder import (
    POLAR_1S_SHA256,
    MACEPolarFeaturizer,
    PrivateMACEFeatures,
    PolarMACEPrivateLayerAdapter,
    load_verified_polar_mace,
    verify_artifact,
)


def test_base_package_import_does_not_require_mace():
    code = """
import builtins
real_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == 'mace' or name.startswith('mace.'):
        raise AssertionError('base import attempted to import optional MACE')
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded
import apnet_pt
print(apnet_pt.__version__)
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


class TinyBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)


def test_artifact_digest_is_verified_before_loader_runs(tmp_path):
    artifact = tmp_path / "polar.model"
    artifact.write_bytes(b"not-a-checkpoint")
    called = False

    def loader(**kwargs):
        nonlocal called
        called = True
        return TinyBackbone()

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_verified_polar_mace(
            artifact,
            expected_sha256="0" * 64,
            loader=loader,
        )
    assert not called


def test_invalid_digest_does_not_import_default_mace_loader(tmp_path, monkeypatch):
    artifact = tmp_path / "polar.model"
    artifact.write_bytes(b"tampered")
    real_import = __import__

    def guarded_import(name, *args, **kwargs):
        if name == "mace" or name.startswith("mace."):
            raise AssertionError("MACE imported before digest verification")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", guarded_import)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_verified_polar_mace(artifact, expected_sha256="0" * 64)


def test_verified_loader_rejects_non_module_result(tmp_path):
    artifact = tmp_path / "polar.model"
    payload = b"verified-model"
    artifact.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    with pytest.raises(TypeError, match="torch.nn.Module"):
        load_verified_polar_mace(
            artifact,
            expected_sha256=digest,
            loader=lambda **kwargs: {"not": "a module"},
        )


def test_verified_loader_freezes_backbone_without_global_dtype_side_effect(tmp_path):
    artifact = tmp_path / "polar.model"
    payload = b"verified-model"
    artifact.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    dtype_before = torch.get_default_dtype()

    model = load_verified_polar_mace(
        artifact,
        expected_sha256=digest,
        loader=lambda **kwargs: TinyBackbone(),
    )

    assert verify_artifact(artifact, digest) == digest
    assert not model.training
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert torch.get_default_dtype() == dtype_before


@pytest.mark.mace_integration
def test_real_verified_polar_checkpoint_loads_as_frozen_module():
    artifact = Path("/tmp/MACE-POLAR-1-S.model")
    if not artifact.is_file():
        pytest.skip("local PolarMACE artifact is not available")
    model = load_verified_polar_mace(
        artifact,
        expected_sha256=POLAR_1S_SHA256,
        offline=True,
    )
    assert type(model).__name__ == "PolarMACE"
    assert not model.training
    assert all(not parameter.requires_grad for parameter in model.parameters())


def test_offline_loader_requires_local_artifact(tmp_path):
    with pytest.raises(FileNotFoundError, match="offline"):
        load_verified_polar_mace(
            tmp_path / "missing.model",
            expected_sha256="0" * 64,
            offline=True,
        )


class ProtocolBackbone(torch.nn.Module):
    def __init__(self, node_width=4):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.node_width = node_width
        self.atomic_numbers = torch.tensor([1, 6, 8])
        self.atomic_multipoles_max_l = 1
        self.calls = []

    def forward(self, data, **kwargs):
        self.calls.append({key: value.detach().clone() for key, value in data.items()})
        positions = data["positions"]
        numbers = data["atomic_numbers"].to(positions)
        count = positions.shape[0]
        target = data["total_charge"][0]
        centered_numbers = numbers - numbers.mean()
        charges = target / count + 0.01 * centered_numbers
        intrinsic = 0.1 * (positions - positions.mean(0, keepdim=True))
        density = torch.cat((charges[:, None], intrinsic[:, [1, 2, 0]]), dim=-1)
        node_feats = torch.stack(
            [numbers + float(index) for index in range(self.node_width)], dim=-1
        )
        dipole = (charges[:, None] * positions + intrinsic).sum(0, keepdim=True)
        return {
            "node_feats": node_feats * self.scale,
            "density_coefficients": density,
            "charges": charges,
            "dipole": dipole,
            "total_charge": charges.sum().reshape(1),
        }


class ProtocolPrivateAdapter:
    version = "protocol-private-v1"

    def extract(self, backbone, graph, public_outputs):
        positions = graph["positions"]
        scalar = public_outputs["node_feats"][:, :1]
        vector_yzx = positions[:, [1, 2, 0]]
        quadrupole = torch.stack(
            (
                positions[:, 0] * positions[:, 1],
                positions[:, 1] * positions[:, 2],
                positions[:, 2].square() - positions[:, 0].square(),
                positions[:, 2] * positions[:, 0],
                positions[:, 0].square() - positions[:, 1].square(),
            ),
            dim=-1,
        )
        hidden = torch.cat((scalar, vector_yzx, quadrupole), dim=-1)
        return PrivateMACEFeatures(
            final_scalars=public_outputs["node_feats"],
            hidden=hidden,
            hidden_irreps="1x0e+1x1o+1x2e",
            layer_count=1,
            adapter_version=self.version,
        )


def protocol_graph_builder(positions, atomic_numbers, total_charge, total_spin, dtype):
    count = atomic_numbers.numel()
    source, target = torch.where(~torch.eye(count, dtype=torch.bool))
    return {
        "positions": positions.to(dtype=dtype),
        "atomic_numbers": atomic_numbers,
        "batch": torch.zeros(count, dtype=torch.long, device=positions.device),
        "edge_index": torch.stack((source, target)).to(positions.device),
        "total_charge": total_charge.to(dtype=dtype),
        "total_spin": total_spin.to(dtype=dtype),
    }


def _protocol_featurizer(**kwargs):
    return MACEPolarFeaturizer(
        ProtocolBackbone(),
        checkpoint_sha256="a" * 64,
        mace_version="0.3.16",
        graph_builder=protocol_graph_builder,
        private_adapter=ProtocolPrivateAdapter(),
        **kwargs,
    )


def test_protocol_dimer_calls_are_isolated_frozen_and_have_runtime_schema():
    featurizer = _protocol_featurizer(feature_mode="all-scalars+norms")
    batch = type("Batch", (), {})()
    batch.RA = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.8]])
    batch.ZA = torch.tensor([8, 1])
    batch.molecule_ind_A = torch.tensor([0, 0])
    batch.total_charge_A = torch.tensor([0.0])
    batch.total_spin_A = torch.tensor([1.0])
    batch.RB = torch.tensor([[4.0, 0.0, 0.0], [4.0, 0.7, 0.0], [4.0, -0.7, 0.0]])
    batch.ZB = torch.tensor([6, 1, 1])
    batch.molecule_ind_B = torch.tensor([0, 0, 0])
    batch.total_charge_B = torch.tensor([0.0])
    batch.total_spin_B = torch.tensor([2.0])

    features_a, direct_a, features_b, direct_b = featurizer.forward_dimer(batch)

    assert [call["positions"].shape[0] for call in featurizer.backbone.calls] == [2, 3]
    for call in featurizer.backbone.calls:
        assert call["batch"].unique().tolist() == [0]
        assert call["edge_index"].max() < call["positions"].shape[0]
    assert features_a.feature_schema.startswith("polar-1-s:mace=0.3.16")
    assert featurizer.metadata["feature_schema"] == features_a.feature_schema
    assert featurizer.metadata["checkpoint_sha256"] == "a" * 64
    assert featurizer.metadata["supported_elements"] == (1, 6, 8)
    assert "inv=7" in features_a.feature_schema
    assert "irreps=1x0e+1x1o+1x2e" in features_a.feature_schema
    assert direct_a.density_coefficients.shape == (2, 4)
    assert direct_b.density_coefficients.shape == (3, 4)
    assert not featurizer.backbone.training
    featurizer.train()
    assert not featurizer.backbone.training
    assert all(not parameter.requires_grad for parameter in featurizer.backbone.parameters())
    assert not features_a.invariant.requires_grad
    assert not direct_b.charges.requires_grad


def test_private_adapter_parity_failure_is_fatal():
    class BadAdapter(ProtocolPrivateAdapter):
        def extract(self, backbone, graph, public_outputs):
            result = super().extract(backbone, graph, public_outputs)
            return PrivateMACEFeatures(
                final_scalars=result.final_scalars + 1.0,
                hidden=result.hidden,
                hidden_irreps=result.hidden_irreps,
                layer_count=result.layer_count,
                adapter_version=result.adapter_version,
            )

    featurizer = MACEPolarFeaturizer(
        ProtocolBackbone(),
        checkpoint_sha256="a" * 64,
        mace_version="0.3.16",
        graph_builder=protocol_graph_builder,
        private_adapter=BadAdapter(),
        feature_mode="all-scalars+norms",
    )
    with pytest.raises(RuntimeError, match="public-final-scalar parity"):
        featurizer.forward_monomer(
            torch.zeros(1, 3),
            torch.tensor([1]),
            torch.tensor([0.0]),
            torch.tensor([1.0]),
        )


def test_protocol_local_dtype_unsupported_elements_and_schema_discovery():
    default_before = torch.get_default_dtype()
    with pytest.raises(ValueError, match="multipole contract"):
        _protocol_featurizer(multipole_contract="unknown-contract")
    featurizer = _protocol_featurizer(
        feature_mode="final-layer-scalars", dtype=torch.float64
    )
    features, direct = featurizer.forward_monomer(
        torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        torch.tensor([1, 8]),
        torch.tensor([0.0]),
        torch.tensor([1.0]),
    )
    assert features.invariant.dtype == torch.float64
    assert direct.density_coefficients.dtype == torch.float64
    assert "inv=4" in features.feature_schema
    assert torch.get_default_dtype() == default_before
    with pytest.raises(ValueError, match="unsupported element.*9"):
        featurizer.forward_monomer(
            torch.zeros(1, 3), torch.tensor([9]), torch.tensor([0.0]), torch.tensor([1.0])
        )


def test_protocol_cache_online_parity_and_exact_invalidation():
    cache = {}
    featurizer = _protocol_featurizer(
        feature_mode="all-scalars+norms", cache=cache
    )
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.7, 0.0, 0.0]])
    numbers = torch.tensor([1, 8])
    args = (positions, numbers, torch.tensor([0.0]), torch.tensor([1.0]))
    online = featurizer.forward_monomer(*args)
    calls = len(featurizer.backbone.calls)
    cached = featurizer.forward_monomer(*args)
    assert len(featurizer.backbone.calls) == calls
    assert torch.equal(online[0].invariant, cached[0].invariant)
    assert torch.equal(online[0].equivariant, cached[0].equivariant)
    assert torch.equal(online[1].density_coefficients, cached[1].density_coefficients)

    variants = [
        (positions + 0.1, numbers, torch.tensor([0.0]), torch.tensor([1.0])),
        (positions.flip(0), numbers.flip(0), torch.tensor([0.0]), torch.tensor([1.0])),
        (positions, numbers, torch.tensor([1.0]), torch.tensor([1.0])),
        (positions, numbers, torch.tensor([0.0]), torch.tensor([2.0])),
    ]
    for variant in variants:
        featurizer.forward_monomer(*variant)
    assert len(featurizer.backbone.calls) == calls + len(variants)

    other_schema = _protocol_featurizer(
        feature_mode="final-layer-scalars", cache=cache
    )
    other_schema.forward_monomer(*args)
    assert len(other_schema.backbone.calls) == 1
    other_dtype = _protocol_featurizer(
        feature_mode="all-scalars+norms", cache=cache, dtype=torch.float64
    )
    other_dtype.forward_monomer(*args)
    assert len(other_dtype.backbone.calls) == 1


def test_protocol_translation_and_rotation_behavior():
    featurizer = _protocol_featurizer(feature_mode="final-layer-scalars")
    positions = torch.tensor([[0.2, -0.1, 0.3], [0.9, 0.4, -0.2]])
    numbers = torch.tensor([1, 8])
    charge = torch.tensor([0.0])
    spin = torch.tensor([1.0])
    features, direct = featurizer.forward_monomer(
        positions, numbers, charge, spin
    )
    shift = torch.tensor([1.5, -0.7, 0.2])
    shifted_features, shifted = featurizer.forward_monomer(
        positions + shift, numbers, charge, spin
    )
    assert torch.allclose(shifted_features.invariant, features.invariant)
    assert torch.allclose(shifted.charges, direct.charges)
    assert torch.allclose(
        shifted.intrinsic_dipole_eangstrom, direct.intrinsic_dipole_eangstrom
    )
    assert torch.allclose(
        shifted.molecular_dipole_eangstrom,
        direct.molecular_dipole_eangstrom,
        atol=1.0e-6,
    )

    rotation = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    rotated_features, rotated = featurizer.forward_monomer(
        positions @ rotation.T, numbers, charge, spin
    )
    assert torch.allclose(rotated_features.invariant, features.invariant)
    assert torch.allclose(
        rotated.intrinsic_dipole_eangstrom,
        direct.intrinsic_dipole_eangstrom @ rotation.T,
        atol=1.0e-6,
    )
    assert torch.allclose(
        rotated.molecular_dipole_eangstrom,
        direct.molecular_dipole_eangstrom @ rotation.T,
        atol=1.0e-6,
    )


def test_protocol_monomer_batch_order_and_atom_permutation():
    featurizer = _protocol_featurizer(feature_mode="final-layer-scalars")
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [3.0, 0.0, 0.0], [3.0, 0.8, 0.0]]
    )
    numbers = torch.tensor([1, 8, 6, 1])
    batch = torch.tensor([0, 0, 1, 1])
    features, direct = featurizer.forward_monomer(
        positions,
        numbers,
        torch.tensor([0.0, 1.0]),
        torch.tensor([1.0, 2.0]),
        batch=batch,
    )
    assert len(featurizer.backbone.calls) == 2
    assert features.batch.tolist() == [0, 0, 1, 1]
    assert torch.allclose(direct.total_charge, torch.tensor([0.0, 1.0]))

    order = torch.tensor([1, 0, 3, 2])
    permuted, permuted_direct = featurizer.forward_monomer(
        positions[order],
        numbers[order],
        torch.tensor([0.0, 1.0]),
        torch.tensor([1.0, 2.0]),
        batch=batch,
    )
    assert torch.allclose(permuted.invariant, features.invariant[order])
    assert torch.allclose(permuted_direct.charges, direct.charges[order])

    monomer_order = torch.tensor([2, 3, 0, 1])
    reordered, reordered_direct = featurizer.forward_monomer(
        positions[monomer_order],
        numbers[monomer_order],
        torch.tensor([1.0, 0.0]),
        torch.tensor([2.0, 1.0]),
        batch=batch,
    )
    assert torch.allclose(reordered.invariant, features.invariant[monomer_order])
    assert torch.allclose(reordered_direct.charges, direct.charges[monomer_order])
    assert torch.allclose(reordered_direct.total_charge, torch.tensor([1.0, 0.0]))


@pytest.mark.mace_integration
def test_real_private_adapter_public_parity_and_direct_contract():
    artifact = Path("/tmp/MACE-POLAR-1-S.model")
    if not artifact.is_file():
        pytest.skip("local PolarMACE artifact is not available")
    backbone = load_verified_polar_mace(
        artifact, expected_sha256=POLAR_1S_SHA256, offline=True
    )
    featurizer = MACEPolarFeaturizer(
        backbone,
        checkpoint_sha256=POLAR_1S_SHA256,
        mace_version="0.3.16",
        feature_mode="all-scalars+norms",
        private_adapter=PolarMACEPrivateLayerAdapter("0.3.16"),
    )
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.758602, 0.0, 0.504284], [-0.758602, 0.0, 0.504284]]
    )
    features, direct = featurizer.forward_monomer(
        positions,
        torch.tensor([8, 1, 1]),
        torch.tensor([0.0]),
        torch.tensor([1.0]),
    )
    assert features.invariant.shape == (3, 2560)
    assert features.equivariant.shape == (3, 8192)
    assert featurizer.last_private_parity_error <= 1.0e-6
    assert torch.allclose(direct.density_coefficients[:, 0], direct.charges, atol=1.0e-7)
    reconstructed = (
        direct.charges[:, None] * positions
        + direct.density_coefficients[:, [3, 1, 2]]
    ).sum(0, keepdim=True)
    assert torch.allclose(reconstructed, direct.molecular_dipole_eangstrom, atol=1.0e-5)

    from e3nn import o3

    rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    rotated_features, rotated_direct = featurizer.forward_monomer(
        positions @ rotation.T,
        torch.tensor([8, 1, 1]),
        torch.tensor([0.0]),
        torch.tensor([1.0]),
    )
    physical_to_mace = torch.tensor(
        [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]
    )
    mace_rotation = physical_to_mace @ rotation @ physical_to_mace.T
    irreps = o3.Irreps("512x0e+512x1o+512x2e+512x3o")
    d_matrix = irreps.D_from_matrix(mace_rotation)
    assert torch.allclose(rotated_features.invariant, features.invariant, atol=5.0e-6)
    assert torch.allclose(
        rotated_features.equivariant,
        features.equivariant @ d_matrix.T,
        atol=5.0e-6,
    )
    assert torch.allclose(rotated_direct.charges, direct.charges, atol=1.0e-6)
    assert torch.allclose(
        rotated_direct.intrinsic_dipole_eangstrom,
        direct.intrinsic_dipole_eangstrom @ rotation.T,
        atol=1.0e-6,
    )
