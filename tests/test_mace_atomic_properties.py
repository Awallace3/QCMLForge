from pathlib import Path

import pytest
import torch

from apnet_pt import constants
from apnet_pt.mace.properties import (
    AtomicPropertyProvider,
    MACEAtomPropertyModel,
    MACEPropertyCompletionHeads,
    PolarDirectPropertyProvider,
)
from apnet_pt.mace.schema import MACEAtomicFeatures, PolarMACEDirectOutputs


def _features(
    *,
    invariant=None,
    equivariant=None,
    batch=None,
    total_charge=None,
):
    torch.manual_seed(7)
    if invariant is None:
        invariant = torch.randn(4, 5)
    if equivariant is None:
        equivariant = torch.randn(4, 18)
    if batch is None:
        batch = torch.tensor([0, 0, 1, 1])
    if total_charge is None:
        total_charge = torch.tensor([0.0, 1.0])
    return MACEAtomicFeatures(
        invariant=invariant,
        equivariant=equivariant,
        batch=batch,
        atomic_numbers=torch.tensor([1, 8, 6, 1]),
        total_charge=total_charge.to(invariant),
        total_spin=torch.ones(total_charge.numel()).to(invariant),
        feature_schema="stub:all-scalars+norms:equiv=2x0e+2x1o+2x2e",
    )


def _heads():
    torch.manual_seed(11)
    return MACEPropertyCompletionHeads(
        invariant_dim=5,
        equivariant_irreps="2x0e+2x1o+2x2e",
        hidden_dim=12,
        geometry_channels=3,
    )


def _assert_bundle_constraints(bundle, features):
    assert bundle.q.shape == (features.natom, 1)
    assert bundle.mu.shape == (features.natom, 3)
    assert bundle.quadrupole.shape == (features.natom, 3, 3)
    assert torch.isfinite(bundle.q).all()
    assert torch.isfinite(bundle.mu).all()
    assert torch.isfinite(bundle.quadrupole).all()
    for value in (bundle.hfvr, bundle.valence_width, bundle.alpha, bundle.damping):
        assert torch.isfinite(value).all()
        assert (value > 0).all()
    assert torch.allclose(
        bundle.quadrupole, bundle.quadrupole.transpose(-1, -2), atol=1.0e-6
    )
    assert torch.allclose(
        bundle.quadrupole.diagonal(dim1=-2, dim2=-1).sum(-1),
        torch.zeros(features.natom),
        atol=1.0e-6,
    )
    for monomer in range(features.total_charge.numel()):
        assert torch.allclose(
            bundle.q[features.batch == monomer].sum(),
            features.total_charge[monomer],
            atol=2.0e-7,
        )
    expected_alpha = (
        constants.polarizability_table.to(bundle.hfvr)[features.atomic_numbers]
        .reshape(-1, 1)
        * bundle.hfvr.pow(4.0 / 3.0)
    )
    assert torch.allclose(bundle.alpha, expected_alpha)


def test_atomhead_c_constraints_one_step_and_no_atommpnn_state():
    features = _features()
    model = MACEAtomPropertyModel(_heads())
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}

    bundle = model.forward_monomer(features)
    _assert_bundle_constraints(bundle, features)
    assert isinstance(model, AtomicPropertyProvider)
    assert "AtomMPNN" not in repr(model)
    assert all("atommpnn" not in key.lower() for key in model.state_dict())

    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    loss = sum(value.square().mean() for value in bundle.__dict__.values())
    optimizer.zero_grad()
    loss.backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    optimizer.step()
    assert any(
        not torch.equal(before[name], value)
        for name, value in model.state_dict().items()
    )


def test_atomhead_c_rotation_covariance_and_atom_permutation():
    from e3nn import o3

    features = _features(batch=torch.zeros(4, dtype=torch.long), total_charge=torch.tensor([0.0]))
    features = MACEAtomicFeatures(
        invariant=features.invariant,
        equivariant=features.equivariant,
        batch=features.batch,
        atomic_numbers=features.atomic_numbers,
        total_charge=features.total_charge,
        total_spin=torch.tensor([1.0]),
        feature_schema=features.feature_schema,
    )
    model = MACEAtomPropertyModel(_heads()).eval()
    reference = model.forward_monomer(features)

    rotation = o3.rand_matrix(dtype=features.invariant.dtype)
    permutation_to_mace = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    mace_rotation = permutation_to_mace @ rotation @ permutation_to_mace.T
    d_matrix = o3.Irreps("2x0e+2x1o+2x2e").D_from_matrix(mace_rotation)
    rotated_features = MACEAtomicFeatures(
        invariant=features.invariant,
        equivariant=features.equivariant @ d_matrix.T,
        batch=features.batch,
        atomic_numbers=features.atomic_numbers,
        total_charge=features.total_charge,
        total_spin=features.total_spin,
        feature_schema=features.feature_schema,
    )
    rotated = model.forward_monomer(rotated_features)
    assert torch.allclose(rotated.q, reference.q, atol=2.0e-5)
    assert torch.allclose(rotated.mu, reference.mu @ rotation.T, atol=2.0e-5)
    expected_quad = rotation @ reference.quadrupole @ rotation.T
    assert torch.allclose(rotated.quadrupole, expected_quad, atol=3.0e-5)

    order = torch.tensor([2, 0, 3, 1])
    permuted_features = MACEAtomicFeatures(
        invariant=features.invariant[order],
        equivariant=features.equivariant[order],
        batch=features.batch[order],
        atomic_numbers=features.atomic_numbers[order],
        total_charge=features.total_charge,
        total_spin=features.total_spin,
        feature_schema=features.feature_schema,
    )
    permuted = model.forward_monomer(permuted_features)
    assert torch.allclose(permuted.q, reference.q[order], atol=2.0e-7)
    assert torch.allclose(permuted.mu, reference.mu[order], atol=2.0e-6)
    assert torch.allclose(permuted.quadrupole, reference.quadrupole[order], atol=2.0e-6)


def _direct_outputs(positions, charges, intrinsic_mu, total_charge=0.0):
    density = torch.cat(
        (charges[:, None], intrinsic_mu[:, [1, 2, 0]]), dim=-1
    )
    batch = torch.zeros(charges.numel(), dtype=torch.long)
    dipole = (charges[:, None] * positions + intrinsic_mu).sum(0, keepdim=True)
    return PolarMACEDirectOutputs(
        density_coefficients=density,
        charges=charges,
        molecular_dipole_eangstrom=dipole,
        positions_angstrom=positions,
        batch=batch,
        total_charge=torch.tensor([total_charge], dtype=positions.dtype),
        multipole_contract="polar-density-l1-yzx-eangstrom-v1",
    )


def test_directpolar_conversion_charge_origin_and_dipole_reconstruction():
    positions = torch.tensor([[0.1, -0.2, 0.3], [1.2, 0.4, -0.5]])
    charges = torch.tensor([-0.4, 0.4])
    intrinsic = torch.tensor([[0.03, -0.02, 0.01], [-0.01, 0.04, 0.02]])
    direct = _direct_outputs(positions, charges, intrinsic)
    features = MACEAtomicFeatures(
        invariant=torch.randn(2, 5),
        equivariant=torch.randn(2, 18),
        batch=torch.zeros(2, dtype=torch.long),
        atomic_numbers=torch.tensor([1, 8]),
        total_charge=torch.tensor([0.0]),
        total_spin=torch.tensor([1.0]),
        feature_schema="stub:equiv=2x0e+2x1o+2x2e",
    )
    completion = _heads()
    provider = PolarDirectPropertyProvider(completion)
    bundle = provider.forward_monomer(features, direct)

    _assert_bundle_constraints(bundle, features)
    assert torch.allclose(bundle.q[:, 0], charges, atol=2.0e-7)
    assert torch.allclose(bundle.mu * constants.au2ang, intrinsic, atol=2.0e-7)
    reconstructed = (
        bundle.q * positions + bundle.mu * constants.au2ang
    ).sum(0, keepdim=True)
    assert torch.allclose(reconstructed, direct.molecular_dipole_eangstrom, atol=1.0e-5)
    assert provider.completion_heads is completion
    assert isinstance(provider, AtomicPropertyProvider)
    assert "AtomMPNN" not in repr(provider)
    assert all("atommpnn" not in key.lower() for key in provider.state_dict())
    optimizer = torch.optim.Adam(provider.parameters(), lr=1.0e-3)
    completion_loss = bundle.quadrupole.square().mean() + bundle.hfvr.square().mean()
    optimizer.zero_grad()
    completion_loss.backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in provider.parameters()
    )
    optimizer.step()

    shift = torch.tensor([2.0, -1.0, 0.5])
    shifted = _direct_outputs(positions + shift, charges, intrinsic)
    shifted_bundle = provider.forward_monomer(features, shifted)
    assert torch.allclose(shifted_bundle.q, bundle.q)
    assert torch.allclose(shifted_bundle.mu, bundle.mu)
    assert torch.allclose(
        shifted.molecular_dipole_eangstrom,
        direct.molecular_dipole_eangstrom,
        atol=1.0e-6,
    )

    charged_q = torch.tensor([0.3, 0.7])
    charged = _direct_outputs(positions, charged_q, intrinsic, total_charge=1.0)
    charged_shifted = _direct_outputs(
        positions + shift, charged_q, intrinsic, total_charge=1.0
    )
    assert torch.allclose(
        charged_shifted.molecular_dipole_eangstrom,
        charged.molecular_dipole_eangstrom + shift.reshape(1, 3),
        atol=1.0e-6,
    )
    charged_features = MACEAtomicFeatures(
        invariant=features.invariant,
        equivariant=features.equivariant,
        batch=features.batch,
        atomic_numbers=features.atomic_numbers,
        total_charge=torch.tensor([1.0]),
        total_spin=features.total_spin,
        feature_schema=features.feature_schema,
    )
    assert torch.allclose(
        provider.forward_monomer(charged_features, charged).mu,
        provider.forward_monomer(charged_features, charged_shifted).mu,
    )


def test_directpolar_rotation_and_incompatible_contract_rejection():
    from e3nn import o3

    positions = torch.tensor([[0.2, 0.3, -0.1], [1.0, -0.4, 0.7]])
    charges = torch.tensor([-0.3, 0.3])
    intrinsic = torch.tensor([[0.04, 0.02, -0.03], [-0.02, 0.01, 0.05]])
    rotation = o3.rand_matrix(dtype=positions.dtype)
    direct = _direct_outputs(positions, charges, intrinsic)
    rotated = _direct_outputs(positions @ rotation.T, charges, intrinsic @ rotation.T)
    assert torch.allclose(
        rotated.density_coefficients[:, [3, 1, 2]],
        direct.density_coefficients[:, [3, 1, 2]] @ rotation.T,
        atol=1.0e-6,
    )
    assert torch.allclose(
        rotated.molecular_dipole_eangstrom,
        direct.molecular_dipole_eangstrom @ rotation.T,
        atol=1.0e-6,
    )
    features = MACEAtomicFeatures(
        invariant=torch.randn(2, 5),
        equivariant=torch.randn(2, 18),
        batch=torch.zeros(2, dtype=torch.long),
        atomic_numbers=torch.tensor([1, 8]),
        total_charge=torch.tensor([0.0]),
        total_spin=torch.tensor([1.0]),
        feature_schema="stub:equiv=2x0e+2x1o+2x2e",
    )
    provider = PolarDirectPropertyProvider(_heads())
    reference_mu = provider.forward_monomer(features, direct).mu
    rotated_mu = provider.forward_monomer(features, rotated).mu
    assert torch.allclose(rotated_mu, reference_mu @ rotation.T, atol=1.0e-6)
    with pytest.raises(ValueError, match="multipole contract"):
        PolarMACEDirectOutputs(
            density_coefficients=direct.density_coefficients,
            charges=direct.charges,
            molecular_dipole_eangstrom=direct.molecular_dipole_eangstrom,
            positions_angstrom=direct.positions_angstrom,
            batch=direct.batch,
            total_charge=direct.total_charge,
            multipole_contract="unknown-spherical-contract",
        )


@pytest.mark.mace_integration
def test_real_checkpoint_directpolar_wiring_is_finite_not_accuracy_claim():
    from apnet_pt.mace.encoder import (
        MACEPolarFeaturizer,
        POLAR_1S_SHA256,
        PolarMACEPrivateLayerAdapter,
        load_verified_polar_mace,
    )

    from tests.mace_integration import polar_mace_artifact

    artifact = polar_mace_artifact()
    backbone = load_verified_polar_mace(
        artifact, expected_sha256=POLAR_1S_SHA256, offline=True
    )
    featurizer = MACEPolarFeaturizer(
        backbone,
        checkpoint_sha256=POLAR_1S_SHA256,
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
    heads = MACEPropertyCompletionHeads(
        invariant_dim=features.invariant.shape[1],
        equivariant_irreps="512x0e+512x1o+512x2e+512x3o",
        hidden_dim=8,
        geometry_channels=2,
    )
    provider = PolarDirectPropertyProvider(heads)
    bundle = provider.forward_monomer(features, direct)
    _assert_bundle_constraints(bundle, features)
    assert torch.allclose(bundle.q[:, 0], direct.charges, atol=1.0e-6)
    reconstructed = (
        bundle.q * positions + bundle.mu * constants.au2ang
    ).sum(0, keepdim=True)
    assert torch.allclose(reconstructed, direct.molecular_dipole_eangstrom, atol=1.0e-5)
