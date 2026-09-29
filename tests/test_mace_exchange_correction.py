"""Public energy and checkpoint contracts for a parent-preserving correction."""

import io
from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from tests.test_mace_h3_pair import _h3_fixture


def _fixture(**overrides):
    from apnet_pt.mace.exchange import MACEExchangeCorrection

    _, batch, fa, fb, _, _ = _h3_fixture("h3l3")
    config = {
        "mace_feature_dim": 16,
        "mace_equivariant_dim": 4,
        "feature_schema": fa.feature_schema,
        "parent_contract_sha256": "a" * 64,
    }
    config.update(overrides)
    return MACEExchangeCorrection(**config), batch, fa, fb


def test_initial_correction_preserves_parent_components_exactly():
    model, batch, fa, fb = _fixture()
    parent = torch.tensor([[-3.5, 7.25, -0.125, -1.75]])
    original = parent.clone()
    actual = model(batch, fa, fb, parent)
    assert torch.equal(actual, original)
    assert torch.equal(parent, original)


def test_public_adapter_learns_only_exchange_and_cannot_train_parent():
    from apnet_pt.mace import MACEExchangeCorrection

    model, batch, fa, fb = _fixture()
    assert isinstance(model, MACEExchangeCorrection)
    parent = torch.tensor([[-3.5, 7.25, -0.125, -1.75]], requires_grad=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    before = model(batch, fa, fb, parent).detach()
    for _ in range(3):
        optimizer.zero_grad()
        prediction = model(batch, fa, fb, parent)
        (prediction[:, 1] - 6.25).square().mean().backward()
        optimizer.step()
    after = model(batch, fa, fb, parent)
    assert after[0, 1] < before[0, 1]
    assert torch.equal(after[:, [0, 2, 3]], before[:, [0, 2, 3]])
    assert parent.grad is None


def test_edges_cannot_silently_mix_parent_dimer_rows():
    model, batch, fa, fb = _fixture()
    fa = replace(
        fa,
        batch=torch.ones_like(fa.batch),
        total_charge=torch.zeros(2),
        total_spin=torch.ones(2),
    )
    batch_b = fb.batch.clone()
    batch_b[-1] = 1
    fb = replace(
        fb, batch=batch_b, total_charge=torch.zeros(2), total_spin=torch.ones(2)
    )
    with pytest.raises(ValueError, match="dimer"):
        model(batch, fa, fb, torch.zeros(2, 4))


def _activate(model, batch, fa, fb, parent):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(2):
        optimizer.zero_grad()
        prediction = model(batch, fa, fb, parent)
        (prediction[:, 1] - parent[:, 1] - 1.0).square().mean().backward()
        optimizer.step()
    return model(batch, fa, fb, parent)


@pytest.mark.parametrize("mode", ["scalar", "axial-tensor"])
@pytest.mark.parametrize("tensor_scaling", ["raw", "unit-ball"])
def test_trained_delta_is_swap_symmetric(mode, tensor_scaling):
    model, batch, fa, fb = _fixture(mode=mode, tensor_scaling=tensor_scaling)
    parent = torch.tensor([[-2.0, 5.0, -1.0, -1.5]])
    expected = _activate(model, batch, fa, fb, parent)
    assert not torch.equal(expected[:, 1], parent[:, 1])
    swapped = deepcopy(batch)
    swapped.RA, swapped.RB = batch.RB, batch.RA
    swapped.ZA, swapped.ZB = batch.ZB, batch.ZA
    swapped.e_ABsr_source = batch.e_ABsr_target
    swapped.e_ABsr_target = batch.e_ABsr_source
    torch.testing.assert_close(
        model(swapped, fb, fa, parent), expected, atol=1e-6, rtol=1e-6
    )


@pytest.mark.parametrize("reflection", [False, True])
@pytest.mark.parametrize("degree", [1, 2, 3])
@pytest.mark.parametrize("tensor_scaling", ["raw", "unit-ball"])
def test_trained_delta_is_joint_o3_and_translation_invariant(
    reflection, degree, tensor_scaling
):
    from apnet_pt.mace.encoder import _e3nn_o3
    from apnet_pt.mace.pair import MACE_E3NN_AXIS_PERMUTATION

    model, batch, fa, fb = _fixture(degree=degree, tensor_scaling=tensor_scaling)
    parent = torch.zeros(1, 4)
    expected = _activate(model, batch, fa, fb, parent)
    o3 = _e3nn_o3()
    transform = o3.rand_matrix()
    if reflection:
        transform = -transform
    permutation = torch.eye(3)[list(MACE_E3NN_AXIS_PERMUTATION)]
    internal = permutation @ transform @ permutation.T

    def rotate(features):
        blocks = [
            (
                features.equivariant_degree(degree)
                @ o3.Irrep(degree, (-1) ** degree).D_from_matrix(internal).T
            ).reshape(features.natom, -1)
            for degree in range(4)
        ]
        return replace(features, equivariant=torch.cat(blocks, dim=-1))

    moved = deepcopy(batch)
    moved.RA = batch.RA @ transform.T + 3.0
    moved.RB = batch.RB @ transform.T + 3.0
    torch.testing.assert_close(
        model(moved, rotate(fa), rotate(fb), parent), expected, atol=2e-6, rtol=1e-5
    )


def test_checkpoint_roundtrip_binds_parent_schema_and_descriptor_semantics():
    from apnet_pt.mace import MACEExchangeCorrection

    model, batch, fa, fb = _fixture()
    parent = torch.zeros(1, 4)
    expected = _activate(model, batch, fa, fb, parent)
    saved = io.BytesIO()
    torch.save(model.state_dict(), saved)
    saved.seek(0)
    state = torch.load(saved, weights_only=True)
    clone = MACEExchangeCorrection(**model.get_config())
    clone.load_state_dict(state)
    assert torch.equal(clone(batch, fa, fb, parent), expected)
    for changes in (
        {"mode": "scalar"},
        {"tensor_scaling": "raw"},
        {"parent_contract_sha256": "b" * 64},
        {"feature_schema": fa.feature_schema.replace("stub", "changed")},
    ):
        incompatible = MACEExchangeCorrection(**(model.get_config() | changes))
        with pytest.raises(RuntimeError, match="contract mismatch"):
            incompatible.load_state_dict(state)


def test_control_has_same_trainable_capacity_and_is_orientation_blind():
    from tests.test_mace_tensor_product import _twist_inputs

    batch, fa, fb, twisted, _, _ = _twist_inputs()
    scalar = _fixture(mode="scalar")[0]
    angular = _fixture(mode="axial-tensor")[0]
    assert sum(p.numel() for p in scalar.parameters()) == sum(
        p.numel() for p in angular.parameters()
    )
    parent = torch.zeros(1, 4)
    scalar_value = _activate(scalar, batch, fa, fb, parent)
    angular_value = _activate(angular, batch, fa, fb, parent)
    torch.testing.assert_close(
        scalar(batch, fa, twisted, parent), scalar_value, atol=1e-7, rtol=0
    )
    assert (angular(batch, fa, twisted, parent) - angular_value).abs().max() > 1e-5


def test_empty_edges_and_cutoff_preserve_parent_after_learning():
    model, batch, fa, fb = _fixture()
    parent = torch.tensor([[-1.0, 3.0, -0.2, -1.5]])
    _activate(model, batch, fa, fb, parent)
    far = deepcopy(batch)
    far.RB = batch.RB + 100.0
    assert torch.equal(model(far, fa, fb, parent), parent)
    empty = deepcopy(batch)
    empty.e_ABsr_source = torch.empty(0, dtype=torch.long)
    empty.e_ABsr_target = torch.empty(0, dtype=torch.long)
    empty.dimer_ind = torch.empty(0, dtype=torch.long)
    assert torch.equal(model(empty, fa, fb, parent), parent)


def test_features_and_parent_are_frozen_and_zero_irreps_are_finite():
    model, batch, fa, fb = _fixture()
    parent = torch.zeros(1, 4, requires_grad=True)
    fa = replace(
        fa,
        invariant=fa.invariant.clone().requires_grad_(),
        equivariant=torch.zeros_like(fa.equivariant).requires_grad_(),
    )
    after = _activate(model, batch, fa, fb, parent)
    assert torch.isfinite(after).all()
    # _activate's target references parent; the model output itself must not.
    parent.grad = None
    model(batch, fa, fb, parent).sum().backward()
    assert parent.grad is None
    assert fa.invariant.grad is None
    assert fa.equivariant.grad is None


def test_nonfinite_geometry_is_rejected_before_zero_readout_can_hide_it():
    model, batch, fa, fb = _fixture()
    batch.RA[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        model(batch, fa, fb, torch.zeros(1, 4))


@pytest.mark.parametrize("degree", [3.0, True, 0, 4])
def test_noninteger_or_unsupported_degree_is_rejected(degree):
    with pytest.raises(ValueError, match="degree"):
        _fixture(degree=degree)


def test_unit_ball_preserves_large_finite_tensor_directions():
    model, batch, fa, fb = _fixture()
    parent = torch.zeros(1, 4)
    _activate(model, batch, fa, fb, parent)
    fa = replace(fa, equivariant=fa.equivariant * 1e20)
    fb = replace(fb, equivariant=fb.equivariant * 1e20)
    reference_model = deepcopy(model).double()
    reference_batch = deepcopy(batch)
    reference_batch.RA = batch.RA.double()
    reference_batch.RB = batch.RB.double()

    def double(features):
        return replace(
            features,
            invariant=features.invariant.double(),
            equivariant=features.equivariant.double(),
            total_charge=features.total_charge.double(),
            total_spin=features.total_spin.double(),
        )

    reference = reference_model(
        reference_batch, double(fa), double(fb), parent.double()
    )
    actual = model(batch, fa, fb, parent)
    torch.testing.assert_close(actual.double(), reference, atol=1e-6, rtol=1e-6)


def test_valid_multidimer_batch_matches_separate_predictions_with_empty_dimer():
    model, batch, fa, fb = _fixture()
    parents = torch.tensor(
        [[-2.0, 3.0, -1.0, -0.5], [-4.0, 6.0, -2.0, -1.0], [1.0, 2.0, 3.0, 4.0]]
    )
    _activate(model, batch, fa, fb, parents[:1])
    expected = torch.cat(
        [model(batch, fa, fb, parents[i : i + 1]) for i in range(2)] + [parents[2:]]
    )

    def repeat(features):
        return replace(
            features,
            invariant=features.invariant.repeat(3, 1),
            equivariant=features.equivariant.repeat(3, 1),
            atomic_numbers=features.atomic_numbers.repeat(3),
            batch=torch.arange(3).repeat_interleave(features.natom),
            total_charge=features.total_charge.repeat(3),
            total_spin=features.total_spin.repeat(3),
        )

    merged = deepcopy(batch)
    merged.RA, merged.RB = batch.RA.repeat(3, 1), batch.RB.repeat(3, 1)
    merged.ZA, merged.ZB = batch.ZA.repeat(3), batch.ZB.repeat(3)
    merged.e_ABsr_source = torch.stack(
        (batch.e_ABsr_source, batch.e_ABsr_source + fa.natom), dim=1
    ).flatten()
    merged.e_ABsr_target = torch.stack(
        (batch.e_ABsr_target, batch.e_ABsr_target + fb.natom), dim=1
    ).flatten()
    merged.dimer_ind = torch.tensor([0, 1]).repeat(batch.dimer_ind.numel())
    torch.testing.assert_close(
        model(merged, repeat(fa), repeat(fb), parents), expected, atol=1e-6, rtol=1e-6
    )


def test_cutoff_value_and_radial_slope_vanish_at_boundary():
    model, batch, fa, fb = _fixture()
    parent = torch.zeros(1, 4)
    _activate(model, batch, fa, fb, parent)
    one = deepcopy(batch)
    one.e_ABsr_source = torch.tensor([0])
    one.e_ABsr_target = torch.tensor([0])
    one.dimer_ind = torch.tensor([0])
    one.RA = batch.RA.clone()
    one.RA[0] = 0
    cutoff = model.get_config()["cutoff"]
    for distance in (cutoff - 0.01, cutoff, cutoff + 0.01):
        one.RB = batch.RB.clone()
        one.RB[0] = torch.tensor([distance, 0, 0])
        one.RB.requires_grad_()
        prediction = model(one, fa, fb, parent)
        gradient = torch.autograd.grad(prediction.sum(), one.RB)[0]
        if distance < cutoff:
            assert 0 < prediction[:, 1].abs().max() < 1e-5
            assert gradient.abs().max() < 1e-3
        else:
            assert torch.equal(prediction, parent)
            torch.testing.assert_close(
                gradient, torch.zeros_like(gradient), atol=1e-7, rtol=0
            )
