from copy import deepcopy
import io

import pytest
import torch

from apnet_pt.AtomPairwiseModels.apnet3_d3_fused import APNet3D3_AtomType_MPNN
from apnet_pt.mace.pair import MACEPairResidualCore
from apnet_pt.mace.schema import MACEAtomicFeatures
from tests.test_mace_h1_pair import (
    _batch,
    _features,
    _permute_bundle,
    _properties,
    _reindex_edges,
    _two_dimer_inputs,
)


def _with_mode(features, mode="all-scalars+norms"):
    return MACEAtomicFeatures(
        invariant=features.invariant,
        equivariant=features.equivariant,
        batch=features.batch,
        atomic_numbers=features.atomic_numbers,
        total_charge=features.total_charge,
        total_spin=features.total_spin,
        feature_schema=(
            f"test:mace=0.3.16:mode={mode}:adapter=stub:"
            f"inv={features.invariant.shape[1]}:equiv=0:layers=4"
        ),
    )


def _h2_fixture(*, no_disp=False, seed=29):
    torch.manual_seed(seed)
    batch = _batch()
    features_a = _features(
        batch.ZA,
        torch.randn(batch.ZA.numel(), 16),
        schema_mode="all-scalars+norms",
    )
    features_b = _features(
        batch.ZB,
        torch.randn(batch.ZB.numel(), 16),
        schema_mode="all-scalars+norms",
    )
    props_a = _properties(batch.ZA)
    props_b = _properties(batch.ZB, 0.1)
    ap3 = APNet3D3_AtomType_MPNN(
        dimer_prop_model=None,
        use_precomputed_classical=True,
        no_disp_nn=no_disp,
    )
    core = MACEPairResidualCore(
        ap3,
        mace_feature_dim=16,
        pair_mode="h2",
        feature_mode="all-scalars+norms",
    )
    return core, batch, features_a, features_b, props_a, props_b


def _call(fixture):
    core, batch, features_a, features_b, props_a, props_b = fixture
    return core(batch, features_a, features_b, props_a, props_b)


def test_h2_resolves_canonical_schema_and_rejects_h1_configuration():
    core, batch, features_a, features_b, props_a, props_b = _h2_fixture()
    assert core.pair_mode == "h2"
    assert core.feature_mode == "all-scalars+norms"
    assert core.architecture_id == "MACE-AP3D3-H2"
    assert core.get_config() == {
        "architecture_id": "MACE-AP3D3-H2",
        "pair_mode": "h2",
        "feature_mode": "all-scalars+norms",
        "mace_feature_dim": 16,
    }
    assert _call((core, batch, features_a, features_b, props_a, props_b)).shape == (
        1,
        4,
    )

    wrong_features = _with_mode(features_a, "final-layer-scalars")
    with pytest.raises(ValueError, match="all-scalars\\+norms"):
        core(batch, wrong_features, features_b, props_a, props_b)
    with pytest.raises(ValueError, match="canonical h2"):
        MACEPairResidualCore(
            deepcopy(core.ap3_core),
            mace_feature_dim=16,
            pair_mode="h2",
            feature_mode="final-layer-scalars",
        )
    with pytest.raises(ValueError, match="canonical h1"):
        MACEPairResidualCore(
            deepcopy(core.ap3_core),
            mace_feature_dim=16,
            pair_mode="h1",
            feature_mode="all-scalars+norms",
        )


def test_h2_bypasses_updates_and_directional_stack_with_same_head_capacity():
    h2 = _h2_fixture()
    h2_core = h2[0]
    # H2 must not even consume intramonomer edge indices.
    for edge_name in (
        "e_AA_source",
        "e_AA_target",
        "e_BB_source",
        "e_BB_target",
    ):
        setattr(h2[1], edge_name, torch.tensor([99]))
    counters = {"update": 0, "directional": 0}

    def count(kind):
        def hook(_module, _inputs, _outputs):
            counters[kind] += 1

        return hook

    handles = [
        layer.register_forward_hook(count("update"))
        for layer in h2_core.ap3_core.update_layers
    ]
    handles.extend(
        layer.register_forward_hook(count("directional"))
        for layer in h2_core.ap3_core.directional_layers
    )
    try:
        h2_result = _call(h2)
    finally:
        for handle in handles:
            handle.remove()
    assert counters == {"update": 0, "directional": 0}

    torch.manual_seed(29)
    batch = _batch()
    h1_features_a = _features(batch.ZA, h2[2].invariant)
    h1_features_b = _features(batch.ZB, h2[3].invariant)
    h1_ap3 = APNet3D3_AtomType_MPNN(
        dimer_prop_model=None, use_precomputed_classical=True
    )
    h1_core = MACEPairResidualCore(
        h1_ap3, mace_feature_dim=16, pair_mode="h1"
    )
    h1_result = h1_core(
        batch, h1_features_a, h1_features_b, h2[4], h2[5]
    )
    assert h1_core.h0_projection.out_features == h2_core.h0_projection.out_features
    assert h1_core.ap3_core.readout_layer_elst[0].in_features == 126
    assert h2_core.ap3_core.readout_layer_elst[0].in_features == 126
    assert h1_result.shape == h2_result.shape == (1, 4)


@pytest.mark.parametrize("no_disp", [False, True])
def test_h2_finite_four_component_contract_and_no_disp_padding(no_disp):
    result = _call(_h2_fixture(no_disp=no_disp))
    assert result.shape == (1, 4)
    assert torch.isfinite(result).all()
    if no_disp:
        assert torch.equal(result[:, 3], torch.zeros_like(result[:, 3]))


def test_h2_swap_rotation_and_translation_invariance():
    fixture = _h2_fixture()
    core, batch, features_a, features_b, props_a, props_b = fixture
    reference = _call(fixture)

    swapped = deepcopy(batch)
    swapped.ZA, swapped.ZB = batch.ZB, batch.ZA
    swapped.RA, swapped.RB = batch.RB, batch.RA
    swapped.e_AA_source, swapped.e_BB_source = batch.e_BB_source, batch.e_AA_source
    swapped.e_AA_target, swapped.e_BB_target = batch.e_BB_target, batch.e_AA_target
    swapped.e_ABsr_source = batch.e_ABsr_target
    swapped.e_ABsr_target = batch.e_ABsr_source
    swapped.e_ABlr_source = batch.e_ABlr_target
    swapped.e_ABlr_target = batch.e_ABlr_source
    swapped.total_charge_A, swapped.total_charge_B = (
        batch.total_charge_B,
        batch.total_charge_A,
    )
    actual_swap = core(swapped, features_b, features_a, props_b, props_a)
    assert torch.allclose(actual_swap, reference, atol=2.0e-6)

    rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    shift = torch.tensor([1.3, -0.4, 2.1])
    transformed = deepcopy(batch)
    transformed.RA = batch.RA @ rotation.T + shift
    transformed.RB = batch.RB @ rotation.T + shift
    actual_transform = core(
        transformed, features_a, features_b, props_a, props_b
    )
    assert torch.allclose(actual_transform, reference, atol=2.0e-6)


def test_h2_atom_permutation_and_batch_order_invariance():
    fixture = _h2_fixture()
    core, batch, features_a, features_b, props_a, props_b = fixture
    reference = _call(fixture)
    order_a = torch.tensor([1, 0])
    order_b = torch.tensor([1, 0])
    inverse_a = torch.argsort(order_a)
    inverse_b = torch.argsort(order_b)
    permuted = deepcopy(batch)
    permuted.ZA, permuted.RA = batch.ZA[order_a], batch.RA[order_a]
    permuted.ZB, permuted.RB = batch.ZB[order_b], batch.RB[order_b]
    permuted.e_AA_source = _reindex_edges(batch.e_AA_source, inverse_a)
    permuted.e_AA_target = _reindex_edges(batch.e_AA_target, inverse_a)
    permuted.e_BB_source = _reindex_edges(batch.e_BB_source, inverse_b)
    permuted.e_BB_target = _reindex_edges(batch.e_BB_target, inverse_b)
    for prefix in ("e_ABsr", "e_ABlr"):
        setattr(
            permuted,
            f"{prefix}_source",
            _reindex_edges(getattr(batch, f"{prefix}_source"), inverse_a),
        )
        setattr(
            permuted,
            f"{prefix}_target",
            _reindex_edges(getattr(batch, f"{prefix}_target"), inverse_b),
        )
    actual_permutation = core(
        permuted,
        _with_mode(_features(permuted.ZA, features_a.invariant[order_a])),
        _with_mode(_features(permuted.ZB, features_b.invariant[order_b])),
        _permute_bundle(props_a, order_a),
        _permute_bundle(props_b, order_b),
    )
    assert torch.allclose(actual_permutation, reference, atol=2.0e-6)

    ordered = list(_two_dimer_inputs([0, 1]))
    reordered = list(_two_dimer_inputs([1, 0]))
    ordered[1], ordered[2] = _with_mode(ordered[1]), _with_mode(ordered[2])
    reordered[1], reordered[2] = (
        _with_mode(reordered[1]),
        _with_mode(reordered[2]),
    )
    batch_core = MACEPairResidualCore(
        APNet3D3_AtomType_MPNN(
            dimer_prop_model=None, use_precomputed_classical=True
        ),
        mace_feature_dim=512,
        pair_mode="h2",
    )
    expected = batch_core(*ordered)
    actual_order = batch_core(*reordered)
    assert torch.allclose(actual_order, expected.flip(0), atol=3.0e-6)


def test_h2_state_dict_round_trip_and_h1_h2_state_rejection():
    fixture = _h2_fixture()
    core, batch, features_a, features_b, props_a, props_b = fixture
    expected = _call(fixture)
    buffer = io.BytesIO()
    torch.save(core.state_dict(), buffer)
    buffer.seek(0)
    restored = _h2_fixture()[0]
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    actual = restored(batch, features_a, features_b, props_a, props_b)
    assert torch.equal(actual, expected)

    h1 = MACEPairResidualCore(
        APNet3D3_AtomType_MPNN(
            dimer_prop_model=None, use_precomputed_classical=True
        ),
        mace_feature_dim=16,
        pair_mode="h1",
    )
    with pytest.raises(RuntimeError, match="architecture configuration"):
        h1.load_state_dict(core.state_dict())

    h1(
        batch,
        _features(batch.ZA, features_a.invariant),
        _features(batch.ZB, features_b.invariant),
        props_a,
        props_b,
    )
    with pytest.raises(RuntimeError, match="architecture configuration"):
        core.load_state_dict(h1.state_dict())
