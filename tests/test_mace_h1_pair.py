from copy import deepcopy
import io
from types import SimpleNamespace

import pytest
import qcelemental as qcel
import torch

from apnet_pt import constants
from apnet_pt.AtomPairwiseModels.apnet3_d3_fused import APNet3D3_AtomType_MPNN
from apnet_pt.mace.pair import MACEPairResidualCore
from apnet_pt.mace.properties import LegacyAtomMPNNPropertyProvider
from apnet_pt.mace.schema import AtomicPropertyBundle, MACEAtomicFeatures


def _legacy_tuple(natom, offset=0.0):
    q = torch.linspace(-0.2, 0.2, natom) + offset
    mu = torch.arange(natom * 3, dtype=torch.float32).reshape(natom, 3) / 20
    quadrupole = torch.zeros(natom, 3, 3)
    quadrupole[:, 0, 0] = 0.2
    quadrupole[:, 1, 1] = -0.1
    quadrupole[:, 2, 2] = -0.1
    hidden = torch.zeros(natom, 2, 3)
    response = torch.stack(
        (torch.linspace(-0.8, 1.2, natom), torch.linspace(0.7, -1.1, natom)),
        dim=-1,
    )
    damping = torch.linspace(-1.5, 2.0, natom)
    return q, mu, quadrupole, hidden, response, damping


class LegacyOutputStub(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.marker = torch.nn.Parameter(torch.tensor(0.0))
        self.outputs = (_legacy_tuple(2), _legacy_tuple(2, 0.1))
        self.calls = 0

    def forward(self, batch):
        del batch
        self.calls += 1
        return self.outputs


def test_legacy_property_adapter_is_exact_narrow_canonical_mapping():
    legacy = LegacyOutputStub()
    provider = LegacyAtomMPNNPropertyProvider(legacy, freeze=True)
    features = _features(torch.tensor([1, 8]), torch.zeros(2, 4))
    props_a, props_b = provider(None, features, features)

    assert legacy.calls == 1
    for bundle, raw in zip((props_a, props_b), legacy.outputs):
        assert torch.equal(bundle.q[:, 0], raw[0])
        assert torch.equal(bundle.mu, raw[1])
        assert torch.equal(bundle.quadrupole, raw[2])
        assert torch.equal(bundle.hfvr[:, 0], raw[-2][:, 0])
        assert torch.equal(bundle.valence_width[:, 0], raw[-2][:, 1])
        assert torch.equal(bundle.damping[:, 0], raw[-1].abs())
        expected_alpha = (
            constants.polarizability_table[bundle.q.new_tensor([1, 8]).long()]
            .reshape(-1, 1)
            .to(bundle.q)
            * bundle.hfvr.abs().pow(4.0 / 3.0)
        )
        assert torch.allclose(bundle.alpha, expected_alpha)
    assert all(not parameter.requires_grad for parameter in legacy.parameters())


def test_real_legacy_stack_adapter_parity_and_injection_seam_noop():
    from apnet_pt.AtomPairwiseModels.apnet3_d3_fused import (
        APNet3D3_AtomType_Model,
    )

    dimer = qcel.models.Molecule.from_data(
        """
        0 1
        O 0.000000 0.000000 0.000000
        H 0.758602 0.000000 0.504284
        H -0.758602 0.000000 0.504284
        --
        0 1
        O 3.000000 0.200000 0.000000
        H 3.758602 0.200000 0.504284
        H 2.241398 0.200000 0.504284
        units angstrom
        """
    )
    harness = APNet3D3_AtomType_Model(
        pre_trained_model_path="./models/ap3d3_ensemble/ap3d3_0_no_disp.pt"
    )
    batch = harness._qcel_example_input(
        [dimer],
        batch_size=1,
        r_cut=harness.model.r_cut,
        r_cut_im=harness.model.r_cut_im,
    )
    harness.model.eval()
    state_keys_before = tuple(harness.model.state_dict())
    with torch.no_grad():
        legacy_default = harness.model(batch)
        explicit_noop = harness.model(
            batch,
            initial_atom_states=None,
            atomic_properties=None,
            residual_only=False,
            pair_energy_envelope=False,
        )
    for expected, actual in zip(legacy_default, explicit_noop):
        if torch.is_tensor(expected):
            assert torch.equal(actual, expected)
    assert tuple(harness.model.state_dict()) == state_keys_before

    harness.dimer_prop_model.set_forward("ap3_atomMPNN")
    with torch.no_grad():
        raw_a, raw_b = harness.dimer_prop_model(batch)
    features_a = _features(batch.ZA, torch.zeros(batch.ZA.numel(), 4))
    features_b = _features(batch.ZB, torch.zeros(batch.ZB.numel(), 4))
    provider = LegacyAtomMPNNPropertyProvider(harness.dimer_prop_model)
    with torch.no_grad():
        props_a, props_b = provider(batch, features_a, features_b)
    for props, raw in ((props_a, raw_a), (props_b, raw_b)):
        assert torch.equal(props.q[:, 0], raw[0].reshape(-1))
        assert torch.equal(props.mu, raw[1])
        assert torch.equal(props.quadrupole, raw[2])
        assert torch.equal(props.hfvr[:, 0], raw[-2][:, 0])
        assert torch.equal(props.valence_width[:, 0], raw[-2][:, 1])
        assert torch.equal(props.damping[:, 0], raw[-1].reshape(-1).abs())

    h1_features_a = _features(batch.ZA, torch.zeros(batch.ZA.numel(), 512))
    h1_features_b = _features(batch.ZB, torch.zeros(batch.ZB.numel(), 512))
    h1 = MACEPairResidualCore(harness.model, mace_feature_dim=512)
    with torch.no_grad():
        residual = h1(batch, h1_features_a, h1_features_b, props_a, props_b)
    assert residual.shape == (1, 4)
    assert torch.isfinite(residual).all()
    assert torch.equal(residual[:, 3], torch.zeros_like(residual[:, 3]))


def _features(
    numbers,
    invariant,
    *,
    schema_mode="final-layer-scalars",
    batch=None,
):
    if batch is None:
        batch = torch.zeros(numbers.numel(), dtype=torch.long, device=numbers.device)
    nmonomer = int(batch.max().item()) + 1
    return MACEAtomicFeatures(
        invariant=invariant,
        equivariant=invariant.new_zeros((numbers.numel(), 0)),
        batch=batch,
        atomic_numbers=numbers,
        total_charge=invariant.new_zeros(nmonomer),
        total_spin=invariant.new_ones(nmonomer),
        feature_schema=(
            f"polar-1-s:mace=0.3.16:mode={schema_mode}:adapter=public:"
            f"inv={invariant.shape[1]}:equiv=0:layers=1:public"
        ),
    )


def _properties(numbers, offset=0.0):
    natom = numbers.numel()
    device = numbers.device
    hfvr = torch.linspace(0.8, 1.2, natom, device=device).reshape(-1, 1)
    return AtomicPropertyBundle(
        q=(torch.linspace(-0.2, 0.2, natom, device=device) + offset).reshape(-1, 1),
        mu=torch.zeros(natom, 3, device=device),
        quadrupole=torch.zeros(natom, 3, 3, device=device),
        hfvr=hfvr,
        valence_width=torch.linspace(0.7, 1.1, natom, device=device).reshape(-1, 1),
        alpha=(
            constants.polarizability_table.to(numbers.device)[numbers]
            .reshape(-1, 1)
            .to(hfvr)
            * hfvr.pow(4.0 / 3.0)
        ),
        damping=torch.ones(natom, 1, device=device),
    )


def _batch():
    return SimpleNamespace(
        ZA=torch.tensor([1, 8]),
        RA=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.7, 0.0]]),
        ZB=torch.tensor([6, 1]),
        RB=torch.tensor([[3.0, 0.1, 0.0], [3.3, 0.8, 0.2]]),
        e_ABsr_source=torch.tensor([0, 0, 1, 1]),
        e_ABsr_target=torch.tensor([0, 1, 0, 1]),
        e_ABlr_source=torch.tensor([0, 0, 1, 1]),
        e_ABlr_target=torch.tensor([0, 1, 0, 1]),
        e_AA_source=torch.tensor([0, 1]),
        e_AA_target=torch.tensor([1, 0]),
        e_BB_source=torch.tensor([0, 1]),
        e_BB_target=torch.tensor([1, 0]),
        dimer_ind=torch.zeros(4, dtype=torch.long),
        total_charge_A=torch.tensor([0.0]),
        total_charge_B=torch.tensor([0.0]),
    )


def _h1_fixture():
    torch.manual_seed(12)
    batch = _batch()
    features_a = _features(batch.ZA, torch.randn(2, 512))
    features_b = _features(batch.ZB, torch.randn(2, 512))
    props_a = _properties(batch.ZA)
    props_b = _properties(batch.ZB, 0.1)
    ap3 = APNet3D3_AtomType_MPNN(
        dimer_prop_model=None,
        use_precomputed_classical=True,
    )
    core = MACEPairResidualCore(ap3, mace_feature_dim=512)
    return core, batch, features_a, features_b, props_a, props_b


def test_h1_contract_replaces_embedding_and_executes_ap3_updates_and_directions():
    core, batch, features_a, features_b, props_a, props_b = _h1_fixture()
    assert core.feature_mode == "final-layer-scalars"
    update_calls = []
    directional_calls = []
    hooks = []
    for layer in core.ap3_core.update_layers:
        hooks.append(layer.register_forward_hook(lambda *args: update_calls.append(1)))
    for layer in core.ap3_core.directional_layers:
        hooks.append(
            layer.register_forward_hook(lambda *args: directional_calls.append(1))
        )

    residual = core(batch, features_a, features_b, props_a, props_b)
    for hook in hooks:
        hook.remove()
    assert residual.shape == (1, 4)
    assert torch.isfinite(residual).all()
    assert len(update_calls) == 2 * core.ap3_core.n_message
    assert len(directional_calls) == 2 * core.ap3_core.n_message
    residual.square().mean().backward()
    assert torch.isfinite(core.h0_projection.weight.grad).all()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in core.ap3_core.update_layers.parameters()
    )

    with torch.no_grad():
        core.ap3_core.embed_layer.weight.fill_(1.0e6)
    changed_embedding = core(batch, features_a, features_b, props_a, props_b)
    assert torch.allclose(changed_embedding, residual)

    wrong_schema = _features(
        batch.ZA, features_a.invariant, schema_mode="all-scalars+norms"
    )
    with pytest.raises(ValueError, match="final-layer-scalars"):
        core(batch, wrong_schema, features_b, props_a, props_b)


def test_h1_bidirectional_shared_readouts_sum_then_aggregate_once():
    core, batch, features_a, features_b, props_a, props_b = _h1_fixture()
    calls = []
    hook = core.ap3_core.readout_layer_elst.register_forward_hook(
        lambda module, inputs, output: calls.append(output.detach())
    )
    residual = core(batch, features_a, features_b, props_a, props_b)
    hook.remove()
    assert len(calls) == 2
    assert not any("ba" in key.lower() for key in core.ap3_core.state_dict())

    h_ab = core.last_h_ab
    h_ba = core.last_h_ba
    pair_sum = core.ap3_core.readouts(h_ab) + core.ap3_core.readouts(h_ba)
    distances, _ = core.ap3_core.get_distances(
        batch.RA,
        batch.RB,
        batch.e_ABsr_source,
        batch.e_ABsr_target,
    )
    pair_sum = pair_sum * (
        core.ap3_core.smooth_pair_energy_envelope(distances) / distances.pow(3)
    ).unsqueeze(-1)
    expected = pair_sum.new_zeros((1, 4))
    expected.index_add_(0, batch.dimer_ind, pair_sum)
    assert torch.allclose(residual, expected)


def _reindex_edges(edges, inverse):
    return inverse.index_select(0, edges)


def _permute_bundle(bundle, order):
    return AtomicPropertyBundle(
        **{name: getattr(bundle, name)[order] for name in bundle.__dataclass_fields__}
    )


def test_h1_swap_and_atom_permutation_behavior():
    core, batch, features_a, features_b, props_a, props_b = _h1_fixture()
    reference = core(batch, features_a, features_b, props_a, props_b)

    swapped = deepcopy(batch)
    swapped.ZA, swapped.ZB = batch.ZB, batch.ZA
    swapped.RA, swapped.RB = batch.RB, batch.RA
    swapped.e_AA_source, swapped.e_BB_source = batch.e_BB_source, batch.e_AA_source
    swapped.e_AA_target, swapped.e_BB_target = batch.e_BB_target, batch.e_AA_target
    swapped.e_ABsr_source, swapped.e_ABsr_target = (
        batch.e_ABsr_target,
        batch.e_ABsr_source,
    )
    swapped.e_ABlr_source, swapped.e_ABlr_target = (
        batch.e_ABlr_target,
        batch.e_ABlr_source,
    )
    swapped.total_charge_A, swapped.total_charge_B = (
        batch.total_charge_B,
        batch.total_charge_A,
    )
    swapped_result = core(
        swapped, features_b, features_a, props_b, props_a
    )
    assert torch.allclose(swapped_result, reference, atol=2.0e-6)

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
    permuted_features_a = _features(
        permuted.ZA, features_a.invariant[order_a]
    )
    permuted_features_b = _features(
        permuted.ZB, features_b.invariant[order_b]
    )
    permuted_result = core(
        permuted,
        permuted_features_a,
        permuted_features_b,
        _permute_bundle(props_a, order_a),
        _permute_bundle(props_b, order_b),
    )
    assert torch.allclose(permuted_result, reference, atol=2.0e-6)


def _single_atom_fixture(distance, *, keep_edge=True, no_disp=False, device="cpu"):
    device = torch.device(device)
    edge = torch.tensor([0], dtype=torch.long, device=device)
    empty = torch.empty(0, dtype=torch.long, device=device)
    pair_edge = edge if keep_edge else empty
    batch = SimpleNamespace(
        ZA=torch.tensor([1], dtype=torch.long, device=device),
        RA=torch.tensor([[0.0, 0.0, 0.0]], device=device),
        ZB=torch.tensor([8], dtype=torch.long, device=device),
        RB=torch.tensor([[distance, 0.0, 0.0]], device=device),
        e_ABsr_source=pair_edge,
        e_ABsr_target=pair_edge,
        e_ABlr_source=pair_edge,
        e_ABlr_target=pair_edge,
        e_AA_source=empty,
        e_AA_target=empty,
        e_BB_source=empty,
        e_BB_target=empty,
        dimer_ind=torch.zeros(pair_edge.numel(), dtype=torch.long, device=device),
        total_charge_A=torch.tensor([0.0], device=device),
        total_charge_B=torch.tensor([0.0], device=device),
    )
    features_a = _features(batch.ZA, torch.zeros(1, 512, device=device))
    features_b = _features(batch.ZB, torch.zeros(1, 512, device=device))
    props_a = _properties(batch.ZA)
    props_b = _properties(batch.ZB)
    ap3 = APNet3D3_AtomType_MPNN(
        dimer_prop_model=None,
        use_precomputed_classical=True,
        no_disp_nn=no_disp,
    ).to(device)
    core = MACEPairResidualCore(ap3, mace_feature_dim=512).to(device)
    return core, batch, features_a, features_b, props_a, props_b


def _constant_readouts(ap3_core):
    for readout in (
        ap3_core.readout_layer_elst,
        ap3_core.readout_layer_exch,
        ap3_core.readout_layer_indu,
        getattr(ap3_core, "readout_layer_disp", None),
    ):
        if readout is None:
            continue
        linear_layers = [
            module for module in readout if isinstance(module, torch.nn.Linear)
        ]
        with torch.no_grad():
            for parameter in readout.parameters():
                parameter.zero_()
            linear_layers[-1].bias.fill_(1.0)


def test_h1_smooth_pair_envelope_at_and_beyond_cutoff_with_edge_removal():
    fixture = _single_atom_fixture(7.0)
    core, batch, features_a, features_b, props_a, props_b = fixture
    core(batch, features_a, features_b, props_a, props_b)
    _constant_readouts(core.ap3_core)

    values = []
    for distance in (
        core.ap3_core.r_cut_im - 1.0e-3,
        core.ap3_core.r_cut_im,
        8.2,
    ):
        batch.RB = torch.tensor([[distance, 0.0, 0.0]])
        values.append(core(batch, features_a, features_b, props_a, props_b))
    assert (values[0].abs() > 0).all()
    assert torch.equal(values[1], torch.zeros_like(values[1]))
    assert torch.equal(values[2], torch.zeros_like(values[2]))

    batch.e_ABsr_source = torch.empty(0, dtype=torch.long)
    batch.e_ABsr_target = torch.empty(0, dtype=torch.long)
    batch.e_ABlr_source = torch.empty(0, dtype=torch.long)
    batch.e_ABlr_target = torch.empty(0, dtype=torch.long)
    batch.dimer_ind = torch.empty(0, dtype=torch.long)
    removed = core(batch, features_a, features_b, props_a, props_b)
    assert torch.equal(removed, torch.zeros_like(removed))


@pytest.mark.parametrize(
    "device",
    ["cpu"] + (["cuda"] if torch.cuda.is_available() else []),
)
def test_h1_single_atom_monomers_use_input_dtype_and_device(device):
    core, batch, features_a, features_b, props_a, props_b = _single_atom_fixture(
        3.0, device=device
    )
    messages = core.ap3_core.get_messages(
        features_a.invariant[:, :8],
        features_a.invariant[:, :8],
        torch.empty(0, 8, device=device),
        batch.e_AA_source,
        batch.e_AA_target,
    )
    assert messages.device.type == device
    assert messages.dtype == features_a.invariant.dtype
    residual = core(batch, features_a, features_b, props_a, props_b)
    assert residual.shape == (1, 4)
    assert torch.isfinite(residual).all()


def test_h1_rotation_and_translation_invariance():
    core, batch, features_a, features_b, props_a, props_b = _h1_fixture()
    reference = core(batch, features_a, features_b, props_a, props_b)
    rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    shift = torch.tensor([2.1, -0.7, 1.3])
    transformed = deepcopy(batch)
    transformed.RA = batch.RA @ rotation.T + shift
    transformed.RB = batch.RB @ rotation.T + shift
    actual = core(transformed, features_a, features_b, props_a, props_b)
    assert torch.allclose(actual, reference, atol=3.0e-6)


def _cat_bundles(bundles):
    return AtomicPropertyBundle(
        **{
            name: torch.cat([getattr(bundle, name) for bundle in bundles], dim=0)
            for name in bundles[0].__dataclass_fields__
        }
    )


def _two_dimer_inputs(order):
    base = _batch()
    a_numbers = []
    b_numbers = []
    a_positions = []
    b_positions = []
    features_a = []
    features_b = []
    props_a = []
    props_b = []
    aa_source = []
    aa_target = []
    bb_source = []
    bb_target = []
    ab_source = []
    ab_target = []
    dimer_indices = []
    for new_index, original_index in enumerate(order):
        offset_a = len(a_numbers) * 2
        offset_b = len(b_numbers) * 2
        displacement = torch.tensor(
            [0.4 * original_index, 2.0 * original_index, 0.0]
        )
        a_numbers.append(base.ZA)
        b_numbers.append(base.ZB)
        a_positions.append(base.RA + displacement)
        b_positions.append(base.RB + displacement)
        generator = torch.Generator().manual_seed(100 + original_index)
        features_a.append(torch.randn(2, 512, generator=generator))
        features_b.append(torch.randn(2, 512, generator=generator))
        props_a.append(_properties(base.ZA, 0.05 * original_index))
        props_b.append(_properties(base.ZB, 0.1 + 0.05 * original_index))
        aa_source.append(base.e_AA_source + offset_a)
        aa_target.append(base.e_AA_target + offset_a)
        bb_source.append(base.e_BB_source + offset_b)
        bb_target.append(base.e_BB_target + offset_b)
        ab_source.append(base.e_ABsr_source + offset_a)
        ab_target.append(base.e_ABsr_target + offset_b)
        dimer_indices.append(torch.full((4,), new_index, dtype=torch.long))
    za = torch.cat(a_numbers)
    zb = torch.cat(b_numbers)
    molecule_batch = torch.arange(len(order)).repeat_interleave(2)
    batch = SimpleNamespace(
        ZA=za,
        RA=torch.cat(a_positions),
        ZB=zb,
        RB=torch.cat(b_positions),
        e_ABsr_source=torch.cat(ab_source),
        e_ABsr_target=torch.cat(ab_target),
        e_ABlr_source=torch.cat(ab_source),
        e_ABlr_target=torch.cat(ab_target),
        e_AA_source=torch.cat(aa_source),
        e_AA_target=torch.cat(aa_target),
        e_BB_source=torch.cat(bb_source),
        e_BB_target=torch.cat(bb_target),
        dimer_ind=torch.cat(dimer_indices),
        total_charge_A=torch.zeros(len(order)),
        total_charge_B=torch.zeros(len(order)),
    )
    return (
        batch,
        _features(za, torch.cat(features_a), batch=molecule_batch),
        _features(zb, torch.cat(features_b), batch=molecule_batch),
        _cat_bundles(props_a),
        _cat_bundles(props_b),
    )


def test_h1_batch_order_equivalence():
    torch.manual_seed(31)
    ap3 = APNet3D3_AtomType_MPNN(
        dimer_prop_model=None, use_precomputed_classical=True
    )
    core = MACEPairResidualCore(ap3, mace_feature_dim=512)
    ordered = _two_dimer_inputs([0, 1])
    reordered = _two_dimer_inputs([1, 0])
    expected = core(*ordered)
    actual = core(*reordered)
    assert torch.allclose(actual, expected.flip(0), atol=3.0e-6)


def test_h1_low_level_state_dict_round_trip_prediction_equality():
    core, batch, features_a, features_b, props_a, props_b = _h1_fixture()
    expected = core(batch, features_a, features_b, props_a, props_b)
    buffer = io.BytesIO()
    torch.save(core.state_dict(), buffer)
    buffer.seek(0)

    restored, _, _, _, _, _ = _h1_fixture()
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    actual = restored(batch, features_a, features_b, props_a, props_b)
    assert torch.equal(actual, expected)
