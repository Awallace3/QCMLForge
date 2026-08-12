from copy import deepcopy

import pytest
import torch

from apnet_pt.AtomPairwiseModels.apnet3_d3_fused import APNet3D3_AtomType_MPNN
from apnet_pt.mace.model import (
    MACEAP3D3,
    MACEAP3D3Model,
    MACEAtomicPropertiesModel,
)
from apnet_pt.mace.pair import MACEPairResidualCore
from apnet_pt.mace.schema import (
    AtomicPropertyBundle,
    ClassicalEnergyBundle,
    InductionDiagnostics,
    MACEAtomicFeatures,
    PhysicsConfig,
    PolarMACEDirectOutputs,
)
from tests.test_mace_h1_pair import _batch, _two_dimer_inputs


ROUTES = {
    "direct-polar": ("h1", "all-scalars+norms", "direct"),
    "hybrid-h1": ("h1", "final-layer-scalars", "legacy"),
    "hybrid-h2": ("h2", "all-scalars+norms", "legacy"),
    "atomhead": ("h1", "all-scalars+norms", "atomhead"),
}


def _augment_batch(batch):
    ndimer = batch.total_charge_A.numel()
    natom_a = batch.ZA.numel() // ndimer
    natom_b = batch.ZB.numel() // ndimer
    batch.molecule_ind_A = torch.arange(ndimer).repeat_interleave(natom_a)
    batch.molecule_ind_B = torch.arange(ndimer).repeat_interleave(natom_b)
    batch.total_spin_A = torch.ones(ndimer)
    batch.total_spin_B = torch.ones(ndimer)
    batch.e_ABfull_source = batch.e_ABsr_source
    batch.e_ABfull_target = batch.e_ABsr_target
    batch.dimer_ind_full = batch.dimer_ind
    batch.natom_per_mol_A = torch.full((ndimer,), natom_a, dtype=torch.long)
    batch.natom_per_mol_B = torch.full((ndimer,), natom_b, dtype=torch.long)
    batch.y = torch.arange(ndimer * 4, dtype=torch.float32).reshape(ndimer, 4)
    return batch


def _feature_values(numbers, supplied=None):
    if supplied is not None:
        return supplied[:, :16]
    z = numbers.float().reshape(-1, 1)
    scales = torch.linspace(0.02, 0.17, 16, device=z.device).reshape(1, -1)
    return torch.sin(z * scales)


class StubFeaturizer(torch.nn.Module):
    def __init__(self, feature_mode):
        super().__init__()
        self.feature_mode = feature_mode
        self.backbone = torch.nn.Linear(1, 1, bias=False)

    def _features(self, numbers, batch, charge, spin, supplied=None):
        invariant = _feature_values(numbers, supplied)
        return MACEAtomicFeatures(
            invariant=invariant,
            equivariant=invariant.new_zeros((numbers.numel(), 0)),
            batch=batch,
            atomic_numbers=numbers,
            total_charge=charge.to(invariant),
            total_spin=spin.to(invariant),
            feature_schema=(
                f"stub:mace=0.3.16:mode={self.feature_mode}:adapter=stub:"
                "inv=16:equiv=0:layers=4"
            ),
        )

    @staticmethod
    def _direct(features, positions):
        density = positions.new_zeros((positions.shape[0], 4))
        molecular_dipole = positions.new_zeros((features.total_charge.numel(), 3))
        return PolarMACEDirectOutputs(
            density_coefficients=density,
            charges=density[:, 0],
            molecular_dipole_eangstrom=molecular_dipole,
            positions_angstrom=positions,
            batch=features.batch,
            total_charge=features.total_charge,
        )

    def forward_dimer(self, batch):
        features_a = self._features(
            batch.ZA,
            batch.molecule_ind_A,
            batch.total_charge_A,
            batch.total_spin_A,
            getattr(batch, "stub_invariant_A", None),
        )
        features_b = self._features(
            batch.ZB,
            batch.molecule_ind_B,
            batch.total_charge_B,
            batch.total_spin_B,
            getattr(batch, "stub_invariant_B", None),
        )
        return (
            features_a,
            self._direct(features_a, batch.RA),
            features_b,
            self._direct(features_b, batch.RB),
        )

    def forward_monomer(
        self, positions, numbers, total_charge, total_spin, *, batch=None
    ):
        if batch is None:
            batch = torch.zeros(numbers.numel(), dtype=torch.long)
        features = self._features(numbers, batch, total_charge, total_spin)
        return features, self._direct(features, positions)


class StubPropertyProvider(torch.nn.Module):
    def __init__(self, provider_kind):
        super().__init__()
        self.provider_kind = provider_kind
        self.scale = torch.nn.Parameter(torch.tensor(0.2))
        self.direct_calls = 0

    def forward_monomer(self, features, direct=None):
        if self.provider_kind == "direct":
            assert direct is not None
            self.direct_calls += 1
        natom = features.natom
        base = self.scale.expand(natom, 1)
        q = base - base.mean()
        hfvr = 1.0 + 0.1 * base
        return AtomicPropertyBundle(
            q=q,
            mu=base.expand(-1, 3) * 0.01,
            quadrupole=base[:, None].expand(-1, 3, 3) * 0.0,
            hfvr=hfvr,
            valence_width=1.0 + 0.05 * base,
            alpha=1.0 + 0.2 * base,
            damping=1.0 + 0.03 * base,
        )

    def forward(
        self,
        batch,
        features_a,
        features_b,
        *,
        direct_a=None,
        direct_b=None,
        **kwargs,
    ):
        del batch, kwargs
        return (
            self.forward_monomer(features_a, direct_a),
            self.forward_monomer(features_b, direct_b),
        )


class StubLongRangeProvider(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0
        self.dispersion_calls = 0
        self.config = PhysicsConfig()

    def forward(self, batch, props_a, props_b):
        del props_a, props_b
        self.calls += 1
        self.dispersion_calls += 1
        ndimer = batch.total_charge_A.numel()
        npair = batch.e_ABfull_source.numel()
        pair = batch.RA.new_zeros(npair)
        return ClassicalEnergyBundle(
            pair_elst=pair + 0.01,
            pair_ind=pair + 0.02,
            pair_disp=pair + 0.03,
            dimer_elst=batch.RA.new_full((ndimer,), 0.1),
            dimer_ind=batch.RA.new_full((ndimer,), 0.2),
            dimer_disp=batch.RA.new_full((ndimer,), 0.3),
            induction_diagnostics=InductionDiagnostics(True, 4, 1.0e-10),
            physics_config_hash=self.config.physics_hash,
        )


def _make_model(route, *, no_disp=False):
    pair_mode, feature_mode, provider_kind = ROUTES[route]
    featurizer = StubFeaturizer(feature_mode)
    provider = StubPropertyProvider(provider_kind)
    ap3 = APNet3D3_AtomType_MPNN(
        dimer_prop_model=None,
        use_precomputed_classical=True,
        no_disp_nn=no_disp,
    )
    pair_kwargs = {}
    if route in {"direct-polar", "atomhead"}:
        pair_kwargs["architecture_id"] = route
    pair_core = MACEPairResidualCore(
        ap3,
        mace_feature_dim=16,
        pair_mode=pair_mode,
        feature_mode=feature_mode,
        **pair_kwargs,
    )
    long_range = StubLongRangeProvider()
    model = MACEAP3D3(
        architecture=route,
        featurizer=featurizer,
        property_provider=provider,
        pair_core=pair_core,
        long_range_provider=long_range,
    )
    return model, provider, long_range


@pytest.mark.parametrize("route", ROUTES)
def test_all_routes_forward_backward_optimizer_and_component_ledgers(route):
    torch.manual_seed(41)
    model, provider, long_range = _make_model(route)
    batch = _augment_batch(_batch())
    labels_before = batch.y.clone()
    details = model(batch, return_details=True)

    assert details.components.shape == (1, 4)
    assert torch.isfinite(details.components).all()
    assert tuple(details.component_ledger) == ("elst", "exch", "indu", "disp")
    assert tuple(details.residual_ledger) == ("elst", "exch", "indu", "disp")
    assert set(details.classical_ledger) == {"elst", "indu", "disp"}
    assert details.induction_diagnostics.converged
    expected = details.residual.clone()
    expected[:, 0] += 0.1
    expected[:, 2] += 0.2
    expected[:, 3] += 0.3
    assert torch.allclose(details.components, expected)
    assert torch.equal(batch.y, labels_before)
    assert long_range.calls == long_range.dispersion_calls == 1
    assert not model.featurizer.backbone.training
    assert all(not parameter.requires_grad for parameter in model.featurizer.backbone.parameters())
    if route == "direct-polar":
        assert provider.direct_calls == 2

    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1.0e-3,
    )
    harness = MACEAP3D3Model(model)
    before = model.pair_core.h0_projection.weight.detach().clone()
    loss = harness.train_step(batch, optimizer)
    assert torch.isfinite(loss)
    assert set(harness.last_component_losses) == {"elst", "exch", "indu", "disp"}
    assert all(torch.isfinite(value) for value in harness.last_component_losses.values())
    assert model.pair_core.h0_projection.weight.grad is not None
    assert torch.isfinite(model.pair_core.h0_projection.weight.grad).all()
    assert not torch.equal(before, model.pair_core.h0_projection.weight)
    assert torch.equal(batch.y, labels_before)
    prediction = harness.predict_batch(batch)
    assert prediction.shape == (1, 4)
    assert torch.isfinite(prediction).all()


def test_no_disp_neural_route_retains_d3_exactly_once():
    model, _, long_range = _make_model("hybrid-h2", no_disp=True)
    batch = _augment_batch(_batch())
    details = model(batch, return_details=True)
    assert torch.equal(details.residual[:, 3], torch.zeros_like(details.residual[:, 3]))
    assert torch.equal(details.components[:, 3], torch.full((1,), 0.3))
    assert long_range.calls == long_range.dispersion_calls == 1


@pytest.mark.parametrize(
    ("route", "changed", "match"),
    [
        ("hybrid-h1", {"provider_kind": "direct"}, "property provider"),
        ("hybrid-h2", {"pair_mode": "h1"}, "pair topology"),
        ("atomhead", {"feature_mode": "final-layer-scalars"}, "feature mode"),
    ],
)
def test_route_topology_provider_and_schema_mismatches_fail(route, changed, match):
    model, _, _ = _make_model(route)
    for name, value in changed.items():
        target = model.property_provider if name == "provider_kind" else model.pair_core
        if name == "feature_mode":
            target = model.featurizer
        setattr(target, name, value)
    with pytest.raises(ValueError, match=match):
        MACEAP3D3(
            architecture=route,
            featurizer=model.featurizer,
            property_provider=model.property_provider,
            pair_core=model.pair_core,
            long_range_provider=model.long_range_provider,
        )
    with pytest.raises(ValueError, match="architecture"):
        MACEAP3D3(
            architecture="MACE-AP3D3-H1",
            featurizer=model.featurizer,
            property_provider=model.property_provider,
            pair_core=model.pair_core,
            long_range_provider=model.long_range_provider,
        )


def _permute_batch(batch, order_a, order_b):
    inverse_a = torch.argsort(order_a)
    inverse_b = torch.argsort(order_b)
    permuted = deepcopy(batch)
    permuted.ZA, permuted.RA = batch.ZA[order_a], batch.RA[order_a]
    permuted.ZB, permuted.RB = batch.ZB[order_b], batch.RB[order_b]
    permuted.molecule_ind_A = batch.molecule_ind_A[order_a]
    permuted.molecule_ind_B = batch.molecule_ind_B[order_b]
    for prefix, inverse in (("e_AA", inverse_a), ("e_BB", inverse_b)):
        setattr(permuted, f"{prefix}_source", inverse[getattr(batch, f"{prefix}_source")])
        setattr(permuted, f"{prefix}_target", inverse[getattr(batch, f"{prefix}_target")])
    for prefix in ("e_ABsr", "e_ABlr", "e_ABfull"):
        setattr(permuted, f"{prefix}_source", inverse_a[getattr(batch, f"{prefix}_source")])
        setattr(permuted, f"{prefix}_target", inverse_b[getattr(batch, f"{prefix}_target")])
    return permuted


@pytest.mark.parametrize("route", ROUTES)
def test_all_routes_rotation_translation_permutation_and_swap_equivalence(route):
    torch.manual_seed(47)
    model, _, _ = _make_model(route)
    batch = _augment_batch(_batch())
    reference = model(batch)

    rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    shift = torch.tensor([1.2, -0.8, 0.4])
    transformed = deepcopy(batch)
    transformed.RA = batch.RA @ rotation.T + shift
    transformed.RB = batch.RB @ rotation.T + shift
    assert torch.allclose(model(transformed), reference, atol=3.0e-6)

    permuted = _permute_batch(batch, torch.tensor([1, 0]), torch.tensor([1, 0]))
    assert torch.allclose(model(permuted), reference, atol=3.0e-6)

    swapped = deepcopy(batch)
    for suffix in ("Z", "R", "molecule_ind", "total_charge", "total_spin"):
        first = getattr(batch, f"{suffix}A" if suffix in {"Z", "R"} else f"{suffix}_A")
        second = getattr(batch, f"{suffix}B" if suffix in {"Z", "R"} else f"{suffix}_B")
        if suffix in {"Z", "R"}:
            setattr(swapped, f"{suffix}A", second)
            setattr(swapped, f"{suffix}B", first)
        else:
            setattr(swapped, f"{suffix}_A", second)
            setattr(swapped, f"{suffix}_B", first)
    swapped.e_AA_source, swapped.e_BB_source = batch.e_BB_source, batch.e_AA_source
    swapped.e_AA_target, swapped.e_BB_target = batch.e_BB_target, batch.e_AA_target
    for prefix in ("e_ABsr", "e_ABlr", "e_ABfull"):
        setattr(swapped, f"{prefix}_source", getattr(batch, f"{prefix}_target"))
        setattr(swapped, f"{prefix}_target", getattr(batch, f"{prefix}_source"))
    swapped.natom_per_mol_A, swapped.natom_per_mol_B = (
        batch.natom_per_mol_B,
        batch.natom_per_mol_A,
    )
    assert torch.allclose(model(swapped), reference, atol=3.0e-6)


@pytest.mark.parametrize("route", ROUTES)
def test_all_routes_batch_order_equivalence(route):
    ordered = list(_two_dimer_inputs([0, 1]))
    reordered = list(_two_dimer_inputs([1, 0]))
    ordered_batch = _augment_batch(ordered[0])
    reordered_batch = _augment_batch(reordered[0])
    ordered_batch.stub_invariant_A = ordered[1].invariant
    ordered_batch.stub_invariant_B = ordered[2].invariant
    reordered_batch.stub_invariant_A = reordered[1].invariant
    reordered_batch.stub_invariant_B = reordered[2].invariant
    torch.manual_seed(53)
    model, _, _ = _make_model(route)
    expected = model(ordered_batch)
    actual = model(reordered_batch)
    assert torch.allclose(actual, expected.flip(0), atol=3.0e-6)


@pytest.mark.parametrize(
    ("property_mode", "provider_kind"),
    [("direct-completion", "direct"), ("learned", "atomhead")],
)
def test_atomic_property_harness_optimizer_step(property_mode, provider_kind):
    featurizer = StubFeaturizer("all-scalars+norms")
    provider = StubPropertyProvider(provider_kind)
    model = MACEAtomicPropertiesModel(
        property_mode=property_mode,
        featurizer=featurizer,
        property_provider=provider,
    )
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.7, 0.0]])
    numbers = torch.tensor([1, 8])
    charge = torch.tensor([0.0])
    spin = torch.tensor([1.0])
    target = AtomicPropertyBundle(
        q=torch.zeros(2, 1),
        mu=torch.zeros(2, 3),
        quadrupole=torch.zeros(2, 3, 3),
        hfvr=torch.ones(2, 1),
        valence_width=torch.ones(2, 1),
        alpha=torch.ones(2, 1),
        damping=torch.ones(2, 1),
    )
    prediction = model(positions, numbers, charge, spin)
    loss, losses = model.compute_loss(prediction, target)
    assert torch.isfinite(loss)
    assert set(losses) == set(target.__dataclass_fields__)
    optimizer = torch.optim.Adam(model.trainable_parameters(), lr=1.0e-3)
    before = provider.scale.detach().clone()
    step_loss = model.train_step(
        positions, numbers, charge, spin, target=target, optimizer=optimizer
    )
    assert torch.isfinite(step_loss)
    assert torch.isfinite(provider.scale.grad)
    assert not torch.equal(provider.scale, before)
    assert all(not parameter.requires_grad for parameter in featurizer.backbone.parameters())
