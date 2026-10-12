"""Stub MACE stack for tests that must not touch the licensed foundation model.

The real featurizer wraps a 33 MB separately-licensed PolarMACE checkpoint, so
nothing in the default suite can construct one.  These stubs stand in for the
three seams ``MACEAP3D3`` composes -- featurizer, atomic-property provider and
classical long-range provider -- with tensors that are cheap but still carry
the properties the routes depend on: the equivariant block rotates with the
geometry, the property bundle is differentiable, and the classical bundle
reports a physics hash.
"""

from types import SimpleNamespace

import torch

from apnet_pt.AtomPairwiseModels.apnet3_d3_fused import APNet3D3_AtomType_MPNN
from apnet_pt.mace.model import MACEAP3D3
from apnet_pt.mace.pair import (
    DIRECTIONAL_DEGREES,
    MACE_E3NN_AXIS_PERMUTATION,
    MACEPairResidualCore,
    real_spherical_harmonics,
)
from apnet_pt.mace.schema import (
    AtomicPropertyBundle,
    ClassicalEnergyBundle,
    InductionDiagnostics,
    MACEAtomicFeatures,
    PhysicsConfig,
    PolarMACEDirectOutputs,
)


def _batch():
    """One two-atom-per-monomer dimer, wired for every edge set a route reads."""

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


ROUTES = {
    "direct-polar": ("h1", "all-scalars+norms", "direct"),
    "hybrid-h1": ("h1", "final-layer-scalars", "legacy"),
    "hybrid-h2": ("h2", "all-scalars+norms", "legacy"),
    "hybrid-h3": ("h3", "all-scalars+norms", "legacy"),
    "hybrid-h3l1": ("h3l1", "all-scalars+norms", "legacy"),
    "hybrid-h3l3": ("h3l3", "all-scalars+norms", "legacy"),
    "hybrid-h3l3q": ("h3l3", "all-scalars+norms", "legacy"),
    "hybrid-h3l3w112": ("h3l3", "all-scalars+norms", "legacy"),
    "hybrid-h3l3p": ("h3l3", "all-scalars+norms", "legacy"),
    "hybrid-h3l3w112p": ("h3l3", "all-scalars+norms", "legacy"),
    "atomhead": ("h1", "all-scalars+norms", "atomhead"),
}

# A narrow stand-in for PolarMACE's ``512x0e+512x1o+512x2e+512x3o``: the H3
# routes only need *an* l=1, l=2 and l=3 block to slice, and four channels keep
# the stub cheap.  Which routes need it is read off ``DIRECTIONAL_DEGREES``
# rather than listed here, so a future degree variant cannot reach this harness
# with a zero-width equivariant block and fail somewhere less obvious.
STUB_IRREPS = "4x0e+4x1o+4x2e+4x3o"
STUB_EQUIVARIANT_CHANNELS = 4
STUB_EQUIVARIANT_WIDTH = 4 * (1 + 3 + 5 + 7)


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
    def __init__(self, feature_mode, equivariant_irreps=""):
        super().__init__()
        self.feature_mode = feature_mode
        self.equivariant_irreps = equivariant_irreps
        self.backbone = torch.nn.Linear(1, 1, bias=False)

    def _equivariant(self, numbers, invariant, positions, molecule_ind):
        """Build a block that actually rotates with the geometry.

        Real MACE equivariants are covariant under rotation, and the H3 routes
        contract them against the interatomic axis.  A block built from atomic
        numbers alone would sit still while the coordinates turned, so the
        harness's rotation check would fail for a reason that has nothing to do
        with the route under test.  Each atom's offset from its own monomer
        centroid is rotation-covariant, translation-invariant, and permutation-
        equivariant, which is everything that check exercises.  It is expressed
        in MACE's own axis order because that is the frame
        ``MACEPairResidualCore`` contracts in.
        """

        if not self.equivariant_irreps:
            return invariant.new_zeros((numbers.numel(), 0))
        ndimer = int(molecule_ind.max().item()) + 1
        ones = positions.new_ones(positions.shape[0])
        counts = positions.new_zeros(ndimer).index_add_(0, molecule_ind, ones)
        totals = positions.new_zeros((ndimer, 3)).index_add_(
            0, molecule_ind, positions
        )
        offset = positions - (totals / counts.unsqueeze(1)).index_select(
            0, molecule_ind
        )
        offset = offset[:, list(MACE_E3NN_AXIS_PERMUTATION)]
        # Guard the centroid-coincident case explicitly; e3nn's own
        # normalization would hand back NaN and the failure would surface as an
        # unrelated assertion.
        offset = offset / offset.norm(dim=1, keepdim=True).clamp_min(1.0e-6)

        z = numbers.float().reshape(-1, 1)
        channels = torch.arange(
            1, STUB_EQUIVARIANT_CHANNELS + 1, device=z.device
        ).reshape(1, -1)
        scale = torch.sin(z * channels * 0.31).to(positions)
        blocks = []
        for degree in (0, 1, 2, 3):
            if degree == 0:
                harmonic = offset.new_ones((numbers.numel(), 1))
            else:
                harmonic = real_spherical_harmonics(degree, offset)
            blocks.append(
                (scale.unsqueeze(2) * harmonic.unsqueeze(1)).reshape(
                    numbers.numel(), STUB_EQUIVARIANT_CHANNELS * (2 * degree + 1)
                )
            )
        return torch.cat(blocks, dim=1)

    def _features(
        self, numbers, batch, charge, spin, positions, supplied=None
    ):
        invariant = _feature_values(numbers, supplied)
        equivariant = self._equivariant(numbers, invariant, positions, batch)
        schema = (
            f"stub:mace=0.3.16:mode={self.feature_mode}:adapter=stub:"
            f"inv=16:equiv={equivariant.shape[1]}:layers=4"
        )
        if self.equivariant_irreps:
            schema = f"{schema}:irreps={self.equivariant_irreps}"
        return MACEAtomicFeatures(
            invariant=invariant,
            equivariant=equivariant,
            batch=batch,
            atomic_numbers=numbers,
            total_charge=charge.to(invariant),
            total_spin=spin.to(invariant),
            feature_schema=schema,
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
            batch.RA,
            getattr(batch, "stub_invariant_A", None),
        )
        features_b = self._features(
            batch.ZB,
            batch.molecule_ind_B,
            batch.total_charge_B,
            batch.total_spin_B,
            batch.RB,
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
        features = self._features(
            numbers, batch, total_charge, total_spin, positions
        )
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
    directional_degree = DIRECTIONAL_DEGREES[pair_mode]
    featurizer = StubFeaturizer(
        feature_mode,
        equivariant_irreps=STUB_IRREPS if directional_degree is not None else "",
    )
    provider = StubPropertyProvider(provider_kind)
    ap3 = APNet3D3_AtomType_MPNN(
        dimer_prop_model=None,
        use_precomputed_classical=True,
        no_disp_nn=no_disp,
    )
    pair_kwargs = {}
    # Derived, not listed: a route names itself to the pair core whenever its
    # id is not recoverable from the pair mode, which is what separates
    # ``hybrid-h3l3q`` from ``hybrid-h3l3``.
    if route != f"hybrid-{pair_mode}":
        pair_kwargs["architecture_id"] = route
    if directional_degree is not None:
        pair_kwargs["mace_equivariant_dim"] = STUB_EQUIVARIANT_CHANNELS
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
