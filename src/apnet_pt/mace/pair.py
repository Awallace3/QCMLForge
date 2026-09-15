"""Shared MACE/AP3D3 short-range residual core."""

from __future__ import annotations

import math
from typing import Any

import torch

from .schema import AtomicPropertyBundle, MACEAtomicFeatures

CANONICAL_AP3D3_DIMENSIONS = {
    "n_message": 3,
    "n_rbf": 8,
    "n_neuron": 128,
    "n_embed": 8,
    "r_cut_im": 8.0,
    "r_cut": 5.0,
}

# The directional slot AP3 has always handed the readout: one contracted float
# per (message, embedding) pair. ``DIRECTIONAL_WIDTH_OVERRIDES`` is what lets a
# route ask for more.
CANONICAL_DIRECTIONAL_WIDTH = (
    CANONICAL_AP3D3_DIMENSIONS["n_message"] * CANONICAL_AP3D3_DIMENSIONS["n_embed"]
)

CANONICAL_PAIR_FEATURE_MODES = {
    "h1": "final-layer-scalars",
    "h2": "all-scalars+norms",
    "h3": "all-scalars+norms",
    "h3l1": "all-scalars+norms",
    "h3l3": "all-scalars+norms",
}

PAIR_ARCHITECTURE_IDS = {
    "h1": "MACE-AP3D3-H1",
    "h2": "MACE-AP3D3-H2",
    "h3": "MACE-AP3D3-H3",
    "h3l1": "MACE-AP3D3-H3L1",
    "h3l3": "MACE-AP3D3-H3L3",
}

PAIR_ROUTE_CONFIGS = {
    "direct-polar": ("h1", "all-scalars+norms"),
    "hybrid-h1": ("h1", "final-layer-scalars"),
    "hybrid-h2": ("h2", "all-scalars+norms"),
    "hybrid-h3": ("h3", "all-scalars+norms"),
    "hybrid-h3l1": ("h3l1", "all-scalars+norms"),
    "hybrid-h3l3": ("h3l3", "all-scalars+norms"),
    "hybrid-h3l3q": ("h3l3", "all-scalars+norms"),
    "hybrid-h3l3w112": ("h3l3", "all-scalars+norms"),
    "hybrid-h3l3p": ("h3l3", "all-scalars+norms"),
    "hybrid-h3l3w112p": ("h3l3", "all-scalars+norms"),
    "atomhead": ("h1", "all-scalars+norms"),
}

# Pair modes that bypass the AP3 intramonomer update stack entirely.
BYPASS_PAIR_MODES = frozenset({"h2", "h3", "h3l1", "h3l3"})

# Scalars ``monomer_conditioning`` appends to each half of the pair feature:
# the monomer's formal charge, that charge spread over its atoms, and its
# unpaired-electron count. Three per monomer, both monomers on every edge.
MONOMER_CONDITIONING_SCALARS = ("formal_charge", "charge_per_atom", "n_unpaired")
MONOMER_CONDITIONING_WIDTH = 2 * len(MONOMER_CONDITIONING_SCALARS)

# Routes that carry the monomer conditioning block. It is orthogonal to the
# pair mode -- ``hybrid-h3l3q`` is ``hybrid-h3l3`` plus these six scalars and
# nothing else -- but it still gets its own architecture id rather than a bare
# constructor flag, because the readout width and the physics the head sees
# both change, and ``expected_resume_semantics`` compares architecture ids. A
# shared id would let a charge-aware run resume from charge-blind weights.
MONOMER_CONDITIONING_ARCHITECTURES = frozenset({"hybrid-h3l3q"})

# Width of the per-edge directional slot each route hands the readout. The
# canonical value is ``n_message * n_embed`` = 24, which is what AP3's own l=1
# directional messages happen to occupy and what every route used before the
# H3 family arrived. It is a hard bottleneck: the contraction compresses all
# 512 PolarMACE channels of one degree into that many floats per edge, and it
# was never resized when the consumed degree went from 1 to 2 to 3, so every
# degree comparison so far has been run through a slot sized for the smallest
# of them. Routes that widen it name themselves here; everything absent keeps
# the canonical width, so no existing checkpoint changes shape.
DIRECTIONAL_WIDTH_OVERRIDES = {
    "hybrid-h3l3w112": 112,
    "hybrid-h3l3w112p": 112,
}

# Routes that give every per-component readout its own directional projection
# instead of sharing one. Electrostatics, exchange, induction and dispersion
# want different angular information out of the same equivariant block --
# exchange is governed by overlap along the axis, induction by the field --
# and a shared projection forces one compromise on all four. Each projection
# stays bias-free and channel-only, so equivariance and the hAB/hBA mirror are
# unaffected; only the number of independent channel mixings changes.
PER_COMPONENT_DIRECTIONAL_ARCHITECTURES = frozenset(
    {"hybrid-h3l3p", "hybrid-h3l3w112p"}
)

# Readout heads the AP3 core exposes, in the order ``readouts`` concatenates
# them. ``disp`` is absent when ``no_disp_nn`` is set.
PAIR_READOUT_COMPONENTS = ("elst", "exch", "indu", "disp")

# Spherical-harmonic degree each H3 variant contracts into the AP3 directional
# slot. H2 (degree ``None``) leaves that slot zeroed, which is the ablation H3
# is designed to undo: MACE's equivariant channels are reduced to norms by the
# ``all-scalars+norms`` invariant, so without this the route carries no atomic
# anisotropy at all.
DIRECTIONAL_DEGREES = {"h1": None, "h2": None, "h3": 2, "h3l1": 1, "h3l3": 3}

# MACE feeds ``_permute_to_e3nn_convention(vectors)`` to its spherical
# harmonics, i.e. Cartesian ``(x, y, z) -> (y, z, x)``. Any contraction against
# MACE's equivariant output must apply the identical permutation or the result
# is not rotationally invariant -- and, being merely wrong rather than broken,
# would still train. ``tests/test_mace_h3_pair.py`` asserts this tuple against
# the MACE function itself so the two cannot drift.
MACE_E3NN_AXIS_PERMUTATION = (1, 2, 0)


def real_spherical_harmonics(degree: int, vectors: torch.Tensor) -> torch.Tensor:
    """``e3nn.o3.spherical_harmonics(degree, v, True, "component")`` in plain torch.

    e3nn evaluates its harmonics inside a ``@torch.jit.script`` function. That
    function is process-global mutable state: a TorchScript graph that has only
    been profiled once can be left in a broken profile by unrelated work
    elsewhere in the process, after which every later call raises
    ``RuntimeError: bad optional access`` from a call site that never changed.
    This route evaluates harmonics on the pair forward path, so a hazard that
    depends on how many times the graph happened to run before is not one to
    carry. The l <= 3 polynomials are a dozen lines; spelling them out removes the
    dependency instead of ordering around it.

    Component ordering, sign convention and the ``sqrt(2l+1)`` component
    normalization all follow e3nn exactly; ``tests/test_mace_h3_pair.py`` pins
    the two against each other elementwise so they cannot drift.
    """
    unit = torch.nn.functional.normalize(vectors, dim=-1)
    x = unit[..., 0]
    y = unit[..., 1]
    z = unit[..., 2]
    if degree == 1:
        # sqrt(3) * [x, y, z]
        return math.sqrt(3.0) * unit
    if degree == 2:
        root3 = math.sqrt(3.0)
        components = torch.stack(
            [
                root3 * x * z,
                root3 * x * y,
                y.pow(2) - 0.5 * (x.pow(2) + z.pow(2)),
                root3 * y * z,
                0.5 * root3 * (z.pow(2) - x.pow(2)),
            ],
            dim=-1,
        )
        return math.sqrt(5.0) * components
    if degree == 3:
        root3 = math.sqrt(3.0)
        y2 = y.pow(2)
        x2z2 = x.pow(2) + z.pow(2)
        # e3nn builds l=3 by recursion off its own l=2 block, so these two are
        # reused verbatim rather than re-derived.
        sh_2_0 = root3 * x * z
        sh_2_4 = 0.5 * root3 * (z.pow(2) - x.pow(2))
        components = torch.stack(
            [
                math.sqrt(5.0 / 6.0) * (sh_2_0 * z + sh_2_4 * x),
                math.sqrt(5.0) * sh_2_0 * y,
                math.sqrt(3.0 / 8.0) * (4.0 * y2 - x2z2) * x,
                0.5 * y * (2.0 * y2 - 3.0 * x2z2),
                math.sqrt(3.0 / 8.0) * z * (4.0 * y2 - x2z2),
                math.sqrt(5.0) * sh_2_4 * y,
                math.sqrt(5.0 / 6.0) * (sh_2_4 * z - sh_2_0 * x),
            ],
            dim=-1,
        )
        return math.sqrt(7.0) * components
    raise ValueError(
        f"real_spherical_harmonics supports degree 1, 2 and 3, got {degree}"
    )


class MACEPairResidualCore(torch.nn.Module):
    """Canonical H1/H2/H3 projection into the existing AP3D3 pair head.

    H1 retains the AP3 intramonomer invariant and directional updates. H2
    bypasses those updates and constructs pairs directly from projected MACE
    invariants. H3 is H2 plus MACE's equivariant features contracted into the
    directional slot H2 leaves zeroed -- ``h3`` takes degree 2, ``h3l1``
    degree 1, ``h3l3`` degree 3 -- which separates "no intramonomer message passing" from "no
    atomic anisotropy", the two things H2 ablates together. Every mode uses
    the same projection output width, AP3 pair feature width, bidirectional
    readouts, cutoff, and dimer aggregation.
    """

    def __init__(
        self,
        ap3_core: torch.nn.Module,
        *,
        mace_feature_dim: int,
        pair_mode: str = "h1",
        feature_mode: str | None = None,
        architecture_id: str | None = None,
        mace_equivariant_dim: int | None = None,
        monomer_conditioning: bool | None = None,
    ) -> None:
        super().__init__()
        if pair_mode not in CANONICAL_PAIR_FEATURE_MODES:
            raise ValueError(f"unsupported MACE pair mode: {pair_mode}")
        if architecture_id is None:
            expected_feature_mode = CANONICAL_PAIR_FEATURE_MODES[pair_mode]
            resolved_architecture_id = PAIR_ARCHITECTURE_IDS[pair_mode]
            error_prefix = f"canonical {pair_mode}"
        else:
            if architecture_id not in PAIR_ROUTE_CONFIGS:
                raise ValueError(
                    f"unsupported MACE pair architecture: {architecture_id}"
                )
            expected_pair_mode, expected_feature_mode = PAIR_ROUTE_CONFIGS[
                architecture_id
            ]
            if pair_mode != expected_pair_mode:
                raise ValueError(
                    f"{architecture_id} requires pair mode {expected_pair_mode}"
                )
            resolved_architecture_id = architecture_id
            error_prefix = architecture_id
        if feature_mode is None:
            feature_mode = expected_feature_mode
        if feature_mode != expected_feature_mode:
            raise ValueError(f"{error_prefix} requires {expected_feature_mode}")
        if mace_feature_dim < 1:
            raise ValueError("mace_feature_dim must be positive")
        for name, expected in CANONICAL_AP3D3_DIMENSIONS.items():
            actual = getattr(ap3_core, name, None)
            if actual != expected:
                raise ValueError(
                    f"canonical {pair_mode} requires AP3 {name}={expected}, "
                    f"got {actual}"
                )
        directional_degree = DIRECTIONAL_DEGREES[pair_mode]
        if directional_degree is None:
            if mace_equivariant_dim is not None:
                raise ValueError(
                    f"{error_prefix} has no directional contraction and takes no "
                    "mace_equivariant_dim"
                )
        elif mace_equivariant_dim is None or mace_equivariant_dim < 1:
            raise ValueError(
                f"{error_prefix} requires a positive mace_equivariant_dim "
                f"(channel count of the l={directional_degree} block)"
            )
        # The route table owns whether monomer conditioning is on; the keyword
        # exists so a test can state the expectation, not so a caller can pair
        # an id with a width it does not describe.
        implied_conditioning = (
            resolved_architecture_id in MONOMER_CONDITIONING_ARCHITECTURES
        )
        if monomer_conditioning is None:
            monomer_conditioning = implied_conditioning
        elif bool(monomer_conditioning) != implied_conditioning:
            raise ValueError(
                f"{error_prefix} sets monomer_conditioning="
                f"{implied_conditioning}; it is a property of the route, not a "
                "free constructor flag"
            )
        self.ap3_core = ap3_core
        self.mace_feature_dim = mace_feature_dim
        self.pair_mode = pair_mode
        self.feature_mode = feature_mode
        self.architecture_id = resolved_architecture_id
        self.directional_degree = directional_degree
        self.mace_equivariant_dim = mace_equivariant_dim
        self.monomer_conditioning = bool(monomer_conditioning)
        self.bypass_intra_updates = pair_mode in BYPASS_PAIR_MODES
        self.directional_width = DIRECTIONAL_WIDTH_OVERRIDES.get(
            resolved_architecture_id, CANONICAL_DIRECTIONAL_WIDTH
        )
        self.per_component_directional = (
            resolved_architecture_id in PER_COMPONENT_DIRECTIONAL_ARCHITECTURES
        )
        if directional_degree is None and (
            self.directional_width != CANONICAL_DIRECTIONAL_WIDTH
            or self.per_component_directional
        ):
            raise ValueError(
                f"{error_prefix} has no directional contraction, so it cannot "
                "widen or split the directional slot"
            )
        self.directional_components = (
            self._resolve_readout_components()
            if self.per_component_directional
            else ()
        )
        self.h0_projection = torch.nn.Linear(
            mace_feature_dim, CANONICAL_AP3D3_DIMENSIONS["n_embed"]
        )
        # Bias-free and applied to the channel axis only: adding a bias or
        # mixing the 2l+1 components would break equivariance, which the loss
        # would absorb rather than report. That holds per component too -- the
        # split changes how many independent channel mixings the block gets,
        # not how any one of them touches the angular index.
        if directional_degree is None:
            self.directional_projection = None
            self.directional_projections = None
        elif self.per_component_directional:
            # The shared attribute stays ``None`` so a per-component
            # checkpoint cannot be silently loaded into a shared-projection
            # route, or the reverse: the parameter names do not overlap.
            self.directional_projection = None
            self.directional_projections = torch.nn.ModuleDict(
                {
                    component: torch.nn.Linear(
                        mace_equivariant_dim, self.directional_width, bias=False
                    )
                    for component in self.directional_components
                }
            )
        else:
            self.directional_projection = torch.nn.Linear(
                mace_equivariant_dim, self.directional_width, bias=False
            )
            self.directional_projections = None
        self._materialize_lazy_layers()
        # MACE projections replace the legacy element embedding on every route.
        self.ap3_core.embed_layer.requires_grad_(False)
        if self.bypass_intra_updates:
            # H2/H3 deliberately bypass the complete intramonomer update stack.
            self.ap3_core.update_layers.requires_grad_(False)
            self.ap3_core.directional_layers.requires_grad_(False)
            self.ap3_core.distance_layer.requires_grad_(False)
        self.last_h_ab: torch.Tensor | None = None
        self.last_h_ba: torch.Tensor | None = None

    def _resolve_readout_components(self) -> tuple[str, ...]:
        """Name the readout heads this AP3 core actually exposes."""

        return tuple(
            component
            for component in PAIR_READOUT_COMPONENTS
            if hasattr(self.ap3_core, f"readout_layer_{component}")
            and not (component == "disp" and self.ap3_core.no_disp_nn)
        )

    def _materialize_lazy_layers(self) -> None:
        """Initialize canonical lazy layers before optimization/checkpointing."""

        message_width = (
            4
            * CANONICAL_AP3D3_DIMENSIONS["n_embed"]
            * CANONICAL_AP3D3_DIMENSIONS["n_rbf"]
            + 4 * CANONICAL_AP3D3_DIMENSIONS["n_embed"]
            + CANONICAL_AP3D3_DIMENSIONS["n_rbf"]
        )
        sample = self.h0_projection.weight.new_empty((0, message_width))
        layers = tuple(self.ap3_core.update_layers) + tuple(
            self.ap3_core.directional_layers
        )
        for layer in layers:
            input_layer = layer[0]
            if isinstance(input_layer, torch.nn.LazyLinear) and (
                input_layer.has_uninitialized_params()
            ):
                input_layer.initialize_parameters(sample)
        pair_width = (
            2 * (self.ap3_core.n_message + 1) * self.ap3_core.n_embed
            + 6
            + self.ap3_core.n_rbf
            # Both halves of the edge contribute one directional block, and a
            # per-component route gives each head its own pair of blocks at the
            # same width -- so the readout input width is identical either way.
            + 2 * self.directional_width
        )
        if self.monomer_conditioning:
            pair_width += MONOMER_CONDITIONING_WIDTH
        pair_sample = self.h0_projection.weight.new_empty((0, pair_width))
        readouts = [
            self.ap3_core.readout_layer_elst,
            self.ap3_core.readout_layer_exch,
            self.ap3_core.readout_layer_indu,
        ]
        if hasattr(self.ap3_core, "readout_layer_disp"):
            readouts.append(self.ap3_core.readout_layer_disp)
        for layer in readouts:
            input_layer = layer[0]
            if isinstance(input_layer, torch.nn.LazyLinear) and (
                input_layer.has_uninitialized_params()
            ):
                input_layer.initialize_parameters(pair_sample)

    def get_config(self) -> dict[str, str | int]:
        """Return the identity required to reconstruct this pair adapter."""

        config: dict[str, str | int] = {
            "architecture_id": self.architecture_id,
            "pair_mode": self.pair_mode,
            "feature_mode": self.feature_mode,
            "mace_feature_dim": self.mace_feature_dim,
        }
        if self.directional_degree is not None:
            # Only the H3 family adds these, so H1/H2 checkpoints written
            # before the family existed still satisfy the strict comparison in
            # ``set_extra_state``.
            config["directional_degree"] = self.directional_degree
            config["mace_equivariant_dim"] = self.mace_equivariant_dim
        if self.directional_width != CANONICAL_DIRECTIONAL_WIDTH:
            # Written only when widened, for the same reason as the keys above:
            # every checkpoint from before the slot was resizable carries the
            # canonical 24 implicitly and must still satisfy the strict
            # comparison in ``set_extra_state``.
            config["directional_width"] = self.directional_width
        if self.per_component_directional:
            config["per_component_directional"] = True
        if self.monomer_conditioning:
            # Added only when on, for the same reason as the H3 keys above: a
            # checkpoint written before the flag existed must still satisfy the
            # strict comparison in ``set_extra_state``. Present-and-true is the
            # only state that changes the readout width, so present-and-true is
            # the only state that has to be recorded.
            config["monomer_conditioning"] = True
        return config

    def get_extra_state(self) -> dict[str, str | int]:
        """Persist topology identity with low-level state dictionaries."""

        return self.get_config()

    def set_extra_state(self, state: dict[str, str | int]) -> None:
        """Reject state dictionaries from a different canonical topology."""

        if state != self.get_config():
            raise RuntimeError(
                "MACE pair architecture configuration does not match state_dict"
            )

    def _validate_features(
        self,
        features: MACEAtomicFeatures,
        properties: AtomicPropertyBundle,
        expected_numbers: torch.Tensor,
        monomer: str,
    ) -> None:
        schema_token = f":mode={self.feature_mode}:"
        if schema_token not in features.feature_schema:
            raise ValueError(
                f"canonical {self.pair_mode} requires {self.feature_mode} features"
            )
        expected_atoms = expected_numbers.numel()
        if features.invariant.shape != (expected_atoms, self.mace_feature_dim):
            raise ValueError(f"monomer {monomer} MACE feature shape is incompatible")
        if not torch.equal(features.atomic_numbers, expected_numbers):
            raise ValueError(f"monomer {monomer} MACE atom order is incompatible")
        if self.directional_degree is not None:
            block = features.equivariant_degree(self.directional_degree)
            if block.shape[1] != self.mace_equivariant_dim:
                raise ValueError(
                    f"monomer {monomer} l={self.directional_degree} block has "
                    f"{block.shape[1]} channels, expected "
                    f"{self.mace_equivariant_dim}"
                )
        if properties.natom != expected_atoms:
            raise ValueError(f"monomer {monomer} properties do not align with atoms")
        if (
            properties.q.device != features.invariant.device
            or expected_numbers.device != features.invariant.device
        ):
            raise ValueError(
                "MACE pair batch, features, and properties must share a device"
            )

    def _pair_directional(
        self,
        batch: Any,
        features_a: MACEAtomicFeatures,
        features_b: MACEAtomicFeatures,
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | dict[str, tuple[torch.Tensor, torch.Tensor]]
    ):
        """Contract MACE's l-th equivariant block onto the interatomic axis.

        Returns the two per-edge tensors AP3 would otherwise build from its own
        directional messages, at this route's directional width -- or, on a
        per-component route, one such pair per readout head, keyed by head
        name. Either way every head sees the same pair feature width.

        The channel projection is applied before the contraction: it costs
        ``n_atom * (2l+1) * C * F`` instead of the per-edge equivalent, and
        because it never mixes the ``2l+1`` components it commutes with the
        rotation, so the two orderings are equivalent up to arithmetic.
        """

        degree = self.directional_degree
        block_a = features_a.equivariant_degree(degree)
        block_b = features_b.equivariant_degree(degree)

        source = batch.e_ABsr_source
        target = batch.e_ABsr_target
        distance, displacement = self.ap3_core.get_distances(
            batch.RA, batch.RB, source, target
        )
        unit = displacement / distance.unsqueeze(1)
        # Match MACE's internal frame before evaluating harmonics.
        unit = unit[:, list(MACE_E3NN_AXIS_PERMUTATION)]

        harmonics_a = real_spherical_harmonics(degree, unit)
        # AP3 gives the two monomers opposite axis polarity. At odd l the
        # harmonic is parity-odd, so this reproduces its own +u/-u convention
        # exactly (l=1 and l=3); at even l the harmonic is parity-even, so A
        # and B share the angular factor and only the per-atom channels
        # distinguish them.
        harmonics_b = real_spherical_harmonics(degree, -unit)

        def contract(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            # [n_atom, channel, m] x [out, channel] -> [n_atom, m, out]
            projected_a = torch.einsum("acm,fc->amf", block_a, weight)
            projected_b = torch.einsum("acm,fc->amf", block_b, weight)
            return (
                torch.einsum(
                    "amf,am->af", projected_a.index_select(0, source), harmonics_a
                ),
                torch.einsum(
                    "amf,am->af", projected_b.index_select(0, target), harmonics_b
                ),
            )

        if self.directional_projections is None:
            return contract(self.directional_projection.weight)
        # The geometry above is shared; only the channel mixing differs per
        # head, so the harmonics are evaluated once no matter how many
        # projections consume them.
        return {
            component: contract(projection.weight)
            for component, projection in self.directional_projections.items()
        }

    def _pair_monomer_scalars(
        self, batch: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Broadcast each monomer's charge and spin onto every pair edge.

        The AP3 pair feature already carries the *predicted* per-atom monopole
        of both edge atoms, but nothing tells the readout what the monomer
        those atoms belong to actually is: a -2 anion and a neutral can present
        the same local charge. Formal charge, charge per atom and unpaired
        electron count are exactly the monomer-level inputs an MLIP handed
        fragment charges gets for free, and are what the PLA15 control showed
        is worth most of MACE-POLAR-1-S's lead there.

        Spin enters as ``multiplicity - 1`` so a closed-shell monomer
        contributes an exact zero rather than a constant that duplicates the
        readout bias. Every dimer in the current training data is a pair of
        singlets, so today this channel is identically zero by construction and
        the arm is a pure charge lever; it starts carrying signal the moment
        open-shell records appear, without a second architecture.
        """

        for name in ("total_charge_A", "total_charge_B",
                     "total_spin_A", "total_spin_B"):
            if getattr(batch, name, None) is None:
                raise ValueError(
                    f"monomer conditioning requires batch.{name}; refusing to "
                    "substitute a default charge or multiplicity"
                )
        dtype = batch.RA.dtype
        charge_a = batch.total_charge_A.to(dtype).reshape(-1)
        charge_b = batch.total_charge_B.to(dtype).reshape(-1)
        unpaired_a = batch.total_spin_A.to(dtype).reshape(-1) - 1.0
        unpaired_b = batch.total_spin_B.to(dtype).reshape(-1) - 1.0
        ndimer = charge_a.numel()
        natom_a = torch.bincount(
            batch.molecule_ind_A, minlength=ndimer
        ).to(dtype).clamp(min=1.0)
        natom_b = torch.bincount(
            batch.molecule_ind_B, minlength=ndimer
        ).to(dtype).clamp(min=1.0)
        per_dimer_a = torch.stack(
            [charge_a, charge_a / natom_a, unpaired_a], dim=1
        )
        per_dimer_b = torch.stack(
            [charge_b, charge_b / natom_b, unpaired_b], dim=1
        )
        edge_dimer = batch.dimer_ind
        edge_a = per_dimer_a.index_select(0, edge_dimer)
        edge_b = per_dimer_b.index_select(0, edge_dimer)
        # A-then-B for hAB, B-then-A for hBA: the same mirror AP3 applies to
        # every other pair feature, so the two readout directions stay exact
        # images of each other and the residual remains swap-symmetric.
        return (
            torch.cat([edge_a, edge_b], dim=1),
            torch.cat([edge_b, edge_a], dim=1),
        )

    def forward(
        self,
        batch: Any,
        features_a: MACEAtomicFeatures,
        features_b: MACEAtomicFeatures,
        props_a: AtomicPropertyBundle,
        props_b: AtomicPropertyBundle,
    ) -> torch.Tensor:
        """Return one four-component short-range residual per dimer."""

        self._validate_features(features_a, props_a, batch.ZA, "A")
        self._validate_features(features_b, props_b, batch.ZB, "B")
        h0_a = self.h0_projection(features_a.invariant)
        h0_b = self.h0_projection(features_b.invariant)
        injected_directional = None
        if self.directional_degree is not None:
            injected_directional = self._pair_directional(
                batch, features_a, features_b
            )
        injected_scalars = None
        if self.monomer_conditioning:
            injected_scalars = self._pair_monomer_scalars(batch)
        result = self.ap3_core(
            batch,
            initial_atom_states=(h0_a, h0_b),
            atomic_properties=(props_a, props_b),
            residual_only=True,
            pair_energy_envelope=True,
            bypass_intra_updates=self.bypass_intra_updates,
            injected_pair_directional=injected_directional,
            injected_pair_scalars=injected_scalars,
        )
        residual = result[0]
        if residual.ndim == 2 and residual.shape[1] == 3:
            residual = torch.cat(
                (residual, residual.new_zeros((residual.shape[0], 1))), dim=1
            )
        expected_dimers = batch.total_charge_A.numel()
        if residual.shape != (expected_dimers, 4):
            raise RuntimeError(
                f"canonical {self.pair_mode} residual must have shape [n_dimer, 4]"
            )
        if not torch.isfinite(residual).all():
            raise RuntimeError(
                f"canonical {self.pair_mode} residual contains non-finite values"
            )
        self.last_h_ab = result[5].detach()
        self.last_h_ba = result[6].detach()
        return residual
