"""Shared MACE/AP3D3 short-range residual core."""

from __future__ import annotations

from typing import Any

import torch

from .encoder import _e3nn_o3
from .schema import AtomicPropertyBundle, MACEAtomicFeatures


CANONICAL_AP3D3_DIMENSIONS = {
    "n_message": 3,
    "n_rbf": 8,
    "n_neuron": 128,
    "n_embed": 8,
    "r_cut_im": 8.0,
    "r_cut": 5.0,
}

CANONICAL_PAIR_FEATURE_MODES = {
    "h1": "final-layer-scalars",
    "h2": "all-scalars+norms",
    "h3": "all-scalars+norms",
    "h3l1": "all-scalars+norms",
}

PAIR_ARCHITECTURE_IDS = {
    "h1": "MACE-AP3D3-H1",
    "h2": "MACE-AP3D3-H2",
    "h3": "MACE-AP3D3-H3",
    "h3l1": "MACE-AP3D3-H3L1",
}

PAIR_ROUTE_CONFIGS = {
    "direct-polar": ("h1", "all-scalars+norms"),
    "hybrid-h1": ("h1", "final-layer-scalars"),
    "hybrid-h2": ("h2", "all-scalars+norms"),
    "hybrid-h3": ("h3", "all-scalars+norms"),
    "hybrid-h3l1": ("h3l1", "all-scalars+norms"),
    "atomhead": ("h1", "all-scalars+norms"),
}

# Pair modes that bypass the AP3 intramonomer update stack entirely.
BYPASS_PAIR_MODES = frozenset({"h2", "h3", "h3l1"})

# Spherical-harmonic degree each H3 variant contracts into the AP3 directional
# slot. H2 (degree ``None``) leaves that slot zeroed, which is the ablation H3
# is designed to undo: MACE's equivariant channels are reduced to norms by the
# ``all-scalars+norms`` invariant, so without this the route carries no atomic
# anisotropy at all.
DIRECTIONAL_DEGREES = {"h1": None, "h2": None, "h3": 2, "h3l1": 1}

# MACE feeds ``_permute_to_e3nn_convention(vectors)`` to its spherical
# harmonics, i.e. Cartesian ``(x, y, z) -> (y, z, x)``. Any contraction against
# MACE's equivariant output must apply the identical permutation or the result
# is not rotationally invariant -- and, being merely wrong rather than broken,
# would still train. ``tests/test_mace_h3_pair.py`` asserts this tuple against
# the MACE function itself so the two cannot drift.
MACE_E3NN_AXIS_PERMUTATION = (1, 2, 0)


class MACEPairResidualCore(torch.nn.Module):
    """Canonical H1/H2/H3 projection into the existing AP3D3 pair head.

    H1 retains the AP3 intramonomer invariant and directional updates. H2
    bypasses those updates and constructs pairs directly from projected MACE
    invariants. H3 is H2 plus MACE's equivariant features contracted into the
    directional slot H2 leaves zeroed -- ``h3`` takes degree 2, ``h3l1``
    degree 1 -- which separates "no intramonomer message passing" from "no
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
        self.ap3_core = ap3_core
        self.mace_feature_dim = mace_feature_dim
        self.pair_mode = pair_mode
        self.feature_mode = feature_mode
        self.architecture_id = resolved_architecture_id
        self.directional_degree = directional_degree
        self.mace_equivariant_dim = mace_equivariant_dim
        self.bypass_intra_updates = pair_mode in BYPASS_PAIR_MODES
        self.h0_projection = torch.nn.Linear(
            mace_feature_dim, CANONICAL_AP3D3_DIMENSIONS["n_embed"]
        )
        if directional_degree is None:
            self.directional_projection = None
        else:
            # Bias-free and applied to the channel axis only: adding a bias or
            # mixing the 2l+1 components would break equivariance, which the
            # loss would absorb rather than report.
            self.directional_projection = torch.nn.Linear(
                mace_equivariant_dim,
                CANONICAL_AP3D3_DIMENSIONS["n_message"]
                * CANONICAL_AP3D3_DIMENSIONS["n_embed"],
                bias=False,
            )
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
            + 2 * self.ap3_core.n_message * self.ap3_core.n_embed
        )
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Contract MACE's l-th equivariant block onto the interatomic axis.

        Returns the two per-edge tensors AP3 would otherwise build from its own
        directional messages, at exactly the AP3 directional width, so the pair
        feature and readout widths are bit-identical across H1, H2, and H3.

        The channel projection is applied before the contraction: it costs
        ``n_atom * (2l+1) * C * F`` instead of the per-edge equivalent, and
        because it never mixes the ``2l+1`` components it commutes with the
        rotation, so the two orderings are equivalent up to arithmetic.
        """

        degree = self.directional_degree
        block_a = features_a.equivariant_degree(degree)
        block_b = features_b.equivariant_degree(degree)
        weight = self.directional_projection.weight
        # [n_atom, channel, m] x [out, channel] -> [n_atom, m, out]
        projected_a = torch.einsum("acm,fc->amf", block_a, weight)
        projected_b = torch.einsum("acm,fc->amf", block_b, weight)

        source = batch.e_ABsr_source
        target = batch.e_ABsr_target
        distance, displacement = self.ap3_core.get_distances(
            batch.RA, batch.RB, source, target
        )
        unit = displacement / distance.unsqueeze(1)
        # Match MACE's internal frame before evaluating harmonics.
        unit = unit[:, list(MACE_E3NN_AXIS_PERMUTATION)]

        o3 = _e3nn_o3()
        harmonics_a = o3.spherical_harmonics(
            degree, unit, normalize=True, normalization="component"
        )
        # AP3 gives the two monomers opposite axis polarity. At l=1 this
        # reproduces its own +u/-u convention exactly; at even l the harmonic
        # is parity-even, so A and B share the angular factor and only the
        # per-atom channels distinguish them.
        harmonics_b = o3.spherical_harmonics(
            degree, -unit, normalize=True, normalization="component"
        )

        directional_a = torch.einsum(
            "amf,am->af", projected_a.index_select(0, source), harmonics_a
        )
        directional_b = torch.einsum(
            "amf,am->af", projected_b.index_select(0, target), harmonics_b
        )
        return directional_a, directional_b

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
        result = self.ap3_core(
            batch,
            initial_atom_states=(h0_a, h0_b),
            atomic_properties=(props_a, props_b),
            residual_only=True,
            pair_energy_envelope=True,
            bypass_intra_updates=self.bypass_intra_updates,
            injected_pair_directional=injected_directional,
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
