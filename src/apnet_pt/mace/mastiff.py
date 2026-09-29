"""Frozen-MACE atomic heads for positive, frame-free eq-l3 exchange."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

import torch

from ..mastiff_exponent import slater_exponent_exchange
from .pair import MACE_E3NN_AXIS_PERMUTATION, real_spherical_harmonics
from .schema import MACEAtomicFeatures, _parse_irreps, irreps_degree_slice


def _bounded_degree(raw: torch.Tensor, rho: float) -> torch.Tensor:
    # Algebraically rho*c/sqrt(1+||c||²), avoiding overflow from c².
    scale = raw.abs().amax(-1, keepdim=True).clamp_min(1)
    scaled = raw / scale
    denominator = (
        scale.reciprocal().square() + scaled.square().sum(-1, keepdim=True)
    ).sqrt()
    return rho * scaled / denominator


class MACEMASTIFFExchange(torch.nn.Module):
    """Replace a learned eq-l3 feature producer with detached MACE features.

    Scalar descriptors predict log-A/log-B corrections to explicit element
    baselines. Bias-free channel mixing of the natural-parity l=1,2,3 blocks
    predicts global-frame coefficient vectors, independently norm-bounded by
    ``rho``. Contracting them with Racah-normalized harmonics modifies the
    decay exponent, not the exchange amplitude.

    This is an exchange replacement, not a parent-preserving additive model.
    Features are detached: geometry derivatives are not complete molecular
    forces. The caller supplies each desired intermonomer pair exactly once;
    the kernel has no intermonomer cutoff. Use all pairs for full exchange.
    Positions are angstrom, A baselines sqrt(kcal/mol), B inverse angstrom.
    Checkpoints bind descriptor schema and external artifact digest; callers
    must still verify that input features came from that artifact.
    """

    def __init__(
        self,
        *,
        mace_feature_dim: int,
        feature_schema: str,
        mace_checkpoint_sha256: str,
        element_baselines: Mapping[int | str, tuple[float, float]],
        hidden_width: int = 64,
        rho: float = 0.4,
        anisotropy: bool = True,
        readout_init_scale: float = 0.01,
    ) -> None:
        super().__init__()
        for value in (mace_feature_dim, hidden_width):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(
                    "feature and hidden dimensions must be positive integers"
                )
        if not isinstance(anisotropy, bool):
            raise TypeError("anisotropy must be boolean")
        if isinstance(rho, bool) or not math.isfinite(rho) or not 0 < rho < 1:
            raise ValueError("rho must be finite and strictly between zero and one")
        if (
            isinstance(readout_init_scale, bool)
            or not math.isfinite(readout_init_scale)
            or readout_init_scale < 0
        ):
            raise ValueError("readout_init_scale must be finite and nonnegative")
        if not re.fullmatch(r"[0-9a-f]{64}", mace_checkpoint_sha256):
            raise ValueError("mace_checkpoint_sha256 must be a lowercase SHA-256")
        _, separator, irreps = feature_schema.partition(":irreps=")
        if not separator:
            raise ValueError("feature_schema must specify an exact irrep layout")
        multiplicities = []
        for degree in (1, 2, 3):
            _, _, channels = irreps_degree_slice(irreps, degree)
            parity = "o" if degree % 2 else "e"
            if channels < 1 or any(
                p != parity
                for _, term_degree, p in _parse_irreps(irreps)
                if term_degree == degree
            ):
                raise ValueError("eq-l3 requires nonempty natural parity blocks")
            multiplicities.append(channels)
        baselines = {}
        for z, values in element_baselines.items():
            if isinstance(z, bool) or str(z) != str(int(z)) or not 1 <= int(z) <= 118:
                raise ValueError("baseline elements must be atomic numbers 1..118")
            if len(values) != 2 or any(
                isinstance(v, bool) or not math.isfinite(v) or v <= 0 for v in values
            ):
                raise ValueError("element baselines must contain positive finite A/B")
            if str(int(z)) in baselines:
                raise ValueError("duplicate element baseline")
            baselines[str(int(z))] = tuple(float(v) for v in values)
        if not baselines:
            raise ValueError("explicit element baselines are required")
        self.config = {
            "mace_feature_dim": mace_feature_dim,
            "feature_schema": feature_schema,
            "mace_checkpoint_sha256": mace_checkpoint_sha256,
            "element_baselines": baselines,
            "hidden_width": hidden_width,
            "rho": float(rho),
            "anisotropy": anisotropy,
            "readout_init_scale": float(readout_init_scale),
        }
        self.radial_head = torch.nn.Sequential(
            torch.nn.Linear(mace_feature_dim, hidden_width),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_width, 2),
        )
        self.coefficient_heads = torch.nn.ModuleList(
            [torch.nn.Linear(n, 1, bias=False) for n in multiplicities]
            if anisotropy
            else []
        )
        with torch.no_grad():
            self.radial_head[-1].weight.mul_(readout_init_scale)
            self.radial_head[-1].bias.zero_()
            for head in self.coefficient_heads:
                head.weight.mul_(readout_init_scale)
        # Store explicit baselines in float64 so model.double() does not
        # inherit rounding of the logarithms through default float32.
        base = torch.zeros(119, 2, dtype=torch.float64)
        domain = torch.zeros(119, dtype=torch.bool)
        for z, values in baselines.items():
            base[int(z)] = torch.tensor(values, dtype=torch.float64).log()
            domain[int(z)] = True
        self.register_buffer("base_log_ab", base)
        self.register_buffer("element_domain", domain)

    def get_config(self) -> dict[str, Any]:
        """Return independent constructor arguments for reconstruction."""
        return deepcopy(self.config)

    def get_extra_state(self) -> dict[str, Any]:
        return {"architecture": "mace-mastiff-eql3-v1", **self.get_config()}

    def set_extra_state(self, state: dict[str, Any]) -> None:
        if state != self.get_extra_state():
            raise RuntimeError("MACE MASTIFF checkpoint contract mismatch")

    def _validate_features(self, features: MACEAtomicFeatures) -> None:
        if features.feature_schema != self.config["feature_schema"]:
            raise ValueError("MACE feature schema mismatch")
        if features.invariant.shape != (
            features.natom,
            self.config["mace_feature_dim"],
        ):
            raise ValueError("MACE invariant feature width mismatch")
        weight = self.radial_head[0].weight
        for value in (features.invariant, features.equivariant):
            if value.dtype != weight.dtype or value.device != weight.device:
                raise ValueError("features and model must share dtype and device")
            if not torch.isfinite(value).all():
                raise ValueError("MACE features must be finite")
        numbers = features.atomic_numbers
        if numbers.device != weight.device or features.batch.device != weight.device:
            raise ValueError("feature indices and model must share device")
        if torch.any((numbers < 1) | (numbers > 118)):
            raise ValueError("unsupported atomic number")
        if not self.element_domain[numbers.long()].all():
            raise ValueError("missing element baseline")
        for degree in (1, 2, 3):
            features.equivariant_degree(degree)

    def atom_quantities(
        self, features: MACEAtomicFeatures
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
        """Return positive A/B and bounded l=1,2,3 coefficient vectors."""
        self._validate_features(features)
        raw = self.radial_head(features.invariant.detach())
        ab = (
            raw + self.base_log_ab[features.atomic_numbers.long()].to(raw.dtype)
        ).exp()
        if not torch.isfinite(ab).all() or torch.any(ab <= 0):
            raise ValueError("radial A/B must remain finite and positive")
        coefficients = []
        for degree in (1, 2, 3):
            if self.config["anisotropy"]:
                block = features.equivariant_degree(degree).detach()
                projected = self.coefficient_heads[degree - 1](
                    block.transpose(1, 2)
                ).squeeze(-1)
            else:
                projected = raw.new_zeros((features.natom, 2 * degree + 1))
            if not torch.isfinite(projected).all():
                raise ValueError("angular coefficients must remain finite")
            coefficients.append(_bounded_degree(projected, self.config["rho"]))
        return ab[:, 0], ab[:, 1], tuple(coefficients)

    def pair_energies(
        self,
        features_a: MACEAtomicFeatures,
        features_b: MACEAtomicFeatures,
        positions_a: torch.Tensor,
        positions_b: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Return kcal/mol exchange per supplied A-to-B edge, without cutoff."""
        qa = self.atom_quantities(features_a)
        qb = self.atom_quantities(features_b)
        for features, positions in (
            (features_a, positions_a),
            (features_b, positions_b),
        ):
            if positions.shape != (features.natom, 3):
                raise ValueError("positions must have shape [n_atom, 3]")
            if (
                positions.dtype != qa[0].dtype
                or positions.device != qa[0].device
                or not torch.isfinite(positions).all()
            ):
                raise ValueError(
                    "positions must be finite and match model dtype/device"
                )
        if features_a.total_charge.numel() != features_b.total_charge.numel():
            raise ValueError("monomer batch counts must match")
        if (
            edge_index.ndim != 2
            or edge_index.shape[0] != 2
            or edge_index.dtype != torch.int64
            or edge_index.device != positions_a.device
        ):
            raise ValueError("edge_index must be an int64 [2, n_pair] tensor")
        source, target = edge_index
        for index, features in ((source, features_a), (target, features_b)):
            if torch.any((index < 0) | (index >= features.natom)):
                raise ValueError("edge atom index outside feature batch")
        if not torch.equal(features_a.batch[source], features_b.batch[target]):
            raise ValueError("edges cannot cross dimer batch identities")
        vectors = positions_b[target] - positions_a[source]
        distance = vectors.norm(dim=-1)
        if torch.any(distance <= 0):
            raise ValueError("intermonomer distances must be positive")
        vectors = vectors[:, list(MACE_E3NN_AXIS_PERMUTATION)]
        si, sj = torch.zeros_like(distance), torch.zeros_like(distance)
        for degree, (ca, cb) in enumerate(zip(qa[2], qb[2]), 1):
            # e3nn component normalization has norm sqrt(2l+1).
            harmonic = real_spherical_harmonics(degree, vectors) / math.sqrt(
                2 * degree + 1
            )
            si = si + (ca[source] * harmonic).sum(-1)
            sj = sj + (cb[target] * harmonic).sum(-1) * (-1) ** degree
        energy = slater_exponent_exchange(
            qa[0][source],
            qb[0][target],
            qa[1][source],
            qb[1][target],
            distance,
            si,
            sj,
        )
        if not torch.isfinite(energy).all():
            raise ValueError("exchange energy exceeded numerical range")
        return energy

    def forward(
        self,
        features_a: MACEAtomicFeatures,
        features_b: MACEAtomicFeatures,
        positions_a: torch.Tensor,
        positions_b: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Sum selected-pair exchange to one kcal/mol value per dimer."""
        energies = self.pair_energies(
            features_a, features_b, positions_a, positions_b, edge_index
        )
        result = energies.new_zeros(features_a.total_charge.numel()).index_add(
            0, features_a.batch[edge_index[0]].long(), energies
        )
        if not torch.isfinite(result).all():
            raise ValueError("dimer exchange sum exceeded numerical range")
        return result
