"""Opt-in, parent-preserving exchange corrections from frozen MACE features."""

from __future__ import annotations

import math
import re
from typing import Any

import torch

from .pair import MACE_E3NN_AXIS_PERMUTATION, real_spherical_harmonics
from .schema import MACEAtomicFeatures, _parse_irreps


class MACEExchangeCorrection(torch.nn.Module):
    """Add only an exchange delta to externally verified parent energies.

    The parent is supplied as a detached ``[n_dimer, 4]`` tensor in kcal/mol,
    ordered elst/exch/indu/disp. This module never loads a parent or evaluates
    atomic-property models. Callers must verify parent provenance and row
    alignment; ``parent_contract_sha256`` binds that contract in checkpoints,
    but cannot authenticate the supplied tensor by itself.

    ``scalar`` uses orientation-blind channel norms and norm products.
    ``axial-tensor`` substitutes axial contractions and tensor inner products.
    Both have identical trainable shapes, bidirectional sharing and cutoffs.
    ``unit-ball`` scales each irrep channel by ``sqrt(1 + ||F||**2)`` before
    either construction; ``raw`` retains its original amplitude.

    All dimensions are materialized at construction. The final readout is
    zero-initialized, preserving the parent at initialization. Parent and MACE
    tensors are detached intentionally: this is an energy-only correction,
    not an end-to-end molecular force model.
    """

    def __init__(
        self,
        *,
        mace_feature_dim: int,
        mace_equivariant_dim: int,
        feature_schema: str,
        parent_contract_sha256: str,
        mode: str = "axial-tensor",
        tensor_scaling: str = "unit-ball",
        degree: int = 3,
        projection_width: int = 8,
        hidden_width: int = 32,
        n_rbf: int = 8,
        cutoff: float = 8.0,
    ) -> None:
        super().__init__()
        dimensions = (
            mace_feature_dim,
            mace_equivariant_dim,
            projection_width,
            hidden_width,
            n_rbf,
        )
        if any(
            isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in dimensions
        ):
            raise ValueError("feature and layer dimensions must be positive integers")
        if (
            not isinstance(degree, int)
            or isinstance(degree, bool)
            or degree not in (1, 2, 3)
        ):
            raise ValueError("degree must be 1, 2 or 3")
        if mode not in ("scalar", "axial-tensor"):
            raise ValueError("mode must be scalar or axial-tensor")
        if tensor_scaling not in ("raw", "unit-ball"):
            raise ValueError("tensor_scaling must be raw or unit-ball")
        if not re.fullmatch(r"[0-9a-f]{64}", parent_contract_sha256):
            raise ValueError("parent_contract_sha256 must be a lowercase SHA-256")
        if not feature_schema:
            raise ValueError("an exact feature_schema is required")
        if isinstance(cutoff, bool) or not math.isfinite(cutoff) or cutoff <= 0:
            raise ValueError("cutoff must be finite and positive")
        self.config = {
            "mace_feature_dim": mace_feature_dim,
            "mace_equivariant_dim": mace_equivariant_dim,
            "feature_schema": feature_schema,
            "parent_contract_sha256": parent_contract_sha256,
            "mode": mode,
            "tensor_scaling": tensor_scaling,
            "degree": degree,
            "projection_width": projection_width,
            "hidden_width": hidden_width,
            "n_rbf": n_rbf,
            "cutoff": float(cutoff),
        }
        self.invariant_projection = torch.nn.Linear(mace_feature_dim, projection_width)
        # Mixing scalar contractions, not m components, preserves O(3).
        self.channel_projection = torch.nn.Linear(
            mace_equivariant_dim, projection_width
        )
        self.readout = torch.nn.Sequential(
            torch.nn.Linear(5 * projection_width + n_rbf, hidden_width),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_width, 1),
        )
        torch.nn.init.zeros_(self.readout[-1].weight)
        torch.nn.init.zeros_(self.readout[-1].bias)
        self.register_buffer("radial_centres", torch.linspace(0.0, cutoff, n_rbf))

    def get_config(self) -> dict[str, Any]:
        """Return constructor arguments for exact reconstruction."""
        return dict(self.config)

    def get_extra_state(self) -> dict[str, Any]:
        return {"architecture": "mace-exchange-correction-v1", **self.get_config()}

    def set_extra_state(self, state: dict[str, Any]) -> None:
        if state != self.get_extra_state():
            raise RuntimeError(
                "exchange correction architecture/parent contract mismatch"
            )

    def _validate(
        self,
        batch: Any,
        fa: MACEAtomicFeatures,
        fb: MACEAtomicFeatures,
        parent: torch.Tensor,
    ) -> None:
        if parent.shape != (fa.total_charge.numel(), 4):
            raise ValueError("parent components must have shape [n_dimer, 4]")
        if not torch.isfinite(parent).all():
            raise ValueError("parent components must be finite")
        for features, numbers, positions in (
            (fa, batch.ZA, batch.RA),
            (fb, batch.ZB, batch.RB),
        ):
            if features.feature_schema != self.config["feature_schema"]:
                raise ValueError("MACE feature schema does not match correction")
            if features.invariant.shape != (
                numbers.numel(),
                self.config["mace_feature_dim"],
            ) or not torch.equal(features.atomic_numbers, numbers):
                raise ValueError("MACE feature atom order or dimensions mismatch")
            if positions.shape != (numbers.numel(), 3):
                raise ValueError("coordinates must have shape [n_atom, 3]")
            if not torch.isfinite(positions).all():
                raise ValueError("coordinates must be finite")
            if features.total_charge.numel() != parent.shape[0]:
                raise ValueError("parent/feature dimer counts mismatch")
            for tensor in (features.invariant, positions, parent):
                if (
                    tensor.dtype != self.radial_centres.dtype
                    or tensor.device != self.radial_centres.device
                ):
                    raise ValueError(
                        "model, features, coordinates and parent must align"
                    )
            block = features.equivariant_degree(self.config["degree"])
            if block.shape[1] != self.config["mace_equivariant_dim"]:
                raise ValueError("equivariant channel count mismatch")
            natural = "o" if self.config["degree"] % 2 else "e"
            if any(
                parity != natural
                for _, degree, parity in _parse_irreps(features.equivariant_irreps)
                if degree == self.config["degree"]
            ):
                raise ValueError("axial correction requires natural parity irreps")
        source, target, dimer = (
            batch.e_ABsr_source,
            batch.e_ABsr_target,
            batch.dimer_ind,
        )
        if source.shape != target.shape or source.shape != dimer.shape:
            raise ValueError("edge source, target and dimer indices must align")
        for index, count in (
            (source, fa.natom),
            (target, fb.natom),
            (dimer, parent.shape[0]),
        ):
            if (
                index.ndim != 1
                or index.dtype != torch.int64
                or index.device != parent.device
                or torch.any((index < 0) | (index >= count))
            ):
                raise ValueError("edge and dimer indices must be valid int64 vectors")
        if not torch.equal(fa.batch[source], dimer) or not torch.equal(
            fb.batch[target], dimer
        ):
            raise ValueError("edge atoms do not belong to the assigned parent dimer")

    def _channel_features(
        self,
        fa: MACEAtomicFeatures,
        fb: MACEAtomicFeatures,
        source: torch.Tensor,
        target: torch.Tensor,
        vectors: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        a = fa.equivariant_degree(self.config["degree"]).detach()[source]
        b = fb.equivariant_degree(self.config["degree"]).detach()[target]
        if self.config["tensor_scaling"] == "unit-ball":

            def bounded(block: torch.Tensor) -> torch.Tensor:
                # Algebraically F/sqrt(1+||F||^2), without squaring large
                # finite inputs into inf and silently erasing their direction.
                scale = block.abs().amax(-1, keepdim=True).clamp_min(1)
                scaled = block / scale
                denominator = torch.sqrt(
                    scale.reciprocal().square() + scaled.square().sum(-1, keepdim=True)
                )
                return scaled / denominator

            a, b = bounded(a), bounded(b)
        if self.config["mode"] == "scalar":
            left, right = a.norm(dim=-1), b.norm(dim=-1)
            correlation = left * right
        else:
            vectors = vectors[:, list(MACE_E3NN_AXIS_PERMUTATION)]
            ya = real_spherical_harmonics(self.config["degree"], vectors)
            yb = real_spherical_harmonics(self.config["degree"], -vectors)
            left = (a * ya[:, None, :]).sum(-1)
            right = (b * yb[:, None, :]).sum(-1)
            correlation = (a * b).sum(-1)
        return tuple(self.channel_projection(x) for x in (left, right, correlation))

    def forward(
        self,
        batch: Any,
        features_a: MACEAtomicFeatures,
        features_b: MACEAtomicFeatures,
        parent_components: torch.Tensor,
    ) -> torch.Tensor:
        """Return parent energies plus a cutoff-localized exchange correction."""
        self._validate(batch, features_a, features_b, parent_components)
        source, target = batch.e_ABsr_source, batch.e_ABsr_target
        vectors = batch.RB[target] - batch.RA[source]
        distance = vectors.norm(dim=-1)
        if torch.any(distance <= 0):
            raise ValueError("intermonomer edges must have positive distance")
        cutoff = self.config["cutoff"]
        radial = torch.exp(
            -(
                (distance[:, None] - self.radial_centres)
                / (cutoff / self.config["n_rbf"])
            ).square()
        )
        a = torch.nn.functional.silu(
            self.invariant_projection(features_a.invariant.detach())
        )[source]
        b = torch.nn.functional.silu(
            self.invariant_projection(features_b.invariant.detach())
        )[target]
        left, right, correlation = self._channel_features(
            features_a, features_b, source, target, vectors
        )
        ab = torch.cat((a, b, left, right, correlation, radial), -1)
        ba = torch.cat((b, a, right, left, correlation, radial), -1)
        edge_delta = 0.5 * (self.readout(ab) + self.readout(ba)).squeeze(-1)
        envelope = 0.5 * (1.0 + torch.cos(math.pi * (distance / cutoff).clamp(max=1)))
        edge_delta = edge_delta * envelope
        delta = parent_components.new_zeros(parent_components.shape[0]).index_add(
            0, batch.dimer_ind, edge_delta
        )
        result = parent_components.detach().clone()
        result[:, 1] = result[:, 1] + delta
        if not torch.isfinite(result).all():
            raise RuntimeError("exchange correction produced nonfinite energies")
        return result
