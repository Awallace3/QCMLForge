"""Constrained atomic-property heads for frozen PolarMACE representations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import torch
import torch.nn.functional as functional

from apnet_pt import constants

from .schema import (
    AtomicPropertyBundle,
    MACEAtomicFeatures,
    POLAR_DENSITY_L1_CONTRACT,
    PolarMACEDirectOutputs,
)


def _e3nn_o3():
    """Import pinned e3nn constants under PyTorch's restricted safe loader."""

    with torch.serialization.safe_globals([slice]):
        from e3nn import o3

    return o3


@runtime_checkable
class AtomicPropertyProvider(Protocol):
    """Structural protocol shared by MACE atomic-property routes."""

    def forward(
        self,
        batch: Any,
        features_a: MACEAtomicFeatures,
        features_b: MACEAtomicFeatures,
        **kwargs: Any,
    ) -> tuple[AtomicPropertyBundle, AtomicPropertyBundle]: ...


@dataclass(frozen=True)
class _HeadOutputs:
    raw_q: torch.Tensor
    mu: torch.Tensor
    quadrupole: torch.Tensor
    hfvr: torch.Tensor
    valence_width: torch.Tensor
    damping: torch.Tensor


def _conserve_charge(
    raw_q: torch.Tensor,
    batch: torch.Tensor,
    total_charge: torch.Tensor,
) -> torch.Tensor:
    """Project atomic charges onto each monomer's exact affine constraint."""

    nmonomer = total_charge.numel()
    sums = raw_q.new_zeros(nmonomer)
    sums.index_add_(0, batch, raw_q[:, 0])
    counts = torch.bincount(batch, minlength=nmonomer).to(raw_q)
    if (counts == 0).any():
        raise ValueError("every monomer must contain at least one atom")
    correction = (total_charge - sums) / counts
    q = raw_q + correction[batch, None]
    # Remove the final floating-point residual without detaching gradients.
    projected_sums = q.new_zeros(nmonomer)
    projected_sums.index_add_(0, batch, q[:, 0])
    residual = total_charge - projected_sums
    final_indices = torch.stack(
        [torch.where(batch == monomer)[0][-1] for monomer in range(nmonomer)]
    )
    q = q.clone()
    q[final_indices, 0] = q[final_indices, 0] + residual
    return q


def _alpha_from_hfvr(
    atomic_numbers: torch.Tensor,
    hfvr: torch.Tensor,
) -> torch.Tensor:
    table = constants.polarizability_table.to(device=hfvr.device, dtype=hfvr.dtype)
    if atomic_numbers.max() >= table.numel():
        raise ValueError("atomic number is outside the free-atom polarizability table")
    return table.index_select(0, atomic_numbers.long()).reshape(-1, 1) * hfvr.pow(
        4.0 / 3.0
    )


class LegacyAtomMPNNPropertyProvider(torch.nn.Module):
    """Adapt existing AtomMPNN/AtomTypeParam outputs to the shared contract.

    The wrapped legacy hierarchy remains the sole owner of all property
    equations. This class only normalizes shapes, preserves the signed
    response values consumed by the legacy pair path, and derives physical
    alpha from ``abs(HFVR)``.
    """

    def __init__(
        self,
        legacy_model: torch.nn.Module,
        *,
        freeze: bool | None = None,
        freeze_atom_model: bool | None = None,
        freeze_dimer_parameters: bool | None = None,
    ) -> None:
        super().__init__()
        if freeze is not None:
            if freeze_atom_model is not None or freeze_dimer_parameters is not None:
                raise ValueError("freeze cannot be combined with selective freeze flags")
            freeze_atom_model = freeze
            freeze_dimer_parameters = freeze
        self.freeze_atom_model = (
            True if freeze_atom_model is None else freeze_atom_model
        )
        self.freeze_dimer_parameters = (
            True if freeze_dimer_parameters is None else freeze_dimer_parameters
        )
        self.freeze = self.freeze_atom_model and self.freeze_dimer_parameters
        self.legacy_model = legacy_model
        atom_model = getattr(self.legacy_model, "atom_model", None)
        if not isinstance(atom_model, torch.nn.Module):
            if not self.freeze_atom_model:
                raise ValueError(
                    "selective atom unfreezing requires legacy_model.atom_model"
                )
        for name, parameter in self.legacy_model.named_parameters():
            is_atom = name == "atom_model" or name.startswith("atom_model.")
            parameter.requires_grad_(
                not (self.freeze_atom_model if is_atom else self.freeze_dimer_parameters)
            )
        if self.freeze:
            self.legacy_model.eval()
        elif self.freeze_atom_model and isinstance(atom_model, torch.nn.Module):
            atom_model.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        atom_model = getattr(self.legacy_model, "atom_model", None)
        if self.freeze:
            self.legacy_model.eval()
        elif self.freeze_atom_model and isinstance(atom_model, torch.nn.Module):
            atom_model.eval()
        return self

    @staticmethod
    def _adapt_monomer(
        raw: tuple[torch.Tensor, ...],
        features: MACEAtomicFeatures,
    ) -> AtomicPropertyBundle:
        if len(raw) < 5:
            raise ValueError("legacy AtomMPNN output is missing AP3 properties")
        response = raw[-2]
        if response.ndim != 2 or response.shape[1] < 2:
            raise ValueError("legacy response output must contain HFVR and width")
        q = raw[0].reshape(-1, 1)
        hfvr = response[:, 0:1]
        valence_width = response[:, 1:2]
        damping = raw[-1].reshape(-1, 1).abs()
        alpha = _alpha_from_hfvr(features.atomic_numbers, hfvr.abs())
        return AtomicPropertyBundle(
            q=q,
            mu=raw[1],
            quadrupole=raw[2],
            hfvr=hfvr,
            valence_width=valence_width,
            alpha=alpha,
            damping=damping,
        )

    def forward(
        self,
        batch: Any,
        features_a: MACEAtomicFeatures,
        features_b: MACEAtomicFeatures,
        **kwargs: Any,
    ) -> tuple[AtomicPropertyBundle, AtomicPropertyBundle]:
        del kwargs
        outputs = self.legacy_model(batch)
        if not isinstance(outputs, (tuple, list)):
            raise TypeError("legacy property model must return monomer tuples")
        if len(outputs) == 2:
            raw_a, raw_b = outputs
        elif len(outputs) == 3:
            _, raw_a, raw_b = outputs
        else:
            raise ValueError("legacy property model returned an unsupported layout")
        return (
            self._adapt_monomer(raw_a, features_a),
            self._adapt_monomer(raw_b, features_b),
        )


class MACEPropertyCompletionHeads(torch.nn.Module):
    """Reusable invariant and equivariant property completion modules.

    Covariant channels are consumed by e3nn ``Linear`` and
    ``FullyConnectedTensorProduct`` blocks.  They are never flattened into an
    unconstrained scalar MLP.
    """

    def __init__(
        self,
        *,
        invariant_dim: int,
        equivariant_irreps: str,
        hidden_dim: int = 64,
        geometry_channels: int = 8,
        positive_epsilon: float = 1.0e-6,
    ) -> None:
        super().__init__()
        if invariant_dim < 1 or hidden_dim < 1 or geometry_channels < 1:
            raise ValueError("property-head dimensions must be positive")
        o3 = _e3nn_o3()

        self.invariant_dim = invariant_dim
        self.equivariant_irreps = str(o3.Irreps(equivariant_irreps))
        self.positive_epsilon = positive_epsilon
        self.invariant_network = torch.nn.Sequential(
            torch.nn.Linear(invariant_dim, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, 4),
        )
        hidden_irreps = o3.Irreps(
            f"{geometry_channels}x1o+{geometry_channels}x2e"
        )
        output_irreps = o3.Irreps("1x1o+1x2e")
        self.geometry_input = o3.Linear(
            o3.Irreps(self.equivariant_irreps), hidden_irreps
        )
        self.geometry_context = torch.nn.Sequential(
            torch.nn.Linear(invariant_dim, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, 1),
            torch.nn.Sigmoid(),
        )
        self.geometry_product = o3.FullyConnectedTensorProduct(
            hidden_irreps,
            o3.Irreps("1x0e"),
            output_irreps,
        )
        # Fixed e3nn-0.4.4 ``1x2e`` to symmetric-traceless Cartesian basis.
        # ``CartesianTensor`` computes this basis through a randomized linear
        # solve and can fail for otherwise valid process RNG states.  Keeping
        # the reviewed basis explicit makes model construction deterministic
        # without consuming or depending on the caller's RNG stream.
        self.register_buffer(
            "_l2_cartesian_basis",
            torch.tensor(
                [
                    [[0.0, 0.0, 2**-0.5], [0.0, 0.0, 0.0], [2**-0.5, 0.0, 0.0]],
                    [[0.0, 2**-0.5, 0.0], [2**-0.5, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    [[-6**-0.5, 0.0, 0.0], [0.0, (2 / 3) ** 0.5, 0.0], [0.0, 0.0, -6**-0.5]],
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 2**-0.5], [0.0, 2**-0.5, 0.0]],
                    [[-2**-0.5, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 2**-0.5]],
                ],
                dtype=torch.float64,
            ),
            persistent=False,
        )
        self.register_buffer(
            "_physical_from_mace",
            torch.tensor([2, 0, 1], dtype=torch.long),
            persistent=False,
        )

    def _geometry(
        self, invariant: torch.Tensor, equivariant: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if equivariant.shape[1] != self.geometry_input.irreps_in.dim:
            raise ValueError(
                "equivariant feature width does not match the declared e3nn irreps"
            )
        hidden = self.geometry_input(equivariant)
        context = self.geometry_context(invariant)
        spherical = self.geometry_product(hidden, context)
        vector_mace = spherical[:, :3]
        l2 = spherical[:, 3:]
        mu = vector_mace.index_select(1, self._physical_from_mace)
        basis = self._l2_cartesian_basis.to(l2)
        quadrupole_mace = torch.einsum("nk,kij->nij", l2, basis)
        axes = self._physical_from_mace
        quadrupole = quadrupole_mace.index_select(1, axes).index_select(2, axes)
        # Guard against accumulated conversion roundoff at the public seam.
        quadrupole = 0.5 * (quadrupole + quadrupole.transpose(-1, -2))
        trace = quadrupole.diagonal(dim1=-2, dim2=-1).sum(-1) / 3.0
        quadrupole = quadrupole - torch.diag_embed(trace[:, None].expand(-1, 3))
        return mu, quadrupole

    def forward(self, features: MACEAtomicFeatures) -> _HeadOutputs:
        if features.invariant.shape[1] != self.invariant_dim:
            raise ValueError("invariant feature width does not match property heads")
        invariant_values = self.invariant_network(features.invariant)
        mu, quadrupole = self._geometry(
            features.invariant, features.equivariant
        )
        return _HeadOutputs(
            raw_q=invariant_values[:, 0:1],
            mu=mu,
            quadrupole=quadrupole,
            hfvr=functional.softplus(invariant_values[:, 1:2])
            + self.positive_epsilon,
            valence_width=functional.softplus(invariant_values[:, 2:3])
            + self.positive_epsilon,
            damping=functional.softplus(invariant_values[:, 3:4])
            + self.positive_epsilon,
        )


class MACEAtomPropertyModel(torch.nn.Module):
    """AtomHead C: all AP3 atomic properties from frozen MACE features."""

    def __init__(self, completion_heads: MACEPropertyCompletionHeads) -> None:
        super().__init__()
        self.completion_heads = completion_heads

    @property
    def property_provider(self) -> "MACEAtomPropertyModel":
        """Expose the provider protocol name without registering a recursive module."""

        return self

    def forward_monomer(self, features: MACEAtomicFeatures) -> AtomicPropertyBundle:
        values = self.completion_heads(features)
        q = _conserve_charge(values.raw_q, features.batch, features.total_charge)
        alpha = _alpha_from_hfvr(features.atomic_numbers, values.hfvr)
        return AtomicPropertyBundle(
            q=q,
            mu=values.mu,
            quadrupole=values.quadrupole,
            hfvr=values.hfvr,
            valence_width=values.valence_width,
            alpha=alpha,
            damping=values.damping,
        )

    def forward(
        self,
        batch: Any,
        features_a: MACEAtomicFeatures,
        features_b: MACEAtomicFeatures,
        **kwargs: Any,
    ) -> tuple[AtomicPropertyBundle, AtomicPropertyBundle]:
        del batch, kwargs
        return self.forward_monomer(features_a), self.forward_monomer(features_b)


class PolarDirectPropertyProvider(torch.nn.Module):
    """DirectPolar A: direct q/μ plus C's Q/response completion modules."""

    def __init__(
        self,
        completion_heads: MACEPropertyCompletionHeads,
        *,
        multipole_contract: str = POLAR_DENSITY_L1_CONTRACT,
        reconstruction_tolerance: float = 1.0e-5,
    ) -> None:
        super().__init__()
        if multipole_contract != POLAR_DENSITY_L1_CONTRACT:
            raise ValueError("incompatible PolarMACE multipole contract")
        self.completion_heads = completion_heads
        self.multipole_contract = multipole_contract
        self.reconstruction_tolerance = reconstruction_tolerance

    def forward_monomer(
        self,
        features: MACEAtomicFeatures,
        direct: PolarMACEDirectOutputs,
    ) -> AtomicPropertyBundle:
        if direct.multipole_contract != self.multipole_contract:
            raise ValueError("incompatible PolarMACE multipole contract")
        if direct.batch.shape != features.batch.shape or not torch.equal(
            direct.batch, features.batch
        ):
            raise ValueError("direct outputs and MACE features use different batches")
        direct_sums = direct.charges.new_zeros(direct.total_charge.numel())
        direct_sums.index_add_(0, direct.batch, direct.charges)
        if not torch.allclose(
            direct_sums,
            direct.total_charge,
            atol=self.reconstruction_tolerance,
            rtol=1.0e-6,
        ):
            raise ValueError("direct charges do not sum to direct.total_charge")
        if not torch.allclose(
            direct.total_charge,
            features.total_charge,
            atol=self.reconstruction_tolerance,
            rtol=1.0e-6,
        ):
            raise ValueError(
                "direct and feature total charges disagree at the monomer seam"
            )
        values = self.completion_heads(features)
        # The PolarMACE public charges are a direct scientific output. They are
        # validated, never projected or otherwise altered by QCMLForge.
        q = direct.charges.reshape(-1, 1)
        mu = direct.intrinsic_dipole_eangstrom / constants.au2ang
        reconstructed = (
            q * direct.positions_angstrom + mu * constants.au2ang
        ).new_zeros(direct.molecular_dipole_eangstrom.shape)
        reconstructed.index_add_(
            0,
            features.batch,
            q * direct.positions_angstrom + mu * constants.au2ang,
        )
        if not torch.allclose(
            reconstructed,
            direct.molecular_dipole_eangstrom,
            atol=self.reconstruction_tolerance,
            rtol=1.0e-6,
        ):
            raise ValueError("direct q/mu fail PolarMACE molecular dipole reconstruction")
        alpha = _alpha_from_hfvr(features.atomic_numbers, values.hfvr)
        return AtomicPropertyBundle(
            q=q,
            mu=mu,
            quadrupole=values.quadrupole,
            hfvr=values.hfvr,
            valence_width=values.valence_width,
            alpha=alpha,
            damping=values.damping,
        )

    def forward(
        self,
        batch: Any,
        features_a: MACEAtomicFeatures,
        features_b: MACEAtomicFeatures,
        *,
        direct_a: PolarMACEDirectOutputs,
        direct_b: PolarMACEDirectOutputs,
        **kwargs: Any,
    ) -> tuple[AtomicPropertyBundle, AtomicPropertyBundle]:
        del batch, kwargs
        return (
            self.forward_monomer(features_a, direct_a),
            self.forward_monomer(features_b, direct_b),
        )
