"""Shared classical SAPT spine for every MACE/AP3D3 architecture."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import warnings

import torch
import torch.nn as nn

from apnet_pt import constants

from .schema import (
    AtomicPropertyBundle,
    ClassicalEnergyBundle,
    InductionDiagnostics,
    PhysicsConfig,
)


def _default_backends():
    from ..AtomPairwiseModels.mtp_mtp import (
        induced_dipole_induction_optimized_no_correction,
        mtp_elst,
        mtp_elst_damping,
        mtp_elst_damping_AMOEBA,
    )
    from qcml_dftd3.d3 import d3

    return (
        {
            "damped-cliff": mtp_elst_damping,
            "damped-amoeba": mtp_elst_damping_AMOEBA,
            "undamped": mtp_elst,
        },
        induced_dipole_induction_optimized_no_correction,
        d3,
    )


def _aggregate_pairs(
    values: torch.Tensor, dimer_index: torch.Tensor, ndimer: int
) -> torch.Tensor:
    values = values.reshape(-1)
    if values.numel() != dimer_index.numel():
        raise ValueError(
            "Classical pair ledger and full-pair dimer index have different lengths"
        )
    result = values.new_zeros(ndimer)
    if values.numel():
        result.index_add_(0, dimer_index.long(), values)
    return result


def _normalize_diagnostics(value: object) -> InductionDiagnostics:
    if isinstance(value, InductionDiagnostics):
        return value
    if isinstance(value, Mapping):
        return InductionDiagnostics(
            converged=bool(value["converged"]),
            iterations=int(value["iterations"]),
            residual=float(value["residual"]),
        )
    raise TypeError("Induction backend must return structured convergence diagnostics")


def _validate_full_pair_edges(batch) -> None:
    source = batch.e_ABfull_source
    target = batch.e_ABfull_target
    dimer_index = batch.dimer_ind_full
    if source.ndim != 1 or target.ndim != 1 or dimer_index.ndim != 1:
        raise ValueError("full intermonomer edge tensors must be rank 1")
    if source.numel() != target.numel() or source.numel() != dimer_index.numel():
        raise ValueError("full intermonomer edge tensors must have matching lengths")
    if not hasattr(batch, "natom_per_mol_A") or not hasattr(batch, "natom_per_mol_B"):
        return
    natom_a = batch.natom_per_mol_A.tolist()
    natom_b = batch.natom_per_mol_B.tolist()
    if len(natom_a) != len(natom_b):
        raise ValueError("monomer atom-count ledgers must align")
    expected_total = sum(int(a) * int(b) for a, b in zip(natom_a, natom_b))
    if source.numel() != expected_total:
        raise ValueError("full edges must contain every intermonomer Cartesian pair")
    offset_a = 0
    offset_b = 0
    for dimer, (count_a, count_b) in enumerate(zip(natom_a, natom_b)):
        mask = dimer_index == dimer
        local_source = source[mask] - offset_a
        local_target = target[mask] - offset_b
        expected = int(count_a) * int(count_b)
        if local_source.numel() != expected:
            raise ValueError("full edges must contain every intermonomer Cartesian pair")
        pairs = set(zip(local_source.tolist(), local_target.tolist()))
        canonical = {
            (a, b) for a in range(int(count_a)) for b in range(int(count_b))
        }
        if pairs != canonical:
            raise ValueError("full edges must contain each Cartesian pair exactly once")
        offset_a += int(count_a)
        offset_b += int(count_b)


class LongRangeSAPTProvider(nn.Module):
    """Evaluate MTP-MTP, Thole-SCF induction, and D3 exactly once.

    Architecture-specific code supplies only canonical atomic properties.  This
    provider owns explicit mode dispatch and both pair and dimer ledgers.
    """

    def __init__(
        self,
        config: PhysicsConfig,
        *,
        electrostatics_kernels: Mapping[str, Callable] | None = None,
        induction_kernel: Callable | None = None,
        dispersion_kernel: Callable | None = None,
    ) -> None:
        super().__init__()
        defaults = None
        if (
            electrostatics_kernels is None
            or induction_kernel is None
            or dispersion_kernel is None
        ):
            defaults = _default_backends()
        self.config = config
        self.electrostatics_kernels = dict(
            electrostatics_kernels
            if electrostatics_kernels is not None
            else defaults[0]
        )
        self.induction_kernel = (
            induction_kernel if induction_kernel is not None else defaults[1]
        )
        self.dispersion_kernel = (
            dispersion_kernel if dispersion_kernel is not None else defaults[2]
        )
        if config.electrostatics_mode not in self.electrostatics_kernels:
            raise ValueError(
                f"No kernel registered for explicit mode {config.electrostatics_mode}"
            )

    def _electrostatics(self, batch, a: AtomicPropertyBundle, b: AtomicPropertyBundle):
        common = dict(
            ZA=batch.ZA,
            RA=batch.RA,
            muA=a.mu,
            quadA=a.quadrupole,
            ZB=batch.ZB,
            RB=batch.RB,
            muB=b.mu,
            quadB=b.quadrupole,
            e_AB_source=batch.e_ABfull_source,
            e_AB_target=batch.e_ABfull_target,
        )
        mode = self.config.electrostatics_mode
        kernel = self.electrostatics_kernels[mode]
        if mode == "undamped":
            return kernel(qA=a.q, qB=b.q, **common)
        if mode == "damped-amoeba":
            if not hasattr(batch, "amoeba_K_A") or not hasattr(batch, "amoeba_K_B"):
                raise ValueError(
                    "AMOEBA electrostatics requires explicit mode-specific damping "
                    "inputs amoeba_K_A and amoeba_K_B"
                )
            damping_a = batch.amoeba_K_A.reshape(-1).to(a.q)
            damping_b = batch.amoeba_K_B.reshape(-1).to(b.q)
            if damping_a.numel() != a.natom or damping_b.numel() != b.natom:
                raise ValueError("AMOEBA damping inputs must contain one K per atom")
            if (
                not torch.isfinite(damping_a).all()
                or not torch.isfinite(damping_b).all()
                or (damping_a <= 0).any()
                or (damping_b <= 0).any()
            ):
                raise ValueError("AMOEBA damping inputs must be finite and positive")
        else:
            damping_a = torch.abs(a.damping).reshape(-1)
            damping_b = torch.abs(b.damping).reshape(-1)
        return kernel(
            qA_0=a.q.reshape(-1),
            Ka=damping_a,
            qB_0=b.q.reshape(-1),
            Kb=damping_b,
            **common,
        )

    @staticmethod
    def _validate_alpha(
        atomic_numbers: torch.Tensor,
        properties: AtomicPropertyBundle,
        monomer: str,
    ) -> None:
        table = constants.polarizability_table.to(properties.alpha)
        numbers = atomic_numbers.long()
        if numbers.numel() != properties.natom or numbers.max() >= table.numel():
            raise ValueError(f"monomer {monomer} alpha atomic numbers are invalid")
        expected = table.index_select(0, numbers).reshape(-1, 1) * (
            properties.hfvr.abs().pow(4.0 / 3.0)
        )
        if not torch.allclose(properties.alpha, expected, atol=1.0e-5, rtol=1.0e-5):
            raise ValueError(
                f"monomer {monomer} alpha does not match canonical "
                "free-atom*abs(HFVR)^(4/3)"
            )

    def _induction(self, batch, a: AtomicPropertyBundle, b: AtomicPropertyBundle):
        result = self.induction_kernel(
            ZA=batch.ZA,
            RA=batch.RA,
            qA=a.q,
            muA=a.mu,
            quadA=a.quadrupole,
            ZB=batch.ZB,
            RB=batch.RB,
            qB=b.q,
            muB=b.mu,
            quadB=b.quadrupole,
            e_AB_source=batch.e_ABfull_source,
            e_AB_target=batch.e_ABfull_target,
            e_AA_source=batch.e_AA_source,
            e_AA_target=batch.e_AA_target,
            e_BB_source=batch.e_BB_source,
            e_BB_target=batch.e_BB_target,
            hirshfeld_volume_ratio_A=torch.abs(a.hfvr).reshape(-1),
            hirshfeld_volume_ratio_B=torch.abs(b.hfvr).reshape(-1),
            max_iterations=self.config.scf_max_iterations,
            convergence_threshold=self.config.scf_tolerance,
            thole_damping_param_direct=self.config.thole_direct,
            thole_damping_param_mutual=self.config.thole_mutual,
            return_diagnostics=True,
        )
        if not isinstance(result, tuple) or len(result) != 2:
            raise TypeError(
                "Induction backend must return (pair_energies, diagnostics)"
            )
        pair_ind, raw_diagnostics = result
        diagnostics = _normalize_diagnostics(raw_diagnostics)
        if not diagnostics.converged:
            message = (
                "Classical induction SCF did not converge after "
                f"{diagnostics.iterations} iterations "
                f"(residual={diagnostics.residual:.3e})"
            )
            if self.config.scf_nonconvergence == "raise":
                raise RuntimeError(message)
            warnings.warn(message, RuntimeWarning, stacklevel=2)
        return pair_ind, diagnostics

    def _d3_parameters(self):
        if not self.config.d3_parameters:
            return None
        if len(self.config.d3_parameters) != 4:
            raise ValueError("d3_parameters must be ordered as (s6, s8, a1, a2)")
        return dict(zip(("s6", "s8", "a1", "a2"), self.config.d3_parameters))

    def forward(
        self,
        batch,
        props_a: AtomicPropertyBundle,
        props_b: AtomicPropertyBundle,
    ) -> ClassicalEnergyBundle:
        _validate_full_pair_edges(batch)
        self._validate_alpha(batch.ZA, props_a, "A")
        self._validate_alpha(batch.ZB, props_b, "B")
        # Legacy MTP code may mutate q tensors.  Each physical term receives
        # an independent clone so kernels cannot alter caller inputs or one
        # another's component semantics.
        pair_elst = self._electrostatics(
            batch, props_a.cloned(), props_b.cloned()
        ).reshape(-1)
        pair_ind, diagnostics = self._induction(
            batch, props_a.cloned(), props_b.cloned()
        )
        pair_ind = pair_ind.reshape(-1)
        pair_disp = self.dispersion_kernel(
            batch, params=self._d3_parameters()
        ).reshape(-1)

        ndimer = int(batch.total_charge_A.numel())
        dimer_index = batch.dimer_ind_full
        return ClassicalEnergyBundle(
            pair_elst=pair_elst,
            pair_ind=pair_ind,
            pair_disp=pair_disp,
            dimer_elst=_aggregate_pairs(pair_elst, dimer_index, ndimer),
            dimer_ind=_aggregate_pairs(pair_ind, dimer_index, ndimer),
            dimer_disp=_aggregate_pairs(pair_disp, dimer_index, ndimer),
            induction_diagnostics=diagnostics,
            physics_config_hash=self.config.physics_hash,
        )


def assemble_sapt_components(
    residual: torch.Tensor,
    classical: ClassicalEnergyBundle,
    *,
    no_disp_nn: bool = False,
) -> torch.Tensor:
    """Assemble ``[ELST, EXCH, IND, DISP]`` without double counting D3."""

    if residual.ndim != 2 or residual.shape[1] != 4:
        raise ValueError("residual predictions must have shape [n_dimer, 4]")
    if residual.shape[0] != classical.dimer_elst.numel():
        raise ValueError("residual and classical batches have different sizes")
    assembled = residual.clone()
    assembled[:, 0] += classical.dimer_elst
    assembled[:, 2] += classical.dimer_ind
    if no_disp_nn:
        assembled[:, 3] = classical.dimer_disp
    else:
        assembled[:, 3] += classical.dimer_disp
    return assembled
