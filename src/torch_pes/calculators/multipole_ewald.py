"""Exact direct-sum Multipolar Ewald (rank 0, 1, 2) for periodic systems.

Evaluates the exact periodic Ewald sum for atomic charges :math:`q_i`,
dipoles :math:`\\boldsymbol{\\mu}_i`, and symmetric traceless Buckingham
quadrupoles :math:`\\boldsymbol{\\Theta}_i` (:math:`\\tilde{Q}_i = \\Theta_i / 3`)
using explicit reciprocal-space :math:`\\mathbf{k}`-vectors and direct-space
Boys functions :math:`B_0 \\dots B_4`.

Serves both as an exact reference oracle for validating
:class:`torch_pes.calculators.MultipolePME` and as a usable direct Ewald
calculator for small periodic cells.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from ..lib.boys import (
    dipole_tensor_coeffs,
    ewald_boys_erf,
    ewald_boys_erfc,
    multipole_pair_energy,
    multipole_pair_field,
)
from ..lib.kvectors import generate_kvectors_for_ewald

_INV_SQRT_PI = 1.0 / math.sqrt(math.pi)


def multipole_self_energy(
    alpha: float | Tensor,
    charges: Tensor,
    dipoles: Tensor | None = None,
    quadrupoles: Tensor | None = None,
) -> Tensor:
    """Exact Gaussian Ewald self-energy (to be subtracted from reciprocal sum).

    .. math::

        E_{\\mathrm{self}} = \\frac{\\alpha}{\\sqrt{\\pi}} \\sum_i q_i^2
            + \\frac{2\\alpha^3}{3\\sqrt{\\pi}} \\sum_i \\|\\boldsymbol{\\mu}_i\\|^2
            + \\frac{8\\alpha^5}{45\\sqrt{\\pi}} \\sum_i (\\boldsymbol{\\Theta}_i : \\boldsymbol{\\Theta}_i)
    """
    e = (alpha * _INV_SQRT_PI) * (charges * charges).sum()
    if dipoles is not None:
        e = e + (2.0 / 3.0 * _INV_SQRT_PI * alpha**3) * (dipoles * dipoles).sum()
    if quadrupoles is not None:
        e = e + (8.0 / 45.0 * _INV_SQRT_PI * alpha**5) * (
            quadrupoles * quadrupoles
        ).sum()
    return e


def multipole_background_energy(
    alpha: float | Tensor,
    volume: Tensor,
    charges: Tensor,
) -> Tensor:
    """Neutralizing uniform background energy :math:`-\\frac{\\pi}{2\\alpha^2 V} (\\sum_i q_i)^2`."""
    q_tot = charges.sum()
    return -(math.pi / (2.0 * alpha * alpha)) * (q_tot * q_tot) / volume


def dipole_self_field_coeff(alpha: float | Tensor) -> float | Tensor:
    """Coefficient :math:`\\frac{4\\alpha^3}{3\\sqrt{\\pi}}` for the dipole self-field.

    * Reciprocal electric field without self-term:
      ``E_rec_no_self = E_rec + coeff * dipoles``
    * Reciprocal polarization matvec without self-term:
      ``(T_pol_rec_no_self @ p) = (T_pol_rec @ p) - coeff * p``
    """
    return (4.0 / 3.0 * _INV_SQRT_PI) * (alpha**3)


class MultipoleEwald(torch.nn.Module):
    """Direct-sum Multipolar Ewald calculator up to traceless quadrupoles.

    Parameters
    ----------
    alpha : float
        Ewald splitting parameter (in inverse units of ``positions`` and ``cell``).
    k_max : tuple[int, int, int] or int
        Maximum reciprocal lattice Miller indices ``(nx, ny, nz)``. The full
        grid ``[-nx..nx] x [-ny..ny] x [-nz..nz]`` is summed (excluding ``k=0``).
    prefactor : float
        Overall unit prefactor (default ``1.0``, i.e. atomic units / Hartree
        when positions are in bohr).
    """

    def __init__(
        self,
        alpha: float,
        k_max: tuple[int, int, int] | int = 6,
        prefactor: float = 1.0,
    ):
        super().__init__()
        if alpha <= 0:
            raise ValueError(f"alpha must be positive, got {alpha}")
        self.alpha = float(alpha)
        if isinstance(k_max, int):
            self.k_max = (k_max, k_max, k_max)
        else:
            self.k_max = (int(k_max[0]), int(k_max[1]), int(k_max[2]))
        self.prefactor = float(prefactor)

    def _kvectors_and_kernel(self, cell: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        ns = torch.tensor(
            [2 * m + 1 for m in self.k_max], dtype=torch.long, device=cell.device
        )
        kv = generate_kvectors_for_ewald(cell=cell, ns=ns)  # (K, 3)
        k_sq = (kv * kv).sum(dim=-1)
        nonzero = k_sq > 0
        kv = kv[nonzero]
        k_sq = k_sq[nonzero]
        g_k = (4.0 * math.pi / k_sq) * torch.exp(-k_sq / (4.0 * self.alpha * self.alpha))
        volume = torch.abs(torch.linalg.det(cell))
        return kv, g_k, volume

    def reciprocal_energy_and_field(
        self,
        positions: Tensor,
        cell: Tensor,
        charges: Tensor,
        dipoles: Tensor | None = None,
        quadrupoles: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Reciprocal-space multipole energy (minus self + background) and
        permanent electric field (minus self-field) at each atom.

        Parameters
        ----------
        positions : Tensor, shape ``(N, 3)``
        cell : Tensor, shape ``(3, 3)``
            Rows are lattice vectors.
        charges : Tensor, shape ``(N,)``
        dipoles : Tensor, shape ``(N, 3)``, optional
        quadrupoles : Tensor, shape ``(N, 3, 3)``, optional
            Symmetric traceless Buckingham quadrupoles :math:`\\Theta`.

        Returns
        -------
        energy : Tensor, scalar
        field : Tensor, shape ``(N, 3)``
        """
        kv, g_k, volume = self._kvectors_and_kernel(cell)
        # phase[k, i] = k . R_i
        kr = kv @ positions.T  # (K, N)
        cos_kr = torch.cos(kr)
        sin_kr = torch.sin(kr)

        # S(k) = sum_j [ (q_j - k^T Q_j k) + i (mu_j . k) ] e^{i k . R_j}
        a_ki = charges[None, :].expand_as(kr)
        if quadrupoles is not None:
            # k^T (Theta_j / 3) k
            ktqk = torch.einsum("ka,iab,kb->ki", kv, quadrupoles, kv) / 3.0
            a_ki = a_ki - ktqk

        if dipoles is not None:
            b_ki = kv @ dipoles.T  # (K, N)
        else:
            b_ki = torch.zeros_like(kr)

        # Real & imag parts of (a + i b) * (cos + i sin)
        s_re = (a_ki * cos_kr - b_ki * sin_kr).sum(dim=1)  # (K,)
        s_im = (a_ki * sin_kr + b_ki * cos_kr).sum(dim=1)  # (K,)

        e_rec = 0.5 / volume * (g_k * (s_re * s_re + s_im * s_im)).sum()
        e_rec = (
            e_rec
            - multipole_self_energy(self.alpha, charges, dipoles, quadrupoles)
            + multipole_background_energy(self.alpha, volume, charges)
        )

        # Potential phi(R_i) = (1/V) sum_k G(k) Re[ S(k) e^{-i k . R_i} ]
        # E(R_i) = -grad_{R_i} phi(R_i)
        #        = (1/V) sum_k G(k) k Im[ S(k) e^{-i k . R_i} ]
        # wait: -grad Re[ (s_re + i s_im)(cos - i sin) ]
        #     = -grad [ s_re cos(k.R) + s_im sin(k.R) ]
        #     = k [ s_re sin(k.R) - s_im cos(k.R) ]
        gs_re = (g_k * s_re)[:, None]  # (K, 1)
        gs_im = (g_k * s_im)[:, None]  # (K, 1)
        weight = gs_re * sin_kr - gs_im * cos_kr  # (K, N)
        field = (weight.T @ kv) / volume  # (N, 3)
        if dipoles is not None:
            field = field + dipole_self_field_coeff(self.alpha) * dipoles

        return self.prefactor * e_rec, self.prefactor * field

    def reciprocal_dipole_matvec(
        self,
        positions: Tensor,
        cell: Tensor,
        dipoles: Tensor,
    ) -> Tensor:
        """Reciprocal-space induced dipole-dipole matvec
        :math:`(T^{\\mathrm{pol,rec}} \\mathbf{p})_i` (minus self-term),
        equal to :math:`-\\mathbf{E}^{\\mathrm{rec,no-self}}(\\mathbf{p})_i`."""
        zeros = torch.zeros(
            positions.shape[0], dtype=positions.dtype, device=positions.device
        )
        _, field = self.reciprocal_energy_and_field(
            positions, cell, zeros, dipoles=dipoles, quadrupoles=None
        )
        return -field

    def direct_energy_and_field(
        self,
        positions: Tensor,
        cell: Tensor,
        charges: Tensor,
        dipoles: Tensor | None,
        quadrupoles: Tensor | None,
        r_cut: float,
    ) -> tuple[Tensor, Tensor]:
        """Exact direct-space Ewald ``erfc`` sum over all periodic images
        within ``r_cut`` (excluding the ``i == j, n == 0`` self-pair).

        Intended as an exact reference for testing against PME.
        """
        dtype, device = positions.dtype, positions.device
        n_atoms = positions.shape[0]
        inv_cell = torch.linalg.inv(cell)
        # Max image shifts needed along each lattice vector
        # Perpendicular heights h_a = 1 / |inv_cell[:, a]|
        heights = 1.0 / torch.linalg.norm(inv_cell, dim=0)
        n_img = [int(math.ceil(r_cut / float(h))) + 1 for h in heights]

        shifts_grid = torch.stack(
            torch.meshgrid(
                torch.arange(-n_img[0], n_img[0] + 1, device=device, dtype=dtype),
                torch.arange(-n_img[1], n_img[1] + 1, device=device, dtype=dtype),
                torch.arange(-n_img[2], n_img[2] + 1, device=device, dtype=dtype),
                indexing="ij",
            ),
            dim=-1,
        ).reshape(-1, 3)
        shift_cart = shifts_grid @ cell  # (S, 3)

        e_tot = torch.zeros((), dtype=dtype, device=device)
        f_tot = torch.zeros((n_atoms, 3), dtype=dtype, device=device)

        idx = torch.arange(n_atoms, device=device)
        ii, jj = torch.meshgrid(idx, idx, indexing="ij")
        ii, jj = ii.reshape(-1), jj.reshape(-1)
        base_disp = positions[jj] - positions[ii]  # (N^2, 3), i -> j

        for s_idx in range(shift_cart.shape[0]):
            is_zero_shift = bool((shifts_grid[s_idx] == 0).all())
            disp = base_disp + shift_cart[s_idx]
            r2 = (disp * disp).sum(dim=-1)
            mask = r2 < (r_cut * r_cut)
            if is_zero_shift:
                mask = mask & (ii != jj)
            if not bool(mask.any()):
                continue
            pi, pj = ii[mask], jj[mask]
            d = disp[mask]
            r = torch.sqrt((d * d).sum(dim=-1))
            rhat = d / r[:, None]
            boys = ewald_boys_erfc(r, self.alpha)

            e_pair = multipole_pair_energy(
                r,
                rhat,
                charges[pi],
                None if dipoles is None else dipoles[pi],
                None if quadrupoles is None else quadrupoles[pi],
                charges[pj],
                None if dipoles is None else dipoles[pj],
                None if quadrupoles is None else quadrupoles[pj],
                boys,
            )
            e_tot = e_tot + 0.5 * e_pair.sum()

            # Field at probe pj due to source pi (rhat points pi -> pj)
            f_pair = multipole_pair_field(
                r,
                rhat,
                charges[pi],
                None if dipoles is None else dipoles[pi],
                None if quadrupoles is None else quadrupoles[pi],
                boys,
            )
            f_tot = f_tot.index_add(0, pj, f_pair)

        return self.prefactor * e_tot, self.prefactor * f_tot

    def direct_dipole_matvec(
        self,
        positions: Tensor,
        cell: Tensor,
        dipoles: Tensor,
        r_cut: float,
    ) -> Tensor:
        """Exact direct-space Ewald ``erfc`` dipole-dipole matvec."""
        zeros = torch.zeros(
            positions.shape[0], dtype=positions.dtype, device=positions.device
        )
        _, field = self.direct_energy_and_field(
            positions, cell, zeros, dipoles=dipoles, quadrupoles=None, r_cut=r_cut
        )
        return -field

    def exclusion_energy_and_field(
        self,
        r: Tensor,
        rhat: Tensor,
        ai: Tensor,
        aj: Tensor,
        charges: Tensor,
        dipoles: Tensor | None = None,
        quadrupoles: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Long-range ``erf`` multipole energy and field for excluded pairs
        ``(ai, aj)`` (half list ``ai < aj``, ``rhat`` points ``ai -> aj``)."""
        boys = ewald_boys_erf(r, self.alpha)
        e_excl = multipole_pair_energy(
            r,
            rhat,
            charges[ai],
            None if dipoles is None else dipoles[ai],
            None if quadrupoles is None else quadrupoles[ai],
            charges[aj],
            None if dipoles is None else dipoles[aj],
            None if quadrupoles is None else quadrupoles[aj],
            boys,
        ).sum()

        # Both directions for field: probe aj <- src ai (+rhat), probe ai <- src aj (-rhat)
        n_atoms = charges.shape[0]
        f_excl = torch.zeros((n_atoms, 3), dtype=charges.dtype, device=charges.device)
        f_on_j = multipole_pair_field(
            r,
            rhat,
            charges[ai],
            None if dipoles is None else dipoles[ai],
            None if quadrupoles is None else quadrupoles[ai],
            boys,
        )
        f_on_i = multipole_pair_field(
            r,
            -rhat,
            charges[aj],
            None if dipoles is None else dipoles[aj],
            None if quadrupoles is None else quadrupoles[aj],
            boys,
        )
        f_excl = f_excl.index_add(0, aj, f_on_j).index_add(0, ai, f_on_i)
        return self.prefactor * e_excl, self.prefactor * f_excl


__all__ = [
    "MultipoleEwald",
    "dipole_self_field_coeff",
    "multipole_background_energy",
    "multipole_self_energy",
]
