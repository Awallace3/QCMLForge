"""Particle Mesh Ewald for :math:`r^{-6}` dispersion (LJPME) with Casimir--Polder
low-rank factorization.

FF-2's dispersion uses an environment-dependent Casimir--Polder combination
rule, :math:`C_{6,ij} = 2 \\alpha_i \\alpha_j \\eta_i \\eta_j / (\\eta_i + \\eta_j)`,
which is non-separable.  We rewrite it via the strictly positive-definite
Cauchy kernel

.. math::

    \\frac{2 \\eta_i \\eta_j}{\\eta_i + \\eta_j}
    = \\sqrt{\\eta_i \\eta_j} \\, \\operatorname{sech}\\!\\left(
        \\frac{\\ln \\eta_i - \\ln \\eta_j}{2}
      \\right)

and approximate the :math:`\\operatorname{sech}` kernel by a Chebyshev--Nyström
rank-:math:`R` factorization on :math:`[\\eta_{\\min}, \\eta_{\\max}]`:

.. math::

    \\tilde{C}_{6,ij} = \\sum_{k=1}^{R} c_{i,k} c_{j,k},

which requires no ``log`` or ``cosh`` at runtime (only
:math:`2\\sqrt{\\eta_i \\eta_k}/(\\eta_i + \\eta_k)`), is strictly positive
semi-definite, fully differentiable, and achieves ``< 2e-3`` relative error
across a 30x range of :math:`\\eta` at ``R = 4`` (and ``< 5e-5`` at ``R = 6``).
Setting ``mode="geometric"`` recovers the rank-1 geometric-mean rule
:math:`\\tilde{C}_{6,ij} = \\alpha_i \\alpha_j \\sqrt{\\eta_i \\eta_j}`.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from ..lib.boys import dispersion_r6_erf, dispersion_r6_erfc
from ..lib.bsplines import bspline_modulus_sq_3d, bspline_weights
from ..lib.kvectors import generate_kvectors_for_ewald
from .multipole_pme import _cell_inverse, _mesh_kvectors_sq


def _chebyshev_nystrom_basis(
    rank: int,
    eta_min: float,
    eta_max: float,
) -> tuple[Tensor, Tensor]:
    """Return ``(eta_nodes (R,), L_inv (R, R))`` for the Chebyshev--Nyström
    SPD factorization of :math:`2\\sqrt{\\eta_i \\eta_j}/(\\eta_i + \\eta_j)`."""
    if rank < 1:
        raise ValueError(f"rank must be >= 1, got {rank}")
    k = torch.arange(rank, dtype=torch.float64)
    theta = (2.0 * k + 1.0) * math.pi / (2.0 * rank)
    log_min, log_max = math.log(eta_min), math.log(eta_max)
    u_c = 0.5 * (log_min + log_max) + 0.5 * (log_max - log_min) * torch.cos(theta)
    eta_nodes = torch.exp(u_c)  # (R,)
    k_cc = (
        2.0
        * torch.sqrt(eta_nodes[:, None] * eta_nodes[None, :])
        / (eta_nodes[:, None] + eta_nodes[None, :])
    )
    chol = torch.linalg.cholesky(k_cc)
    l_inv = torch.linalg.solve_triangular(
        chol, torch.eye(rank, dtype=torch.float64), upper=False
    )
    return eta_nodes, l_inv


def dispersion_kspace_kernel(k_sq: Tensor, alpha: float | Tensor) -> Tensor:
    """Reciprocal-space :math:`r^{-6}` Ewald kernel :math:`\\hat{G}^{(6)}(k)`:

    .. math::

        \\hat{G}^{(6)}(k) = \\frac{\\pi^{3/2} \\alpha^3}{3}
            \\left[ (1 - 2 z) e^{-z} + 2 \\sqrt{\\pi} z^{3/2} \\operatorname{erfc}(\\sqrt{z}) \\right],
        \\quad z = \\frac{k^2}{4 \\alpha^2}.

    Smooth and finite at ``k = 0`` where :math:`\\hat{G}^{(6)}(0) = \\pi^{3/2} \\alpha^3 / 3`.
    """
    z = k_sq / (4.0 * alpha * alpha)
    pos = z > 0
    z_safe = torch.where(pos, z, torch.ones_like(z))
    sqrt_z = torch.sqrt(z_safe)
    term_pos = (1.0 - 2.0 * z_safe) * torch.exp(-z_safe) + (
        2.0 * math.sqrt(math.pi)
    ) * (z_safe * sqrt_z) * torch.erfc(sqrt_z)
    prefac = (math.pi ** 1.5) * (alpha ** 3) / 3.0
    return prefac * torch.where(pos, term_pos, torch.ones_like(z))


class DispersionPME(torch.nn.Module):
    """Smooth Particle Mesh Ewald for attractive :math:`-C_{6,ij}/r^6` dispersion.

    Parameters
    ----------
    alpha : float
        Ewald splitting parameter for :math:`r^{-6}` (inverse length).
    ns_mesh : tuple[int, int, int]
        Static 3D FFT grid dimensions ``(K1, K2, K3)``.
    order : int
        Cardinal B-spline order (default ``4``, smooth enough for scalar
        :math:`r^{-6}` forces since ``M_4'`` is ``C^1``).
    rank : int
        Number of separable Casimir--Polder channels (default ``4``). Ignored
        when ``mode="geometric"`` (always rank 1).
    mode : {"chebyshev_nystrom", "geometric"}
        Separable factorization of the Casimir--Polder combination rule.
    eta_min, eta_max : float
        Expected range of FF-2 effective frequencies :math:`\\eta_i` (Hartree)
        for placing Chebyshev nodes.
    prefactor : float
        Overall unit prefactor (default ``1.0`` for atomic units).
    """

    def __init__(
        self,
        alpha: float,
        ns_mesh: tuple[int, int, int],
        order: int = 4,
        rank: int = 4,
        mode: str = "chebyshev_nystrom",
        eta_min: float = 0.08,
        eta_max: float = 2.5,
        prefactor: float = 1.0,
    ):
        super().__init__()
        if alpha <= 0:
            raise ValueError(f"alpha must be positive, got {alpha}")
        if order < 3:
            raise ValueError(f"order must be >= 3, got {order}")
        if mode not in ("chebyshev_nystrom", "geometric"):
            raise ValueError(f"unknown dispersion mode {mode!r}")

        self.alpha = float(alpha)
        self.ns_mesh = (int(ns_mesh[0]), int(ns_mesh[1]), int(ns_mesh[2]))
        self.order = int(order)
        self.mode = mode
        self.rank = 1 if mode == "geometric" else int(rank)
        self.prefactor = float(prefactor)

        self.register_buffer(
            "mod_sq",
            bspline_modulus_sq_3d(self.ns_mesh, self.order, dtype=torch.float64),
            persistent=False,
        )
        self.register_buffer(
            "shifts",
            torch.arange(self.order, dtype=torch.long),
            persistent=False,
        )
        if mode == "chebyshev_nystrom":
            eta_nodes, l_inv = _chebyshev_nystrom_basis(self.rank, eta_min, eta_max)
            self.register_buffer("eta_nodes", eta_nodes, persistent=False)
            self.register_buffer("l_inv", l_inv, persistent=False)

    def factors(self, alpha_iso: Tensor, eta: Tensor) -> Tensor:
        """Compute separable per-atom dispersion weights ``c`` of shape ``(N, R)``
        such that :math:`\\tilde{C}_{6,ij} = \\mathbf{c}_i \\cdot \\mathbf{c}_j`."""
        sqrt_eta = torch.sqrt(eta)
        base = alpha_iso * sqrt_eta
        if self.mode == "geometric":
            return base[:, None]
        eta_n = self.eta_nodes.to(dtype=eta.dtype, device=eta.device)
        l_inv = self.l_inv.to(dtype=eta.dtype, device=eta.device)
        v = (2.0 * sqrt_eta[:, None] * torch.sqrt(eta_n)[None, :]) / (
            eta[:, None] + eta_n[None, :]
        )  # (N, R)
        return base[:, None] * (v @ l_inv.T)

    @staticmethod
    def pair_c6_tilde(factors: Tensor, ai: Tensor, aj: Tensor) -> Tensor:
        """Separable approximation :math:`\\tilde{C}_{6,ij}` for pairs ``(ai, aj)``."""
        return (factors[ai] * factors[aj]).sum(dim=-1)

    def reciprocal_energy(
        self,
        positions: Tensor,
        cell: Tensor,
        factors: Tensor,
    ) -> Tensor:
        """Attractive reciprocal-space :math:`-\\tilde{C}_{6,ij}/r^6` energy
        (minus self-term; includes the finite ``k = 0`` background term)."""
        k1, k2, k3 = self.ns_mesh
        p = self.order
        n_chan = factors.shape[1]
        dtype, device = positions.dtype, positions.device

        inv_cell = _cell_inverse(cell)
        k_vec = torch.tensor([k1, k2, k3], dtype=dtype, device=device)
        u = positions @ (inv_cell * k_vec[None, :])
        i0 = torch.floor(u)
        delta = u - i0
        i0_l = i0.detach().long()

        w0 = bspline_weights(delta, p, deriv_order=0)[0]  # (N, 3, p)
        shifts = self.shifts.to(device=device)
        ix = torch.remainder(i0_l[:, 0:1] - shifts[None, :], k1)
        iy = torch.remainder(i0_l[:, 1:2] - shifts[None, :], k2)
        iz = torch.remainder(i0_l[:, 2:3] - shifts[None, :], k3)
        flat_idx = (
            ((ix[:, :, None, None] * k2) + iy[:, None, :, None]) * k3
            + iz[:, None, None, :]
        ).reshape(-1)

        # (N, p, p, p) scalar B-spline weights
        w3d = (
            w0[:, 0, :, None, None]
            * w0[:, 1, None, :, None]
            * w0[:, 2, None, None, :]
        ).reshape(-1, 1)  # (N * p^3, 1)
        # Repeat factor rows per stencil point without Python loop over channels
        # factors: (N, R) -> (N, p^3, R) -> (N * p^3, R)
        vals = (
            w3d.reshape(positions.shape[0], p**3, 1) * factors[:, None, :]
        ).reshape(-1, n_chan)

        mesh_flat = factors.new_zeros((k1 * k2 * k3, n_chan))
        mesh_flat = mesh_flat.index_add(0, flat_idx, vals)
        mesh = mesh_flat.T.reshape(n_chan, k1, k2, k3)

        k_sq = _mesh_kvectors_sq(cell, self.ns_mesh)
        volume = torch.abs(torch.linalg.det(cell))
        mod_sq = self.mod_sq.to(dtype=dtype, device=device)
        c_hat = (mod_sq * dispersion_kspace_kernel(k_sq, self.alpha)) / volume

        mesh_hat = torch.fft.rfftn(mesh, dim=(1, 2, 3), norm="backward")
        phi = torch.fft.irfftn(
            mesh_hat * c_hat[None, :, :, :],
            s=self.ns_mesh,
            dim=(1, 2, 3),
            norm="forward",
        )
        e_rec = 0.5 * (mesh * phi).sum()
        # Self term: (1/2) * (alpha^6 / 6) * sum_i ||c_i||^2
        e_self = (self.alpha ** 6 / 12.0) * (factors * factors).sum()
        # Attractive sign!
        return -self.prefactor * (e_rec - e_self)

    def real_space_correction(
        self,
        r: Tensor,
        factors: Tensor,
        ai: Tensor,
        aj: Tensor,
    ) -> Tensor:
        """Per-pair ``+C6_tilde * psi_rec^(6)(r)`` to add to exact BJ dispersion
        within ``r_cut`` (canceling the reciprocal contribution for near pairs)
        or to sum over excluded same-monomer pairs."""
        c6_t = self.pair_c6_tilde(factors, ai, aj)
        return self.prefactor * c6_t * dispersion_r6_erf(r, self.alpha)


class DispersionEwald(torch.nn.Module):
    """Exact direct-sum :math:`r^{-6}` Ewald oracle for validating
    :class:`DispersionPME`."""

    def __init__(
        self,
        alpha: float,
        k_max: tuple[int, int, int] | int = 6,
        prefactor: float = 1.0,
    ):
        super().__init__()
        self.alpha = float(alpha)
        if isinstance(k_max, int):
            self.k_max = (k_max, k_max, k_max)
        else:
            self.k_max = (int(k_max[0]), int(k_max[1]), int(k_max[2]))
        self.prefactor = float(prefactor)

    def reciprocal_energy(
        self, positions: Tensor, cell: Tensor, factors: Tensor
    ) -> Tensor:
        ns = torch.tensor(
            [2 * m + 1 for m in self.k_max], dtype=torch.long, device=cell.device
        )
        kv = generate_kvectors_for_ewald(cell=cell, ns=ns)  # (K, 3), includes k=0!
        k_sq = (kv * kv).sum(dim=-1)
        g_k = dispersion_kspace_kernel(k_sq, self.alpha)
        volume = torch.abs(torch.linalg.det(cell))
        kr = kv @ positions.T  # (K, N)
        s_re = torch.cos(kr) @ factors  # (K, R)
        s_im = torch.sin(kr) @ factors  # (K, R)
        e_rec = 0.5 / volume * (g_k[:, None] * (s_re * s_re + s_im * s_im)).sum()
        e_self = (self.alpha ** 6 / 12.0) * (factors * factors).sum()
        return -self.prefactor * (e_rec - e_self)

    def direct_energy(
        self, positions: Tensor, cell: Tensor, factors: Tensor, r_cut: float
    ) -> Tensor:
        dtype, device = positions.dtype, positions.device
        n_atoms = positions.shape[0]
        inv_cell = torch.linalg.inv(cell)
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
        shift_cart = shifts_grid @ cell

        idx = torch.arange(n_atoms, device=device)
        ii, jj = torch.meshgrid(idx, idx, indexing="ij")
        ii, jj = ii.reshape(-1), jj.reshape(-1)
        base_disp = positions[jj] - positions[ii]

        e_tot = torch.zeros((), dtype=dtype, device=device)
        for s_idx in range(shift_cart.shape[0]):
            is_zero = bool((shifts_grid[s_idx] == 0).all())
            disp = base_disp + shift_cart[s_idx]
            r2 = (disp * disp).sum(dim=-1)
            mask = r2 < (r_cut * r_cut)
            if is_zero:
                mask = mask & (ii != jj)
            if not bool(mask.any()):
                continue
            pi, pj = ii[mask], jj[mask]
            r = torch.sqrt(r2[mask])
            c6_t = (factors[pi] * factors[pj]).sum(dim=-1)
            e_tot = e_tot - 0.5 * (c6_t * dispersion_r6_erfc(r, self.alpha)).sum()
        return self.prefactor * e_tot


def homogeneous_tail_correction(
    volume: Tensor,
    r_cut: float,
    c6_sum: Tensor,
    c8_sum: Tensor | None = None,
    s8: float | Tensor = 1.0,
) -> Tensor:
    """Isotropic long-range dispersion tail correction for ``r >= r_cut``.

    .. math::

        E_{\\mathrm{tail}} = -\\frac{2\\pi}{3 V r_c^3} \\sum_{i,j} C_{6,ij}
                             -\\frac{2\\pi}{5 V r_c^5} s_8 \\sum_{i,j} C_{8,ij}

    where ``c6_sum`` is :math:`\\sum_{i, j} C_{6,ij}` (all ordered pairs, or
    ``N^2 * <C6>``) divided by 2 (the factor :math:`2\\pi = \\frac{1}{2} \\cdot 4\\pi`
    already accounts for pair double counting).
    """
    e = -(2.0 * math.pi / (3.0 * r_cut**3)) * c6_sum / volume
    if c8_sum is not None:
        e = e - (2.0 * math.pi / (5.0 * r_cut**5)) * s8 * c8_sum / volume
    return e


__all__ = [
    "DispersionEwald",
    "DispersionPME",
    "dispersion_kspace_kernel",
    "homogeneous_tail_correction",
]
